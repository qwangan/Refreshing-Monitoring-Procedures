#!/usr/bin/env python3
"""Resumable generator for the locked OPT-1.3B Tournament study.

This module deliberately generates and checkpoints *pivots*, not detector
outputs.  Every detector in the study can therefore be replayed on exactly the
same paths without running OPT-1.3B again.

Locked OPT-1.3B design (1,400 fresh paths)
-------------------------------------------
* T=600 and 20 frozen prompts.
* Temperatures 0.75 and 1.0 only, with no top-k/top-p truncation and no EOS stop.
* Primary two-region schedule [51,250] U [301,500], 500 paths/temperature.
* Four-region stress case: L=50,G=25; 200 paths/temp.

At a watermarked position, thirty fresh full-vocabulary Bernoulli(1/2) layers
implement the exact distributional Tournament recursion. At an ordinary
position the token is sampled directly from the full-vocabulary NTP law before
the selected token's thirty unused table bits are lazily drawn. A separate
stream draws the randomized-Binomial PIT variable V. Checkpoints save the
selected g-vector, S, V, Y and L, so every detector is replayed from one trace.

Random streams are keyed by a stable hash of the scenario, replicate and
stream role.  Consequently paths do not change when jobs are reordered,
resumed, split across processes, or when unrelated scenarios are added.
Checkpoints are written atomically and content-validated before being skipped.

The ``smoke`` profile is intentionally tiny and is not scientific output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from tournament_watermark import (
    LAYERS,
    binomial_randomized_pit,
    normalize_float32_tournament_mass,
    stable_calibrator,
)


ROOT = Path(__file__).resolve().parent


MODEL_NAME = "facebook/opt-1.3b"
MODEL_REVISION = "3f5c25d0bc631cb57ac65913f76e22c2dfb61d62"
HORIZON = 600
MASTER_SEED = 202608010731
STUDY_TEMPERATURES = (0.75, 1.0)
PATHS_PER_TEMPERATURE = 700
TOTAL_MODEL_PATHS = 1_400
LOCKED_DTYPE = "torch.float32"
BIT_GENERATOR_NAME = "PCG64DXSM"
SCHEMA_VERSION = 1
GENERATION_ALGORITHM_VERSION = "opt13b-full-vocab-tournament-m30-v2-roundoff-repair"
DEFAULT_BATCH_SIZE = 10
DEFAULT_TORCH_THREADS = 8
TORCH_INTEROP_THREADS = 1
TABLE_STREAM = 0
ORDINARY_STREAM = 1
PIVOT_STREAM = 2
TOURNAMENT_SAMPLE_STREAM = 3


# Frozen before confirmatory generation.  These are original prompts, span
# several genres, contain no path-specific variables, and all end in a space.
PROMPTS: tuple[str, ...] = (
    "Researchers evaluating scientific claims distinguish the design of an experiment from the interpretation of its results. In a careful report, the assumptions are stated first, the measurements are recorded consistently, and uncertainty is kept visible throughout the discussion. The present example begins with ",
    "The city council released a briefing on public transportation after several months of community meetings. The document compares operating costs, travel times, neighborhood access, and the practical limits of the current street network. According to the briefing, ",
    "A historian opening a box of letters must decide which details are evidence and which are later annotations. Dates, paper, handwriting, and references to public events can each narrow the possibilities, but no single clue settles the matter. In this archive, ",
    "Before servicing the instrument, disconnect external power and record the current calibration settings. Inspect the housing, connectors, and ventilation slots without removing internal components. If the preliminary checks show no visible damage, the next step is to ",
    "At the edge of the harbor, Mara found a narrow shop whose windows were filled with clocks that showed different times. The owner insisted that none of them was broken; each, he said, belonged to a traveler who had not yet returned. That evening, ",
    "A useful travel plan leaves room for delays while identifying the few reservations that cannot be changed. The route below favors trains, short walking connections, and locally owned lodging, with an alternative for severe weather. On the first morning, ",
    "When economists describe a price change, they often separate shifts in demand from changes in production costs. The distinction matters because the same observed increase can have different causes and different policy implications. In the market considered here, ",
    "The wetland survey records water depth, plant cover, bird calls, and signs of recent disturbance along fixed transects. Repeating the same measurements across seasons helps distinguish a persistent ecological change from ordinary variation. During the spring visit, ",
    "A plain-language explanation of a contract should identify the parties, the promised actions, the relevant dates, and the procedure for resolving disagreements. It should also distinguish a summary from legal advice. In the sample agreement, ",
    "Students learning probability often understand a new concept more quickly when a formal definition is paired with a small experiment. The activity begins with a prediction, continues with repeated observations, and ends by comparing the data with the original reasoning. For this lesson, ",
    "A general health article can explain how clinicians evaluate symptoms without diagnosing an individual reader. It should describe common considerations, warning signs that merit prompt care, and the limits of information available outside a clinical examination. In this overview, ",
    "The recipe is organized so that the sauce can be prepared while the vegetables roast. Ingredients are measured before heating begins, and the final seasoning is adjusted only after the components are combined. To start, ",
    "Astronomers infer the properties of distant objects from light collected over many observations. Because atmosphere, instruments, and background sources can alter a measurement, calibration and replication are central to the analysis. For the target described here, ",
    "A philosophical argument is easier to assess when its premises are separated from its conclusion. Ambiguous terms can then be clarified, possible counterexamples tested, and disagreements traced to specific assumptions. The argument in this chapter begins by claiming that ",
    "The project memo summarizes decisions that have already been made, issues that still require an owner, and deadlines that depend on outside review. Estimates are presented as ranges where appropriate, and unresolved risks are listed beside their mitigation plans. This week, ",
    "A biographer comparing interviews with official records may find that the sources disagree about chronology while agreeing on broader themes. Rather than forcing a single seamless story, the account can mark those discrepancies and explain why they matter. In the early years, ",
    "Urban planners studying a busy corridor consider housing, deliveries, pedestrian safety, drainage, trees, and emergency access at the same time. A proposal that improves one measure may create costs elsewhere, so the alternatives are evaluated together. The revised plan would ",
    "A postgame analysis should separate the final score from the sequence of decisions that produced it. Possession, shot quality, substitutions, fatigue, and chance all contribute, and a short run can look more decisive than it was. In the second half, ",
    "The exhibition review considers how the works are arranged as well as the qualities of individual pieces. Lighting, labels, room transitions, and the order of themes shape what a visitor notices and remembers. In the central gallery, ",
    "A policy evaluation starts by specifying the intended outcome, the population affected, and the comparison that will be used. Implementation may differ across locations, so aggregate results are supplemented with transparent subgroup analyses. For the program examined here, ",
)


@dataclass(frozen=True)
class ScheduleSpec:
    schedule_id: str
    intended_regions: tuple[tuple[int, int], ...]  # inclusive, one based


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    schedule: ScheduleSpec
    temperature: float
    n_paths: int
    horizon: int = HORIZON


@dataclass(frozen=True)
class PathSpec:
    path_uid: str
    scenario_id: str
    schedule_id: str
    intended_regions: tuple[tuple[int, int], ...]
    temperature: float
    horizon: int
    replicate: int
    prompt_id: int
    scenario_seed_words: tuple[int, int, int, int]
    generation_batch_uid: str
    generation_batch_size: int
    generation_batch_position: int


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_uint32_words(value: object, n_words: int = 4) -> tuple[int, ...]:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).digest()
    return tuple(int.from_bytes(digest[4 * i : 4 * i + 4], "little") for i in range(n_words))


def temperature_label(value: float) -> str:
    return format(value, ".8g").replace(".", "p")


def two_region_schedule(length: int, gap: int, start: int = 51) -> ScheduleSpec:
    first = (start, start + length - 1)
    second_start = first[1] + gap + 1
    second = (second_start, second_start + length - 1)
    return ScheduleSpec(f"two_l{length:03d}_g{gap:03d}", (first, second))


def four_region_schedule(length: int = 50, gap: int = 25, start: int = 51) -> ScheduleSpec:
    regions = []
    cursor = start
    for _ in range(4):
        regions.append((cursor, cursor + length - 1))
        cursor += length + gap
    return ScheduleSpec(f"four_l{length:03d}_g{gap:03d}", tuple(regions))


def validate_schedule(schedule: ScheduleSpec, horizon: int) -> None:
    previous_end = 0
    for start, end in schedule.intended_regions:
        if not (1 <= start <= end <= horizon):
            raise ValueError(f"invalid region {(start, end)} for horizon {horizon}")
        if start <= previous_end:
            raise ValueError(f"overlapping or unordered regions in {schedule.schedule_id}")
        previous_end = end


def schedule_mask(schedule: ScheduleSpec, horizon: int) -> np.ndarray:
    validate_schedule(schedule, horizon)
    mask = np.zeros(horizon, dtype=np.bool_)
    for start, end in schedule.intended_regions:
        mask[start - 1 : end] = True
    return mask


def _main_scenarios_for_temperatures(
    temperatures: Sequence[float],
) -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []
    for temperature in temperatures:
        # Keep the certified central-scenario ID and seed namespace so the
        # master-seed derivation remains unchanged from the old package.
        schedule = two_region_schedule(200, 50)
        scenarios.append(
            ScenarioSpec(
                scenario_id=f"core_{schedule.schedule_id}_temp{temperature_label(temperature)}",
                schedule=schedule,
                temperature=temperature,
                n_paths=500,
            )
        )

        stress = four_region_schedule()
        scenarios.append(
            ScenarioSpec(
                scenario_id=f"stress_{stress.schedule_id}_temp{temperature_label(temperature)}",
                schedule=stress,
                temperature=temperature,
                n_paths=200,
            )
        )
    return scenarios


def study_scenarios() -> list[ScenarioSpec]:
    """The only scientific Tournament design: 1,400 paths at two temperatures."""

    scenarios = _main_scenarios_for_temperatures(STUDY_TEMPERATURES)
    if sum(s.n_paths for s in scenarios) != TOTAL_MODEL_PATHS:
        raise AssertionError("study manifest must contain exactly 1,400 paths")
    return scenarios


def confirmatory_scenarios() -> list[ScenarioSpec]:
    """Compatibility alias for analysis code; identical to ``study_scenarios``."""

    return study_scenarios()


def development_scenarios() -> list[ScenarioSpec]:
    """Small fresh development panel, never to be labeled confirmatory."""

    return [
        ScenarioSpec(
            scenario_id=f"dev_two_l200_g050_temp{temperature_label(temperature)}",
            schedule=two_region_schedule(200, 50),
            temperature=temperature,
            n_paths=20,
        )
        for temperature in STUDY_TEMPERATURES
    ]


def smoke_scenarios(smoke_tokens: int) -> list[ScenarioSpec]:
    if not 4 <= smoke_tokens <= 32:
        raise ValueError("--smoke-tokens must lie in [4,32]")
    start = 2
    end = min(smoke_tokens - 1, max(start, smoke_tokens // 2))
    schedule = ScheduleSpec("smoke_single_region", ((start, end),))
    return [
        ScenarioSpec(
            scenario_id="smoke_temp1p0",
            schedule=schedule,
            temperature=1.0,
            # Replicates 1 and 21 share prompt 0 and therefore exercise a
            # genuine two-path batch when selected together.
            n_paths=21,
            horizon=smoke_tokens,
        )
    ]


def scenarios_for_profile(profile: str, smoke_tokens: int) -> list[ScenarioSpec]:
    if profile == "study":
        return study_scenarios()
    if profile == "development":
        return development_scenarios()
    if profile == "smoke":
        return smoke_scenarios(smoke_tokens)
    raise ValueError(profile)


def scenario_seed_words(scenario: ScenarioSpec) -> tuple[int, int, int, int]:
    payload = {
        "namespace": "aos-opt13b-fresh-v1",
        "scenario_id": scenario.scenario_id,
        "schedule_id": scenario.schedule.schedule_id,
        "regions": scenario.schedule.intended_regions,
        "temperature": scenario.temperature,
        "horizon": scenario.horizon,
    }
    return stable_uint32_words(payload)  # type: ignore[return-value]


def build_manifest(
    scenarios: Sequence[ScenarioSpec], batch_size: int = DEFAULT_BATCH_SIZE
) -> list[PathSpec]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    paths: list[PathSpec] = []
    for scenario in scenarios:
        validate_schedule(scenario.schedule, scenario.horizon)
        seed_words = scenario_seed_words(scenario)
        replicates_by_prompt: dict[int, list[int]] = {i: [] for i in range(len(PROMPTS))}
        for replicate in range(1, scenario.n_paths + 1):
            replicates_by_prompt[(replicate - 1) % len(PROMPTS)].append(replicate)

        batch_assignment: dict[int, tuple[str, int, int]] = {}
        for prompt_id, replicates in replicates_by_prompt.items():
            for batch_index, start in enumerate(range(0, len(replicates), batch_size)):
                members = replicates[start : start + batch_size]
                batch_uid = (
                    f"{scenario.scenario_id}__p{prompt_id:02d}__b{batch_index:03d}"
                )
                for position, replicate in enumerate(members):
                    batch_assignment[replicate] = (batch_uid, len(members), position)

        # The manifest remains ordered by replicate.  Generation later groups
        # by the precomputed batch UID, so ordering or resumption cannot alter
        # batch membership.
        for replicate in range(1, scenario.n_paths + 1):
            prompt_id = (replicate - 1) % len(PROMPTS)
            batch_uid, actual_batch_size, batch_position = batch_assignment[replicate]
            path_uid = f"{scenario.scenario_id}__r{replicate:04d}__p{prompt_id:02d}"
            paths.append(
                PathSpec(
                    path_uid=path_uid,
                    scenario_id=scenario.scenario_id,
                    schedule_id=scenario.schedule.schedule_id,
                    intended_regions=scenario.schedule.intended_regions,
                    temperature=scenario.temperature,
                    horizon=scenario.horizon,
                    replicate=replicate,
                    prompt_id=prompt_id,
                    scenario_seed_words=seed_words,
                    generation_batch_uid=batch_uid,
                    generation_batch_size=actual_batch_size,
                    generation_batch_position=batch_position,
                )
            )
    if len({p.path_uid for p in paths}) != len(paths):
        raise AssertionError("duplicate path UID")
    return paths


def rng_for_path(spec: PathSpec, stream_id: int) -> np.random.Generator:
    if stream_id not in (
        TABLE_STREAM,
        ORDINARY_STREAM,
        PIVOT_STREAM,
        TOURNAMENT_SAMPLE_STREAM,
    ):
        raise ValueError(f"unknown stream {stream_id}")
    # Each component must be a nonnegative uint32 for SeedSequence.spawn_key.
    spawn_key = (*spec.scenario_seed_words, int(spec.replicate), int(stream_id))
    seed_sequence = np.random.SeedSequence(MASTER_SEED, spawn_key=spawn_key)
    return np.random.Generator(np.random.PCG64DXSM(seed_sequence))


def rng_seed_words(spec: PathSpec, stream_id: int) -> tuple[int, ...]:
    spawn_key = (*spec.scenario_seed_words, int(spec.replicate), int(stream_id))
    sequence = np.random.SeedSequence(MASTER_SEED, spawn_key=spawn_key)
    return tuple(int(x) for x in sequence.generate_state(4))


def prompt_sha256(prompt_id: int) -> str:
    return sha256_text(PROMPTS[prompt_id])


def design_payload(
    profile: str,
    scenarios: Sequence[ScenarioSpec],
    batch_size: int,
    num_threads: int,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generation_algorithm_version": GENERATION_ALGORITHM_VERSION,
        "implementation_sha256": {
            "generator": sha256_file(Path(__file__).resolve()),
            "tournament_math": sha256_file(
                ROOT / "tournament_watermark.py"
            ),
        },
        "profile": profile,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dtype": LOCKED_DTYPE,
        "master_seed": MASTER_SEED,
        "bit_generator": BIT_GENERATOR_NAME,
        "stream_roles": {
            "tournament_table": TABLE_STREAM,
            "ordinary_token": ORDINARY_STREAM,
            "randomized_pit_v": PIVOT_STREAM,
            "tournament_final_sample": TOURNAMENT_SAMPLE_STREAM,
        },
        "planned_batch_size": batch_size,
        "torch_num_threads": num_threads,
        "torch_num_interop_threads": TORCH_INTEROP_THREADS,
        "batching_rule": (
            "stable chunks within identical scenario, prompt, schedule, temperature and horizon; "
            "last chunk may be smaller"
        ),
        "decoder": {
            "temperatures": sorted({float(s.temperature) for s in scenarios}),
            "full_vocabulary": True,
            "top_k": None,
            "top_p": None,
            "repetition_penalty": None,
            "stop_at_eos": False,
            "watermark_sampler": "exact full-vocabulary 30-layer Tournament recursion",
            "null_sampler": "ordinary full-vocabulary categorical sample",
            "tournament_layers": LAYERS,
            "competitors_per_match": 2,
            "g_distribution": "iid Bernoulli(1/2)",
            "fresh_full_table_at_watermarked_positions": True,
            "ordinary_lazy_selected_g_bits": True,
            "repeated_context_masking": False,
            "ideal_random_function_no_hash_or_context_prf": True,
            "separate_table_ordinary_pit_and_tournament_sample_streams": True,
        },
        "prompt_scored": False,
        "prompt_hashes": [prompt_sha256(i) for i in range(len(PROMPTS))],
        "paths_per_temperature": PATHS_PER_TEMPERATURE,
        "total_model_paths": TOTAL_MODEL_PATHS if profile == "study" else len(build_manifest(scenarios, batch_size)),
        "scenarios": [
            {
                **asdict(s),
                "schedule": asdict(s.schedule),
            }
            for s in scenarios
        ],
    }


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_manifest_atomic(path: Path, specs: Sequence[PathSpec]) -> None:
    columns = (
        "path_uid",
        "scenario_id",
        "schedule_id",
        "intended_regions_json",
        "temperature",
        "horizon",
        "replicate",
        "prompt_id",
        "prompt_sha256",
        "generation_batch_uid",
        "generation_batch_size",
        "generation_batch_position",
        "table_seed_words",
        "ordinary_seed_words",
        "pivot_v_seed_words",
        "tournament_sample_seed_words",
        "checkpoint_relpath",
    )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "path_uid": spec.path_uid,
                    "scenario_id": spec.scenario_id,
                    "schedule_id": spec.schedule_id,
                    "intended_regions_json": canonical_json(spec.intended_regions),
                    "temperature": format(spec.temperature, ".17g"),
                    "horizon": spec.horizon,
                    "replicate": spec.replicate,
                    "prompt_id": spec.prompt_id,
                    "prompt_sha256": prompt_sha256(spec.prompt_id),
                    "generation_batch_uid": spec.generation_batch_uid,
                    "generation_batch_size": spec.generation_batch_size,
                    "generation_batch_position": spec.generation_batch_position,
                    "table_seed_words": ",".join(map(str, rng_seed_words(spec, TABLE_STREAM))),
                    "ordinary_seed_words": ",".join(map(str, rng_seed_words(spec, ORDINARY_STREAM))),
                    "pivot_v_seed_words": ",".join(map(str, rng_seed_words(spec, PIVOT_STREAM))),
                    "tournament_sample_seed_words": ",".join(
                        map(str, rng_seed_words(spec, TOURNAMENT_SAMPLE_STREAM))
                    ),
                    "checkpoint_relpath": f"checkpoints/{spec.path_uid}.npz",
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def array_content_sha256(arrays: dict[str, np.ndarray], metadata_without_hash: dict) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json(metadata_without_hash).encode("utf-8"))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(canonical_json(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def checkpoint_metadata(spec: PathSpec, generated: dict, versions: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generation_algorithm_version": GENERATION_ALGORITHM_VERSION,
        "implementation_sha256": {
            "generator": sha256_file(Path(__file__).resolve()),
            "tournament_math": sha256_file(
                ROOT / "tournament_watermark.py"
            ),
        },
        "path_spec": asdict(spec),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "prompt_sha256": prompt_sha256(spec.prompt_id),
        "decoder": {
            "temperature": spec.temperature,
            "full_vocabulary": True,
            "top_k": None,
            "top_p": None,
            "stop_at_eos": False,
        },
        "table_seed_words": rng_seed_words(spec, TABLE_STREAM),
        "ordinary_seed_words": rng_seed_words(spec, ORDINARY_STREAM),
        "pivot_v_seed_words": rng_seed_words(spec, PIVOT_STREAM),
        "tournament_sample_seed_words": rng_seed_words(
            spec, TOURNAMENT_SAMPLE_STREAM
        ),
        "generation_seconds": float(generated["generation_seconds"]),
        "batch_generation_seconds": float(generated["batch_generation_seconds"]),
        "generation_batch_uid": spec.generation_batch_uid,
        "generation_batch_size": spec.generation_batch_size,
        "generation_batch_position": spec.generation_batch_position,
        "versions": versions,
    }


def save_checkpoint_atomic(path: Path, spec: PathSpec, generated: dict, versions: dict) -> None:
    arrays = {
        "token_id": np.asarray(generated["token_id"], dtype=np.int32),
        "selected_g": np.asarray(generated["selected_g"], dtype=np.uint8),
        "pivot_s": np.asarray(generated["pivot_s"], dtype=np.uint8),
        "pivot_v": np.asarray(generated["pivot_v"], dtype=np.float64),
        "pivot_y": np.asarray(generated["pivot_y"], dtype=np.float64),
        "calibrator_l": np.asarray(generated["calibrator_l"], dtype=np.float64),
        "is_watermarked": np.asarray(generated["is_watermarked"], dtype=np.bool_),
        "selected_model_probability": np.asarray(
            generated["selected_model_probability"], dtype=np.float64
        ),
        "max_model_probability": np.asarray(generated["max_model_probability"], dtype=np.float64),
        "model_entropy_nats": np.asarray(generated["model_entropy_nats"], dtype=np.float64),
        "is_eos": np.asarray(generated["is_eos"], dtype=np.bool_),
        "max_mass_correction": np.asarray(
            generated["max_mass_correction"], dtype=np.float64
        ),
    }
    metadata = checkpoint_metadata(spec, generated, versions)
    metadata["content_sha256"] = array_content_sha256(arrays, metadata)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        continuation=np.asarray(generated["continuation"]),
        metadata_json=np.asarray(canonical_json(metadata)),
    )
    # Force bytes to disk before the atomic rename.
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_and_validate_checkpoint(path: Path, expected: PathSpec | None = None) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "token_id",
            "selected_g",
            "pivot_s",
            "pivot_v",
            "pivot_y",
            "calibrator_l",
            "is_watermarked",
            "selected_model_probability",
            "max_model_probability",
            "model_entropy_nats",
            "is_eos",
            "max_mass_correction",
            "continuation",
            "metadata_json",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"checkpoint {path} is missing {sorted(missing)}")
        arrays = {name: np.asarray(data[name]) for name in required if name not in {"continuation", "metadata_json"}}
        continuation = str(data["continuation"].item())
        metadata = json.loads(str(data["metadata_json"].item()))

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"checkpoint {path} has wrong schema version")
    if metadata.get("generation_algorithm_version") != GENERATION_ALGORITHM_VERSION:
        raise ValueError(f"checkpoint {path} has the wrong generation-algorithm version")
    expected_implementation = {
        "generator": sha256_file(Path(__file__).resolve()),
        "tournament_math": sha256_file(
            ROOT / "tournament_watermark.py"
        ),
    }
    if metadata.get("implementation_sha256") != expected_implementation:
        raise ValueError(f"checkpoint {path} was produced by different source code")
    if metadata.get("model_name") != MODEL_NAME or metadata.get("model_revision") != MODEL_REVISION:
        raise ValueError(f"checkpoint {path} was generated from a different model checkpoint")
    content_hash = metadata.get("content_sha256")
    metadata_without_hash = dict(metadata)
    metadata_without_hash.pop("content_sha256", None)
    observed_hash = array_content_sha256(arrays, metadata_without_hash)
    if content_hash != observed_hash:
        raise ValueError(f"checkpoint {path} failed content hash validation")

    if expected is not None:
        saved_spec = metadata.get("path_spec")
        # JSON turns tuples into lists, so canonical comparison is intentional.
        if canonical_json(saved_spec) != canonical_json(asdict(expected)):
            raise ValueError(f"checkpoint {path} belongs to a different path specification")
        if metadata.get("prompt_sha256") != prompt_sha256(expected.prompt_id):
            raise ValueError(f"checkpoint {path} was generated from a different prompt")
        if float(metadata.get("decoder", {}).get("temperature", math.nan)) != expected.temperature:
            raise ValueError(f"checkpoint {path} was generated at a different temperature")
        n = expected.horizon
        for name, array in arrays.items():
            expected_shape = (n, LAYERS) if name == "selected_g" else (n,)
            if array.shape != expected_shape:
                raise ValueError(
                    f"checkpoint {path} has shape {array.shape} for {name}, "
                    f"expected {expected_shape}"
                )
        expected_mask = schedule_mask(
            ScheduleSpec(expected.schedule_id, expected.intended_regions), expected.horizon
        )
        if not np.array_equal(arrays["is_watermarked"], expected_mask):
            raise ValueError(f"checkpoint {path} has the wrong watermark schedule")

    pivots = arrays["pivot_y"]
    if not np.all(np.isfinite(pivots)) or np.any(pivots <= 0.0) or np.any(pivots >= 1.0):
        raise ValueError(f"checkpoint {path} has invalid pivots")
    selected_g = arrays["selected_g"]
    if np.any((selected_g != 0) & (selected_g != 1)):
        raise ValueError(f"checkpoint {path} contains non-Bernoulli selected g-values")
    if not np.array_equal(selected_g.sum(axis=1), arrays["pivot_s"]):
        raise ValueError(f"checkpoint {path} has an inconsistent Tournament score")
    expected_y = binomial_randomized_pit(arrays["pivot_s"], arrays["pivot_v"])
    if not np.array_equal(expected_y, arrays["pivot_y"]):
        raise ValueError(f"checkpoint {path} has an inconsistent randomized PIT")
    if not np.array_equal(stable_calibrator(expected_y), arrays["calibrator_l"]):
        raise ValueError(f"checkpoint {path} has an inconsistent calibrator trace")
    probabilities = arrays["selected_model_probability"]
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0) or not np.all(np.isfinite(probabilities)):
        raise ValueError(f"checkpoint {path} has invalid selected-token probabilities")
    return {**arrays, "continuation": continuation, "metadata": metadata}


def validate_batch(specs: Sequence[PathSpec]) -> None:
    if not specs:
        raise ValueError("a generation batch cannot be empty")
    first = specs[0]
    invariant = (
        first.generation_batch_uid,
        first.scenario_id,
        first.prompt_id,
        first.schedule_id,
        first.intended_regions,
        first.temperature,
        first.horizon,
    )
    for expected_position, spec in enumerate(specs):
        observed = (
            spec.generation_batch_uid,
            spec.scenario_id,
            spec.prompt_id,
            spec.schedule_id,
            spec.intended_regions,
            spec.temperature,
            spec.horizon,
        )
        if observed != invariant:
            raise ValueError("batch members do not share all generation inputs")
        if spec.generation_batch_size != len(specs):
            raise ValueError("manifest batch size does not match actual batch")
        if spec.generation_batch_position != expected_position:
            raise ValueError("batch members are not in their locked positions")


def generate_path_batch(
    model,
    tokenizer,
    prompt_ids,
    specs: Sequence[PathSpec],
    torch,
    logsumexp,
) -> list[dict]:
    """Generate one stable homogeneous batch with independent path RNGs.

    The thirty full-vocabulary Bernoulli layers are materialized only at
    watermarked positions. At ordinary positions only the selected token's
    thirty unused coordinates are drawn, the exact lazy-sampling equivalent.
    """

    del logsumexp
    validate_batch(specs)
    first = specs[0]
    batch_size = len(specs)
    ordinary_rngs = [rng_for_path(spec, ORDINARY_STREAM) for spec in specs]
    pivot_rngs = [rng_for_path(spec, PIVOT_STREAM) for spec in specs]
    tournament_sample_rngs = [
        rng_for_path(spec, TOURNAMENT_SAMPLE_STREAM) for spec in specs
    ]
    watermark_mask = schedule_mask(
        ScheduleSpec(first.schedule_id, first.intended_regions), first.horizon
    )
    vocab_size = int(model.config.vocab_size)
    eos_id = tokenizer.eos_token_id
    device = prompt_ids.device
    table_generators = []
    for spec in specs:
        words = rng_seed_words(spec, TABLE_STREAM)
        seed64 = int(words[0]) | (int(words[1]) << 32)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed64)
        table_generators.append(generator)

    token_ids = np.empty((batch_size, first.horizon), dtype=np.int32)
    selected_g = np.empty((batch_size, first.horizon, LAYERS), dtype=np.uint8)
    pivot_s = np.empty((batch_size, first.horizon), dtype=np.uint8)
    pivot_v = np.empty((batch_size, first.horizon), dtype=np.float64)
    pivots = np.empty((batch_size, first.horizon), dtype=np.float64)
    calibrator_l = np.empty((batch_size, first.horizon), dtype=np.float64)
    selected_probs = np.empty((batch_size, first.horizon), dtype=np.float64)
    max_probs = np.empty((batch_size, first.horizon), dtype=np.float64)
    entropies = np.empty((batch_size, first.horizon), dtype=np.float64)
    is_eos = np.empty((batch_size, first.horizon), dtype=np.bool_)
    max_mass_correction = np.zeros((batch_size, first.horizon), dtype=np.float64)

    started = time.perf_counter()
    with torch.inference_mode():
        # Every row has an identical prompt, but repeat creates an independent
        # batch row and the model creates a fresh mutable KV cache for the batch.
        batch_prompt_ids = prompt_ids.repeat(batch_size, 1)
        initial = model(batch_prompt_ids, use_cache=True)
        next_logits = initial.logits[:, -1, :].detach()
        if next_logits.dtype != torch.float32:
            raise RuntimeError("OPT logits are not float32")
        past = initial.past_key_values

        for zero_t in range(first.horizon):
            scaled_logits = next_logits / first.temperature
            finite = torch.isfinite(scaled_logits)
            scaled_logits = torch.where(
                finite, scaled_logits, torch.full_like(scaled_logits, -torch.inf)
            )
            log_probs = torch.log_softmax(scaled_logits, dim=1)
            probs = torch.softmax(scaled_logits, dim=1)
            if not bool(torch.all(torch.isfinite(probs))) or not bool(
                torch.allclose(
                    probs.sum(dim=1),
                    torch.ones(batch_size, dtype=torch.float32, device=device),
                    rtol=2e-5,
                    atol=2e-5,
                )
            ):
                raise RuntimeError(
                    f"invalid NTP distribution for batch {first.generation_batch_uid}, "
                    f"t={zero_t + 1}"
                )

            if watermark_mask[zero_t]:
                tournament_probs = probs.clone()
                g_layers = []
                largest_correction = torch.zeros(
                    batch_size, dtype=torch.float32, device=device
                )
                for _layer in range(LAYERS):
                    g = torch.stack(
                        [
                            torch.randint(
                                0,
                                2,
                                (vocab_size,),
                                dtype=torch.uint8,
                                device=device,
                                generator=generator,
                            )
                            for generator in table_generators
                        ],
                        dim=0,
                    )
                    g_layers.append(g)
                    g_float = g.to(dtype=torch.float32)
                    q = torch.sum(tournament_probs * g_float, dim=1, keepdim=True)
                    updated_probs = tournament_probs * (1.0 + g_float - q)
                    tournament_probs, correction = normalize_float32_tournament_mass(
                        updated_probs
                    )
                    largest_correction = torch.maximum(
                        largest_correction, correction
                    )
                chosen_list = []
                for j in range(batch_size):
                    u = float(tournament_sample_rngs[j].random())
                    cdf = torch.cumsum(tournament_probs[j], dim=0)
                    chosen = int(
                        torch.searchsorted(
                            cdf,
                            torch.tensor(u, dtype=torch.float32, device=device),
                            right=False,
                        ).item()
                    )
                    chosen_list.append(min(chosen, vocab_size - 1))
                chosen_ids = np.asarray(chosen_list, dtype=np.int64)
                chosen_tensor = torch.as_tensor(chosen_ids, device=device)
                bits = torch.stack(
                    [
                        layer[torch.arange(batch_size, device=device), chosen_tensor]
                        for layer in g_layers
                    ],
                    dim=1,
                )
                selected_g[:, zero_t, :] = bits.cpu().numpy()
                max_mass_correction[:, zero_t] = (
                    largest_correction.double().cpu().numpy()
                )
            else:
                chosen_list = []
                for j in range(batch_size):
                    u = float(ordinary_rngs[j].random())
                    cdf = torch.cumsum(probs[j], dim=0)
                    chosen = int(
                        torch.searchsorted(
                            cdf,
                            torch.tensor(u, dtype=torch.float32, device=device),
                            right=False,
                        ).item()
                    )
                    chosen_list.append(min(chosen, vocab_size - 1))
                chosen_ids = np.asarray(chosen_list, dtype=np.int64)
                # Lazy sampling: these bits are drawn only after ordinary token
                # selection and therefore cannot influence that selection.
                selected_g[:, zero_t, :] = np.stack(
                    [
                        torch.randint(
                            0,
                            2,
                            (LAYERS,),
                            dtype=torch.uint8,
                            device=device,
                            generator=generator,
                        )
                        .cpu()
                        .numpy()
                        for generator in table_generators
                    ],
                    axis=0,
                )

            rows = np.arange(batch_size)
            token_ids[:, zero_t] = chosen_ids
            scores = selected_g[:, zero_t, :].sum(axis=1, dtype=np.uint16)
            pivot_s[:, zero_t] = scores.astype(np.uint8)
            v_now = np.asarray(
                [rng.random() for rng in pivot_rngs], dtype=np.float64
            )
            pivot_v[:, zero_t] = v_now
            y_now = binomial_randomized_pit(scores, v_now)
            pivots[:, zero_t] = y_now
            calibrator_l[:, zero_t] = stable_calibrator(y_now)
            rows_t = torch.arange(batch_size, device=device)
            chosen_t = torch.as_tensor(chosen_ids, dtype=torch.long, device=device)
            selected_probs[:, zero_t] = probs[rows_t, chosen_t].double().cpu().numpy()
            max_probs[:, zero_t] = probs.max(dim=1).values.double().cpu().numpy()
            entropy = -torch.sum(
                torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs)),
                dim=1,
            )
            entropies[:, zero_t] = entropy.double().cpu().numpy()
            if eos_id is None:
                is_eos[:, zero_t] = False
            else:
                is_eos[:, zero_t] = chosen_ids == eos_id

            chosen = torch.as_tensor(chosen_ids[:, None], dtype=torch.long, device=prompt_ids.device)
            updated = model(chosen, past_key_values=past, use_cache=True)
            next_logits = updated.logits[:, -1, :].detach()
            past = updated.past_key_values

    batch_seconds = time.perf_counter() - started
    results: list[dict] = []
    for j, spec in enumerate(specs):
        continuation = tokenizer.decode(
            token_ids[j].tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        results.append(
            {
                "token_id": token_ids[j],
                "selected_g": selected_g[j],
                "pivot_s": pivot_s[j],
                "pivot_v": pivot_v[j],
                "pivot_y": pivots[j],
                "calibrator_l": calibrator_l[j],
                "is_watermarked": watermark_mask.copy(),
                "selected_model_probability": selected_probs[j],
                "max_model_probability": max_probs[j],
                "model_entropy_nats": entropies[j],
                "is_eos": is_eos[j],
                "max_mass_correction": max_mass_correction[j],
                # Per-path share is useful for aggregation; total batch time is
                # retained separately so runtime accounting is never ambiguous.
                "generation_seconds": batch_seconds / batch_size,
                "batch_generation_seconds": batch_seconds,
                "continuation": continuation,
            }
        )
    return results


def install_design_files(
    output_dir: Path,
    profile: str,
    scenarios: Sequence[ScenarioSpec],
    specs: Sequence[PathSpec],
    batch_size: int,
    num_threads: int,
) -> str:
    payload = design_payload(profile, scenarios, batch_size, num_threads)
    design_hash = sha256_text(canonical_json(payload))
    record = {**payload, "design_sha256": design_hash, "number_of_paths": len(specs)}
    design_path = output_dir / "design.json"
    if design_path.exists():
        existing = json.loads(design_path.read_text(encoding="utf-8"))
        if existing.get("design_sha256") != design_hash:
            raise RuntimeError(
                f"{output_dir} already contains a different design; use a new output directory"
            )
    else:
        write_json_atomic(design_path, record)
        write_json_atomic(
            output_dir / "prompts.json",
            {
                "prompt_scored": False,
                "prompts": [
                    {"prompt_id": i, "sha256": prompt_sha256(i), "text": prompt}
                    for i, prompt in enumerate(PROMPTS)
                ],
            },
        )
        write_manifest_atomic(output_dir / "path_manifest.csv", specs)
    return design_hash


def select_paths(
    specs: Sequence[PathSpec],
    scenario_filter: Sequence[str],
    path_uid_filter: Sequence[str],
) -> list[PathSpec]:
    selected = list(specs)
    if scenario_filter:
        requested = set(scenario_filter)
        known = {spec.scenario_id for spec in selected}
        unknown = requested.difference(known)
        if unknown:
            raise ValueError(f"unknown scenario IDs: {sorted(unknown)}")
        selected = [spec for spec in selected if spec.scenario_id in requested]
    if path_uid_filter:
        requested = set(path_uid_filter)
        known = {spec.path_uid for spec in selected}
        unknown = requested.difference(known)
        if unknown:
            raise ValueError(f"unknown or filtered path UIDs: {sorted(unknown)}")
        selected = [spec for spec in selected if spec.path_uid in requested]
    return selected


def stable_batches(specs: Sequence[PathSpec]) -> list[list[PathSpec]]:
    """Recover the locked batches from a complete design manifest."""

    grouped: dict[str, list[PathSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.generation_batch_uid, []).append(spec)
    batches = []
    for members in grouped.values():
        members.sort(key=lambda spec: spec.generation_batch_position)
        validate_batch(members)
        batches.append(members)
    return batches


def select_batches_to_run(
    all_specs: Sequence[PathSpec],
    selected_specs: Sequence[PathSpec],
    checkpoint_dir: Path,
    max_new_paths: int | None,
) -> tuple[list[tuple[list[PathSpec], list[PathSpec]]], list[PathSpec], list[PathSpec]]:
    """Return stable full batches and the selected missing members to save.

    A partially saved batch is recomputed with all original rows, but only its
    missing selected members are written.  This keeps the numerical computation
    invariant to crashes and filters.
    """

    selected_by_uid = {spec.path_uid: spec for spec in selected_specs}
    valid_existing: list[PathSpec] = []
    missing: list[PathSpec] = []
    for spec in selected_specs:
        checkpoint = checkpoint_dir / f"{spec.path_uid}.npz"
        if checkpoint.exists():
            load_and_validate_checkpoint(checkpoint, spec)
            valid_existing.append(spec)
        else:
            missing.append(spec)

    missing_uids = {spec.path_uid for spec in missing}
    jobs: list[tuple[list[PathSpec], list[PathSpec]]] = []
    if max_new_paths == 0:
        return jobs, valid_existing, missing
    selected_missing_so_far = 0
    for full_batch in stable_batches(all_specs):
        to_save = [
            spec
            for spec in full_batch
            if spec.path_uid in selected_by_uid and spec.path_uid in missing_uids
        ]
        if not to_save:
            continue
        if max_new_paths is not None and selected_missing_so_far + len(to_save) > max_new_paths:
            if jobs:
                break
            # Never split a locked batch.  If the first batch is larger than
            # the requested limit, run it and report the soft-limit behavior.
            print(
                f"Note: first locked batch has {len(to_save)} selected missing paths, "
                f"exceeding --max-new-paths={max_new_paths}; running the intact batch.",
                flush=True,
            )
        jobs.append((full_batch, to_save))
        selected_missing_so_far += len(to_save)
        if max_new_paths is not None and selected_missing_so_far >= max_new_paths:
            break
    return jobs, valid_existing, missing


def runtime_versions(torch, transformers, scipy) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "transformers": transformers.__version__,
    }


def make_runtime_fingerprint(
    *,
    versions: dict,
    design_sha256: str,
    profile: str,
    batch_size: int,
    checkpoint_commit: str,
    dtype: str,
    device: str,
) -> dict:
    """Build the exact, time-invariant numerical-runtime lock."""

    return {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "design_sha256": design_sha256,
        "generation_algorithm_version": GENERATION_ALGORITHM_VERSION,
        "planned_batch_size": int(batch_size),
        "python": str(versions["python"]),
        "platform": str(versions["platform"]),
        "numpy": str(versions["numpy"]),
        "scipy": str(versions["scipy"]),
        "torch": str(versions["torch"]),
        "transformers": str(versions["transformers"]),
        "model_name": MODEL_NAME,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_loaded": str(checkpoint_commit),
        "dtype": str(dtype),
        "device": str(device),
        "torch_num_threads": int(versions["torch_num_threads"]),
        "torch_num_interop_threads": int(versions["torch_num_interop_threads"]),
    }


def _fingerprint_differences(expected: object, observed: object, prefix: str = "") -> list[str]:
    """Return concise leaf-level differences for a mismatch error."""

    if isinstance(expected, dict) and isinstance(observed, dict):
        differences: list[str] = []
        keys = sorted(set(expected).union(observed))
        for key in keys:
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences.append(f"{name}: unexpected={observed[key]!r}")
            elif key not in observed:
                differences.append(f"{name}: missing (locked={expected[key]!r})")
            else:
                differences.extend(
                    _fingerprint_differences(expected[key], observed[key], name)
                )
        return differences
    if expected != observed:
        return [f"{prefix}: locked={expected!r}, observed={observed!r}"]
    return []


def establish_runtime_fingerprint(output_dir: Path, observed: dict) -> str:
    """Atomically create or exactly verify ``runtime_fingerprint.json``.

    A hard-link publication step provides atomic create-if-absent semantics: a
    concurrent process cannot overwrite a fingerprint that another process
    just established.  The temporary file is fully flushed before publication.
    """

    path = output_dir / "runtime_fingerprint.json"
    created = False
    if not path.exists():
        temporary = output_dir / (
            f".runtime_fingerprint.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(observed, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Atomic and non-overwriting on the local filesystem.
                os.link(temporary, path)
                created = True
            except FileExistsError:
                # Another process won the race; it must still match exactly.
                pass
        finally:
            temporary.unlink(missing_ok=True)

    try:
        locked = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read valid runtime fingerprint {path}: {error}") from error
    differences = _fingerprint_differences(locked, observed)
    if differences:
        detail = "; ".join(differences[:12])
        if len(differences) > 12:
            detail += f"; plus {len(differences) - 12} more difference(s)"
        raise RuntimeError(
            f"runtime fingerprint mismatch in {path}; refusing to mix numerical "
            f"runtimes: {detail}"
        )
    return "created" if created else "matched"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--profile",
        choices=(
            "study",
            "development",
            "smoke",
        ),
        default="study",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "study_paths",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto selects CUDA when available; the selected device is runtime-locked",
    )
    parser.add_argument("--dry-run", action="store_true", help="write/validate the design and manifest without loading OPT")
    parser.add_argument("--validate-only", action="store_true", help="validate existing selected checkpoints; do not generate")
    parser.add_argument("--scenario", action="append", default=[], help="exact scenario ID; may be repeated")
    parser.add_argument("--path-uid", action="append", default=[], help="exact path UID; may be repeated")
    parser.add_argument(
        "--max-new-paths",
        type=int,
        default=None,
        help="safe chunk size for an overnight invocation; existing checkpoints do not count",
    )
    parser.add_argument("--num-threads", type=int, default=DEFAULT_TORCH_THREADS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "locked maximum generation batch size; paths are batched only when "
            "scenario, prompt, schedule, temperature and horizon are identical"
        ),
    )
    parser.add_argument("--smoke-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    if args.max_new_paths is not None and args.max_new_paths < 0:
        parser.error("--max-new-paths must be nonnegative")
    if args.num_threads <= 0:
        parser.error("--num-threads must be positive")
    if args.batch_size != DEFAULT_BATCH_SIZE:
        parser.error(f"--batch-size is locked to {DEFAULT_BATCH_SIZE}")
    if args.num_threads != DEFAULT_TORCH_THREADS:
        parser.error(f"--num-threads is locked to {DEFAULT_TORCH_THREADS}")
    if args.dry_run and args.validate_only:
        parser.error("--dry-run and --validate-only are mutually exclusive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    continuation_dir = output_dir / "continuations"
    checkpoint_dir.mkdir(exist_ok=True)
    continuation_dir.mkdir(exist_ok=True)

    scenarios = scenarios_for_profile(args.profile, args.smoke_tokens)
    all_specs = build_manifest(scenarios, batch_size=args.batch_size)
    design_hash = install_design_files(
        output_dir,
        args.profile,
        scenarios,
        all_specs,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )
    selected = select_paths(all_specs, args.scenario, args.path_uid)
    print(
        f"profile={args.profile}; design={design_hash}; manifest_paths={len(all_specs)}; "
        f"selected_paths={len(selected)}",
        flush=True,
    )
    if args.dry_run:
        print(f"Dry run complete. Design files are in {output_dir}", flush=True)
        return 0

    jobs, valid_existing, pending = select_batches_to_run(
        all_specs,
        selected,
        checkpoint_dir,
        max_new_paths=None if args.validate_only else args.max_new_paths,
    )
    print(
        f"planned_batch_size={args.batch_size}; valid existing checkpoints={len(valid_existing)}; "
        f"selected pending={len(pending)}; locked batches this invocation={len(jobs)}",
        flush=True,
    )
    if args.validate_only:
        if pending:
            print(f"Validation incomplete: {len(pending)} selected checkpoints are absent.", file=sys.stderr)
            return 2
        print("Every selected checkpoint passed schema, specification, shape and content-hash checks.")
        return 0

    if not jobs:
        print("Nothing to generate.", flush=True)
        return 0

    # Heavy dependencies are intentionally imported only after the manifest and
    # existing checkpoints have been checked.  Dry runs need only NumPy.
    import scipy
    import torch
    import transformers
    from scipy.special import logsumexp
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    versions = runtime_versions(torch, transformers, scipy)
    print(f"Loading {MODEL_NAME}@{MODEL_REVISION} on {device} (float32).", flush=True)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
        dtype=torch.float32,
    )
    model.to(device)
    model.eval()
    model_load_seconds = time.perf_counter() - load_start
    checkpoint_commit = getattr(model.config, "_commit_hash", None)
    if checkpoint_commit != MODEL_REVISION:
        raise RuntimeError(f"checkpoint mismatch: loaded {checkpoint_commit}, expected {MODEL_REVISION}")
    model_dtype = str(next(model.parameters()).dtype)
    if model_dtype != LOCKED_DTYPE:
        raise RuntimeError("the predeclared generation dtype is torch.float32")

    runtime_fingerprint = make_runtime_fingerprint(
        versions=versions,
        design_sha256=design_hash,
        profile=args.profile,
        batch_size=args.batch_size,
        checkpoint_commit=checkpoint_commit,
        dtype=model_dtype,
        device=str(device),
    )
    fingerprint_status = establish_runtime_fingerprint(output_dir, runtime_fingerprint)
    print(
        f"Runtime fingerprint {fingerprint_status}: "
        f"{output_dir / 'runtime_fingerprint.json'}",
        flush=True,
    )

    # Cache tokenized immutable prompts, never mutable model KV caches.
    prompt_ids_by_id = {}
    job_specs = [spec for full_batch, _ in jobs for spec in full_batch]
    for prompt_id in sorted({spec.prompt_id for spec in job_specs}):
        ids = tokenizer(
            PROMPTS[prompt_id], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        max_horizon = max(spec.horizon for spec in job_specs if spec.prompt_id == prompt_id)
        if int(ids.shape[1]) + max_horizon > int(model.config.max_position_embeddings):
            raise RuntimeError(f"prompt {prompt_id} plus continuation exceeds OPT context length")
        prompt_ids_by_id[prompt_id] = ids

    run_start = time.perf_counter()
    generated_count = 0
    computed_path_count = 0
    batch_seconds: list[float] = []
    completed_path_uids: list[str] = []
    total_to_save = sum(len(to_save) for _, to_save in jobs)
    for batch_number, (full_batch, to_save) in enumerate(jobs, start=1):
        generated_batch = generate_path_batch(
            model,
            tokenizer,
            prompt_ids_by_id[full_batch[0].prompt_id],
            full_batch,
            torch,
            logsumexp,
        )
        generated_by_uid = {
            spec.path_uid: generated
            for spec, generated in zip(full_batch, generated_batch, strict=True)
        }
        batch_seconds.append(float(generated_batch[0]["batch_generation_seconds"]))
        computed_path_count += len(full_batch)
        for spec in to_save:
            generated = generated_by_uid[spec.path_uid]
            checkpoint_path = checkpoint_dir / f"{spec.path_uid}.npz"
            save_checkpoint_atomic(checkpoint_path, spec, generated, versions)
            # Re-open immediately: a checkpoint is counted only after full validation.
            load_and_validate_checkpoint(checkpoint_path, spec)
            write_text_atomic(
                continuation_dir / f"{spec.path_uid}.txt", generated["continuation"]
            )
            generated_count += 1
            completed_path_uids.append(spec.path_uid)

        mean_batch_seconds = float(np.mean(batch_seconds))
        eta_seconds = (len(jobs) - batch_number) * mean_batch_seconds
        print(
            f"[batch {batch_number}/{len(jobs)}] {full_batch[0].generation_batch_uid}: "
            f"computed={len(full_batch)}, saved={len(to_save)}, "
            f"{generated_batch[0]['batch_generation_seconds']:.2f}s; "
            f"saved total={generated_count}/{total_to_save}; ETA={eta_seconds / 60:.1f} min",
            flush=True,
        )

    run_record = {
        "schema_version": SCHEMA_VERSION,
        "generation_algorithm_version": GENERATION_ALGORITHM_VERSION,
        "profile": args.profile,
        "design_sha256": design_hash,
        "planned_batch_size": args.batch_size,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "checkpoint_commit": checkpoint_commit,
        "runtime_fingerprint_status": fingerprint_status,
        "runtime_fingerprint": runtime_fingerprint,
        "device": str(device),
        "dtype": model_dtype,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "versions": versions,
        "model_load_seconds": model_load_seconds,
        "generated_this_invocation": generated_count,
        "computed_paths_this_invocation": computed_path_count,
        "computed_batches_this_invocation": len(jobs),
        "selected_paths": len(selected),
        "pending_before_chunk_limit": len(pending),
        "wall_clock_generation_and_save_seconds": time.perf_counter() - run_start,
        "mean_model_batch_seconds": float(np.mean(batch_seconds)),
        "total_model_batch_seconds": float(np.sum(batch_seconds)),
        "completed_path_uids": completed_path_uids,
    }
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    write_json_atomic(output_dir / f"invocation_{timestamp}Z.json", run_record)
    print(f"Completed {generated_count} new paths. Outputs: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
