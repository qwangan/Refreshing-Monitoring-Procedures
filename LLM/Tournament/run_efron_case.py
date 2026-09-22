#!/usr/bin/env python3
"""Run the sentence-aligned Efron human-versus-Tournament case study.

A deterministic sentence-boundary optimizer fixes an approximately
80/100/80/100/80 source partition before generation. The two 100-token target
spans are replaced by complete, context-conditioned, 30-layer
Tournament-watermarked OPT-1.3B sentences. This is one descriptive path, never
a Monte Carlo estimate.
"""

from __future__ import annotations

import argparse
import codecs
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Iterable, Sequence

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sentence_aligned_case import (
    assign_output_intervals,
    build_sentence_aligned_plan,
    finalize_generated_block,
    set_reproducible_seeds,
    should_stop_generated_block,
    write_blind_text,
)
from case_artifacts import render_refreshing_process

from efron_contract import (
    ALLOWED_METRICS,
    BIT_GENERATOR,
    CASE_ID,
    CASE_MASTER_SEED,
    DETECTOR_STRATEGY,
    LAYERS,
    MODEL_DTYPE,
    MODEL_NAME,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    MODEL_VOCAB_SIZE,
    ORDINARY_STREAM,
    PIVOT_STREAM,
    PROMPT,
    ROOT,
    SOURCE_EXCERPT_SHA256,
    SOURCE_OPT_TOKEN_COUNT,
    SOURCE_OPT_TOKEN_SHA256,
    SOURCE_PDF_SHA256,
    SOURCE_URL,
    SWZ_CAP,
    TABLE_STREAM,
    TEMPERATURE,
    THRESHOLD_DISPLAY,
    THRESHOLD_EXACT,
    TOURNAMENT_SAMPLE_STREAM,
    validate_source,
)
from refreshing_swz import Region, path_metrics, run_refreshing_from_pivots
from tournament_watermark import binomial_randomized_pit, stable_calibrator


DEFAULT_SOURCE = ROOT / "efron_excerpt.txt"
DEFAULT_OUTPUT = ROOT.parents[1] / "results" / "llm" / "tournament_efron_case"
TORCH_THREADS = 8
MPL_CACHE = Path(os.environ.get("TOURNAMENT_MPLCONFIGDIR", "/tmp/tournament_opt13b_matplotlib"))
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

FIGURE_INK = "#25343D"
TRUE_REJECTION_COLOR = "#0072B2"
FALSE_REJECTION_COLOR = "#D55E00"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def token_ids_sha256(ids: Sequence[int]) -> str:
    values = np.asarray(ids, dtype="<i4")
    return hashlib.sha256(values.tobytes()).hexdigest()


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_frozen_source(path: Path = DEFAULT_SOURCE) -> str:
    if path.resolve() == DEFAULT_SOURCE.resolve():
        return validate_source(path)
    text = path.read_text(encoding="utf-8")
    if sha256_text(text) != SOURCE_EXCERPT_SHA256:
        raise RuntimeError("source excerpt is not the frozen Efron text")
    return text


def build_block_plan(
    tokenizer,
    source_text: str,
) -> tuple[list[dict], list[int]]:
    source_ids = list(tokenizer(source_text, add_special_tokens=False).input_ids)
    if not source_ids:
        raise RuntimeError("source tokenization is empty")
    decoded = tokenizer.decode(
        source_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != source_text:
        raise RuntimeError("Efron source failed exact OPT tokenizer round trip")
    if sha256_text(source_text) == SOURCE_EXCERPT_SHA256:
        if len(source_ids) != SOURCE_OPT_TOKEN_COUNT:
            raise RuntimeError(
                f"pinned tokenizer produced {len(source_ids)} source tokens; "
                f"expected {SOURCE_OPT_TOKEN_COUNT}"
            )
        if token_ids_sha256(source_ids) != SOURCE_OPT_TOKEN_SHA256:
            raise RuntimeError("pinned Efron source token IDs changed")
    blocks, planned_source_ids = build_sentence_aligned_plan(tokenizer, source_text)
    if source_ids != planned_source_ids:
        raise AssertionError("sentence-aligned planner changed the source tokenization")
    return blocks, source_ids


def report_union_mask(reports: Iterable, horizon: int) -> np.ndarray:
    mask = np.zeros(horizon, dtype=bool)
    for report in reports:
        start = int(report.interval_start)
        end = int(report.interval_end)
        if not 1 <= start <= end <= horizon:
            raise ValueError("report lies outside the monitored document")
        mask[start - 1 : end] = True
    return mask


def case_metrics(detector, regions: Sequence[Region], horizon: int) -> dict[str, float | int]:
    raw = path_metrics(detector.reports, regions, horizon=horizon)
    token_fdp = int(raw["false_positive_tokens"]) / max(int(raw["predicted_tokens"]), 1)
    metrics: dict[str, float | int] = {
        "reports": int(raw["reports"]),
        "token_power": float(raw["token_recall"]),
        "token_fdp": float(token_fdp),
        "token_iou": float(raw["token_iou"]),
        "final_report_fdp": float(raw["final_fdp"]),
        "report_ufdp": float(raw["uniform_fdp"]),
    }
    if tuple(metrics) != ALLOWED_METRICS:
        raise AssertionError("reported metric surface changed")
    return metrics


def validate_loaded_model(model, tokenizer) -> str:
    checkpoint = getattr(model.config, "_commit_hash", None)
    if checkpoint != MODEL_REVISION:
        raise RuntimeError(f"checkpoint mismatch: {checkpoint!r} != {MODEL_REVISION!r}")
    if getattr(model.config, "model_type", None) != "opt":
        raise RuntimeError("loaded checkpoint is not an OPT model")
    if int(model.config.vocab_size) != MODEL_VOCAB_SIZE:
        raise RuntimeError("OPT vocabulary size changed")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != MODEL_PARAMETER_COUNT:
        raise RuntimeError(
            f"parameter count mismatch: {parameter_count} != {MODEL_PARAMETER_COUNT}"
        )
    if str(next(model.parameters()).dtype) != MODEL_DTYPE:
        raise RuntimeError("model is not float32")
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("a fast tokenizer is required for exact figure offsets")
    return str(checkpoint)


def generate_mixed_document(tokenizer, model, torch, blocks: Sequence[dict]) -> dict[str, object]:
    prompt_ids = list(tokenizer(PROMPT, add_special_tokens=False).input_ids)
    if not prompt_ids:
        raise RuntimeError("prompt tokenization is empty")
    maximum_horizon = sum(
        len(block["fixed_token_ids"])
        if block["kind"] == "human"
        else int(block["maximum_tokens"])
        for block in blocks
    )
    if len(prompt_ids) + maximum_horizon > int(model.config.max_position_embeddings):
        raise RuntimeError("mixed document exceeds the pinned model context window")

    vocab_size = int(model.config.vocab_size)
    def seeded_rng(stream: int) -> np.random.Generator:
        return np.random.Generator(
            np.random.PCG64DXSM(
                np.random.SeedSequence(CASE_MASTER_SEED, spawn_key=(stream,))
            )
        )

    pivot_rng = seeded_rng(PIVOT_STREAM)
    tournament_sample_rng = seeded_rng(TOURNAMENT_SAMPLE_STREAM)
    ordinary_seed = np.random.SeedSequence(
        CASE_MASTER_SEED, spawn_key=(ORDINARY_STREAM,)
    ).generate_state(4)
    token_ids: list[int] = []
    selected_g: list[np.ndarray] = []
    pivot_s: list[int] = []
    pivot_v: list[float] = []
    pivots: list[float] = []
    calibrator_l: list[float] = []
    is_watermarked: list[bool] = []
    block_index: list[int] = []
    is_eos: list[bool] = []
    max_mass_correction: list[float] = []
    eos_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    table_words = np.random.SeedSequence(
        CASE_MASTER_SEED, spawn_key=(TABLE_STREAM,)
    ).generate_state(2)
    table_seed = int(table_words[0]) | (int(table_words[1]) << 32)
    table_generator = torch.Generator(device=device)
    table_generator.manual_seed(table_seed)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = time.perf_counter()
    with torch.inference_mode():
        initial = model(prompt_tensor, use_cache=True)
        next_logits = initial.logits[0, -1, :].detach()
        past = initial.past_key_values
        for current_block_index, block in enumerate(blocks):
            human_ids = block["fixed_token_ids"] if block["kind"] == "human" else None
            generated_ids: list[int] = []
            offset = 0
            while True:
                scaled_logits = next_logits / TEMPERATURE
                scaled_logits = torch.where(
                    torch.isfinite(scaled_logits),
                    scaled_logits,
                    torch.full_like(scaled_logits, -torch.inf),
                )
                ordinary_probs = torch.softmax(scaled_logits, dim=0)
                if ordinary_probs.dtype != torch.float32:
                    raise RuntimeError("Tournament generation is not float32")

                # A human token is fixed before the lazy table draw. The
                # separate ordinary sampling stream is intentionally unused.
                chosen_id = int(human_ids[offset]) if human_ids is not None else None
                if chosen_id is None:
                    p = ordinary_probs.clone()
                    g_layers = []
                    largest = 0.0
                    for _layer in range(LAYERS):
                        g = torch.randint(
                            0,
                            2,
                            (vocab_size,),
                            dtype=torch.uint8,
                            device=device,
                            generator=table_generator,
                        )
                        g_layers.append(g)
                        g_float = g.to(torch.float32)
                        q = torch.sum(p * g_float)
                        updated_probs = p * (1.0 + g_float - q)
                        if bool(torch.any(updated_probs < 0.0)):
                            raise RuntimeError("Tournament recursion produced negative mass")
                        total = torch.sum(updated_probs)
                        largest = max(largest, abs(float(total.item()) - 1.0))
                        p = updated_probs / total
                    u = float(tournament_sample_rng.random())
                    chosen_id = min(
                        int(
                            torch.searchsorted(
                                torch.cumsum(p, dim=0),
                                torch.tensor(u, dtype=torch.float32, device=device),
                            ).item()
                        ),
                        vocab_size - 1,
                    )
                    bits = np.asarray(
                        [int(g[chosen_id].item()) for g in g_layers], dtype=np.uint8
                    )
                    mass_correction = largest
                else:
                    # Lazy sampling of the selected coordinates of the unused
                    # full table is exactly equivalent at an ordinary position.
                    bits = (
                        torch.randint(
                            0,
                            2,
                            (LAYERS,),
                            dtype=torch.uint8,
                            device=device,
                            generator=table_generator,
                        )
                        .cpu()
                        .numpy()
                    )
                    mass_correction = 0.0

                score = int(bits.sum())
                randomizer = float(pivot_rng.random())
                pivot = float(
                    binomial_randomized_pit(
                        np.asarray([score]),
                        np.asarray([randomizer]),
                    )[0]
                )
                token_ids.append(chosen_id)
                selected_g.append(np.asarray(bits, dtype=np.uint8))
                pivot_s.append(score)
                pivot_v.append(randomizer)
                pivots.append(pivot)
                calibrator_l.append(float(stable_calibrator(np.asarray([pivot]))[0]))
                is_watermarked.append(block["kind"] == "watermarked")
                block_index.append(current_block_index)
                is_eos.append(eos_id is not None and chosen_id == eos_id)
                max_mass_correction.append(mass_correction)
                generated_ids.append(chosen_id)

                chosen = torch.tensor([[chosen_id]], dtype=torch.long, device=device)
                updated = model(chosen, past_key_values=past, use_cache=True)
                next_logits = updated.logits[0, -1, :].detach()
                past = updated.past_key_values
                offset += 1
                if len(token_ids) % 100 == 0:
                    print(f"generated/scored {len(token_ids)} monitored tokens", flush=True)
                if human_ids is not None:
                    if offset >= len(human_ids):
                        block["length"] = len(generated_ids)
                        block["output_token_ids"] = generated_ids
                        block["output_text_sha256"] = sha256_text(
                            tokenizer.decode(
                                generated_ids,
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            )
                        )
                        break
                elif should_stop_generated_block(
                    tokenizer,
                    generated_ids,
                    minimum_tokens=int(block["minimum_tokens"]),
                    maximum_tokens=int(block["maximum_tokens"]),
                ):
                    finalize_generated_block(tokenizer, block, generated_ids)
                    break

    horizon = assign_output_intervals(blocks)
    if len(token_ids) != horizon:
        raise AssertionError("generated arrays and block intervals disagree")
    expected_truth = np.asarray(
        [
            block["kind"] == "watermarked"
            for block in blocks
            for _ in range(int(block["length"]))
        ],
        dtype=bool,
    )
    truth_array = np.asarray(is_watermarked, dtype=bool)
    if not np.array_equal(truth_array, expected_truth):
        raise AssertionError("truth mask does not match the frozen block plan")
    print(f"generated/scored {horizon}/{horizon} monitored tokens", flush=True)
    return {
        "token_id": np.asarray(token_ids, dtype=np.int32),
        "selected_g": np.asarray(selected_g, dtype=np.uint8),
        "pivot_s": np.asarray(pivot_s, dtype=np.uint8),
        "pivot_v": np.asarray(pivot_v, dtype=np.float64),
        "pivot_y": np.asarray(pivots, dtype=np.float64),
        "calibrator_l": np.asarray(calibrator_l, dtype=np.float64),
        "is_watermarked": truth_array,
        "block_index": np.asarray(block_index, dtype=np.int8),
        "is_eos": np.asarray(is_eos, dtype=bool),
        "max_mass_correction": np.asarray(max_mass_correction, dtype=np.float64),
        "horizon": horizon,
        "generation_seconds": time.perf_counter() - started,
        "prompt_token_count": len(prompt_ids),
        "ordinary_stream_seed_words": [int(value) for value in ordinary_seed],
    }


def token_masks_to_character_masks(
    tokenizer,
    text: str,
    token_ids: Sequence[int],
    truth: Sequence[bool],
    selected: Sequence[bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Map saved OPT token masks without assuming unique BPE re-encoding."""

    if len(truth) != len(token_ids) or len(selected) != len(token_ids):
        raise ValueError("token masks have the wrong length")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded.input_ids) == list(token_ids):
        truth_chars = np.zeros(len(text), dtype=bool)
        selected_chars = np.zeros(len(text), dtype=bool)
        touched = np.zeros(len(text), dtype=bool)
        for (start, end), is_truth, is_selected in zip(
            encoded.offset_mapping,
            truth,
            selected,
            strict=True,
        ):
            if end > start:
                truth_chars[start:end] = bool(is_truth)
                selected_chars[start:end] = bool(is_selected)
                touched[start:end] = True
        visible = ~np.asarray([character.isspace() for character in text])
        if text and not np.all(touched | ~visible):
            raise RuntimeError("fast-tokenizer offsets left visible characters unmapped")
        return truth_chars, selected_chars

    pieces = [
        tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]
    if "".join(pieces) == text:
        truth_chars = np.zeros(len(text), dtype=bool)
        selected_chars = np.zeros(len(text), dtype=bool)
        cursor = 0
        for piece, is_truth, is_selected in zip(pieces, truth, selected, strict=True):
            end = cursor + len(piece)
            truth_chars[cursor:end] = bool(is_truth)
            selected_chars[cursor:end] = bool(is_selected)
            cursor = end
        return truth_chars, selected_chars

    return byte_level_token_masks_to_character_masks(
        tokenizer, text, token_ids, truth, selected
    )


def gpt2_byte_decoder() -> dict[str, int]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = list(byte_values)
    offset = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(256 + offset)
            offset += 1
    return {
        chr(character): byte
        for byte, character in zip(byte_values, unicode_values, strict=True)
    }


def byte_level_token_masks_to_character_masks(
    tokenizer,
    text: str,
    token_ids: Sequence[int],
    truth: Sequence[bool],
    selected: Sequence[bool],
) -> tuple[np.ndarray, np.ndarray]:
    byte_decoder = gpt2_byte_decoder()
    tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
    if isinstance(tokens, str):
        tokens = [tokens]
    if len(tokens) != len(token_ids):
        raise RuntimeError("tokenizer returned the wrong number of token strings")

    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    raw = bytearray()
    byte_owners: list[int] = []
    for index, (token_id, token) in enumerate(zip(token_ids, tokens, strict=True)):
        if int(token_id) in special_ids:
            token_bytes = tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).encode("utf-8")
        else:
            try:
                token_bytes = bytes(byte_decoder[character] for character in token)
            except KeyError as error:
                raise RuntimeError(
                    "saved tokens cannot be mapped with the OPT byte decoder"
                ) from error
        raw.extend(token_bytes)
        byte_owners.extend([index] * len(token_bytes))

    error_spans: list[tuple[int, int]] = []

    def record_decode_error(error: UnicodeDecodeError) -> tuple[str, int]:
        error_spans.append((error.start, error.end))
        return "\ufffd", error.end

    error_handler = f"efron_tournament_character_map_{id(error_spans)}"
    codecs.register_error(error_handler, record_decode_error)
    decoded = bytes(raw).decode("utf-8", errors=error_handler)
    if decoded != text:
        raise RuntimeError("saved OPT token bytes do not reproduce the decoded text")

    truth_chars = np.zeros(len(text), dtype=bool)
    selected_chars = np.zeros(len(text), dtype=bool)
    byte_cursor = 0
    error_cursor = 0
    for character_index, character in enumerate(text):
        if error_cursor < len(error_spans) and error_spans[error_cursor][0] == byte_cursor:
            end = error_spans[error_cursor][1]
            error_cursor += 1
        else:
            encoded_character = character.encode("utf-8")
            end = byte_cursor + len(encoded_character)
            if bytes(raw[byte_cursor:end]) != encoded_character:
                raise RuntimeError("could not align decoded OPT text with token bytes")
        owners = byte_owners[byte_cursor:end]
        truth_chars[character_index] = any(bool(truth[owner]) for owner in owners)
        selected_chars[character_index] = any(bool(selected[owner]) for owner in owners)
        byte_cursor = end
    if byte_cursor != len(raw) or error_cursor != len(error_spans):
        raise RuntimeError("OPT token-byte mapping did not consume the decoded text")
    return truth_chars, selected_chars


def wrap_masked_text(
    text: str,
    truth: np.ndarray,
    selected: np.ndarray,
    *,
    width: int = 92,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Collapse source paragraph breaks and wrap one continuous passage."""

    if len(text) != truth.size or len(text) != selected.size:
        raise ValueError("text and character masks differ in length")

    flat_characters: list[str] = []
    flat_truth: list[bool] = []
    flat_selected: list[bool] = []
    index = 0
    while index < len(text):
        if not text[index].isspace():
            flat_characters.append(text[index])
            flat_truth.append(bool(truth[index]))
            flat_selected.append(bool(selected[index]))
            index += 1
            continue
        run_end = index + 1
        while run_end < len(text) and text[run_end].isspace():
            run_end += 1
        if flat_characters and run_end < len(text):
            flat_characters.append(" ")
            flat_truth.append(bool(np.any(truth[index:run_end])))
            flat_selected.append(bool(np.any(selected[index:run_end])))
        index = run_end

    flat_text = "".join(flat_characters)
    truth_flat = np.asarray(flat_truth, dtype=bool)
    selected_flat = np.asarray(flat_selected, dtype=bool)
    lines: list[tuple[str, np.ndarray, np.ndarray]] = []
    cursor = 0
    while cursor < len(flat_text):
        target = min(cursor + width, len(flat_text))
        if target < len(flat_text):
            cut = flat_text.rfind(" ", cursor, target + 1)
            if cut <= cursor:
                cut = target
        else:
            cut = target
        visible_end = cut
        while visible_end > cursor and flat_text[visible_end - 1] == " ":
            visible_end -= 1
        if visible_end > cursor:
            lines.append(
                (
                    flat_text[cursor:visible_end],
                    truth_flat[cursor:visible_end].copy(),
                    selected_flat[cursor:visible_end].copy(),
                )
            )
        cursor = cut
        while cursor < len(flat_text) and flat_text[cursor] == " ":
            cursor += 1
    return lines


def styled_line_runs(
    line: str,
    truth: np.ndarray,
    selected: np.ndarray,
) -> list[dict[str, object]]:
    """Return the fail-closed four-state visual encoding for one text line."""

    if len(line) != truth.size or len(line) != selected.size:
        raise ValueError("line and character masks differ in length")
    if not line:
        return []
    states = truth.astype(np.uint8) + 2 * selected.astype(np.uint8)
    boundaries = np.flatnonzero(np.r_[True, states[1:] != states[:-1], True])
    runs: list[dict[str, object]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        is_ai = bool(truth[start])
        is_rejected = bool(selected[start])
        if is_rejected and is_ai:
            color = TRUE_REJECTION_COLOR
            rejection_class = "true"
        elif is_rejected:
            color = FALSE_REJECTION_COLOR
            rejection_class = "false"
        else:
            color = FIGURE_INK
            rejection_class = None
        runs.append(
            {
                "start": int(start),
                "end": int(end),
                "text": line[start:end],
                "is_ai": is_ai,
                "is_rejected": is_rejected,
                "fontstyle": "italic" if is_ai else "normal",
                "color": color,
                "underline": is_rejected,
                "rejection_class": rejection_class,
            }
        )
    return runs


def render_case_box(
    tokenizer,
    output_pdf: Path,
    output_png: Path,
    token_ids: np.ndarray,
    truth: np.ndarray,
    selected: np.ndarray,
    metrics: dict[str, float | int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    text = tokenizer.decode(
        token_ids.astype(int).tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    truth_chars, selected_chars = token_masks_to_character_masks(
        tokenizer,
        text,
        token_ids.astype(int).tolist(),
        truth.tolist(),
        selected.tolist(),
    )
    lines = wrap_masked_text(text, truth_chars, selected_chars)
    height = max(4.2, 2.0 + 0.175 * len(lines))
    fig = plt.figure(figsize=(8.0, height), facecolor="white")
    text_ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.91, bottom=0.105)

    text_ax.set_xlim(0.0, 1.0)
    text_ax.set_ylim(len(lines) + 0.3, -0.8)
    text_ax.set_xticks([])
    text_ax.set_yticks([])
    for spine in text_ax.spines.values():
        spine.set_color("#AAB7BE")
        spine.set_linewidth(0.9)
    fontsize = 8.2
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_width_pixels = float(text_ax.get_window_extent(renderer).width)
    if axis_width_pixels <= 0.0:
        raise RuntimeError("figure text box has invalid width")
    text_transform = blended_transform_factory(text_ax.transAxes, text_ax.transData)
    underline_transform = text_transform + ScaledTranslation(
        0.0,
        -0.92 * fontsize / 72.0,
        fig.dpi_scale_trans,
    )
    for line_number, (line, line_truth, line_selected) in enumerate(lines):
        y = line_number + 0.15
        x_pixels = 0.012 * axis_width_pixels
        for run in styled_line_runs(line, line_truth, line_selected):
            font = FontProperties(
                family="DejaVu Serif",
                style=str(run["fontstyle"]),
                size=fontsize,
            )
            run_text = str(run["text"])
            run_width_pixels = float(
                renderer.get_text_width_height_descent(run_text, font, ismath=False)[0]
            )
            x_start = x_pixels / axis_width_pixels
            x_end = (x_pixels + run_width_pixels) / axis_width_pixels
            text_ax.text(
                x_start,
                y,
                run_text,
                ha="left",
                va="top",
                fontproperties=font,
                color=str(run["color"]),
                transform=text_transform,
                zorder=1,
            )
            if bool(run["underline"]):
                text_ax.plot(
                    [x_start, x_end],
                    [y, y],
                    color=str(run["color"]),
                    linewidth=0.9,
                    solid_capstyle="butt",
                    transform=underline_transform,
                    zorder=2,
                )
            x_pixels += run_width_pixels
        if x_pixels > 0.988 * axis_width_pixels:
            raise RuntimeError("wrapped figure line exceeds the text box")

    metric_line = (
        f"Reports {int(metrics['reports'])}   |   "
        f"token Power {float(metrics['token_power']):.3f}   |   "
        f"token FDP {float(metrics['token_fdp']):.3f}   |   "
        f"token IoU {float(metrics['token_iou']):.3f}"
    )
    fig.suptitle(
        "Human/watermarked localization in Bradley Efron's Large-Scale Inference",
        x=0.5,
        y=0.995,
        fontsize=12.5,
        fontweight="bold",
        color=FIGURE_INK,
    )
    fig.text(
        0.5,
        0.958,
        metric_line,
        ha="center",
        va="top",
        fontsize=8.7,
        color=TRUE_REJECTION_COLOR,
    )
    legend = fig.legend(
        handles=[
            Line2D([], [], color="none", label="human text (roman)"),
            Line2D([], [], color="none", label="AI text (italic)"),
            Line2D([], [], color=TRUE_REJECTION_COLOR, linewidth=1.6, label="true rejection (AI)"),
            Line2D([], [], color=FALSE_REJECTION_COLOR, linewidth=1.6, label="false rejection (human)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.026),
        ncol=4,
        frameon=False,
        fontsize=7.8,
    )
    legend.get_texts()[1].set_fontstyle("italic")
    legend.get_texts()[2].set_color(TRUE_REJECTION_COLOR)
    legend.get_texts()[3].set_color(FALSE_REJECTION_COLOR)
    fig.text(
        0.01,
        0.006,
        "Source: Bradley Efron (2010); one prespecified descriptive path, not an FDR estimate.",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=FIGURE_INK,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_png, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def latex_figure_fragment(metrics: dict[str, float | int], pdf_name: str) -> str:
    return rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{{pdf_name}}}
\caption{{A prespecified mixed human/watermarked excerpt from Bradley Efron's
\emph{{Large-Scale Inference}}. Two spans of complete source sentences are
replaced by complete, Tournament-watermarked OPT-1.3B sentences generated from
the preceding context. Paragraph breaks are collapsed for display only. All AI
text is italic. Every rejected span is colored and
underlined: blue for a true rejection of AI text and vermillion for a false
rejection of human text. The one-path values are
Power={float(metrics['token_power']):.3f}, FDP={float(metrics['token_fdp']):.3f},
and IoU={float(metrics['token_iou']):.3f}. They are descriptive and are not
Monte Carlo estimates of power or FDR.}}
\label{{fig:efron-human-watermark-tournament}}
\end{{figure}}
"""


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "%": r"\%",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def direct_mixed_passage_tex(
    tokenizer,
    token_ids: np.ndarray,
    truth: np.ndarray,
    selected: np.ndarray,
    metrics: dict[str, float | int],
) -> str:
    """Render the complete passage as TeX text, never as a rasterized box."""

    text = tokenizer.decode(
        token_ids.astype(int).tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    truth_chars, selected_chars = token_masks_to_character_masks(
        tokenizer, text, token_ids.astype(int).tolist(), truth, selected
    )
    states = truth_chars.astype(np.uint8) + 2 * selected_chars.astype(np.uint8)
    boundaries = np.flatnonzero(np.r_[True, states[1:] != states[:-1], True])
    pieces: list[str] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        state = int(states[start])
        segment = text[start:end]
        cursor = 0
        while cursor < len(segment):
            whitespace = segment[cursor].isspace()
            stop = cursor + 1
            while stop < len(segment) and segment[stop].isspace() == whitespace:
                stop += 1
            part = segment[cursor:stop]
            if whitespace:
                pieces.append("\n\n\\par\n" if part.count("\n") >= 2 else " ")
            else:
                escaped = latex_escape(part)
                wrapper = {
                    0: None,
                    1: "PlainWatermarked",
                    2: "ReportedHuman",
                    3: "ReportedWatermarked",
                }[state]
                pieces.append(escaped if wrapper is None else rf"\{wrapper}{{{escaped}}}")
            cursor = stop
    body = "".join(pieces).strip()
    return rf"""\begingroup
\newcommand{{\PlainWatermarked}}[1]{{{{\itshape #1}}}}
\newcommand{{\ReportedWatermarked}}[1]{{{{\color{{WatermarkBlue}}\itshape\uline{{#1}}}}}}
\newcommand{{\ReportedHuman}}[1]{{{{\color{{HumanReport}}\uline{{#1}}}}}}
\begin{{figure}}[t]
\centering
\begin{{minipage}}{{0.98\textwidth}}
\centering
{{\large\sffamily\bfseries\color{{CaseInk}} Human/watermarked localization in Bradley Efron's Large-Scale Inference\par}}
\vspace{{5pt}}
\setlength{{\fboxsep}}{{8pt}}
\setlength{{\fboxrule}}{{0.55pt}}
\fcolorbox{{CaseBorder}}{{CaseTint}}{{\begin{{minipage}}{{0.945\linewidth}}
\footnotesize\setstretch{{1.02}}\color{{CaseInk}}\setlength{{\parindent}}{{0pt}}\sloppy
{body}
\end{{minipage}}}}
\vspace{{5pt}}
{{\scriptsize\sffamily Human text (roman)\quad
\PlainWatermarked{{Watermarked text (italic)}}\quad
\ReportedWatermarked{{Reported watermarked text}}\quad
\ReportedHuman{{Reported human text}}\par}}
\end{{minipage}}
\caption{{The complete prespecified Efron mixed document under the Tournament watermark. Every reported span is underlined; blue italic text is a true watermarked rejection and vermillion roman text is a false human rejection. The one-path values are Power={float(metrics['token_power']):.3f}, coverage FDP={float(metrics['token_fdp']):.3f}, and IoU={float(metrics['token_iou']):.3f}.}}
\label{{fig:efron-mixed-passage-tournament}}
\end{{figure}}
\endgroup
"""


def serialize_blocks(blocks: Sequence[dict]) -> list[dict]:
    return [
        {
            key: value
            for key, value in block.items()
            if key not in {"fixed_token_ids", "output_token_ids"}
        }
        for block in blocks
    ]


def alternative_regions(blocks: Sequence[dict]) -> tuple[Region, ...]:
    return tuple(
        Region(int(block["start"]), int(block["end"]))
        for block in blocks
        if block["kind"] == "watermarked"
    )


def block_coverage(blocks: Sequence[dict], selected: np.ndarray) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for block in blocks:
        start, end = int(block["start"]), int(block["end"])
        count = int(selected[start - 1 : end].sum())
        result.append(
            {
                "name": block["name"],
                "kind": block["kind"],
                "start": start,
                "end": end,
                "tokens": int(block["length"]),
                "selected_tokens": count,
            }
        )
    return result


def build_derived_artifacts(
    tokenizer,
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    blocks: Sequence[dict],
    metrics: dict[str, float | int],
) -> dict[str, object]:
    pdf_path = output_dir / "efron_tournament_case_box.pdf"
    png_path = output_dir / "efron_tournament_case_box.png"
    tex_path = output_dir / "efron_tournament_case_box.tex"
    direct_tex_path = output_dir / "efron_mixed_passage.tex"
    process_pdf_path = output_dir / "efron_tournament_refreshing_process.pdf"
    process_png_path = output_dir / "efron_tournament_refreshing_process.png"
    mixed_text = tokenizer.decode(
        arrays["token_id"].astype(int).tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    blind_path = write_blind_text(output_dir, mixed_text)
    table_csv_path = output_dir / "efron_tournament_case_metrics.csv"
    table_tex_path = output_dir / "efron_tournament_case_metrics.tex"
    render_case_box(
        tokenizer,
        pdf_path,
        png_path,
        arrays["token_id"],
        arrays["is_watermarked"],
        arrays["selected_by_any_report"],
        metrics,
    )
    write_text_atomic(tex_path, latex_figure_fragment(metrics, pdf_path.name))
    write_text_atomic(
        direct_tex_path,
        direct_mixed_passage_tex(
            tokenizer,
            arrays["token_id"],
            arrays["is_watermarked"],
            arrays["selected_by_any_report"],
            metrics,
        ),
    )
    detector = run_refreshing_from_pivots(
        np.asarray(arrays["pivot_y"], dtype=float),
        strategy=DETECTOR_STRATEGY,
        threshold=THRESHOLD_EXACT,
        cap=SWZ_CAP,
    )
    if not np.array_equal(
        report_union_mask(detector.reports, len(arrays["pivot_y"])),
        arrays["selected_by_any_report"],
    ):
        raise RuntimeError("refreshing-process replay differs from saved selection")
    render_refreshing_process(
        process_pdf_path,
        process_png_path,
        arrays,
        blocks,
        detector,
        threshold=THRESHOLD_EXACT,
        watermark_label="Tournament-watermarked OPT-1.3B",
    )
    write_text_atomic(
        table_csv_path,
        "watermark,reports,token_power,token_fdp,token_iou,final_report_fdp,report_ufdp\n"
        f"Tournament,{int(metrics['reports'])},{float(metrics['token_power']):.9f},"
        f"{float(metrics['token_fdp']):.9f},{float(metrics['token_iou']):.9f},"
        f"{float(metrics['final_report_fdp']):.9f},{float(metrics['report_ufdp']):.9f}\n",
    )
    write_text_atomic(
        table_tex_path,
        "Tournament & "
        f"{int(metrics['reports'])} & {float(metrics['token_power']):.3f} & "
        f"{float(metrics['token_fdp']):.3f} & {float(metrics['token_iou']):.3f} & "
        f"{float(metrics['final_report_fdp']):.3f} & "
        f"{float(metrics['report_ufdp']):.3f} \\\\\n",
    )
    return {
        "figure_pdf": pdf_path.name,
        "figure_pdf_sha256": sha256_file(pdf_path),
        "figure_png": png_path.name,
        "figure_png_sha256": sha256_file(png_path),
        "latex_fragment": tex_path.name,
        "latex_fragment_sha256": sha256_file(tex_path),
        "direct_mixed_passage_tex": direct_tex_path.name,
        "direct_mixed_passage_tex_sha256": sha256_file(direct_tex_path),
        "refreshing_process_pdf": process_pdf_path.name,
        "refreshing_process_pdf_sha256": sha256_file(process_pdf_path),
        "refreshing_process_png": process_png_path.name,
        "refreshing_process_png_sha256": sha256_file(process_png_path),
        "blind_mixed_text": blind_path.name,
        "blind_mixed_text_sha256": sha256_file(blind_path),
        "metrics_csv": table_csv_path.name,
        "metrics_csv_sha256": sha256_file(table_csv_path),
        "metrics_tex": table_tex_path.name,
        "metrics_tex_sha256": sha256_file(table_tex_path),
    }


def replay_saved(tokenizer, output_dir: Path) -> int:
    record_path = output_dir / "case.json"
    arrays_path = output_dir / "case_arrays.npz"
    if not record_path.is_file() or not arrays_path.is_file():
        raise RuntimeError("saved case.json/case_arrays.npz are missing")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["arrays_sha256"] != sha256_file(arrays_path):
        raise RuntimeError("saved array hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    detector = run_refreshing_from_pivots(
        arrays["pivot_y"],
        strategy=DETECTOR_STRATEGY,
        threshold=THRESHOLD_EXACT,
        cap=SWZ_CAP,
    )
    blocks = [dict(block) for block in record["blocks"]]
    regions = alternative_regions(blocks)
    metrics = case_metrics(detector, regions, int(arrays["pivot_y"].size))
    if metrics != record["metrics"]:
        raise RuntimeError("detector replay metrics differ from the saved record")
    selected = report_union_mask(detector.reports, int(arrays["pivot_y"].size))
    if not np.array_equal(selected, arrays["selected_by_any_report"]):
        raise RuntimeError("detector replay selection differs from saved arrays")
    derived = build_derived_artifacts(tokenizer, output_dir, arrays, blocks, metrics)
    manifest = {
        "case": record_path.name,
        "case_sha256": sha256_file(record_path),
        "arrays": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        **derived,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print("replayed saved pivots and rebuilt the box; no text was regenerated")
    return 0


def load_model_and_tokenizer(*, device_name: str, local_files_only: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_config = set_reproducible_seeds(torch, CASE_MASTER_SEED)
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda"
        if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
        dtype=torch.float32,
    )
    model.to(device)
    model.eval()
    checkpoint = validate_loaded_model(model, tokenizer)
    return tokenizer, model, torch, device, checkpoint, seed_config


def run_smoke(tokenizer, model, torch, source_text: str) -> int:
    source_ids = list(tokenizer(source_text, add_special_tokens=False).input_ids)
    smoke_source_ids = source_ids[:13]
    blocks = [
        {
            "name": "Smoke human source",
            "kind": "human",
            "fixed_token_ids": smoke_source_ids,
            "length": len(smoke_source_ids),
        }
    ]
    generated = generate_mixed_document(tokenizer, model, torch, blocks)
    pivots = np.asarray(generated["pivot_y"])
    if pivots.shape != (13,) or not np.all((pivots > 0) & (pivots < 1)):
        raise RuntimeError("model smoke failed")
    detector = run_refreshing_from_pivots(
        pivots,
        strategy=DETECTOR_STRATEGY,
        threshold=THRESHOLD_EXACT,
        cap=SWZ_CAP,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "profile": "non-scientific-smoke",
                "tokens": 13,
                "reports": len(detector.reports),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run a 13-token non-scientific model smoke and exit")
    parser.add_argument("--rebuild-derived", action="store_true", help="replay saved pivots and rebuild only the box")
    args = parser.parse_args(argv)

    validate_source(args.source)
    source_text = load_frozen_source(args.source)

    if args.rebuild_derived:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            local_files_only=args.local_files_only,
            use_fast=True,
        )
        return replay_saved(tokenizer, args.output_dir)

    if not args.smoke and any(
        (args.output_dir / name).exists() for name in ("case.json", "case_arrays.npz")
    ):
        raise SystemExit(
            "scientific case output already exists; the runner refuses regeneration or outcome-based retry"
        )

    print(f"loading {MODEL_NAME}@{MODEL_REVISION} in float32", flush=True)
    load_started = time.perf_counter()
    tokenizer, model, torch, device, checkpoint, seed_config = load_model_and_tokenizer(
        device_name=args.device,
        local_files_only=args.local_files_only,
    )
    model_load_seconds = time.perf_counter() - load_started
    print(f"model loaded on {device} in {model_load_seconds:.1f} seconds", flush=True)
    if args.smoke:
        return run_smoke(tokenizer, model, torch, source_text)

    blocks, source_ids = build_block_plan(tokenizer, source_text)
    generated = generate_mixed_document(tokenizer, model, torch, blocks)
    horizon = int(generated["horizon"])
    pivots = np.asarray(generated["pivot_y"], dtype=np.float64)
    detector = run_refreshing_from_pivots(
        pivots,
        strategy=DETECTOR_STRATEGY,
        threshold=THRESHOLD_EXACT,
        cap=SWZ_CAP,
    )
    regions = alternative_regions(blocks)
    metrics = case_metrics(detector, regions, horizon)
    selected = report_union_mask(detector.reports, horizon)

    arrays = {
        "token_id": np.asarray(generated["token_id"], dtype=np.int32),
        "selected_g": np.asarray(generated["selected_g"], dtype=np.uint8),
        "pivot_s": np.asarray(generated["pivot_s"], dtype=np.uint8),
        "pivot_v": np.asarray(generated["pivot_v"], dtype=np.float64),
        "pivot_y": pivots,
        "calibrator_l": np.asarray(generated["calibrator_l"], dtype=np.float64),
        "is_watermarked": np.asarray(generated["is_watermarked"], dtype=bool),
        "block_index": np.asarray(generated["block_index"], dtype=np.int8),
        "is_eos": np.asarray(generated["is_eos"], dtype=bool),
        "max_mass_correction": np.asarray(
            generated["max_mass_correction"], dtype=np.float64
        ),
        "selected_by_any_report": selected,
        "e_factor": detector.e_factor,
        "bet_fraction": detector.bet_fraction,
        "candidate_logwealth": detector.candidate_logwealth,
        "postreset_logwealth": detector.postreset_logwealth,
        "running_min_time": detector.running_min_time,
        "alarm": detector.alarm,
        "report_after_time": detector.report_after_time,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output_dir / "case_arrays.npz"
    record_path = args.output_dir / "case.json"
    write_npz_atomic(arrays_path, arrays)

    mixed_text = tokenizer.decode(
        arrays["token_id"].astype(int).tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    record = {
        "schema_version": 2,
        "case_id": CASE_ID,
        "single_path_descriptive_only": True,
        "no_seed_selection_regeneration_or_outcome_retry": True,
        "design_change_protocol": (
            "after inspecting seeds 20260921091501 and 20260921091502, the block geometry "
            "was changed to a locked 80/100/80/100/80 sentence-aligned target; this design "
            "uses one fresh seed and its first completed run is accepted without retry"
        ),
        "sentence_aligned_replacements": True,
        "generated_blocks_end_at_sentence_boundaries": True,
        "source_url": SOURCE_URL,
        "source_pdf_sha256": SOURCE_PDF_SHA256,
        "source_excerpt_sha256": SOURCE_EXCERPT_SHA256,
        "source_token_count": len(source_ids),
        "source_token_sha256": token_ids_sha256(source_ids),
        "mixed_text_sha256": sha256_text(mixed_text),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_checkpoint_commit": checkpoint,
        "model_dtype": str(next(model.parameters()).dtype),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "temperature": TEMPERATURE,
        "full_vocabulary": True,
        "watermark_rule": "exact full-vocabulary 30-layer Tournament recursion",
        "human_null_rule": "source token fixed before lazy selected-g sampling",
        "fresh_full_table_at_watermarked_positions": True,
        "ordinary_lazy_selected_g_bits": True,
        "stop_at_eos": False,
        "case_master_seed": CASE_MASTER_SEED,
        "bit_generator": BIT_GENERATOR,
        "table_stream": TABLE_STREAM,
        "ordinary_stream": ORDINARY_STREAM,
        "pivot_v_stream": PIVOT_STREAM,
        "tournament_sample_stream": TOURNAMENT_SAMPLE_STREAM,
        "ordinary_stream_use": "unused because null tokens are fixed human tokens",
        "ordinary_stream_seed_words": generated["ordinary_stream_seed_words"],
        "global_seed_configuration": seed_config,
        "prompt": PROMPT,
        "prompt_token_count": generated["prompt_token_count"],
        "threshold_exact": THRESHOLD_EXACT,
        "threshold_display": THRESHOLD_DISPLAY,
        "eprocess": f"weighted adaptive e-process, cap {SWZ_CAP:g}",
        "localizer": "refreshing plus last global minimum",
        "blocks": serialize_blocks(blocks),
        "alternative_intervals": [[region.start, region.end] for region in regions],
        "reports": [asdict(report) for report in detector.reports],
        "metrics": metrics,
        "audit_counts": {
            "horizon": horizon,
            "human_tokens": int((~arrays["is_watermarked"]).sum()),
            "watermarked_tokens": int(arrays["is_watermarked"].sum()),
            "selected_tokens": int(selected.sum()),
            "watermarked_selected": int((selected & arrays["is_watermarked"]).sum()),
            "human_selected": int((selected & ~arrays["is_watermarked"]).sum()),
            "block_coverage": block_coverage(blocks, selected),
        },
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "generation_seconds": generated["generation_seconds"],
            "torch_threads": TORCH_THREADS,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "arrays_file": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
    }
    record["record_sha256"] = hashlib.sha256(
        canonical_json(record).encode("utf-8")
    ).hexdigest()
    write_json_atomic(record_path, record)
    derived = build_derived_artifacts(tokenizer, args.output_dir, arrays, blocks, metrics)
    manifest = {
        "case": record_path.name,
        "case_sha256": sha256_file(record_path),
        "arrays": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        **derived,
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "completed", "metrics": metrics}, indent=2))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
