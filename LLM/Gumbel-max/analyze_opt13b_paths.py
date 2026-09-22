#!/usr/bin/env python3
"""Paired replay and localization analysis for the fresh OPT-1.3B paths.

This script never generates language-model text.  It validates every saved
checkpoint, replays all predeclared detector/localizer variants on the same
pivot sequence, and writes path-level and aggregate results.  The path—not an
atomic report—is the Monte Carlo unit.

The explicit ``boundary_gap_bridge`` metric is included to preserve the exact
event requested in the development analysis.  For adjacent true regions
``[a,b]`` and ``[c,d]``, a report ``(sigma,tau]`` is a boundary/gap bridge iff

    sigma < b and tau >= c - 1.

Thus, for ``[51,250]`` and ``[301,500]``, it is exactly
``sigma < 250 and tau >= 300``.  This is deliberately distinct from a merge
report, which must overlap both watermark regions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
GENERATION_DIR = HERE
DETECTOR_DIR = HERE
sys.path.insert(0, str(GENERATION_DIR))
sys.path.insert(0, str(DETECTOR_DIR))

import generate_fresh_opt13b as generation  # noqa: E402
from resetting_swz import (  # noqa: E402
    alpha_for_general_fdr_target,
    annotate_reports,
    general_dependence_bound,
    path_metrics,
    run_resetting_from_pivots,
)


SCHEMA_VERSION = 1
ANALYSIS_VERSION = "paired-opt13b-resetting-v5-cap05"
ADAPTIVE_CAP = 0.5
BOOTSTRAP_SEED = 202608020941
DEFAULT_BOOTSTRAP_REPLICATES = 5_000
PROTOCOL_CURRENT_SHA256 = "adfba3b6eeb53d1c10ab6f38da23c015b39794e04f4d35a07424ec0c39cceb17"
PROTOCOL_LOCKED_PREFIX_SHA256 = "8a0bf10c5ce4df650aea5c7d5985be9906a81d7b82c8f8d2c2e7ab0a1b5e4bf7"


FINAL_OUTPUT_FILENAMES = (
    "summary.csv",
    "summary.json",
    "pointwise_fdr.csv",
    "path_metrics.csv",
    "atomic_reports.jsonl",
    "paired_comparisons.csv",
    "generation_diagnostics.csv",
    "generation_randomness_audit.json",
    "batch_size_sensitivity.csv",
    "degeneration_sensitivity.csv",
)


def general_calibration_tag(target: float) -> str:
    percentage = 100.0 * target
    rounded = round(percentage)
    if not math.isclose(percentage, rounded, abs_tol=1e-10):
        raise ValueError("general FDR target must be an integer percentage")
    return f"general_fdr{int(rounded):02d}"


def method_configuration(target: float) -> tuple[tuple[str, ...], dict[str, str], tuple, str]:
    calibration = general_calibration_tag(target)
    general_one_shot = f"swz_one_shot_{calibration}"
    general_whole_block = f"swz_reset_whole_block_{calibration}"
    general_local = f"swz_reset_local_{calibration}"
    percentage = int(round(100.0 * target))
    order = (
        general_one_shot,
        general_whole_block,
        general_local,
    )
    labels = {
        general_one_shot: (
            f"SWZ adaptive, stop after first report (general {percentage}% threshold)"
        ),
        general_whole_block: (
            "SWZ adaptive + reset, whole block "
            f"(general {percentage}% threshold)"
        ),
        general_local: (
            "SWZ adaptive + reset + global-min localizer "
            f"(general {percentage}% threshold)"
        ),
    }
    pairs = (
        (general_local, general_one_shot),
        (general_local, general_whole_block),
    )
    return order, labels, pairs, calibration


METHOD_ORDER, METHOD_LABELS, PAIR_COMPARISONS, DEFAULT_GENERAL_CALIBRATION = (
    method_configuration(0.05)
)
PAIR_METRICS = (
    "all_regions_detected",
    "all_regions_separately_detected",
    "all_regions_single_report_covered",
    "all_regions_iou_ge_0p5",
    "all_regions_iou_ge_0p8",
    "all_regions_detected_timely",
    "token_iou",
    "mean_region_best_iou",
    "any_boundary_gap_bridge",
    "any_merge",
    "uniform_fdp",
    "reports",
)


PRIMARY_SCALAR_METRICS = (
    "reports",
    "any_report",
    "false_reports",
    "uniform_fdp",
    "final_fdp",
    "any_false_report",
    "all_regions_detected",
    "all_regions_detected_timely",
    "all_regions_fully_covered",
    "all_regions_separately_detected",
    "all_regions_single_report_covered",
    "all_regions_iou_ge_0p5",
    "all_regions_iou_ge_0p8",
    "regionwise_separate_recall",
    "regionwise_single_report_coverage",
    "regionwise_iou_ge_0p5",
    "regionwise_iou_ge_0p8",
    "regionwise_recall",
    "regionwise_timely_recall",
    "mean_region_detection_delay_given_timely_detection",
    "token_precision_given_report",
    "token_precision_zero_if_no_report",
    "token_recall",
    "token_f1",
    "token_iou",
    "mean_region_best_iou",
    "mean_region_start_error_given_overlap",
    "mean_region_end_error_given_overlap",
    "mean_absolute_region_start_error_given_overlap",
    "mean_absolute_region_end_error_given_overlap",
    "mean_report_purity",
    "any_merge",
    "merge_reports",
    "any_boundary_gap_bridge",
    "boundary_gap_bridge_reports",
    "any_full_null_gap_covered",
    "fragmented_regions",
    "fragmentation_excess_reports",
    "mean_eta",
    "max_eta",
    "mean_e_factor",
    "mean_null_e_factor",
    "mean_watermark_e_factor",
)


BINARY_METRICS = {
    "any_report",
    "any_false_report",
    "all_regions_detected",
    "all_regions_detected_timely",
    "all_regions_fully_covered",
    "all_regions_separately_detected",
    "all_regions_single_report_covered",
    "all_regions_iou_ge_0p5",
    "all_regions_iou_ge_0p8",
    "any_merge",
    "any_boundary_gap_bridge",
    "any_full_null_gap_covered",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(canonical_json(value.shape).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        canonical_json(_json_safe(value))
                        if isinstance(value, (tuple, list, dict, np.ndarray))
                        else _json_safe(value)
                    )
                    for key, value in row.items()
                }
            )
    os.replace(temporary, path)


def normal_interval(mean: float, mcse: float) -> tuple[float, float]:
    if not math.isfinite(mcse):
        return math.nan, math.nan
    return mean - 1.959963984540054 * mcse, mean + 1.959963984540054 * mcse


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _scenario_fields(spec: generation.PathSpec) -> dict:
    if spec.schedule_id == "all_null":
        return {"scenario_kind": "all_null", "region_length": None, "gap_length": None}
    match = re.fullmatch(r"two_l(\d{3})_g(\d{3})", spec.schedule_id)
    if match:
        return {
            "scenario_kind": "two_region",
            "region_length": int(match.group(1)),
            "gap_length": int(match.group(2)),
        }
    match = re.fullmatch(r"four_l(\d{3})_g(\d{3})", spec.schedule_id)
    if match:
        return {
            "scenario_kind": "four_region_stress",
            "region_length": int(match.group(1)),
            "gap_length": int(match.group(2)),
        }
    raise ValueError(f"unrecognized schedule ID {spec.schedule_id}")


def _whole_block_reports(reports: Sequence) -> list[dict]:
    converted: list[dict] = []
    for report in reports:
        row = report.as_dict()
        row["sigma"] = int(row["previous_tau"])
        row["interval_start"] = int(row["block_start"])
        row["interval_end"] = int(row["tau"])
        row["interval_length"] = int(row["tau"] - row["previous_tau"])
        converted.append(row)
    return converted


def replay_methods(
    pivots: np.ndarray,
    general_threshold: float,
    general_calibration: str = DEFAULT_GENERAL_CALIBRATION,
) -> tuple[dict[str, list[dict]], dict]:
    cumulative_general = run_resetting_from_pivots(
        pivots,
        strategy="adaptive_cumulative",
        threshold=general_threshold,
        cap=ADAPTIVE_CAP,
    )
    general_one_shot = f"swz_one_shot_{general_calibration}"
    general_whole_block = f"swz_reset_whole_block_{general_calibration}"
    general_local = f"swz_reset_local_{general_calibration}"
    reports = {
        general_one_shot: [cumulative_general.reports[0].as_dict()]
        if cumulative_general.reports
        else [],
        general_whole_block: _whole_block_reports(cumulative_general.reports),
        general_local: cumulative_general.report_dicts(),
    }
    traces = {
        general_one_shot: cumulative_general,
        general_whole_block: cumulative_general,
        general_local: cumulative_general,
    }
    return reports, traces


def boundary_gap_bridge_annotations(
    reports: Sequence[Mapping], regions: Sequence[Sequence[int]]
) -> tuple[int, tuple[int, ...]]:
    """Count reports satisfying the user's exact boundary/gap event."""

    bridged: set[int] = set()
    count = 0
    for report in reports:
        sigma = int(report["sigma"])
        tau = int(report["tau"])
        report_bridges = []
        for gap_index, (left, right) in enumerate(zip(regions, regions[1:]), start=1):
            left_end = int(left[1])
            right_start = int(right[0])
            if sigma < left_end and tau >= right_start - 1:
                report_bridges.append(gap_index)
                bridged.add(gap_index)
        count += int(bool(report_bridges))
    return count, tuple(sorted(bridged))


def strict_region_recovery_metrics(
    annotated: Sequence[Mapping], regions: Sequence[Sequence[int]]
) -> dict:
    """Recovery metrics that a single cross-gap bridge cannot satisfy alone.

    Weak ``all_regions_detected`` only requires overlap and is retained for
    continuity.  The metrics here ask for distinct non-merge reports, a
    single report covering an entire planted region, or a minimum best IoU.
    """

    separate: list[bool] = []
    single_covered: list[bool] = []
    iou05: list[bool] = []
    iou08: list[bool] = []
    for region_index, region in enumerate(regions):
        start, end = map(int, region)
        overlapping = [
            report
            for report in annotated
            if int(report["overlap_tokens_by_region"][region_index]) > 0
        ]
        separate.append(any(not bool(report["merge"]) for report in overlapping))
        single_covered.append(
            any(
                int(report["interval_start"]) <= start
                and int(report["interval_end"]) >= end
                for report in annotated
            )
        )
        best_iou = max(
            (float(report["iou_by_region"][region_index]) for report in annotated),
            default=0.0,
        )
        iou05.append(best_iou >= 0.5)
        iou08.append(best_iou >= 0.8)
    n_regions = len(regions)
    return {
        "regions_separately_detected": int(sum(separate)),
        "all_regions_separately_detected": bool(regions) and all(separate),
        "regionwise_separate_recall": sum(separate) / n_regions if n_regions else math.nan,
        "regions_single_report_covered": int(sum(single_covered)),
        "all_regions_single_report_covered": bool(regions) and all(single_covered),
        "regionwise_single_report_coverage": (
            sum(single_covered) / n_regions if n_regions else math.nan
        ),
        "regions_iou_ge_0p5": int(sum(iou05)),
        "all_regions_iou_ge_0p5": bool(regions) and all(iou05),
        "regionwise_iou_ge_0p5": sum(iou05) / n_regions if n_regions else math.nan,
        "regions_iou_ge_0p8": int(sum(iou08)),
        "all_regions_iou_ge_0p8": bool(regions) and all(iou08),
        "regionwise_iou_ge_0p8": sum(iou08) / n_regions if n_regions else math.nan,
    }


def fdp_timeline(annotated: Sequence[Mapping], horizon: int) -> np.ndarray:
    result = np.zeros(horizon, dtype=np.float64)
    false_count = 0
    report_count = 0
    for report in sorted(annotated, key=lambda row: (int(row["tau"]), int(row["report"]))):
        report_count += 1
        false_count += int(bool(report["localized_false"]))
        result[int(report["tau"]) - 1 :] = false_count / report_count
    return result


def prompt_stratified_curve_mcse(
    scenario_id: str,
    method: str,
    total_n: int,
    prompt_sums: Mapping[tuple, np.ndarray],
    prompt_sumsq: Mapping[tuple, np.ndarray],
    prompt_counts: Mapping[tuple, int],
    horizon: int,
) -> np.ndarray:
    """Conditional pointwise MCSE for the locked fixed-prompt allocation."""

    variance_of_mean = np.zeros(horizon, dtype=np.float64)
    matched = 0
    for key, n_prompt in prompt_counts.items():
        name, detector, _prompt_id = key
        if name != scenario_id or detector != method:
            continue
        matched += n_prompt
        if n_prompt < 2:
            return np.full(horizon, np.nan, dtype=np.float64)
        sums = prompt_sums[key]
        sumsq = prompt_sumsq[key]
        sample_variance = np.maximum(
            (sumsq - sums * sums / n_prompt) / (n_prompt - 1), 0.0
        )
        variance_of_mean += n_prompt * sample_variance / (total_n * total_n)
    if matched != total_n:
        raise RuntimeError(
            f"prompt-stratified curve counts mismatch for {(scenario_id, method)}: "
            f"{matched} versus {total_n}"
        )
    return np.sqrt(variance_of_mean)


def finite_mean(values: Iterable[object]) -> tuple[float, float, int]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan, 0
    mean = float(array.mean())
    mcse = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else math.nan
    return mean, mcse, int(array.size)


def prompt_stratified_finite_mean(
    rows: Sequence[Mapping], values: Sequence[object]
) -> tuple[float, float, int]:
    """Mean and conditional MCSE for the fixed prompt-panel allocation.

    The point estimate weights every generated path equally.  Its variance is
    estimated from within-prompt replication, treating the 20-prompt panel and
    each prompt's locked allocation as fixed.  This removes between-prompt
    variation that is not Monte Carlo randomness under the stated estimand.
    """

    if len(rows) != len(values):
        raise ValueError("rows and values must have equal length")
    grouped: dict[int, list[float]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        converted = float(value)
        if math.isfinite(converted):
            if "prompt_id" not in row:
                raise ValueError(
                    "fixed-prompt MCSE requires prompt_id on every finite row"
                )
            grouped[int(row["prompt_id"])].append(converted)
    count = sum(len(group) for group in grouped.values())
    if count == 0:
        return math.nan, math.nan, 0
    mean = sum(sum(group) for group in grouped.values()) / count
    if any(len(group) < 2 for group in grouped.values()):
        # Do not silently replace the fixed-panel variance estimand by an
        # unstratified one for sparse available-case diagnostics.
        return float(mean), math.nan, int(count)
    variance_of_mean = sum(
        len(group) * float(np.var(group, ddof=1)) for group in grouped.values()
    ) / (count * count)
    return float(mean), math.sqrt(max(variance_of_mean, 0.0)), int(count)


def token_degeneracy_diagnostics(token_ids: np.ndarray, is_eos: np.ndarray) -> dict:
    """Post-lock descriptive diagnostics for repetitive OPT continuations.

    The early checkpoint audit motivated these diagnostics.  They never alter
    path inclusion or any confirmatory estimate.  ``distinct_1`` is the share
    of distinct unigrams; ``severe_repetition`` is a scale-free union of a
    longest constant-token run covering at least half the continuation or
    distinct-1 at most 5%.
    """

    tokens = np.asarray(token_ids, dtype=np.int64)
    if tokens.ndim != 1 or tokens.size == 0:
        raise ValueError("token_ids must be a nonempty one-dimensional array")
    values, counts = np.unique(tokens, return_counts=True)
    modal_index = int(np.argmax(counts))
    changes = np.flatnonzero(tokens[1:] != tokens[:-1]) + 1
    run_boundaries = np.concatenate(([0], changes, [tokens.size]))
    longest_run = int(np.diff(run_boundaries).max())
    unique_count = int(values.size)
    distinct_1 = float(unique_count / tokens.size)
    exact_one_token = unique_count == 1
    long_half_run = longest_run >= math.ceil(tokens.size / 2)
    low_distinct_1 = distinct_1 <= 0.05
    modal_token_id = int(values[modal_index])
    return {
        "unique_token_count": unique_count,
        "distinct_1": distinct_1,
        "most_frequent_token_id": modal_token_id,
        "most_frequent_token_count": int(counts[modal_index]),
        "most_frequent_token_fraction": float(counts[modal_index] / tokens.size),
        "longest_identical_token_run": longest_run,
        "eos_count": int(np.count_nonzero(np.asarray(is_eos, dtype=bool))),
        "exact_one_token_trace_flag": bool(exact_one_token),
        "modal_token_is_1437": modal_token_id == 1437,
        "modal_token_is_1437_flag": modal_token_id == 1437,
        "exact_token_1437_trace_flag": bool(exact_one_token and modal_token_id == 1437),
        "long_half_horizon_run_flag": bool(long_half_run),
        "low_distinct_1_flag": bool(low_distinct_1),
        "degeneration_flag": bool(long_half_run or low_distinct_1),
    }


def summarize_groups(
    path_rows: Sequence[dict],
    curve_sums: Mapping[tuple, np.ndarray],
    curve_sumsq: Mapping[tuple, np.ndarray],
    curve_prompt_sums: Mapping[tuple, np.ndarray] | None = None,
    curve_prompt_sumsq: Mapping[tuple, np.ndarray] | None = None,
    curve_prompt_counts: Mapping[tuple, int] | None = None,
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in path_rows:
        key = (row["scenario_id"], row["method"])
        groups[key].append(row)

    summaries: list[dict] = []
    for key in sorted(groups):
        rows = groups[key]
        first = rows[0]
        n = len(rows)
        summary = {
            "scenario_id": first["scenario_id"],
            "schedule_id": first["schedule_id"],
            "scenario_kind": first["scenario_kind"],
            "temperature": first["temperature"],
            "region_length": first["region_length"],
            "gap_length": first["gap_length"],
            "method": first["method"],
            "method_label": first["method_label"],
            "threshold": first["threshold"],
            "alpha": first["alpha"],
            "general_explicit_bound": first["general_explicit_bound"],
            "n_paths": n,
        }
        for metric in PRIMARY_SCALAR_METRICS:
            values = [row.get(metric, math.nan) for row in rows]
            mean, mcse, count = prompt_stratified_finite_mean(rows, values)
            low, high = normal_interval(mean, mcse)
            if metric in BINARY_METRICS and count:
                low, high = max(0.0, low), min(1.0, high)
            summary[f"mean_{metric}"] = mean
            summary[f"mcse_{metric}"] = mcse
            summary[f"ci95_low_{metric}"] = low
            summary[f"ci95_high_{metric}"] = high
            summary[f"n_{metric}"] = count
            summary[f"mcse_basis_{metric}"] = "within_fixed_prompt_strata"
            if metric in BINARY_METRICS and count:
                wilson_low, wilson_high = wilson_interval(
                    int(round(mean * count)), count
                )
                summary[f"wilson_ci95_low_{metric}"] = wilson_low
                summary[f"wilson_ci95_high_{metric}"] = wilson_high
        pointwise = curve_sums[key] / n
        summary["supremum_pointwise_fdr"] = float(pointwise.max())
        summary["time_of_supremum_pointwise_fdr"] = int(np.argmax(pointwise) + 1)
        if (
            curve_prompt_sums is not None
            and curve_prompt_sumsq is not None
            and curve_prompt_counts is not None
        ):
            pointwise_mcse = prompt_stratified_curve_mcse(
                str(first["scenario_id"]),
                str(first["method"]),
                n,
                curve_prompt_sums,
                curve_prompt_sumsq,
                curve_prompt_counts,
                pointwise.size,
            )
            maximizing = int(np.argmax(pointwise))
            summary["mcse_at_supremum_pointwise_fdr_time"] = float(
                pointwise_mcse[maximizing]
            )
            summary["pointwise_mcse_basis"] = "within_fixed_prompt_strata"
        elif n > 1:
            pointwise_variance = np.maximum(
                (curve_sumsq[key] - n * pointwise**2) / (n - 1), 0.0
            )
            pointwise_mcse = np.sqrt(pointwise_variance / n)
            maximizing = int(np.argmax(pointwise))
            summary["mcse_at_supremum_pointwise_fdr_time"] = float(
                pointwise_mcse[maximizing]
            )
        else:
            summary["mcse_at_supremum_pointwise_fdr_time"] = math.nan
        summaries.append(summary)
    return summaries


def paired_bootstrap(
    path_rows: Sequence[dict], bootstrap_replicates: int
) -> list[dict]:
    """Paired whole-path comparisons in the central L=200,G=50 cells."""

    central = [
        row
        for row in path_rows
        if row["scenario_kind"] == "two_region"
        and row["region_length"] == 200
        and row["gap_length"] == 50
    ]
    by_temperature_path: dict[tuple[float, str], dict[str, dict]] = defaultdict(dict)
    for row in central:
        by_temperature_path[(float(row["temperature"]), row["path_uid"])][row["method"]] = row

    result: list[dict] = []
    temperatures = sorted({temperature for temperature, _ in by_temperature_path})
    for temperature in temperatures:
        path_maps = [
            methods
            for (temp, _), methods in sorted(by_temperature_path.items())
            if temp == temperature
        ]
        for method_a, method_b in PAIR_COMPARISONS:
            eligible = [p for p in path_maps if method_a in p and method_b in p]
            for metric in PAIR_METRICS:
                differences = np.asarray(
                    [float(p[method_a][metric]) - float(p[method_b][metric]) for p in eligible],
                    dtype=np.float64,
                )
                prompt_ids = np.asarray(
                    [int(p[method_a]["prompt_id"]) for p in eligible], dtype=np.int64
                )
                finite = np.isfinite(differences)
                differences = differences[finite]
                prompt_ids = prompt_ids[finite]
                if differences.size == 0:
                    continue
                prompt_groups = [
                    np.flatnonzero(prompt_ids == prompt_id)
                    for prompt_id in sorted(np.unique(prompt_ids))
                ]
                if any(group.size < 2 for group in prompt_groups):
                    raise RuntimeError(
                        "central paired bootstrap requires at least two paths per prompt"
                    )
                seed_payload = {
                    "seed": BOOTSTRAP_SEED,
                    "temperature": temperature,
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                }
                seed_words = np.frombuffer(
                    hashlib.sha256(canonical_json(seed_payload).encode("utf-8")).digest()[:16],
                    dtype=np.uint32,
                )
                rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(seed_words)))
                # The 20-prompt panel is fixed. Resample complete paths within
                # prompt, preserving each prompt's observed allocation.
                boot = np.empty(bootstrap_replicates, dtype=np.float64)
                cursor = 0
                while cursor < bootstrap_replicates:
                    stop = min(cursor + 500, bootstrap_replicates)
                    block_sum = np.zeros(stop - cursor, dtype=np.float64)
                    for group in prompt_groups:
                        local = rng.integers(
                            0, group.size, size=(stop - cursor, group.size)
                        )
                        block_sum += differences[group[local]].sum(axis=1)
                    boot[cursor:stop] = block_sum / differences.size
                    cursor = stop
                result.append(
                    {
                        "temperature": temperature,
                        "region_length": 200,
                        "gap_length": 50,
                        "method_a": method_a,
                        "method_b": method_b,
                        "metric": metric,
                        "n_paired_paths": int(differences.size),
                        "mean_paired_difference_a_minus_b": float(differences.mean()),
                        "bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_scheme": "resample_whole_paths_within_fixed_prompt",
                        "n_prompt_strata": len(prompt_groups),
                        "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                        "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                    }
                )
    return result


def summarize_batch_size_sensitivity(path_rows: Sequence[dict]) -> list[dict]:
    """Descriptive central-cell split by locked actual generation batch size."""

    selected = [
        row
        for row in path_rows
        if row["scenario_kind"] == "two_region"
        and row["region_length"] == 200
        and row["gap_length"] == 50
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in selected:
        groups[
            (
                float(row["temperature"]),
                row["method"],
                int(row["generation_batch_size"]),
            )
        ].append(row)
    metrics = (
        "all_regions_separately_detected",
        "all_regions_iou_ge_0p5",
        "token_iou",
        "any_boundary_gap_bridge",
        "uniform_fdp",
        "reports",
    )
    output: list[dict] = []
    for (temperature, method, batch_size), rows in sorted(groups.items()):
        result = {
            "temperature": temperature,
            "region_length": 200,
            "gap_length": 50,
            "method": method,
            "generation_batch_size": batch_size,
            "n_paths": len(rows),
            "interpretation": (
                "descriptive split of different paths by locked actual batch size; not a "
                "paired estimate of a batch-kernel effect (see paired batch smoke audit)"
            ),
        }
        for metric in metrics:
            mean, mcse, count = finite_mean(row[metric] for row in rows)
            result[f"mean_{metric}"] = mean
            result[f"mcse_{metric}"] = mcse
            result[f"n_{metric}"] = count
        output.append(result)
    return output


def summarize_degeneration_sensitivity(path_rows: Sequence[dict]) -> list[dict]:
    """Post-hoc central-cell stratification; never filters primary results."""

    selected = [
        row
        for row in path_rows
        if row["scenario_kind"] == "two_region"
        and row["region_length"] == 200
        and row["gap_length"] == 50
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in selected:
        groups[
            (
                float(row["temperature"]),
                row["method"],
                bool(row["degeneration_flag"]),
            )
        ].append(row)
    metrics = (
        "all_regions_separately_detected",
        "all_regions_single_report_covered",
        "all_regions_iou_ge_0p5",
        "all_regions_iou_ge_0p8",
        "token_iou",
        "mean_region_best_iou",
        "any_boundary_gap_bridge",
        "any_merge",
        "uniform_fdp",
        "reports",
    )
    output: list[dict] = []
    for (temperature, method, degeneration_flag), rows in sorted(groups.items()):
        result = {
            "temperature": temperature,
            "region_length": 200,
            "gap_length": 50,
            "method": method,
            "degeneration_flag": degeneration_flag,
            "degeneration_stratum": (
                "posthoc_degenerate" if degeneration_flag else "posthoc_non_degenerate"
            ),
            "n_paths": len(rows),
            "interpretation": (
                "post-hoc descriptive sensitivity only; no paths were excluded from "
                "the primary analysis; theorem FDR bounds do not condition on this "
                "future-dependent stratum"
            ),
        }
        for metric in metrics:
            mean, mcse, count = finite_mean(row[metric] for row in rows)
            result[f"mean_{metric}"] = mean
            result[f"mcse_{metric}"] = mcse
            result[f"n_{metric}"] = count
        output.append(result)
    return output


def generation_randomness_audit(diagnostic_rows: Sequence[dict]) -> dict:
    """Audit duplicate token traces against independent key/pivot streams."""

    by_tokens: dict[str, list[dict]] = defaultdict(list)
    by_pivots: dict[str, list[dict]] = defaultdict(list)
    for row in diagnostic_rows:
        by_tokens[str(row["token_array_sha256"])].append(row)
        by_pivots[str(row["pivot_array_sha256"])].append(row)
    duplicate_token_groups = []
    for token_hash, rows in sorted(by_tokens.items()):
        if len(rows) < 2:
            continue
        pivot_hashes = {str(row["pivot_array_sha256"]) for row in rows}
        key_seeds = {tuple(row["key_seed_words"]) for row in rows}
        ordinary_seeds = {tuple(row["ordinary_seed_words"]) for row in rows}
        duplicate_token_groups.append(
            {
                "token_array_sha256": token_hash,
                "group_size": len(rows),
                "path_uids": [row["path_uid"] for row in rows],
                "distinct_pivot_array_hashes": len(pivot_hashes),
                "distinct_key_seeds": len(key_seeds),
                "distinct_ordinary_seeds": len(ordinary_seeds),
                "all_pivot_traces_unique_within_group": len(pivot_hashes) == len(rows),
                "all_key_seeds_unique_within_group": len(key_seeds) == len(rows),
                "all_ordinary_seeds_unique_within_group": len(ordinary_seeds) == len(rows),
            }
        )
    duplicate_pivot_groups = [
        {
            "pivot_array_sha256": pivot_hash,
            "group_size": len(rows),
            "path_uids": [row["path_uid"] for row in rows],
        }
        for pivot_hash, rows in sorted(by_pivots.items())
        if len(rows) >= 2
    ]
    duplicate_groups_independent = all(
        group["all_pivot_traces_unique_within_group"]
        and group["all_key_seeds_unique_within_group"]
        and group["all_ordinary_seeds_unique_within_group"]
        for group in duplicate_token_groups
    )
    return {
        "status": "complete",
        "interpretation": (
            "duplicate text is compatible with model degeneration only when the keyed "
            "pivot traces and both RNG stream seeds remain distinct"
        ),
        "n_paths": len(diagnostic_rows),
        "unique_token_array_hashes": len(by_tokens),
        "unique_pivot_array_hashes": len(by_pivots),
        "duplicate_token_trace_group_count": len(duplicate_token_groups),
        "duplicate_pivot_trace_group_count": len(duplicate_pivot_groups),
        "all_duplicate_token_groups_have_distinct_pivots_and_seeds": (
            duplicate_groups_independent
        ),
        "duplicate_token_trace_groups": duplicate_token_groups,
        "duplicate_pivot_trace_groups": duplicate_pivot_groups,
    }


def main(argv: Sequence[str] | None = None) -> int:
    global METHOD_ORDER, METHOD_LABELS, PAIR_COMPARISONS
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=GENERATION_DIR / "confirmatory_fresh_paths",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "opt13b")
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--max-paths", type=int, default=None, help="engineering smoke only")
    parser.add_argument(
        "--general-fdr-target",
        type=float,
        default=0.10,
        help="target for the general-dependence bound (1+log h)/h",
    )
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    if args.max_paths is not None and args.max_paths < 1:
        parser.error("--max-paths must be positive")
    if not math.isclose(args.general_fdr_target, 0.10, abs_tol=1e-15):
        parser.error("--general-fdr-target is locked to 0.10")
    try:
        METHOD_ORDER, METHOD_LABELS, PAIR_COMPARISONS, general_calibration = (
            method_configuration(args.general_fdr_target)
        )
    except ValueError as error:
        parser.error(str(error))

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    design = json.loads((input_dir / "design.json").read_text(encoding="utf-8"))
    profile = str(design.get("profile"))
    if profile == "study":
        scenarios = generation.study_scenarios()
        expected_paths = generation.TOTAL_MODEL_PATHS
    else:
        raise RuntimeError(f"unsupported saved-path profile: {profile}")
    specs = generation.build_manifest(scenarios, batch_size=generation.DEFAULT_BATCH_SIZE)
    expected_design_payload = generation.design_payload(
        profile,
        scenarios,
        generation.DEFAULT_BATCH_SIZE,
        generation.DEFAULT_TORCH_THREADS,
    )
    expected_design_sha = generation.sha256_text(
        generation.canonical_json(expected_design_payload)
    )
    if design.get("design_sha256") != expected_design_sha:
        raise RuntimeError(
            "input design hash does not match the generator's declared saved-path design"
        )
    if int(design.get("number_of_paths", -1)) != expected_paths:
        raise RuntimeError(f"input design is not the complete {expected_paths:,}-path profile")
    if args.max_paths is not None:
        specs = specs[: args.max_paths]
    if args.max_paths is None and len(specs) != expected_paths:
        raise AssertionError(f"analysis requires all {expected_paths:,} manifest paths")

    alpha_general = alpha_for_general_fdr_target(args.general_fdr_target)
    general_threshold = 1.0 / alpha_general
    if not math.isclose(general_threshold, 48.897201698676575, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("general threshold is not the locked exact gamma")
    thresholds = {method: general_threshold for method in METHOD_ORDER}
    config_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "input_design_sha256": design["design_sha256"],
        "input_profile": profile,
        "input_paths": len(specs),
        "engineering_max_paths": args.max_paths,
        "method_order": METHOD_ORDER,
        "method_labels": METHOD_LABELS,
        "thresholds": thresholds,
        "general_alpha": alpha_general,
        "general_fdr_target": args.general_fdr_target,
        "general_calibration": general_calibration,
        "general_threshold": general_threshold,
        "adaptive_cap": ADAPTIVE_CAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": args.bootstrap_replicates,
        "source_sha256": {
            "analysis_runner": sha256_file(Path(__file__).resolve()),
            "detector": sha256_file(DETECTOR_DIR / "resetting_swz.py"),
            "generator": sha256_file(GENERATION_DIR / "generate_fresh_opt13b.py"),
            "protocol_current_with_postlock_amendment": PROTOCOL_CURRENT_SHA256,
            "protocol_locked_prefix_before_amendment": PROTOCOL_LOCKED_PREFIX_SHA256,
        },
        "protocol_postlock_amendment": {
            "date": "2026-07-31",
            "checkpoint_count_when_declared": 170,
            "blinding": (
                "added from a generation-integrity audit before detector replay or "
                "localization outcomes were inspected"
            ),
            "scope": (
                "diagnostic fields, full-sample disclosure, outcome-defined descriptive "
                "stratification, and separately seeded trimmed-prompt sensitivity"
            ),
        },
        "og_and_average_status": (
            "not run: predeclared audit gate was not passed because original SWZ code was "
            "unavailable and an independent exact equation-(13) implementation was not certified"
        ),
        "degeneration_sensitivity_status": (
            "post-hoc descriptive amendment prompted by a blinded early-generation "
            "engineering audit; primary results retain every path"
        ),
        "degeneration_flag_definition": {
            "low_distinct_1": "number of distinct token IDs / horizon <= 0.05",
            "long_half_horizon_run": (
                "longest identical-token run >= ceiling(horizon / 2)"
            ),
            "combined": "low_distinct_1 OR long_half_horizon_run",
        },
    }
    config_payload["analysis_sha256"] = hashlib.sha256(
        canonical_json(config_payload).encode("utf-8")
    ).hexdigest()
    write_json_atomic(output_dir / "analysis_manifest.json", config_payload)

    path_rows: list[dict] = []
    report_count = 0
    report_temporary = output_dir / "atomic_reports.jsonl.tmp"
    report_handle = report_temporary.open("w", encoding="utf-8")
    diagnostic_rows: list[dict] = []
    curve_sums: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(generation.HORIZON))
    curve_sumsq: dict[tuple, np.ndarray] = defaultdict(
        lambda: np.zeros(generation.HORIZON)
    )
    curve_prompt_sums: dict[tuple, np.ndarray] = defaultdict(
        lambda: np.zeros(generation.HORIZON)
    )
    curve_prompt_sumsq: dict[tuple, np.ndarray] = defaultdict(
        lambda: np.zeros(generation.HORIZON)
    )
    curve_prompt_counts: dict[tuple, int] = defaultdict(int)
    begun = time.perf_counter()
    for path_index, spec in enumerate(specs, start=1):
        checkpoint = input_dir / "checkpoints" / f"{spec.path_uid}.npz"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"missing {checkpoint}; generation is incomplete ({path_index-1}/{len(specs)} read)"
            )
        saved = generation.load_and_validate_checkpoint(checkpoint, spec)
        pivots = np.asarray(saved["pivot_y"], dtype=np.float64)
        watermark_mask = np.asarray(saved["is_watermarked"], dtype=bool)
        reports_by_method, traces = replay_methods(
            pivots, general_threshold, general_calibration
        )
        scenario_fields = _scenario_fields(spec)
        regions = tuple(tuple(map(int, region)) for region in spec.intended_regions)
        degeneracy = token_degeneracy_diagnostics(
            np.asarray(saved["token_id"]), np.asarray(saved["is_eos"])
        )
        token_array_sha256 = sha256_array(np.asarray(saved["token_id"]))
        pivot_array_sha256 = sha256_array(pivots)

        diagnostic_rows.append(
            {
                "path_uid": spec.path_uid,
                "scenario_id": spec.scenario_id,
                "schedule_id": spec.schedule_id,
                "temperature": spec.temperature,
                **scenario_fields,
                "prompt_id": spec.prompt_id,
                "generation_batch_uid": spec.generation_batch_uid,
                "generation_batch_size": spec.generation_batch_size,
                "generation_batch_position": spec.generation_batch_position,
                "mean_selected_model_probability": float(
                    np.mean(saved["selected_model_probability"])
                ),
                "mean_max_model_probability": float(np.mean(saved["max_model_probability"])),
                "mean_entropy_nats": float(np.mean(saved["model_entropy_nats"])),
                "mean_null_pivot": float(np.mean(pivots[~watermark_mask]))
                if np.any(~watermark_mask)
                else math.nan,
                "mean_watermark_pivot": float(np.mean(pivots[watermark_mask]))
                if np.any(watermark_mask)
                else math.nan,
                "generation_seconds": float(saved["metadata"]["generation_seconds"]),
                "checkpoint_content_sha256": saved["metadata"]["content_sha256"],
                "token_array_sha256": token_array_sha256,
                "pivot_array_sha256": pivot_array_sha256,
                "key_seed_words": generation.rng_seed_words(spec, generation.KEY_STREAM),
                "ordinary_seed_words": generation.rng_seed_words(
                    spec, generation.ORDINARY_STREAM
                ),
                **degeneracy,
            }
        )

        for method in METHOD_ORDER:
            method_reports = reports_by_method[method]
            annotated = annotate_reports(method_reports, regions)
            metrics = path_metrics(annotated, regions, horizon=spec.horizon)
            strict_metrics = strict_region_recovery_metrics(annotated, regions)
            bridge_count, bridged_gaps = boundary_gap_bridge_annotations(
                method_reports, regions
            )
            delays = np.asarray(metrics["region_detection_delays"], dtype=np.float64)
            finite_delays = delays[np.isfinite(delays)]
            start_errors = np.asarray(metrics["region_start_errors"], dtype=np.float64)
            start_errors = start_errors[np.isfinite(start_errors)]
            end_errors = np.asarray(metrics["region_end_errors"], dtype=np.float64)
            end_errors = end_errors[np.isfinite(end_errors)]
            trace = traces[method]
            threshold = thresholds[method]
            diagnostic_stop = (
                int(method_reports[0]["tau"])
                if method.startswith("swz_one_shot_") and method_reports
                else spec.horizon
            )
            diagnostic_slice = slice(0, diagnostic_stop)
            diagnostic_mask = watermark_mask[diagnostic_slice]
            diagnostic_eta = trace.bet_fraction[diagnostic_slice]
            diagnostic_e = trace.e_factor[diagnostic_slice]
            row = {
                "path_uid": spec.path_uid,
                "scenario_id": spec.scenario_id,
                "schedule_id": spec.schedule_id,
                "temperature": spec.temperature,
                **scenario_fields,
                "replicate": spec.replicate,
                "prompt_id": spec.prompt_id,
                "generation_batch_uid": spec.generation_batch_uid,
                "generation_batch_size": spec.generation_batch_size,
                "generation_batch_position": spec.generation_batch_position,
                **degeneracy,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "threshold": threshold,
                "alpha": 1.0 / threshold,
                "general_explicit_bound": general_dependence_bound(1.0 / threshold),
                **metrics,
                **strict_metrics,
                "any_report": int(metrics["reports"]) > 0,
                "token_precision_given_report": metrics["token_precision"],
                "token_precision_zero_if_no_report": (
                    metrics["token_precision"]
                    if int(metrics["reports"]) > 0
                    else (0.0 if regions else math.nan)
                ),
                "mean_region_detection_delay_given_timely_detection": float(
                    finite_delays.mean()
                )
                if finite_delays.size
                else math.nan,
                "mean_region_start_error_given_overlap": float(start_errors.mean())
                if start_errors.size
                else math.nan,
                "mean_region_end_error_given_overlap": float(end_errors.mean())
                if end_errors.size
                else math.nan,
                "mean_absolute_region_start_error_given_overlap": float(
                    np.abs(start_errors).mean()
                )
                if start_errors.size
                else math.nan,
                "mean_absolute_region_end_error_given_overlap": float(
                    np.abs(end_errors).mean()
                )
                if end_errors.size
                else math.nan,
                "boundary_gap_bridge_reports": bridge_count,
                "boundary_gap_bridge_gap_indices": bridged_gaps,
                "any_boundary_gap_bridge": bridge_count > 0,
                "bettor_diagnostic_tokens": diagnostic_stop,
                "bettor_diagnostic_scope": (
                    "through_first_alarm"
                    if method.startswith("swz_one_shot_") and method_reports
                    else "full_path"
                ),
                "mean_eta": float(np.mean(diagnostic_eta)),
                "max_eta": float(np.max(diagnostic_eta)),
                "mean_e_factor": float(np.mean(diagnostic_e)),
                "mean_null_e_factor": float(np.mean(diagnostic_e[~diagnostic_mask]))
                if np.any(~diagnostic_mask)
                else math.nan,
                "mean_watermark_e_factor": float(np.mean(diagnostic_e[diagnostic_mask]))
                if np.any(diagnostic_mask)
                else math.nan,
            }
            path_rows.append(row)
            group_key = (spec.scenario_id, method)
            timeline = fdp_timeline(annotated, spec.horizon)
            curve_sums[group_key] += timeline
            curve_sumsq[group_key] += timeline * timeline
            prompt_curve_key = (spec.scenario_id, method, spec.prompt_id)
            curve_prompt_sums[prompt_curve_key] += timeline
            curve_prompt_sumsq[prompt_curve_key] += timeline * timeline
            curve_prompt_counts[prompt_curve_key] += 1
            for report in annotated:
                report_handle.write(
                    canonical_json(
                        _json_safe(
                            {
                                "path_uid": spec.path_uid,
                                "scenario_id": spec.scenario_id,
                                "temperature": spec.temperature,
                                "method": method,
                                **report,
                            }
                        )
                    )
                    + "\n"
                )
                report_count += 1
        if path_index % 50 == 0 or path_index == len(specs):
            elapsed = time.perf_counter() - begun
            eta = (len(specs) - path_index) * elapsed / path_index
            print(
                f"replayed {path_index}/{len(specs)} paths; ETA {eta/60:.1f} min",
                flush=True,
            )

    summaries = summarize_groups(
        path_rows,
        curve_sums,
        curve_sumsq,
        curve_prompt_sums,
        curve_prompt_sumsq,
        curve_prompt_counts,
    )
    pointwise_rows: list[dict] = []
    group_counts = defaultdict(int)
    for row in path_rows:
        group_counts[(row["scenario_id"], row["method"])] += 1
    for (scenario_id, method), curve_sum in sorted(curve_sums.items()):
        mean_curve = curve_sum / group_counts[(scenario_id, method)]
        n_group = group_counts[(scenario_id, method)]
        mcse_curve = prompt_stratified_curve_mcse(
            scenario_id,
            method,
            n_group,
            curve_prompt_sums,
            curve_prompt_sumsq,
            curve_prompt_counts,
            mean_curve.size,
        )
        for zero_t, value in enumerate(mean_curve):
            low, high = normal_interval(float(value), float(mcse_curve[zero_t]))
            pointwise_rows.append(
                {
                    "scenario_id": scenario_id,
                    "method": method,
                    "t": zero_t + 1,
                    "n_paths": group_counts[(scenario_id, method)],
                    "pointwise_fdr": float(value),
                    "pointwise_fdr_mcse": float(mcse_curve[zero_t]),
                    "pointwise_fdr_ci95_low": max(0.0, low),
                    "pointwise_fdr_ci95_high": min(1.0, high),
                    "pointwise_mcse_basis": "within_fixed_prompt_strata",
                }
            )
    comparisons = paired_bootstrap(path_rows, args.bootstrap_replicates)
    batch_sensitivity = summarize_batch_size_sensitivity(path_rows)
    degeneration_sensitivity = summarize_degeneration_sensitivity(path_rows)

    write_csv_atomic(output_dir / "summary.csv", summaries)
    write_csv_atomic(output_dir / "pointwise_fdr.csv", pointwise_rows)
    if comparisons:
        write_csv_atomic(output_dir / "paired_comparisons.csv", comparisons)
    if batch_sensitivity:
        write_csv_atomic(output_dir / "batch_size_sensitivity.csv", batch_sensitivity)
    if degeneration_sensitivity:
        write_csv_atomic(
            output_dir / "degeneration_sensitivity.csv", degeneration_sensitivity
        )
    write_csv_atomic(output_dir / "generation_diagnostics.csv", diagnostic_rows)
    randomness_audit = generation_randomness_audit(diagnostic_rows)
    write_json_atomic(
        output_dir / "generation_randomness_audit.json", randomness_audit
    )
    # CSV path rows retain tuple metrics as canonical JSON strings.  JSONL is
    # friendlier for full report-level nesting.
    write_csv_atomic(output_dir / "path_metrics.csv", path_rows)
    report_handle.close()
    os.replace(report_temporary, output_dir / "atomic_reports.jsonl")
    write_json_atomic(output_dir / "summary.json", summaries)

    manifest = json.loads((output_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "completed": True,
            "elapsed_seconds": time.perf_counter() - begun,
            "path_metric_rows": len(path_rows),
            "atomic_report_rows": report_count,
            "summary_rows": len(summaries),
            "degenerate_paths": sum(
                int(bool(row["degeneration_flag"])) for row in diagnostic_rows
            ),
            "generation_randomness_audit_passed": bool(
                randomness_audit[
                    "all_duplicate_token_groups_have_distinct_pivots_and_seeds"
                ]
                and randomness_audit["duplicate_pivot_trace_group_count"] == 0
            ),
            "output_sha256": {
                filename: sha256_file(output_dir / filename)
                for filename in FINAL_OUTPUT_FILENAMES
                if (output_dir / filename).is_file()
            },
        }
    )
    write_json_atomic(output_dir / "analysis_manifest.json", manifest)
    print(f"analysis complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
