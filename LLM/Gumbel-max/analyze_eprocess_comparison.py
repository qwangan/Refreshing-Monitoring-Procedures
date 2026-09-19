#!/usr/bin/env python3
"""Full saved-pivot comparison of SWZ WA, OG, and 50/50 average processes.

The language model is never loaded and text is never regenerated.  Each saved
OPT-1.3B checkpoint is content-validated, its pivot trace is replayed once to
obtain cumulative weighted adaptive and endpoint-adjusted Online Grenander
factors, and every adaptive comparison uses the same general-calibration
threshold.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
GENERATION_DIR = HERE
DETECTOR_DIR = HERE
sys.path.insert(0, str(GENERATION_DIR))
sys.path.insert(0, str(DETECTOR_DIR))

import analyze_opt13b_paths as base  # noqa: E402
import generate_fresh_opt13b as generation  # noqa: E402
from refreshing_swz import (  # noqa: E402
    alpha_for_general_fdr_target,
    annotate_reports,
    general_dependence_bound,
    path_metrics,
    run_average_from_component_evalues,
    run_online_grenander_from_pivots,
    run_refreshing_from_evalues,
    run_refreshing_from_pivots,
)


ANALYSIS_VERSION = "opt13b-wa-og-average-v4-cap05"
NO_CROSSING_THRESHOLD = 1e300
ADAPTIVE_CAP = 0.5

def process_configuration(target: float):
    calibration = base.general_calibration_tag(target)
    order = (
        f"wa_one_shot_{calibration}",
        f"og_one_shot_{calibration}",
        f"average_one_shot_{calibration}",
        f"wa_refresh_{calibration}",
        f"og_refresh_{calibration}",
        f"average_refresh_{calibration}",
    )
    meta = {
        f"wa_one_shot_{calibration}": ("weight_adaptive", "one_shot", calibration),
        f"og_one_shot_{calibration}": ("online_grenander", "one_shot", calibration),
        f"average_one_shot_{calibration}": ("average_50_50", "one_shot", calibration),
        f"wa_refresh_{calibration}": ("weight_adaptive", "refresh", calibration),
        f"og_refresh_{calibration}": ("online_grenander", "refresh", calibration),
        f"average_refresh_{calibration}": ("average_50_50", "refresh", calibration),
    }
    labels = {
        method: f"{process.replace('_', ' ')}; {mode.replace('_', ' ')}; {threshold}"
        for method, (process, mode, threshold) in meta.items()
    }
    pairs = (
        (f"og_one_shot_{calibration}", f"wa_one_shot_{calibration}"),
        (f"average_one_shot_{calibration}", f"wa_one_shot_{calibration}"),
        (f"average_one_shot_{calibration}", f"og_one_shot_{calibration}"),
        (f"og_refresh_{calibration}", f"wa_refresh_{calibration}"),
        (f"average_refresh_{calibration}", f"wa_refresh_{calibration}"),
        (f"average_refresh_{calibration}", f"og_refresh_{calibration}"),
        (f"wa_refresh_{calibration}", f"wa_one_shot_{calibration}"),
        (f"og_refresh_{calibration}", f"og_one_shot_{calibration}"),
        (f"average_refresh_{calibration}", f"average_one_shot_{calibration}"),
    )
    return order, meta, labels, pairs, calibration


METHOD_ORDER, METHOD_META, METHOD_LABELS, PAIRS, DEFAULT_GENERAL_CALIBRATION = (
    process_configuration(0.05)
)

SUMMARY_METRICS = (
    "reports",
    "any_report",
    "false_reports",
    "uniform_fdp",
    "final_fdp",
    "any_false_report",
    "all_regions_detected",
    "all_regions_detected_timely",
    "all_regions_separately_detected",
    "all_regions_single_report_covered",
    "all_regions_iou_ge_0p5",
    "all_regions_iou_ge_0p8",
    "regionwise_separate_recall",
    "regionwise_iou_ge_0p5",
    "token_precision_zero_if_no_report",
    "token_recall",
    "token_f1",
    "token_iou",
    "mean_region_best_iou",
    "any_merge",
    "any_boundary_gap_bridge",
    "fragmented_regions",
    "mean_e_factor",
    "mean_null_e_factor",
    "mean_watermark_e_factor",
)

BINARY_METRICS = {
    "any_report",
    "any_false_report",
    "all_regions_detected",
    "all_regions_detected_timely",
    "all_regions_separately_detected",
    "all_regions_single_report_covered",
    "all_regions_iou_ge_0p5",
    "all_regions_iou_ge_0p8",
    "any_merge",
    "any_boundary_gap_bridge",
}

PAIR_METRICS = (
    "any_report",
    "all_regions_separately_detected",
    "all_regions_iou_ge_0p5",
    "reports",
    "token_iou",
    "mean_region_best_iou",
    "any_boundary_gap_bridge",
    "any_merge",
    "uniform_fdp",
)


def _first_report(result) -> list[dict]:
    return [result.reports[0].as_dict()] if result.reports else []


def replay_methods(
    pivots: np.ndarray,
    general_threshold: float,
    general_calibration: str = DEFAULT_GENERAL_CALIBRATION,
):
    started = time.perf_counter()
    wa_base = run_refreshing_from_pivots(
        pivots,
        strategy="adaptive_cumulative",
        threshold=NO_CROSSING_THRESHOLD,
        cap=ADAPTIVE_CAP,
    )
    wa_seconds = time.perf_counter() - started

    started = time.perf_counter()
    og_base = run_online_grenander_from_pivots(
        pivots,
        strategy="og_cumulative",
        threshold=NO_CROSSING_THRESHOLD,
    )
    og_seconds = time.perf_counter() - started

    wa_general = run_refreshing_from_evalues(
        wa_base.e_factor, threshold=general_threshold
    )
    og_general = run_refreshing_from_evalues(
        og_base.e_factor, threshold=general_threshold
    )
    avg_general = run_average_from_component_evalues(
        wa_base.e_factor, og_base.e_factor, threshold=general_threshold
    )

    wa_one_shot_name = f"wa_one_shot_{general_calibration}"
    og_one_shot_name = f"og_one_shot_{general_calibration}"
    average_one_shot_name = f"average_one_shot_{general_calibration}"
    wa_general_name = f"wa_refresh_{general_calibration}"
    og_general_name = f"og_refresh_{general_calibration}"
    average_general_name = f"average_refresh_{general_calibration}"
    reports = {
        wa_one_shot_name: _first_report(wa_general),
        og_one_shot_name: _first_report(og_general),
        average_one_shot_name: _first_report(avg_general),
        wa_general_name: wa_general.report_dicts(),
        og_general_name: og_general.report_dicts(),
        average_general_name: avg_general.report_dicts(),
    }
    traces = {
        wa_one_shot_name: wa_general,
        og_one_shot_name: og_general,
        average_one_shot_name: avg_general,
        wa_general_name: wa_general,
        og_general_name: og_general,
        average_general_name: avg_general,
    }
    component_factors = {
        "weight_adaptive": wa_base.e_factor,
        "online_grenander": og_base.e_factor,
        f"average_50_50_{general_calibration}": avg_general.e_factor,
    }
    runtime = {"wa_fit_seconds": wa_seconds, "og_fit_seconds": og_seconds}
    return reports, traces, component_factors, runtime


def summarize(path_rows: Sequence[dict], curve_sums: Mapping, curve_counts: Mapping):
    groups = defaultdict(list)
    for row in path_rows:
        groups[(row["scenario_id"], row["method"])].append(row)
    output = []
    for key in sorted(groups):
        rows = groups[key]
        first = rows[0]
        record = {
            "scenario_id": first["scenario_id"],
            "schedule_id": first["schedule_id"],
            "scenario_kind": first["scenario_kind"],
            "temperature": first["temperature"],
            "region_length": first["region_length"],
            "gap_length": first["gap_length"],
            "method": first["method"],
            "method_label": first["method_label"],
            "evidence_process": first["evidence_process"],
            "monitoring_mode": first["monitoring_mode"],
            "threshold_calibration": first["threshold_calibration"],
            "threshold": first["threshold"],
            "n_paths": len(rows),
        }
        for metric in SUMMARY_METRICS:
            values = [row.get(metric, math.nan) for row in rows]
            mean, mcse, count = base.prompt_stratified_finite_mean(rows, values)
            low, high = base.normal_interval(mean, mcse)
            if metric in BINARY_METRICS and count:
                low, high = max(0.0, low), min(1.0, high)
            record[f"mean_{metric}"] = mean
            record[f"mcse_{metric}"] = mcse
            record[f"ci95_low_{metric}"] = low
            record[f"ci95_high_{metric}"] = high
            record[f"n_{metric}"] = count
        pointwise = curve_sums[key] / curve_counts[key]
        record["supremum_pointwise_fdr"] = float(pointwise.max())
        record["time_of_supremum_pointwise_fdr"] = int(np.argmax(pointwise) + 1)
        output.append(record)
    return output


def paired_comparisons(path_rows: Sequence[dict]):
    central = [
        row
        for row in path_rows
        if row["scenario_kind"] == "two_region"
        and row["region_length"] == 200
        and row["gap_length"] == 50
    ]
    by_temp_path = defaultdict(dict)
    for row in central:
        by_temp_path[(float(row["temperature"]), row["path_uid"])][row["method"]] = row
    output = []
    temperatures = sorted({temperature for temperature, _ in by_temp_path})
    for temperature in temperatures:
        maps = [
            methods
            for (temp, _), methods in sorted(by_temp_path.items())
            if temp == temperature
        ]
        for method_a, method_b in PAIRS:
            eligible = [m for m in maps if method_a in m and method_b in m]
            for metric in PAIR_METRICS:
                differences = [
                    float(methods[method_a][metric])
                    - float(methods[method_b][metric])
                    for methods in eligible
                ]
                reference_rows = [methods[method_a] for methods in eligible]
                mean, mcse, count = base.prompt_stratified_finite_mean(
                    reference_rows, differences
                )
                low, high = base.normal_interval(mean, mcse)
                output.append(
                    {
                        "temperature": temperature,
                        "method_a": method_a,
                        "method_b": method_b,
                        "metric": metric,
                        "n_paired_paths": count,
                        "mean_paired_difference_a_minus_b": mean,
                        "mcse_paired_difference": mcse,
                        "ci95_low": low,
                        "ci95_high": high,
                        "mcse_basis": "within_fixed_prompt_strata",
                    }
                )
    return output


def main(argv: list[str] | None = None) -> int:
    global METHOD_ORDER, METHOD_META, METHOD_LABELS, PAIRS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=GENERATION_DIR / "confirmatory_fresh_paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "opt13b_eprocess_comparison",
    )
    parser.add_argument(
        "--smoke-paths-per-temperature",
        type=int,
        default=None,
        help="engineering validation on central paths only",
    )
    parser.add_argument(
        "--smoke-paths-per-scenario",
        type=int,
        default=None,
        help="engineering validation spanning every scenario",
    )
    parser.add_argument(
        "--general-fdr-target",
        type=float,
        default=0.10,
        help="target for the general-dependence bound (1+log h)/h",
    )
    args = parser.parse_args(argv)
    if args.smoke_paths_per_temperature is not None and args.smoke_paths_per_temperature < 1:
        parser.error("--smoke-paths-per-temperature must be positive")
    if args.smoke_paths_per_scenario is not None and args.smoke_paths_per_scenario < 1:
        parser.error("--smoke-paths-per-scenario must be positive")
    if (
        args.smoke_paths_per_temperature is not None
        and args.smoke_paths_per_scenario is not None
    ):
        parser.error("choose only one smoke-selection option")
    if not math.isclose(args.general_fdr_target, 0.10, abs_tol=1e-15):
        parser.error("--general-fdr-target is locked to 0.10")
    try:
        METHOD_ORDER, METHOD_META, METHOD_LABELS, PAIRS, general_calibration = (
            process_configuration(args.general_fdr_target)
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
    expected_payload = generation.design_payload(
        profile,
        scenarios,
        generation.DEFAULT_BATCH_SIZE,
        generation.DEFAULT_TORCH_THREADS,
    )
    expected_sha = generation.sha256_text(generation.canonical_json(expected_payload))
    if design.get("design_sha256") != expected_sha:
        raise RuntimeError("saved input design hash does not match the frozen generator")
    if len(specs) != expected_paths:
        raise RuntimeError(f"expected the complete {expected_paths:,}-path manifest")

    specs = [spec for spec in specs if spec.schedule_id == "two_l200_g050"]
    if len(specs) != 1_000:
        raise RuntimeError("process comparison requires 1,000 primary paths")

    if args.smoke_paths_per_temperature is not None:
        selected = []
        for temperature in sorted({float(spec.temperature) for spec in specs}):
            candidates = [
                spec
                for spec in specs
                if spec.schedule_id == "two_l200_g050"
                and float(spec.temperature) == temperature
            ]
            selected.extend(candidates[: args.smoke_paths_per_temperature])
        specs = selected
    elif args.smoke_paths_per_scenario is not None:
        by_scenario = defaultdict(list)
        for spec in specs:
            by_scenario[spec.scenario_id].append(spec)
        specs = [
            spec
            for scenario_id in sorted(by_scenario)
            for spec in by_scenario[scenario_id][: args.smoke_paths_per_scenario]
        ]

    alpha_general = alpha_for_general_fdr_target(args.general_fdr_target)
    general_threshold = 1.0 / alpha_general
    if not math.isclose(general_threshold, 48.897201698676575, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("general threshold is not the locked exact gamma")
    thresholds = {method: general_threshold for method in METHOD_META}
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "engineering_smoke"
            if args.smoke_paths_per_temperature or args.smoke_paths_per_scenario
            else "full"
        ),
        "input_design_sha256": design["design_sha256"],
        "input_profile": profile,
        "n_input_paths": len(specs),
        "llm_generation_performed": False,
        "methods": METHOD_META,
        "thresholds": thresholds,
        "general_alpha": alpha_general,
        "general_fdr_target": args.general_fdr_target,
        "general_calibration": general_calibration,
        "general_threshold": general_threshold,
        "adaptive_cap": ADAPTIVE_CAP,
        "source_sha256": {
            "analysis": base.sha256_file(Path(__file__).resolve()),
            "detector": base.sha256_file(DETECTOR_DIR / "refreshing_swz.py"),
            "generator": base.sha256_file(GENERATION_DIR / "generate_fresh_opt13b.py"),
        },
        "notes": [
            "SWZ equation-(13) implementation certified against published mathematics",
            "original SWZ source code unavailable; no bit-for-bit private-code claim",
            "one-shot uses the first crossing/report from the same cumulative process",
            "average is the arithmetic average of process capitals, not tokenwise factors",
        ],
    }
    manifest["analysis_sha256"] = hashlib.sha256(
        base.canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    base.write_json_atomic(output_dir / "analysis_manifest.json", manifest)

    path_rows = []
    runtime_rows = []
    curve_sums = defaultdict(lambda: np.zeros(generation.HORIZON))
    curve_counts = defaultdict(int)
    begun = time.perf_counter()
    for path_index, spec in enumerate(specs, start=1):
        checkpoint = input_dir / "checkpoints" / f"{spec.path_uid}.npz"
        saved = generation.load_and_validate_checkpoint(checkpoint, spec)
        pivots = np.asarray(saved["pivot_y"], dtype=np.float64)
        watermark_mask = np.asarray(saved["is_watermarked"], dtype=bool)
        reports_by_method, traces, factors, runtime = replay_methods(
            pivots, general_threshold, general_calibration
        )
        scenario_fields = base._scenario_fields(spec)
        regions = tuple(tuple(map(int, region)) for region in spec.intended_regions)
        runtime_rows.append(
            {
                "path_uid": spec.path_uid,
                "scenario_id": spec.scenario_id,
                "temperature": float(spec.temperature),
                **runtime,
            }
        )
        for method in METHOD_ORDER:
            process, monitoring_mode, calibration = METHOD_META[method]
            method_reports = reports_by_method[method]
            annotated = annotate_reports(method_reports, regions)
            metrics = path_metrics(annotated, regions, horizon=spec.horizon)
            compact_metrics = {
                key: value
                for key, value in metrics.items()
                if key not in {"sequential_fdp", "report_times"}
            }
            strict = base.strict_region_recovery_metrics(annotated, regions)
            bridge_count, bridged_gaps = base.boundary_gap_bridge_annotations(
                method_reports, regions
            )
            if process == "weight_adaptive":
                factor = factors["weight_adaptive"]
            elif process == "online_grenander":
                factor = factors["online_grenander"]
            else:
                factor = factors[f"average_50_50_{calibration}"]
            diagnostic_stop = (
                int(method_reports[0]["tau"])
                if monitoring_mode == "one_shot" and method_reports
                else spec.horizon
            )
            diagnostic_factor = factor[:diagnostic_stop]
            diagnostic_mask = watermark_mask[:diagnostic_stop]
            threshold = thresholds[method]
            row = {
                "path_uid": spec.path_uid,
                "scenario_id": spec.scenario_id,
                "schedule_id": spec.schedule_id,
                "temperature": float(spec.temperature),
                **scenario_fields,
                "replicate": spec.replicate,
                "prompt_id": spec.prompt_id,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "evidence_process": process,
                "monitoring_mode": monitoring_mode,
                "threshold_calibration": calibration,
                "threshold": threshold,
                "alpha": 1 / threshold,
                "general_explicit_bound": general_dependence_bound(1 / threshold),
                **compact_metrics,
                **strict,
                "any_report": int(metrics["reports"]) > 0,
                "token_precision_zero_if_no_report": (
                    metrics["token_precision"]
                    if int(metrics["reports"]) > 0
                    else (0.0 if regions else math.nan)
                ),
                "any_boundary_gap_bridge": bridge_count > 0,
                "boundary_gap_bridge_reports": bridge_count,
                "boundary_gap_bridge_gap_indices": bridged_gaps,
                "first_report_time": (
                    int(method_reports[0]["tau"]) if method_reports else math.nan
                ),
                "mean_e_factor": float(np.mean(diagnostic_factor)),
                "mean_null_e_factor": (
                    float(np.mean(diagnostic_factor[~diagnostic_mask]))
                    if np.any(~diagnostic_mask)
                    else math.nan
                ),
                "mean_watermark_e_factor": (
                    float(np.mean(diagnostic_factor[diagnostic_mask]))
                    if np.any(diagnostic_mask)
                    else math.nan
                ),
            }
            path_rows.append(row)
            key = (spec.scenario_id, method)
            curve_sums[key] += base.fdp_timeline(annotated, spec.horizon)
            curve_counts[key] += 1

        if path_index % 50 == 0 or path_index == len(specs):
            elapsed = time.perf_counter() - begun
            eta = (len(specs) - path_index) * elapsed / path_index
            print(
                f"replayed {path_index}/{len(specs)} paths; ETA {eta/60:.1f} min",
                flush=True,
            )

    summary_rows = summarize(path_rows, curve_sums, curve_counts)
    paired_rows = paired_comparisons(path_rows)
    pointwise_rows = []
    for (scenario_id, method), curve_sum in sorted(curve_sums.items()):
        curve = curve_sum / curve_counts[(scenario_id, method)]
        pointwise_rows.extend(
            {
                "scenario_id": scenario_id,
                "method": method,
                "t": t,
                "pointwise_fdr": float(value),
                "n_paths": curve_counts[(scenario_id, method)],
            }
            for t, value in enumerate(curve, start=1)
        )

    base.write_csv_atomic(output_dir / "path_metrics.csv", path_rows)
    base.write_csv_atomic(output_dir / "summary.csv", summary_rows)
    base.write_csv_atomic(output_dir / "paired_comparisons.csv", paired_rows)
    base.write_csv_atomic(output_dir / "pointwise_fdr.csv", pointwise_rows)
    base.write_csv_atomic(output_dir / "runtime.csv", runtime_rows)
    completion = {
        **manifest,
        "completed": True,
        "n_path_method_rows": len(path_rows),
        "n_summary_rows": len(summary_rows),
        "n_paired_rows": len(paired_rows),
        "wall_seconds": time.perf_counter() - begun,
        "output_sha256": {
            name: base.sha256_file(output_dir / name)
            for name in (
                "path_metrics.csv",
                "summary.csv",
                "paired_comparisons.csv",
                "pointwise_fdr.csv",
                "runtime.csv",
            )
        },
    }
    base.write_json_atomic(output_dir / "analysis_manifest.json", completion)
    print(f"complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
