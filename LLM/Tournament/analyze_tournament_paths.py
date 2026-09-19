#!/usr/bin/env python3
"""Replay all locked detectors on the same saved Tournament pivots."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from refreshing_swz import (
    annotate_reports,
    path_metrics,
    run_average_from_component_evalues,
    run_online_grenander_from_pivots,
    run_refreshing_from_pivots,
)
import generate_tournament_opt13b as generation


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
THRESHOLD = 49.0
WA_CAP = 0.5
FIXED_LAMBDAS = (0.10, 0.25, 0.50, 0.75)
PRIMARY_METHODS = (
    "one_shot_global_min",
    "refresh_whole_block",
    "refresh_global_min",
)
PROCESS_METHODS = (
    "average_50_50",
    "online_grenander",
    "weight_adaptive",
    "fixed_lambda_0p10",
    "fixed_lambda_0p25",
    "fixed_lambda_0p50",
    "fixed_lambda_0p75",
)
REPORTED_METRICS = (
    "mean_reports",
    "power",
    "coverage_fdp",
    "iou",
    "localized_fdp",
    "uniform_localized_fdp",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError("CSV rows do not share one stable schema")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def whole_block_reports(reports: Iterable) -> list[dict]:
    output = []
    for report in reports:
        row = report.as_dict()
        row["sigma"] = int(row["previous_tau"])
        row["interval_start"] = int(row["previous_tau"]) + 1
        row["interval_end"] = int(row["tau"])
        row["interval_length"] = int(row["tau"]) - int(row["previous_tau"])
        output.append(row)
    return output


def method_metrics(reports: Sequence[Mapping] | Sequence, regions, horizon: int) -> dict:
    annotated = annotate_reports(reports, regions)
    raw = path_metrics(annotated, regions, horizon=horizon)
    predicted = int(raw["predicted_tokens"])
    return {
        "mean_reports": int(raw["reports"]),
        "power": float(raw["token_recall"]),
        "coverage_fdp": int(raw["false_positive_tokens"]) / max(predicted, 1),
        "iou": float(raw["token_iou"]),
        "localized_fdp": float(raw["final_fdp"]),
        "uniform_localized_fdp": float(raw["uniform_fdp"]),
    }


def process_results(pivots: np.ndarray) -> tuple[dict[str, object], dict[str, float]]:
    wa = run_refreshing_from_pivots(
        pivots, strategy="adaptive_cumulative", threshold=THRESHOLD, cap=WA_CAP
    )
    og = run_online_grenander_from_pivots(
        pivots, strategy="og_cumulative", threshold=THRESHOLD
    )
    average = run_average_from_component_evalues(
        wa.e_factor, og.e_factor, threshold=THRESHOLD
    )
    processes: dict[str, object] = {
        "average_50_50": average,
        "online_grenander": og,
        "weight_adaptive": wa,
    }
    mean_lambda = {
        "average_50_50": math.nan,
        "online_grenander": math.nan,
        "weight_adaptive": float(np.mean(wa.bet_fraction)),
    }
    for lam in FIXED_LAMBDAS:
        name = f"fixed_lambda_{lam:.2f}".replace(".", "p")
        result = run_refreshing_from_pivots(
            pivots, strategy="fixed", fixed_eta=lam, threshold=THRESHOLD
        )
        processes[name] = result
        mean_lambda[name] = lam
    if tuple(processes) != PROCESS_METHODS:
        raise AssertionError("seven-process order changed")
    return processes, mean_lambda


def save_trace(path: Path, processes: Mapping[str, object]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for method, result in processes.items():
        for name, value in result.trace_dict().items():
            arrays[f"{method}__{name}"] = np.asarray(value)
    temporary = path.with_suffix(".partial.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["experiment"], float(row["temperature"]), row["method"])].append(row)
    summaries = []
    for (experiment, temperature, method), group in sorted(groups.items()):
        record = {
            "experiment": experiment,
            "temperature": temperature,
            "method": method,
            "n_paths": len(group),
            "mean_lambda": float(np.nanmean([row["mean_lambda"] for row in group]))
            if any(math.isfinite(float(row["mean_lambda"])) for row in group)
            else math.nan,
        }
        for metric in REPORTED_METRICS:
            record[metric] = float(np.mean([float(row[metric]) for row in group]))
        summaries.append(record)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "study_paths"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "llm" / "tournament" / "study",
    )
    parser.add_argument("--max-paths", type=int, default=None, help="engineering replay only")
    args = parser.parse_args(argv)

    design = json.loads((args.input_dir / "design.json").read_text(encoding="utf-8"))
    if design.get("profile") != "study" or design.get("number_of_paths") != 1_400:
        raise RuntimeError("input is not the locked 1,400-path Tournament study")
    scenarios = generation.study_scenarios()
    expected_payload = generation.design_payload(
        "study",
        scenarios,
        generation.DEFAULT_BATCH_SIZE,
        generation.DEFAULT_TORCH_THREADS,
    )
    expected_sha = generation.sha256_text(generation.canonical_json(expected_payload))
    if design.get("design_sha256") != expected_sha:
        raise RuntimeError("saved Tournament design does not match the frozen generator")
    specs = generation.build_manifest(
        scenarios, batch_size=generation.DEFAULT_BATCH_SIZE
    )
    if len(specs) != generation.TOTAL_MODEL_PATHS:
        raise AssertionError("analysis requires the complete 1,400-path manifest")
    if args.max_paths is not None:
        if args.max_paths < 1:
            raise ValueError("--max-paths must be positive")
        specs = specs[: args.max_paths]

    rows: list[dict] = []
    reports_path = args.output_dir / "atomic_reports.jsonl"
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    reports_tmp = reports_path.with_suffix(".jsonl.tmp")
    with reports_tmp.open("w", encoding="utf-8") as report_handle:
        for index, spec in enumerate(specs, start=1):
            checkpoint = args.input_dir / "checkpoints" / f"{spec.path_uid}.npz"
            saved = generation.load_and_validate_checkpoint(checkpoint, spec)
            pivots = np.asarray(saved["pivot_y"], dtype=np.float64)
            regions = tuple(tuple(map(int, region)) for region in spec.intended_regions)
            is_primary = spec.schedule_id == "two_l200_g050"

            if is_primary:
                processes, lambdas = process_results(pivots)
                wa = processes["weight_adaptive"]
                primary_reports = {
                    "one_shot_global_min": [wa.reports[0].as_dict()] if wa.reports else [],
                    "refresh_whole_block": whole_block_reports(wa.reports),
                    "refresh_global_min": wa.report_dicts(),
                }
                if tuple(primary_reports) != PRIMARY_METHODS:
                    raise AssertionError("primary reporting-method order changed")
                for method, reports in primary_reports.items():
                    metrics = method_metrics(reports, regions, spec.horizon)
                    rows.append(
                        {
                            "path_uid": spec.path_uid,
                            "prompt_id": spec.prompt_id,
                            "experiment": "primary_reporting",
                            "temperature": spec.temperature,
                            "method": method,
                            "mean_lambda": float(np.mean(wa.bet_fraction)),
                            **metrics,
                        }
                    )
                    for report in annotate_reports(reports, regions):
                        report_handle.write(
                            canonical_json(
                                {
                                    "path_uid": spec.path_uid,
                                    "experiment": "primary_reporting",
                                    "method": method,
                                    **report,
                                }
                            )
                            + "\n"
                        )
                for method, result in processes.items():
                    metrics = method_metrics(result.report_dicts(), regions, spec.horizon)
                    rows.append(
                        {
                            "path_uid": spec.path_uid,
                            "prompt_id": spec.prompt_id,
                            "experiment": "seven_processes",
                            "temperature": spec.temperature,
                            "method": method,
                            "mean_lambda": lambdas[method],
                            **metrics,
                        }
                    )
                save_trace(args.output_dir / "traces" / f"{spec.path_uid}.npz", processes)
            elif spec.schedule_id == "four_l050_g025":
                wa = run_refreshing_from_pivots(
                    pivots,
                    strategy="adaptive_cumulative",
                    threshold=THRESHOLD,
                    cap=WA_CAP,
                )
                metrics = method_metrics(wa.report_dicts(), regions, spec.horizon)
                rows.append(
                    {
                        "path_uid": spec.path_uid,
                        "prompt_id": spec.prompt_id,
                        "experiment": "four_interval_stress",
                        "temperature": spec.temperature,
                        "method": "weight_adaptive",
                        "mean_lambda": float(np.mean(wa.bet_fraction)),
                        **metrics,
                    }
                )
                save_trace(
                    args.output_dir / "traces" / f"{spec.path_uid}.npz",
                    {"weight_adaptive": wa},
                )
            else:
                raise AssertionError(f"unexpected schedule {spec.schedule_id}")
            if index % 50 == 0 or index == len(specs):
                print(f"replayed {index}/{len(specs)} paths", flush=True)
        report_handle.flush()
        os.fsync(report_handle.fileno())
    os.replace(reports_tmp, reports_path)

    summaries = summarize(rows)
    write_csv_atomic(args.output_dir / "path_metrics.csv", rows)
    write_csv_atomic(args.output_dir / "summary.csv", summaries)
    write_json_atomic(args.output_dir / "summary.json", summaries)
    write_json_atomic(
        args.output_dir / "analysis_manifest.json",
        {
            "threshold_exact": THRESHOLD,
            "weighted_adaptive_cap": WA_CAP,
            "fixed_lambdas": list(FIXED_LAMBDAS),
            "same_pivots_for_every_detector": True,
            "n_input_paths": len(specs),
            "reported_metrics": list(REPORTED_METRICS),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
