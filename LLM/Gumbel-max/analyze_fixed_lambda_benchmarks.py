#!/usr/bin/env python3
"""Replay prespecified fixed-lambda SWZ processes on frozen experiment paths.

This analysis never loads OPT-1.3B and never regenerates text. It reuses the
1,000 primary paths from the saved 1,400-path two-temperature design.

For a prespecified lambda, the nonadaptive multiplier is

    E_t(lambda) = (1-lambda) + lambda * {-log(1-Y_t)}.

The declared grid is lambda in {0.10, 0.25, 0.50, 0.75}; every detector uses
refreshing, the global-minimum localizer, and the common comparison threshold
gamma=48.8972017 (displayed as 49).  This is the same threshold used by WA,
Online Grenander, and their average.  Under the independent-multiplier theorem
the fixed-lambda Final report FDR bound is therefore 1/(gamma-1), about 2.09%.
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
from typing import Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
GENERATION_DIR = HERE
DETECTOR_DIR = HERE
sys.path.insert(0, str(GENERATION_DIR))
sys.path.insert(0, str(DETECTOR_DIR))

import analyze_opt13b_paths as opt_base  # noqa: E402
import generate_fresh_opt13b as generation  # noqa: E402
from refreshing_swz import (  # noqa: E402
    annotate_reports,
    fixed_e_factors,
    path_metrics,
    run_refreshing_from_evalues,
)
ANALYSIS_VERSION = "fixed-lambda-opt-v4-common-gamma49"
LAMBDAS = (0.10, 0.25, 0.50, 0.75)
GENERAL_THRESHOLD = 48.897201698676575
THRESHOLD = GENERAL_THRESHOLD


def lambda_tag(lam: float) -> str:
    return f"{lam:.2f}".replace(".", "p")


def method_name(lam: float) -> str:
    return f"fixed_lambda_{lambda_tag(lam)}_refresh_h49"


def method_label(lam: float) -> str:
    return rf"Fixed $\lambda={lam:g}$"


def _interval(mean: float, mcse: float, binary: bool = False) -> tuple[float, float]:
    if not math.isfinite(mean) or not math.isfinite(mcse):
        return math.nan, math.nan
    low, high = mean - 1.96 * mcse, mean + 1.96 * mcse
    if binary:
        low, high = max(0.0, low), min(1.0, high)
    return low, high


def _summarize_opt(path_rows: Sequence[dict]) -> list[dict]:
    metrics = (
        "reports",
        "any_report",
        "uniform_fdp",
        "final_fdp",
        "token_fdp",
        "token_recall",
        "token_iou",
        "any_merge",
        "predicted_tokens",
        "false_positive_tokens",
        "mean_e_factor",
        "mean_null_e_factor",
        "mean_watermark_e_factor",
    )
    binary = {"any_report", "any_merge"}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in path_rows:
        groups[(str(row["scenario_id"]), str(row["method"]))].append(row)
    output: list[dict] = []
    for key in sorted(groups):
        rows = groups[key]
        first = rows[0]
        record = {
            name: first[name]
            for name in (
                "scenario_id",
                "schedule_id",
                "scenario_kind",
                "temperature",
                "region_length",
                "gap_length",
                "method",
                "method_label",
                "fixed_lambda",
                "threshold",
            )
        }
        record["n_paths"] = len(rows)
        for metric in metrics:
            values = [float(row.get(metric, math.nan)) for row in rows]
            mean, mcse, count = opt_base.prompt_stratified_finite_mean(rows, values)
            low, high = _interval(mean, mcse, metric in binary)
            record[f"mean_{metric}"] = mean
            record[f"mcse_{metric}"] = mcse
            record[f"ci95_low_{metric}"] = low
            record[f"ci95_high_{metric}"] = high
            record[f"n_{metric}"] = count
        output.append(record)
    return output


def run_opt(
    input_dir: Path,
    output_dir: Path,
    smoke_per_scenario: int | None,
) -> dict:
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
        raise RuntimeError("saved OPT design hash does not match the frozen generator")
    if len(specs) != expected_paths:
        raise RuntimeError(
            f"expected {expected_paths:,} saved OPT paths, found {len(specs)}"
        )
    specs = [spec for spec in specs if spec.schedule_id == "two_l200_g050"]
    if len(specs) != 1_000:
        raise RuntimeError("fixed-lambda comparison requires 1,000 primary paths")
    if smoke_per_scenario is not None:
        by_scenario: dict[str, list] = defaultdict(list)
        for spec in specs:
            by_scenario[spec.scenario_id].append(spec)
        specs = [
            spec
            for scenario_id in sorted(by_scenario)
            for spec in by_scenario[scenario_id][:smoke_per_scenario]
        ]

    rows: list[dict] = []
    begun = time.perf_counter()
    for path_index, spec in enumerate(specs, start=1):
        checkpoint = input_dir / "checkpoints" / f"{spec.path_uid}.npz"
        saved = generation.load_and_validate_checkpoint(checkpoint, spec)
        pivots = np.asarray(saved["pivot_y"], dtype=np.float64)
        watermark_mask = np.asarray(saved["is_watermarked"], dtype=bool)
        regions = tuple(tuple(map(int, region)) for region in spec.intended_regions)
        scenario_fields = opt_base._scenario_fields(spec)
        for lam in LAMBDAS:
            factor = fixed_e_factors(pivots, lam)
            result = run_refreshing_from_evalues(factor, threshold=THRESHOLD)
            annotated = annotate_reports(result.report_dicts(), regions)
            metrics = path_metrics(annotated, regions, horizon=spec.horizon)
            predicted = int(metrics["predicted_tokens"])
            false_positive = int(metrics["false_positive_tokens"])
            token_fdp = false_positive / max(predicted, 1)
            rows.append(
                {
                    "path_uid": spec.path_uid,
                    "scenario_id": spec.scenario_id,
                    "schedule_id": spec.schedule_id,
                    "temperature": float(spec.temperature),
                    **scenario_fields,
                    "replicate": spec.replicate,
                    "prompt_id": spec.prompt_id,
                    "method": method_name(lam),
                    "method_label": method_label(lam),
                    "fixed_lambda": lam,
                    "threshold": THRESHOLD,
                    "alpha": 1.0 / THRESHOLD,
                    "independent_fdr_bound": 1.0 / (THRESHOLD - 1.0),
                    "reports": int(metrics["reports"]),
                    "any_report": int(metrics["reports"]) > 0,
                    "uniform_fdp": float(metrics["uniform_fdp"]),
                    "final_fdp": float(metrics["final_fdp"]),
                    "token_fdp": token_fdp,
                    "token_recall": float(metrics["token_recall"]),
                    "token_iou": float(metrics["token_iou"]),
                    "any_merge": bool(metrics["any_merge"]),
                    "predicted_tokens": predicted,
                    "false_positive_tokens": false_positive,
                    "mean_e_factor": float(np.mean(factor)),
                    "mean_null_e_factor": (
                        float(np.mean(factor[~watermark_mask]))
                        if np.any(~watermark_mask)
                        else math.nan
                    ),
                    "mean_watermark_e_factor": (
                        float(np.mean(factor[watermark_mask]))
                        if np.any(watermark_mask)
                        else math.nan
                    ),
                }
            )
        if path_index % 250 == 0 or path_index == len(specs):
            elapsed = time.perf_counter() - begun
            eta = (len(specs) - path_index) * elapsed / path_index
            print(
                f"OPT fixed-lambda replay {path_index}/{len(specs)}; ETA {eta/60:.1f} min",
                flush=True,
            )

    summary = _summarize_opt(rows)
    opt_base.write_csv_atomic(output_dir / "opt_path_metrics.csv", rows)
    opt_base.write_csv_atomic(output_dir / "opt_summary.csv", summary)
    return {
        "n_input_paths": len(specs),
        "n_path_method_rows": len(rows),
        "n_summary_rows": len(summary),
        "wall_seconds": time.perf_counter() - begun,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opt-input-dir",
        type=Path,
        default=GENERATION_DIR / "confirmatory_fresh_paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "fixed_lambda_common_gamma49",
    )
    parser.add_argument("--opt-smoke-per-scenario", type=int, default=None)
    args = parser.parse_args(argv)
    if args.opt_smoke_per_scenario is not None and args.opt_smoke_per_scenario < 1:
        parser.error("--opt-smoke-per-scenario must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status = (
        "engineering_smoke"
        if args.opt_smoke_per_scenario is not None
        else "full"
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "status": status,
        "lambdas": list(LAMBDAS),
        "threshold": THRESHOLD,
        "independent_fdr_bound": 1.0 / (THRESHOLD - 1.0),
        "general_threshold": GENERAL_THRESHOLD,
        "general_fdr_bound": (
            1.0 + math.log(GENERAL_THRESHOLD)
        ) / GENERAL_THRESHOLD,
        "formula": "E_t(lambda)=(1-lambda)+lambda*(-log(1-Y_t))",
        "lambda_grid_prespecified": True,
        "llm_generation_performed": False,
        "opt_pivots_reused_from_saved_checkpoints": True,
        "ind_interpretation": (
            "fixed lambda removes factor adaptation; under the declared fresh-key "
            "null design the null multipliers are mutually independent; at the common "
            "comparison threshold gamma=48.8972017 their Final report FDR bound is "
            "1/(gamma-1), about 2.09%"
        ),
        "source_sha256": {
            "analysis": opt_base.sha256_file(Path(__file__).resolve()),
            "detector": opt_base.sha256_file(DETECTOR_DIR / "refreshing_swz.py"),
            "opt_generator": opt_base.sha256_file(
                GENERATION_DIR / "generate_fresh_opt13b.py"
            ),
        },
    }
    manifest["analysis_sha256"] = hashlib.sha256(
        opt_base.canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    opt_base.write_json_atomic(output_dir / "analysis_manifest.json", manifest)

    begun = time.perf_counter()
    manifest["opt"] = run_opt(
        args.opt_input_dir.resolve(), output_dir, args.opt_smoke_per_scenario
    )
    manifest["completed"] = True
    manifest["wall_seconds"] = time.perf_counter() - begun
    outputs = sorted(
        path.name for path in output_dir.glob("*.csv") if path.is_file()
    )
    manifest["output_sha256"] = {
        name: opt_base.sha256_file(output_dir / name) for name in outputs
    }
    opt_base.write_json_atomic(output_dir / "analysis_manifest.json", manifest)
    print(f"complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
