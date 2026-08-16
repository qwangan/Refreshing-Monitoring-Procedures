#!/usr/bin/env python3
"""Create the Section 5.2 tables and figures from saved OPT-1.3B paths."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import analyze_opt13b_paths as primary
from refreshing_swz import Region, path_metrics, run_refreshing_from_pivots


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
RESULTS = REPO_ROOT / "results" / "llm"
PRIMARY = RESULTS / "primary" / "path_metrics.csv"
PROCESSES = RESULTS / "processes" / "path_metrics.csv"
FIXED = RESULTS / "fixed" / "opt_path_metrics.csv"
STUDY_PATHS = ROOT / "study_paths"
HUMAN_WATERMARK = ROOT / "human_watermark.npz"
OUTPUT = RESULTS / "paper_outputs"
FIGURES = OUTPUT / "figures"
THRESHOLD_EXACT = 48.897201698676575
HUMAN_WATERMARK_SHA256 = "57f648d8b35d58db1eeadcd4f02fa838d05635bbdb003c622ed868c7a357fd22"

REPORT_LABELS = {
    "swz_one_shot_general_fdr10": "First alarm + global min.",
    "swz_refresh_whole_block_general_fdr10": "Refresh + whole block",
    "swz_refresh_local_general_fdr10": "Refresh + global min.",
    "wa_refresh_general_fdr10": "WA",
    "og_refresh_general_fdr10": "OG",
    "average_refresh_general_fdr10": "50/50 average",
}

MONITORING_METHODS = (
    "swz_one_shot_general_fdr10",
    "swz_refresh_whole_block_general_fdr10",
    "swz_refresh_local_general_fdr10",
)
PROCESS_METHODS = (
    "average_refresh_general_fdr10",
    "og_refresh_general_fdr10",
    "wa_refresh_general_fdr10",
)


def method_order(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "WA",
        "OG",
        "50/50 average",
        r"Fixed $\lambda=0.1$",
        r"Fixed $\lambda=0.25$",
        r"Fixed $\lambda=0.5$",
        r"Fixed $\lambda=0.75$",
    ]
    labels = list(dict.fromkeys(frame["method_label"].astype(str)))
    return [label for label in preferred if label in labels] + sorted(
        label for label in labels if label not in preferred
    )


def select_representative_path(
    frame: pd.DataFrame,
    schedule_id: str,
    temperature: float,
    label: str,
) -> dict:
    block = frame[
        (frame["schedule_id"] == schedule_id)
        & np.isclose(frame["temperature"], temperature)
    ].copy()
    if block.empty:
        raise RuntimeError(f"no representative-path candidates for {label}")
    median_reports = float(block["reports"].median())
    block["distance"] = (block["reports"] - median_reports).abs()
    row = block.sort_values(["distance", "path_uid"]).iloc[0]
    return {
        "label": label,
        "schedule_id": schedule_id,
        "temperature": float(temperature),
        "path_uid": str(row["path_uid"]),
        "reports": int(row["reports"]),
        "median_reports": median_reports,
        "selection_rule": "report count nearest the cell median, ties by path_uid",
    }


def token_fdp(row: dict) -> float:
    if "token_fdp" in row and pd.notna(row["token_fdp"]):
        return float(row["token_fdp"])
    predicted = float(row.get("predicted_tokens", 0.0))
    false_positive = float(row.get("false_positive_tokens", 0.0))
    return false_positive / max(predicted, 1.0)


def stratified_mean(group: list[dict], values: list[float]) -> float:
    mean, _, _ = primary.prompt_stratified_finite_mean(group, values)
    return float(mean)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    grouped: dict[tuple, list[dict]] = {}
    for row in frame.to_dict("records"):
        key = (
            str(row["scenario_id"]),
            str(row["schedule_id"]),
            float(row["temperature"]),
            str(row["method"]),
            str(row["method_label"]),
        )
        grouped.setdefault(key, []).append(row)

    records = []
    for key, group in sorted(grouped.items()):
        scenario_id, schedule_id, temperature, method, method_label = key
        mean_lambda = math.nan
        if all("fixed_lambda" in row for row in group):
            mean_lambda = float(group[0]["fixed_lambda"])
        elif all("mean_eta" in row for row in group):
            mean_lambda = stratified_mean(
                group, [float(row["mean_eta"]) for row in group]
            )
        records.append(
            {
                "scenario_id": scenario_id,
                "schedule_id": schedule_id,
                "temperature": temperature,
                "method": method,
                "method_label": REPORT_LABELS.get(method, method_label),
                "n_paths": len(group),
                "mean_lambda": mean_lambda,
                "mean_reports": stratified_mean(
                    group, [float(row["reports"]) for row in group]
                ),
                "power": stratified_mean(
                    group, [float(row.get("token_recall", math.nan)) for row in group]
                ),
                "coverage_fdp": stratified_mean(
                    group, [token_fdp(row) for row in group]
                ),
                "iou": stratified_mean(
                    group, [float(row.get("token_iou", math.nan)) for row in group]
                ),
                "final_fdr": stratified_mean(
                    group, [float(row["final_fdp"]) for row in group]
                ),
                "uniform_fdr": stratified_mean(
                    group, [float(row["uniform_fdp"]) for row in group]
                ),
            }
        )
    return pd.DataFrame(records)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [path for path in (PRIMARY, PROCESSES, FIXED) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Saved-path replay is incomplete: " + ", ".join(map(str, missing))
        )

    primary_frame = pd.read_csv(PRIMARY)
    process_frame = pd.read_csv(PROCESSES)
    fixed_frame = pd.read_csv(FIXED)

    table2 = aggregate(
        primary_frame[
            (primary_frame["schedule_id"] == "two_l200_g050")
            & primary_frame["method"].isin(MONITORING_METHODS)
        ]
    )
    table4 = aggregate(
        primary_frame[
            (primary_frame["schedule_id"] == "four_l050_g025")
            & (primary_frame["method"] == "swz_refresh_local_general_fdr10")
        ]
    )
    table5 = pd.concat(
        [
            aggregate(
                process_frame[
                    (process_frame["schedule_id"] == "two_l200_g050")
                    & (process_frame["monitoring_mode"] == "refresh")
                    & process_frame["method"].isin(PROCESS_METHODS)
                ]
            ),
            aggregate(
                fixed_frame[fixed_frame["schedule_id"] == "two_l200_g050"]
            ),
        ],
        ignore_index=True,
    )

    wa_lambda = table2[
        table2["method"] == "swz_refresh_local_general_fdr10"
    ].set_index("temperature")["mean_lambda"]
    for method in ("wa_refresh_general_fdr10", "average_refresh_general_fdr10"):
        mask = table5["method"] == method
        table5.loc[mask, "mean_lambda"] = table5.loc[mask, "temperature"].map(
            wa_lambda
        )

    all_rows = pd.concat(
        [
            table2.assign(table="Table 2"),
            table4.assign(table="Table 4"),
            table5.assign(table="Table 5"),
        ],
        ignore_index=True,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table2.to_csv(OUTPUT / "table2_monitoring_comparison.csv", index=False)
    table4.to_csv(OUTPUT / "table4_four_interval.csv", index=False)
    table5.to_csv(OUTPUT / "table5_process_comparison.csv", index=False)
    all_rows.to_csv(OUTPUT / "section5_llm_tables.csv", index=False)
    return primary_frame, table2, table4, table5


def build_schedule_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    designs = (
        (
            "figure8_two_interval_schedule.png",
            "Two-interval null-alternative sequence",
            ((51, 250), (301, 500)),
        ),
        (
            "figure12_four_interval_schedule.png",
            "Four-interval null-alternative sequence",
            ((51, 100), (126, 175), (201, 250), (276, 325)),
        ),
    )
    for filename, title, regions in designs:
        fig, axis = plt.subplots(figsize=(10.5, 1.9))
        axis.hlines(0, 1, 600, color="#D7DEE2", linewidth=14)
        for start, end in regions:
            axis.hlines(0, start, end, color="#6D597A", linewidth=14)
        axis.set_xlim(1, 600)
        axis.set_ylim(-0.7, 0.7)
        axis.set_yticks([])
        axis.set_xlabel("Token position", fontsize=12)
        axis.set_title(title, fontsize=13)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=10)
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=300, facecolor="white")
        plt.close(fig)


def representative_rows(primary_frame: pd.DataFrame) -> list[dict]:
    local = primary_frame[
        primary_frame["method"] == "swz_refresh_local_general_fdr10"
    ]
    targets = (
        ("two_l200_g050", 0.75, "Two intervals, temperature 0.75"),
        ("two_l200_g050", 1.0, "Two intervals, temperature 1"),
        ("four_l050_g025", 1.0, "Four intervals, temperature 1"),
    )
    return [
        select_representative_path(local, schedule, temperature, label)
        for schedule, temperature, label in targets
    ]


def draw_trajectory(axis: plt.Axes, selection: dict, spec_by_uid: dict) -> None:
    path_uid = selection["path_uid"]
    spec = spec_by_uid[path_uid]
    checkpoint = STUDY_PATHS / "checkpoints" / f"{path_uid}.npz"
    saved = primary.generation.load_and_validate_checkpoint(checkpoint, spec)
    pivots = np.asarray(saved["pivot_y"], dtype=np.float64)
    calibration = primary.general_calibration_tag(0.10)
    reports_by_method, traces = primary.replay_methods(
        pivots, THRESHOLD_EXACT, calibration
    )
    method = "swz_refresh_local_general_fdr10"
    reports = reports_by_method[method]
    wealth = np.maximum(np.asarray(traces[method].candidate_logwealth), -4.0)
    token = np.arange(1, pivots.size + 1)

    for start, end in spec.intended_regions:
        axis.axvspan(start, end, color="#E8C98F", alpha=0.42, linewidth=0)
    axis.plot(token, wealth, color="#174D67", linewidth=1.35)
    axis.axhline(math.log(THRESHOLD_EXACT), color="#A33A35", linestyle="--")
    alarms = np.flatnonzero(np.asarray(traces[method].alarm, dtype=bool)) + 1
    if alarms.size:
        axis.scatter(alarms, wealth[alarms - 1], color="#A33A35", s=18, zorder=5)
    for report in reports:
        axis.hlines(
            -3.72,
            int(report["sigma"]) + 1,
            int(report["tau"]),
            color="#1E8273",
            linewidth=3.4,
        )
    axis.set_ylim(-4.15, max(math.log(THRESHOLD_EXACT) + 1, np.nanmax(wealth) + 0.4))
    axis.set_ylabel("log refreshing process", fontsize=12)
    axis.set_title(
        f"{selection['label']}; localized reports={selection['reports']}",
        loc="left",
        fontsize=12,
    )
    axis.tick_params(labelsize=10)
    axis.grid(axis="y", color="#D8E0E4", linewidth=0.5)


def trajectory_legend() -> list[object]:
    return [
        Line2D([0], [0], color="#174D67", lw=1.5, label="log wealth"),
        Line2D(
            [0], [0], color="#A33A35", lw=1.2, ls="--", label="threshold"
        ),
        Line2D(
            [0], [0], color="#A33A35", marker="o", lw=0, label="alarm/reset"
        ),
        Line2D([0], [0], color="#1E8273", lw=3.2, label="localized report"),
        Patch(facecolor="#E8C98F", edgecolor="none", label="watermarked region"),
    ]


def build_trajectory_figures(primary_frame: pd.DataFrame) -> None:
    selections = representative_rows(primary_frame)
    specs = primary.generation.build_manifest(
        primary.generation.study_scenarios(),
        batch_size=primary.generation.DEFAULT_BATCH_SIZE,
    )
    spec_by_uid = {spec.path_uid: spec for spec in specs}

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for axis, selection in zip(axes, selections[:2], strict=True):
        draw_trajectory(axis, selection, spec_by_uid)
    axes[-1].set_xlabel("Token position", fontsize=12)
    fig.suptitle("Representative refreshing trajectories", fontsize=16)
    fig.legend(
        handles=trajectory_legend(),
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(
        FIGURES / "figure9_representative_two_interval_trajectories.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 4.2))
    draw_trajectory(axis, selections[2], spec_by_uid)
    axis.set_xlabel("Token position", fontsize=12)
    fig.suptitle("Representative refreshing trajectory: four-interval setting", fontsize=16)
    fig.legend(
        handles=trajectory_legend(),
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.93))
    fig.savefig(
        FIGURES / "figure13_representative_four_interval_trajectory.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)

    (OUTPUT / "representative_paths.json").write_text(
        json.dumps(selections, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_tradeoff_figure(table5: pd.DataFrame) -> None:
    labels = method_order(table5)
    palette = plt.get_cmap("tab10").colors[: len(labels)]
    colors = dict(zip(labels, palette, strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, temperature in zip(axes, (0.75, 1.0), strict=True):
        block = table5[np.isclose(table5["temperature"], temperature)].set_index(
            "method_label"
        )
        for label in labels:
            row = block.loc[label]
            axis.scatter(
                float(row["power"]),
                float(row["coverage_fdp"]),
                color=colors[label],
                s=48,
                label=label,
            )
        axis.set_xlabel("Power", fontsize=12)
        axis.set_ylabel("Coverage FDP", fontsize=12)
        axis.set_title(f"Temperature {temperature:g}", fontsize=12)
        axis.grid(color="#D8E0E4", linewidth=0.6)
    handles = [
        Line2D([0], [0], marker="o", lw=0, color=colors[label], label=label)
        for label in labels
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(FIGURES / "figure14_power_coverage_fdp.png", dpi=300, facecolor="white")
    plt.close(fig)


def contiguous_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1) + 1
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def build_human_watermark_figure() -> None:
    import hashlib

    digest = hashlib.sha256(HUMAN_WATERMARK.read_bytes()).hexdigest()
    if digest != HUMAN_WATERMARK_SHA256:
        raise RuntimeError("human/watermark case data failed its SHA-256 check")

    with np.load(HUMAN_WATERMARK, allow_pickle=False) as archive:
        pivots = np.asarray(archive["pivot_y"], dtype=np.float64)
        truth = np.asarray(archive["is_watermarked"], dtype=bool)

    regions = (Region(153, 252), Region(405, 504))
    expected_truth = np.zeros(pivots.size, dtype=bool)
    for region in regions:
        expected_truth[region.start - 1 : region.end] = True
    if pivots.size != 655 or not np.array_equal(truth, expected_truth):
        raise RuntimeError("human/watermark case design changed")

    detector = run_refreshing_from_pivots(
        pivots,
        strategy="adaptive_cumulative",
        threshold=49.0,
        cap=0.5,
    )
    selected = np.zeros(pivots.size, dtype=bool)
    for report in detector.reports:
        selected[report.interval_start - 1 : report.interval_end] = True
    metrics = path_metrics(detector.reports, regions, horizon=pivots.size)
    token_fdp = float(metrics["false_positive_tokens"]) / max(
        int(metrics["predicted_tokens"]), 1
    )
    expected = (26, 0.895, 0.02185792349726776, 0.8774509803921569)
    observed = (
        len(detector.reports),
        float(metrics["token_recall"]),
        token_fdp,
        float(metrics["token_iou"]),
    )
    if observed[0] != expected[0] or not np.allclose(observed[1:], expected[1:]):
        raise RuntimeError("human/watermark detector replay changed")

    human_color = "#D7DEE2"
    watermark_color = "#F4DDAF"
    curve_color = "#0B6E8E"
    alarm_color = "#B94A48"
    report_color = "#188977"
    ink = "#263640"
    grid = "#D5DEE3"
    positions = np.arange(1, pivots.size + 1)
    blocks = (
        (1, 152, "H", human_color),
        (153, 252, "WM", watermark_color),
        (253, 404, "H", human_color),
        (405, 504, "WM", watermark_color),
        (505, 655, "H", human_color),
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.5, 7.4), facecolor="white")
    layout = fig.add_gridspec(3, 1, height_ratios=(0.75, 1.55, 3.0), hspace=0.16)
    source_axis = fig.add_subplot(layout[0])
    pivot_axis = fig.add_subplot(layout[1], sharex=source_axis)
    process_axis = fig.add_subplot(layout[2], sharex=source_axis)

    for start, end, label, color in blocks:
        source_axis.broken_barh(
            [(start - 0.5, end - start + 1)],
            (1.02, 0.56),
            facecolors=color,
            edgecolors="white",
            linewidth=1.1,
        )
        source_axis.text(
            (start + end) / 2,
            1.30,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color=ink,
        )
    for start, end in contiguous_intervals(selected):
        source_axis.broken_barh(
            [(start - 0.5, end - start + 1)],
            (0.18, 0.42),
            facecolors=report_color,
            edgecolors="white",
            linewidth=0.7,
        )
    source_axis.text(-0.015, 1.30, "source", transform=source_axis.get_yaxis_transform(),
                     ha="right", va="center", fontsize=11, color=ink)
    source_axis.text(-0.015, 0.39, "selected", transform=source_axis.get_yaxis_transform(),
                     ha="right", va="center", fontsize=11, color=ink)
    source_axis.set_ylim(0.0, 1.75)
    source_axis.axis("off")

    for region in regions:
        for axis in (pivot_axis, process_axis):
            axis.axvspan(
                region.start - 0.5,
                region.end + 0.5,
                color=watermark_color,
                alpha=0.58,
                linewidth=0,
                zorder=0,
            )
    pivot_axis.plot(positions, pivots, color="#5F6D75", linewidth=0.8, zorder=2)
    pivot_axis.set_ylim(0.0, 1.0)
    pivot_axis.set_ylabel(r"Pivotal statistic $Y_t$", fontsize=11)
    pivot_axis.tick_params(axis="x", labelbottom=False)

    candidate = np.asarray(detector.candidate_logwealth, dtype=float)
    process_axis.plot(positions, candidate, color=curve_color, linewidth=1.35, zorder=3)
    process_axis.axhline(
        np.log(49.0),
        color=alarm_color,
        linestyle="--",
        linewidth=1.4,
        zorder=2,
    )
    alarm_positions = positions[np.asarray(detector.alarm, dtype=bool)]
    process_axis.scatter(
        alarm_positions,
        candidate[np.asarray(detector.alarm, dtype=bool)],
        color=alarm_color,
        s=22,
        zorder=4,
    )
    lower = min(float(candidate.min()), -1.0)
    report_y = lower + 0.35
    for report in detector.reports:
        process_axis.plot(
            [report.interval_start, report.interval_end],
            [report_y, report_y],
            color=report_color,
            linewidth=3.2,
            solid_capstyle="butt",
            zorder=4,
        )
    process_axis.set_ylim(lower - 0.15, max(float(candidate.max()), np.log(49.0)) + 0.55)
    process_axis.set_ylabel("log refreshing process", fontsize=11)
    process_axis.set_xlabel("Token position", fontsize=11)
    process_axis.set_xlim(0.5, pivots.size + 0.5)

    for axis in (pivot_axis, process_axis):
        axis.grid(color=grid, linewidth=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#9FADB5")
        axis.tick_params(labelsize=10, colors=ink)

    fig.suptitle(
        "Refreshing process for mixed Efron and watermarked text",
        fontsize=15,
        color=ink,
        y=0.99,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=curve_color, linewidth=1.8, label="log refreshing process"),
            Line2D([], [], color=alarm_color, linestyle="--", linewidth=1.6,
                   label=r"threshold $\log(49)$"),
            Line2D([], [], marker="o", linestyle="none", color=alarm_color,
                   markersize=6, label="alarm and refresh"),
            Line2D([], [], color=report_color, linewidth=3.2, label="localized report"),
            Patch(facecolor=watermark_color, label="watermarked OPT-1.3B block"),
            Patch(facecolor=human_color, label="fixed Efron block"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.93, bottom=0.15)
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"human_watermark_refreshing_process.{suffix}",
            dpi=300,
            facecolor="white",
        )
    plt.close(fig)


def main() -> int:
    primary_frame, _, _, table5 = build_tables()
    build_schedule_figures()
    build_trajectory_figures(primary_frame)
    build_tradeoff_figure(table5)
    build_human_watermark_figure()
    print(f"Section 5.2 outputs written to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
