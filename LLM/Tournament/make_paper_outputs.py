#!/usr/bin/env python3
"""Build the Tournament figures reported in Section 5.2 and Appendix B."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
RESULTS = REPO_ROOT / "results" / "llm" / "tournament"
STUDY_RESULTS = RESULTS / "study"
CASE_RESULTS = RESULTS / "efron_case"
TRACES = STUDY_RESULTS / "traces"
GENERATED = RESULTS / "paper_outputs"

INK = "#21323D"
NAVY = "#174D67"
RED = "#A33A35"
TEAL = "#1E8273"
BLUE = "#2878B5"
PALE_TAN = "#E8C98F"
GRID = "#D7E0E4"
THRESHOLD = 49.0

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save(fig: plt.Figure, stem: str, *, png_only: bool = False) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    fig.savefig(GENERATED / f"{stem}.png", dpi=300, facecolor="white")
    if not png_only:
        fig.savefig(GENERATED / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def read_summary() -> list[dict[str, str]]:
    with (STUDY_RESULTS / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_trace(path_uid: str) -> dict[str, np.ndarray]:
    with np.load(TRACES / f"{path_uid}.npz", allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def localized_reports(trace: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    alarms = np.flatnonzero(np.asarray(trace["weight_adaptive__alarm"], dtype=bool))
    running_min = np.asarray(trace["weight_adaptive__running_min_time"], dtype=int)
    return [(int(running_min[index]) + 1, int(index) + 1) for index in alarms]


def draw_trajectory(
    axis: plt.Axes,
    trace: dict[str, np.ndarray],
    title: str,
    watermarked_regions: tuple[tuple[int, int], ...],
) -> None:
    wealth = np.maximum(
        np.asarray(trace["weight_adaptive__candidate_logwealth"], dtype=float),
        -4.0,
    )
    positions = np.arange(1, wealth.size + 1)
    for start, end in watermarked_regions:
        axis.axvspan(start - 0.5, end + 0.5, color=PALE_TAN, alpha=0.42, linewidth=0)
    axis.plot(positions, wealth, color=NAVY, linewidth=1.35)
    axis.axhline(math.log(THRESHOLD), color=RED, linestyle="--", linewidth=1.2)
    alarms = np.flatnonzero(np.asarray(trace["weight_adaptive__alarm"], dtype=bool))
    if alarms.size:
        axis.scatter(
            positions[alarms],
            wealth[alarms],
            color=RED,
            s=22,
            zorder=5,
        )
    for start, end in localized_reports(trace):
        axis.hlines(-3.72, start, end, color=TEAL, linewidth=3.5)
    axis.set_xlim(0, wealth.size)
    axis.set_ylim(-4.15, max(math.log(THRESHOLD) + 1.0, float(np.nanmax(wealth)) + 0.4))
    axis.set_ylabel("log refreshing process", fontsize=13)
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=8)
    axis.tick_params(axis="both", labelsize=11, width=0.8, length=4)
    axis.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def trajectory_legend() -> list[object]:
    return [
        Line2D([0], [0], color=NAVY, lw=1.7, label="log refreshing process"),
        Line2D(
            [0],
            [0],
            color=RED,
            lw=1.4,
            ls="--",
            label=r"threshold $\log(49)$",
        ),
        Line2D(
            [0],
            [0],
            color=RED,
            marker="o",
            lw=0,
            markersize=6,
            label="alarm/reset",
        ),
        Line2D([0], [0], color=TEAL, lw=3.5, label="localized report"),
        Patch(facecolor=PALE_TAN, edgecolor="none", label="watermarked region"),
    ]


def build_representative_trajectories() -> None:
    two_regions = ((51, 250), (301, 500))
    two_specs = (
        (
            "core_two_l200_g050_temp0p75__r0001__p00",
            r"$\vartheta=0.75$",
        ),
        (
            "core_two_l200_g050_temp1__r0001__p00",
            r"$\vartheta=1$",
        ),
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for axis, (path_uid, label) in zip(axes, two_specs, strict=True):
        trace = load_trace(path_uid)
        draw_trajectory(axis, trace, label, two_regions)
    axes[-1].set_xlabel("Token position", fontsize=13)
    fig.suptitle(
        "Representative refreshing trajectories",
        color=NAVY,
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles=trajectory_legend(),
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
        fontsize=11,
        handlelength=2.3,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.16, hspace=0.28)
    save(fig, "figure11_primary_trajectories")

    path_uid = "stress_four_l050_g025_temp1__r0001__p00"
    trace = load_trace(path_uid)
    fig, axis = plt.subplots(figsize=(11, 4.25))
    draw_trajectory(
        axis,
        trace,
        r"$\vartheta=1$",
        ((51, 100), (126, 175), (201, 250), (276, 325)),
    )
    axis.set_xlabel("Token position", fontsize=13)
    fig.suptitle(
        "Representative refreshing trajectory: four-interval setting",
        color=NAVY,
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )
    fig.legend(
        handles=trajectory_legend(),
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
        fontsize=11,
        handlelength=2.3,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.79, bottom=0.25)
    save(fig, "figureB19_four_interval_trajectory")


PROCESS = {
    "Average": {"marker": "D", "color": "#8C6BB1"},
    "OG": {"marker": "s", "color": "#4D4D4D"},
    "WA": {"marker": "o", "color": TEAL},
    r"Nonadaptive $\lambda=0.10$": {"marker": "^", "color": "#D95F02"},
    r"Nonadaptive $\lambda=0.25$": {"marker": "v", "color": "#E6AB02"},
    r"Nonadaptive $\lambda=0.50$": {"marker": "P", "color": BLUE},
    r"Nonadaptive $\lambda=0.75$": {"marker": "X", "color": RED},
}

METHOD_LABELS = {
    "average_50_50": "Average",
    "online_grenander": "OG",
    "weight_adaptive": "WA",
    "fixed_lambda_0p10": r"Nonadaptive $\lambda=0.10$",
    "fixed_lambda_0p25": r"Nonadaptive $\lambda=0.25$",
    "fixed_lambda_0p50": r"Nonadaptive $\lambda=0.50$",
    "fixed_lambda_0p75": r"Nonadaptive $\lambda=0.75$",
}


def build_process_tradeoff() -> None:
    rows = [row for row in read_summary() if row["experiment"] == "seven_processes"]
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.55), sharey=True)
    for axis, temperature in zip(axes, (0.75, 1.0), strict=True):
        block = {
            METHOD_LABELS[row["method"]]: row
            for row in rows
            if math.isclose(float(row["temperature"]), temperature)
        }
        for label in PROCESS:
            row = block[label]
            style = PROCESS[label]
            axis.scatter(
                float(row["power"]),
                float(row["coverage_fdp"]),
                s=58,
                marker=style["marker"],
                color=style["color"],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
        axis.set_title(
            rf"Temperature $\vartheta={temperature:g}$",
            color=NAVY,
            fontweight="bold",
        )
        axis.set_xlabel("Power")
        axis.grid(color=GRID, linewidth=0.6, alpha=0.8)
        axis.set_xlim(0.62 if temperature == 0.75 else 0.78, 0.98)
        axis.set_ylim(0.0, 0.155)
    axes[0].set_ylabel("Coverage FDP")
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markersize=7.5,
                label=label,
            )
            for label, style in PROCESS.items()
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=1.3,
    )
    fig.suptitle(
        r"Central Tournament OPT-1.3B process tradeoff: common threshold $\gamma=49$",
        color=INK,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(top=0.78, bottom=0.25, wspace=0.17)
    save(fig, "figureB21_process_power_coverage_fdp")


def build_efron_trajectory() -> None:
    case = json.loads((CASE_RESULTS / "case.json").read_text(encoding="utf-8"))
    with np.load(CASE_RESULTS / "case_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    trace = {
        "weight_adaptive__candidate_logwealth": arrays["candidate_logwealth"],
        "weight_adaptive__alarm": arrays["alarm"],
        "weight_adaptive__running_min_time": arrays["running_min_time"],
    }
    regions = tuple(
        (int(block["start"]), int(block["end"]))
        for block in case["blocks"]
        if block["kind"] == "watermarked"
    )
    fig, axis = plt.subplots(figsize=(11, 4.25))
    draw_trajectory(axis, trace, "Tournament mixed document", regions)
    axis.set_xlabel("Token position", fontsize=13)
    fig.legend(
        handles=trajectory_legend(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.25)
    save(fig, "figure13_efron_trajectory")


def main() -> None:
    build_representative_trajectories()
    build_process_tradeoff()
    build_efron_trajectory()
    print(f"Rebuilt report figures in {GENERATED}")


if __name__ == "__main__":
    main()
