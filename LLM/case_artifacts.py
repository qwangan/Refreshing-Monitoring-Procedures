"""Paper-ready artifacts shared by the two sentence-aligned mixed cases."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np


def contiguous_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1) + 1
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def render_refreshing_process(
    output_pdf: Path,
    output_png: Path,
    arrays: dict[str, np.ndarray],
    blocks: Sequence[dict],
    detector,
    *,
    threshold: float,
    watermark_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    pivots = np.asarray(arrays["pivot_y"], dtype=float)
    selected = np.asarray(arrays["selected_by_any_report"], dtype=bool)
    human_color = "#D7DEE2"
    watermark_color = "#F4DDAF"
    curve_color = "#0B6E8E"
    alarm_color = "#B94A48"
    report_color = "#188977"
    ink = "#263640"
    grid = "#D5DEE3"
    positions = np.arange(1, pivots.size + 1)

    fig = plt.figure(figsize=(12.5, 7.4), facecolor="white")
    layout = fig.add_gridspec(3, 1, height_ratios=(0.75, 1.55, 3.0), hspace=0.16)
    source_axis = fig.add_subplot(layout[0])
    pivot_axis = fig.add_subplot(layout[1], sharex=source_axis)
    process_axis = fig.add_subplot(layout[2], sharex=source_axis)

    for block in blocks:
        start, end = int(block["start"]), int(block["end"])
        watermarked = block["kind"] == "watermarked"
        source_axis.broken_barh(
            [(start - 0.5, end - start + 1)],
            (1.02, 0.56),
            facecolors=watermark_color if watermarked else human_color,
            edgecolors="white",
            linewidth=1.1,
        )
        source_axis.text(
            (start + end) / 2,
            1.30,
            "WM" if watermarked else "H",
            ha="center",
            va="center",
            fontsize=11,
            color=ink,
        )
        if watermarked:
            for axis in (pivot_axis, process_axis):
                axis.axvspan(
                    start - 0.5,
                    end + 0.5,
                    color=watermark_color,
                    alpha=0.58,
                    linewidth=0,
                    zorder=0,
                )
    for start, end in contiguous_intervals(selected):
        source_axis.broken_barh(
            [(start - 0.5, end - start + 1)],
            (0.18, 0.42),
            facecolors=report_color,
            edgecolors="white",
            linewidth=0.7,
        )
    source_axis.text(
        -0.015, 1.30, "source", transform=source_axis.get_yaxis_transform(),
        ha="right", va="center", fontsize=11, color=ink,
    )
    source_axis.text(
        -0.015, 0.39, "selected", transform=source_axis.get_yaxis_transform(),
        ha="right", va="center", fontsize=11, color=ink,
    )
    source_axis.set_ylim(0.0, 1.75)
    source_axis.axis("off")

    pivot_axis.plot(positions, pivots, color="#5F6D75", linewidth=0.8, zorder=2)
    pivot_axis.set_ylim(0.0, 1.0)
    pivot_axis.set_ylabel(r"Pivotal statistic $Y_t$", fontsize=11)
    pivot_axis.tick_params(axis="x", labelbottom=False)

    candidate = np.asarray(detector.candidate_logwealth, dtype=float)
    process_axis.plot(positions, candidate, color=curve_color, linewidth=1.35, zorder=3)
    process_axis.axhline(
        math.log(threshold), color=alarm_color, linestyle="--", linewidth=1.4, zorder=2
    )
    alarm = np.asarray(detector.alarm, dtype=bool)
    process_axis.scatter(positions[alarm], candidate[alarm], color=alarm_color, s=22, zorder=4)
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
    process_axis.set_ylim(
        lower - 0.15,
        max(float(candidate.max()), math.log(threshold)) + 0.55,
    )
    process_axis.set_ylabel("log refreshing process", fontsize=11)
    process_axis.set_xlabel("Token position", fontsize=11)
    process_axis.set_xlim(0.5, pivots.size + 0.5)

    for axis in (pivot_axis, process_axis):
        axis.grid(color=grid, linewidth=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#9FADB5")
        axis.tick_params(labelsize=10, colors=ink)

    fig.suptitle(
        f"Refreshing process for mixed Efron and {watermark_label} text",
        fontsize=15,
        color=ink,
        y=0.99,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=curve_color, linewidth=1.8, label="log refreshing process"),
            Line2D([], [], color=alarm_color, linestyle="--", linewidth=1.6,
                   label=rf"threshold $\log({threshold:g})$"),
            Line2D([], [], marker="o", linestyle="none", color=alarm_color,
                   markersize=6, label="alarm and refresh"),
            Line2D([], [], color=report_color, linewidth=3.2, label="localized report"),
            Patch(facecolor=watermark_color, label=f"{watermark_label} block"),
            Patch(facecolor=human_color, label="fixed Efron block"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.93, bottom=0.15)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, facecolor="white")
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)
