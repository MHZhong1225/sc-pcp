"""Render the code-native SC-PCP method schematic.

This renderer is intentionally independent of experiment artifacts: it draws the
frozen method definition and claim boundary without loading or rerunning a seed.
It publishes an editable working bundle and a PDF-only manuscript directory.

Example
-------
conda run -n ucp python tools/render_method_schematic.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RENDER_PROTOCOL = "scpcp_method_schematic_render_v1"
FIGURE_STEM = "figure_method_schematic"
DEFAULT_WORK_OUTPUT = ROOT / "results/work/method_schematic_20260826"
DEFAULT_PAPER_OUTPUT = ROOT / "results/paper_method_schematic_20260826"

PAPER_FILES = {f"{FIGURE_STEM}.pdf"}
WORK_FILES = {
    f"{FIGURE_STEM}.pdf",
    f"{FIGURE_STEM}.png",
    f"{FIGURE_STEM}.svg",
    f"{FIGURE_STEM}.tiff",
    f"{FIGURE_STEM}_source.csv",
    "figure_contract.json",
    "figure_qa.md",
    "render_manifest.json",
}

BLUE = "#4394F8"
BLUE_DARK = "#2B67A0"
BLUE_PALE = "#E9F3FF"
TEAL = "#42949E"
TEAL_PALE = "#E5F4F4"
VIOLET = "#7A3D9D"
VIOLET_PALE = "#F2EAF7"
NEUTRAL_DARK = "#3F4852"
NEUTRAL_MID = "#7E8C9C"
NEUTRAL_PALE = "#F2F4F6"
GOLD_PALE = "#FFF6D8"
GOLD_EDGE = "#C99A24"

FIGURE_CONTRACT: Mapping[str, Any] = {
    "core_conclusion": (
        "At each stage, SC-PCP transports the post-action score event through the "
        "entire committed policy prefix, selects the narrowest empirically feasible "
        "current radius, and commits it before advancing in sequential time."
    ),
    "archetype": "schematic-led composite",
    "role_in_manuscript": "method and causal mechanism",
    "backend": "Python/matplotlib",
    "final_size_inches": [7.20, 4.95],
    "font": "Times New Roman with Times/DejaVu Serif fallback; STIX math",
    "panel_map": {
        "a": (
            "Nonanticipating deployment: past committed radii determine occupancy at "
            "S_t; the current candidate radius determines actionwise sets, the current "
            "policy, A_t, and the post-action score R_t."
        ),
        "b": (
            "Logged-data calibration: full committed-prefix weighting yields candidate "
            "coverage/width surfaces; the minimum-width feasible candidate is committed."
        ),
    },
    "evidence_hierarchy": {
        "hero": "full current-plus-history prefix and one-stage causal timing",
        "support": "coverage-constrained width selection followed by commitment",
    },
    "statistics": "No empirical statistic is plotted; the panel contains method definitions.",
    "source_data": f"{FIGURE_STEM}_source.csv",
    "image_integrity": "Code-native vector drawing; no raster source image or manipulation.",
    "reviewer_risks": [
        "Do not imply an exact fixed-point construction.",
        "Do not imply global K^T schedule optimization.",
        "Do not imply finite-sample, distribution-free, PAC, or data-conditional validity.",
        "Keep the current-action ratio because R_t is observed after A_t.",
    ],
    "claim_boundary": "plug-in asymptotic per-step marginal coverage",
}


@dataclass(frozen=True)
class Box:
    panel: str
    element_id: str
    x: float
    y: float
    width: float
    height: float
    title: str
    body: str
    note: str
    semantic_role: str
    fill: str
    edge: str
    body_size: float = 6.8


@dataclass(frozen=True)
class Arrow:
    panel: str
    element_id: str
    source_id: str
    target_id: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    label: str
    semantic_role: str
    color: str
    label_dx: float = 0.0
    label_dy: float = 0.0
    connectionstyle: str = "arc3,rad=0"
    linestyle: str = "solid"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-output", type=Path, default=DEFAULT_WORK_OUTPUT)
    parser.add_argument("--paper-output", type=Path, default=DEFAULT_PAPER_OUTPUT)
    args = parser.parse_args()
    render_bundle(args.work_output.resolve(), args.paper_output.resolve())
    print(args.paper_output.resolve())


def render_bundle(work_output: Path, paper_output: Path) -> None:
    """Atomically publish the working bundle and PDF-only paper output."""

    if work_output.exists() or paper_output.exists():
        raise FileExistsError("work-output and paper-output must both be fresh")
    if work_output == paper_output:
        raise ValueError("work-output and paper-output must be different directories")

    work_output.parent.mkdir(parents=True, exist_ok=True)
    paper_output.parent.mkdir(parents=True, exist_ok=True)
    staged_work = Path(
        tempfile.mkdtemp(prefix=f".{work_output.name}-", dir=work_output.parent)
    )
    staged_paper = Path(
        tempfile.mkdtemp(prefix=f".{paper_output.name}-", dir=paper_output.parent)
    )
    try:
        boxes, arrows = schematic_elements()
        _write_source_csv(
            staged_work / f"{FIGURE_STEM}_source.csv", boxes=boxes, arrows=arrows
        )
        _write_json(staged_work / "figure_contract.json", FIGURE_CONTRACT)
        apply_publication_style()
        figure = render_schematic(boxes=boxes, arrows=arrows)
        export_figure(
            figure,
            work_stem=staged_work / FIGURE_STEM,
            paper_path=staged_paper / f"{FIGURE_STEM}.pdf",
        )
        _write_qa(staged_work / "figure_qa.md")
        _write_manifest(
            staged_work / "render_manifest.json",
            work_root=staged_work,
            paper_root=staged_paper,
        )
        validate_bundle(staged_work, staged_paper)
        os.replace(staged_work, work_output)
        os.replace(staged_paper, paper_output)
    except BaseException:
        shutil.rmtree(staged_work, ignore_errors=True)
        shutil.rmtree(staged_paper, ignore_errors=True)
        raise


def apply_publication_style() -> None:
    """Apply the repository's editable Times New Roman paper style."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def schematic_elements() -> tuple[tuple[Box, ...], tuple[Arrow, ...]]:
    boxes = (
        Box(
            "a",
            "committed_history",
            0.025,
            0.570,
            0.175,
            0.215,
            "Committed past",
            r"$\widehat q_{0:t-1}$",
            r"earlier policies and actions",
            "Past radii are fixed before stage t.",
            NEUTRAL_PALE,
            NEUTRAL_MID,
            8.0,
        ),
        Box(
            "a",
            "target_occupancy",
            0.245,
            0.570,
            0.205,
            0.215,
            "Target occupancy at stage t",
            r"$d_t^{\widehat q_{0:t-1}}(S_t)$",
            "history-mediated state shift",
            "Historical policy ratios transport the state occupancy.",
            NEUTRAL_PALE,
            NEUTRAL_MID,
            7.5,
        ),
        Box(
            "a",
            "current_state",
            0.500,
            0.570,
            0.125,
            0.215,
            "Current state",
            r"$S_t$",
            "fixed before choosing $A_t$",
            "Nonanticipation: the current candidate does not alter S_t.",
            NEUTRAL_PALE,
            NEUTRAL_MID,
            9.0,
        ),
        Box(
            "a",
            "candidate_radius",
            0.020,
            0.175,
            0.125,
            0.235,
            "Current candidate",
            r"$r\in\mathcal{G}_t$",
            r"selected later as $\widehat q_t$",
            "Only the current stage grid is scanned.",
            BLUE_PALE,
            BLUE,
            8.4,
        ),
        Box(
            "a",
            "prediction_sets",
            0.180,
            0.175,
            0.165,
            0.235,
            "Actionwise prediction sets",
            r"$C_r(S_t,a)$",
            "one set for each candidate action",
            "The radius changes the actionwise uncertainty sets.",
            BLUE_PALE,
            BLUE,
            8.0,
        ),
        Box(
            "a",
            "indexed_policy",
            0.385,
            0.175,
            0.175,
            0.235,
            r"$q$-indexed treatment policy",
            r"$\pi_t^r(a\mid S_t)$",
            "behavior-anchored, ratio-capped",
            "The candidate set changes the current action distribution.",
            TEAL_PALE,
            TEAL,
            8.0,
        ),
        Box(
            "a",
            "current_action",
            0.600,
            0.175,
            0.095,
            0.235,
            "Treatment",
            r"$A_t$",
            r"drawn from $\pi_t^r$",
            "The current action precedes the score event.",
            TEAL_PALE,
            TEAL,
            9.2,
        ),
        Box(
            "a",
            "post_action_score",
            0.735,
            0.175,
            0.155,
            0.235,
            "Post-action outcome and score",
            r"$Y_{t+1},\ R_t$",
            r"coverage event $\{R_t\leq r\}$",
            "The current-action ratio is required for this post-action event.",
            GOLD_PALE,
            GOLD_EDGE,
            8.1,
        ),
        Box(
            "a",
            "next_state",
            0.915,
            0.175,
            0.075,
            0.235,
            "Next",
            r"$S_{t+1}$",
            "future occupancy",
            "The selected action propagates to the next stage.",
            VIOLET_PALE,
            VIOLET,
            8.6,
        ),
        Box(
            "b",
            "logged_trajectories",
            0.020,
            0.455,
            0.145,
            0.315,
            r"Logged trajectories under $\mu$",
            r"$(S_{i,0:t},A_{i,0:t},R_{i,t})$",
            "frozen nuisance models",
            "Historical patient trajectories provide calibration evidence.",
            NEUTRAL_PALE,
            NEUTRAL_MID,
            6.6,
        ),
        Box(
            "b",
            "full_prefix_weight",
            0.195,
            0.400,
            0.285,
            0.420,
            "Full committed-prefix transport",
            (
                r"$W_{i,t}(q_{0:t})=\prod_{j=0}^{t}$" "\n"
                r"$\dfrac{\pi_j^{q_j}(A_{ij}\mid S_{ij})}"
                r"{\mu_j(A_{ij}\mid S_{ij})}$"
            ),
            r"candidate column: $q_t\leftarrow r$; raw log-prefix is uncapped",
            "Both historical occupancy ratios and the current-action ratio are used.",
            BLUE_PALE,
            BLUE_DARK,
            6.3,
        ),
        Box(
            "b",
            "candidate_surfaces",
            0.515,
            0.440,
            0.175,
            0.340,
            "Candidate target-law surfaces",
            r"$\widehat C_t(r),\ \widehat{\mathcal{W}}_t(r)$",
            "H\u00e1jek coverage, normalized width, ESS",
            "Each candidate is evaluated under the policy law that it induces.",
            TEAL_PALE,
            TEAL,
            7.1,
        ),
        Box(
            "b",
            "stagewise_selection",
            0.725,
            0.390,
            0.250,
            0.430,
            "Choose the narrowest feasible current radius",
            (
                r"$\widehat q_t\in\arg\min_"
                r"{r\in\mathcal{G}_t:\,\widehat C_t(r)\geq 1-\alpha}$" "\n"
                r"$\widehat{\mathcal{W}}_t(r)$"
            ),
            "scan the whole current grid; return unavailable if none is feasible",
            "This is a causal stagewise rule, not a global schedule optimizer.",
            VIOLET_PALE,
            VIOLET,
            6.3,
        ),
        Box(
            "b",
            "commit_and_advance",
            0.725,
            0.075,
            0.250,
            0.190,
            r"Commit $\widehat q_t$ and its raw log-prefix",
            r"advance to stage $t+1$ with $\widehat q_{0:t}$ fixed",
            "sequential time, not fixed-point iteration",
            "The selected prefix becomes historical input at the next stage.",
            GOLD_PALE,
            GOLD_EDGE,
            6.5,
        ),
    )
    arrows = (
        Arrow(
            "a",
            "history_to_occupancy",
            "committed_history",
            "target_occupancy",
            0.200,
            0.677,
            0.245,
            0.677,
            "past actions",
            "Past policies change who reaches the current stage.",
            NEUTRAL_MID,
            label_dy=0.045,
        ),
        Arrow(
            "a",
            "occupancy_to_state",
            "target_occupancy",
            "current_state",
            0.450,
            0.677,
            0.500,
            0.677,
            "realize",
            "The target occupancy induces the distribution of S_t.",
            NEUTRAL_MID,
            label_dy=0.045,
        ),
        Arrow(
            "a",
            "state_to_sets",
            "current_state",
            "prediction_sets",
            0.555,
            0.570,
            0.300,
            0.410,
            "condition on state",
            "Prediction sets and policy are evaluated at the realized current state.",
            NEUTRAL_MID,
            label_dx=-0.010,
            label_dy=0.025,
            connectionstyle="arc3,rad=-0.12",
        ),
        Arrow(
            "a",
            "radius_to_sets",
            "candidate_radius",
            "prediction_sets",
            0.145,
            0.292,
            0.180,
            0.292,
            "sets",
            "The candidate radius indexes the actionwise prediction sets.",
            BLUE_DARK,
            label_dy=0.047,
        ),
        Arrow(
            "a",
            "sets_to_policy",
            "prediction_sets",
            "indexed_policy",
            0.345,
            0.292,
            0.385,
            0.292,
            "decision rule",
            "The prediction sets enter the treatment decision rule.",
            BLUE_DARK,
            label_dy=0.047,
        ),
        Arrow(
            "a",
            "policy_to_action",
            "indexed_policy",
            "current_action",
            0.560,
            0.292,
            0.600,
            0.292,
            "sample",
            "Treatment is sampled from the candidate-indexed policy.",
            TEAL,
            label_dy=0.047,
        ),
        Arrow(
            "a",
            "action_to_score",
            "current_action",
            "post_action_score",
            0.695,
            0.292,
            0.735,
            0.292,
            "observe",
            "Outcome and score occur after the current action.",
            GOLD_EDGE,
            label_dy=0.047,
        ),
        Arrow(
            "a",
            "score_to_next_state",
            "post_action_score",
            "next_state",
            0.890,
            0.292,
            0.915,
            0.292,
            "transition",
            "The action/outcome history propagates to S_{t+1}.",
            VIOLET,
            label_dy=0.047,
        ),
        Arrow(
            "b",
            "logs_to_weights",
            "logged_trajectories",
            "full_prefix_weight",
            0.165,
            0.610,
            0.195,
            0.610,
            "transport",
            "Reweight behavior-policy trajectories toward the candidate target law.",
            BLUE_DARK,
            label_dy=0.050,
        ),
        Arrow(
            "b",
            "weights_to_surfaces",
            "full_prefix_weight",
            "candidate_surfaces",
            0.480,
            0.610,
            0.515,
            0.610,
            "estimate",
            "Self-normalized weights estimate coverage and width.",
            TEAL,
            label_dy=0.050,
        ),
        Arrow(
            "b",
            "surfaces_to_selection",
            "candidate_surfaces",
            "stagewise_selection",
            0.690,
            0.610,
            0.725,
            0.610,
            "optimize",
            "Coverage defines feasibility; width chooses among feasible candidates.",
            VIOLET,
            label_dy=0.050,
        ),
        Arrow(
            "b",
            "selection_to_commit",
            "stagewise_selection",
            "commit_and_advance",
            0.850,
            0.390,
            0.850,
            0.265,
            "commit",
            "The selected radius and raw prefix are frozen before advancing.",
            GOLD_EDGE,
            label_dx=0.055,
        ),
    )
    return boxes, arrows


def render_schematic(*, boxes: Sequence[Box], arrows: Sequence[Arrow]) -> Figure:
    figure = plt.figure(figsize=tuple(FIGURE_CONTRACT["final_size_inches"]))
    axis_a = figure.add_axes((0.025, 0.535, 0.95, 0.405))
    axis_b = figure.add_axes((0.025, 0.095, 0.95, 0.365))
    for axis in (axis_a, axis_b):
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.axis("off")

    _panel_header(
        axis_a,
        "a",
        "Prediction-mediated treatment dynamics",
        "Past radii determine occupancy; the current radius determines the current action before the score is observed.",
    )
    _panel_header(
        axis_b,
        "b",
        "Committed-prefix marginal calibration",
        "Evaluate one current-stage grid under the full causal prefix, choose, then commit.",
    )
    panel_axes = {"a": axis_a, "b": axis_b}
    for box in boxes:
        _draw_box(panel_axes[box.panel], box)
    for arrow in arrows:
        _draw_arrow(panel_axes[arrow.panel], arrow)

    axis_a.text(
        0.675,
        0.680,
        r"Nonanticipation: $r$ cannot change the already realized $S_t$.",
        color=NEUTRAL_DARK,
        fontsize=6.1,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": NEUTRAL_MID,
            "linewidth": 0.7,
        },
    )
    axis_b.text(
        0.020,
        0.155,
        "Candidate evaluation uses uncapped cumulative prefix weights; subtracting a columnwise "
        "maximum log weight is numerical stabilization, not clipping.",
        fontsize=5.6,
        color=NEUTRAL_DARK,
        ha="left",
        va="center",
        linespacing=1.18,
    )
    figure.text(
        0.5,
        0.030,
        r"Statistical target: plug-in asymptotic per-step marginal coverage "
        r"$C_t(\widehat q_{0:t})\geq 1-\alpha-o_p(1)$ under the transport and uniform-convergence assumptions; "
        "not a finite-sample certificate.",
        ha="center",
        va="center",
        fontsize=5.9,
        color=NEUTRAL_DARK,
    )
    return figure


def _panel_header(axis: Axes, label: str, title: str, subtitle: str) -> None:
    axis.text(
        -0.006,
        0.965,
        label,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="top",
    )
    axis.text(
        0.025,
        0.965,
        title,
        fontsize=7.7,
        fontweight="bold",
        ha="left",
        va="top",
    )
    axis.text(
        0.025,
        0.895,
        subtitle,
        fontsize=5.9,
        color=NEUTRAL_DARK,
        ha="left",
        va="top",
    )


def _draw_box(axis: Axes, box: Box) -> None:
    patch = FancyBboxPatch(
        (box.x, box.y),
        box.width,
        box.height,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=box.edge,
        facecolor=box.fill,
        zorder=2,
    )
    axis.add_patch(patch)
    center_x = box.x + box.width / 2.0
    axis.text(
        center_x,
        box.y + box.height * 0.79,
        box.title,
        fontsize=6.35,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=3,
    )
    axis.text(
        center_x,
        box.y + box.height * 0.50,
        box.body,
        fontsize=box.body_size,
        ha="center",
        va="center",
        linespacing=1.05,
        zorder=3,
    )
    axis.text(
        center_x,
        box.y + box.height * 0.17,
        box.note,
        fontsize=4.85,
        color=NEUTRAL_DARK,
        ha="center",
        va="center",
        linespacing=1.05,
        zorder=3,
    )


def _draw_arrow(axis: Axes, arrow: Arrow) -> None:
    patch = FancyArrowPatch(
        (arrow.start_x, arrow.start_y),
        (arrow.end_x, arrow.end_y),
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=1.05,
        linestyle=arrow.linestyle,
        color=arrow.color,
        connectionstyle=arrow.connectionstyle,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=1,
    )
    axis.add_patch(patch)
    midpoint_x = (arrow.start_x + arrow.end_x) / 2.0 + arrow.label_dx
    midpoint_y = (arrow.start_y + arrow.end_y) / 2.0 + arrow.label_dy
    axis.text(
        midpoint_x,
        midpoint_y,
        arrow.label,
        fontsize=4.9,
        color=arrow.color,
        ha="center",
        va="center",
        zorder=4,
    )


def export_figure(
    figure: Figure,
    *,
    work_stem: Path,
    paper_path: Path,
    tiff_dpi: int = 600,
    png_dpi: int = 240,
) -> None:
    title = "SC-PCP committed-prefix marginal calibration"
    pdf_metadata = {
        "Title": title,
        "Creator": "SC-PCP method schematic renderer",
        "CreationDate": None,
        "ModDate": None,
    }
    svg_metadata = {
        "Title": title,
        "Creator": "SC-PCP method schematic renderer",
        "Date": None,
    }
    figure.savefig(
        work_stem.with_suffix(".svg"),
        format="svg",
        metadata=svg_metadata,
    )
    figure.savefig(
        work_stem.with_suffix(".pdf"),
        format="pdf",
        metadata=pdf_metadata,
    )
    shutil.copyfile(work_stem.with_suffix(".pdf"), paper_path)
    figure.savefig(
        work_stem.with_suffix(".tiff"),
        format="tiff",
        dpi=tiff_dpi,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(
        work_stem.with_suffix(".png"),
        format="png",
        dpi=png_dpi,
        metadata={"Software": "SC-PCP method schematic renderer"},
    )
    plt.close(figure)


def _write_source_csv(path: Path, *, boxes: Sequence[Box], arrows: Sequence[Arrow]) -> None:
    fields = (
        "panel",
        "kind",
        "element_id",
        "source_id",
        "target_id",
        "title_or_label",
        "body",
        "note",
        "semantic_role",
        "x",
        "y",
        "width",
        "height",
        "end_x",
        "end_y",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for box in boxes:
            writer.writerow(
                {
                    "panel": box.panel,
                    "kind": "box",
                    "element_id": box.element_id,
                    "source_id": "",
                    "target_id": "",
                    "title_or_label": box.title,
                    "body": box.body,
                    "note": box.note,
                    "semantic_role": box.semantic_role,
                    "x": box.x,
                    "y": box.y,
                    "width": box.width,
                    "height": box.height,
                    "end_x": "",
                    "end_y": "",
                }
            )
        for arrow in arrows:
            writer.writerow(
                {
                    "panel": arrow.panel,
                    "kind": "arrow",
                    "element_id": arrow.element_id,
                    "source_id": arrow.source_id,
                    "target_id": arrow.target_id,
                    "title_or_label": arrow.label,
                    "body": "",
                    "note": "",
                    "semantic_role": arrow.semantic_role,
                    "x": arrow.start_x,
                    "y": arrow.start_y,
                    "width": "",
                    "height": "",
                    "end_x": arrow.end_x,
                    "end_y": arrow.end_y,
                }
            )


def _write_qa(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "# SC-PCP method schematic QA",
                "",
                f"- Core conclusion: {FIGURE_CONTRACT['core_conclusion']}",
                "- Archetype: schematic-led composite; panel a is the causal-timing hero panel.",
                "- Backend: Python/matplotlib only for drawing, previews, exports, and visual QA.",
                "- Final size: 7.20 x 4.95 inches (double-column).",
                "- Typography: Times New Roman convention with serif fallbacks and STIX math; 5--8 pt at final size.",
                "- Panel a separates history-mediated occupancy from the current action-mediated score event.",
                "- Panel b contains the complete current-plus-history likelihood ratio, H\u00e1jek coverage/width surfaces, and the stagewise width argmin subject to estimated marginal coverage.",
                "- Claim boundary: plug-in asymptotic per-step marginal coverage; no finite-sample, distribution-free, PAC, data-conditional, or episode-wise simultaneous guarantee.",
                "- Optimization boundary: one current-stage grid is scanned after fixing the past; the figure does not depict a fixed point or a global K^T optimizer.",
                "- Weight boundary: cumulative prefix weights are uncapped and log-stabilized; the one-step ratio cap belongs to the target-policy definition.",
                "- Source traceability: the companion CSV records every semantic box and directed edge; no empirical seed or result artifact is read.",
                "- Image integrity: code-native vector line art; no source image, crop, contrast change, compositing, or raster manipulation.",
                "- Editable SVG text and TrueType PDF embedding are retained; TIFF is 600 dpi and PNG is 240 dpi.",
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_manifest(path: Path, *, work_root: Path, paper_root: Path) -> None:
    work_files = {
        item.name: _file_contract(item)
        for item in sorted(work_root.iterdir())
        if item.is_file() and item.name != path.name
    }
    paper_files = {
        item.name: _file_contract(item)
        for item in sorted(paper_root.iterdir())
        if item.is_file()
    }
    _write_json(
        path,
        {
            "schema_version": 1,
            "protocol": RENDER_PROTOCOL,
            "status": "complete",
            "scientific_seed_reads": 0,
            "canonical_source_modified": False,
            "work_files": work_files,
            "paper_files": paper_files,
        },
    )


def validate_bundle(work_root: Path, paper_root: Path) -> None:
    observed_work = {path.name for path in work_root.iterdir() if path.is_file()}
    observed_paper = {path.name for path in paper_root.iterdir() if path.is_file()}
    if observed_work != WORK_FILES:
        raise RuntimeError(
            f"work bundle differs: expected {sorted(WORK_FILES)}, found {sorted(observed_work)}"
        )
    if observed_paper != PAPER_FILES or any(
        path.suffix.lower() != ".pdf" for path in paper_root.iterdir()
    ):
        raise RuntimeError("paper output must contain exactly one PDF")

    svg = (work_root / f"{FIGURE_STEM}.svg").read_text(encoding="utf-8")
    if "<text" not in svg or "font-family" not in svg or "<image" in svg:
        raise RuntimeError("method schematic SVG must be editable code-native vector art")
    if not re.search(r"Times New Roman|Times|DejaVu Serif", svg):
        raise RuntimeError("method schematic SVG does not retain the paper font family")
    for root in (work_root, paper_root):
        pdf = root / f"{FIGURE_STEM}.pdf"
        if not pdf.read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"malformed PDF: {pdf}")
    if not (work_root / f"{FIGURE_STEM}.png").read_bytes().startswith(b"\x89PNG"):
        raise RuntimeError("malformed PNG preview")
    if (work_root / f"{FIGURE_STEM}.tiff").read_bytes()[:4] not in {
        b"II*\x00",
        b"MM\x00*",
    }:
        raise RuntimeError("malformed TIFF export")

    rows = list(csv.DictReader((work_root / f"{FIGURE_STEM}_source.csv").open()))
    element_ids = {row["element_id"] for row in rows}
    required = {
        "committed_history",
        "target_occupancy",
        "current_state",
        "prediction_sets",
        "indexed_policy",
        "current_action",
        "post_action_score",
        "full_prefix_weight",
        "candidate_surfaces",
        "stagewise_selection",
        "commit_and_advance",
        "action_to_score",
    }
    if not required <= element_ids:
        raise RuntimeError("method schematic source is missing a required semantic element")

    manifest = _read_json(work_root / "render_manifest.json")
    if (
        manifest.get("protocol") != RENDER_PROTOCOL
        or manifest.get("status") != "complete"
        or manifest.get("scientific_seed_reads") != 0
        or manifest.get("canonical_source_modified") is not False
        or set(_mapping(manifest.get("paper_files"), "paper_files")) != PAPER_FILES
    ):
        raise RuntimeError("render manifest contract differs")
    for group, root in (("work_files", work_root), ("paper_files", paper_root)):
        for name, contract in _mapping(manifest[group], group).items():
            if contract != _file_contract(root / name):
                raise RuntimeError(f"render manifest hash differs: {group}/{name}")


def _file_contract(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"mapping required: {label}")
    return value


if __name__ == "__main__":
    main()
