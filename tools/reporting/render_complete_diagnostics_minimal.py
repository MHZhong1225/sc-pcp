"""Render frozen SC-PCP diagnostics and ablations with minimal figure text.

The command is deterministic post-processing.  It validates six immutable
COMPLETE experiment roots, copies their frozen summary statistics into source
CSVs, and publishes four quantitative figures.  It never imports a science
runner, reruns an experiment, or recomputes a bootstrap.

Example
-------
conda run -n ucp python tools/reporting/render_complete_diagnostics_minimal.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import render_formal_mechanism_results as formal  # noqa: E402
from tools import render_theorem_robustness_results as theorem  # noqa: E402


RENDER_PROTOCOL = "complete_diagnostics_minimal_text_v1"

DEFAULT_EXACT_ROOT = ROOT / "results/work/exact_finite_mdp_20260825"
DEFAULT_HORIZON_ROOT = ROOT / "results/work/horizon_overlap_v1"
DEFAULT_RQ6_ROOT = ROOT / "results/work/rq6_ncal_convergence_v1"
DEFAULT_PROPENSITY_ROOT = ROOT / "results/work/propensity_robustness_v1"
DEFAULT_STRICT_ROOT = ROOT / "results/work/strict_split_robustness_v1_20260826"
DEFAULT_PREFIX_ROOT = (
    ROOT / "results/work/controlled_prefix_ablations_confirm20_20260824"
)
DEFAULT_WORK_OUTPUT = (
    ROOT / "results/work/complete_diagnostics_minimal_text_20260830"
)
DEFAULT_PAPER_OUTPUT = (
    ROOT / "results/paper_complete_diagnostics_minimal_text_20260830"
)

EXACT_STEM = "figure_exact_identification_minimal"
THEORY_STEM = "figure_horizon_overlap_ncal_minimal"
ROBUSTNESS_STEM = "figure_propensity_strict_split_minimal"
PREFIX_STEM = "figure_prefix_ablation_minimal"
FIGURE_STEMS = (EXACT_STEM, THEORY_STEM, ROBUSTNESS_STEM, PREFIX_STEM)

EXACT_SOURCE = "exact_identification_source.csv"
THEORY_SOURCE = "horizon_overlap_ncal_source.csv"
ROBUSTNESS_SOURCE = "propensity_strict_split_source.csv"
PREFIX_SOURCE = "prefix_ablation_source.csv"
SOURCE_FILES = {EXACT_SOURCE, THEORY_SOURCE, ROBUSTNESS_SOURCE, PREFIX_SOURCE}
PAPER_FILES = {f"{stem}.pdf" for stem in FIGURE_STEMS}
WORK_FILES = {
    *(
        f"{stem}.{suffix}"
        for stem in FIGURE_STEMS
        for suffix in ("svg", "pdf", "tiff", "png")
    ),
    *SOURCE_FILES,
    "figure_contract.json",
    "figure_qa.md",
    "render_manifest.json",
    "COMPLETE",
}

PREFIX_PROTOCOL = "controlled_prefix_ablation_confirm20_v1"
PREFIX_SOURCE_TREE_SHA256 = (
    "ab1d0972e3311179f3952ae4e1e27f0ccc3759cc316e1ccde4a28aff96f710ac"
)
PREFIX_TREE_SHA256 = (
    "9661f2bc670ced063824ac0c2dd91b7737ac64e5a29c11d711566f37ad957fb7"
)
PREFIX_METADATA_SHA256 = (
    "d7892d7897ff9bb5b9f4d0217ae7c5c5be7dbabc21767d783ad807f8d18e1728"
)
PREFIX_SUMMARY_SHA256 = (
    "0288fbc8edc1ce889f09dd87cafc6122de418747abf80cfdc7af01b838e5a7bd"
)
PREFIX_GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
PREFIX_SEEDS = tuple(range(12_400, 12_440, 2))
PREFIX_METHODS = (
    "SC-PCP",
    "SC-PCP w/o current-action ratio",
    "SC-PCP current-action only",
    "Frozen-policy Prefix-IW",
    "One-step coupled Prefix-IW",
)
PREFIX_LABELS = {
    "SC-PCP": "Full",
    "SC-PCP w/o current-action ratio": "No-current",
    "SC-PCP current-action only": "Current-only",
    "Frozen-policy Prefix-IW": "Frozen",
    "One-step coupled Prefix-IW": "One-step",
}
PREFIX_COLORS = {
    "SC-PCP": "#1976C9",
    "SC-PCP w/o current-action ratio": "#C7659B",
    "SC-PCP current-action only": "#8C6BB1",
    "Frozen-policy Prefix-IW": "#59636E",
    "One-step coupled Prefix-IW": "#D98C00",
}
PREFIX_MARKERS = {
    "SC-PCP": "s",
    "SC-PCP w/o current-action ratio": "P",
    "SC-PCP current-action only": "D",
    "Frozen-policy Prefix-IW": "o",
    "One-step coupled Prefix-IW": "^",
}
PREFIX_LINESTYLES = {
    "SC-PCP": "-",
    "SC-PCP w/o current-action ratio": (0, (5, 2, 1, 2)),
    "SC-PCP current-action only": (0, (2, 2)),
    "Frozen-policy Prefix-IW": "-",
    "One-step coupled Prefix-IW": (0, (4, 2)),
}
PREFIX_SOURCE_FIELDS = (
    "gamma",
    "variant",
    "variant_label",
    "metric",
    "unit",
    "estimate",
    "ci95_lower",
    "ci95_upper",
    "n_seeds",
    "selection_rate",
    "conditioning",
    "source_json_path",
)

SCPCP_BLUE = "#1976C9"
NEUTRAL_DARK = "#4D4D4D"
TEAL = "#117A65"


@dataclass(frozen=True)
class RenderConfig:
    exact_root: Path = DEFAULT_EXACT_ROOT
    horizon_root: Path = DEFAULT_HORIZON_ROOT
    rq6_root: Path = DEFAULT_RQ6_ROOT
    propensity_root: Path = DEFAULT_PROPENSITY_ROOT
    strict_root: Path = DEFAULT_STRICT_ROOT
    prefix_root: Path = DEFAULT_PREFIX_ROOT
    work_output: Path = DEFAULT_WORK_OUTPUT
    paper_output: Path = DEFAULT_PAPER_OUTPUT


@dataclass(frozen=True)
class FrozenArtifacts:
    exact_summary: Mapping[str, Any]
    horizon_summary: Mapping[str, Any]
    rq6_summary: Mapping[str, Any]
    propensity_summary: Mapping[str, Any]
    strict_summary: Mapping[str, Any]
    prefix_summary: Mapping[str, Any]
    input_contracts: Mapping[str, Mapping[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-root", type=Path, default=DEFAULT_EXACT_ROOT)
    parser.add_argument("--horizon-root", type=Path, default=DEFAULT_HORIZON_ROOT)
    parser.add_argument("--rq6-root", type=Path, default=DEFAULT_RQ6_ROOT)
    parser.add_argument(
        "--propensity-root", type=Path, default=DEFAULT_PROPENSITY_ROOT
    )
    parser.add_argument("--strict-root", type=Path, default=DEFAULT_STRICT_ROOT)
    parser.add_argument("--prefix-root", type=Path, default=DEFAULT_PREFIX_ROOT)
    parser.add_argument("--work-output", type=Path, default=DEFAULT_WORK_OUTPUT)
    parser.add_argument("--paper-output", type=Path, default=DEFAULT_PAPER_OUTPUT)
    args = parser.parse_args()
    render_report(
        RenderConfig(
            exact_root=args.exact_root.resolve(),
            horizon_root=args.horizon_root.resolve(),
            rq6_root=args.rq6_root.resolve(),
            propensity_root=args.propensity_root.resolve(),
            strict_root=args.strict_root.resolve(),
            prefix_root=args.prefix_root.resolve(),
            work_output=args.work_output.resolve(),
            paper_output=args.paper_output.resolve(),
        )
    )
    print(args.paper_output.resolve())


def render_report(config: RenderConfig) -> None:
    if config.work_output.exists() or config.paper_output.exists():
        raise FileExistsError("work-output and paper-output must both be fresh")
    if config.work_output == config.paper_output:
        raise ValueError("work-output and paper-output must differ")

    artifacts = load_frozen_artifacts(config)
    exact_rows = formal.build_exact_source_rows(artifacts.exact_summary)
    theory_rows = _retarget_rows(
        theorem.build_theory_source_rows(
            artifacts.horizon_summary,
            artifacts.rq6_summary,
        ),
        THEORY_STEM,
    )
    robustness_rows = _retarget_rows(
        theorem.build_robustness_source_rows(
            artifacts.propensity_summary,
            artifacts.strict_summary,
        ),
        ROBUSTNESS_STEM,
    )
    prefix_rows = build_prefix_source_rows(artifacts.prefix_summary)

    config.work_output.parent.mkdir(parents=True, exist_ok=True)
    config.paper_output.parent.mkdir(parents=True, exist_ok=True)
    staged_work = Path(
        tempfile.mkdtemp(
            prefix=f".{config.work_output.name}-", dir=config.work_output.parent
        )
    )
    staged_paper = Path(
        tempfile.mkdtemp(
            prefix=f".{config.paper_output.name}-", dir=config.paper_output.parent
        )
    )
    try:
        _write_rows(staged_work / EXACT_SOURCE, formal.EXACT_SOURCE_FIELDS, exact_rows)
        _write_rows(staged_work / THEORY_SOURCE, theorem.SOURCE_FIELDS, theory_rows)
        _write_rows(
            staged_work / ROBUSTNESS_SOURCE,
            theorem.SOURCE_FIELDS,
            robustness_rows,
        )
        _write_rows(
            staged_work / PREFIX_SOURCE,
            PREFIX_SOURCE_FIELDS,
            prefix_rows,
        )
        _write_contract(
            staged_work / "figure_contract.json",
            artifacts=artifacts,
            row_counts={
                EXACT_SOURCE: len(exact_rows),
                THEORY_SOURCE: len(theory_rows),
                ROBUSTNESS_SOURCE: len(robustness_rows),
                PREFIX_SOURCE: len(prefix_rows),
            },
            staged_work=staged_work,
        )
        _write_qa(staged_work / "figure_qa.md")

        apply_publication_style()
        figures = {
            EXACT_STEM: render_exact_figure(exact_rows),
            THEORY_STEM: render_theory_figure(theory_rows),
            ROBUSTNESS_STEM: render_robustness_figure(robustness_rows),
            PREFIX_STEM: render_prefix_figure(prefix_rows),
        }
        for stem, figure in figures.items():
            export_figure(
                figure,
                work_stem=staged_work / stem,
                paper_path=staged_paper / f"{stem}.pdf",
                title=_metadata_title(stem),
            )

        _write_render_manifest(
            staged_work / "render_manifest.json",
            work_root=staged_work,
            paper_root=staged_paper,
        )
        _write_complete(staged_work)
        validate_rendered_outputs(staged_work, staged_paper)
        os.replace(staged_work, config.work_output)
        os.replace(staged_paper, config.paper_output)
    except BaseException:
        shutil.rmtree(staged_work, ignore_errors=True)
        shutil.rmtree(staged_paper, ignore_errors=True)
        raise


def load_frozen_artifacts(config: RenderConfig) -> FrozenArtifacts:
    exact_summary, exact_contract = formal.validate_exact_bundle(config.exact_root)
    horizon_summary, horizon_contract = theorem.validate_horizon_bundle(
        config.horizon_root
    )
    rq6_summary, rq6_contract = theorem.validate_rq6_bundle(config.rq6_root)
    propensity_summary, propensity_contract = theorem.validate_propensity_bundle(
        config.propensity_root
    )
    strict_summary, strict_contract = theorem.validate_strict_bundle(config.strict_root)
    prefix_summary, prefix_contract = validate_prefix_bundle(config.prefix_root)
    return FrozenArtifacts(
        exact_summary=exact_summary,
        horizon_summary=horizon_summary,
        rq6_summary=rq6_summary,
        propensity_summary=propensity_summary,
        strict_summary=strict_summary,
        prefix_summary=prefix_summary,
        input_contracts={
            "exact_identification": exact_contract,
            "horizon_overlap": horizon_contract,
            "ncal_convergence": rq6_contract,
            "propensity_robustness": propensity_contract,
            "strict_split": strict_contract,
            "prefix_ablation": prefix_contract,
        },
    )


def validate_prefix_bundle(
    root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    expected_files = {
        "COMPLETE",
        "metadata.json",
        "summary.json",
        *(f"seed_{seed}.json" for seed in PREFIX_SEEDS),
    }
    observed = {path.name for path in root.iterdir() if path.is_file()}
    if observed != expected_files or any(path.is_symlink() for path in root.iterdir()):
        raise RuntimeError("prefix-ablation root entries differ")
    if (root / "COMPLETE").read_text(encoding="utf-8") != "\n":
        raise RuntimeError("prefix-ablation COMPLETE marker differs")
    if _flat_tree_sha256(root, expected_files) != PREFIX_TREE_SHA256:
        raise RuntimeError("prefix-ablation frozen tree hash differs")
    if _file_sha256(root / "metadata.json") != PREFIX_METADATA_SHA256:
        raise RuntimeError("prefix-ablation metadata hash differs")
    if _file_sha256(root / "summary.json") != PREFIX_SUMMARY_SHA256:
        raise RuntimeError("prefix-ablation summary hash differs")

    metadata = _read_json(root / "metadata.json")
    summary = _read_json(root / "summary.json")
    if (
        metadata.get("protocol") != PREFIX_PROTOCOL
        or metadata.get("role") != "post_confirmatory_explanatory_ablation"
        or metadata.get("source_tree_sha256") != PREFIX_SOURCE_TREE_SHA256
        or metadata.get("canonical_selector_mutation_permitted") is not False
        or tuple(float(value) for value in metadata.get("gammas", ()))
        != PREFIX_GAMMAS
        or tuple(int(value) for value in metadata.get("seeds", ())) != PREFIX_SEEDS
        or tuple(metadata.get("methods", ())) != PREFIX_METHODS
    ):
        raise RuntimeError("prefix-ablation metadata contract differs")
    _validate_prefix_summary(summary)
    return summary, {
        "protocol": PREFIX_PROTOCOL,
        "source_tree_sha256": PREFIX_SOURCE_TREE_SHA256,
        "tree_sha256": PREFIX_TREE_SHA256,
        "metadata_sha256": PREFIX_METADATA_SHA256,
        "summary_sha256": PREFIX_SUMMARY_SHA256,
        "input_path": _project_path(root),
    }


def _validate_prefix_summary(summary: Mapping[str, Any]) -> None:
    if (
        summary.get("protocol") != PREFIX_PROTOCOL
        or summary.get("role") != "post_confirmatory_explanatory_ablation"
        or summary.get("canonical_selector_mutation_permitted") is not False
        or summary.get("primary_metric")
        != "min_t mean_seed(target_coverage_seed_t)"
        or tuple(int(value) for value in summary.get("seeds", ())) != PREFIX_SEEDS
    ):
        raise RuntimeError("prefix-ablation summary identity differs")
    bootstrap = _mapping(summary.get("bootstrap"), "prefix bootstrap")
    if bootstrap.get("resamples") != 10_000 or bootstrap.get("unit") != "complete_seed_vector":
        raise RuntimeError("prefix-ablation bootstrap contract differs")
    aggregates = summary.get("aggregates")
    if not isinstance(aggregates, list) or tuple(
        float(cell.get("gamma")) for cell in aggregates if isinstance(cell, Mapping)
    ) != PREFIX_GAMMAS:
        raise RuntimeError("prefix-ablation gamma grid differs")
    for aggregate in aggregates:
        cell = _mapping(aggregate, "prefix aggregate")
        if int(cell.get("n_seeds", -1)) != len(PREFIX_SEEDS):
            raise RuntimeError("prefix-ablation seed count differs")
        methods = _mapping(cell.get("methods"), "prefix methods")
        paired = _mapping(cell.get("paired_vs_full_prefix"), "prefix paired contrasts")
        if set(methods) != set(PREFIX_METHODS) or set(paired) != set(PREFIX_METHODS[1:]):
            raise RuntimeError("prefix-ablation method grid differs")
        for method in PREFIX_METHODS:
            result = _mapping(methods[method], f"prefix result/{method}")
            if (
                float(result.get("selection_rate", float("nan"))) != 1.0
                or result.get("unavailable_seeds") != []
            ):
                raise RuntimeError("prefix-ablation availability differs")
            _finite_number(result.get("target_marginal_worst_coverage"), "prefix WSC")
            _finite_interval(
                result.get("target_marginal_worst_coverage_ci95"), "prefix WSC CI"
            )
            _finite_number(result.get("mean_target_normalized_width"), "prefix width")
            _finite_number(
                result.get("late_target_q90_to_radius_geometric_ratio"),
                "prefix response ratio",
            )
            _finite_interval(
                result.get("late_target_q90_to_radius_geometric_ratio_ci95"),
                "prefix response ratio CI",
            )
            _finite_number(
                result.get("minimum_selection_ess_fraction"), "prefix ESS fraction"
            )
        for method in PREFIX_METHODS[1:]:
            contrast = _mapping(paired[method], f"prefix paired/{method}")
            if contrast.get("available") is not True:
                raise RuntimeError("prefix-ablation paired contrast is unavailable")
            _finite_interval(
                contrast.get("marginal_worst_coverage_difference_ci95"),
                "prefix paired WSC CI",
            )
            _finite_interval(
                contrast.get("geometric_width_ratio_ci95"),
                "prefix paired width CI",
            )


def build_prefix_source_rows(
    summary: Mapping[str, Any],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metrics = (
        (
            "WSC",
            "%",
            "target_marginal_worst_coverage",
            "target_marginal_worst_coverage_ci95",
            100.0,
            "successful selection",
        ),
        (
            "Mean normalized width",
            "normalized width",
            "mean_target_normalized_width",
            None,
            1.0,
            "successful selection",
        ),
        (
            "Late target Q90/radius ratio",
            "ratio",
            "late_target_q90_to_radius_geometric_ratio",
            "late_target_q90_to_radius_geometric_ratio_ci95",
            1.0,
            "late stages 4:11",
        ),
        (
            "Minimum selection ESS/n",
            "%",
            "minimum_selection_ess_fraction",
            None,
            100.0,
            "successful selection",
        ),
    )
    for aggregate in summary["aggregates"]:
        gamma = float(aggregate["gamma"])
        for method in PREFIX_METHODS:
            result = aggregate["methods"][method]
            for label, unit, field, interval_field, scale, conditioning in metrics:
                interval = result[interval_field] if interval_field else None
                rows.append(
                    {
                        "gamma": gamma,
                        "variant": method,
                        "variant_label": PREFIX_LABELS[method],
                        "metric": label,
                        "unit": unit,
                        "estimate": scale * float(result[field]),
                        "ci95_lower": "" if interval is None else scale * float(interval[0]),
                        "ci95_upper": "" if interval is None else scale * float(interval[1]),
                        "n_seeds": int(aggregate["n_seeds"]),
                        "selection_rate": float(result["selection_rate"]),
                        "conditioning": conditioning,
                        "source_json_path": (
                            f'aggregates[gamma={gamma:g}]["methods"]["{method}"]'
                            f'["{field}"]'
                        ),
                    }
                )
    return rows


def _retarget_rows(
    rows: Sequence[Mapping[str, object]], stem: str
) -> list[dict[str, object]]:
    return [{**row, "figure": stem} for row in rows]


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 6.4,
            "axes.labelsize": 6.7,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "svg.hashsalt": "scpcp-complete-diagnostics-minimal-v1",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def render_exact_figure(rows: Sequence[Mapping[str, object]]) -> plt.Figure:
    by_key = {(row["mechanism"], row["estimator"]): row for row in rows}
    matrix = np.asarray(
        [
            [
                float(by_key[(mechanism, estimator)]["mean_maximum_absolute_bias"])
                for mechanism in formal.MECHANISMS
            ]
            for estimator in formal.ESTIMATORS
        ],
        dtype=np.float64,
    )
    cmap = LinearSegmentedColormap.from_list(
        "identification_bias_minimal",
        ("#F7FBFF", "#C7DCEF", "#78A8CF", "#315F87"),
    )
    figure, axis = plt.subplots(figsize=(7.20, 3.0), constrained_layout=True)
    image = axis.imshow(matrix, cmap=cmap, vmin=0.0, vmax=0.40, aspect="auto")
    axis.set_xticks(
        range(len(formal.MECHANISMS)),
        [formal.MECHANISM_LABELS[value] for value in formal.MECHANISMS],
    )
    axis.set_yticks(
        range(len(formal.ESTIMATORS)),
        [formal.ESTIMATOR_LABELS[value] for value in formal.ESTIMATORS],
    )
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.set_xlabel("Feedback mechanism")
    axis.set_ylabel("Transport diagnostic")
    axis.tick_params(length=0, pad=4)
    for estimator_index, estimator in enumerate(formal.ESTIMATORS):
        for mechanism_index, mechanism in enumerate(formal.MECHANISMS):
            if (mechanism, estimator) in formal.IDENTIFICATION_CORRECT:
                axis.add_patch(
                    Rectangle(
                        (mechanism_index - 0.49, estimator_index - 0.49),
                        0.98,
                        0.98,
                        fill=False,
                        edgecolor=TEAL,
                        linewidth=1.5,
                    )
                )
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.035)
    colorbar.set_label("Max. abs. bias")
    colorbar.ax.tick_params(labelsize=5.5, width=0.55, length=2)
    axis.legend(
        handles=[Patch(facecolor="none", edgecolor=TEAL, linewidth=1.5, label="Exact")],
        loc="lower right",
    )
    _add_panel_label(axis, "a")
    _validate_minimal_text(figure, allowed_axis_text={"a"})
    return figure


def render_theory_figure(rows: Sequence[Mapping[str, object]]) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(7.20, 4.85), constrained_layout=True)
    heatmaps = (
        ("a", "RdBu_r", -2.0, 2.0, r"$\Delta C$ (pp)"),
        ("b", "Blues", 0.0, 100.0, "ESS/n (%)"),
        ("c", "Oranges", 0.0, 4.0, "Sup. error (pp)"),
    )
    for column, (panel, cmap_name, lower, upper, colorbar_label) in enumerate(heatmaps):
        axis = axes[0, column]
        matrix = _heatmap_matrix(rows, panel)
        norm: mpl.colors.Normalize
        if panel == "a":
            norm = mpl.colors.TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
        else:
            norm = mpl.colors.Normalize(vmin=lower, vmax=upper)
        image = axis.imshow(matrix, cmap=cmap_name, norm=norm, aspect="auto")
        axis.set_xticks(range(len(theorem.NOMINAL_POLICY_TVS)))
        axis.set_xticklabels(("0", ".025", ".05", ".10", ".15"))
        axis.set_yticks(range(len(theorem.HORIZONS)))
        if column == 0:
            axis.set_yticklabels(theorem.HORIZONS)
            axis.set_ylabel("Horizon $T$")
        else:
            axis.set_yticklabels([])
        axis.set_xlabel("Policy TV")
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
        colorbar.set_label(colorbar_label)
        colorbar.ax.tick_params(labelsize=5.2, width=0.55, length=2)

    convergence = (
        ("d", "Max. error (pp)", None),
        ("e", "WSC (%)", 90.0),
        ("f", "Width", None),
    )
    for axis, (panel, ylabel, reference) in zip(axes[1], convergence):
        _plot_convergence(axis, _panel_rows(rows, panel), ylabel, reference)
    for label, axis in zip("abcdef", axes.ravel()):
        _add_panel_label(axis, label)
    _validate_minimal_text(figure, allowed_axis_text=set("abcdef"))
    return figure


def render_robustness_figure(rows: Sequence[Mapping[str, object]]) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(7.20, 4.45), constrained_layout=True)
    arm_panels = (
        ("a", "MAE", None),
        ("b", "WSC (%)", 90.0),
        ("c", "Min. ESS/n (%)", None),
        ("d", "Policy TV (%)", None),
    )
    for axis, (panel, ylabel, reference) in zip(
        (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]), arm_panels
    ):
        _plot_arm(axis, _panel_rows(rows, panel), ylabel, reference)
    _plot_forest(
        axes[1, 1],
        _panel_rows(rows, "e"),
        "Strict − canonical WSC (pp)",
    )
    _plot_forest(
        axes[1, 2],
        _panel_rows(rows, "f"),
        "Width change (%)",
    )
    for label, axis in zip("abcdef", axes.ravel()):
        _add_panel_label(axis, label)
    _validate_minimal_text(figure, allowed_axis_text=set("abcdef"))
    return figure


def render_prefix_figure(rows: Sequence[Mapping[str, object]]) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(7.20, 4.45))
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.82, wspace=0.25, hspace=0.34)
    metrics = (
        ("WSC", "WSC (%)", 90.0),
        ("Mean normalized width", "Width", None),
        ("Late target Q90/radius ratio", "Q90/radius", 1.0),
        ("Minimum selection ESS/n", "Min. ESS/n (%)", None),
    )
    for axis, (metric, ylabel, reference) in zip(axes.ravel(), metrics):
        _plot_prefix_metric(axis, rows, metric, ylabel, reference)
    handles = [
        Line2D(
            [0],
            [0],
            color=PREFIX_COLORS[method],
            marker=PREFIX_MARKERS[method],
            linestyle=PREFIX_LINESTYLES[method],
            linewidth=1.0 if method == "SC-PCP" else 0.78,
            markersize=3.2,
            label=PREFIX_LABELS[method],
        )
        for method in PREFIX_METHODS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        columnspacing=0.9,
        handlelength=1.6,
        handletextpad=0.3,
    )
    for label, axis in zip("abcd", axes.ravel()):
        _add_panel_label(axis, label)
    _validate_minimal_text(figure, allowed_axis_text=set("abcd"))
    return figure


def _heatmap_matrix(
    rows: Sequence[Mapping[str, object]], panel: str
) -> np.ndarray:
    lookup = {
        (int(row["horizon"]), float(row["nominal_policy_tv"])): float(row["estimate"])
        for row in _panel_rows(rows, panel)
    }
    return np.asarray(
        [
            [lookup[(horizon, tv)] for tv in theorem.NOMINAL_POLICY_TVS]
            for horizon in theorem.HORIZONS
        ],
        dtype=np.float64,
    )


def _plot_convergence(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    ylabel: str,
    reference: float | None,
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["n_calibration"]))
    x = np.asarray([int(row["n_calibration"]) for row in ordered])
    point = np.asarray([float(row["estimate"]) for row in ordered])
    lower = np.asarray([float(row["ci95_lower"]) for row in ordered])
    upper = np.asarray([float(row["ci95_upper"]) for row in ordered])
    axis.errorbar(
        x,
        point,
        yerr=np.vstack((point - lower, upper - point)),
        color=SCPCP_BLUE,
        marker="s",
        markersize=3.2,
        linewidth=1.05,
        capsize=1.8,
    )
    if reference is not None:
        axis.axhline(reference, color=NEUTRAL_DARK, linestyle=(0, (3, 2)), linewidth=0.75)
    axis.set_xscale("log")
    axis.set_xticks(x)
    axis.set_xticklabels(("250", "500", "1k", "2k", "5k", "10k"))
    axis.set_xlabel(r"$n_{\rm cal}$")
    axis.set_ylabel(ylabel)
    axis.tick_params(width=0.6, length=2.2)
    axis.margins(x=0.04, y=0.13)


def _plot_arm(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    ylabel: str,
    reference: float | None,
) -> None:
    by_arm = {str(row["arm"]): row for row in rows}
    x = np.arange(len(theorem.PROPENSITY_ARMS))
    for index, arm in enumerate(theorem.PROPENSITY_ARMS):
        row = by_arm[arm]
        point = float(row["estimate"])
        lower = float(row["ci95_lower"])
        upper = float(row["ci95_upper"])
        axis.errorbar(
            index,
            point,
            yerr=[[point - lower], [upper - point]],
            color=theorem.ARM_COLORS[arm],
            marker=theorem.ARM_MARKERS[arm],
            markersize=3.8,
            linewidth=0.9,
            capsize=2.0,
        )
    if reference is not None:
        axis.axhline(reference, color=NEUTRAL_DARK, linestyle=(0, (3, 2)), linewidth=0.75)
    axis.set_xticks(x)
    axis.set_xticklabels(("Oracle", "Correct", "Reduced"), rotation=15)
    axis.set_ylabel(ylabel)
    axis.tick_params(width=0.6, length=2.2)
    axis.margins(x=0.20, y=0.16)


def _plot_forest(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    xlabel: str,
) -> None:
    by_setting = {str(row["setting"]): row for row in rows}
    y = np.arange(len(theorem.STRICT_SETTINGS))[::-1]
    for y_value, setting in zip(y, theorem.STRICT_SETTINGS):
        row = by_setting[setting]
        point = float(row["estimate"])
        lower = float(row["ci95_lower"])
        upper = float(row["ci95_upper"])
        axis.plot([lower, upper], [y_value, y_value], color=SCPCP_BLUE, linewidth=1.0)
        axis.plot(
            point,
            y_value,
            color=SCPCP_BLUE,
            marker=theorem.SETTING_MARKERS[setting],
            markersize=3.8,
        )
    axis.axvline(0.0, color=NEUTRAL_DARK, linestyle=(0, (3, 2)), linewidth=0.75)
    axis.set_yticks(y)
    axis.set_yticklabels(
        [theorem.SETTING_LABELS[setting] for setting in theorem.STRICT_SETTINGS]
    )
    axis.set_xlabel(xlabel)
    axis.tick_params(width=0.6, length=2.2)
    axis.margins(x=0.12, y=0.30)


def _plot_prefix_metric(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    metric: str,
    ylabel: str,
    reference: float | None,
) -> None:
    metric_rows = [row for row in rows if row["metric"] == metric]
    for method in PREFIX_METHODS:
        ordered = sorted(
            (row for row in metric_rows if row["variant"] == method),
            key=lambda row: float(row["gamma"]),
        )
        x = np.asarray([float(row["gamma"]) for row in ordered])
        point = np.asarray([float(row["estimate"]) for row in ordered])
        lower_values = [row["ci95_lower"] for row in ordered]
        if all(value != "" for value in lower_values):
            lower = np.asarray([float(value) for value in lower_values])
            upper = np.asarray([float(row["ci95_upper"]) for row in ordered])
            axis.errorbar(
                x,
                point,
                yerr=np.vstack((point - lower, upper - point)),
                color=PREFIX_COLORS[method],
                marker=PREFIX_MARKERS[method],
                linestyle=PREFIX_LINESTYLES[method],
                markersize=3.0,
                linewidth=1.0 if method == "SC-PCP" else 0.72,
                elinewidth=0.48,
                capsize=1.2,
            )
        else:
            axis.plot(
                x,
                point,
                color=PREFIX_COLORS[method],
                marker=PREFIX_MARKERS[method],
                linestyle=PREFIX_LINESTYLES[method],
                markersize=3.0,
                linewidth=1.0 if method == "SC-PCP" else 0.72,
            )
    if reference is not None:
        axis.axhline(reference, color=NEUTRAL_DARK, linestyle=(0, (3, 2)), linewidth=0.75)
    axis.set_xticks(PREFIX_GAMMAS, ("−4", "−2", "0", "+2", "+4"))
    axis.set_xlabel(r"$\gamma$")
    axis.set_ylabel(ylabel)
    axis.tick_params(width=0.6, length=2.2)
    axis.margins(x=0.04, y=0.12)


def _panel_rows(
    rows: Sequence[Mapping[str, object]], panel: str
) -> list[Mapping[str, object]]:
    selected = [row for row in rows if row.get("panel") == panel]
    if not selected:
        raise RuntimeError(f"source data contains no panel {panel}")
    return selected


def _add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.15,
        1.03,
        label,
        transform=axis.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _validate_minimal_text(
    figure: plt.Figure, *, allowed_axis_text: set[str]
) -> None:
    if figure._suptitle is not None or figure.texts:
        raise RuntimeError("minimal-text figure contains figure-level prose")
    for axis in figure.axes:
        if axis.get_title():
            raise RuntimeError("minimal-text figure contains a panel title")
        unexpected = {
            text.get_text() for text in axis.texts if text.get_text() not in allowed_axis_text
        }
        if unexpected:
            raise RuntimeError(f"minimal-text figure contains annotations: {sorted(unexpected)}")


def export_figure(
    figure: plt.Figure,
    *,
    work_stem: Path,
    paper_path: Path,
    title: str,
) -> None:
    creator = "SC-PCP complete diagnostics minimal-text renderer"
    figure.savefig(
        work_stem.with_suffix(".svg"),
        format="svg",
        bbox_inches="tight",
        metadata={"Title": title, "Creator": creator, "Date": None},
    )
    figure.savefig(
        work_stem.with_suffix(".pdf"),
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Creator": creator,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        work_stem.with_suffix(".tiff"),
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(
        work_stem.with_suffix(".png"),
        format="png",
        dpi=240,
        bbox_inches="tight",
        metadata={"Software": creator},
    )
    shutil.copyfile(work_stem.with_suffix(".pdf"), paper_path)
    plt.close(figure)


def _metadata_title(stem: str) -> str:
    return {
        EXACT_STEM: "Exact committed-prefix identification",
        THEORY_STEM: "Horizon overlap and calibration-size diagnostics",
        ROBUSTNESS_STEM: "Propensity and strict-split diagnostics",
        PREFIX_STEM: "Committed-prefix structural ablations",
    }[stem]


def _write_contract(
    path: Path,
    *,
    artifacts: FrozenArtifacts,
    row_counts: Mapping[str, int],
    staged_work: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "protocol": RENDER_PROTOCOL,
        "status": "complete",
        "backend": "Python/matplotlib only",
        "scientific_rng_used": False,
        "bootstrap_recomputed": False,
        "archetype": "quantitative_grid",
        "figures": {
            EXACT_STEM: {
                "panels": {"a": "exact finite-MDP identification matrix"},
                "source_data": EXACT_SOURCE,
            },
            THEORY_STEM: {
                "panels": {
                    "a": "coverage shortfall",
                    "b": "selected-prefix ESS",
                    "c": "committed-surface error",
                    "d": "fixed-grid surface recovery",
                    "e": "exact population WSC",
                    "f": "exact population width",
                },
                "source_data": THEORY_SOURCE,
            },
            ROBUSTNESS_STEM: {
                "panels": {
                    "a": "propensity MAE",
                    "b": "fixed-target-law WSC",
                    "c": "fixed-target-law ESS",
                    "d": "target-law drift",
                    "e": "strict-minus-canonical WSC",
                    "f": "strict/canonical width change",
                },
                "source_data": ROBUSTNESS_SOURCE,
            },
            PREFIX_STEM: {
                "panels": {
                    "a": "WSC",
                    "b": "normalized width",
                    "c": "late target-Q90/radius response",
                    "d": "minimum selection ESS",
                },
                "source_data": PREFIX_SOURCE,
                "variants_are_diagnostics_not_baselines": True,
            },
        },
        "visible_text_policy": {
            "allowed": [
                "lowercase panel letters",
                "axis labels",
                "tick labels",
                "matrix row and column labels",
                "short categorical legends",
            ],
            "forbidden": [
                "suptitle",
                "panel prose title",
                "claim sentence",
                "footer",
                "explanatory annotation",
            ],
        },
        "statistics": {
            "intervals": "copied from frozen summaries",
            "primary_coverage_metric": "min_t mean_seed(C_seed,t)",
            "prefix_ablation_bootstrap": "10000 complete-seed-vector resamples, frozen",
            "no_missing_metric_imputation": True,
        },
        "source_data": {
            name: {
                "rows": int(row_counts[name]),
                "sha256": _file_sha256(staged_work / name),
            }
            for name in sorted(SOURCE_FILES)
        },
        "input_contracts": artifacts.input_contracts,
        "claim_boundary": (
            "Diagnostic and explanatory evidence only. No finite-sample, distribution-free, "
            "PAC, data-conditional, clinical, equivalence, or universal-SOTA claim; prefix "
            "variants are not baseline methods and canonical SC-PCP is unchanged."
        ),
        "export_contract": {
            "paper_directory": "fresh caller-selected output root",
            "paper_files": sorted(PAPER_FILES),
            "paper_directory_policy": "PDF only",
            "work_formats": [
                "editable SVG",
                "TrueType PDF",
                "600-dpi TIFF",
                "240-dpi PNG",
            ],
            "font": "Times New Roman with Times/DejaVu Serif fallback",
            "deterministic": True,
        },
    }
    _write_json(path, payload)


def _write_qa(path: Path) -> None:
    lines = [
        "# Complete diagnostic and ablation figure QA",
        "",
        "- Backend: Python/matplotlib only.",
        "- Inputs: six frozen COMPLETE roots; no science runner or bootstrap executed.",
        "- Figures: four quantitative grids with no visible titles, prose, claims, or footers.",
        "- Visible text: panel letters, axes, ticks, matrix labels, and short legends only.",
        "- Exact identification: 500 paired finite-MDP instances; diagnostics are not baselines.",
        "- Horizon-overlap: 200 paired instances; matrix cells are frozen descriptive summaries.",
        "- Calibration-size: 100 problem clusters x 20 logged resamples; frozen cluster bootstrap.",
        "- Propensity: 100 paired problems; fixed-target and drift layers remain distinct.",
        "- Strict split: paired complete-seed-vector contrasts for three frozen settings.",
        "- Prefix ablation: 20 complete seeds at five signed gamma values; variants are diagnostics.",
        "- Missing intervals are left absent and never imputed or recomputed.",
        "- Exports: editable SVG, identical work/paper PDF, 600-dpi TIFF, 240-dpi PNG.",
        "- Typography: Times New Roman with deterministic serif fallbacks.",
        "- Claim boundary: asymptotic per-step marginal SC-PCP only; no finite-sample certificate.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_render_manifest(path: Path, *, work_root: Path, paper_root: Path) -> None:
    observed_paper = {item.name for item in paper_root.iterdir() if item.is_file()}
    if observed_paper != PAPER_FILES:
        raise RuntimeError("paper staging file set differs")
    work_files = {
        item.name: _file_contract(item)
        for item in sorted(work_root.iterdir())
        if item.is_file() and item.name not in {path.name, "COMPLETE"}
    }
    paper_files = {
        item.name: _file_contract(item) for item in sorted(paper_root.iterdir())
    }
    _write_json(
        path,
        {
            "schema_version": 1,
            "protocol": RENDER_PROTOCOL,
            "status": "complete",
            "work_files": work_files,
            "paper_files": paper_files,
        },
    )


def _write_complete(work_root: Path) -> None:
    (work_root / "COMPLETE").write_text(
        f"protocol={RENDER_PROTOCOL}\n"
        f"manifest_sha256={_file_sha256(work_root / 'render_manifest.json')}\n",
        encoding="utf-8",
    )


def validate_rendered_outputs(work_root: Path, paper_root: Path) -> None:
    observed_work = {item.name for item in work_root.iterdir() if item.is_file()}
    observed_paper = {item.name for item in paper_root.iterdir() if item.is_file()}
    if observed_work != WORK_FILES:
        raise RuntimeError("minimal diagnostic work bundle differs")
    if observed_paper != PAPER_FILES or any(
        item.suffix.lower() != ".pdf" for item in paper_root.iterdir()
    ):
        raise RuntimeError("minimal diagnostic paper bundle must be PDF only")

    for stem in FIGURE_STEMS:
        svg_path = work_root / f"{stem}.svg"
        visible_text = _visible_svg_text(svg_path)
        if not visible_text or "Times New Roman" not in svg_path.read_text(encoding="utf-8"):
            raise RuntimeError(f"editable Times SVG contract differs: {stem}")
        if any(
            phrase in visible_text
            for phrase in (
                "Exact population identification across",
                "Coverage remains",
                "Canonical SC-PCP",
                "diagnostics only",
            )
        ):
            raise RuntimeError(f"visible prose leaked into minimal figure: {stem}")
        work_pdf = work_root / f"{stem}.pdf"
        paper_pdf = paper_root / f"{stem}.pdf"
        if not work_pdf.read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"work PDF is malformed: {stem}")
        if _file_sha256(work_pdf) != _file_sha256(paper_pdf):
            raise RuntimeError(f"work and paper PDFs differ: {stem}")
        if not (work_root / f"{stem}.png").read_bytes().startswith(b"\x89PNG"):
            raise RuntimeError(f"PNG is malformed: {stem}")
        with Image.open(work_root / f"{stem}.tiff") as image:
            dpi = image.info.get("dpi")
            if dpi is None or not all(math.isclose(value, 600.0, rel_tol=1e-3) for value in dpi):
                raise RuntimeError(f"TIFF DPI contract differs: {stem}")

    manifest = _read_json(work_root / "render_manifest.json")
    if (
        manifest.get("protocol") != RENDER_PROTOCOL
        or manifest.get("status") != "complete"
        or set(_mapping(manifest.get("work_files"), "manifest work"))
        != WORK_FILES - {"render_manifest.json", "COMPLETE"}
        or set(_mapping(manifest.get("paper_files"), "manifest paper"))
        != PAPER_FILES
    ):
        raise RuntimeError("minimal diagnostic render manifest differs")
    expected_complete = (
        f"protocol={RENDER_PROTOCOL}\n"
        f"manifest_sha256={_file_sha256(work_root / 'render_manifest.json')}\n"
    )
    if (work_root / "COMPLETE").read_text(encoding="utf-8") != expected_complete:
        raise RuntimeError("minimal diagnostic COMPLETE marker differs")
    for group, root in (("work_files", work_root), ("paper_files", paper_root)):
        for name, contract in _mapping(manifest[group], group).items():
            _validate_file_contract(root / name, _mapping(contract, name))


def _write_rows(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, str(path))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _finite_number(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} must be finite")
    return number


def _finite_interval(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise RuntimeError(f"{label} must contain two endpoints")
    lower, upper = (_finite_number(item, label) for item in value)
    if lower > upper:
        raise RuntimeError(f"{label} endpoints are reversed")
    return lower, upper


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat_tree_sha256(root: Path, names: set[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(f"{_file_sha256(root / name)}  {name}\n".encode("utf-8"))
    return digest.hexdigest()


def _project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _file_contract(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _validate_file_contract(path: Path, contract: Mapping[str, Any]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(contract["bytes"])
        or _file_sha256(path) != contract["sha256"]
    ):
        raise RuntimeError(f"rendered file contract differs: {path.name}")


def _visible_svg_text(path: Path) -> str:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    return " ".join(
        "".join(element.itertext())
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "text"
    )


if __name__ == "__main__":
    main()
