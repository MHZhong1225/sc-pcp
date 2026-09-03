"""Render the frozen gamma=-4 controlled stress stage profile.

The formal all-six artifact is immutable input.  This script performs only
deterministic reporting: it validates the stored seed rows, computes pointwise
seed-vector bootstrap intervals with the frozen gamma=-4 bootstrap stream, and
exports a paper figure plus its auditable source bundle.  It does not fit,
calibrate, select, or evaluate any method.

Example
-------
conda run -n ucp python tools/render_controlled_stress_stage_profile.py
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
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RENDER_PROTOCOL = "controlled_stress_stage_profile_render_v1"
CONTROLLED_PROTOCOL = "controlled_performative_six_method_benchmark_v1"
FROZEN_SOURCE_TREE_SHA256 = (
    "7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643"
)
FROZEN_SUMMARY_SHA256 = (
    "d8533ca5db0c6a3943fed1751f4d450846dcbff17df305a33197a105cc474670"
)
FROZEN_CONFIG_SHA256 = (
    "a9023266d72b6aff04ab446a3236097bd24d10dc1f15b504aeb688c0bbbf9979"
)

DEFAULT_INPUT_ROOT = ROOT / "results/work/controlled_six_method_confirm20_20260825"
DEFAULT_WORK_OUTPUT = ROOT / "results/work/controlled_stress_stage_profile_20260826"
DEFAULT_PAPER_OUTPUT = ROOT / "results/paper_controlled_stress_stage_profile_20260826"

FIGURE_STEM = "figure_controlled_stress_stage_profile"
PAPER_FILES = {f"{FIGURE_STEM}.pdf"}
WORK_FILES = {
    f"{FIGURE_STEM}.svg",
    f"{FIGURE_STEM}.tiff",
    f"{FIGURE_STEM}.png",
    f"{FIGURE_STEM}_source_data.csv",
    "analysis.json",
    "figure_qa.md",
    "render_manifest.json",
}

STRESS_GAMMA = -4.0
TARGET_COVERAGE = 0.90
HORIZON = 12
BOOTSTRAP_RESAMPLES = 10_000
SEEDS = tuple(range(91_000, 91_200, 10))
GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
METHODS = ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")
ONLINE_METHODS = {"ACI", "SPCI", "PRC"}

METHOD_COLORS = {
    "Standard CP": "#4D4D4D",
    "ACI": "#D98C00",
    "MFCS": "#8C6BB1",
    "SPCI": "#56A9D8",
    "PRC": "#C7659B",
    "SC-PCP": "#1976C9",
}
METHOD_MARKERS = {
    "Standard CP": "o",
    "ACI": "^",
    "MFCS": "D",
    "SPCI": "v",
    "PRC": "P",
    "SC-PCP": "s",
}
METHOD_LINESTYLES = {
    "Standard CP": "-",
    "ACI": (0, (4, 2)),
    "MFCS": "-",
    "SPCI": (0, (2, 2)),
    "PRC": (0, (5, 2, 1, 2)),
    "SC-PCP": "-",
}

SOURCE_FIELDS = (
    "setting",
    "gamma",
    "gamma_role",
    "method",
    "information_regime",
    "target_adaptation_trajectories_per_seed",
    "stage_zero_based",
    "seed_count",
    "target_coverage",
    "target_coverage_ci95_lower",
    "target_coverage_ci95_upper",
    "coverage_deviation_from_090_pp",
    "coverage_deviation_ci95_lower_pp",
    "coverage_deviation_ci95_upper_pp",
    "target_normalized_width",
    "target_normalized_width_ci95_lower",
    "target_normalized_width_ci95_upper",
    "interval_definition",
    "source_seed_artifact_pattern",
)


@dataclass(frozen=True)
class RenderConfig:
    input_root: Path = DEFAULT_INPUT_ROOT
    work_output: Path = DEFAULT_WORK_OUTPUT
    paper_output: Path = DEFAULT_PAPER_OUTPUT


@dataclass(frozen=True)
class FrozenControlledStress:
    summary: Mapping[str, Any]
    seed_rows: tuple[Mapping[str, Any], ...]
    input_contract: Mapping[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--work-output", type=Path, default=DEFAULT_WORK_OUTPUT)
    parser.add_argument("--paper-output", type=Path, default=DEFAULT_PAPER_OUTPUT)
    args = parser.parse_args()
    config = RenderConfig(
        input_root=args.input_root.resolve(),
        work_output=args.work_output.resolve(),
        paper_output=args.paper_output.resolve(),
    )
    render_report(config)
    print(config.paper_output)


def render_report(config: RenderConfig) -> None:
    """Validate the frozen artifact and atomically publish one figure bundle."""

    if config.work_output.exists() or config.paper_output.exists():
        raise FileExistsError("work-output and paper-output must both be fresh")
    if config.work_output == config.paper_output:
        raise ValueError("work-output and paper-output must be different directories")

    artifact = load_frozen_controlled_stress(config.input_root)
    source_rows = build_source_rows(artifact.seed_rows, artifact.summary)
    hero = build_hero_metrics(source_rows, artifact.summary)

    config.work_output.parent.mkdir(parents=True, exist_ok=True)
    config.paper_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_work = Path(
        tempfile.mkdtemp(
            prefix=f".{config.work_output.name}-", dir=config.work_output.parent
        )
    )
    temporary_paper = Path(
        tempfile.mkdtemp(
            prefix=f".{config.paper_output.name}-", dir=config.paper_output.parent
        )
    )
    try:
        source_path = temporary_work / f"{FIGURE_STEM}_source_data.csv"
        write_source_csv(source_path, source_rows)
        _write_analysis(
            temporary_work / "analysis.json",
            config=config,
            artifact=artifact,
            source_rows=source_rows,
            source_path=source_path,
            hero=hero,
        )
        _write_qa_notes(temporary_work / "figure_qa.md", hero=hero)
        apply_publication_style()
        export_figure(
            render_figure(source_rows, hero),
            work_stem=temporary_work / FIGURE_STEM,
            paper_path=temporary_paper / f"{FIGURE_STEM}.pdf",
        )
        _write_render_manifest(
            temporary_work / "render_manifest.json",
            work_root=temporary_work,
            paper_root=temporary_paper,
        )
        validate_rendered_outputs(temporary_work, temporary_paper)
        os.replace(temporary_work, config.work_output)
        os.replace(temporary_paper, config.paper_output)
    except BaseException:
        shutil.rmtree(temporary_work, ignore_errors=True)
        shutil.rmtree(temporary_paper, ignore_errors=True)
        raise


def load_frozen_controlled_stress(root: Path) -> FrozenControlledStress:
    seed_names = {f"seed_{seed:05d}.json" for seed in SEEDS}
    _require_exact_root_entries(
        root, seed_names | {"metadata.json", "summary.json", "COMPLETE"}
    )
    if (root / "COMPLETE").read_text(encoding="utf-8") != "\n":
        raise RuntimeError("controlled COMPLETE marker differs")
    metadata = _read_json_mapping(root / "metadata.json")
    config_contract = _mapping(metadata.get("config_contract"), "config contract")
    if (
        metadata.get("protocol") != CONTROLLED_PROTOCOL
        or metadata.get("role") != "fresh_confirmatory_canonical_baseline_comparison"
        or metadata.get("source_tree_sha256") != FROZEN_SOURCE_TREE_SHA256
        or tuple(metadata.get("methods", ())) != METHODS
        or tuple(metadata.get("seeds", ())) != SEEDS
        or tuple(float(value) for value in metadata.get("gammas", ())) != GAMMAS
        or metadata.get("calibration_trajectories") != 3_000
        or metadata.get("grid_trajectories") != 1_000
        or metadata.get("reference_trajectories") != 20_000
        or metadata.get("guarantee_scope") != "asymptotic_per_step_marginal"
        or metadata.get("canonical_selector_mutation_permitted") is not False
        or config_contract.get("active_config_sha256") != FROZEN_CONFIG_SHA256
    ):
        raise RuntimeError("controlled metadata contract differs")
    if _file_sha256(root / "summary.json") != FROZEN_SUMMARY_SHA256:
        raise RuntimeError("controlled frozen summary hash differs")
    summary = _read_json_mapping(root / "summary.json")
    _validate_summary(summary)

    seed_rows: list[Mapping[str, Any]] = []
    seed_hashes: dict[str, str] = {}
    for seed in SEEDS:
        path = root / f"seed_{seed:05d}.json"
        payload = _read_json_mapping(path)
        if (
            payload.get("seed") != seed
            or payload.get("protocol") != CONTROLLED_PROTOCOL
            or payload.get("source_tree_sha256") != FROZEN_SOURCE_TREE_SHA256
            or payload.get("active_config_sha256") != FROZEN_CONFIG_SHA256
            or tuple(payload.get("methods", ())) != METHODS
        ):
            raise RuntimeError(f"controlled seed provenance differs: {path.name}")
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != len(GAMMAS):
            raise RuntimeError(f"controlled seed row count differs: {path.name}")
        stress = [row for row in rows if float(row.get("gamma")) == STRESS_GAMMA]
        if len(stress) != 1:
            raise RuntimeError(f"gamma=-4 row differs: {path.name}")
        _validate_seed_row(stress[0], seed=seed)
        seed_rows.append(stress[0])
        seed_hashes[path.name] = _file_sha256(path)

    _validate_recovery(summary, seed_rows)
    return FrozenControlledStress(
        summary=summary,
        seed_rows=tuple(seed_rows),
        input_contract={
            "protocol": CONTROLLED_PROTOCOL,
            "source_tree_sha256": FROZEN_SOURCE_TREE_SHA256,
            "active_config_sha256": FROZEN_CONFIG_SHA256,
            "summary_sha256": FROZEN_SUMMARY_SHA256,
            "seed_artifact_count": len(seed_hashes),
            "seed_artifact_hashes_sha256": _canonical_sha256(seed_hashes),
            "input_path": _project_path(root),
        },
    )


def _validate_summary(summary: Mapping[str, Any]) -> None:
    if (
        summary.get("protocol") != CONTROLLED_PROTOCOL
        or summary.get("role") != "fresh_confirmatory_canonical_baseline_comparison"
        or tuple(summary.get("methods", ())) != METHODS
        or tuple(summary.get("seeds", ())) != SEEDS
        or summary.get("primary_metric")
        != "min_t mean_seed(target_coverage_seed_t)"
        or summary.get("coverage_conditioning") != "successful_selection"
        or summary.get("selection_rate_denominator") != "all_prespecified_seeds"
    ):
        raise RuntimeError("controlled summary contract differs")
    bootstrap = _mapping(summary.get("bootstrap"), "bootstrap")
    if (
        bootstrap.get("resamples") != BOOTSTRAP_RESAMPLES
        or f"{STRESS_GAMMA:g}" not in bootstrap.get("gamma_seeds", {})
    ):
        raise RuntimeError("controlled bootstrap contract differs")
    stress = _stress_aggregate(summary)
    if stress.get("n_seeds") != len(SEEDS) or set(stress.get("methods", {})) != set(
        METHODS
    ):
        raise RuntimeError("gamma=-4 aggregate contract differs")
    for method in METHODS:
        cell = _mapping(stress["methods"][method], method)
        coverage = _finite_vector(
            cell.get("target_coverage_by_stage"),
            length=HORIZON,
            label=f"{method}/target coverage",
        )
        for field in (
            "target_marginal_worst_coverage",
            "target_mean_coverage",
            "mean_target_normalized_width",
            "selection_rate",
        ):
            _finite_number(cell.get(field), f"{method}/{field}")
        if cell.get("selected_seeds") != 20 or cell.get("total_seeds") != 20:
            raise RuntimeError("gamma=-4 requires complete 20/20 selection")
        if not math.isclose(
            float(coverage.min()),
            float(cell["target_marginal_worst_coverage"]),
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise RuntimeError("stored WSC differs from min stage-mean coverage")
    paired = _mapping(stress.get("paired_scpcp_comparisons"), "paired comparisons")
    if set(paired) != set(METHODS) - {"SC-PCP"}:
        raise RuntimeError("paired comparison set differs")


def _validate_seed_row(row: Mapping[str, Any], *, seed: int) -> None:
    if (
        row.get("seed") != seed
        or float(row.get("gamma")) != STRESS_GAMMA
        or set(row.get("methods", {})) != set(METHODS)
    ):
        raise RuntimeError("controlled stress seed/gamma/method contract differs")
    for method in METHODS:
        values = _mapping(row["methods"][method], f"seed {seed}/{method}")
        if values.get("selection_available") is not True:
            raise RuntimeError("formal stress artifact must select all methods")
        for field in ("target_coverage", "target_normalized_width"):
            vector = _finite_vector(
                values.get(field), length=HORIZON, label=f"{method}/{field}"
            )
            if field == "target_normalized_width" and np.any(vector <= 0.0):
                raise RuntimeError("normalized widths must be positive")


def _validate_recovery(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    if tuple(int(row["seed"]) for row in rows) != SEEDS:
        raise RuntimeError("stress seed order differs")
    stress = _stress_aggregate(summary)
    for method in METHODS:
        coverage = np.asarray(
            [row["methods"][method]["target_coverage"] for row in rows],
            dtype=np.float64,
        )
        width = np.asarray(
            [row["methods"][method]["target_normalized_width"] for row in rows],
            dtype=np.float64,
        )
        stored = stress["methods"][method]
        expected_coverage = np.asarray(
            stored["target_coverage_by_stage"], dtype=np.float64
        )
        if not np.allclose(
            coverage.mean(axis=0), expected_coverage, atol=1e-14, rtol=0.0
        ):
            raise RuntimeError(f"stage coverage recovery differs for {method}")
        if not math.isclose(
            float(width.mean()),
            float(stored["mean_target_normalized_width"]),
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise RuntimeError(f"width recovery differs for {method}")


def build_source_rows(
    seed_rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> list[dict[str, object]]:
    if len(seed_rows) != len(SEEDS):
        raise RuntimeError("stage-profile source requires exactly 20 seed rows")
    stress = _stress_aggregate(summary)
    bootstrap_seed = int(stress["bootstrap_seed"])
    rng = np.random.default_rng(bootstrap_seed)
    uniforms = rng.random(size=(BOOTSTRAP_RESAMPLES, len(SEEDS)))
    indices = np.floor(uniforms * len(SEEDS)).astype(np.int64)
    rows: list[dict[str, object]] = []
    for method in METHODS:
        coverage = np.asarray(
            [row["methods"][method]["target_coverage"] for row in seed_rows],
            dtype=np.float64,
        )
        width = np.asarray(
            [row["methods"][method]["target_normalized_width"] for row in seed_rows],
            dtype=np.float64,
        )
        coverage_draws = coverage[indices].mean(axis=1)
        width_draws = width[indices].mean(axis=1)
        coverage_mean = coverage.mean(axis=0)
        width_mean = width.mean(axis=0)
        coverage_lower = np.quantile(coverage_draws, 0.025, axis=0)
        coverage_upper = np.quantile(coverage_draws, 0.975, axis=0)
        width_lower = np.quantile(width_draws, 0.025, axis=0)
        width_upper = np.quantile(width_draws, 0.975, axis=0)
        adaptation = int(
            stress["methods"][method]["target_adaptation_trajectories_per_seed"]
        )
        for stage in range(HORIZON):
            rows.append(
                {
                    "setting": "controlled_semi_synthetic_stress",
                    "gamma": STRESS_GAMMA,
                    "gamma_role": (
                        "protocol_stress_default_displayed_hero_not_statistical_primary"
                    ),
                    "method": method,
                    "information_regime": (
                        "on_policy_adaptation"
                        if method in ONLINE_METHODS
                        else "offline_logged_data"
                    ),
                    "target_adaptation_trajectories_per_seed": adaptation,
                    "stage_zero_based": stage,
                    "seed_count": len(SEEDS),
                    "target_coverage": float(coverage_mean[stage]),
                    "target_coverage_ci95_lower": float(coverage_lower[stage]),
                    "target_coverage_ci95_upper": float(coverage_upper[stage]),
                    "coverage_deviation_from_090_pp": float(
                        (coverage_mean[stage] - TARGET_COVERAGE) * 100.0
                    ),
                    "coverage_deviation_ci95_lower_pp": float(
                        (coverage_lower[stage] - TARGET_COVERAGE) * 100.0
                    ),
                    "coverage_deviation_ci95_upper_pp": float(
                        (coverage_upper[stage] - TARGET_COVERAGE) * 100.0
                    ),
                    "target_normalized_width": float(width_mean[stage]),
                    "target_normalized_width_ci95_lower": float(width_lower[stage]),
                    "target_normalized_width_ci95_upper": float(width_upper[stage]),
                    "interval_definition": (
                        "pointwise 95% percentile bootstrap over complete seed-stage "
                        "vectors; frozen gamma=-4 bootstrap seed; 10000 resamples"
                    ),
                    "source_seed_artifact_pattern": (
                        "results/work/controlled_six_method_confirm20_20260825/"
                        "seed_91xxx.json"
                    ),
                }
            )
    return rows


def build_hero_metrics(
    rows: Sequence[Mapping[str, object]], summary: Mapping[str, Any]
) -> dict[str, object]:
    stress = _stress_aggregate(summary)
    method_cells = stress["methods"]
    paired_standard = stress["paired_scpcp_comparisons"]["Standard CP"]
    metrics = {
        "standard_wsc": float(
            method_cells["Standard CP"]["target_marginal_worst_coverage"]
        ),
        "scpcp_wsc": float(method_cells["SC-PCP"]["target_marginal_worst_coverage"]),
        "mfcs_wsc": float(method_cells["MFCS"]["target_marginal_worst_coverage"]),
        "standard_mean_width": float(
            method_cells["Standard CP"]["mean_target_normalized_width"]
        ),
        "scpcp_mean_width": float(
            method_cells["SC-PCP"]["mean_target_normalized_width"]
        ),
        "mfcs_mean_width": float(method_cells["MFCS"]["mean_target_normalized_width"]),
        "scpcp_minus_standard_wsc_pp": float(
            paired_standard["scpcp_minus_baseline_wsc"] * 100.0
        ),
        "scpcp_minus_standard_wsc_ci95_pp": [
            float(value * 100.0)
            for value in paired_standard["scpcp_minus_baseline_wsc_ci95"]
        ],
        "scpcp_to_standard_width_ratio": float(
            paired_standard["scpcp_to_baseline_geometric_width_ratio"]
        ),
        "scpcp_to_standard_width_ratio_ci95": [
            float(value)
            for value in paired_standard[
                "scpcp_to_baseline_geometric_width_ratio_ci95"
            ]
        ],
        "standard_worst_stage_zero_based": int(
            method_cells["Standard CP"]["target_worst_stage_zero_based"]
        ),
        "scpcp_worst_stage_zero_based": int(
            method_cells["SC-PCP"]["target_worst_stage_zero_based"]
        ),
    }
    if metrics["standard_worst_stage_zero_based"] != metrics[
        "scpcp_worst_stage_zero_based"
    ]:
        raise RuntimeError("hero annotation assumes a shared worst stage")
    by_method = {
        method: sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["stage_zero_based"]),
        )
        for method in METHODS
    }
    for method in METHODS:
        coverage = np.asarray(
            [float(row["target_coverage"]) for row in by_method[method]]
        )
        width = np.asarray(
            [float(row["target_normalized_width"]) for row in by_method[method]]
        )
        stored = method_cells[method]
        if not math.isclose(
            float(coverage.min()),
            float(stored["target_marginal_worst_coverage"]),
            rel_tol=0.0,
            abs_tol=1e-14,
        ) or not math.isclose(
            float(width.mean()),
            float(stored["mean_target_normalized_width"]),
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise RuntimeError("hero metrics differ from plotted stage source")
    return metrics


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "legend.fontsize": 6.1,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "svg.hashsalt": "scpcp-controlled-stress-stage-profile-v1",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def render_figure(
    rows: Sequence[Mapping[str, object]], hero: Mapping[str, object]
) -> plt.Figure:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.20, 5.40),
        sharex=True,
        gridspec_kw={"height_ratios": (1.08, 1.0)},
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.88,
        bottom=0.175,
        hspace=0.27,
    )
    coverage_axis, width_axis = axes
    stages = np.arange(HORIZON)
    plot_order = ("MFCS", "ACI", "SPCI", "PRC", "Standard CP", "SC-PCP")
    line_handles: dict[str, Any] = {}
    for method in plot_order:
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["stage_zero_based"]),
        )
        if len(selected) != HORIZON:
            raise RuntimeError(f"figure source is incomplete for {method}")
        color = METHOD_COLORS[method]
        linewidth = 1.75 if method == "SC-PCP" else (
            1.35 if method in {"Standard CP", "MFCS"} else 1.0
        )
        band_alpha = 0.13 if method in {"Standard CP", "MFCS", "SC-PCP"} else 0.055
        deviation = np.asarray(
            [float(row["coverage_deviation_from_090_pp"]) for row in selected]
        )
        deviation_lower = np.asarray(
            [float(row["coverage_deviation_ci95_lower_pp"]) for row in selected]
        )
        deviation_upper = np.asarray(
            [float(row["coverage_deviation_ci95_upper_pp"]) for row in selected]
        )
        coverage_axis.fill_between(
            stages,
            deviation_lower,
            deviation_upper,
            color=color,
            alpha=band_alpha,
            linewidth=0.0,
            zorder=1,
        )
        (handle,) = coverage_axis.plot(
            stages,
            deviation,
            color=color,
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=linewidth,
            markersize=3.3,
            label=method,
            zorder=3 if method == "SC-PCP" else 2,
        )
        line_handles[method] = handle

        width = np.asarray(
            [float(row["target_normalized_width"]) for row in selected]
        )
        width_lower = np.asarray(
            [float(row["target_normalized_width_ci95_lower"]) for row in selected]
        )
        width_upper = np.asarray(
            [float(row["target_normalized_width_ci95_upper"]) for row in selected]
        )
        width_axis.fill_between(
            stages,
            width_lower,
            width_upper,
            color=color,
            alpha=band_alpha,
            linewidth=0.0,
            zorder=1,
        )
        width_axis.plot(
            stages,
            width,
            color=color,
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=linewidth,
            markersize=3.3,
            zorder=3 if method == "SC-PCP" else 2,
        )

    coverage_axis.axhline(
        0.0, color="#333333", linewidth=0.85, linestyle=(0, (3, 2)), zorder=0
    )
    coverage_axis.set_ylabel("Coverage deviation from 90% (pp)")
    coverage_axis.set_ylim(-5.0, 7.0)
    coverage_axis.set_title(
        "Per-stage validity: transport closes most of Standard CP's adverse-shift deficit",
        loc="left",
        fontweight="bold",
    )
    worst_stage = int(hero["standard_worst_stage_zero_based"])
    coverage_axis.annotate(
        (
            f"shared worst stage t={worst_stage}: "
            f"{100 * float(hero['standard_wsc']):.2f}% → "
            f"{100 * float(hero['scpcp_wsc']):.2f}% "
            f"(+{float(hero['scpcp_minus_standard_wsc_pp']):.2f} pp)"
        ),
        xy=(worst_stage, (float(hero["scpcp_wsc"]) - TARGET_COVERAGE) * 100.0),
        xytext=(3.15, -4.35),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": METHOD_COLORS["SC-PCP"], "lw": 0.9},
        fontsize=6.2,
        color="#174F7C",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#A9CBE5",
            "linewidth": 0.6,
            "alpha": 0.94,
        },
    )
    coverage_axis.text(
        11.0,
        6.35,
        f"MFCS WSC {100 * float(hero['mfcs_wsc']):.2f}%\n(conservative)",
        ha="right",
        va="top",
        fontsize=6.1,
        color=METHOD_COLORS["MFCS"],
    )

    width_axis.set_ylabel("Normalized prediction width")
    width_axis.set_xlabel("Treatment stage (zero-based)")
    width_axis.set_title(
        "Per-stage efficiency: coverage repair requires width, while MFCS is wider still",
        loc="left",
        fontweight="bold",
    )
    width_axis.set_ylim(3.1, 13.0)
    width_axis.set_xticks(stages)
    width_axis.text(
        11.0,
        12.35,
        (
            "Mean width across stages\n"
            f"Standard {float(hero['standard_mean_width']):.2f}  |  "
            f"SC-PCP {float(hero['scpcp_mean_width']):.2f}  |  "
            f"MFCS {float(hero['mfcs_mean_width']):.2f}"
        ),
        ha="right",
        va="top",
        fontsize=6.1,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#C8D0D7",
            "linewidth": 0.6,
            "alpha": 0.94,
        },
    )

    for label, axis in zip("ab", axes):
        axis.text(
            -0.085,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=-2)
        axis.set_xlim(-0.2, HORIZON - 0.8)

    figure.suptitle(
        r"Controlled semi-synthetic stress: strongest prespecified adverse alignment ($\gamma=-4$)",
        x=0.01,
        y=0.982,
        ha="left",
        fontsize=9.2,
        fontweight="bold",
    )
    figure.legend(
        [line_handles[method] for method in METHODS],
        list(METHODS),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
        ncol=6,
        handlelength=2.2,
        columnspacing=1.05,
        fontsize=6.1,
    )
    figure.text(
        0.01,
        0.006,
        (
            "n=20 prespecified seeds; bands are pointwise 95% seed-vector bootstrap intervals "
            "(not simultaneous).\nACI/SPCI/PRC use 2,000 target-adaptation trajectories per seed. "
            "Default displayed hero = protocol stress endpoint; not the statistical primary."
        ),
        ha="left",
        va="bottom",
        fontsize=5.6,
        color="#444444",
    )
    return figure


def export_figure(
    figure: plt.Figure,
    *,
    work_stem: Path,
    paper_path: Path,
    tiff_dpi: int = 600,
    png_dpi: int = 240,
) -> None:
    title = "Controlled semi-synthetic gamma=-4 stress stage profile"
    figure.savefig(
        work_stem.with_suffix(".svg"),
        format="svg",
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Creator": "SC-PCP controlled stress stage renderer",
            "Date": None,
        },
    )
    figure.savefig(
        paper_path,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Creator": "SC-PCP controlled stress stage renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        work_stem.with_suffix(".tiff"),
        format="tiff",
        dpi=tiff_dpi,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(
        work_stem.with_suffix(".png"),
        format="png",
        dpi=png_dpi,
        bbox_inches="tight",
        metadata={"Software": "SC-PCP controlled stress stage renderer"},
    )
    plt.close(figure)


def write_source_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_analysis(
    path: Path,
    *,
    config: RenderConfig,
    artifact: FrozenControlledStress,
    source_rows: Sequence[Mapping[str, object]],
    source_path: Path,
    hero: Mapping[str, object],
) -> None:
    payload = {
        "schema_version": 1,
        "protocol": RENDER_PROTOCOL,
        "status": "complete",
        "input_contract": artifact.input_contract,
        "figure_contract": {
            "core_conclusion": (
                "Under the strongest prespecified adverse controlled alignment gamma=-4, "
                "Standard CP undercovers; SC-PCP closes most of the deficit with a "
                "coverage-width tradeoff, while conservative baselines can exceed nominal "
                "at larger width."
            ),
            "archetype": "quantitative_grid",
            "backend": "Python/matplotlib only",
            "design_canvas_inches": [7.20, 5.40],
            "panel_map": {
                "a": "per-stage target coverage deviation from 0.90 for all six canonical methods",
                "b": "per-stage target normalized width for all six canonical methods",
            },
            "evidence_hierarchy": {
                "hero": "panel a Standard CP versus SC-PCP adverse-shift coverage",
                "tradeoff": "panel b Standard CP, SC-PCP, and conservative MFCS widths",
                "context": "all canonical comparators remain visible with their true information budgets",
            },
            "statistics": {
                "n": "20 complete prespecified seed-stage vectors",
                "center": "mean across seeds at each stage",
                "interval": (
                    "pointwise 95% percentile bootstrap across complete seed-stage vectors; "
                    "frozen gamma=-4 bootstrap stream; 10000 resamples"
                ),
                "interval_scope": "pointwise, not simultaneous and not a confidence sequence",
            },
            "reviewer_risks": [
                "gamma=-4 is a controlled semi-synthetic stress endpoint, not the statistical primary, natural setting, or default deployment",
                "SC-PCP point WSC is 0.8983 and must not be described as a finite-sample 90% guarantee",
                "ACI, SPCI, and PRC receive 2000 on-policy adaptation trajectories per seed; offline methods receive none",
                "the result is a coverage-efficiency tradeoff and not universal method dominance or universal SOTA",
            ],
            "reporting_convention": (
                "gamma=-4 is the default displayed hero stress; its protocol role remains "
                "stress, not the statistical primary"
            ),
            "source_data": source_path.name,
            "source_data_sha256": _file_sha256(source_path),
            "source_row_count": len(source_rows),
        },
        "hero_metrics": hero,
        "export_contract": {
            "paper_directory": _project_path(config.paper_output),
            "paper_files": sorted(PAPER_FILES),
            "paper_format": "PDF only; TrueType Times New Roman",
            "work_formats": ["editable SVG", "600-dpi TIFF", "240-dpi PNG"],
            "svg_text": "text elements retained",
            "determinism": (
                "frozen artifact hashes and bootstrap stream, fixed row order, palette, "
                "markers, and svg.hashsalt; no experiment rerun"
            ),
        },
        "claim_boundary": (
            "Controlled semi-synthetic stress visualization only. It is not the statistical "
            "primary, a natural/clinical estimate, a finite-sample distribution-free guarantee, "
            "or evidence of universal SOTA."
        ),
    }
    _write_json(path, payload)


def _write_qa_notes(path: Path, *, hero: Mapping[str, object]) -> None:
    path.write_text(
        "\n".join(
            (
                "# Controlled stress stage-profile QA",
                "",
                "- Core conclusion: under the strongest prespecified adverse controlled alignment gamma=-4, Standard CP undercovers; SC-PCP closes most of the deficit with a width tradeoff, while MFCS exceeds nominal at still larger width.",
                "- Archetype: two-panel quantitative grid; coverage deviation is the hero panel and width is the required efficiency panel.",
                "- Backend: Python/matplotlib only; no model fit, calibration, policy rollout, method selection, or scientific seed rerun.",
                "- Design canvas: 7.20 x 5.40 inches (double-column); vector bounds retained on export.",
                "- Input: immutable formal all-six controlled artifact; exact summary and provenance hashes are fail-closed.",
                f"- Frozen experiment source hash: `{FROZEN_SOURCE_TREE_SHA256}`.",
                f"- Frozen summary hash: `{FROZEN_SUMMARY_SHA256}`.",
                "- n: 20 complete prespecified seed-stage vectors; T=12.",
                "- Center: stagewise mean across seeds. Bands: pointwise 95% percentile bootstrap with the frozen gamma=-4 shared seed-vector stream and 10,000 resamples.",
                "- Bands are pointwise, not simultaneous, not a confidence sequence, and not the frozen WSC interval.",
                f"- Standard CP WSC: {float(hero['standard_wsc']):.8f}; SC-PCP WSC: {float(hero['scpcp_wsc']):.8f}; paired improvement: {float(hero['scpcp_minus_standard_wsc_pp']):.3f} pp.",
                f"- Mean normalized width: Standard {float(hero['standard_mean_width']):.3f}, SC-PCP {float(hero['scpcp_mean_width']):.3f}, MFCS {float(hero['mfcs_mean_width']):.3f}.",
                "- SC-PCP remains slightly below 0.90 at its worst stage; no finite-sample validity claim is made.",
                "- Gamma=-4 is explicitly labeled controlled semi-synthetic stress and is not relabeled statistical primary, natural, clinical, or the default deployment setting.",
                "- Reporting convention: gamma=-4 is the default displayed hero stress; its protocol role remains stress, not the statistical primary.",
                "- ACI, SPCI, and PRC each receive 2,000 on-policy adaptation trajectories per seed; Standard CP, MFCS, and SC-PCP receive zero.",
                "- All six canonical methods appear; no diagnostic/ablation row is presented as a baseline.",
                "- The plot supports a setting-specific coverage-efficiency tradeoff, not universal dominance or universal SOTA.",
                "- Typography: Times New Roman; editable SVG text; embedded TrueType PDF fonts.",
                "- Accessibility: restrained palette plus marker and line-style redundancy; no rainbow map.",
                "- Work output contains source CSV, contract/analysis, QA, manifest, editable SVG, 600-dpi TIFF, and PNG preview.",
                "- Paper output contains exactly one PDF and no auxiliary file.",
                "- No raster-image manipulation; the figure is generated directly from stored numerical seed rows.",
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_render_manifest(path: Path, *, work_root: Path, paper_root: Path) -> None:
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
            "work_files": work_files,
            "paper_files": paper_files,
        },
    )


def validate_rendered_outputs(work_root: Path, paper_root: Path) -> None:
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
    if "<text" not in svg or "Times New Roman" not in svg:
        raise RuntimeError("SVG text/font contract differs")
    if not (paper_root / f"{FIGURE_STEM}.pdf").read_bytes().startswith(b"%PDF"):
        raise RuntimeError("PDF header is malformed")
    if not (work_root / f"{FIGURE_STEM}.png").read_bytes().startswith(b"\x89PNG"):
        raise RuntimeError("PNG header is malformed")
    if (work_root / f"{FIGURE_STEM}.tiff").read_bytes()[:4] not in {
        b"II*\x00",
        b"MM\x00*",
    }:
        raise RuntimeError("TIFF header is malformed")
    manifest = _read_json_mapping(work_root / "render_manifest.json")
    if (
        manifest.get("protocol") != RENDER_PROTOCOL
        or manifest.get("status") != "complete"
        or set(_mapping(manifest.get("paper_files"), "manifest paper"))
        != PAPER_FILES
    ):
        raise RuntimeError("render manifest contract differs")
    for group, root in (("work_files", work_root), ("paper_files", paper_root)):
        for name, contract in _mapping(manifest[group], group).items():
            _validate_file_contract(root / name, contract)


def _stress_aggregate(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        cell
        for cell in summary.get("aggregates", ())
        if isinstance(cell, Mapping) and float(cell.get("gamma")) == STRESS_GAMMA
    ]
    if len(matches) != 1:
        raise RuntimeError("summary must contain exactly one gamma=-4 aggregate")
    return matches[0]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"{label} is not finite")
    return resolved


def _finite_vector(value: object, *, length: int, label: str) -> np.ndarray:
    resolved = np.asarray(value, dtype=np.float64)
    if resolved.shape != (length,) or not np.isfinite(resolved).all():
        raise RuntimeError(f"{label} must have finite length {length}")
    return resolved


def _require_exact_root_entries(root: Path, expected: set[str]) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {root}")
    observed = {path.name for path in root.iterdir()}
    if observed != expected:
        raise RuntimeError(
            f"artifact entries differ for {root}: expected {sorted(expected)}, found {sorted(observed)}"
        )


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_contract(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _validate_file_contract(path: Path, contract: object) -> None:
    resolved = _mapping(contract, f"file contract {path.name}")
    if not path.is_file():
        raise RuntimeError(f"missing contracted file: {path}")
    if (
        resolved.get("bytes") != path.stat().st_size
        or resolved.get("sha256") != _file_sha256(path)
    ):
        raise RuntimeError(f"file contract differs: {path}")


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
