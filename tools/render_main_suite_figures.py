"""Render supplementary main-suite figures from the frozen canonical export.

This is a deterministic post-processing entry point.  It validates the exact
2026-08-24 export, extracts figure-specific source tables, and renders two
publication figures without rerunning any scientific experiment or bootstrap.

Example
-------
conda run -n ucp python tools/render_main_suite_figures.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RENDER_PROTOCOL = "frozen_main_suite_figures_v1"
DEFAULT_INPUT = ROOT / "results/work/complete_baseline_results_20260824"
DEFAULT_WORK_OUTPUT = ROOT / "results/work/main_suite_figures_20260826"
DEFAULT_PAPER_OUTPUT = ROOT / "results/paper_main_suite_figures_20260826"

TARGET = 0.90
METHODS = (
    "Standard CP",
    "ACI",
    "MFCS",
    "SPCI",
    "PRC",
    "SC-PCP",
)
DATASETS = ("synthetic", "mimic_iv", "mimic_cxr", "eicu", "inspire")
PROFILE_DATASETS = ("synthetic", "mimic_iv", "inspire")
DATASET_LABELS = {
    "synthetic": "Synthetic",
    "mimic_iv": "MIMIC-IV",
    "mimic_cxr": "MIMIC-CXR + IV/ED",
    "eicu": "eICU",
    "inspire": "INSPIRE",
}
HORIZONS = {
    "synthetic": 12,
    "mimic_iv": 12,
    "mimic_cxr": 6,
    "eicu": 12,
    "inspire": 12,
}
EXPECTED_RUNS = {
    "synthetic": 100,
    "mimic_iv": 20,
    "mimic_cxr": 20,
    "eicu": 20,
    "inspire": 20,
}
HIGHLIGHTED_PROFILE_METHODS = {
    "Standard CP",
    "ACI",
    "MFCS",
    "SC-PCP",
}

METHOD_COLORS = {
    "Standard CP": "#6F7782",
    "ACI": "#B44E4A",
    "MFCS": "#279C9C",
    "SPCI": "#8064A2",
    "PRC": "#568D45",
    "SC-PCP": "#2F80ED",
}
METHOD_MARKERS = {
    "Standard CP": "o",
    "ACI": "v",
    "MFCS": "D",
    "SPCI": "h",
    "PRC": ">",
    "SC-PCP": "*",
}
METHOD_LINESTYLES = {
    "Standard CP": (0, (5, 2)),
    "ACI": "-.",
    "MFCS": ":",
    "SPCI": (0, (2, 2)),
    "PRC": (0, (4, 2, 1, 2)),
    "SC-PCP": "-",
}

PARETO_STEM = "figure_main_pareto"
PROFILE_STEM = "figure_stagewise_profiles"
PAPER_FILES = {f"{PARETO_STEM}.pdf", f"{PROFILE_STEM}.pdf"}
WORK_FILES = {
    *(f"{stem}.{suffix}" for stem in (PARETO_STEM, PROFILE_STEM) for suffix in ("svg", "pdf", "tiff", "png")),
    "figure_main_pareto_source_data.csv",
    "figure_stagewise_profiles_source_data.csv",
    "figure_contract.md",
    "figure_qa.md",
    "analysis.json",
    "render_manifest.json",
}

INPUT_FILES = {
    "README.md": "c9fa6f565c225e06c26ec7447da544d084cd4011e0b30874da50db1583a00b25",
    "metadata.json": "90db917f4208bea5dc0cf956c015f1b1f622a42e873b3788b521d623fcdaf788",
    "per_stage_all_baselines.csv": "aa6915dddfcfae5aa8e1ee7e4c8eca8b96fcb75825799a72f6bbe57244b93e91",
    "rq1_all_baselines.csv": "05f8846aeb09a5feaa1165d40b7d90f200992a264ad9fac6e2d1cc23d9f38347",
    "rq3_all_baselines.csv": "3e1cbf08f984a4b22533521d1014225457c6d69b766618582de21121b86b300a",
}

MAIN_COLUMNS = (
    "section",
    "setting",
    "dataset",
    "method",
    "marginal_worst_coverage",
    "marginal_worst_coverage_ci_low",
    "marginal_worst_coverage_ci_high",
    "average_coverage",
    "average_coverage_ci_low",
    "average_coverage_ci_high",
    "average_normalized_width",
    "average_normalized_width_ci_low",
    "average_normalized_width_ci_high",
    "selection_rate",
    "selection_rate_ci_low",
    "selection_rate_ci_high",
    "marginal_worst_target_met",
    "selection_rate_at_least_95_percent",
    "efficiency_eligible",
    "n_runs",
    "n_selected",
)
STAGE_COLUMNS = (
    "section",
    "setting",
    "dataset",
    "feedback_strength",
    "method",
    "n_runs",
    "n_selected",
    "stage_zero_based",
    "coverage_mean",
    "coverage_ci_low",
    "coverage_ci_high",
    "stage_target_met",
    "normalized_width_mean",
    "normalized_width_ci_low",
    "normalized_width_ci_high",
)


# Repository-specific journal convention.  SVG text stays editable and PDF text
# is embedded as TrueType; the fallback is only used when Times is unavailable.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.hashsalt"] = "scpcp-frozen-main-suite-figures-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--work-output", type=Path, default=DEFAULT_WORK_OUTPUT)
    parser.add_argument("--paper-output", type=Path, default=DEFAULT_PAPER_OUTPUT)
    args = parser.parse_args()
    render_report(
        input_root=args.input.resolve(),
        work_output=args.work_output.resolve(),
        paper_output=args.paper_output.resolve(),
    )
    print(args.paper_output.resolve())


def render_report(*, input_root: Path, work_output: Path, paper_output: Path) -> None:
    """Validate the frozen inputs and atomically publish both figure bundles."""

    if work_output.exists() or paper_output.exists():
        raise FileExistsError("work-output and paper-output must both be fresh")
    if work_output == paper_output:
        raise ValueError("work-output and paper-output must be different directories")

    main_frame, stage_frame, metadata = load_frozen_export(input_root)
    pareto = build_pareto_source(main_frame)
    profiles = build_profile_source(stage_frame)

    work_output.parent.mkdir(parents=True, exist_ok=True)
    paper_output.parent.mkdir(parents=True, exist_ok=True)
    staged_work = Path(
        tempfile.mkdtemp(prefix=f".{work_output.name}-", dir=work_output.parent)
    )
    staged_paper = Path(
        tempfile.mkdtemp(prefix=f".{paper_output.name}-", dir=paper_output.parent)
    )
    try:
        _write_csv(staged_work / "figure_main_pareto_source_data.csv", pareto)
        _write_csv(
            staged_work / "figure_stagewise_profiles_source_data.csv", profiles
        )
        _write_figure_contract(staged_work / "figure_contract.md")
        apply_publication_style()
        export_figure(
            render_pareto_figure(pareto),
            title="Frozen five-dataset coverage-width Pareto comparison",
            work_stem=staged_work / PARETO_STEM,
            paper_path=staged_paper / f"{PARETO_STEM}.pdf",
        )
        export_figure(
            render_profile_figure(profiles),
            title="Frozen stagewise coverage-deviation and width profiles",
            work_stem=staged_work / PROFILE_STEM,
            paper_path=staged_paper / f"{PROFILE_STEM}.pdf",
        )
        _write_analysis(
            staged_work / "analysis.json",
            metadata=metadata,
            pareto=pareto,
            profiles=profiles,
        )
        _write_qa(staged_work / "figure_qa.md", metadata=metadata)
        _write_render_manifest(
            staged_work / "render_manifest.json",
            work_root=staged_work,
            paper_root=staged_paper,
        )
        validate_rendered_outputs(staged_work, staged_paper)
        os.replace(staged_work, work_output)
        os.replace(staged_paper, paper_output)
    except BaseException:
        shutil.rmtree(staged_work, ignore_errors=True)
        shutil.rmtree(staged_paper, ignore_errors=True)
        raise


def load_frozen_export(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(f"frozen export does not exist: {root}")
    observed = {path.name for path in root.iterdir()}
    if observed != set(INPUT_FILES):
        raise RuntimeError(
            "frozen export entry set differs: "
            f"expected {sorted(INPUT_FILES)}, found {sorted(observed)}"
        )
    for name, expected_hash in INPUT_FILES.items():
        observed_hash = _file_sha256(root / name)
        if observed_hash != expected_hash:
            raise RuntimeError(f"frozen export hash differs for {name}")

    metadata = _read_json_mapping(root / "metadata.json")
    main_frame = pd.read_csv(root / "rq1_all_baselines.csv")
    stage_frame = pd.read_csv(root / "per_stage_all_baselines.csv")
    validate_source_frames(main_frame, stage_frame, metadata)
    return main_frame, stage_frame, metadata


def validate_source_frames(
    main_frame: pd.DataFrame,
    stage_frame: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> None:
    if tuple(main_frame.columns) != MAIN_COLUMNS:
        raise RuntimeError("RQ1 source schema differs")
    if tuple(stage_frame.columns) != STAGE_COLUMNS:
        raise RuntimeError("per-stage source schema differs")
    if len(main_frame) != 30 or len(stage_frame) != 612:
        raise RuntimeError("frozen source row counts differ")
    if metadata.get("canonical_methods") != list(METHODS):
        raise RuntimeError("canonical method order differs")
    if metadata.get("coverage_target") != TARGET:
        raise RuntimeError("coverage target differs")
    if metadata.get("rq1_rows") != 30 or metadata.get("per_stage_rows") != 612:
        raise RuntimeError("metadata row counts differ")
    if metadata.get("wsc_definition") != (
        "min_t mean_selected_seed(per_time_coverage_seed_t)"
    ):
        raise RuntimeError("WSC definition differs")
    if metadata.get("suite_method") != (
        "direct_committed_prefix_uncapped_importance_weighting"
    ):
        raise RuntimeError("frozen SC-PCP method differs")
    if metadata.get("suite_protocol") != "committed_prefix_marginal_scpcp":
        raise RuntimeError("frozen suite protocol differs")

    expected_pairs = {(dataset, method) for dataset in DATASETS for method in METHODS}
    observed_pairs = set(zip(main_frame["dataset"], main_frame["method"]))
    if observed_pairs != expected_pairs or main_frame.duplicated(
        ["dataset", "method"]
    ).any():
        raise RuntimeError("RQ1 source must contain one row per dataset and method")
    if set(main_frame["section"]) != {"RQ1"}:
        raise RuntimeError("RQ1 source contains a non-RQ1 row")

    probability_fields = (
        "marginal_worst_coverage",
        "marginal_worst_coverage_ci_low",
        "marginal_worst_coverage_ci_high",
        "average_coverage",
        "average_coverage_ci_low",
        "average_coverage_ci_high",
        "selection_rate",
        "selection_rate_ci_low",
        "selection_rate_ci_high",
    )
    for field in probability_fields:
        values = pd.to_numeric(main_frame[field], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise RuntimeError(f"RQ1 {field} is not a finite probability")
    for point, low, high in (
        (
            "marginal_worst_coverage",
            "marginal_worst_coverage_ci_low",
            "marginal_worst_coverage_ci_high",
        ),
        ("average_coverage", "average_coverage_ci_low", "average_coverage_ci_high"),
        (
            "average_normalized_width",
            "average_normalized_width_ci_low",
            "average_normalized_width_ci_high",
        ),
        ("selection_rate", "selection_rate_ci_low", "selection_rate_ci_high"),
    ):
        _validate_interval_columns(main_frame, point, low, high, label="RQ1")
    widths = main_frame[
        [
            "average_normalized_width",
            "average_normalized_width_ci_low",
            "average_normalized_width_ci_high",
        ]
    ].to_numpy(float)
    if not np.isfinite(widths).all() or np.any(widths <= 0.0):
        raise RuntimeError("RQ1 widths must be finite and positive")

    expected_runs = main_frame["dataset"].map(EXPECTED_RUNS).to_numpy(int)
    if not np.array_equal(main_frame["n_runs"].to_numpy(int), expected_runs):
        raise RuntimeError("RQ1 run counts differ")
    if not np.array_equal(main_frame["n_selected"], main_frame["n_runs"]):
        raise RuntimeError("this frozen export must have complete method selection")

    main_stage = stage_frame[stage_frame["section"].eq("RQ1")].copy()
    if len(main_stage) != sum(HORIZONS.values()) * len(METHODS):
        raise RuntimeError("main per-stage row count differs")
    observed_stage_keys = set(
        zip(
            main_stage["dataset"],
            main_stage["method"],
            main_stage["stage_zero_based"],
        )
    )
    expected_stage_keys = {
        (dataset, method, stage)
        for dataset in DATASETS
        for method in METHODS
        for stage in range(HORIZONS[dataset])
    }
    if observed_stage_keys != expected_stage_keys or main_stage.duplicated(
        ["dataset", "method", "stage_zero_based"]
    ).any():
        raise RuntimeError("main per-stage grid differs")
    if set(main_stage["method"]) != set(METHODS):
        raise RuntimeError("main per-stage methods differ")
    for point, low, high in (
        ("coverage_mean", "coverage_ci_low", "coverage_ci_high"),
        (
            "normalized_width_mean",
            "normalized_width_ci_low",
            "normalized_width_ci_high",
        ),
    ):
        _validate_interval_columns(main_stage, point, low, high, label="per-stage")
    coverage = main_stage[
        ["coverage_mean", "coverage_ci_low", "coverage_ci_high"]
    ].to_numpy(float)
    width = main_stage[
        [
            "normalized_width_mean",
            "normalized_width_ci_low",
            "normalized_width_ci_high",
        ]
    ].to_numpy(float)
    if np.any((coverage < 0.0) | (coverage > 1.0)):
        raise RuntimeError("per-stage coverage is outside [0,1]")
    if np.any(width <= 0.0):
        raise RuntimeError("per-stage widths must be positive")

    recomputed = (
        main_stage.groupby(["dataset", "method"], sort=False)
        .agg(
            recomputed_wsc=("coverage_mean", "min"),
            recomputed_width=("normalized_width_mean", "mean"),
        )
        .reset_index()
    )
    joined = main_frame.merge(recomputed, on=["dataset", "method"], validate="one_to_one")
    if not np.allclose(
        joined["marginal_worst_coverage"], joined["recomputed_wsc"], atol=1e-12
    ):
        raise RuntimeError("RQ1 WSC disagrees with the per-stage means")
    if not np.allclose(
        joined["average_normalized_width"],
        joined["recomputed_width"],
        atol=5e-7,
    ):
        raise RuntimeError("RQ1 width disagrees with the per-stage means")


def _validate_interval_columns(
    frame: pd.DataFrame, point: str, low: str, high: str, *, label: str
) -> None:
    values = frame[[point, low, high]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} interval {point} is not finite")
    if np.any(values[:, 1] > values[:, 0]) or np.any(values[:, 0] > values[:, 2]):
        raise RuntimeError(f"{label} interval {point} does not contain its point")


def build_pareto_source(main_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        group = main_frame[main_frame["dataset"].eq(dataset)].set_index("method")
        eligible = group[group["marginal_worst_coverage"].ge(TARGET)]
        narrowest = str(eligible["average_normalized_width"].idxmin())
        for method in METHODS:
            row = group.loc[method]
            dominated = False
            for other_method in METHODS:
                if other_method == method:
                    continue
                other = group.loc[other_method]
                weakly_better = (
                    other["average_normalized_width"]
                    <= row["average_normalized_width"]
                    and other["marginal_worst_coverage"]
                    >= row["marginal_worst_coverage"]
                )
                strictly_better = (
                    other["average_normalized_width"]
                    < row["average_normalized_width"]
                    or other["marginal_worst_coverage"]
                    > row["marginal_worst_coverage"]
                )
                dominated = dominated or (weakly_better and strictly_better)
            rows.append(
                {
                    "figure": PARETO_STEM,
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "method": method,
                    "wsc": float(row["marginal_worst_coverage"]),
                    "wsc_ci95_lower": float(row["marginal_worst_coverage_ci_low"]),
                    "wsc_ci95_upper": float(row["marginal_worst_coverage_ci_high"]),
                    "average_normalized_width": float(row["average_normalized_width"]),
                    "width_ci95_lower": float(row["average_normalized_width_ci_low"]),
                    "width_ci95_upper": float(row["average_normalized_width_ci_high"]),
                    "point_coverage_eligible": bool(row["marginal_worst_coverage"] >= TARGET),
                    "point_pareto_frontier": not dominated,
                    "narrowest_point_eligible": method == narrowest,
                    "selection_rate": float(row["selection_rate"]),
                    "n_runs": int(row["n_runs"]),
                    "n_selected": int(row["n_selected"]),
                    "source_file": "rq1_all_baselines.csv",
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 30:
        raise RuntimeError("Pareto source must retain all 30 canonical rows")
    scpcp = result[result["method"].eq("SC-PCP")]
    if set(scpcp.loc[scpcp["point_pareto_frontier"], "dataset"]) != set(DATASETS):
        raise RuntimeError("frozen SC-PCP point-frontier audit differs")
    return result


def build_profile_source(stage_frame: pd.DataFrame) -> pd.DataFrame:
    selected = stage_frame[
        stage_frame["section"].eq("RQ1")
        & stage_frame["dataset"].isin(PROFILE_DATASETS)
    ].copy()
    selected["figure"] = PROFILE_STEM
    selected["dataset_label"] = selected["dataset"].map(DATASET_LABELS)
    selected["coverage_deviation_pp"] = 100.0 * (selected["coverage_mean"] - TARGET)
    selected["coverage_deviation_ci95_lower_pp"] = 100.0 * (
        selected["coverage_ci_low"] - TARGET
    )
    selected["coverage_deviation_ci95_upper_pp"] = 100.0 * (
        selected["coverage_ci_high"] - TARGET
    )
    selected["display_role"] = np.where(
        selected["method"].isin(HIGHLIGHTED_PROFILE_METHODS),
        "highlighted",
        "muted_comparator",
    )
    selected["source_file"] = "per_stage_all_baselines.csv"
    columns = (
        "figure",
        "dataset",
        "dataset_label",
        "method",
        "stage_zero_based",
        "coverage_mean",
        "coverage_ci_low",
        "coverage_ci_high",
        "coverage_deviation_pp",
        "coverage_deviation_ci95_lower_pp",
        "coverage_deviation_ci95_upper_pp",
        "normalized_width_mean",
        "normalized_width_ci_low",
        "normalized_width_ci_high",
        "display_role",
        "n_runs",
        "n_selected",
        "source_file",
    )
    selected = selected.loc[:, columns].sort_values(
        ["dataset", "method", "stage_zero_based"],
        key=lambda values: values.map(
            {name: index for index, name in enumerate(PROFILE_DATASETS)}
            if values.name == "dataset"
            else ({name: index for index, name in enumerate(METHODS)} if values.name == "method" else {})
        )
        if values.name in {"dataset", "method"}
        else values,
    )
    if len(selected) != len(PROFILE_DATASETS) * len(METHODS) * 12:
        raise RuntimeError("stage-profile source grid differs")
    if set(selected["method"]) != set(METHODS):
        raise RuntimeError("stage-profile source does not retain all six methods")
    return selected.reset_index(drop=True)


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "scpcp-frozen-main-suite-figures-v1",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def render_pareto_figure(source: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(7.20, 4.30), sharey=True)
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.105,
        top=0.935,
        wspace=0.27,
        hspace=0.37,
    )
    y_limits = (0.889, 0.926)
    for panel_index, dataset in enumerate(DATASETS):
        axis = axes.ravel()[panel_index]
        group = source[source["dataset"].eq(dataset)].set_index("method")
        axis.axhspan(TARGET, y_limits[1], color="#E8F3EC", alpha=0.82, zorder=0)
        axis.axhline(TARGET, color="#3F4A45", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
        frontier = group[group["point_pareto_frontier"]].sort_values(
            "average_normalized_width"
        )
        axis.plot(
            frontier["average_normalized_width"],
            frontier["wsc"],
            color="#B7BDC4",
            linewidth=0.7,
            zorder=1,
        )
        for method in METHODS:
            row = group.loc[method]
            x = float(row["average_normalized_width"])
            y = float(row["wsc"])
            axis.errorbar(
                x,
                y,
                xerr=[
                    [x - float(row["width_ci95_lower"])],
                    [float(row["width_ci95_upper"]) - x],
                ],
                yerr=[
                    [y - float(row["wsc_ci95_lower"])],
                    [float(row["wsc_ci95_upper"]) - y],
                ],
                fmt=METHOD_MARKERS[method],
                markersize=6.4 if method == "SC-PCP" else 4.2,
                color=METHOD_COLORS[method],
                markeredgecolor="#20242A" if method == "SC-PCP" else METHOD_COLORS[method],
                markeredgewidth=0.75 if method == "SC-PCP" else 0.45,
                ecolor=METHOD_COLORS[method],
                elinewidth=0.65,
                capsize=1.6,
                alpha=1.0,
                zorder=4 if method == "SC-PCP" else 3,
            )
            if bool(row["narrowest_point_eligible"]):
                axis.plot(
                    x,
                    y,
                    marker="o",
                    markersize=9.2,
                    markerfacecolor="none",
                    markeredgecolor="#C18F00",
                    markeredgewidth=1.05,
                    linestyle="none",
                    zorder=5,
                )
        x_low = float(group["width_ci95_lower"].min())
        x_high = float(group["width_ci95_upper"].max())
        padding = max(0.02 * max(abs(x_low), abs(x_high)), 0.09 * (x_high - x_low))
        axis.set_xlim(x_low - padding, x_high + padding)
        axis.set_ylim(*y_limits)
        axis.set_title(DATASET_LABELS[dataset], fontweight="bold", pad=4)
        axis.grid(axis="y", color="#D7DADF", linewidth=0.55, alpha=0.75)
        axis.tick_params(width=0.65, length=2.4)
        if panel_index in (0, 3):
            axis.set_ylabel("WSC")
        axis.set_xlabel("Average normalized width")
        add_panel_label(axis, "abcde"[panel_index])

    legend_axis = axes.ravel()[-1]
    legend_axis.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method],
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="#20242A" if method == "SC-PCP" else METHOD_COLORS[method],
            markersize=7 if method == "SC-PCP" else 5,
            label=method,
        )
        for method in METHODS
    ]
    legend_axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
        handletextpad=0.45,
        labelspacing=0.62,
    )
    legend_axis.text(
        0.02,
        0.35,
        "Green field: point WSC ≥ 0.90\n"
        "Gold ring: narrowest eligible method\n"
        "Gray connector: point Pareto frontier\n"
        "Bars: two-sided 95% intervals",
        transform=legend_axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        linespacing=1.35,
        color="#3C4148",
    )
    return figure


def render_profile_figure(source: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(7.20, 4.95), sharex="col")
    figure.subplots_adjust(
        left=0.080,
        right=0.985,
        bottom=0.105,
        top=0.865,
        wspace=0.25,
        hspace=0.22,
    )
    coverage_low = float(source["coverage_deviation_ci95_lower_pp"].min())
    coverage_high = float(source["coverage_deviation_ci95_upper_pp"].max())
    coverage_limits = (
        math.floor((coverage_low - 0.15) * 2.0) / 2.0,
        math.ceil((coverage_high + 0.15) * 2.0) / 2.0,
    )
    legend_handles: list[Line2D] = []
    for column, dataset in enumerate(PROFILE_DATASETS):
        group = source[source["dataset"].eq(dataset)]
        coverage_axis = axes[0, column]
        width_axis = axes[1, column]
        coverage_axis.axhline(0.0, color="#30353A", linestyle=(0, (3, 2)), linewidth=0.8)
        coverage_axis.axhspan(coverage_limits[0], 0.0, color="#F8E9E7", alpha=0.45, zorder=0)
        for method in METHODS:
            rows = group[group["method"].eq(method)].sort_values("stage_zero_based")
            x = rows["stage_zero_based"].to_numpy(float)
            emphasized = method in HIGHLIGHTED_PROFILE_METHODS
            alpha = 0.98 if emphasized else 0.34
            linewidth = 1.35 if method == "SC-PCP" else (1.05 if emphasized else 0.75)
            markersize = 4.2 if method == "SC-PCP" else (2.8 if emphasized else 2.2)
            for axis, point_field, low_field, high_field in (
                (
                    coverage_axis,
                    "coverage_deviation_pp",
                    "coverage_deviation_ci95_lower_pp",
                    "coverage_deviation_ci95_upper_pp",
                ),
                (
                    width_axis,
                    "normalized_width_mean",
                    "normalized_width_ci_low",
                    "normalized_width_ci_high",
                ),
            ):
                point = rows[point_field].to_numpy(float)
                color = METHOD_COLORS[method]
                axis.plot(
                    x,
                    point,
                    color=color,
                    linestyle=METHOD_LINESTYLES[method],
                    marker=METHOD_MARKERS[method],
                    markersize=markersize,
                    linewidth=linewidth,
                    markeredgewidth=0.35,
                    alpha=alpha,
                    zorder=4 if method == "SC-PCP" else (3 if emphasized else 1),
                )
                if emphasized:
                    axis.fill_between(
                        x,
                        rows[low_field].to_numpy(float),
                        rows[high_field].to_numpy(float),
                        color=color,
                        alpha=0.055 if method != "SC-PCP" else 0.085,
                        linewidth=0.0,
                        zorder=0,
                    )
            if column == 0:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=METHOD_COLORS[method],
                        linestyle=METHOD_LINESTYLES[method],
                        marker=METHOD_MARKERS[method],
                        markersize=4.4 if method == "SC-PCP" else 3.2,
                        linewidth=1.35 if method == "SC-PCP" else (1.05 if emphasized else 0.75),
                        alpha=alpha,
                        label=method,
                    )
                )
        coverage_axis.set_title(DATASET_LABELS[dataset], fontweight="bold", pad=4)
        coverage_axis.set_ylim(*coverage_limits)
        coverage_axis.set_xticks(range(12))
        width_values = group[
            ["normalized_width_ci_low", "normalized_width_ci_high"]
        ].to_numpy(float)
        width_min, width_max = float(width_values.min()), float(width_values.max())
        width_padding = 0.07 * (width_max - width_min)
        width_axis.set_ylim(width_min - width_padding, width_max + width_padding)
        for axis in (coverage_axis, width_axis):
            axis.grid(axis="y", color="#D7DADF", linewidth=0.55, alpha=0.75)
            axis.tick_params(width=0.65, length=2.4)
        width_axis.set_xlabel("Treatment stage, t")
        add_panel_label(coverage_axis, "abc"[column])
        add_panel_label(width_axis, "def"[column])
    axes[0, 0].set_ylabel("Coverage deviation (pp)")
    axes[1, 0].set_ylabel("Normalized width")
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.982),
        ncol=6,
        columnspacing=1.05,
        handlelength=2.0,
        handletextpad=0.45,
    )
    figure.text(
        0.5,
        0.018,
        "Lines are conditional on method selection (selection = 100% here); shaded intervals are pointwise 95% CIs for the four highlighted methods.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#3C4148",
    )
    return figure


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.035,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def export_figure(
    figure: plt.Figure,
    *,
    title: str,
    work_stem: Path,
    paper_path: Path,
    tiff_dpi: int = 600,
    png_dpi: int = 240,
) -> None:
    creator = "SC-PCP frozen main-suite renderer"
    figure.savefig(
        work_stem.with_suffix(".svg"),
        format="svg",
        bbox_inches="tight",
        metadata={"Title": title, "Creator": creator, "Date": None},
    )
    work_pdf = work_stem.with_suffix(".pdf")
    figure.savefig(
        work_pdf,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Creator": creator,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    shutil.copyfile(work_pdf, paper_path)
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
        metadata={"Software": creator},
    )
    plt.close(figure)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _write_figure_contract(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "# Frozen main-suite figure contract",
                "",
                "## Coverage-width figure",
                "",
                "- Core conclusion: the frozen five-dataset suite exposes the full six-method validity-efficiency trade-off; SC-PCP is point-nondominated in every dataset, without treating conservative overcoverage as automatically better.",
                "- Archetype: quantitative grid.",
                "- Hero evidence: WSC versus average normalized width for all six canonical comparison rows.",
                "- Validation evidence: two-sided 95% intervals, point target region, descriptive point frontier, and narrowest point-eligible marker.",
                "",
                "## Stagewise figure",
                "",
                "- Core conclusion: coverage deviation and normalized width profiles reveal sequential behavior that a scalar WSC cannot show.",
                "- Archetype: quantitative grid with aligned stages.",
                "- Hero evidence: Synthetic, MIMIC-IV, and INSPIRE stagewise profiles.",
                "- Visual hierarchy: Standard CP, ACI, MFCS, and SC-PCP are highlighted; SPCI and PRC remain present as muted comparison rows.",
                "",
                "## Export and review contract",
                "",
                "- Backend: Python/matplotlib only.",
                "- Final width: 7.20 inches (approximately 183 mm).",
                "- Formats: editable SVG, TrueType PDF, 600-dpi LZW TIFF, and 240-dpi PNG preview.",
                "- Source data: one tidy CSV per figure with all displayed estimates and intervals.",
                "- Reviewer risk: eligibility and Pareto status are descriptive point-estimate summaries, not significance tests or universal SOTA claims.",
                "- Image integrity: no raster input or image manipulation; panels are generated from frozen numerical summaries.",
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_analysis(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    pareto: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    frontier = {
        dataset: pareto[
            pareto["dataset"].eq(dataset) & pareto["point_pareto_frontier"]
        ]["method"].tolist()
        for dataset in DATASETS
    }
    narrowest = {
        dataset: str(
            pareto[
                pareto["dataset"].eq(dataset)
                & pareto["narrowest_point_eligible"]
            ]["method"].iloc[0]
        )
        for dataset in DATASETS
    }
    payload = {
        "schema_version": 1,
        "protocol": RENDER_PROTOCOL,
        "status": "complete",
        "input_experiment_tree_sha256": metadata["experiment_tree_sha256"],
        "input_file_sha256": INPUT_FILES,
        "backend": "Python/matplotlib only",
        "method_order": list(METHODS),
        "coverage_target": TARGET,
        "eligibility_rule": "point WSC >= 0.90",
        "pareto_rule": (
            "point-nondominated when no other method has weakly smaller average "
            "normalized width and weakly larger WSC, with at least one strict"
        ),
        "findings": {
            "scpcp_point_frontier_datasets": [
                DATASET_LABELS[name] for name in DATASETS
            ],
            "point_frontier_methods_by_dataset": frontier,
            "narrowest_point_eligible_method_by_dataset": narrowest,
            "stage_profile_datasets": [DATASET_LABELS[name] for name in PROFILE_DATASETS],
            "stage_profile_methods_retained": list(METHODS),
            "stage_profile_highlighted_methods": sorted(
                HIGHLIGHTED_PROFILE_METHODS,
                key=METHODS.index,
            ),
        },
        "source_data": {
            "figure_main_pareto_source_data.csv": {
                "rows": len(pareto),
                "sha256": _file_sha256(path.parent / "figure_main_pareto_source_data.csv"),
            },
            "figure_stagewise_profiles_source_data.csv": {
                "rows": len(profiles),
                "sha256": _file_sha256(
                    path.parent / "figure_stagewise_profiles_source_data.csv"
                ),
            },
        },
        "claim_boundary": (
            "Descriptive frozen-benchmark comparison only. Method labels identify the "
            "canonical comparison implementations and do not assert that the sequential "
            "evaluation is a native operating regime for every adapter. No finite-sample, "
            "distribution-free, PAC, data-conditional, superiority, or universal SOTA claim."
        ),
    }
    _write_json(path, payload)


def _write_qa(path: Path, *, metadata: Mapping[str, Any]) -> None:
    path.write_text(
        "\n".join(
            (
                "# Frozen main-suite figure QA",
                "",
                "- Backend: Python/matplotlib only; no scientific experiment or bootstrap was rerun.",
                f"- Frozen experiment-tree hash: `{metadata['experiment_tree_sha256']}`.",
                "- Inputs: exactly six canonical comparison names across five RQ1 datasets.",
                "- WSC: `min_t mean_selected_seed(C_seed,t)`; it is the primary scalar validity summary.",
                "- Eligibility: point WSC >= 0.90. It is not based on the confidence interval.",
                "- Pareto status: descriptive point-estimate nondominance in WSC and average normalized width.",
                "- Intervals: WSC uses the frozen 10,000-draw complete-seed-vector percentile bootstrap; width and stagewise intervals use the frozen two-sided 95% Student-t intervals across selected seeds.",
                "- Replicates: Synthetic n=100 seeds; each clinical benchmark n=20 seeds.",
                "- Selection: all displayed rows selected in every prespecified run; curves remain formally conditional on selection.",
                "- Stagewise display: all six methods are retained; SPCI and PRC are visually muted, with all pointwise intervals retained in source data.",
                "- Baseline semantics: comparison labels preserve the frozen adapter names and do not imply a native sequential operating regime.",
                "- Typography: Times New Roman (Times/DejaVu Serif fallback), editable SVG text, TrueType PDF embedding.",
                "- Accessibility: colors are redundant with markers and line styles; target regions also use position and labels.",
                "- Work bundle: source CSV, SVG, PDF, 600-dpi LZW TIFF, 240-dpi PNG, contract, analysis, QA, and SHA-256 manifest.",
                "- Paper bundle: exactly two PDFs and no auxiliary files.",
                "- No raster source, crop, contrast, gamma, pseudo-color, stitching, or other image adjustment.",
                "- Claim boundary: no significance, finite-sample validity, superiority, clinical utility, or universal SOTA claim is inferred from these figures.",
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_render_manifest(path: Path, *, work_root: Path, paper_root: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "protocol": RENDER_PROTOCOL,
            "status": "complete",
            "work_files": {
                item.name: _file_contract(item)
                for item in sorted(work_root.iterdir())
                if item.is_file() and item.name != path.name
            },
            "paper_files": {
                item.name: _file_contract(item)
                for item in sorted(paper_root.iterdir())
                if item.is_file()
            },
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
        raise RuntimeError("paper output must contain exactly the two PDFs")
    for stem in (PARETO_STEM, PROFILE_STEM):
        svg = (work_root / f"{stem}.svg").read_text(encoding="utf-8")
        if "<text" not in svg or "Times New Roman" not in svg:
            raise RuntimeError(f"SVG text/font contract failed for {stem}")
        work_pdf = work_root / f"{stem}.pdf"
        paper_pdf = paper_root / f"{stem}.pdf"
        if not work_pdf.read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"work PDF header is malformed for {stem}")
        if _file_sha256(work_pdf) != _file_sha256(paper_pdf):
            raise RuntimeError(f"work and paper PDF bytes differ for {stem}")
        if not (work_root / f"{stem}.png").read_bytes().startswith(b"\x89PNG"):
            raise RuntimeError(f"PNG header is malformed for {stem}")
        if (work_root / f"{stem}.tiff").read_bytes()[:4] not in {
            b"II*\x00",
            b"MM\x00*",
        }:
            raise RuntimeError(f"TIFF header is malformed for {stem}")
    manifest = _read_json_mapping(work_root / "render_manifest.json")
    if manifest.get("protocol") != RENDER_PROTOCOL or manifest.get("status") != "complete":
        raise RuntimeError("render manifest contract differs")
    if set(_mapping(manifest.get("paper_files"), "paper files")) != PAPER_FILES:
        raise RuntimeError("render manifest paper file set differs")
    for group, root in (("work_files", work_root), ("paper_files", paper_root)):
        for name, contract in _mapping(manifest[group], group).items():
            _validate_file_contract(root / name, _mapping(contract, name))


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"malformed JSON: {path.name}") from error
    return _mapping(payload, path.name)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_contract(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _validate_file_contract(path: Path, contract: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError(f"manifest file is missing: {path.name}")
    if contract.get("bytes") != path.stat().st_size:
        raise RuntimeError(f"manifest byte count differs: {path.name}")
    if contract.get("sha256") != _file_sha256(path):
        raise RuntimeError(f"manifest hash differs: {path.name}")


if __name__ == "__main__":
    main()
