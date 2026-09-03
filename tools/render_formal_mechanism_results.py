"""Render frozen exact-identification and controlled-shift paper evidence.

This module is a deterministic reporting layer.  It validates the immutable
2026-08-25 artifacts, derives tidy source tables, and renders three publication
outputs without fitting a model, rolling out a policy, or changing SC-PCP.

Example
-------
conda run -n ucp python tools/render_formal_mechanism_results.py
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RENDER_PROTOCOL = "formal_mechanism_render_v1"

DEFAULT_EXACT_ROOT = ROOT / "results/work/exact_finite_mdp_20260825"
DEFAULT_CONTROLLED_ROOT = (
    ROOT / "results/work/controlled_six_method_confirm20_20260825"
)
DEFAULT_WORK_OUTPUT = ROOT / "results/work/formal_mechanism_report_20260826"
DEFAULT_PAPER_OUTPUT = ROOT / "results/paper_formal_mechanism_20260826"

EXACT_PROTOCOL = "exact_committed_prefix_finite_mdp_v1"
CONTROLLED_PROTOCOL = "controlled_performative_six_method_benchmark_v1"
FROZEN_SOURCE_TREE_SHA256 = (
    "7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643"
)
FROZEN_EXACT_SUMMARY_SHA256 = (
    "9e3ab3f8f1fadd42685068f735ccd58c55e9be7179d9ed8863f4b83d80e647d6"
)
FROZEN_CONTROLLED_SUMMARY_SHA256 = (
    "d8533ca5db0c6a3943fed1751f4d450846dcbff17df305a33197a105cc474670"
)
FROZEN_CONTROLLED_CONFIG_SHA256 = (
    "a9023266d72b6aff04ab446a3236097bd24d10dc1f15b504aeb688c0bbbf9979"
)

MECHANISMS = (
    "M0_no_feedback",
    "M1_current_only",
    "M2_history_only",
    "M3_full_feedback",
)
MECHANISM_LABELS = {
    "M0_no_feedback": "M0\nNo feedback",
    "M1_current_only": "M1\nCurrent only",
    "M2_history_only": "M2\nHistory only",
    "M3_full_feedback": "M3\nFull feedback",
}
ESTIMATORS = ("unweighted", "history_only", "current_only", "full_prefix")
ESTIMATOR_LABELS = {
    "unweighted": "Unweighted",
    "history_only": "History only",
    "current_only": "Current only",
    "full_prefix": "Full prefix",
}
IDENTIFICATION_CORRECT = {
    ("M0_no_feedback", "unweighted"),
    ("M0_no_feedback", "history_only"),
    ("M0_no_feedback", "current_only"),
    ("M0_no_feedback", "full_prefix"),
    ("M1_current_only", "current_only"),
    ("M1_current_only", "full_prefix"),
    ("M2_history_only", "history_only"),
    ("M2_history_only", "full_prefix"),
    ("M3_full_feedback", "full_prefix"),
}

GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
PRIMARY_GAMMA = -2.0
STRESS_GAMMA = -4.0
METHODS = ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")
SEEDS = tuple(range(91_000, 91_200, 10))
HORIZON = 12
LATE_STAGES = tuple(range(4, 12))
TARGET_COVERAGE = 0.90
BOOTSTRAP_RESAMPLES = 10_000

EXACT_FIGURE = "figure_exact_prefix_identification"
CONTROLLED_FIGURE = "figure_controlled_signed_all_six"
CONTROLLED_TABLE = "table_controlled_signed_all_six"
VISUAL_STEMS = (EXACT_FIGURE, CONTROLLED_FIGURE, CONTROLLED_TABLE)
PAPER_FILES = {f"{stem}.pdf" for stem in VISUAL_STEMS}
WORK_FILES = {
    *(f"{stem}.{suffix}" for stem in VISUAL_STEMS for suffix in ("svg", "tiff", "png")),
    "figure_exact_source_data.csv",
    "figure_controlled_source_data.csv",
    "figure_controlled_mechanism_source_data.csv",
    "analysis.json",
    "figure_qa.md",
    "render_manifest.json",
}

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

EXACT_SOURCE_FIELDS = (
    "mechanism",
    "mechanism_label",
    "estimator",
    "estimator_label",
    "role",
    "identification_correct",
    "instance_count",
    "mean_root_mean_squared_bias",
    "rmse_q05",
    "rmse_q95",
    "mean_maximum_absolute_bias",
    "max_abs_q05",
    "max_abs_q95",
    "source_json_path",
)
CONTROLLED_SOURCE_FIELDS = (
    "gamma",
    "gamma_role",
    "method",
    "wsc",
    "wsc_ci95_lower",
    "wsc_ci95_upper",
    "mean_coverage",
    "mean_coverage_ci95_lower",
    "mean_coverage_ci95_upper",
    "mean_normalized_width",
    "width_ci95_lower",
    "width_ci95_upper",
    "selected_seeds",
    "total_seeds",
    "selection_rate",
    "target_adaptation_trajectories_per_seed",
    "efficiency_winner_among_eligible",
    "source_json_path",
)
MECHANISM_SOURCE_FIELDS = (
    "gamma",
    "gamma_role",
    "metric",
    "unit",
    "estimate",
    "ci95_lower",
    "ci95_upper",
    "seed_count",
    "late_stages_zero_based",
    "method",
    "derivation",
    "source_seed_artifact_pattern",
)


@dataclass(frozen=True)
class RenderConfig:
    exact_root: Path = DEFAULT_EXACT_ROOT
    controlled_root: Path = DEFAULT_CONTROLLED_ROOT
    work_output: Path = DEFAULT_WORK_OUTPUT
    paper_output: Path = DEFAULT_PAPER_OUTPUT


@dataclass(frozen=True)
class FrozenArtifacts:
    exact_summary: Mapping[str, Any]
    controlled_summary: Mapping[str, Any]
    controlled_seed_rows: tuple[Mapping[str, Any], ...]
    input_contracts: Mapping[str, Mapping[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-root", type=Path, default=DEFAULT_EXACT_ROOT)
    parser.add_argument(
        "--controlled-root", type=Path, default=DEFAULT_CONTROLLED_ROOT
    )
    parser.add_argument("--work-output", type=Path, default=DEFAULT_WORK_OUTPUT)
    parser.add_argument("--paper-output", type=Path, default=DEFAULT_PAPER_OUTPUT)
    args = parser.parse_args()
    config = RenderConfig(
        exact_root=args.exact_root.resolve(),
        controlled_root=args.controlled_root.resolve(),
        work_output=args.work_output.resolve(),
        paper_output=args.paper_output.resolve(),
    )
    render_report(config)
    print(config.paper_output)


def render_report(config: RenderConfig) -> None:
    """Validate frozen inputs and atomically publish the report bundle."""

    if config.work_output.exists() or config.paper_output.exists():
        raise FileExistsError("work-output and paper-output must both be fresh")
    if config.work_output == config.paper_output:
        raise ValueError("work-output and paper-output must be different directories")

    artifacts = load_frozen_artifacts(config)
    exact_rows = build_exact_source_rows(artifacts.exact_summary)
    controlled_rows = build_controlled_source_rows(artifacts.controlled_summary)
    mechanism_rows = build_mechanism_source_rows(
        artifacts.controlled_seed_rows,
        artifacts.controlled_summary,
    )
    winners = efficiency_winners(controlled_rows)
    controlled_rows = [
        {
            **row,
            "efficiency_winner_among_eligible": (
                winners[float(row["gamma"])] == row["method"]
            ),
        }
        for row in controlled_rows
    ]

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
        write_source_csv(
            temporary_work / "figure_exact_source_data.csv",
            exact_rows,
            EXACT_SOURCE_FIELDS,
        )
        write_source_csv(
            temporary_work / "figure_controlled_source_data.csv",
            controlled_rows,
            CONTROLLED_SOURCE_FIELDS,
        )
        write_source_csv(
            temporary_work / "figure_controlled_mechanism_source_data.csv",
            mechanism_rows,
            MECHANISM_SOURCE_FIELDS,
        )
        _write_analysis(
            temporary_work / "analysis.json",
            config=config,
            artifacts=artifacts,
            exact_rows=exact_rows,
            controlled_rows=controlled_rows,
            mechanism_rows=mechanism_rows,
            winners=winners,
            staged_work=temporary_work,
        )
        _write_qa_notes(temporary_work / "figure_qa.md", winners=winners)

        apply_publication_style()
        export_visual(
            render_exact_identification_figure(exact_rows),
            title="Exact finite-MDP committed-prefix identification",
            work_stem=temporary_work / EXACT_FIGURE,
            paper_path=temporary_paper / f"{EXACT_FIGURE}.pdf",
        )
        export_visual(
            render_controlled_figure(controlled_rows, mechanism_rows),
            title="Controlled signed shift with all six canonical methods",
            work_stem=temporary_work / CONTROLLED_FIGURE,
            paper_path=temporary_paper / f"{CONTROLLED_FIGURE}.pdf",
        )
        export_visual(
            render_controlled_table(controlled_rows),
            title="Controlled signed benchmark complete results",
            work_stem=temporary_work / CONTROLLED_TABLE,
            paper_path=temporary_paper / f"{CONTROLLED_TABLE}.pdf",
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


def load_frozen_artifacts(config: RenderConfig) -> FrozenArtifacts:
    exact_summary, exact_contract = validate_exact_bundle(config.exact_root)
    controlled_summary, controlled_rows, controlled_contract = (
        validate_controlled_bundle(config.controlled_root)
    )
    return FrozenArtifacts(
        exact_summary=exact_summary,
        controlled_summary=controlled_summary,
        controlled_seed_rows=tuple(controlled_rows),
        input_contracts={
            "exact_finite_mdp": exact_contract,
            "controlled_all_six": controlled_contract,
        },
    )


def validate_exact_bundle(
    root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload_names = {"config.json", "metadata.json", "summary.json", "surfaces.npz"}
    _require_exact_root_entries(root, payload_names | {"manifest.json", "COMPLETE"})
    manifest = _read_json_mapping(root / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol") != EXACT_PROTOCOL
        or manifest.get("status") != "complete"
        or set(_mapping(manifest.get("files"), "exact manifest/files"))
        != payload_names
    ):
        raise RuntimeError("exact finite-MDP manifest contract differs")
    for name, contract in manifest["files"].items():
        _validate_file_contract(root / name, contract)
    complete = _read_json_mapping(root / "COMPLETE")
    if complete != {
        "status": "complete",
        "manifest_sha256": _file_sha256(root / "manifest.json"),
    }:
        raise RuntimeError("exact finite-MDP COMPLETE contract differs")
    metadata = _read_json_mapping(root / "metadata.json")
    if (
        metadata.get("protocol") != EXACT_PROTOCOL
        or metadata.get("schema_version") != 1
        or metadata.get("status") != "complete"
        or metadata.get("source_tree_sha256") != FROZEN_SOURCE_TREE_SHA256
        or metadata.get("canonical_method_unchanged") is not True
        or metadata.get("diagnostic_only") is not True
    ):
        raise RuntimeError("exact finite-MDP metadata contract differs")
    if _file_sha256(root / "summary.json") != FROZEN_EXACT_SUMMARY_SHA256:
        raise RuntimeError("exact finite-MDP frozen summary hash differs")
    summary = _read_json_mapping(root / "summary.json")
    _validate_exact_summary(summary)
    return summary, {
        "protocol": EXACT_PROTOCOL,
        "source_tree_sha256": FROZEN_SOURCE_TREE_SHA256,
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "summary_sha256": FROZEN_EXACT_SUMMARY_SHA256,
        "input_path": _project_path(root),
    }


def validate_controlled_bundle(
    root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], Mapping[str, Any]]:
    seed_names = {f"seed_{seed:05d}.json" for seed in SEEDS}
    _require_exact_root_entries(
        root, seed_names | {"metadata.json", "summary.json", "COMPLETE"}
    )
    if (root / "COMPLETE").read_text(encoding="utf-8") != "\n":
        raise RuntimeError("controlled COMPLETE marker differs")
    metadata = _read_json_mapping(root / "metadata.json")
    config_contract = _mapping(metadata.get("config_contract"), "controlled config")
    if (
        metadata.get("protocol") != CONTROLLED_PROTOCOL
        or metadata.get("role") != "fresh_confirmatory_canonical_baseline_comparison"
        or metadata.get("source_tree_sha256") != FROZEN_SOURCE_TREE_SHA256
        or tuple(metadata.get("methods", ())) != METHODS
        or tuple(metadata.get("seeds", ())) != SEEDS
        or tuple(float(x) for x in metadata.get("gammas", ())) != GAMMAS
        or metadata.get("calibration_trajectories") != 3_000
        or metadata.get("grid_trajectories") != 1_000
        or metadata.get("reference_trajectories") != 20_000
        or tuple(metadata.get("late_stages_zero_based", ())) != LATE_STAGES
        or metadata.get("importance_weights")
        != "uncapped_prefix_float64_log_stabilized"
        or metadata.get("guarantee_scope") != "asymptotic_per_step_marginal"
        or metadata.get("canonical_selector_mutation_permitted") is not False
        or config_contract.get("active_config_sha256")
        != FROZEN_CONTROLLED_CONFIG_SHA256
    ):
        raise RuntimeError("controlled metadata contract differs")
    if _file_sha256(root / "summary.json") != FROZEN_CONTROLLED_SUMMARY_SHA256:
        raise RuntimeError("controlled frozen summary hash differs")
    summary = _read_json_mapping(root / "summary.json")
    _validate_controlled_summary(summary)

    seed_rows: list[Mapping[str, Any]] = []
    seed_hashes: dict[str, str] = {}
    for seed in SEEDS:
        path = root / f"seed_{seed:05d}.json"
        payload = _read_json_mapping(path)
        if (
            payload.get("seed") != seed
            or payload.get("source_tree_sha256") != FROZEN_SOURCE_TREE_SHA256
            or payload.get("protocol") != CONTROLLED_PROTOCOL
            or payload.get("active_config_sha256") != FROZEN_CONTROLLED_CONFIG_SHA256
            or tuple(payload.get("methods", ())) != METHODS
        ):
            raise RuntimeError(f"controlled seed provenance differs: {path.name}")
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != len(GAMMAS):
            raise RuntimeError(f"controlled seed row count differs: {path.name}")
        for row, gamma in zip(rows, GAMMAS):
            _validate_controlled_seed_row(row, seed=seed, gamma=gamma)
            seed_rows.append(row)
        seed_hashes[path.name] = _file_sha256(path)

    _validate_controlled_aggregate_recovery(summary, seed_rows)
    return summary, seed_rows, {
        "protocol": CONTROLLED_PROTOCOL,
        "source_tree_sha256": FROZEN_SOURCE_TREE_SHA256,
        "active_config_sha256": FROZEN_CONTROLLED_CONFIG_SHA256,
        "summary_sha256": FROZEN_CONTROLLED_SUMMARY_SHA256,
        "seed_artifact_count": len(seed_hashes),
        "seed_artifact_hashes_sha256": _canonical_sha256(seed_hashes),
        "input_path": _project_path(root),
    }


def _validate_exact_summary(summary: Mapping[str, Any]) -> None:
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "complete"
        or summary.get("study") != "exact_committed_prefix_finite_mdp"
        or summary.get("population_exact") is not True
        or summary.get("canonical_method_unchanged") is not True
        or summary.get("diagnostic_only") is not True
        or summary.get("finite_sample_claim") is not False
    ):
        raise RuntimeError("exact summary claim contract differs")
    identification = _mapping(
        _mapping(summary.get("population_instance_audit"), "population audit").get(
            "identification"
        ),
        "population identification",
    )
    if set(identification) != set(MECHANISMS):
        raise RuntimeError("exact mechanism set differs")
    for mechanism in MECHANISMS:
        estimators = _mapping(identification[mechanism], mechanism)
        if set(estimators) != set(ESTIMATORS):
            raise RuntimeError(f"exact estimator set differs for {mechanism}")
        for estimator in ESTIMATORS:
            cell = _mapping(estimators[estimator], f"{mechanism}/{estimator}")
            for metric_name in ("root_mean_squared", "maximum_absolute"):
                metric = _mapping(cell.get(metric_name), metric_name)
                if metric.get("count") != 500:
                    raise RuntimeError("exact identification requires 500 instances")
                for field in (
                    "mean",
                    "minimum",
                    "maximum",
                    "median",
                    "q05",
                    "q95",
                ):
                    _finite_number(metric.get(field), f"{mechanism}/{estimator}/{field}")


def _validate_controlled_summary(summary: Mapping[str, Any]) -> None:
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
    bootstrap = _mapping(summary.get("bootstrap"), "controlled bootstrap")
    if (
        bootstrap.get("resamples") != BOOTSTRAP_RESAMPLES
        or set(bootstrap.get("gamma_seeds", {})) != {f"{gamma:g}" for gamma in GAMMAS}
    ):
        raise RuntimeError("controlled bootstrap contract differs")
    aggregates = summary.get("aggregates")
    if not isinstance(aggregates, list) or len(aggregates) != len(GAMMAS):
        raise RuntimeError("controlled aggregate count differs")
    for aggregate, gamma in zip(aggregates, GAMMAS):
        if (
            float(aggregate.get("gamma")) != gamma
            or aggregate.get("n_seeds") != len(SEEDS)
            or set(aggregate.get("methods", {})) != set(METHODS)
        ):
            raise RuntimeError(f"controlled aggregate contract differs at gamma={gamma:g}")
        for method in METHODS:
            cell = _mapping(aggregate["methods"][method], f"{gamma}/{method}")
            for field in (
                "target_marginal_worst_coverage",
                "target_mean_coverage",
                "mean_target_normalized_width",
                "selection_rate",
            ):
                _finite_number(cell.get(field), f"{gamma}/{method}/{field}")
            for field in (
                "target_wsc_ci95",
                "target_mean_coverage_ci95",
                "mean_target_normalized_width_ci95",
                "selection_rate_ci95",
            ):
                _finite_interval(cell.get(field), f"{gamma}/{method}/{field}")
            coverage = _finite_vector(
                cell.get("target_coverage_by_stage"),
                length=HORIZON,
                label=f"{gamma}/{method}/coverage",
            )
            if not math.isclose(
                float(coverage.min()),
                float(cell["target_marginal_worst_coverage"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            ):
                raise RuntimeError("stored WSC does not equal min stage mean coverage")


def _validate_controlled_seed_row(
    row: Mapping[str, Any], *, seed: int, gamma: float
) -> None:
    if (
        row.get("seed") != seed
        or float(row.get("gamma")) != gamma
        or set(row.get("methods", {})) != set(METHODS)
    ):
        raise RuntimeError("controlled seed/gamma/method contract differs")
    for method in METHODS:
        values = _mapping(row["methods"][method], f"seed {seed}/{gamma}/{method}")
        if values.get("selection_available") is not True:
            raise RuntimeError("formal controlled artifact must select all methods")
        for field in (
            "radii",
            "source_coverage",
            "target_coverage",
            "coverage_gap",
            "source_q90",
            "target_q90",
            "q90_relative_gap",
            "target_normalized_width",
            "prefix_ess_fraction",
            "maximum_normalized_weight_share",
            "policy_tv_on_source_states",
        ):
            _finite_vector(values.get(field), length=HORIZON, label=field)
        source = np.asarray(values["source_coverage"], dtype=np.float64)
        target = np.asarray(values["target_coverage"], dtype=np.float64)
        gap = np.asarray(values["coverage_gap"], dtype=np.float64)
        if not np.allclose(gap, target - source, atol=1e-12, rtol=0.0):
            raise RuntimeError("controlled coverage-gap identity differs")
        source_q90 = np.asarray(values["source_q90"], dtype=np.float64)
        target_q90 = np.asarray(values["target_q90"], dtype=np.float64)
        q90_gap = np.asarray(values["q90_relative_gap"], dtype=np.float64)
        expected_q90_gap = target_q90 / np.maximum(source_q90, 1e-8) - 1.0
        # These three vectors were serialized from float32 tensors separately;
        # allow one float32 rounding unit while still rejecting altered data.
        if not np.allclose(q90_gap, expected_q90_gap, atol=1e-7, rtol=0.0):
            raise RuntimeError("controlled Q90-gap identity differs")


def _validate_controlled_aggregate_recovery(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    by_gamma = {float(cell["gamma"]): cell for cell in summary["aggregates"]}
    for gamma in GAMMAS:
        selected = sorted(
            (row for row in rows if float(row["gamma"]) == gamma),
            key=lambda row: int(row["seed"]),
        )
        if tuple(int(row["seed"]) for row in selected) != SEEDS:
            raise RuntimeError("controlled seed ordering/recovery differs")
        for method in METHODS:
            coverage = np.asarray(
                [row["methods"][method]["target_coverage"] for row in selected],
                dtype=np.float64,
            )
            width = np.asarray(
                [
                    np.mean(row["methods"][method]["target_normalized_width"])
                    for row in selected
                ],
                dtype=np.float64,
            )
            recovered_wsc = float(coverage.mean(axis=0).min())
            recovered_mean = float(coverage.mean())
            recovered_width = float(width.mean())
            stored = by_gamma[gamma]["methods"][method]
            for label, recovered, expected in (
                ("WSC", recovered_wsc, stored["target_marginal_worst_coverage"]),
                ("MeanCov", recovered_mean, stored["target_mean_coverage"]),
                ("width", recovered_width, stored["mean_target_normalized_width"]),
            ):
                if not math.isclose(
                    recovered, float(expected), rel_tol=0.0, abs_tol=1e-14
                ):
                    raise RuntimeError(
                        f"controlled per-seed recovery differs for {gamma}/{method}/{label}"
                    )


def build_exact_source_rows(
    summary: Mapping[str, Any]
) -> list[dict[str, object]]:
    identification = summary["population_instance_audit"]["identification"]
    rows: list[dict[str, object]] = []
    for mechanism in MECHANISMS:
        for estimator in ESTIMATORS:
            cell = identification[mechanism][estimator]
            rmse = cell["root_mean_squared"]
            maximum = cell["maximum_absolute"]
            rows.append(
                {
                    "mechanism": mechanism,
                    "mechanism_label": MECHANISM_LABELS[mechanism].replace("\n", " "),
                    "estimator": estimator,
                    "estimator_label": ESTIMATOR_LABELS[estimator],
                    "role": "structural_diagnostic_not_baseline",
                    "identification_correct": (mechanism, estimator)
                    in IDENTIFICATION_CORRECT,
                    "instance_count": int(rmse["count"]),
                    "mean_root_mean_squared_bias": float(rmse["mean"]),
                    "rmse_q05": float(rmse["q05"]),
                    "rmse_q95": float(rmse["q95"]),
                    "mean_maximum_absolute_bias": float(maximum["mean"]),
                    "max_abs_q05": float(maximum["q05"]),
                    "max_abs_q95": float(maximum["q95"]),
                    "source_json_path": "results/work/exact_finite_mdp_20260825/summary.json",
                }
            )
    return rows


def build_controlled_source_rows(
    summary: Mapping[str, Any]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_gamma = {float(cell["gamma"]): cell for cell in summary["aggregates"]}
    for gamma in GAMMAS:
        aggregate = by_gamma[gamma]
        for method in METHODS:
            cell = aggregate["methods"][method]
            rows.append(
                {
                    "gamma": gamma,
                    "gamma_role": _gamma_role(gamma),
                    "method": method,
                    "wsc": float(cell["target_marginal_worst_coverage"]),
                    "wsc_ci95_lower": float(cell["target_wsc_ci95"][0]),
                    "wsc_ci95_upper": float(cell["target_wsc_ci95"][1]),
                    "mean_coverage": float(cell["target_mean_coverage"]),
                    "mean_coverage_ci95_lower": float(
                        cell["target_mean_coverage_ci95"][0]
                    ),
                    "mean_coverage_ci95_upper": float(
                        cell["target_mean_coverage_ci95"][1]
                    ),
                    "mean_normalized_width": float(
                        cell["mean_target_normalized_width"]
                    ),
                    "width_ci95_lower": float(
                        cell["mean_target_normalized_width_ci95"][0]
                    ),
                    "width_ci95_upper": float(
                        cell["mean_target_normalized_width_ci95"][1]
                    ),
                    "selected_seeds": int(cell["selected_seeds"]),
                    "total_seeds": int(cell["total_seeds"]),
                    "selection_rate": float(cell["selection_rate"]),
                    "target_adaptation_trajectories_per_seed": int(
                        cell["target_adaptation_trajectories_per_seed"]
                    ),
                    "efficiency_winner_among_eligible": False,
                    "source_json_path": (
                        "results/work/controlled_six_method_confirm20_20260825/summary.json"
                    ),
                }
            )
    return rows


def build_mechanism_source_rows(
    seed_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> list[dict[str, object]]:
    aggregates = {float(cell["gamma"]): cell for cell in summary["aggregates"]}
    rows: list[dict[str, object]] = []
    for gamma in GAMMAS:
        selected = sorted(
            (row for row in seed_rows if float(row["gamma"]) == gamma),
            key=lambda row: int(row["seed"]),
        )
        if len(selected) != len(SEEDS):
            raise RuntimeError(f"mechanism source requires 20 seeds at gamma={gamma:g}")
        coverage_gap = np.asarray(
            [row["methods"]["Standard CP"]["coverage_gap"] for row in selected],
            dtype=np.float64,
        )[:, LATE_STAGES].mean(axis=1)
        q90_gap = np.asarray(
            [row["methods"]["Standard CP"]["q90_relative_gap"] for row in selected],
            dtype=np.float64,
        )[:, LATE_STAGES].mean(axis=1)
        bootstrap_seed = int(aggregates[gamma]["bootstrap_seed"])
        rng = np.random.default_rng(bootstrap_seed)
        uniforms = rng.random(size=(BOOTSTRAP_RESAMPLES, len(SEEDS)))
        indices = np.floor(uniforms * len(SEEDS)).astype(np.int64)
        for metric, unit, values in (
            ("standard_same_radius_late_coverage_gap", "pp", coverage_gap * 100.0),
            ("standard_late_q90_relative_shift", "%", q90_gap * 100.0),
        ):
            draws = values[indices].mean(axis=1)
            rows.append(
                {
                    "gamma": gamma,
                    "gamma_role": _gamma_role(gamma),
                    "metric": metric,
                    "unit": unit,
                    "estimate": float(values.mean()),
                    "ci95_lower": float(np.quantile(draws, 0.025)),
                    "ci95_upper": float(np.quantile(draws, 0.975)),
                    "seed_count": len(SEEDS),
                    "late_stages_zero_based": "4,5,6,7,8,9,10,11",
                    "method": "Standard CP",
                    "derivation": (
                        "per-seed late-stage mean; frozen per-gamma shared seed-vector "
                        "percentile bootstrap (10000 resamples)"
                    ),
                    "source_seed_artifact_pattern": (
                        "results/work/controlled_six_method_confirm20_20260825/"
                        "seed_91xxx.json"
                    ),
                }
            )
    return rows


def efficiency_winners(
    rows: Sequence[Mapping[str, object]],
) -> dict[float, str]:
    winners: dict[float, str] = {}
    for gamma in GAMMAS:
        eligible = [
            row
            for row in rows
            if float(row["gamma"]) == gamma
            and float(row["wsc"]) >= TARGET_COVERAGE
            and float(row["selection_rate"]) >= 0.95
        ]
        if not eligible:
            raise RuntimeError(f"no point-estimate eligible method at gamma={gamma:g}")
        winner = min(eligible, key=lambda row: float(row["mean_normalized_width"]))
        winners[gamma] = str(winner["method"])
    return winners


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
            "legend.fontsize": 6.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "svg.hashsalt": "scpcp-formal-mechanism-v1",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def render_exact_identification_figure(
    rows: Sequence[Mapping[str, object]],
) -> plt.Figure:
    by_key = {(row["mechanism"], row["estimator"]): row for row in rows}
    matrix = np.asarray(
        [
            [
                float(by_key[(mechanism, estimator)]["mean_maximum_absolute_bias"])
                for mechanism in MECHANISMS
            ]
            for estimator in ESTIMATORS
        ],
        dtype=np.float64,
    )
    cmap = LinearSegmentedColormap.from_list(
        "identification_bias",
        ("#F7FBFF", "#C7DCEF", "#78A8CF", "#315F87"),
    )
    figure, axis = plt.subplots(figsize=(7.20, 3.35), constrained_layout=True)
    image = axis.imshow(matrix, cmap=cmap, vmin=0.0, vmax=0.40, aspect="auto")
    axis.set_xticks(range(len(MECHANISMS)), [MECHANISM_LABELS[x] for x in MECHANISMS])
    axis.set_yticks(range(len(ESTIMATORS)), [ESTIMATOR_LABELS[x] for x in ESTIMATORS])
    axis.xaxis.tick_top()
    axis.tick_params(length=0, pad=5)
    axis.set_xlabel("Feedback mechanism", labelpad=8, fontweight="bold")
    axis.xaxis.set_label_position("top")
    axis.set_ylabel("Transport diagnostic", fontweight="bold")
    for row_index, estimator in enumerate(ESTIMATORS):
        for column_index, mechanism in enumerate(MECHANISMS):
            value = matrix[row_index, column_index]
            correct = (mechanism, estimator) in IDENTIFICATION_CORRECT
            label = "0*" if value < 1e-12 else f"{value:.3f}"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold" if correct else "normal",
                color="white" if value > 0.24 else "#202020",
            )
            if correct:
                axis.add_patch(
                    Rectangle(
                        (column_index - 0.49, row_index - 0.49),
                        0.98,
                        0.98,
                        fill=False,
                        edgecolor="#117A65",
                        linewidth=1.8,
                    )
                )
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.032, pad=0.035)
    colorbar.set_label("Mean maximum absolute bias", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.2, width=0.6, length=2.5)
    axis.set_title(
        "Exact population identification across 500 paired finite-MDP instances",
        loc="left",
        pad=12,
        fontweight="bold",
    )
    axis.text(
        0.0,
        -0.20,
        "* Teal outline: identification-correct (<1e−12). Structural variants are diagnostics, not baseline methods.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color="#333333",
    )
    return figure


def render_controlled_figure(
    controlled_rows: Sequence[Mapping[str, object]],
    mechanism_rows: Sequence[Mapping[str, object]],
) -> plt.Figure:
    figure, axes = plt.subplots(
        1, 3, figsize=(7.20, 3.30), constrained_layout=True
    )
    axis_a, axis_b, axis_c = axes
    x = np.asarray(GAMMAS, dtype=np.float64)

    mechanism_style = {
        "standard_same_radius_late_coverage_gap": (
            "Same-radius coverage gap",
            "#B54A42",
            "o",
        ),
        "standard_late_q90_relative_shift": ("Target Q90 shift", "#7560A8", "D"),
    }
    for metric, (label, color, marker) in mechanism_style.items():
        selected = sorted(
            (row for row in mechanism_rows if row["metric"] == metric),
            key=lambda row: GAMMAS.index(float(row["gamma"])),
        )
        estimate = np.asarray([float(row["estimate"]) for row in selected])
        lower = np.asarray([float(row["ci95_lower"]) for row in selected])
        upper = np.asarray([float(row["ci95_upper"]) for row in selected])
        axis_a.errorbar(
            x,
            estimate,
            yerr=np.vstack((estimate - lower, upper - estimate)),
            label=label,
            color=color,
            marker=marker,
            markersize=3.7,
            linewidth=1.25,
            capsize=1.8,
        )
    axis_a.axhline(0.0, color="#555555", linewidth=0.75, linestyle=(0, (3, 2)))
    axis_a.set_ylabel("Shift (coverage pp; Q90 %)")
    axis_a.set_title("Mechanism at Standard CP radii", loc="left", fontweight="bold")
    axis_a.legend(loc="upper right", fontsize=5.8, handlelength=2.0)

    offsets = dict(zip(METHODS, np.linspace(-0.12, 0.12, len(METHODS))))
    handles: list[Any] = []
    labels: list[str] = []
    for method in METHODS:
        selected = sorted(
            (row for row in controlled_rows if row["method"] == method),
            key=lambda row: GAMMAS.index(float(row["gamma"])),
        )
        width = 1.55 if method == "SC-PCP" else (1.25 if method in {"Standard CP", "MFCS"} else 1.0)
        wsc = np.asarray([float(row["wsc"]) * 100.0 for row in selected])
        wsc_lower = np.asarray(
            [float(row["wsc_ci95_lower"]) * 100.0 for row in selected]
        )
        wsc_upper = np.asarray(
            [float(row["wsc_ci95_upper"]) * 100.0 for row in selected]
        )
        result = axis_b.errorbar(
            x + offsets[method],
            wsc,
            yerr=np.vstack((wsc - wsc_lower, wsc_upper - wsc)),
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=width,
            markersize=3.5,
            capsize=1.6,
            alpha=0.98,
            label=method,
        )
        handles.append(result.lines[0])
        labels.append(method)

        mean_width = np.asarray(
            [float(row["mean_normalized_width"]) for row in selected]
        )
        width_lower = np.asarray(
            [float(row["width_ci95_lower"]) for row in selected]
        )
        width_upper = np.asarray(
            [float(row["width_ci95_upper"]) for row in selected]
        )
        axis_c.errorbar(
            x + offsets[method],
            mean_width,
            yerr=np.vstack((mean_width - width_lower, width_upper - mean_width)),
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=width,
            markersize=3.5,
            capsize=1.6,
            alpha=0.98,
        )

    axis_b.axhline(
        TARGET_COVERAGE * 100.0,
        color="#333333",
        linewidth=0.8,
        linestyle=(0, (3, 2)),
    )
    axis_b.set_ylabel("Marginal WSC (%)")
    axis_b.set_ylim(85.5, 93.5)
    axis_b.set_title("Validity across six methods", loc="left", fontweight="bold")
    axis_c.set_ylabel("Mean normalized width")
    axis_c.set_ylim(bottom=1.35)
    axis_c.set_title("Coverage–efficiency cost", loc="left", fontweight="bold")

    for label, axis in zip("abc", axes):
        axis.text(
            -0.18,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        axis.axvspan(-2.25, -1.75, color="#E8B84A", alpha=0.11, zorder=-3)
        axis.set_xticks(x, ["−4", "−2", "0", "+2", "+4"])
        axis.set_xlabel(r"Signed transition alignment $\gamma$")
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=-4)
        axis.text(
            -2.0,
            0.985,
            "primary",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.4,
            color="#7A5A00",
        )
        axis.text(
            -4.0,
            0.985,
            "stress",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.4,
            color="#555555",
        )
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=6,
        fontsize=6.0,
        handlelength=2.1,
        columnspacing=1.0,
    )
    return figure


def render_controlled_table(
    controlled_rows: Sequence[Mapping[str, object]],
) -> plt.Figure:
    winners = efficiency_winners(controlled_rows)
    by_key = {
        (float(row["gamma"]), str(row["method"])): row for row in controlled_rows
    }
    table_rows: list[list[str]] = []
    winner_indices: set[int] = set()
    gamma_groups: list[int] = []
    for gamma_index, gamma in enumerate(GAMMAS):
        for method_index, method in enumerate(METHODS):
            row = by_key[(gamma, method)]
            gamma_text = _format_gamma(gamma) if method_index == 0 else ""
            table_rows.append(
                [
                    gamma_text,
                    method,
                    _format_percent_interval(
                        row["wsc"], row["wsc_ci95_lower"], row["wsc_ci95_upper"]
                    ),
                    _format_percent_interval(
                        row["mean_coverage"],
                        row["mean_coverage_ci95_lower"],
                        row["mean_coverage_ci95_upper"],
                    ),
                    _format_number_interval(
                        row["mean_normalized_width"],
                        row["width_ci95_lower"],
                        row["width_ci95_upper"],
                    ),
                    f"{int(row['selected_seeds'])}/{int(row['total_seeds'])}",
                ]
            )
            gamma_groups.append(gamma_index)
            if winners[gamma] == method:
                winner_indices.add(len(table_rows) - 1)

    figure, axis = plt.subplots(figsize=(7.20, 8.45))
    figure.subplots_adjust(left=0.01, right=0.99, top=0.985, bottom=0.01)
    axis.axis("off")
    axis.set_title(
        "Controlled signed benchmark: all six canonical methods",
        loc="left",
        fontsize=9,
        fontweight="bold",
        pad=12,
    )
    axis.text(
        0.0,
        0.978,
        r"20 prespecified seeds per $\gamma$; target marginal coverage = 90%",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="#444444",
    )
    table = axis.table(
        cellText=table_rows,
        colLabels=(
            r"$\gamma$",
            "Method",
            "WSC [95% CI]",
            "MeanCov [95% CI]",
            "Normalized width [95% CI]",
            "Selection",
        ),
        colLoc="center",
        cellLoc="center",
        colWidths=(0.07, 0.15, 0.21, 0.21, 0.25, 0.10),
        bbox=(0.0, 0.105, 1.0, 0.84),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.65)
    for column in range(6):
        cell = table[(0, column)]
        cell.set_facecolor("#334E68")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")
        cell.set_linewidth(0.5)
    group_colors = ("#F4F7FA", "#FFFFFF")
    for row_index, group_index in enumerate(gamma_groups, start=1):
        for column in range(6):
            cell = table[(row_index, column)]
            cell.set_facecolor(group_colors[group_index % 2])
            cell.set_edgecolor("#D7DEE5")
            cell.set_linewidth(0.35)
            if column == 1:
                cell.set_text_props(ha="left")
        if table_rows[row_index - 1][1] == "SC-PCP":
            table[(row_index, 1)].get_text().set_color(METHOD_COLORS["SC-PCP"])
        if row_index - 1 in winner_indices:
            table[(row_index, 4)].set_facecolor("#DDECF8")
            table[(row_index, 4)].set_text_props(fontweight="bold")
        if (row_index - 1) % len(METHODS) == 0:
            for column in range(6):
                table[(row_index, column)].set_linewidth(0.75)
                table[(row_index, column)].set_edgecolor("#9AA9B5")
    axis.text(
        0.0,
        0.075,
        r"WSC = $\min_t\,\mathrm{mean}_{seed}(C_{seed,t})$. WSC intervals: seed-vector percentile bootstrap; MeanCov and width intervals: Student-t across selected seeds.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
    )
    axis.text(
        0.0,
        0.047,
        "Eligibility uses point WSC ≥ 90% and Selection ≥ 95%; bold width is the narrowest eligible method within each γ. γ = −2 is primary; γ = −4 is stress.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
    )
    axis.text(
        0.0,
        0.019,
        "Coverage and efficiency trade off across settings; this table does not support a universal-dominance or universal-SOTA claim.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color="#6A3030",
    )
    return figure


def export_visual(
    figure: plt.Figure,
    *,
    title: str,
    work_stem: Path,
    paper_path: Path,
    tiff_dpi: int = 600,
    png_dpi: int = 240,
) -> None:
    pdf_metadata = {
        "Title": title,
        "Creator": "SC-PCP formal mechanism renderer",
        "CreationDate": None,
        "ModDate": None,
    }
    svg_metadata = {
        "Title": title,
        "Creator": "SC-PCP formal mechanism renderer",
        "Date": None,
    }
    figure.savefig(
        work_stem.with_suffix(".svg"),
        format="svg",
        bbox_inches="tight",
        metadata=svg_metadata,
    )
    figure.savefig(
        paper_path,
        format="pdf",
        bbox_inches="tight",
        metadata=pdf_metadata,
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
        metadata={"Software": "SC-PCP formal mechanism renderer"},
    )
    plt.close(figure)


def write_source_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_analysis(
    path: Path,
    *,
    config: RenderConfig,
    artifacts: FrozenArtifacts,
    exact_rows: Sequence[Mapping[str, object]],
    controlled_rows: Sequence[Mapping[str, object]],
    mechanism_rows: Sequence[Mapping[str, object]],
    winners: Mapping[float, str],
    staged_work: Path,
) -> None:
    source_files = {
        "exact": staged_work / "figure_exact_source_data.csv",
        "controlled": staged_work / "figure_controlled_source_data.csv",
        "mechanism": staged_work / "figure_controlled_mechanism_source_data.csv",
    }
    payload = {
        "schema_version": 1,
        "protocol": RENDER_PROTOCOL,
        "status": "complete",
        "backend": "Python/matplotlib only",
        "input_contracts": artifacts.input_contracts,
        "figure_contracts": {
            EXACT_FIGURE: {
                "core_conclusion": (
                    "The required likelihood-ratio factors follow the causal feedback "
                    "structure: current-only is sufficient in M1, history-only in M2, "
                    "and only full prefix is identification-correct in M3."
                ),
                "archetype": "quantitative_grid",
                "design_canvas_inches": [7.20, 3.35],
                "panel_map": {
                    "a": "M0--M3 by four structural transport diagnostics"
                },
                "hero_metric": "mean maximum absolute population identification bias",
                "n": "500 paired finite-MDP problem instances",
                "source_data": source_files["exact"].name,
                "source_data_sha256": _file_sha256(source_files["exact"]),
                "source_row_count": len(exact_rows),
            },
            CONTROLLED_FIGURE: {
                "core_conclusion": (
                    "Signed prediction-mediated treatment shifts move the target score law "
                    "in both directions, producing a coverage--width trade-off rather than "
                    "universal method dominance."
                ),
                "archetype": "quantitative_grid",
                "design_canvas_inches": [7.20, 3.30],
                "panel_map": {
                    "a": "same-radius Standard CP coverage gap and target Q90 shift",
                    "b": "all-six marginal WSC with 95% intervals",
                    "c": "all-six mean normalized width with 95% intervals",
                },
                "n": "20 complete prespecified seed-stage vectors per gamma",
                "source_data": [
                    source_files["controlled"].name,
                    source_files["mechanism"].name,
                ],
                "source_data_sha256": {
                    name: _file_sha256(source) for name, source in source_files.items()
                    if name in {"controlled", "mechanism"}
                },
                "source_row_count": {
                    "controlled": len(controlled_rows),
                    "mechanism": len(mechanism_rows),
                },
            },
            CONTROLLED_TABLE: {
                "core_conclusion": (
                    "Complete all-six results must be judged by validity first and width "
                    "among point-eligible methods, not by maximizing coverage alone."
                ),
                "design_canvas_inches": [7.20, 8.45],
                "eligibility_rule": "point WSC >= 0.90 and Selection >= 0.95",
                "bold_rule": "minimum mean normalized width among eligible methods",
                "efficiency_winners": {f"{gamma:g}": winners[gamma] for gamma in GAMMAS},
                "source_data": source_files["controlled"].name,
            },
        },
        "export_contract": {
            "paper_directory": _project_path(config.paper_output),
            "paper_files": sorted(PAPER_FILES),
            "paper_format": "PDF only; TrueType fonts",
            "work_formats": ["editable SVG", "600-dpi TIFF", "240-dpi PNG"],
            "font": "Times New Roman with Times/DejaVu Serif fallback",
            "svg_text": "text elements retained",
            "determinism": (
                "fixed source hashes, row/category order, palette, markers, svg.hashsalt, "
                "and frozen per-gamma bootstrap seeds; no model fit or rollout"
            ),
        },
        "claim_boundary": (
            "Exact-MDP rows are structural diagnostics, not baseline methods. Controlled "
            "results support a signed benchmark-specific mechanism and trade-off, not "
            "finite-sample distribution-free validity, universal dominance, or universal SOTA."
        ),
    }
    _write_json(path, payload)


def _write_qa_notes(path: Path, *, winners: Mapping[float, str]) -> None:
    winner_text = ", ".join(
        f"gamma={gamma:g}: {winners[gamma]}" for gamma in GAMMAS
    )
    path.write_text(
        "\n".join(
            (
                "# Formal mechanism figure QA",
                "",
                "- Backend: Python/matplotlib only; no model fit, policy rollout, or scientific seed run.",
                f"- Frozen experiment source hash: `{FROZEN_SOURCE_TREE_SHA256}`.",
                f"- Frozen exact summary hash: `{FROZEN_EXACT_SUMMARY_SHA256}`.",
                f"- Frozen controlled summary hash: `{FROZEN_CONTROLLED_SUMMARY_SHA256}`.",
                "- Figure archetype: quantitative grid; each panel carries a distinct estimand.",
                "- Design canvases: exact heatmap 7.20 x 3.35 in; controlled figure 7.20 x 3.30 in; complete table 7.20 x 8.45 in (tight vector bounds retained).",
                "- Typography: Times New Roman, editable SVG text, TrueType PDF embedding.",
                "- Accessibility: restrained non-rainbow palette plus marker/line-style redundancy.",
                "- Exact heatmap n: 500 paired finite-MDP instances; displayed metric is the mean maximum absolute population bias over each coverage surface.",
                "- Exact heatmap zero cells use <1e-12 display tolerance; source CSV retains full-precision values and RMSE diagnostics.",
                "- Structural unweighted/history-only/current-only/full-prefix rows are diagnostics, not paper baselines.",
                "- Controlled n: 20 prespecified seeds per gamma, 12 stages, late mechanism stages 4--11.",
                "- Controlled WSC intervals are the frozen 10,000-resample seed-vector percentile bootstrap; width and MeanCov intervals are frozen Student-t intervals.",
                "- Panel-a mechanism intervals are deterministic post-processing of stored seed rows using each frozen per-gamma bootstrap seed and shared resampling convention.",
                "- Same-radius coverage gap is target minus source at Standard CP radii (percentage points); Q90 shift is (target/source - 1) in percent.",
                "- Gamma=-2 is the frozen primary cell; gamma=-4 is the overlap-stress endpoint.",
                "- Eligibility is based on point WSC >= 0.90 and Selection >= 0.95; CI crossing is not used for bolding.",
                f"- Narrowest eligible widths: {winner_text}.",
                "- All six canonical methods appear in the controlled figure, table, and source CSV; there are no ablation rows.",
                "- The figure and table do not assert universal dominance or universal SOTA.",
                "- Paper output contains PDFs only; work output contains source CSV, analysis, QA, editable SVG, TIFF, and PNG previews.",
                "- No raster-image manipulation; all visuals are generated directly from frozen numerical artifacts.",
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
        raise RuntimeError("paper output must contain exactly the three PDFs")
    for stem in VISUAL_STEMS:
        svg = (work_root / f"{stem}.svg").read_text(encoding="utf-8")
        if "<text" not in svg or "Times New Roman" not in svg:
            raise RuntimeError(f"SVG text/font contract differs: {stem}")
        if not (paper_root / f"{stem}.pdf").read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"PDF header is malformed: {stem}")
        if not (work_root / f"{stem}.png").read_bytes().startswith(b"\x89PNG"):
            raise RuntimeError(f"PNG header is malformed: {stem}")
        tiff_header = (work_root / f"{stem}.tiff").read_bytes()[:4]
        if tiff_header not in {b"II*\x00", b"MM\x00*"}:
            raise RuntimeError(f"TIFF header is malformed: {stem}")
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


def _gamma_role(gamma: float) -> str:
    if gamma == PRIMARY_GAMMA:
        return "primary"
    if gamma == STRESS_GAMMA:
        return "stress"
    if gamma == 0.0:
        return "null"
    return "signed_secondary"


def _format_gamma(gamma: float) -> str:
    if gamma > 0:
        return f"+{gamma:g}"
    return f"{gamma:g}".replace("-", "−")


def _format_percent_interval(point: object, lower: object, upper: object) -> str:
    return f"{100 * float(point):.2f} [{100 * float(lower):.2f}, {100 * float(upper):.2f}]"


def _format_number_interval(point: object, lower: object, upper: object) -> str:
    return f"{float(point):.3f} [{float(lower):.3f}, {float(upper):.3f}]"


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def _finite_interval(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeError(f"{label} is not a two-sided interval")
    lower = _finite_number(value[0], f"{label}/lower")
    upper = _finite_number(value[1], f"{label}/upper")
    if lower > upper:
        raise RuntimeError(f"{label} is reversed")
    return lower, upper


def _finite_vector(value: object, *, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise RuntimeError(f"{label} must have finite length {length}")
    return array


def _require_exact_root_entries(root: Path, expected: set[str]) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {root}")
    observed = {path.name for path in root.iterdir()}
    if observed != expected:
        raise RuntimeError(
            f"artifact entries differ for {root}: expected {sorted(expected)}, found {sorted(observed)}"
        )


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, str(path))


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
