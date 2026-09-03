"""Render the frozen theorem-facing and robustness-facing SC-PCP results.

The four experiment bundles are immutable inputs.  This renderer validates their
completion and hash contracts, extracts tidy source data from the frozen summaries,
and writes two publication figures without rerunning an experiment or bootstrap.

Example
-------
conda run -n ucp python tools/render_theorem_robustness_results.py
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
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_TREE_SHA256 = (
    "296569d628875de774cb5012004c345d624653c1f4ecd2d3b6ff02e292f99226"
)
RENDER_PROTOCOL = "theorem_robustness_render_v1"

DEFAULT_HORIZON_ROOT = ROOT / "results/work/horizon_overlap_v1"
DEFAULT_RQ6_ROOT = ROOT / "results/work/rq6_ncal_convergence_v1"
DEFAULT_PROPENSITY_ROOT = ROOT / "results/work/propensity_robustness_v1"
DEFAULT_STRICT_ROOT = ROOT / "results/work/strict_split_robustness_v1_20260826"
DEFAULT_WORK_OUTPUT = ROOT / "results/work/theorem_robustness_report_20260826"
DEFAULT_PAPER_OUTPUT = ROOT / "results/paper_theorem_robustness_20260826"

HORIZON_PROTOCOL = "finite_mdp_horizon_overlap_v1"
RQ6_PROTOCOL = "rq6_ncal_convergence_v1"
PROPENSITY_PROTOCOL = "propensity_robustness_v1"
STRICT_PROTOCOL = "strict_split_robustness_v1"

HORIZONS = (2, 4, 8, 12, 20)
NOMINAL_POLICY_TVS = (0.0, 0.025, 0.05, 0.10, 0.15)
RQ5_METHODS = (
    "Standard CP",
    "History-only Prefix-IW",
    "Current-only IW",
    "SC-PCP",
)
N_CALIBRATION = (250, 500, 1_000, 2_000, 5_000, 10_000)
PROPENSITY_ARMS = (
    "oracle",
    "correct_multinomial",
    "misspecified_reduced_state",
)
STRICT_SETTINGS = (
    "synthetic_main",
    "mimic_iv",
    "controlled_gamma_minus_2",
)

THEORY_FIGURE = "figure_theory_diagnostics"
ROBUSTNESS_FIGURE = "figure_robustness_audits"
PAPER_FILES = {
    f"{THEORY_FIGURE}.pdf",
    f"{ROBUSTNESS_FIGURE}.pdf",
}
WORK_FILES = {
    f"{THEORY_FIGURE}.svg",
    f"{THEORY_FIGURE}.tiff",
    f"{THEORY_FIGURE}.png",
    f"{ROBUSTNESS_FIGURE}.svg",
    f"{ROBUSTNESS_FIGURE}.tiff",
    f"{ROBUSTNESS_FIGURE}.png",
    "figure_theory_source_data.csv",
    "figure_robustness_source_data.csv",
    "analysis.json",
    "figure_qa.md",
    "render_manifest.json",
}

SCPCP_BLUE = "#4394f8"
NEUTRAL_DARK = "#4D4D4D"
NEUTRAL_MID = "#7e8c9c"
VIOLET = "#9A4D8E"
ARM_COLORS = {
    "oracle": NEUTRAL_DARK,
    "correct_multinomial": SCPCP_BLUE,
    "misspecified_reduced_state": VIOLET,
}
ARM_MARKERS = {
    "oracle": "o",
    "correct_multinomial": "s",
    "misspecified_reduced_state": "D",
}
ARM_LABELS = {
    "oracle": "Oracle",
    "correct_multinomial": "Correct",
    "misspecified_reduced_state": "Misspec.",
}
SETTING_LABELS = {
    "synthetic_main": "Synthetic",
    "mimic_iv": "MIMIC-IV",
    "controlled_gamma_minus_2": r"Controlled $\gamma=-2$",
}
SETTING_MARKERS = {
    "synthetic_main": "o",
    "mimic_iv": "s",
    "controlled_gamma_minus_2": "D",
}

SOURCE_FIELDS = (
    "figure",
    "panel",
    "study",
    "metric",
    "unit",
    "method_or_variant",
    "horizon",
    "nominal_policy_tv",
    "n_calibration",
    "arm",
    "setting",
    "estimate",
    "ci95_lower",
    "ci95_upper",
    "availability_rate",
    "cluster_count",
    "replicates_per_cluster",
    "source_json_path",
    "conditioning",
)


# Preserve the repository's formal Times New Roman convention while retaining
# editable SVG text and TrueType PDF embedding.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.hashsalt"] = "scpcp-theorem-robustness-v1"


@dataclass(frozen=True)
class RenderConfig:
    horizon_root: Path = DEFAULT_HORIZON_ROOT
    rq6_root: Path = DEFAULT_RQ6_ROOT
    propensity_root: Path = DEFAULT_PROPENSITY_ROOT
    strict_root: Path = DEFAULT_STRICT_ROOT
    work_output: Path = DEFAULT_WORK_OUTPUT
    paper_output: Path = DEFAULT_PAPER_OUTPUT


@dataclass(frozen=True)
class FrozenArtifacts:
    horizon_summary: Mapping[str, Any]
    rq6_summary: Mapping[str, Any]
    propensity_summary: Mapping[str, Any]
    strict_summary: Mapping[str, Any]
    input_contracts: Mapping[str, Mapping[str, Any]]
    source_tree_sha256: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-root", type=Path, default=DEFAULT_HORIZON_ROOT)
    parser.add_argument("--rq6-root", type=Path, default=DEFAULT_RQ6_ROOT)
    parser.add_argument(
        "--propensity-root", type=Path, default=DEFAULT_PROPENSITY_ROOT
    )
    parser.add_argument("--strict-root", type=Path, default=DEFAULT_STRICT_ROOT)
    parser.add_argument("--work-output", type=Path, default=DEFAULT_WORK_OUTPUT)
    parser.add_argument("--paper-output", type=Path, default=DEFAULT_PAPER_OUTPUT)
    args = parser.parse_args()
    config = RenderConfig(
        horizon_root=args.horizon_root.resolve(),
        rq6_root=args.rq6_root.resolve(),
        propensity_root=args.propensity_root.resolve(),
        strict_root=args.strict_root.resolve(),
        work_output=args.work_output.resolve(),
        paper_output=args.paper_output.resolve(),
    )
    render_report(config)
    print(config.paper_output)


def render_report(config: RenderConfig) -> None:
    """Validate immutable inputs and atomically publish both figure bundles."""

    if config.work_output.exists() or config.paper_output.exists():
        raise FileExistsError("work-output and paper-output must both be fresh")
    if config.work_output == config.paper_output:
        raise ValueError("work-output and paper-output must be different directories")

    artifacts = load_frozen_artifacts(config)
    theory_rows = build_theory_source_rows(
        artifacts.horizon_summary,
        artifacts.rq6_summary,
    )
    robustness_rows = build_robustness_source_rows(
        artifacts.propensity_summary,
        artifacts.strict_summary,
    )

    config.work_output.parent.mkdir(parents=True, exist_ok=True)
    config.paper_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_work = Path(
        tempfile.mkdtemp(
            prefix=f".{config.work_output.name}-",
            dir=config.work_output.parent,
        )
    )
    temporary_paper = Path(
        tempfile.mkdtemp(
            prefix=f".{config.paper_output.name}-",
            dir=config.paper_output.parent,
        )
    )
    try:
        write_source_csv(
            temporary_work / "figure_theory_source_data.csv", theory_rows
        )
        write_source_csv(
            temporary_work / "figure_robustness_source_data.csv", robustness_rows
        )
        _write_analysis(
            temporary_work / "analysis.json",
            config=config,
            artifacts=artifacts,
            theory_rows=theory_rows,
            robustness_rows=robustness_rows,
            staged_work=temporary_work,
        )
        apply_publication_style()
        theory = render_theory_figure(theory_rows)
        export_figure(
            theory,
            title="Horizon-overlap difficulty and calibration-size convergence",
            work_stem=temporary_work / THEORY_FIGURE,
            paper_path=temporary_paper / f"{THEORY_FIGURE}.pdf",
        )
        robustness = render_robustness_figure(robustness_rows)
        export_figure(
            robustness,
            title="Propensity-estimation and information-split robustness",
            work_stem=temporary_work / ROBUSTNESS_FIGURE,
            paper_path=temporary_paper / f"{ROBUSTNESS_FIGURE}.pdf",
        )
        _write_qa_notes(temporary_work / "figure_qa.md", artifacts)
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
    horizon_summary, horizon_contract = validate_horizon_bundle(config.horizon_root)
    rq6_summary, rq6_contract = validate_rq6_bundle(config.rq6_root)
    propensity_summary, propensity_contract = validate_propensity_bundle(
        config.propensity_root
    )
    strict_summary, strict_contract = validate_strict_bundle(config.strict_root)
    contracts = {
        "horizon_overlap": horizon_contract,
        "rq6_ncal_convergence": rq6_contract,
        "propensity_robustness": propensity_contract,
        "strict_split_robustness": strict_contract,
    }
    source_hashes = {
        str(contract["source_tree_sha256"]) for contract in contracts.values()
    }
    if source_hashes != {EXPECTED_SOURCE_TREE_SHA256}:
        raise RuntimeError(
            "the four input bundles do not share the frozen 2026-08-26 source hash"
        )
    return FrozenArtifacts(
        horizon_summary=horizon_summary,
        rq6_summary=rq6_summary,
        propensity_summary=propensity_summary,
        strict_summary=strict_summary,
        input_contracts=contracts,
        source_tree_sha256=EXPECTED_SOURCE_TREE_SHA256,
    )


def validate_horizon_bundle(
    root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload_names = {"config.json", "metadata.json", "results.npz", "summary.json"}
    _require_exact_root_entries(root, payload_names | {"manifest.json", "COMPLETE"})
    manifest = _read_json_mapping(root / "manifest.json")
    _validate_flat_manifest(
        root,
        manifest,
        protocol=HORIZON_PROTOCOL,
        payload_names=payload_names,
    )
    config = _read_json_mapping(root / "config.json")
    metadata = _read_json_mapping(root / "metadata.json")
    summary = _read_json_mapping(root / "summary.json")
    complete = _read_json_mapping(root / "COMPLETE")
    expected_complete = {
        "status": "complete",
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "config_sha256": metadata.get("config_sha256"),
        "source_tree_sha256": metadata.get("source_tree_sha256"),
        "config_provenance_sha256": metadata.get("config_provenance", {}).get(
            "provenance_sha256"
        ),
        "formal_rng_audit_sha256": metadata.get(
            "formal_rng_collision_audit", {}
        ).get("audit_sha256"),
        "parent_snapshot_contract_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("contract_sha256"),
        "parent_snapshot_manifest_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("manifest_sha256"),
        "parent_snapshot_archive_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("archive_sha256"),
        "parent_source_tree_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("parent_source_tree_sha256"),
        "launch_sha256": _canonical_sha256(metadata.get("launch", {})),
        "environment_versions_sha256": _canonical_sha256(
            metadata.get("environment_versions", {})
        ),
    }
    if complete != expected_complete:
        raise RuntimeError("horizon-overlap COMPLETE hash contract differs")
    if (
        metadata.get("protocol") != HORIZON_PROTOCOL
        or metadata.get("status") != "complete"
        or metadata.get("schema_version") != 1
        or metadata.get("source_tree_sha256") != EXPECTED_SOURCE_TREE_SHA256
        or metadata.get("canonical_method_unchanged") is not True
        or metadata.get("diagnostic_only") is not True
    ):
        raise RuntimeError("horizon-overlap metadata contract differs")
    if (
        config.get("protocol") != HORIZON_PROTOCOL
        or tuple(config.get("horizons", ())) != HORIZONS
        or tuple(float(value) for value in config.get("nominal_policy_tvs", ()))
        != NOMINAL_POLICY_TVS
        or config.get("instances") != 200
        or config.get("calibration_trajectories") != 3_000
        or config.get("bootstrap_resamples") != 10_000
    ):
        raise RuntimeError("horizon-overlap frozen design differs")
    _validate_horizon_summary(summary)
    return summary, {
        "protocol": HORIZON_PROTOCOL,
        "source_tree_sha256": metadata["source_tree_sha256"],
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "summary_sha256": _file_sha256(root / "summary.json"),
        "input_path": _project_path(root),
    }


def _validate_horizon_summary(summary: Mapping[str, Any]) -> None:
    if (
        summary.get("schema_version") != 1
        or summary.get("study") != "finite_mdp_horizon_overlap"
        or summary.get("status") != "complete"
        or summary.get("mechanism_variant") != "RQ5_only_overlap_controlled_M3"
        or summary.get("canonical_method_unchanged") is not True
        or summary.get("diagnostic_only") is not True
        or summary.get("primary_coverage_estimand")
        != "min_stage_mean_instance_conditional_on_availability"
    ):
        raise RuntimeError("horizon-overlap summary contract differs")
    phase = _mapping(summary.get("phase_diagram"), "RQ5 phase diagram")
    if set(phase) != set(RQ5_METHODS):
        raise RuntimeError("RQ5 phase-diagram method set differs")
    metrics = (
        "coverage_shortfall",
        "median_minimum_selected_ess_fraction",
        "median_surface_sup_error",
        "availability_rate",
        "mean_selected_policy_realized_tv",
    )
    for method in RQ5_METHODS:
        method_phase = _mapping(phase.get(method), f"RQ5 phase diagram/{method}")
        for metric in metrics:
            _finite_matrix(
                method_phase.get(metric),
                shape=(len(HORIZONS), len(NOMINAL_POLICY_TVS)),
                label=f"RQ5 {method}/{metric}",
            )
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != 100:
        raise RuntimeError("RQ5 summary must contain exactly 100 method cells")
    by_key = {
        (row.get("horizon"), float(row.get("nominal_policy_tv")), row.get("method")): row
        for row in records
        if isinstance(row, Mapping)
    }
    expected = {
        (horizon, tv, method)
        for horizon in HORIZONS
        for tv in NOMINAL_POLICY_TVS
        for method in RQ5_METHODS
    }
    if set(by_key) != expected:
        raise RuntimeError("RQ5 summary cell set differs")
    for horizon_index, horizon in enumerate(HORIZONS):
        for tv_index, tv in enumerate(NOMINAL_POLICY_TVS):
            for method in RQ5_METHODS:
                row = by_key[(horizon, tv, method)]
                for metric in metrics:
                    phase_value = phase[method][metric][horizon_index][tv_index]
                    if not math.isclose(
                        float(row[metric]), float(phase_value), rel_tol=0.0, abs_tol=1e-14
                    ):
                        raise RuntimeError(f"RQ5 phase diagram differs from records: {metric}")
    bootstrap = _mapping(summary.get("bootstrap"), "RQ5 bootstrap")
    if bootstrap.get("resamples") != 10_000 or bootstrap.get("cluster_unit") != "paired_M3_instance":
        raise RuntimeError("RQ5 bootstrap contract differs")


def validate_propensity_bundle(
    root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload_names = {
        "appendix_end_to_end.csv",
        "arrays.npz",
        "config.json",
        "metadata.json",
        "nuisance_diagnostics.csv",
        "primary_transport_only.csv",
        "summary.json",
    }
    _require_exact_root_entries(root, payload_names | {"manifest.json", "COMPLETE"})
    manifest = _read_json_mapping(root / "manifest.json")
    _validate_flat_manifest(
        root,
        manifest,
        protocol=PROPENSITY_PROTOCOL,
        payload_names=payload_names,
    )
    config = _read_json_mapping(root / "config.json")
    metadata = _read_json_mapping(root / "metadata.json")
    summary = _read_json_mapping(root / "summary.json")
    complete = _read_json_mapping(root / "COMPLETE")
    expected_complete = {
        "status": "complete",
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "config_sha256": metadata.get("config_sha256"),
        "source_tree_sha256": metadata.get("source_tree_sha256"),
        "parent_snapshot_manifest_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("manifest_sha256"),
        "parent_snapshot_archive_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("archive_sha256"),
        "parent_source_tree_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("parent_source_tree_sha256"),
        "formal_rng_audit_sha256": _canonical_sha256(
            metadata.get("formal_rng_collision_audit", {})
        ),
        "launch_sha256": _canonical_sha256(metadata.get("launch", {})),
        "environment_versions_sha256": _canonical_sha256(
            metadata.get("environment_versions", {})
        ),
        "multinomial_fit_semantics_sha256": _canonical_sha256(
            metadata.get("multinomial_propensity_fit_semantics", {})
        ),
    }
    if complete != expected_complete:
        raise RuntimeError("propensity COMPLETE hash contract differs")
    if (
        metadata.get("protocol") != PROPENSITY_PROTOCOL
        or metadata.get("status") != "complete"
        or metadata.get("schema_version") != 1
        or metadata.get("source_tree_sha256") != EXPECTED_SOURCE_TREE_SHA256
        or metadata.get("canonical_method_unchanged") is not True
        or metadata.get("diagnostic_only") is not True
        or metadata.get("formal_scientific_run") is not True
    ):
        raise RuntimeError("propensity metadata contract differs")
    if (
        config.get("protocol") != PROPENSITY_PROTOCOL
        or config.get("instances") != 100
        or config.get("horizon") != 8
        or config.get("calibration_trajectories") != 5_000
        or config.get("nuisance_trajectories") != 5_000
        or config.get("bootstrap_resamples") != 10_000
    ):
        raise RuntimeError("propensity frozen design differs")
    _validate_propensity_summary(summary)
    return summary, {
        "protocol": PROPENSITY_PROTOCOL,
        "source_tree_sha256": metadata["source_tree_sha256"],
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "summary_sha256": _file_sha256(root / "summary.json"),
        "input_path": _project_path(root),
    }


def _validate_propensity_summary(summary: Mapping[str, Any]) -> None:
    if (
        summary.get("schema_version") != 1
        or summary.get("study") != PROPENSITY_PROTOCOL
        or summary.get("status") != "complete"
        or summary.get("diagnostic_only") is not True
        or summary.get("canonical_method_unchanged") is not True
        or summary.get("formal_scientific_run") is not True
        or float(summary.get("target_coverage", float("nan"))) != 0.90
        or tuple(summary.get("propensity_arms", ())) != PROPENSITY_ARMS
    ):
        raise RuntimeError("propensity summary contract differs")
    nuisance = _mapping(summary.get("nuisance_diagnostics"), "propensity nuisance")
    primary = _mapping(summary.get("primary_transport_only"), "propensity primary")
    appendix = _mapping(summary.get("appendix_end_to_end"), "propensity appendix")
    if primary.get("target_law_fingerprint_shared_across_arms") is not True:
        raise RuntimeError("propensity primary target law differs across arms")
    primary_results = _mapping(primary.get("results"), "propensity primary results")
    primary_arms = _mapping(primary_results.get("arms"), "propensity primary arms")
    target_drift = _mapping(
        appendix.get("target_policy_drift_from_oracle"),
        "propensity target-policy drift",
    )
    if set(primary_arms) != set(PROPENSITY_ARMS) or set(target_drift) != set(
        PROPENSITY_ARMS
    ):
        raise RuntimeError("propensity arm set differs")
    for arm in PROPENSITY_ARMS:
        _metric_with_interval(nuisance["mae"][arm], f"propensity MAE/{arm}")
        arm_result = _mapping(primary_arms[arm], f"propensity primary/{arm}")
        _point_and_named_interval(
            arm_result,
            "marginal_worst_step_coverage",
            "marginal_worst_step_coverage_ci95",
            f"propensity WSC/{arm}",
        )
        _point_and_named_interval(
            arm_result,
            "minimum_stage_mean_ess_fraction",
            "minimum_stage_mean_ess_fraction_ci95",
            f"propensity ESS/{arm}",
        )
        if float(arm_result.get("selection_rate", float("nan"))) != 1.0:
            raise RuntimeError("propensity selection availability is not one")
        _metric_with_interval(target_drift[arm], f"propensity target drift/{arm}")


def validate_rq6_bundle(
    root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    required_files = {
        "COMPLETE",
        "artifact_manifest.json",
        "config.yaml",
        "manifest.json",
        "runtime_preflight.json",
        "study_status.json",
        "summary.json",
    }
    problem_dirs = {f"problem_{seed}" for seed in range(97_000, 97_100)}
    _require_exact_root_entries(root, required_files | problem_dirs)
    manifest = _read_json_mapping(root / "manifest.json")
    complete = _read_json_mapping(root / "COMPLETE")
    artifact_manifest = _read_json_mapping(root / "artifact_manifest.json")
    summary = _read_json_mapping(root / "summary.json")
    config = _read_yaml_mapping(root / "config.yaml")
    expected_complete = {
        "status": "complete",
        "protocol": RQ6_PROTOCOL,
        "config_sha256": manifest.get("config_sha256"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "parent_snapshot_manifest_sha256": manifest.get("parent_snapshot", {}).get(
            "manifest_sha256"
        ),
        "parent_snapshot_archive_sha256": manifest.get("parent_snapshot", {}).get(
            "archive_sha256"
        ),
        "parent_source_tree_sha256": manifest.get("parent_snapshot", {}).get(
            "source_tree_sha256"
        ),
        "runtime_environment_sha256": manifest.get("runtime_environment_sha256"),
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "runtime_preflight_sha256": _file_sha256(root / "runtime_preflight.json"),
        "summary_sha256": _file_sha256(root / "summary.json"),
        "artifact_manifest_sha256": _file_sha256(root / "artifact_manifest.json"),
    }
    if complete != expected_complete:
        raise RuntimeError("RQ6 COMPLETE hash contract differs")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol") != RQ6_PROTOCOL
        or manifest.get("study") != "rq6_n_calibration_convergence"
        or manifest.get("canonical_selector_unchanged") is not True
        or manifest.get("source_tree_sha256") != EXPECTED_SOURCE_TREE_SHA256
    ):
        raise RuntimeError("RQ6 manifest contract differs")
    if (
        config.get("protocol") != RQ6_PROTOCOL
        or config.get("horizon") != 4
        or config.get("grid_size") != 7
        or tuple(config.get("n_calibration", ())) != N_CALIBRATION
        or config.get("problem_count") != 100
        or config.get("logged_replicates") != 20
        or config.get("bootstrap_resamples") != 10_000
    ):
        raise RuntimeError("RQ6 frozen design differs")
    _validate_rq6_problem_manifest(root, artifact_manifest)
    status = _read_json_mapping(root / "study_status.json")
    expected_seeds = list(range(97_000, 97_100))
    if (
        status.get("status") != "complete"
        or status.get("error") is not None
        or status.get("expected_problem_seeds") != expected_seeds
        or status.get("completed_problem_seeds") != expected_seeds
        or status.get("missing_problem_seeds") != []
    ):
        raise RuntimeError("RQ6 study status differs")
    _validate_rq6_summary(summary, manifest)
    return summary, {
        "protocol": RQ6_PROTOCOL,
        "source_tree_sha256": manifest["source_tree_sha256"],
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "artifact_manifest_sha256": _file_sha256(root / "artifact_manifest.json"),
        "summary_sha256": _file_sha256(root / "summary.json"),
        "input_path": _project_path(root),
    }


def _validate_rq6_problem_manifest(
    root: Path, artifact_manifest: Mapping[str, Any]
) -> None:
    if (
        artifact_manifest.get("protocol") != RQ6_PROTOCOL
        or artifact_manifest.get("problem_count") != 100
    ):
        raise RuntimeError("RQ6 per-problem artifact manifest differs")
    artifacts = _mapping(
        artifact_manifest.get("problem_artifacts"), "RQ6 problem artifacts"
    )
    expected_keys = {str(seed) for seed in range(97_000, 97_100)}
    if set(artifacts) != expected_keys:
        raise RuntimeError("RQ6 problem artifact set differs")
    expected_files = {"result.json", "metadata.json", "COMPLETE"}
    for seed in range(97_000, 97_100):
        path = root / f"problem_{seed}"
        _require_exact_root_entries(path, expected_files)
        contracts = _mapping(artifacts[str(seed)], f"RQ6 artifact/{seed}")
        if set(contracts) != expected_files:
            raise RuntimeError(f"RQ6 artifact file set differs: {seed}")
        for name in expected_files:
            _validate_file_contract(path / name, contracts[name])


def _validate_rq6_summary(
    summary: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    if (
        summary.get("protocol") != RQ6_PROTOCOL
        or summary.get("status") != "complete"
        or summary.get("config_sha256") != manifest.get("config_sha256")
        or summary.get("source_tree_sha256") != manifest.get("source_tree_sha256")
        or summary.get("runtime_environment_sha256")
        != manifest.get("runtime_environment_sha256")
        or summary.get("formal_problem_count") != 100
    ):
        raise RuntimeError("RQ6 summary contract differs")
    design = _mapping(summary.get("design"), "RQ6 design")
    if (
        tuple(design.get("n_calibration", ())) != N_CALIBRATION
        or design.get("problem_cluster_count") != 100
        or design.get("logged_resamples_per_problem") != 20
        or design.get("bootstrap_resamples") != 10_000
        or design.get("nested_common_random_numbers") is not True
    ):
        raise RuntimeError("RQ6 summary design differs")
    by_n = _mapping(summary.get("by_n_calibration"), "RQ6 n-calibration cells")
    if set(by_n) != {str(value) for value in N_CALIBRATION}:
        raise RuntimeError("RQ6 n-calibration cell set differs")
    for n_calibration in N_CALIBRATION:
        cell = _mapping(by_n[str(n_calibration)], f"RQ6 n={n_calibration}")
        track_a = _mapping(cell.get("track_a_fixed_population_grid"), "RQ6 Track A")
        track_b = _mapping(cell.get("track_b_canonical_empirical_grid"), "RQ6 Track B")
        _point_and_named_interval(
            track_a,
            "mean_surface_sup_error",
            "cluster_bootstrap_95_ci",
            f"RQ6 surface error/{n_calibration}",
        )
        _point_and_named_interval(
            track_b,
            "population_wsc_conditional_on_selection",
            "population_wsc_cluster_bootstrap_95_ci",
            f"RQ6 WSC/{n_calibration}",
        )
        _point_and_named_interval(
            track_b,
            "population_mean_normalized_width_conditional_on_selection",
            "population_width_cluster_bootstrap_95_ci",
            f"RQ6 width/{n_calibration}",
        )
        if float(track_b.get("selection_availability_rate", float("nan"))) != 1.0:
            raise RuntimeError("RQ6 selection availability is not one")
    slope = _mapping(summary.get("track_a_descriptive_log_log_slope"), "RQ6 slope")
    if slope.get("status") != "descriptive_not_a_claimed_rate":
        raise RuntimeError("RQ6 slope claim boundary differs")
    _finite_number(slope.get("value"), "RQ6 descriptive slope")


def validate_strict_bundle(
    root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    required_files = {"COMPLETE", "artifact_manifest.json", "manifest.json", "summary.json"}
    _require_exact_root_entries(root, required_files | set(STRICT_SETTINGS))
    manifest = _read_json_mapping(root / "manifest.json")
    artifact_manifest = _read_json_mapping(root / "artifact_manifest.json")
    summary = _read_json_mapping(root / "summary.json")
    complete = _read_json_mapping(root / "COMPLETE")
    manifest_hash = _file_sha256(root / "manifest.json")
    expected_complete = {
        "status": "complete",
        "protocol": STRICT_PROTOCOL,
        "task_count": 140,
        "manifest_sha256": manifest_hash,
        "summary_sha256": _file_sha256(root / "summary.json"),
        "artifact_manifest_sha256": _file_sha256(root / "artifact_manifest.json"),
        "parent_snapshot_manifest_sha256": manifest.get(
            "parent_formal_snapshot", {}
        ).get("manifest_sha256"),
        "parent_snapshot_archive_sha256": manifest.get(
            "parent_formal_snapshot", {}
        ).get("archive_sha256"),
        "parent_source_tree_sha256": manifest.get("parent_formal_snapshot", {}).get(
            "source_tree_sha256"
        ),
        "active_source_tree_sha256": manifest.get("active_source_tree_sha256"),
    }
    if complete != expected_complete:
        raise RuntimeError("strict-split COMPLETE hash contract differs")
    if (
        manifest.get("protocol") != STRICT_PROTOCOL
        or manifest.get("role") != "theory_aligned_robustness_only"
        or manifest.get("canonical_method_changed") is not False
        or manifest.get("active_source_tree_sha256") != EXPECTED_SOURCE_TREE_SHA256
    ):
        raise RuntimeError("strict-split manifest contract differs")
    _validate_strict_artifact_manifest(root, artifact_manifest, manifest_hash)
    _validate_strict_summary(summary, manifest, artifact_manifest)
    return summary, {
        "protocol": STRICT_PROTOCOL,
        "source_tree_sha256": manifest["active_source_tree_sha256"],
        "manifest_sha256": manifest_hash,
        "artifact_manifest_sha256": _file_sha256(root / "artifact_manifest.json"),
        "summary_sha256": _file_sha256(root / "summary.json"),
        "input_path": _project_path(root),
    }


def _validate_strict_artifact_manifest(
    root: Path,
    artifact_manifest: Mapping[str, Any],
    manifest_hash: str,
) -> None:
    if (
        artifact_manifest.get("protocol") != STRICT_PROTOCOL
        or artifact_manifest.get("manifest_sha256") != manifest_hash
        or artifact_manifest.get("task_count") != 140
    ):
        raise RuntimeError("strict-split artifact manifest differs")
    artifacts = artifact_manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 140:
        raise RuntimeError("strict-split artifact list differs")
    observed_paths: set[str] = set()
    for entry in artifacts:
        contract = _mapping(entry, "strict-split artifact entry")
        relative = contract.get("path")
        if not isinstance(relative, str) or relative in observed_paths:
            raise RuntimeError("strict-split artifact path is malformed or duplicated")
        observed_paths.add(relative)
        path = root / relative
        if path.resolve().parent.parent != root.resolve():
            raise RuntimeError("strict-split artifact path escapes the expected setting")
        _require_exact_root_entries(path, {"row.json", "metadata.json", "COMPLETE"})
        for name, key in (
            ("row.json", "row_sha256"),
            ("metadata.json", "metadata_sha256"),
            ("COMPLETE", "complete_sha256"),
        ):
            if contract.get(key) != _file_sha256(path / name):
                raise RuntimeError(f"strict-split artifact hash differs: {relative}/{name}")


def _validate_strict_summary(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
) -> None:
    if (
        summary.get("protocol") != STRICT_PROTOCOL
        or summary.get("role") != "theory_aligned_robustness_only"
        or summary.get("canonical_method_changed") is not False
        or summary.get("active_source_tree_sha256")
        != manifest.get("active_source_tree_sha256")
        or summary.get("manifest_sha256") != _canonical_file_sha(manifest)
        or summary.get("artifact_manifest_sha256")
        != _canonical_file_sha(artifact_manifest)
    ):
        raise RuntimeError("strict-split summary contract differs")
    settings = _mapping(summary.get("settings"), "strict-split settings")
    if set(settings) != set(STRICT_SETTINGS):
        raise RuntimeError("strict-split setting set differs")
    expected_seeds = {
        "synthetic_main": tuple(range(1_000, 1_100)),
        "mimic_iv": tuple(range(20)),
        "controlled_gamma_minus_2": tuple(range(99_000, 99_200, 10)),
    }
    expected_artifact_paths: set[str] = set()
    for setting in STRICT_SETTINGS:
        cell = _mapping(settings[setting], f"strict-split/{setting}")
        if tuple(cell.get("seeds", ())) != expected_seeds[setting]:
            raise RuntimeError(f"strict-split seed bank differs: {setting}")
        variants = _mapping(cell.get("variants"), f"strict variants/{setting}")
        if set(variants) != {"canonical", "strict"}:
            raise RuntimeError(f"strict-split variant set differs: {setting}")
        for variant in ("canonical", "strict"):
            variant_cell = _mapping(variants[variant], f"strict {setting}/{variant}")
            if float(variant_cell.get("selection_rate", float("nan"))) != 1.0:
                raise RuntimeError(f"strict-split availability is not one: {setting}")
        paired = _mapping(
            cell.get("paired_strict_vs_canonical"), f"strict paired/{setting}"
        )
        _point_and_named_interval(
            paired,
            "strict_minus_canonical_wsc",
            "strict_minus_canonical_wsc_ci95",
            f"strict WSC delta/{setting}",
        )
        _point_and_named_interval(
            paired,
            "strict_to_canonical_geometric_width_ratio",
            "strict_to_canonical_geometric_width_ratio_ci95",
            f"strict width ratio/{setting}",
        )
        if float(paired.get("strict_minus_canonical_availability", float("nan"))) != 0.0:
            raise RuntimeError(f"strict-split availability difference is nonzero: {setting}")
        expected_artifact_paths.update(
            f"{setting}/seed_{seed:05d}" for seed in expected_seeds[setting]
        )
    observed_artifact_paths = {
        str(entry["path"]) for entry in artifact_manifest["artifacts"]
    }
    if observed_artifact_paths != expected_artifact_paths:
        raise RuntimeError("strict-split atomic artifact paths differ from summary seeds")


def _canonical_file_sha(payload: Mapping[str, Any]) -> str:
    """Hash an already parsed JSON file exactly as its sorted on-disk encoding."""

    # Strict-split JSON is emitted as sorted, indented JSON plus a trailing newline.
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_flat_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    protocol: str,
    payload_names: set[str],
) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol") != protocol
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError(f"{protocol} manifest header differs")
    files = _mapping(manifest.get("files"), f"{protocol} manifest files")
    if set(files) != payload_names:
        raise RuntimeError(f"{protocol} manifest payload set differs")
    for name in payload_names:
        _validate_file_contract(root / name, files[name])


def _validate_file_contract(path: Path, contract: object) -> None:
    expected = _mapping(contract, f"file contract for {path.name}")
    observed = {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
    if expected != observed:
        raise RuntimeError(f"artifact payload hash differs: {path}")


def build_theory_source_rows(
    horizon_summary: Mapping[str, Any],
    rq6_summary: Mapping[str, Any],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    phase = horizon_summary["phase_diagram"]["SC-PCP"]
    rq5_metrics = (
        ("a", "coverage_shortfall", "Coverage shortfall", "pp", 100.0),
        (
            "b",
            "median_minimum_selected_ess_fraction",
            "Median minimum selected prefix ESS/n",
            "%",
            100.0,
        ),
        (
            "c",
            "median_surface_sup_error",
            "Median committed-surface sup error",
            "pp",
            100.0,
        ),
    )
    for panel, field, label, unit, scale in rq5_metrics:
        matrix = np.asarray(phase[field], dtype=np.float64)
        for horizon_index, horizon in enumerate(HORIZONS):
            for tv_index, nominal_tv in enumerate(NOMINAL_POLICY_TVS):
                rows.append(
                    _source_row(
                        figure=THEORY_FIGURE,
                        panel=panel,
                        study="RQ5 horizon-overlap",
                        metric=label,
                        unit=unit,
                        method_or_variant="SC-PCP",
                        horizon=horizon,
                        nominal_policy_tv=nominal_tv,
                        estimate=scale * float(matrix[horizon_index, tv_index]),
                        availability_rate=float(
                            phase["availability_rate"][horizon_index][tv_index]
                        ),
                        cluster_count=200,
                        replicates_per_cluster=1,
                        source_json_path=(
                            f'phase_diagram["SC-PCP"]["{field}"]'
                            f"[{horizon_index}][{tv_index}]"
                        ),
                        conditioning="successful selection; availability reported separately",
                    )
                )
    by_n = rq6_summary["by_n_calibration"]
    rq6_metrics = (
        (
            "d",
            "track_a_fixed_population_grid",
            "mean_surface_sup_error",
            "cluster_bootstrap_95_ci",
            "Mean full-prefix surface sup error",
            "pp",
            100.0,
            "all problem/logged cells",
        ),
        (
            "e",
            "track_b_canonical_empirical_grid",
            "population_wsc_conditional_on_selection",
            "population_wsc_cluster_bootstrap_95_ci",
            "Exact population WSC",
            "%",
            100.0,
            "successful selection",
        ),
        (
            "f",
            "track_b_canonical_empirical_grid",
            "population_mean_normalized_width_conditional_on_selection",
            "population_width_cluster_bootstrap_95_ci",
            "Exact population normalized width",
            "normalized width",
            1.0,
            "successful selection",
        ),
    )
    for panel, track, field, interval_field, label, unit, scale, conditioning in rq6_metrics:
        for n_calibration in N_CALIBRATION:
            track_cell = by_n[str(n_calibration)][track]
            interval = track_cell[interval_field]
            availability = by_n[str(n_calibration)][
                "track_b_canonical_empirical_grid"
            ]["selection_availability_rate"]
            rows.append(
                _source_row(
                    figure=THEORY_FIGURE,
                    panel=panel,
                    study="RQ6 n-calibration convergence",
                    metric=label,
                    unit=unit,
                    method_or_variant="SC-PCP",
                    n_calibration=n_calibration,
                    estimate=scale * float(track_cell[field]),
                    ci95_lower=scale * float(interval[0]),
                    ci95_upper=scale * float(interval[1]),
                    availability_rate=float(availability),
                    cluster_count=100,
                    replicates_per_cluster=20,
                    source_json_path=(
                        f'by_n_calibration["{n_calibration}"]["{track}"]'
                        f'["{field}"]'
                    ),
                    conditioning=conditioning,
                )
            )
    return rows


def build_robustness_source_rows(
    propensity_summary: Mapping[str, Any],
    strict_summary: Mapping[str, Any],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    propensity_metrics = (
        (
            "a",
            "Propensity MAE",
            "probability",
            lambda arm: propensity_summary["nuisance_diagnostics"]["mae"][arm],
            1.0,
            "independent nuisance sample",
            'nuisance_diagnostics["mae"]',
        ),
        (
            "b",
            "Primary fixed-target-law WSC",
            "%",
            lambda arm: _named_metric(
                propensity_summary["primary_transport_only"]["results"]["arms"][arm],
                "marginal_worst_step_coverage",
                "marginal_worst_step_coverage_ci95",
            ),
            100.0,
            "jointly available paired problems; target law fixed",
            'primary_transport_only["results"]["arms"]',
        ),
        (
            "c",
            "Primary minimum stage-mean ESS/n",
            "%",
            lambda arm: _named_metric(
                propensity_summary["primary_transport_only"]["results"]["arms"][arm],
                "minimum_stage_mean_ess_fraction",
                "minimum_stage_mean_ess_fraction_ci95",
            ),
            100.0,
            "jointly available paired problems; target law fixed",
            'primary_transport_only["results"]["arms"]',
        ),
        (
            "d",
            "Appendix target-policy drift from oracle",
            "% TV",
            lambda arm: propensity_summary["appendix_end_to_end"][
                "target_policy_drift_from_oracle"
            ][arm],
            100.0,
            "appendix end-to-end target surface; not pooled with primary",
            'appendix_end_to_end["target_policy_drift_from_oracle"]',
        ),
    )
    for panel, label, unit, getter, scale, conditioning, source_prefix in propensity_metrics:
        for arm in PROPENSITY_ARMS:
            metric = getter(arm)
            rows.append(
                _source_row(
                    figure=ROBUSTNESS_FIGURE,
                    panel=panel,
                    study="Propensity robustness",
                    metric=label,
                    unit=unit,
                    method_or_variant="SC-PCP",
                    arm=arm,
                    estimate=scale * float(metric["mean"]),
                    ci95_lower=scale * float(metric["ci95"][0]),
                    ci95_upper=scale * float(metric["ci95"][1]),
                    availability_rate=1.0,
                    cluster_count=100,
                    replicates_per_cluster=1,
                    source_json_path=f'{source_prefix}["{arm}"]',
                    conditioning=conditioning,
                )
            )
    settings = strict_summary["settings"]
    strict_metrics = (
        (
            "e",
            "Strict minus canonical WSC",
            "pp",
            "strict_minus_canonical_wsc",
            "strict_minus_canonical_wsc_ci95",
            lambda value: 100.0 * value,
        ),
        (
            "f",
            "Strict/canonical geometric width change",
            "%",
            "strict_to_canonical_geometric_width_ratio",
            "strict_to_canonical_geometric_width_ratio_ci95",
            lambda value: 100.0 * (value - 1.0),
        ),
    )
    for panel, label, unit, field, interval_field, transform in strict_metrics:
        for setting in STRICT_SETTINGS:
            paired = settings[setting]["paired_strict_vs_canonical"]
            interval = paired[interval_field]
            rows.append(
                _source_row(
                    figure=ROBUSTNESS_FIGURE,
                    panel=panel,
                    study="Strict-split robustness",
                    metric=label,
                    unit=unit,
                    method_or_variant="strict minus canonical",
                    setting=setting,
                    estimate=transform(float(paired[field])),
                    ci95_lower=transform(float(interval[0])),
                    ci95_upper=transform(float(interval[1])),
                    availability_rate=float(
                        settings[setting]["variants"]["strict"]["selection_rate"]
                    ),
                    cluster_count=len(settings[setting]["seeds"]),
                    replicates_per_cluster=1,
                    source_json_path=(
                        f'settings["{setting}"]["paired_strict_vs_canonical"]'
                        f'["{field}"]'
                    ),
                    conditioning="jointly available paired seeds; matched evaluation CRN",
                )
            )
    return rows


def _named_metric(
    payload: Mapping[str, Any], point_field: str, interval_field: str
) -> dict[str, object]:
    return {"mean": payload[point_field], "ci95": payload[interval_field]}


def _source_row(**values: object) -> dict[str, object]:
    return {field: values.get(field, "") for field in SOURCE_FIELDS}


def write_source_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SOURCE_FIELDS})


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "scpcp-theorem-robustness-v1",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def render_theory_figure(rows: Sequence[Mapping[str, object]]) -> plt.Figure:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(7.20, 5.25),
        constrained_layout=True,
    )
    for axis, panel, cmap, vmin, vmax, title in (
        (axes[0, 0], "a", "RdBu_r", -2.0, 2.0, "Coverage shortfall (pp)"),
        (axes[0, 1], "b", "Blues", 0.0, 100.0, "Median min. selected ESS/n (%)"),
        (axes[0, 2], "c", "Oranges", 0.0, 4.0, "Median surface sup error (pp)"),
    ):
        matrix = _heatmap_matrix(rows, panel)
        norm: mpl.colors.Normalize
        if panel == "a":
            norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        _annotate_heatmap(axis, matrix, image.cmap, norm)
        axis.set_xticks(range(len(NOMINAL_POLICY_TVS)))
        axis.set_xticklabels(("0", ".025", ".05", ".10", ".15"))
        axis.set_yticks(range(len(HORIZONS)))
        axis.set_yticklabels(HORIZONS)
        axis.set_xlabel("Nominal one-step policy TV")
        if panel == "a":
            axis.set_ylabel("Horizon $T$")
        axis.set_title(title)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
        colorbar.ax.tick_params(labelsize=5.7, width=0.6, length=2)
        colorbar.outline.set_linewidth(0.6)

    _plot_convergence_panel(
        axes[1, 0],
        _panel_rows(rows, "d"),
        title="Full-prefix surface recovery",
        ylabel="Max surface error (pp)",
        reference=None,
    )
    _plot_convergence_panel(
        axes[1, 1],
        _panel_rows(rows, "e"),
        title="Canonical exact population WSC",
        ylabel="Worst-stage coverage (%)",
        reference=90.0,
    )
    _plot_convergence_panel(
        axes[1, 2],
        _panel_rows(rows, "f"),
        title="Canonical exact population width",
        ylabel="Normalized width",
        reference=None,
    )
    for label, axis in zip("abcdef", axes.ravel()):
        add_panel_label(axis, label)
    return figure


def _heatmap_matrix(
    rows: Sequence[Mapping[str, object]], panel: str
) -> np.ndarray:
    panel_rows = _panel_rows(rows, panel)
    lookup = {
        (int(row["horizon"]), float(row["nominal_policy_tv"])): float(row["estimate"])
        for row in panel_rows
    }
    return np.asarray(
        [[lookup[(horizon, tv)] for tv in NOMINAL_POLICY_TVS] for horizon in HORIZONS],
        dtype=np.float64,
    )


def _annotate_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    colormap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> None:
    for (row, column), value in np.ndenumerate(matrix):
        red, green, blue, _ = colormap(norm(value))
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        axis.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=5.0,
            color="white" if luminance < 0.48 else "black",
        )


def _plot_convergence_panel(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    *,
    title: str,
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
        marker="o",
        markersize=3.6,
        linewidth=1.25,
        capsize=2.0,
        markeredgecolor=SCPCP_BLUE,
    )
    if reference is not None:
        axis.axhline(reference, color=NEUTRAL_DARK, linestyle=(0, (3, 2)), linewidth=0.8)
    axis.set_xscale("log")
    axis.set_xticks(x)
    axis.set_xticklabels(("250", "500", "1k", "2k", "5k", "10k"))
    axis.set_xlabel(r"Total calibration size $n_{\rm cal}$")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(width=0.7, length=2.5)
    axis.margins(x=0.04, y=0.13)


def render_robustness_figure(
    rows: Sequence[Mapping[str, object]],
) -> plt.Figure:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(7.20, 4.80),
        constrained_layout=True,
    )
    _plot_arm_panel(
        axes[0, 0], _panel_rows(rows, "a"), title="Propensity fit error", ylabel="MAE"
    )
    _plot_arm_panel(
        axes[0, 1],
        _panel_rows(rows, "b"),
        title="Primary: fixed target law",
        ylabel="Worst-stage coverage (%)",
        reference=90.0,
    )
    _plot_arm_panel(
        axes[0, 2],
        _panel_rows(rows, "c"),
        title="Primary: overlap",
        ylabel="Minimum stage-mean ESS/n (%)",
    )
    _plot_arm_panel(
        axes[1, 0],
        _panel_rows(rows, "d"),
        title="Appendix: target-law drift",
        ylabel="Mean policy-surface TV (%)",
    )
    _plot_forest_panel(
        axes[1, 1],
        _panel_rows(rows, "e"),
        title="Strict-split coverage",
        xlabel="Strict − canonical WSC (pp)",
    )
    _plot_forest_panel(
        axes[1, 2],
        _panel_rows(rows, "f"),
        title="Strict-split width",
        xlabel="Geometric width change (%)",
    )
    for label, axis in zip("abcdef", axes.ravel()):
        add_panel_label(axis, label)
    return figure


def _plot_arm_panel(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    *,
    title: str,
    ylabel: str,
    reference: float | None = None,
) -> None:
    by_arm = {str(row["arm"]): row for row in rows}
    x = np.arange(len(PROPENSITY_ARMS))
    for index, arm in enumerate(PROPENSITY_ARMS):
        row = by_arm[arm]
        point = float(row["estimate"])
        lower = float(row["ci95_lower"])
        upper = float(row["ci95_upper"])
        axis.errorbar(
            index,
            point,
            yerr=[[point - lower], [upper - point]],
            color=ARM_COLORS[arm],
            marker=ARM_MARKERS[arm],
            markersize=4.0,
            linewidth=1.0,
            capsize=2.2,
        )
    if reference is not None:
        axis.axhline(reference, color=NEUTRAL_DARK, linestyle=(0, (3, 2)), linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels([ARM_LABELS[arm] for arm in PROPENSITY_ARMS], rotation=17)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(width=0.7, length=2.5)
    axis.margins(x=0.20, y=0.16)


def _plot_forest_panel(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    *,
    title: str,
    xlabel: str,
) -> None:
    by_setting = {str(row["setting"]): row for row in rows}
    y = np.arange(len(STRICT_SETTINGS))[::-1]
    for y_value, setting in zip(y, STRICT_SETTINGS):
        row = by_setting[setting]
        point = float(row["estimate"])
        lower = float(row["ci95_lower"])
        upper = float(row["ci95_upper"])
        axis.plot([lower, upper], [y_value, y_value], color=SCPCP_BLUE, linewidth=1.1)
        axis.plot(
            point,
            y_value,
            color=SCPCP_BLUE,
            marker=SETTING_MARKERS[setting],
            markersize=4.2,
        )
    axis.axvline(0.0, color=NEUTRAL_DARK, linestyle=(0, (3, 2)), linewidth=0.8)
    axis.set_yticks(y)
    axis.set_yticklabels([SETTING_LABELS[setting] for setting in STRICT_SETTINGS])
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.tick_params(width=0.7, length=2.5)
    axis.margins(x=0.12, y=0.30)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.16,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _panel_rows(
    rows: Sequence[Mapping[str, object]], panel: str
) -> list[Mapping[str, object]]:
    selected = [row for row in rows if row.get("panel") == panel]
    if not selected:
        raise RuntimeError(f"source data contains no panel {panel}")
    return selected


def export_figure(
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
        "Creator": "SC-PCP theorem robustness renderer",
        "CreationDate": None,
        "ModDate": None,
    }
    svg_metadata = {
        "Title": title,
        "Creator": "SC-PCP theorem robustness renderer",
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
        metadata={"Software": "SC-PCP theorem robustness renderer"},
    )
    plt.close(figure)


def _write_analysis(
    path: Path,
    *,
    config: RenderConfig,
    artifacts: FrozenArtifacts,
    theory_rows: Sequence[Mapping[str, object]],
    robustness_rows: Sequence[Mapping[str, object]],
    staged_work: Path,
) -> None:
    theory_source = staged_work / "figure_theory_source_data.csv"
    robustness_source = staged_work / "figure_robustness_source_data.csv"
    payload = {
        "schema_version": 1,
        "protocol": RENDER_PROTOCOL,
        "status": "complete",
        "source_tree_sha256": artifacts.source_tree_sha256,
        "input_contracts": artifacts.input_contracts,
        "backend": "Python/matplotlib only",
        "figures": {
            THEORY_FIGURE: {
                "archetype": "quantitative_grid",
                "core_conclusion": (
                    "Coverage remains non-short under the frozen grid while overlap and "
                    "horizon degrade prefix ESS and finite-sample surface recovery; larger "
                    "calibration samples reduce the full-prefix surface error."
                ),
                "panels": {
                    "a": "predeclared SC-PCP coverage-shortfall phase diagram",
                    "b": "selected prefix ESS phase diagram",
                    "c": "committed-surface error phase diagram",
                    "d": "fixed-grid surface convergence",
                    "e": "canonical exact population WSC",
                    "f": "canonical exact population width",
                },
                "source_data": "figure_theory_source_data.csv",
                "source_data_sha256": _file_sha256(theory_source),
                "source_row_count": len(theory_rows),
            },
            ROBUSTNESS_FIGURE: {
                "archetype": "quantitative_grid",
                "core_conclusion": (
                    "The frozen benchmark separates transport-denominator sensitivity from "
                    "target-law drift and shows paired stability under an independent "
                    "strict calibration split without changing canonical SC-PCP."
                ),
                "panels": {
                    "a": "propensity nuisance MAE",
                    "b": "fixed-target-law primary WSC",
                    "c": "fixed-target-law primary ESS",
                    "d": "appendix target-law drift",
                    "e": "paired strict-minus-canonical WSC",
                    "f": "paired strict/canonical geometric width change",
                },
                "source_data": "figure_robustness_source_data.csv",
                "source_data_sha256": _file_sha256(robustness_source),
                "source_row_count": len(robustness_rows),
            },
        },
        "export_contract": {
            "paper_directory": _project_path(config.paper_output),
            "paper_files": sorted(PAPER_FILES),
            "paper_format": "PDF only; TrueType fonts",
            "work_formats": ["editable SVG", "600-dpi TIFF", "240-dpi PNG"],
            "font": "Times New Roman with Times/DejaVu Serif fallback",
            "visible_font_size_pt": 7,
            "svg_text": "text elements retained",
            "determinism": (
                "fixed row/category order, fixed palette and markers, fixed svg.hashsalt, "
                "no bootstrap recomputation, PDF creation/modification timestamps omitted"
            ),
        },
        "claim_boundary": (
            "Controlled theorem and robustness diagnostics only; no finite-sample "
            "distribution-free, PAC, data-conditional, clinical, equivalence, or universal "
            "SOTA claim. Canonical SC-PCP is unchanged."
        ),
    }
    _write_json(path, payload)


def _write_qa_notes(path: Path, artifacts: FrozenArtifacts) -> None:
    path.write_text(
        "\n".join(
            (
                "# Theorem and robustness figure QA",
                "",
                "- Backend: Python/matplotlib only.",
                "- Archetype: two quantitative-grid figures; every panel has a unique estimand.",
                "- Inputs: four immutable COMPLETE bundles; no experiment or bootstrap was rerun.",
                f"- Common frozen source hash: `{artifacts.source_tree_sha256}`.",
                "- Theory figure: 7.20 x 5.25 inches; robustness figure: 7.20 x 4.80 inches.",
                "- Typography: Times New Roman, 7 pt body, 8 pt bold lowercase panel labels.",
                "- Accessibility: restrained palettes plus marker/position redundancy; no rainbow map.",
                "- SVG: text retained as editable `<text>` elements; fixed `svg.hashsalt`.",
                "- PDF: TrueType embedding (`pdf.fonttype=42`); creation/modification timestamps omitted.",
                "- Work exports: editable SVG, 600-dpi LZW TIFF, 240-dpi PNG preview, source CSVs, analysis JSON.",
                "- Paper output: exactly two PDF files and no auxiliary files.",
                "- RQ5 n: 200 paired finite-MDP instances; heatmap cells are descriptive medians without CI.",
                "- RQ6 n: 100 problem clusters x 20 paired logged resamples; intervals use the frozen 10,000-resample problem-cluster bootstrap.",
                "- Propensity n: 100 paired finite-MDP problems; primary and appendix layers remain separate.",
                "- Strict split n: Synthetic 100 seeds, MIMIC-IV 20 seeds, controlled gamma=-2 20 seeds; paired complete-seed-vector bootstrap.",
                "- RQ5 negative shortfall means coverage above 0.90; it is not evidence of Standard CP failure.",
                "- RQ6 six-point slope is descriptive, not a claimed asymptotic exponent.",
                "- Propensity stability is benchmark-specific and is not double robustness.",
                "- Strict-split near-zero differences are not an equivalence test or post-hoc method upgrade.",
                "- No raster-image manipulation; every panel is generated directly from frozen numerical summaries.",
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
            f"work figure bundle differs: expected {sorted(WORK_FILES)}, found {sorted(observed_work)}"
        )
    if observed_paper != PAPER_FILES or any(
        path.suffix.lower() != ".pdf" for path in paper_root.iterdir()
    ):
        raise RuntimeError("paper figure output must contain exactly two PDFs")
    for stem in (THEORY_FIGURE, ROBUSTNESS_FIGURE):
        svg = (work_root / f"{stem}.svg").read_text(encoding="utf-8")
        if "<text" not in svg or "font-family" not in svg:
            raise RuntimeError(f"SVG text is not editable: {stem}")
        if not (paper_root / f"{stem}.pdf").read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"paper PDF header is malformed: {stem}")
        if not (work_root / f"{stem}.png").read_bytes().startswith(b"\x89PNG"):
            raise RuntimeError(f"PNG preview header is malformed: {stem}")
        tiff_header = (work_root / f"{stem}.tiff").read_bytes()[:4]
        if tiff_header not in {b"II*\x00", b"MM\x00*"}:
            raise RuntimeError(f"TIFF header is malformed: {stem}")
    manifest = _read_json_mapping(work_root / "render_manifest.json")
    if (
        manifest.get("protocol") != RENDER_PROTOCOL
        or manifest.get("status") != "complete"
        or set(_mapping(manifest.get("paper_files"), "render manifest paper"))
        != PAPER_FILES
    ):
        raise RuntimeError("render manifest contract differs")
    for group, root in (("work_files", work_root), ("paper_files", paper_root)):
        contracts = _mapping(manifest[group], f"render manifest/{group}")
        for name, contract in contracts.items():
            _validate_file_contract(root / name, contract)


def _metric_with_interval(payload: object, label: str) -> None:
    metric = _mapping(payload, label)
    _finite_number(metric.get("mean"), f"{label}/mean")
    _finite_interval(metric.get("ci95"), f"{label}/ci95")


def _point_and_named_interval(
    payload: Mapping[str, Any],
    point_field: str,
    interval_field: str,
    label: str,
) -> None:
    _finite_number(payload.get(point_field), f"{label}/{point_field}")
    _finite_interval(payload.get(interval_field), f"{label}/{interval_field}")


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"{label} is not finite")
    return resolved


def _finite_interval(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeError(f"{label} is not a two-sided interval")
    lower = _finite_number(value[0], f"{label}/lower")
    upper = _finite_number(value[1], f"{label}/upper")
    if lower > upper:
        raise RuntimeError(f"{label} is reversed")
    return lower, upper


def _finite_matrix(value: object, *, shape: tuple[int, int], label: str) -> np.ndarray:
    resolved = np.asarray(value, dtype=np.float64)
    if resolved.shape != shape or not np.isfinite(resolved).all():
        raise RuntimeError(f"{label} must have finite shape {shape}")
    return resolved


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _require_exact_root_entries(root: Path, expected: set[str]) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {root}")
    observed = {path.name for path in root.iterdir()}
    if observed != expected:
        raise RuntimeError(
            f"artifact entry set differs for {root}: expected {sorted(expected)}, found {sorted(observed)}"
        )


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, str(path))


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(payload, str(path))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
