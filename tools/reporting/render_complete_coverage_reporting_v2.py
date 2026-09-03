"""Render the signed-gamma default coverage report from frozen COMPLETE roots.

This command is deterministic post-processing.  It verifies pinned Native and
clinical-v4 publication bundles plus the terminal CXR v5/v6 gate, adapts their
protocol-specific summaries into one long-form reporting schema, and writes an
atomic work/paper bundle.  It does not import or execute any science runner and
draws no scientific RNG.

Example
-------
conda run -n ucp python tools/reporting/render_complete_coverage_reporting_v2.py
"""

from __future__ import annotations

import argparse
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

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import render_complete_coverage_reporting as legacy  # noqa: E402


RENDER_PROTOCOL = (
    "complete_coverage_reporting_v4_signed_gamma_minimal_quantitative_cxr_terminal"
)
DEFAULT_NATIVE_INPUT = (
    ROOT
    / "results/work/native_synthetic_signed_gamma_six_method_science_v1_exact_replay_r1"
)
DEFAULT_CLINICAL_INPUT = (
    ROOT
    / "results/work/controlled_clinical_fidelity_v4_signed_gamma_science_publish_retry_r1"
)
DEFAULT_CXR_V5_CONFIRMATION_INPUT = (
    ROOT / "results/work/controlled_clinical_fidelity_v5_mimic_cxr_confirmation"
)
DEFAULT_CXR_V6_DEVELOPMENT_INPUT = (
    ROOT / "results/work/controlled_clinical_fidelity_v6_mimic_cxr_development"
)
DEFAULT_PRODUCTION_INPUT = ROOT / "results/work/complete_baseline_results_20260824"
DEFAULT_WORK_OUTPUT = (
    ROOT
    / "results/work/complete_coverage_reporting_v4_minimal_quantitative_20260830"
)
DEFAULT_PAPER_OUTPUT = (
    ROOT
    / "results/paper_complete_coverage_reporting_v4_minimal_quantitative_20260830"
)

NATIVE_MANIFEST_SHA256 = "00ac178aaba0d3faa992b2d85b554f995a34a484db638ed2f3749b42371c649d"
NATIVE_COMPLETE_SHA256 = "b201bac5a86596b4e7c4aaec377c3fec908806832e92205edf057ebb869949fc"
CLINICAL_MANIFEST_SHA256 = "c9a5344d135024446c62675628437054927f84f217c2dc41e0b1fc37b4fe37bc"
CLINICAL_COMPLETE_SHA256 = "b89028c85dbf3234f4f4377094dd9e8aa60bfd0138a867e77a0abe8dd7f31c63"
CXR_V5_MANIFEST_SHA256 = "a1a89634c268bd3b4b480db49b481bc3cf135025155e06bd85e4369fa9c6baec"
CXR_V5_COMPLETE_SHA256 = "638b0ba296deabed76d62921d46f6174a8444e15ffd89efe6d5bc39e1a64a3f4"
CXR_V5_FINAL_SHA256 = "5f104c0dff121174b52e5ce0c082583744d544cda47a296f5ad0329474472f18"
CXR_V5_METADATA_SHA256 = "927212270de9215b5b36b112cd59f7ac46701fe96cd2fac8edfb6280ba5db726"
CXR_V6_MANIFEST_SHA256 = "b14d1f401b6bbe4593dd5c24daad70dfb651e98529e45c2c7f2ba409e000d309"
CXR_V6_COMPLETE_SHA256 = "0cc4e853faeee35f18ffd0e47a0dd947aa4d2a48d672c1ab53a93b9e156c0194"
CXR_V6_FINAL_SHA256 = "39c014b9429466849b709a90739ae1b88d72d6eec43f3425ef2281d48fa058a1"
CXR_V6_PARENT_BINDING_SHA256 = (
    "5d2b15f4b840ff86bbda37fe6f592c9b4809ebe1021fafd72bac541c279a9008"
)
NATIVE_PROTOCOL = "native_synthetic_signed_gamma_six_method_science_v1"
CLINICAL_PROTOCOL = "controlled_clinical_fidelity_v4_signed_gamma_science_v1"
CLINICAL_PUBLISH_PROTOCOL = (
    "controlled_clinical_fidelity_v4_signed_gamma_science_publish_retry_r1"
)
CXR_V5_PROTOCOL = "controlled_clinical_fidelity_v5_mimic_cxr"
CXR_V6_PROTOCOL = "controlled_clinical_fidelity_v6_mimic_cxr"
CXR_TERMINAL_FAMILY = "clinical_cxr_terminal_no_go"

TARGET = 0.90
PRIMARY_GAMMA = -4.0
GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
METHODS = legacy.METHODS
DATASETS = ("synthetic", "mimic_iv", "eicu", "inspire", "mimic_cxr")
CURVE_DATASETS = DATASETS[:-1]
PRODUCTION_DATASETS = legacy.PRODUCTION_DATASETS
HORIZONS = legacy.HORIZONS
DATASET_LABELS = legacy.DATASET_LABELS
PRIMARY_METRIC = "min_t mean_seed(target_coverage_seed_t)"
POINT_ELIGIBILITY_RULE = "selection_rate>=0.95 and WSC>=0.90"
CLINICAL_WIDTH_DEFINITION = legacy.NORMALIZED_WIDTH_DEFINITION
NATIVE_WIDTH_DEFINITION = (
    "mean over outcome coordinates of full prediction-box width divided by "
    "the frozen Native one-unit outcome scales [1,1]"
)
INFORMATION_REGIME = legacy.INFORMATION_REGIME
TARGET_ADAPTATION_BUDGET = legacy.TARGET_ADAPTATION_BUDGET
CALIBRATION_BUDGET = 3_000
GRID_BUDGET = 1_000
EVALUATION_BUDGET = 20_000

MAIN_STAGE_STEM = "figure_gamma_minus4_stagewise_profiles"
MAIN_TABLE_STEM = "table_gamma_minus4_complete_metrics"
SIGNED_FIGURE_STEM = "figure_signed_gamma_complete_metrics"
PRODUCTION_TABLE_STEM = "table_production_complete_metrics"
FIGURE_STEMS = (MAIN_STAGE_STEM, SIGNED_FIGURE_STEM)
TABLE_STEMS = (MAIN_TABLE_STEM, PRODUCTION_TABLE_STEM)
OUTPUT_STEMS = (*FIGURE_STEMS, *TABLE_STEMS)

STATUS_COLUMNS = (
    "schema_version",
    "reporting_family",
    "setting_id",
    "display_label",
    "dataset",
    "source_protocol",
    "setting_type",
    "feedback_parameter",
    "feedback_value",
    "horizon",
    "method_count",
    "panel_status",
    "interpretation_status",
    "confirmatory",
    "ranking_permitted",
    "scientific_rows_available",
    "n_prespecified",
    "n_gate_eligible",
    "gate_reason",
    "budget_status",
    "frozen_setting_sha256",
    "source_path",
    "source_sha256",
)

STAGE_COLUMNS = (
    "schema_version",
    "reporting_family",
    "setting_id",
    "display_label",
    "dataset",
    "source_protocol",
    "setting_type",
    "feedback_parameter",
    "feedback_value",
    "analysis_role",
    "panel_status",
    "confirmatory",
    "ranking_permitted",
    "method",
    "information_regime",
    "calibration_trajectories_per_seed",
    "grid_trajectories_per_seed",
    "target_adaptation_trajectories_per_seed",
    "evaluation_trajectories_per_seed",
    "budget_status",
    "horizon",
    "stage_zero_based",
    "n_prespecified",
    "n_gate_eligible",
    "n_selected",
    "coverage_conditioning",
    "coverage_target",
    "coverage_mean",
    "coverage_ci95_lower",
    "coverage_ci95_upper",
    "coverage_interval_definition",
    "coverage_interval_scope",
    "normalized_width_mean",
    "normalized_width_ci95_lower",
    "normalized_width_ci95_upper",
    "normalized_width_interval_definition",
    "normalized_width_interval_scope",
    "normalized_width_definition",
    "source_path",
    "source_sha256",
)

SCALAR_COLUMNS = (
    "schema_version",
    "reporting_family",
    "setting_id",
    "display_label",
    "dataset",
    "source_protocol",
    "setting_type",
    "feedback_parameter",
    "feedback_value",
    "analysis_role",
    "panel_status",
    "confirmatory",
    "ranking_permitted",
    "method",
    "information_regime",
    "calibration_trajectories_per_seed",
    "grid_trajectories_per_seed",
    "target_adaptation_trajectories_per_seed",
    "evaluation_trajectories_per_seed",
    "budget_status",
    "n_prespecified",
    "n_gate_eligible",
    "n_selected",
    "metric_available",
    "coverage_conditioning",
    "selection_rate",
    "selection_rate_ci95_lower",
    "selection_rate_ci95_upper",
    "selection_interval_definition",
    "coverage_target",
    "wsc",
    "wsc_ci95_lower",
    "wsc_ci95_upper",
    "wsc_deviation_from_target_pp",
    "wsc_interval_definition",
    "primary_metric",
    "worst_stage_zero_based",
    "mean_coverage",
    "mean_coverage_ci95_lower",
    "mean_coverage_ci95_upper",
    "mean_coverage_deviation_from_target_pp",
    "mean_coverage_interval_definition",
    "mean_normalized_width",
    "mean_normalized_width_ci95_lower",
    "mean_normalized_width_ci95_upper",
    "mean_width_interval_definition",
    "normalized_width_definition",
    "point_eligibility_rule",
    "point_attainment_at_target",
    "wsc_interval_attainment_at_target",
    "point_eligible",
    "efficiency_rank_defined",
    "narrowest_point_eligible",
    "source_path",
    "source_sha256",
)

PAIRED_COLUMNS = (
    "schema_version",
    "reporting_family",
    "setting_id",
    "dataset",
    "feedback_value",
    "confirmatory",
    "ranking_permitted",
    "baseline",
    "paired_selected_seeds",
    "scpcp_minus_baseline_wsc",
    "scpcp_minus_baseline_wsc_ci95_lower",
    "scpcp_minus_baseline_wsc_ci95_upper",
    "scpcp_to_baseline_geometric_width_ratio",
    "scpcp_to_baseline_geometric_width_ratio_ci95_lower",
    "scpcp_to_baseline_geometric_width_ratio_ci95_upper",
    "interval_definition",
    "source_path",
    "source_sha256",
)

SOURCE_FILES = {
    "setting_status.csv",
    "coverage_stage_profiles.csv",
    "coverage_scalar_summary.csv",
    "paired_scpcp_contrasts.csv",
}
WORK_FILES = {
    *(f"{stem}.{suffix}" for stem in OUTPUT_STEMS for suffix in ("svg", "pdf", "tiff", "png")),
    *SOURCE_FILES,
    "figure_contract.json",
    "figure_qa.md",
    "render_manifest.json",
    "COMPLETE",
}
PAPER_FILES = {f"{stem}.pdf" for stem in OUTPUT_STEMS}


@dataclass(frozen=True)
class ReportingSources:
    status: pd.DataFrame
    stage: pd.DataFrame
    scalar: pd.DataFrame
    paired: pd.DataFrame
    input_contracts: Mapping[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-input", type=Path, default=DEFAULT_NATIVE_INPUT)
    parser.add_argument("--clinical-input", type=Path, default=DEFAULT_CLINICAL_INPUT)
    parser.add_argument(
        "--cxr-v5-confirmation-input",
        type=Path,
        default=DEFAULT_CXR_V5_CONFIRMATION_INPUT,
    )
    parser.add_argument(
        "--cxr-v6-development-input",
        type=Path,
        default=DEFAULT_CXR_V6_DEVELOPMENT_INPUT,
    )
    parser.add_argument("--production-input", type=Path, default=DEFAULT_PRODUCTION_INPUT)
    parser.add_argument("--work-output", type=Path, default=DEFAULT_WORK_OUTPUT)
    parser.add_argument("--paper-output", type=Path, default=DEFAULT_PAPER_OUTPUT)
    args = parser.parse_args()
    render_report(
        native_input=args.native_input.resolve(),
        clinical_input=args.clinical_input.resolve(),
        cxr_v5_confirmation_input=args.cxr_v5_confirmation_input.resolve(),
        cxr_v6_development_input=args.cxr_v6_development_input.resolve(),
        production_input=args.production_input.resolve(),
        work_output=args.work_output.resolve(),
        paper_output=args.paper_output.resolve(),
    )
    print(args.paper_output.resolve())


def build_reporting_sources(
    *,
    native_input: Path,
    clinical_input: Path,
    production_input: Path,
    cxr_v5_confirmation_input: Path = DEFAULT_CXR_V5_CONFIRMATION_INPUT,
    cxr_v6_development_input: Path = DEFAULT_CXR_V6_DEVELOPMENT_INPUT,
) -> ReportingSources:
    native = load_native_sources(native_input)
    clinical = load_clinical_v4_sources(clinical_input)
    cxr = load_cxr_terminal_sources(
        v5_confirmation_root=cxr_v5_confirmation_input,
        v6_development_root=cxr_v6_development_input,
    )
    production = load_production_sources(production_input)
    status = pd.concat(
        [native.status, clinical.status, cxr.status, production.status],
        ignore_index=True,
    ).loc[:, STATUS_COLUMNS]
    stage = pd.concat(
        [native.stage, clinical.stage, production.stage], ignore_index=True
    ).loc[:, STAGE_COLUMNS]
    scalar = pd.DataFrame(
        [
            row
            for frame in (native.scalar, clinical.scalar, cxr.scalar, production.scalar)
            for row in frame.to_dict(orient="records")
        ],
        columns=SCALAR_COLUMNS,
    )
    scalar = assign_efficiency_ranking(scalar)
    paired = pd.concat([native.paired, clinical.paired], ignore_index=True).loc[
        :, PAIRED_COLUMNS
    ]
    sources = ReportingSources(
        status=status,
        stage=stage,
        scalar=scalar,
        paired=paired,
        input_contracts={
            "native": native.input_contracts,
            "clinical_v4": clinical.input_contracts,
            "clinical_cxr_terminal": cxr.input_contracts,
            "production_robustness": production.input_contracts,
        },
    )
    validate_reporting_sources(sources)
    return sources


def load_native_sources(root: Path) -> ReportingSources:
    manifest = _validate_pinned_bundle(
        root,
        expected_manifest_sha256=NATIVE_MANIFEST_SHA256,
        expected_complete_sha256=NATIVE_COMPLETE_SHA256,
        size_field="size_bytes",
    )
    summary_path = root / "summary.json"
    summary = _read_json(summary_path)
    final = _read_json(root / "FINAL_STATUS.json")
    complete = _read_json(root / "COMPLETE")
    if (
        summary.get("protocol") != NATIVE_PROTOCOL
        or final.get("protocol") != NATIVE_PROTOCOL
        or final.get("decision") != "SCIENCE_COMPLETE_EXACT_REPLAY"
        or complete.get("decision") != "SCIENCE_COMPLETE_EXACT_REPLAY"
        or complete.get("downstream_authorized") is not True
        or summary.get("primary_gamma") != PRIMARY_GAMMA
        or tuple(summary.get("gammas", ())) != GAMMAS
        or tuple(summary.get("methods", ())) != METHODS
        or summary.get("primary_metric") != PRIMARY_METRIC
        or complete.get("manifest_sha256") != NATIVE_MANIFEST_SHA256
        or complete.get("summary_sha256") != _file_sha256(summary_path)
    ):
        raise RuntimeError("Native signed-gamma COMPLETE semantics differ")
    source_path = _project_path(summary_path)
    source_sha = _file_sha256(summary_path)
    status_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    information = _mapping(summary["information_budgets"], "Native information budgets")
    aggregates = _gamma_aggregates(summary)
    for gamma in GAMMAS:
        aggregate = aggregates[gamma]
        confirmatory = gamma == PRIMARY_GAMMA
        setting_id = f"synthetic_gamma_{_gamma_id(gamma)}"
        status_rows.append(
            _status_row(
                reporting_family="native_signed_gamma",
                setting_id=setting_id,
                display_label="Synthetic",
                dataset="synthetic",
                source_protocol=NATIVE_PROTOCOL,
                setting_type="native_synthetic_signed_gamma",
                feedback_parameter="gamma",
                feedback_value=gamma,
                horizon=HORIZONS["synthetic"],
                panel_status="CURVES",
                interpretation_status=str(aggregate["gamma_role"]),
                confirmatory=confirmatory,
                ranking_permitted=confirmatory,
                scientific_rows_available=True,
                n_prespecified=int(aggregate["n_prespecified_seeds"]),
                n_gate_eligible=int(aggregate["n_prespecified_seeds"]),
                gate_reason="",
                budget_status="consumed_complete_science",
                frozen_setting_sha256=str(complete["science_config_payload_sha256"]),
                source_path=source_path,
                source_sha256=source_sha,
            )
        )
        methods = _mapping(aggregate["methods"], "Native aggregate methods")
        for method in METHODS:
            cell = _mapping(methods[method], f"Native {gamma}/{method}")
            budget = _mapping(information[method], f"Native budget {method}")
            _validate_budget(
                method,
                calibration=int(budget["logged_calibration_trajectories_per_seed"]),
                grid=int(budget["grid_trajectories_per_seed"]),
                adaptation=int(budget["target_adaptation_trajectories_per_seed"]),
                evaluation=int(budget["fresh_reference_trajectories_per_seed"]),
            )
            common = _method_context(
                reporting_family="native_signed_gamma",
                setting_id=setting_id,
                display_label="Synthetic",
                dataset="synthetic",
                source_protocol=NATIVE_PROTOCOL,
                setting_type="native_synthetic_signed_gamma",
                feedback_value=gamma,
                analysis_role=str(aggregate["gamma_role"]),
                panel_status="CURVES",
                confirmatory=confirmatory,
                ranking_permitted=confirmatory,
                method=method,
                n_prespecified=int(cell["selection_total"]),
                n_gate_eligible=int(cell["selection_total"]),
                n_selected=int(cell["selection_successes"]),
                coverage_conditioning=str(cell["coverage_conditioning"]),
                normalized_width_definition=NATIVE_WIDTH_DEFINITION,
                source_path=source_path,
                source_sha256=source_sha,
            )
            stage_rows.extend(
                _stage_rows_from_cell(
                    common,
                    cell,
                    coverage_ci_field="target_coverage_ci95_by_stage",
                    width_ci_field="target_normalized_width_ci95_by_stage",
                )
            )
            scalar_rows.append(
                _scalar_from_cell(common, cell, stored_point_eligibility=False)
            )
        paired_rows.extend(
            _paired_rows(
                aggregate,
                reporting_family="native_signed_gamma",
                setting_id=setting_id,
                dataset="synthetic",
                feedback_value=gamma,
                confirmatory=confirmatory,
                source_path=source_path,
                source_sha256=source_sha,
            )
        )
    return _sources_from_rows(
        status_rows,
        stage_rows,
        scalar_rows,
        paired_rows,
        input_contracts={
            "protocol": NATIVE_PROTOCOL,
            "input_root": _project_path(root),
            "manifest_sha256": NATIVE_MANIFEST_SHA256,
            "complete_sha256": NATIVE_COMPLETE_SHA256,
            "artifact_count": manifest["artifact_count"],
        },
    )


def load_clinical_v4_sources(root: Path) -> ReportingSources:
    manifest = _validate_pinned_bundle(
        root,
        expected_manifest_sha256=CLINICAL_MANIFEST_SHA256,
        expected_complete_sha256=CLINICAL_COMPLETE_SHA256,
        size_field="bytes",
    )
    final = _read_json(root / "FINAL_STATUS.json")
    metadata = _read_json(root / "metadata.json")
    if (
        manifest.get("protocol") != CLINICAL_PUBLISH_PROTOCOL
        or final.get("protocol") != CLINICAL_PROTOCOL
        or final.get("status") != "COMPLETE_DATASET_INDEPENDENT"
        or tuple(final.get("confirmed_datasets", ())) != ("mimic_iv", "eicu", "inspire")
        or final.get("unopened_datasets")
        != {"mimic_cxr": "CONFIRMATION_NOT_OPENED_DEVELOPMENT_NO_GO"}
        or tuple(final.get("methods", ())) != METHODS
        or tuple(final.get("gammas", ())) != GAMMAS
        or final.get("primary_default_gamma") != PRIMARY_GAMMA
        or final.get("primary_metric") != PRIMARY_METRIC
    ):
        raise RuntimeError("clinical-v4 published COMPLETE semantics differ")
    theta = _mapping(metadata.get("dataset_theta"), "clinical-v4 dataset theta")
    status_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for dataset in ("mimic_iv", "eicu", "inspire"):
        summary_path = root / "science" / dataset / "summary.json"
        summary = _read_json(summary_path)
        dataset_status = _mapping(final["datasets"][dataset], f"{dataset} final status")
        if (
            summary.get("protocol") != CLINICAL_PROTOCOL
            or summary.get("dataset") != dataset
            or summary.get("interpretation_status") != "EMPIRICAL_OVERLAP_SCREEN_PASSED"
            or dataset_status.get("interpretation_status")
            != "EMPIRICAL_OVERLAP_SCREEN_PASSED"
            or summary.get("primary_default_gamma") != PRIMARY_GAMMA
            or summary.get("primary_metric") != PRIMARY_METRIC
            or tuple(summary.get("methods", ())) != METHODS
        ):
            raise RuntimeError(f"{dataset} clinical-v4 summary semantics differ")
        source_path = _project_path(summary_path)
        source_sha = _file_sha256(summary_path)
        aggregates = _gamma_aggregates(summary)
        for gamma in GAMMAS:
            aggregate = aggregates[gamma]
            confirmatory = gamma == PRIMARY_GAMMA
            role = str(aggregate["analysis_role"])
            setting_id = f"{dataset}_gamma_{_gamma_id(gamma)}"
            n_prespecified = int(aggregate["n_prespecified_seeds"])
            n_gate = int(aggregate["n_k0_eligible_seeds"])
            status_rows.append(
                _status_row(
                    reporting_family="clinical_v4_signed_gamma",
                    setting_id=setting_id,
                    display_label=DATASET_LABELS[dataset],
                    dataset=dataset,
                    source_protocol=CLINICAL_PROTOCOL,
                    setting_type="dataset_native_clinical_controlled_v4",
                    feedback_parameter="gamma",
                    feedback_value=gamma,
                    horizon=HORIZONS[dataset],
                    panel_status="CURVES",
                    interpretation_status=role,
                    confirmatory=confirmatory,
                    ranking_permitted=confirmatory,
                    scientific_rows_available=True,
                    n_prespecified=n_prespecified,
                    n_gate_eligible=n_gate,
                    gate_reason="",
                    budget_status="consumed_complete_science",
                    frozen_setting_sha256=_canonical_sha256(theta[dataset]),
                    source_path=source_path,
                    source_sha256=source_sha,
                )
            )
            methods = _mapping(aggregate["methods"], f"{dataset}/{gamma} methods")
            for method in METHODS:
                cell = _mapping(methods[method], f"{dataset}/{gamma}/{method}")
                budget = _mapping(
                    cell["information_budget_per_seed"],
                    f"{dataset}/{gamma}/{method} budget",
                )
                _validate_budget(
                    method,
                    calibration=int(budget["source_calibration_trajectories"]),
                    grid=int(budget["source_grid_trajectories"]),
                    adaptation=int(budget["target_adaptation_trajectories"]),
                    evaluation=int(budget["target_reference_trajectories"]),
                )
                common = _method_context(
                    reporting_family="clinical_v4_signed_gamma",
                    setting_id=setting_id,
                    display_label=DATASET_LABELS[dataset],
                    dataset=dataset,
                    source_protocol=CLINICAL_PROTOCOL,
                    setting_type="dataset_native_clinical_controlled_v4",
                    feedback_value=gamma,
                    analysis_role=role,
                    panel_status="CURVES",
                    confirmatory=confirmatory,
                    ranking_permitted=confirmatory,
                    method=method,
                    n_prespecified=int(cell["n_prespecified"]),
                    n_gate_eligible=int(cell["support_and_k0_eligible_seed_count"]),
                    n_selected=int(cell["n_selected"]),
                    coverage_conditioning=str(summary["coverage_conditioning"]),
                    normalized_width_definition=CLINICAL_WIDTH_DEFINITION,
                    source_path=source_path,
                    source_sha256=source_sha,
                )
                stage_rows.extend(
                    _stage_rows_from_cell(
                        common,
                        cell,
                        coverage_ci_field="target_coverage_by_stage_ci95",
                        width_ci_field="target_normalized_width_by_stage_ci95",
                    )
                )
                scalar_rows.append(
                    _scalar_from_cell(common, cell, stored_point_eligibility=True)
                )
            paired_rows.extend(
                _paired_rows(
                    aggregate,
                    reporting_family="clinical_v4_signed_gamma",
                    setting_id=setting_id,
                    dataset=dataset,
                    feedback_value=gamma,
                    confirmatory=confirmatory,
                    source_path=source_path,
                    source_sha256=source_sha,
                )
            )

    return _sources_from_rows(
        status_rows,
        stage_rows,
        scalar_rows,
        paired_rows,
        input_contracts={
            "protocol": CLINICAL_PROTOCOL,
            "publication_protocol": CLINICAL_PUBLISH_PROTOCOL,
            "input_root": _project_path(root),
            "manifest_sha256": CLINICAL_MANIFEST_SHA256,
            "complete_sha256": CLINICAL_COMPLETE_SHA256,
            "artifact_count": manifest["artifact_count"],
            "dataset_theta_sha256": {
                dataset: _canonical_sha256(theta[dataset])
                for dataset in ("mimic_iv", "eicu", "inspire")
            },
            "historical_cxr_status": str(final["unopened_datasets"]["mimic_cxr"]),
        },
    )


def load_cxr_terminal_sources(
    *, v5_confirmation_root: Path, v6_development_root: Path
) -> ReportingSources:
    v5_manifest = _validate_pinned_bundle(
        v5_confirmation_root,
        expected_manifest_sha256=CXR_V5_MANIFEST_SHA256,
        expected_complete_sha256=CXR_V5_COMPLETE_SHA256,
        size_field="bytes",
    )
    v6_manifest = _validate_pinned_bundle(
        v6_development_root,
        expected_manifest_sha256=CXR_V6_MANIFEST_SHA256,
        expected_complete_sha256=CXR_V6_COMPLETE_SHA256,
        size_field="bytes",
    )
    v5_final_path = v5_confirmation_root / "FINAL_STATUS.json"
    v5_final = _read_json(v5_final_path)
    v5_gate = _read_json(v5_confirmation_root / "gate.json")
    if (
        v5_manifest.get("protocol") != CXR_V5_PROTOCOL
        or _file_sha256(v5_final_path) != CXR_V5_FINAL_SHA256
        or v5_final.get("protocol") != CXR_V5_PROTOCOL
        or v5_final.get("dataset") != "mimic_cxr"
        or v5_final.get("phase") != "confirmation"
        or v5_final.get("status") != "CONFIRMATION_COMPLETE_NO_GO"
        or v5_final.get("confirmed") is not False
        or v5_final.get("coverage_generated") is not False
        or v5_final.get("information_firewall_respected") is not True
        or v5_final.get("candidate_seed_deletions") != 0
        or v5_gate.get("status") != "CONFIRMATION_GATE_NO_GO"
        or v5_gate.get("confirmation_opened") is not True
        or v5_gate.get("k0_pass_count") != 18
        or v5_gate.get("structural_pass_count") != 20
        or v5_gate.get("support_pass_count") != 20
        or v5_gate.get("prespecified_seed_count") != 20
        or v5_gate.get("coverage_generated") is not False
        or v5_gate.get("independent_patient_confirmation_claimed") is not False
    ):
        raise RuntimeError("CXR v5 completed-confirmation semantics differ")

    v6_final_path = v6_development_root / "FINAL_STATUS.json"
    v6_final = _read_json(v6_final_path)
    v6_gate = _read_json(v6_development_root / "development_gate.json")
    v6_frozen = _read_json(v6_development_root / "frozen_settings.json")
    v6_metadata = _read_json(v6_development_root / "metadata.json")
    if (
        v6_manifest.get("protocol") != CXR_V6_PROTOCOL
        or _file_sha256(v6_final_path) != CXR_V6_FINAL_SHA256
        or v6_final.get("protocol") != CXR_V6_PROTOCOL
        or v6_final.get("dataset") != "mimic_cxr"
        or v6_final.get("phase") != "development"
        or v6_final.get("status") != "DEVELOPMENT_NO_GO"
        or v6_final.get("terminal_no_v7") is not True
        or v6_final.get("development_admissible") is not False
        or v6_final.get("further_bridge_repair_permitted") is not False
        or v6_final.get("coverage_generated") is not False
        or v6_final.get("information_firewall_respected") is not True
        or v6_final.get("candidate_seed_deletions") != 0
        or v6_gate.get("status") != "DEVELOPMENT_GATE_NO_GO"
        or v6_gate.get("required_numeric_pass_count_per_lineage") != 20
        or v6_gate.get("required_structural_pass_count_per_lineage") != 20
        or v6_gate.get("scientific_candidate_count") != 1
        or v6_gate.get("selector_present") is not False
        or v6_gate.get("grid_present") is not False
        or v6_gate.get("coverage_generated") is not False
        or v6_gate.get("terminal_no_v7") is not True
        or v6_frozen.get("theta") is not None
        or v6_frozen.get("coverage_generated") is not False
        or v6_frozen.get("terminal_no_v7") is not True
        or v6_frozen.get("parent_v5_binding_sha256")
        != CXR_V6_PARENT_BINDING_SHA256
    ):
        raise RuntimeError("CXR v6 terminal-development semantics differ")

    lineages = _mapping(v6_gate.get("lineage_summaries"), "CXR v6 lineages")
    for name, expected_pass_count in (
        ("v5_development", 19),
        ("v5_failed_confirmation", 18),
    ):
        lineage = _mapping(lineages.get(name), f"CXR v6 lineage {name}")
        if (
            lineage.get("seed_count") != 20
            or lineage.get("pass_count") != expected_pass_count
            or lineage.get("structural_pass_count") != 20
            or lineage.get("nonfinite_numeric_ratio_present") is not False
        ):
            raise RuntimeError(f"CXR v6 lineage semantics differ: {name}")

    parent = _mapping(v6_metadata.get("parent_v5_binding"), "CXR v6 parent binding")
    failed_confirmation = _mapping(
        parent.get("failed_confirmation"), "CXR v6 failed-confirmation binding"
    )
    confirmation_rng = _mapping(
        v6_metadata.get("confirmation_rng_audit"), "CXR v6 confirmation RNG audit"
    )
    fresh_mapping = _mapping(
        confirmation_rng.get("new_rng_stream_mapping"), "CXR v6 fresh RNG mapping"
    )
    fresh_base_seeds = sorted(
        int(key.split("/base_", 1)[1].split("/", 1)[0])
        for key in fresh_mapping
        if key.endswith("/task") and "/base_" in key
    )
    if (
        v6_metadata.get("parent_v5_binding_sha256")
        != CXR_V6_PARENT_BINDING_SHA256
        or failed_confirmation.get("manifest_sha256") != CXR_V5_MANIFEST_SHA256
        or failed_confirmation.get("complete_sha256") != CXR_V5_COMPLETE_SHA256
        or failed_confirmation.get("metadata_sha256") != CXR_V5_METADATA_SHA256
        or failed_confirmation.get("full_semantic_bundle_validation") is not True
        or parent.get("failed_confirmation_reclassified_as_development_only")
        is not True
        or parent.get("scientific_freshness_claimed") is not False
        or confirmation_rng.get("status") != "passed_before_launch"
        or confirmation_rng.get("confirmation_base_seed_count") != 20
        or confirmation_rng.get("new_rng_stream_count") != 341
        or confirmation_rng.get("collision_count") != 0
        or confirmation_rng.get("formal_rng_consumed") is not False
        or confirmation_rng.get("internal_rng_streams_unique") is not True
        or fresh_base_seeds != list(range(120000, 120200, 10))
        or v6_metadata.get("coverage_generation_permitted") is not False
        or v6_metadata.get("scientific_result_execution_path_present") is not False
    ):
        raise RuntimeError("CXR v5/v6 lineage or fresh-bank binding differs")

    v6_confirmation_root = (
        v6_development_root.parent
        / "controlled_clinical_fidelity_v6_mimic_cxr_confirmation"
    )
    if v6_confirmation_root.exists():
        raise RuntimeError("CXR v6 confirmation root must remain absent after terminal NO-GO")

    setting_id = "mimic_cxr_gamma_minus4"
    gate_reason = (
        "V5_CONFIRMATION_COMPLETE_NO_GO;V6_DEVELOPMENT_NO_GO;TERMINAL_NO_V7"
    )
    status_rows = [
        _status_row(
            reporting_family=CXR_TERMINAL_FAMILY,
            setting_id=setting_id,
            display_label=DATASET_LABELS["mimic_cxr"],
            dataset="mimic_cxr",
            source_protocol=CXR_V6_PROTOCOL,
            setting_type="dataset_native_clinical_terminal_precoverage_no_go",
            feedback_parameter="gamma",
            feedback_value=PRIMARY_GAMMA,
            horizon=HORIZONS["mimic_cxr"],
            panel_status="GATE_TERMINAL_NO_GO",
            interpretation_status="PRE_COVERAGE_TERMINAL_DEVELOPMENT_NO_GO",
            confirmatory=False,
            ranking_permitted=False,
            scientific_rows_available=False,
            n_prespecified=20,
            n_gate_eligible=None,
            gate_reason=gate_reason,
            budget_status="not_run_precoverage_gate",
            frozen_setting_sha256=str(v6_frozen["development_config_sha256"]),
            source_path=_project_path(v6_final_path),
            source_sha256=_file_sha256(v6_final_path),
        )
    ]
    scalar_rows = [
        _unavailable_scalar_row(
            reporting_family=CXR_TERMINAL_FAMILY,
            setting_id=setting_id,
            display_label=DATASET_LABELS["mimic_cxr"],
            dataset="mimic_cxr",
            source_protocol=CXR_V6_PROTOCOL,
            setting_type="dataset_native_clinical_terminal_precoverage_no_go",
            analysis_role="precoverage_terminal_no_go",
            panel_status="GATE_TERMINAL_NO_GO",
            method=method,
            n_prespecified=20,
            n_gate_eligible=None,
            source_path=_project_path(v6_final_path),
            source_sha256=_file_sha256(v6_final_path),
        )
        for method in METHODS
    ]
    return _sources_from_rows(
        status_rows,
        [],
        scalar_rows,
        [],
        input_contracts={
            "status": "terminal_precoverage_no_go",
            "v5_confirmation": {
                "protocol": CXR_V5_PROTOCOL,
                "input_root": _project_path(v5_confirmation_root),
                "manifest_sha256": CXR_V5_MANIFEST_SHA256,
                "complete_sha256": CXR_V5_COMPLETE_SHA256,
                "final_status_sha256": CXR_V5_FINAL_SHA256,
                "artifact_count": v5_manifest["artifact_count"],
                "k0_pass_count": 18,
                "structural_pass_count": 20,
                "support_pass_count": 20,
                "coverage_generated": False,
            },
            "v6_development": {
                "protocol": CXR_V6_PROTOCOL,
                "input_root": _project_path(v6_development_root),
                "manifest_sha256": CXR_V6_MANIFEST_SHA256,
                "complete_sha256": CXR_V6_COMPLETE_SHA256,
                "final_status_sha256": CXR_V6_FINAL_SHA256,
                "artifact_count": v6_manifest["artifact_count"],
                "status": "DEVELOPMENT_NO_GO",
                "terminal_no_v7": True,
                "theta": None,
                "coverage_generated": False,
                "lineages": {
                    "v5_development": {"numeric": "19/20", "structural": "20/20"},
                    "v5_failed_confirmation": {
                        "numeric": "18/20",
                        "structural": "20/20",
                    },
                },
                "required_numeric": "20/20 per lineage",
                "fresh_confirmation_bank": {
                    "base_seeds": "120000..120190 step 10",
                    "base_seed_count": 20,
                    "rng_stream_count": 341,
                    "formal_rng_consumed": False,
                    "collision_count": 0,
                    "confirmation_root_present": False,
                },
            },
        },
    )


def load_production_sources(root: Path) -> ReportingSources:
    old_status, old_stage, old_scalar, contract = legacy.load_production_sources(root)
    status_rows = []
    for row in old_status.itertuples(index=False):
        status_rows.append(
            _status_row(
                reporting_family="production_no_gamma_robustness",
                setting_id=str(row.setting_id),
                display_label=str(row.display_label),
                dataset=str(row.dataset),
                source_protocol=str(contract["protocol"]),
                setting_type=str(row.setting_type),
                feedback_parameter=str(row.feedback_parameter),
                feedback_value=_nullable_float(row.feedback_value),
                horizon=int(row.horizon),
                panel_status="CURVES_ROBUSTNESS_ONLY",
                interpretation_status="FROZEN_PRODUCTION_ROBUSTNESS_ONLY",
                confirmatory=False,
                ranking_permitted=False,
                scientific_rows_available=True,
                n_prespecified=int(row.n_prespecified),
                n_gate_eligible=int(row.n_prespecified),
                gate_reason="",
                budget_status="consumed_complete_science",
                frozen_setting_sha256="",
                source_path=str(row.source_path),
                source_sha256=str(row.source_sha256),
            )
        )
    stage_rows = []
    for row in old_stage.itertuples(index=False):
        stage_rows.append(
            _stage_row(
                reporting_family="production_no_gamma_robustness",
                setting_id=str(row.setting_id),
                display_label=str(row.display_label),
                dataset=str(row.dataset),
                source_protocol=str(contract["protocol"]),
                setting_type=str(row.setting_type),
                feedback_parameter=str(row.feedback_parameter),
                feedback_value=_nullable_float(row.feedback_value),
                analysis_role="frozen_production_robustness_only",
                panel_status="CURVES_ROBUSTNESS_ONLY",
                confirmatory=False,
                ranking_permitted=False,
                method=str(row.method),
                calibration=None,
                grid=None,
                adaptation=int(row.target_adaptation_trajectories_per_seed),
                evaluation=int(row.evaluation_trajectories_per_seed),
                budget_status="consumed_complete_science",
                horizon=int(row.horizon),
                stage=int(row.stage_zero_based),
                n_prespecified=int(row.n_prespecified),
                n_gate_eligible=int(row.n_gate_eligible),
                n_selected=int(row.n_selected),
                coverage_conditioning=str(row.coverage_conditioning),
                coverage=float(row.coverage_mean),
                coverage_interval=(
                    float(row.coverage_ci95_lower),
                    float(row.coverage_ci95_upper),
                ),
                coverage_interval_definition=str(row.coverage_interval_definition),
                width=float(row.normalized_width_mean),
                width_interval=(
                    float(row.normalized_width_ci95_lower),
                    float(row.normalized_width_ci95_upper),
                ),
                width_interval_definition=str(row.normalized_width_interval_definition),
                normalized_width_definition=str(row.normalized_width_definition),
                source_path=str(row.source_path),
                source_sha256=str(row.source_sha256),
            )
        )
    scalar_rows = []
    for row in old_scalar.itertuples(index=False):
        scalar_rows.append(
            _scalar_row(
                reporting_family="production_no_gamma_robustness",
                setting_id=str(row.setting_id),
                display_label=str(row.display_label),
                dataset=str(row.dataset),
                source_protocol=str(contract["protocol"]),
                setting_type=str(row.setting_type),
                feedback_parameter=str(row.feedback_parameter),
                feedback_value=_nullable_float(row.feedback_value),
                analysis_role="frozen_production_robustness_only",
                panel_status="CURVES_ROBUSTNESS_ONLY",
                confirmatory=False,
                ranking_permitted=False,
                method=str(row.method),
                calibration=None,
                grid=None,
                adaptation=int(row.target_adaptation_trajectories_per_seed),
                evaluation=int(row.evaluation_trajectories_per_seed),
                budget_status="consumed_complete_science",
                n_prespecified=int(row.n_prespecified),
                n_gate_eligible=int(row.n_gate_eligible),
                n_selected=int(row.n_selected),
                metric_available=True,
                coverage_conditioning=str(row.coverage_conditioning),
                selection_rate=float(row.selection_rate),
                selection_interval=(
                    float(row.selection_rate_ci95_lower),
                    float(row.selection_rate_ci95_upper),
                ),
                wsc=float(row.wsc),
                wsc_interval=(float(row.wsc_ci95_lower), float(row.wsc_ci95_upper)),
                wsc_interval_definition=str(row.wsc_interval_definition),
                worst_stage=int(row.worst_stage_zero_based),
                mean_coverage=float(row.mean_coverage),
                mean_coverage_interval=(
                    float(row.mean_coverage_ci95_lower),
                    float(row.mean_coverage_ci95_upper),
                ),
                mean_width=float(row.mean_normalized_width),
                mean_width_interval=(
                    float(row.mean_normalized_width_ci95_lower),
                    float(row.mean_normalized_width_ci95_upper),
                ),
                normalized_width_definition=str(row.normalized_width_definition),
                point_attainment=None,
                interval_attainment=None,
                point_eligible=None,
                source_path=str(row.source_path),
                source_sha256=str(row.source_sha256),
            )
        )
    return _sources_from_rows(
        status_rows,
        stage_rows,
        scalar_rows,
        [],
        input_contracts={**contract, "analysis_role": "robustness_only_no_gamma"},
    )


def _sources_from_rows(
    status_rows: Sequence[Mapping[str, Any]],
    stage_rows: Sequence[Mapping[str, Any]],
    scalar_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    input_contracts: Mapping[str, Any],
) -> ReportingSources:
    return ReportingSources(
        status=pd.DataFrame(status_rows, columns=STATUS_COLUMNS),
        stage=pd.DataFrame(stage_rows, columns=STAGE_COLUMNS),
        scalar=pd.DataFrame(scalar_rows, columns=SCALAR_COLUMNS),
        paired=pd.DataFrame(paired_rows, columns=PAIRED_COLUMNS),
        input_contracts=input_contracts,
    )


def _status_row(**values: Any) -> dict[str, Any]:
    row = {"schema_version": 2, "method_count": len(METHODS), **values}
    if set(row) != set(STATUS_COLUMNS):
        raise RuntimeError("v2 status row schema differs")
    return row


def _method_context(**values: Any) -> dict[str, Any]:
    method = str(values["method"])
    return {
        **values,
        "feedback_parameter": "gamma",
        "calibration": CALIBRATION_BUDGET,
        "grid": GRID_BUDGET,
        "adaptation": TARGET_ADAPTATION_BUDGET[method],
        "evaluation": EVALUATION_BUDGET,
        "budget_status": "consumed_complete_science",
    }


def _stage_rows_from_cell(
    common: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    coverage_ci_field: str,
    width_ci_field: str,
) -> list[dict[str, Any]]:
    n_selected = int(common["n_selected"])
    if n_selected == 0:
        return []
    horizon = HORIZONS[str(common["dataset"])]
    coverage = _finite_vector(cell.get("target_coverage_by_stage"), horizon, "coverage")
    coverage_ci = _finite_intervals(
        cell.get(coverage_ci_field), coverage, "coverage interval"
    )
    width = _finite_vector(
        cell.get("target_normalized_width_by_stage"),
        horizon,
        "normalized width",
        positive=True,
    )
    width_ci = _finite_intervals(
        cell.get(width_ci_field), width, "normalized-width interval", positive=True
    )
    return [
        _stage_row(
            reporting_family=str(common["reporting_family"]),
            setting_id=str(common["setting_id"]),
            display_label=str(common["display_label"]),
            dataset=str(common["dataset"]),
            source_protocol=str(common["source_protocol"]),
            setting_type=str(common["setting_type"]),
            feedback_parameter=str(common["feedback_parameter"]),
            feedback_value=float(common["feedback_value"]),
            analysis_role=str(common["analysis_role"]),
            panel_status=str(common["panel_status"]),
            confirmatory=bool(common["confirmatory"]),
            ranking_permitted=bool(common["ranking_permitted"]),
            method=str(common["method"]),
            calibration=int(common["calibration"]),
            grid=int(common["grid"]),
            adaptation=int(common["adaptation"]),
            evaluation=int(common["evaluation"]),
            budget_status=str(common["budget_status"]),
            horizon=horizon,
            stage=stage,
            n_prespecified=int(common["n_prespecified"]),
            n_gate_eligible=int(common["n_gate_eligible"]),
            n_selected=n_selected,
            coverage_conditioning=str(common["coverage_conditioning"]),
            coverage=float(coverage[stage]),
            coverage_interval=(
                float(coverage_ci[stage, 0]),
                float(coverage_ci[stage, 1]),
            ),
            coverage_interval_definition=(
                "pointwise two-sided 95% Student-t interval across "
                "method-selected complete seeds"
            ),
            width=float(width[stage]),
            width_interval=(float(width_ci[stage, 0]), float(width_ci[stage, 1])),
            width_interval_definition=(
                "pointwise two-sided 95% Student-t interval across "
                "method-selected complete seeds"
            ),
            normalized_width_definition=str(common["normalized_width_definition"]),
            source_path=str(common["source_path"]),
            source_sha256=str(common["source_sha256"]),
        )
        for stage in range(horizon)
    ]


def _stage_row(
    *,
    reporting_family: str,
    setting_id: str,
    display_label: str,
    dataset: str,
    source_protocol: str,
    setting_type: str,
    feedback_parameter: str,
    feedback_value: float | None,
    analysis_role: str,
    panel_status: str,
    confirmatory: bool,
    ranking_permitted: bool,
    method: str,
    calibration: int | None,
    grid: int | None,
    adaptation: int,
    evaluation: int,
    budget_status: str,
    horizon: int,
    stage: int,
    n_prespecified: int,
    n_gate_eligible: int,
    n_selected: int,
    coverage_conditioning: str,
    coverage: float,
    coverage_interval: tuple[float, float],
    coverage_interval_definition: str,
    width: float,
    width_interval: tuple[float, float],
    width_interval_definition: str,
    normalized_width_definition: str,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    row = {
        "schema_version": 2,
        "reporting_family": reporting_family,
        "setting_id": setting_id,
        "display_label": display_label,
        "dataset": dataset,
        "source_protocol": source_protocol,
        "setting_type": setting_type,
        "feedback_parameter": feedback_parameter,
        "feedback_value": feedback_value,
        "analysis_role": analysis_role,
        "panel_status": panel_status,
        "confirmatory": confirmatory,
        "ranking_permitted": ranking_permitted,
        "method": method,
        "information_regime": INFORMATION_REGIME[method],
        "calibration_trajectories_per_seed": calibration,
        "grid_trajectories_per_seed": grid,
        "target_adaptation_trajectories_per_seed": adaptation,
        "evaluation_trajectories_per_seed": evaluation,
        "budget_status": budget_status,
        "horizon": horizon,
        "stage_zero_based": stage,
        "n_prespecified": n_prespecified,
        "n_gate_eligible": n_gate_eligible,
        "n_selected": n_selected,
        "coverage_conditioning": coverage_conditioning,
        "coverage_target": TARGET,
        "coverage_mean": coverage,
        "coverage_ci95_lower": coverage_interval[0],
        "coverage_ci95_upper": coverage_interval[1],
        "coverage_interval_definition": coverage_interval_definition,
        "coverage_interval_scope": "pointwise, not simultaneous",
        "normalized_width_mean": width,
        "normalized_width_ci95_lower": width_interval[0],
        "normalized_width_ci95_upper": width_interval[1],
        "normalized_width_interval_definition": width_interval_definition,
        "normalized_width_interval_scope": "pointwise, not simultaneous",
        "normalized_width_definition": normalized_width_definition,
        "source_path": source_path,
        "source_sha256": source_sha256,
    }
    if set(row) != set(STAGE_COLUMNS):
        raise RuntimeError("v2 stage row schema differs")
    return row


def _scalar_from_cell(
    common: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    stored_point_eligibility: bool,
) -> dict[str, Any]:
    n_selected = int(common["n_selected"])
    selection = float(cell["selection_rate"])
    selection_ci = _finite_pair(cell["selection_rate_ci95"], "selection interval")
    if n_selected == 0:
        return _scalar_row(
            **_scalar_common_arguments(common),
            metric_available=False,
            selection_rate=selection,
            selection_interval=selection_ci,
            wsc=None,
            wsc_interval=(None, None),
            wsc_interval_definition=(
                "10000-draw complete-seed-stage-vector percentile bootstrap"
            ),
            worst_stage=None,
            mean_coverage=None,
            mean_coverage_interval=(None, None),
            mean_width=None,
            mean_width_interval=(None, None),
            point_attainment=None,
            interval_attainment=None,
            point_eligible=False if bool(common["confirmatory"]) else None,
        )
    wsc = float(cell["target_marginal_worst_coverage"])
    wsc_ci = _finite_pair(cell["target_wsc_ci95"], "WSC interval")
    mean_coverage = float(cell["target_mean_coverage"])
    mean_coverage_ci = _finite_pair(
        cell["target_mean_coverage_ci95"], "MeanCov interval"
    )
    mean_width = float(cell["mean_target_normalized_width"])
    mean_width_ci = _finite_pair(
        cell["mean_target_normalized_width_ci95"], "mean-width interval", positive=True
    )
    confirmatory = bool(common["confirmatory"])
    point_attainment = bool(wsc >= TARGET) if confirmatory else None
    interval_attainment = bool(wsc_ci[0] >= TARGET) if confirmatory else None
    point_eligible = (
        bool(selection >= 0.95 and point_attainment) if confirmatory else None
    )
    if stored_point_eligibility:
        if (
            cell.get("point_attainment_at_0.90") != point_attainment
            or cell.get("wsc_interval_attainment_at_0.90") != interval_attainment
            or cell.get("point_eligible") != point_eligible
        ):
            raise RuntimeError("stored clinical point/interval eligibility differs")
    return _scalar_row(
        **_scalar_common_arguments(common),
        metric_available=True,
        selection_rate=selection,
        selection_interval=selection_ci,
        wsc=wsc,
        wsc_interval=wsc_ci,
        wsc_interval_definition=(
            "10000-draw complete-seed-stage-vector percentile bootstrap"
        ),
        worst_stage=int(cell["target_worst_stage_zero_based"]),
        mean_coverage=mean_coverage,
        mean_coverage_interval=mean_coverage_ci,
        mean_width=mean_width,
        mean_width_interval=mean_width_ci,
        point_attainment=point_attainment,
        interval_attainment=interval_attainment,
        point_eligible=point_eligible,
    )


def _scalar_common_arguments(common: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reporting_family": str(common["reporting_family"]),
        "setting_id": str(common["setting_id"]),
        "display_label": str(common["display_label"]),
        "dataset": str(common["dataset"]),
        "source_protocol": str(common["source_protocol"]),
        "setting_type": str(common["setting_type"]),
        "feedback_parameter": str(common["feedback_parameter"]),
        "feedback_value": float(common["feedback_value"]),
        "analysis_role": str(common["analysis_role"]),
        "panel_status": str(common["panel_status"]),
        "confirmatory": bool(common["confirmatory"]),
        "ranking_permitted": bool(common["ranking_permitted"]),
        "method": str(common["method"]),
        "calibration": int(common["calibration"]),
        "grid": int(common["grid"]),
        "adaptation": int(common["adaptation"]),
        "evaluation": int(common["evaluation"]),
        "budget_status": str(common["budget_status"]),
        "n_prespecified": int(common["n_prespecified"]),
        "n_gate_eligible": int(common["n_gate_eligible"]),
        "n_selected": int(common["n_selected"]),
        "coverage_conditioning": str(common["coverage_conditioning"]),
        "normalized_width_definition": str(common["normalized_width_definition"]),
        "source_path": str(common["source_path"]),
        "source_sha256": str(common["source_sha256"]),
    }


def _scalar_row(
    *,
    reporting_family: str,
    setting_id: str,
    display_label: str,
    dataset: str,
    source_protocol: str,
    setting_type: str,
    feedback_parameter: str,
    feedback_value: float | None,
    analysis_role: str,
    panel_status: str,
    confirmatory: bool,
    ranking_permitted: bool,
    method: str,
    calibration: int | None,
    grid: int | None,
    adaptation: int | None,
    evaluation: int | None,
    budget_status: str,
    n_prespecified: int,
    n_gate_eligible: int | None,
    n_selected: int | None,
    metric_available: bool,
    coverage_conditioning: str,
    selection_rate: float | None,
    selection_interval: tuple[float | None, float | None],
    wsc: float | None,
    wsc_interval: tuple[float | None, float | None],
    wsc_interval_definition: str,
    worst_stage: int | None,
    mean_coverage: float | None,
    mean_coverage_interval: tuple[float | None, float | None],
    mean_width: float | None,
    mean_width_interval: tuple[float | None, float | None],
    normalized_width_definition: str,
    point_attainment: bool | None,
    interval_attainment: bool | None,
    point_eligible: bool | None,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    row = {
        "schema_version": 2,
        "reporting_family": reporting_family,
        "setting_id": setting_id,
        "display_label": display_label,
        "dataset": dataset,
        "source_protocol": source_protocol,
        "setting_type": setting_type,
        "feedback_parameter": feedback_parameter,
        "feedback_value": feedback_value,
        "analysis_role": analysis_role,
        "panel_status": panel_status,
        "confirmatory": confirmatory,
        "ranking_permitted": ranking_permitted,
        "method": method,
        "information_regime": INFORMATION_REGIME[method],
        "calibration_trajectories_per_seed": calibration,
        "grid_trajectories_per_seed": grid,
        "target_adaptation_trajectories_per_seed": adaptation,
        "evaluation_trajectories_per_seed": evaluation,
        "budget_status": budget_status,
        "n_prespecified": n_prespecified,
        "n_gate_eligible": n_gate_eligible,
        "n_selected": n_selected,
        "metric_available": metric_available,
        "coverage_conditioning": coverage_conditioning,
        "selection_rate": selection_rate,
        "selection_rate_ci95_lower": selection_interval[0],
        "selection_rate_ci95_upper": selection_interval[1],
        "selection_interval_definition": (
            "two-sided Wilson 95% interval over all 20 prespecified seeds"
        ),
        "coverage_target": TARGET,
        "wsc": wsc,
        "wsc_ci95_lower": wsc_interval[0],
        "wsc_ci95_upper": wsc_interval[1],
        "wsc_deviation_from_target_pp": (
            None if wsc is None else 100.0 * (wsc - TARGET)
        ),
        "wsc_interval_definition": wsc_interval_definition,
        "primary_metric": PRIMARY_METRIC,
        "worst_stage_zero_based": worst_stage,
        "mean_coverage": mean_coverage,
        "mean_coverage_ci95_lower": mean_coverage_interval[0],
        "mean_coverage_ci95_upper": mean_coverage_interval[1],
        "mean_coverage_deviation_from_target_pp": (
            None if mean_coverage is None else 100.0 * (mean_coverage - TARGET)
        ),
        "mean_coverage_interval_definition": (
            "two-sided 95% Student-t interval across selected seed-level means"
        ),
        "mean_normalized_width": mean_width,
        "mean_normalized_width_ci95_lower": mean_width_interval[0],
        "mean_normalized_width_ci95_upper": mean_width_interval[1],
        "mean_width_interval_definition": (
            "two-sided 95% Student-t interval across selected seed-level means"
        ),
        "normalized_width_definition": normalized_width_definition,
        "point_eligibility_rule": POINT_ELIGIBILITY_RULE,
        "point_attainment_at_target": point_attainment,
        "wsc_interval_attainment_at_target": interval_attainment,
        "point_eligible": point_eligible,
        "efficiency_rank_defined": False,
        "narrowest_point_eligible": None,
        "source_path": source_path,
        "source_sha256": source_sha256,
    }
    if set(row) != set(SCALAR_COLUMNS):
        raise RuntimeError("v2 scalar row schema differs")
    return row


def _unavailable_scalar_row(
    *,
    reporting_family: str,
    setting_id: str,
    display_label: str,
    dataset: str,
    source_protocol: str,
    setting_type: str,
    analysis_role: str,
    panel_status: str,
    method: str,
    n_prespecified: int,
    n_gate_eligible: int | None,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    return _scalar_row(
        reporting_family=reporting_family,
        setting_id=setting_id,
        display_label=display_label,
        dataset=dataset,
        source_protocol=source_protocol,
        setting_type=setting_type,
        feedback_parameter="gamma",
        feedback_value=PRIMARY_GAMMA,
        analysis_role=analysis_role,
        panel_status=panel_status,
        confirmatory=False,
        ranking_permitted=False,
        method=method,
        calibration=None,
        grid=None,
        adaptation=None,
        evaluation=None,
        budget_status="not_run_precoverage_gate",
        n_prespecified=n_prespecified,
        n_gate_eligible=n_gate_eligible,
        n_selected=None,
        metric_available=False,
        coverage_conditioning=(
            "not available: terminal pre-coverage NO-GO; coverage not generated"
        ),
        selection_rate=None,
        selection_interval=(None, None),
        wsc=None,
        wsc_interval=(None, None),
        wsc_interval_definition="not available: terminal pre-coverage NO-GO",
        worst_stage=None,
        mean_coverage=None,
        mean_coverage_interval=(None, None),
        mean_width=None,
        mean_width_interval=(None, None),
        normalized_width_definition=CLINICAL_WIDTH_DEFINITION,
        point_attainment=None,
        interval_attainment=None,
        point_eligible=None,
        source_path=source_path,
        source_sha256=source_sha256,
    )


def _paired_rows(
    aggregate: Mapping[str, Any],
    *,
    reporting_family: str,
    setting_id: str,
    dataset: str,
    feedback_value: float,
    confirmatory: bool,
    source_path: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    if not confirmatory:
        return []
    paired = _mapping(
        aggregate.get("paired_scpcp_comparisons"), "paired SC-PCP comparisons"
    )
    expected = set(METHODS) - {"SC-PCP"}
    if set(paired) != expected:
        raise RuntimeError("confirmatory paired-comparison method set differs")
    rows = []
    for baseline in METHODS:
        if baseline == "SC-PCP":
            continue
        cell = _mapping(paired[baseline], f"paired comparison {baseline}")
        wsc_ci = _nullable_pair(cell["scpcp_minus_baseline_wsc_ci95"])
        width_ci = _nullable_pair(
            cell["scpcp_to_baseline_geometric_width_ratio_ci95"]
        )
        rows.append(
            {
                "schema_version": 2,
                "reporting_family": reporting_family,
                "setting_id": setting_id,
                "dataset": dataset,
                "feedback_value": feedback_value,
                "confirmatory": True,
                "ranking_permitted": True,
                "baseline": baseline,
                "paired_selected_seeds": int(cell["paired_selected_seeds"]),
                "scpcp_minus_baseline_wsc": _nullable_float(
                    cell["scpcp_minus_baseline_wsc"]
                ),
                "scpcp_minus_baseline_wsc_ci95_lower": wsc_ci[0],
                "scpcp_minus_baseline_wsc_ci95_upper": wsc_ci[1],
                "scpcp_to_baseline_geometric_width_ratio": _nullable_float(
                    cell["scpcp_to_baseline_geometric_width_ratio"]
                ),
                "scpcp_to_baseline_geometric_width_ratio_ci95_lower": width_ci[0],
                "scpcp_to_baseline_geometric_width_ratio_ci95_upper": width_ci[1],
                "interval_definition": (
                    "10000-draw paired complete-seed-vector percentile bootstrap"
                ),
                "source_path": source_path,
                "source_sha256": source_sha256,
            }
        )
    return rows


def assign_efficiency_ranking(scalar: pd.DataFrame) -> pd.DataFrame:
    ranked = scalar.copy()
    for _, group in ranked.groupby(["reporting_family", "setting_id"], sort=False):
        permitted = group["ranking_permitted"].map(_as_bool).all()
        if not permitted:
            continue
        eligible = group[group["point_eligible"].map(_as_bool)]
        if eligible.empty:
            continue
        winner = eligible["mean_normalized_width"].astype(float).idxmin()
        ranked.loc[group.index, "efficiency_rank_defined"] = True
        ranked.loc[group.index, "narrowest_point_eligible"] = False
        ranked.loc[winner, "narrowest_point_eligible"] = True
    return ranked.loc[:, SCALAR_COLUMNS]


def _validate_budget(
    method: str,
    *,
    calibration: int,
    grid: int,
    adaptation: int,
    evaluation: int,
) -> None:
    if (
        calibration != CALIBRATION_BUDGET
        or grid != GRID_BUDGET
        or adaptation != TARGET_ADAPTATION_BUDGET[method]
        or evaluation != EVALUATION_BUDGET
    ):
        raise RuntimeError(f"{method} signed-gamma information budget differs")


def _gamma_aggregates(summary: Mapping[str, Any]) -> dict[float, Mapping[str, Any]]:
    values = summary.get("aggregates")
    if not isinstance(values, list) or len(values) != len(GAMMAS):
        raise RuntimeError("signed-gamma aggregate grid differs")
    aggregates = {float(row["gamma"]): _mapping(row, "gamma aggregate") for row in values}
    if tuple(aggregates) != GAMMAS:
        raise RuntimeError("signed-gamma aggregate order differs")
    return aggregates


def _validate_pinned_bundle(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_complete_sha256: str,
    size_field: str,
) -> Mapping[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"COMPLETE input root must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    complete_path = root / "COMPLETE"
    if (
        _file_sha256(manifest_path) != expected_manifest_sha256
        or _file_sha256(complete_path) != expected_complete_sha256
    ):
        raise RuntimeError("pinned manifest or COMPLETE hash differs")
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != manifest.get("artifact_count"):
        raise RuntimeError("pinned manifest artifact count differs")
    expected_paths: set[str] = {"manifest.json", "COMPLETE"}
    for record in artifacts:
        entry = _mapping(record, "manifest artifact")
        relative = _safe_relative_path(entry.get("path"))
        relative_text = relative.as_posix()
        if relative_text in expected_paths:
            raise RuntimeError("manifest artifact path is duplicated")
        expected_paths.add(relative_text)
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"manifest artifact is missing or symbolic: {relative_text}")
        if (
            path.stat().st_size != int(entry[size_field])
            or _file_sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"manifest artifact bytes differ: {relative_text}")
    observed = _observed_files(root)
    if observed != expected_paths:
        raise RuntimeError(
            "pinned COMPLETE artifact set differs; "
            f"missing={sorted(expected_paths - observed)}; "
            f"extra={sorted(observed - expected_paths)}"
        )
    return manifest


def _observed_files(root: Path) -> set[str]:
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symbolic links are forbidden in input bundles: {path}")
        if path.is_file():
            observed.add(path.relative_to(root).as_posix())
    return observed


def _safe_relative_path(value: object) -> Path:
    if not isinstance(value, str):
        raise RuntimeError("manifest path must be a string")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RuntimeError(f"unsafe manifest path: {value!r}")
    return path


def _finite_vector(
    value: object, length: int, label: str, *, positive: bool = False
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise RuntimeError(f"{label} vector differs")
    if positive and np.any(array <= 0.0):
        raise RuntimeError(f"{label} must be positive")
    if not positive and np.any((array < 0.0) | (array > 1.0)):
        raise RuntimeError(f"{label} must lie in [0,1]")
    return array


def _finite_intervals(
    value: object,
    point: np.ndarray,
    label: str,
    *,
    positive: bool = False,
) -> np.ndarray:
    intervals = np.asarray(value, dtype=float)
    if (
        intervals.shape != (len(point), 2)
        or not np.isfinite(intervals).all()
        or np.any(intervals[:, 0] > point)
        or np.any(point > intervals[:, 1])
    ):
        raise RuntimeError(f"{label} differs")
    if positive and np.any(intervals <= 0.0):
        raise RuntimeError(f"{label} must be positive")
    if not positive and np.any((intervals < 0.0) | (intervals > 1.0)):
        raise RuntimeError(f"{label} must lie in [0,1]")
    return intervals


def _finite_pair(
    value: object, label: str, *, positive: bool = False
) -> tuple[float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all() or array[0] > array[1]:
        raise RuntimeError(f"{label} differs")
    if positive and np.any(array <= 0.0):
        raise RuntimeError(f"{label} must be positive")
    if not positive and np.any((array < 0.0) | (array > 1.0)):
        raise RuntimeError(f"{label} must lie in [0,1]")
    return float(array[0]), float(array[1])


def _nullable_pair(value: object) -> tuple[float | None, float | None]:
    if value == [None, None]:
        return None, None
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all() or array[0] > array[1]:
        raise RuntimeError("paired interval differs")
    return float(array[0]), float(array[1])


def _nullable_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("numeric value is not finite")
    return number


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return value


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0", ""}:
            return False
    return bool(value)


def _gamma_id(gamma: float) -> str:
    return f"minus{abs(int(gamma))}" if gamma < 0 else f"plus{int(gamma)}"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def validate_reporting_sources(sources: ReportingSources) -> None:
    status, stage, scalar, paired = (
        sources.status,
        sources.stage,
        sources.scalar,
        sources.paired,
    )
    if tuple(status.columns) != STATUS_COLUMNS:
        raise RuntimeError("v2 status schema differs")
    if tuple(stage.columns) != STAGE_COLUMNS:
        raise RuntimeError("v2 stage schema differs")
    if tuple(scalar.columns) != SCALAR_COLUMNS:
        raise RuntimeError("v2 scalar schema differs")
    if tuple(paired.columns) != PAIRED_COLUMNS:
        raise RuntimeError("v2 paired schema differs")
    if status.duplicated(["reporting_family", "setting_id"]).any():
        raise RuntimeError("v2 status keys are duplicated")
    if stage.duplicated(
        ["reporting_family", "setting_id", "method", "stage_zero_based"]
    ).any():
        raise RuntimeError("v2 stage keys are duplicated")
    if scalar.duplicated(["reporting_family", "setting_id", "method"]).any():
        raise RuntimeError("v2 scalar keys are duplicated")
    if paired.duplicated(["reporting_family", "setting_id", "baseline"]).any():
        raise RuntimeError("v2 paired keys are duplicated")
    if set(stage["method"]) != set(METHODS) or set(scalar["method"]) != set(METHODS):
        raise RuntimeError("v2 canonical method set differs")

    signed_status = status[
        status["reporting_family"].isin(
            {"native_signed_gamma", "clinical_v4_signed_gamma"}
        )
    ]
    expected_signed = {
        (dataset, gamma)
        for dataset in CURVE_DATASETS
        for gamma in GAMMAS
    }
    if set(zip(signed_status["dataset"], signed_status["feedback_value"])) != expected_signed:
        raise RuntimeError("v2 signed dataset/gamma grid differs")
    primary = signed_status[signed_status["feedback_value"].eq(PRIMARY_GAMMA)]
    if tuple(primary["dataset"]) != CURVE_DATASETS:
        raise RuntimeError("v2 default gamma=-4 curve order differs")
    if not primary["confirmatory"].map(_as_bool).all() or not primary[
        "ranking_permitted"
    ].map(_as_bool).all():
        raise RuntimeError("v2 gamma=-4 confirmatory/ranking identity differs")
    nonprimary = signed_status[~signed_status["feedback_value"].eq(PRIMARY_GAMMA)]
    if nonprimary["confirmatory"].map(_as_bool).any() or nonprimary[
        "ranking_permitted"
    ].map(_as_bool).any():
        raise RuntimeError("v2 descriptive signed cells permit ranking")

    cxr = status[status["reporting_family"].eq(CXR_TERMINAL_FAMILY)]
    if (
        len(cxr) != 1
        or cxr.iloc[0]["dataset"] != "mimic_cxr"
        or float(cxr.iloc[0]["feedback_value"]) != PRIMARY_GAMMA
        or cxr.iloc[0]["panel_status"] != "GATE_TERMINAL_NO_GO"
        or cxr.iloc[0]["interpretation_status"]
        != "PRE_COVERAGE_TERMINAL_DEVELOPMENT_NO_GO"
        or cxr.iloc[0]["gate_reason"]
        != "V5_CONFIRMATION_COMPLETE_NO_GO;V6_DEVELOPMENT_NO_GO;TERMINAL_NO_V7"
        or _as_bool(cxr.iloc[0]["scientific_rows_available"])
        or _as_bool(cxr.iloc[0]["ranking_permitted"])
    ):
        raise RuntimeError("v2 CXR terminal pre-coverage gate identity differs")
    if (
        stage["reporting_family"].eq(CXR_TERMINAL_FAMILY)
        & stage["dataset"].eq("mimic_cxr")
    ).any():
        raise RuntimeError("v2 CXR gate contains stage science")
    cxr_scalar = scalar[scalar["reporting_family"].eq(CXR_TERMINAL_FAMILY)]
    cxr_science_columns = [
        "selection_rate",
        "selection_rate_ci95_lower",
        "selection_rate_ci95_upper",
        "wsc",
        "wsc_ci95_lower",
        "wsc_ci95_upper",
        "wsc_deviation_from_target_pp",
        "worst_stage_zero_based",
        "mean_coverage",
        "mean_coverage_ci95_lower",
        "mean_coverage_ci95_upper",
        "mean_coverage_deviation_from_target_pp",
        "mean_normalized_width",
        "mean_normalized_width_ci95_lower",
        "mean_normalized_width_ci95_upper",
        "point_attainment_at_target",
        "wsc_interval_attainment_at_target",
        "point_eligible",
        "narrowest_point_eligible",
    ]
    if (
        tuple(cxr_scalar["method"]) != METHODS
        or cxr_scalar["metric_available"].map(_as_bool).any()
        or cxr_scalar["n_selected"].notna().any()
        or cxr_scalar[cxr_science_columns].notna().any().any()
        or set(cxr_scalar["budget_status"]) != {"not_run_precoverage_gate"}
    ):
        raise RuntimeError("v2 CXR unavailable scalar rows differ")

    production = status[
        status["reporting_family"].eq("production_no_gamma_robustness")
    ]
    if tuple(production["dataset"]) != PRODUCTION_DATASETS:
        raise RuntimeError("v2 production robustness order differs")
    if production["confirmatory"].map(_as_bool).any() or production[
        "ranking_permitted"
    ].map(_as_bool).any():
        raise RuntimeError("v2 production/no-gamma entered confirmatory reporting")

    expected_scalar_keys = {
        (row.reporting_family, row.setting_id, method)
        for row in status.itertuples(index=False)
        for method in METHODS
    }
    observed_scalar_keys = set(
        zip(scalar["reporting_family"], scalar["setting_id"], scalar["method"])
    )
    if observed_scalar_keys != expected_scalar_keys:
        raise RuntimeError("v2 scalar setting/method grid differs")

    if not np.allclose(stage["coverage_target"].to_numpy(float), TARGET, rtol=0, atol=0):
        raise RuntimeError("v2 stage target differs")
    if not np.allclose(scalar["coverage_target"].to_numpy(float), TARGET, rtol=0, atol=0):
        raise RuntimeError("v2 scalar target differs")
    if set(scalar["primary_metric"]) != {PRIMARY_METRIC}:
        raise RuntimeError("v2 primary metric differs")
    native_widths = stage[
        stage["reporting_family"].eq("native_signed_gamma")
        & stage["dataset"].eq("synthetic")
    ]["normalized_width_definition"]
    if set(native_widths) != {NATIVE_WIDTH_DEFINITION}:
        raise RuntimeError("v2 Native width definition differs")
    clinical_widths = stage[
        stage["reporting_family"].eq("clinical_v4_signed_gamma")
        & stage["dataset"].isin({"mimic_iv", "eicu", "inspire"})
    ]["normalized_width_definition"]
    if set(clinical_widths) != {CLINICAL_WIDTH_DEFINITION}:
        raise RuntimeError("v2 clinical width definition differs")

    status_index = status.set_index(["reporting_family", "setting_id"])
    for row in scalar.itertuples(index=False):
        setting = status_index.loc[(row.reporting_family, row.setting_id)]
        profiles = stage[
            stage["reporting_family"].eq(row.reporting_family)
            & stage["setting_id"].eq(row.setting_id)
            & stage["method"].eq(row.method)
        ].sort_values("stage_zero_based")
        available = _as_bool(row.metric_available)
        if not available:
            if not profiles.empty:
                raise RuntimeError("unavailable scalar row has stage profiles")
            continue
        if not _as_bool(setting["scientific_rows_available"]):
            raise RuntimeError("available scalar row belongs to a gated setting")
        horizon = int(setting["horizon"])
        if tuple(profiles["stage_zero_based"].astype(int)) != tuple(range(horizon)):
            raise RuntimeError("available method stage grid differs")
        coverage = profiles["coverage_mean"].to_numpy(float)
        width = profiles["normalized_width_mean"].to_numpy(float)
        if not math.isclose(float(row.wsc), float(coverage.min()), rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("v2 WSC differs from min_t mean_seed(C_seed,t)")
        if int(row.worst_stage_zero_based) != int(coverage.argmin()):
            raise RuntimeError("v2 worst stage differs from first argmin")
        tolerance = 5e-7 if row.reporting_family == "production_no_gamma_robustness" else 1e-12
        if not math.isclose(
            float(row.mean_coverage), float(coverage.mean()), rel_tol=0, abs_tol=tolerance
        ):
            raise RuntimeError("v2 MeanCov differs from stage means")
        if not math.isclose(
            float(row.mean_normalized_width), float(width.mean()), rel_tol=0, abs_tol=5e-7
        ):
            raise RuntimeError("v2 mean width differs from stage widths")
        if not math.isclose(
            float(row.selection_rate),
            int(row.n_selected) / int(row.n_prespecified),
            rel_tol=0,
            abs_tol=1e-14,
        ):
            raise RuntimeError("v2 selection denominator differs")
        if not math.isclose(
            float(row.wsc_deviation_from_target_pp),
            100.0 * (float(row.wsc) - TARGET),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("v2 WSC gap differs")
        if not math.isclose(
            float(row.mean_coverage_deviation_from_target_pp),
            100.0 * (float(row.mean_coverage) - TARGET),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("v2 MeanCov gap differs")
        if row.reporting_family != "production_no_gamma_robustness":
            expected_point = bool(float(row.wsc) >= TARGET) if _as_bool(row.confirmatory) else None
            expected_interval = (
                bool(float(row.wsc_ci95_lower) >= TARGET)
                if _as_bool(row.confirmatory)
                else None
            )
            if (
                _nullable_bool(row.point_attainment_at_target) != expected_point
                or _nullable_bool(row.wsc_interval_attainment_at_target) != expected_interval
            ):
                raise RuntimeError("v2 point/interval attainment differs")

    stage_values = stage[
        ["coverage_ci95_lower", "coverage_mean", "coverage_ci95_upper"]
    ].to_numpy(float)
    width_values = stage[
        [
            "normalized_width_ci95_lower",
            "normalized_width_mean",
            "normalized_width_ci95_upper",
        ]
    ].to_numpy(float)
    if (
        not np.isfinite(stage_values).all()
        or np.any((stage_values < 0) | (stage_values > 1))
        or np.any(stage_values[:, 0] > stage_values[:, 1])
        or np.any(stage_values[:, 1] > stage_values[:, 2])
    ):
        raise RuntimeError("v2 stage coverage intervals differ")
    if (
        not np.isfinite(width_values).all()
        or np.any(width_values <= 0)
        or np.any(width_values[:, 0] > width_values[:, 1])
        or np.any(width_values[:, 1] > width_values[:, 2])
    ):
        raise RuntimeError("v2 stage width intervals differ")
    signed_stage = stage[
        stage["reporting_family"].isin(
            {"native_signed_gamma", "clinical_v4_signed_gamma"}
        )
    ]
    if set(signed_stage["coverage_interval_scope"]) != {"pointwise, not simultaneous"}:
        raise RuntimeError("v2 stage coverage scope differs")
    if not signed_stage["coverage_interval_definition"].str.contains("Student-t").all():
        raise RuntimeError("v2 signed stage coverage CI method differs")
    if not signed_stage["normalized_width_interval_definition"].str.contains(
        "Student-t"
    ).all():
        raise RuntimeError("v2 signed stage width CI method differs")

    for frame in (stage, scalar):
        if not frame["method"].map(INFORMATION_REGIME).eq(frame["information_regime"]).all():
            raise RuntimeError("v2 information regime differs")
        signed = frame[
            frame["reporting_family"].isin(
                {"native_signed_gamma", "clinical_v4_signed_gamma"}
            )
        ]
        if not signed["method"].map(TARGET_ADAPTATION_BUDGET).eq(
            signed["target_adaptation_trajectories_per_seed"]
        ).all():
            raise RuntimeError("v2 signed adaptation budget differs")
        for field, expected in (
            ("calibration_trajectories_per_seed", CALIBRATION_BUDGET),
            ("grid_trajectories_per_seed", GRID_BUDGET),
            ("evaluation_trajectories_per_seed", EVALUATION_BUDGET),
        ):
            if not signed[field].eq(expected).all():
                raise RuntimeError(f"v2 signed {field} differs")

    expected_paired = {
        (dataset, baseline)
        for dataset in CURVE_DATASETS
        for baseline in METHODS
        if baseline != "SC-PCP"
    }
    if set(zip(paired["dataset"], paired["baseline"])) != expected_paired:
        raise RuntimeError("v2 paired contrast grid differs")
    if not paired["confirmatory"].map(_as_bool).all() or not paired[
        "ranking_permitted"
    ].map(_as_bool).all():
        raise RuntimeError("v2 paired contrasts are not confirmatory gamma=-4 only")

    for _, group in scalar.groupby(["reporting_family", "setting_id"], sort=False):
        rank_defined = group["efficiency_rank_defined"].map(_as_bool)
        if rank_defined.any():
            if not rank_defined.all():
                raise RuntimeError("v2 efficiency rank status differs within setting")
            winner = group[group["narrowest_point_eligible"].map(_as_bool)]
            eligible = group[group["point_eligible"].map(_as_bool)]
            if (
                len(winner) != 1
                or eligible.empty
                or float(winner.iloc[0]["mean_normalized_width"])
                != float(eligible["mean_normalized_width"].min())
            ):
                raise RuntimeError("v2 within-dataset eligible-width winner differs")
        elif group["narrowest_point_eligible"].notna().any():
            raise RuntimeError("v2 undefined efficiency ranking contains a winner")


def _nullable_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    return _as_bool(value)


def render_report(
    *,
    native_input: Path,
    clinical_input: Path,
    production_input: Path,
    work_output: Path,
    paper_output: Path,
    cxr_v5_confirmation_input: Path = DEFAULT_CXR_V5_CONFIRMATION_INPUT,
    cxr_v6_development_input: Path = DEFAULT_CXR_V6_DEVELOPMENT_INPUT,
) -> None:
    if work_output.exists() or paper_output.exists():
        raise FileExistsError("work-output and paper-output must both be fresh")
    if work_output == paper_output:
        raise ValueError("work-output and paper-output must differ")
    sources = build_reporting_sources(
        native_input=native_input,
        clinical_input=clinical_input,
        cxr_v5_confirmation_input=cxr_v5_confirmation_input,
        cxr_v6_development_input=cxr_v6_development_input,
        production_input=production_input,
    )
    validate_claim_contract(sources.scalar)

    work_output.parent.mkdir(parents=True, exist_ok=True)
    paper_output.parent.mkdir(parents=True, exist_ok=True)
    staged_work = Path(
        tempfile.mkdtemp(prefix=f".{work_output.name}-", dir=work_output.parent)
    )
    staged_paper = Path(
        tempfile.mkdtemp(prefix=f".{paper_output.name}-", dir=paper_output.parent)
    )
    try:
        _write_csv(staged_work / "setting_status.csv", sources.status)
        _write_csv(staged_work / "coverage_stage_profiles.csv", sources.stage)
        _write_csv(staged_work / "coverage_scalar_summary.csv", sources.scalar)
        _write_csv(staged_work / "paired_scpcp_contrasts.csv", sources.paired)
        _validate_written_sources(staged_work)
        _write_contract(staged_work / "figure_contract.json", sources)
        _write_qa(staged_work / "figure_qa.md", sources)

        apply_publication_style()
        figures = {
            MAIN_STAGE_STEM: render_gamma_minus4_stagewise(sources),
            MAIN_TABLE_STEM: render_gamma_minus4_table(sources),
            SIGNED_FIGURE_STEM: render_signed_gamma_figure(sources),
            PRODUCTION_TABLE_STEM: render_production_table(sources),
        }
        for stem, figure in figures.items():
            export_figure(figure, staged_work / stem, title=_output_title(stem))
        for name in sorted(PAPER_FILES):
            shutil.copyfile(staged_work / name, staged_paper / name)
        _write_render_manifest(
            staged_work / "render_manifest.json",
            work_root=staged_work,
            paper_root=staged_paper,
        )
        _write_complete(staged_work)
        validate_rendered_outputs(staged_work, staged_paper)
        os.replace(staged_work, work_output)
        os.replace(staged_paper, paper_output)
    except BaseException:
        shutil.rmtree(staged_work, ignore_errors=True)
        shutil.rmtree(staged_paper, ignore_errors=True)
        raise


def validate_claim_contract(scalar: pd.DataFrame) -> None:
    clinical = scalar[
        scalar["reporting_family"].eq("clinical_v4_signed_gamma")
        & scalar["feedback_value"].eq(PRIMARY_GAMMA)
    ]
    if set(clinical["dataset"]) != {"mimic_iv", "eicu", "inspire"}:
        raise RuntimeError("clinical gamma=-4 claim grid differs")
    for dataset, group in clinical.groupby("dataset", sort=False):
        rows = group.set_index("method")
        distances = (rows["wsc"].astype(float) - TARGET).abs()
        if distances.idxmin() != "SC-PCP":
            raise RuntimeError(f"{dataset} SC-PCP is not closest to nominal WSC")
        scpcp = rows.loc["SC-PCP"]
        mfcs = rows.loc["MFCS"]
        if (
            float(scpcp["mean_coverage_ci95_lower"]) <= TARGET
            or _as_bool(scpcp["wsc_interval_attainment_at_target"])
            or not _as_bool(mfcs["wsc_interval_attainment_at_target"])
            or float(mfcs["mean_normalized_width"])
            <= float(scpcp["mean_normalized_width"])
        ):
            raise RuntimeError(f"{dataset} audited coverage/width claim differs")
    scpcp = clinical[clinical["method"].eq("SC-PCP")].set_index("dataset")
    point_attainment = {
        dataset: _as_bool(scpcp.loc[dataset, "point_attainment_at_target"])
        for dataset in ("mimic_iv", "eicu", "inspire")
    }
    if point_attainment != {"mimic_iv": False, "eicu": False, "inspire": True}:
        raise RuntimeError("clinical SC-PCP point-WSC attainment claim differs")


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 6.0,
            "axes.labelsize": 6.1,
            "axes.titlesize": 6.7,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.62,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "svg.hashsalt": "scpcp-complete-coverage-reporting-v4-minimal-quantitative",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _default_status(sources: ReportingSources) -> pd.DataFrame:
    curve = sources.status[
        sources.status["reporting_family"].isin(
            {"native_signed_gamma", "clinical_v4_signed_gamma"}
        )
        & sources.status["feedback_value"].eq(PRIMARY_GAMMA)
    ]
    gate = sources.status[
        sources.status["reporting_family"].eq(CXR_TERMINAL_FAMILY)
    ]
    combined = pd.concat([curve, gate], ignore_index=True)
    order = {dataset: index for index, dataset in enumerate(DATASETS)}
    return combined.assign(_order=combined["dataset"].map(order)).sort_values(
        "_order"
    )


def _default_stage(sources: ReportingSources) -> pd.DataFrame:
    return sources.stage[
        sources.stage["reporting_family"].isin(
            {"native_signed_gamma", "clinical_v4_signed_gamma"}
        )
        & sources.stage["feedback_value"].eq(PRIMARY_GAMMA)
    ]


def _default_scalar(sources: ReportingSources) -> pd.DataFrame:
    curve = sources.scalar[
        sources.scalar["reporting_family"].isin(
            {"native_signed_gamma", "clinical_v4_signed_gamma"}
        )
        & sources.scalar["feedback_value"].eq(PRIMARY_GAMMA)
    ]
    gate = sources.scalar[
        sources.scalar["reporting_family"].eq(CXR_TERMINAL_FAMILY)
    ]
    return pd.concat([curve, gate], ignore_index=True)


def render_gamma_minus4_stagewise(sources: ReportingSources) -> plt.Figure:
    status = _default_status(sources)
    profiles = _default_stage(sources)
    figure, axes = plt.subplots(2, 4, figsize=(7.20, 3.65), sharex="col")
    figure.subplots_adjust(
        left=0.068,
        right=0.992,
        bottom=0.145,
        top=0.82,
        wspace=0.36,
        hspace=0.28,
    )
    coverage_limits = _metric_limits(
        100.0 * (profiles["coverage_ci95_lower"].to_numpy(float) - TARGET),
        100.0 * (profiles["coverage_ci95_upper"].to_numpy(float) - TARGET),
        fallback=(-5.0, 5.0),
        quantum=0.5,
    )
    for column, dataset in enumerate(CURVE_DATASETS):
        setting = status[status["dataset"].eq(dataset)].iloc[0]
        coverage_axis, width_axis = axes[:, column]
        group = profiles[profiles["setting_id"].eq(setting["setting_id"])]
        coverage_axis.axhspan(coverage_limits[0], 0.0, color="#F8E9E7", zorder=0)
        coverage_axis.axhline(0.0, color="#20262B", linewidth=0.90, zorder=1)
        for method in METHODS:
            rows = group[group["method"].eq(method)].sort_values("stage_zero_based")
            _plot_stage_interval(coverage_axis, rows, method, metric="coverage")
            _plot_stage_interval(width_axis, rows, method, metric="width")
        coverage_axis.set_title(DATASET_LABELS[dataset], fontweight="bold", pad=3)
        coverage_axis.set_ylim(*coverage_limits)
        _set_width_limits(width_axis, group)
        _set_stage_axis(width_axis, HORIZONS[dataset])
        for axis in (coverage_axis, width_axis):
            axis.grid(axis="y", color="#E1E3E6", linewidth=0.40, zorder=-5)
            axis.tick_params(width=0.58, length=2.0)
        if column == 0:
            coverage_axis.set_ylabel(r"Stage coverage, $C_t-0.90$ (pp)")
            width_axis.set_ylabel("Normalized width")
        width_axis.set_xlabel("Stage, t (0-based)")
        _panel_label(coverage_axis, column)

    figure.legend(
        handles=[_legend_handle(method) for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        columnspacing=0.7,
        handlelength=1.65,
        handletextpad=0.28,
    )
    return figure


def _plot_stage_interval(
    axis: plt.Axes, rows: pd.DataFrame, method: str, *, metric: str
) -> None:
    x = rows["stage_zero_based"].to_numpy(float)
    if metric == "coverage":
        point = 100.0 * (rows["coverage_mean"].to_numpy(float) - TARGET)
        lower = 100.0 * (rows["coverage_ci95_lower"].to_numpy(float) - TARGET)
        upper = 100.0 * (rows["coverage_ci95_upper"].to_numpy(float) - TARGET)
    elif metric == "width":
        point = rows["normalized_width_mean"].to_numpy(float)
        lower = rows["normalized_width_ci95_lower"].to_numpy(float)
        upper = rows["normalized_width_ci95_upper"].to_numpy(float)
    else:
        raise ValueError(f"unknown stage metric: {metric}")
    emphasized = method in {"Standard CP", "MFCS", "SC-PCP"}
    axis.errorbar(
        x,
        point,
        yerr=np.vstack((point - lower, upper - point)),
        color=legacy.METHOD_COLORS[method],
        linestyle=legacy.METHOD_LINESTYLES[method],
        marker=legacy.METHOD_MARKERS[method],
        markersize=2.8 if method == "SC-PCP" else 2.05,
        linewidth=1.12 if method == "SC-PCP" else 0.82 if emphasized else 0.66,
        elinewidth=0.48,
        capsize=1.05,
        capthick=0.44,
        markeredgewidth=0.24,
        zorder=4 if method == "SC-PCP" else 3 if emphasized else 2,
    )


def _panel_label(axis: plt.Axes, column: int) -> None:
    axis.text(
        -0.18,
        1.12,
        chr(ord("a") + column),
        transform=axis.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="top",
    )


def render_gamma_minus4_table(sources: ReportingSources) -> plt.Figure:
    rows = _default_scalar(sources)
    return _render_scalar_table(
        rows,
        group_order=DATASETS,
        height=7.65,
    )


def render_production_table(sources: ReportingSources) -> plt.Figure:
    rows = sources.scalar[
        sources.scalar["reporting_family"].eq("production_no_gamma_robustness")
    ]
    return _render_scalar_table(
        rows,
        group_order=PRODUCTION_DATASETS,
        height=7.65,
    )


def _render_scalar_table(
    rows: pd.DataFrame,
    *,
    group_order: Sequence[str],
    height: float,
) -> plt.Figure:
    dataset_rank = {dataset: index for index, dataset in enumerate(group_order)}
    method_rank = {method: index for index, method in enumerate(METHODS)}
    ordered = rows.assign(
        _dataset_rank=rows["dataset"].map(dataset_rank),
        _method_rank=rows["method"].map(method_rank),
    ).sort_values(["_dataset_rank", "_method_rank"])
    table_rows = []
    unavailable = set()
    for row_index, row in enumerate(ordered.itertuples(index=False)):
        dataset_text = DATASET_LABELS[row.dataset] if row.method == METHODS[0] else ""
        if _as_bool(row.metric_available):
            table_rows.append(
                [
                    dataset_text,
                    row.method,
                    _format_percent_interval_with_gap(
                        row.wsc,
                        row.wsc_ci95_lower,
                        row.wsc_ci95_upper,
                        row.wsc_deviation_from_target_pp,
                    ),
                    str(int(row.worst_stage_zero_based)),
                    _format_percent_interval_with_gap(
                        row.mean_coverage,
                        row.mean_coverage_ci95_lower,
                        row.mean_coverage_ci95_upper,
                        row.mean_coverage_deviation_from_target_pp,
                    ),
                    _format_number_interval(
                        row.mean_normalized_width,
                        row.mean_normalized_width_ci95_lower,
                        row.mean_normalized_width_ci95_upper,
                    ),
                    _format_selection_with_gate(row),
                    _format_attainment(row),
                    _format_budget(row),
                ]
            )
        else:
            table_rows.append(
                [
                    dataset_text,
                    row.method,
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                ]
            )
            unavailable.add(row_index)

    figure, axis = plt.subplots(figsize=(7.20, height))
    figure.subplots_adjust(left=0.008, right=0.992, top=0.986, bottom=0.008)
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=(
            "Dataset",
            "Method",
            "WSC [95% CI]\nΔ vs 90%",
            "t*",
            "MeanCov [95% CI]\nΔ vs 90%",
            "Mean width [95% CI]",
            "Selection [Wilson]\ngate eligible",
            "Point / CI /\neligible",
            "cal/grid/\nadapt/eval",
        ),
        colLoc="center",
        cellLoc="center",
        colWidths=(0.105, 0.10, 0.145, 0.035, 0.145, 0.14, 0.155, 0.085, 0.09),
        bbox=(0.0, 0.0, 1.0, 1.0),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(4.35)
    for column in range(9):
        header = table[(0, column)]
        header.set_facecolor("#334E68")
        header.set_text_props(color="white", fontweight="bold")
        header.set_edgecolor("white")
        header.set_linewidth(0.5)
    group_colors = ("#F4F7FA", "#FFFFFF")
    for row_index, source_row in enumerate(ordered.itertuples(index=False), start=1):
        group_index = dataset_rank[source_row.dataset]
        for column in range(9):
            cell = table[(row_index, column)]
            cell.set_facecolor(
                "#F4F2EF"
                if row_index - 1 in unavailable
                else group_colors[group_index % 2]
            )
            cell.set_edgecolor("#D7DEE5")
            cell.set_linewidth(0.32)
            if column == 1:
                cell.set_text_props(ha="left")
        if source_row.method == "SC-PCP":
            table[(row_index, 1)].get_text().set_color(legacy.METHOD_COLORS["SC-PCP"])
        if _as_bool(source_row.narrowest_point_eligible):
            table[(row_index, 5)].set_facecolor("#DDECF8")
            table[(row_index, 5)].set_text_props(fontweight="bold")
        if (row_index - 1) % len(METHODS) == 0:
            for column in range(9):
                table[(row_index, column)].set_linewidth(0.72)
                table[(row_index, column)].set_edgecolor("#9AA9B5")
    return figure


def render_signed_gamma_figure(sources: ReportingSources) -> plt.Figure:
    rows = sources.scalar[
        sources.scalar["reporting_family"].isin(
            {"native_signed_gamma", "clinical_v4_signed_gamma"}
        )
    ]
    figure, axes = plt.subplots(3, 4, figsize=(7.20, 5.75), sharex="col")
    figure.subplots_adjust(
        left=0.07,
        right=0.993,
        bottom=0.105,
        top=0.86,
        wspace=0.38,
        hspace=0.27,
    )
    metrics = (
        ("wsc", "wsc_ci95_lower", "wsc_ci95_upper", "WSC (%)", 100.0),
        (
            "mean_coverage",
            "mean_coverage_ci95_lower",
            "mean_coverage_ci95_upper",
            "MeanCov (%)",
            100.0,
        ),
        (
            "mean_normalized_width",
            "mean_normalized_width_ci95_lower",
            "mean_normalized_width_ci95_upper",
            "Mean width",
            1.0,
        ),
    )
    offsets = dict(zip(METHODS, np.linspace(-0.08, 0.08, len(METHODS))))
    for column, dataset in enumerate(CURVE_DATASETS):
        dataset_rows = rows[rows["dataset"].eq(dataset)]
        for axis, (point_field, lower_field, upper_field, ylabel, scale) in zip(
            axes[:, column], metrics
        ):
            for method in METHODS:
                selected = dataset_rows[dataset_rows["method"].eq(method)].sort_values(
                    "feedback_value"
                )
                x = selected["feedback_value"].to_numpy(float) + offsets[method]
                point = scale * selected[point_field].to_numpy(float)
                lower = scale * selected[lower_field].to_numpy(float)
                upper = scale * selected[upper_field].to_numpy(float)
                axis.errorbar(
                    x,
                    point,
                    yerr=np.vstack((point - lower, upper - point)),
                    color=legacy.METHOD_COLORS[method],
                    linestyle=legacy.METHOD_LINESTYLES[method],
                    marker=legacy.METHOD_MARKERS[method],
                    markersize=2.5,
                    linewidth=1.0 if method == "SC-PCP" else 0.72,
                    elinewidth=0.46,
                    capsize=1.0,
                    markeredgewidth=0.22,
                )
            axis.axvspan(-4.25, -3.75, color="#E8B84A", alpha=0.14, zorder=-5)
            axis.grid(axis="y", color="#E1E3E6", linewidth=0.40, zorder=-6)
            if column == 0:
                axis.set_ylabel(ylabel)
        axes[0, column].axhline(
            90.0, color="#20262B", linestyle=(0, (3, 2)), linewidth=0.82
        )
        axes[1, column].axhline(
            90.0, color="#20262B", linestyle=(0, (3, 2)), linewidth=0.82
        )
        axes[0, column].set_title(
            DATASET_LABELS[dataset], fontweight="bold", pad=3
        )
        _panel_label(axes[0, column], column)
        axes[-1, column].set_xticks(
            GAMMAS, [_format_gamma(value) for value in GAMMAS]
        )
        axes[-1, column].set_xlabel("γ")
    figure.legend(
        handles=[_legend_handle(method) for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        columnspacing=0.72,
        handlelength=1.65,
    )
    return figure


def _format_percent_interval_with_gap(
    point: float, lower: float, upper: float, gap_pp: float
) -> str:
    return (
        f"{100.0 * float(point):.2f} [{100.0 * float(lower):.2f}, "
        f"{100.0 * float(upper):.2f}]\nΔ={float(gap_pp):+.2f} pp"
    )


def _format_number_interval(point: float, lower: float, upper: float) -> str:
    return f"{float(point):.3f} [{float(lower):.3f}, {float(upper):.3f}]"


def _format_selection_with_gate(row: object) -> str:
    gate = getattr(row, "n_gate_eligible")
    gate_text = "NA" if pd.isna(gate) else f"{int(gate)}/{int(row.n_prespecified)}"
    return (
        f"{int(row.n_selected)}/{int(row.n_prespecified)} "
        f"({100.0 * float(row.selection_rate):.1f}% "
        f"[{100.0 * float(row.selection_rate_ci95_lower):.1f}, "
        f"{100.0 * float(row.selection_rate_ci95_upper):.1f}])\n"
        f"gate={gate_text}"
    )


def _format_attainment(row: object) -> str:
    if not _as_bool(row.confirmatory) or not _as_bool(row.ranking_permitted):
        return "NA"
    point = "Y" if _as_bool(row.point_attainment_at_target) else "N"
    interval = "Y" if _as_bool(row.wsc_interval_attainment_at_target) else "N"
    eligible = "Y" if _as_bool(row.point_eligible) else "N"
    return f"{point} / {interval} / {eligible}"


def _format_budget(row: object) -> str:
    if row.budget_status != "consumed_complete_science":
        return "not run"
    values = (
        row.calibration_trajectories_per_seed,
        row.grid_trajectories_per_seed,
        row.target_adaptation_trajectories_per_seed,
        row.evaluation_trajectories_per_seed,
    )
    return "/".join("--" if pd.isna(value) else _short_budget(int(value)) for value in values)


def _short_budget(value: int) -> str:
    if value == 0:
        return "0"
    return f"{value // 1000}k" if value % 1000 == 0 else str(value)


def _legend_handle(method: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=legacy.METHOD_COLORS[method],
        linestyle=legacy.METHOD_LINESTYLES[method],
        marker=legacy.METHOD_MARKERS[method],
        markersize=3.0,
        linewidth=1.05 if method == "SC-PCP" else 0.78,
        label=method,
    )


def _metric_limits(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    fallback: tuple[float, float],
    quantum: float,
) -> tuple[float, float]:
    if not len(lower):
        return fallback
    padding = max(quantum, 0.06 * (float(upper.max()) - float(lower.min())))
    return (
        math.floor((float(lower.min()) - padding) / quantum) * quantum,
        math.ceil((float(upper.max()) + padding) / quantum) * quantum,
    )


def _set_width_limits(axis: plt.Axes, rows: pd.DataFrame) -> None:
    values = rows[
        ["normalized_width_ci95_lower", "normalized_width_ci95_upper"]
    ].to_numpy(float)
    lower, upper = float(values.min()), float(values.max())
    padding = max(0.02 * max(abs(lower), abs(upper)), 0.07 * (upper - lower))
    axis.set_ylim(lower - padding, upper + padding)


def _set_stage_axis(axis: plt.Axes, horizon: int) -> None:
    axis.set_xlim(-0.15, horizon - 0.85)
    axis.set_xticks(list(range(horizon)) if horizon <= 6 else list(range(0, horizon, 2)))


def _format_gamma(value: float) -> str:
    number = int(value)
    if number > 0:
        return f"+{number}"
    return str(number).replace("-", "−")


def export_figure(figure: plt.Figure, work_stem: Path, *, title: str) -> None:
    creator = "SC-PCP complete coverage reporting v4 minimal quantitative"
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
    plt.close(figure)


def _output_title(stem: str) -> str:
    return {
        MAIN_STAGE_STEM: "Default gamma=-4 complete stagewise coverage and width",
        MAIN_TABLE_STEM: "Default gamma=-4 complete coverage metrics",
        SIGNED_FIGURE_STEM: "Complete signed-gamma coverage and width supplement",
        PRODUCTION_TABLE_STEM: "Production no-gamma robustness metrics",
    }[stem]


def _write_contract(path: Path, sources: ReportingSources) -> None:
    status = _default_status(sources)
    payload = {
        "schema_version": 3,
        "protocol": RENDER_PROTOCOL,
        "status": "complete",
        "backend": "Python/matplotlib only",
        "scientific_rng_used": False,
        "core_conclusion": (
            "At the prespecified gamma=-4 endpoint, SC-PCP is closest to 0.90 WSC "
            "on all three clinical datasets; its MeanCov Student-t lower bound exceeds "
            "0.90 on all three, but only INSPIRE reaches point WSC and none of the three "
            "SC-PCP WSC intervals attains 0.90. MFCS attains by WSC interval on all three "
            "clinical datasets with greater normalized width. CXR terminated at the "
            "pre-coverage environment-fidelity gate, so it has no coverage metrics."
        ),
        "archetype": "four-column quantitative grid plus complete scalar tables",
        "visible_text_policy": {
            "allowed": [
                "axis labels and ticks",
                "dataset labels",
                "canonical method legend",
                "panel letters",
                "table headers and data cells",
            ],
            "forbidden": [
                "figure titles",
                "subtitles",
                "claim sentences",
                "gate prose",
                "footers and explanatory paragraphs",
            ],
            "unavailable_metric_encoding": "omitted from quantitative plots; NA in tables",
        },
        "default_endpoint": {
            "feedback_parameter": "gamma",
            "feedback_value": PRIMARY_GAMMA,
            "figure_dataset_order": list(CURVE_DATASETS),
            "table_dataset_order": list(DATASETS),
            "setting_status": status.drop(columns="_order").to_dict(orient="records"),
            "native_and_clinical_are_separate_strata": True,
            "cross_dataset_pooling_or_ranking": False,
        },
        "figures": {
            MAIN_STAGE_STEM: {
                "panel_map": {
                    "columns": list(CURVE_DATASETS),
                    "top": "all-six stage C_t minus 0.90 with pointwise Student-t intervals",
                    "bottom": "all-six stage normalized width with pointwise Student-t intervals",
                },
                "shared_coverage_axis": True,
                "dataset_specific_width_axes": True,
                "cxr": "not plotted; six NA rows remain in the scalar table",
            },
            SIGNED_FIGURE_STEM: {
                "rows": ["WSC", "MeanCov", "mean normalized width"],
                "columns": list(CURVE_DATASETS),
                "gammas": list(GAMMAS),
                "confirmatory_gamma": PRIMARY_GAMMA,
                "nonprimary_gamma_role": "descriptive_no_ranking",
                "cxr": "not plotted; no signed-gamma science",
            },
        },
        "tables": {
            MAIN_TABLE_STEM: (
                "gamma=-4 WSC/CI/gap, worst stage, MeanCov/CI/gap, mean width/CI, "
                "selection/Wilson/gate count, separate point/CI/eligibility, and budget"
            ),
            PRODUCTION_TABLE_STEM: (
                "production/native no-gamma robustness-only scalar metrics"
            ),
        },
        "metric_contract": {
            "coverage_target": TARGET,
            "primary_metric": PRIMARY_METRIC,
            "stagewise_interval": (
                "pointwise two-sided Student-t across method-selected complete seeds; "
                "not simultaneous"
            ),
            "wsc_interval": "10000-draw complete-seed-vector percentile bootstrap",
            "mean_coverage_interval": "Student-t across selected seed-level means",
            "mean_width_interval": "Student-t across selected seed-level means",
            "selection_interval": "Wilson over all 20 prespecified seeds",
            "selection_denominator": 20,
            "point_eligibility_rule": POINT_ELIGIBILITY_RULE,
            "point_and_interval_attainment_separate": True,
            "native_width_definition": NATIVE_WIDTH_DEFINITION,
            "clinical_width_definition": CLINICAL_WIDTH_DEFINITION,
            "width_comparison_scope": "within dataset only",
        },
        "information_budget_per_seed": {
            "calibration": CALIBRATION_BUDGET,
            "grid": GRID_BUDGET,
            "evaluation": EVALUATION_BUDGET,
            "target_adaptation": dict(TARGET_ADAPTATION_BUDGET),
            "cxr": "not run: terminal pre-coverage NO-GO",
        },
        "source_data": {
            "files": sorted(SOURCE_FILES),
            "status_rows": len(sources.status),
            "stage_rows": len(sources.stage),
            "scalar_rows": len(sources.scalar),
            "paired_rows": len(sources.paired),
        },
        "input_contracts": sources.input_contracts,
        "adapter_rule": (
            "Only pinned protocol-specific adapters are accepted. The CXR adapter binds "
            "the completed v5 confirmation and terminal v6 development roots; unknown, "
            "partial or hash-mismatched roots fail closed."
        ),
        "reviewer_risks": [
            "CXR absence is a terminal pre-coverage environment-fidelity NO-GO, not poor method coverage performance.",
            "Production/no-gamma results are robustness-only and cannot substitute for signed gamma=-4 cells.",
            "Native Synthetic and clinical emulator panels are separate strata and are not pooled.",
            "Pointwise intervals are not simultaneous bands or finite-sample certificates.",
            "Point eligibility and WSC interval attainment are different fields.",
            "Efficiency ranking is within dataset and only among point-eligible methods.",
            "The clinical controlled emulator is calibration-aligned, not a causal treatment effect.",
            "No universal dominance or universal-SOTA claim is made.",
        ],
        "export_contract": {
            "work_formats": [
                "editable SVG",
                "TrueType PDF",
                "600-dpi TIFF",
                "240-dpi PNG",
            ],
            "paper_files": sorted(PAPER_FILES),
            "paper_directory_policy": "PDF only",
        },
    }
    _write_json(path, payload)


def _write_qa(path: Path, sources: ReportingSources) -> None:
    lines = [
        "# Signed-gamma complete coverage reporting QA",
        "",
        "- Backend exclusivity: Python/matplotlib produced all plots, previews and exports.",
        "- Scientific RNG: none; every interval was copied from pinned COMPLETE artifacts.",
        f"- Source rows: status={len(sources.status)}, stage={len(sources.stage)}, scalar={len(sources.scalar)}, paired={len(sources.paired)}.",
        "- Default figure identity: Native Synthetic plus MIMIC-IV, eICU and INSPIRE at gamma=-4; CXR is omitted from quantitative plots and retained as NA table rows.",
        "- Every curve panel contains all six canonical methods and every zero-based stage; no horizon padding.",
        "- The nominal 0.90 line is explicit; the top row reports C_t minus 0.90 in percentage points.",
        "- Stage coverage and width bars are pointwise Student-t intervals, not simultaneous bands.",
        "- WSC is min_t mean_seed(C_seed,t); worst stage is the first zero-based argmin.",
        "- WSC and MeanCov gaps from 0.90 are explicit table and source fields.",
        "- WSC intervals are stored 10,000-draw complete-seed-vector percentile intervals.",
        "- MeanCov/width use Student-t intervals; selection uses Wilson intervals over all 20 prespecified seeds.",
        "- eICU gate eligibility is 19/20 while its selection denominator remains 20.",
        "- Point attainment, WSC-CI attainment and point eligibility are separate fields and table entries.",
        "- Width efficiency is ranked only within dataset and among point-eligible methods.",
        "- Native and clinical normalized-width definitions are separately preserved; no cross-dataset width ranking is defined.",
        "- Audited claim: SC-PCP is closest to 0.90 WSC in each of the three clinical datasets.",
        "- Audited claim: all three SC-PCP MeanCov Student-t lower bounds exceed 0.90.",
        "- Audited claim: only INSPIRE reaches SC-PCP point WSC; no clinical SC-PCP WSC CI attains 0.90.",
        "- Audited claim: MFCS WSC CIs attain 0.90 on all three clinical datasets and MFCS is wider than SC-PCP.",
        "- CXR terminal lineage: v5 confirmation completed NO-GO at 18/20 k0 with 20/20 structural and support checks.",
        "- CXR terminal lineage: v6 development reached 19/20 and 18/20 numeric passes against a required 20/20; both structural checks were 20/20.",
        "- CXR terminal boundary: coverage_generated=false, theta=null, terminal_no_v7=true; the fresh 120k bank remained unconsumed and no v6 confirmation root exists.",
        "- CXR contributes zero stage rows and six unavailable scalar rows; planned budgets are not reported as consumed.",
        "- Production/no-gamma appears only in a robustness table and never in the default gamma=-4 figure/table.",
        "- Typography: Times New Roman with serif fallbacks; SVG text is editable and PDF uses TrueType fonts.",
        "- Visible-text policy: plots contain only axes/ticks, dataset labels, panel letters and the canonical method legend; tables contain only headers and data cells.",
        "- Accessibility: method identity uses color, marker and line-style redundancy; unavailable CXR metrics are not plotted.",
        "- Paper output contains PDFs only; the work bundle contains source CSVs, vector/raster exports, contract, QA and hashes.",
        "- Claim boundary: asymptotic per-step marginal coverage only; no finite-sample, PAC, distribution-free, data-conditional, causal or universal-SOTA claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_written_sources(work_root: Path) -> None:
    sources = ReportingSources(
        status=pd.read_csv(
            work_root / "setting_status.csv", float_precision="round_trip"
        ),
        stage=pd.read_csv(
            work_root / "coverage_stage_profiles.csv", float_precision="round_trip"
        ),
        scalar=pd.read_csv(
            work_root / "coverage_scalar_summary.csv", float_precision="round_trip"
        ),
        paired=pd.read_csv(
            work_root / "paired_scpcp_contrasts.csv", float_precision="round_trip"
        ),
        input_contracts={},
    )
    validate_reporting_sources(sources)
    validate_claim_contract(sources.scalar)


def _write_render_manifest(path: Path, *, work_root: Path, paper_root: Path) -> None:
    observed_paper = {item.name for item in paper_root.iterdir() if item.is_file()}
    if observed_paper != PAPER_FILES:
        raise RuntimeError("paper staging files differ before manifest commit")
    work_files = {
        item.name: _file_contract(item)
        for item in sorted(work_root.iterdir())
        if item.is_file() and item.name not in {path.name, "COMPLETE"}
    }
    paper_files = {
        name: _file_contract(paper_root / name) for name in sorted(PAPER_FILES)
    }
    _write_json(
        path,
        {
            "schema_version": 2,
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
        raise RuntimeError("v2 work bundle entry set differs")
    if observed_paper != PAPER_FILES:
        raise RuntimeError("v2 paper bundle must contain exactly the declared PDFs")
    _validate_written_sources(work_root)
    manifest = _read_json(work_root / "render_manifest.json")
    if (
        manifest.get("protocol") != RENDER_PROTOCOL
        or manifest.get("status") != "complete"
        or set(_mapping(manifest.get("work_files"), "manifest work files"))
        != WORK_FILES - {"render_manifest.json", "COMPLETE"}
        or set(_mapping(manifest.get("paper_files"), "manifest paper files"))
        != PAPER_FILES
    ):
        raise RuntimeError("v2 render manifest contract differs")
    expected_complete = (
        f"protocol={RENDER_PROTOCOL}\n"
        f"manifest_sha256={_file_sha256(work_root / 'render_manifest.json')}\n"
    )
    if (work_root / "COMPLETE").read_text(encoding="utf-8") != expected_complete:
        raise RuntimeError("v2 work COMPLETE marker differs")
    for stem in OUTPUT_STEMS:
        svg = (work_root / f"{stem}.svg").read_text(encoding="utf-8")
        if "<text" not in svg or "Times New Roman" not in svg:
            raise RuntimeError(f"{stem} editable-font SVG contract differs")
        if not (work_root / f"{stem}.pdf").read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"{stem} work PDF is malformed")
        if not (paper_root / f"{stem}.pdf").read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"{stem} paper PDF is malformed")
        if _file_sha256(work_root / f"{stem}.pdf") != _file_sha256(
            paper_root / f"{stem}.pdf"
        ):
            raise RuntimeError(f"{stem} paper PDF differs from work PDF")
        if not (work_root / f"{stem}.png").read_bytes().startswith(b"\x89PNG"):
            raise RuntimeError(f"{stem} PNG is malformed")
        if (work_root / f"{stem}.tiff").read_bytes()[:4] not in {
            b"II*\x00",
            b"MM\x00*",
        }:
            raise RuntimeError(f"{stem} TIFF is malformed")
    for group, root in (("work_files", work_root), ("paper_files", paper_root)):
        contracts = _mapping(manifest[group], group)
        for name, contract in contracts.items():
            _validate_file_contract(root / name, _mapping(contract, name))


def _file_contract(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _validate_file_contract(path: Path, contract: Mapping[str, Any]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(contract["bytes"])
        or _file_sha256(path) != contract["sha256"]
    ):
        raise RuntimeError(f"rendered file contract differs: {path.name}")


if __name__ == "__main__":
    main()
