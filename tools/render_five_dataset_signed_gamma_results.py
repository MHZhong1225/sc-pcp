"""Render the complete five-dataset signed-gamma paper result bundle.

The command performs deterministic post-processing only.  It reuses the
canonical reporting-v2 adapters for Native Synthetic, clinical-v4, and the
production robustness root, and adds the prospectively frozen MIMIC-CXR
environment-support science adapter.

Example
-------
conda run -n ucp python tools/render_five_dataset_signed_gamma_results.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    run_controlled_clinical_mimic_cxr_environment_support_science as cxr_science,
)
from tools.reporting import render_complete_coverage_reporting_v2 as report_v2  # noqa: E402


RENDER_PROTOCOL = "five_dataset_signed_gamma_results_v1"
DEFAULT_NATIVE_INPUT = report_v2.DEFAULT_NATIVE_INPUT
DEFAULT_CLINICAL_INPUT = report_v2.DEFAULT_CLINICAL_INPUT
DEFAULT_CXR_INPUT = cxr_science.OUTPUT_ROOT
DEFAULT_PRODUCTION_INPUT = report_v2.DEFAULT_PRODUCTION_INPUT
DEFAULT_WORK_OUTPUT = ROOT / "results/work/five_dataset_signed_gamma_results_v1_20260903"
DEFAULT_PAPER_OUTPUT = ROOT / "results/paper_five_dataset_signed_gamma_results_v1_20260903"

TARGET = report_v2.TARGET
PRIMARY_GAMMA = report_v2.PRIMARY_GAMMA
GAMMAS = report_v2.GAMMAS
METHODS = report_v2.METHODS
DATASETS = ("synthetic", "mimic_iv", "eicu", "inspire", "mimic_cxr")
SIGNED_FAMILIES = (
    "native_signed_gamma",
    "clinical_v4_signed_gamma",
    "clinical_cxr_environment_support_signed_gamma",
)
PRODUCTION_FAMILY = "production_no_gamma_robustness"
PRODUCTION_DATASETS = report_v2.PRODUCTION_DATASETS
HORIZONS = report_v2.HORIZONS
DATASET_LABELS = {
    **report_v2.DATASET_LABELS,
    "mimic_cxr": "MIMIC-CXR",
}
PRIMARY_METRIC = report_v2.PRIMARY_METRIC
CLINICAL_WIDTH_DEFINITION = report_v2.CLINICAL_WIDTH_DEFINITION
CALIBRATION_BUDGET = report_v2.CALIBRATION_BUDGET
GRID_BUDGET = report_v2.GRID_BUDGET
EVALUATION_BUDGET = report_v2.EVALUATION_BUDGET
TARGET_ADAPTATION_BUDGET = report_v2.TARGET_ADAPTATION_BUDGET

STATUS_COLUMNS = report_v2.STATUS_COLUMNS
STAGE_COLUMNS = report_v2.STAGE_COLUMNS
SCALAR_COLUMNS = report_v2.SCALAR_COLUMNS
PAIRED_COLUMNS = report_v2.PAIRED_COLUMNS
ReportingSources = report_v2.ReportingSources

HERO_STEM = "figure_gamma_minus4_wsc_meancov"
STAGE_STEM = "figure_gamma_minus4_stage_coverage_width"
SIGNED_STEM = "figure_signed_gamma_wsc_meancov_width"
TABLE_STEM = "table_gamma_minus4_complete_metrics"
FIGURE_STEMS = (HERO_STEM, STAGE_STEM, SIGNED_STEM)
TABLE_STEMS = (TABLE_STEM,)
OUTPUT_STEMS = (*FIGURE_STEMS, *TABLE_STEMS)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-input", type=Path, default=DEFAULT_NATIVE_INPUT)
    parser.add_argument("--clinical-input", type=Path, default=DEFAULT_CLINICAL_INPUT)
    parser.add_argument("--cxr-input", type=Path, default=DEFAULT_CXR_INPUT)
    parser.add_argument("--production-input", type=Path, default=DEFAULT_PRODUCTION_INPUT)
    parser.add_argument("--work-output", type=Path, default=DEFAULT_WORK_OUTPUT)
    parser.add_argument("--paper-output", type=Path, default=DEFAULT_PAPER_OUTPUT)
    args = parser.parse_args()
    render_report(
        native_input=args.native_input.resolve(),
        clinical_input=args.clinical_input.resolve(),
        cxr_input=args.cxr_input.resolve(),
        production_input=args.production_input.resolve(),
        work_output=args.work_output.resolve(),
        paper_output=args.paper_output.resolve(),
    )
    print(args.paper_output.resolve())


def build_reporting_sources(
    *,
    native_input: Path,
    clinical_input: Path,
    cxr_input: Path,
    production_input: Path,
) -> ReportingSources:
    native = report_v2.load_native_sources(native_input)
    clinical = report_v2.load_clinical_v4_sources(clinical_input)
    cxr = load_cxr_environment_support_sources(cxr_input)
    production = report_v2.load_production_sources(production_input)
    sources = ReportingSources(
        status=pd.concat(
            [native.status, clinical.status, cxr.status, production.status],
            ignore_index=True,
        ).loc[:, STATUS_COLUMNS],
        stage=pd.concat(
            [native.stage, clinical.stage, cxr.stage, production.stage],
            ignore_index=True,
        ).loc[:, STAGE_COLUMNS],
        scalar=report_v2.assign_efficiency_ranking(
            pd.concat(
                [native.scalar, clinical.scalar, cxr.scalar, production.scalar],
                ignore_index=True,
            ).loc[:, SCALAR_COLUMNS]
        ),
        paired=pd.concat(
            [native.paired, clinical.paired, cxr.paired], ignore_index=True
        ).loc[:, PAIRED_COLUMNS],
        input_contracts={
            "native_exact_replay": native.input_contracts,
            "clinical_v4_publish": clinical.input_contracts,
            "mimic_cxr_environment_support": cxr.input_contracts,
            "production_robustness": production.input_contracts,
        },
    )
    validate_reporting_sources(sources)
    return sources


def load_cxr_environment_support_sources(root: Path) -> ReportingSources:
    root = root.resolve()
    gates, metadata, final, science_final, summary, coverage_audit = (
        _validate_cxr_science_bundle(root)
    )
    summary_path = root / cxr_science.SCIENCE_PHASE / "summary.json"
    source_path = report_v2._project_path(summary_path)
    source_sha = report_v2._file_sha256(summary_path)
    aggregates = report_v2._gamma_aggregates(summary)
    science_eligible = tuple(int(seed) for seed in science_final["science_eligible_seeds"])
    eligible_count = len(science_eligible)
    status_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for gamma in GAMMAS:
        aggregate = aggregates[gamma]
        confirmatory = gamma == PRIMARY_GAMMA
        role = str(aggregate["analysis_role"])
        setting_id = f"mimic_cxr_gamma_{report_v2._gamma_id(gamma)}"
        status_rows.append(
            report_v2._status_row(
                reporting_family=SIGNED_FAMILIES[-1],
                setting_id=setting_id,
                display_label=DATASET_LABELS["mimic_cxr"],
                dataset="mimic_cxr",
                source_protocol=cxr_science.PROTOCOL,
                setting_type="post_failure_environment_support_controlled_clinical",
                feedback_parameter="gamma",
                feedback_value=gamma,
                horizon=HORIZONS["mimic_cxr"],
                panel_status="CURVES",
                interpretation_status=role,
                confirmatory=confirmatory,
                ranking_permitted=confirmatory,
                scientific_rows_available=True,
                n_prespecified=cxr_science.PRESPECIFIED_SEED_COUNT,
                n_gate_eligible=eligible_count,
                gate_reason="",
                budget_status="consumed_complete_science",
                frozen_setting_sha256=str(metadata["config_sha256"]),
                source_path=source_path,
                source_sha256=source_sha,
            )
        )
        methods = report_v2._mapping(
            aggregate["methods"], f"MIMIC-CXR gamma={gamma} methods"
        )
        for method in METHODS:
            cell = report_v2._mapping(
                methods[method], f"MIMIC-CXR gamma={gamma}/{method}"
            )
            _validate_cxr_method_budget(method, cell, gates.science_contract)
            if (
                cell.get("n_k0_eligible") != len(gates.support_k0_eligible_seeds)
                or cell.get("n_support_k0_eligible")
                != len(gates.support_k0_eligible_seeds)
                or cell.get("n_support_k0_overlap_eligible") != eligible_count
            ):
                raise RuntimeError("MIMIC-CXR method eligibility cohorts differ")
            common = report_v2._method_context(
                reporting_family=SIGNED_FAMILIES[-1],
                setting_id=setting_id,
                display_label=DATASET_LABELS["mimic_cxr"],
                dataset="mimic_cxr",
                source_protocol=cxr_science.PROTOCOL,
                setting_type="post_failure_environment_support_controlled_clinical",
                feedback_value=gamma,
                analysis_role=role,
                panel_status="CURVES",
                confirmatory=confirmatory,
                ranking_permitted=confirmatory,
                method=method,
                n_prespecified=int(cell["n_prespecified"]),
                n_gate_eligible=int(cell["n_support_k0_overlap_eligible"]),
                n_selected=int(cell["n_selected"]),
                coverage_conditioning=str(summary["coverage_conditioning"]),
                normalized_width_definition=CLINICAL_WIDTH_DEFINITION,
                source_path=source_path,
                source_sha256=source_sha,
            )
            stage_rows.extend(
                report_v2._stage_rows_from_cell(
                    common,
                    cell,
                    coverage_ci_field="target_coverage_by_stage_ci95",
                    width_ci_field="target_normalized_width_by_stage_ci95",
                )
            )
            scalar_rows.append(
                report_v2._scalar_from_cell(
                    common, cell, stored_point_eligibility=False
                )
            )
        paired_rows.extend(
            report_v2._paired_rows(
                aggregate,
                reporting_family=SIGNED_FAMILIES[-1],
                setting_id=setting_id,
                dataset="mimic_cxr",
                feedback_value=gamma,
                confirmatory=confirmatory,
                source_path=source_path,
                source_sha256=source_sha,
            )
        )

    return report_v2._sources_from_rows(
        status_rows,
        stage_rows,
        scalar_rows,
        paired_rows,
        input_contracts={
            "protocol": cxr_science.PROTOCOL,
            "input_root": report_v2._project_path(root),
            "manifest_sha256": report_v2._file_sha256(root / "manifest.json"),
            "complete_sha256": report_v2._file_sha256(root / "COMPLETE"),
            "final_status_sha256": report_v2._file_sha256(root / "FINAL_STATUS.json"),
            "science_summary_sha256": source_sha,
            "science_coverage_audit_sha256": report_v2._file_sha256(
                root / cxr_science.SCIENCE_PHASE / "coverage_audit.json"
            ),
            "artifact_count": int(
                report_v2._read_json(root / "manifest.json")["artifact_count"]
            ),
            "confirmation_binding": gates.confirmation_binding,
            "confirmation_binding_sha256": cxr_science._json_sha256(
                gates.confirmation_binding
            ),
            "gate_contract_sha256": cxr_science._json_sha256(gates.contract),
            "prespecified_seed_count": cxr_science.PRESPECIFIED_SEED_COUNT,
            "support_k0_eligible_seed_count": len(
                gates.support_k0_eligible_seeds
            ),
            "science_eligible_seed_count": eligible_count,
            "selection_rate_denominator": cxr_science.PRESPECIFIED_SEED_COUNT,
            "final_status": final["status"],
            "science_final_status": science_final["status"],
            "coverage_audit_status": coverage_audit["status"],
        },
    )


def _validate_cxr_science_bundle(
    root: Path,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"CXR science root must be a real COMPLETE directory: {root}")
    devices = ("cuda:0", "cuda:1")
    gates = cxr_science.verify_gate_bundle(devices=devices)
    observed_confirmation = cxr_science._confirmation_binding(
        cxr_science.CONFIRMATION_ROOT.resolve()
    )
    if observed_confirmation != gates.confirmation_binding:
        raise RuntimeError("CXR science confirmation binding differs")

    metadata = cxr_science._read_json(root / "metadata.json")
    audit_hash = cxr_science._json_sha256(gates.contract)
    expected_metadata = cxr_science._science_metadata(
        gates,
        devices=devices,
        audit_go_sha256=audit_hash,
        source_snapshot=report_v2._mapping(
            metadata.get("source_snapshot"), "CXR science source snapshot"
        ),
    )
    if metadata != expected_metadata:
        raise RuntimeError("CXR science metadata/confirmation binding differs")
    cxr_science._verify_manifest(root)
    cxr_science._validate_complete_root(root, expected_metadata, gates)

    final = cxr_science._read_json(root / "FINAL_STATUS.json")
    science_root = root / cxr_science.SCIENCE_PHASE
    science_final = cxr_science._read_json(science_root / "FINAL_STATUS.json")
    summary = cxr_science._read_json(science_root / "summary.json")
    audit = cxr_science._read_json(science_root / "coverage_audit.json")
    unlock = cxr_science._read_json(root / "SCIENCE_UNLOCK.json")
    support_eligible = tuple(int(seed) for seed in gates.support_k0_eligible_seeds)
    overlap = cxr_science._read_json(
        root / cxr_science.OVERLAP_PHASE / "summary.json"
    )
    science_eligible = tuple(int(seed) for seed in unlock.get("science_eligible_seeds", ()))
    science_set = set(science_eligible)
    ordered_subset = tuple(seed for seed in support_eligible if seed in science_set)
    if (
        len(science_eligible) < cxr_science.MINIMUM_JOINT_PASS_COUNT
        or len(science_set) != len(science_eligible)
        or science_eligible != ordered_subset
        or overlap.get("status") != "OVERLAP_GO"
        or overlap.get("science_may_start") is not True
        or tuple(int(seed) for seed in overlap.get("support_k0_eligible_seeds", ()))
        != support_eligible
        or overlap.get("support_k0_eligible_seed_count") != len(support_eligible)
        or tuple(int(seed) for seed in overlap.get("passed_seeds", ()))
        != science_eligible
        or overlap.get("joint_overlap_pass_count") != len(science_eligible)
        or tuple(int(seed) for seed in overlap.get("failed_seeds", ()))
        != tuple(seed for seed in support_eligible if seed not in science_set)
        or unlock != cxr_science._science_unlock(gates, overlap, science_eligible)
    ):
        raise RuntimeError("CXR science unlock/overlap seed binding differs")
    expected_science_final = {
        "protocol": cxr_science.PROTOCOL,
        "dataset": "mimic_cxr",
        "status": "SCIENCE_COMPLETE",
        "methods": list(METHODS),
        "gammas": list(GAMMAS),
        "primary_gamma": PRIMARY_GAMMA,
        "primary_metric": cxr_science.PRIMARY_METRIC,
        "prespecified_seed_count": 20,
        "science_eligible_seed_count": len(science_eligible),
        "science_eligible_seeds": list(science_eligible),
        "seed_deletions": 0,
    }
    expected_final = {
        **expected_science_final,
        "confirmation_status": "CONFIRMATION_GO",
        "overlap_status": "OVERLAP_GO",
        "science_unlocked": True,
        "coverage_generated": True,
        "science_unlock_sha256": cxr_science._json_sha256(unlock),
    }
    if science_final != expected_science_final or final != expected_final:
        raise RuntimeError("CXR SCIENCE_COMPLETE status differs")
    _validate_cxr_summary(summary, audit, gates, science_eligible)
    return gates, metadata, final, science_final, summary, audit


def _validate_cxr_summary(
    summary: Mapping[str, Any],
    audit: Mapping[str, Any],
    gates: Any,
    science_eligible: tuple[int, ...],
) -> None:
    support_eligible = tuple(int(seed) for seed in gates.support_k0_eligible_seeds)
    compatibility = {
        "seeds_k0_eligible": (
            "alias of seeds_support_k0_eligible before donor-overlap screening"
        ),
        "aggregates[].n_k0_eligible_seeds": (
            "count of support/K0-eligible seeds before donor-overlap screening"
        ),
        "aggregates[].methods[].n_k0_eligible": (
            "count of support/K0-eligible seeds before donor-overlap screening"
        ),
    }
    conditioning = "successful method selection among support/K0/overlap-eligible seeds"
    bootstrap = report_v2._mapping(summary.get("bootstrap"), "CXR bootstrap")
    if (
        summary.get("protocol") != cxr_science.PROTOCOL
        or summary.get("dataset") != "mimic_cxr"
        or summary.get("role") != "post_failure_cxr_environment_support_science"
        or summary.get("interpretation_status") != "EMPIRICAL_OVERLAP_SCREEN_PASSED"
        or tuple(summary.get("methods", ())) != METHODS
        or summary.get("primary_gamma") != PRIMARY_GAMMA
        or summary.get("primary_metric") != cxr_science.PRIMARY_METRIC
        or summary.get("mean_coverage_is_supplementary") is not True
        or summary.get("selection_rate_denominator")
        != "all 20 prespecified confirmation seeds"
        or tuple(summary.get("seeds_prespecified", ()))
        != tuple(cxr_science.CONFIRMATION_SEEDS)
        or tuple(summary.get("seeds_support_k0_eligible", ())) != support_eligible
        or tuple(summary.get("seeds_k0_eligible", ())) != support_eligible
        or tuple(summary.get("seeds_support_k0_overlap_eligible", ()))
        != science_eligible
        or summary.get("compatibility_field_semantics") != compatibility
        or summary.get("coverage_conditioning") != conditioning
        or any(seed not in support_eligible for seed in science_eligible)
        or summary.get("seed_deletions") != 0
        or bootstrap.get("resamples") != 10_000
        or bootstrap.get("prespecified_seed_count") != 20
        or bootstrap.get("root_seed") != gates.preset.bootstrap_seed
        or bootstrap.get("uniform_matrix_shape") != [10_000, 20]
        or bootstrap.get("complete_seed_index_matrix_shape") != [10_000, 20]
    ):
        raise RuntimeError("CXR signed-gamma summary contract differs")
    aggregates = report_v2._gamma_aggregates(summary)
    for gamma, aggregate in aggregates.items():
        expected_role = (
            "confirmatory_gamma_minus_4_endpoint"
            if gamma == PRIMARY_GAMMA
            else "descriptive_signed_control_curve"
        )
        if (
            aggregate.get("analysis_role") != expected_role
            or aggregate.get("n_prespecified_seeds") != 20
            or aggregate.get("n_k0_eligible_seeds") != len(support_eligible)
            or aggregate.get("n_support_k0_eligible_seeds")
            != len(support_eligible)
            or aggregate.get("n_support_k0_overlap_eligible_seeds")
            != len(science_eligible)
            or set(report_v2._mapping(aggregate.get("methods"), "CXR methods"))
            != set(METHODS)
        ):
            raise RuntimeError("CXR aggregate grid or role differs")
        for method, cell in report_v2._mapping(
            aggregate.get("methods"), "CXR methods"
        ).items():
            method_summary = report_v2._mapping(cell, f"CXR {method} summary")
            if (
                method_summary.get("n_k0_eligible") != len(support_eligible)
                or method_summary.get("n_support_k0_eligible")
                != len(support_eligible)
                or method_summary.get("n_support_k0_overlap_eligible")
                != len(science_eligible)
            ):
                raise RuntimeError("CXR method eligibility cohorts differ")
    _validate_cxr_coverage_audit(
        audit, aggregates, support_eligible, science_eligible, conditioning
    )


def _validate_cxr_coverage_audit(
    audit: Mapping[str, Any],
    aggregates: Mapping[float, Mapping[str, Any]],
    support_eligible: tuple[int, ...],
    science_eligible: tuple[int, ...],
    conditioning: str,
) -> None:
    records = audit.get("records")
    if (
        audit.get("protocol") != cxr_science.PROTOCOL
        or audit.get("dataset") != "mimic_cxr"
        or audit.get("status") != "COVERAGE_AUDIT_COMPLETE"
        or audit.get("primary_metric") != cxr_science.PRIMARY_METRIC
        or audit.get("formula_verified") is not True
        or audit.get("mean_coverage_is_supplementary") is not True
        or audit.get("all_six_methods_present") is not True
        or audit.get("all_five_gammas_present") is not True
        or audit.get("coverage_conditioning") != conditioning
        or tuple(audit.get("seeds_support_k0_eligible", ())) != support_eligible
        or tuple(audit.get("seeds_support_k0_overlap_eligible", ()))
        != science_eligible
        or audit.get("support_k0_eligible_seed_count") != len(support_eligible)
        or audit.get("support_k0_overlap_eligible_seed_count")
        != len(science_eligible)
        or tuple(audit.get("science_eligible_seeds", ())) != science_eligible
        or audit.get("selection_rate_denominator") != 20
        or not isinstance(records, list)
        or len(records) != len(GAMMAS) * len(METHODS)
    ):
        raise RuntimeError("CXR coverage audit contract differs")
    indexed = {(float(row["gamma"]), str(row["method"])): row for row in records}
    if set(indexed) != {(gamma, method) for gamma in GAMMAS for method in METHODS}:
        raise RuntimeError("CXR coverage audit grid differs")
    for (gamma, method), record in indexed.items():
        cell = report_v2._mapping(aggregates[gamma]["methods"][method], "CXR cell")
        if (
            record.get("n_support_k0_eligible") != len(support_eligible)
            or record.get("n_support_k0_overlap_eligible")
            != len(science_eligible)
            or record.get("n_selected") != cell.get("n_selected")
        ):
            raise RuntimeError("CXR coverage audit selection count differs")
        metrics = report_v2._mapping(record.get("metrics"), "CXR coverage metrics")
        if (
            metrics.get("stage_coverage") != cell.get("target_coverage_by_stage")
            or metrics.get("WSC") != cell.get("target_marginal_worst_coverage")
            or metrics.get("MeanCov") != cell.get("target_mean_coverage")
        ):
            raise RuntimeError("CXR coverage audit metrics differ from summary")


def _validate_cxr_method_budget(
    method: str, cell: Mapping[str, Any], science_contract: Mapping[str, Any]
) -> None:
    adaptation = report_v2._mapping(
        science_contract.get("target_adaptation_trajectories"),
        "CXR target-adaptation budget",
    )
    if (
        science_contract.get("calibration_trajectories") != CALIBRATION_BUDGET
        or science_contract.get("grid_trajectories") != GRID_BUDGET
        or science_contract.get("evaluation_trajectories") != EVALUATION_BUDGET
        or adaptation.get(method) != TARGET_ADAPTATION_BUDGET[method]
        or cell.get("target_adaptation_trajectories_per_seed")
        != TARGET_ADAPTATION_BUDGET[method]
    ):
        raise RuntimeError(f"MIMIC-CXR {method} information budget differs")


def validate_reporting_sources(sources: ReportingSources) -> None:
    status, stage, scalar, paired = (
        sources.status,
        sources.stage,
        sources.scalar,
        sources.paired,
    )
    expected_columns = (
        (status, STATUS_COLUMNS, "status"),
        (stage, STAGE_COLUMNS, "stage"),
        (scalar, SCALAR_COLUMNS, "scalar"),
        (paired, PAIRED_COLUMNS, "paired"),
    )
    for frame, columns, label in expected_columns:
        if tuple(frame.columns) != columns:
            raise RuntimeError(f"five-dataset {label} schema differs")
    if status.duplicated(["reporting_family", "setting_id"]).any():
        raise RuntimeError("five-dataset status keys are duplicated")
    if stage.duplicated(
        ["reporting_family", "setting_id", "method", "stage_zero_based"]
    ).any():
        raise RuntimeError("five-dataset stage keys are duplicated")
    if scalar.duplicated(["reporting_family", "setting_id", "method"]).any():
        raise RuntimeError("five-dataset scalar keys are duplicated")
    if paired.duplicated(["reporting_family", "setting_id", "baseline"]).any():
        raise RuntimeError("five-dataset paired keys are duplicated")

    signed_status = status[status["reporting_family"].isin(SIGNED_FAMILIES)]
    signed_stage = stage[stage["reporting_family"].isin(SIGNED_FAMILIES)]
    signed_scalar = scalar[scalar["reporting_family"].isin(SIGNED_FAMILIES)]
    production_status = status[status["reporting_family"].eq(PRODUCTION_FAMILY)]
    production_stage = stage[stage["reporting_family"].eq(PRODUCTION_FAMILY)]
    production_scalar = scalar[scalar["reporting_family"].eq(PRODUCTION_FAMILY)]

    expected_settings = {(dataset, gamma) for dataset in DATASETS for gamma in GAMMAS}
    if set(zip(signed_status["dataset"], signed_status["feedback_value"])) != expected_settings:
        raise RuntimeError("five-dataset signed-gamma status grid differs")
    primary = _primary_status(sources)
    if tuple(primary["dataset"]) != DATASETS:
        raise RuntimeError("gamma=-4 five-dataset order differs")
    if (
        not primary["confirmatory"].map(report_v2._as_bool).all()
        or not primary["ranking_permitted"].map(report_v2._as_bool).all()
        or not primary["scientific_rows_available"].map(report_v2._as_bool).all()
        or set(primary["panel_status"]) != {"CURVES"}
    ):
        raise RuntimeError("gamma=-4 confirmatory curve identity differs")
    nonprimary = signed_status[~signed_status["feedback_value"].eq(PRIMARY_GAMMA)]
    if nonprimary["confirmatory"].map(report_v2._as_bool).any() or nonprimary[
        "ranking_permitted"
    ].map(report_v2._as_bool).any():
        raise RuntimeError("descriptive signed-gamma cells permit ranking")

    if (
        tuple(production_status["dataset"]) != PRODUCTION_DATASETS
        or len(production_stage) != sum(HORIZONS[name] for name in PRODUCTION_DATASETS) * len(METHODS)
        or len(production_scalar) != len(PRODUCTION_DATASETS) * len(METHODS)
        or production_status["confirmatory"].map(report_v2._as_bool).any()
        or production_status["ranking_permitted"].map(report_v2._as_bool).any()
    ):
        raise RuntimeError("production robustness input contract differs")

    expected_scalar_keys = {
        (family, _setting_id(dataset, gamma), method)
        for dataset, family in _dataset_families().items()
        for gamma in GAMMAS
        for method in METHODS
    }
    observed_scalar_keys = set(
        zip(
            signed_scalar["reporting_family"],
            signed_scalar["setting_id"],
            signed_scalar["method"],
        )
    )
    if observed_scalar_keys != expected_scalar_keys:
        raise RuntimeError("five-dataset scalar method grid differs")
    if len(signed_stage) != sum(HORIZONS[name] for name in DATASETS) * len(GAMMAS) * len(METHODS):
        raise RuntimeError("five-dataset stage grid is incomplete")
    if not signed_scalar["metric_available"].map(report_v2._as_bool).all():
        raise RuntimeError("five-dataset signed-gamma metrics are incomplete")
    if set(signed_stage["method"]) != set(METHODS) or set(signed_scalar["method"]) != set(METHODS):
        raise RuntimeError("canonical six-method set differs")
    if set(signed_scalar["primary_metric"]) != {PRIMARY_METRIC}:
        raise RuntimeError("WSC primary metric definition differs")
    if set(signed_scalar["n_prespecified"].astype(int)) != {20} or set(
        signed_stage["n_prespecified"].astype(int)
    ) != {20}:
        raise RuntimeError("selection denominator is not the 20 prespecified seeds")

    _validate_numeric_rows(signed_stage, signed_scalar)
    _validate_signed_budgets(signed_stage, signed_scalar)
    _validate_paired_rows(paired)

    expected_all_scalar = {
        (row.reporting_family, row.setting_id, method)
        for row in status.itertuples(index=False)
        for method in METHODS
    }
    if set(zip(scalar["reporting_family"], scalar["setting_id"], scalar["method"])) != expected_all_scalar:
        raise RuntimeError("status/scalar setting grid differs")


def _validate_numeric_rows(stage: pd.DataFrame, scalar: pd.DataFrame) -> None:
    coverage = stage[
        ["coverage_ci95_lower", "coverage_mean", "coverage_ci95_upper"]
    ].to_numpy(float)
    width = stage[
        [
            "normalized_width_ci95_lower",
            "normalized_width_mean",
            "normalized_width_ci95_upper",
        ]
    ].to_numpy(float)
    if (
        not np.isfinite(coverage).all()
        or np.any((coverage < 0.0) | (coverage > 1.0))
        or np.any(coverage[:, 0] > coverage[:, 1])
        or np.any(coverage[:, 1] > coverage[:, 2])
    ):
        raise RuntimeError("stage coverage values or intervals differ")
    if (
        not np.isfinite(width).all()
        or np.any(width <= 0.0)
        or np.any(width[:, 0] > width[:, 1])
        or np.any(width[:, 1] > width[:, 2])
    ):
        raise RuntimeError("stage width values or intervals differ")
    if not np.allclose(stage["coverage_target"].to_numpy(float), TARGET, rtol=0, atol=0):
        raise RuntimeError("stage coverage target differs")

    status_index = scalar.set_index(["reporting_family", "setting_id", "method"])
    for key, row in status_index.iterrows():
        profiles = stage[
            stage["reporting_family"].eq(key[0])
            & stage["setting_id"].eq(key[1])
            & stage["method"].eq(key[2])
        ].sort_values("stage_zero_based")
        dataset = str(row["dataset"])
        if tuple(profiles["stage_zero_based"].astype(int)) != tuple(range(HORIZONS[dataset])):
            raise RuntimeError("signed-gamma stage index grid differs")
        stage_coverage = profiles["coverage_mean"].to_numpy(float)
        stage_width = profiles["normalized_width_mean"].to_numpy(float)
        if not math.isclose(float(row["wsc"]), float(stage_coverage.min()), rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("WSC differs from min_t mean_seed(C_seed,t)")
        if int(row["worst_stage_zero_based"]) != int(stage_coverage.argmin()):
            raise RuntimeError("worst stage differs from first argmin")
        if not math.isclose(float(row["mean_coverage"]), float(stage_coverage.mean()), rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("MeanCov differs from the stage mean")
        if not math.isclose(float(row["mean_normalized_width"]), float(stage_width.mean()), rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("mean normalized width differs from stage widths")
        if not math.isclose(float(row["selection_rate"]), int(row["n_selected"]) / 20, rel_tol=0, abs_tol=1e-14):
            raise RuntimeError("selection rate denominator differs from 20")
        if not 0 <= int(row["n_selected"]) <= int(row["n_gate_eligible"]) <= 20:
            raise RuntimeError("selected/gate-eligible counts differ")
        _validate_scalar_interval(row, "selection_rate", "selection_rate_ci95")
        _validate_scalar_interval(row, "wsc", "wsc_ci95")
        _validate_scalar_interval(row, "mean_coverage", "mean_coverage_ci95")
        _validate_scalar_interval(
            row, "mean_normalized_width", "mean_normalized_width_ci95", positive=True
        )
        confirmatory = report_v2._as_bool(row["confirmatory"])
        expected_point = bool(float(row["wsc"]) >= TARGET) if confirmatory else None
        expected_interval = (
            bool(float(row["wsc_ci95_lower"]) >= TARGET) if confirmatory else None
        )
        expected_eligible = (
            bool(float(row["selection_rate"]) >= 0.95 and expected_point)
            if confirmatory
            else None
        )
        if (
            report_v2._nullable_bool(row["point_attainment_at_target"]) != expected_point
            or report_v2._nullable_bool(row["wsc_interval_attainment_at_target"])
            != expected_interval
            or report_v2._nullable_bool(row["point_eligible"]) != expected_eligible
        ):
            raise RuntimeError("point, interval, or selection eligibility differs")


def _validate_scalar_interval(
    row: pd.Series, point: str, interval_prefix: str, *, positive: bool = False
) -> None:
    value = float(row[point])
    lower = float(row[f"{interval_prefix}_lower"])
    upper = float(row[f"{interval_prefix}_upper"])
    if not all(math.isfinite(item) for item in (lower, value, upper)) or not lower <= value <= upper:
        raise RuntimeError(f"{point} interval does not contain its estimate")
    if positive and lower <= 0.0:
        raise RuntimeError(f"{point} interval must be positive")
    if not positive and (lower < 0.0 or upper > 1.0):
        raise RuntimeError(f"{point} interval must lie in [0,1]")


def _validate_signed_budgets(*frames: pd.DataFrame) -> None:
    for frame in frames:
        for field, expected in (
            ("calibration_trajectories_per_seed", CALIBRATION_BUDGET),
            ("grid_trajectories_per_seed", GRID_BUDGET),
            ("evaluation_trajectories_per_seed", EVALUATION_BUDGET),
        ):
            if not frame[field].eq(expected).all():
                raise RuntimeError(f"signed-gamma {field} differs")
        if not frame["method"].map(TARGET_ADAPTATION_BUDGET).eq(
            frame["target_adaptation_trajectories_per_seed"]
        ).all():
            raise RuntimeError("signed-gamma adaptation budget differs")
        if set(frame["budget_status"]) != {"consumed_complete_science"}:
            raise RuntimeError("signed-gamma budget status differs")


def _validate_paired_rows(paired: pd.DataFrame) -> None:
    expected = {
        (dataset, baseline)
        for dataset in DATASETS
        for baseline in METHODS
        if baseline != "SC-PCP"
    }
    if set(zip(paired["dataset"], paired["baseline"])) != expected:
        raise RuntimeError("five-dataset paired contrast grid differs")
    if (
        not paired["confirmatory"].map(report_v2._as_bool).all()
        or not paired["ranking_permitted"].map(report_v2._as_bool).all()
        or not paired["feedback_value"].eq(PRIMARY_GAMMA).all()
        or paired["paired_selected_seeds"].astype(int).lt(1).any()
    ):
        raise RuntimeError("paired contrast identity or availability differs")
    numeric = paired[
        [
            "scpcp_minus_baseline_wsc",
            "scpcp_minus_baseline_wsc_ci95_lower",
            "scpcp_minus_baseline_wsc_ci95_upper",
            "scpcp_to_baseline_geometric_width_ratio",
            "scpcp_to_baseline_geometric_width_ratio_ci95_lower",
            "scpcp_to_baseline_geometric_width_ratio_ci95_upper",
        ]
    ].to_numpy(float)
    if not np.isfinite(numeric).all() or np.any(numeric[:, 3:] <= 0.0):
        raise RuntimeError("paired contrast numeric values are incomplete")
    if (
        np.any(numeric[:, 1] > numeric[:, 0])
        or np.any(numeric[:, 0] > numeric[:, 2])
        or np.any(numeric[:, 4] > numeric[:, 3])
        or np.any(numeric[:, 3] > numeric[:, 5])
    ):
        raise RuntimeError("paired contrast interval does not contain its estimate")


def _dataset_families() -> dict[str, str]:
    return {
        "synthetic": SIGNED_FAMILIES[0],
        "mimic_iv": SIGNED_FAMILIES[1],
        "eicu": SIGNED_FAMILIES[1],
        "inspire": SIGNED_FAMILIES[1],
        "mimic_cxr": SIGNED_FAMILIES[2],
    }


def _setting_id(dataset: str, gamma: float) -> str:
    return f"{dataset}_gamma_{report_v2._gamma_id(float(gamma))}"


def _primary_status(sources: ReportingSources) -> pd.DataFrame:
    rows = sources.status[
        sources.status["reporting_family"].isin(SIGNED_FAMILIES)
        & sources.status["feedback_value"].eq(PRIMARY_GAMMA)
    ]
    order = {dataset: index for index, dataset in enumerate(DATASETS)}
    return rows.assign(_order=rows["dataset"].map(order)).sort_values("_order")


def _primary_stage(sources: ReportingSources) -> pd.DataFrame:
    return sources.stage[
        sources.stage["reporting_family"].isin(SIGNED_FAMILIES)
        & sources.stage["feedback_value"].eq(PRIMARY_GAMMA)
    ]


def _primary_scalar(sources: ReportingSources) -> pd.DataFrame:
    return sources.scalar[
        sources.scalar["reporting_family"].isin(SIGNED_FAMILIES)
        & sources.scalar["feedback_value"].eq(PRIMARY_GAMMA)
    ]


def _signed_scalar(sources: ReportingSources) -> pd.DataFrame:
    return sources.scalar[sources.scalar["reporting_family"].isin(SIGNED_FAMILIES)]


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 5.8,
            "axes.labelsize": 5.9,
            "axes.titlesize": 6.35,
            "xtick.labelsize": 4.8,
            "ytick.labelsize": 4.8,
            "legend.fontsize": 5.0,
            "axes.linewidth": 0.58,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "svg.hashsalt": "scpcp-five-dataset-signed-gamma-results-v1",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def render_gamma_minus4_hero(sources: ReportingSources) -> plt.Figure:
    rows = _primary_scalar(sources)
    figure, axes = plt.subplots(2, 5, figsize=(7.20, 3.35), sharey="row")
    figure.subplots_adjust(
        left=0.064,
        right=0.994,
        bottom=0.19,
        top=0.80,
        wspace=0.18,
        hspace=0.24,
    )
    metrics = (
        ("wsc", "wsc_ci95_lower", "wsc_ci95_upper", "WSC (%)"),
        (
            "mean_coverage",
            "mean_coverage_ci95_lower",
            "mean_coverage_ci95_upper",
            "MeanCov (%)",
        ),
    )
    for column, dataset in enumerate(DATASETS):
        group = rows[rows["dataset"].eq(dataset)].set_index("method")
        for row_index, (point_name, lower_name, upper_name, ylabel) in enumerate(metrics):
            axis = axes[row_index, column]
            for method_index, method in enumerate(METHODS):
                cell = group.loc[method]
                point = 100.0 * float(cell[point_name])
                lower = 100.0 * float(cell[lower_name])
                upper = 100.0 * float(cell[upper_name])
                axis.errorbar(
                    method_index,
                    point,
                    yerr=np.asarray([[point - lower], [upper - point]]),
                    color=report_v2.legacy.METHOD_COLORS[method],
                    marker=report_v2.legacy.METHOD_MARKERS[method],
                    markersize=3.2 if method == "SC-PCP" else 2.55,
                    linestyle="none",
                    elinewidth=0.60,
                    capsize=1.25,
                    capthick=0.50,
                    markeredgewidth=0.28,
                    zorder=4 if method == "SC-PCP" else 3,
                )
            axis.axhline(90.0, color="#20262B", linestyle=(0, (3, 2)), linewidth=0.76)
            axis.grid(axis="y", color="#E1E3E6", linewidth=0.36, zorder=-5)
            axis.tick_params(width=0.52, length=1.8)
            axis.set_xlim(-0.48, len(METHODS) - 0.52)
            axis.set_xticks(range(len(METHODS)))
            if row_index == 1:
                axis.set_xticklabels(
                    ("Std CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP"),
                    rotation=55,
                    ha="right",
                )
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(ylabel)
        axes[0, column].set_title(DATASET_LABELS[dataset], fontweight="bold", pad=3)
        _panel_label(axes[0, column], column)
    _set_shared_percent_limits(axes, rows)
    _add_method_legend(figure)
    return figure


def _set_shared_percent_limits(axes: np.ndarray, rows: pd.DataFrame) -> None:
    for row_index, prefix in enumerate(("wsc", "mean_coverage")):
        lower = 100.0 * rows[f"{prefix}_ci95_lower"].to_numpy(float)
        upper = 100.0 * rows[f"{prefix}_ci95_upper"].to_numpy(float)
        minimum, maximum = report_v2._metric_limits(
            np.append(lower, 90.0),
            np.append(upper, 90.0),
            fallback=(85.0, 100.0),
            quantum=1.0,
        )
        for axis in axes[row_index, :]:
            axis.set_ylim(minimum, maximum)


def render_gamma_minus4_stagewise(sources: ReportingSources) -> plt.Figure:
    profiles = _primary_stage(sources)
    figure, axes = plt.subplots(2, 5, figsize=(7.20, 3.48), sharex="col")
    figure.subplots_adjust(
        left=0.064,
        right=0.994,
        bottom=0.145,
        top=0.80,
        wspace=0.48,
        hspace=0.28,
    )
    coverage_limits = report_v2._metric_limits(
        100.0 * (profiles["coverage_ci95_lower"].to_numpy(float) - TARGET),
        100.0 * (profiles["coverage_ci95_upper"].to_numpy(float) - TARGET),
        fallback=(-5.0, 5.0),
        quantum=0.5,
    )
    for column, dataset in enumerate(DATASETS):
        coverage_axis, width_axis = axes[:, column]
        group = profiles[profiles["dataset"].eq(dataset)]
        coverage_axis.axhspan(coverage_limits[0], 0.0, color="#F8E9E7", zorder=0)
        coverage_axis.axhline(0.0, color="#20262B", linewidth=0.82, zorder=1)
        for method in METHODS:
            method_rows = group[group["method"].eq(method)].sort_values(
                "stage_zero_based"
            )
            report_v2._plot_stage_interval(
                coverage_axis, method_rows, method, metric="coverage"
            )
            report_v2._plot_stage_interval(
                width_axis, method_rows, method, metric="width"
            )
        coverage_axis.set_title(DATASET_LABELS[dataset], fontweight="bold", pad=3)
        coverage_axis.set_ylim(*coverage_limits)
        report_v2._set_width_limits(width_axis, group)
        report_v2._set_stage_axis(width_axis, HORIZONS[dataset])
        for axis in (coverage_axis, width_axis):
            axis.grid(axis="y", color="#E1E3E6", linewidth=0.36, zorder=-5)
            axis.tick_params(width=0.52, length=1.8)
        if column == 0:
            coverage_axis.set_ylabel(r"Stage coverage $-90\%$ (pp)")
            width_axis.set_ylabel("Normalized width")
        width_axis.set_xlabel("Stage, t")
        _panel_label(coverage_axis, column)
    _add_method_legend(figure)
    return figure


def render_signed_gamma_figure(sources: ReportingSources) -> plt.Figure:
    rows = _signed_scalar(sources)
    figure, axes = plt.subplots(3, 5, figsize=(7.20, 5.55), sharex="col")
    figure.subplots_adjust(
        left=0.064,
        right=0.994,
        bottom=0.095,
        top=0.84,
        wspace=0.45,
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
    for column, dataset in enumerate(DATASETS):
        dataset_rows = rows[rows["dataset"].eq(dataset)]
        for axis, (point_name, lower_name, upper_name, ylabel, scale) in zip(
            axes[:, column], metrics, strict=True
        ):
            for method in METHODS:
                selected = dataset_rows[dataset_rows["method"].eq(method)].sort_values(
                    "feedback_value"
                )
                x = selected["feedback_value"].to_numpy(float) + offsets[method]
                point = scale * selected[point_name].to_numpy(float)
                lower = scale * selected[lower_name].to_numpy(float)
                upper = scale * selected[upper_name].to_numpy(float)
                axis.errorbar(
                    x,
                    point,
                    yerr=np.vstack((point - lower, upper - point)),
                    color=report_v2.legacy.METHOD_COLORS[method],
                    linestyle=report_v2.legacy.METHOD_LINESTYLES[method],
                    marker=report_v2.legacy.METHOD_MARKERS[method],
                    markersize=2.45,
                    linewidth=1.0 if method == "SC-PCP" else 0.70,
                    elinewidth=0.44,
                    capsize=0.95,
                    markeredgewidth=0.22,
                )
            axis.axvspan(-4.25, -3.75, color="#E8B84A", alpha=0.14, zorder=-5)
            axis.grid(axis="y", color="#E1E3E6", linewidth=0.36, zorder=-6)
            axis.tick_params(width=0.52, length=1.8)
            if column == 0:
                axis.set_ylabel(ylabel)
        for row_index in (0, 1):
            axes[row_index, column].axhline(
                90.0, color="#20262B", linestyle=(0, (3, 2)), linewidth=0.76
            )
        axes[0, column].set_title(DATASET_LABELS[dataset], fontweight="bold", pad=3)
        _panel_label(axes[0, column], column)
        axes[-1, column].set_xticks(
            GAMMAS, [report_v2._format_gamma(value) for value in GAMMAS]
        )
        axes[-1, column].set_xlabel("γ")
    _set_signed_percent_limits(axes, rows)
    _add_method_legend(figure)
    return figure


def _set_signed_percent_limits(axes: np.ndarray, rows: pd.DataFrame) -> None:
    for row_index, prefix in enumerate(("wsc", "mean_coverage")):
        lower = 100.0 * rows[f"{prefix}_ci95_lower"].to_numpy(float)
        upper = 100.0 * rows[f"{prefix}_ci95_upper"].to_numpy(float)
        limits = report_v2._metric_limits(
            np.append(lower, 90.0),
            np.append(upper, 90.0),
            fallback=(80.0, 100.0),
            quantum=1.0,
        )
        for axis in axes[row_index, :]:
            axis.set_ylim(*limits)


def _panel_label(axis: plt.Axes, column: int) -> None:
    axis.text(
        -0.22,
        1.12,
        chr(ord("a") + column),
        transform=axis.transAxes,
        fontsize=7.6,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _add_method_legend(figure: plt.Figure) -> None:
    figure.legend(
        handles=[report_v2._legend_handle(method) for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        columnspacing=0.68,
        handlelength=1.55,
        handletextpad=0.26,
    )


def render_gamma_minus4_table(sources: ReportingSources) -> plt.Figure:
    rows = _primary_scalar(sources)
    dataset_rank = {dataset: index for index, dataset in enumerate(DATASETS)}
    method_rank = {method: index for index, method in enumerate(METHODS)}
    ordered = rows.assign(
        _dataset_rank=rows["dataset"].map(dataset_rank),
        _method_rank=rows["method"].map(method_rank),
    ).sort_values(["_dataset_rank", "_method_rank"])
    table_rows = []
    for row in ordered.itertuples(index=False):
        table_rows.append(
            [
                DATASET_LABELS[row.dataset] if row.method == METHODS[0] else "",
                row.method,
                report_v2._format_percent_interval_with_gap(
                    row.wsc,
                    row.wsc_ci95_lower,
                    row.wsc_ci95_upper,
                    row.wsc_deviation_from_target_pp,
                ),
                str(int(row.worst_stage_zero_based)),
                report_v2._format_percent_interval_with_gap(
                    row.mean_coverage,
                    row.mean_coverage_ci95_lower,
                    row.mean_coverage_ci95_upper,
                    row.mean_coverage_deviation_from_target_pp,
                ),
                report_v2._format_number_interval(
                    row.mean_normalized_width,
                    row.mean_normalized_width_ci95_lower,
                    row.mean_normalized_width_ci95_upper,
                ),
                (
                    f"{int(row.n_selected)}/20 "
                    f"[{100.0 * float(row.selection_rate_ci95_lower):.1f}, "
                    f"{100.0 * float(row.selection_rate_ci95_upper):.1f}]"
                ),
                f"{int(row.n_gate_eligible)}/20",
                _binary_attainment(row),
                _format_budget(row),
            ]
        )

    figure, axis = plt.subplots(figsize=(7.20, 7.55))
    figure.subplots_adjust(left=0.006, right=0.994, top=0.988, bottom=0.006)
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
            "Selected/20 [Wilson]",
            "Eligible/20",
            "Point/CI/eligible",
            "cal/grid/adapt/eval",
        ),
        colLoc="center",
        cellLoc="center",
        colWidths=(0.09, 0.09, 0.14, 0.03, 0.14, 0.13, 0.135, 0.07, 0.085, 0.09),
        bbox=(0.0, 0.0, 1.0, 1.0),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(4.05)
    for column in range(10):
        header = table[(0, column)]
        header.set_facecolor("#334E68")
        header.set_text_props(color="white", fontweight="bold")
        header.set_edgecolor("white")
        header.set_linewidth(0.50)
    group_colors = ("#F4F7FA", "#FFFFFF")
    for row_index, source_row in enumerate(ordered.itertuples(index=False), start=1):
        for column in range(10):
            cell = table[(row_index, column)]
            cell.set_facecolor(group_colors[dataset_rank[source_row.dataset] % 2])
            cell.set_edgecolor("#D7DEE5")
            cell.set_linewidth(0.30)
            if column == 1:
                cell.set_text_props(ha="left")
        if source_row.method == "SC-PCP":
            table[(row_index, 1)].get_text().set_color(
                report_v2.legacy.METHOD_COLORS["SC-PCP"]
            )
        if (row_index - 1) % len(METHODS) == 0:
            for column in range(10):
                table[(row_index, column)].set_linewidth(0.68)
                table[(row_index, column)].set_edgecolor("#9AA9B5")
    return figure


def _binary_attainment(row: object) -> str:
    return "/".join(
        "1" if report_v2._as_bool(value) else "0"
        for value in (
            row.point_attainment_at_target,
            row.wsc_interval_attainment_at_target,
            row.point_eligible,
        )
    )


def _format_budget(row: object) -> str:
    values = (
        row.calibration_trajectories_per_seed,
        row.grid_trajectories_per_seed,
        row.target_adaptation_trajectories_per_seed,
        row.evaluation_trajectories_per_seed,
    )
    return "/".join(report_v2._short_budget(int(value)) for value in values)


def render_report(
    *,
    native_input: Path,
    clinical_input: Path,
    cxr_input: Path,
    production_input: Path,
    work_output: Path,
    paper_output: Path,
) -> None:
    if work_output.exists() or paper_output.exists():
        raise FileExistsError("work-output and paper-output must both be fresh")
    if work_output == paper_output:
        raise ValueError("work-output and paper-output must differ")
    if not _renderer_is_in_formal_source_snapshot():
        raise RuntimeError("renderer is not inside the top-level tools source-freeze scope")
    sources = build_reporting_sources(
        native_input=native_input,
        clinical_input=clinical_input,
        cxr_input=cxr_input,
        production_input=production_input,
    )

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
            HERO_STEM: render_gamma_minus4_hero(sources),
            STAGE_STEM: render_gamma_minus4_stagewise(sources),
            SIGNED_STEM: render_signed_gamma_figure(sources),
            TABLE_STEM: render_gamma_minus4_table(sources),
        }
        for stem, figure in figures.items():
            _export_figure(figure, staged_work / stem, title=_output_title(stem))
        for name in sorted(PAPER_FILES):
            shutil.copyfile(staged_work / name, staged_paper / name)
        _write_manifest(
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


def _renderer_is_in_formal_source_snapshot() -> bool:
    renderer = Path(__file__).resolve()
    return renderer in {path.resolve() for path in (ROOT / "tools").glob("*.py")}


def _write_contract(path: Path, sources: ReportingSources) -> None:
    payload = {
        "schema_version": 1,
        "protocol": RENDER_PROTOCOL,
        "status": "complete",
        "backend": "Python/matplotlib only",
        "archetype": "five-column quantitative grid",
        "core_conclusion": (
            "The five frozen signed-gamma science records support aligned reporting "
            "of WSC, supplementary MeanCov, normalized width, and selection without "
            "cross-dataset pooling."
        ),
        "scientific_rng_used": False,
        "default_endpoint": {
            "feedback_parameter": "gamma",
            "feedback_value": PRIMARY_GAMMA,
            "dataset_order": list(DATASETS),
            "methods": list(METHODS),
            "source_rule": (
                "Every plotted five-dataset row comes from its dataset's frozen "
                "signed-gamma science record."
            ),
        },
        "panel_map": {
            HERO_STEM: {"rows": ["WSC", "MeanCov"], "columns": list(DATASETS)},
            STAGE_STEM: {
                "rows": ["stage coverage", "stage normalized width"],
                "columns": list(DATASETS),
            },
            SIGNED_STEM: {
                "rows": ["WSC", "MeanCov", "mean normalized width"],
                "columns": list(DATASETS),
                "gammas": list(GAMMAS),
            },
            TABLE_STEM: "complete gamma=-4 scalar metrics for five datasets and six methods",
        },
        "visible_text_policy": {
            "allowed": [
                "axes and ticks",
                "dataset labels",
                "canonical method legend",
                "necessary panel letters",
                "table headers and quantitative cells",
            ],
            "forbidden": [
                "titles and subtitles",
                "claim prose",
                "gate prose",
                "watermarks",
                "footers",
            ],
        },
        "metric_contract": {
            "coverage_target": TARGET,
            "primary_metric": PRIMARY_METRIC,
            "mean_coverage_role": "supplementary and reported separately",
            "wsc_interval": "10000-draw complete-seed-vector percentile bootstrap",
            "stage_interval": "pointwise two-sided Student-t interval",
            "mean_coverage_interval": "two-sided Student-t interval",
            "mean_width_interval": "two-sided Student-t interval",
            "selection_interval": "two-sided Wilson interval",
            "selection_denominator": 20,
            "width_comparison_scope": "within dataset only",
        },
        "information_budget_per_seed": {
            "calibration": CALIBRATION_BUDGET,
            "grid": GRID_BUDGET,
            "evaluation": EVALUATION_BUDGET,
            "target_adaptation": dict(TARGET_ADAPTATION_BUDGET),
        },
        "production_robustness_boundary": {
            "included_in_source_csv": True,
            "included_in_signed_gamma_figures_or_table": False,
            "used_to_fill_signed_gamma_cells": False,
            "reason": "production is a separate no-gamma robustness protocol",
        },
        "source_data": {
            "files": sorted(SOURCE_FILES),
            "status_rows": len(sources.status),
            "stage_rows": len(sources.stage),
            "scalar_rows": len(sources.scalar),
            "paired_rows": len(sources.paired),
        },
        "input_contracts": sources.input_contracts,
        "source_freeze": {
            "renderer_path": "tools/render_five_dataset_signed_gamma_results.py",
            "included_by": "experiment_tree_sha256 tools/*.py",
            "included": _renderer_is_in_formal_source_snapshot(),
            "renderer_sha256": _file_sha256(Path(__file__)),
        },
        "reviewer_risks": [
            "WSC is minimum after stagewise averaging, not mean of seedwise minima.",
            "MeanCov is supplementary and cannot replace WSC.",
            "Selection always uses all 20 prespecified seeds as denominator.",
            "Stage intervals are pointwise rather than simultaneous.",
            "Normalized width is compared within dataset only.",
            "Production robustness values never fill signed-gamma cells.",
            "MIMIC-CXR requires exact confirmation, manifest, COMPLETE, and science bindings.",
        ],
        "export_contract": {
            "work_formats": ["editable SVG", "TrueType PDF", "600-dpi TIFF", "240-dpi PNG"],
            "paper_files": sorted(PAPER_FILES),
            "paper_directory_policy": "PDF only",
        },
    }
    _write_json(path, payload)


def _write_qa(path: Path, sources: ReportingSources) -> None:
    signed = _signed_scalar(sources)
    lines = [
        "# Five-dataset signed-gamma figure QA",
        "",
        "- Backend: Python/matplotlib exclusively; no scientific RNG was drawn.",
        f"- Signed grid: {len(DATASETS)} datasets × {len(GAMMAS)} gammas × {len(METHODS)} methods = {len(signed)} scalar rows.",
        "- Primary endpoint: gamma=-4; WSC=min_t mean_seed(C_seed,t).",
        "- MeanCov is separate supplementary coverage, never substituted for WSC.",
        "- Selection denominator is 20 for every signed-gamma row.",
        "- WSC uses stored 10,000-draw complete-seed-vector percentile intervals.",
        "- MeanCov, width, and stage profiles use stored Student-t intervals; selection uses Wilson intervals.",
        "- All point estimates are finite and lie inside their stored intervals.",
        "- All signed rows use 3,000 calibration, 1,000 grid, 20,000 evaluation, and frozen method-specific adaptation budgets.",
        "- MIMIC-CXR passed science-runner confirmation-binding, manifest, and COMPLETE validation before adaptation.",
        "- Production/no-gamma robustness rows are retained only in source CSV and input provenance; they do not supplement any signed-gamma plot or table cell.",
        "- Plots contain axes/ticks, dataset labels, canonical legend, and necessary panel letters only; the table contains headers and quantitative cells only.",
        "- Times New Roman is requested first; SVG text remains editable and PDF text uses TrueType fonts.",
        "- Paper output contains PDFs only; work output contains all source, audit, vector, and raster files.",
        "- The top-level renderer is inside the formal experiment_tree_sha256 tools/*.py source snapshot.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _export_figure(figure: plt.Figure, stem: Path, *, title: str) -> None:
    creator = "SC-PCP five-dataset signed-gamma results v1"
    figure.savefig(
        stem.with_suffix(".svg"),
        format="svg",
        bbox_inches="tight",
        metadata={"Title": title, "Creator": creator, "Date": None},
    )
    figure.savefig(
        stem.with_suffix(".pdf"),
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
        stem.with_suffix(".tiff"),
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(
        stem.with_suffix(".png"),
        format="png",
        dpi=240,
        bbox_inches="tight",
        metadata={"Software": creator},
    )
    plt.close(figure)


def _output_title(stem: str) -> str:
    return {
        HERO_STEM: "Five-dataset gamma-minus-4 WSC and MeanCov",
        STAGE_STEM: "Five-dataset gamma-minus-4 stage coverage and width",
        SIGNED_STEM: "Five-dataset signed-gamma WSC, MeanCov, and width",
        TABLE_STEM: "Five-dataset gamma-minus-4 complete metrics",
    }[stem]


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_written_sources(work_root: Path) -> None:
    sources = ReportingSources(
        status=pd.read_csv(work_root / "setting_status.csv", float_precision="round_trip"),
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


def _write_manifest(path: Path, *, work_root: Path, paper_root: Path) -> None:
    if {item.name for item in paper_root.iterdir() if item.is_file()} != PAPER_FILES:
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
        raise RuntimeError("five-dataset work bundle entry set differs")
    if observed_paper != PAPER_FILES or any(
        path.suffix != ".pdf" for path in paper_root.iterdir()
    ):
        raise RuntimeError("five-dataset paper bundle must contain PDFs only")
    _validate_written_sources(work_root)
    manifest = report_v2._read_json(work_root / "render_manifest.json")
    if (
        manifest.get("protocol") != RENDER_PROTOCOL
        or manifest.get("status") != "complete"
        or set(report_v2._mapping(manifest.get("work_files"), "work manifest"))
        != WORK_FILES - {"render_manifest.json", "COMPLETE"}
        or set(report_v2._mapping(manifest.get("paper_files"), "paper manifest"))
        != PAPER_FILES
    ):
        raise RuntimeError("five-dataset render manifest differs")
    expected_complete = (
        f"protocol={RENDER_PROTOCOL}\n"
        f"manifest_sha256={_file_sha256(work_root / 'render_manifest.json')}\n"
    )
    if (work_root / "COMPLETE").read_text(encoding="utf-8") != expected_complete:
        raise RuntimeError("five-dataset COMPLETE marker differs")
    for stem in OUTPUT_STEMS:
        svg_path = work_root / f"{stem}.svg"
        svg = svg_path.read_text(encoding="utf-8")
        if "<text" not in svg or "Times New Roman" not in svg:
            raise RuntimeError(f"{stem} editable Times New Roman SVG differs")
        visible = _svg_visible_text(svg_path)
        for forbidden in (
            "Core conclusion",
            "claim",
            "Gate:",
            "watermark",
            "footer",
            "complete science",
        ):
            if forbidden.lower() in visible.lower():
                raise RuntimeError(f"{stem} contains forbidden visible prose")
        work_pdf = work_root / f"{stem}.pdf"
        paper_pdf = paper_root / f"{stem}.pdf"
        if not work_pdf.read_bytes().startswith(b"%PDF") or not paper_pdf.read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"{stem} PDF is malformed")
        if _file_sha256(work_pdf) != _file_sha256(paper_pdf):
            raise RuntimeError(f"{stem} paper PDF differs from work PDF")
        if not (work_root / f"{stem}.png").read_bytes().startswith(b"\x89PNG"):
            raise RuntimeError(f"{stem} PNG is malformed")
        with Image.open(work_root / f"{stem}.tiff") as image:
            dpi = image.info.get("dpi")
            if dpi is None or not all(math.isclose(float(value), 600.0, rel_tol=1e-3) for value in dpi):
                raise RuntimeError(f"{stem} TIFF is not 600 dpi")
    for group, root in (("work_files", work_root), ("paper_files", paper_root)):
        contracts = report_v2._mapping(manifest[group], group)
        for name, contract in contracts.items():
            _validate_file_contract(root / name, report_v2._mapping(contract, name))


def _svg_visible_text(path: Path) -> str:
    root = ElementTree.parse(path).getroot()
    return " ".join(
        "".join(node.itertext())
        for node in root.iter()
        if node.tag.rsplit("}", 1)[-1] == "text"
    )


def _file_contract(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _validate_file_contract(path: Path, contract: Mapping[str, Any]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(contract["bytes"])
        or _file_sha256(path) != contract["sha256"]
    ):
        raise RuntimeError(f"rendered file contract differs: {path.name}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
