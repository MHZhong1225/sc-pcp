"""Render complete coverage reporting from frozen SC-PCP artifacts.

This deterministic reporting command does not fit models, run rollouts, or
resample uncertainty intervals.  It reads already-frozen production and
controlled-clinical summaries, validates their scientific identities, and
publishes a separate figure/table bundle.

Example
-------
conda run -n ucp python tools/render_complete_coverage_reporting.py
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
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RENDER_PROTOCOL = "complete_coverage_reporting_v1"
DEFAULT_PRODUCTION_INPUT = ROOT / "results/work/complete_baseline_results_20260824"
DEFAULT_CLINICAL_INPUT = ROOT / "results/work/controlled_clinical_extension_v2"
DEFAULT_WORK_OUTPUT = ROOT / "results/work/complete_coverage_reporting_20260827"
DEFAULT_PAPER_OUTPUT = ROOT / "results/paper_complete_coverage_reporting_20260827"

MAIN_STAGE_STEM = "figure_gamma_minus4_stagewise_profiles"
MAIN_TABLE_STEM = "table_gamma_minus4_complete_metrics"
SIGNED_FIGURE_STEM = "figure_mimic_iv_v2_signed_gamma_metrics"
SIGNED_TABLE_STEM = "table_mimic_iv_v2_signed_gamma_metrics"
PRODUCTION_TABLE_STEM = "table_production_complete_metrics"
FIGURE_STEMS = (MAIN_STAGE_STEM, SIGNED_FIGURE_STEM)
TABLE_STEMS = (MAIN_TABLE_STEM, SIGNED_TABLE_STEM, PRODUCTION_TABLE_STEM)
OUTPUT_STEMS = (*FIGURE_STEMS, *TABLE_STEMS)

TARGET = 0.90
METHODS = ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")
PRODUCTION_DATASETS = ("synthetic", "mimic_iv", "mimic_cxr", "eicu", "inspire")
CLINICAL_DATASETS = ("mimic_iv", "eicu", "inspire", "mimic_cxr")
SIGNED_GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
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
DATASET_LABELS = {
    "synthetic": "Synthetic",
    "mimic_iv": "MIMIC-IV",
    "mimic_cxr": "MIMIC-CXR + IV/ED",
    "eicu": "eICU",
    "inspire": "INSPIRE",
}
INFORMATION_REGIME = {
    "Standard CP": "offline_logged_data",
    "ACI": "on_policy_adaptation",
    "MFCS": "offline_logged_data",
    "SPCI": "on_policy_adaptation",
    "PRC": "on_policy_adaptation",
    "SC-PCP": "offline_logged_data",
}
TARGET_ADAPTATION_BUDGET = {
    "Standard CP": 0,
    "ACI": 2_000,
    "MFCS": 0,
    "SPCI": 2_000,
    "PRC": 2_000,
    "SC-PCP": 0,
}
PRODUCTION_EVALUATION_BUDGET = 50_000
CONTROLLED_EVALUATION_BUDGET = 20_000
CONTROLLED_CALIBRATION_BUDGET = 3_000
CONTROLLED_GRID_BUDGET = 1_000

NORMALIZED_WIDTH_DEFINITION = (
    "d^{-1} sum_j [2 q_t sigma_hat_{i,t,j} / sigma_out_{s,j}]; "
    "mean normalized coordinate length, not area or log-volume"
)
PRIMARY_METRIC = "min_t mean_seed(target_coverage_seed_t)"
POINT_ELIGIBILITY_RULE = "selection_rate>=0.95 and WSC>=0.90"

METHOD_COLORS = {
    "Standard CP": "#4D5560",
    "ACI": "#C47A16",
    "MFCS": "#7B61A8",
    "SPCI": "#55A7C9",
    "PRC": "#A85B87",
    "SC-PCP": "#1769AA",
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
    "MFCS": (0, (1, 1.5)),
    "SPCI": (0, (2, 2)),
    "PRC": (0, (5, 2, 1, 2)),
    "SC-PCP": "-",
}

STATUS_COLUMNS = (
    "schema_version",
    "reporting_family",
    "setting_id",
    "display_label",
    "dataset",
    "setting_type",
    "feedback_parameter",
    "feedback_value",
    "horizon",
    "method_count",
    "panel_status",
    "interpretation_status",
    "confirmatory",
    "scientific_rows_available",
    "support_available",
    "k0_fidelity_available",
    "n_prespecified",
    "hard_gate_reason",
    "source_path",
    "source_sha256",
)

STAGE_COLUMNS = (
    "schema_version",
    "reporting_family",
    "setting_id",
    "display_label",
    "dataset",
    "setting_type",
    "feedback_parameter",
    "feedback_value",
    "analysis_role",
    "panel_status",
    "confirmatory",
    "method",
    "information_regime",
    "target_adaptation_trajectories_per_seed",
    "evaluation_trajectories_per_seed",
    "calibration_trajectories_per_seed",
    "grid_trajectories_per_seed",
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
    "setting_type",
    "feedback_parameter",
    "feedback_value",
    "analysis_role",
    "panel_status",
    "confirmatory",
    "method",
    "information_regime",
    "target_adaptation_trajectories_per_seed",
    "evaluation_trajectories_per_seed",
    "calibration_trajectories_per_seed",
    "grid_trajectories_per_seed",
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
    "point_eligible",
    "efficiency_rank_defined",
    "narrowest_point_eligible",
    "source_path",
    "source_sha256",
)

SOURCE_FILES = {
    "setting_status.csv",
    "coverage_stage_profiles.csv",
    "coverage_scalar_summary.csv",
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
class ClinicalSource:
    protocol: str
    status: pd.DataFrame
    gamma_minus4_stage: pd.DataFrame
    gamma_minus4_scalar: pd.DataFrame
    signed_stage: pd.DataFrame
    signed_scalar: pd.DataFrame
    contract: Mapping[str, Any]


@dataclass(frozen=True)
class ClinicalArtifactAdapter:
    """An exact, protocol-specific validation and adaptation boundary.

    A future v3 must add its own adapter and protocol identifier.  Unknown
    layouts never fall through to a permissive JSON reader.
    """

    protocol: str
    load: Callable[[Path], ClinicalSource]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-input", type=Path, default=DEFAULT_PRODUCTION_INPUT)
    parser.add_argument("--clinical-input", type=Path, default=DEFAULT_CLINICAL_INPUT)
    parser.add_argument("--work-output", type=Path, default=DEFAULT_WORK_OUTPUT)
    parser.add_argument("--paper-output", type=Path, default=DEFAULT_PAPER_OUTPUT)
    args = parser.parse_args()
    render_report(
        production_input=args.production_input.resolve(),
        clinical_input=args.clinical_input.resolve(),
        work_output=args.work_output.resolve(),
        paper_output=args.paper_output.resolve(),
    )
    print(args.paper_output.resolve())


def render_report(
    *,
    production_input: Path,
    clinical_input: Path,
    work_output: Path,
    paper_output: Path,
) -> None:
    """Validate frozen inputs and atomically publish the reporting bundle."""

    if work_output.exists() or paper_output.exists():
        raise FileExistsError("work-output and paper-output must both be fresh")
    if work_output == paper_output:
        raise ValueError("work-output and paper-output must be different directories")

    status, stage, scalar, input_contract = build_reporting_sources(
        production_input=production_input,
        clinical_input=clinical_input,
    )
    validate_reporting_sources(status, stage, scalar)

    work_output.parent.mkdir(parents=True, exist_ok=True)
    paper_output.parent.mkdir(parents=True, exist_ok=True)
    staged_work = Path(tempfile.mkdtemp(prefix=f".{work_output.name}-", dir=work_output.parent))
    staged_paper = Path(tempfile.mkdtemp(prefix=f".{paper_output.name}-", dir=paper_output.parent))
    try:
        _write_csv(staged_work / "setting_status.csv", status)
        _write_csv(staged_work / "coverage_stage_profiles.csv", stage)
        _write_csv(staged_work / "coverage_scalar_summary.csv", scalar)
        _validate_written_sources(staged_work)
        _write_contract(
            staged_work / "figure_contract.json",
            status=status,
            stage=stage,
            scalar=scalar,
            input_contract=input_contract,
        )
        _write_qa(
            staged_work / "figure_qa.md",
            status=status,
            stage=stage,
            scalar=scalar,
        )

        apply_publication_style()
        figures = {
            MAIN_STAGE_STEM: render_gamma_minus4_stagewise(status, stage),
            MAIN_TABLE_STEM: render_gamma_minus4_table(status, scalar),
            SIGNED_FIGURE_STEM: render_signed_gamma_figure(stage, scalar),
            SIGNED_TABLE_STEM: render_signed_gamma_table(scalar),
            PRODUCTION_TABLE_STEM: render_production_table(scalar),
        }
        for stem, figure in figures.items():
            export_figure(figure, work_stem=staged_work / stem, title=_output_title(stem))

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


def build_reporting_sources(
    *, production_input: Path, clinical_input: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    production_status, production_stage, production_scalar, production_contract = (
        load_production_sources(production_input)
    )
    clinical = load_clinical_sources(clinical_input)
    status = pd.concat([production_status, clinical.status], ignore_index=True).loc[
        :, STATUS_COLUMNS
    ]
    stage = pd.concat(
        [production_stage, clinical.gamma_minus4_stage, clinical.signed_stage],
        ignore_index=True,
    ).loc[:, STAGE_COLUMNS]
    scalar = pd.concat(
        [production_scalar, clinical.gamma_minus4_scalar, clinical.signed_scalar],
        ignore_index=True,
    ).loc[:, SCALAR_COLUMNS]
    scalar = _assign_efficiency_ranking(scalar)
    return status, stage, scalar, {
        "production": production_contract,
        "controlled_clinical": clinical.contract,
    }


def load_production_sources(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Adapt the exact frozen five-setting RQ1 export without resampling."""

    from tools import render_main_suite_figures as frozen

    main_frame, stage_frame, metadata = frozen.load_frozen_export(root)
    main = main_frame[main_frame["section"].eq("RQ1")].copy()
    stages = stage_frame[stage_frame["section"].eq("RQ1")].copy()
    main_index = main.set_index(["dataset", "method"])
    stage_path = root / "per_stage_all_baselines.csv"
    scalar_path = root / "rq1_all_baselines.csv"

    status_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    for dataset in PRODUCTION_DATASETS:
        setting_id = f"production_{dataset}"
        display_label = DATASET_LABELS[dataset]
        status_rows.append(
            _status_row(
                reporting_family="production_no_gamma_supplement",
                setting_id=setting_id,
                display_label=display_label,
                dataset=dataset,
                setting_type="production_native_no_controlled_gamma",
                feedback_parameter="beta" if dataset == "synthetic" else "none",
                feedback_value=1.0 if dataset == "synthetic" else None,
                horizon=HORIZONS[dataset],
                panel_status="CURVES",
                interpretation_status="FROZEN_PRODUCTION_RQ1",
                confirmatory=False,
                scientific_rows_available=True,
                support_available=None,
                k0_fidelity_available=None,
                n_prespecified=EXPECTED_RUNS[dataset],
                hard_gate_reason="",
                source_path=_project_path(root / "metadata.json"),
                source_sha256=frozen.INPUT_FILES["metadata.json"],
            )
        )
        for method in METHODS:
            method_stages = stages[
                stages["dataset"].eq(dataset) & stages["method"].eq(method)
            ].sort_values("stage_zero_based")
            scalar_source = main_index.loc[(dataset, method)]
            worst_stage = int(method_stages["coverage_mean"].to_numpy(float).argmin())
            for row in method_stages.itertuples(index=False):
                stage_rows.append(
                    _stage_row(
                        reporting_family="production_no_gamma_supplement",
                        setting_id=setting_id,
                        display_label=display_label,
                        dataset=dataset,
                        setting_type="production_native_no_controlled_gamma",
                        feedback_parameter="beta" if dataset == "synthetic" else "none",
                        feedback_value=1.0 if dataset == "synthetic" else None,
                        analysis_role="frozen_production_native_rq1",
                        panel_status="CURVES",
                        confirmatory=False,
                        method=method,
                        evaluation_budget=PRODUCTION_EVALUATION_BUDGET,
                        calibration_budget=None,
                        grid_budget=None,
                        horizon=HORIZONS[dataset],
                        stage=int(row.stage_zero_based),
                        n_prespecified=int(row.n_runs),
                        n_gate_eligible=int(row.n_runs),
                        n_selected=int(row.n_selected),
                        coverage_conditioning="successful method selection",
                        coverage=float(row.coverage_mean),
                        coverage_interval=(float(row.coverage_ci_low), float(row.coverage_ci_high)),
                        coverage_interval_definition="pointwise two-sided 95% Student-t interval across selected seeds",
                        width=float(row.normalized_width_mean),
                        width_interval=(float(row.normalized_width_ci_low), float(row.normalized_width_ci_high)),
                        width_interval_definition="pointwise two-sided 95% Student-t interval across selected seeds",
                        source_path=_project_path(stage_path),
                        source_sha256=frozen.INPUT_FILES[stage_path.name],
                    )
                )
            scalar_rows.append(
                _scalar_row(
                    reporting_family="production_no_gamma_supplement",
                    setting_id=setting_id,
                    display_label=display_label,
                    dataset=dataset,
                    setting_type="production_native_no_controlled_gamma",
                    feedback_parameter="beta" if dataset == "synthetic" else "none",
                    feedback_value=1.0 if dataset == "synthetic" else None,
                    analysis_role="frozen_production_native_rq1",
                    panel_status="CURVES",
                    confirmatory=False,
                    method=method,
                    evaluation_budget=PRODUCTION_EVALUATION_BUDGET,
                    calibration_budget=None,
                    grid_budget=None,
                    n_prespecified=int(scalar_source["n_runs"]),
                    n_gate_eligible=int(scalar_source["n_runs"]),
                    n_selected=int(scalar_source["n_selected"]),
                    metric_available=True,
                    coverage_conditioning="successful method selection",
                    selection_rate=float(scalar_source["selection_rate"]),
                    selection_interval=(
                        float(scalar_source["selection_rate_ci_low"]),
                        float(scalar_source["selection_rate_ci_high"]),
                    ),
                    wsc=float(scalar_source["marginal_worst_coverage"]),
                    wsc_interval=(
                        float(scalar_source["marginal_worst_coverage_ci_low"]),
                        float(scalar_source["marginal_worst_coverage_ci_high"]),
                    ),
                    wsc_interval_definition="10000 complete-seed-vector percentile bootstrap draws",
                    worst_stage=worst_stage,
                    mean_coverage=float(scalar_source["average_coverage"]),
                    mean_coverage_interval=(
                        float(scalar_source["average_coverage_ci_low"]),
                        float(scalar_source["average_coverage_ci_high"]),
                    ),
                    mean_width=float(scalar_source["average_normalized_width"]),
                    mean_width_interval=(
                        float(scalar_source["average_normalized_width_ci_low"]),
                        float(scalar_source["average_normalized_width_ci_high"]),
                    ),
                    point_eligible=bool(scalar_source["efficiency_eligible"]),
                    source_path=_project_path(scalar_path),
                    source_sha256=frozen.INPUT_FILES[scalar_path.name],
                )
            )

    return (
        pd.DataFrame(status_rows).loc[:, STATUS_COLUMNS],
        pd.DataFrame(stage_rows).loc[:, STAGE_COLUMNS],
        pd.DataFrame(scalar_rows).loc[:, SCALAR_COLUMNS],
        {
            "protocol": frozen.RENDER_PROTOCOL,
            "input_root": _project_path(root),
            "input_file_sha256": dict(frozen.INPUT_FILES),
            "metadata": metadata,
        },
    )


def load_clinical_sources(root: Path) -> ClinicalSource:
    metadata = _read_json(root / "metadata.json")
    protocol = metadata.get("protocol")
    adapters = _clinical_adapters()
    if protocol not in adapters:
        raise RuntimeError(
            f"unsupported controlled-clinical protocol: {protocol!r}; "
            "register an exact protocol adapter before rendering"
        )
    return adapters[str(protocol)].load(root)


def _clinical_adapters() -> Mapping[str, ClinicalArtifactAdapter]:
    from scripts import run_controlled_clinical_extension as runner

    return {
        runner.PROTOCOL: ClinicalArtifactAdapter(
            protocol=runner.PROTOCOL,
            load=_load_controlled_clinical_v2,
        )
    }


def _load_controlled_clinical_v2(root: Path) -> ClinicalSource:
    """Load only after the formal v2 validators accept the complete bundle."""

    from tools import render_five_setting_stage_profiles as frozen
    from scripts import run_controlled_clinical_extension as runner

    status_source, stage_source, _, contract = frozen.load_complete_clinical(root)
    status_rows: list[dict[str, Any]] = []
    main_stage_rows: list[dict[str, Any]] = []
    main_scalar_rows: list[dict[str, Any]] = []

    for dataset in CLINICAL_DATASETS:
        source_status = status_source[status_source["dataset"].eq(dataset)].iloc[0]
        setting_id = f"{dataset}_gamma_minus4"
        display_label = DATASET_LABELS[dataset]
        panel_status = str(source_status["panel_status"])
        status_rows.append(
            _status_row(
                reporting_family="clinical_gamma_minus4_main",
                setting_id=setting_id,
                display_label=display_label,
                dataset=dataset,
                setting_type="dataset_native_clinical_controlled",
                feedback_parameter="gamma",
                feedback_value=-4.0,
                horizon=HORIZONS[dataset],
                panel_status=panel_status,
                interpretation_status=str(source_status["interpretation_status"]),
                confirmatory=bool(source_status["confirmatory_ranking_included"]),
                scientific_rows_available=bool(source_status["scientific_rows_saved"]),
                support_available=_optional_int(source_status["support_available"]),
                k0_fidelity_available=_optional_int(source_status["k0_fidelity_available"]),
                n_prespecified=EXPECTED_RUNS[dataset],
                hard_gate_reason=_optional_text(source_status["hard_gate_reason"]),
                source_path=str(source_status["source_path"]),
                source_sha256=str(source_status["source_sha256"]),
            )
        )
        if panel_status == "GATE_NO_GO":
            for method in METHODS:
                main_scalar_rows.append(
                    _unavailable_scalar_row(
                        reporting_family="clinical_gamma_minus4_main",
                        setting_id=setting_id,
                        display_label=display_label,
                        dataset=dataset,
                        feedback_value=-4.0,
                        panel_status=panel_status,
                        analysis_role="hard_gate_no_go_no_science_rows",
                        method=method,
                        n_prespecified=EXPECTED_RUNS[dataset],
                        n_gate_eligible=_optional_int(source_status["k0_fidelity_available"]),
                        source_path=str(source_status["source_path"]),
                        source_sha256=str(source_status["source_sha256"]),
                    )
                )
            continue

        science_path = root / dataset / "science" / "summary.json"
        science = runner._read_json(science_path)
        aggregate = _clinical_aggregate(science, gamma=-4.0)
        for method in METHODS:
            cell = _mapping(aggregate["methods"][method], f"{dataset}/gamma=-4/{method}")
            main_stage_rows.extend(
                _clinical_stage_rows(
                    reporting_family="clinical_gamma_minus4_main",
                    setting_id=setting_id,
                    display_label=display_label,
                    dataset=dataset,
                    feedback_value=-4.0,
                    analysis_role=str(aggregate["analysis_role"]),
                    panel_status=panel_status,
                    confirmatory=True,
                    method=method,
                    cell=cell,
                    horizon=HORIZONS[dataset],
                    n_prespecified=EXPECTED_RUNS[dataset],
                    n_gate_eligible=int(aggregate["n_k0_eligible_seeds"]),
                    source_path=_project_path(science_path),
                    source_sha256=_file_sha256(science_path),
                )
            )
            main_scalar_rows.append(
                _clinical_scalar_row(
                    reporting_family="clinical_gamma_minus4_main",
                    setting_id=setting_id,
                    display_label=display_label,
                    dataset=dataset,
                    feedback_value=-4.0,
                    analysis_role=str(aggregate["analysis_role"]),
                    panel_status=panel_status,
                    confirmatory=True,
                    method=method,
                    cell=cell,
                    n_prespecified=EXPECTED_RUNS[dataset],
                    n_gate_eligible=int(aggregate["n_k0_eligible_seeds"]),
                    source_path=_project_path(science_path),
                    source_sha256=_file_sha256(science_path),
                )
            )

    mimic_status = status_source[status_source["dataset"].eq("mimic_iv")].iloc[0]
    if (
        mimic_status["panel_status"] != "CURVES"
        or not bool(mimic_status["confirmatory_ranking_included"])
    ):
        raise RuntimeError("the v2 MIMIC-IV signed supplement requires its formal CURVES endpoint")
    mimic_science_path = root / "mimic_iv" / "science" / "summary.json"
    mimic_science = runner._read_json(mimic_science_path)
    mimic_hash = _file_sha256(mimic_science_path)
    signed_status_rows: list[dict[str, Any]] = []
    signed_stage_rows: list[dict[str, Any]] = []
    signed_scalar_rows: list[dict[str, Any]] = []
    for gamma in SIGNED_GAMMAS:
        aggregate = _clinical_aggregate(mimic_science, gamma=gamma)
        confirmatory = gamma == -4.0
        setting_id = f"mimic_iv_signed_gamma_{_gamma_id(gamma)}"
        signed_status_rows.append(
            _status_row(
                reporting_family="mimic_iv_v2_signed_gamma_supplement",
                setting_id=setting_id,
                display_label=f"MIMIC-IV, gamma={_format_gamma(gamma)}",
                dataset="mimic_iv",
                setting_type="dataset_native_clinical_controlled",
                feedback_parameter="gamma",
                feedback_value=gamma,
                horizon=HORIZONS["mimic_iv"],
                panel_status="CURVES",
                interpretation_status=(
                    "CONFIRMATORY_GAMMA_MINUS4_ENDPOINT"
                    if confirmatory
                    else "DESCRIPTIVE_SIGNED_CONTROL"
                ),
                confirmatory=confirmatory,
                scientific_rows_available=True,
                support_available=int(mimic_status["support_available"]),
                k0_fidelity_available=int(mimic_status["k0_fidelity_available"]),
                n_prespecified=EXPECTED_RUNS["mimic_iv"],
                hard_gate_reason="",
                source_path=_project_path(mimic_science_path),
                source_sha256=mimic_hash,
            )
        )
        for method in METHODS:
            cell = _mapping(aggregate["methods"][method], f"mimic_iv/gamma={gamma}/{method}")
            signed_stage_rows.extend(
                _clinical_stage_rows(
                    reporting_family="mimic_iv_v2_signed_gamma_supplement",
                    setting_id=setting_id,
                    display_label=f"MIMIC-IV, gamma={_format_gamma(gamma)}",
                    dataset="mimic_iv",
                    feedback_value=gamma,
                    analysis_role=str(aggregate["analysis_role"]),
                    panel_status="CURVES",
                    confirmatory=confirmatory,
                    method=method,
                    cell=cell,
                    horizon=HORIZONS["mimic_iv"],
                    n_prespecified=EXPECTED_RUNS["mimic_iv"],
                    n_gate_eligible=int(aggregate["n_k0_eligible_seeds"]),
                    source_path=_project_path(mimic_science_path),
                    source_sha256=mimic_hash,
                )
            )
            signed_scalar_rows.append(
                _clinical_scalar_row(
                    reporting_family="mimic_iv_v2_signed_gamma_supplement",
                    setting_id=setting_id,
                    display_label=f"MIMIC-IV, gamma={_format_gamma(gamma)}",
                    dataset="mimic_iv",
                    feedback_value=gamma,
                    analysis_role=str(aggregate["analysis_role"]),
                    panel_status="CURVES",
                    confirmatory=confirmatory,
                    method=method,
                    cell=cell,
                    n_prespecified=EXPECTED_RUNS["mimic_iv"],
                    n_gate_eligible=int(aggregate["n_k0_eligible_seeds"]),
                    source_path=_project_path(mimic_science_path),
                    source_sha256=mimic_hash,
                )
            )

    status_rows.extend(signed_status_rows)
    return ClinicalSource(
        protocol=runner.PROTOCOL,
        status=pd.DataFrame(status_rows).loc[:, STATUS_COLUMNS],
        gamma_minus4_stage=pd.DataFrame(main_stage_rows, columns=STAGE_COLUMNS),
        gamma_minus4_scalar=pd.DataFrame(main_scalar_rows).loc[:, SCALAR_COLUMNS],
        signed_stage=pd.DataFrame(signed_stage_rows).loc[:, STAGE_COLUMNS],
        signed_scalar=pd.DataFrame(signed_scalar_rows).loc[:, SCALAR_COLUMNS],
        contract={
            **contract,
            "adapter": "exact controlled_clinical_extension_v2",
            "signed_supplement_dataset": "mimic_iv",
            "signed_gammas": list(SIGNED_GAMMAS),
            "future_protocol_rule": (
                "a future v3 requires a new exact ClinicalArtifactAdapter; "
                "unknown protocols fail closed"
            ),
        },
    )


def _clinical_aggregate(science: Mapping[str, Any], *, gamma: float) -> Mapping[str, Any]:
    matches = [
        cell
        for cell in science.get("aggregates", ())
        if isinstance(cell, Mapping) and float(cell.get("gamma")) == gamma
    ]
    if len(matches) != 1:
        raise RuntimeError(f"clinical summary must contain exactly one gamma={gamma:g} aggregate")
    aggregate = matches[0]
    expected_role = (
        "confirmatory_gamma_minus_4_endpoint"
        if gamma == -4.0
        else "descriptive_signed_control_curve"
    )
    if (
        aggregate.get("analysis_role") != expected_role
        or aggregate.get("n_prespecified_seeds") != 20
        or set(_mapping(aggregate.get("methods"), "clinical aggregate methods")) != set(METHODS)
    ):
        raise RuntimeError(f"clinical gamma={gamma:g} aggregate semantics differ")
    return aggregate


def _clinical_stage_rows(
    *,
    reporting_family: str,
    setting_id: str,
    display_label: str,
    dataset: str,
    feedback_value: float,
    analysis_role: str,
    panel_status: str,
    confirmatory: bool,
    method: str,
    cell: Mapping[str, Any],
    horizon: int,
    n_prespecified: int,
    n_gate_eligible: int,
    source_path: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    coverage = _finite_vector(cell.get("target_coverage_by_stage"), horizon, "coverage")
    coverage_ci = _finite_intervals(
        cell.get("target_coverage_by_stage_ci95"), coverage, "coverage interval"
    )
    width = _finite_vector(
        cell.get("target_normalized_width_by_stage"), horizon, "normalized width", positive=True
    )
    width_ci = _finite_intervals(
        cell.get("target_normalized_width_by_stage_ci95"),
        width,
        "normalized-width interval",
        positive=True,
    )
    n_selected = int(cell["n_selected"])
    return [
        _stage_row(
            reporting_family=reporting_family,
            setting_id=setting_id,
            display_label=display_label,
            dataset=dataset,
            setting_type="dataset_native_clinical_controlled",
            feedback_parameter="gamma",
            feedback_value=feedback_value,
            analysis_role=analysis_role,
            panel_status=panel_status,
            confirmatory=confirmatory,
            method=method,
            evaluation_budget=CONTROLLED_EVALUATION_BUDGET,
            calibration_budget=CONTROLLED_CALIBRATION_BUDGET,
            grid_budget=CONTROLLED_GRID_BUDGET,
            horizon=horizon,
            stage=stage,
            n_prespecified=n_prespecified,
            n_gate_eligible=n_gate_eligible,
            n_selected=n_selected,
            coverage_conditioning="successful method selection among K0-eligible seeds",
            coverage=float(coverage[stage]),
            coverage_interval=(float(coverage_ci[stage, 0]), float(coverage_ci[stage, 1])),
            coverage_interval_definition=(
                "pointwise 95% percentile bootstrap across complete selected "
                "seed-stage vectors; frozen dataset stream; 10000 resamples"
            ),
            width=float(width[stage]),
            width_interval=(float(width_ci[stage, 0]), float(width_ci[stage, 1])),
            width_interval_definition=(
                "pointwise 95% percentile bootstrap across complete selected "
                "seed-stage vectors; frozen dataset stream; 10000 resamples"
            ),
            source_path=source_path,
            source_sha256=source_sha256,
        )
        for stage in range(horizon)
    ]


def _clinical_scalar_row(
    *,
    reporting_family: str,
    setting_id: str,
    display_label: str,
    dataset: str,
    feedback_value: float,
    analysis_role: str,
    panel_status: str,
    confirmatory: bool,
    method: str,
    cell: Mapping[str, Any],
    n_prespecified: int,
    n_gate_eligible: int,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    n_selected = int(cell["n_selected"])
    if n_selected < 1:
        raise RuntimeError("complete reporting requires an available method row")
    return _scalar_row(
        reporting_family=reporting_family,
        setting_id=setting_id,
        display_label=display_label,
        dataset=dataset,
        setting_type="dataset_native_clinical_controlled",
        feedback_parameter="gamma",
        feedback_value=feedback_value,
        analysis_role=analysis_role,
        panel_status=panel_status,
        confirmatory=confirmatory,
        method=method,
        evaluation_budget=CONTROLLED_EVALUATION_BUDGET,
        calibration_budget=CONTROLLED_CALIBRATION_BUDGET,
        grid_budget=CONTROLLED_GRID_BUDGET,
        n_prespecified=n_prespecified,
        n_gate_eligible=n_gate_eligible,
        n_selected=n_selected,
        metric_available=True,
        coverage_conditioning="successful method selection among K0-eligible seeds",
        selection_rate=float(cell["selection_rate"]),
        selection_interval=tuple(float(value) for value in cell["selection_rate_ci95"]),
        wsc=float(cell["target_marginal_worst_coverage"]),
        wsc_interval=tuple(float(value) for value in cell["target_wsc_ci95"]),
        wsc_interval_definition=(
            "95% percentile interval over min_t of each complete-seed-vector "
            "bootstrap stage-mean draw; frozen dataset stream; 10000 resamples"
        ),
        worst_stage=int(cell["target_worst_stage_zero_based"]),
        mean_coverage=float(cell["target_mean_coverage"]),
        mean_coverage_interval=tuple(
            float(value) for value in cell["target_mean_coverage_ci95"]
        ),
        mean_width=float(cell["mean_target_normalized_width"]),
        mean_width_interval=tuple(
            float(value) for value in cell["mean_target_normalized_width_ci95"]
        ),
        point_eligible=(cell["point_eligible"] if confirmatory else None),
        source_path=source_path,
        source_sha256=source_sha256,
    )


def _status_row(**values: Any) -> dict[str, Any]:
    row = {"schema_version": 1, "method_count": len(METHODS), **values}
    if set(row) != set(STATUS_COLUMNS):
        raise RuntimeError("setting-status row schema differs")
    return row


def _stage_row(
    *,
    reporting_family: str,
    setting_id: str,
    display_label: str,
    dataset: str,
    setting_type: str,
    feedback_parameter: str,
    feedback_value: float | None,
    analysis_role: str,
    panel_status: str,
    confirmatory: bool,
    method: str,
    evaluation_budget: int,
    calibration_budget: int | None,
    grid_budget: int | None,
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
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    row = {
        "schema_version": 1,
        "reporting_family": reporting_family,
        "setting_id": setting_id,
        "display_label": display_label,
        "dataset": dataset,
        "setting_type": setting_type,
        "feedback_parameter": feedback_parameter,
        "feedback_value": feedback_value,
        "analysis_role": analysis_role,
        "panel_status": panel_status,
        "confirmatory": confirmatory,
        "method": method,
        "information_regime": INFORMATION_REGIME[method],
        "target_adaptation_trajectories_per_seed": TARGET_ADAPTATION_BUDGET[method],
        "evaluation_trajectories_per_seed": evaluation_budget,
        "calibration_trajectories_per_seed": calibration_budget,
        "grid_trajectories_per_seed": grid_budget,
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
        "normalized_width_definition": NORMALIZED_WIDTH_DEFINITION,
        "source_path": source_path,
        "source_sha256": source_sha256,
    }
    if set(row) != set(STAGE_COLUMNS):
        raise RuntimeError("stage-profile row schema differs")
    return row


def _scalar_row(
    *,
    reporting_family: str,
    setting_id: str,
    display_label: str,
    dataset: str,
    setting_type: str,
    feedback_parameter: str,
    feedback_value: float | None,
    analysis_role: str,
    panel_status: str,
    confirmatory: bool,
    method: str,
    evaluation_budget: int,
    calibration_budget: int | None,
    grid_budget: int | None,
    n_prespecified: int,
    n_gate_eligible: int,
    n_selected: int,
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
    point_eligible: bool | None,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    row = {
        "schema_version": 1,
        "reporting_family": reporting_family,
        "setting_id": setting_id,
        "display_label": display_label,
        "dataset": dataset,
        "setting_type": setting_type,
        "feedback_parameter": feedback_parameter,
        "feedback_value": feedback_value,
        "analysis_role": analysis_role,
        "panel_status": panel_status,
        "confirmatory": confirmatory,
        "method": method,
        "information_regime": INFORMATION_REGIME[method],
        "target_adaptation_trajectories_per_seed": TARGET_ADAPTATION_BUDGET[method],
        "evaluation_trajectories_per_seed": evaluation_budget,
        "calibration_trajectories_per_seed": calibration_budget,
        "grid_trajectories_per_seed": grid_budget,
        "n_prespecified": n_prespecified,
        "n_gate_eligible": n_gate_eligible,
        "n_selected": n_selected,
        "metric_available": metric_available,
        "coverage_conditioning": coverage_conditioning,
        "selection_rate": selection_rate,
        "selection_rate_ci95_lower": selection_interval[0],
        "selection_rate_ci95_upper": selection_interval[1],
        "selection_interval_definition": "two-sided Wilson 95% interval over all prespecified seeds",
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
            None
            if mean_coverage is None
            else 100.0 * (mean_coverage - TARGET)
        ),
        "mean_coverage_interval_definition": "two-sided 95% Student-t interval across selected seed-level means",
        "mean_normalized_width": mean_width,
        "mean_normalized_width_ci95_lower": mean_width_interval[0],
        "mean_normalized_width_ci95_upper": mean_width_interval[1],
        "mean_width_interval_definition": "two-sided 95% Student-t interval across selected seed-level means",
        "normalized_width_definition": NORMALIZED_WIDTH_DEFINITION,
        "point_eligibility_rule": POINT_ELIGIBILITY_RULE,
        "point_eligible": point_eligible,
        "efficiency_rank_defined": False,
        "narrowest_point_eligible": None,
        "source_path": source_path,
        "source_sha256": source_sha256,
    }
    if set(row) != set(SCALAR_COLUMNS):
        raise RuntimeError("scalar-summary row schema differs")
    return row


def _assign_efficiency_ranking(scalar: pd.DataFrame) -> pd.DataFrame:
    """Mark width efficiency only inside valid point-eligible groups."""

    ranked = scalar.copy()
    for _, group in ranked.groupby(["reporting_family", "setting_id"], sort=False):
        confirmatory = group["confirmatory"].map(_as_bool).all()
        available = group["metric_available"].map(_as_bool).all()
        if not confirmatory or not available:
            continue
        eligible = group[group["point_eligible"].map(_as_bool)]
        if eligible.empty:
            continue
        winner = eligible["mean_normalized_width"].astype(float).idxmin()
        ranked.loc[group.index, "efficiency_rank_defined"] = True
        ranked.loc[group.index, "narrowest_point_eligible"] = False
        ranked.loc[winner, "narrowest_point_eligible"] = True
    return ranked.loc[:, SCALAR_COLUMNS]


def _unavailable_scalar_row(
    *,
    reporting_family: str,
    setting_id: str,
    display_label: str,
    dataset: str,
    feedback_value: float,
    panel_status: str,
    analysis_role: str,
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
        setting_type="dataset_native_clinical_controlled",
        feedback_parameter="gamma",
        feedback_value=feedback_value,
        analysis_role=analysis_role,
        panel_status=panel_status,
        confirmatory=False,
        method=method,
        evaluation_budget=CONTROLLED_EVALUATION_BUDGET,
        calibration_budget=CONTROLLED_CALIBRATION_BUDGET,
        grid_budget=CONTROLLED_GRID_BUDGET,
        n_prespecified=n_prespecified,
        n_gate_eligible=(0 if n_gate_eligible is None else n_gate_eligible),
        n_selected=0,
        metric_available=False,
        coverage_conditioning="not available: hard preflight NO-GO",
        selection_rate=None,
        selection_interval=(None, None),
        wsc=None,
        wsc_interval=(None, None),
        wsc_interval_definition="not available: hard preflight NO-GO",
        worst_stage=None,
        mean_coverage=None,
        mean_coverage_interval=(None, None),
        mean_width=None,
        mean_width_interval=(None, None),
        point_eligible=None,
        source_path=source_path,
        source_sha256=source_sha256,
    )


def validate_reporting_sources(
    status: pd.DataFrame, stage: pd.DataFrame, scalar: pd.DataFrame
) -> None:
    if tuple(status.columns) != STATUS_COLUMNS:
        raise RuntimeError("setting-status schema differs")
    if tuple(stage.columns) != STAGE_COLUMNS:
        raise RuntimeError("stage-profile schema differs")
    if tuple(scalar.columns) != SCALAR_COLUMNS:
        raise RuntimeError("scalar-summary schema differs")
    if status.duplicated(["reporting_family", "setting_id"]).any():
        raise RuntimeError("setting statuses are duplicated")
    if stage.duplicated(
        ["reporting_family", "setting_id", "method", "stage_zero_based"]
    ).any():
        raise RuntimeError("stage-profile keys are duplicated")
    if scalar.duplicated(["reporting_family", "setting_id", "method"]).any():
        raise RuntimeError("scalar-summary keys are duplicated")
    if set(stage["method"]) != set(METHODS) or set(scalar["method"]) != set(METHODS):
        raise RuntimeError("canonical method set differs")
    if not np.array_equal(
        scalar["coverage_target"].to_numpy(float), np.full(len(scalar), TARGET)
    ) or not np.array_equal(
        stage["coverage_target"].to_numpy(float), np.full(len(stage), TARGET)
    ):
        raise RuntimeError("coverage target differs from 0.90")
    if set(stage["normalized_width_definition"]) != {NORMALIZED_WIDTH_DEFINITION}:
        raise RuntimeError("stage normalized-width definition differs")
    if set(scalar["normalized_width_definition"]) != {NORMALIZED_WIDTH_DEFINITION}:
        raise RuntimeError("scalar normalized-width definition differs")
    if set(scalar["primary_metric"]) != {PRIMARY_METRIC}:
        raise RuntimeError("primary WSC metric differs")

    _validate_family_identity(status, stage, scalar)
    _validate_stage_values(stage)
    _validate_scalar_values(status, stage, scalar)
    _validate_information_budgets(stage, scalar)


def _validate_family_identity(
    status: pd.DataFrame, stage: pd.DataFrame, scalar: pd.DataFrame
) -> None:
    production_status = status[status["reporting_family"].eq("production_no_gamma_supplement")]
    if tuple(production_status["dataset"]) != PRODUCTION_DATASETS:
        raise RuntimeError("production setting order differs")
    if len(production_status) != 5 or not production_status["scientific_rows_available"].map(_as_bool).all():
        raise RuntimeError("production status grid differs")
    if production_status["confirmatory"].map(_as_bool).any():
        raise RuntimeError("production/no-gamma supplement cannot be confirmatory")

    main_status = status[status["reporting_family"].eq("clinical_gamma_minus4_main")]
    if tuple(main_status["dataset"]) != CLINICAL_DATASETS:
        raise RuntimeError("clinical gamma=-4 setting order differs")
    if (
        set(main_status["feedback_parameter"]) != {"gamma"}
        or set(main_status["feedback_value"].astype(float)) != {-4.0}
    ):
        raise RuntimeError("main clinical setting is not exactly gamma=-4")
    if (main_status["dataset"] == "synthetic").any():
        raise RuntimeError("Synthetic beta cannot enter the clinical gamma figure")

    signed_status = status[
        status["reporting_family"].eq("mimic_iv_v2_signed_gamma_supplement")
    ]
    if (
        tuple(signed_status["dataset"]) != ("mimic_iv",) * len(SIGNED_GAMMAS)
        or tuple(signed_status["feedback_value"].astype(float)) != SIGNED_GAMMAS
        or signed_status["confirmatory"].map(_as_bool).tolist()
        != [True, False, False, False, False]
    ):
        raise RuntimeError("MIMIC-IV signed-gamma identity differs")

    expected_scalar_keys = {
        (row.reporting_family, row.setting_id, method)
        for row in status.itertuples(index=False)
        for method in METHODS
    }
    if set(zip(scalar["reporting_family"], scalar["setting_id"], scalar["method"])) != expected_scalar_keys:
        raise RuntimeError("scalar setting-method grid differs")

    for status_row in status.itertuples(index=False):
        rows = stage[
            stage["reporting_family"].eq(status_row.reporting_family)
            & stage["setting_id"].eq(status_row.setting_id)
        ]
        if not _as_bool(status_row.scientific_rows_available):
            if not rows.empty:
                raise RuntimeError("hard-gate status cannot have scientific stage rows")
            continue
        expected = {
            (method, stage_index)
            for method in METHODS
            for stage_index in range(int(status_row.horizon))
        }
        if set(zip(rows["method"], rows["stage_zero_based"])) != expected:
            raise RuntimeError(f"stage grid differs for {status_row.setting_id}")
        for field in (
            "display_label",
            "dataset",
            "setting_type",
            "feedback_parameter",
            "feedback_value",
            "panel_status",
            "confirmatory",
            "horizon",
        ):
            values = rows[field].drop_duplicates()
            expected_value = getattr(status_row, field)
            if len(values) != 1 or not _same_value(values.iloc[0], expected_value):
                raise RuntimeError(f"status/stage join differs for {status_row.setting_id}/{field}")


def _validate_stage_values(stage: pd.DataFrame) -> None:
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
        raise RuntimeError("stage coverage/interval values differ")
    if (
        not np.isfinite(width).all()
        or np.any(width <= 0.0)
        or np.any(width[:, 0] > width[:, 1])
        or np.any(width[:, 1] > width[:, 2])
    ):
        raise RuntimeError("stage width/interval values differ")
    if set(stage["coverage_interval_scope"]) != {"pointwise, not simultaneous"}:
        raise RuntimeError("stage coverage interval scope differs")
    if set(stage["normalized_width_interval_scope"]) != {"pointwise, not simultaneous"}:
        raise RuntimeError("stage width interval scope differs")


def _validate_scalar_values(
    status: pd.DataFrame, stage: pd.DataFrame, scalar: pd.DataFrame
) -> None:
    status_index = status.set_index(["reporting_family", "setting_id"])
    numeric_fields = (
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
    )
    for row in scalar.itertuples(index=False):
        setting = status_index.loc[(row.reporting_family, row.setting_id)]
        available = _as_bool(row.metric_available)
        stages = stage[
            stage["reporting_family"].eq(row.reporting_family)
            & stage["setting_id"].eq(row.setting_id)
            & stage["method"].eq(row.method)
        ].sort_values("stage_zero_based")
        if not available:
            if _as_bool(setting["scientific_rows_available"]):
                raise RuntimeError("an unavailable metric row belongs to a curve setting")
            if not stages.empty or any(not pd.isna(getattr(row, field)) for field in numeric_fields):
                raise RuntimeError("hard-gate scalar row contains scientific values")
            continue
        if not _as_bool(setting["scientific_rows_available"]) or stages.empty:
            raise RuntimeError("available scalar row lacks its scientific stages")
        values = np.asarray(
            [
                row.selection_rate_ci95_lower,
                row.selection_rate,
                row.selection_rate_ci95_upper,
                row.wsc_ci95_lower,
                row.wsc,
                row.wsc_ci95_upper,
                row.mean_coverage_ci95_lower,
                row.mean_coverage,
                row.mean_coverage_ci95_upper,
                row.mean_normalized_width_ci95_lower,
                row.mean_normalized_width,
                row.mean_normalized_width_ci95_upper,
            ],
            dtype=float,
        ).reshape(4, 3)
        if (
            not np.isfinite(values).all()
            or np.any(values[:, 0] > values[:, 1])
            or np.any(values[:, 1] > values[:, 2])
            or np.any((values[:3] < 0.0) | (values[:3] > 1.0))
            or np.any(values[3] <= 0.0)
        ):
            raise RuntimeError("scalar values/intervals differ")
        stage_coverage = stages["coverage_mean"].to_numpy(float)
        stage_width = stages["normalized_width_mean"].to_numpy(float)
        if not math.isclose(float(row.wsc), float(stage_coverage.min()), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("WSC differs from min_t mean_seed(C_seed,t)")
        if int(row.worst_stage_zero_based) != int(stage_coverage.argmin()):
            raise RuntimeError("worst stage differs from the first stage-mean argmin")
        mean_tolerance = 5e-7 if row.reporting_family == "production_no_gamma_supplement" else 1e-12
        if not math.isclose(
            float(row.mean_coverage), float(stage_coverage.mean()), rel_tol=0.0, abs_tol=mean_tolerance
        ):
            raise RuntimeError("MeanCov differs from the mean of stage coverage")
        if not math.isclose(
            float(row.mean_normalized_width), float(stage_width.mean()), rel_tol=0.0, abs_tol=5e-7
        ):
            raise RuntimeError("mean width differs from the mean of stage widths")
        if not math.isclose(
            float(row.selection_rate), int(row.n_selected) / int(row.n_prespecified), rel_tol=0.0, abs_tol=1e-14
        ):
            raise RuntimeError("selection rate differs from selected/prespecified")
        if "complete-seed-vector" not in str(row.wsc_interval_definition):
            raise RuntimeError("WSC CI is not identified as a complete-seed-vector interval")
        if not math.isclose(
            float(row.wsc_deviation_from_target_pp),
            100.0 * (float(row.wsc) - TARGET),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("WSC deviation from target differs")
        if not math.isclose(
            float(row.mean_coverage_deviation_from_target_pp),
            100.0 * (float(row.mean_coverage) - TARGET),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("MeanCov deviation from target differs")

    for _, group in scalar.groupby(["reporting_family", "setting_id"], sort=False):
        rank_defined = group["efficiency_rank_defined"].map(_as_bool)
        if rank_defined.any():
            if not rank_defined.all():
                raise RuntimeError("efficiency-ranking status differs within a setting")
            winner = group[group["narrowest_point_eligible"].map(_as_bool)]
            if len(winner) != 1 or not _as_bool(winner.iloc[0]["point_eligible"]):
                raise RuntimeError("eligible-width winner differs")
            eligible = group[group["point_eligible"].map(_as_bool)]
            if float(winner.iloc[0]["mean_normalized_width"]) != float(
                eligible["mean_normalized_width"].min()
            ):
                raise RuntimeError("width ranking is not restricted to point-eligible methods")
        elif group["narrowest_point_eligible"].notna().any():
            raise RuntimeError("undefined efficiency ranking contains a winner")


def _validate_information_budgets(stage: pd.DataFrame, scalar: pd.DataFrame) -> None:
    for frame in (stage, scalar):
        if not frame["method"].map(INFORMATION_REGIME).eq(frame["information_regime"]).all():
            raise RuntimeError("information regime differs")
        if not frame["method"].map(TARGET_ADAPTATION_BUDGET).eq(
            frame["target_adaptation_trajectories_per_seed"]
        ).all():
            raise RuntimeError("target-adaptation budget differs")
        expected_evaluation = np.where(
            frame["reporting_family"].eq("production_no_gamma_supplement"),
            PRODUCTION_EVALUATION_BUDGET,
            CONTROLLED_EVALUATION_BUDGET,
        )
        if not np.array_equal(
            frame["evaluation_trajectories_per_seed"].to_numpy(int), expected_evaluation
        ):
            raise RuntimeError("evaluation budget differs")


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 6.1,
            "axes.labelsize": 6.2,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.62,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "svg.hashsalt": "scpcp-complete-coverage-reporting-v1",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def render_gamma_minus4_stagewise(
    status: pd.DataFrame, stage: pd.DataFrame
) -> plt.Figure:
    family = "clinical_gamma_minus4_main"
    settings = status[status["reporting_family"].eq(family)]
    profiles = stage[stage["reporting_family"].eq(family)]
    figure, axes = plt.subplots(2, 4, figsize=(7.20, 3.90), sharex="col")
    figure.subplots_adjust(left=0.075, right=0.992, bottom=0.165, top=0.755, wspace=0.34, hspace=0.27)
    coverage_limits = _metric_limits(
        100.0 * (profiles["coverage_ci95_lower"].to_numpy(float) - TARGET),
        100.0 * (profiles["coverage_ci95_upper"].to_numpy(float) - TARGET),
        fallback=(-5.0, 5.0),
        quantum=0.5,
    )
    legend_handles = [_legend_handle(method) for method in METHODS]
    for column, dataset in enumerate(CLINICAL_DATASETS):
        setting = settings[settings["dataset"].eq(dataset)].iloc[0]
        coverage_axis, width_axis = axes[:, column]
        group = profiles[profiles["setting_id"].eq(setting["setting_id"])]
        if setting["panel_status"] == "GATE_NO_GO":
            _render_gate_card(coverage_axis, width_axis, setting)
            continue
        coverage_axis.axhspan(coverage_limits[0], 0.0, color="#F8E9E7", zorder=0)
        coverage_axis.axhline(0.0, color="#30353A", linestyle=(0, (3, 2)), linewidth=0.72)
        for method in METHODS:
            rows = group[group["method"].eq(method)].sort_values("stage_zero_based")
            _plot_profile_errorbars(coverage_axis, rows, method, metric="coverage")
            _plot_profile_errorbars(width_axis, rows, method, metric="width")
        coverage_axis.set_title(f"{DATASET_LABELS[dataset]}\nclinical γ=−4", fontweight="bold", pad=3)
        coverage_axis.set_ylim(*coverage_limits)
        _set_width_limits(width_axis, group)
        _set_stage_axis(width_axis, HORIZONS[dataset])
        for axis in (coverage_axis, width_axis):
            axis.grid(axis="y", color="#E1E3E6", linewidth=0.42, zorder=-5)
            axis.tick_params(width=0.58, length=2.1)
        if column == 0:
            coverage_axis.set_ylabel("Coverage deviation (pp)")
            width_axis.set_ylabel("Normalized width")
        width_axis.set_xlabel("Stage, t (0-based)")

    figure.suptitle(
        "Dataset-native clinical controlled endpoint: complete stagewise reporting at γ=−4",
        x=0.01,
        y=0.99,
        ha="left",
        fontsize=8.2,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.915,
        "A hard K0/support gate failure means no admissible scientific row; it is not a method-performance result.",
        ha="left",
        va="top",
        fontsize=5.65,
        color="#42474D",
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
        ncol=6,
        columnspacing=0.75,
        handlelength=1.7,
        handletextpad=0.3,
    )
    figure.text(
        0.5,
        0.018,
        "Capped error bars show every method's pointwise 95% interval; they are not simultaneous bands. "
        "WSC remains min_t mean_seed(C_seed,t). Controlled stress is calibration-aligned, not a natural or causal treatment effect.",
        ha="center",
        va="bottom",
        fontsize=5.05,
        color="#454A50",
    )
    return figure


def _plot_profile_errorbars(
    axis: plt.Axes, rows: pd.DataFrame, method: str, *, metric: str
) -> ErrorbarContainer:
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
        raise ValueError(f"unknown profile metric: {metric}")
    emphasized = method in {"Standard CP", "MFCS", "SC-PCP"}
    return axis.errorbar(
        x,
        point,
        yerr=np.vstack((point - lower, upper - point)),
        color=METHOD_COLORS[method],
        linestyle=METHOD_LINESTYLES[method],
        marker=METHOD_MARKERS[method],
        markersize=3.0 if method == "SC-PCP" else 2.25,
        linewidth=1.15 if method == "SC-PCP" else 0.86 if emphasized else 0.70,
        elinewidth=0.52,
        capsize=1.15,
        capthick=0.48,
        markeredgewidth=0.25,
        alpha=1.0,
        zorder=4 if method == "SC-PCP" else 3 if emphasized else 2,
    )


def _render_gate_card(
    coverage_axis: plt.Axes, width_axis: plt.Axes, setting: pd.Series
) -> None:
    for axis in (coverage_axis, width_axis):
        axis.set_axis_off()
        axis.set_facecolor("#F4F2EF")
    coverage_axis.set_title(
        f"{setting['display_label']}\nclinical γ=−4", fontweight="bold", pad=3
    )
    coverage_axis.text(
        0.5,
        0.44,
        "HARD GATE\nNO-GO",
        transform=coverage_axis.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="#8A3B31",
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": "#FAECE8",
            "edgecolor": "#C58B82",
            "linewidth": 0.8,
        },
    )
    width_axis.text(
        0.5,
        0.61,
        f"{str(setting['hard_gate_reason']).replace('_', ' ')}\n"
        f"{int(setting['k0_fidelity_available'])}/{int(setting['n_prespecified'])} seeds passed\n\n"
        "No science rows\nNo curve or ranking",
        transform=width_axis.transAxes,
        ha="center",
        va="center",
        fontsize=5.65,
        linespacing=1.32,
        color="#4B4642",
    )


def render_gamma_minus4_table(
    status: pd.DataFrame, scalar: pd.DataFrame
) -> plt.Figure:
    family = "clinical_gamma_minus4_main"
    rows = scalar[scalar["reporting_family"].eq(family)].copy()
    statuses = status[status["reporting_family"].eq(family)].set_index("dataset")
    return _render_complete_table(
        rows,
        title="Clinical controlled endpoint at γ=−4: complete coverage reporting",
        subtitle="WSC is primary; unavailable cells follow formal preflight NO-GO decisions.",
        group_field="dataset",
        group_order=CLINICAL_DATASETS,
        group_labels=DATASET_LABELS,
        height=7.25,
        footer=(
            "WSC=min_t mean_seed(C_seed,t); WSC CI uses the stored 10,000-draw complete-seed-vector bootstrap. "
            "MeanCov and width use Student-t intervals; selection uses Wilson intervals over all 20 prespecified seeds.\n"
            "MIMIC-IV SC-PCP point WSC=90.09% [89.43%, 90.10%]: point-eligible under the frozen point rule, "
            "but its CI crosses 90% and is not a statistical or finite-sample certificate. Width is ranked only "
            "among point-eligible methods.\n"
            + " | ".join(
                f"{DATASET_LABELS[dataset]}: "
                + (
                    "CURVES"
                    if statuses.loc[dataset, "panel_status"] != "GATE_NO_GO"
                    else f"{statuses.loc[dataset, 'hard_gate_reason']} "
                    f"({int(statuses.loc[dataset, 'k0_fidelity_available'])}/20 K0 seeds)"
                )
                for dataset in CLINICAL_DATASETS
            )
        ),
    )


def render_signed_gamma_figure(stage: pd.DataFrame, scalar: pd.DataFrame) -> plt.Figure:
    del stage  # scalar signed curves are the declared supplement; stage rows remain in source data.
    rows = scalar[
        scalar["reporting_family"].eq("mimic_iv_v2_signed_gamma_supplement")
    ]
    figure, axes = plt.subplots(3, 1, figsize=(7.20, 6.35), sharex=True)
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.115, top=0.835, hspace=0.24)
    offsets = dict(zip(METHODS, np.linspace(-0.085, 0.085, len(METHODS))))
    metric_contract = (
        ("wsc", "wsc_ci95_lower", "wsc_ci95_upper", "Marginal WSC (%)", 100.0),
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
            "Mean normalized width",
            1.0,
        ),
    )
    handles: list[Line2D] = []
    for axis, (point_name, lower_name, upper_name, ylabel, scale) in zip(
        axes, metric_contract
    ):
        for method in METHODS:
            selected = rows[rows["method"].eq(method)].sort_values("feedback_value")
            x = selected["feedback_value"].to_numpy(float) + offsets[method]
            point = scale * selected[point_name].to_numpy(float)
            lower = scale * selected[lower_name].to_numpy(float)
            upper = scale * selected[upper_name].to_numpy(float)
            axis.errorbar(
                x,
                point,
                yerr=np.vstack((point - lower, upper - point)),
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                marker=METHOD_MARKERS[method],
                markersize=3.2,
                linewidth=1.1 if method == "SC-PCP" else 0.82,
                elinewidth=0.55,
                capsize=1.35,
                markeredgewidth=0.25,
            )
            if axis is axes[0]:
                handles.append(_legend_handle(method))
        axis.axvspan(-4.25, -3.75, color="#E8B84A", alpha=0.12, zorder=-5)
        axis.grid(axis="y", color="#E1E3E6", linewidth=0.45, zorder=-6)
        axis.set_ylabel(ylabel)
    axes[0].axhline(90.0, color="#30353A", linestyle=(0, (3, 2)), linewidth=0.75)
    axes[1].axhline(90.0, color="#30353A", linestyle=(0, (3, 2)), linewidth=0.75)
    axes[-1].set_xticks(SIGNED_GAMMAS, [_format_gamma(value) for value in SIGNED_GAMMAS])
    axes[-1].set_xlabel("Signed transition alignment, γ")
    axes[0].text(
        -4.0,
        0.97,
        "confirmatory",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=5.3,
        color="#775A00",
    )
    figure.suptitle(
        "MIMIC-IV controlled v2: complete signed-gamma coverage and efficiency",
        x=0.01,
        y=0.99,
        ha="left",
        fontsize=8.4,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.93,
        "γ=−4 is the confirmatory endpoint; all other signed cells are descriptive controls and carry no ranking marker.",
        ha="left",
        va="top",
        fontsize=5.7,
        color="#42474D",
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=6,
        columnspacing=0.9,
        handlelength=1.8,
    )
    figure.text(
        0.5,
        0.018,
        "Intervals are pointwise across signed cells, not a simultaneous curve certificate. "
        "WSC remains the primary coverage metric; full stagewise values and intervals are in the source CSV.",
        ha="center",
        va="bottom",
        fontsize=5.15,
        color="#454A50",
    )
    return figure


def render_signed_gamma_table(scalar: pd.DataFrame) -> plt.Figure:
    rows = scalar[
        scalar["reporting_family"].eq("mimic_iv_v2_signed_gamma_supplement")
    ].copy()
    labels = {gamma: _format_gamma(gamma) for gamma in SIGNED_GAMMAS}
    return _render_complete_table(
        rows,
        title="MIMIC-IV controlled v2: full signed-gamma metrics",
        subtitle="Only γ=−4 is confirmatory; other cells are descriptive and unranked.",
        group_field="feedback_value",
        group_order=SIGNED_GAMMAS,
        group_labels=labels,
        height=8.75,
        footer=(
            "WSC=min_t mean_seed(C_seed,t); t* is the first zero-based stage attaining the minimum. "
            "WSC intervals use complete-seed-vector bootstrap draws; MeanCov/width use Student-t intervals.\n"
            "All methods use 20,000 fresh evaluation trajectories per seed and gamma. "
            "Adaptation budget is 2,000 for ACI/SPCI/PRC and 0 for Standard CP/MFCS/SC-PCP. "
            "Any width highlight is restricted to point-eligible methods in the confirmatory γ=−4 cell."
        ),
    )


def render_production_table(scalar: pd.DataFrame) -> plt.Figure:
    rows = scalar[
        scalar["reporting_family"].eq("production_no_gamma_supplement")
    ].copy()
    return _render_complete_table(
        rows,
        title="Frozen production/native RQ1: complete scalar coverage reporting",
        subtitle="Robustness supplement without controlled γ; it is not the default signed-treatment endpoint.",
        group_field="dataset",
        group_order=PRODUCTION_DATASETS,
        group_labels=DATASET_LABELS,
        height=8.75,
        footer=(
            "WSC=min_t mean_seed(C_seed,t); t* is zero-based. WSC intervals use 10,000 complete-seed-vector "
            "bootstrap draws; MeanCov/width use Student-t intervals; selection uses Wilson intervals.\n"
            "All methods use 50,000 fresh evaluation trajectories per seed. Adaptation budget is 2,000 for "
            "ACI/SPCI/PRC and 0 for Standard CP/MFCS/SC-PCP. No width ranking is made in this robustness table."
        ),
    )


def _render_complete_table(
    rows: pd.DataFrame,
    *,
    title: str,
    subtitle: str,
    group_field: str,
    group_order: Sequence[object],
    group_labels: Mapping[object, str],
    height: float,
    footer: str,
) -> plt.Figure:
    method_rank = {method: index for index, method in enumerate(METHODS)}
    group_rank = {value: index for index, value in enumerate(group_order)}
    rows = rows.assign(
        _group_rank=rows[group_field].map(group_rank),
        _method_rank=rows["method"].map(method_rank),
    ).sort_values(["_group_rank", "_method_rank"])
    table_rows: list[list[str]] = []
    group_indices: list[int] = []
    unavailable_indices: set[int] = set()
    for row_index, row in enumerate(rows.itertuples(index=False)):
        group_value = getattr(row, group_field)
        group_text = str(group_labels[group_value]) if row.method == METHODS[0] else ""
        available = _as_bool(row.metric_available)
        if available:
            values = [
                group_text,
                row.method,
                _format_percent_interval(row.wsc, row.wsc_ci95_lower, row.wsc_ci95_upper),
                str(int(row.worst_stage_zero_based)),
                _format_percent_interval(
                    row.mean_coverage,
                    row.mean_coverage_ci95_lower,
                    row.mean_coverage_ci95_upper,
                ),
                _format_number_interval(
                    row.mean_normalized_width,
                    row.mean_normalized_width_ci95_lower,
                    row.mean_normalized_width_ci95_upper,
                ),
                _format_selection(row),
                _format_budget(row),
            ]
        else:
            values = [group_text, row.method, "--", "--", "--", "--", "not opened", _format_budget(row)]
            unavailable_indices.add(row_index)
        table_rows.append(values)
        group_indices.append(group_rank[group_value])

    figure, axis = plt.subplots(figsize=(7.20, height))
    figure.subplots_adjust(left=0.01, right=0.99, top=0.985, bottom=0.01)
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=8.4, fontweight="bold", pad=12)
    axis.text(
        0.0,
        0.975,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color="#444444",
    )
    table = axis.table(
        cellText=table_rows,
        colLabels=(
            "Setting",
            "Method",
            "WSC [95% CI]\nand delta vs 90%",
            "t*",
            "MeanCov [95% CI]\nand delta vs 90%",
            "Norm. width [95% CI]",
            "Selection [Wilson 95% CI]",
            "Adapt./eval.",
        ),
        colLoc="center",
        cellLoc="center",
        colWidths=(0.13, 0.12, 0.16, 0.045, 0.16, 0.17, 0.15, 0.065),
        bbox=(0.0, 0.11, 1.0, 0.825),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(4.75 if len(table_rows) >= 30 else 5.15)
    for column in range(8):
        header = table[(0, column)]
        header.set_facecolor("#334E68")
        header.set_text_props(color="white", fontweight="bold")
        header.set_edgecolor("white")
        header.set_linewidth(0.5)
    group_colors = ("#F4F7FA", "#FFFFFF")
    for row_index, group_index in enumerate(group_indices, start=1):
        for column in range(8):
            cell = table[(row_index, column)]
            cell.set_facecolor("#F4F2EF" if row_index - 1 in unavailable_indices else group_colors[group_index % 2])
            cell.set_edgecolor("#D7DEE5")
            cell.set_linewidth(0.35)
            if column == 1:
                cell.set_text_props(ha="left")
        if table_rows[row_index - 1][1] == "SC-PCP":
            table[(row_index, 1)].get_text().set_color(METHOD_COLORS["SC-PCP"])
        source_row = rows.iloc[row_index - 1]
        if (
            not pd.isna(source_row["narrowest_point_eligible"])
            and _as_bool(source_row["narrowest_point_eligible"])
        ):
            table[(row_index, 5)].set_facecolor("#DDECF8")
            table[(row_index, 5)].set_text_props(fontweight="bold")
        if (row_index - 1) % len(METHODS) == 0:
            for column in range(8):
                table[(row_index, column)].set_linewidth(0.75)
                table[(row_index, column)].set_edgecolor("#9AA9B5")
    axis.text(
        0.0,
        0.075,
        footer,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.15,
        linespacing=1.35,
    )
    axis.text(
        0.0,
        0.018,
        "Coverage/width condition on successful selection; selection uses all prespecified seeds. "
        "Intervals quantify uncertainty and are not finite-sample certificates.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.0,
        color="#454A50",
    )
    return figure


def _format_percent_interval(point: float, lower: float, upper: float) -> str:
    deviation = 100.0 * (float(point) - TARGET)
    return (
        f"{100.0 * float(point):.2f} [{100.0 * float(lower):.2f}, "
        f"{100.0 * float(upper):.2f}]\ndelta={deviation:+.2f} pp"
    )


def _format_number_interval(point: float, lower: float, upper: float) -> str:
    return f"{float(point):.3f} [{float(lower):.3f}, {float(upper):.3f}]"


def _format_selection(row: object) -> str:
    return (
        f"{int(getattr(row, 'n_selected'))}/{int(getattr(row, 'n_prespecified'))} "
        f"({100.0 * float(getattr(row, 'selection_rate')):.1f}% "
        f"[{100.0 * float(getattr(row, 'selection_rate_ci95_lower')):.1f}, "
        f"{100.0 * float(getattr(row, 'selection_rate_ci95_upper')):.1f}])"
    )


def _format_budget(row: object) -> str:
    adaptation = int(getattr(row, "target_adaptation_trajectories_per_seed"))
    evaluation = int(getattr(row, "evaluation_trajectories_per_seed"))
    adaptation_text = "0" if adaptation == 0 else f"{adaptation // 1000}k"
    return f"{adaptation_text}/{evaluation // 1000}k"


def _legend_handle(method: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=METHOD_COLORS[method],
        linestyle=METHOD_LINESTYLES[method],
        marker=METHOD_MARKERS[method],
        markersize=3.1,
        linewidth=1.1 if method == "SC-PCP" else 0.82,
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


def export_figure(
    figure: plt.Figure,
    *,
    work_stem: Path,
    title: str,
    tiff_dpi: int = 600,
    png_dpi: int = 240,
) -> None:
    creator = "SC-PCP complete coverage reporting renderer"
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


def _write_contract(
    path: Path,
    *,
    status: pd.DataFrame,
    stage: pd.DataFrame,
    scalar: pd.DataFrame,
    input_contract: Mapping[str, Any],
) -> None:
    main_status = status[status["reporting_family"].eq("clinical_gamma_minus4_main")]
    payload = {
        "schema_version": 1,
        "protocol": RENDER_PROTOCOL,
        "status": "complete",
        "backend": "Python/matplotlib only",
        "scientific_rng_used": False,
        "core_conclusion": (
            "Complete coverage reporting keeps WSC primary while exposing every "
            "available stagewise coverage interval, MeanCov, worst stage, selection, "
            "and normalized-width result without replacing formal hard-gate failures."
        ),
        "figures": {
            MAIN_STAGE_STEM: {
                "archetype": "clinical quantitative grid",
                "panel_map": {
                    "columns": list(CLINICAL_DATASETS),
                    "top": "stagewise coverage deviation from 0.90 with every method's pointwise CI",
                    "bottom": "stagewise normalized width with every method's pointwise CI",
                },
                "feedback_parameter": "gamma",
                "feedback_value": -4.0,
                "synthetic_included": False,
                "setting_status": main_status.to_dict(orient="records"),
            },
            SIGNED_FIGURE_STEM: {
                "archetype": "three-row quantitative signed curve",
                "dataset": "mimic_iv",
                "rows": ["WSC", "MeanCov", "mean normalized width"],
                "gammas": list(SIGNED_GAMMAS),
                "confirmatory_gamma": -4.0,
                "ranking_markers": False,
            },
        },
        "tables": {
            MAIN_TABLE_STEM: "complete gamma=-4 clinical scalar metrics and hard-gate availability",
            SIGNED_TABLE_STEM: "complete MIMIC-IV v2 signed-gamma scalar metrics",
            PRODUCTION_TABLE_STEM: "complete frozen production/no-gamma robustness scalar metrics",
        },
        "metric_contract": {
            "coverage_target": TARGET,
            "primary_metric": PRIMARY_METRIC,
            "stagewise_interval_scope": "pointwise, not simultaneous",
            "wsc_ci_rule": "stored complete-seed-vector bootstrap interval only",
            "mean_coverage_ci_rule": "stored Student-t interval across selected seed-level means",
            "selection_ci_rule": "stored Wilson interval over all prespecified seeds",
            "reported_target_deviations": ["WSC minus 0.90", "MeanCov minus 0.90"],
            "normalized_width_definition": NORMALIZED_WIDTH_DEFINITION,
        },
        "source_data": {
            "status_rows": len(status),
            "stage_rows": len(stage),
            "scalar_rows": len(scalar),
            "files": sorted(SOURCE_FILES),
        },
        "input_contracts": input_contract,
        "clinical_protocol_adapter_rule": (
            "Only explicitly registered protocol-specific adapters are accepted; "
            "a future v3 must add its own exact validator and cannot fall through to v2 semantics."
        ),
        "reviewer_risks": [
            "A hard-gate NO-GO is absence of admissible science, not poor method coverage.",
            "Production/no-gamma results are robustness-only supplementary evidence and cannot substitute for controlled gamma cells.",
            "Synthetic beta is not signed clinical gamma and is excluded from the default gamma figure.",
            "Pointwise intervals are not simultaneous bands or finite-sample certificates.",
            "Point eligibility uses frozen point estimates; an interval crossing 0.90 is not statistical certification.",
            "Efficiency ranking is by normalized width only among point-eligible methods, never by raw overcoverage.",
            "The controlled kernel is calibration-aligned, not natural clinical performativity or a causal treatment effect.",
            "No universal dominance or universal-SOTA claim follows from these reports.",
        ],
        "export_contract": {
            "work_formats": ["editable SVG", "TrueType PDF", "600-dpi TIFF", "240-dpi PNG"],
            "paper_files": sorted(PAPER_FILES),
            "paper_directory_policy": "PDF only",
        },
    }
    _write_json(path, payload)


def _write_qa(
    path: Path,
    *,
    status: pd.DataFrame,
    stage: pd.DataFrame,
    scalar: pd.DataFrame,
) -> None:
    main = status[status["reporting_family"].eq("clinical_gamma_minus4_main")]
    curves = main[main["scientific_rows_available"].map(_as_bool)]
    gates = main[~main["scientific_rows_available"].map(_as_bool)]
    lines = [
        "# Complete coverage reporting QA",
        "",
        "- Backend exclusivity: Python/matplotlib produced every visual and QA preview.",
        "- Scientific RNG: none; all confidence intervals were copied from validated frozen artifacts.",
        f"- Source rows: status={len(status)}, stage={len(stage)}, scalar={len(scalar)}.",
        "- Main identity: four dataset-native clinical gamma=-4 columns; Synthetic beta is excluded.",
        "- Main horizons: MIMIC-IV=T12, eICU=T12, INSPIRE=T12, MIMIC-CXR + IV/ED=T6; no padding.",
        f"- Main curve settings: {', '.join(curves['dataset'])}.",
        f"- Main hard-gate settings: {', '.join(gates['dataset'])}.",
        "- All six canonical methods have explicit capped pointwise error bars wherever science exists; no translucent interval band hides a comparator.",
        "- Every hard-gate setting contributes zero stage rows and six unavailable scalar rows.",
        "- WSC is min_t mean_seed(C_seed,t); worst stage is the first zero-based argmin.",
        "- WSC-minus-0.90 and MeanCov-minus-0.90 are explicit scalar source/table fields.",
        "- WSC intervals are stored complete-seed-vector bootstrap intervals, never inferred from stagewise intervals.",
        "- MeanCov and mean width use stored Student-t intervals; selection uses stored Wilson intervals over all prespecified seeds.",
        "- Coverage and width condition on successful selection; selection denominators remain all prespecified seeds.",
        "- Production/no-gamma is robustness-only supplementary evidence; the signed supplement is MIMIC-IV controlled v2 only.",
        "- Signed gamma=-4 is confirmatory; gamma=-2,0,+2,+4 are descriptive and unranked.",
        "- Point eligibility is a frozen point-estimate rule; MIMIC-IV SC-PCP's WSC CI crosses 0.90 and is not a statistical certificate.",
        "- Width efficiency is ranked only among point-eligible methods; raw coverage excess is never the ranking objective.",
        "- Information budgets: production evaluation=50,000; controlled evaluation=20,000; ACI/SPCI/PRC adaptation=2,000; other methods=0.",
        "- Typography: Times New Roman with serif fallback; SVG text remains editable and PDF uses TrueType fonts.",
        "- Accessibility: method identity uses color, marker, and line-style redundancy; gate status uses text and shape, not color alone.",
        "- Paper output contains PDF files only; the work bundle contains source CSV, editable/vector/raster exports, contract, QA, and hashes.",
        "- Claim boundary: asymptotic per-step marginal target only; no finite-sample, distribution-free, PAC, data-conditional, causal, or universal-SOTA claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_render_manifest(path: Path, *, work_root: Path, paper_root: Path) -> None:
    if any(paper_root.iterdir()):
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
            "schema_version": 1,
            "protocol": RENDER_PROTOCOL,
            "status": "complete",
            "work_files": work_files,
            "paper_files": paper_files,
        },
    )


def _write_complete(work_root: Path) -> None:
    manifest_hash = _file_sha256(work_root / "render_manifest.json")
    (work_root / "COMPLETE").write_text(
        f"protocol={RENDER_PROTOCOL}\nmanifest_sha256={manifest_hash}\n",
        encoding="utf-8",
    )


def validate_rendered_outputs(work_root: Path, paper_root: Path) -> None:
    observed_work = {item.name for item in work_root.iterdir() if item.is_file()}
    observed_paper = {item.name for item in paper_root.iterdir() if item.is_file()}
    if observed_work != WORK_FILES:
        raise RuntimeError("work bundle entry set differs")
    if observed_paper != PAPER_FILES:
        raise RuntimeError("paper bundle must contain exactly the five declared PDFs")
    _validate_written_sources(work_root)
    manifest = _read_json(work_root / "render_manifest.json")
    if (
        manifest.get("protocol") != RENDER_PROTOCOL
        or manifest.get("status") != "complete"
        or set(_mapping(manifest.get("paper_files"), "manifest paper files")) != PAPER_FILES
    ):
        raise RuntimeError("render manifest contract differs")
    expected_complete = (
        f"protocol={RENDER_PROTOCOL}\n"
        f"manifest_sha256={_file_sha256(work_root / 'render_manifest.json')}\n"
    )
    if (work_root / "COMPLETE").read_text(encoding="utf-8") != expected_complete:
        raise RuntimeError("work COMPLETE marker differs")
    for stem in OUTPUT_STEMS:
        svg = (work_root / f"{stem}.svg").read_text(encoding="utf-8")
        if "<text" not in svg or "Times New Roman" not in svg:
            raise RuntimeError(f"{stem} editable-font SVG contract differs")
        if not (work_root / f"{stem}.pdf").read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"{stem} work PDF is malformed")
        if not (paper_root / f"{stem}.pdf").read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"{stem} paper PDF is malformed")
        if not (work_root / f"{stem}.png").read_bytes().startswith(b"\x89PNG"):
            raise RuntimeError(f"{stem} PNG is malformed")
        if (work_root / f"{stem}.tiff").read_bytes()[:4] not in {b"II*\x00", b"MM\x00*"}:
            raise RuntimeError(f"{stem} TIFF is malformed")
    for group, root in (("work_files", work_root), ("paper_files", paper_root)):
        for name, contract in _mapping(manifest[group], group).items():
            _validate_file_contract(root / name, contract)


def _validate_written_sources(work_root: Path) -> None:
    status = pd.read_csv(work_root / "setting_status.csv", float_precision="round_trip")
    stage = pd.read_csv(work_root / "coverage_stage_profiles.csv", float_precision="round_trip")
    scalar = pd.read_csv(work_root / "coverage_scalar_summary.csv", float_precision="round_trip")
    validate_reporting_sources(status, stage, scalar)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _output_title(stem: str) -> str:
    return {
        MAIN_STAGE_STEM: "Clinical gamma=-4 complete stagewise coverage reporting",
        MAIN_TABLE_STEM: "Clinical gamma=-4 complete metrics",
        SIGNED_FIGURE_STEM: "MIMIC-IV v2 complete signed-gamma metrics",
        SIGNED_TABLE_STEM: "MIMIC-IV v2 complete signed-gamma table",
        PRODUCTION_TABLE_STEM: "Frozen production complete metrics",
    }[stem]


def _finite_vector(
    value: object, length: int, label: str, *, positive: bool = False
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise RuntimeError(f"{label} must be a finite length-{length} vector")
    if positive and np.any(vector <= 0.0):
        raise RuntimeError(f"{label} must be positive")
    if not positive and np.any((vector < 0.0) | (vector > 1.0)):
        raise RuntimeError(f"{label} must be in [0,1]")
    return vector


def _finite_intervals(
    value: object, points: np.ndarray, label: str, *, positive: bool = False
) -> np.ndarray:
    intervals = np.asarray(value, dtype=np.float64)
    if intervals.shape != (len(points), 2) or not np.isfinite(intervals).all():
        raise RuntimeError(f"{label} must be a finite {len(points)}x2 array")
    if np.any(intervals[:, 0] > points) or np.any(points > intervals[:, 1]):
        raise RuntimeError(f"{label} does not contain its point estimates")
    if positive and np.any(intervals <= 0.0):
        raise RuntimeError(f"{label} must be positive")
    if not positive and np.any((intervals < 0.0) | (intervals > 1.0)):
        raise RuntimeError(f"{label} must be in [0,1]")
    return intervals


def _format_gamma(gamma: float) -> str:
    if gamma > 0:
        return f"+{gamma:g}"
    return f"{gamma:g}".replace("-", "-")


def _gamma_id(gamma: float) -> str:
    if gamma < 0:
        return f"minus{abs(int(gamma))}"
    if gamma > 0:
        return f"plus{int(gamma)}"
    return "zero"


def _optional_int(value: object) -> int | None:
    return None if pd.isna(value) else int(value)


def _optional_text(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in {"True", "False"}:
        return value == "True"
    raise RuntimeError(f"expected a strict boolean, received {value!r}")


def _same_value(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return left == right


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_contract(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _validate_file_contract(path: Path, contract: object) -> None:
    expected = _mapping(contract, f"file contract {path.name}")
    if (
        not path.is_file()
        or expected.get("bytes") != path.stat().st_size
        or expected.get("sha256") != _file_sha256(path)
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
