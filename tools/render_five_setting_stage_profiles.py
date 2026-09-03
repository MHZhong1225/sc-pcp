"""Render the two distinct five-setting stagewise comparison figures.

Figure A expands the frozen production/native RQ1 stage profiles to all five
datasets.  It contains no controlled signed-gamma intervention.  Figure B is
the gate-aware controlled-stress grid: native Synthetic beta=2 is a separate
stratum, while the four dataset-native clinical columns use the formal v2
gamma=-4 endpoint.  A failed clinical hard gate is rendered as a gate card and
never as an imputed curve.

This command only validates and renders completed artifacts.  It does not fit,
calibrate, resample scientific seeds, or run an experiment.

Example
-------
conda run -n ucp python tools/render_five_setting_stage_profiles.py
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
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RENDER_PROTOCOL = "five_setting_stage_profiles_render_v1"
DEFAULT_PRODUCTION_INPUT = ROOT / "results/work/complete_baseline_results_20260824"
DEFAULT_SYNTHETIC_INPUT = (
    ROOT / "results/work/native_synthetic_beta2_contract_20260826"
)
DEFAULT_CLINICAL_INPUT = ROOT / "results/work/controlled_clinical_extension_v2"
DEFAULT_WORK_OUTPUT = ROOT / "results/work/five_setting_stage_profiles_20260826"
DEFAULT_PAPER_OUTPUT = ROOT / "results/paper_five_setting_stage_profiles_20260826"

PRODUCTION_STEM = "figure_stagewise_profiles"
CONTROLLED_STEM = "figure_controlled_stress_grid"
TARGET = 0.90
METHODS = ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")
PRODUCTION_DATASETS = ("synthetic", "mimic_iv", "mimic_cxr", "eicu", "inspire")
CLINICAL_DATASETS = ("mimic_iv", "eicu", "inspire", "mimic_cxr")
CONTROLLED_SETTING_IDS = (
    "synthetic_beta2",
    "mimic_iv_gamma_minus4",
    "eicu_gamma_minus4",
    "inspire_gamma_minus4",
    "mimic_cxr_gamma_minus4",
)
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
PRODUCTION_LABELS = {
    "synthetic": "Synthetic",
    "mimic_iv": "MIMIC-IV",
    "mimic_cxr": "MIMIC-CXR + IV/ED",
    "eicu": "eICU",
    "inspire": "INSPIRE",
}
CONTROLLED_LABELS = {
    "synthetic": "Synthetic\n(native β=2)",
    "mimic_iv": "MIMIC-IV\n(clinical γ=−4)",
    "eicu": "eICU\n(clinical γ=−4)",
    "inspire": "INSPIRE\n(clinical γ=−4)",
    "mimic_cxr": "MIMIC-CXR + IV/ED\n(clinical γ=−4)",
}

NORMALIZED_WIDTH_DEFINITION = (
    "d^{-1} sum_j [2 q_t sigma_hat_{i,t,j} / sigma_out_{s,j}]; "
    "mean normalized coordinate length, not area or log-volume"
)

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
INFORMATION_REGIME = {
    "Standard CP": "offline_logged_data",
    "ACI": "on_policy_adaptation",
    "MFCS": "offline_logged_data",
    "SPCI": "on_policy_adaptation",
    "PRC": "on_policy_adaptation",
    "SC-PCP": "offline_logged_data",
}

FROZEN_SYNTHETIC_HASHES = {
    "manifest.json": "f9abfbb81d9f654b817fa44606c0df8da3add5860ac624b431b51dea762631ad",
    "native_synthetic_beta2_stage_profile_source_data.csv": (
        "ea8bd28f59ef0a1811a0f326cb9cd17172cc719bf4c16f18e12a1f36ef9edf18"
    ),
    "source_contract.json": (
        "5de4a5a621f1dc4c8f89abbef45e5c396410d6d109d5e2e53a86e93cb3352720"
    ),
}
SYNTHETIC_CONTRACT_ID = "native_synthetic_tail_shift_beta2_frozen_rq3_v1"

SETTING_STATUS_COLUMNS = (
    "schema_version",
    "setting_id",
    "column_index",
    "display_label",
    "stratum",
    "dataset",
    "feedback_parameter",
    "feedback_value",
    "signed_scale_comparable_across_strata",
    "uses_clinical_donor_kernel",
    "horizon",
    "method_count",
    "prespecified_seeds",
    "support_available",
    "k0_fidelity_available",
    "panel_status",
    "interpretation_status",
    "hard_gate_reason",
    "scientific_rows_saved",
    "curves_rendered",
    "confirmatory_ranking_included",
    "ranking_status",
    "source_path",
    "source_sha256",
)

STAGE_PROFILE_COLUMNS = (
    "schema_version",
    "figure",
    "setting_id",
    "column_index",
    "display_label",
    "stratum",
    "dataset",
    "feedback_parameter",
    "feedback_value",
    "analysis_role",
    "panel_status",
    "confirmatory_ranking_included",
    "method",
    "information_regime",
    "target_adaptation_trajectories_per_seed",
    "stage_zero_based",
    "n_prespecified",
    "n_selected",
    "metric_available",
    "coverage_target",
    "coverage_mean",
    "coverage_ci95_lower",
    "coverage_ci95_upper",
    "coverage_deviation_from_target_pp",
    "coverage_deviation_ci95_lower_pp",
    "coverage_deviation_ci95_upper_pp",
    "normalized_width_mean",
    "normalized_width_ci95_lower",
    "normalized_width_ci95_upper",
    "normalized_width_definition",
    "interval_definition",
    "interval_scope",
    "source_path",
    "source_sha256",
)

PRODUCTION_SOURCE_COLUMNS = (
    "schema_version",
    "figure",
    "setting_type",
    "dataset",
    "display_label",
    "source_setting",
    "native_feedback_strength",
    "controlled_gamma_used",
    "method",
    "stage_zero_based",
    "n_runs",
    "n_selected",
    "coverage_target",
    "coverage_mean",
    "coverage_ci95_lower",
    "coverage_ci95_upper",
    "coverage_deviation_from_target_pp",
    "coverage_deviation_ci95_lower_pp",
    "coverage_deviation_ci95_upper_pp",
    "normalized_width_mean",
    "normalized_width_ci95_lower",
    "normalized_width_ci95_upper",
    "normalized_width_definition",
    "interval_definition",
    "source_path",
    "source_sha256",
)

METHOD_SUMMARY_COLUMNS = (
    "schema_version",
    "figure",
    "setting_id",
    "display_label",
    "stratum",
    "dataset",
    "feedback_parameter",
    "feedback_value",
    "panel_status",
    "analysis_role",
    "confirmatory_ranking_included",
    "method",
    "metric_available",
    "coverage_target",
    "n_prespecified",
    "n_selected",
    "selection_rate",
    "selection_rate_ci95_lower",
    "selection_rate_ci95_upper",
    "wsc",
    "wsc_ci95_lower",
    "wsc_ci95_upper",
    "worst_stage_zero_based",
    "mean_normalized_width",
    "mean_normalized_width_ci95_lower",
    "mean_normalized_width_ci95_upper",
    "normalized_width_definition",
    "point_eligibility_rule",
    "point_eligible",
    "primary_metric",
    "wsc_interval_definition",
    "mean_width_interval_definition",
    "source_path",
    "source_sha256",
)

WORK_FILES = {
    *(f"{stem}.{suffix}" for stem in (PRODUCTION_STEM, CONTROLLED_STEM) for suffix in ("svg", "pdf", "tiff", "png")),
    "production_stage_profiles.csv",
    "setting_status.csv",
    "stage_profiles.csv",
    "method_summary.csv",
    "figure_contract.json",
    "figure_qa.md",
    "render_manifest.json",
    "COMPLETE",
}
PAPER_FILES = {f"{PRODUCTION_STEM}.pdf", f"{CONTROLLED_STEM}.pdf"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-input", type=Path, default=DEFAULT_PRODUCTION_INPUT)
    parser.add_argument("--synthetic-input", type=Path, default=DEFAULT_SYNTHETIC_INPUT)
    parser.add_argument("--clinical-input", type=Path, default=DEFAULT_CLINICAL_INPUT)
    parser.add_argument("--work-output", type=Path, default=DEFAULT_WORK_OUTPUT)
    parser.add_argument("--paper-output", type=Path, default=DEFAULT_PAPER_OUTPUT)
    args = parser.parse_args()
    render_report(
        production_input=args.production_input.resolve(),
        synthetic_input=args.synthetic_input.resolve(),
        clinical_input=args.clinical_input.resolve(),
        work_output=args.work_output.resolve(),
        paper_output=args.paper_output.resolve(),
    )
    print(args.paper_output.resolve())


def render_report(
    *,
    production_input: Path,
    synthetic_input: Path,
    clinical_input: Path,
    work_output: Path,
    paper_output: Path,
) -> None:
    """Validate all frozen inputs, then atomically publish the two figures."""

    if work_output.exists() or paper_output.exists():
        raise FileExistsError("work-output and paper-output must both be fresh")
    if work_output == paper_output:
        raise ValueError("work-output and paper-output must be different directories")

    production, production_summary, production_contract = load_production_profiles(
        production_input
    )
    (
        synthetic_status,
        synthetic_profiles,
        synthetic_summary,
        synthetic_contract,
    ) = load_native_synthetic(
        synthetic_input, scalar_source_root=production_input
    )
    (
        clinical_status,
        clinical_profiles,
        clinical_summary,
        clinical_contract,
    ) = load_complete_clinical(clinical_input)
    setting_status = pd.concat(
        [synthetic_status, clinical_status], ignore_index=True
    ).loc[:, SETTING_STATUS_COLUMNS]
    stage_profiles = pd.concat(
        [synthetic_profiles, clinical_profiles], ignore_index=True
    ).loc[:, STAGE_PROFILE_COLUMNS]
    method_summary = pd.concat(
        [production_summary, synthetic_summary, clinical_summary], ignore_index=True
    ).loc[:, METHOD_SUMMARY_COLUMNS]
    validate_controlled_render_source(setting_status, stage_profiles)
    validate_method_summary(method_summary, production, setting_status, stage_profiles)

    work_output.parent.mkdir(parents=True, exist_ok=True)
    paper_output.parent.mkdir(parents=True, exist_ok=True)
    staged_work = Path(
        tempfile.mkdtemp(prefix=f".{work_output.name}-", dir=work_output.parent)
    )
    staged_paper = Path(
        tempfile.mkdtemp(prefix=f".{paper_output.name}-", dir=paper_output.parent)
    )
    try:
        _write_csv(staged_work / "production_stage_profiles.csv", production)
        _write_csv(staged_work / "setting_status.csv", setting_status)
        _write_csv(staged_work / "stage_profiles.csv", stage_profiles)
        _write_csv(staged_work / "method_summary.csv", method_summary)
        for path, frame in (
            (staged_work / "production_stage_profiles.csv", production),
            (staged_work / "setting_status.csv", setting_status),
            (staged_work / "stage_profiles.csv", stage_profiles),
            (staged_work / "method_summary.csv", method_summary),
        ):
            _validate_csv_roundtrip(path, frame)
        _write_figure_contract(
            staged_work / "figure_contract.json",
            production_contract=production_contract,
            synthetic_contract=synthetic_contract,
            clinical_contract=clinical_contract,
            setting_status=setting_status,
            production_rows=len(production),
            controlled_rows=len(stage_profiles),
            method_summary_rows=len(method_summary),
        )
        _write_qa(
            staged_work / "figure_qa.md",
            production=production,
            setting_status=setting_status,
            stage_profiles=stage_profiles,
            method_summary=method_summary,
        )
        apply_publication_style()
        export_figure(
            render_production_figure(production),
            title="Five-setting production/native stagewise profiles",
            work_stem=staged_work / PRODUCTION_STEM,
        )
        export_figure(
            render_controlled_figure(setting_status, stage_profiles),
            title="Gate-aware five-setting controlled stress grid",
            work_stem=staged_work / CONTROLLED_STEM,
        )
        _write_render_manifest(
            staged_work / "render_manifest.json",
            work_root=staged_work,
            paper_root=staged_paper,
        )
        _write_work_complete(staged_work)
        _copy_paper_from_completed_work(staged_work, staged_paper)
        validate_rendered_outputs(staged_work, staged_paper)
        _publish_bundles(
            staged_work=staged_work,
            staged_paper=staged_paper,
            work_output=work_output,
            paper_output=paper_output,
        )
    except BaseException:
        shutil.rmtree(staged_work, ignore_errors=True)
        shutil.rmtree(staged_paper, ignore_errors=True)
        raise


def load_production_profiles(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the exact frozen RQ1 grid and retain all five native settings."""

    from tools import render_main_suite_figures as frozen_renderer

    main_frame, stage_frame, metadata = frozen_renderer.load_frozen_export(root)
    selected = stage_frame[stage_frame["section"].eq("RQ1")].copy()
    dataset_order = {name: index for index, name in enumerate(PRODUCTION_DATASETS)}
    method_order = {name: index for index, name in enumerate(METHODS)}
    selected = selected.sort_values(
        ["dataset", "method", "stage_zero_based"],
        key=lambda values: (
            values.map(dataset_order)
            if values.name == "dataset"
            else values.map(method_order)
            if values.name == "method"
            else values
        ),
    )
    result = pd.DataFrame(
        {
            "schema_version": 1,
            "figure": PRODUCTION_STEM,
            "setting_type": "production_native_suite",
            "dataset": selected["dataset"],
            "display_label": selected["dataset"].map(PRODUCTION_LABELS),
            "source_setting": selected["setting"],
            "native_feedback_strength": selected["feedback_strength"],
            "controlled_gamma_used": False,
            "method": selected["method"],
            "stage_zero_based": selected["stage_zero_based"].astype(int),
            "n_runs": selected["n_runs"].astype(int),
            "n_selected": selected["n_selected"].astype(int),
            "coverage_target": TARGET,
            "coverage_mean": selected["coverage_mean"],
            "coverage_ci95_lower": selected["coverage_ci_low"],
            "coverage_ci95_upper": selected["coverage_ci_high"],
            "coverage_deviation_from_target_pp": 100.0
            * (selected["coverage_mean"] - TARGET),
            "coverage_deviation_ci95_lower_pp": 100.0
            * (selected["coverage_ci_low"] - TARGET),
            "coverage_deviation_ci95_upper_pp": 100.0
            * (selected["coverage_ci_high"] - TARGET),
            "normalized_width_mean": selected["normalized_width_mean"],
            "normalized_width_ci95_lower": selected["normalized_width_ci_low"],
            "normalized_width_ci95_upper": selected["normalized_width_ci_high"],
            "normalized_width_definition": NORMALIZED_WIDTH_DEFINITION,
            "interval_definition": metadata["per_stage_interval"],
            "source_path": "complete_baseline_results_20260824/per_stage_all_baselines.csv",
            "source_sha256": frozen_renderer.INPUT_FILES[
                "per_stage_all_baselines.csv"
            ],
        }
    ).reset_index(drop=True)
    result = result.loc[:, PRODUCTION_SOURCE_COLUMNS]
    validate_production_profiles(result)
    summaries: list[dict[str, Any]] = []
    main_index = main_frame.set_index(["dataset", "method"])
    for dataset in PRODUCTION_DATASETS:
        for method in METHODS:
            scalar = main_index.loc[(dataset, method)]
            stages = result[
                result["dataset"].eq(dataset) & result["method"].eq(method)
            ].sort_values("stage_zero_based")
            summaries.append(
                _method_summary_row(
                    figure=PRODUCTION_STEM,
                    setting_id=f"production_{dataset}",
                    display_label=PRODUCTION_LABELS[dataset],
                    stratum="production_native_suite",
                    dataset=dataset,
                    feedback_parameter=(
                        "native_feedback_strength" if dataset == "synthetic" else "none"
                    ),
                    feedback_value=(
                        float(stages["native_feedback_strength"].iloc[0])
                        if dataset == "synthetic"
                        else None
                    ),
                    panel_status="CURVES",
                    analysis_role="frozen_production_native_rq1",
                    confirmatory_ranking_included=False,
                    method=method,
                    metric_available=True,
                    n_prespecified=int(scalar["n_runs"]),
                    n_selected=int(scalar["n_selected"]),
                    selection_rate=float(scalar["selection_rate"]),
                    selection_interval=(
                        float(scalar["selection_rate_ci_low"]),
                        float(scalar["selection_rate_ci_high"]),
                    ),
                    wsc=float(scalar["marginal_worst_coverage"]),
                    wsc_interval=(
                        float(scalar["marginal_worst_coverage_ci_low"]),
                        float(scalar["marginal_worst_coverage_ci_high"]),
                    ),
                    worst_stage=int(stages["coverage_mean"].to_numpy(float).argmin()),
                    mean_width=float(scalar["average_normalized_width"]),
                    mean_width_interval=(
                        float(scalar["average_normalized_width_ci_low"]),
                        float(scalar["average_normalized_width_ci_high"]),
                    ),
                    point_eligible=_as_bool(scalar["efficiency_eligible"]),
                    wsc_interval_definition=metadata["wsc_interval"],
                    mean_width_interval_definition=metadata[
                        "mean_coverage_and_width_interval"
                    ],
                    source_path="complete_baseline_results_20260824/rq1_all_baselines.csv",
                    source_sha256=frozen_renderer.INPUT_FILES[
                        "rq1_all_baselines.csv"
                    ],
                )
            )
    summary_frame = pd.DataFrame(summaries).loc[:, METHOD_SUMMARY_COLUMNS]
    return result, summary_frame, {
        "input_root": _project_path(root),
        "input_file_sha256": dict(frozen_renderer.INPUT_FILES),
        "metadata_sha256": frozen_renderer.INPUT_FILES["metadata.json"],
        "row_count": len(result),
        "setting_order": list(PRODUCTION_DATASETS),
        "controlled_gamma_used": False,
    }


def validate_production_profiles(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != PRODUCTION_SOURCE_COLUMNS:
        raise RuntimeError("production stage-profile schema differs")
    expected = {
        (dataset, method, stage)
        for dataset in PRODUCTION_DATASETS
        for method in METHODS
        for stage in range(HORIZONS[dataset])
    }
    observed = set(
        zip(frame["dataset"], frame["method"], frame["stage_zero_based"])
    )
    if observed != expected or len(frame) != len(expected):
        raise RuntimeError("production stage-profile grid differs")
    if frame.duplicated(["dataset", "method", "stage_zero_based"]).any():
        raise RuntimeError("production stage-profile keys are duplicated")
    if frame["controlled_gamma_used"].astype(bool).any():
        raise RuntimeError("production/native figure must not contain controlled gamma")
    if not all(
        set(frame.loc[frame["dataset"].eq(dataset), "display_label"])
        == {PRODUCTION_LABELS[dataset]}
        for dataset in PRODUCTION_DATASETS
    ):
        raise RuntimeError("production dataset labels differ")
    expected_runs = frame["dataset"].map(EXPECTED_RUNS).to_numpy(int)
    if not np.array_equal(frame["n_runs"].to_numpy(int), expected_runs):
        raise RuntimeError("production run counts differ")
    if not np.array_equal(frame["n_selected"], frame["n_runs"]):
        raise RuntimeError("production source must have complete selection")
    _validate_numeric_profile_columns(frame)


def load_native_synthetic(
    root: Path,
    *,
    scalar_source_root: Path = DEFAULT_PRODUCTION_INPUT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate and adapt the frozen beta=2 native-Synthetic source contract."""

    if not root.is_dir():
        raise FileNotFoundError(f"native-Synthetic contract does not exist: {root}")
    observed = {path.name for path in root.iterdir() if path.is_file()}
    if observed != set(FROZEN_SYNTHETIC_HASHES):
        raise RuntimeError("native-Synthetic contract file set differs")
    for name, expected in FROZEN_SYNTHETIC_HASHES.items():
        if _file_sha256(root / name) != expected:
            raise RuntimeError(f"native-Synthetic frozen hash differs: {name}")
    manifest = _read_json(root / "manifest.json")
    contract = _read_json(root / "source_contract.json")
    source_path = root / "native_synthetic_beta2_stage_profile_source_data.csv"
    if (
        manifest.get("contract_id") != SYNTHETIC_CONTRACT_ID
        or manifest.get("deterministic_postprocessing_only") is not True
        or manifest.get("scientific_rng_used") is not False
        or manifest.get("files")
        != {
            source_path.name: FROZEN_SYNTHETIC_HASHES[source_path.name],
            "source_contract.json": FROZEN_SYNTHETIC_HASHES["source_contract.json"],
        }
    ):
        raise RuntimeError("native-Synthetic manifest contract differs")
    semantics = _mapping(contract.get("semantics"), "synthetic semantics")
    statistics = _mapping(contract.get("statistics"), "synthetic statistics")
    if (
        contract.get("contract_id") != SYNTHETIC_CONTRACT_ID
        or contract.get("status") != "frozen_deterministic_source_only"
        or tuple(contract.get("methods", ())) != METHODS
        or semantics.get("dataset") != "synthetic"
        or semantics.get("feedback_parameter") != "beta"
        or float(semantics.get("feedback_strength")) != 2.0
        or semantics.get("signed_gamma_comparable") is not False
        or semantics.get("uses_clinical_donor_kernel") is not False
        or semantics.get("horizon") != 12
        or float(statistics.get("coverage_target")) != TARGET
        or statistics.get("n_prespecified_seeds") != 100
        or statistics.get("n_selected_seeds") != 100
        or "beta is not the signed gamma scale"
        not in str(contract.get("required_disambiguator"))
    ):
        raise RuntimeError("native-Synthetic semantic contract differs")

    source = pd.read_csv(source_path)
    expected_source_columns = (
        "contract_id",
        "display_label",
        "dataset",
        "scenario",
        "feedback_parameter",
        "feedback_strength",
        "signed_gamma_comparable",
        "uses_clinical_donor_kernel",
        "method",
        "information_regime",
        "stage_zero_based",
        "n_runs",
        "n_selected",
        "coverage_mean",
        "coverage_ci95_lower",
        "coverage_ci95_upper",
        "coverage_deviation_from_090_pp",
        "coverage_deviation_ci95_lower_pp",
        "coverage_deviation_ci95_upper_pp",
        "normalized_width_mean",
        "normalized_width_ci95_lower",
        "normalized_width_ci95_upper",
        "interval_definition",
        "raw_source",
        "aggregate_source",
    )
    if tuple(source.columns) != expected_source_columns:
        raise RuntimeError("native-Synthetic source schema differs")
    expected_keys = {
        (method, stage) for method in METHODS for stage in range(HORIZONS["synthetic"])
    }
    observed_keys = set(zip(source["method"], source["stage_zero_based"]))
    if observed_keys != expected_keys or len(source) != len(expected_keys):
        raise RuntimeError("native-Synthetic method-stage grid differs")
    if (
        set(source["contract_id"]) != {SYNTHETIC_CONTRACT_ID}
        or set(source["feedback_parameter"]) != {"beta"}
        or set(source["feedback_strength"].astype(float)) != {2.0}
        or source["signed_gamma_comparable"].astype(bool).any()
        or source["uses_clinical_donor_kernel"].astype(bool).any()
        or set(source["n_runs"].astype(int)) != {100}
        or set(source["n_selected"].astype(int)) != {100}
    ):
        raise RuntimeError("native-Synthetic row semantics differ")

    status = pd.DataFrame(
        [
            _setting_status_row(
                setting_id="synthetic_beta2",
                column_index=0,
                display_label=CONTROLLED_LABELS["synthetic"],
                stratum="native_synthetic_separate",
                dataset="synthetic",
                feedback_parameter="beta",
                feedback_value=2.0,
                signed_scale_comparable_across_strata=False,
                uses_clinical_donor_kernel=False,
                horizon=12,
                prespecified_seeds=100,
                support_available=None,
                k0_fidelity_available=None,
                panel_status="CURVES",
                interpretation_status="NATIVE_SYNTHETIC_SEPARATE_STRATUM",
                hard_gate_reason="",
                scientific_rows_saved=True,
                curves_rendered=True,
                confirmatory_ranking_included=False,
                ranking_status="SEPARATE_SYNTHETIC_STRATUM_NOT_CLINICAL_RANKING",
                source_path=_project_path(source_path),
                source_sha256=FROZEN_SYNTHETIC_HASHES[source_path.name],
            )
        ]
    )
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        rows.append(
            _stage_profile_row(
                setting_id="synthetic_beta2",
                column_index=0,
                display_label=CONTROLLED_LABELS["synthetic"],
                stratum="native_synthetic_separate",
                dataset="synthetic",
                feedback_parameter="beta",
                feedback_value=2.0,
                analysis_role="native_synthetic_separate_stratum",
                panel_status="CURVES",
                confirmatory_ranking_included=False,
                method=str(row["method"]),
                information_regime=str(row["information_regime"]),
                target_adaptation_trajectories_per_seed=None,
                stage=int(row["stage_zero_based"]),
                n_prespecified=int(row["n_runs"]),
                n_selected=int(row["n_selected"]),
                coverage=float(row["coverage_mean"]),
                coverage_interval=(
                    float(row["coverage_ci95_lower"]),
                    float(row["coverage_ci95_upper"]),
                ),
                width=float(row["normalized_width_mean"]),
                width_interval=(
                    float(row["normalized_width_ci95_lower"]),
                    float(row["normalized_width_ci95_upper"]),
                ),
                interval_definition=str(row["interval_definition"]),
                interval_scope="pointwise, not simultaneous",
                source_path=_project_path(source_path),
                source_sha256=FROZEN_SYNTHETIC_HASHES[source_path.name],
            )
        )
    profiles = pd.DataFrame(rows).loc[:, STAGE_PROFILE_COLUMNS]
    from tools import render_main_suite_figures as frozen_renderer

    frozen_renderer.load_frozen_export(scalar_source_root)
    scalar_path = scalar_source_root / "rq3_all_baselines.csv"
    scalar_source = pd.read_csv(scalar_path)
    scalar_source = scalar_source[
        scalar_source["section"].eq("RQ3")
        & scalar_source["dataset"].eq("synthetic")
        & scalar_source["feedback_strength"].eq(2.0)
    ].copy()
    if (
        len(scalar_source) != len(METHODS)
        or set(scalar_source["method"]) != set(METHODS)
        or scalar_source.duplicated("method").any()
    ):
        raise RuntimeError("native-Synthetic beta=2 scalar source grid differs")
    scalar_index = scalar_source.set_index("method")
    contract_summary = _mapping(contract.get("method_summary"), "synthetic method summary")
    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        scalar = scalar_index.loc[method]
        stages = profiles[profiles["method"].eq(method)].sort_values(
            "stage_zero_based"
        )
        stored = _mapping(contract_summary.get(method), f"synthetic summary/{method}")
        if (
            not math.isclose(
                float(scalar["marginal_worst_coverage"]),
                float(stored["wsc"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            )
            or int(stages["coverage_mean"].to_numpy(float).argmin())
            != int(stored["worst_stage_zero_based"])
        ):
            raise RuntimeError(f"native-Synthetic scalar contract differs for {method}")
        summary_rows.append(
            _method_summary_row(
                figure=CONTROLLED_STEM,
                setting_id="synthetic_beta2",
                display_label=CONTROLLED_LABELS["synthetic"],
                stratum="native_synthetic_separate",
                dataset="synthetic",
                feedback_parameter="beta",
                feedback_value=2.0,
                panel_status="CURVES",
                analysis_role="native_synthetic_separate_stratum",
                confirmatory_ranking_included=False,
                method=method,
                metric_available=True,
                n_prespecified=int(scalar["n_runs"]),
                n_selected=int(scalar["n_selected"]),
                selection_rate=float(scalar["selection_rate"]),
                selection_interval=(
                    float(scalar["selection_rate_ci_low"]),
                    float(scalar["selection_rate_ci_high"]),
                ),
                wsc=float(scalar["marginal_worst_coverage"]),
                wsc_interval=(
                    float(scalar["marginal_worst_coverage_ci_low"]),
                    float(scalar["marginal_worst_coverage_ci_high"]),
                ),
                worst_stage=int(stages["coverage_mean"].to_numpy(float).argmin()),
                mean_width=float(scalar["average_normalized_width"]),
                mean_width_interval=(
                    float(scalar["average_normalized_width_ci_low"]),
                    float(scalar["average_normalized_width_ci_high"]),
                ),
                point_eligible=_as_bool(scalar["efficiency_eligible"]),
                wsc_interval_definition=(
                    "10000 complete-seed-vector percentile bootstrap draws"
                ),
                mean_width_interval_definition=(
                    "two-sided 95% Student-t interval across selected seeds"
                ),
                source_path="complete_baseline_results_20260824/rq3_all_baselines.csv",
                source_sha256=frozen_renderer.INPUT_FILES["rq3_all_baselines.csv"],
            )
        )
    summary_frame = pd.DataFrame(summary_rows).loc[:, METHOD_SUMMARY_COLUMNS]
    return status.loc[:, SETTING_STATUS_COLUMNS], profiles, summary_frame, {
        "contract_id": SYNTHETIC_CONTRACT_ID,
        "input_root": _project_path(root),
        "file_sha256": dict(FROZEN_SYNTHETIC_HASHES),
        "required_disambiguator": contract["required_disambiguator"],
        "scalar_source_path": _project_path(scalar_path),
        "scalar_source_sha256": frozen_renderer.INPUT_FILES[
            "rq3_all_baselines.csv"
        ],
    }


def load_complete_clinical(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load clinical science only after the formal root is durably complete."""

    complete_path = root / "COMPLETE"
    if not complete_path.is_file():
        raise RuntimeError(
            "clinical root COMPLETE is required before any clinical science artifact is read"
        )

    from scripts import run_controlled_clinical_extension as runner
    from scpcp.controlled_clinical_extension import (
        DATASET_NAMES,
        load_extension_config,
    )

    if tuple(DATASET_NAMES) != CLINICAL_DATASETS:
        raise RuntimeError("clinical dataset order differs from the renderer contract")
    runner._verify_manifest(root)
    metadata = runner._read_json(root / "metadata.json")
    runner._verify_source_snapshot(root, metadata["source_snapshot"])
    expected_complete = runner._root_complete_marker(
        metadata["source_snapshot"],
        metadata["precoverage_engineering_retry_amendment"],
        metadata["postcompute_preinspection_retry_amendment"],
    )
    if complete_path.read_text(encoding="utf-8") != expected_complete:
        raise RuntimeError("clinical root COMPLETE binding differs")
    protocol = load_extension_config(runner.CONFIG_PATH)
    root_summary = runner._read_json(root / "summary.json")
    if (
        metadata.get("protocol") != runner.PROTOCOL
        or metadata.get("role") != "fresh_dataset_native_controlled_clinical_extension"
        or tuple(metadata.get("datasets", ())) != CLINICAL_DATASETS
        or tuple(root_summary.get("datasets", ())) != CLINICAL_DATASETS
        or tuple(root_summary.get("completed_datasets", ())) != CLINICAL_DATASETS
    ):
        raise RuntimeError("clinical root dataset/protocol contract differs")

    status_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    dataset_sources: dict[str, Any] = {}
    for clinical_index, dataset in enumerate(CLINICAL_DATASETS, start=1):
        dataset_root = root / dataset
        dataset_metadata = runner._read_json(dataset_root / "metadata.json")
        runner._validate_final_dataset_bundle(
            dataset_root,
            protocol=protocol,
            preset=protocol.datasets[dataset],
            dataset_metadata=dataset_metadata,
        )
        final = runner._read_json(dataset_root / "FINAL_STATUS.json")
        if root_summary["dataset_status"].get(dataset) != final:
            raise RuntimeError(f"{dataset} root and dataset statuses differ")
        gate = runner._read_json(dataset_root / "gate.json")
        support = runner._read_json(dataset_root / "support" / "summary.json")
        k0_path = dataset_root / "k0_fidelity" / "summary.json"
        k0 = runner._read_json(k0_path) if k0_path.is_file() else None
        source_path = (
            dataset_root / "science" / "summary.json"
            if gate.get("panel_status") in {"CURVES", "CURVES_DESCRIPTIVE_ONLY"}
            else dataset_root / "NO_GO.json"
        )
        source_hash = _file_sha256(source_path)
        status, rows, summaries = adapt_clinical_dataset(
            dataset=dataset,
            column_index=clinical_index,
            horizon=protocol.datasets[dataset].horizon,
            prespecified_seeds=len(protocol.datasets[dataset].seeds),
            gate=gate,
            final=final,
            support_summary=support,
            k0_summary=k0,
            science_summary=(
                runner._read_json(source_path)
                if gate.get("panel_status")
                in {"CURVES", "CURVES_DESCRIPTIVE_ONLY"}
                else None
            ),
            source_path=_project_path(source_path),
            source_sha256=source_hash,
        )
        status_rows.append(status)
        profile_rows.extend(rows)
        summary_rows.extend(summaries)
        dataset_sources[dataset] = {
            "panel_status": status["panel_status"],
            "final_status": final["status"],
            "source_path": _project_path(source_path),
            "source_sha256": source_hash,
            "dataset_manifest_sha256": _file_sha256(dataset_root / "manifest.json"),
        }
    return (
        pd.DataFrame(status_rows).loc[:, SETTING_STATUS_COLUMNS],
        pd.DataFrame(profile_rows, columns=STAGE_PROFILE_COLUMNS),
        pd.DataFrame(summary_rows, columns=METHOD_SUMMARY_COLUMNS),
        {
            "protocol": runner.PROTOCOL,
            "input_root": _project_path(root),
            "root_manifest_sha256": _file_sha256(root / "manifest.json"),
            "source_snapshot": metadata["source_snapshot"],
            "source_tree_sha256": metadata["source_tree_sha256"],
            "datasets": dataset_sources,
        },
    )


def adapt_clinical_dataset(
    *,
    dataset: str,
    column_index: int,
    horizon: int,
    prespecified_seeds: int,
    gate: Mapping[str, Any],
    final: Mapping[str, Any],
    support_summary: Mapping[str, Any],
    k0_summary: Mapping[str, Any] | None,
    science_summary: Mapping[str, Any] | None,
    source_path: str,
    source_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Adapt one already-validated dataset bundle into plotting rows."""

    expected_column = CLINICAL_DATASETS.index(dataset) + 1 if dataset in CLINICAL_DATASETS else None
    if (
        dataset not in CLINICAL_DATASETS
        or horizon != HORIZONS[dataset]
        or column_index != expected_column
        or prespecified_seeds != EXPECTED_RUNS[dataset]
        or not _is_sha256(source_sha256)
    ):
        raise RuntimeError("clinical adapter identity/source hash differs")
    panel_status = str(gate.get("panel_status"))
    support_available = int(support_summary.get("n_available"))
    k0_available = (
        int(k0_summary.get("n_available")) if k0_summary is not None else None
    )
    setting_id = f"{dataset}_gamma_minus4"
    common = dict(
        setting_id=setting_id,
        column_index=column_index,
        display_label=CONTROLLED_LABELS[dataset],
        stratum="dataset_native_clinical_controlled",
        dataset=dataset,
        feedback_parameter="gamma",
        feedback_value=-4.0,
        signed_scale_comparable_across_strata=False,
        uses_clinical_donor_kernel=True,
        horizon=horizon,
        prespecified_seeds=prespecified_seeds,
        support_available=support_available,
        k0_fidelity_available=k0_available,
        source_path=source_path,
        source_sha256=source_sha256,
    )
    if panel_status == "GATE_NO_GO":
        reason = str(gate.get("reason"))
        if (
            final.get("status") != reason
            or final.get("scientific_rows_saved") is not False
            or science_summary is not None
            or reason
            not in {"SUPPORT_NO_GO", "STRUCTURAL_NO_GO", "K0_FIDELITY_NO_GO"}
        ):
            raise RuntimeError(f"{dataset} hard-gate status differs")
        unavailable_summaries = [
            _method_summary_row(
                figure=CONTROLLED_STEM,
                setting_id=setting_id,
                display_label=CONTROLLED_LABELS[dataset],
                stratum="dataset_native_clinical_controlled",
                dataset=dataset,
                feedback_parameter="gamma",
                feedback_value=-4.0,
                panel_status="GATE_NO_GO",
                analysis_role="hard_gate_no_go_no_science_rows",
                confirmatory_ranking_included=False,
                method=method,
                metric_available=False,
                n_prespecified=prespecified_seeds,
                n_selected=None,
                selection_rate=None,
                selection_interval=(None, None),
                wsc=None,
                wsc_interval=(None, None),
                worst_stage=None,
                mean_width=None,
                mean_width_interval=(None, None),
                point_eligible=None,
                wsc_interval_definition="not available: hard preflight NO-GO",
                mean_width_interval_definition="not available: hard preflight NO-GO",
                source_path=source_path,
                source_sha256=source_sha256,
            )
            for method in METHODS
        ]
        return (
            _setting_status_row(
                **common,
                panel_status="GATE_NO_GO",
                interpretation_status="HARD_GATE_NO_GO_NO_SCIENCE_ROWS",
                hard_gate_reason=reason,
                scientific_rows_saved=False,
                curves_rendered=False,
                confirmatory_ranking_included=False,
                ranking_status="EXCLUDED_HARD_GATE_NO_GO",
            ),
            [],
            unavailable_summaries,
        )

    if panel_status not in {"CURVES", "CURVES_DESCRIPTIVE_ONLY"}:
        raise RuntimeError(f"unknown clinical panel status: {panel_status}")
    if science_summary is None or final.get("scientific_rows_saved") is not True:
        raise RuntimeError(f"{dataset} curve panel lacks science summary")
    interpretation = str(gate.get("interpretation_status"))
    expected_interpretation = (
        "EMPIRICAL_OVERLAP_SCREEN_PASSED"
        if panel_status == "CURVES"
        else "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
    )
    if (
        interpretation != expected_interpretation
        or final.get("interpretation_status") != expected_interpretation
        or science_summary.get("interpretation_status") != expected_interpretation
        or tuple(science_summary.get("methods", ())) != METHODS
        or science_summary.get("primary_metric")
        != "min_t mean_seed(target_coverage_seed_t)"
    ):
        raise RuntimeError(f"{dataset} curve interpretation contract differs")
    confirmatory = panel_status == "CURVES"
    status = _setting_status_row(
        **common,
        panel_status=panel_status,
        interpretation_status=interpretation,
        hard_gate_reason="",
        scientific_rows_saved=True,
        curves_rendered=True,
        confirmatory_ranking_included=confirmatory,
        ranking_status=(
            "CONFIRMATORY_GAMMA_MINUS4_ENDPOINT"
            if confirmatory
            else "EXCLUDED_LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        ),
    )
    aggregates = [
        cell
        for cell in science_summary.get("aggregates", ())
        if isinstance(cell, Mapping) and float(cell.get("gamma")) == -4.0
    ]
    if len(aggregates) != 1:
        raise RuntimeError(f"{dataset} must have exactly one gamma=-4 aggregate")
    aggregate = aggregates[0]
    expected_role = (
        "confirmatory_gamma_minus_4_endpoint"
        if confirmatory
        else "descriptive_signed_control_curve"
    )
    if (
        aggregate.get("analysis_role") != expected_role
        or aggregate.get("n_prespecified_seeds") != prespecified_seeds
        or set(aggregate.get("methods", {})) != set(METHODS)
    ):
        raise RuntimeError(f"{dataset} gamma=-4 aggregate contract differs")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        cell = _mapping(aggregate["methods"][method], f"{dataset}/{method}")
        count = int(cell.get("n_selected"))
        coverage = cell.get("target_coverage_by_stage")
        coverage_ci = cell.get("target_coverage_by_stage_ci95")
        width = cell.get("target_normalized_width_by_stage")
        width_ci = cell.get("target_normalized_width_by_stage_ci95")
        if count == 0:
            if any(value not in ([], None) for value in (coverage, coverage_ci, width, width_ci)):
                raise RuntimeError(f"{dataset}/{method} unavailable vectors differ")
            summaries.append(
                _method_summary_row(
                    figure=CONTROLLED_STEM,
                    setting_id=setting_id,
                    display_label=CONTROLLED_LABELS[dataset],
                    stratum="dataset_native_clinical_controlled",
                    dataset=dataset,
                    feedback_parameter="gamma",
                    feedback_value=-4.0,
                    panel_status=panel_status,
                    analysis_role=expected_role,
                    confirmatory_ranking_included=confirmatory,
                    method=method,
                    metric_available=False,
                    n_prespecified=prespecified_seeds,
                    n_selected=0,
                    selection_rate=0.0,
                    selection_interval=tuple(cell["selection_rate_ci95"]),
                    wsc=None,
                    wsc_interval=(None, None),
                    worst_stage=None,
                    mean_width=None,
                    mean_width_interval=(None, None),
                    point_eligible=(False if confirmatory else None),
                    wsc_interval_definition=(
                        "not available: method selection unavailable"
                    ),
                    mean_width_interval_definition=(
                        "not available: method selection unavailable"
                    ),
                    source_path=source_path,
                    source_sha256=source_sha256,
                )
            )
            continue
        _validate_stage_vector(coverage, horizon, f"{dataset}/{method}/coverage")
        _validate_interval_vector(
            coverage_ci, coverage, horizon, f"{dataset}/{method}/coverage interval"
        )
        _validate_stage_vector(width, horizon, f"{dataset}/{method}/width", positive=True)
        _validate_interval_vector(
            width_ci, width, horizon, f"{dataset}/{method}/width interval", positive=True
        )
        if not 0 < count <= prespecified_seeds:
            raise RuntimeError(f"{dataset}/{method} selected count differs")
        summaries.append(
            _method_summary_row(
                figure=CONTROLLED_STEM,
                setting_id=setting_id,
                display_label=CONTROLLED_LABELS[dataset],
                stratum="dataset_native_clinical_controlled",
                dataset=dataset,
                feedback_parameter="gamma",
                feedback_value=-4.0,
                panel_status=panel_status,
                analysis_role=expected_role,
                confirmatory_ranking_included=confirmatory,
                method=method,
                metric_available=True,
                n_prespecified=prespecified_seeds,
                n_selected=count,
                selection_rate=float(cell["selection_rate"]),
                selection_interval=(
                    float(cell["selection_rate_ci95"][0]),
                    float(cell["selection_rate_ci95"][1]),
                ),
                wsc=float(cell["target_marginal_worst_coverage"]),
                wsc_interval=(
                    float(cell["target_wsc_ci95"][0]),
                    float(cell["target_wsc_ci95"][1]),
                ),
                worst_stage=int(cell["target_worst_stage_zero_based"]),
                mean_width=float(cell["mean_target_normalized_width"]),
                mean_width_interval=(
                    float(cell["mean_target_normalized_width_ci95"][0]),
                    float(cell["mean_target_normalized_width_ci95"][1]),
                ),
                point_eligible=cell["point_eligible"],
                wsc_interval_definition=(
                    "95% percentile interval over min_t of each complete-seed-vector "
                    "bootstrap stage-mean draw; frozen dataset stream; "
                    "10000 resamples"
                ),
                mean_width_interval_definition=(
                    "two-sided 95% Student-t interval across selected seed-level "
                    "mean normalized widths"
                ),
                source_path=source_path,
                source_sha256=source_sha256,
            )
        )
        for stage in range(horizon):
            rows.append(
                _stage_profile_row(
                    setting_id=setting_id,
                    column_index=column_index,
                    display_label=CONTROLLED_LABELS[dataset],
                    stratum="dataset_native_clinical_controlled",
                    dataset=dataset,
                    feedback_parameter="gamma",
                    feedback_value=-4.0,
                    analysis_role=expected_role,
                    panel_status=panel_status,
                    confirmatory_ranking_included=confirmatory,
                    method=method,
                    information_regime=INFORMATION_REGIME[method],
                    target_adaptation_trajectories_per_seed=int(
                        cell["target_adaptation_trajectories_per_seed"]
                    ),
                    stage=stage,
                    n_prespecified=prespecified_seeds,
                    n_selected=count,
                    coverage=float(coverage[stage]),
                    coverage_interval=(
                        float(coverage_ci[stage][0]),
                        float(coverage_ci[stage][1]),
                    ),
                    width=float(width[stage]),
                    width_interval=(
                        float(width_ci[stage][0]),
                        float(width_ci[stage][1]),
                    ),
                    interval_definition=(
                        "pointwise 95% percentile bootstrap across complete selected "
                        "seed-stage vectors; frozen dataset stream; 10000 resamples"
                    ),
                    interval_scope="pointwise, not simultaneous",
                    source_path=source_path,
                    source_sha256=source_sha256,
                )
            )
    return status, rows, summaries


def _setting_status_row(**values: Any) -> dict[str, Any]:
    row = {"schema_version": 1, "method_count": len(METHODS), **values}
    if set(row) != set(SETTING_STATUS_COLUMNS):
        raise RuntimeError("setting-status row fields differ")
    return row


def _stage_profile_row(
    *,
    setting_id: str,
    column_index: int,
    display_label: str,
    stratum: str,
    dataset: str,
    feedback_parameter: str,
    feedback_value: float,
    analysis_role: str,
    panel_status: str,
    confirmatory_ranking_included: bool,
    method: str,
    information_regime: str,
    target_adaptation_trajectories_per_seed: int | None,
    stage: int,
    n_prespecified: int,
    n_selected: int,
    coverage: float,
    coverage_interval: tuple[float, float],
    width: float,
    width_interval: tuple[float, float],
    interval_definition: str,
    interval_scope: str,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    coverage_lower, coverage_upper = coverage_interval
    width_lower, width_upper = width_interval
    row = {
        "schema_version": 1,
        "figure": CONTROLLED_STEM,
        "setting_id": setting_id,
        "column_index": column_index,
        "display_label": display_label,
        "stratum": stratum,
        "dataset": dataset,
        "feedback_parameter": feedback_parameter,
        "feedback_value": feedback_value,
        "analysis_role": analysis_role,
        "panel_status": panel_status,
        "confirmatory_ranking_included": confirmatory_ranking_included,
        "method": method,
        "information_regime": information_regime,
        "target_adaptation_trajectories_per_seed": (
            target_adaptation_trajectories_per_seed
        ),
        "stage_zero_based": stage,
        "n_prespecified": n_prespecified,
        "n_selected": n_selected,
        "metric_available": True,
        "coverage_target": TARGET,
        "coverage_mean": coverage,
        "coverage_ci95_lower": coverage_lower,
        "coverage_ci95_upper": coverage_upper,
        "coverage_deviation_from_target_pp": 100.0 * (coverage - TARGET),
        "coverage_deviation_ci95_lower_pp": 100.0 * (coverage_lower - TARGET),
        "coverage_deviation_ci95_upper_pp": 100.0 * (coverage_upper - TARGET),
        "normalized_width_mean": width,
        "normalized_width_ci95_lower": width_lower,
        "normalized_width_ci95_upper": width_upper,
        "normalized_width_definition": NORMALIZED_WIDTH_DEFINITION,
        "interval_definition": interval_definition,
        "interval_scope": interval_scope,
        "source_path": source_path,
        "source_sha256": source_sha256,
    }
    if set(row) != set(STAGE_PROFILE_COLUMNS):
        raise RuntimeError("stage-profile row fields differ")
    return row


def _method_summary_row(
    *,
    figure: str,
    setting_id: str,
    display_label: str,
    stratum: str,
    dataset: str,
    feedback_parameter: str,
    feedback_value: float | None,
    panel_status: str,
    analysis_role: str,
    confirmatory_ranking_included: bool,
    method: str,
    metric_available: bool,
    n_prespecified: int,
    n_selected: int | None,
    selection_rate: float | None,
    selection_interval: tuple[float | None, float | None],
    wsc: float | None,
    wsc_interval: tuple[float | None, float | None],
    worst_stage: int | None,
    mean_width: float | None,
    mean_width_interval: tuple[float | None, float | None],
    point_eligible: bool | None,
    wsc_interval_definition: str,
    mean_width_interval_definition: str,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "figure": figure,
        "setting_id": setting_id,
        "display_label": display_label,
        "stratum": stratum,
        "dataset": dataset,
        "feedback_parameter": feedback_parameter,
        "feedback_value": feedback_value,
        "panel_status": panel_status,
        "analysis_role": analysis_role,
        "confirmatory_ranking_included": confirmatory_ranking_included,
        "method": method,
        "metric_available": metric_available,
        "coverage_target": TARGET,
        "n_prespecified": n_prespecified,
        "n_selected": n_selected,
        "selection_rate": selection_rate,
        "selection_rate_ci95_lower": selection_interval[0],
        "selection_rate_ci95_upper": selection_interval[1],
        "wsc": wsc,
        "wsc_ci95_lower": wsc_interval[0],
        "wsc_ci95_upper": wsc_interval[1],
        "worst_stage_zero_based": worst_stage,
        "mean_normalized_width": mean_width,
        "mean_normalized_width_ci95_lower": mean_width_interval[0],
        "mean_normalized_width_ci95_upper": mean_width_interval[1],
        "normalized_width_definition": NORMALIZED_WIDTH_DEFINITION,
        "point_eligibility_rule": (
            "selection_rate>=0.95 and WSC>=0.90; unavailable if hard gate "
            "forbids science"
        ),
        "point_eligible": point_eligible,
        "primary_metric": "min_t mean_seed(target_coverage_seed_t)",
        "wsc_interval_definition": wsc_interval_definition,
        "mean_width_interval_definition": mean_width_interval_definition,
        "source_path": source_path,
        "source_sha256": source_sha256,
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in {"True", "False"}:
        return value == "True"
    raise RuntimeError(f"expected a stored boolean, found {value!r}")


def validate_controlled_render_source(
    status: pd.DataFrame, profiles: pd.DataFrame
) -> None:
    if tuple(status.columns) != SETTING_STATUS_COLUMNS:
        raise RuntimeError("setting-status schema differs")
    if tuple(profiles.columns) != STAGE_PROFILE_COLUMNS:
        raise RuntimeError("controlled stage-profile schema differs")
    if tuple(status["setting_id"]) != CONTROLLED_SETTING_IDS:
        raise RuntimeError("controlled five-setting order differs")
    if tuple(status["column_index"].astype(int)) != tuple(range(5)):
        raise RuntimeError("controlled setting column indices differ")
    if status["setting_id"].duplicated().any():
        raise RuntimeError("controlled setting statuses are duplicated")
    if not np.array_equal(
        status["method_count"].to_numpy(int),
        np.full(len(status), len(METHODS), dtype=int),
    ):
        raise RuntimeError("controlled setting method counts differ")
    expected_datasets = ("synthetic", *CLINICAL_DATASETS)
    if tuple(status["dataset"]) != expected_datasets:
        raise RuntimeError("controlled setting dataset order differs")
    expected_horizons = np.asarray(
        [HORIZONS[dataset] for dataset in expected_datasets], dtype=int
    )
    if not np.array_equal(status["horizon"].to_numpy(int), expected_horizons):
        raise RuntimeError("controlled dataset-native horizons differ")
    if bool(status.iloc[0]["uses_clinical_donor_kernel"]):
        raise RuntimeError("native Synthetic must not use a clinical donor kernel")
    if status.iloc[0]["feedback_parameter"] != "beta":
        raise RuntimeError("native Synthetic must remain on the beta scale")
    clinical = status.iloc[1:]
    if (
        set(clinical["feedback_parameter"]) != {"gamma"}
        or set(clinical["feedback_value"].astype(float)) != {-4.0}
        or not clinical["uses_clinical_donor_kernel"].astype(bool).all()
    ):
        raise RuntimeError("clinical controlled setting semantics differ")
    if status["signed_scale_comparable_across_strata"].astype(bool).any():
        raise RuntimeError("beta and signed gamma must not be declared comparable")
    for _, setting in status.iterrows():
        rows = profiles[profiles["setting_id"].eq(setting["setting_id"])]
        if setting["panel_status"] == "GATE_NO_GO":
            if not rows.empty or bool(setting["scientific_rows_saved"]):
                raise RuntimeError("hard gate cannot have stage-profile rows")
            continue
        expected = {
            (method, stage)
            for method in METHODS
            for stage in range(int(setting["horizon"]))
        }
        observed = set(zip(rows["method"], rows["stage_zero_based"]))
        if observed != expected or rows.empty:
            raise RuntimeError(f"curve grid differs for {setting['setting_id']}")
        if rows["stage_zero_based"].max() != int(setting["horizon"]) - 1:
            raise RuntimeError(f"curve horizon differs for {setting['setting_id']}")
        for field in (
            "column_index",
            "display_label",
            "stratum",
            "dataset",
            "feedback_parameter",
            "feedback_value",
            "panel_status",
            "confirmatory_ranking_included",
        ):
            observed_values = rows[field].drop_duplicates()
            if len(observed_values) != 1 or observed_values.iloc[0] != setting[field]:
                raise RuntimeError(
                    f"setting/profile join differs for {setting['setting_id']}/{field}"
                )
    if profiles.duplicated(["setting_id", "method", "stage_zero_based"]).any():
        raise RuntimeError("controlled stage-profile keys are duplicated")
    _validate_numeric_profile_columns(profiles)


def _validate_numeric_profile_columns(frame: pd.DataFrame) -> None:
    coverage = frame[
        ["coverage_ci95_lower", "coverage_mean", "coverage_ci95_upper"]
    ].to_numpy(float)
    widths = frame[
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
        raise RuntimeError("coverage profile values/intervals differ")
    if (
        not np.isfinite(widths).all()
        or np.any(widths <= 0.0)
        or np.any(widths[:, 0] > widths[:, 1])
        or np.any(widths[:, 1] > widths[:, 2])
    ):
        raise RuntimeError("width profile values/intervals differ")
    targets = frame["coverage_target"].to_numpy(float)
    if not np.array_equal(targets, np.full(len(frame), TARGET, dtype=float)):
        raise RuntimeError("coverage target must be exactly 0.90")
    for coverage_name, deviation_name in (
        ("coverage_mean", "coverage_deviation_from_target_pp"),
        ("coverage_ci95_lower", "coverage_deviation_ci95_lower_pp"),
        ("coverage_ci95_upper", "coverage_deviation_ci95_upper_pp"),
    ):
        expected_deviation = 100.0 * (
            frame[coverage_name].to_numpy(float) - targets
        )
        observed_deviation = frame[deviation_name].to_numpy(float)
        if not np.array_equal(observed_deviation, expected_deviation):
            raise RuntimeError(f"{deviation_name} does not match coverage and target")
    if set(frame["normalized_width_definition"]) != {NORMALIZED_WIDTH_DEFINITION}:
        raise RuntimeError("normalized-width definition differs")


def validate_method_summary(
    summary: pd.DataFrame,
    production_profiles: pd.DataFrame,
    setting_status: pd.DataFrame,
    controlled_profiles: pd.DataFrame,
) -> None:
    if tuple(summary.columns) != METHOD_SUMMARY_COLUMNS:
        raise RuntimeError("method-summary schema differs")
    production_ids = tuple(f"production_{dataset}" for dataset in PRODUCTION_DATASETS)
    expected_keys = {
        (figure, setting_id, method)
        for figure, setting_ids in (
            (PRODUCTION_STEM, production_ids),
            (CONTROLLED_STEM, CONTROLLED_SETTING_IDS),
        )
        for setting_id in setting_ids
        for method in METHODS
    }
    observed_keys = set(zip(summary["figure"], summary["setting_id"], summary["method"]))
    if observed_keys != expected_keys or len(summary) != len(expected_keys):
        raise RuntimeError("method-summary figure/setting/method grid differs")
    if summary.duplicated(["figure", "setting_id", "method"]).any():
        raise RuntimeError("method-summary keys are duplicated")
    if not np.array_equal(
        summary["coverage_target"].to_numpy(float),
        np.full(len(summary), TARGET, dtype=float),
    ):
        raise RuntimeError("method-summary coverage target must be exactly 0.90")
    if set(summary["normalized_width_definition"]) != {NORMALIZED_WIDTH_DEFINITION}:
        raise RuntimeError("method-summary normalized-width definition differs")
    if set(summary["primary_metric"]) != {
        "min_t mean_seed(target_coverage_seed_t)"
    }:
        raise RuntimeError("method-summary primary metric differs")

    controlled_status = setting_status.set_index("setting_id")
    for _, row in summary.iterrows():
        available = _as_bool(row["metric_available"])
        if row["figure"] == PRODUCTION_STEM:
            dataset = str(row["dataset"])
            stages = production_profiles[
                production_profiles["dataset"].eq(dataset)
                & production_profiles["method"].eq(row["method"])
            ].sort_values("stage_zero_based")
            expected_panel = "CURVES"
            expected_label = PRODUCTION_LABELS[dataset]
        else:
            setting = controlled_status.loc[row["setting_id"]]
            stages = controlled_profiles[
                controlled_profiles["setting_id"].eq(row["setting_id"])
                & controlled_profiles["method"].eq(row["method"])
            ].sort_values("stage_zero_based")
            expected_panel = setting["panel_status"]
            expected_label = setting["display_label"]
            if row["dataset"] != setting["dataset"]:
                raise RuntimeError("method-summary setting/dataset join differs")
        if row["panel_status"] != expected_panel or row["display_label"] != expected_label:
            raise RuntimeError("method-summary setting identity differs")
        numeric_names = (
            "n_selected",
            "selection_rate",
            "selection_rate_ci95_lower",
            "selection_rate_ci95_upper",
            "wsc",
            "wsc_ci95_lower",
            "wsc_ci95_upper",
            "worst_stage_zero_based",
            "mean_normalized_width",
            "mean_normalized_width_ci95_lower",
            "mean_normalized_width_ci95_upper",
        )
        if not available:
            if not stages.empty or any(not pd.isna(row[name]) for name in numeric_names):
                raise RuntimeError("unavailable method-summary row contains science values")
            continue
        if stages.empty or any(pd.isna(row[name]) for name in numeric_names):
            raise RuntimeError("available method-summary row lacks stage/scalar values")
        wsc = float(row["wsc"])
        wsc_low = float(row["wsc_ci95_lower"])
        wsc_high = float(row["wsc_ci95_upper"])
        mean_width = float(row["mean_normalized_width"])
        width_low = float(row["mean_normalized_width_ci95_lower"])
        width_high = float(row["mean_normalized_width_ci95_upper"])
        selection = float(row["selection_rate"])
        selection_low = float(row["selection_rate_ci95_lower"])
        selection_high = float(row["selection_rate_ci95_upper"])
        if not (
            0.0 <= wsc_low <= wsc <= wsc_high <= 1.0
            and 0.0 <= selection_low <= selection <= selection_high <= 1.0
            and 0.0 < width_low <= mean_width <= width_high
        ):
            raise RuntimeError("method-summary scalar interval differs")
        stage_coverage = stages["coverage_mean"].to_numpy(float)
        stage_width = stages["normalized_width_mean"].to_numpy(float)
        if not math.isclose(wsc, float(stage_coverage.min()), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("method-summary WSC differs from min stage-mean coverage")
        if int(row["worst_stage_zero_based"]) != int(stage_coverage.argmin()):
            raise RuntimeError("method-summary worst stage differs")
        if not math.isclose(
            mean_width, float(stage_width.mean()), rel_tol=0.0, abs_tol=5e-7
        ):
            raise RuntimeError("method-summary mean width differs from stage profile")
        expected_selection = int(row["n_selected"]) / int(row["n_prespecified"])
        if not math.isclose(selection, expected_selection, rel_tol=0.0, abs_tol=1e-14):
            raise RuntimeError("method-summary selection rate differs from counts")
        if "complete-seed-vector" not in str(row["wsc_interval_definition"]):
            raise RuntimeError("WSC CI must come from complete-seed-vector bootstrap")


def _validate_stage_vector(
    value: object, length: int, label: str, *, positive: bool = False
) -> None:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise RuntimeError(f"{label} must be a finite length-{length} vector")
    if positive and np.any(vector <= 0.0):
        raise RuntimeError(f"{label} must be positive")
    if not positive and np.any((vector < 0.0) | (vector > 1.0)):
        raise RuntimeError(f"{label} must be in [0,1]")


def _validate_interval_vector(
    value: object,
    points: object,
    length: int,
    label: str,
    *,
    positive: bool = False,
) -> None:
    intervals = np.asarray(value, dtype=np.float64)
    point_vector = np.asarray(points, dtype=np.float64)
    if intervals.shape != (length, 2) or not np.isfinite(intervals).all():
        raise RuntimeError(f"{label} must be a finite {length}x2 array")
    if np.any(intervals[:, 0] > point_vector) or np.any(point_vector > intervals[:, 1]):
        raise RuntimeError(f"{label} does not contain its point estimates")
    if positive and np.any(intervals <= 0.0):
        raise RuntimeError(f"{label} must be positive")
    if not positive and np.any((intervals < 0.0) | (intervals > 1.0)):
        raise RuntimeError(f"{label} must be in [0,1]")


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 6.2,
            "axes.labelsize": 6.2,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.62,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "svg.hashsalt": "scpcp-five-setting-stage-profiles-v1",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def render_production_figure(source: pd.DataFrame) -> plt.Figure:
    """Render Figure A: all five frozen production/native settings."""

    validate_production_profiles(source)
    figure, axes = plt.subplots(2, 5, figsize=(7.20, 3.82), sharex="col")
    figure.subplots_adjust(
        left=0.070,
        right=0.992,
        bottom=0.155,
        top=0.790,
        wspace=0.30,
        hspace=0.24,
    )
    coverage_limits = _coverage_limits(source)
    legend_handles: list[Line2D] = []
    for column, dataset in enumerate(PRODUCTION_DATASETS):
        group = source[source["dataset"].eq(dataset)]
        coverage_axis, width_axis = axes[:, column]
        _style_coverage_axis(coverage_axis, coverage_limits)
        for method in METHODS:
            rows = group[group["method"].eq(method)].sort_values("stage_zero_based")
            _plot_method_profiles(coverage_axis, width_axis, rows, method)
            if column == 0:
                legend_handles.append(_legend_handle(method))
        coverage_axis.set_title(PRODUCTION_LABELS[dataset], fontweight="bold", pad=3)
        coverage_axis.set_ylim(*coverage_limits)
        _set_width_limits(width_axis, group)
        _set_stage_axis(width_axis, HORIZONS[dataset])
        for axis in (coverage_axis, width_axis):
            axis.grid(axis="y", color="#E1E3E6", linewidth=0.42, alpha=0.8)
            axis.tick_params(width=0.58, length=2.1)
        if column == 0:
            coverage_axis.set_ylabel("Coverage deviation (pp)")
            width_axis.set_ylabel("Normalized width")
        width_axis.set_xlabel("Stage, t")
    figure.suptitle(
        "Production/native benchmark suite: stagewise profiles across all five datasets",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=8.3,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.925,
        "Frozen RQ1 settings; no controlled signed-γ intervention is used in this figure.",
        ha="left",
        va="top",
        fontsize=5.9,
        color="#42474D",
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=6,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.35,
    )
    figure.text(
        0.5,
        0.018,
        "All six canonical methods; selection is 100%. Bands shown for emphasized methods; all pointwise 95% intervals are in the source data.",
        ha="center",
        va="bottom",
        fontsize=5.25,
        color="#454A50",
    )
    return figure


def render_controlled_figure(
    status: pd.DataFrame, profiles: pd.DataFrame
) -> plt.Figure:
    """Render Figure B with curves, descriptive watermarks, or hard gate cards."""

    validate_controlled_render_source(status, profiles)
    figure, axes = plt.subplots(2, 5, figsize=(7.20, 3.92), sharex="col")
    figure.subplots_adjust(
        left=0.070,
        right=0.992,
        bottom=0.160,
        top=0.765,
        wspace=0.32,
        hspace=0.24,
    )
    coverage_limits = _coverage_limits(profiles)
    legend_handles = [_legend_handle(method) for method in METHODS]
    for column, setting_id in enumerate(CONTROLLED_SETTING_IDS):
        setting = status[status["setting_id"].eq(setting_id)].iloc[0]
        group = profiles[profiles["setting_id"].eq(setting_id)]
        coverage_axis, width_axis = axes[:, column]
        if setting["panel_status"] == "GATE_NO_GO":
            _render_gate_card(coverage_axis, width_axis, setting)
            continue
        _style_coverage_axis(coverage_axis, coverage_limits)
        if setting["panel_status"] == "CURVES_DESCRIPTIVE_ONLY":
            for axis in (coverage_axis, width_axis):
                axis.set_facecolor("#FFF4D6")
                axis.text(
                    0.5,
                    0.5,
                    "DESCRIPTIVE\nONLY",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#C18413",
                    alpha=0.20,
                    rotation=22,
                    zorder=0,
                )
        for method in METHODS:
            rows = group[group["method"].eq(method)].sort_values("stage_zero_based")
            if rows.empty:
                continue
            _plot_method_profiles(coverage_axis, width_axis, rows, method)
        coverage_axis.set_title(str(setting["display_label"]), fontweight="bold", pad=3)
        coverage_axis.set_ylim(*coverage_limits)
        _set_width_limits(width_axis, group)
        _set_stage_axis(width_axis, int(setting["horizon"]))
        for axis in (coverage_axis, width_axis):
            axis.grid(axis="y", color="#E1E3E6", linewidth=0.42, alpha=0.8)
            axis.tick_params(width=0.58, length=2.1)
        if column == 0:
            coverage_axis.set_ylabel("Coverage deviation (pp)")
            width_axis.set_ylabel("Normalized width")
        width_axis.set_xlabel("Stage, t")
    separator_x = (
        axes[0, 0].get_position().x1 + axes[0, 1].get_position().x0
    ) / 2.0
    figure.add_artist(
        mpl.lines.Line2D(
            [separator_x, separator_x],
            [0.14, 0.81],
            transform=figure.transFigure,
            color="#AEB6BE",
            linewidth=0.75,
            linestyle=(0, (3, 2)),
        )
    )
    figure.suptitle(
        "Gate-aware controlled stress grid: native Synthetic β=2 is separate from clinical γ=−4",
        x=0.01,
        y=0.988,
        ha="left",
        fontsize=8.2,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.921,
        "Left: independent synthetic DGP (no clinical donor). Right: dataset-native clinical donor kernels; a clinical hard preflight failure forbids science curves.",
        ha="left",
        va="top",
        fontsize=5.65,
        color="#42474D",
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        ncol=6,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.35,
    )
    figure.text(
        0.5,
        0.018,
        "Bands shown for emphasized methods; all pointwise intervals are in source data. Amber panels are descriptive-only/excluded from ranking.\nClinical stress is calibration-aligned, not natural performativity or a causal treatment effect.",
        ha="center",
        va="bottom",
        fontsize=5.15,
        color="#454A50",
    )
    return figure


def _plot_method_profiles(
    coverage_axis: plt.Axes,
    width_axis: plt.Axes,
    rows: pd.DataFrame,
    method: str,
) -> None:
    x = rows["stage_zero_based"].to_numpy(float)
    emphasized = method in {"Standard CP", "MFCS", "SC-PCP"}
    alpha = 1.0 if emphasized else 0.64
    linewidth = 1.25 if method == "SC-PCP" else 0.92 if emphasized else 0.72
    markersize = 3.4 if method == "SC-PCP" else 2.3 if emphasized else 1.9
    fields = (
        (
            coverage_axis,
            "coverage_deviation_from_target_pp",
            "coverage_deviation_ci95_lower_pp",
            "coverage_deviation_ci95_upper_pp",
        ),
        (
            width_axis,
            "normalized_width_mean",
            "normalized_width_ci95_lower",
            "normalized_width_ci95_upper",
        ),
    )
    for axis, point_name, lower_name, upper_name in fields:
        axis.plot(
            x,
            rows[point_name].to_numpy(float),
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=markersize,
            linewidth=linewidth,
            markeredgewidth=0.25,
            alpha=alpha,
            zorder=4 if method == "SC-PCP" else 3 if emphasized else 2,
        )
        if emphasized:
            axis.fill_between(
                x,
                rows[lower_name].to_numpy(float),
                rows[upper_name].to_numpy(float),
                color=METHOD_COLORS[method],
                alpha=0.075 if method == "SC-PCP" else 0.045,
                linewidth=0.0,
                zorder=1,
            )


def _style_coverage_axis(axis: plt.Axes, limits: tuple[float, float]) -> None:
    axis.axhspan(limits[0], 0.0, color="#F8E9E7", alpha=0.42, zorder=0)
    axis.axhline(0.0, color="#30353A", linestyle=(0, (3, 2)), linewidth=0.72)


def _coverage_limits(source: pd.DataFrame) -> tuple[float, float]:
    lower = float(source["coverage_deviation_ci95_lower_pp"].min())
    upper = float(source["coverage_deviation_ci95_upper_pp"].max())
    return (
        math.floor((lower - 0.15) * 2.0) / 2.0,
        math.ceil((upper + 0.15) * 2.0) / 2.0,
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
    ticks = list(range(horizon)) if horizon <= 6 else list(range(0, horizon, 2))
    axis.set_xticks(ticks)


def _render_gate_card(
    coverage_axis: plt.Axes, width_axis: plt.Axes, setting: pd.Series
) -> None:
    for axis in (coverage_axis, width_axis):
        axis.set_axis_off()
        axis.set_facecolor("#F4F2EF")
    coverage_axis.set_title(str(setting["display_label"]), fontweight="bold", pad=3)
    coverage_axis.text(
        0.5,
        0.44,
        "HARD GATE\nNO-GO",
        transform=coverage_axis.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#8A3B31",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#FAECE8",
            "edgecolor": "#C58B82",
            "linewidth": 0.8,
        },
    )
    available = setting["k0_fidelity_available"]
    available_text = "not reached" if pd.isna(available) else f"{int(available)}/{int(setting['prespecified_seeds'])} seeds passed"
    width_axis.text(
        0.5,
        0.63,
        f"{str(setting['hard_gate_reason']).replace('_', ' ')}\n{available_text}\n\nNo science rows\nNo curve · no ranking",
        transform=width_axis.transAxes,
        ha="center",
        va="center",
        fontsize=5.8,
        linespacing=1.35,
        color="#4B4642",
    )


def _legend_handle(method: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=METHOD_COLORS[method],
        linestyle=METHOD_LINESTYLES[method],
        marker=METHOD_MARKERS[method],
        markersize=3.2,
        linewidth=1.15 if method == "SC-PCP" else 0.85,
        label=method,
    )


def export_figure(
    figure: plt.Figure,
    *,
    title: str,
    work_stem: Path,
    tiff_dpi: int = 600,
    png_dpi: int = 240,
) -> None:
    creator = "SC-PCP five-setting stage-profile renderer"
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


def _write_figure_contract(
    path: Path,
    *,
    production_contract: Mapping[str, Any],
    synthetic_contract: Mapping[str, Any],
    clinical_contract: Mapping[str, Any],
    setting_status: pd.DataFrame,
    production_rows: int,
    controlled_rows: int,
    method_summary_rows: int,
) -> None:
    payload = {
        "schema_version": 1,
        "protocol": RENDER_PROTOCOL,
        "status": "complete",
        "backend": "Python/matplotlib only",
        "archetype": "two quantitative grids",
        "figures": {
            PRODUCTION_STEM: {
                "core_conclusion": (
                    "The complete frozen production/native suite exposes stagewise "
                    "coverage-efficiency behavior for all six canonical methods across "
                    "all five datasets, including the true six-stage MIMIC-CXR horizon."
                ),
                "setting_order": list(PRODUCTION_DATASETS),
                "panel_map": {
                    "top_row": "coverage deviation from 0.90 in percentage points",
                    "bottom_row": "normalized prediction width",
                },
                "controlled_gamma_used": False,
                "source_rows": production_rows,
                "coverage_target": TARGET,
                "normalized_width_definition": NORMALIZED_WIDTH_DEFINITION,
                "reviewer_risks": [
                    "These are frozen production/native RQ1 settings, not controlled gamma cells.",
                    "Stagewise intervals are pointwise and not simultaneous certificates.",
                    "The figure does not establish universal dominance or universal SOTA.",
                ],
            },
            CONTROLLED_STEM: {
                "core_conclusion": (
                    "A gate-aware juxtaposition separates native Synthetic beta=2 from "
                    "four dataset-native clinical gamma=-4 settings and displays a curve "
                    "only when the formal hard preflight permits science."
                ),
                "setting_order": list(CONTROLLED_SETTING_IDS),
                "panel_map": {
                    "top_row": "coverage deviation from 0.90 or hard-gate card",
                    "bottom_row": "normalized prediction width or hard-gate card",
                },
                "source_rows": controlled_rows,
                "coverage_target": TARGET,
                "normalized_width_definition": NORMALIZED_WIDTH_DEFINITION,
                "scalar_summary_rows": method_summary_rows,
                "wsc_ci_source_rule": (
                    "stored complete-seed-vector bootstrap WSC intervals only; "
                    "never inferred from pointwise stage intervals"
                ),
                "setting_status": setting_status.to_dict(orient="records"),
                "reviewer_risks": [
                    "Synthetic beta and clinical signed gamma are different scales and cannot be compared numerically.",
                    "A hard-gate card is absence of admissible science, not poor method performance.",
                    "Low donor overlap requires amber descriptive-only curves and exclusion from ranking.",
                    "The clinical kernel is a calibration-aligned controlled stress, not natural performativity or a causal treatment effect.",
                    "No finite-sample, distribution-free, PAC, data-conditional, or universal SOTA claim is supported.",
                ],
            },
        },
        "input_contracts": {
            "production": production_contract,
            "native_synthetic_beta2": synthetic_contract,
            "controlled_clinical_v2": clinical_contract,
        },
        "export_contract": {
            "canvas_inches": [7.20, 3.82, 7.20, 3.92],
            "work_formats": ["editable SVG", "TrueType PDF", "600-dpi TIFF", "240-dpi PNG"],
            "paper_files": sorted(PAPER_FILES),
            "paper_directory_policy": "PDF only",
            "source_data": [
                "production_stage_profiles.csv",
                "setting_status.csv",
                "stage_profiles.csv",
                "method_summary.csv",
            ],
        },
    }
    _write_json(path, payload)


def _write_qa(
    path: Path,
    *,
    production: pd.DataFrame,
    setting_status: pd.DataFrame,
    stage_profiles: pd.DataFrame,
    method_summary: pd.DataFrame,
) -> None:
    hard_gates = setting_status[setting_status["panel_status"].eq("GATE_NO_GO")]
    curve_settings = setting_status[setting_status["curves_rendered"].astype(bool)]
    lines = [
        "# Five-setting stage-profile QA",
        "",
        "- Backend exclusivity: Python/matplotlib generated every visual output and QA preview.",
        "- Archetype: two distinct quantitative grids; neither figure substitutes for the other.",
        "- Figure A: five production/native RQ1 settings, six canonical methods, coverage deviation plus normalized width; no controlled gamma.",
        f"- Figure A source grid: {len(production)} rows; horizons "
        + ", ".join(f"{name}=T{HORIZONS[name]}" for name in PRODUCTION_DATASETS)
        + ".",
        "- Figure B: native Synthetic beta=2 is a separate stratum; four clinical columns use dataset-native gamma=-4 controlled stress.",
        f"- Figure B curve settings: {', '.join(curve_settings['setting_id'])}.",
        f"- Figure B hard-gate settings: {', '.join(hard_gates['setting_id'])}.",
        f"- Figure B plotted source rows: {len(stage_profiles)}; hard-gate settings contribute exactly zero stage rows.",
        f"- Scalar audit table: {len(method_summary)} setting-method rows; WSC intervals are copied from the frozen complete-seed-vector bootstrap summaries, never derived from pointwise stage bands.",
        f"- Normalized width: `{NORMALIZED_WIDTH_DEFINITION}`.",
        "- MIMIC-CXR is T=6 in both schemas; no stage padding or duplicated tail is permitted.",
        "- A hard K0/support/structural NO-GO is rendered as a gate card with no method values and no ranking.",
        "- LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY, if present, is amber-watermarked and excluded from ranking.",
        "- All six canonical comparison names are fixed: Standard CP, ACI, MFCS, SPCI, PRC, SC-PCP.",
        "- Coverage target is 0.90; primary scalar semantics remain min_t mean_seed, never mean_seed(min_t).",
        "- Bands are pointwise 95% intervals and are not simultaneous confidence bands or finite-sample certificates.",
        "- Typography: Times New Roman with serif fallback; SVG text remains editable and PDF uses TrueType fonts.",
        "- Accessibility: method identity uses color, marker, and line-style redundancy; hard gates use text and shape, not color alone.",
        "- No raster source image, image manipulation, model fit, rollout, scientific RNG, or scientific seed was used.",
        "- Paper output contains exactly two PDF files; work output contains sources, contract, QA, manifest, SVG, PDF, TIFF, and PNG.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_render_manifest(path: Path, *, work_root: Path, paper_root: Path) -> None:
    work_files = {
        item.name: _file_contract(item)
        for item in sorted(work_root.iterdir())
        if item.is_file() and item.name != path.name
    }
    if any(paper_root.iterdir()):
        raise RuntimeError("paper staging directory must be empty before work commit")
    paper_files = {
        name: _file_contract(work_root / name) for name in sorted(PAPER_FILES)
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


def _write_work_complete(work_root: Path) -> None:
    manifest_hash = _file_sha256(work_root / "render_manifest.json")
    (work_root / "COMPLETE").write_text(
        f"complete render_manifest_sha256={manifest_hash}\n", encoding="utf-8"
    )


def _copy_paper_from_completed_work(work_root: Path, paper_root: Path) -> None:
    expected_marker = (
        "complete render_manifest_sha256="
        f"{_file_sha256(work_root / 'render_manifest.json')}\n"
    )
    if (work_root / "COMPLETE").read_text(encoding="utf-8") != expected_marker:
        raise RuntimeError("paper copy requires a completed work bundle")
    manifest = _read_json(work_root / "render_manifest.json")
    paper_contract = _mapping(manifest.get("paper_files"), "paper file plan")
    if set(paper_contract) != PAPER_FILES or any(paper_root.iterdir()):
        raise RuntimeError("paper copy plan/staging directory differs")
    for name in sorted(PAPER_FILES):
        source = work_root / name
        _validate_file_contract(source, paper_contract[name])
        shutil.copyfile(source, paper_root / name)
        _validate_file_contract(paper_root / name, paper_contract[name])


def _publish_bundles(
    *,
    staged_work: Path,
    staged_paper: Path,
    work_output: Path,
    paper_output: Path,
) -> None:
    os.replace(staged_work, work_output)
    try:
        os.replace(staged_paper, paper_output)
    except BaseException as error:
        if paper_output.exists():
            raise RuntimeError(
                "paper publication failed after creating an ambiguous partial target"
            ) from error
        raise RuntimeError(
            "paper publication failed; completed work was retained and no paper "
            "directory was published"
        ) from error


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
    expected_complete = (
        "complete render_manifest_sha256="
        f"{_file_sha256(work_root / 'render_manifest.json')}\n"
    )
    if (work_root / "COMPLETE").read_text(encoding="utf-8") != expected_complete:
        raise RuntimeError("work COMPLETE marker differs")
    for stem in (PRODUCTION_STEM, CONTROLLED_STEM):
        svg = (work_root / f"{stem}.svg").read_text(encoding="utf-8")
        if "<text" not in svg or "Times New Roman" not in svg:
            raise RuntimeError(f"{stem} SVG text/font contract differs")
        if not (work_root / f"{stem}.pdf").read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"{stem} work PDF header is malformed")
        if not (paper_root / f"{stem}.pdf").read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"{stem} paper PDF header is malformed")
        if not (work_root / f"{stem}.png").read_bytes().startswith(b"\x89PNG"):
            raise RuntimeError(f"{stem} PNG header is malformed")
        if (work_root / f"{stem}.tiff").read_bytes()[:4] not in {
            b"II*\x00",
            b"MM\x00*",
        }:
            raise RuntimeError(f"{stem} TIFF header is malformed")
    production = pd.read_csv(
        work_root / "production_stage_profiles.csv", float_precision="round_trip"
    )
    status = pd.read_csv(
        work_root / "setting_status.csv", float_precision="round_trip"
    )
    controlled = pd.read_csv(
        work_root / "stage_profiles.csv", float_precision="round_trip"
    )
    method_summary = pd.read_csv(
        work_root / "method_summary.csv", float_precision="round_trip"
    )
    validate_production_profiles(production)
    validate_controlled_render_source(status, controlled)
    validate_method_summary(method_summary, production, status, controlled)
    for name, frame in (
        ("production_stage_profiles.csv", production),
        ("setting_status.csv", status),
        ("stage_profiles.csv", controlled),
        ("method_summary.csv", method_summary),
    ):
        _validate_csv_roundtrip(work_root / name, frame)
    manifest = _read_json(work_root / "render_manifest.json")
    if (
        manifest.get("protocol") != RENDER_PROTOCOL
        or manifest.get("status") != "complete"
        or set(_mapping(manifest.get("paper_files"), "manifest paper")) != PAPER_FILES
    ):
        raise RuntimeError("render manifest contract differs")
    for group, root in (("work_files", work_root), ("paper_files", paper_root)):
        for name, contract in _mapping(manifest[group], group).items():
            _validate_file_contract(root / name, contract)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _validate_csv_roundtrip(path: Path, source: pd.DataFrame) -> None:
    restored = pd.read_csv(path, float_precision="round_trip")
    if tuple(restored.columns) != tuple(source.columns) or len(restored) != len(source):
        raise RuntimeError(f"CSV roundtrip schema/row count differs: {path.name}")
    for column in source.columns:
        nonmissing = source[column].dropna()
        numeric = not nonmissing.empty and all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in nonmissing
        )
        if numeric:
            before = pd.to_numeric(source[column], errors="raise").to_numpy(float)
            after = pd.to_numeric(restored[column], errors="raise").to_numpy(float)
            if not np.array_equal(before, after, equal_nan=True):
                raise RuntimeError(f"CSV numeric roundtrip differs: {path.name}/{column}")
            continue
        before_text = source[column].map(_csv_text_value).tolist()
        after_text = restored[column].map(_csv_text_value).tolist()
        if before_text != after_text:
            raise RuntimeError(f"CSV text roundtrip differs: {path.name}/{column}")


def _csv_text_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    return str(value)


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


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_contract(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _validate_file_contract(path: Path, contract: object) -> None:
    resolved = _mapping(contract, f"file contract {path.name}")
    if (
        not path.is_file()
        or resolved.get("bytes") != path.stat().st_size
        or resolved.get("sha256") != _file_sha256(path)
    ):
        raise RuntimeError(f"file contract differs: {path}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
