from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import render_complete_coverage_reporting as renderer


@pytest.fixture(scope="module")
def sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    return renderer.build_reporting_sources(
        production_input=renderer.DEFAULT_PRODUCTION_INPUT,
        clinical_input=renderer.DEFAULT_CLINICAL_INPUT,
    )


def test_source_grids_cover_complete_reporting_without_beta_gamma_substitution(
    sources: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    status, stage, scalar, contract = sources

    assert len(status) == 14
    assert len(stage) == 756
    assert len(scalar) == 84
    assert status.groupby("reporting_family").size().to_dict() == {
        "clinical_gamma_minus4_main": 4,
        "mimic_iv_v2_signed_gamma_supplement": 5,
        "production_no_gamma_supplement": 5,
    }
    assert stage.groupby("reporting_family").size().to_dict() == {
        "clinical_gamma_minus4_main": 72,
        "mimic_iv_v2_signed_gamma_supplement": 360,
        "production_no_gamma_supplement": 324,
    }
    assert scalar.groupby("reporting_family").size().to_dict() == {
        "clinical_gamma_minus4_main": 24,
        "mimic_iv_v2_signed_gamma_supplement": 30,
        "production_no_gamma_supplement": 30,
    }
    main = status[status["reporting_family"].eq("clinical_gamma_minus4_main")]
    assert tuple(main["dataset"]) == renderer.CLINICAL_DATASETS
    assert set(main["feedback_parameter"]) == {"gamma"}
    assert set(main["feedback_value"].astype(float)) == {-4.0}
    assert "synthetic" not in set(main["dataset"])
    assert contract["controlled_clinical"]["adapter"] == (
        "exact controlled_clinical_extension_v2"
    )
    production = status[
        status["reporting_family"].eq("production_no_gamma_supplement")
    ]
    assert not production["confirmatory"].astype(bool).any()


def test_main_gamma_minus4_has_one_curve_and_exact_gate_cards(
    sources: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    status, stage, scalar, _ = sources
    main_status = status[status["reporting_family"].eq("clinical_gamma_minus4_main")]
    observed = dict(zip(main_status["dataset"], main_status["panel_status"]))
    assert observed == {
        "mimic_iv": "CURVES",
        "eicu": "GATE_NO_GO",
        "inspire": "GATE_NO_GO",
        "mimic_cxr": "GATE_NO_GO",
    }
    k0 = dict(zip(main_status["dataset"], main_status["k0_fidelity_available"]))
    assert k0 == {"mimic_iv": 20, "eicu": 12, "inspire": 13, "mimic_cxr": 10}
    main_stage = stage[stage["reporting_family"].eq("clinical_gamma_minus4_main")]
    assert set(main_stage["dataset"]) == {"mimic_iv"}
    assert len(main_stage) == 12 * len(renderer.METHODS)
    main_scalar = scalar[scalar["reporting_family"].eq("clinical_gamma_minus4_main")]
    for dataset in ("eicu", "inspire", "mimic_cxr"):
        unavailable = main_scalar[main_scalar["dataset"].eq(dataset)]
        assert len(unavailable) == len(renderer.METHODS)
        assert not unavailable["metric_available"].astype(bool).any()
        assert unavailable["wsc"].isna().all()
        assert unavailable["mean_coverage"].isna().all()
        assert unavailable["mean_normalized_width"].isna().all()


def test_every_available_scalar_reconciles_with_stage_profiles(
    sources: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    status, stage, scalar, _ = sources
    renderer.validate_reporting_sources(status, stage, scalar)

    available = scalar[scalar["metric_available"].astype(bool)]
    for row in available.itertuples(index=False):
        profiles = stage[
            stage["reporting_family"].eq(row.reporting_family)
            & stage["setting_id"].eq(row.setting_id)
            & stage["method"].eq(row.method)
        ].sort_values("stage_zero_based")
        coverage = profiles["coverage_mean"].to_numpy(float)
        width = profiles["normalized_width_mean"].to_numpy(float)
        assert row.wsc == pytest.approx(float(coverage.min()), abs=1e-12)
        assert int(row.worst_stage_zero_based) == int(coverage.argmin())
        mean_tolerance = (
            5e-7
            if row.reporting_family == "production_no_gamma_supplement"
            else 1e-12
        )
        assert row.mean_coverage == pytest.approx(
            float(coverage.mean()), abs=mean_tolerance
        )
        assert row.mean_normalized_width == pytest.approx(
            float(width.mean()), abs=5e-7
        )


def test_mimic_gamma_minus4_contains_every_requested_scalar_field(
    sources: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    _, _, scalar, _ = sources
    rows = scalar[
        scalar["reporting_family"].eq("clinical_gamma_minus4_main")
        & scalar["dataset"].eq("mimic_iv")
    ].set_index("method")

    assert set(rows.index) == set(renderer.METHODS)
    assert rows.loc["Standard CP", "wsc"] == pytest.approx(0.8635799795389175)
    assert rows.loc["SC-PCP", "wsc"] == pytest.approx(0.9008874803781509)
    assert int(rows.loc["SC-PCP", "worst_stage_zero_based"]) == 1
    assert rows.loc["SC-PCP", "mean_coverage"] == pytest.approx(
        0.9033278939624627
    )
    assert rows.loc["SC-PCP", "mean_normalized_width"] == pytest.approx(
        5.067080865303676
    )
    assert rows["selection_rate_ci95_lower"].notna().all()
    assert rows["selection_rate_ci95_upper"].notna().all()
    assert rows.loc["ACI", "target_adaptation_trajectories_per_seed"] == 2_000
    assert rows.loc["SC-PCP", "target_adaptation_trajectories_per_seed"] == 0
    assert set(rows["evaluation_trajectories_per_seed"]) == {20_000}


def test_signed_supplement_has_all_five_v2_cells_and_no_ranking_marker(
    sources: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    status, stage, scalar, _ = sources
    signed_status = status[
        status["reporting_family"].eq("mimic_iv_v2_signed_gamma_supplement")
    ]
    assert tuple(signed_status["feedback_value"].astype(float)) == renderer.SIGNED_GAMMAS
    assert signed_status["confirmatory"].astype(bool).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    signed_stage = stage[
        stage["reporting_family"].eq("mimic_iv_v2_signed_gamma_supplement")
    ]
    signed_scalar = scalar[
        scalar["reporting_family"].eq("mimic_iv_v2_signed_gamma_supplement")
    ]
    assert len(signed_stage) == 5 * 6 * 12
    assert len(signed_scalar) == 5 * 6
    descriptive = signed_scalar[~signed_scalar["confirmatory"].astype(bool)]
    assert descriptive["point_eligible"].isna().all()


def test_unknown_clinical_protocol_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "unknown"
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps({"protocol": "controlled_clinical_extension_v3_unfrozen"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="register an exact protocol adapter"):
        renderer.load_clinical_sources(root)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("wsc", "WSC differs"),
        ("mean_coverage", "MeanCov differs"),
        ("worst_stage_zero_based", "worst stage differs"),
    ],
)
def test_scalar_consistency_tampering_fails_closed(
    sources: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]],
    field: str,
    message: str,
) -> None:
    status, stage, scalar, _ = sources
    malformed = scalar.copy()
    row = malformed[
        malformed["reporting_family"].eq("clinical_gamma_minus4_main")
        & malformed["dataset"].eq("mimic_iv")
        & malformed["method"].eq("SC-PCP")
    ].index[0]
    malformed.loc[row, field] = (
        11
        if field == "worst_stage_zero_based"
        else float(malformed.loc[row, field]) + 1e-5
    )

    with pytest.raises(RuntimeError, match=message):
        renderer.validate_reporting_sources(status, stage, malformed)


def test_hard_gate_stage_injection_fails_closed(
    sources: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    status, stage, scalar, _ = sources
    injected = stage.iloc[[0]].copy()
    injected["reporting_family"] = "clinical_gamma_minus4_main"
    injected["setting_id"] = "eicu_gamma_minus4"
    injected["dataset"] = "eicu"
    injected["display_label"] = "eICU"
    injected["setting_type"] = "dataset_native_clinical_controlled"
    injected["feedback_parameter"] = "gamma"
    injected["feedback_value"] = -4.0
    injected["panel_status"] = "GATE_NO_GO"
    injected["confirmatory"] = False
    malformed = pd.concat([stage, injected], ignore_index=True)

    with pytest.raises(RuntimeError, match="hard-gate status cannot have"):
        renderer.validate_reporting_sources(status, malformed, scalar)


def test_main_figure_uses_six_explicit_errorbar_series_without_interval_fills(
    sources: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    status, stage, _, _ = sources
    renderer.apply_publication_style()
    figure = renderer.render_gamma_minus4_stagewise(status, stage)
    mimic_coverage = figure.axes[0]
    mimic_width = figure.axes[4]

    assert sum(isinstance(container, ErrorbarContainer) for container in mimic_coverage.containers) == 6
    assert sum(isinstance(container, ErrorbarContainer) for container in mimic_width.containers) == 6
    assert not any(isinstance(item, PolyCollection) for item in mimic_coverage.collections)
    assert not any(isinstance(item, PolyCollection) for item in mimic_width.collections)
    gate_text = " ".join(
        text.get_text() for axis in figure.axes[1:] for text in axis.texts
    )
    assert "HARD GATE" in gate_text
    assert "12/20 seeds passed" in gate_text
    assert "13/20 seeds passed" in gate_text
    assert "10/20 seeds passed" in gate_text
    plt.close(figure)


def test_render_bundle_is_pdf_only_and_deterministic(tmp_path: Path) -> None:
    first_work = tmp_path / "work_a"
    first_paper = tmp_path / "paper_a"
    second_work = tmp_path / "work_b"
    second_paper = tmp_path / "paper_b"

    renderer.render_report(
        production_input=renderer.DEFAULT_PRODUCTION_INPUT,
        clinical_input=renderer.DEFAULT_CLINICAL_INPUT,
        work_output=first_work,
        paper_output=first_paper,
    )
    renderer.render_report(
        production_input=renderer.DEFAULT_PRODUCTION_INPUT,
        clinical_input=renderer.DEFAULT_CLINICAL_INPUT,
        work_output=second_work,
        paper_output=second_paper,
    )

    assert {path.name for path in first_work.iterdir()} == renderer.WORK_FILES
    assert {path.name for path in first_paper.iterdir()} == renderer.PAPER_FILES
    assert all(path.suffix == ".pdf" for path in first_paper.iterdir())
    renderer.validate_rendered_outputs(first_work, first_paper)
    renderer.validate_rendered_outputs(second_work, second_paper)
    for name in sorted(renderer.WORK_FILES - {"COMPLETE", "render_manifest.json"}):
        assert _sha256(first_work / name) == _sha256(second_work / name), name
    for name in sorted(renderer.PAPER_FILES):
        assert _sha256(first_paper / name) == _sha256(second_paper / name), name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
