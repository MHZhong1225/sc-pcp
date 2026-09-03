from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import render_five_setting_stage_profiles as renderer


def _clinical_curve_summary(
    *, dataset: str, horizon: int, descriptive_only: bool
) -> dict[str, object]:
    methods = {}
    for method_index, method in enumerate(renderer.METHODS):
        coverage = [
            0.895 + 0.001 * method_index + 0.0002 * stage
            for stage in range(horizon)
        ]
        width = [
            1.1 + 0.05 * method_index + 0.01 * stage
            for stage in range(horizon)
        ]
        methods[method] = {
            "n_selected": 20,
            "selection_rate": 1.0,
            "selection_rate_ci95": [0.84, 1.0],
            "target_adaptation_trajectories_per_seed": (
                2_000 if method in {"ACI", "SPCI", "PRC"} else 0
            ),
            "target_coverage_by_stage": coverage,
            "target_coverage_by_stage_ci95": [
                [value - 0.002, value + 0.002] for value in coverage
            ],
            "target_normalized_width_by_stage": width,
            "target_normalized_width_by_stage_ci95": [
                [value - 0.02, value + 0.02] for value in width
            ],
            "target_marginal_worst_coverage": min(coverage),
            "target_wsc_ci95": [min(coverage) - 0.002, min(coverage) + 0.002],
            "target_worst_stage_zero_based": coverage.index(min(coverage)),
            "mean_target_normalized_width": sum(width) / horizon,
            "mean_target_normalized_width_ci95": [
                sum(width) / horizon - 0.02,
                sum(width) / horizon + 0.02,
            ],
            "point_eligible": (None if descriptive_only else min(coverage) >= 0.90),
        }
    interpretation = (
        "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        if descriptive_only
        else "EMPIRICAL_OVERLAP_SCREEN_PASSED"
    )
    return {
        "dataset": dataset,
        "interpretation_status": interpretation,
        "methods": list(renderer.METHODS),
        "primary_metric": "min_t mean_seed(target_coverage_seed_t)",
        "aggregates": [
            {
                "gamma": -4.0,
                "analysis_role": (
                    "descriptive_signed_control_curve"
                    if descriptive_only
                    else "confirmatory_gamma_minus_4_endpoint"
                ),
                "n_prespecified_seeds": 20,
                "methods": methods,
            }
        ],
    }


def _adapt_curve_fixture(
    *, dataset: str = "mimic_cxr", horizon: int = 6, descriptive_only: bool = True
) -> tuple[
    dict[str, object], list[dict[str, object]], list[dict[str, object]]
]:
    interpretation = (
        "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        if descriptive_only
        else "EMPIRICAL_OVERLAP_SCREEN_PASSED"
    )
    panel = "CURVES_DESCRIPTIVE_ONLY" if descriptive_only else "CURVES"
    return renderer.adapt_clinical_dataset(
        dataset=dataset,
        column_index=4,
        horizon=horizon,
        prespecified_seeds=20,
        gate={
            "panel_status": panel,
            "interpretation_status": interpretation,
        },
        final={
            "status": "COMPLETE",
            "scientific_rows_saved": True,
            "interpretation_status": interpretation,
        },
        support_summary={"n_available": 20},
        k0_summary={"n_available": 20},
        science_summary=_clinical_curve_summary(
            dataset=dataset,
            horizon=horizon,
            descriptive_only=descriptive_only,
        ),
        source_path=f"fixture/{dataset}/science/summary.json",
        source_sha256="a" * 64,
    )


def test_production_adapter_retains_all_five_native_settings_and_no_gamma() -> None:
    profiles, summaries, contract = renderer.load_production_profiles(
        renderer.DEFAULT_PRODUCTION_INPUT
    )

    assert len(profiles) == sum(renderer.HORIZONS.values()) * len(renderer.METHODS)
    assert tuple(profiles["dataset"].drop_duplicates()) == renderer.PRODUCTION_DATASETS
    assert not profiles["controlled_gamma_used"].astype(bool).any()
    assert set(
        profiles.loc[profiles["dataset"].eq("mimic_cxr"), "stage_zero_based"]
    ) == set(range(6))
    assert set(profiles["method"]) == set(renderer.METHODS)
    assert len(summaries) == 5 * len(renderer.METHODS)
    assert summaries["wsc_ci95_lower"].notna().all()
    assert contract["controlled_gamma_used"] is False


def test_native_synthetic_adapter_preserves_separate_beta_stratum() -> None:
    status, profiles, summaries, contract = renderer.load_native_synthetic(
        renderer.DEFAULT_SYNTHETIC_INPUT
    )

    assert len(status) == 1
    assert status.iloc[0]["feedback_parameter"] == "beta"
    assert status.iloc[0]["feedback_value"] == pytest.approx(2.0)
    assert status.iloc[0]["uses_clinical_donor_kernel"] == False
    assert status.iloc[0]["signed_scale_comparable_across_strata"] == False
    assert status.iloc[0]["confirmatory_ranking_included"] == False
    assert len(profiles) == 12 * len(renderer.METHODS)
    assert len(summaries) == len(renderer.METHODS)
    assert summaries["wsc_ci95_lower"].notna().all()
    assert set(profiles["method"]) == set(renderer.METHODS)
    assert "beta is not the signed gamma scale" in contract["required_disambiguator"]


def test_incomplete_clinical_root_fails_before_reading_science(tmp_path: Path) -> None:
    clinical = tmp_path / "clinical"
    clinical.mkdir()
    (clinical / "summary.json").write_text("not-json", encoding="utf-8")
    (clinical / "mimic_iv").mkdir()
    (clinical / "mimic_iv" / "science").mkdir()
    (clinical / "mimic_iv" / "science" / "summary.json").write_text(
        "not-json", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="root COMPLETE is required"):
        renderer.load_complete_clinical(clinical)


def test_hard_k0_no_go_has_gate_card_status_and_zero_stage_rows() -> None:
    status, rows, summaries = renderer.adapt_clinical_dataset(
        dataset="eicu",
        column_index=2,
        horizon=12,
        prespecified_seeds=20,
        gate={
            "panel_status": "GATE_NO_GO",
            "reason": "K0_FIDELITY_NO_GO",
        },
        final={
            "status": "K0_FIDELITY_NO_GO",
            "scientific_rows_saved": False,
        },
        support_summary={"n_available": 20},
        k0_summary={"n_available": 12},
        science_summary=None,
        source_path="fixture/eicu/NO_GO.json",
        source_sha256="b" * 64,
    )

    assert rows == []
    assert len(summaries) == len(renderer.METHODS)
    assert all(row["metric_available"] is False for row in summaries)
    assert all(row["wsc"] is None for row in summaries)
    assert status["panel_status"] == "GATE_NO_GO"
    assert status["k0_fidelity_available"] == 12
    assert status["curves_rendered"] is False
    assert status["confirmatory_ranking_included"] is False
    assert status["ranking_status"] == "EXCLUDED_HARD_GATE_NO_GO"


def test_descriptive_cxr_fixture_is_t6_without_padding_and_excluded_from_ranking() -> None:
    status, rows, summaries = _adapt_curve_fixture()

    assert status["panel_status"] == "CURVES_DESCRIPTIVE_ONLY"
    assert status["confirmatory_ranking_included"] is False
    assert status["ranking_status"] == "EXCLUDED_LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
    assert len(rows) == 6 * len(renderer.METHODS)
    assert len(summaries) == len(renderer.METHODS)
    assert {row["stage_zero_based"] for row in rows} == set(range(6))
    assert max(row["stage_zero_based"] for row in rows) == 5
    assert all(row["analysis_role"] == "descriptive_signed_control_curve" for row in rows)


def test_cxr_adapter_rejects_t12_padding_contract() -> None:
    with pytest.raises(RuntimeError, match="adapter identity"):
        _adapt_curve_fixture(horizon=12)


def test_controlled_validator_rejects_noncanonical_method_grid() -> None:
    synthetic_status, synthetic_profiles, _, _ = renderer.load_native_synthetic(
        renderer.DEFAULT_SYNTHETIC_INPUT
    )
    clinical_status, clinical_profiles, _, _ = renderer.load_complete_clinical(
        renderer.DEFAULT_CLINICAL_INPUT
    )
    status = pd.concat([synthetic_status, clinical_status], ignore_index=True).loc[
        :, renderer.SETTING_STATUS_COLUMNS
    ]
    profiles = pd.concat(
        [synthetic_profiles, clinical_profiles], ignore_index=True
    ).loc[:, renderer.STAGE_PROFILE_COLUMNS]
    malformed = profiles.copy()
    malformed.loc[malformed["setting_id"].eq("synthetic_beta2"), "method"] = "Bogus"

    with pytest.raises(RuntimeError, match="curve grid differs"):
        renderer.validate_controlled_render_source(status, malformed)


def test_profile_validators_reject_target_and_deviation_inconsistency() -> None:
    production, _, _ = renderer.load_production_profiles(
        renderer.DEFAULT_PRODUCTION_INPUT
    )
    malformed_production = production.copy()
    malformed_production.loc[0, "coverage_target"] = 0.81
    with pytest.raises(RuntimeError, match="exactly 0.90"):
        renderer.validate_production_profiles(malformed_production)

    synthetic_status, synthetic_profiles, _, _ = renderer.load_native_synthetic(
        renderer.DEFAULT_SYNTHETIC_INPUT
    )
    clinical_status, clinical_profiles, _, _ = renderer.load_complete_clinical(
        renderer.DEFAULT_CLINICAL_INPUT
    )
    status = pd.concat([synthetic_status, clinical_status], ignore_index=True).loc[
        :, renderer.SETTING_STATUS_COLUMNS
    ]
    profiles = pd.concat(
        [synthetic_profiles, clinical_profiles], ignore_index=True
    ).loc[:, renderer.STAGE_PROFILE_COLUMNS]
    malformed_controlled = profiles.copy()
    malformed_controlled.loc[0, "coverage_deviation_from_target_pp"] = 123.0
    with pytest.raises(RuntimeError, match="does not match coverage"):
        renderer.validate_controlled_render_source(status, malformed_controlled)


def test_descriptive_render_uses_amber_watermark_and_true_t6_ticks() -> None:
    synthetic_status, synthetic_profiles, _, _ = renderer.load_native_synthetic(
        renderer.DEFAULT_SYNTHETIC_INPUT
    )
    clinical_status, clinical_profiles, _, _ = renderer.load_complete_clinical(
        renderer.DEFAULT_CLINICAL_INPUT
    )
    cxr_status, cxr_rows, _ = _adapt_curve_fixture()
    clinical_status = clinical_status[
        ~clinical_status["dataset"].eq("mimic_cxr")
    ]
    status = pd.concat(
        [synthetic_status, clinical_status, pd.DataFrame([cxr_status])],
        ignore_index=True,
    ).sort_values("column_index").loc[:, renderer.SETTING_STATUS_COLUMNS]
    profiles = pd.concat(
        [synthetic_profiles, clinical_profiles, pd.DataFrame(cxr_rows)],
        ignore_index=True,
    ).loc[:, renderer.STAGE_PROFILE_COLUMNS]
    renderer.validate_controlled_render_source(status, profiles)
    renderer.apply_publication_style()
    figure = renderer.render_controlled_figure(status, profiles)
    cxr_top = figure.axes[4]
    cxr_bottom = figure.axes[9]

    assert any(text.get_text() == "DESCRIPTIVE\nONLY" for text in cxr_top.texts)
    assert max(cxr_bottom.get_xticks()) == 5
    assert cxr_top.get_facecolor()[:3] == pytest.approx((1.0, 244 / 255, 214 / 255))
    plt.close(figure)


def test_completed_formal_clinical_bundle_recomputes_to_one_curve_and_three_gates() -> None:
    status, profiles, summaries, contract = renderer.load_complete_clinical(
        renderer.DEFAULT_CLINICAL_INPUT
    )

    observed = dict(zip(status["dataset"], status["panel_status"]))
    assert observed == {
        "mimic_iv": "CURVES",
        "eicu": "GATE_NO_GO",
        "inspire": "GATE_NO_GO",
        "mimic_cxr": "GATE_NO_GO",
    }
    k0 = dict(zip(status["dataset"], status["k0_fidelity_available"]))
    assert k0 == {"mimic_iv": 20, "eicu": 12, "inspire": 13, "mimic_cxr": 10}
    assert set(profiles["setting_id"]) == {"mimic_iv_gamma_minus4"}
    assert len(profiles) == 12 * len(renderer.METHODS)
    assert len(summaries) == 4 * len(renderer.METHODS)
    assert summaries["metric_available"].sum() == len(renderer.METHODS)
    assert contract["protocol"] == "controlled_clinical_extension_v2"


def test_render_bundle_has_two_distinct_figures_and_pdf_only_paper(
    tmp_path: Path,
) -> None:
    production, production_summary, production_contract = renderer.load_production_profiles(
        renderer.DEFAULT_PRODUCTION_INPUT
    )
    synthetic_status, synthetic_profiles, synthetic_summary, synthetic_contract = (
        renderer.load_native_synthetic(renderer.DEFAULT_SYNTHETIC_INPUT)
    )
    clinical_status, clinical_profiles, clinical_summary, clinical_contract = (
        renderer.load_complete_clinical(renderer.DEFAULT_CLINICAL_INPUT)
    )
    status = pd.concat(
        [synthetic_status, clinical_status], ignore_index=True
    ).loc[:, renderer.SETTING_STATUS_COLUMNS]
    profiles = pd.concat(
        [synthetic_profiles, clinical_profiles], ignore_index=True
    ).loc[:, renderer.STAGE_PROFILE_COLUMNS]
    method_summary = pd.concat(
        [production_summary, synthetic_summary, clinical_summary], ignore_index=True
    ).loc[:, renderer.METHOD_SUMMARY_COLUMNS]
    work = tmp_path / "work"
    paper = tmp_path / "paper"
    work.mkdir()
    paper.mkdir()
    renderer._write_csv(work / "production_stage_profiles.csv", production)
    renderer._write_csv(work / "setting_status.csv", status)
    renderer._write_csv(work / "stage_profiles.csv", profiles)
    renderer._write_csv(work / "method_summary.csv", method_summary)
    renderer._write_figure_contract(
        work / "figure_contract.json",
        production_contract=production_contract,
        synthetic_contract=synthetic_contract,
        clinical_contract=clinical_contract,
        setting_status=status,
        production_rows=len(production),
        controlled_rows=len(profiles),
        method_summary_rows=len(method_summary),
    )
    renderer._write_qa(
        work / "figure_qa.md",
        production=production,
        setting_status=status,
        stage_profiles=profiles,
        method_summary=method_summary,
    )
    renderer.apply_publication_style()
    renderer.export_figure(
        renderer.render_production_figure(production),
        title="production fixture",
        work_stem=work / renderer.PRODUCTION_STEM,
        tiff_dpi=72,
        png_dpi=72,
    )
    renderer.export_figure(
        renderer.render_controlled_figure(status, profiles),
        title="controlled fixture",
        work_stem=work / renderer.CONTROLLED_STEM,
        tiff_dpi=72,
        png_dpi=72,
    )
    renderer._write_render_manifest(
        work / "render_manifest.json", work_root=work, paper_root=paper
    )
    renderer._write_work_complete(work)
    renderer._copy_paper_from_completed_work(work, paper)

    renderer.validate_rendered_outputs(work, paper)

    assert {path.name for path in paper.iterdir()} == renderer.PAPER_FILES
    assert "NO-GO" in " ".join(
        text.get_text()
        for axis in renderer.render_controlled_figure(status, profiles).axes
        for text in axis.texts
    )
    plt.close("all")


def test_csv_roundtrip_preserves_exact_profile_numerics(tmp_path: Path) -> None:
    production, _, _ = renderer.load_production_profiles(
        renderer.DEFAULT_PRODUCTION_INPUT
    )
    path = tmp_path / "profiles.csv"

    renderer._write_csv(path, production)
    renderer._validate_csv_roundtrip(path, production)

    restored = pd.read_csv(path, float_precision="round_trip")
    for column in (
        "coverage_mean",
        "coverage_deviation_from_target_pp",
        "normalized_width_mean",
    ):
        assert (
            restored[column].to_numpy(float) == production[column].to_numpy(float)
        ).all()


def test_partial_publication_retains_completed_work_and_publishes_no_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged_work = tmp_path / "staged_work"
    staged_paper = tmp_path / "staged_paper"
    staged_work.mkdir()
    staged_paper.mkdir()
    (staged_work / "COMPLETE").write_text("complete\n", encoding="utf-8")
    (staged_paper / "figure.pdf").write_bytes(b"%PDF-fixture")
    work_output = tmp_path / "work"
    paper_output = tmp_path / "paper"
    real_replace = renderer.os.replace
    calls = 0

    def fail_second_replace(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected paper publication failure")
        real_replace(source, target)

    monkeypatch.setattr(renderer.os, "replace", fail_second_replace)
    with pytest.raises(RuntimeError, match="completed work was retained"):
        renderer._publish_bundles(
            staged_work=staged_work,
            staged_paper=staged_paper,
            work_output=work_output,
            paper_output=paper_output,
        )

    assert work_output.is_dir()
    assert (work_output / "COMPLETE").is_file()
    assert not paper_output.exists()
