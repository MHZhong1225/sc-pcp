from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from xml.etree import ElementTree

from matplotlib.container import ErrorbarContainer
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.reporting import render_complete_coverage_reporting_v2 as renderer


@pytest.fixture(scope="module")
def sources() -> renderer.ReportingSources:
    return renderer.build_reporting_sources(
        native_input=renderer.DEFAULT_NATIVE_INPUT,
        clinical_input=renderer.DEFAULT_CLINICAL_INPUT,
        production_input=renderer.DEFAULT_PRODUCTION_INPUT,
    )


def test_protocol_adapters_build_the_exact_reporting_grids(
    sources: renderer.ReportingSources,
) -> None:
    assert len(sources.status) == 26
    assert len(sources.stage) == 1764
    assert len(sources.scalar) == 156
    assert len(sources.paired) == 20
    assert sources.status.groupby("reporting_family").size().to_dict() == {
        "clinical_cxr_terminal_no_go": 1,
        "clinical_v4_signed_gamma": 15,
        "native_signed_gamma": 5,
        "production_no_gamma_robustness": 5,
    }
    assert sources.stage.groupby("reporting_family").size().to_dict() == {
        "clinical_v4_signed_gamma": 1080,
        "native_signed_gamma": 360,
        "production_no_gamma_robustness": 324,
    }
    assert sources.scalar.groupby("reporting_family").size().to_dict() == {
        "clinical_cxr_terminal_no_go": 6,
        "clinical_v4_signed_gamma": 90,
        "native_signed_gamma": 30,
        "production_no_gamma_robustness": 30,
    }
    renderer.validate_reporting_sources(sources)


def test_default_gamma_minus4_has_four_curve_panels_and_terminal_cxr_gate(
    sources: renderer.ReportingSources,
) -> None:
    status = renderer._default_status(sources)
    assert tuple(status["dataset"]) == renderer.DATASETS
    assert tuple(status.iloc[:4]["panel_status"]) == ("CURVES",) * 4
    assert status.iloc[4]["panel_status"] == "GATE_TERMINAL_NO_GO"
    assert status.iloc[4]["gate_reason"] == (
        "V5_CONFIRMATION_COMPLETE_NO_GO;V6_DEVELOPMENT_NO_GO;TERMINAL_NO_V7"
    )
    stage = renderer._default_stage(sources)
    assert set(stage["dataset"]) == set(renderer.CURVE_DATASETS)
    assert len(stage) == 4 * 6 * 12
    assert not (stage["dataset"] == "mimic_cxr").any()
    scalar = renderer._default_scalar(sources)
    assert len(scalar) == 5 * 6
    cxr = scalar[scalar["dataset"].eq("mimic_cxr")]
    assert tuple(cxr["method"]) == renderer.METHODS
    assert not cxr["metric_available"].astype(bool).any()
    assert cxr["n_selected"].isna().all()
    assert cxr[
        [
            "selection_rate",
            "wsc",
            "wsc_ci95_lower",
            "mean_coverage",
            "mean_coverage_ci95_lower",
            "mean_normalized_width",
            "mean_normalized_width_ci95_lower",
            "point_attainment_at_target",
            "wsc_interval_attainment_at_target",
            "point_eligible",
        ]
    ].isna().all().all()
    assert set(cxr["budget_status"]) == {"not_run_precoverage_gate"}


def test_cxr_terminal_adapter_binds_completed_v5_and_terminal_v6(
    sources: renderer.ReportingSources,
) -> None:
    contract = sources.input_contracts["clinical_cxr_terminal"]
    assert contract["status"] == "terminal_precoverage_no_go"
    assert contract["v5_confirmation"] == {
        "protocol": renderer.CXR_V5_PROTOCOL,
        "input_root": renderer._project_path(renderer.DEFAULT_CXR_V5_CONFIRMATION_INPUT),
        "manifest_sha256": renderer.CXR_V5_MANIFEST_SHA256,
        "complete_sha256": renderer.CXR_V5_COMPLETE_SHA256,
        "final_status_sha256": renderer.CXR_V5_FINAL_SHA256,
        "artifact_count": 47,
        "k0_pass_count": 18,
        "structural_pass_count": 20,
        "support_pass_count": 20,
        "coverage_generated": False,
    }
    v6 = contract["v6_development"]
    assert v6["status"] == "DEVELOPMENT_NO_GO"
    assert v6["terminal_no_v7"] is True
    assert v6["coverage_generated"] is False
    assert v6["theta"] is None
    assert v6["lineages"] == {
        "v5_development": {"numeric": "19/20", "structural": "20/20"},
        "v5_failed_confirmation": {"numeric": "18/20", "structural": "20/20"},
    }
    assert v6["required_numeric"] == "20/20 per lineage"
    assert v6["fresh_confirmation_bank"] == {
        "base_seeds": "120000..120190 step 10",
        "base_seed_count": 20,
        "rng_stream_count": 341,
        "formal_rng_consumed": False,
        "collision_count": 0,
        "confirmation_root_present": False,
    }
    assert not (
        renderer.DEFAULT_CXR_V6_DEVELOPMENT_INPUT.parent
        / "controlled_clinical_fidelity_v6_mimic_cxr_confirmation"
    ).exists()


def test_native_and_clinical_field_aliases_reconcile_to_source_values(
    sources: renderer.ReportingSources,
) -> None:
    native_summary = json.loads(
        (renderer.DEFAULT_NATIVE_INPUT / "summary.json").read_text(encoding="utf-8")
    )
    clinical_summary = json.loads(
        (
            renderer.DEFAULT_CLINICAL_INPUT
            / "science/mimic_iv/summary.json"
        ).read_text(encoding="utf-8")
    )
    native_cell = native_summary["aggregates"][0]["methods"]["SC-PCP"]
    clinical_cell = clinical_summary["aggregates"][0]["methods"]["SC-PCP"]
    native = sources.stage[
        sources.stage["dataset"].eq("synthetic")
        & sources.stage["feedback_value"].eq(-4.0)
        & sources.stage["method"].eq("SC-PCP")
    ].sort_values("stage_zero_based")
    clinical = sources.stage[
        sources.stage["dataset"].eq("mimic_iv")
        & sources.stage["feedback_value"].eq(-4.0)
        & sources.stage["method"].eq("SC-PCP")
        & sources.stage["reporting_family"].eq("clinical_v4_signed_gamma")
    ].sort_values("stage_zero_based")
    assert native["coverage_ci95_lower"].tolist() == pytest.approx(
        [row[0] for row in native_cell["target_coverage_ci95_by_stage"]]
    )
    assert native["normalized_width_ci95_upper"].tolist() == pytest.approx(
        [row[1] for row in native_cell["target_normalized_width_ci95_by_stage"]]
    )
    assert clinical["coverage_ci95_lower"].tolist() == pytest.approx(
        [row[0] for row in clinical_cell["target_coverage_by_stage_ci95"]]
    )
    assert clinical["normalized_width_ci95_upper"].tolist() == pytest.approx(
        [row[1] for row in clinical_cell["target_normalized_width_by_stage_ci95"]]
    )


def test_every_available_scalar_reconciles_with_complete_stage_profiles(
    sources: renderer.ReportingSources,
) -> None:
    for row in sources.scalar[sources.scalar["metric_available"].astype(bool)].itertuples(
        index=False
    ):
        profiles = sources.stage[
            sources.stage["reporting_family"].eq(row.reporting_family)
            & sources.stage["setting_id"].eq(row.setting_id)
            & sources.stage["method"].eq(row.method)
        ].sort_values("stage_zero_based")
        coverage = profiles["coverage_mean"].to_numpy(float)
        width = profiles["normalized_width_mean"].to_numpy(float)
        assert row.wsc == pytest.approx(float(coverage.min()), abs=1e-12)
        assert int(row.worst_stage_zero_based) == int(coverage.argmin())
        tolerance = (
            5e-7
            if row.reporting_family == "production_no_gamma_robustness"
            else 1e-12
        )
        assert row.mean_coverage == pytest.approx(
            float(coverage.mean()), abs=tolerance
        )
        assert row.mean_normalized_width == pytest.approx(
            float(width.mean()), abs=5e-7
        )
        assert row.selection_rate == pytest.approx(
            row.n_selected / row.n_prespecified, abs=1e-14
        )


def test_audited_clinical_claim_is_exact_and_not_overstated(
    sources: renderer.ReportingSources,
) -> None:
    renderer.validate_claim_contract(sources.scalar)
    rows = sources.scalar[
        sources.scalar["reporting_family"].eq("clinical_v4_signed_gamma")
        & sources.scalar["feedback_value"].eq(-4.0)
    ]
    scpcp = rows[rows["method"].eq("SC-PCP")].set_index("dataset")
    assert (scpcp["mean_coverage_ci95_lower"] > 0.90).all()
    assert not scpcp["wsc_interval_attainment_at_target"].astype(bool).any()
    assert scpcp["point_attainment_at_target"].astype(bool).to_dict() == {
        "mimic_iv": False,
        "eicu": False,
        "inspire": True,
    }
    mfcs = rows[rows["method"].eq("MFCS")].set_index("dataset")
    assert mfcs["wsc_interval_attainment_at_target"].astype(bool).all()
    assert (mfcs["mean_normalized_width"] > scpcp["mean_normalized_width"]).all()
    assert not (
        scpcp["point_attainment_at_target"].astype(bool)
        & scpcp["wsc_interval_attainment_at_target"].astype(bool)
    ).any()


def test_eicu_keeps_gate_count_separate_from_selection_denominator(
    sources: renderer.ReportingSources,
) -> None:
    rows = sources.scalar[
        sources.scalar["reporting_family"].eq("clinical_v4_signed_gamma")
        & sources.scalar["dataset"].eq("eicu")
        & sources.scalar["feedback_value"].eq(-4.0)
    ]
    assert set(rows["n_prespecified"]) == {20}
    assert set(rows["n_gate_eligible"]) == {19}
    assert set(rows["n_selected"]) == {19}
    assert set(rows["selection_rate"]) == {0.95}


def test_production_no_gamma_is_robustness_only(
    sources: renderer.ReportingSources,
) -> None:
    production = sources.scalar[
        sources.scalar["reporting_family"].eq("production_no_gamma_robustness")
    ]
    default = renderer._default_scalar(sources)
    assert len(production) == 5 * 6
    assert not production["confirmatory"].astype(bool).any()
    assert not production["ranking_permitted"].astype(bool).any()
    assert production["point_eligible"].isna().all()
    assert not default["reporting_family"].eq(
        "production_no_gamma_robustness"
    ).any()


def test_main_figure_has_all_six_stagewise_interval_series_and_no_prose(
    sources: renderer.ReportingSources,
) -> None:
    renderer.apply_publication_style()
    figure = renderer.render_gamma_minus4_stagewise(sources)
    assert len(figure.axes) == 8
    for column in range(4):
        assert sum(
            isinstance(container, ErrorbarContainer)
            for container in figure.axes[column].containers
        ) == 6
        assert sum(
            isinstance(container, ErrorbarContainer)
            for container in figure.axes[4 + column].containers
        ) == 6
        assert any(line.get_ydata()[0] == 0.0 for line in figure.axes[column].lines)
    assert figure._suptitle is None
    assert figure.texts == []
    assert [axis.get_title() for axis in figure.axes[:4]] == [
        renderer.DATASET_LABELS[dataset] for dataset in renderer.CURVE_DATASETS
    ]
    visible = " ".join(
        [axis.get_title() for axis in figure.axes]
        + [text.get_text() for axis in figure.axes for text in axis.texts]
    )
    for forbidden in ("NO-GO", "confirmation", "coverage_generated", "terminal_no_v7"):
        assert forbidden not in visible
    assert [text.get_text() for text in figure.legends[0].get_texts()] == list(
        renderer.METHODS
    )
    plt.close(figure)


def test_signed_supplement_is_all_dataset_and_all_gamma(
    sources: renderer.ReportingSources,
) -> None:
    renderer.apply_publication_style()
    figure = renderer.render_signed_gamma_figure(sources)
    assert len(figure.axes) == 12
    rows = sources.scalar[
        sources.scalar["reporting_family"].isin(
            {"native_signed_gamma", "clinical_v4_signed_gamma"}
        )
    ]
    assert set(rows["dataset"]) == set(renderer.CURVE_DATASETS)
    assert set(rows["feedback_value"]) == set(renderer.GAMMAS)
    assert len(rows) == 4 * 5 * 6
    for row in range(3):
        for column in range(4):
            axis = figure.axes[row * 4 + column]
            assert sum(
                isinstance(container, ErrorbarContainer)
                for container in axis.containers
            ) == 6
    assert figure._suptitle is None
    assert figure.texts == []
    assert [text.get_text() for text in figure.legends[0].get_texts()] == list(
        renderer.METHODS
    )
    assert [axis.get_title() for axis in figure.axes[:4]] == [
        renderer.DATASET_LABELS[dataset] for dataset in renderer.CURVE_DATASETS
    ]
    for column in range(4):
        labels = [tick.get_text() for tick in figure.axes[8 + column].get_xticklabels()]
        assert labels == [renderer._format_gamma(value) for value in renderer.GAMMAS]
    plt.close(figure)


@pytest.mark.parametrize(
    "render",
    [renderer.render_gamma_minus4_table, renderer.render_production_table],
)
def test_scalar_tables_contain_only_headers_and_data_cells(
    sources: renderer.ReportingSources, render
) -> None:
    renderer.apply_publication_style()
    figure = render(sources)
    axis = figure.axes[0]
    assert axis.get_title() == ""
    assert len(axis.texts) == 0
    assert len(axis.tables) == 1
    visible = " ".join(
        cell.get_text().get_text() for cell in axis.tables[0].get_celld().values()
    )
    for forbidden in (
        "Default",
        "Robustness supplement",
        "WSC=min_t",
        "terminal NO-GO",
        "not run",
        "not defined",
    ):
        assert forbidden not in visible
    if render is renderer.render_gamma_minus4_table:
        assert "NA" in visible
    plt.close(figure)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("wsc", "WSC differs"),
        ("mean_coverage", "MeanCov differs"),
        ("worst_stage_zero_based", "worst stage differs"),
    ],
)
def test_scalar_tampering_fails_closed(
    sources: renderer.ReportingSources, field: str, message: str
) -> None:
    malformed = deepcopy(sources.scalar)
    index = malformed[
        malformed["reporting_family"].eq("clinical_v4_signed_gamma")
        & malformed["dataset"].eq("mimic_iv")
        & malformed["feedback_value"].eq(-4.0)
        & malformed["method"].eq("SC-PCP")
    ].index[0]
    malformed.loc[index, field] = (
        11 if field == "worst_stage_zero_based" else malformed.loc[index, field] + 1e-5
    )
    with pytest.raises(RuntimeError, match=message):
        renderer.validate_reporting_sources(
            renderer.ReportingSources(
                status=sources.status,
                stage=sources.stage,
                scalar=malformed,
                paired=sources.paired,
                input_contracts=sources.input_contracts,
            )
        )


def test_incomplete_or_hash_changed_roots_fail_before_summary_read(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    shutil.copyfile(renderer.DEFAULT_NATIVE_INPUT / "manifest.json", incomplete / "manifest.json")
    shutil.copyfile(renderer.DEFAULT_NATIVE_INPUT / "COMPLETE", incomplete / "COMPLETE")
    with pytest.raises(RuntimeError, match="missing or symbolic"):
        renderer.load_native_sources(incomplete)

    changed = tmp_path / "changed"
    changed.mkdir()
    (changed / "manifest.json").write_text("{}\n", encoding="utf-8")
    shutil.copyfile(renderer.DEFAULT_NATIVE_INPUT / "COMPLETE", changed / "COMPLETE")
    with pytest.raises(RuntimeError, match="pinned manifest or COMPLETE hash"):
        renderer.load_native_sources(changed)


def test_render_bundle_is_pdf_only_deterministic_and_600dpi(tmp_path: Path) -> None:
    first_work = tmp_path / "work_a"
    first_paper = tmp_path / "paper_a"
    second_work = tmp_path / "work_b"
    second_paper = tmp_path / "paper_b"
    for work, paper in ((first_work, first_paper), (second_work, second_paper)):
        renderer.render_report(
            native_input=renderer.DEFAULT_NATIVE_INPUT,
            clinical_input=renderer.DEFAULT_CLINICAL_INPUT,
            production_input=renderer.DEFAULT_PRODUCTION_INPUT,
            work_output=work,
            paper_output=paper,
        )
        renderer.validate_rendered_outputs(work, paper)

    assert {path.name for path in first_work.iterdir()} == renderer.WORK_FILES
    assert {path.name for path in first_paper.iterdir()} == renderer.PAPER_FILES
    assert all(path.suffix == ".pdf" for path in first_paper.iterdir())
    for name in sorted(renderer.WORK_FILES - {"COMPLETE", "render_manifest.json"}):
        assert _sha256(first_work / name) == _sha256(second_work / name), name
    for name in sorted(renderer.PAPER_FILES):
        assert _sha256(first_paper / name) == _sha256(second_paper / name), name
    for stem in renderer.OUTPUT_STEMS:
        with Image.open(first_work / f"{stem}.tiff") as image:
            dpi = image.info.get("dpi")
            assert dpi is not None
            assert dpi[0] == pytest.approx(600, rel=1e-3)
            assert dpi[1] == pytest.approx(600, rel=1e-3)
        svg = (first_work / f"{stem}.svg").read_text(encoding="utf-8")
        assert "<text" in svg
        assert "Times New Roman" in svg
        visible = _svg_visible_text(first_work / f"{stem}.svg")
        for forbidden in (
            "complete stagewise",
            "SC-PCP is closest",
            "Terminal pre-coverage",
            "coverage_generated",
            "Intervals are pointwise",
            "Robustness supplement",
        ):
            assert forbidden not in visible


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _svg_visible_text(path: Path) -> str:
    root = ElementTree.parse(path).getroot()
    return " ".join(
        "".join(node.itertext())
        for node in root.iter()
        if node.tag.rsplit("}", 1)[-1] == "text"
    )
