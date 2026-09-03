from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import render_theorem_robustness_results as renderer


def _theory_summaries() -> tuple[dict[str, object], dict[str, object]]:
    shape = (len(renderer.HORIZONS), len(renderer.NOMINAL_POLICY_TVS))

    def matrix(value: float) -> list[list[float]]:
        return [[value for _ in range(shape[1])] for _ in range(shape[0])]

    horizon = {
        "phase_diagram": {
            "SC-PCP": {
                "coverage_shortfall": matrix(-0.01),
                "median_minimum_selected_ess_fraction": matrix(0.75),
                "median_surface_sup_error": matrix(0.02),
                "availability_rate": matrix(1.0),
            }
        }
    }
    by_n = {}
    for index, n_calibration in enumerate(renderer.N_CALIBRATION):
        error = 0.05 / (index + 1)
        wsc = 0.93 - 0.001 * index
        width = 5.5 - 0.02 * index
        by_n[str(n_calibration)] = {
            "track_a_fixed_population_grid": {
                "mean_surface_sup_error": error,
                "cluster_bootstrap_95_ci": [error - 0.001, error + 0.001],
            },
            "track_b_canonical_empirical_grid": {
                "selection_availability_rate": 1.0,
                "population_wsc_conditional_on_selection": wsc,
                "population_wsc_cluster_bootstrap_95_ci": [wsc - 0.001, wsc + 0.001],
                "population_mean_normalized_width_conditional_on_selection": width,
                "population_width_cluster_bootstrap_95_ci": [width - 0.01, width + 0.01],
            },
        }
    return horizon, {"by_n_calibration": by_n}


def _robustness_summaries() -> tuple[dict[str, object], dict[str, object]]:
    nuisance = {}
    primary_arms = {}
    target_drift = {}
    for index, arm in enumerate(renderer.PROPENSITY_ARMS):
        nuisance[arm] = {"mean": 0.01 * index, "ci95": [0.01 * index, 0.01 * index]}
        primary_arms[arm] = {
            "marginal_worst_step_coverage": 0.91 + 0.001 * index,
            "marginal_worst_step_coverage_ci95": [0.90, 0.92],
            "minimum_stage_mean_ess_fraction": 0.95 - 0.05 * index,
            "minimum_stage_mean_ess_fraction_ci95": [0.80, 0.96],
        }
        target_drift[arm] = {
            "mean": 0.02 * index,
            "ci95": [0.02 * index, 0.02 * index],
        }
    propensity = {
        "nuisance_diagnostics": {"mae": nuisance},
        "primary_transport_only": {"results": {"arms": primary_arms}},
        "appendix_end_to_end": {"target_policy_drift_from_oracle": target_drift},
    }
    seed_counts = {
        "synthetic_main": 100,
        "mimic_iv": 20,
        "controlled_gamma_minus_2": 20,
    }
    settings = {}
    for index, setting in enumerate(renderer.STRICT_SETTINGS):
        wsc = -0.001 + index * 0.001
        ratio = 1.0 + index * 0.001
        settings[setting] = {
            "seeds": list(range(seed_counts[setting])),
            "variants": {"strict": {"selection_rate": 1.0}},
            "paired_strict_vs_canonical": {
                "strict_minus_canonical_wsc": wsc,
                "strict_minus_canonical_wsc_ci95": [wsc - 0.001, wsc + 0.001],
                "strict_to_canonical_geometric_width_ratio": ratio,
                "strict_to_canonical_geometric_width_ratio_ci95": [
                    ratio - 0.001,
                    ratio + 0.001,
                ],
            },
        }
    return propensity, {"settings": settings}


def test_theory_source_rows_apply_declared_units_and_keep_all_cells() -> None:
    horizon, rq6 = _theory_summaries()

    rows = renderer.build_theory_source_rows(horizon, rq6)

    assert len(rows) == 93
    panel_a = [row for row in rows if row["panel"] == "a"]
    panel_d = [row for row in rows if row["panel"] == "d"]
    assert len(panel_a) == 25
    assert panel_a[0]["estimate"] == pytest.approx(-1.0)
    assert panel_a[0]["unit"] == "pp"
    assert panel_d[0]["estimate"] == pytest.approx(5.0)
    assert panel_d[0]["ci95_lower"] == pytest.approx(4.9)
    assert panel_d[0]["cluster_count"] == 100
    assert panel_d[0]["replicates_per_cluster"] == 20


def test_robustness_source_rows_preserve_primary_appendix_and_paired_transforms() -> None:
    propensity, strict = _robustness_summaries()

    rows = renderer.build_robustness_source_rows(propensity, strict)

    assert len(rows) == 18
    panel_b = [row for row in rows if row["panel"] == "b"]
    assert panel_b[0]["estimate"] == pytest.approx(91.0)
    panel_d = [row for row in rows if row["panel"] == "d"]
    assert panel_d[-1]["estimate"] == pytest.approx(4.0)
    panel_e = [row for row in rows if row["panel"] == "e"]
    assert panel_e[0]["estimate"] == pytest.approx(-0.1)
    panel_f = [row for row in rows if row["panel"] == "f"]
    assert panel_f[1]["estimate"] == pytest.approx(0.1)


def test_flat_manifest_validation_fails_after_payload_tampering(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text("{}\n", encoding="utf-8")
    contract = renderer._file_contract(payload)
    manifest = {
        "schema_version": 1,
        "protocol": "test_protocol",
        "status": "complete",
        "files": {"payload.json": contract},
    }
    renderer._validate_flat_manifest(
        tmp_path,
        manifest,
        protocol="test_protocol",
        payload_names={"payload.json"},
    )

    payload.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="payload hash"):
        renderer._validate_flat_manifest(
            tmp_path,
            manifest,
            protocol="test_protocol",
            payload_names={"payload.json"},
        )


def test_render_exports_have_editable_svg_text_and_pdf_only_paper_bundle(
    tmp_path: Path,
) -> None:
    horizon, rq6 = _theory_summaries()
    propensity, strict = _robustness_summaries()
    theory_rows = renderer.build_theory_source_rows(horizon, rq6)
    robustness_rows = renderer.build_robustness_source_rows(propensity, strict)
    work = tmp_path / "work"
    paper = tmp_path / "paper"
    work.mkdir()
    paper.mkdir()
    renderer.write_source_csv(work / "figure_theory_source_data.csv", theory_rows)
    renderer.write_source_csv(
        work / "figure_robustness_source_data.csv", robustness_rows
    )
    (work / "analysis.json").write_text("{}\n", encoding="utf-8")
    (work / "figure_qa.md").write_text("# QA\n", encoding="utf-8")
    renderer.apply_publication_style()
    renderer.export_figure(
        renderer.render_theory_figure(theory_rows),
        title="Theory test",
        work_stem=work / renderer.THEORY_FIGURE,
        paper_path=paper / f"{renderer.THEORY_FIGURE}.pdf",
        tiff_dpi=72,
        png_dpi=72,
    )
    renderer.export_figure(
        renderer.render_robustness_figure(robustness_rows),
        title="Robustness test",
        work_stem=work / renderer.ROBUSTNESS_FIGURE,
        paper_path=paper / f"{renderer.ROBUSTNESS_FIGURE}.pdf",
        tiff_dpi=72,
        png_dpi=72,
    )
    renderer._write_render_manifest(
        work / "render_manifest.json", work_root=work, paper_root=paper
    )

    renderer.validate_rendered_outputs(work, paper)

    assert {path.suffix for path in paper.iterdir()} == {".pdf"}
    svg = (work / f"{renderer.THEORY_FIGURE}.svg").read_text(encoding="utf-8")
    assert "<text" in svg
    assert "Times New Roman" in svg


def test_horizon_summary_rejects_missing_structural_comparator() -> None:
    shape = (len(renderer.HORIZONS), len(renderer.NOMINAL_POLICY_TVS))
    matrix = [[1.0 for _ in range(shape[1])] for _ in range(shape[0])]
    summary = {
        "schema_version": 1,
        "study": "finite_mdp_horizon_overlap",
        "status": "complete",
        "mechanism_variant": "RQ5_only_overlap_controlled_M3",
        "canonical_method_unchanged": True,
        "diagnostic_only": True,
        "primary_coverage_estimand": "min_stage_mean_instance_conditional_on_availability",
        "phase_diagram": {"SC-PCP": {"coverage_shortfall": matrix}},
    }

    with pytest.raises(RuntimeError, match="method set"):
        renderer._validate_horizon_summary(summary)
