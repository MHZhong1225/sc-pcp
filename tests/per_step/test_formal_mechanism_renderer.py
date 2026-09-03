from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import render_formal_mechanism_results as renderer


def _metric(value: float) -> dict[str, float | int]:
    return {
        "count": 500,
        "mean": value,
        "minimum": value,
        "maximum": value,
        "median": value,
        "q05": value,
        "q95": value,
    }


def _exact_summary() -> dict[str, object]:
    identification = {}
    for mechanism_index, mechanism in enumerate(renderer.MECHANISMS):
        cells = {}
        for estimator_index, estimator in enumerate(renderer.ESTIMATORS):
            value = 0.0 if (mechanism, estimator) in renderer.IDENTIFICATION_CORRECT else (
                0.05 * (mechanism_index + estimator_index + 1)
            )
            cells[estimator] = {
                "root_mean_squared": _metric(value / 2.0),
                "maximum_absolute": _metric(value),
            }
        identification[mechanism] = cells
    return {
        "schema_version": 1,
        "status": "complete",
        "study": "exact_committed_prefix_finite_mdp",
        "population_exact": True,
        "canonical_method_unchanged": True,
        "diagnostic_only": True,
        "finite_sample_claim": False,
        "population_instance_audit": {"identification": identification},
    }


def _controlled_rows() -> list[dict[str, object]]:
    rows = []
    for gamma_index, gamma in enumerate(renderer.GAMMAS):
        for method_index, method in enumerate(renderer.METHODS):
            wsc = 0.91 + 0.001 * method_index
            width = 2.0 + 0.2 * method_index + 0.1 * gamma_index
            if method == "SC-PCP":
                width = 1.8 + 0.1 * gamma_index
            if gamma == -4.0 and method == "SC-PCP":
                wsc = 0.899
            if gamma == -4.0 and method == "MFCS":
                width = 1.9
            rows.append(
                {
                    "gamma": gamma,
                    "gamma_role": renderer._gamma_role(gamma),
                    "method": method,
                    "wsc": wsc,
                    "wsc_ci95_lower": wsc - 0.005,
                    "wsc_ci95_upper": wsc + 0.005,
                    "mean_coverage": wsc + 0.005,
                    "mean_coverage_ci95_lower": wsc,
                    "mean_coverage_ci95_upper": wsc + 0.01,
                    "mean_normalized_width": width,
                    "width_ci95_lower": width - 0.1,
                    "width_ci95_upper": width + 0.1,
                    "selected_seeds": 20,
                    "total_seeds": 20,
                    "selection_rate": 1.0,
                    "target_adaptation_trajectories_per_seed": 0,
                    "efficiency_winner_among_eligible": False,
                    "source_json_path": "fixture.json",
                }
            )
    return rows


def _mechanism_rows() -> list[dict[str, object]]:
    rows = []
    for gamma in renderer.GAMMAS:
        for metric, unit, multiplier in (
            ("standard_same_radius_late_coverage_gap", "pp", 0.5),
            ("standard_late_q90_relative_shift", "%", -1.0),
        ):
            estimate = gamma * multiplier
            rows.append(
                {
                    "gamma": gamma,
                    "gamma_role": renderer._gamma_role(gamma),
                    "metric": metric,
                    "unit": unit,
                    "estimate": estimate,
                    "ci95_lower": estimate - 0.1,
                    "ci95_upper": estimate + 0.1,
                    "seed_count": 20,
                    "late_stages_zero_based": "4,5,6,7,8,9,10,11",
                    "method": "Standard CP",
                    "derivation": "fixture",
                    "source_seed_artifact_pattern": "fixture",
                }
            )
    return rows


def test_exact_source_rows_preserve_factorial_identification_pattern() -> None:
    summary = _exact_summary()

    renderer._validate_exact_summary(summary)
    rows = renderer.build_exact_source_rows(summary)

    assert len(rows) == 16
    correct = {
        (row["mechanism"], row["estimator"])
        for row in rows
        if row["identification_correct"]
    }
    assert correct == renderer.IDENTIFICATION_CORRECT
    assert all(row["role"] == "structural_diagnostic_not_baseline" for row in rows)


def test_exact_summary_rejects_missing_structural_diagnostic() -> None:
    summary = deepcopy(_exact_summary())
    del summary["population_instance_audit"]["identification"][
        "M3_full_feedback"
    ]["current_only"]

    with pytest.raises(RuntimeError, match="estimator set"):
        renderer._validate_exact_summary(summary)


def test_mechanism_derivation_uses_late_stage_seed_means() -> None:
    seed_rows = []
    aggregates = []
    for gamma_index, gamma in enumerate(renderer.GAMMAS):
        aggregates.append(
            {
                "gamma": gamma,
                "bootstrap_seed": 10_000 + gamma_index,
            }
        )
        for seed in renderer.SEEDS:
            seed_rows.append(
                {
                    "seed": seed,
                    "gamma": gamma,
                    "methods": {
                        "Standard CP": {
                            "coverage_gap": [gamma * 0.001] * renderer.HORIZON,
                            "q90_relative_gap": [-gamma * 0.002] * renderer.HORIZON,
                        }
                    },
                }
            )
    summary = {"aggregates": aggregates}

    rows = renderer.build_mechanism_source_rows(seed_rows, summary)

    assert len(rows) == 10
    primary_gap = next(
        row
        for row in rows
        if row["gamma"] == -2.0
        and row["metric"] == "standard_same_radius_late_coverage_gap"
    )
    assert primary_gap["estimate"] == pytest.approx(-0.2)
    assert primary_gap["ci95_lower"] == pytest.approx(-0.2)
    assert primary_gap["ci95_upper"] == pytest.approx(-0.2)


def test_efficiency_winner_requires_point_coverage_and_selection() -> None:
    rows = _controlled_rows()

    winners = renderer.efficiency_winners(rows)

    assert winners[-4.0] == "MFCS"
    assert all(winners[gamma] == "SC-PCP" for gamma in renderer.GAMMAS[1:])
    for row in rows:
        if row["gamma"] == 0.0 and row["method"] == "SC-PCP":
            row["selection_rate"] = 0.94
    assert renderer.efficiency_winners(rows)[0.0] == "Standard CP"


def test_render_exports_editable_svg_and_pdf_only_paper_bundle(
    tmp_path: Path,
) -> None:
    exact_rows = renderer.build_exact_source_rows(_exact_summary())
    controlled_rows = _controlled_rows()
    mechanism_rows = _mechanism_rows()
    winners = renderer.efficiency_winners(controlled_rows)
    controlled_rows = [
        {
            **row,
            "efficiency_winner_among_eligible": (
                winners[float(row["gamma"])] == row["method"]
            ),
        }
        for row in controlled_rows
    ]
    work = tmp_path / "work"
    paper = tmp_path / "paper"
    work.mkdir()
    paper.mkdir()
    renderer.write_source_csv(
        work / "figure_exact_source_data.csv",
        exact_rows,
        renderer.EXACT_SOURCE_FIELDS,
    )
    renderer.write_source_csv(
        work / "figure_controlled_source_data.csv",
        controlled_rows,
        renderer.CONTROLLED_SOURCE_FIELDS,
    )
    renderer.write_source_csv(
        work / "figure_controlled_mechanism_source_data.csv",
        mechanism_rows,
        renderer.MECHANISM_SOURCE_FIELDS,
    )
    (work / "analysis.json").write_text("{}\n", encoding="utf-8")
    (work / "figure_qa.md").write_text("# QA\n", encoding="utf-8")
    renderer.apply_publication_style()
    visuals = (
        (
            renderer.EXACT_FIGURE,
            renderer.render_exact_identification_figure(exact_rows),
        ),
        (
            renderer.CONTROLLED_FIGURE,
            renderer.render_controlled_figure(controlled_rows, mechanism_rows),
        ),
        (
            renderer.CONTROLLED_TABLE,
            renderer.render_controlled_table(controlled_rows),
        ),
    )
    for stem, figure in visuals:
        renderer.export_visual(
            figure,
            title="Fixture",
            work_stem=work / stem,
            paper_path=paper / f"{stem}.pdf",
            tiff_dpi=72,
            png_dpi=72,
        )
    renderer._write_render_manifest(
        work / "render_manifest.json", work_root=work, paper_root=paper
    )

    renderer.validate_rendered_outputs(work, paper)

    assert {path.suffix for path in paper.iterdir()} == {".pdf"}
    svg = (work / f"{renderer.EXACT_FIGURE}.svg").read_text(encoding="utf-8")
    assert "<text" in svg
    assert "Times New Roman" in svg
    plt.close("all")
