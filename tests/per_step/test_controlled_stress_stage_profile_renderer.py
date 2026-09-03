from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import render_controlled_stress_stage_profile as renderer


def _summary_and_seed_rows() -> tuple[dict[str, object], list[dict[str, object]]]:
    coverage_bases = {
        "Standard CP": 0.864,
        "ACI": 0.881,
        "MFCS": 0.918,
        "SPCI": 0.895,
        "PRC": 0.878,
        "SC-PCP": 0.898,
    }
    width_bases = {
        "Standard CP": 4.0,
        "ACI": 4.3,
        "MFCS": 5.7,
        "SPCI": 4.8,
        "PRC": 4.4,
        "SC-PCP": 4.9,
    }
    seed_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(renderer.SEEDS):
        centered = seed_index - (len(renderer.SEEDS) - 1) / 2.0
        methods = {}
        for method in renderer.METHODS:
            coverage = np.asarray(
                [
                    coverage_bases[method]
                    + (0.0 if stage == 1 else 0.004 + 0.0002 * stage)
                    + centered * 0.00015
                    for stage in range(renderer.HORIZON)
                ]
            )
            width = np.asarray(
                [
                    width_bases[method]
                    + 0.04 * stage
                    + centered * 0.004
                    for stage in range(renderer.HORIZON)
                ]
            )
            methods[method] = {
                "selection_available": True,
                "target_coverage": coverage.tolist(),
                "target_normalized_width": width.tolist(),
            }
        seed_rows.append(
            {
                "seed": seed,
                "gamma": renderer.STRESS_GAMMA,
                "methods": methods,
            }
        )

    method_summary = {}
    for method in renderer.METHODS:
        coverage = np.asarray(
            [row["methods"][method]["target_coverage"] for row in seed_rows]
        )
        width = np.asarray(
            [row["methods"][method]["target_normalized_width"] for row in seed_rows]
        )
        stage_mean = coverage.mean(axis=0)
        method_summary[method] = {
            "target_coverage_by_stage": stage_mean.tolist(),
            "target_marginal_worst_coverage": float(stage_mean.min()),
            "target_worst_stage_zero_based": int(stage_mean.argmin()),
            "target_mean_coverage": float(stage_mean.mean()),
            "mean_target_normalized_width": float(width.mean()),
            "selection_rate": 1.0,
            "selected_seeds": 20,
            "total_seeds": 20,
            "target_adaptation_trajectories_per_seed": (
                2_000 if method in renderer.ONLINE_METHODS else 0
            ),
        }
    standard_wsc = method_summary["Standard CP"][
        "target_marginal_worst_coverage"
    ]
    scpcp_wsc = method_summary["SC-PCP"]["target_marginal_worst_coverage"]
    standard_width = method_summary["Standard CP"][
        "mean_target_normalized_width"
    ]
    scpcp_width = method_summary["SC-PCP"]["mean_target_normalized_width"]
    aggregate = {
        "gamma": renderer.STRESS_GAMMA,
        "n_seeds": 20,
        "bootstrap_seed": 123_456,
        "methods": method_summary,
        "paired_scpcp_comparisons": {
            method: {
                "scpcp_minus_baseline_wsc": (
                    scpcp_wsc
                    - method_summary[method]["target_marginal_worst_coverage"]
                ),
                "scpcp_minus_baseline_wsc_ci95": [
                    scpcp_wsc
                    - method_summary[method]["target_marginal_worst_coverage"]
                    - 0.001,
                    scpcp_wsc
                    - method_summary[method]["target_marginal_worst_coverage"]
                    + 0.001,
                ],
                "scpcp_to_baseline_geometric_width_ratio": (
                    scpcp_width
                    / method_summary[method]["mean_target_normalized_width"]
                ),
                "scpcp_to_baseline_geometric_width_ratio_ci95": [0.99, 1.01],
            }
            for method in renderer.METHODS
            if method != "SC-PCP"
        },
    }
    summary = {
        "protocol": renderer.CONTROLLED_PROTOCOL,
        "role": "fresh_confirmatory_canonical_baseline_comparison",
        "methods": list(renderer.METHODS),
        "seeds": list(renderer.SEEDS),
        "primary_metric": "min_t mean_seed(target_coverage_seed_t)",
        "coverage_conditioning": "successful_selection",
        "selection_rate_denominator": "all_prespecified_seeds",
        "bootstrap": {
            "resamples": renderer.BOOTSTRAP_RESAMPLES,
            "gamma_seeds": {"-4": 123_456},
        },
        "aggregates": [aggregate],
    }
    assert scpcp_wsc - standard_wsc == pytest.approx(0.034)
    assert scpcp_width > standard_width
    return summary, seed_rows


def test_source_rows_are_complete_pointwise_seed_bootstrap_profiles() -> None:
    summary, seed_rows = _summary_and_seed_rows()

    rows = renderer.build_source_rows(seed_rows, summary)

    assert len(rows) == len(renderer.METHODS) * renderer.HORIZON
    scpcp_stage_one = next(
        row
        for row in rows
        if row["method"] == "SC-PCP" and row["stage_zero_based"] == 1
    )
    assert scpcp_stage_one["target_coverage"] == pytest.approx(0.898)
    assert scpcp_stage_one["coverage_deviation_from_090_pp"] == pytest.approx(
        -0.2
    )
    assert scpcp_stage_one["target_coverage_ci95_lower"] < 0.898
    assert scpcp_stage_one["target_coverage_ci95_upper"] > 0.898
    assert "pointwise" in scpcp_stage_one["interval_definition"]
    assert {
        row["method"] for row in rows
    } == set(renderer.METHODS)


def test_hero_metrics_match_plotted_stage_profiles_and_paired_summary() -> None:
    summary, seed_rows = _summary_and_seed_rows()
    rows = renderer.build_source_rows(seed_rows, summary)

    hero = renderer.build_hero_metrics(rows, summary)

    assert hero["standard_wsc"] == pytest.approx(0.864)
    assert hero["scpcp_wsc"] == pytest.approx(0.898)
    assert hero["scpcp_minus_standard_wsc_pp"] == pytest.approx(3.4)
    assert hero["standard_worst_stage_zero_based"] == 1
    assert hero["scpcp_worst_stage_zero_based"] == 1


def test_seed_validation_rejects_missing_stage() -> None:
    _, seed_rows = _summary_and_seed_rows()
    malformed = deepcopy(seed_rows[0])
    malformed["methods"]["SC-PCP"]["target_coverage"] = [0.9] * 11

    with pytest.raises(RuntimeError, match="finite length 12"):
        renderer._validate_seed_row(malformed, seed=renderer.SEEDS[0])


def test_render_exports_editable_svg_and_pdf_only_paper_bundle(
    tmp_path: Path,
) -> None:
    summary, seed_rows = _summary_and_seed_rows()
    rows = renderer.build_source_rows(seed_rows, summary)
    hero = renderer.build_hero_metrics(rows, summary)
    work = tmp_path / "work"
    paper = tmp_path / "paper"
    work.mkdir()
    paper.mkdir()
    renderer.write_source_csv(
        work / f"{renderer.FIGURE_STEM}_source_data.csv", rows
    )
    (work / "analysis.json").write_text("{}\n", encoding="utf-8")
    (work / "figure_qa.md").write_text("# QA\n", encoding="utf-8")
    renderer.apply_publication_style()
    renderer.export_figure(
        renderer.render_figure(rows, hero),
        work_stem=work / renderer.FIGURE_STEM,
        paper_path=paper / f"{renderer.FIGURE_STEM}.pdf",
        tiff_dpi=72,
        png_dpi=72,
    )
    renderer._write_render_manifest(
        work / "render_manifest.json", work_root=work, paper_root=paper
    )

    renderer.validate_rendered_outputs(work, paper)

    assert {path.suffix for path in paper.iterdir()} == {".pdf"}
    svg = (work / f"{renderer.FIGURE_STEM}.svg").read_text(encoding="utf-8")
    assert "<text" in svg
    assert "Times New Roman" in svg
