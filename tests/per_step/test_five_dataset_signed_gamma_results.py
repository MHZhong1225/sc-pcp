from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from matplotlib.container import ErrorbarContainer
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import render_five_dataset_signed_gamma_results as renderer


@pytest.fixture(scope="module")
def complete_fixture(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("cxr_science")
    support_seeds = tuple(renderer.cxr_science.CONFIRMATION_SEEDS)
    science_seeds = support_seeds[:-1]
    theta_payload = {"fixture": "frozen-bridge"}
    gates = SimpleNamespace(
        support_k0_eligible_seeds=support_seeds,
        prespecified_seeds=support_seeds,
        active_source_tree_sha256="a" * 64,
        theta=SimpleNamespace(to_dict=lambda: theta_payload),
        preset=SimpleNamespace(bootstrap_seed=63_300_019),
        confirmation_binding={"root": "frozen-confirmation", "combined_sha256": "b" * 64},
        contract={"protocol": "fixture-gate-contract"},
        science_contract={
            "calibration_trajectories": 3_000,
            "grid_trajectories": 1_000,
            "evaluation_trajectories": 20_000,
            "target_adaptation_trajectories": dict(renderer.TARGET_ADAPTATION_BUDGET),
        },
    )
    metadata = {
        "source_snapshot": {},
        "config_sha256": "c" * 64,
        "fixture": "exact-science-metadata",
    }
    summary = _cxr_summary(science_seeds)
    audit = _coverage_audit(summary, science_seeds)
    overlap = {
        "protocol": renderer.cxr_science.PROTOCOL,
        "dataset": "mimic_cxr",
        "status": "OVERLAP_GO",
        "gate": "gamma=-4 q_mid+q_high empirical donor-overlap screen",
        "thresholds": {
            "local_ess_p01": 10.0,
            "median_ess_fraction": 0.25,
            "maximum_donor_probability": 0.25,
        },
        "prespecified_seed_count": 20,
        "support_k0_eligible_seed_count": 20,
        "support_k0_eligible_seeds": list(support_seeds),
        "overlap_bank_complete": True,
        "overlap_completed_seed_count": 20,
        "joint_overlap_pass_count": 19,
        "minimum_joint_overlap_pass_count": 19,
        "passed_seeds": list(science_seeds),
        "failed_seeds": [support_seeds[-1]],
        "science_may_start": True,
        "failure_consequence": "OVERLAP_NO_GO_NO_COVERAGE_SCIENCE",
        "seed_deletions": 0,
    }
    science_final = {
        "protocol": renderer.cxr_science.PROTOCOL,
        "dataset": "mimic_cxr",
        "status": "SCIENCE_COMPLETE",
        "methods": list(renderer.METHODS),
        "gammas": list(renderer.GAMMAS),
        "primary_gamma": renderer.PRIMARY_GAMMA,
        "primary_metric": renderer.cxr_science.PRIMARY_METRIC,
        "prespecified_seed_count": 20,
        "science_eligible_seed_count": 19,
        "science_eligible_seeds": list(science_seeds),
        "seed_deletions": 0,
    }
    unlock = renderer.cxr_science._science_unlock(gates, overlap, science_seeds)
    final = {
        **science_final,
        "confirmation_status": "CONFIRMATION_GO",
        "overlap_status": "OVERLAP_GO",
        "science_unlocked": True,
        "coverage_generated": True,
        "science_unlock_sha256": renderer.cxr_science._json_sha256(unlock),
    }
    science_root = root / renderer.cxr_science.SCIENCE_PHASE
    science_root.mkdir()
    _write_json(root / "metadata.json", metadata)
    _write_json(root / "FINAL_STATUS.json", final)
    _write_json(root / "SCIENCE_UNLOCK.json", unlock)
    overlap_root = root / renderer.cxr_science.OVERLAP_PHASE
    overlap_root.mkdir()
    _write_json(overlap_root / "summary.json", overlap)
    _write_json(root / "manifest.json", {"artifact_count": 6})
    (root / "COMPLETE").write_text("fixture-complete\n", encoding="utf-8")
    _write_json(science_root / "FINAL_STATUS.json", science_final)
    _write_json(science_root / "summary.json", summary)
    _write_json(science_root / "coverage_audit.json", audit)

    calls: list[str] = []
    patch = pytest.MonkeyPatch()

    def verify_gate_bundle(*, devices):
        assert devices == ("cuda:0", "cuda:1")
        calls.append("confirmation")
        return gates

    def confirmation_binding(path):
        assert path == renderer.cxr_science.CONFIRMATION_ROOT.resolve()
        calls.append("binding")
        return gates.confirmation_binding

    def science_metadata(bundle, **kwargs):
        assert bundle is gates
        assert kwargs["devices"] == ("cuda:0", "cuda:1")
        assert kwargs["audit_go_sha256"] == renderer.cxr_science._json_sha256(
            gates.contract
        )
        assert kwargs["source_snapshot"] == {}
        calls.append("metadata")
        return metadata

    def verify_manifest(path):
        assert path == root.resolve()
        calls.append("manifest")

    def validate_complete(path, observed_metadata, bundle):
        assert path == root.resolve()
        assert observed_metadata == metadata
        assert bundle is gates
        calls.append("complete")

    patch.setattr(renderer.cxr_science, "verify_gate_bundle", verify_gate_bundle)
    patch.setattr(renderer.cxr_science, "_confirmation_binding", confirmation_binding)
    patch.setattr(renderer.cxr_science, "_science_metadata", science_metadata)
    patch.setattr(renderer.cxr_science, "_verify_manifest", verify_manifest)
    patch.setattr(renderer.cxr_science, "_validate_complete_root", validate_complete)
    cxr = renderer.load_cxr_environment_support_sources(root)
    sources = renderer.build_reporting_sources(
        native_input=renderer.DEFAULT_NATIVE_INPUT,
        clinical_input=renderer.DEFAULT_CLINICAL_INPUT,
        cxr_input=root,
        production_input=renderer.DEFAULT_PRODUCTION_INPUT,
    )
    try:
        yield SimpleNamespace(
            root=root,
            gates=gates,
            metadata=metadata,
            cxr=cxr,
            sources=sources,
            calls=calls,
        )
    finally:
        patch.undo()


def _cxr_summary(seeds: tuple[int, ...]) -> dict:
    eligible_count = len(seeds)
    support_seeds = tuple(renderer.cxr_science.CONFIRMATION_SEEDS)
    support_count = len(support_seeds)
    aggregates = []
    for gamma_index, gamma in enumerate(renderer.GAMMAS):
        method_cells = {}
        for method_index, method in enumerate(renderer.METHODS):
            coverage = np.asarray(
                [
                    0.885
                    + 0.004 * method_index
                    + 0.001 * gamma_index
                    + 0.0015 * stage
                    for stage in range(renderer.HORIZONS["mimic_cxr"])
                ]
            )
            width = np.asarray(
                [1.0 + 0.09 * method_index + 0.02 * gamma_index + 0.01 * stage for stage in range(6)]
            )
            wsc = float(coverage.min())
            mean_coverage = float(coverage.mean())
            mean_width = float(width.mean())
            method_cells[method] = {
                "n_selected": eligible_count,
                "n_prespecified": 20,
                "n_k0_eligible": support_count,
                "n_support_k0_eligible": support_count,
                "n_support_k0_overlap_eligible": eligible_count,
                "selection_rate": eligible_count / 20,
                "selection_rate_ci95": renderer.cxr_science._wilson_interval(
                    eligible_count, 20
                ),
                "target_adaptation_trajectories_per_seed": renderer.TARGET_ADAPTATION_BUDGET[
                    method
                ],
                "target_marginal_worst_coverage": wsc,
                "target_worst_stage_zero_based": int(coverage.argmin()),
                "target_wsc_ci95": [wsc - 0.004, wsc + 0.004],
                "target_coverage_by_stage": coverage.tolist(),
                "target_coverage_by_stage_ci95": [
                    [float(value - 0.005), float(value + 0.005)] for value in coverage
                ],
                "target_mean_coverage": mean_coverage,
                "target_mean_coverage_ci95": [
                    mean_coverage - 0.003,
                    mean_coverage + 0.003,
                ],
                "target_normalized_width_by_stage": width.tolist(),
                "target_normalized_width_by_stage_ci95": [
                    [float(value - 0.03), float(value + 0.03)] for value in width
                ],
                "mean_target_normalized_width": mean_width,
                "mean_target_normalized_width_ci95": [
                    mean_width - 0.02,
                    mean_width + 0.02,
                ],
                "point_eligible": bool(wsc >= 0.90) if gamma == -4.0 else None,
            }
        paired = (
            {
                method: {
                    "paired_selected_seeds": eligible_count,
                    "scpcp_minus_baseline_wsc": 0.01,
                    "scpcp_minus_baseline_wsc_ci95": [0.005, 0.015],
                    "scpcp_to_baseline_geometric_width_ratio": 1.10,
                    "scpcp_to_baseline_geometric_width_ratio_ci95": [1.05, 1.15],
                }
                for method in renderer.METHODS
                if method != "SC-PCP"
            }
            if gamma == -4.0
            else {"status": "EXCLUDED_NON_CONFIRMATORY_GAMMA_SIGNED_CONTROL"}
        )
        aggregates.append(
            {
                "gamma": gamma,
                "analysis_role": (
                    "confirmatory_gamma_minus_4_endpoint"
                    if gamma == -4.0
                    else "descriptive_signed_control_curve"
                ),
                "n_prespecified_seeds": 20,
                "n_k0_eligible_seeds": support_count,
                "n_support_k0_eligible_seeds": support_count,
                "n_support_k0_overlap_eligible_seeds": eligible_count,
                "methods": method_cells,
                "paired_scpcp_comparisons": paired,
            }
        )
    return {
        "protocol": renderer.cxr_science.PROTOCOL,
        "dataset": "mimic_cxr",
        "role": "post_failure_cxr_environment_support_science",
        "interpretation_status": "EMPIRICAL_OVERLAP_SCREEN_PASSED",
        "seeds_prespecified": list(renderer.cxr_science.CONFIRMATION_SEEDS),
        "seeds_support_k0_eligible": list(support_seeds),
        "seeds_k0_eligible": list(support_seeds),
        "seeds_support_k0_overlap_eligible": list(seeds),
        "compatibility_field_semantics": {
            "seeds_k0_eligible": (
                "alias of seeds_support_k0_eligible before donor-overlap screening"
            ),
            "aggregates[].n_k0_eligible_seeds": (
                "count of support/K0-eligible seeds before donor-overlap screening"
            ),
            "aggregates[].methods[].n_k0_eligible": (
                "count of support/K0-eligible seeds before donor-overlap screening"
            ),
        },
        "methods": list(renderer.METHODS),
        "primary_gamma": -4.0,
        "primary_metric": renderer.cxr_science.PRIMARY_METRIC,
        "mean_coverage_is_supplementary": True,
        "coverage_conditioning": (
            "successful method selection among support/K0/overlap-eligible seeds"
        ),
        "selection_rate_denominator": "all 20 prespecified confirmation seeds",
        "seed_deletions": 0,
        "bootstrap": {
            "resamples": 10_000,
            "prespecified_seed_count": 20,
            "root_seed": 63_300_019,
            "uniform_matrix_shape": [10_000, 20],
            "complete_seed_index_matrix_shape": [10_000, 20],
        },
        "aggregates": aggregates,
    }


def _coverage_audit(summary: dict, seeds: tuple[int, ...]) -> dict:
    support_seeds = tuple(renderer.cxr_science.CONFIRMATION_SEEDS)
    records = []
    for aggregate in summary["aggregates"]:
        for method in renderer.METHODS:
            cell = aggregate["methods"][method]
            records.append(
                {
                    "gamma": aggregate["gamma"],
                    "method": method,
                    "n_support_k0_eligible": len(support_seeds),
                    "n_support_k0_overlap_eligible": len(seeds),
                    "n_selected": cell["n_selected"],
                    "selection_rate_denominator": 20,
                    "metrics": {
                        "stage_coverage": cell["target_coverage_by_stage"],
                        "WSC": cell["target_marginal_worst_coverage"],
                        "MeanCov": cell["target_mean_coverage"],
                    },
                }
            )
    return {
        "protocol": renderer.cxr_science.PROTOCOL,
        "dataset": "mimic_cxr",
        "status": "COVERAGE_AUDIT_COMPLETE",
        "primary_metric": renderer.cxr_science.PRIMARY_METRIC,
        "formula_verified": True,
        "mean_coverage_is_supplementary": True,
        "all_six_methods_present": True,
        "all_five_gammas_present": True,
        "coverage_conditioning": (
            "successful method selection among support/K0/overlap-eligible seeds"
        ),
        "seeds_support_k0_eligible": list(support_seeds),
        "seeds_support_k0_overlap_eligible": list(seeds),
        "support_k0_eligible_seed_count": len(support_seeds),
        "support_k0_overlap_eligible_seed_count": len(seeds),
        "science_eligible_seeds": list(seeds),
        "selection_rate_denominator": 20,
        "records": records,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_cxr_adapter_calls_science_runner_validators_and_builds_exact_grid(
    complete_fixture,
) -> None:
    cxr = complete_fixture.cxr
    assert {"confirmation", "binding", "metadata", "manifest", "complete"}.issubset(
        complete_fixture.calls
    )
    assert len(cxr.status) == 5
    assert len(cxr.stage) == 5 * 6 * 6
    assert len(cxr.scalar) == 5 * 6
    assert len(cxr.paired) == 5
    assert set(cxr.scalar["n_prespecified"]) == {20}
    assert set(cxr.scalar["n_gate_eligible"]) == {19}
    assert set(cxr.scalar["selection_rate"]) == {0.95}
    assert cxr.input_contracts["confirmation_binding"] == complete_fixture.gates.confirmation_binding
    assert cxr.input_contracts["support_k0_eligible_seed_count"] == 20
    assert cxr.input_contracts["science_eligible_seed_count"] == 19


def test_cxr_summary_contract_accepts_full_twenty_of_twenty_overlap(
    complete_fixture,
) -> None:
    seeds = tuple(renderer.cxr_science.CONFIRMATION_SEEDS)
    summary = _cxr_summary(seeds)
    audit = _coverage_audit(summary, seeds)
    renderer._validate_cxr_summary(summary, audit, complete_fixture.gates, seeds)


def test_cxr_summary_distinguishes_support_and_overlap_cohorts(
    complete_fixture,
) -> None:
    seeds = tuple(renderer.cxr_science.CONFIRMATION_SEEDS[:-1])
    summary = _cxr_summary(seeds)
    audit = _coverage_audit(summary, seeds)
    summary["aggregates"][0]["methods"]["SC-PCP"][
        "n_support_k0_overlap_eligible"
    ] = 20
    with pytest.raises(RuntimeError, match="method eligibility cohorts"):
        renderer._validate_cxr_summary(summary, audit, complete_fixture.gates, seeds)


def test_five_dataset_signed_grid_is_complete_and_protocol_separated(
    complete_fixture,
) -> None:
    sources = complete_fixture.sources
    renderer.validate_reporting_sources(sources)
    signed = sources.scalar[sources.scalar["reporting_family"].isin(renderer.SIGNED_FAMILIES)]
    assert len(signed) == 5 * 5 * 6
    assert set(signed["dataset"]) == set(renderer.DATASETS)
    assert set(signed["feedback_value"]) == set(renderer.GAMMAS)
    assert signed["metric_available"].astype(bool).all()
    assert not signed["reporting_family"].eq(renderer.PRODUCTION_FAMILY).any()
    assert len(renderer._primary_stage(sources)) == sum(
        renderer.HORIZONS[dataset] for dataset in renderer.DATASETS
    ) * 6
    assert len(renderer._primary_scalar(sources)) == 5 * 6


def test_wsc_meancov_selection_intervals_and_budgets_are_exact(
    complete_fixture,
) -> None:
    sources = complete_fixture.sources
    for row in renderer._signed_scalar(sources).itertuples(index=False):
        profiles = sources.stage[
            sources.stage["reporting_family"].eq(row.reporting_family)
            & sources.stage["setting_id"].eq(row.setting_id)
            & sources.stage["method"].eq(row.method)
        ].sort_values("stage_zero_based")
        coverage = profiles["coverage_mean"].to_numpy(float)
        assert row.wsc == pytest.approx(float(coverage.min()), abs=1e-12)
        assert row.mean_coverage == pytest.approx(float(coverage.mean()), abs=1e-12)
        assert row.selection_rate == pytest.approx(row.n_selected / 20, abs=1e-14)
        assert row.calibration_trajectories_per_seed == 3_000
        assert row.grid_trajectories_per_seed == 1_000
        assert row.evaluation_trajectories_per_seed == 20_000
        assert row.target_adaptation_trajectories_per_seed == renderer.TARGET_ADAPTATION_BUDGET[
            row.method
        ]
        for point, lower, upper in (
            (row.wsc, row.wsc_ci95_lower, row.wsc_ci95_upper),
            (
                row.mean_coverage,
                row.mean_coverage_ci95_lower,
                row.mean_coverage_ci95_upper,
            ),
            (
                row.mean_normalized_width,
                row.mean_normalized_width_ci95_lower,
                row.mean_normalized_width_ci95_upper,
            ),
        ):
            assert lower <= point <= upper


@pytest.mark.parametrize(
    ("frame_name", "field", "message"),
    [
        ("scalar", "wsc", "WSC differs"),
        ("scalar", "selection_rate", "selection rate denominator"),
        ("scalar", "wsc_ci95_lower", "interval does not contain"),
        ("scalar", "evaluation_trajectories_per_seed", "trajectories_per_seed"),
    ],
)
def test_numeric_contract_tampering_fails_closed(
    complete_fixture, frame_name: str, field: str, message: str
) -> None:
    sources = complete_fixture.sources
    frame = deepcopy(getattr(sources, frame_name))
    index = frame[
        frame["reporting_family"].eq(renderer.SIGNED_FAMILIES[-1])
        & frame["feedback_value"].eq(-4.0)
        & frame["method"].eq("SC-PCP")
    ].index[0]
    if field == "wsc_ci95_lower":
        frame.loc[index, field] = frame.loc[index, "wsc"] + 0.01
    elif field == "selection_rate":
        frame.loc[index, field] = 0.90
    elif field == "evaluation_trajectories_per_seed":
        frame.loc[index, field] = 19_999
    else:
        frame.loc[index, field] += 0.001
    changed = renderer.ReportingSources(
        status=sources.status,
        stage=sources.stage,
        scalar=frame,
        paired=sources.paired,
        input_contracts=sources.input_contracts,
    )
    with pytest.raises(RuntimeError, match=message):
        renderer.validate_reporting_sources(changed)


def test_missing_signed_gamma_cell_fails_closed(complete_fixture) -> None:
    sources = complete_fixture.sources
    scalar = sources.scalar.drop(
        sources.scalar[
            sources.scalar["reporting_family"].eq(renderer.SIGNED_FAMILIES[-1])
            & sources.scalar["feedback_value"].eq(4.0)
            & sources.scalar["method"].eq("ACI")
        ].index
    )
    with pytest.raises(RuntimeError, match="scalar method grid"):
        renderer.validate_reporting_sources(
            renderer.ReportingSources(
                status=sources.status,
                stage=sources.stage,
                scalar=scalar,
                paired=sources.paired,
                input_contracts=sources.input_contracts,
            )
        )


@pytest.mark.parametrize(
    ("render", "rows"),
    [
        (renderer.render_gamma_minus4_hero, 2),
        (renderer.render_gamma_minus4_stagewise, 2),
        (renderer.render_signed_gamma_figure, 3),
    ],
)
def test_quantitative_figures_have_exact_grid_and_no_explanatory_prose(
    complete_fixture, render, rows: int
) -> None:
    renderer.apply_publication_style()
    figure = render(complete_fixture.sources)
    assert len(figure.axes) == rows * 5
    assert figure._suptitle is None
    assert figure.texts == []
    assert [axis.get_title() for axis in figure.axes[:5]] == [
        renderer.DATASET_LABELS[dataset] for dataset in renderer.DATASETS
    ]
    for axis in figure.axes:
        assert sum(
            isinstance(container, ErrorbarContainer)
            for container in axis.containers
        ) == 6
    assert [text.get_text() for text in figure.legends[0].get_texts()] == list(
        renderer.METHODS
    )
    visible = " ".join(
        [axis.get_title() for axis in figure.axes]
        + [text.get_text() for axis in figure.axes for text in axis.texts]
    ).lower()
    for forbidden in ("claim", "gate", "watermark", "footer", "no-go"):
        assert forbidden not in visible
    if render is renderer.render_signed_gamma_figure:
        for column in range(5):
            assert [
                tick.get_text() for tick in figure.axes[10 + column].get_xticklabels()
            ] == [renderer.report_v2._format_gamma(value) for value in renderer.GAMMAS]
    plt.close(figure)


def test_complete_table_is_five_by_six_quantitative_only(complete_fixture) -> None:
    renderer.apply_publication_style()
    figure = renderer.render_gamma_minus4_table(complete_fixture.sources)
    axis = figure.axes[0]
    assert axis.get_title() == ""
    assert len(axis.texts) == 0
    assert len(axis.tables) == 1
    cells = axis.tables[0].get_celld()
    assert max(row for row, _ in cells) == 30
    visible = " ".join(cell.get_text().get_text() for cell in cells.values())
    for dataset in renderer.DATASETS:
        assert renderer.DATASET_LABELS[dataset] in visible
    for forbidden in ("claim", "Gate:", "watermark", "footer", "NO-GO"):
        assert forbidden not in visible
    plt.close(figure)


def test_render_bundle_is_fresh_complete_deterministic_and_pdf_only(
    complete_fixture, tmp_path: Path
) -> None:
    roots = [
        (tmp_path / "work_a", tmp_path / "paper_a"),
        (tmp_path / "work_b", tmp_path / "paper_b"),
    ]
    for work, paper in roots:
        renderer.render_report(
            native_input=renderer.DEFAULT_NATIVE_INPUT,
            clinical_input=renderer.DEFAULT_CLINICAL_INPUT,
            cxr_input=complete_fixture.root,
            production_input=renderer.DEFAULT_PRODUCTION_INPUT,
            work_output=work,
            paper_output=paper,
        )
        renderer.validate_rendered_outputs(work, paper)

    first_work, first_paper = roots[0]
    second_work, second_paper = roots[1]
    assert {path.name for path in first_work.iterdir()} == renderer.WORK_FILES
    assert {path.name for path in first_paper.iterdir()} == renderer.PAPER_FILES
    assert all(path.suffix == ".pdf" for path in first_paper.iterdir())
    for name in sorted(renderer.WORK_FILES - {"COMPLETE", "render_manifest.json"}):
        assert _sha256(first_work / name) == _sha256(second_work / name), name
    for name in sorted(renderer.PAPER_FILES):
        assert _sha256(first_paper / name) == _sha256(second_paper / name), name
    contract = json.loads((first_work / "figure_contract.json").read_text())
    assert contract["production_robustness_boundary"] == {
        "included_in_signed_gamma_figures_or_table": False,
        "included_in_source_csv": True,
        "reason": "production is a separate no-gamma robustness protocol",
        "used_to_fill_signed_gamma_cells": False,
    }
    assert contract["source_freeze"]["included"] is True
    for stem in renderer.OUTPUT_STEMS:
        with Image.open(first_work / f"{stem}.tiff") as image:
            assert image.info["dpi"][0] == pytest.approx(600, rel=1e-3)
            assert image.info["dpi"][1] == pytest.approx(600, rel=1e-3)
        assert "<text" in (first_work / f"{stem}.svg").read_text(encoding="utf-8")


def test_existing_or_shared_output_roots_fail_before_render(
    complete_fixture, tmp_path: Path
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        renderer.render_report(
            native_input=renderer.DEFAULT_NATIVE_INPUT,
            clinical_input=renderer.DEFAULT_CLINICAL_INPUT,
            cxr_input=complete_fixture.root,
            production_input=renderer.DEFAULT_PRODUCTION_INPUT,
            work_output=existing,
            paper_output=tmp_path / "paper",
        )
    same = tmp_path / "same"
    with pytest.raises(ValueError, match="differ"):
        renderer.render_report(
            native_input=renderer.DEFAULT_NATIVE_INPUT,
            clinical_input=renderer.DEFAULT_CLINICAL_INPUT,
            cxr_input=complete_fixture.root,
            production_input=renderer.DEFAULT_PRODUCTION_INPUT,
            work_output=same,
            paper_output=same,
        )


def test_renderer_is_inside_formal_top_level_tools_snapshot() -> None:
    assert renderer._renderer_is_in_formal_source_snapshot()
    assert Path(renderer.__file__).resolve().parent == ROOT / "tools"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
