from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
import torch

import scpcp.experiment as experiment
from scpcp.baselines import OnlineBaselineResult
from scpcp.config import ExperimentConfig, PaperConfig
from scpcp.data import DataSplits, TrajectoryBatch
from scpcp.marginal_prefix import MarginalPrefixSelection
from scpcp.selection import RadiusSelection


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import render_paper_results as renderer


def _batch() -> TrajectoryBatch:
    return TrajectoryBatch(
        states=torch.zeros(3, 3, 1),
        actions=torch.zeros(3, 2, dtype=torch.long),
        outcomes=torch.tensor(
            [
                [[1.0, 2.0], [2.0, 4.0]],
                [[3.0, 6.0], [4.0, 8.0]],
                [[5.0, 10.0], [6.0, 12.0]],
            ]
        ),
        patient_ids=torch.arange(3),
    )


_TEST_SOURCE_HASH = "a" * 64
_TEST_GIT_REVISION = "b" * 40


def _write_renderer_study(
    study: Path,
    *,
    source_hash: str = _TEST_SOURCE_HASH,
) -> Path:
    config = {"certification": {"alpha": 0.10}, "seeds": [0]}
    study.mkdir(parents=True)
    (study / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    (study / "study_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "expected_seeds": [0],
                "completed_seeds": [0],
                "missing_seeds": [],
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    (study / "study_metadata.json").write_text(
        json.dumps(
            {
                "source_tree_sha256": source_hash,
                "git_revision": _TEST_GIT_REVISION,
                "seeds": [0],
            }
        ),
        encoding="utf-8",
    )
    seed_root = study / "seed_00000"
    seed_root.mkdir()
    records = []
    for method in renderer.METHOD_ORDER:
        record: dict[str, object] = {
            "track": "empirical_environment",
            "method": method,
        }
        if method == "SC-PCP":
            record.update(
                {
                    "selection_estimand": "per_step_marginal",
                    "selection_parameter": "stagewise_radii",
                    "selection_status": "SELECTED_MARGINAL_POINT",
                    "selection_available": True,
                    "certificate_type": float("nan"),
                    "certificate_formal": False,
                    "certified": False,
                    "lower_bound_min": float("nan"),
                    "guarantee_scope": renderer.SCPCP_GUARANTEE_SCOPE,
                    "selection_evidence": renderer.SCPCP_SELECTION_EVIDENCE,
                }
            )
        records.append(record)
    pd.DataFrame(records).to_csv(seed_root / "records.csv", index=False)
    (seed_root / "metadata.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "source_tree_sha256": source_hash,
                "git_revision": _TEST_GIT_REVISION,
                "config": config,
                "diagnostics": {
                    "protocol": renderer.PAPER_PROTOCOL,
                    "method": renderer.PAPER_METHOD,
                    "guarantee_scope": renderer.SCPCP_GUARANTEE_SCOPE,
                    "scpcp_selection_available": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (seed_root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    (study / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return seed_root


def _write_renderer_suite(root: Path) -> None:
    root.mkdir()
    manifest = {
        "protocol": renderer.PAPER_PROTOCOL,
        "method": renderer.PAPER_METHOD,
        "experiment_tree_sha256": "c" * 64,
        "sections": ["rq1", "rq3"],
        "datasets": list(renderer.DATASET_LABELS),
        "feedback_levels": list(renderer.FEEDBACK_LEVELS),
    }
    (root / "suite_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    for dataset in renderer.DATASET_LABELS:
        _write_renderer_study(root / "rq1" / dataset)
    for beta in renderer.FEEDBACK_LEVELS:
        if beta != 1.0:
            _write_renderer_study(root / "rq3" / f"beta_{beta:g}")
    (root / "COMPLETE").write_text("complete\n", encoding="utf-8")


def test_training_outcome_sd_uses_dpred_and_sample_correction() -> None:
    batch = _batch()

    observed = experiment._training_outcome_sd(batch)
    expected = batch.outcomes.reshape(-1, 2).std(dim=0, unbiased=True)

    assert torch.allclose(observed, expected)


def test_paper_rng_streams_are_stable_and_separated() -> None:
    assert experiment._paper_seed(7, 101) == experiment._paper_seed(7, 101)
    assert experiment._paper_seed(7, 101) != experiment._paper_seed(7, 211)
    assert experiment._paper_seed(7, 101) != experiment._paper_seed(8, 101)


def test_paper_config_validates_protocol_controls() -> None:
    config = ExperimentConfig(paper=PaperConfig(mechanism_seed=0))
    config.validate()

    invalid = ExperimentConfig(paper=PaperConfig(mechanism_seed=-1))
    with pytest.raises(ValueError, match="paper.mechanism_seed"):
        invalid.validate()


def test_metric_placeholders_expose_explicit_selection_and_width() -> None:
    placeholders = experiment._metric_placeholders()

    assert placeholders["selection_available"] is False
    assert math.isnan(placeholders["average_normalized_width"])


def test_renderer_accepts_one_consistent_marginal_suite(tmp_path: Path) -> None:
    suite = tmp_path / "paper"
    _write_renderer_suite(suite)

    renderer.validate_complete_suite(suite)


def test_renderer_rejects_wrong_manifest_protocol(tmp_path: Path) -> None:
    suite = tmp_path / "paper"
    _write_renderer_suite(suite)
    manifest_path = suite / "suite_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol"] = "historical_lcb_scpcp"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="wrong protocol"):
        renderer.validate_complete_suite(suite)


def test_renderer_rejects_noncanonical_method_name(tmp_path: Path) -> None:
    study = tmp_path / "study"
    seed_root = _write_renderer_study(study)
    records_path = seed_root / "records.csv"
    records = pd.read_csv(records_path)
    records.loc[records["method"].eq("SC-PCP"), "method"] = "SC-PCP (old)"
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="exactly the six paper methods"):
        renderer._validate_complete_study(study)


def test_renderer_rejects_changed_alpha(tmp_path: Path) -> None:
    study = tmp_path / "study"
    _write_renderer_study(study)
    config_path = study / "config.yaml"
    config = json.loads(config_path.read_text())
    config["certification"]["alpha"] = 0.11
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="alpha=0.10"):
        renderer._validate_complete_study(study)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("guarantee_scope", "finite_sample_pac"),
        ("certificate_formal", True),
        ("certificate_type", "bootstrap_lcb"),
    ),
)
def test_renderer_rejects_stronger_scpcp_claim_fields(
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    study = tmp_path / "study"
    seed_root = _write_renderer_study(study)
    records_path = seed_root / "records.csv"
    records = pd.read_csv(records_path)
    records[field] = records[field].astype(object)
    records.loc[records["method"].eq("SC-PCP"), field] = bad_value
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="SC-PCP record"):
        renderer._validate_complete_study(study)


def test_renderer_rejects_cross_study_source_mismatch(tmp_path: Path) -> None:
    suite = tmp_path / "paper"
    _write_renderer_suite(suite)
    study = suite / "rq3" / "beta_2"
    different_hash = "d" * 64
    study_metadata_path = study / "study_metadata.json"
    study_metadata = json.loads(study_metadata_path.read_text())
    study_metadata["source_tree_sha256"] = different_hash
    study_metadata_path.write_text(json.dumps(study_metadata), encoding="utf-8")
    seed_metadata_path = study / "seed_00000" / "metadata.json"
    seed_metadata = json.loads(seed_metadata_path.read_text())
    seed_metadata["source_tree_sha256"] = different_hash
    seed_metadata_path.write_text(json.dumps(seed_metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="one consistent source tree"):
        renderer.validate_complete_suite(suite)


def test_run_seed_uses_confirmed_grid_combined_calibration_and_simple_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _batch()
    cot = TrajectoryBatch(
        states=torch.zeros(2, 3, 1),
        actions=torch.zeros(2, 2, dtype=torch.long),
        outcomes=torch.zeros(2, 2, 2),
        patient_ids=torch.tensor([10, 11]),
    )
    certification = TrajectoryBatch(
        states=torch.zeros(3, 3, 1),
        actions=torch.zeros(3, 2, dtype=torch.long),
        outcomes=torch.zeros(3, 2, 2),
        patient_ids=torch.tensor([20, 21, 22]),
    )
    task = experiment._Task(
        environment=object(),
        splits=DataSplits(predictor, None, cot, certification),
        n_actions=2,
        logging_policy=object(),
        name="synthetic",
        policy_config=object(),
    )
    context = experiment._ExperimentContext(
        task=task,
        outcome_model=object(),
        region=object(),
        policy=object(),
        logging_policy=object(),
        outcome_sd=torch.ones(2),
    )
    cot_scores = torch.tensor([[0.2, 1.0], [0.8, 1.6]])
    cert_scores = torch.tensor(
        [[0.4, 1.2], [0.6, 1.4], [1.0, 1.8]]
    )
    score_calls = iter((cot_scores, cert_scores))
    observed: dict[str, object] = {"evaluation_seeds": []}

    monkeypatch.setattr(
        experiment,
        "_prepare_experiment_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        experiment,
        "score_batch",
        lambda *_args, **_kwargs: next(score_calls),
    )

    def select_prefix(batch, scores, *, stage_grids, **_kwargs):
        observed["prefix_patient_ids"] = batch.patient_ids.clone()
        observed["prefix_scores"] = scores.clone()
        observed["stage_grids"] = stage_grids.clone()
        candidate_shape = stage_grids.shape
        return MarginalPrefixSelection(
            radii=torch.tensor([0.8, 1.6]),
            selected_indices=(2, 2),
            estimated_coverage=torch.tensor([0.90, 0.91]),
            estimated_normalized_width=torch.tensor([1.0, 1.1]),
            effective_sample_size=torch.tensor([4.8, 4.7]),
            maximum_raw_log_weight=torch.tensor([0.2, 0.3]),
            raw_log_weight_span=torch.tensor([0.4, 0.5]),
            candidate_effective_sample_size=torch.full(candidate_shape, 4.5),
            candidate_estimated_coverage=torch.full(candidate_shape, 0.9),
            candidate_estimated_normalized_width=torch.ones(candidate_shape),
            candidate_maximum_raw_log_weight=torch.zeros(candidate_shape),
            candidate_raw_log_weight_span=torch.ones(candidate_shape),
            selected_endpoint=True,
            failure_stage=None,
        )

    monkeypatch.setattr(experiment, "select_marginal_prefix_schedule", select_prefix)

    def standard(scores, _alpha):
        observed["standard_scores"] = scores.clone()
        return torch.tensor([0.8, 1.6])

    monkeypatch.setattr(experiment, "standard_cp_stagewise_radii", standard)
    monkeypatch.setattr(
        experiment,
        "finite_depth_mfcs_selection",
        lambda *_args, **_kwargs: (
            RadiusSelection(1.0, 0, "EMPIRICAL_REFERENCE"),
            torch.ones(3, 2),
        ),
    )
    adaptation = OnlineBaselineResult(
        radius_by_time=torch.ones(2),
        target_deployments=4,
        rounds=1,
        adaptation_per_time_coverage=torch.ones(2),
        adaptation_round_worst_coverage=(1.0,),
        adaptation_pathwise_coverage=1.0,
        selected_scale=1.0,
    )
    monkeypatch.setattr(experiment, "aci_style_controller", lambda *_args, **_kwargs: adaptation)
    monkeypatch.setattr(
        experiment,
        "multidim_spci_style_controller",
        lambda *_args, **_kwargs: adaptation,
    )
    monkeypatch.setattr(experiment, "prc_profile_scale", lambda *_args, **_kwargs: adaptation)

    def evaluate_radius(name, _radius, *_args, **_kwargs):
        observed["evaluation_seeds"].append(_args[4])
        return {"method": name, "selection_available": True}

    def evaluate_stagewise(name, _adaptation, *_args, **_kwargs):
        observed["evaluation_seeds"].append(_args[4])
        return {"method": name, "selection_available": True}

    monkeypatch.setattr(experiment, "_evaluate_radius_method", evaluate_radius)
    monkeypatch.setattr(experiment, "_evaluate_stagewise_method", evaluate_stagewise)

    config = ExperimentConfig(horizon=2, q_grid_size=3)
    result = experiment.run_seed(config, seed=7, device="cpu")
    combined_scores = torch.cat((cot_scores, cert_scores), dim=0)
    expected_grids = experiment._committed_prefix_stage_grids(cot_scores, config)

    assert torch.equal(observed["prefix_scores"], combined_scores)
    assert torch.equal(observed["standard_scores"], combined_scores)
    assert torch.equal(observed["prefix_patient_ids"], torch.tensor([10, 11, 20, 21, 22]))
    assert torch.equal(observed["stage_grids"], expected_grids)
    assert tuple(record["method"] for record in result.records) == (
        "Standard CP",
        "ACI",
        "MFCS",
        "SPCI",
        "PRC",
        "SC-PCP",
    )
    assert len(set(observed["evaluation_seeds"])) == 1
    scpcp = result.records[-1]
    assert scpcp["selection_estimand"] == "per_step_marginal"
    assert scpcp["guarantee_scope"] == "asymptotic_per_step_marginal"
    assert scpcp["calibration_trajectories"] == 5
    assert result.diagnostics["stage_grid_role"] == "D_COT"
    assert "scpcp_candidate_coverage" in result.surfaces


def test_paper_curve_ci_uses_metric_appropriate_bounds() -> None:
    matrix = np.array([[0.80, 0.90], [1.00, 1.00]])
    coverage_low, coverage_high = renderer.curve_t_ci(matrix, probability=True)

    widths = 4.0 * matrix
    width_low, width_high = renderer.curve_t_ci(widths, probability=False)

    assert np.all(coverage_low >= 0.0)
    assert np.all(coverage_high <= 1.0)
    assert np.all(width_low >= 0.0)
    assert width_high.max() > 1.0


def test_main_aggregation_uses_marginal_worst_and_seed_bootstrap() -> None:
    records = pd.DataFrame(
        [
            {
                "seed": 1,
                "dataset": "synthetic",
                "method_family": "SC-PCP",
                "selection_available": True,
                "worst_coverage": 0.80,
                "per_time_coverage": "[1.0, 0.8]",
                "average_coverage": 0.90,
                "average_normalized_width": 1.2,
            },
            {
                "seed": 0,
                "dataset": "synthetic",
                "method_family": "SC-PCP",
                "selection_available": True,
                "worst_coverage": 0.80,
                "per_time_coverage": "[0.8, 1.0]",
                "average_coverage": 0.90,
                "average_normalized_width": 1.0,
            },
        ]
    )

    first = renderer.aggregate_main(records).iloc[0]
    second = renderer.aggregate_main(records.iloc[::-1]).iloc[0]

    assert first["marginal_worst_coverage"] == pytest.approx(0.90)
    assert first["marginal_worst_coverage"] != pytest.approx(
        records["worst_coverage"].mean()
    )
    assert first["marginal_worst_coverage_ci_low"] == pytest.approx(0.80)
    assert first["marginal_worst_coverage_ci_high"] == pytest.approx(0.90)
    assert first["marginal_worst_target_met"]
    assert first["efficiency_eligible"]
    assert first["marginal_worst_coverage_ci_low"] == second[
        "marginal_worst_coverage_ci_low"
    ]
    assert first["marginal_worst_coverage_ci_high"] == second[
        "marginal_worst_coverage_ci_high"
    ]


def test_efficiency_eligibility_uses_marginal_target_and_selection_rate() -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "method": "Standard CP",
                "marginal_worst_coverage": 0.91,
                "selection_rate": 0.94,
                "average_normalized_width": 0.5,
            },
            {
                "dataset": "synthetic",
                "method": "ACI",
                "marginal_worst_coverage": 0.899,
                "selection_rate": 1.0,
                "average_normalized_width": 0.4,
            },
            {
                "dataset": "synthetic",
                "method": "SC-PCP",
                "marginal_worst_coverage": 0.90,
                "selection_rate": 0.95,
                "average_normalized_width": 1.0,
            },
            {
                "dataset": "synthetic",
                "method": "MFCS",
                "marginal_worst_coverage": 0.91,
                "selection_rate": 1.0,
                "average_normalized_width": 1.2,
            },
        ]
    )

    assert renderer._efficient_eligible_methods(summary) == {
        ("synthetic", "SC-PCP")
    }


def test_main_table_note_defines_marginal_worst_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "method": "SC-PCP",
                "marginal_worst_coverage": 0.901,
                "marginal_worst_coverage_ci_low": 0.900,
                "marginal_worst_coverage_ci_high": 0.902,
                "average_coverage": 0.902,
                "average_coverage_ci_low": 0.901,
                "average_coverage_ci_high": 0.903,
                "average_normalized_width": 1.1,
                "average_normalized_width_ci_low": 1.0,
                "average_normalized_width_ci_high": 1.2,
                "selection_rate": 1.0,
                "n_selected": 100,
                "n_runs": 100,
            }
        ]
    )
    figure_text: list[str] = []

    def capture_figure(figure: matplotlib.figure.Figure, *_args, **_kwargs) -> None:
        figure_text.extend(text.get_text() for text in figure.texts)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_figure)
    renderer.render_table(
        summary,
        tmp_path / "table.pdf",
        title="Test table",
        include_dataset=False,
    )

    note = "\n".join(figure_text)
    assert "min_t of the across-seed mean per-step coverage" in note
    assert "resamples seeds as whole per-time vectors" in note
    assert "Selection Rate >= 95%" in note


def test_feedback_worst_curve_uses_marginal_per_step_estimand(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = []
    for method in renderer.METHOD_ORDER:
        for feedback in (0.0, 1.0):
            for seed, coverage in enumerate(("[0.8, 1.0]", "[1.0, 0.8]")):
                rows.append(
                    {
                        "seed": seed,
                        "dataset": "synthetic",
                        "feedback_strength": feedback,
                        "method_family": method,
                        "selection_available": True,
                        "worst_coverage": 0.8,
                        "per_time_coverage": coverage,
                        "average_coverage": 0.9,
                        "average_normalized_width": 1.0,
                    }
                )
    captured: list[matplotlib.figure.Figure] = []
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda figure, *_args, **_kwargs: captured.append(figure),
    )

    renderer.render_feedback_stress(
        pd.DataFrame(rows),
        tmp_path / "feedback.pdf",
    )

    assert len(captured) == 1
    method_lines = captured[0].axes[0].lines[: len(renderer.METHOD_ORDER)]
    assert all(np.allclose(line.get_ydata(), 0.90) for line in method_lines)
    assert captured[0].axes[0].get_ylabel() == "Marginal worst-step coverage"


def test_coverage_profile_states_that_curves_are_conditional_on_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = []
    for dataset, feedback_strength in (("synthetic", 2.0), ("mimic_iv", 1.0)):
        rows.append(
            {
                "dataset": dataset,
                "feedback_strength": feedback_strength,
                "method_family": "Standard CP",
                "per_time_coverage": "[0.90, 0.91]",
                "per_time_normalized_width": "[1.20, 1.10]",
            }
        )
    records = pd.DataFrame(rows)
    figure_text: list[str] = []

    def capture_figure(figure: matplotlib.figure.Figure, *_args, **_kwargs) -> None:
        figure_text.extend(text.get_text() for text in figure.texts)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_figure)
    renderer.render_coverage_profiles(records, tmp_path / "coverage.pdf")

    assert renderer.CONDITIONAL_SELECTION_NOTE in figure_text


def test_mechanism_figure_uses_committed_prefix_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    synthetic_root = tmp_path / "rq1" / "synthetic"
    synthetic_root.mkdir(parents=True)
    (synthetic_root / "config.yaml").write_text(
        "paper:\n  mechanism_seed: 7\nseeds: [7]\n",
        encoding="utf-8",
    )
    seed_root = synthetic_root / "seed_00007"
    seed_root.mkdir(parents=True)
    np.savez_compressed(
        seed_root / "surfaces.npz",
        scpcp_stage_grids=np.array([[0.5, 1.0, 1.5], [0.6, 1.1, 1.6]]),
        scpcp_candidate_coverage=np.array(
            [[0.80, 0.90, 0.95], [0.82, 0.91, 0.96]]
        ),
        scpcp_selected_indices=np.array([1, 1]),
        scpcp_selected_radii=np.array([1.0, 1.1]),
        scpcp_selected_effective_sample_size=np.array([120.0, 110.0]),
    )
    pd.DataFrame(
        [{"method": "SC-PCP", "per_time_coverage": "[0.901, 0.902]"}]
    ).to_csv(seed_root / "records.csv", index=False)
    captured: list[matplotlib.figure.Figure] = []
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda figure, *_args, **_kwargs: captured.append(figure),
    )

    renderer.render_mechanism(tmp_path, tmp_path / "mechanism.pdf")

    assert len(captured) == 1
    assert [axis.get_ylabel() for axis in captured[0].axes] == [
        "Per-step coverage",
        "Selected radius",
        "Effective sample size",
    ]
    assert np.array_equal(captured[0].axes[0].lines[0].get_xdata(), [0, 1])


def test_mechanism_figure_rejects_seed_outside_configured_seeds(
    tmp_path: Path,
) -> None:
    synthetic_root = tmp_path / "rq1" / "synthetic"
    synthetic_root.mkdir(parents=True)
    (synthetic_root / "config.yaml").write_text(
        "paper:\n  mechanism_seed: 7\nseeds: [0, 1]\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not in the configured seeds"):
        renderer.render_mechanism(tmp_path, tmp_path / "mechanism.pdf")
