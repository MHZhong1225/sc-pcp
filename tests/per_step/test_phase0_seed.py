from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import scpcp.experiment as experiment
import scpcp.phase0_oracle as phase0_oracle
from scpcp.config import ExperimentConfig, SampleConfig


class _SetupComplete(RuntimeError):
    pass


def _schedule_family() -> experiment._RefinedScheduleFamily:
    vector = torch.tensor([1.0, 2.0])
    matrix = vector[None, :]
    return experiment._RefinedScheduleFamily(
        initial_quantiles=vector,
        initial_profile=vector,
        baseline_scale_grid=torch.tensor([0.5, 1.0, 1.5]),
        profile=vector,
        scale_grid=torch.tensor([0.5, 0.75, 1.0]),
        anchor_scale=torch.tensor(1.0),
        applied_log_correction=vector,
        fold_initial_quantiles=matrix,
        fold_transported_quantiles=matrix,
        fold_effective_sizes=matrix,
        fold_refinement_weights=matrix,
        fold_cap_hit_rates=matrix,
    )


def _install_setup_spies(
    monkeypatch: pytest.MonkeyPatch,
    events: list[tuple[object, ...]],
    *,
    stop_after_family: bool,
) -> SimpleNamespace:
    predictor = object()
    cot = SimpleNamespace(
        current_states=lambda: torch.tensor([[[1.0], [2.0]]]),
        actions=torch.zeros(1, 2, dtype=torch.long),
        outcomes=torch.zeros(1, 2, 2),
    )
    splits = SimpleNamespace(
        predictor=predictor,
        behavior=None,
        cot=cot,
        environment=None,
    )
    logging_policy = object()
    task = experiment._Task(
        environment=object(),
        splits=splits,
        n_actions=3,
        logging_policy=logging_policy,
        name="synthetic",
        policy_config=object(),
    )
    outcome_model = object()
    region = object()
    policy = object()
    outcome_sd = torch.tensor([2.0, 4.0])
    cot_scores = torch.tensor([[0.4, 0.8]])
    family = _schedule_family()

    monkeypatch.setattr(
        experiment.torch,
        "manual_seed",
        lambda seed: events.append(("manual_seed", seed)),
    )

    def prepare_task(
        config: ExperimentConfig, *, seed: int, device: str
    ) -> experiment._Task:
        events.append(("prepare_task", seed, device))
        return task

    def fit_model(
        batch: object,
        *,
        n_actions: int,
        config: object,
        device: str,
        seed: int,
        static_indices: tuple[int, ...],
    ) -> object:
        assert batch is predictor
        events.append(("fit_outcome_model", seed, device, n_actions, static_indices))
        return outcome_model

    def fit_region(model: object) -> object:
        assert model is outcome_model
        events.append(("fit_conformal_region",))
        return region

    def build_policy(**kwargs: object) -> object:
        assert kwargs["outcome_model"] is outcome_model
        assert kwargs["reference_policy"] is logging_policy
        assert kwargs["region"] is region
        events.append(("build_policy",))
        return policy

    def training_sd(batch: object) -> torch.Tensor:
        assert batch is predictor
        events.append(("training_outcome_sd",))
        return outcome_sd

    def score(
        fitted_region: object,
        states: torch.Tensor,
        actions: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> torch.Tensor:
        assert fitted_region is region
        assert torch.equal(states, cot.current_states())
        assert actions is cot.actions
        assert outcomes is cot.outcomes
        events.append(("score_cot",))
        return cot_scores

    def fit_family(
        batch: object,
        scores: torch.Tensor,
        *,
        policy: object,
        logging_policy: object,
        outcome_model: object,
        config: ExperimentConfig,
        device: str,
        seed: int,
    ) -> experiment._RefinedScheduleFamily:
        assert batch is cot
        assert scores is cot_scores
        events.append(("fit_schedule_family", seed, device))
        if stop_after_family:
            raise _SetupComplete
        return family

    monkeypatch.setattr(experiment, "_prepare_task", prepare_task)
    monkeypatch.setattr(experiment, "fit_outcome_model", fit_model)
    monkeypatch.setattr(experiment, "fit_conformal_region", fit_region)
    monkeypatch.setattr(experiment, "BehaviorAnchoredPolicy", build_policy)
    monkeypatch.setattr(experiment, "_training_outcome_sd", training_sd)
    monkeypatch.setattr(experiment, "score_batch", score)
    monkeypatch.setattr(
        experiment,
        "_fit_transport_refined_schedule_family",
        fit_family,
    )
    return SimpleNamespace(
        task=task,
        outcome_model=outcome_model,
        region=region,
        policy=policy,
        logging_policy=logging_policy,
        outcome_sd=outcome_sd,
        cot_scores=cot_scores,
        schedule_family=family,
    )


EXPECTED_SETUP_EVENTS = [
    ("manual_seed", 7),
    ("prepare_task", 7, "cpu"),
    ("fit_outcome_model", 8, "cpu", 3, ()),
    ("fit_conformal_region",),
    ("build_policy",),
    ("training_outcome_sd",),
    ("score_cot",),
    ("fit_schedule_family", 7, "cpu"),
]


def test_run_seed_setup_sequence_and_seeds_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    _install_setup_spies(monkeypatch, events, stop_after_family=True)

    with pytest.raises(_SetupComplete):
        experiment.run_seed(ExperimentConfig(), seed=7, device="cpu")

    assert events == EXPECTED_SETUP_EVENTS


def test_oracle_context_matches_existing_run_seed_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    expected = _install_setup_spies(monkeypatch, events, stop_after_family=False)

    experiment.torch.manual_seed(7)
    observed = experiment._prepare_oracle_context(  # type: ignore[attr-defined]
        ExperimentConfig(),
        seed=7,
        device="cpu",
    )

    assert events == EXPECTED_SETUP_EVENTS
    assert observed == experiment._OracleContext(  # type: ignore[attr-defined]
        task=expected.task,
        outcome_model=expected.outcome_model,
        region=expected.region,
        policy=expected.policy,
        logging_policy=expected.logging_policy,
        outcome_sd=expected.outcome_sd,
        cot_scores=expected.cot_scores,
        schedule_family=expected.schedule_family,
    )


def test_phase0_seed_rejects_non_synthetic_before_any_context_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig()
    non_synthetic = replace(
        config,
        data=replace(config.data, dataset="tabular"),
    )
    work_calls: list[str] = []

    def unexpected_seed(*_args: object, **_kwargs: object) -> None:
        work_calls.append("manual_seed")
        raise AssertionError("global RNG must not be touched")

    def unexpected_context(*_args: object, **_kwargs: object) -> None:
        work_calls.append("prepare_context")
        raise AssertionError("context/model work must not start")

    monkeypatch.setattr(phase0_oracle.torch, "manual_seed", unexpected_seed)
    monkeypatch.setattr(
        phase0_oracle,
        "_prepare_oracle_context",
        unexpected_context,
    )

    with pytest.raises(
        ValueError,
        match="run_phase0_seed requires data.dataset='synthetic'",
    ):
        phase0_oracle.run_phase0_seed(non_synthetic, seed=17, device="cpu")

    assert work_calls == []


def test_phase0_seed_returns_paired_rows_and_keeps_streams_and_grids_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        ExperimentConfig(horizon=2, q_grid_size=3),
        samples=SampleConfig(
            logged=5,
            oracle_surface_rollouts=11,
            oracle_rollouts=13,
            online_rollouts=7,
        ),
    )
    profiles = {
        "standard": torch.tensor([1.0, 2.0]),
        "tail_shift": torch.tensor([2.0, 1.0]),
    }
    scores = {
        "standard": torch.tensor(
            [[1.0, 2.0], [2.0, 6.0], [3.0, 8.0], [4.0, 10.0]]
        ),
        "tail_shift": torch.tensor(
            [[2.0, 1.0], [4.0, 2.0], [8.0, 3.0], [10.0, 4.0]]
        ),
    }
    contexts: dict[str, SimpleNamespace] = {}
    manual_seeds: list[int] = []
    grid_calls: list[tuple[torch.Tensor, int, float, float]] = []
    bundle_calls: list[tuple[int, int, int, str]] = []
    tuning_calls: list[tuple[str, str, object]] = []
    candidate_evaluation_counts = {"standard": 0, "tail_shift": 0}
    validated_scenarios: list[str] = []
    evaluation_calls: list[
        tuple[str, dict[str, torch.Tensor], object, set[int]]
    ] = []
    base_synthetic = config.synthetic
    original_validate = ExperimentConfig.validate

    for scenario in ("standard", "tail_shift"):
        family = _schedule_family()
        family = replace(
            family,
            profile=profiles[scenario],
            scale_grid=torch.tensor([0.5, 0.75, 1.0]),
        )
        contexts[scenario] = SimpleNamespace(
            task=SimpleNamespace(environment=SimpleNamespace(scenario=scenario)),
            outcome_model=object(),
            region=object(),
            policy=object(),
            logging_policy=object(),
            outcome_sd=torch.tensor([1.0, 2.0]),
            cot_scores=scores[scenario],
            schedule_family=family,
        )

    monkeypatch.setattr(
        phase0_oracle.torch,
        "manual_seed",
        lambda seed: manual_seeds.append(seed),
    )

    def validate(derived: ExperimentConfig) -> None:
        original_validate(derived)
        validated_scenarios.append(derived.synthetic.scenario)

    monkeypatch.setattr(ExperimentConfig, "validate", validate)

    def prepare_context(
        scenario_config: ExperimentConfig, *, seed: int, device: str
    ) -> SimpleNamespace:
        assert seed == 17
        assert device == "cpu"
        return contexts[scenario_config.synthetic.scenario]

    def fixed_grid(
        values: torch.Tensor,
        *,
        size: int,
        lower_quantile: float,
        upper_quantile: float,
    ) -> torch.Tensor:
        grid_calls.append((values.clone(), size, lower_quantile, upper_quantile))
        return torch.tensor([0.25, 0.50, 0.75]) + 0.1 * (len(grid_calls) % 3)

    def make_noise(
        *, n: int, horizon: int, seed: int, device: str
    ) -> SimpleNamespace:
        bundle_calls.append((n, horizon, seed, device))
        return SimpleNamespace(seed=seed, marker=object())

    def evaluate_candidates(
        environment: object,
        policy: object,
        outcome_model: object,
        *,
        candidate_schedules: torch.Tensor,
        outcome_sd: torch.Tensor,
        noise: object,
        chunk_size: int,
    ) -> phase0_oracle.CandidateMetrics:
        scenario = environment.scenario
        candidate_evaluation_counts[scenario] += 1
        call_kind = (
            "profiled"
            if candidate_evaluation_counts[scenario] == 1
            else "profiled_common_grid"
        )
        tuning_calls.append((scenario, call_kind, noise))
        if scenario == "tail_shift" and call_kind == "profiled_common_grid":
            return phase0_oracle.CandidateMetrics(
                coverage=torch.tensor(
                    [[0.80, 0.82], [0.85, 0.86], [0.87, 0.88]]
                ),
                normalized_width=torch.tensor(
                    [[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]]
                ),
            )
        offset = 0.01 if scenario == "standard" else 0.02
        return phase0_oracle.CandidateMetrics(
            coverage=torch.tensor(
                [
                    [0.80, 0.82],
                    [0.91 + offset, 0.92 + offset],
                    [0.95, 0.96],
                ]
            ),
            normalized_width=torch.tensor(
                [[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]]
            ),
        )

    def greedy_schedule(
        environment: object,
        policy: object,
        outcome_model: object,
        *,
        stage_grids: torch.Tensor,
        outcome_sd: torch.Tensor,
        noise: object,
        target: float,
        chunk_size: int,
    ) -> phase0_oracle.OracleScheduleResult:
        tuning_calls.append((environment.scenario, "greedy", noise))
        if environment.scenario == "tail_shift":
            return phase0_oracle.OracleScheduleResult(
                radii=None,
                selected_indices=(1,),
                tuning_coverage=None,
                tuning_width=None,
                selection_available=False,
                failure_stage=1,
                selected_endpoint=False,
            )
        return phase0_oracle.OracleScheduleResult(
            radii=torch.tensor([0.5, 0.6]),
            selected_indices=(1, 1),
            tuning_coverage=torch.tensor([0.91, 0.92]),
            tuning_width=torch.tensor([1.1, 1.2]),
            selection_available=True,
            failure_stage=None,
            selected_endpoint=False,
        )

    def evaluate_frozen(
        environment: object,
        policy: object,
        outcome_model: object,
        *,
        schedules: dict[str, torch.Tensor],
        noise: object,
        outcome_sd: torch.Tensor,
        forbidden_noise_seeds: set[int],
    ) -> dict[str, phase0_oracle.FrozenOracleEvaluation]:
        evaluation_calls.append(
            (environment.scenario, schedules, noise, forbidden_noise_seeds)
        )
        return {
            name: phase0_oracle.FrozenOracleEvaluation(
                coverage=torch.tensor([0.93, 0.94]),
                wilson_lower_bound=torch.tensor([0.90, 0.91]),
                normalized_width=torch.tensor([1.3, 1.4]),
                micro_normalized_width=1.35,
                patient_normalized_width=1.35,
                n_rollouts=13,
            )
            for name in schedules
        }

    monkeypatch.setattr(
        phase0_oracle,
        "_prepare_oracle_context",
        prepare_context,
        raising=False,
    )
    monkeypatch.setattr(phase0_oracle, "fixed_q_grid", fixed_grid, raising=False)
    monkeypatch.setattr(
        phase0_oracle,
        "make_synthetic_noise_bundle",
        make_noise,
        raising=False,
    )
    monkeypatch.setattr(
        phase0_oracle,
        "evaluate_profiled_candidates_crn",
        evaluate_candidates,
    )
    monkeypatch.setattr(
        phase0_oracle,
        "greedy_sequential_oracle_schedule",
        greedy_schedule,
    )
    monkeypatch.setattr(
        phase0_oracle,
        "evaluate_frozen_schedules_crn",
        evaluate_frozen,
    )

    result = phase0_oracle.run_phase0_seed(  # type: ignore[attr-defined]
        config,
        seed=17,
        device="cpu",
    )

    assert result.seed == 17
    assert result.device == "cpu"
    assert manual_seeds == [17, 17]
    assert validated_scenarios == ["standard", "tail_shift"]
    assert config.synthetic is base_synthetic
    assert config.synthetic.scenario == "standard"
    assert {(row["scenario"], row["method"]) for row in result.records} == {
        ("standard", "Current Profiled Oracle"),
        ("standard", "Greedy Sequential Oracle"),
        ("tail_shift", "Current Profiled Oracle"),
        ("tail_shift", "Greedy Sequential Oracle"),
    }
    assert len(result.records) == 4

    expected_streams = {
        "standard": (18_300_052, 18_400_052),
        "tail_shift": (18_300_053, 18_400_053),
    }
    for row in result.records:
        tuning_seed, evaluation_seed = expected_streams[row["scenario"]]
        assert row["seed"] == 17
        assert row["tuning_seed"] == tuning_seed
        assert row["evaluation_seed"] == evaluation_seed
        assert tuning_seed != evaluation_seed
        for key in (
            "q_by_time",
            "tuning_coverage",
            "tuning_width",
            "final_coverage",
            "final_wilson_lcb",
            "final_stage_width",
        ):
            assert isinstance(json.loads(row[key]), list)

    failed = next(
        row
        for row in result.records
        if row["scenario"] == "tail_shift"
        and row["method"] == "Greedy Sequential Oracle"
    )
    assert failed["selection_available"] is False
    assert failed["failure_stage"] == 1
    assert failed["selected_endpoint"] is False
    assert failed["q_by_time"] == "[]"
    assert failed["tuning_coverage"] == "[]"
    assert failed["tuning_width"] == "[]"
    assert failed["final_coverage"] == "[]"
    assert failed["final_wilson_lcb"] == "[]"
    assert failed["final_stage_width"] == "[]"
    assert math.isnan(failed["micro_normalized_width"])
    assert math.isnan(failed["patient_normalized_width"])
    assert failed["n_rollouts"] == 0

    assert bundle_calls == [
        (11, 2, 18_300_052, "cpu"),
        (13, 2, 18_400_052, "cpu"),
        (11, 2, 18_300_053, "cpu"),
        (13, 2, 18_400_053, "cpu"),
    ]
    for scenario in ("standard", "tail_shift"):
        scenario_tuning = [
            noise for name, _, noise in tuning_calls if name == scenario
        ]
        assert len(scenario_tuning) == 3
        assert scenario_tuning[0] is scenario_tuning[1] is scenario_tuning[2]

    assert len(evaluation_calls) == 2
    expected_profiled_schedules = {
        "standard": torch.tensor([0.75, 1.50]),
        "tail_shift": torch.tensor([1.50, 0.75]),
    }
    for scenario, schedules, noise, forbidden_seeds in evaluation_calls:
        tuning_seed, evaluation_seed = expected_streams[scenario]
        assert noise.seed == evaluation_seed
        assert forbidden_seeds == {tuning_seed}
        assert "profiled" in schedules
        assert torch.equal(
            schedules["profiled"],
            expected_profiled_schedules[scenario],
        )
        if scenario == "standard":
            assert set(schedules) == {
                "profiled",
                "greedy",
                "profiled_common_grid",
            }
            assert torch.equal(
                schedules["profiled_common_grid"],
                torch.tensor([0.5, 1.0]),
            )
        else:
            assert set(schedules) == {"profiled"}

    assert len(grid_calls) == 6
    for scenario_index, scenario in enumerate(("standard", "tail_shift")):
        first = 3 * scenario_index
        assert torch.equal(grid_calls[first][0], scores[scenario][:, 0])
        assert torch.equal(grid_calls[first + 1][0], scores[scenario][:, 1])
        assert torch.equal(
            grid_calls[first + 2][0],
            scores[scenario] / profiles[scenario][None, :],
        )
        assert {call[1:] for call in grid_calls[first : first + 3]} == {
            (3, config.q_quantile_min, config.q_quantile_max)
        }

    expected_surface_keys = {
        f"{scenario}_{name}"
        for scenario in ("standard", "tail_shift")
        for name in (
            "profiled_scale_grid",
            "profile",
            "profiled_candidate_schedules",
            "profiled_candidate_coverage",
            "profiled_candidate_normalized_width",
            "profiled_selected_schedule",
            "greedy_stage_grids",
            "greedy_selected_schedule",
            "profiled_common_grid_scale_grid",
            "profiled_common_grid_candidate_schedules",
            "profiled_common_grid_candidate_coverage",
            "profiled_common_grid_candidate_normalized_width",
            "profiled_common_grid_selected_schedule",
            "profiled_common_grid_final_coverage",
            "profiled_common_grid_final_wilson_lcb",
            "profiled_common_grid_final_stage_width",
            "profiled_common_grid_micro_normalized_width",
            "profiled_common_grid_patient_normalized_width",
            "profiled_common_grid_n_rollouts",
        )
    }
    assert expected_surface_keys <= result.surfaces.keys()
    assert result.surfaces["tail_shift_greedy_selected_schedule"].numel() == 0
    assert torch.equal(
        result.surfaces["standard_profiled_common_grid_selected_schedule"],
        torch.tensor([0.5, 1.0]),
    )
    assert torch.equal(
        result.surfaces["standard_profiled_common_grid_final_coverage"],
        torch.tensor([0.93, 0.94]),
    )
    assert torch.equal(
        result.surfaces["standard_profiled_common_grid_final_wilson_lcb"],
        torch.tensor([0.90, 0.91]),
    )
    assert torch.equal(
        result.surfaces["standard_profiled_common_grid_final_stage_width"],
        torch.tensor([1.3, 1.4]),
    )
    assert result.surfaces[
        "standard_profiled_common_grid_micro_normalized_width"
    ].item() == pytest.approx(1.35)
    assert result.surfaces[
        "standard_profiled_common_grid_patient_normalized_width"
    ].item() == pytest.approx(1.35)
    assert result.surfaces[
        "standard_profiled_common_grid_n_rollouts"
    ].item() == 13

    assert result.surfaces[
        "tail_shift_profiled_common_grid_selected_schedule"
    ].numel() == 0
    assert result.surfaces[
        "tail_shift_profiled_common_grid_final_coverage"
    ].numel() == 0
    assert result.surfaces[
        "tail_shift_profiled_common_grid_final_wilson_lcb"
    ].numel() == 0
    assert result.surfaces[
        "tail_shift_profiled_common_grid_final_stage_width"
    ].numel() == 0
    assert math.isnan(
        result.surfaces[
            "tail_shift_profiled_common_grid_micro_normalized_width"
        ].item()
    )
    assert math.isnan(
        result.surfaces[
            "tail_shift_profiled_common_grid_patient_normalized_width"
        ].item()
    )
    assert result.surfaces[
        "tail_shift_profiled_common_grid_n_rollouts"
    ].item() == 0

    standard_common = result.diagnostics["standard"]["profiled_common_grid"]
    assert standard_common["selection_available"] is True
    assert standard_common["failure_stage"] is None
    assert standard_common["selected_endpoint"] is False
    assert standard_common["selected_indices"] == [1]
    assert standard_common["selected_schedule"] == pytest.approx([0.5, 1.0])
    assert standard_common["final_coverage"] == pytest.approx([0.93, 0.94])
    assert standard_common["final_wilson_lcb"] == pytest.approx([0.90, 0.91])
    assert standard_common["final_stage_width"] == pytest.approx([1.3, 1.4])
    assert standard_common["micro_normalized_width"] == pytest.approx(1.35)
    assert standard_common["patient_normalized_width"] == pytest.approx(1.35)
    assert standard_common["n_rollouts"] == 13
    tail_common = result.diagnostics["tail_shift"]["profiled_common_grid"]
    assert tail_common["selection_available"] is False
    assert tail_common["failure_stage"] == 0
    assert tail_common["selected_endpoint"] is False
    assert tail_common["selected_indices"] == []
    assert tail_common["selected_schedule"] == []
    assert tail_common["final_coverage"] == []
    assert tail_common["final_wilson_lcb"] == []
    assert tail_common["final_stage_width"] == []
    assert math.isnan(tail_common["micro_normalized_width"])
    assert math.isnan(tail_common["patient_normalized_width"])
    assert tail_common["n_rollouts"] == 0
    assert result.diagnostics["standard"]["tuning_seed"] == 18_300_052
    assert result.diagnostics["tail_shift"]["evaluation_seed"] == 18_400_053
