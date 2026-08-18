from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import scpcp.phase0c_study as study
import scpcp.phase0c_joint_search as joint_search
from scpcp.config import ExperimentConfig, SampleConfig
from scpcp.experiment import _RefinedScheduleFamily, _paper_seed
from scpcp.phase0_oracle import (
    CandidateMetrics,
    FrozenOracleEvaluation,
    OracleScheduleResult,
    evaluate_profiled_candidates_crn,
    greedy_sequential_oracle_schedule,
    select_profiled_oracle_schedule,
)
from scpcp.phase0c_joint_search import (
    CoordinateStep,
    JointSearchCheckpoint,
    JointSearchOutcome,
    SearchStart,
    SearchState,
)
from scpcp.simulator import SyntheticNoiseBundle


SCENARIOS = ("standard", "tail_shift")


class _PrefixPolicy:
    def probabilities(
        self,
        states: torch.Tensor,
        q: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert q is not None
        radii = torch.as_tensor(q, dtype=states.dtype, device=states.device)
        if radii.ndim == 0:
            radii = radii.expand(len(states))
        actions = (radii >= 2.0).long()
        return torch.nn.functional.one_hot(actions, num_classes=2).to(states)


class _PrefixEnvironment:
    def initial_state_from_noise(self, noise: SyntheticNoiseBundle) -> torch.Tensor:
        return noise.initial_normal[:, :1]

    def step_from_noise(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        shared: torch.Tensor,
        independent: torch.Tensor,
        innovations: torch.Tensor,
        difficulty_uniform: torch.Tensor,
        contamination_uniform: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del independent, innovations, difficulty_uniform, contamination_uniform
        outcome = (state[:, 0] + 1.0 + shared)[:, None].expand(-1, 2)
        return state + 3.0 * action.to(state)[:, None], outcome


class _PrefixOutcomeModel:
    def __call__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.where(actions == 0, 2.0, 1.0).to(states)
        return states.new_zeros((len(states), 2)), scale[:, None].expand(-1, 2)


def _real_noise(*, horizon: int, seed: int) -> SyntheticNoiseBundle:
    n = 3
    return SyntheticNoiseBundle(
        initial_normal=torch.zeros(n, 6),
        initial_difficulty_uniform=torch.zeros(n),
        action_uniform=torch.linspace(0.2, 0.8, n).expand(horizon, -1).clone(),
        shared_normal=torch.zeros(horizon, n),
        independent_normal=torch.zeros(horizon, n),
        innovation_normal=torch.zeros(horizon, n, 4),
        difficulty_uniform=torch.zeros(horizon, n),
        contamination_uniform=torch.zeros(horizon, n),
        seed=seed,
    )


def _family(horizon: int = 12) -> _RefinedScheduleFamily:
    profile = torch.linspace(1.0, 2.1, horizon)
    scale_grid = torch.tensor([0.5, 0.75, 1.0])
    matrix = profile[None, :]
    return _RefinedScheduleFamily(
        initial_quantiles=profile,
        initial_profile=profile,
        baseline_scale_grid=scale_grid,
        profile=profile,
        scale_grid=scale_grid,
        anchor_scale=torch.tensor(1.0),
        applied_log_correction=profile,
        fold_initial_quantiles=matrix,
        fold_transported_quantiles=matrix,
        fold_effective_sizes=matrix,
        fold_refinement_weights=matrix,
        fold_cap_hit_rates=matrix,
    )


def _config() -> ExperimentConfig:
    return ExperimentConfig(horizon=12, q_grid_size=101)


def _selection(
    radii: torch.Tensor | None,
    *,
    indices: tuple[int, ...],
    width: float,
) -> OracleScheduleResult:
    horizon = 12
    return OracleScheduleResult(
        radii=radii,
        selected_indices=indices,
        tuning_coverage=(
            None if radii is None else torch.full((horizon,), 0.95)
        ),
        tuning_width=(
            None if radii is None else torch.full((horizon,), width)
        ),
        selection_available=radii is not None,
        failure_stage=None if radii is not None else 0,
        selected_endpoint=False,
    )


def _state(
    name: str,
    radii: torch.Tensor,
    indices: tuple[int | None, ...],
    *,
    width: float,
    pair: int,
) -> SearchState:
    return SearchState(
        start_name=name,
        radii=radii.clone(),
        stage_grid_indices=indices,
        coverage=torch.full((12,), 0.95),
        normalized_width=torch.full((12,), width),
        completed_sweep_pairs=pair,
        converged_at_pair=None,
    )


def _step(name: str, pair: int) -> CoordinateStep:
    return CoordinateStep(
        start_name=name,
        sweep_pair=pair,
        direction="forward",
        stage=0,
        feasible_count=101,
        proposed_grid_index=50,
        before_micro_width=2.0,
        proposed_micro_width=1.5,
        committed=True,
        after_micro_width=1.5,
    )


def _checkpoint(
    pair: int,
    starts: tuple[SearchStart, ...],
) -> JointSearchCheckpoint:
    states = tuple(
        _state(
            start.name,
            start.radii,
            start.stage_grid_indices,
            width=0.8 + pair / 100.0 + index / 10.0,
            pair=pair,
        )
        for index, start in enumerate(starts)
    )
    return JointSearchCheckpoint(
        requested_sweep_pairs=pair,
        executed_sweep_pairs=pair,
        best=states[0],
        per_start=states,
        trace=tuple(_step(state.start_name, pair) for state in states),
        schedule_evaluations=pair * 101,
        committed_updates=pair,
    )


def _canonical_parent_states() -> dict[str, tuple[SearchState, ...]]:
    names = ("profiled", "greedy", "upper_endpoint")
    indices = ((None,) * 12, (40,) * 12, (100,) * 12)
    states = tuple(
        SearchState(
            start_name=name,
            radii=torch.ones(12),
            stage_grid_indices=stage_indices,
            coverage=torch.full((12,), 0.95),
            normalized_width=torch.ones(12),
            completed_sweep_pairs=4,
            converged_at_pair=None,
        )
        for name, stage_indices in zip(names, indices, strict=True)
    )
    return {scenario: states for scenario in SCENARIOS}


def _canonical_state_sha256(state: SearchState) -> str:
    parts = [b"phase0c-search-state-v1", state.start_name.encode("utf-8")]
    for tensor in (state.radii, state.coverage, state.normalized_width):
        canonical = tensor.detach().cpu().contiguous()
        parts.extend(
            (
                json.dumps(list(canonical.shape), separators=(",", ":")).encode(),
                str(canonical.dtype).encode("ascii"),
                canonical.reshape(-1).view(torch.uint8).numpy().tobytes(),
            )
        )
    parts.append(
        json.dumps(
            [
                list(state.stage_grid_indices),
                state.completed_sweep_pairs,
                state.converged_at_pair,
            ],
            separators=(",", ":"),
        ).encode()
    )
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _parent_hashes(
    parents: dict[str, tuple[SearchState, ...]],
) -> dict[str, tuple[str, str, str]]:
    return {
        scenario: tuple(_canonical_state_sha256(state) for state in parents[scenario])
        for scenario in SCENARIOS
    }


def _frozen(schedule: torch.Tensor) -> FrozenOracleEvaluation:
    width = schedule.float() / 10.0
    return FrozenOracleEvaluation(
        coverage=torch.full((12,), 0.96),
        wilson_lower_bound=torch.full((12,), 0.91),
        normalized_width=width,
        micro_normalized_width=float(width.mean().item()),
        patient_normalized_width=float(width.mean().item()),
        n_rollouts=50_000,
    )


def _install_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    config = _config()
    family = _family()
    contexts: dict[str, SimpleNamespace] = {}
    for scenario in SCENARIOS:
        contexts[scenario] = SimpleNamespace(
            task=SimpleNamespace(environment=SimpleNamespace(scenario=scenario)),
            policy=object(),
            outcome_model=object(),
            outcome_sd=torch.ones(2),
            cot_scores=torch.arange(1, 61, dtype=torch.float32).reshape(5, 12),
            schedule_family=family,
        )

    events: list[tuple[object, ...]] = []
    noises: dict[tuple[str, int], SimpleNamespace] = {}
    noise_call_count = 0
    nested_starts: dict[str, tuple[SearchStart, ...]] = {}
    initial_pair4: dict[str, tuple[SearchState, ...]] = {}

    def prepare(
        scenario_config: ExperimentConfig, *, seed: int, device: str
    ) -> SimpleNamespace:
        scenario = scenario_config.synthetic.scenario
        events.append(("context", scenario, seed, device))
        return contexts[scenario]

    grid_calls = 0

    def fixed_grid(
        values: torch.Tensor,
        *,
        size: int,
        lower_quantile: float,
        upper_quantile: float,
    ) -> torch.Tensor:
        nonlocal grid_calls
        assert size == 101
        grid_calls += 1
        return torch.linspace(0.5 + float(values[0]) / 1000.0, 2.5, size)

    def make_noise(
        *, n: int, horizon: int, seed: int, device: str
    ) -> SimpleNamespace:
        nonlocal noise_call_count
        scenario = SCENARIOS[(noise_call_count // 2) % 2]
        noise_call_count += 1
        kind = "tuning" if n == 5_000 else "evaluation"
        noise = SimpleNamespace(seed=seed, n=n, horizon=horizon, kind=kind)
        noises[(scenario, n)] = noise
        events.append(("noise", scenario, n, horizon, seed, device))
        return noise

    candidate_call = {scenario: 0 for scenario in SCENARIOS}

    def evaluate_candidates(
        environment: object,
        policy: object,
        outcome_model: object,
        *,
        candidate_schedules: torch.Tensor,
        outcome_sd: torch.Tensor,
        noise: object,
        chunk_size: int,
    ) -> CandidateMetrics:
        scenario = environment.scenario
        candidate_call[scenario] += 1
        assert noise is noises[(scenario, 5_000)]
        events.append(
            ("candidate", scenario, candidate_call[scenario], tuple(candidate_schedules.shape))
        )
        count = len(candidate_schedules)
        widths = torch.arange(1, count + 1, dtype=torch.float32)[:, None].expand(-1, 12)
        return CandidateMetrics(
            coverage=torch.full((count, 12), 0.95),
            normalized_width=widths,
        )

    def profiled_select(
        schedules: torch.Tensor,
        metrics: CandidateMetrics,
        *,
        target: float,
    ) -> OracleScheduleResult:
        assert target == pytest.approx(0.90)
        return _selection(schedules[0], indices=(0,), width=1.0)

    def greedy(
        environment: object,
        policy: object,
        outcome_model: object,
        *,
        stage_grids: torch.Tensor,
        outcome_sd: torch.Tensor,
        noise: object,
        target: float,
        chunk_size: int,
    ) -> OracleScheduleResult:
        scenario = environment.scenario
        assert noise is noises[(scenario, 5_000)]
        events.append(("greedy", scenario, noise))
        return _selection(
            stage_grids[:, 40],
            indices=(40,) * 12,
            width=1.1,
        )

    class Evaluator:
        def __init__(
            self,
            environment: object,
            policy: object,
            outcome_model: object,
            *,
            starts: tuple[SearchStart, ...],
            outcome_sd: torch.Tensor,
            noise: object,
            chunk_size: int,
        ) -> None:
            scenario = environment.scenario
            assert noise is noises[(scenario, 5_000)]
            nested_starts[scenario] = starts
            events.append(("evaluator", scenario, noise, chunk_size))

    def search(
        starts: tuple[SearchStart, ...],
        stage_grids: torch.Tensor,
        evaluator: object,
        *,
        target: float,
        sweep_pair_checkpoints: tuple[int, ...],
        max_wall_seconds: float,
    ) -> JointSearchOutcome:
        scenario = next(
            scenario for scenario, value in nested_starts.items() if value is starts
        )
        assert sweep_pair_checkpoints == (2, 4)
        assert max_wall_seconds > 0.0
        events.append(("search", scenario, sweep_pair_checkpoints))
        pair2 = _checkpoint(2, starts)
        pair4 = _checkpoint(4, starts)
        initial_pair4[scenario] = pair4.per_start
        return JointSearchOutcome(
            status="SELECTED",
            checkpoints={2: pair2, 4: pair4},
            elapsed_seconds=1.25,
        )

    evaluation_calls: list[tuple[str, dict[str, torch.Tensor], object, set[int]]] = []

    def frozen(
        environment: object,
        policy: object,
        outcome_model: object,
        *,
        schedules: dict[str, torch.Tensor],
        noise: object,
        outcome_sd: torch.Tensor,
        forbidden_noise_seeds: set[int],
    ) -> dict[str, FrozenOracleEvaluation]:
        scenario = environment.scenario
        assert noise is noises[(scenario, 50_000)]
        assert forbidden_noise_seeds == {noises[(scenario, 5_000)].seed}
        evaluation_calls.append((scenario, schedules, noise, forbidden_noise_seeds))
        return {name: _frozen(schedule) for name, schedule in schedules.items()}

    monkeypatch.setattr(study, "_prepare_oracle_context", prepare)
    monkeypatch.setattr(study, "fixed_q_grid", fixed_grid)
    monkeypatch.setattr(study, "make_synthetic_noise_bundle", make_noise)
    monkeypatch.setattr(study, "evaluate_profiled_candidates_crn", evaluate_candidates)
    monkeypatch.setattr(study, "select_profiled_oracle_schedule", profiled_select)
    monkeypatch.setattr(study, "greedy_sequential_oracle_schedule", greedy)
    monkeypatch.setattr(study, "CRNCoordinateEvaluator", Evaluator)
    monkeypatch.setattr(study, "cyclic_joint_coordinate_search", search)
    monkeypatch.setattr(study, "evaluate_frozen_schedules_crn", frozen)
    return SimpleNamespace(
        config=config,
        events=events,
        noises=noises,
        nested_starts=nested_starts,
        initial_pair4=initial_pair4,
        evaluation_calls=evaluation_calls,
    )


def test_seed_contract_uses_shared_crn_nested_search_and_one_fresh_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _install_happy_path(monkeypatch)

    result = study.run_phase0c_seed(
        fixture.config,
        seed=17,
        device="cpu",
        max_seed_wall_seconds=60.0,
    )

    assert study.PHASE0C_METHODS == (
        "current_profiled",
        "greedy",
        "joint_B",
        "joint_2B",
    )
    expected_pairs = {
        (scenario, method)
        for scenario in SCENARIOS
        for method in study.PHASE0C_METHODS
    }
    assert len(result.records) == 8
    assert {(row["scenario"], row["method_id"]) for row in result.records} == expected_pairs
    assert [event for event in fixture.events if event[0] == "search"] == [
        ("search", "standard", (2, 4)),
        ("search", "tail_shift", (2, 4)),
    ]
    assert len(fixture.evaluation_calls) == 2
    assert all(set(call[1]) == set(study.PHASE0C_METHODS) for call in fixture.evaluation_calls)
    assert [event[2] for event in fixture.events if event[0] == "noise"] == [
        5_000,
        50_000,
        5_000,
        50_000,
    ]

    stream_ids = {
        (row["scenario"], row["tuning_stream_id"], row["evaluation_stream_id"])
        for row in result.records
    }
    assert stream_ids == {
        (
            "standard",
            _paper_seed(17, 1_300_001),
            _paper_seed(17, 1_400_001),
        ),
        (
            "tail_shift",
            _paper_seed(17, 1_300_002),
            _paper_seed(17, 1_400_002),
        ),
    }
    assert len({value for item in stream_ids for value in item[1:]}) == 4

    for scenario, starts in fixture.nested_starts.items():
        assert tuple(start.name for start in starts) == (
            "profiled",
            "greedy",
            "upper_endpoint",
        )
        assert starts[0].stage_grid_indices == (None,) * 12
        assert starts[1].stage_grid_indices == (40,) * 12
        assert starts[2].stage_grid_indices == (100,) * 12
        assert torch.equal(
            starts[2].radii,
            result.surfaces[f"{scenario}_stage_grids"][:, -1],
        )
        assert torch.equal(
            result.surfaces[f"{scenario}_pair4_profiled_stage_grid_indices"],
            torch.full((12,), -1, dtype=torch.int64),
        )
        assert result.diagnostics[scenario]["active_start_names"] == [
            "profiled",
            "greedy",
            "upper_endpoint",
        ]
        assert result.diagnostics[scenario]["extension_eligible"] is True
        assert result.diagnostics[scenario]["pair4_state_sha256"] == list(
            _parent_hashes(fixture.initial_pair4)[scenario]
        )
        assert torch.equal(
            result.surfaces[f"{scenario}_active_start_names"],
            torch.tensor([0, 1, 2]),
        )
        assert bool(result.surfaces[f"{scenario}_extension_eligible"].item())

    expected_columns = {
        "schema_version",
        "seed",
        "scenario",
        "method_id",
        "analysis_role",
        "budget_id",
        "sweep_pairs",
        "selection_status",
        "selection_available",
        "tuning_joint_feasible",
        "failure_reason",
        "chosen_initialization",
        "selected_endpoint_stage_count",
        "selected_stage_grid_indices_json",
        "q_by_time_json",
        "tuning_coverage_json",
        "tuning_stage_width_json",
        "tuning_micro_width",
        "final_coverage_json",
        "final_wilson_lcb_json",
        "final_stage_width_json",
        "micro_normalized_width",
        "patient_normalized_width",
        "tuning_stream_id",
        "evaluation_stream_id",
        "n_tuning_rollouts",
        "n_evaluation_rollouts",
        "schedule_evaluations",
        "committed_updates",
        "converged_at_pair",
        "wall_time_seconds",
    }
    for row in result.records:
        assert set(row) == expected_columns
        assert row["selection_status"] in {
            "SELECTED",
            "NO_FEASIBLE_START",
            "WALL_TIME_CAP",
        }
        assert row["n_tuning_rollouts"] == 5_000
        assert row["n_evaluation_rollouts"] == 50_000
        assert len(json.loads(row["q_by_time_json"])) == 12
        assert row["micro_normalized_width"] == pytest.approx(
            sum(json.loads(row["final_stage_width_json"])) / 12,
            abs=1e-7,
        )
        assert row["patient_normalized_width"] == pytest.approx(
            row["micro_normalized_width"], abs=1e-7
        )
        json.dumps(row, allow_nan=False)

    current = next(row for row in result.records if row["method_id"] == "current_profiled")
    assert json.loads(current["selected_stage_grid_indices_json"]) == [0]
    assert current["analysis_role"] == "reference"
    assert current["budget_id"] == "REFERENCE"
    assert current["sweep_pairs"] == 0
    assert next(row for row in result.records if row["method_id"] == "joint_B")["budget_id"] == "B"
    assert next(row for row in result.records if row["method_id"] == "joint_2B")["budget_id"] == "2B"

    pair4_keys = {
        f"{scenario}_pair4_{name}_{field}"
        for scenario in SCENARIOS
        for name in ("profiled", "greedy", "upper_endpoint")
        for field in (
            "radii",
            "stage_grid_indices",
            "coverage",
            "normalized_width",
            "completed_sweep_pairs",
            "converged_at_pair",
        )
    }
    assert pair4_keys <= set(result.surfaces)
    trace = result.diagnostics["standard"]["checkpoints"]["4"]["trace"]
    assert set(trace[0]) == set(CoordinateStep.__dataclass_fields__)
    json.dumps(result.diagnostics, allow_nan=False)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("dataset", "synthetic"),
        ("chunk", "candidate_chunk_size"),
        ("checkpoints", "checkpoints"),
        ("zero_cap", "max_seed_wall_seconds"),
        ("nan_cap", "max_seed_wall_seconds"),
    ],
)
def test_seed_rejects_invalid_contract_before_model_work(
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    message: str,
) -> None:
    config = _config()
    kwargs: dict[str, object] = {
        "seed": 17,
        "device": "cpu",
        "candidate_chunk_size": 16,
        "sweep_pair_checkpoints": (2, 4),
        "max_seed_wall_seconds": 60.0,
    }
    if change == "dataset":
        config = replace(config, data=replace(config.data, dataset="tabular"))
    elif change == "chunk":
        kwargs["candidate_chunk_size"] = 0
    elif change == "checkpoints":
        kwargs["sweep_pair_checkpoints"] = (4, 2)
    elif change == "zero_cap":
        kwargs["max_seed_wall_seconds"] = 0.0
    else:
        kwargs["max_seed_wall_seconds"] = math.nan

    calls: list[str] = []
    monkeypatch.setattr(
        study,
        "_prepare_oracle_context",
        lambda *_args, **_kwargs: calls.append("model"),
    )

    with pytest.raises(ValueError, match=message):
        study.run_phase0c_seed(config, **kwargs)
    assert calls == []


def test_unavailable_starts_stay_canonical_and_missing_rows_are_json_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _install_happy_path(monkeypatch)

    monkeypatch.setattr(
        study,
        "select_profiled_oracle_schedule",
        lambda *_args, **_kwargs: _selection(None, indices=(), width=1.0),
    )
    monkeypatch.setattr(
        study,
        "greedy_sequential_oracle_schedule",
        lambda *_args, **_kwargs: _selection(None, indices=(2,), width=1.0),
    )

    def no_feasible(
        starts: tuple[SearchStart, ...],
        *_args: object,
        **_kwargs: object,
    ) -> JointSearchOutcome:
        assert tuple(start.name for start in starts) == (
            "profiled",
            "greedy",
            "upper_endpoint",
        )
        assert starts[0].stage_grid_indices == (None,) * 12
        assert starts[1].stage_grid_indices == (2,) + (-1,) * 11
        return JointSearchOutcome(
            status="NO_FEASIBLE_START",
            checkpoints={},
            elapsed_seconds=0.5,
        )

    monkeypatch.setattr(study, "cyclic_joint_coordinate_search", no_feasible)
    result = study.run_phase0c_seed(
        fixture.config,
        seed=17,
        device="cpu",
        max_seed_wall_seconds=60.0,
    )

    assert len(result.records) == 8
    for row in result.records:
        assert row["selection_status"] == "NO_FEASIBLE_START"
        assert not row["selection_available"]
        assert row["selected_stage_grid_indices_json"] == "[]"
        assert row["q_by_time_json"] == "[]"
        assert row["final_stage_width_json"] == "[]"
        assert math.isnan(row["micro_normalized_width"])
        safe = {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in row.items()}
        json.dumps(safe, allow_nan=False)
    for scenario in SCENARIOS:
        assert result.diagnostics[scenario]["greedy_partial_indices"] == [2]


def test_prepare_context_matches_direct_phase0_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig(horizon=2, q_grid_size=3)
    family = _family(horizon=2)
    environment = _PrefixEnvironment()
    policy = _PrefixPolicy()
    outcome_model = _PrefixOutcomeModel()
    context = SimpleNamespace(
        task=SimpleNamespace(environment=environment),
        policy=policy,
        outcome_model=outcome_model,
        outcome_sd=torch.ones(2),
        cot_scores=torch.tensor([[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]),
        schedule_family=family,
    )
    tuning_noise = _real_noise(horizon=2, seed=_paper_seed(17, 1_300_001))
    stage_grids = torch.tensor([[0.5, 1.0, 2.0], [0.5, 1.0, 2.0]])
    profiled_schedules = family.scale_grid[:, None] * family.profile[None, :]
    profiled_metrics = evaluate_profiled_candidates_crn(
        environment,
        policy,
        outcome_model,
        candidate_schedules=profiled_schedules,
        outcome_sd=context.outcome_sd,
        noise=tuning_noise,
        chunk_size=2,
    )
    direct_profiled = select_profiled_oracle_schedule(
        profiled_schedules, profiled_metrics, target=0.90
    )
    direct_greedy = greedy_sequential_oracle_schedule(
        environment,
        policy,
        outcome_model,
        stage_grids=stage_grids,
        outcome_sd=context.outcome_sd,
        noise=tuning_noise,
        target=0.90,
        chunk_size=2,
    )

    monkeypatch.setattr(study, "_prepare_oracle_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(
        study,
        "fixed_q_grid",
        lambda values, **_kwargs: stage_grids[0 if values[0] == 0.5 else 1],
    )
    monkeypatch.setattr(
        study,
        "make_synthetic_noise_bundle",
        lambda *, horizon, seed, **_kwargs: _real_noise(horizon=horizon, seed=seed),
    )

    got = study.prepare_phase0c_scenario_context(
        config,
        seed=17,
        scenario="standard",
        scenario_index=0,
        device="cpu",
        candidate_chunk_size=2,
    )
    assert got.profiled_selection.selected_indices == direct_profiled.selected_indices
    assert torch.equal(got.profiled_selection.radii, direct_profiled.radii)
    assert torch.equal(
        got.profiled_selection.tuning_coverage,
        direct_profiled.tuning_coverage,
    )
    assert got.greedy_selection.selected_indices == direct_greedy.selected_indices
    assert torch.equal(got.greedy_selection.radii, direct_greedy.radii)


def test_pair2_survives_pair4_deadline_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _install_happy_path(monkeypatch)

    def deadline(
        starts: tuple[SearchStart, ...],
        *_args: object,
        **_kwargs: object,
    ) -> JointSearchOutcome:
        return JointSearchOutcome(
            status="WALL_TIME_CAP",
            checkpoints={2: _checkpoint(2, starts)},
            elapsed_seconds=59.0,
        )

    monkeypatch.setattr(study, "cyclic_joint_coordinate_search", deadline)
    result = study.run_phase0c_seed(
        fixture.config,
        seed=17,
        device="cpu",
        max_seed_wall_seconds=60.0,
    )
    for scenario in SCENARIOS:
        rows = {row["method_id"]: row for row in result.records if row["scenario"] == scenario}
        assert rows["joint_B"]["selection_status"] == "SELECTED"
        assert rows["joint_2B"]["selection_status"] == "WALL_TIME_CAP"
        assert rows["joint_2B"]["q_by_time_json"] == "[]"
        assert result.diagnostics[scenario]["extension_eligible"] is False
        assert not bool(result.surfaces[f"{scenario}_extension_eligible"].item())


def test_both_scenarios_share_one_seed_wall_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _install_happy_path(monkeypatch)
    clock = iter((100.0, 110.0, 135.0))
    remaining_budgets: list[float] = []
    monkeypatch.setattr(study.time, "monotonic", lambda: next(clock))

    def search(
        starts: tuple[SearchStart, ...],
        *_args: object,
        max_wall_seconds: float,
        **_kwargs: object,
    ) -> JointSearchOutcome:
        remaining_budgets.append(max_wall_seconds)
        return JointSearchOutcome(
            status="SELECTED",
            checkpoints={2: _checkpoint(2, starts), 4: _checkpoint(4, starts)},
            elapsed_seconds=1.0,
        )

    monkeypatch.setattr(study, "cyclic_joint_coordinate_search", search)
    study.run_phase0c_seed(
        fixture.config,
        seed=17,
        device="cpu",
        max_seed_wall_seconds=60.0,
    )
    assert remaining_budgets == [50.0, 25.0]


def test_available_selector_vector_length_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _install_happy_path(monkeypatch)
    malformed = _selection(torch.ones(11), indices=(0,), width=1.0)
    monkeypatch.setattr(
        study,
        "select_profiled_oracle_schedule",
        lambda *_args, **_kwargs: malformed,
    )
    with pytest.raises(ValueError, match="profiled radii must have shape"):
        study.run_phase0c_seed(
            fixture.config,
            seed=17,
            device="cpu",
            max_seed_wall_seconds=60.0,
        )


@pytest.mark.parametrize(
    "config",
    [
        ExperimentConfig(horizon=11, q_grid_size=101),
        ExperimentConfig(horizon=12, q_grid_size=99),
        replace(
            ExperimentConfig(),
            samples=SampleConfig(
                logged=5_000,
                oracle_surface_rollouts=4_999,
                oracle_rollouts=50_000,
                online_rollouts=2_000,
            ),
        ),
        replace(
            ExperimentConfig(),
            samples=SampleConfig(
                logged=5_000,
                oracle_surface_rollouts=5_000,
                oracle_rollouts=49_999,
                online_rollouts=2_000,
            ),
        ),
    ],
)
def test_seed_rejects_noncanonical_scientific_shape_before_model_work(
    monkeypatch: pytest.MonkeyPatch,
    config: ExperimentConfig,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        study,
        "_prepare_oracle_context",
        lambda *_args, **_kwargs: calls.append("model"),
    )
    with pytest.raises(ValueError, match="Phase 0C protocol"):
        study.run_phase0c_seed(
            config,
            seed=17,
            device="cpu",
            max_seed_wall_seconds=60.0,
        )
    assert calls == []


@pytest.mark.parametrize("candidate_chunk_size", (True, 1.5))
def test_seed_rejects_non_integer_chunk_before_model_work(
    monkeypatch: pytest.MonkeyPatch,
    candidate_chunk_size: object,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        study,
        "_prepare_oracle_context",
        lambda *_args, **_kwargs: calls.append("model"),
    )
    with pytest.raises(ValueError, match="candidate_chunk_size"):
        study.run_phase0c_seed(
            _config(),
            seed=17,
            device="cpu",
            candidate_chunk_size=candidate_chunk_size,  # type: ignore[arg-type]
            max_seed_wall_seconds=60.0,
        )
    assert calls == []


@pytest.mark.parametrize("checkpoints", ((2.0, 4.0), [2, 4], (True, 4)))
def test_seed_rejects_non_exact_checkpoint_tuple_before_model_work(
    monkeypatch: pytest.MonkeyPatch,
    checkpoints: object,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        study,
        "_prepare_oracle_context",
        lambda *_args, **_kwargs: calls.append("model"),
    )
    with pytest.raises(ValueError, match="checkpoints"):
        study.run_phase0c_seed(
            _config(),
            seed=17,
            device="cpu",
            sweep_pair_checkpoints=checkpoints,  # type: ignore[arg-type]
            max_seed_wall_seconds=60.0,
        )
    assert calls == []


def test_extension_rejects_boolean_wall_cap_before_stream_or_model_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = _canonical_parent_states()
    work: list[str] = []
    monkeypatch.setattr(study, "_paper_seed", lambda *_args: work.append("stream"))
    monkeypatch.setattr(
        study,
        "_prepare_oracle_context",
        lambda *_args, **_kwargs: work.append("model"),
    )
    with pytest.raises(ValueError, match="max_seed_wall_seconds"):
        study.run_phase0c_extension_seed(
            _config(),
            seed=17,
            device="cpu",
            pair4_states=parents,
            pair4_state_sha256=_parent_hashes(parents),
            extension_eligible={scenario: True for scenario in SCENARIOS},
            max_seed_wall_seconds=True,
        )
    assert work == []


def test_joint_endpoint_count_is_derived_from_all_twelve_stage_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _install_happy_path(monkeypatch)

    def upper_best(
        starts: tuple[SearchStart, ...],
        *_args: object,
        **_kwargs: object,
    ) -> JointSearchOutcome:
        pair2 = _checkpoint(2, starts)
        pair4 = _checkpoint(4, starts)
        pair2 = replace(pair2, best=pair2.per_start[2])
        pair4 = replace(pair4, best=pair4.per_start[2])
        return JointSearchOutcome(
            status="SELECTED",
            checkpoints={2: pair2, 4: pair4},
            elapsed_seconds=1.0,
        )

    monkeypatch.setattr(study, "cyclic_joint_coordinate_search", upper_best)
    result = study.run_phase0c_seed(
        fixture.config,
        seed=17,
        device="cpu",
        max_seed_wall_seconds=60.0,
    )
    joint_rows = [row for row in result.records if row["analysis_role"] == "joint_search"]
    assert len(joint_rows) == 4
    assert all(row["selected_endpoint_stage_count"] == 12 for row in joint_rows)


def test_extension_resumes_validated_pair4_states_and_rejects_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _install_happy_path(monkeypatch)
    initial = study.run_phase0c_seed(
        fixture.config,
        seed=17,
        device="cpu",
        max_seed_wall_seconds=60.0,
    )
    parent_states = fixture.initial_pair4
    parent_hashes = {
        scenario: tuple(initial.diagnostics[scenario]["pair4_state_sha256"])
        for scenario in SCENARIOS
    }
    resumed: list[tuple[str, tuple[SearchState, ...]]] = []
    original_replay = study.evaluate_profiled_candidates_crn

    def replay(
        environment: object,
        policy: object,
        outcome_model: object,
        *,
        candidate_schedules: torch.Tensor,
        outcome_sd: torch.Tensor,
        noise: object,
        chunk_size: int,
    ) -> CandidateMetrics:
        states = parent_states[environment.scenario]
        if torch.equal(candidate_schedules, torch.stack([state.radii for state in states])):
            return CandidateMetrics(
                coverage=torch.stack([state.coverage for state in states]),
                normalized_width=torch.stack([state.normalized_width for state in states]),
            )
        return original_replay(
            environment,
            policy,
            outcome_model,
            candidate_schedules=candidate_schedules,
            outcome_sd=outcome_sd,
            noise=noise,
            chunk_size=chunk_size,
        )

    def resume(
        states: tuple[SearchState, ...],
        stage_grids: torch.Tensor,
        evaluator: object,
        *,
        target: float,
        max_wall_seconds: float,
    ) -> JointSearchOutcome:
        scenario = next(
            name for name, expected in parent_states.items() if states is expected
        )
        assert all(state.completed_sweep_pairs == 4 for state in states)
        resumed.append((scenario, states))
        checkpoint = _checkpoint(
            8,
            tuple(
                SearchStart(
                    name=state.start_name,
                    radii=state.radii,
                    stage_grid_indices=state.stage_grid_indices,
                    coverage=state.coverage,
                    normalized_width=state.normalized_width,
                )
                for state in states
            ),
        )
        return JointSearchOutcome(
            status="SELECTED",
            checkpoints={8: checkpoint},
            elapsed_seconds=2.0,
        )

    monkeypatch.setattr(study, "evaluate_profiled_candidates_crn", replay)
    monkeypatch.setattr(study, "resume_cyclic_joint_coordinate_search", resume)
    fixture.evaluation_calls.clear()
    extension = study.run_phase0c_extension_seed(
        fixture.config,
        seed=17,
        device="cpu",
        pair4_states=parent_states,
        pair4_state_sha256=parent_hashes,
        extension_eligible={scenario: True for scenario in SCENARIOS},
        candidate_chunk_size=16,
        max_seed_wall_seconds=60.0,
    )

    assert [(row["scenario"], row["method_id"]) for row in extension.records] == [
        ("standard", "joint_8SP"),
        ("tail_shift", "joint_8SP"),
    ]
    assert len(resumed) == 2
    assert len(fixture.evaluation_calls) == 2
    assert all(set(call[1]) == {"joint_8SP"} for call in fixture.evaluation_calls)
    for row in extension.records:
        assert row["evaluation_stream_id"] == _paper_seed(
            17, 1_400_001 + SCENARIOS.index(row["scenario"])
        )
        assert row["sweep_pairs"] == 8
        assert row["budget_id"] == "8SP"
    assert initial.records[0]["evaluation_stream_id"] == extension.records[0]["evaluation_stream_id"]

    mutated = {scenario: states for scenario, states in parent_states.items()}
    first = mutated["standard"][0]
    changed = first.radii.clone()
    changed.view(torch.uint8)[0] ^= 1
    mutated["standard"] = (replace(first, radii=changed), *mutated["standard"][1:])
    with pytest.raises(ValueError, match="parent pair-4"):
        study.run_phase0c_extension_seed(
            fixture.config,
            seed=17,
            device="cpu",
            pair4_states=mutated,
            pair4_state_sha256=parent_hashes,
            extension_eligible={scenario: True for scenario in SCENARIOS},
            max_seed_wall_seconds=60.0,
        )


def test_extension_validates_both_parents_before_any_continuation_or_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _install_happy_path(monkeypatch)
    study.run_phase0c_seed(
        fixture.config,
        seed=17,
        device="cpu",
        max_seed_wall_seconds=60.0,
    )
    parents = dict(fixture.initial_pair4)
    original_replay = study.evaluate_profiled_candidates_crn
    events: list[str] = []
    remaining_budgets: list[float] = []

    def replay(
        environment: object,
        policy: object,
        outcome_model: object,
        *,
        candidate_schedules: torch.Tensor,
        outcome_sd: torch.Tensor,
        noise: object,
        chunk_size: int,
    ) -> CandidateMetrics:
        scenario = environment.scenario
        parent_schedules = torch.stack([state.radii for state in parents[scenario]])
        if torch.equal(candidate_schedules.cpu(), parent_schedules.cpu()):
            events.append(f"validate:{scenario}")
            return CandidateMetrics(
                coverage=torch.stack([state.coverage for state in parents[scenario]]),
                normalized_width=torch.stack(
                    [state.normalized_width for state in parents[scenario]]
                ),
            )
        return original_replay(
            environment,
            policy,
            outcome_model,
            candidate_schedules=candidate_schedules,
            outcome_sd=outcome_sd,
            noise=noise,
            chunk_size=chunk_size,
        )

    def forbidden_initial_search(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("extension must not rerun pair1-4 search")

    def resume(
        states: tuple[SearchState, ...],
        stage_grids: torch.Tensor,
        evaluator: object,
        *,
        target: float,
        max_wall_seconds: float,
    ) -> JointSearchOutcome:
        scenario = next(name for name, value in parents.items() if states is value)
        events.append(f"continue:{scenario}")
        remaining_budgets.append(max_wall_seconds)
        checkpoint = _checkpoint(
            8,
            tuple(
                SearchStart(
                    name=state.start_name,
                    radii=state.radii,
                    stage_grid_indices=state.stage_grid_indices,
                    coverage=state.coverage,
                    normalized_width=state.normalized_width,
                )
                for state in states
            ),
        )
        return JointSearchOutcome(
            status="SELECTED",
            checkpoints={8: checkpoint},
            elapsed_seconds=1.0,
        )

    original_frozen = study.evaluate_frozen_schedules_crn

    def frozen(*args: object, **kwargs: object) -> dict[str, FrozenOracleEvaluation]:
        environment = args[0]
        events.append(f"eval:{environment.scenario}")
        return original_frozen(*args, **kwargs)

    now = [0.0]
    original_resume_evaluator = study._resume_evaluator

    def build_resume_evaluator(*args: object, **kwargs: object) -> object:
        scenario = args[0].environment.scenario
        events.append(f"cache:{scenario}")
        now[0] += 10.0
        return original_resume_evaluator(*args, **kwargs)

    monkeypatch.setattr(study.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(study, "evaluate_profiled_candidates_crn", replay)
    monkeypatch.setattr(study, "cyclic_joint_coordinate_search", forbidden_initial_search)
    monkeypatch.setattr(study, "_resume_evaluator", build_resume_evaluator)
    monkeypatch.setattr(study, "resume_cyclic_joint_coordinate_search", resume)
    monkeypatch.setattr(study, "evaluate_frozen_schedules_crn", frozen)

    result = study.run_phase0c_extension_seed(
        fixture.config,
        seed=17,
        device="cpu",
        pair4_states=parents,
        pair4_state_sha256=_parent_hashes(parents),
        extension_eligible={scenario: True for scenario in SCENARIOS},
        max_seed_wall_seconds=100.0,
    )

    assert events == [
        "validate:standard",
        "validate:tail_shift",
        "cache:standard",
        "continue:standard",
        "cache:tail_shift",
        "continue:tail_shift",
        "eval:standard",
        "eval:tail_shift",
    ]
    assert remaining_budgets == [90.0, 80.0]
    assert len(result.records) == 2


@pytest.mark.parametrize(
    ("cap_phase", "expected_resume_calls"),
    (("standard_continuation", ["standard"]), ("before_fresh", list(SCENARIOS))),
)
def test_extension_wall_cap_is_atomic_two_unavailable_rows(
    monkeypatch: pytest.MonkeyPatch,
    cap_phase: str,
    expected_resume_calls: list[str],
) -> None:
    fixture = _install_happy_path(monkeypatch)
    initial = study.run_phase0c_seed(
        fixture.config,
        seed=17,
        device="cpu",
        max_seed_wall_seconds=60.0,
    )
    parents = dict(fixture.initial_pair4)
    parent_hashes = {
        scenario: tuple(initial.diagnostics[scenario]["pair4_state_sha256"])
        for scenario in SCENARIOS
    }
    original_replay = study.evaluate_profiled_candidates_crn

    def replay(
        environment: object,
        policy: object,
        outcome_model: object,
        *,
        candidate_schedules: torch.Tensor,
        outcome_sd: torch.Tensor,
        noise: object,
        chunk_size: int,
    ) -> CandidateMetrics:
        states = parents[environment.scenario]
        if torch.equal(candidate_schedules, torch.stack([state.radii for state in states])):
            return CandidateMetrics(
                coverage=torch.stack([state.coverage for state in states]),
                normalized_width=torch.stack([state.normalized_width for state in states]),
            )
        return original_replay(
            environment,
            policy,
            outcome_model,
            candidate_schedules=candidate_schedules,
            outcome_sd=outcome_sd,
            noise=noise,
            chunk_size=chunk_size,
        )

    now = [0.0]
    tail_finished = [False]
    after_tail_clock_reads = [0]
    resume_calls: list[str] = []

    def resume(
        states: tuple[SearchState, ...],
        stage_grids: torch.Tensor,
        evaluator: object,
        *,
        target: float,
        max_wall_seconds: float,
    ) -> JointSearchOutcome:
        scenario = next(name for name, value in parents.items() if states is value)
        resume_calls.append(scenario)
        if cap_phase == "standard_continuation" and scenario == "standard":
            now[0] = 100.0
            return JointSearchOutcome(
                status="WALL_TIME_CAP",
                checkpoints={},
                elapsed_seconds=100.0,
            )
        checkpoint = _checkpoint(
            8,
            tuple(
                SearchStart(
                    name=state.start_name,
                    radii=state.radii,
                    stage_grid_indices=state.stage_grid_indices,
                    coverage=state.coverage,
                    normalized_width=state.normalized_width,
                )
                for state in states
            ),
        )
        if cap_phase == "before_fresh" and scenario == "tail_shift":
            tail_finished[0] = True
        return JointSearchOutcome(
            status="SELECTED",
            checkpoints={8: checkpoint},
            elapsed_seconds=1.0,
        )

    def clock() -> float:
        if cap_phase == "before_fresh" and tail_finished[0]:
            after_tail_clock_reads[0] += 1
            return 0.0 if after_tail_clock_reads[0] == 1 else 100.0
        return now[0]

    monkeypatch.setattr(study.time, "monotonic", clock)
    monkeypatch.setattr(study, "evaluate_profiled_candidates_crn", replay)
    monkeypatch.setattr(study, "resume_cyclic_joint_coordinate_search", resume)
    fixture.evaluation_calls.clear()

    result = study.run_phase0c_extension_seed(
        fixture.config,
        seed=17,
        device="cpu",
        pair4_states=parents,
        pair4_state_sha256=parent_hashes,
        extension_eligible={scenario: True for scenario in SCENARIOS},
        max_seed_wall_seconds=100.0,
    )

    assert [(row["scenario"], row["method_id"]) for row in result.records] == [
        ("standard", "joint_8SP"),
        ("tail_shift", "joint_8SP"),
    ]
    assert resume_calls == expected_resume_calls
    assert fixture.evaluation_calls == []
    for row in result.records:
        assert row["selection_status"] == "WALL_TIME_CAP"
        assert row["selection_available"] is False
        assert row["q_by_time_json"] == "[]"
        assert row["tuning_coverage_json"] == "[]"
        assert row["final_coverage_json"] == "[]"
        assert row["n_evaluation_rollouts"] == 0
        assert result.surfaces[f"{row['scenario']}_joint_8SP_schedule"].numel() == 0
        assert result.diagnostics[row["scenario"]]["search_status"] == "WALL_TIME_CAP"
        assert result.diagnostics[row["scenario"]]["wall_time_phase"] == cap_phase


@pytest.mark.parametrize(
    "mutation",
    (
        "scenario_keys",
        "states_list",
        "state_count",
        "duplicate_name",
        "completed_pair",
        "radii_shape",
        "bool_index",
        "nonfinite_coverage",
        "mixed_convergence",
    ),
)
def test_extension_rejects_structurally_bad_parent_before_stream_or_model_work(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    parents = _canonical_parent_states()
    parent_hashes = _parent_hashes(parents)
    if mutation == "scenario_keys":
        parents.pop("tail_shift")
    elif mutation == "states_list":
        parents["standard"] = list(parents["standard"])  # type: ignore[assignment]
    elif mutation == "state_count":
        parents["standard"] = parents["standard"][:2]
    elif mutation == "duplicate_name":
        states = parents["standard"]
        parents["standard"] = (states[0], replace(states[1], start_name="profiled"), states[2])
    elif mutation == "completed_pair":
        states = parents["standard"]
        parents["standard"] = (replace(states[0], completed_sweep_pairs=3), *states[1:])
    elif mutation == "radii_shape":
        states = parents["standard"]
        parents["standard"] = (replace(states[0], radii=torch.ones(11)), *states[1:])
    elif mutation == "bool_index":
        states = parents["standard"]
        parents["standard"] = (
            replace(states[0], stage_grid_indices=(True,) + (None,) * 11),
            *states[1:],
        )
    elif mutation == "nonfinite_coverage":
        states = parents["standard"]
        bad = states[0].coverage.clone()
        bad[0] = torch.nan
        parents["standard"] = (replace(states[0], coverage=bad), *states[1:])
    else:
        states = parents["standard"]
        parents["standard"] = (
            replace(states[0], converged_at_pair=2),
            states[1],
            states[2],
        )

    work: list[str] = []
    monkeypatch.setattr(
        study,
        "_paper_seed",
        lambda *_args, **_kwargs: work.append("stream"),
    )
    monkeypatch.setattr(
        study,
        "_prepare_oracle_context",
        lambda *_args, **_kwargs: work.append("model"),
    )
    with pytest.raises((TypeError, ValueError), match="parent pair-4"):
        study.run_phase0c_extension_seed(
            _config(),
            seed=17,
            device="cpu",
            pair4_states=parents,
            pair4_state_sha256=parent_hashes,
            extension_eligible={scenario: True for scenario in SCENARIOS},
            max_seed_wall_seconds=60.0,
        )
    assert work == []


def test_extension_rejects_convergence_mutation_against_original_hash_before_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = _canonical_parent_states()
    parent_hashes = _parent_hashes(parents)
    states = parents["standard"]
    parents["standard"] = tuple(
        replace(state, converged_at_pair=2) for state in states
    )
    work: list[str] = []
    monkeypatch.setattr(study, "_paper_seed", lambda *_args: work.append("stream"))
    monkeypatch.setattr(
        study,
        "_prepare_oracle_context",
        lambda *_args, **_kwargs: work.append("model"),
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        study.run_phase0c_extension_seed(
            _config(),
            seed=17,
            device="cpu",
            pair4_states=parents,
            pair4_state_sha256=parent_hashes,
            extension_eligible={scenario: True for scenario in SCENARIOS},
            max_seed_wall_seconds=60.0,
        )
    assert work == []


@pytest.mark.parametrize(
    "mutation",
    ("scenario_shape", "value_shape", "hash_count", "bad_hex", "wrong_order"),
)
def test_extension_rejects_malformed_parent_hashes_before_stream_or_model_work(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    parents = _canonical_parent_states()
    hashes: object = _parent_hashes(parents)
    if mutation == "scenario_shape":
        hashes = {"standard": hashes["standard"]}  # type: ignore[index]
    elif mutation == "value_shape":
        hashes["standard"] = list(hashes["standard"])  # type: ignore[index]
    elif mutation == "hash_count":
        hashes["standard"] = hashes["standard"][:2]  # type: ignore[index]
    elif mutation == "bad_hex":
        hashes["standard"] = ("G" * 64, *hashes["standard"][1:])  # type: ignore[index]
    else:
        hashes["standard"] = tuple(reversed(hashes["standard"]))  # type: ignore[index]

    work: list[str] = []
    monkeypatch.setattr(study, "_paper_seed", lambda *_args: work.append("stream"))
    monkeypatch.setattr(
        study,
        "_prepare_oracle_context",
        lambda *_args, **_kwargs: work.append("model"),
    )
    with pytest.raises(ValueError, match="pair4_state_sha256"):
        study.run_phase0c_extension_seed(
            _config(),
            seed=17,
            device="cpu",
            pair4_states=parents,
            pair4_state_sha256=hashes,  # type: ignore[arg-type]
            extension_eligible={scenario: True for scenario in SCENARIOS},
            max_seed_wall_seconds=60.0,
        )
    assert work == []


@pytest.mark.parametrize(
    "mutation",
    (
        "radii_raw_bytes",
        "coverage_dtype",
        "width_raw_bytes",
        "name",
        "indices",
        "completed_pair",
        "convergence",
    ),
)
def test_parent_state_sha256_is_byte_and_metadata_exact(mutation: str) -> None:
    base = _canonical_parent_states()["standard"][0]
    if mutation == "radii_raw_bytes":
        base = replace(base, radii=torch.zeros(12))
        changed = base.radii.clone()
        changed[0] = -0.0
        candidate = replace(base, radii=changed)
    elif mutation == "coverage_dtype":
        candidate = replace(base, coverage=base.coverage.double())
    elif mutation == "width_raw_bytes":
        base = replace(base, normalized_width=torch.zeros(12))
        changed = base.normalized_width.clone()
        changed[0] = -0.0
        candidate = replace(base, normalized_width=changed)
    elif mutation == "name":
        candidate = replace(base, start_name="greedy")
    elif mutation == "indices":
        candidate = replace(base, stage_grid_indices=(0,) + (None,) * 11)
    elif mutation == "completed_pair":
        candidate = replace(base, completed_sweep_pairs=3)
    else:
        candidate = replace(base, converged_at_pair=2)
    assert study._state_sha256(base) == _canonical_state_sha256(base)
    assert study._state_sha256(base) != study._state_sha256(candidate)


def test_real_task1_partial_active_search_is_available_but_not_extension_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _install_happy_path(monkeypatch)
    original_candidates = study.evaluate_profiled_candidates_crn
    monkeypatch.setattr(
        study,
        "select_profiled_oracle_schedule",
        lambda *_args, **_kwargs: _selection(None, indices=(), width=1.0),
    )

    def candidates(*args: object, **kwargs: object) -> CandidateMetrics:
        schedules = kwargs["candidate_schedules"]
        if len(schedules) == 1:
            return CandidateMetrics(
                coverage=torch.full((1, 12), 0.80),
                normalized_width=torch.full((1, 12), 1.5),
            )
        return original_candidates(*args, **kwargs)

    class PartialEvaluator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __call__(
            self,
            start_name: str,
            incumbent: torch.Tensor,
            stage: int,
            grid: torch.Tensor,
        ) -> CandidateMetrics:
            assert start_name == "greedy"
            return CandidateMetrics(
                coverage=torch.full((len(grid), 12), 0.95),
                normalized_width=torch.full((len(grid), 12), 1.1),
            )

    monkeypatch.setattr(study, "evaluate_profiled_candidates_crn", candidates)
    monkeypatch.setattr(study, "CRNCoordinateEvaluator", PartialEvaluator)
    monkeypatch.setattr(
        study,
        "cyclic_joint_coordinate_search",
        joint_search.cyclic_joint_coordinate_search,
    )
    result = study.run_phase0c_seed(
        fixture.config,
        seed=17,
        device="cpu",
        max_seed_wall_seconds=60.0,
    )

    for scenario in SCENARIOS:
        rows = {row["method_id"]: row for row in result.records if row["scenario"] == scenario}
        assert rows["joint_B"]["selection_status"] == "SELECTED"
        assert rows["joint_2B"]["selection_status"] == "SELECTED"
        assert result.diagnostics[scenario]["active_start_names"] == ["greedy"]
        assert result.diagnostics[scenario]["extension_eligible"] is False
        assert torch.equal(
            result.surfaces[f"{scenario}_active_start_names"],
            torch.tensor([1]),
        )
        assert not bool(result.surfaces[f"{scenario}_extension_eligible"].item())


@pytest.mark.parametrize(
    "eligibility",
    (
        {"standard": True},
        {"standard": True, "tail_shift": False},
        {"standard": True, "tail_shift": 1},
    ),
)
def test_extension_rejects_ineligible_parent_before_stream_or_model_work(
    monkeypatch: pytest.MonkeyPatch,
    eligibility: dict[str, object],
) -> None:
    parents = _canonical_parent_states()
    work: list[str] = []
    monkeypatch.setattr(study, "_paper_seed", lambda *_args: work.append("stream"))
    monkeypatch.setattr(
        study,
        "_prepare_oracle_context",
        lambda *_args, **_kwargs: work.append("model"),
    )
    with pytest.raises((TypeError, ValueError), match="extension_eligible"):
        study.run_phase0c_extension_seed(
            _config(),
            seed=17,
            device="cpu",
            pair4_states=parents,
            pair4_state_sha256=_parent_hashes(parents),
            extension_eligible=eligibility,  # type: ignore[arg-type]
            max_seed_wall_seconds=60.0,
        )
    assert work == []
