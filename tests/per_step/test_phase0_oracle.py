from __future__ import annotations

import weakref

import torch
from torch import Tensor

import scpcp.phase0_oracle as phase0_oracle
from scpcp.phase0_oracle import (
    CandidateMetrics,
    evaluate_profiled_candidates_crn,
    greedy_sequential_oracle_schedule,
    select_profiled_oracle_schedule,
)
from scpcp.simulator import SyntheticNoiseBundle


class RadiusActionPolicy:
    def __init__(self) -> None:
        self.max_batch = 0

    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        assert q is not None
        self.max_batch = max(self.max_batch, len(states))
        radii = torch.as_tensor(q, dtype=states.dtype, device=states.device)
        if radii.ndim == 0:
            radii = radii.expand(len(states))
        actions = torch.where(
            radii < 0.75,
            torch.zeros_like(radii, dtype=torch.long),
            torch.where(
                radii < 1.25,
                torch.ones_like(radii, dtype=torch.long),
                torch.full_like(radii, 2, dtype=torch.long),
            ),
        )
        return torch.nn.functional.one_hot(actions, num_classes=3).to(states)


class NonMonotoneEnvironment:
    def initial_state_from_noise(self, noise: SyntheticNoiseBundle) -> Tensor:
        return noise.initial_normal[:, :1]

    def step_from_noise(
        self,
        state: Tensor,
        action: Tensor,
        *,
        shared: Tensor,
        independent: Tensor,
        innovations: Tensor,
        difficulty_uniform: Tensor,
        contamination_uniform: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del independent, difficulty_uniform, contamination_uniform
        residual = torch.tensor([0.4, 1.2, 0.2], device=state.device)[action]
        outcome = (residual + shared)[:, None].expand(-1, 2)
        next_state = state + action.to(state)[:, None] + innovations[:, :1]
        return next_state, outcome


class ActionScaleOutcomeModel:
    n_actions = 3

    def __init__(self) -> None:
        self.max_batch = 0

    def __call__(self, states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        self.max_batch = max(self.max_batch, len(states))
        scale = torch.tensor([1.0, 1.0, 0.2], device=states.device)[actions]
        return states.new_zeros((len(states), 2)), scale[:, None].expand(-1, 2)

    def predict_all_actions(self, states: Tensor) -> tuple[Tensor, Tensor]:
        repeated_states = states[:, None, :].expand(-1, self.n_actions, -1)
        flat_states = repeated_states.reshape(len(states) * self.n_actions, -1)
        actions = torch.arange(self.n_actions).repeat(len(states)).to(states.device)
        means, scales = self(flat_states, actions)
        return (
            means.reshape(len(states), self.n_actions, -1),
            scales.reshape(len(states), self.n_actions, -1),
        )


class PredictorMediatedRadiusPolicy(RadiusActionPolicy):
    def __init__(self, predictor: ActionScaleOutcomeModel) -> None:
        super().__init__()
        self.predictor = predictor

    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        self.predictor.predict_all_actions(states)
        return super().probabilities(states, q)


class PrefixPolicy:
    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        assert q is not None
        radii = torch.as_tensor(q, dtype=states.dtype, device=states.device)
        if radii.ndim == 0:
            radii = radii.expand(len(states))
        actions = (radii >= 2.0).long()
        return torch.nn.functional.one_hot(actions, num_classes=2).to(states)


class PrefixSensitiveEnvironment:
    def initial_state_from_noise(self, noise: SyntheticNoiseBundle) -> Tensor:
        return noise.initial_normal[:, :1]

    def step_from_noise(
        self,
        state: Tensor,
        action: Tensor,
        *,
        shared: Tensor,
        independent: Tensor,
        innovations: Tensor,
        difficulty_uniform: Tensor,
        contamination_uniform: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del independent, innovations, difficulty_uniform, contamination_uniform
        outcome = (state[:, 0] + 1.0 + shared)[:, None].expand(-1, 2)
        return state + 3.0 * action.to(state)[:, None], outcome


class PrefixOutcomeModel:
    def __call__(self, states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        scale = torch.where(actions == 0, 2.0, 1.0).to(states)
        return states.new_zeros((len(states), 2)), scale[:, None].expand(-1, 2)


def _noise(*, n: int = 3, horizon: int = 2, shared: Tensor | None = None) -> SyntheticNoiseBundle:
    if shared is None:
        shared = torch.linspace(-0.1, 0.1, n).expand(horizon, -1).clone()
    return SyntheticNoiseBundle(
        initial_normal=torch.zeros(n, 6),
        initial_difficulty_uniform=torch.zeros(n),
        action_uniform=torch.linspace(0.2, 0.8, n).expand(horizon, -1).clone(),
        shared_normal=shared,
        independent_normal=torch.zeros(horizon, n),
        innovation_normal=torch.zeros(horizon, n, 4),
        difficulty_uniform=torch.zeros(horizon, n),
        contamination_uniform=torch.zeros(horizon, n),
        seed=17,
    )


def test_profiled_evaluation_checks_all_nonmonotone_candidates_and_exact_endpoint() -> None:
    schedules = torch.tensor(
        [
            [0.5, 0.5],
            [1.0, 1.5],
            [1.5, 0.5],
        ]
    )
    metrics = evaluate_profiled_candidates_crn(
        NonMonotoneEnvironment(),
        RadiusActionPolicy(),
        ActionScaleOutcomeModel(),
        candidate_schedules=schedules,
        outcome_sd=torch.tensor([1.0, 2.0]),
        noise=_noise(),
        chunk_size=2,
    )

    assert torch.equal(metrics.coverage, torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]]))
    assert torch.allclose(
        metrics.normalized_width,
        torch.tensor([[0.75, 0.75], [1.5, 0.45], [0.45, 0.75]]),
    )

    selected = select_profiled_oracle_schedule(schedules, metrics, target=0.9)

    assert selected.selection_available is True
    assert selected.selected_indices == (2,)
    assert torch.equal(selected.radii, torch.tensor([1.5, 0.5]))
    assert torch.equal(selected.tuning_coverage, torch.tensor([1.0, 1.0]))
    assert torch.allclose(selected.tuning_width, torch.tensor([0.45, 0.75]))
    assert selected.failure_stage is None
    assert selected.selected_endpoint is True


def test_future_radius_never_changes_earlier_hits() -> None:
    schedules = torch.tensor([[0.5, 0.5], [0.5, 1.5]])

    metrics = evaluate_profiled_candidates_crn(
        NonMonotoneEnvironment(),
        RadiusActionPolicy(),
        ActionScaleOutcomeModel(),
        candidate_schedules=schedules,
        outcome_sd=torch.ones(2),
        noise=_noise(),
        chunk_size=1,
    )

    assert metrics.coverage[0, 0].item() == metrics.coverage[1, 0].item() == 1.0
    assert metrics.normalized_width[0, 0].item() == metrics.normalized_width[1, 0].item()


def test_profiled_selection_requires_joint_feasibility_and_breaks_ties_by_index() -> None:
    schedules = torch.tensor([[3.0, 3.0], [1.0, 2.0], [2.0, 1.0]])
    metrics = CandidateMetrics(
        coverage=torch.tensor([[0.95, 0.95], [0.90, 0.91], [0.92, 0.93]]),
        normalized_width=torch.tensor([[3.0, 3.0], [1.0, 2.0], [2.0, 1.0]]),
    )

    selected = select_profiled_oracle_schedule(schedules, metrics, target=0.9)

    assert selected.selected_indices == (1,)
    assert torch.equal(selected.radii, schedules[1])
    assert selected.selected_endpoint is False


def test_profiled_joint_infeasibility_has_no_stage_failure() -> None:
    metrics = CandidateMetrics(
        coverage=torch.tensor([[0.95, 0.80], [0.80, 0.95]]),
        normalized_width=torch.ones(2, 2),
    )

    selected = select_profiled_oracle_schedule(torch.ones(2, 2), metrics, target=0.9)

    assert selected.selection_available is False
    assert selected.radii is None
    assert selected.selected_indices == ()
    assert selected.failure_stage is None
    assert selected.selected_endpoint is False


def test_profiled_failure_reports_earliest_stage_without_any_coverage_pass() -> None:
    metrics = CandidateMetrics(
        coverage=torch.tensor([[0.95, 0.80, 0.70], [0.92, 0.85, 0.95]]),
        normalized_width=torch.ones(2, 3),
    )

    selected = select_profiled_oracle_schedule(torch.ones(2, 3), metrics, target=0.9)

    assert selected.selection_available is False
    assert selected.failure_stage == 1


def test_greedy_commits_chosen_prefix_and_minimizes_current_width() -> None:
    result = greedy_sequential_oracle_schedule(
        PrefixSensitiveEnvironment(),
        PrefixPolicy(),
        PrefixOutcomeModel(),
        stage_grids=torch.tensor([[1.0, 3.0], [1.0, 3.0]]),
        outcome_sd=torch.ones(2),
        noise=_noise(n=3, horizon=2, shared=torch.zeros(2, 3)),
        target=0.9,
        chunk_size=1,
    )

    assert result.selection_available is True
    assert result.selected_indices == (0, 0)
    assert torch.equal(result.radii, torch.tensor([1.0, 1.0]))
    assert torch.equal(result.tuning_coverage, torch.tensor([1.0, 1.0]))
    assert torch.equal(result.tuning_width, torch.tensor([4.0, 4.0]))
    assert result.failure_stage is None
    assert result.selected_endpoint is True


def test_greedy_releases_candidate_state_buffer_before_next_stage(monkeypatch) -> None:
    stage_buffers: list[weakref.ReferenceType[Tensor]] = []

    def tracked_allocation(state: Tensor, candidate_count: int) -> Tensor:
        if stage_buffers:
            assert stage_buffers[-1]() is None
        allocated = state.new_empty((candidate_count, *state.shape))
        stage_buffers.append(weakref.ref(allocated))
        return allocated

    monkeypatch.setattr(
        phase0_oracle,
        "_allocate_candidate_next_states",
        tracked_allocation,
        raising=False,
    )

    result = greedy_sequential_oracle_schedule(
        PrefixSensitiveEnvironment(),
        PrefixPolicy(),
        PrefixOutcomeModel(),
        stage_grids=torch.tensor([[1.0, 3.0], [1.0, 3.0]]),
        outcome_sd=torch.ones(2),
        noise=_noise(n=3, horizon=2, shared=torch.zeros(2, 3)),
        target=0.9,
        chunk_size=1,
    )

    assert result.selection_available is True
    assert len(stage_buffers) == 2
    assert all(reference() is None for reference in stage_buffers)


def test_greedy_tie_uses_lowest_original_candidate_index() -> None:
    result = greedy_sequential_oracle_schedule(
        PrefixSensitiveEnvironment(),
        PrefixPolicy(),
        PrefixOutcomeModel(),
        stage_grids=torch.tensor([[3.0, 1.0, 1.0]]),
        outcome_sd=torch.ones(2),
        noise=_noise(n=3, horizon=1, shared=torch.zeros(1, 3)),
        target=0.9,
        chunk_size=2,
    )

    assert result.selected_indices == (1,)
    assert torch.equal(result.radii, torch.tensor([1.0]))
    assert result.selected_endpoint is False


def test_greedy_reports_first_stage_without_a_feasible_candidate() -> None:
    result = greedy_sequential_oracle_schedule(
        PrefixSensitiveEnvironment(),
        PrefixPolicy(),
        PrefixOutcomeModel(),
        stage_grids=torch.tensor([[1.0, 3.0], [0.1, 0.4]]),
        outcome_sd=torch.ones(2),
        noise=_noise(n=3, horizon=2, shared=torch.zeros(2, 3)),
        target=0.9,
        chunk_size=2,
    )

    assert result.selection_available is False
    assert result.radii is None
    assert result.failure_stage == 1
    assert result.selected_endpoint is False


def test_profiled_metrics_are_invariant_to_candidate_order_and_chunk_size() -> None:
    values = torch.tensor([0.5, 1.0, 1.5])
    schedules = torch.stack(
        (values[torch.arange(101) % 3], values[(torch.arange(101) + 1) % 3]),
        dim=1,
    )
    permutation = torch.randperm(101, generator=torch.Generator().manual_seed(29))
    baseline = evaluate_profiled_candidates_crn(
        NonMonotoneEnvironment(),
        RadiusActionPolicy(),
        ActionScaleOutcomeModel(),
        candidate_schedules=schedules,
        outcome_sd=torch.tensor([1.0, 2.0]),
        noise=_noise(),
        chunk_size=101,
    )

    for chunk_size in (1, 16, 101):
        policy = RadiusActionPolicy()
        outcome_model = ActionScaleOutcomeModel()
        observed = evaluate_profiled_candidates_crn(
            NonMonotoneEnvironment(),
            policy,
            outcome_model,
            candidate_schedules=schedules[permutation],
            outcome_sd=torch.tensor([1.0, 2.0]),
            noise=_noise(),
            chunk_size=chunk_size,
        )
        original_order = torch.argsort(permutation)

        assert torch.equal(observed.coverage[original_order], baseline.coverage)
        assert torch.equal(
            observed.normalized_width[original_order],
            baseline.normalized_width,
        )
        assert policy.max_batch <= chunk_size * 3
        assert outcome_model.max_batch <= chunk_size * 3


def test_policy_all_action_prediction_has_chunk_patient_action_inner_bound() -> None:
    predictor = ActionScaleOutcomeModel()
    policy = PredictorMediatedRadiusPolicy(predictor)

    evaluate_profiled_candidates_crn(
        NonMonotoneEnvironment(),
        policy,
        predictor,
        candidate_schedules=torch.tensor([[0.5], [1.0], [1.5], [0.5], [1.0]]),
        outcome_sd=torch.ones(2),
        noise=_noise(n=3, horizon=1),
        chunk_size=2,
    )

    assert policy.max_batch == 2 * 3
    assert predictor.max_batch == 2 * 3 * predictor.n_actions
