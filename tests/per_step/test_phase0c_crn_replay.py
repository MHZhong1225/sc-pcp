from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch import Tensor

import scpcp.phase0c_joint_search as phase0c_joint_search
from scpcp.phase0_oracle import evaluate_profiled_candidates_crn
from scpcp.phase0c_joint_search import (
    CRNCoordinateEvaluator,
    SearchStart,
    build_schedule_cache,
    coordinate_candidate_schedules,
    evaluate_coordinate_candidates_crn,
)
from scpcp.simulator import SyntheticNoiseBundle


class DeterministicPolicy:
    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        assert q is not None
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


class DeterministicEnvironment:
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


class CountingEnvironment(DeterministicEnvironment):
    def __init__(self) -> None:
        self.initial_state_calls = 0

    def initial_state_from_noise(self, noise: SyntheticNoiseBundle) -> Tensor:
        self.initial_state_calls += 1
        return super().initial_state_from_noise(noise)


class DeterministicOutcomeModel:
    def __call__(self, states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        scale = torch.tensor([1.0, 1.0, 0.2], device=states.device)[actions]
        return states.new_zeros((len(states), 2)), scale[:, None].expand(-1, 2)


def _noise() -> SyntheticNoiseBundle:
    patient_count = 5
    horizon = 3
    return SyntheticNoiseBundle(
        initial_normal=torch.tensor(
            [[0.0], [0.2], [-0.1], [0.3], [-0.2]]
        ).expand(-1, 6).clone(),
        initial_difficulty_uniform=torch.zeros(patient_count),
        action_uniform=torch.tensor(
            [[0.1, 0.3, 0.5, 0.7, 0.9]]
        ).expand(horizon, -1).clone(),
        shared_normal=torch.tensor(
            [
                [-0.10, -0.05, 0.00, 0.05, 0.10],
                [0.10, 0.05, 0.00, -0.05, -0.10],
                [-0.08, 0.04, 0.00, 0.08, -0.04],
            ]
        ),
        independent_normal=torch.zeros(horizon, patient_count),
        innovation_normal=torch.tensor(
            [[[0.01, 0.0, 0.0, 0.0]], [[-0.02, 0.0, 0.0, 0.0]], [[0.03, 0.0, 0.0, 0.0]]]
        ).expand(-1, patient_count, -1).clone(),
        difficulty_uniform=torch.zeros(horizon, patient_count),
        contamination_uniform=torch.zeros(horizon, patient_count),
        seed=1703,
    )


INCUMBENT = torch.tensor([0.5, 1.0, 1.5])
STAGE_GRID = torch.tensor([0.4, 0.8, 1.2, 1.6, 2.0])
OUTCOME_SD = torch.tensor([1.0, 2.0])

LITERAL_COVERAGE = (
    torch.tensor(
        [
            [0.6, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.6, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    ),
    torch.tensor(
        [
            [1.0, 0.6, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.6, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    ),
    torch.tensor(
        [
            [1.0, 0.0, 0.6],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.6],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    ),
)

LITERAL_WIDTH = (
    torch.tensor(
        [
            [0.6, 1.5, 0.4500000476837158],
            [1.2000000476837158, 1.5, 0.4500000476837158],
            [1.8000001907348633, 1.5, 0.4500000476837158],
            [0.48000001907348633, 1.5, 0.4500000476837158],
            [0.6, 1.5, 0.4500000476837158],
        ]
    ),
    torch.tensor(
        [
            [0.75, 0.6, 0.4500000476837158],
            [0.75, 1.2000000476837158, 0.4500000476837158],
            [0.75, 1.8000001907348633, 0.4500000476837158],
            [0.75, 0.48000001907348633, 0.4500000476837158],
            [0.75, 0.6, 0.4500000476837158],
        ]
    ),
    torch.tensor(
        [
            [0.75, 1.5, 0.6],
            [0.75, 1.5, 1.2000000476837158],
            [0.75, 1.5, 1.8000001907348633],
            [0.75, 1.5, 0.48000001907348633],
            [0.75, 1.5, 0.6],
        ]
    ),
)


def _full_replay(stage: int, *, grid: Tensor = STAGE_GRID, chunk_size: int = 2):
    return evaluate_profiled_candidates_crn(
        DeterministicEnvironment(),
        DeterministicPolicy(),
        DeterministicOutcomeModel(),
        candidate_schedules=coordinate_candidate_schedules(
            INCUMBENT,
            stage=stage,
            stage_grid=grid,
        ),
        outcome_sd=OUTCOME_SD,
        noise=_noise(),
        chunk_size=chunk_size,
    )


def _cache(environment: object | None = None):
    return build_schedule_cache(
        environment or DeterministicEnvironment(),
        DeterministicPolicy(),
        DeterministicOutcomeModel(),
        schedule=INCUMBENT,
        outcome_sd=OUTCOME_SD,
        noise=_noise(),
    )


@pytest.mark.parametrize("stage", (0, 1, 2))
def test_old_full_replay_matches_frozen_literal_fixture(stage: int) -> None:
    full = _full_replay(stage)
    assert torch.equal(full.coverage, LITERAL_COVERAGE[stage])
    assert torch.equal(full.normalized_width, LITERAL_WIDTH[stage])


@pytest.mark.parametrize("stage", (0, 1, 2))
@pytest.mark.parametrize("chunk_size", (1, 2, 5))
def test_cached_suffix_is_bitwise_equal_to_full_replay(
    stage: int,
    chunk_size: int,
) -> None:
    full = _full_replay(stage, chunk_size=2)
    cached = evaluate_coordinate_candidates_crn(
        DeterministicEnvironment(),
        DeterministicPolicy(),
        DeterministicOutcomeModel(),
        cache=_cache(),
        incumbent_schedule=INCUMBENT,
        stage=stage,
        stage_grid=STAGE_GRID,
        outcome_sd=OUTCOME_SD,
        noise=_noise(),
        chunk_size=chunk_size,
    )
    assert torch.equal(cached.coverage, full.coverage)
    assert torch.equal(cached.normalized_width, full.normalized_width)


@pytest.mark.parametrize("chunk_size", (1, 2, 5))
def test_cached_suffix_is_invariant_to_candidate_permutation_and_chunking(
    chunk_size: int,
) -> None:
    permutation = torch.tensor([2, 4, 0, 3, 1])
    cached = evaluate_coordinate_candidates_crn(
        DeterministicEnvironment(),
        DeterministicPolicy(),
        DeterministicOutcomeModel(),
        cache=_cache(),
        incumbent_schedule=INCUMBENT,
        stage=1,
        stage_grid=STAGE_GRID[permutation],
        outcome_sd=OUTCOME_SD,
        noise=_noise(),
        chunk_size=chunk_size,
    )
    original_order = torch.argsort(permutation)
    assert torch.equal(cached.coverage[original_order], LITERAL_COVERAGE[1])
    assert torch.equal(cached.normalized_width[original_order], LITERAL_WIDTH[1])


@pytest.mark.parametrize("stage", (0, 1, 2))
def test_suffix_replay_copies_prefix_metrics_without_mutating_cache(stage: int) -> None:
    cache = _cache()
    states_before = tuple(state.clone() for state in cache.states_before)
    coverage = cache.coverage.clone()
    width = cache.normalized_width.clone()

    cached = evaluate_coordinate_candidates_crn(
        DeterministicEnvironment(),
        DeterministicPolicy(),
        DeterministicOutcomeModel(),
        cache=cache,
        incumbent_schedule=INCUMBENT,
        stage=stage,
        stage_grid=STAGE_GRID,
        outcome_sd=OUTCOME_SD,
        noise=_noise(),
        chunk_size=2,
    )

    assert len(cache.states_before) == len(INCUMBENT) + 1
    if stage:
        assert torch.equal(
            cached.coverage[:, :stage],
            coverage[:stage].expand(len(STAGE_GRID), -1),
        )
        assert torch.equal(
            cached.normalized_width[:, :stage],
            width[:stage].expand(len(STAGE_GRID), -1),
        )
    assert torch.equal(cache.coverage, coverage)
    assert torch.equal(cache.normalized_width, width)
    assert all(
        torch.equal(observed, expected)
        for observed, expected in zip(cache.states_before, states_before, strict=True)
    )


def _search_starts() -> tuple[SearchStart, ...]:
    metrics = evaluate_profiled_candidates_crn(
        DeterministicEnvironment(),
        DeterministicPolicy(),
        DeterministicOutcomeModel(),
        candidate_schedules=INCUMBENT.unsqueeze(0),
        outcome_sd=OUTCOME_SD,
        noise=_noise(),
        chunk_size=1,
    )
    return tuple(
        SearchStart(
            name=name,
            radii=INCUMBENT.clone(),
            stage_grid_indices=(None, None, None),
            coverage=metrics.coverage[0].clone(),
            normalized_width=metrics.normalized_width[0].clone(),
        )
        for name in ("profiled", "greedy", "upper_endpoint")
    )


def _coordinate_evaluator(environment: object) -> CRNCoordinateEvaluator:
    return CRNCoordinateEvaluator(
        environment,
        DeterministicPolicy(),
        DeterministicOutcomeModel(),
        starts=_search_starts(),
        outcome_sd=OUTCOME_SD,
        noise=_noise(),
        chunk_size=5,
    )


def test_adapter_rebuilds_only_changed_start_once_after_commit() -> None:
    environment = CountingEnvironment()
    evaluator = _coordinate_evaluator(environment)
    assert environment.initial_state_calls == 3

    evaluator("profiled", INCUMBENT, 0, STAGE_GRID)
    assert environment.initial_state_calls == 3

    committed = INCUMBENT.clone()
    committed[0] = STAGE_GRID[1]
    evaluator("profiled", committed, 1, STAGE_GRID)
    assert environment.initial_state_calls == 4

    evaluator("profiled", committed.clone(), 2, STAGE_GRID)
    assert environment.initial_state_calls == 4


def test_adapter_rejects_unknown_start_and_incompatible_reused_horizon() -> None:
    evaluator = _coordinate_evaluator(DeterministicEnvironment())

    with pytest.raises(ValueError, match="canonical.*profiled.*greedy.*upper_endpoint"):
        evaluator("unknown", INCUMBENT, 0, STAGE_GRID)
    with pytest.raises(ValueError, match="horizon"):
        evaluator("profiled", torch.tensor([0.5, 1.0]), 0, STAGE_GRID)


def test_adapter_rejects_noncanonical_constructor_starts() -> None:
    starts = list(_search_starts())
    starts[2] = SearchStart(
        name="upper",
        radii=starts[2].radii,
        stage_grid_indices=starts[2].stage_grid_indices,
        coverage=starts[2].coverage,
        normalized_width=starts[2].normalized_width,
    )
    with pytest.raises(ValueError, match="canonical.*profiled.*greedy.*upper_endpoint"):
        CRNCoordinateEvaluator(
            DeterministicEnvironment(),
            DeterministicPolicy(),
            DeterministicOutcomeModel(),
            starts=tuple(starts),
            outcome_sd=OUTCOME_SD,
            noise=_noise(),
            chunk_size=5,
        )


def test_candidate_state_buffer_is_released_before_next_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_buffers: list[weakref.ReferenceType[Tensor]] = []

    def tracked_allocation(state: Tensor, candidate_count: int) -> Tensor:
        gc.collect()
        assert all(reference() is None for reference in candidate_buffers)
        allocated = state.new_empty((candidate_count, *state.shape))
        candidate_buffers.append(weakref.ref(allocated))
        return allocated

    monkeypatch.setattr(
        phase0c_joint_search,
        "_allocate_candidate_states",
        tracked_allocation,
        raising=False,
    )
    evaluator = _coordinate_evaluator(DeterministicEnvironment())

    evaluator("profiled", INCUMBENT, 0, STAGE_GRID)
    evaluator("profiled", INCUMBENT, 1, STAGE_GRID)

    assert len(candidate_buffers) == 2
    gc.collect()
    assert all(reference() is None for reference in candidate_buffers)
