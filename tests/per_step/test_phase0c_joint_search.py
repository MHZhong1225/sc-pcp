from __future__ import annotations

from dataclasses import replace
from itertools import product

import pytest
import torch
from torch import Tensor

from scpcp.phase0_oracle import CandidateMetrics
from scpcp.phase0c_joint_search import (
    SearchState,
    SearchStart,
    choose_coordinate_candidate,
    coordinate_candidate_schedules,
    cyclic_joint_coordinate_search,
    resume_cyclic_joint_coordinate_search,
)


class TableEvaluator:
    def __init__(
        self,
        rows: dict[
            tuple[str, int, int, tuple[float, ...]],
            tuple[tuple[float, ...], tuple[float, ...]],
        ],
    ) -> None:
        self.rows = rows
        self.calls: list[tuple[str, int, tuple[float, ...]]] = []

    def __call__(
        self,
        start_name: str,
        incumbent: Tensor,
        stage: int,
        grid: Tensor,
    ) -> CandidateMetrics:
        incumbent_key = tuple(float(value) for value in incumbent.tolist())
        self.calls.append((start_name, stage, incumbent_key))
        coverage = []
        widths = []
        for grid_index in range(len(grid)):
            row_coverage, row_width = self.rows[
                (start_name, stage, grid_index, incumbent_key)
            ]
            coverage.append(row_coverage)
            widths.append(row_width)
        return CandidateMetrics(
            coverage=torch.tensor(coverage),
            normalized_width=torch.tensor(widths),
        )


class LandscapeEvaluator:
    def __init__(self, widths: Tensor, *, clock: ManualClock | None = None) -> None:
        self.widths = widths
        self.clock = clock

    def __call__(
        self,
        start_name: str,
        incumbent: Tensor,
        stage: int,
        grid: Tensor,
    ) -> CandidateMetrics:
        del start_name
        if self.clock is not None:
            self.clock.now += 1.0
        schedules = incumbent.repeat(len(grid), 1)
        schedules[:, stage] = grid
        schedule_indices = schedules.long()
        candidate_widths = torch.tensor(
            [
                float(self.widths[tuple(indices.tolist())].item())
                for indices in schedule_indices
            ]
        )
        if self.clock is not None:
            candidate_widths -= self.clock.now
        horizon = incumbent.shape[0]
        return CandidateMetrics(
            coverage=torch.full((len(grid), horizon), 0.95),
            normalized_width=candidate_widths[:, None].repeat(1, horizon),
        )


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _start(
    name: str,
    radii: tuple[float, ...],
    *,
    coverage: tuple[float, ...] | None = None,
    width: tuple[float, ...] | None = None,
) -> SearchStart:
    horizon = len(radii)
    return SearchStart(
        name=name,
        radii=torch.tensor(radii),
        stage_grid_indices=(None,) * horizon,
        coverage=torch.tensor(coverage or (0.95,) * horizon),
        normalized_width=torch.tensor(width or radii),
    )


def _canonical_starts(
    profiled: SearchStart,
    greedy: SearchStart | None = None,
    upper_endpoint: SearchStart | None = None,
) -> tuple[SearchStart, ...]:
    inactive_radii = tuple(0.0 for _ in profiled.radii)
    if greedy is None:
        greedy = _start(
            "greedy",
            inactive_radii,
            coverage=(0.0,) * len(inactive_radii),
        )
    if upper_endpoint is None:
        upper_endpoint = _start(
            "upper_endpoint",
            inactive_radii,
            coverage=(0.0,) * len(inactive_radii),
        )
    return profiled, greedy, upper_endpoint


def test_coordinate_candidates_replace_only_requested_stage() -> None:
    incumbent = torch.tensor([1.0, 2.0, 3.0])
    grid = torch.tensor([0.5, 1.5, 2.5])
    got = coordinate_candidate_schedules(incumbent, stage=1, stage_grid=grid)
    want = torch.tensor(
        [[1.0, 0.5, 3.0], [1.0, 1.5, 3.0], [1.0, 2.5, 3.0]]
    )
    assert torch.equal(got, want)


def test_coordinate_choice_requires_all_stage_coverage_and_uses_full_mean_width() -> None:
    metrics = CandidateMetrics(
        coverage=torch.tensor([[0.91, 0.89], [0.90, 0.90], [0.91, 0.91]]),
        normalized_width=torch.tensor(
            [[0.10, 0.10], [1.00, 3.00], [2.00, 1.00]]
        ),
    )
    choice = choose_coordinate_candidate(metrics, target=0.90)
    assert choice.grid_index == 2
    assert choice.micro_width == 1.5


def test_duplicate_minimum_uses_lowest_original_grid_index() -> None:
    metrics = CandidateMetrics(
        coverage=torch.full((3, 2), 0.91),
        normalized_width=torch.tensor(
            [[2.0, 2.0], [1.0, 1.0], [1.0, 1.0]]
        ),
    )
    assert choose_coordinate_candidate(metrics, target=0.90).grid_index == 1


def test_equal_width_proposal_is_not_committed() -> None:
    start = _start("profiled", (1.0,), width=(1.0,))
    evaluator = TableEvaluator(
        {
            ("profiled", 0, 0, (1.0,)): ((0.95,), (1.0,)),
            ("profiled", 0, 1, (1.0,)): ((0.95,), (1.0,)),
        }
    )

    outcome = cyclic_joint_coordinate_search(
        _canonical_starts(start),
        torch.tensor([[1.0, 2.0]]),
        evaluator,
        sweep_pair_checkpoints=(2,),
    )

    checkpoint = outcome.checkpoints[2]
    assert checkpoint.committed_updates == 0
    assert checkpoint.best.stage_grid_indices == (None,)
    assert all(not step.committed for step in checkpoint.trace)


def test_coordinate_without_feasible_proposals_preserves_feasible_incumbent() -> None:
    start = _start("profiled", (1.0,), width=(1.0,))
    evaluator = TableEvaluator(
        {
            ("profiled", 0, 0, (1.0,)): ((0.89,), (0.5,)),
            ("profiled", 0, 1, (1.0,)): ((0.80,), (0.4,)),
        }
    )

    outcome = cyclic_joint_coordinate_search(
        _canonical_starts(start),
        torch.tensor([[0.5, 0.75]]),
        evaluator,
        sweep_pair_checkpoints=(2,),
    )

    checkpoint = outcome.checkpoints[2]
    assert torch.equal(checkpoint.best.radii, torch.tensor([1.0]))
    assert checkpoint.trace[0].feasible_count == 0
    assert checkpoint.trace[0].proposed_grid_index is None


def test_all_infeasible_starts_return_no_feasible_start() -> None:
    starts = (
        _start("profiled", (1.0,), coverage=(0.89,)),
        _start("greedy", (2.0,), coverage=(0.80,)),
        _start("upper_endpoint", (2.0,), coverage=(0.70,)),
    )

    def evaluator(
        start_name: str,
        incumbent: Tensor,
        stage: int,
        grid: Tensor,
    ) -> CandidateMetrics:
        raise AssertionError("infeasible starts must not be evaluated")

    outcome = cyclic_joint_coordinate_search(
        starts,
        torch.tensor([[1.0, 2.0]]),
        evaluator,
        sweep_pair_checkpoints=(2,),
    )

    assert outcome.status == "NO_FEASIBLE_START"
    assert outcome.checkpoints == {}


def test_reversed_start_tie_uses_canonical_profiled_priority() -> None:
    profiled = _start("profiled", (1.0,), width=(1.0,))
    greedy = _start("greedy", (2.0,), width=(1.0,))
    upper_endpoint = _start("upper_endpoint", (3.0,), width=(1.0,))

    def no_feasible_proposals(
        start_name: str,
        incumbent: Tensor,
        stage: int,
        grid: Tensor,
    ) -> CandidateMetrics:
        return CandidateMetrics(
            coverage=torch.full((len(grid), 1), 0.80),
            normalized_width=torch.ones(len(grid), 1),
        )

    outcome = cyclic_joint_coordinate_search(
        (upper_endpoint, greedy, profiled),
        torch.tensor([[1.0, 2.0]]),
        no_feasible_proposals,
        sweep_pair_checkpoints=(2,),
    )

    assert outcome.status == "SELECTED"
    assert outcome.checkpoints[2].best.start_name == "profiled"
    assert tuple(
        state.start_name for state in outcome.checkpoints[2].per_start
    ) == ("profiled", "greedy", "upper_endpoint")


@pytest.mark.parametrize(
    "starts",
    [
        (
            _start("profiled", (1.0,)),
            _start("greedy", (1.0,)),
            _start("upper", (1.0,)),
        ),
        (
            _start("profiled", (1.0,)),
            _start("profiled", (1.0,)),
            _start("upper_endpoint", (1.0,)),
        ),
    ],
)
def test_initial_search_rejects_noncanonical_or_duplicate_start_names(
    starts: tuple[SearchStart, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="canonical.*profiled.*greedy.*upper_endpoint",
    ):
        cyclic_joint_coordinate_search(
            starts,
            torch.tensor([[1.0, 2.0]]),
            lambda start_name, incumbent, stage, grid: CandidateMetrics(
                coverage=torch.full((len(grid), 1), 0.95),
                normalized_width=torch.ones(len(grid), 1),
            ),
            sweep_pair_checkpoints=(2,),
        )


def test_trace_records_start_pair_direction_and_stage_order() -> None:
    start = _start("profiled", (0.0, 0.0), width=(1.0, 1.0))

    def no_feasible_proposals(
        start_name: str,
        incumbent: Tensor,
        stage: int,
        grid: Tensor,
    ) -> CandidateMetrics:
        return CandidateMetrics(
            coverage=torch.full((len(grid), 2), 0.80),
            normalized_width=torch.ones(len(grid), 2),
        )

    outcome = cyclic_joint_coordinate_search(
        _canonical_starts(start),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        no_feasible_proposals,
        sweep_pair_checkpoints=(2,),
    )

    assert [
        (step.start_name, step.sweep_pair, step.direction, step.stage)
        for step in outcome.checkpoints[2].trace[:4]
    ] == [
        ("profiled", 1, "forward", 0),
        ("profiled", 1, "forward", 1),
        ("profiled", 1, "reverse", 1),
        ("profiled", 1, "reverse", 0),
    ]


def test_reduced_landscape_matches_exhaustive_best_and_improves_all_starts() -> None:
    widths = torch.tensor(
        [
            [[6.0, 5.0, 4.0], [5.0, 4.0, 3.0], [4.0, 3.0, 2.0]],
            [[5.0, 4.0, 3.0], [4.0, 3.0, 2.0], [3.0, 2.0, 1.0]],
            [[4.0, 3.0, 2.0], [3.0, 2.0, 1.0], [2.0, 1.0, 0.5]],
        ]
    )
    starts = (
        _start("profiled", (0.0, 0.0, 0.0), width=(6.0, 6.0, 6.0)),
        _start("greedy", (1.0, 0.0, 1.0), width=(4.0, 4.0, 4.0)),
        _start(
            "upper_endpoint", (0.0, 2.0, 0.0), width=(4.0, 4.0, 4.0)
        ),
    )
    stage_grids = torch.arange(3.0).repeat(3, 1)

    outcome = cyclic_joint_coordinate_search(
        starts,
        stage_grids,
        LandscapeEvaluator(widths),
        sweep_pair_checkpoints=(2, 4),
    )

    all_27_schedules = tuple(product(range(3), repeat=3))
    exhaustive_width = min(
        float(widths[schedule].item()) for schedule in all_27_schedules
    )
    best = outcome.checkpoints[4].best
    assert bool((best.coverage >= 0.90).all().item())
    assert float(best.normalized_width.mean().item()) == pytest.approx(
        exhaustive_width
    )
    assert float(best.normalized_width.mean().item()) <= min(
        float(start.normalized_width.mean().item()) for start in starts
    )


def test_coordinate_trap_is_reported_only_as_best_found_not_a_global_optimum() -> None:
    widths = torch.tensor([[4.0, 5.0], [5.0, 1.0]])
    start = _start("profiled", (0.0, 0.0), width=(4.0, 4.0))

    outcome = cyclic_joint_coordinate_search(
        _canonical_starts(start),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        LandscapeEvaluator(widths),
        sweep_pair_checkpoints=(2,),
    )

    best_found_width = float(
        outcome.checkpoints[2].best.normalized_width.mean().item()
    )
    assert best_found_width == 4.0
    assert best_found_width > float(widths.min().item())


def test_four_pair_run_reuses_the_exact_two_pair_checkpoint() -> None:
    widths = torch.tensor([[4.0, 3.0], [2.0, 1.0]])
    start = _start("profiled", (0.0, 0.0), width=(4.0, 4.0))
    stage_grids = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    maximum_two = cyclic_joint_coordinate_search(
        _canonical_starts(start),
        stage_grids,
        LandscapeEvaluator(widths),
        sweep_pair_checkpoints=(2,),
    ).checkpoints[2]
    maximum_four = cyclic_joint_coordinate_search(
        _canonical_starts(start),
        stage_grids,
        LandscapeEvaluator(widths),
        sweep_pair_checkpoints=(2, 4),
    ).checkpoints[2]

    assert maximum_four.requested_sweep_pairs == maximum_two.requested_sweep_pairs
    assert maximum_four.executed_sweep_pairs == maximum_two.executed_sweep_pairs
    assert maximum_four.schedule_evaluations == maximum_two.schedule_evaluations
    assert maximum_four.committed_updates == maximum_two.committed_updates
    assert maximum_four.trace == maximum_two.trace
    assert torch.equal(maximum_four.best.radii, maximum_two.best.radii)
    assert torch.equal(maximum_four.best.coverage, maximum_two.best.coverage)
    assert torch.equal(
        maximum_four.best.normalized_width,
        maximum_two.best.normalized_width,
    )


def test_deadline_rolls_back_incomplete_pair_and_omits_its_checkpoint() -> None:
    clock = ManualClock()
    widths = torch.tensor([[4.0, 3.0], [2.0, 1.0]])
    start = _start("profiled", (0.0, 0.0), width=(4.0, 4.0))

    outcome = cyclic_joint_coordinate_search(
        _canonical_starts(start),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        LandscapeEvaluator(widths, clock=clock),
        sweep_pair_checkpoints=(2, 4),
        max_wall_seconds=8.5,
        clock=clock,
    )

    assert outcome.status == "WALL_TIME_CAP"
    assert tuple(outcome.checkpoints) == (2,)
    checkpoint = outcome.checkpoints[2]
    assert checkpoint.executed_sweep_pairs == 2
    assert checkpoint.schedule_evaluations == 16
    assert len(checkpoint.trace) == 8
    assert all(step.sweep_pair <= 2 for step in checkpoint.trace)


def test_deadline_during_last_coordinate_keeps_and_publishes_complete_pair() -> None:
    clock = ManualClock()
    widths = torch.tensor([[4.0, 3.0], [2.0, 1.0]])
    start = _start("profiled", (0.0, 0.0), width=(4.0, 4.0))

    outcome = cyclic_joint_coordinate_search(
        _canonical_starts(start),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        LandscapeEvaluator(widths, clock=clock),
        sweep_pair_checkpoints=(2, 4),
        max_wall_seconds=7.5,
        clock=clock,
    )

    assert outcome.status == "WALL_TIME_CAP"
    assert tuple(outcome.checkpoints) == (2,)
    checkpoint = outcome.checkpoints[2]
    assert checkpoint.executed_sweep_pairs == 2
    assert checkpoint.schedule_evaluations == 16
    assert len(checkpoint.trace) == 8


def test_deadline_after_converged_pair_materializes_later_checkpoints() -> None:
    clock = ManualClock()
    evaluator_calls = 0
    start = _start("profiled", (1.0,), width=(1.0,))

    def equal_width_proposals(
        start_name: str,
        incumbent: Tensor,
        stage: int,
        grid: Tensor,
    ) -> CandidateMetrics:
        nonlocal evaluator_calls
        evaluator_calls += 1
        clock.now += 1.0
        return CandidateMetrics(
            coverage=torch.full((len(grid), 1), 0.95),
            normalized_width=torch.ones(len(grid), 1),
        )

    outcome = cyclic_joint_coordinate_search(
        _canonical_starts(start),
        torch.tensor([[1.0, 2.0]]),
        equal_width_proposals,
        sweep_pair_checkpoints=(2, 4),
        max_wall_seconds=1.5,
        clock=clock,
    )

    assert outcome.status == "WALL_TIME_CAP"
    assert tuple(outcome.checkpoints) == (2, 4)
    assert evaluator_calls == 2
    for requested_pair, checkpoint in outcome.checkpoints.items():
        assert checkpoint.requested_sweep_pairs == requested_pair
        assert checkpoint.executed_sweep_pairs == 1
        assert checkpoint.schedule_evaluations == 4
        assert len(checkpoint.trace) == 2
        assert checkpoint.best.completed_sweep_pairs == requested_pair
        assert checkpoint.best.converged_at_pair == 1


def _pair_four_state(name: str, completed_sweep_pairs: int = 4) -> SearchState:
    return SearchState(
        start_name=name,
        radii=torch.tensor([1.0]),
        stage_grid_indices=(0,),
        coverage=torch.tensor([0.95]),
        normalized_width=torch.tensor([1.0]),
        completed_sweep_pairs=completed_sweep_pairs,
        converged_at_pair=None,
    )


def test_resume_continues_at_pair_five_and_returns_only_pair_eight() -> None:
    def no_feasible_proposals(
        start_name: str,
        incumbent: Tensor,
        stage: int,
        grid: Tensor,
    ) -> CandidateMetrics:
        return CandidateMetrics(
            coverage=torch.full((len(grid), 1), 0.80),
            normalized_width=torch.ones(len(grid), 1),
        )

    outcome = resume_cyclic_joint_coordinate_search(
        (
            _pair_four_state("upper_endpoint"),
            _pair_four_state("greedy"),
            _pair_four_state("profiled"),
        ),
        torch.tensor([[1.0, 2.0]]),
        no_feasible_proposals,
    )

    assert tuple(outcome.checkpoints) == (8,)
    checkpoint = outcome.checkpoints[8]
    assert checkpoint.executed_sweep_pairs == 5
    assert checkpoint.best.completed_sweep_pairs == 8
    assert {step.sweep_pair for step in checkpoint.trace} == {5}
    assert tuple(state.start_name for state in checkpoint.per_start) == (
        "profiled",
        "greedy",
        "upper_endpoint",
    )


@pytest.mark.parametrize(
    "states",
    [
        (
            _pair_four_state("profiled", 3),
            _pair_four_state("greedy", 3),
            _pair_four_state("upper_endpoint", 3),
        ),
        (
            _pair_four_state("profiled", 4),
            _pair_four_state("greedy", 3),
            _pair_four_state("upper_endpoint", 4),
        ),
    ],
)
def test_resume_rejects_non_four_and_mixed_completed_pair_counts(
    states: tuple[SearchState, ...],
) -> None:
    with pytest.raises(ValueError, match="completed_sweep_pairs.*4"):
        resume_cyclic_joint_coordinate_search(
            states,
            torch.tensor([[1.0, 2.0]]),
            lambda start_name, incumbent, stage, grid: CandidateMetrics(
                coverage=torch.full((len(grid), 1), 0.95),
                normalized_width=torch.ones(len(grid), 1),
            ),
        )


def _canonical_pair_four_states() -> tuple[SearchState, ...]:
    return (
        _pair_four_state("profiled"),
        _pair_four_state("greedy"),
        _pair_four_state("upper_endpoint"),
    )


def _unused_evaluator(
    start_name: str,
    incumbent: Tensor,
    stage: int,
    grid: Tensor,
) -> CandidateMetrics:
    return CandidateMetrics(
        coverage=torch.full((len(grid), 1), 0.95),
        normalized_width=torch.ones(len(grid), 1),
    )


def test_resume_rejects_noncanonical_or_duplicate_start_names() -> None:
    states = (
        _pair_four_state("profiled"),
        _pair_four_state("profiled"),
        _pair_four_state("upper_endpoint"),
    )

    with pytest.raises(
        ValueError,
        match="canonical.*profiled.*greedy.*upper_endpoint",
    ):
        resume_cyclic_joint_coordinate_search(
            states,
            torch.tensor([[1.0, 2.0]]),
            _unused_evaluator,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("stage_grid_indices", (0, 0)),
        ("coverage", torch.tensor([0.95, 0.95])),
        ("normalized_width", torch.tensor([1.0, 1.0])),
    ],
)
def test_resume_rejects_state_dimension_mismatch(
    field: str,
    value: tuple[int, ...] | Tensor,
) -> None:
    profiled, greedy, upper_endpoint = _canonical_pair_four_states()
    states = (replace(profiled, **{field: value}), greedy, upper_endpoint)

    with pytest.raises(ValueError, match="state dimensions.*stage-grid horizon"):
        resume_cyclic_joint_coordinate_search(
            states,
            torch.tensor([[1.0, 2.0]]),
            _unused_evaluator,
        )


def test_resume_rejects_infeasible_state() -> None:
    profiled, greedy, upper_endpoint = _canonical_pair_four_states()
    states = (
        profiled,
        replace(greedy, coverage=torch.tensor([0.89])),
        upper_endpoint,
    )

    with pytest.raises(ValueError, match="resume states must be jointly feasible"):
        resume_cyclic_joint_coordinate_search(
            states,
            torch.tensor([[1.0, 2.0]]),
            _unused_evaluator,
        )


@pytest.mark.parametrize(
    "converged_pairs",
    [(2, None, None), (2, 3, 2), (5, 5, 5)],
)
def test_resume_rejects_inconsistent_or_post_pair_four_convergence(
    converged_pairs: tuple[int | None, int | None, int | None],
) -> None:
    states = tuple(
        replace(state, converged_at_pair=converged_pair)
        for state, converged_pair in zip(
            _canonical_pair_four_states(), converged_pairs, strict=True
        )
    )

    with pytest.raises(ValueError, match="convergence metadata.*pair 4"):
        resume_cyclic_joint_coordinate_search(
            states,
            torch.tensor([[1.0, 2.0]]),
            _unused_evaluator,
        )
