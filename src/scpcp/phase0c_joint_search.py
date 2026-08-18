"""Deterministic multi-start coordinate search for Phase 0C."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeVar

import torch
from torch import Tensor

from scpcp.phase0_oracle import CandidateMetrics, _evaluate_stage
from scpcp.simulator import SyntheticNoiseBundle


_CANONICAL_START_NAMES = ("profiled", "greedy", "upper_endpoint")
_NamedSearchObject = TypeVar("_NamedSearchObject")


@dataclass(frozen=True)
class SearchStart:
    name: str
    radii: Tensor
    stage_grid_indices: tuple[int | None, ...]
    coverage: Tensor
    normalized_width: Tensor


@dataclass(frozen=True)
class ScheduleCache:
    states_before: tuple[Tensor, ...]
    coverage: Tensor
    normalized_width: Tensor


@dataclass(frozen=True)
class CoordinateStep:
    start_name: str
    sweep_pair: int
    direction: str
    stage: int
    feasible_count: int
    proposed_grid_index: int | None
    before_micro_width: float
    proposed_micro_width: float | None
    committed: bool
    after_micro_width: float


@dataclass(frozen=True)
class SearchState:
    start_name: str
    radii: Tensor
    stage_grid_indices: tuple[int | None, ...]
    coverage: Tensor
    normalized_width: Tensor
    completed_sweep_pairs: int
    converged_at_pair: int | None


@dataclass(frozen=True)
class JointSearchCheckpoint:
    requested_sweep_pairs: int
    executed_sweep_pairs: int
    best: SearchState
    per_start: tuple[SearchState, ...]
    trace: tuple[CoordinateStep, ...]
    schedule_evaluations: int
    committed_updates: int


@dataclass(frozen=True)
class JointSearchOutcome:
    status: str
    checkpoints: dict[int, JointSearchCheckpoint]
    elapsed_seconds: float


@dataclass(frozen=True)
class _CoordinateChoice:
    grid_index: int | None
    micro_width: float | None
    feasible_count: int


def coordinate_candidate_schedules(
    incumbent: Tensor,
    *,
    stage: int,
    stage_grid: Tensor,
) -> Tensor:
    """Return complete schedules formed by replacing one coordinate."""

    schedules = incumbent.unsqueeze(0).expand(len(stage_grid), -1).clone()
    schedules[:, stage] = stage_grid.to(incumbent)
    return schedules


@torch.no_grad()
def build_schedule_cache(
    environment: object,
    policy: object,
    outcome_model: object,
    *,
    schedule: Tensor,
    outcome_sd: Tensor,
    noise: SyntheticNoiseBundle,
) -> ScheduleCache:
    """Replay one schedule and retain each committed stage boundary."""

    horizon = noise.action_uniform.shape[0]
    if schedule.ndim != 1 or len(schedule) != horizon:
        raise ValueError(f"schedule must have shape ({horizon},)")

    state = environment.initial_state_from_noise(noise)
    schedule = schedule.to(state)
    states_before = [state.clone()]
    coverage = []
    normalized_width = []
    for stage in range(horizon):
        next_states, stage_coverage, stage_width = _evaluate_stage(
            environment,
            policy,
            outcome_model,
            states=state.unsqueeze(0),
            radii=schedule[stage : stage + 1],
            outcome_sd=outcome_sd,
            noise=noise,
            stage=stage,
        )
        state = next_states[0]
        states_before.append(state.clone())
        coverage.append(stage_coverage[0])
        normalized_width.append(stage_width[0])

    return ScheduleCache(
        states_before=tuple(states_before),
        coverage=torch.stack(coverage),
        normalized_width=torch.stack(normalized_width),
    )


def _allocate_candidate_states(state: Tensor, candidate_count: int) -> Tensor:
    return state.new_empty((candidate_count, *state.shape))


@torch.no_grad()
def evaluate_coordinate_candidates_crn(
    environment: object,
    policy: object,
    outcome_model: object,
    *,
    cache: ScheduleCache,
    incumbent_schedule: Tensor,
    stage: int,
    stage_grid: Tensor,
    outcome_sd: Tensor,
    noise: SyntheticNoiseBundle,
    chunk_size: int,
) -> CandidateMetrics:
    """Evaluate one coordinate from its cached incumbent boundary state."""

    horizon = len(incumbent_schedule)
    if (
        incumbent_schedule.ndim != 1
        or len(cache.states_before) != horizon + 1
        or len(cache.coverage) != horizon
        or len(cache.normalized_width) != horizon
        or noise.action_uniform.shape[0] != horizon
    ):
        raise ValueError("cache and incumbent schedule horizons must match")
    if not 0 <= stage < horizon:
        raise ValueError("stage must lie within the schedule horizon")

    candidate_count = len(stage_grid)
    coverage = cache.coverage.unsqueeze(0).expand(candidate_count, -1).clone()
    normalized_width = (
        cache.normalized_width.unsqueeze(0).expand(candidate_count, -1).clone()
    )
    incumbent_schedule = incumbent_schedule.to(cache.states_before[stage])
    stage_grid = stage_grid.to(incumbent_schedule)

    for start in range(0, candidate_count, chunk_size):
        candidate_radii = stage_grid[start : start + chunk_size]
        states = _allocate_candidate_states(
            cache.states_before[stage],
            len(candidate_radii),
        )
        states.copy_(cache.states_before[stage].unsqueeze(0))
        for suffix_stage in range(stage, horizon):
            radii = (
                candidate_radii
                if suffix_stage == stage
                else incumbent_schedule[suffix_stage].expand(len(candidate_radii))
            )
            states, stage_coverage, stage_width = _evaluate_stage(
                environment,
                policy,
                outcome_model,
                states=states,
                radii=radii,
                outcome_sd=outcome_sd,
                noise=noise,
                stage=suffix_stage,
            )
            coverage[start : start + len(candidate_radii), suffix_stage] = (
                stage_coverage
            )
            normalized_width[
                start : start + len(candidate_radii), suffix_stage
            ] = stage_width

    return CandidateMetrics(
        coverage=coverage,
        normalized_width=normalized_width,
    )


class CRNCoordinateEvaluator:
    """Adapt cached CRN suffix replay to the joint-search evaluator contract."""

    def __init__(
        self,
        environment: object,
        policy: object,
        outcome_model: object,
        *,
        starts: tuple[SearchStart, ...],
        outcome_sd: Tensor,
        noise: SyntheticNoiseBundle,
        chunk_size: int,
    ) -> None:
        starts = _canonical_order(starts, tuple(start.name for start in starts))
        self._environment = environment
        self._policy = policy
        self._outcome_model = outcome_model
        self._outcome_sd = outcome_sd
        self._noise = noise
        self._chunk_size = chunk_size
        self._schedules = {start.name: start.radii.clone() for start in starts}
        self._caches = {
            start.name: self._build_cache(start.radii) for start in starts
        }

    def _build_cache(self, schedule: Tensor) -> ScheduleCache:
        return build_schedule_cache(
            self._environment,
            self._policy,
            self._outcome_model,
            schedule=schedule,
            outcome_sd=self._outcome_sd,
            noise=self._noise,
        )

    def __call__(
        self,
        start_name: str,
        incumbent: Tensor,
        stage: int,
        grid: Tensor,
    ) -> CandidateMetrics:
        if start_name not in self._caches:
            raise ValueError(
                "canonical starts must be exactly: "
                "profiled, greedy, upper_endpoint"
            )
        horizon = len(self._schedules[start_name])
        if incumbent.ndim != 1 or len(incumbent) != horizon:
            raise ValueError("reused start name has an incompatible horizon")
        if not 0 <= stage < horizon:
            raise ValueError("stage must lie within the schedule horizon")

        if not torch.equal(incumbent, self._schedules[start_name]):
            self._caches[start_name] = self._build_cache(incumbent)
            self._schedules[start_name] = incumbent.clone()

        return evaluate_coordinate_candidates_crn(
            self._environment,
            self._policy,
            self._outcome_model,
            cache=self._caches[start_name],
            incumbent_schedule=incumbent,
            stage=stage,
            stage_grid=grid,
            outcome_sd=self._outcome_sd,
            noise=self._noise,
            chunk_size=self._chunk_size,
        )


def choose_coordinate_candidate(
    metrics: CandidateMetrics,
    *,
    target: float,
) -> _CoordinateChoice:
    """Select the feasible minimum-width candidate, preserving grid tie order."""

    feasible = (metrics.coverage >= target).all(dim=1)
    feasible_indices = torch.nonzero(feasible, as_tuple=False).flatten()
    if len(feasible_indices) == 0:
        return _CoordinateChoice(None, None, 0)

    micro_widths = metrics.normalized_width.mean(dim=1)
    feasible_widths = micro_widths[feasible_indices]
    feasible_offset = int(torch.argmin(feasible_widths).item())
    grid_index = int(feasible_indices[feasible_offset].item())
    return _CoordinateChoice(
        grid_index=grid_index,
        micro_width=float(micro_widths[grid_index].item()),
        feasible_count=len(feasible_indices),
    )


def _state_from_start(start: SearchStart) -> SearchState:
    return SearchState(
        start_name=start.name,
        radii=start.radii.clone(),
        stage_grid_indices=start.stage_grid_indices,
        coverage=start.coverage.clone(),
        normalized_width=start.normalized_width.clone(),
        completed_sweep_pairs=0,
        converged_at_pair=None,
    )


def _canonical_order(
    values: tuple[_NamedSearchObject, ...],
    names: tuple[str, ...],
) -> tuple[_NamedSearchObject, ...]:
    if len(values) != len(_CANONICAL_START_NAMES) or set(names) != set(
        _CANONICAL_START_NAMES
    ):
        raise ValueError(
            "canonical starts must be exactly: profiled, greedy, upper_endpoint"
        )
    by_name = dict(zip(names, values, strict=True))
    return tuple(by_name[name] for name in _CANONICAL_START_NAMES)


def _micro_width(state: SearchState) -> float:
    return float(state.normalized_width.mean().item())


def _best_state(states: tuple[SearchState, ...]) -> SearchState:
    return min(states, key=_micro_width)


def _checkpoint(
    requested_pair: int,
    executed_pair: int,
    states: tuple[SearchState, ...],
    trace: list[CoordinateStep],
    schedule_evaluations: int,
    committed_updates: int,
) -> JointSearchCheckpoint:
    checkpoint_states = tuple(
        replace(state, completed_sweep_pairs=requested_pair) for state in states
    )
    return JointSearchCheckpoint(
        requested_sweep_pairs=requested_pair,
        executed_sweep_pairs=executed_pair,
        best=_best_state(checkpoint_states),
        per_start=checkpoint_states,
        trace=tuple(trace),
        schedule_evaluations=schedule_evaluations,
        committed_updates=committed_updates,
    )


def _materialize_converged_checkpoints(
    checkpoints: dict[int, JointSearchCheckpoint],
    requested_pairs: tuple[int, ...],
    executed_pair: int,
    states: tuple[SearchState, ...],
    trace: list[CoordinateStep],
    schedule_evaluations: int,
    committed_updates: int,
) -> None:
    for requested_pair in requested_pairs:
        if requested_pair < executed_pair:
            continue
        checkpoints[requested_pair] = _checkpoint(
            requested_pair,
            executed_pair,
            states,
            trace,
            schedule_evaluations,
            committed_updates,
        )


def _validate_search_inputs(
    states: tuple[SearchState, ...],
    stage_grids: Tensor,
    requested_pairs: tuple[int, ...],
    max_wall_seconds: float | None,
) -> None:
    if not requested_pairs or requested_pairs[0] <= 0:
        raise ValueError("sweep-pair checkpoints must be positive")
    if stage_grids.ndim != 2:
        raise ValueError("stage_grids must have shape (horizon, grid_size)")
    if max_wall_seconds is not None and max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive")
    horizon = stage_grids.shape[0]
    if any(
        state.radii.ndim != 1
        or len(state.radii) != horizon
        or len(state.stage_grid_indices) != horizon
        or state.coverage.ndim != 1
        or len(state.coverage) != horizon
        or state.normalized_width.ndim != 1
        or len(state.normalized_width) != horizon
        for state in states
    ):
        raise ValueError("state dimensions must match the stage-grid horizon")


def _run_search(
    states: tuple[SearchState, ...],
    stage_grids: Tensor,
    evaluator: Callable[[str, Tensor, int, Tensor], CandidateMetrics],
    *,
    target: float,
    requested_pairs: tuple[int, ...],
    first_sweep_pair: int,
    max_wall_seconds: float | None,
    clock: Callable[[], float],
    started_at: float,
) -> JointSearchOutcome:
    horizon = stage_grids.shape[0]
    trace: list[CoordinateStep] = []
    checkpoints: dict[int, JointSearchCheckpoint] = {}
    schedule_evaluations = 0
    committed_updates = 0

    for sweep_pair in range(first_sweep_pair, requested_pairs[-1] + 1):
        pair_start_states = states
        pair_trace_length = len(trace)
        pair_start_evaluations = schedule_evaluations
        pair_start_updates = committed_updates
        pair_committed = False
        updated_states = []
        deadline_reached = False
        pair_completed_at_deadline = False
        for state_index, state in enumerate(states):
            current = state
            for direction, stages in (
                ("forward", range(horizon)),
                ("reverse", range(horizon - 1, -1, -1)),
            ):
                for stage in stages:
                    if (
                        max_wall_seconds is not None
                        and clock() - started_at >= max_wall_seconds
                    ):
                        deadline_reached = True
                        break
                    before_width = _micro_width(current)
                    grid = stage_grids[stage]
                    metrics = evaluator(
                        current.start_name,
                        current.radii,
                        stage,
                        grid,
                    )
                    schedule_evaluations += len(grid)
                    choice = choose_coordinate_candidate(metrics, target=target)
                    committed = (
                        choice.micro_width is not None
                        and choice.micro_width < before_width
                    )
                    if committed:
                        assert choice.grid_index is not None
                        radii = current.radii.clone()
                        radii[stage] = grid[choice.grid_index].to(radii)
                        indices = list(current.stage_grid_indices)
                        indices[stage] = choice.grid_index
                        current = replace(
                            current,
                            radii=radii,
                            stage_grid_indices=tuple(indices),
                            coverage=metrics.coverage[choice.grid_index].clone(),
                            normalized_width=metrics.normalized_width[
                                choice.grid_index
                            ].clone(),
                        )
                        pair_committed = True
                        committed_updates += 1
                    trace.append(
                        CoordinateStep(
                            start_name=current.start_name,
                            sweep_pair=sweep_pair,
                            direction=direction,
                            stage=stage,
                            feasible_count=choice.feasible_count,
                            proposed_grid_index=choice.grid_index,
                            before_micro_width=before_width,
                            proposed_micro_width=choice.micro_width,
                            committed=committed,
                            after_micro_width=_micro_width(current),
                        )
                    )
                    if (
                        max_wall_seconds is not None
                        and clock() - started_at >= max_wall_seconds
                    ):
                        deadline_reached = True
                        pair_completed_at_deadline = (
                            state_index == len(states) - 1
                            and direction == "reverse"
                            and stage == 0
                        )
                        break
                if deadline_reached:
                    break
            if deadline_reached:
                if pair_completed_at_deadline:
                    updated_states.append(
                        replace(current, completed_sweep_pairs=sweep_pair)
                    )
                break
            updated_states.append(
                replace(current, completed_sweep_pairs=sweep_pair)
            )

        if deadline_reached and not pair_completed_at_deadline:
            states = pair_start_states
            del trace[pair_trace_length:]
            schedule_evaluations = pair_start_evaluations
            committed_updates = pair_start_updates
            return JointSearchOutcome(
                status="WALL_TIME_CAP",
                checkpoints=checkpoints,
                elapsed_seconds=clock() - started_at,
            )

        states = tuple(updated_states)

        if deadline_reached:
            if not pair_committed:
                states = tuple(
                    replace(state, converged_at_pair=sweep_pair)
                    for state in states
                )
                _materialize_converged_checkpoints(
                    checkpoints,
                    requested_pairs,
                    sweep_pair,
                    states,
                    trace,
                    schedule_evaluations,
                    committed_updates,
                )
            elif sweep_pair in requested_pairs:
                checkpoints[sweep_pair] = _checkpoint(
                    sweep_pair,
                    sweep_pair,
                    states,
                    trace,
                    schedule_evaluations,
                    committed_updates,
                )
            return JointSearchOutcome(
                status="WALL_TIME_CAP",
                checkpoints=checkpoints,
                elapsed_seconds=clock() - started_at,
            )

        if not pair_committed:
            states = tuple(
                replace(state, converged_at_pair=sweep_pair) for state in states
            )
            _materialize_converged_checkpoints(
                checkpoints,
                requested_pairs,
                sweep_pair,
                states,
                trace,
                schedule_evaluations,
                committed_updates,
            )
            break

        if sweep_pair in requested_pairs:
            checkpoints[sweep_pair] = _checkpoint(
                sweep_pair,
                sweep_pair,
                states,
                trace,
                schedule_evaluations,
                committed_updates,
            )

    return JointSearchOutcome(
        status="SELECTED",
        checkpoints=checkpoints,
        elapsed_seconds=clock() - started_at,
    )


def cyclic_joint_coordinate_search(
    starts: tuple[SearchStart, ...],
    stage_grids: Tensor,
    evaluator: Callable[[str, Tensor, int, Tensor], CandidateMetrics],
    *,
    target: float = 0.90,
    sweep_pair_checkpoints: tuple[int, ...] = (2, 4),
    max_wall_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> JointSearchOutcome:
    """Return deterministic best-found checkpoints from cyclic coordinate search."""

    started_at = clock()
    starts = _canonical_order(starts, tuple(start.name for start in starts))
    active_states = tuple(
        _state_from_start(start)
        for start in starts
        if bool((start.coverage >= target).all().item())
    )
    if not active_states:
        return JointSearchOutcome(
            status="NO_FEASIBLE_START",
            checkpoints={},
            elapsed_seconds=clock() - started_at,
        )

    requested_pairs = tuple(sorted(set(sweep_pair_checkpoints)))
    _validate_search_inputs(
        active_states,
        stage_grids,
        requested_pairs,
        max_wall_seconds,
    )
    return _run_search(
        active_states,
        stage_grids,
        evaluator,
        target=target,
        requested_pairs=requested_pairs,
        first_sweep_pair=1,
        max_wall_seconds=max_wall_seconds,
        clock=clock,
        started_at=started_at,
    )


def resume_cyclic_joint_coordinate_search(
    states: tuple[SearchState, ...],
    stage_grids: Tensor,
    evaluator: Callable[[str, Tensor, int, Tensor], CandidateMetrics],
    *,
    target: float = 0.90,
    max_wall_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> JointSearchOutcome:
    """Continue pair-four states at pair five and return only pair eight."""

    started_at = clock()
    states = _canonical_order(
        states,
        tuple(state.start_name for state in states),
    )
    if any(state.completed_sweep_pairs != 4 for state in states):
        raise ValueError("every state must have completed_sweep_pairs equal to 4")

    requested_pairs = (8,)
    _validate_search_inputs(
        states,
        stage_grids,
        requested_pairs,
        max_wall_seconds,
    )
    if any(not bool((state.coverage >= target).all().item()) for state in states):
        raise ValueError("resume states must be jointly feasible")

    converged_pairs = tuple(state.converged_at_pair for state in states)
    present_converged_pairs = tuple(
        pair for pair in converged_pairs if pair is not None
    )
    if present_converged_pairs and (
        len(present_converged_pairs) != len(states)
        or len(set(present_converged_pairs)) != 1
        or not 1 <= present_converged_pairs[0] <= 4
    ):
        raise ValueError(
            "convergence metadata must be consistent and no later than pair 4"
        )
    if all(state.converged_at_pair is not None for state in states):
        executed_pair = present_converged_pairs[0]
        checkpoint = _checkpoint(8, executed_pair, states, [], 0, 0)
        return JointSearchOutcome(
            status="SELECTED",
            checkpoints={8: checkpoint},
            elapsed_seconds=clock() - started_at,
        )

    return _run_search(
        states,
        stage_grids,
        evaluator,
        target=target,
        requested_pairs=requested_pairs,
        first_sweep_pair=5,
        max_wall_seconds=max_wall_seconds,
        clock=clock,
        started_at=started_at,
    )
