"""Analytic finite-MDP schedule-search diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class AnalyticFiniteMDP:
    initial_state_probabilities: Tensor
    transition_probabilities: Tensor
    action_probabilities: Tensor
    radii: Tensor
    predictor_means: Tensor
    predictor_scales: Tensor
    outcome_means: Tensor
    outcome_standard_deviations: Tensor
    outcome_normalization: Tensor


@dataclass(frozen=True)
class ScheduleEvaluation:
    selected_indices: tuple[int, ...]
    coverage: Tensor
    normalized_width: Tensor
    state_occupancy: Tensor


@dataclass(frozen=True)
class SearchDiagnostic:
    search_type: str
    greedy_width: float | None
    best_found_width: float
    true_optimality_gap: float | None
    best_found_gap: float | None
    greedy_available: bool
    greedy_schedule: ScheduleEvaluation | None
    best_found_schedule: ScheduleEvaluation


class NoFeasibleScheduleError(RuntimeError):
    """Raised when a finite search contains no feasible schedule."""


@dataclass(frozen=True)
class _AnalyticKernels:
    problem: AnalyticFiniteMDP
    coverage_by_state: Tensor
    width_by_state: Tensor
    transition_by_state: Tensor


def analytic_schedule_metrics(
    problem: AnalyticFiniteMDP,
    selected_indices: tuple[int, ...],
) -> ScheduleEvaluation:
    """Evaluate one frozen schedule under the exact population recursion."""

    kernels = _analytic_kernels(problem)
    return _evaluate_schedule(kernels, selected_indices)


def greedy_schedule_search(
    problem: AnalyticFiniteMDP,
    *,
    target: float,
) -> ScheduleEvaluation:
    """Choose the narrowest feasible radius at each committed occupancy."""

    _validate_target(target)
    return _greedy_schedule(_analytic_kernels(problem), target=target)


def exact_schedule_search(
    problem: AnalyticFiniteMDP,
    *,
    target: float,
) -> SearchDiagnostic:
    """Enumerate the complete frozen grid and report a true finite-grid gap."""

    _validate_target(target)
    kernels = _analytic_kernels(problem)
    horizon, grid_size = kernels.problem.radii.shape
    feasible: list[ScheduleEvaluation] = []
    for indices in product(range(grid_size), repeat=horizon):
        evaluation = _evaluate_schedule(kernels, indices)
        if bool(evaluation.coverage.ge(target).all()):
            feasible.append(evaluation)
    if not feasible:
        raise NoFeasibleScheduleError("no feasible exact schedule")
    best = min(feasible, key=_schedule_ordering_key)
    try:
        greedy = _greedy_schedule(kernels, target=target)
    except NoFeasibleScheduleError:
        greedy = None
    return _search_diagnostic(
        search_type="exact",
        greedy=greedy,
        best=best,
        exact=True,
    )


def beam_schedule_search(
    problem: AnalyticFiniteMDP,
    *,
    target: float,
    beam_width: int,
) -> SearchDiagnostic:
    """Search feasible prefixes while labeling the result as best-found only."""

    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    _validate_target(target)
    kernels = _analytic_kernels(problem)
    horizon, grid_size = kernels.problem.radii.shape
    prefixes: list[tuple[int, ...]] = [()]
    final_evaluations: list[ScheduleEvaluation] = []
    for stage in range(horizon):
        candidates = []
        for prefix in prefixes:
            for index in range(grid_size):
                evaluation = _evaluate_schedule(kernels, prefix + (index,))
                if float(evaluation.coverage[-1].item()) >= target:
                    candidates.append(evaluation)
        if not candidates:
            raise NoFeasibleScheduleError(
                f"no feasible beam prefix at stage {stage}"
            )
        candidates.sort(key=_schedule_ordering_key)
        final_evaluations = candidates[:beam_width]
        prefixes = [candidate.selected_indices for candidate in final_evaluations]

    best = min(final_evaluations, key=_schedule_ordering_key)
    try:
        greedy = _greedy_schedule(kernels, target=target)
    except NoFeasibleScheduleError:
        greedy = None
    if greedy is not None:
        best = min((best, greedy), key=_schedule_ordering_key)
    return _search_diagnostic(
        search_type="beam",
        greedy=greedy,
        best=best,
        exact=False,
    )


def _canonical_problem(problem: AnalyticFiniteMDP) -> AnalyticFiniteMDP:
    initial = problem.initial_state_probabilities
    if initial.ndim != 1 or not initial.is_floating_point():
        raise ValueError("initial_state_probabilities must have shape [S]")
    device = initial.device

    def as_float64(value: Tensor) -> Tensor:
        return value.to(device=device, dtype=torch.float64)

    canonical = AnalyticFiniteMDP(
        initial_state_probabilities=as_float64(initial),
        transition_probabilities=as_float64(problem.transition_probabilities),
        action_probabilities=as_float64(problem.action_probabilities),
        radii=as_float64(problem.radii),
        predictor_means=as_float64(problem.predictor_means),
        predictor_scales=as_float64(problem.predictor_scales),
        outcome_means=as_float64(problem.outcome_means),
        outcome_standard_deviations=as_float64(
            problem.outcome_standard_deviations
        ),
        outcome_normalization=as_float64(problem.outcome_normalization),
    )
    initial = canonical.initial_state_probabilities
    transition = canonical.transition_probabilities
    action_probabilities = canonical.action_probabilities
    radii = canonical.radii
    predictor_means = canonical.predictor_means
    predictor_scales = canonical.predictor_scales
    outcome_means = canonical.outcome_means
    outcome_standard_deviations = canonical.outcome_standard_deviations
    outcome_normalization = canonical.outcome_normalization

    state_count = len(initial)
    if transition.ndim != 3 or transition.shape[1:] != (
        state_count,
        state_count,
    ):
        raise ValueError("transition_probabilities must have shape [A,S,S]")
    action_count = transition.shape[0]
    if action_probabilities.ndim != 4 or action_probabilities.shape[2:] != (
        state_count,
        action_count,
    ):
        raise ValueError("action_probabilities must have shape [T,K,S,A]")
    horizon, grid_size = action_probabilities.shape[:2]
    if radii.shape != (horizon, grid_size):
        raise ValueError("radii must have shape [T,K]")
    if predictor_means.ndim != 3 or predictor_means.shape[:2] != (
        state_count,
        action_count,
    ):
        raise ValueError("predictor_means must have shape [S,A,D]")
    if predictor_scales.shape != predictor_means.shape:
        raise ValueError("predictor_scales must match predictor_means")
    outcome_dim = predictor_means.shape[2]
    if horizon < 1 or grid_size < 1 or outcome_dim < 1:
        raise ValueError(
            "horizon, grid, and outcome dimensions must be positive"
        )
    if outcome_means.shape != (action_count, state_count, outcome_dim):
        raise ValueError("outcome_means must have shape [A,S_next,D]")
    if outcome_standard_deviations.shape != (outcome_dim,):
        raise ValueError("outcome_standard_deviations must have shape [D]")
    if outcome_normalization.shape != (outcome_dim,):
        raise ValueError("outcome_normalization must have shape [D]")

    values = (
        initial,
        transition,
        action_probabilities,
        radii,
        predictor_means,
        predictor_scales,
        outcome_means,
        outcome_standard_deviations,
        outcome_normalization,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("analytic finite-MDP inputs must be finite")
    if (
        bool((initial < 0.0).any())
        or bool((transition < 0.0).any())
        or bool((action_probabilities < 0.0).any())
        or bool((radii < 0.0).any())
    ):
        raise ValueError("probabilities and radii must be nonnegative")
    if (
        bool((predictor_scales <= 0.0).any())
        or bool((outcome_standard_deviations <= 0.0).any())
        or bool((outcome_normalization <= 0.0).any())
    ):
        raise ValueError("all analytic scales must be positive")
    one = torch.ones((), device=device, dtype=torch.float64)
    if not torch.allclose(initial.sum(), one, atol=1e-6, rtol=0.0):
        raise ValueError("initial-state probabilities must sum to one")
    if not torch.allclose(
        transition.sum(dim=2),
        torch.ones_like(transition[:, :, 0]),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("transition rows must sum to one")
    if not torch.allclose(
        action_probabilities.sum(dim=3),
        torch.ones_like(action_probabilities[:, :, :, 0]),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("action probabilities must sum to one")
    return canonical


def _analytic_kernels(problem: AnalyticFiniteMDP) -> _AnalyticKernels:
    problem = _canonical_problem(problem)
    radii = problem.radii[:, :, None, None, None, None]
    predictor_means = problem.predictor_means[None, None, :, :, None, :]
    predictor_scales = problem.predictor_scales[None, None, :, :, None, :]
    outcome_means = problem.outcome_means[None, None, None, :, :, :]
    outcome_standard_deviations = problem.outcome_standard_deviations[
        None, None, None, None, None, :
    ]
    lower = (predictor_means - radii * predictor_scales - outcome_means) / (
        outcome_standard_deviations
    )
    upper = (predictor_means + radii * predictor_scales - outcome_means) / (
        outcome_standard_deviations
    )
    conditional_hits = (
        (torch.special.ndtr(upper) - torch.special.ndtr(lower))
        .clamp(0.0, 1.0)
        .prod(dim=5)
    )
    coverage_by_state = torch.einsum(
        "tksa,asr,tksar->tks",
        problem.action_probabilities,
        problem.transition_probabilities,
        conditional_hits,
    )
    transition_by_state = torch.einsum(
        "tksa,asr->tksr",
        problem.action_probabilities,
        problem.transition_probabilities,
    )
    base_width = 2.0 * (
        problem.predictor_scales / problem.outcome_normalization[None, None, :]
    ).mean(dim=2)
    conditional_width = (
        problem.radii[:, :, None, None] * base_width[None, None, :, :]
    )
    width_by_state = torch.einsum(
        "tksa,tksa->tks",
        problem.action_probabilities,
        conditional_width,
    )
    return _AnalyticKernels(
        problem=problem,
        coverage_by_state=coverage_by_state,
        width_by_state=width_by_state,
        transition_by_state=transition_by_state,
    )


def _evaluate_schedule(
    kernels: _AnalyticKernels,
    selected_indices: tuple[int, ...],
) -> ScheduleEvaluation:
    horizon, grid_size = kernels.problem.radii.shape
    if len(selected_indices) > horizon:
        raise ValueError("schedule cannot exceed the problem horizon")
    if any(type(index) is not int for index in selected_indices):
        raise TypeError("schedule indices must be integers")
    if any(index < 0 or index >= grid_size for index in selected_indices):
        raise IndexError("schedule index is outside the frozen grid")

    occupancy = kernels.problem.initial_state_probabilities
    occupancies = [occupancy]
    coverage = []
    widths = []
    for stage, index in enumerate(selected_indices):
        coverage.append(
            torch.dot(occupancy, kernels.coverage_by_state[stage, index])
        )
        widths.append(torch.dot(occupancy, kernels.width_by_state[stage, index]))
        occupancy = torch.einsum(
            "s,sr->r",
            occupancy,
            kernels.transition_by_state[stage, index],
        )
        occupancies.append(occupancy)
    like = kernels.problem.initial_state_probabilities
    return ScheduleEvaluation(
        selected_indices=selected_indices,
        coverage=torch.stack(coverage) if coverage else like.new_empty(0),
        normalized_width=torch.stack(widths) if widths else like.new_empty(0),
        state_occupancy=torch.stack(occupancies),
    )


def _greedy_schedule(
    kernels: _AnalyticKernels,
    *,
    target: float,
) -> ScheduleEvaluation:
    horizon, grid_size = kernels.problem.radii.shape
    occupancy = kernels.problem.initial_state_probabilities
    selected_indices = []
    for stage in range(horizon):
        coverage = torch.einsum(
            "s,ks->k",
            occupancy,
            kernels.coverage_by_state[stage],
        )
        widths = torch.einsum(
            "s,ks->k",
            occupancy,
            kernels.width_by_state[stage],
        )
        feasible = coverage.ge(target)
        if not bool(feasible.any()):
            raise NoFeasibleScheduleError(
                f"no feasible greedy candidate at stage {stage}"
            )
        objective = torch.where(
            feasible,
            widths,
            torch.full(
                (grid_size,),
                torch.inf,
                dtype=widths.dtype,
                device=widths.device,
            ),
        )
        index = int(objective.argmin().item())
        selected_indices.append(index)
        occupancy = torch.einsum(
            "s,sr->r",
            occupancy,
            kernels.transition_by_state[stage, index],
        )
    return _evaluate_schedule(kernels, tuple(selected_indices))


def _schedule_ordering_key(
    evaluation: ScheduleEvaluation,
) -> tuple[float, tuple[int, ...]]:
    width = float(evaluation.normalized_width.mean().item())
    return width, evaluation.selected_indices


def _search_diagnostic(
    *,
    search_type: str,
    greedy: ScheduleEvaluation | None,
    best: ScheduleEvaluation,
    exact: bool,
) -> SearchDiagnostic:
    best_width = float(best.normalized_width.mean().item())
    greedy_width = (
        None if greedy is None else float(greedy.normalized_width.mean().item())
    )
    gap = None if greedy_width is None else greedy_width - best_width
    return SearchDiagnostic(
        search_type=search_type,
        greedy_width=greedy_width,
        best_found_width=best_width,
        true_optimality_gap=gap if exact else None,
        best_found_gap=gap,
        greedy_available=greedy is not None,
        greedy_schedule=greedy,
        best_found_schedule=best,
    )


def _validate_target(target: float) -> None:
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target must lie in (0, 1)")
