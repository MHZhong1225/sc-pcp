"""Common-random-number oracle tuning for synthetic rollouts."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from statistics import NormalDist

import torch
from torch import Tensor

from scpcp.simulator import SyntheticNoiseBundle, inverse_cdf_actions


@dataclass(frozen=True)
class CandidateMetrics:
    coverage: Tensor
    normalized_width: Tensor


@dataclass(frozen=True)
class FrozenOracleEvaluation:
    coverage: Tensor
    wilson_lower_bound: Tensor
    normalized_width: Tensor
    micro_normalized_width: float
    patient_normalized_width: float
    n_rollouts: int


@dataclass(frozen=True)
class OracleScheduleResult:
    radii: Tensor | None
    selected_indices: tuple[int, ...]
    tuning_coverage: Tensor | None
    tuning_width: Tensor | None
    selection_available: bool
    failure_stage: int | None
    selected_endpoint: bool


def bonferroni_wilson_lower_bounds(
    hits: Tensor,
    *,
    family_alpha: float = 0.05,
) -> Tensor:
    """Return one-sided Wilson lower bounds for patient-by-stage hits."""

    n_rollouts, horizon = hits.shape
    proportions = hits.float().mean(dim=0)
    z = NormalDist().inv_cdf(1.0 - family_alpha / horizon)
    denominator = 1.0 + z**2 / n_rollouts
    center = proportions + z**2 / (2.0 * n_rollouts)
    spread = z * torch.sqrt(
        proportions * (1.0 - proportions) / n_rollouts
        + z**2 / (4.0 * n_rollouts**2)
    )
    return (center - spread) / denominator


def _repeat_over_candidates(values: Tensor, count: int) -> Tensor:
    expanded = values.unsqueeze(0).expand(count, *values.shape)
    return expanded.reshape(count * len(values), *values.shape[1:])


def _allocate_candidate_next_states(state: Tensor, candidate_count: int) -> Tensor:
    return state.new_empty((candidate_count, *state.shape))


def _evaluate_stage(
    environment: object,
    policy: object,
    outcome_model: object,
    *,
    states: Tensor,
    radii: Tensor,
    outcome_sd: Tensor,
    noise: SyntheticNoiseBundle,
    stage: int,
) -> tuple[Tensor, Tensor, Tensor]:
    candidate_count, patient_count, state_dim = states.shape
    flat_states = states.reshape(candidate_count * patient_count, state_dim)
    flat_radii = radii[:, None].expand(-1, patient_count).reshape(-1)
    probabilities = policy.probabilities(flat_states, flat_radii)
    uniforms = noise.action_uniform[stage][None, :].expand(candidate_count, -1)
    actions = inverse_cdf_actions(probabilities, uniforms.reshape(-1))
    next_states, outcomes = environment.step_from_noise(
        flat_states,
        actions,
        shared=_repeat_over_candidates(noise.shared_normal[stage], candidate_count),
        independent=_repeat_over_candidates(
            noise.independent_normal[stage], candidate_count
        ),
        innovations=_repeat_over_candidates(
            noise.innovation_normal[stage], candidate_count
        ),
        difficulty_uniform=_repeat_over_candidates(
            noise.difficulty_uniform[stage], candidate_count
        ),
        contamination_uniform=_repeat_over_candidates(
            noise.contamination_uniform[stage], candidate_count
        ),
    )
    means, scales = outcome_model(flat_states, actions)
    scores = ((outcomes - means).abs() / scales).amax(dim=1)
    coverage = (scores <= flat_radii).reshape(candidate_count, patient_count).float().mean(dim=1)
    normalized_width = 2.0 * flat_radii[:, None] * scales / outcome_sd.to(scales)[None, :]
    mean_width = normalized_width.reshape(candidate_count, patient_count, -1).mean(dim=(1, 2))
    return next_states.reshape(candidate_count, patient_count, -1), coverage, mean_width


@torch.no_grad()
def evaluate_profiled_candidates_crn(
    environment: object,
    policy: object,
    outcome_model: object,
    *,
    candidate_schedules: Tensor,
    outcome_sd: Tensor,
    noise: SyntheticNoiseBundle,
    chunk_size: int,
) -> CandidateMetrics:
    """Evaluate schedules with a frozen ``(mean, scale)`` predictor.

    ``outcome_model`` is the callable ``GaussianOutcomeModel`` itself, not a
    ``ConformalRegion`` wrapper. Candidate state is materialized for one chunk.
    """

    initial_state = environment.initial_state_from_noise(noise)
    horizon = candidate_schedules.shape[1]
    coverage_chunks = []
    width_chunks = []
    for start in range(0, len(candidate_schedules), chunk_size):
        schedules = candidate_schedules[start : start + chunk_size].to(initial_state)
        states = initial_state[None, :, :].expand(len(schedules), -1, -1).clone()
        stage_coverage = []
        stage_width = []
        for stage in range(horizon):
            states, coverage, width = _evaluate_stage(
                environment,
                policy,
                outcome_model,
                states=states,
                radii=schedules[:, stage],
                outcome_sd=outcome_sd,
                noise=noise,
                stage=stage,
            )
            stage_coverage.append(coverage)
            stage_width.append(width)
        coverage_chunks.append(torch.stack(stage_coverage, dim=1))
        width_chunks.append(torch.stack(stage_width, dim=1))
    return CandidateMetrics(
        coverage=torch.cat(coverage_chunks),
        normalized_width=torch.cat(width_chunks),
    )


@torch.no_grad()
def evaluate_frozen_schedules_crn(
    environment: object,
    policy: object,
    outcome_model: object,
    *,
    schedules: dict[str, Tensor],
    noise: SyntheticNoiseBundle,
    outcome_sd: Tensor,
    forbidden_noise_seeds: Collection[int] = (),
) -> dict[str, FrozenOracleEvaluation]:
    """Evaluate frozen schedules independently on one supplied CRN bundle."""

    if noise.seed in forbidden_noise_seeds:
        raise ValueError(
            f"evaluation noise seed {noise.seed} is forbidden by "
            "tuning/construction streams"
        )
    horizon = noise.action_uniform.shape[0]
    for name, schedule in schedules.items():
        if schedule.ndim != 1 or schedule.shape[0] != horizon:
            raise ValueError(
                f"schedule {name!r} must have shape ({horizon},), "
                f"got {tuple(schedule.shape)}"
            )

    evaluations = {}
    for name, schedule in schedules.items():
        state = environment.initial_state_from_noise(noise)
        stage_hits = []
        stage_widths = []
        patient_stage_widths = []
        for stage, radius in enumerate(schedule.to(state)):
            probabilities = policy.probabilities(state, radius)
            actions = inverse_cdf_actions(probabilities, noise.action_uniform[stage])
            next_state, outcomes = environment.step_from_noise(
                state,
                actions,
                shared=noise.shared_normal[stage],
                independent=noise.independent_normal[stage],
                innovations=noise.innovation_normal[stage],
                difficulty_uniform=noise.difficulty_uniform[stage],
                contamination_uniform=noise.contamination_uniform[stage],
            )
            means, scales = outcome_model(state, actions)
            scores = ((outcomes - means).abs() / scales).amax(dim=1)
            stage_hits.append(scores <= radius)
            normalized_width = (
                2.0 * radius * scales / outcome_sd.to(scales)[None, :]
            )
            patient_width = normalized_width.mean(dim=1)
            stage_widths.append(patient_width.mean())
            patient_stage_widths.append(patient_width)
            state = next_state

        hits = torch.stack(stage_hits, dim=1)
        widths = torch.stack(stage_widths)
        patient_widths = torch.stack(patient_stage_widths, dim=1)
        evaluations[name] = FrozenOracleEvaluation(
            coverage=hits.float().mean(dim=0),
            wilson_lower_bound=bonferroni_wilson_lower_bounds(hits),
            normalized_width=widths,
            micro_normalized_width=float(patient_widths.mean().item()),
            patient_normalized_width=float(
                patient_widths.mean(dim=1).mean().item()
            ),
            n_rollouts=len(noise.initial_normal),
        )
    return evaluations


@torch.no_grad()
def greedy_sequential_oracle_schedule(
    environment: object,
    policy: object,
    outcome_model: object,
    *,
    stage_grids: Tensor,
    outcome_sd: Tensor,
    noise: SyntheticNoiseBundle,
    target: float = 0.90,
    chunk_size: int = 16,
) -> OracleScheduleResult:
    """Choose radii under the committed prefix with a frozen predictor.

    ``outcome_model`` is the callable ``GaussianOutcomeModel`` itself, not a
    ``ConformalRegion`` wrapper.
    """

    state = environment.initial_state_from_noise(noise)
    selected_indices = []
    selected_radii = []
    selected_coverage = []
    selected_width = []
    selected_endpoint = False
    candidate_count = stage_grids.shape[1]
    for stage, stage_grid in enumerate(stage_grids):
        candidate_next_states = _allocate_candidate_next_states(
            state,
            candidate_count,
        )
        coverage = state.new_empty(candidate_count)
        width = state.new_empty(candidate_count)
        for start in range(0, candidate_count, chunk_size):
            radii = stage_grid[start : start + chunk_size].to(state)
            candidate_states = state[None, :, :].expand(len(radii), -1, -1)
            next_states, chunk_coverage, chunk_width = _evaluate_stage(
                environment,
                policy,
                outcome_model,
                states=candidate_states,
                radii=radii,
                outcome_sd=outcome_sd,
                noise=noise,
                stage=stage,
            )
            stop = start + len(radii)
            candidate_next_states[start:stop] = next_states
            coverage[start:stop] = chunk_coverage
            width[start:stop] = chunk_width
        feasible = coverage.ge(target)
        if not bool(feasible.any()):
            return OracleScheduleResult(
                radii=None,
                selected_indices=tuple(selected_indices),
                tuning_coverage=None,
                tuning_width=None,
                selection_available=False,
                failure_stage=stage,
                selected_endpoint=False,
            )
        objective = torch.where(feasible, width, torch.full_like(width, torch.inf))
        index = int(objective.argmin().item())
        selected_indices.append(index)
        selected_radii.append(stage_grid[index])
        selected_coverage.append(coverage[index].clone())
        selected_width.append(width[index].clone())
        selected_endpoint = selected_endpoint or index in {0, candidate_count - 1}
        state = candidate_next_states[index].clone()
        del candidate_next_states
    return OracleScheduleResult(
        radii=torch.stack(selected_radii),
        selected_indices=tuple(selected_indices),
        tuning_coverage=torch.stack(selected_coverage),
        tuning_width=torch.stack(selected_width),
        selection_available=True,
        failure_stage=None,
        selected_endpoint=selected_endpoint,
    )


def select_profiled_oracle_schedule(
    candidate_schedules: Tensor,
    metrics: CandidateMetrics,
    *,
    target: float = 0.90,
) -> OracleScheduleResult:
    """Select the minimum-width schedule satisfying every stage target.

    When no schedule is jointly feasible, ``failure_stage`` identifies only an
    earliest stage where no candidate passes. It remains ``None`` when every
    stage has a passing candidate but those passes belong to different schedules.
    """

    passing = metrics.coverage.ge(target)
    feasible = passing.all(dim=1)
    if not bool(feasible.any()):
        stages_without_pass = (~passing.any(dim=0)).nonzero().flatten()
        failure_stage = (
            int(stages_without_pass[0].item()) if len(stages_without_pass) else None
        )
        return OracleScheduleResult(
            radii=None,
            selected_indices=(),
            tuning_coverage=None,
            tuning_width=None,
            selection_available=False,
            failure_stage=failure_stage,
            selected_endpoint=False,
        )
    mean_width = metrics.normalized_width.mean(dim=1)
    objective = torch.where(
        feasible,
        mean_width,
        torch.full_like(mean_width, torch.inf),
    )
    index = int(objective.argmin().item())
    return OracleScheduleResult(
        radii=candidate_schedules[index],
        selected_indices=(index,),
        tuning_coverage=metrics.coverage[index],
        tuning_width=metrics.normalized_width[index],
        selection_available=True,
        failure_stage=None,
        selected_endpoint=index in {0, len(candidate_schedules) - 1},
    )
