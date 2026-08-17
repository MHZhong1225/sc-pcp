"""Common-random-number oracle tuning for synthetic rollouts."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass, replace
from statistics import NormalDist

import torch
from torch import Tensor

from scpcp.config import ExperimentConfig
from scpcp.coverage import candidate_radius_schedules, fixed_q_grid
from scpcp.experiment import SeedResult, _paper_seed, _prepare_oracle_context
from scpcp.simulator import (
    SyntheticNoiseBundle,
    inverse_cdf_actions,
    make_synthetic_noise_bundle,
)


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


def _json_vector(values: Tensor | None) -> str:
    if values is None:
        return "[]"
    return json.dumps([float(value) for value in values.detach().cpu().tolist()])


def _selected_schedule_surface(selection: OracleScheduleResult, like: Tensor) -> Tensor:
    if selection.radii is None:
        return like.new_empty(0)
    return selection.radii


def _phase0_record(
    *,
    scenario: str,
    method: str,
    seed: int,
    selection: OracleScheduleResult,
    evaluation: FrozenOracleEvaluation | None,
    tuning_seed: int,
    evaluation_seed: int,
) -> dict[str, object]:
    missing = float("nan")
    return {
        "scenario": scenario,
        "method": method,
        "seed": seed,
        "selection_status": (
            "SELECTED" if selection.selection_available else "UNAVAILABLE"
        ),
        "selection_available": selection.selection_available,
        "failure_stage": selection.failure_stage,
        "selected_endpoint": selection.selected_endpoint,
        "q_by_time": _json_vector(selection.radii),
        "tuning_coverage": _json_vector(selection.tuning_coverage),
        "tuning_width": _json_vector(selection.tuning_width),
        "final_coverage": (
            "[]" if evaluation is None else _json_vector(evaluation.coverage)
        ),
        "final_wilson_lcb": (
            "[]"
            if evaluation is None
            else _json_vector(evaluation.wilson_lower_bound)
        ),
        "final_stage_width": (
            "[]"
            if evaluation is None
            else _json_vector(evaluation.normalized_width)
        ),
        "micro_normalized_width": (
            missing if evaluation is None else evaluation.micro_normalized_width
        ),
        "patient_normalized_width": (
            missing if evaluation is None else evaluation.patient_normalized_width
        ),
        "tuning_seed": tuning_seed,
        "evaluation_seed": evaluation_seed,
        "n_rollouts": 0 if evaluation is None else evaluation.n_rollouts,
    }


def _selection_diagnostics(selection: OracleScheduleResult) -> dict[str, object]:
    return {
        "selection_available": selection.selection_available,
        "failure_stage": selection.failure_stage,
        "selected_endpoint": selection.selected_endpoint,
        "selected_indices": list(selection.selected_indices),
    }


def _fresh_evaluation_diagnostics(
    selection: OracleScheduleResult,
    evaluation: FrozenOracleEvaluation | None,
) -> dict[str, object]:
    missing = float("nan")
    return {
        **_selection_diagnostics(selection),
        "selected_schedule": (
            []
            if selection.radii is None
            else [float(value) for value in selection.radii.detach().cpu().tolist()]
        ),
        "final_coverage": (
            []
            if evaluation is None
            else [float(value) for value in evaluation.coverage.detach().cpu().tolist()]
        ),
        "final_wilson_lcb": (
            []
            if evaluation is None
            else [
                float(value)
                for value in evaluation.wilson_lower_bound.detach().cpu().tolist()
            ]
        ),
        "final_stage_width": (
            []
            if evaluation is None
            else [
                float(value)
                for value in evaluation.normalized_width.detach().cpu().tolist()
            ]
        ),
        "micro_normalized_width": (
            missing if evaluation is None else evaluation.micro_normalized_width
        ),
        "patient_normalized_width": (
            missing if evaluation is None else evaluation.patient_normalized_width
        ),
        "n_rollouts": 0 if evaluation is None else evaluation.n_rollouts,
    }


def _fresh_evaluation_surfaces(
    prefix: str,
    evaluation: FrozenOracleEvaluation | None,
    like: Tensor,
) -> dict[str, Tensor]:
    if evaluation is None:
        empty = like.new_empty(0)
        return {
            f"{prefix}final_coverage": empty,
            f"{prefix}final_wilson_lcb": empty,
            f"{prefix}final_stage_width": empty,
            f"{prefix}micro_normalized_width": like.new_tensor(float("nan")),
            f"{prefix}patient_normalized_width": like.new_tensor(float("nan")),
            f"{prefix}n_rollouts": like.new_tensor(0),
        }
    return {
        f"{prefix}final_coverage": evaluation.coverage,
        f"{prefix}final_wilson_lcb": evaluation.wilson_lower_bound,
        f"{prefix}final_stage_width": evaluation.normalized_width,
        f"{prefix}micro_normalized_width": like.new_tensor(
            evaluation.micro_normalized_width
        ),
        f"{prefix}patient_normalized_width": like.new_tensor(
            evaluation.patient_normalized_width
        ),
        f"{prefix}n_rollouts": like.new_tensor(evaluation.n_rollouts),
    }


def run_phase0_seed(
    config: ExperimentConfig,
    *,
    seed: int,
    device: str,
) -> SeedResult:
    """Build the paired standard/tail-shift oracle result for one base seed."""

    if config.data.dataset != "synthetic":
        raise ValueError("run_phase0_seed requires data.dataset='synthetic'")

    records = []
    surfaces = {}
    diagnostics = {}
    target = 1.0 - config.certification.alpha

    for scenario_index, scenario in enumerate(("standard", "tail_shift")):
        torch.manual_seed(seed)
        scenario_config = replace(
            config,
            synthetic=replace(config.synthetic, scenario=scenario),
        )
        scenario_config.validate()
        context = _prepare_oracle_context(
            scenario_config,
            seed=seed,
            device=device,
        )
        profile = context.schedule_family.profile
        profiled_scale_grid = context.schedule_family.scale_grid
        profiled_schedules = candidate_radius_schedules(
            profiled_scale_grid,
            profile,
        )
        stage_grids = torch.stack(
            [
                fixed_q_grid(
                    context.cot_scores[:, stage],
                    size=config.q_grid_size,
                    lower_quantile=config.q_quantile_min,
                    upper_quantile=config.q_quantile_max,
                )
                for stage in range(config.horizon)
            ]
        )
        common_scale_grid = fixed_q_grid(
            context.cot_scores / profile[None, :],
            size=config.q_grid_size,
            lower_quantile=config.q_quantile_min,
            upper_quantile=config.q_quantile_max,
        )
        common_profiled_schedules = candidate_radius_schedules(
            common_scale_grid,
            profile,
        )

        tuning_seed = _paper_seed(seed, 1_300_001 + scenario_index)
        evaluation_seed = _paper_seed(seed, 1_400_001 + scenario_index)
        if tuning_seed == evaluation_seed:
            raise RuntimeError("phase0 tuning and evaluation streams must differ")
        tuning_noise = make_synthetic_noise_bundle(
            n=config.samples.oracle_surface_rollouts,
            horizon=config.horizon,
            seed=tuning_seed,
            device=device,
        )
        profiled_metrics = evaluate_profiled_candidates_crn(
            context.task.environment,
            context.policy,
            context.outcome_model,
            candidate_schedules=profiled_schedules,
            outcome_sd=context.outcome_sd,
            noise=tuning_noise,
            chunk_size=16,
        )
        profiled_selection = select_profiled_oracle_schedule(
            profiled_schedules,
            profiled_metrics,
            target=target,
        )
        common_profiled_metrics = evaluate_profiled_candidates_crn(
            context.task.environment,
            context.policy,
            context.outcome_model,
            candidate_schedules=common_profiled_schedules,
            outcome_sd=context.outcome_sd,
            noise=tuning_noise,
            chunk_size=16,
        )
        common_profiled_selection = select_profiled_oracle_schedule(
            common_profiled_schedules,
            common_profiled_metrics,
            target=target,
        )
        greedy_selection = greedy_sequential_oracle_schedule(
            context.task.environment,
            context.policy,
            context.outcome_model,
            stage_grids=stage_grids,
            outcome_sd=context.outcome_sd,
            noise=tuning_noise,
            target=target,
            chunk_size=16,
        )

        frozen_schedules = {}
        if profiled_selection.radii is not None:
            frozen_schedules["profiled"] = profiled_selection.radii
        if greedy_selection.radii is not None:
            frozen_schedules["greedy"] = greedy_selection.radii
        if common_profiled_selection.radii is not None:
            frozen_schedules["profiled_common_grid"] = (
                common_profiled_selection.radii
            )
        evaluation_noise = make_synthetic_noise_bundle(
            n=config.samples.oracle_rollouts,
            horizon=config.horizon,
            seed=evaluation_seed,
            device=device,
        )
        evaluations = (
            evaluate_frozen_schedules_crn(
                context.task.environment,
                context.policy,
                context.outcome_model,
                schedules=frozen_schedules,
                noise=evaluation_noise,
                outcome_sd=context.outcome_sd,
                forbidden_noise_seeds={tuning_seed},
            )
            if frozen_schedules
            else {}
        )
        common_profiled_evaluation = evaluations.get("profiled_common_grid")

        records.extend(
            (
                _phase0_record(
                    scenario=scenario,
                    method="Current Profiled Oracle",
                    seed=seed,
                    selection=profiled_selection,
                    evaluation=evaluations.get("profiled"),
                    tuning_seed=tuning_seed,
                    evaluation_seed=evaluation_seed,
                ),
                _phase0_record(
                    scenario=scenario,
                    method="Greedy Sequential Oracle",
                    seed=seed,
                    selection=greedy_selection,
                    evaluation=evaluations.get("greedy"),
                    tuning_seed=tuning_seed,
                    evaluation_seed=evaluation_seed,
                ),
            )
        )

        prefix = f"{scenario}_"
        surfaces.update(
            {
                f"{prefix}profiled_scale_grid": profiled_scale_grid,
                f"{prefix}profile": profile,
                f"{prefix}profiled_candidate_schedules": profiled_schedules,
                f"{prefix}profiled_candidate_coverage": profiled_metrics.coverage,
                f"{prefix}profiled_candidate_normalized_width": (
                    profiled_metrics.normalized_width
                ),
                f"{prefix}profiled_selected_schedule": _selected_schedule_surface(
                    profiled_selection,
                    profile,
                ),
                f"{prefix}greedy_stage_grids": stage_grids,
                f"{prefix}greedy_selected_schedule": _selected_schedule_surface(
                    greedy_selection,
                    profile,
                ),
                f"{prefix}profiled_common_grid_scale_grid": common_scale_grid,
                f"{prefix}profiled_common_grid_candidate_schedules": (
                    common_profiled_schedules
                ),
                f"{prefix}profiled_common_grid_candidate_coverage": (
                    common_profiled_metrics.coverage
                ),
                f"{prefix}profiled_common_grid_candidate_normalized_width": (
                    common_profiled_metrics.normalized_width
                ),
                f"{prefix}profiled_common_grid_selected_schedule": (
                    _selected_schedule_surface(common_profiled_selection, profile)
                ),
            }
        )
        surfaces.update(
            _fresh_evaluation_surfaces(
                f"{prefix}profiled_common_grid_",
                common_profiled_evaluation,
                profile,
            )
        )
        diagnostics[scenario] = {
            "tuning_seed": tuning_seed,
            "evaluation_seed": evaluation_seed,
            "tuning_rollouts": config.samples.oracle_surface_rollouts,
            "evaluation_rollouts": config.samples.oracle_rollouts,
            "profiled": _selection_diagnostics(profiled_selection),
            "greedy": _selection_diagnostics(greedy_selection),
            "profiled_common_grid": _fresh_evaluation_diagnostics(
                common_profiled_selection,
                common_profiled_evaluation,
            ),
        }

    return SeedResult(
        seed=seed,
        device=device,
        records=records,
        surfaces=surfaces,
        diagnostics=diagnostics,
    )
