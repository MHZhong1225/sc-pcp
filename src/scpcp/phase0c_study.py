"""Per-seed orchestration for the Phase 0C joint-search audit."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace

import torch
from torch import Tensor

from scpcp.config import ExperimentConfig
from scpcp.coverage import candidate_radius_schedules, fixed_q_grid
from scpcp.experiment import SeedResult, _paper_seed, _prepare_oracle_context
from scpcp.phase0_oracle import (
    FrozenOracleEvaluation,
    OracleScheduleResult,
    evaluate_frozen_schedules_crn,
    evaluate_profiled_candidates_crn,
    greedy_sequential_oracle_schedule,
    select_profiled_oracle_schedule,
)
from scpcp.phase0c_joint_search import (
    CRNCoordinateEvaluator,
    CoordinateStep,
    JointSearchCheckpoint,
    JointSearchOutcome,
    SearchStart,
    SearchState,
    cyclic_joint_coordinate_search,
    resume_cyclic_joint_coordinate_search,
)
from scpcp.simulator import SyntheticNoiseBundle, make_synthetic_noise_bundle


PHASE0C_METHODS = ("current_profiled", "greedy", "joint_B", "joint_2B")

METHOD_METADATA = {
    "current_profiled": ("REFERENCE", 0),
    "greedy": ("REFERENCE", 0),
    "joint_B": ("B", 2),
    "joint_2B": ("2B", 4),
}

_SCENARIOS = ("standard", "tail_shift")
_SCHEMA_VERSION = "phase0c_seed_v1"
_VALID_STATUSES = {"SELECTED", "NO_FEASIBLE_START", "WALL_TIME_CAP"}


@dataclass(frozen=True)
class Phase0CScenarioContext:
    environment: object
    policy: object
    outcome_model: object
    outcome_sd: Tensor
    profile: Tensor
    profiled_scale_grid: Tensor
    profiled_schedules: Tensor
    stage_grids: Tensor
    tuning_noise: SyntheticNoiseBundle
    evaluation_noise: SyntheticNoiseBundle
    starts: tuple[SearchStart, ...]
    profiled_selection: OracleScheduleResult
    greedy_selection: OracleScheduleResult


@dataclass(frozen=True)
class _MethodResult:
    method_id: str
    status: str
    radii: Tensor | None = None
    indices: tuple[int | None, ...] = ()
    tuning_coverage: Tensor | None = None
    tuning_width: Tensor | None = None
    chosen_initialization: str = ""
    endpoint_count: int = 0
    converged_at_pair: int | None = None
    wall_time_seconds: float = 0.0
    schedule_evaluations: int = 0
    committed_updates: int = 0


def _validate_common_inputs(
    config: ExperimentConfig,
    *,
    candidate_chunk_size: int,
    max_seed_wall_seconds: float,
) -> None:
    if config.data.dataset != "synthetic":
        raise ValueError("Phase 0C requires data.dataset='synthetic'")
    if candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be positive")
    if not math.isfinite(max_seed_wall_seconds) or max_seed_wall_seconds <= 0.0:
        raise ValueError("max_seed_wall_seconds must be finite and positive")
    if config.horizon != 12 or config.q_grid_size != 101:
        raise ValueError("Phase 0C protocol requires horizon=12 and q_grid_size=101")
    if (
        config.samples.oracle_surface_rollouts != 5_000
        or config.samples.oracle_rollouts != 50_000
    ):
        raise ValueError(
            "Phase 0C protocol requires 5,000 tuning and 50,000 evaluation rollouts"
        )


def _stream_ids(seed: int) -> tuple[tuple[int, int], ...]:
    streams = tuple(
        (
            _paper_seed(seed, 1_300_001 + scenario_index),
            _paper_seed(seed, 1_400_001 + scenario_index),
        )
        for scenario_index in range(len(_SCENARIOS))
    )
    flattened = tuple(value for pair in streams for value in pair)
    if len(set(flattened)) != len(flattened):
        raise RuntimeError("Phase 0C tuning/evaluation streams must be globally disjoint")
    return streams


def _checked_vector(values: Tensor | None, *, horizon: int, name: str) -> Tensor | None:
    if values is None:
        return None
    if values.ndim != 1 or len(values) != horizon:
        raise ValueError(f"{name} must have shape ({horizon},)")
    return values


def _padded_greedy_indices(
    selected_indices: tuple[int, ...], *, horizon: int
) -> tuple[int, ...]:
    if len(selected_indices) > horizon:
        raise ValueError("greedy selected indices exceed the horizon")
    return (*selected_indices, *((-1,) * (horizon - len(selected_indices))))


def _selection_start(
    *,
    name: str,
    selection: OracleScheduleResult,
    fallback_radii: Tensor,
    fallback_width: Tensor,
    stage_grid_indices: tuple[int | None, ...],
    horizon: int,
) -> SearchStart:
    radii = _checked_vector(selection.radii, horizon=horizon, name=f"{name} radii")
    coverage = _checked_vector(
        selection.tuning_coverage,
        horizon=horizon,
        name=f"{name} tuning coverage",
    )
    width = _checked_vector(
        selection.tuning_width,
        horizon=horizon,
        name=f"{name} tuning width",
    )
    if selection.selection_available:
        if radii is None or coverage is None or width is None:
            raise ValueError(f"available {name} selection has missing metrics")
        return SearchStart(
            name=name,
            radii=radii.clone(),
            stage_grid_indices=stage_grid_indices,
            coverage=coverage.clone(),
            normalized_width=width.clone(),
        )
    if any(value is not None for value in (selection.radii, coverage, width)):
        raise ValueError(f"unavailable {name} selection must not carry metrics")
    return SearchStart(
        name=name,
        radii=fallback_radii.clone(),
        stage_grid_indices=stage_grid_indices,
        coverage=fallback_radii.new_full((horizon,), -torch.inf),
        normalized_width=fallback_width.clone(),
    )


def prepare_phase0c_scenario_context(
    config: ExperimentConfig,
    *,
    seed: int,
    scenario: str,
    scenario_index: int | None = None,
    device: str,
    candidate_chunk_size: int = 16,
) -> Phase0CScenarioContext:
    """Rebuild one Phase 0A context and its three canonical Phase 0C starts."""

    if scenario not in _SCENARIOS:
        raise ValueError("scenario must be standard or tail_shift")
    expected_index = _SCENARIOS.index(scenario)
    if scenario_index is None:
        scenario_index = expected_index
    if scenario_index != expected_index:
        raise ValueError("scenario_index does not match scenario")
    if candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be positive")

    torch.manual_seed(seed)
    scenario_config = replace(config, synthetic=replace(config.synthetic, scenario=scenario))
    scenario_config.validate()
    oracle_context = _prepare_oracle_context(scenario_config, seed=seed, device=device)
    profile = oracle_context.schedule_family.profile
    profiled_scale_grid = oracle_context.schedule_family.scale_grid
    profiled_schedules = candidate_radius_schedules(profiled_scale_grid, profile)
    stage_grids = torch.stack(
        [
            fixed_q_grid(
                oracle_context.cot_scores[:, stage],
                size=config.q_grid_size,
                lower_quantile=config.q_quantile_min,
                upper_quantile=config.q_quantile_max,
            )
            for stage in range(config.horizon)
        ]
    )
    if stage_grids.shape != (config.horizon, config.q_grid_size):
        raise ValueError("stage grids do not match the configured horizon and size")

    tuning_seed = _paper_seed(seed, 1_300_001 + scenario_index)
    evaluation_seed = _paper_seed(seed, 1_400_001 + scenario_index)
    if tuning_seed == evaluation_seed:
        raise RuntimeError("Phase 0C tuning and evaluation streams must differ")
    tuning_noise = make_synthetic_noise_bundle(
        n=config.samples.oracle_surface_rollouts,
        horizon=config.horizon,
        seed=tuning_seed,
        device=device,
    )
    evaluation_noise = make_synthetic_noise_bundle(
        n=config.samples.oracle_rollouts,
        horizon=config.horizon,
        seed=evaluation_seed,
        device=device,
    )

    profiled_metrics = evaluate_profiled_candidates_crn(
        oracle_context.task.environment,
        oracle_context.policy,
        oracle_context.outcome_model,
        candidate_schedules=profiled_schedules,
        outcome_sd=oracle_context.outcome_sd,
        noise=tuning_noise,
        chunk_size=candidate_chunk_size,
    )
    profiled_selection = select_profiled_oracle_schedule(
        profiled_schedules,
        profiled_metrics,
        target=1.0 - config.certification.alpha,
    )
    greedy_selection = greedy_sequential_oracle_schedule(
        oracle_context.task.environment,
        oracle_context.policy,
        oracle_context.outcome_model,
        stage_grids=stage_grids,
        outcome_sd=oracle_context.outcome_sd,
        noise=tuning_noise,
        target=1.0 - config.certification.alpha,
        chunk_size=candidate_chunk_size,
    )

    upper_schedule = stage_grids[:, -1]
    upper_metrics = evaluate_profiled_candidates_crn(
        oracle_context.task.environment,
        oracle_context.policy,
        oracle_context.outcome_model,
        candidate_schedules=upper_schedule.unsqueeze(0),
        outcome_sd=oracle_context.outcome_sd,
        noise=tuning_noise,
        chunk_size=candidate_chunk_size,
    )
    if upper_metrics.coverage.shape != (1, config.horizon) or upper_metrics.normalized_width.shape != (
        1,
        config.horizon,
    ):
        raise ValueError("upper endpoint evaluation has an invalid shape")

    starts = (
        _selection_start(
            name="profiled",
            selection=profiled_selection,
            fallback_radii=upper_schedule,
            fallback_width=upper_metrics.normalized_width[0],
            stage_grid_indices=(None,) * config.horizon,
            horizon=config.horizon,
        ),
        _selection_start(
            name="greedy",
            selection=greedy_selection,
            fallback_radii=upper_schedule,
            fallback_width=upper_metrics.normalized_width[0],
            stage_grid_indices=_padded_greedy_indices(
                greedy_selection.selected_indices, horizon=config.horizon
            ),
            horizon=config.horizon,
        ),
        SearchStart(
            name="upper_endpoint",
            radii=upper_schedule.clone(),
            stage_grid_indices=(config.q_grid_size - 1,) * config.horizon,
            coverage=upper_metrics.coverage[0].clone(),
            normalized_width=upper_metrics.normalized_width[0].clone(),
        ),
    )
    return Phase0CScenarioContext(
        environment=oracle_context.task.environment,
        policy=oracle_context.policy,
        outcome_model=oracle_context.outcome_model,
        outcome_sd=oracle_context.outcome_sd,
        profile=profile,
        profiled_scale_grid=profiled_scale_grid,
        profiled_schedules=profiled_schedules,
        stage_grids=stage_grids,
        tuning_noise=tuning_noise,
        evaluation_noise=evaluation_noise,
        starts=starts,
        profiled_selection=profiled_selection,
        greedy_selection=greedy_selection,
    )


def _json_vector(values: Tensor | None) -> str:
    if values is None:
        return "[]"
    payload = [float(value) for value in values.detach().cpu().tolist()]
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


def _json_indices(values: tuple[int | None, ...]) -> str:
    return json.dumps(
        [-1 if value is None else int(value) for value in values],
        allow_nan=False,
        separators=(",", ":"),
    )


def _endpoint_count(indices: tuple[int | None, ...], *, grid_size: int) -> int:
    return sum(index in {0, grid_size - 1} for index in indices)


def _reference_payload(
    method_id: str,
    selection: OracleScheduleResult,
    *,
    horizon: int,
    grid_size: int,
) -> _MethodResult:
    status = "SELECTED" if selection.selection_available else "NO_FEASIBLE_START"
    if method_id == "current_profiled":
        indices: tuple[int | None, ...] = selection.selected_indices
        chosen = "profiled" if selection.selection_available else ""
        endpoint_count = horizon if selection.selected_endpoint else 0
    else:
        indices = _padded_greedy_indices(selection.selected_indices, horizon=horizon)
        chosen = "greedy" if selection.selection_available else ""
        endpoint_count = _endpoint_count(indices, grid_size=grid_size)
    return _MethodResult(
        method_id=method_id,
        status=status,
        radii=selection.radii,
        indices=indices,
        tuning_coverage=selection.tuning_coverage,
        tuning_width=selection.tuning_width,
        chosen_initialization=chosen,
        endpoint_count=endpoint_count,
    )


def _joint_payload(
    method_id: str,
    outcome: JointSearchOutcome,
    checkpoint: JointSearchCheckpoint | None,
) -> _MethodResult:
    if checkpoint is None:
        return _MethodResult(
            method_id=method_id,
            status=outcome.status,
            wall_time_seconds=outcome.elapsed_seconds,
        )
    best = checkpoint.best
    return _MethodResult(
        method_id=method_id,
        status="SELECTED",
        radii=best.radii,
        indices=best.stage_grid_indices,
        tuning_coverage=best.coverage,
        tuning_width=best.normalized_width,
        chosen_initialization=best.start_name,
        converged_at_pair=best.converged_at_pair,
        wall_time_seconds=outcome.elapsed_seconds,
        schedule_evaluations=checkpoint.schedule_evaluations,
        committed_updates=checkpoint.committed_updates,
    )


def _record(
    *,
    config: ExperimentConfig,
    seed: int,
    scenario: str,
    result: _MethodResult,
    evaluation: FrozenOracleEvaluation | None,
    tuning_stream_id: int,
    evaluation_stream_id: int,
) -> dict[str, object]:
    if result.status not in _VALID_STATUSES:
        raise ValueError(f"invalid Phase 0C selection status: {result.status}")
    budget_id, sweep_pairs = METHOD_METADATA.get(result.method_id, ("8SP", 8))
    available = result.status == "SELECTED" and result.radii is not None
    target = 1.0 - config.certification.alpha
    tuning_feasible = bool(
        available
        and result.tuning_coverage is not None
        and (result.tuning_coverage >= target).all().item()
    )
    missing = float("nan")
    return {
        "schema_version": _SCHEMA_VERSION,
        "seed": seed,
        "scenario": scenario,
        "method_id": result.method_id,
        "analysis_role": "reference" if result.method_id in PHASE0C_METHODS[:2] else "joint_search",
        "budget_id": budget_id,
        "sweep_pairs": sweep_pairs,
        "selection_status": result.status,
        "selection_available": available,
        "tuning_joint_feasible": tuning_feasible,
        "failure_reason": "" if available else result.status,
        "chosen_initialization": result.chosen_initialization,
        "selected_endpoint_stage_count": result.endpoint_count,
        "selected_stage_grid_indices_json": "[]" if not result.indices else _json_indices(result.indices),
        "q_by_time_json": _json_vector(result.radii if available else None),
        "tuning_coverage_json": _json_vector(result.tuning_coverage if available else None),
        "tuning_stage_width_json": _json_vector(result.tuning_width if available else None),
        "tuning_micro_width": (
            missing
            if not available or result.tuning_width is None
            else float(result.tuning_width.mean().item())
        ),
        "final_coverage_json": _json_vector(None if evaluation is None else evaluation.coverage),
        "final_wilson_lcb_json": _json_vector(
            None if evaluation is None else evaluation.wilson_lower_bound
        ),
        "final_stage_width_json": _json_vector(
            None if evaluation is None else evaluation.normalized_width
        ),
        "micro_normalized_width": missing if evaluation is None else evaluation.micro_normalized_width,
        "patient_normalized_width": missing if evaluation is None else evaluation.patient_normalized_width,
        "tuning_stream_id": tuning_stream_id,
        "evaluation_stream_id": evaluation_stream_id,
        "n_tuning_rollouts": config.samples.oracle_surface_rollouts,
        "n_evaluation_rollouts": config.samples.oracle_rollouts,
        "schedule_evaluations": result.schedule_evaluations,
        "committed_updates": result.committed_updates,
        "converged_at_pair": result.converged_at_pair,
        "wall_time_seconds": result.wall_time_seconds,
    }


def _surface_method(
    surfaces: dict[str, Tensor],
    *,
    prefix: str,
    result: _MethodResult,
    evaluation: FrozenOracleEvaluation | None,
    like: Tensor,
) -> None:
    empty = like.new_empty(0)
    surfaces.update({
        f"{prefix}_schedule": empty if result.radii is None else result.radii,
        f"{prefix}_tuning_coverage": empty if result.tuning_coverage is None else result.tuning_coverage,
        f"{prefix}_tuning_stage_width": empty if result.tuning_width is None else result.tuning_width,
        f"{prefix}_final_coverage": empty if evaluation is None else evaluation.coverage,
        f"{prefix}_final_wilson_lcb": empty if evaluation is None else evaluation.wilson_lower_bound,
        f"{prefix}_final_stage_width": empty if evaluation is None else evaluation.normalized_width,
    })


def _state_surfaces(
    surfaces: dict[str, Tensor],
    *,
    scenario: str,
    checkpoint: JointSearchCheckpoint | None,
) -> None:
    if checkpoint is None:
        return
    for state in checkpoint.per_start:
        prefix = f"{scenario}_pair4_{state.start_name}"
        indices = [-1 if value is None else value for value in state.stage_grid_indices]
        converged = -1 if state.converged_at_pair is None else state.converged_at_pair
        surfaces.update({
            f"{prefix}_radii": state.radii,
            f"{prefix}_stage_grid_indices": torch.tensor(indices, dtype=torch.int64),
            f"{prefix}_coverage": state.coverage,
            f"{prefix}_normalized_width": state.normalized_width,
            f"{prefix}_completed_sweep_pairs": torch.tensor(state.completed_sweep_pairs),
            f"{prefix}_converged_at_pair": torch.tensor(converged),
        })


def _trace_diagnostics(trace: tuple[CoordinateStep, ...]) -> list[dict[str, object]]:
    payload = []
    for step in trace:
        row = asdict(step)
        payload.append(
            {
                key: None if isinstance(value, float) and not math.isfinite(value) else value
                for key, value in row.items()
            }
        )
    return payload


def _checkpoint_diagnostics(
    checkpoint: JointSearchCheckpoint,
) -> dict[str, object]:
    return {
        "requested_sweep_pairs": checkpoint.requested_sweep_pairs,
        "executed_sweep_pairs": checkpoint.executed_sweep_pairs,
        "best_start_name": checkpoint.best.start_name,
        "schedule_evaluations": checkpoint.schedule_evaluations,
        "committed_updates": checkpoint.committed_updates,
        "trace": _trace_diagnostics(checkpoint.trace),
    }


def _evaluate_methods(
    context: Phase0CScenarioContext,
    schedules: dict[str, Tensor],
) -> dict[str, FrozenOracleEvaluation]:
    if not schedules:
        return {}
    return evaluate_frozen_schedules_crn(
        context.environment,
        context.policy,
        context.outcome_model,
        schedules=schedules,
        noise=context.evaluation_noise,
        outcome_sd=context.outcome_sd,
        forbidden_noise_seeds={context.tuning_noise.seed},
    )


def _remaining_seconds(started_at: float, cap: float) -> float:
    return cap - (time.monotonic() - started_at)


def _prepare_search_context(
    config: ExperimentConfig,
    *,
    seed: int,
    scenario: str,
    scenario_index: int,
    device: str,
    candidate_chunk_size: int,
    stream_ids: tuple[int, int],
) -> tuple[Phase0CScenarioContext, CRNCoordinateEvaluator]:
    context = prepare_phase0c_scenario_context(
        config,
        seed=seed,
        scenario=scenario,
        scenario_index=scenario_index,
        device=device,
        candidate_chunk_size=candidate_chunk_size,
    )
    if (context.tuning_noise.seed, context.evaluation_noise.seed) != stream_ids:
        raise RuntimeError("Phase 0C context stream identity changed")
    evaluator = CRNCoordinateEvaluator(
        context.environment,
        context.policy,
        context.outcome_model,
        starts=context.starts,
        outcome_sd=context.outcome_sd,
        noise=context.tuning_noise,
        chunk_size=candidate_chunk_size,
    )
    return context, evaluator


def _materialize_method(
    *,
    config: ExperimentConfig,
    seed: int,
    scenario: str,
    result: _MethodResult,
    evaluation: FrozenOracleEvaluation | None,
    tuning_stream_id: int,
    evaluation_stream_id: int,
    records: list[dict[str, object]],
    surfaces: dict[str, Tensor],
    like: Tensor,
) -> None:
    if result.indices and result.method_id.startswith("joint_"):
        result = replace(
            result,
            endpoint_count=_endpoint_count(
                result.indices,
                grid_size=config.q_grid_size,
            ),
        )
    records.append(
        _record(
            config=config,
            seed=seed,
            scenario=scenario,
            result=result,
            evaluation=evaluation,
            tuning_stream_id=tuning_stream_id,
            evaluation_stream_id=evaluation_stream_id,
        )
    )
    if result.status != "SELECTED":
        result = replace(
            result,
            radii=None,
            tuning_coverage=None,
            tuning_width=None,
        )
    _surface_method(
        surfaces,
        prefix=f"{scenario}_{result.method_id}",
        result=result,
        evaluation=evaluation,
        like=like,
    )


def run_phase0c_seed(
    config: ExperimentConfig,
    *,
    seed: int,
    device: str,
    candidate_chunk_size: int = 16,
    sweep_pair_checkpoints: tuple[int, ...] = (2, 4),
    max_seed_wall_seconds: float,
) -> SeedResult:
    """Run both scenarios and the exact reference/B/2B paired schedules."""

    _validate_common_inputs(
        config,
        candidate_chunk_size=candidate_chunk_size,
        max_seed_wall_seconds=max_seed_wall_seconds,
    )
    if sweep_pair_checkpoints != (2, 4):
        raise ValueError("sweep-pair checkpoints must be exactly (2, 4)")
    streams = _stream_ids(seed)
    started_at = time.monotonic()
    records: list[dict[str, object]] = []
    surfaces: dict[str, Tensor] = {}
    diagnostics: dict[str, object] = {}

    for scenario_index, scenario in enumerate(_SCENARIOS):
        tuning_stream_id, evaluation_stream_id = streams[scenario_index]
        context, evaluator = _prepare_search_context(
            config,
            seed=seed,
            scenario=scenario,
            scenario_index=scenario_index,
            device=device,
            candidate_chunk_size=candidate_chunk_size,
            stream_ids=(tuning_stream_id, evaluation_stream_id),
        )
        remaining = _remaining_seconds(started_at, max_seed_wall_seconds)
        if remaining <= 0.0:
            outcome = JointSearchOutcome(
                status="WALL_TIME_CAP",
                checkpoints={},
                elapsed_seconds=time.monotonic() - started_at,
            )
        else:
            outcome = cyclic_joint_coordinate_search(
                context.starts,
                context.stage_grids,
                evaluator,
                target=1.0 - config.certification.alpha,
                sweep_pair_checkpoints=sweep_pair_checkpoints,
                max_wall_seconds=remaining,
            )
        if outcome.status not in _VALID_STATUSES:
            raise RuntimeError("joint search returned an invalid status")

        payloads = {
            "current_profiled": _reference_payload(
                "current_profiled",
                context.profiled_selection,
                horizon=config.horizon,
                grid_size=config.q_grid_size,
            ),
            "greedy": _reference_payload(
                "greedy",
                context.greedy_selection,
                horizon=config.horizon,
                grid_size=config.q_grid_size,
            ),
            "joint_B": _joint_payload(
                "joint_B", outcome, outcome.checkpoints.get(2)
            ),
            "joint_2B": _joint_payload(
                "joint_2B", outcome, outcome.checkpoints.get(4)
            ),
        }
        schedules = {
            method_id: result.radii
            for method_id, result in payloads.items()
            if result.status == "SELECTED" and result.radii is not None
        }
        evaluations = _evaluate_methods(context, schedules)  # one shared fresh bundle
        if set(evaluations) != set(schedules):
            raise RuntimeError("fresh evaluation did not return every frozen schedule")

        surfaces.update(
            {
                f"{scenario}_profile": context.profile,
                f"{scenario}_profiled_scale_grid": context.profiled_scale_grid,
                f"{scenario}_profiled_schedules": context.profiled_schedules,
                f"{scenario}_stage_grids": context.stage_grids,
            }
        )
        for method_id in PHASE0C_METHODS:
            _materialize_method(
                config=config,
                seed=seed,
                scenario=scenario,
                result=payloads[method_id],
                evaluation=evaluations.get(method_id),
                tuning_stream_id=tuning_stream_id,
                evaluation_stream_id=evaluation_stream_id,
                records=records,
                surfaces=surfaces,
                like=context.profile,
            )
        _state_surfaces(
            surfaces,
            scenario=scenario,
            checkpoint=outcome.checkpoints.get(4),
        )
        diagnostics[scenario] = {
            "tuning_stream_id": tuning_stream_id,
            "evaluation_stream_id": evaluation_stream_id,
            "start_order": [start.name for start in context.starts],
            "search_status": outcome.status,
            "checkpoints": {
                str(pair): _checkpoint_diagnostics(checkpoint)
                for pair, checkpoint in sorted(outcome.checkpoints.items())
            },
        }

    json.dumps(diagnostics, allow_nan=False)
    return SeedResult(
        seed=seed,
        device=device,
        records=records,
        surfaces=surfaces,
        diagnostics=diagnostics,
    )


def _states_equal(left: SearchState, right: SearchState) -> bool:
    return (
        left.start_name == right.start_name
        and torch.equal(left.radii, right.radii)
        and left.stage_grid_indices == right.stage_grid_indices
        and torch.equal(left.coverage, right.coverage)
        and torch.equal(left.normalized_width, right.normalized_width)
        and left.completed_sweep_pairs == right.completed_sweep_pairs
        and left.converged_at_pair == right.converged_at_pair
    )


def _validate_parent_states(
    supplied: tuple[SearchState, ...],
    expected: tuple[SearchState, ...],
) -> None:
    if len(supplied) != 3 or len(expected) != 3:
        raise ValueError("parent pair-4 state must contain all three starts")
    if any(state.completed_sweep_pairs != 4 for state in supplied):
        raise ValueError("parent pair-4 state has an invalid completed-pair count")
    if any(not _states_equal(left, right) for left, right in zip(supplied, expected, strict=True)):
        raise ValueError("parent pair-4 state does not match the reconstructed state")


def run_phase0c_extension_seed(
    config: ExperimentConfig,
    *,
    seed: int,
    device: str,
    pair4_states: dict[str, tuple[SearchState, ...]],
    candidate_chunk_size: int = 16,
    max_seed_wall_seconds: float,
) -> SeedResult:
    """Validate pair-four parents and continue both scenarios through 8SP."""

    _validate_common_inputs(
        config,
        candidate_chunk_size=candidate_chunk_size,
        max_seed_wall_seconds=max_seed_wall_seconds,
    )
    if set(pair4_states) != set(_SCENARIOS):
        raise ValueError("parent pair-4 states must contain both scenarios")
    streams = _stream_ids(seed)
    started_at = time.monotonic()
    records: list[dict[str, object]] = []
    surfaces: dict[str, Tensor] = {}
    diagnostics: dict[str, object] = {}

    for scenario_index, scenario in enumerate(_SCENARIOS):
        tuning_stream_id, evaluation_stream_id = streams[scenario_index]
        context, evaluator = _prepare_search_context(
            config,
            seed=seed,
            scenario=scenario,
            scenario_index=scenario_index,
            device=device,
            candidate_chunk_size=candidate_chunk_size,
            stream_ids=(tuning_stream_id, evaluation_stream_id),
        )
        remaining = _remaining_seconds(started_at, max_seed_wall_seconds)
        if remaining <= 0.0:
            raise ValueError("seed wall-time cap expired before parent validation")
        parent_outcome = cyclic_joint_coordinate_search(
            context.starts,
            context.stage_grids,
            evaluator,
            target=1.0 - config.certification.alpha,
            sweep_pair_checkpoints=(2, 4),
            max_wall_seconds=remaining,
        )
        parent_checkpoint = parent_outcome.checkpoints.get(4)
        if parent_checkpoint is None:
            raise ValueError("parent pair-4 state cannot be reconstructed")
        _validate_parent_states(pair4_states[scenario], parent_checkpoint.per_start)

        remaining = _remaining_seconds(started_at, max_seed_wall_seconds)
        if remaining <= 0.0:
            outcome = JointSearchOutcome(
                status="WALL_TIME_CAP",
                checkpoints={},
                elapsed_seconds=time.monotonic() - started_at,
            )
        else:
            outcome = resume_cyclic_joint_coordinate_search(
                parent_checkpoint.per_start,
                context.stage_grids,
                evaluator,
                target=1.0 - config.certification.alpha,
                max_wall_seconds=remaining,
            )
        checkpoint = outcome.checkpoints.get(8)
        result = _joint_payload("joint_8SP", outcome, checkpoint)
        schedules = (
            {"joint_8SP": result.radii}
            if result.status == "SELECTED" and result.radii is not None
            else {}
        )
        evaluations = _evaluate_methods(context, schedules)
        evaluation = evaluations.get("joint_8SP")
        surfaces[f"{scenario}_stage_grids"] = context.stage_grids
        _materialize_method(
            config=config,
            seed=seed,
            scenario=scenario,
            result=result,
            evaluation=evaluation,
            tuning_stream_id=tuning_stream_id,
            evaluation_stream_id=evaluation_stream_id,
            records=records,
            surfaces=surfaces,
            like=context.profile,
        )
        diagnostics[scenario] = {
            "tuning_stream_id": tuning_stream_id,
            "evaluation_stream_id": evaluation_stream_id,
            "search_status": outcome.status,
            "checkpoint": (
                None if checkpoint is None else _checkpoint_diagnostics(checkpoint)
            ),
        }

    json.dumps(diagnostics, allow_nan=False)
    return SeedResult(
        seed=seed,
        device=device,
        records=records,
        surfaces=surfaces,
        diagnostics=diagnostics,
    )
