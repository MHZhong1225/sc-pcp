"""Per-seed orchestration for the Phase 0C joint-search audit."""

from __future__ import annotations

import hashlib
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
_START_NAMES = ("profiled", "greedy", "upper_endpoint")
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
    if type(candidate_chunk_size) is not int or candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be a positive non-bool integer")
    if (
        isinstance(max_seed_wall_seconds, bool)
        or not isinstance(max_seed_wall_seconds, (int, float))
        or not math.isfinite(max_seed_wall_seconds)
        or max_seed_wall_seconds <= 0.0
    ):
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
    if type(candidate_chunk_size) is not int or candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be a positive non-bool integer")

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
        "selected_stage_grid_indices_json": (
            _json_indices(result.indices) if available and result.indices else "[]"
        ),
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
        "n_evaluation_rollouts": (
            config.samples.oracle_rollouts if evaluation is not None else 0
        ),
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
    if (
        type(sweep_pair_checkpoints) is not tuple
        or any(type(value) is not int for value in sweep_pair_checkpoints)
        or sweep_pair_checkpoints != (2, 4)
    ):
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
        checkpoint4 = outcome.checkpoints.get(4)
        active_start_names = (
            tuple(state.start_name for state in checkpoint4.per_start)
            if checkpoint4 is not None
            else tuple(
                start.name
                for start in context.starts
                if bool((start.coverage >= 1.0 - config.certification.alpha).all().item())
            )
        )
        extension_eligible = checkpoint4 is not None and active_start_names == _START_NAMES
        surfaces[f"{scenario}_active_start_names"] = torch.tensor(
            [_START_NAMES.index(name) for name in active_start_names],
            dtype=torch.int64,
        )
        surfaces[f"{scenario}_extension_eligible"] = torch.tensor(
            extension_eligible,
            dtype=torch.bool,
        )
        diagnostics[scenario] = {
            "tuning_stream_id": tuning_stream_id,
            "evaluation_stream_id": evaluation_stream_id,
            "start_order": [start.name for start in context.starts],
            "active_start_names": list(active_start_names),
            "extension_eligible": extension_eligible,
            "pair4_state_sha256": (
                []
                if checkpoint4 is None
                else [_state_sha256(state) for state in checkpoint4.per_start]
            ),
            "greedy_partial_indices": (
                []
                if context.greedy_selection.selection_available
                else list(context.greedy_selection.selected_indices)
            ),
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


def _tensor_fingerprint(tensor: Tensor) -> tuple[tuple[int, ...], torch.dtype, bytes]:
    canonical = tensor.detach().cpu().contiguous()
    raw_bytes = canonical.reshape(-1).view(torch.uint8).numpy().tobytes()
    return tuple(canonical.shape), canonical.dtype, raw_bytes


def _state_sha256(state: SearchState) -> str:
    parts = [b"phase0c-search-state-v1", state.start_name.encode("utf-8")]
    for tensor in (state.radii, state.coverage, state.normalized_width):
        shape, dtype, raw_bytes = _tensor_fingerprint(tensor)
        parts.extend(
            (
                json.dumps(list(shape), separators=(",", ":")).encode(),
                str(dtype).encode("ascii"),
                raw_bytes,
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


def _preflight_parent_states(
    pair4_states: object,
    *,
    horizon: int,
    grid_size: int,
) -> dict[str, tuple[SearchState, ...]]:
    if not isinstance(pair4_states, dict) or set(pair4_states) != set(_SCENARIOS):
        raise ValueError("parent pair-4 states must contain both scenarios")
    for scenario in _SCENARIOS:
        states = pair4_states[scenario]
        if not isinstance(states, tuple) or len(states) != len(_START_NAMES):
            raise TypeError("parent pair-4 states must be a three-state tuple")
        if any(not isinstance(state, SearchState) for state in states):
            raise TypeError("parent pair-4 tuple must contain SearchState values")
        if tuple(state.start_name for state in states) != _START_NAMES:
            raise ValueError("parent pair-4 states must have canonical unique names")
        for state in states:
            vectors = (state.radii, state.coverage, state.normalized_width)
            if any(
                not isinstance(vector, Tensor)
                or vector.ndim != 1
                or len(vector) != horizon
                or not vector.is_floating_point()
                or not bool(torch.isfinite(vector).all().item())
                for vector in vectors
            ):
                raise ValueError("parent pair-4 tensors must be finite horizon vectors")
            if not isinstance(state.stage_grid_indices, tuple) or len(
                state.stage_grid_indices
            ) != horizon:
                raise TypeError("parent pair-4 indices must be a horizon tuple")
            if any(
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value < grid_size
                )
                for value in state.stage_grid_indices
            ):
                raise ValueError("parent pair-4 indices must be integer grid positions or None")
            if (
                isinstance(state.completed_sweep_pairs, bool)
                or not isinstance(state.completed_sweep_pairs, int)
                or state.completed_sweep_pairs != 4
            ):
                raise ValueError("parent pair-4 state must have completed pair 4")
            converged = state.converged_at_pair
            if converged is not None and (
                isinstance(converged, bool)
                or not isinstance(converged, int)
                or not 1 <= converged <= 4
            ):
                raise ValueError("parent pair-4 convergence metadata is invalid")
        convergence = tuple(state.converged_at_pair for state in states)
        if any(value is not None for value in convergence) and len(set(convergence)) != 1:
            raise ValueError("parent pair-4 convergence metadata must be consistent")
    return pair4_states


def _preflight_extension_eligibility(extension_eligible: object) -> None:
    if not isinstance(extension_eligible, dict) or set(extension_eligible) != set(
        _SCENARIOS
    ):
        raise ValueError("extension_eligible must contain both scenarios")
    if any(type(extension_eligible[scenario]) is not bool for scenario in _SCENARIOS):
        raise TypeError("extension_eligible values must be bool")
    if any(not extension_eligible[scenario] for scenario in _SCENARIOS):
        raise ValueError("extension_eligible must be true for both scenarios")


def _preflight_parent_hashes(
    parents: dict[str, tuple[SearchState, ...]],
    pair4_state_sha256: object,
) -> None:
    if not isinstance(pair4_state_sha256, dict) or set(pair4_state_sha256) != set(
        _SCENARIOS
    ):
        raise ValueError("pair4_state_sha256 must contain both scenarios")
    for scenario in _SCENARIOS:
        hashes = pair4_state_sha256[scenario]
        if not isinstance(hashes, tuple) or len(hashes) != len(_START_NAMES):
            raise ValueError("pair4_state_sha256 values must be canonical triples")
        if any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("pair4_state_sha256 values must be lowercase SHA256 hex")
        expected = tuple(_state_sha256(state) for state in parents[scenario])
        if hashes != expected:
            raise ValueError("pair4_state_sha256 SHA256 mismatch for parent pair-4 state")


def _validate_parent_against_context(
    context: Phase0CScenarioContext,
    states: tuple[SearchState, ...],
    *,
    candidate_chunk_size: int,
) -> None:
    profiled_radii = context.profiled_selection.radii
    for state in states:
        for stage, index in enumerate(state.stage_grid_indices):
            if index is None:
                if state.start_name != "profiled" or profiled_radii is None:
                    raise ValueError("parent pair-4 off-grid index is not canonical")
                expected_radius = profiled_radii[stage]
            else:
                expected_radius = context.stage_grids[stage, index]
            if _tensor_fingerprint(state.radii[stage]) != _tensor_fingerprint(
                expected_radius
            ):
                raise ValueError("parent pair-4 indices and radii do not match")

    metrics = evaluate_profiled_candidates_crn(
        context.environment,
        context.policy,
        context.outcome_model,
        candidate_schedules=torch.stack([state.radii for state in states]),
        outcome_sd=context.outcome_sd,
        noise=context.tuning_noise,
        chunk_size=candidate_chunk_size,
    )
    for row, state in enumerate(states):
        if (
            _tensor_fingerprint(state.coverage)
            != _tensor_fingerprint(metrics.coverage[row])
            or _tensor_fingerprint(state.normalized_width)
            != _tensor_fingerprint(metrics.normalized_width[row])
        ):
            raise ValueError("parent pair-4 metrics do not match tuning replay")


def _resume_evaluator(
    context: Phase0CScenarioContext,
    states: tuple[SearchState, ...],
    *,
    candidate_chunk_size: int,
) -> CRNCoordinateEvaluator:
    starts = tuple(
        SearchStart(
            name=state.start_name,
            radii=state.radii,
            stage_grid_indices=state.stage_grid_indices,
            coverage=state.coverage,
            normalized_width=state.normalized_width,
        )
        for state in states
    )
    return CRNCoordinateEvaluator(
        context.environment,
        context.policy,
        context.outcome_model,
        starts=starts,
        outcome_sd=context.outcome_sd,
        noise=context.tuning_noise,
        chunk_size=candidate_chunk_size,
    )


def run_phase0c_extension_seed(
    config: ExperimentConfig,
    *,
    seed: int,
    device: str,
    pair4_states: dict[str, tuple[SearchState, ...]],
    pair4_state_sha256: dict[str, tuple[str, str, str]],
    extension_eligible: dict[str, bool],
    candidate_chunk_size: int = 16,
    max_seed_wall_seconds: float,
) -> SeedResult:
    """Validate pair-four parents and continue both scenarios through 8SP."""

    started_at = time.monotonic()
    _validate_common_inputs(
        config,
        candidate_chunk_size=candidate_chunk_size,
        max_seed_wall_seconds=max_seed_wall_seconds,
    )
    _preflight_extension_eligibility(extension_eligible)
    parents = _preflight_parent_states(
        pair4_states,
        horizon=config.horizon,
        grid_size=config.q_grid_size,
    )
    _preflight_parent_hashes(parents, pair4_state_sha256)
    streams = _stream_ids(seed)

    contexts: dict[str, Phase0CScenarioContext] = {}
    wall_time_phase: str | None = None
    for scenario_index, scenario in enumerate(_SCENARIOS):
        tuning_stream_id, evaluation_stream_id = streams[scenario_index]
        if _remaining_seconds(started_at, max_seed_wall_seconds) <= 0.0:
            wall_time_phase = "parent_validation"
            break
        context = prepare_phase0c_scenario_context(
            config,
            seed=seed,
            scenario=scenario,
            scenario_index=scenario_index,
            device=device,
            candidate_chunk_size=candidate_chunk_size,
        )
        if (context.tuning_noise.seed, context.evaluation_noise.seed) != (
            tuning_stream_id,
            evaluation_stream_id,
        ):
            raise RuntimeError("Phase 0C context stream identity changed")
        contexts[scenario] = context
        if _remaining_seconds(started_at, max_seed_wall_seconds) <= 0.0:
            wall_time_phase = "parent_validation"
            break
        _validate_parent_against_context(
            context,
            parents[scenario],
            candidate_chunk_size=candidate_chunk_size,
        )
        if _remaining_seconds(started_at, max_seed_wall_seconds) <= 0.0:
            wall_time_phase = "parent_validation"
            break

    outcomes: dict[str, JointSearchOutcome] = {}
    results: dict[str, _MethodResult] = {}
    if wall_time_phase is None:
        for scenario in _SCENARIOS:
            if _remaining_seconds(started_at, max_seed_wall_seconds) <= 0.0:
                wall_time_phase = f"{scenario}_cache"
                break
            context = contexts[scenario]
            evaluator = _resume_evaluator(
                context,
                parents[scenario],
                candidate_chunk_size=candidate_chunk_size,
            )
            remaining = _remaining_seconds(started_at, max_seed_wall_seconds)
            if remaining <= 0.0:
                wall_time_phase = f"{scenario}_cache"
                break
            outcome = resume_cyclic_joint_coordinate_search(
                parents[scenario],
                context.stage_grids,
                evaluator,
                target=1.0 - config.certification.alpha,
                max_wall_seconds=remaining,
            )
            checkpoint = outcome.checkpoints.get(8)
            outcomes[scenario] = outcome
            if (
                outcome.status == "WALL_TIME_CAP"
                or _remaining_seconds(started_at, max_seed_wall_seconds) <= 0.0
            ):
                wall_time_phase = f"{scenario}_continuation"
                break
            if outcome.status != "SELECTED" or checkpoint is None:
                raise RuntimeError("extension continuation did not complete pair 8")
            results[scenario] = _joint_payload("joint_8SP", outcome, checkpoint)

    evaluations_by_scenario: dict[str, FrozenOracleEvaluation | None] = {}
    fresh_completed: set[str] = set()
    if wall_time_phase is None:
        for scenario in _SCENARIOS:
            if _remaining_seconds(started_at, max_seed_wall_seconds) <= 0.0:
                wall_time_phase = "before_fresh"
                break
            result = results[scenario]
            if result.radii is None:
                raise RuntimeError("selected pair-8 result has no schedule")
            schedules = {"joint_8SP": result.radii}
            evaluations = _evaluate_methods(contexts[scenario], schedules)
            if set(evaluations) != set(schedules):
                raise RuntimeError("fresh evaluation did not return every frozen schedule")
            evaluations_by_scenario[scenario] = evaluations["joint_8SP"]
            fresh_completed.add(scenario)
            if _remaining_seconds(started_at, max_seed_wall_seconds) <= 0.0:
                wall_time_phase = f"{scenario}_fresh"
                break

    if wall_time_phase is not None:
        elapsed = time.monotonic() - started_at
        results = {
            scenario: _MethodResult(
                method_id="joint_8SP",
                status="WALL_TIME_CAP",
                wall_time_seconds=elapsed,
            )
            for scenario in _SCENARIOS
        }
        evaluations_by_scenario = {scenario: None for scenario in _SCENARIOS}

    records: list[dict[str, object]] = []
    surfaces: dict[str, Tensor] = {}
    diagnostics: dict[str, object] = {}
    for scenario_index, scenario in enumerate(_SCENARIOS):
        context = contexts.get(scenario)
        outcome = outcomes.get(scenario)
        result = results[scenario]
        evaluation = evaluations_by_scenario[scenario]
        tuning_stream_id, evaluation_stream_id = streams[scenario_index]
        checkpoint = None if outcome is None else outcome.checkpoints.get(8)
        if context is not None:
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
            like=torch.empty(0) if context is None else context.profile,
        )
        diagnostics[scenario] = {
            "tuning_stream_id": tuning_stream_id,
            "evaluation_stream_id": evaluation_stream_id,
            "search_status": (
                "WALL_TIME_CAP" if wall_time_phase is not None else outcome.status
            ),
            "continuation_status": None if outcome is None else outcome.status,
            "fresh_evaluation_completed": scenario in fresh_completed,
            "wall_time_phase": wall_time_phase,
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
