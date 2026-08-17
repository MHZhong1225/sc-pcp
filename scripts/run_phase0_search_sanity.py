"""Run the analytic reduced-grid finite-MDP Phase 0 sanity diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import experiment_tree_sha256, source_tree_sha256
from scpcp.config import ExperimentConfig
from scpcp.coverage import fixed_q_grid
from scpcp.device import resolve_device
from scpcp.experiment import _prepare_oracle_context
from scpcp.phase0_search import (
    AnalyticFiniteMDP,
    ScheduleEvaluation,
    SearchDiagnostic,
    exact_schedule_search,
)


DEFAULT_CONFIG = ROOT / "configs" / "per_step_tabular_validation.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "work"
    / "phase0a_finite_mdp_sanity"
    / "finite_mdp_sanity.json"
)
FROZEN_SEED = 0
FROZEN_HORIZON = 4
FROZEN_GRID_SIZE = 5
TARGET_COVERAGE = 0.90


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the analytic exact reduced-grid finite-MDP sanity check"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def reduced_search_config(
    base: ExperimentConfig,
    *,
    device: str,
    output: Path,
) -> ExperimentConfig:
    """Apply only the pre-registered reduced-grid diagnostic overrides."""

    if base.data.dataset != "tabular":
        raise ValueError("finite-MDP sanity requires data.dataset='tabular'")
    config = replace(
        base,
        horizon=FROZEN_HORIZON,
        q_grid_size=FROZEN_GRID_SIZE,
        seeds=(FROZEN_SEED,),
        devices=(device,),
        output_dir=output.parent,
    )
    config.validate()
    return config


@torch.no_grad()
def build_search_problem(
    context: object,
    *,
    horizon: int,
    grid_size: int,
    lower_quantile: float,
    upper_quantile: float,
) -> AnalyticFiniteMDP:
    """Freeze neural outputs, then construct the float64 analytic recursion."""

    environment = context.task.environment
    required = (
        "initial_state_probabilities",
        "transition_probabilities",
        "outcome_distribution_parameters",
        "n_states",
        "n_actions",
    )
    if not all(hasattr(environment, name) for name in required):
        raise TypeError("search sanity requires the existing finite-MDP environment")
    scores = context.cot_scores
    if scores.ndim != 2 or scores.shape[1] != horizon:
        raise ValueError("D_COT scores must have shape [N,T]")
    stage_grids = torch.stack(
        [
            fixed_q_grid(
                scores[:, stage],
                size=grid_size,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
            for stage in range(horizon)
        ]
    )

    predictor = context.outcome_model
    try:
        parameter = next(predictor.parameters())
    except (AttributeError, StopIteration) as error:
        raise TypeError("the frozen predictor must expose model parameters") from error
    states = torch.eye(
        environment.n_states,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    predictor_means, predictor_scales = predictor.predict_all_actions(states)
    action_probabilities = torch.stack(
        [
            context.policy.probabilities_for_grid(
                states,
                stage_grid.to(states),
            ).permute(1, 0, 2)
            for stage_grid in stage_grids
        ]
    )
    analytic_dtype = torch.float64
    device = states.device
    outcome_means, outcome_standard_deviations = (
        environment.outcome_distribution_parameters(device, analytic_dtype)
    )
    return AnalyticFiniteMDP(
        initial_state_probabilities=environment.initial_state_probabilities(
            device,
            analytic_dtype,
        ),
        transition_probabilities=environment.transition_probabilities(
            device,
            analytic_dtype,
        ),
        action_probabilities=action_probabilities.to(dtype=analytic_dtype),
        radii=stage_grids.to(device=device, dtype=analytic_dtype),
        predictor_means=predictor_means.to(dtype=analytic_dtype),
        predictor_scales=predictor_scales.to(dtype=analytic_dtype),
        outcome_means=outcome_means,
        outcome_standard_deviations=outcome_standard_deviations,
        outcome_normalization=context.outcome_sd.to(
            device=device,
            dtype=analytic_dtype,
        ),
    )


def build_sanity_payload(
    problem: AnalyticFiniteMDP,
    diagnostic: SearchDiagnostic,
    *,
    seed: int,
    device: str,
    source_hash: str,
    experiment_hash: str,
    config_hash: str,
    target: float,
) -> dict[str, Any]:
    """Build the locked, non-gating exact finite-grid JSON payload."""

    horizon, grid_size = problem.radii.shape
    if diagnostic.search_type != "exact":
        raise ValueError("sanity payload requires exact enumeration")
    if (horizon, grid_size) != (FROZEN_HORIZON, FROZEN_GRID_SIZE):
        raise ValueError("sanity payload requires the frozen T=4, K=5 grid")
    for name, value in (
        ("source_tree_sha256", source_hash),
        ("experiment_tree_sha256", experiment_hash),
        ("config_sha256", config_hash),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    exact = _schedule_payload(problem, diagnostic.best_found_schedule)
    if diagnostic.greedy_available:
        if (
            diagnostic.greedy_schedule is None
            or diagnostic.true_optimality_gap is None
        ):
            raise ValueError("available greedy search must include its schedule and gap")
        greedy = _schedule_payload(problem, diagnostic.greedy_schedule)
        gap: float | None = float(diagnostic.true_optimality_gap)
        if not math.isfinite(gap) or gap < -1e-12:
            raise ValueError("true finite-grid gap must be finite and nonnegative")
        if not math.isclose(
            gap,
            greedy["mean_normalized_width"] - exact["mean_normalized_width"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("true finite-grid gap does not match the schedule widths")
    else:
        if any(
            value is not None
            for value in (
                diagnostic.greedy_schedule,
                diagnostic.greedy_width,
                diagnostic.true_optimality_gap,
                diagnostic.best_found_gap,
            )
        ):
            raise ValueError("unavailable greedy search must use null schedule and gaps")
        greedy = None
        gap = None
    return {
        "schema_version": 1,
        "diagnostic_type": "analytic_exact_finite_grid_search",
        "status": "complete",
        "non_gating": True,
        "population_exact": True,
        "scope": (
            "The true gap is relative only to the reduced frozen grid and "
            "the frozen predictor and policy."
        ),
        "dataset": "tabular",
        "seed": seed,
        "device": device,
        "horizon": horizon,
        "grid_size": grid_size,
        "schedule_count": grid_size**horizon,
        "target_coverage": target,
        "grid_source": "D_COT stagewise score quantiles frozen before search",
        "gap_definition": (
            "greedy_mean_stage_width_minus_exact_mean_stage_width"
        ),
        "source_tree_sha256": source_hash,
        "experiment_tree_sha256": experiment_hash,
        "config_sha256": config_hash,
        "greedy_available": diagnostic.greedy_available,
        "greedy": greedy,
        "exact": exact,
        "true_finite_grid_gap": gap,
    }


def _schedule_payload(
    problem: AnalyticFiniteMDP,
    evaluation: ScheduleEvaluation,
) -> dict[str, Any]:
    horizon = problem.radii.shape[0]
    if len(evaluation.selected_indices) != horizon:
        raise ValueError("sanity schedules must span all four stages")
    selected_radii = [
        float(problem.radii[stage, index].item())
        for stage, index in enumerate(evaluation.selected_indices)
    ]
    coverage = [float(value) for value in evaluation.coverage.tolist()]
    stage_width = [float(value) for value in evaluation.normalized_width.tolist()]
    mean_width = float(evaluation.normalized_width.mean().item())
    if not all(math.isfinite(value) and value > 0.0 for value in selected_radii):
        raise ValueError("selected radii must be finite and positive")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coverage):
        raise ValueError("coverage values must be finite probabilities")
    if not all(math.isfinite(value) and value > 0.0 for value in stage_width):
        raise ValueError("normalized widths must be finite and positive")
    if not math.isfinite(mean_width) or not math.isclose(
        mean_width,
        sum(stage_width) / horizon,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("mean normalized width must match the stage widths")
    return {
        "selected_indices": list(evaluation.selected_indices),
        "selected_radii": selected_radii,
        "coverage": coverage,
        "normalized_width_by_stage": stage_width,
        "mean_normalized_width": mean_width,
    }


def canonical_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_sanity_json(destination: Path, payload: dict[str, Any]) -> None:
    """Atomically publish strict RFC-compatible JSON."""

    content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}-",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"finite-MDP sanity output already exists: {args.output}"
        )
    device = str(resolve_device(args.device))
    base = ExperimentConfig.from_yaml(args.config)
    config = reduced_search_config(base, device=device, output=args.output)
    torch.manual_seed(FROZEN_SEED)
    context = _prepare_oracle_context(config, seed=FROZEN_SEED, device=device)
    problem = build_search_problem(
        context,
        horizon=FROZEN_HORIZON,
        grid_size=FROZEN_GRID_SIZE,
        lower_quantile=config.q_quantile_min,
        upper_quantile=config.q_quantile_max,
    )
    diagnostic = exact_schedule_search(problem, target=TARGET_COVERAGE)
    payload = build_sanity_payload(
        problem,
        diagnostic,
        seed=FROZEN_SEED,
        device=device,
        source_hash=source_tree_sha256(),
        experiment_hash=experiment_tree_sha256(),
        config_hash=canonical_config_sha256(config.to_dict()),
        target=TARGET_COVERAGE,
    )
    write_sanity_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
