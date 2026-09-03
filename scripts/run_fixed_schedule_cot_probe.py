"""Run the isolated fixed-schedule COT score-law recovery development probe.

This script does not invoke the canonical paper SC-PCP selector.  It uses a
controlled semi-synthetic residual environment only to compare, for a frozen
Standard-CP schedule, direct prefix importance weighting and fixed-schedule
occupancy transport against a fresh target-policy score-law reference.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import source_tree_sha256  # noqa: E402
from scpcp.baselines import standard_cp_stagewise_radii  # noqa: E402
from scpcp.config import ExperimentConfig  # noqa: E402
from scpcp.controlled_policy import ControlledMixturePolicy  # noqa: E402
from scpcp.controlled_transition import (  # noqa: E402
    ControlledResidualEnvironment,
    make_controlled_noise,
    rollout_controlled,
)
from scpcp.experiment import _paper_seed, _prepare_experiment_context  # noqa: E402
from scpcp.fixed_schedule_cot import (  # noqa: E402
    fit_fixed_schedule_cot,
    fixed_schedule_state_action_weights,
)
from scpcp.policy.anchored import BehaviorAnchoredPolicy  # noqa: E402
from scpcp.scores import score_batch  # noqa: E402


PROTOCOL = "fixed_schedule_cot_score_recovery"
DEVELOPMENT_SEEDS = (12100, 12102, 12104, 12106, 12108)
# These seeds are disjoint from the five-seed development screen.  The learner,
# score law, gamma grid, and sample sizes are deliberately identical: this is a
# replication of score-law recovery, not a new tuning opportunity.
REPLICATION_SEEDS = tuple(range(12200, 12240, 2))
GAMMAS = (0.0, -2.0, -3.0, -4.0)
CALIBRATION_TRAJECTORIES = 3_000
FIT_TRAJECTORIES = 3_000
REFERENCE_TRAJECTORIES = 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("development", "replication20"), default="development")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    seeds = DEVELOPMENT_SEEDS if args.study == "development" else REPLICATION_SEEDS
    default_name = "fixed_schedule_cot_probe_dev5_20260824" if args.study == "development" else "fixed_schedule_cot_probe_replication20_20260824"
    output_dir = args.output_dir or ROOT / "results" / "work" / default_name
    run_probe(output_dir.resolve(), device=args.device, seeds=seeds, role=args.study)
    print(output_dir.resolve())


def run_probe(
    output_dir: Path,
    *,
    device: str,
    seeds: tuple[int, ...] = DEVELOPMENT_SEEDS,
    role: str = "development",
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"development output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata = {
        "protocol": PROTOCOL,
        "role": role,
        "base_config": "configs/per_step_mimic_iv.yaml",
        "seeds": seeds,
        "gammas": GAMMAS,
        "calibration_trajectories": CALIBRATION_TRAJECTORIES,
        "fit_trajectories": FIT_TRAJECTORIES,
        "reference_trajectories": REFERENCE_TRAJECTORIES,
        "source_tree_sha256": source_tree_sha256(),
        "learner": "fixed_schedule_mse_occupancy_recursion",
        "reference": "fresh_target_policy_rollout",
    }
    _write_json(output_dir / "metadata.json", metadata)

    rows = []
    for seed in seeds:
        result = run_seed(seed, device=device)
        rows.extend(result)
        _write_json(output_dir / f"seed_{seed:05d}.json", {"seed": seed, "rows": result})

    aggregates = [_aggregate(rows, gamma) for gamma in GAMMAS]
    _write_json(output_dir / "summary.json", {"aggregates": aggregates})
    (output_dir / "COMPLETE").write_text("\n")


def run_seed(seed: int, *, device: str) -> list[dict[str, object]]:
    config = ExperimentConfig.from_yaml(ROOT / "configs" / "per_step_mimic_iv.yaml")
    config = replace(config, policy=replace(config.policy, policy_ratio_cap=3.0))
    torch.manual_seed(seed)
    context = _prepare_experiment_context(config, seed=seed, device=device)
    splits = context.task.splits
    logging_policy = context.logging_policy
    cot_scores = score_batch(
        context.region,
        splits.cot.current_states(),
        splits.cot.actions,
        splits.cot.outcomes,
    )
    q_low = float(torch.quantile(cot_scores.flatten(), 0.80).item())
    q_high = float(torch.quantile(cot_scores.flatten(), 0.95).item())
    alternative_policy = BehaviorAnchoredPolicy(
        outcome_model=context.outcome_model,
        reference_policy=logging_policy,
        config=replace(context.task.policy_config, policy_ratio_cap=3.0),
        region=context.region,
        tilt=20.0,
    )
    target_policy = ControlledMixturePolicy(
        logging_policy=logging_policy,
        alternative_policy=alternative_policy,
        radius_low=q_low,
        radius_high=q_high,
        maximum_response=1.0,
    )
    environment_batch = splits.environment
    if environment_batch is None:
        raise RuntimeError("the controlled clinical probe requires D_env")
    environment_scores = score_batch(
        context.region,
        environment_batch.current_states(),
        environment_batch.actions,
        environment_batch.outcomes,
    )
    difficulty = _empirical_rank_by_stage(environment_scores)
    environment = ControlledResidualEnvironment(
        environment_batch,
        outcome_model=context.outcome_model,
        n_actions=context.task.n_actions,
        difficulty=difficulty,
        history_length=config.model.history_length,
        static_indices=context.task.static_indices,
        state_feature_names=context.task.state_feature_names,
        neighbors=config.data.empirical_neighbors,
        bandwidth=config.data.empirical_bandwidth,
    )
    action_cost = torch.tensor(context.task.policy_config.action_costs, device=device)
    action_coordinate = 2.0 * (action_cost - action_cost.min()) / (action_cost.max() - action_cost.min()) - 1.0
    calibration_noise = make_controlled_noise(
        n=CALIBRATION_TRAJECTORIES,
        horizon=config.horizon,
        initial_count=environment.initial_count,
        seed=_paper_seed(seed, 1_700_101),
        device=device,
    )
    fit_noise = make_controlled_noise(
        n=FIT_TRAJECTORIES,
        horizon=config.horizon,
        initial_count=environment.initial_count,
        seed=_paper_seed(seed, 1_700_102),
        device=device,
    )
    reference_noise = make_controlled_noise(
        n=REFERENCE_TRAJECTORIES,
        horizon=config.horizon,
        initial_count=environment.initial_count,
        seed=_paper_seed(seed, 1_700_103),
        device=device,
    )

    results = []
    for gamma in GAMMAS:
        source_calibration = rollout_controlled(
            environment,
            logging_policy,
            noise=calibration_noise,
            gamma=gamma,
            action_coordinate=action_coordinate,
        )
        schedule = standard_cp_stagewise_radii(
            score_batch(
                context.region,
                source_calibration.trajectories.current_states(),
                source_calibration.trajectories.actions,
                source_calibration.trajectories.outcomes,
            ),
            alpha=0.10,
        )
        source_fit = rollout_controlled(
            environment,
            logging_policy,
            noise=fit_noise,
            gamma=gamma,
            action_coordinate=action_coordinate,
        )
        fitted_cot = fit_fixed_schedule_cot(
            source_fit.trajectories,
            schedule=schedule,
            target_policy=target_policy,
            logging_policy=logging_policy,
            outcome_model=context.outcome_model,
            config=config.cot,
            device=device,
            seed=_paper_seed(seed, int(1_700_200 + gamma)),
        )
        source_reference = rollout_controlled(
            environment,
            logging_policy,
            noise=reference_noise,
            gamma=gamma,
            action_coordinate=action_coordinate,
        )
        target_reference = rollout_controlled(
            environment,
            target_policy,
            noise=reference_noise,
            gamma=gamma,
            action_coordinate=action_coordinate,
            radii=schedule,
        )
        source_scores = score_batch(
            context.region,
            source_reference.trajectories.current_states(),
            source_reference.trajectories.actions,
            source_reference.trajectories.outcomes,
        )
        target_scores = score_batch(
            context.region,
            target_reference.trajectories.current_states(),
            target_reference.trajectories.actions,
            target_reference.trajectories.outcomes,
        )
        prefix_weights = _prefix_weights(
            source_reference.trajectories,
            schedule=schedule,
            target_policy=target_policy,
            logging_policy=logging_policy,
            device=device,
        ).to(source_scores)
        cot_weights, cot_diagnostics = fixed_schedule_state_action_weights(
            fitted_cot,
            source_reference.trajectories,
            target_policy=target_policy,
            logging_policy=logging_policy,
        )
        results.append(
            _metrics(
                seed=seed,
                gamma=gamma,
                schedule=schedule,
                source_scores=source_scores,
                target_scores=target_scores,
                prefix_weights=prefix_weights,
                cot_weights=cot_weights,
                cot_effective_size=cot_diagnostics.effective_sample_size,
                cot_validation_mse=fitted_cot.diagnostics.validation_mse,
                cot_normalization_error=fitted_cot.diagnostics.validation_normalization_error,
            )
        )
    return results


@torch.no_grad()
def _empirical_rank_by_stage(scores: torch.Tensor) -> torch.Tensor:
    order = scores.argsort(dim=0, stable=True).argsort(dim=0, stable=True)
    return order.float() / max(len(scores) - 1, 1)


@torch.no_grad()
def _prefix_weights(
    batch: object,
    *,
    schedule: torch.Tensor,
    target_policy: object,
    logging_policy: object,
    device: str,
) -> torch.Tensor:
    log_weight = torch.zeros(batch.n, dtype=torch.float64, device=device)
    weights = []
    for stage in range(batch.horizon):
        states = batch.states[:, stage]
        actions = batch.actions[:, stage]
        numerator = target_policy.probabilities(states, schedule[stage]).gather(1, actions[:, None]).squeeze(1)
        denominator = logging_policy.probabilities(states).gather(1, actions[:, None]).squeeze(1)
        log_weight += numerator.to(torch.float64).log() - denominator.to(torch.float64).log()
        weights.append((log_weight - log_weight.max()).exp())
    return torch.stack(weights, dim=1)


@torch.no_grad()
def _metrics(
    *,
    seed: int,
    gamma: float,
    schedule: torch.Tensor,
    source_scores: torch.Tensor,
    target_scores: torch.Tensor,
    prefix_weights: torch.Tensor,
    cot_weights: torch.Tensor,
    cot_effective_size: torch.Tensor,
    cot_validation_mse: tuple[float, ...],
    cot_normalization_error: tuple[float, ...],
) -> dict[str, object]:
    target_quantile = _quantile(target_scores)
    prefix_quantile = _weighted_quantile(source_scores, prefix_weights)
    cot_quantile = _weighted_quantile(source_scores, cot_weights)
    prefix_ess = _effective_size(prefix_weights) / len(prefix_weights)
    cot_ess = cot_effective_size / len(cot_weights)
    return {
        "seed": seed,
        "gamma": gamma,
        "schedule": schedule.detach().cpu().tolist(),
        "target_q90": target_quantile.cpu().tolist(),
        "prefix_q90": prefix_quantile.cpu().tolist(),
        "cot_q90": cot_quantile.cpu().tolist(),
        "prefix_q90_absolute_error": (prefix_quantile - target_quantile).abs().cpu().tolist(),
        "cot_q90_absolute_error": (cot_quantile - target_quantile).abs().cpu().tolist(),
        "prefix_cdf_sup_error": _cdf_sup_error(source_scores, prefix_weights, target_scores).cpu().tolist(),
        "cot_cdf_sup_error": _cdf_sup_error(source_scores, cot_weights, target_scores).cpu().tolist(),
        "prefix_ess_fraction": prefix_ess.cpu().tolist(),
        "cot_ess_fraction": cot_ess.cpu().tolist(),
        "cot_validation_mse": list(cot_validation_mse),
        "cot_normalization_error": list(cot_normalization_error),
    }


@torch.no_grad()
def _quantile(scores: torch.Tensor, probability: float = 0.90) -> torch.Tensor:
    index = min(len(scores) - 1, int(torch.ceil(scores.new_tensor(probability * len(scores))).item()) - 1)
    return scores.sort(dim=0).values[index]


@torch.no_grad()
def _weighted_quantile(scores: torch.Tensor, weights: torch.Tensor, probability: float = 0.90) -> torch.Tensor:
    values = []
    for stage in range(scores.shape[1]):
        order = scores[:, stage].argsort(stable=True)
        ordered_scores = scores[order, stage]
        ordered_weights = weights[order, stage]
        index = torch.searchsorted(
            ordered_weights.cumsum(dim=0),
            probability * ordered_weights.sum(),
            right=False,
        ).clamp_max(len(ordered_scores) - 1)
        values.append(ordered_scores[index])
    return torch.stack(values)


@torch.no_grad()
def _cdf_sup_error(source_scores: torch.Tensor, weights: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
    errors = []
    for stage in range(source_scores.shape[1]):
        grid = torch.sort(torch.cat((source_scores[:, stage], target_scores[:, stage]))).values
        order = source_scores[:, stage].argsort(stable=True)
        sorted_scores = source_scores[order, stage]
        cumulative_weight = weights[order, stage].cumsum(dim=0) / weights[:, stage].sum()
        source_index = torch.searchsorted(sorted_scores, grid, right=True).sub(1).clamp_min(0)
        estimate = torch.where(grid < sorted_scores[0], torch.zeros_like(grid), cumulative_weight[source_index])
        target_cdf = (target_scores[:, stage, None] <= grid[None, :]).float().mean(dim=0)
        errors.append((estimate - target_cdf).abs().max())
    return torch.stack(errors)


@torch.no_grad()
def _effective_size(weights: torch.Tensor) -> torch.Tensor:
    return weights.sum(dim=0).square() / weights.square().sum(dim=0).clamp_min(1e-12)


def _aggregate(rows: list[dict[str, object]], gamma: float) -> dict[str, object]:
    selected = [row for row in rows if row["gamma"] == gamma]
    prefix_q_error = torch.tensor([row["prefix_q90_absolute_error"] for row in selected])
    cot_q_error = torch.tensor([row["cot_q90_absolute_error"] for row in selected])
    prefix_cdf_error = torch.tensor([row["prefix_cdf_sup_error"] for row in selected])
    cot_cdf_error = torch.tensor([row["cot_cdf_sup_error"] for row in selected])
    prefix_ess = torch.tensor([row["prefix_ess_fraction"] for row in selected])
    cot_ess = torch.tensor([row["cot_ess_fraction"] for row in selected])
    return {
        "gamma": gamma,
        "n_seeds": len(selected),
        "prefix_q90_error_mean": float(prefix_q_error.mean()),
        "cot_q90_error_mean": float(cot_q_error.mean()),
        "prefix_cdf_sup_error_mean": float(prefix_cdf_error.mean()),
        "cot_cdf_sup_error_mean": float(cot_cdf_error.mean()),
        "prefix_ess_fraction_min": float(prefix_ess.min()),
        "cot_ess_fraction_min": float(cot_ess.min()),
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


if __name__ == "__main__":
    main()
