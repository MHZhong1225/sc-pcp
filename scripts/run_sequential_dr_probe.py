"""Run the isolated sequential-DR score-law recovery development diagnostic.

This is not the paper SC-PCP method.  It holds the controlled environment,
policy, score, stagewise Standard-CP schedule, and 20-seed development bank
fixed, then compares prefix-IW recovery with an independently fitted
sequential doubly robust score-CDF estimator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from multiprocessing import get_context
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
from scpcp.policy.anchored import BehaviorAnchoredPolicy  # noqa: E402
from scpcp.scores import score_batch  # noqa: E402
from scpcp.sequential_dr import (  # noqa: E402
    dr_quantile,
    empirical_cdf,
    fit_fixed_schedule_sequential_dr,
    prefix_action_weights,
    sequential_dr_score_cdf,
)


PROTOCOL = "sequential_dr_score_recovery_development_v1"
DEVELOPMENT_SEEDS = tuple(range(12200, 12240, 2))
GAMMAS = (0.0, -2.0, -3.0, -4.0)
CALIBRATION_TRAJECTORIES = 3_000
FIT_TRAJECTORIES = 3_000
REFERENCE_TRAJECTORIES = 10_000
SCORE_CDF_GRID_SIZE = 201
COT_REFERENCE_ROOT = ROOT / "results" / "work" / "fixed_schedule_cot_probe_replication20_20260824"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "work" / "sequential_dr_probe_dev20_20260824",
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    args = parser.parse_args()
    devices = tuple(device.strip() for device in args.devices.split(",") if device.strip())
    run_probe(args.output_dir.resolve(), devices=devices)
    print(args.output_dir.resolve())


def run_probe(output_dir: Path, *, devices: tuple[str, ...]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"development output already exists: {output_dir}")
    if not devices:
        raise ValueError("at least one CUDA device is required")
    _load_cot_rows(COT_REFERENCE_ROOT)
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "metadata.json",
        {
            "protocol": PROTOCOL,
            "role": "development_only",
            "base_config": "configs/per_step_mimic_iv.yaml",
            "seeds": DEVELOPMENT_SEEDS,
            "gammas": GAMMAS,
            "calibration_trajectories": CALIBRATION_TRAJECTORIES,
            "fit_trajectories": FIT_TRAJECTORIES,
            "reference_trajectories": REFERENCE_TRAJECTORIES,
            "score_cdf_grid_size": SCORE_CDF_GRID_SIZE,
            "devices": devices,
            "source_tree_sha256": source_tree_sha256(),
            "continuation_learner": "fixed_schedule_vector_cdf_mse",
            "reference": "fresh_target_policy_rollout",
            "cot_reference_root": str(COT_REFERENCE_ROOT),
        },
    )

    rows: list[dict[str, object]] = []
    seed_groups = [DEVELOPMENT_SEEDS[offset:: len(devices)] for offset in range(len(devices))]
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=get_context("spawn")) as executor:
        futures = {
            executor.submit(_run_seed_group, seeds, device, COT_REFERENCE_ROOT): device
            for seeds, device in zip(seed_groups, devices)
            if seeds
        }
        for future in as_completed(futures):
            for seed, seed_rows in future.result():
                rows.extend(seed_rows)
                _write_json(output_dir / f"seed_{seed:05d}.json", {"seed": seed, "rows": seed_rows})

    summary = {"aggregates": [_aggregate(rows, gamma) for gamma in GAMMAS]}
    summary["gate"] = _gate(summary["aggregates"])
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "COMPLETE").write_text("\n")


def _run_seed_group(
    seeds: tuple[int, ...],
    device: str,
    cot_reference_root: Path,
) -> list[tuple[int, list[dict[str, object]]]]:
    cot_rows = _load_cot_rows(cot_reference_root)
    return [(seed, run_seed(seed, device=device, cot_rows=cot_rows)) for seed in seeds]


def run_seed(
    seed: int,
    *,
    device: str,
    cot_rows: dict[tuple[int, float], dict[str, object]],
) -> list[dict[str, object]]:
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

    rows = []
    for gamma in GAMMAS:
        source_calibration = rollout_controlled(
            environment, logging_policy, noise=calibration_noise, gamma=gamma, action_coordinate=action_coordinate
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
            environment, logging_policy, noise=fit_noise, gamma=gamma, action_coordinate=action_coordinate
        )
        fit_scores = score_batch(
            context.region,
            source_fit.trajectories.current_states(),
            source_fit.trajectories.actions,
            source_fit.trajectories.outcomes,
        )
        fitted = fit_fixed_schedule_sequential_dr(
            source_fit.trajectories,
            fit_scores,
            schedule=schedule,
            target_policy=target_policy,
            outcome_model=context.outcome_model,
            config=config.cot,
            device=device,
            seed=_paper_seed(seed, int(1_700_300 + gamma)),
        )
        source_reference = rollout_controlled(
            environment, logging_policy, noise=reference_noise, gamma=gamma, action_coordinate=action_coordinate
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
        prefix_weights = prefix_action_weights(
            source_reference.trajectories,
            schedule=schedule,
            target_policy=target_policy,
            logging_policy=logging_policy,
        )
        dr_cdf = sequential_dr_score_cdf(
            fitted,
            source_reference.trajectories,
            source_scores,
            target_policy=target_policy,
            logging_policy=logging_policy,
        )
        grid = fitted.model.score_grid
        prefix_cdf = empirical_cdf(source_scores, grid, weights=prefix_weights)
        target_cdf = empirical_cdf(target_scores, grid)
        target_quantile = _empirical_quantile(target_scores)
        prefix_quantile = dr_quantile(prefix_cdf, grid)
        dr_value = dr_quantile(dr_cdf, grid)
        cot_row = cot_rows[(seed, gamma)]
        if not torch.equal(schedule.cpu(), torch.tensor(cot_row["schedule"])):
            raise RuntimeError("replayed Standard-CP schedule does not match the frozen COT artifact")
        rows.append(
            {
                "seed": seed,
                "gamma": gamma,
                "schedule": schedule.cpu().tolist(),
                "target_q90": target_quantile.cpu().tolist(),
                "prefix_q90": prefix_quantile.cpu().tolist(),
                "dr_q90": dr_value.cpu().tolist(),
                "prefix_q90_absolute_error": (prefix_quantile - target_quantile).abs().cpu().tolist(),
                "dr_q90_absolute_error": (dr_value - target_quantile).abs().cpu().tolist(),
                "prefix_cdf_sup_error": (prefix_cdf - target_cdf).abs().amax(dim=1).cpu().tolist(),
                "dr_cdf_sup_error": (dr_cdf - target_cdf).abs().amax(dim=1).cpu().tolist(),
                "prefix_ess_fraction": _effective_size(prefix_weights).div(len(prefix_weights)).cpu().tolist(),
                "dr_validation_mse": list(fitted.validation_mse),
                "cot_q90_absolute_error_full_grid": cot_row["cot_q90_absolute_error"],
                "cot_cdf_sup_error_full_support": cot_row["cot_cdf_sup_error"],
            }
        )
    return rows


@torch.no_grad()
def _empirical_rank_by_stage(scores: torch.Tensor) -> torch.Tensor:
    order = scores.argsort(dim=0, stable=True).argsort(dim=0, stable=True)
    return order.float() / max(len(scores) - 1, 1)


@torch.no_grad()
def _empirical_quantile(scores: torch.Tensor, probability: float = 0.90) -> torch.Tensor:
    index = min(len(scores) - 1, int(torch.ceil(scores.new_tensor(probability * len(scores))).item()) - 1)
    return scores.sort(dim=0).values[index]


@torch.no_grad()
def _effective_size(weights: torch.Tensor) -> torch.Tensor:
    return weights.sum(dim=0).square() / weights.square().sum(dim=0).clamp_min(1e-12)


def _load_cot_rows(root: Path) -> dict[tuple[int, float], dict[str, object]]:
    if not (root / "COMPLETE").is_file():
        raise FileNotFoundError("the frozen 20-seed COT artifact is required")
    rows: dict[tuple[int, float], dict[str, object]] = {}
    for seed in DEVELOPMENT_SEEDS:
        payload = json.loads((root / f"seed_{seed:05d}.json").read_text())
        for row in payload["rows"]:
            rows[(seed, float(row["gamma"]))] = row
    expected = {(seed, gamma) for seed in DEVELOPMENT_SEEDS for gamma in GAMMAS}
    if rows.keys() != expected:
        raise RuntimeError("the frozen COT artifact is incomplete")
    return rows


def _aggregate(rows: list[dict[str, object]], gamma: float) -> dict[str, object]:
    selected = [row for row in rows if row["gamma"] == gamma]
    prefix_q_error = torch.tensor([row["prefix_q90_absolute_error"] for row in selected]).mean(dim=1)
    dr_q_error = torch.tensor([row["dr_q90_absolute_error"] for row in selected]).mean(dim=1)
    prefix_cdf_error = torch.tensor([row["prefix_cdf_sup_error"] for row in selected]).mean(dim=1)
    dr_cdf_error = torch.tensor([row["dr_cdf_sup_error"] for row in selected]).mean(dim=1)
    cot_q_error = torch.tensor([row["cot_q90_absolute_error_full_grid"] for row in selected]).mean(dim=1)
    cot_cdf_error = torch.tensor([row["cot_cdf_sup_error_full_support"] for row in selected]).mean(dim=1)
    ratio = dr_q_error.mean() / prefix_q_error.mean().clamp_min(1e-8)
    bootstrap = _paired_bootstrap_ratio(
        dr_q_error,
        prefix_q_error,
        seed=1_700_400 + int((gamma + 4.0) * 100),
    )
    return {
        "gamma": gamma,
        "n_seeds": len(selected),
        "prefix_q90_error_mean": float(prefix_q_error.mean()),
        "dr_q90_error_mean": float(dr_q_error.mean()),
        "cot_q90_error_mean_full_grid": float(cot_q_error.mean()),
        "prefix_cdf_sup_error_mean": float(prefix_cdf_error.mean()),
        "dr_cdf_sup_error_mean": float(dr_cdf_error.mean()),
        "cot_cdf_sup_error_mean_full_support": float(cot_cdf_error.mean()),
        "dr_to_prefix_q90_error_ratio": float(ratio),
        "dr_to_prefix_q90_error_ratio_upper_95": float(torch.quantile(bootstrap, 0.975)),
        "dr_cdf_better_than_prefix": bool(dr_cdf_error.mean() < prefix_cdf_error.mean()),
    }


def _paired_bootstrap_ratio(numerator: torch.Tensor, denominator: torch.Tensor, *, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(numerator), (10_000, len(numerator)), generator=generator)
    return numerator[indices].mean(dim=1) / denominator[indices].mean(dim=1).clamp_min(1e-8)


def _gate(aggregates: list[dict[str, object]]) -> dict[str, object]:
    lookup = {float(row["gamma"]): row for row in aggregates}
    strong = [lookup[-3.0], lookup[-4.0]]
    strong_pass = all(
        row["dr_to_prefix_q90_error_ratio"] <= 0.90
        and row["dr_to_prefix_q90_error_ratio_upper_95"] < 1.0
        and row["dr_cdf_better_than_prefix"]
        for row in strong
    )
    mild_pass = lookup[-2.0]["dr_to_prefix_q90_error_ratio"] <= 1.05
    return {
        "development_only": True,
        "strong_shift_pass": strong_pass,
        "mild_shift_non_degradation_pass": mild_pass,
        "status": "GO_TO_FRESH_CONFIRMATION" if strong_pass and mild_pass else "NO_GO",
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


if __name__ == "__main__":
    main()
