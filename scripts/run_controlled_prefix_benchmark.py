"""Evaluate performative coverage drift and committed-prefix SC-PCP together.

This isolated semi-synthetic benchmark uses the same controlled transition
kernel for logging-policy and target-policy trajectories.  It therefore
separates policy-mediated score-law drift from an exogenous environment shift.
The canonical paper runner is not called or modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor


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
from scpcp.coverage import fixed_q_grid  # noqa: E402
from scpcp.experiment import (  # noqa: E402
    _paper_seed,
    _prepare_experiment_context,
    _training_outcome_sd,
)
from scpcp.marginal_prefix import select_marginal_prefix_schedule  # noqa: E402
from scpcp.policy.anchored import BehaviorAnchoredPolicy  # noqa: E402
from scpcp.scores import score_batch  # noqa: E402


PROTOCOL = "controlled_performative_prefix_benchmark_v1"
ENGINEERING_SEEDS = tuple(range(12_300, 12_310, 2))
DEVELOPMENT_SEEDS = tuple(range(12_200, 12_240, 2))
CONFIRM_SEEDS = tuple(range(12_400, 12_440, 2))
GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
METHODS = ("Standard CP", "SC-PCP")
CALIBRATION_TRAJECTORIES = 3_000
GRID_TRAJECTORIES = 1_000
REFERENCE_TRAJECTORIES = 20_000
LATE_STAGES = tuple(range(4, 12))
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 91_733


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=("engineering", "development20", "confirm"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if not devices or any(not value.startswith("cuda:") for value in devices):
        raise ValueError("the benchmark requires explicit CUDA devices")
    seeds = {
        "engineering": ENGINEERING_SEEDS,
        "development20": DEVELOPMENT_SEEDS,
        "confirm": CONFIRM_SEEDS,
    }[args.study]
    run_benchmark(
        args.output_dir.resolve(),
        seeds=seeds,
        role=args.study,
        devices=devices,
        resume=args.resume,
    )
    print(args.output_dir.resolve())


def run_benchmark(
    output_dir: Path,
    *,
    seeds: tuple[int, ...],
    role: str,
    devices: tuple[str, ...],
    resume: bool = False,
) -> None:
    metadata = {
        "protocol": PROTOCOL,
        "role": role,
        "base_config": "configs/per_step_mimic_iv.yaml",
        "seeds": list(seeds),
        "gammas": list(GAMMAS),
        "methods": list(METHODS),
        "calibration_trajectories": CALIBRATION_TRAJECTORIES,
        "grid_trajectories": GRID_TRAJECTORIES,
        "reference_trajectories": REFERENCE_TRAJECTORIES,
        "late_stages_zero_based": list(LATE_STAGES),
        "source_tree_sha256": source_tree_sha256(),
        "estimand": "same_gamma_same_radius_source_target_coverage_gap",
        "importance_weights": "uncapped_prefix_float64_log_stabilized",
        "environment_scope": "isolated_semi_synthetic_calibration_stress",
    }
    metadata_path = output_dir / "metadata.json"
    if resume:
        if not metadata_path.exists():
            raise FileNotFoundError("resume requires an existing metadata.json")
        if json.loads(metadata_path.read_text()) != metadata:
            raise RuntimeError("resume metadata does not match the active protocol")
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh output already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        _write_json(metadata_path, metadata)

    completed = {
        seed
        for seed in seeds
        if _valid_seed_file(output_dir / f"seed_{seed:05d}.json", seed=seed)
    }
    pending = tuple(seed for seed in seeds if seed not in completed)
    groups = [pending[offset:: len(devices)] for offset in range(len(devices))]
    if pending:
        with ProcessPoolExecutor(
            max_workers=len(devices),
            mp_context=get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(_run_seed_group, group, device): device
                for group, device in zip(groups, devices)
                if group
            }
            for future in as_completed(futures):
                for seed, rows in future.result():
                    _write_json(
                        output_dir / f"seed_{seed:05d}.json",
                        {"seed": seed, "rows": rows},
                    )
                    print(f"completed seed {seed}", flush=True)

    rows = []
    for seed in seeds:
        path = output_dir / f"seed_{seed:05d}.json"
        if not _valid_seed_file(path, seed=seed):
            raise RuntimeError(f"invalid or missing seed artifact: {path}")
        rows.extend(json.loads(path.read_text())["rows"])
    _write_json(output_dir / "summary.json", summarize(rows, seeds=seeds, role=role))
    (output_dir / "COMPLETE").write_text("\n")


def _run_seed_group(
    seeds: tuple[int, ...],
    device: str,
) -> list[tuple[int, list[dict[str, Any]]]]:
    torch.cuda.set_device(torch.device(device))
    completed = []
    for seed in seeds:
        completed.append((seed, run_seed(seed, device=device)))
        torch.cuda.empty_cache()
    return completed


def run_seed(seed: int, *, device: str) -> list[dict[str, Any]]:
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
        raise RuntimeError("the controlled benchmark requires D_env")
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
    action_coordinate = 2.0 * (action_cost - action_cost.min()) / (
        action_cost.max() - action_cost.min()
    ) - 1.0
    outcome_sd = _training_outcome_sd(splits.predictor).to(device)
    calibration_noise = make_controlled_noise(
        n=CALIBRATION_TRAJECTORIES,
        horizon=config.horizon,
        initial_count=environment.initial_count,
        seed=_paper_seed(seed, 1_700_101),
        device=device,
    )
    reference_noise = make_controlled_noise(
        n=REFERENCE_TRAJECTORIES,
        horizon=config.horizon,
        initial_count=environment.initial_count,
        seed=_paper_seed(seed, 1_700_401),
        device=device,
    )

    rows = []
    for gamma in GAMMAS:
        source_calibration = rollout_controlled(
            environment,
            logging_policy,
            noise=calibration_noise,
            gamma=gamma,
            action_coordinate=action_coordinate,
        )
        calibration_scores = score_batch(
            context.region,
            source_calibration.trajectories.current_states(),
            source_calibration.trajectories.actions,
            source_calibration.trajectories.outcomes,
        )
        grid_scores = calibration_scores[:GRID_TRAJECTORIES]
        stage_grids = torch.stack(
            [
                fixed_q_grid(
                    grid_scores[:, stage],
                    size=config.q_grid_size,
                    lower_quantile=config.q_quantile_min,
                    upper_quantile=config.q_quantile_max,
                )
                for stage in range(config.horizon)
            ]
        )
        standard = standard_cp_stagewise_radii(
            calibration_scores,
            config.certification.alpha,
        )
        selection = select_marginal_prefix_schedule(
            source_calibration.trajectories,
            calibration_scores,
            stage_grids=stage_grids,
            target_policy=target_policy,
            logging_policy=logging_policy,
            outcome_model=context.outcome_model,
            outcome_sd=outcome_sd,
            target=1.0 - config.certification.alpha,
        )
        if selection.radii is None:
            raise RuntimeError(
                f"SC-PCP failed at seed={seed}, gamma={gamma}, "
                f"stage={selection.failure_stage}"
            )
        schedules = {"Standard CP": standard, "SC-PCP": selection.radii}
        source_reference = rollout_controlled(
            environment,
            logging_policy,
            noise=reference_noise,
            gamma=gamma,
            action_coordinate=action_coordinate,
        )
        source_scores = score_batch(
            context.region,
            source_reference.trajectories.current_states(),
            source_reference.trajectories.actions,
            source_reference.trajectories.outcomes,
        )
        method_rows: dict[str, dict[str, Any]] = {}
        for method, schedule in schedules.items():
            target_reference = rollout_controlled(
                environment,
                target_policy,
                noise=reference_noise,
                gamma=gamma,
                action_coordinate=action_coordinate,
                radii=schedule,
            )
            target_scores = score_batch(
                context.region,
                target_reference.trajectories.current_states(),
                target_reference.trajectories.actions,
                target_reference.trajectories.outcomes,
            )
            weights, ess_fraction, maximum_share, log_span = _prefix_diagnostics(
                source_reference.trajectories,
                schedule=schedule,
                target_policy=target_policy,
                logging_policy=logging_policy,
            )
            del weights
            source_coverage = (source_scores <= schedule[None, :]).float().mean(dim=0)
            target_coverage = (target_scores <= schedule[None, :]).float().mean(dim=0)
            source_q90 = _quantile(source_scores)
            target_q90 = _quantile(target_scores)
            method_rows[method] = {
                "radii": _vector(schedule),
                "source_coverage": _vector(source_coverage),
                "target_coverage": _vector(target_coverage),
                "coverage_gap": _vector(target_coverage - source_coverage),
                "source_q90": _vector(source_q90),
                "target_q90": _vector(target_q90),
                "q90_relative_gap": _vector(target_q90 / source_q90.clamp_min(1e-8) - 1.0),
                "target_normalized_width": _vector(
                    _normalized_width_by_stage(
                        context.outcome_model,
                        target_reference.trajectories,
                        schedule=schedule,
                        outcome_sd=outcome_sd,
                    )
                ),
                "prefix_ess_fraction": _vector(ess_fraction),
                "maximum_normalized_weight_share": _vector(maximum_share),
                "raw_log_weight_span": _vector(log_span),
                "policy_tv_on_source_states": _vector(
                    _policy_tv_by_stage(
                        source_reference.trajectories,
                        schedule=schedule,
                        target_policy=target_policy,
                        logging_policy=logging_policy,
                    )
                ),
                "source_difficulty": _vector(source_reference.donor_difficulty.mean(dim=0)),
                "target_difficulty": _vector(target_reference.donor_difficulty.mean(dim=0)),
                "donor_kernel_ess_fraction_min": float(
                    torch.minimum(
                        source_reference.donor_kernel_ess,
                        target_reference.donor_kernel_ess,
                    ).min().item()
                    / config.data.empirical_neighbors
                ),
                "donor_probability_max": float(
                    torch.maximum(
                        source_reference.donor_probability_max,
                        target_reference.donor_probability_max,
                    ).max().item()
                ),
            }
        rows.append(
            {
                "seed": seed,
                "gamma": gamma,
                "q_low": q_low,
                "q_high": q_high,
                "selection_minimum_ess_fraction": float(
                    selection.effective_sample_size.min().item()
                    / CALIBRATION_TRAJECTORIES
                ),
                "selection_minimum_candidate_ess_fraction": float(
                    selection.candidate_effective_sample_size.min().item()
                    / CALIBRATION_TRAJECTORIES
                ),
                "selection_selected_endpoint": selection.selected_endpoint,
                "methods": method_rows,
            }
        )
    return rows


@torch.no_grad()
def _empirical_rank_by_stage(scores: Tensor) -> Tensor:
    order = scores.argsort(dim=0, stable=True).argsort(dim=0, stable=True)
    return order.float() / max(len(scores) - 1, 1)


@torch.no_grad()
def _quantile(scores: Tensor, probability: float = 0.90) -> Tensor:
    index = min(len(scores) - 1, int(np.ceil(probability * len(scores))) - 1)
    return scores.sort(dim=0).values[index]


@torch.no_grad()
def _policy_tv_by_stage(
    batch: object,
    *,
    schedule: Tensor,
    target_policy: object,
    logging_policy: object,
) -> Tensor:
    values = []
    for stage, radius in enumerate(schedule):
        states = batch.states[:, stage]
        target = target_policy.probabilities(states, radius)
        source = logging_policy.probabilities(states)
        values.append(0.5 * (target - source).abs().sum(dim=1).mean())
    return torch.stack(values)


@torch.no_grad()
def _prefix_diagnostics(
    batch: object,
    *,
    schedule: Tensor,
    target_policy: object,
    logging_policy: object,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    log_weight = torch.zeros(batch.n, dtype=torch.float64, device=batch.actions.device)
    weights, esses, shares, spans = [], [], [], []
    for stage, radius in enumerate(schedule):
        states = batch.states[:, stage]
        actions = batch.actions[:, stage]
        target = target_policy.probabilities(states, radius)
        source = logging_policy.probabilities(states)
        ratio = target.gather(1, actions[:, None]).squeeze(1) / source.gather(
            1, actions[:, None]
        ).squeeze(1)
        log_weight += ratio.to(torch.float64).log()
        stabilized = (log_weight - log_weight.max()).exp()
        total = stabilized.sum().clamp_min(1e-12)
        weights.append(stabilized)
        esses.append(total.square() / stabilized.square().sum().clamp_min(1e-12) / batch.n)
        shares.append(stabilized.max() / total)
        spans.append(log_weight.max() - log_weight.min())
    return (
        torch.stack(weights, dim=1),
        torch.stack(esses),
        torch.stack(shares),
        torch.stack(spans),
    )


@torch.no_grad()
def _normalized_width_by_stage(
    outcome_model: object,
    batch: object,
    *,
    schedule: Tensor,
    outcome_sd: Tensor,
) -> Tensor:
    states, actions, _ = batch.flat_transitions()
    scales = []
    for state_part, action_part in zip(states.split(4_096), actions.split(4_096)):
        _, scale = outcome_model(state_part, action_part)
        scales.append(scale)
    scale = torch.cat(scales).reshape(batch.n, batch.horizon, -1)
    normalized = 2.0 * schedule[None, :, None] * scale / outcome_sd[None, None, :]
    return normalized.mean(dim=(0, 2))


def summarize(
    rows: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    role: str,
) -> dict[str, Any]:
    if len(rows) != len(seeds) * len(GAMMAS):
        raise RuntimeError("summary requires one row per seed and gamma")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = rng.integers(0, len(seeds), size=(BOOTSTRAP_RESAMPLES, len(seeds)))
    aggregates = []
    for gamma in GAMMAS:
        selected = sorted(
            (row for row in rows if float(row["gamma"]) == gamma),
            key=lambda row: int(row["seed"]),
        )
        if tuple(int(row["seed"]) for row in selected) != seeds:
            raise RuntimeError(f"seed mismatch for gamma={gamma}")
        methods = {}
        for method in METHODS:
            coverage = np.asarray(
                [row["methods"][method]["target_coverage"] for row in selected],
                dtype=np.float64,
            )
            source_coverage = np.asarray(
                [row["methods"][method]["source_coverage"] for row in selected],
                dtype=np.float64,
            )
            widths = np.asarray(
                [
                    np.mean(row["methods"][method]["target_normalized_width"])
                    for row in selected
                ],
                dtype=np.float64,
            )
            stage_coverage = coverage.mean(axis=0)
            methods[method] = {
                "target_marginal_worst_coverage": float(stage_coverage.min()),
                "target_worst_stage_zero_based": int(stage_coverage.argmin()),
                "target_coverage_by_stage": stage_coverage.tolist(),
                "source_marginal_worst_coverage": float(source_coverage.mean(axis=0).min()),
                "mean_target_normalized_width": float(widths.mean()),
                "minimum_reference_prefix_ess_fraction": float(
                    min(min(row["methods"][method]["prefix_ess_fraction"]) for row in selected)
                ),
                "maximum_reference_weight_share": float(
                    max(max(row["methods"][method]["maximum_normalized_weight_share"]) for row in selected)
                ),
            }
        standard_gap = np.asarray(
            [row["methods"]["Standard CP"]["coverage_gap"] for row in selected],
            dtype=np.float64,
        )
        standard_q90_gap = np.asarray(
            [row["methods"]["Standard CP"]["q90_relative_gap"] for row in selected],
            dtype=np.float64,
        )
        standard_tv = np.asarray(
            [row["methods"]["Standard CP"]["policy_tv_on_source_states"] for row in selected],
            dtype=np.float64,
        )
        standard_difficulty_gap = np.asarray(
            [
                np.asarray(row["methods"]["Standard CP"]["target_difficulty"])
                - np.asarray(row["methods"]["Standard CP"]["source_difficulty"])
                for row in selected
            ],
            dtype=np.float64,
        )
        standard_width = np.asarray(
            [np.mean(row["methods"]["Standard CP"]["target_normalized_width"]) for row in selected]
        )
        scpcp_width = np.asarray(
            [np.mean(row["methods"]["SC-PCP"]["target_normalized_width"]) for row in selected]
        )
        log_ratio = np.log(scpcp_width / standard_width)
        late_gap = standard_gap[:, LATE_STAGES].mean(axis=1)
        late_q90 = standard_q90_gap[:, LATE_STAGES].mean(axis=1)
        late_difficulty = standard_difficulty_gap[:, LATE_STAGES].mean(axis=1)
        ratio_bootstrap = np.exp(log_ratio[bootstrap].mean(axis=1))
        aggregates.append(
            {
                "gamma": gamma,
                "n_seeds": len(selected),
                "methods": methods,
                "standard_same_radius_coverage_gap_by_stage": standard_gap.mean(axis=0).tolist(),
                "standard_late_coverage_gap": float(late_gap.mean()),
                "standard_late_coverage_gap_ci95": _bootstrap_interval(late_gap, bootstrap),
                "standard_late_q90_relative_gap": float(late_q90.mean()),
                "standard_late_q90_relative_gap_ci95": _bootstrap_interval(late_q90, bootstrap),
                "standard_late_policy_tv": float(standard_tv[:, LATE_STAGES].mean()),
                "standard_late_difficulty_gap": float(late_difficulty.mean()),
                "standard_late_difficulty_gap_ci95": _bootstrap_interval(late_difficulty, bootstrap),
                "scpcp_to_standard_width_ratio": float(np.exp(log_ratio.mean())),
                "scpcp_to_standard_width_ratio_ci95": [
                    float(np.quantile(ratio_bootstrap, 0.025)),
                    float(np.quantile(ratio_bootstrap, 0.975)),
                ],
                "selection_minimum_ess_fraction": float(
                    min(float(row["selection_minimum_ess_fraction"]) for row in selected)
                ),
                "selection_minimum_candidate_ess_fraction": float(
                    min(float(row["selection_minimum_candidate_ess_fraction"]) for row in selected)
                ),
                "selection_endpoint_count": sum(
                    bool(row["selection_selected_endpoint"]) for row in selected
                ),
                "donor_kernel_ess_fraction_min": float(
                    min(
                        row["methods"][method]["donor_kernel_ess_fraction_min"]
                        for row in selected
                        for method in METHODS
                    )
                ),
                "donor_probability_max": float(
                    max(
                        row["methods"][method]["donor_probability_max"]
                        for row in selected
                        for method in METHODS
                    )
                ),
            }
        )
    return {
        "protocol": PROTOCOL,
        "role": role,
        "seeds": list(seeds),
        "primary_metric": "min_t mean_seed(target_coverage_seed_t)",
        "late_stages_zero_based": list(LATE_STAGES),
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "complete_seed_vector",
        },
        "aggregates": aggregates,
    }


def _bootstrap_interval(values: np.ndarray, bootstrap: np.ndarray) -> list[float]:
    draws = values[bootstrap].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _vector(value: Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().tolist()]


def _valid_seed_file(path: Path, *, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("seed") != seed or len(payload.get("rows", ())) != len(GAMMAS):
        return False
    return all(
        row.get("seed") == seed
        and float(row.get("gamma")) == gamma
        and set(row.get("methods", {})) == set(METHODS)
        for row, gamma in zip(payload["rows"], GAMMAS)
    )


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


if __name__ == "__main__":
    main()
