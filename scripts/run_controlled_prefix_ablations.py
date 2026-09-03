"""Run diagnostic ablations for the controlled performative benchmark.

This runner is deliberately separate from the canonical six-method paper
suite.  It reproduces the frozen confirm20 controlled benchmark and compares
the committed-prefix selector with two likelihood-ratio ablations and a
fixed-policy Prefix-IW control.  It never changes the canonical SC-PCP
implementation, and its post-confirmatory results must not be used to modify
that selector.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_controlled_prefix_benchmark import (  # noqa: E402
    _empirical_rank_by_stage,
    _normalized_width_by_stage,
    _policy_tv_by_stage,
    _prefix_diagnostics,
    _quantile,
    _vector,
)
from scpcp.artifacts import source_tree_sha256  # noqa: E402
from scpcp.config import ExperimentConfig  # noqa: E402
from scpcp.controlled_policy import ControlledMixturePolicy  # noqa: E402
from scpcp.controlled_transition import (  # noqa: E402
    ControlledResidualEnvironment,
    make_controlled_noise,
    rollout_controlled,
)
from scpcp.coverage import fixed_q_grid, weighted_stage_score_quantiles  # noqa: E402
from scpcp.data import TrajectoryBatch  # noqa: E402
from scpcp.experiment import (  # noqa: E402
    _paper_seed,
    _prepare_experiment_context,
    _training_outcome_sd,
)
from scpcp.marginal_prefix import select_marginal_prefix_schedule  # noqa: E402
from scpcp.policy.anchored import BehaviorAnchoredPolicy  # noqa: E402
from scpcp.scores import score_batch  # noqa: E402


PROTOCOL = "controlled_prefix_ablation_confirm20_v1"
PARENT_DIR = ROOT / "results" / "work" / "controlled_prefix_benchmark_confirm20_20260824"
PARENT_METADATA_SHA256 = "722198e6078bfe975a97812b2ac125e7a5c576a426e9de57882a5f1f15e8bb63"
PARENT_SUMMARY_SHA256 = "aca39f1f80a72f4e312d02b8a1a11679d7b461731e86b42eea2036c475908569"
PARENT_SEED_BUNDLE_SHA256 = "daa1e968b26125e82d47f08dc78a34ecba3669f0b5775d9f30474d5d2bba015d"
PARENT_SOURCE_TREE_SHA256 = "23403dc6d0282a4b0c22e8894a5e4dbd7f523737454e049f969080c14f3dee0d"

CONFIRM_SEEDS = tuple(range(12_400, 12_440, 2))
GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
CALIBRATION_TRAJECTORIES = 3_000
GRID_TRAJECTORIES = 1_000
REFERENCE_TRAJECTORIES = 20_000
LATE_STAGES = tuple(range(4, 12))
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 91_733

FULL_PREFIX = "SC-PCP"
OMIT_CURRENT = "SC-PCP w/o current-action ratio"
CURRENT_ONLY = "SC-PCP current-action only"
FROZEN_POLICY = "Frozen-policy Prefix-IW"
ONE_STEP_COUPLED = "One-step coupled Prefix-IW"
METHODS = (
    FULL_PREFIX,
    OMIT_CURRENT,
    CURRENT_ONLY,
    FROZEN_POLICY,
    ONE_STEP_COUPLED,
)

WeightMode = Literal["omit_current", "current_only"]


@dataclass(frozen=True)
class ScheduleSelection:
    radii: Tensor | None
    estimated_coverage: Tensor
    estimated_normalized_width: Tensor
    effective_sample_size: Tensor
    minimum_candidate_effective_sample_size: float | None
    selected_endpoint: bool | None
    failure_stage: int | None

    @property
    def available(self) -> bool:
        return self.radii is not None


@dataclass(frozen=True)
class ControlledSeedContext:
    config: ExperimentConfig
    experiment: object
    logging_policy: object
    target_policy: ControlledMixturePolicy
    environment: ControlledResidualEnvironment
    action_coordinate: Tensor
    outcome_sd: Tensor
    calibration_noise: object
    reference_noise: object
    q_low: float
    q_high: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if not devices or any(not value.startswith("cuda:") for value in devices):
        raise ValueError("the ablation benchmark requires explicit CUDA devices")
    run_benchmark(
        args.output_dir.resolve(),
        devices=devices,
        resume=args.resume,
    )
    print(args.output_dir.resolve())


def run_benchmark(
    output_dir: Path,
    *,
    devices: tuple[str, ...],
    resume: bool = False,
) -> None:
    parent_contract = _parent_contract(PARENT_DIR)
    active_source_hash = source_tree_sha256()
    metadata = _metadata(
        devices=devices,
        active_source_hash=active_source_hash,
        parent_contract=parent_contract,
    )
    metadata_path = output_dir / "metadata.json"
    if resume:
        if not metadata_path.exists():
            raise FileNotFoundError("resume requires an existing metadata.json")
        if json.loads(metadata_path.read_text()) != metadata:
            raise RuntimeError("resume metadata does not match the active ablation protocol")
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh output already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        _write_json(metadata_path, metadata)

    seed_contract = _seed_contract(metadata)
    completed = _completed_seeds(
        output_dir,
        seeds=CONFIRM_SEEDS,
        seed_contract=seed_contract,
    )
    pending = tuple(seed for seed in CONFIRM_SEEDS if seed not in completed)
    if pending and (output_dir / "COMPLETE").exists():
        raise RuntimeError("COMPLETE exists but one or more seed artifacts are missing")
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
                        {**seed_contract, "seed": seed, "rows": rows},
                    )
                    print(f"completed seed {seed}", flush=True)

    if source_tree_sha256() != active_source_hash:
        raise RuntimeError("source tree changed while the ablation benchmark was running")
    if _parent_contract(PARENT_DIR) != parent_contract:
        raise RuntimeError("parent confirm artifact changed while the ablation benchmark was running")

    rows: list[dict[str, Any]] = []
    for seed in CONFIRM_SEEDS:
        path = output_dir / f"seed_{seed:05d}.json"
        if not _valid_seed_file(
            path,
            seed=seed,
            seed_contract=seed_contract,
        ):
            raise RuntimeError(f"invalid or missing seed artifact: {path}")
        rows.extend(json.loads(path.read_text())["rows"])
    _write_json(output_dir / "summary.json", summarize(rows, seeds=CONFIRM_SEEDS))
    _write_text(output_dir / "COMPLETE", "\n")


def _metadata(
    *,
    devices: tuple[str, ...],
    active_source_hash: str,
    parent_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "role": "post_confirmatory_explanatory_ablation",
        "artifact_scope": "diagnostic_only_not_canonical_six_method_table",
        "canonical_selector_mutation_permitted": False,
        "base_config": "configs/per_step_mimic_iv.yaml",
        "devices": list(devices),
        "seeds": list(CONFIRM_SEEDS),
        "gammas": list(GAMMAS),
        "methods": list(METHODS),
        "calibration_trajectories": CALIBRATION_TRAJECTORIES,
        "grid_trajectories": GRID_TRAJECTORIES,
        "reference_trajectories": REFERENCE_TRAJECTORIES,
        "late_stages_zero_based": list(LATE_STAGES),
        "source_tree_sha256": active_source_hash,
        "parent_confirm": parent_contract,
        "fixed_policy": {
            "policy_radius": "constant_stagewise_midpoint_of_D_COT_q_low_q_high",
            "response_weight": 0.5,
            "calibration": "log_stabilized_prefix_IW_Hajek_left_quantile",
        },
        "ablation_definitions": {
            OMIT_CURRENT: "estimate stage t with W_0:t-1, then commit rho_t for future stages",
            CURRENT_ONLY: "estimate stage t with rho_t only and discard all committed history",
            FROZEN_POLICY: "calibrate q_IW under pi_qfix and keep pi_qfix at deployment",
            ONE_STEP_COUPLED: "calibrate q_IW under pi_qfix, then deploy pi_qIW",
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "complete_seed_vector",
        },
    }


def _seed_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    parent = metadata["parent_confirm"]
    return {
        "protocol": PROTOCOL,
        "source_tree_sha256": metadata["source_tree_sha256"],
        "parent_metadata_sha256": parent["metadata_sha256"],
        "parent_summary_sha256": parent["summary_sha256"],
        "parent_seed_bundle_sha256": parent["seed_bundle_sha256"],
    }


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


def _prepare_seed_context(seed: int, *, device: str) -> ControlledSeedContext:
    config = ExperimentConfig.from_yaml(ROOT / "configs" / "per_step_mimic_iv.yaml")
    config = replace(config, policy=replace(config.policy, policy_ratio_cap=3.0))
    torch.manual_seed(seed)
    experiment = _prepare_experiment_context(config, seed=seed, device=device)
    splits = experiment.task.splits
    logging_policy = experiment.logging_policy
    cot_scores = score_batch(
        experiment.region,
        splits.cot.current_states(),
        splits.cot.actions,
        splits.cot.outcomes,
    )
    q_low = float(torch.quantile(cot_scores.flatten(), 0.80).item())
    q_high = float(torch.quantile(cot_scores.flatten(), 0.95).item())
    alternative_policy = BehaviorAnchoredPolicy(
        outcome_model=experiment.outcome_model,
        reference_policy=logging_policy,
        config=replace(experiment.task.policy_config, policy_ratio_cap=3.0),
        region=experiment.region,
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
        experiment.region,
        environment_batch.current_states(),
        environment_batch.actions,
        environment_batch.outcomes,
    )
    environment = ControlledResidualEnvironment(
        environment_batch,
        outcome_model=experiment.outcome_model,
        n_actions=experiment.task.n_actions,
        difficulty=_empirical_rank_by_stage(environment_scores),
        history_length=config.model.history_length,
        static_indices=experiment.task.static_indices,
        state_feature_names=experiment.task.state_feature_names,
        neighbors=config.data.empirical_neighbors,
        bandwidth=config.data.empirical_bandwidth,
    )
    action_cost = torch.tensor(experiment.task.policy_config.action_costs, device=device)
    action_coordinate = 2.0 * (action_cost - action_cost.min()) / (
        action_cost.max() - action_cost.min()
    ) - 1.0
    return ControlledSeedContext(
        config=config,
        experiment=experiment,
        logging_policy=logging_policy,
        target_policy=target_policy,
        environment=environment,
        action_coordinate=action_coordinate,
        outcome_sd=_training_outcome_sd(splits.predictor).to(device),
        calibration_noise=make_controlled_noise(
            n=CALIBRATION_TRAJECTORIES,
            horizon=config.horizon,
            initial_count=environment.initial_count,
            seed=_paper_seed(seed, 1_700_101),
            device=device,
        ),
        reference_noise=make_controlled_noise(
            n=REFERENCE_TRAJECTORIES,
            horizon=config.horizon,
            initial_count=environment.initial_count,
            seed=_paper_seed(seed, 1_700_401),
            device=device,
        ),
        q_low=q_low,
        q_high=q_high,
    )


@torch.no_grad()
def select_ratio_ablation_schedule(
    batch: TrajectoryBatch,
    scores: Tensor,
    *,
    stage_grids: Tensor,
    target_policy: object,
    logging_policy: object,
    outcome_model: object,
    outcome_sd: Tensor,
    target: float,
    mode: WeightMode,
) -> ScheduleSelection:
    """Select a schedule after deleting one part of the prefix ratio."""

    if mode not in {"omit_current", "current_only"}:
        raise ValueError(f"unknown ratio ablation: {mode}")
    if scores.shape != batch.actions.shape:
        raise ValueError("scores must have shape [N,T]")
    resolved = batch.to(scores.device)
    grids = stage_grids.to(device=scores.device, dtype=scores.dtype)
    base_width = _normalized_base_width(
        resolved,
        outcome_model=outcome_model,
        outcome_sd=outcome_sd,
    )
    committed_log_weight = torch.zeros(
        batch.n,
        device=scores.device,
        dtype=torch.float64,
    )
    selected_radii: list[Tensor] = []
    selected_coverage: list[Tensor] = []
    selected_width: list[Tensor] = []
    selected_ess: list[Tensor] = []
    candidate_minimums: list[Tensor] = []
    selected_endpoint = False

    for stage, candidate_radii in enumerate(grids):
        states = resolved.states[:, stage]
        actions = resolved.actions[:, stage]
        target_probabilities = target_policy.probabilities_for_grid(
            states,
            candidate_radii,
        )
        logging_probabilities = logging_policy.probabilities(states)
        if (
            not bool(torch.isfinite(target_probabilities).all())
            or not bool(torch.isfinite(logging_probabilities).all())
            or bool((target_probabilities <= 0.0).any())
            or bool((logging_probabilities <= 0.0).any())
        ):
            raise RuntimeError(
                "ratio ablations require finite, strictly positive policies"
            )
        action_index = actions[:, None, None].expand(-1, len(candidate_radii), 1)
        numerator = target_probabilities.gather(2, action_index).squeeze(2)
        denominator = logging_probabilities.gather(1, actions[:, None])
        current_log_ratio = (
            numerator.to(torch.float64).log()
            - denominator.to(torch.float64).log()
        )
        if mode == "omit_current":
            estimation_log_weight = committed_log_weight[:, None].expand_as(
                current_log_ratio
            )
        else:
            estimation_log_weight = current_log_ratio

        maximum = estimation_log_weight.amax(dim=0)
        weights = (estimation_log_weight - maximum[None, :]).exp()
        weight_sum = weights.sum(dim=0).clamp_min(1e-12)
        coverage = (
            weights * (scores[:, stage, None] <= candidate_radii[None, :])
        ).sum(dim=0) / weight_sum
        candidate_width = base_width[:, stage, None] * candidate_radii[None, :]
        normalized_width = (weights * candidate_width).sum(dim=0) / weight_sum
        effective_size = weight_sum.square() / weights.square().sum(dim=0).clamp_min(
            1e-12
        )
        candidate_minimums.append(effective_size.min())

        feasible = coverage >= target
        if not bool(feasible.any()):
            return ScheduleSelection(
                radii=None,
                estimated_coverage=_stack_or_empty(selected_coverage, like=scores),
                estimated_normalized_width=_stack_or_empty(selected_width, like=scores),
                effective_sample_size=_stack_or_empty(selected_ess, like=scores),
                minimum_candidate_effective_sample_size=float(
                    torch.stack(candidate_minimums).min().item()
                ),
                selected_endpoint=selected_endpoint,
                failure_stage=stage,
            )

        objective = torch.where(
            feasible,
            normalized_width,
            torch.full_like(normalized_width, torch.inf),
        )
        index = int(objective.argmin().item())
        selected_radii.append(candidate_radii[index].clone())
        selected_coverage.append(coverage[index].clone())
        selected_width.append(normalized_width[index].clone())
        selected_ess.append(effective_size[index].clone())
        selected_endpoint = selected_endpoint or index in {
            0,
            len(candidate_radii) - 1,
        }
        if mode == "omit_current":
            committed_log_weight = (
                committed_log_weight + current_log_ratio[:, index]
            ).clone()

    return ScheduleSelection(
        radii=torch.stack(selected_radii),
        estimated_coverage=torch.stack(selected_coverage),
        estimated_normalized_width=torch.stack(selected_width),
        effective_sample_size=torch.stack(selected_ess),
        minimum_candidate_effective_sample_size=float(
            torch.stack(candidate_minimums).min().item()
        ),
        selected_endpoint=selected_endpoint,
        failure_stage=None,
    )


def _stack_or_empty(values: list[Tensor], *, like: Tensor) -> Tensor:
    return torch.stack(values) if values else like.new_empty(0)


def _normalized_base_width(
    batch: TrajectoryBatch,
    *,
    outcome_model: object,
    outcome_sd: Tensor,
) -> Tensor:
    states = batch.current_states().reshape(-1, batch.state_dim)
    actions = batch.actions.reshape(-1)
    _, scales = outcome_model(states, actions)
    scales = scales.reshape(batch.n, batch.horizon, -1)
    normalization = outcome_sd.to(scales).clamp_min(1e-6)
    return (2.0 * scales / normalization[None, None, :]).mean(dim=2)


def run_seed(seed: int, *, device: str) -> list[dict[str, Any]]:
    prepared = _prepare_seed_context(seed, device=device)
    parent_rows = _load_parent_seed_rows(seed)
    rows = []
    for gamma in GAMMAS:
        source_calibration = rollout_controlled(
            prepared.environment,
            prepared.logging_policy,
            noise=prepared.calibration_noise,
            gamma=gamma,
            action_coordinate=prepared.action_coordinate,
        )
        calibration_scores = score_batch(
            prepared.experiment.region,
            source_calibration.trajectories.current_states(),
            source_calibration.trajectories.actions,
            source_calibration.trajectories.outcomes,
        )
        stage_grids = torch.stack(
            [
                fixed_q_grid(
                    calibration_scores[:GRID_TRAJECTORIES, stage],
                    size=prepared.config.q_grid_size,
                    lower_quantile=prepared.config.q_quantile_min,
                    upper_quantile=prepared.config.q_quantile_max,
                )
                for stage in range(prepared.config.horizon)
            ]
        )
        target = 1.0 - prepared.config.certification.alpha
        canonical = _canonical_selection(
            source_calibration.trajectories,
            calibration_scores,
            stage_grids=stage_grids,
            prepared=prepared,
            target=target,
        )
        omit_current = select_ratio_ablation_schedule(
            source_calibration.trajectories,
            calibration_scores,
            stage_grids=stage_grids,
            target_policy=prepared.target_policy,
            logging_policy=prepared.logging_policy,
            outcome_model=prepared.experiment.outcome_model,
            outcome_sd=prepared.outcome_sd,
            target=target,
            mode="omit_current",
        )
        current_only = select_ratio_ablation_schedule(
            source_calibration.trajectories,
            calibration_scores,
            stage_grids=stage_grids,
            target_policy=prepared.target_policy,
            logging_policy=prepared.logging_policy,
            outcome_model=prepared.experiment.outcome_model,
            outcome_sd=prepared.outcome_sd,
            target=target,
            mode="current_only",
        )
        fixed_policy_radii = calibration_scores.new_full(
            (prepared.config.horizon,),
            0.5 * (prepared.q_low + prepared.q_high),
        )
        fixed_selection = _fixed_policy_selection(
            source_calibration.trajectories,
            calibration_scores,
            policy_radii=fixed_policy_radii,
            prepared=prepared,
        )

        source_reference = rollout_controlled(
            prepared.environment,
            prepared.logging_policy,
            noise=prepared.reference_noise,
            gamma=gamma,
            action_coordinate=prepared.action_coordinate,
        )
        source_scores = score_batch(
            prepared.experiment.region,
            source_reference.trajectories.current_states(),
            source_reference.trajectories.actions,
            source_reference.trajectories.outcomes,
        )
        method_rows: dict[str, dict[str, Any]] = {}
        for method, selection in (
            (FULL_PREFIX, canonical),
            (OMIT_CURRENT, omit_current),
            (CURRENT_ONLY, current_only),
        ):
            method_rows[method] = _evaluate_selected_method(
                method,
                selection,
                source_reference=source_reference,
                source_scores=source_scores,
                prepared=prepared,
                gamma=gamma,
            )

        if fixed_selection.radii is None:
            raise RuntimeError("fixed-policy Prefix-IW unexpectedly returned no schedule")
        method_rows[FROZEN_POLICY] = _evaluate_selected_method(
            FROZEN_POLICY,
            fixed_selection,
            source_reference=source_reference,
            source_scores=source_scores,
            prepared=prepared,
            gamma=gamma,
            policy_radii=fixed_policy_radii,
        )
        method_rows[ONE_STEP_COUPLED] = _evaluate_selected_method(
            ONE_STEP_COUPLED,
            fixed_selection,
            source_reference=source_reference,
            source_scores=source_scores,
            prepared=prepared,
            gamma=gamma,
            policy_radii=fixed_selection.radii,
        )

        expected_parent = parent_rows[gamma]
        _assert_parent_reproduction(
            seed=seed,
            gamma=gamma,
            q_low=prepared.q_low,
            q_high=prepared.q_high,
            observed=method_rows[FULL_PREFIX],
            expected=expected_parent,
        )
        rows.append(
            {
                "seed": seed,
                "gamma": gamma,
                "q_low": prepared.q_low,
                "q_high": prepared.q_high,
                "fixed_policy_radii": _vector(fixed_policy_radii),
                "methods": method_rows,
            }
        )
    return rows


def _canonical_selection(
    batch: TrajectoryBatch,
    scores: Tensor,
    *,
    stage_grids: Tensor,
    prepared: ControlledSeedContext,
    target: float,
) -> ScheduleSelection:
    result = select_marginal_prefix_schedule(
        batch,
        scores,
        stage_grids=stage_grids,
        target_policy=prepared.target_policy,
        logging_policy=prepared.logging_policy,
        outcome_model=prepared.experiment.outcome_model,
        outcome_sd=prepared.outcome_sd,
        target=target,
    )
    minimum_candidate = (
        float(result.candidate_effective_sample_size.min().item())
        if result.candidate_effective_sample_size.numel()
        else None
    )
    return ScheduleSelection(
        radii=result.radii,
        estimated_coverage=result.estimated_coverage,
        estimated_normalized_width=result.estimated_normalized_width,
        effective_sample_size=result.effective_sample_size,
        minimum_candidate_effective_sample_size=minimum_candidate,
        selected_endpoint=result.selected_endpoint,
        failure_stage=result.failure_stage,
    )


@torch.no_grad()
def _fixed_policy_selection(
    batch: TrajectoryBatch,
    scores: Tensor,
    *,
    policy_radii: Tensor,
    prepared: ControlledSeedContext,
) -> ScheduleSelection:
    weights, ess_fraction, _, _ = _prefix_diagnostics(
        batch,
        schedule=policy_radii,
        target_policy=prepared.target_policy,
        logging_policy=prepared.logging_policy,
    )
    radii = weighted_stage_score_quantiles(
        scores.to(weights),
        weights,
        alpha=prepared.config.certification.alpha,
    ).to(scores)
    weight_sum = weights.sum(dim=0).clamp_min(1e-12)
    estimated_coverage = (
        weights * (scores <= radii[None, :]).to(weights)
    ).sum(dim=0) / weight_sum
    base_width = _normalized_base_width(
        batch.to(scores.device),
        outcome_model=prepared.experiment.outcome_model,
        outcome_sd=prepared.outcome_sd,
    )
    estimated_width = (
        weights * base_width.to(weights) * radii[None, :].to(weights)
    ).sum(dim=0) / weight_sum
    return ScheduleSelection(
        radii=radii,
        estimated_coverage=estimated_coverage,
        estimated_normalized_width=estimated_width,
        effective_sample_size=ess_fraction * batch.n,
        minimum_candidate_effective_sample_size=None,
        selected_endpoint=None,
        failure_stage=None,
    )


def _evaluate_selected_method(
    method: str,
    selection: ScheduleSelection,
    *,
    source_reference: object,
    source_scores: Tensor,
    prepared: ControlledSeedContext,
    gamma: float,
    policy_radii: Tensor | None = None,
) -> dict[str, Any]:
    selection_fields = _selection_fields(selection)
    if selection.radii is None:
        return {
            "method": method,
            **selection_fields,
            "radii": None,
            "policy_radii": None,
        }
    interval_radii = selection.radii
    deployed_policy_radii = interval_radii if policy_radii is None else policy_radii
    target_reference = rollout_controlled(
        prepared.environment,
        prepared.target_policy,
        noise=prepared.reference_noise,
        gamma=gamma,
        action_coordinate=prepared.action_coordinate,
        radii=deployed_policy_radii,
    )
    target_scores = score_batch(
        prepared.experiment.region,
        target_reference.trajectories.current_states(),
        target_reference.trajectories.actions,
        target_reference.trajectories.outcomes,
    )
    _, ess_fraction, maximum_share, log_span = _prefix_diagnostics(
        source_reference.trajectories,
        schedule=deployed_policy_radii,
        target_policy=prepared.target_policy,
        logging_policy=prepared.logging_policy,
    )
    source_q90 = _quantile(source_scores)
    target_q90 = _quantile(target_scores)
    return {
        "method": method,
        **selection_fields,
        "radii": _vector(interval_radii),
        "policy_radii": _vector(deployed_policy_radii),
        "source_coverage": _vector(
            (source_scores <= interval_radii[None, :]).float().mean(dim=0)
        ),
        "target_coverage": _vector(
            (target_scores <= interval_radii[None, :]).float().mean(dim=0)
        ),
        "source_q90": _vector(source_q90),
        "target_q90": _vector(target_q90),
        "target_q90_to_radius_ratio": _vector(
            target_q90 / interval_radii.clamp_min(1e-12)
        ),
        "target_normalized_width": _vector(
            _normalized_width_by_stage(
                prepared.experiment.outcome_model,
                target_reference.trajectories,
                schedule=interval_radii,
                outcome_sd=prepared.outcome_sd,
            )
        ),
        "reference_prefix_ess_fraction": _vector(ess_fraction),
        "maximum_normalized_weight_share": _vector(maximum_share),
        "raw_log_weight_span": _vector(log_span),
        "policy_tv_on_source_states": _vector(
            _policy_tv_by_stage(
                source_reference.trajectories,
                schedule=deployed_policy_radii,
                target_policy=prepared.target_policy,
                logging_policy=prepared.logging_policy,
            )
        ),
        "source_difficulty": _vector(source_reference.donor_difficulty.mean(dim=0)),
        "target_difficulty": _vector(target_reference.donor_difficulty.mean(dim=0)),
        "donor_kernel_ess_fraction_min": float(
            torch.minimum(
                source_reference.donor_kernel_ess,
                target_reference.donor_kernel_ess,
            ).min().item()
            / prepared.config.data.empirical_neighbors
        ),
        "donor_probability_max": float(
            torch.maximum(
                source_reference.donor_probability_max,
                target_reference.donor_probability_max,
            ).max().item()
        ),
    }


def _selection_fields(selection: ScheduleSelection) -> dict[str, Any]:
    return {
        "selection_available": selection.available,
        "selection_estimated_coverage": _vector(selection.estimated_coverage),
        "selection_estimated_normalized_width": _vector(
            selection.estimated_normalized_width
        ),
        "selection_effective_sample_size": _vector(selection.effective_sample_size),
        "selection_minimum_candidate_effective_sample_size": (
            selection.minimum_candidate_effective_sample_size
        ),
        "selection_selected_endpoint": selection.selected_endpoint,
        "selection_failure_stage": selection.failure_stage,
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    gammas: tuple[float, ...] = GAMMAS,
) -> dict[str, Any]:
    if len(rows) != len(seeds) * len(gammas):
        raise RuntimeError("summary requires one row per seed and gamma")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = rng.integers(0, len(seeds), size=(BOOTSTRAP_RESAMPLES, len(seeds)))
    aggregates = []
    for gamma in gammas:
        selected = sorted(
            (row for row in rows if float(row["gamma"]) == gamma),
            key=lambda row: int(row["seed"]),
        )
        if tuple(int(row["seed"]) for row in selected) != seeds:
            raise RuntimeError(f"seed mismatch for gamma={gamma}")
        method_summaries = {
            method: _summarize_method(selected, method=method, bootstrap=bootstrap)
            for method in METHODS
        }
        comparisons = {
            method: _paired_comparison(
                selected,
                method=method,
                reference=FULL_PREFIX,
                bootstrap=bootstrap,
            )
            for method in METHODS
            if method != FULL_PREFIX
        }
        aggregates.append(
            {
                "gamma": gamma,
                "n_seeds": len(selected),
                "methods": method_summaries,
                "paired_vs_full_prefix": comparisons,
            }
        )
    return {
        "protocol": PROTOCOL,
        "role": "post_confirmatory_explanatory_ablation",
        "canonical_selector_mutation_permitted": False,
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


def _summarize_method(
    rows: list[dict[str, Any]],
    *,
    method: str,
    bootstrap: np.ndarray,
) -> dict[str, Any]:
    records = [row["methods"][method] for row in rows]
    unavailable_seeds = [
        int(row["seed"])
        for row, record in zip(rows, records)
        if not bool(record["selection_available"])
    ]
    summary: dict[str, Any] = {
        "selection_rate": 1.0 - len(unavailable_seeds) / len(rows),
        "unavailable_seeds": unavailable_seeds,
        "selection_endpoint_count": sum(
            record["selection_selected_endpoint"] is True for record in records
        ),
    }
    if unavailable_seeds:
        return {
            **summary,
            "target_marginal_worst_coverage": None,
            "target_marginal_worst_coverage_ci95": None,
            "target_coverage_by_stage": None,
            "mean_target_normalized_width": None,
            "minimum_reference_prefix_ess_fraction": None,
            "minimum_selection_ess_fraction": None,
            "minimum_candidate_selection_ess_fraction": None,
            "late_target_q90_to_radius_geometric_ratio": None,
            "late_target_q90_to_radius_geometric_ratio_ci95": None,
        }

    coverage = np.asarray(
        [record["target_coverage"] for record in records],
        dtype=np.float64,
    )
    stage_coverage = coverage.mean(axis=0)
    bootstrap_wsc = coverage[bootstrap].mean(axis=1).min(axis=1)
    widths = np.asarray(
        [np.mean(record["target_normalized_width"]) for record in records],
        dtype=np.float64,
    )
    selection_ess = np.asarray(
        [record["selection_effective_sample_size"] for record in records],
        dtype=np.float64,
    ) / CALIBRATION_TRAJECTORIES
    candidate_ess = [
        record["selection_minimum_candidate_effective_sample_size"]
        for record in records
        if record["selection_minimum_candidate_effective_sample_size"] is not None
    ]
    log_response = np.asarray(
        [
            np.log(np.asarray(record["target_q90_to_radius_ratio"]))[
                list(LATE_STAGES)
            ].mean()
            for record in records
        ],
        dtype=np.float64,
    )
    bootstrap_response = np.exp(log_response[bootstrap].mean(axis=1))
    return {
        **summary,
        "target_marginal_worst_coverage": float(stage_coverage.min()),
        "target_marginal_worst_coverage_ci95": _interval(bootstrap_wsc),
        "target_worst_stage_zero_based": int(stage_coverage.argmin()),
        "target_coverage_by_stage": stage_coverage.tolist(),
        "mean_target_normalized_width": float(widths.mean()),
        "minimum_reference_prefix_ess_fraction": float(
            min(min(record["reference_prefix_ess_fraction"]) for record in records)
        ),
        "minimum_selection_ess_fraction": float(selection_ess.min()),
        "minimum_candidate_selection_ess_fraction": (
            float(min(candidate_ess) / CALIBRATION_TRAJECTORIES)
            if candidate_ess
            else None
        ),
        "late_target_q90_to_radius_geometric_ratio": float(
            np.exp(log_response.mean())
        ),
        "late_target_q90_to_radius_geometric_ratio_ci95": _interval(
            bootstrap_response
        ),
    }


def _paired_comparison(
    rows: list[dict[str, Any]],
    *,
    method: str,
    reference: str,
    bootstrap: np.ndarray,
) -> dict[str, Any]:
    method_records = [row["methods"][method] for row in rows]
    reference_records = [row["methods"][reference] for row in rows]
    if any(
        not record["selection_available"]
        for record in (*method_records, *reference_records)
    ):
        return {
            "available": False,
            "marginal_worst_coverage_difference": None,
            "marginal_worst_coverage_difference_ci95": None,
            "geometric_width_ratio": None,
            "geometric_width_ratio_ci95": None,
        }

    method_coverage = np.asarray(
        [record["target_coverage"] for record in method_records],
        dtype=np.float64,
    )
    reference_coverage = np.asarray(
        [record["target_coverage"] for record in reference_records],
        dtype=np.float64,
    )
    method_wsc = method_coverage.mean(axis=0).min()
    reference_wsc = reference_coverage.mean(axis=0).min()
    bootstrap_difference = (
        method_coverage[bootstrap].mean(axis=1).min(axis=1)
        - reference_coverage[bootstrap].mean(axis=1).min(axis=1)
    )
    method_width = np.asarray(
        [np.mean(record["target_normalized_width"]) for record in method_records]
    )
    reference_width = np.asarray(
        [np.mean(record["target_normalized_width"]) for record in reference_records]
    )
    log_width_ratio = np.log(method_width / reference_width)
    bootstrap_width_ratio = np.exp(log_width_ratio[bootstrap].mean(axis=1))
    return {
        "available": True,
        "marginal_worst_coverage_difference": float(method_wsc - reference_wsc),
        "marginal_worst_coverage_difference_ci95": _interval(
            bootstrap_difference
        ),
        "geometric_width_ratio": float(np.exp(log_width_ratio.mean())),
        "geometric_width_ratio_ci95": _interval(bootstrap_width_ratio),
    }


def _interval(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _assert_parent_reproduction(
    *,
    seed: int,
    gamma: float,
    q_low: float,
    q_high: float,
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if not bool(observed["selection_available"]):
        raise RuntimeError(f"canonical SC-PCP unavailable at seed={seed}, gamma={gamma}")
    _assert_close(q_low, float(expected["q_low"]), label="q_low", seed=seed, gamma=gamma)
    _assert_close(q_high, float(expected["q_high"]), label="q_high", seed=seed, gamma=gamma)
    expected_method = expected["methods"][FULL_PREFIX]
    for field in (
        "radii",
        "source_coverage",
        "target_coverage",
        "target_q90",
        "target_normalized_width",
        "prefix_ess_fraction",
    ):
        observed_field = (
            "reference_prefix_ess_fraction"
            if field == "prefix_ess_fraction"
            else field
        )
        _assert_vector_close(
            observed[observed_field],
            expected_method[field],
            label=field,
            seed=seed,
            gamma=gamma,
        )


def _assert_close(
    observed: float,
    expected: float,
    *,
    label: str,
    seed: int,
    gamma: float,
) -> None:
    if not np.isclose(observed, expected, rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            f"parent reproduction mismatch for {label} at seed={seed}, gamma={gamma}"
        )


def _assert_vector_close(
    observed: object,
    expected: object,
    *,
    label: str,
    seed: int,
    gamma: float,
) -> None:
    observed_array = np.asarray(observed, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if observed_array.shape != expected_array.shape or not np.allclose(
        observed_array,
        expected_array,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"parent reproduction mismatch for {label} at seed={seed}, gamma={gamma}"
        )


def _parent_contract(parent_dir: Path) -> dict[str, Any]:
    if not (parent_dir / "COMPLETE").exists():
        raise RuntimeError(f"parent confirm artifact is incomplete: {parent_dir}")
    metadata_path = parent_dir / "metadata.json"
    summary_path = parent_dir / "summary.json"
    metadata_hash = _file_sha256(metadata_path)
    summary_hash = _file_sha256(summary_path)
    if metadata_hash != PARENT_METADATA_SHA256:
        raise RuntimeError("parent confirm metadata hash mismatch")
    if summary_hash != PARENT_SUMMARY_SHA256:
        raise RuntimeError("parent confirm summary hash mismatch")
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads(summary_path.read_text())
    expected_metadata = {
        "role": "confirm",
        "seeds": list(CONFIRM_SEEDS),
        "gammas": list(GAMMAS),
        "calibration_trajectories": CALIBRATION_TRAJECTORIES,
        "grid_trajectories": GRID_TRAJECTORIES,
        "reference_trajectories": REFERENCE_TRAJECTORIES,
        "source_tree_sha256": PARENT_SOURCE_TREE_SHA256,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"parent confirm metadata mismatch: {key}")
    if summary.get("role") != "confirm" or summary.get("seeds") != list(
        CONFIRM_SEEDS
    ):
        raise RuntimeError("parent confirm summary contract mismatch")
    seed_bundle_hash = _seed_bundle_sha256(parent_dir, CONFIRM_SEEDS)
    if seed_bundle_hash != PARENT_SEED_BUNDLE_SHA256:
        raise RuntimeError("parent confirm seed-bundle hash mismatch")
    return {
        "path": str(parent_dir.resolve()),
        "metadata_sha256": metadata_hash,
        "summary_sha256": summary_hash,
        "seed_bundle_sha256": seed_bundle_hash,
        "source_tree_sha256": PARENT_SOURCE_TREE_SHA256,
    }


def _load_parent_seed_rows(seed: int) -> dict[float, dict[str, Any]]:
    path = PARENT_DIR / f"seed_{seed:05d}.json"
    payload = json.loads(path.read_text())
    if payload.get("seed") != seed or len(payload.get("rows", ())) != len(GAMMAS):
        raise RuntimeError(f"invalid parent seed artifact: {path}")
    rows = {float(row["gamma"]): row for row in payload["rows"]}
    if tuple(rows) != GAMMAS:
        raise RuntimeError(f"parent gamma mismatch: {path}")
    return rows


def _seed_bundle_sha256(parent_dir: Path, seeds: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for seed in seeds:
        path = parent_dir / f"seed_{seed:05d}.json"
        name = path.name.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completed_seeds(
    output_dir: Path,
    *,
    seeds: tuple[int, ...],
    seed_contract: dict[str, Any],
) -> set[int]:
    expected_names = {f"seed_{seed:05d}.json" for seed in seeds}
    unexpected = sorted(
        path.name
        for path in output_dir.glob("seed_*.json")
        if path.name not in expected_names
    )
    if unexpected:
        raise RuntimeError(f"resume found unexpected seed artifacts: {unexpected}")
    completed = set()
    for seed in seeds:
        path = output_dir / f"seed_{seed:05d}.json"
        if not path.exists():
            continue
        if not _valid_seed_file(path, seed=seed, seed_contract=seed_contract):
            raise RuntimeError(f"resume found malformed or mismatched seed artifact: {path}")
        completed.add(seed)
    return completed


def _valid_seed_file(
    path: Path,
    *,
    seed: int,
    seed_contract: dict[str, Any],
) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if any(payload.get(key) != value for key, value in seed_contract.items()):
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


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


if __name__ == "__main__":
    main()
