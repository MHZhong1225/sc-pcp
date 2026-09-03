"""Run the six canonical methods in the frozen controlled-shift environment.

This is an isolated follow-up to ``run_controlled_prefix_benchmark.py``.  It
keeps that benchmark's same-kernel gamma mechanism, 3,000 logged calibration
trajectories, 1,000-trajectory grid split, and 20,000 matched fresh evaluation
rollouts per method, seed, and gamma.  The pasted 50,000 budget belongs to the
general paper suite; retaining 20,000 here preserves Monte Carlo and protocol
parity with the parent controlled confirm benchmark.  The only added
comparison is the complete canonical method set.

No historical COT/profile/ratio ablation is labelled SC-PCP here.  Online
baseline adapters receive exactly 2,000 target-policy trajectories from the
same controlled transition kernel used for evaluation at the active gamma.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
from scipy import stats
import torch
from torch import Tensor
import yaml


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
from scpcp.baselines import (  # noqa: E402
    OnlineBaselineResult,
    aci_style_controller,
    finite_depth_mfcs_selection,
    multidim_spci_style_controller,
    prc_profile_scale,
    standard_cp_stagewise_radii,
)
from scpcp.config import ExperimentConfig  # noqa: E402
from scpcp.controlled_policy import ControlledMixturePolicy  # noqa: E402
from scpcp.controlled_transition import (  # noqa: E402
    ControlledResidualEnvironment,
    make_controlled_noise,
    rollout_controlled,
)
from scpcp.coverage import (  # noqa: E402
    fixed_q_grid,
    profiled_scale_grid,
    stage_score_profile,
)
from scpcp.data import TrajectoryBatch  # noqa: E402
from scpcp.experiment import (  # noqa: E402
    _paper_seed,
    _prepare_experiment_context,
    _training_outcome_sd,
)
from scpcp.marginal_prefix import select_marginal_prefix_schedule  # noqa: E402
from scpcp.policy.anchored import BehaviorAnchoredPolicy  # noqa: E402
from scpcp.scores import score_batch  # noqa: E402


PROTOCOL = "controlled_performative_six_method_benchmark_v1"
BASE_CONFIG_PATH = ROOT / "configs" / "per_step_mimic_iv.yaml"
CONFIRM_SEEDS = tuple(range(91_000, 91_200, 10))
GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
METHODS = ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")
INFORMATION_REGIME = {
    "Standard CP": "offline_logged_data",
    "ACI": "on_policy_adaptation",
    "MFCS": "offline_logged_data",
    "SPCI": "on_policy_adaptation",
    "PRC": "on_policy_adaptation",
    "SC-PCP": "offline_logged_data",
}
TARGET_ADAPTATION_BUDGET = {
    "Standard CP": 0,
    "ACI": 2_000,
    "MFCS": 0,
    "SPCI": 2_000,
    "PRC": 2_000,
    "SC-PCP": 0,
}
CALIBRATION_TRAJECTORIES = 3_000
GRID_TRAJECTORIES = 1_000
REFERENCE_TRAJECTORIES = 20_000
HORIZON = 12
LATE_STAGES = tuple(range(4, 12))
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 9_175_211
Q_LOW_SOURCE_QUANTILE = 0.80
Q_HIGH_SOURCE_QUANTILE = 0.95
ALTERNATIVE_POLICY_TILT = 20.0
MAXIMUM_POLICY_RESPONSE = 1.0
CONTROLLED_POLICY_RATIO_CAP = 3.0
_SEED_NAME = re.compile(r"seed_(\d+)(?:\.json)?$")
_SEED_ASSIGNMENT = re.compile(r"seed", re.IGNORECASE)


@dataclass(frozen=True)
class ControlledOnlineEnvironment:
    """Kernel parameters needed by the online-baseline rollout callback."""

    transition: ControlledResidualEnvironment
    gamma: float
    action_coordinate: Tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if not devices or any(not value.startswith("cuda:") for value in devices):
        raise ValueError("the controlled six-method benchmark requires explicit CUDA devices")
    run_benchmark(args.output_dir.resolve(), devices=devices, resume=args.resume)
    print(args.output_dir.resolve())


def run_benchmark(
    output_dir: Path,
    *,
    devices: tuple[str, ...],
    resume: bool = False,
) -> None:
    active_source_hash = source_tree_sha256()
    active_config_contract = _config_contract()
    seed_to_device = _seed_device_mapping(CONFIRM_SEEDS, devices)
    metadata_path = output_dir / "metadata.json"
    if resume:
        if not metadata_path.exists():
            raise FileNotFoundError("resume requires an existing metadata.json")
        stored_metadata = json.loads(metadata_path.read_text())
        metadata = _metadata(
            devices=devices,
            active_source_hash=active_source_hash,
            config_contract=active_config_contract,
            seed_to_device=seed_to_device,
            seed_audit=stored_metadata.get("base_seed_bank_audit", {}),
        )
        # JSON storage canonicalizes tuples as arrays. Compare the canonical
        # payload rather than Python container types so an artifact can resume
        # under the exact protocol that created it.
        if _json_sha256(stored_metadata) != _json_sha256(metadata):
            raise RuntimeError("resume metadata does not match the active protocol")
        _audit_seed_bank(CONFIRM_SEEDS, output_dir=output_dir)
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh output already exists: {output_dir}")
        seed_audit = _audit_seed_bank(CONFIRM_SEEDS, output_dir=output_dir)
        metadata = _metadata(
            devices=devices,
            active_source_hash=active_source_hash,
            config_contract=active_config_contract,
            seed_to_device=seed_to_device,
            seed_audit=seed_audit,
        )
        output_dir.mkdir(parents=True)
        _write_json(metadata_path, metadata)

    seed_contract = _seed_contract(metadata)
    _reject_unexpected_seed_artifacts(output_dir, CONFIRM_SEEDS)
    completed = _completed_seeds(
        output_dir,
        seeds=CONFIRM_SEEDS,
        seed_to_device=seed_to_device,
        seed_contract=seed_contract,
    )
    pending = tuple(seed for seed in CONFIRM_SEEDS if seed not in completed)
    if pending and (output_dir / "COMPLETE").exists():
        raise RuntimeError("COMPLETE exists but one or more seed artifacts are missing")

    groups = [
        tuple(seed for seed in pending if seed_to_device[seed] == device)
        for device in devices
    ]
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
                for seed, device, rows in future.result():
                    _write_json(
                        output_dir / f"seed_{seed:05d}.json",
                        {
                            **seed_contract,
                            "seed": seed,
                            "device": device,
                            "rows": rows,
                        },
                    )
                    print(f"completed seed {seed}", flush=True)

    if source_tree_sha256() != active_source_hash:
        raise RuntimeError("source tree changed while the benchmark was running")
    if _config_contract() != active_config_contract:
        raise RuntimeError("base or active controlled config changed while running")
    _audit_seed_bank(CONFIRM_SEEDS, output_dir=output_dir)
    rows: list[dict[str, Any]] = []
    for seed in CONFIRM_SEEDS:
        path = output_dir / f"seed_{seed:05d}.json"
        if not _valid_seed_file(
            path,
            seed=seed,
            expected_device=seed_to_device[seed],
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
    config_contract: dict[str, Any],
    seed_to_device: dict[int, str],
    seed_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "role": "fresh_confirmatory_canonical_baseline_comparison",
        "artifact_scope": "isolated_controlled_six_canonical_methods",
        "canonical_selector_mutation_permitted": False,
        "base_config": "configs/per_step_mimic_iv.yaml",
        "config_contract": config_contract,
        "devices": list(devices),
        "seed_to_device": {str(seed): seed_to_device[seed] for seed in CONFIRM_SEEDS},
        "seeds": list(CONFIRM_SEEDS),
        "gammas": list(GAMMAS),
        "methods": list(METHODS),
        "information_regime": INFORMATION_REGIME,
        "target_adaptation_trajectories": TARGET_ADAPTATION_BUDGET,
        "adaptation_round_sizes": [667, 667, 666],
        "calibration_trajectories": CALIBRATION_TRAJECTORIES,
        "grid_trajectories": GRID_TRAJECTORIES,
        "reference_trajectories": REFERENCE_TRAJECTORIES,
        "reference_budget_origin": (
            "20,000 fresh evaluation trajectories per method/seed/gamma from "
            "the frozen parent controlled confirm protocol; the pasted 50,000 "
            "budget is the general paper-suite setting and is intentionally "
            "not substituted because that would change Monte Carlo precision "
            "and break parent-protocol parity"
        ),
        "late_stages_zero_based": list(LATE_STAGES),
        "source_tree_sha256": active_source_hash,
        "base_seed_bank_audit": seed_audit,
        "rng_streams": {
            "calibration": "paper_seed(base_seed, 1700101); reused across gamma",
            "reference": (
                "paper_seed(base_seed, 1700401); matched across methods and gamma"
            ),
            "adaptation_root": "paper_seed(base_seed, 700001)",
            "ACI": "paper_seed(adaptation_root, 101) + 17923 * round",
            "SPCI": "paper_seed(adaptation_root, 211) + 47021 * round",
            "PRC": "paper_seed(adaptation_root, 307) + 61103 * round",
        },
        "common_random_numbers": {
            "source_calibration_across_gamma": True,
            "source_reference_across_gamma": True,
            "target_reference_across_methods_and_gamma": True,
            "online_baselines": "independent method streams reused across gamma",
        },
        "controlled_policy_response": {
            "q_low_source_quantile": Q_LOW_SOURCE_QUANTILE,
            "q_high_source_quantile": Q_HIGH_SOURCE_QUANTILE,
            "alternative_policy_tilt": ALTERNATIVE_POLICY_TILT,
            "maximum_response": MAXIMUM_POLICY_RESPONSE,
            "single_step_policy_ratio_cap": CONTROLLED_POLICY_RATIO_CAP,
        },
        "estimand": "target_policy_per_step_marginal_coverage_and_width",
        "importance_weights": "uncapped_prefix_float64_log_stabilized",
        "environment_scope": "isolated_same_kernel_semi_synthetic_calibration_stress",
        "guarantee_scope": "asymptotic_per_step_marginal",
    }


def _controlled_config() -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(BASE_CONFIG_PATH)
    return replace(
        config,
        policy=replace(
            config.policy,
            policy_ratio_cap=CONTROLLED_POLICY_RATIO_CAP,
        ),
    )


def _config_contract() -> dict[str, Any]:
    yaml_bytes = BASE_CONFIG_PATH.read_bytes()
    active_config = _controlled_config().to_dict()
    return {
        "base_yaml_sha256": hashlib.sha256(yaml_bytes).hexdigest(),
        "base_yaml_size_bytes": len(yaml_bytes),
        "active_config": active_config,
        "active_config_sha256": _json_sha256(active_config),
        "controlled_override": {
            "policy.policy_ratio_cap": CONTROLLED_POLICY_RATIO_CAP
        },
    }


def _seed_device_mapping(
    seeds: tuple[int, ...],
    devices: tuple[str, ...],
) -> dict[int, str]:
    return {
        seed: devices[index % len(devices)]
        for index, seed in enumerate(seeds)
    }


def _seed_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    audit = metadata["base_seed_bank_audit"]
    return {
        "protocol": PROTOCOL,
        "source_tree_sha256": metadata["source_tree_sha256"],
        "base_config_sha256": metadata["config_contract"]["base_yaml_sha256"],
        "active_config_sha256": metadata["config_contract"][
            "active_config_sha256"
        ],
        "base_seed_bank_sha256": audit["reserved_seed_sha256"],
        "rng_stream_mapping_sha256": audit["new_rng_stream_mapping_sha256"],
        "methods": list(METHODS),
        "reference_trajectories": REFERENCE_TRAJECTORIES,
    }


def _run_seed_group(
    seeds: tuple[int, ...],
    device: str,
) -> list[tuple[int, str, list[dict[str, Any]]]]:
    torch.cuda.set_device(torch.device(device))
    completed = []
    for seed in seeds:
        completed.append((seed, device, run_seed(seed, device=device)))
        torch.cuda.empty_cache()
    return completed


def run_seed(seed: int, *, device: str) -> list[dict[str, Any]]:
    config = _controlled_config()
    if config.horizon != HORIZON:
        raise RuntimeError(f"controlled protocol requires horizon={HORIZON}")
    if config.samples.online_rollouts != 2_000:
        raise RuntimeError("the frozen online-baseline budget must be exactly 2,000")
    if config.samples.oracle_rollouts == REFERENCE_TRAJECTORIES:
        raise RuntimeError(
            "controlled reference budget must remain explicitly distinct from the "
            "general paper-suite oracle budget"
        )
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
    q_low = float(
        torch.quantile(cot_scores.flatten(), Q_LOW_SOURCE_QUANTILE).item()
    )
    q_high = float(
        torch.quantile(cot_scores.flatten(), Q_HIGH_SOURCE_QUANTILE).item()
    )
    alternative_policy = BehaviorAnchoredPolicy(
        outcome_model=context.outcome_model,
        reference_policy=logging_policy,
        config=replace(
            context.task.policy_config,
            policy_ratio_cap=CONTROLLED_POLICY_RATIO_CAP,
        ),
        region=context.region,
        tilt=ALTERNATIVE_POLICY_TILT,
    )
    target_policy = ControlledMixturePolicy(
        logging_policy=logging_policy,
        alternative_policy=alternative_policy,
        radius_low=q_low,
        radius_high=q_high,
        maximum_response=MAXIMUM_POLICY_RESPONSE,
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
    environment = ControlledResidualEnvironment(
        environment_batch,
        outcome_model=context.outcome_model,
        n_actions=context.task.n_actions,
        difficulty=_empirical_rank_by_stage(environment_scores),
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
    adaptation_seed = _adaptation_seeds(seed)

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
        initial_profile = stage_score_profile(
            grid_scores,
            alpha=config.certification.alpha,
        )
        baseline_scale_grid = profiled_scale_grid(
            grid_scores,
            initial_profile,
            size=config.q_grid_size,
            lower_quantile=config.q_quantile_min,
            upper_quantile=config.q_quantile_max,
        )
        standard = standard_cp_stagewise_radii(
            calibration_scores,
            config.certification.alpha,
        )
        mfcs, _ = finite_depth_mfcs_selection(
            source_calibration.trajectories,
            calibration_scores,
            q_grid=baseline_scale_grid,
            stage_profile=initial_profile,
            target_policy=target_policy,
            logging_policy=logging_policy,
            depth=config.baselines.mfcs_depth,
            alpha=config.certification.alpha,
            weight_cap=config.cot.weight_cap,
        )
        scpcp = select_marginal_prefix_schedule(
            source_calibration.trajectories,
            calibration_scores,
            stage_grids=stage_grids,
            target_policy=target_policy,
            logging_policy=logging_policy,
            outcome_model=context.outcome_model,
            outcome_sd=outcome_sd,
            target=1.0 - config.certification.alpha,
        )

        online_environment = ControlledOnlineEnvironment(
            transition=environment,
            gamma=gamma,
            action_coordinate=action_coordinate,
        )
        aci = aci_style_controller(
            online_environment,
            target_policy,
            context.region,
            calibration_scores,
            alpha=config.certification.alpha,
            gamma=config.baselines.aci_gamma,
            rounds=config.baselines.online_rounds,
            total_rollouts=config.samples.online_rollouts,
            horizon=config.horizon,
            seed=adaptation_seed["ACI"],
            device=device,
            rollout_fn=_controlled_online_rollout,
        )
        spci = multidim_spci_style_controller(
            online_environment,
            target_policy,
            context.region,
            calibration_scores,
            alpha=config.certification.alpha,
            rounds=config.baselines.online_rounds,
            total_rollouts=config.samples.online_rollouts,
            horizon=config.horizon,
            seed=adaptation_seed["SPCI"],
            device=device,
            residual_window=config.baselines.multidim_buffer,
            rollout_fn=_controlled_online_rollout,
        )
        initial_prc_scale = float(
            (standard / initial_profile.to(standard)).max().item()
        )
        prc = prc_profile_scale(
            online_environment,
            target_policy,
            context.region,
            initial_prc_scale,
            baseline_scale_grid,
            initial_profile,
            alpha=config.certification.alpha,
            delta=config.certification.delta,
            rounds=config.baselines.online_rounds,
            total_rollouts=config.samples.online_rollouts,
            horizon=config.horizon,
            seed=adaptation_seed["PRC"],
            device=device,
            maximum_step=config.baselines.prc_maximum_step,
            rollout_fn=_controlled_online_rollout,
        )
        for name, adaptation in (("ACI", aci), ("SPCI", spci), ("PRC", prc)):
            if adaptation.target_deployments != TARGET_ADAPTATION_BUDGET[name]:
                raise RuntimeError(f"{name} did not consume its exact target-data budget")

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
        schedules: dict[str, Tensor | None] = {
            "Standard CP": standard,
            "ACI": aci.radius_by_time.to(device),
            "MFCS": (
                None
                if mfcs.radius is None
                else mfcs.radius * initial_profile.to(calibration_scores)
            ),
            "SPCI": spci.radius_by_time.to(device),
            "PRC": prc.radius_by_time.to(device),
            "SC-PCP": scpcp.radii,
        }
        adaptations = {"ACI": aci, "SPCI": spci, "PRC": prc}
        method_rows = {
            method: _evaluate_method(
                method,
                schedules[method],
                source_reference=source_reference,
                source_scores=source_scores,
                environment=environment,
                target_policy=target_policy,
                logging_policy=logging_policy,
                reference_noise=reference_noise,
                gamma=gamma,
                action_coordinate=action_coordinate,
                outcome_model=context.outcome_model,
                outcome_sd=outcome_sd,
                adaptation=adaptations.get(method),
                selection_status=(
                    mfcs.status
                    if method == "MFCS"
                    else (
                        "SELECTED_MARGINAL_POINT"
                        if method == "SC-PCP" and scpcp.radii is not None
                        else (
                            "UNAVAILABLE_NO_FEASIBLE_CANDIDATE"
                            if method == "SC-PCP"
                            else "AVAILABLE"
                        )
                    )
                ),
            )
            for method in METHODS
        }
        rows.append(
            {
                "seed": seed,
                "gamma": gamma,
                "q_low": q_low,
                "q_high": q_high,
                "adaptation_seeds": adaptation_seed,
                "scpcp_minimum_ess_fraction": _minimum_fraction(
                    scpcp.effective_sample_size,
                    CALIBRATION_TRAJECTORIES,
                ),
                "scpcp_minimum_candidate_ess_fraction": _minimum_fraction(
                    scpcp.candidate_effective_sample_size,
                    CALIBRATION_TRAJECTORIES,
                ),
                "scpcp_selected_endpoint": scpcp.selected_endpoint,
                "scpcp_failure_stage": scpcp.failure_stage,
                "methods": method_rows,
            }
        )
    return rows


@torch.no_grad()
def _controlled_online_rollout(
    environment: object,
    policy: BehaviorAnchoredPolicy,
    *,
    n: int,
    horizon: int,
    seed: int,
    device: str | torch.device,
    q: Tensor,
) -> TrajectoryBatch:
    if not isinstance(environment, ControlledOnlineEnvironment):
        raise TypeError("controlled online rollout requires ControlledOnlineEnvironment")
    if horizon != environment.transition.horizon or q.shape != (horizon,):
        raise ValueError("controlled online rollout requires one radius per environment stage")
    noise = make_controlled_noise(
        n=n,
        horizon=horizon,
        initial_count=environment.transition.initial_count,
        seed=seed,
        device=device,
    )
    return rollout_controlled(
        environment.transition,
        policy,
        noise=noise,
        gamma=environment.gamma,
        action_coordinate=environment.action_coordinate,
        radii=q,
    ).trajectories


@torch.no_grad()
def _evaluate_method(
    method: str,
    schedule: Tensor | None,
    *,
    source_reference: object,
    source_scores: Tensor,
    environment: ControlledResidualEnvironment,
    target_policy: object,
    logging_policy: object,
    reference_noise: object,
    gamma: float,
    action_coordinate: Tensor,
    outcome_model: object,
    outcome_sd: Tensor,
    adaptation: OnlineBaselineResult | None,
    selection_status: str,
) -> dict[str, Any]:
    adaptation_budget = 0 if adaptation is None else adaptation.target_deployments
    expected_budget = TARGET_ADAPTATION_BUDGET[method]
    if adaptation_budget != expected_budget:
        raise RuntimeError(f"{method} information budget mismatch")
    common: dict[str, Any] = {
        "selection_available": schedule is not None,
        "selection_status": selection_status,
        "information_regime": INFORMATION_REGIME[method],
        "target_adaptation_trajectories": adaptation_budget,
    }
    if adaptation is not None:
        common.update(
            {
                "adaptation_rounds": adaptation.rounds,
                "adaptation_per_time_coverage": _vector(
                    adaptation.adaptation_per_time_coverage
                ),
                "adaptation_round_worst_coverage": list(
                    adaptation.adaptation_round_worst_coverage
                ),
                "adaptation_pathwise_coverage": adaptation.adaptation_pathwise_coverage,
                "selected_scale": adaptation.selected_scale,
            }
        )
    if schedule is None:
        return {**common, "radii": []}

    resolved_schedule = schedule.to(source_scores)
    target_reference = rollout_controlled(
        environment,
        target_policy,
        noise=reference_noise,
        gamma=gamma,
        action_coordinate=action_coordinate,
        radii=resolved_schedule,
    )
    target_scores = score_batch(
        outcome_model,
        target_reference.trajectories.current_states(),
        target_reference.trajectories.actions,
        target_reference.trajectories.outcomes,
    )
    _, ess_fraction, maximum_share, log_span = _prefix_diagnostics(
        source_reference.trajectories,
        schedule=resolved_schedule,
        target_policy=target_policy,
        logging_policy=logging_policy,
    )
    source_coverage = (source_scores <= resolved_schedule[None, :]).float().mean(dim=0)
    target_coverage = (target_scores <= resolved_schedule[None, :]).float().mean(dim=0)
    source_q90 = _quantile(source_scores)
    target_q90 = _quantile(target_scores)
    return {
        **common,
        "radii": _vector(resolved_schedule),
        "source_coverage": _vector(source_coverage),
        "target_coverage": _vector(target_coverage),
        "coverage_gap": _vector(target_coverage - source_coverage),
        "source_q90": _vector(source_q90),
        "target_q90": _vector(target_q90),
        "q90_relative_gap": _vector(
            target_q90 / source_q90.clamp_min(1e-8) - 1.0
        ),
        "target_normalized_width": _vector(
            _normalized_width_by_stage(
                outcome_model,
                target_reference.trajectories,
                schedule=resolved_schedule,
                outcome_sd=outcome_sd,
            )
        ),
        "prefix_ess_fraction": _vector(ess_fraction),
        "maximum_normalized_weight_share": _vector(maximum_share),
        "raw_log_weight_span": _vector(log_span),
        "policy_tv_on_source_states": _vector(
            _policy_tv_by_stage(
                source_reference.trajectories,
                schedule=resolved_schedule,
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
            / environment.neighbors
        ),
        "donor_probability_max": float(
            torch.maximum(
                source_reference.donor_probability_max,
                target_reference.donor_probability_max,
            ).max().item()
        ),
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    if len(rows) != len(seeds) * len(GAMMAS):
        raise RuntimeError("summary requires one row per seed and gamma")
    aggregates = []
    bootstrap_seeds: dict[str, int] = {}
    for gamma_index, gamma in enumerate(GAMMAS):
        gamma_bootstrap_seed = _paper_seed(BOOTSTRAP_SEED, 101 + gamma_index)
        bootstrap_seeds[f"{gamma:g}"] = gamma_bootstrap_seed
        rng = np.random.default_rng(gamma_bootstrap_seed)
        shared_bootstrap_uniforms = rng.random(
            size=(BOOTSTRAP_RESAMPLES, len(seeds))
        )
        selected = sorted(
            (row for row in rows if float(row["gamma"]) == gamma),
            key=lambda row: int(row["seed"]),
        )
        if tuple(int(row["seed"]) for row in selected) != seeds:
            raise RuntimeError(f"seed mismatch for gamma={gamma}")
        methods = {}
        method_arrays: dict[str, dict[str, np.ndarray]] = {}
        for method in METHODS:
            available_mask = np.asarray(
                [
                    bool(row["methods"][method]["selection_available"])
                    for row in selected
                ]
            )
            available = [
                row for row, keep in zip(selected, available_mask) if keep
            ]
            coverage = np.zeros((len(seeds), HORIZON), dtype=np.float64)
            source_coverage = np.zeros_like(coverage)
            widths = np.zeros(len(seeds), dtype=np.float64)
            for index, row in enumerate(selected):
                if not available_mask[index]:
                    continue
                method_row = row["methods"][method]
                coverage[index] = np.asarray(
                    method_row["target_coverage"], dtype=np.float64
                )
                source_coverage[index] = np.asarray(
                    method_row["source_coverage"], dtype=np.float64
                )
                widths[index] = float(np.mean(method_row["target_normalized_width"]))
            method_arrays[method] = {
                "available": available_mask,
                "coverage": coverage,
                "width": widths,
            }
            method_summary: dict[str, Any] = {
                "selected_seeds": len(available),
                "total_seeds": len(selected),
                "selection_rate": len(available) / len(selected),
                "selection_rate_ci95": _wilson_interval(len(available), len(selected)),
                "target_adaptation_trajectories_per_seed": TARGET_ADAPTATION_BUDGET[method],
            }
            if not available:
                method_summary.update(
                    {
                        "target_marginal_worst_coverage": None,
                        "target_wsc_ci95": [None, None],
                        "target_mean_coverage": None,
                        "target_mean_coverage_ci95": [None, None],
                        "mean_target_normalized_width": None,
                        "mean_target_normalized_width_ci95": [None, None],
                    }
                )
                methods[method] = method_summary
                continue
            stage_coverage = coverage[available_mask].mean(axis=0)
            selected_bootstrap = _bootstrap_indices(
                shared_bootstrap_uniforms,
                len(available),
            )
            wsc_draws = _bootstrap_wsc(
                coverage[available_mask],
                selected_bootstrap,
            )
            method_summary.update(
                {
                    "target_marginal_worst_coverage": float(stage_coverage.min()),
                    "target_wsc_ci95": _percentile_interval(wsc_draws),
                    "target_worst_stage_zero_based": int(stage_coverage.argmin()),
                    "target_coverage_by_stage": stage_coverage.tolist(),
                    "target_mean_coverage": float(stage_coverage.mean()),
                    "target_mean_coverage_ci95": _student_t_interval(
                        coverage[available_mask].mean(axis=1)
                    ),
                    "source_marginal_worst_coverage": float(
                        source_coverage[available_mask].mean(axis=0).min()
                    ),
                    "mean_target_normalized_width": float(
                        widths[available_mask].mean()
                    ),
                    "mean_target_normalized_width_ci95": _student_t_interval(
                        widths[available_mask]
                    ),
                    "minimum_reference_prefix_ess_fraction": float(
                        min(
                            min(row["methods"][method]["prefix_ess_fraction"])
                            for row in available
                        )
                    ),
                    "maximum_reference_weight_share": float(
                        max(
                            max(
                                row["methods"][method][
                                    "maximum_normalized_weight_share"
                                ]
                            )
                            for row in available
                        )
                    ),
                }
            )
            methods[method] = method_summary
        paired = {
            baseline: _paired_scpcp_comparison(
                method_arrays["SC-PCP"],
                method_arrays[baseline],
                shared_bootstrap_uniforms,
            )
            for baseline in METHODS
            if baseline != "SC-PCP"
        }
        aggregates.append(
            {
                "gamma": gamma,
                "n_seeds": len(selected),
                "bootstrap_seed": gamma_bootstrap_seed,
                "methods": methods,
                "paired_scpcp_comparisons": paired,
            }
        )
    return {
        "protocol": PROTOCOL,
        "role": "fresh_confirmatory_canonical_baseline_comparison",
        "seeds": list(seeds),
        "methods": list(METHODS),
        "primary_metric": "min_t mean_seed(target_coverage_seed_t)",
        "coverage_conditioning": "successful_selection",
        "selection_rate_denominator": "all_prespecified_seeds",
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "root_seed": BOOTSTRAP_SEED,
            "gamma_seeds": bootstrap_seeds,
            "unit": "complete_seed_stage_vector",
            "coupling": (
                "one shared uniform matrix per gamma; methods or pairs with the "
                "same selected-set size reuse the same exact-size index matrix"
            ),
            "wsc_interval": "seed-vector percentile bootstrap",
            "paired_interval": "paired-seed percentile bootstrap",
            "selection_conditioning": (
                "each method resamples exactly its selected-seed count; paired "
                "comparisons resample exactly the joint-selected count"
            ),
            "mean_coverage_and_width_interval": "Student-t across selected seeds",
        },
        "aggregates": aggregates,
    }


def _bootstrap_wsc(
    coverage: np.ndarray,
    bootstrap: np.ndarray,
) -> np.ndarray:
    return coverage[bootstrap].mean(axis=1).min(axis=1)


def _bootstrap_indices(
    shared_uniforms: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    if sample_size < 1 or sample_size > shared_uniforms.shape[1]:
        raise ValueError("bootstrap sample size must match a nonempty selected set")
    return np.floor(
        shared_uniforms[:, :sample_size] * sample_size
    ).astype(np.int64)


def _paired_scpcp_comparison(
    scpcp: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    shared_bootstrap_uniforms: np.ndarray,
) -> dict[str, Any]:
    paired = scpcp["available"] & baseline["available"]
    pair_count = int(paired.sum())
    if pair_count == 0:
        return {
            "paired_selected_seeds": 0,
            "scpcp_minus_baseline_wsc": None,
            "scpcp_minus_baseline_wsc_ci95": [None, None],
            "scpcp_to_baseline_geometric_width_ratio": None,
            "scpcp_to_baseline_geometric_width_ratio_ci95": [None, None],
        }

    scpcp_coverage = scpcp["coverage"][paired]
    baseline_coverage = baseline["coverage"][paired]
    scpcp_wsc = float(scpcp_coverage.mean(axis=0).min())
    baseline_wsc = float(baseline_coverage.mean(axis=0).min())
    paired_bootstrap = _bootstrap_indices(
        shared_bootstrap_uniforms,
        pair_count,
    )
    scpcp_stage = scpcp_coverage[paired_bootstrap].mean(axis=1)
    baseline_stage = baseline_coverage[paired_bootstrap].mean(axis=1)
    wsc_difference_draws = scpcp_stage.min(axis=1) - baseline_stage.min(axis=1)

    log_width_ratio = np.log(
        scpcp["width"][paired] / baseline["width"][paired]
    )
    width_ratio_draws = np.exp(log_width_ratio[paired_bootstrap].mean(axis=1))
    return {
        "paired_selected_seeds": pair_count,
        "scpcp_minus_baseline_wsc": scpcp_wsc - baseline_wsc,
        "scpcp_minus_baseline_wsc_ci95": _percentile_interval(
            wsc_difference_draws
        ),
        "scpcp_to_baseline_geometric_width_ratio": float(
            np.exp(log_width_ratio.mean())
        ),
        "scpcp_to_baseline_geometric_width_ratio_ci95": _percentile_interval(
            width_ratio_draws
        ),
    }


def _audit_seed_bank(
    reserved: tuple[int, ...],
    *,
    output_dir: Path,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    if len(reserved) != 20 or len(set(reserved)) != 20:
        raise RuntimeError("the confirmatory seed bank must contain 20 unique seeds")
    artifact_root = ROOT / "results" if artifact_root is None else artifact_root
    source_root = ROOT if source_root is None else source_root
    artifact_seeds = _artifact_seeds(artifact_root, excluded_root=output_dir)
    source_seeds = _source_declared_seeds(
        source_root,
        excluded_paths={Path(__file__).resolve()},
    )
    prior_rng_ids = artifact_seeds | source_seeds
    rng_mapping = _new_rng_stream_mapping(reserved)
    collisions = {
        label: rng_id
        for label, rng_id in rng_mapping.items()
        if rng_id in prior_rng_ids
    }
    if collisions:
        raise RuntimeError(
            "new base/derived RNG stream collides with prior use: "
            f"{collisions}"
        )
    _assert_unique_rng_streams(rng_mapping)
    return {
        "status": "passed_before_launch",
        "reserved_seed_sha256": _integer_set_sha256(reserved),
        "artifact_seed_count": len(artifact_seeds),
        "artifact_seed_sha256": _integer_set_sha256(artifact_seeds),
        "source_declared_seed_count": len(source_seeds),
        "source_declared_seed_sha256": _integer_set_sha256(source_seeds),
        "prior_rng_id_count": len(prior_rng_ids),
        "prior_rng_id_sha256": _integer_set_sha256(prior_rng_ids),
        "new_rng_stream_count": len(rng_mapping),
        "new_rng_stream_mapping": rng_mapping,
        "new_rng_stream_mapping_sha256": _json_sha256(rng_mapping),
        "excluded_output": str(output_dir),
        "internal_rng_streams_unique": True,
    }


def _artifact_seeds(root: Path, *, excluded_root: Path) -> set[int]:
    values: set[int] = set()
    if not root.exists():
        return values
    excluded = excluded_root.resolve()
    for path in root.rglob("*"):
        if _is_relative_to(path.resolve(), excluded):
            continue
        match = _SEED_NAME.fullmatch(path.name)
        if match:
            values.add(int(match.group(1)))
        if not path.is_file() or path.name not in {
            "metadata.json",
            "study_metadata.json",
            "manifest.json",
            "summary.json",
        }:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        _collect_named_seed_values(payload, values)
    return values


def _source_declared_seeds(root: Path, *, excluded_paths: set[Path]) -> set[int]:
    values: set[int] = set()
    for directory in ("scripts", "src", "tools", "configs"):
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.resolve() in excluded_paths or not path.is_file():
                continue
            if path.suffix == ".py":
                _collect_python_seed_assignments(path, values)
            elif path.suffix in {".yaml", ".yml"}:
                try:
                    _collect_named_seed_values(yaml.safe_load(path.read_text()), values)
                except (OSError, yaml.YAMLError):
                    continue
    return values


def _collect_python_seed_assignments(path: Path, values: set[int]) -> None:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [name for target in node.targets for name in _target_names(target)]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            names = list(_target_names(node.target))
            value = node.value
        else:
            continue
        if value is None or not any(_SEED_ASSIGNMENT.search(name) for name in names):
            continue
        evaluated = _literal_seed_expression(value)
        if evaluated is not None:
            values.update(evaluated)


def _target_names(target: ast.expr) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_names(element)


def _literal_seed_expression(node: ast.expr) -> set[int] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return {node.value}
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        parts = [_literal_seed_expression(element) for element in node.elts]
        if any(part is None for part in parts):
            return None
        return set().union(*(part or set() for part in parts))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "range":
            arguments = [_literal_integer(argument) for argument in node.args]
            if any(argument is None for argument in arguments):
                return None
            return set(range(*(int(argument) for argument in arguments)))
        if node.func.id in {"tuple", "list", "set"} and len(node.args) == 1:
            return _literal_seed_expression(node.args[0])
    return None


def _literal_integer(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _literal_integer(node.operand)
        return None if operand is None else -operand
    return None


def _collect_named_seed_values(value: object, output: set[int], key: str = "") -> None:
    if isinstance(value, dict):
        if _SEED_ASSIGNMENT.search(key) and {"start", "stop"} <= set(value):
            start, stop = value["start"], value["stop"]
            if isinstance(start, int) and isinstance(stop, int):
                output.update(range(start, stop))
        for child_key, child_value in value.items():
            _collect_named_seed_values(child_value, output, str(child_key))
        return
    if isinstance(value, list):
        for child in value:
            _collect_named_seed_values(child, output, key)
        return
    if _SEED_ASSIGNMENT.search(key) and isinstance(value, int) and not isinstance(value, bool):
        output.add(value)


def _new_rng_stream_mapping(seeds: tuple[int, ...]) -> dict[str, int]:
    mapping: dict[str, int] = {
        f"summary/bootstrap_gamma_{gamma:g}": _paper_seed(
            BOOTSTRAP_SEED,
            101 + gamma_index,
        )
        for gamma_index, gamma in enumerate(GAMMAS)
    }
    for seed in seeds:
        prefix = f"base_{seed}"
        mapping[f"{prefix}/task"] = seed
        mapping[f"{prefix}/outcome_model"] = seed + 1
        mapping[f"{prefix}/behavior_model"] = seed + 2
        mapping[f"{prefix}/calibration"] = _paper_seed(seed, 1_700_101)
        mapping[f"{prefix}/reference"] = _paper_seed(seed, 1_700_401)
        adaptation_root = _paper_seed(seed, 700_001)
        for round_index in range(3):
            mapping[f"{prefix}/ACI_round_{round_index}"] = (
                _paper_seed(adaptation_root, 101) + 17_923 * round_index
            )
            mapping[f"{prefix}/SPCI_round_{round_index}"] = (
                _paper_seed(adaptation_root, 211) + 47_021 * round_index
            )
            mapping[f"{prefix}/PRC_round_{round_index}"] = (
                _paper_seed(adaptation_root, 307) + 61_103 * round_index
            )
    return mapping


def _adaptation_seeds(seed: int) -> dict[str, int]:
    adaptation_root = _paper_seed(seed, 700_001)
    return {
        "ACI": _paper_seed(adaptation_root, 101),
        "SPCI": _paper_seed(adaptation_root, 211),
        "PRC": _paper_seed(adaptation_root, 307),
    }


def _assert_unique_rng_streams(mapping: dict[str, int]) -> None:
    observed: dict[int, str] = {}
    for label, stream_seed in mapping.items():
        if stream_seed in observed:
            raise RuntimeError(
                f"RNG stream collision: {label} and {observed[stream_seed]}"
            )
        observed[stream_seed] = label


def _reject_unexpected_seed_artifacts(root: Path, seeds: tuple[int, ...]) -> None:
    expected = {root / f"seed_{seed:05d}.json" for seed in seeds}
    observed = set(root.glob("seed_*.json"))
    unexpected = sorted(observed - expected)
    if unexpected:
        raise RuntimeError(f"unexpected seed artifacts: {unexpected}")


def _completed_seeds(
    root: Path,
    *,
    seeds: tuple[int, ...],
    seed_to_device: dict[int, str],
    seed_contract: dict[str, Any],
) -> set[int]:
    completed = set()
    for seed in seeds:
        path = root / f"seed_{seed:05d}.json"
        if not path.exists():
            continue
        if not _valid_seed_file(
            path,
            seed=seed,
            expected_device=seed_to_device[seed],
            seed_contract=seed_contract,
        ):
            raise RuntimeError(f"malformed or provenance-mismatched seed artifact: {path}")
        completed.add(seed)
    return completed


def _valid_seed_file(
    path: Path,
    *,
    seed: int,
    expected_device: str,
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
    rows = payload.get("rows", ())
    if (
        payload.get("seed") != seed
        or payload.get("device") != expected_device
        or len(rows) != len(GAMMAS)
    ):
        return False
    for row, gamma in zip(rows, GAMMAS):
        if (
            row.get("seed") != seed
            or float(row.get("gamma")) != gamma
            or set(row.get("methods", {})) != set(METHODS)
            or row.get("adaptation_seeds") != _adaptation_seeds(seed)
        ):
            return False
        for method in METHODS:
            method_row = row["methods"][method]
            if method_row.get("target_adaptation_trajectories") != TARGET_ADAPTATION_BUDGET[method]:
                return False
            if not method_row.get("selection_available"):
                if method in {"Standard CP", "ACI", "SPCI", "PRC"}:
                    return False
                if method_row.get("radii") != []:
                    return False
                continue
            for name in (
                "radii",
                "source_coverage",
                "target_coverage",
                "coverage_gap",
                "target_normalized_width",
                "prefix_ess_fraction",
            ):
                values = method_row.get(name)
                if not isinstance(values, list) or len(values) != HORIZON:
                    return False
                if not all(np.isfinite(value) for value in values):
                    return False
    return True


def _minimum_fraction(values: Tensor, denominator: int) -> float | None:
    if values.numel() == 0:
        return None
    return float(values.min().item() / denominator)


def _wilson_interval(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total**2)) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _student_t_interval(values: np.ndarray) -> list[float]:
    mean = float(values.mean())
    if len(values) == 1:
        return [mean, mean]
    half = float(
        stats.t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / math.sqrt(len(values))
    )
    return [mean - half, mean + half]


def _integer_set_sha256(values: Iterable[int]) -> str:
    payload = json.dumps(sorted(set(values)), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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
