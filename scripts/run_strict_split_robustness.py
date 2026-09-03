#!/usr/bin/env python3
"""Run the frozen canonical-versus-strict calibration-split robustness audit.

This runner is isolated from the paper method.  It never edits or replaces the
canonical SC-PCP row, and its output supports no finite-sample guarantee or
post-hoc method-upgrade rule.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_controlled_prefix_benchmark import (  # noqa: E402
    _empirical_rank_by_stage,
    _normalized_width_by_stage,
)
from scpcp.artifacts import git_revision, source_tree_sha256  # noqa: E402
from scpcp.config import ExperimentConfig  # noqa: E402
from scpcp.controlled_policy import ControlledMixturePolicy  # noqa: E402
from scpcp.controlled_transition import (  # noqa: E402
    ControlledResidualEnvironment,
    make_controlled_noise,
    rollout_controlled,
)
from scpcp.device import resolve_devices  # noqa: E402
from scpcp.experiment import (  # noqa: E402
    _committed_prefix_stage_grids,
    _evaluate_radius_method,
    _paper_seed,
    _prepare_experiment_context,
)
from scpcp.policy.anchored import BehaviorAnchoredPolicy  # noqa: E402
from scpcp.scores import score_batch  # noqa: E402
from scpcp.strict_split_robustness import (  # noqa: E402
    CONFIG_PATH,
    FROZEN_CONFIG,
    PROTOCOL,
    SETTINGS,
    VARIANTS,
    evaluation_payload,
    json_sha256,
    load_frozen_config,
    selection_payload,
    select_strict_split_pair,
    setting_seeds,
    summarize_setting,
    tensor_sha256,
    validate_result_row,
)


PARENT_MANIFEST = (
    ROOT / "results" / "work" / "formal_source_snapshot_7665dfbe_20260825.manifest.json"
)
PARENT_ARCHIVE = (
    ROOT / "results" / "work" / "formal_source_snapshot_7665dfbe_20260825.tar.gz"
)
PARENT_MANIFEST_SHA256 = "e6a1bba7f3be47d39357f212824e7720262e7d5212a14628e3b8981088c64e24"
PARENT_ARCHIVE_SHA256 = "2116b9929240d8a25092f3c9015c362957bc963532930e5e720dc7ec78b2ea0b"
PARENT_ARCHIVE_BYTES = 2_036_776
PARENT_SOURCE_SHA256 = "7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643"

CONTROLLED_BASE_SEEDS = tuple(range(99_000, 99_200, 10))
BOOTSTRAP_RNG = 99_900
COORDINATED_EXTERNAL_RESERVATIONS = {
    "rq5_horizon_overlap": range(95_000, 97_000),
    "rq6_ncal_convergence": range(97_000, 98_000),
    "propensity_robustness": range(98_000, 99_000),
    "score_robustness": range(100_000, 101_000),
}
RNG_KEY = re.compile(r"(?:seed|rng|random)", re.IGNORECASE)
RESERVATION_KEY = re.compile(r"reservation", re.IGNORECASE)
ARTIFACT_ID = re.compile(r"(?:seed|problem|rng)[_-](\d+)(?:\.json)?$")
PROVENANCE_NAMES = {
    "metadata.json",
    "study_metadata.json",
    "manifest.json",
    "summary.json",
    "COMPLETE",
    "config.yaml",
    "suite_manifest.json",
    "study_status.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one reduced, explicitly nonformal engineering task",
    )
    args = parser.parse_args()
    devices = resolve_devices(args.devices)
    output_dir = args.output_dir.resolve()
    if args.smoke:
        if args.resume:
            raise ValueError("engineering smoke does not support resume")
        run_engineering_smoke(output_dir, device=devices[0], argv=tuple(sys.argv))
    else:
        if any(not device.startswith("cuda:") for device in devices):
            raise ValueError("formal strict-split robustness requires explicit CUDA devices")
        run_benchmark(
            output_dir,
            devices=devices,
            resume=args.resume,
            argv=tuple(sys.argv),
        )
    print(output_dir)


def run_benchmark(
    output_dir: Path,
    *,
    devices: tuple[str, ...],
    resume: bool = False,
    argv: tuple[str, ...] = (),
) -> None:
    """Run or exactly resume all three prespecified robustness settings."""

    protocol_config = load_frozen_config()
    _assert_seed_contract(protocol_config)
    parent_snapshot = validate_parent_snapshot()
    active_source_hash = source_tree_sha256()
    config_contract = active_config_contract(protocol_config)
    environment = runtime_environment()
    tasks = formal_tasks(protocol_config)
    device_mapping = stable_device_mapping(tasks, devices)
    current_audit = audit_fresh_controlled_rng_ids(output_dir=output_dir)

    manifest_path = output_dir / "manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("resume requires an existing strict-split manifest")
        stored = _read_json(manifest_path)
        expected = build_manifest(
            protocol_config=protocol_config,
            parent_snapshot=parent_snapshot,
            source_hash=active_source_hash,
            config_contract=config_contract,
            environment=environment,
            devices=devices,
            device_mapping=device_mapping,
            seed_audit=stored.get("fresh_controlled_rng_audit", {}),
            argv=tuple(stored.get("launch_argv", ())),
            created_at_utc=str(stored.get("created_at_utc", "")),
        )
        if stored != expected:
            raise RuntimeError("resume manifest differs from the active frozen protocol")
        if current_audit["status"] != "passed_before_launch":
            raise RuntimeError("fresh controlled RNG re-audit failed on resume")
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh strict-split output already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        for setting in SETTINGS:
            (output_dir / setting).mkdir()
        manifest = build_manifest(
            protocol_config=protocol_config,
            parent_snapshot=parent_snapshot,
            source_hash=active_source_hash,
            config_contract=config_contract,
            environment=environment,
            devices=devices,
            device_mapping=device_mapping,
            seed_audit=current_audit,
            argv=argv,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _atomic_write_json(manifest_path, manifest)

    manifest = _read_json(manifest_path)
    manifest_hash = _file_sha256(manifest_path)
    _reject_unexpected_artifacts(output_dir, tasks)
    completed = _completed_tasks(
        output_dir,
        tasks=tasks,
        manifest_hash=manifest_hash,
        device_mapping=device_mapping,
        source_hash=active_source_hash,
        config_contract_hash=config_contract["contract_sha256"],
        environment_hash=json_sha256(environment),
    )
    pending = tuple(task for task in tasks if task not in completed)
    if (output_dir / "COMPLETE").exists():
        if pending:
            raise RuntimeError("root COMPLETE exists with pending strict-split tasks")
        _validate_complete_bundle(
            output_dir,
            manifest_hash=manifest_hash,
            task_count=len(tasks),
            parent_snapshot=parent_snapshot,
            active_source_hash=active_source_hash,
        )
        _assert_active_contract_unchanged(
            source_hash=active_source_hash,
            config_contract=config_contract,
            parent_snapshot=parent_snapshot,
            environment=environment,
        )
        audit_fresh_controlled_rng_ids(output_dir=output_dir)
        return
    if pending:
        _run_pending(
            pending,
            output_dir=output_dir,
            device_mapping=device_mapping,
            manifest_hash=manifest_hash,
            source_hash=active_source_hash,
            config_contract_hash=config_contract["contract_sha256"],
            environment_hash=json_sha256(environment),
        )

    _assert_active_contract_unchanged(
        source_hash=active_source_hash,
        config_contract=config_contract,
        parent_snapshot=parent_snapshot,
        environment=environment,
    )
    audit_fresh_controlled_rng_ids(output_dir=output_dir)
    rows_by_setting: dict[str, list[dict[str, Any]]] = {setting: [] for setting in SETTINGS}
    artifact_entries = []
    for task in tasks:
        setting, seed = task
        artifact = _artifact_path(output_dir, task)
        row = _load_valid_artifact(
            artifact,
            setting=setting,
            seed=seed,
            expected_device=device_mapping[task],
            manifest_hash=manifest_hash,
            source_hash=active_source_hash,
            config_contract_hash=config_contract["contract_sha256"],
            environment_hash=json_sha256(environment),
        )
        rows_by_setting[setting].append(row)
        artifact_entries.append(
            {
                "setting": setting,
                "seed": seed,
                "path": str(artifact.relative_to(output_dir)),
                "row_sha256": _file_sha256(artifact / "row.json"),
                "metadata_sha256": _file_sha256(artifact / "metadata.json"),
                "complete_sha256": _file_sha256(artifact / "COMPLETE"),
            }
        )
    artifact_manifest = {
        "protocol": PROTOCOL,
        "manifest_sha256": manifest_hash,
        "task_count": len(tasks),
        "artifacts": artifact_entries,
    }
    _atomic_write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    summary = {
        "protocol": PROTOCOL,
        "role": "theory_aligned_robustness_only",
        "canonical_method_changed": False,
        "post_hoc_upgrade_rule": "none",
        "parent_snapshot": parent_snapshot,
        "active_source_tree_sha256": active_source_hash,
        "manifest_sha256": manifest_hash,
        "artifact_manifest_sha256": _file_sha256(output_dir / "artifact_manifest.json"),
        "settings": {
            setting: summarize_setting(
                rows_by_setting[setting],
                setting=setting,
                seeds=setting_seeds(protocol_config, setting),
                bootstrap_resamples=protocol_config["summary"]["bootstrap_resamples"],
                bootstrap_rng=protocol_config["summary"]["bootstrap_rng"],
            )
            for setting in SETTINGS
        },
        "claim_boundary": (
            "robustness-only; the canonical D_COT union D_cert method remains "
            "frozen, and neither variant has a finite-sample guarantee here"
        ),
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    complete = {
        "status": "complete",
        "protocol": PROTOCOL,
        "task_count": len(tasks),
        "manifest_sha256": manifest_hash,
        "summary_sha256": _file_sha256(output_dir / "summary.json"),
        "artifact_manifest_sha256": _file_sha256(output_dir / "artifact_manifest.json"),
        "parent_snapshot_manifest_sha256": parent_snapshot["manifest_sha256"],
        "parent_snapshot_archive_sha256": parent_snapshot["archive_sha256"],
        "parent_source_tree_sha256": parent_snapshot["source_tree_sha256"],
        "active_source_tree_sha256": active_source_hash,
    }
    _atomic_write_json(output_dir / "COMPLETE", complete)


def formal_tasks(config: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    return tuple(
        (setting, seed)
        for setting in SETTINGS
        for seed in setting_seeds(config, setting)
    )


def stable_device_mapping(
    tasks: tuple[tuple[str, int], ...],
    devices: tuple[str, ...],
) -> dict[tuple[str, int], str]:
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("device mapping requires unique devices")
    return {
        task: devices[index % len(devices)]
        for index, task in enumerate(tasks)
    }


def build_manifest(
    *,
    protocol_config: dict[str, Any],
    parent_snapshot: dict[str, Any],
    source_hash: str,
    config_contract: dict[str, Any],
    environment: dict[str, Any],
    devices: tuple[str, ...],
    device_mapping: dict[tuple[str, int], str],
    seed_audit: dict[str, Any],
    argv: tuple[str, ...],
    created_at_utc: str,
) -> dict[str, Any]:
    manifest = {
        "protocol": PROTOCOL,
        "role": "theory_aligned_robustness_only",
        "canonical_method_changed": False,
        "post_hoc_upgrade_rule": "none",
        "guarantee_scope": "asymptotic_per_step_marginal",
        "protocol_config": protocol_config,
        "protocol_config_sha256": json_sha256(protocol_config),
        "parent_formal_snapshot": parent_snapshot,
        "active_git_revision": git_revision(),
        "active_source_tree_sha256": source_hash,
        "active_config_contract": config_contract,
        "runtime_environment": environment,
        "runtime_environment_sha256": json_sha256(environment),
        "devices": list(devices),
        "stable_global_task_device_mapping": {
            f"{setting}/seed_{seed:05d}": device
            for (setting, seed), device in device_mapping.items()
        },
        "fresh_controlled_rng_audit": seed_audit,
        "legacy_main_seed_reuse": {
            "synthetic_main": list(setting_seeds(protocol_config, "synthetic_main")),
            "mimic_iv": list(setting_seeds(protocol_config, "mimic_iv")),
            "status": (
                "intentional paired reuse of frozen legacy mappings; these are "
                "not represented as a new independent RNG design"
            ),
        },
        "launch_argv": list(argv),
        "created_at_utc": created_at_utc,
    }
    return _json_normalized(manifest)


def _json_normalized(value: dict[str, Any]) -> dict[str, Any]:
    """Return the object exactly as JSON persistence will represent it."""

    return json.loads(json.dumps(value, allow_nan=False))


def _run_pending(
    tasks: tuple[tuple[str, int], ...],
    *,
    output_dir: Path,
    device_mapping: dict[tuple[str, int], str],
    manifest_hash: str,
    source_hash: str,
    config_contract_hash: str,
    environment_hash: str,
) -> None:
    devices = tuple(dict.fromkeys(device_mapping[task] for task in tasks))
    groups = {
        device: tuple(task for task in tasks if device_mapping[task] == device)
        for device in devices
    }
    with ProcessPoolExecutor(
        max_workers=len(devices),
        mp_context=get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(_run_task_group, group, device): device
            for device, group in groups.items()
            if group
        }
        for future in as_completed(futures):
            for task, device, row in future.result():
                write_task_artifact(
                    row,
                    output_dir=output_dir,
                    task=task,
                    device=device,
                    manifest_hash=manifest_hash,
                    source_hash=source_hash,
                    config_contract_hash=config_contract_hash,
                    environment_hash=environment_hash,
                )
                print(f"completed strict-split {task[0]} seed {task[1]}", flush=True)


def _run_task_group(
    tasks: tuple[tuple[str, int], ...],
    device: str,
) -> list[tuple[tuple[str, int], str, dict[str, Any]]]:
    if device.startswith("cuda:"):
        torch.cuda.set_device(torch.device(device))
    completed = []
    for setting, seed in tasks:
        row = run_task(setting, seed=seed, device=device)
        completed.append(((setting, seed), device, row))
        if device.startswith("cuda:"):
            torch.cuda.empty_cache()
    return completed


def run_task(setting: str, *, seed: int, device: str) -> dict[str, Any]:
    config = load_frozen_config()
    if seed not in setting_seeds(config, setting):
        raise ValueError(f"seed {seed} is outside {setting}'s frozen bank")
    if setting == "controlled_gamma_minus_2":
        return run_controlled_task(seed=seed, device=device)
    return run_main_task(setting, seed=seed, device=device)


def run_main_task(setting: str, *, seed: int, device: str) -> dict[str, Any]:
    """Run both variants on the frozen Synthetic or MIMIC-IV main setting."""

    protocol_config = load_frozen_config()
    setting_config = protocol_config["settings"][setting]
    config_path = ROOT / setting_config["config"]
    config = ExperimentConfig.from_yaml(config_path)
    torch.manual_seed(seed)
    context = _prepare_experiment_context(config, seed=seed, device=device)
    splits = context.task.splits
    cot_scores = score_batch(
        context.region,
        splits.cot.current_states(),
        splits.cot.actions,
        splits.cot.outcomes,
    )
    certification_scores = score_batch(
        context.region,
        splits.certification.current_states(),
        splits.certification.actions,
        splits.certification.outcomes,
    )
    stage_grids = _committed_prefix_stage_grids(cot_scores, config)
    selections = select_strict_split_pair(
        cot_batch=splits.cot,
        cot_scores=cot_scores,
        certification_batch=splits.certification,
        certification_scores=certification_scores,
        stage_grids=stage_grids,
        target_policy=context.policy,
        logging_policy=context.logging_policy,
        outcome_model=context.outcome_model,
        outcome_sd=context.outcome_sd,
        target=1.0 - config.certification.alpha,
    )
    evaluation_rng = _paper_seed(seed, 900_001)
    variants = {}
    for variant in VARIANTS:
        selection = selections[variant]
        selection_batch_size = (
            splits.cot.n + splits.certification.n
            if variant == "canonical"
            else splits.certification.n
        )
        payload = selection_payload(
            selection,
            calibration_trajectories=selection_batch_size,
            calibration_roles=("D_COT", "D_cert") if variant == "canonical" else ("D_cert",),
        )
        if selection.radii is None:
            evaluation = evaluation_payload(
                coverage=None,
                normalized_width_by_stage=None,
                evaluation_trajectories=0,
                evaluation_rng=evaluation_rng,
            )
        else:
            record = _evaluate_radius_method(
                f"SC-PCP-{variant}",
                selection.radii,
                context.task,
                context.policy,
                context.region,
                config,
                evaluation_rng,
                device,
                outcome_sd=context.outcome_sd,
            )
            evaluation = evaluation_payload(
                coverage=json.loads(record["per_time_coverage"]),
                normalized_width_by_stage=json.loads(
                    record["per_time_normalized_width"]
                ),
                evaluation_trajectories=int(record["oracle_evaluation_trajectories"]),
                evaluation_rng=evaluation_rng,
            )
        variants[variant] = {**payload, "evaluation": evaluation}

    row = {
        "protocol": PROTOCOL,
        "setting": setting,
        "seed": seed,
        "horizon": config.horizon,
        "base_config": str(config_path.relative_to(ROOT)),
        "stage_grid_roles": ["D_COT"],
        "stage_grid_sha256": tensor_sha256(stage_grids),
        "stage_grid_shape": list(stage_grids.shape),
        "matched_evaluation_crn": True,
        "evaluation_rng": evaluation_rng,
        "split_sizes": {
            "D_pred": splits.predictor.n,
            "D_COT": splits.cot.n,
            "D_cert": splits.certification.n,
            "D_env": 0 if splits.environment is None else splits.environment.n,
        },
        "variants": variants,
        "claim_boundary": "robustness_only_asymptotic_per_step_marginal",
    }
    validate_result_row(row, setting=setting)
    return row


def run_controlled_task(*, seed: int, device: str) -> dict[str, Any]:
    """Reproduce the frozen same-kernel gamma=-2 mechanism for both splits."""

    protocol_config = load_frozen_config()
    controlled = protocol_config["settings"]["controlled_gamma_minus_2"]
    config_path = ROOT / controlled["config"]
    config = ExperimentConfig.from_yaml(config_path)
    config = replace(
        config,
        policy=replace(
            config.policy,
            policy_ratio_cap=controlled["policy_ratio_cap"],
        ),
    )
    if config.horizon != controlled["horizon"]:
        raise RuntimeError("controlled strict-split horizon differs from frozen protocol")
    torch.manual_seed(seed)
    context = _prepare_experiment_context(config, seed=seed, device=device)
    splits = context.task.splits
    logged_cot_scores = score_batch(
        context.region,
        splits.cot.current_states(),
        splits.cot.actions,
        splits.cot.outcomes,
    )
    q_low = float(
        torch.quantile(
            logged_cot_scores.flatten(), controlled["q_low_source_quantile"]
        ).item()
    )
    q_high = float(
        torch.quantile(
            logged_cot_scores.flatten(), controlled["q_high_source_quantile"]
        ).item()
    )
    alternative_policy = BehaviorAnchoredPolicy(
        outcome_model=context.outcome_model,
        reference_policy=context.logging_policy,
        config=replace(
            context.task.policy_config,
            policy_ratio_cap=controlled["policy_ratio_cap"],
        ),
        region=context.region,
        tilt=controlled["alternative_policy_tilt"],
    )
    target_policy = ControlledMixturePolicy(
        logging_policy=context.logging_policy,
        alternative_policy=alternative_policy,
        radius_low=q_low,
        radius_high=q_high,
        maximum_response=controlled["maximum_policy_response"],
    )
    environment_batch = splits.environment
    if environment_batch is None:
        raise RuntimeError("controlled strict-split setting requires D_env")
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
    calibration_rng = _paper_seed(seed, 1_700_101)
    reference_rng = _paper_seed(seed, 1_700_401)
    calibration_noise = make_controlled_noise(
        n=controlled["calibration_trajectories"],
        horizon=config.horizon,
        initial_count=environment.initial_count,
        seed=calibration_rng,
        device=device,
    )
    source_calibration = rollout_controlled(
        environment,
        context.logging_policy,
        noise=calibration_noise,
        gamma=controlled["gamma"],
        action_coordinate=action_coordinate,
    ).trajectories
    calibration_scores = score_batch(
        context.region,
        source_calibration.current_states(),
        source_calibration.actions,
        source_calibration.outcomes,
    )
    grid_count = controlled["grid_trajectories"]
    total_count = controlled["calibration_trajectories"]
    cot_indices = torch.arange(grid_count, device=source_calibration.actions.device)
    cert_indices = torch.arange(
        grid_count,
        total_count,
        device=source_calibration.actions.device,
    )
    cot_batch = source_calibration.subset(cot_indices)
    certification_batch = source_calibration.subset(cert_indices)
    cot_scores = calibration_scores[:grid_count]
    certification_scores = calibration_scores[grid_count:]
    if certification_batch.n != controlled["certification_trajectories"]:
        raise RuntimeError("controlled D_cert size differs from frozen protocol")
    stage_grids = _committed_prefix_stage_grids(cot_scores, config)
    selections = select_strict_split_pair(
        cot_batch=cot_batch,
        cot_scores=cot_scores,
        certification_batch=certification_batch,
        certification_scores=certification_scores,
        stage_grids=stage_grids,
        target_policy=target_policy,
        logging_policy=context.logging_policy,
        outcome_model=context.outcome_model,
        outcome_sd=context.outcome_sd.to(device),
        target=1.0 - config.certification.alpha,
    )
    reference_noise = make_controlled_noise(
        n=controlled["reference_trajectories"],
        horizon=config.horizon,
        initial_count=environment.initial_count,
        seed=reference_rng,
        device=device,
    )
    variants = {}
    for variant in VARIANTS:
        selection = selections[variant]
        selection_batch_size = total_count if variant == "canonical" else certification_batch.n
        payload = selection_payload(
            selection,
            calibration_trajectories=selection_batch_size,
            calibration_roles=("D_COT", "D_cert") if variant == "canonical" else ("D_cert",),
        )
        if selection.radii is None:
            evaluation = evaluation_payload(
                coverage=None,
                normalized_width_by_stage=None,
                evaluation_trajectories=0,
                evaluation_rng=reference_rng,
            )
        else:
            target_reference = rollout_controlled(
                environment,
                target_policy,
                noise=reference_noise,
                gamma=controlled["gamma"],
                action_coordinate=action_coordinate,
                radii=selection.radii,
            ).trajectories
            target_scores = score_batch(
                context.region,
                target_reference.current_states(),
                target_reference.actions,
                target_reference.outcomes,
            )
            target_coverage = (
                target_scores <= selection.radii.to(target_scores)[None, :]
            ).float().mean(dim=0)
            target_width = _normalized_width_by_stage(
                context.outcome_model,
                target_reference,
                schedule=selection.radii,
                outcome_sd=context.outcome_sd.to(device),
            )
            evaluation = evaluation_payload(
                coverage=target_coverage,
                normalized_width_by_stage=target_width,
                evaluation_trajectories=target_reference.n,
                evaluation_rng=reference_rng,
            )
        variants[variant] = {**payload, "evaluation": evaluation}

    row = {
        "protocol": PROTOCOL,
        "setting": "controlled_gamma_minus_2",
        "seed": seed,
        "horizon": config.horizon,
        "base_config": str(config_path.relative_to(ROOT)),
        "gamma": controlled["gamma"],
        "q_low": q_low,
        "q_high": q_high,
        "stage_grid_roles": ["D_COT"],
        "stage_grid_sha256": tensor_sha256(stage_grids),
        "stage_grid_shape": list(stage_grids.shape),
        "matched_evaluation_crn": True,
        "calibration_rng": calibration_rng,
        "evaluation_rng": reference_rng,
        "split_sizes": {
            "D_COT": cot_batch.n,
            "D_cert": certification_batch.n,
            "D_env": environment_batch.n,
        },
        "variants": variants,
        "mechanism_scope": "frozen_same_kernel_semi_synthetic_calibration_stress",
        "claim_boundary": "robustness_only_asymptotic_per_step_marginal",
    }
    validate_result_row(row, setting="controlled_gamma_minus_2")
    return row


def write_task_artifact(
    row: dict[str, Any],
    *,
    output_dir: Path,
    task: tuple[str, int],
    device: str,
    manifest_hash: str,
    source_hash: str,
    config_contract_hash: str,
    environment_hash: str,
) -> Path:
    """Atomically publish a complete per-seed task directory."""

    setting, seed = task
    validate_result_row(row, setting=setting)
    if row["seed"] != seed:
        raise RuntimeError("task result seed differs from assigned seed")
    destination = _artifact_path(output_dir, task)
    if destination.exists():
        raise FileExistsError(f"strict-split artifact already exists: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        _write_json_file(temporary / "row.json", row)
        metadata = {
            "protocol": PROTOCOL,
            "setting": setting,
            "seed": seed,
            "device": device,
            "manifest_sha256": manifest_hash,
            "active_source_tree_sha256": source_hash,
            "active_config_contract_sha256": config_contract_hash,
            "runtime_environment_sha256": environment_hash,
            "row_sha256": _file_sha256(temporary / "row.json"),
        }
        _write_json_file(temporary / "metadata.json", metadata)
        _write_json_file(
            temporary / "COMPLETE",
            {
                "status": "complete",
                "protocol": PROTOCOL,
                "setting": setting,
                "seed": seed,
                "manifest_sha256": manifest_hash,
                "row_sha256": metadata["row_sha256"],
                "metadata_sha256": _file_sha256(temporary / "metadata.json"),
            },
        )
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _artifact_path(output_dir: Path, task: tuple[str, int]) -> Path:
    setting, seed = task
    return output_dir / setting / f"seed_{seed:05d}"


def _completed_tasks(
    output_dir: Path,
    *,
    tasks: tuple[tuple[str, int], ...],
    manifest_hash: str,
    device_mapping: dict[tuple[str, int], str],
    source_hash: str,
    config_contract_hash: str,
    environment_hash: str,
) -> set[tuple[str, int]]:
    completed = set()
    for task in tasks:
        path = _artifact_path(output_dir, task)
        if not path.exists():
            continue
        _load_valid_artifact(
            path,
            setting=task[0],
            seed=task[1],
            expected_device=device_mapping[task],
            manifest_hash=manifest_hash,
            source_hash=source_hash,
            config_contract_hash=config_contract_hash,
            environment_hash=environment_hash,
        )
        completed.add(task)
    return completed


def _load_valid_artifact(
    path: Path,
    *,
    setting: str,
    seed: int,
    expected_device: str,
    manifest_hash: str,
    source_hash: str,
    config_contract_hash: str,
    environment_hash: str,
) -> dict[str, Any]:
    expected_files = {"row.json", "metadata.json", "COMPLETE"}
    if not path.is_dir() or {item.name for item in path.iterdir()} != expected_files:
        raise RuntimeError(f"strict-split artifact is malformed: {path}")
    row = _read_json(path / "row.json")
    metadata = _read_json(path / "metadata.json")
    complete = _read_json(path / "COMPLETE")
    validate_result_row(row, setting=setting)
    if row["seed"] != seed:
        raise RuntimeError(f"strict-split artifact seed differs: {path}")
    expected_metadata = {
        "protocol": PROTOCOL,
        "setting": setting,
        "seed": seed,
        "device": expected_device,
        "manifest_sha256": manifest_hash,
        "active_source_tree_sha256": source_hash,
        "active_config_contract_sha256": config_contract_hash,
        "runtime_environment_sha256": environment_hash,
        "row_sha256": _file_sha256(path / "row.json"),
    }
    if metadata != expected_metadata:
        raise RuntimeError(f"strict-split metadata contract differs: {path}")
    expected_complete = {
        "status": "complete",
        "protocol": PROTOCOL,
        "setting": setting,
        "seed": seed,
        "manifest_sha256": manifest_hash,
        "row_sha256": metadata["row_sha256"],
        "metadata_sha256": _file_sha256(path / "metadata.json"),
    }
    if complete != expected_complete:
        raise RuntimeError(f"strict-split COMPLETE contract differs: {path}")
    return row


def _reject_unexpected_artifacts(
    output_dir: Path,
    tasks: tuple[tuple[str, int], ...],
) -> None:
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise RuntimeError("strict-split output root must be one real directory")
    allowed_files = {
        "manifest.json",
        "artifact_manifest.json",
        "summary.json",
        "COMPLETE",
    }
    for child in output_dir.iterdir():
        if child.name in SETTINGS:
            if not child.is_dir() or child.is_symlink():
                raise RuntimeError(
                    f"strict-split setting root is not a real directory: {child}"
                )
            continue
        if child.name in allowed_files:
            if not child.is_file() or child.is_symlink():
                raise RuntimeError(
                    f"strict-split root artifact is not a real file: {child}"
                )
            continue
        raise RuntimeError(f"unexpected strict-split root child: {child}")

    missing_settings = [
        setting for setting in SETTINGS if not (output_dir / setting).is_dir()
    ]
    if missing_settings:
        raise RuntimeError(
            f"strict-split setting directories differ from protocol: {missing_settings}"
        )

    expected_by_setting = {
        setting: {
            _artifact_path(output_dir, task).name
            for task in tasks
            if task[0] == setting
        }
        for setting in SETTINGS
    }
    for setting in SETTINGS:
        setting_root = output_dir / setting
        expected_names = expected_by_setting[setting]
        for child in setting_root.iterdir():
            if child.name not in expected_names:
                raise RuntimeError(
                    f"unexpected strict-split setting child: {child}"
                )
            if not child.is_dir() or child.is_symlink():
                raise RuntimeError(
                    f"strict-split seed artifact is not a real directory: {child}"
                )


def _validate_complete_bundle(
    output_dir: Path,
    *,
    manifest_hash: str,
    task_count: int,
    parent_snapshot: dict[str, Any],
    active_source_hash: str,
) -> None:
    """Validate an already-complete resume target without rewriting it."""

    required = {
        "manifest.json",
        "artifact_manifest.json",
        "summary.json",
        "COMPLETE",
    }
    if not all((output_dir / name).is_file() for name in required):
        raise RuntimeError("complete strict-split bundle is missing root artifacts")
    artifact_manifest = _read_json(output_dir / "artifact_manifest.json")
    summary = _read_json(output_dir / "summary.json")
    complete = _read_json(output_dir / "COMPLETE")
    expected_complete = {
        "status": "complete",
        "protocol": PROTOCOL,
        "task_count": task_count,
        "manifest_sha256": manifest_hash,
        "summary_sha256": _file_sha256(output_dir / "summary.json"),
        "artifact_manifest_sha256": _file_sha256(
            output_dir / "artifact_manifest.json"
        ),
        "parent_snapshot_manifest_sha256": parent_snapshot["manifest_sha256"],
        "parent_snapshot_archive_sha256": parent_snapshot["archive_sha256"],
        "parent_source_tree_sha256": parent_snapshot["source_tree_sha256"],
        "active_source_tree_sha256": active_source_hash,
    }
    if complete != expected_complete:
        raise RuntimeError("complete strict-split root contract differs")
    if (
        artifact_manifest.get("protocol") != PROTOCOL
        or artifact_manifest.get("manifest_sha256") != manifest_hash
        or artifact_manifest.get("task_count") != task_count
        or len(artifact_manifest.get("artifacts", ())) != task_count
    ):
        raise RuntimeError("strict-split artifact manifest differs")
    if (
        summary.get("protocol") != PROTOCOL
        or summary.get("manifest_sha256") != manifest_hash
        or summary.get("artifact_manifest_sha256")
        != _file_sha256(output_dir / "artifact_manifest.json")
        or summary.get("parent_snapshot") != parent_snapshot
        or summary.get("active_source_tree_sha256") != active_source_hash
    ):
        raise RuntimeError("strict-split complete summary contract differs")


def active_config_contract(protocol_config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "strict_split": CONFIG_PATH,
        "synthetic_main": ROOT / protocol_config["settings"]["synthetic_main"]["config"],
        "mimic_iv": ROOT / protocol_config["settings"]["mimic_iv"]["config"],
    }
    files = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    controlled_config = ExperimentConfig.from_yaml(paths["mimic_iv"])
    controlled_config = replace(
        controlled_config,
        policy=replace(
            controlled_config.policy,
            policy_ratio_cap=protocol_config["settings"]["controlled_gamma_minus_2"][
                "policy_ratio_cap"
            ],
        ),
    )
    contract = {
        "files": files,
        "protocol_config_sha256": json_sha256(protocol_config),
        "controlled_active_config": controlled_config.to_dict(),
        "controlled_active_config_sha256": json_sha256(controlled_config.to_dict()),
    }
    return {**contract, "contract_sha256": json_sha256(contract)}


def validate_parent_snapshot() -> dict[str, Any]:
    if not PARENT_MANIFEST.is_file() or not PARENT_ARCHIVE.is_file():
        raise FileNotFoundError("strict-split audit requires the parent formal snapshot")
    manifest_hash = _file_sha256(PARENT_MANIFEST)
    archive_hash = _file_sha256(PARENT_ARCHIVE)
    if manifest_hash != PARENT_MANIFEST_SHA256:
        raise RuntimeError("parent snapshot manifest SHA256 differs")
    if archive_hash != PARENT_ARCHIVE_SHA256:
        raise RuntimeError("parent snapshot archive SHA256 differs")
    if PARENT_ARCHIVE.stat().st_size != PARENT_ARCHIVE_BYTES:
        raise RuntimeError("parent snapshot archive size differs")
    manifest = _read_json(PARENT_MANIFEST)
    expected = {
        "schema_version": 1,
        "role": "content_addressed_formal_source_snapshot",
        "source_tree_sha256": PARENT_SOURCE_SHA256,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(f"parent snapshot {field} differs")
    archive = manifest.get("archive")
    if not isinstance(archive, dict) or archive.get("sha256") != PARENT_ARCHIVE_SHA256:
        raise RuntimeError("parent snapshot archive contract differs")
    if archive.get("bytes") != PARENT_ARCHIVE_BYTES:
        raise RuntimeError("parent snapshot archive byte contract differs")
    return {
        "role": "parent_formal_source_snapshot",
        "manifest_path": str(PARENT_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": manifest_hash,
        "archive_path": str(PARENT_ARCHIVE.relative_to(ROOT)),
        "archive_sha256": archive_hash,
        "archive_bytes": PARENT_ARCHIVE_BYTES,
        "source_tree_sha256": PARENT_SOURCE_SHA256,
        "git_revision": manifest.get("git_revision"),
        "relationship": (
            "strict-split is post-snapshot robustness work; its active source is "
            "bound separately and is not claimed to be inside the parent archive"
        ),
    }


def runtime_environment() -> dict[str, Any]:
    numpy_configuration = getattr(np.__config__, "CONFIG", {})
    build_dependencies = numpy_configuration.get("Build Dependencies", {})
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "numpy": {
            "version": np.__version__,
            "blas": build_dependencies.get("blas"),
            "lapack": build_dependencies.get("lapack"),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "git_version": torch.version.git_version,
        },
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _assert_active_contract_unchanged(
    *,
    source_hash: str,
    config_contract: dict[str, Any],
    parent_snapshot: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    if source_tree_sha256() != source_hash:
        raise RuntimeError("active source changed during strict-split execution")
    if active_config_contract(load_frozen_config()) != config_contract:
        raise RuntimeError("scientific config changed during strict-split execution")
    if validate_parent_snapshot() != parent_snapshot:
        raise RuntimeError("parent formal snapshot changed during strict-split execution")
    if runtime_environment() != environment:
        raise RuntimeError("runtime environment changed during strict-split execution")


def _assert_seed_contract(config: dict[str, Any]) -> None:
    if setting_seeds(config, "synthetic_main") != tuple(range(1000, 1100)):
        raise RuntimeError("synthetic strict-split seed bank differs")
    if setting_seeds(config, "mimic_iv") != tuple(range(20)):
        raise RuntimeError("MIMIC-IV strict-split seed bank differs")
    if setting_seeds(config, "controlled_gamma_minus_2") != CONTROLLED_BASE_SEEDS:
        raise RuntimeError("controlled strict-split seed bank differs")
    if config["summary"]["bootstrap_rng"] != BOOTSTRAP_RNG:
        raise RuntimeError("strict-split bootstrap RNG differs")


def controlled_rng_mapping() -> dict[str, int]:
    """Enumerate every new controlled task/model/behavior/cal/evaluation stream."""

    mapping: dict[str, int] = {"summary/bootstrap": BOOTSTRAP_RNG}
    for seed in CONTROLLED_BASE_SEEDS:
        prefix = f"controlled/base_{seed}"
        mapping[f"{prefix}/task"] = seed
        mapping[f"{prefix}/outcome_model"] = seed + 1
        mapping[f"{prefix}/behavior_model"] = seed + 2
        mapping[f"{prefix}/calibration"] = _paper_seed(seed, 1_700_101)
        mapping[f"{prefix}/reference"] = _paper_seed(seed, 1_700_401)
    if len(mapping) != 101 or len(set(mapping.values())) != 101:
        raise RuntimeError("controlled strict-split RNG mapping is not internally unique")
    return mapping


def audit_fresh_controlled_rng_ids(
    *,
    output_dir: Path,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Reject prior-use or coordinated-reservation collisions before launch."""

    artifact_root = ROOT / "results" if artifact_root is None else artifact_root
    source_root = ROOT if source_root is None else source_root
    mapping = controlled_rng_mapping()
    new_ids = set(mapping.values())
    artifact_ids = _artifact_rng_ids(artifact_root, excluded_root=output_dir)
    excluded_paths = {
        Path(__file__).resolve(),
        (ROOT / "src" / "scpcp" / "strict_split_robustness.py").resolve(),
        CONFIG_PATH.resolve(),
    }
    source_ids = _source_actual_rng_ids(source_root, excluded_paths=excluded_paths)
    external_ids = set().union(
        *(set(values) for values in COORDINATED_EXTERNAL_RESERVATIONS.values())
    )
    prior_ids = artifact_ids | source_ids | external_ids
    collisions = {
        label: rng_id
        for label, rng_id in mapping.items()
        if rng_id in prior_ids
    }
    result = {
        "status": "passed_before_launch" if not collisions else "collision",
        "fresh_base_seeds": list(CONTROLLED_BASE_SEEDS),
        "fresh_base_seed_spacing": 10,
        "bootstrap_rng": BOOTSTRAP_RNG,
        "formal_rng_id_count": len(new_ids),
        "formal_rng_ids": sorted(new_ids),
        "formal_rng_id_sha256": _integer_set_sha256(new_ids),
        "formal_rng_mapping": mapping,
        "formal_rng_mapping_sha256": json_sha256(mapping),
        "internal_rng_streams_unique": True,
        "artifact_actual_rng_id_count": len(artifact_ids),
        "artifact_actual_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "source_actual_rng_id_count": len(source_ids),
        "source_actual_rng_id_sha256": _integer_set_sha256(source_ids),
        "source_actual_use_excludes_reservation_declarations": True,
        "coordinated_external_reservations": {
            name: f"{values.start}..{values.stop - 1}"
            for name, values in COORDINATED_EXTERNAL_RESERVATIONS.items()
        },
        "coordinated_external_rng_id_count": len(external_ids),
        "coordinated_external_rng_id_sha256": _integer_set_sha256(external_ids),
        "collision_count": len(collisions),
        "collisions": collisions,
        "excluded_output": str(output_dir.resolve()),
        "legacy_main_rng_design": (
            "synthetic 1000:1100 and MIMIC-IV 0:20 intentionally reuse the "
            "frozen main-study mapping and are not audited as fresh streams"
        ),
    }
    result["audit_sha256"] = json_sha256(result)
    if collisions:
        raise RuntimeError(f"fresh strict-split RNG IDs collide with prior use: {collisions}")
    return result


def _artifact_rng_ids(root: Path, *, excluded_root: Path) -> set[int]:
    values: set[int] = set()
    if not root.exists():
        return values
    excluded = excluded_root.resolve()
    for path in root.rglob("*"):
        resolved = path.resolve()
        if _is_relative_to(resolved, excluded):
            continue
        match = ARTIFACT_ID.fullmatch(path.name)
        if match:
            values.add(int(match.group(1)))
        if not path.is_file() or path.name not in PROVENANCE_NAMES:
            continue
        try:
            if path.name == "COMPLETE" and not path.read_text().lstrip().startswith(("{", "[")):
                continue
            payload = (
                yaml.safe_load(path.read_text())
                if path.suffix in {".yaml", ".yml"} or path.name == "config.yaml"
                else json.loads(path.read_text())
            )
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
            raise RuntimeError(f"cannot audit RNG-bearing artifact {path}") from error
        _collect_named_rng_values(payload, values)
    return values


def _source_actual_rng_ids(
    root: Path,
    *,
    excluded_paths: set[Path],
) -> set[int]:
    values: set[int] = set()
    for directory in ("scripts", "src", "tools", "configs"):
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            resolved = path.resolve()
            if resolved in excluded_paths or not path.is_file():
                continue
            path_values: set[int] = set()
            if path.suffix == ".py":
                _collect_python_rng_assignments(path, path_values)
            elif path.suffix in {".yaml", ".yml"}:
                try:
                    payload = yaml.safe_load(path.read_text())
                except (OSError, yaml.YAMLError) as error:
                    raise RuntimeError(f"cannot audit source RNG declarations in {path}") from error
                _collect_named_rng_values(payload, path_values)
            values.update(path_values)
    return values


def _collect_python_rng_assignments(path: Path, values: set[int]) -> None:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise RuntimeError(f"cannot parse RNG declarations in {path}") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [name for target in node.targets for name in _target_names(target)]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            names = list(_target_names(node.target))
            value = node.value
        else:
            continue
        if value is None or not any(RNG_KEY.search(name) for name in names):
            continue
        if any(RESERVATION_KEY.search(name) for name in names):
            continue
        evaluated = _literal_rng_expression(value)
        if evaluated is not None:
            values.update(evaluated)


def _target_names(target: ast.expr) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_names(element)


def _literal_rng_expression(node: ast.expr) -> set[int] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return {node.value}
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        parts = [_literal_rng_expression(element) for element in node.elts]
        if any(part is None for part in parts):
            return None
        return set().union(*(part or set() for part in parts))
    if isinstance(node, ast.Dict):
        parts = [_literal_rng_expression(value) for value in node.values]
        return set().union(*(part or set() for part in parts))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "range":
            arguments = [_literal_integer(argument) for argument in node.args]
            if any(argument is None for argument in arguments):
                return None
            return set(range(*(int(argument) for argument in arguments)))
        if node.func.id in {"tuple", "list", "set", "frozenset"} and len(node.args) == 1:
            return _literal_rng_expression(node.args[0])
    return None


def _literal_integer(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _literal_integer(node.operand)
        return None if operand is None else -operand
    return None


def _collect_named_rng_values(
    value: object,
    output: set[int],
    key_path: str = "",
) -> None:
    if RESERVATION_KEY.search(key_path):
        return
    if isinstance(value, dict):
        if RNG_KEY.search(key_path) and {"start", "stop"} <= set(value):
            start, stop = value["start"], value["stop"]
            step = value.get("step", 1)
            if all(isinstance(item, int) for item in (start, stop, step)):
                output.update(range(start, stop, step))
        for key, child in value.items():
            path = f"{key_path}.{key}" if key_path else str(key)
            _collect_named_rng_values(child, output, path)
        return
    if isinstance(value, list):
        for child in value:
            _collect_named_rng_values(child, output, key_path)
        return
    if RNG_KEY.search(key_path) and isinstance(value, int) and not isinstance(value, bool):
        output.add(value)


def run_engineering_smoke(
    output_dir: Path,
    *,
    device: str,
    argv: tuple[str, ...] = (),
) -> None:
    """Run one reduced nonformal synthetic task without opening science seeds."""

    if output_dir.exists():
        raise FileExistsError(f"engineering smoke output exists: {output_dir}")
    smoke_seed = 13_701
    base = ExperimentConfig.from_yaml(ROOT / "configs" / "per_step_synthetic_tail_shift.yaml")
    config = replace(
        base,
        model=replace(
            base.model,
            hidden_dim=32,
            representation_dim=16,
            epochs=3,
            patience=2,
        ),
        samples=replace(
            base.samples,
            logged=1_000,
            oracle_rollouts=1_000,
        ),
        horizon=4,
        q_grid_size=21,
    )
    torch.manual_seed(smoke_seed)
    context = _prepare_experiment_context(config, seed=smoke_seed, device=device)
    splits = context.task.splits
    cot_scores = score_batch(
        context.region,
        splits.cot.current_states(),
        splits.cot.actions,
        splits.cot.outcomes,
    )
    cert_scores = score_batch(
        context.region,
        splits.certification.current_states(),
        splits.certification.actions,
        splits.certification.outcomes,
    )
    grids = _committed_prefix_stage_grids(cot_scores, config)
    selections = select_strict_split_pair(
        cot_batch=splits.cot,
        cot_scores=cot_scores,
        certification_batch=splits.certification,
        certification_scores=cert_scores,
        stage_grids=grids,
        target_policy=context.policy,
        logging_policy=context.logging_policy,
        outcome_model=context.outcome_model,
        outcome_sd=context.outcome_sd,
        target=1.0 - config.certification.alpha,
    )
    evaluation_rng = _paper_seed(smoke_seed, 900_001)
    evaluations = {}
    for variant in VARIANTS:
        selection = selections[variant]
        if selection.radii is None:
            evaluations[variant] = {
                "evaluated": False,
                "coverage_by_stage": [],
                "mean_normalized_width": None,
            }
            continue
        record = _evaluate_radius_method(
            f"SC-PCP-{variant}-engineering-smoke",
            selection.radii,
            context.task,
            context.policy,
            context.region,
            config,
            evaluation_rng,
            device,
            outcome_sd=context.outcome_sd,
        )
        evaluations[variant] = {
            "evaluated": True,
            "coverage_by_stage": json.loads(record["per_time_coverage"]),
            "mean_normalized_width": float(record["average_normalized_width"]),
        }
    output_dir.mkdir(parents=True)
    _atomic_write_json(
        output_dir / "smoke.json",
        {
            "protocol": PROTOCOL,
            "role": "engineering_smoke_nonformal",
            "seed": smoke_seed,
            "science_seed_opened": False,
            "device": device,
            "argv": list(argv),
            "matched_evaluation_crn": True,
            "evaluation_rng": evaluation_rng,
            "evaluation_trajectories_per_available_variant": config.samples.oracle_rollouts,
            "stage_grid_sha256": tensor_sha256(grids),
            "selection_available": {
                variant: selections[variant].selection_available
                for variant in VARIANTS
            },
            "selected_indices": {
                variant: list(selections[variant].selected_indices)
                for variant in VARIANTS
            },
            "evaluations": evaluations,
        },
    )
    _atomic_write_json(
        output_dir / "COMPLETE",
        {"status": "complete", "role": "engineering_smoke_nonformal"},
    )


def _integer_set_sha256(values: Iterable[int]) -> str:
    encoded = json.dumps(sorted(set(values)), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return payload


def _write_json_file(path: Path, value: object) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_write_json(path: Path, value: object) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}-",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    main()
