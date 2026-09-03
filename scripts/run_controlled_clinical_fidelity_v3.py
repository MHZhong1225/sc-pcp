"""Run coverage-blind clinical-v3 K0 development or confirmation.

Examples
--------
Development reuses the frozen v2 base seeds only as a K0 tuning bank::

    python scripts/run_controlled_clinical_fidelity_v3.py development \
      --devices cuda:0,cuda:1 \
      --output-root results/work/controlled_clinical_fidelity_v3_development

Confirmation accepts exactly one already-frozen theta per dataset::

    python scripts/run_controlled_clinical_fidelity_v3.py confirmation \
      --devices cuda:0,cuda:1 \
      --development-root results/work/controlled_clinical_fidelity_v3_development \
      --output-root results/work/controlled_clinical_fidelity_v3_confirmation

This module has no coverage or science execution path.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import tarfile
from typing import Any, Callable, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_controlled_clinical_extension as v2  # noqa: E402
import scripts.run_controlled_six_method_benchmark as six  # noqa: E402
from scpcp.artifacts import experiment_tree_sha256  # noqa: E402
from scpcp.controlled_clinical_extension import (  # noqa: E402
    METHODS,
    ControlledClinicalExtensionConfig,
    DatasetPreset,
)
from scpcp.controlled_clinical_fidelity_v3 import (  # noqa: E402
    DATASETS,
    METRIC_THRESHOLDS,
    PROTOCOL,
    REPAIRED_DATASETS,
    RIDGE_ORDER,
    SELECTOR_VERSION,
    CandidateDatasetSummary,
    FidelityV3Config,
    KernelTheta,
    load_fidelity_v3_config,
    select_dataset_candidate,
    select_shared_candidate,
    stage_a_candidates,
    stage_b_candidates,
    summarize_candidate_dataset,
)
from scpcp.controlled_transition import ControlledResidualEnvironment  # noqa: E402
from scpcp.scores import score_batch  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "controlled_clinical_fidelity_v3.yaml"
V2_CONFIG_PATH = ROOT / "configs" / "controlled_clinical_extension.yaml"
DEVELOPMENT_ROOT = (
    ROOT / "results/work/controlled_clinical_fidelity_v3_development"
).resolve()
CONFIRMATION_ROOT = (
    ROOT / "results/work/controlled_clinical_fidelity_v3_confirmation"
).resolve()
PHASES = ("audit", "development", "confirmation")
FORBIDDEN_PARENT_PATH_TOKENS = (
    "science",
    "coverage",
    "width",
    "method_selection",
)
_OWN_RNG_DECLARATION_PATHS = {
    Path(__file__).resolve(),
    (ROOT / "src/scpcp/controlled_clinical_fidelity_v3.py").resolve(),
    CONFIG_PATH.resolve(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    config = load_fidelity_v3_config(CONFIG_PATH)
    parent_root = (ROOT / config.parent_v2_root).resolve()
    parent_binding = verify_parent_v2(parent_root)
    excluded_roots = tuple(
        path.resolve()
        for path in (args.output_root, args.development_root)
        if path is not None
    )
    rng_audit = audit_confirmation_rng(config, excluded_roots=excluded_roots)
    if args.phase == "audit":
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "parent_binding_sha256": _json_sha256(parent_binding),
                    "confirmation_rng_audit": rng_audit,
                    "coverage_generation_permitted": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output_root is None:
        parser.error("--output-root is required outside audit")
    _validate_devices(devices)
    output_root = args.output_root.resolve()
    if args.phase == "development":
        if args.development_root is not None:
            parser.error("development does not accept --development-root")
        if output_root != DEVELOPMENT_ROOT:
            parser.error(f"development output root is frozen to {DEVELOPMENT_ROOT}")
        run_development(
            output_root,
            config=config,
            devices=devices,
            parent_binding=parent_binding,
            rng_audit=rng_audit,
            resume=args.resume,
        )
    else:
        if args.development_root is None:
            parser.error("confirmation requires --development-root")
        if args.development_root.resolve() != DEVELOPMENT_ROOT:
            parser.error(f"development root is frozen to {DEVELOPMENT_ROOT}")
        if output_root != CONFIRMATION_ROOT:
            parser.error(f"confirmation output root is frozen to {CONFIRMATION_ROOT}")
        run_confirmation(
            output_root,
            development_root=args.development_root.resolve(),
            config=config,
            devices=devices,
            parent_binding=parent_binding,
            rng_audit=rng_audit,
            resume=args.resume,
        )
    print(output_root)


def run_development(
    output_root: Path,
    *,
    config: FidelityV3Config,
    devices: tuple[str, ...],
    parent_binding: Mapping[str, Any],
    rng_audit: Mapping[str, Any],
    resume: bool,
) -> None:
    """Evaluate the frozen K0 grid and publish exactly one development decision."""

    v2_protocol = v2.load_extension_config(V2_CONFIG_PATH)
    source_hash, source_snapshot = _active_source_contract()
    metadata = _root_metadata(
        phase="development",
        output_root=output_root,
        config=config,
        devices=devices,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        parent_binding=parent_binding,
        rng_audit=rng_audit,
    )
    _prepare_root(output_root, metadata, source_snapshot, resume=resume)
    if _complete_and_valid(output_root, metadata):
        return

    stage_a = stage_a_candidates()
    stage_a_rows: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        preset = replace(
            v2_protocol.datasets[dataset],
            seeds=config.development_seeds[dataset],
        )
        stage_a_rows[dataset] = _run_seed_phase(
            output_root / "stage_a" / dataset,
            phase="development_stage_a",
            preset=preset,
            devices=devices,
            candidates=stage_a,
            worker=_development_worker,
            worker_arguments=(v2_protocol,),
            source_hash=source_hash,
            resume=resume,
        )
    stage_a_summaries = _summarize_candidate_matrix(stage_a, stage_a_rows)
    shared_a = select_shared_candidate(stage_a, stage_a_summaries)
    fallback_a = {
        dataset: select_dataset_candidate(dataset, stage_a, stage_a_summaries)
        for dataset in DATASETS
    }
    stage_a_selection = {
        "protocol": PROTOCOL,
        "selector_version": SELECTOR_VERSION,
        "clarification": (
            "stagewise_zscore pools D_env representations over actions within each "
            "stage; donor retrieval remains action-conditional"
        ),
        "prelaunch_clarification_no_formal_seed_run": True,
        "shared": shared_a,
        "dataset_fallbacks_precomputed_before_stage_b": fallback_a,
        "candidate_dataset_summaries": _summary_matrix_to_dict(stage_a_summaries),
    }
    _write_json(output_root / "stage_a" / "selection.json", stage_a_selection)

    shared_a_theta = _theta_from_dict(shared_a["winner"])
    fallback_a_theta = {
        dataset: _theta_from_dict(record["winner"])
        for dataset, record in fallback_a.items()
    }
    stage_b_rows: dict[str, list[dict[str, Any]]] = {}
    stage_b_candidates_by_dataset: dict[str, tuple[KernelTheta, ...]] = {}
    for dataset in DATASETS:
        stage_a_roots = [shared_a_theta]
        if dataset in REPAIRED_DATASETS:
            stage_a_roots.append(fallback_a_theta[dataset])
        unique_roots = {
            candidate.stage_a_id: candidate for candidate in stage_a_roots
        }
        candidates = tuple(
            theta
            for candidate in unique_roots.values()
            for theta in stage_b_candidates(candidate)
        )
        stage_b_candidates_by_dataset[dataset] = candidates
        preset = replace(
            v2_protocol.datasets[dataset],
            seeds=config.development_seeds[dataset],
        )
        stage_b_rows[dataset] = _run_seed_phase(
            output_root / "stage_b" / dataset,
            phase="development_stage_b",
            preset=preset,
            devices=devices,
            candidates=candidates,
            worker=_development_worker,
            worker_arguments=(v2_protocol,),
            source_hash=source_hash,
            resume=resume,
        )
    stage_b_summaries = _summarize_ragged_candidate_matrix(
        stage_b_candidates_by_dataset,
        stage_b_rows,
    )
    decision = _development_decision(
        stage_a=stage_a,
        stage_a_summaries=stage_a_summaries,
        shared_a=shared_a_theta,
        fallback_a=fallback_a_theta,
        stage_b_summaries=stage_b_summaries,
    )
    decision["stage_a_selection_sha256"] = _json_sha256(stage_a_selection)
    decision["selector_contract_sha256"] = _selector_contract_sha256()
    _write_json(output_root / "stage_b" / "selection.json", decision["stage_b_selection"])
    _write_json(output_root / "FINAL_STATUS.json", decision)
    if decision["status"] != "DEVELOPMENT_NO_GO":
        frozen = {
            "protocol": PROTOCOL,
            "role": "frozen_before_fresh_confirmation",
            "status": decision["status"],
            "theta_by_dataset": decision["theta_by_dataset"],
            "selector_contract_sha256": decision["selector_contract_sha256"],
            "development_source_tree_sha256": source_hash,
            "development_config_sha256": metadata["config_sha256"],
            "parent_v2_binding_sha256": metadata["parent_v2_binding_sha256"],
            "development_decision_sha256": _json_sha256(decision),
            "coverage_generation_permitted": False,
        }
        frozen["frozen_theta_sha256"] = _json_sha256(frozen)
        _write_json(output_root / "frozen_theta.json", frozen)
    _finalize_root(
        output_root,
        metadata,
        source_hash=source_hash,
        config=config,
    )


def run_confirmation(
    output_root: Path,
    *,
    development_root: Path,
    config: FidelityV3Config,
    devices: tuple[str, ...],
    parent_binding: Mapping[str, Any],
    rng_audit: Mapping[str, Any],
    resume: bool,
) -> None:
    """Run support and K0 only on untouched splits under frozen theta."""

    development_binding, frozen = _verify_development_for_confirmation(
        development_root,
        current_parent_binding=parent_binding,
    )
    source_hash, source_snapshot = _active_source_contract()
    if source_hash != frozen["development_source_tree_sha256"]:
        raise RuntimeError("source changed after theta freeze; confirmation is blocked")
    v2_protocol = v2.load_extension_config(V2_CONFIG_PATH)
    metadata = _root_metadata(
        phase="confirmation",
        output_root=output_root,
        config=config,
        devices=devices,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        parent_binding=parent_binding,
        rng_audit=rng_audit,
        development_binding=development_binding,
        frozen_theta=frozen,
    )
    _prepare_root(output_root, metadata, source_snapshot, resume=resume)
    if _complete_and_valid(output_root, metadata):
        return

    dataset_status = {}
    for dataset in DATASETS:
        base_preset = v2_protocol.datasets[dataset]
        preset = replace(
            base_preset,
            seeds=config.confirmation_seeds[dataset],
            bootstrap_seed=config.confirmation_bootstrap_seeds[dataset],
        )
        theta = _theta_from_dict(frozen["theta_by_dataset"][dataset])
        support_rows = _run_seed_phase(
            output_root / dataset / "support",
            phase="confirmation_support",
            preset=preset,
            devices=devices,
            candidates=(),
            worker=_confirmation_support_worker,
            worker_arguments=(v2_protocol,),
            source_hash=source_hash,
            resume=resume,
        )
        support_count = sum(bool(row["passed"]) for row in support_rows)
        support_pass = support_count >= 19
        if support_pass:
            k0_rows = _run_seed_phase(
                output_root / dataset / "k0_fidelity",
                phase="confirmation_k0",
                preset=preset,
                devices=devices,
                candidates=(theta,),
                worker=_confirmation_k0_worker,
                worker_arguments=(v2_protocol,),
                source_hash=source_hash,
                resume=resume,
            )
            structural_count = sum(
                bool(row["metrics"]["structural_invariants"]) for row in k0_rows
            )
            k0_count = sum(bool(row["passed"]) for row in k0_rows)
        else:
            structural_count = 0
            k0_count = 0
        status = (
            "CONFIRMATION_GATE_GO"
            if support_pass and structural_count == 20 and k0_count >= 19
            else "CONFIRMATION_GATE_NO_GO"
        )
        dataset_status[dataset] = {
            "protocol": PROTOCOL,
            "dataset": dataset,
            "status": status,
            "support_pass_count": support_count,
            "structural_pass_count": structural_count,
            "k0_pass_count": k0_count,
            "prespecified_seed_count": 20,
            "theta": theta.to_dict(),
            "coverage_generated": False,
            "confirmation_label": "fresh_split_confirmation",
            "independent_patient_confirmation_claimed": False,
        }
        _write_json(output_root / dataset / "gate.json", dataset_status[dataset])
        _write_text(output_root / dataset / "COMPLETE", status.lower() + "\n")
    overall_go = all(
        value["status"] == "CONFIRMATION_GATE_GO"
        for value in dataset_status.values()
    )
    final = {
        "protocol": PROTOCOL,
        "phase": "confirmation",
        "status": "CONFIRMATION_GO" if overall_go else "CONFIRMATION_NO_GO",
        "datasets": dataset_status,
        "coverage_generated": False,
        "science_unlock_present": False,
        "failure_consequence": (
            None
            if overall_go
            else "archive v3; any retry requires a new protocol and seed bank"
        ),
    }
    _write_json(output_root / "FINAL_STATUS.json", final)
    _finalize_root(
        output_root,
        metadata,
        source_hash=source_hash,
        config=config,
    )


def _development_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    candidates: tuple[KernelTheta, ...],
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    if base_context.config.model.representation_dim != 32:
        raise RuntimeError("v3 candidate geometry requires learned representation dim 32")
    candidate_rows = []
    for theta in candidates:
        context = _context_with_theta(base_context, theta)
        metrics, detail = v2._logging_mixture_fidelity(
            context,
            seed=seed,
            protocol=protocol,
        )
        ratio = _normalized_ratio(asdict(metrics))
        candidate_rows.append(
            {
                "theta": theta.to_dict(),
                "metrics": asdict(metrics),
                "passed": v2.k0_fidelity_passes(
                    metrics, protocol.k0_fidelity_gate
                ),
                "normalized_seed_ratio": ratio if math.isfinite(ratio) else None,
                "structural_failure_ratio_is_infinite": not math.isfinite(ratio),
                "systematic_replay": detail,
                "context_identity": _candidate_context_identity(
                    base_context,
                    context.environment,
                    theta,
                ),
            }
        )
    return {
        "seed": seed,
        "dataset": preset.name,
        "phase": "development_k0_only",
        "candidate_count": len(candidates),
        "candidates": candidate_rows,
        "split_audit": v2._split_audit(base_context.splits),
        "coverage_generated": False,
        "information_opened": ["support", "k0_fidelity", "context_identity"],
    }


def _confirmation_support_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    candidates: tuple[KernelTheta, ...],
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    if candidates:
        raise RuntimeError("support must not receive a transition candidate")
    result = v2._support_worker(seed, preset, device, protocol)
    return {
        **result,
        "phase": "confirmation_support",
        "coverage_generated": False,
        "confirmation_label": "fresh_split_confirmation",
    }


def _confirmation_k0_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    candidates: tuple[KernelTheta, ...],
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    if len(candidates) != 1:
        raise RuntimeError("confirmation requires exactly one frozen theta")
    theta = candidates[0]
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    context = _context_with_theta(base_context, theta)
    metrics, detail = v2._logging_mixture_fidelity(
        context,
        seed=seed,
        protocol=protocol,
    )
    ratio = _normalized_ratio(asdict(metrics))
    return {
        "seed": seed,
        "dataset": preset.name,
        "phase": "confirmation_k0",
        "theta": theta.to_dict(),
        "metrics": asdict(metrics),
        "passed": v2.k0_fidelity_passes(metrics, protocol.k0_fidelity_gate),
        "normalized_seed_ratio": ratio if math.isfinite(ratio) else None,
        "structural_failure_ratio_is_infinite": not math.isfinite(ratio),
        "systematic_replay": detail,
        "context_identity": _candidate_context_identity(
            base_context,
            context.environment,
            theta,
        ),
        "split_audit": v2._split_audit(base_context.splits),
        "coverage_generated": False,
        "confirmation_label": "fresh_split_confirmation",
        "independent_patient_confirmation_claimed": False,
    }


def _context_with_theta(
    base_context: v2.ExtensionContext,
    theta: KernelTheta,
) -> v2.ExtensionContext:
    environment_scores = score_batch(
        base_context.region,
        base_context.splits.environment.current_states(),
        base_context.splits.environment.actions,
        base_context.splits.environment.outcomes,
    )
    environment = ControlledResidualEnvironment(
        base_context.splits.environment,
        outcome_model=base_context.outcome_model,
        n_actions=base_context.n_actions,
        difficulty=v2._empirical_rank_by_stage(environment_scores),
        history_length=base_context.config.model.history_length,
        static_indices=base_context.static_indices,
        state_feature_names=base_context.state_feature_names,
        neighbors=theta.neighbors,
        bandwidth=theta.bandwidth,
        ridge=theta.ridge_value,
        representation_geometry=theta.metric,
        donor_weighting=theta.donor_weighting,
        ridge_mode=theta.ridge_mode,
    )
    return replace(base_context, environment=environment)


def _candidate_context_identity(
    base_context: v2.ExtensionContext,
    environment: ControlledResidualEnvironment,
    theta: KernelTheta,
) -> dict[str, Any]:
    base_identity = v2._context_identity(base_context)
    centers = torch.stack(
        [environment._metric_transforms[stage][0] for stage in range(environment.horizon)]
    ).to(torch.float64)
    scales = torch.stack(
        [environment._metric_transforms[stage][1] for stage in range(environment.horizon)]
    ).to(torch.float64)
    transform = {
        "geometry": theta.metric,
        "source": (
            "D_env_only" if theta.metric == "stagewise_zscore" else "identity"
        ),
        "pooling": (
            "per_stage_pooled_over_actions"
            if theta.metric == "stagewise_zscore"
            else "not_applicable"
        ),
        "retrieval": (
            "action_conditional_after_shared_stage_scaling"
            if theta.metric == "stagewise_zscore"
            else "action_conditional_raw_geometry"
        ),
        "coordinate_count": int(centers.shape[1]),
        "stage_count": int(centers.shape[0]),
        "estimation_dtype": "float64",
        "population_sd": True,
        "sd_floor": 1e-4,
        "center_sha256": _tensor_sha256(centers),
        "scale_sha256": _tensor_sha256(scales),
    }
    identity = {
        "base_nuisance_context_sha256": base_identity["combined_sha256"],
        "outcome_model_state_sha256": base_identity["outcome_model_state_sha256"],
        "behavior_policy_state_sha256": base_identity[
            "behavior_policy_state_sha256"
        ],
        "split_patient_id_sha256": base_identity["split_patient_id_sha256"],
        "active_config_sha256": base_identity["active_config_sha256"],
        "theta": theta.to_dict(),
        "metric_transform": transform,
    }
    return {**identity, "combined_sha256": _json_sha256(identity)}


def _run_seed_phase(
    phase_root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    devices: tuple[str, ...],
    candidates: tuple[KernelTheta, ...],
    worker: Callable[..., dict[str, Any]],
    worker_arguments: tuple[object, ...],
    source_hash: str,
    resume: bool,
) -> list[dict[str, Any]]:
    phase_root.mkdir(parents=True, exist_ok=True)
    mapping = _dataset_seed_device_mapping(
        preset.name,
        preset.seeds,
        devices,
    )
    candidate_hash = _json_sha256([candidate.to_dict() for candidate in candidates])
    expected = {phase_root / f"seed_{seed:06d}.json" for seed in preset.seeds}
    unexpected = set(phase_root.glob("seed_*.json")) - expected
    if unexpected:
        raise RuntimeError(f"unexpected {phase} artifacts: {sorted(unexpected)}")
    completed = {}
    for seed in preset.seeds:
        path = phase_root / f"seed_{seed:06d}.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        _validate_seed_payload(
            payload,
            phase=phase,
            preset=preset,
            seed=seed,
            device=mapping[seed],
            source_hash=source_hash,
            candidate_hash=candidate_hash,
        )
        completed[seed] = payload["result"]
    pending = tuple(seed for seed in preset.seeds if seed not in completed)
    if pending and (phase_root / "COMPLETE").exists():
        raise RuntimeError(f"{phase} COMPLETE exists with missing seeds")
    if pending:
        groups = tuple(
            tuple(seed for seed in pending if mapping[seed] == device)
            for device in devices
        )
        with ProcessPoolExecutor(
            max_workers=len(devices),
            mp_context=get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(
                    _phase_group,
                    group,
                    device,
                    preset,
                    candidates,
                    worker,
                    worker_arguments,
                ): device
                for group, device in zip(groups, devices, strict=True)
                if group
            }
            for future in as_completed(futures):
                for seed, device, result in future.result():
                    payload = {
                        "protocol": PROTOCOL,
                        "phase": phase,
                        "dataset": preset.name,
                        "seed": seed,
                        "device": device,
                        "source_tree_sha256": source_hash,
                        "candidate_contract_sha256": candidate_hash,
                        "result": result,
                    }
                    _validate_seed_payload(
                        payload,
                        phase=phase,
                        preset=preset,
                        seed=seed,
                        device=device,
                        source_hash=source_hash,
                        candidate_hash=candidate_hash,
                    )
                    _write_json(phase_root / f"seed_{seed:06d}.json", payload)
                    completed[seed] = result
    if set(completed) != set(preset.seeds):
        raise RuntimeError(f"{phase} did not complete its full seed bank")
    _write_text(phase_root / "COMPLETE", "complete\n")
    return [completed[seed] for seed in preset.seeds]


def _phase_group(
    seeds: tuple[int, ...],
    device: str,
    preset: DatasetPreset,
    candidates: tuple[KernelTheta, ...],
    worker: Callable[..., dict[str, Any]],
    worker_arguments: tuple[object, ...],
) -> list[tuple[int, str, dict[str, Any]]]:
    torch.cuda.set_device(torch.device(device))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    rows = []
    for seed in seeds:
        rows.append(
            (
                seed,
                device,
                worker(seed, preset, device, candidates, *worker_arguments),
            )
        )
        torch.cuda.empty_cache()
    return rows


def _summarize_candidate_matrix(
    candidates: Sequence[KernelTheta],
    rows_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, CandidateDatasetSummary]]:
    expected_ids = tuple(candidate.theta_id for candidate in candidates)
    summaries: dict[str, dict[str, CandidateDatasetSummary]] = {
        candidate_id: {} for candidate_id in expected_ids
    }
    for dataset in DATASETS:
        rows = rows_by_dataset[dataset]
        if len(rows) != 20:
            raise RuntimeError(f"{dataset} candidate matrix is not 20 seeds")
        for candidate_index, candidate_id in enumerate(expected_ids):
            metrics = []
            for row in rows:
                candidate_rows = row.get("candidates")
                if (
                    not isinstance(candidate_rows, list)
                    or tuple(item["theta"]["theta_id"] for item in candidate_rows)
                    != expected_ids
                ):
                    raise RuntimeError(f"{dataset} candidate order differs")
                metrics.append(candidate_rows[candidate_index]["metrics"])
            summaries[candidate_id][dataset] = summarize_candidate_dataset(
                candidate_id,
                dataset,
                metrics,
            )
    return summaries


def _summarize_ragged_candidate_matrix(
    candidates_by_dataset: Mapping[str, Sequence[KernelTheta]],
    rows_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, CandidateDatasetSummary]]:
    summaries: dict[str, dict[str, CandidateDatasetSummary]] = {}
    for dataset in DATASETS:
        candidates = tuple(candidates_by_dataset[dataset])
        local = _summarize_one_dataset_candidates(
            dataset,
            candidates,
            rows_by_dataset[dataset],
        )
        for candidate_id, summary in local.items():
            summaries.setdefault(candidate_id, {})[dataset] = summary
    return summaries


def _summarize_one_dataset_candidates(
    dataset: str,
    candidates: Sequence[KernelTheta],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, CandidateDatasetSummary]:
    if len(rows) != 20:
        raise RuntimeError(f"{dataset} Stage-B matrix is not 20 seeds")
    expected_ids = tuple(candidate.theta_id for candidate in candidates)
    output = {}
    for candidate_index, candidate_id in enumerate(expected_ids):
        metrics = []
        for row in rows:
            candidate_rows = row.get("candidates")
            if (
                not isinstance(candidate_rows, list)
                or tuple(item["theta"]["theta_id"] for item in candidate_rows)
                != expected_ids
            ):
                raise RuntimeError(f"{dataset} Stage-B candidate order differs")
            metrics.append(candidate_rows[candidate_index]["metrics"])
        output[candidate_id] = summarize_candidate_dataset(
            candidate_id,
            dataset,
            metrics,
        )
    return output


def _development_decision(
    *,
    stage_a: Sequence[KernelTheta],
    stage_a_summaries: Mapping[str, Mapping[str, CandidateDatasetSummary]],
    shared_a: KernelTheta,
    fallback_a: Mapping[str, KernelTheta],
    stage_b_summaries: Mapping[str, Mapping[str, CandidateDatasetSummary]],
) -> dict[str, Any]:
    shared_b_candidates = stage_b_candidates(shared_a)
    shared_b = select_shared_candidate(shared_b_candidates, stage_b_summaries)
    shared_theta = _theta_from_dict(shared_b["winner"])
    shared_counts = {
        dataset: stage_b_summaries[shared_theta.theta_id][dataset].pass_count
        for dataset in DATASETS
    }
    shared_admissible = (
        shared_counts["mimic_iv"] == 20
        and all(shared_counts[dataset] >= 19 for dataset in REPAIRED_DATASETS)
    )

    fallback_b = {}
    fallback_theta = {}
    fallback_counts = {}
    for dataset in REPAIRED_DATASETS:
        candidates = stage_b_candidates(fallback_a[dataset])
        record = select_dataset_candidate(dataset, candidates, stage_b_summaries)
        theta = _theta_from_dict(record["winner"])
        fallback_b[dataset] = record
        fallback_theta[dataset] = theta
        fallback_counts[dataset] = stage_b_summaries[theta.theta_id][
            dataset
        ].pass_count
    v2_default = stage_a[0]
    if v2_default.theta_id != "A00_raw_k100_gaussian_b2__raw_ridge_1e-3":
        raise RuntimeError("Stage-A v2 anchor is not candidate A00")
    fallback_theta["mimic_iv"] = v2_default
    fallback_counts["mimic_iv"] = stage_a_summaries[v2_default.theta_id][
        "mimic_iv"
    ].pass_count
    fallback_admissible = (
        fallback_counts["mimic_iv"] == 20
        and all(fallback_counts[dataset] >= 19 for dataset in REPAIRED_DATASETS)
    )

    if shared_admissible:
        status = "DEVELOPMENT_GO_SHARED"
        theta_by_dataset = {dataset: shared_theta.to_dict() for dataset in DATASETS}
        route = {dataset: "shared" for dataset in DATASETS}
    elif fallback_admissible:
        status = "DEVELOPMENT_GO_DATASET_SPECIFIC"
        theta_by_dataset = {
            dataset: fallback_theta[dataset].to_dict() for dataset in DATASETS
        }
        route = {
            "mimic_iv": "fixed_v2_default_anti_regression_fallback",
            **{dataset: "precomputed_dataset_fallback" for dataset in REPAIRED_DATASETS},
        }
    else:
        status = "DEVELOPMENT_NO_GO"
        theta_by_dataset = {}
        route = {}
    return {
        "protocol": PROTOCOL,
        "phase": "development",
        "status": status,
        "coverage_generated": False,
        "candidate_seed_deletions": 0,
        "shared_stage_b": shared_b,
        "shared_pass_counts": shared_counts,
        "shared_admissible": shared_admissible,
        "dataset_stage_b_fallbacks": fallback_b,
        "fallback_pass_counts": fallback_counts,
        "fallback_admissible": fallback_admissible,
        "mimic_iv_fallback_rule": (
            "exact v2 default raw/k100/gaussian_b2/raw-ridge1e-3"
        ),
        "theta_route_by_dataset": route,
        "theta_by_dataset": theta_by_dataset,
        "stage_b_selection": {
            "protocol": PROTOCOL,
            "selector_version": SELECTOR_VERSION,
            "shared": shared_b,
            "dataset_fallbacks": fallback_b,
            "candidate_dataset_summaries": _summary_matrix_to_dict(
                stage_b_summaries
            ),
        },
        "confirmation_rule": (
            "one frozen theta per dataset; exactly 20 fresh-split seeds; no fallback "
            "or retuning after confirmation opens"
        ),
        "failure_consequence": (
            "do not confirm"
            if status == "DEVELOPMENT_NO_GO"
            else None
        ),
    }


def _summary_matrix_to_dict(
    summaries: Mapping[str, Mapping[str, CandidateDatasetSummary]],
) -> dict[str, dict[str, Any]]:
    return {
        candidate_id: {
            dataset: summary.to_dict()
            for dataset, summary in by_dataset.items()
        }
        for candidate_id, by_dataset in summaries.items()
    }


def verify_parent_v2(parent_root: Path) -> dict[str, Any]:
    """Bind v2 provenance without opening any v2 science/result row."""

    expected = (ROOT / "results/work/controlled_clinical_extension_v2").resolve()
    if parent_root != expected:
        raise RuntimeError(f"v3 parent must be the frozen v2 root: {expected}")
    metadata = _read_parent_json(parent_root, Path("metadata.json"))
    if metadata.get("protocol") != v2.PROTOCOL:
        raise RuntimeError("parent metadata is not controlled clinical v2")
    v2_protocol = v2.load_extension_config(V2_CONFIG_PATH)
    live_dataset_contracts = _json_normalize(
        {
            dataset: v2._dataset_contract(
                v2_protocol,
                v2_protocol.datasets[dataset],
            )
            for dataset in DATASETS
        }
    )
    _validate_live_dataset_contracts(metadata, live_dataset_contracts)
    precoverage = v2._verified_precoverage_retry_amendment()
    postcompute = v2._verified_postcompute_retry_amendment()
    v2._validate_retry_amendment_binding(metadata, precoverage)
    v2._validate_postcompute_retry_amendment_binding(metadata, postcompute)
    expected_complete = v2._root_complete_marker(
        metadata["source_snapshot"],
        precoverage,
        postcompute,
    )
    complete = _read_parent_bytes(parent_root, Path("COMPLETE"))
    if complete.decode("utf-8") != expected_complete:
        raise RuntimeError("parent v2 COMPLETE binding differs")
    source_contract = metadata["source_snapshot"]
    source_entries = {}
    for key in ("archive", "manifest"):
        relative = Path(source_contract[f"{key}_path"])
        payload = _read_parent_bytes(parent_root, relative)
        digest = hashlib.sha256(payload).hexdigest()
        if (
            len(payload) != source_contract[f"{key}_bytes"]
            or digest != source_contract[f"{key}_sha256"]
        ):
            raise RuntimeError(f"parent v2 source {key} binding differs")
        source_entries[key] = {
            "path": relative.as_posix(),
            "bytes": len(payload),
            "sha256": digest,
        }
    manifest_entries = {}
    for relative in (
        Path("manifest.json"),
        *(Path(dataset) / "manifest.json" for dataset in DATASETS),
    ):
        payload = _read_parent_bytes(parent_root, relative)
        manifest = json.loads(payload)
        if manifest.get("protocol") != v2.PROTOCOL:
            raise RuntimeError(f"parent manifest protocol differs: {relative}")
        manifest_entries[relative.as_posix()] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "artifact_count": manifest.get("artifact_count"),
            "referenced_artifacts_not_opened": True,
        }
    binding = {
        "parent_protocol": v2.PROTOCOL,
        "parent_root": parent_root.relative_to(ROOT).as_posix(),
        "parent_source_tree_sha256": metadata["source_tree_sha256"],
        "parent_dataset_contracts": live_dataset_contracts,
        "parent_dataset_contracts_sha256": _json_sha256(
            live_dataset_contracts
        ),
        "parent_complete": {
            "bytes": len(complete),
            "sha256": hashlib.sha256(complete).hexdigest(),
        },
        "parent_manifests": manifest_entries,
        "parent_source_snapshot": source_entries,
        "precoverage_retry_amendment": precoverage,
        "precoverage_retry_amendment_sha256": _json_sha256(precoverage),
        "postcompute_retry_amendment": postcompute,
        "postcompute_retry_amendment_sha256": _json_sha256(postcompute),
        "information_firewall": {
            "science_or_coverage_files_opened": False,
            "manifest_referenced_artifacts_opened": False,
        },
    }
    _validate_parent_binding_record(binding)
    return binding


def _validate_live_dataset_contracts(
    parent_metadata: Mapping[str, Any],
    live_contracts: Mapping[str, Any],
) -> None:
    if set(live_contracts) != set(DATASETS):
        raise RuntimeError("live parent dataset contract set differs")
    stored = parent_metadata.get("dataset_contracts")
    if not isinstance(stored, dict) or stored != dict(live_contracts):
        raise RuntimeError(
            "live base config/raw cache/CXR source contract differs from parent v2"
        )
    expected_base_hashes = {
        dataset: live_contracts[dataset]["base_config_sha256"]
        for dataset in DATASETS
    }
    if parent_metadata.get("base_config_sha256_by_dataset") != expected_base_hashes:
        raise RuntimeError("parent v2 base-config hash index differs")


def _validate_parent_binding_record(binding: Mapping[str, Any]) -> None:
    expected_keys = {
        "parent_protocol",
        "parent_root",
        "parent_source_tree_sha256",
        "parent_dataset_contracts",
        "parent_dataset_contracts_sha256",
        "parent_complete",
        "parent_manifests",
        "parent_source_snapshot",
        "precoverage_retry_amendment",
        "precoverage_retry_amendment_sha256",
        "postcompute_retry_amendment",
        "postcompute_retry_amendment_sha256",
        "information_firewall",
    }
    contracts = binding.get("parent_dataset_contracts")
    if (
        set(binding) != expected_keys
        or binding.get("parent_protocol") != v2.PROTOCOL
        or binding.get("parent_root")
        != "results/work/controlled_clinical_extension_v2"
        or not isinstance(contracts, dict)
        or set(contracts) != set(DATASETS)
        or binding.get("parent_dataset_contracts_sha256")
        != _json_sha256(contracts)
        or not v2._is_sha256(binding.get("parent_source_tree_sha256"))
        or binding.get("information_firewall")
        != {
            "science_or_coverage_files_opened": False,
            "manifest_referenced_artifacts_opened": False,
        }
    ):
        raise RuntimeError("parent v2 binding record is malformed")


def audit_confirmation_rng(
    config: FidelityV3Config,
    *,
    excluded_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    """Audit every confirmation and reserved post-GO stream before launch."""

    v2_protocol = v2.load_extension_config(V2_CONFIG_PATH)
    fresh_presets = {
        dataset: replace(
            v2_protocol.datasets[dataset],
            seeds=config.confirmation_seeds[dataset],
            bootstrap_seed=config.confirmation_bootstrap_seeds[dataset],
        )
        for dataset in DATASETS
    }
    fresh_protocol = replace(v2_protocol, datasets=fresh_presets)
    mapping = v2._new_rng_stream_mapping(fresh_protocol, DATASETS)
    v2._assert_unique_rng_streams(mapping)
    excluded = tuple(path.resolve() for path in excluded_roots if path is not None)
    artifact_ids = _metadata_only_artifact_rng_ids(
        ROOT / "results",
        excluded_roots=excluded,
    )
    source_ids = v2._source_declared_seeds(
        ROOT,
        excluded_paths=_OWN_RNG_DECLARATION_PATHS,
    )
    prior = artifact_ids | source_ids
    collisions = {name: value for name, value in mapping.items() if value in prior}
    return {
        "status": "passed_before_launch" if not collisions else "collision",
        "collision_count": len(collisions),
        "collisions": collisions,
        "scan_policy": (
            "seed filenames plus metadata/manifests only; result summaries and all "
            "science/coverage/width files are never opened"
        ),
        "artifact_rng_id_count": len(artifact_ids),
        "artifact_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "source_declared_rng_id_count": len(source_ids),
        "source_declared_rng_id_sha256": _integer_set_sha256(source_ids),
        "prior_rng_id_count": len(prior),
        "prior_rng_id_sha256": _integer_set_sha256(prior),
        "new_rng_stream_count": len(mapping),
        "new_rng_stream_mapping": mapping,
        "new_rng_stream_mapping_sha256": _json_sha256(mapping),
        "internal_rng_streams_unique": True,
        "reserved_future_streams": [
            "donor_overlap_probe",
            "calibration",
            "reference",
            "ACI/SPCI/PRC adaptation",
            "summary bootstrap",
        ],
        "excluded_roots": [str(path) for path in excluded],
    }


def _metadata_only_artifact_rng_ids(
    root: Path,
    *,
    excluded_roots: Sequence[Path],
) -> set[int]:
    values: set[int] = set()
    if not root.exists():
        return values
    for path in root.rglob("*"):
        resolved = path.resolve()
        if any(resolved == excluded or excluded in resolved.parents for excluded in excluded_roots):
            continue
        match = v2._SEED_NAME.fullmatch(path.name)
        if match:
            values.add(int(match.group(1)))
        if not path.is_file() or path.name not in {
            "metadata.json",
            "study_metadata.json",
            "suite_manifest.json",
            "manifest.json",
        }:
            continue
        if _forbidden_parent_artifact_path(path.relative_to(root)):
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        six._collect_named_seed_values(payload, values)
        v2._collect_artifact_rng_values(payload, values)
    return values


def _root_metadata(
    *,
    phase: str,
    output_root: Path,
    config: FidelityV3Config,
    devices: tuple[str, ...],
    source_hash: str,
    source_snapshot: Mapping[str, Any],
    parent_binding: Mapping[str, Any],
    rng_audit: Mapping[str, Any],
    development_binding: Mapping[str, Any] | None = None,
    frozen_theta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_parent_binding_record(parent_binding)
    config_bytes = CONFIG_PATH.read_bytes()
    active_seeds = (
        config.development_seeds
        if phase == "development"
        else config.confirmation_seeds
    )
    metadata = {
        "protocol": PROTOCOL,
        "phase": phase,
        "role": "coverage_blind_k0_development_and_confirmation",
        "canonical_scpcp_mutation_permitted": False,
        "coverage_generation_permitted": False,
        "science_execution_path_present": False,
        "datasets": list(DATASETS),
        "devices": list(devices),
        "output_root": str(output_root),
        "seed_to_device": _seed_device_mapping(active_seeds, devices),
        "source_tree_sha256": source_hash,
        "source_snapshot": source_snapshot,
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_bytes": len(config_bytes),
        "parent_v2_binding": parent_binding,
        "parent_v2_binding_sha256": _json_sha256(parent_binding),
        "confirmation_rng_audit": rng_audit,
        "selector_contract_sha256": _selector_contract_sha256(),
        "candidate_contract": {
            "stage_a": [candidate.to_dict() for candidate in stage_a_candidates()],
            "stage_b_ridge_order": list(RIDGE_ORDER),
            "stagewise_zscore": {
                "source": "D_env_only",
                "pooling": "per_stage_pooled_over_actions",
                "retrieval": "action_conditional_after_shared_stage_scaling",
                "estimation_dtype": "float64",
                "population_sd": True,
                "sd_floor": config.stagewise_sd_floor,
            },
        },
        "k0_gate": {
            **METRIC_THRESHOLDS,
            "minimum_available_seed_fraction": 0.95,
            "systematic_replays": 16,
            "structural_invariants": "all_20_required",
        },
        "post_gate_science_contract_reserved_but_not_executable_here": {
            "methods": list(METHODS),
            "gammas": [-4.0, -2.0, 0.0, 2.0, 4.0],
            "primary_default_gamma": -4.0,
            "other_gamma_role": "supplementary signed curve",
            "calibration_trajectories": 3_000,
            "grid_trajectories": 1_000,
            "reference_trajectories": 20_000,
            "online_trajectories": 2_000,
            "bootstrap_resamples": 10_000,
            "policy_ratio_cap": 3.0,
            "separate_explicit_command_required_after_all_confirmation_gates": True,
        },
        "seed_roles": {
            "development": {
                dataset: list(config.development_seeds[dataset])
                for dataset in DATASETS
            },
            "confirmation": {
                dataset: list(config.confirmation_seeds[dataset])
                for dataset in DATASETS
            },
            "confirmation_label": "fresh_split_confirmation",
            "independent_patient_confirmation_claimed": False,
        },
        "prelaunch_clarification": {
            "original_draft": "stage/action diagonal z-score",
            "frozen_resolution": "stage-only pooled-over-actions diagonal z-score",
            "formal_seed_run_before_resolution": False,
        },
    }
    if phase == "confirmation":
        metadata["development_binding"] = development_binding
        metadata["development_binding_sha256"] = _json_sha256(development_binding)
        metadata["frozen_theta"] = frozen_theta
        metadata["frozen_theta_sha256"] = frozen_theta["frozen_theta_sha256"]
    return metadata


def _selector_contract_sha256() -> str:
    payload = {
        "version": SELECTOR_VERSION,
        "metric_ratios": [
            [name, threshold] for name, threshold in METRIC_THRESHOLDS.items()
        ],
        "seed_ratio": "maximum metric ratio; structural false maps to infinity",
        "dataset_summary": "pass_count and linear q95 over exactly 20 seed ratios",
        "shared_objective": [
            "negative minimum dataset pass count",
            "maximum dataset q95 seed ratio",
            "global mean seed ratio",
            "minimal change tuple",
            "candidate index",
        ],
        "fallback_objective": [
            "negative pass count",
            "q95 seed ratio",
            "mean seed ratio",
            "minimal change tuple",
            "candidate index",
        ],
        "mimic_anti_regression": "shared=20/20; fallback=exact v2 default 20/20",
        "repaired_dataset_requirement": ">=19/20",
    }
    return _json_sha256(payload)


def _seed_device_mapping(
    seeds_by_dataset: Mapping[str, Sequence[int]],
    devices: Sequence[str],
) -> dict[str, str]:
    if tuple(seeds_by_dataset) != DATASETS or not devices:
        raise ValueError("seed/device mapping requires the frozen datasets and devices")
    return {
        f"{dataset}/base_{seed}": device
        for dataset in DATASETS
        for seed, device in _dataset_seed_device_mapping(
            dataset,
            seeds_by_dataset[dataset],
            devices,
        ).items()
    }


def _dataset_seed_device_mapping(
    dataset: str,
    seeds: Sequence[int],
    devices: Sequence[str],
) -> dict[int, str]:
    if dataset not in DATASETS or not devices:
        raise ValueError("unknown dataset or empty device list")
    return {
        seed: devices[(DATASETS.index(dataset) * 20 + index) % len(devices)]
        for index, seed in enumerate(seeds)
    }


def _active_source_contract() -> tuple[str, dict[str, Any]]:
    source_hash = experiment_tree_sha256()
    snapshot = _build_source_snapshot()
    if experiment_tree_sha256() != source_hash:
        raise RuntimeError("source/config changed while building the v3 snapshot")
    return source_hash, snapshot


def _build_source_snapshot() -> dict[str, Any]:
    paths = [
        *sorted((ROOT / "src" / "scpcp").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "tools").glob("*.py")),
        *sorted((ROOT / "configs").glob("*.yaml")),
        ROOT / "pyproject.toml",
    ]
    relative_paths = [path.relative_to(ROOT).as_posix() for path in paths]
    if len(relative_paths) != len(set(relative_paths)) or any(
        not path.is_file() for path in paths
    ):
        raise RuntimeError("v3 source snapshot file set is invalid")
    files = []
    archive_stream = io.BytesIO()
    with tarfile.open(
        fileobj=archive_stream,
        mode="w",
        format=tarfile.PAX_FORMAT,
    ) as archive:
        for path, relative in zip(paths, relative_paths, strict=True):
            content = path.read_bytes()
            files.append(
                {
                    "path": relative,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
            info = tarfile.TarInfo(relative)
            info.size = len(content)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    archive_bytes = archive_stream.getvalue()
    archive_hash = hashlib.sha256(archive_bytes).hexdigest()
    manifest_payload = {
        "protocol": PROTOCOL,
        "format": "deterministic_uncompressed_pax_tar",
        "file_count": len(files),
        "files": files,
    }
    manifest_bytes = (
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    return {
        "archive_bytes": archive_bytes,
        "manifest_bytes": manifest_bytes,
        "contract": {
            "archive_path": f"provenance/source_snapshot_{archive_hash}.tar",
            "archive_sha256": archive_hash,
            "archive_bytes": len(archive_bytes),
            "manifest_path": f"provenance/source_manifest_{manifest_hash}.json",
            "manifest_sha256": manifest_hash,
            "manifest_bytes": len(manifest_bytes),
            "file_count": len(files),
        },
    }


def _prepare_root(
    root: Path,
    metadata: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if metadata["confirmation_rng_audit"]["status"] != "passed_before_launch":
        raise RuntimeError("confirmation RNG collision audit did not pass")
    if resume:
        if not root.is_dir() or not (root / "metadata.json").is_file():
            raise FileNotFoundError("resume requires an existing v3 metadata.json")
        if _read_json(root / "metadata.json") != metadata:
            raise RuntimeError("resume metadata differs from the active v3 contract")
        _verify_source_snapshot(root, metadata["source_snapshot"])
        return
    if root.exists():
        raise FileExistsError(f"fresh v3 output already exists: {root}")
    root.mkdir(parents=True)
    _atomic_write(
        root / source_snapshot["contract"]["archive_path"],
        source_snapshot["archive_bytes"],
    )
    _atomic_write(
        root / source_snapshot["contract"]["manifest_path"],
        source_snapshot["manifest_bytes"],
    )
    _write_json(root / "metadata.json", metadata)
    _verify_source_snapshot(root, metadata["source_snapshot"])


def _complete_and_valid(root: Path, metadata: Mapping[str, Any]) -> bool:
    if not (root / "COMPLETE").exists():
        return False
    _validate_root_bundle(root, metadata)
    return True


def _finalize_root(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    source_hash: str,
    config: FidelityV3Config,
) -> None:
    if experiment_tree_sha256() != source_hash:
        raise RuntimeError("source/config changed during the v3 phase")
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("v3 metadata changed during the phase")
    current_parent = verify_parent_v2((ROOT / config.parent_v2_root).resolve())
    if current_parent != metadata.get("parent_v2_binding"):
        raise RuntimeError("parent v2 data/config provenance changed during v3")
    _assert_no_forbidden_result_paths(root)
    _write_manifest(root)
    final = _read_json(root / "FINAL_STATUS.json")
    complete = (
        f"complete phase={metadata['phase']} source_tree_sha256={source_hash} "
        f"config_sha256={metadata['config_sha256']} "
        f"final_status_sha256={_json_sha256(final)}\n"
    )
    _write_text(root / "COMPLETE", complete)
    _validate_root_bundle(root, metadata)


def _validate_root_bundle(root: Path, metadata: Mapping[str, Any]) -> None:
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("v3 root metadata mismatch")
    parent_binding = metadata.get("parent_v2_binding")
    if (
        not isinstance(parent_binding, dict)
        or metadata.get("parent_v2_binding_sha256")
        != _json_sha256(parent_binding)
    ):
        raise RuntimeError("v3 root parent binding differs")
    _validate_parent_binding_record(parent_binding)
    _verify_source_snapshot(root, metadata["source_snapshot"])
    _verify_manifest(root)
    _assert_no_forbidden_result_paths(root)
    final = _read_json(root / "FINAL_STATUS.json")
    expected = (
        f"complete phase={metadata['phase']} "
        f"source_tree_sha256={metadata['source_tree_sha256']} "
        f"config_sha256={metadata['config_sha256']} "
        f"final_status_sha256={_json_sha256(final)}\n"
    )
    if (root / "COMPLETE").read_text() != expected:
        raise RuntimeError("v3 COMPLETE marker differs")
    if final.get("protocol") != PROTOCOL or final.get("coverage_generated") is not False:
        raise RuntimeError("v3 final status violates the coverage firewall")
    if metadata["phase"] == "development":
        has_theta = (root / "frozen_theta.json").exists()
        if (final.get("status") == "DEVELOPMENT_NO_GO") is has_theta:
            raise RuntimeError("development frozen-theta publication is incoherent")
    elif metadata["phase"] != "confirmation":
        raise RuntimeError("unknown v3 root phase")


def _verify_development_for_confirmation(
    root: Path,
    *,
    current_parent_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _read_json(root / "metadata.json")
    if metadata.get("protocol") != PROTOCOL or metadata.get("phase") != "development":
        raise RuntimeError("confirmation requires a v3 development root")
    _validate_root_bundle(root, metadata)
    final = _read_json(root / "FINAL_STATUS.json")
    if final.get("status") not in {
        "DEVELOPMENT_GO_SHARED",
        "DEVELOPMENT_GO_DATASET_SPECIFIC",
    }:
        raise RuntimeError("development did not freeze an admissible theta")
    frozen = _read_json(root / "frozen_theta.json")
    _validate_development_parent_binding(
        metadata,
        frozen,
        current_parent_binding,
    )
    stored_hash = frozen.get("frozen_theta_sha256")
    unhashed = {key: value for key, value in frozen.items() if key != "frozen_theta_sha256"}
    if stored_hash != _json_sha256(unhashed):
        raise RuntimeError("frozen theta self-hash differs")
    if frozen.get("development_decision_sha256") != _json_sha256(final):
        raise RuntimeError("frozen theta does not bind the development decision")
    if frozen.get("development_config_sha256") != metadata["config_sha256"]:
        raise RuntimeError("frozen theta does not bind the development config")
    theta_by_dataset = frozen.get("theta_by_dataset")
    if not isinstance(theta_by_dataset, dict) or tuple(theta_by_dataset) != DATASETS:
        raise RuntimeError("frozen theta dataset order differs")
    for dataset in DATASETS:
        _theta_from_dict(theta_by_dataset[dataset])
    manifest_bytes = (root / "manifest.json").read_bytes()
    complete_bytes = (root / "COMPLETE").read_bytes()
    binding = {
        "root": str(root),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_bytes": len(manifest_bytes),
        "complete_sha256": hashlib.sha256(complete_bytes).hexdigest(),
        "complete_bytes": len(complete_bytes),
        "final_status_sha256": _json_sha256(final),
        "frozen_theta_sha256": stored_hash,
        "source_tree_sha256": metadata["source_tree_sha256"],
        "config_sha256": metadata["config_sha256"],
    }
    return binding, frozen


def _validate_development_parent_binding(
    development_metadata: Mapping[str, Any],
    frozen_theta: Mapping[str, Any],
    current_parent_binding: Mapping[str, Any],
) -> None:
    current_hash = _json_sha256(current_parent_binding)
    if (
        development_metadata.get("parent_v2_binding")
        != current_parent_binding
        or development_metadata.get("parent_v2_binding_sha256") != current_hash
        or frozen_theta.get("parent_v2_binding_sha256") != current_hash
    ):
        raise RuntimeError(
            "development parent data/config binding differs at confirmation"
        )


def _verify_source_snapshot(root: Path, contract: Mapping[str, Any]) -> None:
    for name in ("archive", "manifest"):
        path = root / contract[f"{name}_path"]
        if (
            not path.is_file()
            or path.stat().st_size != contract[f"{name}_bytes"]
            or _file_sha256(path) != contract[f"{name}_sha256"]
        ):
            raise RuntimeError(f"v3 source snapshot {name} differs")


def _write_manifest(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "COMPLETE"}:
            continue
        if ".tmp-" in path.name:
            raise RuntimeError(f"temporary artifact remains: {path}")
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    _write_json(
        root / "manifest.json",
        {
            "protocol": PROTOCOL,
            "artifact_count": len(entries),
            "artifacts": entries,
        },
    )


def _verify_manifest(root: Path) -> None:
    manifest = _read_json(root / "manifest.json")
    entries = manifest.get("artifacts")
    if manifest.get("protocol") != PROTOCOL or not isinstance(entries, list):
        raise RuntimeError("invalid v3 manifest header")
    expected = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise RuntimeError("malformed v3 manifest entry")
        path = root / entry["path"]
        expected.add(path.resolve())
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or _file_sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"v3 manifest mismatch: {path}")
    observed = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "COMPLETE"}
    }
    if observed != expected or manifest.get("artifact_count") != len(entries):
        raise RuntimeError("v3 manifest file set differs")


def _assert_no_forbidden_result_paths(root: Path) -> None:
    forbidden = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _forbidden_parent_artifact_path(relative):
            forbidden.append(relative.as_posix())
    if forbidden:
        raise RuntimeError(f"coverage firewall rejected result paths: {forbidden}")


def _read_parent_json(parent_root: Path, relative: Path) -> dict[str, Any]:
    return json.loads(_read_parent_bytes(parent_root, relative))


def _read_parent_bytes(parent_root: Path, relative: Path) -> bytes:
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("parent artifact path escapes the v2 root")
    if _forbidden_parent_artifact_path(relative):
        raise RuntimeError(f"information firewall rejected parent artifact: {relative}")
    path = parent_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing parent artifact: {path}")
    return path.read_bytes()


def _forbidden_parent_artifact_path(relative: Path) -> bool:
    normalized = relative.as_posix().lower()
    return any(token in normalized for token in FORBIDDEN_PARENT_PATH_TOKENS)


def _validate_devices(devices: tuple[str, ...]) -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run v3 from the workspace root: {ROOT}")
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("v3 requires exactly two distinct CUDA devices")
    if any(not device.startswith("cuda:") for device in devices):
        raise ValueError("v3 requires explicit CUDA devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
    for device in devices:
        index = torch.device(device).index
        if index is None or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {device}")


def _theta_from_dict(value: Mapping[str, Any]) -> KernelTheta:
    theta = KernelTheta(
        stage_a_id=str(value["stage_a_id"]),
        metric=str(value["metric"]),
        neighbors=int(value["neighbors"]),
        weight=str(value["weight"]),
        ridge=str(value["ridge"]),
    )
    if theta.to_dict() != dict(value):
        raise ValueError("serialized theta differs from the frozen schema")
    return theta


def _normalized_ratio(metrics: Mapping[str, Any]) -> float:
    if metrics.get("structural_invariants") is not True:
        return math.inf
    return max(
        float(metrics[name]) / threshold
        for name, threshold in METRIC_THRESHOLDS.items()
    )


def _validate_seed_payload(
    payload: Mapping[str, Any],
    *,
    phase: str,
    preset: DatasetPreset,
    seed: int,
    device: str,
    source_hash: str,
    candidate_hash: str,
) -> None:
    expected = {
        "protocol": PROTOCOL,
        "phase": phase,
        "dataset": preset.name,
        "seed": seed,
        "device": device,
        "source_tree_sha256": source_hash,
        "candidate_contract_sha256": candidate_hash,
    }
    _validate_seed_payload_firewall(payload)
    if set(payload) != {*expected, "result"}:
        raise RuntimeError(f"{phase} seed payload top-level schema differs for {seed}")
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{phase} seed payload provenance differs for {seed}")
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("coverage_generated") is not False:
        raise RuntimeError(f"{phase} seed result violates coverage firewall")
    if result.get("seed") != seed or result.get("dataset") != preset.name:
        raise RuntimeError(f"{phase} seed result identity differs")
    if phase in {"development_stage_a", "development_stage_b"}:
        _require_exact_keys(
            result,
            {
                "seed",
                "dataset",
                "phase",
                "candidate_count",
                "candidates",
                "split_audit",
                "coverage_generated",
                "information_opened",
            },
            "development result",
        )
        candidates = result.get("candidates")
        split = result.get("split_audit")
        _validate_split_audit_schema(split)
        if (
            result.get("phase") != "development_k0_only"
            or not isinstance(candidates, list)
            or result.get("candidate_count") != len(candidates)
            or not v2._valid_split_audit(split)
            or _json_sha256([candidate.get("theta") for candidate in candidates])
            != candidate_hash
        ):
            raise RuntimeError(f"invalid development candidate matrix for {seed}")
        for candidate in candidates:
            _require_exact_keys(
                candidate,
                {
                    "theta",
                    "metrics",
                    "passed",
                    "normalized_seed_ratio",
                    "structural_failure_ratio_is_infinite",
                    "systematic_replay",
                    "context_identity",
                },
                "development candidate",
            )
            _validate_candidate_k0(
                candidate,
                preset=preset,
                seed=seed,
                split_audit=split,
            )
        return
    if phase == "confirmation_support":
        _require_exact_keys(
            result,
            {
                "seed",
                "dataset",
                "phase",
                "outcome_blind",
                "passed",
                "minimum_unique_patients",
                "failed_cells",
                "unique_patient_counts_by_stage_action",
                "n_actions",
                "active_action_indices",
                "action_mapping",
                "action_costs",
                "environment_episode_support",
                "split_audit",
                "coverage_generated",
                "confirmation_label",
            },
            "confirmation support result",
        )
        _validate_split_audit_schema(result.get("split_audit"))
        if result.get("phase") != "confirmation_support" or not v2._valid_support_result(
            result, preset
        ):
            raise RuntimeError(f"invalid confirmation support result for {seed}")
        return
    if phase == "confirmation_k0":
        _require_exact_keys(
            result,
            {
                "seed",
                "dataset",
                "phase",
                "theta",
                "metrics",
                "passed",
                "normalized_seed_ratio",
                "structural_failure_ratio_is_infinite",
                "systematic_replay",
                "context_identity",
                "split_audit",
                "coverage_generated",
                "confirmation_label",
                "independent_patient_confirmation_claimed",
            },
            "confirmation K0 result",
        )
        _validate_split_audit_schema(result.get("split_audit"))
        if (
            result.get("phase") != "confirmation_k0"
            or result.get("confirmation_label") != "fresh_split_confirmation"
            or result.get("independent_patient_confirmation_claimed") is not False
            or not v2._valid_split_audit(result.get("split_audit"))
            or _json_sha256([result.get("theta")]) != candidate_hash
        ):
            raise RuntimeError(f"invalid confirmation K0 result for {seed}")
        _validate_candidate_k0(
            result,
            preset=preset,
            seed=seed,
            split_audit=result["split_audit"],
        )
        return
    raise RuntimeError(f"unknown v3 seed phase: {phase}")


def _validate_candidate_k0(
    candidate: Mapping[str, Any],
    *,
    preset: DatasetPreset,
    seed: int,
    split_audit: Mapping[str, Any],
) -> None:
    theta = _theta_from_dict(candidate["theta"])
    metrics = candidate.get("metrics")
    detail = candidate.get("systematic_replay")
    identity = candidate.get("context_identity")
    if not isinstance(metrics, dict) or not isinstance(detail, dict):
        raise RuntimeError("candidate K0 metrics are malformed")
    expected_metric_keys = {*METRIC_THRESHOLDS, "structural_invariants"}
    if set(metrics) != expected_metric_keys:
        raise RuntimeError("candidate K0 metric schema differs")
    _require_exact_keys(
        detail,
        {
            "label",
            "episode_weighted",
            "inference_unit",
            "systematic_replays",
            "patient_chunk_size",
            "base_uniform_seed",
            "base_uniform_shape",
            "base_uniform_sha256",
            "expansion_formula",
            "flatten_order",
            "expanded_uniform_sha256",
            "score_ks_by_stage",
            "signed_residual_max_w1_by_stage",
            "successor_mean_w1_by_stage",
            "successor_q95_w1_by_stage",
            "active_successor_coordinates_by_stage",
            "raw_structural_invariants_by_stage",
        },
        "candidate systematic replay",
    )
    if not all(
        math.isfinite(float(metrics[name])) and float(metrics[name]) >= 0.0
        for name in METRIC_THRESHOLDS
    ) or not isinstance(metrics["structural_invariants"], bool):
        raise RuntimeError("candidate K0 headline metrics are invalid")
    vectors = {
        "score_ks_by_stage": "maximum_score_ks",
        "signed_residual_max_w1_by_stage": "maximum_signed_residual_w1",
        "successor_mean_w1_by_stage": "maximum_successor_mean_w1",
        "successor_q95_w1_by_stage": "maximum_successor_q95_w1",
    }
    for vector_name, maximum_name in vectors.items():
        values = detail.get(vector_name)
        if (
            not isinstance(values, list)
            or len(values) != preset.horizon
            or not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values)
            or float(metrics[maximum_name]) != max(float(value) for value in values)
        ):
            raise RuntimeError(f"candidate K0 vector differs: {vector_name}")
    active = detail.get("active_successor_coordinates_by_stage")
    invariants = detail.get("raw_structural_invariants_by_stage")
    if (
        not isinstance(active, list)
        or len(active) != preset.horizon
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in active)
        or not isinstance(invariants, list)
        or len(invariants) != preset.horizon
        or not all(v2._valid_k0_invariant_row(value) for value in invariants)
    ):
        raise RuntimeError("candidate K0 structural detail differs")
    structural = all(value["passed"] for value in invariants) and all(
        value > 0 for value in active
    )
    if metrics["structural_invariants"] is not structural:
        raise RuntimeError("candidate K0 structural headline differs")
    gate = v2.load_extension_config(V2_CONFIG_PATH).k0_fidelity_gate
    resolved = v2.K0FidelityMetrics(**metrics)
    if candidate.get("passed") is not v2.k0_fidelity_passes(resolved, gate):
        raise RuntimeError("candidate K0 pass flag differs")
    fidelity_count = int(split_audit["role_episode_counts"]["fidelity"])
    uniforms = v2._expected_k0_uniform_contract(
        seed=seed,
        horizon=preset.horizon,
        fidelity_episode_count=fidelity_count,
        replay_count=16,
    )
    if (
        detail.get("systematic_replays") != 16
        or detail.get("base_uniform_seed") != uniforms["base_uniform_seed"]
        or detail.get("base_uniform_shape") != uniforms["base_uniform_shape"]
        or detail.get("base_uniform_sha256") != uniforms["base_uniform_sha256"]
        or detail.get("expanded_uniform_sha256")
        != uniforms["expanded_uniform_sha256"]
    ):
        raise RuntimeError("candidate K0 CRN contract differs")
    ratio = _normalized_ratio(metrics)
    if math.isfinite(ratio):
        if (
            float(candidate.get("normalized_seed_ratio")) != ratio
            or candidate.get("structural_failure_ratio_is_infinite") is not False
        ):
            raise RuntimeError("candidate normalized ratio differs")
    elif (
        candidate.get("normalized_seed_ratio") is not None
        or candidate.get("structural_failure_ratio_is_infinite") is not True
    ):
        raise RuntimeError("structural failure must map normalized ratio to infinity")
    if not isinstance(identity, dict):
        raise RuntimeError("candidate context identity is missing")
    _require_exact_keys(
        identity,
        {
            "base_nuisance_context_sha256",
            "outcome_model_state_sha256",
            "behavior_policy_state_sha256",
            "split_patient_id_sha256",
            "active_config_sha256",
            "theta",
            "metric_transform",
            "combined_sha256",
        },
        "candidate context identity",
    )
    combined = identity.get("combined_sha256")
    unhashed = {key: value for key, value in identity.items() if key != "combined_sha256"}
    transform = identity.get("metric_transform")
    split_hashes = identity.get("split_patient_id_sha256")
    expected_transform_source = (
        "D_env_only" if theta.metric == "stagewise_zscore" else "identity"
    )
    expected_transform_pooling = (
        "per_stage_pooled_over_actions"
        if theta.metric == "stagewise_zscore"
        else "not_applicable"
    )
    expected_retrieval = (
        "action_conditional_after_shared_stage_scaling"
        if theta.metric == "stagewise_zscore"
        else "action_conditional_raw_geometry"
    )
    if isinstance(transform, dict):
        _require_exact_keys(
            transform,
            {
                "geometry",
                "source",
                "pooling",
                "retrieval",
                "coordinate_count",
                "stage_count",
                "estimation_dtype",
                "population_sd",
                "sd_floor",
                "center_sha256",
                "scale_sha256",
            },
            "candidate metric transform",
        )
    if (
        combined != _json_sha256(unhashed)
        or identity.get("theta") != theta.to_dict()
        or not isinstance(split_hashes, dict)
        or set(split_hashes) != {"predictor", "fidelity", "environment"}
        or not all(v2._is_sha256(value) for value in split_hashes.values())
        or not isinstance(transform, dict)
        or transform.get("geometry") != theta.metric
        or transform.get("source") != expected_transform_source
        or transform.get("pooling") != expected_transform_pooling
        or transform.get("retrieval") != expected_retrieval
        or transform.get("stage_count") != preset.horizon
        or transform.get("coordinate_count") != 32
        or transform.get("estimation_dtype") != "float64"
        or transform.get("population_sd") is not True
        or transform.get("sd_floor") != 1e-4
        or not v2._is_sha256(transform.get("center_sha256"))
        or not v2._is_sha256(transform.get("scale_sha256"))
    ):
        raise RuntimeError("candidate metric-transform identity differs")


def _require_exact_keys(
    value: object,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError(f"{label} schema differs")


def _validate_split_audit_schema(value: object) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("split audit is missing")
    _require_exact_keys(
        value,
        {
            "role_unique_patient_counts",
            "role_episode_counts",
            "role_patient_id_sha256",
            "patient_sets_pairwise_disjoint",
            "split_fractions",
        },
        "split audit",
    )
    if value.get("split_fractions") != [0.40, 0.20, 0.40]:
        raise RuntimeError("split audit fractions differ")


def _reject_scientific_result_keys(
    value: object,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            normalized = str(child_key).lower()
            child_path = (*path, normalized)
            is_kernel_bandwidth = (
                len(child_path) >= 2
                and child_path[-2:] == ("theta", "bandwidth")
            )
            if not is_kernel_bandwidth and (
                "coverage" in normalized
                or "width" in normalized
                or "science" in normalized
                or "method_selection" in normalized
                or normalized in {"methods", "method_rows", "selection_status"}
            ):
                raise RuntimeError(f"coverage firewall rejected result key: {child_key}")
            _reject_scientific_result_keys(child, child_path)
    elif isinstance(value, list):
        for child in value:
            _reject_scientific_result_keys(child, path)


def _validate_seed_payload_firewall(payload: Mapping[str, Any]) -> None:
    result = payload.get("result")
    if (
        not isinstance(result, dict)
        or result.get("coverage_generated") is not False
    ):
        raise RuntimeError(
            "coverage firewall requires one false result-root marker"
        )
    sanitized_result = {
        key: value
        for key, value in result.items()
        if key != "coverage_generated"
    }
    sanitized_payload = {
        key: sanitized_result if key == "result" else value
        for key, value in payload.items()
    }
    _reject_scientific_result_keys(sanitized_payload)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    _atomic_write(path, payload)


def _write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.encode("utf-8"))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_normalize(value: object) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _integer_set_sha256(values: Iterable[int]) -> str:
    return _json_sha256(sorted(set(int(value) for value in values)))


if __name__ == "__main__":
    main()
