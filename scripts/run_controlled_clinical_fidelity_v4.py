"""Run the coverage-blind, dataset-independent clinical-v4 K0 repair.

Examples
--------
Develop eICU and MIMIC-CXR repairs while importing the two frozen v3 anchors::

    python scripts/run_controlled_clinical_fidelity_v4.py development \
      --devices cuda:0,cuda:1 \
      --output-root results/work/controlled_clinical_fidelity_v4_development

Confirm only datasets whose own development gate passed::

    python scripts/run_controlled_clinical_fidelity_v4.py confirmation \
      --devices cuda:0,cuda:1 \
      --development-root results/work/controlled_clinical_fidelity_v4_development \
      --output-root results/work/controlled_clinical_fidelity_v4_confirmation

Administratively retry a confirmation that failed before any K0 execution::

    python scripts/run_controlled_clinical_fidelity_v4.py confirmation-retry \
      --devices cuda:0,cuda:1 \
      --development-root results/work/controlled_clinical_fidelity_v4_development \
      --output-root \
        results/work/controlled_clinical_fidelity_v4_confirmation_administrative_retry_r1

This runner can compute support and K0 fidelity only.  It has no paper-method
or coverage execution path.
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
    ControlledClinicalExtensionConfig,
    DatasetPreset,
)
from scpcp.controlled_clinical_fidelity_v4 import (  # noqa: E402
    ANCHOR_DATASETS,
    DATASETS,
    FROZEN_ANCHORS,
    METRIC_THRESHOLDS,
    PROTOCOL,
    REPAIR_DATASETS,
    SELECTOR_VERSION,
    FidelityV4Config,
    FrozenAnchor,
    K0CandidateSummary,
    RepairTheta,
    load_fidelity_v4_config,
    normalized_seed_ratio,
    repair_candidates,
    select_dataset_candidate,
    summarize_candidate_dataset,
    validate_parent_v3_bundle,
)
from scpcp.controlled_transition import ControlledResidualEnvironment  # noqa: E402
from scpcp.scores import score_batch  # noqa: E402


Theta = RepairTheta | FrozenAnchor

CONFIG_PATH = ROOT / "configs/controlled_clinical_fidelity_v4.yaml"
V2_CONFIG_PATH = ROOT / "configs/controlled_clinical_extension.yaml"
DEVELOPMENT_ROOT = (
    ROOT / "results/work/controlled_clinical_fidelity_v4_development"
).resolve()
CONFIRMATION_ROOT = (
    ROOT / "results/work/controlled_clinical_fidelity_v4_confirmation"
).resolve()
CONFIRMATION_RETRY_ROOT = (
    ROOT
    / "results/work/controlled_clinical_fidelity_v4_confirmation_administrative_retry_r1"
).resolve()
RETRY_PROTOCOL = "controlled_clinical_fidelity_v4_confirmation_administrative_retry_r1"
PHASES = (
    "audit",
    "retry-audit",
    "development",
    "confirmation",
    "confirmation-retry",
)
FORBIDDEN_RESULT_PATH_TOKENS = (
    "science",
    "coverage",
    "width",
    "method_selection",
)
EXPECTED_DEVELOPMENT_REUSE_MAPPING_SHA256 = (
    "c5d2f96cd8b33339b9abfb2bc572c61bf183048f8b27ff331ca8651a364234e3"
)
EXPECTED_DEVELOPMENT_REUSE_ID_SET_SHA256 = (
    "c75733a9a8d2e69122804e1b2800e4f4f7ab7a51d60ebb37e993ea1f527dcbff"
)
EXPECTED_DEVELOPMENT_REUSE_STREAM_COUNT = 180
EXPECTED_PARENT_SEED_ENVELOPE_SHA256 = (
    "2547b7f476b1fd9aa05809860e9729b51dc3b8a22d6adfac7f70cec6e5395946"
)
EXPECTED_DEVELOPMENT_EXTERNAL_PRIOR_COUNT = 2842
EXPECTED_DEVELOPMENT_EXTERNAL_PRIOR_SHA256 = (
    "7b99c33061f6e2254c2095c5f54126eec71fee52c6dbb23296a18a3156a1719b"
)
EXPECTED_CONFIRMATION_STREAM_COUNT = 1304
EXPECTED_CONFIRMATION_PRIOR_STREAM_COUNT = 5476
EXPECTED_CONFIRMATION_ARTIFACT_COUNT = 5317
EXPECTED_CONFIRMATION_ARTIFACT_SHA256 = (
    "ed4e1bfdc36b484f418bbdd854d3a1404c84bc42d977507f004e7eb0a0880f2d"
)
EXPECTED_CONFIRMATION_SOURCE_COUNT = 887
EXPECTED_CONFIRMATION_SOURCE_SHA256 = (
    "49f32dbfbf3bc90da46101a741d9efdbedadc24f520631144f9c6823ef1eb36f"
)
EXPECTED_CONFIRMATION_PRIOR_SHA256 = (
    "616a14269c9ad698f921d6c535eb14a2b03d2cdef524498ec068864a91c646a6"
)
EXPECTED_CONFIRMATION_MAPPING_ID_SET_SHA256 = (
    "506b914c95553ed222a34015b8e90cf7b1d1b70e7d6108c22ebf446bc4f80ab5"
)
EXPECTED_CONFIRMATION_BASE_SEED_SET_SHA256 = (
    "c0f0e0c5b15516ef5977deea90f4ad749173ef44e4de131fec3b1de33c32231e"
)
EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256 = (
    "9ba785a5d34899bda4dbd4eb3c8998f5be8b7f5dd8951bdd2f60d229628b8e21"
)
EXPECTED_DEVELOPMENT_SOURCE_FILE_COUNT = 113
EXPECTED_DEVELOPMENT_RUNNER_SHA256 = (
    "a39fcf256c809f541fdbd9128303c53a2b4f265cd57c55e4ed09aee1aacfbffb"
)
EXPECTED_FAILED_CONFIRMATION_FILE_COUNT = 24
EXPECTED_FAILED_CONFIRMATION_INVENTORY_SHA256 = (
    "cb9eea72de06c38bdbb74bb2100d10eef8266b9a1b75f7b50eb89497f072dcc8"
)
EXPECTED_FAILED_CONFIRMATION_METADATA_SHA256 = (
    "62177d9c8907bd06684508d8e462ee3f542ee2c0e095334a36b4a4fdb1fac33a"
)
EXPECTED_FAILED_CONFIRMATION_SUPPORT_SHA256 = (
    "dddca33480c3ae35c85610872286b4d81958840def27ebfd20ad662b1916eddb"
)
EXPECTED_FAILED_CONFIRMATION_SUPPORT_COMPLETE_BYTES = 9
EXPECTED_FAILED_CONFIRMATION_SUPPORT_COMPLETE_SHA256 = (
    "37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1"
)
RETRY_CHANGED_SOURCE_PATH = "scripts/run_controlled_clinical_fidelity_v4.py"
_OWN_RNG_DECLARATION_PATHS = {
    Path(__file__).resolve(),
    (ROOT / "src/scpcp/controlled_clinical_fidelity_v4.py").resolve(),
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
    config = load_fidelity_v4_config(CONFIG_PATH)
    parent_binding = _validated_parent_v3_binding(config)
    development_reuse_audit = audit_development_reuse(config)
    failed_attempt_binding: dict[str, Any] | None = None
    if args.phase in {"retry-audit", "confirmation-retry"}:
        failed_attempt_binding = _failed_confirmation_attempt_binding(config)
        confirmation_rng_audit = audit_confirmation_retry_rng(
            config,
            failed_attempt_binding=failed_attempt_binding,
        )
    else:
        confirmation_rng_audit = audit_confirmation_rng(
            config,
            excluded_roots=(DEVELOPMENT_ROOT, CONFIRMATION_ROOT),
        )
    if args.phase == "audit":
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "parent_v3_binding_sha256": _json_sha256(parent_binding),
                    "development_rng_reuse_audit": development_reuse_audit,
                    "confirmation_rng_audit": confirmation_rng_audit,
                    "coverage_generation_permitted": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.phase == "retry-audit":
        development_binding, frozen = _verify_development_for_confirmation(
            DEVELOPMENT_ROOT,
            config=config,
            current_parent_binding=parent_binding,
        )
        source_hash, source_snapshot = _active_source_contract()
        amendment = _administrative_retry_amendment(
            config=config,
            source_hash=source_hash,
            source_snapshot=source_snapshot,
            development_binding=development_binding,
            frozen_settings=frozen,
            failed_attempt_binding=failed_attempt_binding,
            confirmation_rng_audit=confirmation_rng_audit,
        )
        print(
            json.dumps(
                {
                    "protocol": RETRY_PROTOCOL,
                    "status": "ADMINISTRATIVE_RETRY_PREFLIGHT_GO",
                    "source_tree_sha256": source_hash,
                    "amendment_sha256": _json_sha256(amendment),
                    "failed_attempt_inventory_sha256": failed_attempt_binding[
                        "inventory_sha256"
                    ],
                    "formal_rng_consumed": False,
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
            development_reuse_audit=development_reuse_audit,
            confirmation_rng_audit=confirmation_rng_audit,
            resume=args.resume,
        )
    elif args.phase == "confirmation":
        if args.development_root is None:
            parser.error("confirmation requires --development-root")
        development_root = args.development_root.resolve()
        if development_root != DEVELOPMENT_ROOT:
            parser.error(f"development root is frozen to {DEVELOPMENT_ROOT}")
        if output_root != CONFIRMATION_ROOT:
            parser.error(f"confirmation output root is frozen to {CONFIRMATION_ROOT}")
        run_confirmation(
            output_root,
            development_root=development_root,
            config=config,
            devices=devices,
            parent_binding=parent_binding,
            development_reuse_audit=development_reuse_audit,
            confirmation_rng_audit=confirmation_rng_audit,
            resume=args.resume,
        )
    else:
        if args.development_root is None:
            parser.error("confirmation-retry requires --development-root")
        development_root = args.development_root.resolve()
        if development_root != DEVELOPMENT_ROOT:
            parser.error(f"development root is frozen to {DEVELOPMENT_ROOT}")
        if output_root != CONFIRMATION_RETRY_ROOT:
            parser.error(
                "confirmation-retry output root is frozen to "
                f"{CONFIRMATION_RETRY_ROOT}"
            )
        run_confirmation_retry(
            output_root,
            development_root=development_root,
            config=config,
            devices=devices,
            parent_binding=parent_binding,
            development_reuse_audit=development_reuse_audit,
            confirmation_rng_audit=confirmation_rng_audit,
            failed_attempt_binding=failed_attempt_binding,
            resume=args.resume,
        )
    print(output_root)


def run_development(
    output_root: Path,
    *,
    config: FidelityV4Config,
    devices: tuple[str, ...],
    parent_binding: Mapping[str, Any],
    development_reuse_audit: Mapping[str, Any],
    confirmation_rng_audit: Mapping[str, Any],
    resume: bool,
) -> None:
    """Search each repair dataset independently and freeze every local GO."""

    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    source_hash, source_snapshot = _active_source_contract()
    metadata = _root_metadata(
        phase="development",
        output_root=output_root,
        config=config,
        devices=devices,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        parent_binding=parent_binding,
        development_reuse_audit=development_reuse_audit,
        confirmation_rng_audit=confirmation_rng_audit,
    )
    _prepare_root(output_root, metadata, source_snapshot, resume=resume)
    if _complete_and_valid(output_root, metadata, config=config):
        return

    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in REPAIR_DATASETS:
        candidates = repair_candidates(dataset)
        preset = replace(
            protocol.datasets[dataset],
            seeds=config.development_seeds[dataset],
        )
        rows_by_dataset[dataset] = _run_seed_phase(
            output_root / "repair" / dataset,
            phase="development_repair",
            preset=preset,
            devices=devices,
            candidates=candidates,
            worker=_development_worker,
            worker_arguments=(protocol,),
            source_hash=source_hash,
        )

    parent_status = validate_parent_v3_bundle(config, workspace_root=ROOT)
    decision, selections = _development_decision(rows_by_dataset, parent_status)
    for dataset, selection in selections.items():
        _write_json(output_root / "selection" / f"{dataset}.json", selection)
    _write_json(output_root / "FINAL_STATUS.json", decision)
    frozen = _frozen_settings(decision, metadata)
    _write_json(output_root / "frozen_settings.json", frozen)
    _finalize_root(output_root, metadata, source_hash=source_hash, config=config)


def run_confirmation(
    output_root: Path,
    *,
    development_root: Path,
    config: FidelityV4Config,
    devices: tuple[str, ...],
    parent_binding: Mapping[str, Any],
    development_reuse_audit: Mapping[str, Any],
    confirmation_rng_audit: Mapping[str, Any],
    resume: bool,
) -> None:
    """Open a fresh confirmation bank only for each dataset's frozen local GO."""

    development_binding, frozen = _verify_development_for_confirmation(
        development_root,
        config=config,
        current_parent_binding=parent_binding,
    )
    source_hash, source_snapshot = _active_source_contract()
    if source_hash != frozen["development_source_tree_sha256"]:
        raise RuntimeError("source changed after v4 settings freeze")
    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    metadata = _root_metadata(
        phase="confirmation",
        output_root=output_root,
        config=config,
        devices=devices,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        parent_binding=parent_binding,
        development_reuse_audit=development_reuse_audit,
        confirmation_rng_audit=confirmation_rng_audit,
        development_binding=development_binding,
        frozen_settings=frozen,
    )
    _prepare_root(output_root, metadata, source_snapshot, resume=resume)
    if _complete_and_valid(output_root, metadata, config=config):
        return

    gates: dict[str, dict[str, Any]] = {}
    frozen_theta = frozen["theta_by_dataset"]
    for dataset in DATASETS:
        theta_payload = frozen_theta.get(dataset)
        if theta_payload is None:
            gate = _unopened_confirmation_gate(dataset)
        else:
            theta = _theta_from_dict(theta_payload)
            preset = replace(
                protocol.datasets[dataset],
                seeds=config.confirmation_seeds[dataset],
                bootstrap_seed=config.confirmation_bootstrap_seeds[dataset],
            )
            support_rows = _run_seed_phase(
                output_root / dataset / "support",
                phase="confirmation_support",
                preset=preset,
                devices=devices,
                candidates=(),
                worker=_confirmation_support_worker,
                worker_arguments=(protocol,),
                source_hash=source_hash,
            )
            support_count = sum(bool(row["passed"]) for row in support_rows)
            k0_rows: list[dict[str, Any]] = []
            if support_count >= 19:
                k0_rows = _run_seed_phase(
                    output_root / dataset / "k0_fidelity",
                    phase="confirmation_k0",
                    preset=preset,
                    devices=devices,
                    candidates=(theta,),
                    worker=_confirmation_k0_worker,
                    worker_arguments=(protocol,),
                    source_hash=source_hash,
                )
            gate = _confirmation_gate(
                dataset,
                theta_payload,
                support_rows,
                k0_rows,
            )
        gates[dataset] = gate
        _write_json(output_root / dataset / "gate.json", gate)
        _write_text(output_root / dataset / "COMPLETE", gate["status"].lower() + "\n")

    final = _confirmation_final(gates)
    _write_json(output_root / "FINAL_STATUS.json", final)
    _finalize_root(output_root, metadata, source_hash=source_hash, config=config)


def run_confirmation_retry(
    output_root: Path,
    *,
    development_root: Path,
    config: FidelityV4Config,
    devices: tuple[str, ...],
    parent_binding: Mapping[str, Any],
    development_reuse_audit: Mapping[str, Any],
    confirmation_rng_audit: Mapping[str, Any],
    failed_attempt_binding: Mapping[str, Any] | None,
    resume: bool,
) -> None:
    """Recompute the same confirmation after a pre-K0 runner adapter failure."""

    if failed_attempt_binding is None:
        raise RuntimeError("confirmation retry requires the failed-attempt binding")
    development_binding, frozen = _verify_development_for_confirmation(
        development_root,
        config=config,
        current_parent_binding=parent_binding,
    )
    source_hash, source_snapshot = _active_source_contract()
    amendment = _administrative_retry_amendment(
        config=config,
        source_hash=source_hash,
        source_snapshot=source_snapshot,
        development_binding=development_binding,
        frozen_settings=frozen,
        failed_attempt_binding=failed_attempt_binding,
        confirmation_rng_audit=confirmation_rng_audit,
    )
    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    metadata = _root_metadata(
        phase="confirmation_retry",
        output_root=output_root,
        config=config,
        devices=devices,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        parent_binding=parent_binding,
        development_reuse_audit=development_reuse_audit,
        confirmation_rng_audit=confirmation_rng_audit,
        development_binding=development_binding,
        frozen_settings=frozen,
        retry_amendment=amendment,
    )
    _prepare_root(output_root, metadata, source_snapshot, resume=resume)
    _write_once_json(
        output_root / "administrative_retry_amendment.json",
        amendment,
    )
    if _complete_and_valid(output_root, metadata, config=config):
        return

    frozen_theta = frozen["theta_by_dataset"]
    support_by_dataset: dict[str, list[dict[str, Any]]] = {}
    presets: dict[str, DatasetPreset] = {}
    for dataset in DATASETS:
        if dataset not in frozen_theta:
            continue
        preset = replace(
            protocol.datasets[dataset],
            seeds=config.confirmation_seeds[dataset],
            bootstrap_seed=config.confirmation_bootstrap_seeds[dataset],
        )
        presets[dataset] = preset
        support_by_dataset[dataset] = _run_seed_phase(
            output_root / dataset / "support",
            phase="confirmation_support",
            preset=preset,
            devices=devices,
            candidates=(),
            worker=_confirmation_support_worker,
            worker_arguments=(protocol,),
            source_hash=source_hash,
        )

    current_failed_binding = _failed_confirmation_attempt_binding(config)
    if current_failed_binding != dict(failed_attempt_binding):
        raise RuntimeError("failed confirmation evidence changed during support replay")
    support_verification = _verify_retry_support_replay(
        output_root,
        support_by_dataset,
        config=config,
        failed_attempt_binding=failed_attempt_binding,
    )
    _write_once_json(
        output_root / "support_replay_verification.json",
        support_verification,
    )

    gates: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        theta_payload = frozen_theta.get(dataset)
        if theta_payload is None:
            gate = _unopened_confirmation_gate(dataset)
        else:
            theta = _theta_from_dict(theta_payload)
            support_rows = support_by_dataset[dataset]
            support_count = sum(bool(row["passed"]) for row in support_rows)
            k0_rows: list[dict[str, Any]] = []
            if support_count >= 19:
                k0_rows = _run_seed_phase(
                    output_root / dataset / "k0_fidelity",
                    phase="confirmation_k0",
                    preset=presets[dataset],
                    devices=devices,
                    candidates=(theta,),
                    worker=_confirmation_k0_worker,
                    worker_arguments=(protocol,),
                    source_hash=source_hash,
                )
            gate = _confirmation_gate(
                dataset,
                theta_payload,
                support_rows,
                k0_rows,
            )
        gates[dataset] = gate
        _write_json(output_root / dataset / "gate.json", gate)
        _write_text(output_root / dataset / "COMPLETE", gate["status"].lower() + "\n")

    final = _confirmation_retry_final(gates, amendment)
    _write_json(output_root / "FINAL_STATUS.json", final)
    _finalize_root(output_root, metadata, source_hash=source_hash, config=config)


def _development_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    candidates: tuple[RepairTheta, ...],
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    if base_context.config.model.representation_dim != 32:
        raise RuntimeError("v4 candidate geometry requires representation dim 32")
    candidate_rows = []
    for theta in candidates:
        context = _context_with_theta(base_context, theta)
        metrics, detail = v2._logging_mixture_fidelity(
            context,
            seed=seed,
            protocol=protocol,
        )
        metric_payload = asdict(metrics)
        ratio = normalized_seed_ratio(metric_payload)
        candidate_rows.append(
            {
                "theta": theta.to_dict(),
                "metrics": metric_payload,
                "passed": v2.k0_fidelity_passes(metrics, protocol.k0_fidelity_gate),
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
    candidates: tuple[Theta, ...],
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    if candidates:
        raise RuntimeError("support must not receive a transition setting")
    result = v2._support_worker(seed, preset, device, protocol)
    return {
        **result,
        "phase": "confirmation_support",
        "coverage_generated": False,
        "confirmation_label": "fresh_split_operational_gate",
    }


def _confirmation_k0_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    candidates: tuple[Theta, ...],
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    if len(candidates) != 1:
        raise RuntimeError("confirmation requires exactly one frozen setting")
    theta = candidates[0]
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    context = _context_with_theta(base_context, theta)
    metrics, detail = v2._logging_mixture_fidelity(
        context,
        seed=seed,
        protocol=protocol,
    )
    metric_payload = asdict(metrics)
    ratio = normalized_seed_ratio(metric_payload)
    return {
        "seed": seed,
        "dataset": preset.name,
        "phase": "confirmation_k0",
        "theta": _theta_to_dict(theta),
        "metrics": metric_payload,
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
        "confirmation_label": "fresh_split_operational_gate",
        "independent_patient_confirmation_claimed": False,
    }


def _context_with_theta(
    base_context: v2.ExtensionContext,
    theta: Theta,
) -> v2.ExtensionContext:
    """Replace only the transition simulator, passing every frozen mode explicitly."""

    theta_payload = _theta_to_dict(theta)
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
        bandwidth=float(theta_payload["bandwidth"]),
        ridge=theta.ridge_value,
        representation_geometry=theta.metric,
        donor_weighting=str(theta_payload["donor_weighting"]),
        ridge_mode=theta.ridge_mode,
        transition_mode=theta.transition_mode,
        outcome_residual_mode=theta.outcome_residual_mode,
    )
    return replace(base_context, environment=environment)


def _candidate_context_identity(
    base_context: v2.ExtensionContext,
    environment: ControlledResidualEnvironment,
    theta: Theta,
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
        "source": "D_env_only" if theta.metric == "stagewise_zscore" else "identity",
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
    sizes = [
        [
            int(len(environment._libraries[(stage, action)][0]))
            for action in range(base_context.n_actions)
        ]
        for stage in range(environment.horizon)
    ]
    effective = [
        [min(theta.neighbors, size) for size in stage_sizes]
        for stage_sizes in sizes
    ]
    uses_full_cell = theta.neighbors == 10_000
    full_cell_verified = (
        all(
            used == size
            for stage_sizes, stage_used in zip(sizes, effective, strict=True)
            for size, used in zip(stage_sizes, stage_used, strict=True)
        )
        if uses_full_cell
        else None
    )
    if uses_full_cell and full_cell_verified is not True:
        raise RuntimeError("full-cell sentinel did not include an entire donor cell")
    library_support = {
        "requested_neighbors": theta.neighbors,
        "uses_full_cell": uses_full_cell,
        "full_cell_neighbor_sentinel": 10_000,
        "actual_library_sizes_by_stage_action": sizes,
        "effective_neighbor_counts_by_stage_action": effective,
        "full_cell_verified": full_cell_verified,
        "stage_count": environment.horizon,
        "action_count": base_context.n_actions,
    }
    library_support["combined_sha256"] = _json_sha256(library_support)
    identity = {
        "base_nuisance_context_sha256": base_identity["combined_sha256"],
        "outcome_model_state_sha256": base_identity["outcome_model_state_sha256"],
        "behavior_policy_state_sha256": base_identity["behavior_policy_state_sha256"],
        "split_patient_id_sha256": base_identity["split_patient_id_sha256"],
        "active_config_sha256": base_identity["active_config_sha256"],
        "theta": _theta_to_dict(theta),
        "metric_transform": transform,
        "library_support": library_support,
    }
    return {**identity, "combined_sha256": _json_sha256(identity)}


def _run_seed_phase(
    phase_root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    devices: tuple[str, ...],
    candidates: tuple[Theta, ...],
    worker: Callable[..., dict[str, Any]],
    worker_arguments: tuple[object, ...],
    source_hash: str,
) -> list[dict[str, Any]]:
    phase_root.mkdir(parents=True, exist_ok=True)
    mapping = _dataset_seed_device_mapping(preset.name, preset.seeds, devices)
    candidate_hash = _json_sha256([_theta_to_dict(theta) for theta in candidates])
    expected = {phase_root / f"seed_{seed:06d}.json" for seed in preset.seeds}
    unexpected = set(phase_root.glob("seed_*.json")) - expected
    if unexpected:
        raise RuntimeError(f"unexpected {phase} artifacts: {sorted(unexpected)}")

    completed: dict[int, dict[str, Any]] = {}
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
    candidates: tuple[Theta, ...],
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


def _summarize_dataset_candidates(
    dataset: str,
    candidates: Sequence[RepairTheta],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, K0CandidateSummary]:
    if len(rows) != 20:
        raise RuntimeError(f"{dataset} repair matrix is not 20 seeds")
    expected_ids = tuple(theta.candidate_id for theta in candidates)
    summaries: dict[str, K0CandidateSummary] = {}
    for candidate_index, theta in enumerate(candidates):
        metrics = []
        for row in rows:
            candidate_rows = row.get("candidates")
            if (
                not isinstance(candidate_rows, list)
                or tuple(item["theta"]["candidate_id"] for item in candidate_rows)
                != expected_ids
            ):
                raise RuntimeError(f"{dataset} repair candidate order differs")
            metrics.append(candidate_rows[candidate_index]["metrics"])
        summaries[theta.candidate_id] = summarize_candidate_dataset(theta, metrics)
    return summaries


def _development_decision(
    rows_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    parent_status: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Recompute all local decisions without a cross-dataset objective."""

    if tuple(rows_by_dataset) != REPAIR_DATASETS:
        raise RuntimeError("repair result datasets differ from the frozen order")
    selections: dict[str, dict[str, Any]] = {}
    dataset_decisions: dict[str, dict[str, Any]] = {}
    theta_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        if dataset in ANCHOR_DATASETS:
            anchor = FROZEN_ANCHORS[dataset]
            _validate_parent_anchor(parent_status, anchor)
            decision = {
                "protocol": PROTOCOL,
                "dataset": dataset,
                "route": "frozen_v3_anchor",
                "status": "DATASET_DEVELOPMENT_GO",
                "development_admissible": True,
                "development_pass_count": anchor.development_pass_count,
                "development_structural_pass_count": 20,
                "theta": anchor.to_dict(),
                "parent_source_json_pointer": anchor.source_json_pointer,
                "parent_pass_count_json_pointer": anchor.pass_count_json_pointer,
                "candidate_seed_deletions": 0,
            }
        else:
            candidates = repair_candidates(dataset)
            summaries = _summarize_dataset_candidates(
                dataset,
                candidates,
                rows_by_dataset[dataset],
            )
            selection = select_dataset_candidate(dataset, candidates, summaries)
            selection = {
                "protocol": PROTOCOL,
                "candidate_dataset_summaries": {
                    candidate_id: summary.to_dict()
                    for candidate_id, summary in summaries.items()
                },
                **selection,
            }
            selections[dataset] = selection
            decision = {
                "protocol": PROTOCOL,
                "dataset": dataset,
                "route": "dataset_specific_repair_grid",
                "status": selection["status"],
                "development_admissible": selection["development_admissible"],
                "development_pass_count": selection["winner_summary"]["pass_count"],
                "development_structural_pass_count": selection["winner_summary"][
                    "structural_pass_count"
                ],
                "theta": selection["winner"],
                "selection_sha256": _json_sha256(selection),
                "candidate_seed_deletions": 0,
            }
        dataset_decisions[dataset] = decision
        if decision["development_admissible"]:
            theta_by_dataset[dataset] = decision["theta"]

    all_go = len(theta_by_dataset) == len(DATASETS)
    final = {
        "protocol": PROTOCOL,
        "phase": "development",
        "status": (
            "DEVELOPMENT_COMPLETE_ALL_DATASETS_GO"
            if all_go
            else "DEVELOPMENT_COMPLETE_PARTIAL_DATASET_GO"
        ),
        "decision_scope": "per_dataset_independent",
        "cross_dataset_conjunction_used": False,
        "dataset_decisions": dataset_decisions,
        "theta_by_dataset": theta_by_dataset,
        "development_go_datasets": [
            dataset for dataset in DATASETS if dataset in theta_by_dataset
        ],
        "development_no_go_datasets": [
            dataset for dataset in DATASETS if dataset not in theta_by_dataset
        ],
        "coverage_generated": False,
        "confirmation_rule": (
            "open each fresh dataset bank iff that dataset alone passed development"
        ),
        "candidate_seed_deletions": 0,
    }
    return final, selections


def _validate_parent_anchor(
    parent_status: Mapping[str, Any],
    anchor: FrozenAnchor,
) -> None:
    theta = _json_pointer(parent_status, anchor.source_json_pointer)
    pass_count = _json_pointer(parent_status, anchor.pass_count_json_pointer)
    if not isinstance(theta, dict) or theta.get("theta_id") != anchor.theta_id:
        raise RuntimeError(f"v3 parent anchor differs for {anchor.dataset}")
    if pass_count != anchor.development_pass_count:
        raise RuntimeError(f"v3 parent anchor count differs for {anchor.dataset}")


def _frozen_settings(
    decision: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = {
        "protocol": PROTOCOL,
        "role": "dataset_settings_frozen_before_any_fresh_confirmation",
        "decision_scope": "per_dataset_independent",
        "cross_dataset_conjunction_used": False,
        "theta_by_dataset": decision["theta_by_dataset"],
        "development_go_datasets": decision["development_go_datasets"],
        "development_no_go_datasets": decision["development_no_go_datasets"],
        "development_decision_sha256": _json_sha256(decision),
        "development_source_tree_sha256": metadata["source_tree_sha256"],
        "development_config_sha256": metadata["config_sha256"],
        "parent_v3_binding_sha256": metadata["parent_v3_binding_sha256"],
        "coverage_generation_permitted": False,
    }
    return {**frozen, "frozen_settings_sha256": _json_sha256(frozen)}


def _unopened_confirmation_gate(dataset: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "status": "CONFIRMATION_NOT_OPENED_DEVELOPMENT_NO_GO",
        "development_admissible": False,
        "confirmation_opened": False,
        "support_pass_count": 0,
        "structural_pass_count": 0,
        "k0_pass_count": 0,
        "prespecified_seed_count": 20,
        "theta": None,
        "gate_role": "fresh_split_operational_gate",
        "independent_patient_confirmation_claimed": False,
        "coverage_generated": False,
    }


def _confirmation_gate(
    dataset: str,
    theta: Mapping[str, Any],
    support_rows: Sequence[Mapping[str, Any]],
    k0_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(support_rows) != 20:
        raise RuntimeError(f"{dataset} confirmation support is not 20 seeds")
    support_count = sum(bool(row["passed"]) for row in support_rows)
    support_pass = support_count >= 19
    if support_pass and len(k0_rows) != 20:
        raise RuntimeError(f"{dataset} opened K0 bank is not 20 seeds")
    if not support_pass and k0_rows:
        raise RuntimeError(f"{dataset} K0 opened after its support gate failed")
    structural_count = sum(
        bool(row["metrics"]["structural_invariants"]) for row in k0_rows
    )
    k0_count = sum(bool(row["passed"]) for row in k0_rows)
    status = (
        "CONFIRMATION_GATE_GO"
        if support_pass and structural_count == 20 and k0_count >= 19
        else "CONFIRMATION_GATE_NO_GO"
    )
    return {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "status": status,
        "development_admissible": True,
        "confirmation_opened": True,
        "support_pass_count": support_count,
        "structural_pass_count": structural_count,
        "k0_pass_count": k0_count,
        "prespecified_seed_count": 20,
        "theta": dict(theta),
        "gate_role": "fresh_split_operational_gate",
        "independent_patient_confirmation_claimed": False,
        "coverage_generated": False,
    }


def _confirmation_final(
    gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(gates) != DATASETS:
        raise RuntimeError("confirmation gates differ from the frozen dataset order")
    confirmed = [
        dataset
        for dataset in DATASETS
        if gates[dataset]["status"] == "CONFIRMATION_GATE_GO"
    ]
    opened = [
        dataset for dataset in DATASETS if gates[dataset]["confirmation_opened"]
    ]
    return {
        "protocol": PROTOCOL,
        "phase": "confirmation",
        "status": "CONFIRMATION_COMPLETE_DATASET_INDEPENDENT",
        "decision_scope": "per_dataset_independent",
        "cross_dataset_conjunction_used": False,
        "datasets": dict(gates),
        "confirmation_opened_datasets": opened,
        "confirmed_datasets": confirmed,
        "unconfirmed_datasets": [
            dataset for dataset in DATASETS if dataset not in confirmed
        ],
        "gate_role": "fresh_split_operational_gate",
        "independent_patient_confirmation_claimed": False,
        "coverage_generated": False,
    }


def _verify_retry_support_replay(
    retry_root: Path,
    support_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    config: FidelityV4Config,
    failed_attempt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact MIMIC support semantics before the retry can open K0."""

    frozen = _read_json(DEVELOPMENT_ROOT / "frozen_settings.json")
    opened_datasets = tuple(
        dataset for dataset in DATASETS if dataset in frozen["theta_by_dataset"]
    )
    if tuple(support_by_dataset) != opened_datasets:
        raise RuntimeError("retry support datasets differ from the frozen local GO set")
    if (retry_root / "mimic_cxr/support").exists() or (
        retry_root / "mimic_cxr/k0_fidelity"
    ).exists():
        raise RuntimeError("MIMIC-CXR must remain unopened after development no-go")
    if any(len(support_by_dataset[dataset]) != 20 for dataset in opened_datasets):
        raise RuntimeError("retry support did not recompute every opened 20-seed bank")

    old_root = CONFIRMATION_ROOT / "mimic_iv/support"
    new_root = retry_root / "mimic_iv/support"
    per_seed = []
    allowed_delta_keys = ["source_tree_sha256"]
    for seed in config.confirmation_seeds["mimic_iv"]:
        old_payload = _read_json(old_root / f"seed_{seed:06d}.json")
        new_payload = _read_json(new_root / f"seed_{seed:06d}.json")
        if set(old_payload) != set(new_payload):
            raise RuntimeError("retry support top-level schema differs from failed lineage")
        old_without_source = {
            key: value
            for key, value in old_payload.items()
            if key not in allowed_delta_keys
        }
        new_without_source = {
            key: value
            for key, value in new_payload.items()
            if key not in allowed_delta_keys
        }
        if (
            old_without_source != new_without_source
            or old_payload["source_tree_sha256"]
            != EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256
            or new_payload["source_tree_sha256"]
            == EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256
            or old_payload["result"] != new_payload["result"]
        ):
            raise RuntimeError("retry MIMIC support differs from failed support semantics")
        per_seed.append(
            {
                "seed": seed,
                "result_sha256": _json_sha256(new_payload["result"]),
                "old_payload_sha256": _file_sha256(
                    old_root / f"seed_{seed:06d}.json"
                ),
                "new_payload_sha256": _file_sha256(
                    new_root / f"seed_{seed:06d}.json"
                ),
                "result_exact_equal": True,
            }
        )
    expected_result_hashes = failed_attempt_binding["support_result_hashes"]
    observed_result_hashes = [
        {"seed": row["seed"], "result_sha256": row["result_sha256"]}
        for row in per_seed
    ]
    if observed_result_hashes != expected_result_hashes:
        raise RuntimeError("retry support result hashes differ from failed evidence")
    verification = {
        "protocol": RETRY_PROTOCOL,
        "role": "pre_k0_exact_support_replay_verification",
        "same_first_fresh_confirmation_lineage": True,
        "second_fresh_bank_claimed": False,
        "independent_rng_bank_claimed": False,
        "failed_attempt_artifacts_reused": False,
        "support_recomputed_from_scratch": True,
        "opened_support_datasets": list(opened_datasets),
        "all_opened_support_banks_complete": True,
        "mimic_support_seed_count": len(per_seed),
        "mimic_support_result_exact_equal_count": len(per_seed),
        "top_level_allowed_delta_keys": allowed_delta_keys,
        "per_seed": per_seed,
        "failed_support_result_set_sha256": failed_attempt_binding[
            "support_result_set_sha256"
        ],
        "retry_support_result_set_sha256": _json_sha256(observed_result_hashes),
        "mimic_cxr_support_or_k0_opened": False,
        "k0_permitted_after_this_verification": True,
        "coverage_generation_permitted": False,
    }
    return {**verification, "verification_sha256": _json_sha256(verification)}


def _confirmation_retry_final(
    gates: Mapping[str, Mapping[str, Any]],
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    final = _confirmation_final(gates)
    return {
        **final,
        "phase": "confirmation_retry",
        "administrative_retry_protocol": RETRY_PROTOCOL,
        "administrative_retry_amendment_sha256": _json_sha256(amendment),
        "same_first_fresh_confirmation_lineage": True,
        "second_fresh_bank_claimed": False,
        "independent_rng_bank_claimed": False,
        "failed_attempt_artifacts_reused": False,
        "support_recomputed_from_scratch": True,
    }


def _validated_parent_v3_binding(config: FidelityV4Config) -> dict[str, Any]:
    status = validate_parent_v3_bundle(config, workspace_root=ROOT)
    root = config.parent_v3.root
    if not root.is_absolute():
        root = ROOT / root
    metadata = _read_json(root / "metadata.json")
    binding = {
        "root": root.relative_to(ROOT).as_posix(),
        "configured_file_sha256": dict(config.parent_v3.file_sha256),
        "metadata_sha256": _file_sha256(root / "metadata.json"),
        "metadata_source_tree_sha256": metadata["source_tree_sha256"],
        "metadata_parent_v2_binding_sha256": metadata["parent_v2_binding_sha256"],
        "final_status_sha256": _json_sha256(status),
        "manifest_artifact_count": _read_json(root / "manifest.json")["artifact_count"],
        "full_v3_root_validation": True,
        "live_v2_binding_validation": True,
        "controlled_transition_default_parity_validation": True,
        "archived_v3_controlled_transition_sha256": (
            config.archived_v3_controlled_transition_sha256
        ),
        "current_controlled_transition_sha256": (
            config.current_controlled_transition_sha256
        ),
    }
    return binding


def audit_development_reuse(config: FidelityV4Config) -> dict[str, Any]:
    """Prove that every reused nuisance stream is exact, authorized v3 lineage."""

    validate_parent_v3_bundle(config, workspace_root=ROOT)
    mapping = _development_reuse_mapping(config)
    if (
        len(mapping) != EXPECTED_DEVELOPMENT_REUSE_STREAM_COUNT
        or len(set(mapping.values())) != len(mapping)
        or _json_sha256(mapping) != EXPECTED_DEVELOPMENT_REUSE_MAPPING_SHA256
        or _integer_set_sha256(mapping.values())
        != EXPECTED_DEVELOPMENT_REUSE_ID_SET_SHA256
    ):
        raise RuntimeError("development RNG reuse mapping differs")

    parent_root = config.parent_v3.root
    if not parent_root.is_absolute():
        parent_root = ROOT / parent_root
    envelope = []
    covered_labels: set[str] = set()
    for dataset in REPAIR_DATASETS:
        for seed in config.development_seeds[dataset]:
            prefix = f"{dataset}/base_{seed}"
            rng = {
                label: value
                for label, value in mapping.items()
                if label.startswith(prefix + "/")
            }
            covered_labels.update(rng)
            phase_records = []
            for phase in ("stage_a", "stage_b"):
                path = parent_root / phase / dataset / f"seed_{seed:06d}.json"
                payload = _read_json(path)
                result = payload.get("result")
                if (
                    payload.get("protocol") != "controlled_clinical_fidelity_v3"
                    or payload.get("dataset") != dataset
                    or payload.get("seed") != seed
                    or not isinstance(result, dict)
                    or result.get("seed") != seed
                    or result.get("dataset") != dataset
                    or result.get("coverage_generated") is not False
                    or set(result.get("information_opened", ()))
                    != {"support", "k0_fidelity", "context_identity"}
                ):
                    raise RuntimeError(f"v3 parent seed envelope differs: {path}")
                candidates = result.get("candidates")
                if not isinstance(candidates, list) or not candidates:
                    raise RuntimeError(f"v3 parent candidate envelope is empty: {path}")
                uniform_seeds = {
                    item["systematic_replay"]["base_uniform_seed"]
                    for item in candidates
                }
                uniform_hashes = {
                    item["systematic_replay"]["base_uniform_sha256"]
                    for item in candidates
                }
                nuisance_hashes = {
                    item["context_identity"]["base_nuisance_context_sha256"]
                    for item in candidates
                }
                if (
                    uniform_seeds != {v2.K0_UNIFORM_SEED_OFFSET + seed}
                    or len(uniform_hashes) != 1
                    or len(nuisance_hashes) != 1
                ):
                    raise RuntimeError(f"v3 parent CRN envelope differs: {path}")
                phase_records.append(
                    {
                        "phase": phase,
                        "file_sha256": _file_sha256(path),
                        "base_uniform_sha256": next(iter(uniform_hashes)),
                        "base_nuisance_context_sha256": next(iter(nuisance_hashes)),
                        "split_patient_id_sha256": result["split_audit"][
                            "role_patient_id_sha256"
                        ],
                    }
                )
            comparable_fields = (
                "base_uniform_sha256",
                "base_nuisance_context_sha256",
                "split_patient_id_sha256",
            )
            if any(
                phase_records[0][name] != phase_records[1][name]
                for name in comparable_fields
            ):
                raise RuntimeError(f"v3 Stage-A/B lineage differs for {dataset}/{seed}")
            envelope.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "rng": rng,
                    "stage_a": phase_records[0],
                    "stage_b": phase_records[1],
                }
            )
    envelope_hash = _json_sha256(envelope)
    if (
        covered_labels != set(mapping)
        or envelope_hash != EXPECTED_PARENT_SEED_ENVELOPE_SHA256
        or envelope_hash != config.development_reuse_audit["parent_seed_envelope_sha256"]
    ):
        raise RuntimeError("development parent seed envelope binding differs")

    artifact_ids, source_ids = _external_development_prior_rng_ids(
        parent_root=parent_root,
    )
    external_prior = artifact_ids | source_ids
    unauthorized = {
        label: value for label, value in mapping.items() if value in external_prior
    }
    if (
        len(external_prior) != EXPECTED_DEVELOPMENT_EXTERNAL_PRIOR_COUNT
        or _integer_set_sha256(external_prior)
        != EXPECTED_DEVELOPMENT_EXTERNAL_PRIOR_SHA256
        or unauthorized
    ):
        raise RuntimeError("development reuse collides with unauthorized prior RNG use")
    return {
        **dict(config.development_reuse_audit),
        "status": "passed_before_launch",
        "mapping": mapping,
        "verified_parent_seed_envelope_sha256": envelope_hash,
        "verified_parent_seed_envelope_count": len(envelope),
        "authorized_lineage_collision_count": len(mapping),
        "missing_lineage_collision_count": 0,
        "unauthorized_collision_count": len(unauthorized),
        "unauthorized_collisions": unauthorized,
        "external_artifact_rng_id_count": len(artifact_ids),
        "external_artifact_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "external_source_rng_id_count": len(source_ids),
        "external_source_rng_id_sha256": _integer_set_sha256(source_ids),
        "external_prior_rng_id_count": len(external_prior),
        "external_prior_rng_id_sha256": _integer_set_sha256(external_prior),
    }


def _development_reuse_mapping(config: FidelityV4Config) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for dataset in REPAIR_DATASETS:
        for seed in config.development_seeds[dataset]:
            prefix = f"{dataset}/base_{seed}"
            mapping[f"{prefix}/task"] = seed
            mapping[f"{prefix}/outcome_model"] = seed + 1
            mapping[f"{prefix}/behavior_model"] = seed + 2
            if dataset == "mimic_cxr":
                mapping[f"{prefix}/cxr_encoder"] = seed + 701
            mapping[f"{prefix}/k0_base_uniform"] = v2.K0_UNIFORM_SEED_OFFSET + seed
    return mapping


def _external_development_prior_rng_ids(
    *,
    parent_root: Path,
) -> tuple[set[int], set[int]]:
    parent_metadata = _read_json(parent_root / "metadata.json")
    v2_root = ROOT / parent_metadata["parent_v2_binding"]["parent_root"]
    excluded_roots = (
        parent_root.resolve(),
        v2_root.resolve(),
        DEVELOPMENT_ROOT,
        CONFIRMATION_ROOT,
        CONFIRMATION_RETRY_ROOT,
    )
    artifact_ids = _metadata_only_artifact_rng_ids(
        ROOT / "results",
        excluded_roots=excluded_roots,
    )
    inherited_declarations = {
        (ROOT / relative).resolve()
        for relative in (
            "scripts/run_controlled_clinical_extension.py",
            "src/scpcp/controlled_clinical_extension.py",
            "configs/controlled_clinical_extension.yaml",
            "scripts/run_controlled_clinical_fidelity_v3.py",
            "src/scpcp/controlled_clinical_fidelity_v3.py",
            "configs/controlled_clinical_fidelity_v3.yaml",
        )
    }
    source_ids = v2._source_declared_seeds(
        ROOT,
        excluded_paths=inherited_declarations | _OWN_RNG_DECLARATION_PATHS,
    )
    return artifact_ids, source_ids


def audit_confirmation_rng(
    config: FidelityV4Config,
    *,
    excluded_roots: Sequence[Path],
) -> dict[str, Any]:
    """Audit the full v2-derived fresh confirmation stream mapping."""

    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    presets = {
        dataset: replace(
            protocol.datasets[dataset],
            seeds=config.confirmation_seeds[dataset],
            bootstrap_seed=config.confirmation_bootstrap_seeds[dataset],
        )
        for dataset in DATASETS
    }
    mapping = v2._new_rng_stream_mapping(replace(protocol, datasets=presets), DATASETS)
    v2._assert_unique_rng_streams(mapping)
    mapping_values = set(mapping.values())
    base_seeds = set().union(*map(set, config.confirmation_seeds.values()))
    if (
        len(mapping) != EXPECTED_CONFIRMATION_STREAM_COUNT
        or _json_sha256(mapping) != config.confirmation_mapping_sha256
        or _integer_set_sha256(mapping_values)
        != EXPECTED_CONFIRMATION_MAPPING_ID_SET_SHA256
        or len(base_seeds) != 80
        or _integer_set_sha256(base_seeds)
        != EXPECTED_CONFIRMATION_BASE_SEED_SET_SHA256
    ):
        raise RuntimeError("confirmation RNG mapping differs from the frozen audit")

    artifact_ids = _metadata_only_artifact_rng_ids(
        ROOT / "results",
        excluded_roots=tuple(path.resolve() for path in excluded_roots),
    )
    source_ids = v2._source_declared_seeds(
        ROOT,
        excluded_paths=_OWN_RNG_DECLARATION_PATHS,
    )
    prior = artifact_ids | source_ids
    collisions = {label: value for label, value in mapping.items() if value in prior}
    if (
        len(artifact_ids) != EXPECTED_CONFIRMATION_ARTIFACT_COUNT
        or _integer_set_sha256(artifact_ids) != EXPECTED_CONFIRMATION_ARTIFACT_SHA256
        or len(source_ids) != EXPECTED_CONFIRMATION_SOURCE_COUNT
        or _integer_set_sha256(source_ids) != EXPECTED_CONFIRMATION_SOURCE_SHA256
        or len(prior) != EXPECTED_CONFIRMATION_PRIOR_STREAM_COUNT
        or _integer_set_sha256(prior) != EXPECTED_CONFIRMATION_PRIOR_SHA256
        or collisions
    ):
        raise RuntimeError("confirmation RNG collision audit differs")
    return {
        "status": "passed_before_launch",
        "collision_count": 0,
        "collisions": {},
        "artifact_rng_id_count": len(artifact_ids),
        "artifact_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "source_declared_rng_id_count": len(source_ids),
        "source_declared_rng_id_sha256": _integer_set_sha256(source_ids),
        "prior_rng_id_count": len(prior),
        "prior_rng_id_sha256": _integer_set_sha256(prior),
        "new_rng_stream_count": len(mapping),
        "new_rng_stream_mapping": mapping,
        "new_rng_stream_mapping_sha256": _json_sha256(mapping),
        "new_rng_id_set_sha256": _integer_set_sha256(mapping_values),
        "confirmation_base_seed_count": len(base_seeds),
        "confirmation_base_seed_set_sha256": _integer_set_sha256(base_seeds),
        "internal_rng_streams_unique": True,
        "excluded_roots": [str(path.resolve()) for path in excluded_roots],
    }


def audit_confirmation_retry_rng(
    config: FidelityV4Config,
    *,
    failed_attempt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify the reused confirmation mapping as one exact administrative lineage."""

    audit = audit_confirmation_rng(
        config,
        excluded_roots=(
            DEVELOPMENT_ROOT,
            CONFIRMATION_ROOT,
            CONFIRMATION_RETRY_ROOT,
        ),
    )
    failed_metadata = _read_json(CONFIRMATION_ROOT / "metadata.json")
    failed_audit = failed_metadata.get("confirmation_rng_audit")
    mapping = audit["new_rng_stream_mapping"]
    if (
        failed_attempt_binding.get("inventory_sha256")
        != EXPECTED_FAILED_CONFIRMATION_INVENTORY_SHA256
        or not isinstance(failed_audit, dict)
        or failed_audit.get("new_rng_stream_mapping") != mapping
        or failed_audit.get("new_rng_stream_mapping_sha256")
        != audit["new_rng_stream_mapping_sha256"]
        or failed_audit.get("new_rng_id_set_sha256")
        != audit["new_rng_id_set_sha256"]
        or failed_audit.get("new_rng_stream_count")
        != EXPECTED_CONFIRMATION_STREAM_COUNT
    ):
        raise RuntimeError("failed confirmation RNG lineage differs")
    if audit["collision_count"] != 0:
        raise RuntimeError("retry has an unauthorized collision outside failed lineage")
    return {
        **audit,
        "status": "passed_before_launch",
        "role": "exact_administrative_reuse_of_first_confirmation_bank",
        "same_prespecified_mapping_reused": True,
        "same_prespecified_mapping_collision_count": len(mapping),
        "same_prespecified_mapping_collision_sha256": _json_sha256(mapping),
        "authorized_declared_lineage_collision_count": len(mapping),
        "failed_attempt_executed_support_artifact_count": (
            failed_attempt_binding["support_seed_artifact_count"]
        ),
        "failed_attempt_k0_artifact_count": failed_attempt_binding[
            "k0_artifact_count"
        ],
        "unauthorized_collision_count": 0,
        "failed_root_excluded_only_after_exact_binding": True,
        "second_fresh_bank_claimed": False,
        "independent_rng_bank_claimed": False,
    }


def _failed_confirmation_attempt_binding(
    config: FidelityV4Config,
) -> dict[str, Any]:
    """Validate and hash the immutable, pre-K0 failed confirmation attempt."""

    root = CONFIRMATION_ROOT
    if not root.is_dir():
        raise FileNotFoundError("administrative retry requires the failed confirmation root")
    inventory = _artifact_inventory(root)
    inventory_hash = _json_sha256(inventory)
    support_inventory = [
        row
        for row in inventory
        if row["path"].startswith("mimic_iv/support/seed_")
    ]
    expected_support_paths = {
        f"mimic_iv/support/seed_{seed:06d}.json"
        for seed in config.confirmation_seeds["mimic_iv"]
    }
    observed_paths = {row["path"] for row in inventory}
    metadata = _read_json(root / "metadata.json")
    snapshot = metadata.get("source_snapshot")
    if not isinstance(snapshot, dict):
        raise RuntimeError("failed confirmation source snapshot binding is malformed")
    expected_paths = {
        "metadata.json",
        "mimic_iv/support/COMPLETE",
        *expected_support_paths,
        str(snapshot.get("archive_path")),
        str(snapshot.get("manifest_path")),
    }
    support_complete = root / "mimic_iv/support/COMPLETE"
    k0_root = root / "mimic_iv/k0_fidelity"
    k0_files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "k0_fidelity" in path.parts
    ]
    if (
        len(inventory) != EXPECTED_FAILED_CONFIRMATION_FILE_COUNT
        or inventory_hash != EXPECTED_FAILED_CONFIRMATION_INVENTORY_SHA256
        or _file_sha256(root / "metadata.json")
        != EXPECTED_FAILED_CONFIRMATION_METADATA_SHA256
        or len(support_inventory) != 20
        or _json_sha256(support_inventory)
        != EXPECTED_FAILED_CONFIRMATION_SUPPORT_SHA256
        or observed_paths != expected_paths
        or not support_complete.is_file()
        or support_complete.stat().st_size
        != EXPECTED_FAILED_CONFIRMATION_SUPPORT_COMPLETE_BYTES
        or _file_sha256(support_complete)
        != EXPECTED_FAILED_CONFIRMATION_SUPPORT_COMPLETE_SHA256
        or support_complete.read_text() != "complete\n"
        or not k0_root.is_dir()
        or any(k0_root.iterdir())
        or k0_files
        or any((root / name).exists() for name in ("FINAL_STATUS.json", "manifest.json", "COMPLETE"))
    ):
        raise RuntimeError("failed confirmation inventory differs from frozen evidence")
    _verify_source_snapshot(root, snapshot)
    development_frozen = _read_json(DEVELOPMENT_ROOT / "frozen_settings.json")
    if (
        metadata.get("protocol") != PROTOCOL
        or metadata.get("phase") != "confirmation"
        or metadata.get("output_root") != str(root)
        or metadata.get("source_tree_sha256")
        != EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256
        or metadata.get("config_sha256") != _file_sha256(CONFIG_PATH)
        or metadata.get("frozen_settings") != development_frozen
        or metadata.get("frozen_settings_sha256")
        != development_frozen["frozen_settings_sha256"]
    ):
        raise RuntimeError("failed confirmation metadata differs from its frozen contract")

    devices = tuple(metadata["devices"])
    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    preset = replace(
        protocol.datasets["mimic_iv"],
        seeds=config.confirmation_seeds["mimic_iv"],
        bootstrap_seed=config.confirmation_bootstrap_seeds["mimic_iv"],
    )
    mapping = _dataset_seed_device_mapping("mimic_iv", preset.seeds, devices)
    candidate_hash = _json_sha256([])
    result_hashes = []
    for seed in preset.seeds:
        payload = _read_json(root / f"mimic_iv/support/seed_{seed:06d}.json")
        _validate_seed_payload(
            payload,
            phase="confirmation_support",
            preset=preset,
            seed=seed,
            device=mapping[seed],
            source_hash=EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256,
            candidate_hash=candidate_hash,
        )
        result = payload["result"]
        if result.get("outcome_blind") is not True or result.get("passed") is not True:
            raise RuntimeError("failed confirmation support evidence is not 20/20")
        result_hashes.append(
            {"seed": seed, "result_sha256": _json_sha256(result)}
        )
    return {
        "role": "immutable_pre_k0_failed_confirmation_evidence",
        "root": str(root),
        "file_count": len(inventory),
        "inventory_sha256": inventory_hash,
        "metadata_sha256": EXPECTED_FAILED_CONFIRMATION_METADATA_SHA256,
        "support_seed_artifact_count": len(support_inventory),
        "support_seed_file_set_sha256": _json_sha256(support_inventory),
        "support_complete_bytes": support_complete.stat().st_size,
        "support_complete_sha256": _file_sha256(support_complete),
        "support_result_hashes": result_hashes,
        "support_result_set_sha256": _json_sha256(result_hashes),
        "support_outcome_blind_pass_count": 20,
        "k0_directory_present_and_empty": True,
        "k0_artifact_count": 0,
        "root_final_status_present": False,
        "root_manifest_present": False,
        "root_complete_present": False,
        "coverage_or_science_artifact_count": 0,
        "source_tree_sha256": metadata["source_tree_sha256"],
        "frozen_settings_sha256": metadata["frozen_settings_sha256"],
    }


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden in failed evidence: {path}")
        if not path.is_file():
            continue
        raw = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return rows


def _administrative_source_delta(
    *,
    source_hash: str,
    source_snapshot: Mapping[str, Any],
    development_binding: Mapping[str, Any],
) -> dict[str, Any]:
    development_metadata = _read_json(DEVELOPMENT_ROOT / "metadata.json")
    old_contract = development_metadata["source_snapshot"]
    old_manifest_path = DEVELOPMENT_ROOT / old_contract["manifest_path"]
    old_manifest = _read_json(old_manifest_path)
    current_manifest = json.loads(source_snapshot["manifest_bytes"])
    old_rows = {row["path"]: row for row in old_manifest["files"]}
    current_rows = {row["path"]: row for row in current_manifest["files"]}
    changed = sorted(
        path
        for path in old_rows.keys() & current_rows.keys()
        if old_rows[path] != current_rows[path]
    )
    added = sorted(current_rows.keys() - old_rows.keys())
    removed = sorted(old_rows.keys() - current_rows.keys())
    old_runner = old_rows.get(RETRY_CHANGED_SOURCE_PATH)
    current_runner = current_rows.get(RETRY_CHANGED_SOURCE_PATH)
    if (
        development_binding.get("source_tree_sha256")
        != EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256
        or development_metadata.get("source_tree_sha256")
        != EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256
        or source_hash == EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256
        or source_hash != experiment_tree_sha256()
        or old_manifest.get("file_count") != EXPECTED_DEVELOPMENT_SOURCE_FILE_COUNT
        or current_manifest.get("file_count") != EXPECTED_DEVELOPMENT_SOURCE_FILE_COUNT
        or len(old_rows) != EXPECTED_DEVELOPMENT_SOURCE_FILE_COUNT
        or len(current_rows) != EXPECTED_DEVELOPMENT_SOURCE_FILE_COUNT
        or added
        or removed
        or changed != [RETRY_CHANGED_SOURCE_PATH]
        or not isinstance(old_runner, dict)
        or old_runner.get("sha256") != EXPECTED_DEVELOPMENT_RUNNER_SHA256
        or not isinstance(current_runner, dict)
        or current_runner.get("sha256") != _file_sha256(Path(__file__).resolve())
    ):
        raise RuntimeError("retry source differs beyond the one-file administrative delta")
    changed_file = {
        "path": RETRY_CHANGED_SOURCE_PATH,
        "before_bytes": old_runner["bytes"],
        "before_sha256": old_runner["sha256"],
        "after_bytes": current_runner["bytes"],
        "after_sha256": current_runner["sha256"],
    }
    return {
        "role": "one_file_administrative_runtime_adapter_delta",
        "development_source_tree_sha256": EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256,
        "retry_source_tree_sha256": source_hash,
        "source_file_count_before": len(old_rows),
        "source_file_count_after": len(current_rows),
        "changed_file_count": 1,
        "changed_files": [changed_file],
        "added_paths": [],
        "removed_paths": [],
        "unchanged_file_count": len(old_rows) - 1,
        "config_files_changed": False,
        "method_modules_changed": False,
        "repair_grid_or_selector_changed": False,
        "semantic_scope": (
            "deserialize FrozenAnchor bandwidth and donor_weighting from its exact "
            "frozen theta payload before constructing the transition environment"
        ),
        "repair_theta_semantics_changed": False,
        "frozen_theta_changed": False,
    }


def _administrative_retry_amendment(
    *,
    config: FidelityV4Config,
    source_hash: str,
    source_snapshot: Mapping[str, Any],
    development_binding: Mapping[str, Any],
    frozen_settings: Mapping[str, Any],
    failed_attempt_binding: Mapping[str, Any] | None,
    confirmation_rng_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if failed_attempt_binding is None:
        raise RuntimeError("retry amendment requires failed-attempt evidence")
    current_failed = _failed_confirmation_attempt_binding(config)
    if current_failed != dict(failed_attempt_binding):
        raise RuntimeError("failed-attempt evidence changed before retry amendment")
    source_delta = _administrative_source_delta(
        source_hash=source_hash,
        source_snapshot=source_snapshot,
        development_binding=development_binding,
    )
    if (
        frozen_settings.get("development_source_tree_sha256")
        != EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256
        or frozen_settings.get("frozen_settings_sha256")
        != failed_attempt_binding.get("frozen_settings_sha256")
        or confirmation_rng_audit.get("unauthorized_collision_count") != 0
        or confirmation_rng_audit.get("same_prespecified_mapping_collision_count")
        != EXPECTED_CONFIRMATION_STREAM_COUNT
    ):
        raise RuntimeError("retry amendment differs from the frozen confirmation contract")
    return {
        "protocol": RETRY_PROTOCOL,
        "role": "administrative_retry_after_pre_k0_runtime_adapter_failure",
        "failure_stage": "after_mimic_support_before_any_k0",
        "failure_cause": (
            "FrozenAnchor serialized bandwidth and donor_weighting but the runner "
            "incorrectly attempted missing runtime attributes"
        ),
        "failed_attempt_binding": dict(failed_attempt_binding),
        "failed_attempt_binding_sha256": _json_sha256(failed_attempt_binding),
        "source_delta": source_delta,
        "source_delta_sha256": _json_sha256(source_delta),
        "development_binding_sha256": _json_sha256(development_binding),
        "frozen_settings_sha256": frozen_settings["frozen_settings_sha256"],
        "same_frozen_dataset_settings": True,
        "same_prespecified_seed_banks": True,
        "same_first_fresh_confirmation_lineage": True,
        "second_fresh_bank_claimed": False,
        "independent_rng_bank_claimed": False,
        "support_recomputed_from_scratch": True,
        "failed_attempt_artifacts_reused": False,
        "mimic_support_exact_semantic_comparison_required_before_k0": True,
        "failed_attempt_k0_artifact_count": 0,
        "unauthorized_rng_collision_count": 0,
        "coverage_generation_permitted": False,
        "scientific_result_execution_path_present": False,
    }


def _metadata_only_artifact_rng_ids(
    root: Path,
    *,
    excluded_roots: Sequence[Path],
) -> set[int]:
    values: set[int] = set()
    if not root.exists():
        return values
    excluded = tuple(path.resolve() for path in excluded_roots)
    for path in root.rglob("*"):
        resolved = path.resolve()
        if any(resolved == item or item in resolved.parents for item in excluded):
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
        relative = path.relative_to(root)
        if _forbidden_result_path(relative):
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
    config: FidelityV4Config,
    devices: tuple[str, ...],
    source_hash: str,
    source_snapshot: Mapping[str, Any],
    parent_binding: Mapping[str, Any],
    development_reuse_audit: Mapping[str, Any],
    confirmation_rng_audit: Mapping[str, Any],
    development_binding: Mapping[str, Any] | None = None,
    frozen_settings: Mapping[str, Any] | None = None,
    retry_amendment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config_bytes = CONFIG_PATH.read_bytes()
    active_seeds = (
        config.development_seeds
        if phase == "development"
        else config.confirmation_seeds
    )
    active_datasets = REPAIR_DATASETS if phase == "development" else DATASETS
    metadata: dict[str, Any] = {
        "protocol": PROTOCOL,
        "phase": phase,
        "role": "coverage_blind_dataset_independent_k0_repair",
        "decision_scope": "per_dataset_independent",
        "cross_dataset_conjunction_permitted": False,
        "canonical_scpcp_mutation_permitted": False,
        "coverage_generation_permitted": False,
        "scientific_result_execution_path_present": False,
        "datasets": list(DATASETS),
        "active_seed_datasets": list(active_datasets),
        "devices": list(devices),
        "output_root": str(output_root),
        "seed_to_device": _seed_device_mapping(
            active_seeds,
            devices,
            datasets=active_datasets,
        ),
        "source_tree_sha256": source_hash,
        "source_snapshot": dict(source_snapshot),
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_bytes": len(config_bytes),
        "parent_v3_binding": dict(parent_binding),
        "parent_v3_binding_sha256": _json_sha256(parent_binding),
        "development_rng_reuse_audit": dict(development_reuse_audit),
        "confirmation_rng_audit": dict(confirmation_rng_audit),
        "selector_contract": {
            "version": SELECTOR_VERSION,
            "scope": "per_dataset_independent",
            "minimum_pass_count": 19,
            "required_structural_pass_count": 20,
            "cross_dataset_conjunction_permitted": False,
        },
        "candidate_contract": {
            "anchors": {
                dataset: FROZEN_ANCHORS[dataset].to_dict()
                for dataset in ANCHOR_DATASETS
            },
            "repairs": {
                dataset: [theta.to_dict() for theta in repair_candidates(dataset)]
                for dataset in REPAIR_DATASETS
            },
            "stagewise_zscore": {
                "source": "D_env_only",
                "pooling": "per_stage_pooled_over_actions",
                "retrieval": "action_conditional_after_shared_stage_scaling",
                "estimation_dtype": "float64",
                "population_sd": True,
                "sd_floor": config.stagewise_sd_floor,
            },
            "full_cell_neighbor_sentinel": 10_000,
            "effective_library_sizes_recorded_per_seed_candidate": True,
        },
        "k0_gate": {
            **METRIC_THRESHOLDS,
            "minimum_available_seed_fraction": 0.95,
            "systematic_replays": 16,
            "structural_invariants": "all_20_required",
            "interpretation": "operational_gate_not_confidence_interval",
        },
        "seed_roles": {
            "development_reused_v3_k0_lineage": {
                dataset: list(config.development_seeds[dataset])
                for dataset in REPAIR_DATASETS
            },
            "confirmation_fresh_operational_gate": {
                dataset: list(config.confirmation_seeds[dataset])
                for dataset in DATASETS
            },
            "independent_patient_confirmation_claimed": False,
        },
        "default_parity": {
            "archived_v3_controlled_transition_sha256": (
                config.archived_v3_controlled_transition_sha256
            ),
            "current_controlled_transition_sha256": (
                config.current_controlled_transition_sha256
            ),
            "legacy_transition_mode": "ridge_residual",
            "legacy_outcome_residual_mode": "standardized",
        },
    }
    if phase in {"confirmation", "confirmation_retry"}:
        if development_binding is None or frozen_settings is None:
            raise ValueError("confirmation metadata requires a development freeze")
        metadata["development_binding"] = dict(development_binding)
        metadata["development_binding_sha256"] = _json_sha256(development_binding)
        metadata["frozen_settings"] = dict(frozen_settings)
        metadata["frozen_settings_sha256"] = frozen_settings[
            "frozen_settings_sha256"
        ]
    if phase == "confirmation_retry":
        if retry_amendment is None:
            raise ValueError("confirmation retry metadata requires an amendment")
        metadata["role"] = "coverage_blind_administrative_confirmation_retry"
        metadata["administrative_retry_protocol"] = RETRY_PROTOCOL
        metadata["administrative_retry_amendment"] = dict(retry_amendment)
        metadata["administrative_retry_amendment_sha256"] = _json_sha256(
            retry_amendment
        )
        metadata["seed_roles"] = {
            **metadata["seed_roles"],
            "confirmation_retry": {
                "role": "deterministic_replay_of_same_first_fresh_lineage",
                "same_prespecified_seed_banks": {
                    dataset: list(config.confirmation_seeds[dataset])
                    for dataset in DATASETS
                },
                "second_fresh_bank_claimed": False,
                "independent_rng_bank_claimed": False,
                "failed_attempt_artifacts_reused": False,
                "support_recomputed_from_scratch": True,
            },
        }
    return metadata


def _seed_device_mapping(
    seeds_by_dataset: Mapping[str, Sequence[int]],
    devices: Sequence[str],
    *,
    datasets: Sequence[str],
) -> dict[str, str]:
    if tuple(seeds_by_dataset) != tuple(datasets) or not devices:
        raise ValueError("seed/device mapping differs from its active dataset order")
    return {
        f"{dataset}/base_{seed}": device
        for dataset in datasets
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
    offset = DATASETS.index(dataset) * 20
    return {
        seed: devices[(offset + index) % len(devices)]
        for index, seed in enumerate(seeds)
    }


def _active_source_contract() -> tuple[str, dict[str, Any]]:
    source_hash = experiment_tree_sha256()
    snapshot = _build_source_snapshot()
    if experiment_tree_sha256() != source_hash:
        raise RuntimeError("source/config changed while building the v4 snapshot")
    return source_hash, snapshot


def _build_source_snapshot() -> dict[str, Any]:
    paths = [
        *sorted((ROOT / "src/scpcp").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "tools").glob("*.py")),
        *sorted((ROOT / "configs").glob("*.yaml")),
        ROOT / "pyproject.toml",
    ]
    relative_paths = [path.relative_to(ROOT).as_posix() for path in paths]
    if len(relative_paths) != len(set(relative_paths)) or any(
        not path.is_file() for path in paths
    ):
        raise RuntimeError("v4 source snapshot file set is invalid")
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
    manifest = {
        "protocol": PROTOCOL,
        "format": "deterministic_uncompressed_pax_tar",
        "file_count": len(files),
        "files": files,
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
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
    if metadata["development_rng_reuse_audit"].get("status") != "passed_before_launch":
        raise RuntimeError("development RNG reuse audit did not pass")
    if metadata["confirmation_rng_audit"].get("status") != "passed_before_launch":
        raise RuntimeError("confirmation RNG collision audit did not pass")
    if resume:
        if not root.is_dir() or not (root / "metadata.json").is_file():
            raise FileNotFoundError("resume requires an existing v4 metadata.json")
        if _read_json(root / "metadata.json") != metadata:
            raise RuntimeError("resume metadata differs from the active v4 contract")
        _verify_source_snapshot(root, metadata["source_snapshot"])
        return
    if root.exists():
        raise FileExistsError(f"fresh v4 output already exists: {root}")
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


def _complete_and_valid(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV4Config,
) -> bool:
    if not (root / "COMPLETE").exists():
        return False
    try:
        _validate_root_bundle(root, metadata, config=config)
    except (Exception, KeyboardInterrupt):
        _unlink_root_complete(root)
        raise
    return True


def _finalize_root(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    source_hash: str,
    config: FidelityV4Config,
) -> None:
    if experiment_tree_sha256() != source_hash:
        raise RuntimeError("source/config changed during the v4 phase")
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("v4 metadata changed during the phase")
    if _validated_parent_v3_binding(config) != metadata["parent_v3_binding"]:
        raise RuntimeError("parent v3 provenance changed during v4")
    if metadata["phase"] == "confirmation_retry":
        expected_failed = metadata["administrative_retry_amendment"][
            "failed_attempt_binding"
        ]
        if _failed_confirmation_attempt_binding(config) != expected_failed:
            raise RuntimeError("failed confirmation evidence changed before retry commit")
    _assert_no_forbidden_result_paths(root)
    _write_manifest(root)
    _validate_root_contents(root, metadata, config=config)
    final = _read_json(root / "FINAL_STATUS.json")
    manifest_hash = _file_sha256(root / "manifest.json")
    complete = (
        f"complete phase={metadata['phase']} source_tree_sha256={source_hash} "
        f"config_sha256={metadata['config_sha256']} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={manifest_hash}\n"
    )
    # COMPLETE is intentionally the final filesystem mutation.
    _write_text(root / "COMPLETE", complete)
    try:
        _validate_root_bundle(root, metadata, config=config)
    except (Exception, KeyboardInterrupt):
        _unlink_root_complete(root)
        raise


def _validate_root_bundle(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV4Config,
) -> None:
    _validate_root_contents(root, metadata, config=config)
    final = _read_json(root / "FINAL_STATUS.json")
    expected = (
        f"complete phase={metadata['phase']} "
        f"source_tree_sha256={metadata['source_tree_sha256']} "
        f"config_sha256={metadata['config_sha256']} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={_file_sha256(root / 'manifest.json')}\n"
    )
    if not (root / "COMPLETE").is_file() or (root / "COMPLETE").read_text() != expected:
        raise RuntimeError("v4 COMPLETE marker differs")


def _validate_root_contents(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV4Config,
) -> None:
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("v4 root metadata mismatch")
    parent = metadata.get("parent_v3_binding")
    if (
        not isinstance(parent, dict)
        or metadata.get("parent_v3_binding_sha256") != _json_sha256(parent)
        or _validated_parent_v3_binding(config) != parent
    ):
        raise RuntimeError("v4 root parent binding differs")
    _verify_source_snapshot(root, metadata["source_snapshot"])
    _verify_manifest(root)
    _assert_no_forbidden_result_paths(root)
    if metadata["phase"] == "development":
        _validate_development_artifacts(root, metadata, config=config)
    elif metadata["phase"] == "confirmation":
        _validate_confirmation_artifacts(root, metadata, config=config)
    elif metadata["phase"] == "confirmation_retry":
        _validate_confirmation_retry_artifacts(root, metadata, config=config)
    else:
        raise RuntimeError("unknown v4 root phase")
    expected_paths = _expected_artifact_paths(root, metadata, config=config)
    _assert_exact_artifact_file_set(root, expected_paths)


def _verify_source_snapshot(root: Path, contract: Mapping[str, Any]) -> None:
    for name in ("archive", "manifest"):
        path = root / contract[f"{name}_path"]
        if (
            not path.is_file()
            or path.stat().st_size != contract[f"{name}_bytes"]
            or _file_sha256(path) != contract[f"{name}_sha256"]
        ):
            raise RuntimeError(f"v4 source snapshot {name} differs")


def _write_manifest(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden in a v4 bundle: {path}")
        if not path.is_file():
            continue
        relative_path = path.relative_to(root)
        if relative_path in {Path("manifest.json"), Path("COMPLETE")}:
            continue
        if ".tmp-" in path.name:
            raise RuntimeError(f"temporary artifact remains: {path}")
        _resolve_inside_root(root, relative_path)
        entries.append(
            {
                "path": relative_path.as_posix(),
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
    if (
        set(manifest) != {"protocol", "artifact_count", "artifacts"}
        or manifest.get("protocol") != PROTOCOL
        or not isinstance(entries, list)
    ):
        raise RuntimeError("invalid v4 manifest header")
    expected = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise RuntimeError("malformed v4 manifest entry")
        relative = Path(entry["path"])
        if relative in {Path("manifest.json"), Path("COMPLETE")}:
            raise RuntimeError("v4 manifest must not contain root commit files")
        path = _resolve_inside_root(root, relative)
        if path.resolve() in expected:
            raise RuntimeError(f"duplicate v4 manifest entry: {relative.as_posix()}")
        expected.add(path.resolve())
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != entry["bytes"]
            or _file_sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"v4 manifest mismatch: {path}")
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden in a v4 bundle: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative in {Path("manifest.json"), Path("COMPLETE")}:
            continue
        observed.add(_resolve_inside_root(root, relative))
    if (
        observed != expected
        or manifest.get("artifact_count") != len(entries)
        or len(expected) != len(entries)
    ):
        raise RuntimeError("v4 manifest file set differs")


def _expected_artifact_paths(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV4Config,
) -> set[Path]:
    """Rebuild the exact allowed file set after semantic JSON validation.

    Every non-provenance JSON is separately checked against an exact schema or
    a semantically recomputed value.  The two source-snapshot files are trusted
    provenance whose byte lengths and hashes are bound by metadata.
    """

    snapshot = metadata["source_snapshot"]
    expected = {
        Path("metadata.json"),
        Path("FINAL_STATUS.json"),
        Path(snapshot["archive_path"]),
        Path(snapshot["manifest_path"]),
    }
    if metadata["phase"] == "development":
        expected.add(Path("frozen_settings.json"))
        for dataset in REPAIR_DATASETS:
            expected.add(Path("repair") / dataset / "COMPLETE")
            expected.add(Path("selection") / f"{dataset}.json")
            expected.update(
                Path("repair") / dataset / f"seed_{seed:06d}.json"
                for seed in config.development_seeds[dataset]
            )
        return expected

    if metadata["phase"] == "confirmation_retry":
        expected.update(
            {
                Path("administrative_retry_amendment.json"),
                Path("support_replay_verification.json"),
            }
        )
    frozen = metadata["frozen_settings"]
    theta_by_dataset = frozen["theta_by_dataset"]
    for dataset in DATASETS:
        dataset_root = Path(dataset)
        expected.update(
            {
                dataset_root / "gate.json",
                dataset_root / "COMPLETE",
            }
        )
        if dataset not in theta_by_dataset:
            continue
        expected.add(dataset_root / "support" / "COMPLETE")
        expected.update(
            dataset_root / "support" / f"seed_{seed:06d}.json"
            for seed in config.confirmation_seeds[dataset]
        )
        gate = _read_json(root / dataset_root / "gate.json")
        if gate["support_pass_count"] < 19:
            continue
        expected.add(dataset_root / "k0_fidelity" / "COMPLETE")
        expected.update(
            dataset_root / "k0_fidelity" / f"seed_{seed:06d}.json"
            for seed in config.confirmation_seeds[dataset]
        )
    return expected


def _assert_exact_artifact_file_set(
    root: Path,
    expected_relative_paths: set[Path],
) -> None:
    excluded = {Path("manifest.json"), Path("COMPLETE")}
    if any(path.is_absolute() or ".." in path.parts for path in expected_relative_paths):
        raise RuntimeError("expected v4 artifact path escapes its root")
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden in a v4 bundle: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative not in excluded:
            observed.add(relative)
    if observed != expected_relative_paths:
        missing = sorted(path.as_posix() for path in expected_relative_paths - observed)
        extra = sorted(path.as_posix() for path in observed - expected_relative_paths)
        raise RuntimeError(
            f"v4 exact artifact file set differs; missing={missing}; extra={extra}"
        )


def _resolve_inside_root(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("v4 manifest path escapes its root")
    resolved_root = root.resolve()
    path = (root / relative).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise RuntimeError("v4 manifest path escapes its root")
    return path


def _unlink_root_complete(root: Path) -> None:
    complete = root / "COMPLETE"
    complete.unlink(missing_ok=True)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_development_artifacts(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV4Config,
) -> None:
    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    rows_by_dataset = {}
    for dataset in REPAIR_DATASETS:
        candidates = repair_candidates(dataset)
        preset = replace(
            protocol.datasets[dataset],
            seeds=config.development_seeds[dataset],
        )
        rows_by_dataset[dataset] = _load_seed_results(
            root / "repair" / dataset,
            phase="development_repair",
            preset=preset,
            devices=tuple(metadata["devices"]),
            candidates=candidates,
            source_hash=metadata["source_tree_sha256"],
        )
    parent_status = validate_parent_v3_bundle(config, workspace_root=ROOT)
    expected_final, selections = _development_decision(rows_by_dataset, parent_status)
    if _read_json(root / "FINAL_STATUS.json") != expected_final:
        raise RuntimeError("v4 development decision differs on semantic recomputation")
    expected_selection_paths = {
        root / "selection" / f"{dataset}.json" for dataset in REPAIR_DATASETS
    }
    observed_selection_paths = set((root / "selection").glob("*.json"))
    if observed_selection_paths != expected_selection_paths:
        raise RuntimeError("v4 development selection file set differs")
    for dataset, selection in selections.items():
        if _read_json(root / "selection" / f"{dataset}.json") != selection:
            raise RuntimeError(f"{dataset} selection differs on semantic recomputation")
    expected_frozen = _frozen_settings(expected_final, metadata)
    if _read_json(root / "frozen_settings.json") != expected_frozen:
        raise RuntimeError("v4 frozen settings differ from the development decision")


def _verify_development_for_confirmation(
    root: Path,
    *,
    config: FidelityV4Config,
    current_parent_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _read_json(root / "metadata.json")
    if metadata.get("protocol") != PROTOCOL or metadata.get("phase") != "development":
        raise RuntimeError("confirmation requires a v4 development root")
    if not _complete_and_valid(root, metadata, config=config):
        raise RuntimeError("confirmation requires a completed v4 development root")
    if metadata.get("parent_v3_binding") != dict(current_parent_binding):
        raise RuntimeError("development parent binding differs at confirmation")
    final = _read_json(root / "FINAL_STATUS.json")
    frozen = _read_json(root / "frozen_settings.json")
    stored_hash = frozen.get("frozen_settings_sha256")
    unhashed = {
        key: value for key, value in frozen.items() if key != "frozen_settings_sha256"
    }
    if stored_hash != _json_sha256(unhashed):
        raise RuntimeError("frozen settings self-hash differs")
    if frozen.get("development_decision_sha256") != _json_sha256(final):
        raise RuntimeError("frozen settings do not bind the development decision")
    if frozen.get("development_config_sha256") != metadata["config_sha256"]:
        raise RuntimeError("frozen settings do not bind the development config")
    if frozen.get("parent_v3_binding_sha256") != metadata[
        "parent_v3_binding_sha256"
    ]:
        raise RuntimeError("frozen settings do not bind the v3 parent")
    theta_by_dataset = frozen.get("theta_by_dataset")
    if not isinstance(theta_by_dataset, dict) or not set(theta_by_dataset) <= set(DATASETS):
        raise RuntimeError("frozen setting dataset set differs")
    for dataset, theta in theta_by_dataset.items():
        if _theta_from_dict(theta).dataset != dataset:
            raise RuntimeError("frozen setting dataset identity differs")
    manifest_path = root / "manifest.json"
    complete_path = root / "COMPLETE"
    binding = {
        "root": str(root),
        "manifest_sha256": _file_sha256(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "complete_sha256": _file_sha256(complete_path),
        "complete_bytes": complete_path.stat().st_size,
        "final_status_sha256": _json_sha256(final),
        "frozen_settings_sha256": stored_hash,
        "source_tree_sha256": metadata["source_tree_sha256"],
        "config_sha256": metadata["config_sha256"],
        "parent_v3_binding_sha256": metadata["parent_v3_binding_sha256"],
    }
    return binding, frozen


def _load_validated_confirmation_gates(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV4Config,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    development_binding = metadata.get("development_binding")
    frozen = metadata.get("frozen_settings")
    if (
        not isinstance(development_binding, dict)
        or metadata.get("development_binding_sha256")
        != _json_sha256(development_binding)
        or not isinstance(frozen, dict)
        or metadata.get("frozen_settings_sha256")
        != frozen.get("frozen_settings_sha256")
    ):
        raise RuntimeError("confirmation development binding is malformed")
    current_binding, current_frozen = _verify_development_for_confirmation(
        DEVELOPMENT_ROOT,
        config=config,
        current_parent_binding=metadata["parent_v3_binding"],
    )
    if current_binding != development_binding or current_frozen != frozen:
        raise RuntimeError("confirmation development binding changed")

    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    gates: dict[str, dict[str, Any]] = {}
    support_by_dataset: dict[str, list[dict[str, Any]]] = {}
    theta_by_dataset = frozen["theta_by_dataset"]
    for dataset in DATASETS:
        theta_payload = theta_by_dataset.get(dataset)
        dataset_root = root / dataset
        if theta_payload is None:
            if (dataset_root / "support").exists() or (
                dataset_root / "k0_fidelity"
            ).exists():
                raise RuntimeError(f"{dataset} fresh bank opened after development no-go")
            gate = _unopened_confirmation_gate(dataset)
        else:
            theta = _theta_from_dict(theta_payload)
            preset = replace(
                protocol.datasets[dataset],
                seeds=config.confirmation_seeds[dataset],
                bootstrap_seed=config.confirmation_bootstrap_seeds[dataset],
            )
            support_rows = _load_seed_results(
                dataset_root / "support",
                phase="confirmation_support",
                preset=preset,
                devices=tuple(metadata["devices"]),
                candidates=(),
                source_hash=metadata["source_tree_sha256"],
            )
            support_by_dataset[dataset] = support_rows
            support_count = sum(bool(row["passed"]) for row in support_rows)
            if support_count >= 19:
                k0_rows = _load_seed_results(
                    dataset_root / "k0_fidelity",
                    phase="confirmation_k0",
                    preset=preset,
                    devices=tuple(metadata["devices"]),
                    candidates=(theta,),
                    source_hash=metadata["source_tree_sha256"],
                )
            else:
                if (dataset_root / "k0_fidelity").exists():
                    raise RuntimeError(f"{dataset} K0 opened after support no-go")
                k0_rows = []
            gate = _confirmation_gate(
                dataset,
                theta_payload,
                support_rows,
                k0_rows,
            )
        if _read_json(dataset_root / "gate.json") != gate:
            raise RuntimeError(f"{dataset} confirmation gate differs")
        if (dataset_root / "COMPLETE").read_text() != gate["status"].lower() + "\n":
            raise RuntimeError(f"{dataset} confirmation COMPLETE differs")
        gates[dataset] = gate
    return gates, support_by_dataset


def _validate_confirmation_artifacts(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV4Config,
) -> None:
    gates, _ = _load_validated_confirmation_gates(root, metadata, config=config)
    expected_final = _confirmation_final(gates)
    if _read_json(root / "FINAL_STATUS.json") != expected_final:
        raise RuntimeError("v4 confirmation final status differs")


def _validate_confirmation_retry_artifacts(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV4Config,
) -> None:
    amendment = metadata.get("administrative_retry_amendment")
    if (
        metadata.get("administrative_retry_protocol") != RETRY_PROTOCOL
        or not isinstance(amendment, dict)
        or metadata.get("administrative_retry_amendment_sha256")
        != _json_sha256(amendment)
        or _read_json(root / "administrative_retry_amendment.json") != amendment
    ):
        raise RuntimeError("confirmation retry amendment binding differs")
    failed_binding = amendment.get("failed_attempt_binding")
    if (
        not isinstance(failed_binding, dict)
        or amendment.get("failed_attempt_binding_sha256")
        != _json_sha256(failed_binding)
        or _failed_confirmation_attempt_binding(config) != failed_binding
    ):
        raise RuntimeError("confirmation retry failed-attempt binding differs")
    gates, support_by_dataset = _load_validated_confirmation_gates(
        root,
        metadata,
        config=config,
    )
    verification = _verify_retry_support_replay(
        root,
        support_by_dataset,
        config=config,
        failed_attempt_binding=failed_binding,
    )
    if _read_json(root / "support_replay_verification.json") != verification:
        raise RuntimeError("confirmation retry support replay verification differs")
    expected_final = _confirmation_retry_final(gates, amendment)
    if _read_json(root / "FINAL_STATUS.json") != expected_final:
        raise RuntimeError("v4 confirmation retry final status differs")


def _load_seed_results(
    phase_root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    devices: tuple[str, ...],
    candidates: tuple[Theta, ...],
    source_hash: str,
) -> list[dict[str, Any]]:
    expected_paths = {phase_root / f"seed_{seed:06d}.json" for seed in preset.seeds}
    observed_paths = set(phase_root.glob("seed_*.json"))
    if observed_paths != expected_paths or (phase_root / "COMPLETE").read_text() != "complete\n":
        raise RuntimeError(f"{phase} seed artifact set is incomplete")
    mapping = _dataset_seed_device_mapping(preset.name, preset.seeds, devices)
    candidate_hash = _json_sha256([_theta_to_dict(theta) for theta in candidates])
    rows = []
    for seed in preset.seeds:
        payload = _read_json(phase_root / f"seed_{seed:06d}.json")
        _validate_seed_payload(
            payload,
            phase=phase,
            preset=preset,
            seed=seed,
            device=mapping[seed],
            source_hash=source_hash,
            candidate_hash=candidate_hash,
        )
        rows.append(payload["result"])
    return rows


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
    if any(payload.get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"{phase} seed payload provenance differs for {seed}")
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("coverage_generated") is not False:
        raise RuntimeError(f"{phase} seed result violates the information firewall")
    if result.get("seed") != seed or result.get("dataset") != preset.name:
        raise RuntimeError(f"{phase} seed result identity differs")

    if phase == "development_repair":
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
            or set(result.get("information_opened", ()))
            != {"support", "k0_fidelity", "context_identity"}
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
        _validate_shared_seed_candidate_provenance(candidates)
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
        if (
            result.get("phase") != "confirmation_support"
            or result.get("confirmation_label") != "fresh_split_operational_gate"
            or not v2._valid_support_result(result, preset)
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
        split = result.get("split_audit")
        _validate_split_audit_schema(split)
        if (
            result.get("phase") != "confirmation_k0"
            or result.get("confirmation_label") != "fresh_split_operational_gate"
            or result.get("independent_patient_confirmation_claimed") is not False
            or not v2._valid_split_audit(split)
            or _json_sha256([result.get("theta")]) != candidate_hash
        ):
            raise RuntimeError(f"invalid confirmation K0 result for {seed}")
        _validate_candidate_k0(
            result,
            preset=preset,
            seed=seed,
            split_audit=split,
        )
        return
    raise RuntimeError(f"unknown v4 seed phase: {phase}")


def _validate_candidate_k0(
    candidate: Mapping[str, Any],
    *,
    preset: DatasetPreset,
    seed: int,
    split_audit: Mapping[str, Any],
) -> None:
    theta = _theta_from_dict(candidate["theta"])
    if theta.dataset != preset.name:
        raise RuntimeError("candidate setting belongs to another dataset")
    metrics = candidate.get("metrics")
    detail = candidate.get("systematic_replay")
    identity = candidate.get("context_identity")
    if not isinstance(metrics, dict) or not isinstance(detail, dict):
        raise RuntimeError("candidate K0 metrics are malformed")
    if set(metrics) != {*METRIC_THRESHOLDS, "structural_invariants"}:
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
    _validate_systematic_replay_header(detail)
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
            or not all(
                math.isfinite(float(value)) and float(value) >= 0.0
                for value in values
            )
            or float(metrics[maximum_name]) != max(map(float, values))
        ):
            raise RuntimeError(f"candidate K0 vector differs: {vector_name}")
    active = detail.get("active_successor_coordinates_by_stage")
    invariants = detail.get("raw_structural_invariants_by_stage")
    if (
        not isinstance(active, list)
        or len(active) != preset.horizon
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in active
        )
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
    ratio = normalized_seed_ratio(metrics)
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
    _validate_context_identity(
        identity,
        theta=theta,
        preset=preset,
        split_audit=split_audit,
    )


def _validate_systematic_replay_header(detail: Mapping[str, Any]) -> None:
    expected = {
        "label": "logging-mixture one-step fidelity",
        "episode_weighted": True,
        "inference_unit": (
            "patient-disjoint episode query; M=16 quadrature, never 16N "
            "independent observations"
        ),
        "patient_chunk_size": v2.K0_PATIENT_CHUNK_SIZE,
        "expansion_formula": "u[t,i,m]=(U[t,i]+(m+0.5)/16) mod 1",
        "flatten_order": "stage, patient, systematic_offset (offset fastest)",
    }
    if any(detail.get(name) != value for name, value in expected.items()):
        raise RuntimeError("candidate K0 replay semantics differ")


def _validate_shared_seed_candidate_provenance(
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    """All transition candidates must reuse one exact nuisance/data context."""

    identity_fields = (
        "base_nuisance_context_sha256",
        "outcome_model_state_sha256",
        "behavior_policy_state_sha256",
        "split_patient_id_sha256",
        "active_config_sha256",
    )
    for name in identity_fields:
        values = {
            _json_sha256(item["context_identity"][name])
            if isinstance(item["context_identity"][name], dict)
            else item["context_identity"][name]
            for item in candidates
        }
        if len(values) != 1:
            raise RuntimeError(f"candidate nuisance provenance differs within seed: {name}")
    support_values = {
        _json_sha256(
            {
                "actual_library_sizes_by_stage_action": item["context_identity"][
                    "library_support"
                ]["actual_library_sizes_by_stage_action"],
                "stage_count": item["context_identity"]["library_support"][
                    "stage_count"
                ],
                "action_count": item["context_identity"]["library_support"][
                    "action_count"
                ],
            }
        )
        for item in candidates
    }
    if len(support_values) != 1:
        raise RuntimeError("candidate donor-library provenance differs within seed")


def _validate_context_identity(
    identity: object,
    *,
    theta: Theta,
    preset: DatasetPreset,
    split_audit: Mapping[str, Any],
) -> None:
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
            "library_support",
            "combined_sha256",
        },
        "candidate context identity",
    )
    assert isinstance(identity, Mapping)
    combined = identity["combined_sha256"]
    unhashed = {key: value for key, value in identity.items() if key != "combined_sha256"}
    split_hashes = identity["split_patient_id_sha256"]
    transform = identity["metric_transform"]
    library = identity["library_support"]
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
    _require_exact_keys(
        library,
        {
            "requested_neighbors",
            "uses_full_cell",
            "full_cell_neighbor_sentinel",
            "actual_library_sizes_by_stage_action",
            "effective_neighbor_counts_by_stage_action",
            "full_cell_verified",
            "stage_count",
            "action_count",
            "combined_sha256",
        },
        "candidate library support",
    )
    assert isinstance(transform, Mapping) and isinstance(library, Mapping)
    expected_source = "D_env_only" if theta.metric == "stagewise_zscore" else "identity"
    expected_pooling = (
        "per_stage_pooled_over_actions"
        if theta.metric == "stagewise_zscore"
        else "not_applicable"
    )
    expected_retrieval = (
        "action_conditional_after_shared_stage_scaling"
        if theta.metric == "stagewise_zscore"
        else "action_conditional_raw_geometry"
    )
    provenance_hash_names = (
        "base_nuisance_context_sha256",
        "outcome_model_state_sha256",
        "behavior_policy_state_sha256",
        "active_config_sha256",
    )
    if (
        combined != _json_sha256(unhashed)
        or identity["theta"] != _theta_to_dict(theta)
        or not all(v2._is_sha256(identity[name]) for name in provenance_hash_names)
        or not isinstance(split_hashes, dict)
        or set(split_hashes) != {"predictor", "fidelity", "environment"}
        or not all(v2._is_sha256(value) for value in split_hashes.values())
        or split_hashes != split_audit.get("role_patient_id_sha256")
        or transform["geometry"] != theta.metric
        or transform["source"] != expected_source
        or transform["pooling"] != expected_pooling
        or transform["retrieval"] != expected_retrieval
        or transform["stage_count"] != preset.horizon
        or transform["coordinate_count"] != 32
        or transform["estimation_dtype"] != "float64"
        or transform["population_sd"] is not True
        or transform["sd_floor"] != 1e-4
        or not v2._is_sha256(transform["center_sha256"])
        or not v2._is_sha256(transform["scale_sha256"])
    ):
        raise RuntimeError("candidate metric-transform identity differs")
    _validate_library_support(library, theta=theta, preset=preset)


def _validate_library_support(
    library: Mapping[str, Any],
    *,
    theta: Theta,
    preset: DatasetPreset,
) -> None:
    stored_hash = library["combined_sha256"]
    unhashed = {key: value for key, value in library.items() if key != "combined_sha256"}
    sizes = library["actual_library_sizes_by_stage_action"]
    effective = library["effective_neighbor_counts_by_stage_action"]
    action_count = library["action_count"]
    valid_matrix = lambda matrix: (
        isinstance(matrix, list)
        and len(matrix) == preset.horizon
        and all(
            isinstance(row, list)
            and len(row) == action_count
            and all(type(value) is int and value > 0 for value in row)
            for row in matrix
        )
    )
    uses_full_cell = theta.neighbors == 10_000
    if (
        stored_hash != _json_sha256(unhashed)
        or library["requested_neighbors"] != theta.neighbors
        or library["uses_full_cell"] is not uses_full_cell
        or library["full_cell_neighbor_sentinel"] != 10_000
        or library["stage_count"] != preset.horizon
        or type(action_count) is not int
        or action_count <= 0
        or not valid_matrix(sizes)
        or not valid_matrix(effective)
    ):
        raise RuntimeError("candidate effective-library identity differs")
    expected_effective = [
        [min(theta.neighbors, size) for size in stage_sizes]
        for stage_sizes in sizes
    ]
    expected_full = (
        all(
            used == size
            for stage_sizes, stage_used in zip(sizes, effective, strict=True)
            for size, used in zip(stage_sizes, stage_used, strict=True)
        )
        if uses_full_cell
        else None
    )
    if effective != expected_effective or library["full_cell_verified"] is not expected_full:
        raise RuntimeError("full-cell effective donor counts differ")


def _theta_to_dict(theta: Theta) -> dict[str, Any]:
    return theta.to_dict()


def _theta_from_dict(value: Mapping[str, Any]) -> Theta:
    if not isinstance(value, Mapping):
        raise ValueError("serialized setting must be an object")
    if "candidate_id" in value:
        theta = RepairTheta(
            dataset=str(value["dataset"]),
            candidate_id=str(value["candidate_id"]),
            metric=str(value["metric"]),
            neighbors=int(value["neighbors"]),
            weight=str(value["weight"]),
            transition_mode=str(value["transition_mode"]),
            outcome_residual_mode=str(value["outcome_residual_mode"]),
        )
        if theta.to_dict() != dict(value):
            raise ValueError("serialized repair setting differs from the frozen schema")
        return theta
    dataset = str(value.get("dataset"))
    anchor = FROZEN_ANCHORS.get(dataset)
    if anchor is None or anchor.to_dict() != dict(value):
        raise ValueError("serialized anchor differs from the frozen schema")
    return anchor


def _require_exact_keys(
    value: object,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError(f"{label} schema differs")


def _validate_split_audit_schema(value: object) -> None:
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
    assert isinstance(value, Mapping)
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
            is_theta_bandwidth = child_path[-2:] == ("theta", "bandwidth")
            if not is_theta_bandwidth and (
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
    if not isinstance(result, dict) or result.get("coverage_generated") is not False:
        raise RuntimeError("coverage firewall requires one false result-root marker")
    sanitized_result = {
        key: value for key, value in result.items() if key != "coverage_generated"
    }
    sanitized_payload = {
        key: sanitized_result if key == "result" else value
        for key, value in payload.items()
    }
    _reject_scientific_result_keys(sanitized_payload)


def _assert_no_forbidden_result_paths(root: Path) -> None:
    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and _forbidden_result_path(path.relative_to(root))
    ]
    if forbidden:
        raise RuntimeError(f"coverage firewall rejected result paths: {forbidden}")


def _forbidden_result_path(relative: Path) -> bool:
    normalized = relative.as_posix().lower()
    return any(token in normalized for token in FORBIDDEN_RESULT_PATH_TOKENS)


def _validate_devices(devices: tuple[str, ...]) -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run v4 from the workspace root: {ROOT}")
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("v4 requires exactly two distinct CUDA devices")
    if any(not device.startswith("cuda:") for device in devices):
        raise ValueError("v4 requires explicit CUDA devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
    for device in devices:
        index = torch.device(device).index
        if index is None or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {device}")


def _json_pointer(payload: Mapping[str, Any], pointer: str) -> Any:
    value: Any = payload
    for name in pointer.removeprefix("/").split("/"):
        if not isinstance(value, dict) or name not in value:
            raise RuntimeError(f"JSON pointer is missing: {pointer}")
        value = value[name]
    return value


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


def _write_once_json(path: Path, value: object) -> None:
    if path.exists():
        if _read_json(path) != value:
            raise RuntimeError(f"existing immutable JSON differs: {path}")
        return
    _write_json(path, value)


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


def _integer_set_sha256(values: Iterable[int]) -> str:
    return _json_sha256(sorted(set(int(value) for value in values)))


if __name__ == "__main__":
    main()
