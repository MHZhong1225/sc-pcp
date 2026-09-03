"""Run post-confirmation clinical-v3 overlap and all-six science.

The command has no scientific knobs.  It accepts only the frozen clinical-v3
development and confirmation roots, completes the overlap screen for all four
datasets, and only then opens the science phase.

Run::

    python scripts/run_controlled_clinical_fidelity_v3_science.py

Resume the identical root::

    python scripts/run_controlled_clinical_fidelity_v3_science.py --resume
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import tarfile
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_controlled_clinical_extension as v2  # noqa: E402
import scripts.run_controlled_clinical_fidelity_v3 as clinical  # noqa: E402
from scpcp.artifacts import experiment_tree_sha256  # noqa: E402
from scpcp.controlled_clinical_extension import (  # noqa: E402
    METHODS,
    ControlledClinicalExtensionConfig,
    DatasetPreset,
    donor_overlap_passes,
)
from scpcp.controlled_clinical_fidelity_v3 import (  # noqa: E402
    DATASETS,
    FidelityV3Config,
    KernelTheta,
    load_fidelity_v3_config,
)


PROTOCOL = "controlled_clinical_fidelity_v3_science_v1"
OUTPUT_ROOT = (
    ROOT / "results/work/controlled_clinical_fidelity_v3_science"
).resolve()
DEVELOPMENT_ROOT = clinical.DEVELOPMENT_ROOT
CONFIRMATION_ROOT = clinical.CONFIRMATION_ROOT
GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
PRIMARY_GAMMA = -4.0
PRIMARY_METRIC = "min_t mean_seed(target_coverage_seed_t)"
OVERLAP_PHASE = "donor_overlap"
SCIENCE_PHASE = "science"
SCIENCE_CONTRACT = {
    "methods": list(METHODS),
    "gammas": list(GAMMAS),
    "primary_default_gamma": PRIMARY_GAMMA,
    "primary_metric": PRIMARY_METRIC,
    "calibration_trajectories": 3_000,
    "grid_trajectories": 1_000,
    "evaluation_trajectories": 20_000,
    "target_adaptation_trajectories": {
        "Standard CP": 0,
        "ACI": 2_000,
        "MFCS": 0,
        "SPCI": 2_000,
        "PRC": 2_000,
        "SC-PCP": 0,
    },
    "policy_ratio_cap": 3.0,
    "bootstrap_resamples": 10_000,
    "bootstrap_seed_count": 20,
    "bootstrap_rule": (
        "frozen v2 10000x20 complete-seed uniforms/indices with deterministic "
        "selected-method subset projection"
    ),
    "common_random_numbers": {
        "source_calibration_across_gamma": True,
        "source_reference_across_gamma": True,
        "target_reference_across_methods_and_gamma": True,
        "online_baselines": "independent method streams reused across gamma",
    },
    "low_overlap_consequence": (
        "curves descriptive-only; no ranking, attainment, superiority, or "
        "cross-dataset conjunction"
    ),
}


@dataclass(frozen=True)
class ConfirmationAnchor:
    split_audit: Mapping[str, Any]
    kernel_identity: Mapping[str, Any]


@dataclass(frozen=True)
class GateBundle:
    fidelity_config: FidelityV3Config
    science_config: ControlledClinicalExtensionConfig
    source_tree_sha256: str
    parent_binding: Mapping[str, Any]
    rng_audit: Mapping[str, Any]
    development_binding: Mapping[str, Any]
    confirmation_binding: Mapping[str, Any]
    frozen_theta: Mapping[str, KernelTheta]
    seed_to_device: Mapping[str, Mapping[int, str]]
    anchors: Mapping[str, Mapping[int, ConfirmationAnchor]]
    contract: Mapping[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    clinical._validate_devices(devices)
    gates = verify_gate_bundle(devices=devices)
    run_post_confirmation_science(
        OUTPUT_ROOT,
        gates=gates,
        devices=devices,
        resume=args.resume,
    )
    print(OUTPUT_ROOT)


def verify_gate_bundle(*, devices: tuple[str, ...]) -> GateBundle:
    """Revalidate both v3 gates and every confirmation seed before science."""

    fidelity_config = load_fidelity_v3_config(clinical.CONFIG_PATH)
    science_config = v2.load_extension_config(clinical.V2_CONFIG_PATH)
    _validate_science_contract(science_config, fidelity_config)
    parent_root = (ROOT / fidelity_config.parent_v2_root).resolve()
    parent_binding = clinical.verify_parent_v2(parent_root)
    development_binding, frozen = clinical._verify_development_for_confirmation(
        DEVELOPMENT_ROOT,
        current_parent_binding=parent_binding,
    )
    development_metadata = clinical._read_json(DEVELOPMENT_ROOT / "metadata.json")
    confirmation_metadata = clinical._read_json(CONFIRMATION_ROOT / "metadata.json")
    clinical._validate_root_bundle(CONFIRMATION_ROOT, confirmation_metadata)

    active_source = experiment_tree_sha256()
    config_bytes = clinical.CONFIG_PATH.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    expected_confirmation_mapping = clinical._seed_device_mapping(
        fidelity_config.confirmation_seeds,
        devices,
    )
    if (
        development_metadata.get("source_tree_sha256") != active_source
        or confirmation_metadata.get("source_tree_sha256") != active_source
        or tuple(development_metadata.get("devices", ())) != devices
        or tuple(confirmation_metadata.get("devices", ())) != devices
    ):
        raise RuntimeError(
            "development, confirmation, and science must share source/devices"
        )
    if (
        development_metadata.get("protocol") != clinical.PROTOCOL
        or development_metadata.get("phase") != "development"
        or development_metadata.get("output_root") != str(DEVELOPMENT_ROOT)
        or confirmation_metadata.get("protocol") != clinical.PROTOCOL
        or confirmation_metadata.get("phase") != "confirmation"
        or confirmation_metadata.get("output_root") != str(CONFIRMATION_ROOT)
        or development_metadata.get("config_path")
        != clinical.CONFIG_PATH.relative_to(ROOT).as_posix()
        or confirmation_metadata.get("config_path")
        != clinical.CONFIG_PATH.relative_to(ROOT).as_posix()
        or development_metadata.get("config_sha256") != config_sha256
        or confirmation_metadata.get("config_sha256") != config_sha256
        or development_metadata.get("config_bytes") != len(config_bytes)
        or confirmation_metadata.get("config_bytes") != len(config_bytes)
        or development_metadata.get("selector_contract_sha256")
        != clinical._selector_contract_sha256()
        or confirmation_metadata.get("selector_contract_sha256")
        != clinical._selector_contract_sha256()
        or confirmation_metadata.get("seed_to_device")
        != expected_confirmation_mapping
    ):
        raise RuntimeError("development/confirmation live protocol binding differs")
    if (
        confirmation_metadata.get("parent_v2_binding") != parent_binding
        or confirmation_metadata.get("parent_v2_binding_sha256")
        != clinical._json_sha256(parent_binding)
        or confirmation_metadata.get("development_binding") != development_binding
        or confirmation_metadata.get("development_binding_sha256")
        != clinical._json_sha256(development_binding)
        or confirmation_metadata.get("frozen_theta") != frozen
        or confirmation_metadata.get("frozen_theta_sha256")
        != frozen["frozen_theta_sha256"]
    ):
        raise RuntimeError("confirmation does not bind the current development gates")

    current_rng = clinical.audit_confirmation_rng(
        fidelity_config,
        excluded_roots=(DEVELOPMENT_ROOT, CONFIRMATION_ROOT, OUTPUT_ROOT),
    )
    _validate_rng_binding(
        current_rng,
        development_metadata["confirmation_rng_audit"],
        confirmation_metadata["confirmation_rng_audit"],
    )

    final = clinical._read_json(CONFIRMATION_ROOT / "FINAL_STATUS.json")
    _require_exact_keys(
        final,
        {
            "protocol",
            "phase",
            "status",
            "datasets",
            "coverage_generated",
            "science_unlock_present",
            "failure_consequence",
        },
        "confirmation final status",
    )
    if (
        final["protocol"] != clinical.PROTOCOL
        or final["phase"] != "confirmation"
        or final["status"] != "CONFIRMATION_GO"
        or final["coverage_generated"] is not False
        or final["science_unlock_present"] is not False
        or final["failure_consequence"] is not None
        or set(final["datasets"]) != set(DATASETS)
    ):
        raise RuntimeError("all four confirmation gates must be GO before science")

    frozen_theta = {
        dataset: clinical._theta_from_dict(frozen["theta_by_dataset"][dataset])
        for dataset in DATASETS
    }
    seed_to_device: dict[str, dict[int, str]] = {}
    anchors: dict[str, dict[int, ConfirmationAnchor]] = {}
    for dataset in DATASETS:
        preset = replace(
            science_config.datasets[dataset],
            seeds=fidelity_config.confirmation_seeds[dataset],
            bootstrap_seed=fidelity_config.confirmation_bootstrap_seeds[dataset],
        )
        mapping = {
            seed: expected_confirmation_mapping[f"{dataset}/base_{seed}"]
            for seed in preset.seeds
        }
        if set(mapping.values()) != set(devices):
            raise RuntimeError(f"{dataset} confirmation device mapping differs")
        seed_to_device[dataset] = mapping
        anchors[dataset] = _validate_confirmation_dataset(
            dataset,
            preset=preset,
            theta=frozen_theta[dataset],
            source_hash=active_source,
            seed_to_device=mapping,
            final_dataset_status=final["datasets"][dataset],
        )

    confirmation_binding = _root_binding(CONFIRMATION_ROOT)
    contract = {
        "protocol": PROTOCOL,
        "source_tree_sha256": active_source,
        "development_binding": development_binding,
        "development_binding_sha256": clinical._json_sha256(development_binding),
        "confirmation_binding": confirmation_binding,
        "confirmation_binding_sha256": clinical._json_sha256(confirmation_binding),
        "parent_v2_binding": parent_binding,
        "parent_v2_binding_sha256": clinical._json_sha256(parent_binding),
        "rng_stream_mapping_sha256": current_rng[
            "new_rng_stream_mapping_sha256"
        ],
        "frozen_theta_sha256": frozen["frozen_theta_sha256"],
        "frozen_theta_by_dataset": {
            dataset: frozen_theta[dataset].to_dict() for dataset in DATASETS
        },
        "confirmation_seeds": {
            dataset: list(fidelity_config.confirmation_seeds[dataset])
            for dataset in DATASETS
        },
        "confirmation_role": "fresh_split_confirmation",
        "independent_patient_confirmation_claimed": False,
        "all_four_support_k0_gates": "GO",
        "science_contract": SCIENCE_CONTRACT,
    }
    return GateBundle(
        fidelity_config=fidelity_config,
        science_config=science_config,
        source_tree_sha256=active_source,
        parent_binding=parent_binding,
        rng_audit=current_rng,
        development_binding=development_binding,
        confirmation_binding=confirmation_binding,
        frozen_theta=frozen_theta,
        seed_to_device=seed_to_device,
        anchors=anchors,
        contract=contract,
    )


def _validate_science_contract(
    protocol: ControlledClinicalExtensionConfig,
    fidelity_config: FidelityV3Config,
) -> None:
    protocol.validate()
    if (
        tuple(METHODS)
        != ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")
        or tuple(protocol.gammas) != GAMMAS
        or (
            protocol.calibration_trajectories,
            protocol.grid_trajectories,
            protocol.reference_trajectories,
            protocol.online_trajectories,
            protocol.bootstrap_resamples,
        )
        != (3_000, 1_000, 20_000, 2_000, 10_000)
        or protocol.policy_ratio_cap != 3.0
        or dict(v2.TARGET_ADAPTATION_BUDGET)
        != SCIENCE_CONTRACT["target_adaptation_trajectories"]
    ):
        raise RuntimeError("clinical-v3 science constants differ from frozen v2")
    if any(len(fidelity_config.confirmation_seeds[name]) != 20 for name in DATASETS):
        raise RuntimeError("science requires every complete 20-seed confirmation bank")


def _validate_rng_binding(
    current: Mapping[str, Any],
    development: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> None:
    expected_hash = current.get("new_rng_stream_mapping_sha256")
    expected_mapping = current.get("new_rng_stream_mapping")
    for label, value in (
        ("current", current),
        ("development", development),
        ("confirmation", confirmation),
    ):
        if (
            value.get("status") != "passed_before_launch"
            or value.get("collision_count") != 0
            or value.get("collisions") != {}
            or value.get("internal_rng_streams_unique") is not True
            or value.get("new_rng_stream_count") != 1_304
            or value.get("new_rng_stream_mapping_sha256") != expected_hash
            or value.get("new_rng_stream_mapping") != expected_mapping
        ):
            raise RuntimeError(f"{label} confirmation RNG binding differs")


def _validate_confirmation_dataset(
    dataset: str,
    *,
    preset: DatasetPreset,
    theta: KernelTheta,
    source_hash: str,
    seed_to_device: Mapping[int, str],
    final_dataset_status: Mapping[str, Any],
) -> dict[int, ConfirmationAnchor]:
    root = CONFIRMATION_ROOT / dataset
    gate = clinical._read_json(root / "gate.json")
    _require_exact_keys(
        gate,
        {
            "protocol",
            "dataset",
            "status",
            "support_pass_count",
            "structural_pass_count",
            "k0_pass_count",
            "prespecified_seed_count",
            "theta",
            "coverage_generated",
            "confirmation_label",
            "independent_patient_confirmation_claimed",
        },
        f"{dataset} confirmation gate",
    )
    if gate != final_dataset_status or gate.get("status") != "CONFIRMATION_GATE_GO":
        raise RuntimeError(f"{dataset} confirmation gate is not exactly GO")
    if (
        gate.get("theta") != theta.to_dict()
        or gate.get("prespecified_seed_count") != 20
        or gate.get("coverage_generated") is not False
        or gate.get("confirmation_label") != "fresh_split_confirmation"
        or gate.get("independent_patient_confirmation_claimed") is not False
        or (root / "COMPLETE").read_text() != "confirmation_gate_go\n"
    ):
        raise RuntimeError(f"{dataset} frozen-theta confirmation contract differs")

    support_hash = clinical._json_sha256([])
    theta_hash = clinical._json_sha256([theta.to_dict()])
    support_rows = {}
    k0_rows = {}
    for seed in preset.seeds:
        support_payload = clinical._read_json(
            root / "support" / f"seed_{seed:06d}.json"
        )
        clinical._validate_seed_payload(
            support_payload,
            phase="confirmation_support",
            preset=preset,
            seed=seed,
            device=seed_to_device[seed],
            source_hash=source_hash,
            candidate_hash=support_hash,
        )
        k0_payload = clinical._read_json(
            root / "k0_fidelity" / f"seed_{seed:06d}.json"
        )
        clinical._validate_seed_payload(
            k0_payload,
            phase="confirmation_k0",
            preset=preset,
            seed=seed,
            device=seed_to_device[seed],
            source_hash=source_hash,
            candidate_hash=theta_hash,
        )
        support = support_payload["result"]
        k0 = k0_payload["result"]
        if (
            support["split_audit"] != k0["split_audit"]
            or k0["theta"] != theta.to_dict()
            or k0["confirmation_label"] != "fresh_split_confirmation"
            or k0["independent_patient_confirmation_claimed"] is not False
        ):
            raise RuntimeError(f"{dataset}/{seed} confirmation context differs")
        support_rows[seed] = support
        k0_rows[seed] = k0

    _require_exact_seed_files(root / "support", preset.seeds)
    _require_exact_seed_files(root / "k0_fidelity", preset.seeds)
    support_count = sum(bool(row["passed"]) for row in support_rows.values())
    structural_count = sum(
        bool(row["metrics"]["structural_invariants"])
        for row in k0_rows.values()
    )
    k0_count = sum(bool(row["passed"]) for row in k0_rows.values())
    if (
        support_count < 19
        or structural_count != 20
        or k0_count < 19
        or (
            gate["support_pass_count"],
            gate["structural_pass_count"],
            gate["k0_pass_count"],
        )
        != (support_count, structural_count, k0_count)
    ):
        raise RuntimeError(f"{dataset} support/K0 confirmation counts differ")
    return {
        seed: ConfirmationAnchor(
            split_audit=k0_rows[seed]["split_audit"],
            kernel_identity=k0_rows[seed]["context_identity"],
        )
        for seed in preset.seeds
    }


def _require_exact_seed_files(root: Path, seeds: Sequence[int]) -> None:
    expected = {root / f"seed_{seed:06d}.json" for seed in seeds}
    observed = set(root.glob("seed_*.json"))
    if observed != expected or (root / "COMPLETE").read_text() != "complete\n":
        raise RuntimeError(f"confirmation seed artifact set differs: {root}")


def run_post_confirmation_science(
    output_root: Path,
    *,
    gates: GateBundle,
    devices: tuple[str, ...],
    resume: bool,
) -> None:
    if output_root.resolve() != OUTPUT_ROOT:
        raise RuntimeError(f"science output root is frozen to {OUTPUT_ROOT}")
    source_hash, snapshot = _active_source_snapshot()
    if source_hash != gates.source_tree_sha256:
        raise RuntimeError("source changed after confirmation gate verification")
    metadata = _science_metadata(
        gates,
        devices=devices,
        source_snapshot=snapshot["contract"],
    )
    _prepare_root(output_root, metadata, snapshot, resume=resume)
    _require_partial_artifact_subset(output_root, metadata, gates)
    if (output_root / "COMPLETE").exists():
        _validate_complete_root(output_root, metadata, gates)
        return
    if (output_root / SCIENCE_PHASE).exists() and not _valid_global_overlap_marker(
        output_root
    ):
        raise RuntimeError("science artifacts exist without the global overlap gate")

    interpretations: dict[str, str] = {}
    for dataset in DATASETS:
        preset = _confirmation_preset(gates, dataset)
        rows = _run_phase(
            output_root / OVERLAP_PHASE / dataset,
            phase=OVERLAP_PHASE,
            preset=preset,
            theta=gates.frozen_theta[dataset],
            anchors=gates.anchors[dataset],
            seed_to_device=gates.seed_to_device[dataset],
            devices=devices,
            source_hash=source_hash,
            gate_contract_sha256=clinical._json_sha256(gates.contract),
            rng_mapping_sha256=gates.rng_audit["new_rng_stream_mapping_sha256"],
            worker=_overlap_worker,
            worker_arguments=(gates.science_config,),
            resume=resume,
        )
        interpretation = (
            "EMPIRICAL_OVERLAP_SCREEN_PASSED"
            if all(bool(row["passed"]) for row in rows)
            else "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        )
        interpretations[dataset] = interpretation
        _write_json(
            output_root / OVERLAP_PHASE / dataset / "summary.json",
            _overlap_summary(dataset, rows, interpretation),
        )

    _write_global_overlap_marker(output_root, interpretations)
    if not _valid_global_overlap_marker(output_root):
        raise RuntimeError("global overlap marker failed validation")

    dataset_status = {}
    for dataset in DATASETS:
        preset = _confirmation_preset(gates, dataset)
        rows_by_seed = _run_phase(
            output_root / SCIENCE_PHASE / dataset / "seeds",
            phase=SCIENCE_PHASE,
            preset=preset,
            theta=gates.frozen_theta[dataset],
            anchors=gates.anchors[dataset],
            seed_to_device=gates.seed_to_device[dataset],
            devices=devices,
            source_hash=source_hash,
            gate_contract_sha256=clinical._json_sha256(gates.contract),
            rng_mapping_sha256=gates.rng_audit["new_rng_stream_mapping_sha256"],
            worker=_science_worker,
            worker_arguments=(gates.science_config, interpretations[dataset]),
            resume=resume,
        )
        if any(
            result["interpretation_status"] != interpretations[dataset]
            for result in rows_by_seed
        ):
            raise RuntimeError(f"{dataset} science interpretation differs")
        raw_rows = [row for seed_result in rows_by_seed for row in seed_result["rows"]]
        science_root = output_root / SCIENCE_PHASE / dataset
        bootstrap = _ensure_bootstrap_artifacts(science_root, preset)
        summary = _science_summary(
            raw_rows,
            preset=preset,
            interpretation_status=interpretations[dataset],
            bootstrap_contract=bootstrap,
        )
        audit = _coverage_audit(
            raw_rows,
            preset=preset,
            summary=summary,
            interpretation_status=interpretations[dataset],
        )
        _write_json(science_root / "summary.json", summary)
        _write_json(science_root / "coverage_audit.json", audit)
        status = _dataset_science_status(dataset, interpretations[dataset])
        _write_json(science_root / "FINAL_STATUS.json", status)
        _write_text(
            science_root / "COMPLETE",
            (
                "curves\n"
                if interpretations[dataset] == "EMPIRICAL_OVERLAP_SCREEN_PASSED"
                else "curves-descriptive-only\n"
            ),
        )
        dataset_status[dataset] = status

    final = _final_science_status(dataset_status)
    _write_json(output_root / "FINAL_STATUS.json", final)
    _finalize_root(output_root, metadata, gates)


def _confirmation_preset(gates: GateBundle, dataset: str) -> DatasetPreset:
    return replace(
        gates.science_config.datasets[dataset],
        seeds=gates.fidelity_config.confirmation_seeds[dataset],
        bootstrap_seed=gates.fidelity_config.confirmation_bootstrap_seeds[dataset],
    )


def _overlap_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    theta: KernelTheta,
    anchor: ConfirmationAnchor,
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    context = clinical._context_with_theta(base_context, theta)
    kernel_identity = clinical._candidate_context_identity(
        base_context,
        context.environment,
        theta,
    )
    _assert_confirmation_context(anchor, base_context, kernel_identity)
    metrics, diagnostics = v2._donor_overlap_probe(
        context,
        seed=seed,
        protocol=protocol,
    )
    return {
        "seed": seed,
        "dataset": preset.name,
        "phase": OVERLAP_PHASE,
        "passed": donor_overlap_passes(metrics, protocol.donor_overlap_gate),
        "interpretation_if_failed": "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
        "metrics": v2.asdict(metrics),
        "diagnostics": diagnostics,
        "q_low": context.q_low,
        "q_high": context.q_high,
        "q_mid": context.q_low + 0.5 * (context.q_high - context.q_low),
        "n_actions": context.n_actions,
        "action_mapping": {
            str(key): value for key, value in context.action_mapping.items()
        },
        "split_audit": v2._split_audit(context.splits),
        "base_context_identity": v2._context_identity(base_context),
        "kernel_context_identity": kernel_identity,
        "theta": theta.to_dict(),
        "confirmation_anchor_identity_sha256": clinical._json_sha256(
            anchor.kernel_identity
        ),
    }


def _science_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    theta: KernelTheta,
    anchor: ConfirmationAnchor,
    protocol: ControlledClinicalExtensionConfig,
    interpretation_status: str,
) -> dict[str, Any]:
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    context = clinical._context_with_theta(base_context, theta)
    kernel_identity = clinical._candidate_context_identity(
        base_context,
        context.environment,
        theta,
    )
    _assert_confirmation_context(anchor, base_context, kernel_identity)
    rows = v2.run_science_seed(
        seed,
        preset=preset,
        device=device,
        protocol=protocol,
        context=context,
    )
    return {
        "seed": seed,
        "dataset": preset.name,
        "phase": SCIENCE_PHASE,
        "interpretation_status": interpretation_status,
        "rows": rows,
        "q_low": context.q_low,
        "q_high": context.q_high,
        "n_actions": context.n_actions,
        "action_mapping": {
            str(key): value for key, value in context.action_mapping.items()
        },
        "split_audit": v2._split_audit(context.splits),
        "base_context_identity": v2._context_identity(base_context),
        "kernel_context_identity": kernel_identity,
        "theta": theta.to_dict(),
        "confirmation_anchor_identity_sha256": clinical._json_sha256(
            anchor.kernel_identity
        ),
    }


def _assert_confirmation_context(
    anchor: ConfirmationAnchor,
    base_context: v2.ExtensionContext,
    kernel_identity: Mapping[str, Any],
) -> None:
    if (
        v2._split_audit(base_context.splits) != anchor.split_audit
        or kernel_identity != anchor.kernel_identity
    ):
        raise RuntimeError(
            "reconstructed frozen-kernel context differs from confirmation"
        )


def _run_phase(
    root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    theta: KernelTheta,
    anchors: Mapping[int, ConfirmationAnchor],
    seed_to_device: Mapping[int, str],
    devices: tuple[str, ...],
    source_hash: str,
    gate_contract_sha256: str,
    rng_mapping_sha256: str,
    worker: Callable[..., dict[str, Any]],
    worker_arguments: tuple[object, ...],
    resume: bool,
) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    expected_paths = {root / f"seed_{seed:06d}.json" for seed in preset.seeds}
    allowed_paths = {*expected_paths, root / "COMPLETE"}
    if phase == OVERLAP_PHASE:
        allowed_paths.add(root / "summary.json")
    observed_paths = {path for path in root.iterdir() if path.is_file()}
    unexpected = observed_paths - allowed_paths
    if unexpected:
        raise RuntimeError(f"unexpected {phase} seed artifacts: {sorted(unexpected)}")
    if not resume and observed_paths:
        raise RuntimeError(f"fresh {phase} phase already contains artifacts")
    completed = {}
    for seed in preset.seeds:
        path = root / f"seed_{seed:06d}.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        _validate_phase_payload(
            payload,
            phase=phase,
            preset=preset,
            seed=seed,
            device=seed_to_device[seed],
            theta=theta,
            anchor=anchors[seed],
            source_hash=source_hash,
            gate_contract_sha256=gate_contract_sha256,
            rng_mapping_sha256=rng_mapping_sha256,
        )
        completed[seed] = payload["result"]
    pending = tuple(seed for seed in preset.seeds if seed not in completed)
    if pending and (
        (root / "COMPLETE").exists() or (root / "summary.json").exists()
    ):
        raise RuntimeError(f"{phase} completion artifact exists with missing seeds")
    if pending:
        groups = tuple(
            tuple(seed for seed in pending if seed_to_device[seed] == device)
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
                    theta,
                    anchors,
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
                        "gate_contract_sha256": gate_contract_sha256,
                        "rng_stream_mapping_sha256": rng_mapping_sha256,
                        "theta_sha256": clinical._json_sha256(theta.to_dict()),
                        "result": result,
                    }
                    _validate_phase_payload(
                        payload,
                        phase=phase,
                        preset=preset,
                        seed=seed,
                        device=device,
                        theta=theta,
                        anchor=anchors[seed],
                        source_hash=source_hash,
                        gate_contract_sha256=gate_contract_sha256,
                        rng_mapping_sha256=rng_mapping_sha256,
                    )
                    _write_json(root / f"seed_{seed:06d}.json", payload)
                    completed[seed] = result
    if set(completed) != set(preset.seeds):
        raise RuntimeError(f"{phase} did not complete all confirmation seeds")
    if (root / "COMPLETE").exists() and (root / "COMPLETE").read_text() != "complete\n":
        raise RuntimeError(f"{phase} COMPLETE marker differs")
    if phase == OVERLAP_PHASE and (root / "summary.json").exists():
        interpretation = (
            "EMPIRICAL_OVERLAP_SCREEN_PASSED"
            if all(bool(completed[seed]["passed"]) for seed in preset.seeds)
            else "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        )
        expected_summary = _overlap_summary(
            preset.name,
            [completed[seed] for seed in preset.seeds],
            interpretation,
        )
        if _read_json(root / "summary.json") != expected_summary:
            raise RuntimeError(f"{phase} summary differs on resume")
    _write_text(root / "COMPLETE", "complete\n")
    return [completed[seed] for seed in preset.seeds]


def _phase_group(
    seeds: tuple[int, ...],
    device: str,
    preset: DatasetPreset,
    theta: KernelTheta,
    anchors: Mapping[int, ConfirmationAnchor],
    worker: Callable[..., dict[str, Any]],
    worker_arguments: tuple[object, ...],
) -> list[tuple[int, str, dict[str, Any]]]:
    torch.cuda.set_device(torch.device(device))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    rows = []
    for seed in seeds:
        result = worker(
            seed,
            preset,
            device,
            theta,
            anchors[seed],
            *worker_arguments,
        )
        rows.append((seed, device, result))
        torch.cuda.empty_cache()
    return rows


def _validate_phase_payload(
    payload: Mapping[str, Any],
    *,
    phase: str,
    preset: DatasetPreset,
    seed: int,
    device: str,
    theta: KernelTheta,
    anchor: ConfirmationAnchor,
    source_hash: str,
    gate_contract_sha256: str,
    rng_mapping_sha256: str,
) -> None:
    expected = {
        "protocol": PROTOCOL,
        "phase": phase,
        "dataset": preset.name,
        "seed": seed,
        "device": device,
        "source_tree_sha256": source_hash,
        "gate_contract_sha256": gate_contract_sha256,
        "rng_stream_mapping_sha256": rng_mapping_sha256,
        "theta_sha256": clinical._json_sha256(theta.to_dict()),
    }
    _require_exact_keys(payload, {*expected, "result"}, f"{phase} seed wrapper")
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{phase} seed provenance differs for {seed}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"{phase} seed result is malformed")
    if (
        result.get("seed") != seed
        or result.get("dataset") != preset.name
        or result.get("phase") != phase
        or result.get("theta") != theta.to_dict()
        or result.get("kernel_context_identity") != anchor.kernel_identity
        or result.get("split_audit") != anchor.split_audit
        or result.get("confirmation_anchor_identity_sha256")
        != clinical._json_sha256(anchor.kernel_identity)
    ):
        raise RuntimeError(f"{phase} seed identity differs for {seed}")
    if phase == OVERLAP_PHASE:
        _validate_overlap_result(result, preset)
    elif phase == SCIENCE_PHASE:
        _validate_science_result(result, preset)
    else:
        raise RuntimeError(f"unknown post-confirmation phase: {phase}")


def _validate_overlap_result(result: Mapping[str, Any], preset: DatasetPreset) -> None:
    _require_exact_keys(
        result,
        {
            "seed",
            "dataset",
            "phase",
            "passed",
            "interpretation_if_failed",
            "metrics",
            "diagnostics",
            "q_low",
            "q_high",
            "q_mid",
            "n_actions",
            "action_mapping",
            "split_audit",
            "base_context_identity",
            "kernel_context_identity",
            "theta",
            "confirmation_anchor_identity_sha256",
        },
        "overlap result",
    )
    _reject_overlap_science_keys(result)
    _validate_overlap_nested_schemas(result)
    v2_view = dict(result)
    v2_view["context_identity"] = v2_view.pop("base_context_identity")
    for key in (
        "kernel_context_identity",
        "theta",
        "confirmation_anchor_identity_sha256",
    ):
        v2_view.pop(key)
    if not v2._valid_overlap_result(v2_view):
        raise RuntimeError(f"{preset.name} overlap result violates frozen v2 semantics")


def _validate_overlap_nested_schemas(result: Mapping[str, Any]) -> None:
    _require_exact_keys(
        result["metrics"],
        {"local_ess_p01", "median_ess_fraction", "maximum_donor_probability"},
        "overlap metrics",
    )
    diagnostics = result["diagnostics"]
    _require_exact_keys(
        diagnostics,
        {
            "probe_trajectories",
            "gamma",
            "noise_seed",
            "common_random_numbers_across_radii",
            "independent_frozen_stream",
            "patient_aggregated",
            "episode_weighted_transition_patient_aggregated_diagnostics",
            "probes",
            "worst_metrics",
            "screen_status",
            "screen_scope",
            "environment_episode_support",
        },
        "overlap diagnostics",
    )
    _require_exact_keys(
        diagnostics["probes"],
        {"q_mid", "q_high"},
        "overlap probes",
    )
    probe_keys = {
        "radius_fraction",
        "radius",
        "metrics",
        "passed",
        "target_simplex_maximum_error",
        "logging_simplex_maximum_error",
        "minimum_logging_probability",
        "minimum_target_probability",
        "policy_probabilities_finite",
        "maximum_single_step_target_to_logging_ratio",
        "single_step_ratio_cap",
        "local_unique_k_minimum",
        "local_unique_k_median",
        "prefix_overlap_report_only",
    }
    for probe in diagnostics["probes"].values():
        _require_exact_keys(probe, probe_keys, "overlap probe")
        _require_exact_keys(
            probe["metrics"],
            {"local_ess_p01", "median_ess_fraction", "maximum_donor_probability"},
            "overlap probe metrics",
        )
        _require_exact_keys(
            probe["prefix_overlap_report_only"],
            {
                "minimum_ess_fraction",
                "maximum_normalized_weight_share",
                "maximum_raw_log_weight_span",
                "gate_role",
            },
            "overlap prefix report",
        )
        prefix = probe["prefix_overlap_report_only"]
        prefix_values = (
            float(prefix["minimum_ess_fraction"]),
            float(prefix["maximum_normalized_weight_share"]),
            float(prefix["maximum_raw_log_weight_span"]),
        )
        if (
            not math.isfinite(float(probe["local_unique_k_median"]))
            or not all(math.isfinite(value) for value in prefix_values)
            or probe["local_unique_k_median"] < probe["local_unique_k_minimum"]
            or prefix_values[0] <= 0.0
            or prefix_values[0] > 1.0
            or prefix_values[1] <= 0.0
            or prefix_values[1] > 1.0
            or prefix_values[2] < 0.0
            or prefix["gate_role"] != "report-only"
        ):
            raise RuntimeError("overlap report-only prefix diagnostics differ")
    if diagnostics["episode_weighted_transition_patient_aggregated_diagnostics"] is not True:
        raise RuntimeError("overlap patient-aggregation declaration differs")


def _reject_overlap_science_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            tokens = set(normalized.replace("-", "_").split("_"))
            if (
                "coverage" in normalized
                or "width" in tokens
                or "method" in normalized
                or "science" in normalized
            ):
                raise RuntimeError(f"overlap gate contains forbidden science key: {key}")
            _reject_overlap_science_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_overlap_science_keys(child)


def _validate_science_result(result: Mapping[str, Any], preset: DatasetPreset) -> None:
    _require_exact_keys(
        result,
        {
            "seed",
            "dataset",
            "phase",
            "interpretation_status",
            "rows",
            "q_low",
            "q_high",
            "n_actions",
            "action_mapping",
            "split_audit",
            "base_context_identity",
            "kernel_context_identity",
            "theta",
            "confirmation_anchor_identity_sha256",
        },
        "science result",
    )
    v2_view = dict(result)
    v2_view["context_identity"] = v2_view.pop("base_context_identity")
    for key in (
        "kernel_context_identity",
        "theta",
        "confirmation_anchor_identity_sha256",
    ):
        v2_view.pop(key)
    if not v2._valid_science_result(v2_view, preset):
        raise RuntimeError(f"{preset.name} science result violates frozen v2 semantics")
    _validate_science_rows(result["rows"], preset)


def _validate_science_rows(rows: Sequence[Mapping[str, Any]], preset: DatasetPreset) -> None:
    if len(rows) != len(GAMMAS):
        raise RuntimeError("science result requires every signed gamma")
    row_keys = {
        "seed",
        "dataset",
        "gamma",
        "q_low",
        "q_high",
        "adaptation_seeds",
        "scpcp_minimum_ess_fraction",
        "scpcp_minimum_candidate_ess_fraction",
        "scpcp_selected_endpoint",
        "scpcp_failure_stage",
        "methods",
    }
    for row, gamma in zip(rows, GAMMAS, strict=True):
        _require_exact_keys(row, row_keys, "signed-gamma row")
        if float(row["gamma"]) != gamma or set(row["methods"]) != set(METHODS):
            raise RuntimeError("signed-gamma or method order differs")
        fractions = (
            row["scpcp_minimum_ess_fraction"],
            row["scpcp_minimum_candidate_ess_fraction"],
        )
        if (
            any(
                value is not None
                and (
                    not math.isfinite(float(value))
                    or not 0.0 < float(value) <= 1.0
                )
                for value in fractions
            )
            or not isinstance(row["scpcp_selected_endpoint"], bool)
            or (
                row["scpcp_failure_stage"] is not None
                and (
                    not isinstance(row["scpcp_failure_stage"], int)
                    or not 0 <= row["scpcp_failure_stage"] < preset.horizon
                )
            )
        ):
            raise RuntimeError("SC-PCP signed-gamma diagnostics differ")
        for method in METHODS:
            _validate_method_row(method, row["methods"][method], preset.horizon)


def _validate_method_row(method: str, row: Mapping[str, Any], horizon: int) -> None:
    common = {
        "selection_available",
        "selection_status",
        "information_regime",
        "target_adaptation_trajectories",
    }
    adaptation = {
        "adaptation_rounds",
        "adaptation_per_time_coverage",
        "adaptation_round_worst_coverage",
        "adaptation_pathwise_coverage",
        "selected_scale",
    }
    vectors = {
        "radii",
        "source_coverage",
        "target_coverage",
        "coverage_gap",
        "source_q90",
        "target_q90",
        "q90_relative_gap",
        "target_normalized_width",
        "prefix_ess_fraction",
        "maximum_normalized_weight_share",
        "raw_log_weight_span",
        "policy_tv_on_source_states",
        "source_difficulty",
        "target_difficulty",
    }
    scalars = {"donor_kernel_ess_fraction_min", "donor_probability_max"}
    expected = set(common)
    if method in {"ACI", "SPCI", "PRC"}:
        expected |= adaptation
    available = row.get("selection_available") is True
    expected |= vectors | scalars if available else {"radii"}
    _require_exact_keys(row, expected, f"{method} result")
    if not available:
        if method not in {"MFCS", "SC-PCP"} or row.get("radii") != []:
            raise RuntimeError(f"{method} unavailable result differs")
        return
    for name in vectors:
        values = row[name]
        if (
            not isinstance(values, list)
            or len(values) != horizon
            or not all(math.isfinite(float(value)) for value in values)
        ):
            raise RuntimeError(f"{method} {name} vector differs")
    if not all(
        math.isfinite(float(row[name])) for name in scalars
    ):
        raise RuntimeError(f"{method} scalar diagnostics differ")
    if (
        not 0.0 < float(row["donor_kernel_ess_fraction_min"]) <= 1.0
        or not 0.0 <= float(row["donor_probability_max"]) <= 1.0
    ):
        raise RuntimeError(f"{method} donor diagnostics differ")
    if method in {"ACI", "SPCI", "PRC"}:
        per_time = row["adaptation_per_time_coverage"]
        per_round = row["adaptation_round_worst_coverage"]
        pathwise = row["adaptation_pathwise_coverage"]
        selected_scale = row["selected_scale"]
        if (
            row["adaptation_rounds"] != 3
            or len(per_time) != horizon
            or len(per_round) != 3
            or not all(0.0 <= float(value) <= 1.0 for value in per_time)
            or not all(0.0 <= float(value) <= 1.0 for value in per_round)
            or not math.isfinite(float(pathwise))
            or not 0.0 <= float(pathwise) <= 1.0
            or (
                selected_scale is not None
                and (
                    not math.isfinite(float(selected_scale))
                    or float(selected_scale) < 0.0
                )
            )
        ):
            raise RuntimeError(f"{method} target-adaptation trace differs")


def _overlap_summary(
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
    interpretation: str,
) -> dict[str, Any]:
    passed = [int(row["seed"]) for row in rows if bool(row["passed"])]
    return {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "gate": "gamma=-4 q_mid+q_high empirical donor-overlap screen",
        "prespecified_seed_count": 20,
        "passed_seed_count": len(passed),
        "passed_seeds": passed,
        "interpretation_status": interpretation,
        "hard_structural_failure": False,
        "failure_consequence": SCIENCE_CONTRACT["low_overlap_consequence"],
    }


def _dataset_science_status(dataset: str, interpretation: str) -> dict[str, Any]:
    if interpretation not in {
        "EMPIRICAL_OVERLAP_SCREEN_PASSED",
        "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
    }:
        raise RuntimeError(f"unknown overlap interpretation for {dataset}")
    return {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "status": "COMPLETE",
        "interpretation_status": interpretation,
        "prespecified_seed_count": 20,
        "raw_signed_gamma_rows": 20 * len(GAMMAS),
        "methods": list(METHODS),
        "primary_default_gamma": PRIMARY_GAMMA,
        "primary_metric": PRIMARY_METRIC,
        "ranking_permitted": interpretation == "EMPIRICAL_OVERLAP_SCREEN_PASSED",
    }


def _final_science_status(
    dataset_status: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(dataset_status) != tuple(DATASETS):
        raise RuntimeError("final science status requires all datasets in frozen order")
    return {
        "protocol": PROTOCOL,
        "status": "COMPLETE",
        "datasets": dict(dataset_status),
        "methods": list(METHODS),
        "gammas": list(GAMMAS),
        "primary_default_gamma": PRIMARY_GAMMA,
        "primary_metric": PRIMARY_METRIC,
        "complete_confirmation_seeds_per_dataset": 20,
        "bootstrap_resamples": 10_000,
        "low_overlap_datasets": [
            dataset
            for dataset in DATASETS
            if dataset_status[dataset]["interpretation_status"]
            == "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        ],
        "universal_ranking_defined": False,
    }


def _write_global_overlap_marker(
    root: Path,
    interpretations: Mapping[str, str],
) -> None:
    summary = {
        "protocol": PROTOCOL,
        "status": "GLOBAL_OVERLAP_COMPLETE",
        "datasets": dict(interpretations),
        "all_four_dataset_seed_banks_complete": True,
        "science_may_start": True,
        "low_overlap_consequence": SCIENCE_CONTRACT["low_overlap_consequence"],
    }
    _write_json(root / OVERLAP_PHASE / "summary.json", summary)
    _write_text(
        root / OVERLAP_PHASE / "COMPLETE",
        f"global-overlap-complete summary_sha256={clinical._json_sha256(summary)}\n",
    )


def _valid_global_overlap_marker(root: Path) -> bool:
    summary_path = root / OVERLAP_PHASE / "summary.json"
    complete_path = root / OVERLAP_PHASE / "COMPLETE"
    if not summary_path.is_file() or not complete_path.is_file():
        return False
    summary = _read_json(summary_path)
    datasets = summary.get("datasets")
    return (
        set(summary)
        == {
            "protocol",
            "status",
            "datasets",
            "all_four_dataset_seed_banks_complete",
            "science_may_start",
            "low_overlap_consequence",
        }
        and summary.get("protocol") == PROTOCOL
        and summary.get("status") == "GLOBAL_OVERLAP_COMPLETE"
        and isinstance(datasets, dict)
        and set(datasets) == set(DATASETS)
        and all(
            value
            in {
                "EMPIRICAL_OVERLAP_SCREEN_PASSED",
                "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
            }
            for value in datasets.values()
        )
        and summary.get("all_four_dataset_seed_banks_complete") is True
        and summary.get("science_may_start") is True
        and summary.get("low_overlap_consequence")
        == SCIENCE_CONTRACT["low_overlap_consequence"]
        and complete_path.read_text()
        == f"global-overlap-complete summary_sha256={clinical._json_sha256(summary)}\n"
    )


def _ensure_bootstrap_artifacts(
    root: Path,
    preset: DatasetPreset,
    *,
    create_if_missing: bool = True,
) -> dict[str, Any]:
    uniform_path = root / "bootstrap_uniforms.npy"
    index_path = root / "bootstrap_indices.npy"
    if not uniform_path.exists() and not index_path.exists():
        if not create_if_missing:
            raise RuntimeError("bootstrap artifacts are missing from a completed root")
        return v2._write_bootstrap_artifacts(
            root,
            preset=preset,
            resamples=10_000,
        )
    if not uniform_path.is_file() or not index_path.is_file():
        raise RuntimeError("bootstrap artifact pair is incomplete")
    rng = np.random.default_rng(preset.bootstrap_seed)
    expected_uniforms = rng.random((10_000, 20), dtype=np.float64)
    expected_indices = np.floor(expected_uniforms * 20).astype(np.int16)
    uniforms = np.load(uniform_path, allow_pickle=False)
    indices = np.load(index_path, allow_pickle=False)
    if not np.array_equal(uniforms, expected_uniforms) or not np.array_equal(
        indices,
        expected_indices,
    ):
        raise RuntimeError("bootstrap arrays differ from the frozen complete bank")
    return {
        "resamples": 10_000,
        "root_seed": preset.bootstrap_seed,
        "prespecified_seed_count": 20,
        "uniform_matrix_shape": [10_000, 20],
        "uniform_matrix_path": uniform_path.name,
        "uniform_matrix_sha256": _file_sha256(uniform_path),
        "complete_seed_index_matrix_shape": [10_000, 20],
        "complete_seed_index_matrix_path": index_path.name,
        "complete_seed_index_matrix_sha256": _file_sha256(index_path),
        "shared_across": ["methods", "gammas", "stages"],
        "selected_subset_rule": (
            "for selected-set size n, use floor(U[:, :n] * n); the complete "
            "10,000x20 matrix is floor(U*20)"
        ),
    }


def _science_summary(
    rows: list[dict[str, Any]],
    *,
    preset: DatasetPreset,
    interpretation_status: str,
    bootstrap_contract: dict[str, Any],
) -> dict[str, Any]:
    summary = v2.summarize_science(
        rows,
        preset=preset,
        selected_seeds=preset.seeds,
        interpretation_status=interpretation_status,
        bootstrap_contract=bootstrap_contract,
    )
    summary.update(
        {
            "protocol": PROTOCOL,
            "source_summary_semantics": v2.PROTOCOL,
            "role": "post_confirmation_frozen_kernel_signed_gamma_science",
            "seeds_k0_eligible": list(preset.seeds),
            "complete_confirmation_seed_bank_used": True,
            "primary_default_gamma": PRIMARY_GAMMA,
        }
    )
    return summary


def _coverage_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    preset: DatasetPreset,
    summary: Mapping[str, Any],
    interpretation_status: str,
) -> dict[str, Any]:
    if interpretation_status not in {
        "EMPIRICAL_OVERLAP_SCREEN_PASSED",
        "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
    }:
        raise RuntimeError("coverage audit received an unknown overlap interpretation")
    if len(rows) != 20 * len(GAMMAS):
        raise RuntimeError("coverage audit requires all complete-seed signed rows")
    aggregates = {float(row["gamma"]): row for row in summary["aggregates"]}
    if set(aggregates) != set(GAMMAS) or len(summary["aggregates"]) != len(GAMMAS):
        raise RuntimeError("coverage audit requires each signed gamma exactly once")
    records = []
    for gamma in GAMMAS:
        selected = sorted(
            (row for row in rows if float(row["gamma"]) == gamma),
            key=lambda row: int(row["seed"]),
        )
        if tuple(int(row["seed"]) for row in selected) != preset.seeds:
            raise RuntimeError(f"coverage audit seed mismatch for gamma={gamma}")
        for method in METHODS:
            available = [
                row for row in selected if row["methods"][method]["selection_available"]
            ]
            stage_coverage = (
                np.asarray(
                    [row["methods"][method]["target_coverage"] for row in available],
                    dtype=np.float64,
                ).mean(axis=0)
                if available
                else np.asarray([], dtype=np.float64)
            )
            reported = aggregates[gamma]["methods"][method][
                "target_marginal_worst_coverage"
            ]
            computed = None if not available else float(stage_coverage.min())
            if reported != computed:
                raise RuntimeError(f"{preset.name}/{gamma}/{method} WSC differs")
            records.append(
                {
                    "gamma": gamma,
                    "method": method,
                    "raw_seed_rows": 20,
                    "available_seed_rows": len(available),
                    "stage_count": preset.horizon,
                    "computed_wsc": computed,
                    "reported_wsc": reported,
                }
            )
    if interpretation_status == "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY":
        for aggregate in summary["aggregates"]:
            if (
                aggregate["paired_scpcp_comparisons"].get("status")
                != "EXCLUDED_LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
                or aggregate["width_order_among_point_eligible"] != []
                or aggregate["analysis_role"] != "descriptive_signed_control_curve"
            ):
                raise RuntimeError("low-overlap summary contains confirmatory ranking")
    else:
        for gamma, aggregate in aggregates.items():
            if gamma == PRIMARY_GAMMA:
                if aggregate["analysis_role"] != "confirmatory_gamma_minus_4_endpoint":
                    raise RuntimeError("primary gamma lacks its confirmatory role")
            elif (
                aggregate["analysis_role"] != "descriptive_signed_control_curve"
                or aggregate["paired_scpcp_comparisons"].get("status")
                != "EXCLUDED_NON_CONFIRMATORY_GAMMA_SIGNED_CONTROL"
                or aggregate["width_order_among_point_eligible"] != []
            ):
                raise RuntimeError("non-primary gamma contains confirmatory ranking")
    return {
        "protocol": PROTOCOL,
        "dataset": preset.name,
        "status": "COVERAGE_AUDIT_COMPLETE",
        "primary_metric": PRIMARY_METRIC,
        "formula_verified": True,
        "all_six_methods_present": True,
        "all_five_signed_gammas_present": True,
        "all_20_confirmation_seeds_present": True,
        "bootstrap_resamples": 10_000,
        "interpretation_status": interpretation_status,
        "records": records,
    }


def _science_metadata(
    gates: GateBundle,
    *,
    devices: tuple[str, ...],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "role": "post_confirmation_frozen_kernel_signed_gamma_science",
        "output_root": str(OUTPUT_ROOT),
        "source_tree_sha256": gates.source_tree_sha256,
        "source_snapshot": source_snapshot,
        "devices": list(devices),
        "gate_contract": gates.contract,
        "gate_contract_sha256": clinical._json_sha256(gates.contract),
        "parent_v2_binding": gates.parent_binding,
        "parent_v2_binding_sha256": clinical._json_sha256(gates.parent_binding),
        "rng_audit": gates.rng_audit,
        "rng_stream_mapping_sha256": gates.rng_audit[
            "new_rng_stream_mapping_sha256"
        ],
        "seed_to_device": {
            dataset: {
                str(seed): gates.seed_to_device[dataset][seed]
                for seed in gates.fidelity_config.confirmation_seeds[dataset]
            }
            for dataset in DATASETS
        },
        "science_contract": SCIENCE_CONTRACT,
        "phase_order": ["all_dataset_overlap", "global_overlap_complete", "science"],
        "canonical_scpcp_mutation_permitted": False,
    }


def _active_source_snapshot() -> tuple[str, dict[str, Any]]:
    source_hash = experiment_tree_sha256()
    snapshot = _build_source_snapshot()
    if experiment_tree_sha256() != source_hash:
        raise RuntimeError("source/config changed while building science snapshot")
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
        raise RuntimeError("science source snapshot file set is invalid")
    files = []
    archive_stream = io.BytesIO()
    with tarfile.open(fileobj=archive_stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
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
    snapshot: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if metadata["rng_audit"]["status"] != "passed_before_launch":
        raise RuntimeError("science RNG audit did not pass")
    if resume:
        if not root.is_dir() or not (root / "metadata.json").is_file():
            raise FileNotFoundError("science resume requires existing metadata")
        if _read_json(root / "metadata.json") != metadata:
            raise RuntimeError("science resume metadata differs")
        _verify_source_snapshot(root, metadata["source_snapshot"])
        return
    if root.exists():
        raise FileExistsError(f"fresh science root already exists: {root}")
    root.mkdir(parents=True)
    _atomic_write(
        root / snapshot["contract"]["archive_path"],
        snapshot["archive_bytes"],
    )
    _atomic_write(
        root / snapshot["contract"]["manifest_path"],
        snapshot["manifest_bytes"],
    )
    _write_json(root / "metadata.json", metadata)
    _verify_source_snapshot(root, metadata["source_snapshot"])


def _finalize_root(
    root: Path,
    metadata: Mapping[str, Any],
    gates: GateBundle,
) -> None:
    if experiment_tree_sha256() != gates.source_tree_sha256:
        raise RuntimeError("source/config changed during clinical-v3 science")
    refreshed = verify_gate_bundle(devices=tuple(metadata["devices"]))
    if refreshed.contract != gates.contract:
        raise RuntimeError("development/confirmation gate binding changed during science")
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("science root metadata changed")
    _write_manifest(root)
    final = _read_json(root / "FINAL_STATUS.json")
    marker = (
        f"complete source_tree_sha256={gates.source_tree_sha256} "
        f"gate_contract_sha256={clinical._json_sha256(gates.contract)} "
        f"final_status_sha256={clinical._json_sha256(final)}\n"
    )
    _write_text(root / "COMPLETE", marker)
    _validate_complete_root(root, metadata, gates)


def _validate_complete_root(
    root: Path,
    metadata: Mapping[str, Any],
    gates: GateBundle,
) -> None:
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("science complete metadata differs")
    _verify_source_snapshot(root, metadata["source_snapshot"])
    _verify_manifest(root)
    if not _valid_global_overlap_marker(root):
        raise RuntimeError("science root lacks a valid global overlap marker")
    global_overlap = _read_json(root / OVERLAP_PHASE / "summary.json")
    final = _read_json(root / "FINAL_STATUS.json")
    expected_marker = (
        f"complete source_tree_sha256={gates.source_tree_sha256} "
        f"gate_contract_sha256={clinical._json_sha256(gates.contract)} "
        f"final_status_sha256={clinical._json_sha256(final)}\n"
    )
    if (root / "COMPLETE").read_text() != expected_marker:
        raise RuntimeError("science COMPLETE marker differs")
    expected_dataset_status: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        preset = _confirmation_preset(gates, dataset)
        overlap = _load_phase(
            root / OVERLAP_PHASE / dataset,
            phase=OVERLAP_PHASE,
            preset=preset,
            gates=gates,
        )
        interpretation = (
            "EMPIRICAL_OVERLAP_SCREEN_PASSED"
            if all(bool(row["passed"]) for row in overlap)
            else "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        )
        expected_overlap_summary = _overlap_summary(
            dataset,
            overlap,
            interpretation,
        )
        if (
            _read_json(root / OVERLAP_PHASE / dataset / "summary.json")
            != expected_overlap_summary
            or global_overlap["datasets"].get(dataset) != interpretation
        ):
            raise RuntimeError(f"{dataset} overlap summary differs")
        science = _load_phase(
            root / SCIENCE_PHASE / dataset / "seeds",
            phase=SCIENCE_PHASE,
            preset=preset,
            gates=gates,
        )
        if any(
            result["interpretation_status"] != interpretation
            for result in science
        ):
            raise RuntimeError(f"{dataset} science interpretation differs")
        raw_rows = [row for result in science for row in result["rows"]]
        bootstrap = _ensure_bootstrap_artifacts(
            root / SCIENCE_PHASE / dataset,
            preset,
            create_if_missing=False,
        )
        expected_summary = _science_summary(
            raw_rows,
            preset=preset,
            interpretation_status=interpretation,
            bootstrap_contract=bootstrap,
        )
        if _read_json(root / SCIENCE_PHASE / dataset / "summary.json") != expected_summary:
            raise RuntimeError(f"{dataset} science summary differs")
        expected_audit = _coverage_audit(
            raw_rows,
            preset=preset,
            summary=expected_summary,
            interpretation_status=interpretation,
        )
        if _read_json(root / SCIENCE_PHASE / dataset / "coverage_audit.json") != expected_audit:
            raise RuntimeError(f"{dataset} coverage audit differs")
        expected_status = _dataset_science_status(dataset, interpretation)
        if (
            _read_json(root / SCIENCE_PHASE / dataset / "FINAL_STATUS.json")
            != expected_status
        ):
            raise RuntimeError(f"{dataset} final status differs")
        expected_dataset_marker = (
            "curves\n"
            if interpretation == "EMPIRICAL_OVERLAP_SCREEN_PASSED"
            else "curves-descriptive-only\n"
        )
        if (root / SCIENCE_PHASE / dataset / "COMPLETE").read_text() != expected_dataset_marker:
            raise RuntimeError(f"{dataset} science COMPLETE marker differs")
        expected_dataset_status[dataset] = expected_status
    if final != _final_science_status(expected_dataset_status):
        raise RuntimeError("science final status differs")
    _require_complete_artifact_set(root, metadata, gates)


def _require_complete_artifact_set(
    root: Path,
    metadata: Mapping[str, Any],
    gates: GateBundle,
) -> None:
    expected = _expected_complete_artifact_paths(metadata, gates)
    paths = [path for path in root.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("completed science root contains a symbolic link")
    observed = {path.relative_to(root).as_posix() for path in paths}
    if observed != expected:
        raise RuntimeError("completed science root artifact set differs")


def _require_partial_artifact_subset(
    root: Path,
    metadata: Mapping[str, Any],
    gates: GateBundle,
) -> None:
    allowed = _expected_complete_artifact_paths(metadata, gates)
    if any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("science root contains a symbolic link")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if not observed <= allowed:
        raise RuntimeError("partial science root contains an unexpected artifact")


def _expected_complete_artifact_paths(
    metadata: Mapping[str, Any],
    gates: GateBundle,
) -> set[str]:
    expected = {
        "metadata.json",
        "FINAL_STATUS.json",
        "manifest.json",
        "COMPLETE",
        metadata["source_snapshot"]["archive_path"],
        metadata["source_snapshot"]["manifest_path"],
        f"{OVERLAP_PHASE}/summary.json",
        f"{OVERLAP_PHASE}/COMPLETE",
    }
    for dataset in DATASETS:
        preset = _confirmation_preset(gates, dataset)
        expected.update(
            {
                f"{OVERLAP_PHASE}/{dataset}/summary.json",
                f"{OVERLAP_PHASE}/{dataset}/COMPLETE",
                f"{SCIENCE_PHASE}/{dataset}/bootstrap_uniforms.npy",
                f"{SCIENCE_PHASE}/{dataset}/bootstrap_indices.npy",
                f"{SCIENCE_PHASE}/{dataset}/summary.json",
                f"{SCIENCE_PHASE}/{dataset}/coverage_audit.json",
                f"{SCIENCE_PHASE}/{dataset}/FINAL_STATUS.json",
                f"{SCIENCE_PHASE}/{dataset}/COMPLETE",
                f"{SCIENCE_PHASE}/{dataset}/seeds/COMPLETE",
            }
        )
        expected.update(
            f"{OVERLAP_PHASE}/{dataset}/seed_{seed:06d}.json"
            for seed in preset.seeds
        )
        expected.update(
            f"{SCIENCE_PHASE}/{dataset}/seeds/seed_{seed:06d}.json"
            for seed in preset.seeds
        )
    return expected


def _load_phase(
    root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    gates: GateBundle,
) -> list[dict[str, Any]]:
    extras = {root / "summary.json"} if phase == OVERLAP_PHASE else set()
    _require_phase_seed_files(root, preset.seeds, extras=extras)
    rows = []
    for seed in preset.seeds:
        payload = _read_json(root / f"seed_{seed:06d}.json")
        _validate_phase_payload(
            payload,
            phase=phase,
            preset=preset,
            seed=seed,
            device=gates.seed_to_device[preset.name][seed],
            theta=gates.frozen_theta[preset.name],
            anchor=gates.anchors[preset.name][seed],
            source_hash=gates.source_tree_sha256,
            gate_contract_sha256=clinical._json_sha256(gates.contract),
            rng_mapping_sha256=gates.rng_audit["new_rng_stream_mapping_sha256"],
        )
        rows.append(payload["result"])
    return rows


def _require_phase_seed_files(
    root: Path,
    seeds: Sequence[int],
    *,
    extras: set[Path] | None = None,
) -> None:
    expected = {
        *(root / f"seed_{seed:06d}.json" for seed in seeds),
        root / "COMPLETE",
        *(extras or set()),
    }
    observed = {path for path in root.iterdir() if path.is_file()}
    if observed != expected or (root / "COMPLETE").read_text() != "complete\n":
        raise RuntimeError(f"science phase artifact set differs: {root}")


def _root_binding(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "metadata_sha256": _file_sha256(root / "metadata.json"),
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "complete_sha256": _file_sha256(root / "COMPLETE"),
        "final_status_sha256": _file_sha256(root / "FINAL_STATUS.json"),
    }


def _write_manifest(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path in {root / "manifest.json", root / "COMPLETE"}:
            continue
        if ".tmp-" in path.name:
            raise RuntimeError(f"temporary science artifact remains: {path}")
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    _write_json(
        root / "manifest.json",
        {"protocol": PROTOCOL, "artifact_count": len(entries), "artifacts": entries},
    )


def _verify_manifest(root: Path) -> None:
    manifest = _read_json(root / "manifest.json")
    _require_exact_keys(
        manifest,
        {"protocol", "artifact_count", "artifacts"},
        "science manifest",
    )
    entries = manifest.get("artifacts")
    if (
        manifest.get("protocol") != PROTOCOL
        or not isinstance(entries, list)
        or not isinstance(manifest.get("artifact_count"), int)
    ):
        raise RuntimeError("science manifest header differs")
    expected = set()
    for entry in entries:
        _require_exact_keys(entry, {"path", "bytes", "sha256"}, "manifest entry")
        if not isinstance(entry["path"], str):
            raise RuntimeError("science manifest path is malformed")
        path = root / entry["path"]
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()) or resolved in expected:
            raise RuntimeError("science manifest path is unsafe or duplicated")
        expected.add(resolved)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != entry["bytes"]
            or _file_sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"science manifest mismatch: {path}")
    observed = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path not in {root / "manifest.json", root / "COMPLETE"}
    }
    if observed != expected or manifest.get("artifact_count") != len(entries):
        raise RuntimeError("science manifest file set differs")


def _verify_source_snapshot(root: Path, contract: Mapping[str, Any]) -> None:
    for name in ("archive", "manifest"):
        path = root / contract[f"{name}_path"]
        if (
            not path.is_file()
            or path.stat().st_size != contract[f"{name}_bytes"]
            or _file_sha256(path) != contract[f"{name}_sha256"]
        ):
            raise RuntimeError(f"science source snapshot {name} differs")


def _require_exact_keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError(f"{label} schema differs")


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


if __name__ == "__main__":
    main()
