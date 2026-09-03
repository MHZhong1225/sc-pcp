"""Run the frozen MIMIC-CXR environment-support science workflow.

First perform the read-only confirmation audit::

    python scripts/run_controlled_clinical_mimic_cxr_environment_support_science.py audit

After recording the reported contract hash, launch the two-GPU workflow::

    python scripts/run_controlled_clinical_mimic_cxr_environment_support_science.py run \
      --audit-go-sha256 <reported-hash>

The runner completes the full donor-overlap bank before it can publish a
science unlock or compute any method-level coverage result.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import tarfile
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_controlled_clinical_extension as v2  # noqa: E402
import scripts.run_controlled_clinical_fidelity_v4 as v4  # noqa: E402
import scripts.run_controlled_clinical_fidelity_v4_science as v4science  # noqa: E402
import scripts.run_controlled_clinical_fidelity_v5_mimic_cxr as v5  # noqa: E402
import scripts.run_controlled_clinical_mimic_cxr_environment_support as cxr  # noqa: E402
from scripts.run_controlled_six_method_benchmark import (  # noqa: E402
    TARGET_ADAPTATION_BUDGET,
    _wilson_interval,
)
from scpcp.artifacts import experiment_tree_sha256  # noqa: E402
from scpcp.controlled_clinical_extension import (  # noqa: E402
    GAMMAS,
    METHODS,
    ControlledClinicalExtensionConfig,
    DatasetPreset,
    donor_overlap_passes,
)
from scpcp.controlled_clinical_fidelity_v5_mimic_cxr import (  # noqa: E402
    BridgeTheta,
)
from scpcp.controlled_clinical_mimic_cxr_environment_support import (  # noqa: E402
    BRIDGE_CANDIDATE_ID,
    CONFIRMATION_SEEDS,
    DATASET,
    ROLE_SPLIT,
    EnvironmentSupportConfig,
    load_config,
    normalized_k0_ratio,
    summarize_confirmation,
)


PROTOCOL = "controlled_clinical_mimic_cxr_environment_support_science_v1"
CONFIG_PATH = cxr.CONFIG_PATH
CONFIRMATION_ROOT = cxr.CONFIRMATION_ROOT
OUTPUT_ROOT = cxr.SCIENCE_ROOT
PRIMARY_GAMMA = -4.0
PRIMARY_METRIC = "min_t mean_seed(C_seed,t)"
TARGET_COVERAGE = 0.90
PRESPECIFIED_SEED_COUNT = 20
MINIMUM_JOINT_PASS_COUNT = 19
BOOTSTRAP_RESAMPLES = 10_000
OVERLAP_PHASE = "donor_overlap"
SCIENCE_PHASE = "science"
_OWN_RNG_DECLARATION_PATHS = {
    *cxr._OWN_RNG_DECLARATION_PATHS,
    Path(__file__).resolve(),
    (
        ROOT / "tests/per_step/"
        "test_controlled_clinical_mimic_cxr_environment_support_science.py"
    ).resolve(),
}


@dataclass(frozen=True)
class ConfirmationAnchor:
    split_audit: Mapping[str, Any]
    base_context_identity: Mapping[str, Any]
    kernel_context_identity: Mapping[str, Any]
    support_passed: bool
    k0_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_audit": dict(self.split_audit),
            "base_context_identity": dict(self.base_context_identity),
            "kernel_context_identity": dict(self.kernel_context_identity),
            "support_passed": self.support_passed,
            "k0_passed": self.k0_passed,
        }


@dataclass(frozen=True)
class GateBundle:
    config: EnvironmentSupportConfig
    protocol: ControlledClinicalExtensionConfig
    preset: DatasetPreset
    theta: BridgeTheta
    prespecified_seeds: tuple[int, ...]
    support_k0_eligible_seeds: tuple[int, ...]
    anchors: Mapping[int, ConfirmationAnchor]
    eligibility_record: Mapping[str, Any]
    seed_to_device: Mapping[int, str]
    active_source_tree_sha256: str
    confirmation_binding: Mapping[str, Any]
    science_rng_audit: Mapping[str, Any]
    rng_stream_mapping_sha256: str
    science_contract: Mapping[str, Any]
    contract: Mapping[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "run"))
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--audit-go-sha256")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    gates = verify_gate_bundle(devices=devices)
    audit_hash = _json_sha256(gates.contract)
    if args.phase == "audit":
        if args.audit_go_sha256 is not None or args.resume:
            parser.error("audit does not accept --audit-go-sha256 or --resume")
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "status": "READ_ONLY_CONFIRMATION_AUDIT_GO",
                    "audit_contract_sha256": audit_hash,
                    "confirmation_status": "CONFIRMATION_GO",
                    "support_k0_eligible_seed_count": len(
                        gates.support_k0_eligible_seeds
                    ),
                    "coverage_generated": False,
                    "output_root_created": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.audit_go_sha256 != audit_hash:
        raise RuntimeError(
            "run requires the exact hash reported by the read-only audit"
        )
    _validate_cuda_devices(devices)
    run_post_confirmation_science(
        OUTPUT_ROOT,
        gates=gates,
        devices=devices,
        audit_go_sha256=audit_hash,
        resume=args.resume,
    )
    print(OUTPUT_ROOT)


def verify_gate_bundle(
    *,
    devices: tuple[str, ...],
    confirmation_root: Path = CONFIRMATION_ROOT,
) -> GateBundle:
    """Validate the complete frozen confirmation before any output mutation."""

    if devices != ("cuda:0", "cuda:1"):
        raise RuntimeError("science is frozen to devices cuda:0,cuda:1")
    confirmation_root = confirmation_root.resolve()
    config = load_config(CONFIG_PATH)
    protocol = cxr._protocol_for(CONFIRMATION_SEEDS, config)
    preset = protocol.datasets[DATASET]
    theta = cxr._b02()
    science_contract = _validate_frozen_science_contract(protocol, theta)

    cxr._verify_complete_root(confirmation_root)
    metadata = _read_json(confirmation_root / "metadata.json")
    final = _read_json(confirmation_root / "FINAL_STATUS.json")
    gate = _read_json(confirmation_root / "gate.json")
    active_source_hash = experiment_tree_sha256()
    _validate_confirmation_metadata(
        metadata,
        root=confirmation_root,
        config=config,
        theta=theta,
        active_source_hash=active_source_hash,
        devices=devices,
    )
    _validate_parent_source_snapshot(
        confirmation_root,
        metadata,
        active_source_hash=active_source_hash,
    )
    science_rng_audit = _validate_confirmation_rng(metadata, protocol, config)
    rng_mapping_hash = str(science_rng_audit["full_mapping_sha256"])

    seed_to_device = cxr._seed_device_mapping(CONFIRMATION_SEEDS, devices)
    _require_exact_seed_phase(confirmation_root / "support", CONFIRMATION_SEEDS)
    _require_exact_seed_phase(confirmation_root / "k0_fidelity", CONFIRMATION_SEEDS)
    support_rows: list[dict[str, Any]] = []
    k0_rows: list[dict[str, Any]] = []
    anchors: dict[int, ConfirmationAnchor] = {}
    eligibility_rows = []
    for seed in CONFIRMATION_SEEDS:
        support = _load_confirmation_result(
            confirmation_root / "support" / f"seed_{seed:06d}.json",
            phase="confirmation_support",
            seed=seed,
            device=seed_to_device[seed],
            source_hash=active_source_hash,
        )
        k0 = _load_confirmation_result(
            confirmation_root / "k0_fidelity" / f"seed_{seed:06d}.json",
            phase="confirmation_k0",
            seed=seed,
            device=seed_to_device[seed],
            source_hash=active_source_hash,
        )
        _validate_support_result(support, preset=preset, seed=seed)
        _validate_k0_result(k0, theta=theta, seed=seed)
        if support["split_audit"] != k0["split_audit"]:
            raise RuntimeError(f"confirmation split identity differs for seed {seed}")
        support_passed = bool(support["passed"])
        k0_passed = bool(k0["passed"])
        eligible = support_passed and k0_passed
        exclusion_reason = None
        if not support_passed:
            exclusion_reason = {
                "code": "SUPPORT_FAILED",
                "failed_cells": support["failed_cells"],
            }
        elif not k0_passed:
            exclusion_reason = {"code": "K0_FAILED", "metrics": k0["metrics"]}
        eligibility_rows.append(
            {
                "seed": seed,
                "support_passed": support_passed,
                "k0_passed": k0_passed,
                "support_k0_eligible": eligible,
                "exclusion_reason": exclusion_reason,
            }
        )
        support_rows.append(support)
        k0_rows.append(k0)
        if eligible:
            anchors[seed] = ConfirmationAnchor(
                split_audit=k0["split_audit"],
                base_context_identity=k0["base_context_identity"],
                kernel_context_identity=k0["kernel_context_identity"],
                support_passed=True,
                k0_passed=True,
            )

    recomputed_gate = summarize_confirmation(support_rows, k0_rows)
    if gate != recomputed_gate or gate.get("status") != "CONFIRMATION_GO":
        raise RuntimeError("frozen confirmation gate is not an exact CONFIRMATION_GO")
    expected_final = {
        "protocol": cxr.PROTOCOL,
        "dataset": DATASET,
        "phase": "confirmation",
        "status": "CONFIRMATION_GO",
        "confirmation_admissible": True,
        "eligible_seeds": gate["joint_pass_seeds"],
        "coverage_generated": False,
        "science_may_start": True,
    }
    if final != expected_final:
        raise RuntimeError("frozen confirmation FINAL_STATUS differs")

    eligible_seeds = tuple(int(seed) for seed in gate["joint_pass_seeds"])
    if (
        len(eligible_seeds) < MINIMUM_JOINT_PASS_COUNT
        or tuple(seed for seed in CONFIRMATION_SEEDS if seed in anchors)
        != eligible_seeds
    ):
        raise RuntimeError("confirmation eligible seed identity differs")
    eligibility_record = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "rule": "support_passed AND k0_passed",
        "prespecified_seed_count": PRESPECIFIED_SEED_COUNT,
        "selection_rate_denominator": PRESPECIFIED_SEED_COUNT,
        "support_k0_eligible_seed_count": len(eligible_seeds),
        "support_k0_eligible_seeds": list(eligible_seeds),
        "unavailable_before_overlap": [
            row["seed"] for row in eligibility_rows if not row["support_k0_eligible"]
        ],
        "seed_records": eligibility_rows,
        "seed_deletions": 0,
    }
    confirmation_binding = _confirmation_binding(confirmation_root)
    data_contract = v2._dataset_contract(protocol, preset)
    contract = {
        "protocol": PROTOCOL,
        "active_source_tree_sha256": active_source_hash,
        "confirmation_binding": confirmation_binding,
        "confirmation_binding_sha256": _json_sha256(confirmation_binding),
        "confirmation_source_tree_sha256": metadata["source_tree_sha256"],
        "confirmation_source_snapshot": metadata["source_snapshot"],
        "config_sha256": metadata["config_sha256"],
        "science_rng_audit": science_rng_audit,
        "rng_stream_mapping_sha256": rng_mapping_hash,
        "devices": list(devices),
        "seed_to_device": {
            str(seed): seed_to_device[seed] for seed in CONFIRMATION_SEEDS
        },
        "role_split": list(ROLE_SPLIT),
        "bridge_candidate": theta.to_dict(),
        "prespecified_seeds": list(CONFIRMATION_SEEDS),
        "support_k0_eligible_seeds": list(eligible_seeds),
        "confirmation_anchor_sha256": {
            str(seed): _json_sha256(anchors[seed].to_dict()) for seed in eligible_seeds
        },
        "eligibility_record_sha256": _json_sha256(eligibility_record),
        "data_contract": data_contract,
        "science_contract": science_contract,
        "coverage_permitted_before_overlap_unlock": False,
    }
    return GateBundle(
        config=config,
        protocol=protocol,
        preset=preset,
        theta=theta,
        prespecified_seeds=CONFIRMATION_SEEDS,
        support_k0_eligible_seeds=eligible_seeds,
        anchors=anchors,
        eligibility_record=eligibility_record,
        seed_to_device={seed: seed_to_device[seed] for seed in eligible_seeds},
        active_source_tree_sha256=active_source_hash,
        confirmation_binding=confirmation_binding,
        science_rng_audit=science_rng_audit,
        rng_stream_mapping_sha256=rng_mapping_hash,
        science_contract=science_contract,
        contract=contract,
    )


def _validate_frozen_science_contract(
    protocol: ControlledClinicalExtensionConfig,
    theta: BridgeTheta,
) -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    expected_science = {
        "gammas": list(GAMMAS),
        "primary_gamma": PRIMARY_GAMMA,
        "methods": list(METHODS),
        "calibration_trajectories": 3_000,
        "grid_trajectories": 1_000,
        "target_adaptation_trajectories": dict(TARGET_ADAPTATION_BUDGET),
        "evaluation_trajectories": 20_000,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "primary_metric": PRIMARY_METRIC,
        "mean_coverage_is_supplementary": True,
    }
    expected_overlap = {
        "gamma": PRIMARY_GAMMA,
        "probe_radius_fractions": [0.5, 1.0],
        "probe_trajectories": 3_000,
        "local_ess_p01": 10.0,
        "median_ess_fraction": 0.25,
        "maximum_donor_probability": 0.25,
    }
    if payload.get("science") != expected_science:
        raise RuntimeError("science settings differ from the frozen YAML")
    if payload.get("donor_overlap_gate") != expected_overlap:
        raise RuntimeError("donor-overlap settings differ from the frozen YAML")
    if (
        tuple(protocol.gammas) != GAMMAS
        or tuple(METHODS) != ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")
        or (
            protocol.calibration_trajectories,
            protocol.grid_trajectories,
            protocol.reference_trajectories,
            protocol.online_trajectories,
            protocol.bootstrap_resamples,
        )
        != (3_000, 1_000, 20_000, 2_000, 10_000)
        or protocol.policy_ratio_cap != 3.0
        or protocol.split_fractions != ROLE_SPLIT
        or theta.candidate_id != BRIDGE_CANDIDATE_ID
    ):
        raise RuntimeError("runtime protocol differs from the frozen science contract")
    return {
        **expected_science,
        "target_coverage": TARGET_COVERAGE,
        "policy_ratio_cap": protocol.policy_ratio_cap,
        "overlap": expected_overlap,
        "bridge_candidate_id": BRIDGE_CANDIDATE_ID,
        "role_split": list(ROLE_SPLIT),
        "selection_rate_denominator": PRESPECIFIED_SEED_COUNT,
        "science_requires_joint_overlap_pass_count": MINIMUM_JOINT_PASS_COUNT,
    }


def _validate_confirmation_metadata(
    metadata: Mapping[str, Any],
    *,
    root: Path,
    config: EnvironmentSupportConfig,
    theta: BridgeTheta,
    active_source_hash: str,
    devices: tuple[str, ...],
) -> None:
    if (
        metadata.get("protocol") != cxr.PROTOCOL
        or metadata.get("dataset") != DATASET
        or metadata.get("phase") != "confirmation"
        or Path(str(metadata.get("output_root", ""))).resolve() != root
        or metadata.get("devices") != list(devices)
        or metadata.get("source_tree_sha256") != active_source_hash
        or metadata.get("config_path") != CONFIG_PATH.relative_to(ROOT).as_posix()
        or metadata.get("config_sha256") != _file_sha256(CONFIG_PATH)
        or metadata.get("role_split") != list(ROLE_SPLIT)
        or metadata.get("bridge_candidate") != theta.to_dict()
        or metadata.get("coverage_generation_permitted") is not False
        or metadata.get("canonical_scpcp_mutation_permitted") is not False
        or not isinstance(metadata.get("source_snapshot"), Mapping)
        or config.confirmation_seeds != CONFIRMATION_SEEDS
    ):
        raise RuntimeError("frozen confirmation metadata differs")


def _validate_parent_source_snapshot(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    active_source_hash: str,
) -> None:
    contract = metadata["source_snapshot"]
    v4._verify_source_snapshot(root, contract)
    manifest_path = _safe_child(root, Path(str(contract["manifest_path"])))
    archive_path = _safe_child(root, Path(str(contract["archive_path"])))
    manifest = _read_json(manifest_path)
    entries = manifest.get("files")
    if (
        manifest.get("format") != "deterministic_uncompressed_pax_tar"
        or not isinstance(entries, list)
        or manifest.get("file_count") != len(entries)
    ):
        raise RuntimeError("confirmation source manifest is malformed")
    paths = [entry.get("path") for entry in entries if isinstance(entry, Mapping)]
    if len(paths) != len(entries) or len(set(paths)) != len(paths):
        raise RuntimeError("confirmation source manifest paths differ")

    digest = hashlib.sha256()
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != paths or any(
            not member.isfile() for member in members
        ):
            raise RuntimeError("confirmation source archive file set differs")
        for entry, member in zip(entries, members, strict=True):
            relative = Path(str(entry["path"]))
            _safe_child(root, relative)
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError("confirmation source archive member is unreadable")
            content = stream.read()
            if len(content) != entry.get("bytes") or hashlib.sha256(
                content
            ).hexdigest() != entry.get("sha256"):
                raise RuntimeError("confirmation source archive content differs")
            name = relative.as_posix().encode("utf-8")
            digest.update(len(name).to_bytes(4, "big"))
            digest.update(name)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    if digest.hexdigest() != metadata.get("source_tree_sha256") or (
        active_source_hash != metadata.get("source_tree_sha256")
    ):
        raise RuntimeError("confirmation source-tree hash differs")


def _validate_confirmation_rng(
    metadata: Mapping[str, Any],
    protocol: ControlledClinicalExtensionConfig,
    config: EnvironmentSupportConfig,
) -> dict[str, Any]:
    precoverage_mapping = cxr._precoverage_rng_stream_mapping(
        CONFIRMATION_SEEDS, config
    )
    audit = metadata.get("rng_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("new_rng_stream_mapping") != precoverage_mapping
        or audit.get("new_rng_stream_mapping_sha256")
        != _json_sha256(precoverage_mapping)
    ):
        raise RuntimeError("confirmation precoverage RNG mapping differs")
    cxr._validate_rng_audit(audit, phase="confirmation")

    full_mapping = v2._new_rng_stream_mapping(protocol, (DATASET,))
    v2._assert_unique_rng_streams(full_mapping)
    full_hash = _json_sha256(full_mapping)
    frozen_audit = yaml.safe_load(CONFIG_PATH.read_text())[
        "prelaunch_integrity_amendment"
    ]["replacement_rng_audit"]
    if (
        len(full_mapping) != frozen_audit["full_confirmation_stream_count"]
        or full_hash != frozen_audit["full_confirmation_mapping_sha256"]
        or frozen_audit["full_confirmation_internal_collision_count"] != 0
        or frozen_audit["historical_collision_count"] != 0
    ):
        raise RuntimeError("full science RNG mapping differs from the amendment")

    excluded_roots = {cxr.DEVELOPMENT_ROOT, CONFIRMATION_ROOT, OUTPUT_ROOT}
    artifact_ids = cxr._artifact_rng_ids(
        ROOT / "results", excluded_roots=excluded_roots
    )
    source_ids = v2._source_declared_seeds(
        ROOT,
        excluded_paths={path for path in _OWN_RNG_DECLARATION_PATHS if path.exists()},
    )
    prior_ids = artifact_ids | source_ids
    collisions = {
        name: value for name, value in full_mapping.items() if value in prior_ids
    }
    if collisions:
        raise RuntimeError(f"full science RNG collision: {collisions}")
    return {
        "status": "passed_read_only_before_science",
        "full_stream_count": len(full_mapping),
        "full_mapping": full_mapping,
        "full_mapping_sha256": full_hash,
        "full_id_set_sha256": cxr._integer_set_sha256(full_mapping.values()),
        "internal_collision_count": 0,
        "historical_artifact_rng_id_count": len(artifact_ids),
        "historical_artifact_rng_id_sha256": cxr._integer_set_sha256(artifact_ids),
        "historical_source_rng_id_count": len(source_ids),
        "historical_source_rng_id_sha256": cxr._integer_set_sha256(source_ids),
        "historical_union_rng_id_count": len(prior_ids),
        "historical_union_rng_id_sha256": cxr._integer_set_sha256(prior_ids),
        "historical_collision_count": 0,
        "historical_collisions": {},
        "excluded_protocol_roots": sorted(str(path) for path in excluded_roots),
        "own_declaration_paths": sorted(
            str(path) for path in _OWN_RNG_DECLARATION_PATHS if path.exists()
        ),
        "confirmation_precoverage_mapping_sha256": _json_sha256(precoverage_mapping),
    }


def _require_exact_seed_phase(root: Path, seeds: Sequence[int]) -> None:
    expected = {f"seed_{seed:06d}.json" for seed in seeds} | {"COMPLETE"}
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"confirmation seed phase is missing: {root}")
    children = list(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise RuntimeError(f"confirmation seed phase contains an unsafe entry: {root}")
    observed = {path.name for path in children}
    if observed != expected or (root / "COMPLETE").read_text() != "complete\n":
        raise RuntimeError(f"confirmation seed phase file set differs: {root}")


def _load_confirmation_result(
    path: Path,
    *,
    phase: str,
    seed: int,
    device: str,
    source_hash: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    cxr._validate_seed_envelope(
        payload,
        phase=phase,
        seed=seed,
        device=device,
        source_hash=source_hash,
    )
    result = payload["result"]
    if not isinstance(result, dict):
        raise RuntimeError(f"confirmation result is malformed for seed {seed}")
    return result


def _validate_support_result(
    result: Mapping[str, Any],
    *,
    preset: DatasetPreset,
    seed: int,
) -> None:
    if (
        result.get("role_split") != list(ROLE_SPLIT)
        or result.get("coverage_generated") is not False
    ):
        raise RuntimeError(f"support role split differs for seed {seed}")
    v2_result = dict(result)
    v2_result.pop("role_split", None)
    v2_result.pop("coverage_generated", None)
    if not v2._valid_support_result(v2_result, preset):
        raise RuntimeError(f"support result violates the frozen gate for seed {seed}")
    _validate_split_audit(result.get("split_audit"))


def _validate_k0_result(
    result: Mapping[str, Any],
    *,
    theta: BridgeTheta,
    seed: int,
) -> None:
    required = {
        "seed",
        "dataset",
        "passed",
        "metrics",
        "normalized_k0_ratio",
        "theta",
        "role_split",
        "systematic_replay",
        "base_context_identity",
        "kernel_context_identity",
        "split_audit",
        "coverage_generated",
    }
    if set(result) != required:
        raise RuntimeError(f"K0 result schema differs for seed {seed}")
    ratio = normalized_k0_ratio(result["metrics"])
    finite_ratio = ratio if np.isfinite(ratio) else None
    candidate_view = {
        "metrics": result["metrics"],
        "systematic_replay": result["systematic_replay"],
        "context_identity": result["kernel_context_identity"],
        "passed": result["passed"],
        "normalized_seed_ratio": finite_ratio,
        "structural_failure_ratio_is_infinite": not np.isfinite(ratio),
    }
    v5._validate_k0_candidate_row(candidate_view)
    base = result["base_context_identity"]
    kernel = result["kernel_context_identity"]
    if (
        result["seed"] != seed
        or result["dataset"] != DATASET
        or result["theta"] != theta.to_dict()
        or result["role_split"] != list(ROLE_SPLIT)
        or result["normalized_k0_ratio"] != finite_ratio
        or result["coverage_generated"] is not False
        or not v2._valid_context_identity(base)
        or kernel.get("base_nuisance_context_sha256") != base.get("combined_sha256")
        or kernel.get("theta") != theta.to_dict()
    ):
        raise RuntimeError(f"K0 context identity differs for seed {seed}")
    _validate_split_audit(result["split_audit"])
    split_hashes = result["split_audit"]["role_patient_id_sha256"]
    if (
        base.get("split_fractions") != list(ROLE_SPLIT)
        or base.get("split_patient_id_sha256") != split_hashes
        or kernel.get("split_patient_id_sha256") != split_hashes
    ):
        raise RuntimeError(f"K0 split/context binding differs for seed {seed}")


def _validate_split_audit(value: object) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("patient_sets_pairwise_disjoint") is not True
        or value.get("split_fractions") != list(ROLE_SPLIT)
        or set(value.get("role_patient_id_sha256", {}))
        != {"predictor", "fidelity", "environment"}
    ):
        raise RuntimeError("confirmation split audit is not the frozen 20/20/60 split")


def _confirmation_binding(root: Path) -> dict[str, Any]:
    binding = {
        "root": str(root),
        "complete_sha256": _file_sha256(root / "COMPLETE"),
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "metadata_sha256": _file_sha256(root / "metadata.json"),
        "final_status_sha256": _file_sha256(root / "FINAL_STATUS.json"),
        "gate_sha256": _file_sha256(root / "gate.json"),
    }
    return {**binding, "combined_sha256": _json_sha256(binding)}


def run_post_confirmation_science(
    output_root: Path,
    *,
    gates: GateBundle,
    devices: tuple[str, ...],
    audit_go_sha256: str,
    resume: bool,
) -> None:
    if output_root.resolve() != OUTPUT_ROOT:
        raise RuntimeError(f"science output root is frozen to {OUTPUT_ROOT}")
    gate_hash = _json_sha256(gates.contract)
    if audit_go_sha256 != gate_hash:
        raise RuntimeError("audit GO does not match the active confirmation contract")
    source_hash, source_snapshot = _active_source_snapshot()
    if source_hash != gates.active_source_tree_sha256:
        raise RuntimeError("source changed after frozen confirmation")
    metadata = _science_metadata(
        gates,
        devices=devices,
        audit_go_sha256=audit_go_sha256,
        source_snapshot=source_snapshot["contract"],
    )
    _prepare_root(output_root, metadata, source_snapshot, resume=resume)
    if (output_root / "COMPLETE").exists():
        _validate_complete_root(output_root, metadata, gates)
        return
    _validate_partial_root(output_root, metadata, gates)
    _write_or_verify_json(output_root / "eligibility.json", gates.eligibility_record)
    if (output_root / SCIENCE_PHASE).exists() and not _valid_science_unlock(
        output_root, gates
    ):
        raise RuntimeError("science artifacts exist before a valid overlap unlock")

    overlap_preset = replace(gates.preset, seeds=gates.support_k0_eligible_seeds)
    overlap_rows = _run_phase(
        output_root / OVERLAP_PHASE / "seeds",
        phase=OVERLAP_PHASE,
        preset=overlap_preset,
        theta=gates.theta,
        anchors=gates.anchors,
        seed_to_device=gates.seed_to_device,
        devices=devices,
        source_hash=source_hash,
        gate_contract_sha256=gate_hash,
        rng_mapping_sha256=gates.rng_stream_mapping_sha256,
        worker=_overlap_worker,
        worker_arguments=(gates.protocol,),
        resume=resume,
    )
    overlap_summary = summarize_overlap(
        overlap_rows,
        prespecified_seeds=gates.prespecified_seeds,
        support_k0_eligible_seeds=gates.support_k0_eligible_seeds,
    )
    _write_or_verify_json(output_root / OVERLAP_PHASE / "summary.json", overlap_summary)
    _write_or_verify_text(
        output_root / OVERLAP_PHASE / "COMPLETE",
        f"overlap-complete summary_sha256={_json_sha256(overlap_summary)}\n",
    )

    if not overlap_summary["science_may_start"]:
        if (output_root / "SCIENCE_UNLOCK.json").exists() or (
            output_root / SCIENCE_PHASE
        ).exists():
            raise RuntimeError("science exists despite an overlap NO-GO")
        final = {
            "protocol": PROTOCOL,
            "dataset": DATASET,
            "status": "OVERLAP_NO_GO",
            "confirmation_status": "CONFIRMATION_GO",
            "science_unlocked": False,
            "coverage_generated": False,
            "prespecified_seed_count": PRESPECIFIED_SEED_COUNT,
            "support_k0_eligible_seed_count": len(gates.support_k0_eligible_seeds),
            "joint_overlap_pass_count": overlap_summary["joint_overlap_pass_count"],
            "seed_deletions": 0,
        }
        _write_or_verify_json(output_root / "FINAL_STATUS.json", final)
        _finalize_root(output_root, metadata, gates)
        return

    science_seeds = tuple(int(seed) for seed in overlap_summary["passed_seeds"])
    unlock = _science_unlock(gates, overlap_summary, science_seeds)
    _write_or_verify_json(output_root / "SCIENCE_UNLOCK.json", unlock)
    if not _valid_science_unlock(output_root, gates):
        raise RuntimeError("science unlock commit failed validation")

    science_preset = replace(gates.preset, seeds=science_seeds)
    science_results = _run_phase(
        output_root / SCIENCE_PHASE / "seeds",
        phase=SCIENCE_PHASE,
        preset=science_preset,
        theta=gates.theta,
        anchors=gates.anchors,
        seed_to_device=gates.seed_to_device,
        devices=devices,
        source_hash=source_hash,
        gate_contract_sha256=gate_hash,
        rng_mapping_sha256=gates.rng_stream_mapping_sha256,
        worker=_science_worker,
        worker_arguments=(gates.protocol,),
        resume=resume,
    )
    rows = [row for result in science_results for row in result["rows"]]
    bootstrap = _ensure_bootstrap_artifacts(output_root / SCIENCE_PHASE, gates.preset)
    summary = summarize_science(
        rows,
        preset=gates.preset,
        support_k0_eligible_seeds=gates.support_k0_eligible_seeds,
        selected_seeds=science_seeds,
        bootstrap_contract=bootstrap,
    )
    audit = coverage_audit(
        rows,
        summary=summary,
        support_k0_eligible_seeds=gates.support_k0_eligible_seeds,
        selected_seeds=science_seeds,
    )
    science_final = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "SCIENCE_COMPLETE",
        "methods": list(METHODS),
        "gammas": list(GAMMAS),
        "primary_gamma": PRIMARY_GAMMA,
        "primary_metric": PRIMARY_METRIC,
        "prespecified_seed_count": PRESPECIFIED_SEED_COUNT,
        "science_eligible_seed_count": len(science_seeds),
        "science_eligible_seeds": list(science_seeds),
        "seed_deletions": 0,
    }
    _write_or_verify_json(output_root / SCIENCE_PHASE / "summary.json", summary)
    _write_or_verify_json(output_root / SCIENCE_PHASE / "coverage_audit.json", audit)
    _write_or_verify_json(
        output_root / SCIENCE_PHASE / "FINAL_STATUS.json", science_final
    )
    _write_or_verify_text(
        output_root / SCIENCE_PHASE / "COMPLETE", "science-complete\n"
    )
    final = {
        **science_final,
        "confirmation_status": "CONFIRMATION_GO",
        "overlap_status": "OVERLAP_GO",
        "science_unlocked": True,
        "coverage_generated": True,
        "science_unlock_sha256": _json_sha256(unlock),
    }
    _write_or_verify_json(output_root / "FINAL_STATUS.json", final)
    _finalize_root(output_root, metadata, gates)


def _overlap_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    theta: BridgeTheta,
    anchor: ConfirmationAnchor,
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    context, base_identity, kernel_identity = _reconstruct_context(
        seed, preset, device, theta, anchor, protocol
    )
    metrics, diagnostics = v2._donor_overlap_probe(
        context, seed=seed, protocol=protocol
    )
    return {
        "seed": seed,
        "dataset": DATASET,
        "phase": OVERLAP_PHASE,
        "passed": donor_overlap_passes(metrics, protocol.donor_overlap_gate),
        "failure_consequence": "OVERLAP_NO_GO_NO_COVERAGE_SCIENCE",
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
        "base_context_identity": base_identity,
        "kernel_context_identity": kernel_identity,
        "theta": theta.to_dict(),
        "confirmation_anchor_identity_sha256": _json_sha256(anchor.to_dict()),
    }


def _science_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    theta: BridgeTheta,
    anchor: ConfirmationAnchor,
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    context, base_identity, kernel_identity = _reconstruct_context(
        seed, preset, device, theta, anchor, protocol
    )
    rows = v2.run_science_seed(
        seed,
        preset=preset,
        device=device,
        protocol=protocol,
        context=context,
    )
    return {
        "seed": seed,
        "dataset": DATASET,
        "phase": SCIENCE_PHASE,
        "interpretation_status": "EMPIRICAL_OVERLAP_SCREEN_PASSED",
        "rows": rows,
        "q_low": context.q_low,
        "q_high": context.q_high,
        "n_actions": context.n_actions,
        "action_mapping": {
            str(key): value for key, value in context.action_mapping.items()
        },
        "split_audit": v2._split_audit(context.splits),
        "base_context_identity": base_identity,
        "kernel_context_identity": kernel_identity,
        "theta": theta.to_dict(),
        "confirmation_anchor_identity_sha256": _json_sha256(anchor.to_dict()),
    }


def _reconstruct_context(
    seed: int,
    preset: DatasetPreset,
    device: str,
    theta: BridgeTheta,
    anchor: ConfirmationAnchor,
    protocol: ControlledClinicalExtensionConfig,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    context = v5._context_with_theta(base_context, theta)
    split_audit = v2._split_audit(base_context.splits)
    base_identity = v2._context_identity(base_context)
    kernel_identity = v5._candidate_context_identity(
        base_context, context.environment, theta
    )
    if (
        not anchor.support_passed
        or not anchor.k0_passed
        or split_audit != anchor.split_audit
        or base_identity != anchor.base_context_identity
        or kernel_identity != anchor.kernel_context_identity
        or split_audit.get("split_fractions") != list(ROLE_SPLIT)
    ):
        raise RuntimeError(f"reconstructed B02 context differs for seed {seed}")
    return context, base_identity, kernel_identity


def _run_phase(
    root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    theta: BridgeTheta,
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
    expected_files = {f"seed_{seed:06d}.json" for seed in preset.seeds} | {"COMPLETE"}
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise RuntimeError(f"invalid {phase} root: {root}")
    if root.exists() and not resume:
        raise FileExistsError(f"fresh {phase} root already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    children = list(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise RuntimeError(f"unsafe {phase} artifact entry")
    observed = {path.name for path in children}
    if observed - expected_files:
        raise RuntimeError(f"unexpected {phase} artifacts: {sorted(observed)}")
    completed: dict[int, dict[str, Any]] = {}
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
    if pending and (root / "COMPLETE").exists():
        raise RuntimeError(f"{phase} COMPLETE exists with missing seeds")
    if pending:
        groups = {
            device: tuple(seed for seed in pending if seed_to_device[seed] == device)
            for device in devices
        }
        with ProcessPoolExecutor(
            max_workers=len(devices), mp_context=get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(
                    v4science._phase_group,
                    seeds,
                    device,
                    preset,
                    theta,
                    anchors,
                    worker,
                    worker_arguments,
                ): device
                for device, seeds in groups.items()
                if seeds
            }
            for future in as_completed(futures):
                for seed, device, result in future.result():
                    payload = _phase_payload(
                        phase=phase,
                        preset=preset,
                        seed=seed,
                        device=device,
                        theta=theta,
                        anchor=anchors[seed],
                        source_hash=source_hash,
                        gate_contract_sha256=gate_contract_sha256,
                        rng_mapping_sha256=rng_mapping_sha256,
                        result=result,
                    )
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
        raise RuntimeError(f"{phase} did not complete its exact eligible seed bank")
    _write_or_verify_text(root / "COMPLETE", "complete\n")
    return [completed[seed] for seed in preset.seeds]


def _phase_payload(
    *,
    phase: str,
    preset: DatasetPreset,
    seed: int,
    device: str,
    theta: BridgeTheta,
    anchor: ConfirmationAnchor,
    source_hash: str,
    gate_contract_sha256: str,
    rng_mapping_sha256: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "phase": phase,
        "dataset": preset.name,
        "seed": seed,
        "device": device,
        "source_tree_sha256": source_hash,
        "gate_contract_sha256": gate_contract_sha256,
        "rng_stream_mapping_sha256": rng_mapping_sha256,
        "theta_sha256": _json_sha256(theta.to_dict()),
        "confirmation_anchor_sha256": _json_sha256(anchor.to_dict()),
        "result": dict(result),
    }


def _validate_phase_payload(
    payload: Mapping[str, Any],
    *,
    phase: str,
    preset: DatasetPreset,
    seed: int,
    device: str,
    theta: BridgeTheta,
    anchor: ConfirmationAnchor,
    source_hash: str,
    gate_contract_sha256: str,
    rng_mapping_sha256: str,
) -> None:
    expected = _phase_payload(
        phase=phase,
        preset=preset,
        seed=seed,
        device=device,
        theta=theta,
        anchor=anchor,
        source_hash=source_hash,
        gate_contract_sha256=gate_contract_sha256,
        rng_mapping_sha256=rng_mapping_sha256,
        result=payload.get("result", {}),
    )
    if set(payload) != set(expected) or any(
        payload.get(key) != value for key, value in expected.items() if key != "result"
    ):
        raise RuntimeError(f"{phase} provenance differs for seed {seed}")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError(f"{phase} result is malformed for seed {seed}")
    if (
        result.get("seed") != seed
        or result.get("dataset") != DATASET
        or result.get("phase") != phase
        or result.get("theta") != theta.to_dict()
        or result.get("split_audit") != anchor.split_audit
        or result.get("base_context_identity") != anchor.base_context_identity
        or result.get("kernel_context_identity") != anchor.kernel_context_identity
        or result.get("confirmation_anchor_identity_sha256")
        != _json_sha256(anchor.to_dict())
    ):
        raise RuntimeError(f"{phase} context differs for seed {seed}")
    if phase == OVERLAP_PHASE:
        _validate_overlap_result(result, preset)
    elif phase == SCIENCE_PHASE:
        _validate_science_result(result, preset)
    else:
        raise RuntimeError(f"unknown science phase: {phase}")


def _validate_overlap_result(result: Mapping[str, Any], preset: DatasetPreset) -> None:
    if result.get("failure_consequence") != "OVERLAP_NO_GO_NO_COVERAGE_SCIENCE":
        raise RuntimeError("overlap failure consequence differs")
    v2_view = dict(result)
    v2_view["interpretation_if_failed"] = "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
    v2_view["context_identity"] = v2_view.pop("base_context_identity")
    for key in (
        "failure_consequence",
        "kernel_context_identity",
        "theta",
        "confirmation_anchor_identity_sha256",
    ):
        v2_view.pop(key)
    if not v2._valid_overlap_result(v2_view):
        raise RuntimeError(f"{preset.name} overlap result violates frozen semantics")


def _validate_science_result(result: Mapping[str, Any], preset: DatasetPreset) -> None:
    if result.get("interpretation_status") != "EMPIRICAL_OVERLAP_SCREEN_PASSED":
        raise RuntimeError("coverage science requires an overlap-passed interpretation")
    v2_view = dict(result)
    v2_view["context_identity"] = v2_view.pop("base_context_identity")
    for key in (
        "kernel_context_identity",
        "theta",
        "confirmation_anchor_identity_sha256",
    ):
        v2_view.pop(key)
    if not v2._valid_science_result(v2_view, preset):
        raise RuntimeError(f"{preset.name} science result violates frozen semantics")


def summarize_overlap(
    rows: Sequence[Mapping[str, Any]],
    *,
    prespecified_seeds: Sequence[int],
    support_k0_eligible_seeds: Sequence[int],
) -> dict[str, Any]:
    prespecified = tuple(int(seed) for seed in prespecified_seeds)
    indexed = {int(row["seed"]): row for row in rows}
    eligible = tuple(int(seed) for seed in support_k0_eligible_seeds)
    if (
        prespecified != CONFIRMATION_SEEDS
        or len(eligible) < MINIMUM_JOINT_PASS_COUNT
        or tuple(seed for seed in prespecified if seed in set(eligible)) != eligible
        or len(indexed) != len(rows)
        or tuple(indexed) != eligible
        or any(not isinstance(row.get("passed"), bool) for row in rows)
    ):
        raise RuntimeError("overlap rows do not cover the exact eligible seed bank")
    passed = tuple(seed for seed in eligible if bool(indexed[seed]["passed"]))
    science_may_start = len(passed) >= MINIMUM_JOINT_PASS_COUNT
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "OVERLAP_GO" if science_may_start else "OVERLAP_NO_GO",
        "gate": "gamma=-4 q_mid+q_high empirical donor-overlap screen",
        "thresholds": {
            "local_ess_p01": 10.0,
            "median_ess_fraction": 0.25,
            "maximum_donor_probability": 0.25,
        },
        "prespecified_seed_count": len(prespecified),
        "support_k0_eligible_seed_count": len(eligible),
        "support_k0_eligible_seeds": list(eligible),
        "overlap_bank_complete": True,
        "overlap_completed_seed_count": len(rows),
        "joint_overlap_pass_count": len(passed),
        "minimum_joint_overlap_pass_count": MINIMUM_JOINT_PASS_COUNT,
        "passed_seeds": list(passed),
        "failed_seeds": [seed for seed in eligible if seed not in passed],
        "science_may_start": science_may_start,
        "failure_consequence": "OVERLAP_NO_GO_NO_COVERAGE_SCIENCE",
        "seed_deletions": 0,
    }


def _science_unlock(
    gates: GateBundle,
    overlap_summary: Mapping[str, Any],
    science_seeds: Sequence[int],
) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in science_seeds)
    if (
        overlap_summary.get("status") != "OVERLAP_GO"
        or overlap_summary.get("science_may_start") is not True
        or overlap_summary.get("passed_seeds") != list(seeds)
        or len(seeds) < MINIMUM_JOINT_PASS_COUNT
    ):
        raise RuntimeError("science unlock requires the frozen overlap GO")
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "SCIENCE_UNLOCK",
        "confirmation_binding_sha256": _json_sha256(gates.confirmation_binding),
        "gate_contract_sha256": _json_sha256(gates.contract),
        "source_tree_sha256": gates.active_source_tree_sha256,
        "config_sha256": _file_sha256(CONFIG_PATH),
        "overlap_summary_sha256": _json_sha256(overlap_summary),
        "role_split": list(ROLE_SPLIT),
        "bridge_candidate": gates.theta.to_dict(),
        "prespecified_seeds": list(gates.prespecified_seeds),
        "science_eligible_seeds": list(seeds),
        "science_eligible_seed_count": len(seeds),
        "selection_rate_denominator": PRESPECIFIED_SEED_COUNT,
        "science_contract": dict(gates.science_contract),
        "coverage_may_start": True,
        "seed_deletions": 0,
    }


def _valid_science_unlock(root: Path, gates: GateBundle) -> bool:
    path = root / "SCIENCE_UNLOCK.json"
    overlap_path = root / OVERLAP_PHASE / "summary.json"
    complete_path = root / OVERLAP_PHASE / "COMPLETE"
    if not path.is_file() or not overlap_path.is_file() or not complete_path.is_file():
        return False
    try:
        overlap = _read_json(overlap_path)
        unlock = _read_json(path)
    except RuntimeError:
        return False
    expected_marker = f"overlap-complete summary_sha256={_json_sha256(overlap)}\n"
    if (
        overlap.get("status") != "OVERLAP_GO"
        or overlap.get("science_may_start") is not True
        or overlap.get("joint_overlap_pass_count", 0) < MINIMUM_JOINT_PASS_COUNT
        or complete_path.read_text() != expected_marker
    ):
        return False
    seeds = tuple(int(seed) for seed in overlap.get("passed_seeds", ()))
    return unlock == _science_unlock(gates, overlap, seeds)


def _ensure_bootstrap_artifacts(root: Path, preset: DatasetPreset) -> dict[str, Any]:
    contract = v4science._ensure_bootstrap_artifacts(root, preset)
    contract["selected_subset_rule"] = (
        "for selected-set size n, use floor(U[:, :n] * n) while retaining the "
        "complete prespecified 10000x20 seed bank"
    )
    return contract


def marginal_worst_stage_coverage(coverage: np.ndarray) -> float:
    """Compute WSC as the minimum stage after averaging complete seed vectors."""

    values = np.asarray(coverage, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("WSC requires a nonempty seed-by-stage matrix")
    return float(values.mean(axis=0).min())


def summarize_science(
    rows: list[dict[str, Any]],
    *,
    preset: DatasetPreset,
    support_k0_eligible_seeds: tuple[int, ...],
    selected_seeds: tuple[int, ...],
    bootstrap_contract: dict[str, Any],
) -> dict[str, Any]:
    if any(seed not in support_k0_eligible_seeds for seed in selected_seeds):
        raise RuntimeError("overlap-eligible seeds must be support/K0 eligible")
    summary = v2.summarize_science(
        rows,
        preset=preset,
        selected_seeds=selected_seeds,
        interpretation_status="EMPIRICAL_OVERLAP_SCREEN_PASSED",
        bootstrap_contract=bootstrap_contract,
    )
    summary.update(
        {
            "protocol": PROTOCOL,
            "role": "post_failure_cxr_environment_support_science",
            "seeds_prespecified": list(CONFIRMATION_SEEDS),
            "seeds_support_k0_eligible": list(support_k0_eligible_seeds),
            "seeds_support_k0_overlap_eligible": list(selected_seeds),
            "seeds_k0_eligible": list(support_k0_eligible_seeds),
            "compatibility_field_semantics": {
                "seeds_k0_eligible": (
                    "alias of seeds_support_k0_eligible before donor-overlap screening"
                ),
                "aggregates[].n_k0_eligible_seeds": (
                    "count of support/K0-eligible seeds before donor-overlap screening"
                ),
                "aggregates[].methods[].n_k0_eligible": (
                    "count of support/K0-eligible seeds before donor-overlap screening"
                ),
            },
            "coverage_conditioning": (
                "successful method selection among support/K0/overlap-eligible seeds"
            ),
            "selection_rate_denominator": "all 20 prespecified confirmation seeds",
            "primary_gamma": PRIMARY_GAMMA,
            "primary_metric": PRIMARY_METRIC,
            "mean_coverage_is_supplementary": True,
            "seed_deletions": 0,
        }
    )
    aggregates = summary.get("aggregates")
    if not isinstance(aggregates, list) or len(aggregates) != len(GAMMAS):
        raise RuntimeError("science summary lacks all five signed gammas")
    for aggregate, gamma in zip(aggregates, GAMMAS, strict=True):
        aggregate.update(
            {
                "n_k0_eligible_seeds": len(support_k0_eligible_seeds),
                "n_support_k0_eligible_seeds": len(support_k0_eligible_seeds),
                "n_support_k0_overlap_eligible_seeds": len(selected_seeds),
            }
        )
        selected = sorted(
            (row for row in rows if float(row["gamma"]) == gamma),
            key=lambda row: int(row["seed"]),
        )
        if tuple(int(row["seed"]) for row in selected) != selected_seeds:
            raise RuntimeError(f"science summary seed set differs for gamma={gamma}")
        for method in METHODS:
            available = [
                row
                for row in selected
                if bool(row["methods"][method]["selection_available"])
            ]
            method_summary = aggregate["methods"][method]
            count = len(available)
            if (
                method_summary["n_selected"] != count
                or method_summary["n_prespecified"] != PRESPECIFIED_SEED_COUNT
                or method_summary["selection_rate"] != count / PRESPECIFIED_SEED_COUNT
                or method_summary["selection_rate_ci95"]
                != _wilson_interval(count, PRESPECIFIED_SEED_COUNT)
            ):
                raise RuntimeError(f"{method} selection denominator differs")
            method_summary.update(
                {
                    "n_k0_eligible": len(support_k0_eligible_seeds),
                    "n_support_k0_eligible": len(support_k0_eligible_seeds),
                    "n_support_k0_overlap_eligible": len(selected_seeds),
                    "selection_rate_denominator": PRESPECIFIED_SEED_COUNT,
                    "WSC_formula": PRIMARY_METRIC,
                    "MeanCov_role": "supplementary",
                }
            )
            if not available:
                continue
            coverage = np.asarray(
                [row["methods"][method]["target_coverage"] for row in available],
                dtype=np.float64,
            )
            wsc = marginal_worst_stage_coverage(coverage)
            mean_coverage = float(coverage.mean(axis=0).mean())
            if (
                method_summary["target_marginal_worst_coverage"] != wsc
                or method_summary["target_mean_coverage"] != mean_coverage
            ):
                raise RuntimeError(f"{method} coverage aggregation differs")
    return summary


def coverage_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    support_k0_eligible_seeds: tuple[int, ...],
    selected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    if any(seed not in support_k0_eligible_seeds for seed in selected_seeds):
        raise RuntimeError("coverage audit seed is not support/K0 eligible")
    aggregates = {float(row["gamma"]): row for row in summary["aggregates"]}
    records = []
    for gamma in GAMMAS:
        selected = sorted(
            (row for row in rows if float(row["gamma"]) == gamma),
            key=lambda row: int(row["seed"]),
        )
        if tuple(int(row["seed"]) for row in selected) != selected_seeds:
            raise RuntimeError("coverage audit seed bank differs")
        for method in METHODS:
            available = [
                row
                for row in selected
                if bool(row["methods"][method]["selection_available"])
            ]
            metrics = None
            if available:
                coverage = np.asarray(
                    [row["methods"][method]["target_coverage"] for row in available],
                    dtype=np.float64,
                )
                stage = coverage.mean(axis=0)
                metrics = {
                    "stage_coverage": stage.tolist(),
                    "WSC": marginal_worst_stage_coverage(coverage),
                    "MeanCov": float(stage.mean()),
                }
                reported = aggregates[gamma]["methods"][method]
                if (
                    metrics["stage_coverage"] != reported["target_coverage_by_stage"]
                    or metrics["WSC"] != reported["target_marginal_worst_coverage"]
                    or metrics["MeanCov"] != reported["target_mean_coverage"]
                ):
                    raise RuntimeError("coverage audit does not match the summary")
            records.append(
                {
                    "gamma": gamma,
                    "method": method,
                    "n_support_k0_eligible": len(support_k0_eligible_seeds),
                    "n_support_k0_overlap_eligible": len(selected_seeds),
                    "n_selected": len(available),
                    "selection_rate_denominator": PRESPECIFIED_SEED_COUNT,
                    "metrics": metrics,
                }
            )
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "COVERAGE_AUDIT_COMPLETE",
        "primary_metric": PRIMARY_METRIC,
        "formula_verified": True,
        "mean_coverage_is_supplementary": True,
        "all_six_methods_present": True,
        "all_five_gammas_present": True,
        "coverage_conditioning": (
            "successful method selection among support/K0/overlap-eligible seeds"
        ),
        "seeds_support_k0_eligible": list(support_k0_eligible_seeds),
        "seeds_support_k0_overlap_eligible": list(selected_seeds),
        "support_k0_eligible_seed_count": len(support_k0_eligible_seeds),
        "support_k0_overlap_eligible_seed_count": len(selected_seeds),
        "science_eligible_seeds": list(selected_seeds),
        "selection_rate_denominator": PRESPECIFIED_SEED_COUNT,
        "records": records,
    }


def _science_metadata(
    gates: GateBundle,
    *,
    devices: tuple[str, ...],
    audit_go_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = {
        "protocol": PROTOCOL,
        "role": "strict_post_confirmation_cxr_environment_support_science",
        "output_root": str(OUTPUT_ROOT),
        "devices": list(devices),
        "source_tree_sha256": gates.active_source_tree_sha256,
        "source_snapshot": dict(source_snapshot),
        "read_only_audit_status": "GO",
        "read_only_audit_go_sha256": audit_go_sha256,
        "gate_contract": gates.contract,
        "gate_contract_sha256": _json_sha256(gates.contract),
        "confirmation_binding": gates.confirmation_binding,
        "confirmation_binding_sha256": _json_sha256(gates.confirmation_binding),
        "science_rng_audit": gates.science_rng_audit,
        "rng_stream_mapping_sha256": gates.rng_stream_mapping_sha256,
        "seed_to_device": {
            str(seed): gates.seed_to_device[seed]
            for seed in gates.support_k0_eligible_seeds
        },
        "role_split": list(ROLE_SPLIT),
        "bridge_candidate": gates.theta.to_dict(),
        "science_contract": gates.science_contract,
        "coverage_may_start_only_after_science_unlock": True,
        "seed_deletion_permitted": False,
    }
    return json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False))


def _active_source_snapshot() -> tuple[str, dict[str, Any]]:
    source_hash = experiment_tree_sha256()
    snapshot = v4science._build_source_snapshot()
    manifest = json.loads(snapshot["manifest_bytes"])
    manifest["protocol"] = PROTOCOL
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    snapshot["manifest_bytes"] = manifest_bytes
    snapshot["contract"] = {
        **snapshot["contract"],
        "manifest_path": f"provenance/source_manifest_{manifest_hash}.json",
        "manifest_sha256": manifest_hash,
        "manifest_bytes": len(manifest_bytes),
    }
    if experiment_tree_sha256() != source_hash:
        raise RuntimeError("source changed while building the science snapshot")
    return source_hash, snapshot


def _prepare_root(
    root: Path,
    metadata: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if metadata.get("read_only_audit_status") != "GO" or metadata.get(
        "read_only_audit_go_sha256"
    ) != metadata.get("gate_contract_sha256"):
        raise RuntimeError("science root lacks the exact read-only audit GO")
    if resume:
        if not root.is_dir() or root.is_symlink():
            raise FileNotFoundError("science resume requires the existing frozen root")
        if _read_json(root / "metadata.json") != metadata:
            raise RuntimeError("science resume metadata differs")
        v4science._verify_source_snapshot(root, metadata["source_snapshot"])
        return
    if root.exists():
        raise FileExistsError(f"fresh science output root already exists: {root}")
    root.mkdir(parents=True)
    v4._atomic_write(
        root / snapshot["contract"]["archive_path"], snapshot["archive_bytes"]
    )
    v4._atomic_write(
        root / snapshot["contract"]["manifest_path"], snapshot["manifest_bytes"]
    )
    _write_json(root / "metadata.json", metadata)
    v4science._verify_source_snapshot(root, metadata["source_snapshot"])


def _validate_partial_root(
    root: Path, metadata: Mapping[str, Any], gates: GateBundle
) -> None:
    allowed = _all_allowed_paths(metadata, gates)
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"symlinks are forbidden in science artifacts: {symlinks}")
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    unexpected = observed - allowed
    if unexpected:
        raise RuntimeError(f"unexpected science artifacts: {sorted(unexpected)}")
    unlock_exists = (root / "SCIENCE_UNLOCK.json").is_file()
    science_exists = any(path.startswith(f"{SCIENCE_PHASE}/") for path in observed)
    if science_exists and (not unlock_exists or not _valid_science_unlock(root, gates)):
        raise RuntimeError("coverage artifacts exist before a valid science unlock")


def _all_allowed_paths(metadata: Mapping[str, Any], gates: GateBundle) -> set[str]:
    source = metadata["source_snapshot"]
    paths = {
        "metadata.json",
        str(source["archive_path"]),
        str(source["manifest_path"]),
        "eligibility.json",
        f"{OVERLAP_PHASE}/summary.json",
        f"{OVERLAP_PHASE}/COMPLETE",
        f"{OVERLAP_PHASE}/seeds/COMPLETE",
        "SCIENCE_UNLOCK.json",
        f"{SCIENCE_PHASE}/bootstrap_uniforms.npy",
        f"{SCIENCE_PHASE}/bootstrap_indices.npy",
        f"{SCIENCE_PHASE}/summary.json",
        f"{SCIENCE_PHASE}/coverage_audit.json",
        f"{SCIENCE_PHASE}/FINAL_STATUS.json",
        f"{SCIENCE_PHASE}/COMPLETE",
        f"{SCIENCE_PHASE}/seeds/COMPLETE",
        "FINAL_STATUS.json",
        "manifest.json",
        "COMPLETE",
    }
    for seed in gates.support_k0_eligible_seeds:
        paths.add(f"{OVERLAP_PHASE}/seeds/seed_{seed:06d}.json")
        paths.add(f"{SCIENCE_PHASE}/seeds/seed_{seed:06d}.json")
    return paths


def _finalize_root(root: Path, metadata: Mapping[str, Any], gates: GateBundle) -> None:
    if experiment_tree_sha256() != gates.active_source_tree_sha256:
        raise RuntimeError("source changed during CXR science")
    refreshed = verify_gate_bundle(
        devices=tuple(metadata["devices"]), confirmation_root=CONFIRMATION_ROOT
    )
    if refreshed.contract != gates.contract:
        raise RuntimeError("frozen confirmation changed during CXR science")
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("science metadata changed")
    _write_or_verify_manifest(root)
    final = _read_json(root / "FINAL_STATUS.json")
    marker = (
        f"complete source_tree_sha256={gates.active_source_tree_sha256} "
        f"gate_contract_sha256={_json_sha256(gates.contract)} "
        f"confirmation_binding_sha256={_json_sha256(gates.confirmation_binding)} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={_file_sha256(root / 'manifest.json')}\n"
    )
    _write_or_verify_text(root / "COMPLETE", marker)
    _validate_complete_root(root, metadata, gates)


def _validate_complete_root(
    root: Path, metadata: Mapping[str, Any], gates: GateBundle
) -> None:
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("completed science metadata differs")
    v4science._verify_source_snapshot(root, metadata["source_snapshot"])
    _verify_manifest(root)
    final = _read_json(root / "FINAL_STATUS.json")
    expected_paths = _expected_complete_paths(root, metadata, gates, final)
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != expected_paths:
        raise RuntimeError("completed science artifact set differs")
    marker = (
        f"complete source_tree_sha256={gates.active_source_tree_sha256} "
        f"gate_contract_sha256={_json_sha256(gates.contract)} "
        f"confirmation_binding_sha256={_json_sha256(gates.confirmation_binding)} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={_file_sha256(root / 'manifest.json')}\n"
    )
    if (root / "COMPLETE").read_text() != marker:
        raise RuntimeError("science COMPLETE marker differs")


def _expected_complete_paths(
    root: Path,
    metadata: Mapping[str, Any],
    gates: GateBundle,
    final: Mapping[str, Any],
) -> set[str]:
    source = metadata["source_snapshot"]
    paths = {
        "metadata.json",
        str(source["archive_path"]),
        str(source["manifest_path"]),
        "eligibility.json",
        f"{OVERLAP_PHASE}/summary.json",
        f"{OVERLAP_PHASE}/COMPLETE",
        f"{OVERLAP_PHASE}/seeds/COMPLETE",
        "FINAL_STATUS.json",
        "manifest.json",
        "COMPLETE",
    }
    paths.update(
        f"{OVERLAP_PHASE}/seeds/seed_{seed:06d}.json"
        for seed in gates.support_k0_eligible_seeds
    )
    if final.get("status") == "OVERLAP_NO_GO":
        if final.get("coverage_generated") is not False:
            raise RuntimeError("overlap NO-GO contains coverage")
        return paths
    if final.get("status") != "SCIENCE_COMPLETE" or not _valid_science_unlock(
        root, gates
    ):
        raise RuntimeError("completed science status/unlock differs")
    science_seeds = tuple(
        int(seed)
        for seed in _read_json(root / "SCIENCE_UNLOCK.json")["science_eligible_seeds"]
    )
    paths.update(
        {
            "SCIENCE_UNLOCK.json",
            f"{SCIENCE_PHASE}/bootstrap_uniforms.npy",
            f"{SCIENCE_PHASE}/bootstrap_indices.npy",
            f"{SCIENCE_PHASE}/summary.json",
            f"{SCIENCE_PHASE}/coverage_audit.json",
            f"{SCIENCE_PHASE}/FINAL_STATUS.json",
            f"{SCIENCE_PHASE}/COMPLETE",
            f"{SCIENCE_PHASE}/seeds/COMPLETE",
        }
    )
    paths.update(
        f"{SCIENCE_PHASE}/seeds/seed_{seed:06d}.json" for seed in science_seeds
    )
    return paths


def _write_or_verify_manifest(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden in science artifacts: {path}")
        relative = path.relative_to(root)
        if not path.is_file() or relative in {Path("manifest.json"), Path("COMPLETE")}:
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    _write_or_verify_json(
        root / "manifest.json",
        {"protocol": PROTOCOL, "artifact_count": len(entries), "artifacts": entries},
    )


def _verify_manifest(root: Path) -> None:
    manifest = _read_json(root / "manifest.json")
    entries = manifest.get("artifacts")
    if (
        manifest.get("protocol") != PROTOCOL
        or not isinstance(entries, list)
        or manifest.get("artifact_count") != len(entries)
    ):
        raise RuntimeError("science manifest header differs")
    expected = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RuntimeError("science manifest entry is malformed")
        relative = Path(str(entry.get("path", "")))
        path = _safe_child(root, relative)
        if (
            relative in {Path("manifest.json"), Path("COMPLETE")}
            or relative in expected
        ):
            raise RuntimeError("science manifest path is duplicated or reserved")
        expected.add(relative)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != entry.get("bytes")
            or _file_sha256(path) != entry.get("sha256")
        ):
            raise RuntimeError(f"science manifest mismatch: {relative}")
    observed = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root) not in {Path("manifest.json"), Path("COMPLETE")}
    }
    if observed != expected:
        raise RuntimeError("science manifest file set differs")


def _validate_cuda_devices(devices: tuple[str, ...]) -> None:
    if devices != ("cuda:0", "cuda:1"):
        raise RuntimeError("science is frozen to devices cuda:0,cuda:1")
    cxr._validate_devices(devices)


def _safe_child(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError("artifact path escapes its root")
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents:
        raise RuntimeError("artifact path escapes its root")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError(f"JSON artifact may not be a symlink: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return value


def _write_or_verify_json(path: Path, value: object) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or _read_json(path) != value:
            raise RuntimeError(f"existing JSON artifact differs: {path}")
        return
    _write_json(path, value)


def _write_or_verify_text(path: Path, value: str) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_text() != value:
            raise RuntimeError(f"existing text artifact differs: {path}")
        return
    _write_text(path, value)


def _write_json(path: Path, value: object) -> None:
    cxr._write_json(path, value)


def _write_text(path: Path, value: str) -> None:
    cxr._write_text(path, value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    main()
