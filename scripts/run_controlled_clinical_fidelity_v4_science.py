"""Run strict post-confirmation clinical-v4 signed-gamma science.

The read-only audit never creates an output directory or draws random numbers::

    python scripts/run_controlled_clinical_fidelity_v4_science.py audit \
      --devices cuda:0,cuda:1

After an independent review of that audit, run the frozen workflow with the
reported contract hash::

    python scripts/run_controlled_clinical_fidelity_v4_science.py run \
      --devices cuda:0,cuda:1 --audit-go-sha256 <reported-hash>

Resume the same root by appending ``--resume``.  All confirmed datasets finish
their donor-overlap screens before any coverage row may be generated.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import io
import json
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
import scripts.run_controlled_clinical_fidelity_v3_science as v3science  # noqa: E402
import scripts.run_controlled_clinical_fidelity_v4 as v4  # noqa: E402
from scripts.run_controlled_six_method_benchmark import (  # noqa: E402
    TARGET_ADAPTATION_BUDGET,
    _student_t_interval,
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
from scpcp.controlled_clinical_fidelity_v4 import (  # noqa: E402
    DATASETS,
    FidelityV4Config,
    FrozenAnchor,
    RepairTheta,
    load_fidelity_v4_config,
)


Theta = FrozenAnchor | RepairTheta

PROTOCOL = "controlled_clinical_fidelity_v4_signed_gamma_science_v1"
OUTPUT_ROOT = (
    ROOT / "results/work/controlled_clinical_fidelity_v4_signed_gamma_science"
).resolve()
DEVELOPMENT_ROOT = v4.DEVELOPMENT_ROOT
CONFIRMATION_ROOT = v4.CONFIRMATION_RETRY_ROOT
CONFIRMED_DATASETS = ("mimic_iv", "eicu", "inspire")
UNOPENED_DATASETS = ("mimic_cxr",)
PRIMARY_GAMMA = -4.0
TARGET_COVERAGE = 0.90
PRIMARY_METRIC = "min_t mean_seed(target_coverage_seed_t)"
OVERLAP_PHASE = "donor_overlap"
SCIENCE_PHASE = "science"
BOOTSTRAP_RESAMPLES = 10_000

EXPECTED_CONFIRMATION_FILES = {
    "COMPLETE": "e156b19e9cc086a0506aa8cef34f9807ddad66ef670a2b1571705b97924b3fcf",
    "manifest.json": "fe48c9d7f9d356db9765245b62472cc64ce34d7bd0d2b8fb5d900acd9433c69a",
    "metadata.json": "1648e829d8c1cbb8ac4bc174c62b686a1f9403cead6b337c7bf4e09f80b351ca",
    "FINAL_STATUS.json": "25df7a510a929d65847f3d65294bfb7b436cf6bc96b0433bd4b82800425a51ca",
    "administrative_retry_amendment.json": "528fb9f19ca158c4ff255e50cde8c577256e35da358ad282f1cb6cc8b83eb363",
    "support_replay_verification.json": "e1f6d78ec707d7391445cc4d14f56b9b94a11775b3884931e1c9537ef48a7412",
    "provenance/source_manifest_cc6bb64e5f4e8e1d74d23f725236d03fd41ce1e015f4b3553f3b14b062e9a33a.json": "cc6bb64e5f4e8e1d74d23f725236d03fd41ce1e015f4b3553f3b14b062e9a33a",
    "provenance/source_snapshot_a0e13689c5191ea1202e608712adb394719a7d85ee0189cc20521636d3fb4f2c.tar": "a0e13689c5191ea1202e608712adb394719a7d85ee0189cc20521636d3fb4f2c",
}
EXPECTED_DEVELOPMENT_FILES = {
    "COMPLETE": "6d05b9e8e1411c7d75f2247a5d8c8fc2479557fb3d365165682e4e706efff610",
    "manifest.json": "f7d207418590cbb947705caec4c777a0895d710b0665e7323c9a72ff623cc0be",
    "metadata.json": "a7b452b80e13670c5d845fd7ccad0a97a0b39e805869e1c0c884960bd3dfeebc",
    "FINAL_STATUS.json": "a098c77436b8ad8415ad081e6a5af9b5a4dfb329a9eecd4956cd9718c347b368",
    "frozen_settings.json": "0d2f0e676cd19c88772b6972af480b3c431938eed1d10a98db88645f90237ee8",
    "provenance/source_manifest_30e76938be9f247525e5f33ff8c4ad6c53cb77625aa94b297df5c1a67073d687.json": "30e76938be9f247525e5f33ff8c4ad6c53cb77625aa94b297df5c1a67073d687",
    "provenance/source_snapshot_f6f69c5f95e66878ae5a206aab6c378c5246e193d94d4ce547c49e5925003999.tar": "f6f69c5f95e66878ae5a206aab6c378c5246e193d94d4ce547c49e5925003999",
}
EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256 = (
    "9ba785a5d34899bda4dbd4eb3c8998f5be8b7f5dd8951bdd2f60d229628b8e21"
)
EXPECTED_CONFIRMATION_SOURCE_TREE_SHA256 = (
    "6a0e87b120cdae7b31ef640ddb514e62378e1f6157735680d6014b34003c57be"
)
EXPECTED_CONFIRMATION_MAPPING_SHA256 = (
    "3a78ec5afe69f57928de894a38803f5c369b33ab1db3f7c37bd403b974f75c72"
)
EXPECTED_AMENDMENT_SHA256 = (
    "b901a600a0ebba5cc815d53f3dc9c9c3f00d924912663999422bbb70ffd486da"
)
EXPECTED_ELIGIBLE_SEEDS = {
    "mimic_iv": tuple(range(115_000, 115_200, 10)),
    "eicu": tuple(seed for seed in range(116_000, 116_200, 10) if seed != 116_150),
    "inspire": tuple(range(117_000, 117_200, 10)),
}

SCIENCE_CONTRACT = {
    "methods": list(METHODS),
    "gammas": list(GAMMAS),
    "primary_default_gamma": PRIMARY_GAMMA,
    "target_coverage": TARGET_COVERAGE,
    "primary_metric": PRIMARY_METRIC,
    "calibration_trajectories": 3_000,
    "grid_trajectories": 1_000,
    "evaluation_trajectories": 20_000,
    "target_adaptation_trajectories": dict(TARGET_ADAPTATION_BUDGET),
    "policy_ratio_cap": 3.0,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "uncertainty": {
        "selection": "Wilson 95% interval on all 20 prespecified seeds",
        "stage_coverage": "two-sided Student-t across method-selected eligible seeds",
        "mean_coverage": "two-sided Student-t across method-selected eligible seeds",
        "stage_normalized_width": "two-sided Student-t across method-selected eligible seeds",
        "mean_normalized_width": "two-sided Student-t across method-selected eligible seeds",
        "wsc": "10000-draw complete-seed-vector percentile bootstrap",
        "paired": "10000-draw paired-seed-vector percentile bootstrap",
    },
    "eligibility": (
        "support PASS and K0 PASS intersection; denominator remains all 20 "
        "prespecified confirmation seeds"
    ),
    "eicu_seed_116150": (
        "unavailable for every method because support failed at stage 0/action 3 "
        "with 16 unique patients; its K0 PASS does not restore eligibility"
    ),
    "ranking_scope": "gamma=-4 within each overlap-passed dataset only",
    "nonprimary_gamma_role": "descriptive signed control curve",
    "cross_dataset_pooling_or_conjunction": False,
}


@dataclass(frozen=True)
class ConfirmationSeedAnchor:
    split_audit: Mapping[str, Any]
    kernel_identity: Mapping[str, Any]
    support_passed: bool
    k0_passed: bool


@dataclass(frozen=True)
class DatasetGate:
    preset: DatasetPreset
    theta: Theta
    prespecified_seeds: tuple[int, ...]
    eligible_seeds: tuple[int, ...]
    anchors: Mapping[int, ConfirmationSeedAnchor]
    eligibility_record: Mapping[str, Any]
    seed_to_device: Mapping[int, str]


@dataclass(frozen=True)
class GateBundle:
    fidelity_config: FidelityV4Config
    science_config: ControlledClinicalExtensionConfig
    active_source_tree_sha256: str
    confirmation_binding: Mapping[str, Any]
    rng_audit: Mapping[str, Any]
    datasets: Mapping[str, DatasetGate]
    contract: Mapping[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "run"))
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-go-sha256")
    args = parser.parse_args()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    gates = verify_gate_bundle(devices=devices)
    audit_hash = _json_sha256(gates.contract)
    if args.phase == "audit":
        if args.resume or args.audit_go_sha256 is not None:
            parser.error("audit does not accept --resume or --audit-go-sha256")
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "status": "READ_ONLY_AUDIT_GO",
                    "formal_rng_executed": False,
                    "output_root_created": False,
                    "audit_contract_sha256": audit_hash,
                    "confirmed_datasets": list(CONFIRMED_DATASETS),
                    "eligible_seed_counts": {
                        name: len(gates.datasets[name].eligible_seeds)
                        for name in CONFIRMED_DATASETS
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.audit_go_sha256 != audit_hash:
        raise RuntimeError(
            "run requires the exact --audit-go-sha256 reported by an independent audit"
        )
    v4._validate_devices(devices)
    run_post_confirmation_science(
        OUTPUT_ROOT,
        gates=gates,
        devices=devices,
        independent_audit_go_sha256=audit_hash,
        resume=args.resume,
    )
    print(OUTPUT_ROOT)


def verify_gate_bundle(*, devices: tuple[str, ...]) -> GateBundle:
    """Validate immutable v4 evidence with a zero-mutation parent invariant."""

    before = {
        str(root): _read_only_root_inventory(root)
        for root in (DEVELOPMENT_ROOT, CONFIRMATION_ROOT)
    }
    try:
        return _verify_gate_bundle_read_only(devices=devices)
    finally:
        after = {
            str(root): _read_only_root_inventory(root)
            for root in (DEVELOPMENT_ROOT, CONFIRMATION_ROOT)
        }
        if after != before:
            raise RuntimeError("read-only science audit mutated a parent evidence root")


def _verify_gate_bundle_read_only(*, devices: tuple[str, ...]) -> GateBundle:
    """Audit fixed bytes and semantic JSON without RNG or legacy validators."""

    if devices != ("cuda:0", "cuda:1"):
        raise RuntimeError("v4 science is frozen to devices cuda:0,cuda:1")
    fidelity_config = load_fidelity_v4_config(v4.CONFIG_PATH)
    science_config = v2.load_extension_config(v4.V2_CONFIG_PATH)
    _validate_science_constants(science_config, fidelity_config)

    development_binding, frozen = _validate_development_bundle(fidelity_config)
    metadata = _read_json(CONFIRMATION_ROOT / "metadata.json")
    _validate_fixed_confirmation_files()
    _verify_immutable_parent_manifest(CONFIRMATION_ROOT)
    _validate_parent_source_snapshot(
        CONFIRMATION_ROOT,
        metadata,
        expected_source_tree_sha256=EXPECTED_CONFIRMATION_SOURCE_TREE_SHA256,
    )

    final = _read_json(CONFIRMATION_ROOT / "FINAL_STATUS.json")
    amendment = _read_json(
        CONFIRMATION_ROOT / "administrative_retry_amendment.json"
    )
    support_replay = _read_json(
        CONFIRMATION_ROOT / "support_replay_verification.json"
    )
    _validate_parent_terminal_contract(
        metadata,
        final,
        amendment,
        support_replay,
        development_binding=development_binding,
        frozen=frozen,
    )
    rng_audit = _validate_rng_contract(
        metadata,
        science_config=science_config,
        fidelity_config=fidelity_config,
    )

    gates = {
        dataset: _read_json(CONFIRMATION_ROOT / dataset / "gate.json")
        for dataset in DATASETS
    }
    dataset_gates = {
        dataset: _validated_dataset_gate(
            dataset,
            gate=gates[dataset],
            metadata=metadata,
            science_config=science_config,
            fidelity_config=fidelity_config,
        )
        for dataset in CONFIRMED_DATASETS
    }
    _validate_unopened_cxr(gates["mimic_cxr"], final)
    _validate_confirmation_final(final, gates, amendment)
    _validate_parent_complete_marker(CONFIRMATION_ROOT, metadata, final)

    confirmation_binding = _root_binding(CONFIRMATION_ROOT)
    active_source_hash = experiment_tree_sha256()
    data_contracts = {
        dataset: v2._dataset_contract(science_config, dataset_gates[dataset].preset)
        for dataset in CONFIRMED_DATASETS
    }
    contract = {
        "protocol": PROTOCOL,
        "active_science_source_tree_sha256": active_source_hash,
        "confirmation_binding": confirmation_binding,
        "confirmation_binding_sha256": _json_sha256(confirmation_binding),
        "confirmation_source_tree_sha256": metadata["source_tree_sha256"],
        "confirmation_source_snapshot": metadata["source_snapshot"],
        "administrative_retry_amendment_sha256": metadata[
            "administrative_retry_amendment_sha256"
        ],
        "support_replay_verification_sha256": _json_sha256(support_replay),
        "development_binding": metadata["development_binding"],
        "development_binding_sha256": metadata["development_binding_sha256"],
        "parent_v3_binding": metadata["parent_v3_binding"],
        "parent_v3_binding_sha256": metadata["parent_v3_binding_sha256"],
        "rng_stream_mapping_sha256": rng_audit[
            "new_rng_stream_mapping_sha256"
        ],
        "rng_role": "exact_reuse_of_reserved_first_confirmation_mapping",
        "new_rng_bank_claimed": False,
        "confirmed_datasets": list(CONFIRMED_DATASETS),
        "unopened_datasets": list(UNOPENED_DATASETS),
        "dataset_theta": {
            dataset: v4._theta_to_dict(dataset_gates[dataset].theta)
            for dataset in CONFIRMED_DATASETS
        },
        "dataset_eligibility": {
            dataset: dataset_gates[dataset].eligibility_record
            for dataset in CONFIRMED_DATASETS
        },
        "data_contracts": data_contracts,
        "science_contract": SCIENCE_CONTRACT,
        "decision_scope": "per_dataset_independent",
        "pooled_or_universal_claim_permitted": False,
    }
    return GateBundle(
        fidelity_config=fidelity_config,
        science_config=science_config,
        active_source_tree_sha256=active_source_hash,
        confirmation_binding=confirmation_binding,
        rng_audit=rng_audit,
        datasets=dataset_gates,
        contract=contract,
    )


def _validate_science_constants(
    protocol: ControlledClinicalExtensionConfig,
    config: FidelityV4Config,
) -> None:
    protocol.validate()
    config.validate()
    if (
        tuple(METHODS)
        != ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")
        or tuple(protocol.gammas) != (-4.0, -2.0, 0.0, 2.0, 4.0)
        or (
            protocol.calibration_trajectories,
            protocol.grid_trajectories,
            protocol.reference_trajectories,
            protocol.online_trajectories,
            protocol.bootstrap_resamples,
        )
        != (3_000, 1_000, 20_000, 2_000, 10_000)
        or protocol.policy_ratio_cap != 3.0
        or dict(TARGET_ADAPTATION_BUDGET)
        != {
            "Standard CP": 0,
            "ACI": 2_000,
            "MFCS": 0,
            "SPCI": 2_000,
            "PRC": 2_000,
            "SC-PCP": 0,
        }
    ):
        raise RuntimeError("v4 science constants differ from the canonical all-six contract")


def _read_only_root_inventory(root: Path) -> tuple[dict[str, Any], ...]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"immutable parent root is missing or unsafe: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        stat = path.lstat()
        if path.is_symlink():
            rows.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "mode": stat.st_mode,
                    "target": os.readlink(path),
                }
            )
        elif path.is_file():
            rows.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": stat.st_mode,
                    "bytes": stat.st_size,
                    "sha256": _file_sha256(path),
                }
            )
        elif path.is_dir():
            rows.append(
                {
                    "path": relative,
                    "kind": "directory",
                    "mode": stat.st_mode,
                }
            )
        else:
            raise RuntimeError(f"unsupported parent artifact type: {path}")
    return tuple(rows)


def _validate_fixed_confirmation_files() -> None:
    observed = {
        relative: _file_sha256(CONFIRMATION_ROOT / relative)
        for relative in EXPECTED_CONFIRMATION_FILES
    }
    if observed != EXPECTED_CONFIRMATION_FILES:
        raise RuntimeError("v4 administrative confirmation files differ from frozen hashes")


def _validate_development_bundle(
    config: FidelityV4Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = {
        relative: _file_sha256(DEVELOPMENT_ROOT / relative)
        for relative in EXPECTED_DEVELOPMENT_FILES
    }
    if observed != EXPECTED_DEVELOPMENT_FILES:
        raise RuntimeError("v4 development files differ from frozen hashes")
    _verify_immutable_parent_manifest(DEVELOPMENT_ROOT)
    metadata = _read_json(DEVELOPMENT_ROOT / "metadata.json")
    final = _read_json(DEVELOPMENT_ROOT / "FINAL_STATUS.json")
    frozen = _read_json(DEVELOPMENT_ROOT / "frozen_settings.json")
    _validate_parent_source_snapshot(
        DEVELOPMENT_ROOT,
        metadata,
        expected_source_tree_sha256=EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256,
    )
    frozen_unhashed = {
        key: value for key, value in frozen.items() if key != "frozen_settings_sha256"
    }
    if (
        metadata.get("protocol") != v4.PROTOCOL
        or metadata.get("phase") != "development"
        or metadata.get("output_root") != str(DEVELOPMENT_ROOT)
        or metadata.get("config_path")
        != v4.CONFIG_PATH.relative_to(ROOT).as_posix()
        or metadata.get("config_sha256")
        != "3d2939a4d9c46b970b252ed642a4fe764dd1ed8db5c738f6ed81370274e8fded"
        or metadata.get("source_tree_sha256")
        != EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256
        or metadata.get("coverage_generation_permitted") is not False
        or metadata.get("scientific_result_execution_path_present") is not False
        or metadata.get("cross_dataset_conjunction_permitted") is not False
        or metadata.get("decision_scope") != "per_dataset_independent"
        or metadata.get("parent_v3_binding_sha256")
        != _json_sha256(metadata.get("parent_v3_binding"))
        or final.get("protocol") != v4.PROTOCOL
        or final.get("phase") != "development"
        or final.get("status") != "DEVELOPMENT_COMPLETE_PARTIAL_DATASET_GO"
        or tuple(final.get("development_go_datasets", ())) != CONFIRMED_DATASETS
        or tuple(final.get("development_no_go_datasets", ())) != UNOPENED_DATASETS
        or final.get("coverage_generated") is not False
        or final.get("cross_dataset_conjunction_used") is not False
        or final.get("candidate_seed_deletions") != 0
        or frozen.get("protocol") != v4.PROTOCOL
        or frozen.get("role")
        != "dataset_settings_frozen_before_any_fresh_confirmation"
        or frozen.get("development_source_tree_sha256")
        != EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256
        or frozen.get("development_config_sha256") != metadata["config_sha256"]
        or frozen.get("parent_v3_binding_sha256")
        != metadata["parent_v3_binding_sha256"]
        or frozen.get("development_decision_sha256") != _json_sha256(final)
        or frozen.get("frozen_settings_sha256") != _json_sha256(frozen_unhashed)
        or frozen.get("theta_by_dataset") != final.get("theta_by_dataset")
        or tuple(frozen.get("development_go_datasets", ())) != CONFIRMED_DATASETS
        or tuple(frozen.get("development_no_go_datasets", ())) != UNOPENED_DATASETS
        or frozen.get("coverage_generation_permitted") is not False
        or frozen.get("cross_dataset_conjunction_used") is not False
    ):
        raise RuntimeError("v4 development semantic contract differs")
    for dataset in CONFIRMED_DATASETS:
        theta = v4._theta_from_dict(frozen["theta_by_dataset"][dataset])
        if theta.dataset != dataset:
            raise RuntimeError(f"{dataset} frozen development theta differs")
    if "mimic_cxr" in frozen["theta_by_dataset"]:
        raise RuntimeError("MIMIC-CXR must not have a frozen development theta")
    _validate_parent_complete_marker(DEVELOPMENT_ROOT, metadata, final)
    binding = {
        "root": str(DEVELOPMENT_ROOT),
        "manifest_sha256": _file_sha256(DEVELOPMENT_ROOT / "manifest.json"),
        "manifest_bytes": (DEVELOPMENT_ROOT / "manifest.json").stat().st_size,
        "complete_sha256": _file_sha256(DEVELOPMENT_ROOT / "COMPLETE"),
        "complete_bytes": (DEVELOPMENT_ROOT / "COMPLETE").stat().st_size,
        "final_status_sha256": _json_sha256(final),
        "frozen_settings_sha256": frozen["frozen_settings_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "config_sha256": metadata["config_sha256"],
        "parent_v3_binding_sha256": metadata["parent_v3_binding_sha256"],
    }
    return binding, frozen


def _validate_parent_source_snapshot(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    expected_source_tree_sha256: str,
) -> None:
    contract = metadata.get("source_snapshot")
    if (
        metadata.get("source_tree_sha256") != expected_source_tree_sha256
        or not isinstance(contract, dict)
        or contract.get("archive_sha256") != _file_sha256(root / contract["archive_path"])
        or contract.get("manifest_sha256") != _file_sha256(root / contract["manifest_path"])
    ):
        raise RuntimeError("v4 parent source binding differs")
    manifest_path = root / contract["manifest_path"]
    archive_path = root / contract["archive_path"]
    manifest = _read_json(manifest_path)
    entries = manifest.get("files")
    if (
        set(manifest) != {"protocol", "format", "file_count", "files"}
        or manifest.get("protocol") != v4.PROTOCOL
        or manifest.get("format") != "deterministic_uncompressed_pax_tar"
        or not isinstance(entries, list)
        or manifest.get("file_count") != len(entries)
        or len(entries) != contract["file_count"]
    ):
        raise RuntimeError("v4 archived source manifest differs")
    expected = {}
    for entry in entries:
        _require_exact_keys(entry, {"path", "bytes", "sha256"}, "source entry")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in expected:
            raise RuntimeError("v4 archived source path is unsafe or duplicated")
        expected[relative.as_posix()] = entry
    observed = {}
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.name in observed:
                raise RuntimeError("v4 archived source member differs")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError("v4 archived source member is unreadable")
            content = stream.read()
            observed[member.name] = {
                "path": member.name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
    if observed != expected:
        raise RuntimeError("v4 archived source content differs from its manifest")


def _verify_immutable_parent_manifest(root: Path) -> None:
    manifest = _read_json(root / "manifest.json")
    entries = manifest.get("artifacts")
    if (
        set(manifest) != {"protocol", "artifact_count", "artifacts"}
        or manifest.get("protocol") != v4.PROTOCOL
        or not isinstance(entries, list)
        or manifest.get("artifact_count") != len(entries)
    ):
        raise RuntimeError("v4 parent manifest header differs")
    expected = set()
    for entry in entries:
        _require_exact_keys(entry, {"path", "bytes", "sha256"}, "parent manifest entry")
        relative = Path(entry["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative in {Path("manifest.json"), Path("COMPLETE")}
        ):
            raise RuntimeError("v4 parent manifest path is unsafe")
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or path in expected:
            raise RuntimeError("v4 parent manifest path escapes or is duplicated")
        expected.add(path)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != entry["bytes"]
            or _file_sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"v4 parent manifest mismatch: {path}")
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden in a v4 parent root: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative not in {Path("manifest.json"), Path("COMPLETE")}:
            observed.add(path.resolve())
    if observed != expected or len(expected) != len(entries):
        raise RuntimeError("v4 parent manifest does not bind the complete file set")


def _validate_parent_complete_marker(
    root: Path,
    metadata: Mapping[str, Any],
    final: Mapping[str, Any],
) -> None:
    expected = (
        f"complete phase={metadata['phase']} "
        f"source_tree_sha256={metadata['source_tree_sha256']} "
        f"config_sha256={metadata['config_sha256']} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={_file_sha256(root / 'manifest.json')}\n"
    )
    if (root / "COMPLETE").read_text() != expected:
        raise RuntimeError("v4 parent COMPLETE marker differs")


def _validate_parent_terminal_contract(
    metadata: Mapping[str, Any],
    final: Mapping[str, Any],
    amendment: Mapping[str, Any],
    support_replay: Mapping[str, Any],
    *,
    development_binding: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> None:
    replay_without_hash = {
        key: value
        for key, value in support_replay.items()
        if key != "verification_sha256"
    }
    if (
        metadata.get("protocol") != v4.PROTOCOL
        or metadata.get("phase") != "confirmation_retry"
        or metadata.get("output_root") != str(CONFIRMATION_ROOT)
        or tuple(metadata.get("devices", ())) != ("cuda:0", "cuda:1")
        or metadata.get("administrative_retry_protocol") != v4.RETRY_PROTOCOL
        or metadata.get("administrative_retry_amendment") != amendment
        or metadata.get("administrative_retry_amendment_sha256")
        != EXPECTED_AMENDMENT_SHA256
        or _json_sha256(amendment) != EXPECTED_AMENDMENT_SHA256
        or metadata.get("development_binding") != development_binding
        or metadata.get("development_binding_sha256")
        != _json_sha256(development_binding)
        or metadata.get("frozen_settings") != frozen
        or metadata.get("frozen_settings_sha256")
        != frozen.get("frozen_settings_sha256")
        or metadata.get("parent_v3_binding_sha256")
        != _json_sha256(metadata.get("parent_v3_binding"))
        or amendment.get("coverage_generation_permitted") is not False
        or amendment.get("scientific_result_execution_path_present") is not False
        or amendment.get("same_first_fresh_confirmation_lineage") is not True
        or amendment.get("same_frozen_dataset_settings") is not True
        or amendment.get("failed_attempt_artifacts_reused") is not False
        or support_replay.get("k0_permitted_after_this_verification") is not True
        or support_replay.get("all_opened_support_banks_complete") is not True
        or support_replay.get("mimic_cxr_support_or_k0_opened") is not False
        or support_replay.get("verification_sha256")
        != _json_sha256(replay_without_hash)
        or final.get("status") != "CONFIRMATION_COMPLETE_DATASET_INDEPENDENT"
        or tuple(final.get("confirmed_datasets", ())) != CONFIRMED_DATASETS
        or tuple(final.get("unconfirmed_datasets", ())) != UNOPENED_DATASETS
        or final.get("cross_dataset_conjunction_used") is not False
        or final.get("coverage_generated") is not False
        or final.get("independent_patient_confirmation_claimed") is not False
    ):
        raise RuntimeError("v4 confirmation terminal/amendment contract differs")
    failed_binding = amendment.get("failed_attempt_binding")
    source_delta = amendment.get("source_delta")
    if (
        not isinstance(failed_binding, dict)
        or amendment.get("failed_attempt_binding_sha256")
        != _json_sha256(failed_binding)
        or not isinstance(source_delta, dict)
        or amendment.get("source_delta_sha256") != _json_sha256(source_delta)
        or amendment.get("development_binding_sha256")
        != _json_sha256(development_binding)
        or amendment.get("frozen_settings_sha256")
        != frozen.get("frozen_settings_sha256")
    ):
        raise RuntimeError("v4 administrative amendment self-binding differs")


def _validate_confirmation_final(
    final: Mapping[str, Any],
    gates: Mapping[str, Mapping[str, Any]],
    amendment: Mapping[str, Any],
) -> None:
    if tuple(gates) != tuple(DATASETS):
        raise RuntimeError("v4 confirmation gate order differs")
    confirmed = [
        dataset
        for dataset in DATASETS
        if gates[dataset]["status"] == "CONFIRMATION_GATE_GO"
    ]
    opened = [
        dataset for dataset in DATASETS if gates[dataset]["confirmation_opened"]
    ]
    expected = {
        "protocol": v4.PROTOCOL,
        "phase": "confirmation_retry",
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
        "administrative_retry_protocol": v4.RETRY_PROTOCOL,
        "administrative_retry_amendment_sha256": _json_sha256(amendment),
        "same_first_fresh_confirmation_lineage": True,
        "second_fresh_bank_claimed": False,
        "independent_rng_bank_claimed": False,
        "failed_attempt_artifacts_reused": False,
        "support_recomputed_from_scratch": True,
    }
    if final != expected:
        raise RuntimeError("v4 confirmation final status differs on read-only recomputation")


def _validate_rng_contract(
    metadata: Mapping[str, Any],
    *,
    science_config: ControlledClinicalExtensionConfig,
    fidelity_config: FidelityV4Config,
) -> Mapping[str, Any]:
    presets = {
        dataset: replace(
            science_config.datasets[dataset],
            seeds=fidelity_config.confirmation_seeds[dataset],
            bootstrap_seed=fidelity_config.confirmation_bootstrap_seeds[dataset],
        )
        for dataset in DATASETS
    }
    mapping = v2._new_rng_stream_mapping(
        replace(science_config, datasets=presets),
        DATASETS,
    )
    v2._assert_unique_rng_streams(mapping)
    audit = metadata.get("confirmation_rng_audit")
    if (
        not isinstance(audit, dict)
        or audit.get("status") != "passed_before_launch"
        or audit.get("role") != "exact_administrative_reuse_of_first_confirmation_bank"
        or audit.get("new_rng_stream_count") != 1_304
        or audit.get("new_rng_stream_mapping") != mapping
        or audit.get("new_rng_stream_mapping_sha256")
        != EXPECTED_CONFIRMATION_MAPPING_SHA256
        or _json_sha256(mapping) != EXPECTED_CONFIRMATION_MAPPING_SHA256
        or audit.get("internal_rng_streams_unique") is not True
        or audit.get("same_prespecified_mapping_reused") is not True
        or audit.get("same_prespecified_mapping_collision_count") != 1_304
        or audit.get("unauthorized_collision_count") != 0
        or audit.get("second_fresh_bank_claimed") is not False
        or audit.get("independent_rng_bank_claimed") is not False
    ):
        raise RuntimeError("v4 confirmation RNG lineage differs")
    return audit


def _validated_dataset_gate(
    dataset: str,
    *,
    gate: Mapping[str, Any],
    metadata: Mapping[str, Any],
    science_config: ControlledClinicalExtensionConfig,
    fidelity_config: FidelityV4Config,
) -> DatasetGate:
    if gate.get("status") != "CONFIRMATION_GATE_GO":
        raise RuntimeError(f"{dataset} is not independently confirmed")
    theta = v4._theta_from_dict(gate["theta"])
    prespecified = fidelity_config.confirmation_seeds[dataset]
    preset = replace(
        science_config.datasets[dataset],
        seeds=prespecified,
        bootstrap_seed=fidelity_config.confirmation_bootstrap_seeds[dataset],
    )
    seed_to_device = {
        seed: metadata["seed_to_device"][f"{dataset}/base_{seed}"]
        for seed in prespecified
    }
    anchors = {}
    records = []
    support_rows = []
    k0_rows = []
    for seed in prespecified:
        support = _validate_confirmation_seed_payload(
            CONFIRMATION_ROOT
            / dataset
            / "support"
            / f"seed_{seed:06d}.json",
            phase="confirmation_support",
            dataset=dataset,
            seed=seed,
            device=seed_to_device[seed],
            source_tree_sha256=metadata["source_tree_sha256"],
            theta=None,
            horizon=preset.horizon,
        )
        k0 = _validate_confirmation_seed_payload(
            CONFIRMATION_ROOT
            / dataset
            / "k0_fidelity"
            / f"seed_{seed:06d}.json",
            phase="confirmation_k0",
            dataset=dataset,
            seed=seed,
            device=seed_to_device[seed],
            source_tree_sha256=metadata["source_tree_sha256"],
            theta=gate["theta"],
            horizon=preset.horizon,
        )
        if (
            support["split_audit"] != k0["split_audit"]
            or k0["theta"] != gate["theta"]
            or k0["metrics"]["structural_invariants"] is not True
        ):
            raise RuntimeError(f"{dataset}/{seed} confirmation anchor differs")
        support_rows.append(support)
        k0_rows.append(k0)
        support_passed = bool(support["passed"])
        k0_passed = bool(k0["passed"])
        eligible = support_passed and k0_passed
        reason = None
        if not support_passed:
            reason = {
                "code": "SUPPORT_FAILED",
                "minimum_unique_patients": support["minimum_unique_patients"],
                "failed_cells": support["failed_cells"],
                "k0_passed_but_does_not_restore_support": k0_passed,
            }
        elif not k0_passed:
            reason = {"code": "K0_FAILED", "metrics": k0["metrics"]}
        records.append(
            {
                "seed": seed,
                "support_passed": support_passed,
                "k0_passed": k0_passed,
                "science_eligible": eligible,
                "exclusion_reason": reason,
            }
        )
        anchors[seed] = ConfirmationSeedAnchor(
            split_audit=k0["split_audit"],
            kernel_identity=k0["context_identity"],
            support_passed=support_passed,
            k0_passed=k0_passed,
        )
    expected_gate = _recomputed_confirmation_gate(
        dataset,
        gate["theta"],
        support_rows,
        k0_rows,
    )
    if gate != expected_gate:
        raise RuntimeError(f"{dataset} confirmation gate differs on recomputation")
    if (
        (CONFIRMATION_ROOT / dataset / "support" / "COMPLETE").read_text()
        != "complete\n"
        or (CONFIRMATION_ROOT / dataset / "k0_fidelity" / "COMPLETE").read_text()
        != "complete\n"
        or (CONFIRMATION_ROOT / dataset / "COMPLETE").read_text()
        != gate["status"].lower() + "\n"
    ):
        raise RuntimeError(f"{dataset} confirmation phase marker differs")
    eligible_seeds = tuple(row["seed"] for row in records if row["science_eligible"])
    if eligible_seeds != EXPECTED_ELIGIBLE_SEEDS[dataset]:
        raise RuntimeError(f"{dataset} science-eligible seed set differs")
    if dataset == "eicu":
        excluded = [row for row in records if not row["science_eligible"]]
        if (
            excluded
            != [
                {
                    "seed": 116_150,
                    "support_passed": False,
                    "k0_passed": True,
                    "science_eligible": False,
                    "exclusion_reason": {
                        "code": "SUPPORT_FAILED",
                        "minimum_unique_patients": 16,
                        "failed_cells": [[0, 3, 16]],
                        "k0_passed_but_does_not_restore_support": True,
                    },
                }
            ]
        ):
            raise RuntimeError("eICU seed 116150 exclusion semantics differ")
    eligibility_record = {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "rule": "support_passed AND k0_passed",
        "prespecified_seed_count": 20,
        "science_eligible_seed_count": len(eligible_seeds),
        "selection_rate_denominator": 20,
        "maximum_possible_selection_rate": len(eligible_seeds) / 20,
        "eligible_seeds": list(eligible_seeds),
        "unavailable_for_every_method": [
            row["seed"] for row in records if not row["science_eligible"]
        ],
        "seed_records": records,
    }
    return DatasetGate(
        preset=preset,
        theta=theta,
        prespecified_seeds=prespecified,
        eligible_seeds=eligible_seeds,
        anchors={seed: anchors[seed] for seed in eligible_seeds},
        eligibility_record=eligibility_record,
        seed_to_device={seed: seed_to_device[seed] for seed in eligible_seeds},
    )


def _validate_confirmation_seed_payload(
    path: Path,
    *,
    phase: str,
    dataset: str,
    seed: int,
    device: str,
    source_tree_sha256: str,
    theta: Mapping[str, Any] | None,
    horizon: int,
) -> dict[str, Any]:
    payload = _read_json(path)
    candidate_hash = _json_sha256([] if theta is None else [dict(theta)])
    expected_header = {
        "protocol": v4.PROTOCOL,
        "phase": phase,
        "dataset": dataset,
        "seed": seed,
        "device": device,
        "source_tree_sha256": source_tree_sha256,
        "candidate_contract_sha256": candidate_hash,
    }
    _require_exact_keys(payload, {*expected_header, "result"}, "confirmation seed")
    if any(payload.get(key) != value for key, value in expected_header.items()):
        raise RuntimeError(f"{dataset}/{seed} confirmation seed header differs")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"{dataset}/{seed} confirmation result is malformed")
    if phase == "confirmation_support":
        _validate_support_result(result, dataset=dataset, seed=seed, horizon=horizon)
    elif phase == "confirmation_k0" and theta is not None:
        _validate_k0_result(
            result,
            dataset=dataset,
            seed=seed,
            theta=theta,
            horizon=horizon,
        )
    else:
        raise RuntimeError("unknown or malformed confirmation phase")
    return result


def _validate_support_result(
    result: Mapping[str, Any],
    *,
    dataset: str,
    seed: int,
    horizon: int,
) -> None:
    _require_exact_keys(
        result,
        {
            "seed",
            "dataset",
            "phase",
            "passed",
            "n_actions",
            "active_action_indices",
            "action_mapping",
            "action_costs",
            "minimum_unique_patients",
            "unique_patient_counts_by_stage_action",
            "failed_cells",
            "outcome_blind",
            "environment_episode_support",
            "split_audit",
            "coverage_generated",
            "confirmation_label",
        },
        "confirmation support result",
    )
    counts = result["unique_patient_counts_by_stage_action"]
    n_actions = result["n_actions"]
    if (
        result["seed"] != seed
        or result["dataset"] != dataset
        or result["phase"] != "confirmation_support"
        or result["outcome_blind"] is not True
        or result["coverage_generated"] is not False
        or result["confirmation_label"] != "fresh_split_operational_gate"
        or not isinstance(n_actions, int)
        or n_actions < 1
        or not isinstance(counts, list)
        or len(counts) != horizon
        or any(
            not isinstance(row, list)
            or len(row) != n_actions
            or any(not isinstance(value, int) or value < 0 for value in row)
            for row in counts
        )
    ):
        raise RuntimeError(f"{dataset}/{seed} support result differs")
    minimum = min(value for row in counts for value in row)
    failed = [
        [stage, action, value]
        for stage, row in enumerate(counts)
        for action, value in enumerate(row)
        if value < 20
    ]
    if (
        result["minimum_unique_patients"] != minimum
        or result["failed_cells"] != failed
        or result["passed"] is not (not failed)
    ):
        raise RuntimeError(f"{dataset}/{seed} support decision differs")
    _validate_split_audit(result["split_audit"])


def _validate_k0_result(
    result: Mapping[str, Any],
    *,
    dataset: str,
    seed: int,
    theta: Mapping[str, Any],
    horizon: int,
) -> None:
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
    metrics = result["metrics"]
    _require_exact_keys(
        metrics,
        {
            "maximum_score_ks",
            "maximum_signed_residual_w1",
            "maximum_successor_mean_w1",
            "maximum_successor_q95_w1",
            "structural_invariants",
        },
        "confirmation K0 metrics",
    )
    numeric = (
        float(metrics["maximum_score_ks"]),
        float(metrics["maximum_signed_residual_w1"]),
        float(metrics["maximum_successor_mean_w1"]),
        float(metrics["maximum_successor_q95_w1"]),
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in numeric):
        raise RuntimeError(f"{dataset}/{seed} K0 metrics are invalid")
    ratios = (
        numeric[0] / 0.10,
        numeric[1] / 0.25,
        numeric[2] / 0.25,
        numeric[3] / 0.50,
    )
    structural = metrics["structural_invariants"] is True
    passed = structural and max(ratios) <= 1.0
    context = result["context_identity"]
    if not isinstance(context, dict):
        raise RuntimeError(f"{dataset}/{seed} K0 context identity is malformed")
    context_without_hash = {
        key: value for key, value in context.items() if key != "combined_sha256"
    }
    library = context.get("library_support")
    if not isinstance(library, dict):
        raise RuntimeError(f"{dataset}/{seed} K0 library identity is malformed")
    library_without_hash = {
        key: value for key, value in library.items() if key != "combined_sha256"
    }
    replay = result["systematic_replay"]
    if (
        result["seed"] != seed
        or result["dataset"] != dataset
        or result["phase"] != "confirmation_k0"
        or result["theta"] != dict(theta)
        or result["passed"] is not passed
        or result["normalized_seed_ratio"] != max(ratios)
        or result["structural_failure_ratio_is_infinite"] is not (not structural)
        or result["coverage_generated"] is not False
        or result["confirmation_label"] != "fresh_split_operational_gate"
        or result["independent_patient_confirmation_claimed"] is not False
        or context.get("theta") != dict(theta)
        or context.get("combined_sha256") != _json_sha256(context_without_hash)
        or library.get("combined_sha256") != _json_sha256(library_without_hash)
        or not isinstance(replay, dict)
        or replay.get("systematic_replays") != 16
        or replay.get("base_uniform_seed") != 90_000_000 + seed
        or replay.get("base_uniform_shape", [None])[0] != horizon
    ):
        raise RuntimeError(f"{dataset}/{seed} K0 decision/provenance differs")
    _validate_split_audit(result["split_audit"])


def _validate_split_audit(value: object) -> None:
    _require_exact_keys(
        value,
        {
            "role_unique_patient_counts",
            "role_episode_counts",
            "role_patient_id_sha256",
            "patient_sets_pairwise_disjoint",
            "split_fractions",
        },
        "confirmation split audit",
    )
    assert isinstance(value, Mapping)
    roles = {"predictor", "fidelity", "environment"}
    if (
        set(value["role_unique_patient_counts"]) != roles
        or set(value["role_episode_counts"]) != roles
        or set(value["role_patient_id_sha256"]) != roles
        or value["patient_sets_pairwise_disjoint"] is not True
        or value["split_fractions"] != [0.4, 0.2, 0.4]
        or any(
            not isinstance(count, int) or count < 1
            for mapping_name in ("role_unique_patient_counts", "role_episode_counts")
            for count in value[mapping_name].values()
        )
        or any(
            not isinstance(digest, str) or len(digest) != 64
            for digest in value["role_patient_id_sha256"].values()
        )
    ):
        raise RuntimeError("confirmation split audit differs")


def _recomputed_confirmation_gate(
    dataset: str,
    theta: Mapping[str, Any],
    support_rows: Sequence[Mapping[str, Any]],
    k0_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(support_rows) != 20 or len(k0_rows) != 20:
        raise RuntimeError(f"{dataset} confirmation bank is incomplete")
    support_count = sum(bool(row["passed"]) for row in support_rows)
    structural_count = sum(
        bool(row["metrics"]["structural_invariants"]) for row in k0_rows
    )
    k0_count = sum(bool(row["passed"]) for row in k0_rows)
    status = (
        "CONFIRMATION_GATE_GO"
        if support_count >= 19 and structural_count == 20 and k0_count >= 19
        else "CONFIRMATION_GATE_NO_GO"
    )
    return {
        "protocol": v4.PROTOCOL,
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


def _validate_unopened_cxr(
    gate: Mapping[str, Any],
    final: Mapping[str, Any],
) -> None:
    root = CONFIRMATION_ROOT / "mimic_cxr"
    expected = {
        "protocol": v4.PROTOCOL,
        "dataset": "mimic_cxr",
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
    if (
        gate != expected
        or (root / "support").exists()
        or (root / "k0_fidelity").exists()
        or (root / "COMPLETE").read_text() != gate["status"].lower() + "\n"
        or final["datasets"]["mimic_cxr"] != gate
    ):
        raise RuntimeError("MIMIC-CXR must remain unopened and absent from science")


def run_post_confirmation_science(
    output_root: Path,
    *,
    gates: GateBundle,
    devices: tuple[str, ...],
    independent_audit_go_sha256: str,
    resume: bool,
) -> None:
    if output_root.resolve() != OUTPUT_ROOT:
        raise RuntimeError(f"science output root is frozen to {OUTPUT_ROOT}")
    gate_hash = _json_sha256(gates.contract)
    if independent_audit_go_sha256 != gate_hash:
        raise RuntimeError("independent audit GO does not match the active contract")
    source_hash, source_snapshot = _active_source_snapshot()
    if source_hash != gates.active_source_tree_sha256:
        raise RuntimeError("science source changed after the read-only audit")
    metadata = _science_metadata(
        gates,
        devices=devices,
        independent_audit_go_sha256=independent_audit_go_sha256,
        source_snapshot=source_snapshot["contract"],
    )
    _prepare_root(output_root, metadata, source_snapshot, resume=resume)
    try:
        _require_partial_artifact_subset(output_root, metadata, gates)
        if (output_root / "COMPLETE").exists():
            _validate_complete_root(output_root, metadata, gates)
            return
        _publish_eligibility_records(output_root, gates)
        if (output_root / SCIENCE_PHASE).exists() and not _valid_global_overlap_marker(
            output_root, gates
        ):
            raise RuntimeError("science exists before the global overlap commit")

        interpretations = {}
        for dataset in CONFIRMED_DATASETS:
            dataset_gate = gates.datasets[dataset]
            phase_preset = replace(
                dataset_gate.preset,
                seeds=dataset_gate.eligible_seeds,
            )
            rows = _run_phase(
                output_root / OVERLAP_PHASE / dataset,
                phase=OVERLAP_PHASE,
                preset=phase_preset,
                theta=dataset_gate.theta,
                anchors=dataset_gate.anchors,
                seed_to_device=dataset_gate.seed_to_device,
                devices=devices,
                source_hash=source_hash,
                gate_contract_sha256=gate_hash,
                rng_mapping_sha256=gates.rng_audit[
                    "new_rng_stream_mapping_sha256"
                ],
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
            _write_or_verify_json(
                output_root / OVERLAP_PHASE / dataset / "summary.json",
                _overlap_summary(dataset_gate, rows, interpretation),
            )

        _write_global_overlap_marker(output_root, gates, interpretations)
        if not _valid_global_overlap_marker(output_root, gates):
            raise RuntimeError("global overlap commit failed validation")

        statuses = {}
        for dataset in CONFIRMED_DATASETS:
            dataset_gate = gates.datasets[dataset]
            phase_preset = replace(
                dataset_gate.preset,
                seeds=dataset_gate.eligible_seeds,
            )
            seed_results = _run_phase(
                output_root / SCIENCE_PHASE / dataset / "seeds",
                phase=SCIENCE_PHASE,
                preset=phase_preset,
                theta=dataset_gate.theta,
                anchors=dataset_gate.anchors,
                seed_to_device=dataset_gate.seed_to_device,
                devices=devices,
                source_hash=source_hash,
                gate_contract_sha256=gate_hash,
                rng_mapping_sha256=gates.rng_audit[
                    "new_rng_stream_mapping_sha256"
                ],
                worker=_science_worker,
                worker_arguments=(gates.science_config, interpretations[dataset]),
                resume=resume,
            )
            rows = [row for result in seed_results for row in result["rows"]]
            science_root = output_root / SCIENCE_PHASE / dataset
            bootstrap = _ensure_bootstrap_artifacts(
                science_root,
                dataset_gate.preset,
            )
            summary = _science_summary(
                rows,
                dataset_gate=dataset_gate,
                interpretation_status=interpretations[dataset],
                bootstrap_contract=bootstrap,
            )
            audit = _coverage_audit(
                rows,
                dataset_gate=dataset_gate,
                summary=summary,
                interpretation_status=interpretations[dataset],
            )
            status = _dataset_science_status(
                dataset_gate,
                interpretations[dataset],
            )
            _write_or_verify_json(science_root / "summary.json", summary)
            _write_or_verify_json(science_root / "coverage_audit.json", audit)
            _write_or_verify_json(science_root / "FINAL_STATUS.json", status)
            _write_or_verify_text(
                science_root / "COMPLETE",
                (
                    "curves\n"
                    if interpretations[dataset]
                    == "EMPIRICAL_OVERLAP_SCREEN_PASSED"
                    else "curves-descriptive-only\n"
                ),
            )
            statuses[dataset] = status

        _write_or_verify_json(
            output_root / "FINAL_STATUS.json",
            _final_science_status(statuses),
        )
        _finalize_root(output_root, metadata, gates)
    except BaseException:
        _unlink_root_complete(output_root)
        raise


def _publish_eligibility_records(root: Path, gates: GateBundle) -> None:
    for dataset in CONFIRMED_DATASETS:
        _write_or_verify_json(
            root / "eligibility" / f"{dataset}.json",
            gates.datasets[dataset].eligibility_record,
        )


def _overlap_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    theta: Theta,
    anchor: ConfirmationSeedAnchor,
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    context = v4._context_with_theta(base_context, theta)
    kernel_identity = v4._candidate_context_identity(
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
        "theta": v4._theta_to_dict(theta),
        "confirmation_anchor_identity_sha256": _json_sha256(
            anchor.kernel_identity
        ),
    }


def _science_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    theta: Theta,
    anchor: ConfirmationSeedAnchor,
    protocol: ControlledClinicalExtensionConfig,
    interpretation_status: str,
) -> dict[str, Any]:
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    context = v4._context_with_theta(base_context, theta)
    kernel_identity = v4._candidate_context_identity(
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
        "theta": v4._theta_to_dict(theta),
        "confirmation_anchor_identity_sha256": _json_sha256(
            anchor.kernel_identity
        ),
    }


def _assert_confirmation_context(
    anchor: ConfirmationSeedAnchor,
    base_context: v2.ExtensionContext,
    kernel_identity: Mapping[str, Any],
) -> None:
    if (
        not anchor.support_passed
        or not anchor.k0_passed
        or v2._split_audit(base_context.splits) != anchor.split_audit
        or kernel_identity != anchor.kernel_identity
    ):
        raise RuntimeError("reconstructed context differs from an eligible v4 anchor")


def _run_phase(
    root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    theta: Theta,
    anchors: Mapping[int, ConfirmationSeedAnchor],
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
        raise RuntimeError(f"unexpected {phase} artifacts: {sorted(unexpected)}")
    if not resume and observed_paths:
        raise RuntimeError(f"fresh {phase} directory already contains artifacts")

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
    if pending and (root / "COMPLETE").exists():
        raise RuntimeError(f"{phase} COMPLETE exists with missing seeds")
    if pending and phase == OVERLAP_PHASE and (root / "summary.json").exists():
        raise RuntimeError("overlap summary exists with missing seeds")
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
                        "theta_sha256": _json_sha256(v4._theta_to_dict(theta)),
                        "eligibility_anchor_sha256": _json_sha256(
                            {
                                "split_audit": anchors[seed].split_audit,
                                "kernel_identity": anchors[seed].kernel_identity,
                                "support_passed": anchors[seed].support_passed,
                                "k0_passed": anchors[seed].k0_passed,
                            }
                        ),
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
        raise RuntimeError(f"{phase} did not complete its exact eligible seed bank")
    if (root / "COMPLETE").exists() and (root / "COMPLETE").read_text() != "complete\n":
        raise RuntimeError(f"{phase} COMPLETE marker differs")
    _write_or_verify_text(root / "COMPLETE", "complete\n")
    return [completed[seed] for seed in preset.seeds]


def _phase_group(
    seeds: tuple[int, ...],
    device: str,
    preset: DatasetPreset,
    theta: Theta,
    anchors: Mapping[int, ConfirmationSeedAnchor],
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
    theta: Theta,
    anchor: ConfirmationSeedAnchor,
    source_hash: str,
    gate_contract_sha256: str,
    rng_mapping_sha256: str,
) -> None:
    anchor_payload = {
        "split_audit": anchor.split_audit,
        "kernel_identity": anchor.kernel_identity,
        "support_passed": anchor.support_passed,
        "k0_passed": anchor.k0_passed,
    }
    expected = {
        "protocol": PROTOCOL,
        "phase": phase,
        "dataset": preset.name,
        "seed": seed,
        "device": device,
        "source_tree_sha256": source_hash,
        "gate_contract_sha256": gate_contract_sha256,
        "rng_stream_mapping_sha256": rng_mapping_sha256,
        "theta_sha256": _json_sha256(v4._theta_to_dict(theta)),
        "eligibility_anchor_sha256": _json_sha256(anchor_payload),
    }
    _require_exact_keys(payload, {*expected, "result"}, f"{phase} wrapper")
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{phase} provenance differs for seed {seed}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"{phase} result is malformed")
    if (
        not anchor.support_passed
        or not anchor.k0_passed
        or result.get("seed") != seed
        or result.get("dataset") != preset.name
        or result.get("phase") != phase
        or result.get("theta") != v4._theta_to_dict(theta)
        or result.get("kernel_context_identity") != anchor.kernel_identity
        or result.get("split_audit") != anchor.split_audit
        or result.get("confirmation_anchor_identity_sha256")
        != _json_sha256(anchor.kernel_identity)
    ):
        raise RuntimeError(f"{phase} identity differs for seed {seed}")
    if phase == OVERLAP_PHASE:
        v3science._validate_overlap_result(result, preset)
    elif phase == SCIENCE_PHASE:
        v3science._validate_science_result(result, preset)
    else:
        raise RuntimeError(f"unknown phase: {phase}")


def _overlap_summary(
    dataset_gate: DatasetGate,
    rows: Sequence[Mapping[str, Any]],
    interpretation: str,
) -> dict[str, Any]:
    passed = [int(row["seed"]) for row in rows if bool(row["passed"])]
    return {
        "protocol": PROTOCOL,
        "dataset": dataset_gate.preset.name,
        "gate": "gamma=-4 q_mid+q_high empirical donor-overlap screen",
        "prespecified_seed_count": 20,
        "eligible_seed_count": len(dataset_gate.eligible_seeds),
        "eligible_seeds": list(dataset_gate.eligible_seeds),
        "unavailable_for_every_method": dataset_gate.eligibility_record[
            "unavailable_for_every_method"
        ],
        "passed_seed_count": len(passed),
        "passed_seeds": passed,
        "interpretation_status": interpretation,
        "hard_structural_failure": False,
        "failure_consequence": (
            "this dataset remains descriptive-only; no gamma=-4 ranking, attainment, "
            "superiority, or cross-dataset claim"
        ),
    }


def _write_global_overlap_marker(
    root: Path,
    gates: GateBundle,
    interpretations: Mapping[str, str],
) -> None:
    if tuple(interpretations) != CONFIRMED_DATASETS:
        raise RuntimeError("global overlap commit requires all confirmed datasets")
    summary = {
        "protocol": PROTOCOL,
        "status": "GLOBAL_OVERLAP_COMPLETE_BEFORE_COVERAGE",
        "datasets": dict(interpretations),
        "eligible_seed_counts": {
            dataset: len(gates.datasets[dataset].eligible_seeds)
            for dataset in CONFIRMED_DATASETS
        },
        "all_confirmed_dataset_overlap_banks_complete": True,
        "science_may_start": True,
        "cross_dataset_conjunction_used": False,
        "low_overlap_consequence": "per_dataset_descriptive_only",
    }
    _write_or_verify_json(root / OVERLAP_PHASE / "summary.json", summary)
    _write_or_verify_text(
        root / OVERLAP_PHASE / "COMPLETE",
        f"global-overlap-complete summary_sha256={_json_sha256(summary)}\n",
    )


def _valid_global_overlap_marker(root: Path, gates: GateBundle) -> bool:
    summary_path = root / OVERLAP_PHASE / "summary.json"
    complete_path = root / OVERLAP_PHASE / "COMPLETE"
    if not summary_path.is_file() or not complete_path.is_file():
        return False
    try:
        summary = _read_json(summary_path)
    except RuntimeError:
        return False
    interpretations = summary.get("datasets")
    expected_counts = {
        dataset: len(gates.datasets[dataset].eligible_seeds)
        for dataset in CONFIRMED_DATASETS
    }
    return (
        set(summary)
        == {
            "protocol",
            "status",
            "datasets",
            "eligible_seed_counts",
            "all_confirmed_dataset_overlap_banks_complete",
            "science_may_start",
            "cross_dataset_conjunction_used",
            "low_overlap_consequence",
        }
        and summary["protocol"] == PROTOCOL
        and summary["status"] == "GLOBAL_OVERLAP_COMPLETE_BEFORE_COVERAGE"
        and isinstance(interpretations, dict)
        and set(interpretations) == set(CONFIRMED_DATASETS)
        and all(
            value
            in {
                "EMPIRICAL_OVERLAP_SCREEN_PASSED",
                "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
            }
            for value in interpretations.values()
        )
        and summary["eligible_seed_counts"] == expected_counts
        and summary["all_confirmed_dataset_overlap_banks_complete"] is True
        and summary["science_may_start"] is True
        and summary["cross_dataset_conjunction_used"] is False
        and summary["low_overlap_consequence"] == "per_dataset_descriptive_only"
        and complete_path.read_text()
        == f"global-overlap-complete summary_sha256={_json_sha256(summary)}\n"
    )


def _ensure_bootstrap_artifacts(
    root: Path,
    preset: DatasetPreset,
    *,
    create_if_missing: bool = True,
) -> dict[str, Any]:
    uniform_path = root / "bootstrap_uniforms.npy"
    index_path = root / "bootstrap_indices.npy"
    rng = np.random.default_rng(preset.bootstrap_seed)
    expected_uniforms = rng.random((BOOTSTRAP_RESAMPLES, 20), dtype=np.float64)
    expected_indices = np.floor(expected_uniforms * 20).astype(np.int16)
    if not uniform_path.exists() and not index_path.exists():
        if not create_if_missing:
            raise RuntimeError("bootstrap artifacts are missing")
        _write_npy(uniform_path, expected_uniforms)
        _write_npy(index_path, expected_indices)
    if not uniform_path.is_file() or not index_path.is_file():
        raise RuntimeError("bootstrap artifact pair is incomplete")
    uniforms = np.load(uniform_path, allow_pickle=False)
    indices = np.load(index_path, allow_pickle=False)
    if not np.array_equal(uniforms, expected_uniforms) or not np.array_equal(
        indices, expected_indices
    ):
        raise RuntimeError("bootstrap artifacts differ from the frozen 20-seed bank")
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "root_seed": preset.bootstrap_seed,
        "prespecified_seed_count": 20,
        "uniform_matrix_shape": [BOOTSTRAP_RESAMPLES, 20],
        "uniform_matrix_path": uniform_path.name,
        "uniform_matrix_sha256": _file_sha256(uniform_path),
        "complete_seed_index_matrix_shape": [BOOTSTRAP_RESAMPLES, 20],
        "complete_seed_index_matrix_path": index_path.name,
        "complete_seed_index_matrix_sha256": _file_sha256(index_path),
        "unit": "complete_seed_stage_vector",
        "shared_across": ["methods", "gammas", "stages"],
        "selected_subset_rule": (
            "for selected-set size n, use floor(U[:, :n] * n); eICU keeps the "
            "20-column prespecified bank and projects to eligible/selected subsets"
        ),
    }


def _science_summary(
    rows: list[dict[str, Any]],
    *,
    dataset_gate: DatasetGate,
    interpretation_status: str,
    bootstrap_contract: dict[str, Any],
) -> dict[str, Any]:
    preset = dataset_gate.preset
    selected_seeds = dataset_gate.eligible_seeds
    summary = v2.summarize_science(
        rows,
        preset=preset,
        selected_seeds=selected_seeds,
        interpretation_status=interpretation_status,
        bootstrap_contract=bootstrap_contract,
    )
    summary.update(
        {
            "protocol": PROTOCOL,
            "source_summary_semantics": v2.PROTOCOL,
            "role": "v4_confirmed_dataset_specific_theta_signed_gamma_science",
            "seeds_prespecified": list(dataset_gate.prespecified_seeds),
            "seeds_support_and_k0_eligible": list(selected_seeds),
            "unavailable_for_every_method": dataset_gate.eligibility_record[
                "unavailable_for_every_method"
            ],
            "eligibility_record_sha256": _json_sha256(
                dataset_gate.eligibility_record
            ),
            "complete_confirmation_seed_bank_used_as_denominator": True,
            "selection_rate_denominator": "all 20 prespecified seeds",
            "coverage_conditioning": (
                "successful method selection among support-and-K0-eligible seeds"
            ),
            "primary_default_gamma": PRIMARY_GAMMA,
            "target_coverage": TARGET_COVERAGE,
            "point_eligibility_is_not_interval_attainment": True,
            "inference_contract": SCIENCE_CONTRACT["uncertainty"],
            "cross_dataset_pooling_or_claim": False,
        }
    )
    if summary.get("seeds_k0_eligible") != list(selected_seeds):
        raise RuntimeError("base summary changed the frozen eligible seed set")
    summary["seeds_k0_eligible"] = list(selected_seeds)

    aggregates = summary.get("aggregates")
    if not isinstance(aggregates, list) or len(aggregates) != len(GAMMAS):
        raise RuntimeError("science summary lacks the five signed-gamma aggregates")
    for aggregate, gamma in zip(aggregates, GAMMAS, strict=True):
        if float(aggregate.get("gamma")) != gamma:
            raise RuntimeError("science summary gamma order differs")
        confirmatory = (
            gamma == PRIMARY_GAMMA
            and interpretation_status == "EMPIRICAL_OVERLAP_SCREEN_PASSED"
        )
        selected = sorted(
            (row for row in rows if float(row["gamma"]) == gamma),
            key=lambda row: int(row["seed"]),
        )
        if tuple(int(row["seed"]) for row in selected) != selected_seeds:
            raise RuntimeError(f"summary seed set differs for gamma={gamma}")
        for method in METHODS:
            method_summary = aggregate["methods"][method]
            available = [
                row
                for row in selected
                if row["methods"][method]["selection_available"]
            ]
            count = len(available)
            if (
                method_summary["n_selected"] != count
                or method_summary["n_prespecified"] != 20
                or method_summary["n_k0_eligible"] != len(selected_seeds)
                or method_summary["selection_rate"] != count / 20
                or method_summary["selection_rate_ci95"]
                != _wilson_interval(count, 20)
            ):
                raise RuntimeError(f"{method} selection denominator differs")
            method_summary.update(
                {
                    "selection_rate_denominator": 20,
                    "support_and_k0_eligible_seed_count": len(selected_seeds),
                    "unavailable_before_method_selection": 20 - len(selected_seeds),
                    "target_coverage_by_stage_notation": "C_t",
                    "target_mean_coverage_notation": "MeanCov",
                    "target_marginal_worst_coverage_notation": (
                        "WSC=min_t mean_seed(C_seed,t)"
                    ),
                    "stage_coverage_ci_method": "two-sided Student-t",
                    "mean_coverage_ci_method": "two-sided Student-t",
                    "stage_normalized_width_ci_method": "two-sided Student-t",
                    "mean_normalized_width_ci_method": "two-sided Student-t",
                    "wsc_ci_method": (
                        "10000-draw complete-seed-stage-vector percentile bootstrap"
                    ),
                    "information_budget_per_seed": {
                        "source_calibration_trajectories": 3_000,
                        "source_grid_trajectories": 1_000,
                        "target_reference_trajectories": 20_000,
                        "target_adaptation_trajectories": (
                            TARGET_ADAPTATION_BUDGET[method]
                        ),
                    },
                }
            )
            if count == 0:
                method_summary.update(
                    {
                        "target_wsc_gap_to_0.90": None,
                        "target_mean_coverage_gap_to_0.90": None,
                        "point_attainment_at_0.90": None,
                        "wsc_interval_attainment_at_0.90": None,
                        "interval_attainment_rule": (
                            "lower endpoint of target_wsc_ci95 >= 0.90"
                        ),
                    }
                )
                continue
            coverage = np.asarray(
                [row["methods"][method]["target_coverage"] for row in available],
                dtype=np.float64,
            )
            width = np.asarray(
                [
                    row["methods"][method]["target_normalized_width"]
                    for row in available
                ],
                dtype=np.float64,
            )
            stage_coverage = coverage.mean(axis=0)
            stage_width = width.mean(axis=0)
            wsc = float(stage_coverage.min())
            mean_coverage = float(stage_coverage.mean())
            if (
                method_summary["target_coverage_by_stage"]
                != stage_coverage.tolist()
                or method_summary["target_marginal_worst_coverage"] != wsc
                or method_summary["target_worst_stage_zero_based"]
                != int(stage_coverage.argmin())
                or method_summary["target_mean_coverage"] != mean_coverage
                or method_summary["target_normalized_width_by_stage"]
                != stage_width.tolist()
                or method_summary["mean_target_normalized_width"]
                != float(stage_width.mean())
            ):
                raise RuntimeError(f"{method} point estimates differ")
            method_summary["target_coverage_by_stage_ci95"] = (
                _student_t_interval_by_stage(coverage)
            )
            method_summary["target_normalized_width_by_stage_ci95"] = (
                _student_t_interval_by_stage(width)
            )
            wsc_ci = method_summary["target_wsc_ci95"]
            point_attainment = bool(wsc >= TARGET_COVERAGE) if confirmatory else None
            interval_attainment = (
                bool(float(wsc_ci[0]) >= TARGET_COVERAGE)
                if confirmatory
                else None
            )
            method_summary.update(
                {
                    "target_wsc_gap_to_0.90": wsc - TARGET_COVERAGE,
                    "target_mean_coverage_gap_to_0.90": (
                        mean_coverage - TARGET_COVERAGE
                    ),
                    "point_attainment_at_0.90": point_attainment,
                    "wsc_interval_attainment_at_0.90": interval_attainment,
                    "interval_attainment_rule": (
                        "lower endpoint of target_wsc_ci95 >= 0.90"
                    ),
                }
            )
            if confirmatory:
                if (
                    method_summary["confirmatory_attainment_at_0.90"]
                    != point_attainment
                    or method_summary["point_eligible"]
                    != bool(count / 20 >= 0.95 and point_attainment)
                ):
                    raise RuntimeError(f"{method} primary point eligibility differs")
            elif (
                method_summary["confirmatory_attainment_at_0.90"] is not None
                or method_summary["point_eligible"] is not None
            ):
                raise RuntimeError("nonprimary gamma contains confirmatory eligibility")
        if confirmatory:
            if (
                aggregate["analysis_role"]
                != "confirmatory_gamma_minus_4_endpoint"
                or not isinstance(aggregate["paired_scpcp_comparisons"], dict)
            ):
                raise RuntimeError("primary gamma lacks paired/efficiency analysis")
        elif (
            aggregate["analysis_role"] != "descriptive_signed_control_curve"
            or aggregate["width_order_among_point_eligible"] != []
            or set(aggregate["paired_scpcp_comparisons"])
            != {"status"}
        ):
            raise RuntimeError("descriptive gamma contains ranking or paired analysis")
        aggregate.update(
            {
                "target_coverage": TARGET_COVERAGE,
                "ranking_scope": (
                    "within_dataset_point_eligible_methods"
                    if confirmatory
                    else "none_descriptive_only"
                ),
                "cross_dataset_pooling_or_claim": False,
            }
        )
    return summary


def _student_t_interval_by_stage(values: np.ndarray) -> list[list[float]]:
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("Student-t stage intervals require a nonempty seed-stage matrix")
    return [
        _student_t_interval(values[:, stage])
        for stage in range(values.shape[1])
    ]


def _coverage_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_gate: DatasetGate,
    summary: Mapping[str, Any],
    interpretation_status: str,
) -> dict[str, Any]:
    if len(rows) != len(dataset_gate.eligible_seeds) * len(GAMMAS):
        raise RuntimeError("coverage audit requires every eligible seed/gamma row")
    aggregates = {float(row["gamma"]): row for row in summary["aggregates"]}
    if tuple(aggregates) != tuple(GAMMAS):
        raise RuntimeError("coverage audit gamma set differs")
    records = []
    for gamma in GAMMAS:
        selected = sorted(
            (row for row in rows if float(row["gamma"]) == gamma),
            key=lambda row: int(row["seed"]),
        )
        if tuple(int(row["seed"]) for row in selected) != dataset_gate.eligible_seeds:
            raise RuntimeError(f"coverage audit seeds differ for gamma={gamma}")
        for method in METHODS:
            available = [
                row
                for row in selected
                if row["methods"][method]["selection_available"]
            ]
            method_summary = aggregates[gamma]["methods"][method]
            if available:
                coverage = np.asarray(
                    [row["methods"][method]["target_coverage"] for row in available],
                    dtype=np.float64,
                )
                width = np.asarray(
                    [
                        row["methods"][method]["target_normalized_width"]
                        for row in available
                    ],
                    dtype=np.float64,
                )
                stage = coverage.mean(axis=0)
                computed = {
                    "stage_C_t": stage.tolist(),
                    "MeanCov": float(stage.mean()),
                    "WSC": float(stage.min()),
                    "worst_stage_zero_based": int(stage.argmin()),
                    "mean_normalized_width": float(width.mean(axis=0).mean()),
                    "WSC_minus_0.90": float(stage.min() - TARGET_COVERAGE),
                    "MeanCov_minus_0.90": float(stage.mean() - TARGET_COVERAGE),
                }
                reported = {
                    "stage_C_t": method_summary["target_coverage_by_stage"],
                    "MeanCov": method_summary["target_mean_coverage"],
                    "WSC": method_summary["target_marginal_worst_coverage"],
                    "worst_stage_zero_based": method_summary[
                        "target_worst_stage_zero_based"
                    ],
                    "mean_normalized_width": method_summary[
                        "mean_target_normalized_width"
                    ],
                    "WSC_minus_0.90": method_summary["target_wsc_gap_to_0.90"],
                    "MeanCov_minus_0.90": method_summary[
                        "target_mean_coverage_gap_to_0.90"
                    ],
                }
                if computed != reported:
                    raise RuntimeError(
                        f"{dataset_gate.preset.name}/{gamma}/{method} metric audit differs"
                    )
            else:
                computed = None
            records.append(
                {
                    "gamma": gamma,
                    "method": method,
                    "prespecified_seed_denominator": 20,
                    "support_and_k0_eligible_seed_count": len(
                        dataset_gate.eligible_seeds
                    ),
                    "method_selected_seed_count": len(available),
                    "selection_rate": len(available) / 20,
                    "selection_rate_ci95": _wilson_interval(len(available), 20),
                    "metrics": computed,
                }
            )
    return {
        "protocol": PROTOCOL,
        "dataset": dataset_gate.preset.name,
        "status": "COVERAGE_AUDIT_COMPLETE",
        "primary_metric": PRIMARY_METRIC,
        "target_coverage": TARGET_COVERAGE,
        "formula_verified": True,
        "all_six_methods_present": True,
        "all_five_signed_gammas_present": True,
        "all_eligible_seeds_present": True,
        "prespecified_seed_denominator": 20,
        "eligible_seed_count": len(dataset_gate.eligible_seeds),
        "unavailable_for_every_method": dataset_gate.eligibility_record[
            "unavailable_for_every_method"
        ],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "pointwise_mean_and_width_intervals": "two-sided Student-t",
        "wsc_and_paired_intervals": "10000-draw seed-vector bootstrap",
        "point_eligibility_is_not_interval_attainment": True,
        "interpretation_status": interpretation_status,
        "records": records,
    }


def _dataset_science_status(
    dataset_gate: DatasetGate,
    interpretation: str,
) -> dict[str, Any]:
    if interpretation not in {
        "EMPIRICAL_OVERLAP_SCREEN_PASSED",
        "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
    }:
        raise RuntimeError("unknown per-dataset overlap interpretation")
    return {
        "protocol": PROTOCOL,
        "dataset": dataset_gate.preset.name,
        "status": "COMPLETE",
        "interpretation_status": interpretation,
        "prespecified_seed_count": 20,
        "support_and_k0_eligible_seed_count": len(dataset_gate.eligible_seeds),
        "unavailable_for_every_method": dataset_gate.eligibility_record[
            "unavailable_for_every_method"
        ],
        "raw_signed_gamma_rows": len(dataset_gate.eligible_seeds) * len(GAMMAS),
        "methods": list(METHODS),
        "primary_default_gamma": PRIMARY_GAMMA,
        "primary_metric": PRIMARY_METRIC,
        "ranking_permitted_only_at_gamma_minus_4": (
            interpretation == "EMPIRICAL_OVERLAP_SCREEN_PASSED"
        ),
        "cross_dataset_pooling_or_claim": False,
    }


def _final_science_status(
    dataset_status: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(dataset_status) != CONFIRMED_DATASETS:
        raise RuntimeError("final status requires the three confirmed datasets in order")
    return {
        "protocol": PROTOCOL,
        "status": "COMPLETE_DATASET_INDEPENDENT",
        "datasets": dict(dataset_status),
        "confirmed_datasets": list(CONFIRMED_DATASETS),
        "unopened_datasets": {
            "mimic_cxr": "CONFIRMATION_NOT_OPENED_DEVELOPMENT_NO_GO"
        },
        "methods": list(METHODS),
        "gammas": list(GAMMAS),
        "primary_default_gamma": PRIMARY_GAMMA,
        "target_coverage": TARGET_COVERAGE,
        "primary_metric": PRIMARY_METRIC,
        "prespecified_seeds_per_dataset": 20,
        "eligible_seed_counts": {
            dataset: dataset_status[dataset][
                "support_and_k0_eligible_seed_count"
            ]
            for dataset in CONFIRMED_DATASETS
        },
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "low_overlap_datasets": [
            dataset
            for dataset in CONFIRMED_DATASETS
            if dataset_status[dataset]["interpretation_status"]
            == "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        ],
        "cross_dataset_conjunction_used": False,
        "pooled_or_universal_ranking_defined": False,
    }


def _science_metadata(
    gates: GateBundle,
    *,
    devices: tuple[str, ...],
    independent_audit_go_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = {
        "protocol": PROTOCOL,
        "role": "strict_post_v4_confirmation_signed_gamma_science",
        "output_root": str(OUTPUT_ROOT),
        "source_tree_sha256": gates.active_source_tree_sha256,
        "source_snapshot": dict(source_snapshot),
        "devices": list(devices),
        "independent_audit_status": "GO",
        "independent_audit_go_sha256": independent_audit_go_sha256,
        "gate_contract": gates.contract,
        "gate_contract_sha256": _json_sha256(gates.contract),
        "confirmation_binding": gates.confirmation_binding,
        "confirmation_binding_sha256": _json_sha256(
            gates.confirmation_binding
        ),
        "rng_audit": gates.rng_audit,
        "rng_stream_mapping_sha256": gates.rng_audit[
            "new_rng_stream_mapping_sha256"
        ],
        "seed_to_device": {
            dataset: {
                str(seed): gates.datasets[dataset].seed_to_device[seed]
                for seed in gates.datasets[dataset].eligible_seeds
            }
            for dataset in CONFIRMED_DATASETS
        },
        "dataset_theta": {
            dataset: v4._theta_to_dict(gates.datasets[dataset].theta)
            for dataset in CONFIRMED_DATASETS
        },
        "dataset_eligibility": {
            dataset: gates.datasets[dataset].eligibility_record
            for dataset in CONFIRMED_DATASETS
        },
        "coverage_may_start_only_after_global_overlap_commit": True,
        "cross_dataset_conjunction_or_pooling_permitted": False,
        "science_contract": SCIENCE_CONTRACT,
    }
    canonical = json.loads(
        json.dumps(metadata, sort_keys=True, allow_nan=False)
    )
    if not isinstance(canonical, dict):
        raise RuntimeError("science metadata must remain a JSON object")
    return canonical


def _active_source_snapshot() -> tuple[str, dict[str, Any]]:
    source_hash = experiment_tree_sha256()
    snapshot = _build_source_snapshot()
    if experiment_tree_sha256() != source_hash:
        raise RuntimeError("source/config changed while building the science snapshot")
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
    if (
        len(relative_paths) != len(set(relative_paths))
        or any(not path.is_file() for path in paths)
    ):
        raise RuntimeError("science source snapshot file set is invalid")
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
    snapshot: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if (
        metadata.get("independent_audit_status") != "GO"
        or metadata.get("independent_audit_go_sha256")
        != metadata.get("gate_contract_sha256")
    ):
        raise RuntimeError("science root lacks the exact independent-audit GO")
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
    if experiment_tree_sha256() != gates.active_source_tree_sha256:
        raise RuntimeError("source/config changed during v4 science")
    refreshed = verify_gate_bundle(devices=tuple(metadata["devices"]))
    if refreshed.contract != gates.contract:
        raise RuntimeError("v4 confirmation or audit contract changed during science")
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("science metadata changed")
    _write_manifest(root)
    _validate_complete_root_contents(root, metadata, gates)
    final = _read_json(root / "FINAL_STATUS.json")
    marker = (
        f"complete source_tree_sha256={gates.active_source_tree_sha256} "
        f"gate_contract_sha256={_json_sha256(gates.contract)} "
        f"confirmation_binding_sha256={_json_sha256(gates.confirmation_binding)} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={_file_sha256(root / 'manifest.json')}\n"
    )
    _write_text(root / "COMPLETE", marker)
    _validate_complete_root(root, metadata, gates)


def _validate_complete_root(
    root: Path,
    metadata: Mapping[str, Any],
    gates: GateBundle,
) -> None:
    _validate_complete_root_contents(root, metadata, gates)
    final = _read_json(root / "FINAL_STATUS.json")
    expected = (
        f"complete source_tree_sha256={gates.active_source_tree_sha256} "
        f"gate_contract_sha256={_json_sha256(gates.contract)} "
        f"confirmation_binding_sha256={_json_sha256(gates.confirmation_binding)} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={_file_sha256(root / 'manifest.json')}\n"
    )
    if not (root / "COMPLETE").is_file() or (root / "COMPLETE").read_text() != expected:
        raise RuntimeError("science root COMPLETE marker differs")
    _require_complete_artifact_set(root, metadata, gates)


def _validate_complete_root_contents(
    root: Path,
    metadata: Mapping[str, Any],
    gates: GateBundle,
) -> None:
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("science metadata differs")
    _verify_source_snapshot(root, metadata["source_snapshot"])
    _verify_manifest(root)
    if not _valid_global_overlap_marker(root, gates):
        raise RuntimeError("science root lacks the global precoverage overlap commit")
    global_overlap = _read_json(root / OVERLAP_PHASE / "summary.json")
    expected_statuses = {}
    for dataset in CONFIRMED_DATASETS:
        dataset_gate = gates.datasets[dataset]
        if (
            _read_json(root / "eligibility" / f"{dataset}.json")
            != dataset_gate.eligibility_record
        ):
            raise RuntimeError(f"{dataset} eligibility record differs")
        phase_preset = replace(
            dataset_gate.preset,
            seeds=dataset_gate.eligible_seeds,
        )
        overlap = _load_phase(
            root / OVERLAP_PHASE / dataset,
            phase=OVERLAP_PHASE,
            preset=phase_preset,
            dataset_gate=dataset_gate,
            gates=gates,
        )
        interpretation = (
            "EMPIRICAL_OVERLAP_SCREEN_PASSED"
            if all(bool(row["passed"]) for row in overlap)
            else "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        )
        if (
            _read_json(root / OVERLAP_PHASE / dataset / "summary.json")
            != _overlap_summary(dataset_gate, overlap, interpretation)
            or global_overlap["datasets"][dataset] != interpretation
        ):
            raise RuntimeError(f"{dataset} overlap summary differs")
        science = _load_phase(
            root / SCIENCE_PHASE / dataset / "seeds",
            phase=SCIENCE_PHASE,
            preset=phase_preset,
            dataset_gate=dataset_gate,
            gates=gates,
        )
        if any(
            result["interpretation_status"] != interpretation
            for result in science
        ):
            raise RuntimeError(f"{dataset} science interpretation differs")
        rows = [row for result in science for row in result["rows"]]
        science_root = root / SCIENCE_PHASE / dataset
        bootstrap = _ensure_bootstrap_artifacts(
            science_root,
            dataset_gate.preset,
            create_if_missing=False,
        )
        summary = _science_summary(
            rows,
            dataset_gate=dataset_gate,
            interpretation_status=interpretation,
            bootstrap_contract=bootstrap,
        )
        audit = _coverage_audit(
            rows,
            dataset_gate=dataset_gate,
            summary=summary,
            interpretation_status=interpretation,
        )
        status = _dataset_science_status(dataset_gate, interpretation)
        marker = (
            "curves\n"
            if interpretation == "EMPIRICAL_OVERLAP_SCREEN_PASSED"
            else "curves-descriptive-only\n"
        )
        if (
            _read_json(science_root / "summary.json") != summary
            or _read_json(science_root / "coverage_audit.json") != audit
            or _read_json(science_root / "FINAL_STATUS.json") != status
            or (science_root / "COMPLETE").read_text() != marker
        ):
            raise RuntimeError(f"{dataset} science bundle differs")
        expected_statuses[dataset] = status
    if _read_json(root / "FINAL_STATUS.json") != _final_science_status(
        expected_statuses
    ):
        raise RuntimeError("science root final status differs")


def _load_phase(
    root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    dataset_gate: DatasetGate,
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
            device=dataset_gate.seed_to_device[seed],
            theta=dataset_gate.theta,
            anchor=dataset_gate.anchors[seed],
            source_hash=gates.active_source_tree_sha256,
            gate_contract_sha256=_json_sha256(gates.contract),
            rng_mapping_sha256=gates.rng_audit[
                "new_rng_stream_mapping_sha256"
            ],
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
    if not root.is_dir():
        raise RuntimeError(f"phase directory is missing: {root}")
    observed = {path for path in root.iterdir() if path.is_file()}
    if observed != expected or (root / "COMPLETE").read_text() != "complete\n":
        raise RuntimeError(f"phase artifact set differs: {root}")


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
    for dataset in CONFIRMED_DATASETS:
        gate = gates.datasets[dataset]
        expected.update(
            {
                f"eligibility/{dataset}.json",
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
            for seed in gate.eligible_seeds
        )
        expected.update(
            f"{SCIENCE_PHASE}/{dataset}/seeds/seed_{seed:06d}.json"
            for seed in gate.eligible_seeds
        )
    return expected


def _require_complete_artifact_set(
    root: Path,
    metadata: Mapping[str, Any],
    gates: GateBundle,
) -> None:
    expected = _expected_complete_artifact_paths(metadata, gates)
    observed = _observed_artifact_paths(root)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"completed science artifact set differs; missing={missing}; extra={extra}"
        )


def _require_partial_artifact_subset(
    root: Path,
    metadata: Mapping[str, Any],
    gates: GateBundle,
) -> None:
    allowed = _expected_complete_artifact_paths(metadata, gates)
    observed = _observed_artifact_paths(root)
    if not observed <= allowed:
        raise RuntimeError(
            f"partial science root contains unexpected artifacts: {sorted(observed - allowed)}"
        )


def _observed_artifact_paths(root: Path) -> set[str]:
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symbolic links are forbidden in science roots: {path}")
        if path.is_file():
            relative = path.relative_to(root)
            if ".." in relative.parts:
                raise RuntimeError("science artifact escapes its root")
            observed.add(relative.as_posix())
    return observed


def _root_binding(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "metadata_sha256": _file_sha256(root / "metadata.json"),
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "complete_sha256": _file_sha256(root / "COMPLETE"),
        "final_status_sha256": _file_sha256(root / "FINAL_STATUS.json"),
        "administrative_retry_amendment_sha256": _file_sha256(
            root / "administrative_retry_amendment.json"
        ),
        "support_replay_verification_sha256": _file_sha256(
            root / "support_replay_verification.json"
        ),
    }


def _write_manifest(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symbolic links are forbidden in science roots: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative in {Path("manifest.json"), Path("COMPLETE")}:
            continue
        if ".tmp-" in path.name:
            raise RuntimeError(f"temporary science artifact remains: {path}")
        resolved = (root / relative).resolve()
        if root.resolve() not in resolved.parents:
            raise RuntimeError("science manifest path escapes its root")
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    _write_or_verify_json(
        root / "manifest.json",
        {
            "protocol": PROTOCOL,
            "artifact_count": len(entries),
            "artifacts": entries,
        },
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
        relative = Path(entry["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative in {Path("manifest.json"), Path("COMPLETE")}
        ):
            raise RuntimeError("science manifest contains an unsafe/root commit path")
        path = root / relative
        resolved = path.resolve()
        if root.resolve() not in resolved.parents or resolved in expected:
            raise RuntimeError("science manifest path escapes or is duplicated")
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
        if path.is_file()
        and path.relative_to(root) not in {Path("manifest.json"), Path("COMPLETE")}
    }
    if (
        observed != expected
        or manifest["artifact_count"] != len(entries)
        or len(expected) != len(entries)
    ):
        raise RuntimeError("science manifest file set differs")


def _verify_source_snapshot(root: Path, contract: Mapping[str, Any]) -> None:
    for name in ("archive", "manifest"):
        path = root / contract[f"{name}_path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != contract[f"{name}_bytes"]
            or _file_sha256(path) != contract[f"{name}_sha256"]
        ):
            raise RuntimeError(f"science source snapshot {name} differs")


def _write_or_verify_json(path: Path, value: object) -> None:
    if path.exists():
        if not path.is_file() or _read_json(path) != value:
            raise RuntimeError(f"existing JSON artifact differs: {path}")
        return
    _write_json(path, value)


def _write_or_verify_text(path: Path, value: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text() != value:
            raise RuntimeError(f"existing text artifact differs: {path}")
        return
    _write_text(path, value)


def _write_npy(path: Path, value: np.ndarray) -> None:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    _atomic_write(path, stream.getvalue())


def _unlink_root_complete(root: Path) -> None:
    if not root.is_dir():
        return
    complete = root / "COMPLETE"
    complete.unlink(missing_ok=True)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
