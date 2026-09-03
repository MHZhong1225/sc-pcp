"""Replay the Native signed-gamma preflight after one validator-only repair.

This administrative runner reuses the parent mechanism RNG IDs only when the
exact pinned parent bundle is intact.  It changes no scientific setting and
accepts no probe difference other than ``gamma_rows[*].finite_and_structural``.

Validate without consuming any RNG ID:

    conda run -n ucp python \
      scripts/run_native_synthetic_signed_gamma_time_coordinate_repair_r1.py \
      --validate-only
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_native_synthetic_signed_gamma_preflight as base  # noqa: E402
from scpcp.native_signed_gamma import (  # noqa: E402
    NativeSignedGammaBenchmarkConfig,
    mechanism_probe,
    seed_passes_mechanism_gate,
)


REPAIR_PROTOCOL = "native_synthetic_signed_gamma_time_coordinate_repair_r1"
DEFAULT_AMENDMENT = (
    ROOT / "configs/native_synthetic_signed_gamma_time_coordinate_repair_r1.yaml"
)
EXPECTED_PARENT_ROOT = ROOT / "results/work/native_synthetic_signed_gamma_v1"
EXPECTED_OUTPUT_ROOT = (
    ROOT
    / "results/work/native_synthetic_signed_gamma_v1_time_coordinate_repair_r1"
)
REPAIR_RUNNER = Path(__file__).resolve()
BASE_RUNNER = ROOT / "scripts/run_native_synthetic_signed_gamma_preflight.py"
SCIENCE_RUNNER = ROOT / "scripts/run_native_synthetic_signed_gamma_science.py"
NATIVE_MODULE = ROOT / "src/scpcp/native_signed_gamma.py"
BASE_CONFIG = ROOT / "configs/native_synthetic_signed_gamma.yaml"
SCIENCE_CONFIG = ROOT / "configs/native_synthetic_signed_gamma_science.yaml"
FORBIDDEN_RESULT_FIELDS = ("coverage", "width", "q90", "score", "selection")

METADATA_FIELDS = (
    "protocol",
    "role",
    "gate_only",
    "administrative_only",
    "amendment_path",
    "amendment_sha256",
    "amendment_payload_sha256",
    "parent_bundle",
    "scientific_config",
    "scientific_config_sha256",
    "scientific_config_equivalence",
    "output_root",
    "replay_rng_audit",
    "downstream_rng_reservation",
    "seed_device_mapping",
    "seed_device_mapping_sha256",
    "source_tree_sha256",
    "source_snapshot",
    "dependency_files",
    "environment",
    "environment_sha256",
    "invocation",
    "invocation_sha256",
    "artifact_schema_sha256",
    "launch_contract_sha256",
    "repair_contract",
    "information_firewall",
    "downstream_authorization_rule",
)
SUMMARY_FIELDS = (
    "protocol",
    "gate_only",
    "administrative_only",
    "amendment_sha256",
    "parent_manifest_sha256",
    "scientific_config_sha256",
    "replay_rng_audit_sha256",
    "source_tree_sha256",
    "n_prespecified",
    "n_exact_replays",
    "n_repaired_fields_valid",
    "n_passed",
    "passed_rng_ids",
    "required_passed_rng_ids",
    "status",
    "failure_consequence",
    "downstream_authorized",
)
SEED_ARTIFACT_FIELDS = (
    "protocol",
    "scientific_protocol",
    "rng_label",
    "rng_id",
    "device",
    "amendment_sha256",
    "parent_seed_artifact_sha256",
    "scientific_config_sha256",
    "replay_rng_audit_sha256",
    "source_tree_sha256",
    "probe_comparison",
    "probe",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the parent, replay exemption, and reserved downstream bank",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    amendment_path = args.amendment.resolve()
    amendment = load_amendment(amendment_path)
    if args.validate_only:
        print(
            json.dumps(
                validation_payload(amendment, amendment_path),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    run_repair_replay(
        amendment,
        amendment_path=amendment_path,
        resume=args.resume,
        invocation_argv=base._canonical_invocation(raw_argv),
    )


def load_amendment(path: Path = DEFAULT_AMENDMENT) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("repair amendment root must be a mapping")
    if set(raw) != {
        "repair_protocol",
        "role",
        "administrative_only",
        "parent",
        "repair",
        "replay",
        "downstream_rng_reservation",
    }:
        raise RuntimeError("repair amendment fields differ")
    if (
        raw["repair_protocol"] != REPAIR_PROTOCOL
        or raw["role"] != "administrative_validator_repair_replay"
        or raw["administrative_only"] is not True
    ):
        raise RuntimeError("repair amendment identity differs")

    parent = raw.get("parent")
    if not isinstance(parent, dict) or set(parent) != {
        "root",
        "scientific_protocol",
        "decision",
        "config_payload_sha256",
        "rng_audit_sha256",
        "source_tree_sha256",
        "source_snapshot_sha256",
        "artifact_count",
        "artifact_inventory_sha256",
        "bundle_files",
        "artifact_inventory",
        "formal_rng_ids",
    }:
        raise RuntimeError("repair parent contract fields differ")
    if (
        _resolve_project_path(parent["root"]) != EXPECTED_PARENT_ROOT.resolve()
        or parent["scientific_protocol"] != "native_synthetic_signed_gamma_v1"
        or parent["decision"] != "NO_GO"
    ):
        raise RuntimeError("repair parent identity differs")
    _validate_parent_inventory_contract(parent)

    repair = raw.get("repair")
    if not isinstance(repair, dict) or repair != {
        "defect": "time_coordinate_reference_used_torch_arange_division_rounding",
        "scope": "gamma_rows[*].finite_and_structural",
        "old_reference": "torch.arange(T+1,dtype=state.dtype,device=state.device)/T",
        "corrected_reference": "state.new_tensor([stage/T for stage in range(T+1)])",
        "expected_parent_values": [False] * 5,
        "expected_replay_values": [True] * 5,
        "exact_probe_comparison": "remove_only_gamma_rows_finite_and_structural",
        "scientific_changes": [],
        "reused_rng_authority": "exact_pinned_parent_bundle_only",
    }:
        raise RuntimeError("repair scope is not the exact audited validator change")

    replay = raw.get("replay")
    if not isinstance(replay, dict) or set(replay) != {
        "output_root",
        "required_passed_rng_ids",
        "devices",
    }:
        raise RuntimeError("repair replay contract fields differ")
    if (
        _resolve_project_path(replay["output_root"]) != EXPECTED_OUTPUT_ROOT.resolve()
        or replay["required_passed_rng_ids"] != 19
        or replay["devices"] != ["cuda:0", "cuda:1"]
    ):
        raise RuntimeError("repair replay contract differs")

    build_downstream_rng_reservation(raw)
    return raw


def _validate_parent_inventory_contract(parent: Mapping[str, Any]) -> None:
    inventory = parent.get("artifact_inventory")
    if not isinstance(inventory, list) or parent.get("artifact_count") != len(inventory):
        raise RuntimeError("pinned parent artifact count differs")
    paths = []
    for record in inventory:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RuntimeError("pinned parent artifact record fields differ")
        if (
            not isinstance(record["path"], str)
            or not _is_sha256(record["sha256"])
            or not isinstance(record["size_bytes"], int)
            or record["size_bytes"] < 0
        ):
            raise RuntimeError("pinned parent artifact record is malformed")
        paths.append(record["path"])
    if len(paths) != len(set(paths)):
        raise RuntimeError("pinned parent artifact paths are not unique")
    if parent.get("artifact_inventory_sha256") != base._canonical_sha256(inventory):
        raise RuntimeError("pinned parent artifact inventory hash differs")

    bundle_files = parent.get("bundle_files")
    if not isinstance(bundle_files, dict) or set(bundle_files) != {
        "artifact_schema.json",
        "metadata.json",
        "summary.json",
        "manifest.json",
        "COMPLETE",
    } or any(not _is_sha256(value) for value in bundle_files.values()):
        raise RuntimeError("pinned parent bundle-file hashes differ")
    inventory_by_path = {record["path"]: record for record in inventory}
    for name in ("artifact_schema.json", "metadata.json", "summary.json"):
        if inventory_by_path.get(name, {}).get("sha256") != bundle_files[name]:
            raise RuntimeError("pinned parent bundle hash and inventory disagree")
    for name in (
        "config_payload_sha256",
        "rng_audit_sha256",
        "source_tree_sha256",
        "source_snapshot_sha256",
        "artifact_inventory_sha256",
    ):
        if not _is_sha256(parent.get(name)):
            raise RuntimeError(f"pinned parent {name} is malformed")
    formal_ids = parent.get("formal_rng_ids")
    if formal_ids != list(range(121_000, 121_200, 10)):
        raise RuntimeError("pinned parent formal RNG inventory differs")


def validate_parent_bundle(amendment: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every pinned parent byte and its original source snapshot."""

    parent_contract = amendment["parent"]
    parent_root = _resolve_project_path(parent_contract["root"])
    expected_files = {
        record["path"] for record in parent_contract["artifact_inventory"]
    } | {"manifest.json", "COMPLETE"}
    observed_files = {
        path.relative_to(parent_root).as_posix()
        for path in parent_root.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise RuntimeError("parent bundle file inventory differs from amendment")

    for relative, expected_hash in parent_contract["bundle_files"].items():
        path = parent_root / relative
        if not path.is_file() or base._file_sha256(path) != expected_hash:
            raise RuntimeError(f"pinned parent bundle file differs: {relative}")
    for record in parent_contract["artifact_inventory"]:
        path = parent_root / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or base._file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"pinned parent artifact differs: {record['path']}")

    manifest = base._read_json(parent_root / "manifest.json")
    if (
        manifest.get("artifact_count") != parent_contract["artifact_count"]
        or manifest.get("artifacts") != parent_contract["artifact_inventory"]
        or manifest.get("config_payload_sha256")
        != parent_contract["config_payload_sha256"]
        or manifest.get("rng_audit_sha256") != parent_contract["rng_audit_sha256"]
        or manifest.get("source_tree_sha256") != parent_contract["source_tree_sha256"]
    ):
        raise RuntimeError("parent manifest differs from the pinned amendment")
    metadata = base._read_json(parent_root / "metadata.json")
    summary = base._read_json(parent_root / "summary.json")
    complete = base._read_json(parent_root / "COMPLETE")
    if (
        metadata.get("protocol") != parent_contract["scientific_protocol"]
        or metadata.get("config_payload_sha256")
        != parent_contract["config_payload_sha256"]
        or metadata.get("rng_audit", {}).get("audit_sha256")
        != parent_contract["rng_audit_sha256"]
        or metadata.get("source_tree_sha256") != parent_contract["source_tree_sha256"]
        or metadata.get("source_snapshot", {}).get("archive_sha256")
        != parent_contract["source_snapshot_sha256"]
        or summary.get("status") != parent_contract["decision"]
        or complete.get("decision") != parent_contract["decision"]
        or complete.get("manifest_sha256")
        != parent_contract["bundle_files"]["manifest.json"]
    ):
        raise RuntimeError("parent metadata decision or hash chain differs")
    if metadata.get("rng_audit", {}).get("formal_rng_ids") != parent_contract[
        "formal_rng_ids"
    ]:
        raise RuntimeError("parent metadata formal RNG inventory differs")

    snapshot_path = parent_root / metadata["source_snapshot"]["archive_path"]
    with tempfile.TemporaryDirectory(prefix="native-parent-snapshot-") as directory:
        extracted_root = Path(directory)
        _extract_verified_source_snapshot(snapshot_path, extracted_root)
        base.validate_completed_bundle(parent_root, source_root=extracted_root)
        old_native = (extracted_root / "src/scpcp/native_signed_gamma.py").read_text(
            encoding="utf-8"
        )
        if "torch.arange(" not in old_native or ") / trajectory.horizon" not in old_native:
            raise RuntimeError("parent snapshot lacks the audited time-coordinate defect")

    result = {
        "root": parent_contract["root"],
        "protocol": parent_contract["scientific_protocol"],
        "decision": parent_contract["decision"],
        "manifest_sha256": parent_contract["bundle_files"]["manifest.json"],
        "complete_sha256": parent_contract["bundle_files"]["COMPLETE"],
        "config_payload_sha256": parent_contract["config_payload_sha256"],
        "rng_audit_sha256": parent_contract["rng_audit_sha256"],
        "source_tree_sha256": parent_contract["source_tree_sha256"],
        "source_snapshot_sha256": parent_contract["source_snapshot_sha256"],
        "artifact_count": parent_contract["artifact_count"],
        "artifact_inventory_sha256": parent_contract[
            "artifact_inventory_sha256"
        ],
        "formal_rng_ids": list(parent_contract["formal_rng_ids"]),
        "status": "fully_validated",
    }
    result["validation_sha256"] = base._canonical_sha256(result)
    return result


def _extract_verified_source_snapshot(archive_path: Path, target_root: Path) -> None:
    with tarfile.open(archive_path, mode="r") as archive:
        members = archive.getmembers()
        if not members or any(not member.isfile() for member in members):
            raise RuntimeError("parent source snapshot contains a non-file member")
        for member in members:
            target = (target_root / member.name).resolve()
            if not base._is_relative_to(target, target_root.resolve()):
                raise RuntimeError("parent source snapshot path escapes extraction root")
            payload = archive.extractfile(member)
            if payload is None:
                raise RuntimeError("parent source snapshot member is unreadable")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload.read())


def build_effective_config(
    amendment: Mapping[str, Any],
) -> tuple[NativeSignedGammaBenchmarkConfig, dict[str, Any]]:
    parent_root = _resolve_project_path(amendment["parent"]["root"])
    parent_metadata = base._read_json(parent_root / "metadata.json")
    parent_config_payload = parent_metadata["config"]
    parent_config = base._config_from_payload(parent_config_payload)
    output_root = _resolve_project_path(amendment["replay"]["output_root"])
    config = parent_config.with_overrides(output_root=output_root)
    if list(config.devices) != amendment["replay"]["devices"]:
        raise RuntimeError("repair devices differ from the parent scientific protocol")
    parent_without_output = dict(parent_config_payload)
    replay_without_output = config.to_dict()
    parent_output = parent_without_output.pop("output_root")
    replay_output = replay_without_output.pop("output_root")
    if parent_without_output != replay_without_output:
        raise RuntimeError("repair replay changes a scientific config field")
    equivalence = {
        "status": "exact_except_output_root",
        "allowed_difference": "output_root",
        "parent_output_root": parent_output,
        "replay_output_root": replay_output,
        "shared_payload_sha256": base._canonical_sha256(parent_without_output),
    }
    return config, equivalence


def replay_rng_mapping(config: NativeSignedGammaBenchmarkConfig) -> dict[str, int]:
    return base.formal_rng_mapping(config)


def audit_replay_rng_ids(
    config: NativeSignedGammaBenchmarkConfig,
    amendment: Mapping[str, Any],
    *,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
    external_reservations: Mapping[str, Iterable[int]] | None = None,
) -> dict[str, Any]:
    """Authorize only the 20 collisions contributed by the pinned parent."""

    artifact_root = ROOT / "results" if artifact_root is None else artifact_root
    source_root = ROOT if source_root is None else source_root
    parent_root = _resolve_project_path(amendment["parent"]["root"])
    output_root = config.output_root.resolve()
    mapping = replay_rng_mapping(config)
    if list(mapping.values()) != amendment["parent"]["formal_rng_ids"]:
        raise RuntimeError("repair replay RNG mapping differs from the parent")

    # Full parent validation above binds these IDs to the 20 exact seed
    # artifacts.  Re-scanning the parent's stored global audit would mistake
    # its historical inventory for newly consumed parent streams.
    parent_ids = set(amendment["parent"]["formal_rng_ids"])
    other_scan = _artifact_rng_scan_excluding(
        artifact_root,
        excluded_roots=(parent_root, output_root),
    )
    excluded_source_paths = replay_source_exclusions(source_root)
    source_scan = base._source_rng_scan(
        source_root,
        excluded_paths=excluded_source_paths,
    )
    reservations = (
        base.COORDINATED_EXTERNAL_RESERVATIONS
        if external_reservations is None
        else external_reservations
    )
    external_ids = (
        set().union(*(set(values) for values in reservations.values()))
        if reservations
        else set()
    )
    classification = classify_replay_collisions(
        mapping,
        parent_ids=parent_ids,
        other_artifact_ids=other_scan["actual"],
        source_ids=source_scan["actual"],
        external_ids=external_ids,
    )
    audit = {
        "status": classification["status"],
        "policy": "exact_pinned_parent_replay_exemption_v1",
        "formal_rng_mapping": mapping,
        "formal_rng_mapping_sha256": base._canonical_sha256(mapping),
        "formal_rng_id_count": len(mapping),
        "formal_rng_id_sha256": base._integer_set_sha256(mapping.values()),
        "parent_root": amendment["parent"]["root"],
        "parent_manifest_sha256": amendment["parent"]["bundle_files"][
            "manifest.json"
        ],
        "parent_actual_rng_id_count": len(parent_ids),
        "parent_actual_rng_id_sha256": base._integer_set_sha256(
            parent_ids
        ),
        "other_artifact_actual_rng_id_count": len(other_scan["actual"]),
        "other_artifact_actual_rng_id_sha256": base._integer_set_sha256(
            other_scan["actual"]
        ),
        "source_actual_rng_id_count": len(source_scan["actual"]),
        "source_actual_rng_id_sha256": base._integer_set_sha256(
            source_scan["actual"]
        ),
        "external_reserved_rng_id_count": len(external_ids),
        "external_reserved_rng_id_sha256": base._integer_set_sha256(external_ids),
        "raw_collision_count": classification["raw_collision_count"],
        "raw_collisions": classification["raw_collisions"],
        "authorized_parent_collision_count": classification[
            "authorized_parent_collision_count"
        ],
        "authorized_parent_collisions": classification[
            "authorized_parent_collisions"
        ],
        "unauthorized_collision_count": classification[
            "unauthorized_collision_count"
        ],
        "unauthorized_collisions": classification["unauthorized_collisions"],
        "missing_parent_collision_count": classification[
            "missing_parent_collision_count"
        ],
        "missing_parent_collisions": classification["missing_parent_collisions"],
        "excluded_output_root": str(output_root),
        "excluded_parent_root_from_other_scan": str(parent_root.resolve()),
        "exempted_parent_manifest_sha256": amendment["parent"]["bundle_files"][
            "manifest.json"
        ],
    }
    audit["audit_sha256"] = base._canonical_sha256(audit)
    validate_replay_rng_audit(config, amendment, audit)
    return audit


def classify_replay_collisions(
    mapping: Mapping[str, int],
    *,
    parent_ids: set[int],
    other_artifact_ids: set[int],
    source_ids: set[int],
    external_ids: set[int],
) -> dict[str, Any]:
    raw: dict[str, int] = {}
    authorized: dict[str, int] = {}
    unauthorized: dict[str, dict[str, Any]] = {}
    missing_parent: dict[str, int] = {}
    for label, rng_id in mapping.items():
        categories = []
        if rng_id in parent_ids:
            categories.append("exact_parent_bundle")
        else:
            missing_parent[label] = rng_id
        if rng_id in other_artifact_ids:
            categories.append("other_artifact")
        if rng_id in source_ids:
            categories.append("source_actual_use")
        if rng_id in external_ids:
            categories.append("external_reservation")
        if categories:
            raw[label] = rng_id
        if categories == ["exact_parent_bundle"]:
            authorized[label] = rng_id
        elif categories:
            unauthorized[label] = {"rng_id": rng_id, "categories": categories}
    passed = (
        len(raw) == len(mapping)
        and len(authorized) == len(mapping)
        and not unauthorized
        and not missing_parent
    )
    return {
        "status": (
            "passed_with_exact_parent_replay_exemption"
            if passed
            else "unauthorized_replay_collision"
        ),
        "raw_collision_count": len(raw),
        "raw_collisions": raw,
        "authorized_parent_collision_count": len(authorized),
        "authorized_parent_collisions": authorized,
        "unauthorized_collision_count": len(unauthorized),
        "unauthorized_collisions": unauthorized,
        "missing_parent_collision_count": len(missing_parent),
        "missing_parent_collisions": missing_parent,
    }


def validate_replay_rng_audit(
    config: NativeSignedGammaBenchmarkConfig,
    amendment: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    mapping = replay_rng_mapping(config)
    if (
        audit.get("formal_rng_mapping") != mapping
        or audit.get("formal_rng_mapping_sha256") != base._canonical_sha256(mapping)
        or audit.get("formal_rng_id_count") != len(mapping)
        or audit.get("formal_rng_id_sha256")
        != base._integer_set_sha256(mapping.values())
    ):
        raise RuntimeError("repair RNG audit mapping differs")
    if (
        audit.get("parent_manifest_sha256")
        != amendment["parent"]["bundle_files"]["manifest.json"]
        or audit.get("exempted_parent_manifest_sha256")
        != amendment["parent"]["bundle_files"]["manifest.json"]
    ):
        raise RuntimeError("repair RNG exemption is not bound to the exact parent")
    expected_count = len(mapping)
    if (
        audit.get("status") != "passed_with_exact_parent_replay_exemption"
        or audit.get("raw_collision_count") != expected_count
        or audit.get("authorized_parent_collision_count") != expected_count
        or audit.get("unauthorized_collision_count") != 0
        or audit.get("unauthorized_collisions") != {}
        or audit.get("missing_parent_collision_count") != 0
        or audit.get("missing_parent_collisions") != {}
        or audit.get("raw_collisions") != mapping
        or audit.get("authorized_parent_collisions") != mapping
    ):
        raise RuntimeError(
            "repair RNG audit must report raw 20, authorized parent 20, unauthorized 0"
        )
    unhashed = dict(audit)
    stored_hash = unhashed.pop("audit_sha256", None)
    if stored_hash != base._canonical_sha256(unhashed):
        raise RuntimeError("repair RNG audit canonical hash differs")


def _artifact_rng_scan_excluding(
    root: Path,
    *,
    excluded_roots: Sequence[Path],
) -> dict[str, Any]:
    report = base._empty_rng_scan()
    if not root.exists():
        return report
    root_resolved = root.resolve()
    excluded = tuple(path.resolve() for path in excluded_roots)
    for path in sorted(root.rglob("*")):
        resolved = path.resolve()
        if not base._is_relative_to(resolved, root_resolved):
            raise RuntimeError(f"artifact scan path escapes its root: {path}")
        if any(base._is_relative_to(resolved, value) for value in excluded):
            continue
        match = base.SEED_ARTIFACT_NAME.fullmatch(path.name)
        if match:
            report["actual"].add(int(match.group(1)))
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in base.STRUCTURED_ARTIFACT_SUFFIXES or path.name == "COMPLETE":
            payload = base._read_structured_artifact(path)
            base._collect_artifact_rng_fields(
                payload,
                report,
                artifact_path=path,
                artifact_root=root_resolved,
            )
        elif suffix in base.TABULAR_ARTIFACT_SUFFIXES:
            base._collect_tabular_rng_fields(path, report)
    report["binary_bindings"] = sorted(
        report["binary_bindings"],
        key=lambda row: (row["metadata_path"], row["field_path"]),
    )
    return report


def build_downstream_rng_reservation(
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    contract = amendment.get("downstream_rng_reservation")
    if not isinstance(contract, dict) or set(contract) != {
        "status",
        "namespace",
        "reserved_base_seeds",
        "reserved_bootstrap_seed",
        "mapping_formula",
        "reserved_rng_id_count",
        "reserved_rng_mapping_sha256",
    }:
        raise RuntimeError("downstream RNG reservation fields differ")
    if (
        contract["status"] != "reserved_not_consumed"
        or contract["namespace"]
        != "native_synthetic_signed_gamma_science_v1:121400..121590:step10"
        or contract["reserved_base_seeds"] != list(range(121_400, 121_600, 10))
        or contract["reserved_bootstrap_seed"] != 12_140_019
    ):
        raise RuntimeError("downstream RNG reservation identity differs")
    formula = contract["mapping_formula"]
    if formula != {
        "paper_seed_multiplier": 1_000_003,
        "paper_seed_modulus": 2_147_483_647,
        "calibration_stream": 1_700_101,
        "reference_stream": 1_700_401,
        "adaptation_stream": 700_001,
        "methods": {
            "ACI": {"offset": 101, "round_stride": 17_923},
            "SPCI": {"offset": 211, "round_stride": 47_021},
            "PRC": {"offset": 307, "round_stride": 61_103},
        },
        "rounds": [0, 1, 2],
    }:
        raise RuntimeError("downstream RNG reservation formula differs")

    def paper_seed(seed: int, stream: int) -> int:
        return int(
            (formula["paper_seed_multiplier"] * seed + stream)
            % formula["paper_seed_modulus"]
        )

    mapping = {
        "summary/bootstrap_complete_seed_matrix": contract[
            "reserved_bootstrap_seed"
        ]
    }
    for seed in contract["reserved_base_seeds"]:
        prefix = f"science/base_{seed}"
        mapping[f"{prefix}/task"] = seed
        mapping[f"{prefix}/calibration"] = paper_seed(
            seed, formula["calibration_stream"]
        )
        mapping[f"{prefix}/reference"] = paper_seed(
            seed, formula["reference_stream"]
        )
        adaptation = paper_seed(seed, formula["adaptation_stream"])
        for method, values in formula["methods"].items():
            for round_index in formula["rounds"]:
                mapping[f"{prefix}/{method}_round_{round_index}"] = (
                    paper_seed(adaptation, values["offset"])
                    + values["round_stride"] * round_index
                )
    if (
        len(mapping) != contract["reserved_rng_id_count"]
        or len(set(mapping.values())) != len(mapping)
        or base._canonical_sha256(mapping)
        != contract["reserved_rng_mapping_sha256"]
    ):
        raise RuntimeError("downstream RNG reservation mapping differs")
    return mapping


def audit_downstream_rng_reservation(
    amendment: Mapping[str, Any],
    replay_audit: Mapping[str, Any],
    *,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    mapping = build_downstream_rng_reservation(amendment)
    artifact_root = ROOT / "results" if artifact_root is None else artifact_root
    source_root = ROOT if source_root is None else source_root
    parent_root = _resolve_project_path(amendment["parent"]["root"])
    output_root = _resolve_project_path(amendment["replay"]["output_root"])
    other_scan = _artifact_rng_scan_excluding(
        artifact_root,
        excluded_roots=(parent_root, output_root),
    )
    source_scan = base._source_rng_scan(
        source_root,
        excluded_paths=downstream_source_declaration_exclusions(source_root),
    )
    external = set().union(
        *(set(values) for values in base.COORDINATED_EXTERNAL_RESERVATIONS.values())
    )
    replay_ids = set(replay_audit["formal_rng_mapping"].values())
    prior = other_scan["actual"] | source_scan["actual"] | external | replay_ids
    collisions = {
        label: rng_id for label, rng_id in mapping.items() if rng_id in prior
    }
    if collisions:
        raise RuntimeError(
            f"reserved downstream RNG mapping collides with prior use: {collisions}"
        )
    result = {
        "status": "reserved_not_consumed_and_collision_free",
        "namespace": amendment["downstream_rng_reservation"]["namespace"],
        "reserved_rng_mapping": mapping,
        "reserved_rng_mapping_sha256": base._canonical_sha256(mapping),
        "reserved_rng_id_count": len(mapping),
        "reserved_rng_id_sha256": base._integer_set_sha256(mapping.values()),
        "internal_rng_ids_unique": len(set(mapping.values())) == len(mapping),
        "replay_rng_ids_disjoint": not (set(mapping.values()) & replay_ids),
        "collision_count": 0,
        "collisions": {},
        "execution_authorized_by_repair_runner": False,
    }
    result["reservation_sha256"] = base._canonical_sha256(result)
    validate_downstream_rng_reservation(amendment, result)
    return result


def replay_source_exclusions(source_root: Path = ROOT) -> set[Path]:
    """Exclude only files that declare the administrative replay IDs."""

    relative_paths = (
        "scripts/run_native_synthetic_signed_gamma_time_coordinate_repair_r1.py",
        "scripts/run_native_synthetic_signed_gamma_preflight.py",
        "src/scpcp/native_signed_gamma.py",
        "configs/native_synthetic_signed_gamma.yaml",
        "configs/native_synthetic_signed_gamma_time_coordinate_repair_r1.yaml",
    )
    return {(source_root / relative).resolve() for relative in relative_paths}


def downstream_source_declaration_exclusions(
    source_root: Path = ROOT,
) -> set[Path]:
    """Also exclude the exact runner that declares the reserved future bank."""

    return replay_source_exclusions(source_root) | {
        (source_root / "scripts/run_native_synthetic_signed_gamma_science.py").resolve(),
        (source_root / "configs/native_synthetic_signed_gamma_science.yaml").resolve(),
    }


def validate_downstream_rng_reservation(
    amendment: Mapping[str, Any], reservation: Mapping[str, Any]
) -> None:
    mapping = build_downstream_rng_reservation(amendment)
    expected_without_hash = {
        "status": "reserved_not_consumed_and_collision_free",
        "namespace": amendment["downstream_rng_reservation"]["namespace"],
        "reserved_rng_mapping": mapping,
        "reserved_rng_mapping_sha256": base._canonical_sha256(mapping),
        "reserved_rng_id_count": len(mapping),
        "reserved_rng_id_sha256": base._integer_set_sha256(mapping.values()),
        "internal_rng_ids_unique": True,
        "replay_rng_ids_disjoint": True,
        "collision_count": 0,
        "collisions": {},
        "execution_authorized_by_repair_runner": False,
    }
    expected = dict(expected_without_hash)
    expected["reservation_sha256"] = base._canonical_sha256(expected_without_hash)
    if reservation != expected:
        raise RuntimeError("stored downstream RNG reservation contract differs")


def validation_payload(
    amendment: Mapping[str, Any], amendment_path: Path
) -> dict[str, Any]:
    parent = validate_parent_bundle(amendment)
    config, equivalence = build_effective_config(amendment)
    audit = audit_replay_rng_ids(config, amendment)
    reservation = audit_downstream_rng_reservation(amendment, audit)
    return {
        "protocol": REPAIR_PROTOCOL,
        "contract_valid": True,
        "formal_replay_permitted": True,
        "gate_only": True,
        "administrative_only": True,
        "amendment_path": base._project_path(amendment_path),
        "amendment_sha256": base._file_sha256(amendment_path),
        "amendment_payload_sha256": base._canonical_sha256(amendment),
        "parent_bundle": parent,
        "scientific_config_sha256": base._canonical_sha256(config.to_dict()),
        "scientific_config_equivalence": equivalence,
        "output_root": str(config.output_root),
        "replay_rng_audit": audit,
        "downstream_rng_reservation": reservation,
    }


def run_repair_replay(
    amendment: Mapping[str, Any],
    *,
    amendment_path: Path,
    resume: bool = False,
    invocation_argv: Sequence[str] = (),
) -> None:
    parent = validate_parent_bundle(amendment)
    config, equivalence = build_effective_config(amendment)
    rng_audit = audit_replay_rng_ids(config, amendment)
    downstream_reservation = audit_downstream_rng_reservation(amendment, rng_audit)
    source_hash = base._experiment_tree_sha256(ROOT)
    source_snapshot = base._build_source_snapshot(ROOT)
    if base._experiment_tree_sha256(ROOT) != source_hash:
        raise RuntimeError("experiment/source tree changed while building repair snapshot")
    schema = artifact_schema(base._file_sha256(amendment_path))
    metadata = build_metadata(
        amendment,
        amendment_path=amendment_path,
        parent=parent,
        config=config,
        equivalence=equivalence,
        rng_audit=rng_audit,
        downstream_reservation=downstream_reservation,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        schema=schema,
        invocation_argv=invocation_argv,
    )
    _assert_information_firewall(metadata)
    _assert_information_firewall(schema)
    root = config.output_root.resolve()
    if resume and (root / "COMPLETE").is_file():
        validate_completed_bundle(
            root,
            expected_metadata=metadata,
            amendment=amendment,
            amendment_path=amendment_path,
        )
        return
    prepare_root(
        root,
        metadata=metadata,
        schema=schema,
        source_snapshot=source_snapshot,
        amendment_path=amendment_path,
        resume=resume,
    )

    mapping = replay_rng_mapping(config)
    devices_by_label = base.seed_device_mapping(config)
    existing = load_existing_seed_artifacts(
        root,
        mapping=mapping,
        devices_by_label=devices_by_label,
        metadata=metadata,
        amendment=amendment,
    )
    results = dict(existing)
    pending = [(label, rng_id) for label, rng_id in mapping.items() if label not in results]
    if pending:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(config.devices), mp_context=context
        ) as executor:
            future_to_label = {
                executor.submit(
                    run_seed,
                    config,
                    label,
                    rng_id,
                    devices_by_label[label],
                    metadata["amendment_sha256"],
                    metadata["scientific_config_sha256"],
                    rng_audit["audit_sha256"],
                    source_hash,
                    amendment,
                ): label
                for label, rng_id in pending
            }
            for future in as_completed(future_to_label):
                label = future_to_label[future]
                artifact = future.result()
                validate_seed_artifact(
                    artifact,
                    expected_label=label,
                    expected_rng_id=mapping[label],
                    expected_device=devices_by_label[label],
                    metadata=metadata,
                    amendment=amendment,
                )
                results[label] = artifact
                base._write_json(
                    root / "mechanism" / f"seed_{mapping[label]}.json", artifact
                )

    summary = summarize_results(config, results, metadata)
    _assert_information_firewall(summary)
    base._write_json(root / "summary.json", summary)
    finalize_root(
        root,
        metadata=metadata,
        summary=summary,
        amendment=amendment,
        amendment_path=amendment_path,
    )


def run_seed(
    config: NativeSignedGammaBenchmarkConfig,
    rng_label: str,
    rng_id: int,
    device: str,
    amendment_sha256: str,
    config_sha256: str,
    audit_sha256: str,
    source_sha256: str,
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    probe = mechanism_probe(config, seed=rng_id, device=device)
    parent_path = (
        _resolve_project_path(amendment["parent"]["root"])
        / "mechanism"
        / f"seed_{rng_id}.json"
    )
    parent_artifact = base._read_json(parent_path)
    comparison = compare_probe_payloads(parent_artifact["probe"], probe, amendment)
    return {
        "protocol": REPAIR_PROTOCOL,
        "scientific_protocol": config.protocol,
        "rng_label": rng_label,
        "rng_id": rng_id,
        "device": device,
        "amendment_sha256": amendment_sha256,
        "parent_seed_artifact_sha256": base._file_sha256(parent_path),
        "scientific_config_sha256": config_sha256,
        "replay_rng_audit_sha256": audit_sha256,
        "source_tree_sha256": source_sha256,
        "probe_comparison": comparison,
        "probe": probe,
    }


def compare_probe_payloads(
    parent_probe: Mapping[str, Any],
    replay_probe: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    base._validate_probe_payload(parent_probe, expected_seed=int(parent_probe["seed"]))
    base._validate_probe_payload(replay_probe, expected_seed=int(parent_probe["seed"]))
    parent_values = [
        row["finite_and_structural"] for row in parent_probe["gamma_rows"]
    ]
    replay_values = [
        row["finite_and_structural"] for row in replay_probe["gamma_rows"]
    ]
    parent_stripped = _without_repaired_probe_field(parent_probe)
    replay_stripped = _without_repaired_probe_field(replay_probe)
    exact = base._canonical_json_bytes(parent_stripped) == base._canonical_json_bytes(
        replay_stripped
    )
    repaired_values_valid = (
        parent_values == amendment["repair"]["expected_parent_values"]
        and replay_values == amendment["repair"]["expected_replay_values"]
    )
    result = {
        "status": "EXACT_REPLAY" if exact and repaired_values_valid else "INVALID_REPLAY",
        "comparison_rule": amendment["repair"]["exact_probe_comparison"],
        "parent_probe_without_repaired_field_sha256": base._canonical_sha256(
            parent_stripped
        ),
        "replay_probe_without_repaired_field_sha256": base._canonical_sha256(
            replay_stripped
        ),
        "exact_after_removing_only_repaired_field": exact,
        "parent_repaired_field_values": parent_values,
        "replay_repaired_field_values": replay_values,
        "repaired_field_values_valid": repaired_values_valid,
    }
    result["comparison_sha256"] = base._canonical_sha256(result)
    return result


def _without_repaired_probe_field(probe: Mapping[str, Any]) -> dict[str, Any]:
    stripped = deepcopy(dict(probe))
    for row in stripped["gamma_rows"]:
        if set(key for key in row if key == "finite_and_structural") != {
            "finite_and_structural"
        }:
            raise RuntimeError("probe lacks the uniquely allowed repaired field")
        row.pop("finite_and_structural")
    return stripped


def validate_seed_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_label: str,
    expected_rng_id: int,
    expected_device: str,
    metadata: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> None:
    if set(artifact) != set(SEED_ARTIFACT_FIELDS):
        raise RuntimeError("repair seed artifact fields differ")
    parent_path = (
        _resolve_project_path(amendment["parent"]["root"])
        / "mechanism"
        / f"seed_{expected_rng_id}.json"
    )
    if (
        artifact.get("protocol") != REPAIR_PROTOCOL
        or artifact.get("scientific_protocol")
        != metadata["scientific_config"]["protocol"]
        or artifact.get("rng_label") != expected_label
        or artifact.get("rng_id") != expected_rng_id
        or artifact.get("device") != expected_device
        or artifact.get("amendment_sha256") != metadata["amendment_sha256"]
        or artifact.get("parent_seed_artifact_sha256")
        != base._file_sha256(parent_path)
        or artifact.get("scientific_config_sha256")
        != metadata["scientific_config_sha256"]
        or artifact.get("replay_rng_audit_sha256")
        != metadata["replay_rng_audit"]["audit_sha256"]
        or artifact.get("source_tree_sha256") != metadata["source_tree_sha256"]
    ):
        raise RuntimeError(f"repair seed artifact identity differs: {expected_label}")
    probe = artifact.get("probe")
    if not isinstance(probe, dict):
        raise RuntimeError("repair seed probe is malformed")
    base._validate_probe_payload(probe, expected_seed=expected_rng_id)
    parent_probe = base._read_json(parent_path)["probe"]
    expected_comparison = compare_probe_payloads(parent_probe, probe, amendment)
    if artifact.get("probe_comparison") != expected_comparison:
        raise RuntimeError("repair seed probe comparison differs")
    _assert_information_firewall(artifact)


def summarize_results(
    config: NativeSignedGammaBenchmarkConfig,
    results: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    mapping = replay_rng_mapping(config)
    if set(results) != set(mapping):
        raise RuntimeError("cannot summarize an incomplete repair replay")
    exact_ids = [
        mapping[label]
        for label in mapping
        if results[label]["probe_comparison"]["status"] == "EXACT_REPLAY"
        and results[label]["probe_comparison"][
            "exact_after_removing_only_repaired_field"
        ]
    ]
    repaired_ids = [
        mapping[label]
        for label in mapping
        if results[label]["probe_comparison"]["repaired_field_values_valid"]
    ]
    passed = [
        mapping[label]
        for label in mapping
        if seed_passes_mechanism_gate(dict(results[label]["probe"]), config.gate)
    ]
    required = 19
    if required != int(
        np.ceil(config.gate.minimum_available_seed_fraction * len(mapping))
    ):
        raise RuntimeError("repair GO threshold differs from the parent gate")
    if len(exact_ids) != len(mapping) or len(repaired_ids) != len(mapping):
        status = "INVALID_REPLAY"
    else:
        status = "GO" if len(passed) >= required else "NO_GO"
    return {
        "protocol": REPAIR_PROTOCOL,
        "gate_only": True,
        "administrative_only": True,
        "amendment_sha256": metadata["amendment_sha256"],
        "parent_manifest_sha256": metadata["parent_bundle"]["manifest_sha256"],
        "scientific_config_sha256": metadata["scientific_config_sha256"],
        "replay_rng_audit_sha256": metadata["replay_rng_audit"]["audit_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "n_prespecified": len(mapping),
        "n_exact_replays": len(exact_ids),
        "n_repaired_fields_valid": len(repaired_ids),
        "n_passed": len(passed),
        "passed_rng_ids": passed,
        "required_passed_rng_ids": required,
        "status": status,
        "failure_consequence": "no downstream benchmark artifacts",
        "downstream_authorized": status == "GO",
    }


def artifact_schema(amendment_sha256: str) -> dict[str, Any]:
    return {
        "protocol": REPAIR_PROTOCOL,
        "amendment_sha256": amendment_sha256,
        "metadata_fields": list(METADATA_FIELDS),
        "summary_fields": list(SUMMARY_FIELDS),
        "seed_artifact_fields": list(SEED_ARTIFACT_FIELDS),
        "comparison_rule": "remove_only_gamma_rows_finite_and_structural",
        "invalid_replay_consequence": "downstream_unauthorized",
    }


def build_metadata(
    amendment: Mapping[str, Any],
    *,
    amendment_path: Path,
    parent: Mapping[str, Any],
    config: NativeSignedGammaBenchmarkConfig,
    equivalence: Mapping[str, Any],
    rng_audit: Mapping[str, Any],
    downstream_reservation: Mapping[str, Any],
    source_hash: str,
    source_snapshot: Mapping[str, Any],
    schema: Mapping[str, Any],
    invocation_argv: Sequence[str],
) -> dict[str, Any]:
    amendment_sha = base._file_sha256(amendment_path)
    config_payload = config.to_dict()
    devices = base.seed_device_mapping(config)
    environment = base._runtime_environment()
    invocation = {
        "argv": [base._project_path(REPAIR_RUNNER), *invocation_argv],
        "cwd": str(Path.cwd().resolve()),
    }
    bindings = {
        "amendment_sha256": amendment_sha,
        "parent_manifest_sha256": parent["manifest_sha256"],
        "scientific_config_sha256": base._canonical_sha256(config_payload),
        "replay_rng_audit_sha256": rng_audit["audit_sha256"],
        "downstream_rng_reservation_sha256": downstream_reservation[
            "reservation_sha256"
        ],
        "seed_device_mapping_sha256": base._canonical_sha256(devices),
        "source_tree_sha256": source_hash,
        "environment_sha256": base._canonical_sha256(environment),
        "invocation_sha256": base._canonical_sha256(invocation),
        "artifact_schema_sha256": base._canonical_sha256(schema),
    }
    return {
        "protocol": REPAIR_PROTOCOL,
        "role": amendment["role"],
        "gate_only": True,
        "administrative_only": True,
        "amendment_path": base._project_path(amendment_path),
        "amendment_sha256": amendment_sha,
        "amendment_payload_sha256": base._canonical_sha256(amendment),
        "parent_bundle": dict(parent),
        "scientific_config": config_payload,
        "scientific_config_sha256": bindings["scientific_config_sha256"],
        "scientific_config_equivalence": dict(equivalence),
        "output_root": str(config.output_root),
        "replay_rng_audit": dict(rng_audit),
        "downstream_rng_reservation": dict(downstream_reservation),
        "seed_device_mapping": devices,
        "seed_device_mapping_sha256": bindings["seed_device_mapping_sha256"],
        "source_tree_sha256": source_hash,
        "source_snapshot": dict(source_snapshot),
        "dependency_files": {
            "native_module": base._source_contract(NATIVE_MODULE),
            "simulator": base._source_contract(ROOT / "src/scpcp/simulator.py"),
            "base_runner": base._source_contract(BASE_RUNNER),
            "repair_runner": base._source_contract(REPAIR_RUNNER),
            "downstream_runner_declaration": base._source_contract(SCIENCE_RUNNER),
            "base_config": base._source_contract(BASE_CONFIG),
            "amendment": base._source_contract(amendment_path),
            "downstream_config_declaration": base._source_contract(SCIENCE_CONFIG),
            "artifact_io": base._source_contract(ROOT / "src/scpcp/artifacts.py"),
            "project": base._source_contract(ROOT / "pyproject.toml"),
        },
        "environment": environment,
        "environment_sha256": bindings["environment_sha256"],
        "invocation": invocation,
        "invocation_sha256": bindings["invocation_sha256"],
        "artifact_schema_sha256": bindings["artifact_schema_sha256"],
        "launch_contract_sha256": base._canonical_sha256(bindings),
        "repair_contract": dict(amendment["repair"]),
        "information_firewall": {
            "gate_observables_only": True,
            "forbidden_result_fields": list(FORBIDDEN_RESULT_FIELDS),
        },
        "downstream_authorization_rule": {
            "required_status": "GO",
            "required_exact_replays": 20,
            "required_repaired_fields_valid": 20,
            "invalid_replay_authorized": False,
            "reserved_rng_execution_authorized_here": False,
        },
    }


def prepare_root(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    schema: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    amendment_path: Path,
    resume: bool,
) -> None:
    if resume:
        if not root.is_dir():
            raise FileNotFoundError("repair resume requires an existing output root")
        if base._read_json(root / "metadata.json") != metadata:
            raise RuntimeError("repair resume metadata differs from live contract")
        if base._read_json(root / "artifact_schema.json") != schema:
            raise RuntimeError("repair resume artifact schema differs")
        _verify_published_amendment(root, metadata, amendment_path)
        base._verify_source_snapshot(root, metadata["source_snapshot"])
        allowed = expected_bundle_paths(metadata) | {"manifest.json", "COMPLETE"}
        observed = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        unexpected = sorted(observed - allowed)
        if unexpected:
            raise RuntimeError(f"unexpected repair resume artifacts: {unexpected}")
        return
    if root.exists():
        raise FileExistsError(f"fresh repair output already exists: {root}")
    root.mkdir(parents=True)
    (root / "mechanism").mkdir()
    base._atomic_write(
        root / source_snapshot["contract"]["archive_path"],
        source_snapshot["archive_bytes"],
    )
    base._atomic_write(
        root / source_snapshot["contract"]["manifest_path"],
        source_snapshot["manifest_bytes"],
    )
    published_amendment = _published_amendment_path(root, metadata)
    base._atomic_write(published_amendment, amendment_path.read_bytes())
    base._write_json(root / "artifact_schema.json", schema)
    base._write_json(root / "metadata.json", metadata)
    _verify_published_amendment(root, metadata, amendment_path)
    base._verify_source_snapshot(root, metadata["source_snapshot"])


def load_existing_seed_artifacts(
    root: Path,
    *,
    mapping: Mapping[str, int],
    devices_by_label: Mapping[str, str],
    metadata: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    expected_by_id = {rng_id: label for label, rng_id in mapping.items()}
    results: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "mechanism").iterdir()):
        if not path.is_file():
            raise RuntimeError(f"unexpected repair mechanism artifact: {path.name}")
        match = base.SEED_ARTIFACT_NAME.fullmatch(path.name)
        if match is None or int(match.group(1)) not in expected_by_id:
            raise RuntimeError(f"unexpected repair seed artifact: {path.name}")
        rng_id = int(match.group(1))
        label = expected_by_id[rng_id]
        artifact = base._read_json(path)
        validate_seed_artifact(
            artifact,
            expected_label=label,
            expected_rng_id=rng_id,
            expected_device=devices_by_label[label],
            metadata=metadata,
            amendment=amendment,
        )
        if label in results:
            raise RuntimeError(f"duplicate repair RNG label: {label}")
        results[label] = artifact
    return results


def finalize_root(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    summary: Mapping[str, Any],
    amendment: Mapping[str, Any],
    amendment_path: Path,
) -> None:
    validate_parent_bundle(amendment)
    if base._experiment_tree_sha256(ROOT) != metadata["source_tree_sha256"]:
        raise RuntimeError("experiment/source tree changed during repair replay")
    if base._read_json(root / "metadata.json") != metadata:
        raise RuntimeError("repair metadata changed during replay")
    if base._read_json(root / "summary.json") != summary:
        raise RuntimeError("repair summary changed before finalization")
    write_manifest(root, metadata)
    manifest_path = root / "manifest.json"
    complete = {
        "protocol": REPAIR_PROTOCOL,
        "status": "complete",
        "decision": summary["status"],
        "downstream_authorized": summary["downstream_authorized"],
        "amendment_sha256": metadata["amendment_sha256"],
        "parent_manifest_sha256": metadata["parent_bundle"]["manifest_sha256"],
        "manifest_sha256": base._file_sha256(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "metadata_sha256": base._file_sha256(root / "metadata.json"),
        "summary_sha256": base._file_sha256(root / "summary.json"),
        "artifact_schema_sha256": base._file_sha256(root / "artifact_schema.json"),
        "scientific_config_sha256": metadata["scientific_config_sha256"],
        "replay_rng_audit_sha256": metadata["replay_rng_audit"]["audit_sha256"],
        "downstream_rng_reservation_sha256": metadata[
            "downstream_rng_reservation"
        ]["reservation_sha256"],
        "seed_device_mapping_sha256": metadata["seed_device_mapping_sha256"],
        "source_snapshot_sha256": metadata["source_snapshot"]["archive_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "environment_sha256": metadata["environment_sha256"],
        "invocation_sha256": metadata["invocation_sha256"],
        "launch_contract_sha256": metadata["launch_contract_sha256"],
    }
    base._write_json(root / "COMPLETE", complete)
    validate_completed_bundle(
        root,
        expected_metadata=metadata,
        amendment=amendment,
        amendment_path=amendment_path,
    )


def validate_completed_bundle(
    root: Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    amendment: Mapping[str, Any] | None = None,
    amendment_path: Path = DEFAULT_AMENDMENT,
) -> None:
    amendment = load_amendment(amendment_path) if amendment is None else amendment
    parent_contract = validate_parent_bundle(amendment)
    metadata = base._read_json(root / "metadata.json")
    schema = base._read_json(root / "artifact_schema.json")
    summary = base._read_json(root / "summary.json")
    if set(metadata) != set(METADATA_FIELDS) or set(summary) != set(SUMMARY_FIELDS):
        raise RuntimeError("repair metadata or summary fields differ")
    if schema != artifact_schema(metadata["amendment_sha256"]):
        raise RuntimeError("repair artifact schema differs")
    if expected_metadata is not None and metadata != expected_metadata:
        raise RuntimeError("repair metadata differs from live launch contract")
    _assert_information_firewall(metadata)
    _assert_information_firewall(summary)
    _assert_information_firewall(schema)
    if (
        metadata["amendment_sha256"] != base._file_sha256(amendment_path)
        or metadata["amendment_payload_sha256"]
        != base._canonical_sha256(amendment)
        or metadata["artifact_schema_sha256"] != base._canonical_sha256(schema)
        or metadata["output_root"] != str(root.resolve())
    ):
        raise RuntimeError("repair amendment, schema, or output binding differs")
    if metadata["parent_bundle"] != parent_contract:
        raise RuntimeError("repair parent-bundle binding differs")
    _verify_published_amendment(root, metadata, amendment_path)
    config = base._config_from_payload(metadata["scientific_config"])
    expected_config, expected_equivalence = build_effective_config(amendment)
    if (
        config.output_root.resolve() != root.resolve()
        or config.to_dict() != expected_config.to_dict()
        or metadata["scientific_config_sha256"]
        != base._canonical_sha256(config.to_dict())
        or metadata["scientific_config_equivalence"] != expected_equivalence
        or metadata["repair_contract"] != amendment["repair"]
    ):
        raise RuntimeError("repair scientific config binding differs")
    validate_replay_rng_audit(config, amendment, metadata["replay_rng_audit"])
    validate_downstream_rng_reservation(
        amendment, metadata["downstream_rng_reservation"]
    )
    expected_devices = base.seed_device_mapping(config)
    if (
        metadata["seed_device_mapping"] != expected_devices
        or metadata["seed_device_mapping_sha256"]
        != base._canonical_sha256(expected_devices)
    ):
        raise RuntimeError("repair seed-device mapping differs")
    if metadata["environment_sha256"] != base._canonical_sha256(
        metadata["environment"]
    ) or metadata["invocation_sha256"] != base._canonical_sha256(
        metadata["invocation"]
    ):
        raise RuntimeError("repair environment or invocation binding differs")
    launch_bindings = {
        "amendment_sha256": metadata["amendment_sha256"],
        "parent_manifest_sha256": metadata["parent_bundle"]["manifest_sha256"],
        "scientific_config_sha256": metadata["scientific_config_sha256"],
        "replay_rng_audit_sha256": metadata["replay_rng_audit"]["audit_sha256"],
        "downstream_rng_reservation_sha256": metadata[
            "downstream_rng_reservation"
        ]["reservation_sha256"],
        "seed_device_mapping_sha256": metadata["seed_device_mapping_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "environment_sha256": metadata["environment_sha256"],
        "invocation_sha256": metadata["invocation_sha256"],
        "artifact_schema_sha256": metadata["artifact_schema_sha256"],
    }
    if metadata["launch_contract_sha256"] != base._canonical_sha256(
        launch_bindings
    ):
        raise RuntimeError("repair launch contract differs")
    base._verify_source_snapshot(root, metadata["source_snapshot"])
    if base._experiment_tree_sha256(ROOT) != metadata["source_tree_sha256"]:
        raise RuntimeError("active source tree differs from repair replay")
    base._verify_dependency_files(metadata["dependency_files"], ROOT)

    mapping = replay_rng_mapping(config)
    results = load_existing_seed_artifacts(
        root,
        mapping=mapping,
        devices_by_label=expected_devices,
        metadata=metadata,
        amendment=amendment,
    )
    expected_summary = summarize_results(config, results, metadata)
    if summary != expected_summary:
        raise RuntimeError("repair summary does not reconcile with seed artifacts")
    actual_paths = {
        path.relative_to(root).as_posix() for path in iter_bundle_artifacts(root)
    }
    if actual_paths != expected_bundle_paths(metadata):
        raise RuntimeError("repair bundle artifact set differs")
    manifest_hash = verify_manifest(root, metadata)
    expected_complete = {
        "protocol": REPAIR_PROTOCOL,
        "status": "complete",
        "decision": summary["status"],
        "downstream_authorized": summary["downstream_authorized"],
        "amendment_sha256": metadata["amendment_sha256"],
        "parent_manifest_sha256": metadata["parent_bundle"]["manifest_sha256"],
        "manifest_sha256": manifest_hash,
        "manifest_bytes": (root / "manifest.json").stat().st_size,
        "metadata_sha256": base._file_sha256(root / "metadata.json"),
        "summary_sha256": base._file_sha256(root / "summary.json"),
        "artifact_schema_sha256": base._file_sha256(root / "artifact_schema.json"),
        "scientific_config_sha256": metadata["scientific_config_sha256"],
        "replay_rng_audit_sha256": metadata["replay_rng_audit"]["audit_sha256"],
        "downstream_rng_reservation_sha256": metadata[
            "downstream_rng_reservation"
        ]["reservation_sha256"],
        "seed_device_mapping_sha256": metadata["seed_device_mapping_sha256"],
        "source_snapshot_sha256": metadata["source_snapshot"]["archive_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "environment_sha256": metadata["environment_sha256"],
        "invocation_sha256": metadata["invocation_sha256"],
        "launch_contract_sha256": metadata["launch_contract_sha256"],
    }
    if base._read_json(root / "COMPLETE") != expected_complete:
        raise RuntimeError("repair COMPLETE hash chain differs")


def validate_completed_repair_bundle(
    root: Path,
    *,
    source_root: Path = ROOT,
    amendment_path: Path = DEFAULT_AMENDMENT,
) -> dict[str, Any]:
    """Validate a completed replay and return its downstream binding contract."""

    if source_root.resolve() != ROOT.resolve():
        raise RuntimeError("repair validator requires the active project source root")
    root = root.resolve()
    amendment_path = amendment_path.resolve()
    validate_completed_bundle(root, amendment_path=amendment_path)
    metadata = base._read_json(root / "metadata.json")
    summary = base._read_json(root / "summary.json")
    complete = base._read_json(root / "COMPLETE")
    contract = {
        "protocol": REPAIR_PROTOCOL,
        "role": metadata["role"],
        "output_root": str(root),
        "decision": summary["status"],
        "downstream_authorized": summary["downstream_authorized"],
        "amendment_sha256": metadata["amendment_sha256"],
        "parent_manifest_sha256": metadata["parent_bundle"]["manifest_sha256"],
        "scientific_config_sha256": metadata["scientific_config_sha256"],
        "replay_rng_audit_sha256": metadata["replay_rng_audit"]["audit_sha256"],
        "downstream_rng_reservation_sha256": metadata[
            "downstream_rng_reservation"
        ]["reservation_sha256"],
        "reserved_rng_mapping": metadata["downstream_rng_reservation"][
            "reserved_rng_mapping"
        ],
        "reserved_rng_mapping_sha256": metadata[
            "downstream_rng_reservation"
        ]["reserved_rng_mapping_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "source_snapshot_sha256": metadata["source_snapshot"]["archive_sha256"],
        "manifest_sha256": base._file_sha256(root / "manifest.json"),
        "complete_sha256": base._file_sha256(root / "COMPLETE"),
        "metadata_sha256": base._file_sha256(root / "metadata.json"),
        "summary_sha256": base._file_sha256(root / "summary.json"),
    }
    contract["completion_contract_sha256"] = base._canonical_sha256(contract)
    return contract


def write_manifest(root: Path, metadata: Mapping[str, Any]) -> None:
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": base._file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in iter_bundle_artifacts(root)
    ]
    base._write_json(
        root / "manifest.json",
        {
            "protocol": REPAIR_PROTOCOL,
            "amendment_sha256": metadata["amendment_sha256"],
            "parent_manifest_sha256": metadata["parent_bundle"]["manifest_sha256"],
            "scientific_config_sha256": metadata["scientific_config_sha256"],
            "replay_rng_audit_sha256": metadata["replay_rng_audit"][
                "audit_sha256"
            ],
            "source_tree_sha256": metadata["source_tree_sha256"],
            "artifact_count": len(records),
            "artifacts": records,
        },
    )


def verify_manifest(root: Path, metadata: Mapping[str, Any]) -> str:
    manifest = base._read_json(root / "manifest.json")
    expected_header = {
        "protocol": REPAIR_PROTOCOL,
        "amendment_sha256": metadata["amendment_sha256"],
        "parent_manifest_sha256": metadata["parent_bundle"]["manifest_sha256"],
        "scientific_config_sha256": metadata["scientific_config_sha256"],
        "replay_rng_audit_sha256": metadata["replay_rng_audit"]["audit_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
    }
    if any(manifest.get(key) != value for key, value in expected_header.items()):
        raise RuntimeError("repair manifest header differs")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or manifest.get("artifact_count") != len(records):
        raise RuntimeError("repair manifest records are malformed")
    actual_paths = {
        path.relative_to(root).as_posix() for path in iter_bundle_artifacts(root)
    }
    listed_paths = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RuntimeError("repair manifest record fields differ")
        relative = record["path"]
        if relative in listed_paths:
            raise RuntimeError(f"duplicate repair manifest path: {relative}")
        listed_paths.add(relative)
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or base._file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"repair manifest artifact differs: {relative}")
    if listed_paths != actual_paths:
        raise RuntimeError("repair manifest file set differs")
    return base._file_sha256(root / "manifest.json")


def iter_bundle_artifacts(root: Path) -> list[Path]:
    paths = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "COMPLETE"}:
            continue
        if ".tmp-" in path.name:
            raise RuntimeError(f"temporary repair artifact remains: {path}")
        paths.append(path)
    return paths


def expected_bundle_paths(metadata: Mapping[str, Any]) -> set[str]:
    config = base._config_from_payload(metadata["scientific_config"])
    paths = {
        "artifact_schema.json",
        "metadata.json",
        "summary.json",
        metadata["source_snapshot"]["archive_path"],
        metadata["source_snapshot"]["manifest_path"],
        _published_amendment_path(Path(metadata["output_root"]), metadata)
        .relative_to(Path(metadata["output_root"]))
        .as_posix(),
    }
    paths.update(
        f"mechanism/seed_{rng_id}.json"
        for rng_id in replay_rng_mapping(config).values()
    )
    return paths


def _published_amendment_path(root: Path, metadata: Mapping[str, Any]) -> Path:
    return (
        root
        / "provenance"
        / f"amendment_{metadata['amendment_sha256']}.yaml"
    )


def _verify_published_amendment(
    root: Path, metadata: Mapping[str, Any], amendment_path: Path
) -> None:
    published = _published_amendment_path(root, metadata)
    if (
        not published.is_file()
        or base._file_sha256(published) != metadata["amendment_sha256"]
        or published.read_bytes() != amendment_path.read_bytes()
    ):
        raise RuntimeError("published repair amendment differs")


def _assert_information_firewall(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_RESULT_FIELDS):
                location = ".".join((*path, str(key)))
                raise RuntimeError(f"forbidden result field in repair artifact: {location}")
            _assert_information_firewall(nested, (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_information_firewall(nested, (*path, str(index)))


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not base._is_relative_to(resolved, ROOT.resolve()):
        raise RuntimeError(f"repair path escapes project root: {value}")
    return resolved


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


if __name__ == "__main__":
    main()
