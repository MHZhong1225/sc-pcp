"""Exact administrative replay of the failed-publish Native science run.

The quarantined run consumed all 241 frozen streams but failed only when its
published bundle was validated.  This runner authorizes those IDs once, from
that exact byte-pinned quarantine, and requires every regenerated scientific
row, the summary, and both bootstrap arrays to match exactly.  It never prints
scientific values.

Validate the replay contract without consuming an RNG stream::

    conda run -n ucp python \
      scripts/run_native_synthetic_signed_gamma_science_exact_replay_r1.py \
      --validate-only
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_native_synthetic_signed_gamma_preflight as preflight  # noqa: E402
from scripts import run_native_synthetic_signed_gamma_science as science  # noqa: E402


INCIDENT_PROTOCOL = "native_synthetic_signed_gamma_science_failed_publish_replay_r1"
DEFAULT_INCIDENT = (
    ROOT
    / "configs/native_synthetic_signed_gamma_science_failed_publish_replay_r1.yaml"
)
DEFAULT_RETRY_CONFIG = (
    ROOT / "configs/native_synthetic_signed_gamma_science_exact_replay_r1.yaml"
)
ORIGINAL_CONFIG = ROOT / "configs/native_synthetic_signed_gamma_science.yaml"
BASE_SCIENCE_RUNNER = ROOT / "scripts/run_native_synthetic_signed_gamma_science.py"
PREFLIGHT_RUNNER = ROOT / "scripts/run_native_synthetic_signed_gamma_preflight.py"
RETRY_RUNNER = Path(__file__).resolve()
QUARANTINE_ROOT = (
    ROOT
    / "results/work/native_synthetic_signed_gamma_six_method_science_v1_FAILED_PUBLISH_VALIDATION_20260827"
)
ORIGINAL_FAILED_ROOT = (
    ROOT / "results/work/native_synthetic_signed_gamma_six_method_science_v1"
)
RETRY_ROOT = (
    ROOT
    / "results/work/native_synthetic_signed_gamma_six_method_science_v1_exact_replay_r1"
)
NO_GO_FILE = "REPLAY_NO_GO.json"
COMPARISON_FILE = "replay_comparison.json"
RNG_REPLAY_DISCLOSURE = {
    "rng_reused": True,
    "rng_fresh": False,
    "rng_independent": False,
    "rng_reuse_authority": "exact_pinned_quarantine_only",
}

RETRY_METADATA_EXTRA_FIELDS = {
    "administrative_role",
    "incident_amendment_path",
    "incident_amendment_sha256",
    "incident_amendment_payload_sha256",
    "quarantine_binding",
    "repair_gate_amendment",
    "source_delta_contract",
    "retry_config_equivalence",
    "retry_contract",
}
RETRY_SEED_EXTRA_FIELDS = {
    "incident_amendment_sha256",
    "quarantine_seed_artifact_sha256",
    "row_replay_comparison",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.validate_only:
        parser.error("--resume and --validate-only are mutually exclusive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    incident = load_incident_amendment(DEFAULT_INCIDENT)
    if args.validate_only:
        print(
            json.dumps(
                validation_payload(incident),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    run_exact_replay(incident, resume=args.resume)
    print(RETRY_ROOT)


def load_incident_amendment(path: Path = DEFAULT_INCIDENT) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "incident_protocol",
        "role",
        "administrative_only",
        "incident",
        "repair_gate",
        "allowed_source_repair",
        "retry",
    }:
        raise RuntimeError("failed-publish replay amendment fields differ")
    if (
        raw["incident_protocol"] != INCIDENT_PROTOCOL
        or raw["role"] != "administrative_exact_failed_publish_replay"
        or raw["administrative_only"] is not True
    ):
        raise RuntimeError("failed-publish replay amendment identity differs")
    _validate_incident_contract(raw["incident"])
    _validate_repair_gate_contract(raw["repair_gate"])
    _validate_source_repair_contract(raw["allowed_source_repair"])
    _validate_retry_contract(raw["retry"])
    return raw


def _validate_incident_contract(contract: object) -> None:
    if not isinstance(contract, dict) or set(contract) != {
        "original_root",
        "quarantine_root",
        "quarantine_reason",
        "failure",
        "invalid_complete_present",
        "scientific_values_generated",
        "scientific_values_human_inspected",
        "scientific_values_must_not_be_printed",
        "artifact_count",
        "total_size_bytes",
        "artifact_inventory_sha256",
        "old_source_tree_sha256",
        "old_science_runner_sha256",
        "science_config_sha256",
        "source_snapshot_sha256",
        "simulator_sha256",
        "repair_gate_binding_sha256",
        "invalid_complete_sha256",
        "manifest_sha256",
        "summary_sha256",
        "metadata_sha256",
        "artifact_inventory",
    }:
        raise RuntimeError("incident contract fields differ")
    if (
        _resolve_project_path(contract["original_root"]) != ORIGINAL_FAILED_ROOT
        or _resolve_project_path(contract["quarantine_root"]) != QUARANTINE_ROOT
        or contract["quarantine_reason"]
        != "COMPLETE_was_published_before_final_dependency_validation"
        or contract["failure"] != "simulator_dependency_omitted_from_metadata"
        or contract["invalid_complete_present"] is not True
        or contract["scientific_values_generated"] is not True
        or contract["scientific_values_human_inspected"] is not False
        or contract["scientific_values_must_not_be_printed"] is not True
        or contract["artifact_count"] != 31
        or contract["total_size_bytes"] != 10_379_251
    ):
        raise RuntimeError("incident identity or disclosure differs")
    inventory = contract["artifact_inventory"]
    _validate_inventory_records(inventory, expected_count=31)
    if contract["artifact_inventory_sha256"] != science._canonical_sha256(inventory):
        raise RuntimeError("incident inventory hash differs")
    expected_hashes = {
        "artifact_inventory_sha256": "c36da6ac2700c5da0fe907e303a7864e85466aea912d661c5a216604625a2314",
        "old_source_tree_sha256": "51bada1aebf4bc2fe07059858bb5c1af7fcf393bfd9e3e7dfa06ee842a0035f3",
        "old_science_runner_sha256": "fc0d02257b505b84cc885a562c5e477891f3973827f27952fb7d9e33570dff07",
        "science_config_sha256": "cc79e7a6b8433ddc530edc483445ab2ae176021f44dd8d914770b22e13546bdf",
        "source_snapshot_sha256": "6f8d67601d83990cf75dc1cc899601cce8f3db5421615a38596e3aa54455a16a",
        "simulator_sha256": "34460a98c24f45dfaf9b2f5e069094caafd4a12c4bec4482f8c804448bf860de",
        "repair_gate_binding_sha256": "92be89446da7933f57449336c60eebde2023f675e39d57255d5c87b70736fe35",
        "invalid_complete_sha256": "d738d5aa3a52e3e6a23667ae5b90bcc875e786172948a858a660d51f74328f0a",
        "manifest_sha256": "067347b141702bb40ac693d5e61a9ec7cc3e3f6bcb88e63f5fac0bd5ab5a7862",
        "summary_sha256": "911640d859165647d61f6060b2ca277afbc577ebe8ce8df3779306ec3c8da30e",
        "metadata_sha256": "f408cc46b4379a27f5691bffb109fe7475eabee112d4592071cb887810b97c4d",
    }
    if any(contract.get(name) != digest for name, digest in expected_hashes.items()):
        raise RuntimeError("incident pinned hashes differ")


def _validate_repair_gate_contract(contract: object) -> None:
    if not isinstance(contract, dict) or set(contract) != {
        "root",
        "decision",
        "downstream_authorized",
        "artifact_count",
        "total_size_bytes",
        "artifact_inventory_sha256",
        "metadata_sha256",
        "summary_sha256",
        "manifest_sha256",
        "complete_sha256",
        "source_tree_sha256",
        "source_snapshot_sha256",
        "amendment_sha256",
        "binding_sha256",
    }:
        raise RuntimeError("repair-gate amendment fields differ")
    if (
        _resolve_project_path(contract["root"])
        != ROOT / "results/work/native_synthetic_signed_gamma_v1_time_coordinate_repair_r1"
        or contract["decision"] != "GO"
        or contract["downstream_authorized"] is not True
        or contract["artifact_count"] != 28
        or contract["total_size_bytes"] != 3_259_967
        or contract["binding_sha256"]
        != "92be89446da7933f57449336c60eebde2023f675e39d57255d5c87b70736fe35"
    ):
        raise RuntimeError("repair-gate amendment identity differs")
    if any(
        not _is_sha256(contract[name])
        for name in (
            "artifact_inventory_sha256",
            "metadata_sha256",
            "summary_sha256",
            "manifest_sha256",
            "complete_sha256",
            "source_tree_sha256",
            "source_snapshot_sha256",
            "amendment_sha256",
            "binding_sha256",
        )
    ):
        raise RuntimeError("repair-gate amendment hash is malformed")


def _validate_source_repair_contract(contract: object) -> None:
    if not isinstance(contract, dict) or set(contract) != {
        "old_science_runner_sha256",
        "repaired_science_runner_sha256",
        "old_preflight_runner_sha256",
        "repaired_preflight_runner_sha256",
        "unchanged_science_config_sha256",
        "unchanged_simulator_sha256",
        "exactly_changed_existing_sources",
        "permitted_changes",
        "permitted_added_administrative_sources",
        "full_retry_tree_binding",
    }:
        raise RuntimeError("allowed source-repair fields differ")
    if contract != {
        "old_science_runner_sha256": "fc0d02257b505b84cc885a562c5e477891f3973827f27952fb7d9e33570dff07",
        "repaired_science_runner_sha256": "a39da352f9ed7c390bb3cef66a124b2e7d866af2da712226af6c0a0441ce6a78",
        "old_preflight_runner_sha256": "01a83d16995c3da155bb5e4f30607ac8183a4682be63eeb52b1d256b14e822ab",
        "repaired_preflight_runner_sha256": "68286680065fc80ab48e6376ac9707e5356c76d5072584cf404f543d031f1708",
        "unchanged_science_config_sha256": "cc79e7a6b8433ddc530edc483445ab2ae176021f44dd8d914770b22e13546bdf",
        "unchanged_simulator_sha256": "34460a98c24f45dfaf9b2f5e069094caafd4a12c4bec4482f8c804448bf860de",
        "exactly_changed_existing_sources": [
            "scripts/run_native_synthetic_signed_gamma_preflight.py",
            "scripts/run_native_synthetic_signed_gamma_science.py",
        ],
        "permitted_changes": [
            "add_simulator_dependency_binding",
            "validate_all_nonterminal_artifacts_before_COMPLETE",
            "remove_COMPLETE_if_postpublish_validation_fails",
            "ignore_exact_source_dependency_contracts_in_artifact_rng_scan",
            "recurse_exact_structured_rng_config_by_child_field",
        ],
        "permitted_added_administrative_sources": [
            "configs/native_synthetic_signed_gamma_science_exact_replay_r1.yaml",
            "configs/native_synthetic_signed_gamma_science_failed_publish_replay_r1.yaml",
            "scripts/run_native_synthetic_signed_gamma_science_exact_replay_r1.py",
        ],
        "full_retry_tree_binding": "launch_metadata_and_content_addressed_source_snapshot",
    }:
        raise RuntimeError("allowed source repair is not the audited patch")


def _validate_retry_contract(contract: object) -> None:
    if not isinstance(contract, dict) or set(contract) != {
        "config",
        "config_sha256",
        "runner",
        "output_root",
        "scientific_config_difference",
        "reused_rng_authority",
        "rng_reused",
        "rng_fresh",
        "rng_independent",
        "raw_collision_count",
        "authorized_quarantine_collision_count",
        "unauthorized_collision_count",
        "missing_quarantine_collision_count",
        "payload_comparison",
        "complete_publication",
    }:
        raise RuntimeError("retry amendment fields differ")
    if (
        _resolve_project_path(contract["config"]) != DEFAULT_RETRY_CONFIG
        or _resolve_project_path(contract["runner"]) != RETRY_RUNNER
        or _resolve_project_path(contract["output_root"]) != RETRY_ROOT
        or contract["config_sha256"]
        != "db03be19302274ddc14dd0e952a6c29d1dce244e8479d2e8bab262349948c8de"
        or contract["scientific_config_difference"] != "output_root_only"
        or contract["reused_rng_authority"] != "exact_pinned_quarantine_only"
        or contract["rng_reused"] is not True
        or contract["rng_fresh"] is not False
        or contract["rng_independent"] is not False
        or (
            contract["raw_collision_count"],
            contract["authorized_quarantine_collision_count"],
            contract["unauthorized_collision_count"],
            contract["missing_quarantine_collision_count"],
        )
        != (241, 241, 0, 0)
        or contract["complete_publication"]
        != "true_last_after_precommit_validation"
    ):
        raise RuntimeError("retry amendment identity differs")
    if contract["payload_comparison"] != {
        "seed_rows": "exact_canonical_json_per_seed_per_gamma",
        "summary": "exact_canonical_json",
        "bootstrap_uniforms": "exact_array_and_file_sha256",
        "bootstrap_indices": "exact_array_and_file_sha256",
        "mismatch_status": "NO_GO",
    }:
        raise RuntimeError("retry payload-comparison contract differs")


def load_retry_config(
    retry_path: Path = DEFAULT_RETRY_CONFIG,
    original_path: Path = ORIGINAL_CONFIG,
) -> tuple[science.ScienceConfig, dict[str, Any]]:
    retry_raw = yaml.safe_load(retry_path.read_text(encoding="utf-8"))
    original_raw = yaml.safe_load(original_path.read_text(encoding="utf-8"))
    if not isinstance(retry_raw, dict) or not isinstance(original_raw, dict):
        raise RuntimeError("science config payload is malformed")
    retry_without_output = dict(retry_raw)
    original_without_output = dict(original_raw)
    retry_output = retry_without_output.pop("output_root", None)
    original_output = original_without_output.pop("output_root", None)
    if (
        retry_without_output != original_without_output
        or retry_output
        != "results/work/native_synthetic_signed_gamma_six_method_science_v1_exact_replay_r1"
        or original_output
        != "results/work/native_synthetic_signed_gamma_six_method_science_v1"
    ):
        raise RuntimeError("retry science config differs beyond output_root")
    original = science.ScienceConfig.from_yaml(original_path)
    config = replace(original, output_root=Path(retry_output))
    if config.to_dict() != retry_raw:
        raise RuntimeError("effective retry science config does not match its YAML")
    equivalence = {
        "status": "exact_except_administrative_output_root",
        "only_difference": "output_root",
        "original_output_root": original_output,
        "retry_output_root": retry_output,
        "shared_scientific_payload_sha256": science._canonical_sha256(
            retry_without_output
        ),
        "original_config_file_sha256": science._file_sha256(original_path),
        "retry_config_file_sha256": science._file_sha256(retry_path),
    }
    return config, equivalence


def validate_quarantine_bundle(
    incident: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the failed root byte-for-byte without reading result values."""

    contract = incident["incident"]
    root = _resolve_project_path(contract["quarantine_root"])
    if ORIGINAL_FAILED_ROOT.exists():
        raise RuntimeError("failed science root was not exclusively quarantined")
    inventory = _bundle_inventory(root)
    if (
        inventory != contract["artifact_inventory"]
        or len(inventory) != contract["artifact_count"]
        or sum(row["size_bytes"] for row in inventory)
        != contract["total_size_bytes"]
        or science._canonical_sha256(inventory)
        != contract["artifact_inventory_sha256"]
    ):
        raise RuntimeError("quarantined failed-publish bytes differ")

    metadata = science._read_json(root / "metadata.json")
    if (
        metadata.get("protocol") != science.PROTOCOL
        or metadata.get("role") != science.ROLE
        or metadata.get("source_tree_sha256")
        != contract["old_source_tree_sha256"]
        or metadata.get("science_config_file_sha256")
        != contract["science_config_sha256"]
        or metadata.get("source_snapshot", {}).get("archive_sha256")
        != contract["source_snapshot_sha256"]
        or metadata.get("gate_binding", {}).get("binding_sha256")
        != contract["repair_gate_binding_sha256"]
    ):
        raise RuntimeError("quarantine metadata provenance differs")
    dependencies = metadata.get("dependency_files")
    if not isinstance(dependencies, dict) or "simulator" in dependencies:
        raise RuntimeError("quarantine does not preserve the audited missing dependency")

    source_manifest = science._read_json(
        root / metadata["source_snapshot"]["manifest_path"]
    )
    source_by_path = {row["path"]: row for row in source_manifest.get("files", [])}
    for path, expected_hash in {
        "scripts/run_native_synthetic_signed_gamma_science.py": contract[
            "old_science_runner_sha256"
        ],
        "configs/native_synthetic_signed_gamma_science.yaml": contract[
            "science_config_sha256"
        ],
        "src/scpcp/simulator.py": contract["simulator_sha256"],
    }.items():
        if source_by_path.get(path, {}).get("sha256") != expected_hash:
            raise RuntimeError(f"quarantine source snapshot differs at {path}")
    preflight._verify_source_snapshot(root, metadata["source_snapshot"])

    manifest = science._read_json(root / "manifest.json")
    listed = manifest.get("artifacts")
    inventory_without_terminal = [
        row for row in inventory if row["path"] not in {"manifest.json", "COMPLETE"}
    ]
    if (
        not isinstance(listed, list)
        or manifest.get("artifact_count") != len(listed)
        or listed != inventory_without_terminal
        or science._file_sha256(root / "manifest.json") != contract["manifest_sha256"]
    ):
        raise RuntimeError("quarantine manifest differs")
    invalid_complete = science._read_json(root / "COMPLETE")
    if (
        science._file_sha256(root / "COMPLETE")
        != contract["invalid_complete_sha256"]
        or invalid_complete != science._expected_complete_payload(root, metadata=metadata)
    ):
        raise RuntimeError("quarantine invalid COMPLETE bytes or internal hash chain differ")
    old_audit = metadata.get("rng_audit", {})
    old_mapping = old_audit.get("formal_rng_mapping")
    expected_mapping = science.science_rng_mapping()
    if (
        old_mapping != expected_mapping
        or old_audit.get("formal_rng_mapping_sha256") != science.FORMAL_MAPPING_SHA256
        or old_audit.get("status") != "passed_before_launch"
        or old_audit.get("collision_count") != 0
        or old_audit.get("collisions") != {}
        or old_audit.get("internal_rng_ids_unique") is not True
        or old_audit.get("formal_rng_id_count") != 241
        or len(set(old_mapping.values())) != 241
    ):
        raise RuntimeError("quarantine launch RNG audit or mapping differs")
    gate = _gate_from_quarantine_metadata(metadata, incident)
    result = {
        "status": "QUARANTINED_FAILED_PUBLISH_FULLY_VALIDATED",
        "root": contract["quarantine_root"],
        "original_root_absent": True,
        "artifact_count": len(inventory),
        "total_size_bytes": sum(row["size_bytes"] for row in inventory),
        "artifact_inventory_sha256": science._canonical_sha256(inventory),
        "metadata_sha256": science._file_sha256(root / "metadata.json"),
        "summary_sha256": science._file_sha256(root / "summary.json"),
        "manifest_sha256": science._file_sha256(root / "manifest.json"),
        "invalid_complete_sha256": science._file_sha256(root / "COMPLETE"),
        "source_tree_sha256": metadata["source_tree_sha256"],
        "source_snapshot_sha256": metadata["source_snapshot"]["archive_sha256"],
        "repair_gate_binding_sha256": gate.binding_sha256,
        "scientific_values_human_inspected": False,
    }
    result["validation_sha256"] = science._canonical_sha256(result)
    return result


def validate_repair_gate_amendment(
    incident: Mapping[str, Any],
) -> science.GateBinding:
    """Validate the old repair GO by exact bytes, not active-tree relaxation."""

    contract = incident["repair_gate"]
    root = _resolve_project_path(contract["root"])
    inventory = _bundle_inventory(root)
    if (
        len(inventory) != contract["artifact_count"]
        or sum(row["size_bytes"] for row in inventory)
        != contract["total_size_bytes"]
        or science._canonical_sha256(inventory)
        != contract["artifact_inventory_sha256"]
    ):
        raise RuntimeError("amended repair GO bundle inventory differs")
    for name, field in {
        "metadata.json": "metadata_sha256",
        "summary.json": "summary_sha256",
        "manifest.json": "manifest_sha256",
        "COMPLETE": "complete_sha256",
    }.items():
        if science._file_sha256(root / name) != contract[field]:
            raise RuntimeError(f"amended repair GO file differs: {name}")
    metadata = science._read_json(root / "metadata.json")
    summary = science._read_json(root / "summary.json")
    complete = science._read_json(root / "COMPLETE")
    if (
        metadata.get("source_tree_sha256") != contract["source_tree_sha256"]
        or metadata.get("source_snapshot", {}).get("archive_sha256")
        != contract["source_snapshot_sha256"]
        or metadata.get("amendment_sha256") != contract["amendment_sha256"]
        or summary.get("status") != "GO"
        or summary.get("downstream_authorized") is not True
        or summary.get("n_prespecified") != 20
        or summary.get("n_exact_replays") != 20
        or summary.get("n_repaired_fields_valid") != 20
        or summary.get("n_passed") != 20
        or summary.get("required_passed_rng_ids") != 19
        or complete.get("decision") != "GO"
        or complete.get("downstream_authorized") is not True
        or complete.get("manifest_sha256") != contract["manifest_sha256"]
    ):
        raise RuntimeError("amended repair GO decision or provenance differs")
    preflight._verify_source_snapshot(root, metadata["source_snapshot"])
    _verify_manifest_inventory(root)

    quarantine_metadata = science._read_json(QUARANTINE_ROOT / "metadata.json")
    gate = _gate_from_quarantine_metadata(quarantine_metadata, incident)
    if (
        gate.repair_root != contract["root"]
        or gate.decision != "GO"
        or gate.source_tree_sha256 != contract["source_tree_sha256"]
        or gate.source_snapshot_sha256 != contract["source_snapshot_sha256"]
        or gate.manifest_sha256 != contract["manifest_sha256"]
        or gate.complete_sha256 != contract["complete_sha256"]
        or gate.binding_sha256 != contract["binding_sha256"]
    ):
        raise RuntimeError("quarantine gate binding differs from amended repair GO")
    expected_files = {
        "metadata.json": contract["metadata_sha256"],
        "summary.json": contract["summary_sha256"],
        "manifest.json": contract["manifest_sha256"],
        "COMPLETE": contract["complete_sha256"],
    }
    if any(gate.files[name]["sha256"] != digest for name, digest in expected_files.items()):
        raise RuntimeError("amended repair GO file binding differs")
    return gate


def _gate_from_quarantine_metadata(
    metadata: Mapping[str, Any], incident: Mapping[str, Any]
) -> science.GateBinding:
    payload = metadata.get("gate_binding")
    if not isinstance(payload, dict):
        raise RuntimeError("quarantine gate binding is malformed")
    science._require_exact_keys(
        payload,
        set(science.GateBinding.__dataclass_fields__),
        "quarantine repair gate binding",
    )
    core = dict(payload)
    stored_hash = core.pop("binding_sha256", None)
    if (
        stored_hash != science._canonical_sha256(core)
        or stored_hash != incident["repair_gate"]["binding_sha256"]
    ):
        raise RuntimeError("quarantine repair gate binding hash differs")
    return science.GateBinding(**payload)


def validate_allowed_source_delta(
    incident: Mapping[str, Any],
) -> dict[str, Any]:
    contract = incident["allowed_source_repair"]
    quarantine_metadata = science._read_json(QUARANTINE_ROOT / "metadata.json")
    source_manifest = science._read_json(
        QUARANTINE_ROOT / quarantine_metadata["source_snapshot"]["manifest_path"]
    )
    old = {row["path"]: row for row in source_manifest["files"]}
    current_paths = preflight._experiment_paths(ROOT)
    current = {
        path.relative_to(ROOT).as_posix(): {
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": science._file_sha256(path),
        }
        for path in current_paths
    }
    changed, added, missing = validate_source_manifest_delta(old, current, contract)
    if (
        science._file_sha256(ORIGINAL_CONFIG)
        != contract["unchanged_science_config_sha256"]
        or science._file_sha256(ROOT / "src/scpcp/simulator.py")
        != contract["unchanged_simulator_sha256"]
        or science._file_sha256(DEFAULT_RETRY_CONFIG)
        != incident["retry"]["config_sha256"]
    ):
        raise RuntimeError("active source differs beyond the audited administrative repair")
    result = {
        "status": "EXACT_ALLOWED_SOURCE_DELTA",
        "old_source_tree_sha256": incident["incident"]["old_source_tree_sha256"],
        "old_source_manifest_sha256": quarantine_metadata["source_snapshot"][
            "manifest_sha256"
        ],
        "changed_existing_paths": changed,
        "added_administrative_paths": added,
        "missing_paths": missing,
        "repaired_science_runner_sha256": science._file_sha256(BASE_SCIENCE_RUNNER),
        "repaired_preflight_runner_sha256": science._file_sha256(PREFLIGHT_RUNNER),
        "active_source_tree_sha256": preflight._experiment_tree_sha256(ROOT),
    }
    result["contract_sha256"] = science._canonical_sha256(result)
    return result


def validate_source_manifest_delta(
    old: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Require the two audited repairs plus the three replay declarations."""

    changed = sorted(
        path for path in old.keys() & current.keys() if old[path] != current[path]
    )
    added = sorted(current.keys() - old.keys())
    missing = sorted(old.keys() - current.keys())
    expected_changed = sorted(contract["exactly_changed_existing_sources"])
    expected_added = sorted(contract["permitted_added_administrative_sources"])
    expected_hashes = {
        "scripts/run_native_synthetic_signed_gamma_science.py": (
            contract["old_science_runner_sha256"],
            contract["repaired_science_runner_sha256"],
        ),
        "scripts/run_native_synthetic_signed_gamma_preflight.py": (
            contract["old_preflight_runner_sha256"],
            contract["repaired_preflight_runner_sha256"],
        ),
    }
    if (
        changed != expected_changed
        or added != expected_added
        or missing
        or any(
            old[path]["sha256"] != old_hash
            or current[path]["sha256"] != repaired_hash
            for path, (old_hash, repaired_hash) in expected_hashes.items()
        )
    ):
        raise RuntimeError("active source differs beyond the audited administrative repair")
    return changed, added, missing


def audit_retry_rng_ids(
    config: science.ScienceConfig,
    incident: Mapping[str, Any],
    *,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Authorize all 241 IDs only from the exact pinned quarantine."""

    artifact_root = ROOT / "results" if artifact_root is None else artifact_root
    source_root = ROOT if source_root is None else source_root
    mapping = science.science_rng_mapping(
        config.rng.base_seeds, config.rng.bootstrap_seed
    )
    quarantine_metadata = science._read_json(QUARANTINE_ROOT / "metadata.json")
    if (
        quarantine_metadata.get("rng_audit", {}).get("formal_rng_mapping") != mapping
        or science._canonical_sha256(mapping) != science.FORMAL_MAPPING_SHA256
    ):
        raise RuntimeError("quarantine does not bind the exact retry RNG mapping")
    quarantine_ids = set(mapping.values())
    other_scan = _artifact_rng_scan_excluding(
        artifact_root,
        excluded_roots=(QUARANTINE_ROOT, _config_output_root(config)),
    )
    source_scan = preflight._source_rng_scan(
        source_root,
        excluded_paths=retry_source_declaration_exclusions(source_root),
    )
    external_sets = {
        name: set(values)
        for name, values in preflight.COORDINATED_EXTERNAL_RESERVATIONS.items()
    }
    external_ids = set().union(*external_sets.values()) if external_sets else set()
    classification = classify_retry_collisions(
        mapping,
        quarantine_ids=quarantine_ids,
        other_artifact_ids=set(other_scan["actual"]),
        source_ids=set(source_scan["actual"]),
        external_ids=external_ids,
    )
    audit = {
        "status": classification["status"],
        "policy": "exact_pinned_failed_publish_replay_exemption_v1",
        **RNG_REPLAY_DISCLOSURE,
        "seed_namespace": config.rng.seed_namespace,
        "formal_rng_mapping": mapping,
        "formal_rng_mapping_sha256": science._canonical_sha256(mapping),
        "formal_rng_id_count": len(mapping),
        "formal_rng_id_sha256": science._integer_set_sha256(mapping.values()),
        "quarantine_root": incident["incident"]["quarantine_root"],
        "quarantine_inventory_sha256": incident["incident"][
            "artifact_inventory_sha256"
        ],
        "quarantine_rng_id_count": len(quarantine_ids),
        "quarantine_rng_id_sha256": science._integer_set_sha256(quarantine_ids),
        "other_artifact_actual_rng_id_count": len(other_scan["actual"]),
        "other_artifact_actual_rng_id_sha256": science._integer_set_sha256(
            other_scan["actual"]
        ),
        "source_actual_rng_id_count": len(source_scan["actual"]),
        "source_actual_rng_id_sha256": science._integer_set_sha256(
            source_scan["actual"]
        ),
        "external_reserved_rng_id_count": len(external_ids),
        "external_reserved_rng_id_sha256": science._integer_set_sha256(external_ids),
        "raw_collision_count": classification["raw_collision_count"],
        "raw_collisions": classification["raw_collisions"],
        "authorized_quarantine_collision_count": classification[
            "authorized_quarantine_collision_count"
        ],
        "authorized_quarantine_collisions": classification[
            "authorized_quarantine_collisions"
        ],
        "unauthorized_collision_count": classification[
            "unauthorized_collision_count"
        ],
        "unauthorized_collisions": classification["unauthorized_collisions"],
        "missing_quarantine_collision_count": classification[
            "missing_quarantine_collision_count"
        ],
        "missing_quarantine_collisions": classification[
            "missing_quarantine_collisions"
        ],
        "excluded_retry_output": str(_config_output_root(config)),
        "excluded_quarantine_from_other_scan": str(QUARANTINE_ROOT),
        "excluded_source_declarations": sorted(
            str(path) for path in retry_source_declaration_exclusions(source_root)
        ),
    }
    audit["audit_sha256"] = science._canonical_sha256(audit)
    validate_retry_rng_audit(config, incident, audit)
    return audit


def classify_retry_collisions(
    mapping: Mapping[str, int],
    *,
    quarantine_ids: set[int],
    other_artifact_ids: set[int],
    source_ids: set[int],
    external_ids: set[int],
) -> dict[str, Any]:
    raw: dict[str, int] = {}
    authorized: dict[str, int] = {}
    unauthorized: dict[str, dict[str, Any]] = {}
    missing: dict[str, int] = {}
    for label, rng_id in mapping.items():
        categories = []
        if rng_id in quarantine_ids:
            categories.append("exact_pinned_quarantine")
        else:
            missing[label] = rng_id
        if rng_id in other_artifact_ids:
            categories.append("other_artifact")
        if rng_id in source_ids:
            categories.append("source_actual_use")
        if rng_id in external_ids:
            categories.append("external_reservation")
        if categories:
            raw[label] = rng_id
        if categories == ["exact_pinned_quarantine"]:
            authorized[label] = rng_id
        elif categories:
            unauthorized[label] = {"rng_id": rng_id, "categories": categories}
    passed = (
        len(raw) == len(mapping)
        and len(authorized) == len(mapping)
        and not unauthorized
        and not missing
    )
    return {
        "status": (
            "passed_with_exact_quarantine_replay_exemption"
            if passed
            else "unauthorized_failed_publish_replay"
        ),
        "raw_collision_count": len(raw),
        "raw_collisions": raw,
        "authorized_quarantine_collision_count": len(authorized),
        "authorized_quarantine_collisions": authorized,
        "unauthorized_collision_count": len(unauthorized),
        "unauthorized_collisions": unauthorized,
        "missing_quarantine_collision_count": len(missing),
        "missing_quarantine_collisions": missing,
    }


def validate_retry_rng_audit(
    config: science.ScienceConfig,
    incident: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    mapping = science.science_rng_mapping(
        config.rng.base_seeds, config.rng.bootstrap_seed
    )
    if (
        audit.get("formal_rng_mapping") != mapping
        or audit.get("formal_rng_mapping_sha256") != science._canonical_sha256(mapping)
        or audit.get("formal_rng_id_count") != len(mapping)
        or audit.get("formal_rng_id_sha256")
        != science._integer_set_sha256(mapping.values())
        or audit.get("quarantine_inventory_sha256")
        != incident["incident"]["artifact_inventory_sha256"]
        or len(mapping) != len(set(mapping.values()))
    ):
        raise RuntimeError("retry RNG audit mapping or quarantine binding differs")
    if (
        audit.get("status") != "passed_with_exact_quarantine_replay_exemption"
        or any(audit.get(key) != value for key, value in RNG_REPLAY_DISCLOSURE.items())
        or audit.get("raw_collision_count") != 241
        or audit.get("authorized_quarantine_collision_count") != 241
        or audit.get("unauthorized_collision_count") != 0
        or audit.get("unauthorized_collisions") != {}
        or audit.get("missing_quarantine_collision_count") != 0
        or audit.get("missing_quarantine_collisions") != {}
        or audit.get("raw_collisions") != mapping
        or audit.get("authorized_quarantine_collisions") != mapping
    ):
        raise RuntimeError(
            "retry RNG audit must report raw241/authorized241/unauthorized0/missing0"
        )
    unhashed = dict(audit)
    stored_hash = unhashed.pop("audit_sha256", None)
    if stored_hash != science._canonical_sha256(unhashed):
        raise RuntimeError("retry RNG audit canonical hash differs")


def retry_source_declaration_exclusions(source_root: Path = ROOT) -> set[Path]:
    relative = (
        "scripts/run_native_synthetic_signed_gamma_science.py",
        "scripts/run_native_synthetic_signed_gamma_science_exact_replay_r1.py",
        "configs/native_synthetic_signed_gamma_science.yaml",
        "configs/native_synthetic_signed_gamma_science_exact_replay_r1.yaml",
        "configs/native_synthetic_signed_gamma_science_failed_publish_replay_r1.yaml",
    )
    return {(source_root / path).resolve() for path in relative}


def validation_payload(incident: Mapping[str, Any]) -> dict[str, Any]:
    quarantine = validate_quarantine_bundle(incident)
    gate = validate_repair_gate_amendment(incident)
    config, equivalence = load_retry_config()
    source_delta = validate_allowed_source_delta(incident)
    audit = audit_retry_rng_ids(config, incident)
    return {
        "incident_protocol": INCIDENT_PROTOCOL,
        "contract_valid": True,
        "formal_retry_permitted": True,
        "administrative_exact_replay": True,
        "no_rng_consumed": True,
        "no_scientific_values_printed": True,
        "incident_amendment_sha256": science._file_sha256(DEFAULT_INCIDENT),
        "quarantine_binding": quarantine,
        "repair_gate_binding_sha256": gate.binding_sha256,
        "retry_config_equivalence": equivalence,
        "source_delta_contract": source_delta,
        "retry_rng_audit": audit,
        "retry_output_root": str(RETRY_ROOT),
        "retry_output_root_exists": RETRY_ROOT.exists(),
    }


def run_exact_replay(
    incident: Mapping[str, Any],
    *,
    resume: bool = False,
) -> None:
    quarantine = validate_quarantine_bundle(incident)
    gate = validate_repair_gate_amendment(incident)
    config, equivalence = load_retry_config()
    source_delta = validate_allowed_source_delta(incident)
    rng_audit = audit_retry_rng_ids(config, incident)
    source_hash = preflight._experiment_tree_sha256(ROOT)
    if source_hash != source_delta["active_source_tree_sha256"]:
        raise RuntimeError("active source changed after source-delta validation")
    source_snapshot = preflight._build_source_snapshot(ROOT)
    if preflight._experiment_tree_sha256(ROOT) != source_hash:
        raise RuntimeError("active source changed while building retry snapshot")
    schema = artifact_schema()
    metadata = build_metadata(
        incident,
        config=config,
        equivalence=equivalence,
        quarantine=quarantine,
        gate=gate,
        rng_audit=rng_audit,
        source_delta=source_delta,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        schema=schema,
    )
    root = _config_output_root(config)
    if resume and (root / "COMPLETE").is_file():
        try:
            validate_completed_retry_bundle(root, expected_metadata=metadata)
        except BaseException:
            (root / "COMPLETE").unlink(missing_ok=True)
            science._fsync_directory(root)
            raise
        return
    prepare_root(
        root,
        metadata=metadata,
        schema=schema,
        source_snapshot=source_snapshot,
        resume=resume,
    )
    if (root / NO_GO_FILE).exists():
        raise RuntimeError("retry root already records NO_GO and cannot resume")

    payloads = load_seed_payloads(root, config=config, metadata=metadata)
    require_exact_seed_replays(root, payloads)
    pending = tuple(seed for seed in config.rng.base_seeds if seed not in payloads)
    if pending:
        groups = [
            tuple(
                seed
                for seed in pending
                if metadata["seed_device_mapping"][str(seed)] == device
            )
            for device in config.devices
        ]
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(config.devices), mp_context=context
        ) as executor:
            futures = {
                executor.submit(science._run_seed_group, config, group, device): device
                for group, device in zip(groups, config.devices, strict=True)
                if group
            }
            for future in as_completed(futures):
                for seed, device, rows in future.result():
                    payload = make_seed_payload(
                        seed=seed,
                        device=device,
                        rows=rows,
                        config=config,
                        metadata=metadata,
                    )
                    validate_seed_payload(
                        payload,
                        config=config,
                        metadata=metadata,
                        expected_seed=seed,
                        expected_device=device,
                    )
                    science._write_json(root / "seeds" / f"seed_{seed}.json", payload)
                    payloads[seed] = payload
                    if payload["row_replay_comparison"]["status"] != "EXACT_REPLAY":
                        publish_no_go(root, stage="seed_row_comparison")
                    print(f"completed exact-replay seed {seed}", flush=True)

    if set(payloads) != set(config.rng.base_seeds):
        raise RuntimeError("retry cannot summarize an incomplete seed bank")
    uniforms, indices, bootstrap = science._ensure_bootstrap_artifacts(root, config)
    bootstrap_comparison = compare_bootstrap_arrays(
        root, uniforms=uniforms, indices=indices
    )
    if bootstrap_comparison["status"] != "EXACT_REPLAY":
        publish_no_go(root, stage="bootstrap_comparison")

    rows = [
        row
        for seed in config.rng.base_seeds
        for row in payloads[seed]["rows"]
    ]
    summary = science.summarize(
        rows,
        config=config,
        bootstrap_uniforms=uniforms,
        bootstrap_contract=bootstrap,
    )
    summary_comparison = compare_summary_payload(summary)
    if summary_comparison["status"] != "EXACT_REPLAY":
        publish_no_go(root, stage="summary_comparison")
    science._write_json(root / "summary.json", summary)

    coverage_audit = science._coverage_audit(
        payloads,
        summary=summary,
        config=config,
        bootstrap_uniforms=uniforms,
        bootstrap_contract=bootstrap,
    )
    science._write_json(root / "coverage_audit.json", coverage_audit)
    comparison = build_replay_comparison(
        payloads,
        summary_comparison=summary_comparison,
        bootstrap_comparison=bootstrap_comparison,
        metadata=metadata,
    )
    science._write_json(root / COMPARISON_FILE, comparison)
    if comparison["status"] != "EXACT_REPLAY":
        publish_no_go(root, stage="aggregate_replay_comparison")
    final_status = expected_final_status(root, config=config, metadata=metadata)
    science._write_json(root / "FINAL_STATUS.json", final_status)

    final_quarantine = validate_quarantine_bundle(incident)
    final_gate = validate_repair_gate_amendment(incident)
    final_delta = validate_allowed_source_delta(incident)
    final_audit = audit_retry_rng_ids(config, incident)
    if (
        final_quarantine != quarantine
        or final_gate != gate
        or final_delta != source_delta
        or final_audit != rng_audit
        or preflight._experiment_tree_sha256(ROOT) != source_hash
    ):
        raise RuntimeError("retry provenance changed during execution")
    finalize_root(root, metadata=metadata, config=config)


def make_seed_payload(
    *,
    seed: int,
    device: str,
    rows: Sequence[Mapping[str, Any]],
    config: science.ScienceConfig,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    quarantine_path = QUARANTINE_ROOT / "seeds" / f"seed_{seed}.json"
    quarantine_payload = science._read_json(quarantine_path)
    comparison = compare_row_payloads(
        quarantine_payload["rows"], rows, expected_seed=seed
    )
    return {
        **science._seed_contract(metadata, config),
        "incident_amendment_sha256": metadata["incident_amendment_sha256"],
        "quarantine_seed_artifact_sha256": science._file_sha256(quarantine_path),
        "row_replay_comparison": comparison,
        "seed": seed,
        "device": device,
        "rows": list(rows),
    }


def compare_row_payloads(
    quarantine_rows: object,
    retry_rows: object,
    *,
    expected_seed: int,
) -> dict[str, Any]:
    if not isinstance(quarantine_rows, list) or not isinstance(retry_rows, list):
        raise RuntimeError("row comparison requires two row lists")
    if len(quarantine_rows) != len(science.GAMMAS) or len(retry_rows) != len(
        science.GAMMAS
    ):
        raise RuntimeError("row comparison signed-gamma count differs")
    records = []
    for gamma, old, new in zip(
        science.GAMMAS, quarantine_rows, retry_rows, strict=True
    ):
        if (
            not isinstance(old, dict)
            or not isinstance(new, dict)
            or old.get("seed") != expected_seed
            or new.get("seed") != expected_seed
            or float(old.get("gamma")) != gamma
            or float(new.get("gamma")) != gamma
        ):
            raise RuntimeError("row comparison identity differs")
        old_hash = science._canonical_sha256(old)
        new_hash = science._canonical_sha256(new)
        records.append(
            {
                "gamma": gamma,
                "quarantine_row_sha256": old_hash,
                "retry_row_sha256": new_hash,
                "exact": old_hash == new_hash
                and science._canonical_json_bytes(old)
                == science._canonical_json_bytes(new),
            }
        )
    result = {
        "status": (
            "EXACT_REPLAY" if all(row["exact"] for row in records) else "NO_GO"
        ),
        "comparison_rule": "exact_canonical_json_per_seed_per_gamma",
        "seed": expected_seed,
        "row_count": len(records),
        "exact_row_count": sum(bool(row["exact"]) for row in records),
        "rows": records,
    }
    result["comparison_sha256"] = science._canonical_sha256(result)
    return result


def compare_bootstrap_arrays(
    retry_root: Path,
    *,
    uniforms: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    old_uniforms, old_indices = science._read_bootstrap_artifacts(
        QUARANTINE_ROOT, load_retry_config()[0]
    )
    uniform_record = compare_array_payloads(
        old_uniforms,
        uniforms,
        quarantine_sha256=science._file_sha256(
            QUARANTINE_ROOT / "bootstrap_uniforms.npy"
        ),
        retry_sha256=science._file_sha256(retry_root / "bootstrap_uniforms.npy"),
    )
    index_record = compare_array_payloads(
        old_indices,
        indices,
        quarantine_sha256=science._file_sha256(
            QUARANTINE_ROOT / "bootstrap_indices.npy"
        ),
        retry_sha256=science._file_sha256(retry_root / "bootstrap_indices.npy"),
    )
    result = {
        "status": (
            "EXACT_REPLAY"
            if uniform_record["exact"] and index_record["exact"]
            else "NO_GO"
        ),
        "uniforms": uniform_record,
        "indices": index_record,
    }
    result["comparison_sha256"] = science._canonical_sha256(result)
    return result


def compare_array_payloads(
    quarantine: np.ndarray,
    retry: np.ndarray,
    *,
    quarantine_sha256: str,
    retry_sha256: str,
) -> dict[str, Any]:
    return {
        "quarantine_sha256": quarantine_sha256,
        "retry_sha256": retry_sha256,
        "exact": bool(
            quarantine.dtype == retry.dtype
            and quarantine.shape == retry.shape
            and np.array_equal(quarantine, retry)
            and quarantine_sha256 == retry_sha256
        ),
    }


def compare_summary_payload(retry_summary: Mapping[str, Any]) -> dict[str, Any]:
    quarantine_summary = science._read_json(QUARANTINE_ROOT / "summary.json")
    return compare_summary_payloads(quarantine_summary, retry_summary)


def compare_summary_payloads(
    quarantine_summary: Mapping[str, Any],
    retry_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare summaries byte-canonically, without numeric tolerance or coercion."""

    old_hash = science._canonical_sha256(quarantine_summary)
    new_hash = science._canonical_sha256(retry_summary)
    exact = bool(
        old_hash == new_hash
        and science._canonical_json_bytes(quarantine_summary)
        == science._canonical_json_bytes(retry_summary)
    )
    result = {
        "status": "EXACT_REPLAY" if exact else "NO_GO",
        "comparison_rule": "exact_canonical_json",
        "quarantine_summary_sha256": old_hash,
        "retry_summary_sha256": new_hash,
        "exact": exact,
    }
    result["comparison_sha256"] = science._canonical_sha256(result)
    return result


def build_replay_comparison(
    payloads: Mapping[int, Mapping[str, Any]],
    *,
    summary_comparison: Mapping[str, Any],
    bootstrap_comparison: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    seed_comparisons = {
        str(seed): payloads[seed]["row_replay_comparison"]
        for seed in sorted(payloads)
    }
    exact_rows = sum(
        row["exact_row_count"] for row in seed_comparisons.values()
    )
    passed = bool(
        len(seed_comparisons) == 20
        and exact_rows == 100
        and all(row["status"] == "EXACT_REPLAY" for row in seed_comparisons.values())
        and summary_comparison["status"] == "EXACT_REPLAY"
        and bootstrap_comparison["status"] == "EXACT_REPLAY"
    )
    result = {
        "incident_protocol": INCIDENT_PROTOCOL,
        "status": "EXACT_REPLAY" if passed else "NO_GO",
        "incident_amendment_sha256": metadata["incident_amendment_sha256"],
        "quarantine_inventory_sha256": metadata["quarantine_binding"][
            "artifact_inventory_sha256"
        ],
        "seed_count": len(seed_comparisons),
        "row_count": sum(row["row_count"] for row in seed_comparisons.values()),
        "exact_row_count": exact_rows,
        "seed_comparison_sha256": {
            seed: comparison["comparison_sha256"]
            for seed, comparison in seed_comparisons.items()
        },
        "summary": dict(summary_comparison),
        "bootstrap": dict(bootstrap_comparison),
        "scientific_values_printed": False,
        **RNG_REPLAY_DISCLOSURE,
    }
    result["comparison_sha256"] = science._canonical_sha256(result)
    return result


def validate_seed_payload(
    payload: Mapping[str, Any],
    *,
    config: science.ScienceConfig,
    metadata: Mapping[str, Any],
    expected_seed: int,
    expected_device: str,
) -> None:
    science._require_exact_keys(
        payload,
        set(science.SEED_FIELDS) | RETRY_SEED_EXTRA_FIELDS,
        "retry seed payload",
    )
    expected_contract = science._seed_contract(metadata, config)
    for key, value in expected_contract.items():
        if payload[key] != value:
            raise RuntimeError(f"retry seed contract differs at {key}")
    if (
        payload["seed"] != expected_seed
        or payload["device"] != expected_device
        or payload["incident_amendment_sha256"]
        != metadata["incident_amendment_sha256"]
    ):
        raise RuntimeError("retry seed identity or incident binding differs")
    quarantine_path = QUARANTINE_ROOT / "seeds" / f"seed_{expected_seed}.json"
    if payload["quarantine_seed_artifact_sha256"] != science._file_sha256(
        quarantine_path
    ):
        raise RuntimeError("retry seed quarantine artifact binding differs")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(science.GAMMAS):
        raise RuntimeError("retry seed row count differs")
    for row, gamma in zip(rows, science.GAMMAS, strict=True):
        science._validate_science_row(
            row,
            config=config,
            expected_seed=expected_seed,
            expected_gamma=gamma,
        )
    quarantine_rows = science._read_json(quarantine_path)["rows"]
    expected_comparison = compare_row_payloads(
        quarantine_rows, rows, expected_seed=expected_seed
    )
    if payload["row_replay_comparison"] != expected_comparison:
        raise RuntimeError("retry seed row comparison differs")
    science._require_finite_json(payload, "retry seed payload")


def load_seed_payloads(
    root: Path,
    *,
    config: science.ScienceConfig,
    metadata: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    expected_names = {f"seed_{seed}.json" for seed in config.rng.base_seeds}
    observed_names = {path.name for path in (root / "seeds").glob("seed_*.json")}
    unexpected = sorted(observed_names - expected_names)
    if unexpected:
        raise RuntimeError(f"unexpected retry seed artifacts: {unexpected}")
    payloads = {}
    for seed in config.rng.base_seeds:
        path = root / "seeds" / f"seed_{seed}.json"
        if not path.exists():
            continue
        payload = science._read_json(path)
        validate_seed_payload(
            payload,
            config=config,
            metadata=metadata,
            expected_seed=seed,
            expected_device=metadata["seed_device_mapping"][str(seed)],
        )
        payloads[seed] = payload
    return payloads


def require_exact_seed_replays(
    root: Path, payloads: Mapping[int, Mapping[str, Any]]
) -> None:
    if any(
        payload.get("row_replay_comparison", {}).get("status") != "EXACT_REPLAY"
        for payload in payloads.values()
    ):
        publish_no_go(root, stage="seed_row_comparison")


def artifact_schema() -> dict[str, Any]:
    return {
        "protocol": "native_synthetic_signed_gamma_science_exact_replay_schema_r1",
        "metadata_fields": sorted(
            set(science.METADATA_FIELDS) | RETRY_METADATA_EXTRA_FIELDS
        ),
        "seed_fields": sorted(set(science.SEED_FIELDS) | RETRY_SEED_EXTRA_FIELDS),
        "row_fields": sorted(science.ROW_FIELDS),
        "method_common_fields": sorted(science.METHOD_COMMON_FIELDS),
        "method_adaptation_fields": sorted(science.METHOD_ADAPTATION_FIELDS),
        "method_science_fields": sorted(science.METHOD_SCIENCE_FIELDS),
        "comparison_file": COMPARISON_FILE,
        "comparison_rule": "exact_rows_summary_and_bootstrap",
        "mismatch_consequence": "NO_GO_without_COMPLETE",
        "complete_publication": "true_last_after_precommit_validation",
        "downstream_authorization_requires_exact_replay": True,
    }


def retry_contract_payload() -> dict[str, Any]:
    return {
        "incident_protocol": INCIDENT_PROTOCOL,
        "administrative_only": True,
        "exact_row_count_required": 100,
        "exact_summary_required": True,
        "exact_bootstrap_required": True,
        "no_scientific_values_printed": True,
        "complete_publication": "true_last_after_precommit_validation",
        **RNG_REPLAY_DISCLOSURE,
    }


def build_metadata(
    incident: Mapping[str, Any],
    *,
    config: science.ScienceConfig,
    equivalence: Mapping[str, Any],
    quarantine: Mapping[str, Any],
    gate: science.GateBinding,
    rng_audit: Mapping[str, Any],
    source_delta: Mapping[str, Any],
    source_hash: str,
    source_snapshot: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    config_payload = config.to_dict()
    device_mapping = science.seed_device_mapping(config)
    environment = science._runtime_environment(config.devices)
    invocation = [
        "scripts/run_native_synthetic_signed_gamma_science_exact_replay_r1.py"
    ]
    dependencies = dependency_files(config)
    amendment_sha = science._file_sha256(DEFAULT_INCIDENT)
    retry_contract = retry_contract_payload()
    core = {
        "science_config_payload_sha256": science._canonical_sha256(config_payload),
        "gate_binding_sha256": gate.binding_sha256,
        "rng_audit_sha256": rng_audit["audit_sha256"],
        "seed_device_mapping_sha256": science._canonical_sha256(device_mapping),
        "source_tree_sha256": source_hash,
        "environment_sha256": science._canonical_sha256(environment),
        "canonical_invocation_sha256": science._canonical_sha256(invocation),
        "artifact_schema_sha256": science._canonical_sha256(schema),
        "dependency_files_sha256": science._canonical_sha256(dependencies),
        "incident_amendment_sha256": amendment_sha,
        "quarantine_validation_sha256": quarantine["validation_sha256"],
        "source_delta_contract_sha256": source_delta["contract_sha256"],
        "retry_config_equivalence_sha256": science._canonical_sha256(equivalence),
        "retry_contract_sha256": science._canonical_sha256(retry_contract),
    }
    return {
        "protocol": config.protocol,
        "role": config.role,
        "administrative_role": "administrative_exact_failed_publish_replay",
        "science_config": config_payload,
        "science_config_path": DEFAULT_RETRY_CONFIG.relative_to(ROOT).as_posix(),
        "science_config_file_sha256": science._file_sha256(DEFAULT_RETRY_CONFIG),
        "science_config_payload_sha256": core["science_config_payload_sha256"],
        "science_contract": science.SCIENCE_CONTRACT,
        "gate_binding": asdict(gate),
        "rng_audit": dict(rng_audit),
        "seed_device_mapping": device_mapping,
        "seed_device_mapping_sha256": core["seed_device_mapping_sha256"],
        "source_tree_sha256": source_hash,
        "source_snapshot": dict(source_snapshot),
        "dependency_files": dependencies,
        "environment": environment,
        "environment_sha256": core["environment_sha256"],
        "canonical_invocation": invocation,
        "canonical_invocation_sha256": core["canonical_invocation_sha256"],
        "artifact_schema_sha256": core["artifact_schema_sha256"],
        "launch_contract_sha256": science._canonical_sha256(core),
        "incident_amendment_path": DEFAULT_INCIDENT.relative_to(ROOT).as_posix(),
        "incident_amendment_sha256": amendment_sha,
        "incident_amendment_payload_sha256": science._canonical_sha256(incident),
        "quarantine_binding": dict(quarantine),
        "repair_gate_amendment": dict(incident["repair_gate"]),
        "source_delta_contract": dict(source_delta),
        "retry_config_equivalence": dict(equivalence),
        "retry_contract": retry_contract,
    }


def dependency_files(config: science.ScienceConfig) -> dict[str, Any]:
    paths = {
        "base_science_runner": BASE_SCIENCE_RUNNER,
        "retry_runner": RETRY_RUNNER,
        "retry_science_config": DEFAULT_RETRY_CONFIG,
        "original_science_config": ORIGINAL_CONFIG,
        "incident_amendment": DEFAULT_INCIDENT,
        "repair_runner": ROOT / config.parent.repair_runner,
        "repair_config": ROOT / config.parent.repair_config,
        "native_dgp": ROOT / "src/scpcp/native_signed_gamma.py",
        "canonical_scpcp": ROOT / "src/scpcp/marginal_prefix.py",
        "canonical_baselines": ROOT / "src/scpcp/baselines.py",
        "scores": ROOT / "src/scpcp/scores.py",
        "simulator": ROOT / "src/scpcp/simulator.py",
        "coverage": ROOT / "src/scpcp/coverage/per_step.py",
        "experiment_rng": ROOT / "src/scpcp/experiment.py",
        "preflight_provenance": ROOT
        / "scripts/run_native_synthetic_signed_gamma_preflight.py",
        "project": ROOT / "pyproject.toml",
    }
    return {
        name: preflight._source_contract(path.resolve(), ROOT)
        for name, path in paths.items()
    }


def prepare_root(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    schema: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    resume: bool,
) -> None:
    validate_metadata(metadata, schema=schema)
    if resume:
        if not root.is_dir():
            raise FileNotFoundError("retry resume requires an existing output root")
        if science._read_json(root / "metadata.json") != metadata:
            raise RuntimeError("retry resume metadata differs from live contract")
        if science._read_json(root / "artifact_schema.json") != schema:
            raise RuntimeError("retry resume artifact schema differs")
        verify_published_incident(root, metadata)
        preflight._verify_source_snapshot(root, metadata["source_snapshot"])
        observed = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        unexpected = sorted(observed - allowed_partial_paths(metadata))
        if unexpected:
            raise RuntimeError(f"unexpected retry resume artifacts: {unexpected}")
        return
    if root.exists():
        raise FileExistsError(f"fresh retry output already exists: {root}")
    root.mkdir(parents=True)
    (root / "seeds").mkdir()
    contract = source_snapshot["contract"]
    science._atomic_write(root / contract["archive_path"], source_snapshot["archive_bytes"])
    science._atomic_write(
        root / contract["manifest_path"], source_snapshot["manifest_bytes"]
    )
    science._atomic_write(published_incident_path(root, metadata), DEFAULT_INCIDENT.read_bytes())
    science._write_json(root / "artifact_schema.json", schema)
    science._write_json(root / "metadata.json", metadata)
    verify_published_incident(root, metadata)
    preflight._verify_source_snapshot(root, metadata["source_snapshot"])


def validate_metadata(metadata: Mapping[str, Any], *, schema: Mapping[str, Any]) -> None:
    science._require_exact_keys(
        metadata,
        set(science.METADATA_FIELDS) | RETRY_METADATA_EXTRA_FIELDS,
        "retry metadata",
    )
    incident = load_incident_amendment(DEFAULT_INCIDENT)
    config, equivalence = load_retry_config()
    if (
        metadata["protocol"] != config.protocol
        or metadata["role"] != config.role
        or metadata["administrative_role"]
        != "administrative_exact_failed_publish_replay"
        or metadata["science_config"] != config.to_dict()
        or metadata["science_config_path"]
        != DEFAULT_RETRY_CONFIG.relative_to(ROOT).as_posix()
        or metadata["science_config_file_sha256"]
        != science._file_sha256(DEFAULT_RETRY_CONFIG)
        or metadata["science_config_payload_sha256"]
        != science._canonical_sha256(config.to_dict())
        or metadata["science_contract"] != science.SCIENCE_CONTRACT
        or metadata["incident_amendment_path"]
        != DEFAULT_INCIDENT.relative_to(ROOT).as_posix()
        or metadata["incident_amendment_sha256"]
        != science._file_sha256(DEFAULT_INCIDENT)
        or metadata["incident_amendment_payload_sha256"]
        != science._canonical_sha256(incident)
        or metadata["retry_config_equivalence"] != equivalence
        or metadata["repair_gate_amendment"] != incident["repair_gate"]
        or metadata["retry_contract"] != retry_contract_payload()
        or metadata["dependency_files"] != dependency_files(config)
    ):
        raise RuntimeError("retry metadata config or amendment binding differs")
    gate = science.GateBinding(**metadata["gate_binding"])
    validate_retry_rng_audit(config, incident, metadata["rng_audit"])
    expected_devices = science.seed_device_mapping(config)
    if (
        metadata["seed_device_mapping"] != expected_devices
        or metadata["seed_device_mapping_sha256"]
        != science._canonical_sha256(expected_devices)
        or metadata["environment_sha256"]
        != science._canonical_sha256(metadata["environment"])
        or metadata["canonical_invocation"]
        != ["scripts/run_native_synthetic_signed_gamma_science_exact_replay_r1.py"]
        or metadata["canonical_invocation_sha256"]
        != science._canonical_sha256(metadata["canonical_invocation"])
        or metadata["artifact_schema_sha256"] != science._canonical_sha256(schema)
    ):
        raise RuntimeError("retry metadata execution binding differs")
    core = {
        "science_config_payload_sha256": metadata["science_config_payload_sha256"],
        "gate_binding_sha256": gate.binding_sha256,
        "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
        "seed_device_mapping_sha256": metadata["seed_device_mapping_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "environment_sha256": metadata["environment_sha256"],
        "canonical_invocation_sha256": metadata["canonical_invocation_sha256"],
        "artifact_schema_sha256": metadata["artifact_schema_sha256"],
        "dependency_files_sha256": science._canonical_sha256(
            metadata["dependency_files"]
        ),
        "incident_amendment_sha256": metadata["incident_amendment_sha256"],
        "quarantine_validation_sha256": metadata["quarantine_binding"][
            "validation_sha256"
        ],
        "source_delta_contract_sha256": metadata["source_delta_contract"][
            "contract_sha256"
        ],
        "retry_config_equivalence_sha256": science._canonical_sha256(equivalence),
        "retry_contract_sha256": science._canonical_sha256(
            metadata["retry_contract"]
        ),
    }
    if metadata["launch_contract_sha256"] != science._canonical_sha256(core):
        raise RuntimeError("retry metadata launch contract differs")


def expected_final_status(
    root: Path,
    *,
    config: science.ScienceConfig,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": config.protocol,
        "incident_protocol": INCIDENT_PROTOCOL,
        "administrative_role": "administrative_exact_failed_publish_replay",
        "status": "COMPLETE",
        "decision": "SCIENCE_COMPLETE_EXACT_REPLAY",
        "downstream_authorized": True,
        "primary_gamma": science.PRIMARY_GAMMA,
        "primary_metric": science.SCIENCE_CONTRACT["coverage_metric"],
        "signed_curve_interpretation": science.SCIENCE_CONTRACT[
            "signed_curve_interpretation"
        ],
        "n_seeds": len(config.rng.base_seeds),
        "n_signed_gamma_rows": len(config.rng.base_seeds) * len(science.GAMMAS),
        "methods": list(science.METHODS),
        "gate_binding_sha256": metadata["gate_binding"]["binding_sha256"],
        "coverage_audit_sha256": science._file_sha256(root / "coverage_audit.json"),
        "incident_amendment_sha256": metadata["incident_amendment_sha256"],
        "replay_comparison_sha256": science._file_sha256(root / COMPARISON_FILE),
        "scientific_values_printed": False,
        **RNG_REPLAY_DISCLOSURE,
    }


def publish_no_go(root: Path, *, stage: str) -> None:
    complete_path = root / "COMPLETE"
    complete_path.unlink(missing_ok=True)
    science._fsync_directory(root)
    payload = {
        "incident_protocol": INCIDENT_PROTOCOL,
        "administrative_role": "administrative_exact_failed_publish_replay",
        "status": "NO_GO",
        "stage": stage,
        "downstream_authorized": False,
        "scientific_values_printed": False,
        **RNG_REPLAY_DISCLOSURE,
        "failure_consequence": "retry artifacts cannot support reporting or ranking",
    }
    payload["status_sha256"] = science._canonical_sha256(payload)
    science._write_json(root / NO_GO_FILE, payload)
    raise RuntimeError(f"exact failed-publish replay is NO_GO at {stage}")


def finalize_root(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    config: science.ScienceConfig,
) -> None:
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = expected_nonterminal_paths(metadata, config)
    if observed not in (expected, expected | {"manifest.json"}):
        raise RuntimeError("retry pre-finalization artifact set differs")
    if "manifest.json" in observed:
        verify_manifest(root, metadata=metadata, config=config)
    write_manifest(root, metadata=metadata, config=config)
    validate_retry_bundle_contents(
        root,
        expected_metadata=metadata,
        include_complete=False,
    )
    complete = expected_complete_payload(root, metadata=metadata)
    complete_path = root / "COMPLETE"
    try:
        science._write_json(complete_path, complete)
        validate_completed_retry_bundle(root, expected_metadata=metadata)
    except BaseException:
        complete_path.unlink(missing_ok=True)
        science._fsync_directory(root)
        raise


def validate_completed_retry_bundle(
    root: Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    if source_root.resolve() != ROOT.resolve():
        raise RuntimeError("retry validator requires the active project source root")
    metadata, manifest_hash = validate_retry_bundle_contents(
        root,
        expected_metadata=expected_metadata,
        include_complete=True,
    )
    complete = science._read_json(root / "COMPLETE")
    contract = {
        "incident_protocol": INCIDENT_PROTOCOL,
        "administrative_role": metadata["administrative_role"],
        "output_root": str(root.resolve()),
        "decision": complete["decision"],
        "downstream_authorized": True,
        "incident_amendment_sha256": metadata["incident_amendment_sha256"],
        "quarantine_inventory_sha256": metadata["quarantine_binding"][
            "artifact_inventory_sha256"
        ],
        "repair_gate_binding_sha256": metadata["gate_binding"]["binding_sha256"],
        "retry_rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "source_snapshot_sha256": metadata["source_snapshot"]["archive_sha256"],
        "replay_comparison_sha256": science._file_sha256(root / COMPARISON_FILE),
        **RNG_REPLAY_DISCLOSURE,
        "manifest_sha256": manifest_hash,
        "complete_sha256": science._file_sha256(root / "COMPLETE"),
    }
    contract["completion_contract_sha256"] = science._canonical_sha256(contract)
    return contract


def validate_retry_bundle_contents(
    root: Path,
    *,
    expected_metadata: Mapping[str, Any] | None,
    include_complete: bool,
) -> tuple[dict[str, Any], str]:
    incident = load_incident_amendment(DEFAULT_INCIDENT)
    config, equivalence = load_retry_config()
    if root.resolve() != _config_output_root(config):
        raise RuntimeError("retry bundle is not at the frozen output path")
    metadata = science._read_json(root / "metadata.json")
    schema = science._read_json(root / "artifact_schema.json")
    if schema != artifact_schema():
        raise RuntimeError("retry artifact schema differs")
    validate_metadata(metadata, schema=schema)
    if expected_metadata is not None and metadata != expected_metadata:
        raise RuntimeError("retry metadata differs from launch contract")
    quarantine = validate_quarantine_bundle(incident)
    gate = validate_repair_gate_amendment(incident)
    source_delta = validate_allowed_source_delta(incident)
    live_audit = audit_retry_rng_ids(config, incident)
    if (
        metadata["quarantine_binding"] != quarantine
        or metadata["gate_binding"] != asdict(gate)
        or metadata["source_delta_contract"] != source_delta
        or metadata["retry_config_equivalence"] != equivalence
        or metadata["rng_audit"] != live_audit
        or preflight._experiment_tree_sha256(ROOT) != metadata["source_tree_sha256"]
    ):
        raise RuntimeError("retry live provenance differs from metadata")
    preflight._verify_source_snapshot(root, metadata["source_snapshot"])
    preflight._verify_dependency_files(metadata["dependency_files"], ROOT)
    if metadata["dependency_files"].get("simulator", {}).get("sha256") != incident[
        "incident"
    ]["simulator_sha256"]:
        raise RuntimeError("retry simulator dependency binding differs")
    verify_published_incident(root, metadata)

    payloads = load_seed_payloads(root, config=config, metadata=metadata)
    if set(payloads) != set(config.rng.base_seeds):
        raise RuntimeError("retry bundle lacks one or more seed payloads")
    uniforms, indices = science._read_bootstrap_artifacts(root, config)
    bootstrap_comparison = compare_bootstrap_arrays(
        root, uniforms=uniforms, indices=indices
    )
    bootstrap = science._bootstrap_contract(
        config,
        uniform_path=root / "bootstrap_uniforms.npy",
        index_path=root / "bootstrap_indices.npy",
    )
    rows = [
        row
        for seed in config.rng.base_seeds
        for row in payloads[seed]["rows"]
    ]
    summary = science._read_json(root / "summary.json")
    rebuilt_summary = science.summarize(
        rows,
        config=config,
        bootstrap_uniforms=uniforms,
        bootstrap_contract=bootstrap,
    )
    if summary != rebuilt_summary:
        raise RuntimeError("retry summary does not reconcile")
    summary_comparison = compare_summary_payload(summary)
    coverage = science._read_json(root / "coverage_audit.json")
    rebuilt_coverage = science._coverage_audit(
        payloads,
        summary=summary,
        config=config,
        bootstrap_uniforms=uniforms,
        bootstrap_contract=bootstrap,
    )
    if coverage != rebuilt_coverage:
        raise RuntimeError("retry coverage audit differs")
    comparison = build_replay_comparison(
        payloads,
        summary_comparison=summary_comparison,
        bootstrap_comparison=bootstrap_comparison,
        metadata=metadata,
    )
    if (
        comparison["status"] != "EXACT_REPLAY"
        or science._read_json(root / COMPARISON_FILE) != comparison
    ):
        raise RuntimeError("retry scientific payload comparison is not exact")
    if science._read_json(root / "FINAL_STATUS.json") != expected_final_status(
        root, config=config, metadata=metadata
    ):
        raise RuntimeError("retry final status differs")
    require_artifact_set(root, metadata=metadata, config=config, include_complete=include_complete)
    manifest_hash = verify_manifest(root, metadata=metadata, config=config)
    if include_complete:
        complete = science._read_json(root / "COMPLETE")
        expected_complete = expected_complete_payload(root, metadata=metadata)
        if complete != expected_complete or complete["manifest_sha256"] != manifest_hash:
            raise RuntimeError("retry COMPLETE hash chain differs")
    return metadata, manifest_hash


def expected_complete_payload(
    root: Path, *, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "protocol": metadata["protocol"],
        "incident_protocol": INCIDENT_PROTOCOL,
        "administrative_role": metadata["administrative_role"],
        "status": "complete",
        "decision": "SCIENCE_COMPLETE_EXACT_REPLAY",
        "downstream_authorized": True,
        "manifest_sha256": science._file_sha256(root / "manifest.json"),
        "manifest_bytes": (root / "manifest.json").stat().st_size,
        "metadata_sha256": science._file_sha256(root / "metadata.json"),
        "summary_sha256": science._file_sha256(root / "summary.json"),
        "coverage_audit_sha256": science._file_sha256(root / "coverage_audit.json"),
        "final_status_sha256": science._file_sha256(root / "FINAL_STATUS.json"),
        "artifact_schema_sha256": science._file_sha256(root / "artifact_schema.json"),
        "science_config_payload_sha256": metadata["science_config_payload_sha256"],
        "gate_binding_sha256": metadata["gate_binding"]["binding_sha256"],
        "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
        "rng_mapping_sha256": metadata["rng_audit"]["formal_rng_mapping_sha256"],
        "seed_device_mapping_sha256": metadata["seed_device_mapping_sha256"],
        "source_snapshot_sha256": metadata["source_snapshot"]["archive_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "launch_contract_sha256": metadata["launch_contract_sha256"],
        "bootstrap_uniforms_sha256": science._file_sha256(
            root / "bootstrap_uniforms.npy"
        ),
        "bootstrap_indices_sha256": science._file_sha256(
            root / "bootstrap_indices.npy"
        ),
        "incident_amendment_sha256": metadata["incident_amendment_sha256"],
        "quarantine_inventory_sha256": metadata["quarantine_binding"][
            "artifact_inventory_sha256"
        ],
        "replay_comparison_sha256": science._file_sha256(root / COMPARISON_FILE),
        **RNG_REPLAY_DISCLOSURE,
    }


def write_manifest(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    config: science.ScienceConfig,
) -> None:
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": science._file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in iter_bundle_artifacts(root)
    ]
    science._write_json(
        root / "manifest.json",
        {
            "protocol": metadata["protocol"],
            "incident_protocol": INCIDENT_PROTOCOL,
            "incident_amendment_sha256": metadata["incident_amendment_sha256"],
            "quarantine_inventory_sha256": metadata["quarantine_binding"][
                "artifact_inventory_sha256"
            ],
            "science_config_payload_sha256": metadata[
                "science_config_payload_sha256"
            ],
            "gate_binding_sha256": metadata["gate_binding"]["binding_sha256"],
            "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
            "source_tree_sha256": metadata["source_tree_sha256"],
            "artifact_count": len(records),
            "artifacts": records,
        },
    )


def verify_manifest(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    config: science.ScienceConfig,
) -> str:
    del config
    manifest = science._read_json(root / "manifest.json")
    header = {
        "protocol": metadata["protocol"],
        "incident_protocol": INCIDENT_PROTOCOL,
        "incident_amendment_sha256": metadata["incident_amendment_sha256"],
        "quarantine_inventory_sha256": metadata["quarantine_binding"][
            "artifact_inventory_sha256"
        ],
        "science_config_payload_sha256": metadata["science_config_payload_sha256"],
        "gate_binding_sha256": metadata["gate_binding"]["binding_sha256"],
        "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
    }
    if any(manifest.get(key) != value for key, value in header.items()):
        raise RuntimeError("retry manifest header differs")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or manifest.get("artifact_count") != len(records):
        raise RuntimeError("retry manifest records are malformed")
    actual = {
        path.relative_to(root).as_posix() for path in iter_bundle_artifacts(root)
    }
    listed = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RuntimeError("retry manifest record fields differ")
        relative = record["path"]
        if relative in listed:
            raise RuntimeError(f"duplicate retry manifest path: {relative}")
        listed.add(relative)
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or science._file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"retry manifest artifact differs: {relative}")
    if listed != actual:
        raise RuntimeError("retry manifest file set differs")
    return science._file_sha256(root / "manifest.json")


def iter_bundle_artifacts(root: Path) -> list[Path]:
    paths = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "COMPLETE"}:
            continue
        if ".tmp-" in path.name:
            raise RuntimeError(f"temporary retry artifact remains: {path}")
        paths.append(path)
    return paths


def expected_nonterminal_paths(
    metadata: Mapping[str, Any], config: science.ScienceConfig
) -> set[str]:
    paths = {
        "artifact_schema.json",
        "metadata.json",
        "bootstrap_uniforms.npy",
        "bootstrap_indices.npy",
        "summary.json",
        "coverage_audit.json",
        "FINAL_STATUS.json",
        COMPARISON_FILE,
        metadata["source_snapshot"]["archive_path"],
        metadata["source_snapshot"]["manifest_path"],
        published_incident_path(config.output_root, metadata)
        .relative_to(config.output_root)
        .as_posix(),
    }
    paths.update(f"seeds/seed_{seed}.json" for seed in config.rng.base_seeds)
    return paths


def allowed_partial_paths(metadata: Mapping[str, Any]) -> set[str]:
    config, _ = load_retry_config()
    return expected_nonterminal_paths(metadata, config) | {
        "manifest.json",
        "COMPLETE",
        NO_GO_FILE,
    }


def require_artifact_set(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    config: science.ScienceConfig,
    include_complete: bool,
) -> None:
    expected = expected_nonterminal_paths(metadata, config) | {"manifest.json"}
    if include_complete:
        expected.add("COMPLETE")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise RuntimeError("retry completed artifact set differs")


def published_incident_path(root: Path, metadata: Mapping[str, Any]) -> Path:
    return (
        root
        / "provenance"
        / f"failed_publish_incident_{metadata['incident_amendment_sha256']}.yaml"
    )


def verify_published_incident(root: Path, metadata: Mapping[str, Any]) -> None:
    path = published_incident_path(root, metadata)
    if (
        not path.is_file()
        or path.read_bytes() != DEFAULT_INCIDENT.read_bytes()
        or science._file_sha256(path) != metadata["incident_amendment_sha256"]
    ):
        raise RuntimeError("published failed-publish incident differs")


def _artifact_rng_scan_excluding(
    root: Path,
    *,
    excluded_roots: Sequence[Path],
) -> dict[str, Any]:
    report = preflight._empty_rng_scan()
    if not root.exists():
        return report
    root_resolved = root.resolve()
    excluded = tuple(path.resolve() for path in excluded_roots)
    for path in sorted(root.rglob("*")):
        resolved = path.resolve()
        if not preflight._is_relative_to(resolved, root_resolved):
            raise RuntimeError(f"artifact scan path escapes its root: {path}")
        if any(preflight._is_relative_to(resolved, value) for value in excluded):
            continue
        match = preflight.SEED_ARTIFACT_NAME.fullmatch(path.name)
        if match:
            report["actual"].add(int(match.group(1)))
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in preflight.STRUCTURED_ARTIFACT_SUFFIXES or path.name == "COMPLETE":
            payload = preflight._read_structured_artifact(path)
            preflight._collect_artifact_rng_fields(
                payload,
                report,
                artifact_path=path,
                artifact_root=root_resolved,
            )
        elif suffix in preflight.TABULAR_ARTIFACT_SUFFIXES:
            preflight._collect_tabular_rng_fields(path, report)
    report["binary_bindings"] = sorted(
        report["binary_bindings"],
        key=lambda row: (row["metadata_path"], row["field_path"]),
    )
    return report


def _bundle_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise RuntimeError(f"required bundle is missing: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        resolved = path.resolve()
        if not preflight._is_relative_to(resolved, root.resolve()):
            raise RuntimeError(f"bundle path escapes root: {path}")
        if not path.is_file():
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": science._file_sha256(path),
            }
        )
    return records


def _validate_inventory_records(value: object, *, expected_count: int) -> None:
    if not isinstance(value, list) or len(value) != expected_count:
        raise RuntimeError("pinned inventory count differs")
    paths = []
    for record in value:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size_bytes", "sha256"}
            or not isinstance(record["path"], str)
            or not isinstance(record["size_bytes"], int)
            or record["size_bytes"] < 0
            or not _is_sha256(record["sha256"])
        ):
            raise RuntimeError("pinned inventory record is malformed")
        paths.append(record["path"])
    if paths != sorted(set(paths)):
        raise RuntimeError("pinned inventory paths are not unique and sorted")


def _verify_manifest_inventory(root: Path) -> None:
    manifest = science._read_json(root / "manifest.json")
    records = manifest.get("artifacts")
    actual = [
        row
        for row in _bundle_inventory(root)
        if row["path"] not in {"manifest.json", "COMPLETE"}
    ]
    if not isinstance(records, list) or records != actual:
        raise RuntimeError("bundle manifest inventory differs")


def _resolve_project_path(value: object) -> Path:
    if not isinstance(value, str):
        raise RuntimeError("project path must be a string")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not preflight._is_relative_to(resolved, ROOT.resolve()):
        raise RuntimeError(f"project path escapes workspace: {value}")
    return resolved


def _config_output_root(config: science.ScienceConfig) -> Path:
    path = config.output_root
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if resolved != RETRY_ROOT.resolve():
        raise RuntimeError("retry config output root differs from the frozen retry root")
    return resolved


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


if __name__ == "__main__":
    main()
