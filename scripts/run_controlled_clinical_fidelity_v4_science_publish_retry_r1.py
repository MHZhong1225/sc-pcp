"""Publish the completed clinical-v4 science bundle without rerunning science.

Read-only validation (no scientific RNG and no metric output)::

    python scripts/run_controlled_clinical_fidelity_v4_science_publish_retry_r1.py \
      validate-only

Publish the pinned 154-file bundle into the dedicated administrative root::

    python scripts/run_controlled_clinical_fidelity_v4_science_publish_retry_r1.py \
      publish --audit-go-sha256 <validate-only audit_go_sha256>

Append ``--resume`` only to continue that exact administrative byte-copy.  The
original failed-publish root is immutable, and root ``COMPLETE`` is written only
after the copied scientific bytes, retry provenance, manifest, and semantic
invariants have all passed precommit validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_controlled_clinical_fidelity_v4_science as science  # noqa: E402


RETRY_PROTOCOL = (
    "controlled_clinical_fidelity_v4_signed_gamma_science_publish_retry_r1"
)
RETRY_ROLE = "administrative_exact_byte_publish_retry"
CONFIG_PATH = (
    ROOT
    / "configs/controlled_clinical_fidelity_v4_science_publish_retry_r1.yaml"
)
AMENDMENT_PATH = (
    ROOT
    / "configs/controlled_clinical_fidelity_v4_science_publish_retry_r1_amendment.json"
)
ORIGINAL_ROOT = (
    ROOT / "results/work/controlled_clinical_fidelity_v4_signed_gamma_science"
).resolve()
OUTPUT_ROOT = (
    ROOT
    / "results/work/controlled_clinical_fidelity_v4_signed_gamma_science_publish_retry_r1"
).resolve()
RUNNER_PATH = Path(__file__).resolve()

EXPECTED_CONFIG_SHA256 = (
    "6ed3fa542c0aec9e74fc177d219f48b10c7186cc2b06b3c898119c33d7aad76f"
)
EXPECTED_AMENDMENT_SHA256 = (
    "23e952122e70431bf0304ae65bdd022b3ff8caa3aa7b5ba36c7086b7797726f7"
)
EXPECTED_ORIGINAL_SOURCE_TREE_SHA256 = (
    "67db90c1a310ee7bacdd91e82204c931f00fd69dd04853844bd52991e204706f"
)
EXPECTED_ORIGINAL_INVENTORY_SHA256 = (
    "dfc88c08a2f55f73c0c0ad50a06d196b64778577dd6ff8ce6f63107fa5394771"
)
EXPECTED_ARCHIVED_SCIENCE_RUNNER_SHA256 = (
    "b9b910137439fa18a61f59aecd9437f7452883218495dcaed55d3de79c45d709"
)
EXPECTED_REPAIRED_SCIENCE_RUNNER_SHA256 = (
    "d329537ccbf53040fb87c1707e01569266dea0f3c81ac273dc9dad6ab4ef74e6"
)

ADMINISTRATIVE_RECORD_PATH = Path("administrative_publish_retry.json")
PUBLISHED_AMENDMENT_PATH = Path("provenance/publish_retry_amendment.json")
PUBLISHED_CONFIG_PATH = Path("provenance/publish_retry_config.yaml")
MANIFEST_PATH = Path("manifest.json")
COMPLETE_PATH = Path("COMPLETE")
ADMINISTRATIVE_PATHS = (
    ADMINISTRATIVE_RECORD_PATH,
    PUBLISHED_AMENDMENT_PATH,
    PUBLISHED_CONFIG_PATH,
)


@dataclass(frozen=True)
class PublishRetryConfig:
    original_root: Path
    output_root: Path
    amendment_path: Path
    amendment_sha256: str
    devices: tuple[str, ...]
    publication: Mapping[str, Any]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("publish", "validate-only"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-go-sha256")
    args = parser.parse_args(argv)
    if args.phase == "validate-only":
        if args.resume or args.audit_go_sha256 is not None:
            parser.error("validate-only does not accept publish options")
    elif not _is_sha256(args.audit_go_sha256):
        parser.error("publish requires --audit-go-sha256 from validate-only")

    config = load_retry_config()
    amendment = load_retry_amendment(config)
    if args.phase == "validate-only":
        result = validate_only(config, amendment)
    else:
        publish_retry(
            config,
            amendment,
            independent_audit_go_sha256=args.audit_go_sha256,
            resume=args.resume,
        )
        result = {
            "protocol": RETRY_PROTOCOL,
            "status": "ADMINISTRATIVE_PUBLISH_COMPLETE",
            "output_root": str(config.output_root),
            "formal_science_executed": False,
            "rng_streams_executed": 0,
            "independent_audit_go_sha256": args.audit_go_sha256,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


def load_retry_config(path: Path = CONFIG_PATH) -> PublishRetryConfig:
    if _file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("publish-retry config bytes differ")
    raw = yaml.safe_load(path.read_text())
    _require_exact_keys(
        raw,
        {
            "protocol",
            "role",
            "administrative_only",
            "original_root",
            "output_root",
            "amendment_path",
            "amendment_sha256",
            "devices",
            "publication",
        },
        "publish-retry config",
    )
    publication = raw["publication"]
    _require_exact_keys(
        publication,
        {
            "operation",
            "formal_science_execution_permitted",
            "rng_streams_executed",
            "scientific_values_may_be_logged",
            "scientific_artifact_count",
            "scientific_total_bytes",
            "scientific_inventory_sha256",
            "scientific_paths_and_bytes_must_match_original",
            "original_root_must_remain_immutable",
            "semantic_validation_uses_canonical_json_boundary",
            "independent_audit_go_required",
            "fresh_science_claimed",
            "independent_science_claimed",
            "manifest_path",
            "root_complete_path",
            "root_complete_true_last",
            "administrative_artifact_paths",
        },
        "publish-retry publication config",
    )
    expected_publication = {
        "operation": "exact_byte_copy_then_administrative_publish",
        "formal_science_execution_permitted": False,
        "rng_streams_executed": 0,
        "scientific_values_may_be_logged": False,
        "scientific_artifact_count": 154,
        "scientific_total_bytes": 24_426_407,
        "scientific_inventory_sha256": EXPECTED_ORIGINAL_INVENTORY_SHA256,
        "scientific_paths_and_bytes_must_match_original": True,
        "original_root_must_remain_immutable": True,
        "semantic_validation_uses_canonical_json_boundary": True,
        "independent_audit_go_required": True,
        "fresh_science_claimed": False,
        "independent_science_claimed": False,
        "manifest_path": MANIFEST_PATH.as_posix(),
        "root_complete_path": COMPLETE_PATH.as_posix(),
        "root_complete_true_last": True,
        "administrative_artifact_paths": [
            path.as_posix() for path in ADMINISTRATIVE_PATHS
        ],
    }
    if (
        raw["protocol"] != RETRY_PROTOCOL
        or raw["role"] != RETRY_ROLE
        or raw["administrative_only"] is not True
        or publication != expected_publication
        or raw["amendment_sha256"] != EXPECTED_AMENDMENT_SHA256
        or tuple(raw["devices"]) != ("cuda:0", "cuda:1")
    ):
        raise RuntimeError("publish-retry config contract differs")

    config = PublishRetryConfig(
        original_root=(ROOT / raw["original_root"]).resolve(),
        output_root=(ROOT / raw["output_root"]).resolve(),
        amendment_path=(ROOT / raw["amendment_path"]).resolve(),
        amendment_sha256=raw["amendment_sha256"],
        devices=tuple(raw["devices"]),
        publication=publication,
    )
    if (
        config.original_root != ORIGINAL_ROOT
        or config.output_root != OUTPUT_ROOT
        or config.amendment_path != AMENDMENT_PATH
    ):
        raise RuntimeError("publish-retry paths differ from the frozen contract")
    return config


def load_retry_amendment(
    config: PublishRetryConfig,
) -> dict[str, Any]:
    if (
        _file_sha256(config.amendment_path) != config.amendment_sha256
        or config.amendment_sha256 != EXPECTED_AMENDMENT_SHA256
    ):
        raise RuntimeError("publish-retry amendment bytes differ")
    amendment = _read_json(config.amendment_path)
    _validate_amendment(amendment)
    return amendment


def _validate_amendment(amendment: Mapping[str, Any]) -> None:
    _require_exact_keys(
        amendment,
        {
            "protocol",
            "role",
            "administrative_only",
            "incident",
            "authorized_source_delta",
            "retry",
        },
        "publish-retry amendment",
    )
    if (
        amendment["protocol"] != RETRY_PROTOCOL
        or amendment["role"] != RETRY_ROLE
        or amendment["administrative_only"] is not True
    ):
        raise RuntimeError("publish-retry amendment identity differs")

    incident = amendment["incident"]
    required_incident_fields = {
        "original_root",
        "failure_class",
        "failure_stage",
        "formal_science_completed",
        "scientific_values_human_inspected_for_retry",
        "scientific_values_must_not_be_printed",
        "missing_root_commits",
        "present_file_count",
        "present_total_bytes",
        "present_inventory_sha256",
        "source_tree_sha256",
        "metadata_file_sha256",
        "metadata_canonical_sha256",
        "reconstructed_metadata_canonical_sha256",
        "final_status_file_sha256",
        "final_status_canonical_sha256",
        "gate_contract_sha256",
        "confirmation_binding_sha256",
        "source_manifest_sha256",
        "source_snapshot_sha256",
        "archived_science_runner_sha256",
        "mismatch_count",
        "mismatch_kind",
        "mismatch_paths",
        "phase_inventories",
        "science_scope",
        "rng_binding",
        "dataset_artifact_sha256",
        "scientific_artifact_inventory",
    }
    _require_exact_keys(incident, required_incident_fields, "incident amendment")
    inventory = _validate_inventory_rows(incident["scientific_artifact_inventory"])
    expected_mismatch_paths = [
        f"gate_contract.data_contracts.{dataset}.active_config.{field}"
        for dataset in science.CONFIRMED_DATASETS
        for field in (
            "cot.hidden_dims",
            "devices",
            "policy.action_costs",
            "seeds",
        )
    ]
    if (
        incident["original_root"]
        != "results/work/controlled_clinical_fidelity_v4_signed_gamma_science"
        or incident["failure_class"]
        != "metadata_json_roundtrip_representation_mismatch"
        or incident["formal_science_completed"] is not True
        or incident["scientific_values_human_inspected_for_retry"] is not False
        or incident["scientific_values_must_not_be_printed"] is not True
        or incident["missing_root_commits"] != ["manifest.json", "COMPLETE"]
        or incident["present_file_count"] != 154
        or incident["present_total_bytes"] != 24_426_407
        or incident["present_inventory_sha256"]
        != EXPECTED_ORIGINAL_INVENTORY_SHA256
        or incident["source_tree_sha256"]
        != EXPECTED_ORIGINAL_SOURCE_TREE_SHA256
        or incident["metadata_canonical_sha256"]
        != incident["reconstructed_metadata_canonical_sha256"]
        or incident["archived_science_runner_sha256"]
        != EXPECTED_ARCHIVED_SCIENCE_RUNNER_SHA256
        or incident["mismatch_count"] != 12
        or incident["mismatch_paths"] != expected_mismatch_paths
        or len(inventory) != 154
        or sum(row["bytes"] for row in inventory) != 24_426_407
        or _inventory_sha256(inventory) != EXPECTED_ORIGINAL_INVENTORY_SHA256
    ):
        raise RuntimeError("incident amendment binding differs")
    _validate_phase_inventory_bindings(incident, inventory)
    _validate_critical_artifact_bindings(incident, inventory)

    source_delta = amendment["authorized_source_delta"]
    _require_exact_keys(
        source_delta,
        {
            "changed_existing_executable_sources",
            "added_administrative_executable_sources",
            "added_or_changed_non_experiment_tree_sources",
            "science_contract_changed",
            "dgp_changed",
            "seeds_changed",
            "rng_mapping_changed",
            "scientific_results_changed",
        },
        "authorized source delta",
    )
    changed = source_delta["changed_existing_executable_sources"]
    if (
        changed
        != [
            {
                "path": "scripts/run_controlled_clinical_fidelity_v4_science.py",
                "archived_sha256": EXPECTED_ARCHIVED_SCIENCE_RUNNER_SHA256,
                "repaired_sha256": EXPECTED_REPAIRED_SCIENCE_RUNNER_SHA256,
                "only_change": (
                    "canonical JSON roundtrip at _science_metadata return boundary"
                ),
                "strict_equality_checks_preserved": True,
            }
        ]
        or source_delta["added_administrative_executable_sources"]
        != [
            "configs/controlled_clinical_fidelity_v4_science_publish_retry_r1.yaml",
            "scripts/run_controlled_clinical_fidelity_v4_science_publish_retry_r1.py",
        ]
        or any(
            source_delta[field] is not False
            for field in (
                "science_contract_changed",
                "dgp_changed",
                "seeds_changed",
                "rng_mapping_changed",
                "scientific_results_changed",
            )
        )
    ):
        raise RuntimeError("authorized source delta differs")

    retry = amendment["retry"]
    _require_exact_keys(
        retry,
        {
            "output_root",
            "operation",
            "scientific_relative_paths_preserved",
            "scientific_bytes_preserved",
            "scientific_artifact_count",
            "scientific_inventory_sha256",
            "semantic_validation",
            "formal_science_execution_permitted",
            "coverage_value_logging_permitted",
            "fresh_science_claimed",
            "independent_science_claimed",
            "launch_requires_exact_read_only_audit_sha256",
            "root_complete_true_last",
            "old_root_mutation_permitted",
        },
        "publish-retry authorization",
    )
    if (
        retry["output_root"]
        != "results/work/controlled_clinical_fidelity_v4_signed_gamma_science_publish_retry_r1"
        or retry["operation"] != "exact_byte_copy_then_administrative_publish"
        or retry["scientific_artifact_count"] != 154
        or retry["scientific_inventory_sha256"]
        != EXPECTED_ORIGINAL_INVENTORY_SHA256
        or retry["formal_science_execution_permitted"] is not False
        or retry["coverage_value_logging_permitted"] is not False
        or retry["fresh_science_claimed"] is not False
        or retry["independent_science_claimed"] is not False
        or retry["launch_requires_exact_read_only_audit_sha256"] is not True
        or retry["root_complete_true_last"] is not True
        or retry["old_root_mutation_permitted"] is not False
    ):
        raise RuntimeError("publish-retry authorization differs")


def _validate_phase_inventory_bindings(
    incident: Mapping[str, Any],
    inventory: tuple[dict[str, Any], ...],
) -> None:
    prefixes = {
        "eligibility": "eligibility/",
        "donor_overlap_all": "donor_overlap/",
        "donor_overlap_mimic_iv": "donor_overlap/mimic_iv/",
        "donor_overlap_eicu": "donor_overlap/eicu/",
        "donor_overlap_inspire": "donor_overlap/inspire/",
        "science_all": "science/",
        "science_mimic_iv": "science/mimic_iv/",
        "science_eicu": "science/eicu/",
        "science_inspire": "science/inspire/",
    }
    observed = {}
    for label, prefix in prefixes.items():
        rows = tuple(row for row in inventory if row["path"].startswith(prefix))
        binding: dict[str, Any] = {
            "files": len(rows),
            "bytes": sum(row["bytes"] for row in rows),
            "inventory_sha256": _inventory_sha256(rows),
        }
        seed_files = sum(
            "/seed_" in row["path"] and row["path"].endswith(".json")
            for row in rows
        )
        if seed_files:
            binding["seed_files"] = seed_files
        observed[label] = binding
    if incident["phase_inventories"] != observed:
        raise RuntimeError("phase inventory bindings differ")


def _validate_critical_artifact_bindings(
    incident: Mapping[str, Any],
    inventory: tuple[dict[str, Any], ...],
) -> None:
    by_path = {row["path"]: row["sha256"] for row in inventory}
    if (
        by_path["metadata.json"] != incident["metadata_file_sha256"]
        or by_path["FINAL_STATUS.json"] != incident["final_status_file_sha256"]
        or by_path[
            "provenance/"
            "source_manifest_94f7e7c4c8bf0d9404d06453e2018562bfcf9f7c49c8f0470280d7f89193d7f3.json"
        ]
        != incident["source_manifest_sha256"]
        or by_path[
            "provenance/"
            "source_snapshot_0f22e9ac134a12e61ad607bc5cf24ae4261e19e61fb6739132dc6ee01a575782.tar"
        ]
        != incident["source_snapshot_sha256"]
    ):
        raise RuntimeError("critical incident artifact binding differs")
    suffixes = {
        "bootstrap_uniforms": "bootstrap_uniforms.npy",
        "bootstrap_indices": "bootstrap_indices.npy",
        "summary": "summary.json",
        "coverage_audit": "coverage_audit.json",
        "final_status": "FINAL_STATUS.json",
    }
    for dataset, bindings in incident["dataset_artifact_sha256"].items():
        for label, suffix in suffixes.items():
            path = f"science/{dataset}/{suffix}"
            if by_path.get(path) != bindings[label]:
                raise RuntimeError(f"{dataset} {label} binding differs")


def validate_only(
    config: PublishRetryConfig,
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    before = validate_original_root(config, amendment)
    try:
        source_delta = validate_authorized_source_delta(config, amendment)
        if not config.output_root.exists():
            return {
                "protocol": RETRY_PROTOCOL,
                "status": "READ_ONLY_INCIDENT_VALID",
                "published_root_present": False,
                "formal_science_executed": False,
                "rng_streams_executed": 0,
                "scientific_artifact_count": len(before),
                "scientific_inventory_sha256": _inventory_sha256(before),
                "authorized_source_delta_sha256": _canonical_sha256(source_delta),
                "audit_go_sha256": _canonical_sha256(source_delta),
            }
        validate_published_retry(
            config,
            amendment,
            source_delta=source_delta,
        )
        return {
            "protocol": RETRY_PROTOCOL,
            "status": "PUBLISHED_RETRY_VALID",
            "published_root_present": True,
            "formal_science_executed": False,
            "rng_streams_executed": 0,
            "scientific_artifact_count": len(before),
            "scientific_inventory_sha256": _inventory_sha256(before),
            "audit_go_sha256": _canonical_sha256(source_delta),
        }
    finally:
        _require_inventory_unchanged(config.original_root, before)


def publish_retry(
    config: PublishRetryConfig,
    amendment: Mapping[str, Any],
    *,
    independent_audit_go_sha256: str,
    resume: bool,
) -> None:
    before = validate_original_root(config, amendment)
    cleanup_complete_on_failure = False
    try:
        source_delta = validate_authorized_source_delta(config, amendment)
        audit_hash = _canonical_sha256(source_delta)
        if independent_audit_go_sha256 != audit_hash:
            raise RuntimeError(
                "publish requires the exact source-delta hash from validate-only"
            )
        root = config.output_root
        if root.exists() and not resume:
            raise FileExistsError(f"fresh publish-retry root already exists: {root}")
        if resume and not root.is_dir():
            raise FileNotFoundError("publish-retry resume requires an existing root")
        if not root.exists():
            root.mkdir(parents=True, exist_ok=False)
            _fsync_directory(root.parent)
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError("publish-retry root must be a real directory")
        cleanup_complete_on_failure = True
        if (root / COMPLETE_PATH).exists():
            if not resume:
                raise RuntimeError("completed retry root requires --resume")
            validate_published_retry(
                config,
                amendment,
                source_delta=source_delta,
            )
            cleanup_complete_on_failure = False
            return

        inventory = _amendment_inventory(amendment)
        _validate_partial_root(root, inventory)
        for entry in inventory:
            _copy_or_verify_exact(
                config.original_root / entry["path"],
                root / entry["path"],
                entry,
            )
        _write_or_verify_bytes(
            root / PUBLISHED_AMENDMENT_PATH,
            config.amendment_path.read_bytes(),
        )
        _write_or_verify_bytes(root / PUBLISHED_CONFIG_PATH, CONFIG_PATH.read_bytes())

        validate_scientific_semantics(root, config, amendment)
        record = _administrative_record(
            root,
            config,
            amendment,
            source_delta=source_delta,
        )
        _write_or_verify_bytes(
            root / ADMINISTRATIVE_RECORD_PATH,
            _json_bytes(record),
        )
        manifest = _manifest_payload(root, config, amendment, record)
        _write_or_verify_bytes(root / MANIFEST_PATH, _json_bytes(manifest))

        validate_retry_root(
            root,
            config,
            amendment,
            source_delta=source_delta,
            require_complete=False,
        )
        _require_inventory_unchanged(config.original_root, before)
        _atomic_write(root / COMPLETE_PATH, _complete_marker(root, amendment))
        validate_retry_root(
            root,
            config,
            amendment,
            source_delta=source_delta,
            require_complete=True,
        )
        cleanup_complete_on_failure = False
    except BaseException:
        if cleanup_complete_on_failure and _real_directory(config.output_root):
            _unlink_complete(config.output_root)
        raise
    finally:
        _require_inventory_unchanged(config.original_root, before)


def validate_published_retry(
    config: PublishRetryConfig,
    amendment: Mapping[str, Any],
    *,
    source_delta: Mapping[str, Any] | None = None,
) -> None:
    delta = source_delta or validate_authorized_source_delta(config, amendment)
    validate_retry_root(
        config.output_root,
        config,
        amendment,
        source_delta=delta,
        require_complete=True,
    )


def validate_retry_root(
    root: Path,
    config: PublishRetryConfig,
    amendment: Mapping[str, Any],
    *,
    source_delta: Mapping[str, Any],
    require_complete: bool,
) -> None:
    inventory = _amendment_inventory(amendment)
    record = _administrative_record(
        root,
        config,
        amendment,
        source_delta=source_delta,
    )
    manifest = _manifest_payload(root, config, amendment, record)
    expected_paths = {
        *(Path(entry["path"]) for entry in inventory),
        *ADMINISTRATIVE_PATHS,
        MANIFEST_PATH,
    }
    if require_complete:
        expected_paths.add(COMPLETE_PATH)
    _require_exact_tree(root, expected_paths)
    _validate_scientific_bytes(root, inventory)
    _require_file_bytes(
        root / PUBLISHED_AMENDMENT_PATH,
        config.amendment_path.read_bytes(),
    )
    _require_file_bytes(root / PUBLISHED_CONFIG_PATH, CONFIG_PATH.read_bytes())
    _require_file_bytes(root / ADMINISTRATIVE_RECORD_PATH, _json_bytes(record))
    _require_file_bytes(root / MANIFEST_PATH, _json_bytes(manifest))
    _verify_manifest(root, manifest)
    _validate_forensic_bindings(root, amendment)
    validate_scientific_semantics(root, config, amendment)
    if require_complete:
        _require_file_bytes(root / COMPLETE_PATH, _complete_marker(root, amendment))


def validate_original_root(
    config: PublishRetryConfig,
    amendment: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    expected = _amendment_inventory(amendment)
    _require_exact_tree(
        config.original_root,
        {Path(entry["path"]) for entry in expected},
    )
    observed = _inventory(config.original_root)
    if observed != expected:
        raise RuntimeError("original failed-publish inventory differs")
    incident = amendment["incident"]
    if (
        len(observed) != incident["present_file_count"]
        or sum(row["bytes"] for row in observed)
        != incident["present_total_bytes"]
        or _inventory_sha256(observed) != incident["present_inventory_sha256"]
        or (config.original_root / MANIFEST_PATH).exists()
        or (config.original_root / COMPLETE_PATH).exists()
    ):
        raise RuntimeError("original failed-publish root state differs")
    _validate_original_metadata(config.original_root, incident)
    _validate_forensic_bindings(config.original_root, amendment)
    return observed


def _validate_original_metadata(root: Path, incident: Mapping[str, Any]) -> None:
    metadata_path = root / "metadata.json"
    final_path = root / "FINAL_STATUS.json"
    metadata = _read_json(metadata_path)
    final = _read_json(final_path)
    if (
        _file_sha256(metadata_path) != incident["metadata_file_sha256"]
        or _canonical_sha256(metadata) != incident["metadata_canonical_sha256"]
        or _file_sha256(final_path) != incident["final_status_file_sha256"]
        or _canonical_sha256(final) != incident["final_status_canonical_sha256"]
        or metadata.get("source_tree_sha256")
        != EXPECTED_ORIGINAL_SOURCE_TREE_SHA256
        or metadata.get("gate_contract_sha256")
        != incident["gate_contract_sha256"]
        or _canonical_sha256(metadata.get("gate_contract"))
        != incident["gate_contract_sha256"]
        or metadata.get("confirmation_binding_sha256")
        != incident["confirmation_binding_sha256"]
        or _canonical_sha256(metadata.get("confirmation_binding"))
        != incident["confirmation_binding_sha256"]
    ):
        raise RuntimeError("original science metadata binding differs")
    science._verify_source_snapshot(root, metadata["source_snapshot"])


def _validate_forensic_bindings(
    root: Path,
    amendment: Mapping[str, Any],
) -> None:
    metadata = _read_json(root / "metadata.json")
    scope = amendment["incident"]["science_scope"]
    rng_binding = amendment["incident"]["rng_binding"]
    if _canonical_sha256(metadata["seed_to_device"]) != scope[
        "seed_to_device_map_sha256"
    ]:
        raise RuntimeError("seed-to-device map binding differs")

    payload_map: dict[str, dict[str, str]] = {}
    identity_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seed_counts = {}
    row_count = 0
    method_cell_count = 0
    for dataset in science.CONFIRMED_DATASETS:
        payload_map[dataset] = {}
        identity_map[dataset] = {}
        seed_paths = sorted(
            (root / science.SCIENCE_PHASE / dataset / "seeds").glob(
                "seed_*.json"
            )
        )
        seed_counts[dataset] = len(seed_paths)
        for path in seed_paths:
            payload = _read_json(path)
            seed_key = str(payload["seed"])
            rows = payload["result"]["rows"]
            payload_map[dataset][seed_key] = _canonical_sha256(payload)
            identity_map[dataset][seed_key] = [
                {
                    "seed": row["seed"],
                    "dataset": row["dataset"],
                    "gamma": row["gamma"],
                    "methods": sorted(row["methods"]),
                }
                for row in rows
            ]
            row_count += len(rows)
            method_cell_count += sum(len(row["methods"]) for row in rows)
    if (
        seed_counts != scope["eligible_seed_counts"]
        or sum(seed_counts.values()) != scope["seed_payload_count"]
        or row_count != scope["seed_gamma_row_count"]
        or method_cell_count != scope["seed_gamma_method_cell_count"]
        or _canonical_sha256(payload_map) != scope["seed_payload_map_sha256"]
        or _canonical_sha256(identity_map) != scope["row_identity_map_sha256"]
    ):
        raise RuntimeError("science payload forensic binding differs")

    rng_mapping = metadata["rng_audit"]["new_rng_stream_mapping"]
    eligible = {
        dataset: {int(seed) for seed in seeds}
        for dataset, seeds in metadata["seed_to_device"].items()
    }
    used_mapping = {
        key: value
        for key, value in rng_mapping.items()
        if any(
            key == f"{dataset}/summary_bootstrap"
            or any(
                key.startswith(f"{dataset}/base_{seed}/")
                for seed in seeds
            )
            for dataset, seeds in eligible.items()
        )
    }
    if (
        len(rng_mapping) != rng_binding["full_inherited_stream_count"]
        or _canonical_sha256(rng_mapping)
        != rng_binding["full_inherited_mapping_sha256"]
        or len(used_mapping) != rng_binding["eligible_inherited_stream_count"]
        or _canonical_sha256(used_mapping)
        != rng_binding["eligible_inherited_mapping_sha256"]
    ):
        raise RuntimeError("inherited RNG forensic binding differs")


def validate_authorized_source_delta(
    config: PublishRetryConfig,
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    incident = amendment["incident"]
    source_manifest_path = (
        config.original_root
        / "provenance"
        / f"source_manifest_{incident['source_manifest_sha256']}.json"
    )
    if _file_sha256(source_manifest_path) != incident["source_manifest_sha256"]:
        raise RuntimeError("archived source manifest bytes differ")
    source_manifest = _read_json(source_manifest_path)
    _require_exact_keys(
        source_manifest,
        {"protocol", "format", "file_count", "files"},
        "archived source manifest",
    )
    archived_rows = _validate_inventory_rows(
        source_manifest["files"],
        require_sorted=False,
    )
    if source_manifest["file_count"] != len(archived_rows):
        raise RuntimeError("archived source manifest count differs")
    archived = {row["path"]: row for row in archived_rows}
    current_paths = _experiment_source_paths()
    current = {
        path.relative_to(ROOT).as_posix(): {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in current_paths
    }
    changed_path = "scripts/run_controlled_clinical_fidelity_v4_science.py"
    added_paths = set(
        amendment["authorized_source_delta"][
            "added_administrative_executable_sources"
        ]
    )
    if set(current) - set(archived) != added_paths:
        raise RuntimeError("active source contains an unauthorized added path")
    if set(archived) - set(current):
        raise RuntimeError("active source is missing an archived path")
    for path, archived_entry in archived.items():
        if path == changed_path:
            if (
                archived_entry["sha256"]
                != EXPECTED_ARCHIVED_SCIENCE_RUNNER_SHA256
                or current[path]["sha256"]
                != EXPECTED_REPAIRED_SCIENCE_RUNNER_SHA256
            ):
                raise RuntimeError("science metadata-boundary repair differs")
            continue
        if current[path] != archived_entry:
            raise RuntimeError(f"active archived source differs: {path}")
    return {
        "archived_source_tree_sha256": EXPECTED_ORIGINAL_SOURCE_TREE_SHA256,
        "archived_source_manifest_sha256": incident["source_manifest_sha256"],
        "changed_existing_sources": [
            {
                "path": changed_path,
                "archived_sha256": EXPECTED_ARCHIVED_SCIENCE_RUNNER_SHA256,
                "repaired_sha256": EXPECTED_REPAIRED_SCIENCE_RUNNER_SHA256,
                "change": "canonical_JSON_roundtrip_only",
            }
        ],
        "added_administrative_sources": [
            {
                "path": path,
                "bytes": current[path]["bytes"],
                "sha256": current[path]["sha256"],
            }
            for path in sorted(added_paths)
        ],
        "all_other_archived_sources_exact": True,
    }


def validate_scientific_semantics(
    root: Path,
    config: PublishRetryConfig,
    amendment: Mapping[str, Any],
) -> None:
    metadata = _read_json(root / "metadata.json")
    live_gates = science.verify_gate_bundle(devices=config.devices)
    live_contract = _json_roundtrip(live_gates.contract)
    live_contract["active_science_source_tree_sha256"] = (
        EXPECTED_ORIGINAL_SOURCE_TREE_SHA256
    )
    if live_contract != metadata["gate_contract"]:
        raise RuntimeError("historical gate contract cannot be reconstructed")
    historical_gates = replace(
        live_gates,
        active_source_tree_sha256=EXPECTED_ORIGINAL_SOURCE_TREE_SHA256,
        contract=metadata["gate_contract"],
    )
    reconstructed = science._science_metadata(
        historical_gates,
        devices=config.devices,
        independent_audit_go_sha256=metadata["independent_audit_go_sha256"],
        source_snapshot=metadata["source_snapshot"],
    )
    if reconstructed != metadata:
        raise RuntimeError("canonical historical science metadata differs")
    if _canonical_sha256(reconstructed) != amendment["incident"][
        "reconstructed_metadata_canonical_sha256"
    ]:
        raise RuntimeError("canonical metadata hash differs")

    validation_root = root.parent / f".{root.name}.semantic-validation-{os.getpid()}"
    if validation_root.exists():
        raise FileExistsError(f"semantic validation view already exists: {validation_root}")
    validation_root.mkdir(parents=False, exist_ok=False)
    try:
        for entry in _amendment_inventory(amendment):
            source = root / entry["path"]
            target = validation_root / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        science._write_manifest(validation_root)
        pinned_uniforms = {
            historical_gates.datasets[dataset].preset.bootstrap_seed: np.load(
                validation_root
                / science.SCIENCE_PHASE
                / dataset
                / "bootstrap_uniforms.npy",
                allow_pickle=False,
            )
            for dataset in science.CONFIRMED_DATASETS
        }
        pinned_factory = _PinnedBootstrapFactory(pinned_uniforms)
        original_bootstrap_validator = science._ensure_bootstrap_artifacts
        original_default_rng = science.v2.np.random.default_rng
        science._ensure_bootstrap_artifacts = (
            lambda bootstrap_root, preset, create_if_missing=True: (
                _validate_bootstrap_without_rng(
                    bootstrap_root,
                    preset,
                    amendment,
                    create_if_missing=create_if_missing,
                )
            )
        )
        science.v2.np.random.default_rng = pinned_factory
        try:
            science._validate_complete_root_contents(
                validation_root,
                metadata,
                historical_gates,
            )
            pinned_factory.require_all_used_once()
        finally:
            science._ensure_bootstrap_artifacts = original_bootstrap_validator
            science.v2.np.random.default_rng = original_default_rng
    finally:
        _remove_validation_root(validation_root)


class _PinnedBootstrapFactory:
    def __init__(self, uniforms_by_seed: Mapping[int, np.ndarray]) -> None:
        self._uniforms_by_seed = dict(uniforms_by_seed)
        self._calls = {seed: 0 for seed in uniforms_by_seed}

    def __call__(self, seed: int) -> "_PinnedBootstrapFactory":
        normalized_seed = int(seed)
        if normalized_seed not in self._uniforms_by_seed:
            raise RuntimeError("semantic validation requested an unknown RNG seed")
        self._active_seed = normalized_seed
        self._calls[normalized_seed] += 1
        return self

    def random(
        self,
        shape: tuple[int, ...],
        *,
        dtype: Any,
    ) -> np.ndarray:
        if not hasattr(self, "_active_seed"):
            raise RuntimeError("pinned bootstrap reader lacks an active seed")
        uniforms = self._uniforms_by_seed[self._active_seed]
        if shape != uniforms.shape or np.dtype(dtype) != uniforms.dtype:
            raise RuntimeError("semantic validation requested a different bootstrap bank")
        return uniforms.copy()

    def require_all_used_once(self) -> None:
        if any(count != 1 for count in self._calls.values()):
            raise RuntimeError("semantic validation bootstrap use count differs")


def _validate_bootstrap_without_rng(
    root: Path,
    preset: Any,
    amendment: Mapping[str, Any],
    *,
    create_if_missing: bool,
) -> dict[str, Any]:
    if create_if_missing:
        raise RuntimeError("administrative validation cannot create bootstrap artifacts")
    dataset = preset.name
    bindings = amendment["incident"]["dataset_artifact_sha256"][dataset]
    uniform_path = root / "bootstrap_uniforms.npy"
    index_path = root / "bootstrap_indices.npy"
    if (
        not uniform_path.is_file()
        or uniform_path.is_symlink()
        or _file_sha256(uniform_path) != bindings["bootstrap_uniforms"]
        or not index_path.is_file()
        or index_path.is_symlink()
        or _file_sha256(index_path) != bindings["bootstrap_indices"]
    ):
        raise RuntimeError(f"{dataset} pinned bootstrap bytes differ")
    uniforms = np.load(uniform_path, allow_pickle=False)
    indices = np.load(index_path, allow_pickle=False)
    expected_indices = np.floor(uniforms * 20).astype(np.int16)
    if (
        uniforms.dtype != np.dtype(np.float64)
        or uniforms.shape != (science.BOOTSTRAP_RESAMPLES, 20)
        or not np.all(np.isfinite(uniforms))
        or not np.all((uniforms >= 0.0) & (uniforms < 1.0))
        or indices.dtype != np.dtype(np.int16)
        or indices.shape != (science.BOOTSTRAP_RESAMPLES, 20)
        or not np.array_equal(indices, expected_indices)
    ):
        raise RuntimeError(f"{dataset} pinned bootstrap arrays differ")
    return {
        "resamples": science.BOOTSTRAP_RESAMPLES,
        "root_seed": preset.bootstrap_seed,
        "prespecified_seed_count": 20,
        "uniform_matrix_shape": [science.BOOTSTRAP_RESAMPLES, 20],
        "uniform_matrix_path": uniform_path.name,
        "uniform_matrix_sha256": bindings["bootstrap_uniforms"],
        "complete_seed_index_matrix_shape": [science.BOOTSTRAP_RESAMPLES, 20],
        "complete_seed_index_matrix_path": index_path.name,
        "complete_seed_index_matrix_sha256": bindings["bootstrap_indices"],
        "unit": "complete_seed_stage_vector",
        "shared_across": ["methods", "gammas", "stages"],
        "selected_subset_rule": (
            "for selected-set size n, use floor(U[:, :n] * n); eICU keeps the "
            "20-column prespecified bank and projects to eligible/selected subsets"
        ),
    }


def _administrative_record(
    root: Path,
    config: PublishRetryConfig,
    amendment: Mapping[str, Any],
    *,
    source_delta: Mapping[str, Any],
) -> dict[str, Any]:
    incident = amendment["incident"]
    return {
        "protocol": RETRY_PROTOCOL,
        "role": RETRY_ROLE,
        "status": "PRECOMMIT_SEMANTIC_VALIDATION_PASSED",
        "administrative_only": True,
        "published_root": str(root.resolve()),
        "original_root": str(config.original_root),
        "original_root_mutated": False,
        "operation": "exact_byte_copy_then_administrative_publish",
        "formal_science_executed": False,
        "rng_streams_executed": 0,
        "inherited_rng_fresh": False,
        "inherited_rng_independent": False,
        "scientific_values_logged": False,
        "independent_audit": {
            "status": "GO",
            "authorized_source_delta_sha256": _canonical_sha256(source_delta),
        },
        "scientific_artifact_count": incident["present_file_count"],
        "scientific_total_bytes": incident["present_total_bytes"],
        "scientific_inventory_sha256": incident["present_inventory_sha256"],
        "source_tree_at_science_execution_sha256": incident[
            "source_tree_sha256"
        ],
        "metadata_canonical_sha256": incident["metadata_canonical_sha256"],
        "final_status_canonical_sha256": incident[
            "final_status_canonical_sha256"
        ],
        "gate_contract_sha256": incident["gate_contract_sha256"],
        "confirmation_binding_sha256": incident[
            "confirmation_binding_sha256"
        ],
        "source_control": {
            "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "amendment_path": AMENDMENT_PATH.relative_to(ROOT).as_posix(),
            "amendment_sha256": EXPECTED_AMENDMENT_SHA256,
            "publish_runner_path": RUNNER_PATH.relative_to(ROOT).as_posix(),
            "publish_runner_sha256": _file_sha256(RUNNER_PATH),
            "science_runner_repaired_sha256": (
                EXPECTED_REPAIRED_SCIENCE_RUNNER_SHA256
            ),
            "authorized_source_delta": dict(source_delta),
        },
        "claim_boundary": {
            "fresh_science_claimed": False,
            "independent_science_claimed": False,
            "scientific_payload_modified": False,
            "administrative_publish_retry_only": True,
        },
    }


def _manifest_payload(
    root: Path,
    config: PublishRetryConfig,
    amendment: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    scientific = [
        {**entry, "role": "inherited_scientific_exact_byte"}
        for entry in _amendment_inventory(amendment)
    ]
    administrative_payloads = {
        ADMINISTRATIVE_RECORD_PATH: _json_bytes(record),
        PUBLISHED_AMENDMENT_PATH: config.amendment_path.read_bytes(),
        PUBLISHED_CONFIG_PATH: CONFIG_PATH.read_bytes(),
    }
    administrative = [
        {
            "path": path.as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "role": "administrative_retry_provenance",
        }
        for path, payload in administrative_payloads.items()
    ]
    artifacts = sorted(scientific + administrative, key=lambda row: row["path"])
    return {
        "protocol": RETRY_PROTOCOL,
        "artifact_count": len(artifacts),
        "scientific_artifact_count": len(scientific),
        "administrative_artifact_count": len(administrative),
        "scientific_inventory_sha256": amendment["incident"][
            "present_inventory_sha256"
        ],
        "artifacts": artifacts,
    }


def _verify_manifest(root: Path, expected: Mapping[str, Any]) -> None:
    manifest = _read_json(root / MANIFEST_PATH)
    if manifest != expected:
        raise RuntimeError("publish-retry manifest payload differs")
    seen = set()
    for entry in manifest["artifacts"]:
        relative = _safe_relative_path(entry["path"])
        if relative in seen or relative in {MANIFEST_PATH, COMPLETE_PATH}:
            raise RuntimeError("publish-retry manifest path is duplicated or reserved")
        seen.add(relative)
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != entry["bytes"]
            or _file_sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"publish-retry manifest mismatch: {relative}")


def _complete_marker(root: Path, amendment: Mapping[str, Any]) -> bytes:
    record_sha256 = _file_sha256(root / ADMINISTRATIVE_RECORD_PATH)
    manifest_sha256 = _file_sha256(root / MANIFEST_PATH)
    record = _read_json(root / ADMINISTRATIVE_RECORD_PATH)
    audit_hash = record["independent_audit"][
        "authorized_source_delta_sha256"
    ]
    return (
        f"complete protocol={RETRY_PROTOCOL} "
        f"scientific_inventory_sha256={amendment['incident']['present_inventory_sha256']} "
        f"administrative_record_sha256={record_sha256} "
        f"audit_go_sha256={audit_hash} "
        f"manifest_sha256={manifest_sha256} "
        "formal_science_executed=false rng_streams_executed=0\n"
    ).encode("utf-8")


def _validate_partial_root(
    root: Path,
    inventory: tuple[dict[str, Any], ...],
) -> None:
    allowed = {
        *(Path(entry["path"]) for entry in inventory),
        *ADMINISTRATIVE_PATHS,
        MANIFEST_PATH,
    }
    observed = _observed_files(root)
    if not observed <= allowed:
        raise RuntimeError(
            f"partial publish-retry root contains unexpected files: "
            f"{sorted(path.as_posix() for path in observed - allowed)}"
        )
    if MANIFEST_PATH in observed:
        required = allowed - {MANIFEST_PATH}
        if not required <= observed:
            raise RuntimeError("partial root contains a premature manifest")
    _require_allowed_directories(root, allowed)


def _validate_scientific_bytes(
    root: Path,
    inventory: tuple[dict[str, Any], ...],
) -> None:
    observed = []
    for entry in inventory:
        path = root / entry["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != entry["bytes"]
            or _file_sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"copied scientific artifact differs: {entry['path']}")
        observed.append(dict(entry))
    if _inventory_sha256(tuple(observed)) != EXPECTED_ORIGINAL_INVENTORY_SHA256:
        raise RuntimeError("copied scientific inventory hash differs")


def _copy_or_verify_exact(
    source: Path,
    target: Path,
    entry: Mapping[str, Any],
) -> None:
    if target.exists() or target.is_symlink():
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != entry["bytes"]
            or _file_sha256(target) != entry["sha256"]
        ):
            raise RuntimeError(f"existing copied artifact differs: {entry['path']}")
        return
    payload = source.read_bytes()
    if (
        len(payload) != entry["bytes"]
        or hashlib.sha256(payload).hexdigest() != entry["sha256"]
    ):
        raise RuntimeError(f"original artifact changed during copy: {entry['path']}")
    _atomic_write(target, payload)


def _require_inventory_unchanged(
    root: Path,
    expected: tuple[dict[str, Any], ...],
) -> None:
    if _inventory(root) != expected:
        raise RuntimeError("original failed-publish root changed during retry")


def _amendment_inventory(
    amendment: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(entry)
        for entry in amendment["incident"]["scientific_artifact_inventory"]
    )


def _validate_inventory_rows(
    value: object,
    *,
    require_sorted: bool = True,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise RuntimeError("artifact inventory must be a list")
    rows = []
    seen = set()
    for raw in value:
        _require_exact_keys(raw, {"path", "bytes", "sha256"}, "inventory entry")
        relative = _safe_relative_path(raw["path"])
        if relative in seen:
            raise RuntimeError("artifact inventory path is duplicated")
        if (
            not isinstance(raw["bytes"], int)
            or raw["bytes"] < 0
            or not _is_sha256(raw["sha256"])
        ):
            raise RuntimeError("artifact inventory entry is malformed")
        seen.add(relative)
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": raw["bytes"],
                "sha256": raw["sha256"],
            }
        )
    if require_sorted and rows != sorted(rows, key=lambda row: row["path"]):
        raise RuntimeError("artifact inventory must be path-sorted")
    return tuple(rows)


def _inventory(root: Path) -> tuple[dict[str, Any], ...]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"artifact root is missing or unsafe: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symbolic links are forbidden: {path}")
        if not path.is_file():
            continue
        if ".tmp-" in path.name:
            raise RuntimeError(f"temporary artifact remains: {path}")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return tuple(rows)


def _inventory_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(rows),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_exact_tree(root: Path, expected_files: set[Path]) -> None:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"artifact root is missing or unsafe: {root}")
    observed = _observed_files(root)
    if observed != expected_files:
        missing = sorted(path.as_posix() for path in expected_files - observed)
        extra = sorted(path.as_posix() for path in observed - expected_files)
        raise RuntimeError(f"artifact file set differs; missing={missing}; extra={extra}")
    _require_allowed_directories(root, expected_files)


def _observed_files(root: Path) -> set[Path]:
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symbolic links are forbidden: {path}")
        if path.is_file():
            if ".tmp-" in path.name:
                raise RuntimeError(f"temporary artifact remains: {path}")
            observed.add(path.relative_to(root))
    return observed


def _require_allowed_directories(root: Path, files: set[Path]) -> None:
    allowed = set()
    for relative in files:
        parent = relative.parent
        while parent != Path("."):
            allowed.add(parent)
            parent = parent.parent
    observed = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if not observed <= allowed:
        raise RuntimeError(
            "artifact root contains unexpected directories: "
            f"{sorted(path.as_posix() for path in observed - allowed)}"
        )


def _experiment_source_paths() -> tuple[Path, ...]:
    paths = (
        *sorted((ROOT / "src/scpcp").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "tools").glob("*.py")),
        *sorted((ROOT / "configs").glob("*.yaml")),
        ROOT / "pyproject.toml",
    )
    if len(paths) != len(set(paths)) or any(not path.is_file() for path in paths):
        raise RuntimeError("active experiment source file set is invalid")
    return tuple(paths)


def _safe_relative_path(value: object) -> Path:
    if not isinstance(value, str):
        raise RuntimeError("artifact path must be a string")
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise RuntimeError("artifact path is unsafe")
    return relative


def _remove_validation_root(path: Path) -> None:
    if (
        not path.name.startswith(".")
        or ".semantic-validation-" not in path.name
        or path.parent == Path("/")
    ):
        raise RuntimeError("semantic validation path guard failed")
    shutil.rmtree(path)


def _write_or_verify_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        _require_file_bytes(path, payload)
        return
    _atomic_write(path, payload)


def _require_file_bytes(path: Path, payload: bytes) -> None:
    if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
        raise RuntimeError(f"artifact bytes differ: {path}")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _unlink_complete(root: Path) -> None:
    complete = root / COMPLETE_PATH
    complete.unlink(missing_ok=True)
    _fsync_directory(root)


def _real_directory(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _json_roundtrip(value: object) -> dict[str, Any]:
    canonical = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    if not isinstance(canonical, dict):
        raise RuntimeError("canonical JSON value must be an object")
    return canonical


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError(f"{label} fields differ")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


if __name__ == "__main__":
    main()
