"""Run the gate-only native Synthetic signed-gamma preflight.

The runner performs a live, repository-wide audit of every actual RNG ID before
creating its output root.  It never accepts a checked-in PASS flag or digest.

Validate the frozen contract without consuming a formal seed:

    conda run -n ucp python scripts/run_native_synthetic_signed_gamma_preflight.py \
      --validate-only
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from multiprocessing import get_context
import os
from pathlib import Path
import platform
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

from scpcp.artifacts import git_revision  # noqa: E402
from scpcp.native_signed_gamma import (  # noqa: E402
    GAMMAS,
    NativeSignedGammaBenchmarkConfig,
    mechanism_probe,
    seed_passes_mechanism_gate,
)


DEFAULT_CONFIG = ROOT / "configs/native_synthetic_signed_gamma.yaml"
FORBIDDEN_FIELD_TOKENS = (
    "coverage",
    "width",
    "q90",
    "score",
    "selection",
    "science",
)
RNG_FIELD = re.compile(r"(?:seed|rng|random)", re.IGNORECASE)
RESERVATION_FIELD = re.compile(
    r"(?:^|_)(?:reserved|reservation|reservations)(?:_|$)",
    re.IGNORECASE,
)
DECLARATION_FIELD = re.compile(
    r"(?:^|_)(?:declared|declaration|declarations)(?:_|$)",
    re.IGNORECASE,
)
RNG_CALL = re.compile(
    r"(?:manual_seed|default_rng|randomstate|noise|probe|simulate|rollout|"
    r"bootstrap|sample|experiment|run_)",
    re.IGNORECASE,
)
SEED_ARTIFACT_NAME = re.compile(r"(?:seed|rng|problem)[_-](\d+)(?:\.json)?$")
STRUCTURED_ARTIFACT_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
TABULAR_ARTIFACT_SUFFIXES = frozenset({".csv", ".tsv"})
MISSING_TABULAR_VALUES = frozenset({"", "na", "nan", "null", "none"})
RNG_DESCRIPTOR_SUFFIXES = (
    "_bytes",
    "_count",
    "_dir",
    "_fraction",
    "_hash",
    "_label",
    "_namespace",
    "_path",
    "_policy",
    "_sha256",
    "_status",
)
ARTIFACT_SCAN_CONTRACT = {
    "version": "structured_actual_rng_v2",
    "structured_extensions": sorted(STRUCTURED_ARTIFACT_SUFFIXES),
    "tabular_extensions": sorted(TABULAR_ARTIFACT_SUFFIXES),
    "current_output_excluded": True,
    "path_escape": "fail_closed",
    "structured_parse_error": "fail_closed",
    "tabular_parse_error": "fail_closed",
    "noninteger_rng_value": "fail_closed_except_empty_or_NA_tabular_cells",
    "reservation_and_declaration_fields": "reported_separately_not_actual_use",
    "binary_rng_arrays": "verify_declared_adjacent_sha256_do_not_parse_indices_as_ids",
}
SOURCE_SCAN_CONTRACT = {
    "version": "scope_aware_actual_rng_v2",
    "scope": "scripts/src/tools Python trees",
    "consumers": [
        "manual_seed",
        "default_rng",
        "Generator",
        "Random",
        "RandomState",
        "seed",
    ],
    "propagation": "lexical scopes/defaults/containers/subscripts/simple integer arithmetic",
    "unresolved_concrete_seed_expression": "fail_closed_with_path_and_line",
    "reservation_and_declaration_assignments": "reported_separately_not_actual_use",
}


def _paper_rng_id(seed: int, stream: int) -> int:
    return int((1_000_003 * seed + stream) % (2**31 - 1))


def _controlled_clinical_v3_reservation_mapping() -> dict[str, int]:
    """Mirror the frozen 1,304-stream clinical-v3 confirmation reservation."""

    seeds_by_dataset = {
        "mimic_iv": tuple(range(111_000, 111_200, 10)),
        "eicu": tuple(range(112_000, 112_200, 10)),
        "inspire": tuple(range(113_000, 113_200, 10)),
        "mimic_cxr": tuple(range(114_000, 114_200, 10)),
    }
    bootstrap_by_dataset = {
        "mimic_iv": 11_100_019,
        "eicu": 11_200_019,
        "inspire": 11_300_019,
        "mimic_cxr": 11_400_019,
    }
    mapping: dict[str, int] = {}
    for dataset, seeds in seeds_by_dataset.items():
        mapping[f"{dataset}/summary_bootstrap"] = bootstrap_by_dataset[dataset]
        for seed in seeds:
            prefix = f"{dataset}/base_{seed}"
            mapping[f"{prefix}/task"] = seed
            mapping[f"{prefix}/outcome_model"] = seed + 1
            mapping[f"{prefix}/behavior_model"] = seed + 2
            if dataset == "mimic_cxr":
                mapping[f"{prefix}/cxr_encoder"] = seed + 701
            mapping[f"{prefix}/k0_base_uniform"] = 90_000_000 + seed
            mapping[f"{prefix}/donor_overlap_probe"] = _paper_rng_id(seed, 1_700_301)
            mapping[f"{prefix}/calibration"] = _paper_rng_id(seed, 1_700_101)
            mapping[f"{prefix}/reference"] = _paper_rng_id(seed, 1_700_401)
            adaptation_root = _paper_rng_id(seed, 700_001)
            for round_index in range(3):
                mapping[f"{prefix}/ACI_round_{round_index}"] = (
                    _paper_rng_id(adaptation_root, 101) + 17_923 * round_index
                )
                mapping[f"{prefix}/SPCI_round_{round_index}"] = (
                    _paper_rng_id(adaptation_root, 211) + 47_021 * round_index
                )
                mapping[f"{prefix}/PRC_round_{round_index}"] = (
                    _paper_rng_id(adaptation_root, 307) + 61_103 * round_index
                )
    if len(mapping) != 1_304 or len(set(mapping.values())) != 1_304:
        raise RuntimeError("controlled clinical-v3 reservation mapping differs")
    return mapping


CONTROLLED_CLINICAL_V3_RESERVATION_MAPPING = (
    _controlled_clinical_v3_reservation_mapping()
)

# These are coordinated external namespaces, not evidence of prior use.  They
# are kept separate from the actual-use scans and hashed in the live audit.
COORDINATED_EXTERNAL_RESERVATIONS: Mapping[str, frozenset[int]] = {
    "exact_finite_mdp": frozenset(range(52_000, 53_000)),
    "controlled_and_copula": frozenset(range(91_000, 95_000)),
    "rq5_rq6_and_robustness": frozenset(range(95_000, 101_000)),
    "controlled_clinical_fidelity_v3": frozenset(
        CONTROLLED_CLINICAL_V3_RESERVATION_MAPPING.values()
    ),
}

METADATA_FIELDS = (
    "protocol",
    "role",
    "gate_only",
    "config",
    "config_path",
    "config_file_sha256",
    "config_payload_sha256",
    "rng_audit",
    "seed_device_mapping",
    "seed_device_mapping_sha256",
    "source_tree_sha256",
    "source_snapshot",
    "dependency_files",
    "environment",
    "environment_sha256",
    "invocation",
    "invocation_sha256",
    "launch_contract_sha256",
    "artifact_schema_sha256",
    "information_firewall",
    "semantics",
    "downstream_authorization",
)
SUMMARY_FIELDS = (
    "protocol",
    "gate_only",
    "n_prespecified",
    "n_passed",
    "passed_rng_ids",
    "required_passed_rng_ids",
    "status",
    "failure_consequence",
    "config_payload_sha256",
    "rng_audit_sha256",
)
SEED_ARTIFACT_FIELDS = (
    "protocol",
    "rng_label",
    "rng_id",
    "config_payload_sha256",
    "rng_audit_sha256",
    "device",
    "source_tree_sha256",
    "probe",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--devices", default=None, help="comma-separated CUDA devices")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="audit and print the frozen contract without running any RNG ID",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    config_path = args.config.resolve()
    config = NativeSignedGammaBenchmarkConfig.from_yaml(config_path)
    if args.devices:
        devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
        config = config.with_overrides(devices=devices)
    selected_output = config.output_root if args.output_root is None else args.output_root
    if not selected_output.is_absolute():
        selected_output = ROOT / selected_output
    config = config.with_overrides(output_root=selected_output.resolve())
    invocation = _canonical_invocation(raw_argv)
    if args.validate_only:
        print(
            json.dumps(
                validation_payload(config, config_path),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    run_preflight(
        config,
        config_path=config_path,
        resume=args.resume,
        invocation_argv=invocation,
    )


def formal_rng_mapping(config: NativeSignedGammaBenchmarkConfig) -> dict[str, int]:
    """Map each gate worker label to the exact ID passed to its local generator."""

    return {
        f"mechanism/base_{rng_id}/exogenous_noise": rng_id
        for rng_id in config.base_seeds
    }


def seed_device_mapping(config: NativeSignedGammaBenchmarkConfig) -> dict[str, str]:
    mapping = formal_rng_mapping(config)
    return {
        label: config.devices[index % len(config.devices)]
        for index, label in enumerate(mapping)
    }


def validation_payload(
    config: NativeSignedGammaBenchmarkConfig,
    config_path: Path,
    *,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    output_root = config.output_root.resolve()
    audit = audit_formal_rng_ids(
        config,
        output_root=output_root,
        artifact_root=artifact_root,
        source_root=source_root,
        config_path=config_path,
    )
    return {
        "protocol": config.protocol,
        "contract_valid": True,
        "formal_launch_permitted": audit["status"] == "passed_before_launch",
        "formal_launch_blocker": None,
        "gate_only": True,
        "primary_gamma": config.primary_gamma,
        "gammas": list(config.gammas),
        "base_seed_namespace": config.seed_namespace,
        "config_path": str(config_path.resolve()),
        "config_file_sha256": _file_sha256(config_path),
        "config_payload_sha256": _canonical_sha256(config.to_dict()),
        "rng_audit": audit,
    }


def audit_formal_rng_ids(
    config: NativeSignedGammaBenchmarkConfig,
    *,
    output_root: Path,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
    external_reservations: Mapping[str, Iterable[int]] | None = None,
) -> dict[str, Any]:
    """Enumerate actual IDs and reject prior-use or reservation collisions."""

    artifact_root = ROOT / "results" if artifact_root is None else artifact_root
    source_root = ROOT if source_root is None else source_root
    using_default_reservations = external_reservations is None
    reservations = (
        COORDINATED_EXTERNAL_RESERVATIONS
        if external_reservations is None
        else external_reservations
    )
    mapping = formal_rng_mapping(config)
    formal_ids = set(mapping.values())
    if len(mapping) != len(config.base_seeds) or len(formal_ids) != len(mapping):
        raise RuntimeError("formal native signed-gamma RNG mapping is not one-to-one")
    if tuple(mapping.values()) != config.base_seeds:
        raise RuntimeError("formal native signed-gamma RNG mapping differs from config order")

    artifact_scan = _artifact_rng_scan(
        artifact_root,
        excluded_root=output_root,
    )
    excluded_source_paths = {
        Path(__file__).resolve(),
        (ROOT / "src/scpcp/native_signed_gamma.py").resolve(),
        config_path.resolve(),
    }
    source_scan = _source_rng_scan(
        source_root,
        excluded_paths=excluded_source_paths,
    )
    artifact_ids = artifact_scan["actual"]
    source_ids = source_scan["actual"]
    reservation_sets = {
        name: set(values) for name, values in reservations.items()
    }
    external_ids = set().union(*reservation_sets.values()) if reservation_sets else set()
    prior_ids = artifact_ids | source_ids | external_ids
    collisions = {
        label: rng_id for label, rng_id in mapping.items() if rng_id in prior_ids
    }
    config_hash = _canonical_sha256(config.to_dict())
    mapping_hash = _canonical_sha256(mapping)
    formal_id_hash = _integer_set_sha256(formal_ids)
    audit = {
        "status": "passed_before_launch" if not collisions else "collision",
        "policy": config.rng_audit_policy,
        "seed_namespace": config.seed_namespace,
        "base_seed_bank_sha256": _canonical_sha256(list(config.base_seeds)),
        "config_payload_sha256": config_hash,
        "formal_rng_id_count": len(formal_ids),
        "formal_rng_ids": sorted(formal_ids),
        "formal_rng_id_sha256": formal_id_hash,
        "formal_rng_mapping": mapping,
        "formal_rng_mapping_sha256": mapping_hash,
        "internal_rng_ids_unique": True,
        "config_rng_binding_sha256": _canonical_sha256(
            {
                "config_payload_sha256": config_hash,
                "formal_rng_id_sha256": formal_id_hash,
                "formal_rng_mapping_sha256": mapping_hash,
            }
        ),
        "artifact_actual_rng_id_count": len(artifact_ids),
        "artifact_actual_rng_ids": sorted(artifact_ids),
        "artifact_actual_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "artifact_declared_rng_id_count": len(artifact_scan["declared"]),
        "artifact_declared_rng_ids": sorted(artifact_scan["declared"]),
        "artifact_declared_rng_id_sha256": _integer_set_sha256(
            artifact_scan["declared"]
        ),
        "artifact_reserved_rng_id_count": len(artifact_scan["reserved"]),
        "artifact_reserved_rng_ids": sorted(artifact_scan["reserved"]),
        "artifact_reserved_rng_id_sha256": _integer_set_sha256(
            artifact_scan["reserved"]
        ),
        "artifact_binary_rng_bindings": artifact_scan["binary_bindings"],
        "artifact_binary_rng_binding_count": len(
            artifact_scan["binary_bindings"]
        ),
        "artifact_binary_rng_binding_sha256": _canonical_sha256(
            artifact_scan["binary_bindings"]
        ),
        "source_actual_use_rng_id_count": len(source_ids),
        "source_actual_use_rng_ids": sorted(source_ids),
        "source_actual_use_rng_id_sha256": _integer_set_sha256(source_ids),
        "source_declared_rng_id_count": len(source_scan["declared"]),
        "source_declared_rng_ids": sorted(source_scan["declared"]),
        "source_declared_rng_id_sha256": _integer_set_sha256(
            source_scan["declared"]
        ),
        "source_reserved_rng_id_count": len(source_scan["reserved"]),
        "source_reserved_rng_ids": sorted(source_scan["reserved"]),
        "source_reserved_rng_id_sha256": _integer_set_sha256(
            source_scan["reserved"]
        ),
        "source_unresolved_rng_expressions": [],
        "coordinated_external_reservations": {
            name: _reservation_contract(
                values,
                mapping=(
                    CONTROLLED_CLINICAL_V3_RESERVATION_MAPPING
                    if using_default_reservations
                    and name == "controlled_clinical_fidelity_v3"
                    else None
                ),
            )
            for name, values in sorted(reservation_sets.items())
        },
        "coordinated_external_rng_id_count": len(external_ids),
        "coordinated_external_rng_id_sha256": _integer_set_sha256(external_ids),
        "prior_rng_id_count": len(prior_ids),
        "prior_rng_id_sha256": _integer_set_sha256(prior_ids),
        "collision_count": len(collisions),
        "collisions": collisions,
        "collision_sha256": _canonical_sha256(collisions),
        "excluded_output": str(output_root.resolve()),
        "excluded_source_declarations": sorted(
            str(path) for path in excluded_source_paths
        ),
        "source_scan_policy": SOURCE_SCAN_CONTRACT,
        "artifact_scan_policy": ARTIFACT_SCAN_CONTRACT,
        "scan_policy_sha256": _canonical_sha256(
            {
                "artifact": ARTIFACT_SCAN_CONTRACT,
                "source": SOURCE_SCAN_CONTRACT,
            }
        ),
    }
    audit["audit_sha256"] = _canonical_sha256(audit)
    validate_rng_audit(
        config,
        audit,
        external_reservations=(None if using_default_reservations else reservation_sets),
    )
    if collisions:
        raise RuntimeError(
            "formal native signed-gamma RNG IDs collide with prior use or "
            f"coordinated reservations: {collisions}"
        )
    return audit


def validate_rng_audit(
    config: NativeSignedGammaBenchmarkConfig,
    audit: Mapping[str, Any],
    *,
    external_reservations: Mapping[str, Iterable[int]] | None = None,
) -> None:
    mapping = formal_rng_mapping(config)
    formal_ids = set(mapping.values())
    config_hash = _canonical_sha256(config.to_dict())
    expected_binding = _canonical_sha256(
        {
            "config_payload_sha256": config_hash,
            "formal_rng_id_sha256": _integer_set_sha256(formal_ids),
            "formal_rng_mapping_sha256": _canonical_sha256(mapping),
        }
    )
    if audit.get("config_payload_sha256") != config_hash:
        raise RuntimeError("RNG audit does not bind the effective config payload")
    if audit.get("formal_rng_ids") != sorted(formal_ids):
        raise RuntimeError("RNG audit formal ID set differs")
    if audit.get("formal_rng_id_count") != len(formal_ids):
        raise RuntimeError("RNG audit formal ID count differs")
    if audit.get("formal_rng_id_sha256") != _integer_set_sha256(formal_ids):
        raise RuntimeError("RNG audit formal ID hash differs")
    if audit.get("formal_rng_mapping") != mapping:
        raise RuntimeError("RNG audit formal mapping differs")
    if audit.get("formal_rng_mapping_sha256") != _canonical_sha256(mapping):
        raise RuntimeError("RNG audit formal mapping hash differs")
    if audit.get("base_seed_bank_sha256") != _canonical_sha256(
        list(config.base_seeds)
    ):
        raise RuntimeError("RNG audit base seed bank hash differs")
    if audit.get("internal_rng_ids_unique") is not True:
        raise RuntimeError("RNG audit does not certify internal uniqueness")
    if audit.get("config_rng_binding_sha256") != expected_binding:
        raise RuntimeError("RNG audit config-to-mapping binding differs")
    artifact_ids = _validated_integer_list(
        audit.get("artifact_actual_rng_ids"),
        "artifact actual-use IDs",
    )
    source_ids = _validated_integer_list(
        audit.get("source_actual_use_rng_ids"),
        "source actual-use IDs",
    )
    artifact_declared_ids = _validated_integer_list(
        audit.get("artifact_declared_rng_ids"),
        "artifact declared IDs",
    )
    artifact_reserved_ids = _validated_integer_list(
        audit.get("artifact_reserved_rng_ids"),
        "artifact reserved IDs",
    )
    source_declared_ids = _validated_integer_list(
        audit.get("source_declared_rng_ids"),
        "source declared IDs",
    )
    source_reserved_ids = _validated_integer_list(
        audit.get("source_reserved_rng_ids"),
        "source reserved IDs",
    )
    if (
        audit.get("artifact_actual_rng_id_count") != len(artifact_ids)
        or audit.get("artifact_actual_rng_id_sha256")
        != _integer_set_sha256(artifact_ids)
    ):
        raise RuntimeError("RNG audit artifact actual-use set differs")
    if (
        audit.get("source_actual_use_rng_id_count") != len(source_ids)
        or audit.get("source_actual_use_rng_id_sha256")
        != _integer_set_sha256(source_ids)
    ):
        raise RuntimeError("RNG audit source actual-use set differs")
    for prefix, values in (
        ("artifact_declared", artifact_declared_ids),
        ("artifact_reserved", artifact_reserved_ids),
        ("source_declared", source_declared_ids),
        ("source_reserved", source_reserved_ids),
    ):
        if (
            audit.get(f"{prefix}_rng_id_count") != len(values)
            or audit.get(f"{prefix}_rng_id_sha256")
            != _integer_set_sha256(values)
        ):
            raise RuntimeError(f"RNG audit {prefix.replace('_', ' ')} set differs")
    binary_bindings = audit.get("artifact_binary_rng_bindings")
    if not isinstance(binary_bindings, list):
        raise RuntimeError("RNG audit binary bindings must be a list")
    if (
        audit.get("artifact_binary_rng_binding_count") != len(binary_bindings)
        or audit.get("artifact_binary_rng_binding_sha256")
        != _canonical_sha256(binary_bindings)
    ):
        raise RuntimeError("RNG audit binary binding contracts differ")
    expected_scan_policy_hash = _canonical_sha256(
        {
            "artifact": ARTIFACT_SCAN_CONTRACT,
            "source": SOURCE_SCAN_CONTRACT,
        }
    )
    if (
        audit.get("artifact_scan_policy") != ARTIFACT_SCAN_CONTRACT
        or audit.get("source_scan_policy") != SOURCE_SCAN_CONTRACT
        or audit.get("scan_policy_sha256") != expected_scan_policy_hash
        or audit.get("source_unresolved_rng_expressions") != []
    ):
        raise RuntimeError("RNG audit scanner contract differs")
    reservations = (
        COORDINATED_EXTERNAL_RESERVATIONS
        if external_reservations is None
        else external_reservations
    )
    reservation_sets = {name: set(values) for name, values in reservations.items()}
    default_contract = external_reservations is None
    expected_reservation_contracts = {
        name: _reservation_contract(
            values,
            mapping=(
                CONTROLLED_CLINICAL_V3_RESERVATION_MAPPING
                if default_contract and name == "controlled_clinical_fidelity_v3"
                else None
            ),
        )
        for name, values in sorted(reservation_sets.items())
    }
    external_ids = set().union(*reservation_sets.values()) if reservation_sets else set()
    if audit.get("coordinated_external_reservations") != expected_reservation_contracts:
        raise RuntimeError("RNG audit external reservation contracts differ")
    if (
        audit.get("coordinated_external_rng_id_count") != len(external_ids)
        or audit.get("coordinated_external_rng_id_sha256")
        != _integer_set_sha256(external_ids)
    ):
        raise RuntimeError("RNG audit external reservation set differs")
    prior_ids = artifact_ids | source_ids | external_ids
    if (
        audit.get("prior_rng_id_count") != len(prior_ids)
        or audit.get("prior_rng_id_sha256") != _integer_set_sha256(prior_ids)
    ):
        raise RuntimeError("RNG audit prior-use set differs")
    collisions = audit.get("collisions")
    if not isinstance(collisions, dict):
        raise RuntimeError("RNG audit collisions must be a mapping")
    if audit.get("collision_count") != len(collisions):
        raise RuntimeError("RNG audit collision count differs")
    if audit.get("collision_sha256") != _canonical_sha256(collisions):
        raise RuntimeError("RNG audit collision hash differs")
    expected_collisions = {
        label: rng_id for label, rng_id in mapping.items() if rng_id in prior_ids
    }
    if collisions != expected_collisions:
        raise RuntimeError("RNG audit collision mapping differs from prior-use sets")
    unhashed = dict(audit)
    stored_hash = unhashed.pop("audit_sha256", None)
    if stored_hash != _canonical_sha256(unhashed):
        raise RuntimeError("RNG audit canonical hash differs")
    passed = audit.get("status") == "passed_before_launch"
    if passed != (len(collisions) == 0):
        raise RuntimeError("RNG audit status and collisions disagree")


def run_preflight(
    config: NativeSignedGammaBenchmarkConfig,
    *,
    config_path: Path,
    resume: bool = False,
    invocation_argv: Sequence[str] = (),
) -> None:
    """Run or strictly resume the gate-only preflight."""

    output_root = config.output_root.resolve()
    rng_audit = audit_formal_rng_ids(
        config,
        output_root=output_root,
        config_path=config_path,
    )
    source_hash = _experiment_tree_sha256(ROOT)
    source_snapshot = _build_source_snapshot(ROOT)
    if _experiment_tree_sha256(ROOT) != source_hash:
        raise RuntimeError("experiment/source tree changed while building snapshot")
    schema = _artifact_schema()
    metadata = _build_metadata(
        config,
        config_path=config_path,
        rng_audit=rng_audit,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        schema=schema,
        invocation_argv=invocation_argv,
    )
    _assert_field_firewall(metadata)
    _assert_field_firewall(schema)

    if resume and (output_root / "COMPLETE").is_file():
        validate_completed_bundle(
            output_root,
            expected_metadata=metadata,
            source_root=ROOT,
        )
        return
    _prepare_root(
        output_root,
        metadata=metadata,
        schema=schema,
        source_snapshot=source_snapshot,
        resume=resume,
    )

    mapping = formal_rng_mapping(config)
    devices_by_label = seed_device_mapping(config)
    existing = _load_existing_seed_artifacts(
        output_root,
        mapping=mapping,
        devices_by_label=devices_by_label,
        config_hash=metadata["config_payload_sha256"],
        audit_hash=rng_audit["audit_sha256"],
        source_hash=source_hash,
    )
    pending = [(label, rng_id) for label, rng_id in mapping.items() if label not in existing]
    results = dict(existing)
    if pending:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(config.devices),
            mp_context=context,
        ) as executor:
            future_to_label = {
                executor.submit(
                    _run_seed,
                    config,
                    label,
                    rng_id,
                    devices_by_label[label],
                    metadata["config_payload_sha256"],
                    rng_audit["audit_sha256"],
                    source_hash,
                ): label
                for label, rng_id in pending
            }
            for future in as_completed(future_to_label):
                label = future_to_label[future]
                artifact = future.result()
                _validate_seed_artifact(
                    artifact,
                    expected_label=label,
                    expected_rng_id=mapping[label],
                    expected_device=devices_by_label[label],
                    config_hash=metadata["config_payload_sha256"],
                    audit_hash=rng_audit["audit_sha256"],
                    source_hash=source_hash,
                )
                results[label] = artifact
                _write_json(
                    output_root / "mechanism" / f"seed_{mapping[label]}.json",
                    artifact,
                )

    summary = _summarize_results(config, results, rng_audit["audit_sha256"])
    _assert_field_firewall(summary)
    _write_json(output_root / "summary.json", summary)
    _finalize_root(output_root, metadata=metadata, summary=summary)


def _run_seed(
    config: NativeSignedGammaBenchmarkConfig,
    rng_label: str,
    rng_id: int,
    device: str,
    config_hash: str,
    audit_hash: str,
    source_hash: str,
) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    probe = mechanism_probe(config, seed=rng_id, device=device)
    return {
        "protocol": config.protocol,
        "rng_label": rng_label,
        "rng_id": rng_id,
        "config_payload_sha256": config_hash,
        "rng_audit_sha256": audit_hash,
        "device": device,
        "source_tree_sha256": source_hash,
        "probe": probe,
    }


def _summarize_results(
    config: NativeSignedGammaBenchmarkConfig,
    results: Mapping[str, Mapping[str, Any]],
    audit_hash: str,
) -> dict[str, Any]:
    mapping = formal_rng_mapping(config)
    if set(results) != set(mapping):
        missing = sorted(set(mapping) - set(results))
        raise RuntimeError(f"cannot summarize missing preflight RNG labels: {missing}")
    passed = [
        mapping[label]
        for label in mapping
        if seed_passes_mechanism_gate(dict(results[label]["probe"]), config.gate)
    ]
    required = int(np.ceil(config.gate.minimum_available_seed_fraction * len(mapping)))
    return {
        "protocol": config.protocol,
        "gate_only": True,
        "n_prespecified": len(mapping),
        "n_passed": len(passed),
        "passed_rng_ids": passed,
        "required_passed_rng_ids": required,
        "status": "GO" if len(passed) >= required else "NO_GO",
        "failure_consequence": "no downstream benchmark artifacts",
        "config_payload_sha256": _canonical_sha256(config.to_dict()),
        "rng_audit_sha256": audit_hash,
    }


def _build_metadata(
    config: NativeSignedGammaBenchmarkConfig,
    *,
    config_path: Path,
    rng_audit: Mapping[str, Any],
    source_hash: str,
    source_snapshot: Mapping[str, Any],
    schema: Mapping[str, Any],
    invocation_argv: Sequence[str],
    source_root: Path = ROOT,
) -> dict[str, Any]:
    config_payload = config.to_dict()
    devices_by_label = seed_device_mapping(config)
    environment = _runtime_environment()
    invocation = {
        "argv": [Path(__file__).relative_to(ROOT).as_posix(), *invocation_argv],
        "cwd": str(Path.cwd().resolve()),
    }
    environment_hash = _canonical_sha256(environment)
    invocation_hash = _canonical_sha256(invocation)
    schema_hash = _canonical_sha256(schema)
    launch_hash = _canonical_sha256(
        {
            "config_payload_sha256": _canonical_sha256(config_payload),
            "rng_audit_sha256": rng_audit["audit_sha256"],
            "seed_device_mapping_sha256": _canonical_sha256(devices_by_label),
            "source_tree_sha256": source_hash,
            "environment_sha256": environment_hash,
            "invocation_sha256": invocation_hash,
            "artifact_schema_sha256": schema_hash,
        }
    )
    return {
        "protocol": config.protocol,
        "role": "gate_only_native_signed_gamma_preflight",
        "gate_only": True,
        "config": config_payload,
        "config_path": _project_path(config_path),
        "config_file_sha256": _file_sha256(config_path),
        "config_payload_sha256": _canonical_sha256(config_payload),
        "rng_audit": dict(rng_audit),
        "seed_device_mapping": devices_by_label,
        "seed_device_mapping_sha256": _canonical_sha256(devices_by_label),
        "source_tree_sha256": source_hash,
        "source_snapshot": dict(source_snapshot),
        "dependency_files": {
            "native_module": _source_contract(
                source_root / "src/scpcp/native_signed_gamma.py", source_root
            ),
            "simulator": _source_contract(
                source_root / "src/scpcp/simulator.py", source_root
            ),
            "runner": _source_contract(
                source_root / "scripts/run_native_synthetic_signed_gamma_preflight.py",
                source_root,
            ),
            "config": _source_contract(config_path, source_root),
            "artifact_io": _source_contract(
                source_root / "src/scpcp/artifacts.py", source_root
            ),
            "project": _source_contract(source_root / "pyproject.toml", source_root),
        },
        "environment": environment,
        "environment_sha256": environment_hash,
        "invocation": invocation,
        "invocation_sha256": invocation_hash,
        "launch_contract_sha256": launch_hash,
        "artifact_schema_sha256": schema_hash,
        "information_firewall": {
            "gate_observables_only": True,
            "allowed": [
                "policy TV",
                "expected action-coordinate shift",
                "difficulty prevalence",
                "tail prevalence",
                "prefix ESS",
                "incremental action ratio",
                "normalized prefix-weight share",
                "finite/simplex/binary/time/kernel invariants",
            ],
        },
        "semantics": {
            "legacy_beta_reused_or_renamed": False,
            "interaction": "gamma * r(A_t) * observed_H_t",
            "gamma_zero": "exact paired action-invariant kernel placebo",
            "source_target_kernel": "same K_gamma instance within each cell",
            "radius_path": "radius -> target action probabilities only",
        },
        "downstream_authorization": {
            "mechanism_rng_ids_only": True,
            "future_benchmark_rng_mapping_present": False,
            "future_benchmark_execution_authorized": False,
        },
    }


def _artifact_schema() -> dict[str, Any]:
    return {
        "protocol": "native_synthetic_signed_gamma_preflight_artifact_schema_v1",
        "metadata_fields": list(METADATA_FIELDS),
        "summary_fields": list(SUMMARY_FIELDS),
        "seed_artifact_fields": list(SEED_ARTIFACT_FIELDS),
        "seed_probe_fields": [
            "protocol",
            "seed",
            "preflight_only",
            "primary_gamma",
            "gamma_rows",
        ],
        "field_firewall": "recursive key-name audit before publication and validation",
    }


def _prepare_root(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    schema: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    resume: bool,
) -> None:
    config = NativeSignedGammaBenchmarkConfig.from_yaml(
        _resolve_recorded_path(str(metadata["config_path"]))
    )
    config = config.with_overrides(
        devices=tuple(metadata["config"]["devices"]),
        output_root=Path(metadata["config"]["output_root"]),
    )
    validate_rng_audit(config, metadata["rng_audit"])
    if metadata["rng_audit"]["status"] != "passed_before_launch":
        raise RuntimeError("live RNG audit did not pass before launch")
    if resume:
        if not root.is_dir():
            raise FileNotFoundError("resume requires an existing output root")
        stored_metadata = _read_json(root / "metadata.json")
        if stored_metadata != metadata:
            raise RuntimeError("resume metadata differs from the live launch contract")
        if _read_json(root / "artifact_schema.json") != schema:
            raise RuntimeError("resume artifact schema differs")
        _verify_source_snapshot(root, metadata["source_snapshot"])
        allowed = _expected_bundle_paths(metadata) | {"manifest.json", "COMPLETE"}
        observed = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        unexpected = sorted(observed - allowed)
        if unexpected:
            raise RuntimeError(f"unexpected resume artifacts: {unexpected}")
        return
    if root.exists():
        raise FileExistsError(f"fresh preflight output already exists: {root}")
    root.mkdir(parents=True)
    (root / "mechanism").mkdir()
    _atomic_write(
        root / source_snapshot["contract"]["archive_path"],
        source_snapshot["archive_bytes"],
    )
    _atomic_write(
        root / source_snapshot["contract"]["manifest_path"],
        source_snapshot["manifest_bytes"],
    )
    _write_json(root / "artifact_schema.json", schema)
    _write_json(root / "metadata.json", metadata)
    _verify_source_snapshot(root, metadata["source_snapshot"])


def _load_existing_seed_artifacts(
    root: Path,
    *,
    mapping: Mapping[str, int],
    devices_by_label: Mapping[str, str],
    config_hash: str,
    audit_hash: str,
    source_hash: str,
) -> dict[str, dict[str, Any]]:
    expected_by_id = {rng_id: label for label, rng_id in mapping.items()}
    results: dict[str, dict[str, Any]] = {}
    mechanism_root = root / "mechanism"
    for path in sorted(mechanism_root.iterdir()):
        if not path.is_file():
            raise RuntimeError(f"unexpected preflight mechanism artifact: {path.name}")
        match = SEED_ARTIFACT_NAME.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"unexpected preflight seed artifact name: {path.name}")
        rng_id = int(match.group(1))
        if rng_id not in expected_by_id:
            raise RuntimeError(f"unexpected preflight RNG ID: {rng_id}")
        label = expected_by_id[rng_id]
        artifact = _read_json(path)
        _validate_seed_artifact(
            artifact,
            expected_label=label,
            expected_rng_id=rng_id,
            expected_device=devices_by_label[label],
            config_hash=config_hash,
            audit_hash=audit_hash,
            source_hash=source_hash,
        )
        if label in results:
            raise RuntimeError(f"duplicate preflight RNG label: {label}")
        results[label] = artifact
    return results


def _validate_seed_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_label: str,
    expected_rng_id: int,
    expected_device: str,
    config_hash: str,
    audit_hash: str,
    source_hash: str,
) -> None:
    if set(artifact) != set(SEED_ARTIFACT_FIELDS):
        raise RuntimeError("preflight seed artifact fields differ")
    if (
        artifact.get("protocol")
        != NativeSignedGammaBenchmarkConfig().protocol
        or artifact.get("rng_label") != expected_label
        or artifact.get("rng_id") != expected_rng_id
        or artifact.get("config_payload_sha256") != config_hash
        or artifact.get("rng_audit_sha256") != audit_hash
        or artifact.get("device") != expected_device
        or artifact.get("source_tree_sha256") != source_hash
    ):
        raise RuntimeError(f"preflight seed artifact identity differs: {expected_label}")
    probe = artifact.get("probe")
    if not isinstance(probe, dict) or set(probe) != {
        "protocol",
        "seed",
        "preflight_only",
        "primary_gamma",
        "gamma_rows",
    }:
        raise RuntimeError(f"preflight probe fields differ: {expected_label}")
    if (
        probe.get("protocol") != artifact["protocol"]
        or probe.get("seed") != expected_rng_id
        or probe.get("preflight_only") is not True
    ):
        raise RuntimeError(f"preflight probe identity differs: {expected_label}")
    _validate_probe_payload(probe, expected_seed=expected_rng_id)
    _assert_field_firewall(artifact)


def _validate_probe_payload(probe: Mapping[str, Any], *, expected_seed: int) -> None:
    """Validate the exact coverage-blind mechanism-probe schema."""

    if (
        probe.get("seed") != expected_seed
        or probe.get("primary_gamma") != -4.0
        or not isinstance(probe.get("gamma_rows"), list)
        or len(probe["gamma_rows"]) != len(GAMMAS)
    ):
        raise RuntimeError("preflight probe identity or signed-gamma grid differs")

    common_fields = {
        "gamma",
        "kernel_fingerprint",
        "source_target_kernel_shared",
        "mid_policy_tv",
        "high_policy_tv",
        "mid_expected_action_coordinate_shift",
        "high_expected_action_coordinate_shift",
        "late_difficulty_prevalence_shift",
        "late_tail_prevalence_shift",
        "finite_and_structural",
    }
    scalar_fields = (
        "mid_policy_tv",
        "high_policy_tv",
        "mid_expected_action_coordinate_shift",
        "high_expected_action_coordinate_shift",
        "late_difficulty_prevalence_shift",
        "late_tail_prevalence_shift",
    )
    overlap_fields = {
        "minimum_ess_fraction",
        "maximum_incremental_ratio",
        "maximum_normalized_weight_share",
    }
    for expected_gamma, row in zip(GAMMAS, probe["gamma_rows"], strict=True):
        if not isinstance(row, dict) or row.get("gamma") != expected_gamma:
            raise RuntimeError("preflight gamma rows are not unique and ordered")
        expected_fields = set(common_fields)
        if expected_gamma == 0.0:
            expected_fields.add("exact_placebo")
        if expected_gamma == -4.0:
            expected_fields.add("overlap")
        if set(row) != expected_fields:
            raise RuntimeError("preflight gamma-row schema differs")
        if (
            not isinstance(row["kernel_fingerprint"], str)
            or not row["kernel_fingerprint"]
            or not isinstance(row["source_target_kernel_shared"], bool)
            or not isinstance(row["finite_and_structural"], bool)
        ):
            raise RuntimeError("preflight gamma-row identity fields differ")
        for field in scalar_fields:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise RuntimeError("preflight gamma-row scalar is invalid")
        if not 0.0 <= float(row["mid_policy_tv"]) <= 1.0 or not 0.0 <= float(
            row["high_policy_tv"]
        ) <= 1.0:
            raise RuntimeError("preflight policy TV lies outside [0, 1]")
        for field in (
            "late_difficulty_prevalence_shift",
            "late_tail_prevalence_shift",
        ):
            if not -1.0 <= float(row[field]) <= 1.0:
                raise RuntimeError("preflight prevalence shift lies outside [-1, 1]")

        if expected_gamma == 0.0:
            placebo = row["exact_placebo"]
            if not isinstance(placebo, dict) or set(placebo) != {
                "states",
                "outcomes",
                "tails",
            } or any(not isinstance(value, bool) for value in placebo.values()):
                raise RuntimeError("preflight exact-placebo schema differs")
        if expected_gamma == -4.0:
            overlap = row["overlap"]
            if not isinstance(overlap, dict) or set(overlap) != {"mid", "high"}:
                raise RuntimeError("preflight overlap schema differs")
            for radius in ("mid", "high"):
                metrics = overlap[radius]
                if not isinstance(metrics, dict) or set(metrics) != overlap_fields:
                    raise RuntimeError("preflight overlap metric schema differs")
                for field, value in metrics.items():
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not np.isfinite(value)
                        or float(value) < 0.0
                    ):
                        raise RuntimeError("preflight overlap metric is invalid")
                if not 0.0 <= float(metrics["minimum_ess_fraction"]) <= 1.0:
                    raise RuntimeError("preflight overlap ESS fraction is invalid")
                if not 0.0 <= float(metrics["maximum_normalized_weight_share"]) <= 1.0:
                    raise RuntimeError("preflight overlap weight share is invalid")


def _finalize_root(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    summary: Mapping[str, Any],
    source_root: Path = ROOT,
) -> None:
    if _experiment_tree_sha256(source_root) != metadata["source_tree_sha256"]:
        raise RuntimeError("experiment/source tree changed during preflight")
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("preflight metadata changed during execution")
    if _read_json(root / "summary.json") != summary:
        raise RuntimeError("preflight summary changed before finalization")
    _write_manifest(root, metadata)
    manifest_path = root / "manifest.json"
    complete = {
        "protocol": metadata["protocol"],
        "status": "complete",
        "decision": summary["status"],
        "manifest_sha256": _file_sha256(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "metadata_sha256": _file_sha256(root / "metadata.json"),
        "summary_sha256": _file_sha256(root / "summary.json"),
        "artifact_schema_sha256": _file_sha256(root / "artifact_schema.json"),
        "config_payload_sha256": metadata["config_payload_sha256"],
        "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
        "seed_device_mapping_sha256": metadata["seed_device_mapping_sha256"],
        "source_snapshot_sha256": metadata["source_snapshot"]["archive_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "environment_sha256": metadata["environment_sha256"],
        "invocation_sha256": metadata["invocation_sha256"],
        "launch_contract_sha256": metadata["launch_contract_sha256"],
    }
    _write_json(root / "COMPLETE", complete)
    validate_completed_bundle(root, expected_metadata=metadata, source_root=source_root)


def validate_completed_bundle(
    root: Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    source_root: Path | None = None,
) -> None:
    metadata = _read_json(root / "metadata.json")
    schema = _read_json(root / "artifact_schema.json")
    summary = _read_json(root / "summary.json")
    if set(metadata) != set(METADATA_FIELDS):
        raise RuntimeError("preflight metadata fields differ")
    if set(summary) != set(SUMMARY_FIELDS):
        raise RuntimeError("preflight summary fields differ")
    if schema != _artifact_schema():
        raise RuntimeError("preflight artifact schema differs")
    _assert_field_firewall(metadata)
    _assert_field_firewall(summary)
    _assert_field_firewall(schema)
    if expected_metadata is not None and metadata != expected_metadata:
        raise RuntimeError("preflight metadata differs from the live launch contract")

    active_source_root = ROOT if source_root is None else source_root
    config = _config_from_payload(metadata["config"])
    config_hash = _canonical_sha256(config.to_dict())
    if metadata["config_payload_sha256"] != config_hash:
        raise RuntimeError("preflight config payload hash differs")
    recorded_config = _resolve_recorded_path(
        str(metadata["config_path"]), active_source_root
    )
    if metadata["config_file_sha256"] != _file_sha256(recorded_config):
        raise RuntimeError("preflight config file hash differs")
    validate_rng_audit(config, metadata["rng_audit"])
    if metadata["rng_audit"]["status"] != "passed_before_launch":
        raise RuntimeError("completed preflight has a nonpassing RNG audit")
    if metadata["artifact_schema_sha256"] != _canonical_sha256(schema):
        raise RuntimeError("preflight schema binding differs")
    expected_devices = seed_device_mapping(config)
    if (
        metadata["seed_device_mapping"] != expected_devices
        or metadata["seed_device_mapping_sha256"]
        != _canonical_sha256(expected_devices)
    ):
        raise RuntimeError("preflight seed-device mapping differs")
    if metadata["environment_sha256"] != _canonical_sha256(metadata["environment"]):
        raise RuntimeError("preflight environment hash differs")
    if metadata["invocation_sha256"] != _canonical_sha256(metadata["invocation"]):
        raise RuntimeError("preflight invocation hash differs")
    expected_launch_hash = _canonical_sha256(
        {
            "config_payload_sha256": config_hash,
            "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
            "seed_device_mapping_sha256": metadata["seed_device_mapping_sha256"],
            "source_tree_sha256": metadata["source_tree_sha256"],
            "environment_sha256": metadata["environment_sha256"],
            "invocation_sha256": metadata["invocation_sha256"],
            "artifact_schema_sha256": metadata["artifact_schema_sha256"],
        }
    )
    if metadata["launch_contract_sha256"] != expected_launch_hash:
        raise RuntimeError("preflight launch contract hash differs")
    _verify_source_snapshot(root, metadata["source_snapshot"])
    if _experiment_tree_sha256(active_source_root) != metadata["source_tree_sha256"]:
        raise RuntimeError("active experiment/source tree differs from preflight")
    _verify_dependency_files(metadata["dependency_files"], active_source_root)

    mapping = formal_rng_mapping(config)
    results = _load_existing_seed_artifacts(
        root,
        mapping=mapping,
        devices_by_label=expected_devices,
        config_hash=config_hash,
        audit_hash=metadata["rng_audit"]["audit_sha256"],
        source_hash=metadata["source_tree_sha256"],
    )
    expected_summary = _summarize_results(
        config,
        results,
        metadata["rng_audit"]["audit_sha256"],
    )
    if summary != expected_summary:
        raise RuntimeError("preflight summary does not reconcile with seed artifacts")
    actual_artifacts = {
        path.relative_to(root).as_posix() for path in _iter_bundle_artifacts(root)
    }
    if actual_artifacts != _expected_bundle_paths(metadata):
        raise RuntimeError("preflight bundle artifact set differs")
    manifest_hash = _verify_manifest(root, metadata)
    complete = _read_json(root / "COMPLETE")
    expected_complete = {
        "protocol": metadata["protocol"],
        "status": "complete",
        "decision": summary["status"],
        "manifest_sha256": manifest_hash,
        "manifest_bytes": (root / "manifest.json").stat().st_size,
        "metadata_sha256": _file_sha256(root / "metadata.json"),
        "summary_sha256": _file_sha256(root / "summary.json"),
        "artifact_schema_sha256": _file_sha256(root / "artifact_schema.json"),
        "config_payload_sha256": metadata["config_payload_sha256"],
        "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
        "seed_device_mapping_sha256": metadata["seed_device_mapping_sha256"],
        "source_snapshot_sha256": metadata["source_snapshot"]["archive_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "environment_sha256": metadata["environment_sha256"],
        "invocation_sha256": metadata["invocation_sha256"],
        "launch_contract_sha256": metadata["launch_contract_sha256"],
    }
    if complete != expected_complete:
        raise RuntimeError("preflight COMPLETE hash chain differs")


def _write_manifest(root: Path, metadata: Mapping[str, Any]) -> None:
    records = []
    for path in _iter_bundle_artifacts(root):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    _write_json(
        root / "manifest.json",
        {
            "protocol": metadata["protocol"],
            "config_payload_sha256": metadata["config_payload_sha256"],
            "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
            "source_tree_sha256": metadata["source_tree_sha256"],
            "artifact_count": len(records),
            "artifacts": records,
        },
    )


def _verify_manifest(root: Path, metadata: Mapping[str, Any]) -> str:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    expected_header = {
        "protocol": metadata["protocol"],
        "config_payload_sha256": metadata["config_payload_sha256"],
        "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
    }
    if any(manifest.get(key) != value for key, value in expected_header.items()):
        raise RuntimeError("preflight manifest header differs")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or manifest.get("artifact_count") != len(records):
        raise RuntimeError("preflight manifest records are malformed")
    actual_paths = {
        path.relative_to(root).as_posix() for path in _iter_bundle_artifacts(root)
    }
    listed_paths = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RuntimeError("preflight manifest record fields differ")
        relative = record["path"]
        if relative in listed_paths:
            raise RuntimeError(f"duplicate preflight manifest path: {relative}")
        listed_paths.add(relative)
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"preflight manifest artifact differs: {relative}")
    if listed_paths != actual_paths:
        raise RuntimeError("preflight manifest file set differs")
    return _file_sha256(manifest_path)


def _iter_bundle_artifacts(root: Path) -> list[Path]:
    paths = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "COMPLETE"}:
            continue
        if ".tmp-" in path.name:
            raise RuntimeError(f"temporary preflight artifact remains: {path}")
        paths.append(path)
    return paths


def _expected_bundle_paths(metadata: Mapping[str, Any]) -> set[str]:
    config = _config_from_payload(metadata["config"])
    paths = {
        "artifact_schema.json",
        "metadata.json",
        "summary.json",
        str(metadata["source_snapshot"]["archive_path"]),
        str(metadata["source_snapshot"]["manifest_path"]),
    }
    paths.update(
        f"mechanism/seed_{rng_id}.json"
        for rng_id in formal_rng_mapping(config).values()
    )
    return paths


def _assert_field_firewall(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_FIELD_TOKENS):
                location = ".".join((*path, str(key)))
                raise RuntimeError(f"forbidden result field in preflight artifact: {location}")
            _assert_field_firewall(nested, (*path, str(key)))
        return
    if isinstance(value, (list, tuple)):
        field_list = bool(path and path[-1].endswith("_fields"))
        for index, nested in enumerate(value):
            if field_list and isinstance(nested, str):
                lowered = nested.lower()
                if any(token in lowered for token in FORBIDDEN_FIELD_TOKENS):
                    location = ".".join((*path, str(index)))
                    raise RuntimeError(
                        f"forbidden result field in preflight artifact: {location}"
                    )
            _assert_field_firewall(nested, (*path, str(index)))


def _build_source_snapshot(source_root: Path) -> dict[str, Any]:
    paths = _experiment_paths(source_root)
    files = []
    archive_stream = io.BytesIO()
    with tarfile.open(fileobj=archive_stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            relative = path.relative_to(source_root).as_posix()
            content = path.read_bytes()
            files.append(
                {
                    "path": relative,
                    "size_bytes": len(content),
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
        "protocol": NativeSignedGammaBenchmarkConfig().protocol,
        "format": "deterministic_uncompressed_pax_tar",
        "file_count": len(files),
        "files": files,
    }
    manifest_bytes = _canonical_json_bytes(manifest) + b"\n"
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


def _verify_source_snapshot(root: Path, contract: Mapping[str, Any]) -> None:
    for name in ("archive", "manifest"):
        path = root / str(contract[f"{name}_path"])
        if (
            not path.is_file()
            or path.stat().st_size != contract[f"{name}_bytes"]
            or _file_sha256(path) != contract[f"{name}_sha256"]
        ):
            raise RuntimeError(f"preflight source snapshot {name} differs")
    manifest = _read_json(root / str(contract["manifest_path"]))
    if manifest.get("file_count") != contract["file_count"]:
        raise RuntimeError("preflight source snapshot file count differs")
    paths = [row.get("path") for row in manifest.get("files", [])]
    if len(paths) != len(set(paths)) or "src/scpcp/simulator.py" not in paths:
        raise RuntimeError("preflight source snapshot dependency set differs")


def _experiment_paths(root: Path) -> list[Path]:
    paths = [
        *sorted((root / "src/scpcp").rglob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
        *sorted((root / "tools").glob("*.py")),
        *sorted((root / "configs").glob("*.yaml")),
        root / "pyproject.toml",
    ]
    if not paths or any(not path.is_file() for path in paths):
        raise RuntimeError("experiment/source tree file set is incomplete")
    relative = [path.relative_to(root).as_posix() for path in paths]
    if len(relative) != len(set(relative)):
        raise RuntimeError("experiment/source tree contains duplicate paths")
    return paths


def _experiment_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _experiment_paths(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _verify_dependency_files(
    dependencies: Mapping[str, Mapping[str, Any]],
    source_root: Path,
) -> None:
    simulator_seen = False
    for contract in dependencies.values():
        path = source_root / str(contract["path"])
        simulator_seen |= contract["path"] == "src/scpcp/simulator.py"
        if (
            not path.is_file()
            or path.stat().st_size != contract["size_bytes"]
            or _file_sha256(path) != contract["sha256"]
        ):
            raise RuntimeError(f"active dependency differs: {contract['path']}")
    if not simulator_seen:
        raise RuntimeError("simulator.py is missing from dependency provenance")


def _runtime_environment() -> dict[str, Any]:
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
        "git_revision": git_revision(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _empty_rng_scan() -> dict[str, Any]:
    return {
        "actual": set(),
        "declared": set(),
        "reserved": set(),
        "binary_bindings": [],
    }


def _artifact_actual_rng_ids(root: Path, *, excluded_root: Path) -> set[int]:
    return _artifact_rng_scan(root, excluded_root=excluded_root)["actual"]


def _artifact_rng_scan(root: Path, *, excluded_root: Path) -> dict[str, Any]:
    report = _empty_rng_scan()
    if not root.exists():
        return report
    root_resolved = root.resolve()
    excluded = excluded_root.resolve()
    for path in sorted(root.rglob("*")):
        resolved = path.resolve()
        if not _is_relative_to(resolved, root_resolved):
            raise RuntimeError(f"artifact scan path escapes its root: {path}")
        if _is_relative_to(resolved, excluded):
            continue
        match = SEED_ARTIFACT_NAME.fullmatch(path.name)
        if match:
            report["actual"].add(int(match.group(1)))
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in STRUCTURED_ARTIFACT_SUFFIXES or path.name == "COMPLETE":
            payload = _read_structured_artifact(path)
            _collect_artifact_rng_fields(
                payload,
                report,
                artifact_path=path,
                artifact_root=root_resolved,
            )
        elif suffix in TABULAR_ARTIFACT_SUFFIXES:
            _collect_tabular_rng_fields(path, report)
    report["binary_bindings"] = sorted(
        report["binary_bindings"],
        key=lambda row: (row["metadata_path"], row["field_path"]),
    )
    return report


def _read_structured_artifact(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(text)
        if path.name == "COMPLETE":
            stripped = text.strip()
            if not stripped or stripped.lower() == "complete":
                return {}
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                rows: dict[str, Any] = {}
                for line in stripped.splitlines():
                    if "=" not in line:
                        continue
                    key, raw = line.split("=", 1)
                    raw = raw.strip()
                    rows[key.strip()] = int(raw) if re.fullmatch(r"[+-]?\d+", raw) else raw
                return rows
        return json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise RuntimeError(f"cannot parse structured artifact during RNG audit: {path}") from error


def _collect_artifact_rng_fields(
    value: Any,
    report: dict[str, Any],
    *,
    artifact_path: Path,
    artifact_root: Path,
    key_path: tuple[str, ...] = (),
    inherited_role: str | None = None,
) -> None:
    if isinstance(value, dict):
        _verify_declared_binary_rng_bindings(
            value,
            report,
            artifact_path=artifact_path,
            artifact_root=artifact_root,
            key_path=key_path,
        )
        for key, nested in value.items():
            name = str(key)
            if (
                key_path
                and key_path[-1] == "dependency_files"
                and isinstance(nested, dict)
                and set(nested) == {"path", "sha256", "size_bytes"}
            ):
                # Dependency labels such as ``experiment_rng`` identify source
                # files; the source contract is provenance, not an RNG use.
                continue
            if name.lower() == "rng" and isinstance(nested, dict):
                # A structured RNG config contains both actual IDs and
                # descriptors such as mapping hashes.  Let each child field
                # retain its own strict role instead of coercing the dict as
                # one scalar RNG value.
                _collect_artifact_rng_fields(
                    nested,
                    report,
                    artifact_path=artifact_path,
                    artifact_root=artifact_root,
                    key_path=(*key_path, name),
                    inherited_role=inherited_role,
                )
                continue
            role = inherited_role
            if RESERVATION_FIELD.search(name):
                role = "reserved"
            elif DECLARATION_FIELD.search(name):
                role = "declared"
            seed_key_context = any("per_seed" in part.lower() for part in key_path)
            key_ids = _rng_ids_from_mapping_key(name)
            if key_ids and (
                seed_key_context
                or re.search(r"(?:^|[/_-])seed[_-]?\d+(?:$|[/_-])", name, re.IGNORECASE)
            ):
                report[role or "actual"].update(key_ids)
            field_role = _rng_field_role(name)
            if field_role is not None:
                destination = role or field_role
                if _is_rng_formula_mapping(name, nested):
                    destination = "declared" if role is None else role
                ids = _artifact_rng_field_ids(
                    nested,
                    field_name=name,
                    artifact_path=artifact_path,
                )
                report[destination].update(ids)
                continue
            _collect_artifact_rng_fields(
                nested,
                report,
                artifact_path=artifact_path,
                artifact_root=artifact_root,
                key_path=(*key_path, name),
                inherited_role=role,
            )
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _collect_artifact_rng_fields(
                nested,
                report,
                artifact_path=artifact_path,
                artifact_root=artifact_root,
                key_path=(*key_path, str(index)),
                inherited_role=inherited_role,
            )


def _rng_field_role(name: str) -> str | None:
    lowered = name.lower()
    if re.fullmatch(r"seed[_-]?\d+", lowered):
        # The integer is carried by the mapping key itself; descendants may be
        # ordinary metrics and must be scanned recursively on their own terms.
        return None
    if lowered.endswith(RNG_DESCRIPTOR_SUFFIXES) or "per_seed" in lowered:
        return None
    if RESERVATION_FIELD.search(lowered) and RNG_FIELD.search(lowered):
        return "reserved"
    if DECLARATION_FIELD.search(lowered) and RNG_FIELD.search(lowered):
        return "declared"
    if lowered in {
        "seed",
        "seeds",
        "seed_list",
        "seed_lists",
        "rng",
        "rngs",
        "rng_id",
        "rng_ids",
        "random_state",
        "random_states",
    }:
        return "actual"
    if lowered.endswith(
        (
            "_seed",
            "_seeds",
            "_seed_list",
            "_seed_lists",
            "_rng",
            "_rng_id",
            "_rng_ids",
            "_random_state",
            "_random_states",
        )
    ):
        return "actual"
    if (
        RNG_FIELD.search(lowered)
        and lowered.endswith(
            (
                "_bank",
                "_block",
                "_list",
                "_mapping",
                "_range",
                "_range_inclusive",
                "_start",
                "_states",
            )
        )
    ):
        return "actual"
    if re.search(r"(?:seed|rng)_to_", lowered):
        return "actual"
    return None


def _is_rng_formula_mapping(name: str, value: Any) -> bool:
    if not isinstance(value, dict) or "stream" not in name.lower():
        return False
    leaves = list(_nested_leaves(value))
    return bool(leaves) and all(isinstance(item, str) for item in leaves)


def _nested_leaves(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from _nested_leaves(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _nested_leaves(nested)
    else:
        yield value


def _artifact_rng_field_ids(
    value: Any,
    *,
    field_name: str,
    artifact_path: Path,
) -> set[int]:
    if isinstance(value, bool):
        lowered = field_name.lower()
        if (
            lowered not in {"seed", "seeds", "rng", "rng_id", "rng_ids", "random_state"}
            and not lowered.endswith(("_seed", "_rng", "_rng_id"))
        ):
            return set()
        raise RuntimeError(
            f"non-integer RNG value for {field_name!r} in artifact: {artifact_path}"
        )
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple)):
        ids: set[int] = set()
        for nested in value:
            ids.update(
                _artifact_rng_field_ids(
                    nested,
                    field_name=field_name,
                    artifact_path=artifact_path,
                )
            )
        return ids
    if isinstance(value, dict):
        ids: set[int] = set()
        for key, nested in value.items():
            key_ids = _rng_ids_from_mapping_key(str(key))
            ids.update(key_ids)
            if isinstance(nested, str) and key_ids:
                continue
            if isinstance(nested, str) and _is_rng_formula_mapping(
                field_name,
                {str(key): nested},
            ):
                ids.update(int(item) for item in re.findall(r"(?<![.\d])-?\d+(?![.\d])", nested))
                continue
            ids.update(
                _artifact_rng_field_ids(
                    nested,
                    field_name=field_name,
                    artifact_path=artifact_path,
                )
            )
        return ids
    raise RuntimeError(
        f"non-integer RNG value for {field_name!r} in artifact: {artifact_path}"
    )


def _rng_ids_from_mapping_key(value: str) -> set[int]:
    if re.fullmatch(r"[+-]?\d+", value):
        return {int(value)}
    ids = {
        int(match)
        for match in re.findall(
            r"(?:base|problem|rng|seed)[_-](\d+)(?=$|[/_-])",
            value,
            re.IGNORECASE,
        )
    }
    ids.update(int(match) for match in re.findall(r"(?:^|/)(\d+)(?=$|/)", value))
    return ids


def _collect_tabular_rng_fields(path: Path, report: dict[str, Any]) -> None:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter, strict=True)
            roles = {
                name: _rng_field_role(name)
                for name in (reader.fieldnames or [])
                if _rng_field_role(name) is not None
            }
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise RuntimeError(
                        f"malformed tabular artifact at {path}:{row_number}"
                    )
                for name, role in roles.items():
                    raw = (row.get(name) or "").strip()
                    if raw.lower() in MISSING_TABULAR_VALUES:
                        continue
                    if not re.fullmatch(r"[+-]?\d+", raw):
                        raise RuntimeError(
                            f"non-integer RNG value at {path}:{row_number}:{name}"
                        )
                    report[role].add(int(raw))
    except (OSError, UnicodeError, csv.Error) as error:
        raise RuntimeError(f"cannot parse tabular artifact during RNG audit: {path}") from error


def _verify_declared_binary_rng_bindings(
    value: Mapping[str, Any],
    report: dict[str, Any],
    *,
    artifact_path: Path,
    artifact_root: Path,
    key_path: tuple[str, ...],
) -> None:
    for key, nested in value.items():
        name = str(key)
        if not name.lower().endswith("_path") or not isinstance(nested, str):
            continue
        if Path(nested).suffix.lower() not in {".npy", ".npz"}:
            continue
        semantic_path = ".".join((*key_path, name, nested))
        if not re.search(r"(?:seed|rng|random|bootstrap)", semantic_path, re.I):
            continue
        is_recorded_binding = (
            name.lower() == "binary_path"
            and "artifact_binary_rng_bindings" in key_path
            and isinstance(value.get("field_path"), str)
        )
        hash_key = "sha256" if is_recorded_binding else f"{name[:-5]}_sha256"
        expected_hash = value.get(hash_key)
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise RuntimeError(
                f"RNG binary binding lacks a SHA256 at {artifact_path}:{semantic_path}"
            )
        binary_base = artifact_root if is_recorded_binding else artifact_path.parent
        binary_path = (binary_base / nested).resolve()
        if not _is_relative_to(binary_path, artifact_root):
            raise RuntimeError(f"RNG binary binding escapes artifact root: {binary_path}")
        if not binary_path.is_file() or _file_sha256(binary_path) != expected_hash:
            raise RuntimeError(f"RNG binary binding differs: {binary_path}")
        report["binary_bindings"].append(
            {
                "metadata_path": artifact_path.resolve().relative_to(artifact_root).as_posix(),
                "field_path": ".".join((*key_path, name)),
                "binary_path": binary_path.relative_to(artifact_root).as_posix(),
                "sha256": expected_hash,
            }
        )


def _source_actual_rng_ids(
    root: Path,
    *,
    excluded_paths: set[Path],
) -> set[int]:
    return _source_rng_scan(root, excluded_paths=excluded_paths)["actual"]


def _source_rng_scan(root: Path, *, excluded_paths: set[Path]) -> dict[str, Any]:
    report = _empty_rng_scan()
    root_resolved = root.resolve()
    excluded = {path.resolve() for path in excluded_paths}
    for directory in ("scripts", "src", "tools"):
        base = root / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            resolved = path.resolve()
            if not _is_relative_to(resolved, root_resolved):
                raise RuntimeError(f"source scan path escapes its root: {path}")
            if resolved in excluded:
                continue
            _collect_python_rng_report(path, report)
    return report


_UNKNOWN_STATIC = object()
_DYNAMIC_STATIC = object()
_UNRESOLVED_SEED = object()


class _SourceRngAnalyzer(ast.NodeVisitor):
    def __init__(self, path: Path, report: dict[str, Any]) -> None:
        self.path = path
        self.report = report
        self.environments: list[dict[str, Any]] = [{}]

    @property
    def environment(self) -> dict[str, Any]:
        return self.environments[-1]

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value = _static_rng_value(node.value, self.environment)
        for target in node.targets:
            self._bind_target(target, value, node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        self.visit(node.value)
        self._bind_target(
            node.target,
            _static_rng_value(node.value, self.environment),
            node,
        )

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            expression = ast.BinOp(
                left=ast.Name(id=node.target.id, ctx=ast.Load()),
                op=node.op,
                right=node.value,
            )
            self._bind_name(
                node.target.id,
                _static_rng_value(expression, self.environment),
                node,
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for expression in (*node.decorator_list, *node.args.defaults):
            self.visit(expression)
        for expression in node.args.kw_defaults:
            if expression is not None:
                self.visit(expression)
        local = dict(self.environment)
        positional = [*node.args.posonlyargs, *node.args.args]
        default_offset = len(positional) - len(node.args.defaults)
        for index, argument in enumerate(positional):
            if index < default_offset:
                local[argument.arg] = _DYNAMIC_STATIC
            else:
                default = _static_rng_value(
                    node.args.defaults[index - default_offset],
                    self.environment,
                )
                local[argument.arg] = _DYNAMIC_STATIC if default is None else default
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            parsed = None if default is None else _static_rng_value(
                default,
                self.environment,
            )
            local[argument.arg] = (
                _DYNAMIC_STATIC if parsed is None else parsed
            )
        if node.args.vararg is not None:
            local[node.args.vararg.arg] = _DYNAMIC_STATIC
        if node.args.kwarg is not None:
            local[node.args.kwarg.arg] = _DYNAMIC_STATIC
        self.environments.append(local)
        for statement in node.body:
            self.visit(statement)
        self.environments.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases, *[item.value for item in node.keywords]):
            self.visit(expression)
        self.environments.append(dict(self.environment))
        for statement in node.body:
            self.visit(statement)
        self.environments.pop()

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._bind_loop_target(
            node.target,
            _static_rng_value(node.iter, self.environment),
            node.iter,
        )
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_If(self, node: ast.If) -> None:
        """Keep every statically possible binding across control-flow branches."""

        self.visit(node.test)
        baseline = dict(self.environment)
        branches: list[dict[str, Any]] = []
        for statements in (node.body, node.orelse):
            branch = dict(baseline)
            self.environments.append(branch)
            for statement in statements:
                self.visit(statement)
            branches.append(dict(self.environment))
            self.environments.pop()
        if not node.orelse:
            branches[1] = baseline
        merged: dict[str, Any] = {}
        for name in set().union(*(branch.keys() for branch in branches)):
            merged[name] = _merge_branch_static_values(
                tuple(branch.get(name, _UNKNOWN_STATIC) for branch in branches)
            )
        self.environments[-1] = merged

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, (node.key, node.value))

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        expressions: tuple[ast.AST, ...],
    ) -> None:
        self.environments.append(dict(self.environment))
        for generator in generators:
            self.visit(generator.iter)
            self._bind_loop_target(
                generator.target,
                _static_rng_value(generator.iter, self.environment),
                generator.iter,
            )
            for condition in generator.ifs:
                self.visit(condition)
        for expression in expressions:
            self.visit(expression)
        self.environments.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        for candidate in _rng_call_candidates(name, node):
            value = _static_rng_value(candidate, self.environment)
            ids = _static_integer_values(value)
            if ids is not None:
                self.report["actual"].update(ids)
            elif _unresolved_concrete_seed(candidate, value, self.environment):
                raise RuntimeError(
                    "cannot resolve seed-like RNG expression at "
                    f"{self.path}:{getattr(candidate, 'lineno', node.lineno)}"
                )
        self.generic_visit(node)

    def _bind_target(self, target: ast.AST, value: Any, node: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self._bind_name(target.id, value, node)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            items = value if isinstance(value, tuple) else ()
            for index, nested in enumerate(target.elts):
                child = items[index] if index < len(items) else _UNKNOWN_STATIC
                self._bind_target(nested, child, node)

    def _bind_loop_target(self, target: ast.AST, value: Any, node: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, tuple):
            rows = [item for item in value if isinstance(item, tuple)]
            if len(rows) == len(value) and rows:
                for index, nested in enumerate(target.elts):
                    column = tuple(row[index] for row in rows if index < len(row))
                    self._bind_target(nested, column, node)
                return
        self._bind_target(target, value, node)

    def _bind_name(self, name: str, value: Any, node: ast.AST) -> None:
        if RNG_FIELD.search(name) and value is _UNKNOWN_STATIC:
            value = (
                _UNRESOLVED_SEED
                if _ast_has_concrete_integer(node, self.environment)
                else _DYNAMIC_STATIC
            )
        self.environment[name] = value
        ids = _static_integer_values(value)
        if ids is None or not RNG_FIELD.search(name):
            return
        destination = (
            "reserved"
            if RESERVATION_FIELD.search(name)
            else "declared"
        )
        self.report[destination].update(ids)


def _collect_python_rng_report(path: Path, report: dict[str, Any]) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise RuntimeError(f"cannot parse source RNG use: {path}") from error
    _SourceRngAnalyzer(path, report).visit(tree)


def _rng_call_candidates(name: str, node: ast.Call) -> list[ast.AST]:
    basename = name.rsplit(".", 1)[-1]
    direct_consumers = {
        "manual_seed",
        "default_rng",
        "Generator",
        "Random",
        "RandomState",
        "seed",
    }
    is_direct_consumer = basename in direct_consumers or basename.startswith(
        "manual_seed"
    )
    candidates = list(node.args[:1]) if is_direct_consumer else []
    if RNG_CALL.search(name):
        candidates.extend(
            keyword.value
            for keyword in node.keywords
            if keyword.arg is not None and _rng_field_role(keyword.arg) is not None
        )
    return candidates


def _merge_branch_static_values(values: tuple[Any, ...]) -> Any:
    if all(value == values[0] for value in values[1:]):
        return values[0]
    if any(value is _UNRESOLVED_SEED for value in values):
        # A concrete-but-unresolved seed expression is a hard error even when
        # another branch has a fully resolved value.
        return _UNRESOLVED_SEED
    integer_sets = tuple(_static_integer_values(value) for value in values)
    concrete = set().union(
        *(value or set() for value in integer_sets if value is not None)
    )
    if concrete:
        # A dynamic branch does not erase constants reachable through another
        # branch.  The scanner must still report every statically possible ID.
        return tuple(sorted(concrete))
    if any(value is _DYNAMIC_STATIC for value in values):
        return _DYNAMIC_STATIC
    return _UNKNOWN_STATIC


def _static_rng_value(node: ast.AST, environment: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return _UNKNOWN_STATIC
        if isinstance(node.value, (int, float, str)) or node.value is None:
            return node.value
        return _UNKNOWN_STATIC
    if isinstance(node, ast.Name):
        return environment.get(node.id, _UNKNOWN_STATIC)
    if isinstance(node, ast.Attribute):
        return _DYNAMIC_STATIC
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _static_rng_value(node.operand, environment)
        if isinstance(operand, (int, float)) and not isinstance(operand, bool):
            return -operand if isinstance(node.op, ast.USub) else operand
        values = _static_integer_values(operand)
        if values is None:
            return _UNKNOWN_STATIC
        return tuple((-value if isinstance(node.op, ast.USub) else value) for value in values)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = tuple(_static_rng_value(element, environment) for element in node.elts)
        if any(value is _DYNAMIC_STATIC for value in values):
            return _DYNAMIC_STATIC
        return values if all(value is not _UNKNOWN_STATIC for value in values) else _UNKNOWN_STATIC
    if isinstance(node, ast.Dict):
        keys = [_static_rng_value(key, environment) for key in node.keys]
        values = [_static_rng_value(value, environment) for value in node.values]
        if any(value is _DYNAMIC_STATIC for value in (*keys, *values)):
            return _DYNAMIC_STATIC
        if any(value is _UNKNOWN_STATIC for value in (*keys, *values)):
            return _UNKNOWN_STATIC
        try:
            return dict(zip(keys, values))
        except TypeError:
            return _UNKNOWN_STATIC
    if isinstance(node, ast.Subscript):
        container = _static_rng_value(node.value, environment)
        index = _static_rng_value(node.slice, environment)
        if (
            container is _UNKNOWN_STATIC
            or container is _DYNAMIC_STATIC
            or container is _UNRESOLVED_SEED
        ):
            return container
        if index is _DYNAMIC_STATIC:
            if isinstance(container, dict):
                candidates = tuple(container.values())
            elif isinstance(container, (tuple, list)):
                candidates = container
            else:
                return _DYNAMIC_STATIC
            if isinstance(candidates, (tuple, list)):
                flattened: list[Any] = []
                for candidate in candidates:
                    flattened.extend(candidate if isinstance(candidate, tuple) else (candidate,))
                return tuple(flattened)
        if not isinstance(index, (int, str)) or isinstance(index, bool):
            return _UNKNOWN_STATIC
        try:
            return container[index]
        except (IndexError, KeyError, TypeError):
            return _UNKNOWN_STATIC
    if isinstance(node, ast.IfExp):
        body = _static_rng_value(node.body, environment)
        alternate = _static_rng_value(node.orelse, environment)
        if body is _DYNAMIC_STATIC or alternate is _DYNAMIC_STATIC:
            return _DYNAMIC_STATIC
        if body is _UNKNOWN_STATIC or alternate is _UNKNOWN_STATIC:
            return _UNKNOWN_STATIC
        body_values = body if isinstance(body, tuple) else (body,)
        alternate_values = alternate if isinstance(alternate, tuple) else (alternate,)
        return (*body_values, *alternate_values)
    if isinstance(node, ast.BinOp):
        left_value = _static_rng_value(node.left, environment)
        right_value = _static_rng_value(node.right, environment)
        if left_value is _DYNAMIC_STATIC or right_value is _DYNAMIC_STATIC:
            return _DYNAMIC_STATIC
        left = _static_integer_values(left_value)
        right = _static_integer_values(right_value)
        if left is None or right is None:
            return _UNKNOWN_STATIC
        operations = {
            ast.Add: lambda x, y: x + y,
            ast.Sub: lambda x, y: x - y,
            ast.Mult: lambda x, y: x * y,
            ast.FloorDiv: lambda x, y: x // y,
            ast.Mod: lambda x, y: x % y,
        }
        operation = operations.get(type(node.op))
        if operation is None:
            return _UNKNOWN_STATIC
        try:
            return tuple(operation(x, y) for x in left for y in right)
        except (ArithmeticError, OverflowError):
            return _UNKNOWN_STATIC
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name == "range":
            arguments = [
                _single_static_integer(_static_rng_value(argument, environment))
                for argument in node.args
            ]
            if any(argument is None for argument in arguments):
                return _UNKNOWN_STATIC
            return tuple(range(*(int(argument) for argument in arguments)))
        if name == "enumerate" and node.args:
            values = _static_rng_value(node.args[0], environment)
            if not isinstance(values, tuple):
                return _UNKNOWN_STATIC
            start = 0
            if len(node.args) > 1:
                parsed_start = _single_static_integer(
                    _static_rng_value(node.args[1], environment)
                )
                if parsed_start is None:
                    return _UNKNOWN_STATIC
                start = parsed_start
            return tuple(enumerate(values, start=start))
        if name in {"list", "set", "tuple"} and len(node.args) == 1:
            values = _static_rng_value(node.args[0], environment)
            if isinstance(values, tuple):
                return values
        if name.endswith("_paper_seed") and len(node.args) == 2:
            seeds = _static_integer_values(
                _static_rng_value(node.args[0], environment)
            )
            streams = _static_integer_values(
                _static_rng_value(node.args[1], environment)
            )
            if seeds is not None and streams is not None:
                return tuple(
                    (1_000_003 * seed + stream) % (2**31 - 1)
                    for seed in seeds
                    for stream in streams
                )
        if name == "int" and len(node.args) == 1:
            value = _static_rng_value(node.args[0], environment)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
                return int(value)
        arguments = [
            _static_rng_value(argument, environment)
            for argument in (*node.args, *[keyword.value for keyword in node.keywords])
        ]
        if any(value is _DYNAMIC_STATIC for value in arguments):
            return _DYNAMIC_STATIC
    return _UNKNOWN_STATIC


def _static_integer_values(value: Any) -> set[int] | None:
    if (
        isinstance(value, bool)
        or value is _UNKNOWN_STATIC
        or value is _DYNAMIC_STATIC
        or value is _UNRESOLVED_SEED
        or value is None
    ):
        return None
    if isinstance(value, int):
        return {value}
    if isinstance(value, dict):
        parts = [_static_integer_values(nested) for nested in value.values()]
    elif isinstance(value, (tuple, list, set, frozenset)):
        parts = [_static_integer_values(nested) for nested in value]
    else:
        return None
    if any(part is None for part in parts):
        return None
    return set().union(*(part or set() for part in parts))


def _single_static_integer(value: Any) -> int | None:
    values = _static_integer_values(value)
    if values is None or len(values) != 1:
        return None
    return next(iter(values))


def _unresolved_concrete_seed(
    node: ast.AST,
    value: Any,
    environment: Mapping[str, Any],
) -> bool:
    if value is _DYNAMIC_STATIC:
        return False
    if value is _UNRESOLVED_SEED:
        return True
    if isinstance(node, ast.Name):
        bound = environment.get(node.id, _UNKNOWN_STATIC)
        return bound is not _DYNAMIC_STATIC and RNG_FIELD.search(node.id) is not None
    if isinstance(node, ast.Attribute):
        return False
    return any(
        isinstance(child, ast.Constant)
        and isinstance(child.value, int)
        and not isinstance(child.value, bool)
        for child in ast.walk(node)
    )


def _ast_has_concrete_integer(node: ast.AST, environment: Mapping[str, Any]) -> bool:
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Constant)
            and isinstance(child.value, int)
            and not isinstance(child.value, bool)
        ):
            return True
        if isinstance(child, ast.Name):
            values = _static_integer_values(
                environment.get(child.id, _UNKNOWN_STATIC)
            )
            if values:
                return True
    return False


def _call_name(function: ast.AST) -> str:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        parent = _call_name(function.value)
        return f"{parent}.{function.attr}" if parent else function.attr
    if isinstance(function, ast.Call):
        return _call_name(function.func)
    return ""


def _validated_integer_list(value: Any, label: str) -> set[int]:
    if (
        not isinstance(value, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value != sorted(set(value))
    ):
        raise RuntimeError(f"RNG audit {label} must be a sorted unique integer list")
    return set(value)


def _reservation_contract(
    values: set[int],
    *,
    mapping: Mapping[str, int] | None,
) -> dict[str, Any]:
    ordered = sorted(values)
    contract = {
        "count": len(ordered),
        "minimum": None if not ordered else ordered[0],
        "maximum": None if not ordered else ordered[-1],
        "rng_id_sha256": _integer_set_sha256(values),
        "mapping_count": None if mapping is None else len(mapping),
        "mapping_sha256": None if mapping is None else _canonical_sha256(mapping),
    }
    if mapping is not None:
        if set(mapping.values()) != values or len(mapping) != len(values):
            raise RuntimeError("external reservation mapping is not one-to-one")
        contract["mapping"] = dict(mapping)
    return contract


def _config_from_payload(payload: Mapping[str, Any]) -> NativeSignedGammaBenchmarkConfig:
    values = dict(payload)
    from scpcp.native_signed_gamma import NativeSignedGammaDGPConfig, NativeSignedGammaGateConfig

    values["dgp"] = NativeSignedGammaDGPConfig(**values["dgp"])
    values["gate"] = NativeSignedGammaGateConfig(**values["gate"])
    for name in ("gammas", "base_seeds", "devices"):
        values[name] = tuple(values[name])
    values["output_root"] = Path(values["output_root"])
    config = NativeSignedGammaBenchmarkConfig(**values)
    config.validate()
    return config


def _source_contract(path: Path, source_root: Path = ROOT) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        recorded_path = resolved.relative_to(source_root.resolve()).as_posix()
    except ValueError:
        recorded_path = str(resolved)
    return {
        "path": recorded_path,
        "sha256": _file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_recorded_path(value: str, source_root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else source_root / path


def _canonical_invocation(argv: Sequence[str]) -> tuple[str, ...]:
    return tuple(value for value in argv if value not in {"--resume", "--validate-only"})


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _integer_set_sha256(values: Iterable[int]) -> str:
    return _canonical_sha256(sorted(set(values)))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, _canonical_json_bytes(value) + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read preflight JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"preflight JSON artifact must be a mapping: {path}")
    return value


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
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


def _is_relative_to(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


if __name__ == "__main__":
    main()
