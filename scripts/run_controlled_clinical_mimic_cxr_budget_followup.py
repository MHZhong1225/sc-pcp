"""Run the coverage-blind precoverage phase of the MIMIC-CXR v2 follow-up.

Read-only audit::

    python scripts/run_controlled_clinical_mimic_cxr_budget_followup.py audit

Formal launch requires the exact hash printed by that audit::

    python scripts/run_controlled_clinical_mimic_cxr_budget_followup.py run \
      --audit-go-sha256 <hash>
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields, replace
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_controlled_clinical_extension as extension  # noqa: E402
import scripts.run_controlled_clinical_fidelity_v4 as fidelity_v4  # noqa: E402
import scripts.run_controlled_clinical_mimic_cxr_environment_support as v1  # noqa: E402
import scripts.run_controlled_clinical_mimic_cxr_environment_support_science as v1science  # noqa: E402
from scpcp.artifacts import experiment_tree_sha256  # noqa: E402
from scpcp.controlled_clinical_mimic_cxr_budget_followup import (  # noqa: E402
    BOOTSTRAP_SEED,
    CALIBRATION_TRAJECTORIES,
    DATASET,
    EXPECTED_RNG_CONTRACT,
    GRID_TRAJECTORIES,
    PROTOCOL,
    ROLE_SPLIT,
    SEEDS,
    VALIDATION_CLAIMS,
    load_config,
    summarize_precoverage,
    validate_runtime_protocol,
    verify_canonical_scpcp,
    verify_prior_v1_bindings,
    verify_v1_source_diff_allowlist,
)
from scpcp.controlled_clinical_extension import DatasetPreset  # noqa: E402


CONFIG_PATH = ROOT / "configs/controlled_clinical_mimic_cxr_budget_followup_v2.yaml"
BASE_CONFIG_PATH = ROOT / "configs/controlled_clinical_extension.yaml"
OUTPUT_ROOT = (
    ROOT / "results/work/controlled_clinical_mimic_cxr_budget_followup_v2_precoverage"
).resolve()
SCIENCE_ROOT = (
    ROOT / "results/work/controlled_clinical_mimic_cxr_budget_followup_v2_science"
).resolve()
PRECOVERAGE_STREAM_SUFFIXES = (
    "/task",
    "/outcome_model",
    "/behavior_model",
    "/cxr_encoder",
    "/k0_base_uniform",
)
SHARED_CONTEXT_REPLAY_SUFFIXES = (
    "/task",
    "/outcome_model",
    "/behavior_model",
    "/cxr_encoder",
)
FORBIDDEN_PATH_TOKENS = (
    "science",
    "coverage",
    "mean_coverage",
    "width",
    "method_selection",
)
OWN_RNG_DECLARATION_PATHS = {
    CONFIG_PATH.resolve(),
    Path(__file__).resolve(),
    (ROOT / "src/scpcp/controlled_clinical_mimic_cxr_budget_followup.py").resolve(),
    (
        ROOT / "scripts/run_controlled_clinical_mimic_cxr_budget_followup_science.py"
    ).resolve(),
    (
        ROOT / "tests/per_step/test_controlled_clinical_mimic_cxr_budget_followup.py"
    ).resolve(),
    (
        ROOT
        / "tests/per_step/test_controlled_clinical_mimic_cxr_budget_followup_science.py"
    ).resolve(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "run"))
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--audit-go-sha256")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    v1._validate_devices(devices)
    audit = build_read_only_audit(devices)
    audit_hash = _json_sha256(audit)
    if args.phase == "audit":
        if args.audit_go_sha256 is not None or args.resume:
            parser.error("audit does not accept --audit-go-sha256 or --resume")
        assert_fresh_roots()
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "status": "READ_ONLY_PRECOVERAGE_AUDIT_GO",
                    "audit_contract_sha256": audit_hash,
                    "formal_roots_absent": True,
                    "coverage_generated": False,
                    "audit": audit,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.audit_go_sha256 != audit_hash:
        raise RuntimeError("run requires the exact read-only audit hash")
    if not args.resume:
        assert_fresh_roots()
    run_precoverage(
        OUTPUT_ROOT,
        devices=devices,
        audit=audit,
        audit_go_sha256=audit_hash,
        resume=args.resume,
    )
    print(OUTPUT_ROOT)


def runtime_protocol() -> Any:
    """Build the frozen budget adapter without modifying the shared config."""

    protocol = extension.load_extension_config(BASE_CONFIG_PATH)
    preset = replace(
        protocol.datasets[DATASET],
        seeds=SEEDS,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    adapted = replace(
        protocol,
        split_fractions=ROLE_SPLIT,
        calibration_trajectories=CALIBRATION_TRAJECTORIES,
        grid_trajectories=GRID_TRAJECTORIES,
        datasets={DATASET: preset},
    )
    validate_runtime_protocol(adapted)
    validate_exact_only_change(adapted)
    return adapted


def validate_exact_only_change(adapted: Any) -> None:
    """Prove that v2 changes only fresh identities and the two common budgets."""

    v1_config = v1.load_config(v1.CONFIG_PATH)
    v1_protocol = v1._protocol_for(v1.CONFIRMATION_SEEDS, v1_config)
    allowed_protocol_fields = {
        "calibration_trajectories",
        "grid_trajectories",
        "datasets",
    }
    for field in fields(v1_protocol):
        if field.name in allowed_protocol_fields:
            continue
        if getattr(adapted, field.name) != getattr(v1_protocol, field.name):
            raise RuntimeError(f"v2 changed frozen protocol field: {field.name}")
    old_preset = v1_protocol.datasets[DATASET]
    new_preset = adapted.datasets[DATASET]
    for field in fields(old_preset):
        if field.name in {"seeds", "bootstrap_seed"}:
            continue
        if getattr(new_preset, field.name) != getattr(old_preset, field.name):
            raise RuntimeError(f"v2 changed frozen dataset field: {field.name}")
    if (
        v1_protocol.calibration_trajectories != 3_000
        or v1_protocol.grid_trajectories != 1_000
        or adapted.calibration_trajectories != CALIBRATION_TRAJECTORIES
        or adapted.grid_trajectories != GRID_TRAJECTORIES
        or tuple(new_preset.seeds) != SEEDS
        or new_preset.bootstrap_seed != BOOTSTRAP_SEED
    ):
        raise RuntimeError("v2 exact-only change contract differs")


def build_read_only_audit(devices: tuple[str, ...]) -> dict[str, Any]:
    opening_source_hash = experiment_tree_sha256()
    config = load_config(CONFIG_PATH)
    protocol = runtime_protocol()
    prior = verify_prior_v1_bindings(ROOT, config)
    canonical = verify_canonical_scpcp(ROOT)
    source_diff = verify_v1_source_diff_allowlist(ROOT)
    rng = audit_rng(protocol)
    closing_source_diff = verify_v1_source_diff_allowlist(ROOT)
    closing_source_hash = experiment_tree_sha256()
    if opening_source_hash != closing_source_hash or source_diff != closing_source_diff:
        raise RuntimeError("source changed during the read-only audit")
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "devices": list(devices),
        "output_root": str(OUTPUT_ROOT),
        "science_root": str(SCIENCE_ROOT),
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": _file_sha256(CONFIG_PATH),
        "source_tree_sha256": closing_source_hash,
        "prior_v1": prior,
        "canonical_scpcp": canonical,
        "v1_source_diff": source_diff,
        "rng_audit": rng,
        "role_split": list(ROLE_SPLIT),
        "bridge_candidate": v1._b02().to_dict(),
        "calibration_trajectories": protocol.calibration_trajectories,
        "grid_trajectories": protocol.grid_trajectories,
        "reference_trajectories": protocol.reference_trajectories,
        "coverage_generation_permitted": False,
        "seed_deletion_permitted": False,
        "validation_claims": VALIDATION_CLAIMS,
    }


def active_source_snapshot() -> tuple[str, dict[str, Any]]:
    """Build a deterministic source snapshot labeled with the v2 protocol."""

    source_hash = experiment_tree_sha256()
    snapshot = fidelity_v4._build_source_snapshot()
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
        raise RuntimeError("source changed while building the v2 source snapshot")
    return source_hash, snapshot


def verify_source_snapshot(root: Path, contract: Mapping[str, Any]) -> None:
    """Verify bytes plus the v2 identity embedded in the source manifest."""

    fidelity_v4._verify_source_snapshot(root, contract)
    manifest = read_json(safe_child(root, Path(str(contract["manifest_path"]))))
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("format") != "deterministic_uncompressed_pax_tar"
        or manifest.get("file_count") != contract.get("file_count")
        or not isinstance(manifest.get("files"), list)
        or len(manifest["files"]) != contract.get("file_count")
    ):
        raise RuntimeError("source snapshot manifest is not the v2 contract")


def audit_rng(protocol: Any) -> dict[str, Any]:
    full = extension._new_rng_stream_mapping(protocol, (DATASET,))
    extension._assert_unique_rng_streams(full)
    precoverage = {
        name: value
        for name, value in full.items()
        if name.endswith(PRECOVERAGE_STREAM_SUFFIXES)
    }
    shared_context = {
        name: value
        for name, value in precoverage.items()
        if name.endswith(SHARED_CONTEXT_REPLAY_SUFFIXES)
    }
    postunlock_new = {
        name: value for name, value in full.items() if name not in precoverage
    }
    pre_ids = set(precoverage.values())
    post_new_ids = set(postunlock_new.values())
    if pre_ids & post_new_ids:
        raise RuntimeError("precoverage and post-unlock new RNG streams overlap")

    observed_contract = {
        "base_seed_set_sha256": _integer_set_sha256(SEEDS),
        "precoverage_stream_count": len(precoverage),
        "precoverage_mapping_sha256": _json_sha256(precoverage),
        "full_stream_count": len(full),
        "full_mapping_sha256": _json_sha256(full),
        "full_id_set_sha256": _integer_set_sha256(full.values()),
        "internal_collision_count": len(full) - len(set(full.values())),
        "precoverage_vs_postunlock_new_stream_collision_count": len(
            pre_ids & post_new_ids
        ),
    }
    if observed_contract != EXPECTED_RNG_CONTRACT:
        raise RuntimeError("v2 RNG mapping differs from the frozen contract")

    excluded_roots = {OUTPUT_ROOT, SCIENCE_ROOT}
    artifact_ids = v1._artifact_rng_ids(ROOT / "results", excluded_roots=excluded_roots)
    source_ids = extension._source_declared_seeds(
        ROOT,
        excluded_paths={path for path in OWN_RNG_DECLARATION_PATHS if path.exists()},
    )
    prior_ids = artifact_ids | source_ids
    collisions = {name: value for name, value in full.items() if value in prior_ids}
    if collisions:
        raise RuntimeError(f"fresh v2 RNG stream collides with prior use: {collisions}")
    return {
        "status": "passed_read_only_before_launch",
        "full_mapping": full,
        "precoverage_mapping": precoverage,
        "postunlock_new_mapping": postunlock_new,
        "shared_context_replay_streams": shared_context,
        "shared_context_replay_semantics": (
            "anchored deterministic reconstruction of the same task/model/encoder "
            "context; these are replayed identities, not fresh post-unlock draws"
        ),
        **observed_contract,
        "historical_artifact_rng_id_count": len(artifact_ids),
        "historical_artifact_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "historical_source_rng_id_count": len(source_ids),
        "historical_source_rng_id_sha256": _integer_set_sha256(source_ids),
        "historical_union_rng_id_count": len(prior_ids),
        "historical_union_rng_id_sha256": _integer_set_sha256(prior_ids),
        "historical_collision_count": 0,
        "historical_collisions": {},
        "excluded_protocol_roots": sorted(str(path) for path in excluded_roots),
    }


def run_precoverage(
    output_root: Path,
    *,
    devices: tuple[str, ...],
    audit: Mapping[str, Any],
    audit_go_sha256: str,
    resume: bool,
) -> None:
    if output_root.resolve() != OUTPUT_ROOT:
        raise RuntimeError(f"precoverage root is frozen to {OUTPUT_ROOT}")
    if _json_sha256(audit) != audit_go_sha256:
        raise RuntimeError("precoverage audit hash differs")
    if resume and (output_root / "COMPLETE").is_file():
        verify_complete_root(output_root)
        stored = read_json(output_root / "metadata.json")
        if (
            stored.get("read_only_audit") != audit
            or stored.get("source_tree_sha256") != audit.get("source_tree_sha256")
            or experiment_tree_sha256() != audit.get("source_tree_sha256")
        ):
            raise RuntimeError(
                "completed precoverage no longer matches the active audit"
            )
        return
    if SCIENCE_ROOT.exists() or SCIENCE_ROOT.is_symlink():
        raise RuntimeError(
            "science root exists; precoverage cannot be continued or reopened"
        )
    source_hash, source_snapshot = active_source_snapshot()
    if source_hash != audit["source_tree_sha256"]:
        raise RuntimeError("source changed after the read-only audit")
    metadata = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "phase": "precoverage",
        "output_root": str(output_root),
        "devices": list(devices),
        "source_tree_sha256": source_hash,
        "source_snapshot": source_snapshot["contract"],
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": _file_sha256(CONFIG_PATH),
        "read_only_audit_go_sha256": audit_go_sha256,
        "read_only_audit": dict(audit),
        "role_split": list(ROLE_SPLIT),
        "bridge_candidate": v1._b02().to_dict(),
        "coverage_generation_permitted": False,
        "canonical_scpcp_mutation_permitted": False,
        "seed_deletion_permitted": False,
        "validation_claims": VALIDATION_CLAIMS,
    }
    prepare_root(output_root, metadata, source_snapshot, resume=resume)
    if (output_root / "COMPLETE").exists():
        verify_complete_root(output_root)
        return

    protocol = runtime_protocol()
    preset = protocol.datasets[DATASET]
    support_rows = run_seed_phase(
        output_root / "support",
        phase="precoverage_support",
        preset=preset,
        devices=devices,
        worker=v1._support_worker,
        worker_arguments=(protocol,),
        source_hash=source_hash,
    )
    k0_rows = run_seed_phase(
        output_root / "k0_fidelity",
        phase="precoverage_k0",
        preset=preset,
        devices=devices,
        worker=v1._k0_worker,
        worker_arguments=(protocol,),
        source_hash=source_hash,
    )
    gate = summarize_precoverage(support_rows, k0_rows)
    final = precoverage_final_status(gate)
    write_json(output_root / "gate.json", gate)
    write_json(output_root / "FINAL_STATUS.json", final)
    finalize_root(output_root, metadata, source_hash)


def precoverage_final_status(gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "phase": "precoverage",
        "status": gate["status"],
        "precoverage_admissible": gate["precoverage_admissible"],
        "eligible_seeds": gate["joint_pass_seeds"],
        "coverage_generated": False,
        "science_may_start": gate["precoverage_admissible"],
        "seed_deletions": 0,
        "validation_claims": VALIDATION_CLAIMS,
    }


def run_seed_phase(
    phase_root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    devices: tuple[str, ...],
    worker: Callable[..., dict[str, Any]],
    worker_arguments: tuple[object, ...],
    source_hash: str,
) -> list[dict[str, Any]]:
    if phase_root.is_symlink() or (phase_root.exists() and not phase_root.is_dir()):
        raise RuntimeError(f"invalid {phase} root")
    phase_root.mkdir(parents=True, exist_ok=True)
    seed_to_device = v1._seed_device_mapping(preset.seeds, devices)
    expected = {f"seed_{seed:06d}.json" for seed in preset.seeds} | {"COMPLETE"}
    children = list(phase_root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise RuntimeError(f"unsafe {phase} artifact")
    if unexpected := {path.name for path in children} - expected:
        raise RuntimeError(f"unexpected {phase} artifacts: {sorted(unexpected)}")

    complete_path = phase_root / "COMPLETE"
    if complete_path.exists() and complete_path.read_text() != "complete\n":
        raise RuntimeError(f"invalid {phase} COMPLETE marker")
    completed: dict[int, dict[str, Any]] = {}
    for seed in preset.seeds:
        path = phase_root / f"seed_{seed:06d}.json"
        if not path.exists():
            continue
        payload = read_json(path)
        validate_seed_payload(
            payload,
            phase=phase,
            seed=seed,
            device=seed_to_device[seed],
            source_hash=source_hash,
            preset=preset,
        )
        completed[seed] = payload["result"]
    pending = tuple(seed for seed in preset.seeds if seed not in completed)
    if pending and complete_path.exists():
        raise RuntimeError(f"{phase} is complete with missing seeds")
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
                    _phase_group,
                    seeds,
                    device,
                    preset,
                    worker,
                    worker_arguments,
                ): device
                for device, seeds in groups.items()
                if seeds
            }
            for future in as_completed(futures):
                for seed, device, result in future.result():
                    payload = {
                        "protocol": PROTOCOL,
                        "dataset": DATASET,
                        "phase": phase,
                        "seed": seed,
                        "device": device,
                        "source_tree_sha256": source_hash,
                        "result": result,
                    }
                    validate_seed_payload(
                        payload,
                        phase=phase,
                        seed=seed,
                        device=device,
                        source_hash=source_hash,
                        preset=preset,
                    )
                    write_json(phase_root / f"seed_{seed:06d}.json", payload)
                    completed[seed] = result
    if set(completed) != set(preset.seeds):
        raise RuntimeError(f"{phase} did not complete its exact seed bank")
    if not complete_path.exists():
        write_text(complete_path, "complete\n")
    return [completed[seed] for seed in preset.seeds]


def _phase_group(
    seeds: tuple[int, ...],
    device: str,
    preset: DatasetPreset,
    worker: Callable[..., dict[str, Any]],
    worker_arguments: tuple[object, ...],
) -> list[tuple[int, str, dict[str, Any]]]:
    torch.cuda.set_device(torch.device(device))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    rows = []
    for seed in seeds:
        rows.append((seed, device, worker(seed, preset, device, *worker_arguments)))
        torch.cuda.empty_cache()
    return rows


def validate_seed_payload(
    payload: Mapping[str, Any],
    *,
    phase: str,
    seed: int,
    device: str,
    source_hash: str,
    preset: DatasetPreset,
) -> None:
    result = payload.get("result")
    if (
        set(payload)
        != {
            "protocol",
            "dataset",
            "phase",
            "seed",
            "device",
            "source_tree_sha256",
            "result",
        }
        or payload["protocol"] != PROTOCOL
        or payload["dataset"] != DATASET
        or payload["phase"] != phase
        or payload["seed"] != seed
        or payload["device"] != device
        or payload["source_tree_sha256"] != source_hash
        or not isinstance(result, Mapping)
        or result.get("seed") != seed
        or type(result.get("passed")) is not bool
        or result.get("coverage_generated") is not False
        or result.get("role_split") != list(ROLE_SPLIT)
    ):
        raise RuntimeError(f"invalid {phase} seed artifact: {seed}")
    split = result.get("split_audit")
    if (
        not isinstance(split, Mapping)
        or split.get("patient_sets_pairwise_disjoint") is not True
        or split.get("split_fractions") != list(ROLE_SPLIT)
    ):
        raise RuntimeError(f"invalid {phase} split identity: {seed}")
    if phase == "precoverage_support":
        required_support_fields = {
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
            "role_split",
            "coverage_generated",
        }
        if (
            set(result) != required_support_fields
            or result.get("dataset") != DATASET
            or result.get("phase") != "support"
        ):
            raise RuntimeError(f"support artifact schema differs: {seed}")
        v1science._validate_support_result(result, preset=preset, seed=seed)
        return
    if phase != "precoverage_k0":
        raise RuntimeError(f"unknown precoverage phase: {phase}")
    v1science._validate_k0_result(result, theta=v1._b02(), seed=seed)


def prepare_root(
    root: Path,
    metadata: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if resume:
        if root.is_symlink() or not root.is_dir():
            raise FileNotFoundError(
                "resume requires the existing regular precoverage root"
            )
        assert_safe_tree(root)
        assert_no_forbidden_paths(root)
        if read_json(root / "metadata.json") != metadata:
            raise RuntimeError("resume metadata differs")
        assert_partial_artifact_subset(root, metadata)
        verify_source_snapshot(root, metadata["source_snapshot"])
        return
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"fresh precoverage root exists: {root}")
    root.mkdir(parents=True)
    fidelity_v4._atomic_write(
        root / source_snapshot["contract"]["archive_path"],
        source_snapshot["archive_bytes"],
    )
    fidelity_v4._atomic_write(
        root / source_snapshot["contract"]["manifest_path"],
        source_snapshot["manifest_bytes"],
    )
    write_json(root / "metadata.json", metadata)
    verify_source_snapshot(root, metadata["source_snapshot"])


def finalize_root(root: Path, metadata: Mapping[str, Any], source_hash: str) -> None:
    if experiment_tree_sha256() != source_hash:
        raise RuntimeError("source changed during precoverage")
    if read_json(root / "metadata.json") != metadata:
        raise RuntimeError("precoverage metadata changed")
    assert_no_forbidden_paths(root)
    write_manifest(root)
    final = read_json(root / "FINAL_STATUS.json")
    marker = (
        f"complete source_tree_sha256={source_hash} "
        f"config_sha256={metadata['config_sha256']} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={_file_sha256(root / 'manifest.json')}\n"
    )
    write_text(root / "COMPLETE", marker)
    verify_complete_root(root)


def verify_complete_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("completed precoverage root must be a regular directory")
    assert_safe_tree(root)
    assert_no_forbidden_paths(root)
    manifest = read_json(root / "manifest.json")
    entries = manifest.get("artifacts")
    if manifest.get("protocol") != PROTOCOL or not isinstance(entries, list):
        raise RuntimeError("precoverage manifest header differs")
    expected: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RuntimeError("precoverage manifest entry is malformed")
        relative = Path(str(entry.get("path", "")))
        path = safe_child(root, relative)
        if relative in expected or relative in {
            Path("manifest.json"),
            Path("COMPLETE"),
        }:
            raise RuntimeError("precoverage manifest path is duplicated or reserved")
        expected.add(relative)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != entry.get("bytes")
            or _file_sha256(path) != entry.get("sha256")
        ):
            raise RuntimeError(f"precoverage manifest mismatch: {relative}")
    observed = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root) not in {Path("manifest.json"), Path("COMPLETE")}
    }
    if observed != expected or manifest.get("artifact_count") != len(expected):
        raise RuntimeError("precoverage manifest file set differs")
    metadata = read_json(root / "metadata.json")
    final = read_json(root / "FINAL_STATUS.json")
    if {path.relative_to(root) for path in root.rglob("*") if path.is_file()} != (
        precoverage_expected_paths(metadata)
    ):
        raise RuntimeError("completed precoverage artifact set differs")
    verify_source_snapshot(root, metadata["source_snapshot"])
    marker = (
        f"complete source_tree_sha256={metadata['source_tree_sha256']} "
        f"config_sha256={metadata['config_sha256']} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={_file_sha256(root / 'manifest.json')}\n"
    )
    if not (root / "COMPLETE").is_file() or (root / "COMPLETE").read_text() != marker:
        raise RuntimeError("precoverage COMPLETE marker differs")
    support = read_seed_phase(root / "support", "precoverage_support", metadata)
    k0 = read_seed_phase(root / "k0_fidelity", "precoverage_k0", metadata)
    gate = summarize_precoverage(support, k0)
    if read_json(root / "gate.json") != gate or final != precoverage_final_status(gate):
        raise RuntimeError("completed precoverage gate does not recompute")


def verify_precoverage_go(root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    verify_complete_root(root)
    metadata = read_json(root / "metadata.json")
    final = read_json(root / "FINAL_STATUS.json")
    gate = read_json(root / "gate.json")
    support = read_seed_phase(root / "support", "precoverage_support", metadata)
    k0 = read_seed_phase(root / "k0_fidelity", "precoverage_k0", metadata)
    recomputed = summarize_precoverage(support, k0)
    if (
        metadata.get("protocol") != PROTOCOL
        or metadata.get("dataset") != DATASET
        or metadata.get("phase") != "precoverage"
        or metadata.get("output_root") != str(root.resolve())
        or metadata.get("coverage_generation_permitted") is not False
        or metadata.get("canonical_scpcp_mutation_permitted") is not False
        or metadata.get("seed_deletion_permitted") is not False
        or metadata.get("validation_claims") != VALIDATION_CLAIMS
        or gate != recomputed
        or gate.get("status") != "PRECOVERAGE_GO"
        or gate.get("precoverage_admissible") is not True
        or final.get("status") != "PRECOVERAGE_GO"
        or final.get("precoverage_admissible") is not True
        or final.get("eligible_seeds") != gate.get("joint_pass_seeds")
        or final.get("science_may_start") is not True
        or final.get("coverage_generated") is not False
    ):
        raise RuntimeError("science is locked by precoverage NO-GO or tampering")
    binding = {
        "root": str(root.resolve()),
        "complete_sha256": _file_sha256(root / "COMPLETE"),
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "metadata_sha256": _file_sha256(root / "metadata.json"),
        "final_status_sha256": _file_sha256(root / "FINAL_STATUS.json"),
        "gate_sha256": _file_sha256(root / "gate.json"),
    }
    return {
        "metadata": metadata,
        "gate": gate,
        "support": support,
        "k0": k0,
        "binding": {**binding, "combined_sha256": _json_sha256(binding)},
    }


def read_seed_phase(
    root: Path,
    phase: str,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = {f"seed_{seed:06d}.json" for seed in SEEDS} | {"COMPLETE"}
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()} != expected
    ):
        raise RuntimeError(f"{phase} seed bank differs")
    if (root / "COMPLETE").read_text() != "complete\n":
        raise RuntimeError(f"{phase} COMPLETE marker differs")
    devices = tuple(metadata["devices"])
    device_map = v1._seed_device_mapping(SEEDS, devices)
    preset = runtime_protocol().datasets[DATASET]
    rows = []
    for seed in SEEDS:
        payload = read_json(root / f"seed_{seed:06d}.json")
        validate_seed_payload(
            payload,
            phase=phase,
            seed=seed,
            device=device_map[seed],
            source_hash=str(metadata["source_tree_sha256"]),
            preset=preset,
        )
        rows.append(payload["result"])
    return rows


def write_manifest(root: Path) -> None:
    assert_safe_tree(root)
    entries = []
    for path in sorted(root.rglob("*")):
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
    write_json(
        root / "manifest.json",
        {"protocol": PROTOCOL, "artifact_count": len(entries), "artifacts": entries},
    )


def precoverage_allowed_paths(metadata: Mapping[str, Any]) -> set[Path]:
    source = metadata["source_snapshot"]
    paths = {
        Path("metadata.json"),
        Path(str(source["archive_path"])),
        Path(str(source["manifest_path"])),
        Path("gate.json"),
        Path("FINAL_STATUS.json"),
        Path("manifest.json"),
        Path("COMPLETE"),
        Path("support/COMPLETE"),
        Path("k0_fidelity/COMPLETE"),
    }
    for seed in SEEDS:
        paths.add(Path(f"support/seed_{seed:06d}.json"))
        paths.add(Path(f"k0_fidelity/seed_{seed:06d}.json"))
    return paths


def precoverage_expected_paths(metadata: Mapping[str, Any]) -> set[Path]:
    return precoverage_allowed_paths(metadata)


def assert_partial_artifact_subset(root: Path, metadata: Mapping[str, Any]) -> None:
    observed = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
    if unexpected := observed - precoverage_allowed_paths(metadata):
        raise RuntimeError(f"unexpected precoverage artifacts: {sorted(unexpected)}")


def assert_fresh_roots() -> None:
    existing = [
        str(path)
        for path in (OUTPUT_ROOT, SCIENCE_ROOT)
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(f"formal v2 roots must be absent: {existing}")


def assert_safe_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden: {path}")
        if path.is_file() and ".tmp-" in path.name:
            raise RuntimeError(f"temporary artifact remains: {path}")


def assert_no_forbidden_paths(root: Path) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(
            token in part.lower()
            for part in relative.parts
            for token in FORBIDDEN_PATH_TOKENS
        ):
            raise RuntimeError(f"forbidden precoverage artifact path: {relative}")


def safe_child(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError("artifact path escapes its root")
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents:
        raise RuntimeError("artifact path escapes its root")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError(f"JSON artifact may not be a symlink: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    fidelity_v4._write_json(path, value)


def write_text(path: Path, value: str) -> None:
    fidelity_v4._write_text(path, value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer_set_sha256(values: Iterable[int]) -> str:
    return _json_sha256(sorted(set(int(value) for value in values)))


if __name__ == "__main__":
    main()
