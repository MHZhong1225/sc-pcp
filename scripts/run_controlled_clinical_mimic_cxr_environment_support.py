"""Run the coverage-blind MIMIC-CXR environment-support protocol.

This is a new post-failure study, not a continuation of the terminal v6 bridge
repair.  It keeps B02 and all K0 thresholds fixed, and changes only the raw
patient role allocation to predictor/fidelity/environment = 20/20/60.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_controlled_clinical_extension as v2  # noqa: E402
import scripts.run_controlled_clinical_fidelity_v4 as v4  # noqa: E402
import scripts.run_controlled_clinical_fidelity_v5_mimic_cxr as v5  # noqa: E402
from scpcp.artifacts import experiment_tree_sha256  # noqa: E402
from scpcp.controlled_clinical_mimic_cxr_environment_support import (  # noqa: E402
    BRIDGE_CANDIDATE_ID,
    CONFIRMATION_SEEDS,
    DATASET,
    DEVELOPMENT_BLOCKS,
    K0_THRESHOLDS,
    PROTOCOL,
    ROLE_SPLIT,
    EnvironmentSupportConfig,
    load_config,
    normalized_k0_ratio,
    summarize_confirmation,
    summarize_development,
    verify_prior_bindings,
)
from scpcp.controlled_clinical_extension import DatasetPreset  # noqa: E402


CONFIG_PATH = ROOT / "configs/controlled_clinical_mimic_cxr_environment_support_v1.yaml"
V2_CONFIG_PATH = ROOT / "configs/controlled_clinical_extension.yaml"
DEVELOPMENT_ROOT = (
    ROOT / "results/work/controlled_clinical_mimic_cxr_environment_support_v1_development"
).resolve()
CONFIRMATION_ROOT = (
    ROOT / "results/work/controlled_clinical_mimic_cxr_environment_support_v1_confirmation"
).resolve()
SCIENCE_ROOT = (
    ROOT / "results/work/controlled_clinical_mimic_cxr_environment_support_v1_science"
).resolve()
PHASES = ("audit", "development", "confirmation")
FORBIDDEN_PATH_TOKENS = (
    "science",
    "coverage",
    "mean_coverage",
    "width",
    "method_selection",
)
PRECOVERAGE_RNG_STREAM_SUFFIXES = (
    "/task",
    "/outcome_model",
    "/behavior_model",
    "/cxr_encoder",
    "/k0_base_uniform",
)
_OWN_RNG_DECLARATION_PATHS = {
    CONFIG_PATH.resolve(),
    Path(__file__).resolve(),
    (ROOT / "src/scpcp/controlled_clinical_mimic_cxr_environment_support.py").resolve(),
    (ROOT / "scripts/run_controlled_clinical_mimic_cxr_environment_support_science.py").resolve(),
    (ROOT / "tests/per_step/test_controlled_clinical_mimic_cxr_environment_support.py").resolve(),
    (ROOT / "tests/per_step/test_controlled_clinical_mimic_cxr_environment_support_runner.py").resolve(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    prior_binding = verify_prior_bindings(ROOT, config)
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    _validate_devices(devices)
    audits = _rng_audits(config)
    if args.phase == "audit":
        _assert_fresh_roots()
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "status": "PREFLIGHT_GO",
                    "prior_negative_evidence": prior_binding,
                    "development_rng_audit": audits["development"],
                    "confirmation_rng_audit": audits["confirmation"],
                    "formal_roots_absent": True,
                    "coverage_generation_permitted": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.output_root is None:
        parser.error("formal phases require --output-root")
    output_root = args.output_root.resolve()
    if args.phase == "development":
        if output_root != DEVELOPMENT_ROOT:
            parser.error(f"development output root is frozen to {DEVELOPMENT_ROOT}")
        if args.development_root is not None:
            parser.error("development does not accept --development-root")
        if not args.resume:
            _assert_fresh_roots()
        run_development(
            output_root,
            config=config,
            devices=devices,
            prior_binding=prior_binding,
            rng_audit=audits["development"],
            resume=args.resume,
        )
    else:
        if output_root != CONFIRMATION_ROOT:
            parser.error(f"confirmation output root is frozen to {CONFIRMATION_ROOT}")
        if args.development_root is None:
            parser.error("confirmation requires --development-root")
        development_root = args.development_root.resolve()
        if development_root != DEVELOPMENT_ROOT:
            parser.error(f"development root is frozen to {DEVELOPMENT_ROOT}")
        if not args.resume and (SCIENCE_ROOT.exists() or SCIENCE_ROOT.is_symlink()):
            raise FileExistsError(
                f"science root must be absent before confirmation: {SCIENCE_ROOT}"
            )
        run_confirmation(
            output_root,
            development_root=development_root,
            config=config,
            devices=devices,
            prior_binding=prior_binding,
            rng_audit=audits["confirmation"],
            resume=args.resume,
        )
    print(output_root)


def run_development(
    output_root: Path,
    *,
    config: EnvironmentSupportConfig,
    devices: tuple[str, ...],
    prior_binding: Mapping[str, Any],
    rng_audit: Mapping[str, Any],
    resume: bool,
) -> None:
    _validate_rng_audit(rng_audit, phase="development")
    source_hash, source_snapshot = v4._active_source_contract()
    metadata = _metadata(
        phase="development",
        output_root=output_root,
        devices=devices,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        prior_binding=prior_binding,
        rng_audit=rng_audit,
    )
    _prepare_root(output_root, metadata, source_snapshot, resume=resume)
    if _complete_and_valid(output_root, metadata):
        return
    protocol = _protocol_for(tuple(seed for block in DEVELOPMENT_BLOCKS.values() for seed in block), config)
    support_by_block: dict[str, list[dict[str, Any]]] = {}
    k0_by_block: dict[str, list[dict[str, Any]]] = {}
    for block, seeds in DEVELOPMENT_BLOCKS.items():
        preset = replace(protocol.datasets[DATASET], seeds=seeds)
        support_by_block[block] = _run_seed_phase(
            output_root / "support" / block,
            phase=f"development_support_{block}",
            preset=preset,
            devices=devices,
            worker=_support_worker,
            worker_arguments=(protocol,),
            source_hash=source_hash,
        )
        k0_by_block[block] = _run_seed_phase(
            output_root / "k0_fidelity" / block,
            phase=f"development_k0_{block}",
            preset=preset,
            devices=devices,
            worker=_k0_worker,
            worker_arguments=(protocol,),
            source_hash=source_hash,
        )
    gate = summarize_development(support_by_block, k0_by_block)
    final = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "phase": "development",
        "status": gate["status"],
        "development_admissible": gate["development_admissible"],
        "role_split": list(ROLE_SPLIT),
        "bridge_candidate_id": BRIDGE_CANDIDATE_ID,
        "coverage_generated": False,
    }
    frozen = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "FROZEN_FOR_CONFIRMATION" if gate["development_admissible"] else "NOT_FROZEN_DEVELOPMENT_NO_GO",
        "role_split": list(ROLE_SPLIT),
        "bridge_candidate": _b02().to_dict(),
        "development_source_tree_sha256": source_hash,
        "development_gate_sha256": _json_sha256(gate),
        "config_sha256": metadata["config_sha256"],
        "coverage_generated": False,
    }
    _write_json(output_root / "development_gate.json", gate)
    _write_json(output_root / "FINAL_STATUS.json", final)
    _write_json(output_root / "frozen_settings.json", frozen)
    _finalize(output_root, metadata, source_hash)


def run_confirmation(
    output_root: Path,
    *,
    development_root: Path,
    config: EnvironmentSupportConfig,
    devices: tuple[str, ...],
    prior_binding: Mapping[str, Any],
    rng_audit: Mapping[str, Any],
    resume: bool,
) -> None:
    _validate_rng_audit(rng_audit, phase="confirmation")
    development_binding, frozen = _verify_development(development_root)
    source_hash, source_snapshot = v4._active_source_contract()
    if source_hash != frozen["development_source_tree_sha256"]:
        raise RuntimeError("source/config changed after the CXR setting was frozen")
    metadata = _metadata(
        phase="confirmation",
        output_root=output_root,
        devices=devices,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        prior_binding=prior_binding,
        rng_audit=rng_audit,
        development_binding=development_binding,
        frozen_settings=frozen,
    )
    _prepare_root(output_root, metadata, source_snapshot, resume=resume)
    if _complete_and_valid(output_root, metadata):
        return
    protocol = _protocol_for(CONFIRMATION_SEEDS, config)
    preset = protocol.datasets[DATASET]
    support_rows = _run_seed_phase(
        output_root / "support",
        phase="confirmation_support",
        preset=preset,
        devices=devices,
        worker=_support_worker,
        worker_arguments=(protocol,),
        source_hash=source_hash,
    )
    if sum(bool(row["passed"]) for row in support_rows) < 19:
        k0_rows = [
            {
                "seed": seed,
                "passed": False,
                "metrics": {**{name: float("inf") for name in K0_THRESHOLDS}, "structural_invariants": False},
                "not_run_reason": "support_gate_failed",
            }
            for seed in CONFIRMATION_SEEDS
        ]
    else:
        k0_rows = _run_seed_phase(
            output_root / "k0_fidelity",
            phase="confirmation_k0",
            preset=preset,
            devices=devices,
            worker=_k0_worker,
            worker_arguments=(protocol,),
            source_hash=source_hash,
        )
    gate = summarize_confirmation(support_rows, k0_rows)
    final = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "phase": "confirmation",
        "status": gate["status"],
        "confirmation_admissible": gate["confirmation_admissible"],
        "eligible_seeds": gate["joint_pass_seeds"],
        "coverage_generated": False,
        "science_may_start": gate["confirmation_admissible"],
    }
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "FINAL_STATUS.json", final)
    _finalize(output_root, metadata, source_hash)


def _protocol_for(
    seeds: Sequence[int], config: EnvironmentSupportConfig
) -> Any:
    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    preset = replace(
        protocol.datasets[DATASET],
        seeds=tuple(int(seed) for seed in seeds),
        bootstrap_seed=config.confirmation_bootstrap_seed,
    )
    return replace(
        protocol,
        split_fractions=ROLE_SPLIT,
        datasets={DATASET: preset},
    )


def _b02() -> Any:
    matches = [
        theta for theta in v5.bridge_candidates() if theta.candidate_id == BRIDGE_CANDIDATE_ID
    ]
    if len(matches) != 1:
        raise RuntimeError("the frozen B02 bridge is unavailable")
    return matches[0]


def _support_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    protocol: Any,
) -> dict[str, Any]:
    result = v2._support_worker(seed, preset, device, protocol)
    return {
        **result,
        "role_split": list(ROLE_SPLIT),
        "coverage_generated": False,
    }


def _k0_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    protocol: Any,
) -> dict[str, Any]:
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    theta = _b02()
    context = v5._context_with_theta(base_context, theta)
    metrics, detail = v5._logging_mixture_fidelity_v5(
        context, seed=seed, protocol=protocol
    )
    metric_payload = asdict(metrics)
    passed = _k0_passes_from_metrics(metric_payload)
    ratio = normalized_k0_ratio(metric_payload)
    return {
        "seed": seed,
        "dataset": DATASET,
        "passed": passed,
        "metrics": metric_payload,
        "normalized_k0_ratio": ratio if np.isfinite(ratio) else None,
        "theta": theta.to_dict(),
        "role_split": list(ROLE_SPLIT),
        "systematic_replay": detail,
        "base_context_identity": v2._context_identity(base_context),
        "kernel_context_identity": v5._candidate_context_identity(
            base_context, context.environment, theta
        ),
        "split_audit": v2._split_audit(base_context.splits),
        "coverage_generated": False,
    }


def _run_seed_phase(
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
        raise RuntimeError(f"invalid {phase} artifact root")
    phase_root.mkdir(parents=True, exist_ok=True)
    seed_to_device = _seed_device_mapping(preset.seeds, devices)
    expected_names = {
        *(f"seed_{seed:06d}.json" for seed in preset.seeds),
        "COMPLETE",
    }
    children = list(phase_root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise RuntimeError(f"invalid {phase} artifact entry")
    unexpected = [path for path in children if path.name not in expected_names]
    if unexpected:
        raise RuntimeError(f"unexpected {phase} artifacts: {sorted(unexpected)}")
    complete_path = phase_root / "COMPLETE"
    if complete_path.exists() and (
        complete_path.is_symlink()
        or not complete_path.is_file()
        or complete_path.read_text() != "complete\n"
    ):
        raise RuntimeError(f"invalid {phase} COMPLETE marker")
    completed: dict[int, dict[str, Any]] = {}
    for seed in preset.seeds:
        path = phase_root / f"seed_{seed:06d}.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        _validate_seed_envelope(
            payload,
            phase=phase,
            seed=seed,
            device=seed_to_device[seed],
            source_hash=source_hash,
        )
        completed[seed] = payload["result"]
    pending = tuple(seed for seed in preset.seeds if seed not in completed)
    if pending and (phase_root / "COMPLETE").exists():
        raise RuntimeError(f"{phase} is marked complete with missing seeds")
    groups = {
        device: tuple(seed for seed in pending if seed_to_device[seed] == device)
        for device in devices
    }
    if pending:
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
                        "phase": phase,
                        "dataset": DATASET,
                        "seed": seed,
                        "device": device,
                        "source_tree_sha256": source_hash,
                        "result": result,
                    }
                    _validate_seed_envelope(
                        payload,
                        phase=phase,
                        seed=seed,
                        device=device,
                        source_hash=source_hash,
                    )
                    _write_json(phase_root / f"seed_{seed:06d}.json", payload)
                    completed[seed] = result
    if set(completed) != set(preset.seeds):
        raise RuntimeError(f"{phase} did not complete its exact seed bank")
    if not complete_path.exists():
        _write_text(complete_path, "complete\n")
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


def _validate_seed_envelope(
    payload: Mapping[str, Any],
    *,
    phase: str,
    seed: int,
    device: str,
    source_hash: str,
) -> None:
    result = payload.get("result")
    if (
        set(payload)
        != {"protocol", "phase", "dataset", "seed", "device", "source_tree_sha256", "result"}
        or payload["protocol"] != PROTOCOL
        or payload["phase"] != phase
        or payload["dataset"] != DATASET
        or payload["seed"] != seed
        or payload["device"] != device
        or payload["source_tree_sha256"] != source_hash
        or not isinstance(result, Mapping)
        or result.get("seed") != seed
        or not isinstance(result.get("passed"), bool)
        or result.get("coverage_generated") is not False
        or result.get("role_split") != list(ROLE_SPLIT)
    ):
        raise RuntimeError(f"invalid {phase} artifact for seed {seed}")
    split_audit = result.get("split_audit")
    if (
        not isinstance(split_audit, Mapping)
        or split_audit.get("patient_sets_pairwise_disjoint") is not True
        or split_audit.get("split_fractions") != list(ROLE_SPLIT)
    ):
        raise RuntimeError(f"invalid {phase} role-split evidence for seed {seed}")
    if "k0" in phase:
        metrics = result.get("metrics")
        observed_passed = result.get("passed")
        if (
            not isinstance(metrics, Mapping)
            or not isinstance(observed_passed, bool)
            or observed_passed != _k0_passes_from_metrics(metrics)
        ):
            raise RuntimeError(f"invalid {phase} K0 decision for seed {seed}")
        identity = result.get("base_context_identity")
        theta = result.get("theta")
        if (
            not isinstance(identity, Mapping)
            or identity.get("split_fractions") != list(ROLE_SPLIT)
            or not isinstance(theta, Mapping)
            or theta.get("candidate_id") != BRIDGE_CANDIDATE_ID
        ):
            raise RuntimeError(f"invalid {phase} K0 identity for seed {seed}")
    elif "support" in phase:
        if result.get("outcome_blind") is not True:
            raise RuntimeError(f"invalid {phase} support evidence for seed {seed}")
    else:
        raise RuntimeError(f"unknown seed phase: {phase}")


def _k0_passes_from_metrics(metrics: Mapping[str, Any]) -> bool:
    expected_keys = {*K0_THRESHOLDS, "structural_invariants"}
    if set(metrics) != expected_keys or not isinstance(
        metrics.get("structural_invariants"), bool
    ):
        raise RuntimeError("invalid K0 metric schema")
    values: dict[str, float] = {}
    for name in K0_THRESHOLDS:
        value = metrics[name]
        if isinstance(value, bool):
            raise RuntimeError("invalid K0 metric value")
        try:
            values[name] = float(value)
        except (TypeError, ValueError) as error:
            raise RuntimeError("invalid K0 metric value") from error
    if (
        not all(math.isfinite(value) for value in values.values())
        or not 0.0 <= values["maximum_score_ks"] <= 1.0
        or any(value < 0.0 for name, value in values.items() if name != "maximum_score_ks")
    ):
        raise RuntimeError("invalid K0 metric value")
    return bool(metrics["structural_invariants"]) and all(
        values[name] <= threshold for name, threshold in K0_THRESHOLDS.items()
    )


def _verify_development(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _verify_complete_root(root)
    metadata = _read_json(root / "metadata.json")
    final = _read_json(root / "FINAL_STATUS.json")
    gate = _read_json(root / "development_gate.json")
    frozen = _read_json(root / "frozen_settings.json")
    if (
        metadata.get("protocol") != PROTOCOL
        or metadata.get("dataset") != DATASET
        or metadata.get("phase") != "development"
        or metadata.get("output_root") != str(root)
        or metadata.get("role_split") != list(ROLE_SPLIT)
        or metadata.get("coverage_generation_permitted") is not False
        or final.get("protocol") != PROTOCOL
        or final.get("dataset") != DATASET
        or final.get("phase") != "development"
        or final.get("status") != "DEVELOPMENT_GO"
        or final.get("development_admissible") is not True
        or final.get("role_split") != list(ROLE_SPLIT)
        or final.get("bridge_candidate_id") != BRIDGE_CANDIDATE_ID
        or gate.get("development_admissible") is not True
        or gate.get("status") != "DEVELOPMENT_GO"
        or gate.get("role_split") != list(ROLE_SPLIT)
        or gate.get("bridge_candidate_id") != BRIDGE_CANDIDATE_ID
        or gate.get("coverage_generated") is not False
        or frozen.get("protocol") != PROTOCOL
        or frozen.get("dataset") != DATASET
        or frozen.get("status") != "FROZEN_FOR_CONFIRMATION"
        or frozen.get("role_split") != list(ROLE_SPLIT)
        or frozen.get("bridge_candidate") != _b02().to_dict()
        or frozen.get("development_source_tree_sha256")
        != metadata.get("source_tree_sha256")
        or frozen.get("development_gate_sha256") != _json_sha256(gate)
        or frozen.get("config_sha256") != metadata.get("config_sha256")
        or frozen.get("coverage_generated") is not False
        or final.get("coverage_generated") is not False
    ):
        raise RuntimeError("confirmation is locked by development NO-GO")
    binding = {
        "root": str(root),
        "complete_sha256": _file_sha256(root / "COMPLETE"),
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "metadata_sha256": _file_sha256(root / "metadata.json"),
        "final_status_sha256": _file_sha256(root / "FINAL_STATUS.json"),
        "development_gate_sha256": _file_sha256(root / "development_gate.json"),
        "frozen_settings_sha256": _file_sha256(root / "frozen_settings.json"),
    }
    return {**binding, "combined_sha256": _json_sha256(binding)}, frozen


def _metadata(
    *,
    phase: str,
    output_root: Path,
    devices: Sequence[str],
    source_hash: str,
    source_snapshot: Mapping[str, Any],
    prior_binding: Mapping[str, Any],
    rng_audit: Mapping[str, Any],
    development_binding: Mapping[str, Any] | None = None,
    frozen_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "phase": phase,
        "role": "coverage_blind_post_failure_environment_support_reconstruction",
        "output_root": str(output_root),
        "devices": list(devices),
        "source_tree_sha256": source_hash,
        "source_snapshot": dict(source_snapshot),
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": _file_sha256(CONFIG_PATH),
        "prior_negative_evidence": dict(prior_binding),
        "rng_audit": dict(rng_audit),
        "role_split": list(ROLE_SPLIT),
        "bridge_candidate": _b02().to_dict(),
        "development_binding": dict(development_binding or {}),
        "frozen_settings": dict(frozen_settings or {}),
        "coverage_generation_permitted": False,
        "canonical_scpcp_mutation_permitted": False,
    }


def _prepare_root(
    root: Path,
    metadata: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if resume:
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError("resume requires a regular artifact root")
        _assert_safe_artifact_tree(root)
        _assert_no_forbidden_paths(root)
        if _read_json(root / "metadata.json") != metadata:
            raise RuntimeError("resume metadata differs from the frozen launch")
        v4._verify_source_snapshot(root, metadata["source_snapshot"])
        return
    if root.exists():
        raise FileExistsError(f"fresh output root already exists: {root}")
    root.mkdir(parents=True)
    v4._atomic_write(
        root / source_snapshot["contract"]["archive_path"],
        source_snapshot["archive_bytes"],
    )
    v4._atomic_write(
        root / source_snapshot["contract"]["manifest_path"],
        source_snapshot["manifest_bytes"],
    )
    _write_json(root / "metadata.json", metadata)
    v4._verify_source_snapshot(root, metadata["source_snapshot"])


def _complete_and_valid(root: Path, metadata: Mapping[str, Any]) -> bool:
    if not (root / "COMPLETE").exists():
        return False
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("completed root metadata differs")
    _verify_complete_root(root)
    return True


def _finalize(root: Path, metadata: Mapping[str, Any], source_hash: str) -> None:
    if experiment_tree_sha256() != source_hash:
        raise RuntimeError("source/config changed during the formal phase")
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("root metadata changed during the formal phase")
    _assert_no_forbidden_paths(root)
    _write_manifest(root)
    final = _read_json(root / "FINAL_STATUS.json")
    marker = (
        f"complete phase={metadata['phase']} source_tree_sha256={source_hash} "
        f"config_sha256={metadata['config_sha256']} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={_file_sha256(root / 'manifest.json')}\n"
    )
    _write_text(root / "COMPLETE", marker)
    _verify_complete_root(root)


def _write_manifest(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("manifest requires a regular artifact root")
    _assert_safe_artifact_tree(root)
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
    _write_json(
        root / "manifest.json",
        {"protocol": PROTOCOL, "artifact_count": len(entries), "artifacts": entries},
    )


def _verify_complete_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("complete artifact root must be a regular directory")
    _assert_safe_artifact_tree(root)
    _assert_no_forbidden_paths(root)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("protocol") != PROTOCOL or not isinstance(manifest.get("artifacts"), list):
        raise RuntimeError("invalid manifest")
    expected = set()
    for entry in manifest["artifacts"]:
        relative = Path(entry["path"])
        path = _inside(root, relative)
        expected.add(relative)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != entry["bytes"]
            or _file_sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"manifest mismatch: {relative}")
    observed = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root) not in {Path("manifest.json"), Path("COMPLETE")}
    }
    if observed != expected or manifest.get("artifact_count") != len(expected):
        raise RuntimeError("manifest file set differs")
    final = _read_json(root / "FINAL_STATUS.json")
    metadata = _read_json(root / "metadata.json")
    source_snapshot = metadata.get("source_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise RuntimeError("complete root lacks source provenance")
    for name in ("archive", "manifest"):
        relative = Path(str(source_snapshot.get(f"{name}_path", "")))
        _inside(root, relative)
    v4._verify_source_snapshot(root, source_snapshot)
    expected_marker = (
        f"complete phase={metadata['phase']} source_tree_sha256={metadata['source_tree_sha256']} "
        f"config_sha256={metadata['config_sha256']} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={_file_sha256(root / 'manifest.json')}\n"
    )
    if not (root / "COMPLETE").is_file() or (root / "COMPLETE").read_text() != expected_marker:
        raise RuntimeError("COMPLETE marker differs")


def _assert_no_forbidden_paths(root: Path) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(token in part.lower() for part in relative.parts for token in FORBIDDEN_PATH_TOKENS):
            raise RuntimeError(f"forbidden precoverage artifact path: {relative}")


def _assert_safe_artifact_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden: {path}")
        if path.is_file() and ".tmp-" in path.name:
            raise RuntimeError(f"temporary artifact remains: {path}")


def _rng_audits(config: EnvironmentSupportConfig) -> dict[str, Any]:
    development_seeds = tuple(
        seed for seeds in DEVELOPMENT_BLOCKS.values() for seed in seeds
    )
    development_mapping = _precoverage_rng_stream_mapping(development_seeds, config)
    confirmation_mapping = _precoverage_rng_stream_mapping(CONFIRMATION_SEEDS, config)
    cross_bank_collisions = _cross_bank_rng_collisions(
        development_mapping,
        confirmation_mapping,
    )
    if cross_bank_collisions:
        raise RuntimeError(
            "development and confirmation RNG streams overlap: "
            f"{cross_bank_collisions}"
        )
    cross_bank_contract = {
        "development_stream_count": len(development_mapping),
        "development_mapping_sha256": _json_sha256(development_mapping),
        "confirmation_stream_count": len(confirmation_mapping),
        "confirmation_mapping_sha256": _json_sha256(confirmation_mapping),
        "collision_count": 0,
        "collisions": {},
    }
    development = _audit_rng_bank(
        development_seeds,
        config,
        label="development",
        mapping=development_mapping,
    )
    confirmation = _audit_rng_bank(
        CONFIRMATION_SEEDS,
        config,
        label="confirmation",
        mapping=confirmation_mapping,
    )
    return {
        "development": {
            **development,
            "cross_bank_audit": cross_bank_contract,
        },
        "confirmation": {
            **confirmation,
            "cross_bank_audit": cross_bank_contract,
        },
    }


def _audit_rng_bank(
    seeds: Sequence[int],
    config: EnvironmentSupportConfig,
    *,
    label: str,
    mapping: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if label not in {"development", "confirmation"}:
        raise ValueError("unknown RNG audit role")
    mapping = dict(
        _precoverage_rng_stream_mapping(seeds, config)
        if mapping is None
        else mapping
    )
    v2._assert_unique_rng_streams(mapping)
    excluded_roots = {DEVELOPMENT_ROOT, CONFIRMATION_ROOT, SCIENCE_ROOT}
    artifact_ids = _artifact_rng_ids(ROOT / "results", excluded_roots=excluded_roots)
    source_ids = v2._source_declared_seeds(
        ROOT, excluded_paths={path for path in _OWN_RNG_DECLARATION_PATHS if path.exists()}
    )
    prior = artifact_ids | source_ids
    collisions = {name: value for name, value in mapping.items() if value in prior}
    authorized_ids: set[int] = set()
    if label == "development":
        visible_pilot_seeds = tuple(
            seed for seeds in config.pilot_visible_at_freeze.values() for seed in seeds
        )
        authorized_ids = set(
            _precoverage_rng_stream_mapping(visible_pilot_seeds, config).values()
        )
    unauthorized = {
        name: value for name, value in collisions.items() if value not in authorized_ids
    }
    if unauthorized:
        raise RuntimeError(f"unauthorized {label} RNG collision: {unauthorized}")
    if label == "confirmation" and collisions:
        raise RuntimeError(f"fresh confirmation RNG collision: {collisions}")
    return {
        "status": "passed_before_launch" if not collisions else "development_reuse_recorded",
        "role": label,
        "collision_count": len(collisions),
        "collisions": collisions,
        "authorized_collision_count": len(collisions) - len(unauthorized),
        "unauthorized_collision_count": len(unauthorized),
        "unauthorized_collisions": unauthorized,
        "artifact_rng_id_count": len(artifact_ids),
        "artifact_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "source_declared_rng_id_count": len(source_ids),
        "source_declared_rng_id_sha256": _integer_set_sha256(source_ids),
        "prior_rng_id_count": len(prior),
        "prior_rng_id_sha256": _integer_set_sha256(prior),
        "new_rng_stream_count": len(mapping),
        "new_rng_stream_mapping": mapping,
        "new_rng_stream_mapping_sha256": _json_sha256(mapping),
        "new_rng_id_set_sha256": _integer_set_sha256(mapping.values()),
        "base_seed_count": len(seeds),
        "base_seed_set_sha256": _integer_set_sha256(seeds),
        "executed_stream_kinds": [
            suffix.removeprefix("/") for suffix in PRECOVERAGE_RNG_STREAM_SUFFIXES
        ],
        "executed_streams_only": True,
        "scientific_freshness_claimed": label == "confirmation",
    }


def _precoverage_rng_stream_mapping(
    seeds: Sequence[int], config: EnvironmentSupportConfig
) -> dict[str, int]:
    protocol = _protocol_for(seeds, config)
    declared = v2._new_rng_stream_mapping(protocol, (DATASET,))
    mapping = {
        name: value
        for name, value in declared.items()
        if name.endswith(PRECOVERAGE_RNG_STREAM_SUFFIXES)
    }
    expected_count = len(tuple(seeds)) * len(PRECOVERAGE_RNG_STREAM_SUFFIXES)
    if len(mapping) != expected_count:
        raise RuntimeError("precoverage RNG stream mapping is incomplete")
    v2._assert_unique_rng_streams(mapping)
    return mapping


def _cross_bank_rng_collisions(
    development_mapping: Mapping[str, int],
    confirmation_mapping: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    development_by_id: dict[int, list[str]] = {}
    for name, value in development_mapping.items():
        development_by_id.setdefault(int(value), []).append(name)
    return {
        name: {
            "rng_id": int(value),
            "development_streams": sorted(development_by_id[int(value)]),
        }
        for name, value in confirmation_mapping.items()
        if int(value) in development_by_id
    }


def _validate_rng_audit(audit: Mapping[str, Any], *, phase: str) -> None:
    if phase not in {"development", "confirmation"}:
        raise ValueError("unknown RNG audit phase")
    mapping = audit.get("new_rng_stream_mapping")
    collisions = audit.get("collisions")
    unauthorized = audit.get("unauthorized_collisions")
    cross_bank = audit.get("cross_bank_audit")
    base_seed_count = audit.get("base_seed_count")
    expected_status = (
        "development_reuse_recorded"
        if phase == "development" and bool(collisions)
        else "passed_before_launch"
    )
    if (
        audit.get("role") != phase
        or audit.get("status") != expected_status
        or audit.get("unauthorized_collision_count") != 0
        or unauthorized != {}
        or audit.get("executed_streams_only") is not True
        or audit.get("executed_stream_kinds")
        != [suffix.removeprefix("/") for suffix in PRECOVERAGE_RNG_STREAM_SUFFIXES]
        or not isinstance(mapping, Mapping)
        or not all(
            isinstance(name, str)
            and name.endswith(PRECOVERAGE_RNG_STREAM_SUFFIXES)
            and isinstance(value, int)
            and not isinstance(value, bool)
            for name, value in mapping.items()
        )
        or audit.get("new_rng_stream_count") != len(mapping)
        or audit.get("new_rng_stream_mapping_sha256") != _json_sha256(mapping)
        or audit.get("new_rng_id_set_sha256")
        != _integer_set_sha256(mapping.values())
        or len(set(mapping.values())) != len(mapping)
        or not isinstance(collisions, Mapping)
        or audit.get("collision_count") != len(collisions)
        or audit.get("authorized_collision_count") != len(collisions)
        or not isinstance(base_seed_count, int)
        or isinstance(base_seed_count, bool)
        or base_seed_count * len(PRECOVERAGE_RNG_STREAM_SUFFIXES)
        != len(mapping)
        or not _is_sha256(audit.get("base_seed_set_sha256"))
        or audit.get("scientific_freshness_claimed") is not (
            phase == "confirmation"
        )
        or any(
            isinstance(audit.get(f"{name}_rng_id_count"), bool)
            or not isinstance(audit.get(f"{name}_rng_id_count"), int)
            or audit.get(f"{name}_rng_id_count") < 0
            or not _is_sha256(audit.get(f"{name}_rng_id_sha256"))
            for name in ("artifact", "source_declared", "prior")
        )
        or not isinstance(cross_bank, Mapping)
        or cross_bank.get("collision_count") != 0
        or cross_bank.get("collisions") != {}
        or cross_bank.get(f"{phase}_stream_count") != len(mapping)
        or cross_bank.get(f"{phase}_mapping_sha256") != _json_sha256(mapping)
    ):
        raise RuntimeError(f"invalid {phase} RNG audit")
    if phase == "confirmation" and (
        audit.get("collision_count") != 0 or audit.get("collisions") != {}
    ):
        raise RuntimeError("invalid confirmation RNG audit")


def _artifact_rng_ids(root: Path, *, excluded_roots: set[Path]) -> set[int]:
    values: set[int] = set()
    if not root.exists():
        return values
    excluded = {path.resolve() for path in excluded_roots}
    for path in root.rglob("*.json"):
        resolved = path.resolve()
        if any(resolved == item or item in resolved.parents for item in excluded):
            continue
        match = v2._SEED_NAME.fullmatch(path.name)
        if match:
            values.add(int(match.group(1)))
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        v2._collect_artifact_rng_values(payload, values)
    return values


def _assert_fresh_roots() -> None:
    existing = [
        str(path)
        for path in (DEVELOPMENT_ROOT, CONFIRMATION_ROOT, SCIENCE_ROOT)
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(f"formal roots must be absent at audit: {existing}")


def _seed_device_mapping(seeds: Sequence[int], devices: Sequence[str]) -> dict[int, str]:
    return {seed: devices[index % len(devices)] for index, seed in enumerate(seeds)}


def _validate_devices(devices: Sequence[str]) -> None:
    try:
        resolved = tuple(torch.device(value) for value in devices)
    except RuntimeError as error:
        raise ValueError("formal CXR study requires valid CUDA devices") from error
    indices = tuple(device.index for device in resolved)
    if (
        len(resolved) != 2
        or any(device.type != "cuda" or device.index is None for device in resolved)
        or len(set(indices)) != 2
    ):
        raise ValueError("formal CXR study requires exactly two explicit CUDA devices")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
        or max(int(index) for index in indices if index is not None)
        >= torch.cuda.device_count()
    ):
        raise RuntimeError("two CUDA devices are required; CPU fallback is forbidden")


def _inside(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("artifact path escapes root")
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents:
        raise RuntimeError("artifact path escapes root")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: object) -> None:
    v4._write_json(path, value)


def _write_text(path: Path, value: str) -> None:
    v4._write_text(path, value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _integer_set_sha256(values: Iterable[int]) -> str:
    return _json_sha256(sorted(set(int(value) for value in values)))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


if __name__ == "__main__":
    main()
