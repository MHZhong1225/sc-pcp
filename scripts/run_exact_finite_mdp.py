"""Run the isolated exact finite-MDP committed-prefix diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import git_revision, source_tree_sha256
from scpcp.exact_finite_mdp import (
    ESTIMATOR_NAMES,
    MECHANISM_NAMES,
    SEED_NAMESPACE,
    ExactFiniteMDPConfig,
    exact_seed_collision_audit,
)
from scpcp.exact_finite_mdp_study import run_replicated_exact_finite_mdp


DEFAULT_OUTPUT = ROOT / "results" / "work" / "exact_finite_mdp"
PROTOCOL = "exact_committed_prefix_finite_mdp_v1"
PAYLOAD_NAMES = ("config.json", "metadata.json", "summary.json", "surfaces.npz")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact M0--M3 committed-prefix identification diagnostics"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--logged-trajectories", type=int, default=None)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--logged-instances", type=int, default=None)
    parser.add_argument("--logged-replicates", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--surface-chunk-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ExactFiniteMDPConfig:
    overrides = {
        name: value
        for name, value in {
            "seed": args.seed,
            "logged_trajectories": args.logged_trajectories,
            "population_instances": args.instances,
            "logged_instance_count": args.logged_instances,
            "logged_replicates": args.logged_replicates,
            "beam_width": args.beam_width,
            "surface_chunk_size": args.surface_chunk_size,
        }.items()
        if value is not None
    }
    config = replace(ExactFiniteMDPConfig(), **overrides)
    config.validate()
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    run_study(config, args.output, resume=args.resume)
    print(args.output)


def run_study(
    config: ExactFiniteMDPConfig,
    output_dir: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Run once, or validate and reuse one complete immutable bundle."""

    config.validate()
    config_payload = config.to_dict()
    config_hash = _canonical_sha256(config_payload)
    source_hash = source_tree_sha256()
    if resume:
        validate_bundle(
            output_dir,
            expected_config=config_payload,
            expected_config_hash=config_hash,
            expected_source_hash=source_hash,
        )
        return _read_json(output_dir / "summary.json")
    if output_dir.exists():
        raise FileExistsError(f"fresh exact finite-MDP output already exists: {output_dir}")

    started = time.perf_counter()
    result = run_replicated_exact_finite_mdp(config)
    elapsed = time.perf_counter() - started
    if source_tree_sha256() != source_hash:
        raise RuntimeError("source tree changed while exact finite-MDP study was running")
    metadata = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "diagnostic_only": True,
        "canonical_method_unchanged": True,
        "device": "cpu_exact",
        "git_revision": git_revision(),
        "source_tree_sha256": source_hash,
        "config_sha256": config_hash,
        "seed_namespace": (
            SEED_NAMESPACE if 52_000 <= config.seed <= 52_999 else "custom"
        ),
        "seed_collision_audit": exact_seed_collision_audit(config),
        "runtime_seconds": elapsed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _publish_bundle(
        output_dir,
        config_payload=config_payload,
        metadata=metadata,
        summary=result.summary,
        arrays=result.arrays,
    )
    validate_bundle(
        output_dir,
        expected_config=config_payload,
        expected_config_hash=config_hash,
        expected_source_hash=source_hash,
    )
    return result.summary


def validate_bundle(
    output_dir: Path,
    *,
    expected_config: dict[str, Any],
    expected_config_hash: str,
    expected_source_hash: str,
) -> None:
    """Fail closed on partial, mutated, or provenance-mismatched output."""

    if not output_dir.is_dir():
        raise FileNotFoundError(f"exact finite-MDP output does not exist: {output_dir}")
    required = {*PAYLOAD_NAMES, "manifest.json", "COMPLETE"}
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"partial exact finite-MDP bundle; missing files: {missing}")

    manifest = _read_json(output_dir / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("exact finite-MDP manifest has the wrong protocol")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(PAYLOAD_NAMES):
        raise RuntimeError("exact finite-MDP manifest file contract differs")
    for name in PAYLOAD_NAMES:
        path = output_dir / name
        expected = files[name]
        if not isinstance(expected, dict):
            raise RuntimeError(f"manifest entry for {name} is malformed")
        content = path.read_bytes()
        if expected != {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }:
            raise RuntimeError(f"exact finite-MDP payload hash differs: {name}")

    complete = _read_json(output_dir / "COMPLETE")
    manifest_digest = hashlib.sha256((output_dir / "manifest.json").read_bytes()).hexdigest()
    if complete != {
        "status": "complete",
        "manifest_sha256": manifest_digest,
    }:
        raise RuntimeError("exact finite-MDP COMPLETE marker is malformed")
    stored_config = _read_json(output_dir / "config.json")
    if stored_config != expected_config:
        raise RuntimeError("resume config differs from exact finite-MDP bundle")
    metadata = _read_json(output_dir / "metadata.json")
    if metadata.get("config_sha256") != expected_config_hash:
        raise RuntimeError("resume config hash differs from exact finite-MDP bundle")
    if metadata.get("source_tree_sha256") != expected_source_hash:
        raise RuntimeError("resume source tree differs from exact finite-MDP bundle")

    summary = _read_json(output_dir / "summary.json")
    if (
        summary.get("study") != "exact_committed_prefix_finite_mdp"
        or summary.get("status") != "complete"
        or summary.get("schedule_count")
        != expected_config["grid_size"] ** expected_config["horizon"]
    ):
        raise RuntimeError("exact finite-MDP summary contract differs")
    try:
        with np.load(output_dir / "surfaces.npz", allow_pickle=False) as arrays:
            _validate_surface_arrays(arrays, expected_config)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"exact finite-MDP surfaces are unreadable: {error}") from error


def _validate_surface_arrays(
    arrays: Any,
    config: dict[str, Any],
) -> None:
    required = {
        "schedule_indices",
        "radius_grid",
        "mechanism_names",
        "estimator_names",
        "true_coverage",
        "population_coverage",
        "hajek_coverage",
        "identification_bias",
        "finite_sample_sampling_error",
        "total_error",
        "ess_fraction",
        "target_normalized_width",
        "population_problem_seeds",
        "population_identification_maximum_absolute",
        "population_identification_rmse",
        "population_feasible_schedule_fraction",
        "population_global_available",
        "population_greedy_available",
        "population_global_mean_width",
        "population_greedy_mean_width",
        "population_greedy_absolute_regret",
        "population_greedy_relative_regret",
        "population_global_schedule_indices",
        "population_greedy_schedule_indices",
        "logged_problem_seeds",
        "logged_randomness_seeds",
        "logged_sampling_maximum_absolute",
        "logged_sampling_rmse",
        "logged_total_maximum_absolute",
        "logged_total_rmse",
        "logged_ess_fraction_minimum",
        "logged_ess_fraction_median",
    }
    if set(arrays.files) != required:
        raise RuntimeError("exact finite-MDP surface names differ")
    schedule_count = config["grid_size"] ** config["horizon"]
    surface_shape = (
        len(MECHANISM_NAMES),
        len(ESTIMATOR_NAMES),
        schedule_count,
        config["horizon"],
    )
    if arrays["schedule_indices"].shape != (schedule_count, config["horizon"]):
        raise RuntimeError("schedule_indices has the wrong shape")
    if arrays["true_coverage"].shape != (
        len(MECHANISM_NAMES),
        schedule_count,
        config["horizon"],
    ):
        raise RuntimeError("true_coverage has the wrong shape")
    for name in (
        "population_coverage",
        "hajek_coverage",
        "identification_bias",
        "finite_sample_sampling_error",
        "total_error",
        "ess_fraction",
    ):
        if arrays[name].shape != surface_shape:
            raise RuntimeError(f"{name} has the wrong shape")
        if not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"{name} contains non-finite values")
    if arrays["mechanism_names"].tolist() != list(MECHANISM_NAMES):
        raise RuntimeError("mechanism name order differs")
    if arrays["estimator_names"].tolist() != list(ESTIMATOR_NAMES):
        raise RuntimeError("estimator name order differs")

    population_instances = config["population_instances"]
    population_metric_shape = (
        population_instances,
        len(MECHANISM_NAMES),
        len(ESTIMATOR_NAMES),
    )
    if arrays["population_problem_seeds"].shape != (population_instances,):
        raise RuntimeError("population problem seeds have the wrong shape")
    for name in (
        "population_identification_maximum_absolute",
        "population_identification_rmse",
    ):
        if arrays[name].shape != population_metric_shape:
            raise RuntimeError(f"{name} has the wrong shape")
    for name in (
        "population_feasible_schedule_fraction",
        "population_global_available",
        "population_greedy_available",
        "population_global_mean_width",
        "population_greedy_mean_width",
        "population_greedy_absolute_regret",
        "population_greedy_relative_regret",
    ):
        if arrays[name].shape != (population_instances, len(MECHANISM_NAMES)):
            raise RuntimeError(f"{name} has the wrong shape")
        if np.isinf(arrays[name]).any():
            raise RuntimeError(f"{name} contains infinite values")
    population_schedule_shape = (
        population_instances,
        len(MECHANISM_NAMES),
        config["horizon"],
    )
    for name in (
        "population_global_schedule_indices",
        "population_greedy_schedule_indices",
    ):
        if arrays[name].shape != population_schedule_shape:
            raise RuntimeError(f"{name} has the wrong shape")

    logged_instances = config["logged_instance_count"]
    logged_replicates = config["logged_replicates"]
    if arrays["logged_problem_seeds"].shape != (logged_instances,):
        raise RuntimeError("logged problem seeds have the wrong shape")
    if arrays["logged_randomness_seeds"].shape != (
        logged_instances,
        logged_replicates,
    ):
        raise RuntimeError("logged randomness seeds have the wrong shape")
    logged_metric_shape = (
        logged_instances,
        logged_replicates,
        len(MECHANISM_NAMES),
        len(ESTIMATOR_NAMES),
    )
    for name in (
        "logged_sampling_maximum_absolute",
        "logged_sampling_rmse",
        "logged_total_maximum_absolute",
        "logged_total_rmse",
        "logged_ess_fraction_minimum",
        "logged_ess_fraction_median",
    ):
        if arrays[name].shape != logged_metric_shape:
            raise RuntimeError(f"{name} has the wrong shape")
        if not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"{name} contains non-finite values")


def _publish_bundle(
    output_dir: Path,
    *,
    config_payload: dict[str, Any],
    metadata: dict[str, Any],
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        _write_json(temporary / "config.json", config_payload)
        _write_json(temporary / "metadata.json", metadata)
        _write_json(temporary / "summary.json", summary)
        np.savez_compressed(temporary / "surfaces.npz", **arrays)
        _fsync_file(temporary / "surfaces.npz")
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "complete",
            "files": {
                name: _file_contract(temporary / name) for name in PAYLOAD_NAMES
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        _write_json(
            temporary / "COMPLETE",
            {
                "status": "complete",
                "manifest_sha256": hashlib.sha256(
                    (temporary / "manifest.json").read_bytes()
                ).hexdigest(),
            },
        )
        _fsync_directory(temporary)
        os.replace(temporary, output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _file_contract(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def _canonical_sha256(payload: dict[str, Any]) -> str:
    content = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unreadable exact finite-MDP JSON: {path.name}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"exact finite-MDP JSON must contain an object: {path.name}")
    return payload


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    main()
