"""Run the frozen Phase 0 profiled-versus-greedy oracle study."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import (
    experiment_tree_sha256,
    mark_study_complete,
    mark_study_failed,
    source_tree_sha256,
    write_seed_result,
    write_study_metadata,
)
from scpcp.config import ExperimentConfig
from scpcp.device import resolve_devices
from scpcp.phase0_oracle import run_phase0_seed


DEFAULT_CONFIG = ROOT / "configs" / "phase0_oracle.yaml"
PRIMARY_ROWS = {
    ("standard", "Current Profiled Oracle"),
    ("standard", "Greedy Sequential Oracle"),
    ("tail_shift", "Current Profiled Oracle"),
    ("tail_shift", "Greedy Sequential Oracle"),
}
SEED_DIRECTORY = re.compile(r"seed_(\d{5})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Phase 0 profiled-versus-greedy oracle study"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--seeds",
        default=None,
        help="range such as 0:100 or comma-separated seeds",
    )
    parser.add_argument("--devices", default=None, help="comma-separated CUDA devices")
    parser.add_argument(
        "--workers-per-device",
        type=int,
        default=1,
        help="persistent single-process workers assigned to each GPU",
    )
    parser.add_argument("--candidate-chunk-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.workers_per_device < 1:
        parser.error("--workers-per-device must be positive")
    if args.candidate_chunk_size < 1:
        parser.error("--candidate-chunk-size must be positive")

    base = ExperimentConfig.from_yaml(args.config)
    devices = resolve_devices(args.devices or base.devices)
    seeds = parse_seeds(args.seeds, base.seeds)
    output_dir = args.output_dir or base.output_dir
    config = base.with_overrides(
        devices=devices,
        seeds=seeds,
        output_dir=output_dir,
    )
    run_config(
        config,
        output_dir,
        workers_per_device=args.workers_per_device,
        candidate_chunk_size=args.candidate_chunk_size,
        resume=args.resume,
    )
    print(output_dir)


def run_config(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    workers_per_device: int,
    candidate_chunk_size: int,
    resume: bool,
) -> None:
    """Run or safely resume one exact Phase 0 configuration."""

    if workers_per_device < 1:
        raise ValueError("workers_per_device must be positive")
    if candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be positive")
    if config.output_dir != output_dir:
        raise ValueError("config output_dir must exactly match the requested output_dir")

    config_hash = canonical_config_sha256(config.to_dict())
    current_source_hash = source_tree_sha256()
    current_experiment_hash = experiment_tree_sha256()
    execution = {
        "experiment_tree_sha256": current_experiment_hash,
        "config_sha256": config_hash,
        "workers_per_device": workers_per_device,
        "candidate_chunk_size": candidate_chunk_size,
    }

    if resume:
        _validate_resume_provenance(
            output_dir,
            config,
            config_hash=config_hash,
            source_hash=current_source_hash,
            experiment_hash=current_experiment_hash,
            workers_per_device=workers_per_device,
            candidate_chunk_size=candidate_chunk_size,
        )
        completed = _validated_existing_seeds(
            output_dir,
            config.seeds,
            expected_source_hash=current_source_hash,
            expected_config_hash=config_hash,
        )
        if (output_dir / "COMPLETE").is_file() and completed != set(config.seeds):
            missing = sorted(set(config.seeds) - completed)
            raise RuntimeError(
                f"study COMPLETE exists but seed artifacts are missing: {missing}"
            )
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh phase0 output already exists: {output_dir}")
        write_study_metadata(output_dir, config, execution=execution)
        completed = set()

    pending = tuple(seed for seed in config.seeds if seed not in completed)
    try:
        _run_pending_seeds(
            config,
            output_dir,
            pending,
            workers_per_device=workers_per_device,
            candidate_chunk_size=candidate_chunk_size,
        )
        for seed in config.seeds:
            validate_seed_artifact(
                output_dir / f"seed_{seed:05d}",
                seed,
                expected_source_hash=current_source_hash,
                expected_config_hash=config_hash,
            )
        mark_study_complete(output_dir, config.seeds)
    except BaseException as error:
        mark_study_failed(output_dir, config.seeds, error)
        raise


def validate_seed_artifact(
    seed_dir: Path,
    seed: int,
    *,
    expected_source_hash: str | None = None,
    expected_config_hash: str | None = None,
) -> Path:
    """Require the atomic files and exact four-row primary Phase 0 contract."""

    required = ("COMPLETE", "records.csv", "surfaces.npz", "metadata.json")
    missing = [name for name in required if not (seed_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"seed {seed} artifact is partial; missing files: {missing}")

    try:
        metadata = json.loads((seed_dir / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"seed {seed} metadata.json is unreadable: {error}") from error
    if not isinstance(metadata, dict):
        raise RuntimeError(f"seed {seed} metadata.json must contain an object")
    if metadata.get("seed") != seed:
        raise RuntimeError(f"seed {seed} metadata.json has the wrong seed ID")
    if (
        expected_source_hash is not None
        and metadata.get("source_tree_sha256") != expected_source_hash
    ):
        raise RuntimeError(f"seed {seed} source hash differs from the study source")
    if expected_config_hash is not None:
        stored_seed_config = metadata.get("config")
        if not isinstance(stored_seed_config, dict):
            raise RuntimeError(f"seed {seed} metadata config must contain an object")
        if canonical_config_sha256(stored_seed_config) != expected_config_hash:
            raise RuntimeError(f"seed {seed} config differs from the study config")

    try:
        with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as surfaces:
            tuple(surfaces.files)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"seed {seed} surfaces.npz is unreadable: {error}") from error

    try:
        records = pd.read_csv(seed_dir / "records.csv")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise RuntimeError(f"seed {seed} records.csv is unreadable: {error}") from error
    required_columns = {"scenario", "method", "seed"}
    missing_columns = sorted(required_columns - set(records.columns))
    if missing_columns:
        raise RuntimeError(
            f"seed {seed} records.csv is missing columns: {missing_columns}"
        )
    pairs = list(zip(records["scenario"], records["method"], strict=True))
    if len(records) != 4 or set(pairs) != PRIMARY_ROWS or len(set(pairs)) != 4:
        raise RuntimeError(
            f"seed {seed} records.csv must contain exactly four primary rows: "
            "standard/tail_shift x Current Profiled Oracle/Greedy Sequential Oracle"
        )
    try:
        record_seeds = {int(record_seed) for record_seed in records["seed"]}
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"seed {seed} records.csv contains an invalid seed ID"
        ) from error
    if record_seeds != {seed}:
        raise RuntimeError(f"seed {seed} records.csv contains a different seed ID")
    return seed_dir


def canonical_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_resume_provenance(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    config_hash: str,
    source_hash: str,
    experiment_hash: str,
    workers_per_device: int,
    candidate_chunk_size: int,
) -> None:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"resume output does not exist: {output_dir}")
    try:
        metadata = json.loads((output_dir / "study_metadata.json").read_text())
        stored_config = yaml.safe_load((output_dir / "config.yaml").read_text())
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise RuntimeError(f"resume metadata is unreadable in {output_dir}: {error}") from error
    if not isinstance(stored_config, dict):
        raise RuntimeError("resume stored config is not a mapping")
    if canonical_config_sha256(stored_config) != config_hash:
        raise RuntimeError("resume stored config differs from the requested config")
    if metadata.get("seeds") != list(config.seeds):
        raise RuntimeError("resume requested seeds differ from study metadata")
    if metadata.get("devices") != list(config.devices):
        raise RuntimeError("resume requested devices differ from study metadata")
    if metadata.get("source_tree_sha256") != source_hash:
        raise RuntimeError("resume source_tree_sha256 differs from the active source")

    execution = metadata.get("execution")
    if not isinstance(execution, dict):
        raise RuntimeError("resume execution metadata is missing")
    expected_execution = {
        "experiment_tree_sha256": experiment_hash,
        "config_sha256": config_hash,
        "workers_per_device": workers_per_device,
        "candidate_chunk_size": candidate_chunk_size,
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise RuntimeError(f"resume {key} differs from the requested execution")


def _validated_existing_seeds(
    output_dir: Path,
    requested_seeds: tuple[int, ...],
    *,
    expected_source_hash: str,
    expected_config_hash: str,
) -> set[int]:
    requested = set(requested_seeds)
    completed: set[int] = set()
    for path in output_dir.iterdir():
        if path.name.startswith(".seed_"):
            raise RuntimeError(f"partial atomic seed directory blocks resume: {path}")
        if not path.name.startswith("seed_"):
            continue
        match = SEED_DIRECTORY.fullmatch(path.name)
        if match is None or not path.is_dir():
            raise RuntimeError(f"malformed seed path blocks resume: {path}")
        seed = int(match.group(1))
        if seed not in requested:
            raise RuntimeError(f"unexpected seed {seed} directory blocks resume: {path}")
        validate_seed_artifact(
            path,
            seed,
            expected_source_hash=expected_source_hash,
            expected_config_hash=expected_config_hash,
        )
        completed.add(seed)
    return completed


def _run_pending_seeds(
    config: ExperimentConfig,
    output_dir: Path,
    seeds: tuple[int, ...],
    *,
    workers_per_device: int,
    candidate_chunk_size: int,
) -> None:
    if not seeds:
        return
    worker_devices, jobs = _build_seed_jobs(
        seeds,
        config.devices,
        workers_per_device,
    )
    with ExitStack() as stack:
        executors = [
            stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=get_context("spawn"),
                )
            )
            for _ in worker_devices
        ]
        futures = {
            executors[worker_index].submit(
                _run_and_write,
                config,
                seed,
                device,
                output_dir,
                candidate_chunk_size,
            ): seed
            for worker_index, seed, device in jobs
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)


def _run_and_write(
    config: ExperimentConfig,
    seed: int,
    device: str,
    output_dir: Path,
    candidate_chunk_size: int,
) -> str:
    try:
        result = run_phase0_seed(
            config,
            seed=seed,
            device=device,
            candidate_chunk_size=candidate_chunk_size,
        )
        seed_dir = write_seed_result(result, output_dir, config)
        validate_seed_artifact(seed_dir, seed)
        return str(seed_dir)
    finally:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def _build_seed_jobs(
    seeds: tuple[int, ...],
    devices: tuple[str, ...],
    workers_per_device: int,
) -> tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]]:
    if workers_per_device < 1:
        raise ValueError("workers_per_device must be positive")
    worker_devices = tuple(
        device for device in devices for _ in range(workers_per_device)
    )
    jobs = tuple(
        (index % len(worker_devices), seed, worker_devices[index % len(worker_devices)])
        for index, seed in enumerate(seeds)
    )
    return worker_devices, jobs


def parse_seeds(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        seeds = default
    elif ":" in value:
        start, stop = (int(part) for part in value.split(":", maxsplit=1))
        seeds = tuple(range(start, stop))
    else:
        seeds = tuple(int(part) for part in value.split(",") if part)
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


if __name__ == "__main__":
    main()
