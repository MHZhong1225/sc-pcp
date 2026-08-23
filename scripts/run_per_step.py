"""Run the per-step SC-PCP experiment on one or two GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from multiprocessing import get_context
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import (
    mark_study_complete,
    mark_study_failed,
    source_tree_sha256,
    write_seed_result,
    write_study_metadata,
)
from scpcp.config import ExperimentConfig
from scpcp.device import resolve_devices
from scpcp.experiment import run_seed


SEED_DIRECTORY = re.compile(r"seed_(\d{5})")
PAPER_METHODS = {"Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run per-step performative SC-PCP")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--devices", default=None, help="comma-separated CUDA devices; default uses both GPUs")
    parser.add_argument("--seeds", default=None, help="seed range such as 0:20 or comma-separated integers")
    parser.add_argument(
        "--workers-per-device",
        type=int,
        default=1,
        help="independent persistent seed workers per GPU",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only from an exact, validated study directory",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    devices = resolve_devices(args.devices or config.devices)
    if args.workers_per_device < 1:
        parser.error("--workers-per-device must be positive")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires the exact existing --output-dir")
    seeds = _parse_seeds(args.seeds, config.seeds)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or config.output_dir / f"{timestamp}_{config.data.dataset}"
    config = config.with_overrides(devices=devices, seeds=seeds, output_dir=output_dir)
    run_config(
        config,
        output_dir,
        workers_per_device=args.workers_per_device,
        resume=args.resume,
    )
    print(output_dir)


def run_config(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    workers_per_device: int,
    resume: bool = False,
) -> None:
    """Run a fresh study or safely fill missing seeds in an exact study."""

    if workers_per_device < 1:
        raise ValueError("workers_per_device must be positive")
    if config.output_dir != output_dir:
        raise ValueError("config output_dir must exactly match output_dir")

    execution = {"workers_per_device": workers_per_device}
    source_hash = source_tree_sha256()
    if resume:
        _validate_resume_provenance(
            output_dir,
            config,
            source_hash=source_hash,
            execution=execution,
        )
        completed = _validated_existing_seeds(
            output_dir,
            config.seeds,
            expected_source_hash=source_hash,
            expected_config=_normalized_config(config),
        )
        if (output_dir / "COMPLETE").is_file() and completed != set(config.seeds):
            missing = sorted(set(config.seeds) - completed)
            raise RuntimeError(
                f"study COMPLETE exists but seed artifacts are missing: {missing}"
            )
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh study output already exists: {output_dir}")
        write_study_metadata(output_dir, config, execution=execution)
        completed = set()

    pending = tuple(seed for seed in config.seeds if seed not in completed)
    try:
        _run_pending_seeds(
            config,
            output_dir,
            pending,
            workers_per_device=workers_per_device,
        )
        for seed in config.seeds:
            validate_seed_artifact(
                output_dir / f"seed_{seed:05d}",
                seed,
                expected_source_hash=source_hash,
                expected_config=_normalized_config(config),
            )
        mark_study_complete(output_dir, config.seeds)
    except BaseException as error:
        mark_study_failed(output_dir, config.seeds, error)
        raise


def _run_pending_seeds(
    config: ExperimentConfig,
    output_dir: Path,
    seeds: tuple[int, ...],
    *,
    workers_per_device: int,
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
                ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
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
            ): seed
            for worker_index, seed, device in jobs
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)


def validate_seed_artifact(
    seed_dir: Path,
    seed: int,
    *,
    expected_source_hash: str,
    expected_config: dict[str, object],
) -> Path:
    """Validate an atomic six-method seed before treating it as complete."""

    required = ("COMPLETE", "records.csv", "surfaces.npz", "metadata.json")
    missing = [name for name in required if not (seed_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"seed {seed} artifact is partial; missing files: {missing}")

    try:
        marker = json.loads((seed_dir / "COMPLETE").read_text())
        metadata = json.loads((seed_dir / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"seed {seed} metadata is unreadable: {error}") from error
    if marker != {"seed": seed, "status": "complete"}:
        raise RuntimeError(f"seed {seed} COMPLETE marker is invalid")
    if metadata.get("seed") != seed:
        raise RuntimeError(f"seed {seed} metadata has the wrong seed ID")
    if metadata.get("source_tree_sha256") != expected_source_hash:
        raise RuntimeError(f"seed {seed} source hash differs from the study source")
    if metadata.get("config") != expected_config:
        raise RuntimeError(f"seed {seed} config differs from the study config")

    try:
        records = pd.read_csv(seed_dir / "records.csv")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise RuntimeError(f"seed {seed} records.csv is unreadable: {error}") from error
    if "method" not in records or len(records) != len(PAPER_METHODS):
        raise RuntimeError(f"seed {seed} records.csv must contain six method rows")
    methods = records["method"].astype(str)
    if set(methods) != PAPER_METHODS or methods.duplicated().any():
        raise RuntimeError(f"seed {seed} records.csv has an invalid method set")

    try:
        with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as surfaces:
            required_surfaces = {
                "scpcp_stage_grids",
                "scpcp_candidate_coverage",
                "scpcp_selected_indices",
            }
            missing_surfaces = sorted(required_surfaces - set(surfaces.files))
            if missing_surfaces:
                raise RuntimeError(
                    f"seed {seed} surfaces.npz is missing arrays: {missing_surfaces}"
                )
            for name in surfaces.files:
                np.asarray(surfaces[name])
    except (OSError, ValueError) as error:
        raise RuntimeError(f"seed {seed} surfaces.npz is unreadable: {error}") from error
    return seed_dir


def _validate_resume_provenance(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    source_hash: str,
    execution: dict[str, object],
) -> None:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"resume output does not exist: {output_dir}")
    try:
        metadata = json.loads((output_dir / "study_metadata.json").read_text())
        stored_config = yaml.safe_load((output_dir / "config.yaml").read_text())
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise RuntimeError(f"resume metadata is unreadable in {output_dir}: {error}") from error
    if not isinstance(metadata, dict) or not isinstance(stored_config, dict):
        raise RuntimeError("resume metadata and config must contain mappings")
    expected_config = _normalized_config(config)
    if stored_config != expected_config:
        raise RuntimeError("resume stored config differs from the requested config")
    if metadata.get("seeds") != list(config.seeds):
        raise RuntimeError("resume requested seeds differ from study metadata")
    if metadata.get("devices") != list(config.devices):
        raise RuntimeError("resume requested devices differ from study metadata")
    if metadata.get("source_tree_sha256") != source_hash:
        raise RuntimeError("resume source hash differs from the active source")
    if metadata.get("execution") != execution:
        raise RuntimeError("resume execution settings differ from the frozen study")


def _validated_existing_seeds(
    output_dir: Path,
    requested_seeds: tuple[int, ...],
    *,
    expected_source_hash: str,
    expected_config: dict[str, object],
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
            expected_config=expected_config,
        )
        completed.add(seed)
    return completed


def _normalized_config(config: ExperimentConfig) -> dict[str, object]:
    """Match the JSON/YAML representation persisted by artifact writers."""

    return json.loads(json.dumps(config.to_dict()))


def _run_and_write(config: ExperimentConfig, seed: int, device: str, output_dir: Path) -> str:
    try:
        result = run_seed(config, seed=seed, device=device)
        return str(write_seed_result(result, output_dir, config))
    finally:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def _build_seed_jobs(
    seeds: tuple[int, ...],
    devices: tuple[str, ...],
    workers_per_device: int,
) -> tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]]:
    """Assign each seed to a persistent worker pinned to one device."""

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


def _parse_seeds(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        result = default
    elif ":" in value:
        start, stop = (int(part) for part in value.split(":", maxsplit=1))
        result = tuple(range(start, stop))
    else:
        result = tuple(int(part) for part in value.split(",") if part)
    if not result:
        raise ValueError("at least one seed is required")
    if any(seed < 0 for seed in result):
        raise ValueError("seeds must be nonnegative")
    if len(set(result)) != len(result):
        raise ValueError("seeds must be unique")
    return result


if __name__ == "__main__":
    main()
