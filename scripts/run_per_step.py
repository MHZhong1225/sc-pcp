"""Run the per-step SC-PCP experiment on one or two GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import (
    mark_study_complete,
    mark_study_failed,
    write_seed_result,
    write_study_metadata,
)
from scpcp.config import ExperimentConfig
from scpcp.device import resolve_devices
from scpcp.experiment import run_seed


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
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    devices = resolve_devices(args.devices or config.devices)
    if args.workers_per_device < 1:
        parser.error("--workers-per-device must be positive")
    seeds = _parse_seeds(args.seeds, config.seeds)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or config.output_dir / f"{timestamp}_{config.data.dataset}"
    config = config.with_overrides(devices=devices, seeds=seeds, output_dir=output_dir)
    run_config(config, output_dir, workers_per_device=args.workers_per_device)
    print(output_dir)


def run_config(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    workers_per_device: int,
) -> None:
    """Run one frozen config for use by the paper-suite scheduler."""

    write_study_metadata(
        output_dir,
        config,
        execution={"workers_per_device": workers_per_device},
    )
    worker_devices, jobs = _build_seed_jobs(
        config.seeds,
        config.devices,
        workers_per_device,
    )
    try:
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
    except BaseException as error:
        mark_study_failed(output_dir, config.seeds, error)
        raise
    mark_study_complete(output_dir, config.seeds)


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
