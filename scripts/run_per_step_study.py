"""Run prespecified per-step SC-PCP sensitivity studies on one or two GPUs.

This entry point runs one-factor sensitivities and the prespecified
``beta × eta`` performativity factorial.  Every condition is stored as a
normal ``run_per_step.py``-compatible artifact directory, so it supports the
same plotting and audit workflow as a single run.

Examples
--------
python scripts/run_per_step_study.py \
  --config configs/per_step_synthetic.yaml \
  --study feedback --values 0,0.25,0.5,0.75,1 \
  --seeds 0:200 --devices cuda:0,cuda:1 --workers-per-device 4

python scripts/run_per_step_study.py \
  --config configs/per_step_synthetic.yaml \
  --study sample_size --values 1000,2500,5000 \
  --seeds 0:200 --devices cuda:0,cuda:1 --workers-per-device 4

python scripts/run_per_step_study.py \
  --config configs/per_step_synthetic.yaml \
  --study factorial --feedback-values 0,0.5,1,2 \
  --policy-tilt-values 0.25,0.5,1,2 \
  --seeds 0:200 --devices cuda:0,cuda:1 --workers-per-device 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
import json
from multiprocessing import get_context
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import (
    mark_study_complete,
    mark_study_failed,
    source_tree_sha256,
    write_collection_status,
    write_seed_result,
    write_study_metadata,
)
from scpcp.config import ExperimentConfig
from scpcp.device import resolve_devices
from scpcp.experiment import run_seed


DEFAULT_VALUES = {
    "feedback": ("0.0", "0.5", "1.0", "1.5", "2.0"),
    "horizon": ("4", "8", "12", "24"),
    "policy_tilt": ("0.25", "0.5", "1.0", "2.0"),
    "sample_size": ("500", "1000", "2500", "5000", "10000"),
    "ratio_cap": ("1.1", "1.25", "2", "10"),
    "alpha": ("0.05", "0.10", "0.20"),
    "action_cost": ("0", "0.05", "0.10", "0.20"),
    "aci_gamma": ("0.005", "0.01", "0.05", "0.1"),
    "multidim_buffer": ("250", "500", "1000"),
    "mfcs_depth": ("1", "2", "3", "4"),
}
FACTORIAL_FEEDBACK_VALUES = ("0", "0.5", "1", "2")
FACTORIAL_POLICY_TILT_VALUES = ("0.25", "0.5", "1", "2")
FACTORIAL_STUDIES = ("factorial", "feedback_policy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a per-step SC-PCP sensitivity or factorial study")
    parser.add_argument("--config", type=Path, default=Path("configs/per_step_synthetic.yaml"))
    parser.add_argument("--study", choices=tuple(DEFAULT_VALUES) + FACTORIAL_STUDIES, required=True)
    parser.add_argument(
        "--values",
        nargs="*",
        default=None,
        help="space- or comma-separated values; study-specific defaults are used when omitted",
    )
    parser.add_argument(
        "--feedback-values",
        nargs="*",
        default=None,
        help="beta levels for --study factorial; defaults to 0,0.5,1,2",
    )
    parser.add_argument(
        "--policy-tilt-values",
        nargs="*",
        default=None,
        help="eta levels for --study factorial; defaults to 0.25,0.5,1,2",
    )
    parser.add_argument("--devices", default=None, help="comma-separated CUDA devices; defaults to both configured GPUs")
    parser.add_argument("--seeds", default=None, help="seed range such as 0:20 or comma-separated integers")
    parser.add_argument(
        "--workers-per-device",
        type=int,
        default=1,
        help="independent persistent seed workers per GPU",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/per_step_studies"))
    args = parser.parse_args()

    base = ExperimentConfig.from_yaml(args.config)
    devices = resolve_devices(args.devices or base.devices)
    if args.workers_per_device < 1:
        parser.error("--workers-per-device must be positive")
    seeds = parse_seeds(args.seeds, base.seeds)
    study = canonical_study_name(args.study)
    if study == "feedback_policy":
        if args.values:
            parser.error("--values is not used by the factorial; pass --feedback-values and --policy-tilt-values")
        feedback_values = parse_values(args.feedback_values, FACTORIAL_FEEDBACK_VALUES)
        policy_tilt_values = parse_values(args.policy_tilt_values, FACTORIAL_POLICY_TILT_VALUES)
        settings = build_factorial_settings(base, feedback_values, policy_tilt_values)
        study_axes = {
            "feedback_strength": list(feedback_values),
            "policy_tilt": list(policy_tilt_values),
        }
        manifest_values: list[str] | None = None
    else:
        if args.feedback_values or args.policy_tilt_values:
            parser.error("--feedback-values and --policy-tilt-values are only valid for --study factorial")
        values = parse_values(args.values, DEFAULT_VALUES[study])
        settings = build_settings(study, base, values)
        study_axes = {study: list(values)}
        manifest_values = list(values)

    collection_source_hash = source_tree_sha256()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study_root = args.output_dir / f"{timestamp}_{study}"
    study_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "study": study,
        "source_config": str(args.config),
        "values": manifest_values,
        "axes": study_axes,
        "settings": [_setting_manifest_row(label, setting) for label, setting in settings],
        "seeds": list(seeds),
        "devices": list(devices),
        "workers_per_device": args.workers_per_device,
        "source_tree_sha256": collection_source_hash,
    }
    (study_root / "study_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    expected_settings = tuple(label for label, _ in settings)
    completed_settings: list[str] = []
    write_collection_status(study_root, expected_settings, status="running")
    try:
        for label, setting in settings:
            _require_source_hash(collection_source_hash)
            output_dir = study_root / label
            config = setting.with_overrides(devices=devices, seeds=seeds, output_dir=output_dir)
            write_study_metadata(
                output_dir,
                config,
                execution={
                    "workers_per_device": args.workers_per_device,
                    "collection_source_tree_sha256": collection_source_hash,
                },
            )
            run_setting(config, output_dir, workers_per_device=args.workers_per_device)
            completed_settings.append(label)
            write_collection_status(
                study_root,
                expected_settings,
                status="running",
                completed_settings=tuple(completed_settings),
            )
        _require_source_hash(collection_source_hash)
        write_collection_status(
            study_root,
            expected_settings,
            status="complete",
            completed_settings=tuple(completed_settings),
        )
    except BaseException as error:
        write_collection_status(
            study_root,
            expected_settings,
            status="failed",
            completed_settings=tuple(completed_settings),
            error=error,
        )
        raise
    print(study_root)


def build_settings(
    study: str,
    base: ExperimentConfig,
    values: tuple[str, ...],
) -> tuple[tuple[str, ExperimentConfig], ...]:
    """Return readable labels and configurations for one prespecified factor."""

    if study == "feedback":
        if base.data.dataset not in {"synthetic", "tabular"}:
            raise ValueError("feedback sensitivity is defined only for synthetic or tabular environments")
        return tuple(
            (f"feedback_{value}", replace(base, synthetic=replace(base.synthetic, feedback_strength=float(value))))
            for value in values
        )
    if study == "horizon":
        return tuple((f"horizon_{value}", replace(base, horizon=int(value))) for value in values)
    if study == "policy_tilt":
        return tuple(
            (f"policy_tilt_{value}", replace(base, policy=replace(base.policy, tilt=float(value))))
            for value in values
        )
    if study == "sample_size":
        return tuple(
            (f"logged_n_{value}", replace(base, samples=replace(base.samples, logged=int(value))))
            for value in values
        )
    if study == "ratio_cap":
        settings = []
        for value in values:
            cap = float(value)
            # COT's deterministic state-action weight bound is rho_cap times
            # the policy-ratio cap.  Tighten B with the overlap condition so
            # this sensitivity changes both deployment overlap and the valid
            # finite-sample LCB, rather than carrying an unrelated base B.
            settings.append(
                (
                    f"ratio_cap_{value}",
                    replace(
                        base,
                        policy=replace(base.policy, policy_ratio_cap=cap),
                        cot=replace(base.cot, weight_cap=base.cot.rho_cap * cap),
                    ),
                )
            )
        return tuple(settings)
    if study == "alpha":
        return tuple(
            (f"alpha_{value}", replace(base, certification=replace(base.certification, alpha=float(value))))
            for value in values
        )
    if study == "action_cost":
        if base.data.dataset not in {"synthetic", "tabular"}:
            raise ValueError("action-cost sensitivity is defined only for synthetic or tabular environments")
        n_actions = len(base.policy.action_costs)
        if n_actions < 2:
            raise ValueError("action-cost sensitivity requires at least two actions")
        return tuple(
            (
                f"action_cost_{value}",
                replace(
                    base,
                    policy=replace(
                        base.policy,
                        action_costs=tuple(
                            float(value) * index / (n_actions - 1)
                            for index in range(n_actions)
                        ),
                    ),
                ),
            )
            for value in values
        )
    if study == "aci_gamma":
        return tuple(
            (f"aci_gamma_{value}", replace(base, baselines=replace(base.baselines, aci_gamma=float(value))))
            for value in values
        )
    if study == "multidim_buffer":
        return tuple(
            (
                f"multidim_buffer_{value}",
                replace(base, baselines=replace(base.baselines, multidim_buffer=int(value))),
            )
            for value in values
        )
    if study == "mfcs_depth":
        return tuple(
            (f"mfcs_depth_{value}", replace(base, baselines=replace(base.baselines, mfcs_depth=int(value))))
            for value in values
        )
    raise ValueError(f"unknown study {study}")


def build_factorial_settings(
    base: ExperimentConfig,
    feedback_values: tuple[str, ...],
    policy_tilt_values: tuple[str, ...],
) -> tuple[tuple[str, ExperimentConfig], ...]:
    """Build the fixed beta × eta matrix without changing any other factor."""

    if base.data.dataset not in {"synthetic", "tabular"}:
        raise ValueError("the feedback × policy-tilt factorial is defined only for synthetic or tabular environments")
    return tuple(
        (
            f"beta_{feedback}__eta_{tilt}",
            replace(
                base,
                synthetic=replace(base.synthetic, feedback_strength=float(feedback)),
                policy=replace(base.policy, tilt=float(tilt)),
            ),
        )
        for feedback in feedback_values
        for tilt in policy_tilt_values
    )


def canonical_study_name(study: str) -> str:
    return "feedback_policy" if study in FACTORIAL_STUDIES else study


def _setting_manifest_row(label: str, config: ExperimentConfig) -> dict[str, object]:
    return {
        "label": label,
        "feedback_strength": config.synthetic.feedback_strength,
        "policy_tilt": config.policy.tilt,
        "horizon": config.horizon,
        "logged_trajectories": config.samples.logged,
        "policy_ratio_cap": config.policy.policy_ratio_cap,
        "alpha": config.certification.alpha,
        "action_cost_max": max(config.policy.action_costs),
        "aci_gamma": config.baselines.aci_gamma,
        "multidim_buffer": config.baselines.multidim_buffer,
        "mfcs_depth": config.baselines.mfcs_depth,
    }


def run_setting(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    workers_per_device: int = 1,
) -> None:
    if workers_per_device < 1:
        raise ValueError("workers_per_device must be positive")
    worker_devices, jobs = _build_seed_jobs(
        config.seeds,
        config.devices,
        workers_per_device,
    )
    try:
        # Pin every persistent process to one physical GPU for the full setting.
        # Several slots may share a GPU, but workers never switch devices.
        with ExitStack() as stack:
            executors = [
                stack.enter_context(
                    ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
                )
                for _ in worker_devices
            ]
            futures = {
                executors[worker_index].submit(
                    run_and_write,
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


def run_and_write(config: ExperimentConfig, seed: int, device: str, output_dir: Path) -> str:
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


def _require_source_hash(expected: str) -> None:
    observed = source_tree_sha256()
    if observed != expected:
        raise RuntimeError(
            "active source changed during the collection: "
            f"expected {expected}, observed {observed}"
        )


def parse_values(raw_values: list[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not raw_values:
        return default
    values = tuple(value.strip() for item in raw_values for value in item.split(",") if value.strip())
    if not values:
        raise ValueError("at least one study value is required")
    return values


def parse_seeds(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
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
