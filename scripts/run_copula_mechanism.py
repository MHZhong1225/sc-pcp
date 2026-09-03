"""Run the isolated equal-marginal copula mechanism gate.

The command produces no paper-method rows.  It first establishes the causal
chain q -> policy -> observed regime -> normalized-max score law.  A downstream
six-method experiment may start only if :func:`require_six_method_gate` accepts
the completed artifact.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import source_tree_sha256  # noqa: E402
from scpcp.copula_benchmark import (  # noqa: E402
    CopulaKernel,
    CopulaMechanismResult,
    evaluate_mechanism_setting,
    make_copula_noise,
    prepare_source_reference,
)
from scpcp.copula_benchmark_config import (  # noqa: E402
    CopulaBenchmarkConfig,
)


CANONICAL_METHODS = ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")
REQUIRED_SEED_FILES = (
    "COMPLETE",
    "records.csv",
    "setting_diagnostics.json",
    "surfaces.npz",
    "metadata.json",
)
CONTROLLED_CONFIRM_BASES = tuple(range(91_000, 91_200, 10))
CONTROLLED_CONFIRM_SEEDS = frozenset(
    base + offset for base in CONTROLLED_CONFIRM_BASES for offset in (0, 1, 2)
)
FINITE_MDP_RESERVED_SEEDS = frozenset(range(52_000, 53_000))
SEED_ARTIFACT_NAME = re.compile(r"seed_(\d+)(?:\.json)?$")
SEED_ASSIGNMENT = re.compile(r"seed|rng", re.IGNORECASE)
ENGINEERING_CONTAMINATION = {
    "scientific_use": False,
    "inspected_seeds": [1, 93_000],
    "reason": (
        "exploratory engineering checked whether a stronger equal-marginal copula "
        "could create the requested Q90/coverage signal; all viewed cells are "
        "excluded from the frozen v1 confirmation"
    ),
    "viewed_cells": [
        {
            "label": "original-default sanity",
            "easy_correlation": 0.90,
            "beta": [-1.0, 0.0, 1.0],
            "maximum_policy_logit_shift": 1.50,
            "radii": [1.90],
            "observed_summary": (
                "absolute Q90 gaps about +/-0.014--0.016 and coverage gaps about "
                "+0.28pp/-0.32pp at seed 93000"
            ),
        },
        {
            "label": "excluded stronger-copula probes",
            "easy_correlation": [0.99, 0.999],
            "effective_beta_extremes": [-3.0, 3.0],
            "maximum_policy_logit_shift": [2.0, 2.3, 5.0],
            "radii": [1.75, 1.80, 1.90],
            "observed_summary": (
                "relative Q90 gaps ranged roughly 3--7%, same-radius coverage "
                "gaps roughly 1.1--2.4pp, and final-prefix ESS fractions roughly "
                "0.0001--0.08"
            ),
        },
    ],
    "formal_v1_action": (
        "retain the original pre-probe .90/0 copula, beta +/-1 grid, shift 1.5, "
        "and radius 1.90; move formal seeds to untouched 94000..94198 even"
    ),
}


@dataclass(frozen=True)
class CopulaSeedResult:
    seed: int
    device: str
    records: tuple[dict[str, Any], ...]
    setting_diagnostics: tuple[dict[str, Any], ...]
    surfaces: dict[str, np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "copula_mechanism.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--seeds",
        default=None,
        help="frozen-bank subset, e.g. 94000:94020:2 or 94000,94004",
    )
    parser.add_argument("--devices", default=None, help="comma-separated CUDA devices")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = CopulaBenchmarkConfig.from_yaml(args.config)
    seeds = base.seeds if args.seeds is None else parse_seed_subset(args.seeds, base.seeds)
    devices = (
        base.devices
        if args.devices is None
        else tuple(value.strip() for value in args.devices.split(",") if value.strip())
    )
    output_dir = base.output_dir if args.output_dir is None else args.output_dir.resolve()
    config = base.with_overrides(seeds=seeds, devices=devices, output_dir=output_dir)
    run_study(config, resume=args.resume)
    print(config.output_dir)


def parse_seed_subset(specification: str, frozen_bank: tuple[int, ...]) -> tuple[int, ...]:
    if ":" in specification:
        pieces = [int(value) for value in specification.split(":")]
        if len(pieces) not in (2, 3):
            raise ValueError("seed range must be start:stop or start:stop:step")
        seeds = tuple(range(*pieces))
    else:
        seeds = tuple(int(value.strip()) for value in specification.split(",") if value.strip())
    if not seeds or not set(seeds).issubset(frozen_bank):
        raise ValueError("requested seeds are not a nonempty subset of the frozen bank")
    if len(set(seeds)) != len(seeds):
        raise ValueError("requested seeds must be unique")
    return seeds


def run_study(config: CopulaBenchmarkConfig, *, resume: bool) -> None:
    """Run all mechanism cells and publish a fail-closed gate artifact."""

    config.validate()
    _assert_seed_namespaces_do_not_collide(config.seeds)
    output_dir = config.output_dir.resolve()
    config_hash = _json_sha256(config.to_dict())
    source_hash = source_tree_sha256()
    seed_to_device = _seed_device_mapping(config.seeds, config.devices)
    rng_audit = _audit_formal_rng_ids(config.seeds, output_dir=output_dir)
    manifest = _manifest(
        config,
        config_hash=config_hash,
        source_hash=source_hash,
        seed_to_device=seed_to_device,
        rng_audit=rng_audit,
    )
    if resume:
        completed = _validate_resume(output_dir, config, manifest)
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh copula output already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        _atomic_write_text(
            output_dir / "config.yaml",
            yaml.safe_dump(config.to_dict(), sort_keys=False),
        )
        _atomic_write_json(output_dir / "manifest.json", manifest)
        _write_status(output_dir, config.seeds, status="running")
        completed = set()

    pending = tuple(seed for seed in config.seeds if seed not in completed)
    if (output_dir / "COMPLETE").exists() and pending:
        raise RuntimeError("root COMPLETE exists while seed artifacts are missing")
    try:
        _run_pending(config, pending, output_dir, config_hash, source_hash)
        for seed in config.seeds:
            validate_seed_artifact(
                output_dir / f"seed_{seed:05d}",
                config=config,
                seed=seed,
                config_hash=config_hash,
                source_hash=source_hash,
            )
        if source_tree_sha256() != source_hash:
            raise RuntimeError("source tree changed while the mechanism study was running")
        if _audit_formal_rng_ids(config.seeds, output_dir=output_dir) != rng_audit:
            raise RuntimeError("formal RNG collision inventory changed while the study ran")
        records, diagnostics = _load_seed_outputs(output_dir, config.seeds)
        summary = summarize_mechanism(records, diagnostics, config)
        _atomic_write_json(output_dir / "summary.json", summary)
        summary_hash = _file_sha256(output_dir / "summary.json")
        gate = evaluate_mechanism_gate(
            records,
            diagnostics,
            config,
            config_hash=config_hash,
            source_hash=source_hash,
            summary_hash=summary_hash,
        )
        _atomic_write_json(output_dir / "gate.json", gate)
        _write_status(
            output_dir,
            config.seeds,
            status="complete",
            completed=set(config.seeds),
            gate_status=gate["status"],
        )
        complete = {
            "status": "complete",
            "config_sha256": config_hash,
            "manifest_sha256": _file_sha256(output_dir / "manifest.json"),
            "summary_sha256": summary_hash,
            "gate_sha256": _file_sha256(output_dir / "gate.json"),
        }
        _atomic_write_json(output_dir / "COMPLETE", complete)
        _fsync_directory(output_dir)
    except BaseException as error:
        _write_status(
            output_dir,
            config.seeds,
            status="failed",
            completed={
                seed
                for seed in config.seeds
                if (output_dir / f"seed_{seed:05d}" / "COMPLETE").is_file()
            },
            error=error,
        )
        raise


def run_seed(
    config: CopulaBenchmarkConfig,
    seed: int,
    *,
    device: str,
) -> CopulaSeedResult:
    """Run one paired factorial replicate without writing files."""

    noise = make_copula_noise(
        n=config.trajectories,
        horizon=config.horizon,
        seed=seed,
        device=device,
    )
    shape = (len(config.betas), len(config.kappas), len(config.radii), config.horizon)
    surface_names = (
        "policy_tv",
        "action_rate_gap",
        "hard_prevalence_gap",
        "q90_gap",
        "q90_relative_gap",
        "coverage_gap",
        "prefix_ess_fraction",
        "maximum_weight_share",
        "log_weight_span",
    )
    surfaces = {name: np.empty(shape, dtype=np.float64) for name in surface_names}
    records: list[dict[str, Any]] = []
    setting_diagnostics: list[dict[str, Any]] = []
    for beta_index, beta in enumerate(config.betas):
        kernel = CopulaKernel(config.dgp, beta=beta)
        source = prepare_source_reference(kernel, noise, alpha=config.alpha)
        for kappa_index, kappa in enumerate(config.kappas):
            for radius_index, radius in enumerate(config.radii):
                result = evaluate_mechanism_setting(
                    kernel,
                    source,
                    noise,
                    radius=radius,
                    kappa=kappa,
                    alpha=config.alpha,
                )
                records.extend(_stage_records(seed, result))
                setting_diagnostics.append(_setting_diagnostic(seed, result))
                values = _surface_values(result)
                for name in surface_names:
                    surfaces[name][beta_index, kappa_index, radius_index] = values[name]
    surfaces.update(
        {
            "betas": np.asarray(config.betas),
            "kappas": np.asarray(config.kappas),
            "radii": np.asarray(config.radii),
        }
    )
    return CopulaSeedResult(
        seed=seed,
        device=device,
        records=tuple(records),
        setting_diagnostics=tuple(setting_diagnostics),
        surfaces=surfaces,
    )


def summarize_mechanism(
    records: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: CopulaBenchmarkConfig,
) -> dict[str, Any]:
    metric_columns = [
        name
        for name in records.columns
        if name not in {"seed", "beta", "kappa", "radius", "stage", "kernel_fingerprint"}
    ]
    aggregates = (
        records.groupby(["beta", "kappa", "radius", "stage"], sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    audit_columns = [
        "source_maximum_absolute_mean",
        "target_maximum_absolute_mean",
        "source_maximum_variance_error",
        "target_maximum_variance_error",
        "source_maximum_correlation_error",
        "target_maximum_correlation_error",
    ]
    audit_summary = {name: float(diagnostics[name].max()) for name in audit_columns}
    return {
        "protocol": config.protocol,
        "estimand": "same_radius_source_target_normalized_max_coverage_gap",
        "mechanism_chain": [
            "radius",
            "nonanticipating_action_policy",
            "observed_next_regime",
            "cross_outcome_copula",
            "normalized_max_q90",
            "same_radius_coverage",
        ],
        "marginal_contract": (
            "each standardized coordinate is exactly N(0,1) conditional on every "
            "observed regime/action cell; only correlation changes"
        ),
        "n_seeds": len(config.seeds),
        "n_trajectories_per_seed": config.trajectories,
        "aggregates": aggregates.to_dict(orient="records"),
        "maximum_empirical_marginal_audit": audit_summary,
        "late_stage_start_zero_based": config.late_stage_start,
        "late_stage_paired_aggregates": _late_stage_aggregates(records, config),
    }


def evaluate_mechanism_gate(
    records: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: CopulaBenchmarkConfig,
    *,
    config_hash: str,
    source_hash: str,
    summary_hash: str,
) -> dict[str, Any]:
    """Apply frozen placebo, direction, marginal, and overlap checks."""

    gate = config.gate
    maximum_kappa = max(config.kappas)
    negative_beta = min(config.betas)
    positive_beta = max(config.betas)
    primary = records[
        (records["kappa"] == maximum_kappa)
        & (records["radius"] == gate.primary_radius)
        & (records["stage"] >= config.late_stage_start)
    ]
    negative = primary[primary["beta"] == negative_beta]
    positive = primary[primary["beta"] == positive_beta]
    beta_zero = records[records["beta"] == 0.0]
    kappa_zero = records[records["kappa"] == 0.0]
    responsive_beta_zero = beta_zero[
        (beta_zero["kappa"] == maximum_kappa)
        & (beta_zero["radius"] == gate.primary_radius)
    ]

    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, observed: float, operator: str, threshold: float) -> None:
        passed = {
            "<=": observed <= threshold,
            ">=": observed >= threshold,
            ">": observed > threshold,
        }[operator]
        checks[name] = {
            "observed": observed,
            "operator": operator,
            "threshold": threshold,
            "passed": bool(passed),
        }

    add(
        "kappa_zero_policy_placebo",
        _maximum_absolute(kappa_zero, ("policy_tv",)),
        "<=",
        gate.maximum_placebo_policy_tv,
    )
    add(
        "kappa_zero_hard_placebo",
        _maximum_absolute(kappa_zero, ("hard_prevalence_gap",)),
        "<=",
        gate.maximum_placebo_hard_prevalence_gap,
    )
    add(
        "kappa_zero_q90_placebo",
        _maximum_absolute(kappa_zero, ("q90_relative_gap",)),
        "<=",
        gate.maximum_placebo_relative_q90_gap,
    )
    add(
        "kappa_zero_coverage_placebo",
        _maximum_absolute(kappa_zero, ("coverage_gap",)),
        "<=",
        gate.maximum_placebo_coverage_gap,
    )
    add(
        "beta_zero_hard_placebo",
        _maximum_absolute(beta_zero, ("hard_prevalence_gap",)),
        "<=",
        gate.maximum_placebo_hard_prevalence_gap,
    )
    add(
        "beta_zero_q90_placebo",
        _maximum_absolute(beta_zero, ("q90_relative_gap",)),
        "<=",
        gate.maximum_placebo_relative_q90_gap,
    )
    add(
        "beta_zero_coverage_placebo",
        _maximum_absolute(beta_zero, ("coverage_gap",)),
        "<=",
        gate.maximum_placebo_coverage_gap,
    )
    add(
        "beta_zero_policy_changes",
        float(responsive_beta_zero["policy_tv"].mean()),
        ">=",
        gate.minimum_policy_tv,
    )
    add(
        "positive_beta_hard_shift",
        float(positive["hard_prevalence_gap"].mean()),
        ">=",
        gate.minimum_hard_prevalence_shift,
    )
    add(
        "negative_beta_hard_shift",
        float(-negative["hard_prevalence_gap"].mean()),
        ">=",
        gate.minimum_hard_prevalence_shift,
    )
    add(
        "positive_beta_relative_q90_shift",
        float(positive["q90_relative_gap"].mean()),
        ">=",
        gate.minimum_relative_q90_shift,
    )
    add(
        "negative_beta_relative_q90_shift",
        float(-negative["q90_relative_gap"].mean()),
        ">=",
        gate.minimum_relative_q90_shift,
    )
    add(
        "positive_beta_coverage_loss",
        float(-positive["coverage_gap"].mean()),
        ">=",
        gate.minimum_coverage_shift,
    )
    add(
        "negative_beta_coverage_gain",
        float(negative["coverage_gap"].mean()),
        ">=",
        gate.minimum_coverage_shift,
    )
    add(
        "primary_late_minimum_prefix_ess",
        float(primary["prefix_ess_fraction"].min()),
        ">=",
        gate.minimum_prefix_ess_fraction,
    )
    add(
        "primary_late_maximum_incremental_ratio",
        float(primary["maximum_incremental_ratio"].max()),
        "<=",
        gate.maximum_incremental_ratio,
    )
    add(
        "primary_late_maximum_normalized_weight_share",
        float(primary["maximum_weight_share"].max()),
        "<=",
        gate.maximum_normalized_weight_share,
    )
    add(
        "equal_marginal_mean",
        float(
            diagnostics[
                ["source_maximum_absolute_mean", "target_maximum_absolute_mean"]
            ].to_numpy().max()
        ),
        "<=",
        gate.maximum_marginal_mean_error,
    )
    add(
        "equal_marginal_variance",
        float(
            diagnostics[
                ["source_maximum_variance_error", "target_maximum_variance_error"]
            ].to_numpy().max()
        ),
        "<=",
        gate.maximum_marginal_variance_error,
    )
    add(
        "declared_copula_correlation",
        float(
            diagnostics[
                ["source_maximum_correlation_error", "target_maximum_correlation_error"]
            ].to_numpy().max()
        ),
        "<=",
        gate.maximum_correlation_error,
    )
    directional_intervals = {
        "positive_beta_relative_q90": _paired_late_interval(
            positive, "q90_relative_gap", direction=1.0, confidence=gate.paired_confidence_level
        ),
        "negative_beta_relative_q90": _paired_late_interval(
            negative, "q90_relative_gap", direction=-1.0, confidence=gate.paired_confidence_level
        ),
        "positive_beta_coverage_loss": _paired_late_interval(
            positive, "coverage_gap", direction=-1.0, confidence=gate.paired_confidence_level
        ),
        "negative_beta_coverage_gain": _paired_late_interval(
            negative, "coverage_gap", direction=1.0, confidence=gate.paired_confidence_level
        ),
    }
    for name, interval in directional_intervals.items():
        add(
            f"{name}_paired_ci_excludes_zero",
            interval["lower"],
            ">",
            0.0,
        )
    passed = all(check["passed"] for check in checks.values())
    return {
        "status": "pass" if passed else "fail",
        "checks": checks,
        "direction_aligned_seed_paired_late_stage_intervals": directional_intervals,
        "config_sha256": config_hash,
        "source_tree_sha256": source_hash,
        "summary_sha256": summary_hash,
        "optional_six_method_stage": {
            "authorized": passed,
            "canonical_methods": list(CANONICAL_METHODS),
            "reason": (
                "all predeclared mechanism checks passed"
                if passed
                else "mechanism gate failed; do not launch method comparisons"
            ),
        },
    }


def require_six_method_gate(root: str | Path) -> dict[str, Any]:
    """Fail closed unless a complete mechanism run authorizes all six methods."""

    output_dir = Path(root)
    complete = _read_json(output_dir / "COMPLETE")
    manifest = _read_json(output_dir / "manifest.json")
    gate = _read_json(output_dir / "gate.json")
    if complete.get("status") != "complete":
        raise RuntimeError("copula mechanism artifact is not complete")
    expected_hashes = {
        "manifest_sha256": output_dir / "manifest.json",
        "summary_sha256": output_dir / "summary.json",
        "gate_sha256": output_dir / "gate.json",
    }
    for field, path in expected_hashes.items():
        if complete.get(field) != _file_sha256(path):
            raise RuntimeError(f"copula mechanism {field} does not match COMPLETE")
    if gate.get("config_sha256") != manifest.get("config_sha256"):
        raise RuntimeError("copula mechanism gate and manifest config hashes differ")
    if gate.get("summary_sha256") != complete.get("summary_sha256"):
        raise RuntimeError("copula mechanism gate is not bound to the completed summary")
    stage = gate.get("optional_six_method_stage")
    if (
        gate.get("status") != "pass"
        or not isinstance(stage, dict)
        or stage.get("authorized") is not True
        or stage.get("canonical_methods") != list(CANONICAL_METHODS)
    ):
        raise RuntimeError("six-method stage is blocked by the copula mechanism gate")
    return gate


def _run_pending(
    config: CopulaBenchmarkConfig,
    pending: tuple[int, ...],
    output_dir: Path,
    config_hash: str,
    source_hash: str,
) -> None:
    if not pending:
        return
    grouped = _pending_seed_groups(config.seeds, config.devices, pending)
    with ProcessPoolExecutor(
        max_workers=len(config.devices),
        mp_context=get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(_run_seed_group, config, group, device): device
            for device, group in grouped.items()
            if group
        }
        for future in as_completed(futures):
            for result in future.result():
                write_seed_artifact(
                    result,
                    output_dir,
                    config=config,
                    config_hash=config_hash,
                    source_hash=source_hash,
                )
                print(f"completed copula seed {result.seed}", flush=True)


def _run_seed_group(
    config: CopulaBenchmarkConfig,
    seeds: tuple[int, ...],
    device: str,
) -> list[CopulaSeedResult]:
    torch.cuda.set_device(torch.device(device))
    results = []
    for seed in seeds:
        results.append(run_seed(config, seed, device=device))
        torch.cuda.empty_cache()
    return results


def write_seed_artifact(
    result: CopulaSeedResult,
    output_dir: Path,
    *,
    config: CopulaBenchmarkConfig,
    config_hash: str,
    source_hash: str,
) -> Path:
    """Publish one complete seed directory with an atomic rename."""

    expected_device = _seed_device_mapping(config.seeds, config.devices)[result.seed]
    if result.device != expected_device:
        raise RuntimeError(
            f"seed {result.seed} ran on {result.device}, expected {expected_device}"
        )
    destination = output_dir / f"seed_{result.seed:05d}"
    if destination.exists():
        raise FileExistsError(f"seed artifact already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=output_dir))
    try:
        pd.DataFrame(result.records).to_csv(temporary / "records.csv", index=False)
        _write_text(
            temporary / "setting_diagnostics.json",
            json.dumps(result.setting_diagnostics, sort_keys=True, indent=2) + "\n",
        )
        np.savez_compressed(temporary / "surfaces.npz", **result.surfaces)
        payload_files = ("records.csv", "setting_diagnostics.json", "surfaces.npz")
        metadata = {
            "protocol": config.protocol,
            "seed": result.seed,
            "device": result.device,
            "seed_namespace": config.seed_namespace,
            "config_sha256": config_hash,
            "source_tree_sha256": source_hash,
            "files": {name: _file_sha256(temporary / name) for name in payload_files},
        }
        _write_text(
            temporary / "metadata.json",
            json.dumps(metadata, sort_keys=True, indent=2) + "\n",
        )
        complete = {
            "seed": result.seed,
            "status": "complete",
            "metadata_sha256": _file_sha256(temporary / "metadata.json"),
        }
        _write_text(
            temporary / "COMPLETE",
            json.dumps(complete, sort_keys=True) + "\n",
        )
        for name in REQUIRED_SEED_FILES:
            _fsync_file(temporary / name)
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def validate_seed_artifact(
    path: Path,
    *,
    config: CopulaBenchmarkConfig,
    seed: int,
    config_hash: str,
    source_hash: str,
) -> None:
    missing = [name for name in REQUIRED_SEED_FILES if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"partial atomic seed directory {path}: missing {missing}")
    complete = _read_json(path / "COMPLETE")
    metadata = _read_json(path / "metadata.json")
    if complete != {
        "seed": seed,
        "status": "complete",
        "metadata_sha256": _file_sha256(path / "metadata.json"),
    }:
        raise RuntimeError(f"seed {seed} COMPLETE marker is invalid")
    expected_metadata = {
        "protocol": config.protocol,
        "seed": seed,
        "device": _seed_device_mapping(config.seeds, config.devices)[seed],
        "seed_namespace": config.seed_namespace,
        "config_sha256": config_hash,
        "source_tree_sha256": source_hash,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise RuntimeError(f"seed {seed} metadata field {field} is mismatched")
    for name in ("records.csv", "setting_diagnostics.json", "surfaces.npz"):
        if metadata.get("files", {}).get(name) != _file_sha256(path / name):
            raise RuntimeError(f"seed {seed} payload hash differs for {name}")
    records = pd.read_csv(path / "records.csv")
    expected_rows = (
        len(config.betas) * len(config.kappas) * len(config.radii) * config.horizon
    )
    if len(records) != expected_rows or records.isna().any().any():
        raise RuntimeError(f"seed {seed} records have the wrong shape or missing values")
    if set(records["seed"]) != {seed}:
        raise RuntimeError(f"seed {seed} records contain another seed ID")
    diagnostics = _read_json_list(path / "setting_diagnostics.json")
    expected_settings = len(config.betas) * len(config.kappas) * len(config.radii)
    if len(diagnostics) != expected_settings:
        raise RuntimeError(f"seed {seed} setting diagnostics have the wrong length")
    with np.load(path / "surfaces.npz") as surfaces:
        expected_shape = (
            len(config.betas),
            len(config.kappas),
            len(config.radii),
            config.horizon,
        )
        for name in (
            "policy_tv",
            "action_rate_gap",
            "hard_prevalence_gap",
            "q90_gap",
            "q90_relative_gap",
            "coverage_gap",
            "prefix_ess_fraction",
            "maximum_weight_share",
            "log_weight_span",
        ):
            if surfaces[name].shape != expected_shape or not np.isfinite(surfaces[name]).all():
                raise RuntimeError(f"seed {seed} surface {name} is malformed")


def _validate_resume(
    output_dir: Path,
    config: CopulaBenchmarkConfig,
    manifest: dict[str, Any],
) -> set[int]:
    if not output_dir.is_dir():
        raise FileNotFoundError("resume requires an existing output directory")
    if _read_json(output_dir / "manifest.json") != manifest:
        raise RuntimeError("resume manifest differs from the active frozen protocol")
    raw_config = yaml.safe_load((output_dir / "config.yaml").read_text())
    if raw_config != config.to_dict():
        raise RuntimeError("resume config.yaml differs from the active config")
    completed: set[int] = set()
    for seed in config.seeds:
        path = output_dir / f"seed_{seed:05d}"
        if not path.exists():
            continue
        validate_seed_artifact(
            path,
            config=config,
            seed=seed,
            config_hash=manifest["config_sha256"],
            source_hash=manifest["source_tree_sha256"],
        )
        completed.add(seed)
    return completed


def _load_seed_outputs(
    output_dir: Path,
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.concat(
        [pd.read_csv(output_dir / f"seed_{seed:05d}" / "records.csv") for seed in seeds],
        ignore_index=True,
    )
    diagnostics: list[dict[str, Any]] = []
    for seed in seeds:
        diagnostics.extend(
            _read_json_list(output_dir / f"seed_{seed:05d}" / "setting_diagnostics.json")
        )
    return records, pd.DataFrame(diagnostics)


def _stage_records(seed: int, result: CopulaMechanismResult) -> list[dict[str, Any]]:
    values = _surface_values(result)
    records = []
    for stage in range(len(result.source_q90)):
        records.append(
            {
                "seed": seed,
                "beta": result.beta,
                "kappa": result.kappa,
                "radius": result.radius,
                "stage": stage,
                "kernel_fingerprint": result.kernel_fingerprint,
                "policy_tv": float(result.policy_tv_on_source[stage].item()),
                "source_action_rate": float(result.source_action_rate[stage].item()),
                "target_action_rate": float(result.target_action_rate[stage].item()),
                "action_rate_gap": float(values["action_rate_gap"][stage]),
                "source_hard_prevalence": float(
                    result.source_hard_prevalence[stage].item()
                ),
                "target_hard_prevalence": float(
                    result.target_hard_prevalence[stage].item()
                ),
                "hard_prevalence_gap": float(values["hard_prevalence_gap"][stage]),
                "source_q90": float(result.source_q90[stage].item()),
                "target_q90": float(result.target_q90[stage].item()),
                "q90_gap": float(values["q90_gap"][stage]),
                "q90_relative_gap": float(values["q90_relative_gap"][stage]),
                "source_same_radius_coverage": float(
                    result.source_coverage[stage].item()
                ),
                "target_same_radius_coverage": float(
                    result.target_coverage[stage].item()
                ),
                "coverage_gap": float(values["coverage_gap"][stage]),
                "prefix_ess_fraction": float(
                    result.overlap.ess_fraction[stage].item()
                ),
                "maximum_weight_share": float(
                    result.overlap.maximum_normalized_weight_share[stage].item()
                ),
                "log_weight_span": float(result.overlap.log_weight_span[stage].item()),
                "minimum_incremental_ratio": float(
                    result.overlap.minimum_incremental_ratio[stage].item()
                ),
                "maximum_incremental_ratio": float(
                    result.overlap.maximum_incremental_ratio[stage].item()
                ),
            }
        )
    return records


def _setting_diagnostic(seed: int, result: CopulaMechanismResult) -> dict[str, Any]:
    return {
        "seed": seed,
        "beta": result.beta,
        "kappa": result.kappa,
        "radius": result.radius,
        "kernel_fingerprint": result.kernel_fingerprint,
        "source_maximum_absolute_mean": result.source_audit.maximum_absolute_mean,
        "target_maximum_absolute_mean": result.target_audit.maximum_absolute_mean,
        "source_maximum_variance_error": result.source_audit.maximum_variance_error,
        "target_maximum_variance_error": result.target_audit.maximum_variance_error,
        "source_maximum_correlation_error": result.source_audit.maximum_correlation_error,
        "target_maximum_correlation_error": result.target_audit.maximum_correlation_error,
        "source_minimum_regime_action_count": result.source_audit.minimum_regime_action_count,
        "target_minimum_regime_action_count": result.target_audit.minimum_regime_action_count,
    }


def _surface_values(result: CopulaMechanismResult) -> dict[str, np.ndarray]:
    to_numpy = lambda value: value.detach().cpu().numpy()  # noqa: E731
    return {
        "policy_tv": to_numpy(result.policy_tv_on_source),
        "action_rate_gap": to_numpy(result.target_action_rate - result.source_action_rate),
        "hard_prevalence_gap": to_numpy(
            result.target_hard_prevalence - result.source_hard_prevalence
        ),
        "q90_gap": to_numpy(result.target_q90 - result.source_q90),
        "q90_relative_gap": to_numpy(result.target_q90 / result.source_q90 - 1.0),
        "coverage_gap": to_numpy(result.target_coverage - result.source_coverage),
        "prefix_ess_fraction": to_numpy(result.overlap.ess_fraction),
        "maximum_weight_share": to_numpy(
            result.overlap.maximum_normalized_weight_share
        ),
        "log_weight_span": to_numpy(result.overlap.log_weight_span),
    }


def _manifest(
    config: CopulaBenchmarkConfig,
    *,
    config_hash: str,
    source_hash: str,
    seed_to_device: dict[int, str],
    rng_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": config.protocol,
        "study_stage": "mechanism_gate_only",
        "config_sha256": config_hash,
        "source_tree_sha256": source_hash,
        "seed_namespace": config.seed_namespace,
        "seeds": list(config.seeds),
        "seed_to_device": {
            str(seed): seed_to_device[seed] for seed in config.seeds
        },
        "randomness": "one torch float64 CRN bundle per seed, paired across all cells",
        "kernel_contract": (
            "within each beta cell source and target share one kernel; radius and "
            "kappa occur only in the nonanticipating action policy"
        ),
        "outcome_contract": (
            "both standardized coordinates are exactly N(0,1) in every observed "
            "regime/action cell; only cross-coordinate correlation changes"
        ),
        "formal_rng_collision_audit": rng_audit,
        "engineering_contamination": ENGINEERING_CONTAMINATION,
        "optional_six_method_stage": {
            "status": "blocked_until_gate_passes",
            "canonical_methods": list(CANONICAL_METHODS),
        },
    }


def _seed_device_mapping(
    seeds: tuple[int, ...],
    devices: tuple[str, ...],
) -> dict[int, str]:
    """Bind each seed to its global config index, independent of resume state."""

    return {
        seed: devices[index % len(devices)]
        for index, seed in enumerate(seeds)
    }


def _pending_seed_groups(
    seeds: tuple[int, ...],
    devices: tuple[str, ...],
    pending: tuple[int, ...],
) -> dict[str, tuple[int, ...]]:
    mapping = _seed_device_mapping(seeds, devices)
    return {
        device: tuple(seed for seed in pending if mapping[seed] == device)
        for device in devices
    }


def _assert_seed_namespaces_do_not_collide(seeds: tuple[int, ...]) -> None:
    selected = set(seeds)
    if selected & CONTROLLED_CONFIRM_SEEDS:
        raise ValueError("copula seeds collide with controlled six-method confirmation")
    if selected & FINITE_MDP_RESERVED_SEEDS:
        raise ValueError("copula seeds collide with the exact finite-MDP namespace")


def _audit_formal_rng_ids(
    seeds: tuple[int, ...],
    *,
    output_dir: Path,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Enumerate formal base RNG IDs and reject prior artifact/declaration use."""

    artifact_root = ROOT / "results" if artifact_root is None else artifact_root
    source_root = ROOT if source_root is None else source_root
    formal_mapping = {f"base_{seed}/paired_copula_crn": seed for seed in seeds}
    formal_ids = set(formal_mapping.values())
    if len(formal_ids) != len(formal_mapping):
        raise RuntimeError("formal copula RNG IDs are not unique")

    artifact_ids = _artifact_rng_ids(artifact_root, excluded_root=output_dir)
    source_ids = _source_declared_rng_ids(
        source_root,
        excluded_paths={
            Path(__file__).resolve(),
            (ROOT / "src" / "scpcp" / "copula_benchmark_config.py").resolve(),
            (ROOT / "configs" / "copula_mechanism.yaml").resolve(),
        },
        allowed_cross_reservation_ids=formal_ids,
    )
    coordinated_external_ids = set(CONTROLLED_CONFIRM_SEEDS) | set(
        FINITE_MDP_RESERVED_SEEDS
    )
    declared_ids = source_ids | coordinated_external_ids
    prior_ids = artifact_ids | declared_ids
    collisions = {
        label: rng_id
        for label, rng_id in formal_mapping.items()
        if rng_id in prior_ids
    }
    audit = {
        "status": "passed_before_launch" if not collisions else "collision",
        "formal_rng_id_count": len(formal_ids),
        "formal_rng_id_sha256": _integer_set_sha256(formal_ids),
        "formal_rng_mapping": formal_mapping,
        "formal_rng_mapping_sha256": _json_sha256(formal_mapping),
        "artifact_rng_id_count": len(artifact_ids),
        "artifact_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "source_declared_rng_id_count": len(source_ids),
        "source_declared_rng_id_sha256": _integer_set_sha256(source_ids),
        "coordinated_external_rng_id_count": len(coordinated_external_ids),
        "coordinated_external_rng_id_sha256": _integer_set_sha256(
            coordinated_external_ids
        ),
        "prior_declared_or_artifact_rng_id_count": len(prior_ids),
        "prior_declared_or_artifact_rng_id_sha256": _integer_set_sha256(prior_ids),
        "collision_count": len(collisions),
        "collisions": collisions,
        "excluded_output": str(output_dir.resolve()),
        "cross_reservation_acknowledged": {
            "source": "src/scpcp/exact_finite_mdp.py",
            "namespace": "orthogonal_copula_formal",
            "rng_id_sha256": _integer_set_sha256(formal_ids),
        },
    }
    audit["audit_sha256"] = _json_sha256(audit)
    if collisions:
        raise RuntimeError(
            "formal copula RNG IDs collide with prior declaration/artifact use: "
            f"{collisions}"
        )
    return audit


def _artifact_rng_ids(root: Path, *, excluded_root: Path) -> set[int]:
    values: set[int] = set()
    if not root.exists():
        return values
    excluded = excluded_root.resolve()
    for path in root.rglob("*"):
        if _is_relative_to(path.resolve(), excluded):
            continue
        match = SEED_ARTIFACT_NAME.fullmatch(path.name)
        if match:
            values.add(int(match.group(1)))
        if not path.is_file() or path.name not in {
            "metadata.json",
            "study_metadata.json",
            "manifest.json",
            "summary.json",
            "config.yaml",
        }:
            continue
        try:
            payload = (
                yaml.safe_load(path.read_text())
                if path.suffix in {".yaml", ".yml"}
                else json.loads(path.read_text())
            )
        except (OSError, json.JSONDecodeError, yaml.YAMLError):
            continue
        _collect_named_rng_values(payload, values)
    return values


def _source_declared_rng_ids(
    root: Path,
    *,
    excluded_paths: set[Path],
    allowed_cross_reservation_ids: Iterable[int] = (),
) -> set[int]:
    values: set[int] = set()
    for directory in ("scripts", "src", "tools", "configs"):
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.resolve() in excluded_paths or not path.is_file():
                continue
            path_values: set[int] = set()
            if path.suffix == ".py":
                _collect_python_rng_assignments(path, path_values)
            elif path.suffix in {".yaml", ".yml"}:
                try:
                    _collect_named_rng_values(yaml.safe_load(path.read_text()), path_values)
                except (OSError, yaml.YAMLError):
                    continue
            if path.resolve() == (
                root / "src" / "scpcp" / "exact_finite_mdp.py"
            ).resolve():
                path_values.difference_update(allowed_cross_reservation_ids)
            values.update(path_values)
    return values


def _collect_python_rng_assignments(path: Path, values: set[int]) -> None:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [name for target in node.targets for name in _target_names(target)]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            names = list(_target_names(node.target))
            value = node.value
        else:
            continue
        if value is None or not any(SEED_ASSIGNMENT.search(name) for name in names):
            continue
        evaluated = _literal_rng_expression(value)
        if evaluated is not None:
            values.update(evaluated)


def _target_names(target: ast.expr) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_names(element)


def _literal_rng_expression(node: ast.expr) -> set[int] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return {node.value}
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        parts = [_literal_rng_expression(element) for element in node.elts]
        if any(part is None for part in parts):
            return None
        return set().union(*(part or set() for part in parts))
    if isinstance(node, ast.Dict):
        parts = [_literal_rng_expression(value) for value in node.values]
        return set().union(*(part or set() for part in parts))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "range":
            arguments = [_literal_integer(argument) for argument in node.args]
            if any(argument is None for argument in arguments):
                return None
            return set(range(*(int(argument) for argument in arguments)))
        if node.func.id in {"tuple", "list", "set", "frozenset"} and len(node.args) == 1:
            return _literal_rng_expression(node.args[0])
    return None


def _literal_integer(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _literal_integer(node.operand)
        return None if operand is None else -operand
    return None


def _collect_named_rng_values(
    value: object,
    output: set[int],
    key_path: str = "",
) -> None:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            child_path = f"{key_path}.{child_key}" if key_path else str(child_key)
            _collect_named_rng_values(child_value, output, child_path)
        return
    if isinstance(value, list):
        for child in value:
            _collect_named_rng_values(child, output, key_path)
        return
    if SEED_ASSIGNMENT.search(key_path) and isinstance(value, int) and not isinstance(value, bool):
        output.add(value)


def _integer_set_sha256(values: Iterable[int]) -> str:
    encoded = json.dumps(sorted(set(values)), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _maximum_absolute(frame: pd.DataFrame, columns: tuple[str, ...]) -> float:
    if frame.empty:
        raise RuntimeError("mechanism gate selected an empty factorial cell")
    return float(np.abs(frame[list(columns)].to_numpy()).max())


def _late_stage_aggregates(
    records: pd.DataFrame,
    config: CopulaBenchmarkConfig,
) -> list[dict[str, Any]]:
    late = records[records["stage"] >= config.late_stage_start]
    metrics = (
        "policy_tv",
        "hard_prevalence_gap",
        "q90_relative_gap",
        "coverage_gap",
        "prefix_ess_fraction",
    )
    per_seed = (
        late.groupby(["seed", "beta", "kappa", "radius"], sort=True)[list(metrics)]
        .mean()
        .reset_index()
    )
    output = []
    for keys, frame in per_seed.groupby(["beta", "kappa", "radius"], sort=True):
        row: dict[str, Any] = {
            "beta": float(keys[0]),
            "kappa": float(keys[1]),
            "radius": float(keys[2]),
            "n_seeds": len(frame),
        }
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"se_{metric}"] = (
                float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            )
        output.append(row)
    return output


def _paired_late_interval(
    frame: pd.DataFrame,
    metric: str,
    *,
    direction: float,
    confidence: float,
) -> dict[str, Any]:
    values = direction * frame.groupby("seed")[metric].mean().to_numpy(dtype=float)
    if len(values) < 2:
        return {
            "n_seeds": len(values),
            "mean": float(values.mean()),
            "lower": float("-inf"),
            "upper": float("inf"),
            "confidence": confidence,
        }
    standard_error = values.std(ddof=1) / np.sqrt(len(values))
    critical = stats.t.ppf(0.5 + confidence / 2.0, df=len(values) - 1)
    half_width = critical * standard_error
    return {
        "n_seeds": len(values),
        "mean": float(values.mean()),
        "lower": float(values.mean() - half_width),
        "upper": float(values.mean() + half_width),
        "confidence": confidence,
    }


def _write_status(
    output_dir: Path,
    seeds: tuple[int, ...],
    *,
    status: str,
    completed: set[int] | None = None,
    error: BaseException | None = None,
    gate_status: str | None = None,
) -> None:
    completed = set() if completed is None else completed
    _atomic_write_json(
        output_dir / "study_status.json",
        {
            "status": status,
            "expected_seeds": list(seeds),
            "completed_seeds": sorted(completed),
            "missing_seeds": sorted(set(seeds) - completed),
            "gate_status": gate_status,
            "error": None if error is None else f"{type(error).__name__}: {error}",
        },
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}-",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unreadable JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return value


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unreadable JSON artifact: {path}") from error
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise RuntimeError(f"JSON artifact must be a list of objects: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
