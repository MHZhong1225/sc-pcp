"""Run the isolated finite-MDP propensity-robustness study."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import git_revision, source_tree_sha256
from scpcp.propensity_robustness import (
    APPENDIX_LAYER,
    EXTERNAL_SEED_RESERVATIONS,
    PRIMARY_LAYER,
    PROPENSITY_ARMS,
    PROTOCOL,
    SEED_NAMESPACE,
    PropensityRobustnessConfig,
    propensity_seed_collision_audit,
    run_propensity_robustness,
    smoke_config,
)


DEFAULT_OUTPUT = ROOT / "results" / "work" / "propensity_robustness_v1"
PARENT_SNAPSHOT_MANIFEST = (
    ROOT / "results" / "work" / "formal_source_snapshot_7665dfbe_20260825.manifest.json"
)
PARENT_SNAPSHOT_SIDECAR = (
    ROOT / "results" / "work" / "formal_source_snapshot_7665dfbe_20260825.manifest.sha256"
)
PARENT_SNAPSHOT_ARCHIVE = (
    ROOT / "results" / "work" / "formal_source_snapshot_7665dfbe_20260825.tar.gz"
)
EXPECTED_PARENT_MANIFEST_SHA256 = (
    "e6a1bba7f3be47d39357f212824e7720262e7d5212a14628e3b8981088c64e24"
)
EXPECTED_PARENT_ARCHIVE_SHA256 = (
    "2116b9929240d8a25092f3c9015c362957bc963532930e5e720dc7ec78b2ea0b"
)
EXPECTED_PARENT_SOURCE_TREE_SHA256 = (
    "7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643"
)
EXPECTED_PARENT_ARCHIVE_BYTES = 2_036_776
RNG_KEY = re.compile(r"seed|rng", re.IGNORECASE)
RESERVATION_KEY = re.compile(r"reserv|namespace", re.IGNORECASE)
ARTIFACT_ID = re.compile(r"(?:seed|problem)_(\d+)(?:\.json)?$")
PAYLOAD_NAMES = (
    "config.json",
    "metadata.json",
    "summary.json",
    "arrays.npz",
    "nuisance_diagnostics.csv",
    "primary_transport_only.csv",
    "appendix_end_to_end.csv",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paired M3 logging-propensity robustness study"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only a tiny non-scientific contract check",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = smoke_config() if args.smoke else PropensityRobustnessConfig()
    if not args.smoke:
        config.assert_frozen_protocol()
    run_study(config, args.output, resume=args.resume)
    print(args.output)


def run_study(
    config: PropensityRobustnessConfig,
    output_dir: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Run once, or fail-closed while reusing an immutable complete bundle."""

    config.validate()
    formal_scientific_run = config.seed_namespace == SEED_NAMESPACE
    if formal_scientific_run:
        config.assert_frozen_protocol()
    config_payload = config.to_dict()
    config_hash = _canonical_sha256(config_payload)
    source_hash = source_tree_sha256()
    parent_snapshot = _validate_parent_formal_snapshot()
    environment = _environment_versions()
    rng_audit = (
        _audit_formal_rng_ids(config, output_dir=output_dir)
        if formal_scientific_run
        else {
            "status": "not_applicable_nonformal",
            "seed_namespace": config.seed_namespace,
            "source_actual_use_excludes_reservation_declarations": True,
        }
    )
    if resume:
        validate_bundle(
            output_dir,
            expected_config=config_payload,
            expected_config_hash=config_hash,
            expected_source_hash=source_hash,
            expected_parent_snapshot=parent_snapshot,
            expected_environment=environment,
            expected_rng_audit=rng_audit,
        )
        return _read_json(output_dir / "summary.json")
    if output_dir.exists():
        raise FileExistsError(f"fresh propensity output already exists: {output_dir}")

    started = time.perf_counter()
    result = run_propensity_robustness(config)
    runtime = time.perf_counter() - started
    if source_tree_sha256() != source_hash:
        raise RuntimeError("source tree changed while propensity study was running")
    if _validate_parent_formal_snapshot() != parent_snapshot:
        raise RuntimeError("parent formal snapshot changed while propensity study ran")
    if formal_scientific_run and _audit_formal_rng_ids(
        config,
        output_dir=output_dir,
    ) != rng_audit:
        raise RuntimeError("formal RNG inventory changed while propensity study ran")
    result.summary["parent_formal_snapshot"] = parent_snapshot
    result.summary["formal_rng_collision_audit"] = rng_audit
    metadata = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "diagnostic_only": True,
        "canonical_method_unchanged": True,
        "device": "cpu_exact",
        "formal_scientific_run": formal_scientific_run,
        "git_revision": git_revision(),
        "source_tree_sha256": source_hash,
        "config_sha256": config_hash,
        "seed_namespace": config.seed_namespace,
        "seed_collision_audit": propensity_seed_collision_audit(config),
        "formal_rng_collision_audit": rng_audit,
        "parent_formal_snapshot": parent_snapshot,
        "launch": {
            "argv": list(sys.argv),
            "cwd": str(Path.cwd()),
            "executable": sys.executable,
        },
        "environment_versions": environment,
        "multinomial_propensity_fit_semantics": _sklearn_multinomial_semantics(
            config,
            environment,
        ),
        "runtime_seconds": runtime,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "table_separation": {
            PRIMARY_LAYER: "primary_transport_only.csv",
            APPENDIX_LAYER: "appendix_end_to_end.csv",
        },
    }
    _publish_bundle(
        output_dir,
        config_payload=config_payload,
        metadata=metadata,
        summary=result.summary,
        arrays=result.arrays,
        nuisance_records=result.nuisance_records,
        primary_records=result.primary_records,
        appendix_records=result.appendix_records,
    )
    validate_bundle(
        output_dir,
        expected_config=config_payload,
        expected_config_hash=config_hash,
        expected_source_hash=source_hash,
        expected_parent_snapshot=parent_snapshot,
        expected_environment=environment,
        expected_rng_audit=rng_audit,
    )
    return result.summary


def validate_bundle(
    output_dir: Path,
    *,
    expected_config: dict[str, Any],
    expected_config_hash: str,
    expected_source_hash: str,
    expected_parent_snapshot: dict[str, Any],
    expected_environment: dict[str, Any],
    expected_rng_audit: dict[str, Any],
) -> None:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"propensity output does not exist: {output_dir}")
    required = {*PAYLOAD_NAMES, "manifest.json", "COMPLETE"}
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"partial propensity bundle; missing files: {missing}")

    manifest = _read_json(output_dir / "manifest.json")
    if manifest.get("protocol") != PROTOCOL or manifest.get("status") != "complete":
        raise RuntimeError("propensity manifest has the wrong protocol")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(PAYLOAD_NAMES):
        raise RuntimeError("propensity manifest file contract differs")
    for name in PAYLOAD_NAMES:
        content = (output_dir / name).read_bytes()
        observed = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        if files[name] != observed:
            raise RuntimeError(f"propensity payload hash differs: {name}")

    manifest_hash = hashlib.sha256((output_dir / "manifest.json").read_bytes()).hexdigest()
    metadata = _read_json(output_dir / "metadata.json")
    if _read_json(output_dir / "COMPLETE") != {
        "status": "complete",
        "manifest_sha256": manifest_hash,
        "config_sha256": metadata.get("config_sha256"),
        "source_tree_sha256": metadata.get("source_tree_sha256"),
        "parent_snapshot_manifest_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("manifest_sha256"),
        "parent_snapshot_archive_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("archive_sha256"),
        "parent_source_tree_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("parent_source_tree_sha256"),
        "formal_rng_audit_sha256": _canonical_sha256(
            metadata.get("formal_rng_collision_audit", {})
        ),
        "launch_sha256": _canonical_sha256(metadata.get("launch", {})),
        "environment_versions_sha256": _canonical_sha256(
            metadata.get("environment_versions", {})
        ),
        "multinomial_fit_semantics_sha256": _canonical_sha256(
            metadata.get("multinomial_propensity_fit_semantics", {})
        ),
    }:
        raise RuntimeError("propensity COMPLETE marker is malformed")
    if _read_json(output_dir / "config.json") != expected_config:
        raise RuntimeError("resume config differs from propensity bundle")
    if metadata.get("config_sha256") != expected_config_hash:
        raise RuntimeError("resume config hash differs from propensity bundle")
    if metadata.get("source_tree_sha256") != expected_source_hash:
        raise RuntimeError("resume source differs from propensity bundle")
    if metadata.get("parent_formal_snapshot") != expected_parent_snapshot:
        raise RuntimeError("resume parent formal snapshot binding differs")
    if metadata.get("environment_versions") != expected_environment:
        raise RuntimeError("resume runtime environment differs")
    if metadata.get("formal_rng_collision_audit") != expected_rng_audit:
        raise RuntimeError("resume formal RNG collision audit differs")
    if not isinstance(metadata.get("launch", {}).get("argv"), list):
        raise RuntimeError("propensity launch argv is missing")
    expected_fit_semantics = _sklearn_multinomial_semantics(
        PropensityRobustnessConfig(**expected_config),
        expected_environment,
    )
    if metadata.get("multinomial_propensity_fit_semantics") != expected_fit_semantics:
        raise RuntimeError("multinomial propensity fit semantics differ")
    if metadata.get("table_separation") != {
        PRIMARY_LAYER: "primary_transport_only.csv",
        APPENDIX_LAYER: "appendix_end_to_end.csv",
    }:
        raise RuntimeError("primary and appendix table separation differs")

    summary = _read_json(output_dir / "summary.json")
    if (
        summary.get("study") != PROTOCOL
        or summary.get("status") != "complete"
        or not summary.get(PRIMARY_LAYER, {}).get(
            "target_law_fingerprint_shared_across_arms"
        )
        or summary.get("parent_formal_snapshot") != expected_parent_snapshot
        or summary.get("formal_rng_collision_audit") != expected_rng_audit
    ):
        raise RuntimeError("propensity summary contract differs")
    _validate_arrays(output_dir / "arrays.npz", expected_config)
    _validate_tables(output_dir, expected_config)


def _validate_arrays(path: Path, config: dict[str, Any]) -> None:
    try:
        arrays = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"propensity arrays are unreadable: {error}") from error
    with arrays:
        required = {
            "problem_seeds",
            "nuisance_seeds",
            "calibration_seeds",
            "bootstrap_indices",
            "propensity_arms",
            "target_policy_tv",
            "primary_target_law_fingerprints",
            "appendix_target_law_fingerprints",
            "appendix_target_policy_drift",
            "primary_selected_policy_tv_from_oracle_behavior",
            "appendix_selected_policy_tv_from_own_anchor",
            "appendix_selected_policy_tv_from_oracle_target_matched_radii",
            "appendix_deployed_policy_tv_from_primary_oracle_deployment",
            "nuisance_mae",
            "nuisance_log_loss",
            "nuisance_excess_log_loss",
            "nuisance_mean_absolute_relative_error",
            "nuisance_maximum_absolute_relative_error",
            "nuisance_minimum_probability",
            "nuisance_iterations",
        }
        layer_fields = {
            "selected",
            "selected_indices",
            "exact_coverage",
            "exact_normalized_width",
            "estimated_coverage",
            "estimated_normalized_width",
            "ess_fraction",
            "log_weight_span",
            "failure_stage",
        }
        required.update(
            f"{layer}_{field}"
            for layer in (PRIMARY_LAYER, APPENDIX_LAYER)
            for field in layer_fields
        )
        if set(arrays.files) != required:
            raise RuntimeError("propensity array names differ")
        instances = config["instances"]
        horizon = config["horizon"]
        grid_size = config["grid_size"]
        arm_count = len(PROPENSITY_ARMS)
        if arrays["propensity_arms"].tolist() != list(PROPENSITY_ARMS):
            raise RuntimeError("propensity arm order differs")
        expected_problem_seeds = np.arange(
            config["problem_seed_start"],
            config["problem_seed_start"] + instances,
            dtype=np.int64,
        )
        expected_nuisance_seeds = np.arange(
            config["nuisance_seed_start"],
            config["nuisance_seed_start"] + instances,
            dtype=np.int64,
        )
        expected_calibration_seeds = np.arange(
            config["calibration_seed_start"],
            config["calibration_seed_start"] + instances,
            dtype=np.int64,
        )
        for name, expected in (
            ("problem_seeds", expected_problem_seeds),
            ("nuisance_seeds", expected_nuisance_seeds),
            ("calibration_seeds", expected_calibration_seeds),
        ):
            if not np.array_equal(arrays[name], expected):
                raise RuntimeError(f"{name} differs from the exact frozen sequence")

        bootstrap_indices = arrays["bootstrap_indices"]
        if bootstrap_indices.shape != (
            config["bootstrap_resamples"],
            instances,
        ):
            raise RuntimeError("paired bootstrap matrix has the wrong shape")
        expected_bootstrap = np.random.default_rng(config["bootstrap_seed"]).integers(
            0,
            instances,
            size=(config["bootstrap_resamples"], instances),
            dtype=np.int32,
        )
        if not np.array_equal(bootstrap_indices, expected_bootstrap):
            raise RuntimeError("paired bootstrap matrix is not seed-deterministic")

        primary_fingerprints = arrays["primary_target_law_fingerprints"]
        if primary_fingerprints.shape != (instances, arm_count):
            raise RuntimeError("primary fingerprint matrix has the wrong shape")
        if not np.all(primary_fingerprints == primary_fingerprints[:, :1]):
            raise RuntimeError("primary target law is not identical across arms")
        appendix_fingerprints = arrays["appendix_target_law_fingerprints"]
        if appendix_fingerprints.shape != (instances, arm_count):
            raise RuntimeError("appendix fingerprint matrix has the wrong shape")
        if not np.all(appendix_fingerprints[:, 0] == primary_fingerprints[:, 0]):
            raise RuntimeError("appendix oracle target law differs from primary oracle")
        valid_hex = set("0123456789abcdef")
        if any(
            len(str(value)) != 64 or not set(str(value)) <= valid_hex
            for value in np.concatenate(
                (primary_fingerprints.ravel(), appendix_fingerprints.ravel())
            )
        ):
            raise RuntimeError("target-law fingerprint encoding is malformed")

        _require_finite_range(arrays["target_policy_tv"], 0.0, 1.0, "target TV")
        appendix_drift = arrays["appendix_target_policy_drift"]
        if appendix_drift.shape != (instances, arm_count):
            raise RuntimeError("appendix target-policy drift has the wrong shape")
        _require_finite_range(appendix_drift, 0.0, 1.0, "appendix target drift")
        if not np.allclose(appendix_drift[:, 0], 0.0, atol=1e-14, rtol=0.0):
            raise RuntimeError("appendix oracle target-policy drift is nonzero")

        nuisance_names = (
            "mae",
            "log_loss",
            "excess_log_loss",
            "mean_absolute_relative_error",
            "maximum_absolute_relative_error",
            "minimum_probability",
        )
        for name in nuisance_names:
            if arrays[f"nuisance_{name}"].shape != (instances, arm_count):
                raise RuntimeError(f"nuisance {name} has the wrong shape")
            if not np.isfinite(arrays[f"nuisance_{name}"]).all():
                raise RuntimeError(f"nuisance {name} contains nonfinite values")
        _require_finite_range(arrays["nuisance_mae"], 0.0, 1.0, "nuisance MAE")
        if np.any(arrays["nuisance_log_loss"] < 0.0):
            raise RuntimeError("nuisance log loss is negative")
        if np.any(arrays["nuisance_excess_log_loss"] < -1e-10):
            raise RuntimeError("nuisance excess log loss violates the KL identity")
        if np.any(arrays["nuisance_mean_absolute_relative_error"] < 0.0):
            raise RuntimeError("mean absolute relative propensity error is negative")
        if np.any(arrays["nuisance_maximum_absolute_relative_error"] < 0.0):
            raise RuntimeError("maximum absolute relative propensity error is negative")
        if np.any(
            arrays["nuisance_maximum_absolute_relative_error"] + 1e-12
            < arrays["nuisance_mean_absolute_relative_error"]
        ):
            raise RuntimeError("maximum relative error is below the mean relative error")
        minimum_probability = arrays["nuisance_minimum_probability"]
        if np.any((minimum_probability <= 0.0) | (minimum_probability > 1.0)):
            raise RuntimeError("nuisance propensity positivity is invalid")
        for name in (
            "mae",
            "excess_log_loss",
            "mean_absolute_relative_error",
            "maximum_absolute_relative_error",
        ):
            if not np.allclose(
                arrays[f"nuisance_{name}"][:, 0],
                0.0,
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError(f"oracle nuisance {name} is not exactly zero")
        iterations = arrays["nuisance_iterations"]
        if iterations.shape != (instances, arm_count) or not np.issubdtype(
            iterations.dtype, np.integer
        ):
            raise RuntimeError("nuisance iteration counts are malformed")
        if np.any(iterations[:, 0] != 0) or np.any(iterations[:, 1:] <= 0):
            raise RuntimeError("nuisance iteration-count identities differ")
        if np.any(iterations[:, 1:] >= config["logistic_max_iterations"]):
            raise RuntimeError("a stored nuisance fit reached its iteration limit")

        for layer in (PRIMARY_LAYER, APPENDIX_LAYER):
            selected = arrays[f"{layer}_selected"]
            indices = arrays[f"{layer}_selected_indices"]
            failure_stage = arrays[f"{layer}_failure_stage"]
            if selected.shape != (instances, arm_count) or selected.dtype != np.bool_:
                raise RuntimeError(f"{layer} availability has the wrong contract")
            if indices.shape != (instances, arm_count, horizon):
                raise RuntimeError(f"{layer} selected indices have the wrong shape")
            if failure_stage.shape != (instances, arm_count):
                raise RuntimeError(f"{layer} failure stages have the wrong shape")
            if np.any((indices < -1) | (indices >= grid_size)):
                raise RuntimeError(f"{layer} selected radius index is out of range")

            exact_coverage = arrays[f"{layer}_exact_coverage"]
            exact_width = arrays[f"{layer}_exact_normalized_width"]
            estimated_coverage = arrays[f"{layer}_estimated_coverage"]
            estimated_width = arrays[f"{layer}_estimated_normalized_width"]
            ess = arrays[f"{layer}_ess_fraction"]
            span = arrays[f"{layer}_log_weight_span"]
            for name, values in (
                ("exact coverage", exact_coverage),
                ("exact width", exact_width),
                ("estimated coverage", estimated_coverage),
                ("estimated width", estimated_width),
                ("ESS", ess),
                ("log-weight span", span),
            ):
                if values.shape != (instances, arm_count, horizon):
                    raise RuntimeError(f"{layer} {name} has the wrong shape")

            for instance in range(instances):
                for arm in range(arm_count):
                    available = bool(selected[instance, arm])
                    failure = int(failure_stage[instance, arm])
                    completed = horizon if available else failure
                    if available:
                        if failure != -1:
                            raise RuntimeError(f"{layer} available row has a failure stage")
                    elif not 0 <= failure < horizon:
                        raise RuntimeError(f"{layer} unavailable row has invalid failure stage")
                    if np.any(indices[instance, arm, :completed] < 0) or np.any(
                        indices[instance, arm, completed:] != -1
                    ):
                        raise RuntimeError(f"{layer} prefix-selection identity differs")
                    completed_values = (
                        estimated_coverage[instance, arm, :completed],
                        estimated_width[instance, arm, :completed],
                        ess[instance, arm, :completed],
                        span[instance, arm, :completed],
                    )
                    if any(not np.isfinite(values).all() for values in completed_values):
                        raise RuntimeError(f"{layer} completed prefix is nonfinite")
                    remaining_values = (
                        estimated_coverage[instance, arm, completed:],
                        estimated_width[instance, arm, completed:],
                        ess[instance, arm, completed:],
                        span[instance, arm, completed:],
                    )
                    if any(not np.isnan(values).all() for values in remaining_values):
                        raise RuntimeError(f"{layer} failed suffix is not missing")
                    if completed:
                        if np.any(
                            estimated_coverage[instance, arm, :completed]
                            < 1.0 - config["alpha"] - 1e-12
                        ):
                            raise RuntimeError(f"{layer} selected empirical coverage is infeasible")
                        _require_finite_range(
                            estimated_coverage[instance, arm, :completed],
                            0.0,
                            1.0,
                            f"{layer} estimated coverage",
                        )
                        if np.any(estimated_width[instance, arm, :completed] < 0.0):
                            raise RuntimeError(f"{layer} estimated width is negative")
                        _require_finite_range(
                            ess[instance, arm, :completed],
                            np.nextafter(0.0, 1.0),
                            1.0,
                            f"{layer} ESS",
                        )
                        if np.any(span[instance, arm, :completed] < -1e-12):
                            raise RuntimeError(f"{layer} log-weight span is negative")
                    if available:
                        _require_finite_range(
                            exact_coverage[instance, arm],
                            0.0,
                            1.0,
                            f"{layer} exact coverage",
                        )
                        if not np.isfinite(exact_width[instance, arm]).all() or np.any(
                            exact_width[instance, arm] < 0.0
                        ):
                            raise RuntimeError(f"{layer} exact width is invalid")
                    elif not (
                        np.isnan(exact_coverage[instance, arm]).all()
                        and np.isnan(exact_width[instance, arm]).all()
                    ):
                        raise RuntimeError(f"{layer} unavailable exact metrics are populated")

        policy_tv_contracts = {
            "primary_selected_policy_tv_from_oracle_behavior": arrays[
                f"{PRIMARY_LAYER}_selected"
            ],
            "appendix_selected_policy_tv_from_own_anchor": arrays[
                f"{APPENDIX_LAYER}_selected"
            ],
            "appendix_selected_policy_tv_from_oracle_target_matched_radii": arrays[
                f"{APPENDIX_LAYER}_selected"
            ],
            "appendix_deployed_policy_tv_from_primary_oracle_deployment": (
                arrays[f"{APPENDIX_LAYER}_selected"]
                & arrays[f"{PRIMARY_LAYER}_selected"][:, :1]
            ),
        }
        for name, expected_finite in policy_tv_contracts.items():
            values = arrays[name]
            if values.shape != (instances, arm_count, horizon):
                raise RuntimeError(f"{name} has the wrong shape")
            finite_rows = np.isfinite(values).all(axis=2)
            missing_rows = np.isnan(values).all(axis=2)
            if not np.array_equal(finite_rows, expected_finite) or not np.array_equal(
                missing_rows,
                ~expected_finite,
            ):
                raise RuntimeError(f"{name} availability identity differs")
            if bool(expected_finite.any()):
                _require_finite_range(
                    values[expected_finite],
                    0.0,
                    1.0,
                    name,
                )
        oracle = PROPENSITY_ARMS.index("oracle")
        primary_oracle_selected = arrays[f"{PRIMARY_LAYER}_selected"][:, oracle]
        appendix_oracle_selected = arrays[f"{APPENDIX_LAYER}_selected"][:, oracle]
        if not np.array_equal(primary_oracle_selected, appendix_oracle_selected):
            raise RuntimeError("primary and appendix oracle availability differs")
        if not np.array_equal(
            arrays[f"{PRIMARY_LAYER}_selected_indices"][:, oracle],
            arrays[f"{APPENDIX_LAYER}_selected_indices"][:, oracle],
        ):
            raise RuntimeError("primary and appendix oracle schedules differ")
        if bool(primary_oracle_selected.any()):
            selected_rows = primary_oracle_selected
            if not np.allclose(
                arrays["appendix_selected_policy_tv_from_own_anchor"][
                    selected_rows, oracle
                ],
                arrays["primary_selected_policy_tv_from_oracle_behavior"][
                    selected_rows, oracle
                ],
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError("oracle selected-policy TV differs across layers")
            for name in (
                "appendix_selected_policy_tv_from_oracle_target_matched_radii",
                "appendix_deployed_policy_tv_from_primary_oracle_deployment",
            ):
                if not np.allclose(
                    arrays[name][selected_rows, oracle],
                    0.0,
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise RuntimeError(f"oracle arm is nonzero for {name}")


def _require_finite_range(
    values: np.ndarray,
    lower: float,
    upper: float,
    label: str,
) -> None:
    if not np.isfinite(values).all() or np.any(values < lower) or np.any(values > upper):
        raise RuntimeError(f"{label} is nonfinite or outside [{lower}, {upper}]")


def _validate_tables(output_dir: Path, config: dict[str, Any]) -> None:
    nuisance = pd.read_csv(output_dir / "nuisance_diagnostics.csv")
    primary = pd.read_csv(output_dir / "primary_transport_only.csv")
    appendix = pd.read_csv(output_dir / "appendix_end_to_end.csv")
    instances = config["instances"]
    expected_rows = instances * len(PROPENSITY_ARMS)
    if any(len(table) != expected_rows for table in (nuisance, primary, appendix)):
        raise RuntimeError("propensity table row count differs")
    if set(primary["layer"]) != {PRIMARY_LAYER}:
        raise RuntimeError("primary table contains a non-primary layer")
    if set(appendix["layer"]) != {APPENDIX_LAYER}:
        raise RuntimeError("appendix table contains a non-appendix layer")
    if set(primary["target_policy_drift_from_oracle"]) != {0.0}:
        raise RuntimeError("primary table changes the oracle target law")
    expected_problem = np.repeat(
        np.arange(
            config["problem_seed_start"],
            config["problem_seed_start"] + instances,
        ),
        len(PROPENSITY_ARMS),
    )
    expected_nuisance = np.repeat(
        np.arange(
            config["nuisance_seed_start"],
            config["nuisance_seed_start"] + instances,
        ),
        len(PROPENSITY_ARMS),
    )
    expected_calibration = np.repeat(
        np.arange(
            config["calibration_seed_start"],
            config["calibration_seed_start"] + instances,
        ),
        len(PROPENSITY_ARMS),
    )
    expected_arms = list(PROPENSITY_ARMS) * instances
    if any(
        table["problem_seed"].to_numpy().tolist() != expected_problem.tolist()
        for table in (nuisance, primary, appendix)
    ):
        raise RuntimeError("propensity table problem-seed order differs")
    if any(
        table["nuisance_seed"].to_numpy().tolist() != expected_nuisance.tolist()
        for table in (nuisance, primary, appendix)
    ):
        raise RuntimeError("propensity table nuisance-seed order differs")
    if any(
        table["calibration_seed"].to_numpy().tolist()
        != expected_calibration.tolist()
        for table in (nuisance, primary, appendix)
    ):
        raise RuntimeError("propensity table calibration-seed order differs")
    if any(table["arm"].tolist() != expected_arms for table in (nuisance, primary, appendix)):
        raise RuntimeError("propensity table arm order differs")
    if not primary.groupby("problem_seed")["target_law_fingerprint"].nunique().eq(1).all():
        raise RuntimeError("primary table target-law fingerprint differs across arms")
    appendix_oracle = appendix[appendix["arm"] == "oracle"].set_index("problem_seed")
    primary_oracle = primary[primary["arm"] == "oracle"].set_index("problem_seed")
    if appendix_oracle["target_law_fingerprint"].to_dict() != primary_oracle[
        "target_law_fingerprint"
    ].to_dict():
        raise RuntimeError("appendix oracle target law differs from primary table")
    if not np.allclose(
        appendix_oracle["target_policy_drift_from_oracle"].to_numpy(),
        0.0,
        atol=1e-14,
        rtol=0.0,
    ):
        raise RuntimeError("appendix oracle target-policy drift is nonzero")

    with np.load(output_dir / "arrays.npz", allow_pickle=False) as arrays:
        _cross_validate_csv_and_npz(nuisance, primary, appendix, arrays)


def _cross_validate_csv_and_npz(
    nuisance: pd.DataFrame,
    primary: pd.DataFrame,
    appendix: pd.DataFrame,
    arrays: Any,
) -> None:
    """Require every numeric CSV cell to agree with its NPZ source of truth."""

    nuisance_columns = {
        "mae": "nuisance_mae",
        "log_loss": "nuisance_log_loss",
        "excess_log_loss": "nuisance_excess_log_loss",
        "mean_absolute_relative_error": "nuisance_mean_absolute_relative_error",
        "maximum_absolute_relative_error": "nuisance_maximum_absolute_relative_error",
        "minimum_probability": "nuisance_minimum_probability",
        "iterations": "nuisance_iterations",
    }
    for column, array_name in nuisance_columns.items():
        observed = nuisance[column].to_numpy().reshape(arrays[array_name].shape)
        if not np.allclose(observed, arrays[array_name], equal_nan=True):
            raise RuntimeError(f"nuisance CSV differs from NPZ for {column}")
    if not nuisance["converged"].astype(bool).all():
        raise RuntimeError("nuisance CSV contains a nonconverged fitted arm")

    vector_columns = {
        "selected_indices": "selected_indices",
        "estimated_coverage_by_stage": "estimated_coverage",
        "estimated_normalized_width_by_stage": "estimated_normalized_width",
        "exact_coverage_by_stage": "exact_coverage",
        "exact_normalized_width_by_stage": "exact_normalized_width",
        "ess_fraction_by_stage": "ess_fraction",
        "log_weight_span_by_stage": "log_weight_span",
    }
    global_policy_columns = {
        PRIMARY_LAYER: {
            "selected_policy_tv_by_stage": (
                "primary_selected_policy_tv_from_oracle_behavior"
            ),
        },
        APPENDIX_LAYER: {
            "selected_policy_tv_by_stage": (
                "appendix_selected_policy_tv_from_own_anchor"
            ),
            "matched_oracle_target_tv_by_stage": (
                "appendix_selected_policy_tv_from_oracle_target_matched_radii"
            ),
            "primary_oracle_deployment_tv_by_stage": (
                "appendix_deployed_policy_tv_from_primary_oracle_deployment"
            ),
        },
    }
    for layer, table in ((PRIMARY_LAYER, primary), (APPENDIX_LAYER, appendix)):
        selected = table["selection_available"].astype(bool).to_numpy().reshape(
            arrays[f"{layer}_selected"].shape
        )
        if not np.array_equal(selected, arrays[f"{layer}_selected"]):
            raise RuntimeError(f"{layer} CSV availability differs from NPZ")
        failure = table["failure_stage"].fillna(-1).to_numpy().reshape(
            arrays[f"{layer}_failure_stage"].shape
        )
        if not np.array_equal(failure, arrays[f"{layer}_failure_stage"]):
            raise RuntimeError(f"{layer} CSV failure stages differ from NPZ")
        for column, suffix in vector_columns.items():
            observed = np.stack([_csv_vector(value) for value in table[column]])
            observed = observed.reshape(arrays[f"{layer}_{suffix}"].shape)
            if not np.allclose(
                observed,
                arrays[f"{layer}_{suffix}"],
                equal_nan=True,
            ):
                raise RuntimeError(f"{layer} CSV differs from NPZ for {column}")
        for column, array_name in global_policy_columns[layer].items():
            observed = np.stack([_csv_vector(value) for value in table[column]])
            observed = observed.reshape(arrays[array_name].shape)
            if not np.allclose(observed, arrays[array_name], equal_nan=True):
                raise RuntimeError(f"{layer} CSV differs from NPZ for {column}")
            mean_column = column.replace("_by_stage", "_mean")
            expected_means = np.asarray(
                [
                    float(row.mean()) if np.isfinite(row).any() else np.nan
                    for row in arrays[array_name].reshape(-1, arrays[array_name].shape[-1])
                ]
            )
            if not np.allclose(
                table[mean_column].to_numpy(),
                expected_means,
                equal_nan=True,
            ):
                raise RuntimeError(f"{layer} CSV mean differs from NPZ for {column}")
        expected_fingerprints = arrays[f"{layer.split('_')[0]}_target_law_fingerprints"]
        if table["target_law_fingerprint"].tolist() != expected_fingerprints.ravel().tolist():
            raise RuntimeError(f"{layer} CSV target fingerprints differ from NPZ")

    if not np.allclose(
        primary["target_policy_drift_from_oracle"].to_numpy(),
        0.0,
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("primary CSV candidate target drift is nonzero")
    if not np.allclose(
        appendix["target_policy_drift_from_oracle"].to_numpy(),
        arrays["appendix_target_policy_drift"].ravel(),
        equal_nan=True,
    ):
        raise RuntimeError("appendix CSV candidate target drift differs from NPZ")


def _csv_vector(value: object) -> np.ndarray:
    text = str(value).strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise RuntimeError("propensity CSV vector encoding is malformed")
    vector = np.fromstring(text[1:-1], sep=",", dtype=np.float64)
    if vector.size == 0:
        raise RuntimeError("propensity CSV vector is empty")
    return vector


def _validate_parent_formal_snapshot(
    *,
    manifest_path: Path = PARENT_SNAPSHOT_MANIFEST,
    sidecar_path: Path = PARENT_SNAPSHOT_SIDECAR,
    archive_path: Path = PARENT_SNAPSHOT_ARCHIVE,
) -> dict[str, Any]:
    """Validate the content-addressed source snapshot this extension builds on."""

    manifest_path = manifest_path.resolve()
    sidecar_path = sidecar_path.resolve()
    archive_path = archive_path.resolve()
    if not all(path.is_file() for path in (manifest_path, sidecar_path, archive_path)):
        raise FileNotFoundError("propensity robustness requires the parent snapshot bundle")
    manifest_hash = _sha256(manifest_path)
    archive_hash = _sha256(archive_path)
    if manifest_hash != EXPECTED_PARENT_MANIFEST_SHA256:
        raise RuntimeError("parent formal snapshot manifest SHA256 differs")
    if archive_hash != EXPECTED_PARENT_ARCHIVE_SHA256:
        raise RuntimeError("parent formal snapshot archive SHA256 differs")
    if archive_path.stat().st_size != EXPECTED_PARENT_ARCHIVE_BYTES:
        raise RuntimeError("parent formal snapshot archive size differs")
    sidecar_fields = sidecar_path.read_text(encoding="utf-8").split()
    if not sidecar_fields or sidecar_fields[0] != EXPECTED_PARENT_MANIFEST_SHA256:
        raise RuntimeError("parent formal snapshot sidecar differs")

    manifest = _read_json(manifest_path)
    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        raise RuntimeError("parent formal snapshot archive contract is malformed")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("role") != "content_addressed_formal_source_snapshot"
        or manifest.get("source_tree_sha256")
        != EXPECTED_PARENT_SOURCE_TREE_SHA256
        or archive.get("path")
        != "results/work/formal_source_snapshot_7665dfbe_20260825.tar.gz"
        or archive.get("sha256") != EXPECTED_PARENT_ARCHIVE_SHA256
        or archive.get("bytes") != EXPECTED_PARENT_ARCHIVE_BYTES
    ):
        raise RuntimeError("parent formal snapshot manifest contract differs")
    return {
        "role": "parent_formal_source_snapshot",
        "manifest_path": str(PARENT_SNAPSHOT_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": manifest_hash,
        "manifest_sidecar_path": str(PARENT_SNAPSHOT_SIDECAR.relative_to(ROOT)),
        "archive_path": str(PARENT_SNAPSHOT_ARCHIVE.relative_to(ROOT)),
        "archive_sha256": archive_hash,
        "archive_bytes": archive_path.stat().st_size,
        "parent_source_tree_sha256": EXPECTED_PARENT_SOURCE_TREE_SHA256,
        "parent_git_revision": manifest.get("git_revision"),
        "relationship": (
            "propensity robustness is post-snapshot extension work; its active "
            "source hash is bound separately and is not claimed to be in the archive"
        ),
    }


def _environment_versions() -> dict[str, Any]:
    """Record interpreter, numerical stack, and BLAS provenance."""

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
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _sklearn_multinomial_semantics(
    config: PropensityRobustnessConfig,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Make the version-specific sklearn nuisance objective explicit."""

    return {
        "library": "scikit-learn",
        "library_version": environment["scikit_learn"],
        "estimator": "sklearn.linear_model.LogisticRegression",
        "class_count": config.action_count,
        "loss": (
            "multinomial negative log-likelihood over all three action classes, "
            "with L2 penalty"
        ),
        "multiclass_resolution": (
            "multi_class argument omitted; with three classes and lbfgs, the "
            f"scikit-learn {environment['scikit_learn']} behavior resolves to the "
            "multinomial objective"
        ),
        "solver": config.logistic_solver,
        "solver_semantics": "primal L-BFGS optimization of smooth L2-penalized loss",
        "penalty": config.logistic_penalty,
        "inverse_regularization_C": config.logistic_inverse_regularization,
        "fit_intercept": False,
        "class_weight": None,
        "maximum_iterations": config.logistic_max_iterations,
        "convergence_tolerance": config.logistic_tolerance,
        "nonconvergence_policy": "fail_closed",
        "correct_model_features": "saturated full-state one-hot",
        "misspecified_model_features": (
            f"two state bins split at {config.reduced_state_cutpoint}"
        ),
    }


def _formal_rng_mapping(config: PropensityRobustnessConfig) -> dict[str, int]:
    return {
        **{
            f"instance_{index:03d}/problem": seed
            for index, seed in enumerate(config.problem_seeds)
        },
        **{
            f"instance_{index:03d}/nuisance_logging": seed
            for index, seed in enumerate(config.nuisance_seeds)
        },
        **{
            f"instance_{index:03d}/calibration_logging": seed
            for index, seed in enumerate(config.calibration_seeds)
        },
        "summary/paired_problem_bootstrap": config.bootstrap_seed,
    }


def _audit_formal_rng_ids(
    config: PropensityRobustnessConfig,
    *,
    output_dir: Path,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Reject prior actual use of each formal propensity RNG identifier."""

    config.assert_frozen_protocol()
    artifact_root = ROOT / "results" if artifact_root is None else artifact_root
    source_root = ROOT if source_root is None else source_root
    mapping = _formal_rng_mapping(config)
    formal_ids = set(mapping.values())
    expected_count = 3 * config.instances + 1
    if len(mapping) != expected_count or len(formal_ids) != expected_count:
        raise RuntimeError("formal propensity RNG mapping is not one-to-one and complete")
    enumerated = set(propensity_seed_collision_audit(config)["all_rng_ids"])
    if enumerated != formal_ids:
        raise RuntimeError("formal propensity RNG mapping differs from config enumeration")

    artifact_ids = _artifact_rng_ids(artifact_root, excluded_root=output_dir)
    excluded_paths = {
        Path(__file__).resolve(),
        (ROOT / "src" / "scpcp" / "propensity_robustness.py").resolve(),
    }
    source_ids = _source_actual_rng_ids(source_root, excluded_paths=excluded_paths)
    coordinated_external = set().union(
        *(set(values) for values in EXTERNAL_SEED_RESERVATIONS.values())
    )
    prior_ids = artifact_ids | source_ids | coordinated_external
    collisions = {
        label: identifier
        for label, identifier in mapping.items()
        if identifier in prior_ids
    }
    audit = {
        "status": "passed_before_launch" if not collisions else "collision",
        "seed_namespace": config.seed_namespace,
        "formal_rng_id_count": len(formal_ids),
        "formal_rng_ids": sorted(formal_ids),
        "formal_rng_id_sha256": _integer_set_sha256(formal_ids),
        "formal_rng_mapping": mapping,
        "formal_rng_mapping_sha256": _canonical_sha256(mapping),
        "artifact_actual_use_rng_id_count": len(artifact_ids),
        "artifact_actual_use_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "source_actual_use_rng_id_count": len(source_ids),
        "source_actual_use_rng_id_sha256": _integer_set_sha256(source_ids),
        "coordinated_external_rng_id_count": len(coordinated_external),
        "coordinated_external_rng_id_sha256": _integer_set_sha256(
            coordinated_external
        ),
        "collision_count": len(collisions),
        "collisions": collisions,
        "excluded_output": str(output_dir.resolve()),
        "excluded_assignment_sources": sorted(str(path) for path in excluded_paths),
        "source_actual_use_excludes_reservation_declarations": True,
        "artifact_actual_use_excludes_reservation_declarations": True,
    }
    audit["audit_sha256"] = _canonical_sha256(audit)
    if collisions:
        raise RuntimeError(
            "formal propensity RNG IDs collide with prior actual use: "
            f"{collisions}"
        )
    return audit


def _artifact_rng_ids(root: Path, *, excluded_root: Path) -> set[int]:
    values: set[int] = set()
    if not root.exists():
        return values
    excluded = excluded_root.resolve()
    provenance_names = {
        "metadata.json",
        "study_metadata.json",
        "manifest.json",
        "summary.json",
        "config.json",
        "config.yaml",
    }
    for path in root.rglob("*"):
        resolved = path.resolve()
        if _is_relative_to(resolved, excluded):
            continue
        match = ARTIFACT_ID.fullmatch(path.name)
        if match:
            values.add(int(match.group(1)))
        if not path.is_file() or path.name not in provenance_names:
            continue
        try:
            payload = (
                yaml.safe_load(path.read_text(encoding="utf-8"))
                if path.suffix in {".yaml", ".yml"}
                else json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
            raise RuntimeError(f"cannot audit RNG-bearing artifact {path}") from error
        _collect_named_rng_values(payload, values)
    return values


def _source_actual_rng_ids(
    root: Path,
    *,
    excluded_paths: set[Path],
) -> set[int]:
    values: set[int] = set()
    for directory in ("scripts", "src", "tools", "configs"):
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            resolved = path.resolve()
            if resolved in excluded_paths or not path.is_file():
                continue
            path_values: set[int] = set()
            if path.suffix == ".py":
                _collect_python_rng_assignments(path, path_values)
            elif path.suffix in {".yaml", ".yml"}:
                try:
                    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                except (OSError, yaml.YAMLError) as error:
                    raise RuntimeError(
                        f"cannot audit source RNG assignments in {path}"
                    ) from error
                _collect_named_rng_values(payload, path_values)
            values.update(path_values)
    return values


def _collect_python_rng_assignments(path: Path, values: set[int]) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise RuntimeError(f"cannot parse RNG assignments in {path}") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [name for target in node.targets for name in _target_names(target)]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            names = list(_target_names(node.target))
            value = node.value
        else:
            continue
        if value is None or not any(RNG_KEY.search(name) for name in names):
            continue
        if any(RESERVATION_KEY.search(name) for name in names):
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
        if node.func.id in {"tuple", "list", "set", "frozenset"} and len(
            node.args
        ) == 1:
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
    if RESERVATION_KEY.search(key_path):
        return
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            child_path = f"{key_path}.{child_key}" if key_path else str(child_key)
            _collect_named_rng_values(child_value, output, child_path)
        return
    if isinstance(value, list):
        for child in value:
            _collect_named_rng_values(child, output, key_path)
        return
    if RNG_KEY.search(key_path) and isinstance(value, int) and not isinstance(value, bool):
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


def _publish_bundle(
    output_dir: Path,
    *,
    config_payload: dict[str, Any],
    metadata: dict[str, Any],
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    nuisance_records: list[dict[str, Any]],
    primary_records: list[dict[str, Any]],
    appendix_records: list[dict[str, Any]],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _write_json(temporary / "config.json", config_payload)
        _write_json(temporary / "metadata.json", metadata)
        _write_json(temporary / "summary.json", summary)
        np.savez_compressed(temporary / "arrays.npz", **arrays)
        _fsync_file(temporary / "arrays.npz")
        pd.DataFrame(nuisance_records).to_csv(
            temporary / "nuisance_diagnostics.csv", index=False
        )
        pd.DataFrame(primary_records).to_csv(
            temporary / "primary_transport_only.csv", index=False
        )
        pd.DataFrame(appendix_records).to_csv(
            temporary / "appendix_end_to_end.csv", index=False
        )
        for name in (
            "nuisance_diagnostics.csv",
            "primary_transport_only.csv",
            "appendix_end_to_end.csv",
        ):
            _fsync_file(temporary / name)
        files = {}
        for name in PAYLOAD_NAMES:
            content = (temporary / name).read_bytes()
            files[name] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        _write_json(
            temporary / "manifest.json",
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "status": "complete",
                "files": files,
            },
        )
        manifest_hash = hashlib.sha256((temporary / "manifest.json").read_bytes()).hexdigest()
        _write_json(
            temporary / "COMPLETE",
            {
                "status": "complete",
                "manifest_sha256": manifest_hash,
                "config_sha256": metadata["config_sha256"],
                "source_tree_sha256": metadata["source_tree_sha256"],
                "parent_snapshot_manifest_sha256": metadata[
                    "parent_formal_snapshot"
                ]["manifest_sha256"],
                "parent_snapshot_archive_sha256": metadata[
                    "parent_formal_snapshot"
                ]["archive_sha256"],
                "parent_source_tree_sha256": metadata[
                    "parent_formal_snapshot"
                ]["parent_source_tree_sha256"],
                "formal_rng_audit_sha256": _canonical_sha256(
                    metadata["formal_rng_collision_audit"]
                ),
                "launch_sha256": _canonical_sha256(metadata["launch"]),
                "environment_versions_sha256": _canonical_sha256(
                    metadata["environment_versions"]
                ),
                "multinomial_fit_semantics_sha256": _canonical_sha256(
                    metadata["multinomial_propensity_fit_semantics"]
                ),
            },
        )
        _fsync_directory(temporary)
        os.replace(temporary, output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    content = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
