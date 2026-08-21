"""Validate and summarize the Phase 0C joint-search audit."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
import uuid

import numpy as np
import pandas as pd
import yaml


_COVERAGE_CRITICAL_VALUE = 3.0440624276034796
_REQUIRED_SEEDS = 40
_BOOTSTRAP_REPLICATES = 10_000
_BOOTSTRAP_SEED = 2_718_281
_SEEDS = tuple(range(10_000, 10_040))
_SCENARIOS = ("standard", "tail_shift")
_INITIAL_METHODS = ("current_profiled", "greedy", "joint_B", "joint_2B")
_PAYLOAD_NAMES = (
    "phase0c_summary.csv",
    "phase0c_decision.json",
    "phase0c_summary.md",
    "phase0c_joint_search.pdf",
    "phase0c_joint_search.svg",
    "phase0c_joint_search.png",
)
_replace_path = os.replace


def _finite_positive_vector(values: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{label} must be a nonempty vector")
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError(f"{label} must contain finite positive values")
    return array


def paired_geometric_ratio(joint: np.ndarray, current: np.ndarray) -> float:
    """Return the geometric mean of paired positive width ratios."""

    joint_array = _finite_positive_vector(joint, label="joint widths")
    current_array = _finite_positive_vector(current, label="current widths")
    if joint_array.shape != current_array.shape:
        raise ValueError("paired widths must have identical shapes")
    return float(np.exp(np.mean(np.log(joint_array / current_array))))


def relative_budget_gain(r_b: float, r_2b: float) -> float:
    """Return the registered relative gain from B to 2B."""

    if not math.isfinite(r_b) or not math.isfinite(r_2b) or r_b <= 0.0 or r_2b <= 0.0:
        raise ValueError("budget ratios must be finite and positive")
    return (r_b - r_2b) / r_b


def _bootstrap_indices(size: int, supplied: np.ndarray | None) -> np.ndarray:
    if supplied is None:
        return np.random.default_rng(_BOOTSTRAP_SEED).integers(
            0, size, size=(_BOOTSTRAP_REPLICATES, size)
        )
    indices = np.asarray(supplied)
    if (
        indices.ndim != 2
        or indices.shape[1] != size
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= size)
    ):
        raise ValueError("bootstrap indices are invalid for the paired sample")
    return indices


def _ratio_replicates(
    joint: np.ndarray, current: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    log_ratios = np.log(joint / current)
    return np.exp(np.mean(log_ratios[indices], axis=1))


def paired_ratio_summary(
    joint: np.ndarray,
    current: np.ndarray,
    *,
    bootstrap_indices: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Summarize a paired geometric ratio with a descriptive percentile interval."""

    joint_array = _finite_positive_vector(joint, label="joint widths")
    current_array = _finite_positive_vector(current, label="current widths")
    if joint_array.shape != current_array.shape:
        raise ValueError("paired widths must have identical shapes")
    indices = _bootstrap_indices(len(joint_array), bootstrap_indices)
    replicates = _ratio_replicates(joint_array, current_array, indices)
    lower, upper = np.quantile(replicates, (0.025, 0.975))
    return {
        "n_pairs": len(joint_array),
        "point_ratio": paired_geometric_ratio(joint_array, current_array),
        "bootstrap_ci_lower": float(lower),
        "bootstrap_ci_upper": float(upper),
    }


def paired_delta_summary(
    b: np.ndarray,
    b2: np.ndarray,
    current: np.ndarray,
    *,
    bootstrap_indices: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Summarize the registered B-to-2B gain using shared paired resamples."""

    b_array = _finite_positive_vector(b, label="B widths")
    b2_array = _finite_positive_vector(b2, label="2B widths")
    current_array = _finite_positive_vector(current, label="current widths")
    if b_array.shape != b2_array.shape or b_array.shape != current_array.shape:
        raise ValueError("B, 2B, and current widths must have identical shapes")
    indices = _bootstrap_indices(len(b_array), bootstrap_indices)
    b_replicates = _ratio_replicates(b_array, current_array, indices)
    b2_replicates = _ratio_replicates(b2_array, current_array, indices)
    gain_replicates = (b_replicates - b2_replicates) / b_replicates
    lower, upper = np.quantile(gain_replicates, (0.025, 0.975))
    r_b = paired_geometric_ratio(b_array, current_array)
    r_2b = paired_geometric_ratio(b2_array, current_array)
    return {
        "n_pairs": len(b_array),
        "point_gain": relative_budget_gain(r_b, r_2b),
        "bootstrap_ci_lower": float(lower),
        "bootstrap_ci_upper": float(upper),
    }


def coverage_summary(coverage: np.ndarray) -> dict[str, Any]:
    """Compute the registered 24-cell seed-level coverage band."""

    array = np.asarray(coverage, dtype=np.float64)
    if array.shape != (_REQUIRED_SEEDS, 2, 12):
        raise ValueError("coverage must have shape [40,2,12]")
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("coverage must contain finite probabilities")
    stage_seed_mean = np.mean(array, axis=0)
    stage_seed_sd = np.std(array, axis=0, ddof=1)
    simultaneous_lcb = stage_seed_mean - (
        _COVERAGE_CRITICAL_VALUE * stage_seed_sd / math.sqrt(_REQUIRED_SEEDS)
    )
    minimum_lcb = float(np.min(simultaneous_lcb))
    return {
        "n_seeds": _REQUIRED_SEEDS,
        "stage_seed_mean": stage_seed_mean.tolist(),
        "simultaneous_lcb": simultaneous_lcb.tolist(),
        "minimum_stage_seed_mean": float(np.min(stage_seed_mean)),
        "minimum_simultaneous_lcb": minimum_lcb,
        "mean_seedwise_stage_minimum": float(np.mean(np.min(array, axis=(1, 2)))),
        "raw_seed_stage_minimum": float(np.min(array)),
        "coverage_valid": minimum_lcb >= 0.90,
    }


def decide_initial(*, valid: bool, r_2b: float, delta_b: float) -> str:
    """Apply the preregistered initial Phase 0C decision boundaries."""

    if not valid:
        return "STOP_SCALAR_UNAVAILABLE"
    if not math.isfinite(r_2b) or not math.isfinite(delta_b):
        raise ValueError("available initial statistics must be finite")
    if r_2b <= 0.92:
        return "PROMISING_ORACLE_DIAGNOSTIC"
    if delta_b < 0.005:
        return "STOP_SCALAR_SATURATED"
    return "EXTENSION_8SP_REQUIRED"


def decide_extension(*, valid: bool, r_8sp: float) -> str:
    """Apply the preregistered optional-extension decision boundary."""

    if not valid:
        return "STOP_SCALAR_UNAVAILABLE"
    if not math.isfinite(r_8sp):
        raise ValueError("available extension statistic must be finite")
    if r_8sp <= 0.92:
        return "PROMISING_ORACLE_DIAGNOSTIC"
    return "STOP_SCALAR_INSUFFICIENT"


def _runner_module():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("run_phase0c_joint_search")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"study config is unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("study config must contain a mapping")
    return payload


def _validated_study(
    root: Path,
    *,
    kind: str,
    parent_root: Path | None = None,
) -> dict[str, Any]:
    runner = _runner_module()
    root = root.resolve()
    manifest = runner.validate_study_manifest(root, expected_kind=kind)
    if tuple(manifest["ordered_seeds"]) != _SEEDS:
        raise RuntimeError("study ordered_seeds must be exactly 10000..10039")
    metadata = _read_json(root / "study_metadata.json", label="study metadata")
    execution = metadata.get("execution")
    if not isinstance(execution, dict):
        raise RuntimeError("study execution metadata is missing")
    config = _load_config_mapping(root / "config.yaml")
    if runner.canonical_config_sha256(config) != manifest["config_sha256"]:
        raise RuntimeError("study config scientific hash differs")
    if runner.runtime_config_sha256(config) != manifest["runtime_config_sha256"]:
        raise RuntimeError("study config runtime hash differs")

    rows: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    seed_metadata_by_seed: dict[int, dict[str, Any]] = {}
    for seed in _SEEDS:
        seed_dir = root / f"seed_{seed:05d}"
        runner.validate_seed_artifact(
            seed_dir,
            seed,
            mode=kind,
            expected_execution=execution,
            parent_seed_dir=(
                None if parent_root is None else parent_root / f"seed_{seed:05d}"
            ),
        )
        records = pd.read_csv(seed_dir / "records.csv")
        rows.extend(records.to_dict(orient="records"))
        seed_metadata = _read_json(
            seed_dir / "metadata.json", label=f"seed {seed} metadata"
        )
        seed_metadata_by_seed[seed] = seed_metadata
        measurement = seed_metadata["diagnostics"].get("runner_measurement")
        if not isinstance(measurement, dict):
            raise RuntimeError(f"seed {seed} runner measurement is missing")
        measurements.append(measurement)
    runner._validate_global_streams(root, _SEEDS)
    return {
        "root": root,
        "manifest": manifest,
        "metadata": metadata,
        "execution": execution,
        "rows": rows,
        "measurements": measurements,
        "seed_metadata": seed_metadata_by_seed,
    }


def _selected(row: dict[str, Any]) -> bool:
    value = row["selection_available"]
    if not isinstance(value, (bool, np.bool_)):
        raise RuntimeError("validated row availability lost its boolean type")
    return bool(value)


def _row_index(rows: list[dict[str, Any]]) -> dict[tuple[int, str, str], dict[str, Any]]:
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["scenario"]), str(row["method_id"]))
        if key in indexed:
            raise RuntimeError(f"duplicate validated row key: {key}")
        indexed[key] = row
    return indexed


def _pair_count(
    indexed: dict[tuple[int, str, str], dict[str, Any]], method: str
) -> int:
    return sum(
        _selected(indexed[(seed, scenario, "current_profiled")])
        and _selected(indexed[(seed, scenario, method)])
        for seed in _SEEDS
        for scenario in _SCENARIOS
    )


def _coverage_for_method(
    indexed: dict[tuple[int, str, str], dict[str, Any]], method: str
) -> dict[str, Any] | None:
    rows = [
        indexed[(seed, scenario, method)]
        for seed in _SEEDS
        for scenario in _SCENARIOS
    ]
    if not all(_selected(row) for row in rows):
        return None
    coverage = np.asarray(
        [json.loads(row["final_coverage_json"]) for row in rows], dtype=np.float64
    ).reshape(_REQUIRED_SEEDS, 2, 12)
    return coverage_summary(coverage)


def _tail_widths(
    indexed: dict[tuple[int, str, str], dict[str, Any]], method: str
) -> tuple[np.ndarray, np.ndarray] | None:
    current_rows = [
        indexed[(seed, "tail_shift", "current_profiled")] for seed in _SEEDS
    ]
    method_rows = [indexed[(seed, "tail_shift", method)] for seed in _SEEDS]
    if not all(_selected(row) for row in current_rows + method_rows):
        return None
    current = np.asarray(
        [row["micro_normalized_width"] for row in current_rows], dtype=np.float64
    )
    joint = np.asarray(
        [row["micro_normalized_width"] for row in method_rows], dtype=np.float64
    )
    return joint, current


def _extension_eligibility(initial: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, object]] = []
    eligible = 0
    for seed in _SEEDS:
        metadata = initial["seed_metadata"][seed]
        for scenario in _SCENARIOS:
            diagnostics = metadata["diagnostics"][scenario]
            eligible += int(diagnostics["extension_eligible"])
            hashes = diagnostics["pair4_state_sha256"]
            names = diagnostics["active_start_names"][: len(hashes)]
            for start_name, state_hash in zip(names, hashes, strict=True):
                entries.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "start_name": start_name,
                        "sha256": state_hash,
                    }
                )
    state_count = len(entries)
    state_manifest_sha = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "all_eligible": eligible == 80 and state_count == 240,
        "eligible_scenario_seed_count": eligible,
        "required_scenario_seed_count": 80,
        "canonical_state_hash_count": state_count,
        "state_hash_manifest_sha256": state_manifest_sha,
    }


def _checkpoint_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoints: dict[str, Any] = {}
    for method in (*_INITIAL_METHODS, "joint_8SP"):
        method_rows = [row for row in rows if row["method_id"] == method]
        if not method_rows:
            continue
        by_scenario: dict[str, Any] = {}
        for scenario in _SCENARIOS:
            scenario_rows = [row for row in method_rows if row["scenario"] == scenario]
            selected = [row for row in scenario_rows if _selected(row)]
            winner_counts = {
                name: sum(row["chosen_initialization"] == name for row in selected)
                for name in ("profiled", "greedy", "upper_endpoint")
            }
            by_scenario[scenario] = {
                "available_seed_count": len(selected),
                "joint_feasible_seed_count": sum(
                    bool(row["tuning_joint_feasible"]) for row in scenario_rows
                ),
                "winner_counts": winner_counts,
                "winner_denominator": len(selected),
                "endpoint_stage_count": int(
                    np.sum([row["selected_endpoint_stage_count"] for row in selected])
                ),
                "endpoint_stage_denominator": 12 * len(selected),
            }
        checkpoints[method] = {"by_scenario": by_scenario}
    return checkpoints


def _runner_audit(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = np.asarray(
        [measurement["elapsed_seconds"] for measurement in measurements],
        dtype=np.float64,
    )
    q1, median, q3 = np.quantile(elapsed, (0.25, 0.5, 0.75))
    return {
        "n_seed_runs": len(measurements),
        "elapsed_seconds": {
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "max": float(np.max(elapsed)),
        },
        "max_memory_allocated_bytes": int(
            max(item["max_memory_allocated_bytes"] for item in measurements)
        ),
        "max_memory_reserved_bytes": int(
            max(item["max_memory_reserved_bytes"] for item in measurements)
        ),
    }


def _audit_summary(
    initial_rows: list[dict[str, Any]],
    initial_measurements: list[dict[str, Any]],
    extension_rows: list[dict[str, Any]] | None,
    extension_measurements: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "checkpoints": _checkpoint_audit(
            initial_rows + ([] if extension_rows is None else extension_rows)
        ),
        "runner": {
            "initial": _runner_audit(initial_measurements),
            "extension": (
                None
                if extension_measurements is None
                else _runner_audit(extension_measurements)
            ),
        },
    }


def _validate_extension_parent(
    initial: dict[str, Any], extension: dict[str, Any]
) -> None:
    checkpoint_dir = initial["root"] / "checkpoint_analysis"
    validate_summary_bundle(checkpoint_dir)
    checkpoint_decision = _read_json(
        checkpoint_dir / "phase0c_decision.json", label="checkpoint decision"
    )
    if checkpoint_decision.get("decision") != "EXTENSION_8SP_REQUIRED":
        raise RuntimeError("extension parent checkpoint does not authorize 8SP")
    execution = extension["execution"]
    expected = {
        "parent_output_dir": str(initial["root"]),
        "parent_study_manifest_sha256": _file_sha256(
            initial["root"] / "study_manifest.json"
        ),
        "parent_execution_sha256": initial["execution"]["execution_sha256"],
        "checkpoint_decision_sha256": _file_sha256(
            checkpoint_dir / "phase0c_decision.json"
        ),
    }
    for field, value in expected.items():
        if execution.get(field) != value:
            raise RuntimeError(f"extension {field} differs from authenticated parent")
    for field in ("source_tree_sha256", "experiment_tree_sha256", "config_sha256"):
        if extension["manifest"][field] != initial["manifest"][field]:
            raise RuntimeError(f"extension {field} differs from initial study")


def load_validate_analyze(
    input_dir: Path | str,
    extension_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Deeply validate completed studies, then compute registered statistics."""

    initial = _validated_study(Path(input_dir), kind="initial")
    initial_index = _row_index(initial["rows"])
    coverage = {
        method: _coverage_for_method(initial_index, method)
        for method in _INITIAL_METHODS
    }
    b_pairs = _pair_count(initial_index, "joint_B")
    b2_pairs = _pair_count(initial_index, "joint_2B")
    failed: list[str] = []
    if b_pairs != 80:
        failed.append(f"current_b_pairs_{b_pairs}_of_80")
    if b2_pairs != 80:
        failed.append(f"current_2b_pairs_{b2_pairs}_of_80")
    for method in ("joint_B", "joint_2B"):
        method_coverage = coverage[method]
        if method_coverage is None:
            failed.append(f"{method}_coverage_unavailable")
        elif not method_coverage["coverage_valid"]:
            failed.append(f"{method}_coverage_invalid")
    initial_valid = not failed

    bootstrap_indices = _bootstrap_indices(_REQUIRED_SEEDS, None)
    ratios: dict[str, Any] = {"joint_B": None, "joint_2B": None, "joint_8SP": None}
    delta: dict[str, Any] | None = None
    b_widths = _tail_widths(initial_index, "joint_B")
    b2_widths = _tail_widths(initial_index, "joint_2B")
    if b_widths is not None:
        ratios["joint_B"] = paired_ratio_summary(
            *b_widths, bootstrap_indices=bootstrap_indices
        )
    if b2_widths is not None:
        ratios["joint_2B"] = paired_ratio_summary(
            *b2_widths, bootstrap_indices=bootstrap_indices
        )
    if b_widths is not None and b2_widths is not None:
        delta = paired_delta_summary(
            b_widths[0],
            b2_widths[0],
            b_widths[1],
            bootstrap_indices=bootstrap_indices,
        )

    numeric_initial_decision = decide_initial(
        valid=initial_valid,
        r_2b=(math.nan if ratios["joint_2B"] is None else ratios["joint_2B"]["point_ratio"]),
        delta_b=(math.nan if delta is None else delta["point_gain"]),
    )
    eligibility = _extension_eligibility(initial)
    decision = numeric_initial_decision
    if decision == "EXTENSION_8SP_REQUIRED" and not eligibility["all_eligible"]:
        failed.append(
            "extension_eligibility_"
            f"{eligibility['eligible_scenario_seed_count']}_of_80"
        )
        decision = "STOP_SCALAR_UNAVAILABLE"

    extension: dict[str, Any] | None = None
    all_rows = list(initial["rows"])
    analysis_phase = "initial"
    if extension_dir is not None:
        extension = _validated_study(
            Path(extension_dir), kind="extension-8sp", parent_root=initial["root"]
        )
        _validate_extension_parent(initial, extension)
        extension_index = _row_index(extension["rows"])
        combined_index = {**initial_index, **extension_index}
        extension_coverage = _coverage_for_method(extension_index, "joint_8SP")
        coverage["joint_8SP"] = extension_coverage
        extension_pairs = _pair_count(combined_index, "joint_8SP")
        extension_failed: list[str] = []
        if extension_pairs != 80:
            extension_failed.append(f"current_8sp_pairs_{extension_pairs}_of_80")
        if extension_coverage is None:
            extension_failed.append("joint_8SP_coverage_unavailable")
        elif not extension_coverage["coverage_valid"]:
            extension_failed.append("joint_8SP_coverage_invalid")
        extension_widths = _tail_widths(combined_index, "joint_8SP")
        if extension_widths is not None:
            ratios["joint_8SP"] = paired_ratio_summary(
                *extension_widths, bootstrap_indices=bootstrap_indices
            )
        extension_valid = not extension_failed and ratios["joint_8SP"] is not None
        decision = decide_extension(
            valid=extension_valid,
            r_8sp=(
                math.nan
                if ratios["joint_8SP"] is None
                else ratios["joint_8SP"]["point_ratio"]
            ),
        )
        failed.extend(extension_failed)
        all_rows.extend(extension["rows"])
        analysis_phase = "final"
    else:
        coverage["joint_8SP"] = None

    parent_manifest_sha = _file_sha256(initial["root"] / "study_manifest.json")
    convergence_gain_by_seed: list[dict[str, float | int]] = []
    if b_widths is not None and b2_widths is not None:
        convergence_gain_by_seed = [
            {"seed": seed, "value": float(1.0 - width_2b / width_b)}
            for seed, width_b, width_2b in zip(
                _SEEDS, b_widths[0], b2_widths[0], strict=True
            )
        ]
    return {
        "analysis_phase": analysis_phase,
        "decision": decision,
        "numeric_initial_decision": numeric_initial_decision,
        "ordered_seeds": list(_SEEDS),
        "provenance": {
            "parent_study_manifest_sha256": parent_manifest_sha,
            "source_tree_sha256": initial["manifest"]["source_tree_sha256"],
            "experiment_tree_sha256": initial["manifest"]["experiment_tree_sha256"],
            "config_sha256": initial["manifest"]["config_sha256"],
            "initial_execution_sha256": initial["execution"]["execution_sha256"],
            "extension_study_manifest_sha256": (
                None
                if extension is None
                else _file_sha256(extension["root"] / "study_manifest.json")
            ),
        },
        "validity": {
            "initial_valid": initial_valid,
            "current_b_pairs": b_pairs,
            "current_2b_pairs": b2_pairs,
            "required_pairs_each": 80,
            "failed_primitives": failed,
        },
        "coverage": coverage,
        "ratios": ratios,
        "delta_b": delta,
        "convergence_gain_by_seed": convergence_gain_by_seed,
        "extension_eligibility": eligibility,
        "audit": _audit_summary(
            initial["rows"],
            initial["measurements"],
            None if extension is None else extension["rows"],
            None if extension is None else extension["measurements"],
        ),
        "_source_rows": all_rows,
        "_runner_measurements": {
            "initial": initial["measurements"],
            "extension": None if extension is None else extension["measurements"],
        },
        "_input_dir": str(initial["root"]),
        "_extension_dir": None if extension is None else str(extension["root"]),
    }


def _display_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return str(value)


def _summary_rows(analysis: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(
        row_type: str,
        *,
        checkpoint: str = "",
        scenario: str = "",
        stage: str = "",
        metric: str,
        value: object,
        numerator: object = "",
        denominator: object = "",
        detail: object = "",
    ) -> None:
        rows.append(
            {
                "row_type": row_type,
                "checkpoint": checkpoint,
                "scenario": scenario,
                "stage": stage,
                "metric": metric,
                "value": _display_value(value),
                "numerator": _display_value(numerator),
                "denominator": _display_value(denominator),
                "detail": (
                    json.dumps(detail, sort_keys=True, separators=(",", ":"))
                    if isinstance(detail, (dict, list))
                    else _display_value(detail)
                ),
            }
        )

    add("decision", metric="decision", value=analysis["decision"])
    for method, summary in analysis["coverage"].items():
        if summary is None:
            add("coverage_metric", checkpoint=method, metric="availability", value="unavailable")
            continue
        for scenario_index, scenario in enumerate(_SCENARIOS):
            for stage_index in range(12):
                add(
                    "coverage_cell",
                    checkpoint=method,
                    scenario=scenario,
                    stage=str(stage_index + 1),
                    metric="stage_seed_mean",
                    value=summary["stage_seed_mean"][scenario_index][stage_index],
                    denominator=40,
                )
                add(
                    "coverage_cell",
                    checkpoint=method,
                    scenario=scenario,
                    stage=str(stage_index + 1),
                    metric="simultaneous_lcb",
                    value=summary["simultaneous_lcb"][scenario_index][stage_index],
                    denominator=40,
                )
        for metric in (
            "minimum_stage_seed_mean",
            "minimum_simultaneous_lcb",
            "mean_seedwise_stage_minimum",
            "raw_seed_stage_minimum",
            "coverage_valid",
        ):
            add("coverage_metric", checkpoint=method, metric=metric, value=summary[metric])
    for method, summary in analysis["ratios"].items():
        if summary is None:
            add("ratio", checkpoint=method, metric="availability", value="unavailable")
            continue
        for metric, value in summary.items():
            add(
                "ratio",
                checkpoint=method,
                metric=metric,
                value=value,
                denominator=summary["n_pairs"],
            )
    if analysis["delta_b"] is not None:
        for metric, value in analysis["delta_b"].items():
            add(
                "delta",
                checkpoint="B_to_2B",
                metric=metric,
                value=value,
                denominator=analysis["delta_b"]["n_pairs"],
            )
    for item in analysis["convergence_gain_by_seed"]:
        add(
            "convergence_seed",
            checkpoint="B_to_2B",
            scenario="tail_shift",
            metric="one_minus_width_2b_over_width_b",
            value=item["value"],
            detail={"seed": item["seed"]},
        )
    for metric, value in analysis["validity"].items():
        add(
            "validity",
            metric=metric,
            value=value if metric != "failed_primitives" else "",
            detail=value if metric == "failed_primitives" else "",
        )
    for metric, value in analysis["extension_eligibility"].items():
        add("eligibility", metric=metric, value=value)
    for method, checkpoint in analysis["audit"]["checkpoints"].items():
        for scenario, values in checkpoint["by_scenario"].items():
            for metric, value in values.items():
                add(
                    "checkpoint_audit",
                    checkpoint=method,
                    scenario=scenario,
                    metric=metric,
                    value=value,
                )
    for phase, runner in analysis["audit"]["runner"].items():
        if runner is None:
            continue
        add("runner_audit", checkpoint=phase, metric="n_seed_runs", value=runner["n_seed_runs"])
        for metric, value in runner["elapsed_seconds"].items():
            add("runner_audit", checkpoint=phase, metric=f"elapsed_seconds_{metric}", value=value)
        add(
            "runner_audit",
            checkpoint=phase,
            metric="max_memory_allocated_bytes",
            value=runner["max_memory_allocated_bytes"],
        )
        add(
            "runner_audit",
            checkpoint=phase,
            metric="max_memory_reserved_bytes",
            value=runner["max_memory_reserved_bytes"],
        )
    return rows


def _write_csv(path: Path, analysis: dict[str, Any]) -> None:
    columns = (
        "row_type",
        "checkpoint",
        "scenario",
        "stage",
        "metric",
        "value",
        "numerator",
        "denominator",
        "detail",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_summary_rows(analysis))


def _markdown_report(analysis: dict[str, Any]) -> str:
    checkpoint = (
        "joint_2B" if analysis["analysis_phase"] == "initial" else "joint_8SP"
    )
    coverage = analysis["coverage"][checkpoint]
    checkpoint_label = checkpoint.replace("joint_", "")

    def coverage_value(field: str) -> str:
        if coverage is None:
            return "unavailable"
        return f"{coverage[field]:.6f}"

    ratio_lines = []
    for method, label in (("joint_B", "R_B"), ("joint_2B", "R_2B")):
        summary = analysis["ratios"][method]
        value = "unavailable" if summary is None else f"{summary['point_ratio']:.6f}"
        ratio_lines.append(f"- {label}: {value}.")
    ratio_8sp = analysis["ratios"]["joint_8SP"]
    if ratio_8sp is not None:
        ratio_lines.append(f"- R_8SP: {ratio_8sp['point_ratio']:.6f}.")
    elif analysis["analysis_phase"] == "initial":
        ratio_lines.append("- R_8SP: not run in the initial analysis.")
    else:
        ratio_lines.append("- R_8SP: unavailable in the final analysis.")
    delta = analysis["delta_b"]
    delta_value = "unavailable" if delta is None else f"{delta['point_gain']:.6f}"

    return (
        "# Phase 0C joint-search audit\n\n"
        f"Decision: `{analysis['decision']}`.\n\n"
        "This fixed-T, all-active Oracle diagnostic uses 40 development seeds. "
        "It is not deployable, is not a confirmation result, and is not a "
        "state-of-the-art claim. A promising machine decision would only "
        "authorize separate practical-method development.\n\n"
        "## Coverage\n\n"
        f"The latest attempted checkpoint is {checkpoint_label}. Standard and "
        "tail-shift coverage each use 40/40 seed schedules when available. The "
        "registered family contains 24 stage-by-scenario cells, so validity "
        "across both scenarios requires 80/80 scenario-seed schedules. That "
        "80/80 value is a validity denominator, not a width-ratio denominator. "
        "Four summaries answer different questions:\n\n"
        f"- minimum cell seed mean: {coverage_value('minimum_stage_seed_mean')} "
        "(average seeds within each cell, then take the minimum);\n"
        f"- minimum simultaneous LCB: "
        f"{coverage_value('minimum_simultaneous_lcb')} "
        "(take the minimum adjusted lower bound across cells);\n"
        f"- mean seedwise cell minimum: "
        f"{coverage_value('mean_seedwise_stage_minimum')} "
        "(take each seed's minimum cell, then average seeds);\n"
        f"- raw seed-cell minimum: {coverage_value('raw_seed_stage_minimum')} "
        "(take the minimum over every unaggregated seed-cell value).\n\n"
        "Only the simultaneous LCB gates coverage. The other three minima are "
        "descriptive and need not identify the same stage, scenario, or seed.\n\n"
        "## Width and convergence\n\n"
        "Width ratios are tail-only 40-pair geometric means against the current "
        "profiled schedule:\n\n"
        + "\n".join(ratio_lines)
        + "\n\n"
        "The registered convergence statistic is "
        f"Delta_B = 1 - exp(mean(log(W_2B/W_B))) = {delta_value}. The plotted "
        "40 seedwise gains are a distributional audit; their arithmetic mean "
        "does not replace the registered statistic. Bootstrap intervals are "
        "descriptive and do not alter the decision.\n\n"
        "## Feasibility and resource audit\n\n"
        "Winner counts use available schedules as their denominator. Endpoint "
        "use is counted over 12 × available schedules. Runtime uses one runner "
        "elapsed time per seed, and GPU peak memory is the maximum across seed "
        "processes; shared work is never summed across checkpoints or devices.\n"
    )


def _build_diagnostic_figure(analysis: dict[str, Any]) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "svg.hashsalt": "phase0c-joint-search-v1",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(183.0 / 25.4, 120.0 / 25.4),
        gridspec_kw={"height_ratios": (1.05, 0.95)},
    )
    figure.subplots_adjust(
        left=0.085, right=0.98, bottom=0.11, top=0.76, wspace=0.34, hspace=0.58
    )
    coverage_axis, ratio_axis, convergence_axis, audit_axis = axes.flat
    figure.suptitle(analysis["decision"], y=0.985, fontsize=10, fontweight="bold")
    figure.text(
        0.5,
        0.935,
        "Fixed-T joint scalar search · Oracle diagnostic",
        ha="center",
        va="center",
        fontsize=7,
        color="#4D4D4D",
    )

    checkpoint = (
        "joint_2B" if analysis["analysis_phase"] == "initial" else "joint_8SP"
    )
    coverage = analysis["coverage"][checkpoint]
    stage = np.arange(1, 13)
    coverage_handles = []
    coverage_labels = []
    scenario_styles = (
        ("standard", "#0F4D92", "o"),
        ("tail shift", "#B97824", "s"),
    )
    coverage_values = [0.90]
    if coverage is None:
        coverage_axis.text(
            0.5,
            0.62,
            f"{checkpoint.replace('joint_', '')} unavailable",
            transform=coverage_axis.transAxes,
            ha="center",
            va="center",
            color="#606060",
        )
    else:
        for index, (scenario, color, marker) in enumerate(scenario_styles):
            mean = np.asarray(coverage["stage_seed_mean"][index], dtype=np.float64)
            lcb = np.asarray(coverage["simultaneous_lcb"][index], dtype=np.float64)
            coverage_values.extend(mean.tolist())
            coverage_values.extend(lcb.tolist())
            mean_line = coverage_axis.plot(
                stage,
                mean,
                color=color,
                marker=marker,
                markersize=2.8,
                linewidth=1.0,
            )[0]
            lcb_line = coverage_axis.plot(
                stage,
                lcb,
                color=color,
                marker=marker,
                markerfacecolor="white",
                markersize=2.8,
                linewidth=0.9,
                linestyle="--",
            )[0]
            coverage_handles.extend((mean_line, lcb_line))
            coverage_labels.extend(
                (f"{scenario} mean (40/40)", f"{scenario} simultaneous LCB")
            )
    target_line = coverage_axis.axhline(
        0.90, color="#4D4D4D", linewidth=0.9, linestyle=":"
    )
    coverage_handles.append(target_line)
    coverage_labels.append("target 0.90")
    coverage_axis.set_xlim(0.5, 12.5)
    coverage_axis.set_xticks((1, 4, 8, 12))
    coverage_axis.set_xlabel("Stage")
    coverage_axis.set_ylabel("Coverage")
    coverage_axis.set_title(
        f"{checkpoint.replace('joint_', '')} stagewise coverage", loc="left"
    )
    coverage_axis.set_ylim(
        max(0.0, min(coverage_values) - 0.01),
        min(1.0, max(coverage_values) + 0.01),
    )
    figure.legend(
        coverage_handles,
        coverage_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=3,
        frameon=False,
        fontsize=5.5,
        handlelength=2.2,
        columnspacing=1.2,
    )

    ratio_order = (("joint_B", "B"), ("joint_2B", "2B"), ("joint_8SP", "8SP"))
    ratio_values = [0.92, 1.0]
    for x_position, (method, label) in enumerate(ratio_order):
        ratio = analysis["ratios"][method]
        if ratio is None:
            state = (
                "not run"
                if method == "joint_8SP" and analysis["analysis_phase"] == "initial"
                else "unavailable"
            )
            horizontal_alignment = ("left", "center", "right")[x_position]
            ratio_axis.text(
                x_position,
                0.975,
                state,
                ha=horizontal_alignment,
                va="center",
                fontsize=5.5,
                color="#767676",
            )
            continue
        value = float(ratio["point_ratio"])
        ratio_values.append(value)
        ratio_axis.scatter(
            x_position,
            value,
            s=22,
            color="#0F4D92",
            edgecolor="#272727",
            linewidth=0.5,
            zorder=3,
        )
    ratio_axis.axhline(0.92, color="#B97824", linestyle="--", linewidth=0.9)
    ratio_axis.axhline(1.0, color="#4D4D4D", linestyle=":", linewidth=0.9)
    ratio_axis.set_yscale("log")
    ratio_axis.set_ylim(min(ratio_values) * 0.96, max(ratio_values) * 1.04)
    ratio_axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ratio_axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ratio_axis.yaxis.set_minor_formatter(ticker.NullFormatter())
    ratio_axis.set_xticks(range(3), [label for _, label in ratio_order])
    ratio_axis.set_ylabel("Tail width ratio")
    ratio_axis.set_title("Paired geometric width ratio (40 pairs)", loc="left")

    gains = [item["value"] for item in analysis["convergence_gain_by_seed"]]
    if gains:
        convergence_axis.scatter(
            np.linspace(-0.20, 0.20, len(gains)),
            gains,
            s=8,
            facecolor="white",
            edgecolor="#0F4D92",
            linewidth=0.6,
        )
    else:
        convergence_axis.text(
            0.5,
            0.5,
            "unavailable",
            transform=convergence_axis.transAxes,
            ha="center",
            va="center",
            color="#606060",
        )
    convergence_axis.axhline(
        0.005, color="#B97824", linestyle="--", linewidth=0.9
    )
    if analysis["delta_b"] is not None:
        convergence_axis.text(
            0.98,
            0.86,
            f"Registered Delta_B = {analysis['delta_b']['point_gain']:.4f}",
            transform=convergence_axis.transAxes,
            ha="right",
            va="top",
            fontsize=5.6,
            color="#272727",
        )
    convergence_axis.set_xlim(-0.28, 0.28)
    convergence_axis.set_xticks([])
    convergence_axis.set_ylabel(r"Seed gain $1-W_{2B}/W_B$")
    convergence_axis.set_title("B to 2B convergence (40 seeds)", loc="left")

    checkpoint_audit = analysis["audit"]["checkpoints"].get(checkpoint)
    runner_phase = (
        "initial" if analysis["analysis_phase"] == "initial" else "extension"
    )
    runner = analysis["audit"]["runner"][runner_phase]
    audit_lines = [f"Checkpoint: {checkpoint.replace('joint_', '')}"]
    if checkpoint_audit is None:
        audit_lines.append("Audit: unavailable")
    else:
        for scenario, values in checkpoint_audit["by_scenario"].items():
            winners = values["winner_counts"]
            audit_lines.append(
                f"{scenario}: available {values['available_seed_count']}/40; "
                f"feasible {values['joint_feasible_seed_count']}/40"
            )
            audit_lines.append(
                f"  winner denominator {values['winner_denominator']}; P/G/E "
                f"{winners['profiled']}/{winners['greedy']}/"
                f"{winners['upper_endpoint']}"
            )
            audit_lines.append(
                f"  endpoint {values['endpoint_stage_count']}/"
                f"{values['endpoint_stage_denominator']} stages"
            )
    if runner is not None:
        audit_lines.extend(
            (
                f"Runtime median {runner['elapsed_seconds']['median']:.1f} s; "
                f"one value × {runner['n_seed_runs']} seed runs",
                "Peak GPU reserved memory "
                f"{runner['max_memory_reserved_bytes'] / 2**20:.0f} MiB; "
                "maximum across seed processes",
            )
        )
    audit_axis.axis("off")
    audit_axis.set_title("Feasibility and resource audit", loc="left")
    audit_axis.text(
        0.0,
        0.98,
        "\n".join(audit_lines),
        transform=audit_axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        linespacing=1.35,
    )
    for axis in (coverage_axis, ratio_axis, convergence_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.7)
        axis.tick_params(labelsize=5.8)
        axis.xaxis.label.set_size(6.2)
        axis.yaxis.label.set_size(6.2)
        axis.title.set_size(6.8)
    audit_axis.title.set_size(6.8)
    return figure


def _render_figure(analysis: dict[str, Any], output_dir: Path) -> None:
    figure = _build_diagnostic_figure(analysis)
    import matplotlib.pyplot as plt

    try:
        fixed_time = datetime(2026, 8, 18, tzinfo=timezone.utc)
        figure.savefig(
            output_dir / "phase0c_joint_search.pdf",
            metadata={
                "Title": "Phase 0C joint-search audit",
                "Creator": "SC-PCP",
                "CreationDate": fixed_time,
                "ModDate": fixed_time,
            },
        )
        figure.savefig(
            output_dir / "phase0c_joint_search.svg",
            metadata={"Title": "Phase 0C joint-search audit", "Date": "2026-08-18"},
        )
        figure.savefig(
            output_dir / "phase0c_joint_search.png",
            dpi=300,
            metadata={"Software": "SC-PCP", "Creation Time": "2026-08-18"},
        )
    finally:
        plt.close(figure)


def _decision_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    provenance = analysis["provenance"]
    payload = {
        "protocol": "phase0c_joint_search_summary_v1",
        "analysis_phase": analysis["analysis_phase"],
        "decision": analysis["decision"],
        "parent_study_manifest_sha256": provenance[
            "parent_study_manifest_sha256"
        ],
        "ordered_seeds": analysis["ordered_seeds"],
        "source_tree_sha256": provenance["source_tree_sha256"],
        "experiment_tree_sha256": provenance["experiment_tree_sha256"],
        "config_sha256": provenance["config_sha256"],
        "extension_eligibility": analysis["extension_eligibility"],
    }
    if analysis["analysis_phase"] == "final":
        payload["extension_study_manifest_sha256"] = provenance[
            "extension_study_manifest_sha256"
        ]
    return payload


def _file_fact(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _write_bundle(stage: Path, analysis: dict[str, Any]) -> None:
    _write_csv(stage / "phase0c_summary.csv", analysis)
    (stage / "phase0c_decision.json").write_text(
        json.dumps(
            _decision_payload(analysis), sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    (stage / "phase0c_summary.md").write_text(
        _markdown_report(analysis), encoding="utf-8"
    )
    _render_figure(analysis, stage)
    manifest = {
        "protocol": "phase0c_joint_search_summary_manifest_v1",
        "status": "complete",
        "analysis_phase": analysis["analysis_phase"],
        "decision": analysis["decision"],
        "parent_study_manifest_sha256": analysis["provenance"][
            "parent_study_manifest_sha256"
        ],
        "files": {
            name: _file_fact(stage / name) for name in sorted(_PAYLOAD_NAMES)
        },
    }
    if analysis["analysis_phase"] == "final":
        manifest["extension_study_manifest_sha256"] = analysis["provenance"][
            "extension_study_manifest_sha256"
        ]
    (stage / "phase0c_summary_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_summary_bundle(output_dir: Path | str) -> Path:
    output = Path(output_dir)
    expected_names = set(_PAYLOAD_NAMES) | {"phase0c_summary_manifest.json"}
    if not output.is_dir() or {path.name for path in output.iterdir()} != expected_names:
        raise RuntimeError("summary bundle has the wrong exact file set")
    manifest = _read_json(
        output / "phase0c_summary_manifest.json", label="summary manifest"
    )
    manifest_phase = manifest.get("analysis_phase")
    expected_fields = {
        "protocol",
        "status",
        "analysis_phase",
        "decision",
        "parent_study_manifest_sha256",
        "files",
    }
    if manifest_phase == "final":
        expected_fields.add("extension_study_manifest_sha256")
    manifest_extension_hash = manifest.get("extension_study_manifest_sha256")
    if set(manifest) != expected_fields or manifest["protocol"] != (
        "phase0c_joint_search_summary_manifest_v1"
    ) or manifest["status"] != "complete" or manifest_phase not in {
        "initial",
        "final",
    } or (
        manifest_phase == "final"
        and (
            type(manifest_extension_hash) is not str
            or len(manifest_extension_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in manifest_extension_hash
            )
        )
    ):
        raise RuntimeError("summary manifest contract is invalid")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(_PAYLOAD_NAMES):
        raise RuntimeError("summary manifest payload file set differs")
    for name, fact in files.items():
        if (
            not isinstance(fact, dict)
            or set(fact) != {"bytes", "sha256"}
            or type(fact["bytes"]) is not int
            or fact["bytes"] < 0
            or type(fact["sha256"]) is not str
            or len(fact["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in fact["sha256"])
            or fact != _file_fact(output / name)
        ):
            raise RuntimeError(f"summary manifest {name} bytes/hash mismatch")
    decision = _read_json(output / "phase0c_decision.json", label="summary decision")
    phase = decision.get("analysis_phase")
    expected_decision_fields = {
        "protocol",
        "analysis_phase",
        "decision",
        "parent_study_manifest_sha256",
        "ordered_seeds",
        "source_tree_sha256",
        "experiment_tree_sha256",
        "config_sha256",
        "extension_eligibility",
    }
    if phase == "final":
        expected_decision_fields.add("extension_study_manifest_sha256")
    eligibility = decision.get("extension_eligibility")
    decision_literal = decision.get("decision")
    allowed_decisions = {
        "initial": {
            "PROMISING_ORACLE_DIAGNOSTIC",
            "STOP_SCALAR_SATURATED",
            "EXTENSION_8SP_REQUIRED",
            "STOP_SCALAR_UNAVAILABLE",
        },
        "final": {
            "PROMISING_ORACLE_DIAGNOSTIC",
            "STOP_SCALAR_INSUFFICIENT",
            "STOP_SCALAR_UNAVAILABLE",
        },
    }
    expected_eligibility_fields = {
        "all_eligible",
        "eligible_scenario_seed_count",
        "required_scenario_seed_count",
        "canonical_state_hash_count",
        "state_hash_manifest_sha256",
    }
    hashes = (
        decision.get("parent_study_manifest_sha256"),
        decision.get("source_tree_sha256"),
        decision.get("experiment_tree_sha256"),
        decision.get("config_sha256"),
        (
            None
            if not isinstance(eligibility, dict)
            else eligibility.get("state_hash_manifest_sha256")
        ),
        *(
            (decision.get("extension_study_manifest_sha256"),)
            if phase == "final"
            else ()
        ),
    )
    eligibility_complete = (
        isinstance(eligibility, dict)
        and eligibility.get("eligible_scenario_seed_count") == 80
        and eligibility.get("canonical_state_hash_count") == 240
    )
    if (
        set(decision) != expected_decision_fields
        or decision.get("protocol") != "phase0c_joint_search_summary_v1"
        or phase not in allowed_decisions
        or decision_literal not in allowed_decisions.get(phase, set())
        or decision.get("ordered_seeds") != list(_SEEDS)
        or not isinstance(eligibility, dict)
        or set(eligibility) != expected_eligibility_fields
        or type(eligibility.get("all_eligible")) is not bool
        or any(
            type(eligibility.get(field)) is not int
            for field in (
                "eligible_scenario_seed_count",
                "required_scenario_seed_count",
                "canonical_state_hash_count",
            )
        )
        or eligibility.get("required_scenario_seed_count") != 80
        or eligibility.get("all_eligible") != eligibility_complete
        or (
            (phase == "final" or decision_literal == "EXTENSION_8SP_REQUIRED")
            and not eligibility_complete
        )
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        )
        or decision.get("decision") != manifest["decision"]
        or decision.get("analysis_phase") != manifest["analysis_phase"]
        or decision.get("parent_study_manifest_sha256")
        != manifest["parent_study_manifest_sha256"]
        or (
            phase == "final"
            and decision.get("extension_study_manifest_sha256")
            != manifest_extension_hash
        )
    ):
        raise RuntimeError("summary decision/manifest cross-check failed")
    return output


def publish_summary(
    analysis: dict[str, Any], output_dir: Path | str | None = None
) -> Path:
    """Publish all seven summary files with rollback-safe directory replacement."""

    if output_dir is None:
        output = Path(analysis["_input_dir"]) / (
            "checkpoint_analysis"
            if analysis["analysis_phase"] == "initial"
            else "final_analysis"
        )
    else:
        output = Path(output_dir)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
    had_prior = output.exists()
    try:
        if had_prior:
            validate_summary_bundle(output)
        _write_bundle(stage, analysis)
        validate_summary_bundle(stage)
        try:
            if had_prior:
                _replace_path(output, backup)
            _replace_path(stage, output)
            validate_summary_bundle(output)
        except BaseException as error:
            if had_prior and backup.exists() and output.exists():
                shutil.rmtree(output)
            if had_prior and backup.exists() and not output.exists():
                _replace_path(backup, output)
            elif not had_prior and output.exists():
                shutil.rmtree(output)
            raise RuntimeError(f"summary publish failed: {error}") from error
        if backup.exists():
            shutil.rmtree(backup)
        return output
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if backup.exists() and not output.exists():
            _replace_path(backup, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize the Phase 0C audit")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    analysis = load_validate_analyze(args.input_dir, args.extension_dir)
    output = publish_summary(analysis, args.output_dir)
    print(output)
    return output


if __name__ == "__main__":
    main()
