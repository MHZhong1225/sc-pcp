"""Run the fail-closed Phase 0C joint-search audit."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, NamedTuple
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import (
    _atomic_write_text,
    experiment_tree_sha256,
    mark_study_complete,
    mark_study_failed,
    source_tree_sha256,
    write_seed_result,
    write_study_metadata,
)
from scpcp.config import ExperimentConfig
from scpcp.device import resolve_devices
from scpcp.experiment import _paper_seed
from scpcp.phase0c_joint_search import SearchState
from scpcp.phase0c_study import (
    _state_sha256,
    run_phase0c_extension_seed,
    run_phase0c_seed,
)


DEFAULT_CONFIG = ROOT / "configs" / "phase0c_joint_search.yaml"
_RUNTIME_CONFIG_FIELDS = frozenset({"seeds", "devices", "output_dir"})
_SCENARIOS = ("standard", "tail_shift")
_START_NAMES = ("profiled", "greedy", "upper_endpoint")
_INITIAL_METHODS = ("current_profiled", "greedy", "joint_B", "joint_2B")
_METHOD_SPECS = {
    "current_profiled": ("reference", "REFERENCE", 0, 1),
    "greedy": ("reference", "REFERENCE", 0, 12),
    "joint_B": ("joint_search", "B", 2, 12),
    "joint_2B": ("joint_search", "2B", 4, 12),
    "joint_8SP": ("joint_search", "8SP", 8, 12),
}
_SURFACE_FIELDS = {
    "schedule": "q_by_time_json",
    "tuning_coverage": "tuning_coverage_json",
    "tuning_stage_width": "tuning_stage_width_json",
    "final_coverage": "final_coverage_json",
    "final_wilson_lcb": "final_wilson_lcb_json",
    "final_stage_width": "final_stage_width_json",
}
_STATE_SUFFIXES = (
    "radii",
    "stage_grid_indices",
    "coverage",
    "normalized_width",
    "completed_sweep_pairs",
    "converged_at_pair",
)
_RECORD_COLUMNS = (
    "schema_version",
    "seed",
    "scenario",
    "method_id",
    "analysis_role",
    "budget_id",
    "sweep_pairs",
    "selection_status",
    "selection_available",
    "tuning_joint_feasible",
    "failure_reason",
    "chosen_initialization",
    "selected_endpoint_stage_count",
    "selected_stage_grid_indices_json",
    "q_by_time_json",
    "tuning_coverage_json",
    "tuning_stage_width_json",
    "tuning_micro_width",
    "final_coverage_json",
    "final_wilson_lcb_json",
    "final_stage_width_json",
    "micro_normalized_width",
    "patient_normalized_width",
    "tuning_stream_id",
    "evaluation_stream_id",
    "n_tuning_rollouts",
    "n_evaluation_rollouts",
    "schedule_evaluations",
    "committed_updates",
    "converged_at_pair",
    "wall_time_seconds",
)
_SMOKE_FIELDS = {
    "protocol",
    "seed",
    "max_sweep_pairs",
    "elapsed_seconds",
    "max_memory_allocated_bytes",
    "max_memory_reserved_bytes",
    "recommended_max_seed_wall_seconds",
    "source_tree_sha256",
    "experiment_tree_sha256",
    "config_sha256",
}
_SEED_DIRECTORY = re.compile(r"seed_(\d{5})")
_HEX = re.compile(r"[0-9a-f]{64}")
_SMOKE_EXECUTION_CAP_SECONDS = 86_400.0
_RUNNER_MEASUREMENT_FIELDS = {
    "protocol",
    "elapsed_seconds",
    "max_memory_allocated_bytes",
    "max_memory_reserved_bytes",
}


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--workers-per-device", type=_positive_integer, default=1)
    parser.add_argument("--candidate-chunk-size", type=_positive_integer, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 0C joint-search audit")
    modes = parser.add_subparsers(dest="mode", required=True)

    smoke = modes.add_parser("smoke")
    _add_common_arguments(smoke)
    smoke.add_argument("--seed", type=int, default=9999)

    initial = modes.add_parser("initial")
    _add_common_arguments(initial)
    initial.add_argument("--smoke-manifest", type=Path, required=True)

    extension = modes.add_parser("extension-8sp")
    _add_common_arguments(extension)
    extension.add_argument("--parent-dir", type=Path, required=True)
    extension.add_argument("--decision-json", type=Path, required=True)
    return parser


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_config_sha256(config: dict[str, Any]) -> str:
    """Hash scientific settings while excluding the three runtime-only fields."""

    scientific = {
        key: value for key, value in config.items() if key not in _RUNTIME_CONFIG_FIELDS
    }
    return _canonical_sha256(scientific)


def runtime_config_sha256(config: dict[str, Any]) -> str:
    """Hash the complete resolved runtime configuration."""

    return _canonical_sha256(config)


def calibrate_wall_cap(elapsed_seconds: float) -> int:
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0.0
    ):
        raise ValueError("smoke elapsed_seconds must be finite and positive")
    return max(300, math.ceil(1.5 * elapsed_seconds / 300.0) * 300)


def parse_seeds(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        seeds = default
    elif not value:
        seeds = ()
    elif value.count(":") == 1:
        start_text, stop_text = value.split(":")
        if not start_text or not stop_text:
            raise ValueError("seed ranges require both start and stop")
        start, stop = int(start_text), int(stop_text)
        if stop <= start:
            raise ValueError("seed range stop must exceed start")
        seeds = tuple(range(start, stop))
    elif ":" in value:
        raise ValueError("seed range must contain one colon")
    else:
        parts = value.split(",")
        if any(not part for part in parts):
            raise ValueError("comma-separated seeds must not contain empty values")
        seeds = tuple(int(part) for part in parts)
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be lowercase SHA256 hex")
    return value


def validate_smoke_manifest(
    path: Path,
    *,
    source_hash: str,
    experiment_hash: str,
    config_hash: str,
) -> int:
    payload = _load_json_object(path, label="smoke manifest")
    if set(payload) != _SMOKE_FIELDS:
        raise RuntimeError("smoke manifest must contain the exact fields")
    exact = {
        "protocol": "phase0c_smoke_v1",
        "seed": 9999,
        "max_sweep_pairs": 4,
        "source_tree_sha256": source_hash,
        "experiment_tree_sha256": experiment_hash,
        "config_sha256": config_hash,
    }
    for key, expected in exact.items():
        if type(payload.get(key)) is not type(expected) or payload.get(key) != expected:
            raise RuntimeError(f"smoke manifest {key} differs from the active protocol")
    elapsed = payload.get("elapsed_seconds")
    try:
        expected_cap = calibrate_wall_cap(elapsed)  # type: ignore[arg-type]
    except ValueError as error:
        raise RuntimeError(f"smoke manifest elapsed_seconds is invalid: {error}") from error
    for key in ("max_memory_allocated_bytes", "max_memory_reserved_bytes"):
        value = payload.get(key)
        if type(value) is not int or value < 0:
            raise RuntimeError(f"smoke manifest {key} must be a nonnegative integer")
    cap = payload.get("recommended_max_seed_wall_seconds")
    if type(cap) is not int or cap != expected_cap:
        raise RuntimeError("smoke manifest recommended wall cap is not derived from elapsed")
    return cap


def _seed_manifest_sha256(seeds: tuple[int, ...]) -> str:
    return _canonical_sha256({"ordered_seeds": list(seeds)})


def _execution_metadata(
    config: ExperimentConfig,
    *,
    mode: str,
    workers_per_device: int,
    candidate_chunk_size: int,
    max_seed_wall_seconds: float,
    source_hash: str,
    experiment_hash: str,
    parent_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_dict = config.to_dict()
    checkpoints = [2, 4] if mode in {"smoke", "initial"} else [8]
    execution: dict[str, Any] = {
        "protocol": "phase0c_execution_v1",
        "study_kind": mode,
        "source_tree_sha256": source_hash,
        "experiment_tree_sha256": experiment_hash,
        "config_sha256": canonical_config_sha256(config_dict),
        "runtime_config_sha256": runtime_config_sha256(config_dict),
        "ordered_seeds": list(config.seeds),
        "seed_manifest_sha256": _seed_manifest_sha256(config.seeds),
        "devices": list(config.devices),
        "output_dir": str(config.output_dir),
        "workers_per_device": workers_per_device,
        "candidate_chunk_size": candidate_chunk_size,
        "sweep_pair_checkpoints": checkpoints,
        "max_seed_wall_seconds": float(max_seed_wall_seconds),
    }
    if parent_fields:
        execution.update(parent_fields)
    execution["execution_sha256"] = _canonical_sha256(execution)
    return execution


def _validated_runner_measurement(
    value: object,
    *,
    seed: int,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _RUNNER_MEASUREMENT_FIELDS:
        raise RuntimeError(f"seed {seed} runner measurement has wrong exact fields")
    if value["protocol"] != "phase0c_runner_measurement_v1":
        raise RuntimeError(f"seed {seed} runner measurement protocol changed")
    elapsed = value["elapsed_seconds"]
    if type(elapsed) is not float or not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError(
            f"seed {seed} runner measurement elapsed_seconds must be finite and positive"
        )
    for field in ("max_memory_allocated_bytes", "max_memory_reserved_bytes"):
        measured = value[field]
        if type(measured) is not int or measured < 0:
            raise RuntimeError(
                f"seed {seed} runner measurement {field} must be a nonnegative integer"
            )
    return dict(value)


def _runner_result(
    result: Any,
    execution: dict[str, Any],
    measurement: dict[str, Any],
) -> Any:
    diagnostics = dict(result.diagnostics)
    if {"runner_provenance", "runner_measurement"} & diagnostics.keys():
        raise RuntimeError("seed diagnostics already contain runner-owned fields")
    diagnostics["runner_provenance"] = execution
    diagnostics["runner_measurement"] = _validated_runner_measurement(
        measurement, seed=result.seed
    )
    return replace(result, diagnostics=diagnostics)


def _file_fact(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _validate_file_facts(
    root: Path,
    files: object,
    expected_names: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(files, dict) or set(files) != expected_names:
        raise RuntimeError(f"{label} file set differs from the exact contract")
    for relative, expected in files.items():
        if not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}:
            raise RuntimeError(f"{label} {relative} bytes/hash fields are invalid")
        if type(expected["bytes"]) is not int or expected["bytes"] < 0:
            raise RuntimeError(f"{label} {relative} bytes must be an exact integer")
        _require_sha256(expected["sha256"], label=f"{label} {relative} sha256")
        if not (root / relative).is_file() or _file_fact(root / relative) != expected:
            raise RuntimeError(f"{label} {relative} bytes/hash mismatch")
    return files


def _validate_execution_integer_contract(execution: dict[str, Any]) -> None:
    seeds = execution.get("ordered_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise RuntimeError("execution ordered_seeds must contain exact integers")
    for field in ("workers_per_device", "candidate_chunk_size"):
        value = execution.get(field)
        if type(value) is not int or value < 1:
            raise RuntimeError(f"execution {field} must be a positive exact integer")
    checkpoints = execution.get("sweep_pair_checkpoints")
    if not isinstance(checkpoints, list) or any(
        type(value) is not int for value in checkpoints
    ):
        raise RuntimeError(
            "execution sweep_pair_checkpoints must contain exact integers"
        )
    expected_checkpoints = (
        [8] if execution.get("study_kind") == "extension-8sp" else [2, 4]
    )
    if checkpoints != expected_checkpoints:
        raise RuntimeError("execution sweep_pair_checkpoints changed")


def _json_vector(value: object, *, seed: int, field: str, length: int) -> np.ndarray:
    if type(value) is not str:
        raise RuntimeError(f"seed {seed} {field} must be JSON text")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"seed {seed} {field} is invalid JSON") from error
    if not isinstance(payload, list) or len(payload) != length:
        raise RuntimeError(f"seed {seed} {field} must have length {length}")
    array = np.asarray(payload)
    if length and (array.dtype.kind not in "fiu" or not np.isfinite(array).all()):
        raise RuntimeError(f"seed {seed} {field} must contain finite numbers")
    return array


def _finite_number(value: object, *, seed: int, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise RuntimeError(f"seed {seed} {field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"seed {seed} {field} must be finite")
    if positive and number <= 0.0:
        raise RuntimeError(f"seed {seed} {field} must be positive")
    return number


def _matches_float32_reduction(left: float, right: float, *, terms: int) -> bool:
    left32 = np.float32(left)
    right32 = np.float32(right)
    scale = max(abs(float(left32)), abs(float(right32)))
    roundoff_bound = float(np.finfo(np.float32).eps) * scale * terms
    return abs(float(left32) - float(right32)) <= roundoff_bound


def _pandas_integer(
    value: object,
    *,
    seed: int,
    field: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise RuntimeError(f"seed {seed} {field} must be an exact integer")
    parsed = int(value)
    if parsed < minimum or (maximum is not None and parsed > maximum):
        raise RuntimeError(f"seed {seed} {field} is outside its valid range")
    return parsed


def _pandas_boolean(value: object, *, seed: int, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise RuntimeError(f"seed {seed} {field} must be an exact boolean")
    return bool(value)


def _nullable_csv_integer(
    value: object,
    *,
    seed: int,
    field: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise RuntimeError(f"seed {seed} {field} must be an exact integer or null")
        value = int(number)
    return _pandas_integer(
        value,
        seed=seed,
        field=field,
        minimum=minimum,
        maximum=maximum,
    )


def _record_vectors(
    row: dict[str, Any], seed: int, method: str, length: int
) -> dict[str, np.ndarray]:
    return {
        suffix: _json_vector(
            row[field], seed=seed, field=f"{method} {suffix}", length=length
        )
        for suffix, field in _SURFACE_FIELDS.items()
    }


def _load_records(seed_dir: Path, seed: int, *, mode: str) -> pd.DataFrame:
    try:
        records = pd.read_csv(seed_dir / "records.csv")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise RuntimeError(f"seed {seed} records.csv is unreadable: {error}") from error
    if tuple(records.columns) != _RECORD_COLUMNS:
        raise RuntimeError(f"seed {seed} records.csv has the wrong exact columns")
    expected_methods = _INITIAL_METHODS if mode in {"initial", "smoke"} else ("joint_8SP",)
    expected_pairs = {
        (scenario, method) for scenario in _SCENARIOS for method in expected_methods
    }
    pairs = list(zip(records["scenario"], records["method_id"], strict=True))
    if len(records) != len(expected_pairs) or set(pairs) != expected_pairs or len(set(pairs)) != len(pairs):
        raise RuntimeError(f"seed {seed} records.csv has the wrong exact row keys")
    seed_column = records["seed"]
    if pd.api.types.is_bool_dtype(seed_column.dtype) or not pd.api.types.is_integer_dtype(
        seed_column.dtype
    ):
        raise RuntimeError(f"seed {seed} records.csv seed column must have integer dtype")
    if not seed_column.eq(seed).all():
        raise RuntimeError(f"seed {seed} records.csv contains a different seed ID")
    return records


def _validate_record_rows(records: pd.DataFrame, seed: int, *, mode: str) -> None:
    seen_streams: set[int] = set()
    for scenario_index, scenario in enumerate(_SCENARIOS):
        scenario_rows = records.loc[records["scenario"].eq(scenario)]
        expected_tuning = _paper_seed(seed, 1_300_001 + scenario_index)
        expected_evaluation = _paper_seed(seed, 1_400_001 + scenario_index)
        tuning_ids = {
            _pandas_integer(
                value,
                seed=seed,
                field=f"{scenario} tuning_stream_id",
                minimum=0,
            )
            for value in scenario_rows["tuning_stream_id"]
        }
        evaluation_ids = {
            _pandas_integer(
                value,
                seed=seed,
                field=f"{scenario} evaluation_stream_id",
                minimum=0,
            )
            for value in scenario_rows["evaluation_stream_id"]
        }
        if tuning_ids != {expected_tuning} or evaluation_ids != {expected_evaluation}:
            raise RuntimeError(f"seed {seed} {scenario} stream IDs changed")
        scenario_streams = tuning_ids | evaluation_ids
        if tuning_ids & evaluation_ids or seen_streams & scenario_streams:
            raise RuntimeError(f"seed {seed} contains a tuning/evaluation stream collision")
        seen_streams.update(scenario_streams)

    for row in records.to_dict(orient="records"):
        method = row["method_id"]
        if type(method) is not str or method not in _METHOD_SPECS:
            raise RuntimeError(f"seed {seed} method_id changed")
        analysis_role, budget_id, sweep_pairs, index_length = _METHOD_SPECS[method]
        if type(row["schema_version"]) is not str or row["schema_version"] != "phase0c_seed_v1":
            raise RuntimeError(f"seed {seed} schema_version changed")
        if (
            type(row["analysis_role"]) is not str
            or type(row["budget_id"]) is not str
            or row["analysis_role"] != analysis_role
            or row["budget_id"] != budget_id
        ):
            raise RuntimeError(f"seed {seed} {method} method metadata changed")
        if _pandas_integer(
            row["sweep_pairs"],
            seed=seed,
            field=f"{method} sweep_pairs",
            minimum=0,
            maximum=8,
        ) != sweep_pairs:
            raise RuntimeError(f"seed {seed} {method} sweep_pairs changed")
        status = row["selection_status"]
        if type(status) is not str or status not in {
            "SELECTED",
            "NO_FEASIBLE_START",
            "WALL_TIME_CAP",
        }:
            raise RuntimeError(f"seed {seed} {method} has an invalid selection status")
        available = _pandas_boolean(
            row["selection_available"],
            seed=seed,
            field=f"{method} selection_available",
        )
        if available != (status == "SELECTED"):
            raise RuntimeError(f"seed {seed} {method} availability disagrees with status")
        tuning_feasible = _pandas_boolean(
            row["tuning_joint_feasible"],
            seed=seed,
            field=f"{method} tuning_joint_feasible",
        )
        vector_length = 12 if available else 0
        indices = _json_vector(
            row["selected_stage_grid_indices_json"],
            seed=seed,
            field=f"{method} selected indices",
            length=index_length if available else 0,
        )
        minimum_index = -1 if method.startswith("joint_") else 0
        if available and (
            indices.dtype.kind not in "iu"
            or np.any(indices < minimum_index)
            or np.any(indices > 100)
        ):
            raise RuntimeError(f"seed {seed} {method} selected indices are invalid")
        endpoint_count = _pandas_integer(
            row["selected_endpoint_stage_count"],
            seed=seed,
            field=f"{method} selected endpoint count",
            minimum=0,
            maximum=12,
        )
        expected_endpoint_count = (
            int(np.count_nonzero((indices == 0) | (indices == 100)))
            if available
            else 0
        )
        if method == "current_profiled" and expected_endpoint_count:
            expected_endpoint_count = 12
        if endpoint_count != expected_endpoint_count:
            raise RuntimeError(f"seed {seed} {method} endpoint count disagrees")
        expected_initialization = {
            "current_profiled": "profiled",
            "greedy": "greedy",
        }.get(method)
        chosen = row["chosen_initialization"]
        if available:
            if type(chosen) is not str or (
                expected_initialization is not None
                and chosen != expected_initialization
            ) or (
                expected_initialization is None and chosen not in _START_NAMES
            ):
                raise RuntimeError(
                    f"seed {seed} {method} chosen_initialization is invalid"
                )
            if not pd.isna(row["failure_reason"]):
                raise RuntimeError(f"seed {seed} {method} failure_reason must be empty")
        else:
            if not pd.isna(chosen):
                raise RuntimeError(
                    f"seed {seed} unavailable {method} chosen_initialization must be empty"
                )
            if type(row["failure_reason"]) is not str or row["failure_reason"] != status:
                raise RuntimeError(
                    f"seed {seed} unavailable {method} failure_reason disagrees"
                )
        vectors = _record_vectors(row, seed, method, vector_length)
        q = vectors["schedule"]
        tuning_coverage = vectors["tuning_coverage"]
        tuning_width = vectors["tuning_stage_width"]
        final_coverage = vectors["final_coverage"]
        final_lcb = vectors["final_wilson_lcb"]
        final_width = vectors["final_stage_width"]
        expected_tuning_feasible = bool(
            available
            and np.all(
                np.asarray(tuning_coverage, dtype=np.float32)
                >= np.asarray(0.90, dtype=np.float32)
            )
        )
        if available:
            if np.any(q <= 0.0) or np.any(tuning_width <= 0.0) or np.any(final_width <= 0.0):
                raise RuntimeError(f"seed {seed} {method} widths and radii must be positive")
            if (
                np.any((tuning_coverage < 0.0) | (tuning_coverage > 1.0))
                or np.any((final_coverage < 0.0) | (final_coverage > 1.0))
                or np.any((final_lcb < 0.0) | (final_lcb > 1.0))
            ):
                raise RuntimeError(f"seed {seed} {method} coverage is outside [0,1]")
            coverage32 = np.asarray(tuning_coverage, dtype=np.float32)
            target32 = np.asarray(0.90, dtype=coverage32.dtype)
            if method.startswith("joint_") and np.any(coverage32 < target32):
                raise RuntimeError(
                    f"seed {seed} {method} selected tuning coverage is below .90"
                )
            if tuning_feasible != expected_tuning_feasible:
                raise RuntimeError(
                    f"seed {seed} {method} tuning_joint_feasible disagrees"
                )
            tuning_micro = _finite_number(
                row["tuning_micro_width"], seed=seed, field=f"{method} tuning micro width", positive=True
            )
            micro = _finite_number(
                row["micro_normalized_width"], seed=seed, field=f"{method} micro width", positive=True
            )
            patient = _finite_number(
                row["patient_normalized_width"], seed=seed, field=f"{method} patient width", positive=True
            )
            tuning_mean = float(
                np.asarray(tuning_width, dtype=np.float32).mean(dtype=np.float32)
            )
            final_mean = float(
                np.asarray(final_width, dtype=np.float32).mean(dtype=np.float32)
            )
            if not _matches_float32_reduction(
                tuning_micro, tuning_mean, terms=len(tuning_width)
            ):
                raise RuntimeError(f"seed {seed} {method} tuning micro width disagrees")
            if not _matches_float32_reduction(
                micro, final_mean, terms=len(final_width)
            ):
                raise RuntimeError(f"seed {seed} {method} micro width disagrees")
            if not _matches_float32_reduction(patient, micro, terms=len(final_width)):
                raise RuntimeError(f"seed {seed} {method} patient width disagrees")
            if _pandas_integer(
                row["n_evaluation_rollouts"],
                seed=seed,
                field=f"{method} evaluation rollout count",
                minimum=0,
            ) != 50_000:
                raise RuntimeError(f"seed {seed} {method} evaluation rollout count changed")
        else:
            if tuning_feasible != expected_tuning_feasible:
                raise RuntimeError(
                    f"seed {seed} {method} tuning_joint_feasible disagrees"
                )
            for field in (
                "tuning_micro_width",
                "micro_normalized_width",
                "patient_normalized_width",
            ):
                if not pd.isna(row[field]):
                    raise RuntimeError(f"seed {seed} unavailable {method} {field} must be NaN")
            if _pandas_integer(
                row["n_evaluation_rollouts"],
                seed=seed,
                field=f"{method} evaluation rollout count",
                minimum=0,
            ) != 0:
                raise RuntimeError(f"seed {seed} unavailable {method} has evaluation rollouts")
        if _pandas_integer(
            row["n_tuning_rollouts"],
            seed=seed,
            field=f"{method} tuning rollout count",
            minimum=0,
        ) != 5_000:
            raise RuntimeError(f"seed {seed} {method} tuning rollout count changed")
        search_counts = {
            field: _pandas_integer(
                row[field],
                seed=seed,
                field=f"{method} {field}",
                minimum=0,
            )
            for field in ("schedule_evaluations", "committed_updates")
        }
        if (method in {"current_profiled", "greedy"} or not available) and any(
            value != 0 for value in search_counts.values()
        ):
            raise RuntimeError(
                f"seed {seed} {method} unavailable/reference search counts must be exactly zero"
            )
        converged = _nullable_csv_integer(
            row["converged_at_pair"],
            seed=seed,
            field=f"{method} converged_at_pair",
            minimum=1,
            maximum=max(1, sweep_pairs),
        )
        if (method in {"current_profiled", "greedy"} or not available) and converged is not None:
            raise RuntimeError(f"seed {seed} {method} converged_at_pair must be null")
        wall_time = _finite_number(
            row["wall_time_seconds"], seed=seed, field=f"{method} wall time"
        )
        if wall_time < 0.0:
            raise RuntimeError(f"seed {seed} {method} wall time must be nonnegative")
        if method in {"current_profiled", "greedy"} and wall_time != 0.0:
            raise RuntimeError(f"seed {seed} {method} reference wall time must be zero")

    if mode in {"smoke", "initial"}:
        for scenario in _SCENARIOS:
            pair2_wall = _finite_number(
                _row(records, scenario, "joint_B")["wall_time_seconds"],
                seed=seed,
                field=f"{scenario} joint_B wall time",
            )
            pair4_wall = _finite_number(
                _row(records, scenario, "joint_2B")["wall_time_seconds"],
                seed=seed,
                field=f"{scenario} joint_2B wall time",
            )
            if pair2_wall != pair4_wall:
                raise RuntimeError(
                    f"seed {seed} {scenario} nested checkpoint wall times differ"
                )


def _load_npz(seed_dir: Path, seed: int) -> dict[str, np.ndarray]:
    try:
        with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:  # type: ignore[name-defined]
        raise RuntimeError(f"seed {seed} surfaces.npz is unreadable: {error}") from error


def _require_array(
    surfaces: dict[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    *,
    seed: int,
    finite: bool = True,
    dtype: object | None = None,
) -> np.ndarray:
    if name not in surfaces:
        raise RuntimeError(f"seed {seed} surfaces.npz is missing {name}")
    array = surfaces[name]
    if array.shape != shape:
        raise RuntimeError(f"seed {seed} {name} has shape {array.shape}, expected {shape}")
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise RuntimeError(
            f"seed {seed} {name} has dtype {array.dtype}, expected {np.dtype(dtype)}"
        )
    if array.dtype.kind == "O" or (finite and not np.isfinite(array).all()):
        raise RuntimeError(f"seed {seed} {name} must be finite numeric data")
    return array


def _row(records: pd.DataFrame, scenario: str, method: str) -> dict[str, Any]:
    selected = records.loc[
        records["scenario"].eq(scenario) & records["method_id"].eq(method)
    ]
    return selected.iloc[0].to_dict()


def _surface_vectors_for_row(
    surfaces: dict[str, np.ndarray],
    row: dict[str, Any],
    *,
    scenario: str,
    method: str,
    seed: int,
) -> None:
    available = bool(row["selection_available"])
    length = 12 if available else 0
    vectors = _record_vectors(row, seed, method, length)
    for suffix, expected in vectors.items():
        name = f"{scenario}_{method}_{suffix}"
        actual = _require_array(
            surfaces, name, (length,), seed=seed, dtype=np.float32
        )
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"seed {seed} records/NPZ disagrees for {name}")


def _schedule_from_stage_indices(
    indices: np.ndarray,
    stage_grids: np.ndarray,
    current_profiled: np.ndarray,
    *,
    seed: int,
    label: str,
) -> np.ndarray:
    schedule = np.empty(len(indices), dtype=stage_grids.dtype)
    for stage, index_value in enumerate(indices):
        index = int(index_value)
        if index == -1:
            if current_profiled.shape != (12,):
                raise RuntimeError(
                    f"seed {seed} {label} inherits an unavailable current profile"
                )
            schedule[stage] = current_profiled[stage]
        else:
            schedule[stage] = stage_grids[stage, index]
    return schedule


def _state_from_surfaces(
    surfaces: dict[str, np.ndarray],
    *,
    scenario: str,
    start_name: str,
    seed: int,
) -> SearchState:
    prefix = f"{scenario}_pair4_{start_name}"
    radii = _require_array(
        surfaces, f"{prefix}_radii", (12,), seed=seed, dtype=np.float32
    )
    indices = _require_array(
        surfaces,
        f"{prefix}_stage_grid_indices",
        (12,),
        seed=seed,
        dtype=np.int64,
    )
    if indices.dtype.kind not in "iu" or np.any(indices < -1) or np.any(indices > 100):
        raise RuntimeError(f"seed {seed} {prefix} indices are invalid")
    coverage = _require_array(
        surfaces, f"{prefix}_coverage", (12,), seed=seed, dtype=np.float32
    )
    width = _require_array(
        surfaces,
        f"{prefix}_normalized_width",
        (12,),
        seed=seed,
        dtype=np.float32,
    )
    completed = _require_array(
        surfaces,
        f"{prefix}_completed_sweep_pairs",
        (),
        seed=seed,
        dtype=np.int64,
    )
    converged = _require_array(
        surfaces,
        f"{prefix}_converged_at_pair",
        (),
        seed=seed,
        dtype=np.int64,
    )
    if completed.dtype.kind not in "iu" or int(completed) != 4:
        raise RuntimeError(f"seed {seed} {prefix} completed pair count changed")
    converged_value = int(converged)
    if converged.dtype.kind not in "iu" or converged_value not in {-1, 1, 2, 3, 4}:
        raise RuntimeError(f"seed {seed} {prefix} convergence metadata is invalid")
    if np.any(radii <= 0.0) or np.any(width <= 0.0) or np.any(coverage < 0.90):
        raise RuntimeError(f"seed {seed} {prefix} is not a valid feasible state")
    return SearchState(
        start_name=start_name,
        radii=torch.from_numpy(np.array(radii, copy=True)),
        stage_grid_indices=tuple(None if int(value) == -1 else int(value) for value in indices),
        coverage=torch.from_numpy(np.array(coverage, copy=True)),
        normalized_width=torch.from_numpy(np.array(width, copy=True)),
        completed_sweep_pairs=4,
        converged_at_pair=None if converged_value == -1 else converged_value,
    )


_TRACE_FIELDS = {
    "start_name",
    "sweep_pair",
    "direction",
    "stage",
    "feasible_count",
    "proposed_grid_index",
    "before_micro_width",
    "proposed_micro_width",
    "committed",
    "after_micro_width",
}
_CHECKPOINT_FIELDS = {
    "requested_sweep_pairs",
    "executed_sweep_pairs",
    "best_start_name",
    "schedule_evaluations",
    "committed_updates",
    "trace",
}


class _TraceFacts(NamedTuple):
    schedule_evaluations: int
    committed_updates: int
    commits_by_pair: dict[int, int]
    final_width_by_start: dict[str, float]


class _CheckpointFacts(NamedTuple):
    payload: dict[str, Any]
    trace: _TraceFacts


def _exact_int(
    value: object,
    *,
    seed: int,
    label: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise RuntimeError(f"seed {seed} {label} must be an exact integer in range")
    return value


def _positive_json_number(value: object, *, seed: int, label: str) -> float:
    if type(value) not in {int, float}:
        raise RuntimeError(f"seed {seed} {label} trace value must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"seed {seed} {label} trace value must be finite and positive")
    return number


def _validate_coordinate_trace(
    trace: object,
    *,
    seed: int,
    label: str,
    start_names: list[str] | tuple[str, ...],
    minimum_pair: int,
    maximum_pair: int,
) -> _TraceFacts:
    if not isinstance(trace, list):
        raise RuntimeError(f"seed {seed} {label} trace must be a list")
    expected_coordinates = [
        (start_name, sweep_pair, direction, stage)
        for sweep_pair in range(minimum_pair, maximum_pair + 1)
        for start_name in start_names
        for direction, stages in (
            ("forward", range(12)),
            ("reverse", range(11, -1, -1)),
        )
        for stage in stages
    ]
    if len(trace) != len(expected_coordinates):
        raise RuntimeError(f"seed {seed} {label} trace is not a complete canonical sequence")
    committed_count = 0
    commits_by_pair = {
        sweep_pair: 0 for sweep_pair in range(minimum_pair, maximum_pair + 1)
    }
    last_width: dict[str, float] = {}
    for step, expected_coordinate in zip(trace, expected_coordinates, strict=True):
        if not isinstance(step, dict) or set(step) != _TRACE_FIELDS:
            raise RuntimeError(f"seed {seed} {label} trace has wrong exact fields")
        if type(step["start_name"]) is not str or step["start_name"] not in start_names:
            raise RuntimeError(f"seed {seed} {label} trace start name is invalid")
        _exact_int(
            step["sweep_pair"],
            seed=seed,
            label=f"{label} trace sweep pair",
            minimum=minimum_pair,
            maximum=maximum_pair,
        )
        if step["direction"] not in {"forward", "reverse"}:
            raise RuntimeError(f"seed {seed} {label} trace direction is invalid")
        _exact_int(
            step["stage"],
            seed=seed,
            label=f"{label} trace stage",
            minimum=0,
            maximum=11,
        )
        coordinate = (
            step["start_name"],
            step["sweep_pair"],
            step["direction"],
            step["stage"],
        )
        if coordinate != expected_coordinate:
            raise RuntimeError(f"seed {seed} {label} trace order is not canonical")
        feasible_count = _exact_int(
            step["feasible_count"],
            seed=seed,
            label=f"{label} trace feasible count",
            minimum=0,
            maximum=101,
        )
        proposed_index = step["proposed_grid_index"]
        proposed_width = step["proposed_micro_width"]
        if feasible_count == 0:
            if proposed_index is not None or proposed_width is not None:
                raise RuntimeError(f"seed {seed} {label} trace infeasible proposal is invalid")
        else:
            _exact_int(
                proposed_index,
                seed=seed,
                label=f"{label} trace proposed grid index",
                minimum=0,
                maximum=100,
            )
            _positive_json_number(
                proposed_width,
                seed=seed,
                label=f"{label} proposed micro width",
            )
        before = _positive_json_number(
            step["before_micro_width"],
            seed=seed,
            label=f"{label} before micro width",
        )
        after = _positive_json_number(
            step["after_micro_width"],
            seed=seed,
            label=f"{label} after micro width",
        )
        committed = step["committed"]
        if type(committed) is not bool:
            raise RuntimeError(f"seed {seed} {label} trace committed must be bool")
        previous = last_width.get(step["start_name"])
        if previous is not None and before != previous:
            raise RuntimeError(f"seed {seed} {label} trace width chain is broken")
        if committed:
            if (
                proposed_width is None
                or proposed_width >= before
                or after != proposed_width
            ):
                raise RuntimeError(f"seed {seed} {label} committed trace step is invalid")
            committed_count += 1
            commits_by_pair[step["sweep_pair"]] += 1
        elif after != before or (
            proposed_width is not None and proposed_width < before
        ):
            raise RuntimeError(f"seed {seed} {label} uncommitted trace step is invalid")
        last_width[step["start_name"]] = after
    if any(
        commits_by_pair[sweep_pair] == 0
        for sweep_pair in range(minimum_pair, maximum_pair)
    ):
        raise RuntimeError(f"seed {seed} {label} trace continued after convergence")
    return _TraceFacts(
        len(trace) * 101,
        committed_count,
        commits_by_pair,
        last_width,
    )


def _validate_checkpoint_diagnostics(
    checkpoint: object,
    *,
    seed: int,
    label: str,
    requested_pair: int,
    start_names: list[str] | tuple[str, ...],
    extension: bool,
) -> _CheckpointFacts:
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_FIELDS:
        raise RuntimeError(f"seed {seed} {label} checkpoint has wrong exact fields")
    if checkpoint["requested_sweep_pairs"] != requested_pair or type(
        checkpoint["requested_sweep_pairs"]
    ) is not int:
        raise RuntimeError(f"seed {seed} {label} checkpoint request changed")
    executed = _exact_int(
        checkpoint["executed_sweep_pairs"],
        seed=seed,
        label=f"{label} checkpoint executed pairs",
        minimum=1,
        maximum=requested_pair,
    )
    best_start = checkpoint["best_start_name"]
    if type(best_start) is not str or best_start not in start_names:
        raise RuntimeError(f"seed {seed} {label} checkpoint best start is invalid")
    schedule_evaluations = _exact_int(
        checkpoint["schedule_evaluations"],
        seed=seed,
        label=f"{label} checkpoint schedule evaluations",
        minimum=0,
    )
    committed_updates = _exact_int(
        checkpoint["committed_updates"],
        seed=seed,
        label=f"{label} checkpoint committed updates",
        minimum=0,
    )
    trace = checkpoint["trace"]
    if extension and executed <= 4:
        if trace != []:
            raise RuntimeError(f"seed {seed} {label} converged-parent trace must be empty")
        trace_facts = _TraceFacts(0, 0, {}, {})
    else:
        trace_facts = _validate_coordinate_trace(
            trace,
            seed=seed,
            label=label,
            start_names=start_names,
            minimum_pair=5 if extension else 1,
            maximum_pair=executed,
        )
    if (
        schedule_evaluations != trace_facts.schedule_evaluations
        or committed_updates != trace_facts.committed_updates
    ):
        raise RuntimeError(f"seed {seed} {label} checkpoint trace-derived counts differ")
    return _CheckpointFacts(checkpoint, trace_facts)


def _validate_initial_surfaces(
    records: pd.DataFrame,
    surfaces: dict[str, np.ndarray],
    diagnostics: dict[str, Any],
    seed: int,
) -> None:
    expected_keys: set[str] = set()
    for scenario_index, scenario in enumerate(_SCENARIOS):
        scenario_diagnostics = diagnostics.get(scenario)
        if not isinstance(scenario_diagnostics, dict):
            raise RuntimeError(f"seed {seed} diagnostics missing {scenario}")
        required_diagnostics = {
            "tuning_stream_id",
            "evaluation_stream_id",
            "start_order",
            "active_start_names",
            "extension_eligible",
            "pair4_state_sha256",
            "greedy_partial_indices",
            "search_status",
            "checkpoints",
        }
        if set(scenario_diagnostics) != required_diagnostics:
            raise RuntimeError(f"seed {seed} {scenario} diagnostics have wrong exact fields")
        if scenario_diagnostics["tuning_stream_id"] != _paper_seed(
            seed, 1_300_001 + scenario_index
        ) or scenario_diagnostics["evaluation_stream_id"] != _paper_seed(
            seed, 1_400_001 + scenario_index
        ):
            raise RuntimeError(f"seed {seed} {scenario} diagnostics stream IDs changed")
        if scenario_diagnostics["start_order"] != list(_START_NAMES):
            raise RuntimeError(f"seed {seed} {scenario} start order changed")
        active_names = scenario_diagnostics["active_start_names"]
        if (
            not isinstance(active_names, list)
            or active_names != [name for name in _START_NAMES if name in active_names]
            or len(set(active_names)) != len(active_names)
        ):
            raise RuntimeError(f"seed {seed} {scenario} active starts are not canonical")
        checkpoints = scenario_diagnostics["checkpoints"]
        has_pair4 = isinstance(checkpoints, dict) and "4" in checkpoints
        state_names = active_names if has_pair4 else []
        eligible = scenario_diagnostics["extension_eligible"]
        if type(eligible) is not bool or eligible != (
            has_pair4 and active_names == list(_START_NAMES)
        ):
            raise RuntimeError(f"seed {seed} {scenario} extension eligibility is invalid")
        hashes = scenario_diagnostics["pair4_state_sha256"]
        if not isinstance(hashes, list) or len(hashes) != len(state_names):
            raise RuntimeError(f"seed {seed} {scenario} pair4 state hash count changed")
        if any(type(value) is not str or _HEX.fullmatch(value) is None for value in hashes):
            raise RuntimeError(f"seed {seed} {scenario} pair4 state hash is malformed")
        search_status = scenario_diagnostics["search_status"]
        if type(search_status) is not str or search_status not in {
            "SELECTED",
            "NO_FEASIBLE_START",
            "WALL_TIME_CAP",
        }:
            raise RuntimeError(f"seed {seed} {scenario} diagnostic search status is invalid")
        greedy_partial = scenario_diagnostics["greedy_partial_indices"]
        if (
            not isinstance(greedy_partial, list)
            or len(greedy_partial) > 11
            or any(type(value) is not int or not 0 <= value <= 100 for value in greedy_partial)
        ):
            raise RuntimeError(f"seed {seed} {scenario} greedy partial diagnostics are invalid")
        if bool(_row(records, scenario, "greedy")["selection_available"]) and greedy_partial:
            raise RuntimeError(f"seed {seed} {scenario} selected greedy has partial diagnostics")

        base_keys = {
            f"{scenario}_profile",
            f"{scenario}_profiled_scale_grid",
            f"{scenario}_profiled_schedules",
            f"{scenario}_stage_grids",
            f"{scenario}_active_start_names",
            f"{scenario}_extension_eligible",
        }
        expected_keys.update(base_keys)
        profile = _require_array(
            surfaces,
            f"{scenario}_profile",
            (12,),
            seed=seed,
            dtype=np.float32,
        )
        scale_grid = _require_array(
            surfaces,
            f"{scenario}_profiled_scale_grid",
            (101,),
            seed=seed,
            dtype=np.float32,
        )
        profiled_schedules = _require_array(
            surfaces,
            f"{scenario}_profiled_schedules",
            (101, 12),
            seed=seed,
            dtype=np.float32,
        )
        stage_grids = _require_array(
            surfaces,
            f"{scenario}_stage_grids",
            (12, 101),
            seed=seed,
            dtype=np.float32,
        )
        if np.any(profile <= 0.0) or np.any(scale_grid <= 0.0) or np.any(stage_grids <= 0.0):
            raise RuntimeError(f"seed {seed} {scenario} grids/profile must be positive")
        if np.any(np.diff(scale_grid) < 0.0) or np.any(np.diff(stage_grids, axis=1) < 0.0):
            raise RuntimeError(f"seed {seed} {scenario} grids must be ordered")
        if not np.array_equal(profiled_schedules, scale_grid[:, None] * profile[None, :]):
            raise RuntimeError(f"seed {seed} {scenario} profiled schedules disagree")
        active_surface = _require_array(
            surfaces,
            f"{scenario}_active_start_names",
            (len(active_names),),
            seed=seed,
            dtype=np.int64,
        )
        if active_surface.dtype.kind not in "iu" or active_surface.tolist() != [
            _START_NAMES.index(name) for name in active_names
        ]:
            raise RuntimeError(f"seed {seed} {scenario} active-start surface disagrees")
        eligible_surface = _require_array(
            surfaces,
            f"{scenario}_extension_eligible",
            (),
            seed=seed,
            dtype=np.bool_,
        )
        if eligible_surface.dtype.kind != "b" or bool(eligible_surface) != eligible:
            raise RuntimeError(f"seed {seed} {scenario} eligibility surface disagrees")

        current_schedule = surfaces[f"{scenario}_current_profiled_schedule"]
        for method in _INITIAL_METHODS:
            expected_keys.update(
                f"{scenario}_{method}_{suffix}" for suffix in _SURFACE_FIELDS
            )
            method_row = _row(records, scenario, method)
            _surface_vectors_for_row(
                surfaces, method_row, scenario=scenario, method=method, seed=seed
            )
            if bool(method_row["selection_available"]):
                indices_length = 1 if method == "current_profiled" else 12
                indices = _json_vector(
                    method_row["selected_stage_grid_indices_json"],
                    seed=seed,
                    field="selected indices",
                    length=indices_length,
                ).astype(int)
                schedule = surfaces[f"{scenario}_{method}_schedule"]
                if method == "current_profiled":
                    expected_schedule = profiled_schedules[int(indices[0])]
                elif method == "greedy":
                    expected_schedule = stage_grids[np.arange(12), indices]
                else:
                    expected_schedule = _schedule_from_stage_indices(
                        indices,
                        stage_grids,
                        current_schedule,
                        seed=seed,
                        label=f"{scenario} {method}",
                    )
                if not np.array_equal(schedule, expected_schedule):
                    raise RuntimeError(f"seed {seed} {scenario} {method} index mapping disagrees")

        states_by_name: dict[str, SearchState] = {}
        for state_index, start_name in enumerate(state_names):
            prefix = f"{scenario}_pair4_{start_name}"
            state_keys = {f"{prefix}_{suffix}" for suffix in _STATE_SUFFIXES}
            expected_keys.update(state_keys)
            state = _state_from_surfaces(
                surfaces, scenario=scenario, start_name=start_name, seed=seed
            )
            states_by_name[start_name] = state
            for stage, index in enumerate(state.stage_grid_indices):
                expected_radius = (
                    current_schedule[stage] if index is None else stage_grids[stage, index]
                )
                if not np.array_equal(state.radii[stage].numpy(), expected_radius):
                    raise RuntimeError(f"seed {seed} {prefix} index/radius mapping disagrees")
            if _state_sha256(state) != hashes[state_index]:
                raise RuntimeError(f"seed {seed} {scenario} pair4 state hash mismatch")

        if not isinstance(checkpoints, dict):
            raise RuntimeError(f"seed {seed} {scenario} checkpoints are invalid")
        rows_by_pair = {
            2: _row(records, scenario, "joint_B"),
            4: _row(records, scenario, "joint_2B"),
        }
        expected_checkpoint_keys = {
            str(pair)
            for pair, method_row in rows_by_pair.items()
            if method_row["selection_status"] == "SELECTED"
        }
        if set(checkpoints) != expected_checkpoint_keys:
            raise RuntimeError(f"seed {seed} {scenario} checkpoint keys disagree with rows")
        if search_status == "SELECTED" and expected_checkpoint_keys != {"2", "4"}:
            raise RuntimeError(f"seed {seed} {scenario} selected diagnostics lack checkpoints")
        if search_status == "NO_FEASIBLE_START" and expected_checkpoint_keys:
            raise RuntimeError(f"seed {seed} {scenario} infeasible diagnostics have checkpoints")
        checkpoint_facts_by_pair: dict[int, _CheckpointFacts] = {}
        for pair, method_row in rows_by_pair.items():
            if str(pair) not in checkpoints:
                if method_row["selection_status"] != search_status:
                    raise RuntimeError(
                        f"seed {seed} {scenario} checkpoint status disagrees with rows"
                    )
                continue
            checkpoint_facts = _validate_checkpoint_diagnostics(
                checkpoints[str(pair)],
                seed=seed,
                label=f"{scenario} pair{pair}",
                requested_pair=pair,
                start_names=active_names,
                extension=False,
            )
            checkpoint_facts_by_pair[pair] = checkpoint_facts
            checkpoint = checkpoint_facts.payload
            executed = checkpoint["executed_sweep_pairs"]
            expected_convergence = (
                executed
                if checkpoint_facts.trace.commits_by_pair[executed] == 0
                else None
            )
            row_convergence = _nullable_csv_integer(
                method_row["converged_at_pair"],
                seed=seed,
                field=f"{scenario} {method_row['method_id']} converged_at_pair",
                minimum=1,
                maximum=pair,
            )
            if row_convergence != expected_convergence:
                raise RuntimeError(
                    f"seed {seed} {scenario} checkpoint convergence disagrees with row"
                )
            trace_best_start = min(
                active_names,
                key=checkpoint_facts.trace.final_width_by_start.__getitem__,
            )
            if checkpoint["best_start_name"] != trace_best_start:
                raise RuntimeError(
                    f"seed {seed} {scenario} checkpoint winner disagrees with trace"
                )
            trace_best_width = checkpoint_facts.trace.final_width_by_start[
                trace_best_start
            ]
            row_tuning_width = _finite_number(
                method_row["tuning_micro_width"],
                seed=seed,
                field=f"{scenario} {method_row['method_id']} tuning micro width",
                positive=True,
            )
            if not _matches_float32_reduction(
                row_tuning_width, trace_best_width, terms=12
            ):
                raise RuntimeError(
                    f"seed {seed} {scenario} checkpoint row tuning width disagrees with trace"
                )
            if method_row["chosen_initialization"] != checkpoint["best_start_name"]:
                raise RuntimeError(f"seed {seed} {scenario} checkpoint winner disagrees with row")
            if method_row["schedule_evaluations"] != checkpoint["schedule_evaluations"]:
                raise RuntimeError(f"seed {seed} checkpoint schedule count disagrees")
            if method_row["committed_updates"] != checkpoint["committed_updates"]:
                raise RuntimeError(f"seed {seed} checkpoint commit count disagrees")
            if pair == 4:
                for name, state in states_by_name.items():
                    state_width = float(state.normalized_width.mean().item())
                    trace_width = checkpoint_facts.trace.final_width_by_start[name]
                    if not _matches_float32_reduction(
                        state_width, trace_width, terms=12
                    ):
                        raise RuntimeError(
                            f"seed {seed} {scenario} persisted state width disagrees with trace"
                        )
                    if state.converged_at_pair != row_convergence:
                        raise RuntimeError(
                            f"seed {seed} {scenario} state convergence disagrees with row"
                        )
                best_state = states_by_name.get(checkpoint["best_start_name"])
                if best_state is None:
                    raise RuntimeError(f"seed {seed} pair4 checkpoint has no persisted state")
                persisted_best_start = min(
                    state_names,
                    key=lambda name: float(
                        states_by_name[name].normalized_width.mean().item()
                    ),
                )
                if checkpoint["best_start_name"] != persisted_best_start:
                    raise RuntimeError(
                        f"seed {seed} {scenario} checkpoint winner disagrees with states"
                    )
                vectors = _record_vectors(method_row, seed, "joint_2B", 12)
                persisted = {
                    "schedule": best_state.radii.numpy(),
                    "tuning_coverage": best_state.coverage.numpy(),
                    "tuning_stage_width": best_state.normalized_width.numpy(),
                }
                if any(
                    not np.array_equal(vectors[name], expected)
                    for name, expected in persisted.items()
                ):
                    raise RuntimeError(
                        f"seed {seed} {scenario} row disagrees with persisted pair4 state"
                    )
        if 4 in checkpoint_facts_by_pair:
            if 2 not in checkpoint_facts_by_pair:
                raise RuntimeError(
                    f"seed {seed} {scenario} pair4 checkpoint has no nested pair2"
                )
            pair2 = checkpoint_facts_by_pair[2].payload
            pair4 = checkpoint_facts_by_pair[4].payload
            trace2 = pair2["trace"]
            trace4 = pair4["trace"]
            if pair4["executed_sweep_pairs"] < pair2["executed_sweep_pairs"]:
                raise RuntimeError(
                    f"seed {seed} {scenario} nested checkpoint execution regressed"
                )
            if pair4["executed_sweep_pairs"] == pair2["executed_sweep_pairs"]:
                executed = pair4["executed_sweep_pairs"]
                if (
                    checkpoint_facts_by_pair[4].trace.commits_by_pair[executed]
                    != 0
                ):
                    raise RuntimeError(
                        f"seed {seed} {scenario} equal nested checkpoints must be converged"
                    )
                if (
                    trace4 != trace2
                    or pair4["schedule_evaluations"] != pair2["schedule_evaluations"]
                    or pair4["committed_updates"] != pair2["committed_updates"]
                ):
                    raise RuntimeError(
                        f"seed {seed} {scenario} equal nested checkpoints differ"
                    )
                pair2_vectors = _record_vectors(
                    rows_by_pair[2], seed, "joint_B", 12
                )
                pair4_vectors = _record_vectors(
                    rows_by_pair[4], seed, "joint_2B", 12
                )
                for name in ("schedule", "tuning_coverage", "tuning_stage_width"):
                    if not np.array_equal(pair2_vectors[name], pair4_vectors[name]):
                        raise RuntimeError(
                            f"seed {seed} {scenario} equal nested checkpoint rows differ"
                        )
            elif len(trace4) <= len(trace2) or trace4[: len(trace2)] != trace2:
                raise RuntimeError(
                    f"seed {seed} {scenario} pair2 trace is not a strict pair4 prefix"
                )
    if set(surfaces) != expected_keys:
        extra = sorted(set(surfaces) - expected_keys)
        missing = sorted(expected_keys - set(surfaces))
        raise RuntimeError(f"seed {seed} surfaces.npz exact keys differ; missing={missing}, extra={extra}")


_EXTENSION_DIAGNOSTIC_FIELDS = {
    "tuning_stream_id",
    "evaluation_stream_id",
    "search_status",
    "continuation_status",
    "fresh_evaluation_completed",
    "wall_time_phase",
    "checkpoint",
}
_EXTENSION_WALL_PHASES = {
    "parent_validation",
    "standard_cache",
    "standard_continuation",
    "tail_shift_cache",
    "tail_shift_continuation",
    "before_fresh",
    "standard_fresh",
    "tail_shift_fresh",
}


def _validate_extension_diagnostics(
    records: pd.DataFrame,
    diagnostics: dict[str, Any],
    seed: int,
    parent_facts: _ParentSeedFacts,
) -> None:
    phases: list[object] = []
    for scenario_index, scenario in enumerate(_SCENARIOS):
        row = _row(records, scenario, "joint_8SP")
        scenario_diagnostics = diagnostics.get(scenario)
        if not isinstance(scenario_diagnostics, dict) or set(
            scenario_diagnostics
        ) != _EXTENSION_DIAGNOSTIC_FIELDS:
            raise RuntimeError(
                f"seed {seed} {scenario} extension diagnostics have wrong exact fields"
            )
        if scenario_diagnostics["tuning_stream_id"] != _paper_seed(
            seed, 1_300_001 + scenario_index
        ) or scenario_diagnostics["evaluation_stream_id"] != _paper_seed(
            seed, 1_400_001 + scenario_index
        ):
            raise RuntimeError(f"seed {seed} extension diagnostic stream IDs changed")
        phase = scenario_diagnostics["wall_time_phase"]
        phases.append(phase)
        if phase is None:
            if (
                scenario_diagnostics["search_status"] != "SELECTED"
                or scenario_diagnostics["continuation_status"] != "SELECTED"
                or scenario_diagnostics["fresh_evaluation_completed"] is not True
                or row["selection_status"] != "SELECTED"
                or not bool(row["selection_available"])
            ):
                raise RuntimeError(
                    f"seed {seed} {scenario} extension selected status semantics differ"
                )
            checkpoint_facts = _validate_checkpoint_diagnostics(
                scenario_diagnostics["checkpoint"],
                seed=seed,
                label=f"{scenario} pair8",
                requested_pair=8,
                start_names=_START_NAMES,
                extension=True,
            )
            checkpoint = checkpoint_facts.payload
            executed = checkpoint["executed_sweep_pairs"]
            row_convergence = _nullable_csv_integer(
                row["converged_at_pair"],
                seed=seed,
                field=f"{scenario} joint_8SP converged_at_pair",
                minimum=1,
                maximum=8,
            )
            if checkpoint_facts.trace.final_width_by_start:
                expected_convergence = (
                    executed
                    if checkpoint_facts.trace.commits_by_pair[executed] == 0
                    else None
                )
                expected_best_start = min(
                    _START_NAMES,
                    key=checkpoint_facts.trace.final_width_by_start.__getitem__,
                )
                trace_best_width = checkpoint_facts.trace.final_width_by_start[
                    expected_best_start
                ]
                row_tuning_width = _finite_number(
                    row["tuning_micro_width"],
                    seed=seed,
                    field=f"{scenario} joint_8SP tuning micro width",
                    positive=True,
                )
                if not _matches_float32_reduction(
                    row_tuning_width, trace_best_width, terms=12
                ):
                    raise RuntimeError(
                        f"seed {seed} {scenario} extension row width disagrees with trace"
                    )
            else:
                expected_convergence = parent_facts.converged_at_pair[scenario]
                expected_best_start = parent_facts.best_start_name[scenario]
                if expected_convergence is None or executed != expected_convergence:
                    raise RuntimeError(
                        f"seed {seed} {scenario} empty extension trace disagrees with parent"
                    )
            if row_convergence != expected_convergence:
                raise RuntimeError(
                    f"seed {seed} {scenario} extension convergence disagrees"
                )
            if checkpoint["best_start_name"] != expected_best_start:
                raise RuntimeError(
                    f"seed {seed} {scenario} extension checkpoint winner disagrees with trace"
                )
            if checkpoint["best_start_name"] != row["chosen_initialization"]:
                raise RuntimeError(
                    f"seed {seed} {scenario} extension checkpoint winner disagrees"
                )
            if checkpoint["schedule_evaluations"] != row["schedule_evaluations"]:
                raise RuntimeError(f"seed {seed} extension checkpoint schedule count differs")
            if checkpoint["committed_updates"] != row["committed_updates"]:
                raise RuntimeError(f"seed {seed} extension checkpoint commit count differs")
            continue

        if type(phase) is not str or phase not in _EXTENSION_WALL_PHASES:
            raise RuntimeError(f"seed {seed} extension wall-time phase is invalid")
        if (
            scenario_diagnostics["search_status"] != "WALL_TIME_CAP"
            or row["selection_status"] != "WALL_TIME_CAP"
            or bool(row["selection_available"])
            or bool(row["tuning_joint_feasible"])
        ):
            raise RuntimeError(f"seed {seed} extension wall-cap status semantics differ")
        completed_before = {
            "parent_validation": 0,
            "standard_cache": 0,
            "tail_shift_cache": 1,
            "before_fresh": 2,
            "standard_fresh": 2,
            "tail_shift_fresh": 2,
        }.get(phase)
        continuation_index = {
            "standard_continuation": 0,
            "tail_shift_continuation": 1,
        }.get(phase)
        continuation = scenario_diagnostics["continuation_status"]
        if continuation_index is not None:
            if scenario_index < continuation_index:
                expected_continuations = {"SELECTED"}
            elif scenario_index == continuation_index:
                expected_continuations = {"SELECTED", "WALL_TIME_CAP"}
            else:
                expected_continuations = {None}
        else:
            assert completed_before is not None
            expected_continuations = (
                {"SELECTED"} if scenario_index < completed_before else {None}
            )
        if continuation not in expected_continuations:
            raise RuntimeError(f"seed {seed} extension continuation status differs")
        expected_fresh = scenario_index < {
            "standard_fresh": 1,
            "tail_shift_fresh": 2,
        }.get(phase, 0)
        if type(scenario_diagnostics["fresh_evaluation_completed"]) is not bool or (
            scenario_diagnostics["fresh_evaluation_completed"] != expected_fresh
        ):
            raise RuntimeError(f"seed {seed} extension fresh-evaluation status differs")
        checkpoint = scenario_diagnostics["checkpoint"]
        if continuation is None:
            if checkpoint is not None:
                raise RuntimeError(f"seed {seed} extension absent continuation has checkpoint")
        elif continuation == "SELECTED":
            _validate_checkpoint_diagnostics(
                checkpoint,
                seed=seed,
                label=f"{scenario} pair8",
                requested_pair=8,
                start_names=_START_NAMES,
                extension=True,
            )
        elif checkpoint is not None:
            _validate_checkpoint_diagnostics(
                checkpoint,
                seed=seed,
                label=f"{scenario} pair8",
                requested_pair=8,
                start_names=_START_NAMES,
                extension=True,
            )
    if len(set(phases)) != 1:
        raise RuntimeError(f"seed {seed} extension wall-time phase differs by scenario")


def _validate_extension_surfaces(
    records: pd.DataFrame,
    surfaces: dict[str, np.ndarray],
    diagnostics: dict[str, Any],
    seed: int,
    parent_facts: _ParentSeedFacts,
) -> None:
    _validate_extension_diagnostics(records, diagnostics, seed, parent_facts)
    expected_keys: set[str] = set()
    for scenario_index, scenario in enumerate(_SCENARIOS):
        row = _row(records, scenario, "joint_8SP")
        scenario_diagnostics = diagnostics.get(scenario)
        assert isinstance(scenario_diagnostics, dict)
        _surface_vectors_for_row(
            surfaces, row, scenario=scenario, method="joint_8SP", seed=seed
        )
        expected_keys.update(
            f"{scenario}_joint_8SP_{suffix}" for suffix in _SURFACE_FIELDS
        )
        grid_name = f"{scenario}_stage_grids"
        stage_grids = None
        if grid_name in surfaces:
            stage_grids = _require_array(
                surfaces, grid_name, (12, 101), seed=seed, dtype=np.float32
            )
            expected_keys.add(grid_name)
        else:
            exact_early_cap = (
                row["selection_status"] == "WALL_TIME_CAP"
                and not bool(row["selection_available"])
                and not bool(row["tuning_joint_feasible"])
                and row["failure_reason"] == "WALL_TIME_CAP"
                and pd.isna(row["chosen_initialization"])
                and scenario_diagnostics.get("search_status") == "WALL_TIME_CAP"
                and scenario_diagnostics.get("continuation_status") is None
                and scenario_diagnostics.get("fresh_evaluation_completed") is False
                and scenario_diagnostics.get("wall_time_phase") == "parent_validation"
                and scenario_diagnostics.get("checkpoint") is None
            )
            if not exact_early_cap:
                raise RuntimeError(
                    f"seed {seed} {scenario} missing stage grid is not an exact "
                    "parent-validation wall cap"
                )
        if bool(row["selection_available"]):
            if stage_grids is None:
                raise RuntimeError(
                    f"seed {seed} selected {scenario} extension has no stage grid"
                )
            indices = _json_vector(
                row["selected_stage_grid_indices_json"],
                seed=seed,
                field="joint_8SP selected indices",
                length=12,
            ).astype(int)
            schedule = surfaces[f"{scenario}_joint_8SP_schedule"]
            expected_schedule = _schedule_from_stage_indices(
                indices,
                stage_grids,
                parent_facts.current_schedules[scenario],
                seed=seed,
                label=f"{scenario} joint_8SP",
            )
            if not np.array_equal(schedule, expected_schedule):
                raise RuntimeError(f"seed {seed} {scenario} joint_8SP index mapping disagrees")
    if diagnostics["standard"]["wall_time_phase"] == "parent_validation":
        present_stage_grids = tuple(
            scenario
            for scenario in _SCENARIOS
            if f"{scenario}_stage_grids" in surfaces
        )
        if present_stage_grids not in ((), ("standard",), _SCENARIOS):
            raise RuntimeError(
                f"seed {seed} parent-validation stage grid presence is not a canonical prefix"
            )
    if set(surfaces) != expected_keys:
        raise RuntimeError(f"seed {seed} extension surfaces.npz exact keys differ")


def validate_seed_artifact(
    seed_dir: Path,
    seed: int,
    *,
    mode: str = "initial",
    expected_execution: dict[str, Any] | None = None,
    parent_seed_dir: Path | None = None,
) -> Path:
    """Deeply validate one atomic initial/smoke/extension seed artifact."""

    if mode not in {"smoke", "initial", "extension-8sp"}:
        raise ValueError("seed artifact mode is invalid")
    required_files = {"COMPLETE", "records.csv", "surfaces.npz", "metadata.json"}
    if not seed_dir.is_dir():
        raise RuntimeError(f"seed {seed} artifact directory is missing")
    actual_files = {path.name for path in seed_dir.iterdir() if path.is_file()}
    if actual_files != required_files or any(path.is_dir() for path in seed_dir.iterdir()):
        raise RuntimeError(f"seed {seed} artifact must contain exactly four files")
    complete = _load_json_object(seed_dir / "COMPLETE", label=f"seed {seed} COMPLETE")
    if complete != {"seed": seed, "status": "complete"}:
        raise RuntimeError(f"seed {seed} COMPLETE marker is invalid")
    metadata = _load_json_object(seed_dir / "metadata.json", label=f"seed {seed} metadata")
    if type(metadata.get("seed")) is not int or metadata["seed"] != seed:
        raise RuntimeError(f"seed {seed} metadata has the wrong seed ID")
    config = metadata.get("config")
    diagnostics = metadata.get("diagnostics")
    if not isinstance(config, dict) or not isinstance(diagnostics, dict):
        raise RuntimeError(f"seed {seed} metadata config/diagnostics must be objects")
    runner_provenance = diagnostics.get("runner_provenance")
    if not isinstance(runner_provenance, dict):
        raise RuntimeError(f"seed {seed} runner provenance is missing")
    _validate_execution_integer_contract(runner_provenance)
    _validated_runner_measurement(diagnostics.get("runner_measurement"), seed=seed)
    provenance_without_hash = dict(runner_provenance)
    stored_execution_hash = provenance_without_hash.pop("execution_sha256", None)
    if _require_sha256(stored_execution_hash, label=f"seed {seed} execution_sha256") != _canonical_sha256(
        provenance_without_hash
    ):
        raise RuntimeError(f"seed {seed} execution_sha256 does not authenticate provenance")
    if metadata.get("source_tree_sha256") != runner_provenance.get("source_tree_sha256"):
        raise RuntimeError(f"seed {seed} source hash differs from runner provenance")
    if canonical_config_sha256(config) != runner_provenance.get("config_sha256"):
        raise RuntimeError(f"seed {seed} scientific config hash differs")
    if runtime_config_sha256(config) != runner_provenance.get("runtime_config_sha256"):
        raise RuntimeError(f"seed {seed} runtime config hash differs")
    expected_mode = "extension-8sp" if mode == "extension-8sp" else mode
    if runner_provenance.get("study_kind") != expected_mode:
        raise RuntimeError(f"seed {seed} study kind differs")
    if runner_provenance.get("ordered_seeds") != list(config.get("seeds", [])):
        raise RuntimeError(f"seed {seed} ordered seed manifest differs")
    if expected_execution is not None and runner_provenance != expected_execution:
        for key, value in expected_execution.items():
            if runner_provenance.get(key) != value:
                raise RuntimeError(f"seed {seed} runner provenance {key} differs")
        raise RuntimeError(f"seed {seed} runner provenance has extra fields")

    records = _load_records(seed_dir, seed, mode=mode)
    _validate_record_rows(records, seed, mode=mode)
    surfaces = _load_npz(seed_dir, seed)
    scenario_diagnostics = {
        key: value
        for key, value in diagnostics.items()
        if key not in {"runner_provenance", "runner_measurement"}
    }
    if set(scenario_diagnostics) != set(_SCENARIOS):
        raise RuntimeError(f"seed {seed} diagnostics must contain exactly both scenarios")
    if mode in {"smoke", "initial"}:
        if parent_seed_dir is not None:
            raise ValueError("initial/smoke validation does not accept a parent seed")
        _validate_initial_surfaces(records, surfaces, scenario_diagnostics, seed)
    else:
        parent_facts = _validated_parent_current_schedules(
            parent_seed_dir,
            seed,
            extension_execution=runner_provenance,
        )
        _validate_extension_surfaces(
            records,
            surfaces,
            scenario_diagnostics,
            seed,
            parent_facts,
        )
    return seed_dir


class _ParentSeedFacts(NamedTuple):
    current_schedules: dict[str, np.ndarray]
    converged_at_pair: dict[str, int | None]
    best_start_name: dict[str, str]


def _validated_parent_current_schedules(
    parent_seed_dir: Path | None,
    seed: int,
    *,
    extension_execution: dict[str, Any],
) -> _ParentSeedFacts:
    if parent_seed_dir is None:
        raise RuntimeError("extension validation requires an authenticated parent seed")
    parent_root = parent_seed_dir.parent.resolve()
    frozen_parent = extension_execution.get("parent_output_dir")
    if type(frozen_parent) is not str or str(Path(frozen_parent).resolve()) != frozen_parent:
        raise RuntimeError("extension parent_output_dir must be a resolved path")
    if parent_root != Path(frozen_parent) or parent_seed_dir.resolve() != (
        parent_root / f"seed_{seed:05d}"
    ):
        raise RuntimeError("extension parent seed path differs from frozen provenance")
    parent_manifest = validate_study_manifest(parent_root, expected_kind="initial")
    manifest_sha = hashlib.sha256(
        (parent_root / "study_manifest.json").read_bytes()
    ).hexdigest()
    if manifest_sha != extension_execution.get("parent_study_manifest_sha256"):
        raise RuntimeError("extension parent study manifest hash differs")
    parent_metadata = _load_json_object(
        parent_root / "study_metadata.json", label="parent study metadata"
    )
    parent_execution = parent_metadata.get("execution")
    if not isinstance(parent_execution, dict) or parent_execution.get(
        "execution_sha256"
    ) != extension_execution.get("parent_execution_sha256"):
        raise RuntimeError("extension parent execution hash differs")
    if parent_manifest.get("execution_sha256") != parent_execution["execution_sha256"]:
        raise RuntimeError("extension parent manifest/execution cross-check failed")
    validate_seed_artifact(
        parent_seed_dir,
        seed,
        mode="initial",
        expected_execution=parent_execution,
    )
    parent_surfaces = _load_npz(parent_seed_dir, seed)
    current_schedules = {
        scenario: _require_array(
            parent_surfaces,
            f"{scenario}_current_profiled_schedule",
            (12,),
            seed=seed,
            dtype=np.float32,
        )
        for scenario in _SCENARIOS
    }
    parent_seed_metadata = _load_json_object(
        parent_seed_dir / "metadata.json", label=f"parent seed {seed} metadata"
    )
    parent_diagnostics = parent_seed_metadata["diagnostics"]
    converged: dict[str, int | None] = {}
    best_names: dict[str, str] = {}
    for scenario in _SCENARIOS:
        checkpoint = parent_diagnostics[scenario]["checkpoints"]["4"]
        best_name = checkpoint["best_start_name"]
        state = _state_from_surfaces(
            parent_surfaces,
            scenario=scenario,
            start_name=best_name,
            seed=seed,
        )
        converged[scenario] = state.converged_at_pair
        best_names[scenario] = best_name
    return _ParentSeedFacts(current_schedules, converged, best_names)


class ParentContinuation(NamedTuple):
    pair4_states: dict[str, tuple[SearchState, ...]]
    pair4_state_sha256: dict[str, tuple[str, ...]]
    extension_eligible: dict[str, bool]


def load_pair4_states(seed_dir: Path, seed: int) -> ParentContinuation:
    validate_seed_artifact(seed_dir, seed, mode="initial")
    metadata = _load_json_object(seed_dir / "metadata.json", label=f"seed {seed} metadata")
    diagnostics = metadata["diagnostics"]
    surfaces = _load_npz(seed_dir, seed)
    states: dict[str, tuple[SearchState, ...]] = {}
    hashes: dict[str, tuple[str, ...]] = {}
    eligible: dict[str, bool] = {}
    for scenario in _SCENARIOS:
        scenario_diagnostics = diagnostics[scenario]
        names = scenario_diagnostics["active_start_names"]
        state_names = names[: len(scenario_diagnostics["pair4_state_sha256"])]
        states[scenario] = tuple(
            _state_from_surfaces(
                surfaces, scenario=scenario, start_name=start_name, seed=seed
            )
            for start_name in state_names
        )
        hashes[scenario] = tuple(scenario_diagnostics["pair4_state_sha256"])
        eligible[scenario] = scenario_diagnostics["extension_eligible"]
    return ParentContinuation(states, hashes, eligible)


def _study_files(output_dir: Path, seeds: tuple[int, ...]) -> tuple[Path, ...]:
    files = [output_dir / "config.yaml", output_dir / "study_metadata.json"]
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed:05d}"
        files.extend(
            seed_dir / name
            for name in ("COMPLETE", "records.csv", "surfaces.npz", "metadata.json")
        )
    return tuple(files)


def _publish_study_manifest(
    output_dir: Path,
    execution: dict[str, Any],
) -> Path:
    seeds = tuple(execution["ordered_seeds"])
    files = _study_files(output_dir, seeds)
    missing = [str(path.relative_to(output_dir)) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"study manifest cannot publish missing files: {missing}")
    payload = {
        "protocol": "phase0c_study_manifest_v1",
        "status": "complete",
        "study_kind": execution["study_kind"],
        "ordered_seeds": list(seeds),
        "source_tree_sha256": execution["source_tree_sha256"],
        "experiment_tree_sha256": execution["experiment_tree_sha256"],
        "config_sha256": execution["config_sha256"],
        "runtime_config_sha256": execution["runtime_config_sha256"],
        "execution_sha256": execution["execution_sha256"],
        "files": {
            path.relative_to(output_dir).as_posix(): _file_fact(path) for path in files
        },
    }
    path = output_dir / "study_manifest.json"
    _atomic_write_text(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return path


def _validate_root_completion(
    output_dir: Path,
    ordered_seeds: list[int],
) -> None:
    marker = output_dir / "COMPLETE"
    try:
        marker_bytes = marker.read_bytes()
    except OSError as error:
        raise RuntimeError(f"study root COMPLETE is unreadable: {error}") from error
    if marker_bytes != b"complete\n":
        raise RuntimeError("study root COMPLETE must contain exactly complete\\n")

    status = _load_json_object(
        output_dir / "study_status.json", label="study status"
    )
    expected_fields = {
        "status",
        "expected_seeds",
        "completed_seeds",
        "missing_seeds",
        "updated_at_utc",
        "error",
    }
    exact_seed_list = (
        isinstance(ordered_seeds, list)
        and all(type(seed) is int for seed in ordered_seeds)
    )
    status_seed_lists_are_exact = all(
        isinstance(status.get(field), list)
        and all(type(seed) is int for seed in status[field])
        for field in ("expected_seeds", "completed_seeds")
    )
    if (
        set(status) != expected_fields
        or status.get("status") != "complete"
        or not exact_seed_list
        or not status_seed_lists_are_exact
        or status.get("expected_seeds") != ordered_seeds
        or status.get("completed_seeds") != ordered_seeds
        or status.get("missing_seeds") != []
        or type(status.get("updated_at_utc")) is not str
        or not status["updated_at_utc"]
        or status.get("error") is not None
    ):
        raise RuntimeError("study status completion contract is invalid")


def validate_study_manifest(
    output_dir: Path,
    *,
    expected_kind: str,
    require_root_complete: bool = True,
) -> dict[str, Any]:
    manifest = _load_json_object(
        output_dir / "study_manifest.json", label="study manifest"
    )
    expected_fields = {
        "protocol",
        "status",
        "study_kind",
        "ordered_seeds",
        "source_tree_sha256",
        "experiment_tree_sha256",
        "config_sha256",
        "runtime_config_sha256",
        "execution_sha256",
        "files",
    }
    if set(manifest) != expected_fields:
        raise RuntimeError("study manifest has wrong exact fields")
    if (
        manifest["protocol"] != "phase0c_study_manifest_v1"
        or manifest["status"] != "complete"
        or manifest["study_kind"] != expected_kind
    ):
        raise RuntimeError("study manifest protocol/status/study kind changed")
    metadata = _load_json_object(
        output_dir / "study_metadata.json", label="study metadata"
    )
    execution = metadata.get("execution")
    if not isinstance(execution, dict):
        raise RuntimeError("study metadata execution is missing")
    _validate_execution_integer_contract(execution)
    ordered_seeds = manifest["ordered_seeds"]
    metadata_seeds = metadata.get("seeds")
    if (
        not isinstance(ordered_seeds, list)
        or any(type(seed) is not int for seed in ordered_seeds)
        or len(set(ordered_seeds)) != len(ordered_seeds)
        or not isinstance(metadata_seeds, list)
        or any(type(seed) is not int for seed in metadata_seeds)
        or ordered_seeds != execution.get("ordered_seeds")
        or ordered_seeds != metadata_seeds
    ):
        raise RuntimeError("study manifest ordered seeds differ from metadata")
    if metadata.get("source_tree_sha256") != manifest["source_tree_sha256"]:
        raise RuntimeError("study manifest source_tree_sha256 differs")
    for key in (
        "experiment_tree_sha256",
        "config_sha256",
        "runtime_config_sha256",
        "execution_sha256",
    ):
        if execution.get(key) != manifest[key]:
            raise RuntimeError(f"study manifest {key} differs from execution")
    expected_paths = _study_files(output_dir, tuple(ordered_seeds))
    expected_names = {
        path.relative_to(output_dir).as_posix() for path in expected_paths
    }
    _validate_file_facts(
        output_dir, manifest["files"], expected_names, label="study manifest"
    )
    seed_paths = {
        path.name for path in output_dir.iterdir() if path.name.startswith("seed_")
    }
    if seed_paths != {f"seed_{seed:05d}" for seed in ordered_seeds}:
        raise RuntimeError("study root seed directories differ from ordered seeds")
    if any(path.name.startswith(".seed_") for path in output_dir.iterdir()):
        raise RuntimeError("partial atomic seed directory blocks study validation")
    if require_root_complete:
        _validate_root_completion(output_dir, ordered_seeds)
    return manifest


def _validate_global_streams(output_dir: Path, seeds: tuple[int, ...]) -> None:
    seen: set[int] = set()
    for seed in seeds:
        records = pd.read_csv(output_dir / f"seed_{seed:05d}" / "records.csv")
        streams = {
            int(value)
            for field in ("tuning_stream_id", "evaluation_stream_id")
            for value in records[field].unique()
        }
        if seen & streams:
            raise RuntimeError(f"seed {seed} creates a global stream collision")
        seen.update(streams)


def _validated_existing_seeds(
    output_dir: Path,
    execution: dict[str, Any],
) -> set[int]:
    requested = set(execution["ordered_seeds"])
    completed: set[int] = set()
    for path in output_dir.iterdir():
        if path.name.startswith(".seed_"):
            raise RuntimeError(f"partial atomic seed directory blocks resume: {path}")
        if not path.name.startswith("seed_"):
            continue
        match = _SEED_DIRECTORY.fullmatch(path.name)
        if match is None or not path.is_dir():
            raise RuntimeError(f"malformed seed path blocks resume: {path}")
        seed = int(match.group(1))
        if seed not in requested:
            raise RuntimeError(f"unexpected seed {seed} directory blocks resume")
        parent_seed_dir = None
        if execution["study_kind"] == "extension-8sp":
            frozen_parent = execution.get("parent_output_dir")
            if type(frozen_parent) is not str:
                raise RuntimeError("extension resume parent_output_dir is missing")
            parent_seed_dir = Path(frozen_parent) / f"seed_{seed:05d}"
        validate_seed_artifact(
            path,
            seed,
            mode=execution["study_kind"],
            expected_execution=execution,
            parent_seed_dir=parent_seed_dir,
        )
        completed.add(seed)
    return completed


def _validate_resume_provenance(
    output_dir: Path,
    config: ExperimentConfig,
    execution: dict[str, Any],
) -> None:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"resume output does not exist: {output_dir}")
    try:
        stored_config = yaml.safe_load((output_dir / "config.yaml").read_text())
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"resume stored config is unreadable: {error}") from error
    metadata = _load_json_object(
        output_dir / "study_metadata.json", label="resume study metadata"
    )
    if not isinstance(stored_config, dict):
        raise RuntimeError("resume stored config must be a mapping")
    if canonical_config_sha256(stored_config) != execution["config_sha256"]:
        raise RuntimeError("resume config_sha256 differs")
    if runtime_config_sha256(stored_config) != execution["runtime_config_sha256"]:
        raise RuntimeError("resume runtime_config_sha256 differs")
    if metadata.get("source_tree_sha256") != execution["source_tree_sha256"]:
        raise RuntimeError("resume source_tree_sha256 differs")
    if metadata.get("seeds") != list(config.seeds):
        raise RuntimeError("resume ordered seeds differ")
    if metadata.get("devices") != list(config.devices):
        raise RuntimeError("resume devices differ")
    stored_execution = metadata.get("execution")
    if not isinstance(stored_execution, dict):
        raise RuntimeError("resume execution metadata is missing")
    _validate_execution_integer_contract(stored_execution)
    for key, expected in execution.items():
        if stored_execution.get(key) != expected:
            raise RuntimeError(f"resume {key} differs from requested execution")
    if set(stored_execution) != set(execution):
        raise RuntimeError("resume execution metadata has extra fields")


def _execute_jobs(
    worker_devices: tuple[str, ...],
    jobs: tuple[tuple[int, tuple[Any, ...]], ...],
    *,
    worker_function: Callable[..., Any],
) -> tuple[Any, ...]:
    with ExitStack() as stack:
        executors = [
            stack.enter_context(
                ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
            )
            for _ in worker_devices
        ]
        futures = tuple(
            executors[worker_index].submit(worker_function, *arguments)
            for worker_index, arguments in jobs
        )
        return tuple(future.result() for future in futures)


def _build_seed_jobs(
    seeds: tuple[int, ...],
    devices: tuple[str, ...],
    workers_per_device: int,
) -> tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]]:
    if type(workers_per_device) is not int or workers_per_device < 1:
        raise ValueError("workers_per_device must be a positive non-bool integer")
    if not devices:
        raise ValueError("at least one device is required")
    worker_devices = tuple(device for device in devices for _ in range(workers_per_device))
    jobs = tuple(
        (index % len(worker_devices), seed, worker_devices[index % len(worker_devices)])
        for index, seed in enumerate(seeds)
    )
    return worker_devices, jobs


def _run_and_write(
    config: ExperimentConfig,
    seed: int,
    device: str,
    output_dir: Path,
    candidate_chunk_size: int,
    max_seed_wall_seconds: float,
    mode: str,
    execution: dict[str, Any],
    parent_seed_dir: Path | None,
) -> dict[str, Any]:
    def run_and_measure(cuda_device: torch.device | None) -> tuple[Any, dict[str, Any]]:
        if mode == "extension-8sp":
            if parent_seed_dir is None:
                raise RuntimeError("extension worker requires a parent seed directory")
            parent = load_pair4_states(parent_seed_dir, seed)
            if any(not parent.extension_eligible[scenario] for scenario in _SCENARIOS):
                raise RuntimeError(f"seed {seed} parent is not extension eligible")
            if any(
                tuple(state.start_name for state in parent.pair4_states[scenario])
                != _START_NAMES
                for scenario in _SCENARIOS
            ):
                raise RuntimeError(f"seed {seed} parent does not contain three canonical states")
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        started_at = time.monotonic()
        if mode == "extension-8sp":
            result = run_phase0c_extension_seed(
                config,
                seed=seed,
                device=device,
                pair4_states=parent.pair4_states,
                pair4_state_sha256=parent.pair4_state_sha256,
                extension_eligible=parent.extension_eligible,
                candidate_chunk_size=candidate_chunk_size,
                max_seed_wall_seconds=max_seed_wall_seconds,
            )
        else:
            result = run_phase0c_seed(
                config,
                seed=seed,
                device=device,
                candidate_chunk_size=candidate_chunk_size,
                sweep_pair_checkpoints=(2, 4),
                max_seed_wall_seconds=max_seed_wall_seconds,
            )
        elapsed = time.monotonic() - started_at
        measurement = {
            "protocol": "phase0c_runner_measurement_v1",
            "elapsed_seconds": elapsed,
            "max_memory_allocated_bytes": (
                0
                if cuda_device is None
                else int(torch.cuda.max_memory_allocated(cuda_device))
            ),
            "max_memory_reserved_bytes": (
                0
                if cuda_device is None
                else int(torch.cuda.max_memory_reserved(cuda_device))
            ),
        }
        return result, _validated_runner_measurement(measurement, seed=seed)

    def run_and_publish(
        cuda_device: torch.device | None,
    ) -> tuple[Path, dict[str, Any]]:
        result, measurement = run_and_measure(cuda_device)
        result = _runner_result(result, execution, measurement)
        seed_dir = write_seed_result(result, output_dir, config)
        validate_seed_artifact(
            seed_dir,
            seed,
            mode=mode,
            expected_execution=execution,
            parent_seed_dir=parent_seed_dir,
        )
        return seed_dir, measurement

    if device.startswith("cuda"):
        cuda_device = torch.device(device)
        torch.cuda.set_device(cuda_device)
        with torch.cuda.device(cuda_device):
            try:
                seed_dir, measurement = run_and_publish(cuda_device)
            finally:
                torch.cuda.empty_cache()
    else:
        seed_dir, measurement = run_and_publish(None)
    return {
        "seed_dir": str(seed_dir),
        "elapsed_seconds": measurement["elapsed_seconds"],
        "max_memory_allocated_bytes": measurement["max_memory_allocated_bytes"],
        "max_memory_reserved_bytes": measurement["max_memory_reserved_bytes"],
    }


def _run_pending_seeds(
    config: ExperimentConfig,
    output_dir: Path,
    seeds: tuple[int, ...],
    *,
    mode: str,
    workers_per_device: int,
    candidate_chunk_size: int,
    max_seed_wall_seconds: float,
    execution: dict[str, Any],
    parent_dir: Path | None,
) -> tuple[dict[str, Any], ...]:
    if not seeds:
        return ()
    worker_devices, assigned = _build_seed_jobs(
        seeds, config.devices, workers_per_device
    )
    jobs = tuple(
        (
            worker_index,
            (
                config,
                seed,
                device,
                output_dir,
                candidate_chunk_size,
                max_seed_wall_seconds,
                mode,
                execution,
                None if parent_dir is None else parent_dir / f"seed_{seed:05d}",
            ),
        )
        for worker_index, seed, device in assigned
    )
    return _execute_jobs(worker_devices, jobs, worker_function=_run_and_write)


def _persisted_measurement_result(seed_dir: Path, seed: int) -> dict[str, Any]:
    metadata = _load_json_object(
        seed_dir / "metadata.json", label=f"seed {seed} metadata"
    )
    diagnostics = metadata.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError(f"seed {seed} diagnostics are missing")
    measurement = _validated_runner_measurement(
        diagnostics.get("runner_measurement"), seed=seed
    )
    return {
        "seed_dir": str(seed_dir),
        "elapsed_seconds": measurement["elapsed_seconds"],
        "max_memory_allocated_bytes": measurement["max_memory_allocated_bytes"],
        "max_memory_reserved_bytes": measurement["max_memory_reserved_bytes"],
    }


def _write_smoke_result(
    output_dir: Path,
    execution: dict[str, Any],
    measurements: tuple[dict[str, Any], ...],
) -> Path:
    if len(measurements) != 1 or measurements[0].get("elapsed_seconds", 0.0) <= 0.0:
        raise RuntimeError("smoke must produce one positive timing measurement")
    measurement = measurements[0]
    elapsed = float(measurement["elapsed_seconds"])
    payload = {
        "protocol": "phase0c_smoke_v1",
        "seed": 9999,
        "max_sweep_pairs": 4,
        "elapsed_seconds": elapsed,
        "max_memory_allocated_bytes": int(measurement["max_memory_allocated_bytes"]),
        "max_memory_reserved_bytes": int(measurement["max_memory_reserved_bytes"]),
        "recommended_max_seed_wall_seconds": calibrate_wall_cap(elapsed),
        "source_tree_sha256": execution["source_tree_sha256"],
        "experiment_tree_sha256": execution["experiment_tree_sha256"],
        "config_sha256": execution["config_sha256"],
    }
    path = output_dir / "smoke_manifest.json"
    _atomic_write_text(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return path


def _parent_state_hash_manifest(
    parent_dir: Path,
    seeds: tuple[int, ...],
    parent_execution: dict[str, Any],
) -> tuple[int, int, str]:
    entries: list[dict[str, object]] = []
    eligible_count = 0
    for seed in seeds:
        seed_dir = parent_dir / f"seed_{seed:05d}"
        validate_seed_artifact(
            seed_dir,
            seed,
            mode="initial",
            expected_execution=parent_execution,
        )
        metadata = _load_json_object(
            seed_dir / "metadata.json", label=f"seed {seed} metadata"
        )
        for scenario in _SCENARIOS:
            diagnostics = metadata["diagnostics"][scenario]
            eligible_count += int(diagnostics["extension_eligible"])
            names = diagnostics["active_start_names"]
            hashes = diagnostics["pair4_state_sha256"]
            for start_name, state_hash in zip(names, hashes, strict=True):
                entries.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "start_name": start_name,
                        "sha256": state_hash,
                    }
                )
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return eligible_count, len(entries), digest


def authorize_extension(
    parent_dir: Path,
    decision_json: Path,
    *,
    config: ExperimentConfig,
    source_hash: str,
    experiment_hash: str,
) -> dict[str, Any]:
    """Authenticate the initial study and Task 5 decision without writing."""

    expected_decision_path = parent_dir / "checkpoint_analysis" / "phase0c_decision.json"
    if decision_json.resolve() != expected_decision_path.resolve():
        raise RuntimeError("decision path must be parent/checkpoint_analysis/phase0c_decision.json")
    parent_manifest = validate_study_manifest(parent_dir, expected_kind="initial")
    seeds = tuple(parent_manifest["ordered_seeds"])
    if seeds != tuple(range(10_000, 10_040)) or config.seeds != seeds:
        raise RuntimeError("extension ordered_seeds must be exactly 10000..10039")
    if parent_manifest["source_tree_sha256"] != source_hash:
        raise RuntimeError("extension parent source_tree_sha256 differs")
    if parent_manifest["experiment_tree_sha256"] != experiment_hash:
        raise RuntimeError("extension parent experiment_tree_sha256 differs")
    config_hash = canonical_config_sha256(config.to_dict())
    if parent_manifest["config_sha256"] != config_hash:
        raise RuntimeError("extension parent config_sha256 differs")
    parent_manifest_path = parent_dir / "study_manifest.json"
    parent_manifest_sha = hashlib.sha256(parent_manifest_path.read_bytes()).hexdigest()
    parent_metadata = _load_json_object(
        parent_dir / "study_metadata.json", label="parent study metadata"
    )
    parent_execution = parent_metadata.get("execution")
    if not isinstance(parent_execution, dict):
        raise RuntimeError("parent execution metadata is missing")

    analysis_dir = decision_json.parent
    summary_manifest_path = analysis_dir / "phase0c_summary_manifest.json"
    summary_manifest = _load_json_object(
        summary_manifest_path, label="checkpoint summary manifest"
    )
    expected_summary_fields = {
        "protocol",
        "status",
        "analysis_phase",
        "decision",
        "parent_study_manifest_sha256",
        "files",
    }
    if set(summary_manifest) != expected_summary_fields:
        raise RuntimeError("checkpoint summary manifest has wrong exact fields")
    if (
        summary_manifest["protocol"] != "phase0c_joint_search_summary_manifest_v1"
        or summary_manifest["status"] != "complete"
        or summary_manifest["analysis_phase"] != "initial"
    ):
        raise RuntimeError("checkpoint summary manifest protocol/status/phase changed")
    if summary_manifest["parent_study_manifest_sha256"] != parent_manifest_sha:
        raise RuntimeError("checkpoint summary parent_study_manifest_sha256 differs")
    payload_names = {
        "phase0c_decision.json",
        "phase0c_summary.csv",
        "phase0c_summary.md",
        "phase0c_joint_search.pdf",
        "phase0c_joint_search.svg",
        "phase0c_joint_search.png",
    }
    files = _validate_file_facts(
        analysis_dir,
        summary_manifest.get("files"),
        payload_names,
        label="checkpoint summary",
    )

    decision = _load_json_object(decision_json, label="checkpoint decision")
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
    if set(decision) != expected_decision_fields:
        raise RuntimeError("checkpoint decision has wrong exact fields")
    if decision["protocol"] != "phase0c_joint_search_summary_v1" or decision[
        "analysis_phase"
    ] != "initial":
        raise RuntimeError("checkpoint decision protocol/analysis phase changed")
    if decision["decision"] != "EXTENSION_8SP_REQUIRED":
        raise RuntimeError("checkpoint decision does not authorize extension")
    if summary_manifest["decision"] != decision["decision"]:
        raise RuntimeError("checkpoint summary/decision cross-check failed")
    decision_seeds = decision.get("ordered_seeds")
    if not isinstance(decision_seeds, list) or any(
        type(seed) is not int for seed in decision_seeds
    ):
        raise RuntimeError("checkpoint decision ordered_seeds must be exact integers")
    exact = {
        "parent_study_manifest_sha256": parent_manifest_sha,
        "ordered_seeds": list(seeds),
        "source_tree_sha256": source_hash,
        "experiment_tree_sha256": experiment_hash,
        "config_sha256": config_hash,
    }
    for key, expected in exact.items():
        if decision.get(key) != expected:
            raise RuntimeError(f"checkpoint decision {key} differs")

    eligible_count, state_hash_count, state_manifest_sha = _parent_state_hash_manifest(
        parent_dir, seeds, parent_execution
    )
    eligibility = decision.get("extension_eligibility")
    expected_eligibility_fields = {
        "all_eligible",
        "eligible_scenario_seed_count",
        "required_scenario_seed_count",
        "canonical_state_hash_count",
        "state_hash_manifest_sha256",
    }
    if not isinstance(eligibility, dict) or set(eligibility) != expected_eligibility_fields:
        raise RuntimeError("checkpoint extension eligibility has wrong fields")
    for field in (
        "eligible_scenario_seed_count",
        "required_scenario_seed_count",
        "canonical_state_hash_count",
    ):
        if type(eligibility[field]) is not int:
            label = (
                "canonical state hash count"
                if field == "canonical_state_hash_count"
                else f"extension eligibility {field}"
            )
            raise RuntimeError(
                f"checkpoint {label} must be an exact integer"
            )
    if (
        eligibility["all_eligible"] is not True
        or eligibility["eligible_scenario_seed_count"] != 80
        or eligibility["required_scenario_seed_count"] != 80
        or eligible_count != 80
    ):
        raise RuntimeError("checkpoint extension eligibility is not 80/80")
    if eligibility["canonical_state_hash_count"] != 240 or state_hash_count != 240:
        raise RuntimeError("checkpoint canonical state hash count is not 240")
    if eligibility["state_hash_manifest_sha256"] != state_manifest_sha:
        raise RuntimeError("checkpoint state hash manifest SHA256 differs")
    return {
        "parent_study_manifest_sha256": parent_manifest_sha,
        "checkpoint_decision_sha256": files["phase0c_decision.json"]["sha256"],
        "parent_execution_sha256": parent_execution["execution_sha256"],
        "max_seed_wall_seconds": float(parent_execution["max_seed_wall_seconds"]),
    }


def _paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def run_config(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    mode: str,
    workers_per_device: int,
    candidate_chunk_size: int,
    resume: bool,
    smoke_manifest: Path | None = None,
    parent_dir: Path | None = None,
    decision_json: Path | None = None,
) -> None:
    """Run or safely resume one exact Phase 0C study."""

    if mode not in {"smoke", "initial", "extension-8sp"}:
        raise ValueError("mode must be smoke, initial, or extension-8sp")
    if type(workers_per_device) is not int or workers_per_device < 1:
        raise ValueError("workers_per_device must be a positive non-bool integer")
    if type(candidate_chunk_size) is not int or candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be a positive non-bool integer")
    if config.output_dir != output_dir:
        raise ValueError("config output_dir must exactly match the requested output_dir")
    if mode == "smoke" and config.seeds != (9999,):
        raise ValueError("smoke seed bank must be exactly (9999,)")
    if (
        mode == "extension-8sp"
        and parent_dir is not None
        and _paths_overlap(output_dir, parent_dir)
    ):
        raise ValueError("extension output and parent paths must not overlap")

    current_source_hash = source_tree_sha256()
    current_experiment_hash = experiment_tree_sha256()
    scientific_hash = canonical_config_sha256(config.to_dict())
    parent_fields: dict[str, Any] | None = None
    if mode == "initial":
        if smoke_manifest is None:
            raise ValueError("initial mode requires a smoke manifest")
        max_seed_wall_seconds = float(
            validate_smoke_manifest(
                smoke_manifest,
                source_hash=current_source_hash,
                experiment_hash=current_experiment_hash,
                config_hash=scientific_hash,
            )
        )
    elif mode == "smoke":
        if candidate_chunk_size != 16:
            raise ValueError("smoke candidate_chunk_size must be exactly 16")
        max_seed_wall_seconds = _SMOKE_EXECUTION_CAP_SECONDS
    else:
        if parent_dir is None or decision_json is None:
            raise ValueError("extension mode requires parent_dir and decision_json")
        authorization = authorize_extension(
            parent_dir,
            decision_json,
            config=config,
            source_hash=current_source_hash,
            experiment_hash=current_experiment_hash,
        )
        max_seed_wall_seconds = float(authorization["max_seed_wall_seconds"])
        parent_fields = {
            key: authorization[key]
            for key in (
                "parent_study_manifest_sha256",
                "checkpoint_decision_sha256",
                "parent_execution_sha256",
            )
        }
        parent_fields["parent_output_dir"] = str(parent_dir.resolve())

    execution = _execution_metadata(
        config,
        mode=mode,
        workers_per_device=workers_per_device,
        candidate_chunk_size=candidate_chunk_size,
        max_seed_wall_seconds=max_seed_wall_seconds,
        source_hash=current_source_hash,
        experiment_hash=current_experiment_hash,
        parent_fields=parent_fields,
    )

    if resume:
        _validate_resume_provenance(output_dir, config, execution)
        completed = _validated_existing_seeds(output_dir, execution)
        missing = sorted(set(config.seeds) - completed)
        if (output_dir / "COMPLETE").is_file():
            if missing:
                raise RuntimeError(f"study COMPLETE exists but seeds are missing: {missing}")
            validate_study_manifest(output_dir, expected_kind=mode)
            _validate_global_streams(output_dir, config.seeds)
            return
        if (output_dir / "study_manifest.json").exists():
            raise RuntimeError("study manifest exists without root COMPLETE")
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh phase0c output already exists: {output_dir}")
        write_study_metadata(output_dir, config, execution=execution)
        completed = set()

    pending = tuple(seed for seed in config.seeds if seed not in completed)
    try:
        measurements = _run_pending_seeds(
            config,
            output_dir,
            pending,
            mode=mode,
            workers_per_device=workers_per_device,
            candidate_chunk_size=candidate_chunk_size,
            max_seed_wall_seconds=max_seed_wall_seconds,
            execution=execution,
            parent_dir=parent_dir,
        )
        for seed in config.seeds:
            validate_seed_artifact(
                output_dir / f"seed_{seed:05d}",
                seed,
                mode=mode,
                expected_execution=execution,
                parent_seed_dir=(
                    None
                    if parent_dir is None
                    else parent_dir / f"seed_{seed:05d}"
                ),
            )
        _validate_global_streams(output_dir, config.seeds)
        if mode == "smoke":
            persisted_measurements = tuple(
                _persisted_measurement_result(
                    output_dir / f"seed_{seed:05d}", seed
                )
                for seed in config.seeds
            )
            if measurements and measurements != persisted_measurements:
                raise RuntimeError(
                    "smoke worker measurement differs from its atomic seed metadata"
                )
            _write_smoke_result(output_dir, execution, persisted_measurements)
        _publish_study_manifest(output_dir, execution)
        validate_study_manifest(
            output_dir, expected_kind=mode, require_root_complete=False
        )
        mark_study_complete(output_dir, config.seeds)
    except BaseException as error:
        mark_study_failed(output_dir, config.seeds, error)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    base = ExperimentConfig.from_yaml(args.config)
    devices = resolve_devices(args.devices or base.devices)
    if args.workers_per_device != 1:
        parser.error("Phase 0C requires exactly one persistent worker per device")

    if args.mode == "smoke":
        if args.seed != 9999 or args.seeds is not None:
            parser.error("smoke uses only seed 9999")
        if devices != ("cuda:0",):
            parser.error("smoke requires exactly --devices cuda:0")
        if args.candidate_chunk_size != 16:
            parser.error("smoke requires --candidate-chunk-size 16")
        seeds = (9999,)
    else:
        try:
            seeds = parse_seeds(args.seeds, base.seeds)
        except ValueError as error:
            parser.error(str(error))
        if seeds != tuple(range(10_000, 10_040)):
            parser.error("formal Phase 0C seed bank must be exactly 10000:10040")
        if devices != ("cuda:0", "cuda:1"):
            parser.error("formal Phase 0C requires --devices cuda:0,cuda:1")
        if args.mode == "extension-8sp" and args.output_dir is None:
            parser.error("extension-8sp requires an explicit --output-dir")

    output_dir = args.output_dir or base.output_dir
    if args.mode == "extension-8sp" and output_dir.resolve() == args.parent_dir.resolve():
        parser.error("extension output must differ from the immutable parent root")
    config = base.with_overrides(
        devices=devices,
        seeds=seeds,
        output_dir=output_dir,
    )
    run_config(
        config,
        output_dir,
        mode=args.mode,
        workers_per_device=args.workers_per_device,
        candidate_chunk_size=args.candidate_chunk_size,
        resume=args.resume,
        smoke_manifest=getattr(args, "smoke_manifest", None),
        parent_dir=getattr(args, "parent_dir", None),
        decision_json=getattr(args, "decision_json", None),
    )
    print(output_dir)


if __name__ == "__main__":
    main()
