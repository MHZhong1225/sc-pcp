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


def _runner_result(result: Any, execution: dict[str, Any]) -> Any:
    diagnostics = dict(result.diagnostics)
    if "runner_provenance" in diagnostics:
        raise RuntimeError("seed diagnostics already contain runner_provenance")
    diagnostics["runner_provenance"] = execution
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
        if not (root / relative).is_file() or _file_fact(root / relative) != expected:
            raise RuntimeError(f"{label} {relative} bytes/hash mismatch")
    return files


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
        tuning_ids = set(int(value) for value in scenario_rows["tuning_stream_id"])
        evaluation_ids = set(int(value) for value in scenario_rows["evaluation_stream_id"])
        expected_tuning = _paper_seed(seed, 1_300_001 + scenario_index)
        expected_evaluation = _paper_seed(seed, 1_400_001 + scenario_index)
        if tuning_ids != {expected_tuning} or evaluation_ids != {expected_evaluation}:
            raise RuntimeError(f"seed {seed} {scenario} stream IDs changed")
        if tuning_ids & evaluation_ids or seen_streams & (tuning_ids | evaluation_ids):
            raise RuntimeError(f"seed {seed} contains a tuning/evaluation stream collision")
        seen_streams.update(tuning_ids | evaluation_ids)

    for row in records.to_dict(orient="records"):
        method = str(row["method_id"])
        analysis_role, budget_id, sweep_pairs, index_length = _METHOD_SPECS[method]
        if row["schema_version"] != "phase0c_seed_v1":
            raise RuntimeError(f"seed {seed} schema_version changed")
        if row["analysis_role"] != analysis_role or row["budget_id"] != budget_id:
            raise RuntimeError(f"seed {seed} {method} method metadata changed")
        if type(row["sweep_pairs"]) is bool or int(row["sweep_pairs"]) != sweep_pairs:
            raise RuntimeError(f"seed {seed} {method} sweep_pairs changed")
        status = row["selection_status"]
        if status not in {"SELECTED", "NO_FEASIBLE_START", "WALL_TIME_CAP"}:
            raise RuntimeError(f"seed {seed} {method} has an invalid selection status")
        available = bool(row["selection_available"])
        if available != (status == "SELECTED"):
            raise RuntimeError(f"seed {seed} {method} availability disagrees with status")
        vector_length = 12 if available else 0
        indices = _json_vector(
            row["selected_stage_grid_indices_json"],
            seed=seed,
            field=f"{method} selected indices",
            length=index_length if available else 0,
        )
        if available and (
            indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices > 100)
        ):
            raise RuntimeError(f"seed {seed} {method} selected indices are invalid")
        vectors = _record_vectors(row, seed, method, vector_length)
        q = vectors["schedule"]
        tuning_coverage = vectors["tuning_coverage"]
        tuning_width = vectors["tuning_stage_width"]
        final_coverage = vectors["final_coverage"]
        final_lcb = vectors["final_wilson_lcb"]
        final_width = vectors["final_stage_width"]
        if available:
            if np.any(q <= 0.0) or np.any(tuning_width <= 0.0) or np.any(final_width <= 0.0):
                raise RuntimeError(f"seed {seed} {method} widths and radii must be positive")
            if (
                np.any((tuning_coverage < 0.0) | (tuning_coverage > 1.0))
                or np.any((final_coverage < 0.0) | (final_coverage > 1.0))
                or np.any((final_lcb < 0.0) | (final_lcb > 1.0))
            ):
                raise RuntimeError(f"seed {seed} {method} coverage is outside [0,1]")
            if method.startswith("joint_") and np.any(tuning_coverage < 0.90):
                raise RuntimeError(f"seed {seed} {method} selected tuning coverage is below .90")
            tuning_micro = _finite_number(
                row["tuning_micro_width"], seed=seed, field=f"{method} tuning micro width", positive=True
            )
            micro = _finite_number(
                row["micro_normalized_width"], seed=seed, field=f"{method} micro width", positive=True
            )
            patient = _finite_number(
                row["patient_normalized_width"], seed=seed, field=f"{method} patient width", positive=True
            )
            if not math.isclose(tuning_micro, float(tuning_width.mean()), abs_tol=1e-7):
                raise RuntimeError(f"seed {seed} {method} tuning micro width disagrees")
            if not math.isclose(micro, float(final_width.mean()), abs_tol=1e-7):
                raise RuntimeError(f"seed {seed} {method} micro width disagrees")
            if not math.isclose(patient, micro, abs_tol=1e-7):
                raise RuntimeError(f"seed {seed} {method} patient width disagrees")
            if int(row["n_evaluation_rollouts"]) != 50_000:
                raise RuntimeError(f"seed {seed} {method} evaluation rollout count changed")
        else:
            for field in (
                "tuning_micro_width",
                "micro_normalized_width",
                "patient_normalized_width",
            ):
                if not pd.isna(row[field]):
                    raise RuntimeError(f"seed {seed} unavailable {method} {field} must be NaN")
            if int(row["n_evaluation_rollouts"]) != 0:
                raise RuntimeError(f"seed {seed} unavailable {method} has evaluation rollouts")
        if int(row["n_tuning_rollouts"]) != 5_000:
            raise RuntimeError(f"seed {seed} {method} tuning rollout count changed")
        for field in ("schedule_evaluations", "committed_updates"):
            value = row[field]
            if isinstance(value, bool) or not float(value).is_integer() or value < 0:
                raise RuntimeError(f"seed {seed} {method} {field} must be nonnegative integer")
        _finite_number(row["wall_time_seconds"], seed=seed, field=f"{method} wall time")


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
) -> np.ndarray:
    if name not in surfaces:
        raise RuntimeError(f"seed {seed} surfaces.npz is missing {name}")
    array = surfaces[name]
    if array.shape != shape:
        raise RuntimeError(f"seed {seed} {name} has shape {array.shape}, expected {shape}")
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
        actual = _require_array(surfaces, name, (length,), seed=seed)
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"seed {seed} records/NPZ disagrees for {name}")


def _state_from_surfaces(
    surfaces: dict[str, np.ndarray],
    *,
    scenario: str,
    start_name: str,
    seed: int,
) -> SearchState:
    prefix = f"{scenario}_pair4_{start_name}"
    radii = _require_array(surfaces, f"{prefix}_radii", (12,), seed=seed)
    indices = _require_array(
        surfaces, f"{prefix}_stage_grid_indices", (12,), seed=seed
    )
    if indices.dtype.kind not in "iu" or np.any(indices < -1) or np.any(indices > 100):
        raise RuntimeError(f"seed {seed} {prefix} indices are invalid")
    coverage = _require_array(surfaces, f"{prefix}_coverage", (12,), seed=seed)
    width = _require_array(
        surfaces, f"{prefix}_normalized_width", (12,), seed=seed
    )
    completed = _require_array(
        surfaces, f"{prefix}_completed_sweep_pairs", (), seed=seed
    )
    converged = _require_array(
        surfaces, f"{prefix}_converged_at_pair", (), seed=seed
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

        base_keys = {
            f"{scenario}_profile",
            f"{scenario}_profiled_scale_grid",
            f"{scenario}_profiled_schedules",
            f"{scenario}_stage_grids",
            f"{scenario}_active_start_names",
            f"{scenario}_extension_eligible",
        }
        expected_keys.update(base_keys)
        profile = _require_array(surfaces, f"{scenario}_profile", (12,), seed=seed)
        scale_grid = _require_array(
            surfaces, f"{scenario}_profiled_scale_grid", (101,), seed=seed
        )
        profiled_schedules = _require_array(
            surfaces, f"{scenario}_profiled_schedules", (101, 12), seed=seed
        )
        stage_grids = _require_array(
            surfaces, f"{scenario}_stage_grids", (12, 101), seed=seed
        )
        if np.any(profile <= 0.0) or np.any(scale_grid <= 0.0) or np.any(stage_grids <= 0.0):
            raise RuntimeError(f"seed {seed} {scenario} grids/profile must be positive")
        if np.any(np.diff(scale_grid) < 0.0) or np.any(np.diff(stage_grids, axis=1) < 0.0):
            raise RuntimeError(f"seed {seed} {scenario} grids must be ordered")
        if not np.array_equal(profiled_schedules, scale_grid[:, None] * profile[None, :]):
            raise RuntimeError(f"seed {seed} {scenario} profiled schedules disagree")
        active_surface = _require_array(
            surfaces, f"{scenario}_active_start_names", (len(active_names),), seed=seed
        )
        if active_surface.dtype.kind not in "iu" or active_surface.tolist() != [
            _START_NAMES.index(name) for name in active_names
        ]:
            raise RuntimeError(f"seed {seed} {scenario} active-start surface disagrees")
        eligible_surface = _require_array(
            surfaces, f"{scenario}_extension_eligible", (), seed=seed
        )
        if eligible_surface.dtype.kind != "b" or bool(eligible_surface) != eligible:
            raise RuntimeError(f"seed {seed} {scenario} eligibility surface disagrees")

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
                expected_schedule = (
                    profiled_schedules[int(indices[0])]
                    if method == "current_profiled"
                    else stage_grids[np.arange(12), indices]
                )
                if not np.array_equal(schedule, expected_schedule):
                    raise RuntimeError(f"seed {seed} {scenario} {method} index mapping disagrees")

        current_schedule = surfaces[f"{scenario}_current_profiled_schedule"]
        for state_index, start_name in enumerate(state_names):
            prefix = f"{scenario}_pair4_{start_name}"
            state_keys = {f"{prefix}_{suffix}" for suffix in _STATE_SUFFIXES}
            expected_keys.update(state_keys)
            state = _state_from_surfaces(
                surfaces, scenario=scenario, start_name=start_name, seed=seed
            )
            for stage, index in enumerate(state.stage_grid_indices):
                expected_radius = (
                    current_schedule[stage] if index is None else stage_grids[stage, index]
                )
                if not np.array_equal(state.radii[stage].numpy(), expected_radius):
                    raise RuntimeError(f"seed {seed} {prefix} index/radius mapping disagrees")
            if _state_sha256(state) != hashes[state_index]:
                raise RuntimeError(f"seed {seed} {scenario} pair4 state hash mismatch")

        if not isinstance(checkpoints, dict) or not set(checkpoints).issubset({"2", "4"}):
            raise RuntimeError(f"seed {seed} {scenario} checkpoints are invalid")
        for pair_text, checkpoint in checkpoints.items():
            if not isinstance(checkpoint, dict):
                raise RuntimeError(f"seed {seed} checkpoint diagnostics must be objects")
            required = {
                "requested_sweep_pairs",
                "executed_sweep_pairs",
                "best_start_name",
                "schedule_evaluations",
                "committed_updates",
                "trace",
            }
            if set(checkpoint) != required or not isinstance(checkpoint["trace"], list):
                raise RuntimeError(f"seed {seed} checkpoint diagnostics have wrong fields")
            pair = int(pair_text)
            if checkpoint["requested_sweep_pairs"] != pair:
                raise RuntimeError(f"seed {seed} checkpoint request changed")
            method_row = _row(records, scenario, "joint_B" if pair == 2 else "joint_2B")
            if bool(method_row["selection_available"]):
                if int(method_row["schedule_evaluations"]) != checkpoint["schedule_evaluations"]:
                    raise RuntimeError(f"seed {seed} checkpoint schedule count disagrees")
                if int(method_row["committed_updates"]) != checkpoint["committed_updates"]:
                    raise RuntimeError(f"seed {seed} checkpoint commit count disagrees")
    if set(surfaces) != expected_keys:
        extra = sorted(set(surfaces) - expected_keys)
        missing = sorted(expected_keys - set(surfaces))
        raise RuntimeError(f"seed {seed} surfaces.npz exact keys differ; missing={missing}, extra={extra}")


def _validate_extension_surfaces(
    records: pd.DataFrame,
    surfaces: dict[str, np.ndarray],
    diagnostics: dict[str, Any],
    seed: int,
) -> None:
    expected_keys: set[str] = set()
    for scenario_index, scenario in enumerate(_SCENARIOS):
        stage_grids = _require_array(
            surfaces, f"{scenario}_stage_grids", (12, 101), seed=seed
        )
        expected_keys.add(f"{scenario}_stage_grids")
        row = _row(records, scenario, "joint_8SP")
        _surface_vectors_for_row(
            surfaces, row, scenario=scenario, method="joint_8SP", seed=seed
        )
        expected_keys.update(
            f"{scenario}_joint_8SP_{suffix}" for suffix in _SURFACE_FIELDS
        )
        if bool(row["selection_available"]):
            indices = _json_vector(
                row["selected_stage_grid_indices_json"],
                seed=seed,
                field="joint_8SP selected indices",
                length=12,
            ).astype(int)
            schedule = surfaces[f"{scenario}_joint_8SP_schedule"]
            if not np.array_equal(schedule, stage_grids[np.arange(12), indices]):
                raise RuntimeError(f"seed {seed} {scenario} joint_8SP index mapping disagrees")
        scenario_diagnostics = diagnostics.get(scenario)
        if not isinstance(scenario_diagnostics, dict):
            raise RuntimeError(f"seed {seed} extension diagnostics missing {scenario}")
        if scenario_diagnostics.get("tuning_stream_id") != _paper_seed(
            seed, 1_300_001 + scenario_index
        ) or scenario_diagnostics.get("evaluation_stream_id") != _paper_seed(
            seed, 1_400_001 + scenario_index
        ):
            raise RuntimeError(f"seed {seed} extension diagnostic stream IDs changed")
    if set(surfaces) != expected_keys:
        raise RuntimeError(f"seed {seed} extension surfaces.npz exact keys differ")


def validate_seed_artifact(
    seed_dir: Path,
    seed: int,
    *,
    mode: str = "initial",
    expected_execution: dict[str, Any] | None = None,
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
        key: value for key, value in diagnostics.items() if key != "runner_provenance"
    }
    if set(scenario_diagnostics) != set(_SCENARIOS):
        raise RuntimeError(f"seed {seed} diagnostics must contain exactly both scenarios")
    if mode in {"smoke", "initial"}:
        _validate_initial_surfaces(records, surfaces, scenario_diagnostics, seed)
    else:
        _validate_extension_surfaces(records, surfaces, scenario_diagnostics, seed)
    return seed_dir


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
    ordered_seeds = manifest["ordered_seeds"]
    if (
        not isinstance(ordered_seeds, list)
        or any(type(seed) is not int for seed in ordered_seeds)
        or len(set(ordered_seeds)) != len(ordered_seeds)
        or ordered_seeds != execution.get("ordered_seeds")
        or ordered_seeds != metadata.get("seeds")
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
    if require_root_complete and not (output_dir / "COMPLETE").is_file():
        raise RuntimeError("study root COMPLETE is missing")
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
        validate_seed_artifact(
            path,
            seed,
            mode=execution["study_kind"],
            expected_execution=execution,
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
    started_at = time.monotonic()

    def run_and_publish() -> Path:
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
        result = _runner_result(result, execution)
        seed_dir = write_seed_result(result, output_dir, config)
        validate_seed_artifact(
            seed_dir,
            seed,
            mode=mode,
            expected_execution=execution,
        )
        return seed_dir

    allocated = 0
    reserved = 0
    if device.startswith("cuda"):
        cuda_device = torch.device(device)
        torch.cuda.set_device(cuda_device)
        with torch.cuda.device(cuda_device):
            torch.cuda.reset_peak_memory_stats(cuda_device)
            try:
                seed_dir = run_and_publish()
                allocated = int(torch.cuda.max_memory_allocated(cuda_device))
                reserved = int(torch.cuda.max_memory_reserved(cuda_device))
            finally:
                torch.cuda.empty_cache()
    else:
        seed_dir = run_and_publish()
    return {
        "seed_dir": str(seed_dir),
        "elapsed_seconds": time.monotonic() - started_at,
        "max_memory_allocated_bytes": allocated,
        "max_memory_reserved_bytes": reserved,
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
    if not (parent_dir / "COMPLETE").is_file():
        raise RuntimeError("extension parent root COMPLETE is missing")
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
            )
        _validate_global_streams(output_dir, config.seeds)
        if mode == "smoke":
            _write_smoke_result(output_dir, execution, measurements)
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
