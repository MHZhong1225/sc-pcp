"""Validate and summarize the preregistered Phase 0 oracle study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from scpcp.artifacts import (  # noqa: E402
    experiment_tree_sha256,
    git_revision,
    source_tree_sha256,
)
from scpcp.config import ExperimentConfig  # noqa: E402
from run_phase0_oracle import (  # noqa: E402
    canonical_config_sha256,
    validate_seed_artifact,
)
from run_phase0_search_sanity import (  # noqa: E402
    DEFAULT_CONFIG as SANITY_DEFAULT_CONFIG,
    canonical_config_sha256 as sanity_config_sha256,
    reduced_search_config,
)


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2_718_281
HORIZON = 12
GRID_SIZE = 101
EXPECTED_SEEDS = tuple(range(100))
SCENARIOS = ("standard", "tail_shift")
CURRENT = "Current Profiled Oracle"
GREEDY = "Greedy Sequential Oracle"
METHODS = (CURRENT, GREEDY)
COVERAGE_LEGEND_ORDER = (
    "Current mean",
    "Current simultaneous LCB",
    "Greedy mean",
    "Greedy simultaneous LCB",
    "0.90 target",
)
RECORD_COLUMNS = {
    "scenario",
    "method",
    "seed",
    "selection_status",
    "selection_available",
    "failure_stage",
    "selected_endpoint",
    "q_by_time",
    "tuning_coverage",
    "tuning_width",
    "final_coverage",
    "final_wilson_lcb",
    "final_stage_width",
    "micro_normalized_width",
    "patient_normalized_width",
    "tuning_seed",
    "evaluation_seed",
    "n_rollouts",
}
VECTOR_COLUMNS = (
    "q_by_time",
    "tuning_coverage",
    "tuning_width",
    "final_coverage",
    "final_wilson_lcb",
    "final_stage_width",
)
OUTPUT_NAMES = (
    "phase0_summary.csv",
    "phase0_decision.json",
    "phase0_summary.md",
    "phase0_radius_and_coverage.pdf",
    "phase0_radius_and_coverage.svg",
    "phase0_radius_and_coverage.png",
)
MANIFEST_NAME = "phase0_summary_manifest.json"


def compute_coverage_summary(
    coverage: np.ndarray,
    *,
    horizon: int,
) -> dict[str, Any]:
    """Summarize fresh per-stage coverage across successfully selected seeds."""

    values = np.asarray(coverage, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != horizon:
        raise ValueError(f"coverage must have shape [n_selected, {horizon}]")
    if values.shape[0] == 0:
        return {
            "interval_method": "seed_mean_bonferroni_t_lcb",
            "conditioning": "conditional_on_successful_selection",
            "n_selected": 0,
            "critical_value": None,
            "mean": None,
            "lower": None,
            "minimum_stage_seed_mean_coverage": None,
            "minimum_stage_seed_mean_simultaneous_lcb": None,
            "mean_seed_minimum_stage_coverage": None,
            "minimum_seed_stage_coverage": None,
        }
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("coverage must be finite and lie in [0, 1]")

    means = values.mean(axis=0)
    minimum_mean = float(means.min())
    mean_seed_minimum = float(values.min(axis=1).mean())
    minimum_seed_stage = float(values.min())
    n_selected = values.shape[0]
    if n_selected < 2:
        critical = None
        lower = None
        minimum_lower = None
    else:
        critical = float(stats.t.ppf(1.0 - 0.05 / horizon, df=n_selected - 1))
        standard_error = values.std(axis=0, ddof=1) / math.sqrt(n_selected)
        lower_values = means - critical * standard_error
        lower = lower_values.tolist()
        minimum_lower = float(lower_values.min())
    return {
        "interval_method": "seed_mean_bonferroni_t_lcb",
        "conditioning": "conditional_on_successful_selection",
        "n_selected": n_selected,
        "critical_value": critical,
        "mean": means.tolist(),
        "lower": lower,
        "minimum_stage_seed_mean_coverage": minimum_mean,
        "minimum_stage_seed_mean_simultaneous_lcb": minimum_lower,
        "mean_seed_minimum_stage_coverage": mean_seed_minimum,
        "minimum_seed_stage_coverage": minimum_seed_stage,
    }


def compute_paired_width_inference(
    seeds: np.ndarray,
    numerator_micro: np.ndarray,
    denominator_micro: np.ndarray,
    numerator_patient: np.ndarray,
    denominator_patient: np.ndarray,
    *,
    comparison_name: str,
) -> dict[str, Any]:
    """Compute paired geometric width ratios and a seed-cluster bootstrap."""

    seed_values = np.asarray(seeds)
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (
            numerator_micro,
            denominator_micro,
            numerator_patient,
            denominator_patient,
        )
    ]
    if seed_values.ndim != 1 or any(values.shape != seed_values.shape for values in arrays):
        raise ValueError("paired seeds and width arrays must be one-dimensional and aligned")
    if np.issubdtype(seed_values.dtype, np.bool_) or not np.issubdtype(
        seed_values.dtype, np.integer
    ):
        raise ValueError("paired seed identities must be integers")
    if len(np.unique(seed_values)) != len(seed_values):
        raise ValueError("paired seed identities must be unique")
    if any(not np.isfinite(values).all() or np.any(values <= 0.0) for values in arrays):
        raise ValueError("selected widths must be strictly positive and finite")

    order = np.argsort(seed_values, kind="stable")
    micro_log_ratio = np.log(arrays[0][order]) - np.log(arrays[1][order])
    patient_log_ratio = np.log(arrays[2][order]) - np.log(arrays[3][order])
    n_paired = len(order)

    def point(log_ratio: np.ndarray) -> float | None:
        return None if len(log_ratio) == 0 else float(np.exp(log_ratio.mean()))

    micro_point = point(micro_log_ratio)
    patient_point = point(patient_log_ratio)
    if n_paired < 2:
        indices_hash = None
        micro_interval = (None, None)
        patient_interval = (None, None)
    else:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        indices = rng.integers(
            0,
            n_paired,
            size=(BOOTSTRAP_RESAMPLES, n_paired),
        )
        indices_hash = hashlib.sha256(indices.tobytes()).hexdigest()

        def interval(log_ratio: np.ndarray) -> tuple[float, float]:
            draws = np.exp(log_ratio[indices].mean(axis=1))
            limits = np.quantile(draws, [0.025, 0.975], method="linear")
            return float(limits[0]), float(limits[1])

        micro_interval = interval(micro_log_ratio)
        patient_interval = interval(patient_log_ratio)
    return {
        "comparison_name": comparison_name,
        "n_paired": n_paired,
        "micro": {
            "geometric_mean_ratio": micro_point,
            "ci_lower": micro_interval[0],
            "ci_upper": micro_interval[1],
        },
        "patient": {
            "geometric_mean_ratio": patient_point,
            "ci_lower": patient_interval[0],
            "ci_upper": patient_interval[1],
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "rng_seed": BOOTSTRAP_SEED,
            "quantile_method": "linear_percentile_95",
            "paired_index_sha256": indices_hash,
        },
    }


def evaluate_go_no_go(
    *,
    tail_greedy_stage_lcb: np.ndarray | None,
    tail_micro_ratio: float | None,
    tail_micro_bootstrap_upper: float | None,
    tail_patient_ratio: float | None,
    tail_greedy_selection_count: int,
    tail_n_paired: int,
    standard_greedy_stage_lcb: np.ndarray | None,
    standard_micro_ratio: float | None,
    standard_n_paired: int,
) -> dict[str, Any]:
    """Apply the seven preregistered primitive gates without rounding."""

    def stage_gate(identifier: str, values: np.ndarray | None) -> dict[str, Any]:
        available = values is not None and np.asarray(values).shape == (12,)
        value = None if not available else float(np.asarray(values, dtype=float).min())
        passed = bool(available and np.isfinite(value) and value >= 0.90)
        return _gate(identifier, value, 0.90, ">=", available, passed)

    def scalar_gate(
        identifier: str,
        value: float | None,
        threshold: float,
        operator: str,
        paired_count: int,
    ) -> dict[str, Any]:
        available = paired_count >= 2 and value is not None and math.isfinite(value)
        passed = bool(
            available
            and ((value <= threshold) if operator == "<=" else (value < threshold))
        )
        return _gate(
            identifier,
            value if available else None,
            threshold,
            operator,
            available,
            passed,
        )

    gates = [
        stage_gate("tail_stage_lcb", tail_greedy_stage_lcb),
        scalar_gate("tail_micro_ratio", tail_micro_ratio, 0.90, "<=", tail_n_paired),
        scalar_gate(
            "tail_micro_bootstrap_upper",
            tail_micro_bootstrap_upper,
            1.00,
            "<",
            tail_n_paired,
        ),
        scalar_gate("tail_patient_ratio", tail_patient_ratio, 0.92, "<=", tail_n_paired),
        _gate(
            "tail_selection_count",
            tail_greedy_selection_count,
            95,
            ">=",
            type(tail_greedy_selection_count) is int,
            type(tail_greedy_selection_count) is int
            and tail_greedy_selection_count >= 95,
        ),
        stage_gate("standard_stage_lcb", standard_greedy_stage_lcb),
        scalar_gate(
            "standard_micro_ratio",
            standard_micro_ratio,
            1.02,
            "<=",
            standard_n_paired,
        ),
    ]
    return {
        "gates": gates,
        "decision": "GO" if all(gate["passed"] for gate in gates) else "NO_GO",
    }


def _gate(
    identifier: str,
    value: Any,
    threshold: float | int,
    operator: str,
    available: bool,
    passed: bool,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "value": value,
        "threshold": threshold,
        "operator": operator,
        "available": bool(available),
        "passed": bool(passed),
    }


def load_validate_and_analyze(input_dir: Path) -> dict[str, Any]:
    """Fail closed on the frozen artifacts, then compute all registered analyses."""

    root = Path(input_dir)
    config, provenance = _validate_study_root(root)
    config_hash = canonical_config_sha256(config)
    primary: list[dict[str, Any]] = []
    common: list[dict[str, Any]] = []
    tuning_streams: list[int] = []
    evaluation_streams: list[int] = []

    for seed in EXPECTED_SEEDS:
        seed_dir = root / f"seed_{seed:05d}"
        validate_seed_artifact(
            seed_dir,
            seed,
            expected_source_hash=provenance["source_tree_sha256"],
            expected_config_hash=config_hash,
        )
        records = pd.read_csv(seed_dir / "records.csv")
        if set(records.columns) != RECORD_COLUMNS:
            raise RuntimeError(f"seed {seed} records.csv has the wrong Phase 0 schema")
        for column in ("seed", "tuning_seed", "evaluation_seed", "n_rollouts"):
            if pd.api.types.is_bool_dtype(records[column].dtype) or not pd.api.types.is_integer_dtype(
                records[column].dtype
            ):
                raise RuntimeError(f"seed {seed} {column} must have integer dtype")
        metadata = _read_json(seed_dir / "metadata.json", f"seed {seed} metadata")
        if metadata.get("git_revision") != provenance["git_revision"]:
            raise RuntimeError(f"seed {seed} git revision differs from study provenance")
        diagnostics = metadata.get("diagnostics")
        if not isinstance(diagnostics, dict) or set(diagnostics) != set(SCENARIOS):
            raise RuntimeError(f"seed {seed} diagnostics have the wrong scenarios")
        with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as archive:
            surfaces = {name: np.asarray(archive[name]) for name in archive.files}
        _validate_surface_schema(surfaces, seed=seed)

        for scenario in SCENARIOS:
            scenario_rows = records.loc[records["scenario"] == scenario]
            scenario_diagnostics = diagnostics[scenario]
            if not isinstance(scenario_diagnostics, dict):
                raise RuntimeError(f"seed {seed} {scenario} diagnostics must be an object")
            tuning_seed = _strict_int(
                scenario_diagnostics.get("tuning_seed"),
                f"seed {seed} {scenario} tuning_seed",
            )
            evaluation_seed = _strict_int(
                scenario_diagnostics.get("evaluation_seed"),
                f"seed {seed} {scenario} evaluation_seed",
            )
            if tuning_seed == evaluation_seed:
                raise RuntimeError(f"seed {seed} {scenario} tuning/evaluation streams collide")
            tuning_streams.append(tuning_seed)
            evaluation_streams.append(evaluation_seed)
            for method in METHODS:
                row = scenario_rows.loc[scenario_rows["method"] == method].iloc[0]
                if int(row["tuning_seed"]) != tuning_seed or int(row["evaluation_seed"]) != evaluation_seed:
                    raise RuntimeError(f"seed {seed} {scenario} methods do not share stream IDs")
                parsed = _validate_primary_row(row, seed=seed, scenario=scenario, method=method)
                method_key = "profiled" if method == CURRENT else "greedy"
                selected_indices = _validate_selection_diagnostic(
                    scenario_diagnostics.get(method_key),
                    parsed,
                    label=f"seed {seed} {scenario} {method_key}",
                )
                surface_key = f"{scenario}_{method_key}_selected_schedule"
                if surface_key not in surfaces or not _array_equal(
                    surfaces[surface_key], parsed["q_by_time"]
                ):
                    raise RuntimeError(
                        f"seed {seed} {scenario} selected primary schedule disagrees with NPZ"
                    )
                if parsed["selection_available"]:
                    if method_key == "profiled":
                        candidates = surfaces[f"{scenario}_profiled_candidate_schedules"]
                        if not _array_equal(
                            candidates[selected_indices[0]], parsed["q_by_time"]
                        ):
                            raise RuntimeError(
                                f"seed {seed} {scenario} selected profiled candidate disagrees"
                            )
                        candidate_coverage = surfaces[
                            f"{scenario}_profiled_candidate_coverage"
                        ][selected_indices[0]]
                        candidate_width = surfaces[
                            f"{scenario}_profiled_candidate_normalized_width"
                        ][selected_indices[0]]
                        if not (
                            _array_equal(candidate_coverage, parsed["tuning_coverage"])
                            and _array_equal(candidate_width, parsed["tuning_width"])
                        ):
                            raise RuntimeError(
                                f"seed {seed} {scenario} tuning metrics disagree with profiled candidate"
                            )
                    else:
                        grids = surfaces[f"{scenario}_greedy_stage_grids"]
                        selected = grids[np.arange(HORIZON), selected_indices]
                        if not _array_equal(selected, parsed["q_by_time"]):
                            raise RuntimeError(
                                f"seed {seed} {scenario} selected greedy candidate disagrees"
                            )
                primary.append(parsed)
            common.append(
                _validate_common_grid(
                    surfaces,
                    scenario_diagnostics.get("profiled_common_grid"),
                    seed=seed,
                    scenario=scenario,
                    tuning_seed=tuning_seed,
                    evaluation_seed=evaluation_seed,
                )
            )

    if len(primary) != 400:
        raise RuntimeError(f"Phase 0 requires exactly 400 primary rows, found {len(primary)}")
    if len(set(tuning_streams)) != 200:
        raise RuntimeError("all 200 scenario tuning streams must be unique")
    if len(set(evaluation_streams)) != 200:
        raise RuntimeError("all 200 scenario evaluation streams must be unique")
    if set(tuning_streams) & set(evaluation_streams):
        raise RuntimeError("tuning and evaluation stream sets must be disjoint")
    sanity = _validate_finite_mdp_sanity(root, provenance)
    return _analyze(primary, common, provenance, sanity)


def _validate_study_root(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    required = ("COMPLETE", "study_status.json", "study_metadata.json", "config.yaml")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"study is incomplete; missing root files: {missing}")
    status = _read_json(root / "study_status.json", "study status")
    if not _is_exact_seed_list(status.get("expected_seeds")) or not _is_exact_seed_list(
        status.get("completed_seeds")
    ):
        raise RuntimeError("study status must contain strict integer seeds 0..99")
    if (
        status.get("status") != "complete"
        or status.get("missing_seeds") != []
        or status.get("error") is not None
    ):
        raise RuntimeError("study_status.json is not the exact complete 100-seed state")
    try:
        config = yaml.safe_load((root / "config.yaml").read_text())
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"config.yaml is unreadable: {error}") from error
    if not isinstance(config, dict):
        raise RuntimeError("config.yaml must contain an object")
    expected_config = ExperimentConfig.from_yaml(ROOT / "configs" / "phase0_oracle.yaml").to_dict()
    stored_output = Path(str(config.get("output_dir")))
    if not stored_output.is_absolute():
        stored_output = ROOT / stored_output
    if stored_output.resolve() != root.resolve():
        raise RuntimeError("frozen config output_dir differs from the summarized study")
    comparable_config = dict(config)
    comparable_config["output_dir"] = "<study-output-dir>"
    expected_config["output_dir"] = "<study-output-dir>"
    if canonical_config_sha256(comparable_config) != canonical_config_sha256(expected_config):
        raise RuntimeError(
            "frozen config must exactly match configs/phase0_oracle.yaml "
            "apart from the study output_dir"
        )
    devices = config["devices"]

    metadata = _read_json(root / "study_metadata.json", "study metadata")
    execution = metadata.get("execution")
    if not isinstance(execution, dict):
        raise RuntimeError("study execution provenance is missing")
    active_source = source_tree_sha256()
    active_experiment = experiment_tree_sha256()
    active_revision = git_revision()
    checks = {
        "devices": devices,
        "source_tree_sha256": active_source,
        "git_revision": active_revision,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"study {key} differs from frozen active provenance")
    if not _is_exact_seed_list(metadata.get("seeds")):
        raise RuntimeError("study metadata must contain strict integer seeds 0..99")
    execution_checks = {
        "config_sha256": canonical_config_sha256(config),
        "experiment_tree_sha256": active_experiment,
    }
    for key, expected in execution_checks.items():
        if execution.get(key) != expected:
            raise RuntimeError(f"study {key} differs from frozen active provenance")
    for key in ("workers_per_device", "candidate_chunk_size"):
        if type(execution.get(key)) is not int or execution[key] < 1:
            raise RuntimeError(f"study {key} must be a positive integer")

    expected_paths = {f"seed_{seed:05d}" for seed in EXPECTED_SEEDS}
    observed_paths: set[str] = set()
    for path in root.iterdir():
        if path.name.startswith(".seed_"):
            raise RuntimeError(f"partial or unexpected seed path: {path.name}")
        if path.name.startswith("seed_"):
            if not path.is_dir() or re.fullmatch(r"seed_\d{5}", path.name) is None:
                raise RuntimeError(f"partial or unexpected seed path: {path.name}")
            observed_paths.add(path.name)
    if observed_paths != expected_paths:
        raise RuntimeError("study must contain exactly seed_00000 through seed_00099")
    provenance = {
        "git_revision": active_revision,
        "source_tree_sha256": active_source,
        "experiment_tree_sha256": active_experiment,
        "config_sha256": canonical_config_sha256(config),
        "devices": devices,
        "workers_per_device": execution["workers_per_device"],
        "candidate_chunk_size": execution["candidate_chunk_size"],
    }
    return config, provenance


def _validate_primary_row(
    row: pd.Series,
    *,
    seed: int,
    scenario: str,
    method: str,
) -> dict[str, Any]:
    label = f"seed {seed} {scenario} {method}"
    available = _strict_bool(row["selection_available"], f"{label} selection_available")
    if row["selection_status"] != ("SELECTED" if available else "UNAVAILABLE"):
        raise RuntimeError(f"{label} selection status disagrees with availability")
    vectors = {
        name: _parse_vector(row[name], name=f"{label} {name}") for name in VECTOR_COLUMNS
    }
    micro = float(row["micro_normalized_width"])
    patient = float(row["patient_normalized_width"])
    n_rollouts = int(row["n_rollouts"])
    failure_stage = _failure_stage(row["failure_stage"], label)
    endpoint = _strict_bool(row["selected_endpoint"], f"{label} selected_endpoint")
    if available:
        if failure_stage is not None or any(len(values) != HORIZON for values in vectors.values()):
            raise RuntimeError(f"{label} selected vectors must all have length 12")
        if not all(np.isfinite(values).all() for values in vectors.values()):
            raise RuntimeError(f"{label} selected vectors must be finite")
        for name in ("tuning_coverage", "final_coverage", "final_wilson_lcb"):
            if np.any((vectors[name] < 0.0) | (vectors[name] > 1.0)):
                raise RuntimeError(f"{label} {name} must lie in [0, 1]")
        if np.any(vectors["tuning_coverage"] < 0.90 - 1e-7):
            raise RuntimeError(f"{label} selected tuning coverage must be at least 0.90")
        for name in ("q_by_time", "tuning_width", "final_stage_width"):
            if np.any(vectors[name] <= 0.0):
                raise RuntimeError(f"{label} selected widths/radii must be strictly positive")
        if n_rollouts != 50_000 or not _positive_finite(micro, patient):
            raise RuntimeError(f"{label} selected evaluation has invalid rollout/width fields")
        if not (
            math.isclose(micro, float(vectors["final_stage_width"].mean()), rel_tol=1e-5, abs_tol=1e-7)
            and math.isclose(patient, micro, rel_tol=1e-5, abs_tol=1e-7)
        ):
            raise RuntimeError(f"{label} fixed-T micro, patient, and stage widths disagree")
    else:
        if endpoint or any(len(values) for values in vectors.values()):
            raise RuntimeError(f"{label} unavailable vectors must be empty")
        if n_rollouts != 0 or not math.isnan(micro) or not math.isnan(patient):
            raise RuntimeError(f"{label} unavailable widths must be NaN with zero rollouts")
    return {
        "seed": seed,
        "scenario": scenario,
        "method": method,
        "selection_available": available,
        "failure_stage": failure_stage,
        "selected_endpoint": endpoint,
        "micro_normalized_width": micro,
        "patient_normalized_width": patient,
        **vectors,
    }


def _validate_surface_schema(surfaces: dict[str, np.ndarray], *, seed: int) -> None:
    """Lock every Phase 0 candidate and evaluation surface to K=101, T=12."""

    primary_suffixes = {
        "profiled_scale_grid",
        "profile",
        "profiled_candidate_schedules",
        "profiled_candidate_coverage",
        "profiled_candidate_normalized_width",
        "profiled_selected_schedule",
        "greedy_stage_grids",
        "greedy_selected_schedule",
    }
    common_suffixes = {
        "profiled_common_grid_scale_grid",
        "profiled_common_grid_candidate_schedules",
        "profiled_common_grid_candidate_coverage",
        "profiled_common_grid_candidate_normalized_width",
        "profiled_common_grid_selected_schedule",
        "profiled_common_grid_final_coverage",
        "profiled_common_grid_final_wilson_lcb",
        "profiled_common_grid_final_stage_width",
        "profiled_common_grid_micro_normalized_width",
        "profiled_common_grid_patient_normalized_width",
        "profiled_common_grid_n_rollouts",
    }
    expected = {
        f"{scenario}_{suffix}"
        for scenario in SCENARIOS
        for suffix in primary_suffixes | common_suffixes
    }
    missing_primary = sorted(
        f"{scenario}_{suffix}"
        for scenario in SCENARIOS
        for suffix in primary_suffixes
        if f"{scenario}_{suffix}" not in surfaces
    )
    if missing_primary:
        raise RuntimeError(f"seed {seed} primary surface schema is missing: {missing_primary}")
    missing_common = sorted(
        f"{scenario}_{suffix}"
        for scenario in SCENARIOS
        for suffix in common_suffixes
        if f"{scenario}_{suffix}" not in surfaces
    )
    if missing_common:
        raise RuntimeError(f"seed {seed} common-grid surface is missing: {missing_common}")
    if set(surfaces) != expected:
        raise RuntimeError(f"seed {seed} Phase 0 surface schema has unexpected members")

    def checked(
        name: str,
        shape: tuple[int, ...] | tuple[tuple[int, ...], ...],
        *,
        positive: bool = False,
        probability: bool = False,
    ) -> np.ndarray:
        array = surfaces[name]
        shapes = shape if shape and isinstance(shape[0], tuple) else (shape,)
        if array.shape not in shapes or not np.isfinite(array).all():
            raise RuntimeError(f"seed {seed} primary surface schema is invalid: {name}")
        if positive and array.size and np.any(array <= 0.0):
            raise RuntimeError(f"seed {seed} primary surface schema is invalid: {name}")
        if probability and np.any((array < 0.0) | (array > 1.0)):
            raise RuntimeError(f"seed {seed} primary surface schema is invalid: {name}")
        return array

    for scenario in SCENARIOS:
        prefix = f"{scenario}_"
        scale_grid = checked(prefix + "profiled_scale_grid", (GRID_SIZE,), positive=True)
        profile = checked(prefix + "profile", (HORIZON,), positive=True)
        profiled_candidates = checked(
            prefix + "profiled_candidate_schedules",
            (GRID_SIZE, HORIZON),
            positive=True,
        )
        checked(
            prefix + "profiled_candidate_coverage",
            (GRID_SIZE, HORIZON),
            probability=True,
        )
        checked(
            prefix + "profiled_candidate_normalized_width",
            (GRID_SIZE, HORIZON),
            positive=True,
        )
        checked(
            prefix + "profiled_selected_schedule",
            ((0,), (HORIZON,)),
            positive=True,
        )
        checked(prefix + "greedy_stage_grids", (HORIZON, GRID_SIZE), positive=True)
        checked(
            prefix + "greedy_selected_schedule",
            ((0,), (HORIZON,)),
            positive=True,
        )

        common = prefix + "profiled_common_grid_"
        common_scale_grid = checked(common + "scale_grid", (GRID_SIZE,), positive=True)
        common_candidates = checked(
            common + "candidate_schedules", (GRID_SIZE, HORIZON), positive=True
        )
        checked(
            common + "candidate_coverage",
            (GRID_SIZE, HORIZON),
            probability=True,
        )
        checked(
            common + "candidate_normalized_width",
            (GRID_SIZE, HORIZON),
            positive=True,
        )
        checked(common + "selected_schedule", ((0,), (HORIZON,)), positive=True)
        checked(common + "final_coverage", ((0,), (HORIZON,)), probability=True)
        checked(common + "final_wilson_lcb", ((0,), (HORIZON,)), probability=True)
        checked(common + "final_stage_width", ((0,), (HORIZON,)), positive=True)
        for suffix in ("micro_normalized_width", "patient_normalized_width"):
            scalar = surfaces[common + suffix]
            if scalar.shape != () or not (
                np.isnan(scalar).item() or (np.isfinite(scalar).item() and scalar.item() > 0.0)
            ):
                raise RuntimeError(
                    f"seed {seed} primary surface schema is invalid: {common + suffix}"
                )
        n_rollouts = surfaces[common + "n_rollouts"]
        if n_rollouts.shape != () or float(n_rollouts) not in (0.0, 50_000.0):
            raise RuntimeError(
                f"seed {seed} primary surface schema is invalid: {common + 'n_rollouts'}"
            )
        if not _array_equal(profiled_candidates, scale_grid[:, None] * profile[None, :]):
            raise RuntimeError(
                f"seed {seed} {scenario} profiled candidate schedules disagree with scale/profile"
            )
        if not _array_equal(common_candidates, common_scale_grid[:, None] * profile[None, :]):
            raise RuntimeError(
                f"seed {seed} {scenario} common candidate schedules disagree with scale/profile"
            )


def _validate_selection_diagnostic(
    diagnostic: Any,
    row: dict[str, Any],
    *,
    label: str,
) -> list[int]:
    if not isinstance(diagnostic, dict):
        raise RuntimeError(f"{label} selection diagnostic is missing")
    if (
        diagnostic.get("selection_available") is not row["selection_available"]
        or _failure_stage(diagnostic.get("failure_stage"), label) != row["failure_stage"]
        or diagnostic.get("selected_endpoint") is not row["selected_endpoint"]
    ):
        raise RuntimeError(f"{label} record/metadata selection disagreement")
    indices = diagnostic.get("selected_indices")
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        raise RuntimeError(f"{label} selected_indices must be an integer list")
    if any(not 0 <= index < GRID_SIZE for index in indices):
        raise RuntimeError(f"{label} selected_indices must lie in [0, 100]")
    expected_length = 1 if row["method"] == CURRENT else HORIZON
    if row["selection_available"] and len(indices) != expected_length:
        raise RuntimeError(f"{label} selected_indices length must be {expected_length}")
    if not row["selection_available"] and (
        (row["method"] == CURRENT and indices) or len(indices) > HORIZON
    ):
        raise RuntimeError(f"{label} unavailable selected_indices are invalid")
    if row["selection_available"]:
        expected_endpoint = any(index in (0, GRID_SIZE - 1) for index in indices)
        if row["selected_endpoint"] is not expected_endpoint:
            raise RuntimeError(f"{label} selected_endpoint disagrees with selected index")
    elif row["method"] == GREEDY and (
        row["failure_stage"] is None or len(indices) != row["failure_stage"]
    ):
        raise RuntimeError(f"{label} unavailable greedy prefix disagrees with failure_stage")
    return indices


def _validate_common_grid(
    surfaces: dict[str, np.ndarray],
    diagnostic: Any,
    *,
    seed: int,
    scenario: str,
    tuning_seed: int,
    evaluation_seed: int,
) -> dict[str, Any]:
    label = f"seed {seed} {scenario} common-grid"
    if not isinstance(diagnostic, dict):
        raise RuntimeError(f"{label} diagnostic is missing")
    available = _strict_bool(diagnostic.get("selection_available"), f"{label} availability")
    endpoint = _strict_bool(diagnostic.get("selected_endpoint"), f"{label} endpoint")
    failure_stage = _failure_stage(diagnostic.get("failure_stage"), label)
    indices = diagnostic.get("selected_indices")
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        raise RuntimeError(f"{label} selected_indices must be integers")
    if any(not 0 <= index < GRID_SIZE for index in indices):
        raise RuntimeError(f"{label} selected_indices must lie in [0, 100]")
    fields = {
        "q_by_time": "selected_schedule",
        "final_coverage": "final_coverage",
        "final_wilson_lcb": "final_wilson_lcb",
        "final_stage_width": "final_stage_width",
    }
    vectors = {
        output: _numeric_list(diagnostic.get(source), f"{label} {source}")
        for output, source in fields.items()
    }
    micro = _float_value(diagnostic.get("micro_normalized_width"), f"{label} micro width")
    patient = _float_value(diagnostic.get("patient_normalized_width"), f"{label} patient width")
    n_rollouts = _strict_int(diagnostic.get("n_rollouts"), f"{label} n_rollouts")
    prefix = f"{scenario}_profiled_common_grid_"
    required_suffixes = {
        "scale_grid",
        "candidate_schedules",
        "candidate_coverage",
        "candidate_normalized_width",
        "selected_schedule",
        "final_coverage",
        "final_wilson_lcb",
        "final_stage_width",
        "micro_normalized_width",
        "patient_normalized_width",
        "n_rollouts",
    }
    missing_surfaces = sorted(
        suffix for suffix in required_suffixes if prefix + suffix not in surfaces
    )
    if missing_surfaces:
        raise RuntimeError(f"{label} common-grid surface is missing: {missing_surfaces}")
    surface_fields = {
        "q_by_time": "selected_schedule",
        "final_coverage": "final_coverage",
        "final_wilson_lcb": "final_wilson_lcb",
        "final_stage_width": "final_stage_width",
    }
    for output, suffix in surface_fields.items():
        key = prefix + suffix
        if key not in surfaces or not _array_equal(surfaces[key], vectors[output]):
            raise RuntimeError(f"{label} common-grid metadata/surface disagreement")
    for suffix, expected in (
        ("micro_normalized_width", micro),
        ("patient_normalized_width", patient),
        ("n_rollouts", n_rollouts),
    ):
        key = prefix + suffix
        if key not in surfaces or surfaces[key].size != 1 or not math.isclose(
            float(surfaces[key].reshape(-1)[0]), expected, rel_tol=1e-5, abs_tol=1e-7
        ):
            if not (math.isnan(expected) and key in surfaces and np.isnan(surfaces[key]).all()):
                raise RuntimeError(f"{label} common-grid metadata/surface disagreement")
    if available:
        if failure_stage is not None or endpoint not in (True, False):
            raise RuntimeError(f"{label} selected diagnostic is inconsistent")
        if len(indices) != 1 or endpoint is not (indices[0] in (0, GRID_SIZE - 1)):
            raise RuntimeError(f"{label} selected_endpoint disagrees with selected index")
        if any(len(values) != HORIZON for values in vectors.values()):
            raise RuntimeError(f"{label} selected vectors must have length 12")
        if not all(np.isfinite(values).all() for values in vectors.values()):
            raise RuntimeError(f"{label} selected vectors must be finite")
        if np.any((vectors["final_coverage"] < 0) | (vectors["final_coverage"] > 1)) or np.any(
            (vectors["final_wilson_lcb"] < 0) | (vectors["final_wilson_lcb"] > 1)
        ):
            raise RuntimeError(f"{label} coverage must lie in [0, 1]")
        if np.any(vectors["q_by_time"] <= 0) or np.any(vectors["final_stage_width"] <= 0):
            raise RuntimeError(f"{label} radii and widths must be positive")
        if n_rollouts != 50_000 or not _positive_finite(micro, patient):
            raise RuntimeError(f"{label} rollout/width fields are invalid")
        if not (
            math.isclose(micro, float(vectors["final_stage_width"].mean()), rel_tol=1e-5, abs_tol=1e-7)
            and math.isclose(patient, micro, rel_tol=1e-5, abs_tol=1e-7)
        ):
            raise RuntimeError(f"{label} fixed-T widths disagree")
        candidate_key = prefix + "candidate_schedules"
        if candidate_key not in surfaces:
            raise RuntimeError(f"{label} selected candidate index is invalid")
        candidates = surfaces[candidate_key]
        if candidates.ndim != 2 or candidates.shape[1] != HORIZON or not 0 <= indices[0] < len(candidates):
            raise RuntimeError(f"{label} candidate schedule surface is invalid")
        if not _array_equal(candidates[indices[0]], vectors["q_by_time"]):
            raise RuntimeError(f"{label} selected common-grid candidate disagrees")
        candidate_coverage = surfaces[prefix + "candidate_coverage"]
        candidate_width = surfaces[prefix + "candidate_normalized_width"]
        scale_grid = surfaces[prefix + "scale_grid"]
        if (
            candidate_coverage.shape != candidates.shape
            or candidate_width.shape != candidates.shape
            or scale_grid.shape != (len(candidates),)
            or not np.isfinite(candidates).all()
            or not np.isfinite(candidate_coverage).all()
            or not np.isfinite(candidate_width).all()
            or not np.isfinite(scale_grid).all()
            or np.any(candidates <= 0)
            or np.any(candidate_width <= 0)
            or np.any((candidate_coverage < 0) | (candidate_coverage > 1))
        ):
            raise RuntimeError(f"{label} common-grid candidate surfaces are invalid")
    else:
        if endpoint or any(len(values) for values in vectors.values()) or indices:
            raise RuntimeError(f"{label} unavailable vectors/indices must be empty")
        if n_rollouts != 0 or not math.isnan(micro) or not math.isnan(patient):
            raise RuntimeError(f"{label} unavailable widths must be NaN with zero rollouts")
    return {
        "seed": seed,
        "scenario": scenario,
        "method": "Current Profiled Oracle (common grid)",
        "selection_available": available,
        "failure_stage": failure_stage,
        "selected_endpoint": endpoint,
        "micro_normalized_width": micro,
        "patient_normalized_width": patient,
        "tuning_seed": tuning_seed,
        "evaluation_seed": evaluation_seed,
        **vectors,
    }


def _validate_finite_mdp_sanity(root: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    path = root / "finite_mdp_sanity.json"
    if not path.is_file():
        raise RuntimeError("finite_mdp_sanity.json is required")
    value = _read_json(path, "finite-MDP sanity")
    exact = {
        "schema_version": 1,
        "diagnostic_type": "analytic_exact_finite_grid_search",
        "status": "complete",
        "non_gating": True,
        "population_exact": True,
        "dataset": "tabular",
        "horizon": 4,
        "grid_size": 5,
        "schedule_count": 625,
        "target_coverage": 0.9,
        "grid_source": "D_COT stagewise score quantiles frozen before search",
        "seed": 0,
        "device": "cuda:0",
        "gap_definition": "greedy_mean_stage_width_minus_exact_mean_stage_width",
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            if key == "non_gating":
                raise RuntimeError("finite-MDP sanity must be non-gating")
            raise RuntimeError(f"finite-MDP sanity {key} must equal {expected!r}")
    expected_scope = (
        "The true gap is relative only to the reduced frozen grid and "
        "the frozen predictor and policy."
    )
    if value.get("scope") != expected_scope:
        raise RuntimeError("finite-MDP sanity scope must retain the frozen-grid limitation")
    for key in ("source_tree_sha256", "experiment_tree_sha256"):
        if value.get(key) != provenance[key]:
            raise RuntimeError(f"finite-MDP sanity {key} differs from study")
    sanity_config = reduced_search_config(
        ExperimentConfig.from_yaml(SANITY_DEFAULT_CONFIG),
        device="cuda:0",
        output=Path("results/work/phase0a_finite_mdp_sanity.json"),
    )
    expected_config_hash = sanity_config_sha256(sanity_config.to_dict())
    if value.get("config_sha256") != expected_config_hash:
        raise RuntimeError("finite-MDP sanity config_sha256 differs from the frozen run")
    greedy_available = value.get("greedy_available")
    if type(greedy_available) is not bool:
        raise RuntimeError("finite-MDP sanity greedy_available must be boolean")
    branch_names = ("exact", "greedy") if greedy_available else ("exact",)
    branches: dict[str, dict[str, Any]] = {}
    for name in branch_names:
        branch = value.get(name)
        if not isinstance(branch, dict):
            raise RuntimeError(f"finite-MDP sanity {name} branch is missing")
        indices = branch.get("selected_indices")
        if not isinstance(indices, list) or len(indices) != 4 or any(
            type(index) is not int or not 0 <= index < 5 for index in indices
        ):
            raise RuntimeError(f"finite-MDP sanity {name} indices are invalid")
        arrays = {
            key: _numeric_list(branch.get(key), f"finite-MDP {name} {key}")
            for key in ("selected_radii", "coverage", "normalized_width_by_stage")
        }
        if any(len(array) != 4 or not np.isfinite(array).all() for array in arrays.values()):
            raise RuntimeError(f"finite-MDP sanity {name} vectors must be finite length four")
        if np.any(arrays["selected_radii"] <= 0) or np.any(arrays["normalized_width_by_stage"] <= 0):
            raise RuntimeError(f"finite-MDP sanity {name} radii/widths must be positive")
        if np.any((arrays["coverage"] < 0) | (arrays["coverage"] > 1)):
            raise RuntimeError(f"finite-MDP sanity {name} coverage must lie in [0, 1]")
        if np.any(arrays["coverage"] < value["target_coverage"] - 1e-12):
            raise RuntimeError(f"finite-MDP sanity {name} coverage is infeasible")
        mean_width = _float_value(branch.get("mean_normalized_width"), f"finite-MDP {name} mean width")
        if not math.isclose(mean_width, float(arrays["normalized_width_by_stage"].mean()), rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError(f"finite-MDP sanity {name} mean width disagrees")
        branches[name] = {**branch, **{key: array.tolist() for key, array in arrays.items()}}
    if greedy_available:
        gap = _float_value(value.get("true_finite_grid_gap"), "finite-MDP true gap")
        expected_gap = branches["greedy"]["mean_normalized_width"] - branches["exact"]["mean_normalized_width"]
        if gap < -1e-12 or not math.isclose(gap, expected_gap, rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError("finite-MDP true gap disagrees with exact-grid widths")
        greedy = branches["greedy"]
    else:
        if value.get("greedy") is not None or value.get("true_finite_grid_gap") is not None:
            raise RuntimeError("unavailable finite-MDP greedy result and gap must be null")
        greedy, gap = None, None
    return {**value, "greedy": greedy, "exact": branches["exact"], "true_finite_grid_gap": gap, "in_gate": False}


def _analyze(
    primary_rows: list[dict[str, Any]],
    common_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    sanity: dict[str, Any],
) -> dict[str, Any]:
    primary: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, Any] = {}
    sensitivity_scenarios: dict[str, Any] = {}
    for scenario in SCENARIOS:
        primary[scenario] = {}
        for method in METHODS:
            rows = [row for row in primary_rows if row["scenario"] == scenario and row["method"] == method]
            primary[scenario][method] = _method_summary(rows)
        current = [row for row in primary_rows if row["scenario"] == scenario and row["method"] == CURRENT]
        greedy = [row for row in primary_rows if row["scenario"] == scenario and row["method"] == GREEDY]
        common = [row for row in common_rows if row["scenario"] == scenario]
        comparisons[scenario] = _paired_rows(
            greedy,
            current,
            comparison_name=f"{scenario}: Greedy / exact-current Profiled",
        )
        common_summary = _method_summary(common)
        sensitivity_scenarios[scenario] = {
            "common_grid_profiled": common_summary,
            "greedy_over_common_grid": _paired_rows(
                greedy,
                common,
                comparison_name=f"{scenario}: Greedy / common-grid Profiled",
            ),
            "common_grid_over_exact_current": _paired_rows(
                common,
                current,
                comparison_name=f"{scenario}: common-grid / exact-current Profiled",
            ),
        }
    tail_coverage = primary["tail_shift"][GREEDY]["coverage"]
    standard_coverage = primary["standard"][GREEDY]["coverage"]
    tail_comparison = comparisons["tail_shift"]
    standard_comparison = comparisons["standard"]
    decision = evaluate_go_no_go(
        tail_greedy_stage_lcb=None if tail_coverage["lower"] is None else np.asarray(tail_coverage["lower"]),
        tail_micro_ratio=tail_comparison["micro"]["geometric_mean_ratio"],
        tail_micro_bootstrap_upper=tail_comparison["micro"]["ci_upper"],
        tail_patient_ratio=tail_comparison["patient"]["geometric_mean_ratio"],
        tail_greedy_selection_count=primary["tail_shift"][GREEDY]["selection_count"],
        tail_n_paired=tail_comparison["n_paired"],
        standard_greedy_stage_lcb=None if standard_coverage["lower"] is None else np.asarray(standard_coverage["lower"]),
        standard_micro_ratio=standard_comparison["micro"]["geometric_mean_ratio"],
        standard_n_paired=standard_comparison["n_paired"],
    )
    return {
        "schema_version": 1,
        "integrity": {
            "status": "validated_complete",
            "expected_seeds": 100,
            "completed_seeds": 100,
            "primary_rows": 400,
        },
        "provenance": provenance,
        "coverage_estimand": "conditional_on_successful_selection",
        "selection_denominator": 100,
        "bootstrap_contract": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "rng_seed": BOOTSTRAP_SEED,
            "pairing": "same_seed_successfully_selected_by_both_methods",
            "quantile_method": "linear_percentile_95",
        },
        "primary": primary,
        "primary_comparisons": comparisons,
        "sensitivity": {
            "analysis_role": "sensitivity_only_non_gating",
            "in_gate": False,
            "scenarios": sensitivity_scenarios,
        },
        "finite_mdp_sanity": sanity,
        "gates": decision["gates"],
        "decision": decision["decision"],
    }


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = sorted((row for row in rows if row["selection_available"]), key=lambda row: row["seed"])
    matrices = {
        name: np.stack([row[name] for row in selected]) if selected else np.empty((0, HORIZON))
        for name in ("final_coverage", "final_wilson_lcb", "final_stage_width", "q_by_time")
    }
    return {
        "n_total": 100,
        "selection_count": len(selected),
        "selection_rate": len(selected) / 100.0,
        "endpoint_count": sum(row["selected_endpoint"] for row in selected),
        "endpoint_rate_conditional": (
            None if not selected else sum(row["selected_endpoint"] for row in selected) / len(selected)
        ),
        "failure_stage_counts": {
            str(stage): sum(row["failure_stage"] == stage for row in rows) for stage in range(HORIZON)
        },
        "coverage": compute_coverage_summary(matrices["final_coverage"], horizon=HORIZON),
        "wilson_diagnostic_mean": (
            None if not selected else matrices["final_wilson_lcb"].mean(axis=0).tolist()
        ),
        "stage_width_mean": (
            None if not selected else matrices["final_stage_width"].mean(axis=0).tolist()
        ),
        "radius": _stage_mean_interval(matrices["q_by_time"]),
    }


def _stage_mean_interval(values: np.ndarray) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {"n_selected": 0, "mean": None, "lower": None, "upper": None, "interval_method": "two_sided_t_95"}
    mean = values.mean(axis=0)
    if n < 2:
        lower = upper = None
    else:
        half = stats.t.ppf(0.975, df=n - 1) * values.std(axis=0, ddof=1) / math.sqrt(n)
        lower, upper = (mean - half).tolist(), (mean + half).tolist()
    return {"n_selected": n, "mean": mean.tolist(), "lower": lower, "upper": upper, "interval_method": "two_sided_t_95"}


def _paired_rows(
    numerator_rows: list[dict[str, Any]],
    denominator_rows: list[dict[str, Any]],
    *,
    comparison_name: str,
) -> dict[str, Any]:
    numerator = {row["seed"]: row for row in numerator_rows if row["selection_available"]}
    denominator = {row["seed"]: row for row in denominator_rows if row["selection_available"]}
    seeds = sorted(set(numerator) & set(denominator))
    array = lambda rows, field: np.asarray([rows[seed][field] for seed in seeds], dtype=float)
    return compute_paired_width_inference(
        np.asarray(seeds, dtype=int),
        array(numerator, "micro_normalized_width"),
        array(denominator, "micro_normalized_width"),
        array(numerator, "patient_normalized_width"),
        array(denominator, "patient_normalized_width"),
        comparison_name=comparison_name,
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain an object")
    return value


def _parse_vector(value: Any, *, name: str) -> np.ndarray:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} must be a JSON vector") from error
    return _numeric_list(parsed, name)


def _numeric_list(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, list) or any(type(item) not in (int, float) for item in value):
        raise RuntimeError(f"{name} must be a numeric list")
    return np.asarray(value, dtype=np.float64)


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"{name} must be an integer")
    return value


def _is_exact_seed_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(EXPECTED_SEEDS)
        and all(type(seed) is int for seed in value)
        and value == list(EXPECTED_SEEDS)
    )


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) not in (bool, np.bool_):
        raise RuntimeError(f"{name} must be boolean")
    return bool(value)


def _failure_stage(value: Any, label: str) -> int | None:
    if value is None or (isinstance(value, (float, np.floating)) and math.isnan(float(value))):
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        stage = int(value)
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        stage = int(value)
    else:
        raise RuntimeError(f"{label} failure_stage must be integral or null")
    if not 0 <= stage < HORIZON:
        raise RuntimeError(f"{label} failure_stage is out of range")
    return stage


def _float_value(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise RuntimeError(f"{name} must be numeric")
    return float(value)


def _array_equal(first: np.ndarray, second: np.ndarray) -> bool:
    return first.shape == second.shape and bool(np.allclose(first, second, rtol=1e-5, atol=1e-7, equal_nan=True))


def _positive_finite(*values: float) -> bool:
    return all(math.isfinite(value) and value > 0.0 for value in values)


def publish_phase0_summary(input_dir: Path) -> dict[str, Any]:
    """Publish six payloads transactionally, with the manifest installed last."""

    root = Path(input_dir)
    analysis = load_validate_and_analyze(root)
    staging = Path(tempfile.mkdtemp(prefix=".phase0-summary-", dir=root))
    try:
        pd.DataFrame(_source_rows(analysis)).to_csv(
            staging / "phase0_summary.csv",
            index=False,
            float_format="%.17g",
        )
        (staging / "phase0_decision.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        (staging / "phase0_summary.md").write_text(_markdown_report(analysis))
        _render_figure(analysis, staging / "phase0_radius_and_coverage")
        missing = [name for name in OUTPUT_NAMES if not (staging / name).is_file()]
        if missing:
            raise RuntimeError(f"summary staging did not produce: {missing}")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "decision": analysis["decision"],
            "files": {
                name: {
                    "sha256": _file_sha256(staging / name),
                    "bytes": (staging / name).stat().st_size,
                }
                for name in OUTPUT_NAMES
            },
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        for name in OUTPUT_NAMES:
            _fsync_file(staging / name)
        _fsync_file(staging / MANIFEST_NAME)
        _publish_staged_bundle(staging, root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return analysis


def _publish_staged_bundle(staging: Path, root: Path) -> None:
    """Replace a complete bundle, restoring the preceding one on any failure."""

    backup = Path(tempfile.mkdtemp(prefix=".phase0-summary-backup-", dir=root))
    installed: list[str] = []
    backed_up: list[str] = []
    marker_backed_up = False
    marker_installed = False
    try:
        if (root / MANIFEST_NAME).exists():
            os.replace(root / MANIFEST_NAME, backup / MANIFEST_NAME)
            marker_backed_up = True
            _fsync_directory(root)
        for name in OUTPUT_NAMES:
            if (root / name).exists():
                os.replace(root / name, backup / name)
                backed_up.append(name)
        for name in OUTPUT_NAMES:
            os.replace(staging / name, root / name)
            installed.append(name)
        os.replace(staging / MANIFEST_NAME, root / MANIFEST_NAME)
        marker_installed = True
        _fsync_directory(root)
    except BaseException:
        if marker_installed and (root / MANIFEST_NAME).exists():
            (root / MANIFEST_NAME).unlink()
        for name in installed:
            destination = root / name
            if destination.exists():
                destination.unlink()
        for name in backed_up:
            os.replace(backup / name, root / name)
        if marker_backed_up:
            os.replace(backup / MANIFEST_NAME, root / MANIFEST_NAME)
        _fsync_directory(root)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    columns = (
        "row_type", "analysis_role", "scenario", "method", "comparator", "stage",
        "metric", "estimate", "lower", "upper", "n_total", "n_selected", "n_paired",
        "conditioning", "interval_method", "threshold", "operator", "passed",
    )
    rows: list[dict[str, Any]] = []

    def add(**values: Any) -> None:
        rows.append({column: values.get(column) for column in columns})

    def add_method(scenario: str, method: str, result: dict[str, Any], role: str) -> None:
        coverage, radius = result["coverage"], result["radius"]
        for stage in range(HORIZON):
            add(
                row_type="stage", analysis_role=role, scenario=scenario, method=method,
                stage=stage + 1, metric="seed_mean_coverage",
                estimate=None if coverage["mean"] is None else coverage["mean"][stage],
                lower=None if coverage["lower"] is None else coverage["lower"][stage],
                n_total=100, n_selected=coverage["n_selected"],
                conditioning="conditional_on_successful_selection",
                interval_method="seed_mean_bonferroni_t_lcb",
            )
            add(
                row_type="stage_diagnostic", analysis_role=role, scenario=scenario, method=method,
                stage=stage + 1, metric="mean_seed_level_wilson_lcb",
                estimate=None if result["wilson_diagnostic_mean"] is None else result["wilson_diagnostic_mean"][stage],
                n_total=100, n_selected=coverage["n_selected"],
                conditioning="conditional_on_successful_selection",
                interval_method="seed_level_mc_diagnostic_not_gate",
            )
            add(
                row_type="stage", analysis_role=role, scenario=scenario, method=method,
                stage=stage + 1, metric="mean_selected_radius",
                estimate=None if radius["mean"] is None else radius["mean"][stage],
                lower=None if radius["lower"] is None else radius["lower"][stage],
                upper=None if radius["upper"] is None else radius["upper"][stage],
                n_total=100, n_selected=radius["n_selected"],
                conditioning="conditional_on_successful_selection",
                interval_method="two_sided_t_95",
            )
            add(
                row_type="stage", analysis_role=role, scenario=scenario, method=method,
                stage=stage + 1, metric="mean_normalized_width",
                estimate=None if result["stage_width_mean"] is None else result["stage_width_mean"][stage],
                n_total=100, n_selected=coverage["n_selected"],
                conditioning="conditional_on_successful_selection",
            )
        for metric in (
            "minimum_stage_seed_mean_coverage",
            "minimum_stage_seed_mean_simultaneous_lcb",
            "mean_seed_minimum_stage_coverage",
            "minimum_seed_stage_coverage",
        ):
            add(
                row_type="coverage_worst_definition", analysis_role=role, scenario=scenario,
                method=method, metric=metric, estimate=coverage[metric], n_total=100,
                n_selected=coverage["n_selected"],
                conditioning="conditional_on_successful_selection",
                interval_method=("seed_mean_bonferroni_t_lcb" if metric.endswith("simultaneous_lcb") else "descriptive"),
            )
        add(
            row_type="selection", analysis_role=role, scenario=scenario, method=method,
            metric="selection_rate", estimate=result["selection_rate"], n_total=100,
            n_selected=result["selection_count"], conditioning="all_registered_seeds",
        )
        add(
            row_type="boundary", analysis_role=role, scenario=scenario, method=method,
            metric="selected_endpoint_rate", estimate=result["endpoint_rate_conditional"],
            n_total=100, n_selected=result["selection_count"],
            conditioning="conditional_on_successful_selection",
        )

    def add_comparison(scenario: str, result: dict[str, Any], role: str) -> None:
        numerator, denominator = result["comparison_name"].split(": ", 1)[1].split(" / ", 1)
        for width_type in ("micro", "patient"):
            value = result[width_type]
            add(
                row_type="width_ratio", analysis_role=role, scenario=scenario,
                method=numerator, comparator=denominator, metric=f"geometric_{width_type}_width_ratio",
                estimate=value["geometric_mean_ratio"], lower=value["ci_lower"], upper=value["ci_upper"],
                n_total=100, n_paired=result["n_paired"],
                conditioning="same_seed_both_methods_selected",
                interval_method="paired_seed_bootstrap_10000_linear_percentile_95",
            )

    for scenario in SCENARIOS:
        for method in METHODS:
            add_method(scenario, method, analysis["primary"][scenario][method], "primary")
        add_comparison(scenario, analysis["primary_comparisons"][scenario], "primary")
        sensitivity = analysis["sensitivity"]["scenarios"][scenario]
        add_method(scenario, "Current Profiled Oracle (common grid)", sensitivity["common_grid_profiled"], "sensitivity_only_non_gating")
        add_comparison(scenario, sensitivity["greedy_over_common_grid"], "sensitivity_only_non_gating")
        add_comparison(scenario, sensitivity["common_grid_over_exact_current"], "sensitivity_only_non_gating")
    for gate in analysis["gates"]:
        add(
            row_type="gate", analysis_role="preregistered_gate", metric=gate["id"],
            estimate=gate["value"], threshold=gate["threshold"], operator=gate["operator"],
            passed=gate["passed"], n_total=100,
        )
    add(
        row_type="finite_mdp_sanity", analysis_role="diagnostic_non_gating",
        metric="true_finite_grid_gap", estimate=analysis["finite_mdp_sanity"]["true_finite_grid_gap"],
    )
    return rows


def _markdown_report(analysis: dict[str, Any]) -> str:
    lines = [
        f"# Phase 0A decision: {analysis['decision']}",
        "",
        "## Integrity and estimand",
        "",
        "Validated all 100 registered seeds and exactly 400 primary rows before computing statistics. "
        "Coverage and width are conditional on successful selection; selection counts use all 100 seeds. "
        "The experiment is fixed T=12 with all patients active.",
        "",
        "## Preregistered gates",
        "",
        "| Gate | Value | Rule | Result |",
        "|---|---:|:---:|:---:|",
    ]
    for gate in analysis["gates"]:
        value = "unavailable" if gate["value"] is None else _format_number(gate["value"])
        lines.append(f"| {gate['id']} | {value} | {gate['operator']} {gate['threshold']} | {'PASS' if gate['passed'] else 'FAIL'} |")
    lines += [
        "",
        "## Four distinct coverage minima",
        "",
        "These quantities are not interchangeable: the minimum stage seed mean, the minimum simultaneous "
        "LCB (the coverage gate), the mean seedwise minimum, and the raw seed-stage minimum.",
        "",
        "| Scenario | Method | Min stage mean | Min simultaneous LCB | Mean seed minimum | Raw minimum |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for method in METHODS:
            coverage = analysis["primary"][scenario][method]["coverage"]
            values = [_format_number(coverage[key]) for key in (
                "minimum_stage_seed_mean_coverage",
                "minimum_stage_seed_mean_simultaneous_lcb",
                "mean_seed_minimum_stage_coverage",
                "minimum_seed_stage_coverage",
            )]
            lines.append(f"| {scenario} | {method} | {' | '.join(values)} |")
    lines += [
        "",
        "## Primary paired width results",
        "",
        "| Scenario | Paired seeds | Micro ratio (95% CI) | Patient ratio (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        result = analysis["primary_comparisons"][scenario]
        lines.append(
            f"| {scenario} | {result['n_paired']} | {_ratio_ci(result['micro'])} | {_ratio_ci(result['patient'])} |"
        )
    lines += [
        "",
        "## Per-stage seed-mean coverage and simultaneous LCB",
        "",
        "| Scenario | Stage | Current mean / LCB | Greedy mean / LCB |",
        "|---|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        current = analysis["primary"][scenario][CURRENT]["coverage"]
        greedy = analysis["primary"][scenario][GREEDY]["coverage"]
        for stage in range(HORIZON):
            lines.append(
                f"| {scenario} | {stage + 1} | {_stage_pair(current, stage)} | {_stage_pair(greedy, stage)} |"
            )
    lines += [
        "",
        "## Selection and endpoint audit",
        "",
        "| Scenario | Method | Selected / 100 | Endpoint / selected |",
        "|---|---|---:|---:|",
    ]
    for scenario in SCENARIOS:
        sensitivity = analysis["sensitivity"]["scenarios"][scenario]
        for method, result in (
            ("Current", analysis["primary"][scenario][CURRENT]),
            ("Greedy", analysis["primary"][scenario][GREEDY]),
            ("Common-grid", sensitivity["common_grid_profiled"]),
        ):
            endpoint = _format_number(result["endpoint_rate_conditional"])
            lines.append(f"| {scenario} | {method} | {result['selection_count']} | {endpoint} |")
    lines += [
        "",
        "## Common-grid sensitivity",
        "",
        "Common-grid Profiled results are sensitivity-only and never enter a gate.",
        "",
        "| Scenario | Common selected | Common min mean / LCB | Greedy / common-grid micro ratio | Common-grid / exact-current micro ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        sensitivity = analysis["sensitivity"]["scenarios"][scenario]
        common = sensitivity["common_grid_profiled"]
        coverage = common["coverage"]
        coverage_pair = (
            f"{_format_number(coverage['minimum_stage_seed_mean_coverage'])} / "
            f"{_format_number(coverage['minimum_stage_seed_mean_simultaneous_lcb'])}"
        )
        lines.append(
            f"| {scenario} | {common['selection_count']} | {coverage_pair} | "
            f"{_ratio_ci(sensitivity['greedy_over_common_grid']['micro'])}; "
            f"n={sensitivity['greedy_over_common_grid']['n_paired']} | "
            f"{_ratio_ci(sensitivity['common_grid_over_exact_current']['micro'])}; "
            f"n={sensitivity['common_grid_over_exact_current']['n_paired']} |"
        )
    gap = analysis["finite_mdp_sanity"]["true_finite_grid_gap"]
    lines += [
        "",
        "## Finite-MDP diagnostic",
        "",
        "The diagnostic is population-exact only on the reduced frozen grid and frozen predictor/policy, and is non-gating.",
        f"Finite-MDP true grid gap: {_format_number(gap)}",
        "",
        "## Interpretation limits",
        "",
        "The oracle is not a deployable method; greedy search is not globally optimal. Under fixed T=12, micro "
        "and patient width coincide theoretically, although both preregistered thresholds are retained. A GO does "
        "not establish state of the art, and a NO_GO preserves the current method rather than inviting threshold changes.",
        "",
    ]
    return "\n".join(lines)


def _format_number(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, list):
        return _format_number(min(value))
    return f"{float(value):.6f}"


def _ratio_ci(value: dict[str, Any]) -> str:
    if value["geometric_mean_ratio"] is None:
        return "unavailable"
    if value["ci_lower"] is None:
        return f"{value['geometric_mean_ratio']:.4f} (CI unavailable)"
    return f"{value['geometric_mean_ratio']:.4f} ({value['ci_lower']:.4f}, {value['ci_upper']:.4f})"


def _stage_pair(value: dict[str, Any], stage: int) -> str:
    if value["mean"] is None:
        return "unavailable"
    lower = "unavailable" if value["lower"] is None else f"{value['lower'][stage]:.4f}"
    return f"{value['mean'][stage]:.4f} / {lower}"


def _render_figure(analysis: dict[str, Any], output_base: Path) -> None:
    colors = {CURRENT: "#6E6E6E", GREEDY: "#2F6DAE", "common": "#C59A3D"}
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "svg.hashsalt": "phase0-oracle-summary-v1",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "legend.fontsize": 5.7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplot_mosaic(
        [["A", "A", "B"], ["C", "D", "E"]],
        figsize=(183 / 25.4, 120 / 25.4),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.95]},
        constrained_layout=True,
    )
    try:
        layout_engine = fig.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(rect=(0.0, 0.08, 1.0, 0.68))
        coverage_limits = _coverage_limits(analysis)
        _coverage_panel(axes["A"], analysis, "tail_shift", colors, coverage_limits)
        _ratio_panel(axes["B"], analysis, colors)
        _coverage_panel(axes["C"], analysis, "standard", colors, coverage_limits)
        _radius_panel(axes["D"], analysis, colors)
        _selection_panel(axes["E"], analysis, colors)
        coverage_handles: dict[str, Any] = {}
        for panel in ("A", "C"):
            handles, labels = axes[panel].get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                coverage_handles.setdefault(label, handle)
        coverage_labels = [
            label for label in COVERAGE_LEGEND_ORDER if label in coverage_handles
        ]
        coverage_legend = fig.legend(
            [coverage_handles[label] for label in coverage_labels],
            coverage_labels,
            ncol=len(coverage_labels),
            loc="center",
            bbox_to_anchor=(0.5, 0.865),
            bbox_transform=fig.transFigure,
            columnspacing=1.2,
            handlelength=2.0,
        )
        coverage_legend.set_in_layout(False)
        radius_handles, radius_labels = axes["D"].get_legend_handles_labels()
        radius_legend = fig.legend(
            radius_handles,
            radius_labels,
            ncol=len(radius_labels),
            loc="center",
            bbox_to_anchor=(0.5, 0.815),
            bbox_transform=fig.transFigure,
            columnspacing=1.4,
            handlelength=2.0,
        )
        radius_legend.set_in_layout(False)
        for label, axis in axes.items():
            axis.text(-0.12, 1.02, label.lower(), transform=axis.transAxes, fontweight="bold", fontsize=8)
        display_decision = "NO-GO" if analysis["decision"] == "NO_GO" else analysis["decision"]
        fig.suptitle(
            display_decision, y=0.985, fontsize=9, fontweight="bold", color="#272727"
        )
        _draw_gate_strip(fig, analysis)
        fig.text(
            0.5,
            0.018,
            "Panels a,c: dashed = seed-mean Bonferroni-t simultaneous LCB. "
            "Coverage/width conditional on selection; selection denominator = 100. "
            "CI = 10,000 paired-seed resamples.",
            ha="center",
            va="bottom",
            fontsize=5.2,
            color="#4D4D4D",
        )
        fig.savefig(
            output_base.with_suffix(".svg"),
            metadata={"Date": "2026-08-17", "Creator": "SC-PCP Phase 0 summary"},
        )
        fig.savefig(
            output_base.with_suffix(".pdf"),
            metadata={"CreationDate": None, "ModDate": None, "Creator": "SC-PCP Phase 0 summary"},
        )
        fig.savefig(
            output_base.with_suffix(".png"),
            dpi=300,
            metadata={"Software": "SC-PCP Phase 0 summary"},
        )
    finally:
        plt.close(fig)


def _gate_strip_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Format the seven already-computed gates without re-evaluating them."""

    def compact(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, list):
            value = min(value)
        if type(value) is int:
            return str(value)
        return f"{float(value):.4f}"

    rows = []
    for gate in analysis["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        value = compact(gate["value"])
        threshold = compact(gate["threshold"])
        rows.append(
            {
                "id": gate["id"],
                "status": status,
                "passed": gate["passed"],
                "detail": f"{gate['operator']}{threshold} | v={value} | {status}",
            }
        )
    return rows


def _draw_gate_strip(fig: Any, analysis: dict[str, Any]) -> None:
    rows = _gate_strip_rows(analysis)
    left, right = 0.035, 0.985
    bottom, top = 0.895, 0.955
    width = (right - left) / len(rows)
    line_color = "#D5D5D5"
    for y in (bottom, top):
        artist = Line2D(
            [left, right], [y, y], transform=fig.transFigure, color=line_color, lw=0.45
        )
        artist.set_in_layout(False)
        fig.add_artist(artist)
    for index, row in enumerate(rows):
        cell_left = left + index * width
        center = cell_left + width / 2.0
        if index:
            artist = Line2D(
                [cell_left, cell_left],
                [bottom, top],
                transform=fig.transFigure,
                color=line_color,
                lw=0.35,
            )
            artist.set_in_layout(False)
            fig.add_artist(artist)
        marker = Line2D(
            [cell_left + 0.007],
            [0.940],
            transform=fig.transFigure,
            marker="o",
            markersize=2.8,
            linestyle="none",
            color="#2E7D32" if row["passed"] else "#B3261E",
        )
        marker.set_in_layout(False)
        fig.add_artist(marker)
        identifier = fig.text(
            center,
            0.940,
            row["id"],
            ha="center",
            va="center",
            fontsize=4.2,
            color="#272727",
        )
        identifier.set_in_layout(False)
        detail = fig.text(
            center,
            0.912,
            row["detail"],
            ha="center",
            va="center",
            fontsize=3.9,
            color="#272727",
        )
        detail.set_in_layout(False)


def _coverage_limits(analysis: dict[str, Any]) -> tuple[float, float]:
    values = [0.9]
    for scenario in SCENARIOS:
        for method in METHODS:
            coverage = analysis["primary"][scenario][method]["coverage"]
            for key in ("mean", "lower"):
                if coverage[key] is not None:
                    values.extend(coverage[key])
    low, high = min(values), max(values)
    padding = max(0.008, 0.12 * (high - low or 0.05))
    return low - padding, high + padding


def _coverage_panel(
    axis: Any,
    analysis: dict[str, Any],
    scenario: str,
    colors: dict[str, str],
    limits: tuple[float, float],
) -> None:
    stages = np.arange(1, HORIZON + 1)
    for method in METHODS:
        result = analysis["primary"][scenario][method]
        coverage = result["coverage"]
        if coverage["mean"] is None:
            continue
        short = "Current" if method == CURRENT else "Greedy"
        axis.plot(stages, coverage["mean"], color=colors[method], marker="o", ms=2.5, lw=1.3, label=f"{short} mean")
        if coverage["lower"] is not None:
            axis.plot(stages, coverage["lower"], color=colors[method], ls="--", lw=1.0, label=f"{short} simultaneous LCB")
        minimum = coverage["minimum_stage_seed_mean_simultaneous_lcb"]
        note = "LCB unavailable" if minimum is None else f"min LCB {minimum:.3f}"
        axis.text(
            0.02,
            0.98 - (0.16 if method == GREEDY else 0.0),
            f"{short}: {note}; n={result['selection_count']}",
            transform=axis.transAxes,
            va="top",
            color=colors[method],
            fontsize=5.5,
        )
    axis.axhline(0.90, color="#272727", lw=0.8, ls=":", label="0.90 target")
    title = {"tail_shift": "Tail shift", "standard": "Standard"}[scenario]
    axis.set(xlim=(0.7, 12.3), ylim=limits, xlabel="Stage", ylabel="Fresh coverage", title=title)
    axis.set_xticks([1, 3, 6, 9, 12])


def _ratio_panel(axis: Any, analysis: dict[str, Any], colors: dict[str, str]) -> None:
    entries: list[tuple[str, dict[str, Any], str, str]] = []
    for scenario in SCENARIOS:
        primary = analysis["primary_comparisons"][scenario]
        entries.append((f"{scenario}\nG / Current (n={primary['n_paired']})", primary, colors[GREEDY], "o"))
        sensitivity = analysis["sensitivity"]["scenarios"][scenario]
        greedy_common = sensitivity["greedy_over_common_grid"]
        entries.append((f"{scenario}\nG / common (n={greedy_common['n_paired']})", greedy_common, colors["common"], "s"))
        common_current = sensitivity["common_grid_over_exact_current"]
        entries.append((f"{scenario}\ncommon / Current (n={common_current['n_paired']})", common_current, colors["common"], "D"))
    plotted = [0.90, 1.0]
    for y, (label, result, color, marker) in enumerate(reversed(entries)):
        metric = result["micro"]
        point, lower, upper = metric["geometric_mean_ratio"], metric["ci_lower"], metric["ci_upper"]
        if point is None:
            continue
        if lower is not None:
            axis.plot([lower, upper], [y, y], color=color, lw=1.1)
            plotted.extend((lower, upper))
        axis.plot(point, y, marker=marker, ms=3.5, color=color)
        plotted.append(point)
    axis.axvline(1.0, color="#4D4D4D", lw=0.8, ls="--")
    axis.axvline(0.90, color="#2F6DAE", lw=0.7, ls=":")
    axis.set_yticks(range(len(entries)), [entry[0] for entry in reversed(entries)])
    padding = max(0.015, 0.08 * (max(plotted) - min(plotted)))
    axis.set(xlim=(min(plotted) - padding, max(plotted) + padding), xlabel="Geometric micro-width ratio", title="Paired width")
    axis.tick_params(axis="y", labelsize=5)


def _radius_panel(axis: Any, analysis: dict[str, Any], colors: dict[str, str]) -> None:
    stages = np.arange(1, HORIZON + 1)
    series = [
        ("Current", analysis["primary"]["tail_shift"][CURRENT]["radius"], colors[CURRENT], "-"),
        ("Greedy", analysis["primary"]["tail_shift"][GREEDY]["radius"], colors[GREEDY], "-"),
        (
            "Common-grid",
            analysis["sensitivity"]["scenarios"]["tail_shift"]["common_grid_profiled"]["radius"],
            colors["common"],
            "--",
        ),
    ]
    for label, result, color, linestyle in series:
        if result["mean"] is None:
            continue
        axis.plot(stages, result["mean"], color=color, ls=linestyle, lw=1.2, label=f"{label} (n={result['n_selected']})")
        if result["lower"] is not None:
            axis.fill_between(stages, result["lower"], result["upper"], color=color, alpha=0.12, linewidth=0)
    axis.set(xlabel="Stage", ylabel="Selected radius $q_t$", title="Tail-shift radii")
    axis.set_xticks([1, 3, 6, 9, 12])


def _selection_panel(axis: Any, analysis: dict[str, Any], colors: dict[str, str]) -> None:
    labels, selected, endpoints, bar_colors = [], [], [], []
    for scenario, prefix in (("standard", "Std"), ("tail_shift", "Tail")):
        sensitivity = analysis["sensitivity"]["scenarios"][scenario]
        results = (
            ("Current", analysis["primary"][scenario][CURRENT], colors[CURRENT]),
            ("Greedy", analysis["primary"][scenario][GREEDY], colors[GREEDY]),
            ("Common", sensitivity["common_grid_profiled"], colors["common"]),
        )
        for method, result, color in results:
            method_label = {"Current": "Cur", "Greedy": "Grd", "Common": "Com"}[method]
            labels.append(f"{prefix}\n{method_label}")
            selected.append(result["selection_rate"])
            endpoints.append(result["endpoint_rate_conditional"])
            bar_colors.append(color)
    x = np.arange(len(labels))
    axis.bar(x, selected, color=bar_colors, alpha=0.75, width=0.68, label="Selected / 100")
    valid = [(index, value) for index, value in enumerate(endpoints) if value is not None]
    if valid:
        axis.scatter([item[0] for item in valid], [item[1] for item in valid], color="#272727", marker="D", s=10, label="Endpoint / selected", zorder=3)
    axis.set_xticks(x, labels, rotation=0, ha="center")
    axis.tick_params(axis="x", labelsize=5.2)
    axis.set(ylim=(-0.04, 1.07), ylabel="Rate", title="Selection bars\nEndpoint diamonds")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = publish_phase0_summary(args.input_dir)
    print(result["decision"])


if __name__ == "__main__":
    main()
