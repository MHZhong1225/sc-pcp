"""Run the isolated K=101 versus K=401 historical PAC-grid audit."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import (  # noqa: E402
    experiment_tree_sha256,
    mark_study_complete,
    mark_study_failed,
    source_tree_sha256,
    write_seed_result,
    write_study_metadata,
)
from scpcp.config import ExperimentConfig  # noqa: E402
from scpcp.device import resolve_devices  # noqa: E402
from scpcp.pac_grid_study import (  # noqa: E402
    BASE_METHOD,
    DENSE_METHOD,
    run_paired_grid_seed,
)


PROTOCOL = "paired_nested_pac_grid_v1"
DEFAULT_CONFIG = ROOT / "configs" / "phase0_oracle.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "work" / "pac_grid_refinement_20seed"
DEFAULT_SEEDS = tuple(range(20))
METHODS = (BASE_METHOD, DENSE_METHOD)
SEED_DIRECTORY = re.compile(r"seed_(\d{5})")
SUBDIVISIONS = 4
BASE_GRID_SIZE = 101
DENSE_GRID_SIZE = 401
EVALUATION_STREAM = 1_700_001
EVALUATION_ROLLOUTS = 50_000
TARGET = 0.90
DELTA = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 271_828
CERTIFICATE_BOOTSTRAP_RESAMPLES = 2_000

# Frozen before the 20-seed diagnostic.  These thresholds must not be tuned to
# the realized study output.
EXPECTED_DEVELOPMENT_SEEDS = 20
MAXIMUM_WIDTH_RATIO = 0.995
MAXIMUM_WIDTH_RATIO_UCB = 1.0
MAXIMUM_WSC_LOSS = 0.002
WEIGHT_PARITY_TOLERANCE = 0.0
NUMERIC_PARITY_TOLERANCE = 1e-6
FLOAT32_AGGREGATION_TOLERANCE = NUMERIC_PARITY_TOLERANCE
# Re-fitting the GPU pilot COT is deterministic at the selected grid-index
# level but can move the normalized profile by a few float32 ULPs.  This
# tolerance is over two orders of magnitude below one adjacent grid step and
# is used only for replay provenance, never for candidate selection.
REFERENCE_SCHEDULE_TOLERANCE = 1e-4
MAXIMUM_DENSE_CAP_HIT_RATE = 0.01
MINIMUM_DENSE_EFFECTIVE_SAMPLE_SIZE = 25.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare nested K=101/K=401 grids in the retired practical "
            "ordered-IUT path; this does not run the paper SC-PCP method"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--seeds",
        default=None,
        help="defaults to 0:20; accepts a range such as 0:20 or comma-separated seeds",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-separated CUDA devices (defaults to the config's cuda:0,cuda:1)",
    )
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-decomposition",
        type=Path,
        default=None,
        help="optional frozen A/C/D/E root used to audit K=101 replay of old E",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.workers_per_device < 1:
        parser.error("--workers-per-device must be positive")

    base = ExperimentConfig.from_yaml(args.config)
    seeds = parse_seeds(args.seeds, DEFAULT_SEEDS)
    devices = resolve_devices(args.devices or base.devices)
    output_dir = args.output_dir.resolve()
    reference_decomposition = (
        None
        if args.reference_decomposition is None
        else args.reference_decomposition.resolve()
    )
    config = base.with_overrides(
        seeds=seeds,
        devices=devices,
        output_dir=output_dir,
    )
    run_config(
        config,
        output_dir,
        workers_per_device=args.workers_per_device,
        resume=args.resume,
        reference_decomposition=reference_decomposition,
    )
    print(output_dir)


def run_config(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    workers_per_device: int,
    resume: bool,
    reference_decomposition: Path | None = None,
) -> None:
    """Run or fail-closed resume one exact paired-grid configuration."""

    _validate_protocol_config(config, output_dir)
    if workers_per_device < 1:
        raise ValueError("workers_per_device must be positive")

    config_hash = canonical_config_sha256(config.to_dict())
    current_source_hash = source_tree_sha256()
    current_experiment_hash = experiment_tree_sha256()
    reference_fingerprint = (
        None
        if reference_decomposition is None
        else _reference_decomposition_fingerprint(
            reference_decomposition,
            config.seeds,
            horizon=config.horizon,
        )
    )
    execution = {
        "protocol": PROTOCOL,
        "experiment_tree_sha256": current_experiment_hash,
        "config_sha256": config_hash,
        "workers_per_device": workers_per_device,
        "subdivisions": SUBDIVISIONS,
        "evaluation_stream": EVALUATION_STREAM,
        "certificate_patient_bootstrap_resamples": (
            config.certification.practical_bootstrap_resamples
        ),
        "summary_seed_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "summary_seed_bootstrap_seed": BOOTSTRAP_SEED,
        "reference_decomposition": (
            None if reference_decomposition is None else str(reference_decomposition)
        ),
        "reference_decomposition_fingerprint": reference_fingerprint,
        "acceptance_gate": _acceptance_gate_definition(),
    }

    if resume:
        _validate_resume_provenance(
            output_dir,
            config,
            execution=execution,
            source_hash=current_source_hash,
            config_hash=config_hash,
        )
        completed = _validated_existing_seeds(
            output_dir,
            config.seeds,
            expected_source_hash=current_source_hash,
            expected_config_hash=config_hash,
            horizon=config.horizon,
        )
        if (output_dir / "COMPLETE").is_file() and completed != set(config.seeds):
            missing = sorted(set(config.seeds) - completed)
            raise RuntimeError(
                f"study COMPLETE exists but seed artifacts are missing: {missing}"
            )
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh PAC-grid output already exists: {output_dir}")
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
                horizon=config.horizon,
                expected_source_hash=current_source_hash,
                expected_config_hash=config_hash,
            )
        summary = write_summary(
            output_dir,
            config.seeds,
            horizon=config.horizon,
            reference_decomposition=reference_decomposition,
        )
        reference_replay = summary.get("reference_decomposition_replay")
        if (
            reference_decomposition is not None
            and isinstance(reference_replay, dict)
            and not reference_replay["replay_within_tolerance"]
        ):
            raise RuntimeError(
                "K=101 did not replay the frozen decomposition E schedule/index "
                "within the frozen numerical tolerance"
            )
        mark_study_complete(output_dir, config.seeds)
    except BaseException as error:
        mark_study_failed(output_dir, config.seeds, error)
        raise


def _validate_protocol_config(config: ExperimentConfig, output_dir: Path) -> None:
    if config.output_dir != output_dir:
        raise ValueError("config output_dir must exactly match output_dir")
    if config.data.dataset != "synthetic" or config.synthetic.scenario != "standard":
        raise ValueError("the paired-grid audit requires standard synthetic data")
    if config.q_grid_size != BASE_GRID_SIZE:
        raise ValueError("the paired-grid audit requires the frozen K=101 base grid")
    if config.samples.oracle_rollouts != EVALUATION_ROLLOUTS:
        raise ValueError("the paired-grid audit requires 50,000 fresh CRN rollouts")
    if not np.isclose(1.0 - config.certification.alpha, TARGET):
        raise ValueError("the paired-grid audit keeps target coverage fixed at 0.90")
    if not np.isclose(config.certification.delta, DELTA):
        raise ValueError("the paired-grid audit keeps delta fixed at 0.05")
    if (
        config.certification.practical_bootstrap_resamples
        != CERTIFICATE_BOOTSTRAP_RESAMPLES
    ):
        raise ValueError("the paired-grid audit requires 2,000 certificate resamples")


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
    calls = tuple(
        (worker_index, (config, seed, device, output_dir))
        for worker_index, seed, device in jobs
    )
    for result in _execute_jobs(
        worker_devices,
        calls,
        worker_function=_run_and_write,
    ):
        print(result, flush=True)


def _run_and_write(
    config: ExperimentConfig,
    seed: int,
    device: str,
    output_dir: Path,
) -> str:
    def run_and_publish() -> str:
        result = run_paired_grid_seed(
            config,
            seed=seed,
            device=device,
            subdivisions=SUBDIVISIONS,
            evaluation_stream=EVALUATION_STREAM,
        )
        seed_dir = write_seed_result(result, output_dir, config)
        validate_seed_artifact(seed_dir, seed, horizon=config.horizon)
        return str(seed_dir)

    if not device.startswith("cuda"):
        return run_and_publish()
    cuda_device = torch.device(device)
    torch.cuda.set_device(cuda_device)
    with torch.cuda.device(cuda_device):
        try:
            return run_and_publish()
        finally:
            torch.cuda.empty_cache()


def validate_seed_artifact(
    seed_dir: Path,
    seed: int,
    *,
    horizon: int,
    expected_source_hash: str | None = None,
    expected_config_hash: str | None = None,
) -> Path:
    """Validate the exact two-row artifact contract and CRN/parity invariants."""

    required = ("COMPLETE", "records.csv", "surfaces.npz", "metadata.json")
    missing = [name for name in required if not (seed_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"seed {seed} artifact is partial; missing files: {missing}")

    metadata = _read_json(seed_dir / "metadata.json")
    if type(metadata.get("seed")) is not int or metadata["seed"] != seed:
        raise RuntimeError(f"seed {seed} metadata has the wrong seed ID")
    if (
        expected_source_hash is not None
        and metadata.get("source_tree_sha256") != expected_source_hash
    ):
        raise RuntimeError(f"seed {seed} source hash differs from the study source")
    if expected_config_hash is not None:
        stored_config = metadata.get("config")
        if not isinstance(stored_config, dict):
            raise RuntimeError(f"seed {seed} metadata config must contain an object")
        if canonical_config_sha256(stored_config) != expected_config_hash:
            raise RuntimeError(f"seed {seed} config differs from the study config")

    diagnostics = metadata.get("diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("protocol") != PROTOCOL:
        raise RuntimeError(f"seed {seed} has the wrong diagnostic protocol")
    if not np.isclose(float(diagnostics.get("target", np.nan)), TARGET):
        raise RuntimeError(f"seed {seed} changed the target coverage")
    if not np.isclose(float(diagnostics.get("delta", np.nan)), DELTA):
        raise RuntimeError(f"seed {seed} changed delta")
    if diagnostics.get("base_grid_size") != BASE_GRID_SIZE:
        raise RuntimeError(f"seed {seed} has the wrong base-grid size")
    if diagnostics.get("dense_grid_size") != DENSE_GRID_SIZE:
        raise RuntimeError(f"seed {seed} has the wrong dense-grid size")
    if diagnostics.get("evaluation_rollouts") != EVALUATION_ROLLOUTS:
        raise RuntimeError(f"seed {seed} has the wrong fresh rollout count")
    for name in (
        "maximum_base_point_parity_error",
        "maximum_base_lcb_parity_error",
        "maximum_base_width_parity_error",
    ):
        if float(diagnostics.get(name, np.inf)) > NUMERIC_PARITY_TOLERANCE:
            raise RuntimeError(f"seed {seed} exceeds the tolerance for {name}")
    if diagnostics.get("maximum_base_weight_parity_error") != WEIGHT_PARITY_TOLERANCE:
        raise RuntimeError(f"seed {seed} reports nonzero base-weight parity error")
    if diagnostics.get("base_selection_matches_dense_base_knots") is not True:
        raise RuntimeError(f"seed {seed} base selection does not replay dense base knots")

    try:
        records = pd.read_csv(seed_dir / "records.csv")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise RuntimeError(f"seed {seed} records.csv is unreadable: {error}") from error
    required_columns = {
        "seed",
        "method",
        "grid_size",
        "target_coverage",
        "delta",
        "confidence_level",
        "selection_available",
        "selected_index",
        "selected_scale",
        "fresh_per_time_coverage",
        "fresh_average_normalized_width",
        "fresh_per_time_normalized_width",
        "evaluation_seed",
        "evaluation_rollouts",
        "certificate_type",
        "certificate_formal",
    }
    missing_columns = sorted(required_columns - set(records.columns))
    if missing_columns:
        raise RuntimeError(f"seed {seed} records are missing columns: {missing_columns}")
    if len(records) != 2 or tuple(records["method"]) != METHODS:
        raise RuntimeError(f"seed {seed} must contain ordered K=101 and K=401 rows")
    if not records["seed"].eq(seed).all():
        raise RuntimeError(f"seed {seed} records contain a different seed ID")
    if not records["selection_available"].eq(True).all():  # noqa: E712
        raise RuntimeError(f"seed {seed} has an unavailable selection")
    if tuple(records["grid_size"].astype(int)) != (BASE_GRID_SIZE, DENSE_GRID_SIZE):
        raise RuntimeError(f"seed {seed} records have wrong grid sizes")
    if not np.allclose(records["target_coverage"], TARGET, atol=0.0, rtol=0.0):
        raise RuntimeError(f"seed {seed} records changed target coverage")
    if not np.allclose(records["delta"], DELTA, atol=0.0, rtol=0.0):
        raise RuntimeError(f"seed {seed} records changed delta")
    if not np.allclose(records["confidence_level"], 1.0 - DELTA, atol=0.0, rtol=0.0):
        raise RuntimeError(f"seed {seed} records changed confidence level")
    if records["evaluation_seed"].nunique() != 1:
        raise RuntimeError(f"seed {seed} methods do not share one fresh CRN stream")
    if not records["evaluation_rollouts"].eq(EVALUATION_ROLLOUTS).all():
        raise RuntimeError(f"seed {seed} records have wrong fresh rollout count")
    if records["certificate_type"].nunique() != 1:
        raise RuntimeError(f"seed {seed} certificate labels differ between grids")
    if records["certificate_formal"].nunique() != 1:
        raise RuntimeError(f"seed {seed} certificate formality differs between grids")

    try:
        with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as stored:
            surfaces = {name: np.asarray(stored[name]) for name in stored.files}
    except (OSError, ValueError) as error:
        raise RuntimeError(f"seed {seed} surfaces.npz is unreadable: {error}") from error
    _validate_surfaces(seed, records, diagnostics, surfaces, horizon=horizon)
    return seed_dir


def _validate_surfaces(
    seed: int,
    records: pd.DataFrame,
    diagnostics: dict[str, Any],
    surfaces: dict[str, np.ndarray],
    *,
    horizon: int,
) -> None:
    required_shapes = {
        "base_grid": (BASE_GRID_SIZE,),
        "dense_grid": (DENSE_GRID_SIZE,),
        "base_indices_in_dense": (BASE_GRID_SIZE,),
        "stage_profile": (horizon,),
        "base_point_estimates": (BASE_GRID_SIZE, horizon),
        "dense_point_estimates": (DENSE_GRID_SIZE, horizon),
        "base_lower_bounds": (BASE_GRID_SIZE, horizon),
        "dense_lower_bounds": (DENSE_GRID_SIZE, horizon),
        "base_estimated_widths": (BASE_GRID_SIZE,),
        "dense_estimated_widths": (DENSE_GRID_SIZE,),
        "base_selected_schedule": (horizon,),
        "dense_selected_schedule": (horizon,),
        "base_fresh_coverage": (horizon,),
        "dense_fresh_coverage": (horizon,),
        "base_fresh_width": (horizon,),
        "dense_fresh_width": (horizon,),
        "dense_effective_sample_sizes": (DENSE_GRID_SIZE, horizon),
    }
    for name, expected_shape in required_shapes.items():
        value = surfaces.get(name)
        if value is None or value.shape != expected_shape:
            raise RuntimeError(f"seed {seed} surface {name} has the wrong shape")
        if not np.isfinite(value).all():
            raise RuntimeError(f"seed {seed} surface {name} is non-finite")

    expected_indices = np.arange(0, DENSE_GRID_SIZE, SUBDIVISIONS)
    indices = surfaces["base_indices_in_dense"]
    if not np.array_equal(indices, expected_indices):
        raise RuntimeError(f"seed {seed} base indices are not the frozen nested knots")
    if not np.array_equal(surfaces["base_grid"], surfaces["dense_grid"][indices]):
        raise RuntimeError(f"seed {seed} dense grid does not preserve base knots exactly")
    if not np.allclose(
        surfaces["base_point_estimates"],
        surfaces["dense_point_estimates"][indices],
        atol=NUMERIC_PARITY_TOLERANCE,
        rtol=0.0,
    ):
        raise RuntimeError(f"seed {seed} base point estimates exceed parity tolerance")
    if not np.allclose(
        surfaces["base_lower_bounds"],
        surfaces["dense_lower_bounds"][indices],
        atol=NUMERIC_PARITY_TOLERANCE,
        rtol=0.0,
    ):
        raise RuntimeError(f"seed {seed} base LCBs exceed parity tolerance")
    surface_names = {
        BASE_METHOD: ("base_fresh_coverage", "base_fresh_width"),
        DENSE_METHOD: ("dense_fresh_coverage", "dense_fresh_width"),
    }
    for _, row in records.iterrows():
        coverage_name, width_name = surface_names[str(row["method"])]
        coverage = surfaces[coverage_name]
        width = surfaces[width_name]
        if not np.allclose(
            np.asarray(json.loads(row["fresh_per_time_coverage"]), dtype=np.float64),
            coverage,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(f"seed {seed} coverage record disagrees with its surface")
        if not np.allclose(
            np.asarray(json.loads(row["fresh_per_time_normalized_width"]), dtype=np.float64),
            width,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(f"seed {seed} width record disagrees with its surface")
        if not np.isclose(
            float(row["fresh_average_normalized_width"]),
            float(width.mean()),
            atol=FLOAT32_AGGREGATION_TOLERANCE,
            rtol=0.0,
        ):
            raise RuntimeError(f"seed {seed} average width record disagrees")


def write_summary(
    output_dir: Path,
    seeds: tuple[int, ...],
    *,
    horizon: int,
    reference_decomposition: Path | None = None,
) -> dict[str, Any]:
    """Atomically publish paired width, pooled marginal coverage, and gate results."""

    coverage = {method: [] for method in METHODS}
    widths = {method: [] for method in METHODS}
    availability = {method: [] for method in METHODS}
    point_parity_errors: list[float] = []
    lcb_parity_errors: list[float] = []
    weight_parity_errors: list[float] = []
    width_parity_errors: list[float] = []
    selection_replays: list[bool] = []
    certificate_labels: set[str] = set()
    certificate_formality: set[bool] = set()
    dense_wider_seeds: list[int] = []
    minimum_dense_ess_values: list[float] = []
    maximum_dense_cap_hit_rates: list[float] = []
    stopped_blockers: list[dict[str, int | None]] = []

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed:05d}"
        records = pd.read_csv(seed_dir / "records.csv").set_index("method")
        metadata = _read_json(seed_dir / "metadata.json")
        diagnostics = metadata["diagnostics"]
        point_parity_errors.append(float(diagnostics["maximum_base_point_parity_error"]))
        lcb_parity_errors.append(float(diagnostics["maximum_base_lcb_parity_error"]))
        weight_parity_errors.append(
            float(diagnostics["maximum_base_weight_parity_error"])
        )
        width_parity_errors.append(
            float(diagnostics["maximum_base_width_parity_error"])
        )
        selection_replays.append(
            bool(diagnostics["base_selection_matches_dense_base_knots"])
        )
        minimum_dense_ess_values.append(float(diagnostics["minimum_dense_ess"]))
        maximum_dense_cap_hit_rates.append(
            float(diagnostics["maximum_dense_cap_hit_rate"])
        )
        base_stopped = diagnostics["base_stopped_index"]
        dense_stopped = diagnostics["dense_stopped_index"]
        stopped_blockers.append(
            {
                "seed": seed,
                "base_stopped_index": base_stopped,
                "base_stopped_index_on_dense_grid": (
                    None if base_stopped is None else SUBDIVISIONS * base_stopped
                ),
                "dense_stopped_index": dense_stopped,
                "dense_minus_base_equivalent": (
                    None
                    if base_stopped is None or dense_stopped is None
                    else dense_stopped - SUBDIVISIONS * base_stopped
                ),
            }
        )
        with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as surfaces:
            for method, prefix in ((BASE_METHOD, "base"), (DENSE_METHOD, "dense")):
                coverage[method].append(
                    np.asarray(surfaces[f"{prefix}_fresh_coverage"], dtype=np.float64)
                )
                widths[method].append(
                    float(
                        np.asarray(
                            surfaces[f"{prefix}_fresh_width"], dtype=np.float64
                        ).mean()
                    )
                )
                availability[method].append(
                    bool(records.loc[method, "selection_available"])
                )
                certificate_labels.add(str(records.loc[method, "certificate_type"]))
                certificate_formality.add(
                    bool(records.loc[method, "certificate_formal"])
                )
        if widths[DENSE_METHOD][-1] > widths[BASE_METHOD][-1]:
            dense_wider_seeds.append(seed)

    coverage_arrays = {
        method: np.stack(values, axis=0) for method, values in coverage.items()
    }
    width_arrays = {
        method: np.asarray(values, dtype=np.float64) for method, values in widths.items()
    }
    if any(values.shape != (len(seeds), horizon) for values in coverage_arrays.values()):
        raise RuntimeError("summary coverage matrices have the wrong shape")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(
        0,
        len(seeds),
        size=(BOOTSTRAP_RESAMPLES, len(seeds)),
    )
    method_summaries = {
        method: _summarize_method(
            coverage_arrays[method],
            width_arrays[method],
            bootstrap_indices,
        )
        for method in METHODS
    }
    paired_width = _paired_width_summary(
        width_arrays[DENSE_METHOD],
        width_arrays[BASE_METHOD],
        bootstrap_indices,
    )
    base_bootstrap_wsc = (
        coverage_arrays[BASE_METHOD][bootstrap_indices].mean(axis=1).min(axis=1)
    )
    dense_bootstrap_wsc = (
        coverage_arrays[DENSE_METHOD][bootstrap_indices].mean(axis=1).min(axis=1)
    )
    wsc_difference = (
        method_summaries[DENSE_METHOD]["pooled_marginal_wsc"]
        - method_summaries[BASE_METHOD]["pooled_marginal_wsc"]
    )
    wsc_difference_bootstrap = dense_bootstrap_wsc - base_bootstrap_wsc
    wsc_difference_ci = _percentile_interval(wsc_difference_bootstrap)

    paired_available = sum(
        base and dense
        for base, dense in zip(
            availability[BASE_METHOD],
            availability[DENSE_METHOD],
            strict=True,
        )
    )
    point_parity_max = max(point_parity_errors)
    lcb_parity_max = max(lcb_parity_errors)
    weight_parity_max = max(weight_parity_errors)
    width_parity_max = max(width_parity_errors)
    minimum_dense_ess = min(minimum_dense_ess_values)
    maximum_dense_cap_hit_rate = max(maximum_dense_cap_hit_rates)
    identity_preserved = len(certificate_labels) == 1 and len(certificate_formality) == 1
    gate_checks = {
        "twenty_of_twenty_paired_selections_available": (
            len(seeds) == EXPECTED_DEVELOPMENT_SEEDS
            and paired_available == EXPECTED_DEVELOPMENT_SEEDS
        ),
        "base_point_and_lcb_parity_within_frozen_tolerance": (
            point_parity_max <= NUMERIC_PARITY_TOLERANCE
            and lcb_parity_max <= NUMERIC_PARITY_TOLERANCE
        ),
        "independent_base_weight_width_and_selection_replay_within_tolerance": (
            weight_parity_max <= WEIGHT_PARITY_TOLERANCE
            and width_parity_max <= NUMERIC_PARITY_TOLERANCE
            and all(selection_replays)
        ),
        "target_delta_certificate_identity_preserved": identity_preserved,
        "dense_pooled_marginal_wsc_at_least_0_90": (
            method_summaries[DENSE_METHOD]["pooled_marginal_wsc"] >= TARGET
        ),
        "maximum_dense_cap_hit_rate_at_most_0_01": (
            maximum_dense_cap_hit_rate <= MAXIMUM_DENSE_CAP_HIT_RATE
        ),
        "minimum_dense_ess_at_least_25": (
            minimum_dense_ess >= MINIMUM_DENSE_EFFECTIVE_SAMPLE_SIZE
        ),
        "width_ratio_at_most_0_995": (
            paired_width["geometric_mean_ratio"] <= MAXIMUM_WIDTH_RATIO
        ),
        "width_ratio_one_sided_95_ucb_below_1": (
            paired_width["one_sided_95_upper"] < MAXIMUM_WIDTH_RATIO_UCB
        ),
        "pooled_marginal_wsc_loss_paired_95_lcb_at_most_0_002": (
            wsc_difference_ci[0] >= -MAXIMUM_WSC_LOSS
        ),
    }
    summary = {
        "protocol": PROTOCOL,
        "role": "exploratory historical PAC-grid diagnostic; not paper SC-PCP",
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "target_coverage": TARGET,
        "delta": DELTA,
        "confidence_level": 1.0 - DELTA,
        "evaluation_rollouts_per_seed": EVALUATION_ROLLOUTS,
        "coverage_estimands": {
            "pooled_marginal_wsc": "min_t mean_seed coverage(seed,t)",
            "mean_coverage": "mean_t mean_seed coverage(seed,t)",
        },
        "certificate": {
            "method": "patient-cluster practical bootstrap LCB",
            "resamples_per_grid": CERTIFICATE_BOOTSTRAP_RESAMPLES,
            "formal": False,
        },
        "summary_seed_bootstrap": {
            "method": "paired-seed nonparametric percentile bootstrap",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "two_sided_level": 0.95,
            "width_ratio_ucb_level": 0.95,
        },
        "methods": method_summaries,
        "paired_dense_over_base_width": paired_width,
        "paired_pooled_marginal_wsc_difference_dense_minus_base": {
            "estimate": float(wsc_difference),
            "ci95": list(wsc_difference_ci),
        },
        "availability": {
            "paired_available_count": int(paired_available),
            "paired_available_rate": float(paired_available / len(seeds)),
            "by_method": {
                method: {
                    "count": int(sum(availability[method])),
                    "rate": float(np.mean(availability[method])),
                }
                for method in METHODS
            },
        },
        "parity": {
            "maximum_base_weight_parity_error": weight_parity_max,
            "maximum_base_point_parity_error": point_parity_max,
            "maximum_base_lcb_parity_error": lcb_parity_max,
            "maximum_base_width_parity_error": width_parity_max,
            "base_selection_replay_all_seeds": all(selection_replays),
            "certificate_labels": sorted(certificate_labels),
            "certificate_formality": sorted(certificate_formality),
            "numeric_parity_tolerance": NUMERIC_PARITY_TOLERANCE,
            "weight_parity_tolerance": WEIGHT_PARITY_TOLERANCE,
        },
        "dense_diagnostics": {
            "minimum_effective_sample_size": minimum_dense_ess,
            "maximum_cap_hit_rate": maximum_dense_cap_hit_rate,
            "stopped_blockers_by_seed": stopped_blockers,
        },
        "dense_wider_seeds": dense_wider_seeds,
        "dense_wider_seed_count": len(dense_wider_seeds),
        "acceptance_gate": {
            "frozen_definition": _acceptance_gate_definition(),
            "checks": gate_checks,
            "all_passed": all(gate_checks.values()),
        },
    }
    if reference_decomposition is not None:
        reference_replay = _compare_reference_decomposition(
            output_dir,
            reference_decomposition,
            seeds,
            horizon=horizon,
        )
        summary["reference_decomposition_replay"] = reference_replay
        gate_checks[
            "base_grid_replays_frozen_E_schedule_and_index_within_tolerance"
        ] = bool(reference_replay["replay_within_tolerance"])
        summary["acceptance_gate"]["all_passed"] = all(gate_checks.values())

    csv_rows = _summary_csv_rows(summary)
    _atomic_write_text(
        output_dir / "summary.csv",
        pd.DataFrame(csv_rows).to_csv(index=False),
    )
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    return summary


def _summarize_method(
    coverage: np.ndarray,
    widths: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    pooled_per_stage = coverage.mean(axis=0)
    bootstrap_coverage = coverage[bootstrap_indices].mean(axis=1)
    bootstrap_wsc = bootstrap_coverage.min(axis=1)
    bootstrap_mean_coverage = bootstrap_coverage.mean(axis=1)
    log_width = np.log(widths)
    bootstrap_geometric_width = np.exp(log_width[bootstrap_indices].mean(axis=1))
    return {
        "pooled_marginal_wsc": float(pooled_per_stage.min()),
        "pooled_marginal_wsc_ci95": list(_percentile_interval(bootstrap_wsc)),
        "worst_stage_zero_based": int(pooled_per_stage.argmin()),
        "per_stage_marginal_coverage": pooled_per_stage.tolist(),
        "mean_coverage": float(pooled_per_stage.mean()),
        "mean_coverage_ci95": list(_percentile_interval(bootstrap_mean_coverage)),
        "geometric_mean_average_normalized_width": float(np.exp(log_width.mean())),
        "geometric_mean_average_normalized_width_ci95": list(
            _percentile_interval(bootstrap_geometric_width)
        ),
        "fresh_target_met_seed_count": int((coverage.min(axis=1) >= TARGET).sum()),
    }


def _paired_width_summary(
    dense_width: np.ndarray,
    base_width: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    paired_log_ratio = np.log(dense_width / base_width)
    bootstrap = np.exp(paired_log_ratio[bootstrap_indices].mean(axis=1))
    return {
        "numerator": DENSE_METHOD,
        "denominator": BASE_METHOD,
        "geometric_mean_ratio": float(np.exp(paired_log_ratio.mean())),
        "ci95": list(_percentile_interval(bootstrap)),
        "one_sided_95_upper": float(np.quantile(bootstrap, 0.95)),
    }


def _reference_decomposition_fingerprint(
    root: Path,
    seeds: tuple[int, ...],
    *,
    horizon: int,
) -> str:
    """Bind resume provenance to exactly the old E schedules and indices used."""

    if not (root / "COMPLETE").is_file():
        raise RuntimeError(f"reference decomposition is not complete: {root}")
    digest = hashlib.sha256()
    digest.update(PROTOCOL.encode("utf-8"))
    for seed in seeds:
        schedule, index = _load_reference_E(root, seed, horizon=horizon)
        digest.update(seed.to_bytes(8, "big", signed=False))
        digest.update(index.to_bytes(8, "big", signed=False))
        digest.update(schedule.dtype.str.encode("ascii"))
        digest.update(schedule.tobytes(order="C"))
    return digest.hexdigest()


def _compare_reference_decomposition(
    output_dir: Path,
    reference_root: Path,
    seeds: tuple[int, ...],
    *,
    horizon: int,
) -> dict[str, Any]:
    schedule_errors: dict[str, float] = {}
    index_mismatches: list[dict[str, int]] = []
    for seed in seeds:
        reference_schedule, reference_index = _load_reference_E(
            reference_root,
            seed,
            horizon=horizon,
        )
        seed_dir = output_dir / f"seed_{seed:05d}"
        with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as surfaces:
            base_schedule = np.asarray(
                surfaces["base_selected_schedule"], dtype=np.float64
            )
        diagnostics = _read_json(seed_dir / "metadata.json")["diagnostics"]
        base_index = diagnostics.get("base_selected_index")
        if type(base_index) is not int:
            raise RuntimeError(f"seed {seed} base selected index is not an integer")
        error = float(
            np.max(np.abs(base_schedule - reference_schedule.astype(np.float64)))
        )
        schedule_errors[str(seed)] = error
        if base_index != reference_index:
            index_mismatches.append(
                {
                    "seed": seed,
                    "base_selected_index": base_index,
                    "reference_E_index": reference_index,
                }
            )
    maximum_schedule_error = max(schedule_errors.values())
    return {
        "reference_root": str(reference_root),
        "maximum_absolute_schedule_error": maximum_schedule_error,
        "per_seed_maximum_absolute_schedule_error": schedule_errors,
        "index_mismatches": index_mismatches,
        "index_mismatch_count": len(index_mismatches),
        "schedule_tolerance": REFERENCE_SCHEDULE_TOLERANCE,
        "schedule_bitwise_exact": maximum_schedule_error == 0.0,
        "index_exact": not index_mismatches,
        "replay_within_tolerance": (
            maximum_schedule_error <= REFERENCE_SCHEDULE_TOLERANCE
            and not index_mismatches
        ),
    }


def _load_reference_E(
    root: Path,
    seed: int,
    *,
    horizon: int,
) -> tuple[np.ndarray, int]:
    seed_dir = root / f"seed_{seed:05d}"
    for name in ("COMPLETE", "surfaces.npz", "metadata.json"):
        if not (seed_dir / name).is_file():
            raise RuntimeError(f"reference seed {seed} is missing {name}")
    try:
        with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as surfaces:
            schedule = np.asarray(surfaces["E_schedule"]).copy()
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"reference seed {seed} E_schedule is unreadable") from error
    if schedule.shape != (horizon,) or not np.isfinite(schedule).all():
        raise RuntimeError(f"reference seed {seed} E_schedule is invalid")
    metadata = _read_json(seed_dir / "metadata.json")
    try:
        index = metadata["diagnostics"]["indices"]["e"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"reference seed {seed} E index is missing") from error
    if type(index) is not int or not 0 <= index < BASE_GRID_SIZE:
        raise RuntimeError(f"reference seed {seed} E index is invalid")
    return schedule, index


def _acceptance_gate_definition() -> dict[str, Any]:
    return {
        "locked_before_full_20_seed_run": True,
        "required_paired_available_count": EXPECTED_DEVELOPMENT_SEEDS,
        "maximum_base_point_parity_error": NUMERIC_PARITY_TOLERANCE,
        "maximum_base_lcb_parity_error": NUMERIC_PARITY_TOLERANCE,
        "maximum_base_weight_parity_error": WEIGHT_PARITY_TOLERANCE,
        "maximum_base_width_parity_error": NUMERIC_PARITY_TOLERANCE,
        "required_base_selection_replay": True,
        "required_target_coverage": TARGET,
        "required_delta": DELTA,
        "required_certificate_patient_bootstrap_resamples": (
            CERTIFICATE_BOOTSTRAP_RESAMPLES
        ),
        "certificate_label_and_formality_must_match": True,
        "maximum_dense_over_base_geometric_width_ratio": MAXIMUM_WIDTH_RATIO,
        "maximum_dense_over_base_width_ratio_one_sided_95_ucb": (
            MAXIMUM_WIDTH_RATIO_UCB
        ),
        "maximum_allowed_pooled_marginal_wsc_loss_paired_95_lcb": (
            MAXIMUM_WSC_LOSS
        ),
        "minimum_dense_pooled_marginal_wsc": TARGET,
        "maximum_dense_cap_hit_rate": MAXIMUM_DENSE_CAP_HIT_RATE,
        "minimum_dense_effective_sample_size": MINIMUM_DENSE_EFFECTIVE_SAMPLE_SIZE,
        "motivation": "K=401 costs approximately 4x K=101 compute and memory",
        "dense_wider_seeds_must_be_reported": True,
        "reference_E_index_must_replay_exactly_when_provided": True,
        "reference_E_schedule_tolerance_when_provided": REFERENCE_SCHEDULE_TOLERANCE,
    }


def _summary_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        values = summary["methods"][method]
        for metric, ci_name in (
            ("pooled_marginal_wsc", "pooled_marginal_wsc_ci95"),
            ("mean_coverage", "mean_coverage_ci95"),
            (
                "geometric_mean_average_normalized_width",
                "geometric_mean_average_normalized_width_ci95",
            ),
        ):
            interval = values[ci_name]
            rows.append(
                {
                    "category": "method",
                    "method": method,
                    "metric": metric,
                    "estimate": values[metric],
                    "ci95_lower": interval[0],
                    "ci95_upper": interval[1],
                }
            )
    ratio = summary["paired_dense_over_base_width"]
    rows.append(
        {
            "category": "paired",
            "method": f"{DENSE_METHOD} / {BASE_METHOD}",
            "metric": "geometric_mean_width_ratio",
            "estimate": ratio["geometric_mean_ratio"],
            "ci95_lower": ratio["ci95"][0],
            "ci95_upper": ratio["ci95"][1],
        }
    )
    difference = summary["paired_pooled_marginal_wsc_difference_dense_minus_base"]
    rows.append(
        {
            "category": "paired",
            "method": f"{DENSE_METHOD} - {BASE_METHOD}",
            "metric": "pooled_marginal_wsc_difference",
            "estimate": difference["estimate"],
            "ci95_lower": difference["ci95"][0],
            "ci95_upper": difference["ci95"][1],
        }
    )
    return rows


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def _validate_resume_provenance(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    execution: dict[str, Any],
    source_hash: str,
    config_hash: str,
) -> None:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"resume output does not exist: {output_dir}")
    metadata = _read_json(output_dir / "study_metadata.json")
    try:
        stored_config = yaml.safe_load((output_dir / "config.yaml").read_text())
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"resume config is unreadable: {error}") from error
    if not isinstance(stored_config, dict):
        raise RuntimeError("resume stored config is not a mapping")
    if canonical_config_sha256(stored_config) != config_hash:
        raise RuntimeError("resume stored config differs from the requested config")
    if metadata.get("seeds") != list(config.seeds):
        raise RuntimeError("resume requested seeds differ from study metadata")
    if metadata.get("devices") != list(config.devices):
        raise RuntimeError("resume requested devices differ from study metadata")
    if metadata.get("source_tree_sha256") != source_hash:
        raise RuntimeError("resume source tree differs from the active source")
    if metadata.get("execution") != execution:
        raise RuntimeError("resume execution protocol differs from the requested run")


def _validated_existing_seeds(
    output_dir: Path,
    requested_seeds: tuple[int, ...],
    *,
    expected_source_hash: str,
    expected_config_hash: str,
    horizon: int,
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
            horizon=horizon,
            expected_source_hash=expected_source_hash,
            expected_config_hash=expected_config_hash,
        )
        completed.add(seed)
    return completed


def _build_seed_jobs(
    seeds: tuple[int, ...],
    devices: tuple[str, ...],
    workers_per_device: int,
) -> tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]]:
    # Interleave devices so a final partial wave remains balanced across GPUs.
    worker_devices = tuple(
        device for _ in range(workers_per_device) for device in devices
    )
    jobs = tuple(
        (index % len(worker_devices), seed, worker_devices[index % len(worker_devices)])
        for index, seed in enumerate(seeds)
    )
    return worker_devices, jobs


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


def canonical_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"JSON is unreadable at {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON must contain an object at {path}")
    return value


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
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
