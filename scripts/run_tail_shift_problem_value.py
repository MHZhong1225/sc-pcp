"""Audit whether tail shift creates a marginal-coverage problem for Standard CP.

The audit is intentionally separate from the main experiment.  Standard CP is
calibrated on the same D_COT + D_cert budget used by the paper experiment.  Its
frozen stagewise radii and the Phase 0 A/C schedules are then evaluated on the
same 50k common-random-number holdout bundle.  Phase 0 A/C results must replay
bitwise before a seed is accepted.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_phase0_oracle import (  # noqa: E402
    _build_seed_jobs,
    _execute_jobs,
    canonical_config_sha256,
    parse_seeds,
    validate_seed_artifact as validate_phase0_seed,
)
from scpcp.artifacts import (  # noqa: E402
    experiment_tree_sha256,
    mark_study_complete,
    mark_study_failed,
    source_tree_sha256,
    write_seed_result,
    write_study_metadata,
)
from scpcp.baselines import standard_cp_stagewise_radii  # noqa: E402
from scpcp.config import ExperimentConfig  # noqa: E402
from scpcp.data import concatenate_trajectories  # noqa: E402
from scpcp.device import resolve_devices  # noqa: E402
from scpcp.experiment import (  # noqa: E402
    SeedResult,
    _prepare_task,
    _training_outcome_sd,
)
from scpcp.outcome_model import fit_outcome_model  # noqa: E402
from scpcp.phase0_oracle import (  # noqa: E402
    FrozenOracleEvaluation,
    evaluate_frozen_schedules_crn,
)
from scpcp.policy import BehaviorAnchoredPolicy  # noqa: E402
from scpcp.scores import fit_conformal_region, score_batch  # noqa: E402
from scpcp.simulator import make_synthetic_noise_bundle  # noqa: E402


PROTOCOL = "tail_shift_problem_value_common_phase0_holdout_v1"
SCENARIO = "tail_shift"
STANDARD = "Standard CP"
A_ORACLE = "Greedy Sequential Oracle"
C_ORACLE = "Current Profiled Oracle"
METHODS = (STANDARD, A_ORACLE, C_ORACLE)
PAIRWISE_RATIOS = (
    (STANDARD, A_ORACLE),
    (STANDARD, C_ORACLE),
    (C_ORACLE, A_ORACLE),
)
SEED_DIRECTORY = re.compile(r"seed_(\d{5})")
REQUIRED_SEED_FILES = ("COMPLETE", "records.csv", "surfaces.npz", "metadata.json")
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 8_202_681


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fair Standard CP with frozen Phase 0 A/C schedules under "
            "tail shift on their common 50k holdout stream"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "per_step_synthetic.yaml",
    )
    parser.add_argument("--phase0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default=None, help="range such as 0:100")
    parser.add_argument("--devices", default=None, help="comma-separated CUDA devices")
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers_per_device < 1:
        raise ValueError("workers-per-device must be positive")

    reference_config = ExperimentConfig.from_yaml(args.config)
    if reference_config.data.dataset != "synthetic":
        raise ValueError("tail-shift problem-value audit requires synthetic data")
    devices = resolve_devices(args.devices or reference_config.devices)
    seeds = parse_seeds(args.seeds, reference_config.seeds)
    tail_config = replace(
        reference_config,
        synthetic=replace(reference_config.synthetic, scenario=SCENARIO),
    ).with_overrides(
        devices=devices,
        seeds=seeds,
        output_dir=args.output_dir.resolve(),
    )
    run_config(
        tail_config,
        reference_config=reference_config,
        phase0_dir=args.phase0_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        workers_per_device=args.workers_per_device,
        resume=args.resume,
    )
    print(args.output_dir.resolve())


def run_config(
    config: ExperimentConfig,
    *,
    reference_config: ExperimentConfig,
    phase0_dir: Path,
    output_dir: Path,
    workers_per_device: int,
    resume: bool,
) -> None:
    if config.data.dataset != "synthetic" or config.synthetic.scenario != SCENARIO:
        raise ValueError("audit config must be the tail_shift synthetic scenario")
    if config.output_dir != output_dir:
        raise ValueError("config output_dir must match output_dir")
    if config.samples.oracle_rollouts != 50_000:
        raise ValueError("the frozen audit requires the Phase 0 50k holdout size")

    phase0_fingerprint = validate_phase0_inputs(
        phase0_dir,
        reference_config=reference_config,
        seeds=config.seeds,
    )
    source_hash = source_tree_sha256()
    config_hash = canonical_config_sha256(config.to_dict())
    execution = {
        "protocol": PROTOCOL,
        "experiment_tree_sha256": experiment_tree_sha256(),
        "config_sha256": config_hash,
        "phase0_dir": str(phase0_dir),
        "phase0_fingerprint": phase0_fingerprint,
        "workers_per_device": workers_per_device,
        "standard_calibration_roles": ["D_COT", "D_cert"],
        "evaluation_rollouts": config.samples.oracle_rollouts,
        "evaluation_stream": "frozen Phase0 tail_shift evaluation_seed",
    }

    if resume:
        completed = _validate_resume(
            output_dir,
            config,
            execution=execution,
            source_hash=source_hash,
            config_hash=config_hash,
        )
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh audit output already exists: {output_dir}")
        write_study_metadata(output_dir, config, execution=execution)
        completed = set()

    pending = tuple(seed for seed in config.seeds if seed not in completed)
    try:
        _run_pending_seeds(
            config,
            phase0_dir=phase0_dir,
            output_dir=output_dir,
            seeds=pending,
            workers_per_device=workers_per_device,
        )
        for seed in config.seeds:
            validate_output_seed(
                output_dir / f"seed_{seed:05d}",
                seed,
                horizon=config.horizon,
                rollouts=config.samples.oracle_rollouts,
                expected_source_hash=source_hash,
                expected_config_hash=config_hash,
            )
        if validate_phase0_inputs(
            phase0_dir,
            reference_config=reference_config,
            seeds=config.seeds,
        ) != phase0_fingerprint:
            raise RuntimeError("Phase 0 inputs changed while the audit was running")
        write_summary(output_dir, config.seeds, horizon=config.horizon)
        mark_study_complete(output_dir, config.seeds)
    except BaseException as error:
        mark_study_failed(output_dir, config.seeds, error)
        raise


def validate_phase0_inputs(
    root: Path,
    *,
    reference_config: ExperimentConfig,
    seeds: tuple[int, ...],
) -> str:
    for name in (
        "COMPLETE",
        "config.yaml",
        "study_metadata.json",
        "study_status.json",
    ):
        if not (root / name).is_file():
            raise RuntimeError(f"Phase 0 input is missing {name}: {root}")

    status = _read_json(root / "study_status.json")
    expected = status.get("expected_seeds")
    if (
        status.get("status") != "complete"
        or not isinstance(expected, list)
        or status.get("completed_seeds") != expected
        or status.get("missing_seeds") not in ([], None)
        or status.get("error") is not None
    ):
        raise RuntimeError("Phase 0 study is not exactly complete")
    if not set(seeds).issubset(set(expected)):
        raise RuntimeError("requested audit seeds are not all present in Phase 0")

    phase0_config = ExperimentConfig.from_yaml(root / "config.yaml")
    if _normalized_reference_config(phase0_config) != _normalized_reference_config(
        reference_config
    ):
        raise RuntimeError("Phase 0 config differs from per_step_synthetic.yaml")

    metadata = _read_json(root / "study_metadata.json")
    root_source_hash = metadata.get("source_tree_sha256")
    if not isinstance(root_source_hash, str) or not root_source_hash:
        raise RuntimeError("Phase 0 source hash is missing")
    raw_config = yaml.safe_load((root / "config.yaml").read_text())
    if not isinstance(raw_config, dict):
        raise RuntimeError("Phase 0 config must contain a mapping")
    root_config_hash = canonical_config_sha256(raw_config)

    digest = hashlib.sha256()
    for name in ("config.yaml", "study_metadata.json", "study_status.json", "COMPLETE"):
        _update_digest(digest, root / name, root)
    for seed in seeds:
        seed_dir = root / f"seed_{seed:05d}"
        validate_phase0_seed(
            seed_dir,
            seed,
            expected_source_hash=root_source_hash,
            expected_config_hash=root_config_hash,
        )
        for name in REQUIRED_SEED_FILES:
            _update_digest(digest, seed_dir / name, root)
    return digest.hexdigest()


def _normalized_reference_config(config: ExperimentConfig) -> dict[str, Any]:
    values = config.to_dict()
    values["output_dir"] = "<frozen-input-output>"
    values["paper"]["save_mechanism_diagonal"] = False
    values["synthetic"]["scenario"] = "<paired-phase0-scenario>"
    return values


def _run_pending_seeds(
    config: ExperimentConfig,
    *,
    phase0_dir: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
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
        (
            worker_index,
            (config, seed, device, phase0_dir, output_dir),
        )
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
    phase0_dir: Path,
    output_dir: Path,
) -> str:
    def run_and_publish() -> str:
        result = run_seed(
            config,
            seed=seed,
            device=device,
            phase0_seed_dir=phase0_dir / f"seed_{seed:05d}",
        )
        seed_dir = write_seed_result(result, output_dir, config)
        validate_output_seed(
            seed_dir,
            seed,
            horizon=config.horizon,
            rollouts=config.samples.oracle_rollouts,
        )
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


def run_seed(
    config: ExperimentConfig,
    *,
    seed: int,
    device: str,
    phase0_seed_dir: Path,
) -> SeedResult:
    phase0 = load_tail_shift_phase0(
        phase0_seed_dir,
        seed=seed,
        horizon=config.horizon,
        target=1.0 - config.certification.alpha,
        rollouts=config.samples.oracle_rollouts,
    )

    # This reset and replay order matches Phase 0 exactly through model fitting.
    torch.manual_seed(seed)
    task = _prepare_task(config, seed=seed, device=device)
    if task.name != "synthetic" or task.environment is None or task.logging_policy is None:
        raise RuntimeError("tail-shift audit requires the known synthetic task")
    outcome_model = fit_outcome_model(
        task.splits.predictor,
        n_actions=task.n_actions,
        config=config.model,
        device=device,
        seed=seed + 1,
        static_indices=task.static_indices,
    )
    region = fit_conformal_region(outcome_model)
    policy = BehaviorAnchoredPolicy(
        outcome_model=outcome_model,
        reference_policy=task.logging_policy,
        config=task.policy_config,
        region=region,
        tilt=config.policy.tilt,
    )
    outcome_sd = _training_outcome_sd(task.splits.predictor)

    calibration = concatenate_trajectories(
        task.splits.cot,
        task.splits.certification,
    )
    calibration_scores = score_batch(
        region,
        calibration.current_states(),
        calibration.actions,
        calibration.outcomes,
    )
    standard_radii = standard_cp_stagewise_radii(
        calibration_scores,
        config.certification.alpha,
    )
    schedules = {
        STANDARD: standard_radii,
        A_ORACLE: torch.from_numpy(phase0["a_schedule"].copy()),
        C_ORACLE: torch.from_numpy(phase0["c_schedule"].copy()),
    }

    noise = make_synthetic_noise_bundle(
        n=config.samples.oracle_rollouts,
        horizon=config.horizon,
        seed=phase0["evaluation_seed"],
        device=device,
    )
    evaluations = evaluate_frozen_schedules_crn(
        task.environment,
        policy,
        outcome_model,
        schedules=schedules,
        noise=noise,
        outcome_sd=outcome_sd,
        forbidden_noise_seeds={phase0["tuning_seed"]},
    )
    parity = verify_phase0_evaluation_parity(
        evaluations,
        expected_rows=phase0["rows"],
    )

    records = []
    surfaces: dict[str, torch.Tensor] = {}
    for method in METHODS:
        evaluated = evaluations[method]
        schedule = schedules[method].detach().cpu()
        prefix = _method_prefix(method)
        records.append(
            {
                "scenario": SCENARIO,
                "method": method,
                "seed": seed,
                "target_coverage": 1.0 - config.certification.alpha,
                "q_by_time": _json_vector(schedule),
                "coverage_by_time": _json_vector(evaluated.coverage),
                "worst_coverage": float(evaluated.coverage.min().item()),
                "mean_coverage": float(evaluated.coverage.mean().item()),
                "stage_width_by_time": _json_vector(evaluated.normalized_width),
                "mean_normalized_width": evaluated.patient_normalized_width,
                "evaluation_seed": phase0["evaluation_seed"],
                "evaluation_rollouts": evaluated.n_rollouts,
                "calibration_trajectories": calibration.n if method == STANDARD else 0,
            }
        )
        surfaces[f"{prefix}_schedule"] = schedule
        surfaces[f"{prefix}_coverage"] = evaluated.coverage.detach().cpu()
        surfaces[f"{prefix}_wilson_lcb"] = evaluated.wilson_lower_bound.detach().cpu()
        surfaces[f"{prefix}_stage_width"] = evaluated.normalized_width.detach().cpu()

    diagnostics = {
        "protocol": PROTOCOL,
        "scenario": SCENARIO,
        "input_phase0_seed_dir": str(phase0_seed_dir),
        "phase0_parity_verified": True,
        "phase0_parity": parity,
        "evaluation_seed": phase0["evaluation_seed"],
        "tuning_seed": phase0["tuning_seed"],
        "evaluation_rollouts": config.samples.oracle_rollouts,
        "standard_calibration": {
            "roles": ["D_COT", "D_cert"],
            "D_COT_trajectories": task.splits.cot.n,
            "D_cert_trajectories": task.splits.certification.n,
            "total_trajectories": calibration.n,
        },
    }
    return SeedResult(seed, device, records, surfaces, diagnostics)


def load_tail_shift_phase0(
    seed_dir: Path,
    *,
    seed: int,
    horizon: int,
    target: float,
    rollouts: int,
) -> dict[str, Any]:
    records = pd.read_csv(seed_dir / "records.csv")
    rows = records.loc[records["scenario"].eq(SCENARIO)].copy()
    if len(rows) != 2 or set(rows["method"]) != {A_ORACLE, C_ORACLE}:
        raise RuntimeError(f"Phase 0 seed {seed} has an invalid tail_shift contract")
    if not rows["selection_available"].astype(bool).all():
        raise RuntimeError(f"Phase 0 seed {seed} lacks an A or C schedule")
    if not rows["seed"].eq(seed).all() or not rows["n_rollouts"].eq(rollouts).all():
        raise RuntimeError(f"Phase 0 seed {seed} has wrong seed or rollout metadata")
    evaluation_seeds = {int(value) for value in rows["evaluation_seed"]}
    tuning_seeds = {int(value) for value in rows["tuning_seed"]}
    if len(evaluation_seeds) != 1 or len(tuning_seeds) != 1:
        raise RuntimeError(f"Phase 0 seed {seed} A/C streams disagree")
    evaluation_seed = next(iter(evaluation_seeds))
    tuning_seed = next(iter(tuning_seeds))
    if evaluation_seed == tuning_seed:
        raise RuntimeError(f"Phase 0 seed {seed} tuning/evaluation streams collide")

    with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as archive:
        surfaces = {name: np.array(archive[name], copy=True) for name in archive.files}
    required = {
        "tail_shift_greedy_stage_grids",
        "tail_shift_greedy_selected_schedule",
        "tail_shift_profiled_candidate_schedules",
        "tail_shift_profiled_candidate_coverage",
        "tail_shift_profiled_candidate_normalized_width",
        "tail_shift_profiled_selected_schedule",
    }
    missing = sorted(required - set(surfaces))
    if missing:
        raise RuntimeError(f"Phase 0 seed {seed} surfaces are missing {missing}")

    a_schedule = _positive_vector(
        surfaces["tail_shift_greedy_selected_schedule"],
        horizon,
        label="A schedule",
    )
    c_schedule = _positive_vector(
        surfaces["tail_shift_profiled_selected_schedule"],
        horizon,
        label="C schedule",
    )
    a_row = rows.loc[rows["method"].eq(A_ORACLE)].iloc[0]
    c_row = rows.loc[rows["method"].eq(C_ORACLE)].iloc[0]
    _require_json_vector_equal(a_schedule, a_row["q_by_time"], label="A schedule")
    _require_json_vector_equal(c_schedule, c_row["q_by_time"], label="C schedule")

    stage_grids = np.asarray(surfaces["tail_shift_greedy_stage_grids"])
    if stage_grids.ndim != 2 or stage_grids.shape[0] != horizon:
        raise RuntimeError(f"Phase 0 seed {seed} has invalid A stage grids")
    for stage in range(horizon):
        _unique_vector_index(
            stage_grids[stage],
            a_schedule[stage],
            label=f"A stage {stage}",
        )

    candidates = np.asarray(surfaces["tail_shift_profiled_candidate_schedules"])
    coverage = np.asarray(surfaces["tail_shift_profiled_candidate_coverage"])
    widths = np.asarray(
        surfaces["tail_shift_profiled_candidate_normalized_width"]
    )
    if (
        candidates.ndim != 2
        or candidates.shape[1] != horizon
        or coverage.shape != candidates.shape
        or widths.shape != candidates.shape
    ):
        raise RuntimeError(f"Phase 0 seed {seed} has invalid C candidate surfaces")
    c_index = _unique_row_index(candidates, c_schedule, label="C schedule")
    feasible = (coverage >= target).all(axis=1)
    if not bool(feasible.any()):
        raise RuntimeError(f"Phase 0 seed {seed} C surface has no feasible candidate")
    objective = np.where(feasible, widths.mean(axis=1), np.inf)
    if c_index != int(objective.argmin()):
        raise RuntimeError(f"Phase 0 seed {seed} C is not the minimum-width feasible row")
    _require_json_vector_equal(
        coverage[c_index],
        c_row["tuning_coverage"],
        label="C tuning coverage",
    )
    _require_json_vector_equal(
        widths[c_index],
        c_row["tuning_width"],
        label="C tuning width",
    )
    return {
        "a_schedule": a_schedule,
        "c_schedule": c_schedule,
        "evaluation_seed": evaluation_seed,
        "tuning_seed": tuning_seed,
        "rows": {method: rows.loc[rows["method"].eq(method)].iloc[0] for method in (A_ORACLE, C_ORACLE)},
    }


def verify_phase0_evaluation_parity(
    evaluations: dict[str, FrozenOracleEvaluation],
    *,
    expected_rows: dict[str, pd.Series],
) -> dict[str, float]:
    maximum_coverage_error = 0.0
    maximum_width_error = 0.0
    for method in (A_ORACLE, C_ORACLE):
        evaluated = evaluations[method]
        row = expected_rows[method]
        expected_coverage = torch.tensor(
            json.loads(row["final_coverage"]),
            dtype=evaluated.coverage.dtype,
        )
        expected_lcb = torch.tensor(
            json.loads(row["final_wilson_lcb"]),
            dtype=evaluated.wilson_lower_bound.dtype,
        )
        expected_width = torch.tensor(
            json.loads(row["final_stage_width"]),
            dtype=evaluated.normalized_width.dtype,
        )
        actual_coverage = evaluated.coverage.detach().cpu()
        actual_lcb = evaluated.wilson_lower_bound.detach().cpu()
        actual_width = evaluated.normalized_width.detach().cpu()
        if not torch.equal(actual_coverage, expected_coverage):
            raise RuntimeError(f"{method} coverage replay mismatch")
        if not torch.equal(actual_lcb, expected_lcb):
            raise RuntimeError(f"{method} Wilson LCB replay mismatch")
        if not torch.equal(actual_width, expected_width):
            raise RuntimeError(f"{method} width replay mismatch")
        if np.float32(evaluated.patient_normalized_width) != np.float32(
            row["patient_normalized_width"]
        ):
            raise RuntimeError(f"{method} patient width replay mismatch")
        if np.float32(evaluated.micro_normalized_width) != np.float32(
            row["micro_normalized_width"]
        ):
            raise RuntimeError(f"{method} micro width replay mismatch")
        maximum_coverage_error = max(
            maximum_coverage_error,
            float((actual_coverage - expected_coverage).abs().max().item()),
        )
        maximum_width_error = max(
            maximum_width_error,
            float((actual_width - expected_width).abs().max().item()),
        )
    return {
        "maximum_coverage_error": maximum_coverage_error,
        "maximum_stage_width_error": maximum_width_error,
    }


def validate_output_seed(
    seed_dir: Path,
    seed: int,
    *,
    horizon: int,
    rollouts: int,
    expected_source_hash: str | None = None,
    expected_config_hash: str | None = None,
) -> None:
    for name in REQUIRED_SEED_FILES:
        if not (seed_dir / name).is_file():
            raise RuntimeError(f"output seed {seed} is missing {name}")
    metadata = _read_json(seed_dir / "metadata.json")
    if metadata.get("seed") != seed:
        raise RuntimeError(f"output seed {seed} metadata has the wrong seed")
    if expected_source_hash is not None and metadata.get("source_tree_sha256") != expected_source_hash:
        raise RuntimeError(f"output seed {seed} source hash differs")
    if expected_config_hash is not None:
        stored = metadata.get("config")
        if not isinstance(stored, dict) or canonical_config_sha256(stored) != expected_config_hash:
            raise RuntimeError(f"output seed {seed} config differs")
    diagnostics = metadata.get("diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("phase0_parity_verified") is not True:
        raise RuntimeError(f"output seed {seed} lacks passing Phase 0 parity")

    records = pd.read_csv(seed_dir / "records.csv")
    if len(records) != len(METHODS) or tuple(records["method"]) != METHODS:
        raise RuntimeError(f"output seed {seed} must contain ordered Standard/A/C rows")
    if not records["seed"].eq(seed).all() or not records["scenario"].eq(SCENARIO).all():
        raise RuntimeError(f"output seed {seed} records have wrong seed/scenario")
    if not records["evaluation_rollouts"].eq(rollouts).all():
        raise RuntimeError(f"output seed {seed} has wrong rollout count")
    if records["evaluation_seed"].nunique() != 1:
        raise RuntimeError(f"output seed {seed} methods do not share one CRN stream")
    if int(records.loc[records["method"].eq(STANDARD), "calibration_trajectories"].iloc[0]) != 3_000:
        raise RuntimeError(f"output seed {seed} Standard CP did not use D_COT + D_cert")

    with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as surfaces:
        for row_index, method in enumerate(METHODS):
            prefix = _method_prefix(method)
            for suffix in ("schedule", "coverage", "wilson_lcb", "stage_width"):
                key = f"{prefix}_{suffix}"
                if key not in surfaces.files or surfaces[key].shape != (horizon,):
                    raise RuntimeError(f"output seed {seed} has invalid surface {key}")
                if not np.isfinite(surfaces[key]).all():
                    raise RuntimeError(f"output seed {seed} surface {key} is non-finite")
            row = records.iloc[row_index]
            coverage = surfaces[f"{prefix}_coverage"]
            width = surfaces[f"{prefix}_stage_width"]
            if not np.isclose(float(row["worst_coverage"]), float(coverage.min()), atol=1e-8):
                raise RuntimeError(f"output seed {seed} {method} worst coverage disagrees")
            if not np.isclose(float(row["mean_coverage"]), float(coverage.mean()), atol=1e-8):
                raise RuntimeError(f"output seed {seed} {method} mean coverage disagrees")
            if not np.isclose(float(row["mean_normalized_width"]), float(width.mean()), atol=1e-7):
                raise RuntimeError(f"output seed {seed} {method} mean width disagrees")


def write_summary(output_dir: Path, seeds: tuple[int, ...], *, horizon: int) -> None:
    coverage: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    widths: dict[str, list[float]] = {method: [] for method in METHODS}
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed:05d}"
        with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as surfaces:
            for method in METHODS:
                prefix = _method_prefix(method)
                stage_coverage = np.asarray(surfaces[f"{prefix}_coverage"], dtype=np.float64)
                if stage_coverage.shape != (horizon,):
                    raise RuntimeError(f"seed {seed} {method} coverage shape differs")
                coverage[method].append(stage_coverage)
                widths[method].append(
                    float(np.asarray(surfaces[f"{prefix}_stage_width"], dtype=np.float64).mean())
                )

    coverage_arrays = {
        method: np.stack(values, axis=0) for method, values in coverage.items()
    }
    width_arrays = {
        method: np.asarray(values, dtype=np.float64) for method, values in widths.items()
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(
        0,
        len(seeds),
        size=(BOOTSTRAP_RESAMPLES, len(seeds)),
    )

    csv_rows: list[dict[str, Any]] = []
    method_summary: dict[str, Any] = {}
    for method in METHODS:
        per_stage = coverage_arrays[method].mean(axis=0)
        worst_stage = int(per_stage.argmin())
        coverage_bootstrap = coverage_arrays[method][bootstrap_indices].mean(axis=1).min(axis=1)
        marginal_worst = float(per_stage[worst_stage])
        coverage_ci = _percentile_interval(coverage_bootstrap)
        width_values = width_arrays[method]
        width_bootstrap = width_values[bootstrap_indices].mean(axis=1)
        mean_width = float(width_values.mean())
        width_ci = _percentile_interval(width_bootstrap)
        method_summary[method] = {
            "marginal_worst_coverage": marginal_worst,
            "marginal_worst_coverage_ci95": list(coverage_ci),
            "worst_stage_zero_based": worst_stage,
            "per_stage_marginal_coverage": per_stage.tolist(),
            "mean_normalized_width": mean_width,
            "mean_normalized_width_ci95": list(width_ci),
        }
        csv_rows.extend(
            (
                {
                    "category": "method",
                    "method": method,
                    "metric": "marginal_worst_coverage",
                    "estimate": marginal_worst,
                    "ci95_lower": coverage_ci[0],
                    "ci95_upper": coverage_ci[1],
                    "worst_stage_zero_based": worst_stage,
                    "numerator": "",
                    "denominator": "",
                },
                {
                    "category": "method",
                    "method": method,
                    "metric": "mean_normalized_width",
                    "estimate": mean_width,
                    "ci95_lower": width_ci[0],
                    "ci95_upper": width_ci[1],
                    "worst_stage_zero_based": "",
                    "numerator": "",
                    "denominator": "",
                },
            )
        )

    ratio_summary: dict[str, Any] = {}
    for numerator, denominator in PAIRWISE_RATIOS:
        paired_log_ratio = np.log(width_arrays[numerator] / width_arrays[denominator])
        estimate = float(np.exp(paired_log_ratio.mean()))
        bootstrap = np.exp(paired_log_ratio[bootstrap_indices].mean(axis=1))
        interval = _percentile_interval(bootstrap)
        label = f"{numerator} / {denominator}"
        ratio_summary[label] = {
            "geometric_mean": estimate,
            "ci95": list(interval),
        }
        csv_rows.append(
            {
                "category": "paired_ratio",
                "method": "",
                "metric": "geometric_mean_width_ratio",
                "estimate": estimate,
                "ci95_lower": interval[0],
                "ci95_upper": interval[1],
                "worst_stage_zero_based": "",
                "numerator": numerator,
                "denominator": denominator,
            }
        )

    standard_upper = method_summary[STANDARD]["marginal_worst_coverage_ci95"][1]
    summary = {
        "protocol": PROTOCOL,
        "scenario": SCENARIO,
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "evaluation_rollouts_per_seed": 50_000,
        "standard_calibration": "stagewise finite-sample CP on D_COT + D_cert (3000 trajectories)",
        "coverage_estimand": "min_t mean_seed coverage_t",
        "confidence_intervals": {
            "method": "paired-seed nonparametric percentile bootstrap",
            "level": 0.95,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "methods": method_summary,
        "paired_width_ratios": ratio_summary,
        "standard_ci95_upper_below_target": bool(standard_upper < 0.9),
        "target_coverage": 0.9,
    }
    _atomic_write_text(
        output_dir / "summary.csv",
        pd.DataFrame(csv_rows).to_csv(index=False),
    )
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2) + "\n",
    )


def _validate_resume(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    execution: dict[str, Any],
    source_hash: str,
    config_hash: str,
) -> set[int]:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"resume output does not exist: {output_dir}")
    metadata = _read_json(output_dir / "study_metadata.json")
    stored_config = yaml.safe_load((output_dir / "config.yaml").read_text())
    if not isinstance(stored_config, dict) or canonical_config_sha256(stored_config) != config_hash:
        raise RuntimeError("resume config differs")
    if metadata.get("source_tree_sha256") != source_hash:
        raise RuntimeError("resume source tree differs")
    if metadata.get("execution") != execution:
        raise RuntimeError("resume execution or Phase 0 fingerprint differs")

    completed: set[int] = set()
    for path in output_dir.iterdir():
        match = SEED_DIRECTORY.fullmatch(path.name)
        if match is None:
            if path.name.startswith(".seed_"):
                raise RuntimeError(f"partial seed directory blocks resume: {path}")
            continue
        seed = int(match.group(1))
        if seed not in set(config.seeds):
            raise RuntimeError(f"unexpected seed directory blocks resume: {path}")
        validate_output_seed(
            path,
            seed,
            horizon=config.horizon,
            rollouts=config.samples.oracle_rollouts,
            expected_source_hash=source_hash,
            expected_config_hash=config_hash,
        )
        completed.add(seed)
    return completed


def _positive_vector(values: np.ndarray, length: int, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (length,) or not np.issubdtype(array.dtype, np.number):
        raise RuntimeError(f"{label} must be a numeric vector of length {length}")
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise RuntimeError(f"{label} must be finite and positive")
    return np.array(array, copy=True)


def _require_json_vector_equal(values: np.ndarray, payload: str, *, label: str) -> None:
    parsed = np.asarray(json.loads(payload), dtype=values.dtype)
    if not np.array_equal(values, parsed):
        raise RuntimeError(f"{label} differs between records and surfaces")


def _unique_vector_index(values: np.ndarray, target: float, *, label: str) -> int:
    matches = np.flatnonzero(values == np.asarray(target, dtype=values.dtype))
    if len(matches) != 1:
        raise RuntimeError(f"{label} does not have one bitwise grid match")
    return int(matches[0])


def _unique_row_index(values: np.ndarray, target: np.ndarray, *, label: str) -> int:
    matches = np.flatnonzero(np.equal(values, target[None, :]).all(axis=1))
    if len(matches) != 1:
        raise RuntimeError(f"{label} does not have one bitwise candidate match")
    return int(matches[0])


def _json_vector(values: torch.Tensor) -> str:
    return json.dumps(
        [float(value) for value in values.detach().cpu().tolist()],
        separators=(",", ":"),
    )


def _method_prefix(method: str) -> str:
    return {
        STANDARD: "standard_cp",
        A_ORACLE: "a_sequential_oracle",
        C_ORACLE: "c_profiled_oracle",
    }[method]


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, (0.025, 0.975))
    return float(lower), float(upper)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _update_digest(digest: Any, path: Path, root: Path) -> None:
    relative = path.relative_to(root).as_posix().encode("utf-8")
    content = path.read_bytes()
    digest.update(len(relative).to_bytes(4, "big"))
    digest.update(relative)
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)


def _atomic_write_text(path: Path, content: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}-",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
