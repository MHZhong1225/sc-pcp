"""Run the isolated tail-shift marginal prefix-IW development pilot.

The pilot deliberately targets marginal per-step coverage, not SC-PCP's PAC or
data-conditional certificate.  Prefix-IW and Standard CP use the same combined
D_COT + D_cert calibration budget.  Their frozen schedules and the Phase 0 A/C
oracles are evaluated on the original 50k tail-shift holdout stream.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_phase0_oracle import (  # noqa: E402
    _build_seed_jobs,
    _execute_jobs,
    canonical_config_sha256,
    parse_seeds,
)
from scripts.run_tail_shift_problem_value import (  # noqa: E402
    A_ORACLE,
    C_ORACLE,
    SCENARIO,
    STANDARD,
    load_tail_shift_phase0,
    validate_phase0_inputs,
    verify_phase0_evaluation_parity,
)
from scpcp.artifacts import (  # noqa: E402
    mark_study_complete,
    mark_study_failed,
    write_seed_result,
    write_study_metadata,
)
from scpcp.baselines import standard_cp_stagewise_radii  # noqa: E402
from scpcp.config import ExperimentConfig  # noqa: E402
from scpcp.data import concatenate_trajectories  # noqa: E402
from scpcp.device import resolve_devices  # noqa: E402
from scpcp.experiment import SeedResult, _prepare_task, _training_outcome_sd  # noqa: E402
from scpcp.marginal_prefix import (  # noqa: E402
    profile_log_rmse,
    select_marginal_prefix_schedule,
)
from scpcp.outcome_model import fit_outcome_model  # noqa: E402
from scpcp.phase0_oracle import evaluate_frozen_schedules_crn  # noqa: E402
from scpcp.policy import BehaviorAnchoredPolicy  # noqa: E402
from scpcp.scores import fit_conformal_region, score_batch  # noqa: E402
from scpcp.simulator import make_synthetic_noise_bundle  # noqa: E402


PROTOCOL = "tail_shift_marginal_prefix_iw_v2"
PREFIX_IW = "Marginal Prefix-IW"
METHODS = (PREFIX_IW, STANDARD, A_ORACLE, C_ORACLE)
PAIRWISE_RATIOS = (
    (PREFIX_IW, A_ORACLE),
    (PREFIX_IW, STANDARD),
    (PREFIX_IW, C_ORACLE),
)
MINIMUM_EFFECTIVE_SIZE = 50.0
MAXIMUM_A_WIDTH_RATIO = 1.02
MAXIMUM_STANDARD_WIDTH_RATIO = 1.025
MAXIMUM_C_WIDTH_RATIO = 0.99
MINIMUM_OPPORTUNITY_RECOVERY = 0.50
MAXIMUM_MEAN_COVERAGE_GAP_TO_A = 0.003
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 8_202_684
CONFIRM_SEEDS = tuple(range(1_000, 1_100))
REQUIRED_SEED_FILES = ("COMPLETE", "records.csv", "surfaces.npz", "metadata.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit marginal prefix-IW against fair Standard CP and Phase 0 A/C "
            "on tail shift"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "per_step_synthetic.yaml",
    )
    parser.add_argument("--phase0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="0:20")
    parser.add_argument("--devices", default=None)
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument(
        "--study-role",
        choices=("development", "confirm"),
        required=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers_per_device < 1:
        raise ValueError("workers-per-device must be positive")

    reference_config = ExperimentConfig.from_yaml(args.config)
    if reference_config.data.dataset != "synthetic":
        raise ValueError("marginal prefix pilot requires synthetic data")
    seeds = parse_seeds(args.seeds, reference_config.seeds)
    devices = resolve_devices(args.devices or reference_config.devices)
    output_dir = args.output_dir.resolve()
    config = replace(
        reference_config,
        synthetic=replace(reference_config.synthetic, scenario=SCENARIO),
    ).with_overrides(
        devices=devices,
        seeds=seeds,
        output_dir=output_dir,
    )
    phase0_reference_config = _phase0_reference_config(
        reference_config,
        seeds=seeds,
        devices=devices,
        study_role=args.study_role,
    )
    run_config(
        config,
        phase0_reference_config=phase0_reference_config,
        phase0_dir=args.phase0_dir.resolve(),
        output_dir=output_dir,
        workers_per_device=args.workers_per_device,
        study_role=args.study_role,
    )
    print(output_dir)


def _phase0_reference_config(
    reference_config: ExperimentConfig,
    *,
    seeds: tuple[int, ...],
    devices: tuple[str, ...],
    study_role: str,
) -> ExperimentConfig:
    resolved = reference_config.with_overrides(devices=devices)
    if study_role == "development":
        return resolved
    if study_role != "confirm" or seeds != CONFIRM_SEEDS:
        raise ValueError("confirm is precommitted to seeds 1000:1100")
    # The Phase 0 validator intentionally compares its complete seed block.
    # Only an explicit confirm role may replace the paper config's 0:100 block.
    return resolved.with_overrides(seeds=seeds)


def run_config(
    config: ExperimentConfig,
    *,
    phase0_reference_config: ExperimentConfig,
    phase0_dir: Path,
    output_dir: Path,
    workers_per_device: int,
    study_role: str,
) -> None:
    if config.synthetic.scenario != SCENARIO or config.data.dataset != "synthetic":
        raise ValueError("pilot config must be synthetic tail_shift")
    if config.output_dir != output_dir:
        raise ValueError("config output_dir must match output_dir")
    if config.samples.oracle_rollouts != 50_000:
        raise ValueError("pilot requires the original 50k Phase 0 holdout")
    if output_dir.exists():
        raise FileExistsError(f"fresh pilot output already exists: {output_dir}")
    if study_role not in {"development", "confirm"}:
        raise ValueError("study_role must be development or confirm")
    if study_role == "confirm" and config.seeds != CONFIRM_SEEDS:
        raise ValueError("confirm is precommitted to seeds 1000:1100")

    phase0_fingerprint = validate_phase0_inputs(
        phase0_dir,
        reference_config=phase0_reference_config,
        seeds=config.seeds,
    )
    implementation_hash = _implementation_sha256()
    execution = {
        "protocol": PROTOCOL,
        "phase0_dir": str(phase0_dir),
        "phase0_fingerprint": phase0_fingerprint,
        "implementation_sha256": implementation_hash,
        "workers_per_device": workers_per_device,
        "study_role": study_role,
        "precommit_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_config_sha256(config.to_dict()),
        "seed_block": list(config.seeds),
        "calibration_roles": ["D_COT", "D_cert"],
        "calibration_trajectories": 3_000,
        "candidate_grid": "exact Phase 0 tail-shift stage grids (101 per stage)",
        "importance_weights": "uncapped exact prefix RN, float64 column-max log stabilization",
        "development_seeds_previously_inspected": study_role == "development",
    }
    write_study_metadata(output_dir, config, execution=execution)

    try:
        _run_seeds(
            config,
            phase0_dir=phase0_dir,
            output_dir=output_dir,
            workers_per_device=workers_per_device,
            study_role=study_role,
        )
        if _implementation_sha256() != implementation_hash:
            raise RuntimeError("pilot implementation changed while seeds were running")
        for seed in config.seeds:
            validate_seed_output(
                output_dir / f"seed_{seed:05d}",
                seed=seed,
                horizon=config.horizon,
                rollouts=config.samples.oracle_rollouts,
            )
        write_summary(
            output_dir,
            config.seeds,
            horizon=config.horizon,
            study_role=study_role,
        )
        mark_study_complete(output_dir, config.seeds)
    except BaseException as error:
        mark_study_failed(output_dir, config.seeds, error)
        raise


def _run_seeds(
    config: ExperimentConfig,
    *,
    phase0_dir: Path,
    output_dir: Path,
    workers_per_device: int,
    study_role: str,
) -> None:
    worker_devices, jobs = _build_seed_jobs(
        config.seeds,
        config.devices,
        workers_per_device,
    )
    calls = tuple(
        (
            worker_index,
            (config, seed, device, phase0_dir, output_dir, study_role),
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
    study_role: str,
) -> str:
    def run_and_publish() -> str:
        result = run_seed(
            config,
            seed=seed,
            device=device,
            phase0_seed_dir=phase0_dir / f"seed_{seed:05d}",
            study_role=study_role,
        )
        seed_dir = write_seed_result(result, output_dir, config)
        validate_seed_output(
            seed_dir,
            seed=seed,
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
    study_role: str,
) -> SeedResult:
    phase0 = load_tail_shift_phase0(
        phase0_seed_dir,
        seed=seed,
        horizon=config.horizon,
        target=1.0 - config.certification.alpha,
        rollouts=config.samples.oracle_rollouts,
    )
    stage_grids = _load_phase0_stage_grids(
        phase0_seed_dir,
        horizon=config.horizon,
        grid_size=config.q_grid_size,
    )

    # Match Phase 0's task and predictor replay exactly before introducing the
    # calibration-only marginal method.
    torch.manual_seed(seed)
    task = _prepare_task(config, seed=seed, device=device)
    if task.environment is None or task.logging_policy is None:
        raise RuntimeError("pilot requires the known synthetic environment")
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
    if calibration.n != 3_000:
        raise RuntimeError("pilot calibration budget must be exactly 3000 trajectories")
    calibration_scores = score_batch(
        region,
        calibration.current_states(),
        calibration.actions,
        calibration.outcomes,
    )
    prefix_selection = select_marginal_prefix_schedule(
        calibration,
        calibration_scores,
        stage_grids=stage_grids.to(calibration_scores),
        target_policy=policy,
        logging_policy=task.logging_policy,
        outcome_model=outcome_model,
        outcome_sd=outcome_sd,
        target=1.0 - config.certification.alpha,
    )
    if prefix_selection.radii is None:
        raise RuntimeError(
            f"Prefix-IW has no feasible candidate at stage {prefix_selection.failure_stage}"
        )

    standard_radii = standard_cp_stagewise_radii(
        calibration_scores,
        config.certification.alpha,
    )
    a_schedule = torch.from_numpy(phase0["a_schedule"].copy())
    c_schedule = torch.from_numpy(phase0["c_schedule"].copy())
    schedules = {
        PREFIX_IW: prefix_selection.radii,
        STANDARD: standard_radii,
        A_ORACLE: a_schedule,
        C_ORACLE: c_schedule,
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

    records: list[dict[str, Any]] = []
    surfaces: dict[str, torch.Tensor] = {}
    for method in METHODS:
        evaluation = evaluations[method]
        schedule = schedules[method].detach().cpu()
        prefix = _method_prefix(method)
        records.append(
            {
                "scenario": SCENARIO,
                "method": method,
                "seed": seed,
                "target_coverage": 1.0 - config.certification.alpha,
                "q_by_time": _json_vector(schedule),
                "coverage_by_time": _json_vector(evaluation.coverage),
                "worst_coverage": float(evaluation.coverage.min().item()),
                "mean_coverage": float(evaluation.coverage.mean().item()),
                "stage_width_by_time": _json_vector(evaluation.normalized_width),
                "mean_normalized_width": evaluation.patient_normalized_width,
                "oracle_profile_log_rmse": profile_log_rmse(schedule, a_schedule),
                "evaluation_seed": phase0["evaluation_seed"],
                "evaluation_rollouts": evaluation.n_rollouts,
                "calibration_trajectories": (
                    calibration.n if method in {PREFIX_IW, STANDARD} else 0
                ),
                "selected_endpoint": (
                    prefix_selection.selected_endpoint if method == PREFIX_IW else False
                ),
            }
        )
        surfaces[f"{prefix}_schedule"] = schedule
        surfaces[f"{prefix}_coverage"] = evaluation.coverage.detach().cpu()
        surfaces[f"{prefix}_wilson_lcb"] = evaluation.wilson_lower_bound.detach().cpu()
        surfaces[f"{prefix}_stage_width"] = evaluation.normalized_width.detach().cpu()

    surfaces.update(
        {
            "prefix_stage_grids": stage_grids.detach().cpu(),
            "prefix_selected_indices": torch.tensor(prefix_selection.selected_indices),
            "prefix_tuning_coverage": prefix_selection.estimated_coverage.detach().cpu(),
            "prefix_tuning_normalized_width": (
                prefix_selection.estimated_normalized_width.detach().cpu()
            ),
            "prefix_effective_sample_size": (
                prefix_selection.effective_sample_size.detach().cpu()
            ),
            "prefix_maximum_raw_log_weight": (
                prefix_selection.maximum_raw_log_weight.detach().cpu()
            ),
            "prefix_raw_log_weight_span": (
                prefix_selection.raw_log_weight_span.detach().cpu()
            ),
            "prefix_candidate_effective_sample_size": (
                prefix_selection.candidate_effective_sample_size.detach().cpu()
            ),
            "prefix_candidate_maximum_raw_log_weight": (
                prefix_selection.candidate_maximum_raw_log_weight.detach().cpu()
            ),
            "prefix_candidate_raw_log_weight_span": (
                prefix_selection.candidate_raw_log_weight_span.detach().cpu()
            ),
        }
    )
    diagnostics = {
        "protocol": PROTOCOL,
        "scenario": SCENARIO,
        "study_role": study_role,
        "development_seeds_previously_inspected": study_role == "development",
        "input_phase0_seed_dir": str(phase0_seed_dir),
        "phase0_parity_verified": True,
        "phase0_parity": parity,
        "evaluation_seed": phase0["evaluation_seed"],
        "tuning_seed": phase0["tuning_seed"],
        "calibration_roles": ["D_COT", "D_cert"],
        "calibration_trajectories": calibration.n,
        "candidate_grid": "exact Phase 0 tail-shift stage grid",
        "importance_weights": (
            "uncapped exact prefix RN, float64 column-max log stabilization"
        ),
        "selected_indices": list(prefix_selection.selected_indices),
        "selected_endpoint": prefix_selection.selected_endpoint,
        "minimum_effective_sample_size": float(
            prefix_selection.effective_sample_size.min().item()
        ),
        "maximum_raw_log_weight": float(
            prefix_selection.maximum_raw_log_weight.max().item()
        ),
        "maximum_raw_log_weight_span": float(
            prefix_selection.raw_log_weight_span.max().item()
        ),
        "minimum_candidate_effective_sample_size": float(
            prefix_selection.candidate_effective_sample_size.min().item()
        ),
        "maximum_candidate_raw_log_weight": float(
            prefix_selection.candidate_maximum_raw_log_weight.max().item()
        ),
        "maximum_candidate_raw_log_weight_span": float(
            prefix_selection.candidate_raw_log_weight_span.max().item()
        ),
        "oracle_profile_log_rmse": profile_log_rmse(
            prefix_selection.radii.detach().cpu(),
            a_schedule,
        ),
    }
    return SeedResult(seed, device, records, surfaces, diagnostics)


def _load_phase0_stage_grids(
    seed_dir: Path,
    *,
    horizon: int,
    grid_size: int,
) -> torch.Tensor:
    with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as surfaces:
        values = np.asarray(surfaces["tail_shift_greedy_stage_grids"])
    if values.shape != (horizon, grid_size):
        raise RuntimeError("Phase 0 tail-shift stage grid has the wrong shape")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise RuntimeError("Phase 0 tail-shift stage grid must be finite and positive")
    return torch.from_numpy(np.array(values, copy=True))


def validate_seed_output(
    seed_dir: Path,
    *,
    seed: int,
    horizon: int,
    rollouts: int,
) -> None:
    for name in REQUIRED_SEED_FILES:
        if not (seed_dir / name).is_file():
            raise RuntimeError(f"seed {seed} is missing {name}")
    metadata = json.loads((seed_dir / "metadata.json").read_text())
    diagnostics = metadata.get("diagnostics", {})
    if metadata.get("seed") != seed or diagnostics.get("phase0_parity_verified") is not True:
        raise RuntimeError(f"seed {seed} metadata or Phase 0 parity is invalid")

    records = pd.read_csv(seed_dir / "records.csv")
    if tuple(records["method"]) != METHODS or len(records) != len(METHODS):
        raise RuntimeError(f"seed {seed} must contain ordered Prefix/Standard/A/C rows")
    if not records["seed"].eq(seed).all() or not records["evaluation_rollouts"].eq(rollouts).all():
        raise RuntimeError(f"seed {seed} record metadata differs")
    calibrated = records.loc[records["method"].isin((PREFIX_IW, STANDARD))]
    if not calibrated["calibration_trajectories"].eq(3_000).all():
        raise RuntimeError(f"seed {seed} Prefix/Standard calibration budgets differ")

    with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as surfaces:
        for method in METHODS:
            prefix = _method_prefix(method)
            for suffix in ("schedule", "coverage", "wilson_lcb", "stage_width"):
                values = surfaces[f"{prefix}_{suffix}"]
                if values.shape != (horizon,) or not np.isfinite(values).all():
                    raise RuntimeError(f"seed {seed} has invalid {prefix}_{suffix}")
        if surfaces["prefix_stage_grids"].shape[0] != horizon:
            raise RuntimeError(f"seed {seed} prefix grid has the wrong horizon")
        for name in (
            "prefix_selected_indices",
            "prefix_tuning_coverage",
            "prefix_tuning_normalized_width",
            "prefix_effective_sample_size",
            "prefix_maximum_raw_log_weight",
            "prefix_raw_log_weight_span",
        ):
            if surfaces[name].shape != (horizon,) or not np.isfinite(surfaces[name]).all():
                raise RuntimeError(f"seed {seed} has invalid {name}")
        for name in (
            "prefix_candidate_effective_sample_size",
            "prefix_candidate_maximum_raw_log_weight",
            "prefix_candidate_raw_log_weight_span",
        ):
            if (
                surfaces[name].shape != surfaces["prefix_stage_grids"].shape
                or not np.isfinite(surfaces[name]).all()
            ):
                raise RuntimeError(f"seed {seed} has invalid {name}")


def write_summary(
    output_dir: Path,
    seeds: tuple[int, ...],
    *,
    horizon: int,
    study_role: str,
) -> None:
    coverage = {method: [] for method in METHODS}
    widths = {method: [] for method in METHODS}
    profile_errors = {method: [] for method in METHODS}
    endpoint_seeds: list[int] = []
    minimum_ess = float("inf")
    minimum_candidate_ess = float("inf")
    maximum_raw_log_weight = -float("inf")
    maximum_raw_log_weight_span = 0.0
    maximum_candidate_raw_log_weight = -float("inf")
    maximum_candidate_raw_log_weight_span = 0.0

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed:05d}"
        records = pd.read_csv(seed_dir / "records.csv").set_index("method")
        with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as surfaces:
            for method in METHODS:
                prefix = _method_prefix(method)
                stage_coverage = np.asarray(
                    surfaces[f"{prefix}_coverage"],
                    dtype=np.float64,
                )
                if stage_coverage.shape != (horizon,):
                    raise RuntimeError(f"seed {seed} {method} coverage shape differs")
                coverage[method].append(stage_coverage)
                widths[method].append(
                    float(
                        np.asarray(
                            surfaces[f"{prefix}_stage_width"],
                            dtype=np.float64,
                        ).mean()
                    )
                )
                profile_errors[method].append(
                    float(records.loc[method, "oracle_profile_log_rmse"])
                )
            indices = np.asarray(surfaces["prefix_selected_indices"], dtype=np.int64)
            candidate_count = surfaces["prefix_stage_grids"].shape[1]
            if np.any((indices == 0) | (indices == candidate_count - 1)):
                endpoint_seeds.append(seed)
            minimum_ess = min(
                minimum_ess,
                float(np.asarray(surfaces["prefix_effective_sample_size"]).min()),
            )
            minimum_candidate_ess = min(
                minimum_candidate_ess,
                float(
                    np.asarray(
                        surfaces["prefix_candidate_effective_sample_size"]
                    ).min()
                ),
            )
            maximum_raw_log_weight = max(
                maximum_raw_log_weight,
                float(np.asarray(surfaces["prefix_maximum_raw_log_weight"]).max()),
            )
            maximum_raw_log_weight_span = max(
                maximum_raw_log_weight_span,
                float(np.asarray(surfaces["prefix_raw_log_weight_span"]).max()),
            )
            maximum_candidate_raw_log_weight = max(
                maximum_candidate_raw_log_weight,
                float(
                    np.asarray(
                        surfaces["prefix_candidate_maximum_raw_log_weight"]
                    ).max()
                ),
            )
            maximum_candidate_raw_log_weight_span = max(
                maximum_candidate_raw_log_weight_span,
                float(
                    np.asarray(
                        surfaces["prefix_candidate_raw_log_weight_span"]
                    ).max()
                ),
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

    methods: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for method in METHODS:
        per_stage = coverage_arrays[method].mean(axis=0)
        lower, upper, standard_error, critical_lower, critical_upper = (
            _simultaneous_max_t_bands(
                coverage_arrays[method],
                bootstrap_indices,
            )
        )
        per_seed_mean_coverage = coverage_arrays[method].mean(axis=1)
        mean_coverage_bootstrap = per_seed_mean_coverage[bootstrap_indices].mean(axis=1)
        width_bootstrap = width_arrays[method][bootstrap_indices].mean(axis=1)
        methods[method] = {
            "marginal_worst_coverage": float(per_stage.min()),
            "worst_stage_zero_based": int(per_stage.argmin()),
            "per_stage_marginal_coverage": per_stage.tolist(),
            "simultaneous_one_sided_95_lower_by_stage": lower.tolist(),
            "simultaneous_one_sided_95_upper_by_stage": upper.tolist(),
            "simultaneous_worst_coverage_band": [
                float(lower.min()),
                float(upper.min()),
            ],
            "seed_cluster_standard_error_by_stage": standard_error.tolist(),
            "max_t_lower_critical_value": critical_lower,
            "max_t_upper_critical_value": critical_upper,
            "mean_coverage": float(per_seed_mean_coverage.mean()),
            "mean_coverage_ci95": list(
                _percentile_interval(mean_coverage_bootstrap)
            ),
            "mean_normalized_width": float(width_arrays[method].mean()),
            "mean_normalized_width_ci95": list(_percentile_interval(width_bootstrap)),
            "mean_oracle_profile_log_rmse": float(np.mean(profile_errors[method])),
        }
        csv_rows.append(
            {
                "category": "method",
                "method": method,
                "marginal_worst_coverage": methods[method]["marginal_worst_coverage"],
                "simultaneous_worst_upper": float(upper.min()),
                "mean_coverage": methods[method]["mean_coverage"],
                "mean_normalized_width": methods[method]["mean_normalized_width"],
                "mean_oracle_profile_log_rmse": methods[method][
                    "mean_oracle_profile_log_rmse"
                ],
            }
        )

    ratios: dict[str, Any] = {}
    for numerator, denominator in PAIRWISE_RATIOS:
        label = f"{numerator} / {denominator}"
        ratios[label] = _paired_width_ratio(
            width_arrays[numerator],
            width_arrays[denominator],
            bootstrap_indices,
        )
        csv_rows.append(
            {
                "category": "paired_width_ratio",
                "method": label,
                "marginal_worst_coverage": "",
                "simultaneous_worst_upper": "",
                "mean_coverage": "",
                "mean_normalized_width": ratios[label]["geometric_mean"],
                "mean_oracle_profile_log_rmse": "",
            }
        )

    mean_coverage_difference = (
        coverage_arrays[PREFIX_IW].mean(axis=1)
        - coverage_arrays[A_ORACLE].mean(axis=1)
    )
    mean_coverage_difference_bootstrap = mean_coverage_difference[
        bootstrap_indices
    ].mean(axis=1)
    mean_coverage_comparison = {
        "estimate": float(mean_coverage_difference.mean()),
        "one_sided_95_upper": float(
            np.quantile(mean_coverage_difference_bootstrap, 0.95)
        ),
    }
    opportunity = _opportunity_recovery(
        width_arrays,
        bootstrap_indices,
    )

    prefix_upper = np.asarray(
        methods[PREFIX_IW]["simultaneous_one_sided_95_upper_by_stage"]
    )
    standard_upper = np.asarray(
        methods[STANDARD]["simultaneous_one_sided_95_upper_by_stage"]
    )
    method_gates = _method_gate_decisions(
        prefix_point_worst=methods[PREFIX_IW]["marginal_worst_coverage"],
        prefix_simultaneous_upper=prefix_upper,
        standard_simultaneous_upper=standard_upper,
        ratios=ratios,
        opportunity=opportunity,
        mean_coverage_comparison=mean_coverage_comparison,
        endpoint_seeds=endpoint_seeds,
        minimum_ess=minimum_ess,
        minimum_candidate_ess=minimum_candidate_ess,
    )
    confirm_protocol_gates = _confirm_protocol_gates(study_role, seeds)
    summary = {
        "protocol": PROTOCOL,
        "scenario": SCENARIO,
        "study_role": study_role,
        "development_seeds_previously_inspected": study_role == "development",
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "evaluation_rollouts_per_seed": 50_000,
        "calibration_trajectories": 3_000,
        "standard_calibration": "all D_COT + D_cert (size matched, 3000 trajectories)",
        "candidate_grid": "exact Phase 0 tail-shift stage grids (101 per stage)",
        "importance_weights": (
            "uncapped exact prefix RN, float64 column-max log stabilization"
        ),
        "coverage_estimand": "min_t mean_seed coverage_t",
        "coverage_band": (
            "seed-cluster studentized centered max-t bootstrap with fixed original "
            "stage SE and shared resamples"
        ),
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "shared_across_methods_and_stages": True,
        },
        "methods": methods,
        "paired_width_ratios": ratios,
        "opportunity_recovery": opportunity,
        "mean_coverage_prefix_minus_A": mean_coverage_comparison,
        "prefix_diagnostics": {
            "endpoint_seeds": endpoint_seeds,
            "minimum_effective_sample_size": minimum_ess,
            "minimum_candidate_effective_sample_size": minimum_candidate_ess,
            "maximum_raw_log_weight": maximum_raw_log_weight,
            "maximum_raw_log_weight_span": maximum_raw_log_weight_span,
            "maximum_candidate_raw_log_weight": maximum_candidate_raw_log_weight,
            "maximum_candidate_raw_log_weight_span": (
                maximum_candidate_raw_log_weight_span
            ),
        },
        "method_gates": method_gates,
        "development_go": study_role == "development" and all(method_gates.values()),
        "confirm_protocol_gates": confirm_protocol_gates,
        "confirm_go": (
            all(method_gates.values()) and all(confirm_protocol_gates.values())
        ),
    }
    (output_dir / "summary.csv").write_text(
        pd.DataFrame(csv_rows).to_csv(index=False),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def _simultaneous_max_t_bands(
    values_by_seed: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if values_by_seed.ndim != 2 or len(values_by_seed) < 2:
        raise ValueError("coverage values must have shape [S,T] with S >= 2")
    estimate = values_by_seed.mean(axis=0)
    standard_error = values_by_seed.std(axis=0, ddof=1) / np.sqrt(len(values_by_seed))
    if not np.isfinite(standard_error).all() or np.any(standard_error <= 0.0):
        raise RuntimeError("simultaneous max-t band requires positive finite stage SE")
    bootstrap_mean = values_by_seed[bootstrap_indices].mean(axis=1)
    lower_statistics = ((estimate[None, :] - bootstrap_mean) / standard_error).max(
        axis=1
    )
    upper_statistics = ((bootstrap_mean - estimate[None, :]) / standard_error).max(
        axis=1
    )
    critical_lower = float(np.quantile(lower_statistics, 0.95))
    critical_upper = float(np.quantile(upper_statistics, 0.95))
    lower = estimate - critical_lower * standard_error
    upper = estimate + critical_upper * standard_error
    return lower, upper, standard_error, critical_lower, critical_upper


def _opportunity_recovery(
    width_arrays: dict[str, np.ndarray],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    log_width = {method: np.log(values) for method, values in width_arrays.items()}
    denominator = float(log_width[C_ORACLE].mean() - log_width[A_ORACLE].mean())
    numerator = float(log_width[C_ORACLE].mean() - log_width[PREFIX_IW].mean())
    bootstrap_denominator = (
        log_width[C_ORACLE][bootstrap_indices].mean(axis=1)
        - log_width[A_ORACLE][bootstrap_indices].mean(axis=1)
    )
    if denominator <= 0.0 or np.any(bootstrap_denominator <= 0.0):
        return {
            "defined": False,
            "estimate": None,
            "one_sided_95_lower": None,
            "failure": "C-to-A log-width opportunity is non-positive",
        }
    bootstrap_numerator = (
        log_width[C_ORACLE][bootstrap_indices].mean(axis=1)
        - log_width[PREFIX_IW][bootstrap_indices].mean(axis=1)
    )
    bootstrap_recovery = bootstrap_numerator / bootstrap_denominator
    return {
        "defined": True,
        "estimate": numerator / denominator,
        "one_sided_95_lower": float(np.quantile(bootstrap_recovery, 0.05)),
        "denominator_log_width_opportunity": denominator,
    }


def _paired_width_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    if numerator.shape != denominator.shape or np.any(numerator <= 0.0) or np.any(
        denominator <= 0.0
    ):
        raise ValueError("paired widths must be positive vectors with shared shape")
    paired_log_ratio = np.log(numerator / denominator)
    bootstrap = np.exp(paired_log_ratio[bootstrap_indices].mean(axis=1))
    return {
        "geometric_mean": float(np.exp(paired_log_ratio.mean())),
        "ci95": list(_percentile_interval(bootstrap)),
        "one_sided_95_upper": float(np.quantile(bootstrap, 0.95)),
    }


def _method_gate_decisions(
    *,
    prefix_point_worst: float,
    prefix_simultaneous_upper: np.ndarray,
    standard_simultaneous_upper: np.ndarray,
    ratios: dict[str, dict[str, Any]],
    opportunity: dict[str, Any],
    mean_coverage_comparison: dict[str, float],
    endpoint_seeds: list[int],
    minimum_ess: float,
    minimum_candidate_ess: float,
) -> dict[str, bool]:
    return {
        "prefix_point_worst_coverage_at_least_0.9000": bool(
            prefix_point_worst >= 0.9
        ),
        "prefix_no_stage_simultaneous_upper_below_0.9000": bool(
            prefix_simultaneous_upper.min() >= 0.9
        ),
        "standard_has_stage_simultaneous_upper_below_0.9000": bool(
            standard_simultaneous_upper.min() < 0.9
        ),
        "prefix_to_A_width_ratio_q95_at_most_1.02": bool(
            ratios[f"{PREFIX_IW} / {A_ORACLE}"]["one_sided_95_upper"]
            <= MAXIMUM_A_WIDTH_RATIO
        ),
        "prefix_to_Standard_width_ratio_q95_at_most_1.025": bool(
            ratios[f"{PREFIX_IW} / {STANDARD}"]["one_sided_95_upper"]
            <= MAXIMUM_STANDARD_WIDTH_RATIO
        ),
        "prefix_to_C_width_ratio_q95_below_0.99": bool(
            ratios[f"{PREFIX_IW} / {C_ORACLE}"]["one_sided_95_upper"]
            < MAXIMUM_C_WIDTH_RATIO
        ),
        "opportunity_recovery_q05_above_0.50": bool(
            opportunity["defined"]
            and opportunity["one_sided_95_lower"]
            > MINIMUM_OPPORTUNITY_RECOVERY
        ),
        "mean_coverage_minus_A_q95_at_most_0.003": bool(
            mean_coverage_comparison["one_sided_95_upper"]
            <= MAXIMUM_MEAN_COVERAGE_GAP_TO_A
        ),
        "no_endpoint_selection": not endpoint_seeds,
        "no_grid_failure": True,
        "minimum_ess_at_least_50": bool(minimum_ess >= MINIMUM_EFFECTIVE_SIZE),
        "uncapped_exact_prefix_weights": True,
        "size_matched_standard_all_3000": True,
    }


def _confirm_protocol_gates(
    study_role: str,
    seeds: tuple[int, ...],
) -> dict[str, bool]:
    return {
        "study_role_is_confirm": study_role == "confirm",
        "precommitted_seed_block_1000_1100": seeds == CONFIRM_SEEDS,
        "all_100_seed_artifacts_complete": len(seeds) == 100,
    }


def _method_prefix(method: str) -> str:
    return {
        PREFIX_IW: "marginal_prefix_iw",
        STANDARD: "standard_cp",
        A_ORACLE: "a_sequential_oracle",
        C_ORACLE: "c_profiled_oracle",
    }[method]


def _json_vector(values: torch.Tensor) -> str:
    return json.dumps(
        [float(value) for value in values.detach().cpu().tolist()],
        separators=(",", ":"),
    )


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, (0.025, 0.975))
    return float(lower), float(upper)


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src" / "scpcp" / "marginal_prefix.py",
        Path(__file__),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
