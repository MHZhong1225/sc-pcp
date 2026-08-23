"""Fresh common-CRN evaluation of the frozen A/C/D/E decomposition."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_phase0_oracle import (
    _build_seed_jobs,
    _execute_jobs,
    canonical_config_sha256,
    parse_seeds,
    validate_seed_artifact as validate_phase0_seed,
)
from scpcp.artifacts import (
    experiment_tree_sha256,
    mark_study_complete,
    mark_study_failed,
    source_tree_sha256,
    write_seed_result,
    write_study_metadata,
)
from scpcp.config import ExperimentConfig
from scpcp.conservatism_decomposition import (
    FreshEvaluation,
    LAYERS,
    RUNNER_LAYERS,
    canonical_fresh_records,
    load_standard_decomposition,
)
from scpcp.device import resolve_devices
from scpcp.experiment import (
    SeedResult,
    _paper_seed,
    _prepare_task,
    _training_outcome_sd,
)
from scpcp.outcome_model import fit_outcome_model
from scpcp.phase0_oracle import evaluate_frozen_schedules_crn
from scpcp.policy import BehaviorAnchoredPolicy
from scpcp.scores import fit_conformal_region
from scpcp.simulator import make_synthetic_noise_bundle


FRESH_EVALUATION_STREAM = 1_600_001
SEED_DIRECTORY = re.compile(r"seed_(\d{5})")
REQUIRED_INPUT_FILES = ("COMPLETE", "records.csv", "surfaces.npz", "metadata.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen sequential/profiled/COT-point/LCB schedules on one "
            "independent common-random-number rollout stream"
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase0_oracle.yaml")
    parser.add_argument("--phase0-dir", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default=None, help="range such as 0:100 or comma-separated seeds")
    parser.add_argument("--devices", default=None, help="comma-separated CUDA devices")
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=50_000)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers_per_device < 1:
        raise ValueError("workers-per-device must be positive")
    if args.rollouts < 1:
        raise ValueError("rollouts must be positive")

    base = ExperimentConfig.from_yaml(args.config)
    devices = resolve_devices(args.devices or base.devices)
    seeds = parse_seeds(args.seeds, base.seeds)
    config = base.with_overrides(devices=devices, seeds=seeds, output_dir=args.output_dir)
    run_config(
        config,
        phase0_dir=args.phase0_dir.resolve(),
        paper_dir=args.paper_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        workers_per_device=args.workers_per_device,
        rollouts=args.rollouts,
        resume=args.resume,
    )
    print(args.output_dir.resolve())


def run_config(
    config: ExperimentConfig,
    *,
    phase0_dir: Path,
    paper_dir: Path,
    output_dir: Path,
    workers_per_device: int,
    rollouts: int,
    resume: bool,
) -> None:
    if config.data.dataset != "synthetic" or config.synthetic.scenario != "standard":
        raise ValueError("the frozen decomposition requires the standard synthetic scenario")
    if config.output_dir.resolve() != output_dir:
        raise ValueError("config output_dir must match output_dir")
    if rollouts != config.samples.oracle_rollouts:
        raise ValueError(
            "formal decomposition rollouts must equal config.samples.oracle_rollouts"
        )
    _validate_fresh_streams(config.seeds)

    phase0_fingerprint = validate_input_study(phase0_dir, config.seeds, kind="phase0")
    paper_fingerprint = validate_input_study(paper_dir, config.seeds, kind="paper")
    _validate_input_configs(config, phase0_dir, paper_dir)

    current_source_hash = source_tree_sha256()
    config_hash = canonical_config_sha256(config.to_dict())
    execution = {
        "protocol": "frozen_standard_acde_common_fresh_crn",
        "experiment_tree_sha256": experiment_tree_sha256(),
        "config_sha256": config_hash,
        "phase0_dir": str(phase0_dir),
        "phase0_fingerprint": phase0_fingerprint,
        "paper_dir": str(paper_dir),
        "paper_fingerprint": paper_fingerprint,
        "workers_per_device": workers_per_device,
        "fresh_evaluation_rollouts": rollouts,
        "fresh_evaluation_stream": FRESH_EVALUATION_STREAM,
    }

    if resume:
        completed = _validate_resume(
            output_dir,
            config,
            execution=execution,
            source_hash=current_source_hash,
            config_hash=config_hash,
            rollouts=rollouts,
        )
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh decomposition output already exists: {output_dir}")
        write_study_metadata(output_dir, config, execution=execution)
        completed = set()

    pending = tuple(seed for seed in config.seeds if seed not in completed)
    try:
        _run_pending_seeds(
            config,
            phase0_dir=phase0_dir,
            paper_dir=paper_dir,
            output_dir=output_dir,
            seeds=pending,
            workers_per_device=workers_per_device,
            rollouts=rollouts,
        )
        for seed in config.seeds:
            validate_output_seed(
                output_dir / f"seed_{seed:05d}",
                seed,
                horizon=config.horizon,
                rollouts=rollouts,
                expected_source_hash=current_source_hash,
                expected_config_hash=config_hash,
            )
        if validate_input_study(phase0_dir, config.seeds, kind="phase0") != phase0_fingerprint:
            raise RuntimeError("phase0 input changed while the decomposition was running")
        if validate_input_study(paper_dir, config.seeds, kind="paper") != paper_fingerprint:
            raise RuntimeError("paper input changed while the decomposition was running")
        mark_study_complete(output_dir, config.seeds)
    except BaseException as error:
        mark_study_failed(output_dir, config.seeds, error)
        raise


def validate_input_study(root: Path, seeds: tuple[int, ...], *, kind: str) -> str:
    if kind not in {"phase0", "paper"}:
        raise ValueError("input kind must be phase0 or paper")
    for name in ("COMPLETE", "config.yaml", "study_metadata.json", "study_status.json"):
        if not (root / name).is_file():
            raise RuntimeError(f"{kind} input is missing {name}: {root}")

    status = _read_json(root / "study_status.json")
    if (
        status.get("status") != "complete"
        or status.get("expected_seeds") != list(seeds)
        or status.get("completed_seeds") != list(seeds)
        or status.get("missing_seeds") not in ([], None)
        or status.get("error") is not None
    ):
        raise RuntimeError(f"{kind} input study_status is not exactly complete")
    metadata_root = _read_json(root / "study_metadata.json")
    root_source_hash = metadata_root.get("source_tree_sha256")
    root_git_revision = metadata_root.get("git_revision")
    if not isinstance(root_source_hash, str) or not root_source_hash:
        raise RuntimeError(f"{kind} root source hash is missing")
    raw_config = yaml.safe_load((root / "config.yaml").read_text())
    if not isinstance(raw_config, dict):
        raise RuntimeError(f"{kind} root config must contain a mapping")
    root_config_hash = canonical_config_sha256(raw_config)

    observed_seed_dirs = set()
    for path in root.iterdir():
        if path.name.startswith(".seed_"):
            raise RuntimeError(f"partial {kind} seed directory found: {path}")
        match = SEED_DIRECTORY.fullmatch(path.name)
        if match is not None:
            if not path.is_dir():
                raise RuntimeError(f"malformed {kind} seed path: {path}")
            observed_seed_dirs.add(int(match.group(1)))
    if observed_seed_dirs != set(seeds):
        raise RuntimeError(f"{kind} seed directories do not exactly match requested seeds")

    digest = hashlib.sha256()
    for seed in seeds:
        seed_dir = root / f"seed_{seed:05d}"
        for name in REQUIRED_INPUT_FILES:
            path = seed_dir / name
            if not path.is_file():
                raise RuntimeError(f"{kind} seed {seed} is missing {name}")
        metadata = _read_json(seed_dir / "metadata.json")
        if metadata.get("seed") != seed:
            raise RuntimeError(f"{kind} seed {seed} metadata has the wrong seed")
        if metadata.get("source_tree_sha256") != root_source_hash:
            raise RuntimeError(f"{kind} seed {seed} source hash differs from its root")
        if metadata.get("git_revision") != root_git_revision:
            raise RuntimeError(f"{kind} seed {seed} git revision differs from its root")
        seed_config = metadata.get("config")
        if not isinstance(seed_config, dict) or canonical_config_sha256(seed_config) != root_config_hash:
            raise RuntimeError(f"{kind} seed {seed} config differs from its root")
        if kind == "phase0":
            validate_phase0_seed(
                seed_dir,
                seed,
                expected_source_hash=root_source_hash,
                expected_config_hash=root_config_hash,
            )
        else:
            records = pd.read_csv(seed_dir / "records.csv")
            if len(records) != 6 or records["method"].nunique() != 6:
                raise RuntimeError(f"paper seed {seed} must contain exactly six methods")
            if int(records["method"].eq("SC-PCP").sum()) != 1:
                raise RuntimeError(f"paper seed {seed} must contain exactly one SC-PCP row")
        with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as surfaces:
            required = (
                {
                    "standard_profiled_scale_grid",
                    "standard_profile",
                    "standard_profiled_candidate_schedules",
                    "standard_profiled_candidate_coverage",
                    "standard_profiled_candidate_normalized_width",
                    "standard_profiled_selected_schedule",
                    "standard_greedy_stage_grids",
                    "standard_greedy_selected_schedule",
                }
                if kind == "phase0"
                else {
                    "scale_grid",
                    "stage_profile",
                    "candidate_radii",
                    "cot_diagonal",
                    "cot_lower_bounds",
                    "estimated_candidate_widths",
                    "scpcp_selected_radii",
                }
            )
            missing = sorted(required - set(surfaces.files))
            if missing:
                raise RuntimeError(f"{kind} seed {seed} surfaces are missing {missing}")
        for name in REQUIRED_INPUT_FILES:
            _update_file_digest(digest, seed_dir / name, root)
    _update_file_digest(digest, root / "config.yaml", root)
    _update_file_digest(digest, root / "study_metadata.json", root)
    _update_file_digest(digest, root / "study_status.json", root)
    _update_file_digest(digest, root / "COMPLETE", root)
    if kind == "phase0":
        for name in ("phase0_decision.json", "phase0_summary_manifest.json"):
            if not (root / name).is_file():
                raise RuntimeError(f"phase0 input is missing {name}")
            _update_file_digest(digest, root / name, root)
    return digest.hexdigest()


def validate_output_seed(
    seed_dir: Path,
    seed: int,
    *,
    horizon: int,
    rollouts: int,
    expected_source_hash: str | None = None,
    expected_config_hash: str | None = None,
) -> None:
    for name in REQUIRED_INPUT_FILES:
        if not (seed_dir / name).is_file():
            raise RuntimeError(f"output seed {seed} is missing {name}")
    metadata = _read_json(seed_dir / "metadata.json")
    if metadata.get("seed") != seed:
        raise RuntimeError(f"output seed {seed} metadata has the wrong seed")
    if expected_source_hash is not None and metadata.get("source_tree_sha256") != expected_source_hash:
        raise RuntimeError(f"output seed {seed} source hash differs")
    if expected_config_hash is not None:
        stored_config = metadata.get("config")
        if not isinstance(stored_config, dict) or canonical_config_sha256(stored_config) != expected_config_hash:
            raise RuntimeError(f"output seed {seed} config differs")

    records = pd.read_csv(seed_dir / "records.csv")
    if len(records) != len(LAYERS) or tuple(records["layer"]) != LAYERS:
        raise RuntimeError(f"output seed {seed} must contain ordered A/C/D/E rows")
    if not records["seed"].eq(seed).all():
        raise RuntimeError(f"output seed {seed} records contain another seed")
    if not records["oracle_evaluation_trajectories"].eq(rollouts).all():
        raise RuntimeError(f"output seed {seed} has the wrong rollout count")
    if not np.isfinite(records[["worst_coverage", "average_coverage", "average_normalized_width"]]).all().all():
        raise RuntimeError(f"output seed {seed} records contain non-finite metrics")
    diagnostics = metadata.get("diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("phase0_replay_verified") is not True:
        raise RuntimeError(f"output seed {seed} lacks a passing Phase0 replay")

    with np.load(seed_dir / "surfaces.npz", allow_pickle=False) as surfaces:
        for row_index, layer in enumerate(LAYERS):
            for prefix in ("schedule", "fresh_coverage", "fresh_wilson_lcb", "fresh_stage_width"):
                key = f"{layer}_{prefix}"
                if key not in surfaces.files or surfaces[key].shape != (horizon,):
                    raise RuntimeError(f"output seed {seed} has invalid surface {key}")
                if not np.isfinite(surfaces[key]).all():
                    raise RuntimeError(f"output seed {seed} surface {key} is non-finite")
            schedule = surfaces[f"{layer}_schedule"]
            coverage = surfaces[f"{layer}_fresh_coverage"]
            lower = surfaces[f"{layer}_fresh_wilson_lcb"]
            width = surfaces[f"{layer}_fresh_stage_width"]
            if np.any(schedule <= 0.0) or np.any(width <= 0.0):
                raise RuntimeError(f"output seed {seed} {layer} schedule/width must be positive")
            if np.any((coverage < 0.0) | (coverage > 1.0)) or np.any((lower < 0.0) | (lower > 1.0)):
                raise RuntimeError(f"output seed {seed} {layer} coverage/LCB is outside [0,1]")
            row = records.iloc[row_index]
            if not np.isclose(float(row["worst_coverage"]), float(coverage.min()), atol=1e-8):
                raise RuntimeError(f"output seed {seed} {layer} worst coverage disagrees")
            if not np.isclose(float(row["average_coverage"]), float(coverage.mean()), atol=1e-8):
                raise RuntimeError(f"output seed {seed} {layer} average coverage disagrees")
            if bool(row["target_met"]) != bool(coverage.min() >= float(row["target_coverage"])):
                raise RuntimeError(f"output seed {seed} {layer} target_met disagrees")
            if not np.array_equal(
                np.asarray(json.loads(row["q_by_time"]), dtype=schedule.dtype),
                schedule,
            ):
                raise RuntimeError(f"output seed {seed} {layer} schedule record disagrees")


def _run_pending_seeds(
    config: ExperimentConfig,
    *,
    phase0_dir: Path,
    paper_dir: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    workers_per_device: int,
    rollouts: int,
) -> None:
    if not seeds:
        return
    worker_devices, jobs = _build_seed_jobs(seeds, config.devices, workers_per_device)
    calls = tuple(
        (
            worker_index,
            (config, seed, device, phase0_dir, paper_dir, output_dir, rollouts),
        )
        for worker_index, seed, device in jobs
    )
    for result in _execute_jobs(worker_devices, calls, worker_function=_run_and_write):
        print(result, flush=True)


def _run_and_write(
    config: ExperimentConfig,
    seed: int,
    device: str,
    phase0_dir: Path,
    paper_dir: Path,
    output_dir: Path,
    rollouts: int,
) -> str:
    def run_and_publish() -> str:
        result = run_decomposition_seed(
            config,
            seed=seed,
            device=device,
            phase0_seed_dir=phase0_dir / f"seed_{seed:05d}",
            paper_seed_dir=paper_dir / f"seed_{seed:05d}",
            rollouts=rollouts,
        )
        seed_dir = write_seed_result(result, output_dir, config)
        validate_output_seed(
            seed_dir,
            seed,
            horizon=config.horizon,
            rollouts=rollouts,
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


def run_decomposition_seed(
    config: ExperimentConfig,
    *,
    seed: int,
    device: str,
    phase0_seed_dir: Path,
    paper_seed_dir: Path,
    rollouts: int,
) -> SeedResult:
    selection = load_standard_decomposition(
        phase0_seed_dir / "surfaces.npz",
        paper_seed_dir / "surfaces.npz",
        target=1.0 - config.certification.alpha,
    )
    # No Torch operation is allowed between this reset and task/model replay.
    torch.manual_seed(seed)
    task = _prepare_task(config, seed=seed, device=device)
    if task.name != "synthetic" or task.environment is None or task.logging_policy is None:
        raise RuntimeError("fresh decomposition requires the known synthetic task")
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

    schedules_by_layer = selection.schedules.by_layer()
    runner_schedules = {
        runner_name: torch.from_numpy(np.array(schedules_by_layer[layer], copy=True))
        for runner_name, layer in RUNNER_LAYERS.items()
    }
    replay_diagnostics = _verify_phase0_replay(
        config,
        seed=seed,
        phase0_seed_dir=phase0_seed_dir,
        task=task,
        policy=policy,
        outcome_model=outcome_model,
        outcome_sd=outcome_sd,
        schedules=runner_schedules,
        device=device,
    )
    evaluation_seed = _paper_seed(seed, FRESH_EVALUATION_STREAM)
    forbidden = _registered_streams_for_seed(seed)
    if evaluation_seed in forbidden:
        raise RuntimeError("fresh evaluation stream collides with an input study stream")
    noise = make_synthetic_noise_bundle(
        n=rollouts,
        horizon=config.horizon,
        seed=evaluation_seed,
        device=device,
    )
    runner_evaluations = evaluate_frozen_schedules_crn(
        task.environment,
        policy,
        outcome_model,
        schedules=runner_schedules,
        noise=noise,
        outcome_sd=outcome_sd,
        forbidden_noise_seeds=forbidden,
    )

    evaluations: dict[str, FreshEvaluation] = {}
    surfaces: dict[str, torch.Tensor] = {}
    for runner_name, layer in RUNNER_LAYERS.items():
        evaluated = runner_evaluations[runner_name]
        evaluations[layer] = FreshEvaluation(
            coverage=evaluated.coverage.detach().cpu().numpy(),
            normalized_width=evaluated.normalized_width.detach().cpu().numpy(),
            micro_normalized_width=evaluated.micro_normalized_width,
            patient_normalized_width=evaluated.patient_normalized_width,
            n_rollouts=evaluated.n_rollouts,
        )
        surfaces[f"{layer}_schedule"] = runner_schedules[runner_name]
        surfaces[f"{layer}_fresh_coverage"] = evaluated.coverage
        surfaces[f"{layer}_fresh_wilson_lcb"] = evaluated.wilson_lower_bound
        surfaces[f"{layer}_fresh_stage_width"] = evaluated.normalized_width

    records = [dict(row) for row in canonical_fresh_records(seed, selection, evaluations)]
    for record in records:
        layer = str(record["layer"])
        runner_name = next(name for name, short in RUNNER_LAYERS.items() if short == layer)
        evaluated = runner_evaluations[runner_name]
        record["fresh_wilson_lcb_min"] = float(evaluated.wilson_lower_bound.min().item())
        record["fresh_wilson_lcb_by_time"] = json.dumps(
            [float(value) for value in evaluated.wilson_lower_bound.tolist()],
            separators=(",", ":"),
        )
        record["evaluation_seed"] = evaluation_seed

    diagnostics = {
        "protocol": "frozen_standard_acde_common_fresh_crn",
        "evaluation_seed": evaluation_seed,
        "evaluation_rollouts": rollouts,
        "input_phase0_seed_dir": str(phase0_seed_dir),
        "input_paper_seed_dir": str(paper_seed_dir),
        "profile_refit": False,
        "cot_refit": False,
        "phase0_replay_verified": True,
        "phase0_replay": replay_diagnostics,
        "selection": asdict(selection.diagnostics),
        "indices": asdict(selection.indices),
    }
    return SeedResult(seed, device, records, surfaces, diagnostics)


def _verify_phase0_replay(
    config: ExperimentConfig,
    *,
    seed: int,
    phase0_seed_dir: Path,
    task: Any,
    policy: BehaviorAnchoredPolicy,
    outcome_model: Any,
    outcome_sd: torch.Tensor,
    schedules: dict[str, torch.Tensor],
    device: str,
) -> dict[str, object]:
    """Replay the original A/C evaluation stream as a model reconstruction gate."""

    records = pd.read_csv(phase0_seed_dir / "records.csv")
    standard = records.loc[records["scenario"].eq("standard")].copy()
    expected_methods = {"Greedy Sequential Oracle", "Current Profiled Oracle"}
    if len(standard) != 2 or set(standard["method"]) != expected_methods:
        raise RuntimeError(f"phase0 seed {seed} has an invalid standard record contract")
    evaluation_seeds = {int(value) for value in standard["evaluation_seed"]}
    rollout_counts = {int(value) for value in standard["n_rollouts"]}
    if len(evaluation_seeds) != 1 or rollout_counts != {config.samples.oracle_rollouts}:
        raise RuntimeError(f"phase0 seed {seed} replay stream or rollout count differs")
    evaluation_seed = next(iter(evaluation_seeds))
    tuning_seeds = {int(value) for value in standard["tuning_seed"]}
    if evaluation_seed in tuning_seeds:
        raise RuntimeError(f"phase0 seed {seed} tuning/evaluation stream collision")

    noise = make_synthetic_noise_bundle(
        n=config.samples.oracle_rollouts,
        horizon=config.horizon,
        seed=evaluation_seed,
        device=device,
    )
    replay = evaluate_frozen_schedules_crn(
        task.environment,
        policy,
        outcome_model,
        schedules={
            "Greedy Sequential Oracle": schedules["A_sequential"],
            "Current Profiled Oracle": schedules["C_profiled_oracle"],
        },
        noise=noise,
        outcome_sd=outcome_sd,
        forbidden_noise_seeds=tuning_seeds,
    )

    maximum_coverage_error = 0.0
    maximum_width_error = 0.0
    for method in sorted(expected_methods):
        row = standard.loc[standard["method"].eq(method)].iloc[0]
        evaluated = replay[method]
        expected_coverage = torch.tensor(
            json.loads(row["final_coverage"]),
            dtype=evaluated.coverage.dtype,
        )
        expected_width = torch.tensor(
            json.loads(row["final_stage_width"]),
            dtype=evaluated.normalized_width.dtype,
        )
        actual_coverage = evaluated.coverage.detach().cpu()
        actual_width = evaluated.normalized_width.detach().cpu()
        if not torch.equal(actual_coverage, expected_coverage):
            raise RuntimeError(f"phase0 seed {seed} {method} coverage replay mismatch")
        if not torch.equal(actual_width, expected_width):
            raise RuntimeError(f"phase0 seed {seed} {method} width replay mismatch")
        maximum_coverage_error = max(
            maximum_coverage_error,
            float((actual_coverage - expected_coverage).abs().max().item()),
        )
        maximum_width_error = max(
            maximum_width_error,
            float((actual_width - expected_width).abs().max().item()),
        )
    return {
        "evaluation_seed": evaluation_seed,
        "tuning_seeds": sorted(tuning_seeds),
        "rollouts": config.samples.oracle_rollouts,
        "methods": sorted(expected_methods),
        "maximum_coverage_error": maximum_coverage_error,
        "maximum_stage_width_error": maximum_width_error,
    }


def _registered_streams_for_seed(seed: int) -> set[int]:
    adaptation_stream = _paper_seed(seed, 700_001)
    streams = {
        seed,
        seed + 1,
        seed + 2,
        seed + 3,
        seed + 31_337,
        _paper_seed(seed, 300_001),
        _paper_seed(seed, 310_001),
        _paper_seed(seed, 310_002),
        _paper_seed(seed, 310_003),
        adaptation_stream,
        _paper_seed(adaptation_stream, 101),
        _paper_seed(adaptation_stream, 211),
        _paper_seed(adaptation_stream, 307),
        _paper_seed(seed, 900_001),
        _paper_seed(seed, 1_100_001),
        _paper_seed(seed, 1_300_001),
        _paper_seed(seed, 1_300_002),
        _paper_seed(seed, 1_400_001),
        _paper_seed(seed, 1_400_002),
    }
    return streams


def _validate_fresh_streams(seeds: tuple[int, ...]) -> None:
    fresh = {_paper_seed(seed, FRESH_EVALUATION_STREAM) for seed in seeds}
    if len(fresh) != len(seeds):
        raise RuntimeError("fresh evaluation seeds collide with each other")
    registered = set().union(*(_registered_streams_for_seed(seed) for seed in seeds))
    collisions = sorted(fresh & registered)
    if collisions:
        raise RuntimeError(f"fresh evaluation streams collide with registered streams: {collisions}")


def _validate_input_configs(config: ExperimentConfig, phase0_dir: Path, paper_dir: Path) -> None:
    phase0 = ExperimentConfig.from_yaml(phase0_dir / "config.yaml")
    paper = ExperimentConfig.from_yaml(paper_dir / "config.yaml")
    requested = _normalized_input_config(config)
    for label, candidate in (("phase0", phase0), ("paper", paper)):
        comparable = _normalized_input_config(candidate)
        if comparable != requested:
            raise RuntimeError(f"{label} input configuration differs from the requested evaluator config")


def _normalized_input_config(config: ExperimentConfig) -> dict[str, Any]:
    values = config.to_dict()
    values["output_dir"] = "<frozen-input-output>"
    values["paper"]["save_mechanism_diagonal"] = False
    return values


def _validate_resume(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    execution: dict[str, Any],
    source_hash: str,
    config_hash: str,
    rollouts: int,
) -> set[int]:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"resume output does not exist: {output_dir}")
    metadata = _read_json(output_dir / "study_metadata.json")
    stored_config = yaml.safe_load((output_dir / "config.yaml").read_text())
    if not isinstance(stored_config, dict) or canonical_config_sha256(stored_config) != config_hash:
        raise RuntimeError("resume config differs from the requested config")
    if metadata.get("source_tree_sha256") != source_hash:
        raise RuntimeError("resume source hash differs from the active source")
    if metadata.get("execution") != execution:
        raise RuntimeError("resume execution or input fingerprints differ")

    completed: set[int] = set()
    requested = set(config.seeds)
    for path in output_dir.iterdir():
        if path.name.startswith(".seed_"):
            raise RuntimeError(f"partial seed directory blocks resume: {path}")
        match = SEED_DIRECTORY.fullmatch(path.name)
        if match is None:
            continue
        seed = int(match.group(1))
        if seed not in requested:
            raise RuntimeError(f"unexpected seed directory blocks resume: {path}")
        validate_output_seed(
            path,
            seed,
            horizon=config.horizon,
            rollouts=rollouts,
            expected_source_hash=source_hash,
            expected_config_hash=config_hash,
        )
        completed.add(seed)
    return completed


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _update_file_digest(digest: Any, path: Path, root: Path) -> None:
    relative = path.relative_to(root).as_posix().encode("utf-8")
    content = path.read_bytes()
    digest.update(len(relative).to_bytes(4, "big"))
    digest.update(relative)
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)


if __name__ == "__main__":
    main()
