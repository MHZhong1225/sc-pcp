from __future__ import annotations

import importlib.util
from concurrent.futures import Future
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scpcp.config import ExperimentConfig
from scpcp.experiment import SeedResult, _paper_seed
from scpcp.phase0c_joint_search import SearchState
from scpcp.phase0c_study import _state_sha256


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "phase0c_joint_search.yaml"
SCENARIOS = ("standard", "tail_shift")
INITIAL_METHODS = ("current_profiled", "greedy", "joint_B", "joint_2B")
START_NAMES = ("profiled", "greedy", "upper_endpoint")
RECORD_COLUMNS = (
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


def _load_runner():
    path = ROOT / "scripts" / "run_phase0c_joint_search.py"
    spec = importlib.util.spec_from_file_location("run_phase0c_joint_search", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compact(values: list[float] | list[int]) -> str:
    return json.dumps(values, separators=(",", ":"))


def _coordinate_trace(
    start_names: tuple[str, ...] | list[str],
    first_pair: int,
    last_pair: int,
    *,
    commit_last_pair: bool = True,
) -> list[dict[str, object]]:
    widths = {name: 1.5 + index / 10.0 for index, name in enumerate(start_names)}
    trace: list[dict[str, object]] = []
    for sweep_pair in range(first_pair, last_pair + 1):
        for start_name in start_names:
            for direction, stages in (
                ("forward", range(12)),
                ("reverse", range(11, -1, -1)),
            ):
                for stage in stages:
                    before = widths[start_name]
                    commit_name = (
                        "profiled"
                        if first_pair >= 5 or sweep_pair <= 2
                        else "greedy"
                    )
                    committed = (
                        start_name == commit_name
                        and direction == "forward"
                        and stage == 0
                        and (commit_last_pair or sweep_pair < last_pair)
                    )
                    improvement = 0.1 if commit_name == "greedy" else 0.01
                    proposed = before - improvement if committed else before + 0.01
                    after = proposed if committed else before
                    widths[start_name] = after
                    trace.append(
                        {
                            "start_name": start_name,
                            "sweep_pair": sweep_pair,
                            "direction": direction,
                            "stage": stage,
                            "feasible_count": 2,
                            "proposed_grid_index": 25,
                            "before_micro_width": before,
                            "proposed_micro_width": proposed,
                            "committed": committed,
                            "after_micro_width": after,
                        }
                    )
    return trace


def _search_state(name: str, index: int | None) -> SearchState:
    radius = 1.5 if index is None else 1.0 + index / 100.0
    normalized_width = {
        "profiled": 1.48,
        "greedy": 1.4,
        "upper_endpoint": 1.7,
    }[name]
    return SearchState(
        start_name=name,
        radii=torch.full((12,), radius, dtype=torch.float32),
        stage_grid_indices=(index,) * 12,
        coverage=torch.full((12,), 0.91, dtype=torch.float32),
        normalized_width=torch.full((12,), normalized_width, dtype=torch.float32),
        completed_sweep_pairs=4,
        converged_at_pair=None,
    )


def _initial_result(seed: int, device: str, *, partial_active: bool = False) -> SeedResult:
    records: list[dict[str, object]] = []
    surfaces: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, object] = {}
    stage_grids = torch.stack([torch.linspace(1.0, 2.0, 101) for _ in range(12)])
    profiled_schedules = stage_grids.T.contiguous()
    methods = {
        "current_profiled": (50, "profiled", "REFERENCE", 0),
        "greedy": (75, "greedy", "REFERENCE", 0),
        "joint_B": (25, "profiled", "B", 2),
        "joint_2B": (75, "greedy", "2B", 4),
    }
    for scenario_index, scenario in enumerate(SCENARIOS):
        tuning_stream = _paper_seed(seed, 1_300_001 + scenario_index)
        evaluation_stream = _paper_seed(seed, 1_400_001 + scenario_index)
        active_names = list(START_NAMES[:2] if partial_active else START_NAMES)
        pair2_trace = _coordinate_trace(active_names, 1, 2)
        pair4_trace = _coordinate_trace(active_names, 1, 4)
        trace_widths = {
            method: next(
                step["after_micro_width"]
                for step in reversed(trace)
                if step["start_name"] == chosen
            )
            for method, trace, chosen in (
                ("joint_B", pair2_trace, "profiled"),
                ("joint_2B", pair4_trace, "greedy"),
            )
        }
        surfaces.update(
            {
                f"{scenario}_profile": torch.ones(12),
                f"{scenario}_profiled_scale_grid": torch.linspace(1.0, 2.0, 101),
                f"{scenario}_profiled_schedules": profiled_schedules,
                f"{scenario}_stage_grids": stage_grids,
            }
        )
        for method_id, (index, chosen, budget, sweep_pairs) in methods.items():
            schedule = stage_grids[:, index]
            tuning_coverage = torch.full((12,), 0.91)
            tuning_width = torch.full(
                (12,), float(trace_widths.get(method_id, float(schedule.mean())))
            )
            final_coverage = torch.full((12,), 0.93)
            final_lcb = torch.full((12,), 0.925)
            final_width = schedule * 2.0
            selected_indices = [index] if method_id == "current_profiled" else [index] * 12
            records.append(
                {
                    "schema_version": "phase0c_seed_v1",
                    "seed": seed,
                    "scenario": scenario,
                    "method_id": method_id,
                    "analysis_role": (
                        "reference" if method_id in INITIAL_METHODS[:2] else "joint_search"
                    ),
                    "budget_id": budget,
                    "sweep_pairs": sweep_pairs,
                    "selection_status": "SELECTED",
                    "selection_available": True,
                    "tuning_joint_feasible": True,
                    "failure_reason": "",
                    "chosen_initialization": chosen,
                    "selected_endpoint_stage_count": 0,
                    "selected_stage_grid_indices_json": _compact(selected_indices),
                    "q_by_time_json": _compact(schedule.tolist()),
                    "tuning_coverage_json": _compact(tuning_coverage.tolist()),
                    "tuning_stage_width_json": _compact(tuning_width.tolist()),
                    "tuning_micro_width": float(tuning_width.mean()),
                    "final_coverage_json": _compact(final_coverage.tolist()),
                    "final_wilson_lcb_json": _compact(final_lcb.tolist()),
                    "final_stage_width_json": _compact(final_width.tolist()),
                    "micro_normalized_width": float(final_width.mean()),
                    "patient_normalized_width": float(final_width.mean()),
                    "tuning_stream_id": tuning_stream,
                    "evaluation_stream_id": evaluation_stream,
                    "n_tuning_rollouts": 5_000,
                    "n_evaluation_rollouts": 50_000,
                    "schedule_evaluations": 0,
                    "committed_updates": 0,
                    "converged_at_pair": None,
                    "wall_time_seconds": (
                        0.0 if method_id in INITIAL_METHODS[:2] else 1.0
                    ),
                }
            )
            prefix = f"{scenario}_{method_id}"
            surfaces.update(
                {
                    f"{prefix}_schedule": schedule,
                    f"{prefix}_tuning_coverage": tuning_coverage,
                    f"{prefix}_tuning_stage_width": tuning_width,
                    f"{prefix}_final_coverage": final_coverage,
                    f"{prefix}_final_wilson_lcb": final_lcb,
                    f"{prefix}_final_stage_width": final_width,
                }
            )

        states = (
            _search_state("profiled", None),
            _search_state("greedy", 75),
            _search_state("upper_endpoint", 100),
        )
        active_states = states[:2] if partial_active else states
        for state in active_states:
            prefix = f"{scenario}_pair4_{state.start_name}"
            surfaces.update(
                {
                    f"{prefix}_radii": state.radii,
                    f"{prefix}_stage_grid_indices": torch.tensor(
                        [-1 if value is None else value for value in state.stage_grid_indices],
                        dtype=torch.int64,
                    ),
                    f"{prefix}_coverage": state.coverage,
                    f"{prefix}_normalized_width": state.normalized_width,
                    f"{prefix}_completed_sweep_pairs": torch.tensor(4),
                    f"{prefix}_converged_at_pair": torch.tensor(-1),
                }
            )
        active_names = [state.start_name for state in active_states]
        for record in records:
            if record["scenario"] != scenario:
                continue
            trace = {
                "joint_B": pair2_trace,
                "joint_2B": pair4_trace,
            }.get(record["method_id"])
            if trace is not None:
                record["schedule_evaluations"] = 101 * len(trace)
                record["committed_updates"] = sum(
                    bool(step["committed"]) for step in trace
                )
        eligible = not partial_active
        surfaces[f"{scenario}_active_start_names"] = torch.tensor(
            list(range(len(active_names))), dtype=torch.int64
        )
        surfaces[f"{scenario}_extension_eligible"] = torch.tensor(eligible)
        diagnostics[scenario] = {
            "tuning_stream_id": tuning_stream,
            "evaluation_stream_id": evaluation_stream,
            "start_order": list(START_NAMES),
            "active_start_names": active_names,
            "extension_eligible": eligible,
            "pair4_state_sha256": [_state_sha256(state) for state in active_states],
            "greedy_partial_indices": [],
            "search_status": "SELECTED",
            "checkpoints": {
                "2": {
                    "requested_sweep_pairs": 2,
                    "executed_sweep_pairs": 2,
                    "best_start_name": "profiled",
                    "schedule_evaluations": 101 * len(pair2_trace),
                    "committed_updates": sum(
                        bool(step["committed"]) for step in pair2_trace
                    ),
                    "trace": pair2_trace,
                },
                "4": {
                    "requested_sweep_pairs": 4,
                    "executed_sweep_pairs": 4,
                    "best_start_name": "greedy",
                    "schedule_evaluations": 101 * len(pair4_trace),
                    "committed_updates": sum(
                        bool(step["committed"]) for step in pair4_trace
                    ),
                    "trace": pair4_trace,
                },
            },
        }
    return SeedResult(
        seed=seed,
        device=device,
        records=records,
        surfaces=surfaces,
        diagnostics=diagnostics,
    )


def _extension_result(seed: int, device: str) -> SeedResult:
    records: list[dict[str, object]] = []
    surfaces: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, object] = {}
    stage_grids = torch.stack([torch.linspace(1.0, 2.0, 101) for _ in range(12)])
    for scenario_index, scenario in enumerate(SCENARIOS):
        tuning_stream = _paper_seed(seed, 1_300_001 + scenario_index)
        evaluation_stream = _paper_seed(seed, 1_400_001 + scenario_index)
        trace = _coordinate_trace(START_NAMES, 5, 8)
        schedule = stage_grids[:, 25]
        coverage = torch.full((12,), 0.91)
        tuning_width = torch.full((12,), 1.46)
        final_coverage = torch.full((12,), 0.93)
        final_lcb = torch.full((12,), 0.925)
        final_width = schedule * 2.0
        records.append(
            {
                "schema_version": "phase0c_seed_v1",
                "seed": seed,
                "scenario": scenario,
                "method_id": "joint_8SP",
                "analysis_role": "joint_search",
                "budget_id": "8SP",
                "sweep_pairs": 8,
                "selection_status": "SELECTED",
                "selection_available": True,
                "tuning_joint_feasible": True,
                "failure_reason": "",
                "chosen_initialization": "profiled",
                "selected_endpoint_stage_count": 0,
                "selected_stage_grid_indices_json": _compact([25] * 12),
                "q_by_time_json": _compact(schedule.tolist()),
                "tuning_coverage_json": _compact(coverage.tolist()),
                "tuning_stage_width_json": _compact(tuning_width.tolist()),
                "tuning_micro_width": float(tuning_width.mean()),
                "final_coverage_json": _compact(final_coverage.tolist()),
                "final_wilson_lcb_json": _compact(final_lcb.tolist()),
                "final_stage_width_json": _compact(final_width.tolist()),
                "micro_normalized_width": float(final_width.mean()),
                "patient_normalized_width": float(final_width.mean()),
                "tuning_stream_id": tuning_stream,
                "evaluation_stream_id": evaluation_stream,
                "n_tuning_rollouts": 5_000,
                "n_evaluation_rollouts": 50_000,
                "schedule_evaluations": 101 * len(trace),
                "committed_updates": sum(
                    bool(step["committed"]) for step in trace
                ),
                "converged_at_pair": None,
                "wall_time_seconds": 1.0,
            }
        )
        prefix = f"{scenario}_joint_8SP"
        surfaces[f"{scenario}_stage_grids"] = stage_grids
        surfaces.update(
            {
                f"{prefix}_schedule": schedule,
                f"{prefix}_tuning_coverage": coverage,
                f"{prefix}_tuning_stage_width": tuning_width,
                f"{prefix}_final_coverage": final_coverage,
                f"{prefix}_final_wilson_lcb": final_lcb,
                f"{prefix}_final_stage_width": final_width,
            }
        )
        diagnostics[scenario] = {
            "tuning_stream_id": tuning_stream,
            "evaluation_stream_id": evaluation_stream,
            "search_status": "SELECTED",
            "continuation_status": "SELECTED",
            "fresh_evaluation_completed": True,
            "wall_time_phase": None,
            "checkpoint": {
                "requested_sweep_pairs": 8,
                "executed_sweep_pairs": 8,
                "best_start_name": "profiled",
                "schedule_evaluations": 101 * len(trace),
                "committed_updates": sum(
                    bool(step["committed"]) for step in trace
                ),
                "trace": trace,
            },
        }
    return SeedResult(seed, device, records, surfaces, diagnostics)


def _pair2_only_result(seed: int, device: str) -> SeedResult:
    result = _initial_result(seed, device)
    records = [dict(row) for row in result.records]
    surfaces = dict(result.surfaces)
    diagnostics = json.loads(json.dumps(result.diagnostics))
    for scenario in SCENARIOS:
        row = next(
            row
            for row in records
            if row["scenario"] == scenario and row["method_id"] == "joint_2B"
        )
        row.update(
            {
                "selection_status": "WALL_TIME_CAP",
                "selection_available": False,
                "tuning_joint_feasible": False,
                "failure_reason": "WALL_TIME_CAP",
                "chosen_initialization": "",
                "selected_stage_grid_indices_json": "[]",
                "q_by_time_json": "[]",
                "tuning_coverage_json": "[]",
                "tuning_stage_width_json": "[]",
                "tuning_micro_width": float("nan"),
                "final_coverage_json": "[]",
                "final_wilson_lcb_json": "[]",
                "final_stage_width_json": "[]",
                "micro_normalized_width": float("nan"),
                "patient_normalized_width": float("nan"),
                "n_evaluation_rollouts": 0,
                "schedule_evaluations": 0,
                "committed_updates": 0,
            }
        )
        for suffix in (
            "schedule",
            "tuning_coverage",
            "tuning_stage_width",
            "final_coverage",
            "final_wilson_lcb",
            "final_stage_width",
        ):
            surfaces[f"{scenario}_joint_2B_{suffix}"] = torch.empty(0)
        for start_name in START_NAMES:
            for suffix in (
                "radii",
                "stage_grid_indices",
                "coverage",
                "normalized_width",
                "completed_sweep_pairs",
                "converged_at_pair",
            ):
                surfaces.pop(f"{scenario}_pair4_{start_name}_{suffix}")
        surfaces[f"{scenario}_active_start_names"] = torch.tensor([0, 1, 2])
        surfaces[f"{scenario}_extension_eligible"] = torch.tensor(False)
        diagnostics[scenario]["active_start_names"] = list(START_NAMES)
        diagnostics[scenario]["extension_eligible"] = False
        diagnostics[scenario]["pair4_state_sha256"] = []
        diagnostics[scenario]["search_status"] = "WALL_TIME_CAP"
        diagnostics[scenario]["checkpoints"].pop("4")
    return SeedResult(seed, device, records, surfaces, diagnostics)


class _InlineExecutor:
    instances: list["_InlineExecutor"] = []

    def __init__(self, *, max_workers: int, mp_context: object) -> None:
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.submissions: list[tuple[object, tuple[object, ...]]] = []
        self.instances.append(self)

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def submit(self, function: object, *args: object) -> Future[object]:
        self.submissions.append((function, args))
        future: Future[object] = Future()
        try:
            future.set_result(function(*args))  # type: ignore[operator]
        except BaseException as error:
            future.set_exception(error)
        return future


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_npz(
    path: Path,
    *,
    updates: dict[str, np.ndarray] | None = None,
    remove: tuple[str, ...] = (),
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays.update(updates or {})
    for name in remove:
        arrays.pop(name)
    np.savez_compressed(path, **arrays)


def _write_smoke_manifest(runner: object, path: Path, config: ExperimentConfig) -> None:
    payload = {
        "protocol": "phase0c_smoke_v1",
        "seed": 9999,
        "max_sweep_pairs": 4,
        "elapsed_seconds": 200.0,
        "max_memory_allocated_bytes": 123,
        "max_memory_reserved_bytes": 456,
        "recommended_max_seed_wall_seconds": 300,
        "source_tree_sha256": runner.source_tree_sha256(),
        "experiment_tree_sha256": runner.experiment_tree_sha256(),
        "config_sha256": runner.canonical_config_sha256(config.to_dict()),
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _run_initial_fixture(
    runner: object,
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    *,
    seeds: tuple[int, ...] = (10_000,),
    partial_seed: int | None = None,
) -> ExperimentConfig:
    base = ExperimentConfig.from_yaml(CONFIG_PATH)
    config = base.with_overrides(seeds=seeds, devices=("cpu",), output_dir=output)
    smoke = output.parent / f"{output.name}-smoke.json"
    _write_smoke_manifest(runner, smoke, config)
    _InlineExecutor.instances = []
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(
        runner,
        "run_phase0c_seed",
        lambda _config, *, seed, device, **_kwargs: _initial_result(
            seed, device, partial_active=seed == partial_seed
        ),
    )
    runner.run_config(
        config,
        output,
        mode="initial",
        workers_per_device=1,
        candidate_chunk_size=16,
        resume=False,
        smoke_manifest=smoke,
    )
    return config


def _spawn_identity(value: int) -> tuple[int, int]:
    return os.getpid(), value


def _spawn_failure(_value: int) -> None:
    raise RuntimeError("spawn probe failure")


def _state_hash_manifest(parent: Path, seeds: tuple[int, ...]) -> tuple[int, int, str]:
    entries: list[dict[str, object]] = []
    eligible_count = 0
    for seed in seeds:
        metadata = json.loads((parent / f"seed_{seed:05d}" / "metadata.json").read_text())
        for scenario in SCENARIOS:
            diagnostics = metadata["diagnostics"][scenario]
            eligible_count += int(diagnostics["extension_eligible"])
            for start_name, state_hash in zip(
                diagnostics["active_start_names"],
                diagnostics["pair4_state_sha256"],
                strict=True,
            ):
                entries.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "start_name": start_name,
                        "sha256": state_hash,
                    }
                )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return eligible_count, len(entries), hashlib.sha256(payload).hexdigest()


def _write_checkpoint_analysis(
    parent: Path,
    seeds: tuple[int, ...],
    *,
    decision_mutation: tuple[str, object] | None = None,
    manifest_mutation: tuple[str, object] | None = None,
    corrupt_decision_fact: bool = False,
) -> Path:
    analysis = parent / "checkpoint_analysis"
    analysis.mkdir()
    parent_manifest_sha = _sha256(parent / "study_manifest.json")
    study_manifest = json.loads((parent / "study_manifest.json").read_text())
    eligible_count, state_count, state_manifest_sha = _state_hash_manifest(parent, seeds)
    decision = {
        "protocol": "phase0c_joint_search_summary_v1",
        "analysis_phase": "initial",
        "decision": "EXTENSION_8SP_REQUIRED",
        "parent_study_manifest_sha256": parent_manifest_sha,
        "ordered_seeds": list(seeds),
        "source_tree_sha256": study_manifest["source_tree_sha256"],
        "experiment_tree_sha256": study_manifest["experiment_tree_sha256"],
        "config_sha256": study_manifest["config_sha256"],
        "extension_eligibility": {
            "all_eligible": eligible_count == 80,
            "eligible_scenario_seed_count": eligible_count,
            "required_scenario_seed_count": 80,
            "canonical_state_hash_count": state_count,
            "state_hash_manifest_sha256": state_manifest_sha,
        },
    }
    if decision_mutation is not None:
        field, value = decision_mutation
        if field.startswith("extension_eligibility."):
            decision["extension_eligibility"][field.split(".", 1)[1]] = value
        else:
            decision[field] = value
    decision_path = analysis / "phase0c_decision.json"
    decision_path.write_text(json.dumps(decision, sort_keys=True, indent=2) + "\n")
    for name, content in {
        "phase0c_summary.csv": b"metric,value\nfixture,1\n",
        "phase0c_summary.md": b"# Fixture\n",
        "phase0c_joint_search.pdf": b"%PDF-fixture\n",
        "phase0c_joint_search.svg": b"<svg/>\n",
        "phase0c_joint_search.png": b"PNG-fixture\n",
    }.items():
        (analysis / name).write_bytes(content)
    payload_names = {
        "phase0c_decision.json",
        "phase0c_summary.csv",
        "phase0c_summary.md",
        "phase0c_joint_search.pdf",
        "phase0c_joint_search.svg",
        "phase0c_joint_search.png",
    }
    summary_manifest = {
        "protocol": "phase0c_joint_search_summary_manifest_v1",
        "status": "complete",
        "analysis_phase": "initial",
        "decision": decision["decision"],
        "parent_study_manifest_sha256": parent_manifest_sha,
        "files": {name: {"bytes": (analysis / name).stat().st_size, "sha256": _sha256(analysis / name)} for name in sorted(payload_names)},
    }
    if manifest_mutation is not None:
        summary_manifest[manifest_mutation[0]] = manifest_mutation[1]
    if corrupt_decision_fact:
        summary_manifest["files"]["phase0c_decision.json"]["sha256"] = "0" * 64
    (analysis / "phase0c_summary_manifest.json").write_text(
        json.dumps(summary_manifest, sort_keys=True, indent=2) + "\n"
    )
    return decision_path


def test_frozen_config_is_the_phase0c_development_protocol() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    config = ExperimentConfig.from_yaml(CONFIG_PATH)

    assert raw["seeds"] == {"start": 10_000, "stop": 10_040}
    assert config.seeds == tuple(range(10_000, 10_040))
    assert config.devices == ("cuda:0", "cuda:1")
    assert config.output_dir == Path("results/work/phase0c_joint_search")
    assert config.data.dataset == "synthetic"
    assert config.horizon == 12
    assert config.q_grid_size == 101
    assert config.samples.oracle_surface_rollouts == 5_000
    assert config.samples.oracle_rollouts == 50_000


@pytest.mark.parametrize(
    ("text", "want"),
    [
        (None, (10_000, 10_001)),
        ("10000:10003", (10_000, 10_001, 10_002)),
        ("10000,10002", (10_000, 10_002)),
    ],
)
def test_parse_seeds_preserves_exact_order(text: str | None, want: tuple[int, ...]) -> None:
    runner = _load_runner()
    assert runner.parse_seeds(text, (10_000, 10_001)) == want


@pytest.mark.parametrize("text", ["", "-1", "1,1", "1:", ":2", "2:1", "1:2:3"])
def test_parse_seeds_rejects_malformed_banks(text: str) -> None:
    runner = _load_runner()
    with pytest.raises(ValueError):
        runner.parse_seeds(text, (10_000,))


@pytest.mark.parametrize("elapsed", [0.0, -1.0, math.inf, -math.inf, math.nan, True])
def test_smoke_wall_cap_rejects_nonpositive_or_nonfinite_elapsed(elapsed: float) -> None:
    runner = _load_runner()
    with pytest.raises(ValueError, match="finite and positive"):
        runner.calibrate_wall_cap(elapsed)


@pytest.mark.parametrize(
    ("elapsed", "want"),
    [(1.0, 300), (200.0, 300), (200.000_001, 600), (400.0, 600), (600.0, 900)],
)
def test_smoke_wall_cap_uses_preregistered_rounding(elapsed: float, want: int) -> None:
    runner = _load_runner()
    assert runner.calibrate_wall_cap(elapsed) == want


def test_config_hash_excludes_only_runtime_fields() -> None:
    runner = _load_runner()
    first = {
        "horizon": 12,
        "data": {"dataset": "synthetic"},
        "seeds": [9999],
        "devices": ["cuda:0"],
        "output_dir": "smoke",
    }
    second = {
        "output_dir": "formal",
        "devices": ["cuda:0", "cuda:1"],
        "seeds": list(range(10_000, 10_040)),
        "data": {"dataset": "synthetic"},
        "horizon": 12,
    }
    changed_science = {**second, "horizon": 13}

    assert runner.canonical_config_sha256(first) == runner.canonical_config_sha256(second)
    assert runner.canonical_config_sha256(first) != runner.canonical_config_sha256(
        changed_science
    )
    assert runner.runtime_config_sha256(first) != runner.runtime_config_sha256(second)


def test_cli_has_only_the_three_registered_modes_and_fixed_defaults() -> None:
    runner = _load_runner()
    parser = runner.build_parser()
    initial = parser.parse_args(["initial", "--smoke-manifest", "smoke.json"])
    smoke = parser.parse_args(["smoke"])
    extension = parser.parse_args(
        [
            "extension-8sp",
            "--parent-dir",
            "parent",
            "--decision-json",
            "decision.json",
            "--output-dir",
            "extension",
        ]
    )

    assert (initial.mode, smoke.mode, extension.mode) == (
        "initial",
        "smoke",
        "extension-8sp",
    )
    assert initial.config == CONFIG_PATH
    assert initial.workers_per_device == 1
    assert initial.candidate_chunk_size == 16
    assert initial.resume is False
    with pytest.raises(SystemExit):
        parser.parse_args(["unknown"])


@pytest.mark.parametrize(
    "argv",
    [
        ["initial", "--smoke-manifest", "x", "--workers-per-device", "0"],
        ["initial", "--smoke-manifest", "x", "--candidate-chunk-size", "0"],
    ],
)
def test_cli_rejects_nonpositive_execution_counts(argv: list[str]) -> None:
    runner = _load_runner()
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(argv)


def test_main_freezes_initial_seed_bank_and_dispatches_resolved_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    calls: list[tuple[ExperimentConfig, Path, dict[str, object]]] = []
    monkeypatch.setattr(
        runner,
        "resolve_devices",
        lambda value: tuple(value.split(",")) if isinstance(value, str) else tuple(value),
    )
    monkeypatch.setattr(
        runner,
        "run_config",
        lambda config, output, **kwargs: calls.append((config, output, kwargs)),
    )
    output = tmp_path / "initial"
    runner.main(
        [
            "initial",
            "--devices",
            "cuda:0,cuda:1",
            "--smoke-manifest",
            str(tmp_path / "smoke.json"),
            "--output-dir",
            str(output),
        ]
    )
    config, dispatched_output, kwargs = calls.pop()
    assert config.seeds == tuple(range(10_000, 10_040))
    assert config.devices == ("cuda:0", "cuda:1")
    assert config.output_dir == output == dispatched_output
    assert kwargs["mode"] == "initial"
    assert kwargs["smoke_manifest"] == tmp_path / "smoke.json"

    with pytest.raises(SystemExit):
        runner.main(
            [
                "initial",
                "--seeds",
                "10000:10039",
                "--smoke-manifest",
                str(tmp_path / "smoke.json"),
            ]
        )
    assert calls == []


def test_real_spawn_executor_reuses_one_child_and_propagates_failures() -> None:
    runner = _load_runner()
    results = runner._execute_jobs(
        ("cpu",),
        ((0, (11,)), (0, (12,))),
        worker_function=_spawn_identity,
    )
    assert [value for _, value in results] == [11, 12]
    assert len({pid for pid, _ in results}) == 1

    with pytest.raises(RuntimeError, match="spawn probe failure"):
        runner._execute_jobs(
            ("cpu",),
            ((0, (13,)),),
            worker_function=_spawn_failure,
        )


def test_cuda_worker_pins_runs_validates_and_cleans_in_one_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    events: list[str] = []
    published: list[SeedResult] = []
    clock = [10.0]

    class DeviceContext:
        def __init__(self, device: torch.device) -> None:
            self.index = device.index

        def __enter__(self) -> None:
            events.append(f"enter_device:{self.index}")

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeCuda:
        def set_device(self, device: torch.device) -> None:
            events.append(f"set_device:{device.index}")

        def device(self, device: torch.device) -> DeviceContext:
            return DeviceContext(device)

        def empty_cache(self) -> None:
            events.append("empty_cache:0")

        def reset_peak_memory_stats(self, _device: torch.device) -> None:
            return None

        def max_memory_allocated(self, _device: torch.device) -> int:
            return 10

        def max_memory_reserved(self, _device: torch.device) -> int:
            return 20

    config = ExperimentConfig(seeds=(10_000,), devices=("cuda:0",), output_dir=tmp_path)
    provenance = {"execution_sha256": "0" * 64}
    monkeypatch.setattr(runner.torch, "cuda", FakeCuda())
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])

    def run_seed(*_args: object, seed: int, device: str, **_kwargs: object) -> SeedResult:
        events.append(f"run_seed:{seed}")
        clock[0] = 12.5
        return _initial_result(seed, device)

    def write_result(result: SeedResult, *_args: object, **_kwargs: object) -> Path:
        published.append(result)
        clock[0] = 100.0
        return tmp_path / "seed_10000"

    def validate_result(_path: Path, seed: int, **_kwargs: object) -> None:
        events.append(f"validate_seed:{seed}")
        clock[0] = 200.0

    monkeypatch.setattr(
        runner,
        "run_phase0c_seed",
        run_seed,
    )
    monkeypatch.setattr(runner, "write_seed_result", write_result)
    monkeypatch.setattr(runner, "validate_seed_artifact", validate_result)

    returned = runner._run_and_write(
        config,
        10_000,
        "cuda:0",
        tmp_path,
        16,
        300.0,
        "initial",
        provenance,
        None,
    )

    assert events == [
        "set_device:0",
        "enter_device:0",
        "run_seed:10000",
        "validate_seed:10000",
        "empty_cache:0",
    ]
    measurement = published[0].diagnostics["runner_measurement"]
    assert measurement == {
        "protocol": "phase0c_runner_measurement_v1",
        "elapsed_seconds": 2.5,
        "max_memory_allocated_bytes": 10,
        "max_memory_reserved_bytes": 20,
    }
    assert returned == {
        "seed_dir": str(tmp_path / "seed_10000"),
        "elapsed_seconds": 2.5,
        "max_memory_allocated_bytes": 10,
        "max_memory_reserved_bytes": 20,
    }


def test_smoke_manifest_is_strict_and_cap_is_derived_from_elapsed(tmp_path: Path) -> None:
    runner = _load_runner()
    config = ExperimentConfig.from_yaml(CONFIG_PATH)
    manifest = tmp_path / "smoke_manifest.json"
    _write_smoke_manifest(runner, manifest, config)

    assert runner.validate_smoke_manifest(
        manifest,
        source_hash=runner.source_tree_sha256(),
        experiment_hash=runner.experiment_tree_sha256(),
        config_hash=runner.canonical_config_sha256(config.to_dict()),
    ) == 300

    payload = json.loads(manifest.read_text())
    payload["unexpected"] = True
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="exact fields"):
        runner.validate_smoke_manifest(
            manifest,
            source_hash=runner.source_tree_sha256(),
            experiment_hash=runner.experiment_tree_sha256(),
            config_hash=runner.canonical_config_sha256(config.to_dict()),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("seed", 10_000, "seed"),
        ("max_sweep_pairs", 8, "max_sweep_pairs"),
        ("source_tree_sha256", "0" * 64, "source_tree_sha256"),
        ("experiment_tree_sha256", "1" * 64, "experiment_tree_sha256"),
        ("config_sha256", "2" * 64, "config_sha256"),
        ("recommended_max_seed_wall_seconds", 600, "recommended"),
    ],
)
def test_initial_rejects_mutated_smoke_before_creating_output(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    runner = _load_runner()
    output = tmp_path / "formal"
    base = ExperimentConfig.from_yaml(CONFIG_PATH)
    config = base.with_overrides(devices=("cpu",), output_dir=output)
    smoke = tmp_path / "smoke.json"
    _write_smoke_manifest(runner, smoke, config)
    payload = json.loads(smoke.read_text())
    payload[field] = value
    smoke.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match=match):
        runner.run_config(
            config,
            output,
            mode="initial",
            workers_per_device=1,
            candidate_chunk_size=16,
            resume=False,
            smoke_manifest=smoke,
        )
    assert not output.exists()


def test_fresh_initial_run_publishes_deep_seed_manifest_then_root_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "initial"
    config = _run_initial_fixture(runner, monkeypatch, output)

    assert runner.validate_seed_artifact(
        output / "seed_10000",
        10_000,
        mode="initial",
    ) == output / "seed_10000"
    assert (output / "COMPLETE").read_text() == "complete\n"
    manifest = json.loads((output / "study_manifest.json").read_text())
    metadata = json.loads((output / "study_metadata.json").read_text())
    execution = metadata["execution"]
    assert manifest["protocol"] == "phase0c_study_manifest_v1"
    assert manifest["status"] == "complete"
    assert manifest["study_kind"] == "initial"
    assert manifest["ordered_seeds"] == [10_000]
    for key in (
        "source_tree_sha256",
        "experiment_tree_sha256",
        "config_sha256",
        "runtime_config_sha256",
        "execution_sha256",
    ):
        assert manifest[key] == execution.get(key, metadata.get(key))
    expected_files = {
        "config.yaml",
        "study_metadata.json",
        "seed_10000/COMPLETE",
        "seed_10000/records.csv",
        "seed_10000/surfaces.npz",
        "seed_10000/metadata.json",
    }
    assert set(manifest["files"]) == expected_files
    for relative, fact in manifest["files"].items():
        path = output / relative
        assert fact == {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    runner.validate_study_manifest(output, expected_kind="initial")
    assert config.seeds == (10_000,)


@pytest.mark.parametrize(
    ("mutation", "value", "match"),
    [
        ("COMPLETE", b"complete", "root COMPLETE"),
        ("extra", True, "study status"),
        ("status", "running", "study status"),
        ("expected_seeds", [9999], "study status"),
        ("expected_seeds", [10_000.0], "study status"),
        ("completed_seeds", [], "study status"),
        ("completed_seeds", [10_000.0], "study status"),
        ("missing_seeds", [10_000], "study status"),
        ("updated_at_utc", 123, "study status"),
        ("error", "corrupt", "study status"),
    ],
)
def test_study_validation_requires_exact_root_completion_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    value: object,
    match: str,
) -> None:
    runner = _load_runner()
    output = tmp_path / mutation
    _run_initial_fixture(runner, monkeypatch, output)
    if mutation == "COMPLETE":
        (output / "COMPLETE").write_bytes(value)  # type: ignore[arg-type]
    else:
        status_path = output / "study_status.json"
        status = json.loads(status_path.read_text())
        status[mutation] = value
        status_path.write_text(json.dumps(status))

    with pytest.raises(RuntimeError, match=match):
        runner.validate_study_manifest(output, expected_kind="initial")


def test_study_manifest_file_bytes_rejects_equal_float(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "float-file-bytes"
    _run_initial_fixture(runner, monkeypatch, output)
    manifest_path = output / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fact = next(iter(manifest["files"].values()))
    fact["bytes"] = float(fact["bytes"])
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="bytes"):
        runner.validate_study_manifest(output, expected_kind="initial")


def test_resume_and_authorization_share_exact_root_completion_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    config = _run_initial_fixture(runner, monkeypatch, parent)
    status_path = parent / "study_status.json"
    status = json.loads(status_path.read_text())
    status["completed_seeds"] = []
    status_path.write_text(json.dumps(status))
    before = {
        path.relative_to(parent): path.read_bytes()
        for path in parent.rglob("*")
        if path.is_file()
    }
    smoke = tmp_path / "parent-smoke.json"

    with pytest.raises(RuntimeError, match="study status"):
        runner.run_config(
            config,
            parent,
            mode="initial",
            workers_per_device=1,
            candidate_chunk_size=16,
            resume=True,
            smoke_manifest=smoke,
        )
    assert {
        path.relative_to(parent): path.read_bytes()
        for path in parent.rglob("*")
        if path.is_file()
    } == before

    decision = parent / "checkpoint_analysis" / "phase0c_decision.json"
    with pytest.raises(RuntimeError, match="study status"):
        runner.authorize_extension(
            parent,
            decision,
            config=config,
            source_hash=runner.source_tree_sha256(),
            experiment_hash=runner.experiment_tree_sha256(),
        )


def test_partial_active_initial_seed_is_valid_but_not_extension_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "partial"
    _run_initial_fixture(
        runner,
        monkeypatch,
        output,
        seeds=(10_000,),
        partial_seed=10_000,
    )
    assert runner.validate_seed_artifact(
        output / "seed_10000", 10_000, mode="initial"
    ) == output / "seed_10000"
    loaded = runner.load_pair4_states(output / "seed_10000", 10_000)
    assert loaded.extension_eligible == {"standard": False, "tail_shift": False}


def test_pair2_only_wall_cap_is_valid_but_has_no_continuation_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "pair2-only"
    base = ExperimentConfig.from_yaml(CONFIG_PATH)
    config = base.with_overrides(
        seeds=(10_000,), devices=("cpu",), output_dir=output
    )
    smoke = tmp_path / "pair2-smoke.json"
    _write_smoke_manifest(runner, smoke, config)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(
        runner,
        "run_phase0c_seed",
        lambda _config, *, seed, device, **_kwargs: _pair2_only_result(seed, device),
    )
    runner.run_config(
        config,
        output,
        mode="initial",
        workers_per_device=1,
        candidate_chunk_size=16,
        resume=False,
        smoke_manifest=smoke,
    )
    loaded = runner.load_pair4_states(output / "seed_10000", 10_000)
    assert loaded.pair4_states == {"standard": (), "tail_shift": ()}
    assert loaded.extension_eligible == {"standard": False, "tail_shift": False}


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("seed_float", "integer dtype"),
        ("width_zero", "positive"),
        ("width_negative", "positive"),
        ("width_nan", "finite"),
        ("wrong_vector", "length 12"),
        ("stream_collision", "stream"),
        ("record_surface", "disagrees"),
        ("truncated_npz", "unreadable"),
        ("state_hash", "state hash"),
    ],
)
def test_deep_seed_validator_rejects_adversarial_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    runner = _load_runner()
    output = tmp_path / mutation
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    records_path = seed_dir / "records.csv"
    surfaces_path = seed_dir / "surfaces.npz"
    metadata_path = seed_dir / "metadata.json"
    records = pd.read_csv(records_path)
    if mutation == "seed_float":
        records["seed"] = 10_000.5
        records.to_csv(records_path, index=False)
    elif mutation in {"width_zero", "width_negative", "width_nan"}:
        records.loc[0, "micro_normalized_width"] = {
            "width_zero": 0.0,
            "width_negative": -1.0,
            "width_nan": np.nan,
        }[mutation]
        records.to_csv(records_path, index=False)
    elif mutation == "wrong_vector":
        records.loc[0, "final_coverage_json"] = _compact([0.93] * 11)
        records.to_csv(records_path, index=False)
    elif mutation == "stream_collision":
        records["evaluation_stream_id"] = records["tuning_stream_id"]
        records.to_csv(records_path, index=False)
    elif mutation == "record_surface":
        records.loc[0, "q_by_time_json"] = _compact([1.25] * 12)
        records.to_csv(records_path, index=False)
    elif mutation == "truncated_npz":
        member = io.BytesIO()
        np.save(member, np.arange(4))
        with zipfile.ZipFile(surfaces_path, "w") as archive:
            archive.writestr("broken.npy", member.getvalue()[:-8])
    else:
        metadata = json.loads(metadata_path.read_text())
        metadata["diagnostics"]["standard"]["pair4_state_sha256"][0] = "0" * 64
        metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match=match):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


@pytest.mark.parametrize(
    ("name", "wrong_dtype"),
    [
        ("standard_profile", np.complex64),
        ("standard_joint_B_schedule", np.complex64),
        ("standard_pair4_greedy_stage_grid_indices", np.int32),
        ("standard_pair4_greedy_completed_sweep_pairs", np.int32),
        ("standard_active_start_names", np.uint32),
        ("standard_extension_eligible", np.uint8),
    ],
)
def test_npz_arrays_require_exact_producer_dtypes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    wrong_dtype: type[np.generic],
) -> None:
    runner = _load_runner()
    output = tmp_path / f"dtype-{name}-{wrong_dtype.__name__}"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    surfaces_path = seed_dir / "surfaces.npz"
    with np.load(surfaces_path, allow_pickle=False) as archive:
        changed = np.asarray(archive[name]).astype(wrong_dtype)
    _rewrite_npz(surfaces_path, updates={name: changed})

    with pytest.raises(RuntimeError, match="dtype"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def test_record_scalar_helpers_accept_exact_numpy_scalars() -> None:
    runner = _load_runner()

    assert runner._pandas_integer(
        np.int64(4), seed=10_000, field="count", minimum=0
    ) == 4
    assert runner._pandas_boolean(
        np.bool_(True), seed=10_000, field="flag"
    ) is True


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 2, "schema_version"),
        ("seed", 10_000.9, "integer dtype"),
        ("scenario", 2, "row keys"),
        ("method_id", 2, "row keys"),
        ("analysis_role", 2, "method metadata"),
        ("budget_id", 2, "method metadata"),
        ("sweep_pairs", 4.9, "sweep_pairs"),
        ("selection_status", 2, "selection status"),
        ("selection_available", 2, "selection_available"),
        ("tuning_joint_feasible", 2, "tuning_joint_feasible"),
        ("failure_reason", "SELECTED", "failure_reason"),
        ("chosen_initialization", 2, "chosen_initialization"),
        ("selected_endpoint_stage_count", 0.9, "endpoint"),
        ("tuning_stream_id", 1_300_001.9, "tuning_stream_id"),
        ("evaluation_stream_id", 1_400_001.9, "evaluation_stream_id"),
        ("n_tuning_rollouts", 5_000.9, "tuning rollout"),
        ("n_evaluation_rollouts", 50_000.9, "evaluation rollout"),
        ("schedule_evaluations", 0.9, "schedule_evaluations"),
        ("committed_updates", 0.9, "committed_updates"),
        ("converged_at_pair", 1.9, "converged_at_pair"),
        ("wall_time_seconds", "not-a-number", "wall time"),
    ],
)
def test_record_schema_rejects_lossy_scalar_coercion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    match: str,
) -> None:
    runner = _load_runner()
    output = tmp_path / field
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    records_path = seed_dir / "records.csv"
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(
        "joint_2B"
    )
    records[field] = records[field].astype(object)
    records.loc[selected, field] = value
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match=match):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


@pytest.mark.parametrize(
    ("method", "field"),
    [
        ("current_profiled", "schedule_evaluations"),
        ("current_profiled", "committed_updates"),
        ("greedy", "schedule_evaluations"),
        ("greedy", "committed_updates"),
    ],
)
def test_reference_search_counts_are_exactly_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    field: str,
) -> None:
    runner = _load_runner()
    output = tmp_path / f"{method}-{field}"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    records_path = seed_dir / "records.csv"
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(method)
    records.loc[selected, field] = 1
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="exactly zero"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def test_reference_wall_time_is_exactly_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "reference-wall"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    records_path = seed_dir / "records.csv"
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(
        "current_profiled"
    )
    records.loc[selected, "wall_time_seconds"] = 0.001
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="wall time.*zero"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def test_initial_joint_checkpoints_share_one_search_wall_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "nested-wall"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    records_path = seed_dir / "records.csv"
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(
        "joint_2B"
    )
    records.loc[selected, "wall_time_seconds"] = 2.0
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="checkpoint wall times"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


@pytest.mark.parametrize("field", ["schedule_evaluations", "committed_updates"])
def test_unavailable_row_search_counts_are_exactly_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    runner = _load_runner()
    output = tmp_path / f"unavailable-{field}"
    base = ExperimentConfig.from_yaml(CONFIG_PATH)
    config = base.with_overrides(seeds=(10_000,), devices=("cpu",), output_dir=output)
    smoke = tmp_path / f"unavailable-{field}-smoke.json"
    _write_smoke_manifest(runner, smoke, config)
    result = _pair2_only_result(10_000, "cpu")
    row = next(
        row
        for row in result.records
        if row["scenario"] == "standard" and row["method_id"] == "joint_2B"
    )
    row[field] = 1
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(
        runner,
        "run_phase0c_seed",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(RuntimeError, match="exactly zero"):
        runner.run_config(
            config,
            output,
            mode="initial",
            workers_per_device=1,
            candidate_chunk_size=16,
            resume=False,
            smoke_manifest=smoke,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("missing", None),
        ("extra", 1),
        ("protocol", "phase0c_runner_measurement_v0"),
        ("elapsed_seconds", 0.0),
        ("elapsed_seconds", True),
        ("elapsed_seconds", math.nan),
        ("max_memory_allocated_bytes", -1),
        ("max_memory_allocated_bytes", 1.5),
        ("max_memory_allocated_bytes", True),
        ("max_memory_reserved_bytes", -1),
    ],
)
def test_seed_validator_rejects_malformed_atomic_runner_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    runner = _load_runner()
    output = tmp_path / f"measurement-{field}-{value}"
    _run_initial_fixture(runner, monkeypatch, output)
    metadata_path = output / "seed_10000" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    measurement = metadata["diagnostics"]["runner_measurement"]
    if field == "missing":
        metadata["diagnostics"].pop("runner_measurement")
    elif field == "extra":
        measurement["extra"] = value
    else:
        measurement[field] = value
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="runner measurement"):
        runner.validate_seed_artifact(
            output / "seed_10000", 10_000, mode="initial"
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ordered_seeds", [10_000.0]),
        ("workers_per_device", 1.0),
        ("candidate_chunk_size", 16.0),
        ("sweep_pair_checkpoints", [2.0, 4.0]),
    ],
)
def test_seed_execution_counts_require_exact_nonbool_integers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    runner = _load_runner()
    output = tmp_path / f"execution-{field}"
    _run_initial_fixture(runner, monkeypatch, output)
    metadata_path = output / "seed_10000" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    provenance = metadata["diagnostics"]["runner_provenance"]
    provenance[field] = value
    unhashed = dict(provenance)
    unhashed.pop("execution_sha256")
    provenance["execution_sha256"] = runner._canonical_sha256(unhashed)
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="execution.*integer"):
        runner.validate_seed_artifact(
            output / "seed_10000", 10_000, mode="initial"
        )


@pytest.mark.parametrize(
    ("scenario", "tuning_micro", "micro", "patient", "tuning_width", "final_width"),
    [
        (
            "standard",
            1.9110114574432373,
            1.910032033920288,
            1.9100321531295776,
            [
                2.0896928310394287, 1.932094931602478, 1.9361780881881714,
                1.88801908493042, 1.8839011192321777, 1.9341093301773071,
                1.9072444438934326, 1.8646681308746338, 1.895674228668213,
                1.9044526815414429, 1.8746216297149658, 1.8214809894561768,
            ],
            [
                2.085171699523926, 1.930895209312439, 1.9359725713729858,
                1.8873767852783203, 1.8817391395568848, 1.9344522953033447,
                1.9070290327072144, 1.8644696474075317, 1.8942149877548218,
                1.9041286706924438, 1.873592495918274, 1.8213424682617188,
            ],
        ),
        (
            "tail_shift",
            2.022510766983032,
            2.022746324539185,
            2.022746086120605,
            [
                2.2566819190979004, 2.089000940322876, 2.006983995437622,
                2.067286968231201, 1.916393756866455, 2.012202739715576,
                1.9820464849472046, 1.9445734024047852, 2.0600900650024414,
                2.051872968673706, 2.0306272506713867, 1.8523675203323364,
            ],
            [
                2.2563369274139404, 2.0938258171081543, 2.0150740146636963,
                2.067772150039673, 1.9153777360916138, 1.9977521896362305,
                1.975793719291687, 1.952088713645935, 2.0626275539398193,
                2.050555944442749, 2.032233953475952, 1.853514313697815,
            ],
        ),
    ],
)
def test_validator_accepts_real_float32_reduction_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    tuning_micro: float,
    micro: float,
    patient: float,
    tuning_width: list[float],
    final_width: list[float],
) -> None:
    runner = _load_runner()
    output = tmp_path / scenario
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    records_path = seed_dir / "records.csv"
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq(scenario) & records["method_id"].eq(
        "current_profiled"
    )
    records.loc[selected, "tuning_stage_width_json"] = _compact(tuning_width)
    records.loc[selected, "tuning_micro_width"] = tuning_micro
    records.loc[selected, "final_stage_width_json"] = _compact(final_width)
    records.loc[selected, "micro_normalized_width"] = micro
    records.loc[selected, "patient_normalized_width"] = patient
    records.to_csv(records_path, index=False)
    _rewrite_npz(
        seed_dir / "surfaces.npz",
        updates={
            f"{scenario}_current_profiled_tuning_stage_width": np.asarray(
                tuning_width, dtype=np.float32
            ),
            f"{scenario}_current_profiled_final_stage_width": np.asarray(
                final_width, dtype=np.float32
            ),
        },
    )

    assert runner.validate_seed_artifact(seed_dir, 10_000, mode="initial") == seed_dir


def test_validator_rejects_float32_width_corruption_at_one_e_minus_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "corrupt-width"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    records_path = seed_dir / "records.csv"
    records = pd.read_csv(records_path)
    records.loc[0, "micro_normalized_width"] += 1e-4
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="micro width disagrees"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def _set_mixed_joint_schedule(seed_dir: Path, method: str) -> None:
    records_path = seed_dir / "records.csv"
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(method)
    indices = [25, -1, *([25] * 10)]
    schedule = [1.25, 1.5, *([1.25] * 10)]
    records.loc[selected, "selected_stage_grid_indices_json"] = _compact(indices)
    records.loc[selected, "q_by_time_json"] = _compact(schedule)
    records.to_csv(records_path, index=False)
    _rewrite_npz(
        seed_dir / "surfaces.npz",
        updates={
            f"standard_{method}_schedule": np.asarray(schedule, dtype=np.float32)
        },
    )


def test_initial_joint_minus_one_uses_authenticated_current_profiled_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "mixed-initial"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    _set_mixed_joint_schedule(seed_dir, "joint_B")

    assert runner.validate_seed_artifact(seed_dir, 10_000, mode="initial") == seed_dir


@pytest.mark.parametrize("method", ["current_profiled", "greedy"])
def test_reference_indices_still_reject_minus_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    runner = _load_runner()
    output = tmp_path / method
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    records_path = seed_dir / "records.csv"
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(method)
    indices = [-1] if method == "current_profiled" else [-1, *([75] * 11)]
    records.loc[selected, "selected_stage_grid_indices_json"] = _compact(indices)
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="selected indices are invalid"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def _write_extension_seed(
    runner: object,
    parent: Path,
    extension: Path,
    result: SeedResult,
) -> Path:
    extension.mkdir()
    base = ExperimentConfig.from_yaml(CONFIG_PATH)
    config = base.with_overrides(
        seeds=(10_000,), devices=("cpu",), output_dir=extension
    )
    parent_metadata = json.loads((parent / "study_metadata.json").read_text())
    execution = runner._execution_metadata(
        config,
        mode="extension-8sp",
        workers_per_device=1,
        candidate_chunk_size=16,
        max_seed_wall_seconds=300.0,
        source_hash=runner.source_tree_sha256(),
        experiment_hash=runner.experiment_tree_sha256(),
        parent_fields={
            "parent_study_manifest_sha256": _sha256(parent / "study_manifest.json"),
            "checkpoint_decision_sha256": "d" * 64,
            "parent_execution_sha256": parent_metadata["execution"]["execution_sha256"],
            "parent_output_dir": str(parent.resolve()),
        },
    )
    measurement = {
        "protocol": "phase0c_runner_measurement_v1",
        "elapsed_seconds": 1.0,
        "max_memory_allocated_bytes": 0,
        "max_memory_reserved_bytes": 0,
    }
    return runner.write_seed_result(
        runner._runner_result(result, execution, measurement), extension, config
    )


def _write_mixed_extension_seed(
    runner: object,
    parent: Path,
    extension: Path,
    *,
    inherited_radius: float,
) -> Path:
    result = _extension_result(10_000, "cpu")
    indices = [25, -1, *([25] * 10)]
    schedule = torch.tensor([1.25, inherited_radius, *([1.25] * 10)])
    row = next(row for row in result.records if row["scenario"] == "standard")
    row["selected_stage_grid_indices_json"] = _compact(indices)
    row["q_by_time_json"] = _compact(schedule.tolist())
    result.surfaces["standard_joint_8SP_schedule"] = schedule
    return _write_extension_seed(runner, parent, extension, result)


def test_extension_joint_minus_one_uses_authenticated_parent_current_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    _run_initial_fixture(runner, monkeypatch, parent)
    seed_dir = _write_mixed_extension_seed(
        runner, parent, tmp_path / "extension", inherited_radius=1.5
    )

    assert runner.validate_seed_artifact(
        seed_dir,
        10_000,
        mode="extension-8sp",
        parent_seed_dir=parent / "seed_10000",
    ) == seed_dir


def test_extension_joint_minus_one_rejects_grid_last_instead_of_parent_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    _run_initial_fixture(runner, monkeypatch, parent)
    seed_dir = _write_mixed_extension_seed(
        runner, parent, tmp_path / "extension", inherited_radius=2.0
    )

    with pytest.raises(RuntimeError, match="index mapping disagrees"):
        runner.validate_seed_artifact(
            seed_dir,
            10_000,
            mode="extension-8sp",
            parent_seed_dir=parent / "seed_10000",
        )


def _set_joint_tuning_coverage(seed_dir: Path, value: float) -> None:
    records_path = seed_dir / "records.csv"
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(
        "joint_B"
    )
    coverage = [value] * 12
    records.loc[selected, "tuning_coverage_json"] = _compact(coverage)
    records.to_csv(records_path, index=False)
    _rewrite_npz(
        seed_dir / "surfaces.npz",
        updates={
            "standard_joint_B_tuning_coverage": np.asarray(
                coverage, dtype=np.float32
            )
        },
    )


def test_joint_target_accepts_exact_float32_point_nine_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "float32-target"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    _set_joint_tuning_coverage(seed_dir, 0.8999999761581421)

    assert runner.validate_seed_artifact(seed_dir, 10_000, mode="initial") == seed_dir


def test_joint_target_rejects_one_float32_ulp_below_point_nine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "below-target"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    _set_joint_tuning_coverage(seed_dir, 0.8999999165534973)

    with pytest.raises(RuntimeError, match="below .90"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def _early_parent_validation_cap_result(
    *,
    missing_stage_grids: tuple[str, ...],
) -> SeedResult:
    result = _extension_result(10_000, "cpu")
    for row in result.records:
        row.update(
            {
                "selection_status": "WALL_TIME_CAP",
                "selection_available": False,
                "tuning_joint_feasible": False,
                "failure_reason": "WALL_TIME_CAP",
                "chosen_initialization": "",
                "selected_endpoint_stage_count": 0,
                "selected_stage_grid_indices_json": "[]",
                "q_by_time_json": "[]",
                "tuning_coverage_json": "[]",
                "tuning_stage_width_json": "[]",
                "tuning_micro_width": float("nan"),
                "final_coverage_json": "[]",
                "final_wilson_lcb_json": "[]",
                "final_stage_width_json": "[]",
                "micro_normalized_width": float("nan"),
                "patient_normalized_width": float("nan"),
                "n_evaluation_rollouts": 0,
                "schedule_evaluations": 0,
                "committed_updates": 0,
                "converged_at_pair": None,
                "wall_time_seconds": 5.0,
            }
        )
        scenario = str(row["scenario"])
        for suffix in (
            "schedule",
            "tuning_coverage",
            "tuning_stage_width",
            "final_coverage",
            "final_wilson_lcb",
            "final_stage_width",
        ):
            result.surfaces[f"{scenario}_joint_8SP_{suffix}"] = torch.empty(0)
        result.diagnostics[scenario] = {
            "tuning_stream_id": row["tuning_stream_id"],
            "evaluation_stream_id": row["evaluation_stream_id"],
            "search_status": "WALL_TIME_CAP",
            "continuation_status": None,
            "fresh_evaluation_completed": False,
            "wall_time_phase": "parent_validation",
            "checkpoint": None,
        }
    for scenario in missing_stage_grids:
        result.surfaces.pop(f"{scenario}_stage_grids")
    return result


@pytest.mark.parametrize(
    "missing_stage_grids",
    [("standard", "tail_shift"), ("tail_shift",)],
)
def test_extension_accepts_exact_parent_validation_early_cap_without_stage_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_stage_grids: tuple[str, ...],
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    _run_initial_fixture(runner, monkeypatch, parent)
    seed_dir = _write_extension_seed(
        runner,
        parent,
        tmp_path / "extension",
        _early_parent_validation_cap_result(missing_stage_grids=missing_stage_grids),
    )

    assert runner.validate_seed_artifact(
        seed_dir,
        10_000,
        mode="extension-8sp",
        parent_seed_dir=parent / "seed_10000",
    ) == seed_dir


def test_parent_validation_stage_grid_presence_must_be_canonical_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    _run_initial_fixture(runner, monkeypatch, parent)
    seed_dir = _write_extension_seed(
        runner,
        parent,
        tmp_path / "extension",
        _early_parent_validation_cap_result(missing_stage_grids=("standard",)),
    )

    with pytest.raises(RuntimeError, match="stage grid.*prefix"):
        runner.validate_seed_artifact(
            seed_dir,
            10_000,
            mode="extension-8sp",
            parent_seed_dir=parent / "seed_10000",
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("wall_time_phase", "standard_cache", "wall-time phase differs"),
        ("continuation_status", "WALL_TIME_CAP", "continuation status differs"),
        ("fresh_evaluation_completed", True, "fresh-evaluation status differs"),
        ("checkpoint", {"garbage": 1}, "absent continuation has checkpoint"),
    ],
)
def test_extension_missing_stage_grid_requires_exact_parent_validation_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    match: str,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    _run_initial_fixture(runner, monkeypatch, parent)
    result = _early_parent_validation_cap_result(
        missing_stage_grids=("standard", "tail_shift")
    )
    result.diagnostics["standard"][field] = value
    seed_dir = _write_extension_seed(runner, parent, tmp_path / "extension", result)

    with pytest.raises(RuntimeError, match=match):
        runner.validate_seed_artifact(
            seed_dir,
            10_000,
            mode="extension-8sp",
            parent_seed_dir=parent / "seed_10000",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "search_status",
        "checkpoint_keys",
        "pair2_best_start",
        "pair4_best_start",
        "pair4_schedule",
        "pair4_coverage",
        "pair4_width",
    ],
)
def test_initial_diagnostics_cross_bind_rows_and_persisted_pair4_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    runner = _load_runner()
    output = tmp_path / mutation
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    metadata_path = seed_dir / "metadata.json"
    records_path = seed_dir / "records.csv"
    metadata = json.loads(metadata_path.read_text())
    records = pd.read_csv(records_path)
    if mutation == "search_status":
        metadata["diagnostics"]["standard"]["search_status"] = "CORRUPTED"
    elif mutation == "checkpoint_keys":
        metadata["diagnostics"]["standard"]["checkpoints"].pop("2")
    elif mutation == "pair2_best_start":
        metadata["diagnostics"]["standard"]["checkpoints"]["2"][
            "best_start_name"
        ] = "upper_endpoint"
    elif mutation == "pair4_best_start":
        metadata["diagnostics"]["standard"]["checkpoints"]["4"][
            "best_start_name"
        ] = "profiled"
    else:
        selected = records["scenario"].eq("standard") & records["method_id"].eq(
            "joint_2B"
        )
        if mutation == "pair4_schedule":
            records.loc[selected, "selected_stage_grid_indices_json"] = _compact(
                [50] * 12
            )
            records.loc[selected, "q_by_time_json"] = _compact([1.5] * 12)
            updates = {
                "standard_joint_2B_schedule": np.full(12, 1.5, dtype=np.float32)
            }
        elif mutation == "pair4_coverage":
            coverage = [0.9200000166893005] * 12
            records.loc[selected, "tuning_coverage_json"] = _compact(coverage)
            updates = {
                "standard_joint_2B_tuning_coverage": np.asarray(
                    coverage, dtype=np.float32
                )
            }
        else:
            width = [1.7999999523162842] * 12
            records.loc[selected, "tuning_stage_width_json"] = _compact(width)
            records.loc[selected, "tuning_micro_width"] = 1.7999998331069946
            updates = {
                "standard_joint_2B_tuning_stage_width": np.asarray(
                    width, dtype=np.float32
                )
            }
        records.to_csv(records_path, index=False)
        _rewrite_npz(seed_dir / "surfaces.npz", updates=updates)
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="diagnostic|checkpoint|persisted pair4"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def _install_initial_trace(
    seed_dir: Path, trace: list[dict[str, object]]
) -> None:
    metadata_path = seed_dir / "metadata.json"
    records_path = seed_dir / "records.csv"
    metadata = json.loads(metadata_path.read_text())
    checkpoint = metadata["diagnostics"]["standard"]["checkpoints"]["2"]
    checkpoint["trace"] = trace
    checkpoint["schedule_evaluations"] = 101 * len(trace)
    checkpoint["committed_updates"] = sum(
        bool(step["committed"]) for step in trace
    )
    metadata_path.write_text(json.dumps(metadata))
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(
        "joint_B"
    )
    records.loc[selected, "schedule_evaluations"] = 101 * len(trace)
    records.loc[selected, "committed_updates"] = sum(
        bool(step["committed"]) for step in trace
    )
    records.to_csv(records_path, index=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "swapped_steps",
        "missing_step",
        "committed_after",
        "uncommitted_after",
        "broken_chain",
        "no_commit_before_later_pair",
    ],
)
def test_coordinate_trace_requires_complete_canonical_stateful_producer_sequence(
    mutation: str,
) -> None:
    runner = _load_runner()
    trace = _coordinate_trace(START_NAMES, 1, 2)
    if mutation == "swapped_steps":
        trace[0], trace[1] = trace[1], trace[0]
    elif mutation == "missing_step":
        trace.pop(1)
    elif mutation == "committed_after":
        trace[0]["after_micro_width"] = trace[0]["proposed_micro_width"] + 0.001
    elif mutation == "uncommitted_after":
        trace[1]["after_micro_width"] = trace[1]["before_micro_width"] + 0.001
    elif mutation == "broken_chain":
        trace[1]["before_micro_width"] = trace[1]["after_micro_width"] = 1.5
        trace[1]["proposed_micro_width"] = 1.6
    else:
        first = trace[0]
        first["proposed_micro_width"] = first["before_micro_width"]
        first["committed"] = False
        first["after_micro_width"] = first["before_micro_width"]

    with pytest.raises(RuntimeError, match="trace"):
        runner._validate_coordinate_trace(
            trace,
            seed=10_000,
            label="standard pair2",
            start_names=START_NAMES,
            minimum_pair=1,
            maximum_pair=2,
        )


def test_initial_checkpoint_accepts_exact_coordinate_trace_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "valid-trace"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    _install_initial_trace(seed_dir, _coordinate_trace(START_NAMES, 1, 2))

    assert runner.validate_seed_artifact(seed_dir, 10_000, mode="initial") == seed_dir


@pytest.mark.parametrize(
    ("field", "value", "remove"),
    [
        ("start_name", "unknown", False),
        ("sweep_pair", True, False),
        ("sweep_pair", 3, False),
        ("direction", "sideways", False),
        ("stage", 12, False),
        ("feasible_count", -1, False),
        ("proposed_grid_index", 101, False),
        ("before_micro_width", 0.0, False),
        ("proposed_micro_width", math.nan, False),
        ("committed", 1, False),
        ("after_micro_width", -1.0, False),
        ("stage", None, True),
    ],
)
def test_coordinate_trace_rejects_wrong_fields_types_and_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    remove: bool,
) -> None:
    runner = _load_runner()
    output = tmp_path / f"trace-{field}-{value}"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    trace = _coordinate_trace(START_NAMES, 1, 2)
    step = trace[0]
    if remove:
        step.pop(field)
    else:
        step[field] = value
    _install_initial_trace(seed_dir, trace)

    with pytest.raises(RuntimeError, match="trace"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


@pytest.mark.parametrize(
    ("schedule_evaluations", "committed_updates"),
    [(14_545, 2), (14_544, 3)],
)
def test_checkpoint_counts_are_derived_exactly_from_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule_evaluations: int,
    committed_updates: int,
) -> None:
    runner = _load_runner()
    output = tmp_path / f"counts-{schedule_evaluations}-{committed_updates}"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    _install_initial_trace(seed_dir, _coordinate_trace(START_NAMES, 1, 2))
    metadata_path = seed_dir / "metadata.json"
    records_path = seed_dir / "records.csv"
    metadata = json.loads(metadata_path.read_text())
    checkpoint = metadata["diagnostics"]["standard"]["checkpoints"]["2"]
    checkpoint["schedule_evaluations"] = schedule_evaluations
    checkpoint["committed_updates"] = committed_updates
    metadata_path.write_text(json.dumps(metadata))
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(
        "joint_B"
    )
    records.loc[selected, "schedule_evaluations"] = schedule_evaluations
    records.loc[selected, "committed_updates"] = committed_updates
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="trace-derived"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def test_initial_row_convergence_is_derived_from_checkpoint_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "row-convergence"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    records_path = seed_dir / "records.csv"
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(
        "joint_B"
    )
    records.loc[selected, "converged_at_pair"] = 1
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="convergence"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def test_initial_pair2_trace_must_be_exact_pair4_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "nested-prefix"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    metadata_path = seed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["diagnostics"]["standard"]["checkpoints"]["4"]["trace"][0][
        "feasible_count"
    ] = 3
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="prefix"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def test_initial_checkpoint_winner_is_derived_from_trace_even_if_row_agrees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "trace-winner"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    metadata_path = seed_dir / "metadata.json"
    records_path = seed_dir / "records.csv"
    metadata = json.loads(metadata_path.read_text())
    metadata["diagnostics"]["standard"]["checkpoints"]["2"][
        "best_start_name"
    ] = "greedy"
    metadata_path.write_text(json.dumps(metadata))
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(
        "joint_B"
    )
    records.loc[selected, "chosen_initialization"] = "greedy"
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="winner.*trace"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def test_pair4_trace_final_width_cross_checks_row_and_persisted_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "trace-state-width"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    records_path = seed_dir / "records.csv"
    surfaces_path = seed_dir / "surfaces.npz"
    metadata_path = seed_dir / "metadata.json"
    changed_width = np.full(12, 1.41, dtype=np.float32)
    _rewrite_npz(
        surfaces_path,
        updates={
            "standard_pair4_greedy_normalized_width": changed_width,
            "standard_joint_2B_tuning_stage_width": changed_width,
        },
    )
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(
        "joint_2B"
    )
    records.loc[selected, "tuning_stage_width_json"] = _compact(
        changed_width.tolist()
    )
    records.loc[selected, "tuning_micro_width"] = float(changed_width.mean())
    records.to_csv(records_path, index=False)
    metadata = json.loads(metadata_path.read_text())
    surfaces = runner._load_npz(seed_dir, 10_000)
    changed_state = runner._state_from_surfaces(
        surfaces,
        scenario="standard",
        start_name="greedy",
        seed=10_000,
    )
    metadata["diagnostics"]["standard"]["pair4_state_sha256"][1] = runner._state_sha256(
        changed_state
    )
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="trace.*width|width.*trace"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def _make_pair4_materialize_pair2(
    runner: object,
    seed_dir: Path,
    *,
    converged: bool,
) -> None:
    metadata_path = seed_dir / "metadata.json"
    records_path = seed_dir / "records.csv"
    surfaces_path = seed_dir / "surfaces.npz"
    metadata = json.loads(metadata_path.read_text())
    trace = _coordinate_trace(
        START_NAMES,
        1,
        2,
        commit_last_pair=not converged,
    )
    trace_widths = {
        name: next(
            float(step["after_micro_width"])
            for step in reversed(trace)
            if step["start_name"] == name
        )
        for name in START_NAMES
    }
    checkpoint2 = metadata["diagnostics"]["standard"]["checkpoints"]["2"]
    checkpoint2.update(
        {
            "executed_sweep_pairs": 2,
            "best_start_name": "profiled",
            "schedule_evaluations": 101 * len(trace),
            "committed_updates": sum(bool(step["committed"]) for step in trace),
            "trace": trace,
        }
    )
    checkpoint4 = json.loads(json.dumps(checkpoint2))
    checkpoint4["requested_sweep_pairs"] = 4
    metadata["diagnostics"]["standard"]["checkpoints"]["4"] = checkpoint4

    records = pd.read_csv(records_path)
    pair2_selected = records["scenario"].eq("standard") & records["method_id"].eq(
        "joint_B"
    )
    pair4_selected = records["scenario"].eq("standard") & records["method_id"].eq(
        "joint_2B"
    )
    pair2_index = records.index[pair2_selected][0]
    pair4_index = records.index[pair4_selected][0]
    width = np.full(12, trace_widths["profiled"], dtype=np.float32)
    records.loc[pair2_selected, "tuning_stage_width_json"] = _compact(width.tolist())
    records.loc[pair2_selected, "tuning_micro_width"] = float(width.mean())
    records.loc[pair2_selected, "schedule_evaluations"] = 101 * len(trace)
    records.loc[pair2_selected, "committed_updates"] = sum(
        bool(step["committed"]) for step in trace
    )
    for field in (
        "chosen_initialization",
        "selected_endpoint_stage_count",
        "selected_stage_grid_indices_json",
        "q_by_time_json",
        "tuning_coverage_json",
        "tuning_stage_width_json",
        "tuning_micro_width",
        "schedule_evaluations",
        "committed_updates",
    ):
        records.at[pair4_index, field] = records.at[pair2_index, field]
    convergence = 2 if converged else np.nan
    records.loc[pair2_selected | pair4_selected, "converged_at_pair"] = convergence
    records.to_csv(records_path, index=False)

    with np.load(surfaces_path, allow_pickle=False) as archive:
        updates = {
            "standard_joint_B_tuning_stage_width": width,
            "standard_joint_2B_schedule": np.array(
                archive["standard_joint_B_schedule"], copy=True
            ),
            "standard_joint_2B_tuning_coverage": np.array(
                archive["standard_joint_B_tuning_coverage"], copy=True
            ),
            "standard_joint_2B_tuning_stage_width": width,
            "standard_pair4_profiled_radii": np.array(
                archive["standard_joint_B_schedule"], copy=True
            ),
            "standard_pair4_profiled_stage_grid_indices": np.full(
                12, 25, dtype=np.int64
            ),
        }
        for name in START_NAMES:
            updates[f"standard_pair4_{name}_normalized_width"] = np.full(
                12, trace_widths[name], dtype=np.float32
            )
            updates[f"standard_pair4_{name}_converged_at_pair"] = np.asarray(
                2 if converged else -1, dtype=np.int64
            )
    _rewrite_npz(surfaces_path, updates=updates)
    surfaces = runner._load_npz(seed_dir, 10_000)
    metadata["diagnostics"]["standard"]["pair4_state_sha256"] = [
        runner._state_sha256(
            runner._state_from_surfaces(
                surfaces,
                scenario="standard",
                start_name=name,
                seed=10_000,
            )
        )
        for name in START_NAMES
    ]
    metadata_path.write_text(json.dumps(metadata))


def test_equal_executed_nested_checkpoints_accept_exact_converged_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "equal-converged"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    _make_pair4_materialize_pair2(runner, seed_dir, converged=True)

    assert runner.validate_seed_artifact(seed_dir, 10_000, mode="initial") == seed_dir


def test_equal_executed_nested_checkpoints_require_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "equal-with-commit"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    _make_pair4_materialize_pair2(runner, seed_dir, converged=False)

    with pytest.raises(RuntimeError, match="equal.*converg"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def test_equal_executed_nested_checkpoints_require_identical_tuning_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "equal-array-mismatch"
    _run_initial_fixture(runner, monkeypatch, output)
    seed_dir = output / "seed_10000"
    _make_pair4_materialize_pair2(runner, seed_dir, converged=True)
    surfaces_path = seed_dir / "surfaces.npz"
    metadata_path = seed_dir / "metadata.json"
    records_path = seed_dir / "records.csv"
    changed_coverage = np.full(12, 0.92, dtype=np.float32)
    _rewrite_npz(
        surfaces_path,
        updates={
            "standard_joint_2B_tuning_coverage": changed_coverage,
            "standard_pair4_profiled_coverage": changed_coverage,
        },
    )
    records = pd.read_csv(records_path)
    selected = records["scenario"].eq("standard") & records["method_id"].eq(
        "joint_2B"
    )
    records.loc[selected, "tuning_coverage_json"] = _compact(
        changed_coverage.tolist()
    )
    records.to_csv(records_path, index=False)
    metadata = json.loads(metadata_path.read_text())
    surfaces = runner._load_npz(seed_dir, 10_000)
    metadata["diagnostics"]["standard"]["pair4_state_sha256"][0] = runner._state_sha256(
        runner._state_from_surfaces(
            surfaces,
            scenario="standard",
            start_name="profiled",
            seed=10_000,
        )
    )
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="equal nested checkpoint rows"):
        runner.validate_seed_artifact(seed_dir, 10_000, mode="initial")


def test_extension_row_convergence_is_derived_from_checkpoint_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    _run_initial_fixture(runner, monkeypatch, parent)
    result = _extension_result(10_000, "cpu")
    result.records[0]["converged_at_pair"] = 7
    seed_dir = _write_extension_seed(runner, parent, tmp_path / "extension", result)

    with pytest.raises(RuntimeError, match="convergence"):
        runner.validate_seed_artifact(
            seed_dir,
            10_000,
            mode="extension-8sp",
            parent_seed_dir=parent / "seed_10000",
        )


@pytest.mark.parametrize(
    "field",
    [
        "search_status",
        "continuation_status",
        "fresh_evaluation_completed",
        "wall_time_phase",
        "checkpoint",
    ],
)
def test_extension_diagnostics_require_exact_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    _run_initial_fixture(runner, monkeypatch, parent)
    result = _extension_result(10_000, "cpu")
    result.diagnostics["standard"].pop(field)
    seed_dir = _write_extension_seed(runner, parent, tmp_path / "extension", result)

    with pytest.raises(RuntimeError, match="diagnostics.*exact fields"):
        runner.validate_seed_artifact(
            seed_dir,
            10_000,
            mode="extension-8sp",
            parent_seed_dir=parent / "seed_10000",
        )


@pytest.mark.parametrize("mutation", ["best_start", "garbage_trace", "status"])
def test_extension_checkpoint_and_status_cross_bind_selected_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    _run_initial_fixture(runner, monkeypatch, parent)
    result = _extension_result(10_000, "cpu")
    diagnostics = result.diagnostics["standard"]
    if mutation == "best_start":
        diagnostics["checkpoint"]["best_start_name"] = "greedy"
    elif mutation == "garbage_trace":
        diagnostics["checkpoint"]["trace"] = [{"garbage": 1}]
    else:
        diagnostics["continuation_status"] = "CORRUPTED"
    seed_dir = _write_extension_seed(runner, parent, tmp_path / "extension", result)

    with pytest.raises(RuntimeError, match="checkpoint|trace|status"):
        runner.validate_seed_artifact(
            seed_dir,
            10_000,
            mode="extension-8sp",
            parent_seed_dir=parent / "seed_10000",
        )


def test_resume_complete_missing_one_of_forty_fails_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "forty"
    config = _run_initial_fixture(
        runner, monkeypatch, output, seeds=tuple(range(10_000, 10_040))
    )
    shutil.rmtree(output / "seed_10039")
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    monkeypatch.setattr(
        runner,
        "run_phase0c_seed",
        lambda *_args, **_kwargs: pytest.fail("inconsistent COMPLETE must block work"),
    )

    with pytest.raises(RuntimeError, match="COMPLETE.*missing"):
        runner.run_config(
            config,
            output,
            mode="initial",
            workers_per_device=1,
            candidate_chunk_size=16,
            resume=True,
            smoke_manifest=tmp_path / "forty-smoke.json",
        )
    after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert after == before


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("source", "source_tree_sha256"),
        ("experiment", "experiment_tree_sha256"),
        ("config", "config_sha256"),
        ("runtime", "runtime_config_sha256"),
        ("devices", "devices"),
        ("workers", "workers_per_device"),
        ("chunk", "candidate_chunk_size"),
        ("cap", "max_seed_wall_seconds"),
        ("checkpoints", "sweep_pair_checkpoints"),
    ],
)
def test_resume_provenance_mismatch_fails_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    runner = _load_runner()
    output = tmp_path / mutation
    config = _run_initial_fixture(runner, monkeypatch, output)
    (output / "COMPLETE").unlink()
    metadata_path = output / "study_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    target = metadata if mutation in {"source", "devices"} else metadata["execution"]
    keys = {
        "source": "source_tree_sha256",
        "experiment": "experiment_tree_sha256",
        "config": "config_sha256",
        "runtime": "runtime_config_sha256",
        "devices": "devices",
        "workers": "workers_per_device",
        "chunk": "candidate_chunk_size",
        "cap": "max_seed_wall_seconds",
        "checkpoints": "sweep_pair_checkpoints",
    }
    key = keys[mutation]
    target[key] = ["cuda:0"] if mutation == "devices" else (
        [2, 8] if mutation == "checkpoints" else "bad"
    )
    metadata_path.write_text(json.dumps(metadata))
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}

    with pytest.raises(RuntimeError, match=match):
        runner.run_config(
            config,
            output,
            mode="initial",
            workers_per_device=1,
            candidate_chunk_size=16,
            resume=True,
            smoke_manifest=tmp_path / f"{mutation}-smoke.json",
        )
    after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert after == before


def test_smoke_mode_writes_measured_manifest_and_completes_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "smoke"
    base = ExperimentConfig.from_yaml(CONFIG_PATH)
    config = base.with_overrides(seeds=(9999,), devices=("cpu",), output_dir=output)
    _InlineExecutor.instances = []
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(
        runner,
        "run_phase0c_seed",
        lambda _config, *, seed, device, **_kwargs: _initial_result(seed, device),
    )

    runner.run_config(
        config,
        output,
        mode="smoke",
        workers_per_device=1,
        candidate_chunk_size=16,
        resume=False,
    )

    payload = json.loads((output / "smoke_manifest.json").read_text())
    assert set(payload) == {
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
    assert payload["elapsed_seconds"] > 0.0
    assert payload["recommended_max_seed_wall_seconds"] == runner.calibrate_wall_cap(
        payload["elapsed_seconds"]
    )
    assert (output / "COMPLETE").is_file()
    runner.validate_study_manifest(output, expected_kind="smoke")


def test_smoke_resume_recovers_authenticated_seed_measurement_without_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "smoke-resume"
    base = ExperimentConfig.from_yaml(CONFIG_PATH)
    config = base.with_overrides(seeds=(9999,), devices=("cpu",), output_dir=output)
    _InlineExecutor.instances = []
    calls: list[int] = []
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(
        runner,
        "run_phase0c_seed",
        lambda _config, *, seed, device, **_kwargs: (
            calls.append(seed) or _initial_result(seed, device)
        ),
    )
    original_write_smoke = runner._write_smoke_result
    monkeypatch.setattr(
        runner,
        "_write_smoke_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("crash after seed completion")
        ),
    )

    with pytest.raises(RuntimeError, match="crash after seed completion"):
        runner.run_config(
            config,
            output,
            mode="smoke",
            workers_per_device=1,
            candidate_chunk_size=16,
            resume=False,
        )
    seed_metadata = json.loads((output / "seed_09999" / "metadata.json").read_text())
    persisted = seed_metadata["diagnostics"]["runner_measurement"]
    assert calls == [9999]
    assert not (output / "study_manifest.json").exists()
    assert not (output / "COMPLETE").exists()

    monkeypatch.setattr(runner, "_write_smoke_result", original_write_smoke)
    runner.run_config(
        config,
        output,
        mode="smoke",
        workers_per_device=1,
        candidate_chunk_size=16,
        resume=True,
    )

    smoke = json.loads((output / "smoke_manifest.json").read_text())
    assert calls == [9999]
    assert smoke["elapsed_seconds"] == persisted["elapsed_seconds"]
    assert smoke["max_memory_allocated_bytes"] == persisted[
        "max_memory_allocated_bytes"
    ]
    assert smoke["max_memory_reserved_bytes"] == persisted[
        "max_memory_reserved_bytes"
    ]
    runner.validate_study_manifest(output, expected_kind="smoke")


def test_worker_failure_never_publishes_root_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "failed"
    base = ExperimentConfig.from_yaml(CONFIG_PATH)
    config = base.with_overrides(seeds=(10_000,), devices=("cpu",), output_dir=output)
    smoke = tmp_path / "failed-smoke.json"
    _write_smoke_manifest(runner, smoke, config)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(
        runner,
        "run_phase0c_seed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("worker failed")),
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        runner.run_config(
            config,
            output,
            mode="initial",
            workers_per_device=1,
            candidate_chunk_size=16,
            resume=False,
            smoke_manifest=smoke,
        )
    assert not (output / "COMPLETE").exists()
    assert not (output / "study_manifest.json").exists()


def test_authorize_extension_cross_checks_summary_and_parent_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    seeds = tuple(range(10_000, 10_040))
    config = _run_initial_fixture(runner, monkeypatch, parent, seeds=seeds)
    decision = _write_checkpoint_analysis(parent, seeds)

    authorization = runner.authorize_extension(
        parent,
        decision,
        config=config,
        source_hash=runner.source_tree_sha256(),
        experiment_hash=runner.experiment_tree_sha256(),
    )

    summary_manifest = json.loads(
        (parent / "checkpoint_analysis" / "phase0c_summary_manifest.json").read_text()
    )
    assert authorization == {
        "parent_study_manifest_sha256": _sha256(parent / "study_manifest.json"),
        "checkpoint_decision_sha256": summary_manifest["files"]["phase0c_decision.json"]["sha256"],
        "parent_execution_sha256": json.loads((parent / "study_metadata.json").read_text())["execution"]["execution_sha256"],
        "max_seed_wall_seconds": 300.0,
    }


@pytest.mark.parametrize(
    ("decision_mutation", "manifest_mutation", "corrupt_fact", "match"),
    [
        (("decision", "STOP_SCALAR_SATURATED"), None, False, "decision"),
        (("parent_study_manifest_sha256", "0" * 64), None, False, "parent_study_manifest_sha256"),
        (("ordered_seeds", list(range(10_000, 10_039))), None, False, "ordered_seeds"),
        (("ordered_seeds", [float(seed) for seed in range(10_000, 10_040)]), None, False, "ordered_seeds"),
        (("source_tree_sha256", "0" * 64), None, False, "source_tree_sha256"),
        (("extension_eligibility.eligible_scenario_seed_count", 79), None, False, "eligibility"),
        (("extension_eligibility.eligible_scenario_seed_count", 80.0), None, False, "eligibility"),
        (("extension_eligibility.required_scenario_seed_count", 80.0), None, False, "eligibility"),
        (("extension_eligibility.canonical_state_hash_count", 239), None, False, "state hash"),
        (("extension_eligibility.canonical_state_hash_count", 240.0), None, False, "state hash"),
        (("extension_eligibility.state_hash_manifest_sha256", "0" * 64), None, False, "state hash manifest"),
        (None, ("decision", "STOP_SCALAR_SATURATED"), False, "cross-check"),
        (None, None, True, "bytes/hash"),
    ],
)
def test_extension_authorization_fails_closed_on_each_cross_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision_mutation: tuple[str, object] | None,
    manifest_mutation: tuple[str, object] | None,
    corrupt_fact: bool,
    match: str,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    seeds = tuple(range(10_000, 10_040))
    config = _run_initial_fixture(runner, monkeypatch, parent, seeds=seeds)
    decision = _write_checkpoint_analysis(
        parent,
        seeds,
        decision_mutation=decision_mutation,
        manifest_mutation=manifest_mutation,
        corrupt_decision_fact=corrupt_fact,
    )
    with pytest.raises(RuntimeError, match=match):
        runner.authorize_extension(
            parent,
            decision,
            config=config,
            source_hash=runner.source_tree_sha256(),
            experiment_hash=runner.experiment_tree_sha256(),
        )


def test_extension_authorization_rejects_equal_float_summary_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    seeds = tuple(range(10_000, 10_040))
    config = _run_initial_fixture(runner, monkeypatch, parent, seeds=seeds)
    decision = _write_checkpoint_analysis(parent, seeds)
    manifest_path = parent / "checkpoint_analysis" / "phase0c_summary_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fact = manifest["files"]["phase0c_decision.json"]
    fact["bytes"] = float(fact["bytes"])
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="bytes"):
        runner.authorize_extension(
            parent,
            decision,
            config=config,
            source_hash=runner.source_tree_sha256(),
            experiment_hash=runner.experiment_tree_sha256(),
        )


def test_partial_active_parent_cannot_be_authorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    seeds = tuple(range(10_000, 10_040))
    config = _run_initial_fixture(
        runner,
        monkeypatch,
        parent,
        seeds=seeds,
        partial_seed=10_039,
    )
    decision = _write_checkpoint_analysis(parent, seeds)
    with pytest.raises(RuntimeError, match="eligibility"):
        runner.authorize_extension(
            parent,
            decision,
            config=config,
            source_hash=runner.source_tree_sha256(),
            experiment_hash=runner.experiment_tree_sha256(),
        )


@pytest.mark.parametrize("relationship", ["equal", "descendant", "ancestor", "symlink"])
def test_extension_rejects_resolved_parent_output_overlap_before_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relationship: str,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "sentinel").write_text("unchanged\n")
    if relationship == "equal":
        output = parent
    elif relationship == "descendant":
        output = parent / "extension"
    elif relationship == "ancestor":
        output = tmp_path
    else:
        output = tmp_path / "parent-alias"
        output.symlink_to(parent, target_is_directory=True)
    base = ExperimentConfig.from_yaml(CONFIG_PATH)
    config = base.with_overrides(devices=("cpu",), output_dir=output)
    monkeypatch.setattr(
        runner,
        "authorize_extension",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authorization must not run for overlapping paths")
        ),
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    with pytest.raises(ValueError, match="overlap"):
        runner.run_config(
            config,
            output,
            mode="extension-8sp",
            workers_per_device=1,
            candidate_chunk_size=16,
            resume=False,
            parent_dir=parent,
            decision_json=parent / "checkpoint_analysis" / "phase0c_decision.json",
        )

    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_extension_run_passes_authenticated_parents_and_writes_exact_two_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    parent = tmp_path / "parent"
    extension = tmp_path / "extension"
    seeds = tuple(range(10_000, 10_040))
    parent_config = _run_initial_fixture(runner, monkeypatch, parent, seeds=seeds)
    decision = _write_checkpoint_analysis(parent, seeds)
    before = {path.relative_to(parent): path.read_bytes() for path in parent.rglob("*") if path.is_file()}
    calls: list[dict[str, object]] = []

    def fake_extension(
        _config: ExperimentConfig,
        *,
        seed: int,
        device: str,
        pair4_states: dict[str, tuple[SearchState, ...]],
        pair4_state_sha256: dict[str, tuple[str, ...]],
        extension_eligible: dict[str, bool],
        **_kwargs: object,
    ) -> SeedResult:
        calls.append(
            {
                "seed": seed,
                "names": {
                    scenario: tuple(state.start_name for state in pair4_states[scenario])
                    for scenario in SCENARIOS
                },
                "hash_counts": {
                    scenario: len(pair4_state_sha256[scenario]) for scenario in SCENARIOS
                },
                "eligible": extension_eligible,
            }
        )
        return _extension_result(seed, device)

    _InlineExecutor.instances = []
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(runner, "run_phase0c_extension_seed", fake_extension)
    config = parent_config.with_overrides(output_dir=extension)
    runner.run_config(
        config,
        extension,
        mode="extension-8sp",
        workers_per_device=1,
        candidate_chunk_size=16,
        resume=False,
        parent_dir=parent,
        decision_json=decision,
    )

    assert len(calls) == 40
    assert calls[0]["names"] == {scenario: START_NAMES for scenario in SCENARIOS}
    assert calls[0]["hash_counts"] == {scenario: 3 for scenario in SCENARIOS}
    assert calls[0]["eligible"] == {scenario: True for scenario in SCENARIOS}
    records = pd.read_csv(extension / "seed_10000" / "records.csv")
    assert len(records) == 2
    assert set(zip(records["scenario"], records["method_id"], strict=True)) == {
        ("standard", "joint_8SP"),
        ("tail_shift", "joint_8SP"),
    }
    runner.validate_study_manifest(extension, expected_kind="extension-8sp")
    execution = json.loads((extension / "study_metadata.json").read_text())["execution"]
    assert execution["parent_study_manifest_sha256"] == _sha256(parent / "study_manifest.json")
    assert len(execution["checkpoint_decision_sha256"]) == 64
    after = {path.relative_to(parent): path.read_bytes() for path in parent.rglob("*") if path.is_file()}
    assert after == before
