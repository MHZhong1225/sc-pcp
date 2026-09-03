from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_pac_grid_study.py"
    spec = importlib.util.spec_from_file_location("run_pac_grid_study", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested_grids() -> tuple[np.ndarray, np.ndarray]:
    base = np.geomspace(0.5, 2.0, 101).astype(np.float32)
    dense = np.empty(401, dtype=np.float32)
    for index in range(100):
        knots = np.geomspace(base[index], base[index + 1], 5).astype(np.float32)
        dense[4 * index : 4 * index + 4] = knots[:-1]
    dense[-1] = base[-1]
    dense[::4] = base
    return base, dense


def _write_seed(
    runner: object,
    output: Path,
    seed: int,
    *,
    base_coverage: np.ndarray,
    dense_coverage: np.ndarray,
    base_width: float,
    dense_width: float,
    lcb_error: float = 0.0,
    evaluation_seed_offset: int = 0,
    width_mean_roundoff: float = 0.0,
) -> tuple[np.ndarray, int]:
    horizon = len(base_coverage)
    seed_dir = output / f"seed_{seed:05d}"
    seed_dir.mkdir(parents=True)
    base_grid, dense_grid = _nested_grids()
    base_indices = np.arange(0, 401, 4)
    dense_points = np.full((401, horizon), 0.92, dtype=np.float32)
    dense_lcbs = np.full((401, horizon), 0.91, dtype=np.float32)
    base_points = dense_points[base_indices].copy()
    base_lcbs = dense_lcbs[base_indices].copy()
    if lcb_error:
        base_lcbs[0, 0] += lcb_error
    base_index = 62
    dense_index = 247
    profile = np.linspace(0.9, 1.1, horizon, dtype=np.float32)
    base_schedule = base_grid[base_index] * profile
    dense_schedule = dense_grid[dense_index] * profile
    base_stage_width = np.full(horizon, base_width, dtype=np.float32)
    dense_stage_width = np.full(horizon, dense_width, dtype=np.float32)
    evaluation_seed = 1_700_001 + seed

    methods = (
        (
            runner.BASE_METHOD,
            101,
            base_index,
            base_grid[base_index],
            base_coverage,
            base_stage_width,
        ),
        (
            runner.DENSE_METHOD,
            401,
            dense_index,
            dense_grid[dense_index],
            dense_coverage,
            dense_stage_width,
        ),
    )
    records = []
    for row_index, (method, grid_size, index, scale, coverage, width) in enumerate(methods):
        records.append(
            {
                "seed": seed,
                "method": method,
                "grid_size": grid_size,
                "target_coverage": 0.9,
                "confidence_level": 0.95,
                "delta": 0.05,
                "selection_available": True,
                "selection_status": "fixture",
                "selected_index": index,
                "selected_scale": float(scale),
                "estimated_min_coverage": 0.92,
                "lower_bound_min": 0.91,
                "fresh_worst_coverage": float(coverage.min()),
                "fresh_mean_coverage": float(coverage.mean()),
                "fresh_per_time_coverage": json.dumps(coverage.tolist()),
                "fresh_average_normalized_width": float(width.mean())
                + width_mean_roundoff,
                "fresh_per_time_normalized_width": json.dumps(width.tolist()),
                "fresh_target_met": bool(coverage.min() >= 0.9),
                "evaluation_seed": (
                    evaluation_seed + evaluation_seed_offset * row_index
                ),
                "evaluation_rollouts": 50_000,
                "certificate_type": "cluster ordered practical bootstrap LCB",
                "certificate_formal": False,
            }
        )
    pd.DataFrame(records).to_csv(seed_dir / "records.csv", index=False)
    np.savez_compressed(
        seed_dir / "surfaces.npz",
        base_grid=base_grid,
        dense_grid=dense_grid,
        base_indices_in_dense=base_indices,
        stage_profile=profile,
        base_point_estimates=base_points,
        dense_point_estimates=dense_points,
        base_lower_bounds=base_lcbs,
        dense_lower_bounds=dense_lcbs,
        base_estimated_widths=np.full(101, base_width, dtype=np.float32),
        dense_estimated_widths=np.full(401, dense_width, dtype=np.float32),
        base_selected_schedule=base_schedule,
        dense_selected_schedule=dense_schedule,
        base_fresh_coverage=base_coverage,
        dense_fresh_coverage=dense_coverage,
        base_fresh_width=base_stage_width,
        dense_fresh_width=dense_stage_width,
        dense_effective_sample_sizes=np.full((401, horizon), 100.0),
    )
    diagnostics = {
        "protocol": runner.PROTOCOL,
        "target": 0.9,
        "delta": 0.05,
        "base_grid_size": 101,
        "dense_grid_size": 401,
        "evaluation_rollouts": 50_000,
        "base_selected_index": base_index,
        "dense_selected_index": dense_index,
        "base_stopped_index": 61,
        "dense_stopped_index": 246,
        "maximum_base_weight_parity_error": 0.0,
        "maximum_base_point_parity_error": 0.0,
        "maximum_base_lcb_parity_error": abs(lcb_error),
        "maximum_base_width_parity_error": 0.0,
        "base_selection_matches_dense_base_knots": True,
        "minimum_dense_ess": 100.0,
        "maximum_dense_cap_hit_rate": 0.001,
    }
    (seed_dir / "metadata.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "source_tree_sha256": "fixture-source",
                "diagnostics": diagnostics,
                "config": {"fixture": True},
            }
        )
    )
    (seed_dir / "COMPLETE").write_text("complete\n")
    return base_schedule, base_index


def _write_reference(
    root: Path,
    seed_schedules: dict[int, tuple[np.ndarray, int]],
) -> None:
    root.mkdir()
    (root / "COMPLETE").write_text("complete\n")
    for seed, (schedule, index) in seed_schedules.items():
        seed_dir = root / f"seed_{seed:05d}"
        seed_dir.mkdir()
        np.savez_compressed(seed_dir / "surfaces.npz", E_schedule=schedule)
        (seed_dir / "metadata.json").write_text(
            json.dumps({"diagnostics": {"indices": {"e": index}}})
        )
        (seed_dir / "COMPLETE").write_text("complete\n")


def test_cli_defaults_and_acceptance_gate_are_frozen() -> None:
    runner = _load_runner()
    args = runner.build_parser().parse_args([])

    assert args.config == ROOT / "configs" / "phase0_oracle.yaml"
    assert args.workers_per_device == 1
    assert args.resume is False
    assert runner.parse_seeds(args.seeds, runner.DEFAULT_SEEDS) == tuple(range(20))
    gate = runner._acceptance_gate_definition()
    assert gate["maximum_dense_over_base_geometric_width_ratio"] == 0.995
    assert gate["maximum_dense_over_base_width_ratio_one_sided_95_ucb"] == 1.0
    assert gate["maximum_allowed_pooled_marginal_wsc_loss_paired_95_lcb"] == 0.002
    assert gate["maximum_base_lcb_parity_error"] == 1e-6
    assert gate["reference_E_schedule_tolerance_when_provided"] == 1e-4
    assert gate["maximum_dense_cap_hit_rate"] == 0.01
    assert gate["minimum_dense_effective_sample_size"] == 25.0


def test_validation_accepts_independent_lcb_roundoff_within_frozen_tolerance(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    _write_seed(
        runner,
        tmp_path,
        0,
        base_coverage=np.array([0.90, 0.91]),
        dense_coverage=np.array([0.90, 0.92]),
        base_width=2.0,
        dense_width=1.9,
        lcb_error=3.6e-7,
        width_mean_roundoff=1e-7,
    )

    assert runner.validate_seed_artifact(
        tmp_path / "seed_00000", 0, horizon=2
    ).is_dir()

    metadata_path = tmp_path / "seed_00000" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["diagnostics"]["maximum_base_lcb_parity_error"] = 1.1e-6
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="tolerance"):
        runner.validate_seed_artifact(tmp_path / "seed_00000", 0, horizon=2)


def test_validation_rejects_width_aggregation_difference_above_tolerance(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    _write_seed(
        runner,
        tmp_path,
        0,
        base_coverage=np.array([0.90, 0.91]),
        dense_coverage=np.array([0.90, 0.92]),
        base_width=2.0,
        dense_width=1.9,
        width_mean_roundoff=1.1e-6,
    )

    with pytest.raises(RuntimeError, match="average width"):
        runner.validate_seed_artifact(tmp_path / "seed_00000", 0, horizon=2)


def test_seed_jobs_interleave_devices_for_balanced_partial_wave() -> None:
    runner = _load_runner()

    worker_devices, jobs = runner._build_seed_jobs(
        tuple(range(20)),
        ("cuda:0", "cuda:1"),
        workers_per_device=4,
    )

    assert worker_devices == (
        "cuda:0",
        "cuda:1",
        "cuda:0",
        "cuda:1",
        "cuda:0",
        "cuda:1",
        "cuda:0",
        "cuda:1",
    )
    final_wave_devices = [device for _, seed, device in jobs if seed >= 16]
    assert final_wave_devices == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]


def test_summary_reports_pooled_metrics_ratio_dense_wider_and_reference_replay(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    schedules = {
        0: _write_seed(
            runner,
            tmp_path,
            0,
            base_coverage=np.array([0.88, 0.94]),
            dense_coverage=np.array([0.90, 0.92]),
            base_width=2.0,
            dense_width=1.0,
        ),
        1: _write_seed(
            runner,
            tmp_path,
            1,
            base_coverage=np.array([0.92, 0.90]),
            dense_coverage=np.array([0.90, 0.94]),
            base_width=4.0,
            dense_width=5.0,
        ),
    }
    reference = tmp_path / "reference"
    _write_reference(reference, schedules)

    summary = runner.write_summary(
        tmp_path,
        (0, 1),
        horizon=2,
        reference_decomposition=reference,
    )

    base = summary["methods"][runner.BASE_METHOD]
    dense = summary["methods"][runner.DENSE_METHOD]
    assert base["pooled_marginal_wsc"] == pytest.approx(0.90)
    assert base["mean_coverage"] == pytest.approx(0.91)
    assert dense["pooled_marginal_wsc"] == pytest.approx(0.90)
    assert dense["mean_coverage"] == pytest.approx(0.915)
    assert summary["paired_dense_over_base_width"][
        "geometric_mean_ratio"
    ] == pytest.approx(np.sqrt(0.5 * 1.25))
    assert summary["dense_wider_seeds"] == [1]
    assert summary["availability"]["paired_available_count"] == 2
    assert summary["parity"]["maximum_base_lcb_parity_error"] == 0.0
    assert summary["reference_decomposition_replay"][
        "replay_within_tolerance"
    ] is True
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "summary.csv").is_file()


def test_validation_rejects_nonshared_crn(tmp_path: Path) -> None:
    runner = _load_runner()
    _write_seed(
        runner,
        tmp_path,
        0,
        base_coverage=np.array([0.90, 0.91]),
        dense_coverage=np.array([0.90, 0.92]),
        base_width=2.0,
        dense_width=1.9,
        evaluation_seed_offset=1,
    )

    with pytest.raises(RuntimeError, match="CRN"):
        runner.validate_seed_artifact(tmp_path / "seed_00000", 0, horizon=2)


def test_resume_discovery_fails_closed_on_partial_atomic_directory(tmp_path: Path) -> None:
    runner = _load_runner()
    _write_seed(
        runner,
        tmp_path,
        0,
        base_coverage=np.array([0.90, 0.91]),
        dense_coverage=np.array([0.90, 0.92]),
        base_width=2.0,
        dense_width=1.9,
    )
    partial = tmp_path / ".seed_00001-crashed"
    partial.mkdir()

    with pytest.raises(RuntimeError, match="partial atomic"):
        runner._validated_existing_seeds(
            tmp_path,
            (0, 1),
            expected_source_hash="fixture-source",
            expected_config_hash=runner.canonical_config_sha256({"fixture": True}),
            horizon=2,
        )


def test_reference_schedule_or_index_difference_is_reported(tmp_path: Path) -> None:
    runner = _load_runner()
    schedule, index = _write_seed(
        runner,
        tmp_path,
        0,
        base_coverage=np.array([0.90, 0.91]),
        dense_coverage=np.array([0.90, 0.92]),
        base_width=2.0,
        dense_width=1.9,
    )
    reference = tmp_path / "reference"
    changed = schedule.copy()
    changed[0] += 2e-4
    _write_reference(reference, {0: (changed, index + 1)})

    replay = runner._compare_reference_decomposition(
        tmp_path,
        reference,
        (0,),
        horizon=2,
    )

    assert replay["replay_within_tolerance"] is False
    assert replay["maximum_absolute_schedule_error"] > 1e-4
    assert replay["index_mismatch_count"] == 1
