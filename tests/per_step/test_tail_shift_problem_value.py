from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_tail_shift_problem_value.py"
    spec = importlib.util.spec_from_file_location(
        "run_tail_shift_problem_value",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json(values: np.ndarray) -> str:
    return json.dumps([float(value) for value in values.tolist()])


def _phase0_seed(tmp_path: Path) -> Path:
    seed_dir = tmp_path / "seed_00000"
    seed_dir.mkdir()
    stage_grids = np.array(
        [[0.7, 0.8, 0.9], [0.75, 0.85, 0.95]],
        dtype=np.float32,
    )
    a_schedule = np.array([0.8, 0.95], dtype=np.float32)
    candidates = np.array(
        [[0.7, 0.75], [0.8, 0.85], [0.9, 0.95]],
        dtype=np.float32,
    )
    coverage = np.array(
        [[0.88, 0.91], [0.91, 0.90], [0.94, 0.95]],
        dtype=np.float32,
    )
    widths = np.array(
        [[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]],
        dtype=np.float32,
    )
    c_schedule = candidates[1]
    np.savez_compressed(
        seed_dir / "surfaces.npz",
        tail_shift_greedy_stage_grids=stage_grids,
        tail_shift_greedy_selected_schedule=a_schedule,
        tail_shift_profiled_candidate_schedules=candidates,
        tail_shift_profiled_candidate_coverage=coverage,
        tail_shift_profiled_candidate_normalized_width=widths,
        tail_shift_profiled_selected_schedule=c_schedule,
    )
    rows = []
    for method, schedule, tuning_coverage, tuning_width in (
        (
            "Greedy Sequential Oracle",
            a_schedule,
            np.array([0.90, 0.91], dtype=np.float32),
            np.array([1.2, 1.4], dtype=np.float32),
        ),
        (
            "Current Profiled Oracle",
            c_schedule,
            coverage[1],
            widths[1],
        ),
    ):
        rows.append(
            {
                "scenario": "tail_shift",
                "method": method,
                "seed": 0,
                "selection_available": True,
                "q_by_time": _json(schedule),
                "tuning_coverage": _json(tuning_coverage),
                "tuning_width": _json(tuning_width),
                "final_coverage": _json(np.array([0.9, 0.9], dtype=np.float32)),
                "final_wilson_lcb": _json(np.array([0.89, 0.89], dtype=np.float32)),
                "final_stage_width": _json(np.array([1.2, 1.3], dtype=np.float32)),
                "micro_normalized_width": 1.25,
                "patient_normalized_width": 1.25,
                "tuning_seed": 101,
                "evaluation_seed": 202,
                "n_rollouts": 50_000,
            }
        )
    pd.DataFrame(rows).to_csv(seed_dir / "records.csv", index=False)
    return seed_dir


def test_phase0_loader_recovers_exact_tail_shift_a_c_contract(tmp_path: Path) -> None:
    runner = _load_runner()
    seed_dir = _phase0_seed(tmp_path)

    recovered = runner.load_tail_shift_phase0(
        seed_dir,
        seed=0,
        horizon=2,
        target=0.9,
        rollouts=50_000,
    )

    assert np.array_equal(recovered["a_schedule"], np.array([0.8, 0.95], dtype=np.float32))
    assert np.array_equal(recovered["c_schedule"], np.array([0.8, 0.85], dtype=np.float32))
    assert recovered["tuning_seed"] == 101
    assert recovered["evaluation_seed"] == 202


def test_phase0_loader_fails_closed_when_record_schedule_is_tampered(tmp_path: Path) -> None:
    runner = _load_runner()
    seed_dir = _phase0_seed(tmp_path)
    records = pd.read_csv(seed_dir / "records.csv")
    records.loc[records["method"].eq("Current Profiled Oracle"), "q_by_time"] = "[0.8,0.9]"
    records.to_csv(seed_dir / "records.csv", index=False)

    with pytest.raises(RuntimeError, match="differs between records and surfaces"):
        runner.load_tail_shift_phase0(
            seed_dir,
            seed=0,
            horizon=2,
            target=0.9,
            rollouts=50_000,
        )


def test_summary_uses_minimum_of_seed_mean_stage_coverage_and_paired_ratios(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "BOOTSTRAP_RESAMPLES", 200)
    coverage = {
        runner.STANDARD: ([0.88, 0.92], [0.90, 0.94]),
        runner.A_ORACLE: ([0.90, 0.91], [0.92, 0.93]),
        runner.C_ORACLE: ([0.91, 0.92], [0.93, 0.94]),
    }
    widths = {
        runner.STANDARD: (1.0, 1.2),
        runner.A_ORACLE: (2.0, 2.4),
        runner.C_ORACLE: (1.5, 1.8),
    }
    for seed in (0, 1):
        seed_dir = tmp_path / f"seed_{seed:05d}"
        seed_dir.mkdir()
        surfaces = {}
        for method in runner.METHODS:
            prefix = runner._method_prefix(method)
            surfaces[f"{prefix}_coverage"] = np.asarray(coverage[method][seed])
            surfaces[f"{prefix}_stage_width"] = np.full(2, widths[method][seed])
        np.savez_compressed(seed_dir / "surfaces.npz", **surfaces)

    runner.write_summary(tmp_path, (0, 1), horizon=2)
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["methods"][runner.STANDARD]["marginal_worst_coverage"] == pytest.approx(0.89)
    assert summary["methods"][runner.STANDARD]["worst_stage_zero_based"] == 0
    assert summary["paired_width_ratios"][
        f"{runner.STANDARD} / {runner.A_ORACLE}"
    ]["geometric_mean"] == pytest.approx(0.5)


def test_reference_normalization_ignores_only_phase0_pairing_fields() -> None:
    runner = _load_runner()
    from dataclasses import replace
    from scpcp.config import ExperimentConfig

    base = ExperimentConfig()
    paired = replace(
        base,
        synthetic=replace(base.synthetic, scenario="tail_shift"),
        output_dir=Path("elsewhere"),
        paper=replace(base.paper, save_mechanism_diagonal=True),
    )
    different_delta = replace(
        base,
        certification=replace(base.certification, delta=0.10),
    )

    assert runner._normalized_reference_config(base) == runner._normalized_reference_config(paired)
    assert runner._normalized_reference_config(base) != runner._normalized_reference_config(
        different_delta
    )
