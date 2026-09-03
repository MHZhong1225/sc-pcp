from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "render_controlled_prefix_benchmark",
    ROOT / "tools" / "render_controlled_prefix_benchmark.py",
)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def _method_row(seed: int, gamma: float, method: str) -> dict[str, object]:
    seed_offset = (seed - 9.5) * 1e-4
    method_offset = 0.002 if method == "SC-PCP" else 0.0
    source = np.full(12, 0.9 + seed_offset)
    target = source + gamma * 0.001 + method_offset
    radii = np.full(12, 2.0 + method_offset)
    target_q90 = np.full(12, 2.0)
    return {
        "radii": radii.tolist(),
        "source_coverage": source.tolist(),
        "target_coverage": target.tolist(),
        "coverage_gap": (target - source).tolist(),
        "source_q90": np.full(12, 2.0).tolist(),
        "target_q90": target_q90.tolist(),
        "q90_relative_gap": np.full(12, gamma * 0.01).tolist(),
        "target_normalized_width": (radii * 1.5).tolist(),
        "prefix_ess_fraction": np.full(12, 0.6).tolist(),
        "maximum_normalized_weight_share": np.full(12, 0.002).tolist(),
        "raw_log_weight_span": np.full(12, 1.0).tolist(),
        "policy_tv_on_source_states": np.full(12, 0.1).tolist(),
        "source_difficulty": np.full(12, 0.4).tolist(),
        "target_difficulty": np.full(12, 0.5).tolist(),
        "donor_kernel_ess_fraction_min": 0.5,
        "donor_probability_max": 0.02,
    }


def _rows() -> list[dict[str, object]]:
    rows = []
    for seed in range(20):
        for gamma in REPORT.GAMMAS:
            rows.append(
                {
                    "seed": seed,
                    "gamma": gamma,
                    "q_low": 1.0,
                    "q_high": 3.0,
                    "selection_minimum_ess_fraction": 0.5,
                    "selection_minimum_candidate_ess_fraction": 0.4,
                    "selection_selected_endpoint": False,
                    "methods": {
                        method: _method_row(seed, gamma, method)
                        for method in REPORT.METHODS
                    },
                }
            )
    return rows


def test_analysis_uses_worst_stage_mean_and_paired_width_ratio() -> None:
    analysis = REPORT.analyze(_rows())

    cell = analysis["-2"]
    assert cell["methods"]["Standard CP"]["target_wsc"] == pytest.approx(0.898)
    assert cell["methods"]["SC-PCP"]["target_wsc"] == pytest.approx(0.900)
    assert cell["standard_late_coverage_gap"] == pytest.approx(-0.002)
    assert cell["scpcp_to_standard_width_ratio"] == pytest.approx(1.001)
    assert len(cell["methods"]["SC-PCP"]["target_wsc_simultaneous_band"]) == 2


def test_source_table_keeps_development_and_confirmation_separate() -> None:
    analysis = REPORT.analyze(_rows())
    source = REPORT.make_source_rows(analysis, analysis)

    assert len(source) == 10
    assert {row["role"] for row in source} == {"development20", "confirm20"}
    assert sum(row["role"] == "confirm20" for row in source) == 5


def test_row_validation_rejects_inconsistent_coverage_gap() -> None:
    row = _rows()[0]
    row["methods"]["Standard CP"]["coverage_gap"][0] = 1.0

    with pytest.raises(RuntimeError, match="coverage gap"):
        REPORT.validate_row(row, seed=0, gamma=-4.0)
