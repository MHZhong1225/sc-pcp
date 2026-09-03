from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_summary():
    path = ROOT / "scripts" / "summarize_conservatism_decomposition.py"
    spec = importlib.util.spec_from_file_location("summarize_conservatism_decomposition", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summary_uses_paired_log_ratios_and_distinct_coverage_semantics() -> None:
    module = _load_summary()
    widths = np.array(
        [
            [1.0, 1.1, 1.2, 1.4],
            [2.0, 2.2, 2.4, 2.8],
            [4.0, 4.4, 4.8, 5.6],
        ]
    )
    coverage = np.array(
        [
            [[0.90, 0.95], [0.91, 0.95], [0.92, 0.95], [0.93, 0.95]],
            [[0.95, 0.89], [0.95, 0.90], [0.95, 0.91], [0.95, 0.92]],
            [[0.91, 0.91], [0.92, 0.92], [0.93, 0.93], [0.94, 0.94]],
        ]
    )
    arrays = module.DecompositionArrays(
        seeds=np.arange(3),
        widths=widths,
        coverage=coverage,
        target=0.90,
        rollouts=50_000,
    )

    result = module.summarize_decomposition(arrays, n_resamples=1_000, seed=7)

    assert result["paired_width_ratio"]["C/A"] == pytest.approx(1.1)
    assert result["paired_width_ratio"]["D/C"] == pytest.approx(1.2 / 1.1)
    assert result["paired_width_ratio"]["E/D"] == pytest.approx(1.4 / 1.2)
    assert result["paired_width_ratio"]["E/A"] == pytest.approx(1.4)
    assert result["pooled_worst_stage_coverage"]["A"] == pytest.approx((0.95 + 0.89 + 0.91) / 3)
    assert result["mean_seedwise_worst_coverage"]["A"] == pytest.approx((0.90 + 0.89 + 0.91) / 3)
    assert result["fresh_target_met_count"]["A"] == 2
    components = result["mean_log_width_overhead"]
    assert (
        components["profile_C_minus_A"]
        + components["cot_D_minus_C"]
        + components["guard_E_minus_D"]
    ) == pytest.approx(components["total_E_minus_A"])


def test_summary_requires_paired_four_layer_arrays() -> None:
    module = _load_summary()
    arrays = module.DecompositionArrays(
        seeds=np.arange(2),
        widths=np.ones((2, 3)),
        coverage=np.ones((2, 3, 2)),
        target=0.90,
        rollouts=50_000,
    )

    with pytest.raises(ValueError, match="shape"):
        module.summarize_decomposition(arrays, n_resamples=1_000, seed=7)
