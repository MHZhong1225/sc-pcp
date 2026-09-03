from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from scpcp.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_marginal_prefix_pilot.py"
    spec = importlib.util.spec_from_file_location(
        "run_marginal_prefix_pilot",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_max_t_band_uses_shared_resamples_and_fixed_original_se() -> None:
    runner = _load_runner()
    values = np.array(
        [
            [0.89, 0.91],
            [0.90, 0.93],
            [0.92, 0.90],
            [0.91, 0.92],
        ]
    )
    indices = np.array(
        [
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [0, 1, 2, 3],
            [3, 2, 1, 0],
        ]
    )

    lower, upper, standard_error, critical_lower, critical_upper = (
        runner._simultaneous_max_t_bands(values, indices)
    )

    estimate = values.mean(axis=0)
    expected_se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    bootstrap_mean = values[indices].mean(axis=1)
    expected_lower_critical = np.quantile(
        ((estimate - bootstrap_mean) / expected_se).max(axis=1),
        0.95,
    )
    expected_upper_critical = np.quantile(
        ((bootstrap_mean - estimate) / expected_se).max(axis=1),
        0.95,
    )
    assert np.array_equal(standard_error, expected_se)
    assert critical_lower == pytest.approx(expected_lower_critical)
    assert critical_upper == pytest.approx(expected_upper_critical)
    assert np.allclose(lower, estimate - expected_lower_critical * expected_se)
    assert np.allclose(upper, estimate + expected_upper_critical * expected_se)


def test_opportunity_recovery_uses_mean_log_width_and_fails_nonpositive_gap() -> None:
    runner = _load_runner()
    indices = np.array([[0, 1], [1, 0]])
    widths = {
        runner.PREFIX_IW: np.array([2.0**1.5, 2.0**1.5]),
        runner.A_ORACLE: np.array([2.0, 2.0]),
        runner.C_ORACLE: np.array([4.0, 4.0]),
        runner.STANDARD: np.array([2.5, 2.5]),
    }

    recovery = runner._opportunity_recovery(widths, indices)

    assert recovery["defined"] is True
    assert recovery["estimate"] == pytest.approx(0.5)
    assert recovery["one_sided_95_lower"] == pytest.approx(0.5)

    widths[runner.C_ORACLE] = widths[runner.A_ORACLE].copy()
    failed = runner._opportunity_recovery(widths, indices)
    assert failed["defined"] is False
    assert failed["one_sided_95_lower"] is None


def test_confirm_seed_override_is_explicit_and_precommitted() -> None:
    runner = _load_runner()
    reference = ExperimentConfig()

    development = runner._phase0_reference_config(
        reference,
        seeds=(0, 1),
        devices=("cpu",),
        study_role="development",
    )
    confirm = runner._phase0_reference_config(
        reference,
        seeds=runner.CONFIRM_SEEDS,
        devices=("cpu",),
        study_role="confirm",
    )

    assert development.seeds == reference.seeds
    assert confirm.seeds == runner.CONFIRM_SEEDS
    assert all(
        runner._confirm_protocol_gates("confirm", runner.CONFIRM_SEEDS).values()
    )
    assert not all(
        runner._confirm_protocol_gates("development", (0, 1)).values()
    )
    with pytest.raises(ValueError, match="precommitted"):
        runner._phase0_reference_config(
            reference,
            seeds=tuple(range(100, 200)),
            devices=("cpu",),
            study_role="confirm",
        )


def test_paired_width_ratio_reports_shared_bootstrap_q95() -> None:
    runner = _load_runner()
    numerator = np.array([1.0, 2.0, 4.0])
    denominator = np.array([1.0, 1.0, 2.0])
    indices = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [0, 1, 2],
        ]
    )

    result = runner._paired_width_ratio(numerator, denominator, indices)

    log_ratio = np.log(numerator / denominator)
    bootstrap = np.exp(log_ratio[indices].mean(axis=1))
    assert result["geometric_mean"] == pytest.approx(np.exp(log_ratio.mean()))
    assert result["one_sided_95_upper"] == pytest.approx(
        np.quantile(bootstrap, 0.95)
    )


def test_method_gate_decisions_use_the_exact_precommitted_boundaries() -> None:
    runner = _load_runner()
    ratios = {
        f"{runner.PREFIX_IW} / {runner.A_ORACLE}": {
            "one_sided_95_upper": 1.02
        },
        f"{runner.PREFIX_IW} / {runner.STANDARD}": {
            "one_sided_95_upper": 1.025
        },
        f"{runner.PREFIX_IW} / {runner.C_ORACLE}": {
            "one_sided_95_upper": 0.989
        },
    }
    passing = runner._method_gate_decisions(
        prefix_point_worst=0.9,
        prefix_simultaneous_upper=np.array([0.9, 0.901]),
        standard_simultaneous_upper=np.array([0.899, 0.902]),
        ratios=ratios,
        opportunity={"defined": True, "one_sided_95_lower": 0.5001},
        mean_coverage_comparison={"one_sided_95_upper": 0.003},
        endpoint_seeds=[],
        minimum_ess=50.0,
        minimum_candidate_ess=50.0,
    )

    assert all(passing.values())

    ratios[f"{runner.PREFIX_IW} / {runner.C_ORACLE}"][
        "one_sided_95_upper"
    ] = 0.99
    failing = runner._method_gate_decisions(
        prefix_point_worst=0.9,
        prefix_simultaneous_upper=np.array([0.9]),
        standard_simultaneous_upper=np.array([0.899]),
        ratios=ratios,
        opportunity={"defined": True, "one_sided_95_lower": 0.5},
        mean_coverage_comparison={"one_sided_95_upper": 0.0031},
        endpoint_seeds=[7],
        minimum_ess=49.9,
        minimum_candidate_ess=49.9,
    )
    assert failing["prefix_to_C_width_ratio_q95_below_0.99"] is False
    assert failing["opportunity_recovery_q05_above_0.50"] is False
    assert failing["mean_coverage_minus_A_q95_at_most_0.003"] is False
    assert failing["no_endpoint_selection"] is False
    assert failing["minimum_ess_at_least_50"] is False
