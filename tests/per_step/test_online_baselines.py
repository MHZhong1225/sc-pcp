from __future__ import annotations

import pytest
import torch

import scpcp.experiment as experiment
from scpcp.baselines import OnlineBaselineResult
from scpcp.config import BaselineConfig, ExperimentConfig, SampleConfig


def test_default_config_is_valid_and_uses_robust_cot_loss() -> None:
    config = ExperimentConfig()

    config.validate()

    assert config.cot.loss == "huber"
    assert config.cot.weight_cap >= config.cot.rho_cap * config.policy.policy_ratio_cap


def test_config_rejects_an_online_budget_too_small_for_the_round_count() -> None:
    config = ExperimentConfig(
        samples=SampleConfig(online_rollouts=2),
        baselines=BaselineConfig(online_rounds=3),
    )

    with pytest.raises(ValueError, match="online_rollouts must be at least online_rounds"):
        config.validate()


def test_adaptation_target_comparison_accepts_float32_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adaptation = OnlineBaselineResult(
        radius_by_time=torch.tensor([0.5, 0.6]),
        target_deployments=10,
        rounds=2,
        adaptation_per_time_coverage=torch.tensor([0.9, 1.0]),
        adaptation_round_worst_coverage=(0.9, 1.0),
        adaptation_pathwise_coverage=0.8,
    )
    monkeypatch.setattr(
        experiment,
        "per_step_oracle_metrics",
        lambda *args, **kwargs: (torch.ones(2), object(), torch.ones(1, 2)),
    )
    monkeypatch.setattr(experiment, "_deployment_record", lambda *args, **kwargs: {})

    task = type("Task", (), {"environment": object()})()
    record = experiment._evaluate_stagewise_method(
        "online",
        adaptation,
        task,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),
        ExperimentConfig(horizon=2),
        seed=3,
        device="cpu",
    )

    assert record["adaptation_target_coverage"] == pytest.approx(0.9)
    assert record["adaptation_empirical_target_met"] is True
