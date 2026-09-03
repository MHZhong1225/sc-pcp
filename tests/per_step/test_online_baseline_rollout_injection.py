from __future__ import annotations

from typing import Any

import pytest
import torch

from scpcp import baselines
from scpcp.data import TrajectoryBatch


def _batch(n: int, horizon: int) -> TrajectoryBatch:
    return TrajectoryBatch(
        states=torch.zeros((n, horizon + 1, 1)),
        actions=torch.zeros((n, horizon), dtype=torch.long),
        outcomes=torch.zeros((n, horizon, 1)),
        patient_ids=torch.arange(n),
    )


def _constant_scores(
    _model: object,
    states: torch.Tensor,
    _actions: torch.Tensor,
    _outcomes: torch.Tensor,
) -> torch.Tensor:
    return torch.full(states.shape[:2], 0.25)


def test_default_online_rollout_path_matches_explicit_original_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_rollout(
        environment: object,
        policy: object,
        **kwargs: Any,
    ) -> TrajectoryBatch:
        calls.append({"environment": environment, "policy": policy, **kwargs})
        return _batch(kwargs["n"], kwargs["horizon"])

    monkeypatch.setattr(baselines, "rollout", fake_rollout)
    monkeypatch.setattr("scpcp.scores.score_batch", _constant_scores)
    environment, policy = object(), object()
    common = {
        "alpha": 0.1,
        "gamma": 0.01,
        "rounds": 1,
        "total_rollouts": 7,
        "horizon": 2,
        "seed": 13,
        "device": "cpu",
    }
    default = baselines.aci_style_controller(
        environment,
        policy,
        object(),
        torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
        **common,
    )
    default_call = calls.pop()
    injected = baselines.aci_style_controller(
        environment,
        policy,
        object(),
        torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
        rollout_fn=fake_rollout,
        **common,
    )
    injected_call = calls.pop()

    assert default_call.keys() == injected_call.keys()
    assert default_call["environment"] is injected_call["environment"]
    assert default_call["policy"] is injected_call["policy"]
    for name in ("n", "horizon", "seed", "device"):
        assert default_call[name] == injected_call[name]
    assert torch.equal(default_call["q"], injected_call["q"])
    assert torch.equal(default.radius_by_time, injected.radius_by_time)
    assert default.target_deployments == injected.target_deployments == 7


@pytest.mark.parametrize("method", ("ACI", "SPCI", "PRC"))
def test_online_adapters_consume_exactly_two_thousand_deployments(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, torch.Tensor]] = []

    def fake_rollout(
        _environment: object,
        _policy: object,
        *,
        n: int,
        horizon: int,
        seed: int,
        device: str | torch.device,
        q: torch.Tensor,
    ) -> TrajectoryBatch:
        del device
        calls.append((n, seed, q.detach().cpu().clone()))
        return _batch(n, horizon)

    monkeypatch.setattr("scpcp.scores.score_batch", _constant_scores)
    initial_scores = torch.linspace(0.1, 1.0, 20)[:, None].expand(-1, 2).clone()
    common = {
        "rounds": 3,
        "total_rollouts": 2_000,
        "horizon": 2,
        "seed": 101,
        "device": "cpu",
        "rollout_fn": fake_rollout,
    }
    if method == "ACI":
        result = baselines.aci_style_controller(
            object(),
            object(),
            object(),
            initial_scores,
            alpha=0.1,
            gamma=0.01,
            **common,
        )
        expected_seeds = [101, 101 + 17_923, 101 + 2 * 17_923]
    elif method == "SPCI":
        result = baselines.multidim_spci_style_controller(
            object(),
            object(),
            object(),
            initial_scores,
            alpha=0.1,
            residual_window=1_000,
            **common,
        )
        expected_seeds = [101, 101 + 47_021, 101 + 2 * 47_021]
    else:
        result = baselines.prc_profile_scale(
            object(),
            object(),
            object(),
            1.0,
            torch.tensor([0.8, 1.0, 1.2]),
            torch.ones(2),
            alpha=0.1,
            delta=0.05,
            maximum_step=0.35,
            **common,
        )
        expected_seeds = [101, 101 + 61_103, 101 + 2 * 61_103]

    assert [n for n, _, _ in calls] == [667, 667, 666]
    assert [seed for _, seed, _ in calls] == expected_seeds
    assert all(q.shape == (2,) for _, _, q in calls)
    assert result.target_deployments == 2_000
    assert result.rounds == 3
