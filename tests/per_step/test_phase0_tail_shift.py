from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scpcp.config import ExperimentConfig, SampleConfig, SyntheticConfig
from scpcp.experiment import _prepare_task
from scpcp import simulator
from scpcp.simulator import (
    SyntheticBehaviorPolicy,
    SyntheticTreatmentEnvironment,
    rollout,
)


STANDARD_STATES = torch.tensor(
    [
        [
            [
                2.156278133392334,
                0.04113689064979553,
                -0.45599058270454407,
                -0.7175348997116089,
                0.0810621827840805,
                -0.7146333456039429,
            ],
            [
                1.45846688747406,
                0.23923666775226593,
                -0.6207395195960999,
                -0.921209454536438,
                -0.19819921255111694,
                -0.3361726999282837,
            ],
            [
                0.44229209423065186,
                0.7917270660400391,
                -0.13160818815231323,
                -0.7346323728561401,
                -0.17977356910705566,
                0.1564132124185562,
            ],
        ],
        [
            [
                0.17794644832611084,
                -0.2131853997707367,
                -1.429450511932373,
                1.3921992778778076,
                -1.1935831308364868,
                1.5205618143081665,
            ],
            [
                -0.949155330657959,
                0.23260179162025452,
                -0.9614777565002441,
                0.6539797782897949,
                -0.8766911029815674,
                1.1509315967559814,
            ],
            [
                -0.8202700614929199,
                0.8598389625549316,
                -1.0766420364379883,
                0.7255710959434509,
                -0.8373236656188965,
                0.42593321204185486,
            ],
        ],
    ],
    dtype=torch.float32,
)
STANDARD_ACTIONS = torch.tensor([[0, 1], [2, 1]], dtype=torch.int64)
STANDARD_OUTCOMES = torch.tensor(
    [
        [
            [1.45846688747406, 0.23923666775226593],
            [0.44229209423065186, 0.7917270660400391],
        ],
        [
            [-0.949155330657959, 0.23260179162025452],
            [-0.8202700614929199, 0.8598389625549316],
        ],
    ],
    dtype=torch.float32,
)


def _tail_shift_environment() -> object:
    config = SyntheticConfig(scenario="tail_shift")
    ExperimentConfig(synthetic=config).validate()
    return simulator.TailShiftTreatmentEnvironment(config)


def test_standard_rollout_is_bitwise_regression_identical() -> None:
    batch = rollout(
        SyntheticTreatmentEnvironment(SyntheticConfig()),
        SyntheticBehaviorPolicy(),
        n=2,
        horizon=2,
        seed=12345,
        device="cpu",
    )

    assert torch.equal(batch.states, STANDARD_STATES)
    assert torch.equal(batch.actions, STANDARD_ACTIONS)
    assert torch.equal(batch.outcomes, STANDARD_OUTCOMES)


def test_tail_shift_state_exposes_binary_difficulty() -> None:
    environment = _tail_shift_environment()
    generator = torch.Generator().manual_seed(11)

    state = environment.initial_state(512, generator, torch.device("cpu"))

    assert state.shape == (512, 7)
    assert set(state[:, 6].tolist()) == {0.0, 1.0}


def test_treatment_changes_next_difficulty_probability() -> None:
    environment = _tail_shift_environment()
    state = torch.zeros((2, 7))
    action = torch.tensor([0, 2])

    probability = environment.difficulty_probability(state, action)

    expected = torch.tensor([0.11920291930437088, 0.037326887249946594])
    assert torch.allclose(probability, expected)
    assert probability[1] < probability[0]


def test_difficult_state_has_heavier_residual_tail() -> None:
    environment = _tail_shift_environment()
    n = 10_000
    easy_state = torch.zeros((n, 7))
    difficult_state = easy_state.clone()
    difficult_state[:, 6] = 1.0
    action = torch.zeros(n, dtype=torch.int64)

    _, easy_outcome = environment.step(
        easy_state,
        action,
        torch.Generator().manual_seed(29),
    )
    _, difficult_outcome = environment.step(
        difficult_state,
        action,
        torch.Generator().manual_seed(29),
    )

    easy_tail = torch.quantile(easy_outcome.abs(), 0.99)
    difficult_tail = torch.quantile(difficult_outcome.abs(), 0.99)
    assert difficult_tail > 2.0 * easy_tail


def test_tail_shift_fixed_seed_is_reproducible() -> None:
    environment = _tail_shift_environment()
    policy = SyntheticBehaviorPolicy()

    first = rollout(
        environment,
        policy,
        n=32,
        horizon=4,
        seed=73,
        device="cpu",
    )
    second = rollout(
        environment,
        policy,
        n=32,
        horizon=4,
        seed=73,
        device="cpu",
    )

    assert torch.equal(first.states, second.states)
    assert torch.equal(first.actions, second.actions)
    assert torch.equal(first.outcomes, second.outcomes)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"scenario": "unknown"}, "scenario"),
        ({"difficulty_initial_probability": -0.1}, "probabilities"),
        ({"tail_contamination_probability": 1.1}, "probabilities"),
        ({"tail_scale": 0.0}, "tail_scale"),
    ],
)
def test_tail_shift_configuration_validation(
    updates: dict[str, object],
    message: str,
) -> None:
    synthetic = replace(SyntheticConfig(), **updates)
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(synthetic=synthetic).validate()


def test_prepare_task_routes_only_tail_shift_scenario() -> None:
    common = ExperimentConfig(samples=SampleConfig(logged=100), horizon=2)
    standard = _prepare_task(common, seed=5, device="cpu")
    tail_shift = _prepare_task(
        replace(
            common,
            synthetic=replace(common.synthetic, scenario="tail_shift"),
        ),
        seed=5,
        device="cpu",
    )

    assert isinstance(standard.environment, SyntheticTreatmentEnvironment)
    assert isinstance(tail_shift.environment, simulator.TailShiftTreatmentEnvironment)
