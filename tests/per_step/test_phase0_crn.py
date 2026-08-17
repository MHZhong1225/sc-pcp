from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from scpcp.config import SyntheticConfig
from scpcp import simulator


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


def _environments() -> list[object]:
    return [
        simulator.SyntheticTreatmentEnvironment(SyntheticConfig()),
        simulator.TailShiftTreatmentEnvironment(SyntheticConfig(scenario="tail_shift")),
    ]


def _step_kwargs(bundle: object, time: int = 0) -> dict[str, torch.Tensor]:
    return {
        "shared": bundle.shared_normal[time],
        "independent": bundle.independent_normal[time],
        "innovations": bundle.innovation_normal[time],
        "difficulty_uniform": bundle.difficulty_uniform[time],
        "contamination_uniform": bundle.contamination_uniform[time],
    }


def test_noise_bundle_is_reproducible_and_has_expected_shapes() -> None:
    n, horizon, seed = 5, 3, 41
    first = simulator.make_synthetic_noise_bundle(
        n=n,
        horizon=horizon,
        seed=seed,
        device="cpu",
    )
    second = simulator.make_synthetic_noise_bundle(
        n=n,
        horizon=horizon,
        seed=seed,
        device="cpu",
    )

    assert first.seed == seed
    assert first.initial_normal.shape == (n, 6)
    assert first.initial_difficulty_uniform.shape == (n,)
    assert first.action_uniform.shape == (horizon, n)
    assert first.shared_normal.shape == (horizon, n)
    assert first.independent_normal.shape == (horizon, n)
    assert first.innovation_normal.shape == (horizon, n, 4)
    assert first.difficulty_uniform.shape == (horizon, n)
    assert first.contamination_uniform.shape == (horizon, n)

    generator = torch.Generator().manual_seed(seed)
    expected_initial = torch.stack(
        [torch.randn(n, generator=generator) for _ in range(6)],
        dim=1,
    )
    expected = {
        "initial_normal": expected_initial,
        "initial_difficulty_uniform": torch.rand(n, generator=generator),
        "action_uniform": torch.rand((horizon, n), generator=generator),
        "shared_normal": torch.randn((horizon, n), generator=generator),
        "independent_normal": torch.randn((horizon, n), generator=generator),
        "innovation_normal": torch.randn((horizon, n, 4), generator=generator),
        "difficulty_uniform": torch.rand((horizon, n), generator=generator),
        "contamination_uniform": torch.rand((horizon, n), generator=generator),
    }
    for name, expected_tensor in expected.items():
        assert torch.equal(getattr(first, name), expected_tensor)
        assert torch.equal(getattr(first, name), getattr(second, name))

    standard_state = _environments()[0].initial_state_from_noise(first)
    expected_standard = expected_initial.clone()
    expected_standard[:, 0] = 1.3 + 0.6 * expected_standard[:, 0]
    expected_standard[:, 1] = 0.4 + 0.3 * expected_standard[:, 1]
    assert torch.equal(standard_state, expected_standard)

    tail_environment = _environments()[1]
    tail_state = tail_environment.initial_state_from_noise(first)
    expected_difficulty = (
        first.initial_difficulty_uniform
        < tail_environment.config.difficulty_initial_probability
    ).to(expected_standard.dtype)
    assert torch.equal(tail_state[:, :6], expected_standard)
    assert torch.equal(tail_state[:, 6], expected_difficulty)

    with pytest.raises(FrozenInstanceError):
        first.seed = seed + 1


def test_inverse_cdf_action_sampling_uses_shared_uniforms() -> None:
    probabilities = torch.tensor(
        [
            [[0.20, 0.30, 0.50], [0.60, 0.30, 0.10]],
            [[0.10, 0.20, 0.6999998], [0.20, 0.30, 0.50]],
        ],
        dtype=torch.float32,
    )
    almost_one = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    patient_uniforms = torch.tensor([0.10, 0.6001])
    shared_uniforms = patient_uniforms.expand(2, -1).clone()
    shared_uniforms[1, 0] = almost_one
    torch.manual_seed(909)
    rng_state = torch.random.get_rng_state().clone()

    actions = simulator.inverse_cdf_actions(probabilities, shared_uniforms)

    assert torch.equal(actions, torch.tensor([[0, 1], [2, 2]]))
    assert actions.max().item() < probabilities.shape[-1]
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.equal(
        actions,
        simulator.inverse_cdf_actions(probabilities, shared_uniforms),
    )
    with pytest.raises(ValueError, match="uniforms must match"):
        simulator.inverse_cdf_actions(probabilities, shared_uniforms[:, 0])


@pytest.mark.parametrize("environment", _environments())
def test_step_from_noise_is_pure(environment: object) -> None:
    bundle = simulator.make_synthetic_noise_bundle(
        n=4,
        horizon=2,
        seed=73,
        device="cpu",
    )
    state = environment.initial_state_from_noise(bundle)
    action = torch.tensor([0, 1, 2, 1])
    original_state = state.clone()
    original_action = action.clone()
    noise = _step_kwargs(bundle)
    original_noise = {name: value.clone() for name, value in noise.items()}
    torch.manual_seed(918)
    rng_state = torch.random.get_rng_state().clone()

    first_state, first_outcome = environment.step_from_noise(
        state,
        action,
        **noise,
    )
    second_state, second_outcome = environment.step_from_noise(
        state,
        action,
        **noise,
    )

    assert torch.equal(first_state, second_state)
    assert torch.equal(first_outcome, second_outcome)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.equal(state, original_state)
    assert torch.equal(action, original_action)
    for name, value in noise.items():
        assert torch.equal(value, original_noise[name])


@pytest.mark.parametrize("environment", _environments())
def test_candidate_order_does_not_change_patient_noise(environment: object) -> None:
    bundle = simulator.make_synthetic_noise_bundle(
        n=4,
        horizon=1,
        seed=97,
        device="cpu",
    )
    state = environment.initial_state_from_noise(bundle)
    candidate_actions = torch.tensor(
        [
            [0, 1, 2, 0],
            [2, 0, 1, 2],
            [1, 2, 0, 1],
        ]
    )
    noise = _step_kwargs(bundle)

    torch.manual_seed(123)
    ordered = [
        environment.step_from_noise(state, action, **noise)
        for action in candidate_actions
    ]
    torch.manual_seed(123)
    reversed_order = [
        environment.step_from_noise(state, action, **noise)
        for action in candidate_actions.flip(0)
    ]

    for ordered_result, reversed_result in zip(ordered, reversed(reversed_order)):
        assert torch.equal(ordered_result[0], reversed_result[0])
        assert torch.equal(ordered_result[1], reversed_result[1])


def test_legacy_rollout_fixture_remains_exact() -> None:
    batch = simulator.rollout(
        simulator.SyntheticTreatmentEnvironment(SyntheticConfig()),
        simulator.SyntheticBehaviorPolicy(),
        n=2,
        horizon=2,
        seed=12345,
        device="cpu",
    )

    assert torch.equal(batch.states, STANDARD_STATES)
    assert torch.equal(batch.actions, STANDARD_ACTIONS)
    assert torch.equal(batch.outcomes, STANDARD_OUTCOMES)
