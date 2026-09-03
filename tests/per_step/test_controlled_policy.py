from __future__ import annotations

import pytest
import torch

from scpcp.controlled_policy import ControlledMixturePolicy


class _ReferencePolicy:
    n_actions = 2

    def probabilities(self, states: torch.Tensor, q: object = None) -> torch.Tensor:
        return states.new_tensor((0.7, 0.3)).expand(len(states), -1)


class _AlternativePolicy:
    def probabilities(self, states: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        assert torch.equal(q, torch.zeros(len(states), dtype=states.dtype, device=states.device))
        return states.new_tensor((0.1, 0.9)).expand(len(states), -1)


def _policy(response: float = 0.5) -> ControlledMixturePolicy:
    return ControlledMixturePolicy(
        logging_policy=_ReferencePolicy(),
        alternative_policy=_AlternativePolicy(),
        radius_low=1.0,
        radius_high=3.0,
        maximum_response=response,
    )


def test_zero_response_is_the_exact_no_feedback_control() -> None:
    states = torch.zeros(4, 3)
    result = _policy(0.0).probabilities(states, 2.0)

    assert torch.equal(result, _ReferencePolicy().probabilities(states))


def test_endpoint_normalization_has_the_declared_meaning() -> None:
    states = torch.zeros(2, 3)
    policy = _policy(0.5)
    reference = _ReferencePolicy().probabilities(states)
    alternative = _AlternativePolicy().probabilities(states, torch.zeros(2))

    assert torch.equal(policy.probabilities(states, 1.0), reference)
    assert torch.allclose(policy.probabilities(states, 3.0), 0.5 * reference + 0.5 * alternative)


def test_per_row_radii_preserve_probability_mass() -> None:
    states = torch.zeros(3, 3)
    result = _policy().probabilities(states, torch.tensor((1.0, 2.0, 3.0)))

    assert torch.allclose(result.sum(dim=1), torch.ones(3))
    assert torch.all(result[0] == torch.tensor((0.7, 0.3)))
    assert torch.all(result[2] == torch.tensor((0.4, 0.6)))


def test_grid_probabilities_match_individual_radius_queries() -> None:
    states = torch.zeros(3, 3)
    radii = torch.tensor((1.0, 1.5, 2.0, 3.0))
    policy = _policy()

    result = policy.probabilities_for_grid(states, radii)
    expected = torch.stack([policy.probabilities(states, radius) for radius in radii], dim=1)

    assert result.shape == (len(states), len(radii), policy.n_actions)
    assert torch.allclose(result.sum(dim=2), torch.ones((len(states), len(radii))))
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"radius_low": 1.0, "radius_high": 1.0, "maximum_response": 0.5},
        {"radius_low": 1.0, "radius_high": 2.0, "maximum_response": 1.1},
    ),
)
def test_invalid_response_parameters_fail_at_the_public_boundary(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ControlledMixturePolicy(
            logging_policy=_ReferencePolicy(),
            alternative_policy=_AlternativePolicy(),
            **kwargs,
        )
