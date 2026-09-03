"""Policies for isolated controlled-feedback benchmarks.

This module is deliberately not part of the paper's canonical SC-PCP path.
It defines a transparent family in which a prediction radius controls how much
an otherwise fixed alternative clinical preference departs from the logging
policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ControlledMixturePolicy:
    r"""Blend a logging policy with a frozen, radius-independent alternative.

    The response coefficient is an endpoint-normalized sigmoid.  Thus the
    lower design radius always yields the logging policy, the upper radius
    yields a mixture with weight ``maximum_response``, and zero response is an
    exact no-feedback control.
    """

    logging_policy: object
    alternative_policy: object
    radius_low: float
    radius_high: float
    maximum_response: float
    sigmoid_slope: float = 4.0

    def __post_init__(self) -> None:
        if not self.radius_high > self.radius_low:
            raise ValueError("radius_high must exceed radius_low")
        if not 0.0 <= self.maximum_response <= 1.0:
            raise ValueError("maximum_response must lie in [0, 1]")
        if self.sigmoid_slope <= 0.0:
            raise ValueError("sigmoid_slope must be positive")

    @property
    def n_actions(self) -> int:
        value = getattr(self.logging_policy, "n_actions", None)
        if value is None:
            raise AttributeError("logging_policy must expose n_actions")
        return int(value)

    @torch.no_grad()
    def response_weight(self, radius: float | Tensor, *, like: Tensor) -> Tensor:
        """Return the mixture coefficient for a scalar or per-row radius."""

        value = torch.as_tensor(radius, dtype=like.dtype, device=like.device)
        normalized = ((value - self.radius_low) / (self.radius_high - self.radius_low)).clamp(0.0, 1.0)
        half = self.sigmoid_slope / 2.0
        lower = torch.sigmoid(value.new_tensor(-half))
        upper = torch.sigmoid(value.new_tensor(half))
        response = (torch.sigmoid(self.sigmoid_slope * (normalized - 0.5)) - lower) / (upper - lower)
        return self.maximum_response * response

    @torch.no_grad()
    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        if q is None:
            raise ValueError("a radius is required for the controlled policy")
        reference = self.logging_policy.probabilities(states)
        zero = torch.zeros(len(states), dtype=states.dtype, device=states.device)
        alternative = self.alternative_policy.probabilities(states, zero)
        weight = self.response_weight(q, like=states)
        if weight.ndim == 0:
            return (1.0 - weight) * reference + weight * alternative
        if weight.shape != (len(states),):
            raise ValueError("a vector radius must have one value per state")
        return (1.0 - weight[:, None]) * reference + weight[:, None] * alternative

    @torch.no_grad()
    def probabilities_for_grid(self, states: Tensor, radii: Tensor) -> Tensor:
        """Return action probabilities for every radius as ``[N, K, A]``."""

        grid = torch.as_tensor(radii, dtype=states.dtype, device=states.device)
        if grid.ndim != 1:
            raise ValueError("radii must be a one-dimensional grid")
        reference = self.logging_policy.probabilities(states)
        zero = torch.zeros(len(states), dtype=states.dtype, device=states.device)
        alternative = self.alternative_policy.probabilities(states, zero)
        weight = self.response_weight(grid, like=states)
        return (1.0 - weight[None, :, None]) * reference[:, None, :] + weight[None, :, None] * alternative[:, None, :]
