"""Behavior-anchored prediction-mediated treatment policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from config import PolicyConfig
from outcome_model import GaussianOutcomeModel
from scores import ConformalRegion


@dataclass(frozen=True)
class BehaviorAnchoredPolicy:
    r"""\(\pi_q(a\mid s)\propto\mu_{ref}(a\mid s)\exp[-\eta J_q(s,a)]\)."""

    outcome_model: GaussianOutcomeModel
    reference_policy: object
    config: PolicyConfig
    region: ConformalRegion
    tilt: float = 1.0

    @property
    def n_actions(self) -> int:
        return self.outcome_model.n_actions

    @torch.no_grad()
    def clinical_cost(self, states: Tensor, q_grid: Tensor) -> Tensor:
        """Return worst-case clinical cost with shape [N, K, A]."""

        means, scales = self.outcome_model.predict_all_actions(states)
        action_costs = self._action_costs(states)
        baseline = (
            self.config.disease_weight * means[..., 0]
            + self.config.toxicity_weight * means[..., 1]
            + action_costs
        )
        uncertainty = self.region.uncertainty_penalty(
            scales,
            self.config.disease_weight,
            self.config.toxicity_weight,
        )
        return baseline[:, None, :] + q_grid.to(states)[None, :, None] * uncertainty[:, None, :]

    @torch.no_grad()
    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        if q is None:
            raise ValueError("a radius is required for a prediction-mediated policy")
        q_tensor = torch.as_tensor(q, dtype=states.dtype, device=states.device)
        if q_tensor.ndim == 0:
            return self.probabilities_for_grid(states, q_tensor[None])[:, 0, :]
        if q_tensor.ndim == 1 and len(q_tensor) == len(states):
            return self._probabilities_per_row(states, q_tensor)
        if q_tensor.ndim == 1:
            return self.probabilities_for_grid(states, q_tensor)
        raise ValueError("q must be scalar, [N], or [K]")

    @torch.no_grad()
    def probabilities_for_grid(self, states: Tensor, q_grid: Tensor) -> Tensor:
        if q_grid.ndim != 1:
            raise ValueError("q_grid must have shape [K]")
        reference = self.reference_policy.probabilities(states).clamp_min(1e-12)
        costs = self.clinical_cost(states, q_grid)
        return _ratio_capped_tilt(
            reference[:, None, :], costs, self.tilt / self.config.temperature, self.config.policy_ratio_cap
        )

    @torch.no_grad()
    def _probabilities_per_row(self, states: Tensor, radii: Tensor) -> Tensor:
        reference = self.reference_policy.probabilities(states).clamp_min(1e-12)
        means, scales = self.outcome_model.predict_all_actions(states)
        costs = (
            self.config.disease_weight * means[..., 0]
            + self.config.toxicity_weight * means[..., 1]
            + self._action_costs(states)
            + radii[:, None]
            * self.region.uncertainty_penalty(
                scales,
                self.config.disease_weight,
                self.config.toxicity_weight,
            )
        )
        return _ratio_capped_tilt(
            reference, costs, self.tilt / self.config.temperature, self.config.policy_ratio_cap
        )

    def _action_costs(self, states: Tensor) -> Tensor:
        if len(self.config.action_costs) != self.n_actions:
            raise ValueError("policy.action_costs must provide one cost per action")
        return torch.as_tensor(self.config.action_costs, dtype=states.dtype, device=states.device)


def _ratio_capped_tilt(reference: Tensor, costs: Tensor, tilt: float, cap: float) -> Tensor:
    """Exponential tilt projected onto pi(a|s)/mu_ref(a|s) <= cap."""

    shifted = costs - costs.amin(dim=-1, keepdim=True)
    weights = torch.exp((-tilt * shifted).clamp_min(-60.0)).clamp_min(1e-12)
    lower = torch.zeros_like(weights[..., :1])
    upper = weights.amax(dim=-1, keepdim=True).clamp_min(1e-12)
    # For each state, choose z such that
    #   sum_a mu_ref(a|s) min(exp(-eta J_a) / z, cap) = 1.
    # The previous implementation normalized the clipped vector *after* this
    # search, which can silently violate the advertised overlap cap.  Returning
    # the bisection solution directly preserves both unit mass (to numerical
    # precision) and pi / mu_ref <= cap.
    for _ in range(64):
        normalizer = (lower + upper) / 2.0
        ratio = torch.minimum(weights / normalizer, torch.full_like(weights, cap))
        mass = (reference * ratio).sum(dim=-1, keepdim=True)
        lower = torch.where(mass > 1.0, normalizer, lower)
        upper = torch.where(mass > 1.0, upper, normalizer)
    normalizer = (lower + upper) / 2.0
    ratio = torch.minimum(weights / normalizer, torch.full_like(weights, cap))
    return reference * ratio
