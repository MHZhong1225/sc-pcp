"""Conformal scores and prediction-region summaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from scpcp.outcome_model import GaussianOutcomeModel


@dataclass(frozen=True)
class ConformalRegion:
    """The final normalized-max box prediction region."""

    outcome_model: GaussianOutcomeModel

    @torch.no_grad()
    def score(self, states: Tensor, actions: Tensor, outcomes: Tensor) -> Tensor:
        mean, scale = self.outcome_model(states, actions)
        residual = (outcomes - mean) / scale
        return residual.abs().amax(dim=1)

    @torch.no_grad()
    def uncertainty_penalty(self, scales: Tensor, disease_weight: float, toxicity_weight: float) -> Tensor:
        return disease_weight * scales[..., 0] + toxicity_weight * scales[..., 1]

    @torch.no_grad()
    def log_volume(self, scales: Tensor, radius: Tensor) -> Tensor:
        volume = 4.0 * radius.square() * scales[..., 0] * scales[..., 1]
        return (volume + 1e-12).log()


def fit_conformal_region(model: GaussianOutcomeModel) -> ConformalRegion:
    return ConformalRegion(model)


@torch.no_grad()
def normalized_max_score(model: GaussianOutcomeModel, states: Tensor, actions: Tensor, outcomes: Tensor) -> Tensor:
    mean, scale = model(states, actions)
    return ((outcomes - mean).abs() / scale).amax(dim=1)


@torch.no_grad()
def score_batch(model: GaussianOutcomeModel | ConformalRegion, states: Tensor, actions: Tensor, outcomes: Tensor) -> Tensor:
    predictor = model.outcome_model if isinstance(model, ConformalRegion) else model
    device = next(predictor.parameters()).device
    states, actions, outcomes = states.to(device), actions.to(device), outcomes.to(device)
    n, horizon, state_dim = states.shape
    scorer = model.score if isinstance(model, ConformalRegion) else lambda x, a, y: normalized_max_score(model, x, a, y)
    scores = scorer(
        states.reshape(-1, state_dim),
        actions.reshape(-1),
        outcomes.reshape(-1, outcomes.shape[-1]),
    )
    return scores.reshape(n, horizon)
