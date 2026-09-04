"""Conformal scores and prediction-region summaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from outcome_model import GaussianOutcomeModel


_SCORE_INFERENCE_CHUNK_SIZE = 4_096


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
def predict_observed_actions(
    model: GaussianOutcomeModel,
    states: Tensor,
    actions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Predict aligned observed actions without one full-sequence GRU batch."""

    if len(states) == 0:
        shape = (0, 2)
        return states.new_empty(shape), states.new_empty(shape)
    if len(states) <= _SCORE_INFERENCE_CHUNK_SIZE:
        return model(states, actions)
    mean_parts, scale_parts = [], []
    for start in range(0, len(states), _SCORE_INFERENCE_CHUNK_SIZE):
        stop = start + _SCORE_INFERENCE_CHUNK_SIZE
        mean, scale = model(states[start:stop], actions[start:stop])
        mean_parts.append(mean)
        scale_parts.append(scale)
    return torch.cat(mean_parts, dim=0), torch.cat(scale_parts, dim=0)


@torch.no_grad()
def score_batch(
    model: GaussianOutcomeModel | ConformalRegion,
    states: Tensor,
    actions: Tensor,
    outcomes: Tensor,
) -> Tensor:
    predictor = model.outcome_model if isinstance(model, ConformalRegion) else model
    device = next(predictor.parameters()).device
    states, actions, outcomes = states.to(device), actions.to(device), outcomes.to(device)
    n, horizon, state_dim = states.shape
    scorer = (
        model.score
        if isinstance(model, ConformalRegion)
        else lambda x, a, y: normalized_max_score(model, x, a, y)
    )
    flat_states = states.reshape(-1, state_dim)
    flat_actions = actions.reshape(-1)
    flat_outcomes = outcomes.reshape(-1, outcomes.shape[-1])
    if len(flat_states) == 0:
        return states.new_empty((n, horizon))
    if len(flat_states) <= _SCORE_INFERENCE_CHUNK_SIZE:
        scores = scorer(flat_states, flat_actions, flat_outcomes)
        return scores.reshape(n, horizon)
    score_parts = []
    for start in range(0, len(flat_states), _SCORE_INFERENCE_CHUNK_SIZE):
        stop = start + _SCORE_INFERENCE_CHUNK_SIZE
        score_parts.append(
            scorer(
                flat_states[start:stop],
                flat_actions[start:stop],
                flat_outcomes[start:stop],
            )
        )
    scores = torch.cat(score_parts, dim=0)
    return scores.reshape(n, horizon)
