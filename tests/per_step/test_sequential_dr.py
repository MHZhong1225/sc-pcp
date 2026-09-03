from __future__ import annotations

import torch

from scpcp.config import COTConfig, ModelConfig
from scpcp.data import TrajectoryBatch
from scpcp.outcome_model import GaussianOutcomeModel
from scpcp.sequential_dr import (
    dr_quantile,
    empirical_cdf,
    fit_fixed_schedule_sequential_dr,
    make_score_cdf_grid,
    prefix_action_weights,
    sequential_dr_score_cdf,
)


class _UniformPolicy:
    n_actions = 2

    def probabilities(self, states: torch.Tensor, q: object = None) -> torch.Tensor:
        return states.new_full((len(states), 2), 0.5)


def _batch() -> TrajectoryBatch:
    generator = torch.Generator().manual_seed(33)
    return TrajectoryBatch(
        states=torch.randn(24, 4, 3, generator=generator),
        actions=torch.arange(72).reshape(24, 3).remainder(2),
        outcomes=torch.randn(24, 3, 2, generator=generator),
        patient_ids=torch.arange(24),
    )


def test_sequential_dr_returns_a_finite_monotone_score_cdf() -> None:
    batch = _batch()
    scores = torch.linalg.vector_norm(batch.outcomes, dim=2)
    outcome_model = GaussianOutcomeModel(
        state_dim=3,
        n_actions=2,
        config=ModelConfig(hidden_dim=8, representation_dim=4, architecture="mlp"),
    )
    policy = _UniformPolicy()
    grid = make_score_cdf_grid(scores, size=13)
    fitted = fit_fixed_schedule_sequential_dr(
        batch,
        scores,
        schedule=torch.ones(batch.horizon),
        target_policy=policy,
        outcome_model=outcome_model,
        config=COTConfig(hidden_dims=(8,), epochs=3, batch_size=12, patience=1),
        device="cpu",
        seed=34,
        score_grid=grid,
    )

    cdf = sequential_dr_score_cdf(
        fitted,
        batch,
        scores,
        target_policy=policy,
        logging_policy=policy,
    )
    quantile = dr_quantile(cdf, grid)

    assert cdf.shape == grid.shape
    assert quantile.shape == (batch.horizon,)
    assert torch.isfinite(cdf).all()
    assert torch.isfinite(quantile).all()
    assert ((cdf[:, 1:] - cdf[:, :-1]) >= 0).all()
    assert ((cdf >= 0) & (cdf <= 1)).all()


def test_uniform_policy_prefix_weights_and_empirical_cdf_are_identity() -> None:
    batch = _batch()
    scores = torch.linalg.vector_norm(batch.outcomes, dim=2)
    policy = _UniformPolicy()
    weights = prefix_action_weights(
        batch,
        schedule=torch.ones(batch.horizon),
        target_policy=policy,
        logging_policy=policy,
    )
    grid = make_score_cdf_grid(scores, size=9)

    assert torch.equal(weights, torch.ones_like(weights))
    assert torch.allclose(empirical_cdf(scores, grid).to(weights), empirical_cdf(scores, grid, weights=weights))
