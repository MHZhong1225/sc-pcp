from __future__ import annotations

import torch

from scpcp.config import COTConfig, ModelConfig
from scpcp.data import TrajectoryBatch
from scpcp.fixed_schedule_cot import (
    fit_fixed_schedule_cot,
    fixed_schedule_state_action_weights,
)
from scpcp.outcome_model import GaussianOutcomeModel


class _UniformPolicy:
    n_actions = 2

    def probabilities(self, states: torch.Tensor, q: object = None) -> torch.Tensor:
        return states.new_full((len(states), 2), 0.5)


def _batch() -> TrajectoryBatch:
    generator = torch.Generator().manual_seed(8)
    return TrajectoryBatch(
        states=torch.randn(20, 4, 3, generator=generator),
        actions=torch.arange(60).reshape(20, 3).remainder(2),
        outcomes=torch.randn(20, 3, 2, generator=generator),
        patient_ids=torch.arange(20),
    )


def test_fixed_schedule_cot_preserves_the_identity_case() -> None:
    batch = _batch()
    model = GaussianOutcomeModel(
        state_dim=3,
        n_actions=2,
        config=ModelConfig(hidden_dim=8, representation_dim=4, architecture="mlp"),
    )
    fitted = fit_fixed_schedule_cot(
        batch,
        schedule=torch.ones(3),
        target_policy=_UniformPolicy(),
        logging_policy=_UniformPolicy(),
        outcome_model=model,
        config=COTConfig(hidden_dims=(8,), epochs=2, batch_size=10, patience=1),
        device="cpu",
        seed=9,
    )
    weights, diagnostics = fixed_schedule_state_action_weights(
        fitted,
        batch,
        target_policy=_UniformPolicy(),
        logging_policy=_UniformPolicy(),
    )

    assert weights.shape == batch.actions.shape
    assert torch.equal(weights[:, 0], torch.ones(batch.n))
    assert torch.isfinite(weights).all()
    assert torch.isfinite(diagnostics.effective_sample_size).all()
    assert len(fitted.diagnostics.validation_mse) == batch.horizon - 1
