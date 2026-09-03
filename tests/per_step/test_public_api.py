from __future__ import annotations

import torch

from scpcp.data import TrajectoryBatch
from scpcp.experiments import PerStepCalibrationInputs, calibrate_per_step_marginal


class _LoggingPolicy:
    def probabilities(self, states: torch.Tensor) -> torch.Tensor:
        return torch.full((len(states), 2), 0.5, device=states.device)


class _TargetPolicy:
    def probabilities_for_grid(
        self,
        states: torch.Tensor,
        radii: torch.Tensor,
    ) -> torch.Tensor:
        return torch.full((len(states), len(radii), 2), 0.5, device=states.device)


class _UnitScaleOutcome:
    def __call__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((len(states), 1), device=states.device),
            torch.ones((len(states), 1), device=states.device),
        )


def test_public_calibration_api_returns_a_stagewise_schedule() -> None:
    batch = TrajectoryBatch(
        states=torch.zeros((4, 3, 1)),
        actions=torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
        outcomes=torch.zeros((4, 2, 1)),
        patient_ids=torch.arange(4),
    )
    inputs = PerStepCalibrationInputs(
        trajectories=batch,
        scores=torch.tensor([[0.4, 0.8], [0.8, 0.4], [1.2, 1.1], [1.8, 1.7]]),
        stage_grids=torch.tensor([[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]]),
        outcome_sd=torch.ones(1),
        target_coverage=0.75,
    )

    selected = calibrate_per_step_marginal(
        inputs,
        target_policy=_TargetPolicy(),
        logging_policy=_LoggingPolicy(),
        outcome_model=_UnitScaleOutcome(),
    )

    assert selected.selection_available
    assert torch.equal(selected.radii, torch.tensor([1.5, 1.5]))
