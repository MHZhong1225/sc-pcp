from __future__ import annotations

import math

import pytest
import torch

from scpcp.data import TrajectoryBatch
from scpcp.marginal_prefix import (
    profile_log_rmse,
    select_marginal_prefix_schedule,
    unit_geometric_profile,
)


class _UniformPolicy:
    def probabilities(self, states: torch.Tensor) -> torch.Tensor:
        return torch.full((len(states), 2), 0.5, device=states.device)


class _UniformTargetPolicy:
    def probabilities_for_grid(
        self,
        states: torch.Tensor,
        radii: torch.Tensor,
    ) -> torch.Tensor:
        return torch.full(
            (len(states), len(radii), 2),
            0.5,
            device=states.device,
        )


class _RadiusPolicy:
    def probabilities_for_grid(
        self,
        states: torch.Tensor,
        radii: torch.Tensor,
    ) -> torch.Tensor:
        probability_zero = 0.2 + 0.2 * radii
        probabilities = torch.stack(
            (probability_zero, 1.0 - probability_zero),
            dim=1,
        )
        return probabilities[None, :, :].expand(len(states), -1, -1)


class _NearlyDeterministicTargetPolicy:
    def probabilities_for_grid(
        self,
        states: torch.Tensor,
        radii: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.tensor(
            [1.0 - 1e-16, 1e-16],
            device=states.device,
            dtype=torch.float64,
        )
        return probabilities[None, None, :].expand(len(states), len(radii), -1)


class _TinyPositiveLoggingPolicy:
    def probabilities(self, states: torch.Tensor) -> torch.Tensor:
        probabilities = torch.tensor(
            [1e-20, 1.0 - 1e-20],
            device=states.device,
            dtype=torch.float64,
        )
        return probabilities[None, :].expand(len(states), -1)


class _UnitScaleModel:
    def __call__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((len(states), 1), device=states.device),
            torch.ones((len(states), 1), device=states.device),
        )


def _batch(actions: torch.Tensor) -> TrajectoryBatch:
    n, horizon = actions.shape
    return TrajectoryBatch(
        states=torch.zeros((n, horizon + 1, 1)),
        actions=actions,
        outcomes=torch.zeros((n, horizon, 1)),
        patient_ids=torch.arange(n),
    )


def test_uniform_policy_reduces_to_stagewise_weighted_quantiles() -> None:
    batch = _batch(torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
    scores = torch.tensor(
        [
            [0.4, 0.8],
            [0.8, 0.4],
            [1.2, 1.1],
            [1.8, 1.7],
        ]
    )
    grids = torch.tensor([[0.5, 1.0, 1.5, 2.0], [0.5, 1.0, 1.5, 2.0]])

    result = select_marginal_prefix_schedule(
        batch,
        scores,
        stage_grids=grids,
        target_policy=_UniformTargetPolicy(),
        logging_policy=_UniformPolicy(),
        outcome_model=_UnitScaleModel(),
        outcome_sd=torch.ones(1),
        target=0.75,
    )

    assert result.selection_available
    assert result.selected_indices == (2, 2)
    assert torch.equal(result.radii, torch.tensor([1.5, 1.5]))
    assert torch.allclose(
        result.estimated_coverage,
        torch.tensor([0.75, 0.75], dtype=torch.float64),
    )
    assert torch.allclose(
        result.estimated_normalized_width,
        torch.tensor([3.0, 3.0], dtype=torch.float64),
    )
    assert torch.allclose(
        result.effective_sample_size,
        torch.full((2,), 4.0, dtype=torch.float64),
    )
    assert result.candidate_estimated_coverage.shape == grids.shape
    assert result.candidate_estimated_normalized_width.shape == grids.shape
    assert float(result.candidate_estimated_coverage[0, 2]) == pytest.approx(0.75)
    assert not result.selected_endpoint


def test_current_candidate_uses_the_committed_raw_prefix_ratio() -> None:
    batch = _batch(torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
    scores = torch.tensor(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [1.5, 1.5],
            [1.5, 1.5],
        ]
    )
    grids = torch.tensor([[1.0, 2.0], [1.0, 2.0]])

    result = select_marginal_prefix_schedule(
        batch,
        scores,
        stage_grids=grids,
        target_policy=_RadiusPolicy(),
        logging_policy=_UniformPolicy(),
        outcome_model=_UnitScaleModel(),
        outcome_sd=torch.ones(1),
        target=0.55,
    )

    assert result.selection_available
    assert result.selected_indices == (1, 0)
    assert torch.equal(result.radii, torch.tensor([2.0, 1.0]))
    # With only the current-stage ratio this coverage is 0.50.  The committed
    # stage-zero ratios change it to (0.96 + 1.44) / 4 = 0.60.
    assert float(result.estimated_coverage[1]) == pytest.approx(0.60)
    expected_ess = 4.0**2 / (0.96**2 + 1.44**2 + 0.64**2 + 0.96**2)
    assert float(result.effective_sample_size[1]) == pytest.approx(expected_ess)


def test_no_feasible_candidate_reports_the_failure_stage() -> None:
    batch = _batch(torch.tensor([[0], [1], [0]]))
    scores = torch.ones((3, 1))

    result = select_marginal_prefix_schedule(
        batch,
        scores,
        stage_grids=torch.tensor([[0.1, 0.2]]),
        target_policy=_UniformTargetPolicy(),
        logging_policy=_UniformPolicy(),
        outcome_model=_UnitScaleModel(),
        outcome_sd=torch.ones(1),
        target=0.9,
    )

    assert not result.selection_available
    assert result.failure_stage == 0
    assert result.selected_indices == ()
    assert result.estimated_coverage.numel() == 0
    assert result.candidate_estimated_coverage.shape == (1, 2)


def test_uncapped_float64_log_stabilization_preserves_extreme_positive_ratios() -> None:
    horizon = 6
    batch = _batch(torch.zeros((3, horizon), dtype=torch.long))
    scores = torch.full((3, horizon), 0.5)

    result = select_marginal_prefix_schedule(
        batch,
        scores,
        stage_grids=torch.tensor([[1.0, 2.0, 3.0]]).expand(horizon, -1),
        target_policy=_NearlyDeterministicTargetPolicy(),
        logging_policy=_TinyPositiveLoggingPolicy(),
        outcome_model=_UnitScaleModel(),
        outcome_sd=torch.ones(1),
        target=0.9,
    )

    assert result.selection_available
    per_step_log_ratio = math.log((1.0 - 1e-16) / 1e-20)
    assert float(result.maximum_raw_log_weight[-1]) == pytest.approx(
        horizon * per_step_log_ratio
    )
    assert float(result.maximum_raw_log_weight[-1]) > math.log(1e38)
    assert torch.equal(result.raw_log_weight_span, torch.zeros(horizon, dtype=torch.float64))
    assert torch.allclose(
        result.effective_sample_size,
        torch.full((horizon,), 3.0, dtype=torch.float64),
    )


def test_profile_error_is_invariant_to_global_scale() -> None:
    schedule = torch.tensor([1.0, 2.0, 4.0])
    oracle = torch.tensor([1.2, 2.4, 4.8])

    assert torch.allclose(
        unit_geometric_profile(schedule),
        unit_geometric_profile(7.0 * schedule),
    )
    assert profile_log_rmse(schedule, oracle) == pytest.approx(0.0, abs=1e-6)
    assert math.isclose(float(unit_geometric_profile(schedule).log().mean()), 0.0, abs_tol=1e-6)
