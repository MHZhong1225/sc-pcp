"""Public entry point for committed-prefix marginal SC-PCP.

This module intentionally contains no dataset loaders, paper baselines, or
result-writing conventions.  It packages the final calibration estimand in a
small, reusable interface: callers supply logged trajectories, conformal
scores, frozen stage grids, and the two policies that define transport.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from scpcp.data import TrajectoryBatch
from scpcp.marginal_prefix import (
    MarginalPrefixSelection,
    select_marginal_prefix_schedule,
)


@dataclass(frozen=True)
class PerStepCalibrationInputs:
    """Inputs fixed before sequential radius selection.

    ``stage_grids`` has shape ``[T, K]`` and ``scores`` has shape ``[N, T]``.
    ``outcome_sd`` is the training-outcome scale used only to compare candidate
    widths.  Neither arrays nor policies are modified by calibration.
    """

    trajectories: TrajectoryBatch
    scores: Tensor
    stage_grids: Tensor
    outcome_sd: Tensor
    target_coverage: float = 0.90


def calibrate_per_step_marginal(
    inputs: PerStepCalibrationInputs,
    *,
    target_policy: object,
    logging_policy: object,
    outcome_model: object,
) -> MarginalPrefixSelection:
    """Choose the committed-prefix SC-PCP schedule.

    The returned result is unavailable when no candidate meets the empirical
    target at a stage.  This is an asymptotic per-step marginal calibration
    routine; it is not a finite-sample or data-conditional certificate.
    """

    return select_marginal_prefix_schedule(
        inputs.trajectories,
        inputs.scores,
        stage_grids=inputs.stage_grids,
        target_policy=target_policy,
        logging_policy=logging_policy,
        outcome_model=outcome_model,
        outcome_sd=inputs.outcome_sd,
        target=inputs.target_coverage,
    )
