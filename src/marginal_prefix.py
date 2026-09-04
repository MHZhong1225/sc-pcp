"""Committed-prefix importance-weighted calibration for marginal SC-PCP.

The selector targets per-step marginal coverage.  It is an asymptotic
calibration procedure, not a PAC or data-conditional certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from data import TrajectoryBatch


@dataclass(frozen=True)
class MarginalPrefixSelection:
    radii: Tensor | None
    selected_indices: tuple[int, ...]
    estimated_coverage: Tensor
    estimated_normalized_width: Tensor
    effective_sample_size: Tensor
    maximum_raw_log_weight: Tensor
    raw_log_weight_span: Tensor
    candidate_effective_sample_size: Tensor
    candidate_estimated_coverage: Tensor
    candidate_estimated_normalized_width: Tensor
    candidate_maximum_raw_log_weight: Tensor
    candidate_raw_log_weight_span: Tensor
    selected_endpoint: bool
    failure_stage: int | None

    @property
    def selection_available(self) -> bool:
        return self.radii is not None


@torch.no_grad()
def select_marginal_prefix_schedule(
    batch: TrajectoryBatch,
    scores: Tensor,
    *,
    stage_grids: Tensor,
    target_policy: object,
    logging_policy: object,
    outcome_model: object,
    outcome_sd: Tensor,
    target: float = 0.90,
) -> MarginalPrefixSelection:
    """Select one radius per stage with committed prefix action ratios.

    At stage ``t``, every candidate includes the raw action ratios from the
    already selected prefix and the candidate's current action ratio.  The
    cumulative raw log-weight is normalized by its candidate-specific maximum
    before exponentiation.  This leaves every Hájek estimate and ESS unchanged
    while avoiding overflow without truncating the target Radon--Nikodym
    derivative.  All candidates are scanned because both the induced policy and
    weighted width can vary with the radius.
    """

    if scores.shape != batch.actions.shape:
        raise ValueError("scores must have shape [N,T]")
    if stage_grids.ndim != 2 or stage_grids.shape[0] != batch.horizon:
        raise ValueError("stage_grids must have shape [T,K]")
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie in (0, 1)")
    device = scores.device
    resolved = batch.to(device)
    grids = stage_grids.to(device=device, dtype=scores.dtype)
    normalized_base_width = _normalized_base_width(
        resolved,
        outcome_model=outcome_model,
        outcome_sd=outcome_sd,
    )

    raw_log_prefix = torch.zeros(batch.n, device=device, dtype=torch.float64)
    selected_radii: list[Tensor] = []
    selected_indices: list[int] = []
    selected_coverage: list[Tensor] = []
    selected_width: list[Tensor] = []
    selected_ess: list[Tensor] = []
    selected_maximum_log_weight: list[Tensor] = []
    selected_log_weight_span: list[Tensor] = []
    candidate_ess_by_stage: list[Tensor] = []
    candidate_coverage_by_stage: list[Tensor] = []
    candidate_width_by_stage: list[Tensor] = []
    candidate_maximum_log_weight_by_stage: list[Tensor] = []
    candidate_log_weight_span_by_stage: list[Tensor] = []
    selected_endpoint = False

    for stage, candidate_radii in enumerate(grids):
        states = resolved.states[:, stage]
        observed_actions = resolved.actions[:, stage]
        target_probabilities = target_policy.probabilities_for_grid(
            states,
            candidate_radii,
        )
        logging_probabilities = logging_policy.probabilities(states)
        if (
            not bool(torch.isfinite(target_probabilities).all())
            or not bool(torch.isfinite(logging_probabilities).all())
            or bool((target_probabilities <= 0.0).any())
            or bool((logging_probabilities <= 0.0).any())
        ):
            raise RuntimeError(
                "prefix importance weighting requires finite, strictly positive policies"
            )
        action_index = observed_actions[:, None, None].expand(
            -1,
            len(candidate_radii),
            1,
        )
        numerator = target_probabilities.gather(2, action_index).squeeze(2)
        denominator = logging_probabilities.gather(
            1,
            observed_actions[:, None],
        )
        current_log_ratio = (
            numerator.to(torch.float64).log()
            - denominator.to(torch.float64).log()
        )
        candidate_raw_log_weight = raw_log_prefix[:, None] + current_log_ratio
        if not bool(torch.isfinite(candidate_raw_log_weight).all()):
            raise RuntimeError("prefix importance ratios produced non-finite log-weights")
        maximum_log_weight = candidate_raw_log_weight.amax(dim=0)
        minimum_log_weight = candidate_raw_log_weight.amin(dim=0)
        weights = (candidate_raw_log_weight - maximum_log_weight[None, :]).exp()
        weight_sum = weights.sum(dim=0).clamp_min(1e-12)

        hits = scores[:, stage, None] <= candidate_radii[None, :]
        coverage = (weights * hits).sum(dim=0) / weight_sum
        candidate_width = (
            normalized_base_width[:, stage, None]
            * candidate_radii[None, :]
        )
        normalized_width = (weights * candidate_width).sum(dim=0) / weight_sum
        effective_size = weight_sum.square() / weights.square().sum(dim=0).clamp_min(
            1e-12
        )
        log_weight_span = maximum_log_weight - minimum_log_weight
        candidate_ess_by_stage.append(effective_size.clone())
        candidate_coverage_by_stage.append(coverage.clone())
        candidate_width_by_stage.append(normalized_width.clone())
        candidate_maximum_log_weight_by_stage.append(maximum_log_weight.clone())
        candidate_log_weight_span_by_stage.append(log_weight_span.clone())

        feasible = coverage >= target
        if not bool(feasible.any()):
            return _failed_selection(
                like=scores,
                selected_radii=selected_radii,
                selected_indices=selected_indices,
                selected_coverage=selected_coverage,
                selected_width=selected_width,
                selected_ess=selected_ess,
                selected_maximum_log_weight=selected_maximum_log_weight,
                selected_log_weight_span=selected_log_weight_span,
                candidate_ess_by_stage=candidate_ess_by_stage,
                candidate_coverage_by_stage=candidate_coverage_by_stage,
                candidate_width_by_stage=candidate_width_by_stage,
                candidate_maximum_log_weight_by_stage=(
                    candidate_maximum_log_weight_by_stage
                ),
                candidate_log_weight_span_by_stage=(
                    candidate_log_weight_span_by_stage
                ),
                selected_endpoint=selected_endpoint,
                failure_stage=stage,
            )

        objective = torch.where(
            feasible,
            normalized_width,
            torch.full_like(normalized_width, torch.inf),
        )
        index = int(objective.argmin().item())
        selected_radii.append(candidate_radii[index].clone())
        selected_indices.append(index)
        selected_coverage.append(coverage[index].clone())
        selected_width.append(normalized_width[index].clone())
        selected_ess.append(effective_size[index].clone())
        selected_maximum_log_weight.append(maximum_log_weight[index].clone())
        selected_log_weight_span.append(log_weight_span[index].clone())
        selected_endpoint = selected_endpoint or index in {
            0,
            len(candidate_radii) - 1,
        }

        # Commit the uncapped prefix.  Capping here would change every later
        # longitudinal product and is not terminal weight clipping.
        raw_log_prefix = candidate_raw_log_weight[:, index].clone()

    return MarginalPrefixSelection(
        radii=torch.stack(selected_radii),
        selected_indices=tuple(selected_indices),
        estimated_coverage=torch.stack(selected_coverage),
        estimated_normalized_width=torch.stack(selected_width),
        effective_sample_size=torch.stack(selected_ess),
        maximum_raw_log_weight=torch.stack(selected_maximum_log_weight),
        raw_log_weight_span=torch.stack(selected_log_weight_span),
        candidate_effective_sample_size=torch.stack(candidate_ess_by_stage),
        candidate_estimated_coverage=torch.stack(candidate_coverage_by_stage),
        candidate_estimated_normalized_width=torch.stack(candidate_width_by_stage),
        candidate_maximum_raw_log_weight=torch.stack(
            candidate_maximum_log_weight_by_stage
        ),
        candidate_raw_log_weight_span=torch.stack(candidate_log_weight_span_by_stage),
        selected_endpoint=selected_endpoint,
        failure_stage=None,
    )


def unit_geometric_profile(schedule: Tensor) -> Tensor:
    """Normalize a positive schedule to unit geometric mean."""

    if schedule.ndim != 1 or not torch.isfinite(schedule).all():
        raise ValueError("schedule must be a finite vector")
    if bool((schedule <= 0.0).any()):
        raise ValueError("schedule must be strictly positive")
    return schedule / schedule.log().mean().exp()


def profile_log_rmse(schedule: Tensor, oracle_schedule: Tensor) -> float:
    """Return shape-only log RMSE relative to an oracle schedule."""

    if schedule.shape != oracle_schedule.shape:
        raise ValueError("schedule and oracle_schedule must share shape")
    difference = (
        unit_geometric_profile(schedule).log()
        - unit_geometric_profile(oracle_schedule.to(schedule)).log()
    )
    return float(difference.square().mean().sqrt().item())


def _normalized_base_width(
    batch: TrajectoryBatch,
    *,
    outcome_model: object,
    outcome_sd: Tensor,
) -> Tensor:
    states = batch.current_states().reshape(-1, batch.state_dim)
    actions = batch.actions.reshape(-1)
    _, scales = outcome_model(states, actions)
    scales = scales.reshape(batch.n, batch.horizon, -1)
    normalization = outcome_sd.to(scales).clamp_min(1e-6)
    return (2.0 * scales / normalization[None, None, :]).mean(dim=2)


def _failed_selection(
    *,
    like: Tensor,
    selected_radii: list[Tensor],
    selected_indices: list[int],
    selected_coverage: list[Tensor],
    selected_width: list[Tensor],
    selected_ess: list[Tensor],
    selected_maximum_log_weight: list[Tensor],
    selected_log_weight_span: list[Tensor],
    candidate_ess_by_stage: list[Tensor],
    candidate_coverage_by_stage: list[Tensor],
    candidate_width_by_stage: list[Tensor],
    candidate_maximum_log_weight_by_stage: list[Tensor],
    candidate_log_weight_span_by_stage: list[Tensor],
    selected_endpoint: bool,
    failure_stage: int,
) -> MarginalPrefixSelection:
    def stacked(values: list[Tensor]) -> Tensor:
        return torch.stack(values) if values else like.new_empty(0)

    return MarginalPrefixSelection(
        radii=None,
        selected_indices=tuple(selected_indices),
        estimated_coverage=stacked(selected_coverage),
        estimated_normalized_width=stacked(selected_width),
        effective_sample_size=stacked(selected_ess),
        maximum_raw_log_weight=stacked(selected_maximum_log_weight),
        raw_log_weight_span=stacked(selected_log_weight_span),
        candidate_effective_sample_size=stacked(candidate_ess_by_stage),
        candidate_estimated_coverage=stacked(candidate_coverage_by_stage),
        candidate_estimated_normalized_width=stacked(candidate_width_by_stage),
        candidate_maximum_raw_log_weight=stacked(
            candidate_maximum_log_weight_by_stage
        ),
        candidate_raw_log_weight_span=stacked(candidate_log_weight_span_by_stage),
        selected_endpoint=selected_endpoint,
        failure_stage=failure_stage,
    )
