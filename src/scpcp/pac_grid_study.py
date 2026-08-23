"""Paired dense-grid audit for the retired ordered-IUT SC-PCP path."""

from __future__ import annotations

from dataclasses import dataclass
import json

import torch
from torch import Tensor

from scpcp.certification import (
    CertificationResult,
    ordered_pointwise_bootstrap_lower_bounds,
)
from scpcp.config import ExperimentConfig
from scpcp.cot import cot_state_action_weights, fit_cot
from scpcp.coverage import candidate_radius_schedules, effective_sample_sizes
from scpcp.experiment import (
    SeedResult,
    _estimated_candidate_normalized_widths,
    _paper_seed,
    _prepare_oracle_context,
)
from scpcp.pac_grid_refinement import nested_geometric_grid
from scpcp.phase0_oracle import evaluate_frozen_schedules_crn
from scpcp.scores import score_batch
from scpcp.selection import RadiusSelection, select_ordered_lcb_radius
from scpcp.simulator import make_synthetic_noise_bundle


BASE_METHOD = "PAC grid K=101"
DENSE_METHOD = "PAC grid K=401"


@dataclass(frozen=True)
class PairedGridSelections:
    base: RadiusSelection
    dense: RadiusSelection


def run_paired_grid_seed(
    config: ExperimentConfig,
    *,
    seed: int,
    device: str,
    subdivisions: int = 4,
    evaluation_stream: int = 1_700_001,
) -> SeedResult:
    """Fit one K=101 COT and compare nested K=101/K=401 certificates.

    Both selectors use the same D_COT-frozen profile, fitted COT, D_cert
    contributions, patient-bootstrap resamples, target, delta, and fresh CRN
    bundle.  The only difference is whether fixed-sequence testing may stop at
    one of the deterministic intermediate grid knots.
    """

    if config.data.dataset != "synthetic" or config.synthetic.scenario != "standard":
        raise ValueError("the paired dense-grid audit requires standard synthetic data")
    if subdivisions < 2:
        raise ValueError("the paired audit requires at least two subdivisions")

    torch.manual_seed(seed)
    context = _prepare_oracle_context(config, seed=seed, device=device)
    splits = context.task.splits
    family = context.schedule_family
    base_grid = family.scale_grid
    dense_grid, base_indices = nested_geometric_grid(
        base_grid,
        subdivisions=subdivisions,
    )
    expected_dense_size = subdivisions * (len(base_grid) - 1) + 1
    if len(dense_grid) != expected_dense_size:
        raise RuntimeError("dense grid has an unexpected size")

    fitted_cot = fit_cot(
        splits.cot,
        q_grid=base_grid,
        stage_profile=family.profile,
        target_policy=context.policy,
        logging_policy=context.logging_policy,
        outcome_model=context.outcome_model,
        config=config.cot,
        device=device,
        seed=seed + 3,
    )
    cert_scores = score_batch(
        context.region,
        splits.certification.current_states(),
        splits.certification.actions,
        splits.certification.outcomes,
    )
    base_radii = candidate_radius_schedules(base_grid, family.profile)
    dense_radii = candidate_radius_schedules(dense_grid, family.profile)
    base_weights, base_weight_diagnostics = cot_state_action_weights(
        fitted_cot,
        splits.certification,
        q_grid=base_grid,
        target_policy=context.policy,
        logging_policy=context.logging_policy,
        weight_cap=config.cot.weight_cap,
    )
    dense_weights, weight_diagnostics = cot_state_action_weights(
        fitted_cot,
        splits.certification,
        q_grid=dense_grid,
        target_policy=context.policy,
        logging_policy=context.logging_policy,
        weight_cap=config.cot.weight_cap,
    )
    base_widths = _estimated_candidate_normalized_widths(
        context.outcome_model,
        splits.certification,
        base_weights,
        base_radii,
        context.outcome_sd,
    )
    dense_widths = _estimated_candidate_normalized_widths(
        context.outcome_model,
        splits.certification,
        dense_weights,
        dense_radii,
        context.outcome_sd,
    )
    base_certificate = ordered_pointwise_bootstrap_lower_bounds(
        base_weights,
        cert_scores.to(base_weights),
        base_radii.to(base_weights),
        lower_tail=config.certification.delta,
        n_resamples=config.certification.practical_bootstrap_resamples,
        seed=seed + 31_337,
        cluster_ids=splits.certification.patient_ids,
    )
    dense_certificate = ordered_pointwise_bootstrap_lower_bounds(
        dense_weights,
        cert_scores.to(dense_weights),
        dense_radii.to(dense_weights),
        lower_tail=config.certification.delta,
        n_resamples=config.certification.practical_bootstrap_resamples,
        seed=seed + 31_337,
        cluster_ids=splits.certification.patient_ids,
    )
    base_selection = select_ordered_lcb_radius(
        base_grid,
        base_certificate,
        alpha=config.certification.alpha,
        widths=base_widths,
        status="PRACTICAL_CLUSTER_ORDERED_IUT_LCB_BASE_GRID",
    )
    dense_selection = select_ordered_lcb_radius(
        dense_grid,
        dense_certificate,
        alpha=config.certification.alpha,
        widths=dense_widths,
        status="PRACTICAL_CLUSTER_ORDERED_IUT_LCB_DENSE_GRID",
    )
    selections = PairedGridSelections(base_selection, dense_selection)

    schedules = _selected_schedules(selections, family.profile)
    evaluation_seed = _paper_seed(seed, evaluation_stream)
    noise = make_synthetic_noise_bundle(
        n=config.samples.oracle_rollouts,
        horizon=config.horizon,
        seed=evaluation_seed,
        device=device,
    )
    fresh = evaluate_frozen_schedules_crn(
        context.task.environment,
        context.policy,
        context.outcome_model,
        schedules=schedules,
        noise=noise,
        outcome_sd=context.outcome_sd,
    )

    dense_ess = effective_sample_sizes(
        dense_weights,
        cluster_ids=splits.certification.patient_ids,
    )
    records = [
        _selection_record(
            seed,
            BASE_METHOD,
            base_selection,
            base_certificate,
            fresh[BASE_METHOD],
            target=1.0 - config.certification.alpha,
            delta=config.certification.delta,
            grid_size=len(base_grid),
            evaluation_seed=evaluation_seed,
        ),
        _selection_record(
            seed,
            DENSE_METHOD,
            dense_selection,
            dense_certificate,
            fresh[DENSE_METHOD],
            target=1.0 - config.certification.alpha,
            delta=config.certification.delta,
            grid_size=len(dense_grid),
            evaluation_seed=evaluation_seed,
        ),
    ]

    base_schedule = schedules[BASE_METHOD]
    dense_schedule = schedules[DENSE_METHOD]
    surfaces = {
        "base_grid": base_grid,
        "dense_grid": dense_grid,
        "base_indices_in_dense": base_indices,
        "stage_profile": family.profile,
        "base_point_estimates": base_certificate.estimates,
        "dense_point_estimates": dense_certificate.estimates,
        "base_lower_bounds": base_certificate.lower_bounds,
        "dense_lower_bounds": dense_certificate.lower_bounds,
        "base_estimated_widths": base_widths,
        "dense_estimated_widths": dense_widths,
        "base_selected_schedule": base_schedule,
        "dense_selected_schedule": dense_schedule,
        "base_fresh_coverage": fresh[BASE_METHOD].coverage,
        "dense_fresh_coverage": fresh[DENSE_METHOD].coverage,
        "base_fresh_width": fresh[BASE_METHOD].normalized_width,
        "dense_fresh_width": fresh[DENSE_METHOD].normalized_width,
        "dense_effective_sample_sizes": dense_ess,
    }
    diagnostics = {
        "protocol": "paired_nested_pac_grid_v1",
        "target": 1.0 - config.certification.alpha,
        "delta": config.certification.delta,
        "subdivisions": subdivisions,
        "base_grid_size": len(base_grid),
        "dense_grid_size": len(dense_grid),
        "evaluation_seed": evaluation_seed,
        "evaluation_rollouts": config.samples.oracle_rollouts,
        "base_selected_index": base_selection.index,
        "dense_selected_index": dense_selection.index,
        "base_stopped_index": base_selection.stopped_index,
        "dense_stopped_index": dense_selection.stopped_index,
        "dense_selected_is_base_knot": (
            False
            if dense_selection.index is None
            else bool(dense_selection.index % subdivisions == 0)
        ),
        "maximum_base_weight_parity_error": float(
            (dense_weights[:, :, base_indices] - base_weights).abs().max().item()
        ),
        "maximum_base_point_parity_error": float(
            (dense_certificate.estimates[base_indices] - base_certificate.estimates)
            .abs()
            .max()
            .item()
        ),
        "maximum_base_lcb_parity_error": float(
            (dense_certificate.lower_bounds[base_indices] - base_certificate.lower_bounds)
            .abs()
            .max()
            .item()
        ),
        "maximum_base_width_parity_error": float(
            (dense_widths[base_indices] - base_widths).abs().max().item()
        ),
        "base_selection_matches_dense_base_knots": _selection_matches_dense_base_knots(
            base_selection,
            dense_certificate,
            dense_widths,
            base_grid,
            base_indices,
            alpha=config.certification.alpha,
        ),
        "minimum_dense_ess": float(dense_ess.min().item()),
        "mean_base_cap_hit_rate": float(
            base_weight_diagnostics.cap_hit_rate.mean().item()
        ),
        "mean_dense_cap_hit_rate": float(weight_diagnostics.cap_hit_rate.mean().item()),
        "maximum_base_cap_hit_rate": float(
            base_weight_diagnostics.cap_hit_rate.max().item()
        ),
        "maximum_dense_cap_hit_rate": float(weight_diagnostics.cap_hit_rate.max().item()),
        "certificate_label": dense_certificate.label,
        "certificate_formal": dense_certificate.formal,
        "cot": fitted_cot.diagnostics,
    }
    return SeedResult(
        seed=seed,
        device=device,
        records=records,
        surfaces={name: value.detach().cpu() for name, value in surfaces.items()},
        diagnostics=diagnostics,
    )


def _selection_matches_dense_base_knots(
    base_selection: RadiusSelection,
    dense_certificate: CertificationResult,
    dense_widths: Tensor,
    base_grid: Tensor,
    base_indices: Tensor,
    *,
    alpha: float,
) -> bool:
    """Independently reselect after restricting the dense result to old knots."""

    dense_at_base = CertificationResult(
        estimates=dense_certificate.estimates[base_indices],
        lower_bounds=dense_certificate.lower_bounds[base_indices],
        sampling_margin=dense_certificate.sampling_margin,
        ratio_error_bound=dense_certificate.ratio_error_bound[base_indices],
        formal=dense_certificate.formal,
        label=dense_certificate.label,
    )
    replay = select_ordered_lcb_radius(
        base_grid,
        dense_at_base,
        alpha=alpha,
        widths=dense_widths[base_indices],
        status=base_selection.status,
    )
    return (
        replay.index == base_selection.index
        and replay.certified_indices == base_selection.certified_indices
        and replay.stopped_index == base_selection.stopped_index
    )


def _selected_schedules(
    selections: PairedGridSelections,
    profile: Tensor,
) -> dict[str, Tensor]:
    schedules = {}
    for name, selection in (
        (BASE_METHOD, selections.base),
        (DENSE_METHOD, selections.dense),
    ):
        if selection.radius is None:
            raise RuntimeError(f"{name} has no certified candidate")
        schedules[name] = selection.radius * profile
    return schedules


def _selection_record(
    seed: int,
    method: str,
    selection: RadiusSelection,
    certificate: CertificationResult,
    fresh: object,
    *,
    target: float,
    delta: float,
    grid_size: int,
    evaluation_seed: int,
) -> dict[str, object]:
    if selection.index is None or selection.radius is None:
        raise RuntimeError(f"{method} has no certified candidate")
    index = selection.index
    coverage = fresh.coverage
    width = fresh.normalized_width
    if not torch.isfinite(coverage).all() or not torch.isfinite(width).all():
        raise RuntimeError(f"{method} fresh metrics are non-finite")
    return {
        "seed": seed,
        "method": method,
        "grid_size": grid_size,
        "target_coverage": target,
        "confidence_level": 1.0 - delta,
        "delta": delta,
        "selection_available": True,
        "selection_status": selection.status,
        "selected_index": index,
        "selected_scale": float(selection.radius),
        "estimated_min_coverage": float(certificate.estimates[index].min().item()),
        "lower_bound_min": float(certificate.lower_bounds[index].min().item()),
        "fresh_worst_coverage": float(coverage.min().item()),
        "fresh_mean_coverage": float(coverage.mean().item()),
        "fresh_per_time_coverage": json.dumps(
            [float(value) for value in coverage.tolist()]
        ),
        "fresh_average_normalized_width": float(width.mean().item()),
        "fresh_per_time_normalized_width": json.dumps(
            [float(value) for value in width.tolist()]
        ),
        "fresh_target_met": bool(float(coverage.min().item()) >= target),
        "evaluation_seed": evaluation_seed,
        "evaluation_rollouts": fresh.n_rollouts,
        "certificate_type": certificate.label,
        "certificate_formal": certificate.formal,
    }
