"""Simultaneous per-step lower bounds for diagonal SC-PCP selection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from config import CertificationConfig


@dataclass(frozen=True)
class CertificationResult:
    estimates: Tensor
    lower_bounds: Tensor
    sampling_margin: float
    ratio_error_bound: Tensor
    formal: bool
    label: str

    @property
    def aggregate_lower_bound(self) -> Tensor:
        return self.lower_bounds.amin(dim=1)


def simultaneous_lower_bounds(
    estimates: Tensor,
    *,
    n_trajectories: int,
    weight_cap: float,
    config: CertificationConfig,
    allow_oracle: bool = False,
    cluster_ids: Tensor | None = None,
) -> CertificationResult:
    """Bound every (q,t) diagonal constraint without self-normalization.

    A nonzero COT error term is valid only if supplied as a simultaneous L1
    state-action-weight bound by an external analysis.  The result records this
    distinction rather than treating neural fit residuals as a theorem proof.
    When ``cluster_ids`` is supplied, the sampling margin uses the worst-case
    unequal-cluster Hoeffding effective sample size.
    """

    if estimates.ndim != 2:
        raise ValueError("estimates must have shape [K,T]")
    if n_trajectories < 1 or weight_cap <= 0:
        raise ValueError("n_trajectories and weight_cap must be positive")
    if config.ratio_bound_source not in {"none", "declared", "oracle"}:
        raise ValueError("unknown ratio bound source")
    if config.ratio_bound_source == "oracle" and not allow_oracle:
        raise ValueError("oracle ratio bounds are restricted to the internal exact-tabular validation path")
    if config.ratio_bound_source == "declared" and config.ratio_delta <= 0.0:
        raise ValueError("declared statistical ratio bounds require positive ratio_delta")
    k, horizon = estimates.shape
    sample_delta = config.delta - config.ratio_delta
    effective_n = _cluster_count_effective_sample_size(n_trajectories, cluster_ids)
    margin = weight_cap * math.sqrt(math.log(k * horizon / sample_delta) / (2.0 * effective_n))
    formal = config.ratio_bound_source != "none"
    applied_ratio_error = torch.full_like(estimates, config.ratio_error_bound) if formal else torch.zeros_like(estimates)
    # Missing a transport-error premise is not the same as knowing that the
    # error is zero.  Preserve the raw-HT sampling diagnostic, but publish NA
    # in the ratio-error field so downstream tables cannot misread it.
    ratio_error = applied_ratio_error if formal else torch.full_like(estimates, float("nan"))
    label = (
        "oracle_ratio_bound"
        if config.ratio_bound_source == "oracle"
        else "assumption_based_ratio_bound"
        if formal
        else "raw_ht_sampling_only_no_transport_bound"
    )
    lower = (estimates - margin - applied_ratio_error).clamp_min(0.0)
    reported_ratio_error = ratio_error if formal else torch.full_like(estimates, float("nan"))
    return CertificationResult(
        estimates=estimates,
        lower_bounds=lower,
        sampling_margin=margin,
        ratio_error_bound=reported_ratio_error,
        formal=formal,
        label=label,
    )


def ordered_pointwise_ht_lower_bounds(
    estimates: Tensor,
    *,
    n_trajectories: int,
    weight_cap: float,
    config: CertificationConfig,
    allow_oracle: bool = False,
    cluster_ids: Tensor | None = None,
) -> CertificationResult:
    """Pointwise bounded-HT LCBs for widest-to-narrowest fixed-sequence tests.

    The sampling margin deliberately contains neither the candidate count nor
    the horizon.  At one candidate, the unsafe null is a union over stages, so
    rejecting it requires every level-``delta`` component test to reject (an
    intersection-union test).  Across candidates, validity comes from the
    prespecified fixed sequence and its stop-at-first-failure rule; callers
    must therefore use the ordered selector rather than arbitrary screening.
    """

    if estimates.ndim != 2:
        raise ValueError("estimates must have shape [K,T]")
    if n_trajectories < 1 or weight_cap <= 0.0:
        raise ValueError("n_trajectories and weight_cap must be positive")
    if config.ratio_bound_source not in {"none", "declared", "oracle"}:
        raise ValueError("unknown ratio bound source")
    if config.ratio_bound_source == "oracle" and not allow_oracle:
        raise ValueError("oracle ratio bounds are restricted to exact-tabular validation")
    if config.ratio_bound_source == "declared" and config.ratio_delta <= 0.0:
        raise ValueError("declared statistical ratio bounds require positive ratio_delta")
    sample_delta = config.delta - config.ratio_delta
    if not 0.0 < sample_delta < 1.0:
        raise ValueError("delta minus ratio_delta must lie in (0, 1)")
    effective_n = _cluster_count_effective_sample_size(n_trajectories, cluster_ids)
    margin = weight_cap * math.sqrt(math.log(1.0 / sample_delta) / (2.0 * effective_n))
    formal = config.ratio_bound_source != "none"
    applied_error = (
        torch.full_like(estimates, config.ratio_error_bound)
        if formal
        else torch.zeros_like(estimates)
    )
    reported_error = (
        applied_error if formal else torch.full_like(estimates, float("nan"))
    )
    source = (
        "oracle_transport"
        if config.ratio_bound_source == "oracle"
        else "declared_transport"
        if formal
        else "no_transport_bound"
    )
    return CertificationResult(
        estimates=estimates,
        lower_bounds=(estimates - margin - applied_error).clamp(0.0, 1.0),
        sampling_margin=margin,
        ratio_error_bound=reported_error,
        formal=formal,
        label=f"ordered_iut_pointwise_ht_{source}",
    )


def exact_tabular_l1_lower_bounds(
    estimates: Tensor,
    *,
    n_trajectories: int,
    weight_cap: float,
    exact_l1_error_bound: Tensor,
    delta: float,
    cluster_ids: Tensor | None = None,
) -> CertificationResult:
    """Formal internal certificate using an enumerated finite-MDP COT error.

    ``exact_l1_error_bound`` must be the exact distributional L1 discrepancy
    between the capped estimator weights and target state-action ratios.  The
    caller obtains it only by enumerating the known finite MDP, so this helper
    is deliberately an oracle-validation utility rather than a public
    learned-ratio certificate.  Optional cluster IDs replace the trajectory
    count by the corresponding unequal-cluster Hoeffding effective sample size.
    """

    if estimates.ndim != 2 or exact_l1_error_bound.shape != estimates.shape:
        raise ValueError("estimates and exact_l1_error_bound must both have shape [K,T]")
    if n_trajectories < 1 or weight_cap <= 0.0:
        raise ValueError("n_trajectories and weight_cap must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if not torch.isfinite(exact_l1_error_bound).all() or (exact_l1_error_bound < 0.0).any():
        raise ValueError("exact_l1_error_bound must be finite and nonnegative")
    k, horizon = estimates.shape
    effective_n = _cluster_count_effective_sample_size(n_trajectories, cluster_ids)
    margin = weight_cap * math.sqrt(math.log(k * horizon / delta) / (2.0 * effective_n))
    lower = (estimates - margin - exact_l1_error_bound.to(estimates)).clamp_min(0.0)
    return CertificationResult(
        estimates=estimates,
        lower_bounds=lower,
        sampling_margin=margin,
        ratio_error_bound=exact_l1_error_bound.to(estimates),
        formal=True,
        label="tabular_exact_l1_oracle_bound",
    )


def exact_tabular_ordered_pointwise_l1_lower_bounds(
    estimates: Tensor,
    *,
    n_trajectories: int,
    weight_cap: float,
    exact_l1_error_bound: Tensor,
    delta: float,
    cluster_ids: Tensor | None = None,
) -> CertificationResult:
    """Exact finite-MDP transport audit for the ordered pointwise test."""

    if estimates.ndim != 2 or exact_l1_error_bound.shape != estimates.shape:
        raise ValueError("estimates and exact_l1_error_bound must both have shape [K,T]")
    if n_trajectories < 1 or weight_cap <= 0.0:
        raise ValueError("n_trajectories and weight_cap must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if not torch.isfinite(exact_l1_error_bound).all() or (exact_l1_error_bound < 0.0).any():
        raise ValueError("exact_l1_error_bound must be finite and nonnegative")
    effective_n = _cluster_count_effective_sample_size(n_trajectories, cluster_ids)
    margin = weight_cap * math.sqrt(math.log(1.0 / delta) / (2.0 * effective_n))
    error = exact_l1_error_bound.to(estimates)
    return CertificationResult(
        estimates=estimates,
        lower_bounds=(estimates - margin - error).clamp(0.0, 1.0),
        sampling_margin=margin,
        ratio_error_bound=error,
        formal=True,
        label="tabular_ordered_iut_pointwise_exact_l1_oracle_bound",
    )


@torch.no_grad()
def practical_bootstrap_lower_bounds(
    weights: Tensor,
    scores: Tensor,
    q_grid: Tensor,
    *,
    lower_tail: float,
    n_resamples: int,
    seed: int,
    resample_batch_size: int = 32,
    cluster_ids: Tensor | None = None,
) -> CertificationResult:
    """Self-normalized simultaneous cluster-bootstrap LCBs for practical selection.

    This is intentionally not a theorem-level simultaneous certificate: it
    resamples complete independent clusters and constructs one studentized
    max-deviation band over the full frozen ``(q,t)`` family.  This protects
    adaptive radius selection and all per-time constraints without directly
    bootstrapping the non-smooth ``min`` functional.  A simultaneous Wilson
    guard prevents degenerate cells, including all-success boundaries, from
    receiving a zero-width interval.  It exists
    for continuous-state or clinical use where a valid external COT L1 bound
    is unavailable.  Formal selection must continue to use
    :func:`simultaneous_lower_bounds`.
    """

    if weights.ndim != 3 or scores.ndim != 2:
        raise ValueError("weights must be [N,T,K] and scores must be [N,T]")
    if weights.shape[:2] != scores.shape or weights.shape[2] != len(q_grid):
        raise ValueError("bootstrap inputs must share trajectory, time, and q-grid dimensions")
    if not 0.0 < lower_tail < 1.0:
        raise ValueError("bootstrap lower_tail must lie in (0, 1)")
    if n_resamples < 1 or resample_batch_size < 1:
        raise ValueError("bootstrap resample counts must be positive")
    events = (scores[:, :, None] <= q_grid.to(scores)[None, None, :]).to(weights.dtype)
    numerator = (weights * events).permute(0, 2, 1).contiguous()  # [N,K,T]
    denominator = weights.permute(0, 2, 1).contiguous()
    return _practical_bootstrap_from_ratio_contributions(
        numerator,
        denominator,
        lower_tail=lower_tail,
        n_resamples=n_resamples,
        seed=seed,
        resample_batch_size=resample_batch_size,
        cluster_ids=cluster_ids,
        simultaneous=True,
    )


@torch.no_grad()
def ordered_pointwise_bootstrap_lower_bounds(
    weights: Tensor,
    scores: Tensor,
    candidate_radii: Tensor,
    *,
    lower_tail: float,
    n_resamples: int,
    seed: int,
    resample_batch_size: int = 32,
    cluster_ids: Tensor | None = None,
) -> CertificationResult:
    """Marginal patient-cluster LCBs used by practical ordered-IUT SC-PCP.

    The full cluster is resampled jointly, preserving dependence across stages
    and candidates.  Quantiles are nevertheless taken cell by cell: candidate
    multiplicity is handled by the frozen fixed sequence, while stage
    multiplicity is handled by the intersection-union test.  The result is a
    practical bootstrap diagnostic, not a theorem-level certificate.
    """

    if weights.ndim != 3 or scores.ndim != 2:
        raise ValueError("weights must be [N,T,K] and scores must be [N,T]")
    thresholds = _candidate_thresholds(candidate_radii, scores)
    if weights.shape[:2] != scores.shape or weights.shape[2] != thresholds.shape[1]:
        raise ValueError("bootstrap inputs must share trajectory, time, and candidate dimensions")
    if not 0.0 < lower_tail < 1.0:
        raise ValueError("bootstrap lower_tail must lie in (0, 1)")
    if n_resamples < 1 or resample_batch_size < 1:
        raise ValueError("bootstrap resample counts must be positive")
    events = (scores[:, :, None] <= thresholds[None, :, :]).to(weights.dtype)
    numerator = (weights * events).permute(0, 2, 1).contiguous()
    denominator = weights.permute(0, 2, 1).contiguous()
    return _practical_bootstrap_from_ratio_contributions(
        numerator,
        denominator,
        lower_tail=lower_tail,
        n_resamples=n_resamples,
        seed=seed,
        resample_batch_size=resample_batch_size,
        cluster_ids=cluster_ids,
        simultaneous=False,
    )


@torch.no_grad()
def _practical_bootstrap_from_ratio_contributions(
    numerator: Tensor,
    denominator: Tensor,
    *,
    lower_tail: float,
    n_resamples: int,
    seed: int,
    resample_batch_size: int,
    cluster_ids: Tensor | None = None,
    simultaneous: bool = True,
) -> CertificationResult:
    """Bootstrap independent clusters with a probability-preserving ratio."""

    if numerator.ndim != 3 or denominator.shape != numerator.shape:
        raise ValueError("bootstrap numerator and denominator must have shape [N,K,T]")
    if numerator.shape[0] < 1:
        raise ValueError("bootstrap needs at least one trajectory")
    if not 0.0 < lower_tail < 1.0:
        raise ValueError("bootstrap lower_tail must lie in (0, 1)")
    if n_resamples < 1 or resample_batch_size < 1:
        raise ValueError("bootstrap resample counts must be positive")
    cluster_numerator, cluster_denominator = _sum_contributions_by_cluster(
        numerator,
        denominator,
        cluster_ids,
    )
    generator = torch.Generator(device=numerator.device).manual_seed(seed)
    samples: list[Tensor] = []
    for start in range(0, n_resamples, resample_batch_size):
        count = min(resample_batch_size, n_resamples - start)
        indices = torch.randint(
            cluster_numerator.shape[0],
            (count, cluster_numerator.shape[0]),
            generator=generator,
            device=numerator.device,
        )
        multiplicities = torch.zeros(
            (count, cluster_numerator.shape[0]),
            device=numerator.device,
            dtype=numerator.dtype,
        )
        multiplicities.scatter_add_(
            1,
            indices,
            torch.ones_like(indices, dtype=numerator.dtype),
        )
        bootstrap_numerator = torch.einsum("bc,ckt->bkt", multiplicities, cluster_numerator)
        bootstrap_denominator = torch.einsum(
            "bc,ckt->bkt", multiplicities, cluster_denominator
        ).clamp_min(1e-12)
        samples.append(bootstrap_numerator / bootstrap_denominator)
    bootstrap = torch.cat(samples, dim=0)
    estimates = cluster_numerator.sum(dim=0) / cluster_denominator.sum(dim=0).clamp_min(1e-12)
    standard_errors = bootstrap.std(dim=0, unbiased=False)
    stable_errors = standard_errors.clamp_min(torch.finfo(bootstrap.dtype).eps)
    standardized_error = (bootstrap - estimates[None, :, :]) / stable_errors[None, :, :]
    if simultaneous:
        maximum_error = standardized_error.amax(dim=(1, 2))
        critical_value = torch.quantile(maximum_error, 1.0 - lower_tail).clamp_min(0.0)
        margins = critical_value * standard_errors
    else:
        critical_value = torch.quantile(
            standardized_error,
            1.0 - lower_tail,
            dim=0,
        ).clamp_min(0.0)
        margins = critical_value * standard_errors
    lower = (estimates - margins).clamp(0.0, 1.0)

    denominator_ess = _cluster_denominator_effective_sample_sizes(cluster_denominator)
    wilson_lower = _wilson_lower_bounds(
        estimates,
        denominator_ess,
        lower_tail=lower_tail,
        family_size=estimates.numel() if simultaneous else 1,
    )
    numerical_tolerance = math.sqrt(torch.finfo(standard_errors.dtype).eps)
    unstable_cells = standard_errors <= numerical_tolerance
    lower = torch.where(unstable_cells, torch.minimum(lower, wilson_lower), lower)
    realized_margins = (estimates - lower).clamp_min(0.0)
    return CertificationResult(
        estimates=estimates,
        lower_bounds=lower,
        sampling_margin=float(realized_margins.max().item()),
        ratio_error_bound=torch.full_like(lower, float("nan")),
        formal=False,
        label=(
            "practical_hajek_cluster_bootstrap_max_t_wilson_lcb"
            if simultaneous
            else "practical_hajek_patient_cluster_ordered_iut_marginal_bootstrap_wilson_lcb"
        ),
    )


def _cluster_count_effective_sample_size(
    n_trajectories: int,
    cluster_ids: Tensor | None,
) -> float:
    """Worst-case Hoeffding effective n for unequal independent clusters."""

    if cluster_ids is None:
        return float(n_trajectories)
    if cluster_ids.ndim != 1 or len(cluster_ids) != n_trajectories:
        raise ValueError("cluster_ids must have one entry per trajectory")
    cluster_sizes = torch.unique(cluster_ids.detach().cpu(), return_counts=True)[1].to(torch.float64)
    return float(n_trajectories**2 / cluster_sizes.square().sum().item())


def _sum_contributions_by_cluster(
    numerator: Tensor,
    denominator: Tensor,
    cluster_ids: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Return cluster totals while preserving the original Hájek estimate."""

    if cluster_ids is None:
        return numerator, denominator
    if cluster_ids.ndim != 1 or len(cluster_ids) != numerator.shape[0]:
        raise ValueError("cluster_ids must have one entry per trajectory")
    _, inverse = torch.unique(
        cluster_ids.detach().to(numerator.device),
        sorted=True,
        return_inverse=True,
    )
    n_clusters = int(inverse.max().item()) + 1
    cluster_numerator = torch.zeros(
        (n_clusters, *numerator.shape[1:]),
        device=numerator.device,
        dtype=numerator.dtype,
    )
    cluster_denominator = torch.zeros_like(cluster_numerator)
    cluster_numerator.index_add_(0, inverse, numerator)
    cluster_denominator.index_add_(0, inverse, denominator)
    return cluster_numerator, cluster_denominator


def _cluster_denominator_effective_sample_sizes(cluster_denominator: Tensor) -> Tensor:
    """Return denominator-weight ESS for each ``(q,t)`` cluster contribution."""

    total = cluster_denominator.sum(dim=0)
    squared = cluster_denominator.square().sum(dim=0).clamp_min(1e-12)
    return (total.square() / squared).clamp_min(1.0)


def _wilson_lower_bounds(
    estimates: Tensor,
    effective_sample_sizes: Tensor,
    *,
    lower_tail: float,
    family_size: int,
) -> Tensor:
    """One-sided Wilson guard for degenerate boundary cells."""

    if family_size < 1:
        raise ValueError("family_size must be positive")
    cell_tail = lower_tail / family_size
    standard_normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    z_value = float(
        standard_normal.icdf(torch.tensor(1.0 - cell_tail, dtype=torch.float64)).item()
    )
    z_squared = z_value**2
    sample_size = effective_sample_sizes.to(estimates).clamp_min(1.0)
    probability = estimates.clamp(0.0, 1.0)
    denominator = 1.0 + z_squared / sample_size
    center = (probability + z_squared / (2.0 * sample_size)) / denominator
    half_width = z_value / denominator * torch.sqrt(
        probability * (1.0 - probability) / sample_size
        + z_squared / (4.0 * sample_size.square())
    )
    return (center - half_width).clamp(0.0, 1.0)


def _candidate_thresholds(candidate_radii: Tensor, scores: Tensor) -> Tensor:
    """Resolve scalar or stagewise candidates to thresholds with shape [T,K]."""

    resolved = candidate_radii.to(scores)
    if resolved.ndim == 1:
        return resolved[None, :].expand(scores.shape[1], -1)
    if resolved.ndim == 2 and resolved.shape[1] == scores.shape[1]:
        return resolved.transpose(0, 1)
    raise ValueError("candidate_radii must have shape [K] or [K,T]")
