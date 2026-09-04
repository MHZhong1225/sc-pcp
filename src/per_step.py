"""Per-step decoupled coverage surfaces and oracle diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from data import TrajectoryBatch
from anchored import BehaviorAnchoredPolicy
from scores import score_batch
from simulator import rollout


def fixed_q_grid(scores: Tensor, *, size: int, lower_quantile: float, upper_quantile: float) -> Tensor:
    """Freeze the candidate radii from D_COT *step* scores before D_cert."""

    flat = scores.reshape(-1)
    if flat.numel() == 0 or not torch.isfinite(flat).all():
        raise ValueError("q-grid scores must be finite and nonempty")
    probabilities = torch.linspace(lower_quantile, upper_quantile, size, device=flat.device)
    # Keep the prespecified K candidates even when an empirical score
    # distribution has ties.  Duplicate radii are harmless finite-grid points;
    # silently collapsing them changes the union-bound family after D_COT has
    # been inspected and can make the q-conditioned network ill-defined.
    return torch.quantile(flat, probabilities)


def stage_score_quantiles(scores: Tensor, *, alpha: float) -> Tensor:
    """Return finite-sample split-conformal quantiles for every stage."""

    if scores.ndim != 2 or len(scores) == 0:
        raise ValueError("profile scores must have shape [N,T] with N > 0")
    if not torch.isfinite(scores).all() or (scores < 0.0).any():
        raise ValueError("profile scores must be finite and nonnegative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rank = min(len(scores) - 1, math.ceil((len(scores) + 1) * (1.0 - alpha)) - 1)
    return scores.sort(dim=0).values[rank].clamp_min(torch.finfo(scores.dtype).eps)


def stage_score_profile(scores: Tensor, *, alpha: float) -> Tensor:
    """Freeze a positive unit-geometric-mean stage shape from ``D_COT``."""

    return _unit_geometric_mean(stage_score_quantiles(scores, alpha=alpha))


def weighted_stage_score_quantiles(
    scores: Tensor,
    weights: Tensor,
    *,
    alpha: float,
) -> Tensor:
    """Hájek weighted left quantiles for a frozen target-policy schedule."""

    if scores.ndim != 2 or weights.shape != scores.shape or len(scores) == 0:
        raise ValueError("scores and weights must have the same nonempty [N,T] shape")
    if not torch.isfinite(scores).all() or (scores < 0.0).any():
        raise ValueError("scores must be finite and nonnegative")
    if not torch.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("weights must be finite and nonnegative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if (weights.sum(dim=0) <= 0.0).any():
        raise ValueError("every stage must have positive total weight")

    probability = 1.0 - alpha
    quantiles = []
    for time in range(scores.shape[1]):
        order = torch.argsort(scores[:, time], stable=True)
        ordered_scores = scores[order, time]
        ordered_weights = weights[order, time]
        threshold = probability * ordered_weights.sum()
        index = torch.searchsorted(
            ordered_weights.cumsum(dim=0),
            threshold,
            right=False,
        ).clamp_max(len(ordered_scores) - 1)
        quantiles.append(ordered_scores[index])
    return torch.stack(quantiles).clamp_min(torch.finfo(scores.dtype).eps)


def transport_refined_stage_profile(
    initial_quantiles: Tensor,
    fold_initial_quantiles: Tensor,
    fold_transported_quantiles: Tensor,
    fold_effective_sizes: Tensor,
    *,
    refinement_strength: float,
    maximum_profile_ratio: float,
    minimum_effective_size: float,
) -> tuple[Tensor, Tensor]:
    """Apply one regularized OOF transport correction to the stage shape.

    Corrections are learned on patient-held-out folds of ``D_COT``.  A global
    log shift is removed because it belongs to the subsequently certified
    scalar, while a trust region limits the relative change in stage shape.
    """

    if initial_quantiles.ndim != 1:
        raise ValueError("initial_quantiles must have shape [T]")
    expected = fold_initial_quantiles.shape
    if (
        fold_initial_quantiles.ndim != 2
        or fold_transported_quantiles.shape != expected
        or fold_effective_sizes.shape != expected
        or expected[1] != len(initial_quantiles)
    ):
        raise ValueError("fold quantities must share shape [F,T]")
    positive = bool((initial_quantiles > 0.0).all())
    positive = positive and bool((fold_initial_quantiles > 0.0).all())
    positive = positive and bool((fold_transported_quantiles > 0.0).all())
    finite = all(
        torch.isfinite(value).all()
        for value in (
            initial_quantiles,
            fold_initial_quantiles,
            fold_transported_quantiles,
            fold_effective_sizes,
        )
    )
    if not positive or not finite or bool((fold_effective_sizes < 0.0).any()):
        raise ValueError("profile refinement inputs must be finite with positive quantiles")
    if not 0.0 < refinement_strength <= 1.0:
        raise ValueError("refinement_strength must lie in (0, 1]")
    if maximum_profile_ratio <= 1.0 or minimum_effective_size <= 0.0:
        raise ValueError("profile ratio and minimum effective size must be valid")

    fold_corrections = (fold_transported_quantiles / fold_initial_quantiles).log()
    eligible_weights = torch.where(
        fold_effective_sizes >= minimum_effective_size,
        fold_effective_sizes,
        torch.zeros_like(fold_effective_sizes),
    )
    weight_sum = eligible_weights.sum(dim=0)
    mean_correction = (
        (eligible_weights * fold_corrections).sum(dim=0)
        / weight_sum.clamp_min(1e-12)
    )
    mean_correction = torch.where(weight_sum > 0.0, mean_correction, torch.zeros_like(mean_correction))

    active = weight_sum > 0.0
    center = mean_correction[active].mean() if bool(active.any()) else mean_correction.new_zeros(())
    applied = torch.where(
        active,
        refinement_strength * (mean_correction - center),
        torch.zeros_like(mean_correction),
    )
    span = applied.max() - applied.min()
    maximum_span = math.log(maximum_profile_ratio)
    if float(span.item()) > maximum_span:
        applied = applied * (maximum_span / span)
    refined_quantiles = initial_quantiles * applied.exp()
    return _unit_geometric_mean(refined_quantiles), applied


def profiled_local_scale_grid(
    scores: Tensor,
    profile: Tensor,
    *,
    size: int,
    lower_quantile: float,
    upper_quantile: float,
    anchor_scale: float | Tensor,
    focus_fraction: float,
    focus_radius: float,
) -> Tensor:
    """Freeze a broad guarded grid with dense knots near a D_COT anchor."""

    if size < 3:
        raise ValueError("local scale grid needs at least three candidates")
    if not 0.0 < focus_fraction < 1.0 or focus_radius <= 0.0:
        raise ValueError("grid focus settings must be positive and nondegenerate")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("grid quantiles must lie in [0, 1] and be ordered")
    if 2.0 * focus_radius >= upper_quantile - lower_quantile:
        raise ValueError("grid focus radius is too large for the quantile range")
    if scores.ndim != 2 or len(scores) == 0 or profile.shape != (scores.shape[1],):
        raise ValueError("scores and profile must have shapes [N,T] and [T]")
    if not torch.isfinite(scores).all() or (scores < 0.0).any():
        raise ValueError("grid scores must be finite and nonnegative")
    if not torch.isfinite(profile).all() or (profile <= 0.0).any():
        raise ValueError("profile must be finite and positive")
    anchor = torch.as_tensor(anchor_scale, device=scores.device, dtype=scores.dtype)
    if anchor.ndim != 0 or not bool(torch.isfinite(anchor)) or float(anchor.item()) <= 0.0:
        raise ValueError("anchor_scale must be finite and positive")

    normalized = (scores / profile.to(scores)[None, :]).reshape(-1)
    anchor_probability = (normalized <= anchor).to(scores.dtype).mean()
    focus_low = float(
        (anchor_probability - focus_radius).clamp(lower_quantile, upper_quantile).item()
    )
    focus_high = float(
        (anchor_probability + focus_radius).clamp(lower_quantile, upper_quantile).item()
    )
    if focus_high <= focus_low:
        midpoint = min(max(float(anchor_probability.item()), lower_quantile), upper_quantile)
        focus_low = max(lower_quantile, midpoint - focus_radius)
        focus_high = min(upper_quantile, midpoint + focus_radius)

    focus_count = min(size - 2, max(1, round(size * focus_fraction)))
    guard_count = size - focus_count
    lower_count = guard_count // 2
    upper_count = guard_count - lower_count
    lower = torch.linspace(lower_quantile, focus_low, lower_count + 1, device=scores.device)[:-1]
    focus = (
        torch.tensor([(focus_low + focus_high) / 2.0], device=scores.device)
        if focus_count == 1
        else torch.linspace(focus_low, focus_high, focus_count, device=scores.device)
    )
    upper = torch.linspace(focus_high, upper_quantile, upper_count + 1, device=scores.device)[1:]
    probabilities = torch.cat((lower, focus, upper)).to(scores)
    return torch.quantile(normalized, probabilities)


def _unit_geometric_mean(values: Tensor) -> Tensor:
    positive = values.clamp_min(torch.finfo(values.dtype).eps)
    return positive / positive.log().mean().exp()


def profiled_scale_grid(
    scores: Tensor,
    profile: Tensor,
    *,
    size: int,
    lower_quantile: float,
    upper_quantile: float,
) -> Tensor:
    """Freeze a one-dimensional candidate scale grid before certification."""

    if scores.ndim != 2 or profile.shape != (scores.shape[1],):
        raise ValueError("scores and profile must have shapes [N,T] and [T]")
    if not torch.isfinite(profile).all() or (profile <= 0.0).any():
        raise ValueError("profile must be finite and strictly positive")
    return fixed_q_grid(
        scores / profile.to(scores)[None, :],
        size=size,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )


def candidate_radius_schedules(scale_grid: Tensor, profile: Tensor) -> Tensor:
    """Return the frozen candidate radii with shape ``[K,T]``."""

    if scale_grid.ndim != 1 or profile.ndim != 1:
        raise ValueError("scale_grid and profile must both be one-dimensional")
    if not torch.isfinite(scale_grid).all() or (scale_grid < 0.0).any():
        raise ValueError("scale_grid must be finite and nonnegative")
    if not torch.isfinite(profile).all() or (profile <= 0.0).any():
        raise ValueError("profile must be finite and strictly positive")
    return scale_grid[:, None] * profile.to(scale_grid)[None, :]


@torch.no_grad()
def dcov_surface(weights: Tensor, scores: Tensor, q_measure: Tensor) -> Tensor:
    """Estimate F[q_deploy, time, q_measure] from per-step weights.

    ``weights`` has shape [N, T, K_deploy], while the result has shape
    [K_deploy, T, K_measure].
    """

    if weights.shape[:2] != scores.shape:
        raise ValueError("weights and scores must agree on [N,T]")
    events = (scores[:, :, None] <= q_measure.to(scores)[None, None, :]).to(weights.dtype)
    return torch.einsum("ntk,ntm->ktm", weights, events) / scores.shape[0]


@torch.no_grad()
def diagonal_coverage_estimates(weights: Tensor, scores: Tensor, candidate_radii: Tensor) -> Tensor:
    """Return self-consistent coverage estimates with shape ``[K,T]``.

    ``candidate_radii`` may be the legacy scalar grid ``[K]`` or a frozen
    stagewise schedule family ``[K,T]``.
    """

    thresholds = _candidate_thresholds(candidate_radii, scores)
    if weights.shape[:2] != scores.shape or weights.shape[2] != thresholds.shape[1]:
        raise ValueError("weights, scores, and candidate radii must share [N,T,K]")
    events = (scores[:, :, None] <= thresholds[None, :, :]).to(weights.dtype)
    return (weights * events).mean(dim=0).transpose(0, 1)


@torch.no_grad()
def self_normalized_diagonal_coverage_estimates(
    weights: Tensor,
    scores: Tensor,
    candidate_radii: Tensor,
) -> Tensor:
    """Probability-preserving practical coverage estimates of shape [K,T]."""

    thresholds = _candidate_thresholds(candidate_radii, scores)
    if weights.shape[:2] != scores.shape or weights.shape[2] != thresholds.shape[1]:
        raise ValueError("weights, scores, and candidate radii must share [N,T,K]")
    events = (scores[:, :, None] <= thresholds[None, :, :]).to(weights.dtype)
    numerator = (weights * events).sum(dim=0)
    denominator = weights.sum(dim=0).clamp_min(1e-12)
    return (numerator / denominator).transpose(0, 1)


@torch.no_grad()
def self_normalized_dcov_surface(weights: Tensor, scores: Tensor, q_measure: Tensor) -> Tensor:
    """Probability-preserving practical DCov surface [K_deploy,T,K_measure]."""

    if weights.shape[:2] != scores.shape:
        raise ValueError("weights and scores must agree on [N,T]")
    events = (scores[:, :, None] <= q_measure.to(scores)[None, None, :]).to(weights.dtype)
    numerator = torch.einsum("ntk,ntm->ktm", weights, events)
    denominator = weights.sum(dim=0).transpose(0, 1)[:, :, None].clamp_min(1e-12)
    return numerator / denominator


@torch.no_grad()
def effective_sample_sizes(weights: Tensor, cluster_ids: Tensor | None = None) -> Tensor:
    """Return ESS[q, t] over independent trajectory or patient clusters."""

    if weights.ndim != 3:
        raise ValueError("weights must have shape [N,T,K]")
    independent_weights = weights
    if cluster_ids is not None:
        if cluster_ids.ndim != 1 or len(cluster_ids) != len(weights):
            raise ValueError("cluster_ids must have shape [N]")
        _, inverse = torch.unique(cluster_ids.to(weights.device), sorted=True, return_inverse=True)
        independent_weights = torch.zeros(
            (int(inverse.max().item()) + 1, *weights.shape[1:]),
            dtype=weights.dtype,
            device=weights.device,
        )
        independent_weights.index_add_(0, inverse, weights)
    numerator = independent_weights.sum(dim=0).square()
    denominator = independent_weights.square().sum(dim=0).clamp_min(1e-12)
    return (numerator / denominator).transpose(0, 1)


@dataclass(frozen=True)
class OracleSurface:
    surface: Tensor
    q_grid: Tensor

    @property
    def diagonal(self) -> Tensor:
        index = torch.arange(len(self.q_grid), device=self.surface.device)
        return self.surface[index, :, index]


@torch.no_grad()
def estimate_oracle_surface(
    environment: object,
    policy: BehaviorAnchoredPolicy,
    outcome_model: object,
    *,
    q_grid: Tensor,
    horizon: int,
    n_rollouts: int,
    seed: int,
    device: str | torch.device,
) -> OracleSurface:
    """Large-rollout reference for the full per-step DCov surface."""

    q_grid = q_grid.to(device)
    rows = []
    for index, radius in enumerate(q_grid):
        deployed = rollout(
            environment,
            policy,
            n=n_rollouts,
            horizon=horizon,
            seed=seed + 104_729 * index,
            device=device,
            q=radius,
        )
        scores = score_batch(outcome_model, deployed.current_states(), deployed.actions, deployed.outcomes)
        rows.append((scores[:, :, None] <= q_grid[None, None, :]).float().mean(dim=0))
    return OracleSurface(surface=torch.stack(rows), q_grid=q_grid)


@torch.no_grad()
def estimate_oracle_diagonal(
    environment: object,
    policy: BehaviorAnchoredPolicy,
    outcome_model: object,
    *,
    q_grid: Tensor,
    horizon: int,
    n_rollouts: int,
    seed: int,
    device: str | torch.device,
) -> Tensor:
    """Estimate only ``P_q(score_t <= q)`` for the RQ4 mechanism figure.

    The old full-surface routine evaluates every deployment radius against
    every measurement radius.  RQ4 uses only the self-consistent diagonal, so
    materializing the extra ``K x T x K`` surface wastes memory and arithmetic.
    """

    rows = []
    for index, radius in enumerate(q_grid.to(device)):
        coverage, _, _ = per_step_oracle_metrics(
            environment,
            policy,
            outcome_model,
            q=radius,
            horizon=horizon,
            n_rollouts=n_rollouts,
            seed=seed + 104_729 * index,
            device=device,
        )
        rows.append(coverage)
    return torch.stack(rows)


@torch.no_grad()
def per_step_oracle_metrics(
    environment: object,
    policy: BehaviorAnchoredPolicy,
    outcome_model: object,
    *,
    q: float | Tensor,
    horizon: int,
    n_rollouts: int,
    seed: int,
    device: str | torch.device,
) -> tuple[Tensor, TrajectoryBatch, Tensor]:
    """Fresh target-policy coverage, trajectory batch, and scores for one rule."""

    deployed = rollout(
        environment,
        policy,
        n=n_rollouts,
        horizon=horizon,
        seed=seed,
        device=device,
        q=q,
    )
    scores = score_batch(outcome_model, deployed.current_states(), deployed.actions, deployed.outcomes)
    radius = torch.as_tensor(q, device=scores.device, dtype=scores.dtype)
    if radius.ndim == 0:
        coverage = (scores <= radius).float().mean(dim=0)
    elif radius.shape == (horizon,):
        coverage = (scores <= radius[None, :]).float().mean(dim=0)
    else:
        raise ValueError("oracle metric radius must be scalar or [T]")
    return coverage, deployed, scores


def _candidate_thresholds(candidate_radii: Tensor, scores: Tensor) -> Tensor:
    """Resolve scalar or stagewise candidates to a ``[T,K]`` threshold matrix."""

    resolved = candidate_radii.to(scores)
    if resolved.ndim == 1:
        return resolved[None, :].expand(scores.shape[1], -1)
    if resolved.ndim == 2 and resolved.shape[1] == scores.shape[1]:
        return resolved.transpose(0, 1)
    raise ValueError("candidate radii must have shape [K] or [K,T]")


@torch.no_grad()
def response_operator(surface: Tensor, q_grid: Tensor, alpha: float) -> Tensor:
    """Empirical T(q)=max_t F^{-1}_{q,t}(1-alpha) from a DCov surface."""

    target = 1.0 - alpha
    values = []
    for deploy in range(len(q_grid)):
        thresholds = []
        for time in range(surface.shape[1]):
            safe = (surface[deploy, time] >= target).nonzero().squeeze(1)
            thresholds.append(q_grid[safe[0]] if len(safe) else torch.tensor(float("inf"), device=q_grid.device))
        values.append(torch.stack(thresholds).max())
    return torch.stack(values)
