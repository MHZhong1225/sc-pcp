"""Per-step decoupled coverage surfaces and oracle diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from scpcp.data import TrajectoryBatch
from scpcp.policy import BehaviorAnchoredPolicy
from scpcp.scores import score_batch
from scpcp.simulator import rollout


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


def stage_score_profile(scores: Tensor, *, alpha: float) -> Tensor:
    """Freeze the relative stage shape from ``D_COT`` only.

    Each component is the ordinary finite-sample split-conformal quantile for
    that stage.  The subsequent certification step learns only one global
    scale, so the deployed radii are ``q_t(s) = s * b_t``.
    """

    if scores.ndim != 2 or len(scores) == 0:
        raise ValueError("profile scores must have shape [N,T] with N > 0")
    if not torch.isfinite(scores).all() or (scores < 0.0).any():
        raise ValueError("profile scores must be finite and nonnegative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rank = min(len(scores) - 1, math.ceil((len(scores) + 1) * (1.0 - alpha)) - 1)
    raw_profile = scores.sort(dim=0).values[rank].clamp_min(torch.finfo(scores.dtype).eps)
    geometric_mean = raw_profile.log().mean().exp()
    return raw_profile / geometric_mean


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
