"""Task-aligned per-step baselines and transparent adaptations.

The vendored MFCS, MultiDimSPCI, and PRC repositories are valuable references
but are not drop-in multivariate logged-trajectory implementations.  The
functions below are explicitly labelled adapters: they preserve each method's
information regime while using the common frozen score and treatment-policy
interface required for a fair SC-PCP experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from scpcp.coverage import diagonal_coverage_estimates
from scpcp.data import TrajectoryBatch
from scpcp.policy import BehaviorAnchoredPolicy
from scpcp.selection import RadiusSelection, select_empirical_radius
from scpcp.simulator import rollout


def historical_per_step_radius(scores: Tensor, alpha: float) -> float:
    """Conservative split CP radius: the largest of the T stage quantiles."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [N,T]")
    rank = min(scores.shape[0] - 1, math.ceil((scores.shape[0] + 1) * (1.0 - alpha)) - 1)
    stage_radii = scores.sort(dim=0).values[rank]
    return float(stage_radii.max().item())


def finite_depth_mfcs_selection(
    batch: TrajectoryBatch,
    scores: Tensor,
    *,
    q_grid: Tensor,
    target_policy: BehaviorAnchoredPolicy,
    logging_policy: object,
    depth: int,
    alpha: float,
    weight_cap: float,
) -> tuple[RadiusSelection, Tensor]:
    """Split-MFCS-style finite-depth feedback correction on the scalar score."""

    if depth < 1:
        raise ValueError("MFCS depth must be positive")
    device = scores.device
    q_grid = q_grid.to(device)
    weights = []
    for time in range(batch.horizon):
        log_weight = torch.zeros((batch.n, len(q_grid)), device=device)
        for previous in range(max(0, time - depth + 1), time + 1):
            states = batch.states[:, previous]
            pi = target_policy.probabilities_for_grid(states, q_grid)
            action = batch.actions[:, previous, None, None].expand(-1, len(q_grid), 1)
            numerator = pi.gather(2, action).squeeze(2)
            denominator = logging_policy.probabilities(states).gather(1, batch.actions[:, previous, None]).expand(-1, len(q_grid))
            log_weight += (numerator.clamp_min(1e-12) / denominator.clamp_min(1e-12)).log()
        weights.append(log_weight.exp().clamp_max(weight_cap))
    estimates = diagonal_coverage_estimates(torch.stack(weights, dim=1), scores, q_grid)
    return select_empirical_radius(q_grid, estimates, alpha=alpha), estimates


@dataclass(frozen=True)
class OnlineBaselineResult:
    radius_by_time: Tensor
    target_deployments: int
    rounds: int
    adaptation_per_time_coverage: Tensor
    adaptation_round_worst_coverage: tuple[float, ...]
    adaptation_pathwise_coverage: float


@dataclass
class _CoverageAccumulator:
    covered_by_time: Tensor
    trajectories: int = 0
    pathwise_hits: int = 0
    round_worst: list[float] = field(default_factory=list)

    @classmethod
    def create(cls, horizon: int) -> "_CoverageAccumulator":
        return cls(torch.zeros(horizon, dtype=torch.float64))

    def update(self, scores: Tensor, radius: float | Tensor) -> None:
        scores = scores.detach().cpu()
        threshold = torch.as_tensor(radius, dtype=scores.dtype).cpu()
        hits = scores <= threshold
        per_time = hits.to(torch.float64).mean(dim=0)
        self.covered_by_time += hits.to(torch.float64).sum(dim=0)
        self.trajectories += len(scores)
        self.pathwise_hits += int(hits.all(dim=1).sum().item())
        self.round_worst.append(float(per_time.min().item()))

    def finish(self, radii: Tensor, rounds: int) -> OnlineBaselineResult:
        if self.trajectories < 1:
            raise RuntimeError("online adaptation produced no trajectories")
        return OnlineBaselineResult(
            radius_by_time=radii,
            target_deployments=self.trajectories,
            rounds=rounds,
            adaptation_per_time_coverage=(self.covered_by_time / self.trajectories).to(torch.float32),
            adaptation_round_worst_coverage=tuple(self.round_worst),
            adaptation_pathwise_coverage=self.pathwise_hits / self.trajectories,
        )


@torch.no_grad()
def aci_style_controller(
    environment: object,
    policy: BehaviorAnchoredPolicy,
    outcome_model: object,
    initial_scores: Tensor,
    *,
    alpha: float,
    gamma: float,
    rounds: int,
    total_rollouts: int,
    horizon: int,
    seed: int,
    device: str | torch.device,
) -> OnlineBaselineResult:
    """Stagewise online ACI adaptation; it intentionally consumes on-policy data."""

    histories = [initial_scores[:, time].detach().cpu() for time in range(horizon)]
    alpha_time = torch.full((horizon,), alpha)
    round_sizes = _online_round_sizes(total_rollouts, rounds)
    adaptation = _CoverageAccumulator.create(horizon)
    for round_index, rollout_size in enumerate(round_sizes):
        radii = torch.stack([_finite_quantile(history, 1.0 - float(alpha_time[t])) for t, history in enumerate(histories)])
        deployed = rollout(
            environment,
            policy,
            n=rollout_size,
            horizon=horizon,
            seed=seed + 17_923 * round_index,
            device=device,
            q=radii.to(device),
        )
        from scpcp.scores import score_batch

        scores = score_batch(outcome_model, deployed.current_states(), deployed.actions, deployed.outcomes).cpu()
        adaptation.update(scores, radii)
        misses = (scores > radii[None, :]).float().mean(dim=0)
        alpha_time = (alpha_time + gamma * (alpha - misses)).clamp(0.001, 0.999)
        for time in range(horizon):
            histories[time] = torch.cat((histories[time], scores[:, time]))[-10_000:]
    radii = torch.stack([_finite_quantile(history, 1.0 - float(alpha_time[t])) for t, history in enumerate(histories)])
    return adaptation.finish(radii, rounds)


@torch.no_grad()
def repeated_recalibration(
    environment: object,
    policy: BehaviorAnchoredPolicy,
    outcome_model: object,
    initial_radius: float,
    *,
    alpha: float,
    rounds: int,
    total_rollouts: int,
    horizon: int,
    seed: int,
    device: str | torch.device,
) -> OnlineBaselineResult:
    """An on-policy repeated-calibration diagnostic for the DCov off-diagonal."""

    radius = initial_radius
    round_sizes = _online_round_sizes(total_rollouts, rounds)
    adaptation = _CoverageAccumulator.create(horizon)
    for round_index, rollout_size in enumerate(round_sizes):
        deployed = rollout(
            environment,
            policy,
            n=rollout_size,
            horizon=horizon,
            seed=seed + 31_337 * round_index,
            device=device,
            q=radius,
        )
        from scpcp.scores import score_batch

        scores = score_batch(outcome_model, deployed.current_states(), deployed.actions, deployed.outcomes)
        adaptation.update(scores, radius)
        radius = historical_per_step_radius(scores, alpha)
    return adaptation.finish(torch.full((horizon,), radius), rounds)


@torch.no_grad()
def multidim_spci_style_controller(
    environment: object,
    policy: BehaviorAnchoredPolicy,
    outcome_model: object,
    initial_scores: Tensor,
    *,
    alpha: float,
    rounds: int,
    total_rollouts: int,
    horizon: int,
    seed: int,
    device: str | torch.device,
    residual_window: int = 1_000,
) -> OnlineBaselineResult:
    """Online multivariate-score adaptation when native MultiDimSPCI is unavailable.

    It uses the common normalized multivariate score and maintains a separate
    recent residual buffer for every decision stage.  Results are labelled
    ``MultiDimSPCI-style`` rather than presented as a native upstream run.
    """

    histories = [initial_scores[:, time].detach().cpu()[-residual_window:] for time in range(horizon)]
    round_sizes = _online_round_sizes(total_rollouts, rounds)
    adaptation = _CoverageAccumulator.create(horizon)
    for round_index, rollout_size in enumerate(round_sizes):
        radii = torch.stack([_finite_quantile(history, 1.0 - alpha) for history in histories])
        deployed = rollout(
            environment,
            policy,
            n=rollout_size,
            horizon=horizon,
            seed=seed + 47_021 * round_index,
            device=device,
            q=radii.to(device),
        )
        from scpcp.scores import score_batch

        scores = score_batch(outcome_model, deployed.current_states(), deployed.actions, deployed.outcomes).cpu()
        adaptation.update(scores, radii)
        for time in range(horizon):
            histories[time] = torch.cat((histories[time], scores[:, time]))[-residual_window:]
    radii = torch.stack([_finite_quantile(history, 1.0 - alpha) for history in histories])
    return adaptation.finish(radii, rounds)


@torch.no_grad()
def prc_max_time(
    environment: object,
    policy: BehaviorAnchoredPolicy,
    outcome_model: object,
    initial_radius: float,
    q_grid: Tensor,
    *,
    alpha: float,
    delta: float,
    rounds: int,
    total_rollouts: int,
    horizon: int,
    seed: int,
    device: str | torch.device,
    maximum_step: float = 0.35,
) -> OnlineBaselineResult:
    """PRC-MaxTime adapter with explicit on-policy samples and finite-grid moves.

    Unlike native PRC's monotone scalar binary-search interface, this wrapper
    enumerates the fixed grid because performative coverage need not be
    monotone in q.  It may move only within a declared radius sensitivity guard.
    """

    radius = float(initial_radius)
    q_grid = q_grid.detach().cpu()
    round_sizes = _online_round_sizes(total_rollouts, rounds)
    adaptation = _CoverageAccumulator.create(horizon)
    for round_index, rollout_size in enumerate(round_sizes):
        margin = math.sqrt(math.log(len(q_grid) * horizon / delta) / (2.0 * rollout_size))
        deployed = rollout(
            environment,
            policy,
            n=rollout_size,
            horizon=horizon,
            seed=seed + 61_103 * round_index,
            device=device,
            q=radius,
        )
        from scpcp.scores import score_batch

        scores = score_batch(outcome_model, deployed.current_states(), deployed.actions, deployed.outcomes).cpu()
        adaptation.update(scores, radius)
        coverage = (scores[:, :, None] <= q_grid[None, None, :]).float().mean(dim=0).transpose(0, 1)
        safe = (coverage.amin(dim=1) - margin >= 1.0 - alpha).nonzero().squeeze(1)
        guarded = [index for index in safe.tolist() if abs(float(q_grid[index]) - radius) <= maximum_step]
        if guarded:
            radius = float(q_grid[min(guarded)])
    return adaptation.finish(torch.full((horizon,), radius), rounds)


def _finite_quantile(values: Tensor, probability: float) -> Tensor:
    rank = min(len(values) - 1, max(0, math.ceil((len(values) + 1) * probability) - 1))
    return values.sort().values[rank]


def _online_round_sizes(total_rollouts: int, rounds: int) -> tuple[int, ...]:
    """Allocate one fixed on-policy budget without dropping samples."""

    if total_rollouts < 1 or rounds < 1:
        raise ValueError("online total_rollouts and rounds must be positive")
    if total_rollouts < rounds:
        raise ValueError("online total_rollouts must be at least the number of adaptation rounds")
    base, remainder = divmod(total_rollouts, rounds)
    return tuple(base + (round_index < remainder) for round_index in range(rounds))
