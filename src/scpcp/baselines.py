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

from scpcp.coverage import candidate_radius_schedules, diagonal_coverage_estimates
from scpcp.data import TrajectoryBatch
from scpcp.policy import BehaviorAnchoredPolicy
from scpcp.selection import RadiusSelection, select_empirical_radius
from scpcp.simulator import rollout


def standard_cp_stagewise_radii(scores: Tensor, alpha: float) -> Tensor:
    """Ordinary split-CP finite-sample radius at each decision stage."""

    if scores.ndim != 2 or len(scores) == 0:
        raise ValueError("scores must have shape [N,T] with N > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rank = min(scores.shape[0] - 1, math.ceil((scores.shape[0] + 1) * (1.0 - alpha)) - 1)
    return scores.sort(dim=0).values[rank]


def finite_depth_mfcs_selection(
    batch: TrajectoryBatch,
    scores: Tensor,
    *,
    q_grid: Tensor,
    stage_profile: Tensor | None = None,
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
    profile = (
        torch.ones(batch.horizon, device=device, dtype=q_grid.dtype)
        if stage_profile is None
        else stage_profile.to(device=device, dtype=q_grid.dtype)
    )
    if profile.shape != (batch.horizon,):
        raise ValueError("stage_profile must have shape [T]")
    weights = []
    for time in range(batch.horizon):
        log_weight = torch.zeros((batch.n, len(q_grid)), device=device)
        for previous in range(max(0, time - depth + 1), time + 1):
            states = batch.states[:, previous]
            pi = target_policy.probabilities_for_grid(states, q_grid * profile[previous])
            action = batch.actions[:, previous, None, None].expand(-1, len(q_grid), 1)
            numerator = pi.gather(2, action).squeeze(2)
            denominator = logging_policy.probabilities(states).gather(1, batch.actions[:, previous, None]).expand(-1, len(q_grid))
            log_weight += (numerator.clamp_min(1e-12) / denominator.clamp_min(1e-12)).log()
        weights.append(log_weight.exp().clamp_max(weight_cap))
    candidate_radii = candidate_radius_schedules(q_grid, profile)
    estimates = diagonal_coverage_estimates(
        torch.stack(weights, dim=1),
        scores,
        candidate_radii,
    )
    return select_empirical_radius(q_grid, estimates, alpha=alpha), estimates


@dataclass(frozen=True)
class OnlineBaselineResult:
    radius_by_time: Tensor
    target_deployments: int
    rounds: int
    adaptation_per_time_coverage: Tensor
    adaptation_round_worst_coverage: tuple[float, ...]
    adaptation_pathwise_coverage: float
    selected_scale: float | None = None


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

    def finish(
        self,
        radii: Tensor,
        rounds: int,
        *,
        selected_scale: float | None = None,
    ) -> OnlineBaselineResult:
        if self.trajectories < 1:
            raise RuntimeError("online adaptation produced no trajectories")
        return OnlineBaselineResult(
            radius_by_time=radii,
            target_deployments=self.trajectories,
            rounds=rounds,
            adaptation_per_time_coverage=(self.covered_by_time / self.trajectories).to(torch.float32),
            adaptation_round_worst_coverage=tuple(self.round_worst),
            adaptation_pathwise_coverage=self.pathwise_hits / self.trajectories,
            selected_scale=selected_scale,
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
def prc_profile_scale(
    environment: object,
    policy: BehaviorAnchoredPolicy,
    outcome_model: object,
    initial_scale: float,
    scale_grid: Tensor,
    stage_profile: Tensor,
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
    """PRC-MaxTime adapter on the same frozen ``q_t(s)=s b_t`` family."""

    if stage_profile.shape != (horizon,):
        raise ValueError("stage_profile must have shape [T]")
    current_scale = float(initial_scale)
    scale_grid = scale_grid.detach().cpu()
    profile = stage_profile.detach().cpu()
    candidate_radii = candidate_radius_schedules(scale_grid, profile)
    round_sizes = _online_round_sizes(total_rollouts, rounds)
    adaptation = _CoverageAccumulator.create(horizon)
    for round_index, rollout_size in enumerate(round_sizes):
        margin = math.sqrt(math.log(len(scale_grid) * horizon / delta) / (2.0 * rollout_size))
        radii = current_scale * profile
        deployed = rollout(
            environment,
            policy,
            n=rollout_size,
            horizon=horizon,
            seed=seed + 61_103 * round_index,
            device=device,
            q=radii.to(device),
        )
        from scpcp.scores import score_batch

        scores = score_batch(
            outcome_model,
            deployed.current_states(),
            deployed.actions,
            deployed.outcomes,
        ).cpu()
        adaptation.update(scores, radii)
        coverage = (scores[:, None, :] <= candidate_radii[None, :, :]).float().mean(dim=0)
        safe = (coverage.amin(dim=1) - margin >= 1.0 - alpha).nonzero().squeeze(1)
        guarded = [
            index
            for index in safe.tolist()
            if abs(float(scale_grid[index]) - current_scale) <= maximum_step
        ]
        if guarded:
            current_scale = float(scale_grid[min(guarded)].item())
    final_radii = current_scale * profile
    return adaptation.finish(final_radii, rounds, selected_scale=current_scale)


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
