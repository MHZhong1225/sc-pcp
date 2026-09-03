"""Equal-marginal copula benchmark for policy-mediated score-law shift.

This module is isolated from the canonical SC-PCP implementation.  Its two
standardized outcome coordinates are conditionally standard normal for every
observed regime/action cell.  Treatment changes only the observed next regime,
which changes only the cross-outcome correlation and hence the normalized-max
score law.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import Tensor

from scpcp.copula_benchmark_config import CopulaDGPConfig


@dataclass(frozen=True)
class CopulaNoise:
    """Common random numbers shared by source and target rollouts."""

    initial_uniform: Tensor
    action_uniforms: Tensor
    transition_uniforms: Tensor
    outcome_normals: Tensor

    @property
    def n(self) -> int:
        return len(self.initial_uniform)

    @property
    def horizon(self) -> int:
        return self.action_uniforms.shape[1]


@dataclass(frozen=True)
class CopulaTrajectory:
    """One rollout with the observed hard/easy regime included in state."""

    states: Tensor
    actions: Tensor
    residuals: Tensor
    action_one_probabilities: Tensor

    @property
    def current_hard(self) -> Tensor:
        return self.states[:, :-1, 0]

    @property
    def next_hard(self) -> Tensor:
        return self.states[:, 1:, 0]

    @property
    def scores(self) -> Tensor:
        return self.residuals.abs().amax(dim=2)


@dataclass(frozen=True)
class MarginalAudit:
    """Empirical audit of the exact equal-marginal construction."""

    maximum_absolute_mean: float
    maximum_variance_error: float
    maximum_correlation_error: float
    minimum_regime_action_count: int


@dataclass(frozen=True)
class PrefixOverlapDiagnostics:
    ess_fraction: Tensor
    maximum_normalized_weight_share: Tensor
    log_weight_span: Tensor
    minimum_incremental_ratio: Tensor
    maximum_incremental_ratio: Tensor


@dataclass(frozen=True)
class CopulaSourceReference:
    """A logging-policy rollout prepared once for one beta cell."""

    kernel_fingerprint: str
    trajectory: CopulaTrajectory
    q90: Tensor
    audit: MarginalAudit


@dataclass(frozen=True)
class CopulaMechanismResult:
    """Same-radius source/target mechanism metrics for one factorial cell."""

    beta: float
    kappa: float
    radius: float
    kernel_fingerprint: str
    source_q90: Tensor
    target_q90: Tensor
    source_coverage: Tensor
    target_coverage: Tensor
    policy_tv_on_source: Tensor
    source_action_rate: Tensor
    target_action_rate: Tensor
    source_hard_prevalence: Tensor
    target_hard_prevalence: Tensor
    overlap: PrefixOverlapDiagnostics
    source_audit: MarginalAudit
    target_audit: MarginalAudit


@dataclass(frozen=True)
class LoggingPolicy:
    """Known behavior policy depending only on the observed current regime."""

    config: CopulaDGPConfig

    def action_one_probability(self, hard: Tensor) -> Tensor:
        return _action_one_probability(self.config, hard, logit_shift=0.0)


@dataclass(frozen=True)
class RadiusResponsivePolicy:
    """Nonanticipating target policy: radius enters only its action logits."""

    config: CopulaDGPConfig
    radius: float
    kappa: float

    def __post_init__(self) -> None:
        if self.kappa < 0.0:
            raise ValueError("kappa must be nonnegative")

    def action_one_probability(self, hard: Tensor) -> Tensor:
        response = _endpoint_normalized_response(self.config, self.radius, like=hard)
        shift = self.kappa * self.config.maximum_policy_logit_shift * response
        return _action_one_probability(self.config, hard, logit_shift=shift)


@dataclass(frozen=True)
class CopulaKernel:
    """Policy-independent transition/outcome kernel for a fixed beta cell.

    The class deliberately has no radius or kappa field.  Source and target are
    rolled out through the same instance, so q can affect the data law only via
    ``RadiusResponsivePolicy -> action -> next_hard``.
    """

    config: CopulaDGPConfig
    beta: float

    def hard_probability(self, current_hard: Tensor, action: Tensor) -> Tensor:
        logits = (
            self.config.hard_transition_intercept
            + self.config.hard_persistence * current_hard
            + self.beta * (2.0 * action.to(current_hard) - 1.0)
        )
        return torch.sigmoid(logits)

    def correlation(self, next_hard: Tensor) -> Tensor:
        easy = next_hard.new_tensor(self.config.easy_correlation)
        hard = next_hard.new_tensor(self.config.hard_correlation)
        return torch.where(next_hard.bool(), hard, easy)

    def standardized_residuals(self, next_hard: Tensor, normals: Tensor) -> Tensor:
        """Return two N(0,1) marginals with regime-specific correlation."""

        if normals.shape != (*next_hard.shape, 2):
            raise ValueError("normals must have shape [N, 2] aligned with next_hard")
        correlation = self.correlation(next_hard)
        first = normals[..., 0]
        second = correlation * first + torch.sqrt(1.0 - correlation.square()) * normals[..., 1]
        return torch.stack((first, second), dim=-1)

    @property
    def fingerprint(self) -> str:
        payload = {"beta": self.beta, "dgp": self.config.__dict__}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def make_copula_noise(
    *,
    n: int,
    horizon: int,
    seed: int,
    device: str | torch.device,
) -> CopulaNoise:
    """Generate one paired source/target noise bundle."""

    if n < 1 or horizon < 1:
        raise ValueError("n and horizon must be positive")
    generator = torch.Generator(device=device).manual_seed(seed)
    return CopulaNoise(
        initial_uniform=torch.rand(n, generator=generator, device=device, dtype=torch.float64),
        action_uniforms=torch.rand(
            (n, horizon), generator=generator, device=device, dtype=torch.float64
        ),
        transition_uniforms=torch.rand(
            (n, horizon), generator=generator, device=device, dtype=torch.float64
        ),
        outcome_normals=torch.randn(
            (n, horizon, 2), generator=generator, device=device, dtype=torch.float64
        ),
    )


@torch.no_grad()
def rollout_copula(
    kernel: CopulaKernel,
    policy: LoggingPolicy | RadiusResponsivePolicy,
    noise: CopulaNoise,
) -> CopulaTrajectory:
    """Roll out one policy through a fixed, policy-independent kernel."""

    hard = noise.initial_uniform.lt(kernel.config.initial_hard_probability).to(torch.float64)
    state_parts = [
        torch.stack((hard, torch.zeros_like(hard)), dim=1)
    ]
    action_parts: list[Tensor] = []
    probability_parts: list[Tensor] = []
    residual_parts: list[Tensor] = []
    for stage in range(noise.horizon):
        action_probability = policy.action_one_probability(hard)
        action = noise.action_uniforms[:, stage].lt(action_probability).to(torch.long)
        next_probability = kernel.hard_probability(hard, action)
        next_hard = noise.transition_uniforms[:, stage].lt(next_probability).to(torch.float64)
        residual = kernel.standardized_residuals(
            next_hard,
            noise.outcome_normals[:, stage],
        )
        time = torch.full_like(next_hard, (stage + 1) / noise.horizon)
        state_parts.append(torch.stack((next_hard, time), dim=1))
        action_parts.append(action)
        probability_parts.append(action_probability)
        residual_parts.append(residual)
        hard = next_hard
    return CopulaTrajectory(
        states=torch.stack(state_parts, dim=1),
        actions=torch.stack(action_parts, dim=1),
        residuals=torch.stack(residual_parts, dim=1),
        action_one_probabilities=torch.stack(probability_parts, dim=1),
    )


@torch.no_grad()
def prepare_source_reference(
    kernel: CopulaKernel,
    noise: CopulaNoise,
    *,
    alpha: float,
) -> CopulaSourceReference:
    """Roll out and audit the logging distribution once per beta cell."""

    source = rollout_copula(kernel, LoggingPolicy(kernel.config), noise)
    return CopulaSourceReference(
        kernel_fingerprint=kernel.fingerprint,
        trajectory=source,
        q90=empirical_left_quantile(source.scores, 1.0 - alpha, dim=0),
        audit=audit_equal_marginals(source, kernel),
    )


@torch.no_grad()
def evaluate_mechanism_setting(
    kernel: CopulaKernel,
    source: CopulaSourceReference,
    noise: CopulaNoise,
    *,
    radius: float,
    kappa: float,
    alpha: float,
) -> CopulaMechanismResult:
    """Evaluate q -> policy -> regime -> Q90 -> same-q coverage."""

    if source.kernel_fingerprint != kernel.fingerprint:
        raise ValueError("source and target must use the same beta-specific kernel")
    target_policy = RadiusResponsivePolicy(kernel.config, radius=radius, kappa=kappa)
    target = rollout_copula(kernel, target_policy, noise)
    source_scores = source.trajectory.scores
    target_scores = target.scores
    logging_policy = LoggingPolicy(kernel.config)
    logging_probability = logging_policy.action_one_probability(source.trajectory.current_hard)
    target_probability = target_policy.action_one_probability(source.trajectory.current_hard)
    overlap = prefix_overlap_diagnostics(
        source.trajectory.actions,
        logging_probability=logging_probability,
        target_probability=target_probability,
    )
    return CopulaMechanismResult(
        beta=kernel.beta,
        kappa=kappa,
        radius=radius,
        kernel_fingerprint=kernel.fingerprint,
        source_q90=source.q90,
        target_q90=empirical_left_quantile(target_scores, 1.0 - alpha, dim=0),
        source_coverage=source_scores.le(radius).to(torch.float64).mean(dim=0),
        target_coverage=target_scores.le(radius).to(torch.float64).mean(dim=0),
        policy_tv_on_source=(target_probability - logging_probability).abs().mean(dim=0),
        source_action_rate=source.trajectory.actions.to(torch.float64).mean(dim=0),
        target_action_rate=target.actions.to(torch.float64).mean(dim=0),
        source_hard_prevalence=source.trajectory.next_hard.mean(dim=0),
        target_hard_prevalence=target.next_hard.mean(dim=0),
        overlap=overlap,
        source_audit=source.audit,
        target_audit=audit_equal_marginals(target, kernel),
    )


@torch.no_grad()
def prefix_overlap_diagnostics(
    actions: Tensor,
    *,
    logging_probability: Tensor,
    target_probability: Tensor,
) -> PrefixOverlapDiagnostics:
    """Compute exact uncapped committed-prefix ratio diagnostics."""

    if actions.shape != logging_probability.shape or actions.shape != target_probability.shape:
        raise ValueError("actions and both probability tensors must align")
    source_likelihood = torch.where(
        actions.bool(), logging_probability, 1.0 - logging_probability
    )
    target_likelihood = torch.where(
        actions.bool(), target_probability, 1.0 - target_probability
    )
    incremental_ratio = target_likelihood / source_likelihood
    prefix_log_weight = incremental_ratio.log().cumsum(dim=1)
    stabilized = torch.exp(prefix_log_weight - prefix_log_weight.max(dim=0).values)
    normalized = stabilized / stabilized.sum(dim=0)
    ess_fraction = 1.0 / (len(actions) * normalized.square().sum(dim=0))
    return PrefixOverlapDiagnostics(
        ess_fraction=ess_fraction,
        maximum_normalized_weight_share=normalized.max(dim=0).values,
        log_weight_span=(
            prefix_log_weight.max(dim=0).values - prefix_log_weight.min(dim=0).values
        ),
        minimum_incremental_ratio=incremental_ratio.min(dim=0).values,
        maximum_incremental_ratio=incremental_ratio.max(dim=0).values,
    )


def empirical_left_quantile(values: Tensor, probability: float, *, dim: int) -> Tensor:
    """Return the empirical-left quantile at rank ``ceil(probability * n)``."""

    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must lie in (0, 1]")
    count = values.shape[dim]
    if count < 1:
        raise ValueError("quantile input must be nonempty")
    rank = max(1, math.ceil(probability * count))
    return torch.kthvalue(values, rank, dim=dim).values


@torch.no_grad()
def audit_equal_marginals(
    trajectory: CopulaTrajectory,
    kernel: CopulaKernel,
) -> MarginalAudit:
    """Audit N(0,1) marginals and the declared copula in regime/action cells."""

    residuals = trajectory.residuals.reshape(-1, 2)
    regimes = trajectory.next_hard.reshape(-1).to(torch.long)
    actions = trajectory.actions.reshape(-1)
    mean_errors: list[Tensor] = []
    variance_errors: list[Tensor] = []
    correlation_errors: list[Tensor] = []
    counts: list[int] = []
    for regime in (0, 1):
        expected_correlation = (
            kernel.config.easy_correlation if regime == 0 else kernel.config.hard_correlation
        )
        for action in (0, 1):
            cell = residuals[(regimes == regime) & (actions == action)]
            if len(cell) < 2:
                continue
            counts.append(len(cell))
            mean_errors.append(cell.mean(dim=0).abs().max())
            variance_errors.append((cell.square().mean(dim=0) - 1.0).abs().max())
            centered = cell - cell.mean(dim=0)
            covariance = (centered[:, 0] * centered[:, 1]).mean()
            denominator = torch.sqrt(
                centered[:, 0].square().mean() * centered[:, 1].square().mean()
            )
            empirical_correlation = covariance / denominator
            correlation_errors.append(
                (empirical_correlation - expected_correlation).abs()
            )
    if not counts:
        raise ValueError("no regime/action cell has enough rows for a marginal audit")
    return MarginalAudit(
        maximum_absolute_mean=float(torch.stack(mean_errors).max().item()),
        maximum_variance_error=float(torch.stack(variance_errors).max().item()),
        maximum_correlation_error=float(torch.stack(correlation_errors).max().item()),
        minimum_regime_action_count=min(counts),
    )


def _action_one_probability(
    config: CopulaDGPConfig,
    hard: Tensor,
    *,
    logit_shift: float | Tensor,
) -> Tensor:
    logits = config.behavior_logit_intercept + config.behavior_hard_effect * hard
    probability = torch.sigmoid(logits + torch.as_tensor(logit_shift).to(logits))
    return config.propensity_floor + (1.0 - 2.0 * config.propensity_floor) * probability


def _endpoint_normalized_response(
    config: CopulaDGPConfig,
    radius: float,
    *,
    like: Tensor,
) -> Tensor:
    normalized = (
        (like.new_tensor(radius) - config.response_radius_low)
        / (config.response_radius_high - config.response_radius_low)
    ).clamp(0.0, 1.0)
    half = config.response_sigmoid_slope / 2.0
    lower = torch.sigmoid(like.new_tensor(-half))
    upper = torch.sigmoid(like.new_tensor(half))
    return (
        torch.sigmoid(config.response_sigmoid_slope * (normalized - 0.5)) - lower
    ) / (upper - lower)


__all__ = [
    "CopulaKernel",
    "CopulaMechanismResult",
    "CopulaNoise",
    "CopulaSourceReference",
    "CopulaTrajectory",
    "LoggingPolicy",
    "MarginalAudit",
    "PrefixOverlapDiagnostics",
    "RadiusResponsivePolicy",
    "audit_equal_marginals",
    "evaluate_mechanism_setting",
    "empirical_left_quantile",
    "make_copula_noise",
    "prefix_overlap_diagnostics",
    "prepare_source_reference",
    "rollout_copula",
]
