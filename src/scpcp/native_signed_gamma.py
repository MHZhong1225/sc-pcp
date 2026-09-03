"""Native signed-gamma synthetic mechanism for coverage-blind preflights.

This module is isolated from the production synthetic ``beta`` environment.
For a fixed gamma, source and target policies use the same kernel.  Prediction
radii enter only the target policy; the kernel has no radius field or argument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
import yaml

from scpcp.simulator import inverse_cdf_actions


PROTOCOL = "native_synthetic_signed_gamma_v1"
GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
PROVISIONAL_BASE_SEEDS = tuple(range(121_000, 121_200, 10))
SEED_NAMESPACE = "native_synthetic_signed_gamma_v1:121000..121190:step10"
RNG_AUDIT_POLICY = "live_actual_use_v1"


@dataclass(frozen=True)
class NativeSignedGammaDGPConfig:
    """Parameters of the observed-difficulty transition and tail kernel."""

    initial_difficulty_probability: float = 0.25
    logging_exploration: float = 0.12
    radius_low: float = 1.70
    radius_high: float = 2.10
    response_sigmoid_slope: float = 4.0
    policy_tilt: float = 0.4
    policy_ratio_cap: float = 3.0
    difficulty_intercept: float = -1.60
    difficulty_persistence: float = 2.00
    difficulty_state_effect: float = 0.25
    gamma_difficulty_coefficient: float = 0.35
    tail_intercept: float = -3.20
    tail_difficulty_effect: float = 2.20
    gamma_tail_coefficient: float = 0.45
    tail_scale: float = 3.0
    outcome_correlation: float = 0.30
    state_clip: float = 8.0

    def validate(self) -> None:
        if not 0.0 < self.initial_difficulty_probability < 1.0:
            raise ValueError("initial difficulty probability must lie in (0, 1)")
        if not 0.0 <= self.logging_exploration < 1.0:
            raise ValueError("logging exploration must lie in [0, 1)")
        if not self.radius_high > self.radius_low:
            raise ValueError("radius_high must exceed radius_low")
        if self.response_sigmoid_slope <= 0.0 or self.policy_tilt <= 0.0:
            raise ValueError("policy response constants must be positive")
        if self.policy_ratio_cap < 1.0:
            raise ValueError("policy_ratio_cap must be at least one")
        if self.gamma_difficulty_coefficient <= 0.0:
            raise ValueError("gamma difficulty coefficient must be positive")
        if self.gamma_tail_coefficient <= 0.0:
            raise ValueError("gamma tail coefficient must be positive")
        if self.tail_scale <= 1.0:
            raise ValueError("tail_scale must exceed one")
        if not -1.0 < self.outcome_correlation < 1.0:
            raise ValueError("outcome correlation must lie in (-1, 1)")
        if self.state_clip <= 0.0:
            raise ValueError("state_clip must be positive")


@dataclass(frozen=True)
class NativeSignedGammaGateConfig:
    """Coverage-blind checks required before any six-method experiment."""

    radius_mid: float = 1.90
    radius_high: float = 2.10
    minimum_mid_policy_tv: float = 0.03
    minimum_high_policy_tv: float = 0.05
    maximum_mid_action_coordinate_shift: float = -0.03
    maximum_high_action_coordinate_shift: float = -0.05
    minimum_late_difficulty_shift: float = 0.01
    minimum_late_tail_shift: float = 0.01
    minimum_prefix_ess_fraction: float = 0.15
    maximum_incremental_ratio: float = 3.0
    maximum_normalized_weight_share: float = 0.02
    minimum_available_seed_fraction: float = 0.95

    def validate(self, dgp: NativeSignedGammaDGPConfig) -> None:
        if not dgp.radius_low < self.radius_mid < self.radius_high:
            raise ValueError("gate radii must satisfy radius_low < radius_mid < radius_high")
        if self.radius_high != dgp.radius_high:
            raise ValueError("gate radius_high must equal the policy design endpoint")
        if not 0.0 <= self.minimum_mid_policy_tv <= self.minimum_high_policy_tv:
            raise ValueError("policy-TV thresholds must be ordered and nonnegative")
        if not self.maximum_high_action_coordinate_shift <= self.maximum_mid_action_coordinate_shift < 0.0:
            raise ValueError("action-coordinate thresholds must be ordered and negative")
        if self.minimum_late_difficulty_shift <= 0.0 or self.minimum_late_tail_shift <= 0.0:
            raise ValueError("direction thresholds must be positive")
        if not 0.0 < self.minimum_prefix_ess_fraction <= 1.0:
            raise ValueError("minimum prefix ESS fraction must lie in (0, 1]")
        if self.maximum_incremental_ratio != dgp.policy_ratio_cap:
            raise ValueError("overlap ratio threshold must equal the policy ratio cap")
        if not 0.0 < self.maximum_normalized_weight_share <= 1.0:
            raise ValueError("maximum normalized weight share must lie in (0, 1]")
        if self.minimum_available_seed_fraction != 0.95:
            raise ValueError("the frozen setting gate requires 19/20 seeds")


@dataclass(frozen=True)
class NativeSignedGammaBenchmarkConfig:
    """Frozen prelaunch contract for the new native signed-gamma benchmark."""

    dgp: NativeSignedGammaDGPConfig = NativeSignedGammaDGPConfig()
    gate: NativeSignedGammaGateConfig = NativeSignedGammaGateConfig()
    protocol: str = PROTOCOL
    horizon: int = 12
    late_stage_start: int = 4
    mechanism_trajectories: int = 50_000
    gammas: tuple[float, ...] = GAMMAS
    primary_gamma: float = -4.0
    base_seeds: tuple[int, ...] = PROVISIONAL_BASE_SEEDS
    seed_namespace: str = SEED_NAMESPACE
    rng_audit_policy: str = RNG_AUDIT_POLICY
    devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    output_root: Path = Path("results/work/native_synthetic_signed_gamma_v1")
    calibration_trajectories: int = 3_000
    grid_trajectories: int = 1_000
    reference_trajectories: int = 20_000
    online_trajectories: int = 2_000
    bootstrap_resamples: int = 10_000

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NativeSignedGammaBenchmarkConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError("native signed-gamma config root must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown native signed-gamma config fields: {unknown}")
        values = dict(raw)
        values["dgp"] = NativeSignedGammaDGPConfig(**values.get("dgp", {}))
        values["gate"] = NativeSignedGammaGateConfig(**values.get("gate", {}))
        for name in ("gammas", "base_seeds", "devices"):
            if name in values:
                values[name] = tuple(values[name])
        if "output_root" in values:
            values["output_root"] = Path(values["output_root"])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        self.dgp.validate()
        self.gate.validate(self.dgp)
        if self.protocol != PROTOCOL:
            raise ValueError(f"protocol must be {PROTOCOL}")
        if self.horizon != 12 or self.late_stage_start != 4:
            raise ValueError("native signed-gamma horizon contract must be T=12, late=4:12")
        if self.mechanism_trajectories != 50_000:
            raise ValueError("mechanism preflight requires exactly 50,000 trajectories")
        if self.gammas != GAMMAS or self.primary_gamma != -4.0:
            raise ValueError("signed grid must be (-4,-2,0,2,4) with gamma=-4 primary")
        if self.base_seeds != PROVISIONAL_BASE_SEEDS or self.seed_namespace != SEED_NAMESPACE:
            raise ValueError("base seeds differ from the reserved native-gamma namespace")
        if self.rng_audit_policy != RNG_AUDIT_POLICY:
            raise ValueError(f"rng_audit_policy must be {RNG_AUDIT_POLICY}")
        if not self.devices or any(not value.startswith("cuda:") for value in self.devices):
            raise ValueError("formal preflights require explicit CUDA devices")
        if (
            self.calibration_trajectories,
            self.grid_trajectories,
            self.reference_trajectories,
            self.online_trajectories,
            self.bootstrap_resamples,
        ) != (3_000, 1_000, 20_000, 2_000, 10_000):
            raise ValueError("science budgets differ from the clinical-v3 comparison contract")

    def with_overrides(
        self,
        *,
        base_seeds: tuple[int, ...] | None = None,
        devices: tuple[str, ...] | None = None,
        output_root: Path | None = None,
    ) -> "NativeSignedGammaBenchmarkConfig":
        config = replace(
            self,
            base_seeds=self.base_seeds if base_seeds is None else base_seeds,
            devices=self.devices if devices is None else devices,
            output_root=self.output_root if output_root is None else output_root,
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_root"] = str(self.output_root)
        for name in ("gammas", "base_seeds", "devices"):
            values[name] = list(values[name])
        return values


@dataclass(frozen=True)
class NativeSignedGammaNoise:
    """Exogenous noise shared across gamma cells and source/target policies."""

    initial_normals: Tensor
    initial_difficulty_uniforms: Tensor
    action_uniforms: Tensor
    difficulty_uniforms: Tensor
    tail_uniforms: Tensor
    transition_normals: Tensor
    outcome_normals: Tensor
    seed: int

    @property
    def n(self) -> int:
        return len(self.initial_difficulty_uniforms)

    @property
    def horizon(self) -> int:
        return self.action_uniforms.shape[1]


@dataclass(frozen=True)
class NativeSignedGammaTrajectory:
    states: Tensor
    actions: Tensor
    outcomes: Tensor
    tail_indicators: Tensor
    action_probabilities: Tensor
    kernel_fingerprint: str

    @property
    def n(self) -> int:
        return len(self.actions)

    @property
    def horizon(self) -> int:
        return self.actions.shape[1]

    @property
    def current_difficulty(self) -> Tensor:
        return self.states[:, :-1, 2]

    @property
    def next_difficulty(self) -> Tensor:
        return self.states[:, 1:, 2]


@dataclass(frozen=True)
class PrefixOverlap:
    minimum_ess_fraction: float
    maximum_incremental_ratio: float
    maximum_normalized_weight_share: float


def make_native_signed_gamma_noise(
    *,
    n: int,
    horizon: int,
    seed: int,
    device: str | torch.device,
) -> NativeSignedGammaNoise:
    if n < 1 or horizon < 1:
        raise ValueError("n and horizon must be positive")
    resolved = torch.device(device)
    generator = torch.Generator(device=resolved).manual_seed(seed)
    return NativeSignedGammaNoise(
        initial_normals=torch.randn((n, 2), generator=generator, device=resolved, dtype=torch.float64),
        initial_difficulty_uniforms=torch.rand(n, generator=generator, device=resolved, dtype=torch.float64),
        action_uniforms=torch.rand((n, horizon), generator=generator, device=resolved, dtype=torch.float64),
        difficulty_uniforms=torch.rand((n, horizon), generator=generator, device=resolved, dtype=torch.float64),
        tail_uniforms=torch.rand((n, horizon), generator=generator, device=resolved, dtype=torch.float64),
        transition_normals=torch.randn((n, horizon, 2), generator=generator, device=resolved, dtype=torch.float64),
        outcome_normals=torch.randn((n, horizon, 2), generator=generator, device=resolved, dtype=torch.float64),
        seed=seed,
    )


@dataclass(frozen=True)
class NativeSignedGammaLoggingPolicy:
    config: NativeSignedGammaDGPConfig
    n_actions: int = 3

    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        del q
        z1, z2, difficulty = states[:, 0], states[:, 1], states[:, 2]
        logits = torch.stack(
            (
                0.15 - 0.10 * z1 + 0.15 * z2 - 0.20 * difficulty,
                0.10 * z1 - 0.05 * z2,
                -0.10 + 0.25 * z1 - 0.10 * z2 + 0.35 * difficulty,
            ),
            dim=1,
        )
        base = torch.softmax(logits, dim=1)
        exploration = self.config.logging_exploration
        return (1.0 - exploration) * base + exploration / self.n_actions


@dataclass(frozen=True)
class NativeSignedGammaRadiusPolicy:
    """Radius-responsive policy that shifts mass toward the low-r action."""

    logging_policy: NativeSignedGammaLoggingPolicy

    @property
    def n_actions(self) -> int:
        return self.logging_policy.n_actions

    @property
    def config(self) -> NativeSignedGammaDGPConfig:
        return self.logging_policy.config

    def response_weight(self, radius: float | Tensor, *, like: Tensor) -> Tensor:
        value = torch.as_tensor(radius, dtype=like.dtype, device=like.device)
        normalized = ((value - self.config.radius_low) / (self.config.radius_high - self.config.radius_low)).clamp(0.0, 1.0)
        half = self.config.response_sigmoid_slope / 2.0
        lower = torch.sigmoid(value.new_tensor(-half))
        upper = torch.sigmoid(value.new_tensor(half))
        return (
            torch.sigmoid(self.config.response_sigmoid_slope * (normalized - 0.5))
            - lower
        ) / (upper - lower)

    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        if q is None:
            raise ValueError("native signed-gamma target policy requires a radius")
        reference = self.logging_policy.probabilities(states)
        response = self.response_weight(q, like=states)
        if response.ndim == 0:
            response = response.expand(len(states))
        if response.shape != (len(states),):
            raise ValueError("radius must be scalar or have one value per state")
        coordinate = action_coordinate(reference)
        weights = torch.exp(
            -self.config.policy_tilt * response[:, None] * coordinate[None, :]
        )
        return _ratio_capped_tilt(reference, weights, self.config.policy_ratio_cap)


@dataclass(frozen=True)
class NativeSignedGammaKernel:
    """Policy-independent ``K_gamma`` with no prediction-radius field."""

    config: NativeSignedGammaDGPConfig
    gamma: float
    state_dim: int = 4
    outcome_dim: int = 2
    n_actions: int = 3

    def interaction(self, state: Tensor, action: Tensor) -> Tensor:
        coordinate = action_coordinate(state)[action.to(torch.long)]
        observed_difficulty = state[:, 2]
        return self.gamma * coordinate * observed_difficulty

    def difficulty_probability(self, state: Tensor, action: Tensor) -> Tensor:
        interaction = self.interaction(state, action)
        logit = (
            self.config.difficulty_intercept
            + self.config.difficulty_persistence * state[:, 2]
            + self.config.difficulty_state_effect * torch.tanh(state[:, 0])
            + self.config.gamma_difficulty_coefficient * interaction
        )
        return torch.sigmoid(logit)

    def tail_probability(self, state: Tensor, action: Tensor) -> Tensor:
        interaction = self.interaction(state, action)
        logit = (
            self.config.tail_intercept
            + self.config.tail_difficulty_effect * state[:, 2]
            + self.config.gamma_tail_coefficient * interaction
        )
        return torch.sigmoid(logit)

    def initial_state(self, noise: NativeSignedGammaNoise) -> Tensor:
        difficulty = noise.initial_difficulty_uniforms.lt(
            self.config.initial_difficulty_probability
        ).to(noise.initial_normals)
        time = torch.zeros_like(difficulty)
        return torch.cat((noise.initial_normals, difficulty[:, None], time[:, None]), dim=1)

    def step_from_noise(
        self,
        state: Tensor,
        action: Tensor,
        *,
        difficulty_uniform: Tensor,
        tail_uniform: Tensor,
        transition_normals: Tensor,
        outcome_normals: Tensor,
        time: int,
        horizon: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        next_difficulty = difficulty_uniform.lt(
            self.difficulty_probability(state, action)
        ).to(state)
        tail = tail_uniform.lt(self.tail_probability(state, action)).to(state)
        z1, z2 = state[:, 0], state[:, 1]
        next_z1 = (
            0.78 * z1
            + 0.12 * z2
            + 0.30 * next_difficulty
            + 0.25 * transition_normals[:, 0]
        )
        next_z2 = (
            0.08 * z1
            + 0.72 * z2
            + 0.20 * next_difficulty
            + 0.20
            * (0.30 * transition_normals[:, 0] + 0.9539392014 * transition_normals[:, 1])
        )
        continuous = torch.stack((next_z1, next_z2), dim=1).clamp(
            -self.config.state_clip,
            self.config.state_clip,
        )
        correlation = self.config.outcome_correlation
        residual = torch.stack(
            (
                outcome_normals[:, 0],
                correlation * outcome_normals[:, 0]
                + (1.0 - correlation**2) ** 0.5 * outcome_normals[:, 1],
            ),
            dim=1,
        )
        base_scale = state.new_tensor((0.35, 0.25))
        multiplier = 1.0 + (self.config.tail_scale - 1.0) * tail
        mean = continuous + torch.stack(
            (0.25 * next_difficulty, 0.20 * next_difficulty),
            dim=1,
        )
        outcome = mean + multiplier[:, None] * base_scale[None, :] * residual
        next_time = torch.full_like(next_difficulty, (time + 1) / horizon)
        next_state = torch.cat(
            (continuous, next_difficulty[:, None], next_time[:, None]),
            dim=1,
        )
        return next_state, outcome, tail

    @property
    def fingerprint(self) -> str:
        payload = {"protocol": PROTOCOL, "gamma": self.gamma, "dgp": asdict(self.config)}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def rollout_native_signed_gamma(
    kernel: NativeSignedGammaKernel,
    policy: NativeSignedGammaLoggingPolicy | NativeSignedGammaRadiusPolicy,
    noise: NativeSignedGammaNoise,
    *,
    radius: float | Tensor | None = None,
) -> NativeSignedGammaTrajectory:
    if isinstance(radius, Tensor) and radius.ndim == 1 and radius.shape != (noise.horizon,):
        raise ValueError("stagewise radius must have shape [T]")
    state = kernel.initial_state(noise)
    states = [state]
    actions = []
    outcomes = []
    tails = []
    probabilities = []
    for time in range(noise.horizon):
        stage_radius = radius[time] if isinstance(radius, Tensor) and radius.ndim == 1 else radius
        action_probability = policy.probabilities(state, stage_radius)
        action = inverse_cdf_actions(action_probability, noise.action_uniforms[:, time])
        state, outcome, tail = kernel.step_from_noise(
            state,
            action,
            difficulty_uniform=noise.difficulty_uniforms[:, time],
            tail_uniform=noise.tail_uniforms[:, time],
            transition_normals=noise.transition_normals[:, time],
            outcome_normals=noise.outcome_normals[:, time],
            time=time,
            horizon=noise.horizon,
        )
        states.append(state)
        actions.append(action)
        outcomes.append(outcome)
        tails.append(tail)
        probabilities.append(action_probability)
    return NativeSignedGammaTrajectory(
        states=torch.stack(states, dim=1),
        actions=torch.stack(actions, dim=1),
        outcomes=torch.stack(outcomes, dim=1),
        tail_indicators=torch.stack(tails, dim=1),
        action_probabilities=torch.stack(probabilities, dim=1),
        kernel_fingerprint=kernel.fingerprint,
    )


def prefix_overlap(
    source: NativeSignedGammaTrajectory,
    *,
    logging_policy: NativeSignedGammaLoggingPolicy,
    target_policy: NativeSignedGammaRadiusPolicy,
    radius: float,
) -> PrefixOverlap:
    current = source.states[:, :-1].reshape(-1, source.states.shape[-1])
    logging = logging_policy.probabilities(current).reshape(source.n, source.horizon, -1)
    target = target_policy.probabilities(current, radius).reshape(source.n, source.horizon, -1)
    actions = source.actions.to(torch.long)
    chosen_logging = logging.gather(2, actions[:, :, None]).squeeze(2)
    chosen_target = target.gather(2, actions[:, :, None]).squeeze(2)
    incremental = chosen_target / chosen_logging
    log_prefix = incremental.to(torch.float64).log().cumsum(dim=1)
    ess_fractions = []
    maximum_shares = []
    for stage in range(source.horizon):
        log_weight = log_prefix[:, stage]
        weight = torch.exp(log_weight - log_weight.max())
        normalized = weight / weight.sum()
        ess_fractions.append(float((1.0 / normalized.square().sum() / source.n).item()))
        maximum_shares.append(float(normalized.max().item()))
    return PrefixOverlap(
        minimum_ess_fraction=min(ess_fractions),
        maximum_incremental_ratio=float(incremental.max().item()),
        maximum_normalized_weight_share=max(maximum_shares),
    )


def mechanism_probe(
    config: NativeSignedGammaBenchmarkConfig,
    *,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Run one coverage-blind seed across the complete signed-gamma grid."""

    noise = make_native_signed_gamma_noise(
        n=config.mechanism_trajectories,
        horizon=config.horizon,
        seed=seed,
        device=device,
    )
    logging = NativeSignedGammaLoggingPolicy(config.dgp)
    target = NativeSignedGammaRadiusPolicy(logging)
    gamma_rows = []
    for gamma in config.gammas:
        kernel = NativeSignedGammaKernel(config.dgp, gamma)
        source = rollout_native_signed_gamma(kernel, logging, noise)
        target_mid = rollout_native_signed_gamma(
            kernel,
            target,
            noise,
            radius=config.gate.radius_mid,
        )
        target_high = rollout_native_signed_gamma(
            kernel,
            target,
            noise,
            radius=config.gate.radius_high,
        )
        current = source.states[:, :-1].reshape(-1, kernel.state_dim)
        logging_probability = logging.probabilities(current)
        mid_probability = target.probabilities(current, config.gate.radius_mid)
        high_probability = target.probabilities(current, config.gate.radius_high)
        coordinate = action_coordinate(current)
        mid_shift = (
            (mid_probability - logging_probability) * coordinate[None, :]
        ).sum(dim=1).mean()
        high_shift = (
            (high_probability - logging_probability) * coordinate[None, :]
        ).sum(dim=1).mean()
        late = slice(config.late_stage_start, config.horizon)
        row = {
            "gamma": gamma,
            "kernel_fingerprint": kernel.fingerprint,
            "source_target_kernel_shared": bool(
                source.kernel_fingerprint
                == target_mid.kernel_fingerprint
                == target_high.kernel_fingerprint
            ),
            "mid_policy_tv": float(
                (mid_probability - logging_probability).abs().sum(dim=1).mul(0.5).mean().item()
            ),
            "high_policy_tv": float(
                (high_probability - logging_probability).abs().sum(dim=1).mul(0.5).mean().item()
            ),
            "mid_expected_action_coordinate_shift": float(mid_shift.item()),
            "high_expected_action_coordinate_shift": float(high_shift.item()),
            "late_difficulty_prevalence_shift": float(
                (
                    target_high.next_difficulty[:, late].mean()
                    - source.next_difficulty[:, late].mean()
                ).item()
            ),
            "late_tail_prevalence_shift": float(
                (
                    target_high.tail_indicators[:, late].mean()
                    - source.tail_indicators[:, late].mean()
                ).item()
            ),
            "finite_and_structural": _trajectory_invariants(source)
            and _trajectory_invariants(target_mid)
            and _trajectory_invariants(target_high),
        }
        if gamma == 0.0:
            row["exact_placebo"] = {
                "states": torch.equal(source.states, target_high.states),
                "outcomes": torch.equal(source.outcomes, target_high.outcomes),
                "tails": torch.equal(source.tail_indicators, target_high.tail_indicators),
            }
        if gamma == config.primary_gamma:
            row["overlap"] = {
                "mid": asdict(
                    prefix_overlap(
                        source,
                        logging_policy=logging,
                        target_policy=target,
                        radius=config.gate.radius_mid,
                    )
                ),
                "high": asdict(
                    prefix_overlap(
                        source,
                        logging_policy=logging,
                        target_policy=target,
                        radius=config.gate.radius_high,
                    )
                ),
            }
        gamma_rows.append(row)
    return {
        "protocol": config.protocol,
        "seed": seed,
        "preflight_only": True,
        "primary_gamma": config.primary_gamma,
        "gamma_rows": gamma_rows,
    }


def seed_passes_mechanism_gate(
    probe: dict[str, Any],
    gate: NativeSignedGammaGateConfig,
) -> bool:
    rows = {float(row["gamma"]): row for row in probe["gamma_rows"]}
    placebo = rows[0.0].get("exact_placebo", {})
    primary_overlap = rows[-4.0].get("overlap", {})
    overlaps = [primary_overlap.get("mid", {}), primary_overlap.get("high", {})]
    response_rows = (rows[-4.0], rows[0.0], rows[4.0])
    ordered_rows = [rows[gamma] for gamma in GAMMAS]
    return bool(
        probe.get("preflight_only") is True
        and all(row["finite_and_structural"] for row in rows.values())
        and all(row["source_target_kernel_shared"] for row in rows.values())
        and placebo == {"states": True, "outcomes": True, "tails": True}
        and all(row["mid_policy_tv"] >= gate.minimum_mid_policy_tv for row in response_rows)
        and all(row["high_policy_tv"] >= gate.minimum_high_policy_tv for row in response_rows)
        and all(
            row["mid_expected_action_coordinate_shift"]
            <= gate.maximum_mid_action_coordinate_shift
            for row in response_rows
        )
        and all(
            row["high_expected_action_coordinate_shift"]
            <= gate.maximum_high_action_coordinate_shift
            for row in response_rows
        )
        and rows[-4.0]["late_difficulty_prevalence_shift"]
        >= gate.minimum_late_difficulty_shift
        and rows[-4.0]["late_tail_prevalence_shift"] >= gate.minimum_late_tail_shift
        and rows[4.0]["late_difficulty_prevalence_shift"]
        <= -gate.minimum_late_difficulty_shift
        and rows[4.0]["late_tail_prevalence_shift"] <= -gate.minimum_late_tail_shift
        and all(
            left["late_difficulty_prevalence_shift"]
            >= right["late_difficulty_prevalence_shift"]
            for left, right in zip(ordered_rows, ordered_rows[1:])
        )
        and all(
            left["late_tail_prevalence_shift"]
            >= right["late_tail_prevalence_shift"]
            for left, right in zip(ordered_rows, ordered_rows[1:])
        )
        and len(overlaps) == 2
        and all(
            overlap.get("minimum_ess_fraction", -1.0)
            >= gate.minimum_prefix_ess_fraction
            and overlap.get("maximum_incremental_ratio", float("inf"))
            <= gate.maximum_incremental_ratio + 1e-12
            and overlap.get("maximum_normalized_weight_share", float("inf"))
            <= gate.maximum_normalized_weight_share
            for overlap in overlaps
        )
    )


def action_coordinate(like: Tensor) -> Tensor:
    return like.new_tensor((-1.0, 0.0, 1.0))


def _ratio_capped_tilt(reference: Tensor, weights: Tensor, cap: float) -> Tensor:
    lower = torch.zeros_like(weights[:, :1])
    upper = weights.amax(dim=1, keepdim=True).clamp_min(1e-12)
    for _ in range(64):
        normalizer = (lower + upper) / 2.0
        ratio = torch.minimum(weights / normalizer, torch.full_like(weights, cap))
        mass = (reference * ratio).sum(dim=1, keepdim=True)
        lower = torch.where(mass > 1.0, normalizer, lower)
        upper = torch.where(mass > 1.0, upper, normalizer)
    ratio = torch.minimum(weights / ((lower + upper) / 2.0), torch.full_like(weights, cap))
    return reference * ratio


def _trajectory_invariants(trajectory: NativeSignedGammaTrajectory) -> bool:
    finite = all(
        torch.isfinite(value).all().item()
        for value in (
            trajectory.states,
            trajectory.outcomes,
            trajectory.action_probabilities,
        )
    )
    simplex = torch.allclose(
        trajectory.action_probabilities.sum(dim=2),
        torch.ones_like(trajectory.action_probabilities[:, :, 0]),
        atol=1e-10,
        rtol=0.0,
    )
    probability_bounds = bool(
        trajectory.action_probabilities.ge(0.0).all().item()
        and trajectory.action_probabilities.le(1.0).all().item()
    )
    action_bounds = bool(
        trajectory.actions.ge(0).all().item()
        and trajectory.actions.lt(trajectory.action_probabilities.shape[2]).all().item()
    )
    binary = all(
        bool(value.eq(0.0).logical_or(value.eq(1.0)).all().item())
        for value in (trajectory.current_difficulty, trajectory.next_difficulty, trajectory.tail_indicators)
    )
    expected_time = trajectory.states.new_tensor(
        [stage / trajectory.horizon for stage in range(trajectory.horizon + 1)]
    )
    time_coordinates_valid = torch.equal(
        trajectory.states[:, :, 3],
        expected_time[None, :].expand(trajectory.n, -1),
    )
    return bool(
        finite
        and simplex
        and probability_bounds
        and action_bounds
        and binary
        and time_coordinates_valid
    )


__all__ = [
    "GAMMAS",
    "PROTOCOL",
    "PROVISIONAL_BASE_SEEDS",
    "RNG_AUDIT_POLICY",
    "SEED_NAMESPACE",
    "NativeSignedGammaBenchmarkConfig",
    "NativeSignedGammaDGPConfig",
    "NativeSignedGammaGateConfig",
    "NativeSignedGammaKernel",
    "NativeSignedGammaLoggingPolicy",
    "NativeSignedGammaNoise",
    "NativeSignedGammaRadiusPolicy",
    "NativeSignedGammaTrajectory",
    "PrefixOverlap",
    "action_coordinate",
    "make_native_signed_gamma_noise",
    "mechanism_probe",
    "prefix_overlap",
    "rollout_native_signed_gamma",
    "seed_passes_mechanism_gate",
]
