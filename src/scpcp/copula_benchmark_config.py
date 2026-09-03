"""Frozen configuration for the isolated equal-marginal copula benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


COPULA_SCIENTIFIC_SEEDS = tuple(range(94_000, 94_200, 2))
COPULA_SEED_NAMESPACE = "orthogonal_copula_v1:94000..94198:even"


@dataclass(frozen=True)
class CopulaDGPConfig:
    """Parameters of a same-kernel, two-outcome sequential DGP."""

    initial_hard_probability: float = 0.25
    behavior_logit_intercept: float = -0.50
    behavior_hard_effect: float = 0.50
    propensity_floor: float = 0.02
    maximum_policy_logit_shift: float = 1.50
    response_radius_low: float = 1.70
    response_radius_high: float = 2.10
    response_sigmoid_slope: float = 4.0
    hard_transition_intercept: float = -1.50
    hard_persistence: float = 2.20
    easy_correlation: float = 0.90
    hard_correlation: float = 0.00

    def validate(self) -> None:
        probabilities = (self.initial_hard_probability, self.propensity_floor)
        if not 0.0 < probabilities[0] < 1.0:
            raise ValueError("initial_hard_probability must lie in (0, 1)")
        if not 0.0 <= probabilities[1] < 0.5:
            raise ValueError("propensity_floor must lie in [0, 0.5)")
        if self.maximum_policy_logit_shift < 0.0:
            raise ValueError("maximum_policy_logit_shift must be nonnegative")
        if not self.response_radius_high > self.response_radius_low:
            raise ValueError("response_radius_high must exceed response_radius_low")
        if self.response_sigmoid_slope <= 0.0:
            raise ValueError("response_sigmoid_slope must be positive")
        for name, correlation in (
            ("easy_correlation", self.easy_correlation),
            ("hard_correlation", self.hard_correlation),
        ):
            if not -1.0 < correlation < 1.0:
                raise ValueError(f"{name} must lie in (-1, 1)")
        if not self.easy_correlation > self.hard_correlation:
            raise ValueError(
                "easy_correlation must exceed hard_correlation so hard regimes "
                "raise the normalized-max quantile"
            )


@dataclass(frozen=True)
class CopulaGateConfig:
    """Predeclared mechanism checks required before any six-method study."""

    primary_radius: float = 1.90
    maximum_placebo_policy_tv: float = 1e-12
    maximum_placebo_hard_prevalence_gap: float = 0.003
    maximum_placebo_relative_q90_gap: float = 0.01
    maximum_placebo_coverage_gap: float = 0.003
    minimum_policy_tv: float = 0.05
    minimum_hard_prevalence_shift: float = 0.01
    minimum_relative_q90_shift: float = 0.03
    minimum_coverage_shift: float = 0.015
    minimum_prefix_ess_fraction: float = 0.15
    maximum_incremental_ratio: float = 10.0
    maximum_normalized_weight_share: float = 0.02
    maximum_marginal_mean_error: float = 0.02
    maximum_marginal_variance_error: float = 0.03
    maximum_correlation_error: float = 0.03
    paired_confidence_level: float = 0.95

    def validate(self) -> None:
        for name in (
            "maximum_placebo_policy_tv",
            "maximum_placebo_hard_prevalence_gap",
            "maximum_placebo_relative_q90_gap",
            "maximum_placebo_coverage_gap",
            "minimum_policy_tv",
            "minimum_hard_prevalence_shift",
            "minimum_relative_q90_shift",
            "minimum_coverage_shift",
            "minimum_prefix_ess_fraction",
            "maximum_incremental_ratio",
            "maximum_normalized_weight_share",
            "maximum_marginal_mean_error",
            "maximum_marginal_variance_error",
            "maximum_correlation_error",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.minimum_prefix_ess_fraction > 1.0:
            raise ValueError("minimum_prefix_ess_fraction cannot exceed one")
        if self.maximum_incremental_ratio < 1.0:
            raise ValueError("maximum_incremental_ratio must be at least one")
        if self.maximum_normalized_weight_share > 1.0:
            raise ValueError("maximum_normalized_weight_share cannot exceed one")
        if not 0.0 < self.paired_confidence_level < 1.0:
            raise ValueError("paired_confidence_level must lie in (0, 1)")


@dataclass(frozen=True)
class CopulaBenchmarkConfig:
    """Complete mechanism-study configuration.

    The seed bank is independent of the controlled-clinical confirmation bases
    (91_000, 91_010, ..., 91_190 and their +1/+2 streams) and the finite-MDP
    bank (52_000--52_999).
    """

    dgp: CopulaDGPConfig = CopulaDGPConfig()
    gate: CopulaGateConfig = CopulaGateConfig()
    protocol: str = "equal_marginal_copula_mechanism_v1"
    horizon: int = 12
    late_stage_start: int = 4
    trajectories: int = 50_000
    alpha: float = 0.10
    betas: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    kappas: tuple[float, ...] = (0.0, 0.5, 1.0)
    radii: tuple[float, ...] = (1.70, 1.90, 2.10)
    seeds: tuple[int, ...] = COPULA_SCIENTIFIC_SEEDS
    devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    output_dir: Path = Path("results/work/copula_mechanism_v1")
    seed_namespace: str = COPULA_SEED_NAMESPACE

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CopulaBenchmarkConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError("copula configuration root must be a mapping")
        values = dict(raw)
        values["dgp"] = CopulaDGPConfig(**values.get("dgp", {}))
        values["gate"] = CopulaGateConfig(**values.get("gate", {}))
        for name in ("betas", "kappas", "radii", "seeds", "devices"):
            if name in values:
                values[name] = tuple(values[name])
        if "output_dir" in values:
            values["output_dir"] = Path(values["output_dir"])
        config = cls(**values)
        config.validate()
        return config

    def with_overrides(
        self,
        *,
        seeds: tuple[int, ...] | None = None,
        devices: tuple[str, ...] | None = None,
        output_dir: Path | None = None,
    ) -> "CopulaBenchmarkConfig":
        config = replace(
            self,
            seeds=self.seeds if seeds is None else seeds,
            devices=self.devices if devices is None else devices,
            output_dir=self.output_dir if output_dir is None else output_dir,
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.dgp.validate()
        self.gate.validate()
        if self.protocol != "equal_marginal_copula_mechanism_v1":
            raise ValueError("unknown copula benchmark protocol")
        if self.horizon < 1 or self.trajectories < 2:
            raise ValueError("horizon and trajectories must be positive")
        if not 0 <= self.late_stage_start < self.horizon:
            raise ValueError("late_stage_start must index an observed stage")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not self.betas or 0.0 not in self.betas:
            raise ValueError("betas must contain the beta=0 placebo")
        if min(self.betas) >= 0.0 or max(self.betas) <= 0.0:
            raise ValueError("betas must contain both signed directions")
        if not self.kappas or 0.0 not in self.kappas:
            raise ValueError("kappas must contain the kappa=0 placebo")
        if min(self.kappas) < 0.0 or max(self.kappas) <= 0.0:
            raise ValueError("kappas must be nonnegative and contain a response")
        if not self.radii or self.gate.primary_radius not in self.radii:
            raise ValueError("radii must contain gate.primary_radius exactly")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not self.seeds or not set(self.seeds).issubset(COPULA_SCIENTIFIC_SEEDS):
            raise ValueError(
                "seeds must be a nonempty subset of the frozen orthogonal-copula bank"
            )
        if self.seed_namespace != COPULA_SEED_NAMESPACE:
            raise ValueError("seed_namespace differs from the frozen copula namespace")
        if not self.devices or any(not value.startswith("cuda:") for value in self.devices):
            raise ValueError("scientific copula runs require explicit CUDA devices")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        for name in ("betas", "kappas", "radii", "seeds", "devices"):
            values[name] = list(values[name])
        return values
