"""Frozen configuration for the finite-MDP horizon--overlap diagnostic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


PROTOCOL = "finite_mdp_horizon_overlap_v1"
METHOD_NAMES = (
    "Standard CP",
    "History-only Prefix-IW",
    "Current-only IW",
    "SC-PCP",
)

# Whole namespaces are reserved, even when an earlier study used only a subset.
# This keeps later derived streams from silently colliding with existing work.
EXTERNAL_SEED_RESERVATIONS = {
    "exact_finite_mdp": range(52_000, 53_000),
    "controlled_six_method": range(91_000, 92_000),
    "orthogonal_copula": range(94_000, 95_000),
    "rq6_calibration_convergence": range(97_000, 98_000),
    "propensity_robustness": range(98_000, 99_000),
    "strict_split_audit": range(99_000, 100_000),
    "score_robustness": range(100_000, 101_000),
}


@dataclass(frozen=True)
class HorizonOverlapConfig:
    """Complete protocol for the paired M3 horizon--overlap phase diagram."""

    protocol: str = PROTOCOL
    state_count: int = 8
    action_count: int = 3
    horizons: tuple[int, ...] = (2, 4, 8, 12, 20)
    grid_size: int = 7
    alpha: float = 0.10
    calibration_trajectories: int = 3_000
    nominal_policy_tvs: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.15)
    instances: int = 200
    design_seed_start: int = 95_900
    design_seed_count: int = 100
    problem_seed_start: int = 96_000
    logging_seed_start: int = 96_200
    bootstrap_seed: int = 96_400
    bootstrap_resamples: int = 10_000
    radius_minimum: float = 1.4
    radius_maximum: float = 3.5
    mechanism_variant: str = "RQ5_only_overlap_controlled_M3"
    parent_policy_response_center: float = 2.5
    base_policy_center: str = "radius_minimum"
    policy_response_scale: float = 0.7
    policy_response_strength: float = 3.0
    minimum_base_reference_tv: float = 0.15
    output_dir: Path = Path("results/work/horizon_overlap_v1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HorizonOverlapConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError("horizon-overlap configuration root must be a mapping")
        values = dict(raw)
        for name in ("horizons", "nominal_policy_tvs"):
            if name in values:
                values[name] = tuple(values[name])
        if "output_dir" in values:
            values["output_dir"] = Path(values["output_dir"])
        config = cls(**values)
        config.validate()
        return config

    def with_output_dir(self, output_dir: Path) -> "HorizonOverlapConfig":
        config = replace(self, output_dir=output_dir)
        config.validate()
        return config

    @property
    def maximum_horizon(self) -> int:
        return max(self.horizons)

    @property
    def reference_grid_index(self) -> int:
        return self.grid_size // 2

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    @property
    def design_seeds(self) -> tuple[int, ...]:
        return tuple(
            range(self.design_seed_start, self.design_seed_start + self.design_seed_count)
        )

    @property
    def problem_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.problem_seed_start, self.problem_seed_start + self.instances))

    @property
    def logging_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.logging_seed_start, self.logging_seed_start + self.instances))

    @property
    def seed_namespace(self) -> str:
        return (
            f"horizon_overlap_v1:design_{self.design_seed_start}.."
            f"{self.design_seed_start + self.design_seed_count - 1}:problem_"
            f"{self.problem_seed_start}..{self.problem_seed_start + self.instances - 1}:"
            f"logging_{self.logging_seed_start}.."
            f"{self.logging_seed_start + self.instances - 1}:bootstrap_"
            f"{self.bootstrap_seed}"
        )

    def validate(self) -> None:
        if self.protocol != PROTOCOL:
            raise ValueError("unknown horizon-overlap protocol")
        if self.state_count < 4 or self.action_count != 3:
            raise ValueError("the paired M3 family requires at least four states and three actions")
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be nonempty, unique, and increasing")
        if min(self.horizons) < 1:
            raise ValueError("horizons must be positive")
        if self.grid_size != 7:
            raise ValueError("the frozen population radius grid has K=7")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.calibration_trajectories < 2:
            raise ValueError("calibration_trajectories must be at least two")
        if (
            not self.nominal_policy_tvs
            or tuple(sorted(set(self.nominal_policy_tvs))) != self.nominal_policy_tvs
            or self.nominal_policy_tvs[0] != 0.0
            or self.nominal_policy_tvs[-1] > 1.0
        ):
            raise ValueError(
                "nominal_policy_tvs must be unique, increasing, start at zero, and not exceed one"
            )
        if (
            self.instances < 1
            or self.design_seed_count < 1
            or self.bootstrap_resamples < 1
        ):
            raise ValueError(
                "instances, design_seed_count, and bootstrap_resamples must be positive"
            )
        if not self.radius_minimum < self.radius_maximum:
            raise ValueError("radius_minimum must be smaller than radius_maximum")
        if self.mechanism_variant != "RQ5_only_overlap_controlled_M3":
            raise ValueError("mechanism_variant must identify the RQ5-only M3 variant")
        if self.parent_policy_response_center != 2.5:
            raise ValueError("parent_policy_response_center must preserve the M3 provenance")
        if self.base_policy_center != "radius_minimum":
            raise ValueError("base_policy_center is frozen at radius_minimum")
        if self.policy_response_scale <= 0.0 or self.policy_response_strength < 0.0:
            raise ValueError("policy response scale must be positive and strength nonnegative")
        if self.minimum_base_reference_tv < max(self.nominal_policy_tvs):
            raise ValueError(
                "minimum_base_reference_tv must cover every nominal policy-TV level"
            )
        collision_audit = horizon_overlap_seed_collision_audit(self)
        if collision_audit["collision"]:
            details = (
                collision_audit["within_study_duplicates"]
                or collision_audit["external_collisions"]
            )
            raise ValueError(
                f"horizon-overlap RNG IDs collide: {details}"
            )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["horizons"] = list(self.horizons)
        values["nominal_policy_tvs"] = list(self.nominal_policy_tvs)
        values["output_dir"] = str(self.output_dir)
        values["seed_namespace"] = self.seed_namespace
        return values


def horizon_overlap_seed_collision_audit(
    config: HorizonOverlapConfig,
) -> dict[str, Any]:
    """Enumerate every RNG ID and compare it with coordinated namespaces."""

    streams = {
        "policy_design": list(config.design_seeds),
        "problem": list(config.problem_seeds),
        "logging": list(config.logging_seeds),
        "summary_bootstrap": [config.bootstrap_seed],
    }
    owners: dict[int, list[str]] = {}
    for stream, identifiers in streams.items():
        for identifier in identifiers:
            owners.setdefault(identifier, []).append(stream)
    duplicates = {
        str(identifier): stream_names
        for identifier, stream_names in sorted(owners.items())
        if len(stream_names) > 1
    }
    all_ids = sorted(owners)
    active = set(all_ids)
    external = {
        name: sorted(active.intersection(reserved))
        for name, reserved in EXTERNAL_SEED_RESERVATIONS.items()
    }
    external = {name: identifiers for name, identifiers in external.items() if identifiers}
    return {
        "seed_namespace": config.seed_namespace,
        "streams": streams,
        "all_rng_ids": all_ids,
        "rng_id_count": len(all_ids),
        "within_study_duplicates": duplicates,
        "external_reservations": {
            "exact_finite_mdp": "52000..52999",
            "controlled_six_method": "91000..91999",
            "orthogonal_copula": "94000..94999",
            "rq6_calibration_convergence": "97000..97999",
            "propensity_robustness": "98000..98999",
            "strict_split_audit": "99000..99999",
            "score_robustness": "100000..100999",
        },
        "external_collisions": external,
        "collision": bool(duplicates or external),
    }
