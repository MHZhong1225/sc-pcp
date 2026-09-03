"""Run the parent-gated Native signed-gamma six-method science benchmark.

The fixed entry point has no scientific command-line knobs.  It first verifies
the completed time-coordinate repair replay and its downstream RNG reservation,
then performs a live collision audit.  A failed or missing parent gate stops
before the science output root is created.  ``--validate-only`` performs those
checks without consuming a science RNG stream or writing an artifact.

Formal launch (only after an independent prelaunch audit)::

    conda run -n ucp python scripts/run_native_synthetic_signed_gamma_science.py

The primary cell is gamma=-4.  The complete signed-gamma curve is retained as
descriptive mechanism evidence; only the primary cell carries paired method
comparisons.  The primary coverage metric is exactly
``min_t mean_seed(C_seed,t)``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
import hashlib
import importlib.util
import io
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import sys
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy
from scipy import stats
import torch
from torch import Tensor, nn
import yaml


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_native_synthetic_signed_gamma_preflight as preflight  # noqa: E402
from scpcp.baselines import (  # noqa: E402
    OnlineBaselineResult,
    aci_style_controller,
    finite_depth_mfcs_selection,
    multidim_spci_style_controller,
    prc_profile_scale,
    standard_cp_stagewise_radii,
)
from scpcp.coverage import (  # noqa: E402
    fixed_q_grid,
    profiled_scale_grid,
    stage_score_profile,
)
from scpcp.data import TrajectoryBatch  # noqa: E402
from scpcp.experiment import _paper_seed  # noqa: E402
from scpcp.marginal_prefix import select_marginal_prefix_schedule  # noqa: E402
from scpcp.native_signed_gamma import (  # noqa: E402
    GAMMAS as NATIVE_GAMMAS,
    PROTOCOL as NATIVE_PROTOCOL,
    NativeSignedGammaDGPConfig,
    NativeSignedGammaKernel,
    NativeSignedGammaLoggingPolicy,
    NativeSignedGammaRadiusPolicy,
    NativeSignedGammaTrajectory,
    action_coordinate,
    make_native_signed_gamma_noise,
    rollout_native_signed_gamma,
)
from scpcp.scores import score_batch  # noqa: E402


PROTOCOL = "native_synthetic_signed_gamma_six_method_science_v1"
ROLE = "parent_gated_fresh_six_method_science"
DEFAULT_CONFIG = ROOT / "configs/native_synthetic_signed_gamma_science.yaml"
METHODS = ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")
GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
PRIMARY_GAMMA = -4.0
FORMAL_BASE_SEEDS = tuple(range(121_400, 121_600, 10))
FORMAL_BOOTSTRAP_SEED = 12_140_019
FORMAL_MAPPING_COUNT = 241
FORMAL_MAPPING_SHA256 = (
    "c17c3098ee47b5f530300505198983393b95a51b7ffbdb4c3d399c7b43a5546a"
)
OPTIONAL_PREFLIGHT_RESERVE = tuple(range(121_200, 121_400, 10))
REPAIR_REPLAY_IDS = tuple(range(121_000, 121_200, 10))
INFORMATION_REGIME = {
    "Standard CP": "offline_logged_data",
    "ACI": "on_policy_adaptation",
    "MFCS": "offline_logged_data",
    "SPCI": "on_policy_adaptation",
    "PRC": "on_policy_adaptation",
    "SC-PCP": "offline_logged_data",
}
TARGET_ADAPTATION_BUDGET = {
    "Standard CP": 0,
    "ACI": 2_000,
    "MFCS": 0,
    "SPCI": 2_000,
    "PRC": 2_000,
    "SC-PCP": 0,
}
ADAPTIVE_METHODS = ("ACI", "SPCI", "PRC")
SCIENCE_CONTRACT = {
    "methods": list(METHODS),
    "gammas": list(GAMMAS),
    "primary_gamma": PRIMARY_GAMMA,
    "coverage_metric": "min_t mean_seed(target_coverage_seed_t)",
    "mean_coverage_metric": "mean_t mean_seed(target_coverage_seed_t)",
    "coverage_conditioning": "successful_selection",
    "selection_rate_denominator": "all_20_prespecified_fresh_science_seeds",
    "normalized_width": (
        "mean over outcome coordinates of full box width divided by the frozen "
        "one-unit Native outcome scales [1,1]"
    ),
    "signed_curve_interpretation": {
        "gamma_-4": "primary_confirmatory_method_comparison",
        "other_gammas": "descriptive_signed_mechanism_curve_no_ranking_or_superiority",
    },
    "common_random_numbers": (
        "calibration and reference noise are shared across signed gammas; the "
        "target reference noise is shared across methods and signed gammas; "
        "online adaptation uses independent method streams reused across gammas"
    ),
    "uncertainty_intervals": {
        "pointwise_stage_coverage": "Student-t across selected complete seeds",
        "pointwise_stage_width": "Student-t across selected complete seeds",
        "mean_coverage": "Student-t across selected per-seed stage means",
        "mean_width": "Student-t across selected per-seed stage means",
        "wsc": "complete-seed-vector percentile bootstrap",
        "paired_primary_contrasts": "paired complete-seed percentile bootstrap",
        "selection_rate": "Wilson interval over all 20 prespecified seeds",
    },
    "bootstrap": (
        "one 10000x20 complete-seed uniform matrix used only for WSC and primary "
        "paired contrasts; selected subsets use leading columns projected to "
        "their exact selected-set size"
    ),
    "low_overlap_consequence": "no science root; no ranking, attainment, or superiority",
}


@dataclass(frozen=True)
class ParentConfig:
    scientific_protocol: str
    administrative_role: str
    repair_root: Path
    repair_runner: Path
    repair_config: Path
    required_decision: str


@dataclass(frozen=True)
class DesignConfig:
    horizon: int
    gammas: tuple[float, ...]
    primary_gamma: float
    methods: tuple[str, ...]
    alpha: float
    delta: float
    q_grid_size: int
    q_quantile_min: float
    q_quantile_max: float
    mfcs_depth: int
    aci_gamma: float
    online_rounds: int
    multidim_buffer: int
    prc_maximum_step: float
    weight_cap: float
    policy_ratio_cap: float
    outcome_normalization: tuple[float, ...]


@dataclass(frozen=True)
class BudgetConfig:
    calibration_trajectories: int
    grid_trajectories: int
    reference_trajectories: int
    online_trajectories_per_adaptive_method: int
    bootstrap_resamples: int


@dataclass(frozen=True)
class RngConfig:
    base_seeds: tuple[int, ...]
    bootstrap_seed: int
    seed_namespace: str
    mapping_count: int
    mapping_sha256: str
    untouched_optional_preflight_reserve: tuple[int, ...]


@dataclass(frozen=True)
class ScienceConfig:
    protocol: str
    role: str
    parent: ParentConfig
    design: DesignConfig
    budgets: BudgetConfig
    rng: RngConfig
    devices: tuple[str, ...]
    output_root: Path

    @classmethod
    def from_yaml(cls, path: Path) -> "ScienceConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("science config root must be a mapping")
        _require_exact_keys(
            raw,
            {"protocol", "role", "parent", "design", "budgets", "rng", "devices", "output_root"},
            "science config",
        )
        parent_raw = dict(raw["parent"])
        for key in ("repair_root", "repair_runner", "repair_config"):
            parent_raw[key] = Path(parent_raw[key])
        parent = _dataclass_from_mapping(ParentConfig, parent_raw, "parent config")
        design_raw = dict(raw["design"])
        for key in ("gammas", "methods", "outcome_normalization"):
            design_raw[key] = tuple(design_raw[key])
        design = _dataclass_from_mapping(DesignConfig, design_raw, "design config")
        budgets = _dataclass_from_mapping(BudgetConfig, raw["budgets"], "budget config")
        rng_raw = dict(raw["rng"])
        for key in ("base_seeds", "untouched_optional_preflight_reserve"):
            rng_raw[key] = tuple(rng_raw[key])
        rng = _dataclass_from_mapping(RngConfig, rng_raw, "RNG config")
        config = cls(
            protocol=str(raw["protocol"]),
            role=str(raw["role"]),
            parent=parent,
            design=design,
            budgets=budgets,
            rng=rng,
            devices=tuple(raw["devices"]),
            output_root=Path(raw["output_root"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.protocol != PROTOCOL or self.role != ROLE:
            raise ValueError("science protocol or role differs from the frozen contract")
        if self.parent.scientific_protocol != NATIVE_PROTOCOL:
            raise ValueError("parent scientific protocol differs")
        if self.parent.administrative_role != "administrative_validator_repair_replay":
            raise ValueError("parent repair role differs")
        if self.parent.required_decision != "GO":
            raise ValueError("science requires an exact parent GO decision")
        expected_parent_paths = {
            "repair_root": Path("results/work/native_synthetic_signed_gamma_v1_time_coordinate_repair_r1"),
            "repair_runner": Path("scripts/run_native_synthetic_signed_gamma_time_coordinate_repair_r1.py"),
            "repair_config": Path("configs/native_synthetic_signed_gamma_time_coordinate_repair_r1.yaml"),
        }
        for name, expected in expected_parent_paths.items():
            if getattr(self.parent, name) != expected:
                raise ValueError(f"parent {name} differs from the frozen path")
        design = self.design
        if (
            design.horizon != 12
            or design.gammas != GAMMAS
            or design.primary_gamma != PRIMARY_GAMMA
            or design.methods != METHODS
        ):
            raise ValueError("science horizon, gamma grid, primary cell, or methods differ")
        if (
            design.alpha,
            design.delta,
            design.q_grid_size,
            design.q_quantile_min,
            design.q_quantile_max,
            design.mfcs_depth,
            design.aci_gamma,
            design.online_rounds,
            design.multidim_buffer,
            design.prc_maximum_step,
            design.weight_cap,
            design.policy_ratio_cap,
            design.outcome_normalization,
        ) != (0.10, 0.05, 101, 0.50, 0.999, 3, 0.01, 3, 1_000, 0.35, 40.0, 3.0, (1.0, 1.0)):
            raise ValueError("science method hyperparameters differ from the frozen contract")
        budgets = self.budgets
        if tuple(asdict(budgets).values()) != (3_000, 1_000, 20_000, 2_000, 10_000):
            raise ValueError("science budgets differ from the frozen comparison")
        rng = self.rng
        mapping = science_rng_mapping(rng.base_seeds, rng.bootstrap_seed)
        if (
            rng.base_seeds != FORMAL_BASE_SEEDS
            or rng.bootstrap_seed != FORMAL_BOOTSTRAP_SEED
            or rng.mapping_count != FORMAL_MAPPING_COUNT
            or rng.mapping_sha256 != FORMAL_MAPPING_SHA256
            or rng.untouched_optional_preflight_reserve != OPTIONAL_PREFLIGHT_RESERVE
            or len(mapping) != FORMAL_MAPPING_COUNT
            or _canonical_sha256(mapping) != FORMAL_MAPPING_SHA256
            or len(set(mapping.values())) != FORMAL_MAPPING_COUNT
        ):
            raise ValueError("science RNG bank or full stream mapping differs")
        if set(mapping.values()) & (set(REPAIR_REPLAY_IDS) | set(OPTIONAL_PREFLIGHT_RESERVE)):
            raise ValueError("science RNG mapping overlaps repair or optional reserve IDs")
        if self.rng.seed_namespace != "native_synthetic_signed_gamma_science_v1:121400..121590:step10":
            raise ValueError("science seed namespace differs")
        if self.devices != ("cuda:0", "cuda:1"):
            raise ValueError("formal science requires exactly cuda:0,cuda:1")
        if self.output_root != Path("results/work/native_synthetic_signed_gamma_six_method_science_v1"):
            raise ValueError("formal science output root differs")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for section, names in {
            "design": ("gammas", "methods", "outcome_normalization"),
            "rng": ("base_seeds", "untouched_optional_preflight_reserve"),
        }.items():
            for name in names:
                payload[section][name] = list(payload[section][name])
        for name in ("repair_root", "repair_runner", "repair_config"):
            payload["parent"][name] = str(payload["parent"][name])
        payload["devices"] = list(payload["devices"])
        payload["output_root"] = str(payload["output_root"])
        return payload


@dataclass(frozen=True)
class ExecutionBudget:
    calibration: int
    grid: int
    reference: int
    online: int
    q_grid_size: int


@dataclass(frozen=True)
class NativeOnlineEnvironment:
    kernel: NativeSignedGammaKernel


@dataclass(frozen=True)
class GateBinding:
    repair_root: str
    repair_protocol: str
    administrative_role: str
    decision: str
    n_prespecified: int
    n_passed: int
    required_passed: int
    source_tree_sha256: str
    amendment_sha256: str
    repair_config_sha256: str
    parent_manifest_sha256: str
    scientific_config_sha256: str
    replay_rng_audit_sha256: str
    downstream_rng_reservation_sha256: str
    downstream_rng_mapping_sha256: str
    downstream_rng_mapping_count: int
    source_snapshot_sha256: str
    manifest_sha256: str
    complete_sha256: str
    completion_contract_sha256: str
    files: dict[str, dict[str, Any]]
    binding_sha256: str


class NativePolicyGridAdapter:
    """Expose the canonical scalar Native policy on a candidate grid."""

    def __init__(self, policy: NativeSignedGammaRadiusPolicy) -> None:
        self.policy = policy

    @property
    def n_actions(self) -> int:
        return self.policy.n_actions

    @property
    def config(self) -> NativeSignedGammaDGPConfig:
        return self.policy.config

    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        return self.policy.probabilities(states, q)

    def probabilities_for_grid(self, states: Tensor, q_grid: Tensor) -> Tensor:
        if q_grid.ndim != 1 or len(q_grid) < 1:
            raise ValueError("q_grid must be a nonempty vector")
        if not bool(torch.isfinite(q_grid).all()):
            raise ValueError("q_grid must be finite")
        reference = self.policy.logging_policy.probabilities(states)
        response = self.policy.response_weight(q_grid.to(states), like=states)
        coordinate = action_coordinate(states)
        weights = torch.exp(
            -self.config.policy_tilt
            * response[None, :, None]
            * coordinate[None, None, :]
        ).expand(len(states), -1, -1)
        expanded_reference = reference[:, None, :].expand_as(weights)
        probabilities = _ratio_capped_tilt_grid(
            expanded_reference.reshape(-1, self.n_actions),
            weights.reshape(-1, self.n_actions),
            self.config.policy_ratio_cap,
        )
        return probabilities.reshape(len(states), len(q_grid), self.n_actions)


class NativeOutcomeModel(nn.Module):
    """Exact conditional marginal mean/scale for the fixed Native kernel.

    The calculation integrates both Bernoulli difficulty and Gaussian state
    innovations, including the coordinate-wise state clipping.  It has no
    learned parameters and consumes no RNG stream.
    """

    def __init__(self, kernel: NativeSignedGammaKernel, *, device: str | torch.device) -> None:
        super().__init__()
        self.kernel = kernel
        self.device_anchor = nn.Parameter(
            torch.zeros((), dtype=torch.float64, device=device),
            requires_grad=False,
        )

    def forward(self, states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        states = states.to(self.device_anchor)
        actions = actions.to(device=self.device_anchor.device, dtype=torch.long)
        if states.ndim != 2 or states.shape[1] != self.kernel.state_dim:
            raise ValueError("native outcome states must have shape [N,4]")
        if actions.shape != (len(states),):
            raise ValueError("native outcome actions must have shape [N]")
        difficulty_probability = self.kernel.difficulty_probability(states, actions)
        tail_probability = self.kernel.tail_probability(states, actions)
        z1, z2 = states[:, 0], states[:, 1]
        base_means = torch.stack((0.78 * z1 + 0.12 * z2, 0.08 * z1 + 0.72 * z2), dim=1)
        transition_effect = states.new_tensor((0.30, 0.20))
        outcome_effect = states.new_tensor((0.25, 0.20))
        transition_sd = states.new_tensor(
            (0.25, 0.20 * math.sqrt(0.30**2 + 0.9539392014**2))
        )
        clip = float(self.kernel.config.state_clip)

        component_means = []
        component_seconds = []
        for difficulty in (0.0, 1.0):
            latent_mean = base_means + difficulty * transition_effect[None, :]
            clipped_mean, clipped_second = _clipped_normal_moments(
                latent_mean,
                transition_sd[None, :],
                lower=-clip,
                upper=clip,
            )
            shifted_mean = clipped_mean + difficulty * outcome_effect[None, :]
            shifted_second = (
                clipped_second
                + 2.0 * difficulty * outcome_effect[None, :] * clipped_mean
                + difficulty * outcome_effect.square()[None, :]
            )
            component_means.append(shifted_mean)
            component_seconds.append(shifted_second)
        probability = difficulty_probability[:, None]
        mean = (1.0 - probability) * component_means[0] + probability * component_means[1]
        second = (1.0 - probability) * component_seconds[0] + probability * component_seconds[1]
        base_outcome_sd = states.new_tensor((0.35, 0.25))
        multiplier_second = 1.0 + tail_probability[:, None] * (
            self.kernel.config.tail_scale**2 - 1.0
        )
        residual_variance = base_outcome_sd.square()[None, :] * multiplier_second
        variance = (second - mean.square() + residual_variance).clamp_min(1e-12)
        return mean, variance.sqrt()


def _clipped_normal_moments(
    mean: Tensor,
    scale: Tensor,
    *,
    lower: float,
    upper: float,
) -> tuple[Tensor, Tensor]:
    if not lower < upper or bool((scale <= 0.0).any()):
        raise ValueError("clipped-normal bounds/scales are invalid")
    z_lower = (lower - mean) / scale
    z_upper = (upper - mean) / scale
    sqrt_two = math.sqrt(2.0)
    sqrt_two_pi = math.sqrt(2.0 * math.pi)
    cdf_lower = 0.5 * (1.0 + torch.erf(z_lower / sqrt_two))
    cdf_upper = 0.5 * (1.0 + torch.erf(z_upper / sqrt_two))
    pdf_lower = torch.exp(-0.5 * z_lower.square()) / sqrt_two_pi
    pdf_upper = torch.exp(-0.5 * z_upper.square()) / sqrt_two_pi
    inside = cdf_upper - cdf_lower
    first_inside = mean * inside + scale * (pdf_lower - pdf_upper)
    second_inside = (
        (mean.square() + scale.square()) * inside
        + 2.0 * mean * scale * (pdf_lower - pdf_upper)
        + scale.square() * (z_lower * pdf_lower - z_upper * pdf_upper)
    )
    first = lower * cdf_lower + first_inside + upper * (1.0 - cdf_upper)
    second = lower**2 * cdf_lower + second_inside + upper**2 * (1.0 - cdf_upper)
    return first, second


def _ratio_capped_tilt_grid(reference: Tensor, weights: Tensor, cap: float) -> Tensor:
    if reference.shape != weights.shape or reference.ndim != 2:
        raise ValueError("reference and weights must have matching [N,A] shapes")
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


def science_rng_mapping(
    base_seeds: Sequence[int] = FORMAL_BASE_SEEDS,
    bootstrap_seed: int = FORMAL_BOOTSTRAP_SEED,
) -> dict[str, int]:
    mapping: dict[str, int] = {
        "summary/bootstrap_complete_seed_matrix": int(bootstrap_seed)
    }
    for seed_value in base_seeds:
        seed = int(seed_value)
        prefix = f"science/base_{seed}"
        mapping[f"{prefix}/task"] = seed
        mapping[f"{prefix}/calibration"] = _paper_seed(seed, 1_700_101)
        mapping[f"{prefix}/reference"] = _paper_seed(seed, 1_700_401)
        adaptation_root = _paper_seed(seed, 700_001)
        for round_index in range(3):
            mapping[f"{prefix}/ACI_round_{round_index}"] = (
                _paper_seed(adaptation_root, 101) + 17_923 * round_index
            )
            mapping[f"{prefix}/SPCI_round_{round_index}"] = (
                _paper_seed(adaptation_root, 211) + 47_021 * round_index
            )
            mapping[f"{prefix}/PRC_round_{round_index}"] = (
                _paper_seed(adaptation_root, 307) + 61_103 * round_index
            )
    return mapping


def adaptation_seeds(seed: int) -> dict[str, int]:
    root_seed = _paper_seed(seed, 700_001)
    return {
        "ACI": _paper_seed(root_seed, 101),
        "SPCI": _paper_seed(root_seed, 211),
        "PRC": _paper_seed(root_seed, 307),
    }


def execution_rng_mapping(seed: int) -> dict[str, int]:
    """Return every RNG ID consumed by one complete-seed worker."""

    mapping = {
        "task": int(seed),
        "calibration": _paper_seed(seed, 1_700_101),
        "reference": _paper_seed(seed, 1_700_401),
    }
    method_roots = adaptation_seeds(seed)
    strides = {"ACI": 17_923, "SPCI": 47_021, "PRC": 61_103}
    for method in ADAPTIVE_METHODS:
        for round_index in range(3):
            mapping[f"{method}_round_{round_index}"] = (
                method_roots[method] + strides[method] * round_index
            )
    return mapping


def seed_device_mapping(config: ScienceConfig) -> dict[str, str]:
    return {
        str(seed): config.devices[index % len(config.devices)]
        for index, seed in enumerate(config.rng.base_seeds)
    }


def _formal_execution_budget(config: ScienceConfig) -> ExecutionBudget:
    return ExecutionBudget(
        calibration=config.budgets.calibration_trajectories,
        grid=config.budgets.grid_trajectories,
        reference=config.budgets.reference_trajectories,
        online=config.budgets.online_trajectories_per_adaptive_method,
        q_grid_size=config.design.q_grid_size,
    )


def _native_batch(trajectory: NativeSignedGammaTrajectory) -> TrajectoryBatch:
    return TrajectoryBatch(
        states=trajectory.states,
        actions=trajectory.actions,
        outcomes=trajectory.outcomes,
        patient_ids=torch.arange(
            trajectory.n,
            device=trajectory.actions.device,
            dtype=torch.long,
        ),
    )


@torch.no_grad()
def _native_online_rollout(
    environment: object,
    policy: object,
    *,
    n: int,
    horizon: int,
    seed: int,
    device: str | torch.device,
    q: Tensor,
) -> TrajectoryBatch:
    if not isinstance(environment, NativeOnlineEnvironment):
        raise TypeError("native online rollout requires NativeOnlineEnvironment")
    if not isinstance(policy, NativePolicyGridAdapter):
        raise TypeError("native online rollout requires NativePolicyGridAdapter")
    if horizon != 12 or q.shape != (horizon,):
        raise ValueError("native online rollout requires one radius per stage")
    noise = make_native_signed_gamma_noise(
        n=n,
        horizon=horizon,
        seed=seed,
        device=device,
    )
    return _native_batch(
        rollout_native_signed_gamma(
            environment.kernel,
            policy.policy,
            noise,
            radius=q.to(device),
        )
    )


def run_seed(
    config: ScienceConfig,
    seed: int,
    *,
    device: str,
    budget: ExecutionBudget | None = None,
    formal: bool = True,
) -> list[dict[str, Any]]:
    """Run one complete-seed vector; fixture mode rejects every formal ID."""

    if formal:
        if seed not in FORMAL_BASE_SEEDS or not device.startswith("cuda:"):
            raise ValueError("formal Native science requires a reserved base seed and CUDA")
        resolved_budget = _formal_execution_budget(config)
        gammas = GAMMAS
        q_grid_size = config.design.q_grid_size
    else:
        protected = (
            set(science_rng_mapping().values())
            | set(REPAIR_REPLAY_IDS)
            | set(OPTIONAL_PREFLIGHT_RESERVE)
        )
        collisions = set(execution_rng_mapping(seed).values()) & protected
        if collisions:
            raise ValueError(
                "fixture execution cannot consume a protected Native RNG ID: "
                f"{sorted(collisions)}"
            )
        if budget is None:
            raise ValueError("fixture execution requires an explicit tiny budget")
        resolved_budget = budget
        gammas = GAMMAS
        q_grid_size = budget.q_grid_size
    if resolved_budget.grid > resolved_budget.calibration:
        raise ValueError("grid budget cannot exceed calibration budget")

    torch.manual_seed(seed)
    dgp = NativeSignedGammaDGPConfig(policy_ratio_cap=config.design.policy_ratio_cap)
    logging_policy = NativeSignedGammaLoggingPolicy(dgp)
    target_policy = NativePolicyGridAdapter(NativeSignedGammaRadiusPolicy(logging_policy))
    calibration_noise = make_native_signed_gamma_noise(
        n=resolved_budget.calibration,
        horizon=config.design.horizon,
        seed=_paper_seed(seed, 1_700_101),
        device=device,
    )
    reference_noise = make_native_signed_gamma_noise(
        n=resolved_budget.reference,
        horizon=config.design.horizon,
        seed=_paper_seed(seed, 1_700_401),
        device=device,
    )
    online_seeds = adaptation_seeds(seed)
    rows = []
    for gamma in gammas:
        kernel = NativeSignedGammaKernel(dgp, gamma)
        outcome_model = NativeOutcomeModel(kernel, device=device)
        outcome_sd = torch.tensor(
            config.design.outcome_normalization,
            dtype=torch.float64,
            device=device,
        )
        source_calibration_native = rollout_native_signed_gamma(
            kernel,
            logging_policy,
            calibration_noise,
        )
        source_calibration = _native_batch(source_calibration_native)
        calibration_scores = score_batch(
            outcome_model,
            source_calibration.current_states(),
            source_calibration.actions,
            source_calibration.outcomes,
        )
        # The canonical grid constructor inherits PyTorch's default float32
        # quantile grid.  Keep its established dtype contract while retaining
        # float64 trajectories/scores for calibration and evaluation.
        grid_scores = calibration_scores[: resolved_budget.grid].to(torch.float32)
        stage_grids = torch.stack(
            [
                fixed_q_grid(
                    grid_scores[:, stage],
                    size=q_grid_size,
                    lower_quantile=config.design.q_quantile_min,
                    upper_quantile=config.design.q_quantile_max,
                )
                for stage in range(config.design.horizon)
            ]
        )
        stage_profile = stage_score_profile(grid_scores, alpha=config.design.alpha)
        scale_grid = profiled_scale_grid(
            grid_scores,
            stage_profile,
            size=q_grid_size,
            lower_quantile=config.design.q_quantile_min,
            upper_quantile=config.design.q_quantile_max,
        )

        standard = standard_cp_stagewise_radii(calibration_scores, config.design.alpha)
        mfcs, _ = finite_depth_mfcs_selection(
            source_calibration,
            calibration_scores,
            q_grid=scale_grid,
            stage_profile=stage_profile,
            target_policy=target_policy,
            logging_policy=logging_policy,
            depth=config.design.mfcs_depth,
            alpha=config.design.alpha,
            weight_cap=config.design.weight_cap,
        )
        scpcp = select_marginal_prefix_schedule(
            source_calibration,
            calibration_scores,
            stage_grids=stage_grids,
            target_policy=target_policy,
            logging_policy=logging_policy,
            outcome_model=outcome_model,
            outcome_sd=outcome_sd,
            target=1.0 - config.design.alpha,
        )

        online_environment = NativeOnlineEnvironment(kernel)
        aci = aci_style_controller(
            online_environment,
            target_policy,
            outcome_model,
            calibration_scores,
            alpha=config.design.alpha,
            gamma=config.design.aci_gamma,
            rounds=config.design.online_rounds,
            total_rollouts=resolved_budget.online,
            horizon=config.design.horizon,
            seed=online_seeds["ACI"],
            device=device,
            rollout_fn=_native_online_rollout,
        )
        spci = multidim_spci_style_controller(
            online_environment,
            target_policy,
            outcome_model,
            calibration_scores,
            alpha=config.design.alpha,
            rounds=config.design.online_rounds,
            total_rollouts=resolved_budget.online,
            horizon=config.design.horizon,
            seed=online_seeds["SPCI"],
            device=device,
            residual_window=config.design.multidim_buffer,
            rollout_fn=_native_online_rollout,
        )
        initial_prc_scale = float((standard / stage_profile.to(standard)).max().item())
        prc = prc_profile_scale(
            online_environment,
            target_policy,
            outcome_model,
            initial_prc_scale,
            scale_grid,
            stage_profile,
            alpha=config.design.alpha,
            delta=config.design.delta,
            rounds=config.design.online_rounds,
            total_rollouts=resolved_budget.online,
            horizon=config.design.horizon,
            seed=online_seeds["PRC"],
            device=device,
            maximum_step=config.design.prc_maximum_step,
            rollout_fn=_native_online_rollout,
        )
        expected_online_budget = (
            config.budgets.online_trajectories_per_adaptive_method
            if formal
            else resolved_budget.online
        )
        for name, adaptation in (("ACI", aci), ("SPCI", spci), ("PRC", prc)):
            if adaptation.target_deployments != expected_online_budget:
                raise RuntimeError(f"{name} did not consume its exact target-data budget")

        source_reference_native = rollout_native_signed_gamma(
            kernel,
            logging_policy,
            reference_noise,
        )
        source_reference = _native_batch(source_reference_native)
        source_scores = score_batch(
            outcome_model,
            source_reference.current_states(),
            source_reference.actions,
            source_reference.outcomes,
        )
        schedules: dict[str, Tensor | None] = {
            "Standard CP": standard,
            "ACI": aci.radius_by_time.to(device),
            "MFCS": None if mfcs.radius is None else mfcs.radius * stage_profile.to(calibration_scores),
            "SPCI": spci.radius_by_time.to(device),
            "PRC": prc.radius_by_time.to(device),
            "SC-PCP": scpcp.radii,
        }
        adaptations = {"ACI": aci, "SPCI": spci, "PRC": prc}
        method_rows = {
            method: _evaluate_method(
                method,
                schedules[method],
                source_reference=source_reference,
                source_reference_native=source_reference_native,
                source_scores=source_scores,
                kernel=kernel,
                target_policy=target_policy,
                logging_policy=logging_policy,
                reference_noise=reference_noise,
                outcome_model=outcome_model,
                outcome_sd=outcome_sd,
                adaptation=adaptations.get(method),
                expected_online_budget=expected_online_budget,
                selection_status=_selection_status(method, mfcs.status, scpcp.radii is not None),
            )
            for method in METHODS
        }
        rows.append(
            {
                "seed": seed,
                "gamma": gamma,
                "gamma_role": (
                    "primary_confirmatory_method_comparison"
                    if gamma == PRIMARY_GAMMA
                    else "descriptive_signed_mechanism_curve_no_ranking_or_superiority"
                ),
                "kernel_fingerprint": kernel.fingerprint,
                "adaptation_seeds": online_seeds,
                "scpcp_minimum_ess_fraction": _minimum_fraction(
                    scpcp.effective_sample_size,
                    resolved_budget.calibration,
                ),
                "scpcp_minimum_candidate_ess_fraction": _minimum_fraction(
                    scpcp.candidate_effective_sample_size,
                    resolved_budget.calibration,
                ),
                "scpcp_selected_endpoint": scpcp.selected_endpoint,
                "scpcp_failure_stage": scpcp.failure_stage,
                "methods": method_rows,
            }
        )
    return rows


def _selection_status(method: str, mfcs_status: str, scpcp_available: bool) -> str:
    if method == "MFCS":
        return mfcs_status
    if method == "SC-PCP":
        return "SELECTED_MARGINAL_POINT" if scpcp_available else "UNAVAILABLE_NO_FEASIBLE_CANDIDATE"
    return "AVAILABLE"


@torch.no_grad()
def _evaluate_method(
    method: str,
    schedule: Tensor | None,
    *,
    source_reference: TrajectoryBatch,
    source_reference_native: NativeSignedGammaTrajectory,
    source_scores: Tensor,
    kernel: NativeSignedGammaKernel,
    target_policy: NativePolicyGridAdapter,
    logging_policy: NativeSignedGammaLoggingPolicy,
    reference_noise: object,
    outcome_model: NativeOutcomeModel,
    outcome_sd: Tensor,
    adaptation: OnlineBaselineResult | None,
    expected_online_budget: int,
    selection_status: str,
) -> dict[str, Any]:
    adaptation_budget = 0 if adaptation is None else adaptation.target_deployments
    frozen_budget = expected_online_budget if method in ADAPTIVE_METHODS else 0
    if adaptation_budget != frozen_budget:
        raise RuntimeError(f"{method} information budget mismatch")
    common: dict[str, Any] = {
        "selection_available": schedule is not None,
        "selection_status": selection_status,
        "information_regime": INFORMATION_REGIME[method],
        "target_adaptation_trajectories": adaptation_budget,
    }
    if adaptation is not None:
        common.update(
            {
                "adaptation_rounds": adaptation.rounds,
                "adaptation_per_time_coverage": _vector(adaptation.adaptation_per_time_coverage),
                "adaptation_round_worst_coverage": [float(value) for value in adaptation.adaptation_round_worst_coverage],
                "adaptation_pathwise_coverage": float(adaptation.adaptation_pathwise_coverage),
                "selected_scale": None if adaptation.selected_scale is None else float(adaptation.selected_scale),
            }
        )
    if schedule is None:
        return {**common, "radii": []}

    resolved = schedule.to(source_scores)
    target_native = rollout_native_signed_gamma(
        kernel,
        target_policy.policy,
        reference_noise,
        radius=resolved,
    )
    target = _native_batch(target_native)
    target_scores = score_batch(
        outcome_model,
        target.current_states(),
        target.actions,
        target.outcomes,
    )
    ess, maximum_share, log_span = _prefix_diagnostics(
        source_reference,
        schedule=resolved,
        target_policy=target_policy,
        logging_policy=logging_policy,
    )
    source_coverage = (source_scores <= resolved[None, :]).to(torch.float64).mean(dim=0)
    target_coverage = (target_scores <= resolved[None, :]).to(torch.float64).mean(dim=0)
    normalized_width = _normalized_width_by_stage(
        outcome_model,
        target,
        schedule=resolved,
        outcome_sd=outcome_sd,
    )
    return {
        **common,
        "radii": _vector(resolved),
        "source_coverage": _vector(source_coverage),
        "target_coverage": _vector(target_coverage),
        "coverage_gap": _vector(target_coverage - source_coverage),
        "target_normalized_width": _vector(normalized_width),
        "prefix_ess_fraction": _vector(ess),
        "maximum_normalized_weight_share": _vector(maximum_share),
        "raw_log_weight_span": _vector(log_span),
        "policy_tv_on_source_states": _vector(
            _policy_tv_by_stage(
                source_reference,
                schedule=resolved,
                target_policy=target_policy,
                logging_policy=logging_policy,
            )
        ),
        "source_difficulty_by_stage": _vector(source_reference_native.next_difficulty.mean(dim=0)),
        "target_difficulty_by_stage": _vector(target_native.next_difficulty.mean(dim=0)),
        "source_tail_by_stage": _vector(source_reference_native.tail_indicators.mean(dim=0)),
        "target_tail_by_stage": _vector(target_native.tail_indicators.mean(dim=0)),
    }


@torch.no_grad()
def _prefix_diagnostics(
    batch: TrajectoryBatch,
    *,
    schedule: Tensor,
    target_policy: NativePolicyGridAdapter,
    logging_policy: NativeSignedGammaLoggingPolicy,
) -> tuple[Tensor, Tensor, Tensor]:
    log_weight = torch.zeros(batch.n, dtype=torch.float64, device=batch.actions.device)
    ess, shares, spans = [], [], []
    for stage, radius in enumerate(schedule):
        states = batch.states[:, stage]
        actions = batch.actions[:, stage]
        target = target_policy.probabilities(states, radius)
        source = logging_policy.probabilities(states)
        ratio = target.gather(1, actions[:, None]).squeeze(1) / source.gather(1, actions[:, None]).squeeze(1)
        log_weight += ratio.to(torch.float64).log()
        stabilized = (log_weight - log_weight.max()).exp()
        total = stabilized.sum().clamp_min(1e-12)
        ess.append(total.square() / stabilized.square().sum().clamp_min(1e-12) / batch.n)
        shares.append(stabilized.max() / total)
        spans.append(log_weight.max() - log_weight.min())
    return torch.stack(ess), torch.stack(shares), torch.stack(spans)


@torch.no_grad()
def _policy_tv_by_stage(
    batch: TrajectoryBatch,
    *,
    schedule: Tensor,
    target_policy: NativePolicyGridAdapter,
    logging_policy: NativeSignedGammaLoggingPolicy,
) -> Tensor:
    return torch.stack(
        [
            0.5
            * (
                target_policy.probabilities(batch.states[:, stage], radius)
                - logging_policy.probabilities(batch.states[:, stage])
            )
            .abs()
            .sum(dim=1)
            .mean()
            for stage, radius in enumerate(schedule)
        ]
    )


@torch.no_grad()
def _normalized_width_by_stage(
    outcome_model: NativeOutcomeModel,
    batch: TrajectoryBatch,
    *,
    schedule: Tensor,
    outcome_sd: Tensor,
) -> Tensor:
    states, actions, _ = batch.flat_transitions()
    scales = []
    for state_part, action_part in zip(states.split(4_096), actions.split(4_096), strict=True):
        _, scale = outcome_model(state_part, action_part)
        scales.append(scale)
    scale = torch.cat(scales).reshape(batch.n, batch.horizon, -1)
    normalized = 2.0 * schedule[None, :, None] * scale / outcome_sd[None, None, :]
    return normalized.mean(dim=(0, 2))


def _minimum_fraction(values: Tensor, denominator: int) -> float | None:
    return None if values.numel() == 0 else float(values.min().item() / denominator)


def _vector(values: Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().to(torch.float64).tolist()]


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: ScienceConfig,
    bootstrap_uniforms: np.ndarray,
    bootstrap_contract: Mapping[str, Any],
) -> dict[str, Any]:
    seeds = config.rng.base_seeds
    if len(rows) != len(seeds) * len(GAMMAS):
        raise RuntimeError("summary requires every prespecified seed and gamma")
    _validate_bootstrap_uniforms(bootstrap_uniforms, config)
    aggregates = []
    for gamma in GAMMAS:
        selected = sorted(
            (row for row in rows if float(row["gamma"]) == gamma),
            key=lambda row: int(row["seed"]),
        )
        if tuple(int(row["seed"]) for row in selected) != seeds:
            raise RuntimeError(f"summary seed set differs for gamma={gamma:g}")
        methods = {
            method: _summarize_method(
                method,
                selected,
                bootstrap_uniforms=bootstrap_uniforms,
                horizon=config.design.horizon,
            )
            for method in METHODS
        }
        paired = (
            {
                baseline: _paired_scpcp_comparison(
                    selected,
                    baseline=baseline,
                    bootstrap_uniforms=bootstrap_uniforms,
                )
                for baseline in METHODS
                if baseline != "SC-PCP"
            }
            if gamma == PRIMARY_GAMMA
            else {}
        )
        aggregates.append(
            {
                "gamma": gamma,
                "gamma_role": (
                    "primary_confirmatory_method_comparison"
                    if gamma == PRIMARY_GAMMA
                    else "descriptive_signed_mechanism_curve_no_ranking_or_superiority"
                ),
                "n_prespecified_seeds": len(seeds),
                "methods": methods,
                "paired_scpcp_comparisons": paired,
            }
        )
    return {
        "protocol": config.protocol,
        "role": config.role,
        "primary_gamma": PRIMARY_GAMMA,
        "gammas": list(GAMMAS),
        "methods": list(METHODS),
        "seeds": list(seeds),
        "primary_metric": SCIENCE_CONTRACT["coverage_metric"],
        "mean_coverage_metric": SCIENCE_CONTRACT["mean_coverage_metric"],
        "coverage_conditioning": SCIENCE_CONTRACT["coverage_conditioning"],
        "selection_rate_denominator": SCIENCE_CONTRACT["selection_rate_denominator"],
        "signed_curve_interpretation": SCIENCE_CONTRACT["signed_curve_interpretation"],
        "information_budgets": {
            method: {
                "information_regime": INFORMATION_REGIME[method],
                "logged_calibration_trajectories_per_seed": config.budgets.calibration_trajectories,
                "grid_trajectories_per_seed": config.budgets.grid_trajectories,
                "fresh_reference_trajectories_per_seed": config.budgets.reference_trajectories,
                "target_adaptation_trajectories_per_seed": TARGET_ADAPTATION_BUDGET[method],
            }
            for method in METHODS
        },
        "bootstrap": dict(bootstrap_contract),
        "aggregates": aggregates,
    }


def _summarize_method(
    method: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_uniforms: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    available = np.asarray(
        [bool(row["methods"][method]["selection_available"]) for row in rows],
        dtype=bool,
    )
    successes = int(available.sum())
    total = len(rows)
    base = {
        "selection_successes": successes,
        "selection_total": total,
        "selection_rate": successes / total,
        "selection_rate_ci95": _wilson_interval(successes, total),
        "information_regime": INFORMATION_REGIME[method],
        "target_adaptation_trajectories_per_seed": TARGET_ADAPTATION_BUDGET[method],
        "target_adaptation_trajectories_total": TARGET_ADAPTATION_BUDGET[method] * total,
        "coverage_conditioning": "successful_selection",
    }
    if not successes:
        return {
            **base,
            "target_coverage_by_stage": [None] * horizon,
            "target_coverage_ci95_by_stage": [[None, None] for _ in range(horizon)],
            "target_mean_coverage": None,
            "target_mean_coverage_ci95": [None, None],
            "target_marginal_worst_coverage": None,
            "target_wsc_ci95": [None, None],
            "target_worst_stage_zero_based": None,
            "source_marginal_worst_coverage": None,
            "target_normalized_width_by_stage": [None] * horizon,
            "target_normalized_width_ci95_by_stage": [[None, None] for _ in range(horizon)],
            "mean_target_normalized_width": None,
            "mean_target_normalized_width_ci95": [None, None],
            "minimum_reference_prefix_ess_fraction": None,
            "maximum_reference_weight_share": None,
            "maximum_raw_log_weight_span": None,
        }
    selected = [row["methods"][method] for row, keep in zip(rows, available, strict=True) if keep]
    target_coverage = np.asarray([row["target_coverage"] for row in selected], dtype=np.float64)
    source_coverage = np.asarray([row["source_coverage"] for row in selected], dtype=np.float64)
    widths = np.asarray([row["target_normalized_width"] for row in selected], dtype=np.float64)
    bootstrap = _bootstrap_indices(bootstrap_uniforms, successes)
    target_draws = target_coverage[bootstrap].mean(axis=1)
    stage_coverage = target_coverage.mean(axis=0)
    stage_width = widths.mean(axis=0)
    wsc_draws = target_draws.min(axis=1)
    return {
        **base,
        "target_coverage_by_stage": stage_coverage.tolist(),
        "target_coverage_ci95_by_stage": [
            _student_t_interval(target_coverage[:, stage]) for stage in range(horizon)
        ],
        "target_mean_coverage": float(stage_coverage.mean()),
        "target_mean_coverage_ci95": _student_t_interval(
            target_coverage.mean(axis=1)
        ),
        "target_marginal_worst_coverage": float(stage_coverage.min()),
        "target_wsc_ci95": _percentile_interval(wsc_draws),
        "target_worst_stage_zero_based": int(stage_coverage.argmin()),
        "source_marginal_worst_coverage": float(source_coverage.mean(axis=0).min()),
        "target_normalized_width_by_stage": stage_width.tolist(),
        "target_normalized_width_ci95_by_stage": [
            _student_t_interval(widths[:, stage]) for stage in range(horizon)
        ],
        "mean_target_normalized_width": float(stage_width.mean()),
        "mean_target_normalized_width_ci95": _student_t_interval(
            widths.mean(axis=1)
        ),
        "minimum_reference_prefix_ess_fraction": float(
            min(min(row["prefix_ess_fraction"]) for row in selected)
        ),
        "maximum_reference_weight_share": float(
            max(max(row["maximum_normalized_weight_share"]) for row in selected)
        ),
        "maximum_raw_log_weight_span": float(
            max(max(row["raw_log_weight_span"]) for row in selected)
        ),
    }


def _paired_scpcp_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    bootstrap_uniforms: np.ndarray,
) -> dict[str, Any]:
    paired = [
        row
        for row in rows
        if row["methods"]["SC-PCP"]["selection_available"]
        and row["methods"][baseline]["selection_available"]
    ]
    if not paired:
        return {
            "paired_selected_seeds": 0,
            "scpcp_minus_baseline_wsc": None,
            "scpcp_minus_baseline_wsc_ci95": [None, None],
            "scpcp_to_baseline_geometric_width_ratio": None,
            "scpcp_to_baseline_geometric_width_ratio_ci95": [None, None],
        }
    scpcp_coverage = np.asarray(
        [row["methods"]["SC-PCP"]["target_coverage"] for row in paired],
        dtype=np.float64,
    )
    baseline_coverage = np.asarray(
        [row["methods"][baseline]["target_coverage"] for row in paired],
        dtype=np.float64,
    )
    scpcp_width = np.asarray(
        [row["methods"]["SC-PCP"]["target_normalized_width"] for row in paired],
        dtype=np.float64,
    ).mean(axis=1)
    baseline_width = np.asarray(
        [row["methods"][baseline]["target_normalized_width"] for row in paired],
        dtype=np.float64,
    ).mean(axis=1)
    if np.any(scpcp_width <= 0.0) or np.any(baseline_width <= 0.0):
        raise RuntimeError("paired geometric width ratios require positive widths")
    bootstrap = _bootstrap_indices(bootstrap_uniforms, len(paired))
    scpcp_draws = scpcp_coverage[bootstrap].mean(axis=1).min(axis=1)
    baseline_draws = baseline_coverage[bootstrap].mean(axis=1).min(axis=1)
    log_ratio = np.log(scpcp_width / baseline_width)
    ratio_draws = np.exp(log_ratio[bootstrap].mean(axis=1))
    scpcp_wsc = float(scpcp_coverage.mean(axis=0).min())
    baseline_wsc = float(baseline_coverage.mean(axis=0).min())
    return {
        "paired_selected_seeds": len(paired),
        "scpcp_minus_baseline_wsc": scpcp_wsc - baseline_wsc,
        "scpcp_minus_baseline_wsc_ci95": _percentile_interval(scpcp_draws - baseline_draws),
        "scpcp_to_baseline_geometric_width_ratio": float(np.exp(log_ratio.mean())),
        "scpcp_to_baseline_geometric_width_ratio_ci95": _percentile_interval(ratio_draws),
    }


def _bootstrap_indices(uniforms: np.ndarray, sample_size: int) -> np.ndarray:
    if uniforms.ndim != 2 or sample_size < 1 or sample_size > uniforms.shape[1]:
        raise ValueError("bootstrap sample size must fit the complete-seed matrix")
    return np.floor(uniforms[:, :sample_size] * sample_size).astype(np.int64)


def _percentile_interval(values: np.ndarray) -> list[float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("percentile interval requires a finite vector")
    return [float(value) for value in np.quantile(values, (0.025, 0.975), method="linear")]


def _student_t_interval(values: np.ndarray) -> list[float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Student-t interval requires a finite vector")
    mean = float(values.mean())
    if len(values) == 1:
        return [mean, mean]
    half = float(
        stats.t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / math.sqrt(len(values))
    )
    return [mean - half, mean + half]


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Wilson counts are invalid")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _make_bootstrap_arrays(config: ScienceConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.rng.bootstrap_seed)
    uniforms = rng.random(
        (config.budgets.bootstrap_resamples, len(config.rng.base_seeds)),
        dtype=np.float64,
    )
    return uniforms, _bootstrap_indices(uniforms, len(config.rng.base_seeds))


def _validate_bootstrap_uniforms(uniforms: np.ndarray, config: ScienceConfig) -> None:
    expected_shape = (config.budgets.bootstrap_resamples, len(config.rng.base_seeds))
    if (
        uniforms.shape != expected_shape
        or uniforms.dtype != np.float64
        or not np.isfinite(uniforms).all()
        or np.any(uniforms < 0.0)
        or np.any(uniforms >= 1.0)
    ):
        raise RuntimeError("complete-seed bootstrap uniform matrix differs")


def _bootstrap_contract(
    config: ScienceConfig,
    *,
    uniform_path: Path,
    index_path: Path,
) -> dict[str, Any]:
    return {
        "root_seed": config.rng.bootstrap_seed,
        "resamples": config.budgets.bootstrap_resamples,
        "complete_seed_count": len(config.rng.base_seeds),
        "uniform_shape": [config.budgets.bootstrap_resamples, len(config.rng.base_seeds)],
        "index_shape": [config.budgets.bootstrap_resamples, len(config.rng.base_seeds)],
        "uniform_dtype": "float64",
        "index_dtype": "int64",
        "uniform_path": uniform_path.name,
        "uniform_sha256": _file_sha256(uniform_path),
        "index_path": index_path.name,
        "index_sha256": _file_sha256(index_path),
        "unit": "complete_seed_stage_vector",
        "coupling": SCIENCE_CONTRACT["bootstrap"],
        "wsc_formula": SCIENCE_CONTRACT["coverage_metric"],
    }


def audit_science_rng_ids(
    config: ScienceConfig,
    *,
    gate_binding: GateBinding,
    output_root: Path,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Reject any prior actual use of all 241 frozen science RNG IDs."""

    artifact_root = ROOT / "results" if artifact_root is None else artifact_root
    source_root = ROOT if source_root is None else source_root
    mapping = science_rng_mapping(config.rng.base_seeds, config.rng.bootstrap_seed)
    formal_ids = set(mapping.values())
    if (
        len(mapping) != config.rng.mapping_count
        or len(formal_ids) != len(mapping)
        or _canonical_sha256(mapping) != config.rng.mapping_sha256
    ):
        raise RuntimeError("science RNG mapping is not the frozen one-to-one mapping")
    if (
        gate_binding.downstream_rng_mapping_sha256 != config.rng.mapping_sha256
        or gate_binding.downstream_rng_mapping_count != config.rng.mapping_count
    ):
        raise RuntimeError("repair gate did not reserve the exact downstream mapping")

    artifact_scan = preflight._artifact_rng_scan(artifact_root, excluded_root=output_root)
    excluded_source_paths = {
        Path(__file__).resolve(),
        config_path.resolve(),
    }
    source_scan = preflight._source_rng_scan(
        source_root,
        excluded_paths=excluded_source_paths,
    )
    artifact_actual = set(artifact_scan["actual"])
    source_actual = set(source_scan["actual"])
    external_sets = {
        name: set(values)
        for name, values in preflight.COORDINATED_EXTERNAL_RESERVATIONS.items()
    }
    external_ids = set().union(*external_sets.values()) if external_sets else set()
    # The repair artifact is the authority reserving this exact mapping.  Its
    # reservation fields are declarations, never evidence of prior use.
    prior_actual = artifact_actual | source_actual | external_ids
    collisions = {
        label: rng_id for label, rng_id in mapping.items() if rng_id in prior_actual
    }
    audit = {
        "status": "passed_before_launch" if not collisions else "collision",
        "policy": "live_actual_use_full_derived_mapping_v1",
        "seed_namespace": config.rng.seed_namespace,
        "formal_rng_id_count": len(formal_ids),
        "formal_rng_ids": sorted(formal_ids),
        "formal_rng_id_sha256": _integer_set_sha256(formal_ids),
        "formal_rng_mapping": mapping,
        "formal_rng_mapping_sha256": _canonical_sha256(mapping),
        "internal_rng_ids_unique": len(formal_ids) == len(mapping),
        "repair_authorized_reservation_sha256": gate_binding.downstream_rng_mapping_sha256,
        "repair_authorized_reservation_count": gate_binding.downstream_rng_mapping_count,
        "artifact_actual_rng_id_count": len(artifact_actual),
        "artifact_actual_rng_ids": sorted(artifact_actual),
        "artifact_actual_rng_id_sha256": _integer_set_sha256(artifact_actual),
        "artifact_declared_rng_id_count": len(artifact_scan["declared"]),
        "artifact_declared_rng_ids": sorted(artifact_scan["declared"]),
        "artifact_declared_rng_id_sha256": _integer_set_sha256(artifact_scan["declared"]),
        "artifact_reserved_rng_id_count": len(artifact_scan["reserved"]),
        "artifact_reserved_rng_ids": sorted(artifact_scan["reserved"]),
        "artifact_reserved_rng_id_sha256": _integer_set_sha256(artifact_scan["reserved"]),
        "artifact_binary_rng_bindings": artifact_scan["binary_bindings"],
        "artifact_binary_rng_binding_sha256": _canonical_sha256(artifact_scan["binary_bindings"]),
        "source_actual_rng_id_count": len(source_actual),
        "source_actual_rng_ids": sorted(source_actual),
        "source_actual_rng_id_sha256": _integer_set_sha256(source_actual),
        "source_declared_rng_id_count": len(source_scan["declared"]),
        "source_declared_rng_ids": sorted(source_scan["declared"]),
        "source_declared_rng_id_sha256": _integer_set_sha256(source_scan["declared"]),
        "source_reserved_rng_id_count": len(source_scan["reserved"]),
        "source_reserved_rng_ids": sorted(source_scan["reserved"]),
        "source_reserved_rng_id_sha256": _integer_set_sha256(source_scan["reserved"]),
        "source_unresolved_rng_expressions": [],
        "coordinated_external_rng_id_count": len(external_ids),
        "coordinated_external_rng_id_sha256": _integer_set_sha256(external_ids),
        "optional_preflight_reserve": list(config.rng.untouched_optional_preflight_reserve),
        "optional_preflight_reserve_sha256": _integer_set_sha256(
            config.rng.untouched_optional_preflight_reserve
        ),
        "collision_count": len(collisions),
        "collisions": collisions,
        "collision_sha256": _canonical_sha256(collisions),
        "excluded_output": str(output_root.resolve()),
        "excluded_source_declarations": sorted(str(path) for path in excluded_source_paths),
        "source_scan_policy": preflight.SOURCE_SCAN_CONTRACT,
        "artifact_scan_policy": preflight.ARTIFACT_SCAN_CONTRACT,
        "scan_policy_sha256": _canonical_sha256(
            {
                "source": preflight.SOURCE_SCAN_CONTRACT,
                "artifact": preflight.ARTIFACT_SCAN_CONTRACT,
            }
        ),
    }
    audit["audit_sha256"] = _canonical_sha256(audit)
    validate_science_rng_audit(config, gate_binding=gate_binding, audit=audit)
    if collisions:
        raise RuntimeError(f"Native science RNG IDs have prior-use collisions: {collisions}")
    return audit


def validate_science_rng_audit(
    config: ScienceConfig,
    *,
    gate_binding: GateBinding,
    audit: Mapping[str, Any],
) -> None:
    mapping = science_rng_mapping(config.rng.base_seeds, config.rng.bootstrap_seed)
    ids = set(mapping.values())
    if (
        audit.get("status") not in {"passed_before_launch", "collision"}
        or audit.get("formal_rng_id_count") != len(ids)
        or audit.get("formal_rng_ids") != sorted(ids)
        or audit.get("formal_rng_id_sha256") != _integer_set_sha256(ids)
        or audit.get("formal_rng_mapping") != mapping
        or audit.get("formal_rng_mapping_sha256") != _canonical_sha256(mapping)
        or audit.get("internal_rng_ids_unique") is not True
        or audit.get("repair_authorized_reservation_sha256")
        != gate_binding.downstream_rng_mapping_sha256
        or audit.get("repair_authorized_reservation_count")
        != gate_binding.downstream_rng_mapping_count
        or audit.get("source_unresolved_rng_expressions") != []
        or audit.get("collision_count") != len(audit.get("collisions", {}))
        or audit.get("collision_sha256") != _canonical_sha256(audit.get("collisions", {}))
    ):
        raise RuntimeError("Native science RNG audit contract differs")
    for prefix in ("artifact_actual", "artifact_declared", "artifact_reserved", "source_actual", "source_declared", "source_reserved"):
        count_key = f"{prefix}_rng_id_count"
        ids_key = f"{prefix}_rng_ids"
        hash_key = f"{prefix}_rng_id_sha256"
        values = audit.get(ids_key)
        if (
            not isinstance(values, list)
            or values != sorted(set(values))
            or audit.get(count_key) != len(values)
            or audit.get(hash_key) != _integer_set_sha256(values)
        ):
            raise RuntimeError(f"Native science RNG audit {prefix} fields differ")
    expected_status = "passed_before_launch" if not audit["collisions"] else "collision"
    without_hash = dict(audit)
    stored_hash = without_hash.pop("audit_sha256", None)
    if audit["status"] != expected_status or stored_hash != _canonical_sha256(without_hash):
        raise RuntimeError("Native science RNG audit hash or status differs")


def verify_repair_gate(config: ScienceConfig, *, source_root: Path = ROOT) -> GateBinding:
    """Validate the repair bundle and reduce it to an immutable science binding."""

    repair_root = (source_root / config.parent.repair_root).resolve()
    runner_path = (source_root / config.parent.repair_runner).resolve()
    repair_config_path = (source_root / config.parent.repair_config).resolve()
    if not repair_root.is_dir() or not (repair_root / "COMPLETE").is_file():
        raise RuntimeError("Native science is locked: repair COMPLETE is missing")
    if not runner_path.is_file() or not repair_config_path.is_file():
        raise RuntimeError("Native science is locked: repair source/config is missing")
    module = _load_repair_module(runner_path)
    validator = getattr(module, "validate_completed_repair_bundle", None)
    if not callable(validator):
        raise RuntimeError("repair runner lacks validate_completed_repair_bundle")
    completion_contract = validator(
        repair_root,
        source_root=source_root,
        amendment_path=repair_config_path,
    )
    expected_contract_fields = {
        "protocol",
        "role",
        "output_root",
        "decision",
        "downstream_authorized",
        "amendment_sha256",
        "parent_manifest_sha256",
        "scientific_config_sha256",
        "replay_rng_audit_sha256",
        "downstream_rng_reservation_sha256",
        "reserved_rng_mapping",
        "reserved_rng_mapping_sha256",
        "source_tree_sha256",
        "source_snapshot_sha256",
        "manifest_sha256",
        "complete_sha256",
        "metadata_sha256",
        "summary_sha256",
        "completion_contract_sha256",
    }
    _require_exact_keys(
        completion_contract,
        expected_contract_fields,
        "repair completion contract",
    )
    contract_without_hash = dict(completion_contract)
    completion_hash = contract_without_hash.pop("completion_contract_sha256")
    if completion_hash != _canonical_sha256(contract_without_hash):
        raise RuntimeError("repair completion contract hash differs")

    metadata = _read_json(repair_root / "metadata.json")
    summary = _read_json(repair_root / "summary.json")
    complete = _read_json(repair_root / "COMPLETE")
    repair_config = yaml.safe_load(repair_config_path.read_text(encoding="utf-8"))
    if not isinstance(repair_config, dict):
        raise RuntimeError("repair config root is malformed")
    reservation = repair_config.get("downstream_rng_reservation")
    if not isinstance(reservation, dict):
        raise RuntimeError("repair config lacks the downstream science reservation")
    _validate_downstream_reservation(reservation, config)

    decision = complete.get("decision")
    if (
        decision != config.parent.required_decision
        or summary.get("status") != decision
        or completion_contract["decision"] != decision
        or completion_contract["downstream_authorized"] is not True
        or complete.get("downstream_authorized") is not True
        or summary.get("downstream_authorized") is not True
    ):
        raise RuntimeError("Native science is locked: repair decision is not GO")
    if summary.get("n_prespecified") != 20:
        raise RuntimeError("repair summary must contain all 20 replay seeds")
    n_passed = summary.get("n_passed")
    required = summary.get("required_passed_rng_ids")
    passed_ids = summary.get("passed_rng_ids")
    if (
        summary.get("n_exact_replays") != 20
        or summary.get("n_repaired_fields_valid") != 20
        or not isinstance(passed_ids, list)
        or passed_ids != sorted(set(passed_ids))
        or len(passed_ids) != n_passed
        or any(seed not in REPAIR_REPLAY_IDS for seed in passed_ids)
        or not isinstance(n_passed, int)
        or not isinstance(required, int)
        or n_passed < required
        or required != 19
    ):
        raise RuntimeError("repair gate does not satisfy the unchanged 19/20 threshold")
    scientific_config = metadata.get("scientific_config")
    if not isinstance(scientific_config, dict) or scientific_config.get("protocol") != config.parent.scientific_protocol:
        raise RuntimeError("repair metadata scientific protocol differs")
    if (
        scientific_config.get("dgp") != asdict(NativeSignedGammaDGPConfig())
        or scientific_config.get("horizon") != config.design.horizon
        or scientific_config.get("gammas") != list(GAMMAS)
        or scientific_config.get("primary_gamma") != PRIMARY_GAMMA
        or scientific_config.get("calibration_trajectories")
        != config.budgets.calibration_trajectories
        or scientific_config.get("grid_trajectories")
        != config.budgets.grid_trajectories
        or scientific_config.get("reference_trajectories")
        != config.budgets.reference_trajectories
        or scientific_config.get("online_trajectories")
        != config.budgets.online_trajectories_per_adaptive_method
        or scientific_config.get("bootstrap_resamples")
        != config.budgets.bootstrap_resamples
    ):
        raise RuntimeError("repair scientific DGP or frozen science budgets differ")
    if metadata.get("role") != config.parent.administrative_role:
        raise RuntimeError("repair metadata administrative role differs")
    current_source_hash = preflight._experiment_tree_sha256(source_root)
    source_hash = metadata.get("source_tree_sha256")
    if not isinstance(source_hash, str) or source_hash != current_source_hash:
        raise RuntimeError("active experiment/source tree differs from the repair freeze")
    amendment_sha = metadata.get("amendment_sha256")
    config_sha = _file_sha256(repair_config_path)
    if not isinstance(amendment_sha, str) or amendment_sha != config_sha:
        raise RuntimeError("repair amendment/config hash differs")

    expected_mapping = science_rng_mapping(config.rng.base_seeds, config.rng.bootstrap_seed)
    if (
        completion_contract["protocol"]
        != "native_synthetic_signed_gamma_time_coordinate_repair_r1"
        or completion_contract["role"] != config.parent.administrative_role
        or completion_contract["output_root"] != str(repair_root)
        or completion_contract["amendment_sha256"] != config_sha
        or completion_contract["reserved_rng_mapping"] != expected_mapping
        or completion_contract["reserved_rng_mapping_sha256"]
        != config.rng.mapping_sha256
        or completion_contract["source_tree_sha256"] != source_hash
    ):
        raise RuntimeError("repair completion binding differs from Native science")

    files = {
        name: {
            "path": name,
            "sha256": _file_sha256(repair_root / name),
            "size_bytes": (repair_root / name).stat().st_size,
        }
        for name in ("metadata.json", "summary.json", "manifest.json", "COMPLETE")
    }
    contract_file_hashes = {
        "metadata.json": completion_contract["metadata_sha256"],
        "summary.json": completion_contract["summary_sha256"],
        "manifest.json": completion_contract["manifest_sha256"],
        "COMPLETE": completion_contract["complete_sha256"],
    }
    if any(files[name]["sha256"] != digest for name, digest in contract_file_hashes.items()):
        raise RuntimeError("repair completion files changed after public validation")
    core = {
        "repair_root": config.parent.repair_root.as_posix(),
        "repair_protocol": str(metadata.get("protocol")),
        "administrative_role": str(metadata.get("role")),
        "decision": decision,
        "n_prespecified": 20,
        "n_passed": n_passed,
        "required_passed": required,
        "source_tree_sha256": source_hash,
        "amendment_sha256": amendment_sha,
        "repair_config_sha256": config_sha,
        "parent_manifest_sha256": completion_contract["parent_manifest_sha256"],
        "scientific_config_sha256": completion_contract["scientific_config_sha256"],
        "replay_rng_audit_sha256": completion_contract["replay_rng_audit_sha256"],
        "downstream_rng_reservation_sha256": completion_contract[
            "downstream_rng_reservation_sha256"
        ],
        "downstream_rng_mapping_sha256": completion_contract[
            "reserved_rng_mapping_sha256"
        ],
        "downstream_rng_mapping_count": len(expected_mapping),
        "source_snapshot_sha256": completion_contract["source_snapshot_sha256"],
        "manifest_sha256": completion_contract["manifest_sha256"],
        "complete_sha256": completion_contract["complete_sha256"],
        "completion_contract_sha256": completion_contract[
            "completion_contract_sha256"
        ],
        "files": files,
    }
    return GateBinding(**core, binding_sha256=_canonical_sha256(core))


def _validate_downstream_reservation(
    reservation: Mapping[str, Any],
    config: ScienceConfig,
) -> None:
    mapping = science_rng_mapping(config.rng.base_seeds, config.rng.bootstrap_seed)
    expected_formula = {
        "paper_seed_multiplier": 1_000_003,
        "paper_seed_modulus": 2**31 - 1,
        "calibration_stream": 1_700_101,
        "reference_stream": 1_700_401,
        "adaptation_stream": 700_001,
        "methods": {
            "ACI": {"offset": 101, "round_stride": 17_923},
            "SPCI": {"offset": 211, "round_stride": 47_021},
            "PRC": {"offset": 307, "round_stride": 61_103},
        },
        "rounds": [0, 1, 2],
    }
    expected = {
        "status": "reserved_not_consumed",
        "namespace": config.rng.seed_namespace,
        "reserved_base_seeds": list(config.rng.base_seeds),
        "reserved_bootstrap_seed": config.rng.bootstrap_seed,
        "mapping_formula": expected_formula,
        "reserved_rng_id_count": len(mapping),
        "reserved_rng_mapping_sha256": _canonical_sha256(mapping),
    }
    if dict(reservation) != expected:
        raise RuntimeError("repair downstream RNG reservation differs from science contract")


def _load_repair_module(path: Path) -> ModuleType:
    name = "_native_signed_gamma_time_coordinate_repair_r1_binding"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load repair validator module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


METADATA_FIELDS = {
    "protocol",
    "role",
    "science_config",
    "science_config_path",
    "science_config_file_sha256",
    "science_config_payload_sha256",
    "science_contract",
    "gate_binding",
    "rng_audit",
    "seed_device_mapping",
    "seed_device_mapping_sha256",
    "source_tree_sha256",
    "source_snapshot",
    "dependency_files",
    "environment",
    "environment_sha256",
    "canonical_invocation",
    "canonical_invocation_sha256",
    "artifact_schema_sha256",
    "launch_contract_sha256",
}
SEED_FIELDS = {
    "protocol",
    "role",
    "seed",
    "device",
    "source_tree_sha256",
    "science_config_payload_sha256",
    "gate_binding_sha256",
    "rng_audit_sha256",
    "rng_mapping_sha256",
    "methods",
    "gammas",
    "budgets",
    "rows",
}
ROW_FIELDS = {
    "seed",
    "gamma",
    "gamma_role",
    "kernel_fingerprint",
    "adaptation_seeds",
    "scpcp_minimum_ess_fraction",
    "scpcp_minimum_candidate_ess_fraction",
    "scpcp_selected_endpoint",
    "scpcp_failure_stage",
    "methods",
}
METHOD_COMMON_FIELDS = {
    "selection_available",
    "selection_status",
    "information_regime",
    "target_adaptation_trajectories",
    "radii",
}
METHOD_ADAPTATION_FIELDS = {
    "adaptation_rounds",
    "adaptation_per_time_coverage",
    "adaptation_round_worst_coverage",
    "adaptation_pathwise_coverage",
    "selected_scale",
}
METHOD_SCIENCE_FIELDS = {
    "source_coverage",
    "target_coverage",
    "coverage_gap",
    "target_normalized_width",
    "prefix_ess_fraction",
    "maximum_normalized_weight_share",
    "raw_log_weight_span",
    "policy_tv_on_source_states",
    "source_difficulty_by_stage",
    "target_difficulty_by_stage",
    "source_tail_by_stage",
    "target_tail_by_stage",
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only and args.resume:
        parser.error("--validate-only and --resume are mutually exclusive")
    config = ScienceConfig.from_yaml(DEFAULT_CONFIG)
    if args.validate_only:
        print(json.dumps(validation_payload(config), sort_keys=True, allow_nan=False))
        return
    run_science(config, resume=args.resume)
    print((ROOT / config.output_root).resolve())


def validation_payload(
    config: ScienceConfig,
    *,
    source_root: Path = ROOT,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    gate = verify_repair_gate(config, source_root=source_root)
    output_root = (source_root / config.output_root).resolve()
    audit = audit_science_rng_ids(
        config,
        gate_binding=gate,
        output_root=output_root,
        artifact_root=artifact_root,
        source_root=source_root,
        config_path=source_root / DEFAULT_CONFIG.relative_to(ROOT),
    )
    return {
        "protocol": config.protocol,
        "role": config.role,
        "contract_valid": True,
        "formal_launch_permitted": audit["status"] == "passed_before_launch",
        "formal_launch_blocker": None,
        "repair_gate": asdict(gate),
        "science_rng_audit": audit,
        "formal_output_root": str(output_root),
        "output_root_exists": output_root.exists(),
        "no_rng_consumed": True,
        "no_artifact_written": True,
    }


def run_science(config: ScienceConfig, *, resume: bool = False) -> None:
    """Run or strictly resume the fixed formal benchmark."""

    output_root = (ROOT / config.output_root).resolve()
    config_path = DEFAULT_CONFIG.resolve()
    # Gate and live collision audit both happen before output-root creation.
    gate = verify_repair_gate(config)
    rng_audit = audit_science_rng_ids(
        config,
        gate_binding=gate,
        output_root=output_root,
        config_path=config_path,
    )
    if rng_audit["status"] != "passed_before_launch":
        raise RuntimeError("Native science RNG audit did not pass")
    source_hash = preflight._experiment_tree_sha256(ROOT)
    if source_hash != gate.source_tree_sha256:
        raise RuntimeError("active source differs from the validated repair gate")
    source_snapshot = preflight._build_source_snapshot(ROOT)
    if preflight._experiment_tree_sha256(ROOT) != source_hash:
        raise RuntimeError("experiment/source tree changed while building science snapshot")
    schema = _artifact_schema()
    metadata = _build_metadata(
        config,
        config_path=config_path,
        gate=gate,
        rng_audit=rng_audit,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        schema=schema,
    )

    if resume and (output_root / "COMPLETE").is_file():
        validate_completed_bundle(output_root, expected_metadata=metadata)
        return
    _prepare_root(
        output_root,
        metadata=metadata,
        schema=schema,
        source_snapshot=source_snapshot,
        resume=resume,
    )
    seed_contract = _seed_contract(metadata, config)
    existing = _load_seed_payloads(
        output_root,
        config=config,
        metadata=metadata,
        seed_contract=seed_contract,
    )
    pending = tuple(seed for seed in config.rng.base_seeds if seed not in existing)
    if pending:
        groups = [
            tuple(seed for seed in pending if metadata["seed_device_mapping"][str(seed)] == device)
            for device in config.devices
        ]
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(config.devices), mp_context=context) as executor:
            futures = {
                executor.submit(_run_seed_group, config, group, device): device
                for group, device in zip(groups, config.devices, strict=True)
                if group
            }
            for future in as_completed(futures):
                for seed, device, rows in future.result():
                    payload = {
                        **seed_contract,
                        "seed": seed,
                        "device": device,
                        "rows": rows,
                    }
                    _validate_seed_payload(
                        payload,
                        config=config,
                        metadata=metadata,
                        expected_seed=seed,
                        expected_device=device,
                    )
                    _write_json(output_root / "seeds" / f"seed_{seed}.json", payload)
                    existing[seed] = payload
                    print(f"completed Native science seed {seed}", flush=True)

    if set(existing) != set(config.rng.base_seeds):
        raise RuntimeError("Native science cannot summarize an incomplete seed bank")
    uniforms, indices, bootstrap = _ensure_bootstrap_artifacts(output_root, config)
    rows = [
        row
        for seed in config.rng.base_seeds
        for row in existing[seed]["rows"]
    ]
    summary = summarize(
        rows,
        config=config,
        bootstrap_uniforms=uniforms,
        bootstrap_contract=bootstrap,
    )
    _write_json(output_root / "summary.json", summary)
    audit = _coverage_audit(
        existing,
        summary=summary,
        config=config,
        bootstrap_uniforms=uniforms,
        bootstrap_contract=bootstrap,
    )
    _write_json(output_root / "coverage_audit.json", audit)
    final_status = {
        "protocol": config.protocol,
        "status": "COMPLETE",
        "decision": "SCIENCE_COMPLETE",
        "primary_gamma": PRIMARY_GAMMA,
        "primary_metric": SCIENCE_CONTRACT["coverage_metric"],
        "signed_curve_interpretation": SCIENCE_CONTRACT["signed_curve_interpretation"],
        "n_seeds": len(config.rng.base_seeds),
        "n_signed_gamma_rows": len(rows),
        "methods": list(METHODS),
        "gate_binding_sha256": gate.binding_sha256,
        "coverage_audit_sha256": _file_sha256(output_root / "coverage_audit.json"),
    }
    _write_json(output_root / "FINAL_STATUS.json", final_status)

    final_gate = verify_repair_gate(config)
    final_rng_audit = audit_science_rng_ids(
        config,
        gate_binding=final_gate,
        output_root=output_root,
        config_path=config_path,
    )
    if (
        final_gate != gate
        or final_rng_audit != rng_audit
        or preflight._experiment_tree_sha256(ROOT) != source_hash
    ):
        raise RuntimeError("gate, RNG inventory, or source changed during Native science")
    _finalize_root(output_root, metadata=metadata, config=config)


def _run_seed_group(
    config: ScienceConfig,
    seeds: tuple[int, ...],
    device: str,
) -> list[tuple[int, str, list[dict[str, Any]]]]:
    torch.cuda.set_device(torch.device(device))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    completed = []
    for seed in seeds:
        completed.append((seed, device, run_seed(config, seed, device=device)))
        torch.cuda.empty_cache()
    return completed


def _build_metadata(
    config: ScienceConfig,
    *,
    config_path: Path,
    gate: GateBinding,
    rng_audit: Mapping[str, Any],
    source_hash: str,
    source_snapshot: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    config_payload = config.to_dict()
    device_mapping = seed_device_mapping(config)
    environment = _runtime_environment(config.devices)
    invocation = ["scripts/run_native_synthetic_signed_gamma_science.py"]
    dependencies = _dependency_files(config, config_path=config_path)
    core = {
        "science_config_payload_sha256": _canonical_sha256(config_payload),
        "gate_binding_sha256": gate.binding_sha256,
        "rng_audit_sha256": rng_audit["audit_sha256"],
        "seed_device_mapping_sha256": _canonical_sha256(device_mapping),
        "source_tree_sha256": source_hash,
        "environment_sha256": _canonical_sha256(environment),
        "canonical_invocation_sha256": _canonical_sha256(invocation),
        "artifact_schema_sha256": _canonical_sha256(schema),
        "dependency_files_sha256": _canonical_sha256(dependencies),
    }
    return {
        "protocol": config.protocol,
        "role": config.role,
        "science_config": config_payload,
        "science_config_path": config_path.relative_to(ROOT).as_posix(),
        "science_config_file_sha256": _file_sha256(config_path),
        "science_config_payload_sha256": core["science_config_payload_sha256"],
        "science_contract": SCIENCE_CONTRACT,
        "gate_binding": asdict(gate),
        "rng_audit": dict(rng_audit),
        "seed_device_mapping": device_mapping,
        "seed_device_mapping_sha256": core["seed_device_mapping_sha256"],
        "source_tree_sha256": source_hash,
        "source_snapshot": dict(source_snapshot),
        "dependency_files": dependencies,
        "environment": environment,
        "environment_sha256": core["environment_sha256"],
        "canonical_invocation": invocation,
        "canonical_invocation_sha256": core["canonical_invocation_sha256"],
        "artifact_schema_sha256": core["artifact_schema_sha256"],
        "launch_contract_sha256": _canonical_sha256(core),
    }


def _dependency_files(config: ScienceConfig, *, config_path: Path) -> dict[str, Any]:
    paths = {
        "science_runner": Path(__file__),
        "science_config": config_path,
        "repair_runner": ROOT / config.parent.repair_runner,
        "repair_config": ROOT / config.parent.repair_config,
        "native_dgp": ROOT / "src/scpcp/native_signed_gamma.py",
        "canonical_scpcp": ROOT / "src/scpcp/marginal_prefix.py",
        "canonical_baselines": ROOT / "src/scpcp/baselines.py",
        "scores": ROOT / "src/scpcp/scores.py",
        "simulator": ROOT / "src/scpcp/simulator.py",
        "coverage": ROOT / "src/scpcp/coverage/per_step.py",
        "experiment_rng": ROOT / "src/scpcp/experiment.py",
        "preflight_provenance": ROOT / "scripts/run_native_synthetic_signed_gamma_preflight.py",
        "project": ROOT / "pyproject.toml",
    }
    return {
        name: preflight._source_contract(path.resolve(), ROOT)
        for name, path in paths.items()
    }


def _runtime_environment(devices: Sequence[str]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal Native science requires CUDA")
    device_rows = []
    for device in devices:
        index = torch.device(device).index
        if index is None or index >= torch.cuda.device_count():
            raise RuntimeError(f"formal CUDA device is unavailable: {device}")
        properties = torch.cuda.get_device_properties(index)
        device_rows.append(
            {
                "device": device,
                "name": properties.name,
                "total_memory": properties.total_memory,
                "capability": list(properties.major_minor) if hasattr(properties, "major_minor") else [properties.major, properties.minor],
            }
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": device_rows,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _seed_contract(metadata: Mapping[str, Any], config: ScienceConfig) -> dict[str, Any]:
    return {
        "protocol": config.protocol,
        "role": config.role,
        "source_tree_sha256": metadata["source_tree_sha256"],
        "science_config_payload_sha256": metadata["science_config_payload_sha256"],
        "gate_binding_sha256": metadata["gate_binding"]["binding_sha256"],
        "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
        "rng_mapping_sha256": metadata["rng_audit"]["formal_rng_mapping_sha256"],
        "methods": list(METHODS),
        "gammas": list(GAMMAS),
        "budgets": {
            "calibration_trajectories": config.budgets.calibration_trajectories,
            "grid_trajectories": config.budgets.grid_trajectories,
            "reference_trajectories": config.budgets.reference_trajectories,
            "online_trajectories_per_adaptive_method": config.budgets.online_trajectories_per_adaptive_method,
        },
    }


def _artifact_schema() -> dict[str, Any]:
    return {
        "protocol": "native_synthetic_signed_gamma_science_artifact_schema_v1",
        "metadata_fields": sorted(METADATA_FIELDS),
        "seed_fields": sorted(SEED_FIELDS),
        "row_fields": sorted(ROW_FIELDS),
        "method_common_fields": sorted(METHOD_COMMON_FIELDS),
        "method_adaptation_fields": sorted(METHOD_ADAPTATION_FIELDS),
        "method_science_fields": sorted(METHOD_SCIENCE_FIELDS),
        "strict_json": "exact schemas; finite numbers; NaN/Infinity rejected",
        "primary_metric": SCIENCE_CONTRACT["coverage_metric"],
        "signed_curve_interpretation": SCIENCE_CONTRACT["signed_curve_interpretation"],
    }


def _prepare_root(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    schema: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    resume: bool,
) -> None:
    _validate_metadata(metadata, schema=schema)
    if resume:
        if not root.is_dir():
            raise FileNotFoundError("resume requires an existing Native science root")
        if _read_json(root / "metadata.json") != metadata:
            raise RuntimeError("resume metadata differs from the live launch contract")
        if _read_json(root / "artifact_schema.json") != schema:
            raise RuntimeError("resume artifact schema differs")
        preflight._verify_source_snapshot(root, metadata["source_snapshot"])
        observed = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        allowed = _allowed_partial_paths(metadata)
        unexpected = sorted(observed - allowed)
        if unexpected:
            raise RuntimeError(f"unexpected Native science resume artifacts: {unexpected}")
        return
    if root.exists():
        raise FileExistsError(f"fresh Native science output already exists: {root}")
    root.mkdir(parents=True)
    (root / "seeds").mkdir()
    contract = source_snapshot["contract"]
    _atomic_write(root / contract["archive_path"], source_snapshot["archive_bytes"])
    _atomic_write(root / contract["manifest_path"], source_snapshot["manifest_bytes"])
    _write_json(root / "artifact_schema.json", schema)
    _write_json(root / "metadata.json", metadata)
    preflight._verify_source_snapshot(root, metadata["source_snapshot"])


def _validate_metadata(metadata: Mapping[str, Any], *, schema: Mapping[str, Any]) -> None:
    _require_exact_keys(metadata, METADATA_FIELDS, "science metadata")
    config = ScienceConfig.from_yaml(DEFAULT_CONFIG)
    if metadata["protocol"] != config.protocol or metadata["role"] != config.role:
        raise RuntimeError("science metadata protocol or role differs")
    if metadata["science_config"] != config.to_dict():
        raise RuntimeError("science metadata config payload differs")
    if metadata["science_config_payload_sha256"] != _canonical_sha256(config.to_dict()):
        raise RuntimeError("science config payload hash differs")
    config_path = ROOT / str(metadata["science_config_path"])
    if (
        config_path.resolve() != DEFAULT_CONFIG.resolve()
        or metadata["science_config_file_sha256"] != _file_sha256(config_path)
    ):
        raise RuntimeError("science config file binding differs")
    if metadata["science_contract"] != SCIENCE_CONTRACT:
        raise RuntimeError("science reporting contract differs")
    gate = GateBinding(**metadata["gate_binding"])
    core = asdict(gate)
    binding_hash = core.pop("binding_sha256")
    if binding_hash != _canonical_sha256(core):
        raise RuntimeError("repair gate binding hash differs")
    validate_science_rng_audit(config, gate_binding=gate, audit=metadata["rng_audit"])
    expected_devices = seed_device_mapping(config)
    if (
        metadata["seed_device_mapping"] != expected_devices
        or metadata["seed_device_mapping_sha256"] != _canonical_sha256(expected_devices)
    ):
        raise RuntimeError("science seed-device binding differs")
    if metadata["environment_sha256"] != _canonical_sha256(metadata["environment"]):
        raise RuntimeError("science environment hash differs")
    if metadata["canonical_invocation"] != ["scripts/run_native_synthetic_signed_gamma_science.py"]:
        raise RuntimeError("science canonical invocation differs")
    if metadata["canonical_invocation_sha256"] != _canonical_sha256(metadata["canonical_invocation"]):
        raise RuntimeError("science invocation hash differs")
    if metadata["artifact_schema_sha256"] != _canonical_sha256(schema):
        raise RuntimeError("science artifact-schema hash differs")
    launch_core = {
        "science_config_payload_sha256": metadata["science_config_payload_sha256"],
        "gate_binding_sha256": gate.binding_sha256,
        "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
        "seed_device_mapping_sha256": metadata["seed_device_mapping_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "environment_sha256": metadata["environment_sha256"],
        "canonical_invocation_sha256": metadata["canonical_invocation_sha256"],
        "artifact_schema_sha256": metadata["artifact_schema_sha256"],
        "dependency_files_sha256": _canonical_sha256(metadata["dependency_files"]),
    }
    if metadata["launch_contract_sha256"] != _canonical_sha256(launch_core):
        raise RuntimeError("science launch-contract hash differs")


def _load_seed_payloads(
    root: Path,
    *,
    config: ScienceConfig,
    metadata: Mapping[str, Any],
    seed_contract: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    seed_dir = root / "seeds"
    expected_names = {f"seed_{seed}.json" for seed in config.rng.base_seeds}
    observed_names = {path.name for path in seed_dir.glob("seed_*.json")}
    unexpected = sorted(observed_names - expected_names)
    if unexpected:
        raise RuntimeError(f"unexpected Native science seed artifacts: {unexpected}")
    payloads = {}
    for seed in config.rng.base_seeds:
        path = seed_dir / f"seed_{seed}.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        _validate_seed_payload(
            payload,
            config=config,
            metadata=metadata,
            expected_seed=seed,
            expected_device=metadata["seed_device_mapping"][str(seed)],
        )
        for key, value in seed_contract.items():
            if payload[key] != value:
                raise RuntimeError(f"Native science seed {seed} contract differs at {key}")
        payloads[seed] = payload
    return payloads


def _validate_seed_payload(
    payload: Mapping[str, Any],
    *,
    config: ScienceConfig,
    metadata: Mapping[str, Any],
    expected_seed: int,
    expected_device: str,
) -> None:
    _require_exact_keys(payload, SEED_FIELDS, "science seed payload")
    expected_contract = _seed_contract(metadata, config)
    for key, value in expected_contract.items():
        if payload[key] != value:
            raise RuntimeError(f"science seed contract differs at {key}")
    if payload["seed"] != expected_seed or payload["device"] != expected_device:
        raise RuntimeError("science seed identity or device differs")
    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != len(GAMMAS):
        raise RuntimeError("science seed must contain five signed-gamma rows")
    for row, gamma in zip(rows, GAMMAS, strict=True):
        _validate_science_row(
            row,
            config=config,
            expected_seed=expected_seed,
            expected_gamma=gamma,
        )
    _require_finite_json(payload, "science seed payload")


def _validate_science_row(
    row: Mapping[str, Any],
    *,
    config: ScienceConfig,
    expected_seed: int,
    expected_gamma: float,
) -> None:
    _require_exact_keys(row, ROW_FIELDS, "signed-gamma row")
    expected_role = (
        "primary_confirmatory_method_comparison"
        if expected_gamma == PRIMARY_GAMMA
        else "descriptive_signed_mechanism_curve_no_ranking_or_superiority"
    )
    if (
        row["seed"] != expected_seed
        or float(row["gamma"]) != expected_gamma
        or row["gamma_role"] != expected_role
        or set(row["methods"]) != set(METHODS)
        or row["adaptation_seeds"] != adaptation_seeds(expected_seed)
    ):
        raise RuntimeError("signed-gamma row identity or method set differs")
    expected_kernel = NativeSignedGammaKernel(
        NativeSignedGammaDGPConfig(policy_ratio_cap=config.design.policy_ratio_cap),
        expected_gamma,
    ).fingerprint
    if row["kernel_fingerprint"] != expected_kernel:
        raise RuntimeError("signed-gamma kernel fingerprint differs")
    for name in ("scpcp_minimum_ess_fraction", "scpcp_minimum_candidate_ess_fraction"):
        value = row[name]
        if value is not None and (not _is_finite_number(value) or not 0.0 < float(value) <= 1.0):
            raise RuntimeError(f"invalid {name}")
    if not isinstance(row["scpcp_selected_endpoint"], bool):
        raise RuntimeError("SC-PCP selected-endpoint flag must be Boolean")
    failure_stage = row["scpcp_failure_stage"]
    if failure_stage is not None and (not isinstance(failure_stage, int) or not 0 <= failure_stage < config.design.horizon):
        raise RuntimeError("SC-PCP failure stage is invalid")
    for method in METHODS:
        _validate_method_row(method, row["methods"][method], config=config)
    scpcp_available = row["methods"]["SC-PCP"]["selection_available"]
    if scpcp_available != (failure_stage is None):
        raise RuntimeError("SC-PCP availability and failure stage disagree")


def _validate_method_row(
    method: str,
    row: Mapping[str, Any],
    *,
    config: ScienceConfig,
) -> None:
    if not isinstance(row, dict):
        raise RuntimeError(f"{method} row must be a mapping")
    available = row.get("selection_available")
    if not isinstance(available, bool):
        raise RuntimeError(f"{method} selection flag must be Boolean")
    expected_fields = set(METHOD_COMMON_FIELDS)
    if method in ADAPTIVE_METHODS:
        expected_fields |= METHOD_ADAPTATION_FIELDS
    if available:
        expected_fields |= METHOD_SCIENCE_FIELDS
    _require_exact_keys(row, expected_fields, f"{method} row")
    if method in ("Standard CP", *ADAPTIVE_METHODS) and not available:
        raise RuntimeError(f"{method} must always return a schedule")
    if (
        not isinstance(row["selection_status"], str)
        or not row["selection_status"]
        or row["information_regime"] != INFORMATION_REGIME[method]
        or row["target_adaptation_trajectories"] != TARGET_ADAPTATION_BUDGET[method]
    ):
        raise RuntimeError(f"{method} status, information regime, or budget differs")
    horizon = config.design.horizon
    radii = row["radii"]
    if not isinstance(radii, list) or len(radii) != (horizon if available else 0):
        raise RuntimeError(f"{method} radius vector length differs")
    if available and any(not _is_finite_number(value) or float(value) <= 0.0 for value in radii):
        raise RuntimeError(f"{method} radii must be finite and positive")
    if method in ADAPTIVE_METHODS:
        if (
            row["adaptation_rounds"] != config.design.online_rounds
            or len(row["adaptation_per_time_coverage"]) != horizon
            or len(row["adaptation_round_worst_coverage"]) != config.design.online_rounds
            or not 0.0 <= float(row["adaptation_pathwise_coverage"]) <= 1.0
        ):
            raise RuntimeError(f"{method} adaptation diagnostics differ")
        for values in (row["adaptation_per_time_coverage"], row["adaptation_round_worst_coverage"]):
            if any(not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0 for value in values):
                raise RuntimeError(f"{method} adaptation coverage is invalid")
        selected_scale = row["selected_scale"]
        if method == "PRC":
            if not _is_finite_number(selected_scale) or float(selected_scale) <= 0.0:
                raise RuntimeError("PRC selected scale is invalid")
        elif selected_scale is not None:
            raise RuntimeError(f"{method} selected scale must be null")
    if not available:
        return
    bounded_vectors = (
        "source_coverage",
        "target_coverage",
        "prefix_ess_fraction",
        "maximum_normalized_weight_share",
        "policy_tv_on_source_states",
        "source_difficulty_by_stage",
        "target_difficulty_by_stage",
        "source_tail_by_stage",
        "target_tail_by_stage",
    )
    for name in bounded_vectors:
        values = row[name]
        if (
            not isinstance(values, list)
            or len(values) != horizon
            or any(not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0 for value in values)
        ):
            raise RuntimeError(f"{method} {name} is invalid")
    for name in ("coverage_gap",):
        values = row[name]
        if (
            not isinstance(values, list)
            or len(values) != horizon
            or any(not _is_finite_number(value) or not -1.0 <= float(value) <= 1.0 for value in values)
        ):
            raise RuntimeError(f"{method} {name} is invalid")
    for name, strictly_positive in (
        ("target_normalized_width", True),
        ("raw_log_weight_span", False),
    ):
        values = row[name]
        if not isinstance(values, list) or len(values) != horizon:
            raise RuntimeError(f"{method} {name} length differs")
        if any(
            not _is_finite_number(value)
            or (float(value) <= 0.0 if strictly_positive else float(value) < 0.0)
            for value in values
        ):
            raise RuntimeError(f"{method} {name} is invalid")


def _ensure_bootstrap_artifacts(
    root: Path,
    config: ScienceConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    uniform_path = root / "bootstrap_uniforms.npy"
    index_path = root / "bootstrap_indices.npy"
    if uniform_path.exists() != index_path.exists():
        raise RuntimeError("bootstrap artifact pair is incomplete")
    expected_uniforms, expected_indices = _make_bootstrap_arrays(config)
    if not uniform_path.exists():
        _write_npy(uniform_path, expected_uniforms)
        _write_npy(index_path, expected_indices)
    uniforms, indices = _read_bootstrap_artifacts(root, config)
    if not np.array_equal(uniforms, expected_uniforms) or not np.array_equal(indices, expected_indices):
        raise RuntimeError("bootstrap arrays differ from the frozen complete-seed bank")
    return uniforms, indices, _bootstrap_contract(
        config,
        uniform_path=uniform_path,
        index_path=index_path,
    )


def _read_bootstrap_artifacts(
    root: Path,
    config: ScienceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    uniform_path = root / "bootstrap_uniforms.npy"
    index_path = root / "bootstrap_indices.npy"
    if not uniform_path.is_file() or not index_path.is_file():
        raise RuntimeError("completed Native science root lacks bootstrap arrays")
    try:
        uniforms = np.load(uniform_path, allow_pickle=False)
        indices = np.load(index_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RuntimeError("cannot load Native science bootstrap arrays") from error
    _validate_bootstrap_uniforms(uniforms, config)
    expected_shape = (config.budgets.bootstrap_resamples, len(config.rng.base_seeds))
    if (
        indices.shape != expected_shape
        or indices.dtype != np.int64
        or not np.array_equal(indices, _bootstrap_indices(uniforms, len(config.rng.base_seeds)))
    ):
        raise RuntimeError("complete-seed bootstrap index matrix differs")
    return uniforms, indices


def _coverage_audit(
    payloads: Mapping[int, Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    config: ScienceConfig,
    bootstrap_uniforms: np.ndarray,
    bootstrap_contract: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        row
        for seed in config.rng.base_seeds
        for row in payloads[seed]["rows"]
    ]
    reconstructed = summarize(
        rows,
        config=config,
        bootstrap_uniforms=bootstrap_uniforms,
        bootstrap_contract=bootstrap_contract,
    )
    if reconstructed != summary:
        raise RuntimeError("coverage audit cannot reconstruct summary from raw rows")
    raw_hashes = {
        str(seed): _canonical_sha256(payloads[seed]) for seed in config.rng.base_seeds
    }
    return {
        "protocol": config.protocol,
        "status": "PASSED",
        "raw_seed_count": len(payloads),
        "raw_signed_gamma_row_count": len(rows),
        "raw_seed_payload_sha256": raw_hashes,
        "raw_seed_payload_set_sha256": _canonical_sha256(raw_hashes),
        "summary_payload_sha256": _canonical_sha256(summary),
        "reconstructed_summary_payload_sha256": _canonical_sha256(reconstructed),
        "primary_metric_formula": SCIENCE_CONTRACT["coverage_metric"],
        "mean_coverage_formula": SCIENCE_CONTRACT["mean_coverage_metric"],
        "full_stage_vectors_present": True,
        "selection_denominator_all_prespecified_seeds": True,
        "bootstrap_complete_seed_matrix": True,
        "primary_gamma_only_paired_comparisons": True,
        "other_signed_gammas_descriptive_only": True,
    }


def _finalize_root(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    config: ScienceConfig,
) -> None:
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = _expected_artifact_paths(metadata, config)
    if observed not in (expected, expected | {"manifest.json"}):
        raise RuntimeError("Native science pre-finalization artifact set differs")
    if "manifest.json" in observed:
        _verify_manifest(root, metadata=metadata, config=config)
    _write_manifest(root, metadata=metadata, config=config)
    _validate_bundle_contents(
        root,
        expected_metadata=metadata,
        source_root=ROOT,
        include_complete=False,
    )
    complete = _expected_complete_payload(root, metadata=metadata)
    complete_path = root / "COMPLETE"
    try:
        _write_json(complete_path, complete)
        validate_completed_bundle(root, expected_metadata=metadata)
    except BaseException:
        complete_path.unlink(missing_ok=True)
        _fsync_directory(root)
        raise


def validate_completed_bundle(
    root: Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    source_root: Path = ROOT,
) -> None:
    _validate_bundle_contents(
        root,
        expected_metadata=expected_metadata,
        source_root=source_root,
        include_complete=True,
    )


def _validate_bundle_contents(
    root: Path,
    *,
    expected_metadata: Mapping[str, Any] | None,
    source_root: Path,
    include_complete: bool,
) -> tuple[dict[str, Any], str]:
    if source_root.resolve() != ROOT.resolve():
        raise RuntimeError("Native science validator requires the active project source root")
    config_path = source_root / DEFAULT_CONFIG.relative_to(ROOT)
    config = ScienceConfig.from_yaml(config_path)
    expected_root = (source_root / config.output_root).resolve()
    if root.resolve() != expected_root:
        raise RuntimeError("completed Native science root is not the frozen path")
    metadata = _read_json(root / "metadata.json")
    schema = _read_json(root / "artifact_schema.json")
    if schema != _artifact_schema():
        raise RuntimeError("completed Native science artifact schema differs")
    _validate_metadata(metadata, schema=schema)
    if expected_metadata is not None and dict(metadata) != dict(expected_metadata):
        raise RuntimeError("completed Native science metadata differs from launch")

    gate = verify_repair_gate(config, source_root=source_root)
    if metadata["gate_binding"] != asdict(gate):
        raise RuntimeError("completed Native science gate binding differs from live repair")
    if preflight._experiment_tree_sha256(source_root) != metadata["source_tree_sha256"]:
        raise RuntimeError("active source tree differs from completed Native science")
    preflight._verify_source_snapshot(root, metadata["source_snapshot"])
    preflight._verify_dependency_files(metadata["dependency_files"], source_root)
    live_rng = audit_science_rng_ids(
        config,
        gate_binding=gate,
        output_root=root,
        artifact_root=source_root / "results",
        source_root=source_root,
        config_path=config_path,
    )
    if metadata["rng_audit"] != live_rng:
        raise RuntimeError("completed Native science RNG inventory differs from live scan")
    rebuilt_metadata = _build_metadata(
        config,
        config_path=config_path,
        gate=gate,
        rng_audit=live_rng,
        source_hash=metadata["source_tree_sha256"],
        source_snapshot=metadata["source_snapshot"],
        schema=schema,
    )
    if metadata != rebuilt_metadata:
        raise RuntimeError("completed Native science metadata does not rebuild exactly")

    seed_contract = _seed_contract(metadata, config)
    payloads = _load_seed_payloads(
        root,
        config=config,
        metadata=metadata,
        seed_contract=seed_contract,
    )
    if set(payloads) != set(config.rng.base_seeds):
        raise RuntimeError("completed Native science root lacks one or more seed vectors")
    uniforms, indices = _read_bootstrap_artifacts(root, config)
    expected_uniforms, expected_indices = _make_bootstrap_arrays(config)
    if not np.array_equal(uniforms, expected_uniforms) or not np.array_equal(indices, expected_indices):
        raise RuntimeError("completed bootstrap bank differs from frozen seed")
    bootstrap = _bootstrap_contract(
        config,
        uniform_path=root / "bootstrap_uniforms.npy",
        index_path=root / "bootstrap_indices.npy",
    )
    rows = [
        row
        for seed in config.rng.base_seeds
        for row in payloads[seed]["rows"]
    ]
    summary = _read_json(root / "summary.json")
    expected_summary = summarize(
        rows,
        config=config,
        bootstrap_uniforms=uniforms,
        bootstrap_contract=bootstrap,
    )
    if summary != expected_summary:
        raise RuntimeError("completed Native science summary does not reconcile")
    audit = _read_json(root / "coverage_audit.json")
    expected_audit = _coverage_audit(
        payloads,
        summary=summary,
        config=config,
        bootstrap_uniforms=uniforms,
        bootstrap_contract=bootstrap,
    )
    if audit != expected_audit:
        raise RuntimeError("completed Native science coverage audit differs")
    expected_status = {
        "protocol": config.protocol,
        "status": "COMPLETE",
        "decision": "SCIENCE_COMPLETE",
        "primary_gamma": PRIMARY_GAMMA,
        "primary_metric": SCIENCE_CONTRACT["coverage_metric"],
        "signed_curve_interpretation": SCIENCE_CONTRACT["signed_curve_interpretation"],
        "n_seeds": len(config.rng.base_seeds),
        "n_signed_gamma_rows": len(rows),
        "methods": list(METHODS),
        "gate_binding_sha256": gate.binding_sha256,
        "coverage_audit_sha256": _file_sha256(root / "coverage_audit.json"),
    }
    if _read_json(root / "FINAL_STATUS.json") != expected_status:
        raise RuntimeError("completed Native science final status differs")
    _require_complete_artifact_set(
        root,
        metadata=metadata,
        config=config,
        include_complete=include_complete,
    )
    manifest_hash = _verify_manifest(root, metadata=metadata, config=config)
    if include_complete:
        complete = _read_json(root / "COMPLETE")
        expected_complete = _expected_complete_payload(root, metadata=metadata)
        if complete != expected_complete or complete["manifest_sha256"] != manifest_hash:
            raise RuntimeError("completed Native science hash chain differs")
    return metadata, manifest_hash


def _expected_complete_payload(
    root: Path,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": metadata["protocol"],
        "status": "complete",
        "decision": "SCIENCE_COMPLETE",
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "manifest_bytes": (root / "manifest.json").stat().st_size,
        "metadata_sha256": _file_sha256(root / "metadata.json"),
        "summary_sha256": _file_sha256(root / "summary.json"),
        "coverage_audit_sha256": _file_sha256(root / "coverage_audit.json"),
        "final_status_sha256": _file_sha256(root / "FINAL_STATUS.json"),
        "artifact_schema_sha256": _file_sha256(root / "artifact_schema.json"),
        "science_config_payload_sha256": metadata["science_config_payload_sha256"],
        "gate_binding_sha256": metadata["gate_binding"]["binding_sha256"],
        "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
        "rng_mapping_sha256": metadata["rng_audit"]["formal_rng_mapping_sha256"],
        "seed_device_mapping_sha256": metadata["seed_device_mapping_sha256"],
        "source_snapshot_sha256": metadata["source_snapshot"]["archive_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "launch_contract_sha256": metadata["launch_contract_sha256"],
        "bootstrap_uniforms_sha256": _file_sha256(root / "bootstrap_uniforms.npy"),
        "bootstrap_indices_sha256": _file_sha256(root / "bootstrap_indices.npy"),
    }


def _write_manifest(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    config: ScienceConfig,
) -> None:
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _iter_bundle_artifacts(root)
    ]
    _write_json(
        root / "manifest.json",
        {
            "protocol": metadata["protocol"],
            "science_config_payload_sha256": metadata["science_config_payload_sha256"],
            "gate_binding_sha256": metadata["gate_binding"]["binding_sha256"],
            "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
            "source_tree_sha256": metadata["source_tree_sha256"],
            "artifact_count": len(records),
            "expected_seed_count": len(config.rng.base_seeds),
            "artifacts": records,
        },
    )


def _verify_manifest(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    config: ScienceConfig,
) -> str:
    manifest = _read_json(root / "manifest.json")
    _require_exact_keys(
        manifest,
        {
            "protocol",
            "science_config_payload_sha256",
            "gate_binding_sha256",
            "rng_audit_sha256",
            "source_tree_sha256",
            "artifact_count",
            "expected_seed_count",
            "artifacts",
        },
        "science manifest",
    )
    expected_header = {
        "protocol": metadata["protocol"],
        "science_config_payload_sha256": metadata["science_config_payload_sha256"],
        "gate_binding_sha256": metadata["gate_binding"]["binding_sha256"],
        "rng_audit_sha256": metadata["rng_audit"]["audit_sha256"],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "expected_seed_count": len(config.rng.base_seeds),
    }
    if any(manifest[key] != value for key, value in expected_header.items()):
        raise RuntimeError("science manifest header differs")
    records = manifest["artifacts"]
    if not isinstance(records, list) or manifest["artifact_count"] != len(records):
        raise RuntimeError("science manifest records are malformed")
    observed_paths = {path.relative_to(root).as_posix() for path in _iter_bundle_artifacts(root)}
    listed_paths = set()
    for record in records:
        _require_exact_keys(record, {"path", "sha256", "size_bytes"}, "manifest record")
        relative = record["path"]
        if not isinstance(relative, str) or relative in listed_paths:
            raise RuntimeError("science manifest has a duplicate or malformed path")
        listed_paths.add(relative)
        path = (root / relative).resolve()
        if not _is_relative_to(path, root.resolve()):
            raise RuntimeError("science manifest path escapes root")
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"science manifest artifact differs: {relative}")
    if listed_paths != observed_paths:
        raise RuntimeError("science manifest file set differs")
    return _file_sha256(root / "manifest.json")


def _iter_bundle_artifacts(root: Path) -> list[Path]:
    paths = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "COMPLETE"}:
            continue
        if ".tmp-" in path.name:
            raise RuntimeError(f"temporary Native science artifact remains: {path}")
        paths.append(path)
    return paths


def _expected_artifact_paths(
    metadata: Mapping[str, Any],
    config: ScienceConfig,
) -> set[str]:
    return {
        "artifact_schema.json",
        "metadata.json",
        "bootstrap_uniforms.npy",
        "bootstrap_indices.npy",
        "summary.json",
        "coverage_audit.json",
        "FINAL_STATUS.json",
        str(metadata["source_snapshot"]["archive_path"]),
        str(metadata["source_snapshot"]["manifest_path"]),
        *(f"seeds/seed_{seed}.json" for seed in config.rng.base_seeds),
    }


def _allowed_partial_paths(metadata: Mapping[str, Any]) -> set[str]:
    config = ScienceConfig.from_yaml(DEFAULT_CONFIG)
    return _expected_artifact_paths(metadata, config) | {"manifest.json", "COMPLETE"}


def _require_complete_artifact_set(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    config: ScienceConfig,
    include_complete: bool,
) -> None:
    expected = _expected_artifact_paths(metadata, config) | {"manifest.json"}
    if include_complete:
        expected.add("COMPLETE")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise RuntimeError(
            "Native science artifact set differs: "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )


def _dataclass_from_mapping(cls: type[Any], value: object, label: str) -> Any:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    expected = {field.name for field in fields(cls)}
    _require_exact_keys(value, expected, label)
    return cls(**value)


def _require_exact_keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise RuntimeError(
            f"{label} fields differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _require_finite_json(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require_finite_json(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _require_finite_json(nested, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{label} contains a non-finite number")


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"cannot parse strict JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact root must be a mapping: {path}")
    _require_finite_json(value, path.name)
    return value


def _write_json(path: Path, value: object) -> None:
    _require_finite_json(value, path.name)
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def _write_npy(path: Path, value: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    _atomic_write(path, buffer.getvalue())


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _integer_set_sha256(values: Iterable[int]) -> str:
    return _canonical_sha256(sorted(set(int(value) for value in values)))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
