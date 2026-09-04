"""Strict bridge to the official Performative Risk Control (PRC) release.

``rcpp.main.run_trajectory`` owns every PRC update, confidence width, binary
search, and stopping rule.  This adapter contributes only an SC-PCP data
source and a declared scalar loss: a trajectory is lost when any stage is not
covered by its profile-scaled radius.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import itertools
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable, Protocol

import numpy as np
import torch
from torch import Tensor

from data import TrajectoryBatch
from scores import score_batch
from simulator import rollout


OFFICIAL_PRC_REPOSITORY = "https://github.com/livctr/rcpp"
OFFICIAL_PRC_COMMIT = "b11d3964f42622f2e67a8a584a8684108820a1f4"
OFFICIAL_PRC_TAG = "neurips2025"
OFFICIAL_PRC_LICENSE = "MIT"
PRC_DEPLOYMENT_SEED_STRIDE = 61_103
_SOURCE_SHA256 = {
    "LICENSE": "9fbfa5da2c49d115c7bb06aef639cabce31fd1a4946faffad8c03cbb5abef600",
    "rcpp/main.py": "77d4b6b86d4b6dbf4bc72ec47c25a0f96f9aca63cd86396ff61db6960d623e36",
    "rcpp/risk_measure.py": "269fc11f1235769d1070611285b7b6dd79e9f5a0027171e173ed29f361db0401",
    "rcpp/width_calculator.py": "974d48fddcd65ece6c3851a382d96096a7f0e38f617a9f7724117de64201168c",
}


class PRCRollout(Protocol):
    def __call__(
        self,
        environment: object,
        policy: object,
        *,
        n: int,
        horizon: int,
        seed: int,
        device: str | torch.device,
        q: Tensor,
    ) -> TrajectoryBatch: ...


@dataclass(frozen=True)
class NativePRCConfig:
    """Inputs consumed by the original PRC trajectory routine."""

    alpha: float
    delta: float
    tightness: float
    tau: float
    cohort_size: int

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0 or not 0.0 < self.delta < 1.0:
            raise ValueError("PRC alpha and delta must lie in (0, 1)")
        if not 0.0 < self.tightness < self.alpha:
            raise ValueError("PRC tightness must lie in (0, alpha)")
        if self.tau <= 0.0 or self.cohort_size < 2:
            raise ValueError("PRC tau must be positive and cohort_size at least two")


@dataclass(frozen=True)
class NativePRCResult:
    radius_by_time: Tensor
    selected_scale: float
    normalized_lambda: float
    target_deployments: int
    rounds: int
    optimization_rounds: int
    adaptation_per_time_coverage: Tensor
    adaptation_round_worst_coverage: tuple[float, ...]
    adaptation_pathwise_coverage: float
    lambda_path: tuple[float, ...]
    deployed_scale_path: tuple[float, ...]
    empirical_deployed_risk_path: tuple[float, ...]
    empirical_candidate_risk_path: tuple[float, ...]
    guaranteed_iterations: int
    delta_lambda: float
    alpha: float
    delta: float
    tightness: float
    tau: float
    cohort_size: int
    scale_minimum: float
    scale_maximum: float
    upstream_repository: str = OFFICIAL_PRC_REPOSITORY
    upstream_commit: str = OFFICIAL_PRC_COMMIT
    upstream_tag: str = OFFICIAL_PRC_TAG
    upstream_license: str = OFFICIAL_PRC_LICENSE
    upstream_entrypoint: str = "rcpp.main.run_trajectory"
    risk_definition: str = "mean_trajectory_wise_any_stage_miscoverage"


def official_prc_root() -> Path:
    return Path(__file__).resolve().parents[1] / "internal" / "baselines" / "Performative Risk Control"


def verify_official_prc_source(upstream_root: Path | None = None) -> Path:
    """Verify the exact official files before the adapter imports them."""

    root = (official_prc_root() if upstream_root is None else upstream_root).resolve()
    for relative, expected in _SOURCE_SHA256.items():
        file = root / relative
        if not file.is_file():
            raise FileNotFoundError(f"official PRC source is missing: {file}")
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"official PRC source differs from pinned commit {OFFICIAL_PRC_COMMIT}: {file}")
    return root


def _load_official_prc(root: Path) -> tuple[Callable[..., object], type, type]:
    root_text = str(root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        main = importlib.import_module("rcpp.main")
        widths = importlib.import_module("rcpp.width_calculator")
        risks = importlib.import_module("rcpp.risk_measure")
    finally:
        if inserted:
            sys.path.remove(root_text)
    for module in (main, widths, risks):
        _require_module_from_root(module, root)
    return main.run_trajectory, widths.CLTWidth, risks.MeanRiskMeasure


def _require_module_from_root(module: ModuleType, root: Path) -> None:
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f"imported {module.__name__} outside the verified PRC checkout")


class _ProfilePathwiseLoss:
    def __init__(self, profile: np.ndarray, scale_min: float, scale_max: float) -> None:
        self.profile, self.scale_min, self.scale_range = profile, scale_min, scale_max - scale_min

    def scale(self, normalized_lambda: float) -> float:
        return self.scale_min + float(normalized_lambda) * self.scale_range

    def calc_loss(self, shifted_data: list[np.ndarray], normalized_lambda: float, do_new_sample: bool = True) -> np.ndarray:
        del do_new_sample
        scores = shifted_data[0]
        return np.any(scores > self.scale(normalized_lambda) * self.profile[None, :], axis=1).astype(np.float64)


class _TrajectorySimulator:
    """Presents independently split target cohorts to the original PRC code."""

    def __init__(
        self,
        environment: object,
        policy: object,
        outcome_model: object,
        *,
        loss: _ProfilePathwiseLoss,
        profile: Tensor,
        config: NativePRCConfig,
        horizon: int,
        seed: int,
        device: str | torch.device,
        rollout_fn: PRCRollout,
    ) -> None:
        self.environment, self.policy, self.outcome_model = environment, policy, outcome_model
        self.loss, self.profile, self.config = loss, profile.detach().cpu(), config
        self.horizon, self.seed, self.device, self.rollout_fn = horizon, seed, device, rollout_fn
        self.reset()

    def reset(self) -> None:
        self.covered = torch.zeros(self.horizon, dtype=torch.float64)
        self.n = 0
        self.pathwise_hits = 0
        self.worst: list[float] = []
        self.scales: list[float] = []

    def simulate_shift(self, base_data: object, normalized_lambda: float) -> list[np.ndarray]:
        del base_data
        scale = self.loss.scale(normalized_lambda)
        radii = scale * self.profile
        index = len(self.scales)
        # The first N patients are the original PRC selection cohort and the
        # second N are its tracking cohort.  They are never reused as both.
        n = 2 * self.config.cohort_size
        batch = self.rollout_fn(
            self.environment, self.policy, n=n, horizon=self.horizon,
            seed=self.seed + PRC_DEPLOYMENT_SEED_STRIDE * index, device=self.device,
            q=radii.to(self.device),
        )
        scores = score_batch(self.outcome_model, batch.current_states(), batch.actions, batch.outcomes).detach().cpu()
        if scores.shape != (n, self.horizon):
            raise RuntimeError("PRC score function must return [2*cohort_size, horizon]")
        hits = scores <= radii[None, :]
        self.covered += hits.to(torch.float64).sum(dim=0)
        self.n += n
        self.pathwise_hits += int(hits.all(dim=1).sum().item())
        self.worst.append(float(hits.to(torch.float64).mean(dim=0).min().item()))
        self.scales.append(scale)
        return [scores.numpy()]


def _independent_halves(data: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    scores = data[0]
    if len(scores) % 2:
        raise RuntimeError("PRC adapter requires an even target cohort")
    middle = len(scores) // 2
    return [scores[:middle]], [scores[middle:]]


@torch.no_grad()
def native_prc_profile_scale(
    environment: object,
    policy: object,
    outcome_model: object,
    scale_grid: Tensor,
    stage_profile: Tensor,
    *,
    config: NativePRCConfig,
    horizon: int,
    seed: int,
    device: str | torch.device,
    rollout_fn: PRCRollout | None = None,
    upstream_root: Path | None = None,
) -> NativePRCResult:
    """Run unmodified PRC with an explicit profile-family loss interface."""

    grid = scale_grid.detach().cpu().to(torch.float64)
    profile = stage_profile.detach().cpu().to(torch.float64)
    if grid.ndim != 1 or len(grid) < 2 or not bool(torch.all(grid[1:] > grid[:-1])):
        raise ValueError("PRC scale_grid must be a strictly increasing vector")
    if not bool(torch.isfinite(grid).all()) or float(grid[0]) <= 0.0:
        raise ValueError("PRC scale_grid must be finite and positive")
    if profile.shape != (horizon,) or not bool(torch.all(profile > 0)):
        raise ValueError("PRC stage_profile must be positive with shape [horizon]")
    root = verify_official_prc_source(upstream_root)
    run_trajectory, clt_width, mean_risk = _load_official_prc(root)
    loss = _ProfilePathwiseLoss(profile.numpy(), float(grid[0]), float(grid[-1]))
    simulator = _TrajectorySimulator(
        environment, policy, outcome_model, loss=loss, profile=profile, config=config,
        horizon=horizon, seed=seed, device=device, rollout_fn=rollout if rollout_fn is None else rollout_fn,
    )
    trajectory = run_trajectory(
        itertools.repeat(None), _independent_halves, clt_width(config.alpha, loss_max=1.0),
        mean_risk(), simulator, loss,
        SimpleNamespace(alpha=config.alpha, delta=config.delta, tightness=config.tightness,
                        tau=config.tau, N=config.cohort_size, lambda_min=0.0, lambda_safe=1.0),
    )
    normalized_lambda = float(np.asarray(trajectory.lambda_hat).item())
    selected_scale = loss.scale(normalized_lambda)
    if simulator.n == 0:
        raise RuntimeError("official PRC produced no target deployment")
    return NativePRCResult(
        radius_by_time=(selected_scale * profile).to(stage_profile), selected_scale=selected_scale,
        normalized_lambda=normalized_lambda, target_deployments=simulator.n, rounds=len(simulator.scales),
        optimization_rounds=len(trajectory.lambdas) - 1,
        adaptation_per_time_coverage=(simulator.covered / simulator.n).to(torch.float32),
        adaptation_round_worst_coverage=tuple(simulator.worst), adaptation_pathwise_coverage=simulator.pathwise_hits / simulator.n,
        lambda_path=tuple(float(x) for x in trajectory.lambdas), deployed_scale_path=tuple(simulator.scales),
        empirical_deployed_risk_path=tuple(float(x) for x in trajectory.risks_tt),
        empirical_candidate_risk_path=tuple(float(x) for x in trajectory.risks_tm1_t),
        guaranteed_iterations=int(trajectory.guaranteed_T), delta_lambda=float(trajectory.delta_lambda),
        alpha=config.alpha, delta=config.delta, tightness=config.tightness, tau=config.tau,
        cohort_size=config.cohort_size, scale_minimum=float(grid[0]), scale_maximum=float(grid[-1]),
    )
