"""Thin task adapter for the vendored official Performative Risk Control code.

The optimization loop is executed by ``rcpp.main.run_trajectory`` from the
vendored NeurIPS 2025 release.  This module only maps SC-PCP trajectories to
the upstream simulator/loss interfaces and maps the upstream scalar
``lambda`` to the profile family ``q_t(lambda)``.
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable, Protocol

import numpy as np
import torch
from torch import Tensor

from scpcp.data import TrajectoryBatch


OFFICIAL_PRC_REPOSITORY = "https://github.com/livctr/rcpp"
OFFICIAL_PRC_COMMIT = "b11d3964f42622f2e67a8a584a8684108820a1f4"
OFFICIAL_PRC_TAG = "neurips2025"
OFFICIAL_PRC_LICENSE = "MIT"
PRC_DEPLOYMENT_SEED_STRIDE = 61_103

_OFFICIAL_SOURCE_SHA256 = {
    "LICENSE": "9fbfa5da2c49d115c7bb06aef639cabce31fd1a4946faffad8c03cbb5abef600",
    "rcpp/loss_simulator.py": "2039c6f6b03e0780402c69ed328e79411303fbdf9f1b671d789f1be6d06241a4",
    "rcpp/main.py": "77d4b6b86d4b6dbf4bc72ec47c25a0f96f9aca63cd86396ff61db6960d623e36",
    "rcpp/performativity_simulator.py": "64032d1a506716e4094bab554fb2335cb885b0eba8e381d662d1158ce2f832a8",
    "rcpp/risk_measure.py": "269fc11f1235769d1070611285b7b6dd79e9f5a0027171e173ed29f361db0401",
    "rcpp/width_calculator.py": "974d48fddcd65ece6c3851a382d96096a7f0e38f617a9f7724117de64201168c",
}


class PRCRollout(Protocol):
    """Rollout signature used to deploy one upstream PRC iterate."""

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


PRCScore = Callable[[object, Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class NativePRCConfig:
    """Scientific inputs required by the official scalar PRC algorithm."""

    alpha: float
    delta: float
    tightness: float
    tau: float
    cohort_size: int

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("PRC alpha must lie in (0, 1)")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("PRC delta must lie in (0, 1)")
        if not 0.0 < self.tightness < self.alpha:
            raise ValueError("PRC tightness must lie in (0, alpha)")
        if self.tau <= 0.0:
            raise ValueError("PRC tau must be positive")
        if self.cohort_size < 2:
            raise ValueError("PRC cohort_size must be at least two")


@dataclass(frozen=True)
class NativePRCResult:
    """Official PRC output plus experiment-facing adaptation diagnostics."""

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
    upstream_width_calculator: str = "rcpp.width_calculator.CLTWidth"
    upstream_risk_measure: str = "rcpp.risk_measure.MeanRiskMeasure"
    risk_definition: str = "mean_trajectory_wise_any_stage_miscoverage"


@dataclass(frozen=True)
class _OfficialPRC:
    run_trajectory: Callable[..., object]
    clt_width: type
    mean_risk: type


def official_prc_root() -> Path:
    """Return the expected checkout root for the pinned upstream repository."""

    return (
        Path(__file__).resolve().parents[2]
        / "baselines"
        / "Performative Risk Control"
    )


def verify_official_prc_source(upstream_root: Path | None = None) -> Path:
    """Fail closed unless the vendored files match the pinned official release."""

    root = (official_prc_root() if upstream_root is None else upstream_root).resolve()
    for relative_path, expected_digest in _OFFICIAL_SOURCE_SHA256.items():
        source_path = root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"official PRC source is missing: {source_path}; clone "
                f"{OFFICIAL_PRC_REPOSITORY} at {OFFICIAL_PRC_COMMIT}"
            )
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise RuntimeError(
                f"official PRC source differs at {source_path}; expected commit "
                f"{OFFICIAL_PRC_COMMIT}"
            )
    return root


def _import_official_prc(upstream_root: Path | None = None) -> _OfficialPRC:
    root = verify_official_prc_source(upstream_root)
    root_text = str(root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        main = importlib.import_module("rcpp.main")
        risk_measure = importlib.import_module("rcpp.risk_measure")
        width_calculator = importlib.import_module("rcpp.width_calculator")
    finally:
        if inserted:
            sys.path.remove(root_text)

    for module in (main, risk_measure, width_calculator):
        _require_module_from_root(module, root)
    return _OfficialPRC(
        run_trajectory=main.run_trajectory,
        clt_width=width_calculator.CLTWidth,
        mean_risk=risk_measure.MeanRiskMeasure,
    )


def _require_module_from_root(module: ModuleType, root: Path) -> None:
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(root):
        raise RuntimeError(
            f"imported {module.__name__} from {module_path}, not pinned PRC root {root}"
        )


class _ProfilePathwiseLoss:
    def __init__(
        self,
        stage_profile: np.ndarray,
        scale_minimum: float,
        scale_maximum: float,
    ) -> None:
        self._stage_profile = stage_profile
        self._scale_minimum = scale_minimum
        self._scale_range = scale_maximum - scale_minimum

    def scale(self, normalized_lambda: float) -> float:
        return self._scale_minimum + float(normalized_lambda) * self._scale_range

    def calc_loss(
        self,
        shifted_data: list[np.ndarray],
        normalized_lambda: float,
        do_new_sample: bool = True,
    ) -> np.ndarray:
        del do_new_sample
        scores = shifted_data[0]
        radii = self.scale(normalized_lambda) * self._stage_profile
        return np.any(scores > radii[None, :], axis=1).astype(np.float64)


class _TrajectoryPerformativitySimulator:
    def __init__(
        self,
        environment: object,
        policy: object,
        outcome_model: object,
        *,
        loss: _ProfilePathwiseLoss,
        stage_profile: Tensor,
        cohort_size: int,
        horizon: int,
        seed: int,
        device: str | torch.device,
        rollout_fn: PRCRollout,
        score_fn: PRCScore,
    ) -> None:
        self._environment = environment
        self._policy = policy
        self._outcome_model = outcome_model
        self._loss = loss
        self._stage_profile = stage_profile
        self._cohort_size = cohort_size
        self._horizon = horizon
        self._seed = seed
        self._device = device
        self._rollout_fn = rollout_fn
        self._score_fn = score_fn
        self.reset()

    def reset(self) -> None:
        self.covered_by_time = torch.zeros(self._horizon, dtype=torch.float64)
        self.pathwise_hits = 0
        self.deployment_count = 0
        self.worst_coverage: list[float] = []
        self.deployed_scales: list[float] = []

    def simulate_shift(
        self,
        base_data: object,
        normalized_lambda: float,
    ) -> list[np.ndarray]:
        del base_data
        selected_scale = self._loss.scale(normalized_lambda)
        radii = selected_scale * self._stage_profile
        deployment_index = len(self.deployed_scales)
        batch = self._rollout_fn(
            self._environment,
            self._policy,
            n=self._cohort_size,
            horizon=self._horizon,
            seed=self._seed + PRC_DEPLOYMENT_SEED_STRIDE * deployment_index,
            device=self._device,
            q=radii.to(self._device),
        )
        scores = self._score_fn(
            self._outcome_model,
            batch.current_states(),
            batch.actions,
            batch.outcomes,
        ).detach().cpu()
        if scores.shape != (self._cohort_size, self._horizon):
            raise ValueError("PRC score function must return shape [cohort_size, horizon]")

        hits = scores <= radii.detach().cpu()[None, :]
        per_time = hits.to(torch.float64).mean(dim=0)
        self.covered_by_time += hits.to(torch.float64).sum(dim=0)
        self.pathwise_hits += int(hits.all(dim=1).sum().item())
        self.deployment_count += self._cohort_size
        self.worst_coverage.append(float(per_time.min().item()))
        self.deployed_scales.append(selected_scale)
        return [scores.numpy()]


def _reuse_selection_cohort_for_tracking(
    shifted_data: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Keep upstream tracking from consuming a second target cohort.

    The second output affects only ``Trajectory.risks_*`` inside the official
    implementation.  Paper metrics are computed on a separate evaluation set.
    """

    return shifted_data, shifted_data


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
    score_fn: PRCScore | None = None,
    upstream_root: Path | None = None,
) -> NativePRCResult:
    """Run official PRC on a normalized scalar profile family.

    The upstream scalar ``lambda`` lies in ``[0, 1]``.  It is mapped linearly
    onto the endpoints of ``scale_grid``; the upper endpoint is the upstream
    safe initialization.  The bounded loss is one when any stage of a
    trajectory is uncovered.  Controlling this mean pathwise loss implies the
    paper's weaker per-step marginal coverage target.
    """

    if stage_profile.shape != (horizon,):
        raise ValueError("stage_profile must have shape [horizon]")
    if len(scale_grid) < 2 or scale_grid.ndim != 1:
        raise ValueError("scale_grid must be a one-dimensional grid with at least two points")
    if not bool(torch.all(stage_profile > 0)):
        raise ValueError("stage_profile must be strictly positive")
    ordered_grid = scale_grid.detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(ordered_grid).all()):
        raise ValueError("scale_grid must be finite")
    if not bool(torch.all(ordered_grid[1:] > ordered_grid[:-1])):
        raise ValueError("scale_grid must be strictly increasing")
    scale_minimum = float(ordered_grid[0].item())
    scale_maximum = float(ordered_grid[-1].item())
    if scale_minimum <= 0.0:
        raise ValueError("scale_grid must be strictly positive")

    official = _import_official_prc(upstream_root)
    if rollout_fn is None:
        from scpcp.simulator import rollout

        rollout_fn = rollout
    if score_fn is None:
        from scpcp.scores import score_batch

        score_fn = score_batch

    deployment_profile = stage_profile.detach().cpu()
    profile_cpu = deployment_profile.to(torch.float64)
    loss = _ProfilePathwiseLoss(
        profile_cpu.numpy(),
        scale_minimum,
        scale_maximum,
    )
    simulator = _TrajectoryPerformativitySimulator(
        environment,
        policy,
        outcome_model,
        loss=loss,
        stage_profile=deployment_profile,
        cohort_size=config.cohort_size,
        horizon=horizon,
        seed=seed,
        device=device,
        rollout_fn=rollout_fn,
        score_fn=score_fn,
    )
    upstream_args = SimpleNamespace(
        alpha=config.alpha,
        delta=config.delta,
        tightness=config.tightness,
        tau=config.tau,
        N=config.cohort_size,
        lambda_min=0.0,
        lambda_safe=1.0,
    )
    trajectory = official.run_trajectory(
        itertools.repeat(None),
        _reuse_selection_cohort_for_tracking,
        official.clt_width(config.alpha, loss_max=1.0),
        official.mean_risk(),
        simulator,
        loss,
        upstream_args,
    )

    normalized_lambda = float(np.asarray(trajectory.lambda_hat).item())
    selected_scale = loss.scale(normalized_lambda)
    lambda_path = tuple(float(value) for value in trajectory.lambdas)
    deployments = simulator.deployment_count
    if deployments == 0:
        raise RuntimeError("official PRC produced no target deployment")
    return NativePRCResult(
        radius_by_time=(selected_scale * profile_cpu).to(
            device=stage_profile.device,
            dtype=stage_profile.dtype,
        ),
        selected_scale=selected_scale,
        normalized_lambda=normalized_lambda,
        target_deployments=deployments,
        rounds=len(simulator.deployed_scales),
        optimization_rounds=len(lambda_path) - 1,
        adaptation_per_time_coverage=(
            simulator.covered_by_time / deployments
        ).to(torch.float32),
        adaptation_round_worst_coverage=tuple(simulator.worst_coverage),
        adaptation_pathwise_coverage=simulator.pathwise_hits / deployments,
        lambda_path=lambda_path,
        deployed_scale_path=tuple(simulator.deployed_scales),
        empirical_deployed_risk_path=tuple(
            float(value) for value in trajectory.risks_tt
        ),
        empirical_candidate_risk_path=tuple(
            float(value) for value in trajectory.risks_tm1_t
        ),
        guaranteed_iterations=int(trajectory.guaranteed_T),
        delta_lambda=float(trajectory.delta_lambda),
        alpha=config.alpha,
        delta=config.delta,
        tightness=config.tightness,
        tau=config.tau,
        cohort_size=config.cohort_size,
        scale_minimum=scale_minimum,
        scale_maximum=scale_maximum,
    )


def prc_profile_scale(
    environment: object,
    policy: object,
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
    maximum_step: float,
    tightness: float | None = None,
    tau: float = 1.0,
    rollout_fn: PRCRollout | None = None,
    score_fn: PRCScore | None = None,
    upstream_root: Path | None = None,
) -> NativePRCResult:
    """Migration-compatible entry point that executes the official PRC core.

    ``rounds`` and ``maximum_step`` belonged to the retired local grid
    controller.  They are accepted so existing experiment call sites can move
    to this module without a flag-day rewrite, but they do not alter the
    upstream algorithm: official PRC determines its own iteration count and
    step size.  ``total_rollouts`` becomes the upstream per-deployment cohort
    size ``N``.  Consequently the returned target-deployment count is usually
    greater than ``total_rollouts`` and must be reported as such.
    """

    if rounds < 1:
        raise ValueError("legacy PRC rounds must be positive during migration")
    if maximum_step <= 0.0:
        raise ValueError("legacy PRC maximum_step must be positive during migration")
    if not np.isfinite(initial_scale) or initial_scale <= 0.0:
        raise ValueError("initial_scale must be finite and positive")

    ordered_grid = scale_grid.detach().clone()
    if ordered_grid.ndim != 1 or len(ordered_grid) < 2:
        raise ValueError("scale_grid must be a one-dimensional grid with at least two points")
    if initial_scale > float(ordered_grid[-1].item()):
        ordered_grid[-1] = initial_scale

    return native_prc_profile_scale(
        environment,
        policy,
        outcome_model,
        ordered_grid,
        stage_profile,
        config=NativePRCConfig(
            alpha=alpha,
            delta=delta,
            tightness=0.8 * alpha if tightness is None else tightness,
            tau=tau,
            cohort_size=total_rollouts,
        ),
        horizon=horizon,
        seed=seed,
        device=device,
        rollout_fn=rollout_fn,
        score_fn=score_fn,
        upstream_root=upstream_root,
    )
