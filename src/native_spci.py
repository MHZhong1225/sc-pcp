"""Strict bridge to the official MultiDimSPCI implementation.

The upstream method receives a chronological stream at one fixed decision
stage and returns observation-specific prediction ellipsoids.  This adapter
only lays out patient trajectories as that stream (state plus observed action
as features, original multivariate outcome as response); bootstrap LOO,
Mahalanobis scores, QRF, and ellipsoid construction remain upstream code.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import math
import sys
import threading
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression

from data import TrajectoryBatch


UPSTREAM_SPCI_REPOSITORY = "https://github.com/hamrel-cxu/MultiDimSPCI"
UPSTREAM_SPCI_COMMIT = "2b22e47088ed37ebc48d1bb9fdfa192450f289a2"
UPSTREAM_SPCI_REQUIREMENT = "sklearn-quantile==0.0.21"
UPSTREAM_SPCI_LICENSE = "MIT"
_SOURCE_SHA256 = {
    "LICENSE": "06296bc6aada7e0aadcfba16abcb6c2f7f2f5f19db91b384b7874eb891305eb4",
    "helpers/MultiDim_SPCI_class.py": "b6949c16e78f7edb5ed914b8302bb13997d4d73eb7bdc9ec5bbf650305ce2cc2",
    "helpers/utils_SPCI.py": "b6d981c72e70433d040f2b640d32896b79ea69b032632a8108a85317866c0f66",
}
_RUN_LOCK = threading.Lock()


class NativeSPCIUnavailable(RuntimeError):
    """Raised when the pinned SPCI release or required dependency is absent."""


@dataclass(frozen=True)
class NativeSPCIConfig:
    bootstrap_models: int = 15
    stride: int = 1
    past_window: int = 10
    bins: int = 5
    qrf_estimators: int = 10
    qrf_max_depth: int = 2
    qrf_criterion: str = "squared_error"
    small_training_window: bool = False
    weighted_residuals: bool = False
    residual_weight_decay: float = 0.995
    qrf_training_window: int | None = None
    covariance_rank: int | None = None
    local_ellipsoid: bool = False

    def validate(self) -> None:
        if min(self.bootstrap_models, self.stride, self.past_window, self.qrf_estimators, self.qrf_max_depth) < 1:
            raise ValueError("SPCI bootstrap, stride, window, and QRF settings must be positive")
        if self.bins < 2 or not 0.0 < self.residual_weight_decay <= 1.0:
            raise ValueError("SPCI bins must be at least two and decay in (0, 1]")


@dataclass(frozen=True)
class NativeSPCIResult:
    prediction_centers: np.ndarray
    training_mahalanobis_scores: np.ndarray
    prediction_mahalanobis_scores: np.ndarray
    lower_radii: np.ndarray
    upper_radii: np.ndarray
    covariance_matrices: np.ndarray
    covered: np.ndarray
    ellipsoid_volumes: np.ndarray
    mean_coverage: float
    mean_volume: float
    upstream_repository: str = UPSTREAM_SPCI_REPOSITORY
    upstream_commit: str = UPSTREAM_SPCI_COMMIT
    upstream_entrypoint: str = "helpers.MultiDim_SPCI_class.SPCI_and_EnbPI"


@dataclass(frozen=True)
class StagewiseNativeSPCIResult:
    stages: tuple[NativeSPCIResult, ...]

    @property
    def coverage_by_stage(self) -> np.ndarray:
        return np.stack([stage.covered for stage in self.stages], axis=1).mean(axis=0)

    @property
    def volume_by_stage(self) -> np.ndarray:
        return np.stack([stage.ellipsoid_volumes for stage in self.stages], axis=1).mean(axis=0)


def official_spci_root() -> Path:
    return Path(__file__).resolve().parents[1] / "internal" / "baselines" / "MultiDimSPCI"


def verify_official_spci_source(upstream_root: Path | None = None) -> Path:
    root = (official_spci_root() if upstream_root is None else upstream_root).resolve()
    for relative, expected in _SOURCE_SHA256.items():
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"official MultiDimSPCI source is missing: {source}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"official MultiDimSPCI source differs from pinned commit {UPSTREAM_SPCI_COMMIT}: {source}")
    return root


def verify_native_spci_runtime() -> None:
    """Require the dependency revision declared by the official repository."""

    try:
        installed = importlib.metadata.version("sklearn-quantile")
    except importlib.metadata.PackageNotFoundError as error:
        raise NativeSPCIUnavailable(
            f"install the pinned upstream dependency {UPSTREAM_SPCI_REQUIREMENT}"
        ) from error
    if installed != "0.0.21":
        raise NativeSPCIUnavailable(
            f"official MultiDimSPCI requires sklearn-quantile==0.0.21, found {installed}"
        )


@contextmanager
def _import_context(root: Path) -> Iterator[type]:
    root_text = str(root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    # NumPy 2 removed the historical ``np.math`` alias used by the untouched
    # 2024 upstream helper.  Supplying the standard-library alias preserves
    # its exact volume formula without editing the verified upstream source.
    restore_numpy_math = not hasattr(np, "math")
    if restore_numpy_math:
        np.math = math  # type: ignore[attr-defined]
    try:
        try:
            verify_native_spci_runtime()
            module = importlib.import_module("helpers.MultiDim_SPCI_class")
        except ModuleNotFoundError as error:
            if error.name == "sklearn_quantile":
                raise NativeSPCIUnavailable(
                    f"install the pinned upstream dependency {UPSTREAM_SPCI_REQUIREMENT}"
                ) from error
            raise
        module_file = Path(module.__file__).resolve()
        if not module_file.is_relative_to(root):
            raise RuntimeError("MultiDimSPCI was imported outside the verified checkout")
        yield module.SPCI_and_EnbPI
    finally:
        if restore_numpy_math:
            delattr(np, "math")
        if inserted:
            sys.path.remove(root_text)


def run_native_spci(
    x_train: np.ndarray,
    x_predict: np.ndarray,
    y_train: np.ndarray,
    y_predict: np.ndarray,
    *,
    alpha: float,
    seed: int,
    config: NativeSPCIConfig = NativeSPCIConfig(),
    fit_model: RegressorMixin | None = None,
    upstream_root: Path | None = None,
) -> NativeSPCIResult:
    """Execute one unmodified official SPCI fit over a chronological stream."""

    config.validate()
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (x_train, x_predict, y_train, y_predict))
    x_train, x_predict, y_train, y_predict = arrays
    if not 0.0 < alpha < 1.0 or x_train.ndim != 2 or x_predict.ndim != 2 or y_train.ndim != 2 or y_predict.ndim != 2:
        raise ValueError("SPCI requires 2-D feature/response arrays and alpha in (0, 1)")
    if x_train.shape[1] != x_predict.shape[1] or y_train.shape[1] != y_predict.shape[1]:
        raise ValueError("SPCI training and prediction arrays must share feature and response dimensions")
    if len(x_train) <= config.past_window or len(x_predict) < 1:
        raise ValueError("SPCI training stream must exceed past_window and prediction stream be nonempty")
    root = verify_official_spci_source(upstream_root)
    with _RUN_LOCK, _numpy_seed(seed), _import_context(root) as upstream:
        model = upstream(x_train, x_predict, y_train, y_predict, LinearRegression() if fit_model is None else fit_model)
        model.bins, model.n_estimators, model.max_d = config.bins, config.qrf_estimators, config.qrf_max_depth
        model.criterion, model.weigh_residuals, model.c = config.qrf_criterion, config.weighted_residuals, config.residual_weight_decay
        model.T1, model.r, model.use_local_ellipsoid = config.qrf_training_window, config.covariance_rank, config.local_ellipsoid
        model.fit_bootstrap_models_online_multistep(B=config.bootstrap_models, stride=config.stride)
        model.compute_Widths_Ensemble_online(alpha=alpha, stride=config.stride, smallT=config.small_training_window, past_window=config.past_window, use_SPCI=True, quantile_regr="RF")
        mean_coverage, mean_volume = model.get_results()
    return NativeSPCIResult(
        prediction_centers=np.asarray(model.Ensemble_pred_interval_centers).copy(),
        training_mahalanobis_scores=np.asarray(model.train_et).copy(),
        prediction_mahalanobis_scores=np.asarray(model.test_et).copy(),
        lower_radii=model.Width_Ensemble["lower"].to_numpy(copy=True),
        upper_radii=model.Width_Ensemble["upper"].to_numpy(copy=True),
        covariance_matrices=np.asarray(model.cov_matrix_ls if config.local_ellipsoid else [model.global_cov] * len(y_predict)).copy(),
        covered=np.asarray(model.coverages_all, dtype=bool),
        ellipsoid_volumes=np.asarray(model.width_all, dtype=float),
        mean_coverage=float(mean_coverage), mean_volume=float(mean_volume),
    )


def run_stagewise_native_spci(
    training: TrajectoryBatch,
    target_adaptation: TrajectoryBatch,
    *,
    n_actions: int,
    alpha: float,
    seed: int,
    config: NativeSPCIConfig = NativeSPCIConfig(),
    upstream_root: Path | None = None,
) -> StagewiseNativeSPCIResult:
    """Apply original SPCI to each fixed-stage patient-arrival stream.

    ``target_adaptation`` can contain the full 2,000 target-policy arrivals;
    this is an online-information advantage and must be reported alongside any
    resulting coverage or volume summary.  Outputs remain native ellipsoids.
    """

    if training.horizon != target_adaptation.horizon or training.outcome_dim != target_adaptation.outcome_dim:
        raise ValueError("SPCI training and target streams must share horizon and outcome dimension")
    if n_actions < 1:
        raise ValueError("n_actions must be positive")
    return StagewiseNativeSPCIResult(tuple(
        run_native_spci(*_stage_arrays(training, target_adaptation, stage, n_actions), alpha=alpha, seed=seed + 104_729 * stage, config=config, upstream_root=upstream_root)
        for stage in range(training.horizon)
    ))


def _stage_arrays(training: TrajectoryBatch, target: TrajectoryBatch, stage: int, n_actions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def extract(batch: TrajectoryBatch) -> tuple[np.ndarray, np.ndarray]:
        actions = batch.actions[:, stage].detach().cpu().numpy()
        if np.any((actions < 0) | (actions >= n_actions)):
            raise ValueError("SPCI observed actions must lie in [0, n_actions)")
        features = np.concatenate((batch.current_states()[:, stage].detach().cpu().numpy(), np.eye(n_actions)[actions]), axis=1)
        return features, batch.outcomes[:, stage].detach().cpu().numpy()
    x_train, y_train = extract(training)
    x_target, y_target = extract(target)
    return x_train, x_target, y_train, y_target


@contextmanager
def _numpy_seed(seed: int) -> Iterator[None]:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
