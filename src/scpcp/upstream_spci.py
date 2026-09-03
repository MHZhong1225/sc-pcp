"""Thin bridge to the vendored, unmodified MultiDimSPCI implementation.

This module adapts array and :class:`TrajectoryBatch` layouts only.  Bootstrap
prediction, leave-one-out residuals, covariance estimation, Mahalanobis scores,
QRF fitting, interval selection, coverage, and ellipsoid volumes are all
computed by the upstream ``SPCI_and_EnbPI`` class.
"""

from __future__ import annotations

import importlib
import inspect
import math
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
import torch
from torch import Tensor

from scpcp.data import TrajectoryBatch
from scpcp.simulator import rollout


UPSTREAM_COMMIT = "2b22e47088ed37ebc48d1bb9fdfa192450f289a2"
UPSTREAM_QRF_REQUIREMENT = "sklearn-quantile==0.0.21"
_DEFAULT_UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "MultiDimSPCI"
_UPSTREAM_RUN_LOCK = threading.Lock()


class NativeSPCIUnavailable(RuntimeError):
    """Raised when the vendored implementation or its QRF dependency is absent."""


class SPCIRollout(Protocol):
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
class NativeSPCIConfig:
    """Hyperparameters exposed by the upstream ICML 2024 implementation."""

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
        if self.bootstrap_models < 1:
            raise ValueError("bootstrap_models must be positive")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.past_window < 1:
            raise ValueError("past_window must be positive")
        if self.bins < 2:
            raise ValueError("bins must be at least two")
        if self.qrf_estimators < 1:
            raise ValueError("qrf_estimators must be positive")
        if self.qrf_max_depth < 1:
            raise ValueError("qrf_max_depth must be positive")
        if not 0.0 < self.residual_weight_decay <= 1.0:
            raise ValueError("residual_weight_decay must lie in (0, 1]")
        if self.qrf_training_window is not None and self.qrf_training_window < 1:
            raise ValueError("qrf_training_window must be positive when provided")
        if self.covariance_rank is not None and self.covariance_rank < 1:
            raise ValueError("covariance_rank must be positive when provided")


@dataclass(frozen=True)
class NativeSPCIResult:
    """Outputs copied directly from one upstream ``SPCI_and_EnbPI`` run."""

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
    upstream_model: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class StagewiseNativeSPCIResult:
    """Native MultiDimSPCI runs over the incoming-patient stream at each stage."""

    stages: tuple[NativeSPCIResult, ...]

    @property
    def prediction_centers(self) -> np.ndarray:
        return np.stack([result.prediction_centers for result in self.stages], axis=1)

    @property
    def lower_radii(self) -> np.ndarray:
        return np.stack([result.lower_radii for result in self.stages], axis=1)

    @property
    def upper_radii(self) -> np.ndarray:
        return np.stack([result.upper_radii for result in self.stages], axis=1)

    @property
    def mahalanobis_scores(self) -> np.ndarray:
        return np.stack(
            [result.prediction_mahalanobis_scores for result in self.stages], axis=1
        )

    @property
    def covered(self) -> np.ndarray:
        return np.stack([result.covered for result in self.stages], axis=1)

    @property
    def ellipsoid_volumes(self) -> np.ndarray:
        return np.stack([result.ellipsoid_volumes for result in self.stages], axis=1)

    @property
    def coverage_by_stage(self) -> np.ndarray:
        return self.covered.mean(axis=0)

    @property
    def mean_volume_by_stage(self) -> np.ndarray:
        return self.ellipsoid_volumes.mean(axis=0)


@dataclass(frozen=True)
class SourceBackedSPCIResult:
    """Online result for the explicit normalized-score/box mapping."""

    radius_by_time: Tensor
    target_deployments: int
    rounds: int
    adaptation_per_time_coverage: Tensor
    adaptation_round_worst_coverage: tuple[float, ...]
    adaptation_pathwise_coverage: float
    selected_scale: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


def run_native_spci(
    x_train: np.ndarray,
    x_predict: np.ndarray,
    y_train: np.ndarray,
    y_predict: np.ndarray,
    *,
    fit_model: RegressorMixin,
    alpha: float,
    seed: int,
    config: NativeSPCIConfig = NativeSPCIConfig(),
    upstream_root: Path | None = None,
) -> NativeSPCIResult:
    """Run the official MultiDimSPCI class on one chronological stream."""

    config.validate()
    x_train, x_predict, y_train, y_predict = _validated_arrays(
        x_train, x_predict, y_train, y_predict, alpha=alpha, config=config
    )
    root = (upstream_root or _DEFAULT_UPSTREAM_ROOT).resolve()
    upstream_class = _load_upstream_class(str(root))

    with _UPSTREAM_RUN_LOCK, _upstream_numpy_context(seed):
        model = upstream_class(x_train, x_predict, y_train, y_predict, fit_model)
        model.bins = config.bins
        model.n_estimators = config.qrf_estimators
        model.max_d = config.qrf_max_depth
        model.criterion = config.qrf_criterion
        model.weigh_residuals = config.weighted_residuals
        model.c = config.residual_weight_decay
        model.T1 = config.qrf_training_window
        model.r = config.covariance_rank
        model.use_local_ellipsoid = config.local_ellipsoid
        model.fit_bootstrap_models_online_multistep(
            B=config.bootstrap_models,
            stride=config.stride,
        )
        model.compute_Widths_Ensemble_online(
            alpha=alpha,
            stride=config.stride,
            smallT=config.small_training_window,
            past_window=config.past_window,
            use_SPCI=True,
            quantile_regr="RF",
        )
        mean_coverage, mean_volume = model.get_results()

    lower_radii = model.Width_Ensemble["lower"].to_numpy(copy=True)
    upper_radii = model.Width_Ensemble["upper"].to_numpy(copy=True)
    covariance_matrices = _prediction_covariances(model, len(y_predict))
    return NativeSPCIResult(
        prediction_centers=np.asarray(model.Ensemble_pred_interval_centers).copy(),
        training_mahalanobis_scores=np.asarray(model.train_et).copy(),
        prediction_mahalanobis_scores=np.asarray(model.test_et).copy(),
        lower_radii=lower_radii,
        upper_radii=upper_radii,
        covariance_matrices=covariance_matrices,
        covered=np.asarray(model.coverages_all, dtype=bool),
        ellipsoid_volumes=np.asarray(model.width_all, dtype=float),
        mean_coverage=float(mean_coverage),
        mean_volume=float(mean_volume),
        upstream_model=model,
    )


def run_stagewise_native_spci(
    training: TrajectoryBatch,
    prediction: TrajectoryBatch,
    *,
    n_actions: int,
    alpha: float,
    seed: int,
    fit_model_factory: Callable[[], RegressorMixin] = LinearRegression,
    config: NativeSPCIConfig = NativeSPCIConfig(),
    upstream_root: Path | None = None,
) -> StagewiseNativeSPCIResult:
    """Run unmodified MultiDimSPCI separately for every trajectory stage.

    The sequence axis seen by upstream SPCI is the order in which patients
    arrive at a fixed decision stage.  Features are the current state followed
    by a one-hot observed action; outcomes remain the original multivariate
    vector.  Keeping stages separate avoids creating artificial temporal links
    across episode boundaries.

    The returned regions are observation-specific ellipsoids.  They are not a
    ``radius_by_time`` for SC-PCP's normalized-max boxes and cannot be converted
    to that interface without changing the prediction set.
    """

    if training.horizon != prediction.horizon:
        raise ValueError("training and prediction batches must share a horizon")
    if training.state_dim != prediction.state_dim:
        raise ValueError("training and prediction batches must share a state dimension")
    if training.outcome_dim != prediction.outcome_dim:
        raise ValueError("training and prediction batches must share an outcome dimension")

    results = []
    for stage in range(training.horizon):
        x_train, y_train = _stage_arrays(training, stage, n_actions)
        x_predict, y_predict = _stage_arrays(prediction, stage, n_actions)
        results.append(
            run_native_spci(
                x_train,
                x_predict,
                y_train,
                y_predict,
                fit_model=fit_model_factory(),
                alpha=alpha,
                seed=seed + 104_729 * stage,
                config=config,
                upstream_root=upstream_root,
            )
        )
    return StagewiseNativeSPCIResult(tuple(results))


@torch.no_grad()
def upstream_spci_qrf_controller(
    environment: object,
    policy: object,
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
    lag: int = 10,
    bins: int = 5,
    qrf_estimators: int = 10,
    qrf_max_depth: int = 2,
    rollout_fn: SPCIRollout | None = None,
    upstream_root: Path | None = None,
) -> SourceBackedSPCIResult:
    """Run upstream SPCI's conditional QRF on the common box score stream.

    This is the callable bridge for the existing paper pipeline.  It directly
    invokes the vendored class's ``multi_step_QRF`` and ``train_QRF`` methods;
    there is no local quantile-regression implementation.  The interface map is
    deliberate and recorded in ``provenance``: the inputs are frozen-model
    normalized-maximum scores, and the returned upper conditional quantiles are
    radii of the existing axis-aligned boxes.  Consequently this path uses the
    upstream conditional-quantile algorithm but is not its native ellipsoid.
    Use :func:`run_stagewise_native_spci` when native LOO, covariance,
    Mahalanobis, and ellipsoid outputs are required.
    """

    if initial_scores.ndim != 2 or initial_scores.shape[1] != horizon:
        raise ValueError("initial_scores must have shape [N, horizon]")
    if lag < 1:
        raise ValueError("lag must be positive")
    if len(initial_scores) <= lag:
        raise ValueError("initial score history must be longer than the SPCI lag")
    if residual_window <= lag:
        raise ValueError("residual_window must be longer than the SPCI lag")
    if total_rollouts < rounds or rounds < 1:
        raise ValueError("total_rollouts must cover every positive adaptation round")

    root = (upstream_root or _DEFAULT_UPSTREAM_ROOT).resolve()
    histories = [
        initial_scores[:, stage].detach().cpu().to(torch.float64)[-residual_window:]
        for stage in range(horizon)
    ]
    round_sizes = _round_sizes(total_rollouts, rounds)
    covered_by_time = torch.zeros(horizon, dtype=torch.float64)
    adaptation_trajectories = 0
    pathwise_hits = 0
    round_worst: list[float] = []

    for round_index, rollout_size in enumerate(round_sizes):
        radii = _upstream_stagewise_qrf_radii(
            histories,
            alpha=alpha,
            seed=seed + 47_021 * round_index,
            lag=lag,
            bins=bins,
            qrf_estimators=qrf_estimators,
            qrf_max_depth=qrf_max_depth,
            qrf_training_window=residual_window,
            upstream_root=root,
        )
        selected_rollout = rollout if rollout_fn is None else rollout_fn
        deployed = selected_rollout(
            environment,
            policy,
            n=rollout_size,
            horizon=horizon,
            seed=seed + 47_021 * round_index,
            device=device,
            q=radii.to(device=device, dtype=initial_scores.dtype),
        )
        from scpcp.scores import score_batch

        scores = score_batch(
            outcome_model,
            deployed.current_states(),
            deployed.actions,
            deployed.outcomes,
        ).detach().cpu()
        hits = scores <= radii.to(scores)[None, :]
        per_time = hits.to(torch.float64).mean(dim=0)
        covered_by_time += hits.to(torch.float64).sum(dim=0)
        adaptation_trajectories += len(scores)
        pathwise_hits += int(hits.all(dim=1).sum().item())
        round_worst.append(float(per_time.min().item()))
        for stage in range(horizon):
            histories[stage] = torch.cat((histories[stage], scores[:, stage]))[
                -residual_window:
            ]

    final_radii = _upstream_stagewise_qrf_radii(
        histories,
        alpha=alpha,
        seed=seed + 47_021 * rounds,
        lag=lag,
        bins=bins,
        qrf_estimators=qrf_estimators,
        qrf_max_depth=qrf_max_depth,
        qrf_training_window=residual_window,
        upstream_root=root,
    )
    return SourceBackedSPCIResult(
        radius_by_time=final_radii.to(dtype=initial_scores.dtype),
        target_deployments=adaptation_trajectories,
        rounds=rounds,
        adaptation_per_time_coverage=(
            covered_by_time / adaptation_trajectories
        ).to(torch.float32),
        adaptation_round_worst_coverage=tuple(round_worst),
        adaptation_pathwise_coverage=pathwise_hits / adaptation_trajectories,
        provenance={
            "implementation": "vendored_multidim_spci_conditional_qrf",
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_source": str(root / "helpers" / "MultiDim_SPCI_class.py"),
            "upstream_methods": ["multi_step_QRF", "train_QRF"],
            "score_mapping": "frozen_model_normalized_max_nonconformity",
            "prediction_set_mapping": "upper_qrf_quantile_to_axis_aligned_box_radius",
            "native_bootstrap_loo": False,
            "native_covariance_mahalanobis_ellipsoid": False,
            "native_ellipsoid_entry_point": "run_stagewise_native_spci",
            "lag": lag,
            "training_window": residual_window,
            "bins": bins,
            "qrf_estimators": qrf_estimators,
            "qrf_max_depth": qrf_max_depth,
        },
    )


def _upstream_stagewise_qrf_radii(
    histories: list[Tensor],
    *,
    alpha: float,
    seed: int,
    lag: int,
    bins: int,
    qrf_estimators: int,
    qrf_max_depth: int,
    qrf_training_window: int,
    upstream_root: Path,
) -> Tensor:
    radii = [
        _upstream_qrf_upper_radius(
            history.numpy(),
            alpha=alpha,
            seed=seed + 104_729 * stage,
            lag=lag,
            bins=bins,
            qrf_estimators=qrf_estimators,
            qrf_max_depth=qrf_max_depth,
            qrf_training_window=qrf_training_window,
            upstream_root=upstream_root,
        )
        for stage, history in enumerate(histories)
    ]
    return torch.tensor(radii, dtype=torch.float64)


def _upstream_qrf_upper_radius(
    score_history: np.ndarray,
    *,
    alpha: float,
    seed: int,
    lag: int,
    bins: int,
    qrf_estimators: int,
    qrf_max_depth: int,
    qrf_training_window: int,
    upstream_root: Path,
) -> float:
    upstream_class = _load_upstream_class(str(upstream_root))
    dummy_x = np.zeros((2, 1), dtype=float)
    dummy_y = np.zeros((2, 1), dtype=float)
    with _UPSTREAM_RUN_LOCK, _upstream_numpy_context(seed):
        model = upstream_class(
            dummy_x,
            dummy_x[:1],
            dummy_y,
            dummy_y[:1],
            LinearRegression(),
        )
        model.alpha = alpha
        model.past_window = lag
        model.bins = bins
        model.n_estimators = qrf_estimators
        model.max_d = qrf_max_depth
        model.criterion = "squared_error"
        model.weigh_residuals = False
        model.T1 = min(qrf_training_window, len(score_history) - lag)
        model.use_local_ellipsoid = False
        model.global_cov = np.eye(1)
        model.QRF_ls = []
        model.i_star_ls = []
        prediction_features = model.multi_step_QRF(
            np.asarray(score_history, dtype=float),
            i=0,
            s=1,
            n2=lag,
        )
        quantiles = np.asarray(model.QRF_ls[0].predict(prediction_features)).reshape(-1)
        upper_offset = len(quantiles) // 2
        return float(quantiles[upper_offset + model.i_star_ls[0]])


def _round_sizes(total_rollouts: int, rounds: int) -> tuple[int, ...]:
    base, remainder = divmod(total_rollouts, rounds)
    return tuple(base + (round_index < remainder) for round_index in range(rounds))


def _stage_arrays(
    batch: TrajectoryBatch,
    stage: int,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= stage < batch.horizon:
        raise ValueError("stage is outside the trajectory horizon")
    if n_actions < 1:
        raise ValueError("n_actions must be positive")
    actions = batch.actions[:, stage].detach().cpu().long()
    if bool(((actions < 0) | (actions >= n_actions)).any()):
        raise ValueError("actions must be in [0, n_actions)")
    one_hot_actions = np.eye(n_actions, dtype=np.float64)[actions.numpy()]
    states = batch.current_states()[:, stage].detach().cpu().numpy().astype(np.float64)
    outcomes = batch.outcomes[:, stage].detach().cpu().numpy().astype(np.float64)
    return np.concatenate((states, one_hot_actions), axis=1), outcomes


def _validated_arrays(
    x_train: np.ndarray,
    x_predict: np.ndarray,
    y_train: np.ndarray,
    y_predict: np.ndarray,
    *,
    alpha: float,
    config: NativeSPCIConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(
        np.asarray(array, dtype=np.float64)
        for array in (x_train, x_predict, y_train, y_predict)
    )
    x_train, x_predict, y_train, y_predict = arrays
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("SPCI features and outcomes must all be two-dimensional")
    if len(x_train) != len(y_train) or len(x_predict) != len(y_predict):
        raise ValueError("features and outcomes must have matching row counts")
    if x_train.shape[1] != x_predict.shape[1]:
        raise ValueError("training and prediction features must have the same width")
    if y_train.shape[1] != y_predict.shape[1]:
        raise ValueError("training and prediction outcomes must have the same width")
    if y_train.shape[1] < 2:
        raise ValueError("MultiDimSPCI requires at least two outcome dimensions")
    if len(x_predict) < 1:
        raise ValueError("prediction stream must not be empty")
    if len(x_train) <= config.past_window + config.stride - 1:
        raise ValueError("training stream is too short for the configured QRF lag")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("SPCI inputs must be finite")
    return arrays


def _prediction_covariances(model: Any, prediction_count: int) -> np.ndarray:
    if model.use_local_ellipsoid:
        return np.stack([np.asarray(value).copy() for value in model.cov_matrix_ls])
    covariance = np.asarray(model.global_cov).copy()
    return np.repeat(covariance[None, :, :], prediction_count, axis=0)


@lru_cache(maxsize=None)
def _load_upstream_class(upstream_root: str) -> type[Any]:
    root = Path(upstream_root)
    module_path = root / "helpers" / "MultiDim_SPCI_class.py"
    if not module_path.is_file():
        raise NativeSPCIUnavailable(
            f"vendored MultiDimSPCI source is missing: {module_path}"
        )

    existing_helpers = sys.modules.get("helpers")
    if existing_helpers is not None:
        helpers_file = getattr(existing_helpers, "__file__", None)
        helpers_path = Path(helpers_file).resolve() if helpers_file else None
        if helpers_path is None or root not in helpers_path.parents:
            raise NativeSPCIUnavailable(
                "an unrelated top-level 'helpers' package is already imported; "
                "run native SPCI in a clean Python process"
            )

    with _prepend_import_path(root):
        try:
            module = importlib.import_module("helpers.MultiDim_SPCI_class")
        except ModuleNotFoundError as error:
            if error.name == "sklearn_quantile":
                raise NativeSPCIUnavailable(
                    "MultiDimSPCI requires sklearn-quantile; the upstream repository "
                    f"pins {UPSTREAM_QRF_REQUIREMENT}"
                ) from error
            raise

    loaded_path = Path(inspect.getfile(module)).resolve()
    if loaded_path != module_path.resolve():
        raise NativeSPCIUnavailable(
            f"loaded MultiDimSPCI from {loaded_path}, expected {module_path.resolve()}"
        )
    return module.SPCI_and_EnbPI


@contextmanager
def _prepend_import_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        sys.path.remove(value)


@contextmanager
def _upstream_numpy_context(seed: int) -> Iterator[None]:
    """Isolate upstream's global RNG and NumPy 2 ``np.math`` compatibility."""

    random_state = np.random.get_state()
    had_math = hasattr(np, "math")
    if not had_math:
        np.math = math  # type: ignore[attr-defined]
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(random_state)
        if not had_math:
            del np.math  # type: ignore[attr-defined]
