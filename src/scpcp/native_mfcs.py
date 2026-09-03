"""Thin bridge to the vendored conformal-MFCS implementation.

The upstream repository owns the finite-depth replacement recursion and its
weighted-quantile convention.  This module only converts project arrays into
that public interface; it deliberately does not reimplement either algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike, NDArray


UPSTREAM_MFCS_REVISION = "c737536d874fda9f0da6dbc95a16a91c71df2512"
UPSTREAM_MFCS_LICENSE = "MIT"
UPSTREAM_MFCS_REPOSITORY = "https://github.com/drewprinster/conformal-mfcs"
UPSTREAM_MFCS_SOURCE_SHA256 = (
    "766d54864467ce3db635a30da94e5284f1140fba7f5150ce93743896adb0e880"
)
DEFAULT_MFCS_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "baselines"
    / "conformal-mfcs"
    / "calibrate_mfcs.py"
)


@dataclass(frozen=True)
class NativeMFCSCalibration:
    """One native split-MFCS calibration result for one test covariate."""

    radius: float
    unnormalized_weights: NDArray[np.float64]
    normalized_weights: NDArray[np.float64]
    depth: int
    upstream_repository: str = UPSTREAM_MFCS_REPOSITORY
    upstream_commit: str = UPSTREAM_MFCS_REVISION
    upstream_license: str = UPSTREAM_MFCS_LICENSE
    upstream_entrypoint: str = (
        "calibrate_mfcs.compute_w_ptest_split_active_replacement"
    )

    @property
    def has_finite_radius(self) -> bool:
        return bool(np.isfinite(self.radius))


def native_mfcs_split_radius(
    calibration_scores: ArrayLike,
    query_values: ArrayLike,
    *,
    alpha: float,
    depth: int,
    source_file: Path = DEFAULT_MFCS_SOURCE,
) -> NativeMFCSCalibration:
    """Call upstream split-MFCS for one calibration/test query history.

    ``query_values`` follows the upstream ``cal_test_vals_mat`` contract: rows
    are successive feedback/query mechanisms, the first ``N`` columns are the
    calibration covariates, and the final column is the current test covariate.
    The returned radius may be infinite, which is the conservative outcome
    prescribed by the upstream split-MFCS implementation.
    """

    scores = np.asarray(calibration_scores, dtype=np.float64)
    query_matrix = np.asarray(query_values, dtype=np.float64)
    _validate_inputs(scores, query_matrix, alpha=alpha, depth=depth)

    upstream = load_upstream_mfcs(source_file)
    weights = np.asarray(
        upstream.compute_w_ptest_split_active_replacement(
            query_matrix.copy(), depth
        ),
        dtype=np.float64,
    ).reshape(-1)
    if weights.shape != (scores.size + 1,):
        raise RuntimeError(
            "upstream MFCS returned an unexpected number of calibration/test weights"
        )
    total_weight = float(weights.sum())
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or total_weight <= 0.0:
        raise RuntimeError("upstream MFCS returned invalid replacement weights")

    normalized_weights = weights / total_weight
    score_distribution = np.concatenate((scores, np.array([np.inf])))
    radius = float(
        upstream.weighted_quantile(
            score_distribution,
            normalized_weights,
            1.0 - alpha,
        )
    )
    return NativeMFCSCalibration(
        radius=radius,
        unnormalized_weights=weights,
        normalized_weights=normalized_weights,
        depth=depth,
    )


@lru_cache(maxsize=None)
def load_upstream_mfcs(source_file: Path = DEFAULT_MFCS_SOURCE) -> ModuleType:
    """Load the pinned vendored MFCS module without modifying upstream files."""

    resolved_source = Path(source_file).resolve()
    if not resolved_source.is_file():
        raise FileNotFoundError(
            "vendored MFCS source is missing; expected "
            f"{resolved_source} at revision {UPSTREAM_MFCS_REVISION}"
        )
    source_digest = hashlib.sha256(resolved_source.read_bytes()).hexdigest()
    if source_digest != UPSTREAM_MFCS_SOURCE_SHA256:
        raise RuntimeError(
            "vendored MFCS source differs from the pinned upstream file at "
            f"revision {UPSTREAM_MFCS_REVISION}: {resolved_source}"
        )
    spec = importlib.util.spec_from_file_location(
        f"_scpcp_conformal_mfcs_{abs(hash(resolved_source))}",
        resolved_source,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load vendored MFCS source: {resolved_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "compute_w_ptest_split_active_replacement",
        "weighted_quantile",
    ):
        if not callable(getattr(module, name, None)):
            raise ImportError(f"vendored MFCS source does not define callable {name}")
    return module


def _validate_inputs(
    scores: NDArray[np.float64],
    query_matrix: NDArray[np.float64],
    *,
    alpha: float,
    depth: int,
) -> None:
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("calibration_scores must be a nonempty one-dimensional array")
    if not np.isfinite(scores).all() or np.any(scores < 0.0):
        raise ValueError("calibration_scores must be finite and nonnegative")
    if query_matrix.ndim != 2 or query_matrix.shape[1] != scores.size + 1:
        raise ValueError("query_values must have shape [history, N + 1]")
    if not np.isfinite(query_matrix).all() or np.any(query_matrix < 0.0):
        raise ValueError("query_values must be finite and nonnegative")
    if np.any(query_matrix.sum(axis=1) <= 0.0):
        raise ValueError("every query-history row must have positive mass")
    if not isinstance(depth, int) or not 1 <= depth <= query_matrix.shape[0]:
        raise ValueError("depth must be an integer in [1, history]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
