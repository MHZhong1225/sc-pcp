"""Strict bridge to the pinned official conformal-MFCS implementation.

The replacement recursion and weighted-quantile rule are loaded directly from
the upstream release.  This module only checks and reshapes the required
query-law matrix; it never estimates or clips MFCS weights itself.
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


UPSTREAM_MFCS_REPOSITORY = "https://github.com/drewprinster/conformal-mfcs"
UPSTREAM_MFCS_COMMIT = "c737536d874fda9f0da6dbc95a16a91c71df2512"
UPSTREAM_MFCS_LICENSE = "MIT"
UPSTREAM_MFCS_SOURCE_SHA256 = "766d54864467ce3db635a30da94e5284f1140fba7f5150ce93743896adb0e880"
DEFAULT_MFCS_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "internal"
    / "baselines"
    / "conformal-mfcs"
    / "calibrate_mfcs.py"
)


@dataclass(frozen=True)
class NativeMFCSCalibration:
    """One unmodified split-MFCS calibration result for one test covariate."""

    radius: float
    unnormalized_weights: NDArray[np.float64]
    normalized_weights: NDArray[np.float64]
    depth: int
    upstream_repository: str = UPSTREAM_MFCS_REPOSITORY
    upstream_commit: str = UPSTREAM_MFCS_COMMIT
    upstream_license: str = UPSTREAM_MFCS_LICENSE
    upstream_entrypoint: str = "calibrate_mfcs.compute_w_ptest_split_active_replacement"


def native_mfcs_split_radius(
    calibration_scores: ArrayLike,
    query_values: ArrayLike,
    *,
    alpha: float,
    depth: int,
    source_file: Path = DEFAULT_MFCS_SOURCE,
) -> NativeMFCSCalibration:
    """Call the official split-MFCS replacement recursion without alteration.

    ``query_values`` has the exact upstream shape ``[history, N + 1]``.  Each
    row contains the actual query-law values at every calibration covariate and
    the present test covariate.  The final ``+infinity`` score is intentional:
    it is part of the original weighted conformal construction.
    """

    scores = np.asarray(calibration_scores, dtype=np.float64)
    query_matrix = np.asarray(query_values, dtype=np.float64)
    _validate_inputs(scores, query_matrix, alpha=alpha, depth=depth)
    upstream = load_upstream_mfcs(source_file)
    weights = np.asarray(
        upstream.compute_w_ptest_split_active_replacement(query_matrix.copy(), depth),
        dtype=np.float64,
    ).reshape(-1)
    if weights.shape != (scores.size + 1,):
        raise RuntimeError("official MFCS returned an unexpected number of weights")
    total = float(weights.sum())
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or total <= 0.0:
        raise RuntimeError("official MFCS returned invalid replacement weights")
    normalized = weights / total
    radius = float(
        upstream.weighted_quantile(
            np.concatenate((scores, np.array([np.inf]))), normalized, 1.0 - alpha
        )
    )
    return NativeMFCSCalibration(radius, weights, normalized, depth)


@lru_cache(maxsize=None)
def load_upstream_mfcs(source_file: Path = DEFAULT_MFCS_SOURCE) -> ModuleType:
    """Load only the checksum-pinned upstream file, otherwise fail closed."""

    source = Path(source_file).resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"official MFCS source is missing: {source}; clone {UPSTREAM_MFCS_REPOSITORY} "
            f"at {UPSTREAM_MFCS_COMMIT}"
        )
    if hashlib.sha256(source.read_bytes()).hexdigest() != UPSTREAM_MFCS_SOURCE_SHA256:
        raise RuntimeError(
            f"official MFCS source differs from pinned commit {UPSTREAM_MFCS_COMMIT}: {source}"
        )
    spec = importlib.util.spec_from_file_location(f"_scpcp_mfcs_{abs(hash(source))}", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load official MFCS source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("compute_w_ptest_split_active_replacement", "weighted_quantile")
    if not all(callable(getattr(module, name, None)) for name in required):
        raise ImportError("pinned MFCS source does not expose its required entry points")
    return module


def _validate_inputs(
    scores: NDArray[np.float64], query_matrix: NDArray[np.float64], *, alpha: float, depth: int
) -> None:
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all() or np.any(scores < 0.0):
        raise ValueError("calibration_scores must be a nonempty finite nonnegative vector")
    if query_matrix.ndim != 2 or query_matrix.shape[1] != scores.size + 1:
        raise ValueError("query_values must have the official MFCS shape [history, N + 1]")
    if not np.isfinite(query_matrix).all() or np.any(query_matrix < 0.0):
        raise ValueError("query_values must be finite and nonnegative")
    if np.any(query_matrix.sum(axis=1) <= 0.0):
        raise ValueError("every MFCS query-history row must have positive mass")
    if not isinstance(depth, int) or not 1 <= depth <= query_matrix.shape[0]:
        raise ValueError("depth must be an integer in [1, history]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
