"""Shared result types and the one direct split-conformal baseline.

Article-specific baselines live in their own source-backed modules:
``aci``, ``native_mfcs``, ``native_prc``, and ``native_spci``. This module
deliberately contains no local
"style" controller or replacement approximation that could be reported under
one of those paper names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def standard_cp_stagewise_radii(scores: Tensor, alpha: float) -> Tensor:
    """Ordinary split-CP finite-sample radius at every decision stage."""

    if scores.ndim != 2 or len(scores) == 0:
        raise ValueError("scores must have shape [N, T] with N > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rank = math.ceil((scores.shape[0] + 1) * (1.0 - alpha))
    if rank > scores.shape[0]:
        # The standard split-CP convention assigns the remaining conformal
        # mass to +infinity.  Capping it at the largest observed score would
        # change the finite-sample construction.
        return torch.full(
            (scores.shape[1],),
            float("inf"),
            dtype=scores.dtype,
            device=scores.device,
        )
    return scores.sort(dim=0).values[rank - 1]


@dataclass(frozen=True)
class OnlineBaselineResult:
    """Common reporting shape for an online method's actual target stream."""

    radius_by_time: Tensor
    target_deployments: int
    rounds: int
    adaptation_per_time_coverage: Tensor
    adaptation_round_worst_coverage: tuple[float, ...]
    adaptation_pathwise_coverage: float
    selected_scale: float | None = None
