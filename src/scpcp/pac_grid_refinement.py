"""Nested certification grids for the retired PAC/LCB SC-PCP audit.

The construction changes only discretization.  It preserves every original
candidate exactly and inserts deterministic geometric knots between adjacent
positive scales.  Because the grid is frozen from ``D_COT`` before ``D_cert``
is read, it can be used by the same widest-to-narrowest fixed-sequence test
without changing its pointwise confidence level.
"""

from __future__ import annotations

import torch
from torch import Tensor


def nested_geometric_grid(
    base_grid: Tensor,
    *,
    subdivisions: int,
) -> tuple[Tensor, Tensor]:
    """Return a nested positive grid and the exact locations of base knots.

    ``subdivisions`` is the number of equal log-scale intervals replacing each
    adjacent pair.  For example, four subdivisions map a 101-point base grid
    to 401 points.  Original tensor values are copied rather than reconstructed
    through ``exp(log(x))``, so indexing the returned grid at ``base_indices``
    is bitwise equal to ``base_grid``.
    """

    if base_grid.ndim != 1 or len(base_grid) < 2:
        raise ValueError("base_grid must be one-dimensional with at least two points")
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    if not torch.isfinite(base_grid).all() or (base_grid <= 0.0).any():
        raise ValueError("base_grid must be finite and positive")
    if (base_grid[1:] < base_grid[:-1]).any():
        raise ValueError("base_grid must be nondecreasing")

    pieces: list[Tensor] = []
    fractions = torch.arange(
        1,
        subdivisions,
        device=base_grid.device,
        dtype=base_grid.dtype,
    ) / subdivisions
    for left, right in zip(base_grid[:-1], base_grid[1:], strict=True):
        pieces.append(left[None])
        if subdivisions > 1:
            ratio = right / left
            pieces.append(left * ratio.pow(fractions))
    pieces.append(base_grid[-1:])
    dense_grid = torch.cat(pieces)
    base_indices = torch.arange(
        0,
        len(dense_grid),
        subdivisions,
        device=base_grid.device,
        dtype=torch.long,
    )

    if not torch.equal(dense_grid[base_indices], base_grid):
        raise RuntimeError("nested grid failed to preserve the base candidates")
    return dense_grid, base_indices
