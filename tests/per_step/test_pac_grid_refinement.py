from __future__ import annotations

import pytest
import torch

from scpcp.pac_grid_refinement import nested_geometric_grid


def test_nested_geometric_grid_preserves_original_knots_exactly() -> None:
    base = torch.tensor([1.0, 4.0, 16.0], dtype=torch.float64)

    dense, indices = nested_geometric_grid(base, subdivisions=4)

    assert torch.equal(indices, torch.tensor([0, 4, 8]))
    assert torch.equal(dense[indices], base)
    assert torch.allclose(
        dense,
        torch.tensor(
            [
                1.0,
                4.0 ** 0.25,
                2.0,
                4.0 ** 0.75,
                4.0,
                4.0 * 4.0 ** 0.25,
                8.0,
                4.0 * 4.0 ** 0.75,
                16.0,
            ],
            dtype=torch.float64,
        ),
    )


def test_one_subdivision_is_identity() -> None:
    base = torch.tensor([0.8, 1.0, 1.4])

    dense, indices = nested_geometric_grid(base, subdivisions=1)

    assert torch.equal(dense, base)
    assert torch.equal(indices, torch.arange(len(base)))


def test_tied_base_knots_remain_nested_and_nondecreasing() -> None:
    base = torch.tensor([1.0, 1.0, 2.0])

    dense, indices = nested_geometric_grid(base, subdivisions=3)

    assert torch.equal(dense[indices], base)
    assert bool((dense[1:] >= dense[:-1]).all())


@pytest.mark.parametrize(
    ("base", "subdivisions"),
    [
        (torch.tensor([1.0]), 4),
        (torch.ones(2, 2), 4),
        (torch.tensor([1.0, float("nan")]), 4),
        (torch.tensor([0.0, 1.0]), 4),
        (torch.tensor([2.0, 1.0]), 4),
        (torch.tensor([1.0, 2.0]), 0),
    ],
)
def test_invalid_nested_grid_inputs_fail(base: torch.Tensor, subdivisions: int) -> None:
    with pytest.raises(ValueError):
        nested_geometric_grid(base, subdivisions=subdivisions)
