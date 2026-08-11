"""Finite-grid self-consistent radius selection."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from scpcp.certification import CertificationResult


@dataclass(frozen=True)
class RadiusSelection:
    radius: float | None
    index: int | None
    status: str


def select_certified_radius(q_grid: Tensor, certificate: CertificationResult, *, alpha: float) -> RadiusSelection:
    """Select only when the supplied LCB has a stated formal premise."""

    if not certificate.formal:
        return RadiusSelection(radius=None, index=None, status="UNCERTIFIED_NO_RATIO_BOUND")
    return select_lcb_radius(q_grid, certificate, alpha=alpha, status="CERTIFIED")


def select_lcb_radius(
    q_grid: Tensor,
    certificate: CertificationResult,
    *,
    alpha: float,
    status: str | None = None,
) -> RadiusSelection:
    """Enumerate a lower-bound-safe grid point without monotonicity assumptions.

    This is useful for an explicitly labelled practical analysis when no
    external L1 ratio bound is available.  It must not be reported as a formal
    deployment certificate in that case.
    """

    target = 1.0 - alpha
    safe = (certificate.aggregate_lower_bound >= target).nonzero().squeeze(1)
    if len(safe) == 0:
        return RadiusSelection(radius=None, index=None, status="UNCERTIFIED")
    chosen = int(safe[0].item())
    selection_status = status or ("CERTIFIED" if certificate.formal else "PRACTICAL_LCB")
    return RadiusSelection(radius=float(q_grid[chosen].item()), index=chosen, status=selection_status)


def select_empirical_radius(q_grid: Tensor, estimates: Tensor, *, alpha: float) -> RadiusSelection:
    """Select from an empirical reference surface without an LCB claim."""

    target = 1.0 - alpha
    safe = (estimates.amin(dim=1) >= target).nonzero().squeeze(1)
    if len(safe) == 0:
        return RadiusSelection(radius=None, index=None, status="UNAVAILABLE_EMPIRICAL_REFERENCE")
    chosen = int(safe[0].item())
    return RadiusSelection(radius=float(q_grid[chosen].item()), index=chosen, status="EMPIRICAL_REFERENCE")
