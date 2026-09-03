"""Regression checks for source-faithful baseline boundaries."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

import scpcp.aci as aci
from scpcp.native_mfcs import native_mfcs_split_radius
from scpcp.native_prc import verify_official_prc_source
from scpcp.native_spci import NativeSPCIUnavailable, verify_native_spci_runtime, verify_official_spci_source


def test_aci_uses_one_binary_update_per_arrival_without_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    radii: list[float] = []
    observed = iter((torch.tensor([[2.0]]), torch.tensor([[1.0]])))

    def fake_rollout(*_args: object, q: torch.Tensor, **_kwargs: object) -> object:
        radii.append(float(q.item()))
        return SimpleNamespace(
            current_states=lambda: torch.empty(1, 1, 1),
            actions=torch.empty(1, 1),
            outcomes=torch.empty(1, 1, 1),
        )

    monkeypatch.setattr(aci, "score_batch", lambda *_args: next(observed))
    result = aci.run_aci_panel(
        object(), object(), object(), torch.tensor([[1.0]]), alpha=0.5, gamma=0.25,
        target_deployments=2, horizon=1, seed=9, device="cpu", rollout_fn=fake_rollout,
    )
    # First error: .5 + .25(.5 - 1) = .375, so the second pre-arrival
    # conformal radius is the second order statistic of [1, 2], namely 2.
    assert radii == [1.0, 2.0]
    assert result.radius_by_time.tolist() == [1.0]
    assert result.rounds == 2


def test_aci_fails_instead_of_clipping_an_unrepresentable_exact_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aci, "score_batch", lambda *_args: torch.tensor([[2.0]]))
    with pytest.raises(aci.ACIInterfaceError, match="unbounded"):
        aci.run_aci_panel(
            object(), object(), object(), torch.tensor([[1.0]]), alpha=0.1, gamma=1.0,
            target_deployments=1, horizon=1, seed=9, device="cpu", rollout_fn=lambda *_args, **_kwargs: SimpleNamespace(
                current_states=lambda: torch.empty(1, 1, 1), actions=torch.empty(1, 1), outcomes=torch.empty(1, 1, 1)
            ),
        )


def test_native_mfcs_calls_the_pinned_replacement_recursion() -> None:
    result = native_mfcs_split_radius(
        np.array([0.2, 0.6]),
        np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]]),
        alpha=0.2,
        depth=2,
    )
    assert result.normalized_weights.shape == (3,)
    assert np.isclose(result.normalized_weights.sum(), 1.0)


def test_pinned_upstream_sources_are_verified_and_spci_dependency_fails_closed() -> None:
    assert verify_official_prc_source().is_dir()
    assert verify_official_spci_source().is_dir()
    try:
        verify_native_spci_runtime()
    except NativeSPCIUnavailable:
        # A mismatched runtime is an explicit unavailable status, never an
        # invitation to execute the different installed implementation.
        pass
