"""Regression checks for source-faithful baseline boundaries."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest
import torch
from types import SimpleNamespace

import aci as aci
from baselines import standard_cp_stagewise_radii
from native_mfcs import DEFAULT_MFCS_SOURCE, native_mfcs_split_radius
import native_prc as native_prc
from native_prc import NativePRCConfig, native_prc_profile_scale, verify_official_prc_source
from native_spci import (
    NativeSPCIConfig,
    NativeSPCIUnavailable,
    run_native_spci,
    verify_native_spci_runtime,
    verify_official_spci_source,
)


def _require_private_upstream(path: Path) -> None:
    if not path.exists():
        pytest.skip("pinned upstream baseline checkout is intentionally private")


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
    _require_private_upstream(DEFAULT_MFCS_SOURCE)
    result = native_mfcs_split_radius(
        np.array([0.2, 0.6]),
        np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]]),
        alpha=0.2,
        depth=2,
    )
    assert result.normalized_weights.shape == (3,)
    assert np.isclose(result.normalized_weights.sum(), 1.0)


def test_standard_cp_keeps_its_infinite_conformal_atom() -> None:
    radii = standard_cp_stagewise_radii(torch.tensor([[1.0], [2.0]]), alpha=0.01)
    assert torch.isinf(radii).all()


def test_pinned_upstream_sources_are_verified_and_spci_dependency_fails_closed() -> None:
    _require_private_upstream(Path("internal/baselines"))
    assert verify_official_prc_source().is_dir()
    assert verify_official_spci_source().is_dir()
    try:
        verify_native_spci_runtime()
    except NativeSPCIUnavailable:
        # A mismatched runtime is an explicit unavailable status, never an
        # invitation to execute the different installed implementation.
        pass


def test_native_spci_executes_the_verified_upstream_ellipsoid_code() -> None:
    _require_private_upstream(Path("internal/baselines/MultiDimSPCI"))
    try:
        verify_native_spci_runtime()
    except NativeSPCIUnavailable as error:
        pytest.skip(str(error))
    rng = np.random.default_rng(11)
    x_train = rng.normal(size=(32, 3))
    y_train = np.column_stack((x_train[:, 0], x_train[:, 1])) + rng.normal(scale=0.1, size=(32, 2))
    x_target = rng.normal(size=(6, 3))
    y_target = np.column_stack((x_target[:, 0], x_target[:, 1])) + rng.normal(scale=0.1, size=(6, 2))
    result = run_native_spci(
        x_train, x_target, y_train, y_target, alpha=0.2, seed=3,
        config=NativeSPCIConfig(bootstrap_models=2, past_window=3, bins=2, qrf_estimators=2, qrf_max_depth=2),
        upstream_root=Path("internal/baselines/MultiDimSPCI"),
    )
    assert result.covered.shape == (6,)
    assert result.ellipsoid_volumes.shape == (6,)


def test_native_prc_keeps_selection_and_tracking_cohorts_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_private_upstream(Path("internal/baselines/Performative Risk Control"))
    def fake_run(*args: object) -> SimpleNamespace:
        splitter, simulator = args[1], args[4]
        shifted = simulator.simulate_shift(None, 1.0)
        selection, tracking = splitter(shifted)
        assert len(selection[0]) == len(tracking[0]) == 2
        assert not np.shares_memory(selection[0], tracking[0])
        return SimpleNamespace(
            lambda_hat=np.array(0.5), lambdas=np.array([1.0, 0.5]),
            risks_tt=np.array([0.0]), risks_tm1_t=np.array([0.0]), guaranteed_T=1,
            delta_lambda=0.5,
        )

    class Width:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    class MeanRisk:
        pass

    monkeypatch.setattr(native_prc, "verify_official_prc_source", lambda *_args: native_prc.official_prc_root())
    monkeypatch.setattr(native_prc, "_load_official_prc", lambda *_args: (fake_run, Width, MeanRisk))
    monkeypatch.setattr(
        native_prc,
        "score_batch",
        lambda *_args: torch.zeros((4, 1)),
    )
    batch = SimpleNamespace(
        current_states=lambda: torch.empty(4, 1, 1),
        actions=torch.empty(4, 1),
        outcomes=torch.empty(4, 1, 1),
    )
    result = native_prc_profile_scale(
        object(), object(), object(), torch.tensor([1.0, 2.0]), torch.tensor([1.0]),
        config=NativePRCConfig(alpha=0.2, delta=0.1, tightness=0.1, tau=1.0, cohort_size=2),
        horizon=1, seed=7, device="cpu", rollout_fn=lambda *_args, **_kwargs: batch,
    )
    assert result.target_deployments == 4
    assert result.selected_scale == 1.5
