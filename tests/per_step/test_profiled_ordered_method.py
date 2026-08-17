from __future__ import annotations

import math

import pytest
import torch

import scpcp.experiment as experiment
from scpcp.baselines import finite_depth_mfcs_selection, standard_cp_stagewise_radii
from scpcp.certification import (
    CertificationResult,
    ordered_pointwise_bootstrap_lower_bounds,
    ordered_pointwise_ht_lower_bounds,
)
from scpcp.config import (
    COTConfig,
    CertificationConfig,
    ExperimentConfig,
    ModelConfig,
    ProfileConfig,
    SyntheticConfig,
)
from scpcp.cot import QConditionalCOT, _pseudo_target
from scpcp.coverage import (
    candidate_radius_schedules,
    diagonal_coverage_estimates,
    profiled_local_scale_grid,
    profiled_scale_grid,
    stage_score_profile,
    transport_refined_stage_profile,
    weighted_stage_score_quantiles,
)
from scpcp.data import TrajectoryBatch
from scpcp.outcome_model import GaussianOutcomeModel
from scpcp.selection import select_ordered_lcb_radius
from scpcp.simulator import TabularBehaviorPolicy, TabularTreatmentEnvironment


class _UniformLoggingPolicy:
    def probabilities(self, states: torch.Tensor) -> torch.Tensor:
        return torch.full((len(states), 2), 0.5, device=states.device)


class _RecordingPolicy:
    def __init__(self) -> None:
        self.radii: list[torch.Tensor] = []

    def probabilities(self, states: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        self.radii.append(torch.as_tensor(q).detach().cpu().clone())
        return torch.full((len(states), 2), 0.5, device=states.device)

    def probabilities_for_grid(
        self,
        states: torch.Tensor,
        q_grid: torch.Tensor,
    ) -> torch.Tensor:
        self.radii.append(q_grid.detach().cpu().clone())
        return torch.full(
            (len(states), len(q_grid), 2),
            0.5,
            device=states.device,
        )


class _RecordingTabularPolicy:
    n_actions = 3

    def __init__(self) -> None:
        self.radii: list[float] = []

    def probabilities(
        self,
        states: torch.Tensor,
        q: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q is not None:
            self.radii.append(float(torch.as_tensor(q).item()))
        value = 0.0 if q is None else float(torch.as_tensor(q).item())
        logits = torch.tensor([0.0, 0.1 * value, -0.1 * value], device=states.device)
        return torch.softmax(logits, dim=0)[None, :].expand(len(states), -1)


def _batch(n: int = 6, horizon: int = 2, state_dim: int = 2) -> TrajectoryBatch:
    return TrajectoryBatch(
        states=torch.randn(n, horizon + 1, state_dim),
        actions=torch.zeros(n, horizon, dtype=torch.long),
        outcomes=torch.randn(n, horizon, 2),
        patient_ids=torch.arange(n),
    )


def _batch_with_patient_ids(patient_ids: torch.Tensor) -> TrajectoryBatch:
    n = len(patient_ids)
    states = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3, 1)
    return TrajectoryBatch(
        states=states,
        actions=torch.zeros(n, 2, dtype=torch.long),
        outcomes=torch.zeros(n, 2, 2),
        patient_ids=patient_ids,
    )


def test_patient_crossfit_is_disjoint_exhaustive_and_deterministic() -> None:
    batch = _batch_with_patient_ids(
        torch.tensor([10, 10, 20, 30, 30, 40, 50, 60, 60])
    )

    first = experiment._patient_crossfit_indices(batch, folds=3, seed=17)
    second = experiment._patient_crossfit_indices(batch, folds=3, seed=17)
    all_rows = set(range(batch.n))
    held_row_sets = []
    held_patient_counts = []

    assert len(first) == 3
    for (train, held), (repeated_train, repeated_held) in zip(first, second, strict=True):
        train_rows = set(train.tolist())
        held_rows = set(held.tolist())
        train_patients = set(batch.patient_ids[train].tolist())
        held_patients = set(batch.patient_ids[held].tolist())

        assert torch.equal(train, repeated_train)
        assert torch.equal(held, repeated_held)
        assert train.device == batch.patient_ids.device
        assert held.device == batch.patient_ids.device
        assert train_rows.isdisjoint(held_rows)
        assert train_rows | held_rows == all_rows
        assert train_patients.isdisjoint(held_patients)
        held_row_sets.append(held_rows)
        held_patient_counts.append(len(held_patients))

    assert set().union(*held_row_sets) == all_rows
    assert all(
        first_rows.isdisjoint(second_rows)
        for index, first_rows in enumerate(held_row_sets)
        for second_rows in held_row_sets[index + 1 :]
    )
    assert max(held_patient_counts) - min(held_patient_counts) <= 1
    for patient in torch.unique(batch.patient_ids):
        patient_rows = set((batch.patient_ids == patient).nonzero().squeeze(1).tolist())
        assert sum(patient_rows <= held_rows for held_rows in held_row_sets) == 1


def test_patient_crossfit_requires_enough_patients_for_every_fold() -> None:
    batch = _batch_with_patient_ids(torch.tensor([10, 10, 20]))

    with pytest.raises(ValueError, match="one patient per fold"):
        experiment._patient_crossfit_indices(batch, folds=3, seed=5)
    with pytest.raises(ValueError, match="one patient per fold"):
        experiment._patient_crossfit_indices(batch, folds=1, seed=5)


def test_profile_configuration_rejects_invalid_cap_gate_and_focus_window() -> None:
    with pytest.raises(ValueError, match="maximum_cap_hit_rate"):
        ExperimentConfig(profile=ProfileConfig(maximum_cap_hit_rate=1.0)).validate()
    with pytest.raises(ValueError, match="grid focus"):
        ExperimentConfig(
            profile=ProfileConfig(grid_focus_radius=0.25),
            q_quantile_min=0.5,
            q_quantile_max=0.9,
        ).validate()


def test_profile_is_positive_unit_geometric_mean_and_frozen_grid_is_stagewise() -> None:
    scores = torch.tensor(
        [
            [0.3, 1.0, 4.0],
            [0.5, 1.2, 3.0],
            [0.7, 1.4, 5.0],
            [0.9, 1.6, 6.0],
            [1.1, 1.8, 7.0],
        ]
    )
    profile = stage_score_profile(scores, alpha=0.2)
    scale_grid = profiled_scale_grid(
        scores,
        profile,
        size=7,
        lower_quantile=0.5,
        upper_quantile=0.99,
    )
    schedules = candidate_radius_schedules(scale_grid, profile)

    assert torch.all(profile > 0.0)
    assert torch.allclose(profile.log().mean(), torch.tensor(0.0), atol=1e-6)
    assert schedules.shape == (7, 3)
    assert torch.allclose(schedules, scale_grid[:, None] * profile[None, :])


def test_weighted_stage_quantiles_use_hajek_left_rule_and_are_scale_invariant() -> None:
    scores = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 7.0],
            [4.0, 9.0],
            [8.0, 27.0],
        ]
    )
    weights = torch.tensor(
        [
            [1.0, 7.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [7.0, 1.0],
        ]
    )
    permutation = torch.tensor([2, 0, 3, 1])

    observed = weighted_stage_score_quantiles(scores, weights, alpha=0.25)
    rescaled = weighted_stage_score_quantiles(
        scores,
        weights * torch.tensor([13.0, 0.25]),
        alpha=0.25,
    )
    permuted = weighted_stage_score_quantiles(
        scores[permutation],
        weights[permutation],
        alpha=0.25,
    )
    exact_boundary = weighted_stage_score_quantiles(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([[3.0], [1.0], [0.0]]),
        alpha=0.25,
    )

    assert torch.equal(observed, torch.tensor([8.0, 7.0]))
    assert torch.equal(rescaled, observed)
    assert torch.equal(permuted, observed)
    # The target mass is exactly the first cumulative weight, so the inclusive
    # left quantile must return the first score rather than stepping right.
    assert torch.equal(exact_boundary, torch.tensor([1.0]))


def test_weighted_stage_quantiles_reject_invalid_scores_and_weights() -> None:
    scores = torch.tensor([[0.2, 0.4], [0.5, 0.8]])

    with pytest.raises(ValueError, match="positive total weight"):
        weighted_stage_score_quantiles(
            scores,
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            alpha=0.1,
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        weighted_stage_score_quantiles(
            scores,
            torch.tensor([[1.0, -0.1], [1.0, 1.0]]),
            alpha=0.1,
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        weighted_stage_score_quantiles(
            torch.tensor([[0.2, float("nan")], [0.5, 0.8]]),
            torch.ones(2, 2),
            alpha=0.1,
        )


def test_transport_refinement_reduces_frozen_pilot_coverage_imbalance() -> None:
    base = torch.arange(1, 11, dtype=torch.float32) / 10.0
    scores = torch.stack((base, 2.0 * base), dim=1)
    initial_quantiles = torch.ones(2)
    fold_initial = torch.ones(3, 2)
    fold_transported = torch.tensor([[0.8, 1.6]]).expand(3, -1).clone()
    fold_effective_sizes = torch.full((3, 2), 100.0)

    refined, applied = transport_refined_stage_profile(
        initial_quantiles,
        fold_initial,
        fold_transported,
        fold_effective_sizes,
        refinement_strength=0.5,
        maximum_profile_ratio=1.25,
        minimum_effective_size=25.0,
    )
    scale = 1.3
    initial_profile = initial_quantiles / initial_quantiles.log().mean().exp()
    initial_coverage = (scores <= scale * initial_profile).float().mean(dim=0)
    refined_coverage = (scores <= scale * refined).float().mean(dim=0)

    assert refined_coverage.mean() - refined_coverage.min() < (
        initial_coverage.mean() - initial_coverage.min()
    )
    assert refined_coverage.min() >= initial_coverage.min()
    assert applied[0] < applied[1]


def test_transport_refinement_uses_ess_weighted_oof_log_corrections() -> None:
    initial_quantiles = torch.tensor([2.0, 8.0])
    fold_initial = torch.ones(3, 2)
    fold_transported = torch.tensor(
        [
            [1.20, 0.90],
            [0.80, 1.10],
            [9.00, 9.00],
        ]
    )
    effective_sizes = torch.tensor(
        [
            [100.0, 80.0],
            [50.0, 20.0],
            [10.0, 10.0],
        ]
    )

    refined, applied = transport_refined_stage_profile(
        initial_quantiles,
        fold_initial,
        fold_transported,
        effective_sizes,
        refinement_strength=0.5,
        maximum_profile_ratio=10.0,
        minimum_effective_size=25.0,
    )
    mean_correction = torch.tensor(
        [
            (100.0 * math.log(1.20) + 50.0 * math.log(0.80)) / 150.0,
            math.log(0.90),
        ]
    )
    expected_applied = 0.5 * (mean_correction - mean_correction.mean())
    expected_profile = initial_quantiles * expected_applied.exp()
    expected_profile = expected_profile / expected_profile.log().mean().exp()

    assert torch.allclose(applied, expected_applied, atol=1e-6)
    assert torch.allclose(refined, expected_profile, atol=1e-6)
    # The very large third-fold correction and the second-stage fold with ESS
    # below threshold must have no influence.
    assert applied[0] > applied[1]


def test_transport_refinement_preserves_geometry_and_caps_correction_span() -> None:
    initial_quantiles = torch.tensor([0.5, 2.0, 8.0])
    fold_initial = torch.ones(3, 3)
    fold_transported = torch.tensor([[100.0, 0.01, 1.0]]).expand(3, -1).clone()
    effective_sizes = torch.full((3, 3), 100.0)

    first_profile, first_applied = transport_refined_stage_profile(
        initial_quantiles,
        fold_initial,
        fold_transported,
        effective_sizes,
        refinement_strength=1.0,
        maximum_profile_ratio=1.25,
        minimum_effective_size=25.0,
    )
    second_profile, second_applied = transport_refined_stage_profile(
        initial_quantiles,
        fold_initial,
        fold_transported,
        effective_sizes,
        refinement_strength=1.0,
        maximum_profile_ratio=1.25,
        minimum_effective_size=25.0,
    )
    initial_profile = initial_quantiles / initial_quantiles.log().mean().exp()
    correction = first_profile / initial_profile

    assert torch.equal(first_profile, second_profile)
    assert torch.equal(first_applied, second_applied)
    assert torch.isfinite(first_profile).all()
    assert torch.all(first_profile > 0.0)
    assert torch.allclose(first_profile.log().mean(), torch.tensor(0.0), atol=1e-6)
    assert float(correction.max() / correction.min()) <= 1.25 + 1e-6
    assert float((first_applied.max() - first_applied.min()).exp()) <= 1.25 + 1e-6


def test_transport_refinement_is_identity_without_eligible_fold_evidence() -> None:
    initial_quantiles = torch.tensor([0.5, 2.0])
    refined, applied = transport_refined_stage_profile(
        initial_quantiles,
        torch.ones(3, 2),
        torch.tensor([[10.0, 0.1]]).expand(3, -1),
        torch.full((3, 2), 5.0),
        refinement_strength=0.5,
        maximum_profile_ratio=1.25,
        minimum_effective_size=25.0,
    )
    expected = initial_quantiles / initial_quantiles.log().mean().exp()

    assert torch.equal(applied, torch.zeros(2))
    assert torch.allclose(refined, expected, atol=1e-7)


def test_transport_refinement_handles_one_stage_and_rejects_invalid_controls() -> None:
    refined, applied = transport_refined_stage_profile(
        torch.tensor([2.0]),
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([[4.0], [5.0], [6.0]]),
        torch.full((3, 1), 100.0),
        refinement_strength=0.5,
        maximum_profile_ratio=1.25,
        minimum_effective_size=25.0,
    )

    assert torch.equal(refined, torch.ones(1))
    assert torch.equal(applied, torch.zeros(1))
    with pytest.raises(ValueError, match="refinement_strength"):
        transport_refined_stage_profile(
            torch.ones(2),
            torch.ones(3, 2),
            torch.ones(3, 2),
            torch.ones(3, 2),
            refinement_strength=0.0,
            maximum_profile_ratio=1.25,
            minimum_effective_size=25.0,
        )
    with pytest.raises(ValueError, match="profile ratio"):
        transport_refined_stage_profile(
            torch.ones(2),
            torch.ones(3, 2),
            torch.ones(3, 2),
            torch.ones(3, 2),
            refinement_strength=0.5,
            maximum_profile_ratio=1.0,
            minimum_effective_size=25.0,
        )


def test_focused_grid_preserves_broad_guards_and_concentrates_near_anchor() -> None:
    scores = torch.linspace(0.001, 1.0, 1_000)[:, None]
    profile = torch.ones(1)
    broad = profiled_scale_grid(
        scores,
        profile,
        size=101,
        lower_quantile=0.5,
        upper_quantile=0.999,
    )

    focused = profiled_local_scale_grid(
        scores,
        profile,
        size=101,
        lower_quantile=0.5,
        upper_quantile=0.999,
        anchor_scale=0.75,
        focus_fraction=0.8,
        focus_radius=0.075,
    )
    repeated = profiled_local_scale_grid(
        scores,
        profile,
        size=101,
        lower_quantile=0.5,
        upper_quantile=0.999,
        anchor_scale=0.75,
        focus_fraction=0.8,
        focus_radius=0.075,
    )
    focus_limits = torch.quantile(scores[:, 0], torch.tensor([0.675, 0.825]))
    broad_focus_count = ((broad >= focus_limits[0]) & (broad <= focus_limits[1])).sum()
    focused_focus_count = (
        (focused >= focus_limits[0]) & (focused <= focus_limits[1])
    ).sum()

    assert focused.shape == (101,)
    assert torch.equal(focused, repeated)
    assert torch.isfinite(focused).all()
    assert torch.all(focused[1:] >= focused[:-1])
    assert torch.allclose(focused[[0, -1]], broad[[0, -1]], atol=1e-7)
    assert focused_focus_count > broad_focus_count
    assert int(focused_focus_count.item()) >= 79


def test_focused_grid_handles_boundary_anchors_three_points_and_ties() -> None:
    scores = torch.linspace(1.0, 2.0, 200)[:, None]
    profile = torch.ones(1)

    low = profiled_local_scale_grid(
        scores,
        profile,
        size=3,
        lower_quantile=0.5,
        upper_quantile=0.99,
        anchor_scale=0.1,
        focus_fraction=0.8,
        focus_radius=0.075,
    )
    high = profiled_local_scale_grid(
        scores,
        profile,
        size=3,
        lower_quantile=0.5,
        upper_quantile=0.99,
        anchor_scale=10.0,
        focus_fraction=0.8,
        focus_radius=0.075,
    )
    tied = profiled_local_scale_grid(
        torch.full((20, 3), 2.0),
        torch.ones(3),
        size=101,
        lower_quantile=0.5,
        upper_quantile=0.999,
        anchor_scale=2.0,
        focus_fraction=0.8,
        focus_radius=0.075,
    )
    broad = profiled_scale_grid(
        scores,
        profile,
        size=3,
        lower_quantile=0.5,
        upper_quantile=0.99,
    )

    for grid in (low, high):
        assert grid.shape == (3,)
        assert torch.isfinite(grid).all()
        assert torch.all(grid[1:] >= grid[:-1])
        assert torch.allclose(grid[[0, -1]], broad[[0, -1]], atol=1e-7)
    assert tied.shape == (101,)
    assert torch.equal(tied, torch.full((101,), 2.0))


def test_focused_grid_rejects_invalid_focus_and_nonfinite_scores() -> None:
    scores = torch.tensor([[0.2], [0.4], [0.8]])
    common = {
        "size": 11,
        "lower_quantile": 0.5,
        "upper_quantile": 0.999,
        "anchor_scale": 0.5,
        "focus_fraction": 0.8,
        "focus_radius": 0.075,
    }

    with pytest.raises(ValueError, match="at least three"):
        profiled_local_scale_grid(scores, torch.ones(1), **(common | {"size": 2}))
    with pytest.raises(ValueError, match="focus settings"):
        profiled_local_scale_grid(
            scores,
            torch.ones(1),
            **(common | {"focus_fraction": 1.0}),
        )
    with pytest.raises(ValueError, match="too large"):
        profiled_local_scale_grid(
            scores,
            torch.ones(1),
            **(common | {"focus_radius": 0.3}),
        )
    with pytest.raises(ValueError, match="anchor_scale"):
        profiled_local_scale_grid(
            scores,
            torch.ones(1),
            **(common | {"anchor_scale": float("nan")}),
        )
    with pytest.raises(ValueError, match="finite"):
        profiled_local_scale_grid(
            torch.tensor([[0.2], [float("nan")], [0.8]]),
            torch.ones(1),
            **common,
        )


def test_stagewise_candidate_events_match_manual_comparison() -> None:
    scores = torch.tensor([[0.4, 1.4], [0.8, 0.6]])
    schedules = torch.tensor([[0.5, 0.5], [1.0, 1.0], [0.7, 1.5]])
    weights = torch.ones(2, 2, 3)

    observed = diagonal_coverage_estimates(weights, scores, schedules)
    expected = torch.stack(
        [(scores <= schedule[None, :]).float().mean(dim=0) for schedule in schedules]
    )

    assert torch.equal(observed, expected)


def test_ordered_pointwise_ht_margin_has_no_grid_or_horizon_penalty() -> None:
    config = CertificationConfig(alpha=0.1, delta=0.05, ratio_bound_source="none")
    small = ordered_pointwise_ht_lower_bounds(
        torch.full((1, 1), 0.99),
        n_trajectories=1_000,
        weight_cap=1.0,
        config=config,
    )
    large = ordered_pointwise_ht_lower_bounds(
        torch.full((101, 12), 0.99),
        n_trajectories=1_000,
        weight_cap=1.0,
        config=config,
    )
    expected = math.sqrt(math.log(1.0 / 0.05) / 2_000.0)

    assert math.isclose(small.sampling_margin, expected)
    assert math.isclose(large.sampling_margin, expected)


def test_ordered_selector_stops_at_first_failure_and_optimizes_only_prefix() -> None:
    certificate = CertificationResult(
        estimates=torch.ones(4, 2),
        lower_bounds=torch.tensor(
            [
                [0.99, 0.99],  # Individually passes, but lies after the stop.
                [0.89, 0.99],  # First failure in widest-to-narrowest order.
                [0.93, 0.92],
                [0.95, 0.94],
            ]
        ),
        sampling_margin=0.0,
        ratio_error_bound=torch.full((4, 2), float("nan")),
        formal=False,
        label="practical",
    )
    selection = select_ordered_lcb_radius(
        torch.tensor([0.7, 0.9, 1.1, 1.3]),
        certificate,
        alpha=0.1,
        widths=torch.tensor([0.1, 0.2, 0.8, 1.0]),
    )

    assert selection.certified_indices == (3, 2)
    assert selection.stopped_index == 1
    assert selection.index == 2


def test_marginal_bootstrap_does_not_change_when_unrelated_candidate_is_added() -> None:
    generator = torch.Generator().manual_seed(17)
    scores = torch.rand(80, 2, generator=generator)
    base_weights = torch.rand(80, 2, 2, generator=generator) + 0.2
    base_schedules = torch.tensor([[0.4, 0.5], [0.7, 0.8]])
    extra_weights = torch.cat(
        (base_weights, torch.rand(80, 2, 1, generator=generator) + 0.2),
        dim=2,
    )
    extra_schedules = torch.cat((base_schedules, torch.tensor([[0.9, 0.95]])), dim=0)

    base = ordered_pointwise_bootstrap_lower_bounds(
        base_weights,
        scores,
        base_schedules,
        lower_tail=0.05,
        n_resamples=100,
        seed=31,
    )
    extended = ordered_pointwise_bootstrap_lower_bounds(
        extra_weights,
        scores,
        extra_schedules,
        lower_tail=0.05,
        n_resamples=100,
        seed=31,
    )

    assert torch.allclose(base.estimates, extended.estimates[:2], atol=1e-6)
    assert torch.allclose(base.lower_bounds, extended.lower_bounds[:2], atol=1e-6)
    assert "ordered_iut_marginal" in base.label


def test_cot_pseudo_target_uses_current_stage_profile_radius() -> None:
    batch = _batch(n=3, horizon=2, state_dim=2)
    outcome = GaussianOutcomeModel(
        state_dim=2,
        n_actions=2,
        config=ModelConfig(hidden_dim=8, representation_dim=4),
    )
    scales = torch.tensor([1.0, 2.0])
    profile = torch.tensor([0.5, 2.0])
    cot = QConditionalCOT(
        state_dim=2,
        horizon=2,
        outcome_model=outcome,
        q_grid=scales,
        stage_profile=profile,
        config=COTConfig(hidden_dims=(8,), rho_cap=4.0),
    )
    target = _RecordingPolicy()
    values = torch.tensor([1.0, 1.5, 2.0])

    _pseudo_target(
        cot,
        batch,
        1,
        torch.arange(3),
        values,
        target,
        _UniformLoggingPolicy(),
        cot.config,
    )

    assert torch.equal(target.radii[-1], values * profile[1])


def test_mfcs_uses_profile_radius_at_each_history_stage() -> None:
    batch = _batch(n=6, horizon=2, state_dim=2)
    scores = torch.rand(6, 2)
    scales = torch.tensor([1.0, 2.0])
    profile = torch.tensor([0.5, 2.0])
    target = _RecordingPolicy()

    finite_depth_mfcs_selection(
        batch,
        scores,
        q_grid=scales,
        stage_profile=profile,
        target_policy=target,
        logging_policy=_UniformLoggingPolicy(),
        depth=2,
        alpha=0.1,
        weight_cap=10.0,
    )

    assert any(torch.equal(radius, scales * profile[0]) for radius in target.radii)
    assert any(torch.equal(radius, scales * profile[1]) for radius in target.radii)


def test_exact_tabular_occupancy_uses_the_full_prefix_schedule() -> None:
    environment = TabularTreatmentEnvironment(SyntheticConfig(feedback_strength=1.0))
    target = _RecordingTabularPolicy()
    schedules = torch.tensor([[0.2, 0.4, 0.8], [1.0, 2.0, 3.0]])

    ratios = environment.exact_state_ratios(
        target,
        TabularBehaviorPolicy(),
        schedules,
        horizon=3,
        device="cpu",
    )

    assert ratios.shape == (2, 3, environment.n_states)
    assert torch.allclose(torch.tensor(target.radii), torch.tensor([0.2, 0.4, 1.0, 2.0]))


def test_standard_cp_returns_the_finite_sample_radius_for_each_stage() -> None:
    scores = torch.tensor(
        [[0.1, 1.0], [0.2, 0.9], [0.3, 0.8], [0.4, 0.7], [0.5, 0.6]]
    )

    radii = standard_cp_stagewise_radii(scores, alpha=0.4)

    assert torch.equal(radii, torch.tensor([0.4, 0.9]))
