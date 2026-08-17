from __future__ import annotations

import pickle

import torch
import pytest

from scpcp.certification import (
    CertificationResult,
    practical_bootstrap_lower_bounds,
    simultaneous_lower_bounds,
)
from scpcp.behavior import BehaviorPolicy
from scpcp.config import COTConfig, CertificationConfig, ExperimentConfig, ModelConfig, PolicyConfig
from scpcp.cot import QConditionalCOT
from scpcp.cxr import _CXRDataset
from scpcp.coverage import (
    effective_sample_sizes,
    fixed_q_grid,
    self_normalized_diagonal_coverage_estimates,
)
from scpcp.data import TrajectoryBatch, concatenate_trajectories, patient_level_splits
from scpcp.experiment import SCPCP_METHOD
from scpcp.outcome_model import GaussianOutcomeModel
from scpcp.policy.anchored import _ratio_capped_tilt
from scpcp.real_data import (
    CLINICAL_STATE_DEFAULTS,
    _RawClinicalBatch,
    _append_decision_time,
    _assemble_raw,
    _burden_outcomes_from_events,
    _clinical_lab_kind,
    _coarsen_cxr_actions,
    _eicu_fluid_rows,
    _eicu_urine_rows,
    _interval_treatment_grid,
    _merge_rare_actions,
    _respiratory_action_grid,
    _static_context,
)
from scpcp.selection import select_certified_radius, select_lcb_radius
from scpcp.simulator import (
    EmpiricalRolloutContext,
    EmpiricalTransitionEnvironment,
    rollout,
)


def test_final_method_has_one_public_name() -> None:
    assert SCPCP_METHOD == "SC-PCP"


def test_policy_ratio_cap_preserves_mass() -> None:
    reference = torch.softmax(torch.randn(16, 4), dim=1)
    costs = torch.randn(16, 4) * 5.0
    cap = 1.25

    probabilities = _ratio_capped_tilt(reference, costs, tilt=1.0, cap=cap)

    assert torch.allclose(probabilities.sum(dim=1), torch.ones(16), atol=1e-6)
    assert float((probabilities / reference).max()) <= cap + 1e-6


def test_behavior_policy_uses_explicit_stage_bias() -> None:
    policy = BehaviorPolicy(
        state_dim=3,
        n_actions=2,
        model=ModelConfig(hidden_dim=4),
        policy=PolicyConfig(),
        horizon=3,
        decision_time_index=2,
    )
    with torch.no_grad():
        for parameter in policy.network.parameters():
            parameter.zero_()
        policy.stage_bias.copy_(torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]]))
    states = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 / 3.0]])

    probabilities = policy.probabilities(states)

    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[1, 1] > probabilities[1, 0]


def test_tied_quantile_grid_keeps_prespecified_size_and_finite_q_features() -> None:
    scores = torch.ones(12, 3)
    grid = fixed_q_grid(scores, size=11, lower_quantile=0.5, upper_quantile=0.999)
    outcome = GaussianOutcomeModel(
        state_dim=2, n_actions=2, config=ModelConfig(hidden_dim=8, representation_dim=4)
    )
    cot = QConditionalCOT(
        state_dim=2,
        horizon=3,
        outcome_model=outcome,
        q_grid=grid,
        config=COTConfig(hidden_dims=(8,), rho_cap=2.0),
    )

    assert grid.shape == (11,)
    assert torch.isfinite(cot.q_scale)
    assert cot.rho_for_grid(1, torch.randn(5, 2), grid).shape == (5, 11)


def test_q_grid_inference_reuses_state_encoding_and_bounds_head_batches() -> None:
    model_config = ModelConfig(
        architecture="gru",
        history_length=4,
        hidden_dim=8,
        representation_dim=4,
    )
    outcome = GaussianOutcomeModel(state_dim=12, n_actions=2, config=model_config)
    q_grid = torch.linspace(0.5, 2.0, 11)
    cot = QConditionalCOT(
        state_dim=12,
        horizon=3,
        outcome_model=outcome,
        q_grid=q_grid,
        config=COTConfig(hidden_dims=(8,), batch_size=7, rho_cap=4.0),
    )
    torch.manual_seed(5)
    with torch.no_grad():
        for parameter in cot.heads[0].parameters():
            parameter.uniform_(-0.1, 0.1)
    states = torch.randn(23, 12)
    encoded_batch_sizes = []
    head_batch_sizes = []
    encoder_hook = outcome.state_encoder.register_forward_pre_hook(
        lambda _module, inputs: encoded_batch_sizes.append(len(inputs[0]))
    )
    head_hook = cot.heads[0].register_forward_pre_hook(
        lambda _module, inputs: head_batch_sizes.append(len(inputs[0]))
    )

    grid_values = cot.rho_for_grid(1, states, q_grid)
    encoder_hook.remove()
    head_hook.remove()
    expected = torch.stack(
        [cot.rho(1, states, radius.expand(len(states))) for radius in q_grid],
        dim=1,
    )

    assert grid_values.shape == (23, 11)
    assert encoded_batch_sizes == [7, 7, 7, 2]
    assert max(head_batch_sizes) <= cot.config.batch_size
    assert torch.allclose(grid_values, expected, atol=1e-6)


def test_cxr_dataset_is_safe_inside_spawned_gpu_workers() -> None:
    dataset = _CXRDataset(
        ["unused.jpg"],
        torch.zeros((1, 14)),
        torch.tensor([0]),
    )

    pickle.dumps(dataset)


def test_no_ratio_bound_cannot_be_reported_as_certified() -> None:
    estimates = torch.full((3, 2), 0.99)
    result = simultaneous_lower_bounds(
        estimates,
        n_trajectories=100_000,
        weight_cap=1.0,
        config=CertificationConfig(alpha=0.1, delta=0.05, ratio_bound_source="none"),
    )
    grid = torch.tensor([0.5, 1.0, 1.5])

    certified = select_certified_radius(grid, result, alpha=0.1)
    practical = select_lcb_radius(grid, result, alpha=0.1)

    assert certified.radius is None
    assert certified.status == "UNCERTIFIED_NO_RATIO_BOUND"
    assert practical.status == "PRACTICAL_LCB"
    assert result.label == "raw_ht_sampling_only_no_transport_bound"
    assert torch.isnan(result.ratio_error_bound).all()


def test_oracle_ratio_source_is_internal_exact_tabular_only() -> None:
    oracle = CertificationConfig(alpha=0.1, delta=0.05, ratio_bound_source="oracle")
    estimates = torch.full((2, 2), 0.99)

    with pytest.raises(ValueError, match="internal exact-tabular"):
        ExperimentConfig(certification=oracle).validate()
    with pytest.raises(ValueError, match="exact-tabular validation"):
        simultaneous_lower_bounds(
            estimates, n_trajectories=100_000, weight_cap=1.0, config=oracle
        )

    internal = simultaneous_lower_bounds(
        estimates,
        n_trajectories=100_000,
        weight_cap=1.0,
        config=oracle,
        allow_oracle=True,
    )
    assert internal.formal
    assert internal.label == "oracle_ratio_bound"


def test_declared_ratio_bound_requires_its_own_failure_budget() -> None:
    declared = CertificationConfig(alpha=0.1, delta=0.05, ratio_bound_source="declared")
    with pytest.raises(ValueError, match="positive ratio_delta"):
        ExperimentConfig(certification=declared).validate()
    with pytest.raises(ValueError, match="positive ratio_delta"):
        simultaneous_lower_bounds(
            torch.full((2, 2), 0.99),
            n_trajectories=100_000,
            weight_cap=1.0,
            config=declared,
        )


def test_practical_bootstrap_is_nonformal_and_guards_all_success_boundary() -> None:
    weights = torch.ones(40, 2, 3)
    scores = torch.full((40, 2), 0.5)
    result = practical_bootstrap_lower_bounds(
        weights,
        scores,
        torch.tensor([0.4, 0.5, 0.6]),
        lower_tail=0.05,
        n_resamples=50,
        seed=7,
    )

    assert result.lower_bounds.shape == (3, 2)
    assert torch.all(result.lower_bounds[2] < 1.0)
    assert torch.all(result.lower_bounds[2] > 0.0)
    assert not result.formal
    assert torch.isnan(result.ratio_error_bound).all()
    assert result.label == "practical_hajek_cluster_bootstrap_max_t_wilson_lcb"


def test_practical_coverage_is_invariant_to_weight_scale() -> None:
    scores = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    grid = torch.tensor([0.5, 0.9])
    weights = torch.tensor(
        [
            [[0.5, 2.0], [1.0, 3.0]],
            [[1.5, 4.0], [2.0, 1.0]],
            [[1.0, 5.0], [3.0, 2.0]],
        ]
    )

    original = self_normalized_diagonal_coverage_estimates(weights, scores, grid)
    rescaled = self_normalized_diagonal_coverage_estimates(weights * 7.0, scores, grid)

    assert torch.allclose(original, rescaled)
    assert torch.all((0.0 <= original) & (original <= 1.0))


def test_practical_bootstrap_is_invariant_to_per_cell_weight_scale() -> None:
    generator = torch.Generator().manual_seed(41)
    weights = torch.rand(60, 2, 3, generator=generator) + 0.2
    scores = torch.rand(60, 2, generator=generator)
    grid = torch.tensor([0.3, 0.6, 0.9])
    scale = torch.tensor([[0.5, 3.0, 7.0], [5.0, 0.8, 2.0]])

    original = practical_bootstrap_lower_bounds(
        weights,
        scores,
        grid,
        lower_tail=0.05,
        n_resamples=80,
        seed=17,
    )
    rescaled = practical_bootstrap_lower_bounds(
        weights * scale[None, :, :],
        scores,
        grid,
        lower_tail=0.05,
        n_resamples=80,
        seed=17,
    )

    assert torch.allclose(original.estimates, rescaled.estimates, atol=1e-6)
    assert torch.allclose(original.lower_bounds, rescaled.lower_bounds, atol=1e-6)
    assert torch.all(original.lower_bounds <= original.estimates)


def test_studentized_band_uses_wilson_guard_only_for_deterministic_safe_q() -> None:
    generator = torch.Generator().manual_seed(53)
    scores = 0.8 * torch.rand(80, 3, generator=generator)
    weights = torch.rand(80, 3, 2, generator=generator) + 0.1
    result = practical_bootstrap_lower_bounds(
        weights,
        scores,
        torch.tensor([0.35, 1.0]),
        lower_tail=0.05,
        n_resamples=100,
        seed=19,
    )

    assert torch.allclose(result.estimates[1], torch.ones(3))
    assert torch.all(result.lower_bounds[1] < 1.0)
    assert torch.all(result.lower_bounds[1] > 0.8)
    assert torch.any(result.lower_bounds[0] < result.estimates[0])


def test_cluster_bootstrap_is_invariant_to_duplicate_rows_within_cluster() -> None:
    generator = torch.Generator().manual_seed(71)
    weights = torch.rand(24, 2, 3, generator=generator) + 0.2
    scores = torch.rand(24, 2, generator=generator)
    q_grid = torch.tensor([0.3, 0.6, 0.9])
    base = practical_bootstrap_lower_bounds(
        weights,
        scores,
        q_grid,
        lower_tail=0.05,
        n_resamples=100,
        seed=29,
        cluster_ids=torch.arange(24),
    )
    duplicated = practical_bootstrap_lower_bounds(
        weights.repeat_interleave(2, dim=0),
        scores.repeat_interleave(2, dim=0),
        q_grid,
        lower_tail=0.05,
        n_resamples=100,
        seed=29,
        cluster_ids=torch.arange(24).repeat_interleave(2),
    )

    assert torch.allclose(base.estimates, duplicated.estimates, atol=1e-6)
    assert torch.allclose(base.lower_bounds, duplicated.lower_bounds, atol=1e-6)


def test_formal_margin_and_ess_use_independent_patient_clusters() -> None:
    estimates = torch.full((2, 2), 0.95)
    config = CertificationConfig(
        alpha=0.1,
        delta=0.05,
        ratio_error_bound=0.01,
        ratio_bound_source="declared",
        ratio_delta=0.01,
    )
    independent = simultaneous_lower_bounds(
        estimates,
        n_trajectories=4,
        weight_cap=1.0,
        config=config,
    )
    clustered = simultaneous_lower_bounds(
        estimates,
        n_trajectories=4,
        weight_cap=1.0,
        config=config,
        cluster_ids=torch.tensor([10, 10, 20, 30]),
    )
    weights = torch.ones(4, 2, 2)

    assert clustered.sampling_margin > independent.sampling_margin
    assert torch.all(
        effective_sample_sizes(weights, torch.tensor([10, 10, 20, 30])) < 4.0
    )


def test_cot_heads_start_at_identity_ratio_and_calibrate_to_mean_one() -> None:
    grid = torch.linspace(0.5, 1.5, 5)
    outcome = GaussianOutcomeModel(
        state_dim=3,
        n_actions=2,
        config=ModelConfig(hidden_dim=8, representation_dim=4),
    )
    cot = QConditionalCOT(
        state_dim=3,
        horizon=3,
        outcome_model=outcome,
        q_grid=grid,
        config=COTConfig(hidden_dims=(8,), rho_cap=2.0),
    )
    states = torch.randn(40, 3)

    assert torch.allclose(
        cot.rho_for_grid(1, states, grid), torch.ones(40, 5), atol=1e-6
    )
    with torch.no_grad():
        cot.heads[0].network[-1].bias.fill_(-1.0)
    cot.calibrate_head(1, states)

    calibrated = cot.rho_for_grid(1, states, grid)
    assert torch.allclose(calibrated.mean(dim=0), torch.ones(5), atol=1e-5)
    repeated_states = (
        states[:, None, :].expand(-1, len(grid), -1).reshape(-1, states.shape[1])
    )
    repeated_grid = grid.repeat(len(states))
    scalar_path = cot.rho(1, repeated_states, repeated_grid).reshape(
        len(states), len(grid)
    )
    assert torch.allclose(calibrated, scalar_path, atol=1e-6)


def test_grid_selection_enumerates_nonmonotone_safe_points() -> None:
    certificate = CertificationResult(
        estimates=torch.tensor([[0.93, 0.94], [0.70, 0.98], [0.96, 0.95]]),
        lower_bounds=torch.tensor([[0.91, 0.92], [0.69, 0.97], [0.91, 0.95]]),
        sampling_margin=0.0,
        ratio_error_bound=torch.zeros(3, 2),
        formal=True,
        label="oracle_ratio_bound",
    )
    selection = select_certified_radius(
        torch.tensor([0.7, 0.9, 1.2]), certificate, alpha=0.1
    )

    assert selection.index == 0
    assert selection.status == "CERTIFIED"


def test_empirical_environment_excludes_all_rows_from_current_donor_patient() -> None:
    # Patient 10 contributes two episodes.  Both must be excluded, leaving the
    # complete patient-20 transition unchanged, including its static coordinate.
    states = torch.tensor(
        [
            [[0.0, 0.0], [10.0, 100.0]],
            [[0.1, 0.1], [11.0, 110.0]],
            [[5.0, 5.0], [20.0, 200.0]],
        ]
    )
    batch = TrajectoryBatch(
        states=states,
        actions=torch.zeros(3, 1, dtype=torch.long),
        outcomes=torch.tensor([[[10.0, 0.0]], [[11.0, 0.0]], [[20.0, 0.0]]]),
        patient_ids=torch.tensor([10, 10, 20]),
    )
    environment = EmpiricalTransitionEnvironment(
        batch,
        n_actions=1,
        neighbors=3,
        bandwidth=1.0,
        embedding_dim=2,
        static_indices=(0,),
    )
    query = states[:1, 0]
    action = torch.zeros(1, dtype=torch.long)

    next_state, outcome, next_context = environment.step_with_context(
        query,
        action,
        torch.Generator().manual_seed(4),
        time=0,
        context=EmpiricalRolloutContext(torch.tensor([10])),
    )

    assert torch.equal(next_context.donor_patient_ids, torch.tensor([20]))
    assert torch.equal(next_state, states[2:3, 1])
    assert torch.equal(outcome, batch.outcomes[2:3, 0])
    assert next_state[0, 0] == 20.0


def test_empirical_environment_fits_pca_in_double_precision(monkeypatch) -> None:
    states = torch.randn(24, 2, 12, generator=torch.Generator().manual_seed(31))
    # Repeated history coordinates reproduce the rank deficiency seen in the
    # clinical state stacks without depending on a local dataset fixture.
    states[:, :, 6:] = states[:, :, :6]
    batch = TrajectoryBatch(
        states=states,
        actions=torch.zeros(24, 1, dtype=torch.long),
        outcomes=torch.zeros(24, 1, 2),
        patient_ids=torch.arange(24),
    )
    original_eigh = torch.linalg.eigh
    observed_dtypes: list[torch.dtype] = []

    def recording_eigh(matrix: torch.Tensor):
        observed_dtypes.append(matrix.dtype)
        return original_eigh(matrix)

    monkeypatch.setattr(torch.linalg, "eigh", recording_eigh)
    environment = EmpiricalTransitionEnvironment(
        batch,
        n_actions=1,
        neighbors=3,
        bandwidth=1.0,
        embedding_dim=8,
    )

    assert observed_dtypes == [torch.float64]
    assert environment.embedding.dtype == states.dtype
    assert torch.isfinite(environment.embedding).all()


def test_empirical_residual_bootstrap_recomposes_at_the_query_state() -> None:
    class _AffineOutcomeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("device_anchor", torch.zeros(()))

        def forward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            action = actions.to(states.dtype)
            mean = torch.stack(
                (states[:, 0] + 0.5 * action, states[:, 1] - action),
                dim=1,
            )
            scale = torch.stack(
                (1.0 + 0.1 * states[:, 0].abs(), 2.0 + 0.2 * states[:, 1].abs()),
                dim=1,
            )
            return mean, scale

    model = _AffineOutcomeModel()
    current = torch.tensor([[0.0, 0.0], [0.2, -0.1], [5.0, 3.0]])
    successor = torch.tensor([[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]])
    actions = torch.zeros(3, 1, dtype=torch.long)
    signed_residuals = torch.tensor([[0.5, -0.5], [0.25, 1.0], [-1.5, 0.25]])
    donor_mean, donor_scale = model(current, actions[:, 0])
    outcomes = (donor_mean + donor_scale * signed_residuals)[:, None, :]
    batch = TrajectoryBatch(
        states=torch.stack((current, successor), dim=1),
        actions=actions,
        outcomes=outcomes,
        patient_ids=torch.tensor([10, 10, 20]),
    )
    environment = EmpiricalTransitionEnvironment(
        batch,
        n_actions=1,
        neighbors=3,
        bandwidth=1.0,
        embedding_dim=2,
        outcome_model=model,
    )
    query = torch.tensor([[7.0, -4.0]])
    query_action = torch.zeros(1, dtype=torch.long)

    next_state, generated_outcome, next_context = environment.step_with_context(
        query,
        query_action,
        torch.Generator().manual_seed(8),
        time=0,
        context=EmpiricalRolloutContext(torch.tensor([10])),
    )
    query_mean, query_scale = model(query, query_action)
    expected = query_mean + query_scale * signed_residuals[2:3]

    assert torch.equal(next_context.donor_patient_ids, torch.tensor([20]))
    assert torch.equal(next_state, successor[2:3])
    assert torch.allclose(generated_outcome, expected, atol=1e-6)


def test_empirical_residual_bootstrap_preserves_signed_score_vector() -> None:
    class _LocationScaleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("device_anchor", torch.zeros(()))

        def forward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del actions
            mean = torch.stack((2.0 * states[:, 0], -states[:, 0]), dim=1)
            scale = torch.stack(
                (1.0 + states[:, 0].abs(), 0.5 + 0.5 * states[:, 0].abs()),
                dim=1,
            )
            return mean, scale

    model = _LocationScaleModel()
    current = torch.tensor([[0.0], [1.0]])
    successor = torch.tensor([[0.5], [1.5]])
    actions = torch.zeros(2, 1, dtype=torch.long)
    signed_residuals = torch.tensor([[0.2, -0.4], [-1.25, 0.75]])
    donor_mean, donor_scale = model(current, actions[:, 0])
    batch = TrajectoryBatch(
        states=torch.stack((current, successor), dim=1),
        actions=actions,
        outcomes=(donor_mean + donor_scale * signed_residuals)[:, None, :],
        patient_ids=torch.tensor([10, 20]),
    )
    environment = EmpiricalTransitionEnvironment(
        batch,
        n_actions=1,
        neighbors=2,
        bandwidth=1.0,
        embedding_dim=1,
        outcome_model=model,
    )
    query = torch.tensor([[4.0]])
    query_action = torch.zeros(1, dtype=torch.long)

    _, generated_outcome, _ = environment.step_with_context(
        query,
        query_action,
        torch.Generator().manual_seed(9),
        time=0,
        context=EmpiricalRolloutContext(torch.tensor([10])),
    )
    query_mean, query_scale = model(query, query_action)
    generated_residual = (generated_outcome - query_mean) / query_scale

    assert torch.allclose(generated_residual, signed_residuals[1:2], atol=1e-6)
    assert torch.allclose(
        generated_residual.abs().amax(dim=1),
        signed_residuals[1:2].abs().amax(dim=1),
        atol=1e-6,
    )


def test_empirical_rollout_updates_donor_context_and_keeps_stages_separate() -> None:
    patient_ids = torch.tensor([10, 20, 30])
    states = torch.stack(
        [
            torch.tensor(
                [[float(patient), 0.0], [float(patient), 1.0], [float(patient), 2.0]]
            )
            for patient in patient_ids.tolist()
        ]
    )
    outcomes = torch.zeros(3, 2, 2)
    outcomes[:, 0, 0] = patient_ids
    outcomes[:, 1, 0] = 100 + patient_ids
    batch = TrajectoryBatch(
        states=states,
        actions=torch.zeros(3, 2, dtype=torch.long),
        outcomes=outcomes,
        patient_ids=patient_ids,
    )
    environment = EmpiricalTransitionEnvironment(
        batch,
        n_actions=1,
        neighbors=2,
        bandwidth=1.0,
        embedding_dim=2,
    )

    class _OnlyAction:
        n_actions = 1

        def probabilities(
            self, states: torch.Tensor, q: float | torch.Tensor | None = None
        ) -> torch.Tensor:
            return torch.ones(len(states), 1, device=states.device)

    generated = rollout(
        environment,
        _OnlyAction(),
        n=30,
        horizon=2,
        seed=4,
        device="cpu",
    )
    initial_donor = generated.states[:, 0, 0].to(torch.long)
    stage_zero_donor = generated.outcomes[:, 0, 0].to(torch.long)
    stage_one_donor = generated.outcomes[:, 1, 0].to(torch.long) - 100

    assert torch.all(stage_zero_donor != initial_donor)
    assert torch.all(stage_one_donor != stage_zero_donor)
    assert torch.all(generated.outcomes[:, 0, 0] < 100)
    assert torch.all(generated.outcomes[:, 1, 0] >= 100)


def test_empirical_environment_chunks_knn_without_changing_sampling(
    monkeypatch,
) -> None:
    generator = torch.Generator().manual_seed(19)
    current = torch.randn(64, 4, generator=generator)
    successor = current + 0.1 * torch.randn(64, 4, generator=generator)
    outcomes = torch.arange(128, dtype=torch.float32).reshape(64, 1, 2)
    batch = TrajectoryBatch(
        states=torch.stack((current, successor), dim=1),
        actions=torch.zeros(64, 1, dtype=torch.long),
        outcomes=outcomes,
        patient_ids=torch.arange(64),
    )
    environment = EmpiricalTransitionEnvironment(
        batch,
        n_actions=1,
        neighbors=5,
        bandwidth=0.7,
        embedding_dim=3,
        query_batch_size=7,
    )
    query_states = torch.randn(37, 4, generator=generator)
    query_actions = torch.zeros(37, dtype=torch.long)

    candidate_embedding, library_next, library_outcome, _ = (
        environment._library_on_device(0, query_states.device)
    )
    center, scale, embedding = environment._transforms_like(query_states)
    embedded_query = ((query_states - center) / scale).clamp(-10.0, 10.0) @ embedding
    full_distances, full_nearest = torch.cdist(
        embedded_query,
        candidate_embedding,
    ).topk(5, largest=False)
    local_scale = full_distances.median(dim=1).values.clamp_min(1e-6)
    standardized = full_distances / local_scale[:, None]
    probabilities = torch.softmax(-standardized.square() / (2.0 * 0.7**2), dim=1)
    expected_generator = torch.Generator().manual_seed(23)
    expected_draw = torch.multinomial(
        probabilities,
        1,
        generator=expected_generator,
    ).squeeze(1)
    expected_rows = full_nearest[torch.arange(len(query_states)), expected_draw]

    original_cdist = torch.cdist
    observed_batch_sizes = []

    def recording_cdist(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        observed_batch_sizes.append(len(left))
        return original_cdist(left, right)

    monkeypatch.setattr(torch, "cdist", recording_cdist)
    actual_next, actual_outcomes = environment.step(
        query_states,
        query_actions,
        torch.Generator().manual_seed(23),
    )

    assert max(observed_batch_sizes) == 7
    assert len(observed_batch_sizes) == 6
    assert torch.equal(actual_next, library_next[expected_rows])
    assert torch.equal(actual_outcomes, library_outcome[expected_rows])


def test_patient_split_is_disjoint_and_per_step() -> None:
    batch = TrajectoryBatch(
        states=torch.randn(20, 3, 4),
        actions=torch.zeros(20, 2, dtype=torch.long),
        outcomes=torch.randn(20, 2, 2),
        patient_ids=torch.arange(20),
    )
    split = patient_level_splits(batch, seed=3, include_environment=True)
    roles = [
        split.predictor,
        split.behavior,
        split.cot,
        split.certification,
        split.environment,
    ]
    ids = [set(role.patient_ids.tolist()) for role in roles if role is not None]

    assert all(
        first.isdisjoint(second)
        for index, first in enumerate(ids)
        for second in ids[index + 1 :]
    )
    assert not hasattr(batch, "running_max")


def test_clinical_split_reuses_predictor_role_for_propensity_training() -> None:
    batch = TrajectoryBatch(
        states=torch.randn(100, 3, 4),
        actions=torch.zeros(100, 2, dtype=torch.long),
        outcomes=torch.randn(100, 2, 2),
        patient_ids=torch.arange(100),
    )

    split = patient_level_splits(
        batch,
        seed=3,
        include_environment=True,
        include_behavior=False,
    )

    assert split.behavior is None
    assert (
        split.predictor.n,
        split.cot.n,
        split.certification.n,
        split.environment.n,
    ) == (40, 15, 30, 15)


def test_known_environment_split_reuses_the_unneeded_environment_share() -> None:
    batch = TrajectoryBatch(
        states=torch.randn(100, 3, 4),
        actions=torch.zeros(100, 2, dtype=torch.long),
        outcomes=torch.randn(100, 2, 2),
        patient_ids=torch.arange(100),
    )

    split = patient_level_splits(
        batch,
        seed=5,
        include_environment=False,
        include_behavior=False,
    )

    assert split.behavior is None
    assert (split.predictor.n, split.cot.n, split.certification.n) == (40, 20, 40)
    assert split.environment is None
    baseline_calibration = concatenate_trajectories(split.cot, split.certification)
    assert baseline_calibration.n == 60
    assert set(baseline_calibration.patient_ids.tolist()) == (
        set(split.cot.patient_ids.tolist())
        | set(split.certification.patient_ids.tolist())
    )


def test_clinical_decision_time_is_pre_action_and_normalized() -> None:
    states = torch.zeros(2, 4, 3)

    augmented = _append_decision_time(states)

    assert augmented.shape == (2, 4, 4)
    assert torch.allclose(
        augmented[:, :, -1],
        torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]).expand(2, -1),
    )


def test_direct_action_requires_a_stable_prespecified_ontology() -> None:
    actions = torch.tensor([0] * 80 + [1] * 20 + [2] * 20 + [3] * 2)
    with pytest.raises(RuntimeError, match="insufficient support"):
        _merge_rare_actions(
            actions,
            torch.arange(len(actions)),
            expected_actions=4,
            direct=True,
        )


def test_cxr_action_coarsening_is_fixed_before_patient_splits() -> None:
    raw = _RawClinicalBatch(
        states=torch.zeros((2, 3, 1)),
        outcomes=torch.zeros((2, 2, 2)),
        treatments=torch.zeros((2, 2, 2)),
        patient_ids=torch.tensor([1, 2]),
        episode_ids=torch.tensor([10, 20]),
        static_indices=(),
        direct_actions=torch.tensor([[0, 1], [2, 3]]),
        direct_action_count=4,
        original_to_direct_action={0: 0, 1: 1, 2: 2, 3: 3},
    )

    coarsened = _coarsen_cxr_actions(raw)

    assert coarsened.direct_actions.tolist() == [[0, 1], [1, 2]]
    assert coarsened.direct_action_count == 3
    assert coarsened.original_to_direct_action == {0: 0, 1: 1, 2: 1, 3: 2}


def test_respiratory_support_is_carried_forward_between_chart_events() -> None:
    import numpy as np
    import pandas as pd

    devices = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "minutes": [-10.0, 100.0, 50.0],
            "value": ["Nasal cannula", "Ventilator", "HFNC"],
        }
    )
    actions = _respiratory_action_grid(
        devices, np.array([1, 2]), horizon=3, interval=120, action_minutes=60
    )

    assert actions.tolist() == [[1, 3, 3], [2, 2, 2]]


def test_interval_treatment_uses_window_overlap_and_instantaneous_bolus() -> None:
    import numpy as np
    import pandas as pd

    events = pd.DataFrame(
        {
            "stay_id": [10, 10],
            "minutes": [30.0, 130.0],
            "end_minutes": [150.0, 130.0],
            "component": ["pressor", "fluid"],
            "value": [1.0, 500.0],
            "interval_kind": ["active_duration", "amount"],
        }
    )
    grid = _interval_treatment_grid(
        events, np.array([10]), "stay_id", horizon=2, interval=120, action_minutes=60
    )

    # Pressor overlaps 30 min with the first action half and the instantaneous
    # bolus is attributed to the second action half.
    assert grid.shape == (1, 2, 2)
    assert grid[0, 0, 1] == 30.0
    assert grid[0, 1, 0] == 500.0


def test_clinical_state_enrichment_is_shifted_past_the_action_window() -> None:
    import numpy as np
    import pandas as pd

    cohort = pd.DataFrame({"stay_id": [11], "patient_id": [7]})
    static = _static_context(
        pd.DataFrame({"age": [61.0]}, index=pd.Index([11], name="stay_id")),
        numeric=("age",),
    )
    # The minute-2 observations fall in A_0's treatment half-window, so they
    # cannot appear in S_0.  Once the interval is complete they are legitimate
    # history for S_1; the later minute-6 value must be the temporal ``last``.
    events = pd.DataFrame(
        {
            "stay_id": [11] * 11,
            "minutes": [2.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 16.0, 16.0, 16.0, 16.0],
            "kind": [
                "creatinine",
                "map",
                "hr",
                "rr",
                "spo2",
                "sbp",
                "creatinine",
                "map",
                "hr",
                "rr",
                "spo2",
            ],
            "value": [2.0, 75.0, 90.0, 19.0, 95.0, 118.0, 1.4, 76.0, 91.0, 20.0, 96.0],
        }
    )
    raw = _assemble_raw(
        cohort,
        id_column="stay_id",
        patient_column="patient_id",
        static=static,
        vitals=events,
        treatments=pd.DataFrame(columns=["stay_id", "minutes", "component", "value"]),
        horizon=2,
        interval_minutes=10,
        action_minutes=5,
        outcome_kind="hypo_tachy",
    )
    index = {name: position for position, name in enumerate(raw.state_feature_names)}

    assert (
        raw.states[0, 0, index["creatinine_last"]]
        == CLINICAL_STATE_DEFAULTS["creatinine"]
    )
    assert raw.states[0, 0, index["creatinine_missing"]] == 1.0
    assert raw.states[0, 1, index["creatinine_last"]] == pytest.approx(1.4)
    assert raw.states[0, 1, index["creatinine_mean"]] == pytest.approx(1.7)
    assert raw.states[0, 1, index["creatinine_min"]] == pytest.approx(1.4)
    assert raw.states[0, 1, index["creatinine_max"]] == pytest.approx(2.0)
    assert raw.states[0, 1, index["creatinine_missing"]] == 0.0
    assert raw.states[0, 1, index["sbp_last"]] == pytest.approx(118.0)
    assert np.isfinite(raw.states.numpy()).all()


def test_lab_name_mapping_excludes_non_equivalent_assays() -> None:
    import pandas as pd

    kinds = _clinical_lab_kind(
        pd.Series(
            [
                "Creatinine",
                "Lactate",
                "Lactate Dehydrogenase",
                "Platelet Count",
                "platelets x 1000",
                "urinary creatinine",
                "hb",
                "lacate",
            ]
        )
    )

    assert kinds.iloc[0] == "creatinine"
    assert kinds.iloc[1] == "lactate"
    assert pd.isna(kinds.iloc[2])
    assert kinds.iloc[3] == "platelets"
    assert kinds.iloc[4] == "platelets"
    assert pd.isna(kinds.iloc[5])
    assert kinds.iloc[6] == "hemoglobin"
    assert kinds.iloc[7] == "lactate"


def test_response_burden_applies_threshold_before_time_averaging() -> None:
    import numpy as np
    import pandas as pd

    events = pd.DataFrame(
        {
            "stay_id": [9, 9, 9, 9],
            "bin": [0, 0, 0, 0],
            "kind": ["map", "map", "hr", "hr"],
            "value": [50.0, 80.0, 90.0, 110.0],
        }
    )
    full = pd.MultiIndex.from_product([[9], [0]], names=["stay_id", "bin"])
    outcome = _burden_outcomes_from_events(
        events,
        full_index=full,
        valid_ids=np.array([9]),
        id_column="stay_id",
        horizon=1,
        kind="hypo_tachy",
    )

    assert outcome[0, 0, 0] == pytest.approx(0.5)
    assert outcome[0, 0, 1] == pytest.approx(0.125)


def test_static_context_collapses_rare_categories_before_patient_splits() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {"age": [60.0] * 100, "site": ["common"] * 98 + ["rare-a", "rare-b"]}
    )
    encoded = _static_context(frame, numeric=("age",), categorical=("site",))

    assert "site_common" in encoded
    assert "site___other__" in encoded
    assert "site_rare-a" not in encoded
    assert "site_rare-b" not in encoded


def test_eicu_fluid_filter_excludes_total_intake_carriers_and_flushes() -> None:
    import pandas as pd

    prefix = "flowsheet|Flowsheet Cell Labels|I&O|Intake (ml)|Crystalloids (ml)|"
    chunk = pd.DataFrame(
        {
            "patientunitstayid": [1, 1, 1, 1, 2, 1],
            "intakeoutputoffset": [10, 20, 30, 40, 10, 50],
            "cellpath": [
                prefix + value for value in ["NS", "flush", "drug", "albumin", "NS"]
            ]
            + ["flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine"],
            "celllabel": [
                "Volume (mL)-sodium chloride (NORMAL SALINE) 0.9 % bolus 500 mL",
                "Saline Flush (mL)",
                "Volume (mL)-vancomycin in sodium chloride 0.9% 250 mL IVPB",
                "Volume (mL)-albumin human 5 % injection",
                "Volume (mL)-normal saline bolus",
                "Urine",
            ],
            "cellvaluenumeric": [500.0, 10.0, 250.0, 100.0, 1000.0, 75.0],
        }
    )

    selected = _eicu_fluid_rows(chunk, {1})
    urine = _eicu_urine_rows(chunk, {1})

    assert selected.value.tolist() == [500.0, 100.0]
    assert urine.value.tolist() == [75.0]
