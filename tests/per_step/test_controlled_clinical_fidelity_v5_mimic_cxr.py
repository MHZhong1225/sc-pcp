from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scpcp.controlled_clinical_fidelity_v5_mimic_cxr import (
    BRIDGE_RIDGE,
    CANONICAL_ACTION_COUNT,
    C13_STATE_KERNEL,
    BridgeTheta,
    build_cxr_environment,
    bridge_candidates,
    load_fidelity_v5_config,
    outcome_feature_groups,
    select_bridge_candidate,
    successor_clinical_features,
    summarize_candidate,
)
from scpcp.controlled_transition import ControlledResidualEnvironment
from scpcp.data import TrajectoryBatch


ROOT = Path(__file__).resolve().parents[2]


class TinyOutcomeModel(torch.nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(state_dim + CANONICAL_ACTION_COUNT, 2)
        torch.nn.init.uniform_(self.linear.weight, -0.02, 0.02)
        torch.nn.init.zeros_(self.linear.bias)

    def representation(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, :8]

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one_hot = torch.nn.functional.one_hot(
            action.to(torch.long), CANONICAL_ACTION_COUNT
        ).to(state)
        mean = self.linear(torch.cat((state, one_hot), dim=1))
        return mean, torch.full_like(mean, 0.8)


def _feature_names() -> tuple[str, ...]:
    names = [f"feature_{index}" for index in range(153)]
    names[44:48] = ["rr_last", "rr_mean", "rr_min", "rr_max"]
    names[48:52] = ["spo2_last", "spo2_mean", "spo2_min", "spo2_max"]
    return tuple(names)


def _batch() -> tuple[TrajectoryBatch, TinyOutcomeModel, torch.Tensor]:
    generator = torch.Generator().manual_seed(71)
    n, horizon, state_dim = 18, 2, 153
    states = torch.randn((n, horizon + 1, state_dim), generator=generator)
    states[:, :, 44:48] = 18.0 + 8.0 * torch.rand(
        (n, horizon + 1, 4), generator=generator
    )
    states[:, :, 48:52] = 88.0 + 10.0 * torch.rand(
        (n, horizon + 1, 4), generator=generator
    )
    actions = torch.tensor(
        [[(row + stage) % 3 for stage in range(horizon)] for row in range(n)],
        dtype=torch.long,
    )
    clinical = torch.stack(
        [successor_clinical_features(states[:, stage + 1], _feature_names()) for stage in range(horizon)],
        dim=1,
    )
    action_effect = torch.nn.functional.one_hot(actions, 3).to(states)[..., :2]
    outcomes = 0.03 * clinical[..., :2] + 0.4 * clinical[..., 8:10] + action_effect
    batch = TrajectoryBatch(
        states=states,
        actions=actions,
        outcomes=outcomes,
        patient_ids=torch.arange(n),
    )
    model = TinyOutcomeModel(state_dim)
    difficulty = torch.linspace(0.0, 1.0, n * horizon).reshape(n, horizon)
    return batch, model, difficulty


def _environment(theta: BridgeTheta) -> ControlledResidualEnvironment:
    batch, model, difficulty = _batch()
    return build_cxr_environment(
        batch,
        theta=theta,
        outcome_model=model,
        n_actions=3,
        difficulty=difficulty,
        history_length=1,
        static_indices=(),
        state_feature_names=_feature_names(),
    )


def test_contract_is_isolated_and_independent_audit_is_attested() -> None:
    config = load_fidelity_v5_config(
        ROOT / "configs/controlled_clinical_fidelity_v5_mimic_cxr.yaml"
    )
    assert config.development_seeds == tuple(range(92_600, 92_800, 10))
    assert config.confirmation_seeds == tuple(range(119_000, 119_200, 10))
    assert config.independent_audit.status == "GO"
    assert config.independent_audit.permits_formal_launch is True
    config.validate(require_audit_go=True)


def test_candidate_order_and_fixed_kernel_are_exact() -> None:
    candidates = bridge_candidates()
    assert [candidate.candidate_id for candidate in candidates] == [
        "B00_exact_c13_anchor",
        "B01_stagewise_successor_bridge",
        "B02_pooled_successor_bridge_stage_one_hot",
    ]
    assert all(candidate.to_dict()["state_kernel"] == C13_STATE_KERNEL for candidate in candidates)
    with pytest.raises(ValueError, match="candidate ID"):
        BridgeTheta("wrong", "exact_c13_anchor")


def test_successor_features_have_frozen_order_and_hinges() -> None:
    frame = torch.zeros((1, 153))
    frame[0, 48:52] = torch.tensor([90.0, 92.0, 94.0, 82.0])
    frame[0, 44:48] = torch.tensor([20.0, 22.0, 37.0, 7.0])
    value = successor_clinical_features(frame, _feature_names())
    assert value.shape == (1, 16)
    assert torch.equal(value[0, :8], torch.tensor([90, 92, 94, 82, 20, 22, 37, 7.0]))
    assert torch.equal(value[0, 8:12], torch.tensor([0.2, 0.0, 0.0, 1.0]))
    assert torch.equal(value[0, 12:], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert outcome_feature_groups() == (
        (0, 1, 2, 3, 8, 9, 10, 11),
        (4, 5, 6, 7, 12, 13, 14, 15),
    )


def test_formal_n60000_cache_has_the_frozen_feature_schema() -> None:
    stored = torch.load(
        ROOT
        / "data/real_cache/per_step_v17_mimic_cxr_h6_n60000_c271828.pt",
        map_location="cpu",
        weights_only=False,
    )
    names = tuple(stored["state_feature_names"])
    assert stored["states"].shape[-1] == len(names) == 153
    assert names[44:48] == ("rr_last", "rr_mean", "rr_min", "rr_max")
    assert names[48:52] == (
        "spo2_last",
        "spo2_mean",
        "spo2_min",
        "spo2_max",
    )


def test_anchor_factory_is_the_exact_c13_environment() -> None:
    batch, model, difficulty = _batch()
    expected = ControlledResidualEnvironment(
        batch,
        outcome_model=model,
        n_actions=3,
        difficulty=difficulty,
        history_length=1,
        static_indices=(),
        state_feature_names=_feature_names(),
        neighbors=10_000,
        bandwidth=2.0,
        ridge=1e-3,
        representation_geometry="raw",
        donor_weighting="uniform",
        ridge_mode="sample_normalized_no_intercept",
        transition_mode="ridge_residual",
        outcome_residual_mode="raw",
    )
    observed = build_cxr_environment(
        batch,
        theta=bridge_candidates()[0],
        outcome_model=model,
        n_actions=3,
        difficulty=difficulty,
        history_length=1,
        static_indices=(),
        state_feature_names=_feature_names(),
    )
    assert type(observed) is ControlledResidualEnvironment
    assert observed.neighbors == expected.neighbors == 10_000
    for key in expected._libraries:
        for left, right in zip(expected._libraries[key], observed._libraries[key], strict=True):
            assert torch.equal(left, right)
    for left, right in zip(expected._models, observed._models, strict=True):
        assert torch.equal(left.coefficients, right.coefficients)


def test_bridges_change_only_joint_outcome_innovation_not_state_kernel() -> None:
    anchor, stagewise, pooled = (_environment(theta) for theta in bridge_candidates())
    for bridge in (stagewise, pooled):
        assert bridge.neighbors == 10_000
        assert bridge.donor_weighting == "uniform"
        assert bridge.ridge_mode == "sample_normalized_no_intercept"
        for key in anchor._libraries:
            for index in (0, 1, 3, 4):
                assert torch.equal(anchor._libraries[key][index], bridge._libraries[key][index])
        for left, right in zip(anchor._models, bridge._models, strict=True):
            assert torch.equal(left.coefficients, right.coefficients)

    assert len(stagewise._bridge_coefficients) == 2
    assert stagewise._bridge_coefficients[0].shape == (1 + 16 + 3, 2)
    assert len(pooled._bridge_coefficients) == 1
    assert pooled._bridge_coefficients[0].shape == (1 + 16 + 3 + 2, 2)
    assert BRIDGE_RIDGE == 1e-3

    batch, _, _ = _batch()
    state = batch.states[:9, 0]
    action = batch.actions[:9, 0]
    uniform = torch.linspace(0.02, 0.98, len(state))
    coordinate = torch.tensor([-1.0, 0.0, 1.0])
    anchor_result = anchor.step_from_uniform(
        state, action, uniform, time=0, gamma=0.0, action_coordinate=coordinate
    )
    for bridge in (stagewise, pooled):
        result = bridge.step_from_uniform(
            state, action, uniform, time=0, gamma=0.0, action_coordinate=coordinate
        )
        assert torch.equal(anchor_result[0], result[0])
        assert torch.equal(anchor_result[2], result[2])
        assert torch.equal(anchor_result[3], result[3])
        assert torch.equal(anchor_result[4], result[4])


def test_selector_uses_all_twenty_seeds_and_prespecified_order() -> None:
    candidates = bridge_candidates()
    summaries = {}
    pass_counts = (18, 19, 19)
    means = (0.7, 0.8, 0.75)
    for theta, pass_count, mean in zip(candidates, pass_counts, means, strict=True):
        metrics = []
        for index in range(20):
            ratio = mean if index < pass_count else 1.2
            metrics.append(
                {
                    "maximum_score_ks": 0.1 * ratio,
                    "maximum_signed_residual_w1": 0.1,
                    "maximum_successor_mean_w1": 0.1,
                    "maximum_successor_q95_w1": 0.1,
                    "structural_invariants": True,
                }
            )
        summaries[theta.candidate_id] = summarize_candidate(theta, metrics)
    selection = select_bridge_candidate(candidates, summaries)
    assert selection["development_admissible"] is True
    assert selection["winner"]["candidate_id"] == "B02_pooled_successor_bridge_stage_one_hot"
    assert selection["winner_summary"]["pass_count"] == 19
    assert selection["winner_summary"]["structural_pass_count"] == 20
    assert selection["candidate_seed_deletions"] == 0


def test_bridge_requires_three_actions_and_exact_feature_names() -> None:
    batch, model, difficulty = _batch()
    with pytest.raises(ValueError, match="three canonical actions"):
        build_cxr_environment(
            replace(batch, actions=batch.actions.clamp_max(1)),
            theta=bridge_candidates()[1],
            outcome_model=model,
            n_actions=2,
            difficulty=difficulty,
            history_length=1,
            static_indices=(),
            state_feature_names=_feature_names(),
        )
    names = list(_feature_names())
    names[48] = "wrong"
    with pytest.raises(ValueError, match="indices/names"):
        successor_clinical_features(batch.states[:, 1], names)
