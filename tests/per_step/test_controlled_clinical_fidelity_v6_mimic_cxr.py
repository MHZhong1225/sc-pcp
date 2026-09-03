from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scpcp.controlled_clinical_fidelity_v5_mimic_cxr import C13_STATE_KERNEL
from scpcp.controlled_clinical_fidelity_v6_mimic_cxr import (
    CANDIDATE_ID,
    IndependentAudit,
    OXYGEN_FEATURE_INDICES,
    build_b02_regression_anchor,
    build_terminal_environment,
    load_fidelity_v6_config,
    outcome0_design,
    outcome1_design,
    successor_clinical_features,
    terminal_candidate,
)
from scpcp.data import TrajectoryBatch


ROOT = Path(__file__).resolve().parents[2]


class TinyOutcomeModel(torch.nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(state_dim + 3, 2)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()

    def representation(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, :8]

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one_hot = torch.nn.functional.one_hot(action.to(torch.long), 3).to(state)
        mean = self.linear(torch.cat((state, one_hot), dim=1))
        return mean, torch.full_like(mean, 0.8)


def _feature_names() -> tuple[str, ...]:
    names = [f"feature_{index}" for index in range(153)]
    names[44:48] = ["rr_last", "rr_mean", "rr_min", "rr_max"]
    names[48:52] = ["spo2_last", "spo2_mean", "spo2_min", "spo2_max"]
    return tuple(names)


def _batch() -> tuple[TrajectoryBatch, TinyOutcomeModel, torch.Tensor]:
    generator = torch.Generator().manual_seed(601)
    n, horizon, state_dim = 36, 6, 153
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
        [
            successor_clinical_features(states[:, stage + 1], _feature_names())
            for stage in range(horizon)
        ],
        dim=1,
    )
    oxygen = clinical[..., OXYGEN_FEATURE_INDICES]
    supported = actions.eq(1).to(states) + 1.5 * actions.eq(2).to(states)
    stage_scale = torch.arange(1, 7, dtype=states.dtype)[None, :]
    outcome0 = (
        0.02 * clinical[..., :4].sum(dim=2)
        + 0.04 * clinical[..., 8:12].sum(dim=2)
        + 0.01 * stage_scale * supported * oxygen.sum(dim=2)
    )
    outcome1 = (
        0.03 * clinical[..., 4:8].sum(dim=2)
        + 0.05 * clinical[..., 12:16].sum(dim=2)
        + 0.1 * actions.to(states)
        + 0.02 * stage_scale
    )
    batch = TrajectoryBatch(
        states=states,
        actions=actions,
        outcomes=torch.stack((outcome0, outcome1), dim=2),
        patient_ids=torch.arange(n),
    )
    model = TinyOutcomeModel(state_dim)
    difficulty = torch.linspace(0.0, 1.0, n * horizon).reshape(n, horizon)
    return batch, model, difficulty


def _environments():
    batch, model, difficulty = _batch()
    arguments = {
        "outcome_model": model,
        "n_actions": 3,
        "difficulty": difficulty,
        "history_length": 1,
        "static_indices": (),
        "state_feature_names": _feature_names(),
    }
    terminal = build_terminal_environment(
        batch, theta=terminal_candidate(), **arguments
    )
    anchor = build_b02_regression_anchor(batch, **arguments)
    return batch, terminal, anchor


def test_config_is_terminal_and_has_one_candidate_with_explicit_lock() -> None:
    config = load_fidelity_v6_config(
        ROOT / "configs/controlled_clinical_fidelity_v6_mimic_cxr.yaml"
    )
    assert config.development_lineages == {
        "v5_development": tuple(range(92_600, 92_800, 10)),
        "v5_failed_confirmation": tuple(range(119_000, 119_200, 10)),
    }
    assert config.confirmation_seeds == tuple(range(120_000, 120_200, 10))
    assert config.confirmation_bootstrap_seed == 12_000_019
    pending = replace(
        config,
        independent_audit=IndependentAudit(
            status="PENDING",
            attestation_sha256=None,
            expected_prior_count=None,
            expected_prior_sha256=None,
            expected_artifact_count=None,
            expected_artifact_sha256=None,
            expected_source_count=None,
            expected_source_sha256=None,
            expected_v6_source_contract_sha256=None,
        ),
    )
    assert pending.independent_audit.permits_formal_launch is False
    with pytest.raises(RuntimeError, match="independent audit"):
        pending.validate(require_audit_go=True)
    assert terminal_candidate().candidate_id == CANDIDATE_ID


def test_outcome_designs_have_exact_width_order_and_interactions() -> None:
    clinical = torch.arange(48, dtype=torch.float64).reshape(3, 16)
    action = torch.tensor([0, 1, 2])
    design0 = outcome0_design(clinical, action)
    assert design0.shape == (3, 36)
    assert torch.equal(design0[:, 0], torch.ones(3, dtype=torch.float64))
    assert torch.equal(design0[:, 1:17], clinical)
    assert torch.equal(design0[:, 17:20], torch.eye(3, dtype=torch.float64))
    oxygen = clinical[:, OXYGEN_FEATURE_INDICES]
    assert torch.equal(design0[0, 20:], torch.zeros(16, dtype=torch.float64))
    assert torch.equal(design0[1, 20:28], oxygen[1])
    assert torch.equal(design0[1, 28:], torch.zeros(8, dtype=torch.float64))
    assert torch.equal(design0[2, 20:28], torch.zeros(8, dtype=torch.float64))
    assert torch.equal(design0[2, 28:], oxygen[2])

    design1 = outcome1_design(clinical, action, stage=4)
    assert design1.shape == (3, 26)
    assert torch.equal(design1[:, :20], design0[:, :20])
    assert torch.equal(design1[:, 20:26], torch.eye(6)[4].repeat(3, 1))


def test_outcome1_is_exact_b02_and_outcome0_has_six_stage_models() -> None:
    batch, terminal, anchor = _environments()
    assert len(terminal._outcome0_coefficients) == 6
    assert all(value.shape == (36, 1) for value in terminal._outcome0_coefficients)
    assert terminal._outcome1_coefficient.shape == (26, 1)
    assert torch.equal(
        terminal._outcome1_coefficient,
        anchor._bridge_coefficients[0][:, 1:2],
    )
    for stage in range(6):
        frame = batch.states[:, stage + 1]
        action = batch.actions[:, stage]
        terminal_mean = terminal._bridge_mean(frame, action, stage=stage)
        anchor_mean = anchor._bridge_mean(frame, action, stage=stage)
        assert torch.allclose(
            terminal_mean[:, 1], anchor_mean[:, 1], rtol=0.0, atol=1e-6
        )


def test_terminal_bridge_changes_only_joint_outcome_residual_library() -> None:
    _, terminal, anchor = _environments()
    assert terminal.neighbors == anchor.neighbors == 10_000
    assert terminal.donor_weighting == anchor.donor_weighting == "uniform"
    assert terminal.ridge_mode == anchor.ridge_mode == "sample_normalized_no_intercept"
    assert terminal.representation_geometry == anchor.representation_geometry == "raw"
    assert C13_STATE_KERNEL["transition_mode"] == "ridge_residual"
    for key in anchor._libraries:
        for index in (0, 1, 3, 4):
            assert torch.equal(
                anchor._libraries[key][index], terminal._libraries[key][index]
            )
        assert terminal._libraries[key][2].shape[1] == 2
    for left, right in zip(anchor._models, terminal._models, strict=True):
        assert torch.equal(left.coefficients, right.coefficients)


def test_rollout_adds_one_intact_two_outcome_residual_pair() -> None:
    batch, terminal, _ = _environments()
    state = batch.states[:12, 0]
    action = batch.actions[:12, 0]
    uniform = torch.linspace(0.01, 0.99, len(state))
    next_state, outcome, _, _, _ = terminal.step_from_uniform(
        state,
        action,
        uniform,
        time=0,
        gamma=0.0,
        action_coordinate=torch.tensor([-1.0, 0.0, 1.0]),
    )
    next_frame = next_state.reshape(len(state), 1, -1)[:, -1]
    innovation = outcome - terminal._bridge_mean(next_frame, action, stage=0)
    for row, action_value in enumerate(action.tolist()):
        library = terminal._libraries[(0, action_value)][2]
        assert any(
            torch.allclose(
                innovation[row], candidate.to(innovation), rtol=0.0, atol=1e-6
            )
            for candidate in library
        )


def test_formal_cache_feature_schema_matches_v6_design() -> None:
    stored = torch.load(
        ROOT / "data/real_cache/per_step_v17_mimic_cxr_h6_n60000_c271828.pt",
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
