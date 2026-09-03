from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from scpcp.config import ModelConfig
from scpcp.controlled_transition import (
    ControlledResidualEnvironment,
    _lower_median_from_sorted_rows,
    make_controlled_noise,
    rollout_controlled,
)
from scpcp.data import TrajectoryBatch
from scpcp.outcome_model import GaussianOutcomeModel


class _AlternatingPolicy:
    n_actions = 2

    def probabilities(self, states: torch.Tensor, q: object = None) -> torch.Tensor:
        return states.new_tensor((0.35, 0.65)).expand(len(states), -1)


class _ConstantOutcomeModel(torch.nn.Module):
    """Minimal frozen model that makes every donor row equally local."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def representation(self, states: torch.Tensor) -> torch.Tensor:
        return states.new_zeros((len(states), 1)) + self.anchor

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del actions
        return states.new_zeros((len(states), 2)), states.new_ones((len(states), 2))


class _StateScaledOutcomeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def representation(self, states: torch.Tensor) -> torch.Tensor:
        return states[:, :1] + 0.0 * self.anchor

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del actions
        mean = states.new_zeros((len(states), 2)) + 0.0 * self.anchor
        scale = states[:, :1].abs().add(1.0).expand(-1, 2)
        return mean, scale


def _environment() -> ControlledResidualEnvironment:
    generator = torch.Generator().manual_seed(9)
    n, horizon, state_dim = 40, 3, 4
    states = torch.randn(n, horizon + 1, state_dim, generator=generator)
    states[:, :, 0] = torch.arange(n)[:, None]  # immutable patient context
    states[:, :, 3] = torch.arange(horizon + 1)[None] / horizon
    actions = torch.arange(n * horizon).reshape(n, horizon).remainder(2)
    outcomes = torch.randn(n, horizon, 2, generator=generator)
    batch = TrajectoryBatch(states, actions, outcomes, torch.arange(n))
    model = GaussianOutcomeModel(
        state_dim=state_dim,
        n_actions=2,
        config=ModelConfig(hidden_dim=8, representation_dim=4, architecture="mlp"),
        static_indices=(0,),
    )
    difficulty = torch.linspace(0.0, 1.0, n)[:, None].expand(-1, horizon)
    return ControlledResidualEnvironment(
        batch,
        outcome_model=model,
        n_actions=2,
        difficulty=difficulty,
        history_length=1,
        static_indices=(0,),
        state_feature_names=("patient_context", "dynamic_1", "dynamic_2", "decision_time"),
        neighbors=8,
    )


def test_rollout_preserves_static_context_and_is_crn_reproducible() -> None:
    environment = _environment()
    noise = make_controlled_noise(
        n=12,
        horizon=environment.horizon,
        initial_count=environment.initial_count,
        seed=71,
        device="cpu",
    )
    kwargs = dict(
        environment=environment,
        policy=_AlternatingPolicy(),
        noise=noise,
        gamma=0.5,
        action_coordinate=torch.tensor((-1.0, 1.0)),
    )
    first = rollout_controlled(**kwargs)
    second = rollout_controlled(**kwargs)

    static = first.trajectories.states[:, :, 0]
    assert torch.equal(static, static[:, :1].expand_as(static))
    assert torch.equal(first.trajectories.states, second.trajectories.states)
    assert torch.equal(first.trajectories.outcomes, second.trajectories.outcomes)
    assert torch.all(first.donor_kernel_ess >= 1.0)
    assert torch.all(first.donor_probability_max <= 1.0)


def test_patient_aggregated_diagnostics_merge_duplicate_donor_rows() -> None:
    batch = TrajectoryBatch(
        states=torch.zeros((4, 2, 1)),
        actions=torch.zeros((4, 1), dtype=torch.long),
        outcomes=torch.zeros((4, 1, 2)),
        patient_ids=torch.tensor([10, 10, 11, 12]),
    )
    environment = ControlledResidualEnvironment(
        batch,
        outcome_model=_ConstantOutcomeModel(),
        n_actions=1,
        difficulty=torch.zeros((4, 1)),
        history_length=1,
        state_feature_names=("dynamic",),
        neighbors=4,
    )
    query_state = torch.zeros((2, 1))
    query_action = torch.zeros(2, dtype=torch.long)
    action_coordinate = torch.zeros(1)

    _, _, _, row_ess, row_probability_max = environment.step_from_uniform(
        query_state,
        query_action,
        torch.tensor([0.1, 0.9]),
        time=0,
        gamma=0.0,
        action_coordinate=action_coordinate,
    )
    patient_ess, patient_probability_max, unique_k = (
        environment.patient_aggregated_kernel_diagnostics(
            query_state,
            query_action,
            time=0,
            gamma=0.0,
            action_coordinate=action_coordinate,
            chunk_size=1,
        )
    )

    # Four uniform donor rows have masses [1/4, 1/4, 1/4, 1/4].  The first
    # two rows belong to one patient, so patient-level masses are [1/2, 1/4, 1/4].
    expected_patient_masses = torch.tensor([0.5, 0.25, 0.25])
    expected_patient_ess = 1.0 / expected_patient_masses.square().sum()
    assert row_ess.tolist() == pytest.approx([4.0, 4.0])
    assert row_probability_max.tolist() == pytest.approx([0.25, 0.25])
    assert patient_ess.tolist() == pytest.approx([expected_patient_ess] * 2)
    assert patient_probability_max.tolist() == pytest.approx([0.5, 0.5])
    assert unique_k.tolist() == pytest.approx([3.0, 3.0])
    assert torch.all(patient_ess < row_ess)
    assert torch.all(patient_probability_max > row_probability_max)


def test_local_delta_and_raw_outcome_residual_are_explicit_opt_ins() -> None:
    batch = TrajectoryBatch(
        states=torch.tensor([[[0.0], [2.0]], [[10.0], [14.0]]]),
        actions=torch.zeros((2, 1), dtype=torch.long),
        outcomes=torch.full((2, 1, 2), 2.0),
        patient_ids=torch.arange(2),
    )
    common = {
        "batch": batch,
        "outcome_model": _StateScaledOutcomeModel(),
        "n_actions": 1,
        "difficulty": torch.zeros((2, 1)),
        "history_length": 1,
        "state_feature_names": ("dynamic",),
        "neighbors": 2,
        "donor_weighting": "uniform",
    }
    canonical = ControlledResidualEnvironment(**common)
    repaired = ControlledResidualEnvironment(
        **common,
        transition_mode="local_delta",
        outcome_residual_mode="raw",
    )
    arguments = {
        "state": torch.tensor([[1.0]]),
        "action": torch.zeros(1, dtype=torch.long),
        "donor_uniform": torch.tensor([0.1]),
        "time": 0,
        "gamma": 0.0,
        "action_coordinate": torch.zeros(1),
    }

    canonical_next, canonical_outcome, *_ = canonical.step_from_uniform(**arguments)
    repaired_next, repaired_outcome, *_ = repaired.step_from_uniform(**arguments)

    assert repaired_next.item() == pytest.approx(3.0)
    assert torch.equal(repaired_outcome, torch.tensor([[2.0, 2.0]]))
    assert not torch.equal(canonical_next, repaired_next)
    assert torch.equal(canonical_outcome, torch.tensor([[4.0, 4.0]]))

    with pytest.raises(ValueError, match="transition mode"):
        ControlledResidualEnvironment(**common, transition_mode="unknown")
    with pytest.raises(ValueError, match="outcome residual mode"):
        ControlledResidualEnvironment(**common, outcome_residual_mode="unknown")


def test_explicit_legacy_modes_match_the_default_bitwise() -> None:
    batch = TrajectoryBatch(
        states=torch.tensor([[[0.0], [2.0]], [[10.0], [14.0]]]),
        actions=torch.zeros((2, 1), dtype=torch.long),
        outcomes=torch.full((2, 1, 2), 2.0),
        patient_ids=torch.arange(2),
    )
    common = {
        "batch": batch,
        "outcome_model": _StateScaledOutcomeModel(),
        "n_actions": 1,
        "difficulty": torch.zeros((2, 1)),
        "history_length": 1,
        "state_feature_names": ("dynamic",),
        "neighbors": 2,
        "donor_weighting": "uniform",
    }
    default = ControlledResidualEnvironment(**common)
    explicit = ControlledResidualEnvironment(
        **common,
        transition_mode="ridge_residual",
        outcome_residual_mode="standardized",
    )
    arguments = {
        "state": torch.tensor([[1.0], [3.0]]),
        "action": torch.zeros(2, dtype=torch.long),
        "donor_uniform": torch.tensor([0.1, 0.9]),
        "time": 0,
        "gamma": -4.0,
        "action_coordinate": torch.zeros(1),
    }

    for default_value, explicit_value in zip(
        default.step_from_uniform(**arguments),
        explicit.step_from_uniform(**arguments),
        strict=True,
    ):
        assert torch.equal(default_value, explicit_value)


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ),
)
def test_step_from_uniform_uses_deterministic_lower_median(device: str) -> None:
    median_cases = (
        (
            torch.tensor(
                ((1.0, 2.0, 3.0, 4.0), (2.0, 4.0, 8.0, 16.0)),
                device=device,
            ),
            torch.tensor((2.0, 4.0), device=device),
        ),
        (
            torch.tensor(((1.0, 2.0, 3.0), (2.0, 4.0, 8.0)), device=device),
            torch.tensor((2.0, 4.0), device=device),
        ),
    )
    for sorted_distances, expected in median_cases:
        assert torch.equal(
            _lower_median_from_sorted_rows(sorted_distances), expected
        )

    batch = TrajectoryBatch(
        states=torch.arange(8, dtype=torch.float32).reshape(4, 2, 1),
        actions=torch.zeros((4, 1), dtype=torch.long),
        outcomes=torch.zeros((4, 1, 2)),
        patient_ids=torch.arange(4),
    )
    environment = ControlledResidualEnvironment(
        batch,
        outcome_model=_ConstantOutcomeModel().to(device),
        n_actions=1,
        difficulty=torch.linspace(0.0, 1.0, 4)[:, None],
        history_length=1,
        state_feature_names=("dynamic",),
        neighbors=4,
    )

    deterministic_was_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only_was_enabled = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        step_arguments = dict(
            state=torch.tensor(((0.5,), (2.5,)), device=device),
            action=torch.zeros(2, dtype=torch.long, device=device),
            donor_uniform=torch.tensor((0.1, 0.9), device=device),
            time=0,
            gamma=1.0,
            action_coordinate=torch.ones(1, device=device),
        )
        first_outputs = environment.step_from_uniform(**step_arguments)
        second_outputs = environment.step_from_uniform(**step_arguments)
        diagnostics = environment.patient_aggregated_kernel_diagnostics(
            torch.tensor(((0.5,), (2.5,)), device=device),
            torch.zeros(2, dtype=torch.long, device=device),
            time=0,
            gamma=1.0,
            action_coordinate=torch.ones(1, device=device),
        )
    finally:
        torch.use_deterministic_algorithms(
            deterministic_was_enabled, warn_only=warn_only_was_enabled
        )

    assert all(torch.isfinite(value).all() for value in first_outputs)
    assert all(
        torch.equal(first, second)
        for first, second in zip(first_outputs, second_outputs)
    )
    assert all(torch.isfinite(value).all() for value in diagnostics)
