from __future__ import annotations

import math
from statistics import NormalDist

import pytest
import torch
from torch import Tensor

import scpcp.phase0_oracle as phase0_oracle
from scpcp.simulator import SyntheticNoiseBundle


class RadiusProbabilityPolicy:
    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        assert q is not None
        radius = torch.as_tensor(q, dtype=states.dtype, device=states.device)
        probability_zero = radius.expand(len(states))
        return torch.stack((probability_zero, 1.0 - probability_zero), dim=1)


class AuditedEnvironment:
    def __init__(self, horizon: int) -> None:
        self.horizon = horizon
        self.initial_calls: list[tuple[int, int, int, int]] = []
        self.state_means: list[float] = []
        self.action_means: list[float] = []
        self.noise_pointers: list[tuple[int, ...]] = []
        self.first_states: list[Tensor] = []

    def initial_state_from_noise(self, noise: SyntheticNoiseBundle) -> Tensor:
        self.initial_calls.append(
            (
                id(noise),
                noise.seed,
                len(noise.initial_normal),
                noise.initial_normal.data_ptr(),
            )
        )
        return noise.initial_normal[:, :1].clone()

    def step_from_noise(
        self,
        state: Tensor,
        action: Tensor,
        *,
        shared: Tensor,
        independent: Tensor,
        innovations: Tensor,
        difficulty_uniform: Tensor,
        contamination_uniform: Tensor,
    ) -> tuple[Tensor, Tensor]:
        stage = len(self.state_means) % self.horizon
        if stage == 0:
            self.first_states.append(state.clone())
        self.state_means.append(float(state.mean().item()))
        self.action_means.append(float(action.float().mean().item()))
        self.noise_pointers.append(
            tuple(
                value.data_ptr()
                for value in (
                    shared,
                    independent,
                    innovations,
                    difficulty_uniform,
                    contamination_uniform,
                )
            )
        )
        action_float = action.to(state)
        outcomes = torch.stack(
            (0.2 + 0.1 * action_float, 1.2 + 0.4 * action_float),
            dim=1,
        )
        return state + action_float[:, None] + innovations[:, :1], outcomes


class TwoCoordinateOutcomeModel:
    def __call__(self, states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        del actions
        means = states.new_zeros((len(states), 2))
        scales = states.new_tensor((1.0, 2.0)).expand(len(states), -1)
        return means, scales


def _noise(*, n: int, horizon: int, seed: int) -> SyntheticNoiseBundle:
    patient_uniform = (torch.arange(n, dtype=torch.float32) + 0.5) / n
    innovations = torch.zeros(horizon, n, 4)
    innovations[:, :, 0] = 0.01 * torch.arange(1, horizon + 1)[:, None]
    stage_patient = torch.arange(horizon * n, dtype=torch.float32).reshape(horizon, n)
    return SyntheticNoiseBundle(
        initial_normal=torch.zeros(n, 6),
        initial_difficulty_uniform=patient_uniform.clone(),
        action_uniform=patient_uniform.expand(horizon, -1).clone(),
        shared_normal=stage_patient.clone(),
        independent_normal=-stage_patient.clone(),
        innovation_normal=innovations,
        difficulty_uniform=(patient_uniform / 2.0).expand(horizon, -1).clone(),
        contamination_uniform=(patient_uniform / 3.0).expand(horizon, -1).clone(),
        seed=seed,
    )


def test_wilson_matches_independent_one_sided_bonferroni_formula() -> None:
    n = 200
    counts = torch.tensor([0, 1, 20, 50, 90, 100, 120, 150, 180, 199, 200, 133])
    hits = torch.arange(n)[:, None] < counts[None, :]

    observed = phase0_oracle.bonferroni_wilson_lower_bounds(hits)

    z = NormalDist().inv_cdf(1.0 - 0.05 / 12.0)
    expected = []
    for count in counts.tolist():
        proportion = count / n
        denominator = 1.0 + z * z / n
        center = proportion + z * z / (2.0 * n)
        spread = z * math.sqrt(
            proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)
        )
        expected.append((center - spread) / denominator)

    assert torch.allclose(
        observed.double(),
        torch.tensor(expected, dtype=torch.double),
        atol=1e-7,
        rtol=0.0,
    )


def test_wilson_lower_bound_is_monotone_in_hits() -> None:
    n = 100
    counts = torch.tensor([0, 5, 25, 50, 75, 95, 100])
    hits = torch.arange(n)[:, None] < counts[None, :]

    bounds = phase0_oracle.bonferroni_wilson_lower_bounds(hits)

    assert torch.all(bounds[1:] >= bounds[:-1])


@pytest.mark.parametrize(
    "invalid_schedule",
    (torch.ones(11), torch.ones(13), torch.ones(2, 6)),
    ids=("short", "long", "two-dimensional"),
)
def test_frozen_evaluation_rejects_non_horizon_schedule_before_rollout(
    invalid_schedule: Tensor,
) -> None:
    horizon = 12
    noise = _noise(n=4, horizon=horizon, seed=41)
    environment = AuditedEnvironment(horizon)

    with pytest.raises(ValueError, match=r"schedule 'invalid' must have shape \(12,\)"):
        phase0_oracle.evaluate_frozen_schedules_crn(
            environment,
            RadiusProbabilityPolicy(),
            TwoCoordinateOutcomeModel(),
            schedules={
                "valid": torch.ones(horizon),
                "invalid": invalid_schedule,
            },
            noise=noise,
            outcome_sd=torch.ones(2),
        )

    assert environment.initial_calls == []


def test_frozen_evaluation_rejects_forbidden_noise_seed_before_rollout() -> None:
    horizon = 12
    noise = _noise(n=4, horizon=horizon, seed=1_300_001)
    environment = AuditedEnvironment(horizon)

    with pytest.raises(ValueError, match="evaluation noise seed 1300001 is forbidden"):
        phase0_oracle.evaluate_frozen_schedules_crn(
            environment,
            RadiusProbabilityPolicy(),
            TwoCoordinateOutcomeModel(),
            schedules={"profiled": torch.ones(horizon)},
            noise=noise,
            outcome_sd=torch.ones(2),
            forbidden_noise_seeds={1_200_001, 1_300_001},
        )

    assert environment.initial_calls == []


def test_frozen_schedules_use_one_fresh_50000_bundle_and_restart_each_schedule() -> None:
    horizon = 12
    tuning_stream_id = 1_300_001
    evaluation_stream_id = 1_400_001
    noise = _noise(n=50_000, horizon=horizon, seed=evaluation_stream_id)
    environment = AuditedEnvironment(horizon)
    rng_before = torch.random.get_rng_state()

    evaluations = phase0_oracle.evaluate_frozen_schedules_crn(
        environment,
        RadiusProbabilityPolicy(),
        TwoCoordinateOutcomeModel(),
        schedules={
            "profiled": torch.full((horizon,), 0.55),
            "greedy": torch.full((horizon,), 0.75),
        },
        noise=noise,
        outcome_sd=torch.tensor([1.0, 4.0]),
        forbidden_noise_seeds={tuning_stream_id},
    )

    assert [call[0] for call in environment.initial_calls] == [id(noise), id(noise)]
    assert [call[1] for call in environment.initial_calls] == [evaluation_stream_id] * 2
    assert [call[2] for call in environment.initial_calls] == [50_000, 50_000]
    assert environment.noise_pointers[:horizon] == environment.noise_pointers[horizon:]
    assert all(torch.equal(state, torch.zeros_like(state)) for state in environment.first_states)
    assert environment.state_means[1] == pytest.approx(0.46, abs=1e-6)
    assert environment.state_means[horizon + 1] == pytest.approx(0.26, abs=1e-6)
    assert environment.action_means[0] == pytest.approx(0.45, abs=1e-6)
    assert environment.action_means[horizon] == pytest.approx(0.25, abs=1e-6)
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    profiled = evaluations["profiled"]
    greedy = evaluations["greedy"]
    assert profiled.n_rollouts == greedy.n_rollouts == 50_000
    assert torch.equal(profiled.coverage, torch.zeros(horizon))
    assert torch.equal(greedy.coverage, torch.full((horizon,), 0.75))
    assert torch.allclose(profiled.normalized_width, torch.full((horizon,), 0.825))
    assert torch.allclose(greedy.normalized_width, torch.full((horizon,), 1.125))
    assert profiled.micro_normalized_width == pytest.approx(0.825, abs=1e-6)
    assert profiled.patient_normalized_width == pytest.approx(0.825, abs=1e-6)
    assert greedy.micro_normalized_width == pytest.approx(1.125, abs=1e-6)
    assert greedy.patient_normalized_width == pytest.approx(1.125, abs=1e-6)

    z = NormalDist().inv_cdf(1.0 - 0.05 / horizon)
    proportion = 0.75
    denominator = 1.0 + z * z / 50_000
    center = proportion + z * z / 100_000
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / 50_000
        + z * z / (4.0 * 50_000 * 50_000)
    )
    expected_lower_bound = (center - spread) / denominator
    assert torch.allclose(
        greedy.wilson_lower_bound.double(),
        torch.full((horizon,), expected_lower_bound, dtype=torch.double),
        atol=1e-7,
        rtol=0.0,
    )
