from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import pytest
import torch

from scpcp.native_signed_gamma import (
    GAMMAS,
    PROVISIONAL_BASE_SEEDS,
    RNG_AUDIT_POLICY,
    NativeSignedGammaBenchmarkConfig,
    NativeSignedGammaKernel,
    NativeSignedGammaLoggingPolicy,
    NativeSignedGammaRadiusPolicy,
    _trajectory_invariants,
    make_native_signed_gamma_noise,
    mechanism_probe,
    rollout_native_signed_gamma,
    seed_passes_mechanism_gate,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/native_synthetic_signed_gamma.yaml"


def _small_config(n: int = 2_000) -> NativeSignedGammaBenchmarkConfig:
    return replace(
        NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG),
        mechanism_trajectories=n,
    )


def test_contract_is_new_gamma_and_has_no_user_controlled_launch_switch() -> None:
    config = NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG)
    serialized = config.to_dict()

    assert config.gammas == GAMMAS
    assert config.primary_gamma == -4.0
    assert config.base_seeds == PROVISIONAL_BASE_SEEDS
    assert config.rng_audit_policy == RNG_AUDIT_POLICY
    assert "seed_collision_audit_status" not in serialized
    assert "seed_collision_audit_sha256" not in serialized
    assert "beta" not in str(serialized).lower()
    assert "feedback_strength" not in str(serialized)


def test_kernel_has_no_radius_and_gamma_zero_is_exactly_action_invariant() -> None:
    config = NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG)
    kernel = NativeSignedGammaKernel(config.dgp, gamma=0.0)
    noise = make_native_signed_gamma_noise(n=32, horizon=1, seed=11, device="cpu")
    state = kernel.initial_state(noise)
    kwargs = {
        "difficulty_uniform": noise.difficulty_uniforms[:, 0],
        "tail_uniform": noise.tail_uniforms[:, 0],
        "transition_normals": noise.transition_normals[:, 0],
        "outcome_normals": noise.outcome_normals[:, 0],
        "time": 0,
        "horizon": 1,
    }

    action_zero = torch.zeros(32, dtype=torch.long)
    action_two = torch.full((32,), 2, dtype=torch.long)
    zero = kernel.step_from_noise(state, action_zero, **kwargs)
    two = kernel.step_from_noise(state, action_two, **kwargs)

    assert "radius" not in inspect.signature(kernel.step_from_noise).parameters
    assert all(torch.equal(left, right) for left, right in zip(zero, two))


def test_gamma_interaction_requires_observed_difficulty_and_reverses_direction() -> None:
    config = NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG)
    state = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    low = torch.zeros(2, dtype=torch.long)
    high = torch.full((2,), 2, dtype=torch.long)

    negative = NativeSignedGammaKernel(config.dgp, gamma=-4.0)
    positive = NativeSignedGammaKernel(config.dgp, gamma=4.0)

    assert negative.difficulty_probability(state, low)[0] > negative.difficulty_probability(state, high)[0]
    assert negative.tail_probability(state, low)[0] > negative.tail_probability(state, high)[0]
    assert positive.difficulty_probability(state, low)[0] < positive.difficulty_probability(state, high)[0]
    assert positive.tail_probability(state, low)[0] < positive.tail_probability(state, high)[0]
    assert negative.difficulty_probability(state, low)[1] == negative.difficulty_probability(state, high)[1]
    assert negative.tail_probability(state, low)[1] == negative.tail_probability(state, high)[1]


def test_radius_changes_only_policy_and_gamma_zero_is_paired_path_placebo() -> None:
    config = _small_config(512)
    noise = make_native_signed_gamma_noise(
        n=config.mechanism_trajectories,
        horizon=config.horizon,
        seed=13,
        device="cpu",
    )
    kernel = NativeSignedGammaKernel(config.dgp, gamma=0.0)
    logging = NativeSignedGammaLoggingPolicy(config.dgp)
    target = NativeSignedGammaRadiusPolicy(logging)
    source = rollout_native_signed_gamma(kernel, logging, noise)
    deployed = rollout_native_signed_gamma(
        kernel,
        target,
        noise,
        radius=config.gate.radius_high,
    )

    current = source.states[:, :-1].reshape(-1, kernel.state_dim)
    mu = logging.probabilities(current)
    pi = target.probabilities(current, config.gate.radius_high)
    assert (pi - mu).abs().sum(dim=1).mean() > 0.10
    assert (pi / mu).max() <= config.dgp.policy_ratio_cap + 1e-12
    assert not torch.equal(source.actions, deployed.actions)
    assert torch.equal(source.states, deployed.states)
    assert torch.equal(source.outcomes, deployed.outcomes)
    assert torch.equal(source.tail_indicators, deployed.tail_indicators)
    assert source.kernel_fingerprint == deployed.kernel_fingerprint


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_time_coordinate_invariant_uses_the_dgp_scalar_construction(
    device: str,
) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CUDA time-coordinate regression")
    config = _small_config(32)
    noise = make_native_signed_gamma_noise(
        n=config.mechanism_trajectories,
        horizon=config.horizon,
        seed=7,
        device=device,
    )
    trajectory = rollout_native_signed_gamma(
        NativeSignedGammaKernel(config.dgp, gamma=-4.0),
        NativeSignedGammaLoggingPolicy(config.dgp),
        noise,
    )

    assert _trajectory_invariants(trajectory)

    perturbed_states = trajectory.states.clone()
    stage = 5
    perturbed_states[0, stage, 3] = torch.nextafter(
        perturbed_states[0, stage, 3],
        perturbed_states.new_tensor(float("inf")),
    )

    assert not _trajectory_invariants(
        replace(trajectory, states=perturbed_states)
    )


def test_coverage_blind_unit_probe_passes_without_science_fields() -> None:
    config = _small_config(5_000)
    probe = mechanism_probe(config, seed=7, device="cpu")
    serialized = str(probe).lower()

    assert seed_passes_mechanism_gate(probe, config.gate)
    assert [row["gamma"] for row in probe["gamma_rows"]] == list(GAMMAS)
    for forbidden in ("coverage", "width", "q90", "score", "selection"):
        assert forbidden not in serialized


def test_formal_config_rejects_seed_or_science_budget_drift() -> None:
    config = NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG)
    with pytest.raises(ValueError, match="reserved native-gamma namespace"):
        replace(config, base_seeds=config.base_seeds[:-1]).validate()
    with pytest.raises(ValueError, match="science budgets"):
        replace(config, reference_trajectories=19_999).validate()
    with pytest.raises(ValueError, match="rng_audit_policy"):
        replace(config, rng_audit_policy="trust_checked_in_digest").validate()
