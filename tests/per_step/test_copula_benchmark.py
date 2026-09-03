from __future__ import annotations

import torch

from scpcp.copula_benchmark import (
    CopulaKernel,
    LoggingPolicy,
    RadiusResponsivePolicy,
    empirical_left_quantile,
    evaluate_mechanism_setting,
    make_copula_noise,
    prepare_source_reference,
    rollout_copula,
)
from scpcp.copula_benchmark_config import CopulaDGPConfig


def test_empirical_left_quantile_uses_ceil_rank_without_interpolation() -> None:
    values = torch.tensor([[1.0], [2.0], [100.0]], dtype=torch.float64)

    quantile = empirical_left_quantile(values, 0.50, dim=0)

    assert torch.equal(quantile, torch.tensor([2.0], dtype=torch.float64))


def test_each_coordinate_is_standard_normal_and_only_correlation_changes() -> None:
    config = CopulaDGPConfig()
    kernel = CopulaKernel(config, beta=1.0)
    generator = torch.Generator().manual_seed(7)
    normals = torch.randn((200_000, 2), generator=generator, dtype=torch.float64)
    next_hard = torch.cat(
        (torch.zeros(100_000, dtype=torch.float64), torch.ones(100_000, dtype=torch.float64))
    )

    residuals = kernel.standardized_residuals(next_hard, normals)

    for regime, expected_correlation in ((0, config.easy_correlation), (1, config.hard_correlation)):
        cell = residuals[next_hard == regime]
        assert torch.all(cell.mean(dim=0).abs() < 0.01)
        assert torch.all((cell.square().mean(dim=0) - 1.0).abs() < 0.015)
        assert abs(float(torch.corrcoef(cell.T)[0, 1]) - expected_correlation) < 0.01


def test_radius_changes_only_the_nonanticipating_action_policy() -> None:
    config = CopulaDGPConfig()
    hard = torch.tensor([0.0, 1.0], dtype=torch.float64)
    actions = torch.tensor([0, 1])
    kernel = CopulaKernel(config, beta=1.0)
    low = RadiusResponsivePolicy(config, radius=config.response_radius_low, kappa=1.0)
    high = RadiusResponsivePolicy(config, radius=config.response_radius_high, kappa=1.0)

    assert torch.equal(
        low.action_one_probability(hard),
        LoggingPolicy(config).action_one_probability(hard),
    )
    assert torch.all(high.action_one_probability(hard) > low.action_one_probability(hard))
    assert torch.equal(
        kernel.hard_probability(hard, actions),
        kernel.hard_probability(hard, actions),
    )
    assert "radius" not in kernel.__dict__
    assert "kappa" not in kernel.__dict__


def test_kappa_zero_is_an_exact_paired_placebo() -> None:
    config = CopulaDGPConfig()
    noise = make_copula_noise(n=4_096, horizon=4, seed=11, device="cpu")
    kernel = CopulaKernel(config, beta=1.0)
    source = rollout_copula(kernel, LoggingPolicy(config), noise)
    target = rollout_copula(
        kernel,
        RadiusResponsivePolicy(config, radius=1.90, kappa=0.0),
        noise,
    )

    assert torch.equal(source.actions, target.actions)
    assert torch.equal(source.states, target.states)
    assert torch.equal(source.residuals, target.residuals)


def test_beta_zero_changes_actions_but_not_regime_or_score_law_under_crn() -> None:
    config = CopulaDGPConfig()
    noise = make_copula_noise(n=8_192, horizon=5, seed=13, device="cpu")
    kernel = CopulaKernel(config, beta=0.0)
    source = prepare_source_reference(kernel, noise, alpha=0.10)

    result = evaluate_mechanism_setting(
        kernel,
        source,
        noise,
        radius=1.90,
        kappa=1.0,
        alpha=0.10,
    )

    assert torch.all(result.policy_tv_on_source > 0.0)
    assert torch.equal(result.source_hard_prevalence, result.target_hard_prevalence)
    assert torch.equal(result.source_q90, result.target_q90)
    assert torch.equal(result.source_coverage, result.target_coverage)
    assert torch.all((0.0 < result.overlap.ess_fraction) & (result.overlap.ess_fraction <= 1.0))


def test_signed_beta_reverses_the_policy_induced_regime_shift() -> None:
    config = CopulaDGPConfig()
    noise = make_copula_noise(n=20_000, horizon=5, seed=17, device="cpu")
    mean_gaps = []
    for beta in (-1.0, 1.0):
        kernel = CopulaKernel(config, beta=beta)
        source = prepare_source_reference(kernel, noise, alpha=0.10)
        result = evaluate_mechanism_setting(
            kernel,
            source,
            noise,
            radius=1.90,
            kappa=1.0,
            alpha=0.10,
        )
        mean_gaps.append(float((result.target_hard_prevalence - result.source_hard_prevalence).mean()))

    assert mean_gaps[0] < 0.0 < mean_gaps[1]

