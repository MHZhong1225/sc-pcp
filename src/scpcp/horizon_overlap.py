"""Paired exact-MDP horizon--overlap diagnostic for committed-prefix transport.

This module is isolated from the canonical SC-PCP implementation.  It reuses
the validated M3 data-generating family, but performs only an O(T K n)
stagewise search, so the T=20 protocol never enumerates K**T schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from scpcp.exact_finite_mdp import (
    ExactFiniteMDPConfig,
    LoggedTrajectories,
    PairedMechanism,
    build_paired_mechanisms,
    generate_logged_randomness,
    simulate_logged_trajectories,
)
from scpcp.horizon_overlap_config import (
    METHOD_NAMES,
    HorizonOverlapConfig,
    horizon_overlap_seed_collision_audit,
)


@dataclass(frozen=True)
class PolicyMixingSolution:
    mixing_strength: float
    base_reference_tv: float
    realized_reference_tv: float


@dataclass(frozen=True)
class HorizonOverlapInstanceResult:
    problem_seed: int
    logging_seed: int
    base_reference_tv: float
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class HorizonOverlapStudyResult:
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class _PopulationKernels:
    coverage_by_state: np.ndarray
    width_by_state: np.ndarray
    transition_by_state: np.ndarray
    policy_tv_by_state: np.ndarray


def average_policy_tv(
    behavior_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
) -> float:
    """Average one-step policy TV uniformly over all supplied state cells."""

    behavior = np.asarray(behavior_probabilities, dtype=np.float64)
    target = np.asarray(target_probabilities, dtype=np.float64)
    if behavior.ndim != 2 or target.shape[-2:] != behavior.shape:
        raise ValueError("target policies must end in the behavior [S,A] shape")
    return float((0.5 * np.abs(target - behavior).sum(axis=-1)).mean())


def solve_policy_mixing_strength(
    behavior_probabilities: np.ndarray,
    base_reference_probabilities: np.ndarray,
    *,
    nominal_tv: float,
    tolerance: float = 1e-12,
) -> PolicyMixingSolution:
    """Solve kappa from policy probabilities only for (1-kappa)mu+kappa*pi."""

    if not 0.0 <= nominal_tv <= 1.0:
        raise ValueError("nominal_tv must lie in [0, 1]")
    base_tv = average_policy_tv(
        behavior_probabilities,
        base_reference_probabilities,
    )
    if nominal_tv > base_tv + tolerance:
        raise RuntimeError(
            "policy-only overlap target is unattainable: "
            f"requested {nominal_tv:.6f}, base reference TV {base_tv:.6f}"
        )
    kappa = 0.0 if nominal_tv == 0.0 else nominal_tv / base_tv
    if kappa > 1.0 + tolerance:
        raise RuntimeError("policy mixing strength exceeds one")
    kappa = min(kappa, 1.0)
    mixed = mix_target_policy(
        behavior_probabilities,
        base_reference_probabilities,
        mixing_strength=kappa,
    )
    realized = average_policy_tv(behavior_probabilities, mixed)
    if not np.isclose(realized, nominal_tv, atol=tolerance, rtol=0.0):
        raise RuntimeError("analytic policy-TV solve did not attain its target")
    return PolicyMixingSolution(
        mixing_strength=float(kappa),
        base_reference_tv=base_tv,
        realized_reference_tv=realized,
    )


def mix_target_policy(
    behavior_probabilities: np.ndarray,
    base_target_probabilities: np.ndarray,
    *,
    mixing_strength: float,
) -> np.ndarray:
    """Return the convex target-policy path without consulting outcomes."""

    if not 0.0 <= mixing_strength <= 1.0:
        raise ValueError("mixing_strength must lie in [0, 1]")
    behavior = np.asarray(behavior_probabilities, dtype=np.float64)
    base_target = np.asarray(base_target_probabilities, dtype=np.float64)
    if behavior.ndim != 2 or base_target.shape[-2:] != behavior.shape:
        raise ValueError("base target policies must end in the behavior [S,A] shape")
    return (1.0 - mixing_strength) * behavior + mixing_strength * base_target


def audit_policy_design(config: HorizonOverlapConfig) -> dict[str, Any]:
    """Audit design and all formal policies before any score is generated."""

    config.validate()
    design_values = _policy_only_reference_tvs(config, config.design_seeds)
    formal_values = _policy_only_reference_tvs(config, config.problem_seeds)
    design_passed = bool(
        np.all(design_values >= config.minimum_base_reference_tv - 1e-12)
    )
    formal_passed = bool(
        np.all(formal_values >= config.minimum_base_reference_tv - 1e-12)
    )
    audit = {
        "status": "pass" if design_passed and formal_passed else "fail",
        "outcome_blind": True,
        "policy_probabilities_only": True,
        "generated_scores": False,
        "inspected_coverage": False,
        "rq5_only_policy_center_reset": {
            "parent_exact_M3_center": config.parent_policy_response_center,
            "rq5_center": config.radius_minimum,
            "scope": "RQ5_only",
        },
        "design_seed_ids": list(config.design_seeds),
        "formal_problem_seed_ids": list(config.problem_seeds),
        "all_formal_problem_seeds_checked_before_logged_score_generation": True,
        "reference_grid_index": config.reference_grid_index,
        "reference_radius": float(
            np.linspace(
                config.radius_minimum,
                config.radius_maximum,
                config.grid_size,
            )[config.reference_grid_index]
        ),
        "minimum_required_base_reference_tv": config.minimum_base_reference_tv,
        "base_reference_tv": _policy_tv_summary(design_values),
        "independent_design_bank": {
            "status": "pass" if design_passed else "fail",
            "seed_ids": list(config.design_seeds),
            "base_reference_tv": _policy_tv_summary(design_values),
        },
        "formal_problem_bank_attainability": {
            "status": "pass" if formal_passed else "fail",
            "seed_ids": list(config.problem_seeds),
            "base_reference_tv": _policy_tv_summary(formal_values),
            "checked_before_any_logged_score_generation": True,
        },
    }
    if not design_passed or not formal_passed:
        failing_bank = "independent design" if not design_passed else "formal problem"
        failing_values = design_values if not design_passed else formal_values
        raise RuntimeError(
            f"policy-only {failing_bank} preflight failed before score generation: "
            f"minimum base median-grid TV {failing_values.min():.6f} is below "
            f"{config.minimum_base_reference_tv:.6f}"
        )
    return audit


def _policy_only_reference_tvs(
    config: HorizonOverlapConfig,
    seeds: tuple[int, ...],
) -> np.ndarray:
    return np.asarray(
        [
            _base_reference_tv(
                config,
                _build_m3_base(config, problem_seed=seed),
            )
            for seed in seeds
        ],
        dtype=np.float64,
    )


def _policy_tv_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "values": values.tolist(),
    }


def run_horizon_overlap_instance(
    config: HorizonOverlapConfig,
    *,
    problem_seed: int,
    logging_seed: int,
) -> HorizonOverlapInstanceResult:
    """Run all TV levels for one paired M3 kernel and one behavior log."""

    config.validate()
    mechanism = _build_m3_base(config, problem_seed=problem_seed)
    base_reference_tv = _base_reference_tv(config, mechanism)
    if base_reference_tv < config.minimum_base_reference_tv - 1e-12:
        raise RuntimeError(
            "formal policy-only preflight failed before score generation for "
            f"problem seed {problem_seed}: base median-grid TV "
            f"{base_reference_tv:.6f}"
        )

    exact_config = _exact_config(config)
    randomness = generate_logged_randomness(exact_config, seed=logging_seed)
    logged = simulate_logged_trajectories(mechanism, randomness)
    setting_results = [
        _evaluate_overlap_setting(
            config,
            mechanism,
            logged,
            nominal_tv=nominal_tv,
        )
        for nominal_tv in config.nominal_policy_tvs
    ]
    names = setting_results[0]
    arrays = {
        name: np.stack([result[name] for result in setting_results])
        for name in names
    }
    return HorizonOverlapInstanceResult(
        problem_seed=problem_seed,
        logging_seed=logging_seed,
        base_reference_tv=base_reference_tv,
        arrays=arrays,
    )


def run_horizon_overlap_study(
    config: HorizonOverlapConfig,
) -> HorizonOverlapStudyResult:
    """Run the complete 200-instance paired diagnostic without writing files."""

    config.validate()
    design_audit = audit_policy_design(config)
    instance_results = [
        run_horizon_overlap_instance(
            config,
            problem_seed=problem_seed,
            logging_seed=logging_seed,
        )
        for problem_seed, logging_seed in zip(
            config.problem_seeds,
            config.logging_seeds,
            strict=True,
        )
    ]
    instance_array_names = instance_results[0].arrays
    arrays = {
        name: np.stack([result.arrays[name] for result in instance_results])
        for name in instance_array_names
    }
    arrays.update(
        {
            "problem_seeds": np.asarray(config.problem_seeds, dtype=np.int64),
            "logging_seeds": np.asarray(config.logging_seeds, dtype=np.int64),
            "base_reference_tv": np.asarray(
                [result.base_reference_tv for result in instance_results],
                dtype=np.float64,
            ),
            "method_names": np.asarray(METHOD_NAMES),
            "horizons": np.asarray(config.horizons, dtype=np.int16),
            "nominal_policy_tvs": np.asarray(
                config.nominal_policy_tvs,
                dtype=np.float64,
            ),
        }
    )
    arrays["bootstrap_seed"] = np.asarray(config.bootstrap_seed, dtype=np.int64)
    arrays["bootstrap_instance_indices"] = _bootstrap_instance_indices(config)
    summary = _summarize_study(config, arrays)
    summary["policy_design_audit"] = design_audit
    summary["seed_collision_audit"] = horizon_overlap_seed_collision_audit(config)
    return HorizonOverlapStudyResult(summary=summary, arrays=arrays)


def _build_m3_base(
    config: HorizonOverlapConfig,
    *,
    problem_seed: int,
) -> PairedMechanism:
    mechanisms = build_paired_mechanisms(
        _exact_config(config),
        problem_seed=problem_seed,
    )
    return next(
        mechanism
        for mechanism in mechanisms
        if mechanism.name == "M3_full_feedback"
    )


def _exact_config(config: HorizonOverlapConfig) -> ExactFiniteMDPConfig:
    return ExactFiniteMDPConfig(
        state_count=config.state_count,
        action_count=config.action_count,
        horizon=config.maximum_horizon,
        grid_size=config.grid_size,
        alpha=config.alpha,
        logged_trajectories=config.calibration_trajectories,
        radius_minimum=config.radius_minimum,
        radius_maximum=config.radius_maximum,
        policy_response_center=config.radius_minimum,
        policy_response_scale=config.policy_response_scale,
        policy_response_strength=config.policy_response_strength,
    )


def _base_reference_tv(
    config: HorizonOverlapConfig,
    mechanism: PairedMechanism,
) -> float:
    base_target = mechanism.problem.action_probabilities.detach().cpu().numpy()
    return average_policy_tv(
        mechanism.behavior_probabilities,
        base_target[:, config.reference_grid_index],
    )


def _evaluate_overlap_setting(
    config: HorizonOverlapConfig,
    mechanism: PairedMechanism,
    logged: LoggedTrajectories,
    *,
    nominal_tv: float,
) -> dict[str, np.ndarray]:
    problem = mechanism.problem
    base_target = problem.action_probabilities.detach().cpu().numpy()
    behavior = mechanism.behavior_probabilities
    solution = solve_policy_mixing_strength(
        behavior,
        base_target[:, config.reference_grid_index],
        nominal_tv=nominal_tv,
    )
    target_policy = mix_target_policy(
        behavior,
        base_target,
        mixing_strength=solution.mixing_strength,
    )
    kernels = _population_kernels(mechanism, target_policy)
    method_count = len(METHOD_NAMES)
    horizon = problem.radii.shape[0]
    trajectory_count = len(logged.states)

    selected_indices = np.full((method_count, horizon), -1, dtype=np.int16)
    failure_stage = np.full(method_count, -1, dtype=np.int16)
    output_names = (
        "population_coverage",
        "population_width",
        "estimated_coverage",
        "estimated_width",
        "selected_ess_fraction",
        "minimum_candidate_ess_fraction",
        "stage_surface_sup_error",
        "selected_policy_realized_tv",
        "selected_policy_uniform_state_tv",
    )
    outputs = {
        name: np.full((method_count, horizon), np.nan, dtype=np.float64)
        for name in output_names
    }
    active = np.ones(method_count, dtype=bool)
    occupancies = np.broadcast_to(
        problem.initial_state_probabilities.detach().cpu().numpy(),
        (method_count, config.state_count),
    ).copy()
    raw_log_prefix = np.zeros((method_count, trajectory_count), dtype=np.float64)
    radii = problem.radii.detach().cpu().numpy()
    predictor_scales = problem.predictor_scales.detach().cpu().numpy()
    normalization = problem.outcome_normalization.detach().cpu().numpy()
    base_width = 2.0 * np.mean(
        predictor_scales / normalization[None, None, :],
        axis=2,
    )

    for stage in range(horizon):
        states = logged.states[:, stage]
        actions = logged.actions[:, stage]
        observed_target = target_policy[stage, :, states, actions]
        observed_behavior = behavior[states, actions]
        current_log_ratio = np.log(observed_target) - np.log(
            observed_behavior[:, None]
        )
        hits = logged.scores[:, stage, None] <= radii[stage][None, :]
        candidate_width = (
            base_width[states, actions, None] * radii[stage][None, :]
        )

        for method_index in range(method_count):
            if not active[method_index]:
                continue
            log_weights = _method_log_weights(
                method_index,
                raw_log_prefix[method_index],
                current_log_ratio,
            )
            estimated_coverage, estimated_width, ess_fraction = _weighted_curves(
                log_weights,
                hits,
                candidate_width,
            )
            true_coverage = np.einsum(
                "s,ks->k",
                occupancies[method_index],
                kernels.coverage_by_state[stage],
            )
            true_width = np.einsum(
                "s,ks->k",
                occupancies[method_index],
                kernels.width_by_state[stage],
            )
            outputs["minimum_candidate_ess_fraction"][method_index, stage] = (
                ess_fraction.min()
            )
            outputs["stage_surface_sup_error"][method_index, stage] = np.max(
                np.abs(estimated_coverage - true_coverage)
            )

            feasible = estimated_coverage >= config.target_coverage
            if not bool(feasible.any()):
                active[method_index] = False
                failure_stage[method_index] = stage
                continue
            objective = np.where(feasible, estimated_width, np.inf)
            selected = int(objective.argmin())
            selected_indices[method_index, stage] = selected
            outputs["population_coverage"][method_index, stage] = true_coverage[
                selected
            ]
            outputs["population_width"][method_index, stage] = true_width[selected]
            outputs["estimated_coverage"][method_index, stage] = estimated_coverage[
                selected
            ]
            outputs["estimated_width"][method_index, stage] = estimated_width[
                selected
            ]
            outputs["selected_ess_fraction"][method_index, stage] = ess_fraction[
                selected
            ]
            outputs["selected_policy_realized_tv"][method_index, stage] = np.dot(
                occupancies[method_index],
                kernels.policy_tv_by_state[stage, selected],
            )
            outputs["selected_policy_uniform_state_tv"][method_index, stage] = (
                kernels.policy_tv_by_state[stage, selected].mean()
            )
            occupancies[method_index] = np.einsum(
                "s,sr->r",
                occupancies[method_index],
                kernels.transition_by_state[stage, selected],
            )
            if method_index in (1, 3):
                raw_log_prefix[method_index] += current_log_ratio[:, selected]

    availability = np.stack(
        [
            (failure_stage < 0) | (failure_stage >= horizon_value)
            for horizon_value in config.horizons
        ]
    )
    return {
        "mixing_strength": np.asarray(solution.mixing_strength),
        "realized_reference_tv": np.asarray(solution.realized_reference_tv),
        "selected_indices": selected_indices,
        "failure_stage": failure_stage,
        "availability_by_horizon": availability,
        **outputs,
    }


def _method_log_weights(
    method_index: int,
    raw_log_prefix: np.ndarray,
    current_log_ratio: np.ndarray,
) -> np.ndarray:
    if method_index == 0:
        return np.zeros_like(current_log_ratio)
    if method_index == 1:
        return np.broadcast_to(raw_log_prefix[:, None], current_log_ratio.shape)
    if method_index == 2:
        return current_log_ratio
    if method_index == 3:
        return raw_log_prefix[:, None] + current_log_ratio
    raise IndexError("unknown horizon-overlap method index")


def _weighted_curves(
    log_weights: np.ndarray,
    hits: np.ndarray,
    candidate_width: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stabilized = log_weights - log_weights.max(axis=0, keepdims=True)
    weights = np.exp(stabilized)
    weight_sum = weights.sum(axis=0)
    coverage = (weights * hits).sum(axis=0) / weight_sum
    width = (weights * candidate_width).sum(axis=0) / weight_sum
    ess_fraction = weight_sum**2 / (weights**2).sum(axis=0) / len(weights)
    return coverage, width, ess_fraction


def _population_kernels(
    mechanism: PairedMechanism,
    target_policy: np.ndarray,
) -> _PopulationKernels:
    problem = mechanism.problem
    transition = problem.transition_probabilities.detach().cpu().numpy()
    radii = problem.radii.detach().cpu().numpy()
    predictor_means = problem.predictor_means.detach().cpu().numpy()
    predictor_scales = problem.predictor_scales.detach().cpu().numpy()
    outcome_means = problem.outcome_means.detach().cpu().numpy()
    outcome_sd = problem.outcome_standard_deviations.detach().cpu().numpy()
    normalization = problem.outcome_normalization.detach().cpu().numpy()

    standardized_lower = (
        predictor_means[None, None, :, :, None, :]
        - radii[:, :, None, None, None, None]
        * predictor_scales[None, None, :, :, None, :]
        - outcome_means[None, None, None, :, :, :]
    ) / outcome_sd[None, None, None, None, None, :]
    standardized_upper = (
        predictor_means[None, None, :, :, None, :]
        + radii[:, :, None, None, None, None]
        * predictor_scales[None, None, :, :, None, :]
        - outcome_means[None, None, None, :, :, :]
    ) / outcome_sd[None, None, None, None, None, :]
    conditional_hits = np.prod(
        _normal_cdf(standardized_upper) - _normal_cdf(standardized_lower),
        axis=5,
    )
    coverage_by_state = np.einsum(
        "tksa,asr,tksar->tks",
        target_policy,
        transition,
        conditional_hits,
    )
    transition_by_state = np.einsum(
        "tksa,asr->tksr",
        target_policy,
        transition,
    )
    normalized_base_width = 2.0 * np.mean(
        predictor_scales / normalization[None, None, :],
        axis=2,
    )
    conditional_width = (
        radii[:, :, None, None] * normalized_base_width[None, None, :, :]
    )
    width_by_state = np.einsum(
        "tksa,tksa->tks",
        target_policy,
        conditional_width,
    )
    policy_tv_by_state = 0.5 * np.abs(
        target_policy - mechanism.behavior_probabilities[None, None, :, :]
    ).sum(axis=3)
    return _PopulationKernels(
        coverage_by_state=coverage_by_state,
        width_by_state=width_by_state,
        transition_by_state=transition_by_state,
        policy_tv_by_state=policy_tv_by_state,
    )


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    # NumPy does not expose erf in every supported version; torch's exact-MDP
    # implementation uses the same standard-normal CDF.
    import torch

    return torch.special.ndtr(torch.from_numpy(values)).numpy()


def _summarize_study(
    config: HorizonOverlapConfig,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    if "bootstrap_instance_indices" not in arrays:
        arrays["bootstrap_instance_indices"] = _bootstrap_instance_indices(config)
    records = []
    for horizon_index, horizon in enumerate(config.horizons):
        for tv_index, nominal_tv in enumerate(config.nominal_policy_tvs):
            for method_index, method in enumerate(METHOD_NAMES):
                available = arrays["availability_by_horizon"][
                    :, tv_index, horizon_index, method_index
                ]
                selected_count = int(available.sum())
                record: dict[str, Any] = {
                    "horizon": horizon,
                    "nominal_policy_tv": nominal_tv,
                    "method": method,
                    "selected_instances": selected_count,
                    "total_instances": config.instances,
                    "availability_rate": float(available.mean()),
                    "mixing_strength": _quantile_summary(
                        arrays["mixing_strength"][:, tv_index]
                    ),
                    "realized_reference_tv": _quantile_summary(
                        arrays["realized_reference_tv"][:, tv_index]
                    ),
                }
                if selected_count:
                    coverage = arrays["population_coverage"][
                        available, tv_index, method_index, :horizon
                    ]
                    width = arrays["population_width"][
                        available, tv_index, method_index, :horizon
                    ]
                    selected_ess = arrays["selected_ess_fraction"][
                        available, tv_index, method_index, :horizon
                    ]
                    candidate_ess = arrays["minimum_candidate_ess_fraction"][
                        available, tv_index, method_index, :horizon
                    ]
                    surface_error = arrays["stage_surface_sup_error"][
                        available, tv_index, method_index, :horizon
                    ]
                    selected_tv = arrays["selected_policy_realized_tv"][
                        available, tv_index, method_index, :horizon
                    ]
                    uniform_selected_tv = arrays[
                        "selected_policy_uniform_state_tv"
                    ][available, tv_index, method_index, :horizon]
                    per_stage_coverage = coverage.mean(axis=0)
                    marginal_wsc = float(per_stage_coverage.min())
                    selected_ess_summary = _quantile_summary(
                        selected_ess.min(axis=1)
                    )
                    candidate_ess_summary = _quantile_summary(
                        candidate_ess.min(axis=1)
                    )
                    surface_error_summary = _quantile_summary(
                        surface_error.max(axis=1)
                    )
                    selected_tv_summary = _quantile_summary(
                        selected_tv.mean(axis=1)
                    )
                    uniform_selected_tv_summary = _quantile_summary(
                        uniform_selected_tv.mean(axis=1)
                    )
                    record.update(
                        {
                            "marginal_wsc": marginal_wsc,
                            "coverage_shortfall": config.target_coverage
                            - marginal_wsc,
                            "per_stage_mean_coverage": per_stage_coverage.tolist(),
                            "average_normalized_width": float(width.mean()),
                            "minimum_selected_ess_fraction": selected_ess_summary,
                            "median_minimum_selected_ess_fraction": (
                                selected_ess_summary["median"]
                            ),
                            "minimum_candidate_ess_fraction": candidate_ess_summary,
                            "surface_sup_error": surface_error_summary,
                            "median_surface_sup_error": surface_error_summary["median"],
                            "selected_policy_realized_tv": selected_tv_summary,
                            "mean_selected_policy_realized_tv": selected_tv_summary[
                                "mean"
                            ],
                            "selected_policy_uniform_state_tv": (
                                uniform_selected_tv_summary
                            ),
                            "mean_selected_policy_uniform_state_tv": (
                                uniform_selected_tv_summary["mean"]
                            ),
                            "mean_instance_worst_stage_coverage_diagnostic": float(
                                coverage.min(axis=1).mean()
                            ),
                        }
                    )
                else:
                    record.update(
                        {
                            "marginal_wsc": None,
                            "coverage_shortfall": None,
                            "per_stage_mean_coverage": None,
                            "average_normalized_width": None,
                            "minimum_selected_ess_fraction": None,
                            "median_minimum_selected_ess_fraction": None,
                            "minimum_candidate_ess_fraction": None,
                            "surface_sup_error": None,
                            "median_surface_sup_error": None,
                            "selected_policy_realized_tv": None,
                            "mean_selected_policy_realized_tv": None,
                            "selected_policy_uniform_state_tv": None,
                            "mean_selected_policy_uniform_state_tv": None,
                            "mean_instance_worst_stage_coverage_diagnostic": None,
                        }
                    )
                records.append(record)

    bootstrap_comparisons = _attach_bootstrap_intervals(config, arrays, records)
    bootstrap_indices = arrays["bootstrap_instance_indices"]
    return {
        "schema_version": 1,
        "study": "finite_mdp_horizon_overlap",
        "status": "complete",
        "diagnostic_only": True,
        "canonical_method_unchanged": True,
        "mechanism": "M3_full_feedback",
        "mechanism_variant": config.mechanism_variant,
        "rq5_only_policy_center_reset": {
            "scope": "RQ5_horizon_overlap_only",
            "parent_exact_M3_policy_response_center": (
                config.parent_policy_response_center
            ),
            "rq5_policy_response_center": config.radius_minimum,
            "reason": (
                "make the frozen median-grid policy-only TV targets through 0.15 "
                "attainable before observing any outcome, score, or coverage"
            ),
            "canonical_SC_PCP_changed": False,
            "parent_RQ1_results_reinterpreted": False,
        },
        "population_grid_size": config.grid_size,
        "calibration_trajectories": config.calibration_trajectories,
        "paired_longest_horizon_then_truncated": True,
        "paired_behavior_log_across_all_conditions": True,
        "overlap_solve": "outcome_blind_policy_probabilities_only",
        "primary_coverage_estimand": "min_stage_mean_instance_conditional_on_availability",
        "finite_sample_claim": False,
        "policy_tv_reporting": {
            "nominal_and_reference_tv_state_aggregation": "uniform_over_states",
            "selected_policy_realized_tv": "target_occupancy_weighted_over_states",
            "selected_policy_uniform_state_tv": "uniform_over_states",
        },
        "method_roles": {
            "Standard CP": "unweighted structural comparator",
            "History-only Prefix-IW": "diagnostic incomplete-prefix comparator",
            "Current-only IW": "diagnostic incomplete-prefix comparator",
            "SC-PCP": "frozen full committed-prefix method",
        },
        "bootstrap": {
            "resamples": config.bootstrap_resamples,
            "seed": config.bootstrap_seed,
            "rng_registry_label": "summary/instance_cluster_bootstrap",
            "cluster_unit": "paired_M3_instance",
            "shared_index_matrix_across_all_horizon_tv_method_cells": True,
            "selection_conditioning": (
                "each method-cell draw conditions WSC and width on that draw's "
                "resampled available instances; availability uses all resampled instances"
            ),
            "paired_wsc_comparison_conditioning": (
                "report both each-method conditional WSC differences and WSC "
                "differences restricted to the joint-available instance set"
            ),
            "wsc_recomputes_minimum_after_stagewise_resample_means": True,
            "instance_index_matrix_shape": list(bootstrap_indices.shape),
            "instance_index_matrix_sha256": hashlib.sha256(
                np.ascontiguousarray(bootstrap_indices).tobytes()
            ).hexdigest(),
        },
        "bootstrap_wsc_comparisons": bootstrap_comparisons,
        "records": records,
        "phase_diagram": _phase_diagram(config, records),
    }


def _bootstrap_instance_indices(config: HorizonOverlapConfig) -> np.ndarray:
    generator = np.random.default_rng(config.bootstrap_seed)
    return generator.integers(
        0,
        config.instances,
        size=(config.bootstrap_resamples, config.instances),
        dtype=np.int32,
    )


def _attach_bootstrap_intervals(
    config: HorizonOverlapConfig,
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indices = arrays["bootstrap_instance_indices"]
    counts = np.stack(
        [np.bincount(row, minlength=config.instances) for row in indices]
    ).astype(np.float64, copy=False)
    record_by_key = {
        (record["horizon"], record["nominal_policy_tv"], record["method"]): record
        for record in records
    }
    wsc_draws: dict[tuple[int, float, str], np.ndarray] = {}

    for horizon_index, horizon in enumerate(config.horizons):
        for tv_index, nominal_tv in enumerate(config.nominal_policy_tvs):
            for method_index, method in enumerate(METHOD_NAMES):
                key = (horizon, nominal_tv, method)
                record = record_by_key[key]
                available = arrays["availability_by_horizon"][
                    :, tv_index, horizon_index, method_index
                ]
                selected_counts = counts @ available.astype(np.float64)
                availability_draws = selected_counts / config.instances
                record["availability_rate_ci95"] = _percentile_interval(
                    availability_draws
                )
                draws = np.full(config.bootstrap_resamples, np.nan)
                width_draws = np.full_like(draws, np.nan)
                valid = selected_counts > 0.0
                if bool(valid.any()):
                    coverage = arrays["population_coverage"][
                        :, tv_index, method_index, :horizon
                    ]
                    width = arrays["population_width"][
                        :, tv_index, method_index, :horizon
                    ].mean(axis=1)
                    coverage_sums = counts @ np.where(
                        available[:, None],
                        coverage,
                        0.0,
                    )
                    stage_means = coverage_sums[valid] / selected_counts[valid, None]
                    draws[valid] = stage_means.min(axis=1)
                    width_sums = counts @ np.where(available, width, 0.0)
                    width_draws[valid] = width_sums[valid] / selected_counts[valid]
                record["marginal_wsc_ci95"] = _percentile_interval(draws)
                record["average_normalized_width_ci95"] = _percentile_interval(
                    width_draws
                )
                wsc_draws[key] = draws

    comparisons = []
    scpcp_index = METHOD_NAMES.index("SC-PCP")
    for horizon_index, horizon in enumerate(config.horizons):
        for tv_index, nominal_tv in enumerate(config.nominal_policy_tvs):
            scpcp_key = (horizon, nominal_tv, "SC-PCP")
            for comparator in (
                "History-only Prefix-IW",
                "Current-only IW",
            ):
                comparator_index = METHOD_NAMES.index(comparator)
                comparator_key = (horizon, nominal_tv, comparator)
                difference = wsc_draws[scpcp_key] - wsc_draws[comparator_key]
                scpcp_point = record_by_key[scpcp_key]["marginal_wsc"]
                comparator_point = record_by_key[comparator_key]["marginal_wsc"]
                method_conditional_point = (
                    None
                    if scpcp_point is None or comparator_point is None
                    else scpcp_point - comparator_point
                )
                scpcp_available = arrays["availability_by_horizon"][
                    :, tv_index, horizon_index, scpcp_index
                ]
                comparator_available = arrays["availability_by_horizon"][
                    :, tv_index, horizon_index, comparator_index
                ]
                joint_available = scpcp_available & comparator_available
                scpcp_coverage = arrays["population_coverage"][
                    :, tv_index, scpcp_index, :horizon
                ]
                comparator_coverage = arrays["population_coverage"][
                    :, tv_index, comparator_index, :horizon
                ]
                joint_scpcp_draws = _conditional_wsc_draws(
                    counts,
                    joint_available,
                    scpcp_coverage,
                )
                joint_comparator_draws = _conditional_wsc_draws(
                    counts,
                    joint_available,
                    comparator_coverage,
                )
                joint_difference = joint_scpcp_draws - joint_comparator_draws
                joint_scpcp_point = _conditional_wsc_point(
                    joint_available,
                    scpcp_coverage,
                )
                joint_comparator_point = _conditional_wsc_point(
                    joint_available,
                    comparator_coverage,
                )
                joint_point = (
                    None
                    if joint_scpcp_point is None or joint_comparator_point is None
                    else joint_scpcp_point - joint_comparator_point
                )
                comparisons.append(
                    {
                        "horizon": horizon,
                        "nominal_policy_tv": nominal_tv,
                        "comparator": comparator,
                        "comparison_cluster_pairing": "same_instance_bootstrap_indices",
                        "method_conditional_scpcp_selected_instances": record_by_key[
                            scpcp_key
                        ]["selected_instances"],
                        "method_conditional_comparator_selected_instances": (
                            record_by_key[comparator_key]["selected_instances"]
                        ),
                        "method_conditional_scpcp_minus_comparator_wsc": (
                            method_conditional_point
                        ),
                        "method_conditional_scpcp_minus_comparator_wsc_ci95": (
                            _percentile_interval(difference)
                        ),
                        "joint_available_instances": int(joint_available.sum()),
                        "joint_availability_rate": float(joint_available.mean()),
                        "joint_available_scpcp_wsc": joint_scpcp_point,
                        "joint_available_comparator_wsc": joint_comparator_point,
                        "joint_available_scpcp_minus_comparator_wsc": joint_point,
                        "joint_available_scpcp_minus_comparator_wsc_ci95": (
                            _percentile_interval(joint_difference)
                        ),
                    }
                )
    return comparisons


def _conditional_wsc_draws(
    bootstrap_counts: np.ndarray,
    available: np.ndarray,
    coverage: np.ndarray,
) -> np.ndarray:
    selected_counts = bootstrap_counts @ available.astype(np.float64)
    draws = np.full(len(bootstrap_counts), np.nan)
    valid = selected_counts > 0.0
    if bool(valid.any()):
        coverage_sums = bootstrap_counts @ np.where(
            available[:, None],
            coverage,
            0.0,
        )
        stage_means = coverage_sums[valid] / selected_counts[valid, None]
        draws[valid] = stage_means.min(axis=1)
    return draws


def _conditional_wsc_point(
    available: np.ndarray,
    coverage: np.ndarray,
) -> float | None:
    if not bool(available.any()):
        return None
    return float(coverage[available].mean(axis=0).min())


def _percentile_interval(values: np.ndarray) -> dict[str, float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    lower, upper = np.quantile(finite, (0.025, 0.975))
    return {"lower": float(lower), "upper": float(upper)}


def _phase_diagram(
    config: HorizonOverlapConfig,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (record["horizon"], record["nominal_policy_tv"], record["method"]): record
        for record in records
    }
    metric_names = (
        "marginal_wsc",
        "coverage_shortfall",
        "average_normalized_width",
        "availability_rate",
        "median_minimum_selected_ess_fraction",
        "median_surface_sup_error",
        "mean_selected_policy_realized_tv",
        "mean_selected_policy_uniform_state_tv",
    )
    output = {}
    for method in METHOD_NAMES:
        output[method] = {
            metric: [
                [
                    by_key[(horizon, nominal_tv, method)][metric]
                    for nominal_tv in config.nominal_policy_tvs
                ]
                for horizon in config.horizons
            ]
            for metric in metric_names
        }
    return output


def _quantile_summary(values: np.ndarray) -> dict[str, Any]:
    resolved = np.asarray(values, dtype=np.float64)
    if not resolved.size:
        return {"count": 0}
    return {
        "count": int(resolved.size),
        "mean": float(resolved.mean()),
        "standard_deviation": float(resolved.std(ddof=1))
        if resolved.size > 1
        else 0.0,
        "minimum": float(resolved.min()),
        "q05": float(np.quantile(resolved, 0.05)),
        "median": float(np.median(resolved)),
        "q95": float(np.quantile(resolved, 0.95)),
        "maximum": float(resolved.max()),
    }
