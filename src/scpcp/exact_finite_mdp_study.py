"""Replicated population and logged-sample audit for the exact finite MDP."""

from __future__ import annotations

from typing import Any

import numpy as np

from scpcp.exact_finite_mdp import (
    ESTIMATOR_NAMES,
    MECHANISM_NAMES,
    ExactFiniteMDPConfig,
    ExactFiniteMDPResult,
    build_paired_mechanisms,
    enumerate_schedules,
    exact_population_surfaces,
    generate_logged_randomness,
    hajek_surface_estimates,
    run_exact_finite_mdp,
    simulate_logged_trajectories,
)
from scpcp.phase0_search import NoFeasibleScheduleError, greedy_schedule_search


def run_replicated_exact_finite_mdp(
    config: ExactFiniteMDPConfig,
) -> ExactFiniteMDPResult:
    """Run the deterministic fixture plus predeclared random-MDP audits."""

    config.validate()
    fixture = run_exact_finite_mdp(config)
    population_summary, population_arrays = _run_population_instance_audit(config)
    logged_summary, logged_arrays = _run_logged_recovery_audit(config)
    summary = {
        **fixture.summary,
        "instance_protocol": {
            "population_instances": config.population_instances,
            "population_seed_start": config.population_seed_start,
            "logged_instance_count": config.logged_instance_count,
            "logged_replicates": config.logged_replicates,
            "logged_replicate_seed_start": config.logged_replicate_seed_start,
            "deterministic_fixture_is_separate": True,
            "random_instances_are_paired_across_M0_to_M3": True,
            "logged_randomness_is_paired_across_M0_to_M3": True,
        },
        "population_instance_audit": population_summary,
        "logged_recovery_audit": logged_summary,
    }
    arrays = {**fixture.arrays, **population_arrays, **logged_arrays}
    return ExactFiniteMDPResult(summary=summary, arrays=arrays)


def _run_population_instance_audit(
    config: ExactFiniteMDPConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    schedules = enumerate_schedules(config)
    instance_count = config.population_instances
    mechanism_count = len(MECHANISM_NAMES)
    estimator_count = len(ESTIMATOR_NAMES)
    seeds = np.arange(
        config.population_seed_start,
        config.population_seed_start + instance_count,
        dtype=np.int64,
    )
    identification_maximum = np.empty(
        (instance_count, mechanism_count, estimator_count), dtype=np.float64
    )
    identification_rmse = np.empty_like(identification_maximum)
    feasible_fraction = np.empty((instance_count, mechanism_count), dtype=np.float64)
    global_available = np.zeros((instance_count, mechanism_count), dtype=bool)
    greedy_available = np.zeros_like(global_available)
    global_width = np.full((instance_count, mechanism_count), np.nan)
    greedy_width = np.full_like(global_width, np.nan)
    absolute_regret = np.full_like(global_width, np.nan)
    relative_regret = np.full_like(global_width, np.nan)
    global_schedule = np.full(
        (instance_count, mechanism_count, config.horizon), -1, dtype=np.int16
    )
    greedy_schedule = np.full_like(global_schedule, -1)

    for instance_index, problem_seed in enumerate(seeds):
        mechanisms = build_paired_mechanisms(
            config,
            problem_seed=int(problem_seed),
        )
        for mechanism_index, mechanism in enumerate(mechanisms):
            population, width = exact_population_surfaces(mechanism, schedules)
            true_coverage = population[ESTIMATOR_NAMES.index("full_prefix")]
            identification = population - true_coverage[None]
            identification_maximum[instance_index, mechanism_index] = np.max(
                np.abs(identification), axis=(1, 2)
            )
            identification_rmse[instance_index, mechanism_index] = np.sqrt(
                np.mean(identification**2, axis=(1, 2))
            )

            feasible = np.all(true_coverage >= config.target_coverage, axis=1)
            feasible_fraction[instance_index, mechanism_index] = feasible.mean()
            if not bool(feasible.any()):
                continue
            global_available[instance_index, mechanism_index] = True
            mean_width = width.mean(axis=1)
            global_row = int(np.where(feasible, mean_width, np.inf).argmin())
            global_width[instance_index, mechanism_index] = mean_width[global_row]
            global_schedule[instance_index, mechanism_index] = schedules[global_row]
            try:
                greedy = greedy_schedule_search(
                    mechanism.problem,
                    target=config.target_coverage,
                )
            except NoFeasibleScheduleError:
                continue
            greedy_available[instance_index, mechanism_index] = True
            greedy_mean_width = float(greedy.normalized_width.mean().item())
            greedy_width[instance_index, mechanism_index] = greedy_mean_width
            greedy_schedule[instance_index, mechanism_index] = np.asarray(
                greedy.selected_indices,
                dtype=np.int16,
            )
            regret = greedy_mean_width - mean_width[global_row]
            if regret < -1e-10:
                raise RuntimeError("greedy width is smaller than the exact global width")
            regret = 0.0 if abs(regret) <= 1e-12 else regret
            absolute_regret[instance_index, mechanism_index] = regret
            relative_regret[instance_index, mechanism_index] = (
                regret / mean_width[global_row]
            )

    arrays = {
        "population_problem_seeds": seeds,
        "population_identification_maximum_absolute": identification_maximum,
        "population_identification_rmse": identification_rmse,
        "population_feasible_schedule_fraction": feasible_fraction,
        "population_global_available": global_available,
        "population_greedy_available": greedy_available,
        "population_global_mean_width": global_width,
        "population_greedy_mean_width": greedy_width,
        "population_greedy_absolute_regret": absolute_regret,
        "population_greedy_relative_regret": relative_regret,
        "population_global_schedule_indices": global_schedule,
        "population_greedy_schedule_indices": greedy_schedule,
    }
    summary = {
        "instance_count": instance_count,
        "problem_seed_range_inclusive": [int(seeds[0]), int(seeds[-1])],
        "identification": _identification_summary(
            identification_maximum,
            identification_rmse,
        ),
        "search": _search_summary(
            feasible_fraction,
            global_available,
            greedy_available,
            absolute_regret,
            relative_regret,
        ),
        "decision_gate": None,
    }
    return summary, arrays


def _run_logged_recovery_audit(
    config: ExactFiniteMDPConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    instance_count = config.logged_instance_count
    replicate_count = config.logged_replicates
    shape = (
        instance_count,
        replicate_count,
        len(MECHANISM_NAMES),
        len(ESTIMATOR_NAMES),
    )
    sampling_maximum = np.empty(shape, dtype=np.float64)
    sampling_rmse = np.empty(shape, dtype=np.float64)
    total_maximum = np.empty(shape, dtype=np.float64)
    total_rmse = np.empty(shape, dtype=np.float64)
    ess_minimum = np.empty(shape, dtype=np.float64)
    ess_median = np.empty(shape, dtype=np.float64)
    problem_seeds = np.arange(
        config.population_seed_start,
        config.population_seed_start + instance_count,
        dtype=np.int64,
    )
    randomness_seeds = np.arange(
        config.logged_replicate_seed_start,
        config.logged_replicate_seed_start + instance_count * replicate_count,
        dtype=np.int64,
    ).reshape(instance_count, replicate_count)
    schedules = enumerate_schedules(config)

    for instance_index, problem_seed in enumerate(problem_seeds):
        mechanisms = build_paired_mechanisms(
            config,
            problem_seed=int(problem_seed),
        )
        populations = [
            exact_population_surfaces(mechanism, schedules)[0]
            for mechanism in mechanisms
        ]
        for replicate_index, randomness_seed in enumerate(
            randomness_seeds[instance_index]
        ):
            randomness = generate_logged_randomness(
                config,
                seed=int(randomness_seed),
            )
            for mechanism_index, (mechanism, population) in enumerate(
                zip(mechanisms, populations, strict=True)
            ):
                logged = simulate_logged_trajectories(mechanism, randomness)
                estimates, ess = hajek_surface_estimates(
                    mechanism,
                    logged,
                    schedules,
                    chunk_size=config.surface_chunk_size,
                )
                true_coverage = population[ESTIMATOR_NAMES.index("full_prefix")]
                sampling_error = estimates - population
                total_error = estimates - true_coverage[None]
                sampling_maximum[instance_index, replicate_index, mechanism_index] = (
                    np.max(np.abs(sampling_error), axis=(1, 2))
                )
                sampling_rmse[instance_index, replicate_index, mechanism_index] = (
                    np.sqrt(np.mean(sampling_error**2, axis=(1, 2)))
                )
                total_maximum[instance_index, replicate_index, mechanism_index] = (
                    np.max(np.abs(total_error), axis=(1, 2))
                )
                total_rmse[instance_index, replicate_index, mechanism_index] = (
                    np.sqrt(np.mean(total_error**2, axis=(1, 2)))
                )
                ess_minimum[instance_index, replicate_index, mechanism_index] = (
                    np.min(ess, axis=(1, 2))
                )
                ess_median[instance_index, replicate_index, mechanism_index] = (
                    np.median(ess, axis=(1, 2))
                )

    arrays = {
        "logged_problem_seeds": problem_seeds,
        "logged_randomness_seeds": randomness_seeds,
        "logged_sampling_maximum_absolute": sampling_maximum,
        "logged_sampling_rmse": sampling_rmse,
        "logged_total_maximum_absolute": total_maximum,
        "logged_total_rmse": total_rmse,
        "logged_ess_fraction_minimum": ess_minimum,
        "logged_ess_fraction_median": ess_median,
    }
    summary = {
        "instance_count": instance_count,
        "replicates_per_instance": replicate_count,
        "logged_trajectories_per_replicate": config.logged_trajectories,
        "problem_seeds": problem_seeds.tolist(),
        "randomness_seed_range_inclusive": (
            None
            if randomness_seeds.size == 0
            else [int(randomness_seeds.min()), int(randomness_seeds.max())]
        ),
        "sampling_error": _replicate_metric_summary(
            sampling_maximum,
            sampling_rmse,
            first_name="maximum_absolute",
            second_name="root_mean_squared",
        ),
        "total_error": _replicate_metric_summary(
            total_maximum,
            total_rmse,
            first_name="maximum_absolute",
            second_name="root_mean_squared",
        ),
        "ess_fraction": _replicate_metric_summary(
            ess_minimum,
            ess_median,
            first_name="minimum",
            second_name="median",
        ),
    }
    return summary, arrays


def _identification_summary(
    maximum: np.ndarray,
    rmse: np.ndarray,
) -> dict[str, Any]:
    summary = {}
    for mechanism_index, mechanism in enumerate(MECHANISM_NAMES):
        summary[mechanism] = {}
        for estimator_index, estimator in enumerate(ESTIMATOR_NAMES):
            summary[mechanism][estimator] = {
                "maximum_absolute": _quantile_summary(
                    maximum[:, mechanism_index, estimator_index]
                ),
                "root_mean_squared": _quantile_summary(
                    rmse[:, mechanism_index, estimator_index]
                ),
            }
    return summary


def _search_summary(
    feasible_fraction: np.ndarray,
    global_available: np.ndarray,
    greedy_available: np.ndarray,
    absolute_regret: np.ndarray,
    relative_regret: np.ndarray,
) -> dict[str, Any]:
    summary = {}
    for mechanism_index, mechanism in enumerate(MECHANISM_NAMES):
        regret_mask = greedy_available[:, mechanism_index]
        summary[mechanism] = {
            "global_availability_rate": float(
                global_available[:, mechanism_index].mean()
            ),
            "greedy_availability_rate": float(regret_mask.mean()),
            "feasible_schedule_fraction": _quantile_summary(
                feasible_fraction[:, mechanism_index]
            ),
            "greedy_absolute_regret": _quantile_summary(
                absolute_regret[regret_mask, mechanism_index]
            ),
            "greedy_relative_regret": _quantile_summary(
                relative_regret[regret_mask, mechanism_index]
            ),
        }
    return summary


def _replicate_metric_summary(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_name: str,
    second_name: str,
) -> dict[str, Any]:
    summary = {}
    for mechanism_index, mechanism in enumerate(MECHANISM_NAMES):
        summary[mechanism] = {}
        for estimator_index, estimator in enumerate(ESTIMATOR_NAMES):
            summary[mechanism][estimator] = {
                first_name: _quantile_summary(
                    first[:, :, mechanism_index, estimator_index].reshape(-1)
                ),
                second_name: _quantile_summary(
                    second[:, :, mechanism_index, estimator_index].reshape(-1)
                ),
            }
    return summary


def _quantile_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "median": None, "q05": None, "q95": None}
    return {
        "count": int(finite.size),
        "minimum": float(finite.min()),
        "q05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "q95": float(np.quantile(finite, 0.95)),
        "maximum": float(finite.max()),
        "mean": float(finite.mean()),
    }
