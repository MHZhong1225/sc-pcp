"""Exact finite-MDP diagnostics for committed-prefix transport.

This module is intentionally separate from the paper implementation.  It builds
four paired population mechanisms, enumerates their complete radius grids, and
separates population identification bias from finite logged-sample error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math
from typing import Any

import numpy as np
import torch

from scpcp.phase0_search import (
    AnalyticFiniteMDP,
    NoFeasibleScheduleError,
    ScheduleEvaluation,
    beam_schedule_search,
    exact_schedule_search,
)


MECHANISM_NAMES = (
    "M0_no_feedback",
    "M1_current_only",
    "M2_history_only",
    "M3_full_feedback",
)
ESTIMATOR_NAMES = (
    "unweighted",
    "history_only",
    "current_only",
    "full_prefix",
)
SEED_NAMESPACE = "finite_mdp_52000_52999"
EXTERNAL_SEED_RESERVATIONS = {
    "controlled_six_method": tuple(range(91_000, 91_191, 10)),
    "orthogonal_copula_formal": tuple(range(94_000, 94_199, 2)),
}


@dataclass(frozen=True)
class ExactFiniteMDPConfig:
    """Frozen defaults for the isolated exact finite-MDP study."""

    state_count: int = 8
    action_count: int = 3
    horizon: int = 4
    grid_size: int = 7
    alpha: float = 0.10
    logged_trajectories: int = 3_000
    seed: int = 52_081
    population_instances: int = 200
    population_seed_start: int = 52_100
    logged_instance_count: int = 4
    logged_replicates: int = 3
    logged_replicate_seed_start: int = 52_600
    radius_minimum: float = 1.4
    radius_maximum: float = 3.5
    policy_response_center: float = 2.5
    policy_response_scale: float = 0.7
    policy_response_strength: float = 3.0
    beam_width: int = 32
    surface_chunk_size: int = 128

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    @property
    def schedule_count(self) -> int:
        return self.grid_size**self.horizon

    def validate(self) -> None:
        if self.state_count < 4:
            raise ValueError("state_count must be at least four")
        if self.action_count != 3:
            raise ValueError("the paired mechanisms require exactly three actions")
        if self.horizon < 1 or self.grid_size < 2:
            raise ValueError("horizon must be positive and grid_size at least two")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.logged_trajectories < 1:
            raise ValueError("logged_trajectories must be positive")
        if self.population_instances < 1:
            raise ValueError("population_instances must be positive")
        if not 0 <= self.logged_instance_count <= self.population_instances:
            raise ValueError(
                "logged_instance_count must lie between zero and population_instances"
            )
        if self.logged_replicates < 1:
            raise ValueError("logged_replicates must be positive")
        population_seeds = range(
            self.population_seed_start,
            self.population_seed_start + self.population_instances,
        )
        logged_seed_count = self.logged_instance_count * self.logged_replicates
        logged_seeds = range(
            self.logged_replicate_seed_start,
            self.logged_replicate_seed_start + logged_seed_count,
        )
        if self.seed in population_seeds or self.seed in logged_seeds:
            raise ValueError("fixture seed collides with a replicated-study seed")
        if max(population_seeds.start, logged_seeds.start) < min(
            population_seeds.stop,
            logged_seeds.stop,
        ):
            raise ValueError("population and logged-replicate seed ranges overlap")
        collision_audit = exact_seed_collision_audit(self)
        if collision_audit["collision"]:
            raise ValueError(
                "exact finite-MDP seeds collide with another study: "
                f"{collision_audit['colliding_seeds']}"
            )
        if not self.radius_minimum < self.radius_maximum:
            raise ValueError("radius_minimum must be smaller than radius_maximum")
        if self.policy_response_scale <= 0.0:
            raise ValueError("policy_response_scale must be positive")
        if self.policy_response_strength < 0.0:
            raise ValueError("policy_response_strength must be nonnegative")
        if self.beam_width < 1 or self.surface_chunk_size < 1:
            raise ValueError("beam_width and surface_chunk_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def exact_seed_collision_audit(config: ExactFiniteMDPConfig) -> dict[str, Any]:
    """Report collisions against the other frozen scientific seed banks."""

    population = set(
        range(
            config.population_seed_start,
            config.population_seed_start + config.population_instances,
        )
    )
    logged = set(
        range(
            config.logged_replicate_seed_start,
            config.logged_replicate_seed_start
            + config.logged_instance_count * config.logged_replicates,
        )
    )
    active = {config.seed, *population, *logged}
    collisions = {
        name: sorted(active.intersection(reserved))
        for name, reserved in EXTERNAL_SEED_RESERVATIONS.items()
    }
    collisions = {name: values for name, values in collisions.items() if values}
    return {
        "fixture_seed": config.seed,
        "population_seed_range_inclusive": [
            config.population_seed_start,
            config.population_seed_start + config.population_instances - 1,
        ],
        "logged_seed_range_inclusive": (
            None
            if not logged
            else [min(logged), max(logged)]
        ),
        "external_reservations": {
            "controlled_six_method": "91000..91190 step 10",
            "orthogonal_copula_formal": "94000..94198 even",
        },
        "collision": bool(collisions),
        "colliding_seeds": collisions,
    }


@dataclass(frozen=True)
class PairedMechanism:
    name: str
    description: str
    behavior_probabilities: np.ndarray
    problem: AnalyticFiniteMDP


@dataclass(frozen=True)
class LoggedRandomness:
    initial_uniform: np.ndarray
    action_uniform: np.ndarray
    transition_uniform: np.ndarray
    outcome_standard_normal: np.ndarray


@dataclass(frozen=True)
class LoggedTrajectories:
    states: np.ndarray
    actions: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class ExactFiniteMDPResult:
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]


def build_paired_mechanisms(
    config: ExactFiniteMDPConfig,
    *,
    problem_seed: int | None = None,
) -> tuple[PairedMechanism, ...]:
    """Build one paired M0--M3 instance.

    ``problem_seed=None`` is the deterministic fixture.  Integer seeds make
    bounded perturbations of the shared MDP parameters while preserving all
    four mechanism-specific structural equalities.
    """

    config.validate()
    generator = None if problem_seed is None else np.random.default_rng(problem_seed)
    state_count = config.state_count
    action_count = config.action_count
    horizon = config.horizon
    grid_size = config.grid_size
    states = np.arange(state_count, dtype=np.float64)

    center = 3.0 * (state_count - 1) / 7.0
    spread = max(0.8, 1.3 * (state_count - 1) / 7.0)
    if generator is not None:
        center += generator.uniform(-0.2, 0.2) * (state_count - 1) / 7.0
        spread *= generator.uniform(0.92, 1.08)
    initial = np.exp(-0.5 * ((states - center) / spread) ** 2)
    if generator is not None:
        initial *= np.exp(generator.uniform(-0.12, 0.12, state_count))
    initial /= initial.sum()

    behavior_logits = np.broadcast_to(
        np.array([-0.3, 0.6, -0.3], dtype=np.float64),
        (state_count, action_count),
    ).copy()
    if generator is not None:
        behavior_logits += generator.uniform(-0.12, 0.12, action_count)[None, :]
        normalized_state = 2.0 * states / (state_count - 1) - 1.0
        behavior_logits += normalized_state[:, None] * generator.uniform(
            -0.07, 0.07, action_count
        )[None, :]
    behavior = _softmax(behavior_logits)
    action_dependent_transition = _action_dependent_transition(
        state_count,
        generator=generator,
    )
    behavior_transition = np.einsum(
        "sa,asr->sr", behavior, action_dependent_transition
    )
    action_independent_transition = np.broadcast_to(
        behavior_transition,
        (action_count, state_count, state_count),
    ).copy()

    radius_grid = np.linspace(
        config.radius_minimum,
        config.radius_maximum,
        grid_size,
        dtype=np.float64,
    )
    action_direction = np.array([1.0, 0.0, -1.0], dtype=np.float64)
    response_center = config.policy_response_center
    response_strength = config.policy_response_strength
    if generator is not None:
        response_center += generator.uniform(-0.08, 0.08)
        response_strength *= generator.uniform(0.90, 1.10)
    target_by_radius = np.stack(
        [
            _softmax(
                np.log(behavior)
                + response_strength
                * ((radius - response_center) / config.policy_response_scale)
                * action_direction[None, :]
            )
            for radius in radius_grid
        ]
    )
    target_policy = np.broadcast_to(
        target_by_radius,
        (horizon, grid_size, state_count, action_count),
    ).copy()
    behavior_policy = np.broadcast_to(
        behavior,
        (horizon, grid_size, state_count, action_count),
    ).copy()
    radii = np.broadcast_to(radius_grid, (horizon, grid_size)).copy()

    state_fraction = states / (state_count - 1)
    state_scale = 2.1
    state_power = 1.0
    action_effect = np.array([0.1, 0.7, 1.3], dtype=np.float64)
    if generator is not None:
        state_scale *= generator.uniform(0.93, 1.04)
        state_power = generator.uniform(0.90, 1.10)
        action_effect *= generator.uniform(0.92, 1.05)
        action_effect += generator.uniform(-0.025, 0.025, action_count)
        action_effect = np.clip(action_effect, 0.05, 1.35)
    state_effect = 0.05 + state_scale * state_fraction**state_power
    predictor_means = {
        "M0_no_feedback": 0.5 * state_effect[:, None]
        + 0.5 * action_effect[None, :],
        "M1_current_only": np.broadcast_to(
            action_effect,
            (state_count, action_count),
        ).copy(),
        "M2_history_only": np.broadcast_to(
            state_effect[:, None],
            (state_count, action_count),
        ).copy(),
        "M3_full_feedback": 0.75 * state_effect[:, None]
        + 0.5 * action_effect[None, :],
    }
    definitions = (
        (
            "M0_no_feedback",
            "target policy equals behavior; all transport estimands coincide",
            behavior_policy,
            action_independent_transition,
        ),
        (
            "M1_current_only",
            "radius changes only the current-action component of the score law",
            target_policy,
            action_independent_transition,
        ),
        (
            "M2_history_only",
            "past actions change state occupancy; current action has no direct score effect",
            target_policy,
            action_dependent_transition,
        ),
        (
            "M3_full_feedback",
            "past state occupancy and current action both change the score law",
            target_policy,
            action_dependent_transition,
        ),
    )

    mechanisms = []
    for name, description, action_probabilities, transition in definitions:
        problem = AnalyticFiniteMDP(
            initial_state_probabilities=torch.from_numpy(initial.copy()),
            transition_probabilities=torch.from_numpy(transition.copy()),
            action_probabilities=torch.from_numpy(action_probabilities.copy()),
            radii=torch.from_numpy(radii.copy()),
            predictor_means=torch.from_numpy(predictor_means[name][:, :, None]),
            predictor_scales=torch.ones(
                state_count,
                action_count,
                1,
                dtype=torch.float64,
            ),
            outcome_means=torch.zeros(
                action_count,
                state_count,
                1,
                dtype=torch.float64,
            ),
            outcome_standard_deviations=torch.ones(1, dtype=torch.float64),
            outcome_normalization=torch.ones(1, dtype=torch.float64),
        )
        mechanisms.append(
            PairedMechanism(
                name=name,
                description=description,
                behavior_probabilities=behavior.copy(),
                problem=problem,
            )
        )
    return tuple(mechanisms)


def enumerate_schedules(config: ExactFiniteMDPConfig) -> np.ndarray:
    config.validate()
    return np.asarray(
        list(product(range(config.grid_size), repeat=config.horizon)),
        dtype=np.int16,
    )


def exact_population_surfaces(
    mechanism: PairedMechanism,
    schedules: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact partial-weight estimands and target-policy widths.

    Estimator axis order is ``ESTIMATOR_NAMES``.  The full-prefix slice is the
    exact target-policy coverage surface.
    """

    problem = mechanism.problem
    horizon, grid_size = problem.radii.shape
    if schedules.ndim != 2 or schedules.shape[1] != horizon:
        raise ValueError("schedules must have shape [N,T]")
    if np.any(schedules < 0) or np.any(schedules >= grid_size):
        raise ValueError("schedule index is outside the finite grid")

    initial = _numpy(problem.initial_state_probabilities)
    transition = _numpy(problem.transition_probabilities)
    target_policy = _numpy(problem.action_probabilities)
    behavior = mechanism.behavior_probabilities
    conditional_hits = _conditional_hit_probabilities(problem)

    target_coverage_by_state = np.einsum(
        "tksa,asr,tksar->tks",
        target_policy,
        transition,
        conditional_hits,
    )
    behavior_coverage_by_state = np.einsum(
        "sa,asr,tksar->tks",
        behavior,
        transition,
        conditional_hits,
    )
    target_transition = np.einsum(
        "tksa,asr->tksr", target_policy, transition
    )
    behavior_transition = np.einsum("sa,asr->sr", behavior, transition)

    base_width = 2.0 * np.mean(
        _numpy(problem.predictor_scales)
        / _numpy(problem.outcome_normalization)[None, None, :],
        axis=2,
    )
    conditional_width = (
        _numpy(problem.radii)[:, :, None, None]
        * base_width[None, None, :, :]
    )
    target_width_by_state = np.einsum(
        "tksa,tksa->tks", target_policy, conditional_width
    )

    schedule_count = len(schedules)
    population = np.empty(
        (len(ESTIMATOR_NAMES), schedule_count, horizon), dtype=np.float64
    )
    target_width = np.empty((schedule_count, horizon), dtype=np.float64)
    target_occupancy = np.broadcast_to(initial, (schedule_count, len(initial))).copy()
    behavior_occupancy = initial.copy()
    for stage in range(horizon):
        indices = schedules[:, stage]
        target_current = target_coverage_by_state[stage, indices]
        behavior_current = behavior_coverage_by_state[stage, indices]
        population[0, :, stage] = np.einsum(
            "s,ns->n", behavior_occupancy, behavior_current
        )
        population[1, :, stage] = np.einsum(
            "ns,ns->n", target_occupancy, behavior_current
        )
        population[2, :, stage] = np.einsum(
            "s,ns->n", behavior_occupancy, target_current
        )
        population[3, :, stage] = np.einsum(
            "ns,ns->n", target_occupancy, target_current
        )
        target_width[:, stage] = np.einsum(
            "ns,ns->n",
            target_occupancy,
            target_width_by_state[stage, indices],
        )
        target_occupancy = np.einsum(
            "ns,nsr->nr",
            target_occupancy,
            target_transition[stage, indices],
        )
        behavior_occupancy = behavior_occupancy @ behavior_transition
    return population, target_width


def generate_logged_randomness(
    config: ExactFiniteMDPConfig,
    *,
    outcome_dimension: int = 1,
    seed: int | None = None,
) -> LoggedRandomness:
    """Generate common random numbers shared by all four mechanisms."""

    config.validate()
    generator = np.random.default_rng(config.seed if seed is None else seed)
    shape = (config.logged_trajectories, config.horizon)
    return LoggedRandomness(
        initial_uniform=generator.random(config.logged_trajectories),
        action_uniform=generator.random(shape),
        transition_uniform=generator.random(shape),
        outcome_standard_normal=generator.standard_normal(
            (*shape, outcome_dimension)
        ),
    )


def simulate_logged_trajectories(
    mechanism: PairedMechanism,
    randomness: LoggedRandomness,
) -> LoggedTrajectories:
    """Simulate behavior-policy trajectories without target-policy feedback."""

    problem = mechanism.problem
    horizon = problem.radii.shape[0]
    trajectory_count = len(randomness.initial_uniform)
    if randomness.action_uniform.shape != (trajectory_count, horizon):
        raise ValueError("logged action randomness has the wrong shape")
    outcome_dimension = problem.predictor_means.shape[2]
    if randomness.outcome_standard_normal.shape != (
        trajectory_count,
        horizon,
        outcome_dimension,
    ):
        raise ValueError("logged outcome randomness has the wrong shape")

    initial = _numpy(problem.initial_state_probabilities)
    transition = _numpy(problem.transition_probabilities)
    predictor_means = _numpy(problem.predictor_means)
    predictor_scales = _numpy(problem.predictor_scales)
    outcome_means = _numpy(problem.outcome_means)
    outcome_sd = _numpy(problem.outcome_standard_deviations)
    states = np.empty((trajectory_count, horizon), dtype=np.int16)
    actions = np.empty((trajectory_count, horizon), dtype=np.int8)
    scores = np.empty((trajectory_count, horizon), dtype=np.float64)
    current_states = _draw_categorical(
        np.broadcast_to(initial, (trajectory_count, len(initial))),
        randomness.initial_uniform,
    )
    for stage in range(horizon):
        states[:, stage] = current_states
        action_probabilities = mechanism.behavior_probabilities[current_states]
        current_actions = _draw_categorical(
            action_probabilities,
            randomness.action_uniform[:, stage],
        )
        actions[:, stage] = current_actions
        next_probabilities = transition[current_actions, current_states]
        next_states = _draw_categorical(
            next_probabilities,
            randomness.transition_uniform[:, stage],
        )
        outcomes = (
            outcome_means[current_actions, next_states]
            + outcome_sd[None, :]
            * randomness.outcome_standard_normal[:, stage]
        )
        standardized_residual = np.abs(
            (outcomes - predictor_means[current_states, current_actions])
            / predictor_scales[current_states, current_actions]
        )
        scores[:, stage] = standardized_residual.max(axis=1)
        current_states = next_states
    return LoggedTrajectories(states=states, actions=actions, scores=scores)


def hajek_surface_estimates(
    mechanism: PairedMechanism,
    logged: LoggedTrajectories,
    schedules: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover every schedule surface with four uncapped Hájek estimators."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    problem = mechanism.problem
    horizon, grid_size = problem.radii.shape
    trajectory_count = len(logged.states)
    if logged.states.shape != (trajectory_count, horizon):
        raise ValueError("logged states must have shape [N,T]")
    target_policy = _numpy(problem.action_probabilities)
    radii = _numpy(problem.radii)
    ratios = np.empty((horizon, grid_size, trajectory_count), dtype=np.float64)
    hits = np.empty_like(ratios, dtype=bool)
    for stage in range(horizon):
        states = logged.states[:, stage]
        actions = logged.actions[:, stage]
        behavior_probability = mechanism.behavior_probabilities[states, actions]
        target_probability = target_policy[stage, :, states, actions].T
        ratios[stage] = target_probability / behavior_probability[None, :]
        hits[stage] = logged.scores[:, stage][None, :] <= radii[stage, :, None]

    estimates = np.empty(
        (len(ESTIMATOR_NAMES), len(schedules), horizon), dtype=np.float64
    )
    ess_fraction = np.empty_like(estimates)
    for start in range(0, len(schedules), chunk_size):
        stop = min(len(schedules), start + chunk_size)
        chunk = schedules[start:stop]
        prefix_weight = np.ones((len(chunk), trajectory_count), dtype=np.float64)
        for stage in range(horizon):
            indices = chunk[:, stage]
            current_ratio = ratios[stage, indices]
            current_hits = hits[stage, indices]
            weights = (
                np.ones_like(prefix_weight),
                prefix_weight,
                current_ratio,
                prefix_weight * current_ratio,
            )
            for estimator_index, weight in enumerate(weights):
                estimates[estimator_index, start:stop, stage] = _weighted_mean(
                    current_hits,
                    weight,
                )
                ess_fraction[estimator_index, start:stop, stage] = (
                    _effective_sample_size(weight) / trajectory_count
                )
            prefix_weight = weights[3]
    return estimates, ess_fraction


def run_exact_finite_mdp(
    config: ExactFiniteMDPConfig,
) -> ExactFiniteMDPResult:
    """Run the complete analytic and one-sample recovery diagnostic."""

    config.validate()
    mechanisms = build_paired_mechanisms(config)
    schedules = enumerate_schedules(config)
    randomness = generate_logged_randomness(config)
    population = np.empty(
        (
            len(mechanisms),
            len(ESTIMATOR_NAMES),
            len(schedules),
            config.horizon,
        ),
        dtype=np.float64,
    )
    target_width = np.empty(
        (len(mechanisms), len(schedules), config.horizon), dtype=np.float64
    )
    hajek = np.empty_like(population)
    ess_fraction = np.empty_like(population)
    search: dict[str, Any] = {}
    mechanism_descriptions = {}

    for mechanism_index, mechanism in enumerate(mechanisms):
        mechanism_descriptions[mechanism.name] = mechanism.description
        population[mechanism_index], target_width[mechanism_index] = (
            exact_population_surfaces(mechanism, schedules)
        )
        logged = simulate_logged_trajectories(mechanism, randomness)
        hajek[mechanism_index], ess_fraction[mechanism_index] = (
            hajek_surface_estimates(
                mechanism,
                logged,
                schedules,
                chunk_size=config.surface_chunk_size,
            )
        )
        search[mechanism.name] = _run_search_diagnostics(
            mechanism,
            schedules,
            population[mechanism_index, ESTIMATOR_NAMES.index("full_prefix")],
            target_width[mechanism_index],
            target=config.target_coverage,
            beam_width=config.beam_width,
        )

    true_coverage = population[:, ESTIMATOR_NAMES.index("full_prefix")]
    identification_bias = population - true_coverage[:, None]
    sampling_error = hajek - population
    total_error = hajek - true_coverage[:, None]
    summary = {
        "schema_version": 1,
        "study": "exact_committed_prefix_finite_mdp",
        "status": "complete",
        "diagnostic_only": True,
        "canonical_method_unchanged": True,
        "population_exact": True,
        "finite_sample_claim": False,
        "target_coverage": config.target_coverage,
        "schedule_count": len(schedules),
        "seed_namespace": SEED_NAMESPACE,
        "paired_common_random_numbers": True,
        "mechanisms": mechanism_descriptions,
        "estimators": {
            "unweighted": "behavior occupancy and behavior current action",
            "history_only": "target committed history and behavior current action",
            "current_only": "behavior history and target current action",
            "full_prefix": "target committed history and target current action",
        },
        "error_decomposition": (
            "total_error = identification_bias + finite_sample_sampling_error"
        ),
        "search": search,
        "surface_recovery": _surface_recovery_summary(
            identification_bias,
            sampling_error,
            total_error,
            ess_fraction,
        ),
    }
    arrays = {
        "schedule_indices": schedules,
        "radius_grid": _numpy(mechanisms[0].problem.radii),
        "mechanism_names": np.asarray(MECHANISM_NAMES),
        "estimator_names": np.asarray(ESTIMATOR_NAMES),
        "true_coverage": true_coverage,
        "population_coverage": population,
        "hajek_coverage": hajek,
        "identification_bias": identification_bias,
        "finite_sample_sampling_error": sampling_error,
        "total_error": total_error,
        "ess_fraction": ess_fraction,
        "target_normalized_width": target_width,
    }
    return ExactFiniteMDPResult(summary=summary, arrays=arrays)


def _run_search_diagnostics(
    mechanism: PairedMechanism,
    schedules: np.ndarray,
    true_coverage: np.ndarray,
    target_width: np.ndarray,
    *,
    target: float,
    beam_width: int,
) -> dict[str, Any]:
    feasible = np.all(true_coverage >= target, axis=1)
    if not bool(feasible.any()):
        return {
            "grid_schedule_count": len(schedules),
            "feasible_schedule_count": 0,
            "availability_fraction": 0.0,
            "global_available": False,
            "global": None,
            "greedy_available": False,
            "greedy": None,
            "greedy_absolute_regret": None,
            "greedy_relative_regret": None,
            "beam": {
                "diagnostic_only": True,
                "beam_width": beam_width,
                "available": False,
                "best_found": None,
                "best_found_minus_global_width": None,
                "changes_canonical_method": False,
            },
            "decision_gate": None,
        }

    exact = exact_schedule_search(mechanism.problem, target=target)
    try:
        beam = beam_schedule_search(
            mechanism.problem,
            target=target,
            beam_width=beam_width,
        )
    except NoFeasibleScheduleError:
        beam = None
    exact_payload = _schedule_payload(mechanism.problem, exact.best_found_schedule)
    greedy_payload = (
        None
        if exact.greedy_schedule is None
        else _schedule_payload(mechanism.problem, exact.greedy_schedule)
    )
    beam_payload = (
        None
        if beam is None
        else _schedule_payload(mechanism.problem, beam.best_found_schedule)
    )
    global_width = exact_payload["mean_normalized_width"]
    greedy_regret = (
        None
        if greedy_payload is None
        else greedy_payload["mean_normalized_width"] - global_width
    )
    beam_global_gap = (
        None
        if beam_payload is None
        else beam_payload["mean_normalized_width"] - global_width
    )

    global_row = _schedule_row(
        exact.best_found_schedule.selected_indices,
        grid_size=mechanism.problem.radii.shape[1],
    )
    if not np.allclose(
        true_coverage[global_row],
        exact.best_found_schedule.coverage.detach().cpu().numpy(),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("exact surface and phase0 search recursion disagree")
    if not math.isclose(
        float(target_width[global_row].mean()),
        global_width,
        abs_tol=1e-12,
        rel_tol=0.0,
    ):
        raise RuntimeError("exact width surface and phase0 search disagree")

    return {
        "grid_schedule_count": len(schedules),
        "feasible_schedule_count": int(feasible.sum()),
        "availability_fraction": float(feasible.mean()),
        "global_available": True,
        "global": exact_payload,
        "greedy_available": exact.greedy_available,
        "greedy": greedy_payload,
        "greedy_absolute_regret": greedy_regret,
        "greedy_relative_regret": (
            None if greedy_regret is None else greedy_regret / global_width
        ),
        "beam": {
            "diagnostic_only": True,
            "beam_width": beam_width,
            "available": beam_payload is not None,
            "best_found": beam_payload,
            "best_found_minus_global_width": beam_global_gap,
            "changes_canonical_method": False,
        },
        "decision_gate": None,
    }


def _surface_recovery_summary(
    identification_bias: np.ndarray,
    sampling_error: np.ndarray,
    total_error: np.ndarray,
    ess_fraction: np.ndarray,
) -> dict[str, Any]:
    summary = {}
    for mechanism_index, mechanism_name in enumerate(MECHANISM_NAMES):
        estimators = {}
        for estimator_index, estimator_name in enumerate(ESTIMATOR_NAMES):
            estimators[estimator_name] = {
                "identification_bias": _error_metrics(
                    identification_bias[mechanism_index, estimator_index]
                ),
                "finite_sample_sampling_error": _error_metrics(
                    sampling_error[mechanism_index, estimator_index]
                ),
                "total_error": _error_metrics(
                    total_error[mechanism_index, estimator_index]
                ),
                "ess_fraction_minimum": float(
                    ess_fraction[mechanism_index, estimator_index].min()
                ),
                "ess_fraction_median": float(
                    np.median(ess_fraction[mechanism_index, estimator_index])
                ),
            }
        summary[mechanism_name] = estimators
    return summary


def _error_metrics(error: np.ndarray) -> dict[str, Any]:
    return {
        "maximum_absolute": float(np.abs(error).max()),
        "mean_absolute": float(np.abs(error).mean()),
        "root_mean_squared": float(np.sqrt(np.mean(error**2))),
        "stagewise_root_mean_squared": [
            float(np.sqrt(np.mean(error[:, stage] ** 2)))
            for stage in range(error.shape[1])
        ],
    }


def _conditional_hit_probabilities(problem: AnalyticFiniteMDP) -> np.ndarray:
    radii = problem.radii[:, :, None, None, None, None]
    predictor_means = problem.predictor_means[None, None, :, :, None, :]
    predictor_scales = problem.predictor_scales[None, None, :, :, None, :]
    outcome_means = problem.outcome_means[None, None, None, :, :, :]
    outcome_sd = problem.outcome_standard_deviations[
        None, None, None, None, None, :
    ]
    lower = (predictor_means - radii * predictor_scales - outcome_means) / outcome_sd
    upper = (predictor_means + radii * predictor_scales - outcome_means) / outcome_sd
    return _numpy(
        (torch.special.ndtr(upper) - torch.special.ndtr(lower))
        .clamp(0.0, 1.0)
        .prod(dim=5)
    )


def _schedule_payload(
    problem: AnalyticFiniteMDP,
    evaluation: ScheduleEvaluation,
) -> dict[str, Any]:
    stage_indices = np.arange(len(evaluation.selected_indices))
    indices = np.asarray(evaluation.selected_indices, dtype=np.int64)
    radii = _numpy(problem.radii)[stage_indices, indices]
    width = _numpy(evaluation.normalized_width)
    return {
        "schedule_indices": indices.tolist(),
        "radii": radii.tolist(),
        "coverage": _numpy(evaluation.coverage).tolist(),
        "normalized_width": width.tolist(),
        "mean_normalized_width": float(width.mean()),
    }


def _schedule_row(indices: tuple[int, ...], *, grid_size: int) -> int:
    row = 0
    for index in indices:
        row = row * grid_size + index
    return row


def _draw_categorical(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    draws = np.sum(uniforms[:, None] > cumulative, axis=1)
    return np.minimum(draws, probabilities.shape[1] - 1).astype(np.int64)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = weights.sum(axis=1)
    return np.einsum("cn,cn->c", values, weights) / denominator


def _effective_sample_size(weights: np.ndarray) -> np.ndarray:
    return weights.sum(axis=1) ** 2 / np.einsum("cn,cn->c", weights, weights)


def _action_dependent_transition(
    state_count: int,
    *,
    generator: np.random.Generator | None,
) -> np.ndarray:
    transition = np.zeros((3, state_count, state_count), dtype=np.float64)
    shift_size = max(1, round(3 * (state_count - 1) / 7))
    for action, shift in enumerate((-shift_size, 0, shift_size)):
        for state in range(state_count):
            destination = int(np.clip(state + shift, 0, state_count - 1))
            if generator is None:
                diffuse = np.full(state_count, 0.02 / state_count)
                stay_mass = 0.05
            else:
                diffuse_mass = generator.uniform(0.015, 0.035)
                diffuse = diffuse_mass * generator.dirichlet(
                    np.full(state_count, 2.0)
                )
                stay_mass = generator.uniform(0.04, 0.07)
            transition[action, state] += diffuse
            destination_mass = 1.0 - diffuse.sum() - stay_mass
            transition[action, state, destination] += destination_mass
            transition[action, state, state] += stay_mass
    transition /= transition.sum(axis=2, keepdims=True)
    return transition


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=-1, keepdims=True)
    exponential = np.exp(centered)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
