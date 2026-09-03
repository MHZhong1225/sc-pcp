"""Isolated RQ6 calibration-size convergence study.

The module reuses the exact finite-MDP kernels while leaving the canonical
SC-PCP selector untouched.  Track A studies the complete fixed-grid prefix
surface; Track B calls the paper selector on D_COT empirical grids and then
evaluates the selected schedule by exact population recursion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import yaml

from scpcp.coverage import fixed_q_grid
from scpcp.data import TrajectoryBatch
from scpcp.exact_finite_mdp import (
    ESTIMATOR_NAMES,
    ExactFiniteMDPConfig,
    LoggedTrajectories,
    PairedMechanism,
    build_paired_mechanisms,
    enumerate_schedules,
    exact_population_surfaces,
    generate_logged_randomness,
    simulate_logged_trajectories,
)
from scpcp.marginal_prefix import select_marginal_prefix_schedule
from scpcp.phase0_search import analytic_schedule_metrics


PROTOCOL = "rq6_ncal_convergence_v1"
FROZEN_N_CALIBRATION = (250, 500, 1_000, 2_000, 5_000, 10_000)
FROZEN_PROBLEM_SEEDS = tuple(range(97_000, 97_100))
FROZEN_LOGGED_RNG_START = 97_100_000
FROZEN_BOOTSTRAP_RNG = 97_900_000
FROZEN_SEED_NAMESPACE = "rq6_ncal_convergence_v1:97000"
TRACK_A_PREFIX_COUNTS = (7, 49, 343, 2_401)


@dataclass(frozen=True)
class RQ6ConvergenceConfig:
    """Typed configuration for the formal RQ6 protocol."""

    protocol: str = PROTOCOL
    state_count: int = 8
    action_count: int = 3
    horizon: int = 4
    grid_size: int = 7
    alpha: float = 0.10
    radius_minimum: float = 1.40
    radius_maximum: float = 3.50
    q_quantile_minimum: float = 0.50
    q_quantile_maximum: float = 0.999
    policy_reference_tv: float = 0.05
    n_calibration: tuple[int, ...] = FROZEN_N_CALIBRATION
    cot_role_parts: int = 1
    certification_role_parts: int = 2
    problem_seed_start: int = 97_000
    problem_count: int = 100
    logged_replicates: int = 20
    logged_rng_start: int = FROZEN_LOGGED_RNG_START
    bootstrap_rng: int = FROZEN_BOOTSTRAP_RNG
    bootstrap_resamples: int = 10_000
    surface_chunk_size: int = 128
    workers: int = 4
    output_dir: Path = Path("results/work/rq6_ncal_convergence_v1")
    seed_namespace: str = FROZEN_SEED_NAMESPACE

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RQ6ConvergenceConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError("RQ6 configuration root must be a mapping")
        values = dict(raw)
        if "n_calibration" in values:
            values["n_calibration"] = tuple(values["n_calibration"])
        if "output_dir" in values:
            values["output_dir"] = Path(values["output_dir"])
        config = cls(**values)
        config.validate()
        config.assert_frozen_protocol()
        return config

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    @property
    def problem_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.problem_seed_start, self.problem_seed_start + self.problem_count))

    @property
    def reference_radius(self) -> float:
        return 0.5 * (self.radius_minimum + self.radius_maximum)

    @property
    def maximum_role_sizes(self) -> tuple[int, int]:
        return calibration_role_sizes(max(self.n_calibration), self)

    def with_runtime_overrides(
        self,
        *,
        output_dir: Path | None = None,
        workers: int | None = None,
        surface_chunk_size: int | None = None,
    ) -> "RQ6ConvergenceConfig":
        config = replace(
            self,
            output_dir=self.output_dir if output_dir is None else output_dir,
            workers=self.workers if workers is None else workers,
            surface_chunk_size=(
                self.surface_chunk_size
                if surface_chunk_size is None
                else surface_chunk_size
            ),
        )
        config.validate()
        config.assert_frozen_protocol()
        return config

    def validate(self) -> None:
        if self.protocol != PROTOCOL:
            raise ValueError("unknown RQ6 convergence protocol")
        if (self.state_count, self.action_count) != (8, 3):
            raise ValueError("RQ6 finite MDP requires eight states and three actions")
        if self.horizon < 1 or self.grid_size < 2:
            raise ValueError("horizon and grid_size must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not self.radius_minimum < self.radius_maximum:
            raise ValueError("radius bounds are invalid")
        if not 0.0 <= self.q_quantile_minimum < self.q_quantile_maximum <= 1.0:
            raise ValueError("empirical-grid quantiles are invalid")
        if not 0.0 < self.policy_reference_tv < 1.0:
            raise ValueError("policy_reference_tv must lie in (0, 1)")
        if (
            not self.n_calibration
            or tuple(sorted(set(self.n_calibration))) != self.n_calibration
            or min(self.n_calibration) < 3
        ):
            raise ValueError("n_calibration must be a sorted unique positive tuple")
        if self.cot_role_parts != 1 or self.certification_role_parts != 2:
            raise ValueError("RQ6 freezes D_COT:D_cert at 1:2")
        if self.problem_count < 1 or self.logged_replicates < 1:
            raise ValueError("problem_count and logged_replicates must be positive")
        if self.bootstrap_resamples < 1 or self.surface_chunk_size < 1 or self.workers < 1:
            raise ValueError("bootstrap, chunk, and worker counts must be positive")
        if len(set(self.problem_seeds)) != len(self.problem_seeds):
            raise ValueError("problem seeds must be unique")

    def assert_frozen_protocol(self) -> None:
        expected = {
            "protocol": PROTOCOL,
            "state_count": 8,
            "action_count": 3,
            "horizon": 4,
            "grid_size": 7,
            "alpha": 0.10,
            "radius_minimum": 1.40,
            "radius_maximum": 3.50,
            "q_quantile_minimum": 0.50,
            "q_quantile_maximum": 0.999,
            "policy_reference_tv": 0.05,
            "n_calibration": FROZEN_N_CALIBRATION,
            "cot_role_parts": 1,
            "certification_role_parts": 2,
            "problem_seed_start": 97_000,
            "problem_count": 100,
            "logged_replicates": 20,
            "logged_rng_start": FROZEN_LOGGED_RNG_START,
            "bootstrap_rng": FROZEN_BOOTSTRAP_RNG,
            "bootstrap_resamples": 10_000,
            "seed_namespace": FROZEN_SEED_NAMESPACE,
        }
        observed = {name: getattr(self, name) for name in expected}
        if observed != expected:
            raise ValueError("RQ6 YAML differs from the frozen formal protocol")
        if tuple(self.grid_size ** (stage + 1) for stage in range(self.horizon)) != (
            TRACK_A_PREFIX_COUNTS
        ):
            raise ValueError("RQ6 unique-prefix counts differ from 7/49/343/2401")

    def exact_config(self, *, logged_trajectories: int, seed: int) -> ExactFiniteMDPConfig:
        config = ExactFiniteMDPConfig(
            state_count=self.state_count,
            action_count=self.action_count,
            horizon=self.horizon,
            grid_size=self.grid_size,
            alpha=self.alpha,
            logged_trajectories=logged_trajectories,
            seed=seed,
            population_instances=1,
            population_seed_start=52_100,
            logged_instance_count=0,
            logged_replicates=1,
            logged_replicate_seed_start=52_600,
            radius_minimum=self.radius_minimum,
            radius_maximum=self.radius_maximum,
            surface_chunk_size=self.surface_chunk_size,
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["n_calibration"] = list(self.n_calibration)
        values["output_dir"] = str(self.output_dir)
        return values


@dataclass(frozen=True)
class OutcomeBlindRadiusPolicy:
    """Behavior-anchored policy whose radius response never reads outcomes."""

    behavior_probabilities: np.ndarray
    radius_minimum: float
    radius_maximum: float
    logit_strength: float

    def numpy_probabilities(
        self,
        states: np.ndarray,
        radii: np.ndarray,
    ) -> np.ndarray:
        states = np.asarray(states, dtype=np.int64)
        radii = np.asarray(radii, dtype=np.float64)
        response = np.clip(
            (radii - self.radius_minimum) / (self.radius_maximum - self.radius_minimum),
            0.0,
            1.0,
        )
        direction = np.array([1.0, 0.0, -1.0], dtype=np.float64)
        logits = (
            np.log(self.behavior_probabilities[states])[:, None, :]
            + self.logit_strength * response[None, :, None] * direction[None, None, :]
        )
        return _softmax(logits)

    def probabilities_for_grid(self, states: Tensor, radii: Tensor) -> Tensor:
        state_indices = states.reshape(len(states), -1)[:, 0].to(torch.long)
        behavior = torch.as_tensor(
            self.behavior_probabilities,
            device=states.device,
            dtype=radii.dtype,
        )[state_indices]
        response = (
            (radii - self.radius_minimum)
            / (self.radius_maximum - self.radius_minimum)
        ).clamp(0.0, 1.0)
        direction = radii.new_tensor([1.0, 0.0, -1.0])
        logits = behavior.log()[:, None, :] + (
            self.logit_strength * response[None, :, None] * direction[None, None, :]
        )
        return logits.softmax(dim=2)

    def mean_state_tv(self, radius: float) -> float:
        states = np.arange(len(self.behavior_probabilities))
        target = self.numpy_probabilities(states, np.asarray([radius]))[:, 0]
        return float(0.5 * np.abs(target - self.behavior_probabilities).sum(axis=1).mean())


@dataclass(frozen=True)
class KnownLoggingPolicy:
    behavior_probabilities: np.ndarray

    def probabilities(self, states: Tensor) -> Tensor:
        state_indices = states.reshape(len(states), -1)[:, 0].to(torch.long)
        return torch.as_tensor(
            self.behavior_probabilities,
            device=states.device,
            dtype=states.dtype,
        )[state_indices]


@dataclass(frozen=True)
class ExactMDPOutcomeModel:
    """Expose the frozen tabular predictor through the paper-model interface."""

    predictor_means: Tensor
    predictor_scales: Tensor

    def __call__(self, states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        state_indices = states.reshape(len(states), -1)[:, 0].to(torch.long)
        means = self.predictor_means.to(device=states.device, dtype=states.dtype)
        scales = self.predictor_scales.to(device=states.device, dtype=states.dtype)
        return means[state_indices, actions], scales[state_indices, actions]


def calibration_role_sizes(
    n_calibration: int,
    config: RQ6ConvergenceConfig,
) -> tuple[int, int]:
    """Return the nearest integer 1:2 role split while preserving the total."""

    total_parts = config.cot_role_parts + config.certification_role_parts
    cot_size = (n_calibration * config.cot_role_parts + total_parts // 2) // total_parts
    return cot_size, n_calibration - cot_size


def logged_rng_ids(
    config: RQ6ConvergenceConfig,
    problem_index: int,
    replicate_index: int,
) -> tuple[int, int]:
    cell = problem_index * config.logged_replicates + replicate_index
    return config.logged_rng_start + 2 * cell, config.logged_rng_start + 2 * cell + 1


def build_outcome_blind_m3(
    config: RQ6ConvergenceConfig,
    *,
    problem_seed: int,
) -> tuple[PairedMechanism, OutcomeBlindRadiusPolicy]:
    """Build M3 with a behavior-only radius response of reference TV 0.05."""

    exact_config = config.exact_config(logged_trajectories=1, seed=problem_seed)
    base = next(
        mechanism
        for mechanism in build_paired_mechanisms(
            exact_config,
            problem_seed=problem_seed,
        )
        if mechanism.name == "M3_full_feedback"
    )
    strength = _calibrate_logit_strength(
        base.behavior_probabilities,
        target_tv=config.policy_reference_tv,
    )
    policy = OutcomeBlindRadiusPolicy(
        behavior_probabilities=base.behavior_probabilities,
        radius_minimum=config.radius_minimum,
        radius_maximum=config.radius_maximum,
        logit_strength=strength,
    )
    mechanism = mechanism_on_stage_grids(
        base,
        policy,
        base.problem.radii.detach().cpu().numpy(),
    )
    return mechanism, policy


def mechanism_on_stage_grids(
    mechanism: PairedMechanism,
    policy: OutcomeBlindRadiusPolicy,
    stage_grids: np.ndarray,
) -> PairedMechanism:
    """Clone one exact MDP onto arbitrary stage-specific radius grids."""

    grids = np.asarray(stage_grids, dtype=np.float64)
    horizon, grid_size = grids.shape
    if horizon != mechanism.problem.radii.shape[0]:
        raise ValueError("stage grids must align with the MDP horizon")
    states = np.arange(len(mechanism.behavior_probabilities))
    action_probabilities = np.stack(
        [policy.numpy_probabilities(states, grids[stage]).transpose(1, 0, 2) for stage in range(horizon)]
    )
    expected = (horizon, grid_size, len(states), mechanism.behavior_probabilities.shape[1])
    if action_probabilities.shape != expected:
        raise RuntimeError("outcome-blind policy grid has the wrong shape")
    problem = replace(
        mechanism.problem,
        action_probabilities=torch.from_numpy(action_probabilities),
        radii=torch.from_numpy(grids.copy()),
    )
    return replace(mechanism, problem=problem)


def simulate_nested_role_pools(
    config: RQ6ConvergenceConfig,
    mechanism: PairedMechanism,
    *,
    cot_rng: int,
    certification_rng: int,
    problem_seed: int,
) -> tuple[LoggedTrajectories, LoggedTrajectories]:
    """Simulate one maximum D_COT/D_cert pool reused by every smaller n."""

    maximum_cot, maximum_certification = config.maximum_role_sizes
    cot_config = config.exact_config(
        logged_trajectories=maximum_cot,
        seed=problem_seed,
    )
    certification_config = config.exact_config(
        logged_trajectories=maximum_certification,
        seed=problem_seed,
    )
    cot = simulate_logged_trajectories(
        mechanism,
        generate_logged_randomness(cot_config, seed=cot_rng),
    )
    certification = simulate_logged_trajectories(
        mechanism,
        generate_logged_randomness(certification_config, seed=certification_rng),
    )
    return cot, certification


def evaluate_track_a_nested_prefixes(
    mechanism: PairedMechanism,
    schedules: np.ndarray,
    true_surface: np.ndarray,
    cot_pool: LoggedTrajectories,
    certification_pool: LoggedTrajectories,
    *,
    n_calibration: tuple[int, ...],
    role_sizes: tuple[tuple[int, int], ...],
) -> dict[int, dict[str, Any]]:
    """Evaluate all unique q-prefixes once for every nested sample size.

    At stage t there are K**(t+1) distinct prefixes.  Future suffixes do not
    affect the stage-t event, so their 7/49/343/2401 union is exactly the set
    induced by all 2401 complete schedules.
    """

    problem = mechanism.problem
    horizon, grid_size = problem.radii.shape
    if len(schedules) != grid_size**horizon:
        raise ValueError("Track A requires the complete fixed schedule grid")
    combined = _concatenate_logged(cot_pool, certification_pool)
    maximum_cot = len(cot_pool.states)
    trajectory_count = len(combined.states)
    target_policy = problem.action_probabilities.detach().cpu().numpy()
    radii = problem.radii.detach().cpu().numpy()
    previous_weight: np.ndarray | None = None
    stage_sup_error = np.zeros((len(n_calibration), horizon), dtype=np.float64)
    stage_minimum_ess = np.ones_like(stage_sup_error)

    for stage in range(horizon):
        states = combined.states[:, stage]
        actions = combined.actions[:, stage]
        behavior_probability = mechanism.behavior_probabilities[states, actions]
        target_probability = target_policy[stage, :, states, actions].T
        ratios = target_probability / behavior_probability[None, :]
        hits = combined.scores[:, stage][None, :] <= radii[stage, :, None]

        parent_count = grid_size**stage
        weights = np.empty((parent_count, grid_size, trajectory_count), dtype=np.float64)
        if previous_weight is None:
            weights[0] = ratios
        else:
            np.multiply(
                previous_weight[:, None, :],
                ratios[None, :, :],
                out=weights,
            )
            # The expanded prefix matrix now owns every value needed below.
            # Releasing the shorter predecessor before the candidate scans
            # saves K**stage * N doubles at the final stage.
            previous_weight = None
        if not np.isfinite(weights).all():
            raise RuntimeError("Track A prefix weights are non-finite")

        representative_rows = (
            np.arange(parent_count * grid_size, dtype=np.int64)
            * grid_size ** (horizon - stage - 1)
        )
        exact = true_surface[representative_rows, stage].reshape(parent_count, grid_size)
        for sample_index, (cot_size, certification_size) in enumerate(role_sizes):
            denominator = _role_sum(
                weights.reshape(parent_count * grid_size, trajectory_count),
                cot_size=cot_size,
                certification_size=certification_size,
                maximum_cot=maximum_cot,
            ).reshape(parent_count, grid_size)
            squared_sum = _role_squared_sum(
                weights.reshape(parent_count * grid_size, trajectory_count),
                cot_size=cot_size,
                certification_size=certification_size,
                maximum_cot=maximum_cot,
            ).reshape(parent_count, grid_size)
            estimates = np.empty_like(exact)
            for candidate in range(grid_size):
                weighted_hits = weights[:, candidate] * hits[candidate][None, :]
                numerator = _role_sum(
                    weighted_hits,
                    cot_size=cot_size,
                    certification_size=certification_size,
                    maximum_cot=maximum_cot,
                )
                estimates[:, candidate] = numerator / denominator[:, candidate]
            stage_sup_error[sample_index, stage] = float(np.max(np.abs(estimates - exact)))
            ess = np.square(denominator) / squared_sum
            stage_minimum_ess[sample_index, stage] = float(
                np.min(ess) / n_calibration[sample_index]
            )
        previous_weight = weights.reshape(parent_count * grid_size, trajectory_count)

    return {
        n: {
            "surface_sup_error": float(stage_sup_error[index].max()),
            "stagewise_surface_sup_error": stage_sup_error[index].tolist(),
            "minimum_prefix_ess_fraction": float(stage_minimum_ess[index].min()),
            "stagewise_minimum_prefix_ess_fraction": stage_minimum_ess[index].tolist(),
            "unique_prefix_counts": [grid_size ** (stage + 1) for stage in range(horizon)],
            "complete_schedule_count": int(len(schedules)),
            "supremum_definition": (
                "max over every stage and every unique q-prefix induced by all "
                f"{len(schedules)} complete fixed-grid schedules"
            ),
        }
        for index, n in enumerate(n_calibration)
    }


def evaluate_track_b_canonical_selector(
    config: RQ6ConvergenceConfig,
    mechanism: PairedMechanism,
    policy: OutcomeBlindRadiusPolicy,
    cot_pool: LoggedTrajectories,
    certification_pool: LoggedTrajectories,
    *,
    n_calibration: int,
) -> dict[str, Any]:
    """Run the unmodified paper selector on one empirical D_COT grid."""

    cot_size, certification_size = calibration_role_sizes(n_calibration, config)
    cot = _prefix_logged(cot_pool, cot_size)
    certification = _prefix_logged(certification_pool, certification_size)
    calibration = _concatenate_logged(cot, certification)
    # The paper path uses the framework default floating dtype for its
    # empirical quantile grid and selector.  The exact population recursion
    # below remains float64; this cast mirrors, rather than changes, the
    # canonical selector's deployed numerical path.
    selector_dtype = torch.get_default_dtype()
    cot_scores = torch.from_numpy(cot.scores).to(selector_dtype)
    calibration_scores = torch.from_numpy(calibration.scores).to(selector_dtype)
    stage_grids = torch.stack(
        [
            fixed_q_grid(
                cot_scores[:, stage],
                size=config.grid_size,
                lower_quantile=config.q_quantile_minimum,
                upper_quantile=config.q_quantile_maximum,
            )
            for stage in range(config.horizon)
        ]
    )
    batch = _trajectory_batch(calibration)
    selection = select_marginal_prefix_schedule(
        batch,
        calibration_scores,
        stage_grids=stage_grids,
        target_policy=policy,
        logging_policy=KnownLoggingPolicy(mechanism.behavior_probabilities),
        outcome_model=ExactMDPOutcomeModel(
            mechanism.problem.predictor_means,
            mechanism.problem.predictor_scales,
        ),
        outcome_sd=mechanism.problem.outcome_normalization,
        target=config.target_coverage,
    )
    common: dict[str, Any] = {
        "selection_available": selection.selection_available,
        "failure_stage": selection.failure_stage,
        "stage_grids": stage_grids.tolist(),
        "selected_indices": list(selection.selected_indices),
        "selected_endpoint": selection.selected_endpoint,
        "estimated_coverage": selection.estimated_coverage.tolist(),
        "estimated_normalized_width": selection.estimated_normalized_width.tolist(),
        "selected_ess_fraction": (
            selection.effective_sample_size / n_calibration
        ).tolist(),
    }
    if not selection.selection_available:
        return {
            **common,
            "selected_radii": None,
            "population_coverage": None,
            "population_worst_stage_coverage": None,
            "population_mean_normalized_width": None,
            "selected_policy_reference_state_tv": None,
        }

    empirical_mechanism = mechanism_on_stage_grids(
        mechanism,
        policy,
        stage_grids.detach().cpu().numpy(),
    )
    exact = analytic_schedule_metrics(
        empirical_mechanism.problem,
        selection.selected_indices,
    )
    radii = selection.radii.detach().cpu().numpy()
    return {
        **common,
        "selected_radii": radii.tolist(),
        "population_coverage": exact.coverage.tolist(),
        "population_worst_stage_coverage": float(exact.coverage.min().item()),
        "population_mean_normalized_width": float(exact.normalized_width.mean().item()),
        "selected_policy_reference_state_tv": [
            policy.mean_state_tv(float(radius)) for radius in radii
        ],
    }


def run_problem(
    config: RQ6ConvergenceConfig,
    *,
    problem_index: int,
    problem_seed: int,
) -> dict[str, Any]:
    """Run all 20 nested logged replicates for one fixed MDP instance."""

    mechanism, policy = build_outcome_blind_m3(config, problem_seed=problem_seed)
    exact_config = config.exact_config(logged_trajectories=1, seed=problem_seed)
    schedules = enumerate_schedules(exact_config)
    population, _width = exact_population_surfaces(mechanism, schedules)
    true_surface = population[ESTIMATOR_NAMES.index("full_prefix")]
    role_sizes = tuple(calibration_role_sizes(n, config) for n in config.n_calibration)
    rows = []
    for replicate_index in range(config.logged_replicates):
        cot_rng, certification_rng = logged_rng_ids(config, problem_index, replicate_index)
        cot_pool, certification_pool = simulate_nested_role_pools(
            config,
            mechanism,
            cot_rng=cot_rng,
            certification_rng=certification_rng,
            problem_seed=problem_seed,
        )
        track_a = evaluate_track_a_nested_prefixes(
            mechanism,
            schedules,
            true_surface,
            cot_pool,
            certification_pool,
            n_calibration=config.n_calibration,
            role_sizes=role_sizes,
        )
        for n_calibration in config.n_calibration:
            cot_size, certification_size = calibration_role_sizes(n_calibration, config)
            rows.append(
                {
                    "problem_seed": problem_seed,
                    "problem_index": problem_index,
                    "logged_replicate": replicate_index,
                    "cot_rng": cot_rng,
                    "certification_rng": certification_rng,
                    "n_calibration": n_calibration,
                    "n_cot": cot_size,
                    "n_certification": certification_size,
                    "track_a": track_a[n_calibration],
                    "track_b": evaluate_track_b_canonical_selector(
                        config,
                        mechanism,
                        policy,
                        cot_pool,
                        certification_pool,
                        n_calibration=n_calibration,
                    ),
                }
            )
    return {
        "problem_seed": problem_seed,
        "problem_index": problem_index,
        "mechanism": "M3_full_feedback",
        "policy_contract": {
            "outcome_blind": True,
            "formula": "softmax(log(mu) + lambda * clipped_radius_response * [1,0,-1])",
            "reference_radius": config.reference_radius,
            "target_reference_state_mean_tv": config.policy_reference_tv,
            "observed_reference_state_mean_tv": policy.mean_state_tv(config.reference_radius),
            "logit_strength": policy.logit_strength,
        },
        "rows": rows,
    }


def summarize_problem_results(
    problem_results: list[dict[str, Any]],
    config: RQ6ConvergenceConfig,
) -> dict[str, Any]:
    """Summarize RQ6 with the fixed MDP instance as the bootstrap cluster.

    Logged resamples remain paired repeated measurements inside a problem;
    they are never counted as independent clusters.  One shared matrix of
    problem-level bootstrap indices is used for every ``n`` and both tracks.
    """

    ordered = sorted(problem_results, key=lambda result: result["problem_index"])
    if not ordered:
        raise ValueError("at least one problem result is required")
    problem_indices = [int(result["problem_index"]) for result in ordered]
    if len(set(problem_indices)) != len(problem_indices):
        raise ValueError("problem results contain duplicate problem indices")
    problem_count = len(ordered)
    replicate_count = config.logged_replicates
    n_values = config.n_calibration
    n_count = len(n_values)
    horizon = config.horizon
    n_to_index = {n: index for index, n in enumerate(n_values)}

    track_a_error = np.full(
        (problem_count, replicate_count, n_count),
        np.nan,
        dtype=np.float64,
    )
    track_a_ess = np.full_like(track_a_error, np.nan)
    track_b_available = np.zeros(
        (problem_count, replicate_count, n_count),
        dtype=bool,
    )
    track_b_coverage = np.full(
        (problem_count, replicate_count, n_count, horizon),
        np.nan,
        dtype=np.float64,
    )
    track_b_estimated_coverage = np.full_like(track_b_coverage, np.nan)
    track_b_width = np.full_like(track_a_error, np.nan)
    track_b_minimum_ess = np.full_like(track_a_error, np.nan)
    track_b_endpoint = np.zeros_like(track_b_available)
    track_b_failure_stage = np.full(track_b_available.shape, -1, dtype=np.int64)

    for problem_position, result in enumerate(ordered):
        expected_rows = replicate_count * n_count
        if len(result.get("rows", [])) != expected_rows:
            raise ValueError(
                f"problem {result['problem_index']} has {len(result.get('rows', []))} "
                f"rows; expected {expected_rows}"
            )
        seen_cells: set[tuple[int, int]] = set()
        for row in result["rows"]:
            replicate = int(row["logged_replicate"])
            n_calibration = int(row["n_calibration"])
            if not 0 <= replicate < replicate_count or n_calibration not in n_to_index:
                raise ValueError("problem row lies outside the configured design")
            n_index = n_to_index[n_calibration]
            cell = (replicate, n_index)
            if cell in seen_cells:
                raise ValueError("problem result contains a duplicate replicate/n cell")
            seen_cells.add(cell)
            track_a = row["track_a"]
            track_b = row["track_b"]
            track_a_error[problem_position, replicate, n_index] = float(
                track_a["surface_sup_error"]
            )
            track_a_ess[problem_position, replicate, n_index] = float(
                track_a["minimum_prefix_ess_fraction"]
            )
            available = bool(track_b["selection_available"])
            track_b_available[problem_position, replicate, n_index] = available
            track_b_endpoint[problem_position, replicate, n_index] = bool(
                track_b["selected_endpoint"]
            )
            if track_b.get("failure_stage") is not None:
                track_b_failure_stage[problem_position, replicate, n_index] = int(
                    track_b["failure_stage"]
                )
            if available:
                coverage = np.asarray(track_b["population_coverage"], dtype=np.float64)
                estimated = np.asarray(track_b["estimated_coverage"], dtype=np.float64)
                ess = np.asarray(track_b["selected_ess_fraction"], dtype=np.float64)
                if coverage.shape != (horizon,) or estimated.shape != (horizon,):
                    raise ValueError("available Track-B rows must contain every stage")
                track_b_coverage[problem_position, replicate, n_index] = coverage
                track_b_estimated_coverage[problem_position, replicate, n_index] = estimated
                track_b_width[problem_position, replicate, n_index] = float(
                    track_b["population_mean_normalized_width"]
                )
                track_b_minimum_ess[problem_position, replicate, n_index] = float(
                    ess.min()
                )

    if not np.isfinite(track_a_error).all() or not np.isfinite(track_a_ess).all():
        raise ValueError("Track-A summary inputs must be finite and complete")

    bootstrap_generator = np.random.default_rng(config.bootstrap_rng)
    bootstrap_indices = bootstrap_generator.integers(
        0,
        problem_count,
        size=(config.bootstrap_resamples, problem_count),
    )
    by_n: dict[str, Any] = {}
    for n_index, n_calibration in enumerate(n_values):
        cot_size, certification_size = calibration_role_sizes(n_calibration, config)
        error = track_a_error[:, :, n_index]
        ess = track_a_ess[:, :, n_index]
        error_problem_mean = error.mean(axis=1)
        error_bootstrap = error_problem_mean[bootstrap_indices].mean(axis=1)

        available = track_b_available[:, :, n_index]
        available_count = available.sum(axis=1).astype(np.float64)
        available_bootstrap = (
            available_count[bootstrap_indices].sum(axis=1)
            / (problem_count * replicate_count)
        )
        coverage = track_b_coverage[:, :, n_index]
        coverage_numerator = np.nansum(coverage, axis=1)
        coverage_denominator = np.isfinite(coverage).sum(axis=1).astype(np.float64)
        bootstrap_coverage = _cluster_ratio_draws(
            coverage_numerator,
            coverage_denominator,
            bootstrap_indices,
        )
        bootstrap_wsc = _rowwise_finite_minimum(bootstrap_coverage)
        pooled_stage_coverage = _safe_ratio(
            coverage_numerator.sum(axis=0),
            coverage_denominator.sum(axis=0),
        )

        width = track_b_width[:, :, n_index]
        width_numerator = np.nansum(width, axis=1)
        width_denominator = np.isfinite(width).sum(axis=1).astype(np.float64)
        bootstrap_width = _cluster_ratio_draws(
            width_numerator,
            width_denominator,
            bootstrap_indices,
        )
        pooled_width = _finite_scalar_or_none(
            _safe_ratio(width_numerator.sum(), width_denominator.sum())
        )
        estimated = track_b_estimated_coverage[:, :, n_index]
        estimated_stage_mean = _safe_ratio(
            np.nansum(estimated, axis=(0, 1)),
            np.isfinite(estimated).sum(axis=(0, 1)),
        )
        available_wsc = _rowwise_finite_minimum(
            coverage.reshape(problem_count * replicate_count, horizon)
        ).reshape(problem_count, replicate_count)

        by_n[str(n_calibration)] = {
            "budget": {
                "n_calibration": n_calibration,
                "n_cot": cot_size,
                "n_certification": certification_size,
                "role_ratio": "1:2 with nearest-integer D_COT split",
            },
            "track_a_fixed_population_grid": {
                "mean_surface_sup_error": float(error.mean()),
                "cluster_bootstrap_95_ci": _finite_percentile_interval(error_bootstrap),
                "median_surface_sup_error": float(np.median(error)),
                "p90_surface_sup_error": float(np.quantile(error, 0.90)),
                "p95_surface_sup_error": float(np.quantile(error, 0.95)),
                "mean_minimum_prefix_ess_fraction": float(ess.mean()),
                "within_problem_logged_sd": _within_problem_variability(error),
            },
            "track_b_canonical_empirical_grid": {
                "selection_availability_rate": float(available.mean()),
                "selection_availability_cluster_bootstrap_95_ci": (
                    _finite_percentile_interval(available_bootstrap)
                ),
                "failure_stage_rate_all_attempts": {
                    str(stage): float(
                        np.mean(track_b_failure_stage[:, :, n_index] == stage)
                    )
                    for stage in range(horizon)
                },
                "population_stage_coverage_conditional_on_selection": (
                    _optional_list(pooled_stage_coverage)
                ),
                "population_wsc_conditional_on_selection": _finite_scalar_or_none(
                    _rowwise_finite_minimum(pooled_stage_coverage[None, :])[0]
                ),
                "population_wsc_cluster_bootstrap_95_ci": (
                    _finite_percentile_interval(bootstrap_wsc)
                ),
                "population_mean_normalized_width_conditional_on_selection": (
                    pooled_width
                ),
                "population_width_cluster_bootstrap_95_ci": (
                    _finite_percentile_interval(bootstrap_width)
                ),
                "selected_schedule_target_attainment_rate_conditional_on_selection": (
                    _finite_mean(available_wsc >= config.target_coverage, mask=available)
                ),
                "selection_and_target_attainment_rate": float(
                    np.mean(available & (available_wsc >= config.target_coverage))
                ),
                "estimated_stage_coverage_conditional_on_selection": (
                    _optional_list(estimated_stage_mean)
                ),
                "mean_selected_minimum_ess_fraction_conditional_on_selection": (
                    _finite_mean(track_b_minimum_ess[:, :, n_index])
                ),
                "selected_endpoint_rate_conditional_on_selection": _finite_mean(
                    track_b_endpoint[:, :, n_index],
                    mask=available,
                ),
                "endpoint_reached_rate_all_attempts": float(
                    track_b_endpoint[:, :, n_index].mean()
                ),
                "within_problem_logged_variability": {
                    "availability_sd": _within_problem_variability(
                        available.astype(np.float64)
                    ),
                    "population_wsc_sd_conditional_on_selection": (
                        _within_problem_variability(available_wsc)
                    ),
                    "population_width_sd_conditional_on_selection": (
                        _within_problem_variability(width)
                    ),
                },
            },
        }

    mean_errors = np.asarray(
        [by_n[str(n)]["track_a_fixed_population_grid"]["mean_surface_sup_error"] for n in n_values]
    )
    log_log_slope = (
        None
        if len(n_values) < 2
        else float(
            np.polyfit(
                np.log(np.asarray(n_values, dtype=np.float64)),
                np.log(mean_errors),
                1,
            )[0]
        )
    )
    return {
        "protocol": config.protocol,
        "design": {
            "problem_cluster_count": problem_count,
            "logged_resamples_per_problem": replicate_count,
            "n_calibration": list(n_values),
            "nested_common_random_numbers": True,
            "bootstrap_unit": "fixed MDP problem instance",
            "bootstrap_resamples": config.bootstrap_resamples,
            "bootstrap_rng": config.bootstrap_rng,
            "shared_bootstrap_indices_across_n_and_tracks": True,
        },
        "estimands": {
            "track_a": (
                "For each logged resample, max absolute full-prefix Hajek surface "
                "error over every stage and every unique q-prefix induced by all "
                f"{config.grid_size ** config.horizon} fixed-grid schedules."
            ),
            "track_b": (
                "Selection availability plus exact population per-step marginal "
                "coverage and normalized width of the unmodified canonical selector "
                "on a D_COT-frozen empirical grid; performance values condition on "
                "a schedule being available."
            ),
            "uncertainty": (
                "Percentile bootstrap over fixed MDP problem clusters; the logged "
                "resamples are paired within-problem repetitions, not independent clusters."
            ),
        },
        "claim_boundary": (
            "RQ6 diagnoses finite-sample convergence in the frozen outcome-blind M3 "
            "benchmark. It does not establish finite-sample, distribution-free, PAC, "
            "or data-conditional coverage, and it is not a universal SOTA claim."
        ),
        "track_a_descriptive_log_log_slope": {
            "value": log_log_slope,
            "status": "descriptive_not_a_claimed_rate",
            "reason": (
                "six fixed finite-sample design points are insufficient to claim "
                "an asymptotic convergence exponent"
            ),
        },
        "by_n_calibration": by_n,
    }


def _cluster_ratio_draws(
    numerator_by_problem: np.ndarray,
    denominator_by_problem: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> np.ndarray:
    numerator = numerator_by_problem[bootstrap_indices].sum(axis=1)
    denominator = denominator_by_problem[bootstrap_indices].sum(axis=1)
    return _safe_ratio(numerator, denominator)


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    output_shape = np.broadcast_shapes(numerator_array.shape, denominator_array.shape)
    output = np.full(output_shape, np.nan, dtype=np.float64)
    return np.divide(numerator_array, denominator_array, out=output, where=denominator_array > 0)


def _finite_percentile_interval(values: np.ndarray) -> list[float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [None, None]
    lower, upper = np.quantile(finite, (0.025, 0.975))
    return [float(lower), float(upper)]


def _finite_scalar_or_none(value: np.ndarray | float) -> float | None:
    scalar = float(np.asarray(value, dtype=np.float64))
    return scalar if np.isfinite(scalar) else None


def _optional_list(values: np.ndarray) -> list[float | None]:
    return [_finite_scalar_or_none(value) for value in np.asarray(values)]


def _rowwise_finite_minimum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("rowwise minimum requires a matrix")
    replaced = np.where(np.isfinite(values), values, np.inf)
    minimum = replaced.min(axis=1)
    minimum[~np.isfinite(minimum)] = np.nan
    return minimum


def _within_problem_variability(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    standard_deviations = []
    for row in values:
        finite = row[np.isfinite(row)]
        if len(finite) >= 2:
            standard_deviations.append(float(np.std(finite, ddof=1)))
    if not standard_deviations:
        return {"mean_sd": None, "median_sd": None}
    return {
        "mean_sd": float(np.mean(standard_deviations)),
        "median_sd": float(np.median(standard_deviations)),
    }


def _finite_mean(values: np.ndarray, *, mask: np.ndarray | None = None) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not finite.any():
        return None
    return float(values[finite].mean())


def _calibrate_logit_strength(
    behavior_probabilities: np.ndarray,
    *,
    target_tv: float,
) -> float:
    direction = np.array([1.0, 0.0, -1.0], dtype=np.float64)

    def tv(strength: float) -> float:
        target = _softmax(
            np.log(behavior_probabilities) + 0.5 * strength * direction[None, :]
        )
        return float(0.5 * np.abs(target - behavior_probabilities).sum(axis=1).mean())

    lower, upper = 0.0, 1.0
    while tv(upper) < target_tv:
        upper *= 2.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if tv(middle) < target_tv:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _prefix_logged(logged: LoggedTrajectories, count: int) -> LoggedTrajectories:
    return LoggedTrajectories(
        states=logged.states[:count],
        actions=logged.actions[:count],
        scores=logged.scores[:count],
    )


def _concatenate_logged(
    first: LoggedTrajectories,
    second: LoggedTrajectories,
) -> LoggedTrajectories:
    return LoggedTrajectories(
        states=np.concatenate((first.states, second.states), axis=0),
        actions=np.concatenate((first.actions, second.actions), axis=0),
        scores=np.concatenate((first.scores, second.scores), axis=0),
    )


def _trajectory_batch(logged: LoggedTrajectories) -> TrajectoryBatch:
    n, horizon = logged.states.shape
    states = torch.zeros((n, horizon + 1, 1), dtype=torch.float64)
    states[:, :horizon, 0] = torch.from_numpy(logged.states)
    return TrajectoryBatch(
        states=states,
        actions=torch.from_numpy(logged.actions).to(torch.long),
        outcomes=torch.zeros((n, horizon, 1), dtype=torch.float64),
        patient_ids=torch.arange(n),
    )


def _role_sum(
    values: np.ndarray,
    *,
    cot_size: int,
    certification_size: int,
    maximum_cot: int,
) -> np.ndarray:
    return values[:, :cot_size].sum(axis=1) + values[
        :, maximum_cot : maximum_cot + certification_size
    ].sum(axis=1)


def _role_squared_sum(
    values: np.ndarray,
    *,
    cot_size: int,
    certification_size: int,
    maximum_cot: int,
) -> np.ndarray:
    """Sum squares over both nested roles without allocating ``values**2``."""

    cot = values[:, :cot_size]
    certification = values[
        :, maximum_cot : maximum_cot + certification_size
    ]
    return np.einsum("ij,ij->i", cot, cot) + np.einsum(
        "ij,ij->i",
        certification,
        certification,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=-1, keepdims=True)
    exponential = np.exp(centered)
    return exponential / exponential.sum(axis=-1, keepdims=True)
