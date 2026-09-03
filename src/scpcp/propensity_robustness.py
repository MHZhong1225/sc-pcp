"""Isolated finite-MDP robustness study for fitted logging propensities.

The primary layer fixes the target policy at the oracle logging-policy anchor
and changes only the transport denominator.  The appendix layer deliberately
changes both the anchor and denominator and is reported separately.  Neither
layer changes the canonical SC-PCP implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import warnings
from typing import Any

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from scpcp.exact_finite_mdp import (
    ExactFiniteMDPConfig,
    LoggedTrajectories,
    PairedMechanism,
    build_paired_mechanisms,
    generate_logged_randomness,
    simulate_logged_trajectories,
)
from scpcp.phase0_search import AnalyticFiniteMDP, analytic_schedule_metrics


PROTOCOL = "propensity_robustness_v1"
SEED_NAMESPACE = "propensity_robustness_v1:98000..98300"
PRIMARY_LAYER = "primary_transport_only"
APPENDIX_LAYER = "appendix_end_to_end"
PROPENSITY_ARMS = (
    "oracle",
    "correct_multinomial",
    "misspecified_reduced_state",
)

EXTERNAL_SEED_RESERVATIONS = {
    "exact_finite_mdp": range(52_000, 53_000),
    "controlled_six_method": range(91_000, 92_000),
    "orthogonal_copula": range(94_000, 95_000),
    "rq5_horizon_overlap": range(96_000, 97_000),
    "rq6_calibration_convergence": range(97_000, 98_000),
    "strict_split_audit": range(99_000, 100_000),
    "score_robustness": range(100_000, 101_000),
}


@dataclass(frozen=True)
class PropensityRobustnessConfig:
    """Frozen protocol for the paired M3 propensity experiment."""

    protocol: str = PROTOCOL
    state_count: int = 8
    action_count: int = 3
    horizon: int = 8
    grid_size: int = 7
    alpha: float = 0.10
    instances: int = 100
    nuisance_trajectories: int = 5_000
    calibration_trajectories: int = 5_000
    problem_seed_start: int = 98_000
    nuisance_seed_start: int = 98_100
    calibration_seed_start: int = 98_200
    bootstrap_seed: int = 98_300
    bootstrap_resamples: int = 10_000
    seed_namespace: str = SEED_NAMESPACE
    radius_minimum: float = 1.4
    radius_maximum: float = 3.5
    policy_response_center: float = 2.5
    policy_response_scale: float = 0.7
    policy_response_strength: float = 0.24
    behavior_state_strength: float = 0.8
    reduced_state_cutpoint: int = 4
    logistic_solver: str = "lbfgs"
    logistic_penalty: str = "l2"
    logistic_inverse_regularization: float = 1_000_000.0
    logistic_max_iterations: int = 1_000
    logistic_tolerance: float = 1e-10

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    @property
    def problem_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.problem_seed_start, self.problem_seed_start + self.instances))

    @property
    def nuisance_seeds(self) -> tuple[int, ...]:
        return tuple(
            range(self.nuisance_seed_start, self.nuisance_seed_start + self.instances)
        )

    @property
    def calibration_seeds(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.calibration_seed_start,
                self.calibration_seed_start + self.instances,
            )
        )

    def validate(self) -> None:
        if self.protocol != PROTOCOL:
            raise ValueError("unknown propensity-robustness protocol")
        if (self.state_count, self.action_count) != (8, 3):
            raise ValueError("the frozen M3 study requires S=8 and A=3")
        if self.horizon < 1 or self.grid_size < 2:
            raise ValueError("horizon and grid size must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.instances < 1:
            raise ValueError("instances must be positive")
        if min(self.nuisance_trajectories, self.calibration_trajectories) < 2:
            raise ValueError("nuisance and calibration samples must contain two trajectories")
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")
        if not self.radius_minimum < self.radius_maximum:
            raise ValueError("radius bounds are invalid")
        if self.policy_response_scale <= 0.0 or self.policy_response_strength < 0.0:
            raise ValueError("policy response settings are invalid")
        if self.behavior_state_strength < 0.0:
            raise ValueError("behavior_state_strength must be nonnegative")
        if not 1 <= self.reduced_state_cutpoint < self.state_count:
            raise ValueError("reduced_state_cutpoint must split the state space")
        if (
            self.logistic_solver != "lbfgs"
            or self.logistic_penalty != "l2"
            or self.logistic_inverse_regularization <= 0.0
            or self.logistic_max_iterations < 1
            or self.logistic_tolerance <= 0.0
        ):
            raise ValueError("the multinomial fitting contract is invalid")
        audit = propensity_seed_collision_audit(self)
        if audit["collision"]:
            raise ValueError(
                "propensity robustness RNG IDs collide: "
                f"{audit['within_study_duplicates'] or audit['external_collisions']}"
            )

    def assert_frozen_protocol(self) -> None:
        # Every dataclass field affects either the estimand, nuisance fit,
        # Monte-Carlo population, or uncertainty calculation. There are no
        # scientific command-line overrides for a formal run; artifact output
        # and resume behavior live exclusively in the runner.
        expected = asdict(PropensityRobustnessConfig())
        observed = asdict(self)
        if observed != expected:
            differences = {
                name: {"expected": expected[name], "observed": observed[name]}
                for name in expected
                if observed[name] != expected[name]
            }
            raise ValueError(
                "propensity robustness config differs from the frozen protocol: "
                f"{differences}"
            )

    def exact_config(self, *, logged_trajectories: int) -> ExactFiniteMDPConfig:
        config = ExactFiniteMDPConfig(
            state_count=self.state_count,
            action_count=self.action_count,
            horizon=self.horizon,
            grid_size=self.grid_size,
            alpha=self.alpha,
            logged_trajectories=logged_trajectories,
            seed=52_081,
            population_instances=1,
            population_seed_start=52_100,
            logged_instance_count=0,
            logged_replicates=1,
            logged_replicate_seed_start=52_600,
            radius_minimum=self.radius_minimum,
            radius_maximum=self.radius_maximum,
            policy_response_center=self.policy_response_center,
            policy_response_scale=self.policy_response_scale,
            policy_response_strength=self.policy_response_strength,
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PropensityRobustnessResult:
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]
    nuisance_records: list[dict[str, Any]]
    primary_records: list[dict[str, Any]]
    appendix_records: list[dict[str, Any]]


@dataclass(frozen=True)
class _PrefixSelection:
    available: bool
    selected_indices: np.ndarray
    estimated_coverage: np.ndarray
    estimated_width: np.ndarray
    ess_fraction: np.ndarray
    log_weight_span: np.ndarray
    failure_stage: int | None


def propensity_seed_collision_audit(
    config: PropensityRobustnessConfig,
) -> dict[str, Any]:
    """Enumerate every actual RNG ID and compare coordinated reservations."""

    streams = {
        "problem": list(config.problem_seeds),
        "nuisance_logging": list(config.nuisance_seeds),
        "calibration_logging": list(config.calibration_seeds),
        "paired_summary_bootstrap": [config.bootstrap_seed],
    }
    owners: dict[int, list[str]] = {}
    for stream, identifiers in streams.items():
        for identifier in identifiers:
            owners.setdefault(identifier, []).append(stream)
    duplicates = {
        str(identifier): names
        for identifier, names in sorted(owners.items())
        if len(names) > 1
    }
    active = set(owners)
    external = {
        name: sorted(active.intersection(reservation))
        for name, reservation in EXTERNAL_SEED_RESERVATIONS.items()
    }
    external = {name: values for name, values in external.items() if values}
    return {
        "seed_namespace": config.seed_namespace,
        "streams": streams,
        "all_rng_ids": sorted(active),
        "rng_id_count": len(active),
        "within_study_duplicates": duplicates,
        "external_reservations": {
            "exact_finite_mdp": "52000..52999",
            "controlled_six_method": "91000..91999",
            "orthogonal_copula": "94000..94999",
            "rq5_horizon_overlap": "96000..96999",
            "rq6_calibration_convergence": "97000..97999",
            "strict_split_audit": "99000..99999",
            "score_robustness": "100000..100999",
        },
        "external_collisions": external,
        "collision": bool(duplicates or external),
    }


def smoke_config() -> PropensityRobustnessConfig:
    """Return a tiny non-scientific contract configuration."""

    return PropensityRobustnessConfig(
        instances=2,
        nuisance_trajectories=96,
        calibration_trajectories=128,
        problem_seed_start=701,
        nuisance_seed_start=711,
        calibration_seed_start=721,
        bootstrap_seed=731,
        bootstrap_resamples=64,
        seed_namespace="custom_smoke_non_scientific",
        logistic_max_iterations=300,
        logistic_tolerance=1e-8,
    )


def run_propensity_robustness(
    config: PropensityRobustnessConfig,
) -> PropensityRobustnessResult:
    """Run paired primary and appendix layers over the configured M3 instances."""

    config.validate()
    instance_count = config.instances
    arm_count = len(PROPENSITY_ARMS)
    horizon = config.horizon
    nuisance_shape = (instance_count, arm_count)
    layer_shape = (instance_count, arm_count, horizon)

    nuisance_arrays = {
        "mae": np.empty(nuisance_shape),
        "log_loss": np.empty(nuisance_shape),
        "excess_log_loss": np.empty(nuisance_shape),
        "mean_absolute_relative_error": np.empty(nuisance_shape),
        "maximum_absolute_relative_error": np.empty(nuisance_shape),
        "minimum_probability": np.empty(nuisance_shape),
        "iterations": np.empty(nuisance_shape, dtype=np.int32),
    }
    layers = {
        PRIMARY_LAYER: _empty_layer_arrays(layer_shape),
        APPENDIX_LAYER: _empty_layer_arrays(layer_shape),
    }
    target_policy_tv = np.empty(instance_count)
    primary_fingerprints = np.empty((instance_count, arm_count), dtype="U64")
    appendix_fingerprints = np.empty((instance_count, arm_count), dtype="U64")
    appendix_target_drift = np.empty(nuisance_shape)
    primary_selected_tv_from_oracle_behavior = np.full(layer_shape, np.nan)
    appendix_selected_tv_from_own_anchor = np.full(layer_shape, np.nan)
    appendix_selected_tv_from_oracle_target_matched_radii = np.full(
        layer_shape,
        np.nan,
    )
    appendix_deployed_tv_from_primary_oracle_deployment = np.full(
        layer_shape,
        np.nan,
    )
    nuisance_records: list[dict[str, Any]] = []
    primary_records: list[dict[str, Any]] = []
    appendix_records: list[dict[str, Any]] = []

    for instance, (problem_seed, nuisance_seed, calibration_seed) in enumerate(
        zip(
            config.problem_seeds,
            config.nuisance_seeds,
            config.calibration_seeds,
            strict=True,
        )
    ):
        mechanism = _build_m3_mechanism(config, problem_seed=problem_seed)
        nuisance_logged = _simulate_logged(
            mechanism,
            config,
            trajectories=config.nuisance_trajectories,
            seed=nuisance_seed,
        )
        calibration_logged = _simulate_logged(
            mechanism,
            config,
            trajectories=config.calibration_trajectories,
            seed=calibration_seed,
        )
        oracle_behavior = mechanism.behavior_probabilities
        state_occupancy = _logging_state_occupancy(mechanism)
        propensity_estimates, fit_iterations = _fit_propensity_arms(
            nuisance_logged,
            oracle_behavior,
            config,
        )
        oracle_target = np.asarray(mechanism.problem.action_probabilities)
        target_policy_tv[instance] = _mean_policy_tv(
            oracle_target,
            oracle_behavior,
            state_occupancy,
        )
        primary_oracle_deployment: np.ndarray | None = None

        for arm_index, arm in enumerate(PROPENSITY_ARMS):
            estimate = propensity_estimates[arm]
            diagnostics = _nuisance_diagnostics(
                oracle_behavior,
                estimate,
                state_occupancy,
            )
            for name in nuisance_arrays:
                nuisance_arrays[name][instance, arm_index] = (
                    fit_iterations[arm] if name == "iterations" else diagnostics[name]
                )
            nuisance_records.append(
                {
                    "problem_seed": problem_seed,
                    "nuisance_seed": nuisance_seed,
                    "calibration_seed": calibration_seed,
                    "arm": arm,
                    **diagnostics,
                    "iterations": fit_iterations[arm],
                    "converged": True,
                }
            )

            primary_problem = mechanism.problem
            primary_fingerprint = _target_law_fingerprint(primary_problem)
            primary_fingerprints[instance, arm_index] = primary_fingerprint
            primary_selection = _select_prefix_schedule(
                primary_problem,
                calibration_logged,
                denominator=estimate,
                target=config.target_coverage,
            )
            _store_layer_result(
                layers[PRIMARY_LAYER],
                instance,
                arm_index,
                primary_problem,
                primary_selection,
            )
            primary_deployment = _selected_policy(
                oracle_target,
                primary_selection,
            )
            if primary_deployment is not None:
                primary_selected_tv_from_oracle_behavior[instance, arm_index] = (
                    _policy_tv_by_stage(
                        primary_deployment,
                        oracle_behavior,
                        state_occupancy,
                    )
                )
            if arm == "oracle":
                primary_oracle_deployment = primary_deployment
            primary_records.append(
                _layer_record(
                    layer=PRIMARY_LAYER,
                    arm=arm,
                    problem_seed=problem_seed,
                    nuisance_seed=nuisance_seed,
                    calibration_seed=calibration_seed,
                    target_law_fingerprint=primary_fingerprint,
                    target_policy_drift_from_oracle=0.0,
                    selection=primary_selection,
                    arrays=layers[PRIMARY_LAYER],
                    instance=instance,
                    arm_index=arm_index,
                    selected_policy_tv_by_stage=(
                        primary_selected_tv_from_oracle_behavior[
                            instance, arm_index
                        ]
                    ),
                    selected_policy_tv_definition=(
                        "selected/deployed primary policy versus oracle logging "
                        "policy mu, stagewise TV averaged over oracle-mu state occupancy"
                    ),
                )
            )

            appendix_target = _target_policy_from_anchor(
                estimate,
                np.asarray(mechanism.problem.radii),
                config,
            )
            appendix_problem = replace(
                mechanism.problem,
                action_probabilities=torch.from_numpy(appendix_target.copy()),
            )
            appendix_fingerprint = _target_law_fingerprint(appendix_problem)
            appendix_fingerprints[instance, arm_index] = appendix_fingerprint
            appendix_target_drift[instance, arm_index] = _mean_target_policy_drift(
                appendix_target,
                oracle_target,
                state_occupancy,
            )
            appendix_selection = _select_prefix_schedule(
                appendix_problem,
                calibration_logged,
                denominator=estimate,
                target=config.target_coverage,
            )
            _store_layer_result(
                layers[APPENDIX_LAYER],
                instance,
                arm_index,
                appendix_problem,
                appendix_selection,
            )
            appendix_deployment = _selected_policy(
                appendix_target,
                appendix_selection,
            )
            if appendix_deployment is not None:
                appendix_selected_tv_from_own_anchor[instance, arm_index] = (
                    _policy_tv_by_stage(
                        appendix_deployment,
                        estimate,
                        state_occupancy,
                    )
                )
                oracle_at_matched_radii = _selected_policy(
                    oracle_target,
                    appendix_selection,
                )
                if oracle_at_matched_radii is None:  # pragma: no cover - same selection
                    raise RuntimeError("matched-radius oracle target is unavailable")
                appendix_selected_tv_from_oracle_target_matched_radii[
                    instance, arm_index
                ] = _policy_tv_by_stage(
                    appendix_deployment,
                    oracle_at_matched_radii,
                    state_occupancy,
                )
                if primary_oracle_deployment is not None:
                    appendix_deployed_tv_from_primary_oracle_deployment[
                        instance, arm_index
                    ] = _policy_tv_by_stage(
                        appendix_deployment,
                        primary_oracle_deployment,
                        state_occupancy,
                    )
            appendix_records.append(
                _layer_record(
                    layer=APPENDIX_LAYER,
                    arm=arm,
                    problem_seed=problem_seed,
                    nuisance_seed=nuisance_seed,
                    calibration_seed=calibration_seed,
                    target_law_fingerprint=appendix_fingerprint,
                    target_policy_drift_from_oracle=appendix_target_drift[
                        instance, arm_index
                    ],
                    selection=appendix_selection,
                    arrays=layers[APPENDIX_LAYER],
                    instance=instance,
                    arm_index=arm_index,
                    selected_policy_tv_by_stage=(
                        appendix_selected_tv_from_own_anchor[instance, arm_index]
                    ),
                    selected_policy_tv_definition=(
                        "selected/deployed appendix policy versus that arm's fitted "
                        "anchor, stagewise TV averaged over oracle-mu state occupancy"
                    ),
                    matched_oracle_target_tv_by_stage=(
                        appendix_selected_tv_from_oracle_target_matched_radii[
                            instance, arm_index
                        ]
                    ),
                    primary_oracle_deployment_tv_by_stage=(
                        appendix_deployed_tv_from_primary_oracle_deployment[
                            instance, arm_index
                        ]
                    ),
                )
            )

        if len(set(primary_fingerprints[instance])) != 1:
            raise RuntimeError("primary target-law fingerprint differs across propensity arms")

    bootstrap_indices = np.random.default_rng(config.bootstrap_seed).integers(
        0,
        instance_count,
        size=(config.bootstrap_resamples, instance_count),
        dtype=np.int32,
    )
    summary = {
        "schema_version": 1,
        "study": PROTOCOL,
        "status": "complete",
        "diagnostic_only": True,
        "canonical_method_unchanged": True,
        "formal_scientific_run": config.seed_namespace == SEED_NAMESPACE,
        "target_coverage": config.target_coverage,
        "propensity_arms": list(PROPENSITY_ARMS),
        "seed_namespace": config.seed_namespace,
        "seed_collision_audit": propensity_seed_collision_audit(config),
        "nuisance_fit_contract": {
            "correct_features": "full_state_one_hot",
            "misspecified_features": (
                f"two_bin_state_indicator_cut_at_{config.reduced_state_cutpoint}"
            ),
            "solver": config.logistic_solver,
            "penalty": config.logistic_penalty,
            "inverse_regularization": config.logistic_inverse_regularization,
            "maximum_iterations": config.logistic_max_iterations,
            "convergence_tolerance": config.logistic_tolerance,
            "nonconvergence_policy": "fail_closed",
        },
        "moderate_policy_tv": _metric_summary(target_policy_tv),
        "nuisance_diagnostics": _nuisance_summary(
            nuisance_arrays,
            bootstrap_indices,
        ),
        PRIMARY_LAYER: {
            "description": (
                "oracle target-policy anchor shared exactly; only the transport "
                "denominator changes"
            ),
            "target_law_fingerprint_shared_across_arms": True,
            "selected_deployed_policy_tv": _policy_tv_summary(
                primary_selected_tv_from_oracle_behavior,
                layers[PRIMARY_LAYER]["selected"],
                bootstrap_indices,
                reference=(
                    "oracle logging policy mu; stagewise state TV is averaged under "
                    "the oracle-mu state occupancy, then stages and paired instances "
                    "are averaged"
                ),
            ),
            "results": _layer_summary(layers[PRIMARY_LAYER], bootstrap_indices),
        },
        APPENDIX_LAYER: {
            "description": (
                "fitted propensity changes both target-policy anchor and transport "
                "denominator; never pooled with the primary layer"
            ),
            "target_policy_drift_from_oracle": _arm_metric_summary(
                appendix_target_drift,
                bootstrap_indices,
            ),
            "target_policy_drift_from_oracle_definition": (
                "mean TV between the full fitted-anchor and oracle-anchor candidate "
                "policy surfaces, averaged over stages, all candidate radii, and "
                "oracle-mu state occupancy; this is not the selected-policy TV"
            ),
            "selected_deployed_policy_tv": {
                "from_own_anchor": _policy_tv_summary(
                    appendix_selected_tv_from_own_anchor,
                    layers[APPENDIX_LAYER]["selected"],
                    bootstrap_indices,
                    reference=(
                        "each arm's own oracle/fitted propensity anchor; stagewise "
                        "state TV is averaged under oracle-mu state occupancy"
                    ),
                ),
                "from_oracle_target_at_matched_selected_radii": _policy_tv_summary(
                    appendix_selected_tv_from_oracle_target_matched_radii,
                    layers[APPENDIX_LAYER]["selected"],
                    bootstrap_indices,
                    reference=(
                        "oracle-mu-anchored target evaluated at the same selected "
                        "radius at each stage; isolates anchor drift from radius choice"
                    ),
                ),
                "from_primary_oracle_deployment": _policy_tv_summary(
                    appendix_deployed_tv_from_primary_oracle_deployment,
                    layers[APPENDIX_LAYER]["selected"],
                    bootstrap_indices,
                    reference=(
                        "primary oracle arm's selected/deployed policy; includes both "
                        "anchor-induced target drift and any change in selected radii"
                    ),
                ),
            },
            "results": _layer_summary(layers[APPENDIX_LAYER], bootstrap_indices),
        },
    }
    arrays: dict[str, np.ndarray] = {
        "problem_seeds": np.asarray(config.problem_seeds, dtype=np.int64),
        "nuisance_seeds": np.asarray(config.nuisance_seeds, dtype=np.int64),
        "calibration_seeds": np.asarray(config.calibration_seeds, dtype=np.int64),
        "bootstrap_indices": bootstrap_indices,
        "propensity_arms": np.asarray(PROPENSITY_ARMS),
        "target_policy_tv": target_policy_tv,
        "primary_target_law_fingerprints": primary_fingerprints,
        "appendix_target_law_fingerprints": appendix_fingerprints,
        "appendix_target_policy_drift": appendix_target_drift,
        "primary_selected_policy_tv_from_oracle_behavior": (
            primary_selected_tv_from_oracle_behavior
        ),
        "appendix_selected_policy_tv_from_own_anchor": (
            appendix_selected_tv_from_own_anchor
        ),
        "appendix_selected_policy_tv_from_oracle_target_matched_radii": (
            appendix_selected_tv_from_oracle_target_matched_radii
        ),
        "appendix_deployed_policy_tv_from_primary_oracle_deployment": (
            appendix_deployed_tv_from_primary_oracle_deployment
        ),
    }
    arrays.update({f"nuisance_{name}": value for name, value in nuisance_arrays.items()})
    for layer_name, layer_arrays in layers.items():
        arrays.update({f"{layer_name}_{name}": value for name, value in layer_arrays.items()})
    return PropensityRobustnessResult(
        summary=summary,
        arrays=arrays,
        nuisance_records=nuisance_records,
        primary_records=primary_records,
        appendix_records=appendix_records,
    )


def _empty_layer_arrays(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    instance_count, arm_count, horizon = shape
    return {
        "selected": np.zeros((instance_count, arm_count), dtype=bool),
        "selected_indices": np.full(shape, -1, dtype=np.int16),
        "exact_coverage": np.full(shape, np.nan),
        "exact_normalized_width": np.full(shape, np.nan),
        "estimated_coverage": np.full(shape, np.nan),
        "estimated_normalized_width": np.full(shape, np.nan),
        "ess_fraction": np.full(shape, np.nan),
        "log_weight_span": np.full(shape, np.nan),
        "failure_stage": np.full((instance_count, arm_count), -1, dtype=np.int16),
    }


def _build_m3_mechanism(
    config: PropensityRobustnessConfig,
    *,
    problem_seed: int,
) -> PairedMechanism:
    exact_config = config.exact_config(logged_trajectories=config.calibration_trajectories)
    base = build_paired_mechanisms(exact_config, problem_seed=problem_seed)[3]
    behavior = _state_dependent_behavior(config)
    target = _target_policy_from_anchor(
        behavior,
        np.asarray(base.problem.radii),
        config,
    )
    return PairedMechanism(
        name="M3_full_feedback_propensity_robustness",
        description=(
            "M3 with state-dependent logging propensity and moderate "
            "prediction-mediated policy shift"
        ),
        behavior_probabilities=behavior,
        problem=replace(
            base.problem,
            action_probabilities=torch.from_numpy(target.copy()),
        ),
    )


def _state_dependent_behavior(config: PropensityRobustnessConfig) -> np.ndarray:
    normalized_state = np.linspace(-1.0, 1.0, config.state_count)
    base_logits = np.array([-0.3, 0.6, -0.3], dtype=np.float64)
    direction = np.array([1.0, 0.0, -1.0], dtype=np.float64)
    logits = (
        base_logits[None, :]
        + config.behavior_state_strength
        * normalized_state[:, None]
        * direction[None, :]
    )
    return _softmax(logits)


def _target_policy_from_anchor(
    anchor: np.ndarray,
    radii: np.ndarray,
    config: PropensityRobustnessConfig,
) -> np.ndarray:
    direction = np.array([1.0, 0.0, -1.0], dtype=np.float64)
    response = config.policy_response_strength * (
        (radii - config.policy_response_center) / config.policy_response_scale
    )
    logits = (
        np.log(anchor)[None, None, :, :]
        + response[:, :, None, None] * direction[None, None, None, :]
    )
    return _softmax(logits)


def _simulate_logged(
    mechanism: PairedMechanism,
    config: PropensityRobustnessConfig,
    *,
    trajectories: int,
    seed: int,
) -> LoggedTrajectories:
    exact_config = config.exact_config(logged_trajectories=trajectories)
    randomness = generate_logged_randomness(exact_config, seed=seed)
    return simulate_logged_trajectories(mechanism, randomness)


def _fit_propensity_arms(
    logged: LoggedTrajectories,
    oracle: np.ndarray,
    config: PropensityRobustnessConfig,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    states = logged.states.reshape(-1).astype(np.int64)
    actions = logged.actions.reshape(-1).astype(np.int64)
    estimates = {"oracle": oracle.copy()}
    iterations = {"oracle": 0}
    for arm, reduced in (
        ("correct_multinomial", False),
        ("misspecified_reduced_state", True),
    ):
        estimate, iteration_count = _fit_multinomial(
            states,
            actions,
            reduced=reduced,
            config=config,
        )
        estimates[arm] = estimate
        iterations[arm] = iteration_count
    return estimates, iterations


def _fit_multinomial(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    reduced: bool,
    config: PropensityRobustnessConfig,
) -> tuple[np.ndarray, int]:
    category_count = 2 if reduced else config.state_count
    categories = (
        (states >= config.reduced_state_cutpoint).astype(np.int64)
        if reduced
        else states
    )
    features = np.eye(category_count, dtype=np.float64)[categories]
    model = LogisticRegression(
        solver=config.logistic_solver,
        penalty=config.logistic_penalty,
        C=config.logistic_inverse_regularization,
        fit_intercept=False,
        max_iter=config.logistic_max_iterations,
        tol=config.logistic_tolerance,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(features, actions)
        except ConvergenceWarning as error:
            raise RuntimeError("multinomial propensity fit did not converge") from error
    iteration_count = int(np.max(model.n_iter_))
    if iteration_count >= config.logistic_max_iterations:
        raise RuntimeError("multinomial propensity fit reached the iteration limit")
    if model.classes_.tolist() != list(range(config.action_count)):
        raise RuntimeError("multinomial propensity fit did not observe every action")
    state_categories = (
        (np.arange(config.state_count) >= config.reduced_state_cutpoint).astype(np.int64)
        if reduced
        else np.arange(config.state_count)
    )
    state_features = np.eye(category_count, dtype=np.float64)[state_categories]
    probabilities = model.predict_proba(state_features)
    if (
        probabilities.shape != (config.state_count, config.action_count)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities <= 0.0)
    ):
        raise RuntimeError("multinomial propensity fit returned invalid probabilities")
    return probabilities, iteration_count


def _logging_state_occupancy(mechanism: PairedMechanism) -> np.ndarray:
    problem = mechanism.problem
    initial = np.asarray(problem.initial_state_probabilities)
    transition = np.asarray(problem.transition_probabilities)
    behavior_transition = np.einsum(
        "sa,asr->sr",
        mechanism.behavior_probabilities,
        transition,
    )
    occupancies = []
    current = initial.copy()
    for _ in range(problem.radii.shape[0]):
        occupancies.append(current)
        current = current @ behavior_transition
    return np.stack(occupancies)


def _nuisance_diagnostics(
    truth: np.ndarray,
    estimate: np.ndarray,
    state_occupancy: np.ndarray,
) -> dict[str, float]:
    state_weight = state_occupancy.mean(axis=0)
    absolute_error = np.abs(estimate - truth)
    relative_error = absolute_error / truth
    log_loss = -np.sum(
        state_weight[:, None] * truth * np.log(estimate)
    )
    oracle_log_loss = -np.sum(state_weight[:, None] * truth * np.log(truth))
    return {
        "mae": float(np.sum(state_weight * absolute_error.mean(axis=1))),
        "log_loss": float(log_loss),
        "excess_log_loss": float(log_loss - oracle_log_loss),
        "mean_absolute_relative_error": float(
            np.sum(state_weight * relative_error.mean(axis=1))
        ),
        "maximum_absolute_relative_error": float(relative_error.max()),
        "minimum_probability": float(estimate.min()),
    }


def _select_prefix_schedule(
    problem: AnalyticFiniteMDP,
    logged: LoggedTrajectories,
    *,
    denominator: np.ndarray,
    target: float,
) -> _PrefixSelection:
    target_policy = np.asarray(problem.action_probabilities)
    radii = np.asarray(problem.radii)
    predictor_scales = np.asarray(problem.predictor_scales)
    outcome_normalization = np.asarray(problem.outcome_normalization)
    base_width = 2.0 * np.mean(
        predictor_scales / outcome_normalization[None, None, :],
        axis=2,
    )
    trajectory_count, horizon = logged.states.shape
    raw_log_prefix = np.zeros(trajectory_count, dtype=np.float64)
    selected_indices = np.full(horizon, -1, dtype=np.int16)
    estimated_coverage = np.full(horizon, np.nan)
    estimated_width = np.full(horizon, np.nan)
    ess_fraction = np.full(horizon, np.nan)
    log_weight_span = np.full(horizon, np.nan)

    for stage in range(horizon):
        states = logged.states[:, stage]
        actions = logged.actions[:, stage]
        numerator = target_policy[stage, :, states, actions].T
        observed_denominator = denominator[states, actions]
        if np.any(observed_denominator <= 0.0):
            raise RuntimeError("transport denominator is not strictly positive")
        candidate_log_weight = (
            raw_log_prefix[None, :]
            + np.log(numerator)
            - np.log(observed_denominator)[None, :]
        )
        maximum = candidate_log_weight.max(axis=1)
        minimum = candidate_log_weight.min(axis=1)
        weights = np.exp(candidate_log_weight - maximum[:, None])
        weight_sum = weights.sum(axis=1)
        hits = logged.scores[:, stage][None, :] <= radii[stage, :, None]
        coverage = np.sum(weights * hits, axis=1) / weight_sum
        observed_base_width = base_width[states, actions]
        candidate_width = radii[stage, :, None] * observed_base_width[None, :]
        width = np.sum(weights * candidate_width, axis=1) / weight_sum
        ess = weight_sum**2 / np.sum(weights**2, axis=1) / trajectory_count
        feasible = coverage >= target
        if not bool(feasible.any()):
            return _PrefixSelection(
                available=False,
                selected_indices=selected_indices,
                estimated_coverage=estimated_coverage,
                estimated_width=estimated_width,
                ess_fraction=ess_fraction,
                log_weight_span=log_weight_span,
                failure_stage=stage,
            )
        objective = np.where(feasible, width, np.inf)
        selected = int(objective.argmin())
        selected_indices[stage] = selected
        estimated_coverage[stage] = coverage[selected]
        estimated_width[stage] = width[selected]
        ess_fraction[stage] = ess[selected]
        log_weight_span[stage] = maximum[selected] - minimum[selected]
        raw_log_prefix = candidate_log_weight[selected].copy()
    return _PrefixSelection(
        available=True,
        selected_indices=selected_indices,
        estimated_coverage=estimated_coverage,
        estimated_width=estimated_width,
        ess_fraction=ess_fraction,
        log_weight_span=log_weight_span,
        failure_stage=None,
    )


def _store_layer_result(
    arrays: dict[str, np.ndarray],
    instance: int,
    arm: int,
    problem: AnalyticFiniteMDP,
    selection: _PrefixSelection,
) -> None:
    arrays["selected"][instance, arm] = selection.available
    arrays["selected_indices"][instance, arm] = selection.selected_indices
    arrays["estimated_coverage"][instance, arm] = selection.estimated_coverage
    arrays["estimated_normalized_width"][instance, arm] = selection.estimated_width
    arrays["ess_fraction"][instance, arm] = selection.ess_fraction
    arrays["log_weight_span"][instance, arm] = selection.log_weight_span
    arrays["failure_stage"][instance, arm] = (
        -1 if selection.failure_stage is None else selection.failure_stage
    )
    if not selection.available:
        return
    exact = analytic_schedule_metrics(
        problem,
        tuple(int(index) for index in selection.selected_indices),
    )
    arrays["exact_coverage"][instance, arm] = np.asarray(exact.coverage)
    arrays["exact_normalized_width"][instance, arm] = np.asarray(
        exact.normalized_width
    )


def _layer_record(
    *,
    layer: str,
    arm: str,
    problem_seed: int,
    nuisance_seed: int,
    calibration_seed: int,
    target_law_fingerprint: str,
    target_policy_drift_from_oracle: float,
    selection: _PrefixSelection,
    arrays: dict[str, np.ndarray],
    instance: int,
    arm_index: int,
    selected_policy_tv_by_stage: np.ndarray,
    selected_policy_tv_definition: str,
    matched_oracle_target_tv_by_stage: np.ndarray | None = None,
    primary_oracle_deployment_tv_by_stage: np.ndarray | None = None,
) -> dict[str, Any]:
    record = {
        "layer": layer,
        "problem_seed": problem_seed,
        "nuisance_seed": nuisance_seed,
        "calibration_seed": calibration_seed,
        "arm": arm,
        "selection_available": selection.available,
        "failure_stage": selection.failure_stage,
        "selected_indices": selection.selected_indices.tolist(),
        "estimated_coverage_by_stage": selection.estimated_coverage.tolist(),
        "estimated_normalized_width_by_stage": selection.estimated_width.tolist(),
        "exact_coverage_by_stage": arrays["exact_coverage"][instance, arm_index].tolist(),
        "exact_normalized_width_by_stage": arrays["exact_normalized_width"][
            instance, arm_index
        ].tolist(),
        "ess_fraction_by_stage": selection.ess_fraction.tolist(),
        "log_weight_span_by_stage": selection.log_weight_span.tolist(),
        "target_law_fingerprint": target_law_fingerprint,
        "target_policy_drift_from_oracle": target_policy_drift_from_oracle,
        "selected_policy_tv_by_stage": selected_policy_tv_by_stage.tolist(),
        "selected_policy_tv_mean": _finite_mean_or_nan(
            selected_policy_tv_by_stage
        ),
        "selected_policy_tv_definition": selected_policy_tv_definition,
    }
    if matched_oracle_target_tv_by_stage is not None:
        record["matched_oracle_target_tv_by_stage"] = (
            matched_oracle_target_tv_by_stage.tolist()
        )
        record["matched_oracle_target_tv_mean"] = _finite_mean_or_nan(
            matched_oracle_target_tv_by_stage
        )
    if primary_oracle_deployment_tv_by_stage is not None:
        record["primary_oracle_deployment_tv_by_stage"] = (
            primary_oracle_deployment_tv_by_stage.tolist()
        )
        record["primary_oracle_deployment_tv_mean"] = _finite_mean_or_nan(
            primary_oracle_deployment_tv_by_stage
        )
    return record


def _selected_policy(
    policy_surface: np.ndarray,
    selection: _PrefixSelection,
) -> np.ndarray | None:
    """Return the stagewise policy actually deployed by a selected schedule."""

    if not selection.available:
        return None
    horizon = len(selection.selected_indices)
    return np.stack(
        [
            policy_surface[stage, int(selection.selected_indices[stage])]
            for stage in range(horizon)
        ]
    )


def _policy_tv_by_stage(
    deployed_policy: np.ndarray,
    reference_policy: np.ndarray,
    state_occupancy: np.ndarray,
) -> np.ndarray:
    """Stagewise action TV under a common oracle-logging state measure."""

    reference = np.asarray(reference_policy)
    if reference.ndim == 2:
        reference = np.broadcast_to(reference, deployed_policy.shape)
    if reference.shape != deployed_policy.shape:
        raise ValueError("policy-TV reference has the wrong shape")
    state_tv = 0.5 * np.abs(deployed_policy - reference).sum(axis=2)
    return np.einsum("ts,ts->t", state_occupancy, state_tv)


def _finite_mean_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def _mean_policy_tv(
    target_policy: np.ndarray,
    behavior: np.ndarray,
    state_occupancy: np.ndarray,
) -> float:
    tv_by_state = 0.5 * np.abs(
        target_policy - behavior[None, None, :, :]
    ).sum(axis=3)
    return float(np.einsum("ts,tks->tk", state_occupancy, tv_by_state).mean())


def _mean_target_policy_drift(
    target_policy: np.ndarray,
    oracle_target: np.ndarray,
    state_occupancy: np.ndarray,
) -> float:
    tv_by_state = 0.5 * np.abs(target_policy - oracle_target).sum(axis=3)
    return float(np.einsum("ts,tks->tk", state_occupancy, tv_by_state).mean())


def _target_law_fingerprint(problem: AnalyticFiniteMDP) -> str:
    digest = hashlib.sha256()
    for value in (
        problem.initial_state_probabilities,
        problem.transition_probabilities,
        problem.action_probabilities,
        problem.radii,
        problem.predictor_means,
        problem.predictor_scales,
        problem.outcome_means,
        problem.outcome_standard_deviations,
        problem.outcome_normalization,
    ):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _nuisance_summary(
    arrays: dict[str, np.ndarray],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    return {
        metric: _arm_metric_summary(values, bootstrap_indices)
        for metric, values in arrays.items()
        if metric != "iterations"
    } | {
        "iterations": {
            arm: _metric_summary(values[:, arm_index])
            for arm_index, arm in enumerate(PROPENSITY_ARMS)
            for values in (arrays["iterations"],)
        }
    }


def _arm_metric_summary(
    values: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for arm_index, arm in enumerate(PROPENSITY_ARMS):
        observed = values[:, arm_index]
        draws = observed[bootstrap_indices].mean(axis=1)
        result[arm] = {
            "mean": float(observed.mean()),
            "ci95": _percentile_interval(draws),
        }
    return result


def _policy_tv_summary(
    values: np.ndarray,
    selected: np.ndarray,
    bootstrap_indices: np.ndarray,
    *,
    reference: str,
) -> dict[str, Any]:
    """Summarize paired selected-policy TV on a joint-availability population."""

    finite = np.isfinite(values).all(axis=2)
    joint = selected.all(axis=1) & finite.all(axis=1)
    arms: dict[str, Any] = {}
    for arm_index, arm in enumerate(PROPENSITY_ARMS):
        observed = values[joint, arm_index]
        point = float(observed.mean()) if len(observed) else float("nan")
        stage_mean = (
            observed.mean(axis=0).tolist()
            if len(observed)
            else [float("nan")] * values.shape[2]
        )
        draws = np.full(len(bootstrap_indices), np.nan)
        for draw_index, sampled in enumerate(bootstrap_indices):
            sampled_joint = joint[sampled]
            if bool(sampled_joint.any()):
                draws[draw_index] = values[
                    sampled[sampled_joint],
                    arm_index,
                ].mean()
        arms[arm] = {
            "mean_over_stage_and_paired_instance": point,
            "mean_by_stage": stage_mean,
            "ci95": _percentile_interval(draws),
        }
    return {
        "reference_and_measure": reference,
        "aggregation_population": "jointly_available_paired_problem_instances",
        "joint_complete_case_count": int(joint.sum()),
        "arms": arms,
        "bootstrap": {
            "resamples": len(bootstrap_indices),
            "interval": "paired_problem_seed_percentile",
            "same_problem_seed_matrix_for_every_arm": True,
        },
    }


def _layer_summary(
    arrays: dict[str, np.ndarray],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    selected = arrays["selected"]
    joint = selected.all(axis=1)
    point = _layer_metrics(arrays, joint)
    draws = {
        "marginal_worst_step_coverage": np.full(
            (len(bootstrap_indices), len(PROPENSITY_ARMS)), np.nan
        ),
        "mean_normalized_width": np.full(
            (len(bootstrap_indices), len(PROPENSITY_ARMS)), np.nan
        ),
        "minimum_stage_mean_ess_fraction": np.full(
            (len(bootstrap_indices), len(PROPENSITY_ARMS)), np.nan
        ),
    }
    for draw_index, sampled in enumerate(bootstrap_indices):
        sampled_joint = joint[sampled]
        if not bool(sampled_joint.any()):
            continue
        sampled_arrays = {
            name: value[sampled]
            for name, value in arrays.items()
            if value.shape[0] == len(selected)
        }
        sampled_metrics = _layer_metrics(sampled_arrays, sampled_joint)
        for metric in draws:
            draws[metric][draw_index] = [
                sampled_metrics[arm][metric] for arm in PROPENSITY_ARMS
            ]

    arm_summary = {}
    for arm_index, arm in enumerate(PROPENSITY_ARMS):
        arm_summary[arm] = {
            **point[arm],
            "selection_rate": float(selected[:, arm_index].mean()),
            "joint_complete_case_count": int(joint.sum()),
        }
        for metric, values in draws.items():
            arm_summary[arm][f"{metric}_ci95"] = _percentile_interval(
                values[:, arm_index]
            )

    oracle_index = PROPENSITY_ARMS.index("oracle")
    comparisons = {}
    for arm_index, arm in enumerate(PROPENSITY_ARMS[1:], start=1):
        coverage_degradation = (
            draws["marginal_worst_step_coverage"][:, oracle_index]
            - draws["marginal_worst_step_coverage"][:, arm_index]
        )
        width_difference = (
            draws["mean_normalized_width"][:, arm_index]
            - draws["mean_normalized_width"][:, oracle_index]
        )
        ess_difference = (
            draws["minimum_stage_mean_ess_fraction"][:, arm_index]
            - draws["minimum_stage_mean_ess_fraction"][:, oracle_index]
        )
        comparisons[arm] = {
            "wsc_degradation_from_nuisance": (
                point["oracle"]["marginal_worst_step_coverage"]
                - point[arm]["marginal_worst_step_coverage"]
            ),
            "wsc_degradation_from_nuisance_ci95": _percentile_interval(
                coverage_degradation
            ),
            "width_difference_vs_oracle": (
                point[arm]["mean_normalized_width"]
                - point["oracle"]["mean_normalized_width"]
            ),
            "width_difference_vs_oracle_ci95": _percentile_interval(
                width_difference
            ),
            "minimum_stage_mean_ess_fraction_difference_vs_oracle": (
                point[arm]["minimum_stage_mean_ess_fraction"]
                - point["oracle"]["minimum_stage_mean_ess_fraction"]
            ),
            "minimum_stage_mean_ess_fraction_difference_vs_oracle_ci95": (
                _percentile_interval(ess_difference)
            ),
        }
    return {
        "aggregation_population": "jointly_available_paired_problem_instances",
        "joint_complete_case_count": int(joint.sum()),
        "arms": arm_summary,
        "paired_comparisons_vs_oracle": comparisons,
        "bootstrap": {
            "resamples": len(bootstrap_indices),
            "same_problem_seed_matrix_for_every_arm_and_metric": True,
            "interval": "paired_seed_percentile",
        },
    }


def _layer_metrics(
    arrays: dict[str, np.ndarray],
    include: np.ndarray,
) -> dict[str, dict[str, float]]:
    if not bool(include.any()):
        return {
            arm: {
                "marginal_worst_step_coverage": float("nan"),
                "mean_normalized_width": float("nan"),
                "minimum_stage_mean_ess_fraction": float("nan"),
            }
            for arm in PROPENSITY_ARMS
        }
    result = {}
    for arm_index, arm in enumerate(PROPENSITY_ARMS):
        coverage = arrays["exact_coverage"][include, arm_index]
        width = arrays["exact_normalized_width"][include, arm_index]
        ess = arrays["ess_fraction"][include, arm_index]
        result[arm] = {
            "marginal_worst_step_coverage": float(coverage.mean(axis=0).min()),
            "mean_normalized_width": float(width.mean()),
            "minimum_stage_mean_ess_fraction": float(ess.mean(axis=0).min()),
        }
    return result


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _percentile_interval(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return [None, None]
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return [float(lower), float(upper)]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)
