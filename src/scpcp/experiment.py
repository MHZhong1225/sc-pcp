"""End-to-end per-step SC-PCP experiment orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from scpcp.baselines import (
    aci_style_controller,
    finite_depth_mfcs_selection,
    historical_per_step_radius,
    multidim_spci_style_controller,
    prc_max_time,
    repeated_recalibration,
)
from scpcp.behavior import fit_behavior_policy
from scpcp.certification import (
    CertificationResult,
    exact_tabular_l1_lower_bounds,
    practical_bootstrap_lower_bounds,
    simultaneous_lower_bounds,
)
from scpcp.config import ExperimentConfig
from scpcp.cot import (
    cot_state_action_weights,
    exact_tabular_cot_l1_error_bound,
    exact_tabular_state_action_weights,
    fit_cot,
    prefix_importance_weights,
)
from scpcp.coverage import (
    dcov_surface,
    diagonal_coverage_estimates,
    effective_sample_sizes,
    estimate_oracle_surface,
    fixed_q_grid,
    per_step_oracle_metrics,
    self_normalized_dcov_surface,
    self_normalized_diagonal_coverage_estimates,
)
from scpcp.data import DataSplits, TrajectoryBatch, concatenate_trajectories, patient_level_splits
from scpcp.outcome_model import fit_outcome_model
from scpcp.policy import BehaviorAnchoredPolicy
from scpcp.scores import fit_conformal_region, score_batch
from scpcp.selection import RadiusSelection, select_certified_radius, select_empirical_radius, select_lcb_radius
from scpcp.simulator import (
    EmpiricalTransitionEnvironment,
    SyntheticBehaviorPolicy,
    SyntheticTreatmentEnvironment,
    TabularBehaviorPolicy,
    TabularTreatmentEnvironment,
    rollout,
)


SCPCP_METHOD = "SC-PCP"
IW_ABLATION_METHOD = "IW-SC-PCP"


@dataclass(frozen=True)
class SeedResult:
    seed: int
    device: str
    records: list[dict[str, Any]]
    surfaces: dict[str, Tensor]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _Task:
    environment: object
    splits: DataSplits
    n_actions: int
    logging_policy: object | None
    name: str
    policy_config: object
    static_indices: tuple[int, ...] = ()
    action_mapping: dict[int, int] | None = None
    state_feature_names: tuple[str, ...] = ()


def _deployment_selection(
    certificate: CertificationResult,
    certified_selection: RadiusSelection,
    practical_selection: RadiusSelection,
) -> RadiusSelection:
    """Use one method name while retaining an explicit certificate status."""

    return certified_selection if certificate.formal else practical_selection


def run_seed(config: ExperimentConfig, *, seed: int, device: str) -> SeedResult:
    """Fit once on the prespecified roles, then evaluate all frozen methods."""

    torch.manual_seed(seed)
    task = _prepare_task(config, seed=seed, device=device)
    splits = task.splits
    outcome_model = fit_outcome_model(
        splits.predictor,
        n_actions=task.n_actions,
        config=config.model,
        device=device,
        seed=seed + 1,
        static_indices=task.static_indices,
    )
    region = fit_conformal_region(outcome_model)
    behavior_model = None
    if splits.behavior is not None:
        behavior_model = fit_behavior_policy(
            splits.behavior,
            n_actions=task.n_actions,
            model_config=config.model,
            policy_config=task.policy_config,
            device=device,
            seed=seed + 2,
            static_indices=task.static_indices,
        )
    reference_policy = task.logging_policy if task.logging_policy is not None else behavior_model
    if reference_policy is None:
        raise RuntimeError("a known or fitted logging policy is required")
    policy = BehaviorAnchoredPolicy(
        outcome_model=outcome_model,
        reference_policy=reference_policy,
        config=task.policy_config,
        region=region,
        tilt=config.policy.tilt,
    )
    # The policy anchor and the OPE denominator are conceptually distinct.
    # In observational clinical data both are necessarily the fitted propensity
    # model.  Synthetic and tabular experiments instead retain the known
    # data-generating logging policy in the denominator, so COT/IW error is not
    # conflated with a propensity-estimation error in theorem validation.
    logging_policy = task.logging_policy if task.logging_policy is not None else behavior_model
    if logging_policy is None:
        raise RuntimeError("the OPE denominator is unavailable")
    cot_scores = score_batch(region, splits.cot.current_states(), splits.cot.actions, splits.cot.outcomes)
    q_grid = fixed_q_grid(
        cot_scores,
        size=config.q_grid_size,
        lower_quantile=config.q_quantile_min,
        upper_quantile=config.q_quantile_max,
    )
    fitted_cot = fit_cot(
        splits.cot,
        q_grid=q_grid,
        target_policy=policy,
        logging_policy=logging_policy,
        outcome_model=outcome_model,
        config=config.cot,
        device=device,
        seed=seed + 3,
    )
    cert_scores = score_batch(
        region, splits.certification.current_states(), splits.certification.actions, splits.certification.outcomes
    )
    baseline_batch = concatenate_trajectories(splits.cot, splits.certification)
    baseline_scores = torch.cat((cot_scores, cert_scores), dim=0)
    cot_weights, cot_weight_diagnostics = cot_state_action_weights(
        fitted_cot,
        splits.certification,
        q_grid=q_grid,
        target_policy=policy,
        logging_policy=logging_policy,
        weight_cap=config.cot.weight_cap,
    )
    iw_weights, iw_weight_diagnostics = prefix_importance_weights(
        splits.certification,
        q_grid=q_grid,
        target_policy=policy,
        logging_policy=logging_policy,
        weight_cap=config.cot.weight_cap,
    )
    cot_diagonal = diagonal_coverage_estimates(cot_weights, cert_scores.to(cot_weights), q_grid.to(cot_weights))
    iw_diagonal = diagonal_coverage_estimates(iw_weights, cert_scores.to(iw_weights), q_grid.to(iw_weights))
    cot_practical_diagonal = self_normalized_diagonal_coverage_estimates(
        cot_weights,
        cert_scores.to(cot_weights),
        q_grid.to(cot_weights),
    )
    iw_practical_diagonal = self_normalized_diagonal_coverage_estimates(
        iw_weights,
        cert_scores.to(iw_weights),
        q_grid.to(iw_weights),
    )
    cot_certificate = simultaneous_lower_bounds(
        cot_diagonal,
        n_trajectories=splits.certification.n,
        weight_cap=config.cot.weight_cap,
        config=config.certification,
        cluster_ids=splits.certification.patient_ids,
    )
    # A declared occupancy-ratio theorem concerns COT, not the distinct
    # trajectory-prefix IW estimator.  Until an IW-specific simultaneous error
    # premise is implemented, keep this ablation practical-only.
    iw_certification_config = (
        config.certification
        if config.certification.ratio_bound_source != "declared"
        else replace(config.certification, ratio_error_bound=0.0, ratio_bound_source="none", ratio_delta=0.0)
    )
    iw_certificate = simultaneous_lower_bounds(
        iw_diagonal,
        n_trajectories=splits.certification.n,
        weight_cap=config.cot.weight_cap,
        config=iw_certification_config,
        cluster_ids=splits.certification.patient_ids,
    )
    cot_selection = select_certified_radius(q_grid, cot_certificate, alpha=config.certification.alpha)
    cot_bootstrap_certificate = None if cot_certificate.formal else practical_bootstrap_lower_bounds(
        cot_weights,
        cert_scores.to(cot_weights),
        q_grid.to(cot_weights),
        lower_tail=config.certification.delta,
        n_resamples=config.certification.practical_bootstrap_resamples,
        seed=seed + 31_337,
        cluster_ids=splits.certification.patient_ids,
    )
    cot_practical_selection = (
        select_lcb_radius(q_grid, cot_certificate, alpha=config.certification.alpha)
        if cot_bootstrap_certificate is None
        else select_lcb_radius(
            q_grid,
            cot_bootstrap_certificate,
            alpha=config.certification.alpha,
            status="PRACTICAL_CLUSTER_MAX_T_LCB",
        )
    )
    iw_selection = select_certified_radius(q_grid, iw_certificate, alpha=config.certification.alpha)
    iw_bootstrap_certificate = None if iw_certificate.formal else practical_bootstrap_lower_bounds(
        iw_weights,
        cert_scores.to(iw_weights),
        q_grid.to(iw_weights),
        lower_tail=config.certification.delta,
        n_resamples=config.certification.practical_bootstrap_resamples,
        seed=seed + 47_021,
        cluster_ids=splits.certification.patient_ids,
    )
    iw_practical_selection = (
        select_lcb_radius(q_grid, iw_certificate, alpha=config.certification.alpha)
        if iw_bootstrap_certificate is None
        else select_lcb_radius(
            q_grid,
            iw_bootstrap_certificate,
            alpha=config.certification.alpha,
            status="PRACTICAL_CLUSTER_MAX_T_LCB",
        )
    )
    exact_diagonal = exact_certificate = exact_ess = None
    learned_cot_oracle_l1_error_bound = learned_cot_oracle_l1_certificate = learned_cot_oracle_l1_selection = None
    exact_cot_l1_error = exact_cot_cdf_error = exact_iw_cdf_error = None
    if task.name == "tabular" and hasattr(task.environment, "exact_state_ratios") and task.logging_policy is not None:
        # This finite-MDP-only branch enumerates the population discrepancy of
        # the *capped learned* COT weights.  Unlike an empirical D_cert error,
        # it is a deterministic oracle-validation term and can support the
        # dedicated formal certificate below.
        learned_cot_oracle_l1_error_bound = exact_tabular_cot_l1_error_bound(
            fitted_cot,
            task.environment,
            q_grid=q_grid,
            target_policy=policy,
            logging_policy=task.logging_policy,
            weight_cap=config.cot.weight_cap,
        )
        learned_cot_oracle_l1_certificate = exact_tabular_l1_lower_bounds(
            cot_diagonal,
            n_trajectories=splits.certification.n,
            weight_cap=config.cot.weight_cap,
            exact_l1_error_bound=learned_cot_oracle_l1_error_bound,
            delta=config.certification.delta,
            cluster_ids=splits.certification.patient_ids,
        )
        learned_cot_oracle_l1_selection = select_certified_radius(
            q_grid,
            learned_cot_oracle_l1_certificate,
            alpha=config.certification.alpha,
        )
        exact_weights, exact_weight_bound = exact_tabular_state_action_weights(
            task.environment,
            splits.certification,
            q_grid=q_grid,
            target_policy=policy,
            logging_policy=task.logging_policy,
        )
        exact_diagonal = diagonal_coverage_estimates(exact_weights, cert_scores.to(exact_weights), q_grid.to(exact_weights))
        oracle_certificate_config = replace(
            config.certification,
            ratio_error_bound=0.0,
            ratio_bound_source="oracle",
            ratio_delta=0.0,
        )
        exact_certificate = simultaneous_lower_bounds(
            exact_diagonal,
            n_trajectories=splits.certification.n,
            weight_cap=exact_weight_bound,
            config=oracle_certificate_config,
            allow_oracle=True,
            cluster_ids=splits.certification.patient_ids,
        )
        exact_ess = effective_sample_sizes(exact_weights, splits.certification.patient_ids)
        # Held-out empirical oracle diagnostics for the COT error premise and
        # the RQ4 CDF comparison.  These are available only in the finite MDP;
        # they are never recycled into radius selection.
        exact_cot_l1_error = (cot_weights - exact_weights.to(cot_weights)).abs().mean(dim=0).transpose(0, 1)
        exact_cot_cdf_error = (cot_diagonal - exact_diagonal.to(cot_diagonal)).abs()
        exact_iw_cdf_error = (iw_diagonal - exact_diagonal.to(iw_diagonal)).abs()
    # Baselines that do not fit COT receive the union of the two post-predictor
    # calibration roles, so they are not disadvantaged in total data access.
    historical_radius = historical_per_step_radius(baseline_scores, config.certification.alpha)
    mfcs_selection, mfcs_diagonal = finite_depth_mfcs_selection(
        baseline_batch.to(device),
        baseline_scores.to(device),
        q_grid=q_grid.to(device),
        target_policy=policy,
        logging_policy=logging_policy,
        depth=3,
        alpha=config.certification.alpha,
        weight_cap=config.cot.weight_cap,
    )
    # A full on-policy response surface is an internal simulator diagnostic,
    # not part of SC-PCP.  Clinical configurations cannot expose a true oracle
    # and previously paid hundreds of thousands of unnecessary rollouts here.
    oracle_surface = oracle_selection = None
    if task.name in {"synthetic", "tabular"}:
        oracle_surface = estimate_oracle_surface(
            task.environment,
            policy,
            region,
            q_grid=q_grid,
            horizon=config.horizon,
            n_rollouts=config.samples.oracle_surface_rollouts,
            seed=seed + 4,
            device=device,
        )
        oracle_selection = select_empirical_radius(
            q_grid,
            oracle_surface.diagonal,
            alpha=config.certification.alpha,
        )
    cot_deployment_selection = _deployment_selection(cot_certificate, cot_selection, cot_practical_selection)
    iw_deployment_selection = _deployment_selection(iw_certificate, iw_selection, iw_practical_selection)
    cot_deployment_certificate = cot_certificate if cot_bootstrap_certificate is None else cot_bootstrap_certificate
    iw_deployment_certificate = iw_certificate if iw_bootstrap_certificate is None else iw_bootstrap_certificate
    cot_deployment_diagonal = cot_deployment_certificate.estimates
    iw_deployment_diagonal = iw_deployment_certificate.estimates
    records = [
        _evaluate_scalar_method("Historical CP", historical_radius, task, policy, region, config, seed, device),
        _evaluate_scalar_method("MFCS-style (depth=3)", mfcs_selection.radius, task, policy, region, config, seed + 11, device, selection=mfcs_selection),
        _evaluate_scalar_method(IW_ABLATION_METHOD, iw_deployment_selection.radius, task, policy, region, config, seed + 22, device, selection=iw_deployment_selection, certificate=iw_deployment_certificate),
        _evaluate_scalar_method(SCPCP_METHOD, cot_deployment_selection.radius, task, policy, region, config, seed + 44, device, selection=cot_deployment_selection, certificate=cot_deployment_certificate),
    ]
    if oracle_selection is not None:
        records.append(
            _evaluate_scalar_method(
                "MC-oracle SC-PCP (reference)",
                oracle_selection.radius,
                task,
                policy,
                region,
                config,
                seed + 55,
                device,
                selection=oracle_selection,
                target_deployments=len(q_grid) * config.samples.oracle_surface_rollouts,
                information_regime="on_policy_oracle_reference",
            )
        )
    online = aci_style_controller(
        task.environment,
        policy,
        region,
        baseline_scores,
        alpha=config.certification.alpha,
        gamma=0.01,
        rounds=3,
        total_rollouts=config.samples.online_rollouts,
        horizon=config.horizon,
        seed=seed + 66,
        device=device,
    )
    records.append(
        _evaluate_stagewise_method("ACI-style online", online.radius_by_time, task, policy, region, config, seed + 77, device, online.target_deployments)
    )
    spci = multidim_spci_style_controller(
        task.environment,
        policy,
        region,
        baseline_scores,
        alpha=config.certification.alpha,
        rounds=3,
        total_rollouts=config.samples.online_rollouts,
        horizon=config.horizon,
        seed=seed + 82,
        device=device,
    )
    records.append(
        _evaluate_stagewise_method("MultiDimSPCI-style online", spci.radius_by_time, task, policy, region, config, seed + 83, device, spci.target_deployments)
    )
    repeated = repeated_recalibration(
        task.environment,
        policy,
        region,
        historical_radius,
        alpha=config.certification.alpha,
        rounds=3,
        total_rollouts=config.samples.online_rollouts,
        horizon=config.horizon,
        seed=seed + 88,
        device=device,
    )
    records.append(
        _evaluate_stagewise_method("Repeated recalibration", repeated.radius_by_time, task, policy, region, config, seed + 99, device, repeated.target_deployments)
    )
    prc = prc_max_time(
        task.environment,
        policy,
        region,
        historical_radius,
        q_grid,
        alpha=config.certification.alpha,
        delta=config.certification.delta,
        rounds=3,
        total_rollouts=config.samples.online_rollouts,
        horizon=config.horizon,
        seed=seed + 103,
        device=device,
    )
    records.append(
        _evaluate_stagewise_method(
            "PRC-MaxTime-style online (grid-adapted)",
            prc.radius_by_time,
            task,
            policy,
            region,
            config,
            seed + 104,
            device,
            prc.target_deployments,
        )
    )
    cot_ess = effective_sample_sizes(cot_weights, splits.certification.patient_ids)
    iw_ess = effective_sample_sizes(iw_weights, splits.certification.patient_ids)
    records.extend(
        (
            _logged_record("Historical CP", historical_radius, cert_scores, None, None, None, policy, logging_policy, splits.certification, config, q_grid),
            _logged_record("MFCS-style (depth=3)", mfcs_selection.radius, cert_scores, mfcs_diagonal, None, None, policy, logging_policy, splits.certification, config, q_grid, mfcs_selection),
            _logged_record(IW_ABLATION_METHOD, iw_deployment_selection.radius, cert_scores, iw_deployment_diagonal, iw_deployment_certificate, iw_ess, policy, logging_policy, splits.certification, config, q_grid, iw_deployment_selection),
            _logged_record(SCPCP_METHOD, cot_deployment_selection.radius, cert_scores, cot_deployment_diagonal, cot_deployment_certificate, cot_ess, policy, logging_policy, splits.certification, config, q_grid, cot_deployment_selection),
        )
    )
    cot_raw_dcov = dcov_surface(cot_weights, cert_scores.to(cot_weights), q_grid.to(cot_weights))
    iw_raw_dcov = dcov_surface(iw_weights, cert_scores.to(iw_weights), q_grid.to(iw_weights))
    cot_practical_dcov = self_normalized_dcov_surface(
        cot_weights,
        cert_scores.to(cot_weights),
        q_grid.to(cot_weights),
    )
    iw_practical_dcov = self_normalized_dcov_surface(
        iw_weights,
        cert_scores.to(iw_weights),
        q_grid.to(iw_weights),
    )
    surfaces = {
        "q_grid": q_grid,
        "cot_dcov": cot_raw_dcov if cot_certificate.formal else cot_practical_dcov,
        "iw_dcov": iw_raw_dcov if iw_certificate.formal else iw_practical_dcov,
        "cot_dcov_raw": cot_raw_dcov,
        "iw_dcov_raw": iw_raw_dcov,
        "cot_dcov_self_normalized": cot_practical_dcov,
        "iw_dcov_self_normalized": iw_practical_dcov,
        "cot_diagonal": cot_deployment_diagonal,
        "iw_diagonal": iw_deployment_diagonal,
        "cot_diagonal_raw": cot_diagonal,
        "iw_diagonal_raw": iw_diagonal,
        "cot_diagonal_self_normalized": cot_practical_diagonal,
        "iw_diagonal_self_normalized": iw_practical_diagonal,
        "mfcs_diagonal": mfcs_diagonal,
        "cot_lower_bounds": cot_deployment_certificate.lower_bounds,
        "iw_lower_bounds": iw_deployment_certificate.lower_bounds,
        "cot_ht_lower_bounds": cot_certificate.lower_bounds,
        "iw_ht_lower_bounds": iw_certificate.lower_bounds,
        "cot_sampling_margin": torch.tensor(cot_deployment_certificate.sampling_margin),
        "iw_sampling_margin": torch.tensor(iw_deployment_certificate.sampling_margin),
        "cot_ht_sampling_margin": torch.tensor(cot_certificate.sampling_margin),
        "iw_ht_sampling_margin": torch.tensor(iw_certificate.sampling_margin),
        "cot_ratio_error_bound": cot_deployment_certificate.ratio_error_bound,
        "iw_ratio_error_bound": iw_deployment_certificate.ratio_error_bound,
        "cot_ht_ratio_error_bound": cot_certificate.ratio_error_bound,
        "iw_ht_ratio_error_bound": iw_certificate.ratio_error_bound,
        "cot_ess": cot_ess,
        "iw_ess": iw_ess,
        "cot_weight_variance_pre_cap": cot_weight_diagnostics.raw_variance,
        "iw_weight_variance_pre_cap": iw_weight_diagnostics.raw_variance,
        "cot_cap_hit_rate": cot_weight_diagnostics.cap_hit_rate,
        "iw_cap_hit_rate": iw_weight_diagnostics.cap_hit_rate,
        "cot_weight_maximum_pre_cap": cot_weight_diagnostics.raw_maximum,
        "iw_weight_maximum_pre_cap": iw_weight_diagnostics.raw_maximum,
    }
    if oracle_surface is not None:
        surfaces["oracle_dcov"] = oracle_surface.surface
    if learned_cot_oracle_l1_certificate is not None:
        surfaces["learned_cot_oracle_l1_error_bound"] = learned_cot_oracle_l1_error_bound
        surfaces["learned_cot_oracle_l1_lower_bounds"] = learned_cot_oracle_l1_certificate.lower_bounds
        surfaces["learned_cot_oracle_l1_sampling_margin"] = torch.tensor(
            learned_cot_oracle_l1_certificate.sampling_margin
        )
    if exact_diagonal is not None:
        surfaces["exact_cot_diagonal"] = exact_diagonal
        surfaces["exact_cot_lower_bounds"] = exact_certificate.lower_bounds
        surfaces["exact_cot_ess"] = exact_ess
        surfaces["exact_cot_sampling_margin"] = torch.tensor(exact_certificate.sampling_margin)
        surfaces["exact_cot_ratio_error_bound"] = exact_certificate.ratio_error_bound
        surfaces["exact_cot_l1_error_on_dcert"] = exact_cot_l1_error
        surfaces["exact_cot_cdf_error"] = exact_cot_cdf_error
        surfaces["exact_iw_cdf_error"] = exact_iw_cdf_error
    diagnostics = {
        "dataset": task.name,
        "q_grid_size": len(q_grid),
        "cot": fitted_cot.diagnostics,
        "cot_cap_hit_rate": float(cot_weight_diagnostics.cap_hit_rate.mean().item()),
        "iw_cap_hit_rate": float(iw_weight_diagnostics.cap_hit_rate.mean().item()),
        "cot_certificate": cot_certificate.label,
        "iw_certificate": iw_certificate.label,
        "cot_practical_certificate": "" if cot_bootstrap_certificate is None else cot_bootstrap_certificate.label,
        "iw_practical_certificate": "" if iw_bootstrap_certificate is None else iw_bootstrap_certificate.label,
        "scpcp_deployment_certificate": cot_deployment_certificate.label,
        "scpcp_deployment_certificate_formal": cot_deployment_certificate.formal,
        "iw_scpcp_deployment_certificate": iw_deployment_certificate.label,
        "iw_scpcp_deployment_certificate_formal": iw_deployment_certificate.formal,
        "learned_cot_oracle_l1_certificate": ""
        if learned_cot_oracle_l1_certificate is None
        else learned_cot_oracle_l1_certificate.label,
        "learned_cot_oracle_l1_error_bound_max": None
        if learned_cot_oracle_l1_error_bound is None
        else float(learned_cot_oracle_l1_error_bound.max().item()),
        "learned_cot_oracle_l1_selection_status": ""
        if learned_cot_oracle_l1_selection is None
        else learned_cot_oracle_l1_selection.status,
        "exact_cot_certificate": "" if exact_certificate is None else exact_certificate.label,
        "exact_cot_l1_error_max_on_dcert": None if exact_cot_l1_error is None else float(exact_cot_l1_error.max().item()),
        "exact_cot_cdf_error_max_on_dcert": None if exact_cot_cdf_error is None else float(exact_cot_cdf_error.max().item()),
        "exact_iw_cdf_error_max_on_dcert": None if exact_iw_cdf_error is None else float(exact_iw_cdf_error.max().item()),
        "ope_logging_policy": "known" if task.logging_policy is not None else "estimated",
        "policy_reference": "known_logging" if task.logging_policy is not None else "estimated_behavior",
        "original_to_model_action": {} if task.action_mapping is None else task.action_mapping,
        "base_state_feature_names": list(task.state_feature_names),
        "state_history_length": (
            config.model.history_length if config.model.architecture == "gru" else 1
        ),
        "split_sizes": {
            "pred": splits.predictor.n,
            "beh": 0 if splits.behavior is None else splits.behavior.n,
            "cot": splits.cot.n,
            "cert": splits.certification.n,
            "env": 0 if splits.environment is None else splits.environment.n,
        },
    }
    return SeedResult(seed=seed, device=device, records=records, surfaces=surfaces, diagnostics=diagnostics)


def _prepare_task(config: ExperimentConfig, *, seed: int, device: str) -> _Task:
    if config.data.dataset == "synthetic":
        environment = SyntheticTreatmentEnvironment(config.synthetic)
        logging = SyntheticBehaviorPolicy()
        logged = rollout(
            environment,
            logging,
            n=config.samples.logged,
            horizon=config.horizon,
            seed=seed,
            device=device,
        )
        splits = patient_level_splits(
            logged,
            seed=seed,
            include_environment=False,
            include_behavior=False,
        )
        return _Task(environment, splits, environment.n_actions, logging, "synthetic", config.policy)
    if config.data.dataset == "tabular":
        environment = TabularTreatmentEnvironment(config.synthetic)
        logging = TabularBehaviorPolicy()
        logged = rollout(
            environment,
            logging,
            n=config.samples.logged,
            horizon=config.horizon,
            seed=seed,
            device=device,
        )
        splits = patient_level_splits(
            logged,
            seed=seed,
            include_environment=False,
            include_behavior=False,
        )
        return _Task(environment, splits, environment.n_actions, logging, "tabular", config.policy)
    from scpcp.real_data import load_clinical_trajectories

    (
        logged,
        n_actions,
        static_indices,
        action_costs,
        action_mapping,
        state_feature_names,
    ) = load_clinical_trajectories(config, seed=seed, device=device)
    splits = patient_level_splits(logged, seed=seed, include_environment=True)
    if splits.environment is None:
        raise RuntimeError("clinical Track A requires D_env")
    environment = EmpiricalTransitionEnvironment(
        splits.environment,
        n_actions=n_actions,
        neighbors=config.data.empirical_neighbors,
        bandwidth=config.data.empirical_bandwidth,
        embedding_dim=config.data.empirical_embedding_dim,
        static_indices=static_indices,
        history_length=config.model.history_length if config.model.architecture == "gru" else 1,
    )
    return _Task(
        environment,
        splits,
        n_actions,
        None,
        config.data.dataset,
        replace(config.policy, action_costs=action_costs),
        static_indices=static_indices,
        action_mapping=action_mapping,
        state_feature_names=state_feature_names,
    )


def _evaluate_scalar_method(
    name: str,
    radius: float | None,
    task: _Task,
    policy: BehaviorAnchoredPolicy,
    outcome_model: object,
    config: ExperimentConfig,
    seed: int,
    device: str,
    *,
    selection: RadiusSelection | None = None,
    certificate: CertificationResult | None = None,
    target_deployments: int = 0,
    information_regime: str | None = None,
) -> dict[str, Any]:
    if radius is None:
        return _unavailable_record(
            name,
            selection,
            certificate,
            information_regime=information_regime,
            target_deployments=target_deployments,
        )
    coverage, deployed, scores = per_step_oracle_metrics(
        task.environment,
        policy,
        outcome_model,
        q=radius,
        horizon=config.horizon,
        n_rollouts=config.samples.oracle_rollouts,
        seed=seed,
        device=device,
    )
    return _deployment_record(
        name,
        radius,
        coverage,
        deployed,
        scores,
        config,
        policy,
        outcome_model,
        selection,
        certificate,
        target_deployments,
        information_regime,
    )


def _evaluate_stagewise_method(
    name: str,
    radii: Tensor,
    task: _Task,
    policy: BehaviorAnchoredPolicy,
    outcome_model: object,
    config: ExperimentConfig,
    seed: int,
    device: str,
    target_deployments: int,
) -> dict[str, Any]:
    coverage, deployed, scores = per_step_oracle_metrics(
        task.environment,
        policy,
        outcome_model,
        q=radii.to(device),
        horizon=config.horizon,
        n_rollouts=config.samples.oracle_rollouts,
        seed=seed,
        device=device,
    )
    return _deployment_record(
        name, radii.cpu(), coverage, deployed, scores, config, policy, outcome_model, None, None, target_deployments
    )


def _metric_placeholders() -> dict[str, float | str]:
    """Provide explicit NA fields for metrics belonging to the other track.

    Track A and Track B deliberately estimate different populations.  Keeping
    the non-applicable fields present (rather than relying on a sparse CSV
    union) prevents a generic downstream aggregation from silently treating a
    logged-clinician summary as a target-policy deployment result.
    """

    missing = float("nan")
    return {
        "mean_log_volume": missing,
        "median_volume": missing,
        "clinical_cost": missing,
        "clinical_utility": missing,
        "logged_descriptive_mean_log_volume": missing,
        "logged_descriptive_median_volume": missing,
        "logged_descriptive_clinical_cost": missing,
        "logged_descriptive_clinical_utility": missing,
        "logged_descriptive_per_time_clinical_cost": "[]",
        "logged_state_model_estimated_clinical_cost": missing,
        "logged_state_model_estimated_clinical_utility": missing,
    }


@torch.no_grad()
def _logged_descriptive_metrics(
    batch: TrajectoryBatch,
    *,
    radius: float,
    policy: BehaviorAnchoredPolicy,
) -> dict[str, float | str]:
    """Summarize a selected set on observed logged-source state-action pairs.

    These are post-selection, *logged-data descriptive* statistics.  The raw
    outcome/action cost is an observed logging-policy quantity.  The
    model-based value averages the frozen policy's one-step conditional cost
    over the logged-state distribution; it is useful for comparing selected
    radii, but is not a target-policy rollout or a clinical deployment value.
    """

    device = next(policy.outcome_model.parameters()).device
    states = batch.current_states().reshape(-1, batch.state_dim).to(device)
    actions = batch.actions.reshape(-1).to(device)
    outcomes = batch.outcomes.reshape(-1, batch.outcome_dim).to(device)
    _, scales = policy.outcome_model(states, actions)
    radii = torch.full((len(scales),), float(radius), device=device, dtype=scales.dtype)
    log_volumes = policy.region.log_volume(scales, radii)
    volumes = log_volumes.exp()

    action_costs = torch.as_tensor(policy.config.action_costs, dtype=outcomes.dtype, device=device)
    observed_cost = (
        policy.config.disease_weight * outcomes[:, 0]
        + policy.config.toxicity_weight * outcomes[:, 1]
        + action_costs[actions]
    )
    per_time_observed_cost = observed_cost.reshape(batch.n, batch.horizon).mean(dim=0)

    means, _ = policy.outcome_model.predict_all_actions(states)
    target_probabilities = policy.probabilities(states, float(radius))
    actionwise_model_cost = (
        policy.config.disease_weight * means[..., 0]
        + policy.config.toxicity_weight * means[..., 1]
        + action_costs[None, :]
    )
    conditional_policy_cost = (target_probabilities * actionwise_model_cost).sum(dim=1).mean()
    observed_mean_cost = observed_cost.mean()
    return {
        "logged_descriptive_mean_log_volume": float(log_volumes.mean().item()),
        "logged_descriptive_median_volume": float(volumes.median().item()),
        "logged_descriptive_clinical_cost": float(observed_mean_cost.item()),
        # Utility is deliberately defined as negative cost so that its
        # direction is unambiguous: larger values are better.
        "logged_descriptive_clinical_utility": float((-observed_mean_cost).item()),
        "logged_descriptive_per_time_clinical_cost": json.dumps(
            [float(value) for value in per_time_observed_cost.tolist()]
        ),
        "logged_state_model_estimated_clinical_cost": float(conditional_policy_cost.item()),
        "logged_state_model_estimated_clinical_utility": float((-conditional_policy_cost).item()),
    }


def _deployment_record(
    name: str,
    radius: float | Tensor,
    coverage: Tensor,
    deployed: TrajectoryBatch,
    scores: Tensor,
    config: ExperimentConfig,
    policy: BehaviorAnchoredPolicy,
    outcome_model: object,
    selection: RadiusSelection | None,
    certificate: CertificationResult | None,
    target_deployments: int,
    information_regime: str | None = None,
) -> dict[str, Any]:
    action_cost = torch.as_tensor(policy.config.action_costs, device=deployed.actions.device)[deployed.actions]
    realized_cost = (
        policy.config.disease_weight * deployed.outcomes[..., 0]
        + policy.config.toxicity_weight * deployed.outcomes[..., 1]
        + action_cost
    ).mean()
    flat_states, flat_actions, _ = deployed.flat_transitions()
    predictor = outcome_model.outcome_model if hasattr(outcome_model, "outcome_model") else outcome_model
    _, scales = predictor(flat_states, flat_actions)
    if isinstance(radius, Tensor):
        step_radius = radius.to(scales)[None, :].expand(deployed.n, -1).reshape(-1)
        pathwise_coverage = (scores <= radius.to(scores)[None, :]).all(dim=1).float().mean()
    else:
        step_radius = torch.full((len(scales),), radius, device=scales.device)
        pathwise_coverage = (scores <= radius).all(dim=1).float().mean()
    if hasattr(outcome_model, "log_volume"):
        log_volumes = outcome_model.log_volume(scales, step_radius)
        volumes = log_volumes.exp()
    else:
        volumes = 4.0 * step_radius.square() * scales[:, 0] * scales[:, 1]
        log_volumes = (volumes + 1e-12).log()
    return {
        **_metric_placeholders(),
        "track": "empirical_environment",
        "evaluation_scope": "fresh_target_policy_rollouts_in_frozen_empirical_environment",
        "prediction_set_metric_scope": "fresh_target_policy_state_action_pairs",
        "clinical_value_metric_scope": "realized_outcomes_and_actions_from_fresh_target_policy_rollouts",
        "clinical_utility_definition": "negative_clinical_cost_higher_is_better",
        "method": name,
        "information_regime": information_regime or ("on_policy_adaptation" if target_deployments else "offline_logged_data"),
        "selection_estimand": "per_step",
        "selected_q": float(radius) if not isinstance(radius, Tensor) else float(radius.mean().item()),
        "q_by_time": "" if not isinstance(radius, Tensor) else json.dumps([float(value) for value in radius.tolist()]),
        "selection_status": "FIXED" if selection is None else selection.status,
        "certificate_type": "" if certificate is None else certificate.label,
        "certificate_formal": False if certificate is None else certificate.formal,
        "worst_coverage": float(coverage.min().item()),
        "average_coverage": float(coverage.mean().item()),
        "pathwise_coverage": float(pathwise_coverage.item()),
        "worst_gap": float((1.0 - config.certification.alpha - coverage).clamp_min(0.0).max().item()),
        "per_time_coverage": json.dumps([float(value) for value in coverage.tolist()]),
        "mean_log_volume": float(log_volumes.mean().item()),
        "median_volume": float(volumes.median().item()),
        "clinical_cost": float(realized_cost.item()),
        "clinical_utility": float((-realized_cost).item()),
        "target_policy_trajectories": target_deployments,
        "oracle_evaluation_trajectories": deployed.n,
        "score_mean": float(scores.mean().item()),
    }


def _unavailable_record(
    name: str,
    selection: RadiusSelection | None,
    certificate: CertificationResult | None,
    *,
    information_regime: str | None = None,
    target_deployments: int = 0,
) -> dict[str, Any]:
    return {
        **_metric_placeholders(),
        "track": "empirical_environment",
        "evaluation_scope": "unavailable_target_policy_evaluation",
        "prediction_set_metric_scope": "unavailable",
        "clinical_value_metric_scope": "unavailable",
        "clinical_utility_definition": "negative_clinical_cost_higher_is_better",
        "method": name,
        "information_regime": information_regime or "offline_logged_data",
        "selection_estimand": "per_step",
        "selected_q": float("nan"),
        "q_by_time": "",
        "selection_status": "UNCERTIFIED" if selection is None else selection.status,
        "certificate_type": "" if certificate is None else certificate.label,
        "certificate_formal": False if certificate is None else certificate.formal,
        "worst_coverage": float("nan"),
        "average_coverage": float("nan"),
        "pathwise_coverage": float("nan"),
        "worst_gap": float("nan"),
        "per_time_coverage": "[]",
        "target_policy_trajectories": target_deployments,
        "oracle_evaluation_trajectories": 0,
        "score_mean": float("nan"),
    }


@torch.no_grad()
def _logged_record(
    name: str,
    radius: float | None,
    scores: Tensor,
    estimates: Tensor | None,
    certificate: CertificationResult | None,
    ess: Tensor | None,
    policy: BehaviorAnchoredPolicy,
    logging_policy: object,
    batch: TrajectoryBatch,
    config: ExperimentConfig,
    q_grid: Tensor,
    selection: RadiusSelection | None = None,
    information_regime: str = "offline_logged_data",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        **_metric_placeholders(),
        "track": "logged_data",
        "evaluation_scope": "logged_source_trajectories_descriptive_not_target_policy_deployment",
        "prediction_set_metric_scope": "observed_logged_state_action_pairs_post_selection_descriptive",
        "clinical_value_metric_scope": (
            "observed_logged_source_cost_descriptive_and_frozen_model_target_action_cost_"
            "conditioned_on_logged_states_not_target_policy_deployment"
        ),
        "clinical_utility_definition": "negative_clinical_cost_higher_is_better",
        "method": name,
        "information_regime": information_regime,
        "selection_estimand": "per_step",
        "selected_q": float("nan") if radius is None else radius,
        "q_by_time": "",
        "selection_status": "FIXED" if selection is None else selection.status,
        "certificate_type": "" if certificate is None else certificate.label,
        "certificate_formal": False if certificate is None else certificate.formal,
        "certified": bool(radius is not None and certificate is not None and certificate.formal and selection is not None and selection.status == "CERTIFIED"),
        "worst_coverage": float("nan"),
        "average_coverage": float("nan"),
        "pathwise_coverage": float("nan"),
        "worst_gap": float("nan"),
        "per_time_coverage": "[]",
        "estimated_min_coverage": float("nan"),
        "lower_bound_min": float("nan"),
        "mean_ess": float("nan"),
        "minimum_ess": float("nan"),
        "median_policy_kl": float("nan"),
        "maximum_policy_ratio": float("nan"),
        "prediction_set_mean_score": float(scores.mean().item()),
        "target_policy_trajectories": 0,
        "oracle_evaluation_trajectories": 0,
        "score_mean": float("nan"),
    }
    if radius is None:
        return record
    if estimates is None:
        coverage = (scores <= radius).float().mean(dim=0)
        record["estimated_min_coverage"] = float(coverage.min().item())
    else:
        nearest = torch.argmin((torch.as_tensor(radius, device=estimates.device) - q_grid.to(estimates)).abs())
        index = int(nearest.item())
        record["estimated_min_coverage"] = float(estimates[index].amin().item())
        if certificate is not None:
            record["lower_bound_min"] = float(certificate.lower_bounds[index].amin().item())
        if ess is not None:
            record["mean_ess"] = float(ess[index].mean().item())
            record["minimum_ess"] = float(ess[index].min().item())
    states = batch.current_states().reshape(-1, batch.state_dim).to(next(policy.outcome_model.parameters()).device)
    pi = policy.probabilities(states, radius)
    mu = logging_policy.probabilities(states)
    kl = (pi * (pi.clamp_min(1e-12).log() - mu.clamp_min(1e-12).log())).sum(dim=1)
    record["median_policy_kl"] = float(kl.median().item())
    record["maximum_policy_ratio"] = float((pi / mu.clamp_min(1e-12)).max().item())
    record.update(
        _logged_descriptive_metrics(
            batch,
            radius=float(radius),
            policy=policy,
        )
    )
    return record
