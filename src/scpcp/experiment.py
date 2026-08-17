"""End-to-end per-step SC-PCP experiment orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from scpcp.baselines import (
    OnlineBaselineResult,
    aci_style_controller,
    finite_depth_mfcs_selection,
    multidim_spci_style_controller,
    prc_profile_scale,
    standard_cp_stagewise_radii,
)
from scpcp.behavior import fit_behavior_policy
from scpcp.certification import (
    CertificationResult,
    exact_tabular_ordered_pointwise_l1_lower_bounds,
    ordered_pointwise_bootstrap_lower_bounds,
    ordered_pointwise_ht_lower_bounds,
)
from scpcp.config import ExperimentConfig
from scpcp.cot import (
    cot_state_action_weights,
    exact_tabular_cot_l1_error_bound,
    fit_cot,
)
from scpcp.coverage import (
    candidate_radius_schedules,
    diagonal_coverage_estimates,
    effective_sample_sizes,
    estimate_oracle_diagonal,
    per_step_oracle_metrics,
    profiled_local_scale_grid,
    profiled_scale_grid,
    self_normalized_diagonal_coverage_estimates,
    stage_score_profile,
    stage_score_quantiles,
    transport_refined_stage_profile,
    weighted_stage_score_quantiles,
)
from scpcp.data import DataSplits, TrajectoryBatch, concatenate_trajectories, patient_level_splits
from scpcp.outcome_model import fit_outcome_model
from scpcp.policy import BehaviorAnchoredPolicy
from scpcp.scores import fit_conformal_region, score_batch
from scpcp.selection import (
    RadiusSelection,
    select_ordered_certified_radius,
    select_ordered_lcb_radius,
)
from scpcp.simulator import (
    EmpiricalTransitionEnvironment,
    SyntheticBehaviorPolicy,
    SyntheticTreatmentEnvironment,
    TailShiftTreatmentEnvironment,
    TabularBehaviorPolicy,
    TabularTreatmentEnvironment,
    rollout,
)


SCPCP_METHOD = "SC-PCP"


@dataclass(frozen=True)
class SeedResult:
    seed: int
    device: str
    records: list[dict[str, Any]]
    surfaces: dict[str, Tensor]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _Task:
    environment: object | None
    splits: DataSplits
    n_actions: int
    logging_policy: object | None
    name: str
    policy_config: object
    static_indices: tuple[int, ...] = ()
    action_mapping: dict[int, int] | None = None
    state_feature_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RefinedScheduleFamily:
    """D_COT-measurable schedule artifacts frozen before certification."""

    initial_quantiles: Tensor
    initial_profile: Tensor
    baseline_scale_grid: Tensor
    profile: Tensor
    scale_grid: Tensor
    anchor_scale: Tensor
    applied_log_correction: Tensor
    fold_initial_quantiles: Tensor
    fold_transported_quantiles: Tensor
    fold_effective_sizes: Tensor
    fold_refinement_weights: Tensor
    fold_cap_hit_rates: Tensor


def run_seed(config: ExperimentConfig, *, seed: int, device: str) -> SeedResult:
    """Run the sole ordered-IUT SC-PCP method and its five baselines."""

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
    if splits.environment is not None:
        task = replace(
            task,
            environment=EmpiricalTransitionEnvironment(
                splits.environment,
                n_actions=task.n_actions,
                neighbors=config.data.empirical_neighbors,
                bandwidth=config.data.empirical_bandwidth,
                embedding_dim=config.data.empirical_embedding_dim,
                static_indices=task.static_indices,
                history_length=(
                    config.model.history_length
                    if config.model.architecture == "gru"
                    else 1
                ),
                outcome_model=outcome_model,
            ),
        )
    region = fit_conformal_region(outcome_model)
    behavior_model = None
    if task.logging_policy is None:
        behavior_training = (
            splits.predictor
            if splits.behavior is None
            else concatenate_trajectories(splits.predictor, splits.behavior)
        )
        behavior_model = fit_behavior_policy(
            behavior_training,
            n_actions=task.n_actions,
            model_config=config.model,
            policy_config=task.policy_config,
            device=device,
            seed=seed + 2,
            static_indices=task.static_indices,
            decision_time_index=(
                task.state_feature_names.index("decision_time")
                if "decision_time" in task.state_feature_names
                else None
            ),
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
    logging_policy = task.logging_policy if task.logging_policy is not None else behavior_model
    if logging_policy is None:
        raise RuntimeError("the OPE denominator is unavailable")

    outcome_sd = _training_outcome_sd(splits.predictor)
    cot_scores = score_batch(
        region,
        splits.cot.current_states(),
        splits.cot.actions,
        splits.cot.outcomes,
    )
    schedule_family = _fit_transport_refined_schedule_family(
        splits.cot,
        cot_scores,
        policy=policy,
        logging_policy=logging_policy,
        outcome_model=outcome_model,
        config=config,
        device=device,
        seed=seed,
    )
    initial_stage_profile = schedule_family.initial_profile
    baseline_scale_grid = schedule_family.baseline_scale_grid
    stage_profile = schedule_family.profile
    scale_grid = schedule_family.scale_grid
    candidate_radii = candidate_radius_schedules(scale_grid, stage_profile)
    fitted_cot = fit_cot(
        splits.cot,
        q_grid=scale_grid,
        stage_profile=stage_profile,
        target_policy=policy,
        logging_policy=logging_policy,
        outcome_model=outcome_model,
        config=config.cot,
        device=device,
        seed=seed + 3,
    )
    exact_profiled_l1_error = None
    if task.name == "tabular" and hasattr(task.environment, "exact_state_ratios"):
        exact_profiled_l1_error = exact_tabular_cot_l1_error_bound(
            fitted_cot,
            task.environment,
            q_grid=scale_grid,
            target_policy=policy,
            logging_policy=logging_policy,
            weight_cap=config.cot.weight_cap,
        )
    # The profile, candidate grid, and final transport learner are now frozen.
    # D_cert is first touched only after this point.
    cert_scores = score_batch(
        region,
        splits.certification.current_states(),
        splits.certification.actions,
        splits.certification.outcomes,
    )
    cot_weights, cot_weight_diagnostics = cot_state_action_weights(
        fitted_cot,
        splits.certification,
        q_grid=scale_grid,
        target_policy=policy,
        logging_policy=logging_policy,
        weight_cap=config.cot.weight_cap,
    )
    cot_ht_diagonal = diagonal_coverage_estimates(
        cot_weights,
        cert_scores.to(cot_weights),
        candidate_radii.to(cot_weights),
    )
    cot_diagonal = self_normalized_diagonal_coverage_estimates(
        cot_weights,
        cert_scores.to(cot_weights),
        candidate_radii.to(cot_weights),
    )
    candidate_widths = _estimated_candidate_normalized_widths(
        outcome_model,
        splits.certification,
        cot_weights,
        candidate_radii,
        outcome_sd,
    )
    if exact_profiled_l1_error is not None:
        cot_certificate = exact_tabular_ordered_pointwise_l1_lower_bounds(
            cot_ht_diagonal,
            n_trajectories=splits.certification.n,
            weight_cap=config.cot.weight_cap,
            exact_l1_error_bound=exact_profiled_l1_error,
            delta=config.certification.delta,
            cluster_ids=splits.certification.patient_ids,
        )
        cot_selection = select_ordered_certified_radius(
            scale_grid,
            cot_certificate,
            alpha=config.certification.alpha,
            widths=candidate_widths,
        )
        selection_evidence = "exact_finite_mdp_transport_audit"
    elif config.certification.ratio_bound_source == "declared":
        cot_certificate = ordered_pointwise_ht_lower_bounds(
            cot_ht_diagonal,
            n_trajectories=splits.certification.n,
            weight_cap=config.cot.weight_cap,
            config=config.certification,
            cluster_ids=splits.certification.patient_ids,
        )
        cot_selection = select_ordered_certified_radius(
            scale_grid,
            cot_certificate,
            alpha=config.certification.alpha,
            widths=candidate_widths,
        )
        selection_evidence = "declared_simultaneous_transport_bound"
    else:
        cot_certificate = ordered_pointwise_bootstrap_lower_bounds(
            cot_weights,
            cert_scores.to(cot_weights),
            candidate_radii.to(cot_weights),
            lower_tail=config.certification.delta,
            n_resamples=config.certification.practical_bootstrap_resamples,
            seed=seed + 31_337,
            cluster_ids=splits.certification.patient_ids,
        )
        cot_selection = select_ordered_lcb_radius(
            scale_grid,
            cot_certificate,
            alpha=config.certification.alpha,
            widths=candidate_widths,
            status="PRACTICAL_CLUSTER_ORDERED_IUT_LCB",
        )
        selection_evidence = "practical_patient_cluster_bootstrap"

    baseline_batch = concatenate_trajectories(splits.cot, splits.certification)
    baseline_scores = torch.cat((cot_scores, cert_scores), dim=0)
    standard_radii = standard_cp_stagewise_radii(
        baseline_scores,
        config.certification.alpha,
    )
    mfcs_method = f"MFCS task-adapted (depth={config.baselines.mfcs_depth})"
    mfcs_selection, _ = finite_depth_mfcs_selection(
        baseline_batch.to(device),
        baseline_scores.to(device),
        q_grid=baseline_scale_grid.to(device),
        stage_profile=initial_stage_profile.to(device),
        target_policy=policy,
        logging_policy=logging_policy,
        depth=config.baselines.mfcs_depth,
        alpha=config.certification.alpha,
        weight_cap=config.cot.weight_cap,
    )

    adaptation_stream = _paper_seed(seed, 700_001)
    aci_adaptation_seed = _paper_seed(adaptation_stream, 101)
    multidim_adaptation_seed = _paper_seed(adaptation_stream, 211)
    prc_adaptation_seed = _paper_seed(adaptation_stream, 307)
    aci = aci_style_controller(
        task.environment,
        policy,
        region,
        baseline_scores,
        alpha=config.certification.alpha,
        gamma=config.baselines.aci_gamma,
        rounds=config.baselines.online_rounds,
        total_rollouts=config.samples.online_rollouts,
        horizon=config.horizon,
        seed=aci_adaptation_seed,
        device=device,
    )
    multidim = multidim_spci_style_controller(
        task.environment,
        policy,
        region,
        baseline_scores,
        alpha=config.certification.alpha,
        rounds=config.baselines.online_rounds,
        total_rollouts=config.samples.online_rollouts,
        horizon=config.horizon,
        seed=multidim_adaptation_seed,
        device=device,
        residual_window=config.baselines.multidim_buffer,
    )
    initial_prc_scale = float(
        (standard_radii / initial_stage_profile.to(standard_radii)).max().item()
    )
    prc = prc_profile_scale(
        task.environment,
        policy,
        region,
        initial_prc_scale,
        baseline_scale_grid,
        initial_stage_profile,
        alpha=config.certification.alpha,
        delta=config.certification.delta,
        rounds=config.baselines.online_rounds,
        total_rollouts=config.samples.online_rollouts,
        horizon=config.horizon,
        seed=prc_adaptation_seed,
        device=device,
        maximum_step=config.baselines.prc_maximum_step,
    )

    evaluation_seed = _paper_seed(seed, 900_001)
    records = [
        _evaluate_radius_method(
            "Standard CP",
            standard_radii,
            task,
            policy,
            region,
            config,
            evaluation_seed,
            device,
            outcome_sd=outcome_sd,
        ),
        _evaluate_radius_method(
            mfcs_method,
            None
            if mfcs_selection.radius is None
            else mfcs_selection.radius * initial_stage_profile,
            task,
            policy,
            region,
            config,
            evaluation_seed,
            device,
            selection=mfcs_selection,
            selected_scale=mfcs_selection.radius,
            stage_profile=initial_stage_profile,
            outcome_sd=outcome_sd,
        ),
        _evaluate_stagewise_method(
            f"ACI stagewise adaptation (gamma={config.baselines.aci_gamma:g})",
            aci,
            task,
            policy,
            region,
            config,
            evaluation_seed,
            device,
            outcome_sd=outcome_sd,
        ),
        _evaluate_stagewise_method(
            f"MultiDimSPCI task-adapted (buffer={config.baselines.multidim_buffer})",
            multidim,
            task,
            policy,
            region,
            config,
            evaluation_seed,
            device,
            outcome_sd=outcome_sd,
        ),
        _evaluate_stagewise_method(
            "PRC grid-adapted",
            prc,
            task,
            policy,
            region,
            config,
            evaluation_seed,
            device,
            outcome_sd=outcome_sd,
            stage_profile=initial_stage_profile,
        ),
        _evaluate_radius_method(
            SCPCP_METHOD,
            None
            if cot_selection.radius is None
            else cot_selection.radius * stage_profile,
            task,
            policy,
            region,
            config,
            evaluation_seed,
            device,
            selection=cot_selection,
            certificate=cot_certificate,
            selected_scale=cot_selection.radius,
            stage_profile=stage_profile,
            outcome_sd=outcome_sd,
        ),
    ]

    surfaces = {
        "q_grid": scale_grid,
        "scale_grid": scale_grid,
        "stage_profile": stage_profile,
        "initial_stage_quantiles": schedule_family.initial_quantiles,
        "initial_stage_profile": initial_stage_profile,
        "baseline_scale_grid": baseline_scale_grid,
        "profile_anchor_scale": schedule_family.anchor_scale,
        "profile_applied_log_correction": schedule_family.applied_log_correction,
        "profile_fold_initial_quantiles": schedule_family.fold_initial_quantiles,
        "profile_fold_transported_quantiles": schedule_family.fold_transported_quantiles,
        "profile_fold_effective_sizes": schedule_family.fold_effective_sizes,
        "profile_fold_refinement_weights": schedule_family.fold_refinement_weights,
        "profile_fold_cap_hit_rates": schedule_family.fold_cap_hit_rates,
        "candidate_radii": candidate_radii,
        "cot_ht_diagonal": cot_ht_diagonal,
        "cot_diagonal": cot_diagonal,
        "cot_lower_bounds": cot_certificate.lower_bounds,
        "estimated_candidate_widths": candidate_widths,
    }
    if cot_selection.radius is not None:
        surfaces["scpcp_selected_scale"] = torch.tensor(cot_selection.radius)
        surfaces["scpcp_selected_radii"] = cot_selection.radius * stage_profile
    if exact_profiled_l1_error is not None:
        surfaces["exact_profiled_cot_l1_error"] = exact_profiled_l1_error
    if config.paper.save_mechanism_diagonal and seed == config.paper.mechanism_seed:
        if task.name not in {"synthetic", "tabular"}:
            raise ValueError("the mechanism diagonal is available only in known environments")
        surfaces["oracle_diagonal"] = estimate_oracle_diagonal(
            task.environment,
            policy,
            region,
            q_grid=candidate_radii,
            horizon=config.horizon,
            n_rollouts=config.samples.oracle_surface_rollouts,
            seed=_paper_seed(seed, 1_100_001),
            device=device,
        )
    diagnostics = {
        "dataset": task.name,
        "protocol": "transport_refined_four_rq_paper_protocol",
        "baseline_scope": "task_adapted_common_prediction_set_interface",
        "matched_evaluation_random_stream": True,
        "adaptation_seeds": {
            "aci": aci_adaptation_seed,
            "multidim_spci": multidim_adaptation_seed,
            "prc": prc_adaptation_seed,
        },
        "evaluation_seed": evaluation_seed,
        "training_outcome_sd": [float(value) for value in outcome_sd.tolist()],
        "cot": fitted_cot.diagnostics,
        "cot_cap_hit_rate": float(cot_weight_diagnostics.cap_hit_rate.mean().item()),
        "selection_evidence": selection_evidence,
        "profile_learning": "patient_crossfit_transport_refinement",
        "initial_stage_profile": [
            float(value) for value in initial_stage_profile.tolist()
        ],
        "stage_profile": [float(value) for value in stage_profile.tolist()],
        "profile_anchor_scale": float(schedule_family.anchor_scale.item()),
        "profile_applied_log_correction": [
            float(value) for value in schedule_family.applied_log_correction.tolist()
        ],
        "profile_fold_minimum_effective_size": [
            float(value)
            for value in schedule_family.fold_effective_sizes.min(dim=1).values.tolist()
        ],
        "profile_fold_mean_cap_hit_rate": [
            float(value) for value in schedule_family.fold_cap_hit_rates.mean(dim=1).tolist()
        ],
        "profile_gated_fold_stage_count": int(
            (schedule_family.fold_refinement_weights == 0.0).sum().item()
        ),
        "baseline_candidate_count": len(baseline_scale_grid),
        "ordered_candidate_count": len(scale_grid),
        "ordered_certified_indices": list(cot_selection.certified_indices),
        "ordered_stopped_index": cot_selection.stopped_index,
        "split_sizes": {
            "pred": splits.predictor.n,
            "beh": 0 if splits.behavior is None else splits.behavior.n,
            "cot": splits.cot.n,
            "cert": splits.certification.n,
            "env": 0 if splits.environment is None else splits.environment.n,
        },
    }
    return SeedResult(
        seed=seed,
        device=device,
        records=records,
        surfaces=surfaces,
        diagnostics=diagnostics,
    )


def _fit_transport_refined_schedule_family(
    batch: TrajectoryBatch,
    scores: Tensor,
    *,
    policy: object,
    logging_policy: object,
    outcome_model: object,
    config: ExperimentConfig,
    device: str,
    seed: int,
) -> _RefinedScheduleFamily:
    """Learn the SC-PCP schedule shape using only patient-crossfit D_COT data."""

    alpha = config.certification.alpha
    initial_quantiles = stage_score_quantiles(scores, alpha=alpha)
    initial_profile = stage_score_profile(scores, alpha=alpha)
    baseline_scale_grid = profiled_scale_grid(
        scores,
        initial_profile,
        size=config.q_grid_size,
        lower_quantile=config.q_quantile_min,
        upper_quantile=config.q_quantile_max,
    )

    fold_initial_quantiles = []
    fold_transported_quantiles = []
    fold_effective_sizes = []
    fold_cap_hit_rates = []
    folds = _patient_crossfit_indices(
        batch,
        folds=config.profile.refinement_folds,
        seed=_paper_seed(seed, 300_001),
    )
    for fold, (train_indices, held_indices) in enumerate(folds):
        train_batch = batch.subset(train_indices)
        held_batch = batch.subset(held_indices)
        train_scores = scores[train_indices.to(scores.device)]
        held_scores = scores[held_indices.to(scores.device)]

        fold_quantiles = stage_score_quantiles(train_scores, alpha=alpha)
        fold_anchor = fold_quantiles.log().mean().exp()
        fold_profile = fold_quantiles / fold_anchor
        # A one-schedule pilot is enough to estimate the transported shape and
        # is much cheaper than fitting another full K-candidate family.
        pilot_seed = _paper_seed(seed, 310_001 + fold)
        resolved_device = torch.device(device)
        rng_devices = (
            []
            if resolved_device.type != "cuda"
            else [
                torch.cuda.current_device()
                if resolved_device.index is None
                else resolved_device.index
            ]
        )
        # Pilot fitting has a private RNG stream and cannot perturb the final
        # COT learner or any baseline's random stream.
        with torch.random.fork_rng(devices=rng_devices):
            torch.random.default_generator.manual_seed(pilot_seed)
            if rng_devices:
                with torch.cuda.device(rng_devices[0]):
                    torch.cuda.manual_seed(pilot_seed)
            pilot_cot = fit_cot(
                train_batch,
                q_grid=fold_anchor[None],
                stage_profile=fold_profile,
                target_policy=policy,
                logging_policy=logging_policy,
                outcome_model=outcome_model,
                config=config.cot,
                device=device,
                seed=pilot_seed,
            )
        held_weights, held_diagnostics = cot_state_action_weights(
            pilot_cot,
            held_batch,
            q_grid=fold_anchor[None],
            target_policy=policy,
            logging_policy=logging_policy,
            weight_cap=config.cot.weight_cap,
        )
        transported = weighted_stage_score_quantiles(
            held_scores.to(held_weights),
            held_weights[:, :, 0],
            alpha=alpha,
        )
        effective_sizes = effective_sample_sizes(
            held_weights,
            cluster_ids=held_batch.patient_ids,
        )[0]

        fold_initial_quantiles.append(fold_quantiles.to(scores))
        fold_transported_quantiles.append(transported.to(scores))
        fold_effective_sizes.append(effective_sizes.to(scores))
        fold_cap_hit_rates.append(held_diagnostics.cap_hit_rate[0].to(scores))
        del pilot_cot, held_weights

    stacked_initial = torch.stack(fold_initial_quantiles)
    stacked_transported = torch.stack(fold_transported_quantiles)
    stacked_effective_sizes = torch.stack(fold_effective_sizes)
    stacked_cap_hit_rates = torch.stack(fold_cap_hit_rates)
    refinement_weights = torch.where(
        stacked_cap_hit_rates <= config.profile.maximum_cap_hit_rate,
        stacked_effective_sizes,
        torch.zeros_like(stacked_effective_sizes),
    )
    refined_profile, applied_log_correction = transport_refined_stage_profile(
        initial_quantiles,
        stacked_initial,
        stacked_transported,
        refinement_weights,
        refinement_strength=config.profile.refinement_strength,
        maximum_profile_ratio=config.profile.maximum_profile_ratio,
        minimum_effective_size=config.profile.minimum_effective_size,
    )
    refined_quantiles = initial_quantiles * applied_log_correction.exp()
    anchor_scale = refined_quantiles.log().mean().exp()
    scale_grid = profiled_local_scale_grid(
        scores,
        refined_profile,
        size=config.q_grid_size,
        lower_quantile=config.q_quantile_min,
        upper_quantile=config.q_quantile_max,
        anchor_scale=anchor_scale,
        focus_fraction=config.profile.grid_focus_fraction,
        focus_radius=config.profile.grid_focus_radius,
    )
    return _RefinedScheduleFamily(
        initial_quantiles=initial_quantiles,
        initial_profile=initial_profile,
        baseline_scale_grid=baseline_scale_grid,
        profile=refined_profile,
        scale_grid=scale_grid,
        anchor_scale=anchor_scale,
        applied_log_correction=applied_log_correction,
        fold_initial_quantiles=stacked_initial,
        fold_transported_quantiles=stacked_transported,
        fold_effective_sizes=stacked_effective_sizes,
        fold_refinement_weights=refinement_weights,
        fold_cap_hit_rates=stacked_cap_hit_rates,
    )


def _patient_crossfit_indices(
    batch: TrajectoryBatch,
    *,
    folds: int,
    seed: int,
) -> tuple[tuple[Tensor, Tensor], ...]:
    """Return deterministic train/held row indices with patient-level isolation."""

    patient_ids = batch.patient_ids.detach().cpu()
    unique_patients = torch.unique(patient_ids, sorted=True)
    if folds < 2 or len(unique_patients) < folds:
        raise ValueError("patient cross-fitting requires at least one patient per fold")
    generator = torch.Generator().manual_seed(seed)
    shuffled = unique_patients[
        torch.randperm(len(unique_patients), generator=generator)
    ]
    device = batch.patient_ids.device
    indices = []
    for held_patients in torch.tensor_split(shuffled, folds):
        held_mask = torch.isin(patient_ids, held_patients)
        held = held_mask.nonzero().squeeze(1).to(device)
        train = (~held_mask).nonzero().squeeze(1).to(device)
        indices.append((train, held))
    return tuple(indices)


def _training_outcome_sd(batch: TrajectoryBatch) -> Tensor:
    """Outcome normalization scale computed from D_pred only."""

    flat = batch.outcomes.reshape(-1, batch.outcome_dim).float()
    return flat.std(dim=0, unbiased=True).clamp_min(1e-6)


@torch.no_grad()
def _estimated_candidate_normalized_widths(
    outcome_model: object,
    batch: TrajectoryBatch,
    weights: Tensor,
    candidate_radii: Tensor,
    outcome_sd: Tensor,
) -> Tensor:
    """Estimate target-policy normalized width for every frozen schedule."""

    if candidate_radii.ndim != 2 or candidate_radii.shape[1] != batch.horizon:
        raise ValueError("candidate_radii must have shape [K,T]")
    if weights.shape != (batch.n, batch.horizon, len(candidate_radii)):
        raise ValueError("weights must have shape [N,T,K]")
    predictor = outcome_model.outcome_model if hasattr(outcome_model, "outcome_model") else outcome_model
    device = weights.device
    resolved = batch.to(device)
    states = resolved.current_states().reshape(-1, batch.state_dim)
    actions = resolved.actions.reshape(-1)
    _, scales = predictor(states, actions)
    scales = scales.reshape(batch.n, batch.horizon, -1)
    normalization = outcome_sd.to(scales).clamp_min(1e-6)
    base_width = (2.0 * scales / normalization[None, None, :]).mean(dim=2)
    radii_by_time = candidate_radii.to(weights).transpose(0, 1)
    logged_width = base_width[:, :, None] * radii_by_time[None, :, :]
    weighted = (weights * logged_width).sum(dim=0)
    target_width_by_time = weighted / weights.sum(dim=0).clamp_min(1e-12)
    return target_width_by_time.mean(dim=0)


def _paper_seed(seed: int, stream: int) -> int:
    """Stable disjoint RNG streams shared across methods within a split seed."""

    return int((1_000_003 * seed + stream) % (2**31 - 1))


def _prepare_task(config: ExperimentConfig, *, seed: int, device: str) -> _Task:
    if config.data.dataset == "synthetic":
        environment = (
            TailShiftTreatmentEnvironment(config.synthetic)
            if config.synthetic.scenario == "tail_shift"
            else SyntheticTreatmentEnvironment(config.synthetic)
        )
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
    splits = patient_level_splits(
        logged,
        seed=seed,
        include_environment=True,
        include_behavior=False,
    )
    if splits.environment is None:
        raise RuntimeError("clinical Track A requires D_env")
    return _Task(
        None,
        splits,
        n_actions,
        None,
        config.data.dataset,
        replace(config.policy, action_costs=action_costs),
        static_indices=static_indices,
        action_mapping=action_mapping,
        state_feature_names=state_feature_names,
    )


def _evaluate_radius_method(
    name: str,
    radius: float | Tensor | None,
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
    outcome_sd: Tensor | None = None,
    selected_scale: float | None = None,
    stage_profile: Tensor | None = None,
) -> dict[str, Any]:
    if radius is None:
        return _unavailable_record(
            name,
            selection,
            certificate,
            information_regime=information_regime,
            target_deployments=target_deployments,
            selected_scale=selected_scale,
            stage_profile=stage_profile,
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
        outcome_sd,
        selected_scale,
        stage_profile,
    )


def _evaluate_stagewise_method(
    name: str,
    adaptation: OnlineBaselineResult,
    task: _Task,
    policy: BehaviorAnchoredPolicy,
    outcome_model: object,
    config: ExperimentConfig,
    seed: int,
    device: str,
    *,
    outcome_sd: Tensor | None = None,
    stage_profile: Tensor | None = None,
) -> dict[str, Any]:
    radii = adaptation.radius_by_time
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
    record = _deployment_record(
        name,
        radii.cpu(),
        coverage,
        deployed,
        scores,
        config,
        policy,
        outcome_model,
        None,
        None,
        adaptation.target_deployments,
        outcome_sd=outcome_sd,
        selected_scale=adaptation.selected_scale,
        stage_profile=stage_profile,
    )
    per_time = adaptation.adaptation_per_time_coverage
    adaptation_target = 1.0 - config.certification.alpha
    record.update(
        {
            "adaptation_trajectories": adaptation.target_deployments,
            "adaptation_rounds": adaptation.rounds,
            "adaptation_target_coverage": adaptation_target,
            "adaptation_empirical_target_met": bool(
                float(per_time.min().item()) >= adaptation_target - 1e-7
            ),
            "adaptation_worst_coverage": float(per_time.min().item()),
            "adaptation_average_coverage": float(per_time.mean().item()),
            "adaptation_pathwise_coverage": adaptation.adaptation_pathwise_coverage,
            "adaptation_per_time_coverage": json.dumps([float(value) for value in per_time.tolist()]),
            "adaptation_round_worst_coverage": json.dumps(list(adaptation.adaptation_round_worst_coverage)),
        }
    )
    return record


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
        "average_normalized_width": missing,
        "per_time_normalized_width": "[]",
        "clinical_cost": missing,
        "clinical_utility": missing,
        "logged_descriptive_mean_log_volume": missing,
        "logged_descriptive_median_volume": missing,
        "logged_descriptive_clinical_cost": missing,
        "logged_descriptive_clinical_utility": missing,
        "logged_descriptive_per_time_clinical_cost": "[]",
        "logged_state_model_estimated_clinical_cost": missing,
        "logged_state_model_estimated_clinical_utility": missing,
        "estimated_min_coverage": missing,
        "lower_bound_min": missing,
        "mean_ess": missing,
        "minimum_ess": missing,
        "median_policy_kl": missing,
        "maximum_policy_ratio": missing,
        "prediction_set_mean_score": missing,
        "certified": False,
        "selection_available": False,
        "adaptation_trajectories": 0,
        "adaptation_rounds": 0,
        "adaptation_target_coverage": missing,
        "adaptation_empirical_target_met": missing,
        "adaptation_worst_coverage": missing,
        "adaptation_average_coverage": missing,
        "adaptation_pathwise_coverage": missing,
        "adaptation_per_time_coverage": "[]",
        "adaptation_round_worst_coverage": "[]",
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
    outcome_sd: Tensor | None = None,
    selected_scale: float | None = None,
    stage_profile: Tensor | None = None,
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
    normalized_width = float("nan")
    per_time_normalized_width = torch.full(
        (deployed.horizon,),
        float("nan"),
        device=scales.device,
    )
    if outcome_sd is not None:
        normalization = outcome_sd.to(scales).clamp_min(1e-6)
        normalized_coordinate_width = (
            2.0 * step_radius[:, None] * scales / normalization[None, :]
        )
        normalized_width = float(normalized_coordinate_width.mean().item())
        per_time_normalized_width = (
            normalized_coordinate_width.mean(dim=1)
            .reshape(deployed.n, deployed.horizon)
            .mean(dim=0)
        )
    selected_index = None if selection is None else selection.index
    has_selected_certificate = certificate is not None and selected_index is not None
    estimated_min = (
        float(certificate.estimates[selected_index].amin().item())
        if has_selected_certificate
        else float("nan")
    )
    lower_bound_min = (
        float(certificate.lower_bounds[selected_index].amin().item())
        if has_selected_certificate
        else float("nan")
    )
    formally_certified = bool(
        has_selected_certificate
        and certificate is not None
        and certificate.formal
        and selection is not None
        and selection.status.startswith("CERTIFIED")
    )
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
        "selection_parameter": (
            "global_scale"
            if selected_scale is not None
            else "stagewise_radii"
            if isinstance(radius, Tensor)
            else "scalar_radius"
        ),
        "selected_q": float(radius) if not isinstance(radius, Tensor) else float(radius.mean().item()),
        "selected_scale": float("nan") if selected_scale is None else float(selected_scale),
        "stage_profile": (
            "[]"
            if stage_profile is None
            else json.dumps([float(value) for value in stage_profile.tolist()])
        ),
        "q_by_time": "" if not isinstance(radius, Tensor) else json.dumps([float(value) for value in radius.tolist()]),
        "selection_status": "FIXED" if selection is None else selection.status,
        "certificate_type": "" if certificate is None else certificate.label,
        "certificate_formal": False if certificate is None else certificate.formal,
        "selection_available": True,
        "worst_coverage": float(coverage.min().item()),
        "average_coverage": float(coverage.mean().item()),
        "pathwise_coverage": float(pathwise_coverage.item()),
        "worst_gap": float((1.0 - config.certification.alpha - coverage).clamp_min(0.0).max().item()),
        "per_time_coverage": json.dumps([float(value) for value in coverage.tolist()]),
        "mean_log_volume": float(log_volumes.mean().item()),
        "median_volume": float(volumes.median().item()),
        "average_normalized_width": normalized_width,
        "per_time_normalized_width": json.dumps(
            [float(value) for value in per_time_normalized_width.tolist()]
        ),
        "clinical_cost": float(realized_cost.item()),
        "clinical_utility": float((-realized_cost).item()),
        "target_policy_trajectories": target_deployments,
        "oracle_evaluation_trajectories": deployed.n,
        "score_mean": float(scores.mean().item()),
        "estimated_min_coverage": estimated_min,
        "lower_bound_min": lower_bound_min,
        "certified": formally_certified,
    }


def _unavailable_record(
    name: str,
    selection: RadiusSelection | None,
    certificate: CertificationResult | None,
    *,
    information_regime: str | None = None,
    target_deployments: int = 0,
    selected_scale: float | None = None,
    stage_profile: Tensor | None = None,
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
        "selection_parameter": "global_scale" if stage_profile is not None else "scalar_radius",
        "selected_q": float("nan"),
        "selected_scale": float("nan") if selected_scale is None else float(selected_scale),
        "stage_profile": (
            "[]"
            if stage_profile is None
            else json.dumps([float(value) for value in stage_profile.tolist()])
        ),
        "q_by_time": "",
        "selection_status": "UNCERTIFIED" if selection is None else selection.status,
        "certificate_type": "" if certificate is None else certificate.label,
        "certificate_formal": False if certificate is None else certificate.formal,
        "selection_available": False,
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
        "selection_available": radius is not None,
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
