"""Run the isolated, coverage-blind MIMIC-CXR v5 outcome-bridge repair.

The ``audit`` phase consumes no formal RNG.  Development and confirmation are
hard-locked until the frozen config contains an independent-audit GO
attestation.

Examples
--------
::

    conda run -n ucp python scripts/run_controlled_clinical_fidelity_v5_mimic_cxr.py audit
    conda run -n ucp python scripts/run_controlled_clinical_fidelity_v5_mimic_cxr.py development \
      --devices cuda:0,cuda:1 \
      --output-root results/work/controlled_clinical_fidelity_v5_mimic_cxr_development
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_controlled_clinical_extension as v2  # noqa: E402
import scripts.run_controlled_clinical_fidelity_v4 as v4  # noqa: E402
import scripts.run_controlled_six_method_benchmark as six  # noqa: E402
from scpcp.artifacts import experiment_tree_sha256  # noqa: E402
from scpcp.controlled_clinical_extension import (  # noqa: E402
    ControlledClinicalExtensionConfig,
    DatasetPreset,
    K0FidelityMetrics,
    empirical_ks,
    equal_sample_wasserstein_1,
)
from scpcp.controlled_clinical_fidelity_v4 import (  # noqa: E402
    load_fidelity_v4_config,
)
from scpcp.controlled_clinical_fidelity_v5_mimic_cxr import (  # noqa: E402
    CONFIRMATION_BASE_SET_SHA256,
    CONFIRMATION_BOOTSTRAP_SEED,
    CONFIRMATION_ID_SET_SHA256,
    CONFIRMATION_MAPPING_SHA256,
    CONFIRMATION_SEEDS,
    DATASET,
    DEVELOPMENT_ID_SET_SHA256,
    DEVELOPMENT_MAPPING_SHA256,
    DEVELOPMENT_MINIMUM_PASS_COUNT,
    DEVELOPMENT_SEEDS,
    K0_THRESHOLDS,
    PROTOCOL,
    REQUIRED_STRUCTURAL_PASS_COUNT,
    SELECTOR_VERSION,
    BridgeTheta,
    FidelityV5Config,
    IndependentAudit,
    build_cxr_environment,
    bridge_candidates,
    bridge_contract,
    independent_audit_attestation_sha256,
    load_fidelity_v5_config,
    normalized_seed_ratio,
    outcome_feature_groups,
    select_bridge_candidate,
    successor_clinical_features,
    summarize_candidate,
)
from scpcp.scores import score_batch  # noqa: E402


CONFIG_PATH = ROOT / "configs/controlled_clinical_fidelity_v5_mimic_cxr.yaml"
V2_CONFIG_PATH = ROOT / "configs/controlled_clinical_extension.yaml"
V4_CONFIG_PATH = ROOT / "configs/controlled_clinical_fidelity_v4.yaml"
DEVELOPMENT_ROOT = (
    ROOT / "results/work/controlled_clinical_fidelity_v5_mimic_cxr_development"
).resolve()
CONFIRMATION_ROOT = (
    ROOT / "results/work/controlled_clinical_fidelity_v5_mimic_cxr_confirmation"
).resolve()
PHASES = ("audit", "development", "confirmation")
FORBIDDEN_RESULT_PATH_TOKENS = ("science", "coverage", "width", "method_selection")
_OWN_RNG_DECLARATION_PATHS = {
    Path(__file__).resolve(),
    (ROOT / "src/scpcp/controlled_clinical_fidelity_v5_mimic_cxr.py").resolve(),
    CONFIG_PATH.resolve(),
    (ROOT / "scripts/run_controlled_clinical_extension.py").resolve(),
    (ROOT / "src/scpcp/controlled_clinical_extension.py").resolve(),
    V2_CONFIG_PATH.resolve(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_fidelity_v5_config(CONFIG_PATH)
    parent_binding = validate_parent_v4_bundles(config)
    development_audit = audit_development_reuse(config, parent_binding=parent_binding)
    confirmation_audit = audit_confirmation_rng(config)
    if args.phase == "audit":
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "status": "PREFLIGHT_READY_FOR_INDEPENDENT_AUDIT",
                    "parent_v4_binding": parent_binding,
                    "development_rng_reuse_audit": development_audit,
                    "confirmation_rng_audit": confirmation_audit,
                    "formal_launch_authorized": config.independent_audit.permits_formal_launch,
                    "formal_rng_consumed": False,
                    "coverage_generation_permitted": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    config.validate(require_audit_go=True)
    _validate_frozen_audit_snapshot(config, confirmation_audit)
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    _validate_devices(devices)
    if args.output_root is None:
        parser.error("formal phases require --output-root")
    output_root = args.output_root.resolve()
    if args.phase == "development":
        if args.development_root is not None:
            parser.error("development does not accept --development-root")
        if output_root != DEVELOPMENT_ROOT:
            parser.error(f"development output root is frozen to {DEVELOPMENT_ROOT}")
        run_development(
            output_root,
            config=config,
            devices=devices,
            parent_binding=parent_binding,
            development_audit=development_audit,
            confirmation_audit=confirmation_audit,
            resume=args.resume,
        )
    else:
        if args.development_root is None:
            parser.error("confirmation requires --development-root")
        development_root = args.development_root.resolve()
        if development_root != DEVELOPMENT_ROOT:
            parser.error(f"development root is frozen to {DEVELOPMENT_ROOT}")
        if output_root != CONFIRMATION_ROOT:
            parser.error(f"confirmation output root is frozen to {CONFIRMATION_ROOT}")
        run_confirmation(
            output_root,
            development_root=development_root,
            config=config,
            devices=devices,
            parent_binding=parent_binding,
            development_audit=development_audit,
            confirmation_audit=confirmation_audit,
            resume=args.resume,
        )
    print(output_root)


def run_development(
    output_root: Path,
    *,
    config: FidelityV5Config,
    devices: tuple[str, ...],
    parent_binding: Mapping[str, Any],
    development_audit: Mapping[str, Any],
    confirmation_audit: Mapping[str, Any],
    resume: bool,
) -> None:
    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    source_hash, source_snapshot = v4._active_source_contract()
    metadata = _root_metadata(
        phase="development",
        output_root=output_root,
        config=config,
        devices=devices,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        parent_binding=parent_binding,
        development_audit=development_audit,
        confirmation_audit=confirmation_audit,
    )
    _prepare_root(output_root, metadata, source_snapshot, resume=resume)
    if _complete_and_valid(output_root, metadata, config=config):
        return

    candidates = bridge_candidates()
    preset = replace(protocol.datasets[DATASET], seeds=config.development_seeds)
    rows = _run_seed_phase(
        output_root / "repair",
        phase="development_k0_only",
        preset=preset,
        devices=devices,
        candidates=candidates,
        worker=_development_worker,
        worker_arguments=(protocol,),
        source_hash=source_hash,
    )
    selection = _development_selection(rows)
    final = _development_final(selection)
    frozen = _frozen_settings(final, metadata)
    _write_json(output_root / "selection.json", selection)
    _write_json(output_root / "FINAL_STATUS.json", final)
    _write_json(output_root / "frozen_settings.json", frozen)
    _finalize_root(output_root, metadata, source_hash=source_hash, config=config)


def run_confirmation(
    output_root: Path,
    *,
    development_root: Path,
    config: FidelityV5Config,
    devices: tuple[str, ...],
    parent_binding: Mapping[str, Any],
    development_audit: Mapping[str, Any],
    confirmation_audit: Mapping[str, Any],
    resume: bool,
) -> None:
    development_binding, frozen = _verify_development_for_confirmation(
        development_root, config=config, parent_binding=parent_binding
    )
    source_hash, source_snapshot = v4._active_source_contract()
    if source_hash != frozen["development_source_tree_sha256"]:
        raise RuntimeError("source/config changed after the v5 bridge was selected")
    metadata = _root_metadata(
        phase="confirmation",
        output_root=output_root,
        config=config,
        devices=devices,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        parent_binding=parent_binding,
        development_audit=development_audit,
        confirmation_audit=confirmation_audit,
        development_binding=development_binding,
        frozen_settings=frozen,
    )
    _prepare_root(output_root, metadata, source_snapshot, resume=resume)
    if _complete_and_valid(output_root, metadata, config=config):
        return

    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    preset = replace(
        protocol.datasets[DATASET],
        seeds=config.confirmation_seeds,
        bootstrap_seed=config.confirmation_bootstrap_seed,
    )
    theta = _theta_from_dict(frozen["theta"])
    support_rows = _run_seed_phase(
        output_root / "support",
        phase="confirmation_support",
        preset=preset,
        devices=devices,
        candidates=(),
        worker=_confirmation_support_worker,
        worker_arguments=(protocol,),
        source_hash=source_hash,
    )
    support_count = sum(bool(row["passed"]) for row in support_rows)
    k0_rows: list[dict[str, Any]] = []
    if support_count >= DEVELOPMENT_MINIMUM_PASS_COUNT:
        k0_rows = _run_seed_phase(
            output_root / "k0_fidelity",
            phase="confirmation_k0",
            preset=preset,
            devices=devices,
            candidates=(theta,),
            worker=_confirmation_k0_worker,
            worker_arguments=(protocol,),
            source_hash=source_hash,
        )
    gate = _confirmation_gate(theta, support_rows, k0_rows)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "FINAL_STATUS.json", _confirmation_final(gate))
    _finalize_root(output_root, metadata, source_hash=source_hash, config=config)


def _development_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    candidates: tuple[BridgeTheta, ...],
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    candidate_rows = []
    for theta in candidates:
        context = _context_with_theta(base_context, theta)
        metrics, detail = _logging_mixture_fidelity_v5(
            context, seed=seed, protocol=protocol
        )
        metric_payload = asdict(metrics)
        ratio = normalized_seed_ratio(metric_payload)
        candidate_rows.append(
            {
                "theta": theta.to_dict(),
                "metrics": metric_payload,
                "passed": v2.k0_fidelity_passes(metrics, protocol.k0_fidelity_gate),
                "normalized_seed_ratio": ratio if math.isfinite(ratio) else None,
                "structural_failure_ratio_is_infinite": not math.isfinite(ratio),
                "systematic_replay": detail,
                "context_identity": _candidate_context_identity(
                    base_context, context.environment, theta
                ),
            }
        )
    return {
        "seed": seed,
        "dataset": DATASET,
        "phase": "development_k0_only",
        "candidate_count": len(candidates),
        "candidates": candidate_rows,
        "split_audit": v2._split_audit(base_context.splits),
        "coverage_generated": False,
        "information_opened": [
            "support",
            "k0_fidelity",
            "context_identity",
            "descriptive_diagnostics",
        ],
    }


def _confirmation_support_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    candidates: tuple[BridgeTheta, ...],
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    if candidates:
        raise RuntimeError("support must remain outcome/bridge blind")
    result = v2._support_worker(seed, preset, device, protocol)
    return {
        **result,
        "phase": "confirmation_support",
        "coverage_generated": False,
        "confirmation_label": "fresh_split_operational_gate",
    }


def _confirmation_k0_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    candidates: tuple[BridgeTheta, ...],
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    if len(candidates) != 1:
        raise RuntimeError("confirmation requires exactly one frozen bridge")
    theta = candidates[0]
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    context = _context_with_theta(base_context, theta)
    metrics, detail = _logging_mixture_fidelity_v5(
        context, seed=seed, protocol=protocol
    )
    metric_payload = asdict(metrics)
    ratio = normalized_seed_ratio(metric_payload)
    return {
        "seed": seed,
        "dataset": DATASET,
        "phase": "confirmation_k0",
        "theta": theta.to_dict(),
        "metrics": metric_payload,
        "passed": v2.k0_fidelity_passes(metrics, protocol.k0_fidelity_gate),
        "normalized_seed_ratio": ratio if math.isfinite(ratio) else None,
        "structural_failure_ratio_is_infinite": not math.isfinite(ratio),
        "systematic_replay": detail,
        "context_identity": _candidate_context_identity(
            base_context, context.environment, theta
        ),
        "split_audit": v2._split_audit(base_context.splits),
        "coverage_generated": False,
        "confirmation_label": "fresh_split_operational_gate",
        "independent_patient_confirmation_claimed": False,
    }


def _context_with_theta(
    base_context: v2.ExtensionContext,
    theta: BridgeTheta,
) -> v2.ExtensionContext:
    if base_context.config.data.dataset != DATASET or base_context.n_actions != 3:
        raise RuntimeError("v5 context is restricted to three-action MIMIC-CXR")
    environment_scores = score_batch(
        base_context.region,
        base_context.splits.environment.current_states(),
        base_context.splits.environment.actions,
        base_context.splits.environment.outcomes,
    )
    environment = build_cxr_environment(
        base_context.splits.environment,
        theta=theta,
        outcome_model=base_context.outcome_model,
        n_actions=base_context.n_actions,
        difficulty=v2._empirical_rank_by_stage(environment_scores),
        history_length=base_context.config.model.history_length,
        static_indices=base_context.static_indices,
        state_feature_names=base_context.state_feature_names,
    )
    return replace(base_context, environment=environment)


@torch.no_grad()
def _logging_mixture_fidelity_v5(
    context: v2.ExtensionContext,
    *,
    seed: int,
    protocol: ControlledClinicalExtensionConfig,
) -> tuple[K0FidelityMetrics, dict[str, Any]]:
    """Compute the unchanged aggregate K0 gate plus non-gating diagnostics."""

    fidelity = context.splits.fidelity
    replay_count = protocol.k0_fidelity_gate.systematic_replays
    if replay_count != 16:
        raise RuntimeError("v5 K0 systematic replay count must remain M=16")
    uniform_seed = v2.K0_UNIFORM_SEED_OFFSET + seed
    generator = torch.Generator(device="cpu").manual_seed(uniform_seed)
    base_uniform = torch.rand(
        (fidelity.horizon, fidelity.n),
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    offsets = (torch.arange(replay_count, dtype=torch.float64) + 0.5) / replay_count
    systematic_uniform = (
        base_uniform[:, :, None] + offsets[None, None, :]
    ).remainder(1.0)

    score_ks: list[float] = []
    residual_max_w1: list[float] = []
    successor_mean_w1: list[float] = []
    successor_q95_w1: list[float] = []
    active_counts: list[int] = []
    invariant_rows: list[dict[str, Any]] = []
    residual_by_outcome: list[list[float]] = []
    clinical_successor_mean_by_outcome: list[list[float]] = []
    clinical_successor_q95_by_outcome: list[list[float]] = []
    action_diagnostics: list[list[dict[str, Any]]] = []
    feature_groups = outcome_feature_groups()

    for stage in range(fidelity.horizon):
        state = fidelity.states[:, stage].to(context.action_coordinate.device)
        action = fidelity.actions[:, stage].to(state.device)
        true_outcome = fidelity.outcomes[:, stage].to(state.device)
        true_score_original = score_batch(
            context.region,
            state[:, None, :],
            action[:, None],
            true_outcome[:, None, :],
        ).flatten()
        mean, scale = context.outcome_model(state, action)
        replay_score_parts = []
        replay_residual_parts = []
        replay_representation_parts = []
        replay_clinical_parts = []
        invariant_parts = []
        for start, stop, uniform_chunk in v2._systematic_uniform_chunks(
            systematic_uniform[stage], chunk_size=v2.K0_PATIENT_CHUNK_SIZE
        ):
            state_chunk = state[start:stop]
            action_chunk = action[start:stop]
            repeated_state = state_chunk.repeat_interleave(replay_count, dim=0)
            repeated_action = action_chunk.repeat_interleave(replay_count, dim=0)
            replay_next, replay_outcome, _, replay_ess, replay_max = (
                context.environment.step_from_uniform(
                    repeated_state,
                    repeated_action,
                    uniform_chunk.to(state),
                    time=stage,
                    gamma=0.0,
                    action_coordinate=context.action_coordinate,
                )
            )
            replay_score_parts.append(
                score_batch(
                    context.region,
                    repeated_state[:, None, :],
                    repeated_action[:, None],
                    replay_outcome[:, None, :],
                ).flatten().cpu()
            )
            repeated_mean = mean[start:stop].repeat_interleave(replay_count, dim=0)
            repeated_scale = scale[start:stop].clamp_min(1e-6).repeat_interleave(
                replay_count, dim=0
            )
            replay_residual_parts.append(
                ((replay_outcome - repeated_mean) / repeated_scale).cpu()
            )
            replay_representation_parts.append(
                v2._representation(context.outcome_model, replay_next).to(torch.float64)
            )
            replay_frame = replay_next.reshape(
                len(replay_next),
                context.config.model.history_length,
                -1,
            )[:, -1]
            replay_clinical_parts.append(
                successor_clinical_features(
                    replay_frame, context.state_feature_names
                ).to(torch.float64).cpu()
            )
            invariant_parts.append(
                v2._raw_transition_invariants(
                    context,
                    state=repeated_state,
                    next_state=replay_next,
                    outcome=replay_outcome,
                    ess=replay_ess,
                    probability_max=replay_max,
                    stage=stage,
                )
            )

        replay_score = torch.cat(replay_score_parts)
        replay_residual = torch.cat(replay_residual_parts)
        replay_representation = torch.cat(replay_representation_parts)
        replay_clinical = torch.cat(replay_clinical_parts)
        true_score = true_score_original.cpu().repeat_interleave(replay_count)
        score_ks.append(empirical_ks(true_score, replay_score))

        true_residual = ((true_outcome - mean) / scale.clamp_min(1e-6)).cpu()
        coordinate_residual_w1 = equal_sample_wasserstein_1(
            true_residual.repeat_interleave(replay_count, dim=0), replay_residual
        )
        residual_max_w1.append(float(coordinate_residual_w1.max().item()))
        residual_by_outcome.append(
            [float(value) for value in coordinate_residual_w1.tolist()]
        )

        environment_true_representation = v2._representation(
            context.outcome_model,
            context.splits.environment.states[:, stage + 1],
        ).to(torch.float64)
        center = environment_true_representation.mean(dim=0)
        representation_scale = environment_true_representation.std(
            dim=0, unbiased=False
        )
        active = (
            representation_scale
            > protocol.k0_fidelity_gate.active_coordinate_sd_floor
        )
        active_counts.append(int(active.sum().item()))
        if active.any():
            true_representation = v2._representation(
                context.outcome_model, fidelity.states[:, stage + 1]
            ).to(torch.float64)
            true_standardized = (
                (true_representation - center) / representation_scale
            )[:, active].repeat_interleave(replay_count, dim=0)
            replay_standardized = (
                (replay_representation - center) / representation_scale
            )[:, active]
            successor_w1 = equal_sample_wasserstein_1(
                true_standardized, replay_standardized
            )
            successor_mean_w1.append(float(successor_w1.mean().item()))
            successor_q95_w1.append(
                float(
                    torch.quantile(
                        successor_w1, 0.95, interpolation="linear"
                    ).item()
                )
            )
        else:
            successor_mean_w1.append(0.0)
            successor_q95_w1.append(0.0)

        true_clinical = _clinical_successor_features(
            fidelity.states[:, stage + 1], context
        )
        environment_clinical = _clinical_successor_features(
            context.splits.environment.states[:, stage + 1], context
        )
        clinical_center = environment_clinical.mean(dim=0)
        clinical_scale = environment_clinical.std(dim=0, unbiased=False)
        clinical_active = (
            clinical_scale > protocol.k0_fidelity_gate.active_coordinate_sd_floor
        )
        true_clinical_standardized = (true_clinical - clinical_center) / clinical_scale.clamp_min(
            protocol.k0_fidelity_gate.active_coordinate_sd_floor
        )
        replay_clinical_standardized = (
            replay_clinical - clinical_center
        ) / clinical_scale.clamp_min(
            protocol.k0_fidelity_gate.active_coordinate_sd_floor
        )
        clinical_mean_row, clinical_q95_row = _outcome_group_successor_w1(
            true_clinical_standardized,
            replay_clinical_standardized,
            clinical_active,
            replay_count=replay_count,
            groups=feature_groups,
        )
        clinical_successor_mean_by_outcome.append(clinical_mean_row)
        clinical_successor_q95_by_outcome.append(clinical_q95_row)
        action_diagnostics.append(
            _action_stratified_diagnostics(
                action=action.cpu(),
                true_score=true_score_original.cpu(),
                replay_score=replay_score,
                true_residual=true_residual,
                replay_residual=replay_residual,
                true_clinical=true_clinical_standardized,
                replay_clinical=replay_clinical_standardized,
                clinical_active=clinical_active,
                replay_count=replay_count,
                groups=feature_groups,
            )
        )
        invariant_rows.append(v2._merge_invariant_rows(invariant_parts))

    structural = all(row["passed"] for row in invariant_rows) and all(
        count > 0 for count in active_counts
    )
    metrics = K0FidelityMetrics(
        maximum_score_ks=max(score_ks),
        maximum_signed_residual_w1=max(residual_max_w1),
        maximum_successor_mean_w1=max(successor_mean_w1),
        maximum_successor_q95_w1=max(successor_q95_w1),
        structural_invariants=structural,
    )
    detail = {
        "label": "logging-mixture one-step fidelity",
        "episode_weighted": True,
        "inference_unit": (
            "patient-disjoint episode query; M=16 quadrature, never 16N independent observations"
        ),
        "systematic_replays": replay_count,
        "patient_chunk_size": v2.K0_PATIENT_CHUNK_SIZE,
        "base_uniform_seed": uniform_seed,
        "base_uniform_shape": list(base_uniform.shape),
        "base_uniform_sha256": _tensor_sha256(base_uniform),
        "expansion_formula": "u[t,i,m]=(U[t,i]+(m+0.5)/16) mod 1",
        "flatten_order": "stage, patient, systematic_offset (offset fastest)",
        "expanded_uniform_sha256": _tensor_sha256(systematic_uniform),
        "score_ks_by_stage": score_ks,
        "signed_residual_max_w1_by_stage": residual_max_w1,
        "successor_mean_w1_by_stage": successor_mean_w1,
        "successor_q95_w1_by_stage": successor_q95_w1,
        "active_successor_coordinates_by_stage": active_counts,
        "raw_structural_invariants_by_stage": invariant_rows,
        "descriptive_diagnostics_non_gating": True,
        "score_ks_semantics": "one_scalar_per_stage",
        "signed_residual_w1_by_stage_outcome": residual_by_outcome,
        "clinical_successor_mean_w1_by_stage_outcome": (
            clinical_successor_mean_by_outcome
        ),
        "clinical_successor_q95_w1_by_stage_outcome": (
            clinical_successor_q95_by_outcome
        ),
        "action_stratified_by_stage": action_diagnostics,
        "aggregate_gate_unchanged": True,
    }
    return metrics, detail


def _clinical_successor_features(
    history_state: torch.Tensor,
    context: v2.ExtensionContext,
) -> torch.Tensor:
    frame = history_state.reshape(
        len(history_state), context.config.model.history_length, -1
    )[:, -1]
    return successor_clinical_features(frame, context.state_feature_names).to(
        torch.float64
    ).cpu()


def _outcome_group_successor_w1(
    true_values: torch.Tensor,
    replay_values: torch.Tensor,
    active: torch.Tensor,
    *,
    replay_count: int,
    groups: Sequence[Sequence[int]],
) -> tuple[list[float], list[float]]:
    means, q95s = [], []
    for group in groups:
        indices = [index for index in group if bool(active[index])]
        if not indices:
            means.append(0.0)
            q95s.append(0.0)
            continue
        coordinate_w1 = equal_sample_wasserstein_1(
            true_values[:, indices].repeat_interleave(replay_count, dim=0),
            replay_values[:, indices],
        )
        means.append(float(coordinate_w1.mean().item()))
        q95s.append(
            float(
                torch.quantile(
                    coordinate_w1, 0.95, interpolation="linear"
                ).item()
            )
        )
    return means, q95s


def _action_stratified_diagnostics(
    *,
    action: torch.Tensor,
    true_score: torch.Tensor,
    replay_score: torch.Tensor,
    true_residual: torch.Tensor,
    replay_residual: torch.Tensor,
    true_clinical: torch.Tensor,
    replay_clinical: torch.Tensor,
    clinical_active: torch.Tensor,
    replay_count: int,
    groups: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    replay_score_matrix = replay_score.reshape(len(action), replay_count)
    replay_residual_matrix = replay_residual.reshape(
        len(action), replay_count, -1
    )
    replay_clinical_matrix = replay_clinical.reshape(
        len(action), replay_count, -1
    )
    rows = []
    for action_value in range(3):
        selected = action.eq(action_value)
        query_count = int(selected.sum().item())
        if query_count == 0:
            rows.append(
                {
                    "action": action_value,
                    "available": False,
                    "query_count": 0,
                    "replay_count": 0,
                    "score_ks": None,
                    "signed_residual_w1_by_outcome": None,
                    "clinical_successor_mean_w1_by_outcome": None,
                    "clinical_successor_q95_w1_by_outcome": None,
                }
            )
            continue
        selected_replay_score = replay_score_matrix[selected].reshape(-1)
        selected_replay_residual = replay_residual_matrix[selected].reshape(
            -1, true_residual.shape[1]
        )
        selected_replay_clinical = replay_clinical_matrix[selected].reshape(
            -1, true_clinical.shape[1]
        )
        residual_w1 = equal_sample_wasserstein_1(
            true_residual[selected].repeat_interleave(replay_count, dim=0),
            selected_replay_residual,
        )
        successor_mean, successor_q95 = _outcome_group_successor_w1(
            true_clinical[selected],
            selected_replay_clinical,
            clinical_active,
            replay_count=replay_count,
            groups=groups,
        )
        rows.append(
            {
                "action": action_value,
                "available": True,
                "query_count": query_count,
                "replay_count": query_count * replay_count,
                "score_ks": empirical_ks(
                    true_score[selected].repeat_interleave(replay_count),
                    selected_replay_score,
                ),
                "signed_residual_w1_by_outcome": [
                    float(value) for value in residual_w1.tolist()
                ],
                "clinical_successor_mean_w1_by_outcome": successor_mean,
                "clinical_successor_q95_w1_by_outcome": successor_q95,
            }
        )
    return rows


def _candidate_context_identity(
    base_context: v2.ExtensionContext,
    environment: object,
    theta: BridgeTheta,
) -> dict[str, Any]:
    base = v2._context_identity(base_context)
    sizes = [
        [
            int(len(environment._libraries[(stage, action)][0]))
            for action in range(base_context.n_actions)
        ]
        for stage in range(environment.horizon)
    ]
    if not all(size < 10_000 for stage in sizes for size in stage):
        raise RuntimeError("C13 full-cell sentinel no longer exceeds donor cells")
    state_kernel = {
        "metric": environment.representation_geometry,
        "requested_neighbors": environment.neighbors,
        "actual_library_sizes_by_stage_action": sizes,
        "effective_neighbor_counts_by_stage_action": sizes,
        "full_cell_verified": True,
        "donor_weighting": environment.donor_weighting,
        "bandwidth": environment.bandwidth,
        "ridge": environment.ridge,
        "ridge_mode": environment.ridge_mode,
        "transition_mode": environment.transition_mode,
        "outcome_residual_mode": environment.outcome_residual_mode,
        "state_model_coefficient_sha256": [
            _tensor_sha256(model.coefficients.to(torch.float64))
            for model in environment._models
        ],
        "state_payload_sha256_by_stage_action": [
            [
                _tensor_sha256(
                    environment._libraries[(stage, action)][1].to(torch.float64)
                )
                for action in range(base_context.n_actions)
            ]
            for stage in range(environment.horizon)
        ],
    }
    bridge = (
        {
            **bridge_contract(theta.bridge_mode),
            "coefficient_count": 0,
            "coefficient_sha256": [],
            "joint_residual_libraries": [
                {
                    "stage": stage,
                    "action": action,
                    "rows": len(environment._libraries[(stage, action)][2]),
                    "sha256": _tensor_sha256(
                        environment._libraries[(stage, action)][2].to(torch.float64)
                    ),
                }
                for stage in range(environment.horizon)
                for action in range(base_context.n_actions)
            ],
        }
        if theta.bridge_mode == "exact_c13_anchor"
        else environment.bridge_identity()
    )
    identity = {
        "base_nuisance_context_sha256": base["combined_sha256"],
        "outcome_model_state_sha256": base["outcome_model_state_sha256"],
        "behavior_policy_state_sha256": base["behavior_policy_state_sha256"],
        "split_patient_id_sha256": base["split_patient_id_sha256"],
        "active_config_sha256": base["active_config_sha256"],
        "theta": theta.to_dict(),
        "state_kernel": state_kernel,
        "outcome_bridge": bridge,
    }
    return {**identity, "combined_sha256": _json_sha256(identity)}


def _development_selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 20:
        raise RuntimeError("v5 development requires all 20 old-lineage seeds")
    candidates = bridge_candidates()
    expected_ids = tuple(theta.candidate_id for theta in candidates)
    summaries = {}
    for candidate_index, theta in enumerate(candidates):
        metrics = []
        for row in rows:
            candidate_rows = row.get("candidates")
            if (
                not isinstance(candidate_rows, list)
                or tuple(item["theta"]["candidate_id"] for item in candidate_rows)
                != expected_ids
            ):
                raise RuntimeError("v5 candidate order differs within development")
            metrics.append(candidate_rows[candidate_index]["metrics"])
        summaries[theta.candidate_id] = summarize_candidate(theta, metrics)
    selection = select_bridge_candidate(candidates, summaries)
    return {
        **selection,
        "candidate_dataset_summaries": {
            candidate_id: summary.to_dict()
            for candidate_id, summary in summaries.items()
        },
    }


def _development_final(selection: Mapping[str, Any]) -> dict[str, Any]:
    admissible = bool(selection["development_admissible"])
    return {
        "protocol": PROTOCOL,
        "phase": "development",
        "dataset": DATASET,
        "status": (
            "DEVELOPMENT_GO" if admissible else "DEVELOPMENT_NO_GO"
        ),
        "development_admissible": admissible,
        "development_pass_count": selection["winner_summary"]["pass_count"],
        "development_structural_pass_count": selection["winner_summary"][
            "structural_pass_count"
        ],
        "winner": selection["winner"],
        "selection_sha256": _json_sha256(selection),
        "candidate_seed_deletions": 0,
        "coverage_generated": False,
        "information_firewall_respected": True,
    }


def _frozen_settings(
    final: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "development_admissible": final["development_admissible"],
        "theta": final["winner"] if final["development_admissible"] else None,
        "development_source_tree_sha256": metadata["source_tree_sha256"],
        "development_config_sha256": metadata["config_sha256"],
        "parent_v4_binding_sha256": metadata["parent_v4_binding_sha256"],
        "selection_sha256": final["selection_sha256"],
        "candidate_seed_deletions": 0,
        "coverage_generated": False,
    }


def _confirmation_gate(
    theta: BridgeTheta,
    support_rows: Sequence[Mapping[str, Any]],
    k0_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(support_rows) != 20:
        raise RuntimeError("v5 confirmation support requires all 20 fresh seeds")
    support_count = sum(bool(row["passed"]) for row in support_rows)
    if support_count >= DEVELOPMENT_MINIMUM_PASS_COUNT and len(k0_rows) != 20:
        raise RuntimeError("v5 confirmation K0 requires all 20 fresh seeds")
    if support_count < DEVELOPMENT_MINIMUM_PASS_COUNT and k0_rows:
        raise RuntimeError("v5 K0 opened after support NO-GO")
    k0_count = sum(bool(row["passed"]) for row in k0_rows)
    structural_count = sum(
        bool(row["metrics"]["structural_invariants"]) for row in k0_rows
    )
    confirmed = (
        support_count >= DEVELOPMENT_MINIMUM_PASS_COUNT
        and k0_count >= DEVELOPMENT_MINIMUM_PASS_COUNT
        and structural_count == REQUIRED_STRUCTURAL_PASS_COUNT
    )
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "CONFIRMATION_GATE_GO" if confirmed else "CONFIRMATION_GATE_NO_GO",
        "confirmation_opened": True,
        "development_admissible": True,
        "support_pass_count": support_count,
        "k0_pass_count": k0_count,
        "structural_pass_count": structural_count,
        "prespecified_seed_count": 20,
        "candidate_seed_deletions": 0,
        "theta": theta.to_dict(),
        "gate_role": "fresh_split_operational_gate",
        "independent_patient_confirmation_claimed": False,
        "coverage_generated": False,
    }


def _confirmation_final(gate: Mapping[str, Any]) -> dict[str, Any]:
    confirmed = gate["status"] == "CONFIRMATION_GATE_GO"
    return {
        "protocol": PROTOCOL,
        "phase": "confirmation",
        "dataset": DATASET,
        "status": (
            "CONFIRMATION_COMPLETE_GO" if confirmed else "CONFIRMATION_COMPLETE_NO_GO"
        ),
        "confirmed": confirmed,
        "gate": dict(gate),
        "candidate_seed_deletions": 0,
        "coverage_generated": False,
        "information_firewall_respected": True,
    }


def validate_parent_v4_bundles(config: FidelityV5Config) -> dict[str, Any]:
    v4_config = load_fidelity_v4_config(V4_CONFIG_PATH)
    bindings = {}
    for label, root_binding in (
        ("development", config.parent_development),
        ("confirmation_retry", config.parent_confirmation_retry),
    ):
        root = root_binding.root
        if not root.is_absolute():
            root = ROOT / root
        for relative, expected_hash in root_binding.file_sha256.items():
            path = root / relative
            if not path.is_file() or _file_sha256(path) != expected_hash:
                raise RuntimeError(f"frozen v4 {label} binding differs: {relative}")
        metadata = _read_json(root / "metadata.json")
        v4._validate_root_bundle(root, metadata, config=v4_config)
        bindings[label] = {
            "root": root.relative_to(ROOT).as_posix(),
            "manifest_sha256": _file_sha256(root / "manifest.json"),
            "complete_sha256": _file_sha256(root / "COMPLETE"),
            "metadata_sha256": _file_sha256(root / "metadata.json"),
            "full_semantic_bundle_validation": True,
        }

    development_status = _read_json(
        ROOT / config.parent_development.root / "FINAL_STATUS.json"
    )
    cxr = development_status["dataset_decisions"][DATASET]
    selection = _read_json(
        ROOT / config.parent_development.root / "selection/mimic_cxr.json"
    )
    c13 = selection["winner"]
    if (
        cxr["status"] != "DATASET_DEVELOPMENT_NO_GO"
        or cxr["development_pass_count"] != 18
        or cxr["development_structural_pass_count"] != 20
        or c13["candidate_id"] != "C13_raw_k10000_uniform_ridge_residual_raw"
        or c13["metric"] != "raw"
        or c13["neighbors"] != 10_000
        or c13["weight"] != "uniform"
        or c13["transition_mode"] != "ridge_residual"
        or c13["outcome_residual_mode"] != "raw"
        or c13["ridge_mode"] != "sample_normalized_no_intercept"
        or c13["ridge_value"] != 1e-3
        or cxr["selection_sha256"]
        != "0312cbd3b8a46888ca3fd8eba63aa4e72d4f693a4628aa9cf83ddd54a23af783"
    ):
        raise RuntimeError("frozen v4 C13 development decision differs")
    retry_status = _read_json(
        ROOT / config.parent_confirmation_retry.root / "FINAL_STATUS.json"
    )
    retry_cxr = retry_status["datasets"][DATASET]
    if (
        retry_cxr["status"] != "CONFIRMATION_NOT_OPENED_DEVELOPMENT_NO_GO"
        or retry_cxr["confirmation_opened"] is not False
        or retry_cxr["coverage_generated"] is not False
        or retry_status["confirmed_datasets"] != ["mimic_iv", "eicu", "inspire"]
        or retry_status["coverage_generated"] is not False
    ):
        raise RuntimeError("frozen v4 confirmation-retry decision differs")
    binding = {
        "protocol": PROTOCOL,
        "development": bindings["development"],
        "confirmation_retry": bindings["confirmation_retry"],
        "c13_selection_sha256": cxr["selection_sha256"],
        "c13_development_pass_count": 18,
        "c13_development_structural_pass_count": 20,
        "cxr_confirmation_was_never_opened": True,
        "coverage_generated": False,
    }
    return {**binding, "combined_sha256": _json_sha256(binding)}


def audit_development_reuse(
    config: FidelityV5Config,
    *,
    parent_binding: Mapping[str, Any],
) -> dict[str, Any]:
    del parent_binding
    mapping = _development_reuse_mapping()
    if (
        len(mapping) != 100
        or len(set(mapping.values())) != 100
        or _json_sha256(mapping) != DEVELOPMENT_MAPPING_SHA256
        or _integer_set_sha256(mapping.values()) != DEVELOPMENT_ID_SET_SHA256
    ):
        raise RuntimeError("v5 development RNG reuse mapping differs")
    parent_root = ROOT / config.parent_development.root
    parent_metadata = _read_json(parent_root / "metadata.json")
    parent_mapping = parent_metadata["development_rng_reuse_audit"]["mapping"]
    if any(parent_mapping.get(label) != value for label, value in mapping.items()):
        raise RuntimeError("v5 development streams are not exact v4 CXR lineage")
    envelope = []
    for seed in config.development_seeds:
        path = parent_root / "repair" / DATASET / f"seed_{seed:06d}.json"
        payload = _read_json(path)
        result = payload["result"]
        candidate_rows = result["candidates"]
        c13_rows = [
            row
            for row in candidate_rows
            if row["theta"]["candidate_id"]
            == "C13_raw_k10000_uniform_ridge_residual_raw"
        ]
        if (
            len(c13_rows) != 1
            or result["coverage_generated"] is not False
            or c13_rows[0]["systematic_replay"]["base_uniform_seed"]
            != v2.K0_UNIFORM_SEED_OFFSET + seed
        ):
            raise RuntimeError("v4 C13 seed envelope differs")
        envelope.append(
            {
                "seed": seed,
                "file_sha256": _file_sha256(path),
                "base_uniform_sha256": c13_rows[0]["systematic_replay"][
                    "base_uniform_sha256"
                ],
                "base_nuisance_context_sha256": c13_rows[0]["context_identity"][
                    "base_nuisance_context_sha256"
                ],
                "split_patient_id_sha256": result["split_audit"][
                    "role_patient_id_sha256"
                ],
            }
        )
    return {
        "status": "passed_before_launch",
        "role": "exact_authorized_reuse_of_v4_cxr_development_lineage",
        "base_seed_count": 20,
        "stream_count": len(mapping),
        "mapping": mapping,
        "mapping_sha256": _json_sha256(mapping),
        "rng_id_set_sha256": _integer_set_sha256(mapping.values()),
        "parent_seed_envelope_count": len(envelope),
        "parent_seed_envelope_sha256": _json_sha256(envelope),
        "authorized_lineage_collision_count": len(mapping),
        "unauthorized_collision_count": 0,
        "common_random_numbers_across_candidates": True,
        "scientific_freshness_claimed": False,
    }


def _development_reuse_mapping() -> dict[str, int]:
    mapping = {}
    for seed in DEVELOPMENT_SEEDS:
        prefix = f"{DATASET}/base_{seed}"
        mapping[f"{prefix}/task"] = seed
        mapping[f"{prefix}/outcome_model"] = seed + 1
        mapping[f"{prefix}/behavior_model"] = seed + 2
        mapping[f"{prefix}/cxr_encoder"] = seed + 701
        mapping[f"{prefix}/k0_base_uniform"] = v2.K0_UNIFORM_SEED_OFFSET + seed
    return mapping


def audit_confirmation_rng(config: FidelityV5Config) -> dict[str, Any]:
    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    preset = replace(
        protocol.datasets[DATASET],
        seeds=config.confirmation_seeds,
        bootstrap_seed=config.confirmation_bootstrap_seed,
    )
    active_protocol = replace(
        protocol, datasets={**protocol.datasets, DATASET: preset}
    )
    mapping = v2._new_rng_stream_mapping(active_protocol, (DATASET,))
    v2._assert_unique_rng_streams(mapping)
    mapping_values = set(mapping.values())
    if (
        len(mapping) != 341
        or _json_sha256(mapping) != CONFIRMATION_MAPPING_SHA256
        or _integer_set_sha256(mapping_values) != CONFIRMATION_ID_SET_SHA256
        or _integer_set_sha256(config.confirmation_seeds)
        != CONFIRMATION_BASE_SET_SHA256
    ):
        raise RuntimeError("v5 confirmation derived RNG mapping differs")

    excluded_roots = (DEVELOPMENT_ROOT, CONFIRMATION_ROOT)
    artifact_ids = _artifact_rng_ids(
        ROOT / "results", excluded_roots=excluded_roots
    )
    source_ids = v2._source_declared_seeds(
        ROOT, excluded_paths=_OWN_RNG_DECLARATION_PATHS
    )
    prior = artifact_ids | source_ids
    collisions = {
        label: value for label, value in mapping.items() if value in prior
    }
    result = {
        "status": "passed_before_launch" if not collisions else "collision_detected",
        "collision_count": len(collisions),
        "collisions": collisions,
        "artifact_rng_id_count": len(artifact_ids),
        "artifact_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "source_declared_rng_id_count": len(source_ids),
        "source_declared_rng_id_sha256": _integer_set_sha256(source_ids),
        "prior_rng_id_count": len(prior),
        "prior_rng_id_sha256": _integer_set_sha256(prior),
        "new_rng_stream_count": len(mapping),
        "new_rng_stream_mapping": mapping,
        "new_rng_stream_mapping_sha256": _json_sha256(mapping),
        "new_rng_id_set_sha256": _integer_set_sha256(mapping_values),
        "confirmation_base_seed_count": len(config.confirmation_seeds),
        "confirmation_base_seed_set_sha256": _integer_set_sha256(
            config.confirmation_seeds
        ),
        "internal_rng_streams_unique": len(mapping_values) == len(mapping),
        "excluded_roots": [str(path) for path in excluded_roots],
        "formal_rng_consumed": False,
    }
    proposed = IndependentAudit(
        status="GO",
        attestation_sha256=None,
        expected_prior_count=result["prior_rng_id_count"],
        expected_prior_sha256=result["prior_rng_id_sha256"],
        expected_artifact_count=result["artifact_rng_id_count"],
        expected_artifact_sha256=result["artifact_rng_id_sha256"],
        expected_source_count=result["source_declared_rng_id_count"],
        expected_source_sha256=result["source_declared_rng_id_sha256"],
    )
    result["proposed_independent_audit"] = {
        **asdict(proposed),
        "attestation_sha256": independent_audit_attestation_sha256(proposed),
    }
    return result


def _artifact_rng_ids(
    root: Path, *, excluded_roots: Sequence[Path]
) -> set[int]:
    values: set[int] = set()
    excluded = tuple(path.resolve() for path in excluded_roots)
    if not root.exists():
        return values
    named_files = {
        "metadata.json",
        "study_metadata.json",
        "suite_manifest.json",
        "manifest.json",
        "summary.json",
    }
    for path in root.rglob("*"):
        resolved = path.resolve()
        if any(resolved == item or item in resolved.parents for item in excluded):
            continue
        match = v2._SEED_NAME.fullmatch(path.name)
        if match:
            values.add(int(match.group(1)))
        if not path.is_file() or path.suffix != ".json":
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if path.name in named_files:
            six._collect_named_seed_values(payload, values)
        v2._collect_artifact_rng_values(payload, values)
    return values


def _validate_frozen_audit_snapshot(
    config: FidelityV5Config, audit: Mapping[str, Any]
) -> None:
    frozen = config.independent_audit
    observed = (
        audit["prior_rng_id_count"],
        audit["prior_rng_id_sha256"],
        audit["artifact_rng_id_count"],
        audit["artifact_rng_id_sha256"],
        audit["source_declared_rng_id_count"],
        audit["source_declared_rng_id_sha256"],
    )
    expected = (
        frozen.expected_prior_count,
        frozen.expected_prior_sha256,
        frozen.expected_artifact_count,
        frozen.expected_artifact_sha256,
        frozen.expected_source_count,
        frozen.expected_source_sha256,
    )
    if audit["collision_count"] != 0 or observed != expected:
        raise RuntimeError("v5 independent RNG audit snapshot differs")


def _run_seed_phase(
    phase_root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    devices: tuple[str, ...],
    candidates: tuple[BridgeTheta, ...],
    worker: Callable[..., dict[str, Any]],
    worker_arguments: tuple[object, ...],
    source_hash: str,
) -> list[dict[str, Any]]:
    phase_root.mkdir(parents=True, exist_ok=True)
    mapping = _seed_device_mapping(preset.seeds, devices)
    candidate_hash = _json_sha256(
        [candidate.to_dict() for candidate in candidates]
    )
    expected = {
        phase_root / f"seed_{seed:06d}.json" for seed in preset.seeds
    }
    unexpected = set(phase_root.glob("seed_*.json")) - expected
    if unexpected:
        raise RuntimeError(f"unexpected {phase} artifacts: {sorted(unexpected)}")

    completed: dict[int, dict[str, Any]] = {}
    for seed in preset.seeds:
        path = phase_root / f"seed_{seed:06d}.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        _validate_seed_payload(
            payload,
            phase=phase,
            preset=preset,
            seed=seed,
            device=mapping[seed],
            source_hash=source_hash,
            candidate_hash=candidate_hash,
            candidates=candidates,
        )
        completed[seed] = payload["result"]

    pending = tuple(seed for seed in preset.seeds if seed not in completed)
    if pending and (phase_root / "COMPLETE").exists():
        raise RuntimeError(f"{phase} COMPLETE exists with missing seeds")
    if pending:
        groups = tuple(
            tuple(seed for seed in pending if mapping[seed] == device)
            for device in devices
        )
        with ProcessPoolExecutor(
            max_workers=len(devices), mp_context=get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(
                    _phase_group,
                    group,
                    device,
                    preset,
                    candidates,
                    worker,
                    worker_arguments,
                ): device
                for group, device in zip(groups, devices, strict=True)
                if group
            }
            for future in as_completed(futures):
                for seed, device, result in future.result():
                    payload = {
                        "protocol": PROTOCOL,
                        "phase": phase,
                        "dataset": DATASET,
                        "seed": seed,
                        "device": device,
                        "source_tree_sha256": source_hash,
                        "candidate_contract_sha256": candidate_hash,
                        "result": result,
                    }
                    _validate_seed_payload(
                        payload,
                        phase=phase,
                        preset=preset,
                        seed=seed,
                        device=device,
                        source_hash=source_hash,
                        candidate_hash=candidate_hash,
                        candidates=candidates,
                    )
                    _write_json(phase_root / f"seed_{seed:06d}.json", payload)
                    completed[seed] = result
    if set(completed) != set(preset.seeds):
        raise RuntimeError(f"{phase} did not complete all 20 seeds")
    _write_text(phase_root / "COMPLETE", "complete\n")
    return [completed[seed] for seed in preset.seeds]


def _phase_group(
    seeds: tuple[int, ...],
    device: str,
    preset: DatasetPreset,
    candidates: tuple[BridgeTheta, ...],
    worker: Callable[..., dict[str, Any]],
    worker_arguments: tuple[object, ...],
) -> list[tuple[int, str, dict[str, Any]]]:
    torch.cuda.set_device(torch.device(device))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    rows = []
    for seed in seeds:
        rows.append(
            (
                seed,
                device,
                worker(seed, preset, device, candidates, *worker_arguments),
            )
        )
        torch.cuda.empty_cache()
    return rows


def _validate_seed_payload(
    payload: Mapping[str, Any],
    *,
    phase: str,
    preset: DatasetPreset,
    seed: int,
    device: str,
    source_hash: str,
    candidate_hash: str,
    candidates: tuple[BridgeTheta, ...],
) -> None:
    expected_keys = {
        "protocol",
        "phase",
        "dataset",
        "seed",
        "device",
        "source_tree_sha256",
        "candidate_contract_sha256",
        "result",
    }
    if (
        set(payload) != expected_keys
        or payload["protocol"] != PROTOCOL
        or payload["phase"] != phase
        or payload["dataset"] != DATASET
        or payload["seed"] != seed
        or payload["device"] != device
        or payload["source_tree_sha256"] != source_hash
        or payload["candidate_contract_sha256"] != candidate_hash
    ):
        raise RuntimeError(f"invalid v5 seed envelope for {seed}")
    result = payload["result"]
    if (
        not isinstance(result, Mapping)
        or result.get("seed") != seed
        or result.get("dataset") != DATASET
        or result.get("phase") != phase
        or result.get("coverage_generated") is not False
    ):
        raise RuntimeError(f"invalid v5 seed result for {seed}")
    if phase == "confirmation_support":
        support = {**result, "phase": "support"}
        support.pop("confirmation_label", None)
        if not v2._valid_support_result(support, preset):
            raise RuntimeError(f"invalid v5 support result for {seed}")
        return
    if phase == "development_k0_only":
        rows = result.get("candidates")
        if (
            not isinstance(rows, list)
            or len(rows) != len(candidates)
            or tuple(row["theta"] for row in rows)
            != tuple(theta.to_dict() for theta in candidates)
        ):
            raise RuntimeError(f"invalid v5 development candidates for {seed}")
        for row in rows:
            _validate_k0_candidate_row(row)
        return
    if phase == "confirmation_k0":
        if len(candidates) != 1 or result.get("theta") != candidates[0].to_dict():
            raise RuntimeError(f"invalid v5 confirmation theta for {seed}")
        _validate_k0_candidate_row(result)
        return
    raise RuntimeError("unknown v5 seed phase")


def _validate_k0_candidate_row(row: Mapping[str, Any]) -> None:
    metrics = row.get("metrics")
    detail = row.get("systematic_replay")
    identity = row.get("context_identity")
    if (
        not isinstance(metrics, Mapping)
        or set(metrics)
        != {
            "maximum_score_ks",
            "maximum_signed_residual_w1",
            "maximum_successor_mean_w1",
            "maximum_successor_q95_w1",
            "structural_invariants",
        }
        or not isinstance(detail, Mapping)
        or detail.get("aggregate_gate_unchanged") is not True
        or detail.get("descriptive_diagnostics_non_gating") is not True
        or detail.get("score_ks_semantics") != "one_scalar_per_stage"
        or len(detail.get("signed_residual_w1_by_stage_outcome", ())) != 6
        or any(
            len(values) != 2
            for values in detail.get("signed_residual_w1_by_stage_outcome", ())
        )
        or len(detail.get("action_stratified_by_stage", ())) != 6
        or any(
            len(values) != 3
            for values in detail.get("action_stratified_by_stage", ())
        )
        or not isinstance(identity, Mapping)
        or identity.get("combined_sha256")
        != _json_sha256(
            {key: value for key, value in identity.items() if key != "combined_sha256"}
        )
    ):
        raise RuntimeError("invalid v5 K0 candidate payload")
    ratio = normalized_seed_ratio(metrics)
    expected_ratio = ratio if math.isfinite(ratio) else None
    if (
        row.get("passed")
        != (
            bool(metrics["structural_invariants"])
            and all(float(metrics[name]) <= threshold for name, threshold in K0_THRESHOLDS.items())
        )
        or row.get("normalized_seed_ratio") != expected_ratio
        or row.get("structural_failure_ratio_is_infinite")
        is not (not math.isfinite(ratio))
    ):
        raise RuntimeError("v5 K0 candidate decision differs")


def _root_metadata(
    *,
    phase: str,
    output_root: Path,
    config: FidelityV5Config,
    devices: Sequence[str],
    source_hash: str,
    source_snapshot: Mapping[str, Any],
    parent_binding: Mapping[str, Any],
    development_audit: Mapping[str, Any],
    confirmation_audit: Mapping[str, Any],
    development_binding: Mapping[str, Any] | None = None,
    frozen_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config_bytes = CONFIG_PATH.read_bytes()
    active_seeds = (
        config.development_seeds
        if phase == "development"
        else config.confirmation_seeds
    )
    metadata = {
        "protocol": PROTOCOL,
        "phase": phase,
        "role": "coverage_blind_mimic_cxr_outcome_bridge_repair",
        "dataset": DATASET,
        "devices": list(devices),
        "output_root": str(output_root),
        "seed_to_device": {
            str(seed): device
            for seed, device in _seed_device_mapping(active_seeds, devices).items()
        },
        "source_tree_sha256": source_hash,
        "source_snapshot": dict(source_snapshot),
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_bytes": len(config_bytes),
        "parent_v4_binding": dict(parent_binding),
        "parent_v4_binding_sha256": _json_sha256(parent_binding),
        "development_rng_reuse_audit": dict(development_audit),
        "confirmation_rng_audit": dict(confirmation_audit),
        "independent_audit": asdict(config.independent_audit),
        "candidate_contract": [
            theta.to_dict() for theta in bridge_candidates()
        ],
        "selector_contract": {
            "version": SELECTOR_VERSION,
            "minimum_pass_count": DEVELOPMENT_MINIMUM_PASS_COUNT,
            "required_structural_pass_count": REQUIRED_STRUCTURAL_PASS_COUNT,
            "candidate_seed_deletion_permitted": False,
        },
        "diagnostic_contract": {
            "aggregate_gate_unchanged": True,
            "score_ks": "stage_scalar",
            "outcome_coordinate_arrays": [
                "signed_residual_w1",
                "clinical_successor_mean_w1",
                "clinical_successor_q95_w1",
            ],
            "action_stratification": "descriptive_non_gating",
        },
        "coverage_generation_permitted": False,
        "scientific_result_execution_path_present": False,
        "canonical_scpcp_mutation_permitted": False,
    }
    if phase == "confirmation":
        metadata["development_binding"] = dict(development_binding or {})
        metadata["development_binding_sha256"] = _json_sha256(
            development_binding
        )
        metadata["frozen_settings"] = dict(frozen_settings or {})
        metadata["frozen_settings_sha256"] = _json_sha256(frozen_settings)
        metadata["independent_patient_confirmation_claimed"] = False
    return metadata


def _prepare_root(
    root: Path,
    metadata: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if resume:
        if not root.is_dir() or _read_json(root / "metadata.json") != metadata:
            raise RuntimeError("v5 resume metadata differs")
        v4._verify_source_snapshot(root, metadata["source_snapshot"])
        return
    if root.exists():
        raise FileExistsError(f"fresh v5 output already exists: {root}")
    root.mkdir(parents=True)
    v4._atomic_write(
        root / source_snapshot["contract"]["archive_path"],
        source_snapshot["archive_bytes"],
    )
    v4._atomic_write(
        root / source_snapshot["contract"]["manifest_path"],
        source_snapshot["manifest_bytes"],
    )
    _write_json(root / "metadata.json", metadata)
    v4._verify_source_snapshot(root, metadata["source_snapshot"])


def _complete_and_valid(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV5Config,
) -> bool:
    if not (root / "COMPLETE").exists():
        return False
    try:
        _validate_root_bundle(root, metadata, config=config)
    except (Exception, KeyboardInterrupt):
        v4._unlink_root_complete(root)
        raise
    return True


def _finalize_root(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    source_hash: str,
    config: FidelityV5Config,
) -> None:
    if experiment_tree_sha256() != source_hash:
        raise RuntimeError("source/config changed during the formal v5 phase")
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("v5 metadata changed during the formal phase")
    if validate_parent_v4_bundles(config) != metadata["parent_v4_binding"]:
        raise RuntimeError("v4 parent evidence changed during v5")
    _assert_no_forbidden_result_paths(root)
    _write_manifest(root)
    _validate_root_contents(root, metadata, config=config)
    final = _read_json(root / "FINAL_STATUS.json")
    complete = (
        f"complete phase={metadata['phase']} "
        f"source_tree_sha256={source_hash} "
        f"config_sha256={metadata['config_sha256']} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={_file_sha256(root / 'manifest.json')}\n"
    )
    _write_text(root / "COMPLETE", complete)
    try:
        _validate_root_bundle(root, metadata, config=config)
    except (Exception, KeyboardInterrupt):
        v4._unlink_root_complete(root)
        raise


def _validate_root_bundle(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV5Config,
) -> None:
    _validate_root_contents(root, metadata, config=config)
    final = _read_json(root / "FINAL_STATUS.json")
    expected = (
        f"complete phase={metadata['phase']} "
        f"source_tree_sha256={metadata['source_tree_sha256']} "
        f"config_sha256={metadata['config_sha256']} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={_file_sha256(root / 'manifest.json')}\n"
    )
    if not (root / "COMPLETE").is_file() or (root / "COMPLETE").read_text() != expected:
        raise RuntimeError("v5 COMPLETE marker differs")


def _validate_root_contents(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV5Config,
) -> None:
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("v5 root metadata differs")
    if (
        metadata.get("parent_v4_binding_sha256")
        != _json_sha256(metadata.get("parent_v4_binding"))
        or validate_parent_v4_bundles(config)
        != metadata.get("parent_v4_binding")
    ):
        raise RuntimeError("v5 parent binding differs")
    v4._verify_source_snapshot(root, metadata["source_snapshot"])
    _verify_manifest(root)
    _assert_no_forbidden_result_paths(root)
    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    if metadata["phase"] == "development":
        preset = replace(protocol.datasets[DATASET], seeds=config.development_seeds)
        rows = _load_seed_results(
            root / "repair",
            phase="development_k0_only",
            preset=preset,
            devices=tuple(metadata["devices"]),
            candidates=bridge_candidates(),
            source_hash=metadata["source_tree_sha256"],
        )
        selection = _development_selection(rows)
        final = _development_final(selection)
        frozen = _frozen_settings(final, metadata)
        if (
            _read_json(root / "selection.json") != selection
            or _read_json(root / "FINAL_STATUS.json") != final
            or _read_json(root / "frozen_settings.json") != frozen
        ):
            raise RuntimeError("v5 development semantic recomputation differs")
        expected = {
            Path("metadata.json"),
            Path("FINAL_STATUS.json"),
            Path("selection.json"),
            Path("frozen_settings.json"),
            Path("repair/COMPLETE"),
            *(
                Path("repair") / f"seed_{seed:06d}.json"
                for seed in config.development_seeds
            ),
        }
    elif metadata["phase"] == "confirmation":
        preset = replace(
            protocol.datasets[DATASET],
            seeds=config.confirmation_seeds,
            bootstrap_seed=config.confirmation_bootstrap_seed,
        )
        theta = _theta_from_dict(metadata["frozen_settings"]["theta"])
        support = _load_seed_results(
            root / "support",
            phase="confirmation_support",
            preset=preset,
            devices=tuple(metadata["devices"]),
            candidates=(),
            source_hash=metadata["source_tree_sha256"],
        )
        support_count = sum(bool(row["passed"]) for row in support)
        k0 = []
        expected = {
            Path("metadata.json"),
            Path("FINAL_STATUS.json"),
            Path("gate.json"),
            Path("support/COMPLETE"),
            *(
                Path("support") / f"seed_{seed:06d}.json"
                for seed in config.confirmation_seeds
            ),
        }
        if support_count >= DEVELOPMENT_MINIMUM_PASS_COUNT:
            k0 = _load_seed_results(
                root / "k0_fidelity",
                phase="confirmation_k0",
                preset=preset,
                devices=tuple(metadata["devices"]),
                candidates=(theta,),
                source_hash=metadata["source_tree_sha256"],
            )
            expected.add(Path("k0_fidelity/COMPLETE"))
            expected.update(
                Path("k0_fidelity") / f"seed_{seed:06d}.json"
                for seed in config.confirmation_seeds
            )
        gate = _confirmation_gate(theta, support, k0)
        if (
            _read_json(root / "gate.json") != gate
            or _read_json(root / "FINAL_STATUS.json") != _confirmation_final(gate)
        ):
            raise RuntimeError("v5 confirmation semantic recomputation differs")
    else:
        raise RuntimeError("unknown v5 root phase")
    expected.update(
        {
            Path(metadata["source_snapshot"]["archive_path"]),
            Path(metadata["source_snapshot"]["manifest_path"]),
        }
    )
    _assert_exact_artifact_file_set(root, expected)


def _verify_development_for_confirmation(
    root: Path,
    *,
    config: FidelityV5Config,
    parent_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _read_json(root / "metadata.json")
    _validate_root_bundle(root, metadata, config=config)
    final = _read_json(root / "FINAL_STATUS.json")
    frozen = _read_json(root / "frozen_settings.json")
    if (
        final["status"] != "DEVELOPMENT_GO"
        or frozen["development_admissible"] is not True
        or frozen["theta"] is None
        or metadata["parent_v4_binding"] != parent_binding
        or metadata["parent_v4_binding_sha256"] != _json_sha256(parent_binding)
    ):
        raise RuntimeError("v5 development did not authorize confirmation")
    binding = {
        "root": root.relative_to(ROOT).as_posix(),
        "complete_sha256": _file_sha256(root / "COMPLETE"),
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "metadata_sha256": _file_sha256(root / "metadata.json"),
        "final_status_sha256": _file_sha256(root / "FINAL_STATUS.json"),
        "frozen_settings_sha256": _file_sha256(root / "frozen_settings.json"),
        "source_tree_sha256": metadata["source_tree_sha256"],
        "selection_sha256": final["selection_sha256"],
        "full_semantic_bundle_validation": True,
    }
    return {**binding, "combined_sha256": _json_sha256(binding)}, frozen


def _load_seed_results(
    phase_root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    devices: tuple[str, ...],
    candidates: tuple[BridgeTheta, ...],
    source_hash: str,
) -> list[dict[str, Any]]:
    if (phase_root / "COMPLETE").read_text() != "complete\n":
        raise RuntimeError(f"invalid {phase} COMPLETE marker")
    mapping = _seed_device_mapping(preset.seeds, devices)
    candidate_hash = _json_sha256(
        [candidate.to_dict() for candidate in candidates]
    )
    rows = []
    for seed in preset.seeds:
        payload = _read_json(phase_root / f"seed_{seed:06d}.json")
        _validate_seed_payload(
            payload,
            phase=phase,
            preset=preset,
            seed=seed,
            device=mapping[seed],
            source_hash=source_hash,
            candidate_hash=candidate_hash,
            candidates=candidates,
        )
        rows.append(payload["result"])
    return rows


def _write_manifest(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden in v5 bundle: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative in {Path("manifest.json"), Path("COMPLETE")}:
            continue
        if ".tmp-" in path.name:
            raise RuntimeError(f"temporary v5 artifact remains: {path}")
        _resolve_inside_root(root, relative)
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    _write_json(
        root / "manifest.json",
        {
            "protocol": PROTOCOL,
            "artifact_count": len(entries),
            "artifacts": entries,
        },
    )


def _verify_manifest(root: Path) -> None:
    manifest = _read_json(root / "manifest.json")
    entries = manifest.get("artifacts")
    if (
        set(manifest) != {"protocol", "artifact_count", "artifacts"}
        or manifest.get("protocol") != PROTOCOL
        or not isinstance(entries, list)
    ):
        raise RuntimeError("invalid v5 manifest header")
    expected = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "bytes", "sha256"}:
            raise RuntimeError("malformed v5 manifest entry")
        relative = Path(entry["path"])
        if relative in {Path("manifest.json"), Path("COMPLETE")}:
            raise RuntimeError("v5 manifest contains a root commit file")
        path = _resolve_inside_root(root, relative)
        if path in expected:
            raise RuntimeError("duplicate v5 manifest entry")
        expected.add(path)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != entry["bytes"]
            or _file_sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"v5 manifest mismatch: {path}")
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden in v5 bundle: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative in {Path("manifest.json"), Path("COMPLETE")}:
            continue
        observed.add(_resolve_inside_root(root, relative))
    if (
        observed != expected
        or manifest["artifact_count"] != len(entries)
        or len(expected) != len(entries)
    ):
        raise RuntimeError("v5 manifest file set differs")


def _assert_exact_artifact_file_set(root: Path, expected: set[Path]) -> None:
    if any(path.is_absolute() or ".." in path.parts for path in expected):
        raise RuntimeError("expected v5 path escapes root")
    excluded = {Path("manifest.json"), Path("COMPLETE")}
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden in v5 bundle: {path}")
        if path.is_file() and path.relative_to(root) not in excluded:
            observed.add(path.relative_to(root))
    if observed != expected:
        raise RuntimeError(
            "v5 exact artifact file set differs; "
            f"missing={sorted(map(str, expected - observed))}; "
            f"extra={sorted(map(str, observed - expected))}"
        )


def _assert_no_forbidden_result_paths(root: Path) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(token in part.lower() for part in relative.parts for token in FORBIDDEN_RESULT_PATH_TOKENS):
            raise RuntimeError(f"forbidden result path in v5 bundle: {relative}")


def _resolve_inside_root(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("v5 path escapes its root")
    resolved_root = root.resolve()
    path = (root / relative).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise RuntimeError("v5 path escapes its root")
    return path


def _theta_from_dict(value: Mapping[str, Any]) -> BridgeTheta:
    theta = BridgeTheta(
        candidate_id=str(value["candidate_id"]),
        bridge_mode=str(value["bridge_mode"]),
    )
    if theta.to_dict() != dict(value):
        raise ValueError("serialized v5 bridge differs from frozen schema")
    return theta


def _seed_device_mapping(
    seeds: Sequence[int], devices: Sequence[str]
) -> dict[int, str]:
    if not devices:
        raise ValueError("v5 seed/device mapping needs at least one GPU")
    return {seed: devices[index % len(devices)] for index, seed in enumerate(seeds)}


def _validate_devices(devices: Sequence[str]) -> None:
    if len(devices) != 2 or any(
        not value.startswith("cuda:") for value in devices
    ):
        raise ValueError("formal v5 requires exactly two explicit CUDA devices")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: object) -> None:
    v4._write_json(path, value)


def _write_text(path: Path, value: str) -> None:
    v4._write_text(path, value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _integer_set_sha256(values: Iterable[int]) -> str:
    return _json_sha256(sorted(set(int(value) for value in values)))


if __name__ == "__main__":
    main()
