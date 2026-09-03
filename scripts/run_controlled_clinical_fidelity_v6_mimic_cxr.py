"""Run the terminal, coverage-blind MIMIC-CXR v6 bridge repair.

The audit phase is read-only. Development and confirmation remain locked until
an independent audit attests the exact source tree and fresh RNG namespace.
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
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_controlled_clinical_extension as v2  # noqa: E402
import scripts.run_controlled_clinical_fidelity_v5_mimic_cxr as v5run  # noqa: E402
from scpcp.artifacts import experiment_tree_sha256  # noqa: E402
from scpcp.controlled_clinical_extension import (  # noqa: E402
    ControlledClinicalExtensionConfig,
    DatasetPreset,
)
from scpcp.controlled_clinical_fidelity_v5_mimic_cxr import (  # noqa: E402
    load_fidelity_v5_config,
)
from scpcp.controlled_clinical_fidelity_v6_mimic_cxr import (  # noqa: E402
    CANDIDATE_ID,
    CONFIRMATION_BASE_SET_SHA256,
    CONFIRMATION_BOOTSTRAP_SEED,
    CONFIRMATION_ID_SET_SHA256,
    CONFIRMATION_MAPPING_SHA256,
    CONFIRMATION_MINIMUM_PASS_COUNT,
    DATASET,
    DEVELOPMENT_BASE_SET_SHA256,
    DEVELOPMENT_ID_SET_SHA256,
    DEVELOPMENT_LINEAGES,
    DEVELOPMENT_MAPPING_SHA256,
    DEVELOPMENT_REQUIRED_PASS_COUNT,
    HORIZON,
    K0_THRESHOLDS,
    PROTOCOL,
    REQUIRED_STRUCTURAL_PASS_COUNT,
    FidelityV6Config,
    IndependentAudit,
    TerminalBridgeTheta,
    build_terminal_environment,
    independent_audit_attestation_sha256,
    load_fidelity_v6_config,
    numeric_seed_ratio,
    normalized_seed_ratio,
    terminal_candidate,
)
from scpcp.scores import score_batch  # noqa: E402


CONFIG_PATH = ROOT / "configs/controlled_clinical_fidelity_v6_mimic_cxr.yaml"
V2_CONFIG_PATH = ROOT / "configs/controlled_clinical_extension.yaml"
V5_CONFIG_PATH = ROOT / "configs/controlled_clinical_fidelity_v5_mimic_cxr.yaml"
DEVELOPMENT_ROOT = (
    ROOT / "results/work/controlled_clinical_fidelity_v6_mimic_cxr_development"
).resolve()
CONFIRMATION_ROOT = (
    ROOT / "results/work/controlled_clinical_fidelity_v6_mimic_cxr_confirmation"
).resolve()
PHASES = ("audit", "development", "confirmation")
FORBIDDEN_RESULT_PATH_TOKENS = (
    "science",
    "coverage",
    "width",
    "method_selection",
)
_K0_DETAIL_KEYS = {
    "label",
    "systematic_replays",
    "base_uniform_seed",
    "base_uniform_shape",
    "base_uniform_sha256",
    "expanded_uniform_sha256",
    "expansion_formula",
    "flatten_order",
    "inference_unit",
    "episode_weighted",
    "patient_chunk_size",
    "score_ks_semantics",
    "score_ks_by_stage",
    "signed_residual_w1_by_stage_outcome",
    "signed_residual_max_w1_by_stage",
    "successor_mean_w1_by_stage",
    "successor_q95_w1_by_stage",
    "clinical_successor_mean_w1_by_stage_outcome",
    "clinical_successor_q95_w1_by_stage_outcome",
    "active_successor_coordinates_by_stage",
    "raw_structural_invariants_by_stage",
    "action_stratified_by_stage",
    "aggregate_gate_unchanged",
    "descriptive_diagnostics_non_gating",
}
_CONTEXT_IDENTITY_KEYS = {
    "base_nuisance_context_sha256",
    "outcome_model_state_sha256",
    "behavior_policy_state_sha256",
    "split_patient_id_sha256",
    "active_config_sha256",
    "theta",
    "state_kernel",
    "outcome_bridge",
    "combined_sha256",
}
_SPLIT_AUDIT_KEYS = {
    "role_unique_patient_counts",
    "role_episode_counts",
    "role_patient_id_sha256",
    "patient_sets_pairwise_disjoint",
    "split_fractions",
}
_OWN_RNG_DECLARATION_PATHS = {
    Path(__file__).resolve(),
    (ROOT / "src/scpcp/controlled_clinical_fidelity_v6_mimic_cxr.py").resolve(),
    CONFIG_PATH.resolve(),
    (ROOT / "scripts/run_controlled_clinical_extension.py").resolve(),
    (ROOT / "src/scpcp/controlled_clinical_extension.py").resolve(),
    V2_CONFIG_PATH.resolve(),
}
_V6_SOURCE_CONTRACT_PATHS = (
    CONFIG_PATH,
    (ROOT / "src/scpcp/controlled_clinical_fidelity_v6_mimic_cxr.py").resolve(),
    Path(__file__).resolve(),
    (
        ROOT / "tests/per_step/test_controlled_clinical_fidelity_v6_mimic_cxr.py"
    ).resolve(),
    (
        ROOT / "tests/per_step/test_controlled_clinical_fidelity_v6_mimic_cxr_runner.py"
    ).resolve(),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_fidelity_v6_config(CONFIG_PATH)
    if args.phase == "audit":
        _assert_formal_roots_absent()
    parent_binding = validate_parent_v5_bundles(config)
    development_audit = audit_development_reuse(config, parent_binding=parent_binding)
    confirmation_audit = audit_confirmation_rng(config)
    if args.phase == "audit":
        ready = confirmation_audit["collision_count"] == 0
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "status": (
                        "PREFLIGHT_READY_FOR_INDEPENDENT_AUDIT"
                        if ready
                        else "PREFLIGHT_NO_GO_RNG_COLLISION"
                    ),
                    "parent_v5_binding": parent_binding,
                    "development_rng_reuse_audit": development_audit,
                    "confirmation_rng_audit": confirmation_audit,
                    "formal_launch_authorized": (
                        ready and config.independent_audit.permits_formal_launch
                    ),
                    "formal_roots_absent_at_audit": True,
                    "formal_rng_consumed": False,
                    "coverage_generation_permitted": False,
                    "terminal_no_v7": True,
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
    config: FidelityV6Config,
    devices: tuple[str, ...],
    parent_binding: Mapping[str, Any],
    development_audit: Mapping[str, Any],
    confirmation_audit: Mapping[str, Any],
    resume: bool,
) -> None:
    source_hash, source_snapshot = v5run.v4._active_source_contract()
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
    if (output_root / "COMPLETE").exists():
        _assert_no_symlinks(output_root)
        if _complete_and_valid(output_root, metadata, config=config):
            return
    else:
        _validate_incomplete_resume_root(
            output_root,
            phase="development",
            metadata=metadata,
            config=config,
        )

    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    theta = terminal_candidate()
    lineage_rows = {}
    for lineage, seeds in config.development_lineages.items():
        preset = replace(protocol.datasets[DATASET], seeds=seeds)
        lineage_rows[lineage] = _run_seed_phase(
            output_root / "k0_fidelity" / lineage,
            phase="development_k0",
            preset=preset,
            devices=devices,
            candidates=(theta,),
            worker=_development_worker,
            worker_arguments=(protocol, lineage),
            source_hash=source_hash,
        )
    gate = _development_gate(lineage_rows)
    final = _development_final(gate)
    frozen = _frozen_settings(final, metadata)
    _write_json(output_root / "development_gate.json", gate)
    _write_json(output_root / "FINAL_STATUS.json", final)
    _write_json(output_root / "frozen_settings.json", frozen)
    _finalize_root(output_root, metadata, source_hash=source_hash, config=config)


def run_confirmation(
    output_root: Path,
    *,
    development_root: Path,
    config: FidelityV6Config,
    devices: tuple[str, ...],
    parent_binding: Mapping[str, Any],
    development_audit: Mapping[str, Any],
    confirmation_audit: Mapping[str, Any],
    resume: bool,
) -> None:
    development_binding, frozen = _verify_development_for_confirmation(
        development_root, config=config, parent_binding=parent_binding
    )
    source_hash, source_snapshot = v5run.v4._active_source_contract()
    if source_hash != frozen["development_source_tree_sha256"]:
        raise RuntimeError("source/config changed after the v6 bridge was frozen")
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
    if (output_root / "COMPLETE").exists():
        _assert_no_symlinks(output_root)
        if _complete_and_valid(output_root, metadata, config=config):
            return
    else:
        _validate_incomplete_resume_root(
            output_root,
            phase="confirmation",
            metadata=metadata,
            config=config,
        )

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
    if support_count >= CONFIRMATION_MINIMUM_PASS_COUNT:
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
    candidates: tuple[TerminalBridgeTheta, ...],
    protocol: ControlledClinicalExtensionConfig,
    lineage: str,
) -> dict[str, Any]:
    if candidates != (terminal_candidate(),):
        raise RuntimeError("v6 development has exactly one scientific candidate")
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    context = _context_with_terminal_bridge(base_context, candidates[0])
    metrics, detail = v5run._logging_mixture_fidelity_v5(
        context, seed=seed, protocol=protocol
    )
    row = _k0_row(base_context, context, candidates[0], metrics, detail)
    return {
        "seed": seed,
        "dataset": DATASET,
        "phase": "development_k0",
        "development_lineage": lineage,
        "candidate_count": 1,
        "selector_opened": False,
        "grid_opened": False,
        **row,
        "split_audit": v2._split_audit(base_context.splits),
        "coverage_generated": False,
    }


def _confirmation_support_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    candidates: tuple[TerminalBridgeTheta, ...],
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    if candidates:
        raise RuntimeError("v6 support must remain outcome-bridge blind")
    result = v2._support_worker(seed, preset, device, protocol)
    return {
        **result,
        "phase": "confirmation_support",
        "coverage_generated": False,
        "confirmation_label": "fresh_split_terminal_gate",
    }


def _confirmation_k0_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    candidates: tuple[TerminalBridgeTheta, ...],
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    if candidates != (terminal_candidate(),):
        raise RuntimeError("v6 confirmation requires the frozen R01 bridge")
    base_context = v2._prepare_extension_context(seed, preset, device, protocol)
    context = _context_with_terminal_bridge(base_context, candidates[0])
    metrics, detail = v5run._logging_mixture_fidelity_v5(
        context, seed=seed, protocol=protocol
    )
    row = _k0_row(base_context, context, candidates[0], metrics, detail)
    return {
        "seed": seed,
        "dataset": DATASET,
        "phase": "confirmation_k0",
        **row,
        "split_audit": v2._split_audit(base_context.splits),
        "coverage_generated": False,
        "confirmation_label": "fresh_split_terminal_gate",
        "independent_patient_confirmation_claimed": False,
    }


def _k0_row(
    base_context: v2.ExtensionContext,
    context: v2.ExtensionContext,
    theta: TerminalBridgeTheta,
    metrics: object,
    detail: Mapping[str, Any],
) -> dict[str, Any]:
    metric_payload = asdict(metrics)
    ratio = normalized_seed_ratio(metric_payload)
    passed = bool(metric_payload["structural_invariants"]) and all(
        float(metric_payload[name]) <= threshold
        for name, threshold in K0_THRESHOLDS.items()
    )
    return {
        "theta": theta.to_dict(),
        "metrics": metric_payload,
        "passed": passed,
        "normalized_seed_ratio": ratio if math.isfinite(ratio) else None,
        "structural_failure_ratio_is_infinite": not math.isfinite(ratio),
        "systematic_replay": dict(detail),
        "context_identity": _candidate_context_identity(
            base_context, context.environment, theta
        ),
    }


def _context_with_terminal_bridge(
    base_context: v2.ExtensionContext,
    theta: TerminalBridgeTheta,
) -> v2.ExtensionContext:
    if (
        base_context.config.data.dataset != DATASET
        or base_context.n_actions != 3
        or base_context.splits.environment.horizon != HORIZON
    ):
        raise RuntimeError("v6 context requires six-stage, three-action MIMIC-CXR")
    environment_scores = score_batch(
        base_context.region,
        base_context.splits.environment.current_states(),
        base_context.splits.environment.actions,
        base_context.splits.environment.outcomes,
    )
    environment = build_terminal_environment(
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


def _candidate_context_identity(
    base_context: v2.ExtensionContext,
    environment: object,
    theta: TerminalBridgeTheta,
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
        raise RuntimeError("v6 C13 full-cell sentinel no longer exceeds donor cells")
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
    identity = {
        "base_nuisance_context_sha256": base["combined_sha256"],
        "outcome_model_state_sha256": base["outcome_model_state_sha256"],
        "behavior_policy_state_sha256": base["behavior_policy_state_sha256"],
        "split_patient_id_sha256": base["split_patient_id_sha256"],
        "active_config_sha256": base["active_config_sha256"],
        "theta": theta.to_dict(),
        "state_kernel": state_kernel,
        "outcome_bridge": environment.bridge_identity(),
    }
    return {**identity, "combined_sha256": _json_sha256(identity)}


def _lineage_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 20:
        raise RuntimeError("each v6 development lineage requires all 20 seeds")
    ratios = tuple(numeric_seed_ratio(row["metrics"]) for row in rows)
    structural = tuple(bool(row["metrics"]["structural_invariants"]) for row in rows)
    all_finite = all(math.isfinite(value) for value in ratios)
    numeric = np.asarray(ratios, dtype=np.float64) if all_finite else None
    return {
        "seed_count": 20,
        "pass_count": sum(value <= 1.0 for value in ratios),
        "structural_pass_count": sum(structural),
        "q95_seed_ratio": (
            float(np.quantile(numeric, 0.95, method="linear"))
            if numeric is not None
            else None
        ),
        "mean_seed_ratio": float(numeric.mean()) if numeric is not None else None,
        "seed_ratios": [value if math.isfinite(value) else None for value in ratios],
        "structural_pass_flags": list(structural),
        "nonfinite_numeric_ratio_present": not all_finite,
    }


def _development_gate(
    lineage_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if set(lineage_rows) != set(DEVELOPMENT_LINEAGES):
        raise RuntimeError("v6 development lineages differ")
    summaries = {
        lineage: _lineage_summary(lineage_rows[lineage])
        for lineage in DEVELOPMENT_LINEAGES
    }
    admissible = all(
        summary["pass_count"] == DEVELOPMENT_REQUIRED_PASS_COUNT
        and summary["structural_pass_count"] == REQUIRED_STRUCTURAL_PASS_COUNT
        for summary in summaries.values()
    )
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "DEVELOPMENT_GATE_GO" if admissible else "DEVELOPMENT_GATE_NO_GO",
        "development_admissible": admissible,
        "scientific_candidate": terminal_candidate().to_dict(),
        "scientific_candidate_count": 1,
        "selector_present": False,
        "grid_present": False,
        "b02_anchor_role": "regression_only_not_a_candidate",
        "lineage_summaries": summaries,
        "required_numeric_pass_count_per_lineage": DEVELOPMENT_REQUIRED_PASS_COUNT,
        "required_structural_pass_count_per_lineage": REQUIRED_STRUCTURAL_PASS_COUNT,
        "candidate_seed_deletions": 0,
        "coverage_generated": False,
        "terminal_no_v7": True,
    }


def _development_final(gate: Mapping[str, Any]) -> dict[str, Any]:
    admissible = bool(gate["development_admissible"])
    return {
        "protocol": PROTOCOL,
        "phase": "development",
        "dataset": DATASET,
        "status": "DEVELOPMENT_GO" if admissible else "DEVELOPMENT_NO_GO",
        "development_admissible": admissible,
        "gate": dict(gate),
        "candidate_seed_deletions": 0,
        "coverage_generated": False,
        "information_firewall_respected": True,
        "further_bridge_repair_permitted": False,
        "terminal_no_v7": True,
    }


def _frozen_settings(
    final: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    admissible = bool(final["development_admissible"])
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "development_admissible": admissible,
        "theta": terminal_candidate().to_dict() if admissible else None,
        "development_source_tree_sha256": metadata["source_tree_sha256"],
        "development_config_sha256": metadata["config_sha256"],
        "parent_v5_binding_sha256": metadata["parent_v5_binding_sha256"],
        "development_gate_sha256": _json_sha256(final["gate"]),
        "candidate_seed_deletions": 0,
        "coverage_generated": False,
        "terminal_no_v7": True,
    }


def _confirmation_gate(
    theta: TerminalBridgeTheta,
    support_rows: Sequence[Mapping[str, Any]],
    k0_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(support_rows) != 20:
        raise RuntimeError("v6 confirmation support requires all 20 fresh seeds")
    support_count = sum(bool(row["passed"]) for row in support_rows)
    if support_count >= CONFIRMATION_MINIMUM_PASS_COUNT and len(k0_rows) != 20:
        raise RuntimeError("v6 confirmation K0 requires all 20 fresh seeds")
    if support_count < CONFIRMATION_MINIMUM_PASS_COUNT and k0_rows:
        raise RuntimeError("v6 K0 opened after support NO-GO")
    k0_count = sum(bool(row["passed"]) for row in k0_rows)
    structural_count = sum(
        bool(row["metrics"]["structural_invariants"]) for row in k0_rows
    )
    confirmed = (
        support_count >= CONFIRMATION_MINIMUM_PASS_COUNT
        and k0_count >= CONFIRMATION_MINIMUM_PASS_COUNT
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
        "independent_patient_confirmation_claimed": False,
        "coverage_generated": False,
        "further_bridge_repair_permitted": False,
        "terminal_no_v7": True,
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
        "further_bridge_repair_permitted": False,
        "terminal_no_v7": True,
    }


def validate_parent_v5_bundles(config: FidelityV6Config) -> dict[str, Any]:
    v5_config = load_fidelity_v5_config(V5_CONFIG_PATH)
    bindings = {}
    for label, root_binding in (
        ("development", config.parent_development),
        ("failed_confirmation", config.parent_failed_confirmation),
    ):
        root = (
            root_binding.root
            if root_binding.root.is_absolute()
            else ROOT / root_binding.root
        )
        for relative, expected_hash in root_binding.file_sha256.items():
            path = root / relative
            if not path.is_file() or _file_sha256(path) != expected_hash:
                raise RuntimeError(f"frozen v5 {label} binding differs: {relative}")
        metadata = _read_json(root / "metadata.json")
        v5run._validate_root_bundle(root, metadata, config=v5_config)
        bindings[label] = {
            "root": root.relative_to(ROOT).as_posix(),
            "manifest_sha256": _file_sha256(root / "manifest.json"),
            "complete_sha256": _file_sha256(root / "COMPLETE"),
            "metadata_sha256": _file_sha256(root / "metadata.json"),
            "full_semantic_bundle_validation": True,
        }

    development_root = ROOT / config.parent_development.root
    development_final = _read_json(development_root / "FINAL_STATUS.json")
    failed_root = ROOT / config.parent_failed_confirmation.root
    failed_final = _read_json(failed_root / "FINAL_STATUS.json")
    failed_gate = failed_final["gate"]
    if (
        development_final["status"] != "DEVELOPMENT_GO"
        or development_final["winner"]["candidate_id"]
        != "B02_pooled_successor_bridge_stage_one_hot"
        or development_final["development_pass_count"] != 19
        or development_final["development_structural_pass_count"] != 20
        or failed_final["status"] != "CONFIRMATION_COMPLETE_NO_GO"
        or failed_gate["support_pass_count"] != 20
        or failed_gate["k0_pass_count"] != 18
        or failed_gate["structural_pass_count"] != 20
        or development_final["coverage_generated"] is not False
        or failed_final["coverage_generated"] is not False
    ):
        raise RuntimeError("v5 development/failed-confirmation decision differs")

    failures = []
    expected_failures = {
        119_120: (1, 0, 0.2956583463636382),
        119_180: (3, 0, 0.3200767965422734),
    }
    for seed in sorted(expected_failures):
        payload = _read_json(failed_root / "k0_fidelity" / f"seed_{seed:06d}.json")
        result = payload["result"]
        stage, outcome, expected_value = expected_failures[seed]
        observed = result["systematic_replay"]["signed_residual_w1_by_stage_outcome"][
            stage
        ][outcome]
        if (
            result["passed"] is not False
            or result["metrics"]["structural_invariants"] is not True
            or observed != expected_value
        ):
            raise RuntimeError("v5 public failed-seed record differs")
        failures.append(
            {
                "seed": seed,
                "stage": stage,
                "outcome": outcome,
                "maximum_signed_residual_w1": observed,
            }
        )
    observed_failed = {
        int(path.stem.split("_")[1])
        for path in (failed_root / "k0_fidelity").glob("seed_*.json")
        if not _read_json(path)["result"]["passed"]
    }
    if observed_failed != set(expected_failures):
        raise RuntimeError("v5 failed confirmation seed set differs")

    binding = {
        "protocol": PROTOCOL,
        "development": bindings["development"],
        "failed_confirmation": bindings["failed_confirmation"],
        "v5_source_tree_sha256": "bf7ff256327c76130ff93b626d4c414748b0e2cb9be0d269aca332d0236c893a",
        "b02_anchor_role": "regression_only_not_a_candidate",
        "failed_confirmation_reclassified_as_development_only": True,
        "scientific_freshness_claimed": False,
        "public_failed_confirmation": {
            "support_pass_count": 20,
            "k0_pass_count": 18,
            "structural_pass_count": 20,
            "failed_seeds": failures,
        },
        "coverage_generated": False,
    }
    return {**binding, "combined_sha256": _json_sha256(binding)}


def audit_development_reuse(
    config: FidelityV6Config,
    *,
    parent_binding: Mapping[str, Any],
) -> dict[str, Any]:
    del parent_binding
    mapping = _development_reuse_mapping()
    values = set(mapping.values())
    seeds = tuple(seed for lineage in DEVELOPMENT_LINEAGES.values() for seed in lineage)
    if (
        len(mapping) != 200
        or len(values) != 200
        or _json_sha256(mapping) != DEVELOPMENT_MAPPING_SHA256
        or _integer_set_sha256(values) != DEVELOPMENT_ID_SET_SHA256
        or _integer_set_sha256(seeds) != DEVELOPMENT_BASE_SET_SHA256
    ):
        raise RuntimeError("v6 development RNG reuse mapping differs")

    v5_development = _read_json(
        ROOT / config.parent_development.root / "metadata.json"
    )["development_rng_reuse_audit"]["mapping"]
    v5_confirmation = _read_json(
        ROOT / config.parent_failed_confirmation.root / "metadata.json"
    )["confirmation_rng_audit"]["new_rng_stream_mapping"]
    for label, value in mapping.items():
        parent = v5_development if "base_92" in label else v5_confirmation
        if parent.get(label) != value:
            raise RuntimeError(f"v6 development stream is not exact v5 reuse: {label}")
    return {
        "status": "passed_before_launch",
        "role": "exact_authorized_reuse_of_two_exposed_v5_lineages",
        "base_seed_count": len(seeds),
        "stream_count": len(mapping),
        "mapping": mapping,
        "mapping_sha256": _json_sha256(mapping),
        "rng_id_set_sha256": _integer_set_sha256(values),
        "base_seed_set_sha256": _integer_set_sha256(seeds),
        "authorized_lineage_collision_count": len(mapping),
        "unauthorized_collision_count": 0,
        "common_random_numbers_with_v5": True,
        "scientific_freshness_claimed": False,
        "formal_rng_consumed": False,
    }


def _development_reuse_mapping() -> dict[str, int]:
    mapping = {}
    for lineage in DEVELOPMENT_LINEAGES.values():
        for seed in lineage:
            prefix = f"{DATASET}/base_{seed}"
            mapping[f"{prefix}/task"] = seed
            mapping[f"{prefix}/outcome_model"] = seed + 1
            mapping[f"{prefix}/behavior_model"] = seed + 2
            mapping[f"{prefix}/cxr_encoder"] = seed + 701
            mapping[f"{prefix}/k0_base_uniform"] = v2.K0_UNIFORM_SEED_OFFSET + seed
    return mapping


def v6_prelaunch_source_contract() -> dict[str, Any]:
    """Bind the audited v6 files and normalized executable experiment tree."""

    paths = [
        *sorted((ROOT / "src/scpcp").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "tools").glob("*.py")),
        *sorted((ROOT / "configs").glob("*.yaml")),
        ROOT / "pyproject.toml",
    ]
    if (
        len(paths) != len({path.resolve() for path in paths})
        or any(not path.is_file() or path.is_symlink() for path in paths)
        or any(
            not path.is_file() or path.is_symlink()
            for path in _V6_SOURCE_CONTRACT_PATHS
        )
    ):
        raise RuntimeError("v6 source contract file set is invalid")

    def content(path: Path) -> tuple[bytes, str]:
        if path.resolve() != CONFIG_PATH.resolve():
            return path.read_bytes(), "raw_bytes"
        payload = yaml.safe_load(path.read_text())
        payload["independent_audit"] = {
            key: "__INDEPENDENT_AUDIT_DYNAMIC__"
            for key in sorted(payload["independent_audit"])
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return encoded, "canonical_yaml_with_independent_audit_values_normalized"

    exact_files = []
    for path in _V6_SOURCE_CONTRACT_PATHS:
        raw, normalization = content(path)
        exact_files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "normalization": normalization,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )

    digest = hashlib.sha256()
    for path in paths:
        raw, _ = content(path)
        relative = path.relative_to(ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    contract = {
        "protocol": PROTOCOL,
        "role": "independent_prelaunch_normalized_source_contract",
        "exact_v6_file_count": len(exact_files),
        "exact_v6_files": exact_files,
        "normalized_experiment_tree_file_count": len(paths),
        "normalized_experiment_tree_sha256": digest.hexdigest(),
    }
    if contract["exact_v6_file_count"] != 5:
        raise RuntimeError(
            "v6 isolated source contract must contain exactly five files"
        )
    return {**contract, "combined_sha256": _json_sha256(contract)}


def audit_confirmation_rng(config: FidelityV6Config) -> dict[str, Any]:
    source_contract = v6_prelaunch_source_contract()
    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    preset = replace(
        protocol.datasets[DATASET],
        seeds=config.confirmation_seeds,
        bootstrap_seed=config.confirmation_bootstrap_seed,
    )
    active_protocol = replace(protocol, datasets={**protocol.datasets, DATASET: preset})
    mapping = v2._new_rng_stream_mapping(active_protocol, (DATASET,))
    v2._assert_unique_rng_streams(mapping)
    values = set(mapping.values())
    if (
        len(mapping) != 341
        or _json_sha256(mapping) != CONFIRMATION_MAPPING_SHA256
        or _integer_set_sha256(values) != CONFIRMATION_ID_SET_SHA256
        or _integer_set_sha256(config.confirmation_seeds)
        != CONFIRMATION_BASE_SET_SHA256
    ):
        raise RuntimeError("v6 confirmation derived RNG mapping differs")

    excluded_roots = (DEVELOPMENT_ROOT, CONFIRMATION_ROOT)
    artifact_ids = v5run._artifact_rng_ids(
        ROOT / "results", excluded_roots=excluded_roots
    )
    source_ids = v2._source_declared_seeds(
        ROOT, excluded_paths=_OWN_RNG_DECLARATION_PATHS
    )
    prior = artifact_ids | source_ids
    collisions = {label: value for label, value in mapping.items() if value in prior}
    if v6_prelaunch_source_contract() != source_contract:
        raise RuntimeError("source changed during the v6 prelaunch audit")
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
        "new_rng_id_set_sha256": _integer_set_sha256(values),
        "confirmation_base_seed_count": len(config.confirmation_seeds),
        "confirmation_base_seed_set_sha256": _integer_set_sha256(
            config.confirmation_seeds
        ),
        "internal_rng_streams_unique": len(values) == len(mapping),
        "excluded_roots": [str(path) for path in excluded_roots],
        "formal_rng_consumed": False,
        "v6_source_contract": source_contract,
        "v6_source_contract_sha256": source_contract["combined_sha256"],
    }
    if collisions:
        result["proposed_independent_audit"] = None
    else:
        proposed = IndependentAudit(
            status="GO",
            attestation_sha256=None,
            expected_prior_count=result["prior_rng_id_count"],
            expected_prior_sha256=result["prior_rng_id_sha256"],
            expected_artifact_count=result["artifact_rng_id_count"],
            expected_artifact_sha256=result["artifact_rng_id_sha256"],
            expected_source_count=result["source_declared_rng_id_count"],
            expected_source_sha256=result["source_declared_rng_id_sha256"],
            expected_v6_source_contract_sha256=result["v6_source_contract_sha256"],
        )
        result["proposed_independent_audit"] = {
            **asdict(proposed),
            "attestation_sha256": independent_audit_attestation_sha256(proposed),
        }
    return result


def _assert_formal_roots_absent() -> None:
    present = [
        str(path) for path in (DEVELOPMENT_ROOT, CONFIRMATION_ROOT) if path.exists()
    ]
    if present:
        raise RuntimeError(
            f"v6 prelaunch audit requires absent formal roots: {present}"
        )


def _validate_frozen_audit_snapshot(
    config: FidelityV6Config, audit: Mapping[str, Any]
) -> None:
    frozen = config.independent_audit
    current_source_contract = v6_prelaunch_source_contract()
    if (
        audit.get("v6_source_contract") != current_source_contract
        or audit.get("v6_source_contract_sha256")
        != current_source_contract["combined_sha256"]
    ):
        raise RuntimeError("v6 independently audited source contract differs")
    observed = (
        audit["prior_rng_id_count"],
        audit["prior_rng_id_sha256"],
        audit["artifact_rng_id_count"],
        audit["artifact_rng_id_sha256"],
        audit["source_declared_rng_id_count"],
        audit["source_declared_rng_id_sha256"],
        audit["v6_source_contract_sha256"],
    )
    expected = (
        frozen.expected_prior_count,
        frozen.expected_prior_sha256,
        frozen.expected_artifact_count,
        frozen.expected_artifact_sha256,
        frozen.expected_source_count,
        frozen.expected_source_sha256,
        frozen.expected_v6_source_contract_sha256,
    )
    if audit["collision_count"] != 0 or observed != expected:
        raise RuntimeError("v6 independent RNG audit snapshot differs")


def _run_seed_phase(
    phase_root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    devices: tuple[str, ...],
    candidates: tuple[TerminalBridgeTheta, ...],
    worker: Callable[..., dict[str, Any]],
    worker_arguments: tuple[object, ...],
    source_hash: str,
) -> list[dict[str, Any]]:
    phase_root.mkdir(parents=True, exist_ok=True)
    mapping = _seed_device_mapping(preset.seeds, devices)
    candidate_hash = _json_sha256([candidate.to_dict() for candidate in candidates])
    expected = {phase_root / f"seed_{seed:06d}.json" for seed in preset.seeds}
    unexpected = set(phase_root.glob("seed_*.json")) - expected
    if unexpected:
        raise RuntimeError(f"unexpected {phase} artifacts: {sorted(unexpected)}")

    completed = {}
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
    candidates: tuple[TerminalBridgeTheta, ...],
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
    candidates: tuple[TerminalBridgeTheta, ...],
) -> None:
    if (
        set(payload)
        != {
            "protocol",
            "phase",
            "dataset",
            "seed",
            "device",
            "source_tree_sha256",
            "candidate_contract_sha256",
            "result",
        }
        or payload["protocol"] != PROTOCOL
        or payload["phase"] != phase
        or payload["dataset"] != DATASET
        or payload["seed"] != seed
        or payload["device"] != device
        or payload["source_tree_sha256"] != source_hash
        or payload["candidate_contract_sha256"] != candidate_hash
    ):
        raise RuntimeError(f"invalid v6 seed envelope for {seed}")
    result = payload["result"]
    if (
        not isinstance(result, Mapping)
        or result.get("seed") != seed
        or result.get("dataset") != DATASET
        or result.get("phase") != phase
        or result.get("coverage_generated") is not False
    ):
        raise RuntimeError(f"invalid v6 seed result for {seed}")
    _assert_seed_result_firewall(result)
    if phase == "confirmation_support":
        if set(result) != {
            "seed",
            "dataset",
            "phase",
            "outcome_blind",
            "passed",
            "minimum_unique_patients",
            "failed_cells",
            "unique_patient_counts_by_stage_action",
            "n_actions",
            "active_action_indices",
            "action_mapping",
            "action_costs",
            "environment_episode_support",
            "split_audit",
            "coverage_generated",
            "confirmation_label",
        }:
            raise RuntimeError(f"invalid v6 support result schema for {seed}")
        support = {**result, "phase": "support"}
        support.pop("confirmation_label", None)
        if not v2._valid_support_result(support, preset):
            raise RuntimeError(f"invalid v6 support result for {seed}")
        return
    if (
        candidates != (terminal_candidate(),)
        or result.get("theta") != candidates[0].to_dict()
    ):
        raise RuntimeError(f"invalid v6 R01 candidate for {seed}")
    if phase == "development_k0":
        expected_keys = {
            "seed",
            "dataset",
            "phase",
            "development_lineage",
            "candidate_count",
            "selector_opened",
            "grid_opened",
            "theta",
            "metrics",
            "passed",
            "normalized_seed_ratio",
            "structural_failure_ratio_is_infinite",
            "systematic_replay",
            "context_identity",
            "split_audit",
            "coverage_generated",
        }
        expected_lineage = next(
            (
                lineage
                for lineage, seeds in DEVELOPMENT_LINEAGES.items()
                if seed in seeds
            ),
            None,
        )
        if (
            set(result) != expected_keys
            or result.get("development_lineage") != expected_lineage
            or result.get("candidate_count") != 1
            or result.get("selector_opened") is not False
            or result.get("grid_opened") is not False
        ):
            raise RuntimeError(f"invalid v6 development role for {seed}")
    elif phase == "confirmation_k0":
        expected_keys = {
            "seed",
            "dataset",
            "phase",
            "theta",
            "metrics",
            "passed",
            "normalized_seed_ratio",
            "structural_failure_ratio_is_infinite",
            "systematic_replay",
            "context_identity",
            "split_audit",
            "coverage_generated",
            "confirmation_label",
            "independent_patient_confirmation_claimed",
        }
        if (
            set(result) != expected_keys
            or result.get("confirmation_label") != "fresh_split_terminal_gate"
            or result.get("independent_patient_confirmation_claimed") is not False
        ):
            raise RuntimeError(f"invalid v6 confirmation role for {seed}")
    else:
        raise RuntimeError("unknown v6 seed phase")
    if (
        set(result["systematic_replay"]) != _K0_DETAIL_KEYS
        or set(result["context_identity"]) != _CONTEXT_IDENTITY_KEYS
        or set(result["split_audit"]) != _SPLIT_AUDIT_KEYS
    ):
        raise RuntimeError(f"invalid v6 nested K0 schema for {seed}")
    v5run._validate_k0_candidate_row(result)


def _assert_seed_result_firewall(value: object, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower()
            matched = any(
                token in name
                for token in ("science", "coverage", "width", "method_selection")
            )
            allowed = (name == "coverage_generated" and item is False) or name in {
                "bandwidth",
                "design_width",
            }
            if matched and not allowed:
                location = ".".join((*path, str(key)))
                raise RuntimeError(f"forbidden v6 seed-result content: {location}")
            _assert_seed_result_firewall(item, path=(*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_seed_result_firewall(item, path=(*path, str(index)))


def _root_metadata(
    *,
    phase: str,
    output_root: Path,
    config: FidelityV6Config,
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
    if phase == "development":
        seed_to_device = {
            lineage: {
                str(seed): device
                for seed, device in _seed_device_mapping(seeds, devices).items()
            }
            for lineage, seeds in config.development_lineages.items()
        }
    else:
        seed_to_device = {
            str(seed): device
            for seed, device in _seed_device_mapping(
                config.confirmation_seeds, devices
            ).items()
        }
    metadata = {
        "protocol": PROTOCOL,
        "phase": phase,
        "role": "terminal_coverage_blind_mimic_cxr_outcome_bridge_repair",
        "dataset": DATASET,
        "devices": list(devices),
        "output_root": str(output_root),
        "seed_to_device": seed_to_device,
        "source_tree_sha256": source_hash,
        "source_snapshot": dict(source_snapshot),
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_bytes": len(config_bytes),
        "parent_v5_binding": dict(parent_binding),
        "parent_v5_binding_sha256": _json_sha256(parent_binding),
        "development_rng_reuse_audit": dict(development_audit),
        "confirmation_rng_audit": dict(confirmation_audit),
        "independent_audit": asdict(config.independent_audit),
        "scientific_candidate": terminal_candidate().to_dict(),
        "scientific_candidate_count": 1,
        "selector_present": False,
        "grid_present": False,
        "b02_anchor_role": "regression_only_not_a_candidate",
        "development_gate_contract": {
            "numeric_pass_count_per_lineage": 20,
            "structural_pass_count_per_lineage": 20,
            "candidate_seed_deletion_permitted": False,
        },
        "confirmation_gate_contract": {
            "support_minimum_pass_count": 19,
            "k0_minimum_pass_count": 19,
            "required_structural_pass_count": 20,
            "candidate_seed_deletion_permitted": False,
        },
        "coverage_generation_permitted": False,
        "scientific_result_execution_path_present": False,
        "canonical_scpcp_mutation_permitted": False,
        "further_bridge_repair_permitted": False,
        "terminal_no_v7": True,
    }
    if phase == "confirmation":
        metadata["development_binding"] = dict(development_binding or {})
        metadata["development_binding_sha256"] = _json_sha256(development_binding)
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
            raise RuntimeError("v6 resume metadata differs")
        v5run.v4._verify_source_snapshot(root, metadata["source_snapshot"])
        return
    if root.exists():
        raise FileExistsError(f"fresh v6 output already exists: {root}")
    root.mkdir(parents=True)
    v5run.v4._atomic_write(
        root / source_snapshot["contract"]["archive_path"],
        source_snapshot["archive_bytes"],
    )
    v5run.v4._atomic_write(
        root / source_snapshot["contract"]["manifest_path"],
        source_snapshot["manifest_bytes"],
    )
    _write_json(root / "metadata.json", metadata)
    v5run.v4._verify_source_snapshot(root, metadata["source_snapshot"])


def _validate_incomplete_resume_root(
    root: Path,
    *,
    phase: str,
    metadata: Mapping[str, Any],
    config: FidelityV6Config,
) -> None:
    """Validate every recoverable partial artifact before any worker starts."""

    if (root / "COMPLETE").exists():
        raise RuntimeError("completed v6 root must use the full bundle validator")
    _assert_no_symlinks(root)
    _assert_no_forbidden_result_paths(root)
    for path in root.rglob("*"):
        if not path.is_file() and not path.is_dir():
            raise RuntimeError(f"non-regular v6 partial artifact: {path}")
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("v6 partial-root metadata differs")

    snapshot = metadata["source_snapshot"]
    base_files = {
        Path("metadata.json"),
        Path(snapshot["archive_path"]),
        Path(snapshot["manifest_path"]),
    }
    allowed_directories = {Path("provenance")}
    terminal_files = {Path("manifest.json"), Path("FINAL_STATUS.json")}
    terminal_expected: dict[Path, Mapping[str, Any]] = {}
    terminal_order: tuple[Path, ...] = ()
    terminal_ready = False
    protocol = v2.load_extension_config(V2_CONFIG_PATH)

    if phase == "development":
        terminal_files.update(
            {Path("development_gate.json"), Path("frozen_settings.json")}
        )
        allowed_directories.add(Path("k0_fidelity"))
        allowed_files = set(base_files)
        completion = {}
        lineage_rows = {}
        for lineage, seeds in config.development_lineages.items():
            relative_root = Path("k0_fidelity") / lineage
            allowed_directories.add(relative_root)
            preset = replace(protocol.datasets[DATASET], seeds=seeds)
            phase_files = {
                relative_root / f"seed_{seed:06d}.json" for seed in seeds
            } | {relative_root / "COMPLETE"}
            allowed_files.update(phase_files)
            lineage_rows[lineage], completion[lineage] = _validate_partial_seed_phase(
                root / relative_root,
                phase="development_k0",
                preset=preset,
                devices=tuple(metadata["devices"]),
                candidates=(terminal_candidate(),),
                source_hash=metadata["source_tree_sha256"],
            )
        second_root = root / "k0_fidelity" / "v5_failed_confirmation"
        if second_root.exists() and not completion["v5_development"]:
            raise RuntimeError(
                "v6 second development lineage opened before the first completed"
            )
        terminal_ready = all(completion.values())
        if terminal_ready:
            gate = _development_gate(lineage_rows)
            final = _development_final(gate)
            terminal_expected = {
                Path("development_gate.json"): gate,
                Path("FINAL_STATUS.json"): final,
                Path("frozen_settings.json"): _frozen_settings(final, metadata),
            }
            terminal_order = (
                Path("development_gate.json"),
                Path("FINAL_STATUS.json"),
                Path("frozen_settings.json"),
                Path("manifest.json"),
            )
    elif phase == "confirmation":
        terminal_files.add(Path("gate.json"))
        allowed_directories.add(Path("support"))
        preset = replace(
            protocol.datasets[DATASET],
            seeds=config.confirmation_seeds,
            bootstrap_seed=config.confirmation_bootstrap_seed,
        )
        support_files = {
            Path("support") / f"seed_{seed:06d}.json"
            for seed in config.confirmation_seeds
        } | {Path("support/COMPLETE")}
        allowed_files = base_files | support_files
        support_rows, support_complete = _validate_partial_seed_phase(
            root / "support",
            phase="confirmation_support",
            preset=preset,
            devices=tuple(metadata["devices"]),
            candidates=(),
            source_hash=metadata["source_tree_sha256"],
        )
        k0_root = root / "k0_fidelity"
        k0_rows: list[dict[str, Any]] = []
        k0_complete = False
        support_count = sum(bool(row["passed"]) for row in support_rows)
        if k0_root.exists():
            if not support_complete:
                raise RuntimeError("v6 K0 opened before support completed")
            if support_count < CONFIRMATION_MINIMUM_PASS_COUNT:
                raise RuntimeError("v6 K0 exists after support NO-GO")
            allowed_directories.add(Path("k0_fidelity"))
            k0_files = {
                Path("k0_fidelity") / f"seed_{seed:06d}.json"
                for seed in config.confirmation_seeds
            } | {Path("k0_fidelity/COMPLETE")}
            allowed_files.update(k0_files)
            theta = _theta_from_dict(metadata["frozen_settings"]["theta"])
            k0_rows, k0_complete = _validate_partial_seed_phase(
                k0_root,
                phase="confirmation_k0",
                preset=preset,
                devices=tuple(metadata["devices"]),
                candidates=(theta,),
                source_hash=metadata["source_tree_sha256"],
            )
        terminal_ready = support_complete and (
            support_count < CONFIRMATION_MINIMUM_PASS_COUNT or k0_complete
        )
        if terminal_ready:
            theta = _theta_from_dict(metadata["frozen_settings"]["theta"])
            gate = _confirmation_gate(theta, support_rows, k0_rows)
            terminal_expected = {
                Path("gate.json"): gate,
                Path("FINAL_STATUS.json"): _confirmation_final(gate),
            }
            terminal_order = (
                Path("gate.json"),
                Path("FINAL_STATUS.json"),
                Path("manifest.json"),
            )
    else:
        raise RuntimeError("unknown v6 partial-root phase")

    observed_files = {
        path.relative_to(root) for path in root.rglob("*") if path.is_file()
    }
    present_terminal = observed_files & terminal_files
    if present_terminal and not terminal_ready:
        raise RuntimeError(
            f"premature v6 terminal artifacts in incomplete root: {present_terminal}"
        )
    if terminal_ready:
        allowed_files.update(terminal_files)
        _validate_partial_terminal_fragments(
            root,
            expected=terminal_expected,
            order=terminal_order,
        )
    unexpected_files = observed_files - allowed_files
    if unexpected_files:
        raise RuntimeError(f"unexpected v6 partial-root files: {unexpected_files}")
    if not base_files <= observed_files:
        raise RuntimeError("v6 partial root is missing metadata/source provenance")
    observed_directories = {
        path.relative_to(root) for path in root.rglob("*") if path.is_dir()
    }
    unexpected_directories = observed_directories - allowed_directories
    if unexpected_directories:
        raise RuntimeError(
            f"unexpected v6 partial-root directories: {unexpected_directories}"
        )


def _validate_partial_seed_phase(
    phase_root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    devices: tuple[str, ...],
    candidates: tuple[TerminalBridgeTheta, ...],
    source_hash: str,
) -> tuple[list[dict[str, Any]], bool]:
    if not phase_root.exists():
        return [], False
    if not phase_root.is_dir() or phase_root.is_symlink():
        raise RuntimeError(f"invalid v6 partial phase root: {phase_root}")
    mapping = _seed_device_mapping(preset.seeds, devices)
    candidate_hash = _json_sha256([candidate.to_dict() for candidate in candidates])
    rows = []
    observed_seeds = set()
    for seed in preset.seeds:
        path = phase_root / f"seed_{seed:06d}.json"
        if not path.exists():
            continue
        try:
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
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise RuntimeError(
                f"invalid existing v6 {phase} seed payload: {seed}"
            ) from error
        observed_seeds.add(seed)
        rows.append(payload["result"])
    complete = phase_root / "COMPLETE"
    if complete.exists():
        if complete.read_text() != "complete\n" or observed_seeds != set(preset.seeds):
            raise RuntimeError(f"invalid incomplete-root {phase} COMPLETE marker")
        return rows, True
    return rows, False


def _validate_partial_terminal_fragments(
    root: Path,
    *,
    expected: Mapping[Path, Mapping[str, Any]],
    order: tuple[Path, ...],
) -> None:
    """Accept only an exact prefix of the deterministic terminal commit sequence."""

    manifest_path = Path("manifest.json")
    if not order or order[-1] != manifest_path or set(expected) != set(order[:-1]):
        raise RuntimeError("invalid internal v6 terminal-fragment contract")

    missing_predecessor = False
    for relative in order:
        path = root / relative
        if not path.exists():
            missing_predecessor = True
            continue
        if missing_predecessor:
            raise RuntimeError("v6 terminal fragments are out of commit order")
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"invalid v6 terminal fragment: {relative}")
        if relative == manifest_path:
            try:
                _verify_manifest(root)
            except (OSError, TypeError, ValueError, KeyError) as error:
                raise RuntimeError("invalid partial v6 manifest") from error
            continue
        try:
            observed = _read_json(path)
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise RuntimeError(f"invalid v6 terminal fragment: {relative}") from error
        if observed != expected[relative]:
            raise RuntimeError(f"stale v6 terminal fragment: {relative}")


def _complete_and_valid(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV6Config,
) -> bool:
    if not (root / "COMPLETE").exists():
        return False
    try:
        _validate_root_bundle(root, metadata, config=config)
    except (Exception, KeyboardInterrupt):
        v5run.v4._unlink_root_complete(root)
        raise
    return True


def _finalize_root(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    source_hash: str,
    config: FidelityV6Config,
) -> None:
    if experiment_tree_sha256() != source_hash:
        raise RuntimeError("source/config changed during the formal v6 phase")
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("v6 metadata changed during the formal phase")
    if validate_parent_v5_bundles(config) != metadata["parent_v5_binding"]:
        raise RuntimeError("v5 parent evidence changed during v6")
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
        v5run.v4._unlink_root_complete(root)
        raise


def _validate_root_bundle(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV6Config,
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
        raise RuntimeError("v6 COMPLETE marker differs")


def _validate_root_contents(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    config: FidelityV6Config,
) -> None:
    if _read_json(root / "metadata.json") != metadata:
        raise RuntimeError("v6 root metadata differs")
    if metadata.get("parent_v5_binding_sha256") != _json_sha256(
        metadata.get("parent_v5_binding")
    ) or validate_parent_v5_bundles(config) != metadata.get("parent_v5_binding"):
        raise RuntimeError("v6 parent binding differs")
    v5run.v4._verify_source_snapshot(root, metadata["source_snapshot"])
    _verify_manifest(root)
    _assert_no_forbidden_result_paths(root)
    protocol = v2.load_extension_config(V2_CONFIG_PATH)
    if metadata["phase"] == "development":
        lineage_rows = {}
        expected = {
            Path("metadata.json"),
            Path("FINAL_STATUS.json"),
            Path("development_gate.json"),
            Path("frozen_settings.json"),
        }
        for lineage, seeds in config.development_lineages.items():
            preset = replace(protocol.datasets[DATASET], seeds=seeds)
            phase_root = root / "k0_fidelity" / lineage
            lineage_rows[lineage] = _load_seed_results(
                phase_root,
                phase="development_k0",
                preset=preset,
                devices=tuple(metadata["devices"]),
                candidates=(terminal_candidate(),),
                source_hash=metadata["source_tree_sha256"],
            )
            expected.add(Path("k0_fidelity") / lineage / "COMPLETE")
            expected.update(
                Path("k0_fidelity") / lineage / f"seed_{seed:06d}.json"
                for seed in seeds
            )
        gate = _development_gate(lineage_rows)
        final = _development_final(gate)
        frozen = _frozen_settings(final, metadata)
        if (
            _read_json(root / "development_gate.json") != gate
            or _read_json(root / "FINAL_STATUS.json") != final
            or _read_json(root / "frozen_settings.json") != frozen
        ):
            raise RuntimeError("v6 development semantic recomputation differs")
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
        if support_count >= CONFIRMATION_MINIMUM_PASS_COUNT:
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
        if _read_json(root / "gate.json") != gate or _read_json(
            root / "FINAL_STATUS.json"
        ) != _confirmation_final(gate):
            raise RuntimeError("v6 confirmation semantic recomputation differs")
    else:
        raise RuntimeError("unknown v6 root phase")
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
    config: FidelityV6Config,
    parent_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _read_json(root / "metadata.json")
    _validate_root_bundle(root, metadata, config=config)
    final = _read_json(root / "FINAL_STATUS.json")
    frozen = _read_json(root / "frozen_settings.json")
    if (
        final["status"] != "DEVELOPMENT_GO"
        or frozen["development_admissible"] is not True
        or frozen["theta"] != terminal_candidate().to_dict()
        or metadata["parent_v5_binding"] != parent_binding
        or metadata["parent_v5_binding_sha256"] != _json_sha256(parent_binding)
    ):
        raise RuntimeError("v6 development did not authorize confirmation")
    binding = {
        "root": root.relative_to(ROOT).as_posix(),
        "complete_sha256": _file_sha256(root / "COMPLETE"),
        "manifest_sha256": _file_sha256(root / "manifest.json"),
        "metadata_sha256": _file_sha256(root / "metadata.json"),
        "final_status_sha256": _file_sha256(root / "FINAL_STATUS.json"),
        "frozen_settings_sha256": _file_sha256(root / "frozen_settings.json"),
        "source_tree_sha256": metadata["source_tree_sha256"],
        "development_gate_sha256": frozen["development_gate_sha256"],
        "full_semantic_bundle_validation": True,
    }
    return {**binding, "combined_sha256": _json_sha256(binding)}, frozen


def _load_seed_results(
    phase_root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    devices: tuple[str, ...],
    candidates: tuple[TerminalBridgeTheta, ...],
    source_hash: str,
) -> list[dict[str, Any]]:
    if (phase_root / "COMPLETE").read_text() != "complete\n":
        raise RuntimeError(f"invalid {phase} COMPLETE marker")
    mapping = _seed_device_mapping(preset.seeds, devices)
    candidate_hash = _json_sha256([candidate.to_dict() for candidate in candidates])
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
            raise RuntimeError(f"symlink forbidden in v6 bundle: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative in {Path("manifest.json"), Path("COMPLETE")}:
            continue
        if ".tmp-" in path.name:
            raise RuntimeError(f"temporary v6 artifact remains: {path}")
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
        {"protocol": PROTOCOL, "artifact_count": len(entries), "artifacts": entries},
    )


def _verify_manifest(root: Path) -> None:
    _assert_no_symlinks(root)
    manifest = _read_json(root / "manifest.json")
    entries = manifest.get("artifacts")
    if (
        set(manifest) != {"protocol", "artifact_count", "artifacts"}
        or manifest.get("protocol") != PROTOCOL
        or not isinstance(entries, list)
    ):
        raise RuntimeError("invalid v6 manifest header")
    expected = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "bytes", "sha256"}:
            raise RuntimeError("malformed v6 manifest entry")
        relative = Path(entry["path"])
        if relative in {Path("manifest.json"), Path("COMPLETE")}:
            raise RuntimeError("v6 manifest contains a root commit file")
        path = _resolve_inside_root(root, relative)
        if path in expected:
            raise RuntimeError("duplicate v6 manifest entry")
        expected.add(path)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != entry["bytes"]
            or _file_sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"v6 manifest mismatch: {path}")
    observed = {
        _resolve_inside_root(root, path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root) not in {Path("manifest.json"), Path("COMPLETE")}
    }
    if observed != expected or manifest["artifact_count"] != len(entries):
        raise RuntimeError("v6 manifest file set differs")


def _assert_exact_artifact_file_set(root: Path, expected: set[Path]) -> None:
    _assert_no_symlinks(root)
    observed = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root) not in {Path("manifest.json"), Path("COMPLETE")}
    }
    if observed != expected:
        raise RuntimeError(
            "v6 exact artifact file set differs; "
            f"missing={sorted(map(str, expected - observed))}; "
            f"extra={sorted(map(str, observed - expected))}"
        )


def _assert_no_symlinks(root: Path) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("symlink forbidden in v6 bundle")


def _assert_no_forbidden_result_paths(root: Path) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(
            token in part.lower()
            for part in relative.parts
            for token in FORBIDDEN_RESULT_PATH_TOKENS
        ):
            raise RuntimeError(f"forbidden result path in v6 bundle: {relative}")


def _resolve_inside_root(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("v6 path escapes its root")
    resolved_root = root.resolve()
    path = (root / relative).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise RuntimeError("v6 path escapes its root")
    return path


def _theta_from_dict(value: Mapping[str, Any]) -> TerminalBridgeTheta:
    theta = TerminalBridgeTheta(candidate_id=str(value["candidate_id"]))
    if theta.to_dict() != dict(value):
        raise ValueError("serialized v6 bridge differs from frozen schema")
    return theta


def _seed_device_mapping(
    seeds: Sequence[int], devices: Sequence[str]
) -> dict[int, str]:
    if not devices:
        raise ValueError("v6 seed/device mapping needs at least one GPU")
    return {seed: devices[index % len(devices)] for index, seed in enumerate(seeds)}


def _validate_devices(devices: Sequence[str]) -> None:
    if len(devices) != 2 or any(not value.startswith("cuda:") for value in devices):
        raise ValueError("formal v6 requires exactly two explicit CUDA devices")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: object) -> None:
    v5run.v4._write_json(path, value)


def _write_text(path: Path, value: str) -> None:
    v5run.v4._write_text(path, value)


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


def _integer_set_sha256(values: Sequence[int] | set[int]) -> str:
    return _json_sha256(sorted(set(int(value) for value in values)))


if __name__ == "__main__":
    main()
