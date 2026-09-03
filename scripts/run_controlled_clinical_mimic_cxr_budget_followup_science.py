"""Run post-unlock science for the MIMIC-CXR calibration-budget follow-up.

The read-only audit must succeed before the science root may be created::

    python scripts/run_controlled_clinical_mimic_cxr_budget_followup_science.py audit
    python scripts/run_controlled_clinical_mimic_cxr_budget_followup_science.py run \
      --audit-go-sha256 <hash>
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_controlled_clinical_extension as extension  # noqa: E402
import scripts.run_controlled_clinical_fidelity_v4 as fidelity_v4  # noqa: E402
import scripts.run_controlled_clinical_fidelity_v5_mimic_cxr as fidelity_v5  # noqa: E402
import scripts.run_controlled_clinical_mimic_cxr_budget_followup as precoverage  # noqa: E402
import scripts.run_controlled_clinical_mimic_cxr_environment_support_science as v1science  # noqa: E402
from scpcp.artifacts import experiment_tree_sha256  # noqa: E402
from scpcp.controlled_clinical_extension import (  # noqa: E402
    GAMMAS,
    METHODS,
    DatasetPreset,
    donor_overlap_passes,
)
from scpcp.controlled_clinical_mimic_cxr_budget_followup import (  # noqa: E402
    BOOTSTRAP_SEED,
    BOOTSTRAP_RESAMPLES,
    CALIBRATION_TRAJECTORIES,
    DATASET,
    GRID_TRAJECTORIES,
    MINIMUM_OVERLAP_JOINT,
    MINIMUM_SELECTED_SEEDS,
    MINIMUM_SELECTION_RATE,
    PRIMARY_GAMMA,
    PROTOCOL,
    REFERENCE_TRAJECTORIES,
    ROLE_SPLIT,
    SEEDS,
    TARGET_COVERAGE,
    VALIDATION_CLAIMS,
    load_config,
    primary_success_gate,
)


OUTPUT_ROOT = precoverage.SCIENCE_ROOT
SCIENCE_ROOT = OUTPUT_ROOT
PRECOVERAGE_ROOT = precoverage.OUTPUT_ROOT
OVERLAP_PHASE = "donor_overlap"
SCIENCE_PHASE = "science"
CONFIRMATION_SEEDS = SEEDS
PRIMARY_METRIC = "min_t mean_seed(C_seed,t)"
SCIENCE_CONTRACT = {
    "gammas": list(GAMMAS),
    "default_gamma": PRIMARY_GAMMA,
    "primary_gamma": PRIMARY_GAMMA,
    "methods": list(METHODS),
    "calibration_trajectories": CALIBRATION_TRAJECTORIES,
    "grid_trajectories": GRID_TRAJECTORIES,
    "calibration_pool_shared_by_all_methods": True,
    "grid_prefix_shared_by_grid_using_methods": ["MFCS", "PRC", "SC-PCP"],
    "evaluation_trajectories": REFERENCE_TRAJECTORIES,
    "target_adaptation_trajectories": {
        "Standard CP": 0,
        "ACI": 2_000,
        "MFCS": 0,
        "SPCI": 2_000,
        "PRC": 2_000,
        "SC-PCP": 0,
    },
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "primary_metric": PRIMARY_METRIC,
    "primary_success": {
        "method": "SC-PCP",
        "selection_denominator": len(SEEDS),
        "minimum_selected_seeds": MINIMUM_SELECTED_SEEDS,
        "minimum_selection_rate": MINIMUM_SELECTION_RATE,
        "minimum_wsc_point": TARGET_COVERAGE,
        "confidence_interval_is_gating": False,
        "mean_coverage_is_gating": False,
        "width_is_gating": False,
    },
    "role_split": list(ROLE_SPLIT),
    "bridge_candidate_id": precoverage.v1.BRIDGE_CANDIDATE_ID,
    "canonical_scpcp_mutation_permitted": False,
}
SCIENCE_ROW_FIELDS = {
    "seed",
    "dataset",
    "gamma",
    "q_low",
    "q_high",
    "adaptation_seeds",
    "scpcp_minimum_ess_fraction",
    "scpcp_minimum_candidate_ess_fraction",
    "scpcp_selected_endpoint",
    "scpcp_failure_stage",
    "methods",
}
AVAILABLE_VECTOR_FIELDS = (
    "radii",
    "source_coverage",
    "target_coverage",
    "coverage_gap",
    "source_q90",
    "target_q90",
    "q90_relative_gap",
    "target_normalized_width",
    "prefix_ess_fraction",
    "maximum_normalized_weight_share",
    "raw_log_weight_span",
    "policy_tv_on_source_states",
    "source_difficulty",
    "target_difficulty",
)
ADAPTATION_FIELDS = {
    "adaptation_rounds",
    "adaptation_per_time_coverage",
    "adaptation_round_worst_coverage",
    "adaptation_pathwise_coverage",
    "selected_scale",
}
METHOD_COMMON_FIELDS = {
    "selection_available",
    "selection_status",
    "information_regime",
    "target_adaptation_trajectories",
}
AVAILABLE_SCALAR_FIELDS = {
    "donor_kernel_ess_fraction_min",
    "donor_probability_max",
}


@dataclass(frozen=True)
class ConfirmationAnchor:
    split_audit: Mapping[str, Any]
    base_context_identity: Mapping[str, Any]
    kernel_context_identity: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_audit": dict(self.split_audit),
            "base_context_identity": dict(self.base_context_identity),
            "kernel_context_identity": dict(self.kernel_context_identity),
        }


@dataclass(frozen=True)
class GateBundle:
    protocol: Any
    preset: DatasetPreset
    theta: Any
    eligible_seeds: tuple[int, ...]
    anchors: Mapping[int, ConfirmationAnchor]
    seed_to_device: Mapping[int, str]
    active_source_tree_sha256: str
    precoverage_binding: Mapping[str, Any]
    rng_mapping_sha256: str
    eligibility_record: Mapping[str, Any]
    contract: Mapping[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "run"))
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--audit-go-sha256")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    precoverage.v1._validate_devices(devices)
    gates = verify_gate_bundle(devices=devices)
    audit_hash = _json_sha256(gates.contract)
    if args.phase == "audit":
        if args.audit_go_sha256 is not None or args.resume:
            parser.error("audit does not accept --audit-go-sha256 or --resume")
        if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
            raise FileExistsError(f"fresh science root already exists: {OUTPUT_ROOT}")
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "status": "READ_ONLY_SCIENCE_AUDIT_GO",
                    "audit_contract_sha256": audit_hash,
                    "precoverage_status": "PRECOVERAGE_GO",
                    "eligible_seed_count": len(gates.eligible_seeds),
                    "coverage_generated": False,
                    "output_root_created": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.audit_go_sha256 != audit_hash:
        raise RuntimeError("run requires the exact read-only science audit hash")
    run_science(
        OUTPUT_ROOT,
        gates=gates,
        devices=devices,
        audit_go_sha256=audit_hash,
        resume=args.resume,
    )
    print(OUTPUT_ROOT)


def verify_gate_bundle(
    *,
    devices: tuple[str, ...],
    precoverage_root: Path = PRECOVERAGE_ROOT,
) -> GateBundle:
    """Verify the complete precoverage result without mutating science state."""

    load_config(precoverage.CONFIG_PATH)
    bundle = precoverage.verify_precoverage_go(precoverage_root)
    metadata = bundle["metadata"]
    active_audit = precoverage.build_read_only_audit(devices)
    if metadata.get("read_only_audit") != active_audit:
        raise RuntimeError("precoverage audit no longer matches active source/history")
    active_source_hash = experiment_tree_sha256()
    if metadata.get("source_tree_sha256") != active_source_hash:
        raise RuntimeError("source changed after precoverage")
    protocol = precoverage.runtime_protocol()
    preset = protocol.datasets[DATASET]
    theta = precoverage.v1._b02()
    eligible = tuple(int(seed) for seed in bundle["gate"]["joint_pass_seeds"])
    if (
        len(eligible) < MINIMUM_OVERLAP_JOINT
        or tuple(seed for seed in SEEDS if seed in set(eligible)) != eligible
    ):
        raise RuntimeError("precoverage eligible seed identity differs")

    support_by_seed = {int(row["seed"]): row for row in bundle["support"]}
    k0_by_seed = {int(row["seed"]): row for row in bundle["k0"]}
    anchors: dict[int, ConfirmationAnchor] = {}
    eligibility_rows = []
    for seed in SEEDS:
        support_row = support_by_seed[seed]
        k0_row = k0_by_seed[seed]
        base_identity = k0_row["base_context_identity"]
        if (
            support_row.get("split_audit") != k0_row.get("split_audit")
            or support_row.get("n_actions") != base_identity.get("n_actions")
            or support_row.get("action_mapping") != base_identity.get("action_mapping")
        ):
            raise RuntimeError(f"support/K0 cohort identity differs for seed {seed}")
        support_passed = bool(support_row["passed"])
        k0_passed = bool(k0_row["passed"])
        is_eligible = support_passed and k0_passed
        if is_eligible:
            k0 = k0_by_seed[seed]
            anchors[seed] = ConfirmationAnchor(
                split_audit=dict(k0["split_audit"]),
                base_context_identity=dict(k0["base_context_identity"]),
                kernel_context_identity=dict(k0["kernel_context_identity"]),
            )
        eligibility_rows.append(
            {
                "seed": seed,
                "support_passed": support_passed,
                "k0_passed": k0_passed,
                "support_k0_eligible": is_eligible,
                "exclusion_reason": None if is_eligible else "SUPPORT_OR_K0_FAILED",
            }
        )
    if tuple(seed for seed in SEEDS if seed in anchors) != eligible:
        raise RuntimeError("precoverage anchors differ from the eligible seed set")

    rng_hash = str(active_audit["rng_audit"]["full_mapping_sha256"])
    device_map = precoverage.v1._seed_device_mapping(SEEDS, devices)
    eligibility = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "prespecified_seed_count": len(SEEDS),
        "support_k0_eligible_seed_count": len(eligible),
        "support_k0_eligible_seeds": list(eligible),
        "selection_rate_denominator": len(SEEDS),
        "seed_records": eligibility_rows,
        "seed_deletions": 0,
        "validation_claims": VALIDATION_CLAIMS,
    }
    contract = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "devices": list(devices),
        "precoverage_binding": bundle["binding"],
        "source_tree_sha256": active_source_hash,
        "rng_stream_mapping_sha256": rng_hash,
        "eligible_seeds": list(eligible),
        "anchor_sha256": {
            str(seed): _json_sha256(anchors[seed].to_dict()) for seed in eligible
        },
        "science_contract": SCIENCE_CONTRACT,
        "coverage_permitted_before_overlap_unlock": False,
        "validation_claims": VALIDATION_CLAIMS,
    }
    return GateBundle(
        protocol=protocol,
        preset=preset,
        theta=theta,
        eligible_seeds=eligible,
        anchors=anchors,
        seed_to_device={seed: device_map[seed] for seed in eligible},
        active_source_tree_sha256=active_source_hash,
        precoverage_binding=bundle["binding"],
        rng_mapping_sha256=rng_hash,
        eligibility_record=eligibility,
        contract=contract,
    )


def run_science(
    output_root: Path,
    *,
    gates: GateBundle,
    devices: tuple[str, ...],
    audit_go_sha256: str,
    resume: bool,
) -> None:
    if output_root.resolve() != OUTPUT_ROOT:
        raise RuntimeError(f"science root is frozen to {OUTPUT_ROOT}")
    gate_hash = _json_sha256(gates.contract)
    if audit_go_sha256 != gate_hash:
        raise RuntimeError("science audit hash differs")
    source_hash, source_snapshot = precoverage.active_source_snapshot()
    if source_hash != gates.active_source_tree_sha256:
        raise RuntimeError("source changed after the science audit")
    metadata = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "phase": "post_precoverage_science",
        "output_root": str(output_root),
        "devices": list(devices),
        "source_tree_sha256": source_hash,
        "source_snapshot": source_snapshot["contract"],
        "read_only_audit_go_sha256": audit_go_sha256,
        "gate_contract": dict(gates.contract),
        "gate_contract_sha256": gate_hash,
        "precoverage_binding": dict(gates.precoverage_binding),
        "rng_stream_mapping_sha256": gates.rng_mapping_sha256,
        "science_contract": SCIENCE_CONTRACT,
        "role_split": list(ROLE_SPLIT),
        "bridge_candidate": gates.theta.to_dict(),
        "coverage_may_start_only_after_overlap_unlock": True,
        "canonical_scpcp_mutation_permitted": False,
        "seed_deletion_permitted": False,
        "validation_claims": VALIDATION_CLAIMS,
    }
    prepare_science_root(output_root, metadata, source_snapshot, resume=resume)
    if (output_root / "COMPLETE").exists():
        _verify_complete_root(output_root, metadata, gates)
        return
    validate_partial_root(output_root, metadata, gates)
    write_or_verify_json(output_root / "eligibility.json", gates.eligibility_record)

    overlap_preset = replace(gates.preset, seeds=gates.eligible_seeds)
    overlap_rows = run_phase(
        output_root / OVERLAP_PHASE / "seeds",
        phase=OVERLAP_PHASE,
        preset=overlap_preset,
        gates=gates,
        devices=devices,
        worker=overlap_worker,
        source_hash=source_hash,
        gate_hash=gate_hash,
    )
    overlap_summary = summarize_overlap(overlap_rows, gates.eligible_seeds)
    write_or_verify_json(output_root / OVERLAP_PHASE / "summary.json", overlap_summary)
    write_or_verify_text(
        output_root / OVERLAP_PHASE / "COMPLETE",
        f"overlap-complete summary_sha256={_json_sha256(overlap_summary)}\n",
    )
    if not overlap_summary["science_may_start"]:
        if (output_root / "SCIENCE_UNLOCK.json").exists() or (
            output_root / SCIENCE_PHASE
        ).exists():
            raise RuntimeError("coverage artifacts exist despite overlap NO-GO")
        final = overlap_no_go_status(gates, overlap_summary)
        write_or_verify_json(output_root / "FINAL_STATUS.json", final)
        finalize_science_root(output_root, metadata, gates)
        return

    science_seeds = tuple(int(seed) for seed in overlap_summary["passed_seeds"])
    unlock = science_unlock(gates, overlap_summary, science_seeds)
    write_or_verify_json(output_root / "SCIENCE_UNLOCK.json", unlock)
    if not valid_science_unlock(output_root, gates):
        raise RuntimeError("science unlock commit failed")

    science_preset = replace(gates.preset, seeds=science_seeds)
    science_results = run_phase(
        output_root / SCIENCE_PHASE / "seeds",
        phase=SCIENCE_PHASE,
        preset=science_preset,
        gates=gates,
        devices=devices,
        worker=science_worker,
        source_hash=source_hash,
        gate_hash=gate_hash,
    )
    rows = [row for result in science_results for row in result["rows"]]
    bootstrap = ensure_bootstrap_artifacts(output_root / SCIENCE_PHASE, gates.preset)
    summary = summarize_science(
        rows,
        preset=gates.preset,
        support_k0_eligible_seeds=gates.eligible_seeds,
        selected_seeds=science_seeds,
        bootstrap_contract=bootstrap,
    )
    audit = coverage_audit(
        rows,
        summary=summary,
        support_k0_eligible_seeds=gates.eligible_seeds,
        selected_seeds=science_seeds,
    )
    gate = {
        **primary_success_gate(summary),
        "full_five_gamma_six_method_matrix_verified": True,
        "coverage_audit_sha256": _json_sha256(audit),
    }
    summary["primary_success_gate"] = gate
    summary["primary_status"] = gate["status"]
    summary["default_gamma"] = PRIMARY_GAMMA
    summary["validation_claims"] = VALIDATION_CLAIMS
    science_final = science_complete_status(gate, science_seeds)
    write_or_verify_json(output_root / SCIENCE_PHASE / "summary.json", summary)
    write_or_verify_json(output_root / SCIENCE_PHASE / "coverage_audit.json", audit)
    write_or_verify_json(
        output_root / SCIENCE_PHASE / "FINAL_STATUS.json", science_final
    )
    write_or_verify_text(output_root / SCIENCE_PHASE / "COMPLETE", "science-complete\n")
    final = top_level_science_status(science_final, unlock)
    write_or_verify_json(output_root / "FINAL_STATUS.json", final)
    finalize_science_root(output_root, metadata, gates)


def overlap_no_go_status(
    gates: GateBundle, overlap_summary: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "OVERLAP_NO_GO",
        "primary_status": "NOT_EVALUATED_OVERLAP_NO_GO",
        "science_unlocked": False,
        "coverage_generated": False,
        "prespecified_seed_count": len(SEEDS),
        "support_k0_eligible_seed_count": len(gates.eligible_seeds),
        "joint_overlap_pass_count": overlap_summary["joint_overlap_pass_count"],
        "seed_deletions": 0,
        "validation_claims": VALIDATION_CLAIMS,
    }


def science_complete_status(
    gate: Mapping[str, Any], science_seeds: tuple[int, ...]
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "SCIENCE_COMPLETE",
        "primary_status": gate["status"],
        "primary_success": gate["passed"],
        "primary_success_gate_sha256": _json_sha256(gate),
        "methods": list(METHODS),
        "gammas": list(GAMMAS),
        "default_gamma": PRIMARY_GAMMA,
        "primary_gamma": PRIMARY_GAMMA,
        "primary_metric": PRIMARY_METRIC,
        "prespecified_seed_count": len(SEEDS),
        "science_eligible_seed_count": len(science_seeds),
        "science_eligible_seeds": list(science_seeds),
        "seed_deletions": 0,
        "validation_claims": VALIDATION_CLAIMS,
    }


def top_level_science_status(
    science_final: Mapping[str, Any], unlock: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **science_final,
        "precoverage_status": "PRECOVERAGE_GO",
        "overlap_status": "OVERLAP_GO",
        "science_unlocked": True,
        "coverage_generated": True,
        "science_unlock_sha256": _json_sha256(unlock),
    }


def overlap_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    gates: GateBundle,
) -> dict[str, Any]:
    context, base_identity, kernel_identity = reconstruct_context(
        seed, preset, device, gates
    )
    metrics, diagnostics = extension._donor_overlap_probe(
        context, seed=seed, protocol=gates.protocol
    )
    return {
        "seed": seed,
        "dataset": DATASET,
        "phase": OVERLAP_PHASE,
        "passed": donor_overlap_passes(metrics, gates.protocol.donor_overlap_gate),
        "failure_consequence": "OVERLAP_NO_GO_NO_COVERAGE_SCIENCE",
        "metrics": extension.asdict(metrics),
        "diagnostics": diagnostics,
        "q_low": context.q_low,
        "q_high": context.q_high,
        "q_mid": context.q_low + 0.5 * (context.q_high - context.q_low),
        "n_actions": context.n_actions,
        "action_mapping": {
            str(key): value for key, value in context.action_mapping.items()
        },
        "split_audit": extension._split_audit(context.splits),
        "base_context_identity": base_identity,
        "kernel_context_identity": kernel_identity,
        "theta": gates.theta.to_dict(),
        "confirmation_anchor_identity_sha256": _json_sha256(
            gates.anchors[seed].to_dict()
        ),
    }


def science_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    gates: GateBundle,
) -> dict[str, Any]:
    context, base_identity, kernel_identity = reconstruct_context(
        seed, preset, device, gates
    )
    rows = extension.run_science_seed(
        seed,
        preset=preset,
        device=device,
        protocol=gates.protocol,
        context=context,
    )
    return {
        "seed": seed,
        "dataset": DATASET,
        "phase": SCIENCE_PHASE,
        "interpretation_status": "EMPIRICAL_OVERLAP_SCREEN_PASSED",
        "rows": rows,
        "q_low": context.q_low,
        "q_high": context.q_high,
        "n_actions": context.n_actions,
        "action_mapping": {
            str(key): value for key, value in context.action_mapping.items()
        },
        "split_audit": extension._split_audit(context.splits),
        "base_context_identity": base_identity,
        "kernel_context_identity": kernel_identity,
        "theta": gates.theta.to_dict(),
        "confirmation_anchor_identity_sha256": _json_sha256(
            gates.anchors[seed].to_dict()
        ),
        "calibration_trajectories": CALIBRATION_TRAJECTORIES,
        "grid_trajectories": GRID_TRAJECTORIES,
        "evaluation_trajectories": REFERENCE_TRAJECTORIES,
        "science_contract_sha256": _json_sha256(SCIENCE_CONTRACT),
    }


def reconstruct_context(
    seed: int,
    preset: DatasetPreset,
    device: str,
    gates: GateBundle,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    base = extension._prepare_extension_context(seed, preset, device, gates.protocol)
    context = fidelity_v5._context_with_theta(base, gates.theta)
    split = extension._split_audit(base.splits)
    base_identity = extension._context_identity(base)
    kernel_identity = fidelity_v5._candidate_context_identity(
        base, context.environment, gates.theta
    )
    anchor = gates.anchors[seed]
    if (
        split != anchor.split_audit
        or base_identity != anchor.base_context_identity
        or kernel_identity != anchor.kernel_context_identity
        or split.get("split_fractions") != list(ROLE_SPLIT)
    ):
        raise RuntimeError(f"reconstructed context differs for seed {seed}")
    return context, base_identity, kernel_identity


def run_phase(
    root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    gates: GateBundle,
    devices: tuple[str, ...],
    worker: Callable[..., dict[str, Any]],
    source_hash: str,
    gate_hash: str,
) -> list[dict[str, Any]]:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise RuntimeError(f"invalid {phase} root")
    root.mkdir(parents=True, exist_ok=True)
    expected = {f"seed_{seed:06d}.json" for seed in preset.seeds} | {"COMPLETE"}
    children = list(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise RuntimeError(f"unsafe {phase} artifact")
    if unexpected := {path.name for path in children} - expected:
        raise RuntimeError(f"unexpected {phase} artifacts: {sorted(unexpected)}")
    complete = root / "COMPLETE"
    if complete.exists() and complete.read_text() != "complete\n":
        raise RuntimeError(f"invalid {phase} COMPLETE marker")

    completed: dict[int, dict[str, Any]] = {}
    for seed in preset.seeds:
        path = root / f"seed_{seed:06d}.json"
        if not path.exists():
            continue
        payload = precoverage.read_json(path)
        validate_phase_payload(
            payload,
            phase=phase,
            seed=seed,
            device=gates.seed_to_device[seed],
            source_hash=source_hash,
            gate_hash=gate_hash,
            gates=gates,
        )
        completed[seed] = payload["result"]
    pending = tuple(seed for seed in preset.seeds if seed not in completed)
    if pending and complete.exists():
        raise RuntimeError(f"{phase} is complete with missing seeds")
    if pending:
        groups = {
            device: tuple(
                seed for seed in pending if gates.seed_to_device[seed] == device
            )
            for device in devices
        }
        with ProcessPoolExecutor(
            max_workers=len(devices), mp_context=get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(
                    precoverage._phase_group,
                    seeds,
                    device,
                    preset,
                    worker,
                    (gates,),
                ): device
                for device, seeds in groups.items()
                if seeds
            }
            for future in as_completed(futures):
                for seed, device, result in future.result():
                    anchor_hash = _json_sha256(gates.anchors[seed].to_dict())
                    payload = {
                        "protocol": PROTOCOL,
                        "dataset": DATASET,
                        "phase": phase,
                        "seed": seed,
                        "device": device,
                        "source_tree_sha256": source_hash,
                        "gate_contract_sha256": gate_hash,
                        "rng_stream_mapping_sha256": gates.rng_mapping_sha256,
                        "confirmation_anchor_sha256": anchor_hash,
                        "result": result,
                    }
                    validate_phase_payload(
                        payload,
                        phase=phase,
                        seed=seed,
                        device=device,
                        source_hash=source_hash,
                        gate_hash=gate_hash,
                        gates=gates,
                    )
                    precoverage.write_json(root / f"seed_{seed:06d}.json", payload)
                    completed[seed] = result
    if set(completed) != set(preset.seeds):
        raise RuntimeError(f"{phase} did not complete its exact seed bank")
    if not complete.exists():
        precoverage.write_text(complete, "complete\n")
    return [completed[seed] for seed in preset.seeds]


def validate_phase_payload(
    payload: Mapping[str, Any],
    *,
    phase: str,
    seed: int,
    device: str,
    source_hash: str,
    gate_hash: str,
    gates: GateBundle,
) -> None:
    result = payload.get("result")
    expected_anchor = _json_sha256(gates.anchors[seed].to_dict())
    if (
        set(payload)
        != {
            "protocol",
            "dataset",
            "phase",
            "seed",
            "device",
            "source_tree_sha256",
            "gate_contract_sha256",
            "rng_stream_mapping_sha256",
            "confirmation_anchor_sha256",
            "result",
        }
        or payload["protocol"] != PROTOCOL
        or payload["dataset"] != DATASET
        or payload["phase"] != phase
        or payload["seed"] != seed
        or payload["device"] != device
        or payload["source_tree_sha256"] != source_hash
        or payload["gate_contract_sha256"] != gate_hash
        or payload["rng_stream_mapping_sha256"] != gates.rng_mapping_sha256
        or payload["confirmation_anchor_sha256"] != expected_anchor
        or not isinstance(result, Mapping)
        or result.get("seed") != seed
        or result.get("dataset") != DATASET
        or result.get("phase") != phase
        or result.get("split_audit") != gates.anchors[seed].split_audit
        or result.get("base_context_identity")
        != gates.anchors[seed].base_context_identity
        or result.get("kernel_context_identity")
        != gates.anchors[seed].kernel_context_identity
        or result.get("theta") != gates.theta.to_dict()
        or result.get("confirmation_anchor_identity_sha256") != expected_anchor
    ):
        raise RuntimeError(f"invalid {phase} artifact for seed {seed}")
    if phase == OVERLAP_PHASE:
        metrics = result.get("metrics")
        if (
            set(result)
            != {
                "seed",
                "dataset",
                "phase",
                "passed",
                "failure_consequence",
                "metrics",
                "diagnostics",
                "q_low",
                "q_high",
                "q_mid",
                "n_actions",
                "action_mapping",
                "split_audit",
                "base_context_identity",
                "kernel_context_identity",
                "theta",
                "confirmation_anchor_identity_sha256",
            }
            or not isinstance(metrics, Mapping)
            or type(result.get("passed")) is not bool
            or result["passed"]
            != donor_overlap_passes(
                extension.DonorOverlapMetrics(**metrics),
                gates.protocol.donor_overlap_gate,
            )
        ):
            raise RuntimeError(f"overlap decision differs for seed {seed}")
        v1science._validate_overlap_result(result, gates.preset)
        return
    if phase != SCIENCE_PHASE:
        raise RuntimeError(f"unknown science phase: {phase}")
    rows = result.get("rows")
    expected_result_fields = {
        "seed",
        "dataset",
        "phase",
        "interpretation_status",
        "rows",
        "q_low",
        "q_high",
        "n_actions",
        "action_mapping",
        "split_audit",
        "base_context_identity",
        "kernel_context_identity",
        "theta",
        "confirmation_anchor_identity_sha256",
        "calibration_trajectories",
        "grid_trajectories",
        "evaluation_trajectories",
        "science_contract_sha256",
    }
    if (
        set(result) != expected_result_fields
        or result.get("interpretation_status") != "EMPIRICAL_OVERLAP_SCREEN_PASSED"
        or not isinstance(rows, list)
        or len(rows) != len(GAMMAS)
        or tuple(float(row.get("gamma")) for row in rows) != GAMMAS
        or any(set(row.get("methods", {})) != set(METHODS) for row in rows)
        or result.get("calibration_trajectories") != CALIBRATION_TRAJECTORIES
        or result.get("grid_trajectories") != GRID_TRAJECTORIES
        or result.get("evaluation_trajectories") != REFERENCE_TRAJECTORIES
        or result.get("science_contract_sha256") != _json_sha256(SCIENCE_CONTRACT)
    ):
        raise RuntimeError(f"science matrix differs for seed {seed}")
    v1science._validate_science_result(result, gates.preset)
    _validate_science_rows(rows, seed=seed, preset=gates.preset)


def _validate_science_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int, preset: DatasetPreset
) -> None:
    expected_adaptation = extension._adaptation_seeds(seed)
    for row in rows:
        if (
            set(row) != SCIENCE_ROW_FIELDS
            or row.get("seed") != seed
            or row.get("dataset") != DATASET
            or row.get("adaptation_seeds") != expected_adaptation
        ):
            raise RuntimeError(f"science row identity differs for seed {seed}")
        for name in (
            "scpcp_minimum_ess_fraction",
            "scpcp_minimum_candidate_ess_fraction",
        ):
            value = row.get(name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise RuntimeError(f"science ESS evidence differs for seed {seed}")
        for method in METHODS:
            method_row = row["methods"][method]
            if not isinstance(method_row, Mapping):
                raise RuntimeError(f"science method row differs for seed {seed}")
            available = method_row.get("selection_available")
            if type(available) is not bool:
                raise RuntimeError(f"science selection flag differs for seed {seed}")
            expected_fields = set(METHOD_COMMON_FIELDS)
            if method in {"ACI", "SPCI", "PRC"}:
                expected_fields.update(ADAPTATION_FIELDS)
            if available:
                expected_fields.update(AVAILABLE_VECTOR_FIELDS)
                expected_fields.update(AVAILABLE_SCALAR_FIELDS)
                for name in AVAILABLE_VECTOR_FIELDS:
                    values = method_row.get(name)
                    if (
                        not isinstance(values, list)
                        or len(values) != preset.horizon
                        or not all(np.isfinite(float(value)) for value in values)
                    ):
                        raise RuntimeError(
                            f"science finite horizon differs for {method}, seed {seed}"
                        )
                if any(
                    not np.isfinite(float(method_row[name]))
                    for name in AVAILABLE_SCALAR_FIELDS
                ):
                    raise RuntimeError(
                        f"science finite scalar differs for {method}, seed {seed}"
                    )
            else:
                expected_fields.add("radii")
                if method_row.get("radii") != []:
                    raise RuntimeError(f"unavailable method has radii for seed {seed}")
            if set(method_row) != expected_fields:
                raise RuntimeError(
                    f"science method schema differs for {method}, seed {seed}"
                )
            if method in {"ACI", "SPCI", "PRC"}:
                adaptation = method_row.get("adaptation_per_time_coverage")
                round_coverage = method_row.get("adaptation_round_worst_coverage")
                rounds = method_row.get("adaptation_rounds")
                selected_scale = method_row.get("selected_scale")
                if (
                    not isinstance(adaptation, list)
                    or len(adaptation) != preset.horizon
                    or not all(
                        np.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
                        for value in adaptation
                    )
                    or method_row.get("target_adaptation_trajectories") != 2_000
                    or type(rounds) is not int
                    or rounds <= 0
                    or not isinstance(round_coverage, list)
                    or len(round_coverage) != rounds
                    or not all(np.isfinite(float(value)) for value in round_coverage)
                    or not np.isfinite(
                        float(method_row.get("adaptation_pathwise_coverage"))
                    )
                    or (
                        method == "PRC"
                        and (
                            isinstance(selected_scale, bool)
                            or not isinstance(selected_scale, (int, float))
                            or not np.isfinite(float(selected_scale))
                        )
                    )
                    or (method != "PRC" and selected_scale is not None)
                ):
                    raise RuntimeError(
                        f"science adaptation evidence differs for {method}, seed {seed}"
                    )


def summarize_overlap(
    rows: Sequence[Mapping[str, Any]], eligible_seeds: tuple[int, ...]
) -> dict[str, Any]:
    indexed = {int(row["seed"]): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(eligible_seeds):
        raise RuntimeError("overlap rows do not cover the exact eligible bank")
    passed = tuple(seed for seed in eligible_seeds if bool(indexed[seed]["passed"]))
    go = len(passed) >= MINIMUM_OVERLAP_JOINT
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "OVERLAP_GO" if go else "OVERLAP_NO_GO",
        "gate": "gamma=-4 q_mid+q_high empirical donor-overlap screen",
        "prespecified_seed_count": len(SEEDS),
        "support_k0_eligible_seed_count": len(eligible_seeds),
        "support_k0_eligible_seeds": list(eligible_seeds),
        "joint_overlap_pass_count": len(passed),
        "minimum_joint_overlap_pass_count": MINIMUM_OVERLAP_JOINT,
        "passed_seeds": list(passed),
        "failed_seeds": [seed for seed in eligible_seeds if seed not in passed],
        "overlap_bank_complete": True,
        "science_may_start": go,
        "failure_consequence": "OVERLAP_NO_GO_NO_COVERAGE_SCIENCE",
        "seed_deletions": 0,
    }


def science_unlock(
    gates: GateBundle,
    overlap_summary: Mapping[str, Any],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    if (
        overlap_summary.get("status") != "OVERLAP_GO"
        or overlap_summary.get("science_may_start") is not True
        or overlap_summary.get("passed_seeds") != list(seeds)
        or len(seeds) < MINIMUM_OVERLAP_JOINT
    ):
        raise RuntimeError("overlap result cannot unlock science")
    payload = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "SCIENCE_UNLOCKED",
        "gate_contract_sha256": _json_sha256(gates.contract),
        "precoverage_binding_sha256": _json_sha256(gates.precoverage_binding),
        "overlap_summary_sha256": _json_sha256(overlap_summary),
        "prespecified_seeds": list(SEEDS),
        "science_eligible_seeds": list(seeds),
        "science_eligible_seed_count": len(seeds),
        "coverage_may_start": True,
        "seed_deletions": 0,
    }
    return {**payload, "combined_sha256": _json_sha256(payload)}


def valid_science_unlock(root: Path, gates: GateBundle) -> bool:
    try:
        _, overlap = load_verified_overlap(root, gates)
        unlock = precoverage.read_json(root / "SCIENCE_UNLOCK.json")
    except (
        OSError,
        json.JSONDecodeError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
    ):
        return False
    seeds = tuple(int(seed) for seed in overlap.get("passed_seeds", ()))
    return unlock == science_unlock(gates, overlap, seeds)


def load_verified_overlap(
    root: Path, gates: GateBundle
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_verified_phase(
        root / OVERLAP_PHASE / "seeds",
        phase=OVERLAP_PHASE,
        seeds=gates.eligible_seeds,
        gates=gates,
    )
    expected = summarize_overlap(rows, gates.eligible_seeds)
    observed = precoverage.read_json(root / OVERLAP_PHASE / "summary.json")
    marker = root / OVERLAP_PHASE / "COMPLETE"
    expected_marker = f"overlap-complete summary_sha256={_json_sha256(expected)}\n"
    if (
        observed != expected
        or not marker.is_file()
        or marker.is_symlink()
        or marker.read_text() != expected_marker
    ):
        raise RuntimeError("completed overlap result differs from its seed bank")
    return rows, observed


def read_verified_phase(
    root: Path,
    *,
    phase: str,
    seeds: tuple[int, ...],
    gates: GateBundle,
) -> list[dict[str, Any]]:
    """Read an exact completed phase bank and validate every seed payload."""

    expected = {f"seed_{seed:06d}.json" for seed in seeds} | {"COMPLETE"}
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"{phase} seed bank is missing")
    children = list(root.iterdir())
    if (
        any(path.is_symlink() or not path.is_file() for path in children)
        or {path.name for path in children} != expected
        or (root / "COMPLETE").read_text() != "complete\n"
    ):
        raise RuntimeError(f"{phase} seed bank differs")
    gate_hash = _json_sha256(gates.contract)
    results = []
    for seed in seeds:
        payload = precoverage.read_json(root / f"seed_{seed:06d}.json")
        validate_phase_payload(
            payload,
            phase=phase,
            seed=seed,
            device=gates.seed_to_device[seed],
            source_hash=gates.active_source_tree_sha256,
            gate_hash=gate_hash,
            gates=gates,
        )
        results.append(payload["result"])
    return results


def summarize_science(
    rows: list[dict[str, Any]],
    *,
    preset: DatasetPreset,
    support_k0_eligible_seeds: tuple[int, ...],
    selected_seeds: tuple[int, ...],
    bootstrap_contract: dict[str, Any],
) -> dict[str, Any]:
    summary = extension.summarize_science(
        rows,
        preset=preset,
        selected_seeds=selected_seeds,
        interpretation_status="EMPIRICAL_OVERLAP_SCREEN_PASSED",
        bootstrap_contract=bootstrap_contract,
    )
    aggregates = summary.get("aggregates")
    if (
        not isinstance(aggregates, list)
        or len(aggregates) != len(GAMMAS)
        or tuple(float(row["gamma"]) for row in aggregates) != GAMMAS
        or any(set(row.get("methods", {})) != set(METHODS) for row in aggregates)
    ):
        raise RuntimeError("science summary lacks the full five-by-six matrix")
    summary.update(
        {
            "protocol": PROTOCOL,
            "role": "fresh_task_rng_budget_only_followup",
            "seeds_prespecified": list(SEEDS),
            "seeds_support_k0_eligible": list(support_k0_eligible_seeds),
            "seeds_support_k0_overlap_eligible": list(selected_seeds),
            "seeds_k0_eligible": list(support_k0_eligible_seeds),
            "compatibility_field_semantics": {
                "seeds_k0_eligible": (
                    "alias of seeds_support_k0_eligible before donor-overlap screening"
                ),
                "aggregates[].n_k0_eligible_seeds": (
                    "count of support/K0-eligible seeds before donor-overlap screening"
                ),
                "aggregates[].methods[].n_k0_eligible": (
                    "count of support/K0-eligible seeds before donor-overlap screening"
                ),
            },
            "coverage_conditioning": (
                "successful method selection among support/K0/overlap-eligible seeds"
            ),
            "selection_rate_denominator": "all 20 prespecified seeds",
            "default_gamma": PRIMARY_GAMMA,
            "primary_gamma": PRIMARY_GAMMA,
            "primary_metric": PRIMARY_METRIC,
            "mean_coverage_is_supplementary": True,
            "confidence_intervals_are_supplementary": True,
            "width_is_supplementary_to_primary_attainment": True,
            "seed_deletions": 0,
            "validation_claims": VALIDATION_CLAIMS,
        }
    )
    for aggregate in aggregates:
        aggregate["n_support_k0_eligible_seeds"] = len(support_k0_eligible_seeds)
        aggregate["n_support_k0_overlap_eligible_seeds"] = len(selected_seeds)
        aggregate["n_k0_eligible_seeds"] = len(support_k0_eligible_seeds)
        for method in METHODS:
            method_summary = aggregate["methods"][method]
            method_summary["n_support_k0_eligible"] = len(support_k0_eligible_seeds)
            method_summary["n_support_k0_overlap_eligible"] = len(selected_seeds)
            method_summary["n_k0_eligible"] = len(support_k0_eligible_seeds)
            method_summary["selection_rate_denominator"] = len(SEEDS)
            method_summary["WSC_formula"] = PRIMARY_METRIC
            method_summary["MeanCov_role"] = "supplementary"
    return summary


def coverage_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    support_k0_eligible_seeds: tuple[int, ...],
    selected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    audit = v1science.coverage_audit(
        rows,
        summary=summary,
        support_k0_eligible_seeds=support_k0_eligible_seeds,
        selected_seeds=selected_seeds,
    )
    records = audit.get("records")
    expected_pairs = {(gamma, method) for gamma in GAMMAS for method in METHODS}
    observed_pairs = (
        {(float(row["gamma"]), str(row["method"])) for row in records}
        if isinstance(records, list)
        else set()
    )
    if observed_pairs != expected_pairs or len(records) != len(expected_pairs):
        raise RuntimeError("coverage audit lacks the full five-by-six matrix")
    return {
        **audit,
        "protocol": PROTOCOL,
        "default_gamma": PRIMARY_GAMMA,
        "primary_gamma": PRIMARY_GAMMA,
        "full_five_gamma_six_method_matrix_verified": True,
    }


def verify_bootstrap_artifacts(root: Path, preset: DatasetPreset) -> dict[str, Any]:
    """Read and replay the frozen bootstrap bank without creating artifacts."""

    contract = v1science.v4science._ensure_bootstrap_artifacts(
        root, preset, create_if_missing=False
    )
    contract["selected_subset_rule"] = (
        "for selected-set size n, use floor(U[:, :n] * n) while retaining the "
        "complete prespecified 10000x20 seed bank"
    )
    if (
        contract.get("resamples") != BOOTSTRAP_RESAMPLES
        or contract.get("root_seed") != preset.bootstrap_seed
        or contract.get("prespecified_seed_count") != len(SEEDS)
    ):
        raise RuntimeError("bootstrap contract differs from the frozen v2 bank")
    return contract


def ensure_bootstrap_artifacts(root: Path, preset: DatasetPreset) -> dict[str, Any]:
    """Create or safely repair the deterministic two-file bootstrap commit."""

    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise RuntimeError("bootstrap root must be a regular directory")
    root.mkdir(parents=True, exist_ok=True)
    uniform_path = root / "bootstrap_uniforms.npy"
    index_path = root / "bootstrap_indices.npy"
    rng = np.random.default_rng(preset.bootstrap_seed)
    expected_uniforms = rng.random((BOOTSTRAP_RESAMPLES, len(SEEDS)), dtype=np.float64)
    expected_indices = np.floor(expected_uniforms * len(SEEDS)).astype(np.int16)
    for path, expected in (
        (uniform_path, expected_uniforms),
        (index_path, expected_indices),
    ):
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"unsafe bootstrap artifact: {path}")
            try:
                observed = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise RuntimeError(f"malformed bootstrap artifact: {path}") from error
            if not np.array_equal(observed, expected):
                raise RuntimeError(
                    f"bootstrap artifact differs from frozen seed: {path}"
                )
    if not uniform_path.exists():
        v1science.v4science._write_npy(uniform_path, expected_uniforms)
    if not index_path.exists():
        v1science.v4science._write_npy(index_path, expected_indices)
    return verify_bootstrap_artifacts(root, preset)


def prepare_science_root(
    root: Path,
    metadata: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if resume:
        if root.is_symlink() or not root.is_dir():
            raise FileNotFoundError("resume requires the existing regular science root")
        precoverage.assert_safe_tree(root)
        if precoverage.read_json(root / "metadata.json") != metadata:
            raise RuntimeError("science resume metadata differs")
        precoverage.verify_source_snapshot(root, metadata["source_snapshot"])
        return
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"fresh science root exists: {root}")
    root.mkdir(parents=True)
    fidelity_v4._atomic_write(
        root / source_snapshot["contract"]["archive_path"],
        source_snapshot["archive_bytes"],
    )
    fidelity_v4._atomic_write(
        root / source_snapshot["contract"]["manifest_path"],
        source_snapshot["manifest_bytes"],
    )
    precoverage.write_json(root / "metadata.json", metadata)
    precoverage.verify_source_snapshot(root, metadata["source_snapshot"])


def validate_partial_root(
    root: Path, metadata: Mapping[str, Any], gates: GateBundle
) -> None:
    precoverage.assert_safe_tree(root)
    allowed = allowed_paths(metadata, gates)
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if unexpected := observed - allowed:
        raise RuntimeError(f"unexpected science artifacts: {sorted(unexpected)}")
    has_coverage = any(path.startswith(f"{SCIENCE_PHASE}/") for path in observed)
    if has_coverage and not valid_science_unlock(root, gates):
        raise RuntimeError("coverage artifacts exist before a valid overlap unlock")


def allowed_paths(metadata: Mapping[str, Any], gates: GateBundle) -> set[str]:
    source = metadata["source_snapshot"]
    paths = {
        "metadata.json",
        str(source["archive_path"]),
        str(source["manifest_path"]),
        "eligibility.json",
        f"{OVERLAP_PHASE}/summary.json",
        f"{OVERLAP_PHASE}/COMPLETE",
        f"{OVERLAP_PHASE}/seeds/COMPLETE",
        "SCIENCE_UNLOCK.json",
        f"{SCIENCE_PHASE}/bootstrap_uniforms.npy",
        f"{SCIENCE_PHASE}/bootstrap_indices.npy",
        f"{SCIENCE_PHASE}/summary.json",
        f"{SCIENCE_PHASE}/coverage_audit.json",
        f"{SCIENCE_PHASE}/FINAL_STATUS.json",
        f"{SCIENCE_PHASE}/COMPLETE",
        f"{SCIENCE_PHASE}/seeds/COMPLETE",
        "FINAL_STATUS.json",
        "manifest.json",
        "COMPLETE",
    }
    for seed in gates.eligible_seeds:
        paths.add(f"{OVERLAP_PHASE}/seeds/seed_{seed:06d}.json")
        paths.add(f"{SCIENCE_PHASE}/seeds/seed_{seed:06d}.json")
    return paths


def finalize_science_root(
    root: Path, metadata: Mapping[str, Any], gates: GateBundle
) -> None:
    if experiment_tree_sha256() != gates.active_source_tree_sha256:
        raise RuntimeError("source changed during v2 science")
    refreshed = verify_gate_bundle(
        devices=tuple(metadata["devices"]), precoverage_root=PRECOVERAGE_ROOT
    )
    if refreshed.contract != gates.contract:
        raise RuntimeError("precoverage contract changed during science")
    if precoverage.read_json(root / "metadata.json") != metadata:
        raise RuntimeError("science metadata changed")
    write_manifest(root)
    final = precoverage.read_json(root / "FINAL_STATUS.json")
    marker = (
        f"complete source_tree_sha256={gates.active_source_tree_sha256} "
        f"gate_contract_sha256={_json_sha256(gates.contract)} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={precoverage._file_sha256(root / 'manifest.json')}\n"
    )
    write_or_verify_text(root / "COMPLETE", marker)
    _verify_complete_root(root, metadata, gates)


def write_manifest(root: Path) -> None:
    precoverage.assert_safe_tree(root)
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or relative in {Path("manifest.json"), Path("COMPLETE")}:
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": precoverage._file_sha256(path),
            }
        )
    write_or_verify_json(
        root / "manifest.json",
        {"protocol": PROTOCOL, "artifact_count": len(entries), "artifacts": entries},
    )


def verify_complete_root(
    root: Path = SCIENCE_ROOT,
    devices: tuple[str, ...] = ("cuda:0", "cuda:1"),
) -> None:
    """Public read-only verification entry for a completed formal science root."""

    gates = verify_gate_bundle(devices=devices)
    metadata = precoverage.read_json(root / "metadata.json")
    _verify_complete_root(root, metadata, gates)


def _verify_complete_root(
    root: Path, metadata: Mapping[str, Any], gates: GateBundle
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("completed science root must be a regular directory")
    precoverage.assert_safe_tree(root)
    if (
        metadata.get("protocol") != PROTOCOL
        or metadata.get("dataset") != DATASET
        or metadata.get("output_root") != str(root.resolve())
        or metadata.get("source_tree_sha256") != gates.active_source_tree_sha256
        or metadata.get("gate_contract") != gates.contract
        or metadata.get("gate_contract_sha256") != _json_sha256(gates.contract)
        or metadata.get("science_contract") != SCIENCE_CONTRACT
        or metadata.get("canonical_scpcp_mutation_permitted") is not False
        or metadata.get("coverage_may_start_only_after_overlap_unlock") is not True
        or metadata.get("seed_deletion_permitted") is not False
        or metadata.get("validation_claims") != VALIDATION_CLAIMS
    ):
        raise RuntimeError("completed science metadata differs")
    precoverage.verify_source_snapshot(root, metadata["source_snapshot"])
    manifest = precoverage.read_json(root / "manifest.json")
    entries = manifest.get("artifacts")
    if manifest.get("protocol") != PROTOCOL or not isinstance(entries, list):
        raise RuntimeError("science manifest header differs")
    expected: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RuntimeError("science manifest entry is malformed")
        relative = Path(str(entry.get("path", "")))
        path = precoverage.safe_child(root, relative)
        if relative in expected or relative in {
            Path("manifest.json"),
            Path("COMPLETE"),
        }:
            raise RuntimeError("science manifest path is duplicated or reserved")
        expected.add(relative)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != entry.get("bytes")
            or precoverage._file_sha256(path) != entry.get("sha256")
        ):
            raise RuntimeError(f"science manifest mismatch: {relative}")
    observed = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root) not in {Path("manifest.json"), Path("COMPLETE")}
    }
    if observed != expected or manifest.get("artifact_count") != len(expected):
        raise RuntimeError("science manifest file set differs")
    final = precoverage.read_json(root / "FINAL_STATUS.json")
    expected_paths = expected_complete_paths(root, metadata, gates, final)
    all_files = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
    if all_files != expected_paths:
        raise RuntimeError("completed science artifact set differs")
    marker = (
        f"complete source_tree_sha256={gates.active_source_tree_sha256} "
        f"gate_contract_sha256={_json_sha256(gates.contract)} "
        f"final_status_sha256={_json_sha256(final)} "
        f"manifest_sha256={precoverage._file_sha256(root / 'manifest.json')}\n"
    )
    if (root / "COMPLETE").read_text() != marker:
        raise RuntimeError("science COMPLETE marker differs")
    if precoverage.read_json(root / "eligibility.json") != gates.eligibility_record:
        raise RuntimeError("completed eligibility record differs")
    _, overlap = load_verified_overlap(root, gates)
    if final.get("status") == "OVERLAP_NO_GO":
        if overlap.get("status") != "OVERLAP_NO_GO" or final != overlap_no_go_status(
            gates, overlap
        ):
            raise RuntimeError("completed overlap NO-GO status differs")
        return
    if final.get("status") != "SCIENCE_COMPLETE" or not valid_science_unlock(
        root, gates
    ):
        raise RuntimeError("completed science lacks a valid overlap unlock")

    unlock = precoverage.read_json(root / "SCIENCE_UNLOCK.json")
    science_seeds = tuple(int(seed) for seed in unlock["science_eligible_seeds"])
    science_results = read_verified_phase(
        root / SCIENCE_PHASE / "seeds",
        phase=SCIENCE_PHASE,
        seeds=science_seeds,
        gates=gates,
    )
    rows = [row for result in science_results for row in result["rows"]]
    bootstrap = verify_bootstrap_artifacts(root / SCIENCE_PHASE, gates.preset)
    expected_summary = summarize_science(
        rows,
        preset=gates.preset,
        support_k0_eligible_seeds=gates.eligible_seeds,
        selected_seeds=science_seeds,
        bootstrap_contract=bootstrap,
    )
    expected_audit = coverage_audit(
        rows,
        summary=expected_summary,
        support_k0_eligible_seeds=gates.eligible_seeds,
        selected_seeds=science_seeds,
    )
    expected_gate = {
        **primary_success_gate(expected_summary),
        "full_five_gamma_six_method_matrix_verified": True,
        "coverage_audit_sha256": _json_sha256(expected_audit),
    }
    expected_summary["primary_success_gate"] = expected_gate
    expected_summary["primary_status"] = expected_gate["status"]
    expected_summary["default_gamma"] = PRIMARY_GAMMA
    expected_summary["validation_claims"] = VALIDATION_CLAIMS
    expected_science_final = science_complete_status(expected_gate, science_seeds)
    if (
        precoverage.read_json(root / SCIENCE_PHASE / "summary.json") != expected_summary
        or precoverage.read_json(root / SCIENCE_PHASE / "coverage_audit.json")
        != expected_audit
        or precoverage.read_json(root / SCIENCE_PHASE / "FINAL_STATUS.json")
        != expected_science_final
        or final != top_level_science_status(expected_science_final, unlock)
        or (root / SCIENCE_PHASE / "COMPLETE").read_text() != "science-complete\n"
    ):
        raise RuntimeError("completed science does not recompute from seed rows")


def expected_complete_paths(
    root: Path,
    metadata: Mapping[str, Any],
    gates: GateBundle,
    final: Mapping[str, Any],
) -> set[Path]:
    source = metadata["source_snapshot"]
    paths = {
        Path("metadata.json"),
        Path(str(source["archive_path"])),
        Path(str(source["manifest_path"])),
        Path("eligibility.json"),
        Path(f"{OVERLAP_PHASE}/summary.json"),
        Path(f"{OVERLAP_PHASE}/COMPLETE"),
        Path(f"{OVERLAP_PHASE}/seeds/COMPLETE"),
        Path("FINAL_STATUS.json"),
        Path("manifest.json"),
        Path("COMPLETE"),
    }
    paths.update(
        Path(f"{OVERLAP_PHASE}/seeds/seed_{seed:06d}.json")
        for seed in gates.eligible_seeds
    )
    if final.get("status") == "OVERLAP_NO_GO":
        if final.get("coverage_generated") is not False:
            raise RuntimeError("overlap NO-GO contains coverage")
        return paths
    if final.get("status") != "SCIENCE_COMPLETE" or not valid_science_unlock(
        root, gates
    ):
        raise RuntimeError("completed science status or unlock differs")
    science_seeds = tuple(
        int(seed)
        for seed in precoverage.read_json(root / "SCIENCE_UNLOCK.json")[
            "science_eligible_seeds"
        ]
    )
    paths.update(
        {
            Path("SCIENCE_UNLOCK.json"),
            Path(f"{SCIENCE_PHASE}/bootstrap_uniforms.npy"),
            Path(f"{SCIENCE_PHASE}/bootstrap_indices.npy"),
            Path(f"{SCIENCE_PHASE}/summary.json"),
            Path(f"{SCIENCE_PHASE}/coverage_audit.json"),
            Path(f"{SCIENCE_PHASE}/FINAL_STATUS.json"),
            Path(f"{SCIENCE_PHASE}/COMPLETE"),
            Path(f"{SCIENCE_PHASE}/seeds/COMPLETE"),
        }
    )
    paths.update(
        Path(f"{SCIENCE_PHASE}/seeds/seed_{seed:06d}.json") for seed in science_seeds
    )
    return paths


def load_verified_science_bundle(
    root: Path = SCIENCE_ROOT,
    devices: tuple[str, ...] = ("cuda:0", "cuda:1"),
) -> dict[str, Any]:
    """Load a completed result only after full provenance and matrix verification."""

    verify_complete_root(root=root, devices=devices)
    final = precoverage.read_json(root / "FINAL_STATUS.json")
    bundle: dict[str, Any] = {
        "protocol": PROTOCOL,
        "root": str(root.resolve()),
        "final_status": final,
        "science_contract": SCIENCE_CONTRACT,
    }
    if final.get("status") == "SCIENCE_COMPLETE":
        bundle["summary"] = precoverage.read_json(root / SCIENCE_PHASE / "summary.json")
        bundle["coverage_audit"] = precoverage.read_json(
            root / SCIENCE_PHASE / "coverage_audit.json"
        )
    return bundle


def write_or_verify_json(path: Path, value: object) -> None:
    if path.exists():
        if path.is_symlink() or precoverage.read_json(path) != value:
            raise RuntimeError(f"existing JSON artifact differs: {path}")
        return
    precoverage.write_json(path, value)


def write_or_verify_text(path: Path, value: str) -> None:
    if path.exists():
        if path.is_symlink() or path.read_text() != value:
            raise RuntimeError(f"existing text artifact differs: {path}")
        return
    precoverage.write_text(path, value)


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    main()
