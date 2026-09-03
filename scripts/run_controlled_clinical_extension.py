"""Run the isolated v2 dataset-native controlled clinical extension.

The runner first executes outcome-blind support, held-out logging-mixture K0
fidelity, and gamma=-4 donor-overlap preflights.  Coverage is never opened for
a dataset that fails support or K0 fidelity.  Low donor overlap instead marks
all saved curves descriptive-only; it does not erase the scientific rows.

This file parameterizes the frozen controlled-v1 mechanism without changing
the canonical SC-PCP implementation or the existing MIMIC-IV benchmark.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import re
import sys
import tarfile
from typing import Any, Iterable, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import Tensor
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_controlled_prefix_benchmark import (  # noqa: E402
    _empirical_rank_by_stage,
    _prefix_diagnostics,
)
from scripts.run_controlled_six_method_benchmark import (  # noqa: E402
    INFORMATION_REGIME,
    TARGET_ADAPTATION_BUDGET,
    ControlledOnlineEnvironment,
    _adaptation_seeds,
    _artifact_seeds,
    _assert_unique_rng_streams,
    _bootstrap_indices,
    _controlled_online_rollout,
    _evaluate_method,
    _paired_scpcp_comparison,
    _percentile_interval,
    _student_t_interval,
    _source_declared_seeds,
    _wilson_interval,
)
from scpcp.artifacts import experiment_tree_sha256  # noqa: E402
from scpcp.baselines import (  # noqa: E402
    aci_style_controller,
    finite_depth_mfcs_selection,
    multidim_spci_style_controller,
    prc_profile_scale,
    standard_cp_stagewise_radii,
)
from scpcp.behavior import fit_behavior_policy  # noqa: E402
from scpcp.config import ExperimentConfig  # noqa: E402
from scpcp.controlled_clinical_extension import (  # noqa: E402
    DATASET_NAMES,
    GAMMAS,
    METHODS,
    PROTOCOL,
    ClinicalExtensionSplits,
    ControlledClinicalExtensionConfig,
    DatasetPreset,
    DonorOverlapMetrics,
    K0FidelityMetrics,
    donor_overlap_passes,
    empirical_ks,
    equal_sample_wasserstein_1,
    evaluate_support_gate,
    k0_fidelity_passes,
    load_extension_config,
    setting_availability_passes,
    split_clinical_extension_roles,
    unique_patient_action_counts,
)
from scpcp.controlled_policy import ControlledMixturePolicy  # noqa: E402
from scpcp.controlled_transition import (  # noqa: E402
    ControlledResidualEnvironment,
    make_controlled_noise,
    rollout_controlled,
)
from scpcp.coverage import fixed_q_grid, profiled_scale_grid, stage_score_profile  # noqa: E402
from scpcp.experiment import _paper_seed, _training_outcome_sd  # noqa: E402
from scpcp.marginal_prefix import select_marginal_prefix_schedule  # noqa: E402
from scpcp.outcome_model import fit_outcome_model  # noqa: E402
from scpcp.policy.anchored import BehaviorAnchoredPolicy  # noqa: E402
from scpcp.real_data import (  # noqa: E402
    _coarsen_cxr_actions,
    _discretize_actions,
    _load_or_build_raw,
    _predictor_rows,
    load_clinical_trajectories,
)
from scpcp.scores import fit_conformal_region, score_batch  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "controlled_clinical_extension.yaml"
PRECOVERAGE_FAILURE_ARCHIVE = Path(
    "results/work/precoverage_failure_archives/"
    "controlled_clinical_extension_v2_failed_precoverage_cuda_median_"
    "20260826_210800.tar"
)
PRECOVERAGE_FAILURE_ARCHIVE_SHA256 = (
    "0fec4f676d86dce583bd135622bd7e888d0dac0198aa63a12d3b64ef96664906"
)
PRECOVERAGE_FAILURE_ARCHIVE_BYTES = 3_194_880
FAILED_SOURCE_TREE_SHA256 = (
    "ffbe0be73c5666358128f772e0e94029335632e90b221864df789de839b144c4"
)
FAILED_SOURCE_ARCHIVE_SHA256 = (
    "d16dae929122a96baae9133b5316dca2ad914ba2abb5b67d3c2a41b472c06580"
)
POSTCOMPUTE_FAILURE_ARCHIVE = Path(
    "results/work/precoverage_failure_archives/"
    "controlled_clinical_extension_v2_failed_postcompute_preinspection_"
    "json_key_order_20260826_223159.tar"
)
POSTCOMPUTE_FAILURE_ARCHIVE_SHA256 = (
    "bfd28a92a574bac0e25e3ec5f3b03ef5c5c33ef319ac996e57b766e292d9e54e"
)
POSTCOMPUTE_FAILURE_ARCHIVE_BYTES = 10_168_320
POSTCOMPUTE_FAILED_SOURCE_TREE_SHA256 = (
    "4dc07d23dd7dc7c89c13952235ef00af44fcc6fc9cb63de7b60c27339e2f7c54"
)
POSTCOMPUTE_FAILED_SOURCE_ARCHIVE_SHA256 = (
    "6e65fc60f3f2532f00001408f7f46962cb8bc184e1880c089a25d4754efee3e3"
)
BOOTSTRAP_RESAMPLES = 10_000
K0_UNIFORM_SEED_OFFSET = 90_000_000
K0_SYSTEMATIC_REPLAYS = 16
K0_PATIENT_CHUNK_SIZE = 128
OVERLAP_STREAM_SALT = 1_700_301
CALIBRATION_STREAM_SALT = 1_700_101
REFERENCE_STREAM_SALT = 1_700_401
_SEED_NAME = re.compile(r"seed_(\d+)(?:\.json)?$")
_SEED_ASSIGNMENT = re.compile(r"seed", re.IGNORECASE)

# Dataset-native quantities (horizon, donor bandwidth, and action ontology) are
# deliberately excluded.  Everything below is the frozen common scientific
# contract shared by all four clinical settings.
COMMON_SCIENTIFIC_CONTRACT: dict[str, Any] = {
    "model": {
        "architecture": "gru",
        "history_length": 4,
        "hidden_dim": 128,
        "representation_dim": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 512,
        "epochs": 100,
        "patience": 10,
        "min_scale": 1e-3,
        "gradient_clip": 5.0,
    },
    "policy_without_action_ontology_or_controlled_cap": {
        "tilt": 1.0,
        "temperature": 1.0,
        "disease_weight": 0.5,
        "toxicity_weight": 0.5,
        "propensity_floor": 0.01,
    },
    "cot": {
        "hidden_dims": (128, 64),
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 512,
        "epochs": 100,
        "patience": 10,
        "gradient_clip": 5.0,
        "rho_cap": 4.0,
        "weight_cap": 40.0,
        "normalization_penalty": 1.0,
        "validation_fraction": 0.15,
        "q_samples_per_batch": 4,
        "loss": "huber",
    },
    "profile": {
        "refinement_folds": 3,
        "refinement_strength": 0.5,
        "maximum_profile_ratio": 1.25,
        "minimum_effective_size": 25.0,
        "maximum_cap_hit_rate": 0.01,
        "grid_focus_fraction": 0.80,
        "grid_focus_radius": 0.075,
    },
    "certification": {
        "alpha": 0.10,
        "delta": 0.05,
        "ratio_error_bound": 0.0,
        "ratio_bound_source": "none",
        "ratio_delta": 0.0,
        "practical_bootstrap_resamples": 2_000,
    },
    "samples": {
        "logged": 5_000,
        "oracle_rollouts": 50_000,
        "oracle_surface_rollouts": 5_000,
        "online_rollouts": 2_000,
    },
    "baselines": {
        "mfcs_depth": 3,
        "aci_gamma": 0.01,
        "multidim_buffer": 1_000,
        "online_rounds": 3,
        "prc_maximum_step": 0.35,
    },
    "candidate_grid": {
        "q_grid_size": 101,
        "q_quantile_min": 0.50,
        "q_quantile_max": 0.999,
    },
}


@dataclass(frozen=True)
class ExtensionContext:
    config: ExperimentConfig
    splits: ClinicalExtensionSplits
    n_actions: int
    static_indices: tuple[int, ...]
    action_costs: tuple[float, ...]
    action_mapping: dict[int, int]
    state_feature_names: tuple[str, ...]
    outcome_model: object
    region: object
    logging_policy: object
    target_policy: ControlledMixturePolicy
    environment: ControlledResidualEnvironment
    action_coordinate: Tensor
    outcome_sd: Tensor
    q_low: float
    q_high: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(DATASET_NAMES),
        help="comma-separated frozen clinical dataset names",
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    run_extension(
        args.output_root.resolve(),
        datasets=datasets,
        devices=devices,
        resume=args.resume,
    )
    print(args.output_root.resolve())


def run_extension(
    output_root: Path,
    *,
    datasets: tuple[str, ...],
    devices: tuple[str, ...],
    resume: bool = False,
) -> None:
    protocol = load_extension_config(CONFIG_PATH)
    retry_amendment = _verified_precoverage_retry_amendment()
    postcompute_retry_amendment = _verified_postcompute_retry_amendment()
    _validate_launch(protocol, datasets=datasets, devices=devices)
    active_hash = experiment_tree_sha256()
    snapshot = _build_source_snapshot()
    contracts = {
        dataset: _dataset_contract(protocol, protocol.datasets[dataset])
        for dataset in datasets
    }
    seed_mapping = _stable_device_mapping(protocol, datasets, devices)
    stored_metadata = None
    if resume:
        metadata_path = output_root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError("resume requires an existing root metadata.json")
        stored_metadata = _read_json(metadata_path)
    fresh_rng_audit = _audit_rng_banks(protocol, datasets, output_root=output_root)
    rng_audit = (
        fresh_rng_audit
        if stored_metadata is None
        else _resume_rng_identity(stored_metadata, fresh_rng_audit)
    )
    metadata = _root_metadata(
        protocol,
        datasets=datasets,
        devices=devices,
        source_hash=active_hash,
        contracts=contracts,
        seed_mapping=seed_mapping,
        rng_audit=rng_audit,
        source_snapshot=snapshot["contract"],
        retry_amendment=retry_amendment,
        postcompute_retry_amendment=postcompute_retry_amendment,
    )
    _validate_retry_amendment_binding(metadata, retry_amendment)
    _validate_postcompute_retry_amendment_binding(
        metadata, postcompute_retry_amendment
    )
    metadata_path = output_root / "metadata.json"
    if resume:
        if _json_sha256(stored_metadata) != _json_sha256(metadata):
            raise RuntimeError("resume metadata does not match the active v2 protocol")
        _verify_source_snapshot(output_root, snapshot["contract"])
        if (output_root / "COMPLETE").exists():
            _verify_manifest(output_root)
            expected_complete = _root_complete_marker(
                snapshot["contract"], retry_amendment, postcompute_retry_amendment
            )
            if (output_root / "COMPLETE").read_text() != expected_complete:
                raise RuntimeError("root COMPLETE source-snapshot binding differs")
    else:
        if output_root.exists():
            raise FileExistsError(f"fresh output already exists: {output_root}")
        output_root.mkdir(parents=True)
        _publish_source_snapshot(output_root, snapshot)
        _write_json(metadata_path, metadata)

    completed = []
    for dataset in datasets:
        dataset_root = output_root / dataset
        _run_dataset(
            dataset_root,
            protocol=protocol,
            preset=protocol.datasets[dataset],
            devices=devices,
            source_hash=active_hash,
            contract=contracts[dataset],
            seed_to_device={
                seed: seed_mapping[f"{dataset}/{seed}"]
                for seed in protocol.datasets[dataset].seeds
            },
            rng_audit=rng_audit,
            retry_amendment=retry_amendment,
            postcompute_retry_amendment=postcompute_retry_amendment,
            resume=resume,
        )
        completed.append(dataset)
    if experiment_tree_sha256() != active_hash:
        raise RuntimeError("source/config tree changed while the extension was running")
    for dataset in datasets:
        closing_contract = _dataset_contract(protocol, protocol.datasets[dataset])
        if _json_sha256(closing_contract) != _json_sha256(contracts[dataset]):
            raise RuntimeError(
                f"{dataset} raw/cache/image/checkpoint provenance changed during the run"
            )
    closing_retry_amendment = _verified_precoverage_retry_amendment()
    if _json_sha256(closing_retry_amendment) != _json_sha256(retry_amendment):
        raise RuntimeError(
            "precoverage retry provenance changed while the extension was running"
        )
    closing_postcompute_retry_amendment = _verified_postcompute_retry_amendment()
    if _json_sha256(closing_postcompute_retry_amendment) != _json_sha256(
        postcompute_retry_amendment
    ):
        raise RuntimeError(
            "postcompute retry provenance changed while the extension was running"
        )
    _validate_retry_amendment_binding(_read_json(metadata_path), retry_amendment)
    _validate_postcompute_retry_amendment_binding(
        _read_json(metadata_path), postcompute_retry_amendment
    )
    retry_amendment_sha256 = _json_sha256(retry_amendment)
    postcompute_retry_amendment_sha256 = _json_sha256(postcompute_retry_amendment)
    _write_json(
        output_root / "summary.json",
        {
            "protocol": PROTOCOL,
            "datasets": list(datasets),
            "completed_datasets": completed,
            "dataset_status": {
                dataset: _read_json(output_root / dataset / "FINAL_STATUS.json")
                for dataset in datasets
            },
            "precoverage_engineering_retry_amendment_sha256": (
                retry_amendment_sha256
            ),
            "postcompute_preinspection_retry_amendment_sha256": (
                postcompute_retry_amendment_sha256
            ),
        },
    )
    _write_manifest(output_root)
    _write_text(
        output_root / "COMPLETE",
        _root_complete_marker(
            snapshot["contract"], retry_amendment, postcompute_retry_amendment
        ),
    )


def _validate_launch(
    protocol: ControlledClinicalExtensionConfig,
    *,
    datasets: tuple[str, ...],
    devices: tuple[str, ...],
) -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from the workspace root so relative caches resolve exactly: {ROOT}")
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("datasets must be a nonempty duplicate-free list")
    unknown = set(datasets) - set(DATASET_NAMES)
    if unknown:
        raise ValueError(f"unknown controlled clinical datasets: {sorted(unknown)}")
    if datasets != tuple(name for name in DATASET_NAMES if name in datasets):
        raise ValueError("datasets must follow the frozen protocol order")
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("the v2 extension requires exactly two distinct CUDA devices")
    if any(not device.startswith("cuda:") for device in devices):
        raise ValueError("the extension requires explicit CUDA devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is not permitted")
    for device in devices:
        index = torch.device(device).index
        if index is None or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {device}")
    protocol.validate()


def _run_dataset(
    output_root: Path,
    *,
    protocol: ControlledClinicalExtensionConfig,
    preset: DatasetPreset,
    devices: tuple[str, ...],
    source_hash: str,
    contract: dict[str, Any],
    seed_to_device: dict[int, str],
    rng_audit: dict[str, Any],
    retry_amendment: dict[str, Any],
    postcompute_retry_amendment: dict[str, Any],
    resume: bool,
) -> None:
    retry_amendment_sha256 = _json_sha256(retry_amendment)
    dataset_metadata = {
        "protocol": PROTOCOL,
        "dataset": preset.name,
        "source_tree_sha256": source_hash,
        "dataset_contract": contract,
        "dataset_contract_sha256": _json_sha256(contract),
        "seed_to_device": {str(seed): seed_to_device[seed] for seed in preset.seeds},
        "rng_stream_mapping_sha256": rng_audit["new_rng_stream_mapping_sha256"],
        "precoverage_engineering_retry_amendment": retry_amendment,
        "precoverage_engineering_retry_amendment_sha256": (
            retry_amendment_sha256
        ),
        "postcompute_preinspection_retry_amendment": postcompute_retry_amendment,
        "postcompute_preinspection_retry_amendment_sha256": _json_sha256(
            postcompute_retry_amendment
        ),
    }
    _validate_retry_amendment_binding(dataset_metadata, retry_amendment)
    _validate_postcompute_retry_amendment_binding(
        dataset_metadata, postcompute_retry_amendment
    )
    if resume:
        if not output_root.exists():
            output_root.mkdir(parents=True)
            _write_json(output_root / "metadata.json", dataset_metadata)
        else:
            stored = _read_json(output_root / "metadata.json")
            if _json_sha256(stored) != _json_sha256(dataset_metadata):
                raise RuntimeError(f"{preset.name} resume metadata mismatch")
        final = output_root / "FINAL_STATUS.json"
        complete = output_root / "COMPLETE"
        if complete.exists() and not final.exists():
            raise RuntimeError(f"{preset.name} COMPLETE exists without FINAL_STATUS")
        if final.exists() and complete.exists():
            _validate_final_dataset_bundle(
                output_root,
                protocol=protocol,
                preset=preset,
                dataset_metadata=dataset_metadata,
            )
            return
        # FINAL_STATUS without COMPLETE is only a crash-recoverable draft.
        # Resume rederives every phase and republishes the terminal bundle.
    else:
        if output_root.exists():
            raise FileExistsError(f"fresh dataset output already exists: {output_root}")
        output_root.mkdir(parents=True)
        _write_json(output_root / "metadata.json", dataset_metadata)

    support = _run_phase(
        output_root / "support",
        preset=preset,
        seed_to_device=seed_to_device,
        phase="support",
        worker=_support_worker,
        worker_arguments=(protocol,),
        seed_contract=dataset_metadata,
        resume=resume,
    )
    supported = tuple(int(row["seed"]) for row in support if bool(row["passed"]))
    support_pass = setting_availability_passes(
        len(supported), len(preset.seeds), protocol.support_gate.minimum_available_seed_fraction
    )
    _write_json(
        output_root / "support" / "summary.json",
        _gate_summary("support", preset.seeds, supported, support_pass),
    )
    if not support_pass:
        _publish_no_go(
            output_root,
            preset=preset,
            reason="SUPPORT_NO_GO",
            detail="fewer than 19/20 prespecified seeds passed unique D_env support",
        )
        _validate_final_dataset_bundle(
            output_root,
            protocol=protocol,
            preset=preset,
            dataset_metadata=dataset_metadata,
        )
        return

    fidelity = _run_phase(
        output_root / "k0_fidelity",
        preset=replace(preset, seeds=supported),
        seed_to_device=seed_to_device,
        phase="k0_fidelity",
        worker=_k0_worker,
        worker_arguments=(protocol,),
        seed_contract=dataset_metadata,
        resume=resume,
    )
    _assert_support_context_consistency(support, fidelity)
    fidelity_passed = tuple(int(row["seed"]) for row in fidelity if bool(row["passed"]))
    structural_failures = tuple(
        int(row["seed"])
        for row in fidelity
        if not bool(row["metrics"]["structural_invariants"])
    )
    k0_pass = setting_availability_passes(
        len(fidelity_passed),
        len(preset.seeds),
        protocol.k0_fidelity_gate.minimum_available_seed_fraction,
    )
    _write_json(
        output_root / "k0_fidelity" / "summary.json",
        {
            **_gate_summary(
                "logging-mixture one-step fidelity",
                preset.seeds,
                fidelity_passed,
                k0_pass and not structural_failures,
            ),
            "structural_failure_seeds": list(structural_failures),
            "structural_rule": "any exact-invariant failure is terminal",
            "numeric_availability_rule": ">=19/20 only after all exact invariants pass",
        },
    )
    if structural_failures:
        _publish_no_go(
            output_root,
            preset=preset,
            reason="STRUCTURAL_NO_GO",
            detail=(
                "one or more K0 seeds violated exact transition invariants: "
                f"{list(structural_failures)}"
            ),
        )
        _validate_final_dataset_bundle(
            output_root,
            protocol=protocol,
            preset=preset,
            dataset_metadata=dataset_metadata,
        )
        return
    if not k0_pass:
        _publish_no_go(
            output_root,
            preset=preset,
            reason="K0_FIDELITY_NO_GO",
            detail="fewer than 19/20 prespecified seeds passed K0 logging-mixture one-step fidelity",
        )
        _validate_final_dataset_bundle(
            output_root,
            protocol=protocol,
            preset=preset,
            dataset_metadata=dataset_metadata,
        )
        return

    overlap = _run_phase(
        output_root / "donor_overlap",
        preset=replace(preset, seeds=fidelity_passed),
        seed_to_device=seed_to_device,
        phase="donor_overlap",
        worker=_overlap_worker,
        worker_arguments=(protocol,),
        seed_contract=dataset_metadata,
        resume=resume,
    )
    _assert_context_consistency(fidelity, overlap, label="K0/donor-overlap")
    donor_interpretation_pass = all(bool(row["passed"]) for row in overlap)
    interpretation = (
        "EMPIRICAL_OVERLAP_SCREEN_PASSED"
        if donor_interpretation_pass
        else "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
    )
    _write_json(
        output_root / "donor_overlap" / "summary.json",
        {
            **_gate_summary(
                "gamma=-4 q_mid+q_high empirical donor-overlap screen",
                fidelity_passed,
                tuple(int(row["seed"]) for row in overlap if bool(row["passed"])),
                donor_interpretation_pass,
            ),
            "failure_consequence": (
                "save and render full curves, but exclude confirmatory ranking, attainment, "
                "superiority, and cross-dataset conjunction"
            ),
            "interpretation_status": interpretation,
        },
    )

    science = _run_phase(
        output_root / "science",
        preset=replace(preset, seeds=fidelity_passed),
        seed_to_device=seed_to_device,
        phase="science",
        worker=_science_worker,
        worker_arguments=(protocol, interpretation),
        seed_contract=dataset_metadata,
        resume=resume,
    )
    _assert_context_consistency(fidelity, science, label="K0/science")
    rows = [row for seed_payload in science for row in seed_payload["rows"]]
    bootstrap_contract = _write_bootstrap_artifacts(
        output_root / "science",
        preset=preset,
        resamples=protocol.bootstrap_resamples,
    )
    summary = summarize_science(
        rows,
        preset=preset,
        selected_seeds=fidelity_passed,
        interpretation_status=interpretation,
        bootstrap_contract=bootstrap_contract,
    )
    _write_json(output_root / "science" / "summary.json", summary)
    _write_json(
        output_root / "FINAL_STATUS.json",
        {
            "protocol": PROTOCOL,
            "dataset": preset.name,
            "status": "COMPLETE",
            "scientific_rows_saved": True,
            "interpretation_status": interpretation,
            "support_available": len(supported),
            "k0_fidelity_available": len(fidelity_passed),
            "prespecified_seeds": len(preset.seeds),
        },
    )
    _write_json(
        output_root / "gate.json",
        {
            "protocol": PROTOCOL,
            "dataset": preset.name,
            "panel_status": (
                "CURVES" if donor_interpretation_pass else "CURVES_DESCRIPTIVE_ONLY"
            ),
            "interpretation_status": interpretation,
        },
    )
    _write_json(
        output_root / "summary.json",
        {
            "protocol": PROTOCOL,
            "dataset": preset.name,
            "status": "COMPLETE",
            "interpretation_status": interpretation,
            "science_summary_path": "science/summary.json",
            "scientific_rows_saved": True,
        },
    )
    _write_manifest(output_root)
    _write_text(
        output_root / "COMPLETE",
        "curves\n" if donor_interpretation_pass else "curves-descriptive-only\n",
    )
    _validate_final_dataset_bundle(
        output_root,
        protocol=protocol,
        preset=preset,
        dataset_metadata=dataset_metadata,
    )


def _publish_no_go(
    output_root: Path,
    *,
    preset: DatasetPreset,
    reason: str,
    detail: str,
) -> None:
    science_root = output_root / "science"
    if science_root.exists():
        raise RuntimeError("a preflight NO-GO cannot coexist with scientific rows")
    status = {
        "protocol": PROTOCOL,
        "dataset": preset.name,
        "status": reason,
        "detail": detail,
        "scientific_rows_saved": False,
    }
    _write_json(
        output_root / "gate.json",
        {
            "protocol": PROTOCOL,
            "dataset": preset.name,
            "panel_status": "GATE_NO_GO",
            "reason": reason,
        },
    )
    _write_json(output_root / "summary.json", status)
    _write_json(output_root / "NO_GO.json", status)
    _write_json(output_root / "FINAL_STATUS.json", status)
    _write_manifest(output_root)
    _write_text(output_root / "COMPLETE", "gate-no-go\n")


def _run_phase(
    phase_root: Path,
    *,
    preset: DatasetPreset,
    seed_to_device: Mapping[int, str],
    phase: str,
    worker: object,
    worker_arguments: tuple[object, ...],
    seed_contract: dict[str, Any],
    resume: bool,
) -> list[dict[str, Any]]:
    phase_root.mkdir(parents=True, exist_ok=True)
    expected = {phase_root / f"seed_{seed:05d}.json" for seed in preset.seeds}
    unexpected = set(phase_root.glob("seed_*.json")) - expected
    if unexpected:
        raise RuntimeError(f"unexpected {phase} seed artifacts: {sorted(unexpected)}")
    completed: dict[int, dict[str, Any]] = {}
    for seed in preset.seeds:
        path = phase_root / f"seed_{seed:05d}.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        if not _valid_phase_payload(
            payload,
            phase=phase,
            preset=preset,
            seed=seed,
            device=seed_to_device[seed],
            seed_contract=seed_contract,
        ):
            raise RuntimeError(f"malformed or provenance-mismatched phase artifact: {path}")
        completed[seed] = payload["result"]
    pending = tuple(seed for seed in preset.seeds if seed not in completed)
    if pending and (phase_root / "COMPLETE").exists():
        raise RuntimeError(f"{phase} COMPLETE exists with missing seed artifacts")
    groups = tuple(
        tuple(seed for seed in pending if seed_to_device[seed] == device)
        for device in dict.fromkeys(seed_to_device.values())
    )
    devices = tuple(dict.fromkeys(seed_to_device.values()))
    if pending:
        with ProcessPoolExecutor(
            max_workers=len(devices),
            mp_context=get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(
                    _phase_group,
                    group,
                    device,
                    preset,
                    phase,
                    worker,
                    worker_arguments,
                ): device
                for group, device in zip(groups, devices)
                if group
            }
            for future in as_completed(futures):
                for seed, device, result in future.result():
                    payload = {
                        "protocol": PROTOCOL,
                        "dataset": preset.name,
                        "phase": phase,
                        "seed": seed,
                        "device": device,
                        "dataset_contract_sha256": seed_contract["dataset_contract_sha256"],
                        "source_tree_sha256": seed_contract["source_tree_sha256"],
                        "rng_stream_mapping_sha256": seed_contract["rng_stream_mapping_sha256"],
                        "result": result,
                    }
                    if not _valid_phase_payload(
                        payload,
                        phase=phase,
                        preset=preset,
                        seed=seed,
                        device=device,
                        seed_contract=seed_contract,
                    ):
                        raise RuntimeError(
                            f"worker returned an invalid {phase} payload for seed {seed}"
                        )
                    _write_json(phase_root / f"seed_{seed:05d}.json", payload)
                    completed[seed] = result
    if set(completed) != set(preset.seeds):
        raise RuntimeError(f"{phase} did not complete every requested seed")
    _write_text(phase_root / "COMPLETE", "complete\n")
    return [completed[seed] for seed in preset.seeds]


def _phase_group(
    seeds: tuple[int, ...],
    device: str,
    preset: DatasetPreset,
    phase: str,
    worker: object,
    worker_arguments: tuple[object, ...],
) -> list[tuple[int, str, dict[str, Any]]]:
    torch.cuda.set_device(torch.device(device))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    results = []
    for seed in seeds:
        result = worker(seed, preset, device, *worker_arguments)  # type: ignore[operator]
        results.append((seed, device, result))
        torch.cuda.empty_cache()
    return results


def _support_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    """Outcome-blind D_env support preflight; outcomes/scores are never read."""

    del device
    config = _controlled_config(preset, protocol)
    raw = _load_or_build_raw(config, seed=seed)
    if preset.name == "mimic_cxr":
        raw = _coarsen_cxr_actions(raw)
    predictor_rows = _predictor_rows(
        raw,
        seed,
        predictor_fraction=protocol.split_fractions[0],
    )
    actions, active_actions, direct_to_model = _discretize_actions(raw, predictor_rows)
    original_to_direct = raw.original_to_direct_action or {
        action: action for action in range(max(active_actions) + 1)
    }
    action_mapping = {
        int(original): int(direct_to_model[direct])
        for original, direct in original_to_direct.items()
    }
    placeholder = _trajectory_placeholder(raw.states, actions, raw.patient_ids)
    splits = split_clinical_extension_roles(
        placeholder,
        seed=seed,
        fractions=protocol.split_fractions,
    )
    counts = unique_patient_action_counts(splits.environment, len(active_actions))
    gate = evaluate_support_gate(counts, protocol.support_gate)
    episode = _episode_support_summary(splits.environment.patient_ids)
    return {
        "seed": seed,
        "dataset": preset.name,
        "phase": "support",
        "outcome_blind": True,
        "passed": gate.passed,
        "minimum_unique_patients": gate.minimum_unique_patients,
        "failed_cells": [list(cell) for cell in gate.failed_cells],
        "unique_patient_counts_by_stage_action": counts,
        "n_actions": len(active_actions),
        "active_action_indices": list(active_actions),
        "action_mapping": {str(key): value for key, value in action_mapping.items()},
        "action_costs": [float(config.policy.action_costs[index]) for index in active_actions],
        "environment_episode_support": episode,
        "split_audit": _split_audit(splits),
    }


def _k0_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    context = _prepare_extension_context(seed, preset, device, protocol)
    metrics, detail = _logging_mixture_fidelity(context, seed=seed, protocol=protocol)
    passed = k0_fidelity_passes(metrics, protocol.k0_fidelity_gate)
    return {
        "seed": seed,
        "dataset": preset.name,
        "phase": "k0_fidelity",
        "gate_name": "logging-mixture one-step fidelity",
        "passed": passed,
        "metrics": asdict(metrics),
        "systematic_replay": detail,
        "q_low": context.q_low,
        "q_high": context.q_high,
        "n_actions": context.n_actions,
        "action_mapping": {str(key): value for key, value in context.action_mapping.items()},
        "split_audit": _split_audit(context.splits),
        "context_identity": _context_identity(context),
    }


def _overlap_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    context = _prepare_extension_context(seed, preset, device, protocol)
    metrics, diagnostics = _donor_overlap_probe(context, seed=seed, protocol=protocol)
    passed = donor_overlap_passes(metrics, protocol.donor_overlap_gate)
    return {
        "seed": seed,
        "dataset": preset.name,
        "phase": "donor_overlap",
        "passed": passed,
        "interpretation_if_failed": "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
        "metrics": asdict(metrics),
        "diagnostics": diagnostics,
        "q_low": context.q_low,
        "q_high": context.q_high,
        "q_mid": context.q_low + 0.5 * (context.q_high - context.q_low),
        "n_actions": context.n_actions,
        "action_mapping": {str(key): value for key, value in context.action_mapping.items()},
        "split_audit": _split_audit(context.splits),
        "context_identity": _context_identity(context),
    }


def _science_worker(
    seed: int,
    preset: DatasetPreset,
    device: str,
    protocol: ControlledClinicalExtensionConfig,
    interpretation_status: str,
) -> dict[str, Any]:
    context = _prepare_extension_context(seed, preset, device, protocol)
    context_rows = run_science_seed(
        seed,
        preset=preset,
        device=device,
        protocol=protocol,
        context=context,
    )
    return {
        "seed": seed,
        "dataset": preset.name,
        "phase": "science",
        "interpretation_status": interpretation_status,
        "rows": context_rows,
        "q_low": context.q_low,
        "q_high": context.q_high,
        "n_actions": context.n_actions,
        "action_mapping": {str(key): value for key, value in context.action_mapping.items()},
        "split_audit": _split_audit(context.splits),
        "context_identity": _context_identity(context),
    }


def _controlled_config(
    preset: DatasetPreset,
    protocol: ControlledClinicalExtensionConfig,
) -> ExperimentConfig:
    path = ROOT / preset.base_config
    config = ExperimentConfig.from_yaml(path)
    if config.data.dataset != preset.name or config.horizon != preset.horizon:
        raise RuntimeError(f"{preset.name} base config does not match its frozen preset")
    if config.samples.online_rollouts != protocol.online_trajectories:
        raise RuntimeError("online baseline budget differs from the frozen v2 contract")
    if config.data.empirical_neighbors != protocol.support_gate.neighbors:
        raise RuntimeError("donor-neighbor count differs from the frozen v2 contract")
    observed_common = _common_scientific_contract(config)
    if observed_common != COMMON_SCIENTIFIC_CONTRACT:
        raise RuntimeError(
            "base config common scientific tuple differs from the frozen v2 contract"
        )
    return replace(
        config,
        policy=replace(config.policy, policy_ratio_cap=protocol.policy_ratio_cap),
    )


def _common_scientific_contract(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "model": asdict(config.model),
        "policy_without_action_ontology_or_controlled_cap": {
            "tilt": config.policy.tilt,
            "temperature": config.policy.temperature,
            "disease_weight": config.policy.disease_weight,
            "toxicity_weight": config.policy.toxicity_weight,
            "propensity_floor": config.policy.propensity_floor,
        },
        "cot": asdict(config.cot),
        "profile": asdict(config.profile),
        "certification": asdict(config.certification),
        "samples": asdict(config.samples),
        "baselines": asdict(config.baselines),
        "candidate_grid": {
            "q_grid_size": config.q_grid_size,
            "q_quantile_min": config.q_quantile_min,
            "q_quantile_max": config.q_quantile_max,
        },
    }


def _prepare_extension_context(
    seed: int,
    preset: DatasetPreset,
    device: str,
    protocol: ControlledClinicalExtensionConfig,
) -> ExtensionContext:
    config = _controlled_config(preset, protocol)
    torch.manual_seed(seed)
    (
        batch,
        n_actions,
        static_indices,
        action_costs,
        action_mapping,
        state_feature_names,
    ) = load_clinical_trajectories(
        config,
        seed=seed,
        device=device,
        predictor_fraction=protocol.split_fractions[0],
    )
    splits = split_clinical_extension_roles(
        batch,
        seed=seed,
        fractions=protocol.split_fractions,
    )
    _seed_torch_stream(seed + 1, device)
    outcome_model = fit_outcome_model(
        splits.predictor,
        n_actions=n_actions,
        config=config.model,
        device=device,
        seed=seed + 1,
        static_indices=static_indices,
    )
    region = fit_conformal_region(outcome_model)
    _seed_torch_stream(seed + 2, device)
    logging_policy = fit_behavior_policy(
        splits.predictor,
        n_actions=n_actions,
        model_config=config.model,
        policy_config=replace(config.policy, action_costs=action_costs),
        device=device,
        seed=seed + 2,
        static_indices=static_indices,
        decision_time_index=(
            state_feature_names.index("decision_time")
            if "decision_time" in state_feature_names
            else None
        ),
    )
    fidelity_scores = score_batch(
        region,
        splits.fidelity.current_states(),
        splits.fidelity.actions,
        splits.fidelity.outcomes,
    )
    q_low = float(torch.quantile(fidelity_scores.flatten(), protocol.q_low_source_quantile).item())
    q_high = float(torch.quantile(fidelity_scores.flatten(), protocol.q_high_source_quantile).item())
    if not math.isfinite(q_low) or not math.isfinite(q_high) or not q_high > q_low:
        raise RuntimeError("D_fidelity did not produce a valid q80/q95 policy range")
    policy_config = replace(
        config.policy,
        action_costs=action_costs,
        policy_ratio_cap=protocol.policy_ratio_cap,
    )
    alternative_policy = BehaviorAnchoredPolicy(
        outcome_model=outcome_model,
        reference_policy=logging_policy,
        config=policy_config,
        region=region,
        tilt=protocol.alternative_policy_tilt,
    )
    target_policy = ControlledMixturePolicy(
        logging_policy=logging_policy,
        alternative_policy=alternative_policy,
        radius_low=q_low,
        radius_high=q_high,
        maximum_response=protocol.maximum_policy_response,
    )
    environment_scores = score_batch(
        region,
        splits.environment.current_states(),
        splits.environment.actions,
        splits.environment.outcomes,
    )
    environment = ControlledResidualEnvironment(
        splits.environment,
        outcome_model=outcome_model,
        n_actions=n_actions,
        difficulty=_empirical_rank_by_stage(environment_scores),
        history_length=config.model.history_length,
        static_indices=static_indices,
        state_feature_names=state_feature_names,
        neighbors=protocol.support_gate.neighbors,
        bandwidth=config.data.empirical_bandwidth,
        ridge=protocol.transition_ridge,
    )
    action_cost = torch.tensor(action_costs, device=device)
    span = action_cost.max() - action_cost.min()
    if float(span.item()) <= 0.0:
        raise RuntimeError("controlled action coordinate requires nonconstant action costs")
    action_coordinate = 2.0 * (action_cost - action_cost.min()) / span - 1.0
    return ExtensionContext(
        config=config,
        splits=splits,
        n_actions=n_actions,
        static_indices=static_indices,
        action_costs=action_costs,
        action_mapping=action_mapping,
        state_feature_names=state_feature_names,
        outcome_model=outcome_model,
        region=region,
        logging_policy=logging_policy,
        target_policy=target_policy,
        environment=environment,
        action_coordinate=action_coordinate,
        outcome_sd=_training_outcome_sd(splits.predictor).to(device),
        q_low=q_low,
        q_high=q_high,
    )


def _seed_torch_stream(seed: int, device: str) -> None:
    """Reset the explicit nuisance stream after any upstream ambient RNG use."""

    torch.manual_seed(seed)
    resolved = torch.device(device)
    if resolved.type == "cuda":
        with torch.cuda.device(resolved):
            torch.cuda.manual_seed(seed)


@torch.no_grad()
def _logging_mixture_fidelity(
    context: ExtensionContext,
    *,
    seed: int,
    protocol: ControlledClinicalExtensionConfig,
) -> tuple[K0FidelityMetrics, dict[str, Any]]:
    fidelity = context.splits.fidelity
    replay_count = protocol.k0_fidelity_gate.systematic_replays
    if replay_count != K0_SYSTEMATIC_REPLAYS:
        raise RuntimeError("K0 systematic mixture size must remain M=16")
    uniform_seed = K0_UNIFORM_SEED_OFFSET + seed
    generator = torch.Generator(device="cpu").manual_seed(uniform_seed)
    base_uniform = torch.rand(
        (fidelity.horizon, fidelity.n),
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    offsets = (torch.arange(replay_count, dtype=torch.float64) + 0.5) / replay_count
    systematic_uniform = (base_uniform[:, :, None] + offsets[None, None, :]).remainder(1.0)
    base_uniform_hash = hashlib.sha256(base_uniform.numpy().tobytes(order="C")).hexdigest()
    uniform_hash = hashlib.sha256(systematic_uniform.numpy().tobytes(order="C")).hexdigest()

    score_ks = []
    residual_w1 = []
    successor_mean_w1 = []
    successor_q95_w1 = []
    active_counts = []
    invariant_rows = []
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
        true_score = true_score_original.repeat_interleave(replay_count)
        replay_score_parts = []
        replay_residual_parts = []
        replay_representation_parts = []
        invariant_parts = []
        mean, scale = context.outcome_model(state, action)
        for start, stop, uniform_chunk in _systematic_uniform_chunks(
            systematic_uniform[stage],
            chunk_size=K0_PATIENT_CHUNK_SIZE,
        ):
            state_chunk = state[start:stop]
            action_chunk = action[start:stop]
            repeated_state = state_chunk.repeat_interleave(replay_count, dim=0)
            repeated_action = action_chunk.repeat_interleave(replay_count, dim=0)
            donor_uniform = uniform_chunk.to(state)
            replay_next, replay_outcome, _, replay_ess, replay_max = (
                context.environment.step_from_uniform(
                    repeated_state,
                    repeated_action,
                    donor_uniform,
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
                replay_count,
                dim=0,
            )
            replay_residual_parts.append(
                ((replay_outcome - repeated_mean) / repeated_scale).cpu()
            )
            replay_representation_parts.append(
                _representation(context.outcome_model, replay_next).to(torch.float64)
            )
            invariant_parts.append(
                _raw_transition_invariants(
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
        replay_rep = torch.cat(replay_representation_parts)
        score_ks.append(empirical_ks(true_score, replay_score))
        true_residual = (true_outcome - mean) / scale.clamp_min(1e-6)
        coordinate_w1 = equal_sample_wasserstein_1(
            true_residual.repeat_interleave(replay_count, dim=0),
            replay_residual,
        )
        residual_w1.append(float(coordinate_w1.max().item()))

        environment_true_rep = _representation(
            context.outcome_model,
            context.splits.environment.states[:, stage + 1],
        ).to(torch.float64)
        center = environment_true_rep.mean(dim=0)
        scale_rep = environment_true_rep.std(dim=0, unbiased=False)
        active = scale_rep > protocol.k0_fidelity_gate.active_coordinate_sd_floor
        active_counts.append(int(active.sum().item()))
        if active.any():
            true_rep = _representation(
                context.outcome_model,
                fidelity.states[:, stage + 1],
            ).to(torch.float64)
            true_standardized = (
                (true_rep - center) / scale_rep
            )[:, active].repeat_interleave(replay_count, dim=0)
            replay_standardized = ((replay_rep - center) / scale_rep)[:, active]
            successor_w1 = equal_sample_wasserstein_1(
                true_standardized,
                replay_standardized,
            )
            successor_mean_w1.append(float(successor_w1.mean().item()))
            successor_q95_w1.append(
                float(torch.quantile(successor_w1, 0.95, interpolation="linear").item())
            )
        else:
            successor_mean_w1.append(0.0)
            successor_q95_w1.append(0.0)
        invariant_rows.append(_merge_invariant_rows(invariant_parts))
    structural = all(row["passed"] for row in invariant_rows) and all(
        count > 0 for count in active_counts
    )
    metrics = K0FidelityMetrics(
        maximum_score_ks=max(score_ks),
        maximum_signed_residual_w1=max(residual_w1),
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
        "patient_chunk_size": K0_PATIENT_CHUNK_SIZE,
        "base_uniform_seed": uniform_seed,
        "base_uniform_shape": list(base_uniform.shape),
        "base_uniform_sha256": base_uniform_hash,
        "expansion_formula": "u[t,i,m]=(U[t,i]+(m+0.5)/16) mod 1",
        "flatten_order": "stage, patient, systematic_offset (offset fastest)",
        "expanded_uniform_sha256": uniform_hash,
        "score_ks_by_stage": score_ks,
        "signed_residual_max_w1_by_stage": residual_w1,
        "successor_mean_w1_by_stage": successor_mean_w1,
        "successor_q95_w1_by_stage": successor_q95_w1,
        "active_successor_coordinates_by_stage": active_counts,
        "raw_structural_invariants_by_stage": invariant_rows,
    }
    return metrics, detail


def _systematic_uniform_chunks(
    stage_uniform: Tensor,
    *,
    chunk_size: int,
) -> Iterable[tuple[int, int, Tensor]]:
    """Yield patient chunks while preserving patient-major, offset-fastest order."""

    if stage_uniform.ndim != 2 or len(stage_uniform) == 0 or chunk_size < 1:
        raise ValueError("systematic uniforms must be nonempty [N,M] with positive chunk size")
    for start in range(0, len(stage_uniform), chunk_size):
        stop = min(start + chunk_size, len(stage_uniform))
        yield start, stop, stage_uniform[start:stop].reshape(-1)


@torch.no_grad()
def _donor_overlap_probe(
    context: ExtensionContext,
    *,
    seed: int,
    protocol: ControlledClinicalExtensionConfig,
) -> tuple[DonorOverlapMetrics, dict[str, Any]]:
    gate = protocol.donor_overlap_gate
    noise_seed = _paper_seed(seed, OVERLAP_STREAM_SALT)
    noise = make_controlled_noise(
        n=gate.probe_trajectories,
        horizon=context.config.horizon,
        initial_count=context.environment.initial_count,
        seed=noise_seed,
        device=context.action_coordinate.device,
    )
    source = rollout_controlled(
        context.environment,
        context.logging_policy,
        noise=noise,
        gamma=gate.gamma,
        action_coordinate=context.action_coordinate,
    )
    labels = ("q_mid", "q_high")
    probes = {}
    for label, fraction in zip(labels, gate.probe_radius_fractions, strict=True):
        radius = context.q_low + fraction * (context.q_high - context.q_low)
        metrics, diagnostics = _single_donor_overlap_probe(
            context,
            source_batch=source.trajectories,
            noise=noise,
            radius=radius,
            protocol=protocol,
        )
        probes[label] = {
            "radius_fraction": fraction,
            "radius": radius,
            "metrics": asdict(metrics),
            "passed": donor_overlap_passes(metrics, gate),
            **diagnostics,
        }
    worst = DonorOverlapMetrics(
        local_ess_p01=min(float(probe["metrics"]["local_ess_p01"]) for probe in probes.values()),
        median_ess_fraction=min(
            float(probe["metrics"]["median_ess_fraction"]) for probe in probes.values()
        ),
        maximum_donor_probability=max(
            float(probe["metrics"]["maximum_donor_probability"])
            for probe in probes.values()
        ),
    )
    return worst, {
        "probe_trajectories": gate.probe_trajectories,
        "gamma": gate.gamma,
        "noise_seed": noise_seed,
        "common_random_numbers_across_radii": True,
        "independent_frozen_stream": True,
        "patient_aggregated": True,
        "episode_weighted_transition_patient_aggregated_diagnostics": True,
        "probes": probes,
        "worst_metrics": asdict(worst),
        "screen_status": (
            "EMPIRICAL_OVERLAP_SCREEN_PASSED"
            if all(bool(probe["passed"]) for probe in probes.values())
            else "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        ),
        "screen_scope": "gamma=-4 q_mid and q_high=max-response; empirical, not a guarantee",
        "environment_episode_support": _episode_support_summary(
            context.splits.environment.patient_ids
        ),
    }


@torch.no_grad()
def _single_donor_overlap_probe(
    context: ExtensionContext,
    *,
    source_batch: object,
    noise: object,
    radius: float,
    protocol: ControlledClinicalExtensionConfig,
) -> tuple[DonorOverlapMetrics, dict[str, Any]]:
    gate = protocol.donor_overlap_gate
    target = rollout_controlled(
        context.environment,
        context.target_policy,
        noise=noise,
        gamma=gate.gamma,
        action_coordinate=context.action_coordinate,
        radii=torch.full(
            (context.config.horizon,),
            radius,
            device=context.action_coordinate.device,
        ),
    )
    patient_ess, patient_maximum, unique_k = [], [], []
    target_simplex_errors = []
    logging_simplex_errors = []
    maximum_policy_ratios = []
    minimum_logging_probabilities = []
    minimum_target_probabilities = []
    policy_probabilities_finite = []
    for stage in range(context.config.horizon):
        states = target.trajectories.states[:, stage]
        actions = target.trajectories.actions[:, stage]
        probability = context.target_policy.probabilities(states, radius)
        logging_probability = context.logging_policy.probabilities(states)
        target_simplex_errors.append(
            float((probability.sum(dim=1) - 1.0).abs().max().item())
        )
        logging_simplex_errors.append(
            float((logging_probability.sum(dim=1) - 1.0).abs().max().item())
        )
        minimum_logging_probabilities.append(float(logging_probability.min().item()))
        minimum_target_probabilities.append(float(probability.min().item()))
        policy_probabilities_finite.append(
            bool(torch.isfinite(probability).all() and torch.isfinite(logging_probability).all())
        )
        ratio = probability / logging_probability
        maximum_policy_ratios.append(float(ratio.max().item()))
        ess, maximum, local_unique = context.environment.patient_aggregated_kernel_diagnostics(
            states,
            actions,
            time=stage,
            gamma=gate.gamma,
            action_coordinate=context.action_coordinate,
        )
        patient_ess.append(ess)
        patient_maximum.append(maximum)
        unique_k.append(local_unique)
    ess = torch.stack(patient_ess, dim=1)
    maximum = torch.stack(patient_maximum, dim=1)
    local_k = torch.stack(unique_k, dim=1)
    structural = (
        ess.numel() > 0
        and torch.isfinite(ess).all()
        and torch.isfinite(maximum).all()
        and torch.isfinite(local_k).all()
        and bool(ess.ge(1.0 - 1e-6).all())
        and bool(maximum.ge(0.0).logical_and(maximum.le(1.0 + 1e-6)).all())
        and bool(local_k.ge(1.0).all())
        and max(target_simplex_errors) <= 1e-5
        and max(logging_simplex_errors) <= 1e-5
        and min(minimum_logging_probabilities) > 0.0
        and min(minimum_target_probabilities) >= 0.0
        and all(policy_probabilities_finite)
        and all(math.isfinite(value) for value in maximum_policy_ratios)
        and max(maximum_policy_ratios) <= protocol.policy_ratio_cap + 1e-5
    )
    if not structural:
        raise RuntimeError("structural donor-overlap stop: nonfinite/empty/ESS/simplex failure")
    ess_fraction = ess / local_k
    metrics = DonorOverlapMetrics(
        local_ess_p01=float(
            torch.quantile(ess.flatten(), gate.local_ess_quantile, interpolation="linear").item()
        ),
        median_ess_fraction=float(torch.median(ess_fraction.flatten()).item()),
        maximum_donor_probability=float(maximum.max().item()),
    )
    prefix = _prefix_overlap_report(context, source_batch, radius=radius)
    return metrics, {
        "target_simplex_maximum_error": max(target_simplex_errors),
        "logging_simplex_maximum_error": max(logging_simplex_errors),
        "minimum_logging_probability": min(minimum_logging_probabilities),
        "minimum_target_probability": min(minimum_target_probabilities),
        "policy_probabilities_finite": all(policy_probabilities_finite),
        "maximum_single_step_target_to_logging_ratio": max(maximum_policy_ratios),
        "single_step_ratio_cap": protocol.policy_ratio_cap,
        "local_unique_k_minimum": float(local_k.min().item()),
        "local_unique_k_median": float(local_k.median().item()),
        "prefix_overlap_report_only": prefix,
    }


@torch.no_grad()
def _prefix_overlap_report(
    context: ExtensionContext,
    target_batch: object,
    *,
    radius: float,
) -> dict[str, Any]:
    schedule = torch.full(
        (context.config.horizon,),
        radius,
        device=context.action_coordinate.device,
    )
    _, ess_fraction, maximum_share, log_span = _prefix_diagnostics(
        target_batch,
        schedule=schedule,
        target_policy=context.target_policy,
        logging_policy=context.logging_policy,
    )
    return {
        "minimum_ess_fraction": float(ess_fraction.min().item()),
        "maximum_normalized_weight_share": float(maximum_share.max().item()),
        "maximum_raw_log_weight_span": float(log_span.max().item()),
        "gate_role": "report-only",
    }


def run_science_seed(
    seed: int,
    *,
    preset: DatasetPreset,
    device: str,
    protocol: ControlledClinicalExtensionConfig,
    context: ExtensionContext | None = None,
) -> list[dict[str, Any]]:
    """Evaluate the six frozen methods after all structural gates are open."""

    if context is None:
        context = _prepare_extension_context(seed, preset, device, protocol)
    config = context.config
    calibration_noise = make_controlled_noise(
        n=protocol.calibration_trajectories,
        horizon=config.horizon,
        initial_count=context.environment.initial_count,
        seed=_paper_seed(seed, CALIBRATION_STREAM_SALT),
        device=device,
    )
    reference_noise = make_controlled_noise(
        n=protocol.reference_trajectories,
        horizon=config.horizon,
        initial_count=context.environment.initial_count,
        seed=_paper_seed(seed, REFERENCE_STREAM_SALT),
        device=device,
    )
    adaptation_seed = _adaptation_seeds(seed)
    rows: list[dict[str, Any]] = []
    for gamma in protocol.gammas:
        source_calibration = rollout_controlled(
            context.environment,
            context.logging_policy,
            noise=calibration_noise,
            gamma=gamma,
            action_coordinate=context.action_coordinate,
        )
        calibration_scores = score_batch(
            context.region,
            source_calibration.trajectories.current_states(),
            source_calibration.trajectories.actions,
            source_calibration.trajectories.outcomes,
        )
        grid_scores = calibration_scores[: protocol.grid_trajectories]
        stage_grids = torch.stack(
            [
                fixed_q_grid(
                    grid_scores[:, stage],
                    size=config.q_grid_size,
                    lower_quantile=config.q_quantile_min,
                    upper_quantile=config.q_quantile_max,
                )
                for stage in range(config.horizon)
            ]
        )
        initial_profile = stage_score_profile(
            grid_scores,
            alpha=config.certification.alpha,
        )
        scale_grid = profiled_scale_grid(
            grid_scores,
            initial_profile,
            size=config.q_grid_size,
            lower_quantile=config.q_quantile_min,
            upper_quantile=config.q_quantile_max,
        )
        standard = standard_cp_stagewise_radii(
            calibration_scores,
            config.certification.alpha,
        )
        mfcs, _ = finite_depth_mfcs_selection(
            source_calibration.trajectories,
            calibration_scores,
            q_grid=scale_grid,
            stage_profile=initial_profile,
            target_policy=context.target_policy,
            logging_policy=context.logging_policy,
            depth=config.baselines.mfcs_depth,
            alpha=config.certification.alpha,
            weight_cap=config.cot.weight_cap,
        )
        scpcp = select_marginal_prefix_schedule(
            source_calibration.trajectories,
            calibration_scores,
            stage_grids=stage_grids,
            target_policy=context.target_policy,
            logging_policy=context.logging_policy,
            outcome_model=context.outcome_model,
            outcome_sd=context.outcome_sd,
            target=1.0 - config.certification.alpha,
        )
        online_environment = ControlledOnlineEnvironment(
            transition=context.environment,
            gamma=gamma,
            action_coordinate=context.action_coordinate,
        )
        aci = aci_style_controller(
            online_environment,
            context.target_policy,
            context.region,
            calibration_scores,
            alpha=config.certification.alpha,
            gamma=config.baselines.aci_gamma,
            rounds=config.baselines.online_rounds,
            total_rollouts=protocol.online_trajectories,
            horizon=config.horizon,
            seed=adaptation_seed["ACI"],
            device=device,
            rollout_fn=_controlled_online_rollout,
        )
        spci = multidim_spci_style_controller(
            online_environment,
            context.target_policy,
            context.region,
            calibration_scores,
            alpha=config.certification.alpha,
            rounds=config.baselines.online_rounds,
            total_rollouts=protocol.online_trajectories,
            horizon=config.horizon,
            seed=adaptation_seed["SPCI"],
            device=device,
            residual_window=config.baselines.multidim_buffer,
            rollout_fn=_controlled_online_rollout,
        )
        initial_prc_scale = float((standard / initial_profile.to(standard)).max().item())
        prc = prc_profile_scale(
            online_environment,
            context.target_policy,
            context.region,
            initial_prc_scale,
            scale_grid,
            initial_profile,
            alpha=config.certification.alpha,
            delta=config.certification.delta,
            rounds=config.baselines.online_rounds,
            total_rollouts=protocol.online_trajectories,
            horizon=config.horizon,
            seed=adaptation_seed["PRC"],
            device=device,
            maximum_step=config.baselines.prc_maximum_step,
            rollout_fn=_controlled_online_rollout,
        )
        for name, adaptation in (("ACI", aci), ("SPCI", spci), ("PRC", prc)):
            if adaptation.target_deployments != TARGET_ADAPTATION_BUDGET[name]:
                raise RuntimeError(f"{name} did not consume its exact target-data budget")

        source_reference = rollout_controlled(
            context.environment,
            context.logging_policy,
            noise=reference_noise,
            gamma=gamma,
            action_coordinate=context.action_coordinate,
        )
        source_scores = score_batch(
            context.region,
            source_reference.trajectories.current_states(),
            source_reference.trajectories.actions,
            source_reference.trajectories.outcomes,
        )
        schedules: dict[str, Tensor | None] = {
            "Standard CP": standard,
            "ACI": aci.radius_by_time.to(device),
            "MFCS": None if mfcs.radius is None else mfcs.radius * initial_profile.to(calibration_scores),
            "SPCI": spci.radius_by_time.to(device),
            "PRC": prc.radius_by_time.to(device),
            "SC-PCP": scpcp.radii,
        }
        adaptations = {"ACI": aci, "SPCI": spci, "PRC": prc}
        method_rows = {
            method: _evaluate_method(
                method,
                schedules[method],
                source_reference=source_reference,
                source_scores=source_scores,
                environment=context.environment,
                target_policy=context.target_policy,
                logging_policy=context.logging_policy,
                reference_noise=reference_noise,
                gamma=gamma,
                action_coordinate=context.action_coordinate,
                outcome_model=context.outcome_model,
                outcome_sd=context.outcome_sd,
                adaptation=adaptations.get(method),
                selection_status=(
                    mfcs.status
                    if method == "MFCS"
                    else (
                        "SELECTED_MARGINAL_POINT"
                        if method == "SC-PCP" and scpcp.radii is not None
                        else (
                            "UNAVAILABLE_NO_FEASIBLE_CANDIDATE"
                            if method == "SC-PCP"
                            else "AVAILABLE"
                        )
                    )
                ),
            )
            for method in METHODS
        }
        rows.append(
            {
                "seed": seed,
                "dataset": preset.name,
                "gamma": gamma,
                "q_low": context.q_low,
                "q_high": context.q_high,
                "adaptation_seeds": adaptation_seed,
                "scpcp_minimum_ess_fraction": _minimum_fraction(
                    scpcp.effective_sample_size,
                    protocol.calibration_trajectories,
                ),
                "scpcp_minimum_candidate_ess_fraction": _minimum_fraction(
                    scpcp.candidate_effective_sample_size,
                    protocol.calibration_trajectories,
                ),
                "scpcp_selected_endpoint": scpcp.selected_endpoint,
                "scpcp_failure_stage": scpcp.failure_stage,
                "methods": method_rows,
            }
        )
    return rows


def _trajectory_placeholder(states: Tensor, actions: Tensor, patient_ids: Tensor) -> object:
    """Build a shape-valid action-only batch without consulting clinical outcomes."""

    from scpcp.data import TrajectoryBatch

    outcome = torch.zeros(
        (*actions.shape, 1),
        dtype=states.dtype,
        device=states.device,
    )
    return TrajectoryBatch(states, actions, outcome, patient_ids)


def _split_audit(splits: ClinicalExtensionSplits) -> dict[str, Any]:
    identifiers = {
        "predictor": sorted(set(int(value) for value in splits.predictor.patient_ids.cpu().tolist())),
        "fidelity": sorted(set(int(value) for value in splits.fidelity.patient_ids.cpu().tolist())),
        "environment": sorted(set(int(value) for value in splits.environment.patient_ids.cpu().tolist())),
    }
    sets = {name: set(values) for name, values in identifiers.items()}
    disjoint = (
        sets["predictor"].isdisjoint(sets["fidelity"])
        and sets["predictor"].isdisjoint(sets["environment"])
        and sets["fidelity"].isdisjoint(sets["environment"])
    )
    if not disjoint:
        raise RuntimeError("clinical role allocation is not patient-disjoint")
    return {
        "role_unique_patient_counts": {name: len(values) for name, values in identifiers.items()},
        "role_episode_counts": {
            "predictor": splits.predictor.n,
            "fidelity": splits.fidelity.n,
            "environment": splits.environment.n,
        },
        "role_patient_id_sha256": {
            name: _integer_set_sha256(values) for name, values in identifiers.items()
        },
        "patient_sets_pairwise_disjoint": True,
        "split_fractions": list(splits.split_fractions),
    }


def _context_identity(context: ExtensionContext) -> dict[str, Any]:
    split = _split_audit(context.splits)
    identity = {
        "outcome_model_state_sha256": _module_state_sha256(context.outcome_model),
        "behavior_policy_state_sha256": _module_state_sha256(context.logging_policy),
        "q_low": context.q_low,
        "q_high": context.q_high,
        "n_actions": context.n_actions,
        "action_mapping": {
            str(key): value for key, value in sorted(context.action_mapping.items())
        },
        "action_costs": list(context.action_costs),
        "donor_neighbors": context.environment.neighbors,
        "donor_bandwidth": context.environment.bandwidth,
        "transition_ridge": context.environment.ridge,
        "environment_patient_id_sha256": split["role_patient_id_sha256"]["environment"],
        "split_patient_id_sha256": split["role_patient_id_sha256"],
        "split_fractions": split["split_fractions"],
        "active_config_sha256": _json_sha256(context.config.to_dict()),
    }
    return {**identity, "combined_sha256": _json_sha256(identity)}


def _module_state_sha256(module: object) -> str:
    digest = hashlib.sha256()
    state = module.state_dict()  # type: ignore[union-attr]
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        shape_bytes = json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(dtype_bytes).to_bytes(2, "big"))
        digest.update(dtype_bytes)
        digest.update(len(shape_bytes).to_bytes(4, "big"))
        digest.update(shape_bytes)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _episode_support_summary(patient_ids: Tensor) -> dict[str, Any]:
    _, counts = torch.unique(patient_ids.cpu(), return_counts=True)
    row_count = int(len(patient_ids))
    unique_count = int(len(counts))
    return {
        "episode_row_count": row_count,
        "unique_patient_count": unique_count,
        "maximum_episodes_per_patient": int(counts.max().item()),
        "duplicate_episode_rate": 1.0 - unique_count / row_count,
        "donor_contract": (
            "episode-weighted transition; patient-aggregated overlap diagnostics"
        ),
    }


@torch.no_grad()
def _representation(model: object, states: Tensor) -> Tensor:
    device = next(model.parameters()).device  # type: ignore[union-attr]
    parts = [model.representation(part.to(device)).cpu() for part in states.split(4_096)]  # type: ignore[union-attr]
    return torch.cat(parts, dim=0)


def _raw_transition_invariants(
    context: ExtensionContext,
    *,
    state: Tensor,
    next_state: Tensor,
    outcome: Tensor,
    ess: Tensor,
    probability_max: Tensor,
    stage: int,
) -> dict[str, Any]:
    history = context.config.model.history_length
    base_dim = state.shape[1] // history
    current_sequence = state.reshape(len(state), history, base_dim)
    next_sequence = next_state.reshape(len(state), history, base_dim)
    rolling_history_exact = bool(torch.equal(next_sequence[:, :-1], current_sequence[:, 1:]))
    static_base = tuple(sorted({index % base_dim for index in context.static_indices}))
    static_exact = (
        True
        if not static_base
        else bool(
            torch.equal(
                next_sequence[:, -1, static_base],
                current_sequence[:, -1, static_base],
            )
        )
    )
    cumulative = tuple(
        index
        for index, name in enumerate(context.state_feature_names)
        if name.startswith("cumulative_")
    )
    cumulative_monotone = (
        True
        if not cumulative
        else bool(
            (
                next_sequence[:, -1, cumulative]
                >= current_sequence[:, -1, cumulative] - 1e-6
            ).all()
        )
    )
    decision_time_exact = True
    if "decision_time" in context.state_feature_names:
        time_index = context.state_feature_names.index("decision_time")
        expected = (stage + 1) / context.config.horizon
        decision_time_exact = bool(
            torch.allclose(
                next_sequence[:, -1, time_index],
                torch.full_like(next_sequence[:, -1, time_index], expected),
                atol=1e-6,
                rtol=0.0,
            )
        )
    finite = bool(
        torch.isfinite(next_state).all()
        and torch.isfinite(outcome).all()
        and torch.isfinite(ess).all()
        and torch.isfinite(probability_max).all()
    )
    ess_valid = bool(ess.ge(1.0 - 1e-6).all())
    probability_valid = bool(
        probability_max.ge(0.0).logical_and(probability_max.le(1.0 + 1e-6)).all()
    )
    passed = all(
        (
            rolling_history_exact,
            static_exact,
            cumulative_monotone,
            decision_time_exact,
            finite,
            ess_valid,
            probability_valid,
        )
    )
    return {
        "passed": passed,
        "rolling_history_exact": rolling_history_exact,
        "static_coordinates_exact": static_exact,
        "cumulative_coordinates_monotone": cumulative_monotone,
        "decision_time_exact": decision_time_exact,
        "finite": finite,
        "row_kernel_ess_at_least_one": ess_valid,
        "row_kernel_probability_in_unit_interval": probability_valid,
    }


def _merge_invariant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one transition-invariant chunk is required")
    keys = tuple(rows[0])
    required_keys = set(keys)
    if any(set(row) != required_keys for row in rows):
        raise ValueError("transition-invariant chunks have different schemas")
    return {key: all(bool(row[key]) for row in rows) for key in keys}


def _minimum_fraction(values: Tensor, denominator: int) -> float | None:
    if values.numel() == 0:
        return None
    return float(values.min().item() / denominator)


def _dataset_contract(
    protocol: ControlledClinicalExtensionConfig,
    preset: DatasetPreset,
) -> dict[str, Any]:
    base_path = ROOT / preset.base_config
    base_bytes = base_path.read_bytes()
    active_config = _controlled_config(preset, protocol).to_dict()
    cache_path = _raw_cache_path(_controlled_config(preset, protocol))
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"formal v2 provenance requires the frozen raw clinical cache: {cache_path}"
        )
    contract = {
        "dataset": preset.name,
        "base_config": preset.base_config.as_posix(),
        "base_config_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "base_config_size_bytes": len(base_bytes),
        "active_config": active_config,
        "active_config_sha256": _json_sha256(active_config),
        "controlled_override": {
            "policy.policy_ratio_cap": protocol.policy_ratio_cap,
        },
        "horizon": preset.horizon,
        "late_stages_zero_based": list(preset.late_stages),
        "raw_clinical_cache": {
            "path": cache_path.relative_to(ROOT).as_posix(),
            "size_bytes": cache_path.stat().st_size,
            "sha256": _file_sha256(cache_path),
            "schema": "per_step_v17",
        },
    }
    if preset.name == "mimic_cxr":
        contract["mimic_cxr_sources"] = _cxr_source_contract(
            cache_path,
            data_root=Path(active_config["data"]["data_root"]),
        )
    return contract


def _cxr_source_contract(cache_path: Path, *, data_root: Path) -> dict[str, Any]:
    """Bind every ordered index image and the exact pretrained checkpoint."""

    from PIL import __version__ as pillow_version
    import torch.hub
    import torchvision
    from torchvision.models import DenseNet121_Weights

    stored = torch.load(cache_path, map_location="cpu", weights_only=False)
    image_paths = tuple(str(path) for path in stored.get("cxr_paths", ()))
    if not image_paths:
        raise RuntimeError("MIMIC-CXR cache does not contain ordered index image paths")
    entries = []
    total_bytes = 0
    for index, raw_path in enumerate(image_paths):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing ordered MIMIC-CXR image: {path}")
        size = path.stat().st_size
        total_bytes += size
        try:
            stored_path = path.resolve().relative_to(data_root.resolve()).as_posix()
        except ValueError:
            stored_path = str(path.resolve())
        entries.append(
            {
                "index": index,
                "path": stored_path,
                "bytes": size,
                "sha256": _file_sha256(path),
            }
        )
    manifest_hash = _json_sha256(entries)
    weights = DenseNet121_Weights.IMAGENET1K_V1
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "formal MIMIC-CXR provenance requires the cached DenseNet-121 checkpoint: "
            f"{checkpoint_path}"
        )
    return {
        "ordered_image_count": len(entries),
        "ordered_image_total_bytes": total_bytes,
        "ordered_image_manifest": entries,
        "ordered_image_manifest_sha256": manifest_hash,
        "pretrained_weight_enum": "DenseNet121_Weights.IMAGENET1K_V1",
        "pretrained_weight_url": weights.url,
        "pretrained_checkpoint_path": str(checkpoint_path.resolve()),
        "pretrained_checkpoint_bytes": checkpoint_path.stat().st_size,
        "pretrained_checkpoint_sha256": _file_sha256(checkpoint_path),
        "torchvision_version": str(torchvision.__version__),
        "pillow_version": str(pillow_version),
    }


def _build_source_snapshot() -> dict[str, Any]:
    """Build a deterministic, content-addressed snapshot of executable inputs."""

    paths = [
        *sorted((ROOT / "src" / "scpcp").rglob("*.py")),
        *sorted((ROOT / "scripts").rglob("*.py")),
        *sorted((ROOT / "tools").rglob("*.py")),
        *sorted((ROOT / "configs").rglob("*.yaml")),
        ROOT / "pyproject.toml",
    ]
    relative_paths = [path.relative_to(ROOT).as_posix() for path in paths]
    if len(relative_paths) != len(set(relative_paths)) or any(not path.is_file() for path in paths):
        raise RuntimeError("source snapshot file set is invalid")
    files = []
    archive_stream = io.BytesIO()
    with tarfile.open(fileobj=archive_stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path, relative in zip(paths, relative_paths):
            content = path.read_bytes()
            files.append(
                {
                    "path": relative,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
            info = tarfile.TarInfo(relative)
            info.size = len(content)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    archive_bytes = archive_stream.getvalue()
    archive_hash = hashlib.sha256(archive_bytes).hexdigest()
    manifest_payload = {
        "protocol": PROTOCOL,
        "format": "deterministic_uncompressed_pax_tar",
        "file_count": len(files),
        "files": files,
    }
    manifest_bytes = (
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    archive_relative = f"provenance/source_snapshot_{archive_hash}.tar"
    manifest_relative = f"provenance/source_manifest_{manifest_hash}.json"
    return {
        "archive_bytes": archive_bytes,
        "manifest_bytes": manifest_bytes,
        "contract": {
            "archive_path": archive_relative,
            "archive_sha256": archive_hash,
            "archive_bytes": len(archive_bytes),
            "manifest_path": manifest_relative,
            "manifest_sha256": manifest_hash,
            "manifest_bytes": len(manifest_bytes),
            "file_count": len(files),
            "relationship": (
                "content-addressed recovery snapshot of the active dirty/untracked "
                "source and configuration tree used by this extension"
            ),
        },
    }


def _publish_source_snapshot(root: Path, snapshot: dict[str, Any]) -> None:
    contract = snapshot["contract"]
    archive_path = root / contract["archive_path"]
    manifest_path = root / contract["manifest_path"]
    _atomic_write(archive_path, snapshot["archive_bytes"])
    _atomic_write(manifest_path, snapshot["manifest_bytes"])
    _verify_source_snapshot(root, contract)


def _verify_source_snapshot(root: Path, contract: dict[str, Any]) -> None:
    archive_path = root / contract["archive_path"]
    manifest_path = root / contract["manifest_path"]
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != contract["archive_bytes"]
        or _file_sha256(archive_path) != contract["archive_sha256"]
        or not manifest_path.is_file()
        or manifest_path.stat().st_size != contract["manifest_bytes"]
        or _file_sha256(manifest_path) != contract["manifest_sha256"]
    ):
        raise RuntimeError("source/config recovery snapshot does not match metadata")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("format") != "deterministic_uncompressed_pax_tar"
        or manifest.get("file_count") != contract["file_count"]
        or len(manifest.get("files", ())) != contract["file_count"]
    ):
        raise RuntimeError("source/config recovery manifest is malformed")


def _verified_precoverage_retry_amendment() -> dict[str, Any]:
    """Bind the one documented precoverage CUDA failure to every retry artifact."""

    archive_path = ROOT / PRECOVERAGE_FAILURE_ARCHIVE
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"precoverage failure archive is required before launch: {archive_path}"
        )
    if (
        archive_path.stat().st_size != PRECOVERAGE_FAILURE_ARCHIVE_BYTES
        or _file_sha256(archive_path) != PRECOVERAGE_FAILURE_ARCHIVE_SHA256
    ):
        raise RuntimeError("precoverage failure archive differs from the frozen retry contract")
    return {
        "amendment_id": "precoverage_cuda_indexed_median_retry_20260826",
        "classification": "precoverage_engineering_retry",
        "failure_archive": {
            "path": PRECOVERAGE_FAILURE_ARCHIVE.as_posix(),
            "bytes": PRECOVERAGE_FAILURE_ARCHIVE_BYTES,
            "sha256": PRECOVERAGE_FAILURE_ARCHIVE_SHA256,
        },
        "failed_source_tree_sha256": FAILED_SOURCE_TREE_SHA256,
        "failed_source_archive_sha256": FAILED_SOURCE_ARCHIVE_SHA256,
        "failure": {
            "dataset": "mimic_iv",
            "phase": "k0_fidelity",
            "cause": (
                "torch.median(dim=1) selected CUDA's indexed-median kernel while "
                "deterministic algorithms were required"
            ),
            "coverage_opened": False,
            "scientific_rows_opened": False,
        },
        "durable_failed_attempt_inventory": {
            "support_seed_artifacts": 20,
            "k0_fidelity_seed_artifacts": 0,
            "donor_overlap_seed_artifacts": 0,
            "science_seed_artifacts": 0,
            "root_complete_marker": False,
            "root_final_status": False,
        },
        "retry_execution": {
            "same_prespecified_seed_banks_reused": True,
            "failed_attempt_support_artifacts_reused": False,
            "support_recomputed_from_scratch": True,
            "failed_attempt_phase_artifacts_reused": False,
            "seed_interpretation": (
                "prespecified seeds reused after a documented precoverage "
                "implementation failure; not described as previously unused"
            ),
        },
        "implementation_fix": {
            "old_operation": "torch.median(sorted_topk_distances, dim=1).values",
            "new_operation": "sorted_topk_distances[:, (k-1)//2]",
            "semantic_equivalence": (
                "exact lower median for both odd and even k, matching PyTorch's "
                "median(dim=1) value semantics"
            ),
            "scientific_contract_changed": False,
            "dgp_gate_model_seed_or_threshold_changed": False,
        },
    }


def _verified_postcompute_retry_amendment() -> dict[str, Any]:
    """Bind the uninspected postcompute validator failure to the fresh rerun."""

    archive_path = ROOT / POSTCOMPUTE_FAILURE_ARCHIVE
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"postcompute failure archive is required before launch: {archive_path}"
        )
    if (
        archive_path.stat().st_size != POSTCOMPUTE_FAILURE_ARCHIVE_BYTES
        or _file_sha256(archive_path) != POSTCOMPUTE_FAILURE_ARCHIVE_SHA256
    ):
        raise RuntimeError(
            "postcompute failure archive differs from the frozen retry contract"
        )
    return {
        "amendment_id": "postcompute_preinspection_json_key_order_retry_20260826",
        "classification": "postcompute_preinspection_validator_only_retry",
        "failure_archive": {
            "path": POSTCOMPUTE_FAILURE_ARCHIVE.as_posix(),
            "bytes": POSTCOMPUTE_FAILURE_ARCHIVE_BYTES,
            "sha256": POSTCOMPUTE_FAILURE_ARCHIVE_SHA256,
        },
        "failed_source_tree_sha256": POSTCOMPUTE_FAILED_SOURCE_TREE_SHA256,
        "failed_source_archive_sha256": POSTCOMPUTE_FAILED_SOURCE_ARCHIVE_SHA256,
        "failure": {
            "dataset": "mimic_iv",
            "phase": "final_dataset_bundle_validation",
            "cause": (
                "canonical JSON serialization used sort_keys=True while two reload "
                "validators incorrectly treated object-key order as semantic"
            ),
            "coverage_rows_generated": True,
            "scientific_rows_generated": True,
            "coverage_or_science_values_inspected_or_used": False,
            "result_guided_change": False,
        },
        "durable_failed_attempt_inventory": {
            "support_seed_artifacts": 20,
            "k0_fidelity_seed_artifacts": 20,
            "donor_overlap_seed_artifacts": 20,
            "science_seed_artifacts": 20,
            "science_summary": True,
            "bootstrap_uniform_matrix": True,
            "bootstrap_index_matrix": True,
            "dataset_final_status": True,
            "dataset_complete_marker": True,
            "dataset_manifest": True,
            "root_complete_marker": False,
            "other_dataset_directories": [],
        },
        "retry_execution": {
            "same_prespecified_seed_banks_reused": True,
            "all_phases_recomputed_from_scratch": True,
            "failed_attempt_phase_artifacts_reused": False,
            "old_dataset_bundle_reused": False,
            "seed_interpretation": (
                "prespecified seeds rerun after a documented postcompute, preinspection "
                "validator failure; no result value informed the validator-only fix"
            ),
        },
        "validator_fix": {
            "json_object_order_is_semantically_irrelevant": True,
            "k0_invariant_schema": (
                "replace tuple(dict)==ordered_keys with equality of the exact key set"
            ),
            "overlap_probe_schema": (
                "replace tuple(dict)==('q_mid','q_high') with equality of the exact "
                "{'q_mid','q_high'} key set"
            ),
            "value_type_and_cross_field_checks_preserved": True,
            "scientific_contract_changed": False,
            "dgp_gate_model_method_seed_threshold_or_metric_changed": False,
        },
    }


def _validate_retry_amendment_binding(
    metadata: dict[str, Any],
    retry_amendment: dict[str, Any],
) -> None:
    expected_sha256 = _json_sha256(retry_amendment)
    if (
        metadata.get("precoverage_engineering_retry_amendment") != retry_amendment
        or metadata.get("precoverage_engineering_retry_amendment_sha256")
        != expected_sha256
    ):
        raise RuntimeError("metadata does not bind the frozen precoverage retry amendment")


def _validate_postcompute_retry_amendment_binding(
    metadata: dict[str, Any],
    retry_amendment: dict[str, Any],
) -> None:
    expected_sha256 = _json_sha256(retry_amendment)
    if (
        metadata.get("postcompute_preinspection_retry_amendment") != retry_amendment
        or metadata.get("postcompute_preinspection_retry_amendment_sha256")
        != expected_sha256
    ):
        raise RuntimeError(
            "metadata does not bind the frozen postcompute retry amendment"
        )


def _root_complete_marker(
    source_snapshot: dict[str, Any],
    retry_amendment: dict[str, Any],
    postcompute_retry_amendment: dict[str, Any],
) -> str:
    return (
        f"complete source_snapshot_sha256={source_snapshot['archive_sha256']} "
        "precoverage_retry_amendment_sha256="
        f"{_json_sha256(retry_amendment)} "
        "precoverage_failure_archive_sha256="
        f"{retry_amendment['failure_archive']['sha256']} "
        "postcompute_retry_amendment_sha256="
        f"{_json_sha256(postcompute_retry_amendment)} "
        "postcompute_failure_archive_sha256="
        f"{postcompute_retry_amendment['failure_archive']['sha256']}\n"
    )


def _raw_cache_path(config: ExperimentConfig) -> Path:
    limit = config.data.max_patients or 60_000
    path = Path(config.data.cache_dir)
    if not path.is_absolute():
        path = ROOT / path
    return path / (
        f"per_step_v17_{config.data.dataset}_h{config.horizon}_"
        f"n{limit}_c{config.data.cohort_seed}.pt"
    )


def _root_metadata(
    protocol: ControlledClinicalExtensionConfig,
    *,
    datasets: tuple[str, ...],
    devices: tuple[str, ...],
    source_hash: str,
    contracts: dict[str, dict[str, Any]],
    seed_mapping: dict[str, str],
    rng_audit: dict[str, Any],
    source_snapshot: dict[str, Any],
    retry_amendment: dict[str, Any],
    postcompute_retry_amendment: dict[str, Any],
) -> dict[str, Any]:
    config_bytes = CONFIG_PATH.read_bytes()
    return {
        "protocol": PROTOCOL,
        "role": "fresh_dataset_native_controlled_clinical_extension",
        "canonical_scpcp_mutation_permitted": False,
        "existing_mimic_v1_substitution_permitted": False,
        "datasets": list(datasets),
        "devices": list(devices),
        "working_directory": str(Path.cwd().resolve()),
        "seed_to_device": seed_mapping,
        "source_tree_sha256": source_hash,
        "source_snapshot": source_snapshot,
        "precoverage_engineering_retry_amendment": retry_amendment,
        "precoverage_engineering_retry_amendment_sha256": _json_sha256(
            retry_amendment
        ),
        "postcompute_preinspection_retry_amendment": postcompute_retry_amendment,
        "postcompute_preinspection_retry_amendment_sha256": _json_sha256(
            postcompute_retry_amendment
        ),
        "extension_config": {
            "path": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "size_bytes": len(config_bytes),
        },
        "common_scientific_contract": COMMON_SCIENTIFIC_CONTRACT,
        "base_config_sha256_by_dataset": {
            dataset: contracts[dataset]["base_config_sha256"] for dataset in datasets
        },
        "dataset_contracts": contracts,
        "split_contract": {
            "roles": ["D_pred", "D_fidelity", "D_env"],
            "fractions": list(protocol.split_fractions),
            "unit": "unique patient",
            "D_pred": "outcome model and behavior policy only",
            "D_fidelity": "q80/q95 and logging-mixture one-step K0 fidelity only",
            "D_env": "episode-weighted donor transition environment",
            "patient_sets_pairwise_disjoint": True,
        },
        "donor_contract": {
            "transition_sampling": "episode-weighted",
            "overlap_diagnostics": "probability aggregated by unique donor patient",
            "local_k": "unique donor patients among the selected neighbor rows",
            "no_pooling_or_action_coarsening_beyond_dataset_ontology": True,
            "kernel_formula": (
                "h_t is the same-stage D_env realized conformity-score empirical rank in "
                "[0,1]; "
                "r(a)=2*(cost(a)-min_cost)/(max_cost-min_cost)-1; local donor logit "
                "equals the base kNN logit plus gamma*r(a)*h_t"
            ),
            "source_target_kernel": "the same K_gamma is used for source and target rollouts",
            "neighbors": protocol.support_gate.neighbors,
            "bandwidth": {
                dataset: contracts[dataset]["active_config"]["data"]["empirical_bandwidth"]
                for dataset in datasets
            },
            "ridge": protocol.transition_ridge,
            "stress_interpretation": (
                "h_t is a calibration-aligned controlled stress coordinate, not future "
                "clinical acuity, not natural performativity, and not a causal treatment effect"
            ),
            "action_costs_and_mappings": "persisted per seed in support and context identities",
        },
        "science": {
            "gammas": list(protocol.gammas),
            "methods": list(METHODS),
            "information_regime": INFORMATION_REGIME,
            "target_adaptation_trajectories": TARGET_ADAPTATION_BUDGET,
            "calibration_trajectories": protocol.calibration_trajectories,
            "grid_trajectories": protocol.grid_trajectories,
            "reference_trajectories": protocol.reference_trajectories,
            "online_trajectories": protocol.online_trajectories,
            "bootstrap_resamples": protocol.bootstrap_resamples,
            "confirmatory_endpoint_gamma": -4.0,
            "other_gamma_role": "full signed control curves, descriptive only",
            "primary_metric": "min_t mean_seed(target_coverage_seed_t)",
            "guarantee_scope": "asymptotic per-step marginal coverage",
        },
        "gate_consequences": {
            "SUPPORT_NO_GO": "no scientific score or coverage rows",
            "STRUCTURAL_NO_GO": (
                "IMPLEMENTATION_INVALID: any exact K0 invariant failure stops all science"
            ),
            "K0_FIDELITY_NO_GO": "no scientific score or coverage rows",
            "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY": (
                "save/render curves; exclude confirmatory ranking, attainment, "
                "superiority, and cross-dataset conjunction"
            ),
            "EMPIRICAL_OVERLAP_SCREEN_PASSED": (
                "both gamma=-4 q_mid and q_high=max-response probes pass; empirical "
                "interpretation screen only, not a support or coverage guarantee"
            ),
        },
        "overlap_screen_amendment": {
            "timing": "frozen before any coverage launch",
            "radii": ["q_mid", "q_high=max_response"],
            "gamma": -4.0,
            "probe_trajectories": protocol.donor_overlap_gate.probe_trajectories,
            "common_random_numbers": "same frozen noise bundle for both radii",
            "rationale": "screen maximal policy response in addition to the midpoint",
            "rng_map_changed": False,
        },
        "mimic_iv_seed_bank_amendment": {
            "timing": "before any coverage or scientific launch",
            "original_formal_seeds": list(range(93_200, 93_400, 10)),
            "revised_formal_seeds": list(range(93_600, 93_800, 10)),
            "original_bootstrap_seed": 9_321_019,
            "revised_bootstrap_seed": 9_361_019,
            "reason": (
                "pure RNG/provenance collision: MIMIC-CXR encoder seed+701 overlapped "
                "the original MIMIC-IV outcome-model seed+1 stream"
            ),
            "scientific_rows_opened_before_amendment": False,
            "outcome_blind_support_reaudit": {
                "available_seeds": 20,
                "prespecified_seeds": 20,
                "seedwise_minimum_cell_count_range": [88, 115],
                "n_actions": 6,
            },
        },
        "rng_audit": rng_audit,
        "environment_versions": _environment_versions(),
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "nuisance_stream_reset": (
                "torch CPU/current-CUDA reset to seed+1 immediately before outcome fit "
                "and seed+2 immediately before behavior fit"
            ),
        },
    }


def _stable_device_mapping(
    protocol: ControlledClinicalExtensionConfig,
    datasets: tuple[str, ...],
    devices: tuple[str, ...],
) -> dict[str, str]:
    mapping = {}
    for dataset in datasets:
        dataset_offset = DATASET_NAMES.index(dataset) * 20
        for index, seed in enumerate(protocol.datasets[dataset].seeds):
            mapping[f"{dataset}/{seed}"] = devices[(dataset_offset + index) % len(devices)]
    return mapping


def _new_rng_stream_mapping(
    protocol: ControlledClinicalExtensionConfig,
    datasets: tuple[str, ...],
) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for dataset in datasets:
        preset = protocol.datasets[dataset]
        mapping[f"{dataset}/summary_bootstrap"] = preset.bootstrap_seed
        for seed in preset.seeds:
            prefix = f"{dataset}/base_{seed}"
            mapping[f"{prefix}/task"] = seed
            mapping[f"{prefix}/outcome_model"] = seed + 1
            mapping[f"{prefix}/behavior_model"] = seed + 2
            if dataset == "mimic_cxr":
                mapping[f"{prefix}/cxr_encoder"] = seed + 701
            mapping[f"{prefix}/k0_base_uniform"] = K0_UNIFORM_SEED_OFFSET + seed
            mapping[f"{prefix}/donor_overlap_probe"] = _paper_seed(seed, OVERLAP_STREAM_SALT)
            mapping[f"{prefix}/calibration"] = _paper_seed(seed, CALIBRATION_STREAM_SALT)
            mapping[f"{prefix}/reference"] = _paper_seed(seed, REFERENCE_STREAM_SALT)
            adaptation_root = _paper_seed(seed, 700_001)
            for round_index in range(3):
                mapping[f"{prefix}/ACI_round_{round_index}"] = (
                    _paper_seed(adaptation_root, 101) + 17_923 * round_index
                )
                mapping[f"{prefix}/SPCI_round_{round_index}"] = (
                    _paper_seed(adaptation_root, 211) + 47_021 * round_index
                )
                mapping[f"{prefix}/PRC_round_{round_index}"] = (
                    _paper_seed(adaptation_root, 307) + 61_103 * round_index
                )
    return mapping


def _audit_rng_banks(
    protocol: ControlledClinicalExtensionConfig,
    datasets: tuple[str, ...],
    *,
    output_root: Path,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    artifact_root = ROOT / "results" if artifact_root is None else artifact_root
    source_root = ROOT if source_root is None else source_root
    excluded = {
        Path(__file__).resolve(),
        (ROOT / "src/scpcp/controlled_clinical_extension.py").resolve(),
        CONFIG_PATH.resolve(),
    }
    artifact_ids = _artifact_seeds(artifact_root, excluded_root=output_root)
    artifact_ids |= _all_artifact_rng_ids(artifact_root, excluded_root=output_root)
    source_ids = _source_declared_seeds(source_root, excluded_paths=excluded)
    prior = artifact_ids | source_ids
    mapping = _new_rng_stream_mapping(protocol, datasets)
    _assert_unique_rng_streams(mapping)
    collisions = {name: value for name, value in mapping.items() if value in prior}
    if collisions:
        raise RuntimeError(f"new extension RNG stream collides with prior use: {collisions}")
    return {
        "status": "passed_before_launch",
        "collision_count": 0,
        "collisions": {},
        "artifact_rng_id_count": len(artifact_ids),
        "artifact_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "source_declared_rng_id_count": len(source_ids),
        "source_declared_rng_id_sha256": _integer_set_sha256(source_ids),
        "prior_rng_id_count": len(prior),
        "prior_rng_id_sha256": _integer_set_sha256(prior),
        "new_rng_stream_count": len(mapping),
        "new_rng_stream_mapping": mapping,
        "new_rng_stream_mapping_sha256": _json_sha256(mapping),
        "internal_rng_streams_unique": True,
        "excluded_output": str(output_root),
    }


def _resume_rng_identity(
    stored_metadata: dict[str, Any],
    fresh_audit: dict[str, Any],
) -> dict[str, Any]:
    """Reuse the launch snapshot while rejecting new collisions or identity drift."""

    stored = stored_metadata.get("rng_audit")
    if not isinstance(stored, dict):
        raise RuntimeError("resume metadata lacks the launch-time RNG audit")
    if (
        fresh_audit.get("status") != "passed_before_launch"
        or fresh_audit.get("collision_count") != 0
        or fresh_audit.get("collisions") != {}
        or
        stored.get("status") != "passed_before_launch"
        or stored.get("collision_count") != 0
        or stored.get("collisions") != {}
        or stored.get("new_rng_stream_mapping_sha256")
        != fresh_audit.get("new_rng_stream_mapping_sha256")
        or stored.get("new_rng_stream_mapping")
        != fresh_audit.get("new_rng_stream_mapping")
    ):
        raise RuntimeError("resume RNG identity differs from the launch contract")
    return stored


def _all_artifact_rng_ids(root: Path, *, excluded_root: Path) -> set[int]:
    """Scan ordinary seed JSON too, including nested derived-stream mappings."""

    values: set[int] = set()
    if not root.exists():
        return values
    excluded = excluded_root.resolve()
    for path in root.rglob("*.json"):
        resolved = path.resolve()
        if resolved == excluded or excluded in resolved.parents:
            continue
        match = _SEED_NAME.fullmatch(path.name)
        if match:
            values.add(int(match.group(1)))
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        _collect_artifact_rng_values(payload, values)
    return values


def _collect_artifact_rng_values(value: object, output: set[int], key: str = "") -> None:
    normalized = key.lower()
    if isinstance(value, dict):
        if normalized == "seed_to_device":
            output.update(int(child) for child in value if str(child).isdigit())
        if normalized in {
            "rng_stream_mapping",
            "new_rng_stream_mapping",
            "adaptation_seeds",
            "bootstrap_seeds",
        }:
            _collect_integer_leaves(value, output)
            return
        for child_key, child_value in value.items():
            _collect_artifact_rng_values(child_value, output, str(child_key))
        return
    if isinstance(value, list):
        if normalized == "seeds" or normalized.endswith("_seeds"):
            _collect_integer_leaves(value, output)
            return
        for child in value:
            _collect_artifact_rng_values(child, output, key)
        return
    ignored_suffixes = ("count", "bytes", "size", "sha256")
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and _SEED_ASSIGNMENT.search(key)
        and not normalized.endswith(ignored_suffixes)
    ):
        output.add(value)


def _collect_integer_leaves(value: object, output: set[int]) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _collect_integer_leaves(child, output)
    elif isinstance(value, list):
        for child in value:
            _collect_integer_leaves(child, output)
    elif isinstance(value, int) and not isinstance(value, bool):
        output.add(value)


def _environment_versions() -> dict[str, Any]:
    import pandas
    import scipy

    numpy_configuration = getattr(np.__config__, "CONFIG", {})
    build_dependencies = numpy_configuration.get("Build Dependencies", {})
    gpu_devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu_devices.append(
            {
                "logical_index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [properties.major, properties.minor],
                "uuid": str(getattr(properties, "uuid", "unavailable")),
            }
        )
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "numpy": {
            "version": str(np.__version__),
            "blas": build_dependencies.get("blas"),
            "lapack": build_dependencies.get("lapack"),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "git_version": torch.version.git_version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpu_identity": gpu_devices,
        },
        "scipy": str(scipy.__version__),
        "pandas": str(pandas.__version__),
        "pyyaml": str(yaml.__version__),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _write_bootstrap_artifacts(
    science_root: Path,
    *,
    preset: DatasetPreset,
    resamples: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(preset.bootstrap_seed)
    uniforms = rng.random((resamples, len(preset.seeds)), dtype=np.float64)
    full_indices = np.floor(uniforms * len(preset.seeds)).astype(np.int16)
    uniform_path = science_root / "bootstrap_uniforms.npy"
    index_path = science_root / "bootstrap_indices.npy"
    _write_npy(uniform_path, uniforms)
    _write_npy(index_path, full_indices)
    return {
        "resamples": resamples,
        "root_seed": preset.bootstrap_seed,
        "prespecified_seed_count": len(preset.seeds),
        "uniform_matrix_shape": list(uniforms.shape),
        "uniform_matrix_path": uniform_path.name,
        "uniform_matrix_sha256": _file_sha256(uniform_path),
        "complete_seed_index_matrix_shape": list(full_indices.shape),
        "complete_seed_index_matrix_path": index_path.name,
        "complete_seed_index_matrix_sha256": _file_sha256(index_path),
        "shared_across": ["methods", "gammas", "stages"],
        "selected_subset_rule": (
            "for selected-set size n, use floor(U[:, :n] * n); the complete "
            "10,000x20 matrix is floor(U*20)"
        ),
    }


def summarize_science(
    rows: list[dict[str, Any]],
    *,
    preset: DatasetPreset,
    selected_seeds: tuple[int, ...],
    interpretation_status: str,
    bootstrap_contract: dict[str, Any],
) -> dict[str, Any]:
    if len(rows) != len(selected_seeds) * len(GAMMAS):
        raise RuntimeError("science summary requires one row per eligible seed and gamma")
    if not selected_seeds:
        raise RuntimeError("science summary needs at least one K0-eligible seed")
    rng = np.random.default_rng(preset.bootstrap_seed)
    uniforms = rng.random(
        (BOOTSTRAP_RESAMPLES, len(preset.seeds)),
        dtype=np.float64,
    )
    aggregates = []
    for gamma in GAMMAS:
        confirmatory_endpoint = (
            gamma == -4.0
            and interpretation_status == "EMPIRICAL_OVERLAP_SCREEN_PASSED"
        )
        selected = sorted(
            (row for row in rows if float(row["gamma"]) == gamma),
            key=lambda row: int(row["seed"]),
        )
        if tuple(int(row["seed"]) for row in selected) != selected_seeds:
            raise RuntimeError(f"science seed mismatch for gamma={gamma}")
        method_arrays: dict[str, dict[str, np.ndarray]] = {}
        method_summaries: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            available_mask = np.asarray(
                [bool(row["methods"][method]["selection_available"]) for row in selected],
                dtype=bool,
            )
            coverage = np.zeros((len(selected), preset.horizon), dtype=np.float64)
            source_coverage = np.zeros_like(coverage)
            width = np.zeros_like(coverage)
            for index, row in enumerate(selected):
                if not available_mask[index]:
                    continue
                payload = row["methods"][method]
                coverage[index] = np.asarray(payload["target_coverage"], dtype=np.float64)
                source_coverage[index] = np.asarray(payload["source_coverage"], dtype=np.float64)
                width[index] = np.asarray(payload["target_normalized_width"], dtype=np.float64)
            method_arrays[method] = {
                "available": available_mask,
                "coverage": coverage,
                "width": width.mean(axis=1),
            }
            count = int(available_mask.sum())
            summary: dict[str, Any] = {
                "n_selected": count,
                "n_prespecified": len(preset.seeds),
                "n_k0_eligible": len(selected_seeds),
                "selection_rate": count / len(preset.seeds),
                "selection_rate_ci95": _wilson_interval(count, len(preset.seeds)),
                "target_adaptation_trajectories_per_seed": TARGET_ADAPTATION_BUDGET[method],
            }
            if count == 0:
                summary.update(
                    {
                        "target_marginal_worst_coverage": None,
                        "target_worst_stage_zero_based": None,
                        "target_wsc_ci95": [None, None],
                        "target_coverage_by_stage": [],
                        "target_coverage_by_stage_ci95": [],
                        "target_mean_coverage": None,
                        "target_mean_coverage_ci95": [None, None],
                        "source_marginal_worst_coverage": None,
                        "target_normalized_width_by_stage": [],
                        "target_normalized_width_by_stage_ci95": [],
                        "mean_target_normalized_width": None,
                        "mean_target_normalized_width_ci95": [None, None],
                        "minimum_reference_prefix_ess_fraction": None,
                        "maximum_reference_weight_share": None,
                        # No selected point means attainment is not estimable;
                        # point_eligible remains False because availability failed.
                        "confirmatory_attainment_at_0.90": None,
                        "point_eligibility_rule": (
                            "selection_rate>=0.95 and target_marginal_worst_coverage>=0.90"
                        ),
                        "point_eligible": False if confirmatory_endpoint else None,
                    }
                )
                method_summaries[method] = summary
                continue
            chosen_coverage = coverage[available_mask]
            chosen_source = source_coverage[available_mask]
            chosen_width = width[available_mask]
            bootstrap = _bootstrap_indices(uniforms, count)
            coverage_draws = chosen_coverage[bootstrap].mean(axis=1)
            width_draws = chosen_width[bootstrap].mean(axis=1)
            stage_coverage = chosen_coverage.mean(axis=0)
            stage_width = chosen_width.mean(axis=0)
            wsc_draws = coverage_draws.min(axis=1)
            summary.update(
                {
                    "target_marginal_worst_coverage": float(stage_coverage.min()),
                    "target_worst_stage_zero_based": int(stage_coverage.argmin()),
                    "target_wsc_ci95": _percentile_interval(wsc_draws),
                    "target_coverage_by_stage": stage_coverage.tolist(),
                    "target_coverage_by_stage_ci95": _pointwise_interval(coverage_draws),
                    "target_mean_coverage": float(stage_coverage.mean()),
                    "target_mean_coverage_ci95": _student_t_interval(
                        chosen_coverage.mean(axis=1)
                    ),
                    "source_marginal_worst_coverage": float(
                        chosen_source.mean(axis=0).min()
                    ),
                    "target_normalized_width_by_stage": stage_width.tolist(),
                    "target_normalized_width_by_stage_ci95": _pointwise_interval(width_draws),
                    "mean_target_normalized_width": float(stage_width.mean()),
                    "mean_target_normalized_width_ci95": _student_t_interval(
                        chosen_width.mean(axis=1)
                    ),
                    "minimum_reference_prefix_ess_fraction": float(
                        min(
                            min(row["methods"][method]["prefix_ess_fraction"])
                            for row in selected
                            if row["methods"][method]["selection_available"]
                        )
                    ),
                    "maximum_reference_weight_share": float(
                        max(
                            max(row["methods"][method]["maximum_normalized_weight_share"])
                            for row in selected
                            if row["methods"][method]["selection_available"]
                        )
                    ),
                    "confirmatory_attainment_at_0.90": (
                        bool(stage_coverage.min() >= 0.90)
                        if confirmatory_endpoint
                        else None
                    ),
                    "point_eligibility_rule": (
                        "selection_rate>=0.95 and target_marginal_worst_coverage>=0.90"
                    ),
                    "point_eligible": (
                        bool(
                            count / len(preset.seeds) >= 0.95
                            and stage_coverage.min() >= 0.90
                        )
                        if confirmatory_endpoint
                        else None
                    ),
                }
            )
            method_summaries[method] = summary
        if confirmatory_endpoint:
            paired: object = {
                baseline: _paired_scpcp_comparison(
                    method_arrays["SC-PCP"],
                    method_arrays[baseline],
                    uniforms,
                )
                for baseline in METHODS
                if baseline != "SC-PCP"
            }
            width_order = sorted(
                (
                    {
                        "method": method,
                        "mean_target_normalized_width": method_summaries[method][
                            "mean_target_normalized_width"
                        ],
                    }
                    for method in METHODS
                    if method_summaries[method].get("point_eligible") is True
                ),
                key=lambda item: float(item["mean_target_normalized_width"]),
            )
        else:
            paired = {
                "status": (
                    "EXCLUDED_LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
                    if interpretation_status == "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
                    else "EXCLUDED_NON_CONFIRMATORY_GAMMA_SIGNED_CONTROL"
                )
            }
            width_order = []
        aggregates.append(
            {
                "gamma": gamma,
                "analysis_role": (
                    "confirmatory_gamma_minus_4_endpoint"
                    if confirmatory_endpoint
                    else "descriptive_signed_control_curve"
                ),
                "n_prespecified_seeds": len(preset.seeds),
                "n_k0_eligible_seeds": len(selected_seeds),
                "methods": method_summaries,
                "paired_scpcp_comparisons": paired,
                "width_order_among_point_eligible": width_order,
                "universal_ranking_defined": False,
            }
        )
    return {
        "protocol": PROTOCOL,
        "dataset": preset.name,
        "role": "fresh_dataset_native_controlled_clinical_extension",
        "interpretation_status": interpretation_status,
        "seeds_prespecified": list(preset.seeds),
        "seeds_k0_eligible": list(selected_seeds),
        "methods": list(METHODS),
        "primary_metric": "min_t mean_seed(target_coverage_seed_t)",
        "coverage_conditioning": "successful method selection among K0-eligible seeds",
        "selection_rate_denominator": "all 20 prespecified seeds",
        "bootstrap": bootstrap_contract,
        "aggregates": aggregates,
    }


def _pointwise_interval(draws: np.ndarray) -> list[list[float]]:
    lower = np.quantile(draws, 0.025, axis=0)
    upper = np.quantile(draws, 0.975, axis=0)
    return [[float(lo), float(hi)] for lo, hi in zip(lower, upper)]


def _gate_summary(
    name: str,
    prespecified: tuple[int, ...],
    available: tuple[int, ...],
    passed: bool,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "gate": name,
        "passed": passed,
        "available_seeds": list(available),
        "n_available": len(available),
        "n_prespecified": len(prespecified),
        "availability_fraction": len(available) / len(prespecified),
    }


def _assert_support_context_consistency(
    support_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
) -> None:
    support = {int(row["seed"]): row for row in support_rows}
    for row in context_rows:
        seed = int(row["seed"])
        anchor = support.get(seed)
        if anchor is None:
            raise RuntimeError(f"context seed {seed} lacks a support preflight")
        identity = row["context_identity"]
        if (
            row.get("n_actions") != anchor.get("n_actions")
            or row.get("action_mapping") != anchor.get("action_mapping")
            or identity.get("split_patient_id_sha256")
            != anchor["split_audit"].get("role_patient_id_sha256")
        ):
            raise RuntimeError(f"support/context identity mismatch for seed {seed}")


def _assert_context_consistency(
    anchor_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    anchors = {int(row["seed"]): row["context_identity"] for row in anchor_rows}
    for row in candidate_rows:
        seed = int(row["seed"])
        if row.get("context_identity") != anchors.get(seed):
            raise RuntimeError(f"{label} context identity mismatch for seed {seed}")


def _valid_phase_payload(
    payload: object,
    *,
    phase: str,
    preset: DatasetPreset,
    seed: int,
    device: str,
    seed_contract: dict[str, Any],
) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {
        "protocol": PROTOCOL,
        "dataset": preset.name,
        "phase": phase,
        "seed": seed,
        "device": device,
        "dataset_contract_sha256": seed_contract["dataset_contract_sha256"],
        "source_tree_sha256": seed_contract["source_tree_sha256"],
        "rng_stream_mapping_sha256": seed_contract["rng_stream_mapping_sha256"],
    }
    if any(payload.get(key) != value for key, value in required.items()):
        return False
    result = payload.get("result")
    if not isinstance(result, dict):
        return False
    if (
        result.get("seed") != seed
        or result.get("dataset") != preset.name
        or result.get("phase") != phase
    ):
        return False
    try:
        if phase == "support":
            return _valid_support_result(result, preset)
        if phase == "k0_fidelity":
            return _valid_k0_result(result, preset)
        if phase == "donor_overlap":
            return _valid_overlap_result(result)
        if phase == "science":
            return _valid_science_result(result, preset)
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        return False
    return False


def _valid_support_result(result: dict[str, Any], preset: DatasetPreset) -> bool:
    counts = result.get("unique_patient_counts_by_stage_action")
    n_actions = result.get("n_actions")
    if (
        isinstance(n_actions, bool)
        or not isinstance(n_actions, int)
        or n_actions < 1
        or not isinstance(counts, list)
        or len(counts) != preset.horizon
        or any(not isinstance(row, list) or len(row) != n_actions for row in counts)
    ):
        return False
    gate = load_extension_config(CONFIG_PATH).support_gate
    evaluated = evaluate_support_gate(counts, gate)
    active = result.get("active_action_indices")
    costs = result.get("action_costs")
    mapping = result.get("action_mapping")
    return (
        result.get("outcome_blind") is True
        and result.get("passed") is evaluated.passed
        and result.get("minimum_unique_patients") == evaluated.minimum_unique_patients
        and result.get("failed_cells") == [list(cell) for cell in evaluated.failed_cells]
        and isinstance(active, list)
        and len(active) == n_actions
        and all(isinstance(value, int) and value >= 0 for value in active)
        and isinstance(costs, list)
        and len(costs) == n_actions
        and all(math.isfinite(float(value)) for value in costs)
        and isinstance(mapping, dict)
        and mapping
        and all(
            isinstance(value, int) and 0 <= value < n_actions
            for value in mapping.values()
        )
        and _valid_split_audit(result.get("split_audit"))
        and _valid_episode_summary(result.get("environment_episode_support"))
    )


def _valid_k0_result(result: dict[str, Any], preset: DatasetPreset) -> bool:
    values = result.get("metrics")
    detail = result.get("systematic_replay")
    if not isinstance(values, dict) or not isinstance(detail, dict):
        return False
    metrics = K0FidelityMetrics(**values)
    gate = load_extension_config(CONFIG_PATH).k0_fidelity_gate
    identity = result.get("context_identity")
    split = result.get("split_audit")
    if not _valid_split_audit(split):
        return False
    fidelity_episode_count = int(split["role_episode_counts"]["fidelity"])
    expected_uniforms = _expected_k0_uniform_contract(
        seed=int(result["seed"]),
        horizon=preset.horizon,
        fidelity_episode_count=fidelity_episode_count,
        replay_count=gate.systematic_replays,
    )
    stage_vectors = {
        "score_ks_by_stage": (0.0, 1.0),
        "signed_residual_max_w1_by_stage": (0.0, None),
        "successor_mean_w1_by_stage": (0.0, None),
        "successor_q95_w1_by_stage": (0.0, None),
    }
    resolved_vectors: dict[str, list[float]] = {}
    for name, bounds in stage_vectors.items():
        raw = detail.get(name)
        if not _valid_finite_stage_vector(raw, preset.horizon, *bounds):
            return False
        resolved_vectors[name] = [float(value) for value in raw]
    active_counts = detail.get("active_successor_coordinates_by_stage")
    invariant_rows = detail.get("raw_structural_invariants_by_stage")
    if (
        not isinstance(active_counts, list)
        or len(active_counts) != preset.horizon
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in active_counts)
        or not isinstance(invariant_rows, list)
        or len(invariant_rows) != preset.horizon
        or not all(_valid_k0_invariant_row(row) for row in invariant_rows)
    ):
        return False
    structural = all(bool(row["passed"]) for row in invariant_rows) and all(
        count > 0 for count in active_counts
    )
    headline = {
        "maximum_score_ks": max(resolved_vectors["score_ks_by_stage"]),
        "maximum_signed_residual_w1": max(
            resolved_vectors["signed_residual_max_w1_by_stage"]
        ),
        "maximum_successor_mean_w1": max(
            resolved_vectors["successor_mean_w1_by_stage"]
        ),
        "maximum_successor_q95_w1": max(
            resolved_vectors["successor_q95_w1_by_stage"]
        ),
        "structural_invariants": structural,
    }
    return (
        result.get("gate_name") == "logging-mixture one-step fidelity"
        and detail.get("label") == "logging-mixture one-step fidelity"
        and detail.get("episode_weighted") is True
        and result.get("passed") is k0_fidelity_passes(metrics, gate)
        and values == headline
        and detail.get("systematic_replays") == K0_SYSTEMATIC_REPLAYS
        and detail.get("patient_chunk_size") == K0_PATIENT_CHUNK_SIZE
        and detail.get("base_uniform_seed") == expected_uniforms["base_uniform_seed"]
        and detail.get("base_uniform_shape") == expected_uniforms["base_uniform_shape"]
        and detail.get("base_uniform_sha256") == expected_uniforms["base_uniform_sha256"]
        and detail.get("expansion_formula") == "u[t,i,m]=(U[t,i]+(m+0.5)/16) mod 1"
        and detail.get("flatten_order")
        == "stage, patient, systematic_offset (offset fastest)"
        and detail.get("inference_unit")
        == "patient-disjoint episode query; M=16 quadrature, never 16N independent observations"
        and detail.get("expanded_uniform_sha256")
        == expected_uniforms["expanded_uniform_sha256"]
        and _valid_context_identity(identity)
        and float(result.get("q_low")) == float(identity["q_low"])
        and float(result.get("q_high")) == float(identity["q_high"])
        and result.get("n_actions") == identity["n_actions"]
        and result.get("action_mapping") == identity["action_mapping"]
    )


def _expected_k0_uniform_contract(
    *,
    seed: int,
    horizon: int,
    fidelity_episode_count: int,
    replay_count: int,
) -> dict[str, Any]:
    if horizon < 1 or fidelity_episode_count < 1 or replay_count != K0_SYSTEMATIC_REPLAYS:
        raise ValueError("invalid frozen K0 uniform dimensions")
    uniform_seed = K0_UNIFORM_SEED_OFFSET + seed
    generator = torch.Generator(device="cpu").manual_seed(uniform_seed)
    base = torch.rand(
        (horizon, fidelity_episode_count),
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    offsets = (torch.arange(replay_count, dtype=torch.float64) + 0.5) / replay_count
    expanded = (base[:, :, None] + offsets[None, None, :]).remainder(1.0)
    return {
        "base_uniform_seed": uniform_seed,
        "base_uniform_shape": [horizon, fidelity_episode_count],
        "base_uniform_sha256": hashlib.sha256(
            base.numpy().tobytes(order="C")
        ).hexdigest(),
        "expanded_uniform_sha256": hashlib.sha256(
            expanded.numpy().tobytes(order="C")
        ).hexdigest(),
    }


def _valid_finite_stage_vector(
    value: object,
    horizon: int,
    minimum: float,
    maximum: float | None,
) -> bool:
    return (
        isinstance(value, list)
        and len(value) == horizon
        and all(
            math.isfinite(float(item))
            and float(item) >= minimum
            and (maximum is None or float(item) <= maximum)
            for item in value
        )
    )


def _valid_k0_invariant_row(value: object) -> bool:
    keys = (
        "passed",
        "rolling_history_exact",
        "static_coordinates_exact",
        "cumulative_coordinates_monotone",
        "decision_time_exact",
        "finite",
        "row_kernel_ess_at_least_one",
        "row_kernel_probability_in_unit_interval",
    )
    if not isinstance(value, dict) or set(value) != set(keys):
        return False
    if any(not isinstance(value[key], bool) for key in keys):
        return False
    return value["passed"] is all(value[key] for key in keys[1:])


def _valid_overlap_result(result: dict[str, Any]) -> bool:
    values = result.get("metrics")
    diagnostics = result.get("diagnostics")
    if not isinstance(values, dict) or not isinstance(diagnostics, dict):
        return False
    metrics = DonorOverlapMetrics(**values)
    gate = load_extension_config(CONFIG_PATH).donor_overlap_gate
    identity = result.get("context_identity")
    probes = diagnostics.get("probes")
    if not isinstance(probes, dict) or set(probes) != {"q_mid", "q_high"}:
        return False
    q_low = float(result.get("q_low"))
    q_high = float(result.get("q_high"))
    if (
        not math.isfinite(q_low)
        or not math.isfinite(q_high)
        or not q_high > q_low
        or float(result.get("q_mid")) != q_low + 0.5 * (q_high - q_low)
    ):
        return False
    resolved_probe_metrics: dict[str, DonorOverlapMetrics] = {}
    for label, fraction in zip(("q_mid", "q_high"), gate.probe_radius_fractions, strict=True):
        probe = probes[label]
        expected_radius = q_low + fraction * (q_high - q_low)
        if (
            not isinstance(probe, dict)
            or float(probe.get("radius_fraction")) != fraction
            or float(probe.get("radius")) != expected_radius
        ):
            return False
        probe_metrics = probe.get("metrics")
        if not isinstance(probe_metrics, dict):
            return False
        resolved = DonorOverlapMetrics(**probe_metrics)
        resolved_probe_metrics[label] = resolved
        if (
            probe.get("passed") is not donor_overlap_passes(resolved, gate)
            or not _valid_overlap_probe_diagnostics(probe)
        ):
            return False
    all_passed = all(bool(probe["passed"]) for probe in probes.values())
    expected_worst = {
        "local_ess_p01": min(
            resolved_probe_metrics[label].local_ess_p01 for label in ("q_mid", "q_high")
        ),
        "median_ess_fraction": min(
            resolved_probe_metrics[label].median_ess_fraction
            for label in ("q_mid", "q_high")
        ),
        "maximum_donor_probability": max(
            resolved_probe_metrics[label].maximum_donor_probability
            for label in ("q_mid", "q_high")
        ),
    }
    return (
        values == expected_worst
        and result.get("passed") is all_passed
        and result.get("interpretation_if_failed")
        == "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        and diagnostics.get("patient_aggregated") is True
        and diagnostics.get("probe_trajectories") == gate.probe_trajectories
        and float(diagnostics.get("gamma")) == gate.gamma
        and diagnostics.get("noise_seed") == _paper_seed(result["seed"], OVERLAP_STREAM_SALT)
        and diagnostics.get("independent_frozen_stream") is True
        and diagnostics.get("common_random_numbers_across_radii") is True
        and diagnostics.get("screen_scope")
        == "gamma=-4 q_mid and q_high=max-response; empirical, not a guarantee"
        and diagnostics.get("worst_metrics") == expected_worst
        and diagnostics.get("screen_status")
        == (
            "EMPIRICAL_OVERLAP_SCREEN_PASSED"
            if all_passed
            else "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        )
        and donor_overlap_passes(metrics, gate) is all_passed
        and _valid_episode_summary(diagnostics.get("environment_episode_support"))
        and _valid_split_audit(result.get("split_audit"))
        and _valid_context_identity(identity)
        and float(result.get("q_low")) == float(identity["q_low"])
        and float(result.get("q_high")) == float(identity["q_high"])
        and result.get("n_actions") == identity["n_actions"]
        and result.get("action_mapping") == identity["action_mapping"]
    )


def _valid_overlap_probe_diagnostics(
    probe: dict[str, Any],
) -> bool:
    return (
        math.isfinite(float(probe.get("radius")))
        and float(probe.get("target_simplex_maximum_error")) <= 1e-5
        and float(probe.get("logging_simplex_maximum_error")) <= 1e-5
        and float(probe.get("minimum_logging_probability")) > 0.0
        and float(probe.get("minimum_target_probability")) >= 0.0
        and probe.get("policy_probabilities_finite") is True
        and float(probe.get("maximum_single_step_target_to_logging_ratio"))
        <= float(probe.get("single_step_ratio_cap")) + 1e-5
        and float(probe.get("single_step_ratio_cap"))
        == load_extension_config(CONFIG_PATH).policy_ratio_cap
        and float(probe.get("local_unique_k_minimum")) >= 1.0
        and isinstance(probe.get("prefix_overlap_report_only"), dict)
    )


def _valid_science_result(result: dict[str, Any], preset: DatasetPreset) -> bool:
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != len(GAMMAS):
        return False
    interpretation = result.get("interpretation_status")
    if interpretation not in {
        "EMPIRICAL_OVERLAP_SCREEN_PASSED",
        "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
    }:
        return False
    identity = result.get("context_identity")
    if (
        not _valid_split_audit(result.get("split_audit"))
        or not _valid_context_identity(identity)
        or float(result.get("q_low")) != float(identity["q_low"])
        or float(result.get("q_high")) != float(identity["q_high"])
        or result.get("n_actions") != identity["n_actions"]
        or result.get("action_mapping") != identity["action_mapping"]
    ):
        return False
    for row, gamma in zip(rows, GAMMAS):
        if (
            row.get("seed") != result["seed"]
            or row.get("dataset") != preset.name
            or float(row.get("gamma")) != gamma
            or set(row.get("methods", {})) != set(METHODS)
            or row.get("adaptation_seeds") != _adaptation_seeds(result["seed"])
            or not math.isfinite(float(row.get("q_low")))
            or not math.isfinite(float(row.get("q_high")))
            or not float(row.get("q_high")) > float(row.get("q_low"))
            or float(row.get("q_low")) != float(result["q_low"])
            or float(row.get("q_high")) != float(result["q_high"])
        ):
            return False
        for method in METHODS:
            method_row = row["methods"][method]
            if method_row.get("target_adaptation_trajectories") != TARGET_ADAPTATION_BUDGET[method]:
                return False
            if method_row.get("information_regime") != INFORMATION_REGIME[method]:
                return False
            available = method_row.get("selection_available")
            if not isinstance(available, bool):
                return False
            if not available:
                if method not in {"MFCS", "SC-PCP"} or method_row.get("radii") != []:
                    return False
                continue
            for name in (
                "radii",
                "source_coverage",
                "target_coverage",
                "coverage_gap",
                "target_normalized_width",
                "prefix_ess_fraction",
                "maximum_normalized_weight_share",
            ):
                values = method_row.get(name)
                if (
                    not isinstance(values, list)
                    or len(values) != preset.horizon
                    or not all(math.isfinite(float(value)) for value in values)
                ):
                    return False
            if not all(0.0 <= float(value) <= 1.0 for value in method_row["target_coverage"]):
                return False
            if not all(0.0 <= float(value) <= 1.0 for value in method_row["source_coverage"]):
                return False
            if any(
                not math.isclose(
                    float(gap),
                    float(target) - float(source),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                for gap, target, source in zip(
                    method_row["coverage_gap"],
                    method_row["target_coverage"],
                    method_row["source_coverage"],
                    strict=True,
                )
            ):
                return False
            if not all(float(value) >= 0.0 for value in method_row["target_normalized_width"]):
                return False
            if not all(float(value) >= 0.0 for value in method_row["radii"]):
                return False
            if not all(0.0 < float(value) <= 1.0 for value in method_row["prefix_ess_fraction"]):
                return False
            if not all(
                0.0 < float(value) <= 1.0
                for value in method_row["maximum_normalized_weight_share"]
            ):
                return False
    return True


def _valid_split_audit(value: object) -> bool:
    if not isinstance(value, dict) or value.get("patient_sets_pairwise_disjoint") is not True:
        return False
    hashes = value.get("role_patient_id_sha256")
    counts = value.get("role_unique_patient_counts")
    episodes = value.get("role_episode_counts")
    roles = {"predictor", "fidelity", "environment"}
    return (
        isinstance(hashes, dict)
        and set(hashes) == roles
        and all(_is_sha256(digest) for digest in hashes.values())
        and isinstance(counts, dict)
        and set(counts) == roles
        and all(isinstance(count, int) and count > 0 for count in counts.values())
        and isinstance(episodes, dict)
        and set(episodes) == roles
        and all(isinstance(count, int) and count > 0 for count in episodes.values())
    )


def _valid_context_identity(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    required_hashes = (
        "outcome_model_state_sha256",
        "behavior_policy_state_sha256",
        "environment_patient_id_sha256",
        "active_config_sha256",
        "combined_sha256",
    )
    if not all(_is_sha256(value.get(name)) for name in required_hashes):
        return False
    split_hashes = value.get("split_patient_id_sha256")
    if (
        not isinstance(split_hashes, dict)
        or set(split_hashes) != {"predictor", "fidelity", "environment"}
        or not all(_is_sha256(digest) for digest in split_hashes.values())
        or value.get("environment_patient_id_sha256") != split_hashes["environment"]
    ):
        return False
    identity = {key: child for key, child in value.items() if key != "combined_sha256"}
    q_low = value.get("q_low")
    q_high = value.get("q_high")
    n_actions = value.get("n_actions")
    action_costs = value.get("action_costs")
    action_mapping = value.get("action_mapping")
    return (
        value["combined_sha256"] == _json_sha256(identity)
        and isinstance(q_low, (int, float))
        and isinstance(q_high, (int, float))
        and math.isfinite(float(q_low))
        and math.isfinite(float(q_high))
        and float(q_high) > float(q_low)
        and isinstance(n_actions, int)
        and n_actions >= 2
        and isinstance(action_costs, list)
        and len(action_costs) == n_actions
        and all(math.isfinite(float(cost)) for cost in action_costs)
        and isinstance(action_mapping, dict)
        and action_mapping
        and value.get("donor_neighbors") == 100
        and math.isfinite(float(value.get("donor_bandwidth")))
        and float(value.get("donor_bandwidth")) > 0.0
        and float(value.get("transition_ridge")) == 1e-3
    )


def _valid_episode_summary(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    rows = value.get("episode_row_count")
    unique = value.get("unique_patient_count")
    maximum = value.get("maximum_episodes_per_patient")
    duplicate = value.get("duplicate_episode_rate")
    return (
        isinstance(rows, int)
        and isinstance(unique, int)
        and isinstance(maximum, int)
        and rows >= unique >= 1
        and maximum >= 1
        and isinstance(duplicate, (int, float))
        and math.isfinite(float(duplicate))
        and 0.0 <= float(duplicate) < 1.0
        and abs(float(duplicate) - (1.0 - unique / rows)) <= 1e-12
        and value.get("donor_contract")
        == "episode-weighted transition; patient-aggregated overlap diagnostics"
    )


def _validate_final_dataset_bundle(
    root: Path,
    *,
    protocol: ControlledClinicalExtensionConfig,
    preset: DatasetPreset,
    dataset_metadata: dict[str, Any],
) -> None:
    retry_amendment = _verified_precoverage_retry_amendment()
    _validate_retry_amendment_binding(dataset_metadata, retry_amendment)
    postcompute_retry_amendment = _verified_postcompute_retry_amendment()
    _validate_postcompute_retry_amendment_binding(
        dataset_metadata, postcompute_retry_amendment
    )
    _verify_manifest(root)
    if _json_sha256(_read_json(root / "metadata.json")) != _json_sha256(dataset_metadata):
        raise RuntimeError("final dataset metadata differs from the launch contract")
    final = _read_json(root / "FINAL_STATUS.json")
    gate = _read_json(root / "gate.json")
    root_summary = _read_json(root / "summary.json")
    if (
        final.get("dataset") != preset.name
        or final.get("protocol") != PROTOCOL
        or gate.get("dataset") != preset.name
        or gate.get("protocol") != PROTOCOL
    ):
        raise RuntimeError("final dataset status has wrong identity")
    support_rows = _load_phase_results(
        root / "support",
        phase="support",
        preset=preset,
        seeds=preset.seeds,
        dataset_metadata=dataset_metadata,
    )
    supported = tuple(int(row["seed"]) for row in support_rows if bool(row["passed"]))
    support_pass = setting_availability_passes(
        len(supported),
        len(preset.seeds),
        protocol.support_gate.minimum_available_seed_fraction,
    )
    expected_support_summary = _gate_summary(
        "support", preset.seeds, supported, support_pass
    )
    if _json_sha256(_read_json(root / "support" / "summary.json")) != _json_sha256(
        expected_support_summary
    ):
        raise RuntimeError("support summary does not recompute from raw seed artifacts")
    panel = gate.get("panel_status")
    if panel == "GATE_NO_GO":
        reason = final.get("status")
        if (
            gate.get("reason") != reason
            or final.get("scientific_rows_saved") is not False
            or root_summary != final
            or _read_json(root / "NO_GO.json") != final
            or (root / "science").exists()
            or (root / "donor_overlap").exists()
        ):
            raise RuntimeError("gate NO-GO bundle contains scientific rows")
        if reason == "SUPPORT_NO_GO":
            if support_pass or (root / "k0_fidelity").exists():
                raise RuntimeError("SUPPORT_NO_GO does not recompute from support artifacts")
        elif reason in {"K0_FIDELITY_NO_GO", "STRUCTURAL_NO_GO"}:
            if not support_pass:
                raise RuntimeError("K0 NO-GO cannot follow a failed support gate")
            fidelity_rows = _load_phase_results(
                root / "k0_fidelity",
                phase="k0_fidelity",
                preset=preset,
                seeds=supported,
                dataset_metadata=dataset_metadata,
            )
            _assert_support_context_consistency(support_rows, fidelity_rows)
            numeric_passed = tuple(
                int(row["seed"]) for row in fidelity_rows if bool(row["passed"])
            )
            structural = tuple(
                int(row["seed"])
                for row in fidelity_rows
                if not bool(row["metrics"]["structural_invariants"])
            )
            numeric_setting_pass = setting_availability_passes(
                len(numeric_passed),
                len(preset.seeds),
                protocol.k0_fidelity_gate.minimum_available_seed_fraction,
            )
            expected_k0 = {
                **_gate_summary(
                    "logging-mixture one-step fidelity",
                    preset.seeds,
                    numeric_passed,
                    numeric_setting_pass and not structural,
                ),
                "structural_failure_seeds": list(structural),
                "structural_rule": "any exact-invariant failure is terminal",
                "numeric_availability_rule": ">=19/20 only after all exact invariants pass",
            }
            if _json_sha256(
                _read_json(root / "k0_fidelity" / "summary.json")
            ) != _json_sha256(expected_k0):
                raise RuntimeError("K0 summary does not recompute from raw seed artifacts")
            if reason == "STRUCTURAL_NO_GO" and not structural:
                raise RuntimeError("STRUCTURAL_NO_GO lacks an exact-invariant failure")
            if reason == "K0_FIDELITY_NO_GO" and (structural or numeric_setting_pass):
                raise RuntimeError("K0_FIDELITY_NO_GO numerical decision is stale")
        else:
            raise RuntimeError(f"unknown hard gate status: {reason}")
        if (root / "COMPLETE").read_text() != "gate-no-go\n":
            raise RuntimeError("gate NO-GO COMPLETE marker differs")
        return
    if panel not in {"CURVES", "CURVES_DESCRIPTIVE_ONLY"}:
        raise RuntimeError("unknown dataset panel status")
    if final.get("scientific_rows_saved") is not True or not support_pass:
        raise RuntimeError("curve bundle lacks scientific rows")
    fidelity_rows = _load_phase_results(
        root / "k0_fidelity",
        phase="k0_fidelity",
        preset=preset,
        seeds=supported,
        dataset_metadata=dataset_metadata,
    )
    _assert_support_context_consistency(support_rows, fidelity_rows)
    structural = tuple(
        int(row["seed"])
        for row in fidelity_rows
        if not bool(row["metrics"]["structural_invariants"])
    )
    seeds = tuple(int(row["seed"]) for row in fidelity_rows if bool(row["passed"]))
    k0_pass = setting_availability_passes(
        len(seeds),
        len(preset.seeds),
        protocol.k0_fidelity_gate.minimum_available_seed_fraction,
    )
    if structural or not k0_pass:
        raise RuntimeError("curve bundle bypassed the K0 hard gate")
    expected_k0 = {
        **_gate_summary(
            "logging-mixture one-step fidelity", preset.seeds, seeds, True
        ),
        "structural_failure_seeds": [],
        "structural_rule": "any exact-invariant failure is terminal",
        "numeric_availability_rule": ">=19/20 only after all exact invariants pass",
    }
    if _json_sha256(_read_json(root / "k0_fidelity" / "summary.json")) != _json_sha256(
        expected_k0
    ):
        raise RuntimeError("curve K0 summary is stale")
    overlap_rows = _load_phase_results(
        root / "donor_overlap",
        phase="donor_overlap",
        preset=preset,
        seeds=seeds,
        dataset_metadata=dataset_metadata,
    )
    science_payloads = _load_phase_results(
        root / "science",
        phase="science",
        preset=preset,
        seeds=seeds,
        dataset_metadata=dataset_metadata,
    )
    _assert_context_consistency(fidelity_rows, overlap_rows, label="K0/donor-overlap")
    _assert_context_consistency(fidelity_rows, science_payloads, label="K0/science")
    donor_pass = all(bool(row["passed"]) for row in overlap_rows)
    expected_interpretation = (
        "EMPIRICAL_OVERLAP_SCREEN_PASSED"
        if donor_pass
        else "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
    )
    expected_panel = "CURVES" if donor_pass else "CURVES_DESCRIPTIVE_ONLY"
    if panel != expected_panel or gate.get("interpretation_status") != expected_interpretation:
        raise RuntimeError("curve panel and interpretation status disagree")
    if any(
        payload.get("interpretation_status") != expected_interpretation
        for payload in science_payloads
    ):
        raise RuntimeError("science seed interpretation status disagrees with overlap screen")
    expected_overlap_summary = {
        **_gate_summary(
            "gamma=-4 q_mid+q_high empirical donor-overlap screen",
            seeds,
            tuple(int(row["seed"]) for row in overlap_rows if bool(row["passed"])),
            donor_pass,
        ),
        "failure_consequence": (
            "save and render full curves, but exclude confirmatory ranking, attainment, "
            "superiority, and cross-dataset conjunction"
        ),
        "interpretation_status": expected_interpretation,
    }
    if _json_sha256(
        _read_json(root / "donor_overlap" / "summary.json")
    ) != _json_sha256(expected_overlap_summary):
        raise RuntimeError("donor-overlap summary is stale")
    summary = _read_json(root / "science" / "summary.json")
    bootstrap_contract = _validate_bootstrap_artifacts(
        root / "science", summary.get("bootstrap"), preset, protocol
    )
    science_rows = [
        row for payload in science_payloads for row in payload["rows"]
    ]
    recomputed = summarize_science(
        science_rows,
        preset=preset,
        selected_seeds=seeds,
        interpretation_status=expected_interpretation,
        bootstrap_contract=bootstrap_contract,
    )
    if _json_sha256(summary) != _json_sha256(recomputed):
        raise RuntimeError("science summary does not recompute from raw rows/bootstrap")
    expected_final = {
        "protocol": PROTOCOL,
        "dataset": preset.name,
        "status": "COMPLETE",
        "scientific_rows_saved": True,
        "interpretation_status": expected_interpretation,
        "support_available": len(supported),
        "k0_fidelity_available": len(seeds),
        "prespecified_seeds": len(preset.seeds),
    }
    if final != expected_final:
        raise RuntimeError("FINAL_STATUS does not recompute from gate artifacts")
    expected_root_summary = {
        "protocol": PROTOCOL,
        "dataset": preset.name,
        "status": "COMPLETE",
        "interpretation_status": expected_interpretation,
        "science_summary_path": "science/summary.json",
        "scientific_rows_saved": True,
    }
    if root_summary != expected_root_summary:
        raise RuntimeError("dataset root summary is stale")
    expected_complete = "curves\n" if panel == "CURVES" else "curves-descriptive-only\n"
    if (root / "COMPLETE").read_text() != expected_complete:
        raise RuntimeError("curve COMPLETE marker differs")


def _load_phase_results(
    root: Path,
    *,
    phase: str,
    preset: DatasetPreset,
    seeds: tuple[int, ...],
    dataset_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if (root / "COMPLETE").read_text() != "complete\n":
        raise RuntimeError(f"{phase} phase COMPLETE marker differs")
    expected_files = {root / f"seed_{seed:05d}.json" for seed in seeds}
    if set(root.glob("seed_*.json")) != expected_files:
        raise RuntimeError(f"{phase} phase seed file set differs")
    results = []
    for seed in seeds:
        payload = _read_json(root / f"seed_{seed:05d}.json")
        if not _valid_phase_payload(
            payload,
            phase=phase,
            preset=replace(preset, seeds=seeds),
            seed=seed,
            device=dataset_metadata["seed_to_device"][str(seed)],
            seed_contract=dataset_metadata,
        ):
            raise RuntimeError(f"invalid {phase} artifact for seed {seed}")
        results.append(payload["result"])
    return results


def _validate_bootstrap_artifacts(
    science_root: Path,
    value: object,
    preset: DatasetPreset,
    protocol: ControlledClinicalExtensionConfig,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("science summary lacks the bootstrap contract")
    expected_keys = {
        "resamples",
        "root_seed",
        "prespecified_seed_count",
        "uniform_matrix_shape",
        "uniform_matrix_path",
        "uniform_matrix_sha256",
        "complete_seed_index_matrix_shape",
        "complete_seed_index_matrix_path",
        "complete_seed_index_matrix_sha256",
        "shared_across",
        "selected_subset_rule",
    }
    if set(value) != expected_keys:
        raise RuntimeError("bootstrap contract fields differ")
    if (
        value.get("uniform_matrix_path") != "bootstrap_uniforms.npy"
        or value.get("complete_seed_index_matrix_path") != "bootstrap_indices.npy"
    ):
        raise RuntimeError("bootstrap paths must be the frozen local filenames")
    uniform_path = science_root / "bootstrap_uniforms.npy"
    index_path = science_root / "bootstrap_indices.npy"
    if (
        value.get("resamples") != protocol.bootstrap_resamples
        or value.get("root_seed") != preset.bootstrap_seed
        or value.get("prespecified_seed_count") != len(preset.seeds)
        or value.get("uniform_matrix_shape")
        != [protocol.bootstrap_resamples, len(preset.seeds)]
        or value.get("complete_seed_index_matrix_shape")
        != [protocol.bootstrap_resamples, len(preset.seeds)]
        or value.get("shared_across") != ["methods", "gammas", "stages"]
        or value.get("selected_subset_rule")
        != (
            "for selected-set size n, use floor(U[:, :n] * n); the complete "
            "10,000x20 matrix is floor(U*20)"
        )
        or _file_sha256(uniform_path) != value.get("uniform_matrix_sha256")
        or _file_sha256(index_path) != value.get("complete_seed_index_matrix_sha256")
    ):
        raise RuntimeError("bootstrap artifact contract differs")
    uniforms = np.load(uniform_path, allow_pickle=False)
    indices = np.load(index_path, allow_pickle=False)
    expected_indices = np.floor(uniforms * len(preset.seeds)).astype(np.int16)
    if (
        uniforms.dtype != np.float64
        or indices.dtype != np.int16
        or uniforms.shape != (protocol.bootstrap_resamples, len(preset.seeds))
        or indices.shape != uniforms.shape
        or not np.isfinite(uniforms).all()
        or not ((uniforms >= 0.0) & (uniforms < 1.0)).all()
        or not np.array_equal(indices, expected_indices)
    ):
        raise RuntimeError("bootstrap arrays are malformed or incoherent")
    rng = np.random.default_rng(preset.bootstrap_seed)
    expected_uniforms = rng.random(uniforms.shape, dtype=np.float64)
    if not np.array_equal(uniforms, expected_uniforms):
        raise RuntimeError("bootstrap uniform matrix does not match its frozen seed")
    return value


def _write_manifest(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "COMPLETE"}:
            continue
        if ".tmp-" in path.name or path.suffix == ".tmp":
            raise RuntimeError(f"temporary artifact remains before manifest: {path}")
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
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
    if manifest.get("protocol") != PROTOCOL or not isinstance(entries, list):
        raise RuntimeError("invalid artifact manifest header")
    expected = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise RuntimeError("malformed artifact manifest entry")
        path = root / entry["path"]
        expected.add(path.resolve())
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or _file_sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"artifact manifest mismatch: {path}")
    observed = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "COMPLETE"}
    }
    if observed != expected or manifest.get("artifact_count") != len(entries):
        raise RuntimeError("artifact manifest file set differs")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))


def _write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.encode("utf-8"))


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_set_sha256(values: Iterable[int]) -> str:
    payload = json.dumps(sorted(set(int(value) for value in values)), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


if __name__ == "__main__":
    main()
