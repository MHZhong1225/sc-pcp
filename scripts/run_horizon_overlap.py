"""Run the isolated finite-MDP horizon--overlap diagnostic."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import git_revision, source_tree_sha256  # noqa: E402
from scpcp.horizon_overlap import run_horizon_overlap_study  # noqa: E402
from scpcp.horizon_overlap_config import (  # noqa: E402
    EXTERNAL_SEED_RESERVATIONS,
    METHOD_NAMES,
    PROTOCOL,
    HorizonOverlapConfig,
    horizon_overlap_seed_collision_audit,
)


DEFAULT_CONFIG = ROOT / "configs" / "horizon_overlap.yaml"
PARENT_SNAPSHOT_MANIFEST = (
    ROOT / "results" / "work" / "formal_source_snapshot_7665dfbe_20260825.manifest.json"
)
PARENT_SNAPSHOT_SIDECAR = (
    ROOT / "results" / "work" / "formal_source_snapshot_7665dfbe_20260825.manifest.sha256"
)
PARENT_SNAPSHOT_ARCHIVE = (
    ROOT / "results" / "work" / "formal_source_snapshot_7665dfbe_20260825.tar.gz"
)
EXPECTED_PARENT_MANIFEST_SHA256 = (
    "e6a1bba7f3be47d39357f212824e7720262e7d5212a14628e3b8981088c64e24"
)
EXPECTED_PARENT_ARCHIVE_SHA256 = (
    "2116b9929240d8a25092f3c9015c362957bc963532930e5e720dc7ec78b2ea0b"
)
EXPECTED_PARENT_SOURCE_TREE_SHA256 = (
    "7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643"
)
SEED_ARTIFACT_NAME = re.compile(r"seed_(\d+)(?:\.json)?$")
RNG_ASSIGNMENT = re.compile(r"seed|rng", re.IGNORECASE)
RESERVATION_ASSIGNMENT = re.compile(r"reserv|namespace", re.IGNORECASE)
PAYLOAD_NAMES = ("config.json", "metadata.json", "summary.json", "results.npz")
RESULT_ARRAY_NAMES = {
    "mixing_strength",
    "realized_reference_tv",
    "selected_indices",
    "failure_stage",
    "availability_by_horizon",
    "population_coverage",
    "population_width",
    "estimated_coverage",
    "estimated_width",
    "selected_ess_fraction",
    "minimum_candidate_ess_fraction",
    "stage_surface_sup_error",
    "selected_policy_realized_tv",
    "selected_policy_uniform_state_tv",
    "problem_seeds",
    "logging_seeds",
    "base_reference_tv",
    "method_names",
    "horizons",
    "nominal_policy_tvs",
    "bootstrap_seed",
    "bootstrap_instance_indices",
}
CONFIG_PROVENANCE_IDENTITY_FIELDS = (
    "selected_raw_config_sha256",
    "default_raw_config_sha256",
    "raw_config_is_default_byte_identical",
    "canonical_default_config_sha256",
    "scientific_config_contract_sha256",
)
FORMAL_RNG_IDENTITY_FIELDS = (
    "formal_rng_id_count",
    "formal_rng_id_sha256",
    "formal_rng_mapping",
    "formal_rng_mapping_sha256",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_formal_config(args.config)
    if args.output_dir is not None:
        config = config.with_output_dir(args.output_dir.resolve())
    run_study(config, resume=args.resume, config_path=args.config)
    print(config.output_dir)


def load_formal_config(path: Path) -> HorizonOverlapConfig:
    """Load only a byte-identical copy of the frozen default protocol."""

    selected = path.resolve()
    default = DEFAULT_CONFIG.resolve()
    if _sha256(selected) != _sha256(default):
        raise RuntimeError(
            "formal horizon-overlap config must be byte-identical to DEFAULT_CONFIG"
        )
    config = HorizonOverlapConfig.from_yaml(selected)
    canonical = HorizonOverlapConfig.from_yaml(default)
    if _scientific_config_contract(config) != _scientific_config_contract(canonical):
        raise RuntimeError("formal horizon-overlap canonical config contract differs")
    return config


def run_study(
    config: HorizonOverlapConfig,
    *,
    resume: bool,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run once, or validate and reuse one immutable completed bundle."""

    config.validate()
    selected_config_path = DEFAULT_CONFIG if config_path is None else config_path
    config_provenance = _formal_config_provenance(selected_config_path, config)
    parent_snapshot = _validate_parent_formal_snapshot()
    config_payload = config.to_dict()
    output_dir = config.output_dir.resolve()
    rng_audit = _audit_formal_rng_ids(
        config,
        output_dir=output_dir,
        selected_config_path=selected_config_path,
    )
    if resume:
        validate_bundle(
            output_dir,
            expected_config=config_payload,
            expected_config_provenance=config_provenance,
            expected_rng_audit=rng_audit,
            expected_parent_snapshot=parent_snapshot,
        )
        return _read_json(output_dir / "summary.json")
    if output_dir.exists():
        raise FileExistsError(f"fresh horizon-overlap output already exists: {output_dir}")

    seed_audit = horizon_overlap_seed_collision_audit(config)
    config_hash = _canonical_sha256(config_payload)
    source_hash = source_tree_sha256()
    started = time.perf_counter()
    result = run_horizon_overlap_study(config)
    elapsed = time.perf_counter() - started
    if source_tree_sha256() != source_hash:
        raise RuntimeError("source tree changed while the horizon-overlap study was running")
    if _formal_config_provenance(selected_config_path, config) != config_provenance:
        raise RuntimeError("formal config changed while the horizon-overlap study was running")
    if _validate_parent_formal_snapshot() != parent_snapshot:
        raise RuntimeError("parent formal snapshot changed while the study was running")
    if (
        _audit_formal_rng_ids(
            config,
            output_dir=output_dir,
            selected_config_path=selected_config_path,
        )
        != rng_audit
    ):
        raise RuntimeError("formal RNG inventory changed while the study was running")
    result.summary["parent_formal_snapshot"] = parent_snapshot
    result.summary["rq5_only_policy_center_reset"] = _policy_center_reset(config)
    metadata = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "diagnostic_only": True,
        "canonical_method_unchanged": True,
        "device": "cpu_exact",
        "git_revision": git_revision(),
        "source_tree_sha256": source_hash,
        "config_sha256": config_hash,
        "config_provenance": config_provenance,
        "seed_namespace": config.seed_namespace,
        "seed_collision_audit": seed_audit,
        "formal_rng_collision_audit": rng_audit,
        "parent_formal_snapshot": parent_snapshot,
        "rq5_only_policy_center_reset": _policy_center_reset(config),
        "policy_design_audit": result.summary["policy_design_audit"],
        "launch": {
            "argv": list(sys.argv),
            "cwd": str(Path.cwd()),
            "executable": sys.executable,
        },
        "environment_versions": _environment_versions(),
        "runtime_seconds": elapsed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _publish_bundle(
        output_dir,
        config_payload=config_payload,
        metadata=metadata,
        summary=result.summary,
        arrays=result.arrays,
    )
    validate_bundle(
        output_dir,
        expected_config=config_payload,
        expected_config_provenance=config_provenance,
        expected_rng_audit=rng_audit,
        expected_parent_snapshot=parent_snapshot,
    )
    return result.summary


def validate_bundle(
    output_dir: Path,
    *,
    expected_config: dict[str, Any],
    expected_config_provenance: dict[str, Any],
    expected_rng_audit: dict[str, Any],
    expected_parent_snapshot: dict[str, Any],
) -> None:
    """Fail closed on partial, mutated, or provenance-mismatched output.

    A completed-bundle resume validates and returns stored results; it does not
    execute the current numerical implementation.  The stored source and launch
    environment therefore remain bundle-bound, while later non-conflicting
    workspace growth is permitted.
    """

    if not output_dir.is_dir():
        raise FileNotFoundError(f"horizon-overlap output does not exist: {output_dir}")
    required = {*PAYLOAD_NAMES, "manifest.json", "COMPLETE"}
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"partial horizon-overlap bundle; missing files: {missing}")

    manifest = _read_json(output_dir / "manifest.json")
    if manifest != {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "files": {
            name: _file_contract(output_dir / name) for name in PAYLOAD_NAMES
        },
    }:
        raise RuntimeError("horizon-overlap manifest contract differs")
    metadata = _read_json(output_dir / "metadata.json")
    complete = _read_json(output_dir / "COMPLETE")
    if complete != {
        "status": "complete",
        "manifest_sha256": _sha256(output_dir / "manifest.json"),
        "config_sha256": metadata.get("config_sha256"),
        "source_tree_sha256": metadata.get("source_tree_sha256"),
        "config_provenance_sha256": metadata.get("config_provenance", {}).get(
            "provenance_sha256"
        ),
        "formal_rng_audit_sha256": metadata.get(
            "formal_rng_collision_audit", {}
        ).get("audit_sha256"),
        "parent_snapshot_contract_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("contract_sha256"),
        "parent_snapshot_manifest_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("manifest_sha256"),
        "parent_snapshot_archive_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("archive_sha256"),
        "parent_source_tree_sha256": metadata.get(
            "parent_formal_snapshot", {}
        ).get("parent_source_tree_sha256"),
        "launch_sha256": _canonical_sha256(metadata.get("launch", {})),
        "environment_versions_sha256": _canonical_sha256(
            metadata.get("environment_versions", {})
        ),
    }:
        raise RuntimeError("horizon-overlap COMPLETE marker is malformed")
    stored_config = _read_json(output_dir / "config.json")
    _validate_resume_config(
        stored_config,
        expected_config=expected_config,
        metadata=metadata,
        output_dir=output_dir,
    )
    if metadata.get("config_sha256") != _canonical_sha256(stored_config):
        raise RuntimeError("stored horizon-overlap config hash differs")
    if re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("source_tree_sha256"))) is None:
        raise RuntimeError("stored horizon-overlap source hash is malformed")
    _validate_config_provenance(
        metadata.get("config_provenance"),
        expected_config_provenance,
    )
    _validate_formal_rng_audit(
        metadata.get("formal_rng_collision_audit"),
        expected_rng_audit,
    )
    if metadata.get("parent_formal_snapshot") != expected_parent_snapshot:
        raise RuntimeError("resume parent formal snapshot binding differs")
    if metadata.get("canonical_method_unchanged") is not True:
        raise RuntimeError("horizon-overlap bundle changed the canonical method")
    if metadata.get("seed_collision_audit", {}).get("collision") is not False:
        raise RuntimeError("horizon-overlap bundle has an RNG collision")
    if metadata.get("policy_design_audit", {}).get("status") != "pass":
        raise RuntimeError("horizon-overlap policy-only design audit did not pass")
    formal_policy_audit = metadata.get("policy_design_audit", {}).get(
        "formal_problem_bank_attainability", {}
    )
    expected_problem_ids = list(
        range(
            expected_config["problem_seed_start"],
            expected_config["problem_seed_start"] + expected_config["instances"],
        )
    )
    if (
        formal_policy_audit.get("status") != "pass"
        or formal_policy_audit.get("seed_ids") != expected_problem_ids
        or formal_policy_audit.get("checked_before_any_logged_score_generation")
        is not True
    ):
        raise RuntimeError("not all formal problem policies passed pre-score audit")
    if not isinstance(metadata.get("launch", {}).get("argv"), list):
        raise RuntimeError("horizon-overlap launch argv is missing")
    if not isinstance(metadata.get("environment_versions"), dict):
        raise RuntimeError("horizon-overlap environment versions are missing")

    summary = _read_json(output_dir / "summary.json")
    if (
        summary.get("study") != "finite_mdp_horizon_overlap"
        or summary.get("status") != "complete"
        or summary.get("canonical_method_unchanged") is not True
        or summary.get("primary_coverage_estimand")
        != "min_stage_mean_instance_conditional_on_availability"
        or summary.get("parent_formal_snapshot") != expected_parent_snapshot
    ):
        raise RuntimeError("horizon-overlap summary contract differs")
    try:
        with np.load(output_dir / "results.npz", allow_pickle=False) as arrays:
            _validate_result_arrays(arrays, expected_config)
            _validate_summary(summary, arrays, expected_config)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"horizon-overlap results are unreadable: {error}") from error


def _validate_summary(
    summary: dict[str, Any],
    arrays: Any,
    config: dict[str, Any],
) -> None:
    expected_record_count = (
        len(config["horizons"])
        * len(config["nominal_policy_tvs"])
        * len(METHOD_NAMES)
    )
    if len(summary.get("records", ())) != expected_record_count:
        raise RuntimeError("horizon-overlap summary record count differs")
    bootstrap = summary.get("bootstrap", {})
    bootstrap_indices = arrays["bootstrap_instance_indices"]
    if (
        bootstrap.get("resamples") != config["bootstrap_resamples"]
        or bootstrap.get("seed") != config["bootstrap_seed"]
        or bootstrap.get("instance_index_matrix_shape")
        != list(bootstrap_indices.shape)
        or bootstrap.get("instance_index_matrix_sha256")
        != hashlib.sha256(np.ascontiguousarray(bootstrap_indices).tobytes()).hexdigest()
        or bootstrap.get(
            "shared_index_matrix_across_all_horizon_tv_method_cells"
        )
        is not True
        or bootstrap.get("wsc_recomputes_minimum_after_stagewise_resample_means")
        is not True
    ):
        raise RuntimeError("horizon-overlap bootstrap contract differs")
    expected_comparison_count = (
        len(config["horizons"]) * len(config["nominal_policy_tvs"]) * 2
    )
    comparisons = summary.get("bootstrap_wsc_comparisons", ())
    if len(comparisons) != expected_comparison_count:
        raise RuntimeError("horizon-overlap bootstrap comparison count differs")
    required_comparison_fields = {
        "method_conditional_scpcp_minus_comparator_wsc",
        "method_conditional_scpcp_minus_comparator_wsc_ci95",
        "joint_available_instances",
        "joint_availability_rate",
        "joint_available_scpcp_minus_comparator_wsc",
        "joint_available_scpcp_minus_comparator_wsc_ci95",
    }
    if any(not required_comparison_fields <= set(record) for record in comparisons):
        raise RuntimeError("horizon-overlap paired WSC comparison contract differs")
    if summary.get("policy_tv_reporting", {}).get(
        "selected_policy_uniform_state_tv"
    ) != "uniform_over_states":
        raise RuntimeError("uniform-state selected policy TV summary is missing")


def _validate_result_arrays(arrays: Any, config: dict[str, Any]) -> None:
    if set(arrays.files) != RESULT_ARRAY_NAMES:
        raise RuntimeError("horizon-overlap result array names differ")
    instances = config["instances"]
    tv_count = len(config["nominal_policy_tvs"])
    method_count = len(METHOD_NAMES)
    horizon_count = len(config["horizons"])
    maximum_horizon = max(config["horizons"])
    stage_shape = (instances, tv_count, method_count, maximum_horizon)

    if arrays["problem_seeds"].tolist() != list(
        range(config["problem_seed_start"], config["problem_seed_start"] + instances)
    ):
        raise RuntimeError("problem RNG IDs differ from the frozen config")
    if arrays["logging_seeds"].tolist() != list(
        range(config["logging_seed_start"], config["logging_seed_start"] + instances)
    ):
        raise RuntimeError("logging RNG IDs differ from the frozen config")
    if arrays["method_names"].tolist() != list(METHOD_NAMES):
        raise RuntimeError("horizon-overlap method names differ")
    if arrays["horizons"].tolist() != config["horizons"]:
        raise RuntimeError("horizon grid differs")
    if arrays["nominal_policy_tvs"].tolist() != config["nominal_policy_tvs"]:
        raise RuntimeError("nominal policy-TV grid differs")
    if arrays["mixing_strength"].shape != (instances, tv_count):
        raise RuntimeError("mixing_strength has the wrong shape")
    if arrays["realized_reference_tv"].shape != (instances, tv_count):
        raise RuntimeError("realized_reference_tv has the wrong shape")
    if arrays["base_reference_tv"].shape != (instances,):
        raise RuntimeError("base_reference_tv has the wrong shape")
    if arrays["bootstrap_seed"].shape != () or int(arrays["bootstrap_seed"]) != config[
        "bootstrap_seed"
    ]:
        raise RuntimeError("bootstrap_seed differs from the frozen config")
    if arrays["bootstrap_instance_indices"].shape != (
        config["bootstrap_resamples"],
        instances,
    ):
        raise RuntimeError("bootstrap_instance_indices has the wrong shape")
    if (
        np.any(arrays["bootstrap_instance_indices"] < 0)
        or np.any(arrays["bootstrap_instance_indices"] >= instances)
    ):
        raise RuntimeError("bootstrap_instance_indices contains an invalid instance")
    if arrays["selected_indices"].shape != stage_shape:
        raise RuntimeError("selected_indices has the wrong shape")
    if arrays["failure_stage"].shape != (instances, tv_count, method_count):
        raise RuntimeError("failure_stage has the wrong shape")
    if arrays["availability_by_horizon"].shape != (
        instances,
        tv_count,
        horizon_count,
        method_count,
    ):
        raise RuntimeError("availability_by_horizon has the wrong shape")
    for name in (
        "population_coverage",
        "population_width",
        "estimated_coverage",
        "estimated_width",
        "selected_ess_fraction",
        "minimum_candidate_ess_fraction",
        "stage_surface_sup_error",
        "selected_policy_realized_tv",
        "selected_policy_uniform_state_tv",
    ):
        if arrays[name].shape != stage_shape:
            raise RuntimeError(f"{name} has the wrong shape")
        if np.isinf(arrays[name]).any():
            raise RuntimeError(f"{name} contains infinite values")
    for name in ("mixing_strength", "realized_reference_tv", "base_reference_tv"):
        if not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"{name} contains non-finite values")

    failure = arrays["failure_stage"]
    expected_availability = np.stack(
        [
            (failure < 0) | (failure >= horizon)
            for horizon in config["horizons"]
        ],
        axis=2,
    )
    if not np.array_equal(arrays["availability_by_horizon"], expected_availability):
        raise RuntimeError("availability does not agree with failure_stage")
    for horizon_index, horizon in enumerate(config["horizons"]):
        available = arrays["availability_by_horizon"][:, :, horizon_index, :]
        finite_prefix = np.isfinite(arrays["population_coverage"][:, :, :, :horizon]).all(
            axis=3
        )
        if not np.all(finite_prefix[available]):
            raise RuntimeError("available selection has a non-finite coverage prefix")


def _scientific_config_contract(config: HorizonOverlapConfig) -> dict[str, Any]:
    contract = config.to_dict()
    contract.pop("output_dir")
    return contract


def _validate_resume_config(
    stored_config: dict[str, Any],
    *,
    expected_config: dict[str, Any],
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    stored_science = dict(stored_config)
    expected_science = dict(expected_config)
    stored_output = stored_science.pop("output_dir", None)
    expected_science.pop("output_dir", None)
    if stored_science != expected_science:
        raise RuntimeError("resume scientific config differs from horizon-overlap bundle")
    if not isinstance(stored_output, str):
        raise RuntimeError("stored horizon-overlap output directory is missing")

    recorded_output = Path(stored_output)
    if not recorded_output.is_absolute():
        launch_cwd = metadata.get("launch", {}).get("cwd")
        if not isinstance(launch_cwd, str):
            raise RuntimeError("relative stored output directory has no launch cwd")
        recorded_output = Path(launch_cwd) / recorded_output
    if recorded_output.resolve() != output_dir.resolve():
        raise RuntimeError("resume output directory differs from horizon-overlap bundle")


def _validate_config_provenance(
    stored: object,
    expected: dict[str, Any],
) -> None:
    if not isinstance(stored, dict):
        raise RuntimeError("stored horizon-overlap config provenance is missing")
    stored_without_hash = dict(stored)
    provenance_hash = stored_without_hash.pop("provenance_sha256", None)
    if provenance_hash != _canonical_sha256(stored_without_hash):
        raise RuntimeError("stored horizon-overlap config provenance hash differs")
    if any(
        stored.get(name) != expected.get(name)
        for name in CONFIG_PROVENANCE_IDENTITY_FIELDS
    ):
        raise RuntimeError("resume raw/default/canonical config hashes differ")


def _validate_formal_rng_audit(stored: object, expected: dict[str, Any]) -> None:
    if not isinstance(stored, dict):
        raise RuntimeError("stored horizon-overlap formal RNG audit is missing")
    stored_without_hash = dict(stored)
    audit_hash = stored_without_hash.pop("audit_sha256", None)
    if audit_hash != _canonical_sha256(stored_without_hash):
        raise RuntimeError("stored horizon-overlap formal RNG audit hash differs")
    if (
        stored.get("status") != "passed_before_launch"
        or stored.get("collision_count") != 0
        or stored.get("collisions") != {}
    ):
        raise RuntimeError("stored horizon-overlap formal RNG audit did not pass")
    if any(
        stored.get(name) != expected.get(name)
        for name in FORMAL_RNG_IDENTITY_FIELDS
    ):
        raise RuntimeError("resume formal RNG identity contract differs")


def _policy_center_reset(config: HorizonOverlapConfig) -> dict[str, Any]:
    return {
        "scope": "RQ5_horizon_overlap_only",
        "parent_exact_M3_policy_response_center": (
            config.parent_policy_response_center
        ),
        "rq5_policy_response_center": config.radius_minimum,
        "reason": (
            "make the frozen median-grid policy-only TV targets through 0.15 "
            "attainable before observing any outcome, score, or coverage"
        ),
        "canonical_SC_PCP_changed": False,
        "parent_RQ1_results_reinterpreted": False,
    }


def _environment_versions() -> dict[str, Any]:
    import torch

    numpy_configuration = getattr(np.__config__, "CONFIG", {})
    build_dependencies = numpy_configuration.get("Build Dependencies", {})
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
            "git_version": torch.version.git_version,
        },
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


def _formal_config_provenance(
    selected_config_path: Path,
    effective_config: HorizonOverlapConfig,
) -> dict[str, Any]:
    selected = selected_config_path.resolve()
    default = DEFAULT_CONFIG.resolve()
    selected_raw_hash = _sha256(selected)
    default_raw_hash = _sha256(default)
    if selected_raw_hash != default_raw_hash:
        raise RuntimeError(
            "formal horizon-overlap config must be byte-identical to DEFAULT_CONFIG"
        )
    selected_config = HorizonOverlapConfig.from_yaml(selected)
    default_config = HorizonOverlapConfig.from_yaml(default)
    default_contract = _scientific_config_contract(default_config)
    if _scientific_config_contract(selected_config) != default_contract:
        raise RuntimeError("selected config differs from the canonical default contract")
    if _scientific_config_contract(effective_config) != default_contract:
        raise RuntimeError(
            "formal horizon-overlap config permits only an output-dir override"
        )
    provenance = {
        "selected_config_path": str(selected),
        "default_config_path": str(default),
        "selected_raw_config_sha256": selected_raw_hash,
        "default_raw_config_sha256": default_raw_hash,
        "raw_config_is_default_byte_identical": True,
        "canonical_default_config_sha256": _canonical_sha256(
            default_config.to_dict()
        ),
        "scientific_config_contract_sha256": _canonical_sha256(default_contract),
        "effective_config_sha256": _canonical_sha256(effective_config.to_dict()),
        "explicit_output_dir_override": (
            effective_config.output_dir.resolve() != default_config.output_dir.resolve()
        ),
    }
    provenance["provenance_sha256"] = _canonical_sha256(provenance)
    return provenance


def _validate_parent_formal_snapshot(
    *,
    manifest_path: Path = PARENT_SNAPSHOT_MANIFEST,
    sidecar_path: Path = PARENT_SNAPSHOT_SIDECAR,
    archive_path: Path = PARENT_SNAPSHOT_ARCHIVE,
) -> dict[str, Any]:
    """Validate the content-addressed parent M3 source snapshot."""

    manifest_path = manifest_path.resolve()
    sidecar_path = sidecar_path.resolve()
    archive_path = archive_path.resolve()
    manifest_hash = _sha256(manifest_path)
    archive_hash = _sha256(archive_path)
    if manifest_hash != EXPECTED_PARENT_MANIFEST_SHA256:
        raise RuntimeError("parent formal snapshot manifest SHA256 differs")
    if archive_hash != EXPECTED_PARENT_ARCHIVE_SHA256:
        raise RuntimeError("parent formal source archive SHA256 differs")
    sidecar_fields = sidecar_path.read_text(encoding="utf-8").split()
    if not sidecar_fields or sidecar_fields[0] != EXPECTED_PARENT_MANIFEST_SHA256:
        raise RuntimeError("parent formal snapshot manifest sidecar differs")

    manifest = _read_json(manifest_path)
    archive = manifest.get("archive", {})
    exact_parent = manifest.get("bound_formal_studies", {}).get(
        "exact_finite_mdp_20260825"
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("role") != "content_addressed_formal_source_snapshot"
        or manifest.get("source_tree_sha256")
        != EXPECTED_PARENT_SOURCE_TREE_SHA256
        or archive.get("sha256") != EXPECTED_PARENT_ARCHIVE_SHA256
        or archive.get("bytes") != archive_path.stat().st_size
        or archive.get("path")
        != "results/work/formal_source_snapshot_7665dfbe_20260825.tar.gz"
        or not isinstance(exact_parent, dict)
        or exact_parent.get("root") != "results/work/exact_finite_mdp_20260825"
    ):
        raise RuntimeError("parent formal snapshot manifest contract differs")
    contract = {
        "role": "parent_exact_M3_formal_source_snapshot",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest_sidecar_path": str(sidecar_path),
        "archive_path": str(archive_path),
        "archive_sha256": archive_hash,
        "archive_bytes": archive_path.stat().st_size,
        "parent_source_tree_sha256": EXPECTED_PARENT_SOURCE_TREE_SHA256,
        "parent_git_revision": manifest.get("git_revision"),
        "parent_exact_finite_mdp": exact_parent,
        "provenance_boundary": (
            "RQ5 extends the parent M3 family from this snapshot but runs under "
            "its own active source hash and RQ5-only policy-center reset"
        ),
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _audit_formal_rng_ids(
    config: HorizonOverlapConfig,
    *,
    output_dir: Path,
    selected_config_path: Path,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Reject prior use of every design, problem, logging, and bootstrap ID."""

    artifact_root = ROOT / "results" if artifact_root is None else artifact_root
    source_root = ROOT if source_root is None else source_root
    mapping = {
        **{
            f"policy_design/base_{seed}": seed
            for seed in config.design_seeds
        },
        **{
            f"instance_{index:03d}/problem": seed
            for index, seed in enumerate(config.problem_seeds)
        },
        **{
            f"instance_{index:03d}/logging": seed
            for index, seed in enumerate(config.logging_seeds)
        },
        "summary/instance_cluster_bootstrap": config.bootstrap_seed,
    }
    formal_ids = set(mapping.values())
    if len(formal_ids) != len(mapping):
        raise RuntimeError("formal horizon-overlap RNG streams are not unique")
    enumerated = set(
        horizon_overlap_seed_collision_audit(config)["all_rng_ids"]
    )
    if formal_ids != enumerated:
        raise RuntimeError("formal RNG mapping differs from the config enumeration")

    artifact_ids = _artifact_rng_ids(artifact_root, excluded_root=output_dir)
    excluded_paths = {
        Path(__file__).resolve(),
        (ROOT / "src" / "scpcp" / "horizon_overlap_config.py").resolve(),
        DEFAULT_CONFIG.resolve(),
        selected_config_path.resolve(),
    }
    source_ids = _source_declared_rng_ids(
        source_root,
        excluded_paths=excluded_paths,
    )
    coordinated_external_ids = set().union(
        *(set(values) for values in EXTERNAL_SEED_RESERVATIONS.values())
    )
    prior_ids = artifact_ids | source_ids | coordinated_external_ids
    collisions = {
        label: identifier
        for label, identifier in mapping.items()
        if identifier in prior_ids
    }
    audit = {
        "status": "passed_before_launch" if not collisions else "collision",
        "formal_rng_id_count": len(formal_ids),
        "formal_rng_id_sha256": _integer_set_sha256(formal_ids),
        "formal_rng_mapping": mapping,
        "formal_rng_mapping_sha256": _canonical_sha256(mapping),
        "artifact_rng_id_count": len(artifact_ids),
        "artifact_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "source_declared_rng_id_count": len(source_ids),
        "source_declared_rng_id_sha256": _integer_set_sha256(source_ids),
        "coordinated_external_rng_id_count": len(coordinated_external_ids),
        "coordinated_external_rng_id_sha256": _integer_set_sha256(
            coordinated_external_ids
        ),
        "prior_declared_or_artifact_rng_id_count": len(prior_ids),
        "prior_declared_or_artifact_rng_id_sha256": _integer_set_sha256(prior_ids),
        "collision_count": len(collisions),
        "collisions": collisions,
        "excluded_output": str(output_dir.resolve()),
        "excluded_source_declarations": sorted(str(path) for path in excluded_paths),
    }
    audit["audit_sha256"] = _canonical_sha256(audit)
    if collisions:
        raise RuntimeError(
            "formal horizon-overlap RNG IDs collide with prior declaration/artifact use: "
            f"{collisions}"
        )
    return audit


def _artifact_rng_ids(root: Path, *, excluded_root: Path) -> set[int]:
    values: set[int] = set()
    if not root.exists():
        return values
    excluded = excluded_root.resolve()
    payload_names = {
        "metadata.json",
        "study_metadata.json",
        "manifest.json",
        "summary.json",
        "config.json",
        "config.yaml",
    }
    for path in root.rglob("*"):
        if _is_relative_to(path.resolve(), excluded):
            continue
        match = SEED_ARTIFACT_NAME.fullmatch(path.name)
        if match:
            values.add(int(match.group(1)))
        if not path.is_file() or path.name not in payload_names:
            continue
        try:
            payload = (
                yaml.safe_load(path.read_text())
                if path.suffix in {".yaml", ".yml"}
                else json.loads(path.read_text())
            )
        except (OSError, json.JSONDecodeError, yaml.YAMLError):
            continue
        _collect_named_rng_values(payload, values)
    return values


def _source_declared_rng_ids(
    root: Path,
    *,
    excluded_paths: set[Path],
) -> set[int]:
    values: set[int] = set()
    for directory in ("scripts", "src", "tools", "configs"):
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.resolve() in excluded_paths or not path.is_file():
                continue
            path_values: set[int] = set()
            if path.suffix == ".py":
                _collect_python_rng_assignments(path, path_values)
            elif path.suffix in {".yaml", ".yml"}:
                try:
                    _collect_named_rng_values(
                        yaml.safe_load(path.read_text()),
                        path_values,
                    )
                except (OSError, yaml.YAMLError):
                    continue
            values.update(path_values)
    return values


def _collect_python_rng_assignments(path: Path, values: set[int]) -> None:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [name for target in node.targets for name in _target_names(target)]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            names = list(_target_names(node.target))
            value = node.value
        else:
            continue
        if (
            value is None
            or not any(RNG_ASSIGNMENT.search(name) for name in names)
            or any(RESERVATION_ASSIGNMENT.search(name) for name in names)
        ):
            continue
        evaluated = _literal_rng_expression(value)
        if evaluated is not None:
            values.update(evaluated)


def _target_names(target: ast.expr) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_names(element)


def _literal_rng_expression(node: ast.expr) -> set[int] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return {node.value}
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        parts = [_literal_rng_expression(element) for element in node.elts]
        if any(part is None for part in parts):
            return None
        return set().union(*(part or set() for part in parts))
    if isinstance(node, ast.Dict):
        parts = [_literal_rng_expression(value) for value in node.values]
        return set().union(*(part or set() for part in parts))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "range":
            arguments = [_literal_integer(argument) for argument in node.args]
            if any(argument is None for argument in arguments):
                return None
            return set(range(*(int(argument) for argument in arguments)))
        if node.func.id in {"tuple", "list", "set", "frozenset"} and len(
            node.args
        ) == 1:
            return _literal_rng_expression(node.args[0])
    return None


def _literal_integer(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _literal_integer(node.operand)
        return None if operand is None else -operand
    return None


def _collect_named_rng_values(
    value: object,
    output: set[int],
    key_path: str = "",
) -> None:
    if RESERVATION_ASSIGNMENT.search(key_path):
        return
    if isinstance(value, dict):
        if RNG_ASSIGNMENT.search(key_path) and {"start", "stop"} <= set(value):
            start, stop = value["start"], value["stop"]
            if isinstance(start, int) and isinstance(stop, int):
                output.update(range(start, stop))
        for child_key, child_value in value.items():
            child_path = f"{key_path}.{child_key}" if key_path else str(child_key)
            _collect_named_rng_values(child_value, output, child_path)
        return
    if isinstance(value, list):
        for child in value:
            _collect_named_rng_values(child, output, key_path)
        return
    if (
        RNG_ASSIGNMENT.search(key_path)
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        output.add(value)


def _integer_set_sha256(values: Iterable[int]) -> str:
    encoded = json.dumps(sorted(set(values)), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _publish_bundle(
    output_dir: Path,
    *,
    config_payload: dict[str, Any],
    metadata: dict[str, Any],
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        _write_json(temporary / "config.json", config_payload)
        _write_json(temporary / "metadata.json", metadata)
        _write_json(temporary / "summary.json", summary)
        np.savez_compressed(temporary / "results.npz", **arrays)
        _fsync_file(temporary / "results.npz")
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "complete",
            "files": {
                name: _file_contract(temporary / name) for name in PAYLOAD_NAMES
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        _write_json(
            temporary / "COMPLETE",
            {
                "status": "complete",
                "manifest_sha256": _sha256(temporary / "manifest.json"),
                "config_sha256": metadata["config_sha256"],
                "source_tree_sha256": metadata["source_tree_sha256"],
                "config_provenance_sha256": metadata["config_provenance"][
                    "provenance_sha256"
                ],
                "formal_rng_audit_sha256": metadata[
                    "formal_rng_collision_audit"
                ]["audit_sha256"],
                "parent_snapshot_contract_sha256": metadata[
                    "parent_formal_snapshot"
                ]["contract_sha256"],
                "parent_snapshot_manifest_sha256": metadata[
                    "parent_formal_snapshot"
                ]["manifest_sha256"],
                "parent_snapshot_archive_sha256": metadata[
                    "parent_formal_snapshot"
                ]["archive_sha256"],
                "parent_source_tree_sha256": metadata[
                    "parent_formal_snapshot"
                ]["parent_source_tree_sha256"],
                "launch_sha256": _canonical_sha256(metadata["launch"]),
                "environment_versions_sha256": _canonical_sha256(
                    metadata["environment_versions"]
                ),
            },
        )
        _fsync_directory(temporary)
        os.replace(temporary, output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _canonical_sha256(payload: dict[str, Any]) -> str:
    content = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _file_contract(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unreadable horizon-overlap JSON: {path.name}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"horizon-overlap JSON must contain an object: {path.name}")
    return payload


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    main()
