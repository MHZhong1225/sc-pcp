"""Run the isolated RQ6 calibration-size convergence protocol.

The formal launch is CPU-exact and problem-parallel.  It never edits or wraps
the canonical SC-PCP selector.  Use ``--preflight-only`` for the engineering
runtime/memory smoke; that mode does not consume any formal scientific RNG ID.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import git_revision, source_tree_sha256  # noqa: E402
from scpcp.exact_finite_mdp import (  # noqa: E402
    ESTIMATOR_NAMES,
    enumerate_schedules,
    exact_population_surfaces,
)
from scpcp.rq6_ncal_convergence import (  # noqa: E402
    RQ6ConvergenceConfig,
    build_outcome_blind_m3,
    calibration_role_sizes,
    evaluate_track_a_nested_prefixes,
    evaluate_track_b_canonical_selector,
    logged_rng_ids,
    run_problem,
    simulate_nested_role_pools,
    summarize_problem_results,
)


DEFAULT_CONFIG = ROOT / "configs" / "rq6_ncal_convergence.yaml"
PARENT_SNAPSHOT_MANIFEST = (
    ROOT / "results" / "work" / "formal_source_snapshot_7665dfbe_20260825.manifest.json"
)
PARENT_SNAPSHOT_ARCHIVE = (
    ROOT / "results" / "work" / "formal_source_snapshot_7665dfbe_20260825.tar.gz"
)
PARENT_SNAPSHOT_MANIFEST_SHA256 = (
    "e6a1bba7f3be47d39357f212824e7720262e7d5212a14628e3b8981088c64e24"
)
PARENT_SNAPSHOT_ARCHIVE_SHA256 = (
    "2116b9929240d8a25092f3c9015c362957bc963532930e5e720dc7ec78b2ea0b"
)
PARENT_SOURCE_TREE_SHA256 = (
    "7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643"
)
PARENT_SNAPSHOT_ARCHIVE_BYTES = 2_036_776
PROBLEM_FILES = ("result.json", "metadata.json", "COMPLETE")
RNG_KEY = re.compile(r"seed|rng", re.IGNORECASE)
RESERVATION_KEY = re.compile(r"reservation", re.IGNORECASE)
ARTIFACT_ID = re.compile(r"(?:seed|problem)_(\d+)(?:\.json)?$")

# These are coordinated whole namespaces, not merely the IDs used by earlier
# completed artifacts.  The assigned RQ6 problem namespace is 97000..97999;
# its derived logged streams live at 97.1 million and are enumerated separately.
COORDINATED_EXTERNAL_RESERVATIONS = {
    "exact_finite_mdp": range(52_000, 53_000),
    "controlled_six_method": range(91_000, 92_000),
    "orthogonal_copula": range(94_000, 95_000),
    "rq5_horizon_overlap": range(96_000, 97_000),
    "propensity_robustness": range(98_000, 99_000),
    "strict_split_audit": range(99_000, 100_000),
    "future_score_robustness": range(100_000, 101_000),
}
COORDINATED_RESERVATION_LABELS = {
    "exact_finite_mdp": "52000..52999",
    "controlled_six_method": "91000..91999",
    "orthogonal_copula": "94000..94999",
    "rq5_horizon_overlap": "96000..96999",
    "rq6_ncal_convergence": "97000..97999 (assigned here)",
    "propensity_robustness": "98000..98999",
    "strict_split_audit": "99000..99999",
    "future_score_robustness": "100000..100999",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--surface-chunk-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run only the non-scientific engineering runtime/memory smoke",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = RQ6ConvergenceConfig.from_yaml(args.config)
    output = None if args.output_dir is None else args.output_dir.resolve()
    config = base.with_runtime_overrides(
        output_dir=output,
        workers=args.workers,
        surface_chunk_size=args.surface_chunk_size,
    )
    if args.preflight_only:
        print(json.dumps(run_runtime_preflight(config), sort_keys=True, indent=2))
        return
    run_study(config, resume=args.resume)
    print(config.output_dir.resolve())


def run_study(config: RQ6ConvergenceConfig, *, resume: bool) -> dict[str, Any]:
    """Run or fail-closed resume the complete frozen RQ6 study."""

    config.validate()
    config.assert_frozen_protocol()
    output_dir = config.output_dir.resolve()
    config_payload = config.to_dict()
    config_hash = _json_sha256(config_payload)
    source_hash = source_tree_sha256()
    parent_snapshot = validate_parent_snapshot()
    environment = runtime_environment()
    environment_hash = _json_sha256(environment)
    invocation_argv = list(sys.argv)
    launch_argv = invocation_argv
    if resume:
        if not (output_dir / "manifest.json").is_file():
            raise FileNotFoundError("RQ6 resume requires an existing manifest.json")
        stored_manifest = _read_json(output_dir / "manifest.json")
        launch_argv = stored_manifest.get("launch_argv")
        if not isinstance(launch_argv, list) or not all(
            isinstance(value, str) for value in launch_argv
        ):
            raise RuntimeError("RQ6 stored launch argv is malformed")
    rng_audit = audit_formal_rng_ids(config, output_dir=output_dir)
    manifest = _root_manifest(
        config,
        config_hash=config_hash,
        source_hash=source_hash,
        rng_audit=rng_audit,
        parent_snapshot=parent_snapshot,
        environment=environment,
        launch_argv=launch_argv,
    )

    if resume:
        completed = validate_resume(output_dir, config=config, manifest=manifest)
        if (output_dir / "COMPLETE").is_file():
            validate_complete_bundle(output_dir, config=config, manifest=manifest)
            return _read_json(output_dir / "summary.json")
    else:
        if output_dir.exists():
            raise FileExistsError(f"fresh RQ6 output already exists: {output_dir}")
        # The tiny engineering run is mandatory and uses no formal IDs.  It is
        # completed before any formal output directory or job is created.
        preflight = run_runtime_preflight(
            config,
            parent_snapshot=parent_snapshot,
            environment=environment,
            invocation_argv=invocation_argv,
        )
        if source_tree_sha256() != source_hash:
            raise RuntimeError("source tree changed during the RQ6 runtime preflight")
        output_dir.mkdir(parents=True)
        _atomic_write_text(
            output_dir / "config.yaml",
            yaml.safe_dump(config_payload, sort_keys=False),
        )
        _atomic_write_json(output_dir / "manifest.json", manifest)
        _atomic_write_json(output_dir / "runtime_preflight.json", preflight)
        completed = set()

    _reject_unknown_or_temporary_problem_paths(output_dir, config)
    pending = tuple(
        (index, seed)
        for index, seed in enumerate(config.problem_seeds)
        if seed not in completed
    )
    if (output_dir / "COMPLETE").exists() and pending:
        raise RuntimeError("root COMPLETE exists while problem artifacts are missing")
    _write_status(
        output_dir,
        config,
        completed=completed,
        status="running",
        invocation_argv=invocation_argv,
    )
    try:
        _run_pending_problems(
            config,
            pending,
            output_dir=output_dir,
            config_hash=config_hash,
            source_hash=source_hash,
            parent_snapshot=parent_snapshot,
            environment_hash=environment_hash,
        )
        results = []
        for problem_index, problem_seed in enumerate(config.problem_seeds):
            path = output_dir / f"problem_{problem_seed}"
            result = validate_problem_artifact(
                path,
                config=config,
                problem_index=problem_index,
                problem_seed=problem_seed,
                config_hash=config_hash,
                source_hash=source_hash,
                parent_snapshot=parent_snapshot,
                environment_hash=environment_hash,
            )
            results.append(result)
        if source_tree_sha256() != source_hash:
            raise RuntimeError("source tree changed while RQ6 was running")
        if audit_formal_rng_ids(config, output_dir=output_dir) != rng_audit:
            raise RuntimeError("RNG collision inventory changed while RQ6 was running")
        if validate_parent_snapshot() != parent_snapshot:
            raise RuntimeError("parent source snapshot changed while RQ6 was running")
        if runtime_environment() != environment:
            raise RuntimeError("Python/NumPy/Torch/BLAS environment changed during RQ6")

        summary = summarize_problem_results(results, config)
        summary.update(
            {
                "status": "complete",
                "source_tree_sha256": source_hash,
                "config_sha256": config_hash,
                "formal_problem_count": len(results),
                "parent_snapshot": parent_snapshot,
                "runtime_environment_sha256": environment_hash,
            }
        )
        _atomic_write_json(output_dir / "summary.json", summary)
        artifact_manifest = _artifact_manifest(output_dir, config)
        _atomic_write_json(output_dir / "artifact_manifest.json", artifact_manifest)
        _write_status(
            output_dir,
            config,
            completed=set(config.problem_seeds),
            status="complete",
            invocation_argv=invocation_argv,
        )
        complete = {
            "status": "complete",
            "protocol": config.protocol,
            "config_sha256": config_hash,
            "source_tree_sha256": source_hash,
            "parent_snapshot_manifest_sha256": parent_snapshot["manifest_sha256"],
            "parent_snapshot_archive_sha256": parent_snapshot["archive_sha256"],
            "parent_source_tree_sha256": parent_snapshot["source_tree_sha256"],
            "runtime_environment_sha256": environment_hash,
            "manifest_sha256": _file_sha256(output_dir / "manifest.json"),
            "runtime_preflight_sha256": _file_sha256(
                output_dir / "runtime_preflight.json"
            ),
            "summary_sha256": _file_sha256(output_dir / "summary.json"),
            "artifact_manifest_sha256": _file_sha256(
                output_dir / "artifact_manifest.json"
            ),
        }
        _atomic_write_json(output_dir / "COMPLETE", complete)
        _fsync_directory(output_dir)
        validate_complete_bundle(output_dir, config=config, manifest=manifest)
        return summary
    except BaseException as error:
        completed_now = {
            seed
            for seed in config.problem_seeds
            if (output_dir / f"problem_{seed}" / "COMPLETE").is_file()
        }
        _write_status(
            output_dir,
            config,
            completed=completed_now,
            status="failed",
            error=error,
            invocation_argv=invocation_argv,
        )
        raise


def run_runtime_preflight(
    config: RQ6ConvergenceConfig,
    *,
    parent_snapshot: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    invocation_argv: list[str] | None = None,
) -> dict[str, Any]:
    """Time one excluded 250-trajectory cell and extrapolate formal resources."""

    parent_snapshot = (
        validate_parent_snapshot() if parent_snapshot is None else parent_snapshot
    )
    environment = runtime_environment() if environment is None else environment
    invocation_argv = list(sys.argv) if invocation_argv is None else invocation_argv
    smoke_n = min(config.n_calibration)
    smoke = replace(
        config,
        n_calibration=(smoke_n,),
        problem_seed_start=7,
        problem_count=1,
        logged_replicates=1,
        bootstrap_resamples=10,
        workers=1,
        seed_namespace="engineering_runtime_smoke_excluded",
    )
    smoke.validate()
    timings: dict[str, float] = {}
    started = time.perf_counter()
    mechanism, policy = build_outcome_blind_m3(smoke, problem_seed=7)
    schedules = enumerate_schedules(smoke.exact_config(logged_trajectories=1, seed=7))
    population, _ = exact_population_surfaces(mechanism, schedules)
    true_surface = population[ESTIMATOR_NAMES.index("full_prefix")]
    timings["population_setup_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    cot_pool, certification_pool = simulate_nested_role_pools(
        smoke,
        mechanism,
        cot_rng=11,
        certification_rng=13,
        problem_seed=7,
    )
    timings["maximum_pool_simulation_seconds"] = time.perf_counter() - started
    role_sizes = (calibration_role_sizes(smoke_n, smoke),)

    started = time.perf_counter()
    track_a = evaluate_track_a_nested_prefixes(
        mechanism,
        schedules,
        true_surface,
        cot_pool,
        certification_pool,
        n_calibration=(smoke_n,),
        role_sizes=role_sizes,
    )[smoke_n]
    timings["track_a_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    track_b = evaluate_track_b_canonical_selector(
        smoke,
        mechanism,
        policy,
        cot_pool,
        certification_pool,
        n_calibration=smoke_n,
    )
    timings["track_b_seconds"] = time.perf_counter() - started
    timings["total_seconds"] = sum(timings.values())

    maximum_n = max(config.n_calibration)
    final_weight_bytes = config.grid_size**config.horizon * maximum_n * 8
    adjacent_matrix_bytes = config.grid_size ** (config.horizon - 1) * maximum_n * 8
    analytic_peak_bytes = final_weight_bytes + adjacent_matrix_bytes
    sample_sum_scale = sum(config.n_calibration) / smoke_n
    maximum_pool_scale = maximum_n / smoke_n
    estimated_replicate_seconds = (
        timings["maximum_pool_simulation_seconds"] * maximum_pool_scale
        + timings["track_a_seconds"] * sample_sum_scale
        + timings["track_b_seconds"] * sample_sum_scale
    )
    estimated_problem_seconds = (
        timings["population_setup_seconds"]
        + config.logged_replicates * estimated_replicate_seconds
    )
    estimated_wall_hours = (
        estimated_problem_seconds * config.problem_count / config.workers / 3_600.0
    )
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    observed_process_bytes = maximum_rss * 1_024
    planning_peak_bytes = observed_process_bytes + analytic_peak_bytes
    # Linux reports KiB; macOS reports bytes.  This workspace is Linux, but the
    # explicit field name keeps the observed unit honest.
    return {
        "status": "passed",
        "scientific_use": False,
        "formal_rng_ids_consumed": [],
        "engineering_rng_ids": {
            "problem": 7,
            "cot_pool": 11,
            "certification_pool": 13,
        },
        "geometry": {
            "horizon": config.horizon,
            "grid_size": config.grid_size,
            "smoke_n_calibration": smoke_n,
            "formal_n_calibration": list(config.n_calibration),
            "unique_prefix_counts": track_a["unique_prefix_counts"],
            "complete_schedule_count": track_a["complete_schedule_count"],
        },
        "contract_checks": {
            "track_a_complete_prefix_surface": (
                track_a["unique_prefix_counts"] == [7, 49, 343, 2_401]
            ),
            "track_b_canonical_selector_completed": isinstance(
                track_b["selection_available"], bool
            ),
            "reference_policy_tv": policy.mean_state_tv(config.reference_radius),
        },
        "observed_timings": timings,
        "extrapolation": {
            "model": (
                "kernel-only linear extrapolation: maximum pool size for simulation, "
                "and conservative sum(n) scaling for each six-n Track-A/Track-B scan"
            ),
            "maximum_pool_scale_from_smoke": maximum_pool_scale,
            "sum_n_scale_from_smoke": sample_sum_scale,
            "estimated_seconds_per_formal_problem": estimated_problem_seconds,
            "kernel_only_estimated_formal_wall_hours_at_configured_workers": (
                estimated_wall_hours
            ),
            "planning_wall_hours_with_3x_process_io_margin": 3.0 * estimated_wall_hours,
            "configured_workers": config.workers,
        },
        "memory": {
            "final_track_a_weight_matrix_bytes": final_weight_bytes,
            "adjacent_prefix_or_candidate_matrix_bytes": adjacent_matrix_bytes,
            "analytic_peak_incremental_bytes_per_worker": analytic_peak_bytes,
            "recommended_incremental_reserve_bytes_per_worker": 512 * 2**20,
            "recommended_incremental_reserve_bytes_all_workers": (
                config.workers * 512 * 2**20
            ),
            "observed_process_ru_maxrss_kib": maximum_rss,
            "observed_smoke_process_maxrss_bytes": observed_process_bytes,
            "planning_peak_bytes_per_worker": planning_peak_bytes,
            "recommended_total_reserve_bytes_per_worker": 1 * 2**30,
            "recommended_total_reserve_bytes_all_workers": config.workers * 1 * 2**30,
        },
        "source_tree_sha256": source_tree_sha256(),
        "config_sha256": _json_sha256(config.to_dict()),
        "parent_snapshot": parent_snapshot,
        "runtime_environment": environment,
        "runtime_environment_sha256": _json_sha256(environment),
        "argv": invocation_argv,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def validate_parent_snapshot(
    *,
    manifest_path: Path | None = None,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and return the content-addressed parent formal snapshot."""

    manifest_path = (
        PARENT_SNAPSHOT_MANIFEST if manifest_path is None else manifest_path
    ).resolve()
    archive_path = (
        PARENT_SNAPSHOT_ARCHIVE if archive_path is None else archive_path
    ).resolve()
    if not manifest_path.is_file() or not archive_path.is_file():
        raise FileNotFoundError("RQ6 requires the parent formal source snapshot bundle")
    manifest_hash = _file_sha256(manifest_path)
    archive_hash = _file_sha256(archive_path)
    if manifest_hash != PARENT_SNAPSHOT_MANIFEST_SHA256:
        raise RuntimeError("parent source snapshot manifest hash differs")
    manifest = _read_json(manifest_path)
    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        raise RuntimeError("parent source snapshot archive contract is malformed")
    expected_manifest_fields = {
        "schema_version": 1,
        "role": "content_addressed_formal_source_snapshot",
        "source_tree_sha256": PARENT_SOURCE_TREE_SHA256,
    }
    for field, expected in expected_manifest_fields.items():
        if manifest.get(field) != expected:
            raise RuntimeError(f"parent source snapshot field {field} differs")
    expected_archive_fields = {
        "path": (
            "results/work/formal_source_snapshot_7665dfbe_20260825.tar.gz"
        ),
        "sha256": PARENT_SNAPSHOT_ARCHIVE_SHA256,
        "bytes": PARENT_SNAPSHOT_ARCHIVE_BYTES,
    }
    for field, expected in expected_archive_fields.items():
        if archive.get(field) != expected:
            raise RuntimeError(f"parent source snapshot archive field {field} differs")
    if archive_hash != PARENT_SNAPSHOT_ARCHIVE_SHA256:
        raise RuntimeError("parent source snapshot archive hash differs")
    if archive_path.stat().st_size != PARENT_SNAPSHOT_ARCHIVE_BYTES:
        raise RuntimeError("parent source snapshot archive size differs")
    return {
        "role": "parent_formal_source_snapshot",
        "manifest_path": str(PARENT_SNAPSHOT_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": manifest_hash,
        "archive_path": str(PARENT_SNAPSHOT_ARCHIVE.relative_to(ROOT)),
        "archive_sha256": archive_hash,
        "archive_bytes": archive_path.stat().st_size,
        "source_tree_sha256": manifest["source_tree_sha256"],
        "git_revision": manifest.get("git_revision"),
        "relationship": (
            "RQ6 is post-snapshot extension work; active RQ6 source is bound "
            "separately and is not claimed to be contained in the parent archive"
        ),
    }


def runtime_environment() -> dict[str, Any]:
    """Return stable Python/NumPy/Torch and BLAS provenance fields."""

    numpy_configuration = getattr(np.__config__, "CONFIG", {})
    build_dependencies = numpy_configuration.get("Build Dependencies", {})
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "numpy": {
            "version": np.__version__,
            "blas": build_dependencies.get("blas"),
            "lapack": build_dependencies.get("lapack"),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "git_version": torch.version.git_version,
        },
        "platform": platform.platform(),
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


def formal_rng_mapping(config: RQ6ConvergenceConfig) -> dict[str, int]:
    """Enumerate every problem, logged-role, and summary-bootstrap RNG ID."""

    mapping: dict[str, int] = {}
    for problem_index, problem_seed in enumerate(config.problem_seeds):
        mapping[f"problem/{problem_index:03d}/mdp"] = problem_seed
        for replicate in range(config.logged_replicates):
            cot_rng, certification_rng = logged_rng_ids(
                config,
                problem_index,
                replicate,
            )
            prefix = f"problem/{problem_index:03d}/logged/{replicate:02d}"
            mapping[f"{prefix}/cot"] = cot_rng
            mapping[f"{prefix}/certification"] = certification_rng
    mapping["summary/problem_cluster_bootstrap"] = config.bootstrap_rng
    return mapping


def audit_formal_rng_ids(
    config: RQ6ConvergenceConfig,
    *,
    output_dir: Path,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Reject collisions after publishing the full formal RNG mapping."""

    artifact_root = ROOT / "results" if artifact_root is None else artifact_root
    source_root = ROOT if source_root is None else source_root
    mapping = formal_rng_mapping(config)
    formal_ids = set(mapping.values())
    expected_count = (
        config.problem_count
        + 2 * config.problem_count * config.logged_replicates
        + 1
    )
    if len(mapping) != expected_count or len(formal_ids) != expected_count:
        raise RuntimeError("formal RQ6 RNG mapping is not one-to-one and complete")
    if (
        config.problem_count == 100
        and config.logged_replicates == 20
        and expected_count != 4_101
    ):
        raise RuntimeError("formal RQ6 RNG mapping must contain exactly 4101 IDs")

    artifact_ids = _artifact_rng_ids(artifact_root, excluded_root=output_dir)
    excluded_paths = {
        Path(__file__).resolve(),
        (ROOT / "src" / "scpcp" / "rq6_ncal_convergence.py").resolve(),
        (ROOT / "configs" / "rq6_ncal_convergence.yaml").resolve(),
    }
    source_ids = _source_actual_rng_ids(
        source_root,
        excluded_paths=excluded_paths,
    )
    coordinated_external = set().union(
        *(set(values) for values in COORDINATED_EXTERNAL_RESERVATIONS.values())
    )
    prior_ids = artifact_ids | source_ids | coordinated_external
    collisions = {
        label: rng_id for label, rng_id in mapping.items() if rng_id in prior_ids
    }
    audit = {
        "status": "passed_before_launch" if not collisions else "collision",
        "seed_namespace": config.seed_namespace,
        "formal_rng_id_count": len(formal_ids),
        "formal_rng_ids": sorted(formal_ids),
        "formal_rng_id_sha256": _integer_set_sha256(formal_ids),
        "formal_rng_mapping": mapping,
        "formal_rng_mapping_sha256": _json_sha256(mapping),
        "problem_seed_range_inclusive": [
            min(config.problem_seeds),
            max(config.problem_seeds),
        ],
        "logged_rng_range_inclusive": [
            config.logged_rng_start,
            config.logged_rng_start
            + 2 * config.problem_count * config.logged_replicates
            - 1,
        ],
        "bootstrap_rng": config.bootstrap_rng,
        "artifact_rng_id_count": len(artifact_ids),
        "artifact_rng_id_sha256": _integer_set_sha256(artifact_ids),
        "source_actual_use_rng_id_count": len(source_ids),
        "source_actual_use_rng_id_sha256": _integer_set_sha256(source_ids),
        "coordinated_reservations": COORDINATED_RESERVATION_LABELS,
        "coordinated_external_rng_id_count": len(coordinated_external),
        "coordinated_external_rng_id_sha256": _integer_set_sha256(
            coordinated_external
        ),
        "collision_count": len(collisions),
        "collisions": collisions,
        "excluded_output": str(output_dir.resolve()),
        "acknowledged_assignment_source": (
            "coordinated reservation declarations are inventoried separately and "
            "excluded from the source actual-use scan"
        ),
        "source_actual_use_excludes_reservation_declarations": True,
    }
    audit["audit_sha256"] = _json_sha256(audit)
    if collisions:
        raise RuntimeError(f"formal RQ6 RNG IDs collide with prior use: {collisions}")
    return audit


def _run_pending_problems(
    config: RQ6ConvergenceConfig,
    pending: tuple[tuple[int, int], ...],
    *,
    output_dir: Path,
    config_hash: str,
    source_hash: str,
    parent_snapshot: dict[str, Any],
    environment_hash: str,
) -> None:
    if not pending:
        return
    with ProcessPoolExecutor(
        max_workers=config.workers,
        mp_context=get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(
                run_problem,
                config,
                problem_index=problem_index,
                problem_seed=problem_seed,
            ): (problem_index, problem_seed)
            for problem_index, problem_seed in pending
        }
        for future in as_completed(futures):
            problem_index, problem_seed = futures[future]
            result = future.result()
            write_problem_artifact(
                result,
                output_dir,
                config=config,
                problem_index=problem_index,
                problem_seed=problem_seed,
                config_hash=config_hash,
                source_hash=source_hash,
                parent_snapshot=parent_snapshot,
                environment_hash=environment_hash,
            )
            print(f"completed RQ6 problem {problem_seed}", flush=True)


def write_problem_artifact(
    result: dict[str, Any],
    output_dir: Path,
    *,
    config: RQ6ConvergenceConfig,
    problem_index: int,
    problem_seed: int,
    config_hash: str,
    source_hash: str,
    parent_snapshot: dict[str, Any] | None = None,
    environment_hash: str | None = None,
) -> Path:
    """Atomically publish one complete problem cluster."""

    parent_snapshot = (
        validate_parent_snapshot() if parent_snapshot is None else parent_snapshot
    )
    environment_hash = (
        _json_sha256(runtime_environment())
        if environment_hash is None
        else environment_hash
    )
    destination = output_dir / f"problem_{problem_seed}"
    if destination.exists():
        raise FileExistsError(f"problem artifact already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=output_dir))
    try:
        _write_text(
            temporary / "result.json",
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        )
        metadata = {
            "protocol": config.protocol,
            "problem_index": problem_index,
            "problem_seed": problem_seed,
            "seed_namespace": config.seed_namespace,
            "config_sha256": config_hash,
            "source_tree_sha256": source_hash,
            "parent_snapshot_manifest_sha256": parent_snapshot["manifest_sha256"],
            "parent_snapshot_archive_sha256": parent_snapshot["archive_sha256"],
            "parent_source_tree_sha256": parent_snapshot["source_tree_sha256"],
            "runtime_environment_sha256": environment_hash,
            "row_count": len(result.get("rows", [])),
            "result_sha256": _file_sha256(temporary / "result.json"),
        }
        _write_text(
            temporary / "metadata.json",
            json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n",
        )
        complete = {
            "status": "complete",
            "problem_index": problem_index,
            "problem_seed": problem_seed,
            "parent_snapshot_manifest_sha256": parent_snapshot["manifest_sha256"],
            "parent_snapshot_archive_sha256": parent_snapshot["archive_sha256"],
            "parent_source_tree_sha256": parent_snapshot["source_tree_sha256"],
            "runtime_environment_sha256": environment_hash,
            "metadata_sha256": _file_sha256(temporary / "metadata.json"),
        }
        _write_text(
            temporary / "COMPLETE",
            json.dumps(complete, sort_keys=True, allow_nan=False) + "\n",
        )
        for name in PROBLEM_FILES:
            _fsync_file(temporary / name)
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def validate_problem_artifact(
    path: Path,
    *,
    config: RQ6ConvergenceConfig,
    problem_index: int,
    problem_seed: int,
    config_hash: str,
    source_hash: str,
    parent_snapshot: dict[str, Any] | None = None,
    environment_hash: str | None = None,
) -> dict[str, Any]:
    parent_snapshot = (
        validate_parent_snapshot() if parent_snapshot is None else parent_snapshot
    )
    environment_hash = (
        _json_sha256(runtime_environment())
        if environment_hash is None
        else environment_hash
    )
    missing = [name for name in PROBLEM_FILES if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"partial RQ6 problem artifact {path}: missing {missing}")
    metadata = _read_json(path / "metadata.json")
    complete = _read_json(path / "COMPLETE")
    expected_complete = {
        "status": "complete",
        "problem_index": problem_index,
        "problem_seed": problem_seed,
        "parent_snapshot_manifest_sha256": parent_snapshot["manifest_sha256"],
        "parent_snapshot_archive_sha256": parent_snapshot["archive_sha256"],
        "parent_source_tree_sha256": parent_snapshot["source_tree_sha256"],
        "runtime_environment_sha256": environment_hash,
        "metadata_sha256": _file_sha256(path / "metadata.json"),
    }
    if complete != expected_complete:
        raise RuntimeError(f"problem {problem_seed} COMPLETE marker is malformed")
    expected_metadata = {
        "protocol": config.protocol,
        "problem_index": problem_index,
        "problem_seed": problem_seed,
        "seed_namespace": config.seed_namespace,
        "config_sha256": config_hash,
        "source_tree_sha256": source_hash,
        "parent_snapshot_manifest_sha256": parent_snapshot["manifest_sha256"],
        "parent_snapshot_archive_sha256": parent_snapshot["archive_sha256"],
        "parent_source_tree_sha256": parent_snapshot["source_tree_sha256"],
        "runtime_environment_sha256": environment_hash,
        "row_count": config.logged_replicates * len(config.n_calibration),
        "result_sha256": _file_sha256(path / "result.json"),
    }
    if metadata != expected_metadata:
        raise RuntimeError(f"problem {problem_seed} metadata contract differs")
    result = _read_json(path / "result.json")
    _validate_problem_result(
        result,
        config=config,
        problem_index=problem_index,
        problem_seed=problem_seed,
    )
    return result


def _validate_problem_result(
    result: dict[str, Any],
    *,
    config: RQ6ConvergenceConfig,
    problem_index: int,
    problem_seed: int,
) -> None:
    if (
        result.get("problem_seed") != problem_seed
        or result.get("problem_index") != problem_index
        or result.get("mechanism") != "M3_full_feedback"
    ):
        raise RuntimeError(f"problem {problem_seed} result identity differs")
    policy = result.get("policy_contract", {})
    if (
        policy.get("outcome_blind") is not True
        or policy.get("formula")
        != "softmax(log(mu) + lambda * clipped_radius_response * [1,0,-1])"
        or policy.get("reference_radius") != config.reference_radius
        or policy.get("target_reference_state_mean_tv")
        != config.policy_reference_tv
        or not np.isclose(
            policy.get("observed_reference_state_mean_tv", np.nan),
            config.policy_reference_tv,
            atol=1e-12,
            rtol=0.0,
        )
        or not np.isfinite(policy.get("logit_strength", np.nan))
        or policy.get("logit_strength", 0.0) <= 0.0
    ):
        raise RuntimeError(f"problem {problem_seed} policy contract differs")
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != config.logged_replicates * len(
        config.n_calibration
    ):
        raise RuntimeError(f"problem {problem_seed} row count differs")
    seen: set[tuple[int, int]] = set()
    for row in rows:
        replicate = row.get("logged_replicate")
        n_calibration = row.get("n_calibration")
        cell = (replicate, n_calibration)
        if cell in seen:
            raise RuntimeError(f"problem {problem_seed} has duplicate cells")
        seen.add(cell)
        if (
            not isinstance(replicate, int)
            or not 0 <= replicate < config.logged_replicates
            or n_calibration not in config.n_calibration
            or row.get("problem_seed") != problem_seed
            or row.get("problem_index") != problem_index
        ):
            raise RuntimeError(f"problem {problem_seed} cell lies outside the design")
        expected_cot_rng, expected_certification_rng = logged_rng_ids(
            config,
            problem_index,
            replicate,
        )
        if (row.get("cot_rng"), row.get("certification_rng")) != (
            expected_cot_rng,
            expected_certification_rng,
        ):
            raise RuntimeError(f"problem {problem_seed} logged RNG mapping differs")
        if (row.get("n_cot"), row.get("n_certification")) != calibration_role_sizes(
            n_calibration,
            config,
        ):
            raise RuntimeError(f"problem {problem_seed} calibration budget differs")
        track_a = row.get("track_a", {})
        stage_error = np.asarray(
            track_a.get("stagewise_surface_sup_error", []),
            dtype=np.float64,
        )
        stage_ess = np.asarray(
            track_a.get("stagewise_minimum_prefix_ess_fraction", []),
            dtype=np.float64,
        )
        surface_error = float(track_a.get("surface_sup_error", np.nan))
        minimum_ess = float(track_a.get("minimum_prefix_ess_fraction", np.nan))
        prefix_counts = [
            config.grid_size ** (stage + 1) for stage in range(config.horizon)
        ]
        if (
            track_a.get("unique_prefix_counts") != prefix_counts
            or track_a.get("complete_schedule_count")
            != config.grid_size**config.horizon
            or stage_error.shape != (config.horizon,)
            or stage_ess.shape != (config.horizon,)
            or not np.isfinite(stage_error).all()
            or not np.isfinite(stage_ess).all()
            or bool(((stage_error < 0.0) | (stage_error > 1.0)).any())
            or bool(((stage_ess <= 0.0) | (stage_ess > 1.0 + 1e-12)).any())
            or not np.isclose(surface_error, stage_error.max(), atol=1e-14, rtol=0.0)
            or not np.isclose(minimum_ess, stage_ess.min(), atol=1e-14, rtol=0.0)
            or track_a.get("supremum_definition")
            != (
                "max over every stage and every unique q-prefix induced by all "
                f"{config.grid_size**config.horizon} complete fixed-grid schedules"
            )
        ):
            raise RuntimeError(f"problem {problem_seed} Track-A contract differs")
        track_b = row.get("track_b", {})
        grids = np.asarray(track_b.get("stage_grids", []), dtype=np.float64)
        if (
            grids.shape != (config.horizon, config.grid_size)
            or not np.isfinite(grids).all()
            or bool((grids < 0.0).any())
            or bool((np.diff(grids, axis=1) < 0.0).any())
        ):
            raise RuntimeError(f"problem {problem_seed} Track-B grid differs")
        available = track_b.get("selection_available")
        if not isinstance(available, bool):
            raise RuntimeError(f"problem {problem_seed} Track-B availability differs")
        indices = track_b.get("selected_indices")
        if (
            not isinstance(indices, list)
            or any(
                not isinstance(index, int)
                or not 0 <= index < config.grid_size
                for index in indices
            )
            or not isinstance(track_b.get("selected_endpoint"), bool)
            or track_b["selected_endpoint"]
            != any(index in {0, config.grid_size - 1} for index in indices)
        ):
            raise RuntimeError(f"problem {problem_seed} Track-B indices differ")
        selected_count = len(indices)
        estimated_coverage = np.asarray(
            track_b.get("estimated_coverage", []), dtype=np.float64
        )
        estimated_width = np.asarray(
            track_b.get("estimated_normalized_width", []), dtype=np.float64
        )
        selected_ess = np.asarray(
            track_b.get("selected_ess_fraction", []), dtype=np.float64
        )
        if (
            estimated_coverage.shape != (selected_count,)
            or estimated_width.shape != (selected_count,)
            or selected_ess.shape != (selected_count,)
            or not np.isfinite(estimated_coverage).all()
            or not np.isfinite(estimated_width).all()
            or not np.isfinite(selected_ess).all()
            or bool(
                ((estimated_coverage < 0.0) | (estimated_coverage > 1.0)).any()
            )
            or bool((estimated_width < 0.0).any())
            or bool(((selected_ess <= 0.0) | (selected_ess > 1.0 + 1e-12)).any())
        ):
            raise RuntimeError(f"problem {problem_seed} Track-B estimates differ")
        if available:
            coverage = np.asarray(track_b.get("population_coverage"), dtype=np.float64)
            selected_radii = np.asarray(track_b.get("selected_radii"), dtype=np.float64)
            selected_tv = np.asarray(
                track_b.get("selected_policy_reference_state_tv"),
                dtype=np.float64,
            )
            expected_radii = grids[np.arange(config.horizon), np.asarray(indices)]
            if (
                track_b.get("failure_stage") is not None
                or selected_count != config.horizon
                or coverage.shape != (config.horizon,)
                or selected_radii.shape != (config.horizon,)
                or selected_tv.shape != (config.horizon,)
                or not np.isfinite(coverage).all()
                or not np.isfinite(selected_radii).all()
                or not np.isfinite(selected_tv).all()
                or bool(((coverage < 0.0) | (coverage > 1.0)).any())
                or bool(((selected_tv < 0.0) | (selected_tv > 1.0)).any())
                or not np.allclose(
                    selected_radii,
                    expected_radii,
                    atol=1e-7,
                    rtol=1e-7,
                )
                or not np.isclose(
                    track_b.get("population_worst_stage_coverage", np.nan),
                    coverage.min(),
                    atol=1e-14,
                    rtol=0.0,
                )
                or not np.isfinite(
                    track_b.get("population_mean_normalized_width", np.nan)
                )
                or track_b.get("population_mean_normalized_width", -1.0) < 0.0
            ):
                raise RuntimeError(f"problem {problem_seed} Track-B coverage differs")
        else:
            failure_stage = track_b.get("failure_stage")
            if (
                not isinstance(failure_stage, int)
                or not 0 <= failure_stage < config.horizon
                or selected_count != failure_stage
                or any(
                    track_b.get(field) is not None
                    for field in (
                        "selected_radii",
                        "population_coverage",
                        "population_worst_stage_coverage",
                        "population_mean_normalized_width",
                        "selected_policy_reference_state_tv",
                    )
                )
            ):
                raise RuntimeError(f"problem {problem_seed} Track-B failure differs")
    expected_cells = {
        (replicate, n_calibration)
        for replicate in range(config.logged_replicates)
        for n_calibration in config.n_calibration
    }
    if seen != expected_cells:
        raise RuntimeError(f"problem {problem_seed} replicate/n grid differs")


def validate_resume(
    output_dir: Path,
    *,
    config: RQ6ConvergenceConfig,
    manifest: dict[str, Any],
) -> set[int]:
    if not output_dir.is_dir():
        raise FileNotFoundError("RQ6 resume requires an existing output directory")
    for name in ("config.yaml", "manifest.json", "runtime_preflight.json"):
        if not (output_dir / name).is_file():
            raise RuntimeError(f"RQ6 resume is missing {name}")
    if _read_json(output_dir / "manifest.json") != manifest:
        raise RuntimeError("RQ6 resume manifest differs from the active protocol")
    parent_snapshot = validate_parent_snapshot()
    if manifest.get("parent_snapshot") != parent_snapshot:
        raise RuntimeError("RQ6 resume parent snapshot binding differs")
    environment = runtime_environment()
    environment_hash = _json_sha256(environment)
    if (
        manifest.get("runtime_environment") != environment
        or manifest.get("runtime_environment_sha256") != environment_hash
    ):
        raise RuntimeError("RQ6 resume runtime environment differs")
    try:
        stored_config = yaml.safe_load((output_dir / "config.yaml").read_text())
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError("RQ6 stored config is unreadable") from error
    if stored_config != config.to_dict():
        raise RuntimeError("RQ6 resume config differs from the active config")
    preflight = _read_json(output_dir / "runtime_preflight.json")
    if (
        preflight.get("status") != "passed"
        or preflight.get("config_sha256") != manifest["config_sha256"]
        or preflight.get("source_tree_sha256") != manifest["source_tree_sha256"]
        or preflight.get("parent_snapshot") != parent_snapshot
        or preflight.get("runtime_environment_sha256") != environment_hash
    ):
        raise RuntimeError("RQ6 runtime preflight provenance differs")
    _reject_unknown_or_temporary_problem_paths(output_dir, config)
    completed: set[int] = set()
    for problem_index, problem_seed in enumerate(config.problem_seeds):
        path = output_dir / f"problem_{problem_seed}"
        if not path.exists():
            continue
        validate_problem_artifact(
            path,
            config=config,
            problem_index=problem_index,
            problem_seed=problem_seed,
            config_hash=manifest["config_sha256"],
            source_hash=manifest["source_tree_sha256"],
            parent_snapshot=parent_snapshot,
            environment_hash=environment_hash,
        )
        completed.add(problem_seed)
    return completed


def validate_complete_bundle(
    output_dir: Path,
    *,
    config: RQ6ConvergenceConfig,
    manifest: dict[str, Any],
) -> None:
    if manifest.get("parent_snapshot") != validate_parent_snapshot():
        raise RuntimeError("RQ6 complete bundle parent snapshot differs")
    environment = runtime_environment()
    if (
        manifest.get("runtime_environment") != environment
        or manifest.get("runtime_environment_sha256") != _json_sha256(environment)
    ):
        raise RuntimeError("RQ6 complete bundle runtime environment differs")
    complete = _read_json(output_dir / "COMPLETE")
    expected = {
        "status": "complete",
        "protocol": config.protocol,
        "config_sha256": manifest["config_sha256"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "parent_snapshot_manifest_sha256": manifest["parent_snapshot"][
            "manifest_sha256"
        ],
        "parent_snapshot_archive_sha256": manifest["parent_snapshot"][
            "archive_sha256"
        ],
        "parent_source_tree_sha256": manifest["parent_snapshot"][
            "source_tree_sha256"
        ],
        "runtime_environment_sha256": manifest["runtime_environment_sha256"],
        "manifest_sha256": _file_sha256(output_dir / "manifest.json"),
        "runtime_preflight_sha256": _file_sha256(output_dir / "runtime_preflight.json"),
        "summary_sha256": _file_sha256(output_dir / "summary.json"),
        "artifact_manifest_sha256": _file_sha256(
            output_dir / "artifact_manifest.json"
        ),
    }
    if complete != expected:
        raise RuntimeError("RQ6 root COMPLETE hash contract differs")
    artifact_manifest = _read_json(output_dir / "artifact_manifest.json")
    if artifact_manifest != _artifact_manifest(output_dir, config):
        raise RuntimeError("RQ6 per-problem artifact manifest differs")
    summary = _read_json(output_dir / "summary.json")
    if (
        summary.get("status") != "complete"
        or summary.get("protocol") != config.protocol
        or summary.get("config_sha256") != manifest["config_sha256"]
        or summary.get("source_tree_sha256") != manifest["source_tree_sha256"]
        or summary.get("parent_snapshot") != manifest["parent_snapshot"]
        or summary.get("runtime_environment_sha256")
        != manifest["runtime_environment_sha256"]
        or summary.get("formal_problem_count") != config.problem_count
        or summary.get("design", {}).get("n_calibration")
        != list(config.n_calibration)
        or summary.get("design", {}).get("bootstrap_rng") != config.bootstrap_rng
    ):
        raise RuntimeError("RQ6 complete summary contract differs")


def _root_manifest(
    config: RQ6ConvergenceConfig,
    *,
    config_hash: str,
    source_hash: str,
    rng_audit: dict[str, Any],
    parent_snapshot: dict[str, Any],
    environment: dict[str, Any],
    launch_argv: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": config.protocol,
        "study": "rq6_n_calibration_convergence",
        "canonical_selector_unchanged": True,
        "mechanism": "M3_full_feedback_with_outcome_blind_radius_policy",
        "config_sha256": config_hash,
        "source_tree_sha256": source_hash,
        "git_revision": git_revision(),
        "parent_snapshot": parent_snapshot,
        "runtime_environment": environment,
        "runtime_environment_sha256": _json_sha256(environment),
        "launch_argv": launch_argv,
        "seed_namespace": config.seed_namespace,
        "formal_rng_collision_audit": rng_audit,
        "pairing": (
            "one maximum independent D_COT/D_cert pool per problem/logged resample; "
            "every smaller n is a role-specific prefix of that same pool"
        ),
        "track_a": (
            "fixed population K=7 grid; stage-t supremum over all K**(t+1) unique "
            "q-prefixes, exactly equivalent to all 2401 schedule-stage cells"
        ),
        "track_b": (
            "D_COT empirical grid plus unmodified committed-prefix selector; exact "
            "population evaluation of each selected schedule"
        ),
        "inference": (
            "shared percentile bootstrap over problem clusters; within-problem logged "
            "resamples reported separately"
        ),
        "claim_boundary": (
            "diagnostic of asymptotic per-step marginal calibration convergence; "
            "not finite-sample distribution-free, PAC, data-conditional, or universal SOTA"
        ),
    }


def _artifact_manifest(output_dir: Path, config: RQ6ConvergenceConfig) -> dict[str, Any]:
    artifacts = {}
    for problem_seed in config.problem_seeds:
        path = output_dir / f"problem_{problem_seed}"
        artifacts[str(problem_seed)] = {
            name: {
                "sha256": _file_sha256(path / name),
                "bytes": (path / name).stat().st_size,
            }
            for name in PROBLEM_FILES
        }
    return {
        "protocol": config.protocol,
        "problem_count": len(artifacts),
        "problem_artifacts": artifacts,
    }


def _reject_unknown_or_temporary_problem_paths(
    output_dir: Path,
    config: RQ6ConvergenceConfig,
) -> None:
    expected = {f"problem_{seed}" for seed in config.problem_seeds}
    for path in output_dir.iterdir():
        if path.name.startswith(".problem_"):
            raise RuntimeError(f"abandoned temporary RQ6 artifact exists: {path.name}")
        if path.name.startswith("problem_") and path.name not in expected:
            raise RuntimeError(f"unknown RQ6 problem artifact exists: {path.name}")


def _write_status(
    output_dir: Path,
    config: RQ6ConvergenceConfig,
    *,
    completed: set[int],
    status: str,
    error: BaseException | None = None,
    invocation_argv: list[str] | None = None,
) -> None:
    _atomic_write_json(
        output_dir / "study_status.json",
        {
            "status": status,
            "expected_problem_seeds": list(config.problem_seeds),
            "completed_problem_seeds": sorted(completed),
            "missing_problem_seeds": sorted(set(config.problem_seeds) - completed),
            "error": None if error is None else f"{type(error).__name__}: {error}",
            "invocation_argv": list(sys.argv) if invocation_argv is None else invocation_argv,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _artifact_rng_ids(root: Path, *, excluded_root: Path) -> set[int]:
    values: set[int] = set()
    if not root.exists():
        return values
    excluded = excluded_root.resolve()
    provenance_names = {
        "metadata.json",
        "study_metadata.json",
        "manifest.json",
        "summary.json",
        "config.yaml",
    }
    for path in root.rglob("*"):
        resolved = path.resolve()
        if _is_relative_to(resolved, excluded):
            continue
        match = ARTIFACT_ID.fullmatch(path.name)
        if match:
            values.add(int(match.group(1)))
        if not path.is_file() or path.name not in provenance_names:
            continue
        try:
            payload = (
                yaml.safe_load(path.read_text())
                if path.suffix in {".yaml", ".yml"}
                else json.loads(path.read_text())
            )
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
            raise RuntimeError(f"cannot audit RNG-bearing artifact {path}") from error
        _collect_named_rng_values(payload, values)
    return values


def _source_actual_rng_ids(
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
            resolved = path.resolve()
            if resolved in excluded_paths or not path.is_file():
                continue
            path_values: set[int] = set()
            if path.suffix == ".py":
                _collect_python_rng_assignments(path, path_values)
            elif path.suffix in {".yaml", ".yml"}:
                try:
                    payload = yaml.safe_load(path.read_text())
                except (OSError, yaml.YAMLError) as error:
                    raise RuntimeError(f"cannot audit source RNG declarations in {path}") from error
                _collect_named_rng_values(payload, path_values)
            values.update(path_values)
    return values


def _collect_python_rng_assignments(path: Path, values: set[int]) -> None:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise RuntimeError(f"cannot parse RNG declarations in {path}") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [name for target in node.targets for name in _target_names(target)]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            names = list(_target_names(node.target))
            value = node.value
        else:
            continue
        if value is None or not any(RNG_KEY.search(name) for name in names):
            continue
        if any(RESERVATION_KEY.search(name) for name in names):
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
        if node.func.id in {"tuple", "list", "set", "frozenset"} and len(node.args) == 1:
            return _literal_rng_expression(node.args[0])
    return None


def _literal_integer(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _literal_integer(node.operand)
        return None if operand is None else -operand
    return None


def _collect_named_rng_values(value: object, output: set[int], key_path: str = "") -> None:
    if RESERVATION_KEY.search(key_path):
        return
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{key_path}.{key}" if key_path else str(key)
            _collect_named_rng_values(child, output, path)
        return
    if isinstance(value, list):
        for child in value:
            _collect_named_rng_values(child, output, key_path)
        return
    if RNG_KEY.search(key_path) and isinstance(value, int) and not isinstance(value, bool):
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


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )


def _atomic_write_text(path: Path, content: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}-",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unreadable JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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
