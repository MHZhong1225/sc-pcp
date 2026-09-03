from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch
import yaml

from scpcp.coverage import fixed_q_grid
from scpcp.exact_finite_mdp import (
    ESTIMATOR_NAMES,
    LoggedTrajectories,
    enumerate_schedules,
    exact_population_surfaces,
    hajek_surface_estimates,
)
import scpcp.rq6_ncal_convergence as rq6
from scpcp.rq6_ncal_convergence import (
    RQ6ConvergenceConfig,
    build_outcome_blind_m3,
    calibration_role_sizes,
    evaluate_track_a_nested_prefixes,
    evaluate_track_b_canonical_selector,
    logged_rng_ids,
    simulate_nested_role_pools,
    summarize_problem_results,
)


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_rq6_ncal_convergence.py"
    name = "test_run_rq6_ncal_convergence"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _small_config(**overrides) -> RQ6ConvergenceConfig:
    base = RQ6ConvergenceConfig.from_yaml(ROOT / "configs" / "rq6_ncal_convergence.yaml")
    values = {
        "horizon": 2,
        "grid_size": 3,
        "n_calibration": (30, 60),
        "problem_seed_start": 701,
        "problem_count": 1,
        "logged_replicates": 1,
        "logged_rng_start": 801,
        "bootstrap_rng": 901,
        "bootstrap_resamples": 100,
        "surface_chunk_size": 16,
        "workers": 1,
        "seed_namespace": "rq6_test_only",
    }
    values.update(overrides)
    config = replace(base, **values)
    config.validate()
    return config


def _small_problem(config: RQ6ConvergenceConfig):
    mechanism, policy = build_outcome_blind_m3(config, problem_seed=701)
    schedules = enumerate_schedules(
        config.exact_config(logged_trajectories=1, seed=701)
    )
    population, _ = exact_population_surfaces(mechanism, schedules)
    true_surface = population[ESTIMATOR_NAMES.index("full_prefix")]
    cot_pool, certification_pool = simulate_nested_role_pools(
        config,
        mechanism,
        cot_rng=811,
        certification_rng=812,
        problem_seed=701,
    )
    return mechanism, policy, schedules, true_surface, cot_pool, certification_pool


def test_frozen_protocol_and_full_rng_mapping_are_exact(tmp_path: Path) -> None:
    runner = _load_runner()
    config = RQ6ConvergenceConfig.from_yaml(
        ROOT / "configs" / "rq6_ncal_convergence.yaml"
    )

    assert config.problem_seeds == tuple(range(97_000, 97_100))
    assert [calibration_role_sizes(n, config) for n in config.n_calibration] == [
        (83, 167),
        (167, 333),
        (333, 667),
        (667, 1_333),
        (1_667, 3_333),
        (3_333, 6_667),
    ]
    mapping = runner.formal_rng_mapping(config)
    assert len(mapping) == len(set(mapping.values())) == 4_101
    assert mapping["problem/000/mdp"] == 97_000
    assert mapping["problem/099/mdp"] == 97_099
    assert mapping["problem/000/logged/00/cot"] == 97_100_000
    assert mapping["problem/099/logged/19/certification"] == 97_103_999
    assert mapping["summary/problem_cluster_bootstrap"] == 97_900_000

    artifact_root = tmp_path / "artifacts"
    source_root = tmp_path / "source"
    artifact_root.mkdir()
    (source_root / "scripts").mkdir(parents=True)
    (source_root / "scripts" / "coordinator.py").write_text(
        "EXTERNAL_SEED_RESERVATIONS = {\n"
        "    'rq6_calibration_convergence': range(97_000, 98_000),\n"
        "    'propensity_robustness': range(98_000, 99_000),\n"
        "}\n"
    )
    audit = runner.audit_formal_rng_ids(
        config,
        output_dir=tmp_path / "new_output",
        artifact_root=artifact_root,
        source_root=source_root,
    )
    assert audit["collision_count"] == 0
    assert audit["formal_rng_id_count"] == 4_101
    assert set(audit["coordinated_reservations"]) == {
        "exact_finite_mdp",
        "controlled_six_method",
        "orthogonal_copula",
        "rq5_horizon_overlap",
        "rq6_ncal_convergence",
        "propensity_robustness",
        "strict_split_audit",
        "future_score_robustness",
    }

    (source_root / "scripts" / "prior.py").write_text("OLD_RNG = (97100000,)\n")
    with pytest.raises(RuntimeError, match="collide"):
        runner.audit_formal_rng_ids(
            config,
            output_dir=tmp_path / "new_output",
            artifact_root=artifact_root,
            source_root=source_root,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("alpha", 0.11),
        ("radius_minimum", 1.30),
        ("q_quantile_maximum", 0.995),
        ("policy_reference_tv", 0.06),
        ("cot_role_parts", 2),
        ("bootstrap_resamples", 9_999),
    ),
)
def test_formal_protocol_locks_every_scientific_field(field: str, value) -> None:
    config = RQ6ConvergenceConfig.from_yaml(
        ROOT / "configs" / "rq6_ncal_convergence.yaml"
    )
    with pytest.raises(ValueError, match="frozen formal protocol"):
        replace(config, **{field: value}).assert_frozen_protocol()

    runtime_only = config.with_runtime_overrides(
        output_dir=Path("another-output"),
        workers=2,
        surface_chunk_size=64,
    )
    runtime_only.assert_frozen_protocol()


def test_parent_snapshot_and_runtime_environment_are_content_bound(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    binding = runner.validate_parent_snapshot()
    assert binding["manifest_sha256"] == (
        "e6a1bba7f3be47d39357f212824e7720262e7d5212a14628e3b8981088c64e24"
    )
    assert binding["archive_sha256"] == (
        "2116b9929240d8a25092f3c9015c362957bc963532930e5e720dc7ec78b2ea0b"
    )
    assert binding["source_tree_sha256"] == (
        "7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643"
    )
    environment = runner.runtime_environment()
    assert environment["python"]["version"]
    assert environment["numpy"]["version"]
    assert environment["numpy"]["blas"]
    assert environment["torch"]["version"]

    tampered_archive = tmp_path / "snapshot.tar.gz"
    shutil.copyfile(runner.PARENT_SNAPSHOT_ARCHIVE, tampered_archive)
    with tampered_archive.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="archive hash differs"):
        runner.validate_parent_snapshot(archive_path=tampered_archive)


def test_outcome_blind_policy_is_behavior_anchored_and_has_reference_tv() -> None:
    config = _small_config()
    mechanism, policy = build_outcome_blind_m3(config, problem_seed=701)
    states = np.arange(config.state_count)
    endpoints = policy.numpy_probabilities(
        states,
        np.asarray([config.radius_minimum, config.reference_radius]),
    )

    np.testing.assert_allclose(
        endpoints[:, 0],
        mechanism.behavior_probabilities,
        atol=1e-14,
        rtol=0.0,
    )
    assert policy.mean_state_tv(config.reference_radius) == pytest.approx(0.05, abs=1e-12)
    assert policy.behavior_probabilities is mechanism.behavior_probabilities
    assert not hasattr(policy, "outcome_means")
    assert not hasattr(policy, "predictor_means")


def test_unique_prefix_track_matches_all_schedule_hajek_surface() -> None:
    config = _small_config()
    mechanism, _policy, schedules, true_surface, cot_pool, certification_pool = (
        _small_problem(config)
    )
    role_sizes = tuple(calibration_role_sizes(n, config) for n in config.n_calibration)
    track = evaluate_track_a_nested_prefixes(
        mechanism,
        schedules,
        true_surface,
        cot_pool,
        certification_pool,
        n_calibration=config.n_calibration,
        role_sizes=role_sizes,
    )

    maximum_cot = len(cot_pool.states)
    for n_calibration, (cot_size, certification_size) in zip(
        config.n_calibration,
        role_sizes,
        strict=True,
    ):
        logged = LoggedTrajectories(
            states=np.concatenate(
                (
                    cot_pool.states[:cot_size],
                    certification_pool.states[:certification_size],
                )
            ),
            actions=np.concatenate(
                (
                    cot_pool.actions[:cot_size],
                    certification_pool.actions[:certification_size],
                )
            ),
            scores=np.concatenate(
                (
                    cot_pool.scores[:cot_size],
                    certification_pool.scores[:certification_size],
                )
            ),
        )
        estimates, ess = hajek_surface_estimates(
            mechanism,
            logged,
            schedules,
            chunk_size=16,
        )
        full_prefix = ESTIMATOR_NAMES.index("full_prefix")
        expected_sup = np.abs(estimates[full_prefix] - true_surface).max()
        expected_ess = ess[full_prefix].min()
        assert track[n_calibration]["surface_sup_error"] == pytest.approx(
            expected_sup,
            abs=1e-12,
        )
        assert track[n_calibration]["minimum_prefix_ess_fraction"] == pytest.approx(
            expected_ess,
            abs=1e-12,
        )
        assert track[n_calibration]["unique_prefix_counts"] == [3, 9]
        assert track[n_calibration]["complete_schedule_count"] == 9
    assert maximum_cot == calibration_role_sizes(60, config)[0]


def test_track_b_uses_dcot_grid_and_unmodified_canonical_selector(monkeypatch) -> None:
    config = _small_config()
    mechanism, policy, _schedules, _surface, cot_pool, certification_pool = (
        _small_problem(config)
    )
    original = rq6.select_marginal_prefix_schedule
    captured = {}

    def recording_selector(batch, scores, **kwargs):
        captured["n"] = batch.n
        captured["grids"] = kwargs["stage_grids"].clone()
        return original(batch, scores, **kwargs)

    monkeypatch.setattr(rq6, "select_marginal_prefix_schedule", recording_selector)
    result = evaluate_track_b_canonical_selector(
        config,
        mechanism,
        policy,
        cot_pool,
        certification_pool,
        n_calibration=60,
    )

    cot_size, _ = calibration_role_sizes(60, config)
    expected = torch.stack(
        [
            fixed_q_grid(
                torch.from_numpy(cot_pool.scores[:cot_size, stage]).to(
                    torch.get_default_dtype()
                ),
                size=config.grid_size,
                lower_quantile=config.q_quantile_minimum,
                upper_quantile=config.q_quantile_maximum,
            )
            for stage in range(config.horizon)
        ]
    )
    assert captured["n"] == 60
    torch.testing.assert_close(captured["grids"], expected)
    assert np.asarray(result["stage_grids"]).shape == (2, 3)
    assert isinstance(result["selection_available"], bool)


def test_summary_bootstraps_problem_clusters_and_reports_within_problem_sd() -> None:
    config = _small_config(
        n_calibration=(30,),
        problem_count=3,
        logged_replicates=2,
        bootstrap_resamples=200,
    )
    problem_results = []
    coverages = (
        ((0.88, 0.91), (0.90, 0.92)),
        ((0.89, 0.93), None),
        ((0.91, 0.94), (0.92, 0.95)),
    )
    for problem_index in range(3):
        rows = []
        for replicate in range(2):
            coverage = coverages[problem_index][replicate]
            available = coverage is not None
            rows.append(
                {
                    "logged_replicate": replicate,
                    "n_calibration": 30,
                    "track_a": {
                        "surface_sup_error": 0.10 + 0.01 * problem_index + 0.005 * replicate,
                        "minimum_prefix_ess_fraction": 0.7,
                    },
                    "track_b": {
                        "selection_available": available,
                        "selected_endpoint": replicate == 0,
                        "population_coverage": None if not available else list(coverage),
                        "estimated_coverage": None if not available else [0.90, 0.91],
                        "population_mean_normalized_width": None if not available else 4.0,
                        "selected_ess_fraction": [] if not available else [0.7, 0.6],
                    },
                }
            )
        problem_results.append(
            {
                "problem_index": problem_index,
                "problem_seed": 701 + problem_index,
                "rows": rows,
            }
        )

    summary = summarize_problem_results(problem_results, config)
    track_b = summary["by_n_calibration"]["30"][
        "track_b_canonical_empirical_grid"
    ]
    assert summary["design"]["bootstrap_unit"] == "fixed MDP problem instance"
    assert summary["design"]["shared_bootstrap_indices_across_n_and_tracks"] is True
    assert track_b["selection_availability_rate"] == pytest.approx(5 / 6)
    assert track_b["population_stage_coverage_conditional_on_selection"] == pytest.approx(
        [0.90, 0.93]
    )
    assert track_b["population_wsc_conditional_on_selection"] == pytest.approx(0.90)
    assert track_b["population_mean_normalized_width_conditional_on_selection"] == 4.0
    assert track_b["within_problem_logged_variability"]["availability_sd"][
        "mean_sd"
    ] > 0.0


def test_problem_artifact_is_atomic_and_hash_bound(tmp_path: Path) -> None:
    runner = _load_runner()
    config = RQ6ConvergenceConfig.from_yaml(
        ROOT / "configs" / "rq6_ncal_convergence.yaml"
    )
    result = _formal_shape_fake_problem(config, problem_index=0, problem_seed=97_000)
    config_hash = runner._json_sha256(config.to_dict())
    source_hash = "source"

    path = runner.write_problem_artifact(
        result,
        tmp_path,
        config=config,
        problem_index=0,
        problem_seed=97_000,
        config_hash=config_hash,
        source_hash=source_hash,
    )
    observed = runner.validate_problem_artifact(
        path,
        config=config,
        problem_index=0,
        problem_seed=97_000,
        config_hash=config_hash,
        source_hash=source_hash,
    )
    assert observed["problem_seed"] == 97_000

    payload = json.loads((path / "result.json").read_text())
    payload["rows"][0]["track_a"]["surface_sup_error"] = 9.0
    (path / "result.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="metadata contract differs"):
        runner.validate_problem_artifact(
            path,
            config=config,
            problem_index=0,
            problem_seed=97_000,
            config_hash=config_hash,
            source_hash=source_hash,
        )


def test_real_engineering_problem_satisfies_strict_result_validator() -> None:
    runner = _load_runner()
    config = _small_config(
        horizon=4,
        grid_size=7,
        n_calibration=(30,),
        problem_count=1,
        logged_replicates=1,
    )
    result = rq6.run_problem(config, problem_index=0, problem_seed=701)

    runner._validate_problem_result(
        result,
        config=config,
        problem_index=0,
        problem_seed=701,
    )


def test_resume_validates_config_preflight_and_completed_problem(tmp_path: Path) -> None:
    runner = _load_runner()
    config = _small_config(
        horizon=4,
        grid_size=7,
        n_calibration=(30,),
        problem_count=1,
        logged_replicates=1,
        output_dir=tmp_path,
    )
    config_hash = runner._json_sha256(config.to_dict())
    source_hash = "source"
    parent_snapshot = runner.validate_parent_snapshot()
    environment = runner.runtime_environment()
    environment_hash = runner._json_sha256(environment)
    manifest = {
        "config_sha256": config_hash,
        "source_tree_sha256": source_hash,
        "parent_snapshot": parent_snapshot,
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_hash,
        "launch_argv": ["run_rq6_ncal_convergence.py"],
    }
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False)
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "runtime_preflight.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "config_sha256": config_hash,
                "source_tree_sha256": source_hash,
                "parent_snapshot": parent_snapshot,
                "runtime_environment_sha256": environment_hash,
            }
        )
    )
    result = _formal_shape_fake_problem(config, problem_index=0, problem_seed=701)
    runner.write_problem_artifact(
        result,
        tmp_path,
        config=config,
        problem_index=0,
        problem_seed=701,
        config_hash=config_hash,
        source_hash=source_hash,
    )

    assert runner.validate_resume(tmp_path, config=config, manifest=manifest) == {701}

    summary = {
        "status": "complete",
        "protocol": config.protocol,
        "config_sha256": config_hash,
        "source_tree_sha256": source_hash,
        "parent_snapshot": parent_snapshot,
        "runtime_environment_sha256": environment_hash,
        "formal_problem_count": 1,
        "design": {
            "n_calibration": [30],
            "bootstrap_rng": config.bootstrap_rng,
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    artifact_manifest = runner._artifact_manifest(tmp_path, config)
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(artifact_manifest))
    complete = {
        "status": "complete",
        "protocol": config.protocol,
        "config_sha256": config_hash,
        "source_tree_sha256": source_hash,
        "parent_snapshot_manifest_sha256": parent_snapshot["manifest_sha256"],
        "parent_snapshot_archive_sha256": parent_snapshot["archive_sha256"],
        "parent_source_tree_sha256": parent_snapshot["source_tree_sha256"],
        "runtime_environment_sha256": environment_hash,
        "manifest_sha256": runner._file_sha256(tmp_path / "manifest.json"),
        "runtime_preflight_sha256": runner._file_sha256(
            tmp_path / "runtime_preflight.json"
        ),
        "summary_sha256": runner._file_sha256(tmp_path / "summary.json"),
        "artifact_manifest_sha256": runner._file_sha256(
            tmp_path / "artifact_manifest.json"
        ),
    }
    (tmp_path / "COMPLETE").write_text(json.dumps(complete))
    runner.validate_complete_bundle(tmp_path, config=config, manifest=manifest)

    stored = yaml.safe_load((tmp_path / "config.yaml").read_text())
    stored["alpha"] = 0.2
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(stored))
    with pytest.raises(RuntimeError, match="config differs"):
        runner.validate_resume(tmp_path, config=config, manifest=manifest)


def _formal_shape_fake_problem(
    config: RQ6ConvergenceConfig,
    *,
    problem_index: int,
    problem_seed: int,
) -> dict:
    rows = []
    for replicate in range(config.logged_replicates):
        cot_rng, certification_rng = logged_rng_ids(config, problem_index, replicate)
        for n_calibration in config.n_calibration:
            n_cot, n_certification = calibration_role_sizes(n_calibration, config)
            rows.append(
                {
                    "problem_seed": problem_seed,
                    "problem_index": problem_index,
                    "logged_replicate": replicate,
                    "cot_rng": cot_rng,
                    "certification_rng": certification_rng,
                    "n_calibration": n_calibration,
                    "n_cot": n_cot,
                    "n_certification": n_certification,
                    "track_a": {
                        "surface_sup_error": 0.1,
                        "stagewise_surface_sup_error": [0.1] * config.horizon,
                        "minimum_prefix_ess_fraction": 0.5,
                        "stagewise_minimum_prefix_ess_fraction": [0.5] * config.horizon,
                        "unique_prefix_counts": [7, 49, 343, 2_401],
                        "complete_schedule_count": 2_401,
                        "supremum_definition": (
                            "max over every stage and every unique q-prefix induced "
                            "by all 2401 complete fixed-grid schedules"
                        ),
                    },
                    "track_b": {
                        "selection_available": False,
                        "failure_stage": 0,
                        "selected_endpoint": False,
                        "selected_indices": [],
                        "stage_grids": [
                            [1.0] * config.grid_size for _ in range(config.horizon)
                        ],
                        "estimated_coverage": [],
                        "estimated_normalized_width": [],
                        "selected_ess_fraction": [],
                        "selected_radii": None,
                        "population_coverage": None,
                        "population_worst_stage_coverage": None,
                        "population_mean_normalized_width": None,
                        "selected_policy_reference_state_tv": None,
                    },
                }
            )
    return {
        "problem_seed": problem_seed,
        "problem_index": problem_index,
        "mechanism": "M3_full_feedback",
        "policy_contract": {
            "outcome_blind": True,
            "formula": (
                "softmax(log(mu) + lambda * clipped_radius_response * [1,0,-1])"
            ),
            "reference_radius": config.reference_radius,
            "target_reference_state_mean_tv": config.policy_reference_tv,
            "observed_reference_state_mean_tv": 0.05,
            "logit_strength": 1.0,
        },
        "rows": rows,
    }
