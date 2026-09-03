from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

import scpcp.horizon_overlap as horizon_overlap
from scpcp.horizon_overlap import (
    audit_policy_design,
    mix_target_policy,
    run_horizon_overlap_instance,
    solve_policy_mixing_strength,
)
from scpcp.horizon_overlap_config import (
    METHOD_NAMES,
    HorizonOverlapConfig,
    horizon_overlap_seed_collision_audit,
)
from scpcp.phase0_search import analytic_schedule_metrics


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_horizon_overlap.py"
    name = "test_run_horizon_overlap"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_config(tmp_path: Path | None = None) -> HorizonOverlapConfig:
    return replace(
        HorizonOverlapConfig(),
        horizons=(2,),
        alpha=0.20,
        calibration_trajectories=256,
        nominal_policy_tvs=(0.0,),
        instances=1,
        design_seed_start=110_100,
        design_seed_count=2,
        problem_seed_start=110_200,
        logging_seed_start=110_300,
        bootstrap_seed=110_400,
        bootstrap_resamples=100,
        output_dir=Path("tiny") if tmp_path is None else tmp_path / "tiny",
    )


def _write_config(path: Path, config: HorizonOverlapConfig) -> None:
    payload = config.to_dict()
    payload.pop("seed_namespace")
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_frozen_config_and_collision_audit_enumerate_every_rng_id() -> None:
    config = HorizonOverlapConfig.from_yaml(ROOT / "configs" / "horizon_overlap.yaml")
    audit = horizon_overlap_seed_collision_audit(config)

    assert config.horizons == (2, 4, 8, 12, 20)
    assert config.nominal_policy_tvs == (0.0, 0.025, 0.05, 0.10, 0.15)
    assert config.grid_size == 7
    assert config.calibration_trajectories == 3_000
    assert config.instances == 200
    assert config.bootstrap_resamples == 10_000
    assert config.bootstrap_seed == 96_400
    assert audit["streams"]["policy_design"] == list(range(95_900, 96_000))
    assert audit["streams"]["problem"] == list(range(96_000, 96_200))
    assert audit["streams"]["logging"] == list(range(96_200, 96_400))
    assert audit["streams"]["summary_bootstrap"] == [96_400]
    assert audit["all_rng_ids"] == list(range(95_900, 96_401))
    assert audit["rng_id_count"] == 501
    assert audit["collision"] is False
    assert {
        "rq6_calibration_convergence",
        "propensity_robustness",
        "strict_split_audit",
        "score_robustness",
    }.issubset(audit["external_reservations"])

    with pytest.raises(ValueError, match="RNG IDs collide"):
        replace(config, logging_seed_start=96_000).validate()


def test_policy_tv_solve_is_analytic_and_fails_if_unattainable() -> None:
    behavior = np.asarray([[0.6, 0.4], [0.3, 0.7]])
    base = np.asarray([[0.2, 0.8], [0.7, 0.3]])
    solution = solve_policy_mixing_strength(
        behavior,
        base,
        nominal_tv=0.10,
    )
    mixed = mix_target_policy(
        behavior,
        base,
        mixing_strength=solution.mixing_strength,
    )

    assert solution.base_reference_tv == pytest.approx(0.4)
    assert solution.mixing_strength == pytest.approx(0.25)
    assert solution.realized_reference_tv == pytest.approx(0.10)
    assert mixed.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])

    with pytest.raises(RuntimeError, match="unattainable"):
        solve_policy_mixing_strength(behavior, base, nominal_tv=0.41)


def test_policy_preflight_uses_only_independent_design_seeds() -> None:
    config = _tiny_config()

    audit = audit_policy_design(config)

    assert audit["status"] == "pass"
    assert audit["outcome_blind"] is True
    assert audit["policy_probabilities_only"] is True
    assert audit["generated_scores"] is False
    assert audit["inspected_coverage"] is False
    assert audit["design_seed_ids"] == [110_100, 110_101]
    assert audit["formal_problem_seed_ids"] == [110_200]
    assert audit[
        "all_formal_problem_seeds_checked_before_logged_score_generation"
    ] is True
    assert audit["formal_problem_bank_attainability"]["status"] == "pass"
    assert audit["base_reference_tv"]["minimum"] >= 0.15


def test_all_problem_policies_are_checked_before_any_instance_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_tiny_config(), instances=3, design_seed_count=1)
    last_problem_seed = config.problem_seeds[-1]
    instance_started = False

    monkeypatch.setattr(
        horizon_overlap,
        "_build_m3_base",
        lambda _config, *, problem_seed: problem_seed,
    )
    monkeypatch.setattr(
        horizon_overlap,
        "_base_reference_tv",
        lambda _config, mechanism: 0.10
        if mechanism == last_problem_seed
        else 0.20,
    )

    def fail_if_instance_starts(*args, **kwargs):
        nonlocal instance_started
        instance_started = True
        raise AssertionError("no logged instance may start before the whole policy audit")

    monkeypatch.setattr(
        horizon_overlap,
        "run_horizon_overlap_instance",
        fail_if_instance_starts,
    )

    with pytest.raises(RuntimeError, match="formal problem preflight failed"):
        horizon_overlap.run_horizon_overlap_study(config)
    assert instance_started is False


def test_formal_instance_policy_check_precedes_score_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_tiny_config(), minimum_base_reference_tv=0.99)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("score randomness must not be generated")

    monkeypatch.setattr(horizon_overlap, "generate_logged_randomness", fail_if_called)
    with pytest.raises(RuntimeError, match="before score generation"):
        run_horizon_overlap_instance(
            config,
            problem_seed=110_200,
            logging_seed=110_300,
        )


def test_zero_tv_makes_all_four_stagewise_estimators_identical() -> None:
    config = _tiny_config()

    result = run_horizon_overlap_instance(
        config,
        problem_seed=110_200,
        logging_seed=110_300,
    )
    arrays = result.arrays

    assert arrays["selected_indices"].shape == (1, len(METHOD_NAMES), 2)
    assert arrays["availability_by_horizon"].shape == (1, 1, len(METHOD_NAMES))
    assert arrays["availability_by_horizon"].all()
    for name in (
        "selected_indices",
        "population_coverage",
        "population_width",
        "estimated_coverage",
        "estimated_width",
        "selected_ess_fraction",
        "minimum_candidate_ess_fraction",
        "stage_surface_sup_error",
    ):
        values = arrays[name][0]
        assert np.allclose(values, values[0], equal_nan=True)
    assert arrays["mixing_strength"].item() == 0.0
    assert arrays["realized_reference_tv"].item() == 0.0
    assert np.allclose(arrays["selected_ess_fraction"], 1.0)
    assert np.allclose(arrays["selected_policy_realized_tv"], 0.0)


def test_selected_population_coverage_matches_existing_exact_mdp_api() -> None:
    config = replace(
        _tiny_config(),
        calibration_trajectories=1_024,
        nominal_policy_tvs=(0.0, 0.05),
    )
    result = run_horizon_overlap_instance(
        config,
        problem_seed=110_200,
        logging_seed=110_300,
    )
    selected = result.arrays["selected_indices"][1, METHOD_NAMES.index("SC-PCP")]
    assert np.all(selected >= 0)

    mechanism = horizon_overlap._build_m3_base(config, problem_seed=110_200)
    base_target = mechanism.problem.action_probabilities.numpy()
    solution = solve_policy_mixing_strength(
        mechanism.behavior_probabilities,
        base_target[:, config.reference_grid_index],
        nominal_tv=0.05,
    )
    target = mix_target_policy(
        mechanism.behavior_probabilities,
        base_target,
        mixing_strength=solution.mixing_strength,
    )
    exact_problem = replace(
        mechanism.problem,
        action_probabilities=torch.from_numpy(target),
    )
    exact = analytic_schedule_metrics(
        exact_problem,
        tuple(int(index) for index in selected),
    )

    assert result.arrays["population_coverage"][1, -1].tolist() == pytest.approx(
        exact.coverage.tolist(),
        abs=1e-12,
    )
    assert result.arrays["population_width"][1, -1].tolist() == pytest.approx(
        exact.normalized_width.tolist(),
        abs=1e-12,
    )


def test_summary_uses_minimum_of_stagewise_instance_means() -> None:
    config = replace(
        _tiny_config(),
        instances=2,
        problem_seed_start=110_200,
        logging_seed_start=110_300,
    )
    instance_count = config.instances
    tv_count = len(config.nominal_policy_tvs)
    method_count = len(METHOD_NAMES)
    horizon = config.maximum_horizon
    stage_shape = (instance_count, tv_count, method_count, horizon)
    coverage = np.full(stage_shape, 0.9)
    coverage[:, 0, 0] = np.asarray([[0.8, 1.0], [1.0, 0.8]])
    coverage[:, 0, METHOD_NAMES.index("SC-PCP")] = np.asarray(
        [[0.95, 0.95], [0.85, 0.85]]
    )
    availability = np.ones(
        (instance_count, tv_count, 1, method_count),
        dtype=bool,
    )
    availability[1, 0, 0, METHOD_NAMES.index("History-only Prefix-IW")] = False
    arrays = {
        "availability_by_horizon": availability,
        "mixing_strength": np.zeros((instance_count, tv_count)),
        "realized_reference_tv": np.zeros((instance_count, tv_count)),
        "population_coverage": coverage,
        "population_width": np.ones(stage_shape),
        "selected_ess_fraction": np.ones(stage_shape),
        "minimum_candidate_ess_fraction": np.ones(stage_shape),
        "stage_surface_sup_error": np.zeros(stage_shape),
        "selected_policy_realized_tv": np.zeros(stage_shape),
        "selected_policy_uniform_state_tv": np.zeros(stage_shape),
    }

    summary = horizon_overlap._summarize_study(config, arrays)
    standard = next(
        record
        for record in summary["records"]
        if record["method"] == "Standard CP"
    )

    assert standard["marginal_wsc"] == pytest.approx(0.9)
    assert standard["mean_instance_worst_stage_coverage_diagnostic"] == pytest.approx(
        0.8
    )
    assert summary["bootstrap"]["instance_index_matrix_shape"] == [100, 2]
    assert summary["bootstrap"]["seed"] == 110_400
    assert standard["marginal_wsc_ci95"] is not None
    assert standard["average_normalized_width_ci95"] is not None
    assert standard["availability_rate_ci95"] is not None
    assert len(summary["bootstrap_wsc_comparisons"]) == 2
    assert {
        comparison["comparator"]
        for comparison in summary["bootstrap_wsc_comparisons"]
    } == {"History-only Prefix-IW", "Current-only IW"}
    history_comparison = next(
        comparison
        for comparison in summary["bootstrap_wsc_comparisons"]
        if comparison["comparator"] == "History-only Prefix-IW"
    )
    assert history_comparison["joint_available_instances"] == 1
    assert history_comparison[
        "method_conditional_scpcp_minus_comparator_wsc"
    ] == pytest.approx(0.0)
    assert history_comparison[
        "joint_available_scpcp_minus_comparator_wsc"
    ] == pytest.approx(0.05)


def test_parent_formal_snapshot_is_content_addressed(tmp_path: Path) -> None:
    runner = _load_runner()

    contract = runner._validate_parent_formal_snapshot()

    assert contract["manifest_sha256"] == (
        "e6a1bba7f3be47d39357f212824e7720262e7d5212a14628e3b8981088c64e24"
    )
    assert contract["archive_sha256"] == (
        "2116b9929240d8a25092f3c9015c362957bc963532930e5e720dc7ec78b2ea0b"
    )
    assert contract["parent_source_tree_sha256"] == (
        "7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643"
    )

    mutated_manifest = tmp_path / "mutated.manifest.json"
    mutated_manifest.write_bytes(runner.PARENT_SNAPSHOT_MANIFEST.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="manifest SHA256 differs"):
        runner._validate_parent_formal_snapshot(manifest_path=mutated_manifest)


def test_formal_loader_rejects_any_non_output_config_mutation(tmp_path: Path) -> None:
    runner = _load_runner()
    canonical = ROOT / "configs" / "horizon_overlap.yaml"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(canonical.read_text().replace("alpha: 0.10", "alpha: 0.11"))

    with pytest.raises(RuntimeError, match="byte-identical"):
        runner.load_formal_config(mutated)

    base = HorizonOverlapConfig.from_yaml(canonical)
    output_override = base.with_output_dir(tmp_path / "allowed-output")
    provenance = runner._formal_config_provenance(canonical, output_override)
    assert provenance["explicit_output_dir_override"] is True
    with pytest.raises(RuntimeError, match="only an output-dir override"):
        runner._formal_config_provenance(
            canonical,
            replace(base, alpha=0.11),
        )


def test_rng_audit_scans_source_and_artifact_metadata(tmp_path: Path) -> None:
    runner = _load_runner()
    config = _tiny_config()
    artifact_root = tmp_path / "artifacts"
    source_root = tmp_path / "source"
    artifact_root.mkdir()
    (source_root / "scripts").mkdir(parents=True)
    selected_config = tmp_path / "selected.yaml"
    _write_config(selected_config, config)

    (source_root / "scripts" / "reservations.py").write_text(
        "MUTUAL_RNG_RESERVATIONS = {'rq5': range(110100, 110401)}\n"
    )
    reservation_artifact = artifact_root / "reservation"
    reservation_artifact.mkdir()
    (reservation_artifact / "metadata.json").write_text(
        json.dumps({"mutual_rng_reservations": {"rq5": [110_200]}})
    )
    clean = runner._audit_formal_rng_ids(
        config,
        output_dir=tmp_path / "new_output",
        selected_config_path=selected_config,
        artifact_root=artifact_root,
        source_root=source_root,
    )
    assert clean["status"] == "passed_before_launch"
    assert clean["formal_rng_id_count"] == 5
    assert len(clean["formal_rng_id_sha256"]) == 64

    (source_root / "scripts" / "old.py").write_text(
        "OLD_RNG_IDS = (110200,)\n"
    )
    with pytest.raises(RuntimeError, match="prior declaration/artifact use"):
        runner._audit_formal_rng_ids(
            config,
            output_dir=tmp_path / "new_output",
            selected_config_path=selected_config,
            artifact_root=artifact_root,
            source_root=source_root,
        )

    (source_root / "scripts" / "old.py").write_text("NO_IDS = ()\n")
    old_artifact = artifact_root / "old"
    old_artifact.mkdir()
    (old_artifact / "metadata.json").write_text(
        json.dumps({"logging_seed": 110_300})
    )
    with pytest.raises(RuntimeError, match="prior declaration/artifact use"):
        runner._audit_formal_rng_ids(
            config,
            output_dir=tmp_path / "new_output",
            selected_config_path=selected_config,
            artifact_root=artifact_root,
            source_root=source_root,
        )


def test_runner_publishes_atomic_bundle_and_resumes_after_workspace_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.chdir(tmp_path)
    config = _tiny_config()
    canonical_config = tmp_path / "canonical.yaml"
    _write_config(canonical_config, config)
    monkeypatch.setattr(runner, "DEFAULT_CONFIG", canonical_config)
    monkeypatch.setattr(runner, "source_tree_sha256", lambda: "a" * 64)
    monkeypatch.setattr(runner, "git_revision", lambda: "deadbeef")

    fresh = runner.run_study(config, resume=False)
    resumed = runner.run_study(config, resume=True)

    assert fresh == resumed
    assert {
        "config.json",
        "metadata.json",
        "summary.json",
        "results.npz",
        "manifest.json",
        "COMPLETE",
    } == {path.name for path in config.output_dir.iterdir()}
    metadata = json.loads((config.output_dir / "metadata.json").read_text())
    assert metadata["canonical_method_unchanged"] is True
    assert metadata["seed_collision_audit"]["collision"] is False
    assert metadata["policy_design_audit"]["status"] == "pass"
    assert metadata["policy_design_audit"]["formal_problem_seed_ids"] == [110_200]
    assert metadata["config_provenance"]["raw_config_is_default_byte_identical"] is True
    assert metadata["formal_rng_collision_audit"]["collision_count"] == 0
    assert metadata["formal_rng_collision_audit"]["formal_rng_mapping"][
        "summary/instance_cluster_bootstrap"
    ] == 110_400
    assert metadata["parent_formal_snapshot"]["manifest_sha256"] == (
        "e6a1bba7f3be47d39357f212824e7720262e7d5212a14628e3b8981088c64e24"
    )
    assert metadata["rq5_only_policy_center_reset"]["scope"] == (
        "RQ5_horizon_overlap_only"
    )
    assert isinstance(metadata["launch"]["argv"], list)
    assert metadata["environment_versions"]["python"]["version"]
    assert metadata["environment_versions"]["numpy"]["blas"]
    assert metadata["environment_versions"]["numpy"]["lapack"]
    complete = json.loads((config.output_dir / "COMPLETE").read_text())
    assert complete["parent_snapshot_archive_sha256"] == (
        "2116b9929240d8a25092f3c9015c362957bc963532930e5e720dc7ec78b2ea0b"
    )
    assert complete["parent_source_tree_sha256"] == (
        "7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643"
    )

    stored_rng_audit = metadata["formal_rng_collision_audit"]
    grown_rng_audit = {
        **stored_rng_audit,
        "artifact_rng_id_count": stored_rng_audit["artifact_rng_id_count"] + 1,
        "artifact_rng_id_sha256": "b" * 64,
        "prior_declared_or_artifact_rng_id_count": (
            stored_rng_audit["prior_declared_or_artifact_rng_id_count"] + 1
        ),
        "prior_declared_or_artifact_rng_id_sha256": "c" * 64,
    }
    grown_rng_audit.pop("audit_sha256")
    grown_rng_audit["audit_sha256"] = runner._canonical_sha256(grown_rng_audit)
    monkeypatch.setattr(
        runner,
        "_audit_formal_rng_ids",
        lambda *args, **kwargs: grown_rng_audit,
    )
    monkeypatch.setattr(
        runner,
        "source_tree_sha256",
        lambda: (_ for _ in ()).throw(
            AssertionError("resume must not hash current source")
        ),
    )

    absolute_config = config.with_output_dir(config.output_dir.resolve())
    resumed_after_growth = runner.run_study(absolute_config, resume=True)

    assert resumed_after_growth == fresh


def test_resume_accepts_legacy_environment_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    config = _tiny_config(tmp_path)
    canonical_config = tmp_path / "canonical.yaml"
    _write_config(canonical_config, config)
    monkeypatch.setattr(runner, "DEFAULT_CONFIG", canonical_config)
    monkeypatch.setattr(runner, "source_tree_sha256", lambda: "a" * 64)
    legacy_environment = {
        "python": "3.11.13",
        "python_implementation": "CPython",
        "platform": "legacy-platform",
        "numpy": "2.2.6",
        "torch": "2.8.0+cu128",
        "pyyaml": "6.0.3",
        "conda_environment": "ucp",
    }
    current_environment = runner._environment_versions
    monkeypatch.setattr(runner, "_environment_versions", lambda: legacy_environment)

    fresh = runner.run_study(config, resume=False)
    monkeypatch.setattr(runner, "_environment_versions", current_environment)
    resumed = runner.run_study(config, resume=True)

    assert resumed == fresh
    metadata = json.loads((config.output_dir / "metadata.json").read_text())
    assert metadata["environment_versions"] == legacy_environment
