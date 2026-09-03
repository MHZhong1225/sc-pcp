from __future__ import annotations

from dataclasses import fields, replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import scpcp.propensity_robustness as robustness
from scpcp.propensity_robustness import (
    APPENDIX_LAYER,
    PRIMARY_LAYER,
    PROPENSITY_ARMS,
    SEED_NAMESPACE,
    PropensityRobustnessConfig,
    propensity_seed_collision_audit,
    run_propensity_robustness,
    smoke_config,
)


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_propensity_robustness.py"
    spec = importlib.util.spec_from_file_location("run_propensity_robustness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_contract_and_seed_namespace_are_disjoint() -> None:
    config = PropensityRobustnessConfig()
    config.validate()
    config.assert_frozen_protocol()
    audit = propensity_seed_collision_audit(config)

    assert config.seed_namespace == SEED_NAMESPACE
    assert config.problem_seeds == tuple(range(98_000, 98_100))
    assert config.nuisance_seeds == tuple(range(98_100, 98_200))
    assert config.calibration_seeds == tuple(range(98_200, 98_300))
    assert audit["rng_id_count"] == 301
    assert audit["all_rng_ids"][0] == 98_000
    assert audit["all_rng_ids"][-1] == 98_300
    assert not audit["collision"]
    assert audit["external_reservations"]["strict_split_audit"] == "99000..99999"
    assert audit["external_reservations"]["score_robustness"] == "100000..100999"


def test_every_formal_config_field_is_frozen() -> None:
    config = PropensityRobustnessConfig()
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, str):
            changed = value + "_changed"
        elif isinstance(value, int):
            changed = value + 1
        elif isinstance(value, float):
            changed = value + 0.03125
        else:  # pragma: no cover - guards future unsupported config fields
            raise AssertionError(f"unhandled config field type: {field.name}")
        with pytest.raises(ValueError, match="frozen protocol"):
            replace(config, **{field.name: changed}).assert_frozen_protocol()


def test_actual_use_rng_scan_excludes_reservations_and_rejects_real_use(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    source = tmp_path / "source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    reservation = scripts / "reservation.py"
    reservation.write_text(
        "EXTERNAL_SEED_RESERVATIONS = {'propensity': range(98_000, 99_000)}\n"
    )
    assert 98_000 not in runner._source_actual_rng_ids(
        source,
        excluded_paths=set(),
    )

    actual = scripts / "actual.py"
    actual.write_text("FORMAL_RNG_SEED = 98_000\n")
    assert 98_000 in runner._source_actual_rng_ids(
        source,
        excluded_paths=set(),
    )
    with pytest.raises(RuntimeError, match="prior actual use"):
        runner._audit_formal_rng_ids(
            PropensityRobustnessConfig(),
            output_dir=tmp_path / "out",
            artifact_root=tmp_path / "artifacts",
            source_root=source,
        )


def test_artifact_rng_scan_rejects_actual_seed_but_not_reservation(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metadata.json").write_text(
        json.dumps({"external_seed_reservation": [98_000, 98_001]})
    )
    assert 98_000 not in runner._artifact_rng_ids(
        artifacts,
        excluded_root=tmp_path / "out",
    )
    (artifacts / "summary.json").write_text(json.dumps({"problem_seed": 98_042}))
    assert 98_042 in runner._artifact_rng_ids(
        artifacts,
        excluded_root=tmp_path / "out",
    )


def test_parent_snapshot_environment_and_formal_rng_contracts_are_bound(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    parent = runner._validate_parent_formal_snapshot()
    environment = runner._environment_versions()
    rng_audit = runner._audit_formal_rng_ids(
        PropensityRobustnessConfig(),
        output_dir=tmp_path / "out",
        artifact_root=tmp_path / "empty_artifacts",
    )

    assert parent["manifest_sha256"] == runner.EXPECTED_PARENT_MANIFEST_SHA256
    assert parent["archive_sha256"] == runner.EXPECTED_PARENT_ARCHIVE_SHA256
    assert parent["parent_source_tree_sha256"] == (
        runner.EXPECTED_PARENT_SOURCE_TREE_SHA256
    )
    assert environment["python"]["version"]
    assert environment["numpy"]["version"]
    assert environment["numpy"]["blas"]
    assert environment["torch"]["version"]
    assert environment["scikit_learn"]
    assert rng_audit["status"] == "passed_before_launch"
    assert rng_audit["formal_rng_id_count"] == 301
    assert rng_audit["source_actual_use_excludes_reservation_declarations"]


def test_primary_target_law_is_shared_and_appendix_is_separate() -> None:
    result = run_propensity_robustness(smoke_config())

    primary_fingerprints = result.arrays["primary_target_law_fingerprints"]
    appendix_fingerprints = result.arrays["appendix_target_law_fingerprints"]
    assert np.all(primary_fingerprints == primary_fingerprints[:, :1])
    assert np.all(appendix_fingerprints[:, 0] == primary_fingerprints[:, 0])
    assert np.any(appendix_fingerprints[:, 1:] != primary_fingerprints[:, :1])
    assert result.summary[PRIMARY_LAYER][
        "target_law_fingerprint_shared_across_arms"
    ]
    assert all(record["layer"] == PRIMARY_LAYER for record in result.primary_records)
    assert all(record["layer"] == APPENDIX_LAYER for record in result.appendix_records)
    assert all(
        record["target_policy_drift_from_oracle"] == 0.0
        for record in result.primary_records
    )
    assert 0.04 <= result.summary["moderate_policy_tv"]["mean"] <= 0.07
    primary_tv = result.arrays[
        "primary_selected_policy_tv_from_oracle_behavior"
    ]
    appendix_anchor_tv = result.arrays[
        "appendix_selected_policy_tv_from_own_anchor"
    ]
    appendix_matched_tv = result.arrays[
        "appendix_selected_policy_tv_from_oracle_target_matched_radii"
    ]
    appendix_deployed_tv = result.arrays[
        "appendix_deployed_policy_tv_from_primary_oracle_deployment"
    ]
    assert primary_tv.shape == (
        2,
        3,
        result.arrays["primary_transport_only_exact_coverage"].shape[-1],
    )
    assert np.allclose(appendix_anchor_tv[:, 0], primary_tv[:, 0])
    assert np.allclose(appendix_matched_tv[:, 0], 0.0)
    assert np.allclose(appendix_deployed_tv[:, 0], 0.0)
    assert np.any(appendix_matched_tv[:, 1:] > 0.0)
    assert "oracle logging policy mu" in result.summary[PRIMARY_LAYER][
        "selected_deployed_policy_tv"
    ]["reference_and_measure"]
    appendix_summary = result.summary[APPENDIX_LAYER]["selected_deployed_policy_tv"]
    assert set(appendix_summary) == {
        "from_own_anchor",
        "from_oracle_target_at_matched_selected_radii",
        "from_primary_oracle_deployment",
    }


def test_smoke_result_has_all_nuisance_and_scpcp_outputs() -> None:
    config = smoke_config()
    result = run_propensity_robustness(config)

    assert result.arrays["propensity_arms"].tolist() == list(PROPENSITY_ARMS)
    assert result.arrays["nuisance_mae"].shape == (config.instances, 3)
    assert np.allclose(result.arrays["nuisance_mae"][:, 0], 0.0)
    assert np.allclose(result.arrays["nuisance_excess_log_loss"][:, 0], 0.0)
    assert np.isfinite(result.arrays["nuisance_log_loss"]).all()
    assert np.isfinite(result.arrays["nuisance_mean_absolute_relative_error"]).all()
    assert result.arrays["bootstrap_indices"].shape == (
        config.bootstrap_resamples,
        config.instances,
    )
    for layer in (PRIMARY_LAYER, APPENDIX_LAYER):
        assert result.arrays[f"{layer}_exact_coverage"].shape == (
            config.instances,
            3,
            config.horizon,
        )
        layer_summary = result.summary[layer]["results"]
        assert layer_summary["bootstrap"][
            "same_problem_seed_matrix_for_every_arm_and_metric"
        ]
        assert set(layer_summary["paired_comparisons_vs_oracle"]) == set(
            PROPENSITY_ARMS[1:]
        )


def test_selected_policy_tv_matches_direct_policy_probability_calculation() -> None:
    config = smoke_config()
    result = run_propensity_robustness(config)
    mechanism = robustness._build_m3_mechanism(  # type: ignore[attr-defined]
        config,
        problem_seed=config.problem_seeds[0],
    )
    nuisance_logged = robustness._simulate_logged(  # type: ignore[attr-defined]
        mechanism,
        config,
        trajectories=config.nuisance_trajectories,
        seed=config.nuisance_seeds[0],
    )
    estimates, _ = robustness._fit_propensity_arms(  # type: ignore[attr-defined]
        nuisance_logged,
        mechanism.behavior_probabilities,
        config,
    )
    occupancy = robustness._logging_state_occupancy(  # type: ignore[attr-defined]
        mechanism
    )
    oracle_target = np.asarray(mechanism.problem.action_probabilities)

    def deployed(surface: np.ndarray, layer: str, arm_index: int) -> np.ndarray:
        indices = result.arrays[f"{layer}_selected_indices"][0, arm_index]
        return np.stack(
            [surface[stage, int(index)] for stage, index in enumerate(indices)]
        )

    primary_oracle = deployed(oracle_target, PRIMARY_LAYER, 0)
    for arm_index, arm in enumerate(PROPENSITY_ARMS):
        primary = deployed(oracle_target, PRIMARY_LAYER, arm_index)
        direct_primary = robustness._policy_tv_by_stage(  # type: ignore[attr-defined]
            primary,
            mechanism.behavior_probabilities,
            occupancy,
        )
        assert np.allclose(
            direct_primary,
            result.arrays["primary_selected_policy_tv_from_oracle_behavior"][
                0, arm_index
            ],
        )

        appendix_surface = robustness._target_policy_from_anchor(  # type: ignore[attr-defined]
            estimates[arm],
            np.asarray(mechanism.problem.radii),
            config,
        )
        appendix = deployed(appendix_surface, APPENDIX_LAYER, arm_index)
        matched_oracle = deployed(oracle_target, APPENDIX_LAYER, arm_index)
        for reference, array_name in (
            (estimates[arm], "appendix_selected_policy_tv_from_own_anchor"),
            (
                matched_oracle,
                "appendix_selected_policy_tv_from_oracle_target_matched_radii",
            ),
            (
                primary_oracle,
                "appendix_deployed_policy_tv_from_primary_oracle_deployment",
            ),
        ):
            direct = robustness._policy_tv_by_stage(  # type: ignore[attr-defined]
                appendix,
                reference,
                occupancy,
            )
            assert np.allclose(direct, result.arrays[array_name][0, arm_index])


def test_array_validator_locks_seeds_bootstrap_fingerprints_and_identities(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    config = smoke_config()
    result = run_propensity_robustness(config)
    base = {name: value.copy() for name, value in result.arrays.items()}
    valid_path = tmp_path / "valid.npz"
    np.savez_compressed(valid_path, **base)
    runner._validate_arrays(valid_path, config.to_dict())

    corruptions = []
    wrong_seed = {name: value.copy() for name, value in base.items()}
    wrong_seed["problem_seeds"][0] += 1
    corruptions.append(("seed", wrong_seed, "exact frozen sequence"))

    wrong_bootstrap = {name: value.copy() for name, value in base.items()}
    wrong_bootstrap["bootstrap_indices"][0, 0] = (
        int(wrong_bootstrap["bootstrap_indices"][0, 0]) + 1
    ) % config.instances
    corruptions.append(("bootstrap", wrong_bootstrap, "seed-deterministic"))

    wrong_fingerprint = {name: value.copy() for name, value in base.items()}
    wrong_fingerprint["primary_target_law_fingerprints"][0, 1] = "0" * 64
    corruptions.append(("fingerprint", wrong_fingerprint, "identical across arms"))

    wrong_range = {name: value.copy() for name, value in base.items()}
    wrong_range[f"{PRIMARY_LAYER}_ess_fraction"][0, 0, 0] = 1.1
    corruptions.append(("range", wrong_range, "ESS"))

    wrong_identity = {name: value.copy() for name, value in base.items()}
    wrong_identity[f"{PRIMARY_LAYER}_failure_stage"][0, 0] = 0
    corruptions.append(("identity", wrong_identity, "failure stage"))

    for label, arrays, message in corruptions:
        path = tmp_path / f"{label}.npz"
        np.savez_compressed(path, **arrays)
        with pytest.raises(RuntimeError, match=message):
            runner._validate_arrays(path, config.to_dict())


def test_multinomial_nonconvergence_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = smoke_config()

    class _NonconvergedLogistic:
        def __init__(self, **_kwargs):
            self.n_iter_ = np.array([config.logistic_max_iterations])
            self.classes_ = np.arange(config.action_count)

        def fit(self, _features, _actions):
            return self

        def predict_proba(self, features):
            return np.full((len(features), config.action_count), 1.0 / config.action_count)

    monkeypatch.setattr(robustness, "LogisticRegression", _NonconvergedLogistic)
    states = np.tile(np.arange(config.state_count), 12)
    actions = np.tile(np.arange(config.action_count), 32)
    with pytest.raises(RuntimeError, match="iteration limit"):
        robustness._fit_multinomial(  # type: ignore[attr-defined]
            states,
            actions,
            reduced=False,
            config=config,
        )


def test_runner_publishes_separate_atomic_tables_and_resume_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    config = smoke_config()
    output = tmp_path / "propensity"
    monkeypatch.setattr(runner, "source_tree_sha256", lambda: "a" * 64)
    monkeypatch.setattr(runner, "git_revision", lambda: "deadbeef")
    fsynced_files: list[str] = []
    fsynced_directories: list[Path] = []
    original_fsync_file = runner._fsync_file
    original_fsync_directory = runner._fsync_directory

    def record_fsync_file(path: Path) -> None:
        fsynced_files.append(path.name)
        original_fsync_file(path)

    def record_fsync_directory(path: Path) -> None:
        fsynced_directories.append(path)
        original_fsync_directory(path)

    monkeypatch.setattr(runner, "_fsync_file", record_fsync_file)
    monkeypatch.setattr(runner, "_fsync_directory", record_fsync_directory)

    fresh = runner.run_study(config, output, resume=False)
    resumed = runner.run_study(config, output, resume=True)

    assert fresh == resumed
    assert (output / "COMPLETE").is_file()
    assert (output / "primary_transport_only.csv").is_file()
    assert (output / "appendix_end_to_end.csv").is_file()
    assert not (output / "combined_layers.csv").exists()
    assert set(fsynced_files) == {
        "arrays.npz",
        "nuisance_diagnostics.csv",
        "primary_transport_only.csv",
        "appendix_end_to_end.csv",
    }
    assert len(fsynced_directories) == 2
    assert fsynced_directories[-1] == output.parent
    metadata = json.loads((output / "metadata.json").read_text())
    complete = json.loads((output / "COMPLETE").read_text())
    assert metadata["parent_formal_snapshot"]["manifest_sha256"] == (
        runner.EXPECTED_PARENT_MANIFEST_SHA256
    )
    assert metadata["parent_formal_snapshot"]["archive_sha256"] == (
        runner.EXPECTED_PARENT_ARCHIVE_SHA256
    )
    assert metadata["parent_formal_snapshot"]["parent_source_tree_sha256"] == (
        runner.EXPECTED_PARENT_SOURCE_TREE_SHA256
    )
    assert isinstance(metadata["launch"]["argv"], list)
    assert metadata["environment_versions"]["numpy"]["blas"]
    fit_semantics = metadata["multinomial_propensity_fit_semantics"]
    assert fit_semantics["library_version"] == metadata["environment_versions"][
        "scikit_learn"
    ]
    assert "multinomial negative log-likelihood" in fit_semantics["loss"]
    assert fit_semantics["solver"] == config.logistic_solver
    assert "L-BFGS" in fit_semantics["solver_semantics"]
    assert complete["parent_snapshot_manifest_sha256"] == (
        runner.EXPECTED_PARENT_MANIFEST_SHA256
    )
    assert complete["parent_snapshot_archive_sha256"] == (
        runner.EXPECTED_PARENT_ARCHIVE_SHA256
    )
    assert complete["parent_source_tree_sha256"] == (
        runner.EXPECTED_PARENT_SOURCE_TREE_SHA256
    )
    for name in (
        "nuisance_diagnostics.csv",
        "primary_transport_only.csv",
        "appendix_end_to_end.csv",
    ):
        table = runner.pd.read_csv(output / name)
        assert {"problem_seed", "nuisance_seed", "calibration_seed"} <= set(
            table.columns
        )

    nuisance = runner.pd.read_csv(output / "nuisance_diagnostics.csv")
    primary = runner.pd.read_csv(output / "primary_transport_only.csv")
    appendix = runner.pd.read_csv(output / "appendix_end_to_end.csv")
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        runner._cross_validate_csv_and_npz(nuisance, primary, appendix, arrays)
        nuisance.loc[0, "mae"] += 0.01
        with pytest.raises(RuntimeError, match="nuisance CSV differs from NPZ"):
            runner._cross_validate_csv_and_npz(nuisance, primary, appendix, arrays)

    original_parent_validator = runner._validate_parent_formal_snapshot
    changed_parent = dict(metadata["parent_formal_snapshot"])
    changed_parent["relationship"] = "tampered"
    monkeypatch.setattr(
        runner,
        "_validate_parent_formal_snapshot",
        lambda: changed_parent,
    )
    with pytest.raises(RuntimeError, match="parent formal snapshot binding"):
        runner.run_study(config, output, resume=True)
    monkeypatch.setattr(
        runner,
        "_validate_parent_formal_snapshot",
        original_parent_validator,
    )

    (output / "summary.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="payload hash differs"):
        runner.run_study(config, output, resume=True)
