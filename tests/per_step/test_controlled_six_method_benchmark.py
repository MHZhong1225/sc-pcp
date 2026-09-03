from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_controlled_six_method_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "run_controlled_six_method_benchmark",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_contains_exactly_the_six_canonical_methods_and_budgets() -> None:
    runner = _load_runner()

    assert runner.METHODS == (
        "Standard CP",
        "ACI",
        "MFCS",
        "SPCI",
        "PRC",
        "SC-PCP",
    )
    assert runner.TARGET_ADAPTATION_BUDGET == {
        "Standard CP": 0,
        "ACI": 2_000,
        "MFCS": 0,
        "SPCI": 2_000,
        "PRC": 2_000,
        "SC-PCP": 0,
    }
    assert runner.REFERENCE_TRAJECTORIES == 20_000
    assert runner.CALIBRATION_TRAJECTORIES == 3_000
    assert runner.GRID_TRAJECTORIES == 1_000
    assert runner.CONFIRM_SEEDS == tuple(range(91_000, 91_200, 10))
    metadata = runner._metadata(
        devices=("cuda:0", "cuda:1"),
        active_source_hash="active-source",
        config_contract={"active": "config"},
        seed_to_device=runner._seed_device_mapping(
            runner.CONFIRM_SEEDS,
            ("cuda:0", "cuda:1"),
        ),
        seed_audit={"status": "test"},
    )
    note = metadata["reference_budget_origin"]
    assert "20,000 fresh evaluation trajectories per method/seed/gamma" in note
    assert "pasted 50,000" in note
    assert "parent-protocol parity" in note


def test_resume_metadata_comparison_survives_json_tuple_canonicalization() -> None:
    runner = _load_runner()
    metadata = runner._metadata(
        devices=("cuda:0", "cuda:1"),
        active_source_hash="active-source",
        config_contract=runner._config_contract(),
        seed_to_device=runner._seed_device_mapping(
            runner.CONFIRM_SEEDS,
            ("cuda:0", "cuda:1"),
        ),
        seed_audit={"status": "test"},
    )
    stored_metadata = json.loads(json.dumps(metadata))

    assert stored_metadata != metadata
    assert runner._json_sha256(stored_metadata) == runner._json_sha256(metadata)


def test_controlled_online_callback_uses_explicit_gamma_seed_and_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    transition = SimpleNamespace(horizon=2, initial_count=17)
    action_coordinate = torch.tensor([-1.0, 1.0])
    wrapped = runner.ControlledOnlineEnvironment(
        transition=transition,
        gamma=-2.0,
        action_coordinate=action_coordinate,
    )
    noise = object()
    trajectories = object()
    observed: dict[str, object] = {}

    def fake_noise(**kwargs):
        observed["noise_kwargs"] = kwargs
        return noise

    def fake_rollout(environment, policy, **kwargs):
        observed["environment"] = environment
        observed["policy"] = policy
        observed["rollout_kwargs"] = kwargs
        return SimpleNamespace(trajectories=trajectories)

    monkeypatch.setattr(runner, "make_controlled_noise", fake_noise)
    monkeypatch.setattr(runner, "rollout_controlled", fake_rollout)
    policy = object()
    radii = torch.tensor([1.0, 1.5])
    result = runner._controlled_online_rollout(
        wrapped,
        policy,
        n=667,
        horizon=2,
        seed=12345,
        device="cpu",
        q=radii,
    )

    assert result is trajectories
    assert observed["noise_kwargs"] == {
        "n": 667,
        "horizon": 2,
        "initial_count": 17,
        "seed": 12345,
        "device": "cpu",
    }
    assert observed["environment"] is transition
    assert observed["policy"] is policy
    assert observed["rollout_kwargs"]["noise"] is noise
    assert observed["rollout_kwargs"]["gamma"] == -2.0
    assert torch.equal(observed["rollout_kwargs"]["action_coordinate"], action_coordinate)
    assert torch.equal(observed["rollout_kwargs"]["radii"], radii)


def test_seed_bank_audit_rejects_artifact_and_source_collisions(tmp_path: Path) -> None:
    runner = _load_runner()
    artifact_root = tmp_path / "results"
    source_root = tmp_path / "source"
    (artifact_root / "old" / "seed_91000").mkdir(parents=True)
    (source_root / "scripts").mkdir(parents=True)
    (source_root / "scripts" / "old.py").write_text("OLD_SEEDS = (1, 3)\n")

    with pytest.raises(RuntimeError, match="collides with prior use"):
        runner._audit_seed_bank(
            runner.CONFIRM_SEEDS,
            output_dir=tmp_path / "new_output",
            artifact_root=artifact_root,
            source_root=source_root,
        )

    (artifact_root / "old" / "seed_91000").rmdir()
    (source_root / "scripts" / "old.py").write_text("OLD_SEEDS = (91010,)\n")
    with pytest.raises(RuntimeError, match="collides with prior use"):
        runner._audit_seed_bank(
            runner.CONFIRM_SEEDS,
            output_dir=tmp_path / "new_output",
            artifact_root=artifact_root,
            source_root=source_root,
        )


def test_seed_bank_audit_records_collision_free_snapshot(tmp_path: Path) -> None:
    runner = _load_runner()
    artifact_root = tmp_path / "results"
    source_root = tmp_path / "source"
    (artifact_root / "old" / "seed_00007").mkdir(parents=True)
    (source_root / "scripts").mkdir(parents=True)
    (source_root / "scripts" / "old.py").write_text(
        "DEVELOPMENT_SEEDS = tuple(range(100, 120, 2))\n"
    )

    audit = runner._audit_seed_bank(
        runner.CONFIRM_SEEDS,
        output_dir=tmp_path / "new_output",
        artifact_root=artifact_root,
        source_root=source_root,
    )

    assert audit["status"] == "passed_before_launch"
    assert audit["artifact_seed_count"] == 1
    assert audit["source_declared_seed_count"] == 10
    assert audit["internal_rng_streams_unique"] is True
    assert audit["new_rng_stream_count"] == 285
    assert len(audit["reserved_seed_sha256"]) == 64


def _available_method_row(coverage: list[float]) -> dict[str, object]:
    return {
        "selection_available": True,
        "target_coverage": coverage,
        "source_coverage": [0.9] * 12,
        "target_normalized_width": [2.0] * 12,
        "prefix_ess_fraction": [0.5] * 12,
        "maximum_normalized_weight_share": [0.01] * 12,
    }


def test_summary_uses_minimum_stagewise_seed_mean_and_selection_rate() -> None:
    runner = _load_runner()
    rows = []
    for seed_index, seed in enumerate(runner.CONFIRM_SEEDS):
        coverage = (
            [0.80, 0.95, *([0.95] * 10)]
            if seed_index < 10
            else [0.90, 0.85, *([0.95] * 10)]
        )
        for gamma in runner.GAMMAS:
            methods = {
                method: _available_method_row(coverage)
                for method in runner.METHODS
            }
            if seed_index == 0:
                methods["MFCS"] = {
                    "selection_available": False,
                    "radii": [],
                }
            rows.append({"seed": seed, "gamma": gamma, "methods": methods})

    summary = runner.summarize(rows, seeds=runner.CONFIRM_SEEDS)
    standard = summary["aggregates"][0]["methods"]["Standard CP"]
    mfcs = summary["aggregates"][0]["methods"]["MFCS"]

    assert standard["target_coverage_by_stage"][:2] == pytest.approx([0.85, 0.90])
    assert standard["target_marginal_worst_coverage"] == pytest.approx(0.85)
    assert standard["target_marginal_worst_coverage"] != pytest.approx(
        np.mean([0.80, 0.85])
    )
    assert mfcs["selected_seeds"] == 19
    assert mfcs["selection_rate"] == pytest.approx(0.95)
    gamma_seed = summary["aggregates"][0]["bootstrap_seed"]
    shared_uniforms = np.random.default_rng(gamma_seed).random(
        size=(runner.BOOTSTRAP_RESAMPLES, len(runner.CONFIRM_SEEDS))
    )
    mfcs_indices = runner._bootstrap_indices(shared_uniforms, 19)
    mfcs_coverage = np.asarray(
        [
            (
                [0.80, 0.95, *([0.95] * 10)]
                if seed_index < 10
                else [0.90, 0.85, *([0.95] * 10)]
            )
            for seed_index in range(1, 20)
        ]
    )
    expected_mfcs_draws = mfcs_coverage[mfcs_indices].mean(axis=1).min(axis=1)
    assert mfcs_indices.shape == (runner.BOOTSTRAP_RESAMPLES, 19)
    assert mfcs["target_wsc_ci95"] == pytest.approx(
        np.quantile(expected_mfcs_draws, [0.025, 0.975])
    )
    assert summary["primary_metric"] == "min_t mean_seed(target_coverage_seed_t)"
    paired = summary["aggregates"][0]["paired_scpcp_comparisons"]["Standard CP"]
    assert paired["paired_selected_seeds"] == 20
    assert paired["scpcp_minus_baseline_wsc"] == pytest.approx(0.0)
    assert paired["scpcp_to_baseline_geometric_width_ratio"] == pytest.approx(1.0)


def test_paired_bootstrap_resamples_exact_joint_selected_set() -> None:
    runner = _load_runner()
    shared_uniforms = np.asarray(
        [
            [0.00, 0.99, 0.10, 0.20],
            [0.10, 0.20, 0.30, 0.40],
            [0.80, 0.90, 0.70, 0.60],
            [0.45, 0.55, 0.50, 0.50],
        ]
    )
    scpcp = {
        "available": np.asarray([True, True, True, True]),
        "coverage": np.asarray(
            [
                [0.90, 0.95],
                [0.50, 0.50],
                [0.70, 0.95],
                [0.50, 0.50],
            ]
        ),
        "width": np.asarray([2.0, 9.0, 4.0, 9.0]),
    }
    baseline = {
        "available": np.asarray([True, False, True, False]),
        "coverage": np.asarray(
            [
                [0.80, 0.95],
                [0.00, 0.00],
                [1.00, 0.95],
                [0.00, 0.00],
            ]
        ),
        "width": np.asarray([4.0, 0.0, 2.0, 0.0]),
    }

    comparison = runner._paired_scpcp_comparison(
        scpcp,
        baseline,
        shared_uniforms,
    )
    joint_indices = runner._bootstrap_indices(shared_uniforms, 2)
    scpcp_joint = scpcp["coverage"][[0, 2]]
    baseline_joint = baseline["coverage"][[0, 2]]
    expected_difference = (
        scpcp_joint[joint_indices].mean(axis=1).min(axis=1)
        - baseline_joint[joint_indices].mean(axis=1).min(axis=1)
    )
    expected_width_ratio = np.exp(
        np.log(np.asarray([0.5, 2.0]))[joint_indices].mean(axis=1)
    )

    assert joint_indices.shape == (len(shared_uniforms), 2)
    assert comparison["paired_selected_seeds"] == 2
    assert comparison["scpcp_minus_baseline_wsc_ci95"] == pytest.approx(
        np.quantile(expected_difference, [0.025, 0.975])
    )
    assert comparison[
        "scpcp_to_baseline_geometric_width_ratio_ci95"
    ] == pytest.approx(np.quantile(expected_width_ratio, [0.025, 0.975]))


def test_seed_to_device_mapping_is_stable_under_resume_subsets() -> None:
    runner = _load_runner()
    devices = ("cuda:0", "cuda:1")
    mapping = runner._seed_device_mapping(runner.CONFIRM_SEEDS, devices)

    assert mapping[runner.CONFIRM_SEEDS[0]] == "cuda:0"
    assert mapping[runner.CONFIRM_SEEDS[1]] == "cuda:1"
    assert mapping[runner.CONFIRM_SEEDS[2]] == "cuda:0"
    pending = runner.CONFIRM_SEEDS[3:]
    grouped = {
        device: tuple(seed for seed in pending if mapping[seed] == device)
        for device in devices
    }
    assert grouped["cuda:1"][0] == runner.CONFIRM_SEEDS[3]
    assert grouped["cuda:0"][0] == runner.CONFIRM_SEEDS[4]


def test_config_contract_binds_yaml_bytes_and_active_override() -> None:
    runner = _load_runner()
    contract = runner._config_contract()

    assert len(contract["base_yaml_sha256"]) == 64
    assert len(contract["active_config_sha256"]) == 64
    assert contract["base_yaml_size_bytes"] == runner.BASE_CONFIG_PATH.stat().st_size
    assert contract["active_config"]["policy"]["policy_ratio_cap"] == 3.0
    assert contract["controlled_override"] == {"policy.policy_ratio_cap": 3.0}


def test_seed_artifact_validation_binds_the_global_device_mapping(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    seed = runner.CONFIRM_SEEDS[0]
    contract = {
        "protocol": runner.PROTOCOL,
        "source_tree_sha256": "source",
        "base_config_sha256": "base-config",
        "active_config_sha256": "active-config",
        "base_seed_bank_sha256": "seed-bank",
        "methods": list(runner.METHODS),
        "reference_trajectories": runner.REFERENCE_TRAJECTORIES,
    }
    rows = []
    for gamma in runner.GAMMAS:
        methods = {}
        for method in runner.METHODS:
            methods[method] = {
                "selection_available": True,
                "target_adaptation_trajectories": runner.TARGET_ADAPTATION_BUDGET[
                    method
                ],
                "radii": [1.0] * runner.HORIZON,
                "source_coverage": [0.9] * runner.HORIZON,
                "target_coverage": [0.9] * runner.HORIZON,
                "coverage_gap": [0.0] * runner.HORIZON,
                "target_normalized_width": [2.0] * runner.HORIZON,
                "prefix_ess_fraction": [0.5] * runner.HORIZON,
            }
        rows.append(
            {
                "seed": seed,
                "gamma": gamma,
                "adaptation_seeds": runner._adaptation_seeds(seed),
                "methods": methods,
            }
        )
    path = tmp_path / f"seed_{seed:05d}.json"
    runner._write_json(
        path,
        {**contract, "seed": seed, "device": "cuda:0", "rows": rows},
    )

    assert runner._valid_seed_file(
        path,
        seed=seed,
        expected_device="cuda:0",
        seed_contract=contract,
    )
    assert not runner._valid_seed_file(
        path,
        seed=seed,
        expected_device="cuda:1",
        seed_contract=contract,
    )
