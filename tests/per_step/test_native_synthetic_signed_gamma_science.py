from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import yaml

from scpcp.native_signed_gamma import (
    NativeSignedGammaBenchmarkConfig,
    NativeSignedGammaDGPConfig,
    NativeSignedGammaKernel,
    NativeSignedGammaLoggingPolicy,
    NativeSignedGammaRadiusPolicy,
    make_native_signed_gamma_noise,
    rollout_native_signed_gamma,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_native_synthetic_signed_gamma_science.py"
CONFIG_PATH = ROOT / "configs/native_synthetic_signed_gamma_science.yaml"


def _load_runner():
    name = "test_run_native_synthetic_signed_gamma_science"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


science = _load_runner()


def _config() -> science.ScienceConfig:
    return science.ScienceConfig.from_yaml(CONFIG_PATH)


def _gate_binding() -> science.GateBinding:
    core = {
        "repair_root": "results/work/native_synthetic_signed_gamma_v1_time_coordinate_repair_r1",
        "repair_protocol": "native_synthetic_signed_gamma_time_coordinate_repair_r1",
        "administrative_role": "administrative_validator_repair_replay",
        "decision": "GO",
        "n_prespecified": 20,
        "n_passed": 20,
        "required_passed": 19,
        "source_tree_sha256": "1" * 64,
        "amendment_sha256": "2" * 64,
        "repair_config_sha256": "2" * 64,
        "parent_manifest_sha256": "3" * 64,
        "scientific_config_sha256": "4" * 64,
        "replay_rng_audit_sha256": "5" * 64,
        "downstream_rng_reservation_sha256": "6" * 64,
        "downstream_rng_mapping_sha256": science.FORMAL_MAPPING_SHA256,
        "downstream_rng_mapping_count": science.FORMAL_MAPPING_COUNT,
        "source_snapshot_sha256": "7" * 64,
        "manifest_sha256": "8" * 64,
        "complete_sha256": "9" * 64,
        "completion_contract_sha256": "a" * 64,
        "files": {},
    }
    return science.GateBinding(
        **core,
        binding_sha256=science._canonical_sha256(core),
    )


def _fixture_uniforms(rows: int, columns: int) -> np.ndarray:
    values = np.arange(rows * columns, dtype=np.float64).reshape(rows, columns)
    return np.mod(values * 0.6180339887498948 + 0.123, 1.0)


def _summary_method_row(
    coverage: list[float],
    width: list[float],
) -> dict[str, object]:
    horizon = len(coverage)
    return {
        "selection_available": True,
        "target_coverage": coverage,
        "source_coverage": coverage,
        "target_normalized_width": width,
        "prefix_ess_fraction": [0.8] * horizon,
        "maximum_normalized_weight_share": [0.02] * horizon,
        "raw_log_weight_span": [1.0] * horizon,
    }


def test_frozen_config_and_full_rng_mapping_are_exact() -> None:
    config = _config()
    mapping = science.science_rng_mapping()

    assert config.rng.base_seeds == tuple(range(121_400, 121_600, 10))
    assert config.rng.bootstrap_seed == 12_140_019
    assert len(mapping) == len(set(mapping.values())) == 241
    assert science._canonical_sha256(mapping) == science.FORMAL_MAPPING_SHA256
    assert mapping["summary/bootstrap_complete_seed_matrix"] == 12_140_019
    assert set(mapping.values()).isdisjoint(science.REPAIR_REPLAY_IDS)
    assert set(mapping.values()).isdisjoint(science.OPTIONAL_PREFLIGHT_RESERVE)
    for seed in config.rng.base_seeds:
        worker = science.execution_rng_mapping(seed)
        assert set(worker.values()) == {
            value
            for label, value in mapping.items()
            if label.startswith(f"science/base_{seed}/")
        }


def test_runtime_environment_binds_scipy_version() -> None:
    environment = science._runtime_environment(("cuda:0", "cuda:1"))

    assert environment["scipy"] == science.scipy.__version__


def test_cli_exposes_no_scientific_knobs() -> None:
    with pytest.raises(SystemExit):
        science.main(["--gamma", "0"])
    with pytest.raises(SystemExit):
        science.main(["--output-root", "/tmp/not-allowed"])
    with pytest.raises(SystemExit):
        science.main(["--devices", "cpu"])


def test_policy_grid_adapter_equals_scalar_policy_and_preserves_cap() -> None:
    dgp = NativeSignedGammaDGPConfig()
    logging = NativeSignedGammaLoggingPolicy(dgp)
    scalar = NativeSignedGammaRadiusPolicy(logging)
    policy = science.NativePolicyGridAdapter(scalar)
    states = torch.tensor(
        [
            [-1.0, 0.5, 0.0, 0.0],
            [0.2, -0.4, 1.0, 0.5],
            [1.2, 0.7, 1.0, 0.8],
        ],
        dtype=torch.float64,
    )
    grid = torch.tensor([1.6, 1.7, 1.9, 2.1, 2.4], dtype=torch.float64)

    vectorized = policy.probabilities_for_grid(states, grid)
    scalar_loop = torch.stack(
        [policy.probabilities(states, radius) for radius in grid],
        dim=1,
    )
    reference = logging.probabilities(states)[:, None, :]

    assert torch.equal(vectorized, scalar_loop)
    assert torch.allclose(vectorized.sum(dim=2), torch.ones((3, 5), dtype=torch.float64))
    assert bool((vectorized > 0.0).all())
    assert float((vectorized / reference).max()) <= dgp.policy_ratio_cap + 1e-10


def test_exact_outcome_model_matches_native_monte_carlo_and_gamma_zero_placebo() -> None:
    dgp = NativeSignedGammaDGPConfig()
    kernel = NativeSignedGammaKernel(dgp, -4.0)
    model = science.NativeOutcomeModel(kernel, device="cpu")
    state = torch.tensor([[0.4, -0.3, 1.0, 0.25]], dtype=torch.float64)
    action = torch.tensor([2], dtype=torch.long)
    predicted_mean, predicted_scale = model(state, action)

    n = 80_000
    generator = torch.Generator().manual_seed(23)
    repeated_state = state.expand(n, -1)
    repeated_action = action.expand(n)
    _, outcomes, _ = kernel.step_from_noise(
        repeated_state,
        repeated_action,
        difficulty_uniform=torch.rand(n, generator=generator, dtype=torch.float64),
        tail_uniform=torch.rand(n, generator=generator, dtype=torch.float64),
        transition_normals=torch.randn((n, 2), generator=generator, dtype=torch.float64),
        outcome_normals=torch.randn((n, 2), generator=generator, dtype=torch.float64),
        time=3,
        horizon=12,
    )
    assert torch.allclose(outcomes.mean(dim=0), predicted_mean[0], atol=0.012, rtol=0.0)
    assert torch.allclose(outcomes.std(dim=0, unbiased=False), predicted_scale[0], atol=0.015, rtol=0.0)

    zero_model = science.NativeOutcomeModel(NativeSignedGammaKernel(dgp, 0.0), device="cpu")
    repeated = state.expand(3, -1)
    means, scales = zero_model(repeated, torch.tensor([0, 1, 2]))
    assert torch.equal(means[0], means[1]) and torch.equal(means[1], means[2])
    assert torch.equal(scales[0], scales[1]) and torch.equal(scales[1], scales[2])


def test_native_batch_and_online_callback_keep_exact_kernel_radius_and_seed() -> None:
    dgp = NativeSignedGammaDGPConfig()
    kernel = NativeSignedGammaKernel(dgp, -2.0)
    logging = NativeSignedGammaLoggingPolicy(dgp)
    target = science.NativePolicyGridAdapter(NativeSignedGammaRadiusPolicy(logging))
    schedule = torch.linspace(1.7, 2.1, 12, dtype=torch.float64)

    batch = science._native_online_rollout(
        science.NativeOnlineEnvironment(kernel),
        target,
        n=11,
        horizon=12,
        seed=29,
        device="cpu",
        q=schedule,
    )
    direct = rollout_native_signed_gamma(
        kernel,
        target.policy,
        make_native_signed_gamma_noise(n=11, horizon=12, seed=29, device="cpu"),
        radius=schedule,
    )

    assert torch.equal(batch.states, direct.states)
    assert torch.equal(batch.actions, direct.actions)
    assert torch.equal(batch.outcomes, direct.outcomes)
    assert torch.equal(batch.patient_ids, torch.arange(11))


def test_fixture_worker_calls_all_six_canonical_implementations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {name: 0 for name in ("standard", "aci", "mfcs", "spci", "prc", "scpcp")}
    names = {
        "standard_cp_stagewise_radii": "standard",
        "aci_style_controller": "aci",
        "finite_depth_mfcs_selection": "mfcs",
        "multidim_spci_style_controller": "spci",
        "prc_profile_scale": "prc",
        "select_marginal_prefix_schedule": "scpcp",
    }
    for attribute, label in names.items():
        original = getattr(science, attribute)

        def wrapped(*args: object, _original=original, _label=label, **kwargs: object):
            called[_label] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(science, attribute, wrapped)

    rows = science.run_seed(
        _config(),
        17,
        device="cpu",
        budget=science.ExecutionBudget(
            calibration=36,
            grid=24,
            reference=48,
            online=9,
            q_grid_size=7,
        ),
        formal=False,
    )

    assert len(rows) == 5
    assert [row["gamma"] for row in rows] == list(science.GAMMAS)
    assert set(rows[0]["methods"]) == set(science.METHODS)
    assert called == {name: 5 for name in called}
    for row in rows:
        for method in science.ADAPTIVE_METHODS:
            assert row["methods"][method]["target_adaptation_trajectories"] == 9


def test_wsc_is_minimum_stage_after_seed_mean_not_mean_seed_minima() -> None:
    rows = [
        {"methods": {"SC-PCP": _summary_method_row([1.0, 0.8], [2.0, 2.0])}},
        {"methods": {"SC-PCP": _summary_method_row([0.8, 1.0], [2.2, 1.8])}},
    ]
    result = science._summarize_method(
        "SC-PCP",
        rows,
        bootstrap_uniforms=_fixture_uniforms(200, 2),
        horizon=2,
    )

    assert result["target_coverage_by_stage"] == pytest.approx([0.9, 0.9])
    assert result["target_mean_coverage"] == pytest.approx(0.9)
    assert result["target_marginal_worst_coverage"] == pytest.approx(0.9)
    assert np.mean([min(1.0, 0.8), min(0.8, 1.0)]) == pytest.approx(0.8)
    assert result["target_marginal_worst_coverage"] != pytest.approx(0.8)


def test_pointwise_and_mean_intervals_are_student_t_not_bootstrap() -> None:
    coverage = ([0.60, 0.70], [0.80, 0.90], [1.00, 0.95])
    widths = ([1.0, 2.0], [2.0, 4.0], [4.0, 8.0])
    rows = [
        {"methods": {"SC-PCP": _summary_method_row(list(cov), list(width))}}
        for cov, width in zip(coverage, widths, strict=True)
    ]
    uniforms = _fixture_uniforms(500, 3)
    result = science._summarize_method(
        "SC-PCP",
        rows,
        bootstrap_uniforms=uniforms,
        horizon=2,
    )

    coverage_array = np.asarray(coverage, dtype=np.float64)
    width_array = np.asarray(widths, dtype=np.float64)
    expected_stage_coverage = [
        science._student_t_interval(coverage_array[:, stage]) for stage in range(2)
    ]
    expected_stage_width = [
        science._student_t_interval(width_array[:, stage]) for stage in range(2)
    ]
    assert np.asarray(result["target_coverage_ci95_by_stage"]) == pytest.approx(
        np.asarray(expected_stage_coverage)
    )
    assert np.asarray(result["target_normalized_width_ci95_by_stage"]) == pytest.approx(
        np.asarray(expected_stage_width)
    )
    assert result["target_mean_coverage_ci95"] == pytest.approx(
        science._student_t_interval(coverage_array.mean(axis=1))
    )
    assert result["mean_target_normalized_width_ci95"] == pytest.approx(
        science._student_t_interval(width_array.mean(axis=1))
    )

    bootstrap = science._bootstrap_indices(uniforms, 3)
    bootstrap_stage_zero = science._percentile_interval(
        coverage_array[bootstrap].mean(axis=1)[:, 0]
    )
    assert result["target_coverage_ci95_by_stage"][0] != pytest.approx(
        bootstrap_stage_zero
    )
    # WSC remains the complete-seed-vector percentile bootstrap.
    expected_wsc = science._percentile_interval(
        coverage_array[bootstrap].mean(axis=1).min(axis=1)
    )
    assert result["target_wsc_ci95"] == pytest.approx(expected_wsc)


def test_bootstrap_projection_uses_complete_seed_matrix() -> None:
    uniforms = _fixture_uniforms(100, 20)
    full = science._bootstrap_indices(uniforms, 20)
    selected = science._bootstrap_indices(uniforms, 7)

    assert full.shape == (100, 20)
    assert selected.shape == (100, 7)
    assert np.array_equal(selected, np.floor(uniforms[:, :7] * 7).astype(np.int64))
    with pytest.raises(ValueError):
        science._bootstrap_indices(uniforms, 21)


def test_fixture_worker_rejects_any_protected_derived_rng_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = next(iter(science.science_rng_mapping().values()))
    monkeypatch.setattr(
        science,
        "execution_rng_mapping",
        lambda seed: {"task": seed, "calibration": protected},
    )
    with pytest.raises(ValueError, match="protected Native RNG"):
        science.run_seed(
            _config(),
            31,
            device="cpu",
            budget=science.ExecutionBudget(12, 8, 12, 3, 5),
            formal=False,
        )


def test_live_rng_audit_accepts_empty_inventory_and_rejects_actual_collision(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "results"
    source_root = tmp_path / "source"
    artifact_root.mkdir()
    for name in ("scripts", "src", "tools"):
        (source_root / name).mkdir(parents=True)
    output_root = artifact_root / "science"
    config = _config()
    gate = _gate_binding()

    audit = science.audit_science_rng_ids(
        config,
        gate_binding=gate,
        output_root=output_root,
        artifact_root=artifact_root,
        source_root=source_root,
        config_path=CONFIG_PATH.resolve(),
    )
    assert audit["status"] == "passed_before_launch"
    assert audit["collision_count"] == 0

    collision = artifact_root / "seed_121400.json"
    collision.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="prior-use collisions"):
        science.audit_science_rng_ids(
            config,
            gate_binding=gate,
            output_root=output_root,
            artifact_root=artifact_root,
            source_root=source_root,
            config_path=CONFIG_PATH.resolve(),
        )


def test_strict_json_and_method_schema_reject_tampering(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"target_coverage": NaN}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="strict JSON"):
        science._read_json(bad)

    config = _config()
    row = {
        "selection_available": False,
        "selection_status": "UNAVAILABLE_NO_FEASIBLE_CANDIDATE",
        "information_regime": "offline_logged_data",
        "target_adaptation_trajectories": 0,
        "radii": [],
        "stage_coverage": [0.9] * 12,
    }
    with pytest.raises(RuntimeError, match="fields differ"):
        science._validate_method_row("SC-PCP", row, config=config)


def test_repair_gate_binding_uses_public_completion_contract_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    repair_root = tmp_path / config.parent.repair_root
    repair_root.mkdir(parents=True)
    runner_path = tmp_path / config.parent.repair_runner
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("# fixture repair API\n", encoding="utf-8")
    amendment_path = tmp_path / config.parent.repair_config
    amendment_path.parent.mkdir(parents=True)
    amendment = yaml.safe_load(
        Path("configs/native_synthetic_signed_gamma_time_coordinate_repair_r1.yaml").read_text()
    )
    amendment_path.write_text(yaml.safe_dump(amendment, sort_keys=False), encoding="utf-8")
    source_hash = "b" * 64
    native_config = NativeSignedGammaBenchmarkConfig.from_yaml(
        Path("configs/native_synthetic_signed_gamma.yaml")
    ).to_dict()
    metadata = {
        "protocol": "native_synthetic_signed_gamma_time_coordinate_repair_r1",
        "role": config.parent.administrative_role,
        "scientific_config": native_config,
        "source_tree_sha256": source_hash,
        "amendment_sha256": science._file_sha256(amendment_path),
    }
    summary = {
        "status": "GO",
        "downstream_authorized": True,
        "n_prespecified": 20,
        "n_exact_replays": 20,
        "n_repaired_fields_valid": 20,
        "n_passed": 20,
        "passed_rng_ids": list(science.REPAIR_REPLAY_IDS),
        "required_passed_rng_ids": 19,
    }
    complete = {"decision": "GO", "downstream_authorized": True}
    for name, payload in (
        ("metadata.json", metadata),
        ("summary.json", summary),
        ("COMPLETE", complete),
        ("manifest.json", {"fixture": True}),
    ):
        (repair_root / name).write_text(json.dumps(payload) + "\n", encoding="utf-8")

    contract = {
        "protocol": "native_synthetic_signed_gamma_time_coordinate_repair_r1",
        "role": config.parent.administrative_role,
        "output_root": str(repair_root.resolve()),
        "decision": "GO",
        "downstream_authorized": True,
        "amendment_sha256": science._file_sha256(amendment_path),
        "parent_manifest_sha256": "1" * 64,
        "scientific_config_sha256": science._canonical_sha256(native_config),
        "replay_rng_audit_sha256": "2" * 64,
        "downstream_rng_reservation_sha256": "3" * 64,
        "reserved_rng_mapping": science.science_rng_mapping(),
        "reserved_rng_mapping_sha256": science.FORMAL_MAPPING_SHA256,
        "source_tree_sha256": source_hash,
        "source_snapshot_sha256": "4" * 64,
        "manifest_sha256": science._file_sha256(repair_root / "manifest.json"),
        "complete_sha256": science._file_sha256(repair_root / "COMPLETE"),
        "metadata_sha256": science._file_sha256(repair_root / "metadata.json"),
        "summary_sha256": science._file_sha256(repair_root / "summary.json"),
    }
    contract["completion_contract_sha256"] = science._canonical_sha256(contract)
    fake_module = SimpleNamespace(
        validate_completed_repair_bundle=lambda *args, **kwargs: contract
    )
    monkeypatch.setattr(science, "_load_repair_module", lambda path: fake_module)
    monkeypatch.setattr(science.preflight, "_experiment_tree_sha256", lambda path: source_hash)

    binding = science.verify_repair_gate(config, source_root=tmp_path)
    assert binding.decision == "GO"
    assert binding.downstream_rng_mapping_count == 241
    assert binding.completion_contract_sha256 == contract["completion_contract_sha256"]

    summary["status"] = "NO_GO"
    (repair_root / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="decision is not GO"):
        science.verify_repair_gate(config, source_root=tmp_path)


def test_gamma_roles_and_information_budgets_are_frozen() -> None:
    config = _config()
    assert config.design.primary_gamma == -4.0
    assert config.design.gammas == (-4.0, -2.0, 0.0, 2.0, 4.0)
    assert config.design.methods == science.METHODS
    assert science.TARGET_ADAPTATION_BUDGET == {
        "Standard CP": 0,
        "ACI": 2_000,
        "MFCS": 0,
        "SPCI": 2_000,
        "PRC": 2_000,
        "SC-PCP": 0,
    }
    assert science.SCIENCE_CONTRACT["coverage_metric"] == (
        "min_t mean_seed(target_coverage_seed_t)"
    )
    assert "independent method streams reused across gammas" in science.SCIENCE_CONTRACT[
        "common_random_numbers"
    ]
    intervals = science.SCIENCE_CONTRACT["uncertainty_intervals"]
    assert intervals["pointwise_stage_coverage"].startswith("Student-t")
    assert intervals["pointwise_stage_width"].startswith("Student-t")
    assert intervals["mean_coverage"].startswith("Student-t")
    assert intervals["mean_width"].startswith("Student-t")
    assert intervals["wsc"] == "complete-seed-vector percentile bootstrap"
    assert intervals["selection_rate"].startswith("Wilson")


def test_science_dependency_provenance_binds_simulator_and_fails_closed() -> None:
    dependencies = science._dependency_files(
        _config(),
        config_path=CONFIG_PATH.resolve(),
    )

    assert dependencies["simulator"]["path"] == "src/scpcp/simulator.py"
    science.preflight._verify_dependency_files(dependencies, ROOT)

    missing = dict(dependencies)
    missing.pop("simulator")
    with pytest.raises(RuntimeError, match="simulator.py is missing"):
        science.preflight._verify_dependency_files(missing, ROOT)

    tampered = {name: dict(contract) for name, contract in dependencies.items()}
    tampered["simulator"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="active dependency differs: src/scpcp/simulator.py"):
        science.preflight._verify_dependency_files(tampered, ROOT)


def test_finalize_removes_complete_when_public_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "science"
    root.mkdir()
    metadata = {
        "protocol": science.PROTOCOL,
        "science_config_payload_sha256": "1" * 64,
        "gate_binding": {"binding_sha256": "2" * 64},
        "rng_audit": {"audit_sha256": "3" * 64},
        "source_tree_sha256": "4" * 64,
    }
    events: list[str] = []

    monkeypatch.setattr(science, "_expected_artifact_paths", lambda *args: set())
    monkeypatch.setattr(
        science,
        "_expected_complete_payload",
        lambda *args, **kwargs: {"status": "complete"},
    )

    def validate_nonterminal(
        selected_root: Path,
        *,
        expected_metadata: object,
        source_root: Path,
        include_complete: bool,
    ) -> tuple[dict[str, object], str]:
        assert selected_root == root
        assert expected_metadata is metadata
        assert source_root == ROOT
        assert include_complete is False
        assert (root / "manifest.json").is_file()
        assert not (root / "COMPLETE").exists()
        events.append("precommit")
        return metadata, science._file_sha256(root / "manifest.json")

    def fail_public_validation(
        selected_root: Path,
        *,
        expected_metadata: object,
    ) -> None:
        assert selected_root == root
        assert expected_metadata is metadata
        assert (root / "COMPLETE").is_file()
        events.append("public")
        raise RuntimeError("forced completed validation failure")

    monkeypatch.setattr(science, "_validate_bundle_contents", validate_nonterminal)
    monkeypatch.setattr(science, "validate_completed_bundle", fail_public_validation)

    with pytest.raises(RuntimeError, match="forced completed validation failure"):
        science._finalize_root(root, metadata=metadata, config=_config())

    assert events == ["precommit", "public"]
    assert (root / "manifest.json").is_file()
    assert not (root / "COMPLETE").exists()


def test_clipped_normal_moments_handle_degenerate_tail_limits() -> None:
    mean = torch.tensor([[-20.0, 0.0, 20.0]], dtype=torch.float64)
    scale = torch.ones_like(mean)
    first, second = science._clipped_normal_moments(mean, scale, lower=-8.0, upper=8.0)

    assert torch.allclose(first, torch.tensor([[-8.0, 0.0, 8.0]], dtype=torch.float64), atol=1e-12)
    assert second[0, 0] == pytest.approx(64.0)
    assert second[0, 2] == pytest.approx(64.0)
    assert second[0, 1] == pytest.approx(1.0, abs=1e-12)
