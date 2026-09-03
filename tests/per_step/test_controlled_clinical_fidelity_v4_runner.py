from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from scpcp.controlled_clinical_fidelity_v4 import (
    DATASETS,
    FROZEN_ANCHORS,
    REPAIR_DATASETS,
    load_fidelity_v4_config,
    repair_candidates,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/controlled_clinical_fidelity_v4.yaml"


def _load_runner():
    path = ROOT / "scripts/run_controlled_clinical_fidelity_v4.py"
    spec = importlib.util.spec_from_file_location(
        "run_controlled_clinical_fidelity_v4",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _metric_payload(ratio: float, *, structural: bool = True) -> dict[str, object]:
    return {
        "maximum_score_ks": 0.10 * ratio,
        "maximum_signed_residual_w1": 0.25 * ratio,
        "maximum_successor_mean_w1": 0.25 * ratio,
        "maximum_successor_q95_w1": 0.50 * ratio,
        "structural_invariants": structural,
    }


def _candidate_rows(dataset: str, *, structural: bool) -> list[dict[str, object]]:
    candidates = repair_candidates(dataset)
    return [
        {
            "candidates": [
                {
                    "theta": theta.to_dict(),
                    "metrics": _metric_payload(0.8, structural=structural),
                }
                for theta in candidates
            ]
        }
        for _ in range(20)
    ]


def test_live_parent_and_rng_audits_match_the_frozen_contract() -> None:
    config = load_fidelity_v4_config(CONFIG_PATH)

    development = runner.audit_development_reuse(config)
    confirmation = runner.audit_confirmation_rng(
        config,
        excluded_roots=(runner.DEVELOPMENT_ROOT, runner.CONFIRMATION_ROOT),
    )

    assert development["status"] == "passed_before_launch"
    assert development["verified_parent_seed_envelope_sha256"] == (
        runner.EXPECTED_PARENT_SEED_ENVELOPE_SHA256
    )
    assert development["authorized_lineage_collision_count"] == 180
    assert development["missing_lineage_collision_count"] == 0
    assert development["unauthorized_collision_count"] == 0
    assert confirmation["status"] == "passed_before_launch"
    assert confirmation["new_rng_stream_count"] == 1304
    assert confirmation["collision_count"] == 0

    failed = runner._failed_confirmation_attempt_binding(config)
    retry = runner.audit_confirmation_retry_rng(
        config,
        failed_attempt_binding=failed,
    )
    assert failed["inventory_sha256"] == (
        runner.EXPECTED_FAILED_CONFIRMATION_INVENTORY_SHA256
    )
    assert failed["support_seed_artifact_count"] == 20
    assert failed["k0_artifact_count"] == 0
    assert retry["authorized_declared_lineage_collision_count"] == 1304
    assert retry["failed_attempt_executed_support_artifact_count"] == 20
    assert retry["failed_attempt_k0_artifact_count"] == 0
    assert retry["unauthorized_collision_count"] == 0
    assert retry["second_fresh_bank_claimed"] is False


def test_context_builder_passes_every_dataset_specific_mode_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _EnvironmentBatch:
        actions = torch.zeros((2, 1), dtype=torch.long)
        outcomes = torch.zeros((2, 1, 2))

        @staticmethod
        def current_states() -> torch.Tensor:
            return torch.zeros((2, 1, 4))

    def fake_environment(batch: object, **kwargs: object) -> object:
        captured["batch"] = batch
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runner, "ControlledResidualEnvironment", fake_environment)
    monkeypatch.setattr(runner, "score_batch", lambda *args: torch.zeros((2, 1)))
    monkeypatch.setattr(
        runner.v2,
        "_empirical_rank_by_stage",
        lambda scores: torch.zeros_like(scores),
    )
    base = runner.v2.ExtensionContext(
        config=SimpleNamespace(model=SimpleNamespace(history_length=2)),
        splits=SimpleNamespace(environment=_EnvironmentBatch()),
        n_actions=2,
        static_indices=(0,),
        action_costs=(0.0, 1.0),
        action_mapping={0: 0, 1: 1},
        state_feature_names=("a", "b"),
        outcome_model=object(),
        region=object(),
        logging_policy=object(),
        target_policy=object(),
        environment=object(),
        action_coordinate=torch.tensor([-1.0, 1.0]),
        outcome_sd=torch.ones(2),
        q_low=0.0,
        q_high=1.0,
    )
    theta = next(
        candidate
        for candidate in repair_candidates("mimic_cxr")
        if candidate.metric == "stagewise_zscore"
        and candidate.neighbors == 10_000
        and candidate.weight == "gaussian_b2"
        and candidate.transition_mode == "local_delta"
        and candidate.outcome_residual_mode == "raw"
    )

    runner._context_with_theta(base, theta)

    assert captured["neighbors"] == 10_000
    assert captured["bandwidth"] == 2.0
    assert captured["ridge"] == 1e-3
    assert captured["representation_geometry"] == "stagewise_zscore"
    assert captured["donor_weighting"] == "gaussian"
    assert captured["ridge_mode"] == "sample_normalized_no_intercept"
    assert captured["transition_mode"] == "local_delta"
    assert captured["outcome_residual_mode"] == "raw"


@pytest.mark.parametrize("dataset", ("mimic_iv", "inspire"))
def test_context_builder_deserializes_both_frozen_anchor_kernel_fields(
    dataset: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _EnvironmentBatch:
        actions = torch.zeros((2, 1), dtype=torch.long)
        outcomes = torch.zeros((2, 1, 2))

        @staticmethod
        def current_states() -> torch.Tensor:
            return torch.zeros((2, 1, 4))

    def fake_environment(batch: object, **kwargs: object) -> object:
        captured["batch"] = batch
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runner, "ControlledResidualEnvironment", fake_environment)
    monkeypatch.setattr(runner, "score_batch", lambda *args: torch.zeros((2, 1)))
    monkeypatch.setattr(
        runner.v2,
        "_empirical_rank_by_stage",
        lambda scores: torch.zeros_like(scores),
    )
    base = runner.v2.ExtensionContext(
        config=SimpleNamespace(model=SimpleNamespace(history_length=2)),
        splits=SimpleNamespace(environment=_EnvironmentBatch()),
        n_actions=2,
        static_indices=(0,),
        action_costs=(0.0, 1.0),
        action_mapping={0: 0, 1: 1},
        state_feature_names=("a", "b"),
        outcome_model=object(),
        region=object(),
        logging_policy=object(),
        target_policy=object(),
        environment=object(),
        action_coordinate=torch.tensor([-1.0, 1.0]),
        outcome_sd=torch.ones(2),
        q_low=0.0,
        q_high=1.0,
    )
    anchor = FROZEN_ANCHORS[dataset]

    runner._context_with_theta(base, anchor)

    assert not hasattr(anchor, "bandwidth")
    assert not hasattr(anchor, "donor_weighting")
    assert captured["bandwidth"] == anchor.to_dict()["bandwidth"] == 2.0
    assert captured["donor_weighting"] == (
        anchor.to_dict()["donor_weighting"]
    ) == "gaussian"
    assert captured["ridge"] == anchor.ridge_value
    assert captured["ridge_mode"] == anchor.ridge_mode
    assert captured["transition_mode"] == anchor.transition_mode
    assert captured["outcome_residual_mode"] == anchor.outcome_residual_mode


def test_context_identity_records_actual_full_cell_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.v2,
        "_context_identity",
        lambda context: {
            "combined_sha256": "a" * 64,
            "outcome_model_state_sha256": "b" * 64,
            "behavior_policy_state_sha256": "c" * 64,
            "split_patient_id_sha256": {
                "predictor": "d" * 64,
                "fidelity": "e" * 64,
                "environment": "f" * 64,
            },
            "active_config_sha256": "1" * 64,
        },
    )
    theta = next(
        candidate
        for candidate in repair_candidates("eicu")
        if candidate.neighbors == 10_000 and candidate.weight == "gaussian_b2"
    )
    transforms = {
        stage: (torch.zeros(32, dtype=torch.float64), torch.ones(32, dtype=torch.float64))
        for stage in range(2)
    }
    sizes = ((23, 41), (37, 29))
    libraries = {
        (stage, action): (torch.zeros((sizes[stage][action], 32)),)
        for stage in range(2)
        for action in range(2)
    }
    environment = SimpleNamespace(
        horizon=2,
        _metric_transforms=transforms,
        _libraries=libraries,
    )

    identity = runner._candidate_context_identity(
        SimpleNamespace(n_actions=2),
        environment,
        theta,
    )
    support = identity["library_support"]

    assert support["actual_library_sizes_by_stage_action"] == [[23, 41], [37, 29]]
    assert support["effective_neighbor_counts_by_stage_action"] == [[23, 41], [37, 29]]
    assert support["full_cell_verified"] is True
    runner._validate_library_support(
        support,
        theta=theta,
        preset=SimpleNamespace(horizon=2),
    )
    split_audit = {
        "role_patient_id_sha256": {
            "predictor": "d" * 64,
            "fidelity": "e" * 64,
            "environment": "f" * 64,
        }
    }
    runner._validate_context_identity(
        identity,
        theta=theta,
        preset=SimpleNamespace(horizon=2),
        split_audit=split_audit,
    )
    tampered = {**identity, "base_nuisance_context_sha256": "not-a-sha"}
    tampered["combined_sha256"] = runner._json_sha256(
        {key: value for key, value in tampered.items() if key != "combined_sha256"}
    )
    with pytest.raises(RuntimeError, match="metric-transform identity"):
        runner._validate_context_identity(
            tampered,
            theta=theta,
            preset=SimpleNamespace(horizon=2),
            split_audit=split_audit,
        )
    wrong_split = {
        "role_patient_id_sha256": {
            **split_audit["role_patient_id_sha256"],
            "environment": "2" * 64,
        }
    }
    with pytest.raises(RuntimeError, match="metric-transform identity"):
        runner._validate_context_identity(
            identity,
            theta=theta,
            preset=SimpleNamespace(horizon=2),
            split_audit=wrong_split,
        )


def test_development_and_confirmation_decisions_are_dataset_independent() -> None:
    parent_status = json.loads(
        (
            ROOT
            / "results/work/controlled_clinical_fidelity_v3_development/FINAL_STATUS.json"
        ).read_text()
    )
    rows = {
        "eicu": _candidate_rows("eicu", structural=True),
        "mimic_cxr": _candidate_rows("mimic_cxr", structural=False),
    }

    decision, _ = runner._development_decision(rows, parent_status)

    assert decision["cross_dataset_conjunction_used"] is False
    assert decision["development_go_datasets"] == ["mimic_iv", "eicu", "inspire"]
    assert decision["development_no_go_datasets"] == ["mimic_cxr"]
    assert "mimic_cxr" not in decision["theta_by_dataset"]
    gates = {
        dataset: runner._unopened_confirmation_gate(dataset) for dataset in DATASETS
    }
    gates["mimic_iv"] = {
        **gates["mimic_iv"],
        "status": "CONFIRMATION_GATE_GO",
        "development_admissible": True,
        "confirmation_opened": True,
    }
    final = runner._confirmation_final(gates)
    assert final["status"] == "CONFIRMATION_COMPLETE_DATASET_INDEPENDENT"
    assert final["confirmed_datasets"] == ["mimic_iv"]
    assert final["cross_dataset_conjunction_used"] is False


def test_manifest_binds_nested_complete_and_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    nested = root / "repair/eicu"
    nested.mkdir(parents=True)
    (root / "metadata.json").write_text("{}")
    (nested / "COMPLETE").write_text("complete\n")

    runner._write_manifest(root)
    manifest = json.loads((root / "manifest.json").read_text())

    assert "repair/eicu/COMPLETE" in {
        entry["path"] for entry in manifest["artifacts"]
    }
    runner._verify_manifest(root)
    manifest["diagnostics"] = {"mean_width": 1.0}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="manifest header"):
        runner._verify_manifest(root)
    runner._write_manifest(root)
    duplicate = json.loads((root / "manifest.json").read_text())
    duplicate["artifacts"].append(duplicate["artifacts"][0])
    duplicate["artifact_count"] += 1
    (root / "manifest.json").write_text(json.dumps(duplicate))
    with pytest.raises(RuntimeError, match="duplicate v4 manifest entry"):
        runner._verify_manifest(root)
    (root / "diagnostics.json").write_text('{"mean_width": 1.0}')
    runner._write_manifest(root)
    runner._verify_manifest(root)
    with pytest.raises(RuntimeError, match="exact artifact file set"):
        runner._assert_exact_artifact_file_set(
            root,
            {Path("metadata.json"), Path("repair/eicu/COMPLETE")},
        )
    (root / "diagnostics.json").unlink()
    (root / "unsafe").symlink_to(root / "metadata.json")
    with pytest.raises(RuntimeError, match="symlink"):
        runner._write_manifest(root)


@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt))
def test_invalid_completed_resume_unlinks_stale_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "COMPLETE").write_text("stale\n")

    def fail(*args: object, **kwargs: object) -> None:
        raise error_type("invalid bundle")

    monkeypatch.setattr(runner, "_validate_root_bundle", fail)
    with pytest.raises(error_type):
        runner._complete_and_valid(root, {}, config=object())
    assert not (root / "COMPLETE").exists()


def test_result_firewall_allows_only_theta_bandwidth() -> None:
    runner._validate_seed_payload_firewall(
        {
            "result": {
                "coverage_generated": False,
                "candidates": [{"theta": {"bandwidth": 2.0}}],
            }
        }
    )
    with pytest.raises(RuntimeError, match="coverage firewall"):
        runner._validate_seed_payload_firewall(
            {
                "result": {
                    "coverage_generated": False,
                    "kernel": {"bandwidth": 2.0},
                }
            }
        )


def test_k0_replay_semantics_are_value_bound() -> None:
    detail = {
        "label": "logging-mixture one-step fidelity",
        "episode_weighted": True,
        "inference_unit": (
            "patient-disjoint episode query; M=16 quadrature, never 16N "
            "independent observations"
        ),
        "patient_chunk_size": 128,
        "expansion_formula": "u[t,i,m]=(U[t,i]+(m+0.5)/16) mod 1",
        "flatten_order": "stage, patient, systematic_offset (offset fastest)",
    }
    runner._validate_systematic_replay_header(detail)
    detail["inference_unit"] = "16N independent observations"
    with pytest.raises(RuntimeError, match="replay semantics"):
        runner._validate_systematic_replay_header(detail)
    with pytest.raises(RuntimeError, match="coverage firewall"):
        runner._validate_seed_payload_firewall(
            {
                "result": {
                    "coverage_generated": False,
                    "nested": {"mean_width": 1.0},
                }
            }
        )


def test_retry_source_delta_is_exactly_one_runner_file() -> None:
    config = load_fidelity_v4_config(CONFIG_PATH)
    parent_binding = runner._validated_parent_v3_binding(config)
    development_binding, frozen = runner._verify_development_for_confirmation(
        runner.DEVELOPMENT_ROOT,
        config=config,
        current_parent_binding=parent_binding,
    )
    source_hash, source_snapshot = runner._active_source_contract()
    delta = runner._administrative_source_delta(
        source_hash=source_hash,
        source_snapshot=source_snapshot,
        development_binding=development_binding,
    )

    assert delta["changed_file_count"] == 1
    assert delta["changed_files"][0]["path"] == runner.RETRY_CHANGED_SOURCE_PATH
    assert delta["changed_files"][0]["before_sha256"] == (
        runner.EXPECTED_DEVELOPMENT_RUNNER_SHA256
    )
    assert delta["source_file_count_before"] == 113
    assert delta["source_file_count_after"] == 113
    assert delta["config_files_changed"] is False
    assert delta["method_modules_changed"] is False
    assert delta["repair_grid_or_selector_changed"] is False
    assert frozen["development_source_tree_sha256"] == (
        runner.EXPECTED_DEVELOPMENT_SOURCE_TREE_SHA256
    )


def test_retry_support_comparison_allows_only_new_source_provenance(
    tmp_path: Path,
) -> None:
    config = load_fidelity_v4_config(CONFIG_PATH)
    failed = runner._failed_confirmation_attempt_binding(config)
    retry_root = tmp_path / "retry"
    new_support = retry_root / "mimic_iv/support"
    new_support.mkdir(parents=True)
    mimic_rows = []
    for seed in config.confirmation_seeds["mimic_iv"]:
        old_path = runner.CONFIRMATION_ROOT / f"mimic_iv/support/seed_{seed:06d}.json"
        payload = json.loads(old_path.read_text())
        payload["source_tree_sha256"] = "f" * 64
        new_path = new_support / f"seed_{seed:06d}.json"
        new_path.write_text(json.dumps(payload))
        mimic_rows.append(payload["result"])
    support = {
        "mimic_iv": mimic_rows,
        "eicu": [dict(seed=index, passed=True) for index in range(20)],
        "inspire": [dict(seed=index, passed=True) for index in range(20)],
    }

    verification = runner._verify_retry_support_replay(
        retry_root,
        support,
        config=config,
        failed_attempt_binding=failed,
    )

    assert verification["mimic_support_result_exact_equal_count"] == 20
    assert verification["top_level_allowed_delta_keys"] == ["source_tree_sha256"]
    assert verification["failed_attempt_artifacts_reused"] is False
    assert verification["second_fresh_bank_claimed"] is False
    tampered_path = new_support / "seed_115000.json"
    tampered = json.loads(tampered_path.read_text())
    tampered["device"] = "cuda:9"
    tampered_path.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="support semantics"):
        runner._verify_retry_support_replay(
            retry_root,
            support,
            config=config,
            failed_attempt_binding=failed,
        )


def test_formal_v4_retry_root_is_absent_before_launch() -> None:
    assert (runner.DEVELOPMENT_ROOT / "COMPLETE").is_file()
    assert runner.CONFIRMATION_ROOT.is_dir()
    assert not (runner.CONFIRMATION_ROOT / "COMPLETE").exists()
    assert not runner.CONFIRMATION_RETRY_ROOT.exists()
    assert tuple(repair_candidates(dataset) for dataset in REPAIR_DATASETS)
