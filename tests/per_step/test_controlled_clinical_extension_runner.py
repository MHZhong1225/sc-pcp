from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from scpcp.data import TrajectoryBatch


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_controlled_clinical_extension.py"
    name = "test_run_controlled_clinical_extension"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _split_audit() -> dict[str, object]:
    roles = ("predictor", "fidelity", "environment")
    return {
        "patient_sets_pairwise_disjoint": True,
        "role_patient_id_sha256": {role: "a" * 64 for role in roles},
        "role_unique_patient_counts": {role: 20 for role in roles},
        "role_episode_counts": {role: 20 for role in roles},
    }


def _episode_summary() -> dict[str, object]:
    return {
        "episode_row_count": 20,
        "unique_patient_count": 20,
        "maximum_episodes_per_patient": 1,
        "duplicate_episode_rate": 0.0,
        "donor_contract": (
            "episode-weighted transition; patient-aggregated overlap diagnostics"
        ),
    }


def _k0_invariant_row() -> dict[str, bool]:
    return {
        "passed": True,
        "rolling_history_exact": True,
        "static_coordinates_exact": True,
        "cumulative_coordinates_monotone": True,
        "decision_time_exact": True,
        "finite": True,
        "row_kernel_ess_at_least_one": True,
        "row_kernel_probability_in_unit_interval": True,
    }


def _context_identity(runner) -> dict[str, object]:
    split_hashes = {
        "predictor": "a" * 64,
        "fidelity": "a" * 64,
        "environment": "a" * 64,
    }
    identity = {
        "outcome_model_state_sha256": "e" * 64,
        "behavior_policy_state_sha256": "f" * 64,
        "q_low": 1.0,
        "q_high": 2.0,
        "n_actions": 2,
        "action_mapping": {"0": 0, "1": 1},
        "action_costs": [0.0, 1.0],
        "donor_neighbors": 100,
        "donor_bandwidth": 2.0,
        "transition_ridge": 1e-3,
        "environment_patient_id_sha256": split_hashes["environment"],
        "split_patient_id_sha256": split_hashes,
        "active_config_sha256": "1" * 64,
    }
    return {**identity, "combined_sha256": runner._json_sha256(identity)}


def _overlap_probe(
    runner,
    *,
    fraction: float,
    radius: float,
    metrics: dict[str, float],
) -> dict[str, object]:
    gate = runner.load_extension_config(runner.CONFIG_PATH).donor_overlap_gate
    resolved = runner.DonorOverlapMetrics(**metrics)
    return {
        "radius_fraction": fraction,
        "radius": radius,
        "metrics": metrics,
        "passed": runner.donor_overlap_passes(resolved, gate),
        "target_simplex_maximum_error": 0.0,
        "logging_simplex_maximum_error": 0.0,
        "minimum_logging_probability": 0.1,
        "minimum_target_probability": 0.05,
        "policy_probabilities_finite": True,
        "maximum_single_step_target_to_logging_ratio": 2.0,
        "single_step_ratio_cap": 3.0,
        "local_unique_k_minimum": 20.0,
        "prefix_overlap_report_only": {},
    }


def _phase_payload(
    runner,
    *,
    preset,
    seed: int,
    phase: str,
    result: dict[str, object],
) -> tuple[dict[str, object], dict[str, str]]:
    contract = {
        "dataset_contract_sha256": "b" * 64,
        "source_tree_sha256": "c" * 64,
        "rng_stream_mapping_sha256": "d" * 64,
    }
    payload = {
        "protocol": runner.PROTOCOL,
        "dataset": preset.name,
        "phase": phase,
        "seed": seed,
        "device": "cuda:0",
        **contract,
        "result": {
            "seed": seed,
            "dataset": preset.name,
            "phase": phase,
            **result,
        },
    }
    return payload, contract


def _method_row(runner, method: str, horizon: int) -> dict[str, object]:
    return {
        "target_adaptation_trajectories": runner.TARGET_ADAPTATION_BUDGET[method],
        "information_regime": runner.INFORMATION_REGIME[method],
        "selection_available": True,
        "radii": [1.0] * horizon,
        "source_coverage": [0.9] * horizon,
        "target_coverage": [0.9] * horizon,
        "coverage_gap": [0.0] * horizon,
        "target_normalized_width": [2.0] * horizon,
        "prefix_ess_fraction": [0.5] * horizon,
        "maximum_normalized_weight_share": [0.01] * horizon,
    }


def _summary_rows(runner, preset, selected_seeds: tuple[int, ...]) -> list[dict[str, object]]:
    widths = {
        "Standard CP": 3.0,
        "ACI": 2.5,
        "MFCS": 1.0,
        "SPCI": 2.0,
        "PRC": 1.8,
        "SC-PCP": 1.5,
    }
    rows = []
    for gamma in runner.GAMMAS:
        for index, seed in enumerate(selected_seeds):
            coverage = (
                [0.82, 1.0, *([0.95] * (preset.horizon - 2))]
                if index < 9
                else [1.0, 0.82, *([0.95] * (preset.horizon - 2))]
            )
            methods = {}
            for method in runner.METHODS:
                if method == "MFCS" and index == 0:
                    methods[method] = {
                        "target_adaptation_trajectories": (
                            runner.TARGET_ADAPTATION_BUDGET[method]
                        ),
                        "information_regime": runner.INFORMATION_REGIME[method],
                        "selection_available": False,
                        "radii": [],
                    }
                    continue
                methods[method] = {
                    "target_adaptation_trajectories": (
                        runner.TARGET_ADAPTATION_BUDGET[method]
                    ),
                    "information_regime": runner.INFORMATION_REGIME[method],
                    "selection_available": True,
                    "radii": [1.0] * preset.horizon,
                    "source_coverage": [0.91] * preset.horizon,
                    "target_coverage": coverage,
                    "coverage_gap": [value - 0.91 for value in coverage],
                    "target_normalized_width": [widths[method]] * preset.horizon,
                    "prefix_ess_fraction": [0.5] * preset.horizon,
                    "maximum_normalized_weight_share": [0.02] * preset.horizon,
                }
            rows.append(
                {
                    "seed": seed,
                    "dataset": preset.name,
                    "gamma": gamma,
                    "methods": methods,
                }
            )
    return rows


def _valid_phase_fixtures(runner):
    protocol = runner.load_extension_config(runner.CONFIG_PATH)
    preset = protocol.datasets["mimic_cxr"]
    seed = preset.seeds[0]
    k0_uniforms = runner._expected_k0_uniform_contract(
        seed=seed,
        horizon=preset.horizon,
        fidelity_episode_count=20,
        replay_count=runner.K0_SYSTEMATIC_REPLAYS,
    )

    support, contract = _phase_payload(
        runner,
        preset=preset,
        seed=seed,
        phase="support",
        result={
            "outcome_blind": True,
            "passed": True,
            "minimum_unique_patients": 20,
            "failed_cells": [],
            "unique_patient_counts_by_stage_action": [[20, 21]] * preset.horizon,
            "n_actions": 2,
            "active_action_indices": [0, 1],
            "action_costs": [0.0, 1.0],
            "action_mapping": {"0": 0, "1": 1},
            "split_audit": _split_audit(),
            "environment_episode_support": _episode_summary(),
        },
    )
    k0, _ = _phase_payload(
        runner,
        preset=preset,
        seed=seed,
        phase="k0_fidelity",
        result={
            "gate_name": "logging-mixture one-step fidelity",
            "passed": True,
            "metrics": {
                "maximum_score_ks": 0.10,
                "maximum_signed_residual_w1": 0.25,
                "maximum_successor_mean_w1": 0.25,
                "maximum_successor_q95_w1": 0.50,
                "structural_invariants": True,
            },
            "systematic_replay": {
                "label": "logging-mixture one-step fidelity",
                "episode_weighted": True,
                "systematic_replays": 16,
                "patient_chunk_size": runner.K0_PATIENT_CHUNK_SIZE,
                **k0_uniforms,
                "expansion_formula": "u[t,i,m]=(U[t,i]+(m+0.5)/16) mod 1",
                "flatten_order": (
                    "stage, patient, systematic_offset (offset fastest)"
                ),
                "inference_unit": (
                    "patient-disjoint episode query; M=16 quadrature, never 16N "
                    "independent observations"
                ),
                "score_ks_by_stage": [0.10] * preset.horizon,
                "signed_residual_max_w1_by_stage": [0.25] * preset.horizon,
                "successor_mean_w1_by_stage": [0.25] * preset.horizon,
                "successor_q95_w1_by_stage": [0.50] * preset.horizon,
                "active_successor_coordinates_by_stage": [1] * preset.horizon,
                "raw_structural_invariants_by_stage": [
                    _k0_invariant_row() for _ in range(preset.horizon)
                ],
            },
            "q_low": 1.0,
            "q_high": 2.0,
            "n_actions": 2,
            "action_mapping": {"0": 0, "1": 1},
            "split_audit": _split_audit(),
            "context_identity": _context_identity(runner),
        },
    )
    q_mid_metrics = {
        "local_ess_p01": 12.0,
        "median_ess_fraction": 0.40,
        "maximum_donor_probability": 0.20,
    }
    q_high_metrics = {
        "local_ess_p01": 10.0,
        "median_ess_fraction": 0.30,
        "maximum_donor_probability": 0.24,
    }
    worst_metrics = {
        "local_ess_p01": 10.0,
        "median_ess_fraction": 0.30,
        "maximum_donor_probability": 0.24,
    }
    overlap, _ = _phase_payload(
        runner,
        preset=preset,
        seed=seed,
        phase="donor_overlap",
        result={
            "passed": True,
            "interpretation_if_failed": "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
            "metrics": worst_metrics,
            "diagnostics": {
                "patient_aggregated": True,
                "probe_trajectories": 3_000,
                "gamma": -4.0,
                "noise_seed": runner._paper_seed(seed, runner.OVERLAP_STREAM_SALT),
                "independent_frozen_stream": True,
                "common_random_numbers_across_radii": True,
                "probes": {
                    "q_mid": _overlap_probe(
                        runner,
                        fraction=0.50,
                        radius=1.5,
                        metrics=q_mid_metrics,
                    ),
                    "q_high": _overlap_probe(
                        runner,
                        fraction=1.00,
                        radius=2.0,
                        metrics=q_high_metrics,
                    ),
                },
                "worst_metrics": worst_metrics,
                "screen_status": "EMPIRICAL_OVERLAP_SCREEN_PASSED",
                "screen_scope": (
                    "gamma=-4 q_mid and q_high=max-response; empirical, not a "
                    "guarantee"
                ),
                "environment_episode_support": _episode_summary(),
            },
            "q_low": 1.0,
            "q_high": 2.0,
            "q_mid": 1.5,
            "n_actions": 2,
            "action_mapping": {"0": 0, "1": 1},
            "split_audit": _split_audit(),
            "context_identity": _context_identity(runner),
        },
    )
    rows = []
    for gamma in runner.GAMMAS:
        rows.append(
            {
                "seed": seed,
                "dataset": preset.name,
                "gamma": gamma,
                "q_low": 1.0,
                "q_high": 2.0,
                "adaptation_seeds": runner._adaptation_seeds(seed),
                "methods": {
                    method: _method_row(runner, method, preset.horizon)
                    for method in runner.METHODS
                },
            }
        )
    science, _ = _phase_payload(
        runner,
        preset=preset,
        seed=seed,
        phase="science",
        result={
            "interpretation_status": "EMPIRICAL_OVERLAP_SCREEN_PASSED",
            "rows": rows,
            "q_low": 1.0,
            "q_high": 2.0,
            "n_actions": 2,
            "action_mapping": {"0": 0, "1": 1},
            "split_audit": _split_audit(),
            "context_identity": _context_identity(runner),
        },
    )
    return preset, seed, contract, {
        "support": support,
        "k0_fidelity": k0,
        "donor_overlap": overlap,
        "science": science,
    }


def test_stable_device_mapping_is_balanced_and_independent_of_dataset_subset() -> None:
    runner = _load_runner()
    protocol = runner.load_extension_config(runner.CONFIG_PATH)
    devices = ("cuda:0", "cuda:1")
    all_mapping = runner._stable_device_mapping(
        protocol,
        runner.DATASET_NAMES,
        devices,
    )
    subset_mapping = runner._stable_device_mapping(
        protocol,
        ("eicu", "mimic_cxr"),
        devices,
    )

    assert len(all_mapping) == 80
    for dataset in runner.DATASET_NAMES:
        seeds = protocol.datasets[dataset].seeds
        assignments = [all_mapping[f"{dataset}/{seed}"] for seed in seeds]
        assert assignments == [devices[index % 2] for index in range(20)]
        assert assignments.count("cuda:0") == assignments.count("cuda:1") == 10
    assert subset_mapping == {
        key: value
        for key, value in all_mapping.items()
        if key.startswith(("eicu/", "mimic_cxr/"))
    }


def test_rng_stream_mapping_enumerates_unique_dataset_specific_streams() -> None:
    runner = _load_runner()
    protocol = runner.load_extension_config(runner.CONFIG_PATH)
    mapping = runner._new_rng_stream_mapping(protocol, runner.DATASET_NAMES)

    assert len(mapping) == len(set(mapping.values())) == 1_304
    mimic_seed = protocol.datasets["mimic_iv"].seeds[0]
    cxr_seed = protocol.datasets["mimic_cxr"].seeds[0]
    assert mapping[f"mimic_iv/base_{mimic_seed}/task"] == mimic_seed
    assert mapping[f"mimic_iv/base_{mimic_seed}/k0_base_uniform"] == (
        runner.K0_UNIFORM_SEED_OFFSET + mimic_seed
    )
    assert f"mimic_iv/base_{mimic_seed}/cxr_encoder" not in mapping
    assert mapping[f"mimic_cxr/base_{cxr_seed}/cxr_encoder"] == cxr_seed + 701
    assert mapping["eicu/summary_bootstrap"] == protocol.datasets["eicu"].bootstrap_seed


def test_explicit_torch_seed_and_module_fingerprint_are_reproducible() -> None:
    runner = _load_runner()

    torch.manual_seed(999)
    torch.rand(7)
    runner._seed_torch_stream(123_456, "cpu")
    first_draw = torch.rand(8)
    torch.manual_seed(17)
    torch.rand(19)
    runner._seed_torch_stream(123_456, "cpu")
    second_draw = torch.rand(8)
    assert torch.equal(first_draw, second_draw)

    torch.manual_seed(91)
    first_model = torch.nn.Linear(3, 2)
    second_model = torch.nn.Linear(3, 2)
    second_model.load_state_dict(first_model.state_dict())
    first_hash = runner._module_state_sha256(first_model)
    assert first_hash == runner._module_state_sha256(second_model)
    assert len(first_hash) == 64
    with torch.no_grad():
        second_model.bias[0].add_(1.0)
    assert first_hash != runner._module_state_sha256(second_model)


def test_controlled_configs_reject_common_scientific_tuple_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    protocol = runner.load_extension_config(runner.CONFIG_PATH)
    for preset in protocol.datasets.values():
        controlled = runner._controlled_config(preset, protocol)
        assert runner._common_scientific_contract(controlled) == (
            runner.COMMON_SCIENTIFIC_CONTRACT
        )

    preset = protocol.datasets["eicu"]
    original = runner.ExperimentConfig.from_yaml(runner.ROOT / preset.base_config)
    drifted = replace(
        original,
        model=replace(original.model, hidden_dim=original.model.hidden_dim + 1),
    )
    monkeypatch.setattr(
        runner.ExperimentConfig,
        "from_yaml",
        classmethod(lambda cls, path: drifted),
    )
    with pytest.raises(RuntimeError, match="common scientific tuple differs"):
        runner._controlled_config(preset, protocol)


def test_split_audit_and_chunk_invariant_merge_are_order_independent() -> None:
    runner = _load_runner()

    def batch(patient_ids: list[int]) -> TrajectoryBatch:
        count = len(patient_ids)
        return TrajectoryBatch(
            states=torch.zeros((count, 2, 1)),
            actions=torch.zeros((count, 1), dtype=torch.long),
            outcomes=torch.zeros((count, 1, 2)),
            patient_ids=torch.tensor(patient_ids),
        )

    splits = runner.ClinicalExtensionSplits(
        predictor=batch([1, 1, 2]),
        fidelity=batch([3]),
        environment=batch([4, 4, 5]),
        split_fractions=(0.20, 0.20, 0.60),
    )
    audit = runner._split_audit(splits)
    assert audit["patient_sets_pairwise_disjoint"] is True
    assert audit["role_unique_patient_counts"] == {
        "predictor": 2,
        "fidelity": 1,
        "environment": 2,
    }
    assert audit["role_episode_counts"] == {
        "predictor": 3,
        "fidelity": 1,
        "environment": 3,
    }
    assert audit["split_fractions"] == [0.20, 0.20, 0.60]

    context = SimpleNamespace(
        splits=splits,
        outcome_model=torch.nn.Linear(1, 1),
        logging_policy=torch.nn.Linear(1, 1),
        q_low=0.5,
        q_high=0.9,
        n_actions=2,
        action_mapping={0: 0, 1: 1},
        action_costs=(0.0, 1.0),
        environment=SimpleNamespace(neighbors=100, bandwidth=2.0, ridge=1e-3),
        config=SimpleNamespace(to_dict=lambda: {"name": "test"}),
    )
    identity = runner._context_identity(context)
    assert identity["split_fractions"] == [0.20, 0.20, 0.60]
    assert identity["combined_sha256"] == runner._json_sha256(
        {key: value for key, value in identity.items() if key != "combined_sha256"}
    )

    chunks = [
        {"passed": True, "finite": True, "history": True},
        {"passed": False, "finite": True, "history": False},
    ]
    expected = {"passed": False, "finite": True, "history": False}
    assert runner._merge_invariant_rows(chunks) == expected
    assert runner._merge_invariant_rows(list(reversed(chunks))) == expected
    reordered_keys = [chunks[0], dict(reversed(tuple(chunks[1].items())))]
    assert runner._merge_invariant_rows(reordered_keys) == expected
    with pytest.raises(ValueError, match="at least one"):
        runner._merge_invariant_rows([])
    with pytest.raises(ValueError, match="different schemas"):
        runner._merge_invariant_rows([chunks[0], {"passed": True}])


def test_resume_rng_identity_ignores_prior_set_growth_but_rejects_drift_and_collision() -> None:
    runner = _load_runner()
    protocol = runner.load_extension_config(runner.CONFIG_PATH)
    mapping = runner._new_rng_stream_mapping(protocol, ("eicu",))
    digest = runner._json_sha256(mapping)
    stored_audit = {
        "status": "passed_before_launch",
        "collision_count": 0,
        "collisions": {},
        "prior_rng_id_count": 7,
        "prior_rng_id_sha256": "1" * 64,
        "new_rng_stream_mapping": mapping,
        "new_rng_stream_mapping_sha256": digest,
    }
    fresh_audit = {
        **stored_audit,
        "prior_rng_id_count": 99,
        "prior_rng_id_sha256": "2" * 64,
    }

    observed = runner._resume_rng_identity(
        {"rng_audit": stored_audit},
        fresh_audit,
    )
    assert observed is stored_audit
    changed_mapping = dict(mapping)
    first_key = next(iter(changed_mapping))
    changed_mapping[first_key] += 1
    with pytest.raises(RuntimeError, match="identity differs"):
        runner._resume_rng_identity(
            {"rng_audit": stored_audit},
            {
                **fresh_audit,
                "new_rng_stream_mapping": changed_mapping,
                "new_rng_stream_mapping_sha256": runner._json_sha256(changed_mapping),
            },
        )
    with pytest.raises(RuntimeError, match="identity differs"):
        runner._resume_rng_identity(
            {"rng_audit": stored_audit},
            {
                **fresh_audit,
                "status": "collision_detected",
                "collision_count": 1,
                "collisions": {first_key: mapping[first_key]},
            },
        )


def test_artifact_rng_scan_finds_nested_derived_streams_and_blocks_collision(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    protocol = runner.load_extension_config(runner.CONFIG_PATH)
    mapping = runner._new_rng_stream_mapping(protocol, ("eicu",))
    adaptation_id = mapping["eicu/base_92000/ACI_round_0"]
    reference_id = mapping["eicu/base_92000/reference"]
    artifact_root = tmp_path / "artifacts"
    output_root = artifact_root / "new_extension"
    prior_seed_file = artifact_root / "prior_study" / "seed_12345.json"
    prior_seed_file.parent.mkdir(parents=True)
    prior_seed_file.write_text(
        json.dumps(
            {
                "seed": 12_345,
                "result": {
                    "adaptation_seeds": {"ACI": adaptation_id},
                    "nested": {
                        "new_rng_stream_mapping": {"reference": reference_id}
                    },
                },
            }
        )
    )
    empty_source = tmp_path / "empty_source"
    empty_source.mkdir()

    detected = runner._all_artifact_rng_ids(
        artifact_root,
        excluded_root=output_root,
    )
    assert {12_345, adaptation_id, reference_id}.issubset(detected)
    with pytest.raises(RuntimeError, match="collides with prior use"):
        runner._audit_rng_banks(
            protocol,
            ("eicu",),
            output_root=output_root,
            artifact_root=artifact_root,
            source_root=empty_source,
        )


def test_manifest_verification_detects_content_and_file_set_tampering(tmp_path: Path) -> None:
    runner = _load_runner()
    root = tmp_path / "bundle"
    runner._write_json(root / "metadata.json", {"protocol": runner.PROTOCOL})
    runner._write_text(root / "nested" / "row.txt", "original\n")
    runner._write_manifest(root)
    runner._write_text(root / "COMPLETE", "complete\n")

    runner._verify_manifest(root)
    assert not list(root.rglob("*.tmp"))
    assert not [path for path in root.rglob("*") if ".tmp-" in path.name]
    (root / "nested" / "row.txt").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        runner._verify_manifest(root)

    (root / "nested" / "row.txt").write_text("original\n")
    (root / "unexpected.json").write_text("{}")
    with pytest.raises(RuntimeError, match="file set differs"):
        runner._verify_manifest(root)


def test_source_snapshot_is_deterministic_content_addressed_and_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    source = tmp_path / "source"
    for relative, content in {
        "src/scpcp/model.py": "VALUE = 1\n",
        "scripts/run.py": "print('run')\n",
        "tools/render.py": "print('render')\n",
        "configs/frozen.yaml": "alpha: 0.1\n",
        "pyproject.toml": "[project]\nname='fixture'\nversion='0.0.0'\n",
    }.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr(runner, "ROOT", source)

    first = runner._build_source_snapshot()
    second = runner._build_source_snapshot()
    assert first == second
    contract = first["contract"]
    assert contract["file_count"] == 5
    assert contract["archive_sha256"] in contract["archive_path"]
    assert contract["manifest_sha256"] in contract["manifest_path"]
    manifest = json.loads(first["manifest_bytes"])
    assert [entry["path"] for entry in manifest["files"]] == [
        "src/scpcp/model.py",
        "scripts/run.py",
        "tools/render.py",
        "configs/frozen.yaml",
        "pyproject.toml",
    ]

    bundle = tmp_path / "bundle"
    runner._publish_source_snapshot(bundle, first)
    runner._verify_source_snapshot(bundle, contract)
    archive_path = bundle / contract["archive_path"]
    archive_path.write_bytes(archive_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="does not match metadata"):
        runner._verify_source_snapshot(bundle, contract)


def test_precoverage_retry_archive_is_required_hashed_and_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    archive_relative = Path("failure_archive.tar")
    archive = tmp_path / archive_relative
    archive.write_bytes(b"frozen precoverage failure")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PRECOVERAGE_FAILURE_ARCHIVE", archive_relative)
    monkeypatch.setattr(
        runner,
        "PRECOVERAGE_FAILURE_ARCHIVE_SHA256",
        runner._file_sha256(archive),
    )
    monkeypatch.setattr(
        runner,
        "PRECOVERAGE_FAILURE_ARCHIVE_BYTES",
        archive.stat().st_size,
    )

    amendment = runner._verified_precoverage_retry_amendment()
    assert amendment["failure_archive"]["path"] == archive_relative.as_posix()
    assert amendment["failed_source_tree_sha256"] == runner.FAILED_SOURCE_TREE_SHA256
    assert amendment["failed_source_archive_sha256"] == (
        runner.FAILED_SOURCE_ARCHIVE_SHA256
    )
    assert amendment["failure"]["coverage_opened"] is False
    assert amendment["failure"]["scientific_rows_opened"] is False
    assert amendment["durable_failed_attempt_inventory"] == {
        "support_seed_artifacts": 20,
        "k0_fidelity_seed_artifacts": 0,
        "donor_overlap_seed_artifacts": 0,
        "science_seed_artifacts": 0,
        "root_complete_marker": False,
        "root_final_status": False,
    }
    assert amendment["retry_execution"]["same_prespecified_seed_banks_reused"]
    assert amendment["retry_execution"]["support_recomputed_from_scratch"]
    assert not amendment["retry_execution"]["failed_attempt_support_artifacts_reused"]
    assert not amendment["retry_execution"]["failed_attempt_phase_artifacts_reused"]
    assert not amendment["implementation_fix"]["scientific_contract_changed"]
    assert amendment["implementation_fix"]["new_operation"] == (
        "sorted_topk_distances[:, (k-1)//2]"
    )

    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="differs from the frozen retry contract"):
        runner._verified_precoverage_retry_amendment()
    archive.unlink()
    with pytest.raises(FileNotFoundError, match="required before launch"):
        runner._verified_precoverage_retry_amendment()


def test_postcompute_retry_archive_is_required_hashed_and_preinspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    archive_relative = Path("postcompute_failure_archive.tar")
    archive = tmp_path / archive_relative
    archive.write_bytes(b"frozen postcompute preinspection failure")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "POSTCOMPUTE_FAILURE_ARCHIVE", archive_relative)
    monkeypatch.setattr(
        runner,
        "POSTCOMPUTE_FAILURE_ARCHIVE_SHA256",
        runner._file_sha256(archive),
    )
    monkeypatch.setattr(
        runner,
        "POSTCOMPUTE_FAILURE_ARCHIVE_BYTES",
        archive.stat().st_size,
    )

    amendment = runner._verified_postcompute_retry_amendment()
    assert amendment["failure_archive"]["path"] == archive_relative.as_posix()
    assert amendment["failed_source_tree_sha256"] == (
        runner.POSTCOMPUTE_FAILED_SOURCE_TREE_SHA256
    )
    assert amendment["failed_source_archive_sha256"] == (
        runner.POSTCOMPUTE_FAILED_SOURCE_ARCHIVE_SHA256
    )
    assert amendment["failure"]["coverage_rows_generated"] is True
    assert amendment["failure"]["scientific_rows_generated"] is True
    assert amendment["failure"]["coverage_or_science_values_inspected_or_used"] is False
    assert amendment["failure"]["result_guided_change"] is False
    inventory = amendment["durable_failed_attempt_inventory"]
    assert [
        inventory[f"{phase}_seed_artifacts"]
        for phase in ("support", "k0_fidelity", "donor_overlap", "science")
    ] == [20, 20, 20, 20]
    assert inventory["dataset_complete_marker"] is True
    assert inventory["root_complete_marker"] is False
    assert inventory["other_dataset_directories"] == []
    assert amendment["retry_execution"]["same_prespecified_seed_banks_reused"]
    assert amendment["retry_execution"]["all_phases_recomputed_from_scratch"]
    assert not amendment["retry_execution"]["failed_attempt_phase_artifacts_reused"]
    assert amendment["validator_fix"][
        "json_object_order_is_semantically_irrelevant"
    ]
    assert not amendment["validator_fix"]["scientific_contract_changed"]

    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="differs from the frozen retry contract"):
        runner._verified_postcompute_retry_amendment()
    archive.unlink()
    with pytest.raises(FileNotFoundError, match="required before launch"):
        runner._verified_postcompute_retry_amendment()


def test_retry_amendment_binds_root_metadata_and_complete_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    protocol = runner.load_extension_config(runner.CONFIG_PATH)
    amendment = {
        "failure_archive": {"sha256": "a" * 64},
        "fixture": True,
    }
    postcompute_amendment = {
        "failure_archive": {"sha256": "e" * 64},
        "fixture": True,
    }
    contract = {
        "base_config_sha256": "b" * 64,
        "active_config": {"data": {"empirical_bandwidth": 2.0}},
    }
    monkeypatch.setattr(runner, "_environment_versions", lambda: {})
    metadata = runner._root_metadata(
        protocol,
        datasets=("eicu",),
        devices=("cuda:0", "cuda:1"),
        source_hash="c" * 64,
        contracts={"eicu": contract},
        seed_mapping={},
        rng_audit={},
        source_snapshot={"archive_sha256": "d" * 64},
        retry_amendment=amendment,
        postcompute_retry_amendment=postcompute_amendment,
    )
    runner._validate_retry_amendment_binding(metadata, amendment)
    runner._validate_postcompute_retry_amendment_binding(
        metadata, postcompute_amendment
    )
    assert metadata["precoverage_engineering_retry_amendment"] == amendment
    amendment_sha256 = runner._json_sha256(amendment)
    assert metadata["precoverage_engineering_retry_amendment_sha256"] == (
        amendment_sha256
    )
    postcompute_amendment_sha256 = runner._json_sha256(postcompute_amendment)
    assert metadata["postcompute_preinspection_retry_amendment"] == (
        postcompute_amendment
    )
    assert metadata["postcompute_preinspection_retry_amendment_sha256"] == (
        postcompute_amendment_sha256
    )
    marker = runner._root_complete_marker(
        metadata["source_snapshot"], amendment, postcompute_amendment
    )
    assert marker == (
        f"complete source_snapshot_sha256={'d' * 64} "
        f"precoverage_retry_amendment_sha256={amendment_sha256} "
        f"precoverage_failure_archive_sha256={'a' * 64} "
        f"postcompute_retry_amendment_sha256={postcompute_amendment_sha256} "
        f"postcompute_failure_archive_sha256={'e' * 64}\n"
    )

    tampered = deepcopy(metadata)
    tampered["precoverage_engineering_retry_amendment_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="does not bind"):
        runner._validate_retry_amendment_binding(tampered, amendment)
    tampered = deepcopy(metadata)
    tampered["postcompute_preinspection_retry_amendment_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="does not bind"):
        runner._validate_postcompute_retry_amendment_binding(
            tampered, postcompute_amendment
        )


@pytest.mark.parametrize("dataset", ["mimic_cxr", "eicu"])
def test_science_summary_uses_stagewise_seed_mean_and_frozen_eligibility(
    dataset: str,
) -> None:
    runner = _load_runner()
    protocol = runner.load_extension_config(runner.CONFIG_PATH)
    preset = protocol.datasets[dataset]
    selected_seeds = preset.seeds[:-1]
    rows = _summary_rows(runner, preset, selected_seeds)

    summary = runner.summarize_science(
        rows,
        preset=preset,
        selected_seeds=selected_seeds,
        interpretation_status="EMPIRICAL_OVERLAP_SCREEN_PASSED",
        bootstrap_contract={"fixture": True},
    )

    assert preset.horizon in {6, 12}
    assert summary["primary_metric"] == "min_t mean_seed(target_coverage_seed_t)"
    assert summary["selection_rate_denominator"] == "all 20 prespecified seeds"
    assert len(summary["seeds_prespecified"]) == 20
    assert len(summary["seeds_k0_eligible"]) == 19
    gamma_minus_four = next(
        aggregate for aggregate in summary["aggregates"] if aggregate["gamma"] == -4.0
    )
    standard = gamma_minus_four["methods"]["Standard CP"]
    mfcs = gamma_minus_four["methods"]["MFCS"]
    expected_stagewise_worst = (9 * 1.0 + 10 * 0.82) / 19

    assert standard["target_marginal_worst_coverage"] == pytest.approx(
        expected_stagewise_worst
    )
    assert standard["target_marginal_worst_coverage"] != pytest.approx(0.82)
    assert standard["n_selected"] == 19
    assert standard["n_k0_eligible"] == 19
    assert standard["n_prespecified"] == 20
    assert standard["selection_rate"] == pytest.approx(19 / 20)
    assert standard["point_eligible"] is True
    assert standard["confirmatory_attainment_at_0.90"] is True
    assert mfcs["n_selected"] == 18
    assert mfcs["selection_rate"] == pytest.approx(18 / 20)
    assert mfcs["selection_rate"] != pytest.approx(18 / 19)
    assert mfcs["selection_rate_ci95"] == pytest.approx(
        runner._wilson_interval(18, 20)
    )
    assert mfcs["point_eligible"] is False
    assert gamma_minus_four["paired_scpcp_comparisons"]["MFCS"][
        "paired_selected_seeds"
    ] == 18
    assert [
        item["method"]
        for item in gamma_minus_four["width_order_among_point_eligible"]
    ] == ["SC-PCP", "PRC", "SPCI", "ACI", "Standard CP"]

    for aggregate in summary["aggregates"]:
        if aggregate["gamma"] == -4.0:
            assert aggregate["analysis_role"] == "confirmatory_gamma_minus_4_endpoint"
            continue
        assert aggregate["analysis_role"] == "descriptive_signed_control_curve"
        assert aggregate["paired_scpcp_comparisons"] == {
            "status": "EXCLUDED_NON_CONFIRMATORY_GAMMA_SIGNED_CONTROL"
        }
        assert aggregate["width_order_among_point_eligible"] == []
        assert aggregate["methods"]["SC-PCP"]["point_eligible"] is None
        assert aggregate["methods"]["SC-PCP"][
            "confirmatory_attainment_at_0.90"
        ] is None


def test_low_donor_overlap_removes_point_eligibility_and_width_order() -> None:
    runner = _load_runner()
    protocol = runner.load_extension_config(runner.CONFIG_PATH)
    preset = protocol.datasets["mimic_cxr"]
    selected_seeds = preset.seeds[:-1]
    summary = runner.summarize_science(
        _summary_rows(runner, preset, selected_seeds),
        preset=preset,
        selected_seeds=selected_seeds,
        interpretation_status="LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
        bootstrap_contract={"fixture": True},
    )

    for aggregate in summary["aggregates"]:
        assert aggregate["analysis_role"] == "descriptive_signed_control_curve"
        assert aggregate["paired_scpcp_comparisons"] == {
            "status": "EXCLUDED_LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        }
        assert aggregate["width_order_among_point_eligible"] == []
        for method in runner.METHODS:
            assert aggregate["methods"][method]["point_eligible"] is None
            assert aggregate["methods"][method][
                "confirmatory_attainment_at_0.90"
            ] is None


def test_zero_selection_summary_keeps_schema_and_uses_three_state_eligibility() -> None:
    runner = _load_runner()
    protocol = runner.load_extension_config(runner.CONFIG_PATH)
    preset = protocol.datasets["mimic_cxr"]
    selected_seeds = preset.seeds[:-1]
    rows = _summary_rows(runner, preset, selected_seeds)
    for row in rows:
        row["methods"]["MFCS"] = {
            "target_adaptation_trajectories": runner.TARGET_ADAPTATION_BUDGET["MFCS"],
            "information_regime": runner.INFORMATION_REGIME["MFCS"],
            "selection_available": False,
            "radii": [],
        }

    confirmatory = runner.summarize_science(
        rows,
        preset=preset,
        selected_seeds=selected_seeds,
        interpretation_status="EMPIRICAL_OVERLAP_SCREEN_PASSED",
        bootstrap_contract={"fixture": True},
    )
    gamma_minus_four = next(
        aggregate
        for aggregate in confirmatory["aggregates"]
        if aggregate["gamma"] == -4.0
    )
    other_gamma = next(
        aggregate
        for aggregate in confirmatory["aggregates"]
        if aggregate["gamma"] != -4.0
    )
    empty_confirmatory = gamma_minus_four["methods"]["MFCS"]
    empty_nonconfirmatory = other_gamma["methods"]["MFCS"]
    populated = gamma_minus_four["methods"]["Standard CP"]

    assert set(empty_confirmatory) == set(empty_nonconfirmatory) == set(populated)
    assert empty_confirmatory == {
        "n_selected": 0,
        "n_prespecified": 20,
        "n_k0_eligible": 19,
        "selection_rate": 0.0,
        "selection_rate_ci95": pytest.approx(runner._wilson_interval(0, 20)),
        "target_adaptation_trajectories_per_seed": (
            runner.TARGET_ADAPTATION_BUDGET["MFCS"]
        ),
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
        "confirmatory_attainment_at_0.90": None,
        "point_eligibility_rule": (
            "selection_rate>=0.95 and target_marginal_worst_coverage>=0.90"
        ),
        "point_eligible": False,
    }
    assert empty_nonconfirmatory["confirmatory_attainment_at_0.90"] is None
    assert empty_nonconfirmatory["point_eligible"] is None

    low_overlap = runner.summarize_science(
        rows,
        preset=preset,
        selected_seeds=selected_seeds,
        interpretation_status="LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
        bootstrap_contract={"fixture": True},
    )
    low_overlap_gamma_minus_four = next(
        aggregate
        for aggregate in low_overlap["aggregates"]
        if aggregate["gamma"] == -4.0
    )
    empty_low_overlap = low_overlap_gamma_minus_four["methods"]["MFCS"]
    assert set(empty_low_overlap) == set(populated)
    assert empty_low_overlap["confirmatory_attainment_at_0.90"] is None
    assert empty_low_overlap["point_eligible"] is None


def test_phase_payload_validation_accepts_all_four_shapes_and_fails_closed() -> None:
    runner = _load_runner()
    preset, seed, contract, fixtures = _valid_phase_fixtures(runner)

    for phase, payload in fixtures.items():
        assert runner._valid_phase_payload(
            payload,
            phase=phase,
            preset=preset,
            seed=seed,
            device="cuda:0",
            seed_contract=contract,
        )

    wrong_provenance = deepcopy(fixtures["support"])
    wrong_provenance["source_tree_sha256"] = "0" * 64
    assert not runner._valid_phase_payload(
        wrong_provenance,
        phase="support",
        preset=preset,
        seed=seed,
        device="cuda:0",
        seed_contract=contract,
    )
    malformed_support = deepcopy(fixtures["support"])
    malformed_support["result"]["passed"] = False
    assert not runner._valid_phase_payload(
        malformed_support,
        phase="support",
        preset=preset,
        seed=seed,
        device="cuda:0",
        seed_contract=contract,
    )
    malformed_k0 = deepcopy(fixtures["k0_fidelity"])
    malformed_k0["result"]["systematic_replay"]["systematic_replays"] = 15
    assert not runner._valid_phase_payload(
        malformed_k0,
        phase="k0_fidelity",
        preset=preset,
        seed=seed,
        device="cuda:0",
        seed_contract=contract,
    )
    wrong_context = deepcopy(fixtures["k0_fidelity"])
    wrong_context["result"]["context_identity"]["combined_sha256"] = "0" * 64
    assert not runner._valid_phase_payload(
        wrong_context,
        phase="k0_fidelity",
        preset=preset,
        seed=seed,
        device="cuda:0",
        seed_contract=contract,
    )
    malformed_overlap = deepcopy(fixtures["donor_overlap"])
    malformed_overlap["result"]["diagnostics"]["patient_aggregated"] = False
    assert not runner._valid_phase_payload(
        malformed_overlap,
        phase="donor_overlap",
        preset=preset,
        seed=seed,
        device="cuda:0",
        seed_contract=contract,
    )
    malformed_science = deepcopy(fixtures["science"])
    del malformed_science["result"]["rows"][0]["methods"]["SC-PCP"]
    assert not runner._valid_phase_payload(
        malformed_science,
        phase="science",
        preset=preset,
        seed=seed,
        device="cuda:0",
        seed_contract=contract,
    )


def test_sorted_json_roundtrip_preserves_k0_and_overlap_payload_validity() -> None:
    runner = _load_runner()
    preset, seed, contract, fixtures = _valid_phase_fixtures(runner)

    roundtripped = {
        phase: json.loads(json.dumps(fixtures[phase], sort_keys=True))
        for phase in ("k0_fidelity", "donor_overlap")
    }
    k0_row = roundtripped["k0_fidelity"]["result"]["systematic_replay"][
        "raw_structural_invariants_by_stage"
    ][0]
    overlap_probes = roundtripped["donor_overlap"]["result"]["diagnostics"][
        "probes"
    ]
    assert tuple(k0_row) != tuple(_k0_invariant_row())
    assert tuple(overlap_probes) == ("q_high", "q_mid")

    for phase, payload in roundtripped.items():
        assert runner._valid_phase_payload(
            payload,
            phase=phase,
            preset=preset,
            seed=seed,
            device="cuda:0",
            seed_contract=contract,
        )


def test_k0_payload_binds_m16_detail_stage_headlines_invariants_and_uniforms() -> None:
    runner = _load_runner()
    preset, seed, contract, fixtures = _valid_phase_fixtures(runner)
    valid = fixtures["k0_fidelity"]
    detail_path = valid["result"]["systematic_replay"]
    assert detail_path["base_uniform_shape"] == [preset.horizon, 20]

    mutations = {
        "M16 replay count": lambda payload: payload["result"][
            "systematic_replay"
        ].__setitem__("systematic_replays", 15),
        "stage-vector length": lambda payload: payload["result"][
            "systematic_replay"
        ]["score_ks_by_stage"].pop(),
        "stage-vector finite": lambda payload: payload["result"][
            "systematic_replay"
        ]["signed_residual_max_w1_by_stage"].__setitem__(0, float("nan")),
        "headline maximum": lambda payload: payload["result"]["metrics"].__setitem__(
            "maximum_score_ks", 0.09
        ),
        "raw invariant stage schema": lambda payload: payload["result"][
            "systematic_replay"
        ]["raw_structural_invariants_by_stage"][0].pop("finite"),
        "raw invariant total": lambda payload: (
            payload["result"]["systematic_replay"][
                "raw_structural_invariants_by_stage"
            ][0].__setitem__("finite", False),
            payload["result"]["systematic_replay"][
                "raw_structural_invariants_by_stage"
            ][0].__setitem__("passed", False),
        ),
        "active successor count": lambda payload: payload["result"][
            "systematic_replay"
        ]["active_successor_coordinates_by_stage"].__setitem__(0, 0),
        "base uniform seed": lambda payload: payload["result"][
            "systematic_replay"
        ].__setitem__(
            "base_uniform_seed",
            payload["result"]["systematic_replay"]["base_uniform_seed"] + 1,
        ),
        "base uniform shape": lambda payload: payload["result"][
            "systematic_replay"
        ].__setitem__("base_uniform_shape", [preset.horizon, 21]),
        "base uniform hash": lambda payload: payload["result"][
            "systematic_replay"
        ].__setitem__("base_uniform_sha256", "0" * 64),
        "systematic uniform hash": lambda payload: payload["result"][
            "systematic_replay"
        ].__setitem__("expanded_uniform_sha256", "0" * 64),
        "fidelity episode count": lambda payload: payload["result"]["split_audit"][
            "role_episode_counts"
        ].__setitem__("fidelity", 21),
    }

    for label, mutate in mutations.items():
        tampered = deepcopy(valid)
        mutate(tampered)
        assert not runner._valid_phase_payload(
            tampered,
            phase="k0_fidelity",
            preset=preset,
            seed=seed,
            device="cuda:0",
            seed_contract=contract,
        ), label


def test_overlap_payload_validation_rejects_frozen_probe_contract_tampering() -> None:
    runner = _load_runner()
    preset, seed, contract, fixtures = _valid_phase_fixtures(runner)
    valid = fixtures["donor_overlap"]
    mutations = {
        "reconstructed probe radius": lambda payload: payload["result"][
            "diagnostics"
        ]["probes"]["q_mid"].__setitem__("radius", 1.51),
        "root q_mid": lambda payload: payload["result"].__setitem__("q_mid", 1.51),
        "noise seed": lambda payload: payload["result"]["diagnostics"].__setitem__(
            "noise_seed",
            payload["result"]["diagnostics"]["noise_seed"] + 1,
        ),
        "independent frozen stream": lambda payload: payload["result"][
            "diagnostics"
        ].__setitem__("independent_frozen_stream", False),
        "screen scope": lambda payload: payload["result"]["diagnostics"].__setitem__(
            "screen_scope", "changed after launch"
        ),
    }

    for label, mutate in mutations.items():
        tampered = deepcopy(valid)
        mutate(tampered)
        assert not runner._valid_phase_payload(
            tampered,
            phase="donor_overlap",
            preset=preset,
            seed=seed,
            device="cuda:0",
            seed_contract=contract,
        ), label


@pytest.mark.parametrize("container", ["metrics", "worst_metrics"])
@pytest.mark.parametrize(
    "metric",
    ["local_ess_p01", "median_ess_fraction", "maximum_donor_probability"],
)
def test_overlap_payload_recomputes_each_worst_metric_from_both_probes(
    container: str,
    metric: str,
) -> None:
    runner = _load_runner()
    preset, seed, contract, fixtures = _valid_phase_fixtures(runner)
    tampered = deepcopy(fixtures["donor_overlap"])
    if container == "metrics":
        target = tampered["result"]["metrics"]
    else:
        target = tampered["result"]["diagnostics"]["worst_metrics"]
    target[metric] = float(target[metric]) + 0.01

    assert not runner._valid_phase_payload(
        tampered,
        phase="donor_overlap",
        preset=preset,
        seed=seed,
        device="cuda:0",
        seed_contract=contract,
    )


def test_science_payload_binds_coverage_gap_to_target_minus_source() -> None:
    runner = _load_runner()
    preset, seed, contract, fixtures = _valid_phase_fixtures(runner)
    within_tolerance = deepcopy(fixtures["science"])
    within_tolerance["result"]["rows"][0]["methods"]["Standard CP"][
        "coverage_gap"
    ][0] = 0.5e-6
    assert runner._valid_phase_payload(
        within_tolerance,
        phase="science",
        preset=preset,
        seed=seed,
        device="cuda:0",
        seed_contract=contract,
    )

    tampered = deepcopy(fixtures["science"])
    tampered["result"]["rows"][0]["methods"]["Standard CP"]["coverage_gap"][
        0
    ] = 1.1e-6
    assert not runner._valid_phase_payload(
        tampered,
        phase="science",
        preset=preset,
        seed=seed,
        device="cuda:0",
        seed_contract=contract,
    )


def test_overlap_screen_uses_worst_of_two_and_either_failure_is_descriptive() -> None:
    runner = _load_runner()
    preset, seed, contract, fixtures = _valid_phase_fixtures(runner)
    passing = fixtures["donor_overlap"]
    probes = passing["result"]["diagnostics"]["probes"]
    assert passing["result"]["metrics"] == {
        "local_ess_p01": min(
            probes["q_mid"]["metrics"]["local_ess_p01"],
            probes["q_high"]["metrics"]["local_ess_p01"],
        ),
        "median_ess_fraction": min(
            probes["q_mid"]["metrics"]["median_ess_fraction"],
            probes["q_high"]["metrics"]["median_ess_fraction"],
        ),
        "maximum_donor_probability": max(
            probes["q_mid"]["metrics"]["maximum_donor_probability"],
            probes["q_high"]["metrics"]["maximum_donor_probability"],
        ),
    }
    assert passing["result"]["passed"] is True
    assert passing["result"]["diagnostics"]["common_random_numbers_across_radii"] is True
    assert passing["result"]["diagnostics"]["screen_status"] == (
        "EMPIRICAL_OVERLAP_SCREEN_PASSED"
    )

    one_failed = deepcopy(passing)
    high = one_failed["result"]["diagnostics"]["probes"]["q_high"]
    high["metrics"]["local_ess_p01"] = 9.0
    high["passed"] = False
    worst = {
        "local_ess_p01": 9.0,
        "median_ess_fraction": 0.30,
        "maximum_donor_probability": 0.24,
    }
    one_failed["result"]["metrics"] = worst
    one_failed["result"]["passed"] = False
    one_failed["result"]["diagnostics"]["worst_metrics"] = worst
    one_failed["result"]["diagnostics"]["screen_status"] = (
        "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
    )
    assert runner._valid_phase_payload(
        one_failed,
        phase="donor_overlap",
        preset=preset,
        seed=seed,
        device="cuda:0",
        seed_contract=contract,
    )

    selected_seeds = preset.seeds[:-1]
    summary = runner.summarize_science(
        _summary_rows(runner, preset, selected_seeds),
        preset=preset,
        selected_seeds=selected_seeds,
        interpretation_status=one_failed["result"]["diagnostics"]["screen_status"],
        bootstrap_contract={"fixture": True},
    )
    gamma_minus_four = next(
        aggregate for aggregate in summary["aggregates"] if aggregate["gamma"] == -4.0
    )
    assert gamma_minus_four["analysis_role"] == "descriptive_signed_control_curve"
    assert gamma_minus_four["width_order_among_point_eligible"] == []
    assert gamma_minus_four["methods"]["SC-PCP"]["point_eligible"] is None

    wrong_crn = deepcopy(passing)
    wrong_crn["result"]["diagnostics"]["common_random_numbers_across_radii"] = False
    assert not runner._valid_phase_payload(
        wrong_crn,
        phase="donor_overlap",
        preset=preset,
        seed=seed,
        device="cuda:0",
        seed_contract=contract,
    )


def test_no_go_bundle_contains_no_science_and_is_manifest_verified(tmp_path: Path) -> None:
    runner = _load_runner()
    protocol = runner.load_extension_config(runner.CONFIG_PATH)
    preset = protocol.datasets["eicu"]
    root = tmp_path / "eicu"
    runner._write_json(root / "metadata.json", {"role": "test"})

    runner._publish_no_go(
        root,
        preset=preset,
        reason="K0_FIDELITY_NO_GO",
        detail="test gate failure",
    )

    assert not (root / "science").exists()
    assert (root / "COMPLETE").read_text() == "gate-no-go\n"
    assert runner._read_json(root / "gate.json")["panel_status"] == "GATE_NO_GO"
    assert runner._read_json(root / "FINAL_STATUS.json")["scientific_rows_saved"] is False
    runner._verify_manifest(root)

    blocked = tmp_path / "blocked"
    (blocked / "science").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="cannot coexist"):
        runner._publish_no_go(
            blocked,
            preset=preset,
            reason="SUPPORT_NO_GO",
            detail="test gate failure",
        )


def test_any_exact_k0_invariant_failure_is_structural_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    protocol = runner.load_extension_config(runner.CONFIG_PATH)
    preset = protocol.datasets["eicu"]
    phases: list[str] = []
    split_audit = _split_audit()
    identity = _context_identity(runner)

    def fake_phase(*args, phase: str, preset, **kwargs):
        del args, kwargs
        phases.append(phase)
        if phase == "support":
            return [
                {
                    "seed": seed,
                    "passed": True,
                    "n_actions": 2,
                    "action_mapping": {"0": 0, "1": 1},
                    "split_audit": split_audit,
                }
                for seed in preset.seeds
            ]
        if phase == "k0_fidelity":
            return [
                {
                    "seed": seed,
                    "passed": seed != preset.seeds[0],
                    "n_actions": 2,
                    "action_mapping": {"0": 0, "1": 1},
                    "metrics": {
                        "structural_invariants": seed != preset.seeds[0],
                    },
                    "context_identity": identity,
                }
                for seed in preset.seeds
            ]
        raise AssertionError(f"phase {phase} must not run after structural failure")

    monkeypatch.setattr(runner, "_run_phase", fake_phase)
    monkeypatch.setattr(runner, "_validate_final_dataset_bundle", lambda *args, **kwargs: None)
    seed_to_device = {
        seed: ("cuda:0" if index % 2 == 0 else "cuda:1")
        for index, seed in enumerate(preset.seeds)
    }
    root = tmp_path / "eicu"
    retry_amendment = {
        "failure_archive": {"sha256": "e" * 64},
        "fixture": True,
    }
    postcompute_retry_amendment = {
        "failure_archive": {"sha256": "f" * 64},
        "fixture": True,
    }
    runner._run_dataset(
        root,
        protocol=protocol,
        preset=preset,
        devices=("cuda:0", "cuda:1"),
        source_hash="c" * 64,
        contract={"fixture": True},
        seed_to_device=seed_to_device,
        rng_audit={"new_rng_stream_mapping_sha256": "d" * 64},
        retry_amendment=retry_amendment,
        postcompute_retry_amendment=postcompute_retry_amendment,
        resume=False,
    )

    assert phases == ["support", "k0_fidelity"]
    assert not (root / "donor_overlap").exists()
    assert not (root / "science").exists()
    assert runner._read_json(root / "FINAL_STATUS.json")["status"] == (
        "STRUCTURAL_NO_GO"
    )
    assert runner._read_json(root / "gate.json")["reason"] == "STRUCTURAL_NO_GO"
    assert (root / "COMPLETE").read_text() == "gate-no-go\n"
    dataset_metadata = runner._read_json(root / "metadata.json")
    assert dataset_metadata["precoverage_engineering_retry_amendment"] == (
        retry_amendment
    )
    assert dataset_metadata["precoverage_engineering_retry_amendment_sha256"] == (
        runner._json_sha256(retry_amendment)
    )
    assert dataset_metadata["postcompute_preinspection_retry_amendment"] == (
        postcompute_retry_amendment
    )
    assert dataset_metadata["postcompute_preinspection_retry_amendment_sha256"] == (
        runner._json_sha256(postcompute_retry_amendment)
    )
    runner._verify_manifest(root)
