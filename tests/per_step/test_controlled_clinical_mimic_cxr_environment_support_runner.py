from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = (
        ROOT
        / "scripts"
        / "run_controlled_clinical_mimic_cxr_environment_support.py"
    )
    name = "test_run_controlled_clinical_mimic_cxr_environment_support"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _snapshot() -> dict[str, Any]:
    archive = b"source archive"
    manifest = b'{"files":[]}\n'
    contract = {
        "archive_path": "provenance/source.tar",
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "archive_bytes": len(archive),
        "manifest_path": "provenance/source.json",
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "manifest_bytes": len(manifest),
        "file_count": 0,
    }
    return {
        "archive_bytes": archive,
        "manifest_bytes": manifest,
        "contract": contract,
    }


def _metadata(root: Path, snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    contract = _snapshot()["contract"] if snapshot is None else snapshot["contract"]
    return {
        "protocol": runner.PROTOCOL,
        "dataset": runner.DATASET,
        "phase": "development",
        "output_root": str(root),
        "source_tree_sha256": "a" * 64,
        "source_snapshot": contract,
        "config_sha256": "b" * 64,
        "role_split": list(runner.ROLE_SPLIT),
        "coverage_generation_permitted": False,
    }


def _seal_root(root: Path) -> None:
    runner._write_manifest(root)
    metadata = runner._read_json(root / "metadata.json")
    final = runner._read_json(root / "FINAL_STATUS.json")
    marker = (
        f"complete phase={metadata['phase']} "
        f"source_tree_sha256={metadata['source_tree_sha256']} "
        f"config_sha256={metadata['config_sha256']} "
        f"final_status_sha256={runner._json_sha256(final)} "
        f"manifest_sha256={runner._file_sha256(root / 'manifest.json')}\n"
    )
    runner._write_text(root / "COMPLETE", marker)


def _support_result(seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "passed": True,
        "outcome_blind": True,
        "role_split": list(runner.ROLE_SPLIT),
        "split_audit": {
            "patient_sets_pairwise_disjoint": True,
            "split_fractions": list(runner.ROLE_SPLIT),
        },
        "coverage_generated": False,
    }


def _seed_envelope(
    result: Mapping[str, Any], *, phase: str, seed: int, source_hash: str = "c" * 64
) -> dict[str, Any]:
    return {
        "protocol": runner.PROTOCOL,
        "phase": phase,
        "dataset": runner.DATASET,
        "seed": seed,
        "device": "cuda:0",
        "source_tree_sha256": source_hash,
        "result": dict(result),
    }


def _k0_metrics(ratio: float, *, structural: bool = True) -> dict[str, Any]:
    return {
        name: threshold * ratio
        for name, threshold in runner.K0_THRESHOLDS.items()
    } | {"structural_invariants": structural}


def _rng_audit(phase: str) -> dict[str, Any]:
    mapping = {f"mimic_cxr/base_1/{kind}": index for index, kind in enumerate(
        (suffix.removeprefix("/") for suffix in runner.PRECOVERAGE_RNG_STREAM_SUFFIXES),
        start=1,
    )}
    empty_sha256 = runner._integer_set_sha256(())
    other_phase = "confirmation" if phase == "development" else "development"
    return {
        "status": "passed_before_launch",
        "role": phase,
        "collision_count": 0,
        "collisions": {},
        "authorized_collision_count": 0,
        "unauthorized_collision_count": 0,
        "unauthorized_collisions": {},
        "new_rng_stream_count": len(mapping),
        "new_rng_stream_mapping": mapping,
        "new_rng_stream_mapping_sha256": runner._json_sha256(mapping),
        "new_rng_id_set_sha256": runner._integer_set_sha256(mapping.values()),
        "artifact_rng_id_count": 0,
        "artifact_rng_id_sha256": empty_sha256,
        "source_declared_rng_id_count": 0,
        "source_declared_rng_id_sha256": empty_sha256,
        "prior_rng_id_count": 0,
        "prior_rng_id_sha256": empty_sha256,
        "base_seed_count": 1,
        "base_seed_set_sha256": runner._integer_set_sha256((1,)),
        "executed_stream_kinds": [
            suffix.removeprefix("/")
            for suffix in runner.PRECOVERAGE_RNG_STREAM_SUFFIXES
        ],
        "executed_streams_only": True,
        "scientific_freshness_claimed": phase == "confirmation",
        "cross_bank_audit": {
            f"{phase}_stream_count": len(mapping),
            f"{phase}_mapping_sha256": runner._json_sha256(mapping),
            f"{other_phase}_stream_count": 0,
            f"{other_phase}_mapping_sha256": empty_sha256,
            "collision_count": 0,
            "collisions": {},
        },
    }


def test_fresh_root_and_resume_metadata_are_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "formal"
    snapshot = _snapshot()
    metadata = _metadata(root, snapshot)

    runner._prepare_root(root, metadata, snapshot, resume=False)

    assert runner._read_json(root / "metadata.json") == metadata
    runner.v4._verify_source_snapshot(root, metadata["source_snapshot"])
    with pytest.raises(FileExistsError, match="already exists"):
        runner._prepare_root(root, metadata, snapshot, resume=False)

    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    runner._prepare_root(root, metadata, snapshot, resume=True)
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before

    changed = {**metadata, "devices": ["cuda:1", "cuda:0"]}
    with pytest.raises(RuntimeError, match="resume metadata differs"):
        runner._prepare_root(root, changed, snapshot, resume=True)
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_preflight_fresh_root_check_detects_broken_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = tmp_path / "development"
    confirmation = tmp_path / "confirmation"
    science = tmp_path / "science_root"
    monkeypatch.setattr(runner, "DEVELOPMENT_ROOT", development)
    monkeypatch.setattr(runner, "CONFIRMATION_ROOT", confirmation)
    monkeypatch.setattr(runner, "SCIENCE_ROOT", science)

    runner._assert_fresh_roots()
    development.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(FileExistsError, match="formal roots must be absent"):
        runner._assert_fresh_roots()


def test_formal_devices_are_two_distinct_explicit_cuda_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 2)

    runner._validate_devices(("cuda:0", "cuda:1"))
    for devices in (
        ("cuda:0",),
        ("cuda:0", "cuda:0"),
        ("cuda", "cuda:1"),
        ("cpu", "cuda:1"),
    ):
        with pytest.raises(ValueError, match="two explicit CUDA"):
            runner._validate_devices(devices)
    with pytest.raises(RuntimeError, match="two CUDA devices"):
        runner._validate_devices(("cuda:0", "cuda:2"))


def test_resume_rejects_symlinks_before_using_partial_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "formal"
    snapshot = _snapshot()
    metadata = _metadata(root, snapshot)
    runner._prepare_root(root, metadata, snapshot, resume=False)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (root / "seed_000001.json").symlink_to(outside)

    with pytest.raises(RuntimeError, match="symlink is forbidden"):
        runner._prepare_root(root, metadata, snapshot, resume=True)


def test_nested_phase_complete_is_manifested_and_tamper_evident(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    snapshot = _snapshot()
    runner._prepare_root(root, _metadata(root, snapshot), snapshot, resume=False)
    runner._write_json(root / "FINAL_STATUS.json", {"status": "DEVELOPMENT_GO"})
    runner._write_text(root / "support" / "block_a" / "COMPLETE", "complete\n")
    runner._write_json(root / "support" / "block_a" / "seed_000001.json", {})
    _seal_root(root)

    paths = {
        entry["path"]
        for entry in runner._read_json(root / "manifest.json")["artifacts"]
    }
    assert "support/block_a/COMPLETE" in paths
    runner._verify_complete_root(root)

    runner._write_text(root / "support" / "block_a" / "COMPLETE", "changed\n")
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        runner._verify_complete_root(root)


def test_completed_phase_marker_is_validated_not_silently_rewritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_root = tmp_path / "support"
    seed = 11
    phase = "development_support_block_a"
    source_hash = "c" * 64
    runner._write_json(
        phase_root / f"seed_{seed:06d}.json",
        _seed_envelope(
            _support_result(seed),
            phase=phase,
            seed=seed,
            source_hash=source_hash,
        ),
    )
    runner._write_text(phase_root / "COMPLETE", "complete\n")
    monkeypatch.setattr(
        runner,
        "_write_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected write")),
    )

    rows = runner._run_seed_phase(
        phase_root,
        phase=phase,
        preset=SimpleNamespace(seeds=(seed,)),
        devices=("cuda:0",),
        worker=lambda *args: {},
        worker_arguments=(),
        source_hash=source_hash,
    )
    assert rows == [_support_result(seed)]

    (phase_root / "COMPLETE").write_text("corrupt\n")
    with pytest.raises(RuntimeError, match="invalid .* COMPLETE"):
        runner._run_seed_phase(
            phase_root,
            phase=phase,
            preset=SimpleNamespace(seeds=(seed,)),
            devices=("cuda:0",),
            worker=lambda *args: {},
            worker_arguments=(),
            source_hash=source_hash,
        )


@pytest.mark.parametrize("token", runner.FORBIDDEN_PATH_TOKENS)
def test_forbidden_precoverage_paths_fail_on_resume_and_complete_verification(
    tmp_path: Path,
    token: str,
) -> None:
    root = tmp_path / token
    snapshot = _snapshot()
    metadata = _metadata(root, snapshot)
    runner._prepare_root(root, metadata, snapshot, resume=False)
    runner._write_json(root / "FINAL_STATUS.json", {"status": "DEVELOPMENT_GO"})
    runner._write_json(root / token / "payload.json", {})

    with pytest.raises(RuntimeError, match="forbidden precoverage artifact path"):
        runner._prepare_root(root, metadata, snapshot, resume=True)

    _seal_root(root)
    with pytest.raises(RuntimeError, match="forbidden precoverage artifact path"):
        runner._verify_complete_root(root)


def test_k0_decision_is_derived_only_from_frozen_metrics() -> None:
    assert runner._k0_passes_from_metrics(_k0_metrics(1.0)) is True
    assert runner._k0_passes_from_metrics(_k0_metrics(1.0, structural=False)) is False
    for name in runner.K0_THRESHOLDS:
        metrics = _k0_metrics(0.5)
        metrics[name] = runner.K0_THRESHOLDS[name] * 1.01
        assert runner._k0_passes_from_metrics(metrics) is False

    invalid = _k0_metrics(0.5)
    invalid["maximum_score_ks"] = float("nan")
    with pytest.raises(RuntimeError, match="invalid K0 metric value"):
        runner._k0_passes_from_metrics(invalid)


def test_k0_worker_and_envelope_bind_passed_to_metrics_and_20_20_60_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = runner.CONFIRMATION_SEEDS[0]
    base_context = SimpleNamespace(splits=object())
    context = SimpleNamespace(environment=object())
    theta = SimpleNamespace(
        to_dict=lambda: {"candidate_id": runner.BRIDGE_CANDIDATE_ID}
    )
    metrics = runner.v2.K0FidelityMetrics(**_k0_metrics(0.5))
    monkeypatch.setattr(runner.v2, "_prepare_extension_context", lambda *args: base_context)
    monkeypatch.setattr(runner, "_b02", lambda: theta)
    monkeypatch.setattr(runner.v5, "_context_with_theta", lambda *args: context)
    monkeypatch.setattr(
        runner.v5,
        "_logging_mixture_fidelity_v5",
        lambda *args, **kwargs: (metrics, {"replays": 16}),
    )
    monkeypatch.setattr(
        runner.v2,
        "_context_identity",
        lambda value: {"split_fractions": list(runner.ROLE_SPLIT)},
    )
    monkeypatch.setattr(
        runner.v5,
        "_candidate_context_identity",
        lambda *args: {"combined_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        runner.v2,
        "_split_audit",
        lambda value: {
            "patient_sets_pairwise_disjoint": True,
            "split_fractions": list(runner.ROLE_SPLIT),
        },
    )

    result = runner._k0_worker(
        seed,
        SimpleNamespace(),
        "cuda:0",
        SimpleNamespace(),
    )
    assert result["passed"] is True
    assert result["role_split"] == [0.20, 0.20, 0.60]
    assert result["base_context_identity"]["split_fractions"] == [0.20, 0.20, 0.60]
    assert result["split_audit"]["split_fractions"] == [0.20, 0.20, 0.60]

    phase = "confirmation_k0"
    envelope = _seed_envelope(result, phase=phase, seed=seed)
    runner._validate_seed_envelope(
        envelope,
        phase=phase,
        seed=seed,
        device="cuda:0",
        source_hash="c" * 64,
    )
    forged = json.loads(json.dumps(envelope))
    forged["result"]["passed"] = False
    with pytest.raises(RuntimeError, match="K0 decision"):
        runner._validate_seed_envelope(
            forged,
            phase=phase,
            seed=seed,
            device="cuda:0",
            source_hash="c" * 64,
        )


def test_seed_envelope_rejects_non_20_20_60_split_identity() -> None:
    seed = 1
    phase = "confirmation_support"
    result = _support_result(seed)
    result["split_audit"] = {
        **result["split_audit"],
        "split_fractions": [0.40, 0.20, 0.40],
    }
    with pytest.raises(RuntimeError, match="role-split evidence"):
        runner._validate_seed_envelope(
            _seed_envelope(result, phase=phase, seed=seed),
            phase=phase,
            seed=seed,
            device="cuda:0",
            source_hash="c" * 64,
        )


def test_development_no_go_locks_confirmation_before_root_creation(
    tmp_path: Path,
) -> None:
    development_root = tmp_path / "development"
    confirmation_root = tmp_path / "confirmation"
    snapshot = _snapshot()
    metadata = _metadata(development_root, snapshot)
    runner._prepare_root(
        development_root,
        metadata,
        snapshot,
        resume=False,
    )
    runner._write_json(
        development_root / "FINAL_STATUS.json",
        {
            "protocol": runner.PROTOCOL,
            "dataset": runner.DATASET,
            "phase": "development",
            "status": "DEVELOPMENT_NO_GO",
            "coverage_generated": False,
        },
    )
    runner._write_json(
        development_root / "development_gate.json",
        {"development_admissible": False},
    )
    runner._write_json(
        development_root / "frozen_settings.json",
        {"status": "NOT_FROZEN_DEVELOPMENT_NO_GO"},
    )
    _seal_root(development_root)

    with pytest.raises(RuntimeError, match="locked by development NO-GO"):
        runner.run_confirmation(
            confirmation_root,
            development_root=development_root,
            config=SimpleNamespace(),
            devices=("cuda:0", "cuda:1"),
            prior_binding={},
            rng_audit=_rng_audit("confirmation"),
            resume=False,
        )
    assert not confirmation_root.exists()


def test_development_binding_recomputes_the_frozen_gate_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "development"
    snapshot = _snapshot()
    metadata = _metadata(root, snapshot)
    runner._prepare_root(root, metadata, snapshot, resume=False)
    gate = {
        "status": "DEVELOPMENT_GO",
        "development_admissible": True,
        "role_split": list(runner.ROLE_SPLIT),
        "bridge_candidate_id": runner.BRIDGE_CANDIDATE_ID,
        "coverage_generated": False,
    }
    final = {
        "protocol": runner.PROTOCOL,
        "dataset": runner.DATASET,
        "phase": "development",
        "status": "DEVELOPMENT_GO",
        "development_admissible": True,
        "role_split": list(runner.ROLE_SPLIT),
        "bridge_candidate_id": runner.BRIDGE_CANDIDATE_ID,
        "coverage_generated": False,
    }
    frozen = {
        "protocol": runner.PROTOCOL,
        "dataset": runner.DATASET,
        "status": "FROZEN_FOR_CONFIRMATION",
        "role_split": list(runner.ROLE_SPLIT),
        "bridge_candidate": runner._b02().to_dict(),
        "development_source_tree_sha256": metadata["source_tree_sha256"],
        "development_gate_sha256": runner._json_sha256(gate),
        "config_sha256": metadata["config_sha256"],
        "coverage_generated": False,
    }
    runner._write_json(root / "development_gate.json", gate)
    runner._write_json(root / "FINAL_STATUS.json", final)
    runner._write_json(root / "frozen_settings.json", frozen)
    _seal_root(root)

    binding, observed_frozen = runner._verify_development(root)
    assert observed_frozen == frozen
    assert binding["metadata_sha256"] == runner._file_sha256(root / "metadata.json")

    runner._write_json(root / "development_gate.json", {**gate, "tampered": True})
    _seal_root(root)
    with pytest.raises(RuntimeError, match="locked by development NO-GO"):
        runner._verify_development(root)


def test_precoverage_rng_map_contains_only_streams_the_runner_executes() -> None:
    config = runner.load_config(runner.CONFIG_PATH)
    seeds = (700_000, 700_010)
    mapping = runner._precoverage_rng_stream_mapping(seeds, config)

    assert len(mapping) == len(seeds) * len(runner.PRECOVERAGE_RNG_STREAM_SUFFIXES)
    assert len(set(mapping.values())) == len(mapping)
    assert all(name.endswith(runner.PRECOVERAGE_RNG_STREAM_SUFFIXES) for name in mapping)
    assert not any(
        token in name
        for name in mapping
        for token in ("summary_bootstrap", "calibration", "reference", "ACI_round")
    )


def test_configured_development_and_confirmation_actual_streams_are_disjoint() -> None:
    config = runner.load_config(runner.CONFIG_PATH)
    development_seeds = tuple(
        seed for block in runner.DEVELOPMENT_BLOCKS.values() for seed in block
    )
    development = runner._precoverage_rng_stream_mapping(development_seeds, config)
    confirmation = runner._precoverage_rng_stream_mapping(
        runner.CONFIRMATION_SEEDS,
        config,
    )

    assert runner._cross_bank_rng_collisions(development, confirmation) == {}
    assert len(development) == len(development_seeds) * 5
    assert len(confirmation) == len(runner.CONFIRMATION_SEEDS) * 5


def test_cross_bank_rng_overlap_is_explicit_and_validation_is_fail_closed() -> None:
    development = {"development/task": 10, "development/outcome_model": 11}
    confirmation = {"confirmation/task": 20, "confirmation/outcome_model": 11}

    assert runner._cross_bank_rng_collisions(development, {"confirmation/task": 20}) == {}
    assert runner._cross_bank_rng_collisions(development, confirmation) == {
        "confirmation/outcome_model": {
            "rng_id": 11,
            "development_streams": ["development/outcome_model"],
        }
    }

    audit = _rng_audit("confirmation")
    runner._validate_rng_audit(audit, phase="confirmation")
    audit["cross_bank_audit"] = {
        "collision_count": 1,
        "collisions": {"confirmation/outcome_model": 11},
    }
    with pytest.raises(RuntimeError, match="invalid confirmation RNG audit"):
        runner._validate_rng_audit(audit, phase="confirmation")


def test_protocol_adapter_uses_the_frozen_20_20_60_role_split() -> None:
    config = runner.load_config(runner.CONFIG_PATH)
    seeds = tuple(config.confirmation_seeds[:2])
    protocol = runner._protocol_for(seeds, config)

    assert protocol.split_fractions == (0.20, 0.20, 0.60)
    assert tuple(protocol.datasets) == (runner.DATASET,)
    assert protocol.datasets[runner.DATASET].seeds == seeds
