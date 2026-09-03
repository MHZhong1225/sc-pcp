from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from scpcp.controlled_clinical_fidelity_v6_mimic_cxr import (
    CONFIRMATION_MAPPING_SHA256,
    DEVELOPMENT_MAPPING_SHA256,
    IndependentAudit,
    independent_audit_attestation_sha256,
    load_fidelity_v6_config,
    terminal_candidate,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/controlled_clinical_fidelity_v6_mimic_cxr.yaml"


def _load_runner():
    path = ROOT / "scripts/run_controlled_clinical_fidelity_v6_mimic_cxr.py"
    spec = importlib.util.spec_from_file_location(
        "run_controlled_clinical_fidelity_v6_mimic_cxr", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _metrics(*, ratio: float = 0.8, structural: bool = True) -> dict[str, object]:
    return {
        name: threshold * ratio for name, threshold in runner.K0_THRESHOLDS.items()
    } | {"structural_invariants": structural}


def _development_rows(
    *, failing_index: int | None = None, structural_index: int | None = None
) -> list[dict[str, object]]:
    return [
        {
            "metrics": _metrics(
                ratio=1.2 if index == failing_index else 0.8,
                structural=index != structural_index,
            )
        }
        for index in range(20)
    ]


def _confirmation_rows(pass_count: int, *, structural_count: int = 20):
    return [
        {
            "passed": index < pass_count,
            "metrics": {"structural_invariants": index < structural_count},
        }
        for index in range(20)
    ]


def _partial_metadata(root: Path, phase: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "phase": phase,
        "source_tree_sha256": "a" * 64,
        "devices": ["cuda:0", "cuda:1"],
        "source_snapshot": {
            "archive_path": "provenance/source.tar",
            "manifest_path": "provenance/source.json",
        },
    }
    if phase == "confirmation":
        metadata["frozen_settings"] = {"theta": terminal_candidate().to_dict()}
    (root / "provenance").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "provenance/source.tar").write_bytes(b"source")
    (root / "provenance/source.json").write_text("{}")
    return metadata


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, bytes | str], ...]:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append((relative, "symlink", os.readlink(path)))
        elif path.is_dir():
            rows.append((relative, "directory", ""))
        else:
            rows.append((relative, "file", path.read_bytes()))
    return tuple(rows)


def test_read_only_audit_binds_both_v5_roots_and_fresh_rng() -> None:
    config = load_fidelity_v6_config(CONFIG)
    parent = runner.validate_parent_v5_bundles(config)
    development = runner.audit_development_reuse(config, parent_binding=parent)
    confirmation = runner.audit_confirmation_rng(config)

    assert parent["failed_confirmation_reclassified_as_development_only"] is True
    assert parent["public_failed_confirmation"] == {
        "support_pass_count": 20,
        "k0_pass_count": 18,
        "structural_pass_count": 20,
        "failed_seeds": [
            {
                "seed": 119_120,
                "stage": 1,
                "outcome": 0,
                "maximum_signed_residual_w1": 0.2956583463636382,
            },
            {
                "seed": 119_180,
                "stage": 3,
                "outcome": 0,
                "maximum_signed_residual_w1": 0.3200767965422734,
            },
        ],
    }
    assert development["stream_count"] == 200
    assert development["mapping_sha256"] == DEVELOPMENT_MAPPING_SHA256
    assert development["unauthorized_collision_count"] == 0
    assert development["scientific_freshness_claimed"] is False
    assert confirmation["new_rng_stream_count"] == 341
    assert confirmation["new_rng_stream_mapping_sha256"] == (
        CONFIRMATION_MAPPING_SHA256
    )
    assert confirmation["collision_count"] == 0
    assert confirmation["formal_rng_consumed"] is False
    assert confirmation["v6_source_contract"]["exact_v6_file_count"] == 5
    assert (
        confirmation["v6_source_contract_sha256"]
        == confirmation["v6_source_contract"]["combined_sha256"]
    )


def test_independent_attestation_is_exact_and_pending_locks_formal() -> None:
    config = load_fidelity_v6_config(CONFIG)
    audit = runner.audit_confirmation_rng(config)
    unsigned = IndependentAudit(
        status="GO",
        attestation_sha256=None,
        expected_prior_count=audit["prior_rng_id_count"],
        expected_prior_sha256=audit["prior_rng_id_sha256"],
        expected_artifact_count=audit["artifact_rng_id_count"],
        expected_artifact_sha256=audit["artifact_rng_id_sha256"],
        expected_source_count=audit["source_declared_rng_id_count"],
        expected_source_sha256=audit["source_declared_rng_id_sha256"],
        expected_v6_source_contract_sha256=audit["v6_source_contract_sha256"],
    )
    signed = replace(
        unsigned,
        attestation_sha256=independent_audit_attestation_sha256(unsigned),
    )
    authorized = replace(config, independent_audit=signed)
    authorized.validate(require_audit_go=True)
    runner._validate_frozen_audit_snapshot(authorized, audit)

    drifted_source = {**audit, "v6_source_contract_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="source contract differs"):
        runner._validate_frozen_audit_snapshot(authorized, drifted_source)

    with pytest.raises(RuntimeError, match="snapshot differs"):
        runner._validate_frozen_audit_snapshot(
            replace(
                authorized,
                independent_audit=replace(signed, expected_prior_count=0),
            ),
            audit,
        )
    pending = replace(
        config,
        independent_audit=IndependentAudit(
            status="PENDING",
            attestation_sha256=None,
            expected_prior_count=None,
            expected_prior_sha256=None,
            expected_artifact_count=None,
            expected_artifact_sha256=None,
            expected_source_count=None,
            expected_source_sha256=None,
            expected_v6_source_contract_sha256=None,
        ),
    )
    with pytest.raises(RuntimeError, match="independent audit"):
        pending.validate(require_audit_go=True)


def test_development_requires_twenty_of_twenty_in_each_exposed_lineage() -> None:
    passing = {lineage: _development_rows() for lineage in runner.DEVELOPMENT_LINEAGES}
    gate = runner._development_gate(passing)
    assert gate["status"] == "DEVELOPMENT_GATE_GO"
    assert gate["scientific_candidate_count"] == 1
    assert gate["selector_present"] is False
    assert gate["grid_present"] is False
    assert gate["candidate_seed_deletions"] == 0
    assert gate["b02_anchor_role"] == "regression_only_not_a_candidate"

    one_failure = dict(passing)
    one_failure["v5_failed_confirmation"] = _development_rows(failing_index=19)
    assert runner._development_gate(one_failure)["status"] == "DEVELOPMENT_GATE_NO_GO"

    structural_failure = dict(passing)
    structural_failure["v5_development"] = _development_rows(structural_index=0)
    failed = runner._development_gate(structural_failure)
    summary = failed["lineage_summaries"]["v5_development"]
    assert failed["status"] == "DEVELOPMENT_GATE_NO_GO"
    assert summary["pass_count"] == 20
    assert summary["structural_pass_count"] == 19
    assert summary["q95_seed_ratio"] == pytest.approx(0.8)
    assert summary["mean_seed_ratio"] == pytest.approx(0.8)
    assert summary["seed_ratios"][0] == pytest.approx(0.8)
    runner._json_sha256(failed)


def test_confirmation_gate_keeps_support_k0_and_structural_thresholds() -> None:
    theta = terminal_candidate()
    support = _confirmation_rows(20)
    gate = runner._confirmation_gate(theta, support, _confirmation_rows(19))
    assert gate["status"] == "CONFIRMATION_GATE_GO"
    assert gate["support_pass_count"] == 20
    assert gate["k0_pass_count"] == 19
    assert gate["structural_pass_count"] == 20
    assert gate["candidate_seed_deletions"] == 0
    assert gate["further_bridge_repair_permitted"] is False
    assert gate["terminal_no_v7"] is True

    assert (
        runner._confirmation_gate(theta, support, _confirmation_rows(18))["status"]
        == "CONFIRMATION_GATE_NO_GO"
    )
    assert (
        runner._confirmation_gate(
            theta, support, _confirmation_rows(19, structural_count=19)
        )["status"]
        == "CONFIRMATION_GATE_NO_GO"
    )
    with pytest.raises(RuntimeError, match="opened after support NO-GO"):
        runner._confirmation_gate(theta, _confirmation_rows(18), _confirmation_rows(20))


def test_development_seed_envelope_binds_lineage_and_no_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = runner.v2.load_extension_config(runner.V2_CONFIG_PATH)
    seed = 92_600
    preset = replace(
        protocol.datasets[runner.DATASET],
        seeds=runner.DEVELOPMENT_LINEAGES["v5_development"],
    )
    candidate = terminal_candidate()
    candidate_hash = runner._json_sha256([candidate.to_dict()])
    result = {
        "seed": seed,
        "dataset": runner.DATASET,
        "phase": "development_k0",
        "coverage_generated": False,
        "theta": candidate.to_dict(),
        "development_lineage": "v5_development",
        "candidate_count": 1,
        "selector_opened": False,
        "grid_opened": False,
        "metrics": _metrics(),
        "passed": True,
        "normalized_seed_ratio": 0.8,
        "structural_failure_ratio_is_infinite": False,
        "systematic_replay": {key: None for key in runner._K0_DETAIL_KEYS},
        "context_identity": {key: None for key in runner._CONTEXT_IDENTITY_KEYS},
        "split_audit": {key: None for key in runner._SPLIT_AUDIT_KEYS},
    }
    payload = {
        "protocol": runner.PROTOCOL,
        "phase": "development_k0",
        "dataset": runner.DATASET,
        "seed": seed,
        "device": "cuda:0",
        "source_tree_sha256": "a" * 64,
        "candidate_contract_sha256": candidate_hash,
        "result": result,
    }
    monkeypatch.setattr(runner.v5run, "_validate_k0_candidate_row", lambda row: None)
    arguments = {
        "phase": "development_k0",
        "preset": preset,
        "seed": seed,
        "device": "cuda:0",
        "source_hash": "a" * 64,
        "candidate_hash": candidate_hash,
        "candidates": (candidate,),
    }
    runner._validate_seed_payload(payload, **arguments)

    result["development_lineage"] = "v5_failed_confirmation"
    with pytest.raises(RuntimeError, match="development role"):
        runner._validate_seed_payload(payload, **arguments)

    result["development_lineage"] = "v5_development"
    result["grid_opened"] = True
    with pytest.raises(RuntimeError, match="development role"):
        runner._validate_seed_payload(payload, **arguments)

    result["grid_opened"] = False
    result["systematic_replay"]["coverage"] = [0.9]
    with pytest.raises(RuntimeError, match="forbidden v6 seed-result content"):
        runner._validate_seed_payload(payload, **arguments)


def test_rng_collision_is_no_go_without_an_attestation_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_fidelity_v6_config(CONFIG)
    monkeypatch.setattr(
        runner.v5run,
        "_artifact_rng_ids",
        lambda root, excluded_roots: {120_000},
    )
    monkeypatch.setattr(
        runner.v2,
        "_source_declared_seeds",
        lambda root, excluded_paths: set(),
    )
    audit = runner.audit_confirmation_rng(config)
    assert audit["status"] == "collision_detected"
    assert audit["collision_count"] == 1
    assert audit["proposed_independent_audit"] is None


def test_manifest_is_exact_rejects_symlinks_and_firewall_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    (root / "k0_fidelity").mkdir(parents=True)
    (root / "metadata.json").write_text("{}")
    (root / "k0_fidelity/COMPLETE").write_text("complete\n")
    (root / "COMPLETE").write_text("old-root-marker\n")
    runner._write_manifest(root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert {entry["path"] for entry in manifest["artifacts"]} == {
        "metadata.json",
        "k0_fidelity/COMPLETE",
    }
    runner._verify_manifest(root)

    (root / "alias.json").symlink_to(root / "metadata.json")
    with pytest.raises(RuntimeError, match="symlink forbidden"):
        runner._verify_manifest(root)

    firewall = tmp_path / "firewall"
    firewall.mkdir()
    (firewall / "coverage.json").write_text("{}")
    with pytest.raises(RuntimeError, match="forbidden result path"):
        runner._assert_no_forbidden_result_paths(firewall)


def test_prelaunch_audit_requires_both_formal_roots_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development = tmp_path / "development"
    confirmation = tmp_path / "confirmation"
    monkeypatch.setattr(runner, "DEVELOPMENT_ROOT", development)
    monkeypatch.setattr(runner, "CONFIRMATION_ROOT", confirmation)
    runner._assert_formal_roots_absent()
    development.mkdir()
    with pytest.raises(RuntimeError, match="absent formal roots"):
        runner._assert_formal_roots_absent()


def test_resume_with_mismatched_metadata_fails_before_any_write(
    tmp_path: Path,
) -> None:
    root = tmp_path / "development"
    root.mkdir()
    (root / "metadata.json").write_text("{}")
    with pytest.raises(RuntimeError, match="resume metadata differs"):
        runner._prepare_root(
            root,
            {"protocol": runner.PROTOCOL},
            {},
            resume=True,
        )
    assert json.loads((root / "metadata.json").read_text()) == {}


@pytest.mark.parametrize("phase", ["development", "confirmation"])
def test_clean_partial_root_is_recoverable_before_workers(
    tmp_path: Path, phase: str
) -> None:
    root = tmp_path / phase
    root.mkdir()
    metadata = _partial_metadata(root, phase)
    runner._validate_incomplete_resume_root(
        root,
        phase=phase,
        metadata=metadata,
        config=load_fidelity_v6_config(CONFIG),
    )


def test_exact_terminal_crash_fragments_are_recoverable_but_stale_are_not(
    tmp_path: Path,
) -> None:
    gate_path = Path("development_gate.json")
    final_path = Path("FINAL_STATUS.json")
    manifest_path = Path("manifest.json")
    expected = {gate_path: {"gate": "exact"}, final_path: {"final": "exact"}}
    order = (gate_path, final_path, manifest_path)
    root = tmp_path / "terminal_fragments"
    root.mkdir()

    (root / gate_path).write_text(json.dumps(expected[gate_path]))
    runner._validate_partial_terminal_fragments(
        root,
        expected=expected,
        order=order,
    )
    (root / final_path).write_text(json.dumps(expected[final_path]))
    runner._validate_partial_terminal_fragments(
        root,
        expected=expected,
        order=order,
    )
    runner._write_manifest(root)
    runner._validate_partial_terminal_fragments(
        root,
        expected=expected,
        order=order,
    )

    (root / final_path).write_text(json.dumps({"final": "stale"}))
    with pytest.raises(RuntimeError, match="stale v6 terminal fragment"):
        runner._validate_partial_terminal_fragments(
            root,
            expected=expected,
            order=order,
        )

    out_of_order = tmp_path / "out_of_order"
    out_of_order.mkdir()
    (out_of_order / final_path).write_text(json.dumps(expected[final_path]))
    with pytest.raises(RuntimeError, match="out of commit order"):
        runner._validate_partial_terminal_fragments(
            out_of_order,
            expected=expected,
            order=order,
        )


@pytest.mark.parametrize(
    ("phase", "corruption"),
    [
        ("confirmation", "support_symlink"),
        ("confirmation", "k0_symlink"),
        ("development", "coverage_file"),
        ("development", "arbitrary_extra"),
        ("development", "unexpected_phase_file"),
        ("development", "wrong_seed_filename"),
        ("development", "invalid_seed_envelope"),
        ("development", "phase_complete_missing_seeds"),
        ("confirmation", "k0_before_support"),
        ("development", "stale_manifest"),
        ("development", "stale_gate"),
        ("development", "stale_final"),
    ],
)
def test_corrupt_partial_resume_starts_zero_workers_and_writes_zero_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    corruption: str,
) -> None:
    root = tmp_path / "partial"
    root.mkdir()
    metadata = _partial_metadata(root, phase)
    outside = tmp_path / "outside"
    outside.mkdir()

    if corruption == "support_symlink":
        (root / "support").symlink_to(outside, target_is_directory=True)
    elif corruption == "k0_symlink":
        (root / "k0_fidelity").symlink_to(outside, target_is_directory=True)
    elif corruption == "coverage_file":
        (root / "coverage.json").write_text("{}")
    elif corruption == "arbitrary_extra":
        (root / "extra.json").write_text("{}")
    elif corruption == "unexpected_phase_file":
        (root / "k0_fidelity/v5_development").mkdir(parents=True)
        (root / "k0_fidelity/v5_development/notes.json").write_text("{}")
    elif corruption == "wrong_seed_filename":
        (root / "k0_fidelity/v5_development").mkdir(parents=True)
        (root / "k0_fidelity/v5_development/seed_999999.json").write_text("{}")
    elif corruption == "invalid_seed_envelope":
        (root / "k0_fidelity/v5_development").mkdir(parents=True)
        (root / "k0_fidelity/v5_development/seed_092600.json").write_text("{}")
    elif corruption == "phase_complete_missing_seeds":
        (root / "k0_fidelity/v5_development").mkdir(parents=True)
        (root / "k0_fidelity/v5_development/COMPLETE").write_text("complete\n")
    elif corruption == "k0_before_support":
        (root / "k0_fidelity").mkdir()
    elif corruption == "stale_manifest":
        (root / "manifest.json").write_text("{}")
    elif corruption == "stale_gate":
        (root / "development_gate.json").write_text("{}")
    elif corruption == "stale_final":
        (root / "FINAL_STATUS.json").write_text("{}")
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(corruption)

    before = _tree_snapshot(root)
    calls = {"workers": 0, "writes": 0}

    def forbidden_worker(*args, **kwargs):
        del args, kwargs
        calls["workers"] += 1
        raise AssertionError("worker must not start from a corrupt partial root")

    def forbidden_write(*args, **kwargs):
        del args, kwargs
        calls["writes"] += 1
        raise AssertionError("corrupt partial-root validation must remain read-only")

    class ForbiddenExecutor:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            calls["workers"] += 1
            raise AssertionError("executor must not start from a corrupt partial root")

    source_snapshot = {"contract": metadata["source_snapshot"]}
    monkeypatch.setattr(
        runner.v5run.v4,
        "_active_source_contract",
        lambda: (metadata["source_tree_sha256"], source_snapshot),
    )
    monkeypatch.setattr(runner, "_root_metadata", lambda **kwargs: metadata)
    monkeypatch.setattr(runner, "_prepare_root", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_run_seed_phase", forbidden_worker)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ForbiddenExecutor)
    monkeypatch.setattr(runner, "_write_json", forbidden_write)
    monkeypatch.setattr(runner, "_write_text", forbidden_write)
    config = load_fidelity_v6_config(CONFIG)

    with pytest.raises(RuntimeError):
        if phase == "development":
            runner.run_development(
                root,
                config=config,
                devices=("cuda:0", "cuda:1"),
                parent_binding={},
                development_audit={},
                confirmation_audit={},
                resume=True,
            )
        else:
            frozen = {
                "development_source_tree_sha256": metadata["source_tree_sha256"],
                "theta": terminal_candidate().to_dict(),
            }
            monkeypatch.setattr(
                runner,
                "_verify_development_for_confirmation",
                lambda *args, **kwargs: ({}, frozen),
            )
            runner.run_confirmation(
                root,
                development_root=tmp_path / "development_evidence",
                config=config,
                devices=("cuda:0", "cuda:1"),
                parent_binding={},
                development_audit={},
                confirmation_audit={},
                resume=True,
            )

    assert calls == {"workers": 0, "writes": 0}
    assert _tree_snapshot(root) == before
