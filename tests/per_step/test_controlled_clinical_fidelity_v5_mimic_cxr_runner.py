from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from scpcp.controlled_clinical_fidelity_v5_mimic_cxr import (
    IndependentAudit,
    bridge_candidates,
    independent_audit_attestation_sha256,
    load_fidelity_v5_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/controlled_clinical_fidelity_v5_mimic_cxr.yaml"


def _load_runner():
    path = ROOT / "scripts/run_controlled_clinical_fidelity_v5_mimic_cxr.py"
    spec = importlib.util.spec_from_file_location(
        "run_controlled_clinical_fidelity_v5_mimic_cxr", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_read_only_audit_binds_parents_and_fresh_rng() -> None:
    config = load_fidelity_v5_config(CONFIG)
    parent = runner.validate_parent_v4_bundles(config)
    development = runner.audit_development_reuse(config, parent_binding=parent)
    confirmation = runner.audit_confirmation_rng(config)
    assert parent["c13_development_pass_count"] == 18
    assert parent["c13_development_structural_pass_count"] == 20
    assert parent["cxr_confirmation_was_never_opened"] is True
    assert development["stream_count"] == 100
    assert development["authorized_lineage_collision_count"] == 100
    assert development["unauthorized_collision_count"] == 0
    assert confirmation["new_rng_stream_count"] == 341
    assert confirmation["collision_count"] == 0
    assert confirmation["new_rng_stream_mapping_sha256"] == (
        "43d832a650352de3e97fc6694178c61969e707a3658abb33958668080cf3e40e"
    )
    assert confirmation["formal_rng_consumed"] is False


def test_independent_audit_snapshot_is_exact_and_formal_lock_is_explicit() -> None:
    config = load_fidelity_v5_config(CONFIG)
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
    )
    frozen = replace(
        unsigned,
        attestation_sha256=independent_audit_attestation_sha256(unsigned),
    )
    authorized = replace(config, independent_audit=frozen)
    authorized.validate(require_audit_go=True)
    runner._validate_frozen_audit_snapshot(authorized, audit)
    with pytest.raises(RuntimeError, match="snapshot differs"):
        runner._validate_frozen_audit_snapshot(
            replace(
                authorized,
                independent_audit=replace(frozen, expected_prior_count=0),
            ),
            audit,
        )
    pending = replace(
        config,
        independent_audit=replace(
            config.independent_audit,
            status="PENDING",
            attestation_sha256=None,
        ),
    )
    with pytest.raises(RuntimeError, match="independent audit"):
        pending.validate(require_audit_go=True)


def test_manifest_includes_nested_complete_but_not_root_commit_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    (root / "repair").mkdir(parents=True)
    (root / "metadata.json").write_text("{}")
    (root / "repair/COMPLETE").write_text("complete\n")
    (root / "COMPLETE").write_text("old-root-marker\n")
    runner._write_manifest(root)
    manifest = json.loads((root / "manifest.json").read_text())
    paths = {entry["path"] for entry in manifest["artifacts"]}
    assert paths == {"metadata.json", "repair/COMPLETE"}
    runner._verify_manifest(root)
    (root / "repair/COMPLETE").write_text("changed\n")
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        runner._verify_manifest(root)


def test_action_diagnostics_mark_unavailable_cells_and_remain_non_gating() -> None:
    action = torch.tensor([0, 0, 1])
    replay_count = 2
    true_score = torch.tensor([0.1, 0.2, 0.3])
    replay_score = true_score.repeat_interleave(replay_count)
    true_residual = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
    replay_residual = true_residual.repeat_interleave(replay_count, dim=0)
    true_clinical = torch.arange(48, dtype=torch.float64).reshape(3, 16)
    replay_clinical = true_clinical.repeat_interleave(replay_count, dim=0)
    rows = runner._action_stratified_diagnostics(
        action=action,
        true_score=true_score,
        replay_score=replay_score,
        true_residual=true_residual,
        replay_residual=replay_residual,
        true_clinical=true_clinical,
        replay_clinical=replay_clinical,
        clinical_active=torch.ones(16, dtype=torch.bool),
        replay_count=replay_count,
        groups=((0, 1, 2, 3, 8, 9, 10, 11), (4, 5, 6, 7, 12, 13, 14, 15)),
    )
    assert [row["available"] for row in rows] == [True, True, False]
    assert [row["query_count"] for row in rows] == [2, 1, 0]
    assert rows[2]["score_ks"] is None
    assert rows[0]["signed_residual_w1_by_outcome"] == [0.0, 0.0]


def test_development_decision_never_deletes_a_seed() -> None:
    candidates = bridge_candidates()
    rows = []
    pass_counts = (18, 19, 20)
    for seed_index in range(20):
        candidate_rows = []
        for theta, pass_count in zip(candidates, pass_counts, strict=True):
            ratio = 0.8 if seed_index < pass_count else 1.2
            candidate_rows.append(
                {
                    "theta": theta.to_dict(),
                    "metrics": {
                        "maximum_score_ks": 0.1 * ratio,
                        "maximum_signed_residual_w1": 0.1,
                        "maximum_successor_mean_w1": 0.1,
                        "maximum_successor_q95_w1": 0.1,
                        "structural_invariants": True,
                    },
                }
            )
        rows.append({"candidates": candidate_rows})
    selection = runner._development_selection(rows)
    final = runner._development_final(selection)
    assert selection["winner"]["candidate_id"] == (
        "B02_pooled_successor_bridge_stage_one_hot"
    )
    assert selection["winner_summary"]["pass_count"] == 20
    assert selection["candidate_seed_deletions"] == 0
    assert final["candidate_seed_deletions"] == 0
    assert final["coverage_generated"] is False


def test_forbidden_science_or_coverage_artifact_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "coverage.json").write_text("{}")
    with pytest.raises(RuntimeError, match="forbidden result path"):
        runner._assert_no_forbidden_result_paths(root)
