from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_controlled_clinical_fidelity_v4_science.py"


def _load_runner():
    name = "test_run_controlled_clinical_fidelity_v4_science"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


science = _load_runner()


@pytest.fixture(scope="module")
def gates() -> science.GateBundle:
    return science.verify_gate_bundle(devices=("cuda:0", "cuda:1"))


def test_read_only_gate_audit_binds_exact_three_dataset_contract(
    gates: science.GateBundle,
) -> None:
    assert tuple(gates.datasets) == science.CONFIRMED_DATASETS
    assert {
        dataset: len(gate.eligible_seeds)
        for dataset, gate in gates.datasets.items()
    } == {"mimic_iv": 20, "eicu": 19, "inspire": 20}
    assert gates.contract["unopened_datasets"] == ["mimic_cxr"]
    assert gates.contract["pooled_or_universal_claim_permitted"] is False
    assert gates.contract["rng_role"] == (
        "exact_reuse_of_reserved_first_confirmation_mapping"
    )
    assert gates.contract["new_rng_bank_claimed"] is False
    assert gates.rng_audit["new_rng_stream_mapping_sha256"] == (
        science.EXPECTED_CONFIRMATION_MAPPING_SHA256
    )


def test_audit_passes_when_numpy_and_torch_rng_consumers_are_forbidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_rng(*args, **kwargs):
        raise AssertionError("read-only audit attempted to consume RNG")

    numpy_consumers = (
        "default_rng",
        "RandomState",
        "seed",
        "rand",
        "randn",
        "random",
        "random_sample",
        "randint",
        "permutation",
    )
    for name in numpy_consumers:
        if hasattr(np.random, name):
            monkeypatch.setattr(np.random, name, forbidden_rng)
    torch_consumers = (
        "Generator",
        "manual_seed",
        "seed",
        "rand",
        "randn",
        "randint",
        "randperm",
    )
    for name in torch_consumers:
        if hasattr(science.torch, name):
            monkeypatch.setattr(science.torch, name, forbidden_rng)
    for name in ("manual_seed", "manual_seed_all"):
        if hasattr(science.torch.cuda, name):
            monkeypatch.setattr(science.torch.cuda, name, forbidden_rng)

    audited = science.verify_gate_bundle(devices=("cuda:0", "cuda:1"))
    assert tuple(audited.datasets) == science.CONFIRMED_DATASETS


def test_audit_preserves_exact_parent_root_inventories() -> None:
    roots = (science.DEVELOPMENT_ROOT, science.CONFIRMATION_ROOT)
    before = {str(root): science._read_only_root_inventory(root) for root in roots}
    science.verify_gate_bundle(devices=("cuda:0", "cuda:1"))
    after = {str(root): science._read_only_root_inventory(root) for root in roots}
    assert after == before
    assert science._file_sha256(science.DEVELOPMENT_ROOT / "COMPLETE") == (
        "6d05b9e8e1411c7d75f2247a5d8c8fc2479557fb3d365165682e4e706efff610"
    )
    assert science._file_sha256(science.CONFIRMATION_ROOT / "COMPLETE") == (
        "e156b19e9cc086a0506aa8cef34f9807ddad66ef670a2b1571705b97924b3fcf"
    )


def test_audit_exception_has_zero_write_and_zero_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = (science.DEVELOPMENT_ROOT, science.CONFIRMATION_ROOT)
    before = {str(root): science._read_only_root_inventory(root) for root in roots}

    def forced_failure() -> None:
        raise RuntimeError("forced read-only audit failure")

    def forbidden_mutation(*args, **kwargs):
        raise AssertionError("audit attempted a filesystem mutation")

    monkeypatch.setattr(science, "_validate_fixed_confirmation_files", forced_failure)
    monkeypatch.setattr(science, "_atomic_write", forbidden_mutation)
    monkeypatch.setattr(science, "_write_json", forbidden_mutation)
    monkeypatch.setattr(science, "_write_text", forbidden_mutation)
    monkeypatch.setattr(science, "_unlink_root_complete", forbidden_mutation)
    monkeypatch.setattr(Path, "unlink", forbidden_mutation)
    monkeypatch.setattr(os, "replace", forbidden_mutation)

    with pytest.raises(RuntimeError, match="forced read-only audit failure"):
        science.verify_gate_bundle(devices=("cuda:0", "cuda:1"))
    after = {str(root): science._read_only_root_inventory(root) for root in roots}
    assert after == before


def test_eicu_failed_support_seed_is_explicitly_unavailable_for_every_method(
    gates: science.GateBundle,
) -> None:
    gate = gates.datasets["eicu"]
    assert 116_150 not in gate.eligible_seeds
    assert gate.eligibility_record["selection_rate_denominator"] == 20
    assert gate.eligibility_record["maximum_possible_selection_rate"] == 0.95
    assert gate.eligibility_record["unavailable_for_every_method"] == [116_150]
    record = next(
        row
        for row in gate.eligibility_record["seed_records"]
        if row["seed"] == 116_150
    )
    assert record == {
        "seed": 116_150,
        "support_passed": False,
        "k0_passed": True,
        "science_eligible": False,
        "exclusion_reason": {
            "code": "SUPPORT_FAILED",
            "minimum_unique_patients": 16,
            "failed_cells": [[0, 3, 16]],
            "k0_passed_but_does_not_restore_support": True,
        },
    }


def test_science_contract_freezes_primary_gamma_metrics_and_intervals() -> None:
    assert tuple(science.GAMMAS) == (-4.0, -2.0, 0.0, 2.0, 4.0)
    assert science.PRIMARY_GAMMA == -4.0
    assert science.TARGET_COVERAGE == 0.90
    assert science.PRIMARY_METRIC == "min_t mean_seed(target_coverage_seed_t)"
    assert tuple(science.METHODS) == (
        "Standard CP",
        "ACI",
        "MFCS",
        "SPCI",
        "PRC",
        "SC-PCP",
    )
    assert science.SCIENCE_CONTRACT["uncertainty"] == {
        "selection": "Wilson 95% interval on all 20 prespecified seeds",
        "stage_coverage": "two-sided Student-t across method-selected eligible seeds",
        "mean_coverage": "two-sided Student-t across method-selected eligible seeds",
        "stage_normalized_width": "two-sided Student-t across method-selected eligible seeds",
        "mean_normalized_width": "two-sided Student-t across method-selected eligible seeds",
        "wsc": "10000-draw complete-seed-vector percentile bootstrap",
        "paired": "10000-draw paired-seed-vector percentile bootstrap",
    }


def test_science_metadata_is_json_canonical_across_fresh_and_resume(
    gates: science.GateBundle,
    tmp_path: Path,
) -> None:
    archive = b"archived source\n"
    source_manifest = b"{}\n"
    archive_sha256 = science.hashlib.sha256(archive).hexdigest()
    manifest_sha256 = science.hashlib.sha256(source_manifest).hexdigest()
    source_snapshot = {
        "archive_bytes": archive,
        "manifest_bytes": source_manifest,
        "contract": {
            "archive_path": f"provenance/source_snapshot_{archive_sha256}.tar",
            "archive_sha256": archive_sha256,
            "archive_bytes": len(archive),
            "manifest_path": f"provenance/source_manifest_{manifest_sha256}.json",
            "manifest_sha256": manifest_sha256,
            "manifest_bytes": len(source_manifest),
            "file_count": 1,
        },
    }
    gate_hash = science._json_sha256(gates.contract)
    metadata = science._science_metadata(
        gates,
        devices=("cuda:0", "cuda:1"),
        independent_audit_go_sha256=gate_hash,
        source_snapshot=source_snapshot["contract"],
    )

    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
    assert not _contains_tuple(metadata)

    output_root = tmp_path / "fresh_then_resume"
    science._prepare_root(
        output_root,
        metadata,
        source_snapshot,
        resume=False,
    )
    science._prepare_root(
        output_root,
        metadata,
        source_snapshot,
        resume=True,
    )


def test_summary_keeps_eicu_denominator_and_primary_vs_descriptive_roles(
    gates: science.GateBundle,
    tmp_path: Path,
) -> None:
    gate = gates.datasets["eicu"]
    rows = _synthetic_rows(gate)
    bootstrap = science._ensure_bootstrap_artifacts(tmp_path, gate.preset)
    summary = science._science_summary(
        rows,
        dataset_gate=gate,
        interpretation_status="EMPIRICAL_OVERLAP_SCREEN_PASSED",
        bootstrap_contract=bootstrap,
    )

    primary = summary["aggregates"][0]
    standard = primary["methods"]["Standard CP"]
    assert primary["gamma"] == -4.0
    assert primary["analysis_role"] == "confirmatory_gamma_minus_4_endpoint"
    assert standard["n_selected"] == 19
    assert standard["n_prespecified"] == 20
    assert standard["selection_rate"] == 0.95
    assert standard["selection_rate_ci95"] == science._wilson_interval(19, 20)
    assert standard["target_wsc_gap_to_0.90"] == pytest.approx(
        standard["target_marginal_worst_coverage"] - 0.90
    )
    assert standard["target_mean_coverage_gap_to_0.90"] == pytest.approx(
        standard["target_mean_coverage"] - 0.90
    )
    assert standard["point_attainment_at_0.90"] is True
    assert isinstance(standard["wsc_interval_attainment_at_0.90"], bool)
    assert standard["point_eligible"] is True
    assert standard["point_eligibility_rule"] == (
        "selection_rate>=0.95 and target_marginal_worst_coverage>=0.90"
    )
    expected_stage_ci = science._student_t_interval_by_stage(
        np.asarray(
            [
                row["methods"]["Standard CP"]["target_coverage"]
                for row in rows
                if row["gamma"] == -4.0
            ],
            dtype=np.float64,
        )
    )
    assert standard["target_coverage_by_stage_ci95"] == expected_stage_ci

    for aggregate in summary["aggregates"][1:]:
        assert aggregate["analysis_role"] == "descriptive_signed_control_curve"
        assert aggregate["width_order_among_point_eligible"] == []
        assert set(aggregate["paired_scpcp_comparisons"]) == {"status"}
        assert aggregate["methods"]["SC-PCP"]["point_eligible"] is None
        assert (
            aggregate["methods"]["SC-PCP"]["point_attainment_at_0.90"]
            is None
        )


def test_global_overlap_commit_is_dataset_complete_not_a_pass_conjunction(
    gates: science.GateBundle,
    tmp_path: Path,
) -> None:
    interpretations = {
        "mimic_iv": "EMPIRICAL_OVERLAP_SCREEN_PASSED",
        "eicu": "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
        "inspire": "EMPIRICAL_OVERLAP_SCREEN_PASSED",
    }
    science._write_global_overlap_marker(tmp_path, gates, interpretations)
    assert science._valid_global_overlap_marker(tmp_path, gates)
    summary = json.loads((tmp_path / science.OVERLAP_PHASE / "summary.json").read_text())
    assert summary["science_may_start"] is True
    assert summary["cross_dataset_conjunction_used"] is False
    assert summary["datasets"] == interpretations


def test_manifest_binds_nested_complete_but_excludes_only_root_commits(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "COMPLETE").write_text("complete\n")
    (tmp_path / "payload.json").write_text("{}")
    (tmp_path / "COMPLETE").write_text("draft\n")
    science._write_manifest(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    paths = {entry["path"] for entry in manifest["artifacts"]}
    assert "nested/COMPLETE" in paths
    assert "payload.json" in paths
    assert "COMPLETE" not in paths
    assert "manifest.json" not in paths
    science._verify_manifest(tmp_path)
    (tmp_path / "nested" / "COMPLETE").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        science._verify_manifest(tmp_path)


def test_bootstrap_bank_remains_twenty_columns_for_eicu_projection(
    gates: science.GateBundle,
    tmp_path: Path,
) -> None:
    contract = science._ensure_bootstrap_artifacts(
        tmp_path,
        gates.datasets["eicu"].preset,
    )
    uniforms = np.load(tmp_path / "bootstrap_uniforms.npy", allow_pickle=False)
    assert uniforms.shape == (10_000, 20)
    assert contract["prespecified_seed_count"] == 20
    projected = science.v2._bootstrap_indices(uniforms, 19)
    assert projected.shape == (10_000, 19)
    assert int(projected.max()) <= 18


def _synthetic_rows(gate: science.DatasetGate) -> list[dict[str, object]]:
    rows = []
    horizon = gate.preset.horizon
    for seed_index, seed in enumerate(gate.eligible_seeds):
        for gamma in science.GAMMAS:
            methods = {}
            for method_index, method in enumerate(science.METHODS):
                coverage = [
                    0.912
                    + 0.0001 * seed_index
                    + 0.0002 * stage
                    + 0.0001 * method_index
                    for stage in range(horizon)
                ]
                width = [
                    1.0
                    + 0.002 * seed_index
                    + 0.01 * stage
                    + 0.02 * method_index
                    for stage in range(horizon)
                ]
                methods[method] = {
                    "selection_available": True,
                    "target_coverage": coverage,
                    "source_coverage": [value + 0.001 for value in coverage],
                    "target_normalized_width": width,
                    "prefix_ess_fraction": [0.5] * horizon,
                    "maximum_normalized_weight_share": [0.1] * horizon,
                }
            rows.append(
                {
                    "seed": seed,
                    "dataset": gate.preset.name,
                    "gamma": gamma,
                    "methods": methods,
                }
            )
    return rows


def _contains_tuple(value: object) -> bool:
    if isinstance(value, tuple):
        return True
    if isinstance(value, dict):
        return any(_contains_tuple(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_tuple(child) for child in value)
    return False
