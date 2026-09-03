from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.run_controlled_clinical_mimic_cxr_environment_support as cxr
import scripts.run_controlled_clinical_mimic_cxr_environment_support_science as science
from scpcp.controlled_clinical_extension import GAMMAS, METHODS
from scpcp.controlled_clinical_mimic_cxr_environment_support import (
    CONFIRMATION_SEEDS,
    K0_THRESHOLDS,
    ROLE_SPLIT,
    load_config,
)


def _anchor(seed: int = 633_000) -> science.ConfirmationAnchor:
    split = {
        "role_unique_patient_counts": {
            "predictor": 398,
            "fidelity": 398,
            "environment": 1197,
        },
        "role_episode_counts": {
            "predictor": 398,
            "fidelity": 398,
            "environment": 1197,
        },
        "role_patient_id_sha256": {
            "predictor": "1" * 64,
            "fidelity": "2" * 64,
            "environment": "3" * 64,
        },
        "patient_sets_pairwise_disjoint": True,
        "split_fractions": list(ROLE_SPLIT),
    }
    base_without_hash: dict[str, Any] = {
        "outcome_model_state_sha256": "4" * 64,
        "behavior_policy_state_sha256": "5" * 64,
        "q_low": 0.1,
        "q_high": 0.9,
        "n_actions": 3,
        "action_mapping": {"0": 0, "1": 1, "2": 2},
        "action_costs": [0.0, 0.5, 1.0],
        "donor_neighbors": 10_000,
        "donor_bandwidth": 2.0,
        "transition_ridge": 0.001,
        "environment_patient_id_sha256": "3" * 64,
        "split_patient_id_sha256": split["role_patient_id_sha256"],
        "split_fractions": list(ROLE_SPLIT),
        "active_config_sha256": "6" * 64,
    }
    base = {
        **base_without_hash,
        "combined_sha256": science._json_sha256(base_without_hash),
    }
    kernel_without_hash = {
        "base_nuisance_context_sha256": base["combined_sha256"],
        "outcome_model_state_sha256": base["outcome_model_state_sha256"],
        "behavior_policy_state_sha256": base["behavior_policy_state_sha256"],
        "split_patient_id_sha256": split["role_patient_id_sha256"],
        "active_config_sha256": base["active_config_sha256"],
        "theta": cxr._b02().to_dict(),
        "state_kernel": {"seed": seed},
        "outcome_bridge": {"seed": seed},
    }
    kernel = {
        **kernel_without_hash,
        "combined_sha256": science._json_sha256(kernel_without_hash),
    }
    return science.ConfirmationAnchor(
        split_audit=split,
        base_context_identity=base,
        kernel_context_identity=kernel,
        support_passed=True,
        k0_passed=True,
    )


def _bundle(tmp_path: Path, seeds: tuple[int, ...]) -> science.GateBundle:
    config = load_config(cxr.CONFIG_PATH)
    protocol = cxr._protocol_for(CONFIRMATION_SEEDS, config)
    theta = cxr._b02()
    anchors = {seed: _anchor(seed) for seed in seeds}
    contract = {"root": str(tmp_path), "seeds": list(seeds)}
    return science.GateBundle(
        config=config,
        protocol=protocol,
        preset=protocol.datasets["mimic_cxr"],
        theta=theta,
        prespecified_seeds=CONFIRMATION_SEEDS,
        support_k0_eligible_seeds=seeds,
        anchors=anchors,
        eligibility_record={"seeds": list(seeds)},
        seed_to_device={
            seed: ("cuda:0" if index % 2 == 0 else "cuda:1")
            for index, seed in enumerate(seeds)
        },
        active_source_tree_sha256="a" * 64,
        confirmation_binding={"combined_sha256": "b" * 64},
        science_rng_audit={"full_mapping_sha256": "c" * 64},
        rng_stream_mapping_sha256="c" * 64,
        science_contract={"primary_metric": science.PRIMARY_METRIC},
        contract=contract,
    )


def test_science_contract_is_the_frozen_yaml_contract() -> None:
    config = load_config(cxr.CONFIG_PATH)
    protocol = cxr._protocol_for(CONFIRMATION_SEEDS, config)

    contract = science._validate_frozen_science_contract(protocol, cxr._b02())

    assert tuple(contract["gammas"]) == GAMMAS
    assert tuple(contract["methods"]) == METHODS
    assert contract["primary_gamma"] == -4.0
    assert contract["primary_metric"] == "min_t mean_seed(C_seed,t)"
    assert contract["calibration_trajectories"] == 3_000
    assert contract["grid_trajectories"] == 1_000
    assert contract["evaluation_trajectories"] == 20_000
    assert contract["target_adaptation_trajectories"] == {
        "Standard CP": 0,
        "ACI": 2_000,
        "MFCS": 0,
        "SPCI": 2_000,
        "PRC": 2_000,
        "SC-PCP": 0,
    }
    assert contract["role_split"] == [0.2, 0.2, 0.6]
    assert contract["bridge_candidate_id"].startswith("B02_")
    assert CONFIRMATION_SEEDS == config.confirmation_seeds
    assert CONFIRMATION_SEEDS[0] == 633_000


def test_confirmation_rng_audit_separates_precoverage_and_full_science_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(cxr.CONFIG_PATH)
    protocol = cxr._protocol_for(CONFIRMATION_SEEDS, config)
    precoverage = cxr._precoverage_rng_stream_mapping(CONFIRMATION_SEEDS, config)
    metadata = {
        "rng_audit": {
            "new_rng_stream_mapping": precoverage,
            "new_rng_stream_mapping_sha256": science._json_sha256(precoverage),
        }
    }
    monkeypatch.setattr(cxr, "_validate_rng_audit", lambda *_args, **_kwargs: None)
    observed_exclusions: dict[str, set[Path]] = {}

    def artifact_ids(_root: Path, *, excluded_roots: set[Path]) -> set[int]:
        observed_exclusions["roots"] = excluded_roots
        return {7}

    def source_ids(_root: Path, *, excluded_paths: set[Path]) -> set[int]:
        observed_exclusions["paths"] = excluded_paths
        return {11}

    monkeypatch.setattr(cxr, "_artifact_rng_ids", artifact_ids)
    monkeypatch.setattr(science.v2, "_source_declared_seeds", source_ids)

    audit = science._validate_confirmation_rng(metadata, protocol, config)

    frozen = yaml.safe_load(cxr.CONFIG_PATH.read_text())[
        "prelaunch_integrity_amendment"
    ]["replacement_rng_audit"]
    assert len(precoverage) == 100
    assert audit["full_stream_count"] == 341
    assert audit["full_mapping_sha256"] == frozen["full_confirmation_mapping_sha256"]
    assert audit["confirmation_precoverage_mapping_sha256"] == science._json_sha256(
        precoverage
    )
    assert audit["historical_collision_count"] == 0
    assert observed_exclusions["roots"] == {
        cxr.DEVELOPMENT_ROOT,
        science.CONFIRMATION_ROOT,
        science.OUTPUT_ROOT,
    }
    assert Path(science.__file__).resolve() in observed_exclusions["paths"]
    assert Path(__file__).resolve() in observed_exclusions["paths"]

    tampered = {
        "rng_audit": {
            **metadata["rng_audit"],
            "new_rng_stream_mapping": {},
        }
    }
    with pytest.raises(RuntimeError, match="precoverage RNG mapping differs"):
        science._validate_confirmation_rng(tampered, protocol, config)

    colliding = next(iter(audit["full_mapping"].values()))
    monkeypatch.setattr(cxr, "_artifact_rng_ids", lambda *_args, **_kwargs: {colliding})
    with pytest.raises(RuntimeError, match="full science RNG collision"):
        science._validate_confirmation_rng(metadata, protocol, config)


def test_wsc_is_minimum_of_stagewise_seed_mean() -> None:
    coverage = np.asarray(
        [
            [0.70, 1.00],
            [1.00, 0.70],
        ]
    )

    observed = science.marginal_worst_stage_coverage(coverage)

    assert observed == pytest.approx(0.85)
    assert observed != pytest.approx(float(coverage.min(axis=1).mean()))


def test_science_reporting_distinguishes_support_and_overlap_cohorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support_k0_seeds = CONFIRMATION_SEEDS
    overlap_seeds = support_k0_seeds[:-1]
    target_coverage = [0.91, 0.92]
    coverage_matrix = np.asarray(
        [target_coverage for _seed in overlap_seeds], dtype=np.float64
    )
    rows = [
        {
            "seed": seed,
            "gamma": gamma,
            "methods": {
                method: {
                    "selection_available": True,
                    "target_coverage": target_coverage,
                }
                for method in METHODS
            },
        }
        for gamma in GAMMAS
        for seed in overlap_seeds
    ]
    method_summary = {
        "n_selected": len(overlap_seeds),
        "n_prespecified": len(support_k0_seeds),
        "n_k0_eligible": len(overlap_seeds),
        "selection_rate": len(overlap_seeds) / len(support_k0_seeds),
        "selection_rate_ci95": science._wilson_interval(
            len(overlap_seeds), len(support_k0_seeds)
        ),
        "target_marginal_worst_coverage": science.marginal_worst_stage_coverage(
            coverage_matrix
        ),
        "target_mean_coverage": float(coverage_matrix.mean(axis=0).mean()),
        "target_coverage_by_stage": coverage_matrix.mean(axis=0).tolist(),
    }
    inherited = {
        "seeds_k0_eligible": list(overlap_seeds),
        "coverage_conditioning": "successful method selection among K0-eligible seeds",
        "aggregates": [
            {
                "gamma": gamma,
                "n_k0_eligible_seeds": len(overlap_seeds),
                "methods": {method: dict(method_summary) for method in METHODS},
            }
            for gamma in GAMMAS
        ],
    }
    monkeypatch.setattr(
        science.v2,
        "summarize_science",
        lambda *_args, **_kwargs: inherited,
    )
    preset = cxr._protocol_for(
        CONFIRMATION_SEEDS, load_config(cxr.CONFIG_PATH)
    ).datasets["mimic_cxr"]

    summary = science.summarize_science(
        rows,
        preset=preset,
        support_k0_eligible_seeds=support_k0_seeds,
        selected_seeds=overlap_seeds,
        bootstrap_contract={"resamples": 10_000},
    )
    audit = science.coverage_audit(
        rows,
        summary=summary,
        support_k0_eligible_seeds=support_k0_seeds,
        selected_seeds=overlap_seeds,
    )

    assert len(summary["seeds_support_k0_eligible"]) == 20
    assert len(summary["seeds_support_k0_overlap_eligible"]) == 19
    assert summary["seeds_k0_eligible"] == list(support_k0_seeds)
    assert summary["coverage_conditioning"] == (
        "successful method selection among support/K0/overlap-eligible seeds"
    )
    assert (
        "before donor-overlap screening"
        in summary["compatibility_field_semantics"]["seeds_k0_eligible"]
    )
    for aggregate in summary["aggregates"]:
        assert aggregate["n_k0_eligible_seeds"] == 20
        assert aggregate["n_support_k0_eligible_seeds"] == 20
        assert aggregate["n_support_k0_overlap_eligible_seeds"] == 19
        for cell in aggregate["methods"].values():
            assert cell["n_k0_eligible"] == 20
            assert cell["n_support_k0_eligible"] == 20
            assert cell["n_support_k0_overlap_eligible"] == 19
    assert audit["seeds_support_k0_eligible"] == list(support_k0_seeds)
    assert audit["seeds_support_k0_overlap_eligible"] == list(overlap_seeds)
    assert audit["science_eligible_seeds"] == list(overlap_seeds)
    assert audit["support_k0_eligible_seed_count"] == 20
    assert audit["support_k0_overlap_eligible_seed_count"] == 19
    assert all(
        record["n_support_k0_overlap_eligible"] == 19 for record in audit["records"]
    )


def test_overlap_gate_requires_a_complete_joint_19_of_20_bank() -> None:
    eligible = CONFIRMATION_SEEDS
    passing = [{"seed": seed, "passed": True} for seed in eligible]
    passing[-1]["passed"] = False

    go = science.summarize_overlap(
        passing,
        prespecified_seeds=CONFIRMATION_SEEDS,
        support_k0_eligible_seeds=eligible,
    )

    assert go["status"] == "OVERLAP_GO"
    assert go["joint_overlap_pass_count"] == 19
    assert go["science_may_start"] is True
    assert go["overlap_completed_seed_count"] == 20
    assert go["seed_deletions"] == 0

    passing[-2]["passed"] = False
    no_go = science.summarize_overlap(
        passing,
        prespecified_seeds=CONFIRMATION_SEEDS,
        support_k0_eligible_seeds=eligible,
    )
    assert no_go["status"] == "OVERLAP_NO_GO"
    assert no_go["joint_overlap_pass_count"] == 18
    assert no_go["science_may_start"] is False

    with pytest.raises(RuntimeError, match="exact eligible seed bank"):
        science.summarize_overlap(
            passing[:-1],
            prespecified_seeds=CONFIRMATION_SEEDS,
            support_k0_eligible_seeds=eligible,
        )


def test_science_unlock_requires_committed_overlap_go(
    tmp_path: Path,
) -> None:
    seeds = CONFIRMATION_SEEDS[:19]
    gates = _bundle(tmp_path, seeds)
    rows = [{"seed": seed, "passed": True} for seed in seeds]
    overlap = science.summarize_overlap(
        rows,
        prespecified_seeds=CONFIRMATION_SEEDS,
        support_k0_eligible_seeds=seeds,
    )
    unlock = science._science_unlock(gates, overlap, seeds)
    science._write_json(tmp_path / "donor_overlap/summary.json", overlap)
    science._write_text(
        tmp_path / "donor_overlap/COMPLETE",
        f"overlap-complete summary_sha256={science._json_sha256(overlap)}\n",
    )
    science._write_json(tmp_path / "SCIENCE_UNLOCK.json", unlock)

    assert science._valid_science_unlock(tmp_path, gates) is True

    (tmp_path / "donor_overlap/COMPLETE").write_text("stale\n")
    assert science._valid_science_unlock(tmp_path, gates) is False

    rows[-1]["passed"] = False
    no_go = science.summarize_overlap(
        rows,
        prespecified_seeds=CONFIRMATION_SEEDS,
        support_k0_eligible_seeds=seeds,
    )
    with pytest.raises(RuntimeError, match="requires the frozen overlap GO"):
        science._science_unlock(gates, no_go, no_go["passed_seeds"])


def test_reconstructed_context_must_equal_both_confirmation_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = CONFIRMATION_SEEDS[0]
    anchor = _anchor(seed)
    protocol = cxr._protocol_for(CONFIRMATION_SEEDS, load_config(cxr.CONFIG_PATH))
    preset = replace(protocol.datasets["mimic_cxr"], seeds=(seed,))
    base_context = SimpleNamespace(splits=object())
    context = SimpleNamespace(environment=object())
    monkeypatch.setattr(
        science.v2, "_prepare_extension_context", lambda *_args, **_kwargs: base_context
    )
    monkeypatch.setattr(
        science.v5, "_context_with_theta", lambda *_args, **_kwargs: context
    )
    monkeypatch.setattr(
        science.v2, "_split_audit", lambda _splits: dict(anchor.split_audit)
    )
    monkeypatch.setattr(
        science.v2,
        "_context_identity",
        lambda _context: dict(anchor.base_context_identity),
    )
    monkeypatch.setattr(
        science.v5,
        "_candidate_context_identity",
        lambda *_args, **_kwargs: dict(anchor.kernel_context_identity),
    )

    rebuilt, base, kernel = science._reconstruct_context(
        seed,
        preset,
        "cuda:0",
        cxr._b02(),
        anchor,
        protocol,
    )

    assert rebuilt is context
    assert base == anchor.base_context_identity
    assert kernel == anchor.kernel_context_identity

    wrong = replace(
        anchor,
        base_context_identity={**anchor.base_context_identity, "q_low": 0.2},
    )
    with pytest.raises(RuntimeError, match="reconstructed B02 context differs"):
        science._reconstruct_context(
            seed,
            preset,
            "cuda:0",
            cxr._b02(),
            wrong,
            protocol,
        )


def test_k0_artifact_binds_base_kernel_and_20_20_60_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = CONFIRMATION_SEEDS[0]
    anchor = _anchor(seed)
    metrics = {name: threshold * 0.5 for name, threshold in K0_THRESHOLDS.items()} | {
        "structural_invariants": True
    }
    result = {
        "seed": seed,
        "dataset": "mimic_cxr",
        "passed": True,
        "metrics": metrics,
        "normalized_k0_ratio": 0.5,
        "theta": cxr._b02().to_dict(),
        "role_split": list(ROLE_SPLIT),
        "systematic_replay": {},
        "base_context_identity": dict(anchor.base_context_identity),
        "kernel_context_identity": dict(anchor.kernel_context_identity),
        "split_audit": dict(anchor.split_audit),
        "coverage_generated": False,
    }
    monkeypatch.setattr(science.v5, "_validate_k0_candidate_row", lambda _row: None)
    monkeypatch.setattr(science.v2, "_valid_context_identity", lambda _value: True)

    science._validate_k0_result(result, theta=cxr._b02(), seed=seed)

    broken = {
        **result,
        "kernel_context_identity": {
            **result["kernel_context_identity"],
            "base_nuisance_context_sha256": "0" * 64,
        },
    }
    with pytest.raises(RuntimeError, match="K0 context identity differs"):
        science._validate_k0_result(broken, theta=cxr._b02(), seed=seed)


def test_phase_payload_rejects_context_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = CONFIRMATION_SEEDS[0]
    anchor = _anchor(seed)
    protocol = cxr._protocol_for(CONFIRMATION_SEEDS, load_config(cxr.CONFIG_PATH))
    preset = replace(protocol.datasets["mimic_cxr"], seeds=(seed,))
    result = {
        "seed": seed,
        "dataset": "mimic_cxr",
        "phase": science.OVERLAP_PHASE,
        "theta": cxr._b02().to_dict(),
        "split_audit": anchor.split_audit,
        "base_context_identity": anchor.base_context_identity,
        "kernel_context_identity": anchor.kernel_context_identity,
        "confirmation_anchor_identity_sha256": science._json_sha256(anchor.to_dict()),
    }
    payload = science._phase_payload(
        phase=science.OVERLAP_PHASE,
        preset=preset,
        seed=seed,
        device="cuda:0",
        theta=cxr._b02(),
        anchor=anchor,
        source_hash="a" * 64,
        gate_contract_sha256="b" * 64,
        rng_mapping_sha256="c" * 64,
        result=result,
    )
    monkeypatch.setattr(science, "_validate_overlap_result", lambda *_args: None)
    science._validate_phase_payload(
        payload,
        phase=science.OVERLAP_PHASE,
        preset=preset,
        seed=seed,
        device="cuda:0",
        theta=cxr._b02(),
        anchor=anchor,
        source_hash="a" * 64,
        gate_contract_sha256="b" * 64,
        rng_mapping_sha256="c" * 64,
    )

    payload["result"]["kernel_context_identity"] = {"combined_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="context differs"):
        science._validate_phase_payload(
            payload,
            phase=science.OVERLAP_PHASE,
            preset=preset,
            seed=seed,
            device="cuda:0",
            theta=cxr._b02(),
            anchor=anchor,
            source_hash="a" * 64,
            gate_contract_sha256="b" * 64,
            rng_mapping_sha256="c" * 64,
        )


def test_resume_metadata_and_manifest_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_contract = {"fixed": True}
    metadata = {
        "read_only_audit_status": "GO",
        "read_only_audit_go_sha256": science._json_sha256(gate_contract),
        "gate_contract_sha256": science._json_sha256(gate_contract),
        "source_snapshot": {},
    }
    tmp_path.mkdir(exist_ok=True)
    science._write_json(tmp_path / "metadata.json", metadata)
    monkeypatch.setattr(
        science.v4science, "_verify_source_snapshot", lambda *_args: None
    )

    science._prepare_root(tmp_path, metadata, {}, resume=True)

    changed = {**metadata, "source_snapshot": {"changed": True}}
    with pytest.raises(RuntimeError, match="resume metadata differs"):
        science._prepare_root(tmp_path, changed, {}, resume=True)

    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"value": 1}))
    science._write_or_verify_manifest(tmp_path)
    science._verify_manifest(tmp_path)
    artifact.write_text(json.dumps({"value": 2}))
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        science._verify_manifest(tmp_path)


def test_parent_source_snapshot_recomputes_the_tree_hash(tmp_path: Path) -> None:
    source_files = [
        ("src/scpcp/example.py", b"VALUE = 1\n"),
        ("configs/example.yaml", b"value: 1\n"),
    ]
    tree_digest = hashlib.sha256()
    archive_stream = io.BytesIO()
    manifest_rows = []
    with tarfile.open(
        fileobj=archive_stream, mode="w", format=tarfile.PAX_FORMAT
    ) as archive:
        for relative, content in source_files:
            name = relative.encode("utf-8")
            tree_digest.update(len(name).to_bytes(4, "big"))
            tree_digest.update(name)
            tree_digest.update(len(content).to_bytes(8, "big"))
            tree_digest.update(content)
            info = tarfile.TarInfo(relative)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
            manifest_rows.append(
                {
                    "path": relative,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    archive_bytes = archive_stream.getvalue()
    source_manifest = {
        "protocol": "parent",
        "format": "deterministic_uncompressed_pax_tar",
        "file_count": len(manifest_rows),
        "files": manifest_rows,
    }
    manifest_bytes = json.dumps(source_manifest).encode("utf-8")
    archive_path = tmp_path / "provenance/source.tar"
    manifest_path = tmp_path / "provenance/source.json"
    archive_path.parent.mkdir()
    archive_path.write_bytes(archive_bytes)
    manifest_path.write_bytes(manifest_bytes)
    contract = {
        "archive_path": archive_path.relative_to(tmp_path).as_posix(),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_bytes": len(archive_bytes),
        "manifest_path": manifest_path.relative_to(tmp_path).as_posix(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_bytes": len(manifest_bytes),
        "file_count": len(manifest_rows),
    }
    source_hash = tree_digest.hexdigest()
    metadata = {
        "source_snapshot": contract,
        "source_tree_sha256": source_hash,
    }

    science._validate_parent_source_snapshot(
        tmp_path, metadata, active_source_hash=source_hash
    )

    with pytest.raises(RuntimeError, match="source-tree hash differs"):
        science._validate_parent_source_snapshot(
            tmp_path, metadata, active_source_hash="0" * 64
        )
