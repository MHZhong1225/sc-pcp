from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "scripts/run_controlled_clinical_fidelity_v4_science_publish_retry_r1.py"
)


def _load_runner():
    name = "test_run_controlled_clinical_fidelity_v4_science_publish_retry_r1"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


@pytest.fixture(scope="module")
def contract():
    config = runner.load_retry_config()
    amendment = runner.load_retry_amendment(config)
    return config, amendment


@pytest.fixture(scope="module")
def published_retry(contract, tmp_path_factory):
    config, amendment = contract
    output_root = tmp_path_factory.mktemp("v4-publish-retry") / "bundle"
    test_config = replace(config, output_root=output_root)
    original_before = runner._inventory(config.original_root)
    root_writes = []
    atomic_write = runner._atomic_write

    def tracked_write(path: Path, payload: bytes) -> None:
        atomic_write(path, payload)
        if output_root in path.parents:
            root_writes.append(path.relative_to(output_root).as_posix())

    runner._atomic_write = tracked_write
    try:
        runner.publish_retry(
            test_config,
            amendment,
            independent_audit_go_sha256=_audit_hash(test_config, amendment),
            resume=False,
        )
    finally:
        runner._atomic_write = atomic_write

    assert runner._inventory(config.original_root) == original_before
    assert root_writes[-1] == "COMPLETE"
    return test_config, amendment


def test_source_controlled_contract_pins_all_scientific_bytes(contract) -> None:
    config, amendment = contract
    incident = amendment["incident"]
    inventory = runner._amendment_inventory(amendment)

    assert config.output_root == runner.OUTPUT_ROOT
    assert len(inventory) == 154
    assert sum(row["bytes"] for row in inventory) == 24_426_407
    assert runner._inventory_sha256(inventory) == (
        "dfc88c08a2f55f73c0c0ad50a06d196b64778577dd6ff8ce6f63107fa5394771"
    )
    assert incident["missing_root_commits"] == ["manifest.json", "COMPLETE"]
    assert incident["mismatch_count"] == 12
    assert amendment["retry"]["formal_science_execution_permitted"] is False
    assert amendment["retry"]["fresh_science_claimed"] is False
    assert amendment["retry"]["independent_science_claimed"] is False
    assert incident["rng_binding"]["publish_retry_rng_streams_executed"] == 0


def test_read_only_incident_validation_preserves_original_root(contract) -> None:
    config, amendment = contract
    before = runner._inventory(config.original_root)
    observed = runner.validate_original_root(config, amendment)
    source_delta = runner.validate_authorized_source_delta(config, amendment)
    after = runner._inventory(config.original_root)

    assert observed == before == after
    assert source_delta["all_other_archived_sources_exact"] is True
    assert source_delta["changed_existing_sources"] == [
        {
            "path": "scripts/run_controlled_clinical_fidelity_v4_science.py",
            "archived_sha256": (
                "b9b910137439fa18a61f59aecd9437f7452883218495dcaed55d3de79c45d709"
            ),
            "repaired_sha256": (
                "d329537ccbf53040fb87c1707e01569266dea0f3c81ac273dc9dad6ab4ef74e6"
            ),
            "change": "canonical_JSON_roundtrip_only",
        }
    ]


def test_forensic_payload_and_rng_hashes_are_independently_recomputed(
    contract,
) -> None:
    config, amendment = contract
    runner._validate_forensic_bindings(config.original_root, amendment)

    changed = deepcopy(amendment)
    changed["incident"]["science_scope"]["row_identity_map_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="science payload forensic binding differs"):
        runner._validate_forensic_bindings(config.original_root, changed)

    changed = deepcopy(amendment)
    changed["incident"]["rng_binding"]["eligible_inherited_stream_count"] -= 1
    with pytest.raises(RuntimeError, match="inherited RNG forensic binding differs"):
        runner._validate_forensic_bindings(config.original_root, changed)


def test_publish_requires_exact_read_only_audit_hash(
    contract,
    tmp_path: Path,
) -> None:
    config, amendment = contract
    test_config = replace(config, output_root=tmp_path / "wrong-audit")
    with pytest.raises(RuntimeError, match="exact source-delta hash"):
        runner.publish_retry(
            test_config,
            amendment,
            independent_audit_go_sha256="0" * 64,
            resume=False,
        )
    assert not test_config.output_root.exists()


def test_publish_is_exact_semantic_and_complete_true_last(published_retry) -> None:
    config, amendment = published_retry
    runner.validate_published_retry(config, amendment)
    runner.publish_retry(
        config,
        amendment,
        independent_audit_go_sha256=_audit_hash(config, amendment),
        resume=True,
    )

    original = runner._amendment_inventory(amendment)
    for entry in original:
        assert runner._file_sha256(config.output_root / entry["path"]) == entry[
            "sha256"
        ]
    manifest = json.loads((config.output_root / "manifest.json").read_text())
    assert manifest["scientific_artifact_count"] == 154
    assert manifest["administrative_artifact_count"] == 3
    assert manifest["artifact_count"] == 157
    assert (config.output_root / "COMPLETE").is_file()


def test_completed_validate_uses_zero_scientific_rng_and_no_formal_dispatch(
    published_retry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, amendment = published_retry

    def forbidden(*args, **kwargs):
        raise AssertionError("publish validation attempted scientific RNG or dispatch")

    for name in (
        "default_rng",
        "RandomState",
        "seed",
        "rand",
        "randn",
        "random",
        "random_sample",
        "randint",
        "permutation",
    ):
        if hasattr(np.random, name):
            monkeypatch.setattr(np.random, name, forbidden)
    for name in (
        "Generator",
        "manual_seed",
        "seed",
        "rand",
        "randn",
        "randint",
        "randperm",
    ):
        if hasattr(runner.science.torch, name):
            monkeypatch.setattr(runner.science.torch, name, forbidden)
    for name in ("manual_seed", "manual_seed_all"):
        if hasattr(runner.science.torch.cuda, name):
            monkeypatch.setattr(runner.science.torch.cuda, name, forbidden)
    monkeypatch.setattr(runner.science, "run_post_confirmation_science", forbidden)
    monkeypatch.setattr(runner.science, "_overlap_worker", forbidden)
    monkeypatch.setattr(runner.science, "_science_worker", forbidden)

    runner.validate_published_retry(config, amendment)


def test_validation_fails_closed_on_scientific_tamper_and_extra_symlink(
    published_retry,
) -> None:
    config, amendment = published_retry
    source = config.output_root / "eligibility/eicu.json"
    original = source.read_bytes()
    source.write_bytes(original + b" ")
    try:
        with pytest.raises(RuntimeError, match="copied scientific artifact differs"):
            runner.validate_published_retry(config, amendment)
    finally:
        source.write_bytes(original)

    extra = config.output_root / "unexpected-link"
    extra.symlink_to(config.output_root / "metadata.json")
    try:
        with pytest.raises(RuntimeError, match="symbolic links are forbidden"):
            runner.validate_published_retry(config, amendment)
    finally:
        extra.unlink()
    runner.validate_published_retry(config, amendment)


def test_resume_removes_complete_when_completed_bundle_is_tampered(
    published_retry,
) -> None:
    config, amendment = published_retry
    artifact = config.output_root / "eligibility/eicu.json"
    complete = config.output_root / "COMPLETE"
    artifact_bytes = artifact.read_bytes()
    complete_bytes = complete.read_bytes()
    artifact.write_bytes(artifact_bytes + b" ")
    try:
        with pytest.raises(RuntimeError, match="copied scientific artifact differs"):
            runner.publish_retry(
                config,
                amendment,
                independent_audit_go_sha256=_audit_hash(config, amendment),
                resume=True,
            )
        assert not complete.exists()
    finally:
        artifact.write_bytes(artifact_bytes)
        complete.write_bytes(complete_bytes)
    runner.validate_published_retry(config, amendment)


def test_wrong_audit_hash_does_not_mutate_completed_root(published_retry) -> None:
    config, amendment = published_retry
    complete = config.output_root / "COMPLETE"
    before = complete.read_bytes()
    with pytest.raises(RuntimeError, match="exact source-delta hash"):
        runner.publish_retry(
            config,
            amendment,
            independent_audit_go_sha256="0" * 64,
            resume=True,
        )
    assert complete.read_bytes() == before


def test_partial_resume_rejects_premature_manifest(contract, tmp_path: Path) -> None:
    _, amendment = contract
    root = tmp_path / "partial"
    root.mkdir()
    first = runner._amendment_inventory(amendment)[0]
    source = runner.ORIGINAL_ROOT / first["path"]
    target = root / first["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    (root / "manifest.json").write_text("{}")

    with pytest.raises(RuntimeError, match="premature manifest"):
        runner._validate_partial_root(root, runner._amendment_inventory(amendment))


def test_complete_write_failure_removes_root_complete(
    contract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, amendment = contract
    test_config = replace(config, output_root=tmp_path / "postcommit-failure")
    atomic_write = runner._atomic_write

    def write_then_fail(path: Path, payload: bytes) -> None:
        atomic_write(path, payload)
        if path == test_config.output_root / "COMPLETE":
            raise RuntimeError("forced COMPLETE fsync failure")

    monkeypatch.setattr(runner, "_atomic_write", write_then_fail)
    monkeypatch.setattr(runner, "validate_scientific_semantics", lambda *args: None)
    with pytest.raises(RuntimeError, match="forced COMPLETE fsync failure"):
        runner.publish_retry(
            test_config,
            amendment,
            independent_audit_go_sha256=_audit_hash(test_config, amendment),
            resume=False,
        )
    assert not (test_config.output_root / "COMPLETE").exists()


def test_postcommit_validation_failure_removes_root_complete(
    contract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, amendment = contract
    test_config = replace(config, output_root=tmp_path / "postcommit-validation")
    validate = runner.validate_retry_root

    def validate_then_fail(*args, require_complete: bool, **kwargs) -> None:
        validate(*args, require_complete=require_complete, **kwargs)
        if require_complete:
            raise RuntimeError("forced postcommit validation failure")

    monkeypatch.setattr(runner, "validate_retry_root", validate_then_fail)
    monkeypatch.setattr(runner, "validate_scientific_semantics", lambda *args: None)
    with pytest.raises(RuntimeError, match="forced postcommit validation failure"):
        runner.publish_retry(
            test_config,
            amendment,
            independent_audit_go_sha256=_audit_hash(test_config, amendment),
            resume=False,
        )
    assert not (test_config.output_root / "COMPLETE").exists()


def _audit_hash(config, amendment) -> str:
    source_delta = runner.validate_authorized_source_delta(config, amendment)
    return runner._canonical_sha256(source_delta)
