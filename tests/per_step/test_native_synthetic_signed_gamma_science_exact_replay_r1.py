from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT / "scripts/run_native_synthetic_signed_gamma_science_exact_replay_r1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "test_run_native_synthetic_signed_gamma_science_exact_replay_r1",
    RUNNER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_amendment_pins_exact_failed_publish_inventory_and_discloses_rng_reuse() -> None:
    amendment = runner.load_incident_amendment()
    incident = amendment["incident"]
    retry = amendment["retry"]

    assert incident["artifact_count"] == len(incident["artifact_inventory"]) == 31
    assert incident["total_size_bytes"] == 10_379_251
    assert incident["artifact_inventory_sha256"] == runner.science._canonical_sha256(
        incident["artifact_inventory"]
    )
    assert incident["artifact_inventory_sha256"] == (
        "c36da6ac2700c5da0fe907e303a7864e85466aea912d661c5a216604625a2314"
    )
    assert incident["scientific_values_human_inspected"] is False
    assert retry["rng_reused"] is True
    assert retry["rng_fresh"] is False
    assert retry["rng_independent"] is False


def test_retry_config_differs_only_by_administrative_output_root() -> None:
    retry_raw = yaml.safe_load(runner.DEFAULT_RETRY_CONFIG.read_text(encoding="utf-8"))
    original_raw = yaml.safe_load(runner.ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    retry_output = retry_raw.pop("output_root")
    original_output = original_raw.pop("output_root")

    assert retry_raw == original_raw
    assert original_output == "results/work/native_synthetic_signed_gamma_six_method_science_v1"
    assert retry_output.endswith("_exact_replay_r1")
    config, equivalence = runner.load_retry_config()
    assert equivalence["only_difference"] == "output_root"
    assert runner._config_output_root(config) == runner.RETRY_ROOT


def test_output_root_resolution_is_workspace_relative_not_cwd_relative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, _ = runner.load_retry_config()
    monkeypatch.chdir(tmp_path)

    assert runner._config_output_root(config) == runner.RETRY_ROOT


def test_incident_inventory_mutation_fails_closed() -> None:
    incident = deepcopy(runner.load_incident_amendment()["incident"])
    incident["artifact_inventory"][0]["sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="incident inventory hash differs"):
        runner._validate_incident_contract(incident)


def test_quarantine_and_explicit_old_tree_repair_gate_validate_without_values() -> None:
    amendment = runner.load_incident_amendment()
    quarantine = runner.validate_quarantine_bundle(amendment)
    gate = runner.validate_repair_gate_amendment(amendment)

    assert quarantine["status"] == "QUARANTINED_FAILED_PUBLISH_FULLY_VALIDATED"
    assert quarantine["artifact_count"] == 31
    assert quarantine["original_root_absent"] is True
    assert quarantine["scientific_values_human_inspected"] is False
    assert gate.decision == "GO"
    assert gate.binding_sha256 == amendment["repair_gate"]["binding_sha256"]


def test_ordinary_gate_validation_remains_strict_on_old_source_tree() -> None:
    config, _ = runner.load_retry_config()

    with pytest.raises(RuntimeError, match="active source tree differs"):
        runner.science.verify_repair_gate(config)


def test_source_delta_allows_only_audited_repair_and_three_declarations() -> None:
    contract = runner.load_incident_amendment()["allowed_source_repair"]
    science_path = "scripts/run_native_synthetic_signed_gamma_science.py"
    preflight_path = "scripts/run_native_synthetic_signed_gamma_preflight.py"
    old = {
        science_path: {
            "path": science_path,
            "sha256": contract["old_science_runner_sha256"],
        },
        preflight_path: {
            "path": preflight_path,
            "sha256": contract["old_preflight_runner_sha256"],
        },
    }
    current = {
        science_path: {
            "path": science_path,
            "sha256": contract["repaired_science_runner_sha256"],
        },
        preflight_path: {
            "path": preflight_path,
            "sha256": contract["repaired_preflight_runner_sha256"],
        },
    }
    current.update(
        {
            path: {"path": path, "sha256": "a" * 64}
            for path in contract["permitted_added_administrative_sources"]
        }
    )

    observed = runner.validate_source_manifest_delta(old, current, contract)
    assert observed == (
        sorted(contract["exactly_changed_existing_sources"]),
        sorted(contract["permitted_added_administrative_sources"]),
        [],
    )

    current["scripts/unauthorized_science_change.py"] = {
        "path": "scripts/unauthorized_science_change.py",
        "sha256": "b" * 64,
    }
    with pytest.raises(RuntimeError, match="beyond the audited administrative repair"):
        runner.validate_source_manifest_delta(old, current, contract)


def test_live_source_delta_and_rng_audit_are_exact() -> None:
    amendment = runner.load_incident_amendment()
    config, _ = runner.load_retry_config()
    delta = runner.validate_allowed_source_delta(amendment)
    audit = runner.audit_retry_rng_ids(config, amendment)

    assert delta["changed_existing_paths"] == [
        "scripts/run_native_synthetic_signed_gamma_preflight.py",
        "scripts/run_native_synthetic_signed_gamma_science.py"
    ]
    assert len(delta["added_administrative_paths"]) == 3
    assert delta["missing_paths"] == []
    assert audit["raw_collision_count"] == 241
    assert audit["authorized_quarantine_collision_count"] == 241
    assert audit["unauthorized_collision_count"] == 0
    assert audit["missing_quarantine_collision_count"] == 0
    assert audit["rng_reused"] is True
    assert audit["rng_fresh"] is False
    assert audit["rng_independent"] is False


def test_collision_classifier_rejects_duplicate_or_missing_authority() -> None:
    mapping = {"science/a": 11, "science/b": 12}
    exact = runner.classify_retry_collisions(
        mapping,
        quarantine_ids={11, 12},
        other_artifact_ids=set(),
        source_ids=set(),
        external_ids=set(),
    )
    assert exact["raw_collision_count"] == 2
    assert exact["authorized_quarantine_collision_count"] == 2
    assert exact["unauthorized_collision_count"] == 0
    assert exact["missing_quarantine_collision_count"] == 0

    duplicate = runner.classify_retry_collisions(
        mapping,
        quarantine_ids={11, 12},
        other_artifact_ids={12},
        source_ids=set(),
        external_ids=set(),
    )
    assert duplicate["status"] == "unauthorized_failed_publish_replay"
    assert duplicate["authorized_quarantine_collision_count"] == 1
    assert duplicate["unauthorized_collision_count"] == 1
    assert duplicate["unauthorized_collisions"]["science/b"]["categories"] == [
        "exact_pinned_quarantine",
        "other_artifact",
    ]

    missing = runner.classify_retry_collisions(
        mapping,
        quarantine_ids={11},
        other_artifact_ids=set(),
        source_ids=set(),
        external_ids=set(),
    )
    assert missing["status"] == "unauthorized_failed_publish_replay"
    assert missing["missing_quarantine_collisions"] == {"science/b": 12}


def test_rogue_source_rng_use_is_unauthorized(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "rogue.py").write_text(
        "import torch\ntorch.manual_seed(121400)\n", encoding="utf-8"
    )
    scan = runner.preflight._source_rng_scan(tmp_path, excluded_paths=set())
    classified = runner.classify_retry_collisions(
        {"science/base_121400/task": 121400},
        quarantine_ids={121400},
        other_artifact_ids=set(),
        source_ids=set(scan["actual"]),
        external_ids=set(),
    )

    assert scan["actual"] == {121400}
    assert classified["unauthorized_collision_count"] == 1
    assert classified["authorized_quarantine_collision_count"] == 0


def test_dependency_contract_exemption_is_exact_and_rng_fields_stay_strict(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "metadata.json"
    artifact.write_text("{}\n", encoding="utf-8")
    exact_report = runner.preflight._empty_rng_scan()
    runner.preflight._collect_artifact_rng_fields(
        {
            "dependency_files": {
                "experiment_rng": {
                    "path": "src/scpcp/experiment.py",
                    "sha256": "a" * 64,
                    "size_bytes": 123,
                }
            }
        },
        exact_report,
        artifact_path=artifact,
        artifact_root=tmp_path,
    )
    assert exact_report["actual"] == set()

    numeric_report = runner.preflight._empty_rng_scan()
    runner.preflight._collect_artifact_rng_fields(
        {"dependency_files": {"experiment_rng": 121400}},
        numeric_report,
        artifact_path=artifact,
        artifact_root=tmp_path,
    )
    assert numeric_report["actual"] == {121400}

    container_report = runner.preflight._empty_rng_scan()
    runner.preflight._collect_artifact_rng_fields(
        {"formal_rng_ids": [121400, 121401]},
        container_report,
        artifact_path=artifact,
        artifact_root=tmp_path,
    )
    assert container_report["actual"] == {121400, 121401}

    with pytest.raises(RuntimeError, match="non-integer RNG value"):
        runner.preflight._collect_artifact_rng_fields(
            {"dependency_files": {"experiment_rng": "src/scpcp/experiment.py"}},
            runner.preflight._empty_rng_scan(),
            artifact_path=artifact,
            artifact_root=tmp_path,
        )
    with pytest.raises(RuntimeError, match="non-integer RNG value"):
        runner.preflight._collect_artifact_rng_fields(
            {
                "dependency_files": {
                    "experiment_rng": {
                        "path": "src/scpcp/experiment.py",
                        "sha256": "a" * 64,
                        "size_bytes": 123,
                        "extra": "not-an-exact-source-contract",
                    }
                }
            },
            runner.preflight._empty_rng_scan(),
            artifact_path=artifact,
            artifact_root=tmp_path,
        )


def test_structured_rng_config_recurses_but_scalar_list_and_string_stay_strict(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "metadata.json"
    artifact.write_text("{}\n", encoding="utf-8")
    structured = runner.preflight._empty_rng_scan()
    runner.preflight._collect_artifact_rng_fields(
        {
            "science_config": {
                "rng": {
                    "base_seeds": [121400, 121410],
                    "bootstrap_seed": 12140019,
                    "mapping_count": 3,
                    "mapping_sha256": "a" * 64,
                    "seed_namespace": "descriptive-only",
                }
            }
        },
        structured,
        artifact_path=artifact,
        artifact_root=tmp_path,
    )
    assert structured["actual"] == {121400, 121410, 12140019}

    scalar = runner.preflight._empty_rng_scan()
    runner.preflight._collect_artifact_rng_fields(
        {"rng": 121400},
        scalar,
        artifact_path=artifact,
        artifact_root=tmp_path,
    )
    assert scalar["actual"] == {121400}

    container = runner.preflight._empty_rng_scan()
    runner.preflight._collect_artifact_rng_fields(
        {"rng": [121400, 121410]},
        container,
        artifact_path=artifact,
        artifact_root=tmp_path,
    )
    assert container["actual"] == {121400, 121410}

    with pytest.raises(RuntimeError, match="non-integer RNG value"):
        runner.preflight._collect_artifact_rng_fields(
            {"rng": "not-an-integer"},
            runner.preflight._empty_rng_scan(),
            artifact_path=artifact,
            artifact_root=tmp_path,
        )


def test_multi_exclusion_scan_keeps_global_binary_binding_coordinates(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    quarantine = results / "quarantine"
    retry = results / "retry"
    metadata = results / "other" / "metadata.json"
    binary = results / "prior" / "bootstrap_indices.npy"
    quarantine.mkdir(parents=True)
    retry.mkdir()
    metadata.parent.mkdir()
    binary.parent.mkdir()
    (quarantine / "seed_11.json").write_text("{}\n", encoding="utf-8")
    (retry / "seed_12.json").write_text("{}\n", encoding="utf-8")
    (results / "seed_13.json").write_text("{}\n", encoding="utf-8")
    binary.write_bytes(b"global-root-bound-bootstrap")
    binding = {
        "binary_path": binary.relative_to(results).as_posix(),
        "field_path": "bootstrap.seed_index_matrix_path",
        "sha256": runner.science._file_sha256(binary),
    }
    metadata.write_text(
        json.dumps({"rng_audit": {"artifact_binary_rng_bindings": [binding]}}),
        encoding="utf-8",
    )

    report = runner._artifact_rng_scan_excluding(
        results,
        excluded_roots=(quarantine, retry),
    )

    assert report["actual"] == {13}
    assert report["binary_bindings"] == [
        {
            "metadata_path": "other/metadata.json",
            "field_path": "rng_audit.artifact_binary_rng_bindings.0.binary_path",
            "binary_path": "prior/bootstrap_indices.npy",
            "sha256": binding["sha256"],
        }
    ]


def _rows(seed: int) -> list[dict[str, object]]:
    return [
        {"seed": seed, "gamma": gamma, "payload": {"token": index}}
        for index, gamma in enumerate(runner.science.GAMMAS)
    ]


def test_row_and_summary_comparison_are_exact_and_tolerance_free() -> None:
    old_rows = _rows(7)
    exact_rows = runner.compare_row_payloads(old_rows, deepcopy(old_rows), expected_seed=7)
    changed_rows = deepcopy(old_rows)
    changed_rows[2]["payload"]["token"] = 99
    mismatch_rows = runner.compare_row_payloads(old_rows, changed_rows, expected_seed=7)

    assert exact_rows["status"] == "EXACT_REPLAY"
    assert exact_rows["exact_row_count"] == 5
    assert mismatch_rows["status"] == "NO_GO"
    assert mismatch_rows["exact_row_count"] == 4

    exact_summary = runner.compare_summary_payloads(
        {"count": 1, "nested": [True]}, {"count": 1, "nested": [True]}
    )
    type_changed_summary = runner.compare_summary_payloads(
        {"count": 1, "nested": [True]}, {"count": 1.0, "nested": [True]}
    )
    signed_zero_summary = runner.compare_summary_payloads(
        {"value": -0.0}, {"value": 0.0}
    )
    assert exact_summary["status"] == "EXACT_REPLAY"
    assert type_changed_summary["status"] == "NO_GO"
    assert signed_zero_summary["status"] == "NO_GO"


def test_bootstrap_array_comparison_requires_array_and_file_hash_identity() -> None:
    old = np.array([[0.1, 0.2]], dtype=np.float64)
    exact = runner.compare_array_payloads(
        old, old.copy(), quarantine_sha256="a" * 64, retry_sha256="a" * 64
    )
    changed_value = runner.compare_array_payloads(
        old,
        np.array([[0.1, 0.3]], dtype=np.float64),
        quarantine_sha256="a" * 64,
        retry_sha256="a" * 64,
    )
    changed_file = runner.compare_array_payloads(
        old, old.copy(), quarantine_sha256="a" * 64, retry_sha256="b" * 64
    )
    changed_dtype = runner.compare_array_payloads(
        old,
        old.astype(np.float32),
        quarantine_sha256="a" * 64,
        retry_sha256="a" * 64,
    )
    changed_shape = runner.compare_array_payloads(
        old,
        old.reshape(2, 1),
        quarantine_sha256="a" * 64,
        retry_sha256="a" * 64,
    )

    assert exact["exact"] is True
    assert changed_value["exact"] is False
    assert changed_file["exact"] is False
    assert changed_dtype["exact"] is False
    assert changed_shape["exact"] is False


def test_no_go_removes_complete_and_discloses_nonindependent_rng(tmp_path: Path) -> None:
    (tmp_path / "COMPLETE").write_text("invalid\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="NO_GO at summary_comparison"):
        runner.publish_no_go(tmp_path, stage="summary_comparison")

    payload = json.loads((tmp_path / runner.NO_GO_FILE).read_text(encoding="utf-8"))
    assert not (tmp_path / "COMPLETE").exists()
    assert payload["downstream_authorized"] is False
    assert payload["rng_reused"] is True
    assert payload["rng_fresh"] is False
    assert payload["rng_independent"] is False


def test_resumed_mismatched_seed_is_normalized_to_no_go_before_dispatch(
    tmp_path: Path,
) -> None:
    (tmp_path / "COMPLETE").write_text("stale\n", encoding="utf-8")
    payloads = {
        121400: {"row_replay_comparison": {"status": "NO_GO"}},
    }

    with pytest.raises(RuntimeError, match="NO_GO at seed_row_comparison"):
        runner.require_exact_seed_replays(tmp_path, payloads)

    status = json.loads((tmp_path / runner.NO_GO_FILE).read_text(encoding="utf-8"))
    assert status["stage"] == "seed_row_comparison"
    assert status["downstream_authorized"] is False
    assert not (tmp_path / "COMPLETE").exists()

    clean = tmp_path / "clean"
    clean.mkdir()
    runner.require_exact_seed_replays(
        clean,
        {121400: {"row_replay_comparison": {"status": "EXACT_REPLAY"}}},
    )
    assert list(clean.iterdir()) == []


def test_finalize_is_true_last_and_removes_complete_on_public_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    metadata: dict[str, object] = {}

    monkeypatch.setattr(runner, "expected_nonterminal_paths", lambda *args: set())

    def write_manifest(root: Path, **_: object) -> None:
        runner.science._write_json(root / "manifest.json", {"status": "precommit"})

    def validate_precommit(
        root: Path, *, expected_metadata: object, include_complete: bool
    ) -> tuple[dict[str, object], str]:
        assert expected_metadata is metadata
        assert include_complete is False
        assert (root / "manifest.json").is_file()
        assert not (root / "COMPLETE").exists()
        events.append("precommit")
        return metadata, "a" * 64

    def fail_public(root: Path, *, expected_metadata: object) -> None:
        assert expected_metadata is metadata
        assert (root / "COMPLETE").is_file()
        events.append("public")
        raise RuntimeError("injected completed-validator failure")

    monkeypatch.setattr(runner, "write_manifest", write_manifest)
    monkeypatch.setattr(runner, "validate_retry_bundle_contents", validate_precommit)
    monkeypatch.setattr(
        runner, "expected_complete_payload", lambda *args, **kwargs: {"status": "complete"}
    )
    monkeypatch.setattr(runner, "validate_completed_retry_bundle", fail_public)

    with pytest.raises(RuntimeError, match="injected completed-validator failure"):
        runner.finalize_root(tmp_path, metadata=metadata, config=object())

    assert events == ["precommit", "public"]
    assert (tmp_path / "manifest.json").is_file()
    assert not (tmp_path / "COMPLETE").exists()


def test_precommit_failure_never_writes_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata: dict[str, object] = {}
    writes: list[str] = []
    original_write_json = runner.science._write_json

    monkeypatch.setattr(runner, "expected_nonterminal_paths", lambda *args: set())

    def tracked_write(path: Path, payload: object) -> None:
        writes.append(path.name)
        original_write_json(path, payload)

    def write_manifest(root: Path, **_: object) -> None:
        runner.science._write_json(root / "manifest.json", {"status": "precommit"})

    def fail_precommit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected precommit failure")

    monkeypatch.setattr(runner.science, "_write_json", tracked_write)
    monkeypatch.setattr(runner, "write_manifest", write_manifest)
    monkeypatch.setattr(runner, "validate_retry_bundle_contents", fail_precommit)

    with pytest.raises(RuntimeError, match="injected precommit failure"):
        runner.finalize_root(tmp_path, metadata=metadata, config=object())

    assert writes == ["manifest.json"]
    assert not (tmp_path / "COMPLETE").exists()


def test_postpublish_baseexception_removes_complete_and_fsyncs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata: dict[str, object] = {}
    complete_presence_at_fsync: list[bool] = []

    monkeypatch.setattr(runner, "expected_nonterminal_paths", lambda *args: set())
    monkeypatch.setattr(
        runner,
        "write_manifest",
        lambda root, **kwargs: runner.science._write_json(
            root / "manifest.json", {"status": "precommit"}
        ),
    )
    monkeypatch.setattr(
        runner,
        "validate_retry_bundle_contents",
        lambda *args, **kwargs: (metadata, "a" * 64),
    )
    monkeypatch.setattr(
        runner, "expected_complete_payload", lambda *args, **kwargs: {"status": "complete"}
    )

    def fail_public(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt("injected BaseException")

    def record_fsync(path: Path) -> None:
        assert path == tmp_path
        complete_presence_at_fsync.append((path / "COMPLETE").exists())

    monkeypatch.setattr(runner, "validate_completed_retry_bundle", fail_public)
    monkeypatch.setattr(runner.science, "_fsync_directory", record_fsync)

    with pytest.raises(KeyboardInterrupt, match="injected BaseException"):
        runner.finalize_root(tmp_path, metadata=metadata, config=object())

    assert not (tmp_path / "COMPLETE").exists()
    assert True in complete_presence_at_fsync
    assert complete_presence_at_fsync[-1] is False


def test_successful_finalize_writes_complete_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata: dict[str, object] = {}
    events: list[str] = []
    original_write_json = runner.science._write_json

    monkeypatch.setattr(runner, "expected_nonterminal_paths", lambda *args: set())

    def tracked_write(path: Path, payload: object) -> None:
        events.append(f"write:{path.name}")
        original_write_json(path, payload)

    def write_manifest(root: Path, **_: object) -> None:
        runner.science._write_json(root / "manifest.json", {"status": "precommit"})

    def validate_precommit(*args: object, **kwargs: object) -> tuple[dict, str]:
        assert not (tmp_path / "COMPLETE").exists()
        events.append("precommit")
        return metadata, "a" * 64

    def validate_public(*args: object, **kwargs: object) -> dict[str, bool]:
        assert (tmp_path / "COMPLETE").is_file()
        events.append("public")
        return {"valid": True}

    monkeypatch.setattr(runner.science, "_write_json", tracked_write)
    monkeypatch.setattr(runner, "write_manifest", write_manifest)
    monkeypatch.setattr(runner, "validate_retry_bundle_contents", validate_precommit)
    monkeypatch.setattr(
        runner, "expected_complete_payload", lambda *args, **kwargs: {"status": "complete"}
    )
    monkeypatch.setattr(runner, "validate_completed_retry_bundle", validate_public)

    runner.finalize_root(tmp_path, metadata=metadata, config=object())

    assert events == ["write:manifest.json", "precommit", "write:COMPLETE", "public"]
    assert (tmp_path / "COMPLETE").is_file()


def _patch_completed_resume_prerequisites(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    config, _ = runner.load_retry_config()
    monkeypatch.setattr(runner, "validate_quarantine_bundle", lambda *args: {})
    monkeypatch.setattr(runner, "validate_repair_gate_amendment", lambda *args: object())
    monkeypatch.setattr(runner, "load_retry_config", lambda *args: (config, {}))
    monkeypatch.setattr(
        runner,
        "validate_allowed_source_delta",
        lambda *args: {"active_source_tree_sha256": "tree"},
    )
    monkeypatch.setattr(runner, "audit_retry_rng_ids", lambda *args: {})
    monkeypatch.setattr(runner.preflight, "_experiment_tree_sha256", lambda *args: "tree")
    monkeypatch.setattr(
        runner.preflight,
        "_build_source_snapshot",
        lambda *args: {"contract": {}},
    )
    monkeypatch.setattr(runner, "build_metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_config_output_root", lambda *args: root)
    monkeypatch.setattr(
        runner.science,
        "_run_seed_group",
        lambda *args, **kwargs: pytest.fail("formal replay was dispatched"),
    )


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_resume_with_stale_invalid_complete_removes_marker_before_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_type: type[BaseException],
) -> None:
    root = tmp_path / "stale_retry"
    root.mkdir()
    (root / "COMPLETE").write_text("invalid\n", encoding="utf-8")
    _patch_completed_resume_prerequisites(monkeypatch, root)
    fsync_states: list[bool] = []

    def reject_stale(*args: object, **kwargs: object) -> None:
        raise error_type("stale invalid COMPLETE rejected")

    def record_fsync(path: Path) -> None:
        assert path == root
        fsync_states.append((path / "COMPLETE").exists())

    monkeypatch.setattr(runner, "validate_completed_retry_bundle", reject_stale)
    monkeypatch.setattr(runner.science, "_fsync_directory", record_fsync)

    with pytest.raises(error_type, match="stale invalid COMPLETE rejected"):
        runner.run_exact_replay({}, resume=True)

    assert not (root / "COMPLETE").exists()
    assert fsync_states == [False]


def test_successful_completed_resume_only_validates_and_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "completed_retry"
    root.mkdir()
    complete = root / "COMPLETE"
    complete.write_bytes(b"already-complete\n")
    original_bytes = complete.read_bytes()
    _patch_completed_resume_prerequisites(monkeypatch, root)
    validations: list[Path] = []

    def accept_completed(
        selected_root: Path, *, expected_metadata: object
    ) -> dict[str, bool]:
        assert expected_metadata == {}
        validations.append(selected_root)
        return {"valid": True}

    monkeypatch.setattr(runner, "validate_completed_retry_bundle", accept_completed)
    monkeypatch.setattr(
        runner.science,
        "_write_json",
        lambda *args, **kwargs: pytest.fail("completed resume wrote an artifact"),
    )

    runner.run_exact_replay({}, resume=True)

    assert validations == [root]
    assert complete.read_bytes() == original_bytes


def test_fresh_root_and_resume_artifact_sets_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(runner, "validate_metadata", lambda *args, **kwargs: None)
    with pytest.raises(FileExistsError, match="fresh retry output already exists"):
        runner.prepare_root(
            existing,
            metadata={},
            schema={},
            source_snapshot={},
            resume=False,
        )

    metadata = {"source_snapshot": {}}
    schema: dict[str, object] = {}
    runner.science._write_json(existing / "metadata.json", metadata)
    runner.science._write_json(existing / "artifact_schema.json", schema)
    (existing / "rogue.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "verify_published_incident", lambda *args: None)
    monkeypatch.setattr(runner.preflight, "_verify_source_snapshot", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "allowed_partial_paths",
        lambda *args: {"metadata.json", "artifact_schema.json"},
    )
    with pytest.raises(RuntimeError, match="unexpected retry resume artifacts"):
        runner.prepare_root(
            existing,
            metadata=metadata,
            schema=schema,
            source_snapshot={},
            resume=True,
        )


def test_validate_only_never_dispatches_replay_rng(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runner, "load_incident_amendment", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "validation_payload",
        lambda *args: {"no_rng_consumed": True, "formal_retry_permitted": True},
    )

    def forbidden_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("formal RNG replay was dispatched")

    monkeypatch.setattr(runner, "run_exact_replay", forbidden_run)
    runner.main(["--validate-only"])

    assert json.loads(capsys.readouterr().out) == {
        "formal_retry_permitted": True,
        "no_rng_consumed": True,
    }


def test_retry_metadata_has_distinct_administrative_role_and_rng_disclosure(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    amendment = runner.load_incident_amendment()
    config, equivalence = runner.load_retry_config()
    quarantine = runner.validate_quarantine_bundle(amendment)
    gate = runner.validate_repair_gate_amendment(amendment)
    rng_audit = runner.audit_retry_rng_ids(config, amendment)
    source_delta = runner.validate_allowed_source_delta(amendment)
    monkeypatch.setattr(
        runner.science,
        "_runtime_environment",
        lambda devices: {"devices": list(devices)},
    )
    metadata = runner.build_metadata(
        amendment,
        config=config,
        equivalence=equivalence,
        quarantine=quarantine,
        gate=gate,
        rng_audit=rng_audit,
        source_delta=source_delta,
        source_hash="d" * 64,
        source_snapshot={},
        schema=runner.artifact_schema(),
    )

    assert metadata["role"] == config.role
    assert metadata["administrative_role"] == "administrative_exact_failed_publish_replay"
    assert metadata["retry_contract"]["rng_reused"] is True
    assert metadata["retry_contract"]["rng_fresh"] is False
    assert metadata["retry_contract"]["rng_independent"] is False
    runner.validate_metadata(metadata, schema=runner.artifact_schema())

    tampered = deepcopy(metadata)
    tampered["retry_contract"]["rng_independent"] = True
    with pytest.raises(RuntimeError, match="metadata config or amendment binding differs"):
        runner.validate_metadata(tampered, schema=runner.artifact_schema())

    missing_dependency = deepcopy(metadata)
    missing_dependency["dependency_files"].pop("simulator")
    with pytest.raises(RuntimeError, match="metadata config or amendment binding differs"):
        runner.validate_metadata(
            missing_dependency,
            schema=runner.artifact_schema(),
        )


def test_base_science_runner_fix_binds_simulator_and_retry_root_stays_absent() -> None:
    config, _ = runner.load_retry_config()
    dependencies = runner.dependency_files(config)

    assert runner.science._file_sha256(runner.BASE_SCIENCE_RUNNER) == (
        "a39da352f9ed7c390bb3cef66a124b2e7d866af2da712226af6c0a0441ce6a78"
    )
    assert dependencies["simulator"]["sha256"] == (
        "34460a98c24f45dfaf9b2f5e069094caafd4a12c4bec4482f8c804448bf860de"
    )
    assert not runner.RETRY_ROOT.exists()
