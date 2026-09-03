from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT / "scripts/run_native_synthetic_signed_gamma_time_coordinate_repair_r1.py"
)
AMENDMENT_PATH = (
    ROOT / "configs/native_synthetic_signed_gamma_time_coordinate_repair_r1.yaml"
)
PARENT_ROOT = ROOT / "results/work/native_synthetic_signed_gamma_v1"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "test_run_native_synthetic_signed_gamma_time_coordinate_repair_r1",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parent_probe(runner, seed: int = 121000) -> dict:
    return runner.base._read_json(PARENT_ROOT / f"mechanism/seed_{seed}.json")[
        "probe"
    ]


def _repaired_probe(runner, seed: int = 121000) -> dict:
    probe = deepcopy(_parent_probe(runner, seed))
    for row in probe["gamma_rows"]:
        row["finite_and_structural"] = True
    return probe


def _empty_audit_roots(tmp_path: Path) -> tuple[Path, Path]:
    artifact_root = tmp_path / "artifacts"
    source_root = tmp_path / "source"
    artifact_root.mkdir()
    source_root.mkdir()
    return artifact_root, source_root


def test_amendment_pins_exact_parent_inventory_and_repair_scope(runner) -> None:
    amendment = runner.load_amendment(AMENDMENT_PATH)
    parent = amendment["parent"]
    assert parent["root"] == "results/work/native_synthetic_signed_gamma_v1"
    assert parent["artifact_count"] == 25
    assert parent["artifact_inventory_sha256"] == runner.base._canonical_sha256(
        parent["artifact_inventory"]
    )
    assert parent["formal_rng_ids"] == list(range(121000, 121200, 10))
    assert amendment["repair"]["scientific_changes"] == []
    assert amendment["repair"]["scope"] == "gamma_rows[*].finite_and_structural"


def test_parent_bundle_is_fully_validated_from_its_pinned_snapshot(runner) -> None:
    contract = runner.validate_parent_bundle(runner.load_amendment(AMENDMENT_PATH))
    assert contract["status"] == "fully_validated"
    assert contract["decision"] == "NO_GO"
    assert contract["artifact_count"] == 25
    assert contract["formal_rng_ids"] == list(range(121000, 121200, 10))


def test_parent_inventory_or_hash_mutation_fails_closed(runner) -> None:
    amendment = runner.load_amendment(AMENDMENT_PATH)
    mutated = deepcopy(amendment["parent"])
    mutated["artifact_inventory"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="inventory hash differs"):
        runner._validate_parent_inventory_contract(mutated)


def test_replay_collision_classifier_exempts_only_exact_parent(runner) -> None:
    mapping = {f"mechanism/base_{seed}/exogenous_noise": seed for seed in (11, 12)}
    passed = runner.classify_replay_collisions(
        mapping,
        parent_ids={11, 12},
        other_artifact_ids=set(),
        source_ids=set(),
        external_ids=set(),
    )
    assert passed["status"] == "passed_with_exact_parent_replay_exemption"
    assert passed["raw_collision_count"] == 2
    assert passed["authorized_parent_collision_count"] == 2
    assert passed["unauthorized_collision_count"] == 0

    duplicated_elsewhere = runner.classify_replay_collisions(
        mapping,
        parent_ids={11, 12},
        other_artifact_ids={12},
        source_ids=set(),
        external_ids=set(),
    )
    assert duplicated_elsewhere["status"] == "unauthorized_replay_collision"
    assert duplicated_elsewhere["authorized_parent_collision_count"] == 1
    assert duplicated_elsewhere["unauthorized_collision_count"] == 1
    assert duplicated_elsewhere["unauthorized_collisions"][
        "mechanism/base_12/exogenous_noise"
    ]["categories"] == ["exact_parent_bundle", "other_artifact"]


def test_live_replay_audit_reports_raw20_authorized20_unauthorized0(
    runner, tmp_path: Path
) -> None:
    amendment = runner.load_amendment(AMENDMENT_PATH)
    config, _ = runner.build_effective_config(amendment)
    artifact_root, source_root = _empty_audit_roots(tmp_path)
    audit = runner.audit_replay_rng_ids(
        config,
        amendment,
        artifact_root=artifact_root,
        source_root=source_root,
    )
    assert audit["raw_collision_count"] == 20
    assert audit["authorized_parent_collision_count"] == 20
    assert audit["unauthorized_collision_count"] == 0
    assert audit["missing_parent_collision_count"] == 0


def test_ordinary_preflight_still_rejects_parent_seed_reuse(
    runner, tmp_path: Path
) -> None:
    config = runner.base.NativeSignedGammaBenchmarkConfig.from_yaml(
        runner.BASE_CONFIG
    ).with_overrides(output_root=tmp_path / "ordinary_fresh_root")
    with pytest.raises(RuntimeError, match="collide with prior use"):
        runner.base.audit_formal_rng_ids(
            config,
            output_root=config.output_root,
            config_path=runner.BASE_CONFIG,
        )


def test_probe_comparison_allows_only_the_repaired_boolean(runner) -> None:
    amendment = runner.load_amendment(AMENDMENT_PATH)
    parent = _parent_probe(runner)
    replay = _repaired_probe(runner)
    comparison = runner.compare_probe_payloads(parent, replay, amendment)
    assert comparison["status"] == "EXACT_REPLAY"
    assert comparison["exact_after_removing_only_repaired_field"] is True
    assert comparison["repaired_field_values_valid"] is True

    changed_metric = deepcopy(replay)
    changed_metric["gamma_rows"][0]["mid_policy_tv"] += 1e-15
    invalid = runner.compare_probe_payloads(parent, changed_metric, amendment)
    assert invalid["status"] == "INVALID_REPLAY"
    assert invalid["exact_after_removing_only_repaired_field"] is False


def test_wrong_repaired_boolean_is_invalid_replay(runner) -> None:
    amendment = runner.load_amendment(AMENDMENT_PATH)
    replay = _repaired_probe(runner)
    replay["gamma_rows"][2]["finite_and_structural"] = False
    comparison = runner.compare_probe_payloads(
        _parent_probe(runner), replay, amendment
    )
    assert comparison["exact_after_removing_only_repaired_field"] is True
    assert comparison["repaired_field_values_valid"] is False
    assert comparison["status"] == "INVALID_REPLAY"


def test_reserved_downstream_mapping_is_exact_unique_and_unconsumed(
    runner, tmp_path: Path
) -> None:
    amendment = runner.load_amendment(AMENDMENT_PATH)
    mapping = runner.build_downstream_rng_reservation(amendment)
    assert len(mapping) == len(set(mapping.values())) == 241
    assert mapping["summary/bootstrap_complete_seed_matrix"] == 12140019
    assert runner.base._canonical_sha256(mapping) == (
        "c17c3098ee47b5f530300505198983393b95a51b7ffbdb4c3d399c7b43a5546a"
    )
    config, _ = runner.build_effective_config(amendment)
    artifact_root, source_root = _empty_audit_roots(tmp_path)
    replay_audit = runner.audit_replay_rng_ids(
        config,
        amendment,
        artifact_root=artifact_root,
        source_root=source_root,
    )
    reservation = runner.audit_downstream_rng_reservation(
        amendment,
        replay_audit,
        artifact_root=artifact_root,
        source_root=source_root,
    )
    assert reservation["status"] == "reserved_not_consumed_and_collision_free"
    assert reservation["collision_count"] == 0
    assert reservation["execution_authorized_by_repair_runner"] is False
    runner.validate_downstream_rng_reservation(amendment, reservation)
    tampered = deepcopy(reservation)
    tampered["reserved_rng_mapping"]["science/base_121400/task"] += 1
    with pytest.raises(RuntimeError, match="reservation contract differs"):
        runner.validate_downstream_rng_reservation(amendment, tampered)


def test_only_designated_downstream_source_may_declare_reserved_ids(
    runner, tmp_path: Path
) -> None:
    scripts = tmp_path / "scripts"
    tools = tmp_path / "tools"
    source = tmp_path / "src/scpcp"
    scripts.mkdir(parents=True)
    tools.mkdir()
    source.mkdir(parents=True)
    designated = scripts / "run_native_synthetic_signed_gamma_science.py"
    designated.write_text(
        "import torch\n"
        "RESERVED_BASE_SEEDS = (121400,)\n"
        "def future(seed=RESERVED_BASE_SEEDS[0]):\n"
        "    torch.manual_seed(seed)\n",
        encoding="utf-8",
    )
    excluded = runner.downstream_source_declaration_exclusions(tmp_path)
    declared_only = runner.base._source_rng_scan(
        tmp_path, excluded_paths=excluded
    )
    assert 121400 not in declared_only["actual"]

    (tools / "rogue.py").write_text(
        "import torch\ntorch.manual_seed(121400)\n", encoding="utf-8"
    )
    with_rogue = runner.base._source_rng_scan(tmp_path, excluded_paths=excluded)
    assert 121400 in with_rogue["actual"]


def test_downstream_runner_is_not_exempt_from_repair_seed_reuse(
    runner, tmp_path: Path
) -> None:
    scripts = tmp_path / "scripts"
    (tmp_path / "src/scpcp").mkdir(parents=True)
    (tmp_path / "tools").mkdir()
    scripts.mkdir()
    (scripts / "run_native_synthetic_signed_gamma_science.py").write_text(
        "import torch\ntorch.manual_seed(121000)\n", encoding="utf-8"
    )
    scan = runner.base._source_rng_scan(
        tmp_path, excluded_paths=runner.replay_source_exclusions(tmp_path)
    )
    assert 121000 in scan["actual"]


def test_effective_scientific_config_changes_only_output_root(runner) -> None:
    amendment = runner.load_amendment(AMENDMENT_PATH)
    config, equivalence = runner.build_effective_config(amendment)
    assert equivalence["status"] == "exact_except_output_root"
    assert equivalence["allowed_difference"] == "output_root"
    assert config.output_root == runner.EXPECTED_OUTPUT_ROOT
    assert config.devices == ("cuda:0", "cuda:1")


def test_validate_only_path_never_calls_mechanism_probe(
    runner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runner,
        "mechanism_probe",
        lambda *_args, **_kwargs: pytest.fail("validate-only consumed a formal seed"),
    )
    amendment = runner.load_amendment(AMENDMENT_PATH)
    artifact_root, source_root = _empty_audit_roots(tmp_path)
    live_replay_audit = runner.audit_replay_rng_ids
    live_reservation_audit = runner.audit_downstream_rng_reservation
    monkeypatch.setattr(
        runner,
        "audit_replay_rng_ids",
        lambda config, selected_amendment: live_replay_audit(
            config,
            selected_amendment,
            artifact_root=artifact_root,
            source_root=source_root,
        ),
    )
    monkeypatch.setattr(
        runner,
        "audit_downstream_rng_reservation",
        lambda selected_amendment, replay_audit: live_reservation_audit(
            selected_amendment,
            replay_audit,
            artifact_root=artifact_root,
            source_root=source_root,
        ),
    )
    payload = runner.validation_payload(amendment, AMENDMENT_PATH)
    assert payload["formal_replay_permitted"] is True
    assert payload["replay_rng_audit"]["raw_collision_count"] == 20
    assert payload["downstream_rng_reservation"]["reserved_rng_id_count"] == 241


def test_fresh_root_and_resume_metadata_are_strict(
    runner, tmp_path: Path
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        runner.prepare_root(
            existing,
            metadata={},
            schema={},
            source_snapshot={},
            amendment_path=AMENDMENT_PATH,
            resume=False,
        )

    runner.base._write_json(existing / "metadata.json", {"identity": "stored"})
    runner.base._write_json(existing / "artifact_schema.json", {})
    with pytest.raises(RuntimeError, match="metadata differs"):
        runner.prepare_root(
            existing,
            metadata={"identity": "live"},
            schema={},
            source_snapshot={},
            amendment_path=AMENDMENT_PATH,
            resume=True,
        )


def test_schema_and_summary_bind_amendment_and_invalid_replay_blocks_downstream(
    runner,
) -> None:
    amendment_sha = runner.base._file_sha256(AMENDMENT_PATH)
    schema = runner.artifact_schema(amendment_sha)
    assert schema["amendment_sha256"] == amendment_sha
    assert schema["invalid_replay_consequence"] == "downstream_unauthorized"

    amendment = runner.load_amendment(AMENDMENT_PATH)
    config, _ = runner.build_effective_config(amendment)
    mapping = runner.replay_rng_mapping(config)
    invalid_probe = _repaired_probe(runner)
    invalid_probe["gamma_rows"][0]["mid_policy_tv"] += 1e-15
    results = {}
    for label, seed in mapping.items():
        parent_probe = _parent_probe(runner, seed)
        replay_probe = _repaired_probe(runner, seed)
        if seed == 121000:
            replay_probe = invalid_probe
        results[label] = {
            "probe": replay_probe,
            "probe_comparison": runner.compare_probe_payloads(
                parent_probe, replay_probe, amendment
            ),
        }
    metadata = {
        "amendment_sha256": amendment_sha,
        "parent_bundle": {
            "manifest_sha256": amendment["parent"]["bundle_files"]["manifest.json"]
        },
        "scientific_config_sha256": runner.base._canonical_sha256(config.to_dict()),
        "replay_rng_audit": {"audit_sha256": "a" * 64},
        "source_tree_sha256": "b" * 64,
    }
    summary = runner.summarize_results(config, results, metadata)
    assert summary["status"] == "INVALID_REPLAY"
    assert summary["downstream_authorized"] is False


def test_complete_repair_bundle_and_public_validator_bind_every_contract(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    amendment = runner.load_amendment(AMENDMENT_PATH)
    parent = runner.validate_parent_bundle(amendment)
    original_config, original_equivalence = runner.build_effective_config(amendment)
    root = (tmp_path / "repair_bundle").resolve()
    config = original_config.with_overrides(output_root=root)
    equivalence = dict(original_equivalence)
    equivalence["replay_output_root"] = str(root)
    monkeypatch.setattr(
        runner,
        "build_effective_config",
        lambda _amendment: (config, equivalence),
    )
    artifact_root, source_root = _empty_audit_roots(tmp_path)
    live_replay_audit = runner.audit_replay_rng_ids
    live_reservation_audit = runner.audit_downstream_rng_reservation
    monkeypatch.setattr(
        runner,
        "audit_replay_rng_ids",
        lambda selected_config, selected_amendment: live_replay_audit(
            selected_config,
            selected_amendment,
            artifact_root=artifact_root,
            source_root=source_root,
        ),
    )
    monkeypatch.setattr(
        runner,
        "audit_downstream_rng_reservation",
        lambda selected_amendment, replay_audit: live_reservation_audit(
            selected_amendment,
            replay_audit,
            artifact_root=artifact_root,
            source_root=source_root,
        ),
    )
    rng_audit = runner.audit_replay_rng_ids(config, amendment)
    reservation = runner.audit_downstream_rng_reservation(amendment, rng_audit)
    source_hash = runner.base._experiment_tree_sha256(ROOT)
    source_snapshot = runner.base._build_source_snapshot(ROOT)
    amendment_sha = runner.base._file_sha256(AMENDMENT_PATH)
    schema = runner.artifact_schema(amendment_sha)
    metadata = runner.build_metadata(
        amendment,
        amendment_path=AMENDMENT_PATH,
        parent=parent,
        config=config,
        equivalence=equivalence,
        rng_audit=rng_audit,
        downstream_reservation=reservation,
        source_hash=source_hash,
        source_snapshot=source_snapshot["contract"],
        schema=schema,
        invocation_argv=("--fixture",),
    )
    runner.prepare_root(
        root,
        metadata=metadata,
        schema=schema,
        source_snapshot=source_snapshot,
        amendment_path=AMENDMENT_PATH,
        resume=False,
    )
    results = {}
    devices = runner.base.seed_device_mapping(config)
    for label, seed in runner.replay_rng_mapping(config).items():
        parent_path = PARENT_ROOT / f"mechanism/seed_{seed}.json"
        probe = _repaired_probe(runner, seed)
        artifact = {
            "protocol": runner.REPAIR_PROTOCOL,
            "scientific_protocol": config.protocol,
            "rng_label": label,
            "rng_id": seed,
            "device": devices[label],
            "amendment_sha256": amendment_sha,
            "parent_seed_artifact_sha256": runner.base._file_sha256(parent_path),
            "scientific_config_sha256": metadata["scientific_config_sha256"],
            "replay_rng_audit_sha256": rng_audit["audit_sha256"],
            "source_tree_sha256": source_hash,
            "probe_comparison": runner.compare_probe_payloads(
                _parent_probe(runner, seed), probe, amendment
            ),
            "probe": probe,
        }
        runner.validate_seed_artifact(
            artifact,
            expected_label=label,
            expected_rng_id=seed,
            expected_device=devices[label],
            metadata=metadata,
            amendment=amendment,
        )
        runner.base._write_json(root / f"mechanism/seed_{seed}.json", artifact)
        results[label] = artifact
    summary = runner.summarize_results(config, results, metadata)
    assert summary["status"] == "GO"
    assert summary["n_exact_replays"] == summary["n_repaired_fields_valid"] == 20
    runner.base._write_json(root / "summary.json", summary)
    runner.finalize_root(
        root,
        metadata=metadata,
        summary=summary,
        amendment=amendment,
        amendment_path=AMENDMENT_PATH,
    )
    contract = runner.validate_completed_repair_bundle(
        root, source_root=ROOT, amendment_path=AMENDMENT_PATH
    )
    assert contract["decision"] == "GO"
    assert contract["downstream_authorized"] is True
    assert contract["amendment_sha256"] == amendment_sha
    assert contract["reserved_rng_mapping_sha256"] == (
        "c17c3098ee47b5f530300505198983393b95a51b7ffbdb4c3d399c7b43a5546a"
    )
    assert len(contract["reserved_rng_mapping"]) == 241

    seed_path = root / "mechanism/seed_121000.json"
    tampered = runner.base._read_json(seed_path)
    tampered["probe"]["gamma_rows"][0]["mid_policy_tv"] += 1e-15
    runner.base._write_json(seed_path, tampered)
    with pytest.raises(RuntimeError, match="probe comparison differs"):
        runner.validate_completed_repair_bundle(
            root, source_root=ROOT, amendment_path=AMENDMENT_PATH
        )


def test_runner_and_amendment_contain_no_formal_execution_side_effects_on_import(
    runner,
) -> None:
    parsed = runner.parse_args(["--validate-only"])
    assert parsed.validate_only is True
    assert json.loads(json.dumps(runner.load_amendment(AMENDMENT_PATH)))[
        "repair_protocol"
    ] == runner.REPAIR_PROTOCOL
