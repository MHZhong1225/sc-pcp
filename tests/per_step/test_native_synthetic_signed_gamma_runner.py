from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_native_synthetic_signed_gamma_preflight.py"
CONFIG_PATH = ROOT / "configs/native_synthetic_signed_gamma.yaml"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "test_run_native_synthetic_signed_gamma_preflight",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def _empty_scan_roots(tmp_path: Path) -> tuple[Path, Path]:
    artifacts = tmp_path / "artifacts"
    source = tmp_path / "source"
    artifacts.mkdir(parents=True)
    (source / "scripts").mkdir(parents=True)
    return artifacts, source


def _passing_probe(config, rng_id: int) -> dict[str, object]:
    rows = []
    for gamma in config.gammas:
        signed_shift = -0.005 * gamma
        row: dict[str, object] = {
            "gamma": gamma,
            "kernel_fingerprint": f"kernel-{gamma:+g}",
            "source_target_kernel_shared": True,
            "mid_policy_tv": 0.10,
            "high_policy_tv": 0.20,
            "mid_expected_action_coordinate_shift": -0.10,
            "high_expected_action_coordinate_shift": -0.20,
            "late_difficulty_prevalence_shift": signed_shift,
            "late_tail_prevalence_shift": signed_shift,
            "finite_and_structural": True,
        }
        if gamma == 0.0:
            row["exact_placebo"] = {
                "states": True,
                "outcomes": True,
                "tails": True,
            }
        if gamma == config.primary_gamma:
            overlap = {
                "minimum_ess_fraction": 0.50,
                "maximum_incremental_ratio": 2.0,
                "maximum_normalized_weight_share": 0.01,
            }
            row["overlap"] = {"mid": dict(overlap), "high": dict(overlap)}
        rows.append(row)
    return {
        "protocol": config.protocol,
        "seed": rng_id,
        "preflight_only": True,
        "primary_gamma": config.primary_gamma,
        "gamma_rows": rows,
    }


def _build_completed_bundle(
    runner,
    tmp_path: Path,
) -> tuple[object, Path, dict[str, object]]:
    output = tmp_path / "bundle"
    config = runner.NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG_PATH)
    config = config.with_overrides(output_root=output)
    artifacts, source = _empty_scan_roots(tmp_path)
    audit = runner.audit_formal_rng_ids(
        config,
        output_root=output,
        artifact_root=artifacts,
        source_root=source,
        config_path=CONFIG_PATH,
    )
    source_hash = runner._experiment_tree_sha256(ROOT)
    snapshot = runner._build_source_snapshot(ROOT)
    schema = runner._artifact_schema()
    metadata = runner._build_metadata(
        config,
        config_path=CONFIG_PATH,
        rng_audit=audit,
        source_hash=source_hash,
        source_snapshot=snapshot["contract"],
        schema=schema,
        invocation_argv=(),
    )
    runner._prepare_root(
        output,
        metadata=metadata,
        schema=schema,
        source_snapshot=snapshot,
        resume=False,
    )
    results = {}
    devices_by_label = runner.seed_device_mapping(config)
    for label, rng_id in runner.formal_rng_mapping(config).items():
        artifact = {
            "protocol": config.protocol,
            "rng_label": label,
            "rng_id": rng_id,
            "config_payload_sha256": metadata["config_payload_sha256"],
            "rng_audit_sha256": audit["audit_sha256"],
            "device": devices_by_label[label],
            "source_tree_sha256": source_hash,
            "probe": _passing_probe(config, rng_id),
        }
        results[label] = artifact
        runner._write_json(output / "mechanism" / f"seed_{rng_id}.json", artifact)
    summary = runner._summarize_results(config, results, audit["audit_sha256"])
    runner._write_json(output / "summary.json", summary)
    runner._finalize_root(output, metadata=metadata, summary=summary)
    return config, output, metadata


def test_validate_only_runs_live_audit_without_running_formal_ids(
    runner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG_PATH)
    config = config.with_overrides(output_root=tmp_path / "never-created")
    artifacts, source = _empty_scan_roots(tmp_path)
    monkeypatch.setattr(
        runner,
        "mechanism_probe",
        lambda *_args, **_kwargs: pytest.fail("validate-only consumed a formal RNG ID"),
    )
    payload = runner.validation_payload(
        config,
        CONFIG_PATH,
        artifact_root=artifacts,
        source_root=source,
    )

    assert payload["contract_valid"]
    assert payload["gate_only"]
    assert payload["formal_launch_permitted"]
    assert payload["formal_launch_blocker"] is None
    assert payload["rng_audit"]["collision_count"] == 0
    clinical_v3 = payload["rng_audit"]["coordinated_external_reservations"][
        "controlled_clinical_fidelity_v3"
    ]
    assert clinical_v3["count"] == clinical_v3["mapping_count"] == 1_304
    assert len(clinical_v3["mapping"]) == 1_304
    assert not config.output_root.exists()


def test_live_rng_mapping_hashes_and_config_binding_are_exact(
    runner,
    tmp_path: Path,
) -> None:
    config = runner.NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG_PATH)
    config = config.with_overrides(output_root=tmp_path / "output")
    artifacts, source = _empty_scan_roots(tmp_path)
    audit = runner.audit_formal_rng_ids(
        config,
        output_root=config.output_root,
        artifact_root=artifacts,
        source_root=source,
        config_path=CONFIG_PATH,
        external_reservations={},
    )
    mapping = runner.formal_rng_mapping(config)

    assert tuple(mapping.values()) == config.base_seeds
    assert len(mapping) == len(set(mapping.values())) == 20
    assert audit["formal_rng_mapping"] == mapping
    assert audit["internal_rng_ids_unique"] is True
    assert audit["base_seed_bank_sha256"] == runner._canonical_sha256(
        list(config.base_seeds)
    )
    assert audit["formal_rng_mapping_sha256"] == runner._canonical_sha256(mapping)
    assert audit["formal_rng_id_sha256"] == runner._integer_set_sha256(mapping.values())
    assert audit["config_payload_sha256"] == runner._canonical_sha256(config.to_dict())
    assert audit["prior_rng_id_sha256"] == runner._integer_set_sha256(set())
    assert audit["collision_sha256"] == runner._canonical_sha256({})
    unhashed = dict(audit)
    stored = unhashed.pop("audit_sha256")
    assert stored == runner._canonical_sha256(unhashed)
    runner.validate_rng_audit(config, audit, external_reservations={})


def test_clinical_v3_external_reservation_matches_its_actual_full_mapping(
    runner,
) -> None:
    from scpcp.controlled_clinical_fidelity_v3 import (
        DATASETS,
        load_fidelity_v3_config,
    )
    spec = importlib.util.spec_from_file_location(
        "test_run_controlled_clinical_extension_for_rng_reservation",
        ROOT / "scripts/run_controlled_clinical_extension.py",
    )
    assert spec is not None and spec.loader is not None
    v2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v2
    spec.loader.exec_module(v2)

    fidelity = load_fidelity_v3_config(
        ROOT / "configs/controlled_clinical_fidelity_v3.yaml"
    )
    protocol = v2.load_extension_config(
        ROOT / "configs/controlled_clinical_extension.yaml"
    )
    fresh = {
        dataset: replace(
            protocol.datasets[dataset],
            seeds=fidelity.confirmation_seeds[dataset],
            bootstrap_seed=fidelity.confirmation_bootstrap_seeds[dataset],
        )
        for dataset in DATASETS
    }
    expected = v2._new_rng_stream_mapping(
        replace(protocol, datasets=fresh),
        DATASETS,
    )
    assert runner.CONTROLLED_CLINICAL_V3_RESERVATION_MAPPING == expected
    assert len(expected) == len(set(expected.values())) == 1_304


def test_source_scan_counts_actual_use_but_excludes_declaration(
    runner,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    declaration = source / "scripts/declaration.py"
    declaration.write_text("DECLARED_SEED = 121000\n", encoding="utf-8")
    assert runner._source_actual_rng_ids(source, excluded_paths=set()) == set()

    declaration.write_text(
        "import torch\nDECLARED_SEED = 121000\ntorch.manual_seed(DECLARED_SEED)\n",
        encoding="utf-8",
    )
    assert runner._source_actual_rng_ids(source, excluded_paths=set()) == {121000}


def test_source_scan_resolves_local_default_container_arithmetic_and_nested_shadowing(
    runner,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/prior.py").write_text(
        "import numpy as np\n"
        "import torch\n"
        "OUTER_SEED = 121000\n"
        "SEED_MAP = {'formal': 121010}\n"
        "def local():\n"
        "    seed = 121020\n"
        "    torch.manual_seed(seed)\n"
        "def default(seed=SEED_MAP['formal'] + 1):\n"
        "    np.random.default_rng(seed)\n"
        "def outer():\n"
        "    seed = 17\n"
        "    def inner(seed=seed + 1):\n"
        "        torch.Generator().manual_seed(seed)\n",
        encoding="utf-8",
    )

    report = runner._source_rng_scan(source, excluded_paths=set())

    assert report["actual"] == {18, 121011, 121020}
    assert {17, 121000, 121010, 121020}.issubset(report["declared"])


def test_source_scan_fails_closed_on_unresolved_seed_like_expression(
    runner,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    prior = source / "scripts/prior.py"
    prior.write_text(
        "import torch\n"
        "def used():\n"
        "    seed = opaque(121000)\n"
        "    torch.manual_seed(seed)\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"prior\.py:4"):
        runner._source_rng_scan(source, excluded_paths=set())


def test_source_reservations_are_reported_separately_from_actual_use(
    runner,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/prior.py").write_text(
        "RESERVED_SEEDS = [121000, 121010]\n",
        encoding="utf-8",
    )

    report = runner._source_rng_scan(source, excluded_paths=set())

    assert report["actual"] == set()
    assert report["reserved"] == {121000, 121010}


@pytest.mark.parametrize(
    "payload",
    [
        {"seed": 121000},
        {"arbitrary": {"nested": [{"random_state": 121000}]}},
    ],
)
def test_all_structured_artifact_names_reject_nested_actual_seed_collision(
    runner,
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    artifacts, source = _empty_scan_roots(tmp_path)
    (artifacts / "ordinary-result.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    config = runner.NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG_PATH)
    config = config.with_overrides(output_root=output)

    with pytest.raises(RuntimeError, match="collide"):
        runner.audit_formal_rng_ids(
            config,
            output_root=output,
            artifact_root=artifacts,
            source_root=source,
            config_path=CONFIG_PATH,
            external_reservations={},
        )
    assert not output.exists()


def test_artifact_scan_is_fail_closed_but_ignores_benign_numeric_fields(
    runner,
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output = tmp_path / "output"
    benign = artifacts / "metrics.json"
    benign.write_text(json.dumps({"epoch": 121000, "loss": 0.1}), encoding="utf-8")
    assert runner._artifact_actual_rng_ids(artifacts, excluded_root=output) == set()

    benign.write_text('{"seed":', encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot parse structured artifact"):
        runner._artifact_actual_rng_ids(artifacts, excluded_root=output)

    benign.write_text(json.dumps({"seed": "121000"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-integer RNG value"):
        runner._artifact_actual_rng_ids(artifacts, excluded_root=output)


def test_artifact_scan_rejects_path_escape_and_separates_reservations(
    runner,
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output = tmp_path / "output"
    (artifacts / "reservation.json").write_text(
        json.dumps({"reserved_seeds": [121000]}),
        encoding="utf-8",
    )
    report = runner._artifact_rng_scan(artifacts, excluded_root=output)
    assert report["actual"] == set()
    assert report["reserved"] == {121000}

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"seed": 121000}), encoding="utf-8")
    (artifacts / "escape.json").symlink_to(outside)
    with pytest.raises(RuntimeError, match="escapes its root"):
        runner._artifact_rng_scan(artifacts, excluded_root=output)


def test_tabular_rng_columns_are_scanned_and_noninteger_values_fail_closed(
    runner,
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output = tmp_path / "output"
    table = artifacts / "ordinary-result.csv"
    table.write_text("metric,seed,evaluation_seed\n0.5,121000,NA\n", encoding="utf-8")
    assert runner._artifact_actual_rng_ids(artifacts, excluded_root=output) == {121000}

    table.write_text("metric,seed\n121000,not-an-integer\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-integer RNG value"):
        runner._artifact_actual_rng_ids(artifacts, excluded_root=output)


def test_rng_binary_index_binding_is_hashed_without_treating_indices_as_seeds(
    runner,
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output = tmp_path / "output"
    indices = artifacts / "bootstrap_indices.npy"
    indices.write_bytes(b"not-loaded-as-seed-ids")
    metadata = artifacts / "summary.json"
    metadata.write_text(
        json.dumps(
            {
                "bootstrap": {
                    "seed_index_matrix_path": indices.name,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="binary binding lacks a SHA256"):
        runner._artifact_rng_scan(artifacts, excluded_root=output)

    metadata.write_text(
        json.dumps(
            {
                "bootstrap": {
                    "seed_index_matrix_path": indices.name,
                    "seed_index_matrix_sha256": runner._file_sha256(indices),
                }
            }
        ),
        encoding="utf-8",
    )
    report = runner._artifact_rng_scan(artifacts, excluded_root=output)
    assert report["actual"] == set()
    assert len(report["binary_bindings"]) == 1

    indices.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="binary binding differs"):
        runner._artifact_rng_scan(artifacts, excluded_root=output)


def test_rng_audit_binary_binding_records_are_root_relative_and_revalidated(
    runner,
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    binary = artifacts / "prior" / "bootstrap_indices.npy"
    metadata = artifacts / "completed" / "metadata.json"
    binary.parent.mkdir(parents=True)
    metadata.parent.mkdir(parents=True)
    binary.write_bytes(b"previously-verified-bootstrap-indices")
    binding = {
        "binary_path": binary.relative_to(artifacts).as_posix(),
        "field_path": "bootstrap.seed_index_matrix_path",
        "sha256": runner._file_sha256(binary),
    }
    metadata.write_text(
        json.dumps(
            {"rng_audit": {"artifact_binary_rng_bindings": [binding]}}
        ),
        encoding="utf-8",
    )

    report = runner._artifact_rng_scan(
        artifacts,
        excluded_root=tmp_path / "output",
    )

    assert report["actual"] == set()
    assert report["binary_bindings"] == [
        {
            "metadata_path": "completed/metadata.json",
            "field_path": (
                "rng_audit.artifact_binary_rng_bindings.0.binary_path"
            ),
            "binary_path": "prior/bootstrap_indices.npy",
            "sha256": binding["sha256"],
        }
    ]

    binary.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="binary binding differs"):
        runner._artifact_rng_scan(artifacts, excluded_root=tmp_path / "output")


def test_rng_audit_binary_binding_record_rejects_escape_and_seed_collision(
    runner,
    tmp_path: Path,
) -> None:
    artifacts, source = _empty_scan_roots(tmp_path)
    outside = tmp_path / "outside.npy"
    outside.write_bytes(b"outside")
    metadata = artifacts / "metadata.json"
    binding = {
        "binary_path": "../outside.npy",
        "field_path": "bootstrap.seed_index_matrix_path",
        "sha256": runner._file_sha256(outside),
    }
    metadata.write_text(
        json.dumps(
            {"rng_audit": {"artifact_binary_rng_bindings": [binding]}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="binary binding escapes artifact root"):
        runner._artifact_rng_scan(artifacts, excluded_root=tmp_path / "output")

    inside = artifacts / "bootstrap_indices.npy"
    inside.write_bytes(b"inside")
    binding["binary_path"] = inside.relative_to(artifacts).as_posix()
    binding["sha256"] = runner._file_sha256(inside)
    metadata.write_text(
        json.dumps(
            {
                "rng_audit": {"artifact_binary_rng_bindings": [binding]},
                "seed_list": [121000],
            }
        ),
        encoding="utf-8",
    )
    config = runner.NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG_PATH)
    config = config.with_overrides(output_root=tmp_path / "output")
    with pytest.raises(RuntimeError, match="collide"):
        runner.audit_formal_rng_ids(
            config,
            output_root=config.output_root,
            artifact_root=artifacts,
            source_root=source,
            config_path=CONFIG_PATH,
            external_reservations={},
        )


@pytest.mark.parametrize("collision_source", ["artifact", "source", "reservation"])
def test_live_audit_rejects_every_prior_collision_before_output_creation(
    runner,
    tmp_path: Path,
    collision_source: str,
) -> None:
    output = tmp_path / "output"
    config = runner.NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG_PATH)
    config = config.with_overrides(output_root=output)
    artifacts, source = _empty_scan_roots(tmp_path)
    reservations: dict[str, set[int]] = {}
    if collision_source == "artifact":
        prior = artifacts / "prior"
        prior.mkdir()
        (prior / "metadata.json").write_text(
            json.dumps({"actual_rng_ids": [121000]}),
            encoding="utf-8",
        )
    elif collision_source == "source":
        (source / "scripts/prior.py").write_text(
            "import torch\ntorch.manual_seed(121000)\n",
            encoding="utf-8",
        )
    else:
        reservations = {"coordinated_other_study": {121000}}

    with pytest.raises(RuntimeError, match="collide"):
        runner.audit_formal_rng_ids(
            config,
            output_root=output,
            artifact_root=artifacts,
            source_root=source,
            config_path=CONFIG_PATH,
            external_reservations=reservations,
        )
    assert not output.exists()


def test_current_output_is_excluded_from_prior_artifact_scan(
    runner,
    tmp_path: Path,
) -> None:
    artifacts, source = _empty_scan_roots(tmp_path)
    output = artifacts / "current"
    output.mkdir()
    (output / "metadata.json").write_text(
        json.dumps({"actual_rng_ids": [121000]}),
        encoding="utf-8",
    )
    config = runner.NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG_PATH)
    config = config.with_overrides(output_root=output)
    audit = runner.audit_formal_rng_ids(
        config,
        output_root=output,
        artifact_root=artifacts,
        source_root=source,
        config_path=CONFIG_PATH,
        external_reservations={},
    )
    assert audit["collision_count"] == 0


def test_zero_digest_config_bypass_and_rehashed_audit_bypass_fail(
    runner,
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["seed_collision_audit_status"] = "PASS"
    raw["seed_collision_audit_sha256"] = "0" * 64
    malicious = tmp_path / "malicious.yaml"
    malicious.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown native signed-gamma config fields"):
        runner.NativeSignedGammaBenchmarkConfig.from_yaml(malicious)

    config = runner.NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG_PATH)
    config = config.with_overrides(output_root=tmp_path / "output")
    artifacts, source = _empty_scan_roots(tmp_path / "audit")
    audit = runner.audit_formal_rng_ids(
        config,
        output_root=config.output_root,
        artifact_root=artifacts,
        source_root=source,
        config_path=CONFIG_PATH,
        external_reservations={},
    )
    tampered = deepcopy(audit)
    tampered["config_payload_sha256"] = "0" * 64
    tampered["config_rng_binding_sha256"] = "0" * 64
    unhashed = dict(tampered)
    unhashed.pop("audit_sha256")
    tampered["audit_sha256"] = runner._canonical_sha256(unhashed)
    with pytest.raises(RuntimeError, match="effective config payload"):
        runner.validate_rng_audit(config, tampered)


@pytest.mark.parametrize("artifact_name", ["metadata", "summary", "schema"])
@pytest.mark.parametrize(
    "forbidden",
    ["coverage", "normalized_width", "q90", "score", "selection_rate", "science"],
)
def test_metadata_summary_and_schema_firewall_fail_closed(
    runner,
    artifact_name: str,
    forbidden: str,
) -> None:
    if artifact_name == "schema":
        payload = runner._artifact_schema()
        payload["seed_probe_fields"].append(forbidden)
    else:
        payload = {"identity": {forbidden: 0.0}}
    with pytest.raises(RuntimeError, match="forbidden result field"):
        runner._assert_field_firewall(payload)


def test_snapshot_is_deterministic_and_includes_simulator(runner) -> None:
    first = runner._build_source_snapshot(ROOT)
    second = runner._build_source_snapshot(ROOT)
    assert first["contract"] == second["contract"]
    assert first["archive_bytes"] == second["archive_bytes"]
    manifest = json.loads(first["manifest_bytes"])
    paths = {row["path"] for row in manifest["files"]}
    assert "src/scpcp/simulator.py" in paths
    assert "src/scpcp/native_signed_gamma.py" in paths
    assert "scripts/run_native_synthetic_signed_gamma_preflight.py" in paths
    assert "configs/native_synthetic_signed_gamma.yaml" in paths


def test_atomic_hash_chain_strict_resume_and_complete_tamper(
    runner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, output, metadata = _build_completed_bundle(runner, tmp_path)
    runner.validate_completed_bundle(output, expected_metadata=metadata)
    complete = json.loads((output / "COMPLETE").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_prespecified"] == 20
    assert summary["required_passed_rng_ids"] == 19
    assert complete["manifest_sha256"] == runner._file_sha256(output / "manifest.json")
    assert complete["rng_audit_sha256"] == metadata["rng_audit"]["audit_sha256"]
    assert complete["source_snapshot_sha256"] == metadata["source_snapshot"]["archive_sha256"]
    assert complete["environment_sha256"] == metadata["environment_sha256"]
    assert complete["invocation_sha256"] == metadata["invocation_sha256"]
    assert complete["launch_contract_sha256"] == metadata["launch_contract_sha256"]

    class ForbiddenExecutor:
        def __init__(self, *_args, **_kwargs):
            pytest.fail("a valid completed resume attempted to run a formal RNG ID")

    monkeypatch.setattr(
        runner,
        "audit_formal_rng_ids",
        lambda *_args, **_kwargs: metadata["rng_audit"],
    )
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ForbiddenExecutor)
    runner.run_preflight(config, config_path=CONFIG_PATH, resume=True)

    tampered_summary = dict(summary)
    tampered_summary["n_passed"] = 19
    runner._write_json(output / "summary.json", tampered_summary)
    with pytest.raises(RuntimeError, match="does not reconcile"):
        runner.validate_completed_bundle(output, expected_metadata=metadata)
    runner._write_json(output / "summary.json", summary)

    complete["manifest_sha256"] = "0" * 64
    runner._write_json(output / "COMPLETE", complete)
    with pytest.raises(RuntimeError, match="COMPLETE hash chain"):
        runner.run_preflight(config, config_path=CONFIG_PATH, resume=True)


def test_artifact_tamper_and_active_source_drift_fail_closed(
    runner,
    tmp_path: Path,
) -> None:
    _, output, metadata = _build_completed_bundle(runner, tmp_path)
    seed_path = next((output / "mechanism").glob("seed_*.json"))
    seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
    seed_payload["rng_id"] += 1
    runner._write_json(seed_path, seed_payload)
    with pytest.raises(RuntimeError, match="identity differs"):
        runner.validate_completed_bundle(output, expected_metadata=metadata)

    _, clean_output, clean_metadata = _build_completed_bundle(
        runner,
        tmp_path / "clean",
    )
    copied_source = tmp_path / "copied-source"
    for path in runner._experiment_paths(ROOT):
        destination = copied_source / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    runner.validate_completed_bundle(
        clean_output,
        expected_metadata=clean_metadata,
        source_root=copied_source,
    )
    simulator = copied_source / "src/scpcp/simulator.py"
    simulator.write_text(
        simulator.read_text(encoding="utf-8") + "\n# source drift\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="source tree differs"):
        runner.validate_completed_bundle(
            clean_output,
            expected_metadata=clean_metadata,
            source_root=copied_source,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"seed_list": [121000]},
        {"random_states": [121000]},
        {"per_seed": {"121000": {"metric": 0.5}}},
        {"results": {"seed_121000": {"metric": 0.5}}},
    ],
)
def test_live_artifact_scan_catches_common_seed_containers(
    runner,
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    report = runner._artifact_rng_scan(
        artifacts,
        excluded_root=tmp_path / "output",
    )
    assert 121000 in report["actual"]


def test_source_scan_catches_manual_seed_all_and_branch_possibilities(
    runner,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    prior = source / "scripts/prior.py"
    prior.write_text(
        "import torch\n"
        "torch.cuda.manual_seed_all(121000)\n"
        "seed = 121000\n"
        "if flag:\n"
        "    seed = 42\n"
        "torch.manual_seed(seed)\n",
        encoding="utf-8",
    )
    report = runner._source_rng_scan(source, excluded_paths=set())
    assert {42, 121000}.issubset(report["actual"])


@pytest.mark.parametrize(
    "branch",
    [
        "seed = external\nif flag:\n    seed = 121000\n",
        "if flag:\n    seed = 121000\nelse:\n    seed = external\n",
    ],
)
def test_source_scan_keeps_concrete_seed_from_mixed_dynamic_branch(
    runner,
    tmp_path: Path,
    branch: str,
) -> None:
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/prior.py").write_text(
        "import torch\n"
        "def used(external, flag):\n"
        + "".join(f"    {line}\n" for line in branch.splitlines())
        + "    torch.manual_seed(seed)\n",
        encoding="utf-8",
    )
    report = runner._source_rng_scan(source, excluded_paths=set())
    assert 121000 in report["actual"]


def test_source_scan_rejects_unresolved_concrete_seed_in_any_branch(
    runner,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/prior.py").write_text(
        "import torch\n"
        "def used(flag):\n"
        "    if flag:\n"
        "        seed = 42\n"
        "    else:\n"
        "        seed = opaque(121000)\n"
        "    torch.manual_seed(seed)\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="cannot resolve seed-like RNG expression"):
        runner._source_rng_scan(source, excluded_paths=set())


def test_probe_schema_rejects_gamma_or_arbitrary_metric_tampering(runner) -> None:
    config = runner.NativeSignedGammaBenchmarkConfig.from_yaml(CONFIG_PATH)
    probe = _passing_probe(config, 121000)
    runner._validate_probe_payload(probe, expected_seed=121000)

    wrong_primary = deepcopy(probe)
    wrong_primary["primary_gamma"] = 999.0
    with pytest.raises(RuntimeError, match="identity or signed-gamma grid"):
        runner._validate_probe_payload(wrong_primary, expected_seed=121000)

    wrong_order = deepcopy(probe)
    wrong_order["gamma_rows"] = list(reversed(wrong_order["gamma_rows"]))
    with pytest.raises(RuntimeError, match="unique and ordered"):
        runner._validate_probe_payload(wrong_order, expected_seed=121000)

    arbitrary_metric = deepcopy(probe)
    arbitrary_metric["gamma_rows"][0]["metric"] = 0.91
    with pytest.raises(RuntimeError, match="gamma-row schema"):
        runner._validate_probe_payload(arbitrary_metric, expected_seed=121000)
