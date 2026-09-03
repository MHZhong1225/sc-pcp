from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from scpcp.copula_benchmark_config import (
    COPULA_SCIENTIFIC_SEEDS,
    CopulaBenchmarkConfig,
)


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_copula_mechanism.py"
    name = "test_run_copula_mechanism"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_config_uses_untouched_independent_seed_namespace() -> None:
    config = CopulaBenchmarkConfig.from_yaml(ROOT / "configs" / "copula_mechanism.yaml")

    assert config.seeds == COPULA_SCIENTIFIC_SEEDS
    assert config.seeds[0] == 94_000
    assert config.seeds[-1] == 94_198
    assert config.dgp.easy_correlation == 0.90
    assert config.dgp.hard_correlation == 0.0
    assert config.dgp.maximum_policy_logit_shift == 1.50
    assert config.gate.primary_radius == 1.90
    assert config.gate.minimum_relative_q90_shift == 0.03
    assert config.gate.minimum_coverage_shift == 0.015
    assert config.gate.minimum_prefix_ess_fraction == 0.15
    assert config.gate.maximum_incremental_ratio == 10.0
    assert config.gate.maximum_normalized_weight_share == 0.02


def test_global_seed_device_mapping_is_stable_for_resume_subsets() -> None:
    runner = _load_runner()
    seeds = (94_000, 94_002, 94_004, 94_006, 94_008)
    devices = ("cuda:0", "cuda:1")
    mapping = runner._seed_device_mapping(seeds, devices)
    pending = seeds[2:]
    grouped = runner._pending_seed_groups(seeds, devices, pending)

    assert mapping == {
        94_000: "cuda:0",
        94_002: "cuda:1",
        94_004: "cuda:0",
        94_006: "cuda:1",
        94_008: "cuda:0",
    }
    assert grouped == {
        "cuda:0": (94_004, 94_008),
        "cuda:1": (94_006,),
    }


def test_run_seed_records_stagewise_and_late_stage_inputs_without_method_rows() -> None:
    runner = _load_runner()
    base = CopulaBenchmarkConfig.from_yaml(ROOT / "configs" / "copula_mechanism.yaml")
    config = replace(
        base,
        horizon=3,
        late_stage_start=1,
        trajectories=2_000,
        betas=(-1.0, 0.0, 1.0),
        kappas=(0.0, 1.0),
        radii=(1.90,),
        seeds=(94_000,),
    )

    result = runner.run_seed(config, 7, device="cpu")
    records = pd.DataFrame(result.records)

    assert len(records) == 3 * 2 * 1 * 3
    assert {"q90_relative_gap", "coverage_gap", "prefix_ess_fraction"}.issubset(
        records.columns
    )
    assert "method" not in records.columns
    assert result.surfaces["q90_relative_gap"].shape == (3, 2, 1, 3)


def test_gate_requires_strong_signed_late_effects_and_paired_intervals() -> None:
    runner = _load_runner()
    base = CopulaBenchmarkConfig.from_yaml(ROOT / "configs" / "copula_mechanism.yaml")
    config = replace(base, seeds=(94_000, 94_002))
    records = _gate_records(config, relative_q90=0.04, coverage=0.02)
    diagnostics = _gate_diagnostics(config)

    gate = runner.evaluate_mechanism_gate(
        records,
        diagnostics,
        config,
        config_hash="config",
        source_hash="source",
        summary_hash="summary",
    )

    assert gate["status"] == "pass"
    assert gate["optional_six_method_stage"]["authorized"] is True
    assert all(
        interval["lower"] > 0.0
        for interval in gate["direction_aligned_seed_paired_late_stage_intervals"].values()
    )

    weak = _gate_records(config, relative_q90=0.02, coverage=0.01)
    failed = runner.evaluate_mechanism_gate(
        weak,
        diagnostics,
        config,
        config_hash="config",
        source_hash="source",
        summary_hash="summary",
    )
    assert failed["status"] == "fail"
    assert failed["optional_six_method_stage"]["authorized"] is False


def test_gate_uses_worst_overlap_and_marginal_cells() -> None:
    runner = _load_runner()
    base = CopulaBenchmarkConfig.from_yaml(ROOT / "configs" / "copula_mechanism.yaml")
    config = replace(base, seeds=(94_000, 94_002))
    records = _gate_records(config, relative_q90=0.04, coverage=0.02)
    diagnostics = _gate_diagnostics(config)

    bad_index = records[
        (records["beta"] == max(config.betas))
        & (records["kappa"] == max(config.kappas))
        & (records["radius"] == config.gate.primary_radius)
        & (records["stage"] >= config.late_stage_start)
    ].index[0]
    records.loc[bad_index, "prefix_ess_fraction"] = 0.149
    records.loc[bad_index, "maximum_incremental_ratio"] = 10.01
    records.loc[bad_index, "maximum_weight_share"] = 0.0201
    diagnostics.loc[diagnostics.index[0], "target_maximum_absolute_mean"] = 0.0201

    gate = runner.evaluate_mechanism_gate(
        records,
        diagnostics,
        config,
        config_hash="config",
        source_hash="source",
        summary_hash="summary",
    )

    assert gate["status"] == "fail"
    assert gate["checks"]["primary_late_minimum_prefix_ess"]["observed"] == pytest.approx(0.149)
    assert gate["checks"]["primary_late_maximum_incremental_ratio"]["observed"] == pytest.approx(10.01)
    assert gate["checks"]["primary_late_maximum_normalized_weight_share"]["observed"] == pytest.approx(0.0201)
    assert gate["checks"]["equal_marginal_mean"]["observed"] == pytest.approx(0.0201)


def test_atomic_seed_validation_rejects_payload_tampering(tmp_path: Path) -> None:
    runner = _load_runner()
    base = CopulaBenchmarkConfig.from_yaml(ROOT / "configs" / "copula_mechanism.yaml")
    config = replace(
        base,
        horizon=2,
        late_stage_start=1,
        betas=(-1.0, 0.0, 1.0),
        kappas=(0.0, 1.0),
        radii=(1.90,),
        seeds=(94_000,),
        output_dir=tmp_path,
    )
    result = _fake_seed_result(runner, config, seed=94_000)
    config_hash = runner._json_sha256(config.to_dict())
    source_hash = "source"

    path = runner.write_seed_artifact(
        result,
        tmp_path,
        config=config,
        config_hash=config_hash,
        source_hash=source_hash,
    )
    runner.validate_seed_artifact(
        path,
        config=config,
        seed=94_000,
        config_hash=config_hash,
        source_hash=source_hash,
    )

    with (path / "records.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(RuntimeError, match="payload hash differs"):
        runner.validate_seed_artifact(
            path,
            config=config,
            seed=94_000,
            config_hash=config_hash,
            source_hash=source_hash,
        )


def test_seed_metadata_binds_the_global_device_assignment(tmp_path: Path) -> None:
    runner = _load_runner()
    base = CopulaBenchmarkConfig.from_yaml(ROOT / "configs" / "copula_mechanism.yaml")
    config = replace(
        base,
        horizon=2,
        late_stage_start=1,
        betas=(-1.0, 0.0, 1.0),
        kappas=(0.0, 1.0),
        radii=(1.90,),
        seeds=(94_000,),
        output_dir=tmp_path,
    )
    result = _fake_seed_result(runner, config, seed=94_000)
    config_hash = runner._json_sha256(config.to_dict())
    path = runner.write_seed_artifact(
        result,
        tmp_path,
        config=config,
        config_hash=config_hash,
        source_hash="source",
    )
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["device"] = "cuda:1"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    complete = json.loads((path / "COMPLETE").read_text())
    complete["metadata_sha256"] = _sha256(metadata_path)
    (path / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="metadata field device"):
        runner.validate_seed_artifact(
            path,
            config=config,
            seed=94_000,
            config_hash=config_hash,
            source_hash="source",
        )


def test_formal_rng_audit_enumerates_digests_and_rejects_prior_use(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    artifact_root = tmp_path / "artifacts"
    source_root = tmp_path / "source"
    (source_root / "scripts").mkdir(parents=True)
    artifact_root.mkdir()
    (source_root / "scripts" / "old.py").write_text(
        "OLD_SEEDS = tuple(range(91000, 91020, 10))\n"
    )
    seeds = (94_000, 94_002)

    audit = runner._audit_formal_rng_ids(
        seeds,
        output_dir=tmp_path / "new_output",
        artifact_root=artifact_root,
        source_root=source_root,
    )

    assert audit["formal_rng_id_count"] == 2
    assert audit["formal_rng_mapping"] == {
        "base_94000/paired_copula_crn": 94_000,
        "base_94002/paired_copula_crn": 94_002,
    }
    assert audit["collision_count"] == 0
    assert len(audit["formal_rng_mapping_sha256"]) == 64
    assert len(audit["prior_declared_or_artifact_rng_id_sha256"]) == 64
    assert len(audit["audit_sha256"]) == 64

    (artifact_root / "seed_94002").mkdir()
    with pytest.raises(RuntimeError, match="collide"):
        runner._audit_formal_rng_ids(
            seeds,
            output_dir=tmp_path / "new_output",
            artifact_root=artifact_root,
            source_root=source_root,
        )
    (artifact_root / "seed_94002").rmdir()
    (source_root / "scripts" / "old.py").write_text("OLD_RNG_IDS = (94000,)\n")
    with pytest.raises(RuntimeError, match="collide"):
        runner._audit_formal_rng_ids(
            seeds,
            output_dir=tmp_path / "new_output",
            artifact_root=artifact_root,
            source_root=source_root,
        )


def test_six_method_gate_is_bound_to_completed_artifacts(tmp_path: Path) -> None:
    runner = _load_runner()
    manifest = {"config_sha256": "config"}
    summary = {"result": "frozen"}
    gate = {
        "status": "pass",
        "config_sha256": "config",
        "summary_sha256": "",
        "optional_six_method_stage": {
            "authorized": True,
            "canonical_methods": list(runner.CANONICAL_METHODS),
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    gate["summary_sha256"] = _sha256(tmp_path / "summary.json")
    (tmp_path / "gate.json").write_text(json.dumps(gate))
    complete = {
        "status": "complete",
        "manifest_sha256": _sha256(tmp_path / "manifest.json"),
        "summary_sha256": _sha256(tmp_path / "summary.json"),
        "gate_sha256": _sha256(tmp_path / "gate.json"),
    }
    (tmp_path / "COMPLETE").write_text(json.dumps(complete))

    assert runner.require_six_method_gate(tmp_path)["status"] == "pass"
    (tmp_path / "summary.json").write_text(json.dumps({"result": "tampered"}))
    with pytest.raises(RuntimeError, match="summary_sha256"):
        runner.require_six_method_gate(tmp_path)


def _gate_records(
    config: CopulaBenchmarkConfig,
    *,
    relative_q90: float,
    coverage: float,
) -> pd.DataFrame:
    rows = []
    for seed in config.seeds:
        for beta in config.betas:
            for kappa in config.kappas:
                for radius in config.radii:
                    for stage in range(config.horizon):
                        policy_tv = 0.0 if kappa == 0.0 else 0.10
                        if kappa == 0.0 or beta == 0.0:
                            hard_gap = q90_gap = coverage_gap = 0.0
                        else:
                            direction = 1.0 if beta > 0.0 else -1.0
                            hard_gap = direction * 0.05
                            q90_gap = direction * relative_q90
                            coverage_gap = -direction * coverage
                        rows.append(
                            {
                                "seed": seed,
                                "beta": beta,
                                "kappa": kappa,
                                "radius": radius,
                                "stage": stage,
                                "kernel_fingerprint": str(beta),
                                "policy_tv": policy_tv,
                                "source_action_rate": 0.4,
                                "target_action_rate": 0.4 + policy_tv,
                                "action_rate_gap": policy_tv,
                                "source_hard_prevalence": 0.4,
                                "target_hard_prevalence": 0.4 + hard_gap,
                                "hard_prevalence_gap": hard_gap,
                                "source_q90": 1.9,
                                "target_q90": 1.9 * (1.0 + q90_gap),
                                "q90_gap": 1.9 * q90_gap,
                                "q90_relative_gap": q90_gap,
                                "source_same_radius_coverage": 0.9,
                                "target_same_radius_coverage": 0.9 + coverage_gap,
                                "coverage_gap": coverage_gap,
                                "prefix_ess_fraction": 0.5,
                                "maximum_weight_share": 0.01,
                                "log_weight_span": 2.0,
                                "minimum_incremental_ratio": 0.5,
                                "maximum_incremental_ratio": 2.0,
                            }
                        )
    return pd.DataFrame(rows)


def _gate_diagnostics(config: CopulaBenchmarkConfig) -> pd.DataFrame:
    rows = []
    for seed in config.seeds:
        for beta in config.betas:
            for kappa in config.kappas:
                for radius in config.radii:
                    rows.append(
                        {
                            "seed": seed,
                            "beta": beta,
                            "kappa": kappa,
                            "radius": radius,
                            "source_maximum_absolute_mean": 0.005,
                            "target_maximum_absolute_mean": 0.005,
                            "source_maximum_variance_error": 0.01,
                            "target_maximum_variance_error": 0.01,
                            "source_maximum_correlation_error": 0.01,
                            "target_maximum_correlation_error": 0.01,
                        }
                    )
    return pd.DataFrame(rows)


def _fake_seed_result(runner, config: CopulaBenchmarkConfig, *, seed: int):
    records = _gate_records(config, relative_q90=0.04, coverage=0.02)
    records = records[records["seed"] == seed]
    diagnostics = _gate_diagnostics(config)
    diagnostics = diagnostics[diagnostics["seed"] == seed]
    shape = (len(config.betas), len(config.kappas), len(config.radii), config.horizon)
    surfaces = {
        name: np.zeros(shape)
        for name in (
            "policy_tv",
            "action_rate_gap",
            "hard_prevalence_gap",
            "q90_gap",
            "q90_relative_gap",
            "coverage_gap",
            "prefix_ess_fraction",
            "maximum_weight_share",
            "log_weight_span",
        )
    }
    surfaces.update(
        {
            "betas": np.asarray(config.betas),
            "kappas": np.asarray(config.kappas),
            "radii": np.asarray(config.radii),
        }
    )
    return runner.CopulaSeedResult(
        seed=seed,
        device="cuda:0",
        records=tuple(records.to_dict(orient="records")),
        setting_diagnostics=tuple(diagnostics.to_dict(orient="records")),
        surfaces=surfaces,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
