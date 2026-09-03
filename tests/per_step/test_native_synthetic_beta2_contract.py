from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import export_native_synthetic_beta2_contract as exporter


def test_frozen_beta2_source_is_complete_and_explicitly_not_gamma() -> None:
    exporter.validate_frozen_sources(
        exporter.DEFAULT_AGGREGATE_ROOT,
        exporter.DEFAULT_RAW_ROOT,
    )
    rows = exporter.build_source_rows(
        exporter.load_aggregate_rows(exporter.DEFAULT_AGGREGATE_ROOT)
    )

    assert len(rows) == len(exporter.METHODS) * exporter.HORIZON
    assert {row["method"] for row in rows} == set(exporter.METHODS)
    assert {row["stage_zero_based"] for row in rows} == set(
        range(exporter.HORIZON)
    )
    assert all(row["feedback_parameter"] == "beta" for row in rows)
    assert all(row["signed_gamma_comparable"] is False for row in rows)
    assert all(row["uses_clinical_donor_kernel"] is False for row in rows)


def test_contract_has_reviewer_safe_claim_boundary_and_frozen_metrics() -> None:
    rows = exporter.build_source_rows(
        exporter.load_aggregate_rows(exporter.DEFAULT_AGGREGATE_ROOT)
    )
    contract = exporter.build_contract(rows)

    assert contract["semantics"]["scenario"] == "tail_shift"
    assert contract["semantics"]["feedback_strength"] == 2.0
    assert contract["semantics"]["signed_gamma_comparable"] is False
    assert contract["semantics"]["uses_clinical_donor_kernel"] is False
    assert "beta is not the signed gamma scale" in contract["required_disambiguator"]
    assert contract["method_summary"]["Standard CP"]["wsc"] == pytest.approx(
        0.899429178238
    )
    assert contract["method_summary"]["SC-PCP"]["wsc"] == pytest.approx(
        0.902030578256
    )
    assert contract["contrast_audit"]["scpcp_minus_standard_wsc_pp"] == pytest.approx(
        0.2601400018
    )
    assert "not a large adverse-undercoverage rescue" in contract[
        "contrast_audit"
    ]["interpretation"]


def test_hash_validation_rejects_mutated_frozen_aggregate(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    for name in ("metadata.json", "per_stage_all_baselines.csv"):
        shutil.copy2(exporter.DEFAULT_AGGREGATE_ROOT / name, aggregate / name)
    with (aggregate / "per_stage_all_baselines.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(RuntimeError, match="hash differs"):
        exporter.validate_frozen_sources(aggregate, exporter.DEFAULT_RAW_ROOT)


def test_export_writes_only_source_contract_assets(tmp_path: Path) -> None:
    output = tmp_path / "native_synthetic"
    exporter.export_contract(
        exporter.DEFAULT_AGGREGATE_ROOT,
        exporter.DEFAULT_RAW_ROOT,
        output,
    )

    assert {path.name for path in output.iterdir()} == {
        "native_synthetic_beta2_stage_profile_source_data.csv",
        "source_contract.json",
        "manifest.json",
    }
    rows = list(
        csv.DictReader(
            (output / "native_synthetic_beta2_stage_profile_source_data.csv").open(
                encoding="utf-8",
                newline="",
            )
        )
    )
    assert len(rows) == 72
    assert "gamma" not in rows[0]
    contract = json.loads((output / "source_contract.json").read_text())
    assert contract["status"] == "frozen_deterministic_source_only"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["scientific_rng_used"] is False
