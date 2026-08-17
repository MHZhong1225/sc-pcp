from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import summarize_tabular_validation as report


SEEDS = (0, 1)
SOURCE_HASH = "c" * 64


def _write_complete(path: Path, payload: dict | None = None) -> None:
    content = "complete\n" if payload is None else json.dumps(payload) + "\n"
    path.write_text(content, encoding="utf-8")


def _scpcp_record(*, selected: bool, worst_coverage: float = 0.92) -> dict:
    profile = [0.8, 1.25]
    if selected:
        selected_scale = 2.1
        return {
            "track": report.TRACK_A,
            "evaluation_scope": report.FRESH_SCOPE,
            "method": report.AUDIT_METHOD,
            "information_regime": report.AUDIT_REGIME,
            "selection_estimand": "per_step",
            "selection_parameter": report.SELECTION_PARAMETER,
            "selection_status": report.CERTIFIED_STATUS,
            "selection_available": True,
            "certificate_type": report.AUDIT_CERTIFICATE,
            "certificate_formal": True,
            "certified": True,
            "selected_scale": selected_scale,
            "stage_profile": json.dumps(profile),
            "q_by_time": json.dumps([selected_scale * value for value in profile]),
            "estimated_min_coverage": 0.95,
            "lower_bound_min": 0.91,
            "worst_coverage": worst_coverage,
            "average_coverage": 0.94,
            "worst_gap": max(0.0, 0.9 - worst_coverage),
            "oracle_evaluation_trajectories": 50_000,
        }
    return {
        "track": report.TRACK_A,
        "evaluation_scope": report.UNAVAILABLE_SCOPE,
        "method": report.AUDIT_METHOD,
        "information_regime": report.AUDIT_REGIME,
        "selection_estimand": "per_step",
        "selection_parameter": report.SELECTION_PARAMETER,
        "selection_status": report.UNCERTIFIED_STATUS,
        "selection_available": False,
        "certificate_type": report.AUDIT_CERTIFICATE,
        "certificate_formal": True,
        "certified": False,
        "selected_scale": float("nan"),
        "stage_profile": json.dumps(profile),
        "q_by_time": "",
        "estimated_min_coverage": float("nan"),
        "lower_bound_min": float("nan"),
        "worst_coverage": float("nan"),
        "average_coverage": float("nan"),
        "worst_gap": float("nan"),
        "oracle_evaluation_trajectories": 0,
    }


def _write_run(root: Path) -> None:
    root.mkdir()
    config = {
        "data": {"dataset": "tabular"},
        "certification": {"alpha": 0.1},
        "samples": {"oracle_rollouts": 50_000},
        "cot": {"loss": "huber"},
        "horizon": 2,
        "seeds": list(SEEDS),
        "devices": ["cuda:0", "cuda:1"],
        "output_dir": str(root),
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (root / "study_metadata.json").write_text(
        json.dumps({"source_tree_sha256": SOURCE_HASH, "seeds": list(SEEDS)}),
        encoding="utf-8",
    )
    (root / "study_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "expected_seeds": list(SEEDS),
                "completed_seeds": list(SEEDS),
                "missing_seeds": [],
            }
        ),
        encoding="utf-8",
    )
    _write_complete(root / "COMPLETE")

    for seed, selected in zip(SEEDS, (True, False)):
        seed_root = root / f"seed_{seed:05d}"
        seed_root.mkdir()
        pd.DataFrame([_scpcp_record(selected=selected)]).to_csv(
            seed_root / "records.csv", index=False
        )
        (seed_root / "metadata.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "source_tree_sha256": SOURCE_HASH,
                    "config": config,
                }
            ),
            encoding="utf-8",
        )
        _write_complete(
            seed_root / "COMPLETE", {"seed": seed, "status": "complete"}
        )


def test_tabular_report_separates_formal_selection_from_empirical_attainment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tabular"
    _write_run(root)

    audit = report.audit_validation_run(root, expected_seeds=SEEDS)
    summary = report.summarize_validation(audit).iloc[0]

    assert summary["formal_bound_available_rate"] == 1.0
    assert summary["formal_certificate_rate"] == 0.5
    assert summary["formal_selection_rate"] == 0.5
    assert summary["abstention_rate"] == 0.5
    assert summary["fresh_target_met_rate_all_runs_abstention_as_failure"] == 0.5
    assert summary["fresh_target_met_rate_evaluated"] == 1.0
    assert summary["mean_worst_coverage_evaluated"] == pytest.approx(0.92)
    assert summary["mean_selected_scale"] == pytest.approx(2.1)
    assert summary["mean_selected_estimated_min_coverage"] == pytest.approx(0.95)
    assert summary["mean_selected_lower_bound_min"] == pytest.approx(0.91)
    assert summary["total_fresh_evaluation_trajectories"] == 50_000

    output = tmp_path / "report"
    report.write_report(output, audit, report.summarize_validation(audit))
    payload = json.loads((output / "theorem_validation_summary.json").read_text())
    markdown = (output / "theorem_validation_report.md").read_text()
    assert (output / "COMPLETE").is_file()
    assert payload["interpretation"]["clinical_extrapolation"] == "not permitted"
    assert "single SC-PCP record" in markdown
    assert "abstention = failure" in markdown
    assert payload["method_schema"] == "profiled_scale_ordered_iut"
    assert payload["selection_parameter"] == "global_scale"


@pytest.mark.parametrize(
    "corruption", ["missing_complete", "source_hash", "duplicate_scpcp"]
)
def test_report_refuses_incomplete_or_inconsistent_artifacts(
    tmp_path: Path,
    corruption: str,
) -> None:
    root = tmp_path / "tabular"
    _write_run(root)
    if corruption == "missing_complete":
        (root / "seed_00001" / "COMPLETE").unlink()
        match = "missing"
    elif corruption == "source_hash":
        metadata_path = root / "seed_00001" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["source_tree_sha256"] = "d" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        match = "source hash mismatch"
    else:
        records_path = root / "seed_00000" / "records.csv"
        records = pd.read_csv(records_path)
        duplicate = records.iloc[0].copy()
        pd.concat([records, duplicate.to_frame().T], ignore_index=True).to_csv(
            records_path, index=False
        )
        match = "exactly one profiled ordered-IUT"
    with pytest.raises(report.AuditError, match=match):
        report.audit_validation_run(root, expected_seeds=SEEDS)
