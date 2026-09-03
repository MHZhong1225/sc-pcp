"""Export the frozen native-Synthetic beta=2 stage-profile source contract.

This is deterministic post-processing only. It selects the prespecified RQ3
tail-shift cell from the frozen six-method paper suite and makes its semantics
explicit for reuse in a multi-setting figure. It never simulates trajectories,
resamples seeds, or maps beta onto the controlled signed-gamma scale.

Run from the repository root:

    conda run -n ucp python tools/export_native_synthetic_beta2_contract.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGGREGATE_ROOT = (
    ROOT / "results" / "work" / "complete_baseline_results_20260824"
)
DEFAULT_RAW_ROOT = (
    ROOT
    / "results"
    / "work"
    / "paper_marginal_final_20260822"
    / "rq3"
    / "beta_2"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "work" / "native_synthetic_beta2_contract_20260826"
)

CONTRACT_ID = "native_synthetic_tail_shift_beta2_frozen_rq3_v1"
DISPLAY_LABEL = "Native Synthetic tail-shift — strongest frozen feedback (beta=2)"
METHODS = ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")
INFORMATION_REGIME = {
    "Standard CP": "offline_logged_data",
    "ACI": "on_policy_adaptation",
    "MFCS": "offline_logged_data",
    "SPCI": "on_policy_adaptation",
    "PRC": "on_policy_adaptation",
    "SC-PCP": "offline_logged_data",
}
HORIZON = 12
SEEDS = tuple(range(1000, 1100))
TARGET = 0.90

EXPECTED_HASHES = {
    "aggregate_metadata": "90db917f4208bea5dc0cf956c015f1b1f622a42e873b3788b521d623fcdaf788",
    "aggregate_stage_rows": "aa6915dddfcfae5aa8e1ee7e4c8eca8b96fcb75825799a72f6bbe57244b93e91",
    "raw_config": "816a13850f991d930ba03c4ff3ff01606c04b0a6625eb3b855952c4b4e74a38e",
    "raw_study_metadata": "50ed02f9f0f4663662cce91e2045a6ba1f8b0ba1e6276dee8ad08fcbff574bda",
    "raw_study_status": "9dee9fa3eba036e927263f91b795211030ebd83f5cb210f86aa4d38e13c4bbdb",
}

SOURCE_COLUMNS = (
    "contract_id",
    "display_label",
    "dataset",
    "scenario",
    "feedback_parameter",
    "feedback_strength",
    "signed_gamma_comparable",
    "uses_clinical_donor_kernel",
    "method",
    "information_regime",
    "stage_zero_based",
    "n_runs",
    "n_selected",
    "coverage_mean",
    "coverage_ci95_lower",
    "coverage_ci95_upper",
    "coverage_deviation_from_090_pp",
    "coverage_deviation_ci95_lower_pp",
    "coverage_deviation_ci95_upper_pp",
    "normalized_width_mean",
    "normalized_width_ci95_lower",
    "normalized_width_ci95_upper",
    "interval_definition",
    "raw_source",
    "aggregate_source",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _validate_hash(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise RuntimeError(f"frozen source hash differs: {path.name}")


def validate_frozen_sources(aggregate_root: Path, raw_root: Path) -> Mapping[str, Any]:
    """Validate frozen provenance and the complete per-seed stage-vector grid."""

    aggregate_metadata_path = aggregate_root / "metadata.json"
    aggregate_stage_path = aggregate_root / "per_stage_all_baselines.csv"
    raw_config_path = raw_root / "config.yaml"
    raw_metadata_path = raw_root / "study_metadata.json"
    raw_status_path = raw_root / "study_status.json"

    for key, path in (
        ("aggregate_metadata", aggregate_metadata_path),
        ("aggregate_stage_rows", aggregate_stage_path),
        ("raw_config", raw_config_path),
        ("raw_study_metadata", raw_metadata_path),
        ("raw_study_status", raw_status_path),
    ):
        _validate_hash(path, EXPECTED_HASHES[key])

    aggregate_metadata = _read_json(aggregate_metadata_path)
    if aggregate_metadata.get("canonical_methods") != list(METHODS):
        raise RuntimeError("frozen canonical method order differs")
    if aggregate_metadata.get("coverage_target") != TARGET:
        raise RuntimeError("frozen coverage target differs")
    if aggregate_metadata.get("per_stage_interval") != (
        "pointwise two-sided 95% Student-t interval across selected seeds"
    ):
        raise RuntimeError("frozen stagewise interval definition differs")

    raw_config = yaml.safe_load(raw_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise RuntimeError("frozen beta=2 config is not a mapping")
    synthetic = raw_config.get("synthetic", {})
    if (
        raw_config.get("data", {}).get("dataset") != "synthetic"
        or synthetic.get("scenario") != "tail_shift"
        or synthetic.get("feedback_strength") != 2.0
        or raw_config.get("horizon") != HORIZON
    ):
        raise RuntimeError("frozen beta=2 native-Synthetic config differs")

    status = _read_json(raw_status_path)
    if (
        status.get("status") != "complete"
        or tuple(status.get("expected_seeds", ())) != SEEDS
        or tuple(status.get("completed_seeds", ())) != SEEDS
        or status.get("missing_seeds") != []
    ):
        raise RuntimeError("frozen beta=2 seed completion contract differs")

    for seed in SEEDS:
        seed_root = raw_root / f"seed_{seed:05d}"
        if not (seed_root / "COMPLETE").is_file():
            raise RuntimeError(f"seed {seed} is not complete")
        validate_seed_records(seed_root / "records.csv", seed=seed)

    return {
        "aggregate_metadata": aggregate_metadata,
        "raw_config": raw_config,
        "raw_study_metadata": _read_json(raw_metadata_path),
        "raw_study_status": status,
    }


def validate_seed_records(path: Path, *, seed: int) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if tuple(row.get("method") for row in rows) != METHODS:
        raise RuntimeError(f"seed {seed} canonical method rows differ")
    for row in rows:
        for field in ("per_time_coverage", "per_time_normalized_width"):
            values = ast.literal_eval(row[field])
            if not isinstance(values, list) or len(values) != HORIZON:
                raise RuntimeError(
                    f"seed {seed} {row['method']} {field} must have {HORIZON} stages"
                )


def load_aggregate_rows(aggregate_root: Path) -> list[dict[str, str]]:
    path = aggregate_root / "per_stage_all_baselines.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["section"] == "RQ3"
        and row["setting"] == "Synthetic beta=2"
        and row["dataset"] == "synthetic"
        and float(row["feedback_strength"]) == 2.0
    ]
    validate_aggregate_rows(selected)
    return selected


def validate_aggregate_rows(rows: Iterable[Mapping[str, str]]) -> None:
    indexed = {
        (row["method"], int(row["stage_zero_based"])): row for row in rows
    }
    expected = {(method, stage) for method in METHODS for stage in range(HORIZON)}
    if set(indexed) != expected or len(indexed) != len(expected):
        raise RuntimeError("frozen beta=2 aggregate method-stage grid differs")

    for method, stage in expected:
        row = indexed[(method, stage)]
        if row["n_runs"] != "100" or row["n_selected"] != "100":
            raise RuntimeError("frozen beta=2 selection count differs")
        coverage = float(row["coverage_mean"])
        coverage_low = float(row["coverage_ci_low"])
        coverage_high = float(row["coverage_ci_high"])
        width = float(row["normalized_width_mean"])
        width_low = float(row["normalized_width_ci_low"])
        width_high = float(row["normalized_width_ci_high"])
        if not 0.0 <= coverage_low <= coverage <= coverage_high <= 1.0:
            raise RuntimeError("frozen beta=2 coverage interval differs")
        if not 0.0 < width_low <= width <= width_high:
            raise RuntimeError("frozen beta=2 width interval differs")


def build_source_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    method_order = {method: index for index, method in enumerate(METHODS)}
    for row in sorted(
        rows,
        key=lambda item: (
            method_order[item["method"]],
            int(item["stage_zero_based"]),
        ),
    ):
        coverage = float(row["coverage_mean"])
        coverage_low = float(row["coverage_ci_low"])
        coverage_high = float(row["coverage_ci_high"])
        result.append(
            {
                "contract_id": CONTRACT_ID,
                "display_label": DISPLAY_LABEL,
                "dataset": "synthetic",
                "scenario": "tail_shift",
                "feedback_parameter": "beta",
                "feedback_strength": 2.0,
                "signed_gamma_comparable": False,
                "uses_clinical_donor_kernel": False,
                "method": row["method"],
                "information_regime": INFORMATION_REGIME[row["method"]],
                "stage_zero_based": int(row["stage_zero_based"]),
                "n_runs": int(row["n_runs"]),
                "n_selected": int(row["n_selected"]),
                "coverage_mean": coverage,
                "coverage_ci95_lower": coverage_low,
                "coverage_ci95_upper": coverage_high,
                "coverage_deviation_from_090_pp": 100.0 * (coverage - TARGET),
                "coverage_deviation_ci95_lower_pp": 100.0
                * (coverage_low - TARGET),
                "coverage_deviation_ci95_upper_pp": 100.0
                * (coverage_high - TARGET),
                "normalized_width_mean": float(row["normalized_width_mean"]),
                "normalized_width_ci95_lower": float(
                    row["normalized_width_ci_low"]
                ),
                "normalized_width_ci95_upper": float(
                    row["normalized_width_ci_high"]
                ),
                "interval_definition": (
                    "pointwise two-sided 95% Student-t interval across 100 selected seeds"
                ),
                "raw_source": "paper_marginal_final_20260822/rq3/beta_2",
                "aggregate_source": (
                    "complete_baseline_results_20260824/per_stage_all_baselines.csv"
                ),
            }
        )
    return result


def build_contract(source_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    method_summary: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        rows = [row for row in source_rows if row["method"] == method]
        coverages = [float(row["coverage_mean"]) for row in rows]
        widths = [float(row["normalized_width_mean"]) for row in rows]
        worst_stage = min(range(HORIZON), key=coverages.__getitem__)
        method_summary[method] = {
            "wsc": min(coverages),
            "worst_stage_zero_based": worst_stage,
            "mean_coverage": sum(coverages) / HORIZON,
            "mean_normalized_width": sum(widths) / HORIZON,
            "point_wsc_at_least_090": min(coverages) >= TARGET,
            "n_selected": 100,
        }

    standard = method_summary["Standard CP"]
    scpcp = method_summary["SC-PCP"]
    return {
        "contract_id": CONTRACT_ID,
        "status": "frozen_deterministic_source_only",
        "display_label": DISPLAY_LABEL,
        "required_disambiguator": (
            "independent synthetic DGP; no clinical donor kernel; beta is not "
            "the signed gamma scale"
        ),
        "figure_contract": {
            "core_conclusion": (
                "Under the strongest frozen native-Synthetic feedback setting, "
                "SC-PCP keeps stagewise mean coverage above 0.90 with a modest "
                "width increase over Standard CP; conservative methods can be "
                "wider, and the cell is not an adverse-rescue analogue."
            ),
            "archetype": "quantitative_grid",
            "panel_map": {
                "top": "stagewise coverage deviation from 0.90 in percentage points",
                "bottom": "stagewise normalized width",
            },
            "reviewer_risk": (
                "beta and controlled signed gamma are different intervention "
                "scales and must not be compared numerically"
            ),
        },
        "semantics": {
            "dataset": "synthetic",
            "scenario": "tail_shift",
            "feedback_parameter": "beta",
            "feedback_strength": 2.0,
            "strongest_frozen_feedback_cell": True,
            "signed_gamma_comparable": False,
            "uses_clinical_donor_kernel": False,
            "uses_native_synthetic_noise": True,
            "horizon": HORIZON,
            "tail_contamination_probability": 0.10,
            "tail_scale": 4.0,
        },
        "statistics": {
            "coverage_target": TARGET,
            "primary_coverage_metric": (
                "min_t mean_selected_seed(per_time_coverage_seed_t)"
            ),
            "stagewise_intervals": (
                "pointwise two-sided 95% Student-t intervals across selected seeds"
            ),
            "n_prespecified_seeds": len(SEEDS),
            "n_selected_seeds": len(SEEDS),
        },
        "methods": list(METHODS),
        "method_summary": method_summary,
        "contrast_audit": {
            "scpcp_minus_standard_wsc_pp": 100.0
            * (float(scpcp["wsc"]) - float(standard["wsc"])),
            "scpcp_to_standard_mean_width_ratio": float(
                scpcp["mean_normalized_width"]
            )
            / float(standard["mean_normalized_width"]),
            "interpretation": (
                "near-nominal native stress profile, not a large "
                "adverse-undercoverage rescue"
            ),
        },
        "claim_boundaries": [
            "Do not relabel beta=2 as gamma=-4 or as any signed-gamma cell.",
            "Do not describe this native synthetic environment as using clinical donors.",
            "Do not use this cell as evidence of a large adverse-undercoverage rescue.",
            "Do not claim universal SOTA or finite-sample coverage from this panel.",
        ],
        "source_provenance": {
            "raw_source": "results/work/paper_marginal_final_20260822/rq3/beta_2",
            "aggregate_source": (
                "results/work/complete_baseline_results_20260824/per_stage_all_baselines.csv"
            ),
            "sha256": dict(EXPECTED_HASHES),
            "source_rows": len(source_rows),
        },
    }


def write_bundle(
    output: Path,
    source_rows: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError(f"output must be new: {output}")
    output.mkdir(parents=True)

    source_path = output / "native_synthetic_beta2_stage_profile_source_data.csv"
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(source_rows)

    contract_path = output / "source_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "contract_id": CONTRACT_ID,
        "deterministic_postprocessing_only": True,
        "scientific_rng_used": False,
        "files": {
            source_path.name: _sha256(source_path),
            contract_path.name: _sha256(contract_path),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_contract(aggregate_root: Path, raw_root: Path, output: Path) -> None:
    validate_frozen_sources(aggregate_root, raw_root)
    source_rows = build_source_rows(load_aggregate_rows(aggregate_root))
    contract = build_contract(source_rows)
    write_bundle(output, source_rows, contract)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-root", type=Path, default=DEFAULT_AGGREGATE_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_contract(args.aggregate_root, args.raw_root, args.output)


if __name__ == "__main__":
    main()
