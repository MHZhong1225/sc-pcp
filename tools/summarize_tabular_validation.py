"""Audit and summarize the exact finite-MDP theorem-validation run.

The input must be one complete ``run_per_step.py`` tabular run using the sole
paper method: profiled-scale, ordered-IUT SC-PCP. Exact finite-MDP transport
evidence belongs to that single SC-PCP record.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from select_baseline_hyperparameters import (
    AuditError,
    _exact_seed_directories,
    parse_seed_spec,
    read_json,
    read_yaml,
    require_complete_marker,
    require_status,
)


AUDIT_METHOD = "SC-PCP"
AUDIT_REGIME = "offline_logged_data"
AUDIT_CERTIFICATE = "tabular_ordered_iut_pointwise_exact_l1_oracle_bound"
LEGACY_AUDIT_METHOD = "SC-PCP (exact-MDP oracle-bound audit)"
SELECTION_PARAMETER = "global_scale"
CERTIFIED_STATUS = "CERTIFIED_ORDERED_IUT"
UNCERTIFIED_STATUS = "UNCERTIFIED_ORDERED_IUT"
TRACK_A = "empirical_environment"
FRESH_SCOPE = "fresh_target_policy_rollouts_in_frozen_empirical_environment"
UNAVAILABLE_SCOPE = "unavailable_target_policy_evaluation"
SOURCE_HASH = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ValidationAudit:
    root: Path
    source_hash: str
    config: dict[str, Any]
    target_coverage: float
    configured_fresh_rollouts: int
    seed_rows: pd.DataFrame


def _as_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    raise AuditError(f"{label} is not a Boolean: {value!r}")


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AuditError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise AuditError(f"{label} is not finite: {value!r}")
    return result


def _optional_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AuditError(f"{label} is not numeric or NA: {value!r}") from error
    if math.isinf(result):
        raise AuditError(f"{label} is infinite: {value!r}")
    return result


def _integer(value: Any, *, label: str) -> int:
    number = _finite_float(value, label=label)
    rounded = int(round(number))
    if not math.isclose(number, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise AuditError(f"{label} is not an integer: {value!r}")
    return rounded


def _positive_vector(value: Any, *, label: str, length: int) -> np.ndarray:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        vector = np.asarray(parsed, dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise AuditError(f"{label} is not a numeric JSON vector") from error
    if vector.shape != (length,):
        raise AuditError(f"{label} must have length {length}")
    if not np.isfinite(vector).all() or (vector <= 0.0).any():
        raise AuditError(f"{label} must be finite and strictly positive")
    return vector


def _nested(mapping: Mapping[str, Any], *keys: str, label: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise AuditError(f"missing {label}")
        value = value[key]
    return value


def _validate_optional_manifest(root: Path, source_hash: str, seeds: tuple[int, ...]) -> None:
    """Validate a launcher manifest when a single run has been wrapped by one."""

    path = root / "study_manifest.json"
    if not path.exists():
        return
    manifest = read_json(path)
    if manifest.get("source_tree_sha256") != source_hash:
        raise AuditError(f"source hash mismatch in optional manifest {path}")
    if "seeds" in manifest and manifest["seeds"] != list(seeds):
        raise AuditError(f"seed set mismatch in optional manifest {path}")


def _extract_audit_record(frame: pd.DataFrame, records_path: Path) -> pd.Series:
    if "method" not in frame:
        raise AuditError(f"{records_path} is missing required column: method")
    if frame["method"].eq(LEGACY_AUDIT_METHOD).any():
        raise AuditError(
            f"{records_path} uses the retired two-row scalar/max-t SC-PCP schema"
        )

    required = {
        "track",
        "evaluation_scope",
        "method",
        "information_regime",
        "selection_estimand",
        "selection_status",
        "selection_parameter",
        "selection_available",
        "certificate_type",
        "certificate_formal",
        "certified",
        "selected_scale",
        "stage_profile",
        "q_by_time",
        "estimated_min_coverage",
        "lower_bound_min",
        "worst_coverage",
        "average_coverage",
        "worst_gap",
        "oracle_evaluation_trajectories",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AuditError(f"{records_path} is missing required columns: {', '.join(missing)}")

    audit_rows = frame[frame["method"].eq(AUDIT_METHOD)]
    if len(audit_rows) != 1:
        raise AuditError(
            f"{records_path} must contain exactly one profiled ordered-IUT "
            f"{AUDIT_METHOD!r} row; "
            f"found {len(audit_rows)}"
        )
    return audit_rows.iloc[0]


def _audit_seed_record(
    row: pd.Series,
    *,
    seed: int,
    records_path: Path,
    target: float,
    fresh_rollouts: int,
    horizon: int,
) -> dict[str, Any]:
    if row["track"] != TRACK_A:
        raise AuditError(f"exact-MDP audit has the wrong track in {records_path}")
    if row["information_regime"] != AUDIT_REGIME:
        raise AuditError(f"SC-PCP has the wrong information regime in {records_path}")
    if row["selection_estimand"] != "per_step":
        raise AuditError(f"SC-PCP is not per-step in {records_path}")
    if row["selection_parameter"] != SELECTION_PARAMETER:
        raise AuditError(
            f"SC-PCP is not a profiled global-scale selection in {records_path}"
        )
    if row["certificate_type"] != AUDIT_CERTIFICATE:
        raise AuditError(f"SC-PCP has the wrong ordered-IUT certificate type in {records_path}")
    formal_bound = _as_bool(
        row["certificate_formal"], label=f"certificate_formal in {records_path}"
    )
    if not formal_bound:
        raise AuditError(f"exact-MDP SC-PCP is not marked formal in {records_path}")

    profile = _positive_vector(
        row["stage_profile"],
        label=f"stage_profile in {records_path}",
        length=horizon,
    )
    if not math.isclose(
        float(np.exp(np.log(profile).mean())),
        1.0,
        rel_tol=0.0,
        abs_tol=2e-6,
    ):
        raise AuditError(f"stage_profile does not have geometric mean one in {records_path}")

    status = str(row["selection_status"])
    if status not in {CERTIFIED_STATUS, UNCERTIFIED_STATUS}:
        raise AuditError(f"unexpected ordered-IUT selection status {status!r} in {records_path}")
    selected = status == CERTIFIED_STATUS
    certified = _as_bool(row["certified"], label=f"certified in {records_path}")
    if certified != selected:
        raise AuditError(f"certified flag and selection status disagree in {records_path}")
    selection_available = _as_bool(
        row["selection_available"],
        label=f"selection_available in {records_path}",
    )
    if selection_available != selected:
        raise AuditError(f"selection_available and selection status disagree in {records_path}")

    metrics = {
        name: _optional_float(row[name], label=f"{name} in {records_path}")
        for name in (
            "selected_scale",
            "estimated_min_coverage",
            "lower_bound_min",
            "worst_coverage",
            "average_coverage",
            "worst_gap",
        )
    }
    evaluation_rollouts = _integer(
        row["oracle_evaluation_trajectories"],
        label=f"oracle_evaluation_trajectories in {records_path}",
    )

    if selected:
        nonfinite = [name for name, value in metrics.items() if not math.isfinite(value)]
        if nonfinite:
            raise AuditError(
                f"certified exact-MDP selection has missing metrics in {records_path}: "
                f"{', '.join(nonfinite)}"
            )
        if row["evaluation_scope"] != FRESH_SCOPE:
            raise AuditError(f"certified SC-PCP selection lacks fresh evaluation in {records_path}")
        if evaluation_rollouts != fresh_rollouts:
            raise AuditError(
                f"fresh rollout budget in {records_path} is {evaluation_rollouts}, "
                f"expected {fresh_rollouts}"
            )
        if not all(
            0.0 <= metrics[name] <= 1.0
            for name in ("estimated_min_coverage", "lower_bound_min", "worst_coverage", "average_coverage", "worst_gap")
        ):
            raise AuditError(f"coverage metric lies outside [0,1] in {records_path}")
        if metrics["lower_bound_min"] < target - 2e-6:
            raise AuditError(f"selected formal LCB is below target in {records_path}")
        if metrics["estimated_min_coverage"] < metrics["lower_bound_min"] - 2e-6:
            raise AuditError(f"selected estimate is below its LCB in {records_path}")
        deployed_radii = _positive_vector(
            row["q_by_time"],
            label=f"q_by_time in {records_path}",
            length=horizon,
        )
        expected_radii = metrics["selected_scale"] * profile
        if not np.allclose(deployed_radii, expected_radii, rtol=2e-6, atol=2e-6):
            raise AuditError(
                f"q_by_time does not equal selected_scale times stage_profile in {records_path}"
            )
        expected_gap = max(0.0, target - metrics["worst_coverage"])
        if not math.isclose(
            metrics["worst_gap"], expected_gap, rel_tol=0.0, abs_tol=2e-6
        ):
            raise AuditError(f"worst_gap is inconsistent in {records_path}")
    else:
        if row["evaluation_scope"] != UNAVAILABLE_SCOPE:
            raise AuditError(f"abstention has the wrong evaluation scope in {records_path}")
        if evaluation_rollouts != 0:
            raise AuditError(f"abstention consumed fresh evaluation rollouts in {records_path}")
        present = [name for name, value in metrics.items() if math.isfinite(value)]
        if present:
            raise AuditError(
                f"abstained exact-MDP record has deployment metrics in {records_path}: "
                f"{', '.join(present)}"
            )

    target_met = bool(selected and metrics["worst_coverage"] >= target - 1e-7)
    return {
        "seed": seed,
        "target_coverage": target,
        "formal_bound_available": True,
        "formal_selected": selected,
        "abstained": not selected,
        "fresh_evaluated": selected,
        "fresh_target_met_all_runs": target_met,
        "fresh_target_met_evaluated": target_met if selected else np.nan,
        "stage_profile_geometric_mean": float(np.exp(np.log(profile).mean())),
        **metrics,
        "fresh_evaluation_trajectories": evaluation_rollouts,
    }


def audit_validation_run(
    root: Path,
    *,
    expected_seeds: tuple[int, ...],
    fresh_rollouts: int = 50_000,
) -> ValidationAudit:
    root = root.resolve()
    if not root.is_dir():
        raise AuditError(f"validation root is not a directory: {root}")
    if fresh_rollouts <= 0:
        raise AuditError("fresh rollout budget must be positive")

    require_complete_marker(root, "tabular theorem-validation run")
    require_status(
        root,
        expected_key="expected_seeds",
        completed_key="completed_seeds",
        expected_values=list(expected_seeds),
    )
    metadata = read_json(root / "study_metadata.json")
    source_hash = str(metadata.get("source_tree_sha256", ""))
    if SOURCE_HASH.fullmatch(source_hash) is None:
        raise AuditError(f"invalid source_tree_sha256 in {root / 'study_metadata.json'}")
    if metadata.get("seeds") != list(expected_seeds):
        raise AuditError("study metadata does not contain the prespecified seed set")
    execution = metadata.get("execution", {})
    if not isinstance(execution, Mapping):
        raise AuditError(f"execution metadata is not a mapping in {root / 'study_metadata.json'}")
    collection_hash = execution.get("collection_source_tree_sha256")
    if collection_hash is not None and collection_hash != source_hash:
        raise AuditError("collection and setting source hashes disagree in study metadata")

    config = read_yaml(root / "config.yaml")
    if config.get("seeds") != list(expected_seeds):
        raise AuditError("root config does not contain the prespecified seed set")
    dataset = _nested(config, "data", "dataset", label="data.dataset in root config")
    if dataset != "tabular":
        raise AuditError(f"the theorem-validation dataset must be tabular, found {dataset!r}")
    horizon = _integer(config.get("horizon"), label="horizon in root config")
    if horizon < 1:
        raise AuditError("horizon must be positive")
    alpha = _finite_float(
        _nested(config, "certification", "alpha", label="certification.alpha in root config"),
        label="certification.alpha in root config",
    )
    if not 0.0 < alpha < 1.0:
        raise AuditError("certification.alpha must lie strictly between zero and one")
    configured_rollouts = _integer(
        _nested(config, "samples", "oracle_rollouts", label="samples.oracle_rollouts in root config"),
        label="samples.oracle_rollouts in root config",
    )
    if configured_rollouts != fresh_rollouts:
        raise AuditError(
            f"root config requests {configured_rollouts} fresh rollouts; expected {fresh_rollouts}"
        )
    _validate_optional_manifest(root, source_hash, expected_seeds)

    seed_directories = _exact_seed_directories(root, expected_seeds)
    rows: list[dict[str, Any]] = []
    target = 1.0 - alpha
    for seed in expected_seeds:
        seed_root = seed_directories[seed]
        require_complete_marker(seed_root, f"tabular seed {seed}")
        seed_metadata_path = seed_root / "metadata.json"
        seed_metadata = read_json(seed_metadata_path)
        if seed_metadata.get("seed") != seed:
            raise AuditError(f"seed ID mismatch in {seed_metadata_path}")
        if seed_metadata.get("source_tree_sha256") != source_hash:
            raise AuditError(f"source hash mismatch in {seed_metadata_path}")
        if seed_metadata.get("config") != config:
            raise AuditError(f"seed config differs from root config in {seed_metadata_path}")
        records_path = seed_root / "records.csv"
        try:
            frame = pd.read_csv(records_path)
        except (OSError, pd.errors.ParserError) as error:
            raise AuditError(f"cannot read {records_path}: {error}") from error
        audit_record = _extract_audit_record(frame, records_path)
        rows.append(
            _audit_seed_record(
                audit_record,
                seed=seed,
                records_path=records_path,
                target=target,
                fresh_rollouts=fresh_rollouts,
                horizon=horizon,
            )
        )

    return ValidationAudit(
        root=root,
        source_hash=source_hash,
        config=config,
        target_coverage=target,
        configured_fresh_rollouts=configured_rollouts,
        seed_rows=pd.DataFrame(rows).sort_values("seed").reset_index(drop=True),
    )


def _mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.mean()) if not finite.empty else math.nan


def _standard_error(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return math.nan
    if len(finite) == 1:
        return 0.0
    return float(finite.std(ddof=1) / math.sqrt(len(finite)))


def summarize_validation(audit: ValidationAudit) -> pd.DataFrame:
    rows = audit.seed_rows
    evaluated = rows[rows["fresh_evaluated"]]
    selected = rows[rows["formal_selected"]]
    n_runs = len(rows)
    n_selected = int(rows["formal_selected"].sum())
    n_target_met = int(rows["fresh_target_met_all_runs"].sum())
    return pd.DataFrame(
        [
            {
                "method": AUDIT_METHOD,
                "information_regime": AUDIT_REGIME,
                "source_tree_sha256": audit.source_hash,
                "n_seeds": n_runs,
                "target_coverage": audit.target_coverage,
                "formal_bound_available_rate": float(rows["formal_bound_available"].mean()),
                "formal_certificate_rate": n_selected / n_runs,
                "formal_selection_rate": n_selected / n_runs,
                "n_formal_selected": n_selected,
                "abstention_rate": float(rows["abstained"].mean()),
                "n_abstained": int(rows["abstained"].sum()),
                "n_fresh_evaluated": len(evaluated),
                "fresh_evaluation_rate": len(evaluated) / n_runs,
                "fresh_target_met_count_all_runs": n_target_met,
                "fresh_target_met_rate_all_runs_abstention_as_failure": n_target_met / n_runs,
                "fresh_target_met_rate_evaluated": _mean(
                    evaluated["fresh_target_met_evaluated"]
                ),
                "mean_worst_coverage_evaluated": _mean(evaluated["worst_coverage"]),
                "se_worst_coverage_evaluated": _standard_error(evaluated["worst_coverage"]),
                "min_worst_coverage_evaluated": (
                    float(evaluated["worst_coverage"].min()) if not evaluated.empty else math.nan
                ),
                "mean_worst_gap_evaluated": _mean(evaluated["worst_gap"]),
                "se_worst_gap_evaluated": _standard_error(evaluated["worst_gap"]),
                "max_worst_gap_evaluated": (
                    float(evaluated["worst_gap"].max()) if not evaluated.empty else math.nan
                ),
                "mean_selected_scale": _mean(selected["selected_scale"]),
                "mean_selected_estimated_min_coverage": _mean(
                    selected["estimated_min_coverage"]
                ),
                "mean_selected_lower_bound_min": _mean(selected["lower_bound_min"]),
                "min_selected_lower_bound_min": (
                    float(selected["lower_bound_min"].min()) if not selected.empty else math.nan
                ),
                "configured_fresh_rollouts_per_selected_seed": audit.configured_fresh_rollouts,
                "mean_fresh_rollouts_per_selected_seed": _mean(
                    evaluated["fresh_evaluation_trajectories"]
                ),
                "total_fresh_evaluation_trajectories": int(
                    rows["fresh_evaluation_trajectories"].sum()
                ),
            }
        ]
    )


def _format(value: Any, *, percent: bool = False) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not math.isfinite(value)):
        return "NA"
    if percent:
        return f"{100.0 * float(value):.1f}%"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def build_markdown(audit: ValidationAudit, summary: pd.DataFrame) -> str:
    row = summary.iloc[0]
    return "\n".join(
        [
            "# Exact finite-MDP theorem validation",
            "",
            f"Input: `{audit.root}`  ",
            f"Source tree SHA-256: `{audit.source_hash}`  ",
            f"Seeds: {int(row['n_seeds'])}; target per-step coverage: {row['target_coverage']:.3f}.",
            "",
            "| Quantity | Result |",
            "| --- | ---: |",
            f"| Formal bound available | {_format(row['formal_bound_available_rate'], percent=True)} |",
            f"| Formal ordered-IUT certificate / scale selected | {_format(row['formal_selection_rate'], percent=True)} ({int(row['n_formal_selected'])}/{int(row['n_seeds'])}) |",
            f"| Abstained | {_format(row['abstention_rate'], percent=True)} ({int(row['n_abstained'])}/{int(row['n_seeds'])}) |",
            f"| Fresh target met, all runs (abstention = failure) | {_format(row['fresh_target_met_rate_all_runs_abstention_as_failure'], percent=True)} ({int(row['fresh_target_met_count_all_runs'])}/{int(row['n_seeds'])}) |",
            f"| Fresh target met, evaluated runs only | {_format(row['fresh_target_met_rate_evaluated'], percent=True)} |",
            f"| Evaluated worst-step coverage, mean (SE) | {_format(row['mean_worst_coverage_evaluated'])} ({_format(row['se_worst_coverage_evaluated'])}) |",
            f"| Evaluated worst-step coverage, minimum | {_format(row['min_worst_coverage_evaluated'])} |",
            f"| Evaluated worst gap, mean (SE) | {_format(row['mean_worst_gap_evaluated'])} ({_format(row['se_worst_gap_evaluated'])}) |",
            f"| Selected estimated minimum coverage, mean | {_format(row['mean_selected_estimated_min_coverage'])} |",
            f"| Selected formal LCB minimum, mean | {_format(row['mean_selected_lower_bound_min'])} |",
            f"| Selected global scale, mean | {_format(row['mean_selected_scale'])} |",
            f"| Fresh rollout budget per selected seed | {_format(row['configured_fresh_rollouts_per_selected_seed'])} |",
            f"| Total fresh evaluation trajectories | {_format(row['total_fresh_evaluation_trajectories'])} |",
            "",
            "## Interpretation boundary",
            "",
            "This is a theorem-path validation in an exactly enumerable finite MDP. The formal "
            "premise is the internally computed population L1 error bound for the capped learned "
            "COT weights. The stage profile is frozen on D_COT, and the same SC-PCP record selects "
            "one global scale by stagewise IUT plus a widest-to-narrowest fixed sequence. Formal "
            "selection and empirical fresh-rollout target attainment are reported separately; "
            "abstentions count as failures in the all-run attainment rate.",
            "",
            "The single SC-PCP record carries the exact finite-MDP evidence. This evidence does not "
            "turn the practical patient-cluster ordered-IUT LCB used in continuous or clinical "
            "experiments into a theorem certificate, and the internal oracle-L1 premise must not "
            "be extrapolated to clinical datasets.",
            "",
        ]
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_report(output: Path, audit: ValidationAudit, summary: pd.DataFrame) -> None:
    output = output.resolve()
    if output.exists():
        raise AuditError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        audit.seed_rows.to_csv(
            temporary / "theorem_validation_seed_level.csv",
            index=False,
            float_format="%.10g",
        )
        summary.to_csv(
            temporary / "theorem_validation_summary.csv",
            index=False,
            float_format="%.10g",
        )
        (temporary / "theorem_validation_report.md").write_text(
            build_markdown(audit, summary), encoding="utf-8"
        )
        payload = {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_root": str(audit.root),
            "source_tree_sha256": audit.source_hash,
            "expected_seeds": audit.seed_rows["seed"].astype(int).tolist(),
            "method": AUDIT_METHOD,
            "information_regime": AUDIT_REGIME,
            "selection_parameter": SELECTION_PARAMETER,
            "method_schema": "profiled_scale_ordered_iut",
            "fresh_rollout_budget_per_selected_seed": audit.configured_fresh_rollouts,
            "summary": _jsonable(summary.iloc[0].to_dict()),
            "interpretation": {
                "theorem_scope": "exactly enumerable finite MDP only",
                "formal_premise": (
                    "internal population L1 error bound for capped learned COT weights"
                ),
                "abstention_rule": "abstention counts as failure in the all-run fresh target-met rate",
                "continuous_or_clinical_scpcp": (
                    "uses practical patient-cluster ordered-IUT bounds and remains non-formal"
                ),
                "clinical_extrapolation": "not permitted",
            },
        }
        (temporary / "theorem_validation_summary.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        complete = {
            "status": "complete",
            "source_tree_sha256": audit.source_hash,
            "expected_seeds": payload["expected_seeds"],
        }
        (temporary / "COMPLETE").write_text(
            json.dumps(complete, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--expected-seeds",
        type=parse_seed_spec,
        default=parse_seed_spec("0:200"),
        metavar="0:200",
        help="exact seed range or list (default: 0:200)",
    )
    parser.add_argument(
        "--fresh-rollouts",
        type=int,
        default=50_000,
        help="required fresh target-policy rollout budget for every selected seed",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit = audit_validation_run(
            args.input_root,
            expected_seeds=args.expected_seeds,
            fresh_rollouts=args.fresh_rollouts,
        )
        summary = summarize_validation(audit)
        write_report(args.output, audit, summary)
    except AuditError as error:
        print(f"theorem-validation report refused: {error}", file=sys.stderr)
        return 2
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
