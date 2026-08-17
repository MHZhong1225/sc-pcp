"""Select held-out hyperparameters for the two dynamic online baselines.

The selector is deliberately strict: it reads only completed Track-A
deployment records, requires the exact held-out seed set, and rejects a study
unless its collection, setting, and seed completion markers all agree.  The
selection rule is fixed before reading the metrics:

1. maximize fresh-target target-met rate;
2. maximize mean fresh-target worst-step coverage;
3. minimize mean fresh-target log volume.

No partial report is produced.  The output bundle is published atomically and
its ``COMPLETE`` marker is written last inside the temporary bundle.
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
import yaml


TRACK_A = "empirical_environment"
FRESH_SCOPE = "fresh_target_policy_rollouts_in_frozen_empirical_environment"
ONLINE_REGIME = "on_policy_adaptation"
SEED_DIRECTORY = re.compile(r"seed_(\d+)$")
METHOD_PATTERNS = {
    "aci_gamma": re.compile(r"ACI-style online \(gamma=([^()]+)\)$"),
    "multidim_buffer": re.compile(r"MultiDimSPCI-style online \(buffer=([^()]+)\)$"),
}
METHOD_NAMES = {
    "aci_gamma": "ACI-style online",
    "multidim_buffer": "MultiDimSPCI-style online",
}


class AuditError(RuntimeError):
    """An input artifact is incomplete, inconsistent, or ambiguous."""


@dataclass(frozen=True)
class StudyAudit:
    family: str
    root: Path
    source_hash: str
    context: dict[str, Any]
    seed_rows: pd.DataFrame


def parse_seed_spec(value: str) -> tuple[int, ...]:
    """Parse a half-open range (``200:220``) or an explicit seed list."""

    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("the expected seed specification is empty")
    try:
        if ":" in value:
            parts = value.split(":")
            if len(parts) != 2:
                raise ValueError
            start, stop = (int(part) for part in parts)
            seeds = tuple(range(start, stop))
        else:
            seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected seeds must be a half-open range such as 200:220 or a comma-separated list"
        ) from error
    if not seeds:
        raise argparse.ArgumentTypeError("the expected seed set is empty")
    if min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("expected seeds must be unique non-negative integers")
    return tuple(sorted(seeds))


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AuditError(f"required JSON file is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise AuditError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AuditError(f"expected a JSON object in {path}")
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AuditError(f"required YAML file is missing: {path}") from error
    except yaml.YAMLError as error:
        raise AuditError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AuditError(f"expected a YAML mapping in {path}")
    return payload


def require_complete_marker(root: Path, label: str) -> None:
    marker = root / "COMPLETE"
    try:
        text = marker.read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise AuditError(f"{label} is partial: missing {marker}") from error
    if text.lower() == "complete":
        return
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise AuditError(f"{label} has an invalid COMPLETE marker: {marker}") from error
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise AuditError(f"{label} does not have status=complete in {marker}")


def require_status(
    root: Path,
    *,
    expected_key: str,
    completed_key: str,
    expected_values: Sequence[int | str],
) -> None:
    path = root / "study_status.json"
    status = read_json(path)
    if status.get("status") != "complete":
        raise AuditError(f"partial study status in {path}: {status.get('status')!r}")
    expected = list(expected_values)
    if status.get(expected_key) != expected:
        raise AuditError(f"{expected_key} in {path} does not equal the prespecified set")
    if status.get(completed_key) != expected:
        raise AuditError(f"{completed_key} in {path} does not equal the prespecified set")
    missing_key = "missing_seeds" if expected_key == "expected_seeds" else "missing_settings"
    if status.get(missing_key) != []:
        raise AuditError(f"{missing_key} is nonempty in {path}")


def _float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AuditError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise AuditError(f"{label} is not finite: {value!r}")
    return result


def _integer(value: Any, *, label: str) -> int:
    numeric = _float(value, label=label)
    rounded = int(round(numeric))
    if not math.isclose(numeric, rounded, abs_tol=1e-9):
        raise AuditError(f"{label} is not an integer: {value!r}")
    return rounded


def _exact_seed_directories(root: Path, expected_seeds: tuple[int, ...]) -> dict[int, Path]:
    observed: dict[int, Path] = {}
    unexpected_directories: list[str] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        match = SEED_DIRECTORY.fullmatch(path.name)
        if match is None:
            unexpected_directories.append(path.name)
            continue
        seed = int(match.group(1))
        if seed in observed:
            raise AuditError(
                f"duplicate seed directory IDs under {root}: {observed[seed].name}, {path.name}"
            )
        observed[seed] = path
    if unexpected_directories:
        raise AuditError(
            f"unexpected directories under setting {root}: {', '.join(unexpected_directories)}"
        )
    expected = set(expected_seeds)
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise AuditError(f"seed directories under {root} mismatch; missing={missing}, extra={extra}")
    return observed


def _same_number(left: Any, right: Any, *, label: str) -> None:
    a = _float(left, label=f"{label} (left)")
    b = _float(right, label=f"{label} (right)")
    if not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12):
        raise AuditError(f"{label} mismatch: {a:g} != {b:g}")


def _context_without_selected_hyperparameters(config: Mapping[str, Any]) -> dict[str, Any]:
    context = json.loads(json.dumps(config))
    context.pop("output_dir", None)
    # Device placement is execution metadata, not a statistical or method
    # hyperparameter.  Held-out studies may occupy different physical GPUs as
    # long as their frozen source and every substantive config field agree.
    context.pop("devices", None)
    baselines = context.get("baselines")
    if isinstance(baselines, dict):
        baselines.pop("aci_gamma", None)
        baselines.pop("multidim_buffer", None)
    return context


def _extract_dynamic_row(
    frame: pd.DataFrame,
    *,
    family: str,
    configured_value: float,
    path: Path,
) -> pd.Series:
    required = {
        "track",
        "evaluation_scope",
        "method",
        "information_regime",
        "selection_estimand",
        "worst_coverage",
        "worst_gap",
        "mean_log_volume",
        "target_policy_trajectories",
        "oracle_evaluation_trajectories",
        "adaptation_trajectories",
        "adaptation_worst_coverage",
        "adaptation_average_coverage",
        "adaptation_pathwise_coverage",
        "adaptation_per_time_coverage",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AuditError(f"{path} is missing required columns: {', '.join(missing)}")
    pattern = METHOD_PATTERNS[family]
    matches = frame[
        frame["method"].fillna("").astype(str).str.fullmatch(pattern)
        & frame["track"].eq(TRACK_A)
        & frame["selection_estimand"].eq("per_step")
    ]
    if len(matches) != 1:
        raise AuditError(
            f"{path} must contain exactly one corresponding dynamic-baseline Track-A row; "
            f"found {len(matches)}"
        )
    row = matches.iloc[0]
    method_match = pattern.fullmatch(str(row["method"]))
    assert method_match is not None
    _same_number(
        method_match.group(1),
        configured_value,
        label=f"method label and configured {family} in {path}",
    )
    if row["information_regime"] != ONLINE_REGIME:
        raise AuditError(f"dynamic baseline does not use {ONLINE_REGIME} in {path}")
    if row["evaluation_scope"] != FRESH_SCOPE:
        raise AuditError(f"dynamic baseline is not evaluated on fresh target rollouts in {path}")
    return row


def _validate_adaptation_metrics(row: pd.Series, *, horizon: int, path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for column in (
        "adaptation_worst_coverage",
        "adaptation_average_coverage",
        "adaptation_pathwise_coverage",
    ):
        value = _float(row[column], label=f"{column} in {path}")
        if not 0.0 <= value <= 1.0:
            raise AuditError(f"{column} lies outside [0,1] in {path}")
        values[column] = value
    try:
        per_time = json.loads(str(row["adaptation_per_time_coverage"]))
    except json.JSONDecodeError as error:
        raise AuditError(f"invalid adaptation_per_time_coverage in {path}") from error
    if not isinstance(per_time, list) or len(per_time) != horizon:
        raise AuditError(
            f"adaptation_per_time_coverage in {path} must have horizon={horizon} entries"
        )
    per_time_values = [
        _float(value, label=f"adaptation_per_time_coverage in {path}") for value in per_time
    ]
    if any(value < 0.0 or value > 1.0 for value in per_time_values):
        raise AuditError(f"adaptation_per_time_coverage lies outside [0,1] in {path}")
    if not math.isclose(
        float(np.mean(per_time_values)),
        values["adaptation_average_coverage"],
        rel_tol=0.0,
        abs_tol=2e-6,
    ):
        raise AuditError(f"adaptation cumulative coverage is internally inconsistent in {path}")
    if not math.isclose(
        min(per_time_values),
        values["adaptation_worst_coverage"],
        rel_tol=0.0,
        abs_tol=2e-6,
    ):
        raise AuditError(f"adaptation worst coverage is internally inconsistent in {path}")
    return values


def _audit_seed(
    seed_root: Path,
    *,
    seed: int,
    source_hash: str,
    setting_config: dict[str, Any],
    family: str,
    hyperparameter: float,
    target_coverage: float,
) -> dict[str, Any]:
    require_complete_marker(seed_root, f"seed {seed}")
    metadata = read_json(seed_root / "metadata.json")
    if metadata.get("seed") != seed:
        raise AuditError(f"seed ID mismatch in {seed_root / 'metadata.json'}")
    if metadata.get("source_tree_sha256") != source_hash:
        raise AuditError(f"source hash mismatch in {seed_root / 'metadata.json'}")
    if metadata.get("config") != setting_config:
        raise AuditError(f"seed config differs from setting config in {seed_root / 'metadata.json'}")
    record_path = seed_root / "records.csv"
    try:
        frame = pd.read_csv(record_path)
    except (OSError, pd.errors.ParserError) as error:
        raise AuditError(f"cannot read {record_path}: {error}") from error
    row = _extract_dynamic_row(
        frame,
        family=family,
        configured_value=hyperparameter,
        path=record_path,
    )
    horizon = _integer(setting_config.get("horizon"), label=f"horizon in {record_path}")
    adaptation = _validate_adaptation_metrics(row, horizon=horizon, path=record_path)
    worst = _float(row["worst_coverage"], label=f"worst_coverage in {record_path}")
    worst_gap = _float(row["worst_gap"], label=f"worst_gap in {record_path}")
    log_volume = _float(row["mean_log_volume"], label=f"mean_log_volume in {record_path}")
    if not 0.0 <= worst <= 1.0 or not 0.0 <= worst_gap <= 1.0:
        raise AuditError(f"fresh coverage metric lies outside [0,1] in {record_path}")
    expected_gap = max(0.0, target_coverage - worst)
    if not math.isclose(worst_gap, expected_gap, rel_tol=0.0, abs_tol=2e-6):
        raise AuditError(f"worst_gap does not match target minus worst coverage in {record_path}")
    adaptation_budget = _integer(
        row["adaptation_trajectories"], label=f"adaptation_trajectories in {record_path}"
    )
    target_budget = _integer(
        row["target_policy_trajectories"],
        label=f"target_policy_trajectories in {record_path}",
    )
    evaluation_budget = _integer(
        row["oracle_evaluation_trajectories"],
        label=f"oracle_evaluation_trajectories in {record_path}",
    )
    if adaptation_budget <= 0 or adaptation_budget != target_budget:
        raise AuditError(f"online adaptation budget is invalid or inconsistent in {record_path}")
    if evaluation_budget <= 0:
        raise AuditError(f"fresh evaluation budget is non-positive in {record_path}")
    return {
        "family": family,
        "method": METHOD_NAMES[family],
        "hyperparameter": hyperparameter,
        "seed": seed,
        "target_coverage": target_coverage,
        "fresh_target_met": float(worst >= target_coverage - 1e-7),
        "worst_coverage": worst,
        "worst_gap": worst_gap,
        "mean_log_volume": log_volume,
        "adaptation_cumulative_coverage": adaptation["adaptation_average_coverage"],
        "adaptation_worst_coverage": adaptation["adaptation_worst_coverage"],
        "adaptation_pathwise_coverage": adaptation["adaptation_pathwise_coverage"],
        "adaptation_trajectories": adaptation_budget,
        "evaluation_trajectories": evaluation_budget,
    }


def audit_study(root: Path, family: str, expected_seeds: tuple[int, ...]) -> StudyAudit:
    """Audit one multi-setting baseline tuning collection and extract its rows."""

    root = root.resolve()
    if family not in METHOD_PATTERNS:
        raise ValueError(f"unsupported baseline family: {family}")
    if not root.is_dir():
        raise AuditError(f"study root is not a directory: {root}")
    require_complete_marker(root, f"{family} collection")
    manifest = read_json(root / "study_manifest.json")
    if manifest.get("study") != family:
        raise AuditError(
            f"expected study={family!r} in {root / 'study_manifest.json'}, "
            f"found {manifest.get('study')!r}"
        )
    if manifest.get("seeds") != list(expected_seeds):
        raise AuditError(f"manifest seeds in {root} do not equal the prespecified held-out set")
    settings = manifest.get("settings")
    if not isinstance(settings, list) or not settings:
        raise AuditError(f"study manifest has no settings: {root / 'study_manifest.json'}")
    labels: list[str] = []
    manifest_rows: dict[str, dict[str, Any]] = {}
    for item in settings:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            raise AuditError(f"invalid setting entry in {root / 'study_manifest.json'}")
        label = item["label"]
        if label in manifest_rows:
            raise AuditError(f"duplicate setting label {label!r} in {root / 'study_manifest.json'}")
        labels.append(label)
        manifest_rows[label] = item
    expected_labels = sorted(labels)
    require_status(
        root,
        expected_key="expected_settings",
        completed_key="completed_settings",
        expected_values=expected_labels,
    )
    actual_setting_dirs = sorted(
        path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if actual_setting_dirs != expected_labels:
        raise AuditError(
            f"setting directories in {root} disagree with the manifest; "
            f"expected={expected_labels}, observed={actual_setting_dirs}"
        )

    seed_rows: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    contexts: list[dict[str, Any]] = []
    hyperparameters: set[float] = set()
    for label in labels:
        setting_root = root / label
        require_complete_marker(setting_root, f"setting {label}")
        require_status(
            setting_root,
            expected_key="expected_seeds",
            completed_key="completed_seeds",
            expected_values=list(expected_seeds),
        )
        study_metadata = read_json(setting_root / "study_metadata.json")
        source_hash = str(study_metadata.get("source_tree_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
            raise AuditError(
                f"missing or invalid source_tree_sha256 in {setting_root / 'study_metadata.json'}"
            )
        source_hashes.add(source_hash)
        if study_metadata.get("seeds") != list(expected_seeds):
            raise AuditError(f"study metadata seeds mismatch in {setting_root}")
        config = read_yaml(setting_root / "config.yaml")
        if config.get("seeds") != list(expected_seeds):
            raise AuditError(f"config seeds mismatch in {setting_root / 'config.yaml'}")
        baselines = config.get("baselines")
        if not isinstance(baselines, dict) or family not in baselines:
            raise AuditError(f"missing baselines.{family} in {setting_root / 'config.yaml'}")
        hyperparameter = _float(
            baselines[family], label=f"baselines.{family} in {setting_root / 'config.yaml'}"
        )
        if family == "multidim_buffer" and hyperparameter != int(hyperparameter):
            raise AuditError(f"MultiDim buffer must be an integer in {setting_root / 'config.yaml'}")
        if hyperparameter in hyperparameters:
            raise AuditError(f"duplicate {family} value {hyperparameter:g} in {root}")
        hyperparameters.add(hyperparameter)
        manifest_value = manifest_rows[label].get(family)
        _same_number(
            manifest_value,
            hyperparameter,
            label=f"manifest and config {family} for setting {label}",
        )
        target_coverage = 1.0 - _float(
            config.get("certification", {}).get("alpha"),
            label=f"certification.alpha in {setting_root / 'config.yaml'}",
        )
        if not 0.0 < target_coverage < 1.0:
            raise AuditError(f"target coverage lies outside (0,1) in {setting_root / 'config.yaml'}")
        contexts.append(_context_without_selected_hyperparameters(config))
        seed_directories = _exact_seed_directories(setting_root, expected_seeds)
        for seed in expected_seeds:
            seed_rows.append(
                _audit_seed(
                    seed_directories[seed],
                    seed=seed,
                    source_hash=source_hash,
                    setting_config=config,
                    family=family,
                    hyperparameter=hyperparameter,
                    target_coverage=target_coverage,
                )
            )
    if len(source_hashes) != 1:
        raise AuditError(f"settings in {root} do not share one source hash: {sorted(source_hashes)}")
    reference_context = contexts[0]
    if any(context != reference_context for context in contexts[1:]):
        raise AuditError(f"settings in {root} differ beyond the selected baseline hyperparameter")
    return StudyAudit(
        family=family,
        root=root,
        source_hash=next(iter(source_hashes)),
        context=reference_context,
        seed_rows=pd.DataFrame(seed_rows),
    )


def standard_error(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0
    return float(numeric.std(ddof=1) / math.sqrt(len(numeric)))


def summarize_and_select(seed_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    grouped = seed_rows.groupby(["family", "method", "hyperparameter"], sort=True)
    for (family, method, hyperparameter), group in grouped:
        target_values = group["target_coverage"].unique()
        if len(target_values) != 1:
            raise AuditError(f"target coverage changes within {family}={hyperparameter:g}")
        rows.append(
            {
                "family": family,
                "method": method,
                "hyperparameter": hyperparameter,
                "n_seeds": len(group),
                "target_coverage": float(target_values[0]),
                "fresh_target_met_rate": float(group["fresh_target_met"].mean()),
                "mean_worst_coverage": float(group["worst_coverage"].mean()),
                "se_worst_coverage": standard_error(group["worst_coverage"]),
                "mean_worst_gap": float(group["worst_gap"].mean()),
                "se_worst_gap": standard_error(group["worst_gap"]),
                "mean_log_volume": float(group["mean_log_volume"].mean()),
                "se_log_volume": standard_error(group["mean_log_volume"]),
                "mean_adaptation_cumulative_coverage": float(
                    group["adaptation_cumulative_coverage"].mean()
                ),
                "se_adaptation_cumulative_coverage": standard_error(
                    group["adaptation_cumulative_coverage"]
                ),
                "mean_adaptation_worst_coverage": float(
                    group["adaptation_worst_coverage"].mean()
                ),
                "mean_adaptation_pathwise_coverage": float(
                    group["adaptation_pathwise_coverage"].mean()
                ),
                "mean_adaptation_trajectories": float(group["adaptation_trajectories"].mean()),
                "total_adaptation_trajectories": int(group["adaptation_trajectories"].sum()),
                "mean_evaluation_trajectories": float(group["evaluation_trajectories"].mean()),
                "total_evaluation_trajectories": int(group["evaluation_trajectories"].sum()),
            }
        )
    summary = pd.DataFrame(rows)
    selected: dict[str, float] = {}
    ranked_frames: list[pd.DataFrame] = []
    for family, family_rows in summary.groupby("family", sort=True):
        ranked = family_rows.sort_values(
            ["fresh_target_met_rate", "mean_worst_coverage", "mean_log_volume", "hyperparameter"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).copy()
        if len(ranked) > 1:
            columns = ["fresh_target_met_rate", "mean_worst_coverage", "mean_log_volume"]
            if ranked.iloc[0][columns].equals(ranked.iloc[1][columns]):
                raise AuditError(
                    f"the prespecified three-part rule leaves an exact tie for {family}; "
                    "no unregistered tie-break is applied"
                )
        ranked["lexicographic_rank"] = range(1, len(ranked) + 1)
        ranked["selected"] = ranked["lexicographic_rank"].eq(1)
        selected[family] = float(ranked.iloc[0]["hyperparameter"])
        ranked_frames.append(ranked)
    result = pd.concat(ranked_frames, ignore_index=True)
    result = result.sort_values(["family", "lexicographic_rank"]).reset_index(drop=True)
    return result, selected


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "family",
        "hyperparameter",
        "n_seeds",
        "fresh_target_met_rate",
        "mean_worst_coverage",
        "mean_worst_gap",
        "mean_log_volume",
        "se_log_volume",
        "mean_adaptation_cumulative_coverage",
        "mean_adaptation_trajectories",
        "mean_evaluation_trajectories",
        "lexicographic_rank",
        "selected",
    ]
    labels = [
        "Baseline",
        "Hyperparameter",
        "Seeds",
        "Fresh target-met",
        "Worst coverage",
        "Worst gap",
        "Log volume",
        "Log-volume SE",
        "Adapt. cumulative cov.",
        "Adapt. budget/seed",
        "Eval. budget/seed",
        "Rank",
        "Selected",
    ]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(labels)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if isinstance(value, (bool, np.bool_)):
                values.append("yes" if value else "")
            elif isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def write_bundle(
    output: Path,
    *,
    summary: pd.DataFrame,
    selected: Mapping[str, float],
    audits: Sequence[StudyAudit],
    expected_seeds: tuple[int, ...],
) -> None:
    output = output.resolve()
    if output.exists():
        raise AuditError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        summary.to_csv(
            temporary / "baseline_hyperparameter_summary.csv",
            index=False,
            float_format="%.10g",
        )
        seed_rows = pd.concat(
            [audit.seed_rows.assign(study_root=str(audit.root)) for audit in audits],
            ignore_index=True,
        )
        seed_rows.to_csv(
            temporary / "baseline_hyperparameter_seed_level.csv",
            index=False,
            float_format="%.10g",
        )
        report = [
            "# Dynamic-baseline held-out hyperparameter selection",
            "",
            "Selection rule (fixed in this order): maximize fresh-target target-met rate; "
            "maximize mean fresh-target worst-step coverage; minimize mean fresh-target log volume.",
            "",
            markdown_table(summary),
            "",
            "Adaptation cumulative coverage is the coverage pooled across adaptation "
            "trajectories and decision stages. Adaptation and fresh-evaluation budgets are "
            "reported separately.",
            "",
        ]
        (temporary / "baseline_hyperparameter_summary.md").write_text(
            "\n".join(report), encoding="utf-8"
        )
        source_hash = audits[0].source_hash
        payload = {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_rule": [
                "maximize fresh_target_met_rate",
                "maximize mean_worst_coverage",
                "minimize mean_log_volume",
            ],
            "expected_seeds": list(expected_seeds),
            "source_tree_sha256": source_hash,
            "inputs": {audit.family: str(audit.root) for audit in audits},
            "selected": {
                "aci_gamma": selected["aci_gamma"],
                "multidim_buffer": int(selected["multidim_buffer"]),
            },
            "summary": _json_records(summary),
        }
        (temporary / "baseline_hyperparameter_selection.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        complete = {
            "status": "complete",
            "source_tree_sha256": source_hash,
            "expected_seeds": list(expected_seeds),
            "selected": payload["selected"],
        }
        (temporary / "COMPLETE").write_text(
            json.dumps(complete, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def select_baselines(
    aci_root: Path,
    multidim_root: Path,
    expected_seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, float], tuple[StudyAudit, StudyAudit]]:
    aci = audit_study(aci_root, "aci_gamma", expected_seeds)
    multidim = audit_study(multidim_root, "multidim_buffer", expected_seeds)
    if aci.source_hash != multidim.source_hash:
        raise AuditError(
            "ACI and MultiDim studies use different source hashes: "
            f"{aci.source_hash} != {multidim.source_hash}"
        )
    if aci.context != multidim.context:
        raise AuditError(
            "ACI and MultiDim studies differ beyond their selected hyperparameters/output roots"
        )
    combined = pd.concat([aci.seed_rows, multidim.seed_rows], ignore_index=True)
    summary, selected = summarize_and_select(combined)
    return summary, selected, (aci, multidim)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aci-root", type=Path, required=True)
    parser.add_argument("--multidim-root", type=Path, required=True)
    parser.add_argument(
        "--expected-seeds",
        type=parse_seed_spec,
        default=parse_seed_spec("200:220"),
        metavar="200:220",
        help="exact held-out seed range or list (default: 200:220)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary, selected, audits = select_baselines(
            args.aci_root,
            args.multidim_root,
            args.expected_seeds,
        )
        write_bundle(
            args.output,
            summary=summary,
            selected=selected,
            audits=audits,
            expected_seeds=args.expected_seeds,
        )
    except AuditError as error:
        print(f"baseline selection refused: {error}", file=sys.stderr)
        return 2
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
