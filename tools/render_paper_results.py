"""Aggregate the four-RQ experiment and render the five manuscript PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


TARGET = 0.90
EXPECTED_ALPHA = 0.10
PAPER_PROTOCOL = "committed_prefix_marginal_scpcp"
PAPER_METHOD = "direct_committed_prefix_uncapped_importance_weighting"
SCPCP_GUARANTEE_SCOPE = "asymptotic_per_step_marginal"
SCPCP_SELECTION_EVIDENCE = "committed_prefix_uncapped_hajek_point_estimate"
FEEDBACK_LEVELS = (0.0, 0.5, 1.0, 2.0)
WORST_COVERAGE_BOOTSTRAP_RESAMPLES = 10_000
WORST_COVERAGE_BOOTSTRAP_SEED = 8_202_686
WORST_COVERAGE_BOOTSTRAP_BATCH_SIZE = 500
CONDITIONAL_SELECTION_NOTE = (
    "Coverage and width curves are conditional on successful method selection; "
    "see Selection Rate for abstentions."
)
METHOD_ORDER = (
    "Standard CP",
    "ACI",
    "MFCS",
    "SPCI",
    "PRC",
    "SC-PCP",
)
METHOD_LABELS = {
    "Standard CP": "Standard CP",
    "ACI": "ACI",
    "MFCS": "MFCS",
    "SPCI": "SPCI",
    "PRC": "PRC",
    "SC-PCP": "SC-PCP",
}
COLORS = {
    "Standard CP": "#7e8c9c",
    "ACI": "#aa3831",
    "MFCS": "#02bec4",
    "SPCI": "#7a3d9d",
    "PRC": "#448c27",
    "SC-PCP": "#4394f8",
}
MARKERS = {
    "Standard CP": "o",
    "ACI": "v",
    "MFCS": "D",
    "SPCI": "h",
    "PRC": ">",
    "SC-PCP": "*",
}
LINESTYLES = {
    "Standard CP": (0, (7, 3)),
    "ACI": "-.",
    "MFCS": ":",
    "SPCI": (0, (3, 2)),
    "PRC": (0, (5, 2, 1, 2, 1, 2)),
    "SC-PCP": "-",
}
DATASET_LABELS = {
    "synthetic": "Synthetic",
    "mimic_iv": "MIMIC-IV",
    "mimic_cxr": "MIMIC-CXR + IV/ED",
    "eicu": "eICU",
    "inspire": "INSPIRE",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the SC-PCP paper results")
    parser.add_argument("--input", type=Path, default=Path("results/work/paper_final"))
    parser.add_argument("--output", type=Path, default=Path("results/paper_final"))
    args = parser.parse_args()

    validate_complete_suite(args.input)
    records = load_suite_records(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    _style()
    render_table(
        aggregate_main(
            records[
                records["dataset"].eq("synthetic")
                & records["feedback_strength"].eq(1.0)
            ]
        ),
        args.output / "table_1_synthetic_main.pdf",
        title="Table 1 | Synthetic main results (target marginal worst-step coverage = 0.90)",
        include_dataset=False,
    )
    render_table(
        aggregate_main(records[~records["dataset"].eq("synthetic")]),
        args.output / "table_2_clinical_main.pdf",
        title="Table 2 | Clinical main results in frozen held-out empirical environments",
        include_dataset=True,
    )
    render_coverage_profiles(records, args.output / "figure_1_per_step_coverage.pdf")
    render_feedback_stress(records, args.output / "figure_2_feedback_stress.pdf")
    render_mechanism(args.input, args.output / "figure_3_committed_prefix_mechanism.pdf")
    _assert_pdf_only(args.output)
    print(args.output)


def validate_complete_suite(root: Path) -> None:
    """Fail closed unless every prespecified paper run is complete and intact."""

    if not (root / "COMPLETE").is_file():
        raise RuntimeError(f"paper suite is not complete: {root}")
    manifest_path = root / "suite_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("paper suite manifest is missing")
    manifest = _read_json_mapping(manifest_path)
    if manifest.get("protocol") != PAPER_PROTOCOL:
        raise RuntimeError("paper suite manifest has the wrong protocol")
    if manifest.get("method") != PAPER_METHOD:
        raise RuntimeError("paper suite manifest has the wrong SC-PCP method")
    if not _is_sha256(manifest.get("experiment_tree_sha256")):
        raise RuntimeError("paper suite manifest has an invalid experiment source hash")
    if set(manifest.get("sections", ())) != {"rq1", "rq3"}:
        raise RuntimeError("formal PDF rendering requires complete RQ1 and RQ3 sections")
    if set(manifest.get("datasets", ())) != set(DATASET_LABELS):
        raise RuntimeError("formal PDF rendering requires all five prespecified datasets")
    if tuple(manifest.get("feedback_levels", ())) != FEEDBACK_LEVELS:
        raise RuntimeError("formal PDF rendering requires the prespecified feedback levels")
    studies = [root / "rq1" / dataset for dataset in manifest["datasets"]]
    studies.extend(
        root / "rq3" / f"beta_{float(beta):g}"
        for beta in manifest["feedback_levels"]
        if float(beta) != 1.0
    )
    source_hashes: set[str] = set()
    git_revisions: set[str] = set()
    validated_records: set[Path] = set()
    for study in studies:
        source_hash, git_revision, record_paths = _validate_complete_study(study)
        source_hashes.add(source_hash)
        git_revisions.add(git_revision)
        validated_records.update(record_paths)
    if len(source_hashes) != 1 or len(git_revisions) != 1:
        raise RuntimeError("paper studies were not produced from one consistent source tree")
    observed_records = {path.resolve() for path in root.rglob("records.csv")}
    if observed_records != validated_records:
        raise RuntimeError("paper suite contains records outside the validated studies")


def _validate_complete_study(study: Path) -> tuple[str, str, set[Path]]:
    if not (study / "COMPLETE").is_file():
        raise RuntimeError(f"paper study is incomplete: {study}")
    config_path = study / "config.yaml"
    if not config_path.is_file():
        raise RuntimeError(f"paper study config is missing: {study}")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise RuntimeError(f"paper study config must be a mapping: {study}")
    certification = config.get("certification")
    alpha = None if not isinstance(certification, dict) else certification.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or float(alpha) != EXPECTED_ALPHA:
        raise RuntimeError(f"paper study must use alpha={EXPECTED_ALPHA:.2f}: {study}")
    expected_seeds = tuple(int(seed) for seed in config["seeds"])
    status = _read_json_mapping(study / "study_status.json")
    if (
        status.get("status") != "complete"
        or status.get("expected_seeds") != list(expected_seeds)
        or status.get("completed_seeds") != list(expected_seeds)
        or status.get("missing_seeds") not in ([], None)
        or status.get("error") is not None
    ):
        raise RuntimeError(f"paper study status is inconsistent: {study}")
    study_metadata = _read_json_mapping(study / "study_metadata.json")
    source_hash = study_metadata.get("source_tree_sha256")
    git_revision = study_metadata.get("git_revision")
    if not _is_sha256(source_hash) or not _is_git_revision(git_revision):
        raise RuntimeError(f"paper study has invalid source provenance: {study}")
    if study_metadata.get("seeds") != list(expected_seeds):
        raise RuntimeError(f"paper study metadata has a different seed set: {study}")
    observed_seed_dirs = {
        path.name
        for path in study.glob("seed_*")
        if path.is_dir()
    }
    expected_seed_dirs = {f"seed_{seed:05d}" for seed in expected_seeds}
    if observed_seed_dirs != expected_seed_dirs:
        raise RuntimeError(
            f"seed set mismatch in {study}: expected {len(expected_seed_dirs)}, "
            f"found {len(observed_seed_dirs)}"
        )
    record_paths: set[Path] = set()
    for seed_dir in sorted(expected_seed_dirs):
        seed_root = study / seed_dir
        if not (seed_root / "COMPLETE").is_file():
            raise RuntimeError(f"seed is incomplete: {seed_root}")
        record_path = seed_root / "records.csv"
        metadata_path = seed_root / "metadata.json"
        if not record_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"paper seed is missing records or metadata: {seed_root}")
        seed = int(seed_dir.split("_")[-1])
        seed_metadata = _read_json_mapping(metadata_path)
        if (
            seed_metadata.get("seed") != seed
            or seed_metadata.get("source_tree_sha256") != source_hash
            or seed_metadata.get("git_revision") != git_revision
            or seed_metadata.get("config") != config
        ):
            raise RuntimeError(f"paper seed provenance differs from its study: {seed_root}")
        diagnostics = seed_metadata.get("diagnostics")
        if not isinstance(diagnostics, dict) or (
            diagnostics.get("protocol") != PAPER_PROTOCOL
            or diagnostics.get("method") != PAPER_METHOD
            or diagnostics.get("guarantee_scope") != SCPCP_GUARANTEE_SCOPE
        ):
            raise RuntimeError(f"paper seed has the wrong SC-PCP protocol: {seed_root}")

        frame = pd.read_csv(record_path)
        empirical = frame[frame["track"].eq("empirical_environment")]
        methods = tuple(empirical["method"].astype(str))
        if len(methods) != len(METHOD_ORDER) or set(methods) != set(METHOD_ORDER):
            raise RuntimeError(f"seed does not contain exactly the six paper methods: {seed_root}")
        scpcp = empirical.loc[empirical["method"].eq("SC-PCP")].iloc[0]
        selection_available = _validate_scpcp_claim_record(scpcp, seed_root)
        if _strict_bool(
            diagnostics.get("scpcp_selection_available"),
            field="diagnostics.scpcp_selection_available",
            source=seed_root,
        ) != selection_available:
            raise RuntimeError(f"paper seed SC-PCP selection status disagrees: {seed_root}")
        record_paths.add(record_path.resolve())
    return str(source_hash), str(git_revision), record_paths


def _validate_scpcp_claim_record(record: pd.Series, source: Path) -> bool:
    expected_text = {
        "selection_estimand": "per_step_marginal",
        "selection_parameter": "stagewise_radii",
        "guarantee_scope": SCPCP_GUARANTEE_SCOPE,
        "selection_evidence": SCPCP_SELECTION_EVIDENCE,
    }
    required = {
        *expected_text,
        "selection_status",
        "selection_available",
        "certificate_type",
        "certificate_formal",
        "certified",
        "lower_bound_min",
    }
    missing = sorted(required - set(record.index))
    if missing:
        raise RuntimeError(f"SC-PCP record is missing claim fields {missing}: {source}")
    for field, expected in expected_text.items():
        if record[field] != expected:
            raise RuntimeError(f"SC-PCP record has invalid {field}: {source}")
    if not _is_missing(record["certificate_type"]) or not _is_missing(record["lower_bound_min"]):
        raise RuntimeError(f"marginal SC-PCP record must not contain a certificate: {source}")
    for field in ("certificate_formal", "certified"):
        if _strict_bool(record[field], field=field, source=source):
            raise RuntimeError(f"marginal SC-PCP record cannot set {field}=true: {source}")
    available = _strict_bool(
        record["selection_available"],
        field="selection_available",
        source=source,
    )
    expected_status = (
        "SELECTED_MARGINAL_POINT"
        if available
        else "UNAVAILABLE_NO_FEASIBLE_CANDIDATE"
    )
    if record["selection_status"] != expected_status:
        raise RuntimeError(f"SC-PCP record has inconsistent selection status: {source}")
    return available


def _read_json_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"required JSON artifact is missing: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"JSON artifact is malformed: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_git_revision(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _is_missing(value: object) -> bool:
    return value is None or (not isinstance(value, str) and bool(pd.isna(value))) or (
        isinstance(value, str) and not value.strip()
    )


def _strict_bool(value: object, *, field: str, source: Path) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{field} must be a boolean: {source}")


def load_suite_records(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("records.csv")):
        if not (path.parent / "COMPLETE").is_file():
            continue
        run_root = path.parent.parent
        config = yaml.safe_load((run_root / "config.yaml").read_text())
        frame = pd.read_csv(path)
        frame = frame[frame["track"].eq("empirical_environment")].copy()
        frame["seed"] = int(path.parent.name.split("_")[-1])
        frame["run_root"] = str(run_root.resolve())
        frame["dataset"] = str(config["data"]["dataset"])
        frame["feedback_strength"] = float(config["synthetic"]["feedback_strength"])
        frame["method_family"] = frame["method"].map(method_family)
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"no completed paper records below {root}")
    records = pd.concat(rows, ignore_index=True)
    records = records[records["method_family"].isin(METHOD_ORDER)].copy()
    return records


def method_family(name: object) -> str:
    value = str(name)
    for family in METHOD_ORDER:
        if value == family or value.startswith(family + " ("):
            return family
    return value


def aggregate_main(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), group in records.groupby(["dataset", "method_family"], sort=False):
        selected = group["selection_available"].map(_as_bool)
        evaluated = group[selected]
        marginal_worst, marginal_worst_lo, marginal_worst_hi = (
            marginal_worst_coverage_ci(
                evaluated,
                dataset=str(dataset),
                method=str(method),
            )
        )
        average_cov_mean, average_cov_lo, average_cov_hi = mean_ci(
            evaluated["average_coverage"]
        )
        width_mean, width_lo, width_hi = mean_ci(evaluated["average_normalized_width"])
        selection_rate = float(selected.mean())
        selection_lo, selection_hi = wilson_ci(int(selected.sum()), len(selected))
        marginal_target_met = bool(
            np.isfinite(marginal_worst) and marginal_worst >= TARGET
        )
        selection_rate_eligible = selection_rate >= 0.95
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "marginal_worst_coverage": marginal_worst,
                "marginal_worst_coverage_ci_low": marginal_worst_lo,
                "marginal_worst_coverage_ci_high": marginal_worst_hi,
                "average_coverage": average_cov_mean,
                "average_coverage_ci_low": average_cov_lo,
                "average_coverage_ci_high": average_cov_hi,
                "average_normalized_width": width_mean,
                "average_normalized_width_ci_low": width_lo,
                "average_normalized_width_ci_high": width_hi,
                "selection_rate": selection_rate,
                "selection_rate_ci_low": selection_lo,
                "selection_rate_ci_high": selection_hi,
                "marginal_worst_target_met": marginal_target_met,
                "selection_rate_at_least_95_percent": selection_rate_eligible,
                "efficiency_eligible": (
                    marginal_target_met and selection_rate_eligible
                ),
                "n_runs": len(group),
                "n_selected": int(selected.sum()),
            }
        )
    result = pd.DataFrame(rows)
    method_rank = {name: rank for rank, name in enumerate(METHOD_ORDER)}
    dataset_rank = {name: rank for rank, name in enumerate(DATASET_LABELS)}
    return result.sort_values(
        ["dataset", "method"],
        key=lambda values: values.map(dataset_rank if values.name == "dataset" else method_rank),
    )


def render_table(summary: pd.DataFrame, path: Path, *, title: str, include_dataset: bool) -> None:
    rows = []
    efficient = _efficient_eligible_methods(summary)
    for row in summary.itertuples(index=False):
        values = []
        if include_dataset:
            values.append(DATASET_LABELS[row.dataset])
        values.extend(
            (
                METHOD_LABELS[row.method],
                ci_text(
                    row.marginal_worst_coverage,
                    row.marginal_worst_coverage_ci_low,
                    row.marginal_worst_coverage_ci_high,
                    digits=4,
                ),
                ci_text(
                    row.average_coverage,
                    row.average_coverage_ci_low,
                    row.average_coverage_ci_high,
                    digits=4,
                ),
                ci_text(
                    row.average_normalized_width,
                    row.average_normalized_width_ci_low,
                    row.average_normalized_width_ci_high,
                ),
                rate_text(row.selection_rate, row.n_selected, row.n_runs),
            )
        )
        rows.append(values)
    columns = (["Dataset"] if include_dataset else []) + [
        "Method",
        "Marginal worst-step\ncoverage point\n[seed-bootstrap 95% CI]",
        "Mean coverage\nmean [95% CI]",
        "Average normalized\nwidth ↓\nmean [95% CI]",
        "Selection rate ↑\nmean (selected/runs)",
    ]
    height = max(5.0, 0.43 * len(rows) + 2.3)
    figure, axis = plt.subplots(figsize=(18, height))
    axis.axis("off")
    axis.set_title(title, loc="left", fontweight="bold", pad=18)
    column_widths = (
        [0.17, 0.12, 0.19, 0.16, 0.18, 0.18]
        if include_dataset
        else None
    )
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        colWidths=column_widths,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(17)
    table.scale(1.0, 1.85)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_facecolor("#E8EEF7")
            cell.set_text_props(fontweight="bold")
            cell.set_height(cell.get_height() * 2.1)
        elif row_index % 2 == 0:
            cell.set_facecolor("#F7F7F7")
    width_column = 4 if include_dataset else 3
    method_column = 1 if include_dataset else 0
    for row_index, (_, source_row) in enumerate(summary.iterrows(), start=1):
        key = (str(source_row["dataset"]), str(source_row["method"]))
        if key in efficient:
            table[(row_index, width_column)].set_text_props(fontweight="bold")
            table[(row_index, method_column)].set_text_props(fontweight="bold")
    figure.text(
        0.01,
        0.015,
        "Marginal WSC is min_t of the across-seed mean per-step coverage over selected runs. Its percentile "
        f"95% CI resamples seeds as whole per-time vectors ({WORST_COVERAGE_BOOTSTRAP_RESAMPLES:,} draws).\n"
        "Mean coverage and width are averaged over selected runs; Selection Rate uses all prespecified runs. "
        "Bold marks the narrowest method with marginal WSC >= 0.90 and Selection Rate >= 95%.",
        ha="left",
        va="bottom",
        fontsize=17,
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def render_coverage_profiles(records: pd.DataFrame, path: Path) -> None:
    rows = (
        (records[records["dataset"].eq("synthetic") & records["feedback_strength"].eq(2.0)], "Synthetic, strong feedback"),
        (records[records["dataset"].eq("mimic_iv")], "MIMIC-IV"),
    )
    columns = (
        ("per_time_coverage", "Per-step coverage"),
        ("per_time_normalized_width", "Per-step normalized width"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(22, 15))
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.10,
        top=0.88,
        hspace=0.34,
        wspace=0.27,
    )
    panel_label = iter("abcd")
    for row_index, (source, dataset_label) in enumerate(rows):
        for column_index, (field, y_label) in enumerate(columns):
            axis = axes[row_index, column_index]
            for method in METHOD_ORDER:
                curves = [
                    parse_curve(value)
                    for value in source.loc[source["method_family"].eq(method), field]
                ]
                curves = [curve for curve in curves if curve.size]
                if not curves:
                    continue
                matrix = np.stack(curves)
                mean = matrix.mean(axis=0)
                low, high = curve_t_ci(
                    matrix,
                    probability=field == "per_time_coverage",
                )
                times = np.arange(len(mean))
                axis.errorbar(
                    times,
                    mean,
                    yerr=np.vstack(((mean - low).clip(min=0.0), (high - mean).clip(min=0.0))),
                    errorevery=2,
                    capsize=2.5,
                    alpha=0.95,
                    label=METHOD_LABELS[method],
                    **_method_plot_kwargs(method),
                )
            if field == "per_time_coverage":
                axis.axhline(TARGET, color="#111827", linestyle=(0, (1, 2)), linewidth=1.8)
            axis.set_xlabel("Treatment stage, t")
            axis.set_ylabel(y_label)
            axis.grid(True, color="#D1D5DB", linewidth=0.8, alpha=0.65)
            axis.text(-0.10, 1.02, next(panel_label), transform=axis.transAxes, fontweight="bold")
            horizons = [len(line.get_xdata()) for line in axis.lines if len(line.get_xdata()) > 1]
            if horizons:
                axis.set_xticks(np.arange(max(horizons)))
            if row_index == 0:
                axis.set_title(y_label, fontweight="bold")
        axes[row_index, 0].text(
            1.05,
            1.12,
            dataset_label,
            transform=axes[row_index, 0].transAxes,
            ha="center",
            fontweight="bold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.98))
    figure.text(
        0.5,
        0.025,
        CONDITIONAL_SELECTION_NOTE,
        ha="center",
        va="bottom",
        fontsize=17,
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def render_feedback_stress(records: pd.DataFrame, path: Path) -> None:
    source = records[records["dataset"].eq("synthetic")]
    figure, axes = plt.subplots(2, 2, figsize=(22, 15), constrained_layout=True)
    flat_axes = axes.ravel()
    for method in METHOD_ORDER:
        method_rows = source[source["method_family"].eq(method)]
        series = [[] for _ in range(10)]
        for beta, group in method_rows.groupby("feedback_strength"):
            selected = group["selection_available"].map(_as_bool)
            evaluated = group[selected]
            # The bootstrap key deliberately excludes beta, so each method
            # reuses the same seed-resampling stream across feedback levels.
            c, clo, chi = marginal_worst_coverage_ci(
                evaluated,
                dataset="synthetic",
                method=method,
            )
            m, mlo, mhi = mean_ci(evaluated["average_coverage"])
            w, wlo, whi = mean_ci(evaluated["average_normalized_width"])
            values = (beta, c, clo, chi, m, mlo, mhi, w, wlo, whi)
            for destination, value in zip(series, values):
                destination.append(value)
        x, *metric_values = series
        order = np.argsort(x)
        x = np.asarray(x)[order]
        for axis, (center, low, high) in zip(
            flat_axes,
            (
                metric_values[0:3],
                metric_values[3:6],
                metric_values[6:9],
            ),
        ):
            center = np.asarray(center)[order]
            low = np.asarray(low)[order]
            high = np.asarray(high)[order]
            axis.plot(
                x,
                center,
                label=METHOD_LABELS[method],
                **_method_plot_kwargs(method),
            )
            axis.fill_between(x, low, high, color=COLORS[method], alpha=0.10)
    selection_axis = flat_axes[3]
    feedback_levels = np.asarray(sorted(source["feedback_strength"].unique()))
    selection_rates = []
    for beta in feedback_levels:
        beta_rows = source[source["feedback_strength"].eq(beta)]
        for method in METHOD_ORDER:
            method_rows = beta_rows[beta_rows["method_family"].eq(method)]
            selection_rates.append(
                float(method_rows["selection_available"].map(_as_bool).mean())
            )
    if selection_rates and np.allclose(selection_rates, selection_rates[0]):
        selection_axis.plot(
            feedback_levels,
            np.full_like(feedback_levels, selection_rates[0], dtype=float),
            color="#7e8c9c",
            linewidth=2.2,
            marker="o",
            markersize=6,
        )
        selection_axis.text(
            0.5,
            0.50,
            f"All six methods: {100 * selection_rates[0]:.0f}% selection\nat every feedback strength",
            transform=selection_axis.transAxes,
            ha="center",
            va="center",
        )
        selection_axis.set_ylim(max(0.0, selection_rates[0] - 0.02), 1.002)
    else:
        raise RuntimeError("selection-rate panel requires an explicit multi-method rendering")
    flat_axes[0].axhline(TARGET, color="#111827", linestyle=(0, (1, 2)), linewidth=1.8)
    flat_axes[1].axhline(TARGET, color="#111827", linestyle=(0, (1, 2)), linewidth=1.8)
    labels = (
        "Marginal worst-step coverage",
        "Mean coverage",
        "Average normalized width",
        "Selection rate",
    )
    for panel, (axis, label) in enumerate(zip(flat_axes, labels)):
        axis.set_ylabel(label)
        axis.set_xlabel("Feedback strength, β")
        axis.set_xticks(sorted(source["feedback_strength"].unique()))
        axis.grid(True, color="#D1D5DB", linewidth=0.8, alpha=0.65)
        axis.text(-0.08, 1.03, "abcd"[panel], transform=axis.transAxes, fontweight="bold")
    handles, legend_labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def render_mechanism(root: Path, path: Path) -> None:
    synthetic_root = root / "rq1" / "synthetic"
    mechanism_seed = configured_mechanism_seed(synthetic_root)
    seed_root = synthetic_root / f"seed_{mechanism_seed:05d}"
    with np.load(seed_root / "surfaces.npz") as arrays:
        required = {
            "scpcp_stage_grids",
            "scpcp_candidate_coverage",
            "scpcp_selected_indices",
            "scpcp_selected_radii",
            "scpcp_selected_effective_sample_size",
        }
        missing = required - set(arrays.files)
        if missing:
            raise RuntimeError(f"RQ4 arrays are missing: {sorted(missing)}")
        stage_grids = arrays["scpcp_stage_grids"]
        candidate_coverage = arrays["scpcp_candidate_coverage"]
        selected_indices = arrays["scpcp_selected_indices"].astype(int)
        selected_radii = arrays["scpcp_selected_radii"]
        selected_ess = arrays["scpcp_selected_effective_sample_size"]
    records = pd.read_csv(seed_root / "records.csv")
    selected = records[records["method"].eq("SC-PCP")]
    if len(selected) != 1:
        raise RuntimeError("RQ4 requires exactly one SC-PCP record")
    fresh_coverage = parse_curve(selected["per_time_coverage"].iloc[0])
    horizon = len(selected_indices)
    expected_shapes = {
        "stage_grids": stage_grids.shape,
        "candidate_coverage": candidate_coverage.shape,
    }
    if (
        stage_grids.ndim != 2
        or candidate_coverage.shape != stage_grids.shape
        or selected_radii.shape != (horizon,)
        or selected_ess.shape != (horizon,)
        or fresh_coverage.shape != (horizon,)
        or np.any(selected_indices < 0)
        or np.any(selected_indices >= stage_grids.shape[1])
    ):
        raise RuntimeError(f"RQ4 arrays have incompatible shapes: {expected_shapes}")
    selected_estimated_coverage = candidate_coverage[
        np.arange(horizon), selected_indices
    ]
    stages = np.arange(horizon)

    figure, axes = plt.subplots(1, 3, figsize=(20, 6.5), constrained_layout=True)
    axes[0].plot(
        stages,
        selected_estimated_coverage,
        color="#4394f8",
        marker="o",
        linewidth=2.4,
        label="Calibration IW estimate",
    )
    axes[0].plot(
        stages,
        fresh_coverage,
        color="#448c27",
        marker="s",
        linewidth=2.2,
        label="Fresh evaluation",
    )
    axes[0].axhline(TARGET, color="#111827", linestyle=":", linewidth=2.0)
    axes[0].set_ylabel("Per-step coverage")
    axes[0].set_ylim(0.84, 0.96)
    axes[0].legend(loc="best")

    axes[1].plot(stages, selected_radii, color="#4394f8", marker="o", linewidth=2.4)
    axes[1].set_ylabel("Selected radius")
    axes[2].plot(stages, selected_ess, color="#7a3d9d", marker="D", linewidth=2.4)
    axes[2].set_ylabel("Effective sample size")
    for panel, axis in enumerate(axes):
        axis.set_xlabel("Sequential stage, t")
        axis.set_xticks(stages)
        axis.grid(True, color="#D1D5DB", linewidth=0.8, alpha=0.65)
        axis.text(-0.10, 1.03, "abc"[panel], transform=axis.transAxes, fontweight="bold")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def mean_ci(values: object) -> tuple[float, float, float]:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(array):
        return math.nan, math.nan, math.nan
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array)))
    return mean, mean - half, mean + half


def marginal_worst_coverage_ci(
    records: pd.DataFrame,
    *,
    dataset: str,
    method: str,
) -> tuple[float, float, float]:
    """Estimate ``min_t mean_seed coverage_t`` and its seed bootstrap CI."""

    if records.empty:
        return math.nan, math.nan, math.nan
    ordered = records.sort_values("seed") if "seed" in records else records
    curves = [parse_curve(value) for value in ordered["per_time_coverage"]]
    if any(not curve.size for curve in curves):
        raise RuntimeError(
            f"{dataset}/{method} contains an invalid per_time_coverage curve"
        )
    try:
        matrix = np.stack(curves)
    except ValueError as error:
        raise RuntimeError(
            f"{dataset}/{method} per_time_coverage horizons differ"
        ) from error
    if (
        not np.isfinite(matrix).all()
        or np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
    ):
        raise RuntimeError(
            f"{dataset}/{method} per_time_coverage must be finite probabilities"
        )

    estimate = float(matrix.mean(axis=0).min())
    if len(matrix) == 1:
        return estimate, estimate, estimate

    rng = np.random.default_rng(_group_bootstrap_seed(dataset, method))
    bootstrap_minima = np.empty(WORST_COVERAGE_BOOTSTRAP_RESAMPLES)
    for start in range(
        0,
        WORST_COVERAGE_BOOTSTRAP_RESAMPLES,
        WORST_COVERAGE_BOOTSTRAP_BATCH_SIZE,
    ):
        stop = min(
            start + WORST_COVERAGE_BOOTSTRAP_BATCH_SIZE,
            WORST_COVERAGE_BOOTSTRAP_RESAMPLES,
        )
        indices = rng.integers(0, len(matrix), size=(stop - start, len(matrix)))
        bootstrap_minima[start:stop] = matrix[indices].mean(axis=1).min(axis=1)
    low, high = np.quantile(bootstrap_minima, (0.025, 0.975))
    return estimate, float(low), float(high)


def _group_bootstrap_seed(dataset: str, method: str) -> int:
    key = f"{WORST_COVERAGE_BOOTSTRAP_SEED}\0{dataset}\0{method}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def configured_mechanism_seed(synthetic_root: Path) -> int:
    """Read the mechanism seed from the completed synthetic study config."""

    config_path = synthetic_root / "config.yaml"
    if not config_path.is_file():
        raise RuntimeError(f"RQ4 synthetic config is missing: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise RuntimeError("RQ4 synthetic config must be a mapping")
    paper = config.get("paper")
    mechanism_seed = None if not isinstance(paper, dict) else paper.get("mechanism_seed")
    if (
        isinstance(mechanism_seed, bool)
        or not isinstance(mechanism_seed, int)
        or mechanism_seed < 0
    ):
        raise RuntimeError("RQ4 paper.mechanism_seed must be a nonnegative integer")

    configured_seeds = config.get("seeds")
    if isinstance(configured_seeds, dict) and set(configured_seeds) == {"start", "stop"}:
        start = configured_seeds["start"]
        stop = configured_seeds["stop"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, stop)):
            raise RuntimeError("RQ4 configured seed range must contain integers")
        seeds = set(range(start, stop))
    elif isinstance(configured_seeds, (list, tuple)):
        if any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in configured_seeds
        ):
            raise RuntimeError("RQ4 configured seeds must be integers")
        seeds = set(configured_seeds)
    else:
        raise RuntimeError("RQ4 synthetic config has an invalid seeds field")
    if mechanism_seed not in seeds:
        raise RuntimeError(
            f"RQ4 mechanism seed {mechanism_seed} is not in the configured seeds"
        )
    return mechanism_seed


def _efficient_eligible_methods(summary: pd.DataFrame) -> set[tuple[str, str]]:
    selected = set()
    for dataset, group in summary.groupby("dataset"):
        eligible = group[
            group["marginal_worst_coverage"].ge(TARGET)
            & group["selection_rate"].ge(0.95)
            & group["average_normalized_width"].notna()
        ]
        if eligible.empty:
            continue
        row = eligible.loc[eligible["average_normalized_width"].idxmin()]
        selected.add((str(dataset), str(row["method"])))
    return selected


def curve_t_ci(
    matrix: np.ndarray,
    *,
    probability: bool,
) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    if len(matrix) == 1:
        lower = upper = mean
    else:
        half = (
            stats.t.ppf(0.975, len(matrix) - 1)
            * matrix.std(axis=0, ddof=1)
            / math.sqrt(len(matrix))
        )
        lower, upper = mean - half, mean + half
    if probability:
        return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)
    return np.clip(lower, 0.0, None), upper


def wilson_ci(successes: int, total: int) -> tuple[float, float]:
    if total < 1:
        return math.nan, math.nan
    z = stats.norm.ppf(0.975)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def parse_curve(value: object) -> np.ndarray:
    try:
        array = np.asarray(json.loads(str(value)), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError):
        return np.asarray([], dtype=float)
    return array if array.ndim == 1 else np.asarray([], dtype=float)


def ci_text(mean: float, low: float, high: float, *, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "—"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def rate_ci_text(mean: float, low: float, high: float, selected: int, total: int) -> str:
    if not np.isfinite(mean):
        return "—"
    return (
        f"{100 * mean:.1f}% [{100 * low:.1f}, {100 * high:.1f}] "
        f"({selected}/{total})"
    )


def rate_text(mean: float, selected: int, total: int) -> str:
    if not np.isfinite(mean):
        return "—"
    return f"{100 * mean:.1f}% ({selected}/{total})"


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _method_plot_kwargs(method: str) -> dict[str, object]:
    """Use color, line, and marker redundancy for print accessibility."""

    return {
        "color": COLORS[method],
        "linestyle": LINESTYLES[method],
        "marker": MARKERS[method],
        "markerfacecolor": "none" if method == "Standard CP" else COLORS[method],
        "markeredgecolor": COLORS[method],
        "linewidth": 2.6 if method == "SC-PCP" else 1.8,
        "markersize": 9 if method == "SC-PCP" else 6,
        "markevery": 2,
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 17,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def _assert_pdf_only(output: Path) -> None:
    unexpected = [path for path in output.iterdir() if path.is_file() and path.suffix.lower() != ".pdf"]
    if unexpected:
        raise RuntimeError(f"non-PDF paper outputs found: {unexpected}")
    expected = {
        "table_1_synthetic_main.pdf",
        "table_2_clinical_main.pdf",
        "figure_1_per_step_coverage.pdf",
        "figure_2_feedback_stress.pdf",
        "figure_3_committed_prefix_mechanism.pdf",
    }
    observed = {path.name for path in output.glob("*.pdf")}
    if observed != expected:
        raise RuntimeError(f"paper PDF bundle mismatch: expected {sorted(expected)}, found {sorted(observed)}")


if __name__ == "__main__":
    main()
