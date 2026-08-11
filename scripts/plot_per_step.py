"""Render tables and figures for per-step performative SC-PCP runs.

The script deliberately reads only the artifacts emitted by ``run_per_step.py``:
one ``records.csv`` and ``surfaces.npz`` per seed.  It keeps the two evaluation
tracks separate:

* ``empirical_environment`` is the fresh-deployment result in the frozen
  synthetic/tabular/empirical environment (Track A).
* ``logged_data`` is the logged-data diagnostic (Track B), not a deployment
  coverage claim.

Examples
--------
python scripts/plot_per_step.py \
  --input results/final/main \
  --output results/final/main_figures

python scripts/plot_per_step.py \
  --input results/per_step_studies/20260810T000000Z_feedback \
  --output results/figures/feedback
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


TRACK_A = "empirical_environment"
TRACK_B = "logged_data"
SEED_PATTERN = re.compile(r"seed_(\d+)$")

METHOD_STYLES = {
    "Historical CP": ("#6B7280", "o"),
    "MFCS-style (depth=3)": ("#16A085", "D"),
    "IW-SC-PCP": ("#D97706", "X"),
    "SC-PCP": ("#0F4C81", "*"),
    "MC-oracle SC-PCP (reference)": ("#111827", "^"),
    "ACI-style online": ("#DC2626", "v"),
    "MultiDimSPCI-style online": ("#BE185D", "h"),
    "Repeated recalibration": ("#0891B2", "<"),
    "PRC-MaxTime-style online (grid-adapted)": ("#65A30D", ">"),
}

DISPLAY_LABELS = {
    "Historical CP": "Historical CP",
    "MFCS-style (depth=3)": "MFCS-style (depth 3)",
    "IW-SC-PCP": "IW-SC-PCP",
    "SC-PCP": "SC-PCP (ours)",
    "MC-oracle SC-PCP (reference)": "MC oracle",
    "ACI-style online": "ACI-style (online)",
    "MultiDimSPCI-style online": "MultiDimSPCI-style (online)",
    "Repeated recalibration": "Recalibration (online)",
    "PRC-MaxTime-style online (grid-adapted)": "PRC-MaxTime-style (online, grid-adapted)",
}

MAIN_RESULT_METHODS = tuple(METHOD_STYLES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render per-step SC-PCP tables, per-time coverage, DCov, and study diagnostics"
    )
    parser.add_argument("--input", type=Path, required=True, help="one run directory or a study directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--surface-seed",
        type=int,
        default=None,
        help="seed used for each DCov diagnostic; defaults to the first available seed",
    )
    args = parser.parse_args()

    records = load_per_step_records(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output / "all_records.csv", index=False)

    track_a = records[records["track"] == TRACK_A].copy()
    track_b = records[records["track"] == TRACK_B].copy()
    if not track_a.empty:
        summary = summarize_track_a(track_a)
        summary.to_csv(args.output / "track_a_summary.csv", index=False)
        main_results = build_main_results(summary)
        main_results.to_csv(args.output / "main_results.csv", index=False)
        write_main_results_markdown(main_results, args.output / "main_results.md")
        certification_summary = track_a_certification_summary(summary)
        certification_summary.to_csv(args.output / "track_a_certification_summary.csv", index=False)
        coverage_source = build_per_time_source(track_a)
        coverage_source.to_csv(args.output / "track_a_per_time_coverage.csv", index=False)
        for setting, setting_records in track_a.groupby("setting", sort=False):
            setting_source = coverage_source[coverage_source["setting"] == setting]
            target = _setting_target(setting_records)
            plot_per_time_coverage(setting_source, target, args.output / _figure_stem("track_a_per_time_coverage", setting))
            plot_tradeoff(setting_records, target, args.output / _figure_stem("track_a_tradeoff", setting))
        factorial_summary = summarize_factorial_track_a(track_a)
        if not factorial_summary.empty:
            factorial_summary.to_csv(args.output / "track_a_factorial_summary.csv", index=False)
            plot_factorial_worst_coverage(
                factorial_summary,
                args.output / "track_a_factorial_worst_coverage",
            )
    if not track_b.empty:
        summarize_track_b(track_b).to_csv(args.output / "track_b_summary.csv", index=False)

    surface_diagnostics = summarize_cot_iw_surface_diagnostics(records)
    if not surface_diagnostics.empty:
        surface_diagnostics.to_csv(args.output / "cot_iw_surface_diagnostics.csv", index=False)
        horizon_diagnostics = summarize_cot_iw_horizon_diagnostics(surface_diagnostics)
        horizon_diagnostics.to_csv(args.output / "cot_iw_horizon_summary.csv", index=False)
        plot_cot_iw_horizon_diagnostics(
            horizon_diagnostics,
            args.output / "cot_vs_prefix_iw_horizon_diagnostics",
        )

    for run_root in sorted(find_run_roots(args.input)):
        surface_file = select_surface_file(run_root, args.surface_seed)
        if surface_file is None:
            continue
        arrays = load_surface_arrays(surface_file)
        seed = seed_from_path(surface_file.parent)
        run_records = records[(records["run_root"] == str(run_root)) & (records["seed"] == seed)]
        target = _setting_target(run_records)
        label = f"{run_root.name}_seed_{seed:05d}"
        plot_dcov_heatmaps(arrays, run_records, target, args.output / _figure_stem("dcov_min", label))
        plot_certification_diagnostics(arrays, run_records, target, args.output / _figure_stem("certification", label))

    print(args.output)


def load_per_step_records(input_root: Path) -> pd.DataFrame:
    """Load records while retaining seed and condition provenance."""

    root = input_root.resolve()
    rows: list[pd.DataFrame] = []
    for record_file in sorted(root.rglob("records.csv")):
        seed_dir = record_file.parent
        if not (seed_dir / "COMPLETE").is_file():
            continue
        run_root = seed_dir.parent
        seed = seed_from_path(seed_dir)
        frame = pd.read_csv(record_file)
        provenance = run_provenance(run_root)
        frame["seed"] = seed
        frame["run_root"] = str(run_root)
        frame["study_root"] = str(run_root.parent)
        frame["setting"] = run_root.name
        frame["dataset"] = provenance["dataset"]
        frame["target_coverage"] = provenance["target_coverage"]
        frame["feedback_strength"] = provenance["feedback_strength"]
        frame["policy_tilt"] = provenance["policy_tilt"]
        frame["horizon"] = provenance["horizon"]
        frame["logged_trajectories"] = provenance["logged_trajectories"]
        frame["policy_ratio_cap"] = provenance["policy_ratio_cap"]
        if "selection_estimand" not in frame:
            frame["selection_estimand"] = "per_step"
        if "information_regime" not in frame:
            frame["information_regime"] = "unknown"
        frame["method_family"] = frame["method"].map(method_family)
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"no completed per-step seed artifacts under {input_root}")
    return add_mc_oracle_q_errors(pd.concat(rows, ignore_index=True))


def find_run_roots(input_root: Path) -> Iterable[Path]:
    """Yield directories that own one or more seed artifact directories."""

    roots = {
        file.parent.parent.resolve()
        for file in input_root.resolve().rglob("surfaces.npz")
        if (file.parent / "COMPLETE").is_file()
    }
    return roots


def seed_from_path(seed_dir: Path) -> int:
    match = SEED_PATTERN.fullmatch(seed_dir.name)
    if match is None:
        raise ValueError(f"expected seed_<integer> directory, found {seed_dir}")
    return int(match.group(1))


def target_coverage_for_run(run_root: Path) -> float:
    return float(run_provenance(run_root)["target_coverage"])


def run_provenance(run_root: Path) -> dict[str, float | str]:
    """Read the frozen condition values recorded next to a seed artifact."""

    config_file = run_root / "config.yaml"
    if not config_file.exists():
        return {
            "dataset": "",
            "target_coverage": float("nan"),
            "feedback_strength": float("nan"),
            "policy_tilt": float("nan"),
            "horizon": float("nan"),
            "logged_trajectories": float("nan"),
            "policy_ratio_cap": float("nan"),
        }
    config = yaml.safe_load(config_file.read_text()) or {}
    certification = config.get("certification", {})
    synthetic = config.get("synthetic", {})
    policy = config.get("policy", {})
    samples = config.get("samples", {})
    data = config.get("data", {})
    alpha = certification.get("alpha")
    return {
        "dataset": str(data.get("dataset", "")),
        "target_coverage": float("nan") if alpha is None else 1.0 - float(alpha),
        "feedback_strength": _as_float(synthetic.get("feedback_strength")),
        "policy_tilt": _as_float(policy.get("tilt")),
        "horizon": _as_float(config.get("horizon")),
        "logged_trajectories": _as_float(samples.get("logged")),
        "policy_ratio_cap": _as_float(policy.get("policy_ratio_cap")),
    }


def add_mc_oracle_q_errors(records: pd.DataFrame) -> pd.DataFrame:
    """Attach a scalar per-step selected-radius error to each Track-A record.

    The comparison reference is the explicitly labelled on-policy
    ``MC-oracle SC-PCP (reference)`` row from the same split seed.  A
    nonconstant stagewise controller has no single q and remains undefined.
    """

    result = records.copy()
    result["abs_selected_q_error_to_mc_oracle"] = float("nan")
    required = {"run_root", "seed", "track", "method", "selected_q"}
    if not required.issubset(result):
        return result
    for _, group in result.groupby(["run_root", "seed"], sort=False):
        track_a = group[group["track"].eq(TRACK_A)]
        oracle = track_a[track_a["method"].eq("MC-oracle SC-PCP (reference)")]
        oracle_q = pd.to_numeric(oracle["selected_q"], errors="coerce").dropna()
        if oracle_q.empty:
            continue
        reference_q = float(oracle_q.iloc[0])
        for index, row in track_a.iterrows():
            if not _is_scalar_per_step_record(row):
                continue
            selected_q = _as_float(row.get("selected_q"))
            if np.isfinite(selected_q):
                result.at[index, "abs_selected_q_error_to_mc_oracle"] = abs(selected_q - reference_q)
    return result


def summarize_track_a(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, information_regime, selection_estimand, method), group in records.groupby(
        ["setting", "information_regime", "selection_estimand", "method"], sort=False
    ):
        evaluated = group[group["worst_coverage"].notna()]
        target = _setting_target(group)
        certified = _certificate_success(group)
        fresh_target_met_rate = _all_run_target_met_rate(group, target, "worst_coverage")
        abstention_rate = _status_rate(group, "UNCERTIFIED")
        rows.append(
            {
                "setting": setting,
                "information_regime": information_regime,
                "selection_estimand": selection_estimand,
                "method": method,
                "n_runs": len(group),
                "n_evaluated": len(evaluated),
                "n_abstained": int(round(len(group) * abstention_rate)),
                "abstention_rate": abstention_rate,
                # Retained for older downstream readers.  The new name states
                # what it measures: no radius was available on the fixed grid.
                "uncertified_rate": abstention_rate,
                "formal_certificate_rate": float(certified.mean()),
                "target_coverage": target,
                "target_met_rate": _target_met_rate(evaluated, target),
                "primary_per_step_target_met_rate": _target_met_rate(evaluated, target),
                "fresh_target_met_rate_all_runs": fresh_target_met_rate,
                # Backward-compatible alias. This is an empirical fresh-rollout
                # success rate, not the probability of issuing a certificate.
                "cert_rate": fresh_target_met_rate,
                "cert_rate_among_evaluated": _target_met_rate(evaluated, target),
                "mean_worst_coverage": _mean(evaluated, "worst_coverage"),
                "se_worst_coverage": _standard_error(evaluated, "worst_coverage"),
                "mean_average_coverage": _mean(evaluated, "average_coverage"),
                "se_average_coverage": _standard_error(evaluated, "average_coverage"),
                "mean_pathwise_coverage": _mean(evaluated, "pathwise_coverage"),
                "se_pathwise_coverage": _standard_error(evaluated, "pathwise_coverage"),
                "mean_log_volume": _mean(evaluated, "mean_log_volume"),
                "se_log_volume": _standard_error(evaluated, "mean_log_volume"),
                "mean_median_volume": _mean(evaluated, "median_volume"),
                "se_median_volume": _standard_error(evaluated, "median_volume"),
                "mean_worst_gap": _mean(evaluated, "worst_gap"),
                "se_worst_gap": _standard_error(evaluated, "worst_gap"),
                "mean_clinical_cost": _mean(evaluated, "clinical_cost"),
                "se_clinical_cost": _standard_error(evaluated, "clinical_cost"),
                "mean_selected_q": _mean(evaluated, "selected_q"),
                "mean_abs_selected_q_error_to_mc_oracle": _mean(
                    evaluated, "abs_selected_q_error_to_mc_oracle"
                ),
                "se_abs_selected_q_error_to_mc_oracle": _standard_error(
                    evaluated, "abs_selected_q_error_to_mc_oracle"
                ),
                "mean_target_policy_trajectories": _mean(evaluated, "target_policy_trajectories"),
                "mean_evaluation_trajectories": _mean(evaluated, "oracle_evaluation_trajectories"),
            }
        )
    return pd.DataFrame(rows)


def build_main_results(summary: pd.DataFrame) -> pd.DataFrame:
    """Build the main comparison table without mixing information regimes."""

    if summary.empty:
        return pd.DataFrame()
    primary = summary[
        summary["selection_estimand"].fillna("per_step").eq("per_step")
        & summary["method"].isin(MAIN_RESULT_METHODS)
    ].copy()
    regime_panel = {
        "offline_logged_data": "offline",
        "on_policy_adaptation": "online_with_adaptation_data",
        "on_policy_oracle_reference": "oracle_reference",
    }
    primary.insert(
        1,
        "comparison_panel",
        primary["information_regime"].map(regime_panel).fillna("other"),
    )
    method_order = {method: index for index, method in enumerate(MAIN_RESULT_METHODS)}
    primary["_method_order"] = primary["method"].map(method_order)
    primary = primary.sort_values(["setting", "comparison_panel", "_method_order"])
    columns = [
        "setting",
        "comparison_panel",
        "information_regime",
        "method",
        "n_runs",
        "n_evaluated",
        "n_abstained",
        "target_coverage",
        "fresh_target_met_rate_all_runs",
        "abstention_rate",
        "formal_certificate_rate",
        "mean_worst_coverage",
        "se_worst_coverage",
        "mean_average_coverage",
        "se_average_coverage",
        "mean_worst_gap",
        "se_worst_gap",
        "mean_log_volume",
        "se_log_volume",
        "mean_median_volume",
        "se_median_volume",
        "mean_clinical_cost",
        "se_clinical_cost",
        "mean_target_policy_trajectories",
        "mean_evaluation_trajectories",
    ]
    return primary.loc[:, columns].reset_index(drop=True)


def write_main_results_markdown(results: pd.DataFrame, output: Path) -> None:
    """Write a compact human-readable companion to the numeric main table."""

    header = (
        "| Setting | Panel | Method | Runs | Target met | WorstCov | AvgCov | "
        "MeanLogVolume | Clinical cost | Adaptation trajectories |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for row in results.itertuples(index=False):
        rows.append(
            "| {setting} | {panel} | {method} | {evaluated}/{runs} | {met} | "
            "{worst} | {average} | {volume} | {cost} | {adaptation} |".format(
                setting=row.setting,
                panel=row.comparison_panel,
                method=display_label(row.method),
                evaluated=row.n_evaluated,
                runs=row.n_runs,
                met=_format_rate(row.fresh_target_met_rate_all_runs),
                worst=_format_mean_se(row.mean_worst_coverage, row.se_worst_coverage, row.n_evaluated),
                average=_format_mean_se(row.mean_average_coverage, row.se_average_coverage, row.n_evaluated),
                volume=_format_mean_se(row.mean_log_volume, row.se_log_volume, row.n_evaluated),
                cost=_format_mean_se(row.mean_clinical_cost, row.se_clinical_cost, row.n_evaluated),
                adaptation=f"{row.mean_target_policy_trajectories:.0f}",
            )
        )
    note = (
        "\nValues are mean ± s.e. across evaluated patient-split seeds. "
        "Target-met rates count abstentions as failures. Offline and online rows are "
        "separate information regimes; clinical deployment metrics come from frozen "
        "empirical environments, not real interventions.\n"
    )
    output.write_text(header + "\n".join(rows) + note)


def _format_mean_se(mean: float, standard_error: float, n: int) -> str:
    if not np.isfinite(mean):
        return "—"
    if n < 2 or not np.isfinite(standard_error):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {standard_error:.4f}"


def _format_rate(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.1%}"


def track_a_certification_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return fresh target attainment, abstention, and formal-status rates.

    ``fresh_target_met_rate_all_runs`` is empirical deployment performance,
    not certificate issuance. It counts abstentions as failures. Formal
    theorem status is reported separately by ``formal_certificate_rate``.
    """

    columns = [
        "setting",
        "information_regime",
        "selection_estimand",
        "method",
        "n_runs",
        "n_evaluated",
        "n_abstained",
        "target_coverage",
        "fresh_target_met_rate_all_runs",
        "cert_rate",
        "cert_rate_among_evaluated",
        "abstention_rate",
        "formal_certificate_rate",
    ]
    return summary.loc[:, columns].copy()


def summarize_factorial_track_a(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the beta × eta matrix for the per-step target."""

    required = {"feedback_strength", "policy_tilt", "method_family"}
    if not required.issubset(records):
        return pd.DataFrame()
    primary = records[records["selection_estimand"].fillna("per_step").eq("per_step")].copy()
    primary = primary[primary["feedback_strength"].notna() & primary["policy_tilt"].notna()]
    if primary["feedback_strength"].nunique() < 2 or primary["policy_tilt"].nunique() < 2:
        return pd.DataFrame()
    rows = []
    group_columns = ("method_family", "feedback_strength", "policy_tilt")
    for (method, beta, eta), group in primary.groupby(list(group_columns), sort=False):
        target = _setting_target(group)
        evaluated = group[group["worst_coverage"].notna()]
        abstention_rate = _status_rate(group, "UNCERTIFIED")
        rows.append(
            {
                "method": method,
                "feedback_strength": beta,
                "policy_tilt": eta,
                "n_runs": len(group),
                "n_evaluated": len(evaluated),
                "abstention_rate": abstention_rate,
                "cert_rate": _all_run_target_met_rate(group, target, "worst_coverage"),
                "cert_rate_among_evaluated": _target_met_rate(evaluated, target),
                "mean_worst_coverage": _mean(evaluated, "worst_coverage"),
                "se_worst_coverage": _standard_error(evaluated, "worst_coverage"),
                "mean_log_volume": _mean(evaluated, "mean_log_volume"),
                "mean_clinical_cost": _mean(evaluated, "clinical_cost"),
                "target_coverage": target,
            }
        )
    return pd.DataFrame(rows)


def plot_factorial_worst_coverage(summary: pd.DataFrame, output_stem: Path) -> None:
    """Draw the prespecified beta × eta per-step deployment heatmaps."""

    if summary.empty:
        return
    preferred = ("Historical CP", "IW-SC-PCP", "SC-PCP", "Repeated recalibration")
    methods = [method for method in preferred if method in set(summary["method"])]
    if not methods:
        methods = list(summary["method"].drop_duplicates())[:4]
    columns = 2
    rows = int(np.ceil(len(methods) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(9.4, 3.7 * rows), constrained_layout=True, squeeze=False)
    beta_values = np.sort(summary["feedback_strength"].unique())
    eta_values = np.sort(summary["policy_tilt"].unique())
    image = None
    for axis, method in zip(axes.flat, methods):
        data = summary[summary["method"].eq(method)]
        coverage = data.pivot(index="feedback_strength", columns="policy_tilt", values="mean_worst_coverage").reindex(
            index=beta_values, columns=eta_values
        )
        abstention = data.pivot(index="feedback_strength", columns="policy_tilt", values="abstention_rate").reindex(
            index=beta_values, columns=eta_values
        )
        image = axis.imshow(coverage.to_numpy(dtype=float), origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        for beta_index, eta_index in np.ndindex(coverage.shape):
            value = coverage.iat[beta_index, eta_index]
            abstained = abstention.iat[beta_index, eta_index]
            if np.isfinite(value):
                text_color = "white" if value < 0.58 else "black"
                suffix = "" if not np.isfinite(abstained) or abstained == 0.0 else f"\nA={abstained:.0%}"
                axis.text(eta_index, beta_index, f"{value:.3f}{suffix}", ha="center", va="center", fontsize=7, color=text_color)
            elif np.isfinite(abstained) and abstained > 0.0:
                axis.text(
                    eta_index,
                    beta_index,
                    f"No radius\nA={abstained:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#111827",
                )
        axis.set_title(method)
        axis.set_xticks(np.arange(len(eta_values)), [f"{value:g}" for value in eta_values])
        axis.set_yticks(np.arange(len(beta_values)), [f"{value:g}" for value in beta_values])
        axis.set_xlabel(r"Policy tilt $\eta$")
        axis.set_ylabel(r"Feedback strength $\beta$")
    for axis in axes.flat[len(methods) :]:
        axis.set_visible(False)
    if image is not None:
        colorbar = figure.colorbar(image, ax=list(axes.flat), shrink=0.9)
        colorbar.set_label("Mean fresh-deployment WorstCov")
    figure.suptitle("Per-step performative coverage: beta × eta matrix (A = abstention rate)", fontsize=11)
    save_figure(figure, output_stem)


def summarize_cot_iw_surface_diagnostics(records: pd.DataFrame) -> pd.DataFrame:
    """Extract q-grid summaries without loading full DCov cubes for every seed.

    The ESS and raw-variance summaries first reduce across decision times and
    then take the median over the frozen ``D_COT`` q-grid.  This uses neither a
    selected radius.  Exact CDF-error fields are populated only by the
    finite-MDP oracle experiment.
    """

    provenance_columns = [
        "run_root",
        "study_root",
        "seed",
        "setting",
        "dataset",
        "horizon",
        "feedback_strength",
        "policy_tilt",
        "target_coverage",
        "logged_trajectories",
        "policy_ratio_cap",
    ]
    if not set(provenance_columns).issubset(records):
        return pd.DataFrame()
    rows = []
    q_errors = _selected_q_error_lookup(records)
    provenance = records.loc[:, provenance_columns].drop_duplicates()
    for row in provenance.itertuples(index=False):
        surface_file = Path(row.run_root) / f"seed_{int(row.seed):05d}" / "surfaces.npz"
        if not surface_file.exists():
            continue
        with np.load(surface_file, allow_pickle=False) as arrays:
            if "cot_ess" not in arrays or "iw_ess" not in arrays:
                continue
            rows.append(
                {
                    "run_root": row.run_root,
                    "study_root": row.study_root,
                    "seed": row.seed,
                    "setting": row.setting,
                    "dataset": row.dataset,
                    "horizon": row.horizon,
                    "feedback_strength": row.feedback_strength,
                    "policy_tilt": row.policy_tilt,
                    "target_coverage": row.target_coverage,
                    "logged_trajectories": row.logged_trajectories,
                    "policy_ratio_cap": row.policy_ratio_cap,
                    "q_grid_size": len(arrays["q_grid"]) if "q_grid" in arrays else float("nan"),
                    "cot_grid_median_min_ess": _grid_median_time_extreme(arrays, "cot_ess", "min"),
                    "iw_grid_median_min_ess": _grid_median_time_extreme(arrays, "iw_ess", "min"),
                    "cot_grid_median_max_pre_cap_variance": _grid_median_time_extreme(
                        arrays, "cot_weight_variance_pre_cap", "max"
                    ),
                    "iw_grid_median_max_pre_cap_variance": _grid_median_time_extreme(
                        arrays, "iw_weight_variance_pre_cap", "max"
                    ),
                    "cot_grid_median_max_cdf_error": _grid_median_time_extreme(
                        arrays, "exact_cot_cdf_error", "max"
                    ),
                    "iw_grid_median_max_cdf_error": _grid_median_time_extreme(
                        arrays, "exact_iw_cdf_error", "max"
                    ),
                    "cot_abs_selected_q_error_to_mc_oracle": q_errors.get(
                        (str(row.run_root), int(row.seed), "SC-PCP"), float("nan")
                    ),
                    "iw_abs_selected_q_error_to_mc_oracle": q_errors.get(
                        (str(row.run_root), int(row.seed), "IW-SC-PCP"), float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_cot_iw_horizon_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only controlled studies whose common context varies horizon."""

    if diagnostics.empty or "horizon" not in diagnostics:
        return pd.DataFrame()
    numeric_horizon = pd.to_numeric(diagnostics["horizon"], errors="coerce")
    valid = diagnostics.loc[numeric_horizon.notna()].copy()
    valid["horizon"] = numeric_horizon.loc[valid.index]
    if valid.empty:
        return pd.DataFrame()
    context_columns = [
        "study_root",
        "dataset",
        "feedback_strength",
        "policy_tilt",
        "target_coverage",
        "logged_trajectories",
        "policy_ratio_cap",
    ]
    if not set(context_columns).issubset(valid):
        return pd.DataFrame()
    controlled = []
    for _, context in valid.groupby(context_columns, dropna=False, sort=False):
        if context["horizon"].nunique() > 1:
            controlled.append(context)
    if not controlled:
        return pd.DataFrame()
    valid = pd.concat(controlled, ignore_index=True)
    metrics = (
        "cot_grid_median_min_ess",
        "iw_grid_median_min_ess",
        "cot_grid_median_max_pre_cap_variance",
        "iw_grid_median_max_pre_cap_variance",
        "cot_grid_median_max_cdf_error",
        "iw_grid_median_max_cdf_error",
        "cot_abs_selected_q_error_to_mc_oracle",
        "iw_abs_selected_q_error_to_mc_oracle",
    )
    rows = []
    for context_values, context in valid.groupby(context_columns, dropna=False, sort=False):
        context_result = dict(zip(context_columns, context_values))
        for horizon, group in context.groupby("horizon", sort=True):
            result: dict[str, float | int | str] = {
                **context_result,
                "horizon": float(horizon),
                "n_runs": len(group),
            }
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                result[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
                result[f"{metric}_se"] = _series_standard_error(values)
                result[f"{metric}_n"] = int(len(values))
            rows.append(result)
    return pd.DataFrame(rows)


def plot_cot_iw_horizon_diagnostics(summary: pd.DataFrame, output_stem: Path) -> None:
    """Plot the dedicated horizon-scaling comparison required for COT versus IW."""

    if summary.empty or summary["horizon"].nunique() < 2:
        return
    context_columns = ["study_root", "dataset", "feedback_strength", "policy_tilt", "target_coverage", "logged_trajectories", "policy_ratio_cap"]
    if summary[context_columns].drop_duplicates().shape[0] != 1:
        return
    has_exact_cdf_error = any(
        np.isfinite(summary[f"{prefix}_grid_median_max_cdf_error_mean"]).any()
        for prefix in ("cot", "iw")
    )
    metrics = [
        (
            "grid_median_min_ess",
            "Median over q of min-time ESS",
            "linear",
        ),
        (
            "grid_median_max_pre_cap_variance",
            "Median over q of max-time raw weight variance",
            "symlog",
        ),
    ]
    if has_exact_cdf_error:
        metrics.append(
            (
                "grid_median_max_cdf_error",
                "Median over q of max-time CDF error\n(exact-tabular only)",
                "linear",
            )
        )
    has_selected_q_error = any(
        np.isfinite(summary[f"{prefix}_abs_selected_q_error_to_mc_oracle_mean"]).any()
        for prefix in ("cot", "iw")
    )
    if has_selected_q_error:
        metrics.append(
            (
                "abs_selected_q_error_to_mc_oracle",
                "Mean $|\\hat q-q_{\\rm MC-oracle}|$\n(scalar per-step selections)",
                "linear",
            )
        )
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4.0), constrained_layout=True, squeeze=False)
    horizons = summary["horizon"].to_numpy(dtype=float)
    for axis, (metric, ylabel, scale) in zip(axes.flat, metrics):
        for prefix, label, color, marker in (
            ("cot", "COT", "#0F4C81", "o"),
            ("iw", "Prefix-IW", "#D97706", "X"),
        ):
            mean = summary[f"{prefix}_{metric}_mean"].to_numpy(dtype=float)
            error = summary[f"{prefix}_{metric}_se"].to_numpy(dtype=float)
            finite = np.isfinite(mean)
            if finite.any():
                axis.errorbar(
                    horizons[finite],
                    mean[finite],
                    yerr=np.nan_to_num(error[finite], nan=0.0),
                    color=color,
                    marker=marker,
                    linewidth=2.0,
                    capsize=2.5,
                    label=label,
                )
        axis.set_xlabel("Horizon T")
        axis.set_ylabel(ylabel)
        axis.set_xticks(horizons)
        if scale == "symlog":
            axis.set_yscale("symlog", linthresh=1e-3)
        elif metric.endswith("cdf_error"):
            axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        axis.legend(fontsize=8)
    figure.suptitle("COT versus prefix-IW: frozen-grid horizon diagnostics", fontsize=11)
    save_figure(figure, output_stem)


def summarize_track_b(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, information_regime, selection_estimand, method), group in records.groupby(
        ["setting", "information_regime", "selection_estimand", "method"], sort=False
    ):
        rows.append(
            {
                "setting": setting,
                "information_regime": information_regime,
                "selection_estimand": selection_estimand,
                "method": method,
                "n_runs": len(group),
                "uncertified_rate": _status_rate(group, "UNCERTIFIED"),
                "formal_certificate_rate": float(_certificate_success(group).mean()),
                "mean_estimated_min_coverage": _mean(group, "estimated_min_coverage"),
                "se_estimated_min_coverage": _standard_error(group, "estimated_min_coverage"),
                "mean_lower_bound_min": _mean(group, "lower_bound_min"),
                "se_lower_bound_min": _standard_error(group, "lower_bound_min"),
                "mean_ess": _mean(group, "mean_ess"),
                "mean_minimum_ess": _mean(group, "minimum_ess"),
                "median_policy_kl": _mean(group, "median_policy_kl"),
                "maximum_policy_ratio": _mean(group, "maximum_policy_ratio"),
                "mean_logged_descriptive_mean_log_volume": _mean(
                    group, "logged_descriptive_mean_log_volume"
                ),
                "mean_logged_descriptive_median_volume": _mean(
                    group, "logged_descriptive_median_volume"
                ),
                "mean_logged_descriptive_clinical_cost": _mean(
                    group, "logged_descriptive_clinical_cost"
                ),
                "mean_logged_descriptive_clinical_utility": _mean(
                    group, "logged_descriptive_clinical_utility"
                ),
                "mean_logged_state_model_estimated_clinical_cost": _mean(
                    group, "logged_state_model_estimated_clinical_cost"
                ),
                "mean_logged_state_model_estimated_clinical_utility": _mean(
                    group, "logged_state_model_estimated_clinical_utility"
                ),
                "evaluation_scope": _first_text(group, "evaluation_scope"),
                "prediction_set_metric_scope": _first_text(
                    group, "prediction_set_metric_scope"
                ),
                "clinical_value_metric_scope": _first_text(
                    group, "clinical_value_metric_scope"
                ),
                "clinical_utility_definition": _first_text(
                    group, "clinical_utility_definition"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_per_time_source(records: pd.DataFrame) -> pd.DataFrame:
    """Build the primary per-step curve source."""

    rows = []
    for row in primary_per_step_records(records).itertuples(index=False):
        values = parse_curve(getattr(row, "per_time_coverage", "[]"))
        for time, coverage in enumerate(values, start=1):
            rows.append(
                {
                    "setting": row.setting,
                    "seed": row.seed,
                    "method": row.method,
                    "time": time,
                    "coverage": coverage,
                }
            )
    return pd.DataFrame(rows, columns=["setting", "seed", "method", "time", "coverage"])


def parse_curve(value: object) -> np.ndarray:
    try:
        result = np.asarray(json.loads(value), dtype=float)
    except (TypeError, json.JSONDecodeError):
        return np.empty(0, dtype=float)
    return result[np.isfinite(result)]


def plot_per_time_coverage(source: pd.DataFrame, target: float, output_stem: Path) -> None:
    if source.empty:
        return
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for method, group in source.groupby("method", sort=False):
        aggregate = group.groupby("time", sort=True)["coverage"].agg(["mean", "count", "std"])
        time = aggregate.index.to_numpy(dtype=float)
        mean = aggregate["mean"].to_numpy()
        count = aggregate["count"].to_numpy()
        standard_error = aggregate["std"].fillna(0.0).to_numpy() / np.sqrt(count)
        color, marker = METHOD_STYLES.get(method, ("#4B5563", "o"))
        axis.plot(
            time,
            mean,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.5,
            label=display_label(method),
        )
        if np.any(standard_error > 0.0):
            axis.fill_between(time, np.clip(mean - 1.96 * standard_error, 0.0, 1.0), np.clip(mean + 1.96 * standard_error, 0.0, 1.0), color=color, alpha=0.12)
    if np.isfinite(target):
        axis.axhline(target, color="#111827", linestyle="--", linewidth=1.2, label=f"Target ({target:.2f})")
    axis.set_xlabel("Decision time")
    axis.set_ylabel("Fresh-deployment per-step coverage")
    axis.set_ylim(*coverage_axis_limits(source["coverage"], target))
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=6.4,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    axis.text(
        0.99,
        0.02,
        f"Shading: 95% normal-approx. CI; {seed_count_note(source)}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="#4B5563",
    )
    save_figure(figure, output_stem)


def plot_tradeoff(records: pd.DataFrame, target: float, output_stem: Path) -> None:
    """Plot the primary per-step efficiency comparison."""

    evaluated = primary_per_step_records(records)
    evaluated = evaluated[evaluated["worst_coverage"].notna()]
    if evaluated.empty:
        return
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for method, group in evaluated.groupby("method", sort=False):
        x = group["mean_log_volume"].mean()
        y = group["worst_coverage"].mean()
        x_error = _standard_error(group, "mean_log_volume")
        y_error = _standard_error(group, "worst_coverage")
        color, marker = METHOD_STYLES.get(method, ("#4B5563", "o"))
        online = _first_text(group, "information_regime") == "on_policy_adaptation"
        axis.errorbar(
            x,
            y,
            xerr=x_error,
            yerr=y_error,
            color=color,
            marker=marker,
            markerfacecolor="white" if online else color,
            markeredgewidth=1.2,
            markersize=8 if method == "SC-PCP" else 6,
            capsize=2.5,
            linewidth=1.0,
            label=display_label(method),
            zorder=4 if method == "SC-PCP" else 3,
        )
    if np.isfinite(target):
        axis.axhline(target, color="#111827", linestyle="--", linewidth=1.2, label=f"Target ({target:.2f})")
    axis.set_xlabel("Mean log prediction-set volume (smaller is better)")
    axis.set_ylabel("Worst per-step fresh-deployment coverage")
    axis.set_ylim(*coverage_axis_limits(evaluated["worst_coverage"], target))
    axis.grid(color="#E5E7EB", linewidth=0.7)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=6.4,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    axis.text(
        0.99,
        0.02,
        f"Points: seed mean; error bars: ±1 s.e.; {seed_count_note(evaluated)}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="#4B5563",
    )
    save_figure(figure, output_stem)

def select_surface_file(run_root: Path, requested_seed: int | None) -> Path | None:
    candidates = sorted(
        path
        for path in run_root.glob("seed_*/surfaces.npz")
        if (path.parent / "COMPLETE").is_file()
    )
    if requested_seed is None:
        return candidates[0] if candidates else None
    target = run_root / f"seed_{requested_seed:05d}" / "surfaces.npz"
    return target if target.exists() and (target.parent / "COMPLETE").is_file() else None


def load_surface_arrays(surface_file: Path) -> dict[str, np.ndarray]:
    with np.load(surface_file) as arrays:
        return {name: arrays[name] for name in arrays.files}


def plot_dcov_heatmaps(
    arrays: dict[str, np.ndarray],
    records: pd.DataFrame,
    target: float,
    output_stem: Path,
) -> None:
    """Render DCov geometry and scalar per-step selection locations."""

    required = ("q_grid", "cot_dcov", "iw_dcov", "oracle_dcov")
    if not all(name in arrays for name in required):
        return
    q_grid = arrays["q_grid"]
    selection_points = per_step_scalar_selection_points(records, q_grid)
    names = (("cot_dcov", "COT estimate"), ("iw_dcov", "Prefix-IW estimate"), ("oracle_dcov", "Fresh-environment oracle"))
    figure, axes = plt.subplots(1, len(names), figsize=(11.5, 4.25), constrained_layout=True, sharey=True)
    image = None
    for panel_index, (axis, (key, title)) in enumerate(zip(axes, names)):
        minimum_coverage = np.min(arrays[key], axis=1)
        image = axis.pcolormesh(q_grid, q_grid, minimum_coverage.T, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        axis.plot(q_grid, q_grid, color="white", linewidth=1.2, linestyle="--")
        if np.isfinite(target) and minimum_coverage.min() <= target <= minimum_coverage.max():
            axis.contour(q_grid, q_grid, minimum_coverage.T, levels=[target], colors=["#EF4444"], linewidths=1.5)
        for label, radius, color, marker in selection_points:
            axis.scatter(
                radius,
                radius,
                color=color,
                marker=marker,
                s=58,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
                label=label if panel_index == 0 else "_nolegend_",
            )
        axis.set_title(title)
        axis.set_xlabel(r"Deployment radius $q_D$")
    axes[0].set_ylabel(r"Measurement radius $q_M$")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.94)
    colorbar.set_label(r"$\min_t\,\widehat{\mathrm{DCov}}_t(q_D,q_M)$")
    if selection_points:
        axes[0].legend(loc="upper left", fontsize=6.5, framealpha=0.82)
    figure.text(
        0.5,
        0.005,
        "Markers are scalar per-step selections; stagewise controllers are not projected here.",
        ha="center",
        fontsize=7,
    )
    save_figure(figure, output_stem)


def plot_certification_diagnostics(
    arrays: dict[str, np.ndarray],
    records: pd.DataFrame,
    target: float,
    output_stem: Path,
) -> None:
    if "q_grid" not in arrays:
        return
    q_grid = arrays["q_grid"]
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 3.8), constrained_layout=True)
    coverage_axis, ess_axis, variance_axis = axes
    lines = [
        ("cot_diagonal", "COT diagonal", "#0F4C81"),
        ("iw_diagonal", "Prefix-IW diagonal", "#D97706"),
        ("exact_cot_diagonal", "Exact tabular COT", "#1D4ED8"),
    ]
    for key, label, color in lines:
        if key in arrays:
            coverage_axis.plot(q_grid, arrays[key].min(axis=1), color=color, linewidth=2.0, label=label)
    for key, label, color in (
        ("cot_lower_bounds", "SC-PCP deployment LCB", "#0F4C81"),
        ("iw_lower_bounds", "IW-SC-PCP deployment LCB", "#D97706"),
        ("exact_cot_lower_bounds", "Exact COT simultaneous LCB", "#1D4ED8"),
        (
            "learned_cot_oracle_l1_lower_bounds",
            "Learned COT + exact oracle-L1 LCB",
            "#2563EB",
        ),
    ):
        if key in arrays:
            coverage_axis.plot(q_grid, arrays[key].min(axis=1), color=color, linestyle="--", linewidth=1.4, label=label)
    if "oracle_dcov" in arrays:
        oracle_diagonal = grid_diagonal(arrays["oracle_dcov"])
        coverage_axis.plot(q_grid, oracle_diagonal.min(axis=1), color="#111827", linewidth=1.5, label="Fresh-environment oracle")
    for _, radius, color, _ in per_step_scalar_selection_points(records, q_grid):
        coverage_axis.axvline(radius, color=color, linewidth=1.0, alpha=0.65)
    if np.isfinite(target):
        coverage_axis.axhline(target, color="#111827", linestyle=":", linewidth=1.2, label="Target")
    coverage_axis.set_xlabel("Radius q")
    coverage_axis.set_ylabel("Minimum over decision times")
    coverage_axis.set_ylim(0.0, 1.02)
    coverage_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    coverage_axis.legend(fontsize=6.5, ncol=2)

    for key, label, color in (("cot_ess", "COT", "#0F4C81"), ("iw_ess", "Prefix-IW", "#D97706"), ("exact_cot_ess", "Exact tabular COT", "#1D4ED8")):
        if key in arrays:
            ess_axis.plot(q_grid, arrays[key].min(axis=1), color=color, linewidth=2.0, label=label)
    ess_axis.set_xlabel("Radius q")
    ess_axis.set_ylabel("Minimum per-time ESS")
    ess_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ess_axis.legend(fontsize=7)

    for key, label, color in (
        ("cot_weight_variance_pre_cap", "COT pre-cap variance", "#0F4C81"),
        ("iw_weight_variance_pre_cap", "Prefix-IW pre-cap variance", "#D97706"),
    ):
        if key in arrays:
            variance_axis.plot(q_grid, arrays[key].max(axis=1), color=color, linewidth=2.0, label=label)
    for key, label, color in (
        ("cot_cap_hit_rate", "COT cap-hit rate", "#0F4C81"),
        ("iw_cap_hit_rate", "Prefix-IW cap-hit rate", "#D97706"),
    ):
        if key in arrays:
            variance_axis.plot(q_grid, arrays[key].max(axis=1), color=color, linestyle="--", linewidth=1.2, label=label)
    variance_axis.set_xlabel("Radius q")
    variance_axis.set_ylabel("Max per-time variance / cap-hit rate")
    variance_axis.set_yscale("symlog", linthresh=1e-3)
    variance_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    variance_axis.legend(fontsize=6.5)
    save_figure(figure, output_stem)


def grid_diagonal(surface: np.ndarray) -> np.ndarray:
    indices = np.arange(surface.shape[0])
    return surface[indices, :, indices]


def method_family(method: object) -> str:
    """Return the stable method name used by the final experiment."""

    return str(method)


def per_step_scalar_selection_points(
    records: pd.DataFrame,
    q_grid: np.ndarray,
) -> list[tuple[str, float, str, str]]:
    """Return valid scalar selections in the per-step geometry only.

    A stagewise controller has no single deployment coordinate in the DCov
    surface, so nonconstant ``q_by_time`` rows are intentionally omitted.
    """

    if records.empty or "track" not in records or "method" not in records:
        return []
    q_grid = np.asarray(q_grid, dtype=float)
    if q_grid.ndim != 1 or not np.isfinite(q_grid).any():
        return []
    candidates = records[records["track"].eq(TRACK_A)]
    if "selection_estimand" in candidates:
        candidates = candidates[candidates["selection_estimand"].fillna("per_step").eq("per_step")]
    points: dict[str, tuple[str, float, str, str]] = {}
    lower, upper = float(np.nanmin(q_grid)), float(np.nanmax(q_grid))
    tolerance = max(1e-8, 1e-6 * (upper - lower))
    for row in candidates.itertuples(index=False):
        method = str(getattr(row, "method"))
        radius = _as_float(getattr(row, "selected_q", None))
        q_by_time = parse_curve(getattr(row, "q_by_time", ""))
        if q_by_time.size:
            if not np.allclose(q_by_time, q_by_time[0], rtol=1e-6, atol=tolerance):
                continue
            if not np.isfinite(radius):
                radius = float(q_by_time[0])
        if not np.isfinite(radius) or radius < lower - tolerance or radius > upper + tolerance:
            continue
        family = method_family(method)
        color, marker = METHOD_STYLES.get(method, METHOD_STYLES.get(family, ("#4B5563", "o")))
        points.setdefault(family, (family, float(radius), color, marker))
    return list(points.values())


def _is_scalar_per_step_record(row: pd.Series) -> bool:
    if str(row.get("selection_estimand", "per_step")) != "per_step":
        return False
    q_by_time = parse_curve(row.get("q_by_time", ""))
    return not q_by_time.size or np.allclose(q_by_time, q_by_time[0], rtol=1e-6, atol=1e-8)


def primary_per_step_records(records: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose selection target is the manuscript's primary estimand."""

    selected = records.copy()
    if "selection_estimand" in selected:
        selected = selected[selected["selection_estimand"].fillna("per_step").eq("per_step")]
    return selected


def display_label(method: object) -> str:
    return DISPLAY_LABELS.get(str(method), str(method))


def seed_count_note(records: pd.DataFrame) -> str:
    """Describe the per-method split-seed count without assuming balance."""

    if "seed" not in records or "method" not in records:
        return "split seeds"
    counts = records.groupby("method", sort=False)["seed"].nunique()
    if counts.empty:
        return "split seeds"
    lower, upper = int(counts.min()), int(counts.max())
    if lower == upper:
        return f"n={lower} split seed{'s' if lower != 1 else ''}"
    return f"n={lower}–{upper} split seeds per method"


def coverage_axis_limits(values: object, target: float, *, minimum_span: float = 0.10) -> tuple[float, float]:
    """Zoom coverage panels while retaining a visible, honest target margin."""

    finite = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if np.isfinite(target):
        finite = np.append(finite, target)
    if not finite.size:
        return 0.0, 1.02
    lower_data, upper_data = float(finite.min()), float(finite.max())
    span = max(minimum_span, upper_data - lower_data)
    center = 0.5 * (lower_data + upper_data)
    lower = max(0.0, center - 0.5 * span - 0.01)
    upper = min(1.005, center + 0.5 * span + 0.01)
    if upper - lower < minimum_span:
        if lower == 0.0:
            upper = min(1.005, minimum_span)
        else:
            lower = max(0.0, upper - minimum_span)
    return lower, upper


def _selected_q_error_lookup(records: pd.DataFrame) -> dict[tuple[str, int, str], float]:
    required = {
        "run_root",
        "seed",
        "track",
        "method",
        "abs_selected_q_error_to_mc_oracle",
    }
    if not required.issubset(records):
        return {}
    selected = records[records["track"].eq(TRACK_A)].copy()
    selected = selected[selected.apply(_is_scalar_per_step_record, axis=1)]
    selected["method_family"] = selected["method"].map(method_family)
    lookup = {}
    for (run_root, seed, family), group in selected.groupby(
        ["run_root", "seed", "method_family"], sort=False
    ):
        values = pd.to_numeric(group["abs_selected_q_error_to_mc_oracle"], errors="coerce").dropna()
        if not values.empty:
            lookup[(str(run_root), int(seed), str(family))] = float(values.iloc[0])
    return lookup


def _grid_median_time_extreme(
    arrays: object,
    key: str,
    extreme: str,
) -> float:
    if key not in arrays:
        return float("nan")
    values = np.asarray(arrays[key], dtype=float)
    if values.ndim != 2:
        return float("nan")
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    if extreme == "min":
        per_q = np.where(finite, values, np.inf).min(axis=1)
    elif extreme == "max":
        per_q = np.where(finite, values, -np.inf).max(axis=1)
    else:
        raise ValueError("extreme must be min or max")
    per_q[~finite.any(axis=1)] = np.nan
    return float(np.nanmedian(per_q))


def _series_standard_error(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else float("nan")
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _certificate_success(records: pd.DataFrame) -> pd.Series:
    formal = records.get("certificate_formal", pd.Series(False, index=records.index)).map(_as_bool)
    status = records.get("selection_status", pd.Series("", index=records.index)).fillna("").eq("CERTIFIED")
    return formal & status


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _status_rate(records: pd.DataFrame, prefix: str) -> float:
    status = records.get("selection_status", pd.Series("", index=records.index)).fillna("").astype(str)
    return float(status.str.startswith(prefix).mean())


def _target_met_rate(records: pd.DataFrame, target: float) -> float:
    if records.empty or not np.isfinite(target):
        return float("nan")
    return float((records["worst_coverage"] >= target - 1e-7).mean())


def _all_run_target_met_rate(records: pd.DataFrame, target: float, column: str) -> float:
    if records.empty or not np.isfinite(target) or column not in records:
        return float("nan")
    values = pd.to_numeric(records[column], errors="coerce")
    return float((values >= target - 1e-7).fillna(False).mean())


def _mean(records: pd.DataFrame, column: str) -> float:
    if column not in records:
        return float("nan")
    return float(records[column].mean())


def _first_text(records: pd.DataFrame, column: str) -> str:
    if column not in records:
        return ""
    values = records[column].dropna().astype(str)
    values = values[values.ne("")]
    return "" if values.empty else str(values.iloc[0])


def _standard_error(records: pd.DataFrame, column: str) -> float:
    if column not in records:
        return float("nan")
    values = records[column].dropna()
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _setting_target(records: pd.DataFrame) -> float:
    values = records.get("target_coverage", pd.Series(dtype=float)).dropna()
    return float(values.iloc[0]) if not values.empty else float("nan")


def _figure_stem(prefix: str, label: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return Path(prefix + "_" + safe)


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    # Setting labels legitimately contain decimal values such as ``eta_0.25``.
    # Appending the render extension preserves that value; ``Path.with_suffix``
    # would otherwise mistake ``.25`` for an existing file suffix.
    figure.savefig(Path(f"{output_stem}.svg"), bbox_inches="tight")
    figure.savefig(Path(f"{output_stem}.pdf"), bbox_inches="tight")
    figure.savefig(Path(f"{output_stem}.png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
