"""Render the held-out controlled performative benchmark.

The experiment artifacts are immutable inputs.  This renderer recomputes all
reported quantities from per-seed JSON files, keeps development and
confirmation separate, and writes editable working exports plus one PDF-only
paper output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROTOCOL = "controlled_performative_prefix_benchmark_v1"
GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
METHODS = ("Standard CP", "SC-PCP")
TARGET = 0.90
HORIZON = 12
LATE_STAGES = tuple(range(4, 12))
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 5_129_041
STANDARD_COLOR = "#7e8c9c"
SCPCP_COLOR = "#4394f8"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--confirm", type=Path, required=True)
    parser.add_argument("--work-output", type=Path, required=True)
    parser.add_argument("--paper-output", type=Path, required=True)
    args = parser.parse_args()

    development_meta, development_rows = load_study(
        args.development.resolve(), expected_role="development20"
    )
    confirm_meta, confirm_rows = load_study(
        args.confirm.resolve(), expected_role="confirm"
    )
    if development_meta["source_tree_sha256"] != confirm_meta["source_tree_sha256"]:
        raise RuntimeError("development and confirmation used different source trees")
    if set(development_meta["seeds"]) & set(confirm_meta["seeds"]):
        raise RuntimeError("development and confirmation seed banks overlap")

    if args.work_output.exists() or args.paper_output.exists():
        raise FileExistsError("both output directories must be fresh")
    args.work_output.mkdir(parents=True)
    args.paper_output.mkdir(parents=True)

    development = analyze(development_rows)
    confirmation = analyze(confirm_rows)
    source_rows = make_source_rows(development, confirmation)
    write_csv(args.work_output / "controlled_prefix_source_data.csv", source_rows)
    write_json(
        args.work_output / "controlled_prefix_analysis.json",
        {
            "protocol": PROTOCOL,
            "generated_from_source_tree_sha256": confirm_meta["source_tree_sha256"],
            "development_seeds": development_meta["seeds"],
            "confirm_seeds": confirm_meta["seeds"],
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "unit": "complete seed-stage vector",
                "wsc_interval": "studentized centered max-t simultaneous stagewise band",
                "paired_interval": "percentile paired-seed bootstrap",
            },
            "development": development,
            "confirmation": confirmation,
        },
    )
    render_figure(
        confirmation,
        paper_path=args.paper_output / "figure_controlled_performative_benchmark.pdf",
        work_stem=args.work_output / "figure_controlled_performative_benchmark",
    )
    write_qa_notes(args.work_output / "figure_qa.md", confirm_meta)
    observed = sorted(path.suffix for path in args.paper_output.iterdir())
    if observed != [".pdf"]:
        raise RuntimeError("paper output must contain exactly one PDF")
    print(args.paper_output.resolve())


def load_study(root: Path, *, expected_role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not (root / "COMPLETE").is_file():
        raise RuntimeError(f"study is incomplete: {root}")
    metadata = read_json(root / "metadata.json")
    if metadata.get("protocol") != PROTOCOL or metadata.get("role") != expected_role:
        raise RuntimeError(f"wrong protocol or role: {root}")
    if tuple(float(value) for value in metadata.get("gammas", ())) != GAMMAS:
        raise RuntimeError("gamma grid does not match the frozen protocol")
    if tuple(metadata.get("methods", ())) != METHODS:
        raise RuntimeError("method set does not match the frozen protocol")
    if metadata.get("late_stages_zero_based") != list(LATE_STAGES):
        raise RuntimeError("late-stage definition does not match the frozen protocol")
    source_hash = metadata.get("source_tree_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise RuntimeError("invalid experiment source hash")

    seeds = tuple(int(value) for value in metadata.get("seeds", ()))
    if len(seeds) != 20 or len(set(seeds)) != 20:
        raise RuntimeError("the formal study requires exactly 20 unique seeds")
    expected_paths = {root / f"seed_{seed:05d}.json" for seed in seeds}
    observed_paths = set(root.glob("seed_*.json"))
    if observed_paths != expected_paths:
        raise RuntimeError("seed artifact set does not match metadata")

    rows: list[dict[str, Any]] = []
    for seed, path in zip(seeds, sorted(expected_paths)):
        payload = read_json(path)
        if payload.get("seed") != seed or len(payload.get("rows", ())) != len(GAMMAS):
            raise RuntimeError(f"malformed seed artifact: {path}")
        for row, gamma in zip(payload["rows"], GAMMAS):
            validate_row(row, seed=seed, gamma=gamma)
            rows.append(row)
    return metadata, rows


def validate_row(row: dict[str, Any], *, seed: int, gamma: float) -> None:
    if row.get("seed") != seed or float(row.get("gamma")) != gamma:
        raise RuntimeError("seed/gamma mismatch in benchmark row")
    if set(row.get("methods", {})) != set(METHODS):
        raise RuntimeError("method mismatch in benchmark row")
    for method in METHODS:
        values = row["methods"][method]
        for name in (
            "radii",
            "source_coverage",
            "target_coverage",
            "coverage_gap",
            "source_q90",
            "target_q90",
            "q90_relative_gap",
            "target_normalized_width",
            "prefix_ess_fraction",
            "maximum_normalized_weight_share",
            "policy_tv_on_source_states",
        ):
            vector = np.asarray(values.get(name), dtype=np.float64)
            if vector.shape != (HORIZON,) or not np.isfinite(vector).all():
                raise RuntimeError(f"invalid {method}/{name} vector")
        source = np.asarray(values["source_coverage"], dtype=np.float64)
        target = np.asarray(values["target_coverage"], dtype=np.float64)
        gap = np.asarray(values["coverage_gap"], dtype=np.float64)
        if not np.allclose(gap, target - source, atol=1e-12, rtol=0.0):
            raise RuntimeError("stored coverage gap is inconsistent")


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seed_order = tuple(sorted({int(row["seed"]) for row in rows}))
    if len(seed_order) != 20:
        raise RuntimeError("analysis requires exactly 20 complete seeds")
    seed_index = {seed: index for index, seed in enumerate(seed_order)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = rng.integers(
        0, len(seed_order), size=(BOOTSTRAP_RESAMPLES, len(seed_order))
    )
    output: dict[str, Any] = {}
    for gamma in GAMMAS:
        selected = [row for row in rows if float(row["gamma"]) == gamma]
        selected.sort(key=lambda row: seed_index[int(row["seed"])])
        if len(selected) != len(seed_order):
            raise RuntimeError(f"incomplete gamma cell: {gamma}")
        methods: dict[str, Any] = {}
        for method in METHODS:
            coverage = stack(selected, method, "target_coverage")
            source_coverage = stack(selected, method, "source_coverage")
            radii = stack(selected, method, "radii")
            target_q90 = stack(selected, method, "target_q90")
            widths = stack(selected, method, "target_normalized_width")
            stage_mean = coverage.mean(axis=0)
            lower, upper = simultaneous_stage_band(coverage, bootstrap)
            actual_width = widths.mean(axis=1)
            response_width = (widths / radii * target_q90).mean(axis=1)
            response_ratio = geometric_mean(actual_width / response_width)
            response_draws = np.exp(
                np.log(actual_width[bootstrap] / response_width[bootstrap]).mean(axis=1)
            )
            q_response_ratio = geometric_mean(radii.ravel() / target_q90.ravel())
            q_response_draws = np.exp(
                np.log(radii[bootstrap] / target_q90[bootstrap]).mean(axis=(1, 2))
            )
            methods[method] = {
                "target_wsc": float(stage_mean.min()),
                "target_mean_coverage": float(stage_mean.mean()),
                "target_worst_stage_zero_based": int(stage_mean.argmin()),
                "target_stage_coverage": stage_mean.tolist(),
                "target_wsc_simultaneous_band": [
                    float(lower.min()),
                    float(upper.min()),
                ],
                "source_wsc": float(source_coverage.mean(axis=0).min()),
                "mean_target_width": float(actual_width.mean()),
                "width_to_fixed_policy_q90_response": response_ratio,
                "width_to_fixed_policy_q90_response_ci95": percentile_interval(
                    response_draws
                ),
                "radius_to_target_q90_geometric_ratio": q_response_ratio,
                "radius_to_target_q90_geometric_ratio_ci95": percentile_interval(
                    q_response_draws
                ),
                "minimum_reference_prefix_ess_fraction": float(
                    min(
                        min(row["methods"][method]["prefix_ess_fraction"])
                        for row in selected
                    )
                ),
                "maximum_reference_weight_share": float(
                    max(
                        max(row["methods"][method]["maximum_normalized_weight_share"])
                        for row in selected
                    )
                ),
            }

        standard_gap = stack(selected, "Standard CP", "coverage_gap")
        standard_q90 = stack(selected, "Standard CP", "q90_relative_gap")
        late_gap = standard_gap[:, LATE_STAGES].mean(axis=1)
        late_q90 = standard_q90[:, LATE_STAGES].mean(axis=1)
        standard_width = stack(selected, "Standard CP", "target_normalized_width").mean(axis=1)
        scpcp_width = stack(selected, "SC-PCP", "target_normalized_width").mean(axis=1)
        width_log_ratio = np.log(scpcp_width / standard_width)
        width_draws = np.exp(width_log_ratio[bootstrap].mean(axis=1))
        output[f"{gamma:g}"] = {
            "gamma": gamma,
            "n_seeds": len(seed_order),
            "methods": methods,
            "standard_late_coverage_gap": float(late_gap.mean()),
            "standard_late_coverage_gap_ci95": percentile_interval(
                late_gap[bootstrap].mean(axis=1)
            ),
            "standard_late_q90_relative_gap": float(late_q90.mean()),
            "standard_late_q90_relative_gap_ci95": percentile_interval(
                late_q90[bootstrap].mean(axis=1)
            ),
            "standard_late_policy_tv": float(
                stack(selected, "Standard CP", "policy_tv_on_source_states")[:, LATE_STAGES].mean()
            ),
            "scpcp_to_standard_width_ratio": geometric_mean(
                scpcp_width / standard_width
            ),
            "scpcp_to_standard_width_ratio_ci95": percentile_interval(width_draws),
            "selection_minimum_ess_fraction": float(
                min(row["selection_minimum_ess_fraction"] for row in selected)
            ),
            "selection_minimum_candidate_ess_fraction": float(
                min(row["selection_minimum_candidate_ess_fraction"] for row in selected)
            ),
            "selection_endpoint_count": int(
                sum(bool(row["selection_selected_endpoint"]) for row in selected)
            ),
            "donor_kernel_ess_fraction_min": float(
                min(
                    row["methods"][method]["donor_kernel_ess_fraction_min"]
                    for row in selected
                    for method in METHODS
                )
            ),
            "donor_probability_max": float(
                max(
                    row["methods"][method]["donor_probability_max"]
                    for row in selected
                    for method in METHODS
                )
            ),
        }
    return output


def simultaneous_stage_band(
    values: np.ndarray, bootstrap: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    standard_error = values.std(axis=0, ddof=1) / math.sqrt(len(values))
    if np.any(standard_error <= 0.0):
        raise RuntimeError("simultaneous band requires positive stagewise variation")
    resampled = values[bootstrap].mean(axis=1)
    lower_statistic = np.max((mean[None, :] - resampled) / standard_error[None, :], axis=1)
    upper_statistic = np.max((resampled - mean[None, :]) / standard_error[None, :], axis=1)
    lower = mean - np.quantile(lower_statistic, 0.95) * standard_error
    upper = mean + np.quantile(upper_statistic, 0.95) * standard_error
    return lower, upper


def make_source_rows(
    development: dict[str, Any], confirmation: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for role, analysis in (("development20", development), ("confirm20", confirmation)):
        for gamma in GAMMAS:
            cell = analysis[f"{gamma:g}"]
            row: dict[str, Any] = {
                "role": role,
                "gamma": gamma,
                "n_seeds": cell["n_seeds"],
                "standard_late_coverage_gap": cell["standard_late_coverage_gap"],
                "standard_late_coverage_gap_ci_low": cell["standard_late_coverage_gap_ci95"][0],
                "standard_late_coverage_gap_ci_high": cell["standard_late_coverage_gap_ci95"][1],
                "standard_late_q90_relative_gap": cell["standard_late_q90_relative_gap"],
                "standard_late_q90_relative_gap_ci_low": cell["standard_late_q90_relative_gap_ci95"][0],
                "standard_late_q90_relative_gap_ci_high": cell["standard_late_q90_relative_gap_ci95"][1],
                "standard_late_policy_tv": cell["standard_late_policy_tv"],
                "scpcp_to_standard_width_ratio": cell["scpcp_to_standard_width_ratio"],
                "scpcp_to_standard_width_ratio_ci_low": cell["scpcp_to_standard_width_ratio_ci95"][0],
                "scpcp_to_standard_width_ratio_ci_high": cell["scpcp_to_standard_width_ratio_ci95"][1],
                "selection_minimum_ess_fraction": cell["selection_minimum_ess_fraction"],
                "selection_minimum_candidate_ess_fraction": cell["selection_minimum_candidate_ess_fraction"],
                "selection_endpoint_count": cell["selection_endpoint_count"],
            }
            for method, prefix in (("Standard CP", "standard"), ("SC-PCP", "scpcp")):
                method_cell = cell["methods"][method]
                row[f"{prefix}_target_wsc"] = method_cell["target_wsc"]
                row[f"{prefix}_target_wsc_band_low"] = method_cell[
                    "target_wsc_simultaneous_band"
                ][0]
                row[f"{prefix}_target_wsc_band_high"] = method_cell[
                    "target_wsc_simultaneous_band"
                ][1]
                row[f"{prefix}_target_mean_coverage"] = method_cell[
                    "target_mean_coverage"
                ]
                row[f"{prefix}_mean_target_width"] = method_cell[
                    "mean_target_width"
                ]
                row[f"{prefix}_width_to_q90_response"] = method_cell[
                    "width_to_fixed_policy_q90_response"
                ]
                row[f"{prefix}_radius_to_target_q90"] = method_cell[
                    "radius_to_target_q90_geometric_ratio"
                ]
            rows.append(row)
    return rows


def render_figure(
    confirmation: dict[str, Any], *, paper_path: Path, work_stem: Path
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    x = np.asarray(GAMMAS)
    figure, axes = plt.subplots(2, 2, figsize=(7.20, 4.75), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    offsets = {"Standard CP": -0.07, "SC-PCP": 0.07}
    for method, color, marker in (
        ("Standard CP", STANDARD_COLOR, "o"),
        ("SC-PCP", SCPCP_COLOR, "s"),
    ):
        point = np.asarray(
            [confirmation[f"{gamma:g}"]["methods"][method]["target_wsc"] for gamma in x]
        )
        band = np.asarray(
            [
                confirmation[f"{gamma:g}"]["methods"][method][
                    "target_wsc_simultaneous_band"
                ]
                for gamma in x
            ]
        )
        ax_a.errorbar(
            x + offsets[method],
            point,
            yerr=np.vstack((point - band[:, 0], band[:, 1] - point)),
            color=color,
            marker=marker,
            markersize=4.2,
            linewidth=1.35,
            capsize=2.2,
            label=method,
        )
    ax_a.axhline(TARGET, color="#333333", linestyle=(0, (3, 2)), linewidth=0.9)
    ax_a.set_ylabel("Target marginal WSC")
    ax_a.set_ylim(0.845, 0.918)
    ax_a.legend(loc="lower right", fontsize=6.4)

    gap = np.asarray(
        [confirmation[f"{gamma:g}"]["standard_late_coverage_gap"] for gamma in x]
    ) * 100.0
    gap_ci = np.asarray(
        [confirmation[f"{gamma:g}"]["standard_late_coverage_gap_ci95"] for gamma in x]
    ) * 100.0
    ax_b.errorbar(
        x,
        gap,
        yerr=np.vstack((gap - gap_ci[:, 0], gap_ci[:, 1] - gap)),
        color=STANDARD_COLOR,
        marker="o",
        markersize=4.2,
        linewidth=1.35,
        capsize=2.2,
    )
    ax_b.axhline(0.0, color="#333333", linestyle=(0, (3, 2)), linewidth=0.9)
    ax_b.set_ylabel("Standard CP coverage drift (pp)")

    q90 = np.asarray(
        [confirmation[f"{gamma:g}"]["standard_late_q90_relative_gap"] for gamma in x]
    ) * 100.0
    q90_ci = np.asarray(
        [confirmation[f"{gamma:g}"]["standard_late_q90_relative_gap_ci95"] for gamma in x]
    ) * 100.0
    ax_c.errorbar(
        x,
        q90,
        yerr=np.vstack((q90 - q90_ci[:, 0], q90_ci[:, 1] - q90)),
        color="#9A4D8E",
        marker="D",
        markersize=4.0,
        linewidth=1.35,
        capsize=2.2,
    )
    ax_c.axhline(0.0, color="#333333", linestyle=(0, (3, 2)), linewidth=0.9)
    ax_c.set_ylabel("Target score Q90 shift (%)")
    ax_c.set_xlabel(r"Signed transition alignment $\gamma$")

    ratio = np.asarray(
        [confirmation[f"{gamma:g}"]["scpcp_to_standard_width_ratio"] for gamma in x]
    )
    ratio_ci = np.asarray(
        [confirmation[f"{gamma:g}"]["scpcp_to_standard_width_ratio_ci95"] for gamma in x]
    )
    ax_d.errorbar(
        x,
        ratio,
        yerr=np.vstack((ratio - ratio_ci[:, 0], ratio_ci[:, 1] - ratio)),
        color=SCPCP_COLOR,
        marker="s",
        markersize=4.2,
        linewidth=1.35,
        capsize=2.2,
    )
    ax_d.axhline(1.0, color="#333333", linestyle=(0, (3, 2)), linewidth=0.9)
    ax_d.set_ylabel("SC-PCP / Standard width")
    ax_d.set_xlabel(r"Signed transition alignment $\gamma$")

    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        axis.text(
            -0.16,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )
        axis.set_xticks(x)
        axis.tick_params(width=0.7, length=3)
        axis.axvline(-2.0, color="#FFD700", alpha=0.18, linewidth=6, zorder=0)

    figure.savefig(paper_path, bbox_inches="tight")
    figure.savefig(work_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(work_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    figure.savefig(work_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_qa_notes(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            (
                "# Controlled benchmark figure QA",
                "",
                "- Core conclusion: held-out signed policy-mediated score shifts make Standard CP under- or over-cover, while SC-PCP remains near the 0.90 target and adjusts width in both directions.",
                "- Archetype: quantitative grid; panel a is the hero coverage result.",
                "- Backend: Python/matplotlib only.",
                "- Final size: 7.20 x 4.75 inches (double-column).",
                "- n: 20 complete confirmation seed-stage vectors per gamma; T=12.",
                "- WSC: min_t mean_seed coverage. Error bars are simultaneous 95% stagewise max-t bands; the displayed interval is [min_t L_t, min_t U_t].",
                "- Panels b-d: paired complete-seed 10,000-resample percentile bootstrap intervals.",
                "- Gold vertical marker: development-selected primary mechanism strength gamma=-2; no outcome-dependent retuning occurred before confirmation.",
                "- Gamma=-4 is an overlap stress endpoint; minimum selected calibration ESS/n is reported in source data and must be disclosed.",
                f"- Experiment source hash: {metadata['source_tree_sha256']}",
                "- Editable SVG and PDF text retained; PDF uses TrueType embedding.",
                "- No image manipulation; all panels are direct numerical summaries of immutable per-seed JSON artifacts.",
                "",
            )
        )
    )


def stack(rows: list[dict[str, Any]], method: str, field: str) -> np.ndarray:
    value = np.asarray([row["methods"][method][field] for row in rows], dtype=np.float64)
    if value.shape != (len(rows), HORIZON) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid stacked field: {method}/{field}")
    return value


def percentile_interval(draws: np.ndarray) -> list[float]:
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def geometric_mean(values: np.ndarray) -> float:
    if np.any(values <= 0.0) or not np.isfinite(values).all():
        raise RuntimeError("geometric mean requires finite positive values")
    return float(np.exp(np.log(values).mean()))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
