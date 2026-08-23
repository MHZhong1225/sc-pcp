"""Validate, summarize, and render the paired A/C/D/E decomposition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scpcp.conservatism_decomposition import LAYERS


COLORS = {
    "A": "#7e8c9c",
    "C": "#448c27",
    "D": "#02bec4",
    "E": "#aa3831",
}
DISPLAY_NAMES = {
    "A": "Sequential reference",
    "C": "Profiled oracle",
    "D": "COT point",
    "E": "SC-PCP LCB",
}


@dataclass(frozen=True)
class DecompositionArrays:
    seeds: np.ndarray
    widths: np.ndarray
    coverage: np.ndarray
    target: float
    rollouts: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the fresh A/C/D/E decomposition")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=271_828)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_pdf.suffix.lower() != ".pdf":
        raise ValueError("output must be a PDF")
    if args.bootstrap_resamples < 1_000:
        raise ValueError("bootstrap-resamples must be at least 1000")
    if args.output_pdf.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output_pdf}")
    arrays = load_decomposition_arrays(args.input_dir)
    summary = summarize_decomposition(
        arrays,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    render_decomposition_pdf(arrays, summary, args.output_pdf)
    print(json.dumps(summary, indent=2))
    print(args.output_pdf.resolve())


def load_decomposition_arrays(input_dir: Path) -> DecompositionArrays:
    """Load only a complete, internally consistent four-layer study."""

    for name in ("COMPLETE", "config.yaml", "study_status.json"):
        if not (input_dir / name).is_file():
            raise RuntimeError(f"decomposition input is missing {name}")
    config = yaml.safe_load((input_dir / "config.yaml").read_text())
    status = json.loads((input_dir / "study_status.json").read_text())
    if not isinstance(config, dict) or not isinstance(status, dict):
        raise RuntimeError("decomposition config/status must be mappings")
    seeds = tuple(int(seed) for seed in config.get("seeds", ()))
    if (
        not seeds
        or status.get("status") != "complete"
        or status.get("expected_seeds") != list(seeds)
        or status.get("completed_seeds") != list(seeds)
        or status.get("error") is not None
    ):
        raise RuntimeError("decomposition study is not exactly complete")

    records_by_seed = []
    horizon = int(config["horizon"])
    target = 1.0 - float(config["certification"]["alpha"])
    rollout_counts: set[int] = set()
    evaluation_seeds: set[int] = set()
    for seed in seeds:
        seed_dir = input_dir / f"seed_{seed:05d}"
        if not (seed_dir / "COMPLETE").is_file():
            raise RuntimeError(f"decomposition seed {seed} is incomplete")
        records = pd.read_csv(seed_dir / "records.csv")
        if len(records) != len(LAYERS) or tuple(records["layer"]) != LAYERS:
            raise RuntimeError(f"decomposition seed {seed} does not contain ordered A/C/D/E")
        if not records["seed"].eq(seed).all():
            raise RuntimeError(f"decomposition seed {seed} record IDs differ")
        if not np.allclose(records["target_coverage"], target, atol=0.0, rtol=0.0):
            raise RuntimeError(f"decomposition seed {seed} target differs")
        rollout_counts.update(int(value) for value in records["oracle_evaluation_trajectories"])
        seed_evaluations = {int(value) for value in records["evaluation_seed"]}
        if len(seed_evaluations) != 1:
            raise RuntimeError(f"decomposition seed {seed} does not use one common CRN stream")
        evaluation_seed = next(iter(seed_evaluations))
        if evaluation_seed in evaluation_seeds:
            raise RuntimeError("fresh evaluation seeds are not unique across split seeds")
        evaluation_seeds.add(evaluation_seed)
        per_time = np.stack(
            [np.asarray(json.loads(value), dtype=np.float64) for value in records["per_time_coverage"]]
        )
        if per_time.shape != (len(LAYERS), horizon):
            raise RuntimeError(f"decomposition seed {seed} coverage has the wrong shape")
        if not np.isfinite(per_time).all() or np.any((per_time < 0.0) | (per_time > 1.0)):
            raise RuntimeError(f"decomposition seed {seed} coverage is invalid")
        widths = records["average_normalized_width"].to_numpy(dtype=np.float64)
        if not np.isfinite(widths).all() or np.any(widths <= 0.0):
            raise RuntimeError(f"decomposition seed {seed} widths are invalid")
        if not np.allclose(records["worst_coverage"], per_time.min(axis=1), atol=1e-7):
            raise RuntimeError(f"decomposition seed {seed} worst coverage disagrees")
        if not np.allclose(records["average_coverage"], per_time.mean(axis=1), atol=1e-7):
            raise RuntimeError(f"decomposition seed {seed} average coverage disagrees")
        records_by_seed.append((widths, per_time))
    if len(rollout_counts) != 1:
        raise RuntimeError("decomposition rows use different rollout counts")

    return DecompositionArrays(
        seeds=np.asarray(seeds, dtype=np.int64),
        widths=np.stack([item[0] for item in records_by_seed]),
        coverage=np.stack([item[1] for item in records_by_seed]),
        target=target,
        rollouts=next(iter(rollout_counts)),
    )


def summarize_decomposition(
    arrays: DecompositionArrays,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Return paired point estimates and percentile intervals."""

    n, layer_count = arrays.widths.shape
    if layer_count != len(LAYERS) or arrays.coverage.shape[:2] != (n, len(LAYERS)):
        raise ValueError("decomposition arrays must have shape [N,4] and [N,4,T]")
    if n < 2:
        raise ValueError("at least two paired seeds are required")
    generator = np.random.default_rng(seed)
    resamples = generator.integers(0, n, size=(n_resamples, n))

    log_width = np.log(arrays.widths)
    width_boot = np.exp(log_width[resamples].mean(axis=1))
    width_point = np.exp(log_width.mean(axis=0))

    adjacent_pairs = ((0, 1), (1, 2), (2, 3), (0, 3))
    ratio_names = ("C/A", "D/C", "E/D", "E/A")
    ratio_point: dict[str, float] = {}
    ratio_ci: dict[str, list[float]] = {}
    for name, (denominator, numerator) in zip(ratio_names, adjacent_pairs, strict=True):
        paired_log_ratio = log_width[:, numerator] - log_width[:, denominator]
        ratio_point[name] = float(np.exp(paired_log_ratio.mean()))
        boot = np.exp(paired_log_ratio[resamples].mean(axis=1))
        ratio_ci[name] = _percentile_interval(boot)

    seedwise_worst = arrays.coverage.min(axis=2)
    average_coverage = arrays.coverage.mean(axis=2)
    seedwise_worst_boot = seedwise_worst[resamples].mean(axis=1)
    average_coverage_boot = average_coverage[resamples].mean(axis=1)
    pooled_stage_boot = arrays.coverage[resamples].mean(axis=1).min(axis=2)

    log_components = np.column_stack(
        (
            log_width[:, 1] - log_width[:, 0],
            log_width[:, 2] - log_width[:, 1],
            log_width[:, 3] - log_width[:, 2],
        )
    )
    component_means = log_components.mean(axis=0)
    total = float(component_means.sum())
    shares = np.full(3, np.nan) if abs(total) < 1e-15 else component_means / total

    return {
        "paired_seeds": int(n),
        "rollouts_per_schedule": arrays.rollouts,
        "target": arrays.target,
        "geometric_mean_width": {
            layer: float(width_point[index]) for index, layer in enumerate(LAYERS)
        },
        "geometric_mean_width_95ci": {
            layer: _percentile_interval(width_boot[:, index])
            for index, layer in enumerate(LAYERS)
        },
        "paired_width_ratio": ratio_point,
        "paired_width_ratio_95ci": ratio_ci,
        "mean_seedwise_worst_coverage": {
            layer: float(seedwise_worst[:, index].mean())
            for index, layer in enumerate(LAYERS)
        },
        "mean_seedwise_worst_coverage_95ci": {
            layer: _percentile_interval(seedwise_worst_boot[:, index])
            for index, layer in enumerate(LAYERS)
        },
        "pooled_worst_stage_coverage": {
            layer: float(arrays.coverage[:, index].mean(axis=0).min())
            for index, layer in enumerate(LAYERS)
        },
        "pooled_worst_stage_coverage_95ci": {
            layer: _percentile_interval(pooled_stage_boot[:, index])
            for index, layer in enumerate(LAYERS)
        },
        "mean_coverage": {
            layer: float(average_coverage[:, index].mean())
            for index, layer in enumerate(LAYERS)
        },
        "mean_coverage_95ci": {
            layer: _percentile_interval(average_coverage_boot[:, index])
            for index, layer in enumerate(LAYERS)
        },
        "fresh_target_met_count": {
            layer: int((seedwise_worst[:, index] >= arrays.target).sum())
            for index, layer in enumerate(LAYERS)
        },
        "mean_log_width_overhead": {
            "profile_C_minus_A": float(component_means[0]),
            "cot_D_minus_C": float(component_means[1]),
            "guard_E_minus_D": float(component_means[2]),
            "total_E_minus_A": total,
        },
        "log_overhead_share": {
            "profile": float(shares[0]),
            "cot_point": float(shares[1]),
            "guard": float(shares[2]),
        },
    }


def render_decomposition_pdf(
    arrays: DecompositionArrays,
    summary: dict[str, Any],
    output_pdf: Path,
) -> None:
    """Render one four-panel PDF; no PNG/SVG side products are created."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 17,
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
        }
    )
    x = np.arange(len(LAYERS))
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12.5), constrained_layout=False)
    ax_width, ax_coverage, ax_overhead, ax_stage = axes.flat

    width = np.array([summary["geometric_mean_width"][layer] for layer in LAYERS])
    width_ci = np.array([summary["geometric_mean_width_95ci"][layer] for layer in LAYERS])
    width_error = np.vstack((width - width_ci[:, 0], width_ci[:, 1] - width))
    ax_width.errorbar(
        x,
        width,
        yerr=width_error,
        fmt="none",
        ecolor="#333333",
        elinewidth=1.5,
        capsize=4,
        zorder=3,
    )
    ax_width.bar(x, width, color=[COLORS[layer] for layer in LAYERS], width=0.68)
    ax_width.set_xticks(x, LAYERS)
    ax_width.set_ylabel("Average normalized width")
    ax_width.set_title("a  Common-fresh width")
    ax_width.grid(axis="y", alpha=0.25)

    seed_worst = np.array(
        [summary["mean_seedwise_worst_coverage"][layer] for layer in LAYERS]
    )
    pooled_worst = np.array(
        [summary["pooled_worst_stage_coverage"][layer] for layer in LAYERS]
    )
    mean_coverage = np.array([summary["mean_coverage"][layer] for layer in LAYERS])
    ax_coverage.axhline(arrays.target, color="#111111", linestyle=":", linewidth=2, label="Target")
    ax_coverage.plot(x, seed_worst, "o-", color="#aa3831", linewidth=2, label=r"$E[\min_t C_t]$")
    ax_coverage.plot(x, pooled_worst, "s-", color="#448c27", linewidth=2, label=r"$\min_t E[C_t]$")
    ax_coverage.plot(x, mean_coverage, "^-", color="#4394f8", linewidth=2, label="Mean coverage")
    ax_coverage.set_xticks(x, LAYERS)
    ax_coverage.set_ylabel("Coverage")
    ax_coverage.set_ylim(min(0.885, seed_worst.min() - 0.003), max(0.935, mean_coverage.max() + 0.003))
    ax_coverage.set_title("b  Coverage semantics")
    ax_coverage.legend(frameon=False, loc="best")
    ax_coverage.grid(alpha=0.25)

    component_keys = ("profile_C_minus_A", "cot_D_minus_C", "guard_E_minus_D")
    component_labels = ("Profile\nC−A", "COT point\nD−C", "LCB guard\nE−D")
    component_colors = (COLORS["C"], COLORS["D"], COLORS["E"])
    components = 100.0 * np.array(
        [summary["mean_log_width_overhead"][key] for key in component_keys]
    )
    ax_overhead.axhline(0.0, color="#333333", linewidth=1)
    ax_overhead.bar(np.arange(3), components, color=component_colors, width=0.65)
    ax_overhead.set_xticks(np.arange(3), component_labels)
    ax_overhead.set_ylabel("Mean log-width overhead (%)")
    ax_overhead.set_title("c  Telescoping overhead")
    ax_overhead.grid(axis="y", alpha=0.25)
    for index, value in enumerate(components):
        ax_overhead.text(index, value + np.sign(value or 1) * 0.08, f"{value:.2f}", ha="center", va="bottom")

    stage = np.arange(1, arrays.coverage.shape[2] + 1)
    for index, layer in enumerate(LAYERS):
        mean = arrays.coverage[:, index].mean(axis=0)
        se = arrays.coverage[:, index].std(axis=0, ddof=1) / math.sqrt(len(arrays.seeds))
        ax_stage.plot(stage, mean, color=COLORS[layer], linewidth=2, marker="o", markersize=4, label=layer)
        ax_stage.fill_between(stage, mean - 1.984 * se, mean + 1.984 * se, color=COLORS[layer], alpha=0.10)
    ax_stage.axhline(arrays.target, color="#111111", linestyle=":", linewidth=2)
    ax_stage.set_xlabel("Decision step")
    ax_stage.set_ylabel("Mean coverage")
    ax_stage.set_xticks(stage)
    ax_stage.set_title("d  Per-step target-policy coverage")
    ax_stage.legend(frameon=False, ncol=4, loc="best")
    ax_stage.grid(alpha=0.25)

    ratio = summary["paired_width_ratio"]
    counts = summary["fresh_target_met_count"]
    caption = (
        f"{summary['paired_seeds']} paired seeds; {arrays.rollouts:,} fresh rollouts per frozen schedule.  "
        f"Width ratios: C/A={ratio['C/A']:.4f}, D/C={ratio['D/C']:.4f}, "
        f"E/D={ratio['E/D']:.4f}, E/A={ratio['E/A']:.4f}.  "
        f"Fresh target-met seeds A/C/D/E: {counts['A']}/{counts['C']}/{counts['D']}/{counts['E']}."
    )
    fig.suptitle("SC-PCP conservatism decomposition", y=0.985, fontsize=23)
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=14)
    fig.tight_layout(rect=(0.025, 0.055, 0.985, 0.955), h_pad=2.0, w_pad=2.0)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _percentile_interval(values: np.ndarray) -> list[float]:
    lower, upper = np.quantile(values, (0.025, 0.975))
    return [float(lower), float(upper)]


if __name__ == "__main__":
    main()
