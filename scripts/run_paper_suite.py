"""Run the prespecified four-RQ paper experiment with one global GPU queue."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_per_step import run_config
from scpcp.artifacts import experiment_tree_sha256
from scpcp.config import ExperimentConfig
from scpcp.device import resolve_devices


CONFIGS = {
    "synthetic": ROOT / "configs" / "per_step_synthetic_tail_shift.yaml",
    "mimic_iv": ROOT / "configs" / "per_step_mimic_iv.yaml",
    "mimic_cxr": ROOT / "configs" / "per_step_mimic_cxr.yaml",
    "eicu": ROOT / "configs" / "per_step_eicu.yaml",
    "inspire": ROOT / "configs" / "per_step_inspire.yaml",
}
FEEDBACK_LEVELS = (0.0, 0.5, 1.0, 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete SC-PCP paper suite")
    parser.add_argument(
        "--sections",
        default="rq1,rq3",
        help="comma-separated rq1,rq3; RQ2 and RQ4 reuse those artifacts",
    )
    parser.add_argument("--datasets", default=",".join(CONFIGS), help="RQ1 datasets")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--output-root", type=Path, default=Path("results/work/paper_final"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an exact suite and validate every completed setting",
    )
    args = parser.parse_args()

    sections = _parse_choices(args.sections, {"rq1", "rq3"})
    datasets = _parse_choices(args.datasets, set(CONFIGS))
    devices = resolve_devices(args.devices)
    root = args.output_root
    mechanism_seed = ExperimentConfig.from_yaml(CONFIGS["synthetic"]).paper.mechanism_seed
    manifest = {
        "protocol": "committed_prefix_marginal_scpcp",
        "method": "direct_committed_prefix_uncapped_importance_weighting",
        "experiment_tree_sha256": experiment_tree_sha256(),
        "sections": sections,
        "datasets": datasets,
        "feedback_levels": FEEDBACK_LEVELS,
        "devices": devices,
        "reuse": {
            "rq2_mimic_iv": "rq1/mimic_iv",
            "rq2_synthetic_strong": "rq3/beta_2",
            "rq3_beta_1": "rq1/synthetic",
            "rq4": (
                f"rq1/synthetic/seed_{mechanism_seed:05d} "
                "committed-prefix surfaces"
            ),
        },
    }
    _prepare_suite_root(root, manifest, resume=args.resume)

    if "rq1" in sections:
        for dataset in datasets:
            config = ExperimentConfig.from_yaml(CONFIGS[dataset]).with_overrides(
                devices=devices,
                output_dir=root / "rq1" / dataset,
            )
            run_config(
                config,
                config.output_dir,
                workers_per_device=_workers_per_device(dataset),
                resume=args.resume and config.output_dir.exists(),
            )

    if "rq3" in sections:
        base = ExperimentConfig.from_yaml(CONFIGS["synthetic"])
        if not (root / "rq1" / "synthetic" / "COMPLETE").is_file():
            config = base.with_overrides(
                devices=devices,
                output_dir=root / "rq1" / "synthetic",
            )
            run_config(
                config,
                config.output_dir,
                workers_per_device=_workers_per_device("synthetic"),
                resume=args.resume and config.output_dir.exists(),
            )
        for beta in FEEDBACK_LEVELS:
            if beta == 1.0:
                continue
            label = f"beta_{beta:g}"
            config = replace(
                base,
                synthetic=replace(base.synthetic, feedback_strength=beta),
                paper=replace(base.paper, save_mechanism_diagonal=False),
            ).with_overrides(
                devices=devices,
                output_dir=root / "rq3" / label,
            )
            run_config(
                config,
                config.output_dir,
                workers_per_device=_workers_per_device("synthetic"),
                resume=args.resume and config.output_dir.exists(),
            )

    (root / "COMPLETE").write_text("complete\n")
    print(root)


def _prepare_suite_root(
    root: Path,
    manifest: dict[str, object],
    *,
    resume: bool,
) -> None:
    """Create a new suite or validate an exact resumable suite manifest."""

    manifest_path = root / "suite_manifest.json"
    normalized_manifest = json.loads(json.dumps(manifest))
    if resume:
        if not root.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(f"resumable paper suite is missing: {root}")
        stored = json.loads(manifest_path.read_text())
        if stored != normalized_manifest:
            raise RuntimeError("paper-suite manifest differs from the active experiment")
        return
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"paper-suite output already contains files: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "rq1").mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(normalized_manifest, indent=2) + "\n")


def _parse_choices(value: str, allowed: set[str]) -> tuple[str, ...]:
    choices = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(choices) - allowed
    if not choices or unknown:
        raise ValueError(f"invalid choices: {sorted(unknown)}")
    return choices


def _workers_per_device(dataset: str) -> int:
    if dataset == "mimic_cxr":
        return 1
    if dataset == "synthetic":
        return 4
    return 2


if __name__ == "__main__":
    main()
