"""Run the frozen SC-PCP paper experiment from one short command."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from internal.scripts import run_strict_five_dataset_signed_gamma_20260904 as suite


DEFAULT_OUTPUT_ROOT = Path("results/work/main_experiment")
DATASET_CHOICES = (*suite.DATASETS, "all")


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    output_root: Path
    resume: bool


def parse_config(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Run one dataset shard or the complete SC-PCP paper suite."
    )
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    return RunConfig(
        dataset=args.dataset,
        output_root=args.output_root,
        resume=args.resume,
    )


def run_experiment(config: RunConfig) -> None:
    settings = suite.SuiteSettings()
    settings.validate()
    output_root = config.output_root.resolve()

    if config.dataset == "all":
        suite.run_suite(output_root, settings=settings, resume=config.resume)
        print(f"Completed all datasets: {output_root}")
        return

    _prepare_result_root(output_root, settings=settings, resume=config.resume)
    suite._run_dataset(
        output_root,
        dataset=config.dataset,
        settings=settings,
        resume=config.resume,
    )
    _finalize_when_complete(output_root, settings=settings)


def _prepare_result_root(
    output_root: Path,
    *,
    settings: suite.SuiteSettings,
    resume: bool,
) -> None:
    metadata = suite._metadata(settings)
    metadata_path = output_root / "metadata.json"

    if resume:
        if not metadata_path.is_file():
            raise FileNotFoundError("--resume requires an existing compatible run")
        if suite._json_sha256(suite._read_json(metadata_path)) != suite._json_sha256(
            metadata
        ):
            raise RuntimeError("the existing run has different settings or source code")
        return

    if output_root.exists():
        raise FileExistsError(
            f"output root already exists; use --resume or choose another root: {output_root}"
        )
    output_root.mkdir(parents=True)
    suite._write_json(metadata_path, metadata)


def _finalize_when_complete(
    output_root: Path,
    *,
    settings: suite.SuiteSettings,
) -> None:
    remaining = [
        dataset
        for dataset in suite.DATASETS
        if not (output_root / "records" / dataset / "COMPLETE").is_file()
    ]
    if remaining:
        print(f"Dataset complete. Remaining: {', '.join(remaining)}")
        return

    summary = suite._summarize(output_root, settings)
    suite._write_json(output_root / "summary.json", summary)
    suite._write_json(
        output_root / "FINAL_STATUS.json",
        {
            "protocol": suite.PROTOCOL,
            "status": "COMPLETE",
            "primary_gamma": suite.PRIMARY_GAMMA,
            "strict_baselines": True,
            "summary_path": "summary.json",
        },
    )
    (output_root / "COMPLETE").write_text("complete\n")
    print(f"Completed all datasets: {output_root}")


def main() -> None:
    run_experiment(parse_config())


if __name__ == "__main__":
    main()
