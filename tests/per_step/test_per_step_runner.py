from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

import scpcp.artifacts as artifacts
from scpcp.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]
SOURCE_HASH = "a" * 64
METHODS = ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")


def _load_runner():
    path = ROOT / "scripts" / "run_per_step.py"
    spec = importlib.util.spec_from_file_location("run_per_step_resume_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=seed,
        device="cpu",
        records=[{"method": method} for method in METHODS],
        surfaces={
            "scpcp_stage_grids": torch.ones(2, 3),
            "scpcp_candidate_coverage": torch.ones(2, 3),
            "scpcp_selected_indices": torch.ones(2, dtype=torch.long),
        },
        diagnostics={},
    )


def _freeze_source_hash(monkeypatch: pytest.MonkeyPatch, runner: object) -> None:
    monkeypatch.setattr(artifacts, "source_tree_sha256", lambda: SOURCE_HASH)
    monkeypatch.setattr(runner, "source_tree_sha256", lambda: SOURCE_HASH)


def _initialize_study(runner: object, config: ExperimentConfig) -> None:
    runner.write_study_metadata(
        config.output_dir,
        config,
        execution={"workers_per_device": 1},
    )


def test_resume_skips_valid_seed_and_fills_only_missing_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    _freeze_source_hash(monkeypatch, runner)
    output = tmp_path / "study"
    config = ExperimentConfig(
        devices=("cpu",),
        seeds=(0, 1),
        output_dir=output,
    )
    _initialize_study(runner, config)
    runner.write_seed_result(_result(0), output, config)
    pending_calls: list[tuple[int, ...]] = []

    def fill_pending(config, output_dir, seeds, *, workers_per_device):
        pending_calls.append(seeds)
        for seed in seeds:
            runner.write_seed_result(_result(seed), output_dir, config)

    monkeypatch.setattr(runner, "_run_pending_seeds", fill_pending)

    runner.run_config(config, output, workers_per_device=1, resume=True)

    assert pending_calls == [(1,)]
    assert (output / "seed_00000" / "COMPLETE").is_file()
    assert (output / "seed_00001" / "COMPLETE").is_file()
    assert (output / "COMPLETE").is_file()


def test_resume_rejects_partial_seed_without_overwriting_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    _freeze_source_hash(monkeypatch, runner)
    output = tmp_path / "study"
    config = ExperimentConfig(devices=("cpu",), seeds=(0,), output_dir=output)
    _initialize_study(runner, config)
    partial = output / "seed_00000"
    partial.mkdir()
    (partial / "COMPLETE").write_text('{"seed": 0, "status": "complete"}\n')
    called = False

    def unexpected_run(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runner, "_run_pending_seeds", unexpected_run)

    with pytest.raises(RuntimeError, match="artifact is partial"):
        runner.run_config(config, output, workers_per_device=1, resume=True)

    assert called is False
    assert tuple(path.name for path in partial.iterdir()) == ("COMPLETE",)


def test_resume_rejects_source_execution_and_unexpected_seed_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    _freeze_source_hash(monkeypatch, runner)
    output = tmp_path / "study"
    config = ExperimentConfig(devices=("cpu",), seeds=(0,), output_dir=output)
    _initialize_study(runner, config)
    metadata_path = output / "study_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_tree_sha256"] = "b" * 64
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="source hash differs"):
        runner.run_config(config, output, workers_per_device=1, resume=True)

    metadata["source_tree_sha256"] = SOURCE_HASH
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="execution settings differ"):
        runner.run_config(config, output, workers_per_device=2, resume=True)

    unexpected = output / "seed_00099"
    unexpected.mkdir()
    with pytest.raises(RuntimeError, match="unexpected seed 99"):
        runner.run_config(config, output, workers_per_device=1, resume=True)


def test_fresh_mode_still_refuses_an_existing_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    _freeze_source_hash(monkeypatch, runner)
    output = tmp_path / "study"
    output.mkdir()
    config = ExperimentConfig(devices=("cpu",), seeds=(0,), output_dir=output)

    with pytest.raises(FileExistsError, match="fresh study output already exists"):
        runner.run_config(config, output, workers_per_device=1)
