from __future__ import annotations

from concurrent.futures import Future
import io
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scpcp.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "phase0_oracle.yaml"
EXPECTED_PRIMARY_ROWS = {
    ("standard", "Current Profiled Oracle"),
    ("standard", "Greedy Sequential Oracle"),
    ("tail_shift", "Current Profiled Oracle"),
    ("tail_shift", "Greedy Sequential Oracle"),
}


def _spawn_identity(value: int) -> tuple[int, int]:
    return os.getpid(), value


def _spawn_failure(_value: int) -> None:
    raise RuntimeError("spawn probe failure")


def _load_runner():
    path = ROOT / "scripts" / "run_phase0_oracle.py"
    spec = importlib.util.spec_from_file_location("run_phase0_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(seed: int, device: str) -> SimpleNamespace:
    records = [
        {"scenario": scenario, "method": method, "seed": seed}
        for scenario, method in sorted(EXPECTED_PRIMARY_ROWS)
    ]
    return SimpleNamespace(
        seed=seed,
        device=device,
        records=records,
        surfaces={"profiled_scale_grid": torch.tensor([0.5, 1.0])},
        diagnostics={},
    )


class _InlineExecutor:
    instances: list["_InlineExecutor"] = []

    def __init__(self, *, max_workers: int, mp_context: object) -> None:
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.submissions: list[tuple[object, tuple[object, ...]]] = []
        self.instances.append(self)

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def submit(self, function: object, *args: object) -> Future[str]:
        self.submissions.append((function, args))
        future: Future[str] = Future()
        try:
            future.set_result(function(*args))  # type: ignore[operator]
        except BaseException as error:
            future.set_exception(error)
        return future


def _run_fake_study(
    runner: object,
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    *,
    seeds: tuple[int, ...] = (0, 1),
    candidate_chunk_size: int = 7,
) -> tuple[ExperimentConfig, list[tuple[int, str, int]]]:
    calls: list[tuple[int, str, int]] = []

    def fake_seed(
        config: ExperimentConfig,
        *,
        seed: int,
        device: str,
        candidate_chunk_size: int,
    ) -> SimpleNamespace:
        assert config.seeds == seeds
        calls.append((seed, device, candidate_chunk_size))
        return _result(seed, device)

    _InlineExecutor.instances = []
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(runner, "run_phase0_seed", fake_seed)
    config = ExperimentConfig(
        seeds=seeds,
        devices=("cpu",),
        output_dir=output,
    )
    runner.run_config(
        config,
        output,
        workers_per_device=1,
        candidate_chunk_size=candidate_chunk_size,
        resume=False,
    )
    return config, calls


def test_frozen_config_is_the_prespecified_phase0_protocol() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    config = ExperimentConfig.from_yaml(CONFIG_PATH)

    assert raw["seeds"] == list(range(100))
    assert config.data.dataset == "synthetic"
    assert config.horizon == 12
    assert config.q_grid_size == 101
    assert (config.q_quantile_min, config.q_quantile_max) == (0.50, 0.999)
    assert config.samples.logged == 5_000
    assert config.samples.oracle_surface_rollouts == 5_000
    assert config.samples.oracle_rollouts == 50_000
    assert config.seeds == tuple(range(100))
    assert config.output_dir == Path("results/work/phase0a_profiled_vs_greedy")
    assert "paper_final" not in config.output_dir.parts


def test_cli_defaults_to_one_spawn_worker_per_gpu_and_chunk_16() -> None:
    runner = _load_runner()
    args = runner.build_parser().parse_args([])

    assert args.config == CONFIG_PATH
    assert args.workers_per_device == 1
    assert args.candidate_chunk_size == 16
    assert args.resume is False
    worker_devices, jobs = runner._build_seed_jobs(
        (0, 1, 2, 3, 4),
        ("cuda:0", "cuda:1"),
        args.workers_per_device,
    )
    assert worker_devices == ("cuda:0", "cuda:1")
    assert jobs == (
        (0, 0, "cuda:0"),
        (1, 1, "cuda:1"),
        (0, 2, "cuda:0"),
        (1, 3, "cuda:1"),
        (0, 4, "cuda:0"),
    )


def test_real_spawn_executor_reuses_one_child_and_propagates_failures() -> None:
    runner = _load_runner()
    results = runner._execute_jobs(
        ("cpu",),
        ((0, (11,)), (0, (12,))),
        worker_function=_spawn_identity,
    )

    assert [value for _, value in results] == [11, 12]
    assert len({pid for pid, _ in results}) == 1

    with pytest.raises(RuntimeError, match="spawn probe failure"):
        runner._execute_jobs(
            ("cpu",),
            ((0, (13,)),),
            worker_function=_spawn_failure,
        )


def test_cuda_worker_pins_device_before_oracle_and_cleans_same_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    events: list[tuple[str, ...]] = []

    class DeviceContext:
        def __init__(self, device: torch.device) -> None:
            self.device = str(device)

        def __enter__(self) -> None:
            events.append(("enter", self.device))

        def __exit__(self, *_args: object) -> None:
            events.append(("exit", self.device))

    class FakeCuda:
        def set_device(self, device: torch.device) -> None:
            events.append(("set_device", str(device)))

        def device(self, device: torch.device) -> DeviceContext:
            events.append(("device_context", str(device)))
            return DeviceContext(device)

        def empty_cache(self) -> None:
            events.append(("empty_cache",))

    def fail_after_pin(
        _config: ExperimentConfig,
        *,
        seed: int,
        device: str,
        candidate_chunk_size: int,
    ) -> None:
        assert (seed, candidate_chunk_size) == (3, 5)
        events.append(("oracle", device))
        raise RuntimeError("oracle failed")

    monkeypatch.setattr(runner.torch, "cuda", FakeCuda())
    monkeypatch.setattr(runner, "run_phase0_seed", fail_after_pin)
    config = ExperimentConfig(
        seeds=(3,),
        devices=("cuda:1",),
        output_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="oracle failed"):
        runner._run_and_write(config, 3, "cuda:1", tmp_path, 5)

    assert events == [
        ("set_device", "cuda:1"),
        ("device_context", "cuda:1"),
        ("enter", "cuda:1"),
        ("oracle", "cuda:1"),
        ("empty_cache",),
        ("exit", "cuda:1"),
    ]


def test_config_hash_is_canonical_across_yaml_key_order() -> None:
    runner = _load_runner()
    first = yaml.safe_load("horizon: 12\ndata:\n  dataset: synthetic\n")
    second = yaml.safe_load("data: {dataset: synthetic}\nhorizon: 12\n")

    assert runner.canonical_config_sha256(first) == runner.canonical_config_sha256(second)
    assert runner.canonical_config_sha256(first) != runner.canonical_config_sha256(
        {"horizon": 13, "data": {"dataset": "synthetic"}}
    )


def test_fresh_run_uses_persistent_spawn_executor_and_resume_skips_valid_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "phase0"
    config, calls = _run_fake_study(runner, monkeypatch, output)

    assert calls == [(0, "cpu", 7), (1, "cpu", 7)]
    assert len(_InlineExecutor.instances) == 1
    assert _InlineExecutor.instances[0].max_workers == 1
    assert _InlineExecutor.instances[0].mp_context.get_start_method() == "spawn"
    assert len(_InlineExecutor.instances[0].submissions) == 2
    metadata = json.loads((output / "study_metadata.json").read_text())
    execution = metadata["execution"]
    assert len(metadata["source_tree_sha256"]) == 64
    assert len(execution["experiment_tree_sha256"]) == 64
    assert len(execution["config_sha256"]) == 64
    assert execution["workers_per_device"] == 1
    assert execution["candidate_chunk_size"] == 7
    assert (output / "COMPLETE").is_file()

    monkeypatch.setattr(
        runner,
        "run_phase0_seed",
        lambda *_args, **_kwargs: pytest.fail("contract-valid seed must be skipped"),
    )
    runner.run_config(
        config,
        output,
        workers_per_device=1,
        candidate_chunk_size=7,
        resume=True,
    )


def test_fresh_run_refuses_every_existing_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "earlier-paper-output"
    output.mkdir()
    (output / "COMPLETE").write_text("complete\n")
    config = ExperimentConfig(seeds=(0,), devices=("cpu",), output_dir=output)
    monkeypatch.setattr(
        runner,
        "run_phase0_seed",
        lambda *_args, **_kwargs: pytest.fail("existing output must fail before work"),
    )

    with pytest.raises(FileExistsError, match="fresh phase0 output already exists"):
        runner.run_config(
            config,
            output,
            workers_per_device=1,
            candidate_chunk_size=16,
            resume=False,
        )

    assert (output / "COMPLETE").read_text() == "complete\n"


def test_resume_runs_only_missing_seed_when_the_suite_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "interrupted"
    config, calls = _run_fake_study(runner, monkeypatch, output, seeds=(0, 1, 2))
    calls.clear()
    (output / "COMPLETE").unlink()
    shutil.rmtree(output / "seed_00001")

    runner.run_config(
        config,
        output,
        workers_per_device=1,
        candidate_chunk_size=7,
        resume=True,
    )

    assert calls == [(1, "cpu", 7)]
    assert (output / "COMPLETE").is_file()


def test_resume_rejects_inconsistent_suite_complete_without_running_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "inconsistent-complete"
    config, _ = _run_fake_study(runner, monkeypatch, output, seeds=(0, 1))
    shutil.rmtree(output / "seed_00001")
    monkeypatch.setattr(
        runner,
        "run_phase0_seed",
        lambda *_args, **_kwargs: pytest.fail(
            "an inconsistent suite COMPLETE must block resume"
        ),
    )

    with pytest.raises(RuntimeError, match="study COMPLETE"):
        runner.run_config(
            config,
            output,
            workers_per_device=1,
            candidate_chunk_size=7,
            resume=True,
        )

    assert (output / "COMPLETE").is_file()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("source", "source_tree_sha256"),
        ("experiment", "experiment_tree_sha256"),
        ("config_hash", "config_sha256"),
        ("seeds", "requested seeds"),
        ("devices", "requested devices"),
        ("workers", "workers_per_device"),
        ("chunk", "candidate_chunk_size"),
        ("stored_config", "stored config"),
    ],
)
def test_resume_aborts_before_work_on_any_provenance_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    runner = _load_runner()
    output = tmp_path / mutation
    config, calls = _run_fake_study(runner, monkeypatch, output, seeds=(0,))
    assert calls == [(0, "cpu", 7)]

    metadata_path = output / "study_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if mutation == "source":
        metadata["source_tree_sha256"] = "0" * 64
    elif mutation == "experiment":
        metadata["execution"]["experiment_tree_sha256"] = "1" * 64
    elif mutation == "config_hash":
        metadata["execution"]["config_sha256"] = "2" * 64
    elif mutation == "seeds":
        metadata["seeds"] = [1]
    elif mutation == "devices":
        metadata["devices"] = ["cuda:0"]
    elif mutation == "workers":
        metadata["execution"]["workers_per_device"] = 2
    elif mutation == "chunk":
        metadata["execution"]["candidate_chunk_size"] = 16
    elif mutation == "stored_config":
        stored = yaml.safe_load((output / "config.yaml").read_text())
        stored["output_dir"] = "results/work/paper_final/rq1/synthetic"
        (output / "config.yaml").write_text(yaml.safe_dump(stored))
    metadata_path.write_text(json.dumps(metadata))

    monkeypatch.setattr(
        runner,
        "run_phase0_seed",
        lambda *_args, **_kwargs: pytest.fail("mismatched resume must not run seeds"),
    )
    with pytest.raises(RuntimeError, match=match):
        runner.run_config(
            config,
            output,
            workers_per_device=1,
            candidate_chunk_size=7,
            resume=True,
        )


def test_complete_marker_is_insufficient_without_exact_four_row_contract(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    seed_dir = tmp_path / "seed_00000"
    seed_dir.mkdir()
    (seed_dir / "COMPLETE").write_text('{"seed": 0, "status": "complete"}\n')
    pd.DataFrame(_result(0, "cpu").records).to_csv(seed_dir / "records.csv", index=False)
    np.savez_compressed(seed_dir / "surfaces.npz", q=np.array([1.0]))
    (seed_dir / "metadata.json").write_text(json.dumps({"seed": 0}))

    assert runner.validate_seed_artifact(seed_dir, 0) == seed_dir

    records = pd.read_csv(seed_dir / "records.csv")
    records.iloc[:-1].to_csv(seed_dir / "records.csv", index=False)
    with pytest.raises(RuntimeError, match="exactly four primary rows"):
        runner.validate_seed_artifact(seed_dir, 0)


@pytest.mark.parametrize(
    "failure",
    [
        "records",
        "surfaces",
        "metadata",
        "bad_metadata",
        "metadata_shape",
        "bad_surfaces",
        "bad_seed",
        "truncated_member",
        "seed_source",
        "seed_config",
    ],
)
def test_partial_or_malformed_seed_is_actionable_and_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    runner = _load_runner()
    output = tmp_path / failure
    config, _ = _run_fake_study(runner, monkeypatch, output, seeds=(0,))
    seed_dir = output / "seed_00000"
    (output / "COMPLETE").unlink()
    if failure == "bad_metadata":
        (seed_dir / "metadata.json").write_text("not json")
    elif failure == "metadata_shape":
        (seed_dir / "metadata.json").write_text("[]")
    elif failure == "bad_surfaces":
        (seed_dir / "surfaces.npz").write_text("not an npz")
    elif failure == "bad_seed":
        records = pd.read_csv(seed_dir / "records.csv")
        records["seed"] = 0.5
        records.to_csv(seed_dir / "records.csv", index=False)
    elif failure == "truncated_member":
        member = io.BytesIO()
        np.save(member, np.arange(4))
        with zipfile.ZipFile(seed_dir / "surfaces.npz", "w") as archive:
            archive.writestr("truncated.npy", member.getvalue()[:-8])
    elif failure in {"seed_source", "seed_config"}:
        metadata = json.loads((seed_dir / "metadata.json").read_text())
        if failure == "seed_source":
            metadata["source_tree_sha256"] = "0" * 64
        else:
            metadata["config"]["output_dir"] = "results/work/paper_final/rq1/synthetic"
        (seed_dir / "metadata.json").write_text(json.dumps(metadata))
    else:
        missing_name = {
            "records": "records.csv",
            "surfaces": "surfaces.npz",
            "metadata": "metadata.json",
        }[failure]
        (seed_dir / missing_name).unlink()
    before = sorted(path.name for path in seed_dir.iterdir())
    completed_calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        runner,
        "run_phase0_seed",
        lambda *_args, **_kwargs: pytest.fail("malformed seed must not be overwritten"),
    )
    monkeypatch.setattr(
        runner,
        "mark_study_complete",
        lambda _output, seeds: completed_calls.append(seeds),
    )

    with pytest.raises(RuntimeError, match="seed 0"):
        runner.run_config(
            config,
            output,
            workers_per_device=1,
            candidate_chunk_size=7,
            resume=True,
        )

    assert sorted(path.name for path in seed_dir.iterdir()) == before
    assert completed_calls == []


@pytest.mark.parametrize(
    ("expected_seed", "stored_seed"),
    [(1, True), (0, 0.0)],
)
def test_resume_rejects_non_integer_json_metadata_seed_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_seed: int,
    stored_seed: object,
) -> None:
    runner = _load_runner()
    output = tmp_path / f"metadata-{expected_seed}-{stored_seed}"
    config, _ = _run_fake_study(
        runner,
        monkeypatch,
        output,
        seeds=(expected_seed,),
    )
    seed_dir = output / f"seed_{expected_seed:05d}"
    (output / "COMPLETE").unlink()
    metadata_path = seed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["seed"] = stored_seed
    metadata_path.write_text(json.dumps(metadata))
    completion_calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        runner,
        "mark_study_complete",
        lambda _output, seeds: completion_calls.append(seeds),
    )

    with pytest.raises(RuntimeError, match=f"seed {expected_seed}"):
        runner.run_config(
            config,
            output,
            workers_per_device=1,
            candidate_chunk_size=7,
            resume=True,
        )

    assert completion_calls == []


def test_suite_complete_is_published_only_after_all_100_seeds_revalidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "full"
    config, calls = _run_fake_study(
        runner,
        monkeypatch,
        output,
        seeds=tuple(range(100)),
    )
    assert len(calls) == 100
    assert (output / "COMPLETE").is_file()

    (output / "COMPLETE").unlink()
    broken_records = pd.read_csv(output / "seed_00099" / "records.csv").iloc[:-1]
    broken_records.to_csv(output / "seed_00099" / "records.csv", index=False)
    mark_calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        runner,
        "mark_study_complete",
        lambda _output, seeds: mark_calls.append(seeds),
    )

    with pytest.raises(RuntimeError, match="seed 99"):
        runner.run_config(
            config,
            output,
            workers_per_device=1,
            candidate_chunk_size=7,
            resume=True,
        )

    assert mark_calls == []
    assert not (output / "COMPLETE").exists()
