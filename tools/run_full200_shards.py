"""Run the prespecified 200-replication synthetic study suite in shards.

This launcher deliberately lives outside ``scripts/`` because it only
orchestrates immutable experiment entry points.  Each shard is a normal,
auditable ``run_per_step_study.py`` collection with its own completion
markers.  The central condition is not repeated: the completed 200-seed
synthetic main experiment supplies beta=1, eta=1, T=12, n=5000, cap=10,
alpha=0.10, action-cost lambda=0.10, and MFCS depth=3.  Every non-central
condition is a normal 0:200 collection; aggregation verifies reuse explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import tempfile
from threading import Lock, Thread
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpcp.artifacts import source_tree_sha256
from scpcp.device import resolve_devices


EXPECTED_SEEDS = tuple(range(200))


@dataclass(frozen=True)
class Task:
    section: str
    label: str
    study: str
    seeds: str
    extra_args: tuple[str, ...]


def build_tasks() -> tuple[Task, ...]:
    tasks: list[Task] = []
    for beta in ("0", "0.5", "1", "2"):
        for eta in ("0.25", "0.5", "1", "2"):
            if beta == "1" and eta == "1":
                continue
            tasks.append(
                Task(
                    section="factorial",
                    label=f"beta_{beta}__eta_{eta}",
                    study="factorial",
                    seeds="0:200",
                    extra_args=(
                        "--feedback-values",
                        beta,
                        "--policy-tilt-values",
                        eta,
                    ),
                )
            )

    one_factor_specs = (
        ("feedback_extra", "feedback_1.5", "feedback", "1.5"),
        ("horizon", "horizon_4", "horizon", "4"),
        ("horizon", "horizon_8", "horizon", "8"),
        ("horizon", "horizon_24", "horizon", "24"),
        ("sample_size", "logged_n_500", "sample_size", "500"),
        ("sample_size", "logged_n_1000", "sample_size", "1000"),
        ("sample_size", "logged_n_2500", "sample_size", "2500"),
        ("sample_size", "logged_n_10000", "sample_size", "10000"),
        ("ratio_cap", "ratio_cap_1.1", "ratio_cap", "1.1"),
        ("ratio_cap", "ratio_cap_1.25", "ratio_cap", "1.25"),
        ("ratio_cap", "ratio_cap_2", "ratio_cap", "2"),
        ("alpha", "alpha_0.05", "alpha", "0.05"),
        ("alpha", "alpha_0.20", "alpha", "0.20"),
        ("action_cost", "action_cost_0", "action_cost", "0"),
        ("action_cost", "action_cost_0.05", "action_cost", "0.05"),
        ("action_cost", "action_cost_0.2", "action_cost", "0.2"),
        ("mfcs_depth", "mfcs_depth_1", "mfcs_depth", "1"),
        ("mfcs_depth", "mfcs_depth_2", "mfcs_depth", "2"),
        ("mfcs_depth", "mfcs_depth_4", "mfcs_depth", "4"),
    )
    tasks.extend(
        Task(
            section=section,
            label=label,
            study=study,
            seeds="0:200",
            extra_args=("--values", value),
        )
        for section, label, study, value in one_factor_specs
    )
    return tuple(tasks)


def completed_collection(
    task_dir: Path,
    *,
    task: Task,
    expected_source_hash: str,
) -> Path | None:
    """Return the one complete collection matching this exact frozen task."""

    complete_roots = sorted(path.parent for path in task_dir.glob("*/COMPLETE"))
    matching = [
        root
        for root in complete_roots
        if _collection_matches(root, task, expected_source_hash)
    ]
    if len(matching) > 1:
        raise RuntimeError(f"multiple matching completed collections found under {task_dir}")
    return matching[0] if matching else None


def _collection_matches(root: Path, task: Task, expected_source_hash: str) -> bool:
    try:
        manifest = json.loads((root / "study_manifest.json").read_text())
        collection_status = json.loads((root / "study_status.json").read_text())
        setting = root / task.label
        study_metadata = json.loads((setting / "study_metadata.json").read_text())
        setting_status = json.loads((setting / "study_status.json").read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if not all(
        isinstance(payload, dict)
        for payload in (manifest, collection_status, study_metadata, setting_status)
    ):
        return False

    expected_study = "feedback_policy" if task.study == "factorial" else task.study
    setting_rows = manifest.get("settings")
    setting_labels = (
        [row.get("label") for row in setting_rows]
        if isinstance(setting_rows, list) and all(isinstance(row, dict) for row in setting_rows)
        else []
    )
    execution = study_metadata.get("execution")
    if not isinstance(execution, dict):
        return False
    expected_seed_list = list(EXPECTED_SEEDS)
    if any(
        (
            task.seeds != "0:200",
            manifest.get("study") != expected_study,
            manifest.get("seeds") != expected_seed_list,
            manifest.get("source_tree_sha256") != expected_source_hash,
            setting_labels != [task.label],
            collection_status.get("status") != "complete",
            collection_status.get("completed_settings") != [task.label],
            collection_status.get("missing_settings") != [],
            study_metadata.get("seeds") != expected_seed_list,
            study_metadata.get("source_tree_sha256") != expected_source_hash,
            execution.get("collection_source_tree_sha256") != expected_source_hash,
            setting_status.get("status") != "complete",
            setting_status.get("expected_seeds") != expected_seed_list,
            setting_status.get("completed_seeds") != expected_seed_list,
            setting_status.get("missing_seeds") != [],
            not (setting / "COMPLETE").is_file(),
        )
    ):
        return False
    completed_seed_dirs = {
        path.parent.name
        for path in setting.glob("seed_*/COMPLETE")
    }
    return completed_seed_dirs == {f"seed_{seed:05d}" for seed in EXPECTED_SEEDS}


def build_command(
    task: Task,
    *,
    device: str,
    config: Path,
    task_dir: Path,
    seed_workers_per_device: int,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_per_step_study.py"),
        "--config",
        str(config),
        "--study",
        task.study,
        "--seeds",
        task.seeds,
        "--devices",
        device,
        "--workers-per-device",
        str(seed_workers_per_device),
        "--output-dir",
        str(task_dir),
        *task.extra_args,
    ]


def run_task(
    task: Task,
    *,
    device: str,
    config: Path,
    output_root: Path,
    logs_root: Path,
    seed_workers_per_device: int,
    expected_source_hash: str,
) -> dict[str, Any]:
    task_dir = output_root / task.section / task.label
    task_dir.mkdir(parents=True, exist_ok=True)
    lock_path = task_dir / ".launcher.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "task": asdict(task),
                "device": device,
                "status": "claimed_elsewhere",
                "collection": None,
            }
        observed_source_hash = source_tree_sha256()
        if observed_source_hash != expected_source_hash:
            return {
                "task": asdict(task),
                "device": device,
                "status": "source_changed",
                "expected_source_tree_sha256": expected_source_hash,
                "observed_source_tree_sha256": observed_source_hash,
                "collection": None,
            }
        existing = completed_collection(
            task_dir,
            task=task,
            expected_source_hash=expected_source_hash,
        )
        if existing is not None:
            return {
                "task": asdict(task),
                "device": device,
                "status": "skipped_complete",
                "collection": str(existing),
            }
        incompatible_complete = sorted(
            str(path.parent) for path in task_dir.glob("*/COMPLETE")
        )
        if incompatible_complete:
            return {
                "task": asdict(task),
                "device": device,
                "status": "incompatible_complete",
                "collections": incompatible_complete,
                "collection": None,
            }

        command = build_command(
            task,
            device=device,
            config=config,
            task_dir=task_dir,
            seed_workers_per_device=seed_workers_per_device,
        )
        log_path = logs_root / f"{task.section}__{task.label}.log"
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        started_at = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"START {started_at} device={device} "
                f"seed_workers={seed_workers_per_device}\n"
            )
            log.flush()
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            finished_at = datetime.now(timezone.utc).isoformat()
            log.write(f"END {finished_at} returncode={result.returncode}\n")
            log.flush()

        observed_source_hash = source_tree_sha256()
        collection = completed_collection(
            task_dir,
            task=task,
            expected_source_hash=expected_source_hash,
        )
        status = (
            "complete"
            if result.returncode == 0
            and collection is not None
            and observed_source_hash == expected_source_hash
            else "failed"
        )
        return {
            "task": asdict(task),
            "device": device,
            "status": status,
            "returncode": result.returncode,
            "collection": None if collection is None else str(collection),
            "log": str(log_path),
            "started_at": started_at,
            "finished_at": finished_at,
            "expected_source_tree_sha256": expected_source_hash,
            "observed_source_tree_sha256": observed_source_hash,
        }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}-",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "per_step_synthetic.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "final" / "studies" / "full200_shards",
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--tasks-per-device", type=int, default=1)
    parser.add_argument("--seed-workers-per-device", type=int, default=4)
    parser.add_argument(
        "--sections",
        default=None,
        help="optional comma-separated task sections, useful for safe parallel backfill",
    )
    args = parser.parse_args()

    try:
        devices = resolve_devices(args.devices)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if args.tasks_per_device < 1 or args.seed_workers_per_device < 1:
        parser.error("task and seed worker counts must be positive")

    output_root = args.output_root.resolve()
    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    selected_sections = (
        None
        if args.sections is None
        else {item.strip() for item in args.sections.split(",") if item.strip()}
    )
    selected_tasks = tuple(
        task
        for task in build_tasks()
        if selected_sections is None or task.section in selected_sections
    )
    if not selected_tasks:
        parser.error("the section filter selected no tasks")
    tasks: Queue[Task] = Queue()
    for task in selected_tasks:
        tasks.put(task)
    expected_source_hash = source_tree_sha256()

    results: list[dict[str, Any]] = []
    results_lock = Lock()

    def worker(device: str) -> None:
        while True:
            try:
                task = tasks.get_nowait()
            except Empty:
                return
            try:
                outcome = run_task(
                    task,
                    device=device,
                    config=args.config.resolve(),
                    output_root=output_root,
                    logs_root=logs_root,
                    seed_workers_per_device=args.seed_workers_per_device,
                    expected_source_hash=expected_source_hash,
                )
            except BaseException as error:
                outcome = {
                    "task": asdict(task),
                    "device": device,
                    "status": "launcher_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            with results_lock:
                results.append(outcome)
                print(json.dumps(outcome, sort_keys=True), flush=True)
            tasks.task_done()

    threads = [
        Thread(target=worker, args=(device,), daemon=False)
        for device in devices
        for _ in range(args.tasks_per_device)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    summary = {
        "status": "complete" if all(row["status"] in {"complete", "skipped_complete"} for row in results) else "failed",
        "task_count": len(selected_tasks),
        "selected_sections": None if selected_sections is None else sorted(selected_sections),
        "source_tree_sha256": expected_source_hash,
        "results": sorted(results, key=lambda row: (row["task"]["section"], row["task"]["label"])),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_suffix = "" if selected_sections is None else "__" + "_".join(sorted(selected_sections))
    summary_path = output_root / f"launcher_summary{summary_suffix}.json"
    _atomic_write_json(summary_path, summary)
    print(summary_path, flush=True)
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
