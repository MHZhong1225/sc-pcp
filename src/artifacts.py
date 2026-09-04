from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from config import ExperimentConfig
from device import runtime_metadata


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def source_tree_sha256() -> str:
    """Hash the active implementation when the workspace has no Git commit."""

    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "src" / "scpcp").rglob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
        *sorted((root / "tools").glob("*.py")),
        root / "pyproject.toml",
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def experiment_tree_sha256() -> str:
    """Hash executable sources and paper configs for a whole-suite freeze."""

    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "src" / "scpcp").rglob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
        *sorted((root / "tools").glob("*.py")),
        *sorted((root / "configs").glob("*.yaml")),
        root / "pyproject.toml",
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def write_study_metadata(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    execution: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_text(output_dir / "config.yaml", yaml.safe_dump(config.to_dict(), sort_keys=False))
    metadata = {
        "git_revision": git_revision(),
        "source_tree_sha256": source_tree_sha256(),
        "devices": list(config.devices),
        "seeds": list(config.seeds),
        "execution": _jsonable(execution or {}),
    }
    _write_text(output_dir / "study_metadata.json", json.dumps(metadata, indent=2))
    _write_study_status(output_dir, config.seeds, status="running")


def mark_study_complete(output_dir: Path, expected_seeds: tuple[int, ...]) -> None:
    """Publish a complete status only after every requested seed is durable."""

    completed = _completed_seed_ids(output_dir, expected_seeds)
    missing = sorted(set(expected_seeds) - set(completed))
    if missing:
        _write_study_status(
            output_dir,
            expected_seeds,
            status="incomplete",
            completed_seeds=completed,
            error=f"missing COMPLETE markers for seeds {missing}",
        )
        raise RuntimeError(f"study is missing completed seed artifacts: {missing}")
    consistency_errors = _artifact_consistency_errors(output_dir, expected_seeds)
    if consistency_errors:
        message = "; ".join(consistency_errors)
        _write_study_status(
            output_dir,
            expected_seeds,
            status="incomplete",
            completed_seeds=completed,
            error=message,
        )
        raise RuntimeError(f"study artifact consistency check failed: {message}")
    _write_study_status(
        output_dir,
        expected_seeds,
        status="complete",
        completed_seeds=completed,
    )
    _write_text(output_dir / "COMPLETE", "complete\n")
    _fsync_directory(output_dir)


def mark_study_failed(
    output_dir: Path,
    expected_seeds: tuple[int, ...],
    error: BaseException,
) -> None:
    """Record a terminal failure without hiding seeds that finished safely."""

    _write_study_status(
        output_dir,
        expected_seeds,
        status="failed",
        completed_seeds=_completed_seed_ids(output_dir, expected_seeds),
        error=f"{type(error).__name__}: {error}",
    )


def write_collection_status(
    output_dir: Path,
    expected_settings: tuple[str, ...],
    *,
    status: str,
    completed_settings: tuple[str, ...] = (),
    error: BaseException | None = None,
) -> None:
    """Atomically publish aggregate progress for a multi-setting study."""

    expected = sorted(set(expected_settings))
    completed = sorted(set(completed_settings))
    missing = sorted(set(expected) - set(completed))
    invalid_complete = status == "complete" and bool(missing)
    published_status = "incomplete" if invalid_complete else status
    published_error = (
        f"missing completed settings: {missing}"
        if invalid_complete
        else None if error is None else f"{type(error).__name__}: {error}"
    )
    payload = {
        "status": published_status,
        "expected_settings": expected,
        "completed_settings": completed,
        "missing_settings": missing,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "error": published_error,
    }
    _atomic_write_text(output_dir / "study_status.json", json.dumps(payload, indent=2) + "\n")
    if invalid_complete:
        raise RuntimeError(f"collection is missing completed settings: {missing}")
    if published_status == "complete":
        _write_text(output_dir / "COMPLETE", "complete\n")
        _fsync_directory(output_dir)


def write_seed_result(result: Any, output_dir: Path, config: ExperimentConfig) -> Path:
    destination = output_dir / f"seed_{result.seed:05d}"
    if destination.exists():
        raise FileExistsError(f"result already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=output_dir))
    try:
        pd.DataFrame(result.records).to_csv(temporary / "records.csv", index=False)
        np.savez_compressed(
            temporary / "surfaces.npz",
            **{name: _to_numpy(value) for name, value in result.surfaces.items()},
        )
        metadata = {
            "seed": result.seed,
            "device": result.device,
            "git_revision": git_revision(),
            "source_tree_sha256": source_tree_sha256(),
            "runtime": runtime_metadata(result.device),
            "diagnostics": _jsonable(result.diagnostics),
            "config": config.to_dict(),
        }
        _write_text(temporary / "metadata.json", json.dumps(metadata, indent=2))
        _fsync_file(temporary / "records.csv")
        _fsync_file(temporary / "surfaces.npz")
        # This marker is written last.  Since the entire directory is then
        # atomically renamed, readers can never observe a partial seed that also
        # claims completion.
        _write_text(
            temporary / "COMPLETE",
            json.dumps({"seed": result.seed, "status": "complete"}) + "\n",
        )
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_records(path: str | Path) -> pd.DataFrame:
    root = Path(path)
    files = sorted(
        file
        for file in root.rglob("records.csv")
        if (file.parent / "COMPLETE").is_file()
    )
    if not files:
        raise FileNotFoundError(f"no completed records.csv files under {root}")
    return pd.concat((pd.read_csv(file) for file in files), ignore_index=True)


def _write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_write_text(path: Path, content: str) -> None:
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
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_study_status(
    output_dir: Path,
    expected_seeds: tuple[int, ...],
    *,
    status: str,
    completed_seeds: tuple[int, ...] = (),
    error: str | None = None,
) -> None:
    completed = sorted(set(completed_seeds))
    expected = sorted(set(expected_seeds))
    payload = {
        "status": status,
        "expected_seeds": expected,
        "completed_seeds": completed,
        "missing_seeds": sorted(set(expected) - set(completed)),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    _atomic_write_text(output_dir / "study_status.json", json.dumps(payload, indent=2) + "\n")


def _completed_seed_ids(output_dir: Path, expected_seeds: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        seed
        for seed in expected_seeds
        if (output_dir / f"seed_{seed:05d}" / "COMPLETE").is_file()
    )


def _artifact_consistency_errors(output_dir: Path, expected_seeds: tuple[int, ...]) -> list[str]:
    """Verify that every seed used the study's frozen source and config."""

    study_metadata = json.loads((output_dir / "study_metadata.json").read_text())
    study_config = yaml.safe_load((output_dir / "config.yaml").read_text())
    expected_hash = study_metadata.get("source_tree_sha256")
    errors = []
    if study_metadata.get("seeds") != list(expected_seeds):
        errors.append("study metadata seeds differ from requested seeds")
    if source_tree_sha256() != expected_hash:
        errors.append("active source changed after the study started")
    for seed in expected_seeds:
        metadata_path = output_dir / f"seed_{seed:05d}" / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"seed {seed} metadata unreadable: {error}")
            continue
        if metadata.get("seed") != seed:
            errors.append(f"seed {seed} metadata has the wrong seed ID")
        if metadata.get("source_tree_sha256") != expected_hash:
            errors.append(f"seed {seed} source hash differs from study source hash")
        if metadata.get("config") != study_config:
            errors.append(f"seed {seed} config differs from study config")
    return errors


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
