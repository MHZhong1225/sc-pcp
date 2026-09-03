from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest

from scpcp.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_conservatism_decomposition.py"
    spec = importlib.util.spec_from_file_location("run_conservatism_decomposition", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fresh_stream_is_globally_disjoint_for_registered_100_seeds() -> None:
    runner = _load_runner()

    runner._validate_fresh_streams(tuple(range(100)))


def test_fresh_stream_collision_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "FRESH_EVALUATION_STREAM", 900_001)

    with pytest.raises(RuntimeError, match="collide"):
        runner._validate_fresh_streams(tuple(range(3)))


def test_input_config_normalization_ignores_only_output_and_mechanism_flag() -> None:
    runner = _load_runner()
    base = ExperimentConfig()
    allowed_difference = replace(
        base,
        output_dir=Path("somewhere-else"),
        paper=replace(base.paper, save_mechanism_diagonal=True),
    )
    different_seeds = replace(base, seeds=(0, 1))
    different_devices = replace(base, devices=("cuda:0",))
    different_delta = replace(
        base,
        certification=replace(base.certification, delta=0.10),
    )

    normalized = runner._normalized_input_config(base)
    assert runner._normalized_input_config(allowed_difference) == normalized
    assert runner._normalized_input_config(different_seeds) != normalized
    assert runner._normalized_input_config(different_devices) != normalized
    assert runner._normalized_input_config(different_delta) != normalized
