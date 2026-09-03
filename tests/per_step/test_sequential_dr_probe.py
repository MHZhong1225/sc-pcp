from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_sequential_dr_probe",
    ROOT / "scripts" / "run_sequential_dr_probe.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _aggregate(gamma: float, ratio: float, upper: float, cdf_better: bool) -> dict[str, object]:
    return {
        "gamma": gamma,
        "dr_to_prefix_q90_error_ratio": ratio,
        "dr_to_prefix_q90_error_ratio_upper_95": upper,
        "dr_cdf_better_than_prefix": cdf_better,
    }


def test_dr_gate_requires_every_prespecified_strong_shift_condition() -> None:
    passing = [
        _aggregate(0.0, 1.0, 1.0, False),
        _aggregate(-2.0, 1.04, 1.1, False),
        _aggregate(-3.0, 0.86, 0.98, True),
        _aggregate(-4.0, 0.88, 0.99, True),
    ]
    failing = [*passing]
    failing[-1] = _aggregate(-4.0, 0.88, 1.01, True)

    assert MODULE._gate(passing)["status"] == "GO_TO_FRESH_CONFIRMATION"
    assert MODULE._gate(failing)["status"] == "NO_GO"
