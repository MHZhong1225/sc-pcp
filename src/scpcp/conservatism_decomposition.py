"""Pure recovery and reporting helpers for the A/C/D/E width decomposition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
import torch


LAYERS = ("A", "C", "D", "E")
LAYER_METHODS = {
    "A": "Greedy Sequential Reference",
    "C": "Profiled Oracle Point",
    "D": "Learned-COT Point",
    "E": "SC-PCP Bootstrap/LCB",
}
RUNNER_LAYERS = {
    "A_sequential": "A",
    "C_profiled_oracle": "C",
    "D_cot_point": "D",
    "E_lcb": "E",
}


@dataclass(frozen=True)
class DecompositionSchedules:
    a: NDArray[np.floating]
    c: NDArray[np.floating]
    d: NDArray[np.floating]
    e: NDArray[np.floating]

    def by_layer(self) -> dict[str, NDArray[np.floating]]:
        return {"A": self.a, "C": self.c, "D": self.d, "E": self.e}


@dataclass(frozen=True)
class DecompositionIndices:
    a: tuple[int, ...]
    c: int
    d: int
    e: int

    def by_layer(self) -> dict[str, tuple[int, ...]]:
        return {
            "A": self.a,
            "C": (self.c,),
            "D": (self.d,),
            "E": (self.e,),
        }


@dataclass(frozen=True)
class DecompositionDiagnostics:
    target: float
    horizon: int
    candidate_count: int
    d_ordered_prefix: tuple[int, ...]
    d_stopped_index: int | None
    e_ordered_prefix: tuple[int, ...]
    e_stopped_index: int | None
    c_oracle_min_coverage: float
    d_point_min_coverage: float
    e_lcb_min: float


@dataclass(frozen=True)
class DecompositionSelection:
    schedules: DecompositionSchedules
    indices: DecompositionIndices
    diagnostics: DecompositionDiagnostics


@dataclass(frozen=True)
class FrozenDecompositionSchedules:
    """Runner-facing schedules recovered from two completed seed directories."""

    schedules: dict[str, Tensor]
    indices: dict[str, int | None]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class OrderedPointSelection:
    index: int
    passing_prefix: tuple[int, ...]
    stopped_index: int | None


@dataclass(frozen=True)
class FreshEvaluation:
    coverage: NDArray[np.floating]
    normalized_width: NDArray[np.floating]
    micro_normalized_width: float
    patient_normalized_width: float
    n_rollouts: int


@dataclass(frozen=True)
class LogRatioDecomposition:
    a_to_c: float
    c_to_d: float
    d_to_e: float
    a_to_e: float
    closure_error: float


def load_standard_decomposition(
    phase0_surfaces_path: str | Path,
    paper_surfaces_path: str | Path,
    *,
    target: float = 0.90,
) -> DecompositionSelection:
    """Load two non-pickle NPZ archives and recover their standard A/C/D/E rules."""

    with np.load(phase0_surfaces_path, allow_pickle=False) as archive:
        phase0 = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(paper_surfaces_path, allow_pickle=False) as archive:
        paper = {name: np.array(archive[name], copy=True) for name in archive.files}
    return recover_standard_decomposition(phase0, paper, target=target)


def load_decomposition_schedules(
    phase0_seed_dir: Path,
    paper_seed_dir: Path,
    *,
    alpha: float,
) -> FrozenDecompositionSchedules:
    """Return immutable-by-convention CPU schedules for a fresh-evaluation runner."""

    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and lie in (0, 1)")
    recovered = load_standard_decomposition(
        phase0_seed_dir / "surfaces.npz",
        paper_seed_dir / "surfaces.npz",
        target=1.0 - alpha,
    )
    short_schedules = recovered.schedules.by_layer()
    short_indices = recovered.indices.by_layer()
    schedules = {
        runner_layer: torch.from_numpy(
            np.array(short_schedules[short_layer], copy=True)
        )
        for runner_layer, short_layer in RUNNER_LAYERS.items()
    }
    indices = {
        runner_layer: (
            None if short_layer == "A" else short_indices[short_layer][0]
        )
        for runner_layer, short_layer in RUNNER_LAYERS.items()
    }
    diagnostics = {
        "target": recovered.diagnostics.target,
        "horizon": recovered.diagnostics.horizon,
        "candidate_count": recovered.diagnostics.candidate_count,
        "A_stage_indices": recovered.indices.a,
        "D_ordered_prefix": recovered.diagnostics.d_ordered_prefix,
        "D_stopped_index": recovered.diagnostics.d_stopped_index,
        "E_ordered_prefix": recovered.diagnostics.e_ordered_prefix,
        "E_stopped_index": recovered.diagnostics.e_stopped_index,
        "C_oracle_min_coverage": recovered.diagnostics.c_oracle_min_coverage,
        "D_point_min_coverage": recovered.diagnostics.d_point_min_coverage,
        "E_lcb_min": recovered.diagnostics.e_lcb_min,
    }
    return FrozenDecompositionSchedules(schedules, indices, diagnostics)


def recover_standard_decomposition(
    phase0: Mapping[str, NDArray[np.generic]],
    paper: Mapping[str, NDArray[np.generic]],
    *,
    target: float = 0.90,
) -> DecompositionSelection:
    """Recover schedules while failing closed on provenance-relevant invariants."""

    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target must be finite and lie in (0, 1)")

    phase0_grid = _numeric_array(phase0, "standard_profiled_scale_grid")
    phase0_profile = _numeric_array(phase0, "standard_profile")
    phase0_candidates = _numeric_array(
        phase0, "standard_profiled_candidate_schedules"
    )
    paper_grid = _numeric_array(paper, "scale_grid")
    paper_profile = _numeric_array(paper, "stage_profile")
    paper_candidates = _numeric_array(paper, "candidate_radii")

    _require_bitwise_equal("scale grid", phase0_grid, paper_grid)
    _require_bitwise_equal("stage profile", phase0_profile, paper_profile)
    _require_bitwise_equal(
        "candidate radii", phase0_candidates, paper_candidates
    )
    _validate_family(paper_grid, paper_profile, paper_candidates)
    candidate_count, horizon = paper_candidates.shape

    oracle_coverage = _matrix(
        phase0,
        "standard_profiled_candidate_coverage",
        (candidate_count, horizon),
    )
    oracle_stage_width = _matrix(
        phase0,
        "standard_profiled_candidate_normalized_width",
        (candidate_count, horizon),
        positive=True,
    )
    c_index = _minimum_width_feasible_index(
        oracle_coverage,
        oracle_stage_width.mean(axis=1),
        target=target,
    )
    c_saved = _vector(
        phase0, "standard_profiled_selected_schedule", horizon, positive=True
    )
    _require_schedule_match("C", c_saved, paper_candidates[c_index])

    greedy_grids = _matrix(
        phase0,
        "standard_greedy_stage_grids",
        (horizon, candidate_count),
        positive=True,
    )
    a_schedule = _vector(
        phase0, "standard_greedy_selected_schedule", horizon, positive=True
    )
    a_indices = tuple(
        _unique_bitwise_index(greedy_grids[stage], a_schedule[stage], label=f"A stage {stage}")
        for stage in range(horizon)
    )

    estimated_widths = _vector(
        paper, "estimated_candidate_widths", candidate_count, positive=True
    )
    cot_point = _matrix(
        paper, "cot_diagonal", (candidate_count, horizon)
    )
    d_selection = select_ordered_point_index(
        paper_grid,
        cot_point,
        estimated_widths,
        target=target,
    )

    cot_lower = _matrix(
        paper, "cot_lower_bounds", (candidate_count, horizon)
    )
    e_selection = select_ordered_point_index(
        paper_grid,
        cot_lower,
        estimated_widths,
        target=target,
    )
    e_saved = _vector(
        paper, "scpcp_selected_radii", horizon, positive=True
    )
    e_index = _unique_row_index(paper_candidates, e_saved, label="E")
    if e_index != e_selection.index:
        raise ValueError(
            f"saved E schedule is candidate {e_index}, expected {e_selection.index}"
        )

    _reject_endpoint("A", a_indices, candidate_count)
    for layer, index in (("C", c_index), ("D", d_selection.index), ("E", e_index)):
        _reject_endpoint(layer, (index,), candidate_count)

    schedules = DecompositionSchedules(
        a=_read_only(a_schedule),
        c=_read_only(paper_candidates[c_index]),
        d=_read_only(paper_candidates[d_selection.index]),
        e=_read_only(e_saved),
    )
    indices = DecompositionIndices(
        a=a_indices,
        c=c_index,
        d=d_selection.index,
        e=e_index,
    )
    diagnostics = DecompositionDiagnostics(
        target=target,
        horizon=horizon,
        candidate_count=candidate_count,
        d_ordered_prefix=d_selection.passing_prefix,
        d_stopped_index=d_selection.stopped_index,
        e_ordered_prefix=e_selection.passing_prefix,
        e_stopped_index=e_selection.stopped_index,
        c_oracle_min_coverage=float(oracle_coverage[c_index].min()),
        d_point_min_coverage=float(cot_point[d_selection.index].min()),
        e_lcb_min=float(cot_lower[e_index].min()),
    )
    return DecompositionSelection(schedules, indices, diagnostics)


def select_ordered_point_index(
    candidate_grid: NDArray[np.floating],
    point_surface: NDArray[np.floating],
    widths: NDArray[np.floating],
    *,
    target: float = 0.90,
) -> OrderedPointSelection:
    """Apply the current widest-to-narrowest prefix rule to a point surface."""

    grid = np.asarray(candidate_grid)
    surface = np.asarray(point_surface)
    candidate_widths = np.asarray(widths)
    if grid.ndim != 1 or len(grid) < 1:
        raise ValueError("candidate grid must be a nonempty vector")
    if surface.ndim != 2 or surface.shape[0] != len(grid):
        raise ValueError("point surface must have shape [K, T]")
    if candidate_widths.shape != grid.shape:
        raise ValueError("widths must have shape [K]")
    _require_finite("candidate grid", grid)
    _require_finite("point surface", surface)
    _require_finite("widths", candidate_widths)
    if np.any(np.diff(grid) < 0.0):
        raise ValueError("candidate grid must be nondecreasing")
    if np.any(candidate_widths <= 0.0):
        raise ValueError("widths must be strictly positive")
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target must be finite and lie in (0, 1)")

    prefix: list[int] = []
    stopped_index = None
    for index in range(len(grid) - 1, -1, -1):
        if bool(np.all(surface[index] >= target)):
            prefix.append(index)
            continue
        stopped_index = index
        break
    if not prefix:
        raise ValueError("ordered point selector has no feasible candidate")
    chosen = min(
        prefix,
        key=lambda index: (
            float(candidate_widths[index]),
            float(grid[index]),
            index,
        ),
    )
    return OrderedPointSelection(chosen, tuple(prefix), stopped_index)


def canonical_fresh_records(
    seed: int,
    selection: DecompositionSelection,
    evaluations: Mapping[str, FreshEvaluation],
) -> tuple[dict[str, object], ...]:
    """Convert one common-CRN fresh evaluation into stable row dictionaries."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if set(evaluations) != set(LAYERS):
        raise ValueError("fresh evaluations must contain exactly A, C, D, and E")

    schedules = selection.schedules.by_layer()
    indices = selection.indices.by_layer()
    horizon = selection.diagnostics.horizon
    records = []
    for layer in LAYERS:
        evaluation = evaluations[layer]
        coverage = np.asarray(evaluation.coverage)
        stage_width = np.asarray(evaluation.normalized_width)
        if coverage.shape != (horizon,) or stage_width.shape != (horizon,):
            raise ValueError(f"fresh {layer} coverage and width must have shape [T]")
        _require_finite(f"fresh {layer} coverage", coverage)
        _require_finite(f"fresh {layer} width", stage_width)
        if np.any((coverage < 0.0) | (coverage > 1.0)):
            raise ValueError(f"fresh {layer} coverage must lie in [0, 1]")
        if np.any(stage_width <= 0.0):
            raise ValueError(f"fresh {layer} width must be strictly positive")
        if (
            not math.isfinite(evaluation.micro_normalized_width)
            or not math.isfinite(evaluation.patient_normalized_width)
            or evaluation.micro_normalized_width <= 0.0
            or evaluation.patient_normalized_width <= 0.0
        ):
            raise ValueError(f"fresh {layer} aggregate widths must be finite and positive")
        if type(evaluation.n_rollouts) is not int or evaluation.n_rollouts < 1:
            raise ValueError(f"fresh {layer} n_rollouts must be a positive integer")

        records.append(
            {
                "seed": seed,
                "layer": layer,
                "method": LAYER_METHODS[layer],
                "selection_indices": _json_numbers(indices[layer]),
                "q_by_time": _json_numbers(schedules[layer]),
                "target_coverage": selection.diagnostics.target,
                "target_met": bool(coverage.min() >= selection.diagnostics.target),
                "worst_coverage": float(coverage.min()),
                "average_coverage": float(coverage.mean()),
                "per_time_coverage": _json_numbers(coverage),
                "average_normalized_width": float(evaluation.micro_normalized_width),
                "patient_normalized_width": float(evaluation.patient_normalized_width),
                "per_time_normalized_width": _json_numbers(stage_width),
                "oracle_evaluation_trajectories": evaluation.n_rollouts,
                "evaluation_scope": "common_crn_fresh_target_policy_rollouts",
            }
        )
    return tuple(records)


def log_ratio_decomposition(
    *,
    a_width: float,
    c_width: float,
    d_width: float,
    e_width: float,
) -> LogRatioDecomposition:
    """Return the exact telescoping decomposition on the log-width scale."""

    widths = (a_width, c_width, d_width, e_width)
    if any(not math.isfinite(value) or value <= 0.0 for value in widths):
        raise ValueError("all widths must be finite and strictly positive")
    a_to_c = math.log(c_width / a_width)
    c_to_d = math.log(d_width / c_width)
    d_to_e = math.log(e_width / d_width)
    a_to_e = math.log(e_width / a_width)
    closure_error = a_to_c + c_to_d + d_to_e - a_to_e
    return LogRatioDecomposition(
        a_to_c=a_to_c,
        c_to_d=c_to_d,
        d_to_e=d_to_e,
        a_to_e=a_to_e,
        closure_error=closure_error,
    )


def _numeric_array(
    surfaces: Mapping[str, NDArray[np.generic]], key: str
) -> NDArray[np.generic]:
    if key not in surfaces:
        raise ValueError(f"missing surface: {key}")
    values = np.asarray(surfaces[key])
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"surface {key} must be numeric")
    _require_finite(key, values)
    return values


def _vector(
    surfaces: Mapping[str, NDArray[np.generic]],
    key: str,
    length: int,
    *,
    positive: bool = False,
) -> NDArray[np.generic]:
    values = _numeric_array(surfaces, key)
    if values.shape != (length,):
        raise ValueError(f"surface {key} must have shape ({length},)")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"surface {key} must be strictly positive")
    return values


def _matrix(
    surfaces: Mapping[str, NDArray[np.generic]],
    key: str,
    shape: tuple[int, int],
    *,
    positive: bool = False,
) -> NDArray[np.generic]:
    values = _numeric_array(surfaces, key)
    if values.shape != shape:
        raise ValueError(f"surface {key} must have shape {shape}")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"surface {key} must be strictly positive")
    return values


def _validate_family(
    grid: NDArray[np.generic],
    profile: NDArray[np.generic],
    candidates: NDArray[np.generic],
) -> None:
    if grid.ndim != 1 or len(grid) < 3:
        raise ValueError("scale grid must be a vector with at least three candidates")
    if profile.ndim != 1 or len(profile) < 1:
        raise ValueError("stage profile must be a nonempty vector")
    if candidates.shape != (len(grid), len(profile)):
        raise ValueError("candidate radii must have shape [K, T]")
    if np.any(np.diff(grid) < 0.0):
        raise ValueError("scale grid must be nondecreasing")
    if np.any(grid < 0.0) or np.any(profile <= 0.0) or np.any(candidates <= 0.0):
        raise ValueError("profiled family must be positive")


def _minimum_width_feasible_index(
    coverage: NDArray[np.generic],
    widths: NDArray[np.generic],
    *,
    target: float,
) -> int:
    feasible = np.all(coverage >= target, axis=1)
    if not bool(feasible.any()):
        raise ValueError("profiled oracle has no feasible candidate")
    objective = np.where(feasible, widths, np.inf)
    return int(objective.argmin())


def _require_bitwise_equal(
    label: str, first: NDArray[np.generic], second: NDArray[np.generic]
) -> None:
    equal = (
        first.shape == second.shape
        and first.dtype == second.dtype
        and first.tobytes(order="C") == second.tobytes(order="C")
    )
    if not equal:
        raise ValueError(f"phase0 and paper {label} are not bitwise identical")


def _require_schedule_match(
    label: str, observed: NDArray[np.generic], expected: NDArray[np.generic]
) -> None:
    if (
        observed.shape != expected.shape
        or observed.dtype != expected.dtype
        or observed.tobytes(order="C") != expected.tobytes(order="C")
    ):
        raise ValueError(f"saved {label} schedule disagrees with recovered candidate")


def _unique_row_index(
    candidates: NDArray[np.generic],
    schedule: NDArray[np.generic],
    *,
    label: str,
) -> int:
    matches = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.dtype == schedule.dtype
        and candidate.tobytes(order="C") == schedule.tobytes(order="C")
    ]
    if len(matches) != 1:
        raise ValueError(f"saved {label} schedule must match exactly one candidate")
    return matches[0]


def _unique_bitwise_index(
    grid: NDArray[np.generic], value: np.generic, *, label: str
) -> int:
    scalar = np.asarray(value)
    matches = [
        index
        for index, candidate in enumerate(grid)
        if np.asarray(candidate).dtype == scalar.dtype
        and np.asarray(candidate).tobytes(order="C") == scalar.tobytes(order="C")
    ]
    if len(matches) != 1:
        raise ValueError(f"saved {label} radius must match exactly one grid point")
    return matches[0]


def _reject_endpoint(label: str, indices: tuple[int, ...], count: int) -> None:
    if any(index in {0, count - 1} for index in indices):
        raise ValueError(f"{label} selected a grid endpoint")


def _require_finite(label: str, values: NDArray[np.generic]) -> None:
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{label} must be finite")


def _read_only(values: NDArray[np.generic]) -> NDArray[np.floating]:
    frozen = np.array(values, copy=True)
    frozen.flags.writeable = False
    return frozen


def _json_numbers(values: object) -> str:
    array = np.asarray(values).reshape(-1)
    converted = [int(value) if np.issubdtype(array.dtype, np.integer) else float(value) for value in array]
    return json.dumps(converted, separators=(",", ":"))
