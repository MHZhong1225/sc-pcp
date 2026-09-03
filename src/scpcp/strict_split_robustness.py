"""Theory-aligned strict-split robustness utilities for marginal SC-PCP.

This module does not define another paper method.  It applies the unchanged
committed-prefix selector to one D_COT-frozen grid under two information sets:
the canonical D_COT union D_cert calibration sample, and D_cert alone.  The
claim boundary remains asymptotic per-step marginal coverage.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import yaml

from scpcp.data import TrajectoryBatch, concatenate_trajectories
from scpcp.marginal_prefix import MarginalPrefixSelection, select_marginal_prefix_schedule


PROTOCOL = "strict_split_robustness_v1"
VARIANTS = ("canonical", "strict")
SETTINGS = ("synthetic_main", "mimic_iv", "controlled_gamma_minus_2")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "strict_split_robustness.yaml"

FROZEN_CONFIG: dict[str, Any] = {
    "protocol": PROTOCOL,
    "role": "theory_aligned_robustness_only",
    "canonical_method_changed": False,
    "post_hoc_upgrade_rule": "none",
    "guarantee_scope": "asymptotic_per_step_marginal",
    "variants": {
        "canonical": {
            "grid_role": "D_COT",
            "selection_roles": ["D_COT", "D_cert"],
        },
        "strict": {
            "grid_role": "D_COT",
            "selection_roles": ["D_cert"],
        },
    },
    "settings": {
        "synthetic_main": {
            "config": "configs/per_step_synthetic_tail_shift.yaml",
            "seeds": {"start": 1000, "stop": 1100, "step": 1},
            "seed_status": "paired_reuse_of_frozen_main_seed_bank",
            "evaluation_stream": "paper_seed(base_seed, 900001)",
        },
        "mimic_iv": {
            "config": "configs/per_step_mimic_iv.yaml",
            "seeds": {"start": 0, "stop": 20, "step": 1},
            "seed_status": "paired_reuse_of_frozen_main_seed_bank",
            "evaluation_stream": "paper_seed(base_seed, 900001)",
        },
        "controlled_gamma_minus_2": {
            "config": "configs/per_step_mimic_iv.yaml",
            "seeds": {"start": 99000, "stop": 99200, "step": 10},
            "seed_status": "fresh_reserved_base_seed_bank",
            "gamma": -2.0,
            "horizon": 12,
            "calibration_trajectories": 3000,
            "grid_trajectories": 1000,
            "certification_trajectories": 2000,
            "reference_trajectories": 20000,
            "q_low_source_quantile": 0.80,
            "q_high_source_quantile": 0.95,
            "alternative_policy_tilt": 20.0,
            "maximum_policy_response": 1.0,
            "policy_ratio_cap": 3.0,
            "calibration_stream": "paper_seed(base_seed, 1700101)",
            "reference_stream": "paper_seed(base_seed, 1700401)",
        },
    },
    "summary": {
        "bootstrap_resamples": 10000,
        "bootstrap_rng": 99900,
        "bootstrap_unit": "complete_seed_stage_vector",
        "primary_coverage_metric": "min_t mean_seed(C_seed_t)",
        "coverage_conditioning": "successful_selection",
        "paired_comparisons": "joint_available_seeds",
    },
    "parent_formal_snapshot": {
        "manifest": "results/work/formal_source_snapshot_7665dfbe_20260825.manifest.json",
        "manifest_sha256": "e6a1bba7f3be47d39357f212824e7720262e7d5212a14628e3b8981088c64e24",
        "archive": "results/work/formal_source_snapshot_7665dfbe_20260825.tar.gz",
        "archive_sha256": "2116b9929240d8a25092f3c9015c362957bc963532930e5e720dc7ec78b2ea0b",
        "archive_bytes": 2036776,
        "source_tree_sha256": "7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643",
    },
}


def load_frozen_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the protocol and reject every scientific-field modification."""

    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"cannot load strict-split protocol: {path}") from error
    if payload != FROZEN_CONFIG:
        raise RuntimeError("strict-split scientific protocol differs from frozen v1")
    return payload


def setting_seeds(config: dict[str, Any], setting: str) -> tuple[int, ...]:
    if setting not in SETTINGS:
        raise ValueError(f"unknown strict-split setting: {setting}")
    seed_spec = config["settings"][setting]["seeds"]
    return tuple(range(seed_spec["start"], seed_spec["stop"], seed_spec["step"]))


@torch.no_grad()
def select_strict_split_pair(
    *,
    cot_batch: TrajectoryBatch,
    cot_scores: Tensor,
    certification_batch: TrajectoryBatch,
    certification_scores: Tensor,
    stage_grids: Tensor,
    target_policy: object,
    logging_policy: object,
    outcome_model: object,
    outcome_sd: Tensor,
    target: float,
) -> dict[str, MarginalPrefixSelection]:
    """Run the unchanged selector twice on one physically shared frozen grid."""

    if cot_scores.shape != cot_batch.actions.shape:
        raise ValueError("D_COT scores must match D_COT actions")
    if certification_scores.shape != certification_batch.actions.shape:
        raise ValueError("D_cert scores must match D_cert actions")
    if cot_batch.horizon != certification_batch.horizon:
        raise ValueError("D_COT and D_cert must have the same horizon")
    if stage_grids.shape[0] != cot_batch.horizon:
        raise ValueError("stage grids must have one row per stage")
    overlap = torch.isin(
        cot_batch.patient_ids.detach().cpu(),
        certification_batch.patient_ids.detach().cpu(),
    )
    if bool(overlap.any()):
        raise RuntimeError("D_COT and D_cert patient identifiers overlap")

    canonical_batch = concatenate_trajectories(cot_batch, certification_batch)
    canonical_scores = torch.cat((cot_scores, certification_scores), dim=0)
    common = {
        "stage_grids": stage_grids.to(canonical_scores),
        "target_policy": target_policy,
        "logging_policy": logging_policy,
        "outcome_model": outcome_model,
        "outcome_sd": outcome_sd,
        "target": target,
    }
    canonical = select_marginal_prefix_schedule(
        canonical_batch,
        canonical_scores,
        **common,
    )
    strict = select_marginal_prefix_schedule(
        certification_batch,
        certification_scores,
        **{
            **common,
            "stage_grids": stage_grids.to(certification_scores),
        },
    )
    return {"canonical": canonical, "strict": strict}


def selection_payload(
    selection: MarginalPrefixSelection,
    *,
    calibration_trajectories: int,
    calibration_roles: tuple[str, ...],
) -> dict[str, Any]:
    """Serialize selection and ESS diagnostics without nonstandard JSON NaNs."""

    if calibration_trajectories < 1:
        raise ValueError("calibration_trajectories must be positive")
    selected_fraction = selection.effective_sample_size / calibration_trajectories
    candidate_fraction = (
        selection.candidate_effective_sample_size / calibration_trajectories
    )
    return {
        "selection_available": selection.selection_available,
        "calibration_roles": list(calibration_roles),
        "calibration_trajectories": calibration_trajectories,
        "radii": _tensor_vector_or_empty(selection.radii),
        "selected_indices": list(selection.selected_indices),
        "selected_endpoint": selection.selected_endpoint,
        "failure_stage": selection.failure_stage,
        "estimated_coverage": _tensor_vector(selection.estimated_coverage),
        "estimated_normalized_width": _tensor_vector(
            selection.estimated_normalized_width
        ),
        "selected_ess": _tensor_vector(selection.effective_sample_size),
        "selected_ess_fraction": _tensor_vector(selected_fraction),
        "selected_minimum_ess_fraction": _tensor_min_or_none(selected_fraction),
        "candidate_ess": _tensor_matrix(selection.candidate_effective_sample_size),
        "candidate_ess_fraction": _tensor_matrix(candidate_fraction),
        "candidate_minimum_ess_fraction": _tensor_min_or_none(candidate_fraction),
        "maximum_raw_log_weight": _tensor_vector(
            selection.maximum_raw_log_weight
        ),
        "raw_log_weight_span": _tensor_vector(selection.raw_log_weight_span),
    }


def evaluation_payload(
    *,
    coverage: Tensor | list[float] | None,
    normalized_width_by_stage: Tensor | list[float] | None,
    evaluation_trajectories: int,
    evaluation_rng: int,
) -> dict[str, Any]:
    """Normalize fresh-evaluation output shared by main and controlled paths."""

    if coverage is None or normalized_width_by_stage is None:
        return {
            "evaluated": False,
            "evaluation_trajectories": 0,
            "evaluation_rng": evaluation_rng,
            "coverage_by_stage": [],
            "normalized_width_by_stage": [],
            "mean_normalized_width": None,
        }
    coverage_array = torch.as_tensor(coverage, dtype=torch.float64)
    width_array = torch.as_tensor(normalized_width_by_stage, dtype=torch.float64)
    if (
        coverage_array.ndim != 1
        or width_array.shape != coverage_array.shape
        or not bool(torch.isfinite(coverage_array).all())
        or not bool(torch.isfinite(width_array).all())
        or bool(((coverage_array < 0.0) | (coverage_array > 1.0)).any())
        or bool((width_array <= 0.0).any())
    ):
        raise RuntimeError("fresh evaluation metrics are malformed")
    return {
        "evaluated": True,
        "evaluation_trajectories": evaluation_trajectories,
        "evaluation_rng": evaluation_rng,
        "coverage_by_stage": _tensor_vector(coverage_array),
        "normalized_width_by_stage": _tensor_vector(width_array),
        "mean_normalized_width": float(width_array.mean().item()),
    }


def summarize_setting(
    rows: list[dict[str, Any]],
    *,
    setting: str,
    seeds: tuple[int, ...],
    bootstrap_resamples: int,
    bootstrap_rng: int,
) -> dict[str, Any]:
    """Compute WSC and paired seed-vector uncertainty without changing metrics."""

    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    ordered = sorted(rows, key=lambda row: int(row["seed"]))
    if tuple(int(row["seed"]) for row in ordered) != seeds:
        raise RuntimeError(f"{setting} rows do not match the prespecified seed bank")
    for row in ordered:
        validate_result_row(row, setting=setting)
    horizon = int(ordered[0]["horizon"])
    uniforms = np.random.default_rng(bootstrap_rng).random(
        size=(bootstrap_resamples, len(seeds))
    )

    arrays: dict[str, dict[str, np.ndarray]] = {}
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        available = np.asarray(
            [row["variants"][variant]["selection_available"] for row in ordered],
            dtype=bool,
        )
        coverage = np.full((len(seeds), horizon), np.nan, dtype=np.float64)
        width = np.full(len(seeds), np.nan, dtype=np.float64)
        selected_ess = np.full(len(seeds), np.nan, dtype=np.float64)
        candidate_ess = np.full(len(seeds), np.nan, dtype=np.float64)
        for index, row in enumerate(ordered):
            payload = row["variants"][variant]
            if not available[index]:
                continue
            coverage[index] = np.asarray(
                payload["evaluation"]["coverage_by_stage"], dtype=np.float64
            )
            width[index] = float(payload["evaluation"]["mean_normalized_width"])
            selected_ess[index] = float(payload["selected_minimum_ess_fraction"])
            candidate_ess[index] = float(payload["candidate_minimum_ess_fraction"])
        arrays[variant] = {
            "available": available,
            "coverage": coverage,
            "width": width,
            "selected_ess": selected_ess,
            "candidate_ess": candidate_ess,
        }
        variants[variant] = _variant_summary(
            arrays[variant],
            uniforms=uniforms,
            total_seeds=len(seeds),
        )

    paired = _paired_summary(
        arrays["canonical"],
        arrays["strict"],
        uniforms=uniforms,
    )
    return {
        "protocol": PROTOCOL,
        "setting": setting,
        "seeds": list(seeds),
        "primary_metric": "min_t mean_seed(target_coverage_seed_t)",
        "coverage_conditioning": "successful_selection",
        "selection_rate_denominator": "all_prespecified_seeds",
        "ess_conditioning": "successful_selection",
        "paired_conditioning": "joint_available_seeds_except_availability",
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "rng": bootstrap_rng,
            "unit": "complete_seed_stage_vector",
            "shared_uniforms_across_variants": True,
        },
        "variants": variants,
        "paired_strict_vs_canonical": paired,
        "claim_boundary": (
            "robustness-only comparison of information splits; canonical SC-PCP "
            "is unchanged and no finite-sample or post-hoc upgrade claim is made"
        ),
    }


def validate_result_row(row: dict[str, Any], *, setting: str | None = None) -> None:
    """Fail closed on malformed, mislabeled, or unmatched per-seed output."""

    if row.get("protocol") != PROTOCOL:
        raise RuntimeError("strict-split row protocol differs")
    if setting is not None and row.get("setting") != setting:
        raise RuntimeError("strict-split row setting differs")
    if not isinstance(row.get("seed"), int) or not isinstance(row.get("horizon"), int):
        raise RuntimeError("strict-split row seed/horizon is malformed")
    if row["horizon"] < 1 or set(row.get("variants", {})) != set(VARIANTS):
        raise RuntimeError("strict-split row variant family is malformed")
    if row.get("stage_grid_roles") != ["D_COT"]:
        raise RuntimeError("strict-split row did not use the frozen D_COT grid")
    grid_hash = row.get("stage_grid_sha256")
    if not isinstance(grid_hash, str) or len(grid_hash) != 64:
        raise RuntimeError("strict-split row grid fingerprint is malformed")
    grid_shape = row.get("stage_grid_shape")
    if (
        not isinstance(grid_shape, list)
        or len(grid_shape) != 2
        or grid_shape[0] != row["horizon"]
        or not isinstance(grid_shape[1], int)
        or grid_shape[1] < 2
    ):
        raise RuntimeError("strict-split row grid shape is malformed")
    if row.get("matched_evaluation_crn") is not True:
        raise RuntimeError("strict-split variants did not declare matched evaluation CRN")
    split_sizes = row.get("split_sizes")
    if not isinstance(split_sizes, dict):
        raise RuntimeError("strict-split row split sizes are malformed")
    cot_size = split_sizes.get("D_COT")
    certification_size = split_sizes.get("D_cert")
    if (
        not isinstance(cot_size, int)
        or not isinstance(certification_size, int)
        or cot_size < 1
        or certification_size < 1
    ):
        raise RuntimeError("strict-split D_COT/D_cert sizes are malformed")
    evaluation_rngs = set()
    for variant in VARIANTS:
        payload = row["variants"][variant]
        expected_roles = ["D_COT", "D_cert"] if variant == "canonical" else ["D_cert"]
        if payload.get("calibration_roles") != expected_roles:
            raise RuntimeError(f"{variant} selection roles differ")
        available = payload.get("selection_available")
        evaluation = payload.get("evaluation")
        if not isinstance(available, bool) or not isinstance(evaluation, dict):
            raise RuntimeError(f"{variant} availability/evaluation is malformed")
        if bool(evaluation.get("evaluated")) != available:
            raise RuntimeError(f"{variant} evaluation availability differs")
        expected_calibration = (
            cot_size + certification_size
            if variant == "canonical"
            else certification_size
        )
        if payload.get("calibration_trajectories") != expected_calibration:
            raise RuntimeError(f"{variant} calibration sample size differs")
        evaluation_rng = evaluation.get("evaluation_rng")
        if not isinstance(evaluation_rng, int):
            raise RuntimeError(f"{variant} evaluation RNG is malformed")
        evaluation_rngs.add(evaluation_rng)
        if available:
            horizon = row["horizon"]
            for field in ("radii", "selected_indices", "selected_ess_fraction"):
                if len(payload.get(field, ())) != horizon:
                    raise RuntimeError(f"{variant} {field} does not span the horizon")
            if len(evaluation.get("coverage_by_stage", ())) != horizon:
                raise RuntimeError(f"{variant} coverage does not span the horizon")
            if len(evaluation.get("normalized_width_by_stage", ())) != horizon:
                raise RuntimeError(f"{variant} width does not span the horizon")
            radii = np.asarray(payload["radii"], dtype=np.float64)
            indices = payload["selected_indices"]
            coverage = np.asarray(
                evaluation["coverage_by_stage"], dtype=np.float64
            )
            width = np.asarray(
                evaluation["normalized_width_by_stage"], dtype=np.float64
            )
            if (
                not np.isfinite(radii).all()
                or np.any(radii <= 0.0)
                or not all(
                    isinstance(index, int) and 0 <= index < grid_shape[1]
                    for index in indices
                )
                or not np.isfinite(coverage).all()
                or np.any((coverage < 0.0) | (coverage > 1.0))
                or not np.isfinite(width).all()
                or np.any(width <= 0.0)
            ):
                raise RuntimeError(f"{variant} selected/evaluation values are malformed")
            expected_evaluation = (
                20_000 if row["setting"] == "controlled_gamma_minus_2" else 50_000
            )
            if evaluation.get("evaluation_trajectories") != expected_evaluation:
                raise RuntimeError(f"{variant} evaluation budget differs")
            for field in (
                "selected_minimum_ess_fraction",
                "candidate_minimum_ess_fraction",
            ):
                value = payload.get(field)
                if not isinstance(value, (int, float)) or not 0.0 < value <= 1.0:
                    raise RuntimeError(f"{variant} {field} is malformed")
        elif evaluation.get("coverage_by_stage") or evaluation.get("normalized_width_by_stage"):
            raise RuntimeError(f"{variant} unavailable evaluation contains metrics")
    if len(evaluation_rngs) != 1 or row.get("evaluation_rng") not in evaluation_rngs:
        raise RuntimeError("strict-split variants do not share the exact evaluation RNG")
    if row["setting"] == "controlled_gamma_minus_2":
        if row.get("gamma") != -2.0 or (cot_size, certification_size) != (1000, 2000):
            raise RuntimeError("controlled strict-split mechanism/budget differs")


def tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _variant_summary(
    values: dict[str, np.ndarray],
    *,
    uniforms: np.ndarray,
    total_seeds: int,
) -> dict[str, Any]:
    mask = values["available"]
    count = int(mask.sum())
    base = {
        "selected_seeds": count,
        "total_seeds": total_seeds,
        "selection_rate": count / total_seeds,
        "selection_rate_ci95": _wilson_interval(count, total_seeds),
    }
    if count == 0:
        return {
            **base,
            "target_marginal_worst_coverage": None,
            "target_wsc_ci95": [None, None],
            "target_coverage_by_stage": [],
            "mean_normalized_width": None,
            "mean_normalized_width_ci95": [None, None],
            "mean_selected_minimum_ess_fraction": None,
            "mean_selected_minimum_ess_fraction_ci95": [None, None],
            "worst_selected_minimum_ess_fraction": None,
            "mean_candidate_minimum_ess_fraction": None,
            "mean_candidate_minimum_ess_fraction_ci95": [None, None],
            "worst_candidate_minimum_ess_fraction": None,
        }
    index = _bootstrap_indices(uniforms, count)
    coverage = values["coverage"][mask]
    width = values["width"][mask]
    selected_ess = values["selected_ess"][mask]
    candidate_ess = values["candidate_ess"][mask]
    stage_coverage = coverage.mean(axis=0)
    wsc_draws = coverage[index].mean(axis=1).min(axis=1)
    return {
        **base,
        "target_marginal_worst_coverage": float(stage_coverage.min()),
        "target_wsc_ci95": _percentile_interval(wsc_draws),
        "target_worst_stage_zero_based": int(stage_coverage.argmin()),
        "target_coverage_by_stage": stage_coverage.tolist(),
        "mean_normalized_width": float(width.mean()),
        "mean_normalized_width_ci95": _percentile_interval(
            width[index].mean(axis=1)
        ),
        "mean_selected_minimum_ess_fraction": float(selected_ess.mean()),
        "mean_selected_minimum_ess_fraction_ci95": _percentile_interval(
            selected_ess[index].mean(axis=1)
        ),
        "worst_selected_minimum_ess_fraction": float(selected_ess.min()),
        "mean_candidate_minimum_ess_fraction": float(candidate_ess.mean()),
        "mean_candidate_minimum_ess_fraction_ci95": _percentile_interval(
            candidate_ess[index].mean(axis=1)
        ),
        "worst_candidate_minimum_ess_fraction": float(candidate_ess.min()),
    }


def _paired_summary(
    canonical: dict[str, np.ndarray],
    strict: dict[str, np.ndarray],
    *,
    uniforms: np.ndarray,
) -> dict[str, Any]:
    availability_difference = (
        strict["available"].astype(np.float64)
        - canonical["available"].astype(np.float64)
    )
    full_index = _bootstrap_indices(uniforms, len(availability_difference))
    joint = canonical["available"] & strict["available"]
    count = int(joint.sum())
    result: dict[str, Any] = {
        "strict_minus_canonical_availability": float(availability_difference.mean()),
        "strict_minus_canonical_availability_ci95": _percentile_interval(
            availability_difference[full_index].mean(axis=1)
        ),
        "joint_available_seeds": count,
    }
    if count == 0:
        return {
            **result,
            "strict_minus_canonical_wsc": None,
            "strict_minus_canonical_wsc_ci95": [None, None],
            "strict_minus_canonical_mean_width": None,
            "strict_minus_canonical_mean_width_ci95": [None, None],
            "strict_to_canonical_geometric_width_ratio": None,
            "strict_to_canonical_geometric_width_ratio_ci95": [None, None],
            "strict_minus_canonical_selected_minimum_ess_fraction": None,
            "strict_minus_canonical_selected_minimum_ess_fraction_ci95": [None, None],
            "strict_minus_canonical_candidate_minimum_ess_fraction": None,
            "strict_minus_canonical_candidate_minimum_ess_fraction_ci95": [None, None],
        }
    index = _bootstrap_indices(uniforms, count)
    canonical_coverage = canonical["coverage"][joint]
    strict_coverage = strict["coverage"][joint]
    canonical_stage_draws = canonical_coverage[index].mean(axis=1)
    strict_stage_draws = strict_coverage[index].mean(axis=1)
    wsc_draws = strict_stage_draws.min(axis=1) - canonical_stage_draws.min(axis=1)
    wsc_difference = (
        strict_coverage.mean(axis=0).min()
        - canonical_coverage.mean(axis=0).min()
    )
    width_difference = strict["width"][joint] - canonical["width"][joint]
    log_width_ratio = np.log(strict["width"][joint] / canonical["width"][joint])
    selected_ess_difference = (
        strict["selected_ess"][joint] - canonical["selected_ess"][joint]
    )
    candidate_ess_difference = (
        strict["candidate_ess"][joint] - canonical["candidate_ess"][joint]
    )
    return {
        **result,
        "strict_minus_canonical_wsc": float(wsc_difference),
        "strict_minus_canonical_wsc_ci95": _percentile_interval(wsc_draws),
        "strict_minus_canonical_mean_width": float(width_difference.mean()),
        "strict_minus_canonical_mean_width_ci95": _percentile_interval(
            width_difference[index].mean(axis=1)
        ),
        "strict_to_canonical_geometric_width_ratio": float(
            np.exp(log_width_ratio.mean())
        ),
        "strict_to_canonical_geometric_width_ratio_ci95": _percentile_interval(
            np.exp(log_width_ratio[index].mean(axis=1))
        ),
        "strict_minus_canonical_selected_minimum_ess_fraction": float(
            selected_ess_difference.mean()
        ),
        "strict_minus_canonical_selected_minimum_ess_fraction_ci95": (
            _percentile_interval(selected_ess_difference[index].mean(axis=1))
        ),
        "strict_minus_canonical_candidate_minimum_ess_fraction": float(
            candidate_ess_difference.mean()
        ),
        "strict_minus_canonical_candidate_minimum_ess_fraction_ci95": (
            _percentile_interval(candidate_ess_difference[index].mean(axis=1))
        ),
    }


def _bootstrap_indices(uniforms: np.ndarray, sample_size: int) -> np.ndarray:
    if sample_size < 1 or sample_size > uniforms.shape[1]:
        raise ValueError("bootstrap sample size is outside the shared uniform matrix")
    return np.floor(uniforms[:, :sample_size] * sample_size).astype(np.int64)


def _percentile_interval(values: np.ndarray) -> list[float]:
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("bootstrap draws must be one finite vector")
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("invalid Wilson inputs")
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _tensor_vector(value: Tensor) -> list[float]:
    if value.ndim != 1 or not bool(torch.isfinite(value).all()):
        raise RuntimeError("expected one finite tensor vector")
    return [float(item) for item in value.detach().cpu().tolist()]


def _tensor_vector_or_empty(value: Tensor | None) -> list[float]:
    return [] if value is None else _tensor_vector(value)


def _tensor_matrix(value: Tensor) -> list[list[float]]:
    if value.ndim != 2 or not bool(torch.isfinite(value).all()):
        raise RuntimeError("expected one finite tensor matrix")
    return [
        [float(item) for item in row]
        for row in value.detach().cpu().tolist()
    ]


def _tensor_min_or_none(value: Tensor) -> float | None:
    if value.numel() == 0:
        return None
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("ESS diagnostics must be finite")
    result = float(value.min().item())
    if not 0.0 < result <= 1.0 + 1e-10:
        raise RuntimeError("ESS fraction is outside (0, 1]")
    return min(result, 1.0)
