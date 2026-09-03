"""Frozen, coverage-blind clinical-v4 repair contract.

The contract inherits two already-passing clinical settings from the completed
v3 development bundle and searches transition simulators independently for
eICU and MIMIC-CXR.  It contains no paper-method or scientific-result logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import tarfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


PROTOCOL = "controlled_clinical_fidelity_v4"
SELECTOR_VERSION = "controlled_clinical_fidelity_v4_dataset_selector_v1"
DATASETS = ("mimic_iv", "eicu", "inspire", "mimic_cxr")
ANCHOR_DATASETS = ("mimic_iv", "inspire")
REPAIR_DATASETS = ("eicu", "mimic_cxr")

METRIC_ORDER = ("raw", "stagewise_zscore")
NEIGHBOR_ORDER = (200, 10_000)
WEIGHT_ORDER = ("gaussian_b2", "uniform")
TRANSITION_ORDER = ("ridge_residual", "local_delta")
OUTCOME_RESIDUAL_ORDER = {
    "eicu": ("standardized",),
    "mimic_cxr": ("standardized", "raw"),
}
RIDGE_BY_DATASET = {
    "eicu": ("sample_normalized_no_intercept", 1e-3),
    "mimic_cxr": ("sample_normalized_no_intercept", 1e-3),
}
METRIC_THRESHOLDS = {
    "maximum_score_ks": 0.10,
    "maximum_signed_residual_w1": 0.25,
    "maximum_successor_mean_w1": 0.25,
    "maximum_successor_q95_w1": 0.50,
}
DEVELOPMENT_MINIMUM_PASS_COUNT = 19
CONTROLLED_TRANSITION_DEFAULTS = {
    "transition_mode": "ridge_residual",
    "outcome_residual_mode": "standardized",
}
ARCHIVED_V3_CONTROLLED_TRANSITION_SHA256 = (
    "ba878ad3ae89a9a3aef84734a6347f321baf3fce84f7ec13089ff4c41e0a4fd0"
)
CURRENT_CONTROLLED_TRANSITION_SHA256 = (
    "e6b433cda750fa745893eafafa429bbacfc504f603585b02ae4d68105c6a8a70"
)

_PARENT_FILES = {
    "FINAL_STATUS.json": (
        "cd7cfa7ec21814c01a847e24f3b706beb0a70916f8d75754a770b6b2e3bc1308"
    ),
    "manifest.json": (
        "9d01274c652a6eab3c13cdfdc62b6b79114ef8878ad3d0627b95e7812bca8d6f"
    ),
    "COMPLETE": (
        "c2278657ed61e7d8a182f2e0424c2cb2f63a27f15f139269ec520a65b8492ccf"
    ),
}
_DEVELOPMENT_REUSE_AUDIT = {
    "role": "exact_authorized_v3_k0_lineage_reuse",
    "base_seed_count": 40,
    "stream_count": 180,
    "mapping_sha256": (
        "c5d2f96cd8b33339b9abfb2bc572c61bf183048f8b27ff331ca8651a364234e3"
    ),
    "rng_id_set_sha256": (
        "c75733a9a8d2e69122804e1b2800e4f4f7ab7a51d60ebb37e993ea1f527dcbff"
    ),
    "parent_seed_envelope_sha256": (
        "2547b7f476b1fd9aa05809860e9729b51dc3b8a22d6adfac7f70cec6e5395946"
    ),
    "required_authorized_lineage_collision_count": 180,
    "required_missing_lineage_collision_count": 0,
    "required_unauthorized_collision_count": 0,
    "common_random_numbers_across_candidates": True,
    "scientific_freshness_claimed": False,
}


@dataclass(frozen=True)
class FrozenAnchor:
    """A v3 setting inherited without any v4 selection."""

    dataset: str
    source_json_pointer: str
    pass_count_json_pointer: str
    theta_id: str
    stage_a_id: str
    metric: str
    neighbors: int
    weight: str
    ridge: str
    ridge_mode: str
    ridge_value: float
    penalize_intercept: bool
    transition_mode: str
    outcome_residual_mode: str
    development_pass_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_json_pointer": self.source_json_pointer,
            "pass_count_json_pointer": self.pass_count_json_pointer,
            "theta_id": self.theta_id,
            "stage_a_id": self.stage_a_id,
            "metric": self.metric,
            "neighbors": self.neighbors,
            "weight": self.weight,
            "bandwidth": 2.0,
            "donor_weighting": "gaussian",
            "ridge": self.ridge,
            "ridge_mode": self.ridge_mode,
            "ridge_value": self.ridge_value,
            "penalize_intercept": self.penalize_intercept,
            "transition_mode": self.transition_mode,
            "outcome_residual_mode": self.outcome_residual_mode,
            "development_pass_count": self.development_pass_count,
        }


FROZEN_ANCHORS = {
    "mimic_iv": FrozenAnchor(
        dataset="mimic_iv",
        source_json_pointer="/shared_stage_b/winner",
        pass_count_json_pointer="/shared_pass_counts/mimic_iv",
        theta_id="A03_raw_k200_gaussian_b2__normalized_ridge_1e-2",
        stage_a_id="A03_raw_k200_gaussian_b2",
        metric="raw",
        neighbors=200,
        weight="gaussian_b2",
        ridge="normalized_ridge_1e-2",
        ridge_mode="sample_normalized_no_intercept",
        ridge_value=1e-2,
        penalize_intercept=False,
        transition_mode="ridge_residual",
        outcome_residual_mode="standardized",
        development_pass_count=20,
    ),
    "inspire": FrozenAnchor(
        dataset="inspire",
        source_json_pointer="/dataset_stage_b_fallbacks/inspire/winner",
        pass_count_json_pointer="/fallback_pass_counts/inspire",
        theta_id="A03_raw_k200_gaussian_b2__raw_ridge_1e-3",
        stage_a_id="A03_raw_k200_gaussian_b2",
        metric="raw",
        neighbors=200,
        weight="gaussian_b2",
        ridge="raw_ridge_1e-3",
        ridge_mode="v2_raw",
        ridge_value=1e-3,
        penalize_intercept=True,
        transition_mode="ridge_residual",
        outcome_residual_mode="standardized",
        development_pass_count=20,
    ),
}


@dataclass(frozen=True)
class RepairTheta:
    """One dataset-specific transition-repair candidate."""

    dataset: str
    candidate_id: str
    metric: str
    neighbors: int
    weight: str
    transition_mode: str
    outcome_residual_mode: str

    def __post_init__(self) -> None:
        if self.dataset not in REPAIR_DATASETS:
            raise ValueError(f"repair candidates are not defined for {self.dataset!r}")
        axes = _candidate_axes(self.dataset)
        values = (
            self.metric,
            self.neighbors,
            self.weight,
            self.transition_mode,
            self.outcome_residual_mode,
        )
        try:
            index = axes.index(values)
        except ValueError as error:
            raise ValueError("repair candidate lies outside the frozen dataset grid") from error
        prefix = "E" if self.dataset == "eicu" else "C"
        expected_id = (
            f"{prefix}{index:02d}_{self.metric}_k{self.neighbors}_{self.weight}_"
            f"{self.transition_mode}_{self.outcome_residual_mode}"
        )
        if self.candidate_id != expected_id:
            raise ValueError("repair candidate ID does not match its frozen parameters")

    @property
    def bandwidth(self) -> float:
        return 2.0

    @property
    def donor_weighting(self) -> str:
        return "gaussian" if self.weight == "gaussian_b2" else "uniform"

    @property
    def ridge_mode(self) -> str:
        return RIDGE_BY_DATASET[self.dataset][0]

    @property
    def ridge_value(self) -> float:
        return RIDGE_BY_DATASET[self.dataset][1]

    @property
    def uses_full_cell(self) -> bool:
        return self.neighbors == 10_000

    @property
    def minimal_change_tuple(self) -> tuple[int, ...]:
        reference = {
            "eicu": ("raw", 200, "uniform", "ridge_residual", "standardized"),
            "mimic_cxr": (
                "raw",
                200,
                "gaussian_b2",
                "ridge_residual",
                "standardized",
            ),
        }[self.dataset]
        values = (
            self.metric,
            self.neighbors,
            self.weight,
            self.transition_mode,
            self.outcome_residual_mode,
        )
        changes = tuple(int(value != anchor) for value, anchor in zip(values, reference))
        return (sum(changes), *changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "dataset": self.dataset,
            "metric": self.metric,
            "neighbors": self.neighbors,
            "uses_full_cell": self.uses_full_cell,
            "full_cell_neighbor_sentinel": 10_000,
            "weight": self.weight,
            "bandwidth": self.bandwidth,
            "donor_weighting": self.donor_weighting,
            "transition_mode": self.transition_mode,
            "outcome_residual_mode": self.outcome_residual_mode,
            "ridge_mode": self.ridge_mode,
            "ridge_value": self.ridge_value,
            "penalize_intercept": False,
            "minimal_change_tuple": list(self.minimal_change_tuple),
        }


@dataclass(frozen=True)
class ParentV3Binding:
    root: Path
    file_sha256: Mapping[str, str]


@dataclass(frozen=True)
class FidelityV4Config:
    parent_v3: ParentV3Binding
    development_seeds: Mapping[str, tuple[int, ...]]
    development_reuse_audit: Mapping[str, Any]
    confirmation_seeds: Mapping[str, tuple[int, ...]]
    confirmation_bootstrap_seeds: Mapping[str, int]
    confirmation_mapping_sha256: str
    stagewise_sd_floor: float
    archived_v3_controlled_transition_sha256: str
    current_controlled_transition_sha256: str

    def validate(self) -> None:
        if self.parent_v3.root != Path(
            "results/work/controlled_clinical_fidelity_v3_development"
        ):
            raise ValueError("v4 parent root differs from the completed v3 development")
        if dict(self.parent_v3.file_sha256) != _PARENT_FILES:
            raise ValueError("v4 parent artifact hashes differ from the frozen binding")
        expected_development = {
            "eicu": tuple(range(92_000, 92_200, 10)),
            "mimic_cxr": tuple(range(92_600, 92_800, 10)),
        }
        if dict(self.development_seeds) != expected_development:
            raise ValueError("v4 development must reuse exactly the v3 repair banks")
        if dict(self.development_reuse_audit) != _DEVELOPMENT_REUSE_AUDIT:
            raise ValueError("v4 development reuse audit binding differs")
        expected_confirmation = {
            "mimic_iv": tuple(range(115_000, 115_200, 10)),
            "eicu": tuple(range(116_000, 116_200, 10)),
            "inspire": tuple(range(117_000, 117_200, 10)),
            "mimic_cxr": tuple(range(118_000, 118_200, 10)),
        }
        if dict(self.confirmation_seeds) != expected_confirmation:
            raise ValueError("v4 confirmation seeds differ from the audited fresh banks")
        expected_bootstrap = {
            "mimic_iv": 11_500_019,
            "eicu": 11_600_019,
            "inspire": 11_700_019,
            "mimic_cxr": 11_800_019,
        }
        if dict(self.confirmation_bootstrap_seeds) != expected_bootstrap:
            raise ValueError("v4 confirmation bootstrap seeds differ")
        if self.confirmation_mapping_sha256 != (
            "3a78ec5afe69f57928de894a38803f5c369b33ab1db3f7c37bd403b974f75c72"
        ):
            raise ValueError("v4 confirmation RNG mapping binding differs")
        banks = tuple(self.confirmation_seeds.values())
        if any(len(seeds) != 20 or len(set(seeds)) != 20 for seeds in banks):
            raise ValueError("confirmation requires exactly 20 distinct seeds per dataset")
        if len(set().union(*map(set, banks))) != 80:
            raise ValueError("confirmation base-seed banks must be disjoint across datasets")
        if set().union(*map(set, banks)) & set().union(
            *map(set, self.development_seeds.values())
        ):
            raise ValueError("development and confirmation base seeds overlap")
        if self.stagewise_sd_floor != 1e-4:
            raise ValueError("stagewise representation sd floor must be 1e-4")
        if self.archived_v3_controlled_transition_sha256 != (
            ARCHIVED_V3_CONTROLLED_TRANSITION_SHA256
        ) or self.current_controlled_transition_sha256 != (
            CURRENT_CONTROLLED_TRANSITION_SHA256
        ):
            raise ValueError("controlled-transition default-parity binding differs")


@dataclass(frozen=True)
class K0CandidateSummary:
    candidate_id: str
    dataset: str
    pass_count: int
    structural_pass_count: int
    q95_seed_ratio: float
    mean_seed_ratio: float
    seed_ratios: tuple[float, ...]
    structural_pass_flags: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.dataset not in REPAIR_DATASETS:
            raise ValueError("K0 repair summary has an unknown dataset")
        if len(self.seed_ratios) != 20 or len(self.structural_pass_flags) != 20:
            raise ValueError("K0 repair summary requires exactly 20 seed records")
        if any(type(value) is not bool for value in self.structural_pass_flags):
            raise ValueError("K0 structural pass flags must be exact booleans")
        if any(math.isnan(value) or value < 0.0 for value in self.seed_ratios):
            raise ValueError("K0 seed ratios must be nonnegative and not NaN")
        expected_pass_count = sum(value <= 1.0 for value in self.seed_ratios)
        if self.pass_count != expected_pass_count:
            raise ValueError("K0 pass count does not match the seed ratios")
        if self.structural_pass_count != sum(self.structural_pass_flags):
            raise ValueError("K0 structural pass count does not match its flags")
        if any(
            not structural and math.isfinite(ratio)
            for structural, ratio in zip(
                self.structural_pass_flags,
                self.seed_ratios,
            )
        ):
            raise ValueError("K0 structural failures must have infinite seed ratios")
        finite = all(math.isfinite(value) for value in self.seed_ratios)
        expected_q95 = (
            float(
                np.quantile(
                    np.asarray(sorted(self.seed_ratios), dtype=np.float64),
                    0.95,
                    method="linear",
                )
            )
            if finite
            else math.inf
        )
        expected_mean = (
            float(np.mean(np.asarray(self.seed_ratios, dtype=np.float64)))
            if finite
            else math.inf
        )
        if not _same_float(self.q95_seed_ratio, expected_q95):
            raise ValueError("K0 q95 ratio does not match the seed ratios")
        if not _same_float(self.mean_seed_ratio, expected_mean):
            raise ValueError("K0 mean ratio does not match the seed ratios")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "dataset": self.dataset,
            "pass_count": self.pass_count,
            "structural_pass_count": self.structural_pass_count,
            "q95_seed_ratio": _finite_or_none(self.q95_seed_ratio),
            "mean_seed_ratio": _finite_or_none(self.mean_seed_ratio),
            "seed_ratios": [_finite_or_none(value) for value in self.seed_ratios],
            "structural_pass_flags": list(self.structural_pass_flags),
            "structural_failure_ratio_is_infinite": any(
                not math.isfinite(value) for value in self.seed_ratios
            ),
        }


def load_fidelity_v4_config(path: Path) -> FidelityV4Config:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        raise ValueError(f"protocol must be {PROTOCOL}")
    expected_top_level = {
        "protocol",
        "parent_v3",
        "datasets",
        "devices_required",
        "development",
        "anchors",
        "confirmation",
        "repair_grids",
        "representation",
        "default_parity",
        "k0_gate",
        "support_gate",
        "selection",
        "information_firewall",
    }
    if set(payload) != expected_top_level:
        raise ValueError("v4 config keys differ from the frozen schema")
    _validate_parent_section(payload["parent_v3"])
    if tuple(payload["datasets"]) != DATASETS or payload["devices_required"] != 2:
        raise ValueError("v4 dataset order or CUDA device count differs")
    _validate_development_section(payload["development"])
    expected_anchors = {name: FROZEN_ANCHORS[name].to_dict() for name in ANCHOR_DATASETS}
    if payload["anchors"] != expected_anchors:
        raise ValueError("v4 frozen anchors differ from the completed v3 evidence")
    _validate_confirmation_section(payload["confirmation"])
    _validate_grid_section(payload["repair_grids"])
    if payload["representation"] != {
        "stagewise_zscore": {
            "source": "D_env_only",
            "pooling": "per_stage_pooled_over_actions",
            "center_dtype": "float64",
            "scale_dtype": "float64",
            "population_sd": True,
            "sd_floor": 1e-4,
        }
    }:
        raise ValueError("v4 representation contract differs from v3")
    if payload["default_parity"] != {
        "controlled_transition_defaults": CONTROLLED_TRANSITION_DEFAULTS,
        "inherited_anchor_datasets": list(ANCHOR_DATASETS),
        "archived_v3_controlled_transition_sha256": (
            ARCHIVED_V3_CONTROLLED_TRANSITION_SHA256
        ),
        "current_controlled_transition_sha256": CURRENT_CONTROLLED_TRANSITION_SHA256,
        "required_bitwise_test": (
            "tests/per_step/test_controlled_transition.py::"
            "test_explicit_legacy_modes_match_the_default_bitwise"
        ),
    }:
        raise ValueError("v4 controlled-transition default parity differs")
    expected_k0 = {
        "systematic_replays": 16,
        **METRIC_THRESHOLDS,
        "minimum_available_seed_fraction": 0.95,
        "active_coordinate_sd_floor": 1e-4,
    }
    if payload["k0_gate"] != expected_k0:
        raise ValueError("v4 K0 thresholds differ from v3")
    if payload["support_gate"] != {
        "minimum_unique_patients_per_stage_action": 20,
        "minimum_available_seed_fraction": 0.95,
    }:
        raise ValueError("v4 support thresholds differ from v3")
    if payload["selection"] != {
        "version": SELECTOR_VERSION,
        "scope": "per_dataset_independent",
        "seed_ratio_quantile": 0.95,
        "seed_ratio_quantile_method": "linear",
        "order": [
            "maximize_pass_count",
            "minimize_q95_seed_ratio",
            "minimize_mean_seed_ratio",
            "minimize_change_from_v3",
            "candidate_id",
        ],
        "development_seed_count": 20,
        "development_minimum_pass_count": DEVELOPMENT_MINIMUM_PASS_COUNT,
        "development_required_structural_pass_count": 20,
        "development_gate_interpretation": "operational_not_confidence_interval",
        "failure": "DATASET_DEVELOPMENT_NO_GO",
        "cross_dataset_conjunction_permitted": False,
    }:
        raise ValueError("v4 selector contract differs")
    if payload["information_firewall"] != {
        "allowed": ["support", "k0_fidelity", "context_identity", "provenance"],
        "forbidden": ["science", "coverage", "width", "method_selection"],
        "scientific_outputs_permitted": False,
    }:
        raise ValueError("v4 information firewall differs")

    parent = payload["parent_v3"]
    confirmation = payload["confirmation"]
    representation = payload["representation"]["stagewise_zscore"]
    config = FidelityV4Config(
        parent_v3=ParentV3Binding(
            root=Path(parent["root"]),
            file_sha256=dict(parent["file_sha256"]),
        ),
        development_seeds=_seed_banks(
            payload["development"]["seeds"], REPAIR_DATASETS
        ),
        development_reuse_audit=dict(payload["development"]["rng_reuse_audit"]),
        confirmation_seeds=_seed_banks(confirmation["seeds"], DATASETS),
        confirmation_bootstrap_seeds={
            name: int(seed) for name, seed in confirmation["bootstrap_seeds"].items()
        },
        confirmation_mapping_sha256=confirmation["rng_audit"]["mapping_sha256"],
        stagewise_sd_floor=float(representation["sd_floor"]),
        archived_v3_controlled_transition_sha256=payload["default_parity"][
            "archived_v3_controlled_transition_sha256"
        ],
        current_controlled_transition_sha256=payload["default_parity"][
            "current_controlled_transition_sha256"
        ],
    )
    config.validate()
    return config


def repair_candidates(dataset: str) -> tuple[RepairTheta, ...]:
    if dataset not in REPAIR_DATASETS:
        raise ValueError(f"repair candidates are not defined for {dataset!r}")
    prefix = "E" if dataset == "eicu" else "C"
    candidates = []
    for index, values in enumerate(_candidate_axes(dataset)):
        metric, neighbors, weight, transition, outcome_residual = values
        candidates.append(
            RepairTheta(
                dataset=dataset,
                candidate_id=(
                    f"{prefix}{index:02d}_{metric}_k{neighbors}_{weight}_"
                    f"{transition}_{outcome_residual}"
                ),
                metric=metric,
                neighbors=neighbors,
                weight=weight,
                transition_mode=transition,
                outcome_residual_mode=outcome_residual,
            )
        )
    return tuple(candidates)


def normalized_seed_ratio(metrics: Mapping[str, Any]) -> float:
    if not isinstance(metrics, Mapping) or set(metrics) != {
        *METRIC_THRESHOLDS,
        "structural_invariants",
    }:
        raise ValueError("K0 metric payload differs from the exact schema")
    if metrics.get("structural_invariants") is not True:
        return math.inf
    try:
        ratios = [
            float(metrics[name]) / threshold
            for name, threshold in METRIC_THRESHOLDS.items()
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("K0 metric payload differs from the exact schema") from error
    if not all(math.isfinite(value) and value >= 0.0 for value in ratios):
        return math.inf
    return max(ratios)


def summarize_candidate_dataset(
    candidate: RepairTheta,
    metrics_by_seed: Sequence[Mapping[str, Any]],
) -> K0CandidateSummary:
    if len(metrics_by_seed) != 20:
        raise ValueError("candidate summary requires one full 20-seed dataset bank")
    ratios = tuple(normalized_seed_ratio(metrics) for metrics in metrics_by_seed)
    structural_flags = tuple(
        metrics["structural_invariants"] is True for metrics in metrics_by_seed
    )
    return summarize_seed_ratios(
        candidate,
        ratios,
        structural_pass_flags=structural_flags,
    )


def summarize_seed_ratios(
    candidate: RepairTheta,
    seed_ratios: Sequence[float],
    *,
    structural_pass_flags: Sequence[bool],
) -> K0CandidateSummary:
    ratios = tuple(float(value) for value in seed_ratios)
    structural_flags = tuple(structural_pass_flags)
    finite = all(math.isfinite(value) for value in ratios)
    q95 = (
        float(
            np.quantile(
                np.asarray(sorted(ratios), dtype=np.float64),
                0.95,
                method="linear",
            )
        )
        if finite and len(ratios) == 20
        else math.inf
    )
    mean = (
        float(np.mean(np.asarray(ratios, dtype=np.float64)))
        if finite and len(ratios) == 20
        else math.inf
    )
    return K0CandidateSummary(
        candidate_id=candidate.candidate_id,
        dataset=candidate.dataset,
        pass_count=sum(value <= 1.0 for value in ratios),
        structural_pass_count=sum(structural_flags),
        q95_seed_ratio=q95,
        mean_seed_ratio=mean,
        seed_ratios=ratios,
        structural_pass_flags=structural_flags,
    )


def select_dataset_candidate(
    dataset: str,
    candidates: Sequence[RepairTheta],
    summaries: Mapping[str, K0CandidateSummary],
) -> dict[str, Any]:
    """Select one repair without consulting any other dataset."""

    expected = repair_candidates(dataset)
    if tuple(candidates) != expected:
        raise ValueError("selector requires the complete frozen dataset grid in order")
    if tuple(summaries) != tuple(candidate.candidate_id for candidate in expected):
        raise ValueError("selector summaries differ from the frozen candidate order")
    scored: list[tuple[tuple[Any, ...], RepairTheta, K0CandidateSummary]] = []
    for candidate in expected:
        summary = summaries[candidate.candidate_id]
        if summary.dataset != dataset or summary.candidate_id != candidate.candidate_id:
            raise ValueError("selector summary identity differs from its candidate")
        objective = (
            -summary.pass_count,
            summary.q95_seed_ratio,
            summary.mean_seed_ratio,
            candidate.minimal_change_tuple,
            candidate.candidate_id,
        )
        scored.append((objective, candidate, summary))
    ordered = sorted(scored, key=lambda item: item[0])
    winner_objective, winner, winner_summary = ordered[0]
    substantive_objective = winner_objective[:-1]
    admissible = (
        winner_summary.pass_count >= DEVELOPMENT_MINIMUM_PASS_COUNT
        and winner_summary.structural_pass_count == 20
    )
    return {
        "selector_version": SELECTOR_VERSION,
        "dataset": dataset,
        "winner": winner.to_dict(),
        "winner_summary": winner_summary.to_dict(),
        "objective": _jsonable(winner_objective),
        "substantive_ties_before_candidate_id": [
            candidate.candidate_id
            for objective, candidate, _ in ordered
            if objective[:-1] == substantive_objective
        ],
        "development_minimum_pass_count": DEVELOPMENT_MINIMUM_PASS_COUNT,
        "development_required_structural_pass_count": 20,
        "development_admissible": admissible,
        "status": "DATASET_DEVELOPMENT_GO" if admissible else "DATASET_DEVELOPMENT_NO_GO",
        "ordered_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "objective": _jsonable(objective),
            }
            for objective, candidate, _ in ordered
        ],
    }


def validate_parent_v3_bundle(
    config: FidelityV4Config,
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    """Verify that v4 is bound to the exact completed v3 no-go bundle."""

    root = config.parent_v3.root
    if not root.is_absolute():
        root = workspace_root / root
    for name, expected_sha256 in config.parent_v3.file_sha256.items():
        path = root / name
        if not path.is_file() or _file_sha256(path) != expected_sha256:
            raise RuntimeError(f"v3 parent artifact binding differs: {name}")
    try:
        metadata = json.loads((root / "metadata.json").read_text())
        status = json.loads((root / "FINAL_STATUS.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("v3 parent metadata or FINAL_STATUS.json is malformed") from error
    if not isinstance(metadata, dict) or not isinstance(status, dict):
        raise RuntimeError("v3 parent metadata and final status must be objects")
    v3_runner = _load_v3_runner(workspace_root)
    v3_runner._validate_root_bundle(root, metadata)
    live_v2_root = workspace_root / "results/work/controlled_clinical_extension_v2"
    live_v2_binding = v3_runner.verify_parent_v2(live_v2_root.resolve())
    if metadata.get("parent_v2_binding") != live_v2_binding:
        raise RuntimeError("v3 parent live v2 data/config binding differs")
    validate_controlled_transition_default_parity(
        config,
        workspace_root=workspace_root,
    )
    required = {
        "protocol": "controlled_clinical_fidelity_v3",
        "phase": "development",
        "status": "DEVELOPMENT_NO_GO",
    }
    if any(status.get(name) != value for name, value in required.items()):
        raise RuntimeError("v3 parent is not the frozen completed DEVELOPMENT_NO_GO")
    for dataset, anchor in FROZEN_ANCHORS.items():
        theta = _json_pointer(status, anchor.source_json_pointer)
        pass_count = _json_pointer(status, anchor.pass_count_json_pointer)
        if not isinstance(theta, dict) or theta.get("theta_id") != anchor.theta_id:
            raise RuntimeError(f"v3 parent anchor differs for {dataset}")
        if pass_count != anchor.development_pass_count:
            raise RuntimeError(f"v3 parent anchor pass count differs for {dataset}")
    return status


def validate_controlled_transition_default_parity(
    config: FidelityV4Config,
    *,
    workspace_root: Path,
) -> None:
    path = workspace_root / "src/scpcp/controlled_transition.py"
    if _file_sha256(path) != config.current_controlled_transition_sha256:
        raise RuntimeError("current controlled-transition dependency binding differs")
    parent_root = config.parent_v3.root
    if not parent_root.is_absolute():
        parent_root = workspace_root / parent_root
    try:
        metadata = json.loads((parent_root / "metadata.json").read_text())
        archive_path = parent_root / metadata["source_snapshot"]["archive_path"]
        with tarfile.open(archive_path, mode="r:") as archive:
            source = archive.extractfile("src/scpcp/controlled_transition.py")
            if source is None:
                raise RuntimeError("v3 snapshot lacks controlled_transition.py")
            archived_sha256 = hashlib.sha256(source.read()).hexdigest()
    except (KeyError, OSError, json.JSONDecodeError, tarfile.TarError) as error:
        raise RuntimeError("v3 controlled-transition snapshot is malformed") from error
    if archived_sha256 != config.archived_v3_controlled_transition_sha256:
        raise RuntimeError("archived v3 controlled-transition binding differs")
    if any(
        anchor.transition_mode != CONTROLLED_TRANSITION_DEFAULTS["transition_mode"]
        or anchor.outcome_residual_mode
        != CONTROLLED_TRANSITION_DEFAULTS["outcome_residual_mode"]
        for anchor in FROZEN_ANCHORS.values()
    ):
        raise RuntimeError("inherited anchors differ from controlled-transition defaults")


def _candidate_axes(dataset: str) -> tuple[tuple[str, int, str, str, str], ...]:
    if dataset not in REPAIR_DATASETS:
        raise ValueError(f"repair candidates are not defined for {dataset!r}")
    candidates = tuple(
        itertools.product(
            METRIC_ORDER,
            NEIGHBOR_ORDER,
            WEIGHT_ORDER,
            TRANSITION_ORDER,
            OUTCOME_RESIDUAL_ORDER[dataset],
        )
    )
    return tuple(
        candidate
        for candidate in candidates
        if candidate[:3] != ("stagewise_zscore", 10_000, "uniform")
    )


def _validate_parent_section(parent: Any) -> None:
    expected = {
        "root": "results/work/controlled_clinical_fidelity_v3_development",
        "required_protocol": "controlled_clinical_fidelity_v3",
        "required_phase": "development",
        "required_status": "DEVELOPMENT_NO_GO",
        "file_sha256": _PARENT_FILES,
    }
    if parent != expected:
        raise ValueError("v4 parent binding differs from the frozen v3 bundle")


def _validate_development_section(development: Any) -> None:
    if not isinstance(development, dict) or set(development) != {
        "role",
        "datasets",
        "seeds",
        "rng_reuse_audit",
    }:
        raise ValueError("v4 development schema differs")
    if development["role"] != "k0_only_reuse_of_v3_development_banks":
        raise ValueError("v4 development role differs")
    if tuple(development["datasets"]) != REPAIR_DATASETS:
        raise ValueError("v4 development must repair eICU and MIMIC-CXR independently")
    _seed_banks(development["seeds"], REPAIR_DATASETS)
    if development["rng_reuse_audit"] != _DEVELOPMENT_REUSE_AUDIT:
        raise ValueError("v4 development reuse audit binding differs")


def _validate_confirmation_section(confirmation: Any) -> None:
    if not isinstance(confirmation, dict) or set(confirmation) != {
        "role",
        "decision_scope",
        "independent_patient_confirmation_claimed",
        "seeds",
        "bootstrap_seeds",
        "rng_audit",
    }:
        raise ValueError("v4 confirmation schema differs")
    if confirmation["role"] != "fresh_split_confirmation" or confirmation[
        "decision_scope"
    ] != "per_dataset_independent":
        raise ValueError("v4 confirmation is not dataset-independent")
    if confirmation["independent_patient_confirmation_claimed"] is not False:
        raise ValueError("v4 must not claim independent-patient confirmation")
    _seed_banks(confirmation["seeds"], DATASETS)
    if tuple(confirmation["bootstrap_seeds"]) != DATASETS:
        raise ValueError("v4 confirmation bootstrap dataset order differs")
    if confirmation["rng_audit"] != {
        "scope": "v2_full_derived_mapping_against_current_artifacts_and_source",
        "derived_stream_count": 1304,
        "prior_stream_count": 5476,
        "collision_count": 0,
        "mapping_sha256": (
            "3a78ec5afe69f57928de894a38803f5c369b33ab1db3f7c37bd403b974f75c72"
        ),
    }:
        raise ValueError("v4 confirmation RNG audit binding differs")


def _validate_grid_section(grids: Any) -> None:
    if not isinstance(grids, dict) or tuple(grids) != REPAIR_DATASETS:
        raise ValueError("v4 repair-grid datasets differ")
    common = {
        "metric_order": list(METRIC_ORDER),
        "neighbor_order": list(NEIGHBOR_ORDER),
        "full_cell_neighbor_sentinel": 10_000,
        "weight_order": list(WEIGHT_ORDER),
        "transition_order": list(TRANSITION_ORDER),
        "bandwidth": 2.0,
        "ridge_mode": "sample_normalized_no_intercept",
        "penalize_intercept": False,
        "semantic_deduplication": "omit_stagewise_zscore_k10000_uniform",
    }
    expected = {
        "eicu": {
            **common,
            "outcome_residual_order": ["standardized"],
            "ridge_value": 1e-3,
            "candidate_count": 14,
        },
        "mimic_cxr": {
            **common,
            "outcome_residual_order": ["standardized", "raw"],
            "ridge_value": 1e-3,
            "candidate_count": 28,
        },
    }
    if grids != expected:
        raise ValueError("v4 repair grid differs from the frozen dataset-specific search")


def _seed_banks(
    payload: Mapping[str, Mapping[str, int]],
    datasets: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    if not isinstance(payload, dict) or tuple(payload) != datasets:
        raise ValueError("seed banks differ from the frozen dataset order")
    if any(not isinstance(spec, dict) or set(spec) != {"start", "stop", "step"} for spec in payload.values()):
        raise ValueError("seed-bank schema differs from start/stop/step")
    return {
        name: tuple(range(int(spec["start"]), int(spec["stop"]), int(spec["step"])))
        for name, spec in payload.items()
    }


def _json_pointer(payload: Mapping[str, Any], pointer: str) -> Any:
    value: Any = payload
    for name in pointer.removeprefix("/").split("/"):
        if not isinstance(value, dict) or name not in value:
            raise RuntimeError(f"v3 parent JSON pointer is missing: {pointer}")
        value = value[name]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_float(left: float, right: float) -> bool:
    if math.isinf(left) or math.isinf(right):
        return left == right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "inf"
    return value


def _load_v3_runner(workspace_root: Path) -> Any:
    name = "_scpcp_controlled_clinical_fidelity_v3_parent_validator"
    if name in sys.modules:
        return sys.modules[name]
    path = workspace_root / "scripts/run_controlled_clinical_fidelity_v3.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the v3 parent validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module
