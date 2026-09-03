"""Frozen K0-only development and confirmation contract for clinical v3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


PROTOCOL = "controlled_clinical_fidelity_v3"
SELECTOR_VERSION = "controlled_clinical_fidelity_v3_selector_v1"
DATASETS = ("mimic_iv", "eicu", "inspire", "mimic_cxr")
REPAIRED_DATASETS = ("eicu", "inspire", "mimic_cxr")
METRIC_ORDER = ("raw", "stagewise_zscore")
NEIGHBOR_ORDER = (100, 200)
WEIGHT_ORDER = ("gaussian_b2", "gaussian_b4", "uniform")
RIDGE_ORDER = (
    "raw_ridge_1e-3",
    "normalized_ridge_1e-3",
    "normalized_ridge_1e-2",
)
METRIC_THRESHOLDS = {
    "maximum_score_ks": 0.10,
    "maximum_signed_residual_w1": 0.25,
    "maximum_successor_mean_w1": 0.25,
    "maximum_successor_q95_w1": 0.50,
}


@dataclass(frozen=True)
class KernelTheta:
    """One controlled-transition setting; it never changes SC-PCP itself."""

    stage_a_id: str
    metric: str
    neighbors: int
    weight: str
    ridge: str = "raw_ridge_1e-3"

    def __post_init__(self) -> None:
        if self.metric not in METRIC_ORDER:
            raise ValueError(f"unknown representation metric: {self.metric}")
        if self.neighbors not in NEIGHBOR_ORDER:
            raise ValueError(f"unknown neighbor count: {self.neighbors}")
        if self.weight not in WEIGHT_ORDER:
            raise ValueError(f"unknown donor weight: {self.weight}")
        if self.ridge not in RIDGE_ORDER:
            raise ValueError(f"unknown ridge candidate: {self.ridge}")
        index = (
            METRIC_ORDER.index(self.metric) * len(NEIGHBOR_ORDER) * len(WEIGHT_ORDER)
            + NEIGHBOR_ORDER.index(self.neighbors) * len(WEIGHT_ORDER)
            + WEIGHT_ORDER.index(self.weight)
        )
        expected = f"A{index:02d}_{self.metric}_k{self.neighbors}_{self.weight}"
        if self.stage_a_id != expected:
            raise ValueError("Stage-A candidate ID does not match its parameters")

    @property
    def theta_id(self) -> str:
        return f"{self.stage_a_id}__{self.ridge}"

    @property
    def donor_weighting(self) -> str:
        return "uniform" if self.weight == "uniform" else "gaussian"

    @property
    def bandwidth(self) -> float:
        return {"gaussian_b2": 2.0, "gaussian_b4": 4.0, "uniform": 2.0}[
            self.weight
        ]

    @property
    def ridge_mode(self) -> str:
        return (
            "v2_raw"
            if self.ridge == "raw_ridge_1e-3"
            else "sample_normalized_no_intercept"
        )

    @property
    def ridge_value(self) -> float:
        return {
            "raw_ridge_1e-3": 1e-3,
            "normalized_ridge_1e-3": 1e-3,
            "normalized_ridge_1e-2": 1e-2,
        }[self.ridge]

    @property
    def minimal_change_tuple(self) -> tuple[int, int, int]:
        weight_rank = {name: index for index, name in enumerate(WEIGHT_ORDER)}
        return (
            int(self.metric != "raw"),
            abs(self.neighbors - 100),
            weight_rank[self.weight],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta_id": self.theta_id,
            "stage_a_id": self.stage_a_id,
            "metric": self.metric,
            "neighbors": self.neighbors,
            "weight": self.weight,
            "bandwidth": self.bandwidth,
            "donor_weighting": self.donor_weighting,
            "ridge": self.ridge,
            "ridge_mode": self.ridge_mode,
            "ridge_value": self.ridge_value,
            "penalize_intercept": self.ridge_mode == "v2_raw",
            "minimal_change_tuple": list(self.minimal_change_tuple),
        }


@dataclass(frozen=True)
class FidelityV3Config:
    parent_v2_root: Path
    development_seeds: Mapping[str, tuple[int, ...]]
    confirmation_seeds: Mapping[str, tuple[int, ...]]
    confirmation_bootstrap_seeds: Mapping[str, int]
    stagewise_sd_floor: float

    def validate(self) -> None:
        if tuple(self.development_seeds) != DATASETS:
            raise ValueError("development datasets differ from the frozen order")
        if tuple(self.confirmation_seeds) != DATASETS:
            raise ValueError("confirmation datasets differ from the frozen order")
        if tuple(self.confirmation_bootstrap_seeds) != DATASETS:
            raise ValueError("confirmation bootstrap datasets differ")
        expected_development = {
            "mimic_iv": tuple(range(93_600, 93_800, 10)),
            "eicu": tuple(range(92_000, 92_200, 10)),
            "inspire": tuple(range(92_300, 92_500, 10)),
            "mimic_cxr": tuple(range(92_600, 92_800, 10)),
        }
        expected_confirmation = {
            "mimic_iv": tuple(range(111_000, 111_200, 10)),
            "eicu": tuple(range(112_000, 112_200, 10)),
            "inspire": tuple(range(113_000, 113_200, 10)),
            "mimic_cxr": tuple(range(114_000, 114_200, 10)),
        }
        if dict(self.development_seeds) != expected_development:
            raise ValueError("development must reuse exactly the v2 seed banks")
        if dict(self.confirmation_seeds) != expected_confirmation:
            raise ValueError("confirmation seeds differ from the fresh frozen banks")
        expected_bootstrap = {
            "mimic_iv": 11_100_019,
            "eicu": 11_200_019,
            "inspire": 11_300_019,
            "mimic_cxr": 11_400_019,
        }
        if dict(self.confirmation_bootstrap_seeds) != expected_bootstrap:
            raise ValueError("confirmation bootstrap seeds differ")
        if any(len(seeds) != 20 or len(set(seeds)) != 20 for seeds in self.confirmation_seeds.values()):
            raise ValueError("confirmation requires exactly 20 distinct seeds per dataset")
        if set().union(*map(set, self.confirmation_seeds.values())) & set().union(
            *map(set, self.development_seeds.values())
        ):
            raise ValueError("development and confirmation base seeds overlap")
        if self.stagewise_sd_floor != 1e-4:
            raise ValueError("stagewise representation sd floor must be 1e-4")


@dataclass(frozen=True)
class CandidateDatasetSummary:
    candidate_id: str
    dataset: str
    pass_count: int
    q95_seed_ratio: float
    mean_seed_ratio: float
    seed_ratios: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "dataset": self.dataset,
            "pass_count": self.pass_count,
            "q95_seed_ratio": (
                self.q95_seed_ratio if math.isfinite(self.q95_seed_ratio) else None
            ),
            "mean_seed_ratio": (
                self.mean_seed_ratio if math.isfinite(self.mean_seed_ratio) else None
            ),
            "seed_ratios": [
                value if math.isfinite(value) else None for value in self.seed_ratios
            ],
            "structural_failure_ratio_is_infinite": any(
                not math.isfinite(value) for value in self.seed_ratios
            ),
        }


def load_fidelity_v3_config(path: Path) -> FidelityV3Config:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        raise ValueError(f"protocol must be {PROTOCOL}")
    expected_top_level = {
        "protocol",
        "parent_v2_root",
        "datasets",
        "devices_required",
        "development",
        "confirmation",
        "stage_a",
        "stage_b",
        "k0_gate",
        "support_gate",
        "selection",
        "information_firewall",
    }
    if set(payload) != expected_top_level:
        raise ValueError("v3 config keys differ from the frozen schema")
    if payload.get("parent_v2_root") != (
        "results/work/controlled_clinical_extension_v2"
    ):
        raise ValueError("v3 parent root differs from the frozen v2 root")
    if tuple(payload.get("datasets", ())) != DATASETS:
        raise ValueError("v3 datasets differ from the frozen order")
    if payload.get("devices_required") != 2:
        raise ValueError("v3 requires exactly two CUDA devices")
    if payload["development"].get("role") != (
        "k0_only_reuse_of_v2_prespecified_seed_bank"
    ) or set(payload["development"]) != {"role", "seeds"}:
        raise ValueError("development role differs from the frozen contract")
    if (
        payload["confirmation"].get("role") != "fresh_split_confirmation"
        or set(payload["confirmation"])
        != {"role", "seeds", "bootstrap_seeds"}
    ):
        raise ValueError("confirmation role differs from the frozen contract")
    stage_a = payload["stage_a"]
    stage_b = payload["stage_b"]
    if set(stage_a) != {
        "metric_order",
        "neighbor_order",
        "weight_order",
        "stagewise_zscore",
        "ridge",
    } or set(stage_b) != {"ridge_order", "normalized_penalty"}:
        raise ValueError("v3 candidate schema differs from the frozen grid")
    if (
        tuple(stage_a["metric_order"]) != METRIC_ORDER
        or tuple(stage_a["neighbor_order"]) != NEIGHBOR_ORDER
        or tuple(stage_a["weight_order"]) != WEIGHT_ORDER
        or tuple(stage_b["ridge_order"]) != RIDGE_ORDER
    ):
        raise ValueError("v3 candidate order differs from the frozen grid")
    zscore = stage_a["stagewise_zscore"]
    if zscore != {
        "source": "D_env_only",
        "pooling": "per_stage_pooled_over_actions",
        "center_dtype": "float64",
        "scale_dtype": "float64",
        "population_sd": True,
        "sd_floor": 1e-4,
    }:
        raise ValueError("stagewise z-score contract differs")
    if stage_a.get("ridge") != {
        "mode": "v2_raw",
        "value": 1e-3,
        "penalize_intercept": True,
    }:
        raise ValueError("Stage-A ridge differs from the v2 anchor")
    if stage_b.get("normalized_penalty") != {
        "gram": "XTX_over_n",
        "right_hand_side": "XTY_over_n",
        "penalize_intercept": False,
    }:
        raise ValueError("Stage-B normalized ridge contract differs")
    expected_gate = {
        "systematic_replays": 16,
        **METRIC_THRESHOLDS,
        "minimum_available_seed_fraction": 0.95,
        "active_coordinate_sd_floor": 1e-4,
    }
    if payload["k0_gate"] != expected_gate:
        raise ValueError("K0 gate was changed from v2")
    if payload["support_gate"] != {
        "minimum_unique_patients_per_stage_action": 20,
        "minimum_available_seed_fraction": 0.95,
    }:
        raise ValueError("support gate was changed from v2")
    expected_selection = {
        "version": SELECTOR_VERSION,
        "seed_ratio_quantile": 0.95,
        "seed_ratio_quantile_method": "linear",
        "shared_stage_a_order": [
            "maximize_minimum_dataset_pass_count",
            "minimize_maximum_dataset_q95_seed_ratio",
            "minimize_global_mean_seed_ratio",
            "minimize_change_from_v2",
            "candidate_index",
        ],
        "dataset_fallback_order": [
            "maximize_pass_count",
            "minimize_q95_seed_ratio",
            "minimize_mean_seed_ratio",
            "minimize_change_from_v2",
            "candidate_index",
        ],
        "shared_admissibility": {
            "repaired_dataset_minimum_pass_count": 19,
            "mimic_iv_exact_pass_count": 20,
        },
        "failure": "DEVELOPMENT_NO_GO",
    }
    if payload["selection"] != expected_selection:
        raise ValueError("selector contract differs")
    if payload["information_firewall"] != {
        "allowed": ["support", "k0_fidelity", "context_identity", "provenance"],
        "forbidden": ["science", "coverage", "width", "method_selection"],
        "coverage_generation_permitted": False,
    }:
        raise ValueError("v3 information firewall differs")
    config = FidelityV3Config(
        parent_v2_root=Path(payload["parent_v2_root"]),
        development_seeds=_seed_banks(payload["development"]["seeds"]),
        confirmation_seeds=_seed_banks(payload["confirmation"]["seeds"]),
        confirmation_bootstrap_seeds={
            name: int(seed)
            for name, seed in payload["confirmation"]["bootstrap_seeds"].items()
        },
        stagewise_sd_floor=float(zscore["sd_floor"]),
    )
    config.validate()
    return config


def stage_a_candidates() -> tuple[KernelTheta, ...]:
    candidates = []
    index = 0
    for metric in METRIC_ORDER:
        for neighbors in NEIGHBOR_ORDER:
            for weight in WEIGHT_ORDER:
                stage_a_id = f"A{index:02d}_{metric}_k{neighbors}_{weight}"
                candidates.append(
                    KernelTheta(
                        stage_a_id=stage_a_id,
                        metric=metric,
                        neighbors=neighbors,
                        weight=weight,
                    )
                )
                index += 1
    return tuple(candidates)


def stage_b_candidates(stage_a: KernelTheta) -> tuple[KernelTheta, ...]:
    return tuple(
        KernelTheta(
            stage_a_id=stage_a.stage_a_id,
            metric=stage_a.metric,
            neighbors=stage_a.neighbors,
            weight=stage_a.weight,
            ridge=ridge,
        )
        for ridge in RIDGE_ORDER
    )


def normalized_seed_ratio(metrics: Mapping[str, Any]) -> float:
    if metrics.get("structural_invariants") is not True:
        return math.inf
    ratios = [
        float(metrics[name]) / threshold
        for name, threshold in METRIC_THRESHOLDS.items()
    ]
    if not all(math.isfinite(value) and value >= 0.0 for value in ratios):
        return math.inf
    return max(ratios)


def summarize_candidate_dataset(
    candidate_id: str,
    dataset: str,
    metrics_by_seed: Sequence[Mapping[str, Any]],
) -> CandidateDatasetSummary:
    if dataset not in DATASETS or len(metrics_by_seed) != 20:
        raise ValueError("candidate summary requires one full 20-seed dataset bank")
    ratios = tuple(normalized_seed_ratio(metrics) for metrics in metrics_by_seed)
    finite = all(math.isfinite(value) for value in ratios)
    q95 = (
        float(np.quantile(np.asarray(sorted(ratios), dtype=np.float64), 0.95, method="linear"))
        if finite
        else math.inf
    )
    mean = float(np.mean(np.asarray(ratios, dtype=np.float64))) if finite else math.inf
    return CandidateDatasetSummary(
        candidate_id=candidate_id,
        dataset=dataset,
        pass_count=sum(value <= 1.0 for value in ratios),
        q95_seed_ratio=q95,
        mean_seed_ratio=mean,
        seed_ratios=ratios,
    )


def select_shared_candidate(
    candidates: Sequence[KernelTheta],
    summaries: Mapping[str, Mapping[str, CandidateDatasetSummary]],
) -> dict[str, Any]:
    """Apply the frozen cross-dataset lexicographic selector."""

    scored = []
    for index, candidate in enumerate(candidates):
        by_dataset = summaries[candidate.theta_id]
        if tuple(by_dataset) != DATASETS:
            raise ValueError("shared selector requires every dataset in frozen order")
        values = tuple(by_dataset[name] for name in DATASETS)
        objective = (
            -min(value.pass_count for value in values),
            max(value.q95_seed_ratio for value in values),
            float(np.mean([value.mean_seed_ratio for value in values])),
            candidate.minimal_change_tuple,
            index,
        )
        scored.append((objective, candidate))
    return _selection_record(scored)


def select_dataset_candidate(
    dataset: str,
    candidates: Sequence[KernelTheta],
    summaries: Mapping[str, Mapping[str, CandidateDatasetSummary]],
) -> dict[str, Any]:
    """Apply the frozen per-dataset fallback selector."""

    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    scored = []
    for index, candidate in enumerate(candidates):
        value = summaries[candidate.theta_id][dataset]
        objective = (
            -value.pass_count,
            value.q95_seed_ratio,
            value.mean_seed_ratio,
            candidate.minimal_change_tuple,
            index,
        )
        scored.append((objective, candidate))
    return _selection_record(scored)


def _selection_record(
    scored: Sequence[tuple[tuple[Any, ...], KernelTheta]],
) -> dict[str, Any]:
    if not scored:
        raise ValueError("selector candidate list is empty")
    ordered = sorted(scored, key=lambda item: item[0])
    winner_objective, winner = ordered[0]
    substantive = winner_objective[:-1]
    ties = [candidate.theta_id for objective, candidate in ordered if objective[:-1] == substantive]
    return {
        "selector_version": SELECTOR_VERSION,
        "winner": winner.to_dict(),
        "objective": _jsonable_objective(winner_objective),
        "substantive_ties_before_candidate_index": ties,
        "ordered_candidates": [
            {
                "theta_id": candidate.theta_id,
                "objective": _jsonable_objective(objective),
            }
            for objective, candidate in ordered
        ],
    }


def _jsonable_objective(value: tuple[Any, ...]) -> list[Any]:
    output = []
    for item in value:
        if isinstance(item, tuple):
            output.append(list(item))
        elif isinstance(item, float) and not math.isfinite(item):
            output.append("inf")
        else:
            output.append(item)
    return output


def _seed_banks(payload: Mapping[str, Mapping[str, int]]) -> dict[str, tuple[int, ...]]:
    if tuple(payload) != DATASETS:
        raise ValueError("seed banks differ from the frozen dataset order")
    if any(set(spec) != {"start", "stop", "step"} for spec in payload.values()):
        raise ValueError("seed-bank schema differs from start/stop/step")
    return {
        name: tuple(range(int(spec["start"]), int(spec["stop"]), int(spec["step"])))
        for name, spec in payload.items()
    }
