"""Contract for the post-failure MIMIC-CXR environment-support study.

The study keeps the previously selected B02 outcome bridge and every numerical
gate fixed.  Its sole scientific change is a patient-disjoint 20/20/60 raw-data
role allocation, giving the empirical environment enough opportunity to
contain the exceptionally sparse hypoxemia tail.  This module contains no
coverage or method-comparison code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


PROTOCOL = "controlled_clinical_mimic_cxr_environment_support_v1"
DATASET = "mimic_cxr"
BRIDGE_CANDIDATE_ID = "B02_pooled_successor_bridge_stage_one_hot"
ROLE_SPLIT = (0.20, 0.20, 0.60)
DEVELOPMENT_BLOCKS = {
    "block_a": tuple(range(631_000, 631_200, 10)),
    "block_b": tuple(range(631_200, 631_400, 10)),
}
ORIGINAL_CONFIRMATION_SEEDS = tuple(range(632_000, 632_200, 10))
ORIGINAL_CONFIRMATION_BOOTSTRAP_SEED = 63_200_019
CONFIRMATION_SEEDS = tuple(range(633_000, 633_200, 10))
CONFIRMATION_BOOTSTRAP_SEED = 63_300_019
PRELAUNCH_AMENDMENT_ID = "prelaunch_integrity_amendment_20260903"
ORIGINAL_FROZEN_AT_UTC = "2026-09-03T05:51:54Z"
FROZEN_AT_UTC = "2026-09-03T06:11:42Z"
DEVELOPMENT_MINIMUM_JOINT_BY_BLOCK = 19
DEVELOPMENT_MINIMUM_JOINT_TOTAL = 39
DEVELOPMENT_REQUIRED_STRUCTURAL_TOTAL = 40
DEVELOPMENT_MAXIMUM_Q95_RATIO = 0.95
CONFIRMATION_MINIMUM_JOINT = 19
CONFIRMATION_REQUIRED_STRUCTURAL = 20
K0_THRESHOLDS = {
    "maximum_score_ks": 0.10,
    "maximum_signed_residual_w1": 0.25,
    "maximum_successor_mean_w1": 0.25,
    "maximum_successor_q95_w1": 0.50,
}
FROZEN_CONFIG_PAYLOAD_SHA256 = (
    "66b5ed532ad763a41f24a4451a0cc1e6677c7cc592a77b4d5acac8a46b1138e7"
)
_PRIOR_V5_ROOT = Path(
    "results/work/controlled_clinical_fidelity_v5_mimic_cxr_confirmation"
)
_PRIOR_V5_FILES = {
    "FINAL_STATUS.json": (
        "5f104c0dff121174b52e5ce0c082583744d544cda47a296f5ad0329474472f18"
    ),
    "gate.json": (
        "d663b2c5b5d6a7280efe2dceb31c743b94a7ec674825caf3eaad7701d04e6a5b"
    ),
}
_PRIOR_V6_ROOT = Path(
    "results/work/controlled_clinical_fidelity_v6_mimic_cxr_development"
)
_PRIOR_V6_FILES = {
    "FINAL_STATUS.json": (
        "39c014b9429466849b709a90739ae1b88d72d6eec43f3425ef2281d48fa058a1"
    ),
    "development_gate.json": (
        "45a9768412fc4fae47384581b0235a43e21a0d965b25782b0986603a9e8aa4bc"
    ),
}


@dataclass(frozen=True)
class PriorBinding:
    root: Path
    files: Mapping[str, str]


@dataclass(frozen=True)
class EnvironmentSupportConfig:
    prior_v5: PriorBinding
    prior_v6: PriorBinding
    development_blocks: Mapping[str, tuple[int, ...]]
    confirmation_seeds: tuple[int, ...]
    confirmation_bootstrap_seed: int
    original_frozen_at_utc: str
    frozen_at_utc: str
    pilot_visible_at_freeze: Mapping[str, tuple[int, ...]]
    changes_after_freeze: tuple[str, ...]

    def validate(self) -> None:
        if tuple(self.development_blocks) != ("block_a", "block_b"):
            raise ValueError("development block order differs from the frozen contract")
        if dict(self.development_blocks) != DEVELOPMENT_BLOCKS:
            raise ValueError("development seed blocks differ from the frozen contract")
        if self.confirmation_seeds != CONFIRMATION_SEEDS:
            raise ValueError("confirmation seed bank differs from the frozen contract")
        if self.confirmation_bootstrap_seed != CONFIRMATION_BOOTSTRAP_SEED:
            raise ValueError("confirmation bootstrap seed differs")
        if self.prior_v5.root != _PRIOR_V5_ROOT or dict(
            self.prior_v5.files
        ) != _PRIOR_V5_FILES:
            raise ValueError("v5 prior binding differs from the frozen contract")
        if self.prior_v6.root != _PRIOR_V6_ROOT or dict(
            self.prior_v6.files
        ) != _PRIOR_V6_FILES:
            raise ValueError("v6 prior binding differs from the frozen contract")
        if self.original_frozen_at_utc != ORIGINAL_FROZEN_AT_UTC:
            raise ValueError("original design freeze timestamp differs")
        if self.frozen_at_utc != FROZEN_AT_UTC:
            raise ValueError("amended design freeze timestamp differs")
        if self.changes_after_freeze != (PRELAUNCH_AMENDMENT_ID,):
            raise ValueError("post-freeze amendment record differs")
        expected_pilot = {
            "block_a": DEVELOPMENT_BLOCKS["block_a"][:5],
            "block_b": DEVELOPMENT_BLOCKS["block_b"][:5],
        }
        if dict(self.pilot_visible_at_freeze) != expected_pilot:
            raise ValueError("development exposure record differs")


def normalized_k0_ratio(metrics: Mapping[str, Any]) -> float:
    """Return the largest unchanged K0 metric-to-threshold ratio."""

    expected = {*K0_THRESHOLDS, "structural_invariants"}
    if set(metrics) != expected:
        raise ValueError("K0 metrics do not match the frozen schema")
    structural = metrics["structural_invariants"]
    if type(structural) is not bool:
        raise ValueError("K0 structural_invariants must be boolean")
    values = []
    for name, limit in K0_THRESHOLDS.items():
        raw_value = metrics[name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise ValueError(f"K0 metric {name} must be numeric")
        value = float(raw_value)
        if value < 0.0:
            raise ValueError(f"K0 metric {name} must be nonnegative")
        values.append(value / limit)
    if not structural:
        return math.inf
    return max(values) if all(math.isfinite(value) for value in values) else math.inf


def summarize_development(
    support_by_block: Mapping[str, Sequence[Mapping[str, Any]]],
    k0_by_block: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Apply the frozen two-block joint development gate."""

    if tuple(support_by_block) != tuple(DEVELOPMENT_BLOCKS):
        raise ValueError("support development blocks differ")
    if tuple(k0_by_block) != tuple(DEVELOPMENT_BLOCKS):
        raise ValueError("K0 development blocks differ")
    summaries: dict[str, Any] = {}
    all_ratios: list[float] = []
    joint_total = 0
    structural_total = 0
    for block, seeds in DEVELOPMENT_BLOCKS.items():
        support = _rows_by_seed(support_by_block[block], seeds)
        k0 = _rows_by_seed(k0_by_block[block], seeds)
        support_passed = {
            seed: _validated_support_passed(support[seed]) for seed in seeds
        }
        k0_results = {seed: _validated_k0_row(k0[seed]) for seed in seeds}
        joint = [
            seed
            for seed in seeds
            if support_passed[seed] and k0_results[seed][0]
        ]
        structural = [seed for seed in seeds if k0_results[seed][1]]
        ratios = [k0_results[seed][2] for seed in seeds]
        all_ratios.extend(ratios)
        joint_total += len(joint)
        structural_total += len(structural)
        summaries[block] = {
            "seeds": list(seeds),
            "support_pass_count": sum(support_passed.values()),
            "k0_pass_count": sum(value[0] for value in k0_results.values()),
            "joint_pass_count": len(joint),
            "joint_pass_seeds": joint,
            "structural_pass_count": len(structural),
            "normalized_k0_ratios": ratios,
        }
    q95 = (
        float(np.quantile(np.asarray(all_ratios), 0.95, method="linear"))
        if all(math.isfinite(value) for value in all_ratios)
        else math.inf
    )
    admissible = (
        all(
            value["joint_pass_count"] >= DEVELOPMENT_MINIMUM_JOINT_BY_BLOCK
            for value in summaries.values()
        )
        and joint_total >= DEVELOPMENT_MINIMUM_JOINT_TOTAL
        and structural_total == DEVELOPMENT_REQUIRED_STRUCTURAL_TOTAL
        and q95 <= DEVELOPMENT_MAXIMUM_Q95_RATIO
    )
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "DEVELOPMENT_GO" if admissible else "DEVELOPMENT_NO_GO",
        "development_admissible": admissible,
        "role_split": list(ROLE_SPLIT),
        "bridge_candidate_id": BRIDGE_CANDIDATE_ID,
        "candidate_count": 1,
        "selector_present": False,
        "blocks": summaries,
        "joint_pass_count_total": joint_total,
        "structural_pass_count_total": structural_total,
        "q95_normalized_k0_ratio": q95,
        "maximum_q95_normalized_k0_ratio": DEVELOPMENT_MAXIMUM_Q95_RATIO,
        "seed_deletions": 0,
        "coverage_generated": False,
    }


def summarize_confirmation(
    support_rows: Sequence[Mapping[str, Any]],
    k0_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the unchanged 19/20 fresh confirmation rule."""

    support = _rows_by_seed(support_rows, CONFIRMATION_SEEDS)
    k0 = _rows_by_seed(k0_rows, CONFIRMATION_SEEDS)
    support_passed = {
        seed: _validated_support_passed(support[seed]) for seed in CONFIRMATION_SEEDS
    }
    k0_results = {
        seed: _validated_k0_row(k0[seed]) for seed in CONFIRMATION_SEEDS
    }
    joint = [
        seed
        for seed in CONFIRMATION_SEEDS
        if support_passed[seed] and k0_results[seed][0]
    ]
    structural = [
        seed
        for seed in CONFIRMATION_SEEDS
        if k0_results[seed][1]
    ]
    passed = (
        len(joint) >= CONFIRMATION_MINIMUM_JOINT
        and len(structural) == CONFIRMATION_REQUIRED_STRUCTURAL
    )
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "CONFIRMATION_GO" if passed else "CONFIRMATION_NO_GO",
        "confirmation_admissible": passed,
        "prespecified_seed_count": 20,
        "support_pass_count": sum(support_passed.values()),
        "k0_pass_count": sum(value[0] for value in k0_results.values()),
        "joint_pass_count": len(joint),
        "joint_pass_seeds": joint,
        "structural_pass_count": len(structural),
        "seed_deletions": 0,
        "coverage_generated": False,
    }


def load_config(path: Path) -> EnvironmentSupportConfig:
    payload = yaml.safe_load(path.read_text())
    if _json_sha256(payload) != FROZEN_CONFIG_PAYLOAD_SHA256:
        raise ValueError("environment-support config differs from the frozen contract")
    if payload["protocol"] != PROTOCOL or payload["dataset"] != DATASET:
        raise ValueError("environment-support protocol identity differs")
    role_split = tuple(
        float(payload["role_split"][key])
        for key in ("predictor", "fidelity", "environment")
    )
    if role_split != ROLE_SPLIT:
        raise ValueError("CXR role split differs")
    environment = payload["environment"]
    if (
        environment["bridge_candidate_id"] != BRIDGE_CANDIDATE_ID
        or int(environment["candidate_count"]) != 1
        or bool(environment["selector_present"])
        or bool(environment["bridge_search_permitted"])
    ):
        raise ValueError("B02 must be the single non-selected bridge")
    if {key: float(payload["k0_gate"][key]) for key in K0_THRESHOLDS} != K0_THRESHOLDS:
        raise ValueError("K0 thresholds differ")
    development = payload["development"]
    confirmation = payload["confirmation"]
    config = EnvironmentSupportConfig(
        prior_v5=PriorBinding(
            Path(payload["prior_negative_evidence"]["v5_confirmation"]["root"]),
            {
                "FINAL_STATUS.json": payload["prior_negative_evidence"][
                    "v5_confirmation"
                ]["final_status_sha256"],
                "gate.json": payload["prior_negative_evidence"]["v5_confirmation"]
                ["gate_sha256"],
            },
        ),
        prior_v6=PriorBinding(
            Path(payload["prior_negative_evidence"]["v6_development"]["root"]),
            {
                "FINAL_STATUS.json": payload["prior_negative_evidence"][
                    "v6_development"
                ]["final_status_sha256"],
                "development_gate.json": payload["prior_negative_evidence"]
                ["v6_development"]["development_gate_sha256"],
            },
        ),
        development_blocks={
            name: tuple(
                range(
                    int(values["start"]),
                    int(values["stop"]),
                    int(values["step"]),
                )
            )
            for name, values in (
                (name, development[name]) for name in ("block_a", "block_b")
            )
        },
        confirmation_seeds=tuple(
            range(
                int(confirmation["seeds"]["start"]),
                int(confirmation["seeds"]["stop"]),
                int(confirmation["seeds"]["step"]),
            )
        ),
        confirmation_bootstrap_seed=int(confirmation["bootstrap_seed"]),
        original_frozen_at_utc=str(
            payload["design_freeze"]["original_frozen_at_utc"]
        ),
        frozen_at_utc=str(payload["design_freeze"]["frozen_at_utc"]),
        pilot_visible_at_freeze={
            name: tuple(int(seed) for seed in values)
            for name, values in payload["design_freeze"][
                "development_pilot_visible_at_freeze"
            ].items()
        },
        changes_after_freeze=tuple(
            str(change)
            for change in payload["design_freeze"]["changes_after_freeze"]
        ),
    )
    config.validate()
    return config


def verify_prior_bindings(root: Path, config: EnvironmentSupportConfig) -> dict[str, Any]:
    bindings = {}
    prior_bindings = (
        ("v5_confirmation", config.prior_v5),
        ("v6_development", config.prior_v6),
    )
    for label, binding in prior_bindings:
        observed = {}
        for relative, expected in binding.files.items():
            path = root / binding.root / relative
            if not path.is_file():
                raise FileNotFoundError(f"missing prior negative evidence: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise RuntimeError(f"prior negative evidence changed: {path}")
            observed[relative] = digest
        bindings[label] = {"root": binding.root.as_posix(), "files": observed}
    return {**bindings, "combined_sha256": _json_sha256(bindings)}


def _rows_by_seed(
    rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> dict[int, Mapping[str, Any]]:
    if any(type(row.get("seed")) is not int for row in rows):
        raise ValueError("artifact seeds must be integers")
    indexed = {row["seed"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(seeds):
        raise ValueError("rows do not cover the exact prespecified seed set")
    return indexed


def _validated_support_passed(row: Mapping[str, Any]) -> bool:
    passed = row.get("passed")
    if type(passed) is not bool:
        raise ValueError("support passed flag must be boolean")
    return passed


def _validated_k0_row(row: Mapping[str, Any]) -> tuple[bool, bool, float]:
    artifact_passed = row.get("passed")
    if type(artifact_passed) is not bool:
        raise ValueError("K0 passed flag must be boolean")
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("K0 metrics must be a mapping")
    ratio = normalized_k0_ratio(metrics)
    computed_passed = ratio <= 1.0
    if artifact_passed != computed_passed:
        raise ValueError("K0 passed flag disagrees with the frozen thresholds")
    return computed_passed, metrics["structural_invariants"], ratio


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
