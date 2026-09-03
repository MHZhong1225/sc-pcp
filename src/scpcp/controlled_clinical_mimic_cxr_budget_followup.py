"""Frozen contract for the MIMIC-CXR calibration-budget follow-up.

The follow-up is explicitly motivated by the completed v1 coverage result.  It
changes only the common source-calibration and grid-prefix budgets; SC-PCP and
all five comparison methods retain their existing implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from scpcp.controlled_clinical_extension import GAMMAS, METHODS
from scpcp.controlled_clinical_mimic_cxr_environment_support import (
    BRIDGE_CANDIDATE_ID,
    ROLE_SPLIT,
    normalized_k0_ratio,
)


PROTOCOL = "controlled_clinical_mimic_cxr_budget_followup_v2"
DATASET = "mimic_cxr"
FROZEN_AT_UTC = "2026-09-03T07:58:00Z"
SEEDS = tuple(range(641_000, 641_200, 10))
BOOTSTRAP_SEED = 64_100_019
CALIBRATION_TRAJECTORIES = 12_000
GRID_TRAJECTORIES = 4_000
REFERENCE_TRAJECTORIES = 20_000
ONLINE_TRAJECTORIES = 2_000
BOOTSTRAP_RESAMPLES = 10_000
PRIMARY_GAMMA = -4.0
TARGET_COVERAGE = 0.90
MINIMUM_SELECTED_SEEDS = 19
MINIMUM_SELECTION_RATE = 0.95
MINIMUM_PRECOVERAGE_JOINT = 19
REQUIRED_STRUCTURAL_PASS_COUNT = 20
MINIMUM_OVERLAP_JOINT = 19
VALIDATION_CLAIMS = {
    "rng_and_split_fresh": True,
    "new_patient_cohort": False,
    "external_validation": False,
}
CANONICAL_SCPCP_PATH = Path("src/scpcp/marginal_prefix.py")
CANONICAL_SCPCP_SHA256 = (
    "97a14397143f7f5a9304fcc281ede2093860fff85f9631a409d03d1c2a21f3f0"
)
V1_SOURCE_MANIFEST = Path(
    "results/work/controlled_clinical_mimic_cxr_environment_support_v1_science/"
    "provenance/source_manifest_a527d42cde0b16cde0d362b4300e9acf2b0e81f0d426d8794911c53ddaf1156e.json"
)
UNCHANGED_SHARED_PATHS = (
    Path("configs/controlled_clinical_extension.yaml"),
    Path("scripts/run_controlled_clinical_extension.py"),
    Path("scripts/run_controlled_prefix_benchmark.py"),
    Path("scripts/run_controlled_six_method_benchmark.py"),
    Path("src/scpcp/baselines.py"),
    Path("src/scpcp/controlled_clinical_extension.py"),
    Path("src/scpcp/marginal_prefix.py"),
)
V1_ALLOWED_CHANGED_PATHS = (
    Path("scripts/summarize_phase0c_joint_search.py"),
    Path("tools/render_five_dataset_signed_gamma_results.py"),
)
V1_ALLOWED_ADDED_PATHS = (
    Path("configs/controlled_clinical_mimic_cxr_budget_followup_v2.yaml"),
    Path("scripts/run_controlled_clinical_mimic_cxr_budget_followup.py"),
    Path("scripts/run_controlled_clinical_mimic_cxr_budget_followup_science.py"),
    Path("src/scpcp/controlled_clinical_mimic_cxr_budget_followup.py"),
    Path("tools/interpret_five_dataset_signed_gamma_results.py"),
)
FROZEN_CONFIG_PAYLOAD_SHA256 = (
    "7a170c9b1017fbc876a46708872911d0d9fd2abc050791d28e9b65c4c1953343"
)
EXPECTED_RNG_CONTRACT = {
    "base_seed_set_sha256": (
        "0ad2c4656d798678733086eb0e0da86d854d9d7bf1c2e3689afa010c5ec0d735"
    ),
    "precoverage_stream_count": 100,
    "precoverage_mapping_sha256": (
        "cbed16cac407278cf7dee7e3be98871504dc6c957d4080b40badbac436f830bb"
    ),
    "full_stream_count": 341,
    "full_mapping_sha256": (
        "6c67a07a9de16c60646de18b9640109c076087f2149ac484c235b3c9515fd400"
    ),
    "full_id_set_sha256": (
        "6883cc424a2850581bacec59636fb6926b26c52d45a58307f26269529843795f"
    ),
    "internal_collision_count": 0,
    "precoverage_vs_postunlock_new_stream_collision_count": 0,
}


@dataclass(frozen=True)
class BudgetFollowupConfig:
    prior_roots: Mapping[str, Path]
    prior_files: Mapping[str, str]

    def validate(self) -> None:
        if tuple(self.prior_roots) != ("development", "confirmation", "science"):
            raise ValueError("v1 prior roots differ from the frozen contract")
        if len(self.prior_files) != 10:
            raise ValueError("v1 prior file binding is incomplete")
        if any(not _is_sha256(value) for value in self.prior_files.values()):
            raise ValueError("v1 prior file hash is malformed")


def load_config(path: Path) -> BudgetFollowupConfig:
    """Load and validate the exact frozen v2 YAML payload."""

    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("budget-follow-up config must be a mapping")
    if _json_sha256(payload) != FROZEN_CONFIG_PAYLOAD_SHA256:
        raise ValueError("budget-follow-up config differs from the frozen contract")
    if (
        payload["protocol"] != PROTOCOL
        or payload["dataset"] != DATASET
        or payload["design_freeze"]["frozen_at_utc"] != FROZEN_AT_UTC
        or payload["design_freeze"]["v1_coverage_inspected"] is not True
        or payload["design_freeze"]["further_changes_permitted"] is not False
        or payload["design_freeze"]["seed_deletion_permitted"] is not False
    ):
        raise ValueError("budget-follow-up identity or disclosure differs")

    split = payload["role_split"]
    if (
        tuple(float(split[name]) for name in ("predictor", "fidelity", "environment"))
        != ROLE_SPLIT
    ):
        raise ValueError("budget-follow-up role split differs")
    environment = payload["environment"]
    if environment != {
        "bridge_candidate_id": BRIDGE_CANDIDATE_ID,
        "candidate_count": 1,
        "selector_present": False,
        "bridge_search_permitted": False,
    }:
        raise ValueError("budget-follow-up B02 contract differs")

    precoverage = payload["precoverage"]
    seeds = tuple(
        range(
            int(precoverage["seeds"]["start"]),
            int(precoverage["seeds"]["stop"]),
            int(precoverage["seeds"]["step"]),
        )
    )
    if (
        seeds != SEEDS
        or precoverage["minimum_joint_pass_count"] != MINIMUM_PRECOVERAGE_JOINT
        or precoverage["required_structural_pass_count"]
        != REQUIRED_STRUCTURAL_PASS_COUNT
        or precoverage["seed_deletion_permitted"] is not False
    ):
        raise ValueError("budget-follow-up precoverage contract differs")

    science = payload["science"]
    success = science["success_gate"]
    expected_adaptation = {
        "Standard CP": 0,
        "ACI": 2_000,
        "MFCS": 0,
        "SPCI": 2_000,
        "PRC": 2_000,
        "SC-PCP": 0,
    }
    if (
        tuple(float(value) for value in science["gammas"]) != GAMMAS
        or float(science["default_gamma"]) != PRIMARY_GAMMA
        or float(science["primary_gamma"]) != PRIMARY_GAMMA
        or tuple(science["methods"]) != METHODS
        or science["calibration_trajectories"] != CALIBRATION_TRAJECTORIES
        or science["grid_trajectories"] != GRID_TRAJECTORIES
        or science["calibration_pool_shared_by_all_methods"] is not True
        or tuple(science["grid_prefix_shared_by_grid_using_methods"])
        != ("MFCS", "PRC", "SC-PCP")
        or science["target_adaptation_trajectories"] != expected_adaptation
        or science["evaluation_trajectories"] != REFERENCE_TRAJECTORIES
        or science["bootstrap_resamples"] != BOOTSTRAP_RESAMPLES
        or science["bootstrap_seed"] != BOOTSTRAP_SEED
        or success
        != {
            "method": "SC-PCP",
            "selection_denominator": 20,
            "minimum_selected_seeds": MINIMUM_SELECTED_SEEDS,
            "minimum_selection_rate": MINIMUM_SELECTION_RATE,
            "minimum_primary_wsc_point": TARGET_COVERAGE,
            "confidence_interval_is_gating": False,
            "mean_coverage_is_gating": False,
            "width_is_gating": False,
        }
    ):
        raise ValueError("budget-follow-up science or success gate differs")

    canonical = payload["canonical_scpcp"]
    if canonical != {
        "source_path": CANONICAL_SCPCP_PATH.as_posix(),
        "source_sha256": CANONICAL_SCPCP_SHA256,
        "feasibility_rule": "estimated_coverage_greater_than_or_equal_to_one_minus_alpha",
        "mutation_permitted": False,
        "lcb_margin_or_radius_scaling_permitted": False,
        "v1_source_manifest": V1_SOURCE_MANIFEST.as_posix(),
        "required_unchanged_shared_paths": [
            path.as_posix() for path in UNCHANGED_SHARED_PATHS
        ],
    }:
        raise ValueError("canonical SC-PCP binding differs")
    if payload.get("source_diff_allowlist") != {
        "prior_source_manifest": V1_SOURCE_MANIFEST.as_posix(),
        "missing_prior_paths_permitted": False,
        "allowed_changed_paths": [path.as_posix() for path in V1_ALLOWED_CHANGED_PATHS],
        "allowed_added_paths": [path.as_posix() for path in V1_ALLOWED_ADDED_PATHS],
    }:
        raise ValueError("v1 source-diff allowlist differs")
    if payload["rng_contract"] != EXPECTED_RNG_CONTRACT:
        raise ValueError("budget-follow-up RNG contract differs")
    if payload.get("validation_claims") != VALIDATION_CLAIMS:
        raise ValueError("budget-follow-up validation claims differ")

    prior = payload["prior_v1"]
    config = BudgetFollowupConfig(
        prior_roots={
            "development": Path(prior["development_root"]),
            "confirmation": Path(prior["confirmation_root"]),
            "science": Path(prior["science_root"]),
        },
        prior_files={str(name): str(value) for name, value in prior["files"].items()},
    )
    config.validate()
    return config


def validate_runtime_protocol(protocol: object) -> None:
    """Fail closed unless the shared runner received the exact v2 budget adapter."""

    preset = protocol.datasets[DATASET]  # type: ignore[attr-defined]
    if (
        tuple(protocol.datasets) != (DATASET,)  # type: ignore[attr-defined]
        or protocol.split_fractions != ROLE_SPLIT  # type: ignore[attr-defined]
        or tuple(protocol.gammas) != GAMMAS  # type: ignore[attr-defined]
        or protocol.calibration_trajectories != CALIBRATION_TRAJECTORIES  # type: ignore[attr-defined]
        or protocol.grid_trajectories != GRID_TRAJECTORIES  # type: ignore[attr-defined]
        or protocol.reference_trajectories != REFERENCE_TRAJECTORIES  # type: ignore[attr-defined]
        or protocol.online_trajectories != ONLINE_TRAJECTORIES  # type: ignore[attr-defined]
        or protocol.bootstrap_resamples != BOOTSTRAP_RESAMPLES  # type: ignore[attr-defined]
        or tuple(preset.seeds) != SEEDS
        or preset.bootstrap_seed != BOOTSTRAP_SEED
        or preset.horizon != 6
    ):
        raise ValueError("runtime protocol differs from the frozen v2 adapter")


def verify_prior_v1_bindings(
    root: Path, config: BudgetFollowupConfig
) -> dict[str, Any]:
    """Verify the completed v1 lineage and its disclosed near-miss values."""

    observed: dict[str, str] = {}
    for label, expected in config.prior_files.items():
        root_name, relative = label.split("/", 1)
        path = root / config.prior_roots[root_name] / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing v1 prior artifact: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"v1 prior artifact changed: {path}")
        observed[label] = digest

    summary_path = root / config.prior_roots["science"] / "science/summary.json"
    summary = json.loads(summary_path.read_text())
    primary = next(
        row for row in summary["aggregates"] if float(row["gamma"]) == PRIMARY_GAMMA
    )
    scpcp = primary["methods"]["SC-PCP"]
    mfcs = primary["methods"]["MFCS"]
    if (
        len(summary["seeds_prespecified"]) != 20
        or len(summary["seeds_support_k0_eligible"]) != 19
        or summary["seeds_support_k0_eligible"]
        != [seed for seed in range(633_000, 633_200, 10) if seed != 633_120]
        or scpcp["target_marginal_worst_coverage"] != 0.8970868211043509
        or scpcp["target_mean_coverage"] != 0.9006083084825883
        or scpcp["mean_target_normalized_width"] != 7.33784418357046
        or scpcp["target_wsc_ci95"] != [0.8918578092204897, 0.9002578698490795]
        or mfcs["target_marginal_worst_coverage"] != 0.9137262858842549
        or mfcs["mean_target_normalized_width"] != 9.101523307331822
        or primary["paired_scpcp_comparisons"]["MFCS"][
            "scpcp_to_baseline_geometric_width_ratio"
        ]
        != 0.8242967947048239
    ):
        raise RuntimeError("v1 near-miss disclosure differs from the bound summary")
    return {
        "status": "verified_honest_v1_near_miss",
        "files": observed,
        "combined_sha256": _json_sha256(observed),
    }


def verify_canonical_scpcp(root: Path) -> dict[str, str]:
    """Match the active shared six-method implementation to the bound v1 snapshot."""

    manifest = json.loads((root / V1_SOURCE_MANIFEST).read_text())
    entries = {Path(entry["path"]): entry["sha256"] for entry in manifest["files"]}
    observed: dict[str, str] = {}
    for relative in UNCHANGED_SHARED_PATHS:
        expected = entries.get(relative)
        path = root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is None or digest != expected:
            raise RuntimeError(f"shared v1 implementation changed: {relative}")
        observed[relative.as_posix()] = digest
    if observed[CANONICAL_SCPCP_PATH.as_posix()] != CANONICAL_SCPCP_SHA256:
        raise RuntimeError("canonical SC-PCP implementation changed")
    return {
        "path": CANONICAL_SCPCP_PATH.as_posix(),
        "sha256": CANONICAL_SCPCP_SHA256,
        "shared_implementation_sha256": _json_sha256(observed),
    }


def verify_v1_source_diff_allowlist(root: Path) -> dict[str, Any]:
    """Prove that active executable/config deltas from v1 are exactly allowlisted."""

    manifest = json.loads((root / V1_SOURCE_MANIFEST).read_text())
    rows = manifest.get("files")
    if not isinstance(rows, list) or manifest.get("file_count") != len(rows):
        raise RuntimeError("v1 source manifest is malformed")
    prior = {Path(row["path"]): str(row["sha256"]) for row in rows}
    if len(prior) != len(rows) or len(prior) != 127:
        raise RuntimeError("v1 source manifest file identities differ")
    active_paths = [
        *sorted((root / "src/scpcp").rglob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
        *sorted((root / "tools").glob("*.py")),
        *sorted((root / "configs").glob("*.yaml")),
        root / "pyproject.toml",
    ]
    active = {
        path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in active_paths
    }
    missing = set(prior) - set(active)
    added = set(active) - set(prior)
    changed = {path for path in set(prior) & set(active) if prior[path] != active[path]}
    if (
        missing
        or changed != set(V1_ALLOWED_CHANGED_PATHS)
        or added != set(V1_ALLOWED_ADDED_PATHS)
    ):
        raise RuntimeError(
            "active source differs from the frozen v1 allowlist: "
            f"missing={sorted(map(str, missing))}, "
            f"changed={sorted(map(str, changed))}, "
            f"added={sorted(map(str, added))}"
        )
    delta_hashes = {
        path.as_posix(): active[path]
        for path in (*V1_ALLOWED_CHANGED_PATHS, *V1_ALLOWED_ADDED_PATHS)
    }
    return {
        "status": "exact_v1_source_diff_allowlist_verified",
        "prior_file_count": len(prior),
        "unchanged_prior_file_count": len(prior) - len(changed),
        "missing_prior_paths": [],
        "allowed_changed_paths": [path.as_posix() for path in V1_ALLOWED_CHANGED_PATHS],
        "allowed_added_paths": [path.as_posix() for path in V1_ALLOWED_ADDED_PATHS],
        "active_allowed_delta_sha256": _json_sha256(delta_hashes),
        "active_allowed_delta_file_sha256": delta_hashes,
    }


def summarize_precoverage(
    support_rows: Sequence[Mapping[str, Any]],
    k0_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen 19/20 joint support-and-K0 gate."""

    support = _rows_by_seed(support_rows)
    k0 = _rows_by_seed(k0_rows)
    support_passed = {seed: _strict_bool(support[seed].get("passed")) for seed in SEEDS}
    k0_results = {seed: _validated_k0(k0[seed]) for seed in SEEDS}
    joint = tuple(
        seed for seed in SEEDS if support_passed[seed] and k0_results[seed][0]
    )
    structural = sum(result[1] for result in k0_results.values())
    passed = (
        len(joint) >= MINIMUM_PRECOVERAGE_JOINT
        and structural == REQUIRED_STRUCTURAL_PASS_COUNT
    )
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "status": "PRECOVERAGE_GO" if passed else "PRECOVERAGE_NO_GO",
        "precoverage_admissible": passed,
        "prespecified_seed_count": len(SEEDS),
        "support_pass_count": sum(support_passed.values()),
        "k0_pass_count": sum(result[0] for result in k0_results.values()),
        "joint_pass_count": len(joint),
        "joint_pass_seeds": list(joint),
        "structural_pass_count": structural,
        "seed_deletions": 0,
        "coverage_generated": False,
    }


def primary_success_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen point-estimate gate without changing SC-PCP."""

    aggregates = summary.get("aggregates")
    if not isinstance(aggregates, Sequence):
        raise ValueError("science summary lacks gamma aggregates")
    matches = [row for row in aggregates if float(row["gamma"]) == PRIMARY_GAMMA]
    if len(matches) != 1:
        raise ValueError("science summary lacks the unique primary gamma")
    method = matches[0]["methods"]["SC-PCP"]
    selected = method.get("n_selected")
    selection_rate = method.get("selection_rate")
    wsc = method.get("target_marginal_worst_coverage")
    if (
        type(selected) is not int
        or not isinstance(selection_rate, Real)
        or isinstance(selection_rate, bool)
        or float(selection_rate) != selected / len(SEEDS)
        or wsc is None
        or not isinstance(wsc, Real)
        or isinstance(wsc, bool)
        or not math.isfinite(float(wsc))
    ):
        raise ValueError("primary SC-PCP summary is malformed")
    selection_passed = (
        selected >= MINIMUM_SELECTED_SEEDS
        and float(selection_rate) >= MINIMUM_SELECTION_RATE
    )
    wsc_passed = float(wsc) >= TARGET_COVERAGE
    success = selection_passed and wsc_passed
    return {
        "method": "SC-PCP",
        "gamma": PRIMARY_GAMMA,
        "status": "PRIMARY_SUCCESS" if success else "PRIMARY_NO_GO",
        "passed": success,
        "selection": {
            "selected_seeds": selected,
            "denominator": len(SEEDS),
            "selection_rate": float(selection_rate),
            "minimum_selected_seeds": MINIMUM_SELECTED_SEEDS,
            "minimum_selection_rate": MINIMUM_SELECTION_RATE,
            "passed": selection_passed,
        },
        "wsc": {
            "point_estimate": float(wsc),
            "minimum": TARGET_COVERAGE,
            "passed": wsc_passed,
        },
        "confidence_interval_is_gating": False,
        "mean_coverage_is_gating": False,
        "width_is_gating": False,
    }


def _rows_by_seed(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    if any(type(row.get("seed")) is not int for row in rows):
        raise ValueError("seed rows must use integer identities")
    indexed = {int(row["seed"]): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(SEEDS):
        raise ValueError("rows do not cover the exact frozen seed bank")
    return indexed


def _validated_k0(row: Mapping[str, Any]) -> tuple[bool, bool]:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("K0 row lacks metrics")
    ratio = normalized_k0_ratio(metrics)
    passed = ratio <= 1.0
    if _strict_bool(row.get("passed")) != passed:
        raise ValueError("K0 passed flag differs from its metrics")
    return passed, _strict_bool(metrics.get("structural_invariants"))


def _strict_bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("gate flag must be boolean")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
