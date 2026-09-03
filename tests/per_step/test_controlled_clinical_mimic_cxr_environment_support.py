from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import yaml

from scpcp.controlled_clinical_mimic_cxr_environment_support import (
    BRIDGE_CANDIDATE_ID,
    CONFIRMATION_BOOTSTRAP_SEED,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_BLOCKS,
    FROZEN_AT_UTC,
    K0_THRESHOLDS,
    ORIGINAL_CONFIRMATION_BOOTSTRAP_SEED,
    ORIGINAL_CONFIRMATION_SEEDS,
    ORIGINAL_FROZEN_AT_UTC,
    PRELAUNCH_AMENDMENT_ID,
    PROTOCOL,
    ROLE_SPLIT,
    PriorBinding,
    load_config,
    normalized_k0_ratio,
    summarize_confirmation,
    summarize_development,
    verify_prior_bindings,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "configs/controlled_clinical_mimic_cxr_environment_support_v1.yaml"
)


def _metrics(
    ratio: float = 0.5, *, structural_invariants: bool = True
) -> dict[str, Any]:
    return {
        name: threshold * ratio for name, threshold in K0_THRESHOLDS.items()
    } | {"structural_invariants": structural_invariants}


def _rows(
    seeds: Sequence[int],
    *,
    failed_support: set[int] | None = None,
    failed_k0: set[int] | None = None,
    structural_failures: set[int] | None = None,
    ratios: float | Mapping[int, float] = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failed_support = set() if failed_support is None else failed_support
    failed_k0 = set() if failed_k0 is None else failed_k0
    structural_failures = (
        set() if structural_failures is None else structural_failures
    )
    support_rows = [
        {"seed": seed, "passed": seed not in failed_support} for seed in seeds
    ]
    k0_rows = []
    for seed in seeds:
        ratio = ratios[seed] if isinstance(ratios, Mapping) else ratios
        if seed in failed_k0 and seed not in structural_failures:
            ratio = max(ratio, 1.01)
        k0_rows.append(
            {
                "seed": seed,
                "passed": seed not in failed_k0,
                "metrics": _metrics(
                    ratio,
                    structural_invariants=seed not in structural_failures,
                ),
            }
        )
    return support_rows, k0_rows


def _development_rows(
    *,
    failed_support_by_block: Mapping[str, set[int]] | None = None,
    failed_k0_by_block: Mapping[str, set[int]] | None = None,
    structural_failures_by_block: Mapping[str, set[int]] | None = None,
    ratio: float = 0.5,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    failed_support_by_block = (
        {} if failed_support_by_block is None else failed_support_by_block
    )
    failed_k0_by_block = {} if failed_k0_by_block is None else failed_k0_by_block
    structural_failures_by_block = (
        {}
        if structural_failures_by_block is None
        else structural_failures_by_block
    )
    support_by_block = {}
    k0_by_block = {}
    for block, seeds in DEVELOPMENT_BLOCKS.items():
        support, k0 = _rows(
            seeds,
            failed_support=failed_support_by_block.get(block, set()),
            failed_k0=failed_k0_by_block.get(block, set()),
            structural_failures=structural_failures_by_block.get(block, set()),
            ratios=ratio,
        )
        support_by_block[block] = support
        k0_by_block[block] = k0
    return support_by_block, k0_by_block


def test_config_freezes_the_exact_post_failure_protocol() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    config = load_config(CONFIG_PATH)

    assert list(payload) == [
        "protocol",
        "dataset",
        "study_role",
        "not_a_v7_bridge_repair",
        "prior_negative_evidence",
        "design_freeze",
        "prelaunch_integrity_amendment",
        "role_split",
        "environment",
        "development",
        "confirmation",
        "k0_gate",
        "donor_overlap_gate",
        "science",
        "information_firewall",
    ]
    assert payload["protocol"] == PROTOCOL
    assert payload["dataset"] == "mimic_cxr"
    assert payload["study_role"] == (
        "post_failure_prospectively_frozen_environment_support_reconstruction"
    )
    assert payload["not_a_v7_bridge_repair"] is True
    assert payload["prior_negative_evidence"] == {
        "v5_confirmation": {
            "root": (
                "results/work/controlled_clinical_fidelity_v5_mimic_cxr_"
                "confirmation"
            ),
            "final_status_sha256": (
                "5f104c0dff121174b52e5ce0c082583744d544cda47a296f5ad0329474472f18"
            ),
            "gate_sha256": (
                "d663b2c5b5d6a7280efe2dceb31c743b94a7ec674825caf3eaad7701d04e6a5b"
            ),
            "support_pass_count": 20,
            "k0_pass_count": 18,
            "structural_pass_count": 20,
        },
        "v6_development": {
            "root": (
                "results/work/controlled_clinical_fidelity_v6_mimic_cxr_"
                "development"
            ),
            "final_status_sha256": (
                "39c014b9429466849b709a90739ae1b88d72d6eec43f3425ef2281d48fa058a1"
            ),
            "development_gate_sha256": (
                "45a9768412fc4fae47384581b0235a43e21a0d965b25782b0986603a9e8aa4bc"
            ),
            "terminal_no_v7": True,
        },
    }
    assert payload["design_freeze"] == {
        "original_frozen_at_utc": ORIGINAL_FROZEN_AT_UTC,
        "frozen_at_utc": FROZEN_AT_UTC,
        "rationale": (
            "sparse_hypoxemia_tail_is_concentrated_in_four_patients_and_requires_"
            "a_larger_environment_library"
        ),
        "coverage_or_width_inspected": False,
        "development_is_scientifically_fresh": False,
        "development_pilot_visible_at_freeze": {
            "block_a": [631_000, 631_010, 631_020, 631_030, 631_040],
            "block_b": [631_200, 631_210, 631_220, 631_230, 631_240],
        },
        "changes_after_freeze": [PRELAUNCH_AMENDMENT_ID],
        "further_changes_permitted": False,
    }
    assert payload["prelaunch_integrity_amendment"] == {
        "amendment_id": PRELAUNCH_AMENDMENT_ID,
        "amended_at_utc": FROZEN_AT_UTC,
        "timing": "before_formal_launch",
        "evidence_opened_before_amendment": {
            "development_support_k0_pilots": 10,
            "formal_development_run": False,
            "formal_confirmation_run": False,
            "confirmation_support_or_k0": False,
            "coverage_mean_coverage_width_or_selection": False,
        },
        "scientific_design_changed": False,
        "performance_tuning": False,
        "only_protocol_parameter_change": "confirmation_rng_identity",
        "confirmation_rng_bank": {
            "voided_seeds": {"start": 632_000, "stop": 632_200, "step": 10},
            "voided_bootstrap_seed": ORIGINAL_CONFIRMATION_BOOTSTRAP_SEED,
            "voided_bank_status": "unconsumed_and_invalidated",
            "invalidation_reason": (
                "rng_collision_with_development_block_b_cxr_encoder"
            ),
            "collision_rule": (
                "confirmation_outcome_model_seed_plus_1_equals_development_cxr_"
                "encoder_seed_plus_701"
            ),
            "collision_count": 10,
            "collision_rng_ids": [
                632_001,
                632_011,
                632_021,
                632_031,
                632_041,
                632_051,
                632_061,
                632_071,
                632_081,
                632_091,
            ],
            "replacement_seeds": {
                "start": 633_000,
                "stop": 633_200,
                "step": 10,
            },
            "replacement_bootstrap_seed": CONFIRMATION_BOOTSTRAP_SEED,
        },
        "replacement_rng_audit": {
            "historical_artifact_rng_id_count": 7_200,
            "historical_source_rng_id_count": 1_488,
            "historical_union_rng_id_count": 7_947,
            "historical_union_rng_id_sha256": (
                "66720aecf35b5f7b47200c488e6e01ac0428277c1783cfb068efb244f1518655"
            ),
            "full_confirmation_stream_count": 341,
            "full_confirmation_mapping_sha256": (
                "c31fcc88d50e9a5b9d6ed94be7b6864b79d25c6c475b10af7f7291ddce22b6cb"
            ),
            "full_confirmation_internal_collision_count": 0,
            "historical_collision_count": 0,
            "development_actual_stream_count": 200,
            "development_actual_mapping_sha256": (
                "7d5140bd3a2e42399edb394ee3167583e4bae6cde2a1f5768bddc4278cae9108"
            ),
            "confirmation_precoverage_stream_count": 100,
            "confirmation_precoverage_mapping_sha256": (
                "1ffe184c6deba56207e2949a37c55d9fd845da5f93e65282cd4fe0a289a58bd3"
            ),
            "development_actual_collision_count": 0,
            "bootstrap_historical_collision": False,
            "bootstrap_development_actual_collision": False,
        },
        "encoder_training_scope_audit": {
            "status": "passed_after_prelaunch_contract_conformance_fix",
            "predictor_fraction": 0.20,
            "patient_permutation": (
                "same_seeded_sorted_unique_patient_permutation_as_three_role_split"
            ),
            "training_rows": (
                "D_pred_patient_ids_intersect_official_mimic_cxr_train_mask"
            ),
            "fidelity_patients_train_encoder": False,
            "environment_patients_train_encoder": False,
            "scientific_parameter_change": False,
            "performance_evidence_used": False,
        },
    }
    assert payload["role_split"] == {
        "predictor": 0.20,
        "fidelity": 0.20,
        "environment": 0.60,
        "patient_disjoint": True,
        "expected_patient_counts_for_frozen_cohort": {
            "predictor": 398,
            "fidelity": 398,
            "environment": 1197,
        },
    }
    assert payload["environment"] == {
        "bridge_candidate_id": BRIDGE_CANDIDATE_ID,
        "candidate_count": 1,
        "selector_present": False,
        "bridge_search_permitted": False,
        "state_kernel": {
            "metric": "raw",
            "neighbors": 10_000,
            "uses_full_cell": True,
            "donor_weighting": "uniform",
            "bandwidth": 2.0,
            "transition_mode": "ridge_residual",
            "outcome_residual_mode": "raw",
            "ridge_mode": "sample_normalized_no_intercept",
            "ridge_value": 0.001,
        },
    }
    assert payload["development"] == {
        "block_a": {"start": 631_000, "stop": 631_200, "step": 10},
        "block_b": {"start": 631_200, "stop": 631_400, "step": 10},
        "minimum_joint_pass_count_by_block": 19,
        "minimum_joint_pass_count_total": 39,
        "required_structural_pass_count_total": 40,
        "maximum_q95_normalized_k0_ratio": 0.95,
        "seed_deletion_permitted": False,
        "scientific_outputs_permitted": False,
    }
    assert payload["confirmation"] == {
        "seeds": {"start": 633_000, "stop": 633_200, "step": 10},
        "bootstrap_seed": 63_300_019,
        "minimum_joint_pass_count": 19,
        "required_structural_pass_count": 20,
        "seed_deletion_permitted": False,
        "independent_patient_confirmation_claimed": False,
    }
    assert payload["k0_gate"] == {
        "systematic_replays": 16,
        **K0_THRESHOLDS,
        "active_coordinate_sd_floor": 0.0001,
    }
    assert payload["donor_overlap_gate"] == {
        "gamma": -4.0,
        "probe_radius_fractions": [0.5, 1.0],
        "probe_trajectories": 3_000,
        "local_ess_p01": 10.0,
        "median_ess_fraction": 0.25,
        "maximum_donor_probability": 0.25,
    }
    assert payload["science"] == {
        "gammas": [-4.0, -2.0, 0.0, 2.0, 4.0],
        "primary_gamma": -4.0,
        "methods": ["Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP"],
        "calibration_trajectories": 3_000,
        "grid_trajectories": 1_000,
        "target_adaptation_trajectories": {
            "Standard CP": 0,
            "ACI": 2_000,
            "MFCS": 0,
            "SPCI": 2_000,
            "PRC": 2_000,
            "SC-PCP": 0,
        },
        "evaluation_trajectories": 20_000,
        "bootstrap_resamples": 10_000,
        "primary_metric": "min_t mean_seed(C_seed,t)",
        "mean_coverage_is_supplementary": True,
    }
    assert payload["information_firewall"] == {
        "precoverage_allowed": [
            "support",
            "k0_fidelity",
            "context_identity",
            "provenance",
            "descriptive_diagnostics",
        ],
        "precoverage_forbidden": [
            "science",
            "coverage",
            "mean_coverage",
            "width",
            "method_selection",
        ],
    }

    assert config.development_blocks == DEVELOPMENT_BLOCKS
    assert config.confirmation_seeds == CONFIRMATION_SEEDS
    assert config.confirmation_bootstrap_seed == CONFIRMATION_BOOTSTRAP_SEED
    assert ORIGINAL_CONFIRMATION_SEEDS == tuple(range(632_000, 632_200, 10))
    assert config.original_frozen_at_utc == ORIGINAL_FROZEN_AT_UTC
    assert config.frozen_at_utc == FROZEN_AT_UTC
    assert config.changes_after_freeze == (PRELAUNCH_AMENDMENT_ID,)
    assert ROLE_SPLIT == (0.20, 0.20, 0.60)
    assert config.pilot_visible_at_freeze == {
        "block_a": DEVELOPMENT_BLOCKS["block_a"][:5],
        "block_b": DEVELOPMENT_BLOCKS["block_b"][:5],
    }


@pytest.mark.parametrize(
    ("field_path", "changed_value"),
    [
        (("study_role",), "retrospective_repair"),
        (("prior_negative_evidence", "v5_confirmation", "k0_pass_count"), 19),
        (("prior_negative_evidence", "v6_development", "terminal_no_v7"), False),
        (("design_freeze", "coverage_or_width_inspected"), True),
        (
            ("design_freeze", "changes_after_freeze"),
            [PRELAUNCH_AMENDMENT_ID, "unrecorded_change"],
        ),
        (
            ("prelaunch_integrity_amendment", "performance_tuning"),
            True,
        ),
        (
            (
                "prelaunch_integrity_amendment",
                "confirmation_rng_bank",
                "collision_count",
            ),
            9,
        ),
        (
            (
                "prelaunch_integrity_amendment",
                "encoder_training_scope_audit",
                "fidelity_patients_train_encoder",
            ),
            True,
        ),
        (
            (
                "role_split",
                "expected_patient_counts_for_frozen_cohort",
                "environment",
            ),
            1_196,
        ),
        (("environment", "state_kernel", "neighbors"), 9_999),
        (("development", "minimum_joint_pass_count_total"), 38),
        (("confirmation", "minimum_joint_pass_count"), 18),
        (("k0_gate", "systematic_replays"), 15),
        (("donor_overlap_gate", "gamma"), -2.0),
        (("science", "primary_gamma"), -2.0),
        (
            ("information_firewall", "precoverage_forbidden"),
            ["science", "coverage", "mean_coverage", "method_selection"],
        ),
        (("unexpected_field",), True),
    ],
)
def test_load_config_rejects_changes_to_any_frozen_protocol_section(
    tmp_path: Path,
    field_path: tuple[str, ...],
    changed_value: object,
) -> None:
    payload = deepcopy(yaml.safe_load(CONFIG_PATH.read_text()))
    destination = payload
    for key in field_path[:-1]:
        destination = destination[key]
    destination[field_path[-1]] = changed_value
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="config differs from the frozen contract"):
        load_config(changed_path)


def test_prior_negative_evidence_is_bound_by_exact_hashes() -> None:
    config = load_config(CONFIG_PATH)
    observed = verify_prior_bindings(ROOT, config)

    assert observed["v5_confirmation"] == {
        "root": "results/work/controlled_clinical_fidelity_v5_mimic_cxr_confirmation",
        "files": {
            "FINAL_STATUS.json": (
                "5f104c0dff121174b52e5ce0c082583744d544cda47a296f5ad0329474472f18"
            ),
            "gate.json": (
                "d663b2c5b5d6a7280efe2dceb31c743b94a7ec674825caf3eaad7701d04e6a5b"
            ),
        },
    }
    assert observed["v6_development"] == {
        "root": "results/work/controlled_clinical_fidelity_v6_mimic_cxr_development",
        "files": {
            "FINAL_STATUS.json": (
                "39c014b9429466849b709a90739ae1b88d72d6eec43f3425ef2281d48fa058a1"
            ),
            "development_gate.json": (
                "45a9768412fc4fae47384581b0235a43e21a0d965b25782b0986603a9e8aa4bc"
            ),
        },
    }
    assert len(observed["combined_sha256"]) == 64

    wrong_v5 = replace(
        config,
        prior_v5=PriorBinding(
            root=config.prior_v5.root,
            files={**config.prior_v5.files, "gate.json": "0" * 64},
        ),
    )
    with pytest.raises(RuntimeError, match="prior negative evidence changed"):
        verify_prior_bindings(ROOT, wrong_v5)


def test_normalized_k0_ratio_uses_the_largest_unchanged_threshold_ratio() -> None:
    metrics = _metrics(0.5)
    metrics["maximum_signed_residual_w1"] = 0.225
    assert normalized_k0_ratio(metrics) == pytest.approx(0.9)
    assert math.isinf(
        normalized_k0_ratio(_metrics(0.5, structural_invariants=False))
    )


def test_development_accepts_total_39_and_q95_at_the_exact_boundary() -> None:
    first_seed = DEVELOPMENT_BLOCKS["block_a"][0]
    support, k0 = _development_rows(
        failed_support_by_block={"block_a": {first_seed}},
        ratio=0.95,
    )

    summary = summarize_development(support, k0)

    assert summary["development_admissible"] is True
    assert summary["status"] == "DEVELOPMENT_GO"
    assert summary["blocks"]["block_a"]["joint_pass_count"] == 19
    assert summary["blocks"]["block_b"]["joint_pass_count"] == 20
    assert summary["joint_pass_count_total"] == 39
    assert summary["structural_pass_count_total"] == 40
    assert summary["q95_normalized_k0_ratio"] == pytest.approx(0.95)
    assert summary["candidate_count"] == 1
    assert summary["selector_present"] is False


def test_development_rejects_total_38_even_when_both_blocks_reach_19() -> None:
    failed_support = {
        block: {seeds[0]} for block, seeds in DEVELOPMENT_BLOCKS.items()
    }
    support, k0 = _development_rows(
        failed_support_by_block=failed_support,
        ratio=0.5,
    )

    summary = summarize_development(support, k0)

    assert [
        summary["blocks"][block]["joint_pass_count"]
        for block in DEVELOPMENT_BLOCKS
    ] == [19, 19]
    assert summary["joint_pass_count_total"] == 38
    assert summary["development_admissible"] is False
    assert summary["status"] == "DEVELOPMENT_NO_GO"


def test_development_rejects_q95_above_095_with_every_joint_seed_passing() -> None:
    support, k0 = _development_rows(ratio=0.950_001)

    summary = summarize_development(support, k0)

    assert summary["joint_pass_count_total"] == 40
    assert summary["structural_pass_count_total"] == 40
    assert summary["q95_normalized_k0_ratio"] > 0.95
    assert summary["development_admissible"] is False


def test_development_rejects_one_structural_failure_at_total_39() -> None:
    failed_seed = DEVELOPMENT_BLOCKS["block_a"][0]
    support, k0 = _development_rows(
        failed_k0_by_block={"block_a": {failed_seed}},
        structural_failures_by_block={"block_a": {failed_seed}},
        ratio=0.5,
    )

    summary = summarize_development(support, k0)

    assert summary["joint_pass_count_total"] == 39
    assert summary["structural_pass_count_total"] == 39
    assert math.isinf(summary["q95_normalized_k0_ratio"])
    assert summary["development_admissible"] is False


@pytest.mark.parametrize(
    ("failed_support_indices", "failed_k0_indices", "structural_indices", "joint", "admissible"),
    [
        ({0}, set(), set(), 19, True),
        ({0}, {1}, set(), 18, False),
        (set(), {0}, {0}, 19, False),
    ],
)
def test_confirmation_requires_19_joint_passes_and_all_structural_invariants(
    failed_support_indices: set[int],
    failed_k0_indices: set[int],
    structural_indices: set[int],
    joint: int,
    admissible: bool,
) -> None:
    failed_support = {CONFIRMATION_SEEDS[index] for index in failed_support_indices}
    failed_k0 = {CONFIRMATION_SEEDS[index] for index in failed_k0_indices}
    structural_failures = {
        CONFIRMATION_SEEDS[index] for index in structural_indices
    }
    support, k0 = _rows(
        CONFIRMATION_SEEDS,
        failed_support=failed_support,
        failed_k0=failed_k0,
        structural_failures=structural_failures,
    )

    summary = summarize_confirmation(support, k0)

    assert summary["joint_pass_count"] == joint
    assert summary["confirmation_admissible"] is admissible
    assert summary["status"] == (
        "CONFIRMATION_GO" if admissible else "CONFIRMATION_NO_GO"
    )


def test_gate_summaries_reject_incomplete_or_duplicate_seed_rows() -> None:
    support, k0 = _rows(CONFIRMATION_SEEDS)
    with pytest.raises(ValueError, match="exact prespecified seed set"):
        summarize_confirmation(support[:-1], k0)

    duplicate = [*support[:-1], support[0]]
    with pytest.raises(ValueError, match="exact prespecified seed set"):
        summarize_confirmation(duplicate, k0)


def test_gate_summaries_reject_truthy_non_boolean_support_flags() -> None:
    support, k0 = _rows(CONFIRMATION_SEEDS)
    support[0]["passed"] = 1

    with pytest.raises(ValueError, match="support passed flag must be boolean"):
        summarize_confirmation(support, k0)


@pytest.mark.parametrize(
    ("artifact_passed", "ratio", "structural_invariants"),
    [
        (True, 1.01, True),
        (False, 0.5, True),
        (True, 0.5, False),
    ],
)
def test_confirmation_rejects_k0_pass_flags_inconsistent_with_metrics(
    artifact_passed: bool,
    ratio: float,
    structural_invariants: bool,
) -> None:
    support, k0 = _rows(CONFIRMATION_SEEDS)
    k0[0]["passed"] = artifact_passed
    k0[0]["metrics"] = _metrics(
        ratio,
        structural_invariants=structural_invariants,
    )

    with pytest.raises(ValueError, match="K0 passed flag disagrees"):
        summarize_confirmation(support, k0)


def test_development_rejects_non_boolean_k0_pass_flags() -> None:
    support, k0 = _development_rows()
    k0["block_a"][0]["passed"] = "true"

    with pytest.raises(ValueError, match="K0 passed flag must be boolean"):
        summarize_development(support, k0)
