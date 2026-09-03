from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from scpcp.controlled_clinical_fidelity_v4 import (
    CONTROLLED_TRANSITION_DEFAULTS,
    DATASETS,
    DEVELOPMENT_MINIMUM_PASS_COUNT,
    FROZEN_ANCHORS,
    METRIC_THRESHOLDS,
    REPAIR_DATASETS,
    RepairTheta,
    load_fidelity_v4_config,
    normalized_seed_ratio,
    repair_candidates,
    select_dataset_candidate,
    summarize_candidate_dataset,
    summarize_seed_ratios,
    validate_controlled_transition_default_parity,
    validate_parent_v3_bundle,
)
from scpcp.controlled_transition import ControlledResidualEnvironment


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/controlled_clinical_fidelity_v4.yaml"


def _metrics(ratio: float, *, structural: bool = True) -> dict[str, object]:
    return {
        "maximum_score_ks": 0.10 * ratio,
        "maximum_signed_residual_w1": 0.25 * ratio,
        "maximum_successor_mean_w1": 0.25 * ratio,
        "maximum_successor_q95_w1": 0.50 * ratio,
        "structural_invariants": structural,
    }


def _summaries(
    dataset: str,
    *,
    default_ratios: tuple[float, ...] = (0.8,) * 20,
) -> dict[str, object]:
    return {
        candidate.candidate_id: _ratio_summary(candidate, default_ratios)
        for candidate in repair_candidates(dataset)
    }


def _ratio_summary(
    candidate: RepairTheta,
    ratios: tuple[float, ...],
    *,
    structural_pass_flags: tuple[bool, ...] = (True,) * 20,
):
    return summarize_seed_ratios(
        candidate,
        ratios,
        structural_pass_flags=structural_pass_flags,
    )


def _find_candidate(dataset: str, **values: object) -> RepairTheta:
    matches = [
        candidate
        for candidate in repair_candidates(dataset)
        if all(getattr(candidate, name) == value for name, value in values.items())
    ]
    assert len(matches) == 1
    return matches[0]


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def test_config_binds_exact_completed_v3_parent_and_frozen_anchors() -> None:
    config = load_fidelity_v4_config(CONFIG_PATH)

    status = validate_parent_v3_bundle(config, workspace_root=ROOT)
    assert status["status"] == "DEVELOPMENT_NO_GO"
    assert FROZEN_ANCHORS["mimic_iv"].theta_id == (
        "A03_raw_k200_gaussian_b2__normalized_ridge_1e-2"
    )
    assert FROZEN_ANCHORS["inspire"].theta_id == (
        "A03_raw_k200_gaussian_b2__raw_ridge_1e-3"
    )
    assert all(anchor.development_pass_count == 20 for anchor in FROZEN_ANCHORS.values())


def test_config_freezes_fresh_independent_confirmation_banks() -> None:
    config = load_fidelity_v4_config(CONFIG_PATH)

    assert config.confirmation_seeds == {
        "mimic_iv": tuple(range(115_000, 115_200, 10)),
        "eicu": tuple(range(116_000, 116_200, 10)),
        "inspire": tuple(range(117_000, 117_200, 10)),
        "mimic_cxr": tuple(range(118_000, 118_200, 10)),
    }
    assert config.confirmation_bootstrap_seeds == {
        "mimic_iv": 11_500_019,
        "eicu": 11_600_019,
        "inspire": 11_700_019,
        "mimic_cxr": 11_800_019,
    }
    assert config.confirmation_mapping_sha256 == (
        "3a78ec5afe69f57928de894a38803f5c369b33ab1db3f7c37bd403b974f75c72"
    )
    assert all(len(seeds) == 20 for seeds in config.confirmation_seeds.values())
    assert len(set().union(*map(set, config.confirmation_seeds.values()))) == 80
    assert tuple(config.confirmation_seeds) == DATASETS


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("parent_v3", "required_status"), "DEVELOPMENT_GO", "parent binding"),
        (("repair_grids", "eicu", "ridge_value"), 1e-2, "repair grid"),
        (("k0_gate", "maximum_successor_q95_w1"), 0.55, "K0 thresholds"),
        (("selection", "cross_dataset_conjunction_permitted"), True, "selector"),
    ),
)
def test_config_is_fail_closed(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    target = payload
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match=message):
        load_fidelity_v4_config(changed)


def test_dataset_grids_are_exact_and_keep_fixed_ridges() -> None:
    eicu = repair_candidates("eicu")
    cxr = repair_candidates("mimic_cxr")

    assert len(eicu) == 14
    assert len(cxr) == 28
    assert eicu[0].candidate_id == (
        "E00_raw_k200_gaussian_b2_ridge_residual_standardized"
    )
    assert eicu[-1].candidate_id == (
        "E13_stagewise_zscore_k10000_gaussian_b2_local_delta_standardized"
    )
    assert cxr[-1].candidate_id == (
        "C27_stagewise_zscore_k10000_gaussian_b2_local_delta_raw"
    )
    assert {candidate.ridge_value for candidate in eicu} == {1e-3}
    assert {candidate.ridge_value for candidate in cxr} == {1e-3}
    assert {candidate.ridge_mode for candidate in (*eicu, *cxr)} == {
        "sample_normalized_no_intercept"
    }
    assert {candidate.outcome_residual_mode for candidate in eicu} == {
        "standardized"
    }
    assert {candidate.outcome_residual_mode for candidate in cxr} == {
        "standardized",
        "raw",
    }
    assert any(candidate.uses_full_cell for candidate in (*eicu, *cxr))
    assert not any(
        candidate.metric == "stagewise_zscore"
        and candidate.uses_full_cell
        and candidate.weight == "uniform"
        for candidate in (*eicu, *cxr)
    )


def test_default_parity_keeps_inherited_anchors_executable() -> None:
    config = load_fidelity_v4_config(CONFIG_PATH)
    parameters = inspect.signature(ControlledResidualEnvironment).parameters

    assert CONTROLLED_TRANSITION_DEFAULTS == {
        "transition_mode": "ridge_residual",
        "outcome_residual_mode": "standardized",
    }
    assert parameters["transition_mode"].default == "ridge_residual"
    assert parameters["outcome_residual_mode"].default == "standardized"
    assert all(
        anchor.transition_mode == parameters["transition_mode"].default
        and anchor.outcome_residual_mode
        == parameters["outcome_residual_mode"].default
        for anchor in FROZEN_ANCHORS.values()
    )
    validate_controlled_transition_default_parity(config, workspace_root=ROOT)


def test_candidate_identity_and_complete_grid_are_fail_closed() -> None:
    candidate = repair_candidates("eicu")[0]
    with pytest.raises(ValueError, match="ID does not match"):
        RepairTheta(
            dataset=candidate.dataset,
            candidate_id="E99_tampered",
            metric=candidate.metric,
            neighbors=candidate.neighbors,
            weight=candidate.weight,
            transition_mode=candidate.transition_mode,
            outcome_residual_mode=candidate.outcome_residual_mode,
        )

    candidates = repair_candidates("eicu")
    with pytest.raises(ValueError, match="complete frozen dataset grid"):
        select_dataset_candidate("eicu", candidates[:-1], _summaries("eicu"))
    with pytest.raises(ValueError, match="not defined"):
        repair_candidates("mimic_iv")


def test_k0_ratio_keeps_v3_thresholds_and_requires_exact_metric_schema() -> None:
    assert METRIC_THRESHOLDS == {
        "maximum_score_ks": 0.10,
        "maximum_signed_residual_w1": 0.25,
        "maximum_successor_mean_w1": 0.25,
        "maximum_successor_q95_w1": 0.50,
    }
    assert normalized_seed_ratio(_metrics(0.8)) == pytest.approx(0.8)
    assert math.isinf(normalized_seed_ratio(_metrics(0.1, structural=False)))
    extra = {**_metrics(0.8), "mean_width": 1.0}
    with pytest.raises(ValueError, match="exact schema"):
        normalized_seed_ratio(extra)
    missing = _metrics(0.8)
    missing.pop("maximum_score_ks")
    with pytest.raises(ValueError, match="exact schema"):
        normalized_seed_ratio(missing)


def test_summary_uses_exact_linear_q95_and_structural_failures() -> None:
    candidate = repair_candidates("eicu")[0]
    values = tuple(float(value) for value in np.linspace(0.1, 1.0, 20))
    summary = summarize_candidate_dataset(candidate, [_metrics(value) for value in values])

    assert summary.pass_count == 20
    assert summary.q95_seed_ratio == pytest.approx(
        np.quantile(values, 0.95, method="linear")
    )
    failed = summarize_candidate_dataset(
        candidate,
        [*[_metrics(value) for value in values[:-1]], _metrics(0.1, structural=False)],
    )
    assert failed.pass_count == 19
    assert math.isinf(failed.q95_seed_ratio)
    assert failed.to_dict()["q95_seed_ratio"] is None


def test_selector_is_per_dataset_and_obeys_frozen_lexicographic_order() -> None:
    candidates = repair_candidates("eicu")
    summaries = _summaries("eicu", default_ratios=(0.99,) * 20)
    tempting_failure = candidates[-1]
    summaries[tempting_failure.candidate_id] = _ratio_summary(
        tempting_failure,
        (*([0.1] * 19), 1.01),
    )
    winner = select_dataset_candidate("eicu", candidates, summaries)

    assert winner["winner_summary"]["pass_count"] == 20
    assert winner["dataset"] == "eicu"
    assert winner["development_admissible"] is True

    wrong_dataset = copy.copy(summaries)
    wrong = repair_candidates("mimic_cxr")[0]
    wrong_dataset[candidates[0].candidate_id] = _ratio_summary(
        wrong, (0.8,) * 20
    )
    with pytest.raises(ValueError, match="identity"):
        select_dataset_candidate("eicu", candidates, wrong_dataset)


def test_minimal_change_and_nineteen_of_twenty_gate_are_explicit() -> None:
    candidates = repair_candidates("eicu")
    summaries = _summaries("eicu", default_ratios=(*([0.8] * 18), 1.1, 1.2))
    reference = _find_candidate(
        "eicu",
        metric="raw",
        neighbors=200,
        weight="uniform",
        transition_mode="ridge_residual",
        outcome_residual_mode="standardized",
    )
    summaries[reference.candidate_id] = _ratio_summary(
        reference, (*([0.8] * 19), 1.1)
    )
    decision = select_dataset_candidate("eicu", candidates, summaries)

    assert DEVELOPMENT_MINIMUM_PASS_COUNT == 19
    assert decision["winner"]["candidate_id"] == reference.candidate_id
    assert decision["winner_summary"]["pass_count"] == 19
    assert decision["status"] == "DATASET_DEVELOPMENT_GO"

    all_eighteen = _summaries(
        "mimic_cxr", default_ratios=(*([0.8] * 18), 1.1, 1.2)
    )
    no_go = select_dataset_candidate(
        "mimic_cxr", repair_candidates("mimic_cxr"), all_eighteen
    )
    assert no_go["development_admissible"] is False
    assert no_go["status"] == "DATASET_DEVELOPMENT_NO_GO"


def test_structural_pass_count_must_be_twenty_even_at_nineteen_metric_passes() -> None:
    candidates = repair_candidates("eicu")
    summaries = _summaries("eicu", default_ratios=(*([0.8] * 18), 1.1, 1.2))
    candidate = candidates[0]
    summaries[candidate.candidate_id] = _ratio_summary(
        candidate,
        (*([0.8] * 19), math.inf),
        structural_pass_flags=(*([True] * 19), False),
    )

    decision = select_dataset_candidate("eicu", candidates, summaries)

    assert decision["winner_summary"]["pass_count"] == 19
    assert decision["winner_summary"]["structural_pass_count"] == 19
    assert decision["development_required_structural_pass_count"] == 20
    assert decision["development_admissible"] is False


@pytest.mark.parametrize("dataset", REPAIR_DATASETS)
def test_contract_outputs_contain_only_k0_selection_fields(dataset: str) -> None:
    output = select_dataset_candidate(
        dataset,
        repair_candidates(dataset),
        _summaries(dataset),
    )

    forbidden = {"coverage", "stage_coverage", "mean_width", "science", "method_selection"}
    assert _all_keys(output).isdisjoint(forbidden)
