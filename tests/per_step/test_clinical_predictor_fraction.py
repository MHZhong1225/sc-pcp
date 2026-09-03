from __future__ import annotations

import torch

from scpcp.controlled_clinical_extension import split_clinical_extension_roles
from scpcp.data import TrajectoryBatch
from scpcp.real_data import _RawClinicalBatch, _predictor_rows


def _raw(patient_ids: torch.Tensor) -> _RawClinicalBatch:
    rows = len(patient_ids)
    return _RawClinicalBatch(
        states=torch.zeros(rows, 3, 2),
        outcomes=torch.zeros(rows, 2, 1),
        treatments=torch.zeros(rows, 2, 1),
        patient_ids=patient_ids,
        episode_ids=patient_ids,
        static_indices=(),
        state_feature_names=("x0", "x1"),
    )


def _placeholder(raw: _RawClinicalBatch) -> TrajectoryBatch:
    return TrajectoryBatch(
        raw.states,
        torch.zeros(len(raw.patient_ids), 2, dtype=torch.long),
        raw.outcomes,
        raw.patient_ids,
    )


def test_predictor_rows_match_custom_controlled_role_exactly() -> None:
    patient_ids = torch.arange(1_993).repeat_interleave(2)
    raw = _raw(patient_ids)

    rows = _predictor_rows(raw, 631_000, predictor_fraction=0.20)
    roles = split_clinical_extension_roles(
        _placeholder(raw),
        seed=631_000,
        fractions=(0.20, 0.20, 0.60),
    )

    assert torch.equal(
        torch.unique(raw.patient_ids[rows], sorted=True),
        torch.unique(roles.predictor.patient_ids, sorted=True),
    )
    assert len(torch.unique(raw.patient_ids[rows])) == 398


def test_default_predictor_rows_are_bitwise_unchanged() -> None:
    raw = _raw(torch.arange(101).repeat_interleave(3))

    implicit = _predictor_rows(raw, 92_000)
    explicit = _predictor_rows(raw, 92_000, predictor_fraction=0.40)

    assert torch.equal(implicit, explicit)


def test_predictor_fraction_rejects_invalid_values() -> None:
    raw = _raw(torch.arange(5))
    for value in (0.0, 1.0, -0.1, float("nan"), float("inf")):
        try:
            _predictor_rows(raw, 1, predictor_fraction=value)
        except ValueError:
            continue
        raise AssertionError(f"invalid predictor_fraction accepted: {value}")
