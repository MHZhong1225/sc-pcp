from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scpcp.controlled_clinical_extension import (
    DATASET_NAMES,
    GAMMAS,
    METHODS,
    DonorOverlapMetrics,
    K0FidelityMetrics,
    donor_overlap_passes,
    empirical_ks,
    equal_sample_wasserstein_1,
    evaluate_support_gate,
    k0_fidelity_passes,
    load_extension_config,
    setting_availability_passes,
    split_clinical_extension_roles,
    unique_patient_action_counts,
)
from scpcp.data import TrajectoryBatch


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "controlled_clinical_extension.yaml"


def test_frozen_extension_config_matches_the_registered_protocol() -> None:
    config = load_extension_config(CONFIG_PATH)

    assert config.protocol == "controlled_clinical_extension_v2"
    assert tuple(config.datasets) == DATASET_NAMES == (
        "mimic_iv",
        "eicu",
        "inspire",
        "mimic_cxr",
    )
    assert config.split_fractions == (0.40, 0.20, 0.40)
    assert config.gammas == GAMMAS == (-4.0, -2.0, 0.0, 2.0, 4.0)
    assert METHODS == (
        "Standard CP",
        "ACI",
        "MFCS",
        "SPCI",
        "PRC",
        "SC-PCP",
    )
    assert (
        config.calibration_trajectories,
        config.grid_trajectories,
        config.reference_trajectories,
        config.online_trajectories,
    ) == (3_000, 1_000, 20_000, 2_000)
    assert config.bootstrap_resamples == 10_000
    assert (
        config.q_low_source_quantile,
        config.q_high_source_quantile,
        config.alternative_policy_tilt,
        config.maximum_policy_response,
        config.policy_ratio_cap,
    ) == (0.80, 0.95, 20.0, 1.0, 3.0)
    assert config.k0_fidelity_gate.systematic_replays == 16
    assert config.donor_overlap_gate.probe_radius_fractions == (0.50, 1.00)


def test_dataset_presets_lock_horizons_stages_and_fresh_seed_banks() -> None:
    config = load_extension_config(CONFIG_PATH)
    expected = {
        "mimic_iv": (12, tuple(range(4, 12)), tuple(range(93_600, 93_800, 10)), 9_361_019),
        "eicu": (12, tuple(range(4, 12)), tuple(range(92_000, 92_200, 10)), 9_201_019),
        "inspire": (12, tuple(range(4, 12)), tuple(range(92_300, 92_500, 10)), 9_231_019),
        "mimic_cxr": (6, tuple(range(2, 6)), tuple(range(92_600, 92_800, 10)), 9_261_019),
    }

    for name, (horizon, late_stages, seeds, bootstrap_seed) in expected.items():
        preset = config.datasets[name]
        assert preset.name == name
        assert preset.horizon == horizon
        assert preset.late_stages == late_stages
        assert preset.seeds == seeds
        assert preset.bootstrap_seed == bootstrap_seed
        assert len(preset.seeds) == len(set(preset.seeds)) == 20

    all_seeds = [seed for preset in config.datasets.values() for seed in preset.seeds]
    assert len(all_seeds) == len(set(all_seeds)) == 80


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split_fractions", (0.40, 0.15, 0.45), "clinical split"),
        ("gammas", (-4.0, 0.0, 4.0), "gamma grid"),
        ("calibration_trajectories", 2_999, "trajectory budgets"),
        ("policy_ratio_cap", 2.0, "policy constants"),
    ],
)
def test_config_validation_fails_closed_on_scientific_changes(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_extension_config(CONFIG_PATH)

    with pytest.raises(ValueError, match=message):
        replace(config, **{field: value}).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_config", Path("configs/not_the_frozen_eicu.yaml")),
        ("horizon", 13),
        ("late_stage_start", 5),
        ("seeds", tuple(range(120_000, 120_200, 10))),
        ("bootstrap_seed", 9_201_020),
    ],
)
def test_every_dataset_preset_field_is_frozen(field: str, value: object) -> None:
    config = load_extension_config(CONFIG_PATH)
    datasets = dict(config.datasets)
    datasets["eicu"] = replace(datasets["eicu"], **{field: value})

    with pytest.raises(ValueError, match="dataset presets differ"):
        replace(config, datasets=datasets).validate()


def test_support_gate_records_every_failed_stage_action_cell() -> None:
    config = load_extension_config(CONFIG_PATH)
    counts = [
        [20, 21, 19],
        [25, 18, 20],
    ]

    result = evaluate_support_gate(counts, config.support_gate)

    assert result.passed is False
    assert result.minimum_unique_patients == 18
    assert result.failed_cells == ((0, 2, 19), (1, 1, 18))
    assert evaluate_support_gate([[20, 100], [21, 20]], config.support_gate).passed


def test_clinical_split_is_exactly_patient_disjoint_and_deterministic() -> None:
    patient_ids = torch.arange(10).repeat_interleave(2)
    batch = TrajectoryBatch(
        states=torch.arange(20 * 3 * 2, dtype=torch.float32).reshape(20, 3, 2),
        actions=torch.zeros((20, 2), dtype=torch.long),
        outcomes=torch.zeros((20, 2, 1)),
        patient_ids=patient_ids,
    )

    first = split_clinical_extension_roles(batch, seed=92_000)
    repeated = split_clinical_extension_roles(batch, seed=92_000)
    explicit_default = split_clinical_extension_roles(
        batch,
        seed=92_000,
        fractions=(0.40, 0.20, 0.40),
    )
    role_ids = [
        set(role.patient_ids.tolist())
        for role in (first.predictor, first.fidelity, first.environment)
    ]

    assert [len(ids) for ids in role_ids] == [4, 2, 4]
    assert [
        role.n for role in (first.predictor, first.fidelity, first.environment)
    ] == [8, 4, 8]
    assert first.predictor.patient_ids.tolist() == [1, 1, 3, 3, 5, 5, 9, 9]
    assert first.fidelity.patient_ids.tolist() == [2, 2, 6, 6]
    assert first.environment.patient_ids.tolist() == [0, 0, 4, 4, 7, 7, 8, 8]
    assert role_ids[0].isdisjoint(role_ids[1])
    assert role_ids[0].isdisjoint(role_ids[2])
    assert role_ids[1].isdisjoint(role_ids[2])
    assert set.union(*role_ids) == set(range(10))
    assert torch.equal(first.predictor.patient_ids, repeated.predictor.patient_ids)
    assert torch.equal(first.fidelity.patient_ids, repeated.fidelity.patient_ids)
    assert torch.equal(first.environment.patient_ids, repeated.environment.patient_ids)
    for role_name in ("predictor", "fidelity", "environment"):
        default_role = getattr(first, role_name)
        explicit_role = getattr(explicit_default, role_name)
        assert torch.equal(default_role.states, explicit_role.states)
        assert torch.equal(default_role.actions, explicit_role.actions)
        assert torch.equal(default_role.outcomes, explicit_role.outcomes)
        assert torch.equal(default_role.patient_ids, explicit_role.patient_ids)


def test_cxr_role_split_is_exactly_20_20_60_and_patient_disjoint() -> None:
    patient_count = 1_993
    batch = TrajectoryBatch(
        states=torch.zeros((patient_count, 2, 1)),
        actions=torch.zeros((patient_count, 1), dtype=torch.long),
        outcomes=torch.zeros((patient_count, 1, 1)),
        patient_ids=torch.arange(patient_count),
    )

    splits = split_clinical_extension_roles(
        batch,
        seed=122_000,
        fractions=(0.20, 0.20, 0.60),
    )
    role_ids = [
        set(role.patient_ids.tolist())
        for role in (splits.predictor, splits.fidelity, splits.environment)
    ]

    assert splits.split_fractions == (0.20, 0.20, 0.60)
    assert [len(ids) for ids in role_ids] == [398, 398, 1_197]
    assert role_ids[0].isdisjoint(role_ids[1])
    assert role_ids[0].isdisjoint(role_ids[2])
    assert role_ids[1].isdisjoint(role_ids[2])
    assert set.union(*role_ids) == set(range(patient_count))


@pytest.mark.parametrize(
    ("fractions", "message"),
    (
        ((0.50, 0.50), "exactly three"),
        ((0.40, 0.00, 0.60), "finite and positive"),
        ((0.40, -0.10, 0.70), "finite and positive"),
        ((0.40, 0.20, float("nan")), "finite and positive"),
        ((0.40, 0.20, 0.50), "sum to one"),
    ),
)
def test_clinical_split_rejects_invalid_fractions(
    fractions: tuple[float, ...],
    message: str,
) -> None:
    patient_count = 10
    batch = TrajectoryBatch(
        states=torch.zeros((patient_count, 2, 1)),
        actions=torch.zeros((patient_count, 1), dtype=torch.long),
        outcomes=torch.zeros((patient_count, 1, 1)),
        patient_ids=torch.arange(patient_count),
    )

    with pytest.raises(ValueError, match=message):
        split_clinical_extension_roles(
            batch,
            seed=1,
            fractions=fractions,  # type: ignore[arg-type]
        )


def test_clinical_split_rejects_fractions_that_leave_an_empty_role() -> None:
    batch = TrajectoryBatch(
        states=torch.zeros((3, 2, 1)),
        actions=torch.zeros((3, 1), dtype=torch.long),
        outcomes=torch.zeros((3, 1, 1)),
        patient_ids=torch.arange(3),
    )

    with pytest.raises(ValueError, match="empty role"):
        split_clinical_extension_roles(
            batch,
            seed=1,
            fractions=(0.80, 0.10, 0.10),
        )


def test_clinical_split_rejects_too_few_patients_before_allocating_roles() -> None:
    batch = TrajectoryBatch(
        states=torch.zeros((2, 2, 1)),
        actions=torch.zeros((2, 1), dtype=torch.long),
        outcomes=torch.zeros((2, 1, 1)),
        patient_ids=torch.tensor([10, 11]),
    )

    with pytest.raises(ValueError, match="at least three patients"):
        split_clinical_extension_roles(batch, seed=1)


def test_support_counts_unique_patients_not_rows() -> None:
    batch = TrajectoryBatch(
        states=torch.zeros((5, 3, 1)),
        actions=torch.tensor(
            [
                [0, 1],
                [0, 1],
                [0, 0],
                [1, 1],
                [1, 0],
            ]
        ),
        outcomes=torch.zeros((5, 2, 1)),
        patient_ids=torch.tensor([10, 10, 11, 12, 13]),
    )

    assert unique_patient_action_counts(batch, n_actions=3) == [
        [2, 2, 0],
        [2, 2, 0],
    ]
    with pytest.raises(ValueError, match="n_actions must be positive"):
        unique_patient_action_counts(batch, n_actions=0)


def test_equal_sample_wasserstein_is_coordinatewise_and_order_invariant() -> None:
    first = torch.tensor([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    second = torch.tensor([[3.0, 31.0], [1.0, 21.0], [5.0, 11.0]])

    distance = equal_sample_wasserstein_1(first, second)

    assert distance.dtype == torch.float64
    assert distance.tolist() == pytest.approx([1.0, 1.0])
    assert equal_sample_wasserstein_1(second.flip(0), first).tolist() == pytest.approx(
        [1.0, 1.0]
    )
    with pytest.raises(ValueError, match="nonempty equal-shape"):
        equal_sample_wasserstein_1(first, second[:2])


def test_empirical_ks_handles_ties_and_rejects_nonfinite_inputs() -> None:
    first = torch.tensor([0.0, 0.0, 1.0, 1.0])
    second = torch.tensor([1.0, 1.0, 2.0, 2.0])

    assert empirical_ks(first, first.flip(0)) == pytest.approx(0.0)
    assert empirical_ks(first, second) == pytest.approx(0.5)
    assert empirical_ks(torch.zeros(4), torch.ones(4)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="finite and nonempty"):
        empirical_ks(first, torch.tensor([float("nan")]))


def test_setting_availability_requires_nineteen_of_twenty_prespecified_seeds() -> None:
    config = load_extension_config(CONFIG_PATH)
    threshold = config.support_gate.minimum_available_seed_fraction

    assert setting_availability_passes(20, 20, threshold)
    assert setting_availability_passes(19, 20, threshold)
    assert not setting_availability_passes(18, 20, threshold)
    with pytest.raises(ValueError, match="at least one prespecified seed"):
        setting_availability_passes(0, 0, threshold)


def test_donor_overlap_gate_is_inclusive_at_all_three_frozen_boundaries() -> None:
    gate = load_extension_config(CONFIG_PATH).donor_overlap_gate
    boundary = DonorOverlapMetrics(
        local_ess_p01=10.0,
        median_ess_fraction=0.25,
        maximum_donor_probability=0.25,
    )

    assert donor_overlap_passes(boundary, gate)
    assert not donor_overlap_passes(replace(boundary, local_ess_p01=9.999), gate)
    assert not donor_overlap_passes(replace(boundary, median_ess_fraction=0.249), gate)
    assert not donor_overlap_passes(
        replace(boundary, maximum_donor_probability=0.251),
        gate,
    )


def test_k0_fidelity_gate_is_inclusive_and_requires_structural_invariants() -> None:
    gate = load_extension_config(CONFIG_PATH).k0_fidelity_gate
    boundary = K0FidelityMetrics(
        maximum_score_ks=0.10,
        maximum_signed_residual_w1=0.25,
        maximum_successor_mean_w1=0.25,
        maximum_successor_q95_w1=0.50,
        structural_invariants=True,
    )

    assert k0_fidelity_passes(boundary, gate)
    assert not k0_fidelity_passes(
        replace(boundary, maximum_score_ks=0.100_001),
        gate,
    )
    assert not k0_fidelity_passes(
        replace(boundary, maximum_signed_residual_w1=0.250_001),
        gate,
    )
    assert not k0_fidelity_passes(
        replace(boundary, maximum_successor_mean_w1=0.250_001),
        gate,
    )
    assert not k0_fidelity_passes(
        replace(boundary, maximum_successor_q95_w1=0.500_001),
        gate,
    )
    assert not k0_fidelity_passes(
        replace(boundary, structural_invariants=False),
        gate,
    )
