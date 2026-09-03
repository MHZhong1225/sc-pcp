"""Frozen configuration and gates for the clinical controlled extension."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
import yaml

from scpcp.data import TrajectoryBatch


PROTOCOL = "controlled_clinical_extension_v2"
DATASET_NAMES = ("mimic_iv", "eicu", "inspire", "mimic_cxr")
GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
METHODS = ("Standard CP", "ACI", "MFCS", "SPCI", "PRC", "SC-PCP")


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    base_config: Path
    horizon: int
    late_stage_start: int
    seeds: tuple[int, ...]
    bootstrap_seed: int

    @property
    def late_stages(self) -> tuple[int, ...]:
        return tuple(range(self.late_stage_start, self.horizon))


@dataclass(frozen=True)
class SupportGate:
    minimum_unique_patients: int
    minimum_available_seed_fraction: float
    neighbors: int


@dataclass(frozen=True)
class DonorOverlapGate:
    gamma: float
    probe_trajectories: int
    probe_radius_fractions: tuple[float, ...]
    local_ess_quantile: float
    minimum_local_ess_quantile: float
    minimum_median_ess_fraction: float
    maximum_donor_probability: float
    amendment: str


@dataclass(frozen=True)
class K0FidelityGate:
    systematic_replays: int
    maximum_score_ks: float
    maximum_signed_residual_w1: float
    maximum_successor_mean_w1: float
    maximum_successor_q95_w1: float
    minimum_available_seed_fraction: float
    active_coordinate_sd_floor: float


@dataclass(frozen=True)
class ControlledClinicalExtensionConfig:
    protocol: str
    split_fractions: tuple[float, float, float]
    gammas: tuple[float, ...]
    bootstrap_resamples: int
    calibration_trajectories: int
    grid_trajectories: int
    reference_trajectories: int
    online_trajectories: int
    q_low_source_quantile: float
    q_high_source_quantile: float
    alternative_policy_tilt: float
    maximum_policy_response: float
    policy_ratio_cap: float
    transition_ridge: float
    support_gate: SupportGate
    donor_overlap_gate: DonorOverlapGate
    k0_fidelity_gate: K0FidelityGate
    datasets: Mapping[str, DatasetPreset]

    def validate(self) -> None:
        if self.protocol != PROTOCOL:
            raise ValueError(f"protocol must be {PROTOCOL}")
        if tuple(self.datasets) != DATASET_NAMES:
            raise ValueError(f"datasets must appear in frozen order {DATASET_NAMES}")
        if self.split_fractions != (0.40, 0.20, 0.40):
            raise ValueError("clinical split must be D_pred/D_fidelity/D_env=40/20/40")
        if self.gammas != GAMMAS:
            raise ValueError(f"gamma grid must be {GAMMAS}")
        if self.bootstrap_resamples != 10_000:
            raise ValueError("bootstrap resamples must be exactly 10,000")
        if (
            self.calibration_trajectories,
            self.grid_trajectories,
            self.reference_trajectories,
            self.online_trajectories,
        ) != (3_000, 1_000, 20_000, 2_000):
            raise ValueError("controlled trajectory budgets differ from the frozen v2 contract")
        if (
            self.q_low_source_quantile,
            self.q_high_source_quantile,
            self.alternative_policy_tilt,
            self.maximum_policy_response,
            self.policy_ratio_cap,
        ) != (0.80, 0.95, 20.0, 1.0, 3.0):
            raise ValueError("controlled policy constants differ from the frozen v2 contract")
        if self.transition_ridge != 1e-3:
            raise ValueError("controlled transition ridge differs from the frozen v2 contract")
        if self.support_gate != SupportGate(20, 0.95, 100):
            raise ValueError("support gate differs from the frozen v2 contract")
        if self.donor_overlap_gate != DonorOverlapGate(
            -4.0,
            3_000,
            (0.50, 1.00),
            0.01,
            10.0,
            0.25,
            0.25,
            "q_high_max_response_probe_added_before_any_coverage_launch",
        ):
            raise ValueError("donor-overlap gate differs from the frozen v2 contract")
        if self.k0_fidelity_gate != K0FidelityGate(
            16,
            0.10,
            0.25,
            0.25,
            0.50,
            0.95,
            1e-4,
        ):
            raise ValueError("K0 fidelity gate differs from the frozen v2 contract")
        for preset in self.datasets.values():
            if len(preset.seeds) != 20 or len(set(preset.seeds)) != 20:
                raise ValueError(f"{preset.name} must have exactly 20 fresh seeds")
            if not 0 <= preset.late_stage_start < preset.horizon:
                raise ValueError(f"{preset.name} late-stage range is invalid")
        expected_presets = {
            "mimic_iv": DatasetPreset(
                "mimic_iv",
                Path("configs/per_step_mimic_iv.yaml"),
                12,
                4,
                tuple(range(93_600, 93_800, 10)),
                9_361_019,
            ),
            "eicu": DatasetPreset(
                "eicu",
                Path("configs/per_step_eicu.yaml"),
                12,
                4,
                tuple(range(92_000, 92_200, 10)),
                9_201_019,
            ),
            "inspire": DatasetPreset(
                "inspire",
                Path("configs/per_step_inspire.yaml"),
                12,
                4,
                tuple(range(92_300, 92_500, 10)),
                9_231_019,
            ),
            "mimic_cxr": DatasetPreset(
                "mimic_cxr",
                Path("configs/per_step_mimic_cxr.yaml"),
                6,
                2,
                tuple(range(92_600, 92_800, 10)),
                9_261_019,
            ),
        }
        if dict(self.datasets) != expected_presets:
            raise ValueError("dataset presets differ from the frozen v2 contract")


@dataclass(frozen=True)
class SupportGateResult:
    passed: bool
    minimum_unique_patients: int
    failed_cells: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class DonorOverlapMetrics:
    local_ess_p01: float
    median_ess_fraction: float
    maximum_donor_probability: float


@dataclass(frozen=True)
class K0FidelityMetrics:
    maximum_score_ks: float
    maximum_signed_residual_w1: float
    maximum_successor_mean_w1: float
    maximum_successor_q95_w1: float
    structural_invariants: bool


@dataclass(frozen=True)
class ClinicalExtensionSplits:
    """Patient-disjoint roles for the controlled clinical study."""

    predictor: TrajectoryBatch
    fidelity: TrajectoryBatch
    environment: TrajectoryBatch
    split_fractions: tuple[float, float, float] = (0.40, 0.20, 0.40)


def split_clinical_extension_roles(
    batch: TrajectoryBatch,
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.40, 0.20, 0.40),
) -> ClinicalExtensionSplits:
    """Create a deterministic D_pred/D_fidelity/D_env patient allocation."""

    if len(fractions) != 3:
        raise ValueError("clinical split fractions must contain exactly three values")
    split_fractions = tuple(float(value) for value in fractions)
    if not all(math.isfinite(value) and value > 0.0 for value in split_fractions):
        raise ValueError("clinical split fractions must be finite and positive")
    if not math.isclose(sum(split_fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("clinical split fractions must sum to one")

    unique_ids = torch.unique(batch.patient_ids.cpu(), sorted=True)
    if len(unique_ids) < 3:
        raise ValueError("the clinical extension needs at least three patients")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shuffled = unique_ids[torch.randperm(len(unique_ids), generator=generator)]
    predictor_count = max(1, int(split_fractions[0] * len(shuffled)))
    fidelity_count = max(1, int(split_fractions[1] * len(shuffled)))
    if predictor_count + fidelity_count >= len(shuffled):
        raise ValueError("clinical split fractions produced an empty role")
    predictor_ids = shuffled[:predictor_count]
    fidelity_ids = shuffled[predictor_count : predictor_count + fidelity_count]
    environment_ids = shuffled[predictor_count + fidelity_count :]

    def subset(patient_ids: Tensor) -> TrajectoryBatch:
        rows = torch.isin(batch.patient_ids.cpu(), patient_ids).nonzero().squeeze(1)
        return batch.subset(rows.to(batch.patient_ids.device))

    return ClinicalExtensionSplits(
        predictor=subset(predictor_ids),
        fidelity=subset(fidelity_ids),
        environment=subset(environment_ids),
        split_fractions=split_fractions,
    )


def unique_patient_action_counts(batch: TrajectoryBatch, n_actions: int) -> list[list[int]]:
    """Count unique D_env patients in every prespecified stage/action cell."""

    if n_actions < 1:
        raise ValueError("n_actions must be positive")
    counts: list[list[int]] = []
    for stage in range(batch.horizon):
        row = []
        for action in range(n_actions):
            selected = batch.actions[:, stage].eq(action)
            row.append(int(torch.unique(batch.patient_ids[selected]).numel()))
        counts.append(row)
    return counts


def equal_sample_wasserstein_1(first: Tensor, second: Tensor) -> Tensor:
    """Coordinatewise W1 for equal-size empirical samples."""

    if first.shape != second.shape or first.ndim != 2 or len(first) == 0:
        raise ValueError("W1 inputs must be nonempty equal-shape [N,D] tensors")
    first64 = first.detach().to(device="cpu", dtype=torch.float64)
    second64 = second.detach().to(device="cpu", dtype=torch.float64)
    return (first64.sort(dim=0).values - second64.sort(dim=0).values).abs().mean(dim=0)


def empirical_ks(first: Tensor, second: Tensor) -> float:
    """Two-sample empirical KS distance without asymptotic p-values."""

    x = first.detach().flatten().to(device="cpu", dtype=torch.float64)
    y = second.detach().flatten().to(device="cpu", dtype=torch.float64)
    if len(x) == 0 or len(y) == 0 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("KS inputs must be finite and nonempty")
    support = torch.cat((x, y)).sort().values
    x_cdf = torch.searchsorted(x.sort().values, support, right=True) / len(x)
    y_cdf = torch.searchsorted(y.sort().values, support, right=True) / len(y)
    return float((x_cdf - y_cdf).abs().max().item())


def load_extension_config(path: Path) -> ControlledClinicalExtensionConfig:
    payload = yaml.safe_load(path.read_text())
    science = payload["science"]
    support = payload["support_gate"]
    overlap = payload["donor_overlap_gate"]
    fidelity = payload["k0_fidelity_gate"]
    fractions = payload["split_fractions"]
    datasets = {
        name: _dataset_preset(name, values)
        for name, values in payload["datasets"].items()
    }
    config = ControlledClinicalExtensionConfig(
        protocol=str(payload["protocol"]),
        split_fractions=(
            float(fractions["predictor"]),
            float(fractions["fidelity"]),
            float(fractions["environment"]),
        ),
        gammas=tuple(float(value) for value in science["gammas"]),
        bootstrap_resamples=int(science["bootstrap_resamples"]),
        calibration_trajectories=int(science["calibration_trajectories"]),
        grid_trajectories=int(science["grid_trajectories"]),
        reference_trajectories=int(science["reference_trajectories"]),
        online_trajectories=int(science["online_trajectories"]),
        q_low_source_quantile=float(science["q_low_source_quantile"]),
        q_high_source_quantile=float(science["q_high_source_quantile"]),
        alternative_policy_tilt=float(science["alternative_policy_tilt"]),
        maximum_policy_response=float(science["maximum_policy_response"]),
        policy_ratio_cap=float(science["policy_ratio_cap"]),
        transition_ridge=float(science["transition_ridge"]),
        support_gate=SupportGate(
            minimum_unique_patients=int(support["minimum_unique_patients_per_stage_action"]),
            minimum_available_seed_fraction=float(support["minimum_available_seed_fraction"]),
            neighbors=int(support["neighbors"]),
        ),
        donor_overlap_gate=DonorOverlapGate(
            gamma=float(overlap["gamma"]),
            probe_trajectories=int(overlap["probe_trajectories"]),
            probe_radius_fractions=tuple(
                float(value) for value in overlap["probe_radius_fractions"]
            ),
            local_ess_quantile=float(overlap["local_ess_quantile"]),
            minimum_local_ess_quantile=float(overlap["minimum_local_ess_quantile"]),
            minimum_median_ess_fraction=float(overlap["minimum_median_ess_fraction"]),
            maximum_donor_probability=float(overlap["maximum_donor_probability"]),
            amendment=str(overlap["amendment"]),
        ),
        k0_fidelity_gate=K0FidelityGate(
            systematic_replays=int(fidelity["systematic_replays"]),
            maximum_score_ks=float(fidelity["maximum_score_ks"]),
            maximum_signed_residual_w1=float(fidelity["maximum_signed_residual_w1"]),
            maximum_successor_mean_w1=float(fidelity["maximum_successor_mean_w1"]),
            maximum_successor_q95_w1=float(fidelity["maximum_successor_q95_w1"]),
            minimum_available_seed_fraction=float(fidelity["minimum_available_seed_fraction"]),
            active_coordinate_sd_floor=float(fidelity["active_coordinate_sd_floor"]),
        ),
        datasets=datasets,
    )
    config.validate()
    return config


def evaluate_support_gate(
    unique_patient_counts: list[list[int]],
    gate: SupportGate,
) -> SupportGateResult:
    if not unique_patient_counts or any(not row for row in unique_patient_counts):
        raise ValueError("support counts must be a nonempty rectangular table")
    width = len(unique_patient_counts[0])
    if any(len(row) != width for row in unique_patient_counts):
        raise ValueError("support counts must be rectangular")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for row in unique_patient_counts
        for count in row
    ):
        raise ValueError("support counts must be nonnegative integers")
    failed = tuple(
        (stage, action, int(count))
        for stage, row in enumerate(unique_patient_counts)
        for action, count in enumerate(row)
        if count < gate.minimum_unique_patients
    )
    minimum = min(count for row in unique_patient_counts for count in row)
    return SupportGateResult(not failed, int(minimum), failed)


def donor_overlap_passes(metrics: DonorOverlapMetrics, gate: DonorOverlapGate) -> bool:
    values = (
        metrics.local_ess_p01,
        metrics.median_ess_fraction,
        metrics.maximum_donor_probability,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("donor-overlap metrics must be finite")
    if metrics.local_ess_p01 < 1.0:
        raise ValueError("patient-aggregated donor ESS must be at least one")
    if not 0.0 <= metrics.median_ess_fraction <= 1.0:
        raise ValueError("median ESS/k must lie in [0,1]")
    if not 0.0 <= metrics.maximum_donor_probability <= 1.0:
        raise ValueError("maximum donor mass must lie in [0,1]")
    return (
        metrics.local_ess_p01 >= gate.minimum_local_ess_quantile
        and metrics.median_ess_fraction >= gate.minimum_median_ess_fraction
        and metrics.maximum_donor_probability <= gate.maximum_donor_probability
    )


def k0_fidelity_passes(metrics: K0FidelityMetrics, gate: K0FidelityGate) -> bool:
    values = (
        metrics.maximum_score_ks,
        metrics.maximum_signed_residual_w1,
        metrics.maximum_successor_mean_w1,
        metrics.maximum_successor_q95_w1,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("K0 fidelity metrics must be finite")
    if not 0.0 <= metrics.maximum_score_ks <= 1.0:
        raise ValueError("KS distance must lie in [0,1]")
    if any(value < 0.0 for value in values[1:]):
        raise ValueError("Wasserstein distances must be nonnegative")
    return (
        metrics.structural_invariants
        and metrics.maximum_score_ks <= gate.maximum_score_ks
        and metrics.maximum_signed_residual_w1 <= gate.maximum_signed_residual_w1
        and metrics.maximum_successor_mean_w1 <= gate.maximum_successor_mean_w1
        and metrics.maximum_successor_q95_w1 <= gate.maximum_successor_q95_w1
    )


def setting_availability_passes(
    available: int,
    total: int,
    minimum_fraction: float,
) -> bool:
    if total < 1:
        raise ValueError("a setting must contain at least one prespecified seed")
    if not 0 <= available <= total:
        raise ValueError("available must lie between zero and total")
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must lie in [0,1]")
    return available / total >= minimum_fraction


def _dataset_preset(name: str, payload: Mapping[str, Any]) -> DatasetPreset:
    seeds = tuple(
        range(
            int(payload["seed_start"]),
            int(payload["seed_stop"]),
            int(payload["seed_step"]),
        )
    )
    return DatasetPreset(
        name=name,
        base_config=Path(str(payload["base_config"])),
        horizon=int(payload["horizon"]),
        late_stage_start=int(payload["late_stage_start"]),
        seeds=seeds,
        bootstrap_seed=int(payload["bootstrap_seed"]),
    )
