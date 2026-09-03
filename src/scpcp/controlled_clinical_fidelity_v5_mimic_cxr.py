"""Coverage-blind MIMIC-CXR outcome-bridge repair contract.

V5 freezes the complete v4 C13 state/donor kernel and varies only the
conditional mean used for the two MIMIC-CXR outcomes.  It deliberately has no
SC-PCP, coverage, width, or paper-result code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import yaml

from scpcp.controlled_transition import ControlledResidualEnvironment
from scpcp.data import TrajectoryBatch


PROTOCOL = "controlled_clinical_fidelity_v5_mimic_cxr"
DATASET = "mimic_cxr"
SELECTOR_VERSION = "controlled_clinical_fidelity_v5_mimic_cxr_selector_v1"

DEVELOPMENT_SEEDS = tuple(range(92_600, 92_800, 10))
CONFIRMATION_SEEDS = tuple(range(119_000, 119_200, 10))
CONFIRMATION_BOOTSTRAP_SEED = 11_900_019

DEVELOPMENT_MAPPING_SHA256 = (
    "88ff47c1cf4516777b63cfc892155b426e0a34b3268414c27e2911b3ecdda9c6"
)
DEVELOPMENT_ID_SET_SHA256 = (
    "bfafe31040f154d3f4a157f74cb441d0c535b494463779a6cc359b9c1f2e18cc"
)
CONFIRMATION_MAPPING_SHA256 = (
    "43d832a650352de3e97fc6694178c61969e707a3658abb33958668080cf3e40e"
)
CONFIRMATION_ID_SET_SHA256 = (
    "c3caf2b4e5137b62c52fe71e97a559ff2fff513cdcd597865819f21852a80742"
)
CONFIRMATION_BASE_SET_SHA256 = (
    "8724d28e631cda56727563b6904abd3407a32c6484672a72b634e83b08d719bd"
)

K0_THRESHOLDS = {
    "maximum_score_ks": 0.10,
    "maximum_signed_residual_w1": 0.25,
    "maximum_successor_mean_w1": 0.25,
    "maximum_successor_q95_w1": 0.50,
}
DEVELOPMENT_MINIMUM_PASS_COUNT = 19
REQUIRED_STRUCTURAL_PASS_COUNT = 20

C13_STATE_KERNEL = {
    "metric": "raw",
    "neighbors": 10_000,
    "uses_full_cell": True,
    "donor_weighting": "uniform",
    "bandwidth": 2.0,
    "transition_mode": "ridge_residual",
    "outcome_residual_mode": "raw",
    "ridge_mode": "sample_normalized_no_intercept",
    "ridge_value": 1e-3,
    "penalize_intercept": False,
}

SPO2_FEATURE_NAMES = (
    "spo2_last",
    "spo2_mean",
    "spo2_min",
    "spo2_max",
)
RR_FEATURE_NAMES = ("rr_last", "rr_mean", "rr_min", "rr_max")
SUCCESSOR_SOURCE_FEATURE_NAMES = SPO2_FEATURE_NAMES + RR_FEATURE_NAMES
# The formal n=60,000 cache has one more fitted static-category column than the
# small n=3,000 development cache.  Bind the paper run's exact v17 schema.
SUCCESSOR_SOURCE_INDICES = (48, 49, 50, 51, 44, 45, 46, 47)
SPO2_HINGE_THRESHOLD = 92.0
SPO2_HINGE_SCALE = 10.0
RR_HINGE_THRESHOLD = 22.0
RR_HINGE_SCALE = 15.0
BRIDGE_RIDGE = 1e-3
CANONICAL_ACTION_COUNT = 3

_BRIDGE_MODES = (
    "exact_c13_anchor",
    "stagewise_successor_bridge",
    "pooled_successor_bridge_stage_one_hot",
)

_PARENT_DEVELOPMENT_FILES = {
    "COMPLETE": "6d05b9e8e1411c7d75f2247a5d8c8fc2479557fb3d365165682e4e706efff610",
    "FINAL_STATUS.json": "a098c77436b8ad8415ad081e6a5af9b5a4dfb329a9eecd4956cd9718c347b368",
    "frozen_settings.json": "0d2f0e676cd19c88772b6972af480b3c431938eed1d10a98db88645f90237ee8",
    "manifest.json": "f7d207418590cbb947705caec4c777a0895d710b0665e7323c9a72ff623cc0be",
    "metadata.json": "a7b452b80e13670c5d845fd7ccad0a97a0b39e805869e1c0c884960bd3dfeebc",
    "selection/mimic_cxr.json": "1b7d30389a84d38255ea15395430eb99d74ddb4259c20a9451e9aebca99fec63",
}
_PARENT_CONFIRMATION_RETRY_FILES = {
    "COMPLETE": "e156b19e9cc086a0506aa8cef34f9807ddad66ef670a2b1571705b97924b3fcf",
    "FINAL_STATUS.json": "25df7a510a929d65847f3d65294bfb7b436cf6bc96b0433bd4b82800425a51ca",
    "administrative_retry_amendment.json": "528fb9f19ca158c4ff255e50cde8c577256e35da358ad282f1cb6cc8b83eb363",
    "manifest.json": "fe48c9d7f9d356db9765245b62472cc64ce34d7bd0d2b8fb5d900acd9433c69a",
    "metadata.json": "1648e829d8c1cbb8ac4bc174c62b686a1f9403cead6b337c7bf4e09f80b351ca",
    "support_replay_verification.json": "e1f6d78ec707d7391445cc4d14f56b9b94a11775b3884931e1c9537ef48a7412",
    "mimic_cxr/COMPLETE": "9f0f4cf6ea28485b6f322b5c52755352d7a1f45d5201be22e966e247928fdd91",
    "mimic_cxr/gate.json": "a5824cc58334539b598114ae87e1e640c97dea4bb7b105a096c1aab9f5a7b3d6",
}


@dataclass(frozen=True)
class BridgeTheta:
    """One prespecified outcome-mean candidate over the frozen C13 kernel."""

    candidate_id: str
    bridge_mode: str

    def __post_init__(self) -> None:
        try:
            index = _BRIDGE_MODES.index(self.bridge_mode)
        except ValueError as error:
            raise ValueError("unknown MIMIC-CXR bridge mode") from error
        expected = f"B{index:02d}_{self.bridge_mode}"
        if self.candidate_id != expected:
            raise ValueError("bridge candidate ID differs from its frozen mode")

    @property
    def minimal_change_rank(self) -> int:
        return _BRIDGE_MODES.index(self.bridge_mode)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "dataset": DATASET,
            "bridge_mode": self.bridge_mode,
            "minimal_change_rank": self.minimal_change_rank,
            "state_kernel": dict(C13_STATE_KERNEL),
            "bridge": bridge_contract(self.bridge_mode),
        }


def bridge_candidates() -> tuple[BridgeTheta, ...]:
    return tuple(
        BridgeTheta(f"B{index:02d}_{mode}", mode)
        for index, mode in enumerate(_BRIDGE_MODES)
    )


def bridge_contract(mode: str) -> dict[str, Any]:
    if mode not in _BRIDGE_MODES:
        raise ValueError("unknown MIMIC-CXR bridge mode")
    return {
        "source": "D_env_only",
        "fit_scope": {
            "exact_c13_anchor": "not_applicable",
            "stagewise_successor_bridge": "one_model_per_stage",
            "pooled_successor_bridge_stage_one_hot": "pooled_over_stages",
        }[mode],
        "successor_source_feature_names": list(SUCCESSOR_SOURCE_FEATURE_NAMES),
        "successor_source_feature_indices": list(SUCCESSOR_SOURCE_INDICES),
        "clinical_coordinate_order": [
            *SUCCESSOR_SOURCE_FEATURE_NAMES,
            *(f"hinge_hypox_{name}" for name in SPO2_FEATURE_NAMES),
            *(f"hinge_tachyp_{name}" for name in RR_FEATURE_NAMES),
        ],
        "spo2_hinge": {
            "formula": "max(92-spo2,0)/10",
            "threshold": SPO2_HINGE_THRESHOLD,
            "scale": SPO2_HINGE_SCALE,
        },
        "rr_hinge": {
            "formula": "max(rr-22,0)/15",
            "threshold": RR_HINGE_THRESHOLD,
            "scale": RR_HINGE_SCALE,
        },
        "current_action_one_hot": {
            "included": mode != "exact_c13_anchor",
            "action_count": CANONICAL_ACTION_COUNT,
            "order": [0, 1, 2],
            "penalized": True,
        },
        "stage_one_hot": {
            "included": mode == "pooled_successor_bridge_stage_one_hot",
            "penalized": True,
        },
        "ridge_mode": "sample_normalized_no_intercept",
        "ridge_value": BRIDGE_RIDGE,
        "intercept_included": True,
        "intercept_penalized": False,
        "outcome_residual": "observed_joint_two_vector_minus_bridge_prediction",
        "rollout_mean": "bridge_generated_successor_and_current_action",
        "joint_donor_innovation": True,
    }


@dataclass(frozen=True)
class RootBinding:
    root: Path
    file_sha256: Mapping[str, str]


@dataclass(frozen=True)
class IndependentAudit:
    status: str
    attestation_sha256: str | None
    expected_prior_count: int | None
    expected_prior_sha256: str | None
    expected_artifact_count: int | None
    expected_artifact_sha256: str | None
    expected_source_count: int | None
    expected_source_sha256: str | None

    @property
    def permits_formal_launch(self) -> bool:
        return (
            self.status == "GO"
            and _is_sha256(self.attestation_sha256)
            and self.attestation_sha256
            == independent_audit_attestation_sha256(self)
        )


def independent_audit_attestation_sha256(audit: IndependentAudit) -> str:
    """Bind GO to the exact collision-free prior snapshot and RNG contract."""

    payload = {
        "protocol": PROTOCOL,
        "role": "independent_read_only_prelaunch_audit",
        "status": "GO",
        "formal_roots_absent_at_audit": True,
        "formal_rng_consumed": False,
        "collision_count": 0,
        "confirmation_mapping_sha256": CONFIRMATION_MAPPING_SHA256,
        "confirmation_id_set_sha256": CONFIRMATION_ID_SET_SHA256,
        "confirmation_base_set_sha256": CONFIRMATION_BASE_SET_SHA256,
        "prior_rng_id_count": audit.expected_prior_count,
        "prior_rng_id_sha256": audit.expected_prior_sha256,
        "artifact_rng_id_count": audit.expected_artifact_count,
        "artifact_rng_id_sha256": audit.expected_artifact_sha256,
        "source_declared_rng_id_count": audit.expected_source_count,
        "source_declared_rng_id_sha256": audit.expected_source_sha256,
    }
    return _json_sha256(payload)


@dataclass(frozen=True)
class FidelityV5Config:
    parent_development: RootBinding
    parent_confirmation_retry: RootBinding
    development_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    confirmation_bootstrap_seed: int
    independent_audit: IndependentAudit

    def validate(self, *, require_audit_go: bool = False) -> None:
        if self.parent_development.root != Path(
            "results/work/controlled_clinical_fidelity_v4_development"
        ):
            raise ValueError("v5 development parent differs from frozen v4")
        if self.parent_confirmation_retry.root != Path(
            "results/work/controlled_clinical_fidelity_v4_confirmation_administrative_retry_r1"
        ):
            raise ValueError("v5 confirmation parent differs from frozen v4 retry")
        if dict(self.parent_development.file_sha256) != _PARENT_DEVELOPMENT_FILES:
            raise ValueError("v5 development-parent hashes differ")
        if (
            dict(self.parent_confirmation_retry.file_sha256)
            != _PARENT_CONFIRMATION_RETRY_FILES
        ):
            raise ValueError("v5 confirmation-parent hashes differ")
        if self.development_seeds != DEVELOPMENT_SEEDS:
            raise ValueError("v5 must reuse the exact v4 CXR development lineage")
        if self.confirmation_seeds != CONFIRMATION_SEEDS:
            raise ValueError("v5 fresh confirmation bank differs")
        if self.confirmation_bootstrap_seed != CONFIRMATION_BOOTSTRAP_SEED:
            raise ValueError("v5 confirmation bootstrap seed differs")
        if require_audit_go and not self.independent_audit.permits_formal_launch:
            raise RuntimeError(
                "formal v5 RNG is locked until an independent audit records GO"
            )


@dataclass(frozen=True)
class K0CandidateSummary:
    candidate_id: str
    pass_count: int
    structural_pass_count: int
    q95_seed_ratio: float
    mean_seed_ratio: float
    seed_ratios: tuple[float, ...]
    structural_pass_flags: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.candidate_id not in {
            candidate.candidate_id for candidate in bridge_candidates()
        }:
            raise ValueError("unknown bridge candidate summary")
        if len(self.seed_ratios) != 20 or len(self.structural_pass_flags) != 20:
            raise ValueError("bridge summary requires all 20 development seeds")
        if any(type(value) is not bool for value in self.structural_pass_flags):
            raise ValueError("structural flags must be exact booleans")
        if self.pass_count != sum(ratio <= 1.0 for ratio in self.seed_ratios):
            raise ValueError("bridge numeric pass count differs from seed ratios")
        if self.structural_pass_count != sum(self.structural_pass_flags):
            raise ValueError("bridge structural pass count differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "dataset": DATASET,
            "pass_count": self.pass_count,
            "structural_pass_count": self.structural_pass_count,
            "q95_seed_ratio": self.q95_seed_ratio,
            "mean_seed_ratio": self.mean_seed_ratio,
            "seed_ratios": list(self.seed_ratios),
            "structural_pass_flags": list(self.structural_pass_flags),
            "structural_failure_ratio_is_infinite": any(
                not math.isfinite(value) for value in self.seed_ratios
            ),
        }


def normalized_seed_ratio(metrics: Mapping[str, Any]) -> float:
    if not bool(metrics.get("structural_invariants")):
        return math.inf
    ratios = [
        float(metrics[name]) / threshold
        for name, threshold in K0_THRESHOLDS.items()
    ]
    return max(ratios) if all(math.isfinite(value) for value in ratios) else math.inf


def summarize_candidate(
    theta: BridgeTheta,
    metrics_by_seed: Sequence[Mapping[str, Any]],
) -> K0CandidateSummary:
    if len(metrics_by_seed) != 20:
        raise ValueError("bridge development summary requires exactly 20 seeds")
    ratios = tuple(normalized_seed_ratio(metrics) for metrics in metrics_by_seed)
    structural = tuple(
        bool(metrics.get("structural_invariants")) for metrics in metrics_by_seed
    )
    finite = np.asarray(ratios, dtype=np.float64)
    return K0CandidateSummary(
        candidate_id=theta.candidate_id,
        pass_count=sum(value <= 1.0 for value in ratios),
        structural_pass_count=sum(structural),
        q95_seed_ratio=float(np.quantile(finite, 0.95, method="linear")),
        mean_seed_ratio=float(finite.mean()),
        seed_ratios=ratios,
        structural_pass_flags=structural,
    )


def select_bridge_candidate(
    candidates: Sequence[BridgeTheta],
    summaries: Mapping[str, K0CandidateSummary],
) -> dict[str, Any]:
    if tuple(candidates) != bridge_candidates():
        raise ValueError("bridge candidate order differs from the frozen contract")
    if set(summaries) != {candidate.candidate_id for candidate in candidates}:
        raise ValueError("bridge summaries do not cover the exact candidate set")

    def objective(theta: BridgeTheta) -> tuple[Any, ...]:
        summary = summaries[theta.candidate_id]
        return (
            -summary.pass_count,
            summary.q95_seed_ratio,
            summary.mean_seed_ratio,
            theta.minimal_change_rank,
            theta.candidate_id,
        )

    ordered = sorted(candidates, key=objective)
    winner = ordered[0]
    summary = summaries[winner.candidate_id]
    admissible = (
        summary.pass_count >= DEVELOPMENT_MINIMUM_PASS_COUNT
        and summary.structural_pass_count == REQUIRED_STRUCTURAL_PASS_COUNT
    )
    substantive = objective(winner)[:-1]
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "selector_version": SELECTOR_VERSION,
        "status": (
            "DATASET_DEVELOPMENT_GO" if admissible else "DATASET_DEVELOPMENT_NO_GO"
        ),
        "development_admissible": admissible,
        "development_minimum_pass_count": DEVELOPMENT_MINIMUM_PASS_COUNT,
        "development_required_structural_pass_count": REQUIRED_STRUCTURAL_PASS_COUNT,
        "winner": winner.to_dict(),
        "winner_summary": summary.to_dict(),
        "objective": list(objective(winner)),
        "ordered_candidates": [
            {"candidate_id": theta.candidate_id, "objective": list(objective(theta))}
            for theta in ordered
        ],
        "substantive_ties_before_candidate_id": [
            theta.candidate_id
            for theta in ordered
            if objective(theta)[:-1] == substantive
        ],
        "candidate_seed_deletions": 0,
        "coverage_generated": False,
    }


def successor_clinical_features(
    frame: Tensor,
    state_feature_names: Sequence[str],
) -> Tensor:
    """Return the frozen 16-coordinate successor bridge representation."""

    if frame.ndim != 2:
        raise ValueError("successor frame must be a matrix")
    if tuple(state_feature_names[index] for index in SUCCESSOR_SOURCE_INDICES) != (
        SUCCESSOR_SOURCE_FEATURE_NAMES
    ):
        raise ValueError("MIMIC-CXR successor feature indices/names differ")
    raw = frame[:, SUCCESSOR_SOURCE_INDICES]
    spo2 = raw[:, :4]
    rr = raw[:, 4:]
    spo2_hinge = (SPO2_HINGE_THRESHOLD - spo2).clamp_min(0.0) / SPO2_HINGE_SCALE
    rr_hinge = (rr - RR_HINGE_THRESHOLD).clamp_min(0.0) / RR_HINGE_SCALE
    return torch.cat((raw, spo2_hinge, rr_hinge), dim=1)


def outcome_feature_groups() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Indices linking successor diagnostics to hypoxemia and tachypnea."""

    return ((0, 1, 2, 3, 8, 9, 10, 11), (4, 5, 6, 7, 12, 13, 14, 15))


class CxrSuccessorOutcomeBridgeEnvironment(ControlledResidualEnvironment):
    """Frozen C13 state kernel with a D_env-only successor-outcome bridge."""

    def __init__(
        self,
        batch: TrajectoryBatch,
        *,
        bridge_mode: str,
        outcome_model: object,
        n_actions: int,
        difficulty: Tensor,
        history_length: int,
        static_indices: tuple[int, ...],
        state_feature_names: tuple[str, ...],
    ) -> None:
        if bridge_mode not in _BRIDGE_MODES[1:]:
            raise ValueError("successor bridge environment requires B01 or B02")
        if n_actions != CANONICAL_ACTION_COUNT:
            raise ValueError("MIMIC-CXR bridge requires exactly three canonical actions")
        self.bridge_mode = bridge_mode
        self.bridge_state_feature_names = state_feature_names
        super().__init__(
            batch,
            outcome_model=outcome_model,
            n_actions=n_actions,
            difficulty=difficulty,
            history_length=history_length,
            static_indices=static_indices,
            state_feature_names=state_feature_names,
            neighbors=int(C13_STATE_KERNEL["neighbors"]),
            bandwidth=float(C13_STATE_KERNEL["bandwidth"]),
            ridge=float(C13_STATE_KERNEL["ridge_value"]),
            representation_geometry=str(C13_STATE_KERNEL["metric"]),
            donor_weighting=str(C13_STATE_KERNEL["donor_weighting"]),
            ridge_mode=str(C13_STATE_KERNEL["ridge_mode"]),
            transition_mode=str(C13_STATE_KERNEL["transition_mode"]),
            outcome_residual_mode=str(C13_STATE_KERNEL["outcome_residual_mode"]),
        )
        self._fit_outcome_bridge(batch)

    def _fit_outcome_bridge(self, batch: TrajectoryBatch) -> None:
        next_frames = batch.states[:, 1:].cpu().reshape(
            batch.n,
            batch.horizon,
            self.history_length,
            self.base_state_dim,
        )[:, :, -1]
        actions = batch.actions.cpu()
        outcomes = batch.outcomes.cpu()
        clinical = torch.stack(
            [
                successor_clinical_features(
                    next_frames[:, stage], self.bridge_state_feature_names
                )
                for stage in range(self.horizon)
            ],
            dim=1,
        )
        if self.bridge_mode == "stagewise_successor_bridge":
            coefficients = []
            predictions = []
            for stage in range(self.horizon):
                design = self._bridge_design(
                    clinical[:, stage], actions[:, stage], stage=stage
                )
                coefficient = self._fit_ridge(
                    design,
                    outcomes[:, stage],
                    BRIDGE_RIDGE,
                    mode="sample_normalized_no_intercept",
                )
                coefficients.append(coefficient)
                predictions.append(design @ coefficient)
            self._bridge_coefficients = tuple(coefficients)
            predicted = torch.stack(predictions, dim=1)
        else:
            designs = [
                self._bridge_design(clinical[:, stage], actions[:, stage], stage=stage)
                for stage in range(self.horizon)
            ]
            pooled_design = torch.cat(designs, dim=0)
            pooled_outcome = outcomes.transpose(0, 1).reshape(-1, outcomes.shape[-1])
            coefficient = self._fit_ridge(
                pooled_design,
                pooled_outcome,
                BRIDGE_RIDGE,
                mode="sample_normalized_no_intercept",
            )
            self._bridge_coefficients = (coefficient,)
            predicted = torch.stack(
                [design @ coefficient for design in designs], dim=1
            )

        residual = outcomes - predicted
        for stage in range(self.horizon):
            for action in range(self.n_actions):
                rows = actions[:, stage].eq(action)
                library = self._libraries[(stage, action)]
                self._libraries[(stage, action)] = (
                    library[0],
                    library[1],
                    residual[rows, stage],
                    library[3],
                    library[4],
                )

    def _bridge_design(
        self,
        clinical: Tensor,
        action: Tensor,
        *,
        stage: int,
    ) -> Tensor:
        one_hot = torch.nn.functional.one_hot(
            action.to(torch.long), self.n_actions
        ).to(clinical)
        parts = [
            torch.ones((len(clinical), 1), dtype=clinical.dtype, device=clinical.device),
            clinical,
            one_hot,
        ]
        if self.bridge_mode == "pooled_successor_bridge_stage_one_hot":
            stage_column = torch.full(
                (len(clinical),), stage, dtype=torch.long, device=clinical.device
            )
            parts.append(
                torch.nn.functional.one_hot(stage_column, self.horizon).to(clinical)
            )
        return torch.cat(parts, dim=1)

    def _bridge_mean(self, frame: Tensor, action: Tensor, *, stage: int) -> Tensor:
        clinical = successor_clinical_features(
            frame, self.bridge_state_feature_names
        )
        design = self._bridge_design(clinical, action, stage=stage)
        coefficient = (
            self._bridge_coefficients[stage]
            if self.bridge_mode == "stagewise_successor_bridge"
            else self._bridge_coefficients[0]
        )
        return design @ coefficient.to(design)

    @torch.no_grad()
    def step_from_uniform(
        self,
        state: Tensor,
        action: Tensor,
        donor_uniform: Tensor,
        *,
        time: int,
        gamma: float,
        action_coordinate: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if donor_uniform.shape != (len(state),):
            raise ValueError("donor_uniform must have one draw per state")
        next_state = torch.empty_like(state)
        outcome = state.new_empty((len(state), 2))
        selected_difficulty = state.new_empty(len(state))
        kernel_ess = state.new_empty(len(state))
        probability_max = state.new_empty(len(state))
        representation = self._representation(state)
        current_frame = state.reshape(
            len(state), self.history_length, self.base_state_dim
        )[:, -1]
        for action_value in range(self.n_actions):
            rows = action.eq(action_value).nonzero().squeeze(1)
            if len(rows) == 0:
                continue
            (
                library_rep,
                state_payload,
                outcome_residual,
                difficulty,
                cumulative_increment,
            ) = self._library(time, action_value, state.device, state.dtype)
            query = self._metric_query_representation(
                representation[rows], time=time, action=action_value
            )
            distance = torch.cdist(query, library_rep)
            count = min(self.neighbors, len(library_rep))
            nearest_distance, nearest = distance.topk(count, largest=False, sorted=True)
            logits = self._base_donor_logits(nearest_distance)
            logits = (
                logits
                + gamma
                * action_coordinate.to(logits)[action_value]
                * difficulty[nearest]
            )
            probability = torch.softmax(logits, dim=1)
            draw = torch.searchsorted(
                probability.cumsum(dim=1), donor_uniform[rows, None]
            ).squeeze(1)
            draw = draw.clamp_max(count - 1)
            chosen = nearest[
                torch.arange(len(rows), device=state.device), draw
            ]

            features = self._features(representation[rows], action[rows])
            predicted_frame = features @ self._models[time].coefficients.to(features)
            frame = predicted_frame + state_payload[chosen]
            if self.static_base_indices:
                frame[:, self.static_base_indices] = current_frame[rows][
                    :, self.static_base_indices
                ]
            if self.cumulative_indices:
                frame[:, self.cumulative_indices] = (
                    current_frame[rows][:, self.cumulative_indices]
                    + cumulative_increment[chosen]
                )
            if self.decision_time_index is not None:
                frame[:, self.decision_time_index] = (time + 1) / self.horizon
            sequence = state[rows].reshape(
                len(rows), self.history_length, self.base_state_dim
            )
            next_state[rows] = torch.cat(
                (sequence[:, 1:], frame[:, None]), dim=1
            ).reshape(len(rows), -1)
            outcome[rows] = self._bridge_mean(
                frame, action[rows], stage=time
            ) + outcome_residual[chosen]
            selected_difficulty[rows] = difficulty[chosen]
            kernel_ess[rows] = 1.0 / probability.square().sum(dim=1)
            probability_max[rows] = probability.max(dim=1).values
        return next_state, outcome, selected_difficulty, kernel_ess, probability_max

    def bridge_identity(self) -> dict[str, Any]:
        coefficients = [
            _tensor_sha256(value.to(torch.float64))
            for value in self._bridge_coefficients
        ]
        residual_libraries = [
            {
                "stage": stage,
                "action": action,
                "rows": len(self._libraries[(stage, action)][2]),
                "sha256": _tensor_sha256(
                    self._libraries[(stage, action)][2].to(torch.float64)
                ),
            }
            for stage in range(self.horizon)
            for action in range(self.n_actions)
        ]
        identity = {
            **bridge_contract(self.bridge_mode),
            "coefficient_sha256": coefficients,
            "coefficient_count": len(coefficients),
            "joint_residual_libraries": residual_libraries,
        }
        return {**identity, "combined_sha256": _json_sha256(identity)}


def build_cxr_environment(
    batch: TrajectoryBatch,
    *,
    theta: BridgeTheta,
    outcome_model: object,
    n_actions: int,
    difficulty: Tensor,
    history_length: int,
    static_indices: tuple[int, ...],
    state_feature_names: tuple[str, ...],
) -> ControlledResidualEnvironment:
    """Build B00 bitwise as C13, or one of the two isolated bridges."""

    if theta.bridge_mode == "exact_c13_anchor":
        return ControlledResidualEnvironment(
            batch,
            outcome_model=outcome_model,
            n_actions=n_actions,
            difficulty=difficulty,
            history_length=history_length,
            static_indices=static_indices,
            state_feature_names=state_feature_names,
            neighbors=int(C13_STATE_KERNEL["neighbors"]),
            bandwidth=float(C13_STATE_KERNEL["bandwidth"]),
            ridge=float(C13_STATE_KERNEL["ridge_value"]),
            representation_geometry=str(C13_STATE_KERNEL["metric"]),
            donor_weighting=str(C13_STATE_KERNEL["donor_weighting"]),
            ridge_mode=str(C13_STATE_KERNEL["ridge_mode"]),
            transition_mode=str(C13_STATE_KERNEL["transition_mode"]),
            outcome_residual_mode=str(C13_STATE_KERNEL["outcome_residual_mode"]),
        )
    return CxrSuccessorOutcomeBridgeEnvironment(
        batch,
        bridge_mode=theta.bridge_mode,
        outcome_model=outcome_model,
        n_actions=n_actions,
        difficulty=difficulty,
        history_length=history_length,
        static_indices=static_indices,
        state_feature_names=state_feature_names,
    )


def load_fidelity_v5_config(path: Path) -> FidelityV5Config:
    payload = yaml.safe_load(path.read_text())
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("unknown clinical-v5 protocol")

    def binding(section: Mapping[str, Any]) -> RootBinding:
        return RootBinding(
            root=Path(section["root"]),
            file_sha256=dict(section["file_sha256"]),
        )

    audit = payload["independent_audit"]
    config = FidelityV5Config(
        parent_development=binding(payload["parent_v4"]["development"]),
        parent_confirmation_retry=binding(
            payload["parent_v4"]["confirmation_retry"]
        ),
        development_seeds=_seed_tuple(payload["development"]["seeds"]),
        confirmation_seeds=_seed_tuple(payload["confirmation"]["seeds"]),
        confirmation_bootstrap_seed=int(
            payload["confirmation"]["bootstrap_seed"]
        ),
        independent_audit=IndependentAudit(
            status=str(audit["status"]),
            attestation_sha256=audit.get("attestation_sha256"),
            expected_prior_count=audit.get("expected_prior_count"),
            expected_prior_sha256=audit.get("expected_prior_sha256"),
            expected_artifact_count=audit.get("expected_artifact_count"),
            expected_artifact_sha256=audit.get("expected_artifact_sha256"),
            expected_source_count=audit.get("expected_source_count"),
            expected_source_sha256=audit.get("expected_source_sha256"),
        ),
    )
    config.validate(require_audit_go=False)
    _validate_yaml_contract(payload)
    return config


def _validate_yaml_contract(payload: Mapping[str, Any]) -> None:
    if payload["dataset"] != DATASET:
        raise ValueError("v5 is isolated to MIMIC-CXR")
    if payload["candidate_order"] != [
        candidate.candidate_id for candidate in bridge_candidates()
    ]:
        raise ValueError("v5 candidate order differs")
    if payload["state_kernel"] != C13_STATE_KERNEL:
        raise ValueError("v5 C13 state kernel differs")
    expected_bridge = {
        "source": "D_env_only",
        "successor_source_feature_names": list(SUCCESSOR_SOURCE_FEATURE_NAMES),
        "successor_source_feature_indices": list(SUCCESSOR_SOURCE_INDICES),
        "clinical_coordinate_count": 16,
        "spo2_hinge": {
            "threshold": SPO2_HINGE_THRESHOLD,
            "scale": SPO2_HINGE_SCALE,
            "direction": "lower_tail",
        },
        "rr_hinge": {
            "threshold": RR_HINGE_THRESHOLD,
            "scale": RR_HINGE_SCALE,
            "direction": "upper_tail",
        },
        "current_action_one_hot": {
            "included_for": [
                "B01_stagewise_successor_bridge",
                "B02_pooled_successor_bridge_stage_one_hot",
            ],
            "action_order": [0, 1, 2],
            "penalized": True,
        },
        "stage_one_hot": {
            "included_for": [
                "B02_pooled_successor_bridge_stage_one_hot"
            ],
            "stage_order": [0, 1, 2, 3, 4, 5],
            "penalized": True,
        },
        "ridge_mode": "sample_normalized_no_intercept",
        "ridge_value": BRIDGE_RIDGE,
        "intercept_penalized": False,
        "joint_two_outcome_donor_residual": True,
    }
    if payload["outcome_bridge"] != expected_bridge:
        raise ValueError("v5 outcome-bridge contract differs")
    if payload["development"]["rng_audit"] != {
        "base_seed_count": 20,
        "stream_count": 100,
        "mapping_sha256": DEVELOPMENT_MAPPING_SHA256,
        "rng_id_set_sha256": DEVELOPMENT_ID_SET_SHA256,
        "common_random_numbers_across_candidates": True,
        "scientific_freshness_claimed": False,
    }:
        raise ValueError("v5 development RNG declaration differs")
    confirmation_audit = payload["confirmation"]["rng_audit"]
    if confirmation_audit != {
        "derived_stream_count": 341,
        "mapping_sha256": CONFIRMATION_MAPPING_SHA256,
        "rng_id_set_sha256": CONFIRMATION_ID_SET_SHA256,
        "base_seed_set_sha256": CONFIRMATION_BASE_SET_SHA256,
        "required_collision_count": 0,
    }:
        raise ValueError("v5 confirmation RNG declaration differs")
    if payload["k0_gate"] != {
        "systematic_replays": 16,
        **K0_THRESHOLDS,
        "active_coordinate_sd_floor": 1e-4,
    }:
        raise ValueError("v5 aggregate K0 gate differs")
    firewall = payload["information_firewall"]
    if firewall.get("scientific_outputs_permitted") is not False or set(
        firewall.get("forbidden", ())
    ) != {"science", "coverage", "width", "method_selection"}:
        raise ValueError("v5 information firewall differs")


def _seed_tuple(section: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(
        range(int(section["start"]), int(section["stop"]), int(section["step"]))
    )


def _tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
