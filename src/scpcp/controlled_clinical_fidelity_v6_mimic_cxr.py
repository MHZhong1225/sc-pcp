"""Terminal, coverage-blind MIMIC-CXR outcome-bridge repair."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
import yaml

from scpcp.controlled_clinical_fidelity_v5_mimic_cxr import (
    BRIDGE_RIDGE,
    CANONICAL_ACTION_COUNT,
    C13_STATE_KERNEL,
    K0_THRESHOLDS,
    RR_FEATURE_NAMES,
    RR_HINGE_SCALE,
    RR_HINGE_THRESHOLD,
    SPO2_FEATURE_NAMES,
    SPO2_HINGE_SCALE,
    SPO2_HINGE_THRESHOLD,
    SUCCESSOR_SOURCE_FEATURE_NAMES,
    SUCCESSOR_SOURCE_INDICES,
    CxrSuccessorOutcomeBridgeEnvironment,
    build_cxr_environment,
    bridge_candidates,
    successor_clinical_features,
)
from scpcp.controlled_transition import ControlledResidualEnvironment
from scpcp.data import TrajectoryBatch


PROTOCOL = "controlled_clinical_fidelity_v6_mimic_cxr"
DATASET = "mimic_cxr"
CANDIDATE_ID = "R01_o0_stagewise_supported_spo2_slopes_o1_b02"
REGRESSION_ANCHOR_ID = "B02_pooled_successor_bridge_stage_one_hot"
HORIZON = 6
OXYGEN_FEATURE_INDICES = (0, 1, 2, 3, 8, 9, 10, 11)
DEVELOPMENT_LINEAGES = {
    "v5_development": tuple(range(92_600, 92_800, 10)),
    "v5_failed_confirmation": tuple(range(119_000, 119_200, 10)),
}
CONFIRMATION_SEEDS = tuple(range(120_000, 120_200, 10))
CONFIRMATION_BOOTSTRAP_SEED = 12_000_019
DEVELOPMENT_MAPPING_SHA256 = (
    "23d506013a014b1f62b072da87512d38cf741c3a5de6e5a07258a2c1f5ae9dee"
)
DEVELOPMENT_ID_SET_SHA256 = (
    "3056d442a6e0ab0afe087a057ce2c4c5fb149fe8a486a9230d88ff675c58e767"
)
DEVELOPMENT_BASE_SET_SHA256 = (
    "a4d22c564d82cf93a13e4c25ea026b919b38f33668a6332ec2cc3bbfe8b27142"
)
CONFIRMATION_MAPPING_SHA256 = (
    "7ebd9467ddb8f05edd44762c16d85e3506d9bbc263034981278e760da36f45d9"
)
CONFIRMATION_ID_SET_SHA256 = (
    "2c4eca8a8a20e1f911723dc7b17c6a82b25b32aab2d14b918f9ad57d8fd860d8"
)
CONFIRMATION_BASE_SET_SHA256 = (
    "91d90f120a5628dd0d07190a949cad4c76827b16bde3bccfc4f3e729468e170b"
)
DEVELOPMENT_REQUIRED_PASS_COUNT = 20
CONFIRMATION_MINIMUM_PASS_COUNT = 19
REQUIRED_STRUCTURAL_PASS_COUNT = 20

_PARENT_DEVELOPMENT_FILES = {
    "COMPLETE": "a9d968ba1ca338c9f417887bba48a3c003d5c5b23eabace5feb2189efdeb44d2",
    "FINAL_STATUS.json": "661261a8970e36be82fc7746d05886d9724857806f0c6568f4e5d7ae84b448de",
    "frozen_settings.json": "24db41f5ec974a833989793808121fbbdb1e4c7aa6a4afab5174c10edc22c1ce",
    "manifest.json": "5cab76c1341148ad58ee2de6821455ac7738acb65968b249107d30dde49369af",
    "metadata.json": "4b64cd6ad7d861bed5aaa739a137ac037a539ab5f5fd2ab3aa02b7fce8a09d0b",
    "selection.json": "e26fc40c63d7529e200fccb8ac80b1114438312fffcbd3498ddb4fe066095b57",
}
_PARENT_CONFIRMATION_FILES = {
    "COMPLETE": "638b0ba296deabed76d62921d46f6174a8444e15ffd89efe6d5bc39e1a64a3f4",
    "FINAL_STATUS.json": "5f104c0dff121174b52e5ce0c082583744d544cda47a296f5ad0329474472f18",
    "gate.json": "d663b2c5b5d6a7280efe2dceb31c743b94a7ec674825caf3eaad7701d04e6a5b",
    "manifest.json": "a1a89634c268bd3b4b480db49b481bc3cf135025155e06bd85e4369fa9c6baec",
    "metadata.json": "927212270de9215b5b36b112cd59f7ac46701fe96cd2fac8edfb6280ba5db726",
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
    expected_v6_source_contract_sha256: str | None

    @property
    def permits_formal_launch(self) -> bool:
        return (
            self.status == "GO"
            and _is_sha256(self.attestation_sha256)
            and _is_sha256(self.expected_v6_source_contract_sha256)
            and self.attestation_sha256 == independent_audit_attestation_sha256(self)
        )


@dataclass(frozen=True)
class FidelityV6Config:
    parent_development: RootBinding
    parent_failed_confirmation: RootBinding
    development_lineages: Mapping[str, tuple[int, ...]]
    confirmation_seeds: tuple[int, ...]
    confirmation_bootstrap_seed: int
    independent_audit: IndependentAudit

    def validate(self, *, require_audit_go: bool = False) -> None:
        if self.parent_development.root != Path(
            "results/work/controlled_clinical_fidelity_v5_mimic_cxr_development"
        ):
            raise ValueError("v6 development parent differs")
        if self.parent_failed_confirmation.root != Path(
            "results/work/controlled_clinical_fidelity_v5_mimic_cxr_confirmation"
        ):
            raise ValueError("v6 failed-confirmation parent differs")
        if dict(self.parent_development.file_sha256) != _PARENT_DEVELOPMENT_FILES:
            raise ValueError("v6 development-parent hashes differ")
        if (
            dict(self.parent_failed_confirmation.file_sha256)
            != _PARENT_CONFIRMATION_FILES
        ):
            raise ValueError("v6 failed-confirmation-parent hashes differ")
        if dict(self.development_lineages) != DEVELOPMENT_LINEAGES:
            raise ValueError("v6 development lineages differ")
        if self.confirmation_seeds != CONFIRMATION_SEEDS:
            raise ValueError("v6 fresh confirmation bank differs")
        if self.confirmation_bootstrap_seed != CONFIRMATION_BOOTSTRAP_SEED:
            raise ValueError("v6 confirmation bootstrap seed differs")
        audit_values = (
            self.independent_audit.expected_prior_count,
            self.independent_audit.expected_prior_sha256,
            self.independent_audit.expected_artifact_count,
            self.independent_audit.expected_artifact_sha256,
            self.independent_audit.expected_source_count,
            self.independent_audit.expected_source_sha256,
            self.independent_audit.expected_v6_source_contract_sha256,
        )
        if self.independent_audit.status == "PENDING":
            if self.independent_audit.attestation_sha256 is not None or any(
                value is not None for value in audit_values
            ):
                raise ValueError("pending v6 audit must not contain an attestation")
        elif self.independent_audit.status == "GO":
            counts = audit_values[:6:2]
            hashes = (*audit_values[1:6:2], audit_values[6])
            if (
                any(type(value) is not int or value < 0 for value in counts)
                or any(not _is_sha256(value) for value in hashes)
                or not self.independent_audit.permits_formal_launch
            ):
                raise ValueError("v6 independent GO attestation differs")
        else:
            raise ValueError("v6 independent audit status must be PENDING or GO")
        if require_audit_go and not self.independent_audit.permits_formal_launch:
            raise RuntimeError(
                "formal v6 RNG is locked until an independent audit records GO"
            )


@dataclass(frozen=True)
class TerminalBridgeTheta:
    candidate_id: str = CANDIDATE_ID

    def __post_init__(self) -> None:
        if self.candidate_id != CANDIDATE_ID:
            raise ValueError("v6 has exactly one scientific candidate")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": CANDIDATE_ID,
            "dataset": DATASET,
            "state_kernel": dict(C13_STATE_KERNEL),
            "outcome_bridge": terminal_bridge_contract(),
            "regression_anchor": {
                "candidate_id": REGRESSION_ANCHOR_ID,
                "role": "regression_only_not_a_candidate",
            },
        }


def terminal_candidate() -> TerminalBridgeTheta:
    return TerminalBridgeTheta()


def terminal_bridge_contract() -> dict[str, Any]:
    clinical_names = [
        *SUCCESSOR_SOURCE_FEATURE_NAMES,
        *(f"hinge_hypox_{name}" for name in SPO2_FEATURE_NAMES),
        *(f"hinge_tachyp_{name}" for name in RR_FEATURE_NAMES),
    ]
    return {
        "source": "D_env_only",
        "clinical_coordinate_order": clinical_names,
        "outcome0": {
            "fit_scope": "one_model_per_stage",
            "stage_order": list(range(HORIZON)),
            "design_order": [
                "intercept",
                "clinical_X16",
                "current_action_one_hot_0_1_2",
                "action1_times_spo2_X8",
                "action2_times_spo2_X8",
            ],
            "design_width": 36,
            "spo2_interaction_indices_in_X16": list(OXYGEN_FEATURE_INDICES),
            "supported_action_reference": "action0_none",
        },
        "outcome1": {
            "fit_scope": "pooled_over_stages",
            "stage_row_order": list(range(HORIZON)),
            "design_order": [
                "intercept",
                "clinical_X16",
                "current_action_one_hot_0_1_2",
                "stage_one_hot_0_1_2_3_4_5",
            ],
            "design_width": 26,
            "exact_v5_b02_regression": True,
        },
        "ridge_mode": "sample_normalized_no_intercept",
        "ridge_value": BRIDGE_RIDGE,
        "intercept_penalized": False,
        "joint_two_outcome_donor_residual": True,
        "same_donor_state_and_outcome_innovation": True,
    }


def outcome0_design(clinical: Tensor, action: Tensor) -> Tensor:
    """Build the fixed 36-column stage-local outcome-0 design."""

    if clinical.ndim != 2 or clinical.shape[1] != 16:
        raise ValueError("outcome-0 clinical design must have 16 columns")
    one_hot = torch.nn.functional.one_hot(
        action.to(torch.long), CANONICAL_ACTION_COUNT
    ).to(clinical)
    oxygen = clinical[:, OXYGEN_FEATURE_INDICES]
    return torch.cat(
        (
            clinical.new_ones((len(clinical), 1)),
            clinical,
            one_hot,
            one_hot[:, 1:2] * oxygen,
            one_hot[:, 2:3] * oxygen,
        ),
        dim=1,
    )


def outcome1_design(clinical: Tensor, action: Tensor, *, stage: int) -> Tensor:
    """Build the exact 26-column pooled B02 outcome-1 design."""

    if clinical.ndim != 2 or clinical.shape[1] != 16:
        raise ValueError("outcome-1 clinical design must have 16 columns")
    if not 0 <= stage < HORIZON:
        raise ValueError("outcome-1 stage is outside the frozen horizon")
    action_one_hot = torch.nn.functional.one_hot(
        action.to(torch.long), CANONICAL_ACTION_COUNT
    ).to(clinical)
    stage_value = torch.full(
        (len(clinical),), stage, dtype=torch.long, device=clinical.device
    )
    stage_one_hot = torch.nn.functional.one_hot(stage_value, HORIZON).to(clinical)
    return torch.cat(
        (
            clinical.new_ones((len(clinical), 1)),
            clinical,
            action_one_hot,
            stage_one_hot,
        ),
        dim=1,
    )


class CxrTerminalOutcomeBridgeEnvironment(CxrSuccessorOutcomeBridgeEnvironment):
    """Frozen C13 dynamics with the single terminal two-mean bridge."""

    def __init__(
        self,
        batch: TrajectoryBatch,
        *,
        outcome_model: object,
        n_actions: int,
        difficulty: Tensor,
        history_length: int,
        static_indices: tuple[int, ...],
        state_feature_names: tuple[str, ...],
    ) -> None:
        if n_actions != CANONICAL_ACTION_COUNT or batch.horizon != HORIZON:
            raise ValueError("v6 requires six-stage, three-action MIMIC-CXR")
        if batch.outcome_dim != 2:
            raise ValueError("v6 requires exactly two outcome coordinates")
        self.bridge_mode = CANDIDATE_ID
        self.bridge_state_feature_names = state_feature_names
        ControlledResidualEnvironment.__init__(
            self,
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
        self._fit_terminal_bridge(batch)

    def _fit_terminal_bridge(self, batch: TrajectoryBatch) -> None:
        next_frames = (
            batch.states[:, 1:]
            .cpu()
            .reshape(
                batch.n,
                batch.horizon,
                self.history_length,
                self.base_state_dim,
            )[:, :, -1]
        )
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

        outcome1_designs = [
            outcome1_design(clinical[:, stage], actions[:, stage], stage=stage)
            for stage in range(self.horizon)
        ]
        pooled_design = torch.cat(outcome1_designs, dim=0)
        pooled_outcomes = outcomes.transpose(0, 1).reshape(-1, 2)
        b02_joint_coefficient = self._fit_ridge(
            pooled_design,
            pooled_outcomes,
            BRIDGE_RIDGE,
            mode="sample_normalized_no_intercept",
        )
        self._b02_joint_coefficient = b02_joint_coefficient
        self._outcome1_coefficient = b02_joint_coefficient[:, 1:2]

        outcome0_coefficients = []
        outcome0_predictions = []
        for stage in range(self.horizon):
            design = outcome0_design(clinical[:, stage], actions[:, stage])
            coefficient = self._fit_ridge(
                design,
                outcomes[:, stage, 0:1],
                BRIDGE_RIDGE,
                mode="sample_normalized_no_intercept",
            )
            outcome0_coefficients.append(coefficient)
            outcome0_predictions.append((design @ coefficient).squeeze(1))
        self._outcome0_coefficients = tuple(outcome0_coefficients)

        predicted0 = torch.stack(outcome0_predictions, dim=1)
        predicted1 = torch.stack(
            [
                (design @ self._outcome1_coefficient).squeeze(1)
                for design in outcome1_designs
            ],
            dim=1,
        )
        joint_residual = outcomes - torch.stack((predicted0, predicted1), dim=2)
        for stage in range(self.horizon):
            for action in range(self.n_actions):
                selected = actions[:, stage].eq(action)
                library = self._libraries[(stage, action)]
                self._libraries[(stage, action)] = (
                    library[0],
                    library[1],
                    joint_residual[selected, stage],
                    library[3],
                    library[4],
                )

    def _bridge_mean(self, frame: Tensor, action: Tensor, *, stage: int) -> Tensor:
        clinical = successor_clinical_features(frame, self.bridge_state_feature_names)
        mean0 = outcome0_design(clinical, action) @ self._outcome0_coefficients[
            stage
        ].to(clinical)
        mean1 = outcome1_design(
            clinical, action, stage=stage
        ) @ self._outcome1_coefficient.to(clinical)
        return torch.cat((mean0, mean1), dim=1)

    def bridge_identity(self) -> dict[str, Any]:
        identity = {
            **terminal_bridge_contract(),
            "outcome0_coefficient_sha256_by_stage": [
                _tensor_sha256(value.to(torch.float64))
                for value in self._outcome0_coefficients
            ],
            "outcome1_coefficient_sha256": _tensor_sha256(
                self._outcome1_coefficient.to(torch.float64)
            ),
            "b02_joint_coefficient_sha256": _tensor_sha256(
                self._b02_joint_coefficient.to(torch.float64)
            ),
            "joint_residual_libraries": [
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
            ],
        }
        return {**identity, "combined_sha256": _json_sha256(identity)}


def build_terminal_environment(
    batch: TrajectoryBatch,
    *,
    theta: TerminalBridgeTheta,
    outcome_model: object,
    n_actions: int,
    difficulty: Tensor,
    history_length: int,
    static_indices: tuple[int, ...],
    state_feature_names: tuple[str, ...],
) -> CxrTerminalOutcomeBridgeEnvironment:
    if theta != terminal_candidate():
        raise ValueError("v6 received an unknown scientific candidate")
    return CxrTerminalOutcomeBridgeEnvironment(
        batch,
        outcome_model=outcome_model,
        n_actions=n_actions,
        difficulty=difficulty,
        history_length=history_length,
        static_indices=static_indices,
        state_feature_names=state_feature_names,
    )


def build_b02_regression_anchor(
    batch: TrajectoryBatch,
    *,
    outcome_model: object,
    n_actions: int,
    difficulty: Tensor,
    history_length: int,
    static_indices: tuple[int, ...],
    state_feature_names: tuple[str, ...],
) -> ControlledResidualEnvironment:
    anchor = next(
        value
        for value in bridge_candidates()
        if value.candidate_id == REGRESSION_ANCHOR_ID
    )
    return build_cxr_environment(
        batch,
        theta=anchor,
        outcome_model=outcome_model,
        n_actions=n_actions,
        difficulty=difficulty,
        history_length=history_length,
        static_indices=static_indices,
        state_feature_names=state_feature_names,
    )


def normalized_seed_ratio(metrics: Mapping[str, Any]) -> float:
    if not bool(metrics.get("structural_invariants")):
        return math.inf
    ratios = [
        float(metrics[name]) / threshold for name, threshold in K0_THRESHOLDS.items()
    ]
    return max(ratios) if all(math.isfinite(value) for value in ratios) else math.inf


def numeric_seed_ratio(metrics: Mapping[str, Any]) -> float:
    """Return the aggregate numeric K0 ratio independently of structure."""

    ratios = [
        float(metrics[name]) / threshold for name, threshold in K0_THRESHOLDS.items()
    ]
    return max(ratios) if all(math.isfinite(value) for value in ratios) else math.inf


def independent_audit_attestation_sha256(audit: IndependentAudit) -> str:
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
        "v6_source_contract_sha256": audit.expected_v6_source_contract_sha256,
    }
    return _json_sha256(payload)


def load_fidelity_v6_config(path: Path) -> FidelityV6Config:
    payload = yaml.safe_load(path.read_text())
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("unknown clinical-v6 protocol")

    def binding(section: Mapping[str, Any]) -> RootBinding:
        return RootBinding(
            root=Path(section["root"]),
            file_sha256=dict(section["file_sha256"]),
        )

    audit = payload["independent_audit"]
    config = FidelityV6Config(
        parent_development=binding(payload["parent_v5"]["development"]),
        parent_failed_confirmation=binding(payload["parent_v5"]["failed_confirmation"]),
        development_lineages={
            name: _seed_tuple(section)
            for name, section in payload["development"]["lineages"].items()
        },
        confirmation_seeds=_seed_tuple(payload["confirmation"]["seeds"]),
        confirmation_bootstrap_seed=int(payload["confirmation"]["bootstrap_seed"]),
        independent_audit=IndependentAudit(
            status=str(audit["status"]),
            attestation_sha256=audit.get("attestation_sha256"),
            expected_prior_count=audit.get("expected_prior_count"),
            expected_prior_sha256=audit.get("expected_prior_sha256"),
            expected_artifact_count=audit.get("expected_artifact_count"),
            expected_artifact_sha256=audit.get("expected_artifact_sha256"),
            expected_source_count=audit.get("expected_source_count"),
            expected_source_sha256=audit.get("expected_source_sha256"),
            expected_v6_source_contract_sha256=audit.get(
                "expected_v6_source_contract_sha256"
            ),
        ),
    )
    config.validate(require_audit_go=False)
    _validate_yaml_contract(payload)
    return config


def _validate_yaml_contract(payload: Mapping[str, Any]) -> None:
    if set(payload) != {
        "protocol",
        "dataset",
        "terminal_protocol",
        "further_bridge_repair_permitted",
        "terminal_no_v7",
        "parent_v5",
        "development",
        "confirmation",
        "independent_audit",
        "scientific_candidate",
        "state_kernel",
        "outcome_bridge",
        "k0_gate",
        "gates",
        "diagnostics",
        "information_firewall",
    }:
        raise ValueError("v6 YAML top-level schema differs")
    if (
        payload["dataset"] != DATASET
        or payload["terminal_protocol"] is not True
        or payload["further_bridge_repair_permitted"] is not False
        or payload["terminal_no_v7"] is not True
    ):
        raise ValueError("v6 terminal protocol declaration differs")
    if payload["state_kernel"] != C13_STATE_KERNEL:
        raise ValueError("v6 C13 state kernel differs")
    parent = payload["parent_v5"]
    if (
        set(parent) != {"development", "failed_confirmation"}
        or set(parent["development"]) != {"role", "root", "file_sha256"}
        or parent["development"]["role"] != "development_evidence"
        or set(parent["failed_confirmation"])
        != {"role", "root", "file_sha256", "public_failure_record"}
        or parent["failed_confirmation"]["role"] != "reclassified_as_development_only"
        or parent["failed_confirmation"]["public_failure_record"]
        != {
            "support_pass_count": 20,
            "k0_pass_count": 18,
            "structural_pass_count": 20,
            "failed_seeds": [
                {
                    "seed": 119_120,
                    "stage": 1,
                    "outcome": 0,
                    "maximum_signed_residual_w1": 0.2956583463636382,
                },
                {
                    "seed": 119_180,
                    "stage": 3,
                    "outcome": 0,
                    "maximum_signed_residual_w1": 0.3200767965422734,
                },
            ],
        }
    ):
        raise ValueError("v6 parent roles/public failure record differ")
    if payload["scientific_candidate"] != {
        "candidate_id": CANDIDATE_ID,
        "candidate_count": 1,
        "selector_present": False,
        "grid_present": False,
        "anchor": {
            "candidate_id": REGRESSION_ANCHOR_ID,
            "role": "regression_only_not_a_candidate",
        },
        "outcome0": terminal_bridge_contract()["outcome0"],
        "outcome1": terminal_bridge_contract()["outcome1"],
    }:
        raise ValueError("v6 single-candidate contract differs")
    if payload["outcome_bridge"] != {
        "source": "D_env_only",
        "successor_source_feature_names": list(SUCCESSOR_SOURCE_FEATURE_NAMES),
        "successor_source_feature_indices": list(SUCCESSOR_SOURCE_INDICES),
        "clinical_coordinate_count": 16,
        "spo2_interaction_indices_in_X16": list(OXYGEN_FEATURE_INDICES),
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
            "action_order": list(range(CANONICAL_ACTION_COUNT)),
            "penalized": True,
        },
        "stage_one_hot": {
            "stage_order": list(range(HORIZON)),
            "penalized": True,
        },
        "supported_action_interactions": {
            "action_order": [1, 2],
            "penalized": True,
        },
        "ridge_mode": "sample_normalized_no_intercept",
        "ridge_value": BRIDGE_RIDGE,
        "intercept_penalized": False,
        "joint_two_outcome_donor_residual": True,
        "same_donor_state_and_outcome_innovation": True,
    }:
        raise ValueError("v6 terminal bridge declaration differs")
    if set(payload["independent_audit"]) != {
        "status",
        "attestation_sha256",
        "expected_prior_count",
        "expected_prior_sha256",
        "expected_artifact_count",
        "expected_artifact_sha256",
        "expected_source_count",
        "expected_source_sha256",
        "expected_v6_source_contract_sha256",
    }:
        raise ValueError("v6 independent audit schema differs")
    development = payload["development"]
    if (
        set(development) != {"role", "lineages", "rng_audit"}
        or development["role"] != "two_exposed_lineages_development_only"
        or development["lineages"]
        != {
            "v5_development": {"start": 92_600, "stop": 92_800, "step": 10},
            "v5_failed_confirmation": {
                "start": 119_000,
                "stop": 119_200,
                "step": 10,
            },
        }
    ):
        raise ValueError("v6 development role differs")
    development_audit = development["rng_audit"]
    if development_audit != {
        "base_seed_count": 40,
        "stream_count": 200,
        "mapping_sha256": DEVELOPMENT_MAPPING_SHA256,
        "rng_id_set_sha256": DEVELOPMENT_ID_SET_SHA256,
        "base_seed_set_sha256": DEVELOPMENT_BASE_SET_SHA256,
        "exact_authorized_reuse": True,
        "scientific_freshness_claimed": False,
    }:
        raise ValueError("v6 development RNG declaration differs")
    confirmation = payload["confirmation"]
    if (
        set(confirmation)
        != {
            "role",
            "seeds",
            "bootstrap_seed",
            "independent_patient_confirmation_claimed",
            "rng_audit",
        }
        or confirmation["role"] != "fresh_split_terminal_confirmation"
        or confirmation["seeds"] != {"start": 120_000, "stop": 120_200, "step": 10}
        or confirmation["bootstrap_seed"] != CONFIRMATION_BOOTSTRAP_SEED
        or confirmation["independent_patient_confirmation_claimed"] is not False
    ):
        raise ValueError("v6 confirmation role differs")
    confirmation_audit = confirmation["rng_audit"]
    if confirmation_audit != {
        "derived_stream_count": 341,
        "mapping_sha256": CONFIRMATION_MAPPING_SHA256,
        "rng_id_set_sha256": CONFIRMATION_ID_SET_SHA256,
        "base_seed_set_sha256": CONFIRMATION_BASE_SET_SHA256,
        "required_collision_count": 0,
    }:
        raise ValueError("v6 confirmation RNG declaration differs")
    if payload["k0_gate"] != {
        "systematic_replays": 16,
        **K0_THRESHOLDS,
        "active_coordinate_sd_floor": 1e-4,
    }:
        raise ValueError("v6 aggregate K0 gate differs")
    gates = payload["gates"]
    if gates != {
        "development": {
            "per_lineage_seed_count": 20,
            "per_lineage_numeric_pass_count": 20,
            "per_lineage_structural_pass_count": 20,
            "seed_deletion_permitted": False,
        },
        "confirmation": {
            "prespecified_seed_count": 20,
            "support_minimum_pass_count": 19,
            "k0_minimum_pass_count": 19,
            "structural_pass_count": 20,
            "seed_deletion_permitted": False,
        },
    }:
        raise ValueError("v6 gate contract differs")
    if payload["diagnostics"] != {
        "aggregate_gate_unchanged": True,
        "score_ks": "stage_scalar",
        "outcome_coordinate_arrays": [
            "signed_residual_w1",
            "successor_mean_w1",
            "successor_q95_w1",
        ],
        "action_stratification": "descriptive_non_gating",
    }:
        raise ValueError("v6 descriptive diagnostic contract differs")
    if payload["information_firewall"] != {
        "allowed": [
            "support",
            "k0_fidelity",
            "context_identity",
            "provenance",
            "descriptive_diagnostics",
        ],
        "forbidden": ["science", "coverage", "width", "method_selection"],
        "scientific_outputs_permitted": False,
    }:
        raise ValueError("v6 information firewall differs")


def _seed_tuple(section: Mapping[str, Any]) -> tuple[int, ...]:
    if set(section) != {"start", "stop", "step"}:
        raise ValueError("v6 seed-range schema differs")
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
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
