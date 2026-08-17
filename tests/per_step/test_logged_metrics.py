from __future__ import annotations

import json
import math

import torch

from scpcp.config import ExperimentConfig, ModelConfig, PolicyConfig
from scpcp.data import TrajectoryBatch
from scpcp.experiment import _logged_record, _unavailable_record
from scpcp.outcome_model import GaussianOutcomeModel
from scpcp.policy import BehaviorAnchoredPolicy
from scpcp.scores import ConformalRegion


class _UniformPolicy:
    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions

    def probabilities(self, states: torch.Tensor) -> torch.Tensor:
        return torch.full((len(states), self.n_actions), 1.0 / self.n_actions, device=states.device)


def _policy_and_batch() -> tuple[BehaviorAnchoredPolicy, TrajectoryBatch, ExperimentConfig, _UniformPolicy]:
    model_config = ModelConfig(hidden_dim=8, representation_dim=4)
    policy_config = PolicyConfig(
        tilt=0.0,
        disease_weight=0.4,
        toxicity_weight=0.6,
        action_costs=(0.0, 0.25),
    )
    config = ExperimentConfig(model=model_config, policy=policy_config, horizon=2)
    outcome = GaussianOutcomeModel(state_dim=2, n_actions=2, config=model_config)
    with torch.no_grad():
        for parameter in outcome.parameters():
            parameter.zero_()
    reference = _UniformPolicy(2)
    region = ConformalRegion(outcome)
    policy = BehaviorAnchoredPolicy(outcome, reference, policy_config, region=region, tilt=0.0)
    batch = TrajectoryBatch(
        states=torch.tensor(
            [
                [[0.0, 0.0], [0.1, -0.1], [0.2, -0.2]],
                [[0.3, 0.0], [0.4, 0.1], [0.5, 0.2]],
            ]
        ),
        actions=torch.tensor([[0, 1], [1, 0]]),
        outcomes=torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 1.0], [4.0, 3.0]],
            ]
        ),
        patient_ids=torch.tensor([11, 12]),
    )
    return policy, batch, config, reference


def test_logged_record_uses_region_volume_and_explicit_nondeployment_value_scope() -> None:
    policy, batch, config, reference = _policy_and_batch()
    radius = 1.5
    scores = torch.full((batch.n, batch.horizon), 0.5)

    record = _logged_record(
        "test",
        radius,
        scores,
        None,
        None,
        None,
        policy,
        reference,
        batch,
        config,
        torch.tensor([radius]),
    )

    states, actions, _ = batch.flat_transitions()
    _, scales = policy.outcome_model(states, actions)
    expected_log_volume = policy.region.log_volume(
        scales, torch.full((len(scales),), radius)
    ).mean()
    observed_cost = (
        config.policy.disease_weight * batch.outcomes[..., 0]
        + config.policy.toxicity_weight * batch.outcomes[..., 1]
        + torch.tensor(config.policy.action_costs)[batch.actions]
    )

    assert record["evaluation_scope"] == "logged_source_trajectories_descriptive_not_target_policy_deployment"
    assert record["prediction_set_metric_scope"] == "observed_logged_state_action_pairs_post_selection_descriptive"
    assert math.isclose(record["logged_descriptive_mean_log_volume"], float(expected_log_volume), rel_tol=1e-6)
    assert math.isclose(record["logged_descriptive_clinical_cost"], float(observed_cost.mean()), rel_tol=1e-6)
    assert math.isclose(record["logged_descriptive_clinical_utility"], -float(observed_cost.mean()), rel_tol=1e-6)
    assert json.loads(record["logged_descriptive_per_time_clinical_cost"]) == [
        float(value) for value in observed_cost.mean(dim=0)
    ]
    assert math.isfinite(record["logged_state_model_estimated_clinical_cost"])
    assert math.isnan(record["clinical_cost"])


def test_unavailable_record_keeps_track_specific_metrics_explicitly_missing() -> None:
    record = _unavailable_record("test", None, None)

    assert record["evaluation_scope"] == "unavailable_target_policy_evaluation"
    assert record["prediction_set_metric_scope"] == "unavailable"
    assert math.isnan(record["mean_log_volume"])
    assert math.isnan(record["logged_descriptive_mean_log_volume"])
    assert math.isnan(record["logged_state_model_estimated_clinical_cost"])
