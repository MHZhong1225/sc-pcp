from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scpcp.certification import exact_tabular_l1_lower_bounds
from scpcp.certification import CertificationResult
from scpcp.config import COTConfig, ExperimentConfig, ModelConfig, PolicyConfig, SyntheticConfig
from scpcp.cot import COTDiagnostics, FittedCOT, QConditionalCOT, exact_tabular_cot_l1_error_bound
from scpcp.data import TrajectoryBatch
import scpcp.experiment as experiment
from scpcp.outcome_model import GaussianOutcomeModel
from scpcp.selection import RadiusSelection
from scpcp.simulator import TabularBehaviorPolicy, TabularTreatmentEnvironment


class _GridTabularPolicy(TabularBehaviorPolicy):
    """Known tabular policy adapter exposing the grid interface used by COT."""

    def probabilities_for_grid(self, states: torch.Tensor, q_grid: torch.Tensor) -> torch.Tensor:
        return self.probabilities(states)[:, None, :].expand(-1, len(q_grid), -1)


def test_tabular_validation_config_is_full_scale() -> None:
    path = Path(__file__).parents[2] / "configs" / "per_step_tabular_validation.yaml"
    config = ExperimentConfig.from_yaml(path)

    assert config.data.dataset == "tabular"
    assert config.seeds == tuple(range(200))
    assert config.samples.oracle_rollouts == 50_000
    assert config.cot.loss == "huber"


def test_exact_tabular_l1_bound_enumerates_capped_learned_weights() -> None:
    q_grid = torch.tensor([0.5, 1.0, 1.5])
    outcome = GaussianOutcomeModel(
        state_dim=5,
        n_actions=3,
        config=ModelConfig(hidden_dim=8, representation_dim=4),
    )
    model = QConditionalCOT(
        state_dim=5,
        horizon=2,
        outcome_model=outcome,
        q_grid=q_grid,
        config=COTConfig(hidden_dims=(8,), rho_cap=2.0),
    )
    fitted = FittedCOT(model=model, q_grid=q_grid, diagnostics=COTDiagnostics((), (), ()))
    environment = TabularTreatmentEnvironment(SyntheticConfig(feedback_strength=0.8))
    policy = _GridTabularPolicy()

    error = exact_tabular_cot_l1_error_bound(
        fitted,
        environment,
        q_grid=q_grid,
        target_policy=policy,  # type: ignore[arg-type]
        logging_policy=policy,
        weight_cap=4.0,
    )

    assert error.shape == (len(q_grid), 2)
    assert torch.isfinite(error).all()
    assert (error >= 0.0).all()


def test_exact_tabular_l1_certificate_is_formal_and_cellwise() -> None:
    result = exact_tabular_l1_lower_bounds(
        torch.full((3, 2), 0.99),
        n_trajectories=1_000_000,
        weight_cap=1.0,
        exact_l1_error_bound=torch.full((3, 2), 0.01),
        delta=0.05,
    )

    assert result.formal
    assert result.label == "tabular_exact_l1_oracle_bound"
    assert result.lower_bounds.shape == (3, 2)
    assert torch.all(result.lower_bounds < result.estimates)


def test_exact_mdp_audit_uses_fresh_rollouts_and_selected_certificate_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployed = TrajectoryBatch(
        states=torch.zeros(3, 3, 1),
        actions=torch.zeros(3, 2, dtype=torch.long),
        outcomes=torch.zeros(3, 2, 2),
        patient_ids=torch.arange(3),
    )
    requested_rollouts: list[int] = []

    def fake_oracle_metrics(*args: object, **kwargs: object) -> tuple[torch.Tensor, TrajectoryBatch, torch.Tensor]:
        del args
        requested_rollouts.append(int(kwargs["n_rollouts"]))
        return torch.tensor([0.93, 0.94]), deployed, torch.full((3, 2), 0.4)

    monkeypatch.setattr(experiment, "per_step_oracle_metrics", fake_oracle_metrics)
    model_config = ModelConfig(hidden_dim=8, representation_dim=4)
    outcome_model = GaussianOutcomeModel(state_dim=1, n_actions=1, config=model_config)
    policy = SimpleNamespace(config=PolicyConfig(action_costs=(0.0,)))
    task = SimpleNamespace(environment=object())
    certificate = CertificationResult(
        estimates=torch.tensor([[0.85, 0.86], [0.96, 0.95]]),
        lower_bounds=torch.tensor([[0.75, 0.76], [0.92, 0.91]]),
        sampling_margin=0.03,
        ratio_error_bound=torch.full((2, 2), 0.01),
        formal=True,
        label="tabular_exact_l1_oracle_bound",
    )
    selection = RadiusSelection(radius=1.2, index=1, status="CERTIFIED")

    record = experiment._evaluate_radius_method(
        experiment.SCPCP_METHOD,
        selection.radius,
        task,  # type: ignore[arg-type]
        policy,  # type: ignore[arg-type]
        outcome_model,
        ExperimentConfig(model=model_config, horizon=2),
        seed=11,
        device="cpu",
        selection=selection,
        certificate=certificate,
        information_regime="internal_oracle_ratio_bound_validation_only",
    )

    assert requested_rollouts == [50_000]
    assert record["information_regime"] == "internal_oracle_ratio_bound_validation_only"
    assert record["oracle_evaluation_trajectories"] == deployed.n
    assert record["estimated_min_coverage"] == pytest.approx(0.95)
    assert record["lower_bound_min"] == pytest.approx(0.91)
    assert record["certificate_formal"] is True
    assert record["certified"] is True


def test_exact_mdp_audit_abstention_skips_fresh_rollouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_evaluated(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("an abstained method must not be deployed for fresh evaluation")

    monkeypatch.setattr(experiment, "per_step_oracle_metrics", fail_if_evaluated)
    certificate = CertificationResult(
        estimates=torch.full((2, 2), 0.95),
        lower_bounds=torch.full((2, 2), 0.80),
        sampling_margin=0.1,
        ratio_error_bound=torch.zeros(2, 2),
        formal=True,
        label="tabular_exact_l1_oracle_bound",
    )
    selection = RadiusSelection(radius=None, index=None, status="UNCERTIFIED")

    record = experiment._evaluate_radius_method(
        experiment.SCPCP_METHOD,
        selection.radius,
        SimpleNamespace(environment=object()),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),
        ExperimentConfig(),
        seed=13,
        device="cpu",
        selection=selection,
        certificate=certificate,
        information_regime="internal_oracle_ratio_bound_validation_only",
    )

    assert record["selection_status"] == "UNCERTIFIED"
    assert record["certificate_formal"] is True
    assert record["certified"] is False
    assert record["oracle_evaluation_trajectories"] == 0
    assert torch.isnan(torch.tensor(record["worst_coverage"]))
