from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scpcp.config import SyntheticConfig
from scpcp.phase0_search import (
    AnalyticFiniteMDP,
    NoFeasibleScheduleError,
    analytic_schedule_metrics,
    beam_schedule_search,
    exact_schedule_search,
    greedy_schedule_search,
)
from scpcp.simulator import TabularTreatmentEnvironment


ROOT = Path(__file__).resolve().parents[2]


def _load_search_script():
    path = ROOT / "scripts" / "run_phase0_search_sanity.py"
    spec = importlib.util.spec_from_file_location("run_phase0_search_sanity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _one_state_problem(*, radii: torch.Tensor) -> AnalyticFiniteMDP:
    horizon, grid_size = radii.shape
    return AnalyticFiniteMDP(
        initial_state_probabilities=torch.tensor([1.0]),
        transition_probabilities=torch.ones(1, 1, 1),
        action_probabilities=torch.ones(horizon, grid_size, 1, 1),
        radii=radii,
        predictor_means=torch.zeros(1, 1, 2),
        predictor_scales=torch.ones(1, 1, 2),
        outcome_means=torch.zeros(1, 1, 2),
        outcome_standard_deviations=torch.ones(2),
        outcome_normalization=torch.tensor([2.0, 2.0]),
    )


def _triangular_problem() -> AnalyticFiniteMDP:
    # State 1 is the initial state. At stage 0, radius index 0 sends it to
    # state 0, while index 1 sends it to state 2. Both other states self-loop.
    transition = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ]
    )
    action_probabilities = torch.zeros(2, 2, 3, 2)
    action_probabilities[0, 0, :, 0] = 1.0
    action_probabilities[0, 1, :, 1] = 1.0
    action_probabilities[1, :, :, 0] = 1.0
    predictor_means = torch.zeros(3, 2, 1)
    predictor_means[0, 0, 0] = 3.0
    return AnalyticFiniteMDP(
        initial_state_probabilities=torch.tensor([0.0, 1.0, 0.0]),
        transition_probabilities=transition,
        action_probabilities=action_probabilities,
        radii=torch.tensor([[2.0, 2.1], [2.0, 6.0]]),
        predictor_means=predictor_means,
        predictor_scales=torch.ones(3, 2, 1),
        outcome_means=torch.zeros(2, 3, 1),
        outcome_standard_deviations=torch.ones(1),
        outcome_normalization=torch.ones(1),
    )


def _frozen_grid_greedy_failure_problem() -> AnalyticFiniteMDP:
    base = _triangular_problem()
    action_probabilities = torch.zeros(4, 5, 3, 2)
    action_probabilities[0, 0, :, 0] = 1.0
    action_probabilities[0, 1:, :, 1] = 1.0
    action_probabilities[1:, :, :, 0] = 1.0
    return replace(
        base,
        action_probabilities=action_probabilities,
        radii=torch.tensor(
            [
                [2.0, 2.1, 2.1, 2.1, 2.1],
                [2.0, 2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0, 2.0],
            ]
        ),
    )


def test_tabular_step_fixed_seed_legacy_fixture_is_bitwise_unchanged() -> None:
    environment = TabularTreatmentEnvironment(
        SyntheticConfig(feedback_strength=0.8)
    )
    state = torch.eye(5)[torch.tensor([0, 1, 3, 4])]
    action = torch.tensor([0, 1, 2, 1])
    generator = torch.Generator(device="cpu").manual_seed(24680)

    next_state, outcome = environment.step(state, action, generator, time=2)

    assert torch.equal(
        next_state,
        torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert torch.equal(
        outcome,
        torch.tensor(
            [
                [0.24464425444602966, -0.09009982645511627],
                [-0.10334055125713348, 0.1503487527370453],
                [0.45932286977767944, 0.34942829608917236],
                [0.9449310898780823, 0.23083065450191498],
            ]
        ),
    )


def test_tabular_analytic_helpers_match_the_legacy_population_law() -> None:
    environment = TabularTreatmentEnvironment(
        SyntheticConfig(feedback_strength=0.8)
    )

    initial = environment.initial_state_probabilities(
        torch.device("cpu"), torch.float64
    )
    means, standard_deviations = environment.outcome_distribution_parameters(
        torch.device("cpu"), torch.float64
    )

    assert torch.equal(initial, torch.full((5,), 0.2, dtype=torch.float64))
    assert means.shape == (3, 5, 2)
    assert means[0, 4].tolist() == pytest.approx([1.0, 0.2])
    assert means[2, 0].tolist() == pytest.approx([-0.12, 0.144])
    assert torch.equal(
        standard_deviations,
        torch.tensor([0.08, 0.08], dtype=torch.float64),
    )


def test_one_stage_population_metric_matches_hand_calculation() -> None:
    problem = _one_state_problem(radii=torch.tensor([[1.0]]))

    result = analytic_schedule_metrics(problem, (0,))

    # P(|Z| <= 1)^2 for two independent standard-normal outcomes.
    assert result.coverage.tolist() == pytest.approx(
        [0.4660649426743922], abs=1e-12
    )
    assert result.normalized_width.tolist() == pytest.approx([1.0], abs=1e-12)
    assert torch.equal(
        result.state_occupancy,
        torch.tensor([[1.0], [1.0]], dtype=torch.float64),
    )


def test_two_stage_metric_propagates_target_policy_occupancy_exactly() -> None:
    transition = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    action_probabilities = torch.zeros(2, 1, 2, 2)
    action_probabilities[0, 0, :, 1] = 1.0
    action_probabilities[1, 0, :, 0] = 1.0
    problem = AnalyticFiniteMDP(
        initial_state_probabilities=torch.tensor([1.0, 0.0]),
        transition_probabilities=transition,
        action_probabilities=action_probabilities,
        radii=torch.tensor([[4.0], [4.0]]),
        predictor_means=torch.zeros(2, 2, 1),
        predictor_scales=torch.ones(2, 2, 1),
        outcome_means=torch.zeros(2, 2, 1),
        outcome_standard_deviations=torch.ones(1),
        outcome_normalization=torch.ones(1),
    )

    result = analytic_schedule_metrics(problem, (0, 0))

    assert torch.equal(
        result.state_occupancy,
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        ),
    )
    assert result.coverage.tolist() == pytest.approx(
        [0.9999366575163338, 0.9999366575163338], abs=1e-12
    )


def test_exact_search_exposes_greedy_triangular_suboptimality() -> None:
    result = exact_schedule_search(_triangular_problem(), target=0.90)

    assert result.search_type == "exact"
    assert result.greedy_schedule is not None
    assert result.greedy_schedule.selected_indices == (0, 1)
    assert result.best_found_schedule.selected_indices == (1, 0)
    assert result.greedy_width == pytest.approx(8.0)
    assert result.best_found_width == pytest.approx(4.1)
    assert result.true_optimality_gap == pytest.approx(3.9)
    assert result.best_found_gap == pytest.approx(3.9)


def test_exact_and_greedy_search_break_width_ties_by_lowest_index() -> None:
    problem = _one_state_problem(radii=torch.tensor([[2.0, 2.0]]))

    greedy = greedy_schedule_search(problem, target=0.90)
    exact = exact_schedule_search(problem, target=0.90)

    assert greedy.selected_indices == (0,)
    assert exact.best_found_schedule.selected_indices == (0,)


def test_no_feasible_grid_is_an_explicit_search_failure() -> None:
    problem = _one_state_problem(radii=torch.tensor([[0.01, 0.02]]))

    with pytest.raises(NoFeasibleScheduleError, match="no feasible greedy candidate at stage 0"):
        greedy_schedule_search(problem, target=0.90)
    with pytest.raises(NoFeasibleScheduleError, match="no feasible exact schedule"):
        exact_schedule_search(problem, target=0.90)
    with pytest.raises(NoFeasibleScheduleError, match="no feasible beam prefix at stage 0"):
        beam_schedule_search(problem, target=0.90, beam_width=2)


def test_beam_reports_only_a_best_found_gap() -> None:
    result = beam_schedule_search(
        _triangular_problem(), target=0.90, beam_width=2
    )

    assert result.search_type == "beam"
    assert result.best_found_schedule.selected_indices == (1, 0)
    assert result.true_optimality_gap is None
    assert result.best_found_gap == pytest.approx(3.9)


def test_exact_uses_null_greedy_fields_when_greedy_prefix_fails() -> None:
    problem = replace(
        _triangular_problem(),
        radii=torch.tensor([[2.0, 2.1], [2.0, 2.1]]),
    )

    result = exact_schedule_search(problem, target=0.90)

    assert result.greedy_available is False
    assert result.greedy_schedule is None
    assert result.greedy_width is None
    assert result.true_optimality_gap is None
    assert result.best_found_gap is None
    assert result.best_found_schedule.selected_indices == (1, 0)


def test_search_rejects_nonpositive_beam_width() -> None:
    with pytest.raises(ValueError, match="beam_width must be positive"):
        beam_schedule_search(
            _one_state_problem(radii=torch.tensor([[2.0]])),
            target=0.90,
            beam_width=0,
        )


def test_population_metric_rejects_nonintegral_schedule_indices() -> None:
    with pytest.raises(TypeError, match="schedule indices must be integers"):
        analytic_schedule_metrics(
            _one_state_problem(radii=torch.tensor([[2.0]])),
            (0.0,),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("empty_dimension", ["horizon", "grid", "outcome"])
def test_population_metric_rejects_empty_problem_dimensions(
    empty_dimension: str,
) -> None:
    problem = _one_state_problem(radii=torch.tensor([[2.0]]))
    if empty_dimension == "horizon":
        problem = replace(
            problem,
            radii=torch.empty(0, 1),
            action_probabilities=torch.empty(0, 1, 1, 1),
        )
    elif empty_dimension == "grid":
        problem = replace(
            problem,
            radii=torch.empty(1, 0),
            action_probabilities=torch.empty(1, 0, 1, 1),
        )
    else:
        problem = replace(
            problem,
            predictor_means=torch.empty(1, 1, 0),
            predictor_scales=torch.empty(1, 1, 0),
            outcome_means=torch.empty(1, 1, 0),
            outcome_standard_deviations=torch.empty(0),
            outcome_normalization=torch.empty(0),
        )

    with pytest.raises(
        ValueError,
        match="horizon, grid, and outcome dimensions must be positive",
    ):
        analytic_schedule_metrics(problem, ())


def test_script_builds_float64_problem_from_frozen_tabular_context() -> None:
    script = _load_search_script()
    environment = TabularTreatmentEnvironment(SyntheticConfig(feedback_strength=0.8))

    class FrozenPredictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def predict_all_actions(
            self, states: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            shape = (len(states), environment.n_actions, environment.outcome_dim)
            return torch.zeros(shape, device=states.device), torch.ones(
                shape, device=states.device
            )

    class FrozenPolicy:
        def probabilities_for_grid(
            self, states: torch.Tensor, q_grid: torch.Tensor
        ) -> torch.Tensor:
            return torch.full(
                (len(states), len(q_grid), environment.n_actions),
                1.0 / environment.n_actions,
                device=states.device,
            )

    context = SimpleNamespace(
        task=SimpleNamespace(environment=environment),
        outcome_model=FrozenPredictor(),
        policy=FrozenPolicy(),
        cot_scores=torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
        ),
        outcome_sd=torch.tensor([1.0, 2.0]),
    )

    problem = script.build_search_problem(
        context,
        horizon=4,
        grid_size=5,
        lower_quantile=0.50,
        upper_quantile=0.999,
    )

    assert problem.radii.dtype == torch.float64
    assert problem.action_probabilities.dtype == torch.float64
    assert problem.radii.tolist() == [
        [1.0] * 5,
        [2.0] * 5,
        [3.0] * 5,
        [4.0] * 5,
    ]
    assert problem.action_probabilities.shape == (4, 5, 5, 3)
    assert torch.equal(
        problem.initial_state_probabilities,
        torch.full((5,), 0.2, dtype=torch.float64),
    )


def test_sanity_payload_has_locked_exact_non_gating_schema() -> None:
    script = _load_search_script()
    problem = _one_state_problem(radii=torch.full((4, 5), 2.0))
    diagnostic = exact_schedule_search(problem, target=0.90)

    payload = script.build_sanity_payload(
        problem,
        diagnostic,
        seed=0,
        device="cpu",
        source_hash="1" * 64,
        experiment_hash="2" * 64,
        config_hash="3" * 64,
        target=0.90,
    )

    assert payload["schema_version"] == 1
    assert payload["diagnostic_type"] == "analytic_exact_finite_grid_search"
    assert payload["status"] == "complete"
    assert payload["non_gating"] is True
    assert payload["population_exact"] is True
    assert payload["dataset"] == "tabular"
    assert payload["horizon"] == 4
    assert payload["grid_size"] == 5
    assert payload["schedule_count"] == 625
    assert payload["target_coverage"] == pytest.approx(0.90)
    assert payload["gap_definition"] == (
        "greedy_mean_stage_width_minus_exact_mean_stage_width"
    )
    assert payload["true_finite_grid_gap"] == pytest.approx(0.0)
    for name in ("greedy", "exact"):
        assert len(payload[name]["selected_indices"]) == 4
        assert len(payload[name]["selected_radii"]) == 4
        assert len(payload[name]["coverage"]) == 4
        assert len(payload[name]["normalized_width_by_stage"]) == 4
    json.dumps(payload, allow_nan=False)


def test_sanity_payload_uses_json_null_when_exact_is_feasible_but_greedy_fails() -> None:
    script = _load_search_script()
    problem = _frozen_grid_greedy_failure_problem()
    diagnostic = exact_schedule_search(problem, target=0.90)
    assert diagnostic.greedy_available is False

    payload = script.build_sanity_payload(
        problem,
        diagnostic,
        seed=0,
        device="cpu",
        source_hash="1" * 64,
        experiment_hash="2" * 64,
        config_hash="3" * 64,
        target=0.90,
    )

    assert payload["greedy_available"] is False
    assert payload["greedy"] is None
    assert payload["true_finite_grid_gap"] is None
    assert payload["exact"]["selected_indices"] == [1, 0, 0, 0]
    json.dumps(payload, allow_nan=False)


def test_atomic_json_writer_rejects_nonfinite_values_and_replaces_cleanly(
    tmp_path: Path,
) -> None:
    script = _load_search_script()
    destination = tmp_path / "finite_mdp_sanity.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        script.write_sanity_json(destination, {"gap": math.inf})
    assert not destination.exists()

    script.write_sanity_json(destination, {"status": "complete", "gap": 0.0})

    assert json.loads(destination.read_text()) == {"status": "complete", "gap": 0.0}
    assert list(tmp_path.iterdir()) == [destination]
