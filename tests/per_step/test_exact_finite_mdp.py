from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scpcp.exact_finite_mdp import (
    ESTIMATOR_NAMES,
    MECHANISM_NAMES,
    ExactFiniteMDPConfig,
    build_paired_mechanisms,
    enumerate_schedules,
    exact_population_surfaces,
    run_exact_finite_mdp,
)
from scpcp.phase0_search import analytic_schedule_metrics
from scpcp.exact_finite_mdp_study import run_replicated_exact_finite_mdp


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_exact_finite_mdp.py"
    spec = importlib.util.spec_from_file_location("run_exact_finite_mdp", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_default_contract_is_eight_by_three_by_four_by_seven() -> None:
    config = ExactFiniteMDPConfig()

    assert (
        config.state_count,
        config.action_count,
        config.horizon,
        config.grid_size,
    ) == (8, 3, 4, 7)
    assert config.schedule_count == 2_401
    assert config.seed == 52_081
    assert config.logged_trajectories == 3_000
    assert config.population_instances == 200
    assert MECHANISM_NAMES == (
        "M0_no_feedback",
        "M1_current_only",
        "M2_history_only",
        "M3_full_feedback",
    )
    assert ESTIMATOR_NAMES[-1] == "full_prefix"

    for reserved_seed in (91_000, 94_000):
        with pytest.raises(ValueError, match="collide with another study"):
            replace(
                config,
                population_seed_start=reserved_seed,
                population_instances=2,
                logged_instance_count=0,
            ).validate()


def test_exact_population_surfaces_isolate_current_history_and_full_prefix() -> None:
    config = ExactFiniteMDPConfig()
    schedules = enumerate_schedules(config)
    mechanisms = build_paired_mechanisms(config)
    surfaces = {}

    for mechanism in mechanisms:
        population, _width = exact_population_surfaces(mechanism, schedules)
        surfaces[mechanism.name] = population
        phase0 = analytic_schedule_metrics(
            mechanism.problem,
            tuple(int(index) for index in schedules[317]),
        )
        assert population[3, 317].tolist() == pytest.approx(
            phase0.coverage.tolist(), abs=1e-12
        )

    m0 = surfaces["M0_no_feedback"]
    assert np.max(np.abs(m0 - m0[3])) < 1e-12

    m1 = surfaces["M1_current_only"]
    assert np.max(np.abs(m1[2] - m1[3])) < 1e-12
    assert np.max(np.abs(m1[0] - m1[3])) > 0.10
    assert np.max(np.abs(m1[1] - m1[3])) > 0.10

    m2 = surfaces["M2_history_only"]
    assert np.max(np.abs(m2[1] - m2[3])) < 1e-12
    assert np.max(np.abs(m2[0] - m2[3])) > 0.20
    assert np.max(np.abs(m2[2] - m2[3])) > 0.20

    m3 = surfaces["M3_full_feedback"]
    assert min(np.max(np.abs(m3[index] - m3[3])) for index in range(3)) > 0.05


def test_finite_sample_output_separates_identification_and_sampling_error() -> None:
    config = replace(
        ExactFiniteMDPConfig(),
        horizon=2,
        grid_size=3,
        logged_trajectories=256,
        beam_width=4,
        surface_chunk_size=4,
    )

    result = run_exact_finite_mdp(config)
    arrays = result.arrays

    assert arrays["population_coverage"].shape == (4, 4, 9, 2)
    assert np.allclose(
        arrays["identification_bias"]
        + arrays["finite_sample_sampling_error"],
        arrays["total_error"],
        atol=0.0,
        rtol=0.0,
    )
    assert np.allclose(
        arrays["population_coverage"][:, 3],
        arrays["true_coverage"],
        atol=0.0,
        rtol=0.0,
    )
    assert np.allclose(
        arrays["hajek_coverage"][0],
        arrays["hajek_coverage"][0, 0],
        atol=0.0,
        rtol=0.0,
    )
    assert np.isfinite(arrays["ess_fraction"]).all()
    assert result.summary["finite_sample_claim"] is False


def test_random_problem_instances_preserve_the_structural_equalities() -> None:
    config = replace(
        ExactFiniteMDPConfig(),
        horizon=2,
        grid_size=3,
    )
    schedules = enumerate_schedules(config)

    for problem_seed in (52_100, 52_101):
        mechanisms = build_paired_mechanisms(config, problem_seed=problem_seed)
        surfaces = {
            mechanism.name: exact_population_surfaces(mechanism, schedules)[0]
            for mechanism in mechanisms
        }
        assert np.max(
            np.abs(surfaces["M0_no_feedback"] - surfaces["M0_no_feedback"][3])
        ) < 1e-12
        assert np.max(
            np.abs(
                surfaces["M1_current_only"][2]
                - surfaces["M1_current_only"][3]
            )
        ) < 1e-12
        assert np.max(
            np.abs(
                surfaces["M2_history_only"][1]
                - surfaces["M2_history_only"][3]
            )
        ) < 1e-12


def test_replicated_study_reports_quantiles_over_predeclared_instances() -> None:
    config = replace(
        ExactFiniteMDPConfig(),
        horizon=2,
        grid_size=3,
        logged_trajectories=64,
        population_instances=3,
        logged_instance_count=1,
        logged_replicates=2,
        beam_width=4,
        surface_chunk_size=4,
    )

    result = run_replicated_exact_finite_mdp(config)

    assert result.arrays["population_problem_seeds"].tolist() == [
        52_100,
        52_101,
        52_102,
    ]
    assert result.arrays["logged_sampling_rmse"].shape == (1, 2, 4, 4)
    search = result.summary["population_instance_audit"]["search"]
    assert search["M2_history_only"]["greedy_relative_regret"]["count"] == 3
    assert "median" in search["M2_history_only"]["greedy_relative_regret"]
    assert result.summary["population_instance_audit"]["decision_gate"] is None


def test_default_grid_reports_exact_greedy_regret_without_a_beam_gate() -> None:
    config = replace(
        ExactFiniteMDPConfig(),
        logged_trajectories=32,
        surface_chunk_size=64,
    )

    result = run_exact_finite_mdp(config)
    m2 = result.summary["search"]["M2_history_only"]

    assert m2["greedy_available"] is True
    assert m2["greedy_absolute_regret"] > 0.0
    assert m2["global"]["mean_normalized_width"] < m2["greedy"][
        "mean_normalized_width"
    ]
    assert m2["beam"]["diagnostic_only"] is True
    assert m2["beam"]["changes_canonical_method"] is False
    assert m2["decision_gate"] is None


def test_runner_publishes_atomic_bundle_and_resume_fails_on_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    config = replace(
        ExactFiniteMDPConfig(),
        horizon=2,
        grid_size=3,
        logged_trajectories=64,
        population_instances=2,
        logged_instance_count=1,
        logged_replicates=2,
        beam_width=4,
        surface_chunk_size=4,
    )
    output = tmp_path / "exact"
    monkeypatch.setattr(runner, "source_tree_sha256", lambda: "a" * 64)
    monkeypatch.setattr(runner, "git_revision", lambda: "deadbeef")

    fresh = runner.run_study(config, output, resume=False)
    resumed = runner.run_study(config, output, resume=True)

    assert fresh == resumed
    assert {
        "config.json",
        "metadata.json",
        "summary.json",
        "surfaces.npz",
        "manifest.json",
        "COMPLETE",
    } == {path.name for path in output.iterdir()}
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["device"] == "cpu_exact"
    assert metadata["seed_collision_audit"]["collision"] is False

    monkeypatch.setattr(runner, "source_tree_sha256", lambda: "b" * 64)
    with pytest.raises(RuntimeError, match="source tree differs"):
        runner.run_study(config, output, resume=True)
