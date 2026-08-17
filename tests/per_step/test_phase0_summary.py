from __future__ import annotations

import io
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scpcp.artifacts import (
    experiment_tree_sha256,
    git_revision,
    source_tree_sha256,
)
from scpcp.phase0_search import AnalyticFiniteMDP, ScheduleEvaluation, SearchDiagnostic
from scpcp.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "phase0_oracle.yaml"
SCENARIOS = ("standard", "tail_shift")
METHODS = ("Current Profiled Oracle", "Greedy Sequential Oracle")
PAYLOAD_NAMES = (
    "phase0_summary.csv",
    "phase0_decision.json",
    "phase0_summary.md",
    "phase0_radius_and_coverage.pdf",
    "phase0_radius_and_coverage.svg",
    "phase0_radius_and_coverage.png",
)
MANIFEST_NAME = "phase0_summary_manifest.json"
OUTPUT_NAMES = PAYLOAD_NAMES + (MANIFEST_NAME,)
GATE_IDS = (
    "tail_stage_lcb",
    "tail_micro_ratio",
    "tail_micro_bootstrap_upper",
    "tail_patient_ratio",
    "tail_selection_count",
    "standard_stage_lcb",
    "standard_micro_ratio",
)


def _load_summary():
    path = ROOT / "scripts" / "summarize_phase0_oracle.py"
    spec = importlib.util.spec_from_file_location("summarize_phase0_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_search_sanity():
    path = ROOT / "scripts" / "run_phase0_search_sanity.py"
    spec = importlib.util.spec_from_file_location("run_phase0_search_sanity_for_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_complete_study(
    root: Path,
    *,
    unavailable_primary: tuple[int, str, str] | set[tuple[int, str, str]] | None = None,
    relative_output_dir: bool = False,
) -> None:
    summary = _load_summary()
    root.mkdir()
    config = ExperimentConfig.from_yaml(CONFIG_PATH).to_dict()
    config["output_dir"] = os.path.relpath(root, ROOT) if relative_output_dir else str(root)
    source_hash = source_tree_sha256()
    experiment_hash = experiment_tree_sha256()
    config_hash = summary.canonical_config_sha256(config)
    revision = git_revision()
    metadata = {
        "git_revision": revision,
        "source_tree_sha256": source_hash,
        "devices": config["devices"],
        "seeds": list(range(100)),
        "execution": {
            "experiment_tree_sha256": experiment_hash,
            "config_sha256": config_hash,
            "workers_per_device": 1,
            "candidate_chunk_size": 16,
        },
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "study_metadata.json").write_text(json.dumps(metadata))
    (root / "study_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "expected_seeds": list(range(100)),
                "completed_seeds": list(range(100)),
                "missing_seeds": [],
                "error": None,
            }
        )
    )
    (root / "COMPLETE").write_text("complete\n")

    unavailable_rows = (
        set()
        if unavailable_primary is None
        else {unavailable_primary}
        if isinstance(unavailable_primary, tuple)
        else unavailable_primary
    )
    for seed in range(100):
        seed_dir = root / f"seed_{seed:05d}"
        seed_dir.mkdir()
        records: list[dict[str, object]] = []
        diagnostics: dict[str, object] = {}
        surfaces: dict[str, np.ndarray] = {}
        for scenario_index, scenario in enumerate(SCENARIOS):
            tuning_seed = 10_000 + 2 * seed + scenario_index
            evaluation_seed = 20_000 + 2 * seed + scenario_index
            scenario_diagnostics: dict[str, object] = {
                "tuning_seed": tuning_seed,
                "evaluation_seed": evaluation_seed,
                "tuning_rollouts": 5_000,
                "evaluation_rollouts": 50_000,
            }
            for method in METHODS:
                unavailable = (seed, scenario, method) in unavailable_rows
                method_key = (
                    "profiled"
                    if method == "Current Profiled Oracle"
                    else "greedy"
                )
                base_q = 1.0 if method_key == "profiled" else 0.9
                q_by_time = np.array(
                    [base_q + 0.01 * stage for stage in range(12)],
                    dtype=float,
                )
                width = (
                    1.0
                    if method_key == "profiled"
                    else (0.98 if scenario == "standard" else 0.80)
                )
                selected_endpoint = False
                if unavailable:
                    row = {
                        "scenario": scenario,
                        "method": method,
                        "seed": seed,
                        "selection_status": "UNAVAILABLE",
                        "selection_available": False,
                        "failure_stage": 0,
                        "selected_endpoint": False,
                        "q_by_time": "[]",
                        "tuning_coverage": "[]",
                        "tuning_width": "[]",
                        "final_coverage": "[]",
                        "final_wilson_lcb": "[]",
                        "final_stage_width": "[]",
                        "micro_normalized_width": float("nan"),
                        "patient_normalized_width": float("nan"),
                        "tuning_seed": tuning_seed,
                        "evaluation_seed": evaluation_seed,
                        "n_rollouts": 0,
                    }
                    schedule = np.empty(0)
                    selection_diagnostics = {
                        "selection_available": False,
                        "failure_stage": 0,
                        "selected_endpoint": False,
                        "selected_indices": [],
                    }
                else:
                    row = {
                        "scenario": scenario,
                        "method": method,
                        "seed": seed,
                        "selection_status": "SELECTED",
                        "selection_available": True,
                        "failure_stage": None,
                        "selected_endpoint": selected_endpoint,
                        "q_by_time": json.dumps(q_by_time.tolist()),
                        "tuning_coverage": json.dumps([0.95] * 12),
                        "tuning_width": json.dumps([width] * 12),
                        "final_coverage": json.dumps([0.95] * 12),
                        "final_wilson_lcb": json.dumps([0.94] * 12),
                        "final_stage_width": json.dumps([width] * 12),
                        "micro_normalized_width": width,
                        "patient_normalized_width": width,
                        "tuning_seed": tuning_seed,
                        "evaluation_seed": evaluation_seed,
                        "n_rollouts": 50_000,
                    }
                    schedule = q_by_time
                    selection_diagnostics = {
                        "selection_available": True,
                        "failure_stage": None,
                        "selected_endpoint": selected_endpoint,
                        "selected_indices": [1]
                        if method_key == "profiled"
                        else [1] * 12,
                    }
                records.append(row)
                scenario_diagnostics[method_key] = selection_diagnostics
                surfaces[f"{scenario}_{method_key}_selected_schedule"] = schedule
                if method_key == "profiled":
                    scale_grid = 0.99 + 0.01 * np.arange(101, dtype=float)
                    profile = q_by_time.copy()
                    candidate_schedules = scale_grid[:, None] * profile[None, :]
                    surfaces.update(
                        {
                            f"{scenario}_profiled_scale_grid": scale_grid,
                            f"{scenario}_profile": profile,
                            f"{scenario}_profiled_candidate_schedules": candidate_schedules,
                            f"{scenario}_profiled_candidate_coverage": np.full(
                                (101, 12), 0.95
                            ),
                            f"{scenario}_profiled_candidate_normalized_width": np.full(
                                (101, 12), width
                            ),
                        }
                    )
                else:
                    offsets = -0.01 + 0.01 * np.arange(101, dtype=float)
                    surfaces[f"{scenario}_greedy_stage_grids"] = (
                        q_by_time[:, None] + offsets[None, :]
                    )

            frozen_profile = np.array([1.0 + 0.01 * stage for stage in range(12)])
            common_schedule = 0.95 * frozen_profile
            common_width = 0.95 if scenario == "standard" else 0.85
            common_coverage = [0.945] * 12
            common = {
                "selection_available": True,
                "failure_stage": None,
                "selected_endpoint": False,
                "selected_indices": [1],
                "selected_schedule": common_schedule.tolist(),
                "final_coverage": common_coverage,
                "final_wilson_lcb": [0.935] * 12,
                "final_stage_width": [common_width] * 12,
                "micro_normalized_width": common_width,
                "patient_normalized_width": common_width,
                "n_rollouts": 50_000,
            }
            scenario_diagnostics["profiled_common_grid"] = common
            diagnostics[scenario] = scenario_diagnostics
            common_scale_grid = 0.94 + 0.01 * np.arange(101, dtype=float)
            candidate_schedules = (
                common_scale_grid[:, None] * frozen_profile[None, :]
            )
            prefix = f"{scenario}_profiled_common_grid_"
            surfaces.update(
                {
                    f"{prefix}scale_grid": common_scale_grid,
                    f"{prefix}candidate_schedules": candidate_schedules,
                    f"{prefix}candidate_coverage": np.full((101, 12), 0.945),
                    f"{prefix}candidate_normalized_width": np.full(
                        (101, 12), common_width
                    ),
                    f"{prefix}selected_schedule": common_schedule,
                    f"{prefix}final_coverage": np.array(common_coverage),
                    f"{prefix}final_wilson_lcb": np.full(12, 0.935),
                    f"{prefix}final_stage_width": np.full(12, common_width),
                    f"{prefix}micro_normalized_width": np.array(common_width),
                    f"{prefix}patient_normalized_width": np.array(common_width),
                    f"{prefix}n_rollouts": np.array(50_000),
                }
            )
        pd.DataFrame(records).to_csv(seed_dir / "records.csv", index=False)
        np.savez_compressed(seed_dir / "surfaces.npz", **surfaces)
        seed_metadata = {
            "seed": seed,
            "device": config["devices"][seed % 2],
            "git_revision": revision,
            "source_tree_sha256": source_hash,
            "runtime": {},
            "diagnostics": diagnostics,
            "config": config,
        }
        (seed_dir / "metadata.json").write_text(json.dumps(seed_metadata))
        (seed_dir / "COMPLETE").write_text(
            json.dumps({"seed": seed, "status": "complete"})
        )

    problem = AnalyticFiniteMDP(
        initial_state_probabilities=torch.ones(1),
        transition_probabilities=torch.ones(1, 1, 1),
        action_probabilities=torch.ones(4, 5, 1, 1),
        radii=torch.full((4, 5), 2.0),
        predictor_means=torch.zeros(1, 1, 1),
        predictor_scales=torch.ones(1, 1, 1),
        outcome_means=torch.zeros(1, 1, 1),
        outcome_standard_deviations=torch.ones(1),
        outcome_normalization=torch.ones(1),
    )
    schedule = ScheduleEvaluation(
        selected_indices=(0, 0, 0, 0),
        coverage=torch.full((4,), 0.95),
        normalized_width=torch.full((4,), 4.0),
        state_occupancy=torch.ones(5, 1),
    )
    diagnostic = SearchDiagnostic(
        search_type="exact",
        greedy_width=4.0,
        best_found_width=4.0,
        true_optimality_gap=0.0,
        best_found_gap=0.0,
        greedy_available=True,
        greedy_schedule=schedule,
        best_found_schedule=schedule,
    )
    search_sanity = _load_search_sanity()
    reduced = search_sanity.reduced_search_config(
        ExperimentConfig.from_yaml(search_sanity.DEFAULT_CONFIG),
        device="cuda:0",
        output=Path("results/work/phase0a_finite_mdp_sanity.json"),
    )
    sanity = search_sanity.build_sanity_payload(
        problem,
        diagnostic,
        seed=0,
        device="cuda:0",
        source_hash=source_hash,
        experiment_hash=experiment_hash,
        config_hash=search_sanity.canonical_config_sha256(reduced.to_dict()),
        target=0.9,
    )
    (root / "finite_mdp_sanity.json").write_text(json.dumps(sanity))


def test_seed_mean_bonferroni_t_lcb_matches_hand_checked_oracle() -> None:
    summary = _load_summary()
    coverage = np.repeat(
        np.array([[0.89], [0.90], [0.91], [0.92], [0.93]]),
        12,
        axis=1,
    )

    result = summary.compute_coverage_summary(coverage, horizon=12)

    assert result["interval_method"] == "seed_mean_bonferroni_t_lcb"
    assert result["n_selected"] == 5
    assert result["critical_value"] == pytest.approx(4.851008443005503)
    assert result["mean"] == pytest.approx([0.91] * 12)
    assert result["lower"] == pytest.approx([0.875698190343576] * 12)
    assert result["minimum_stage_seed_mean_coverage"] == pytest.approx(0.91)
    assert result["minimum_stage_seed_mean_simultaneous_lcb"] == pytest.approx(
        0.875698190343576
    )


def test_coverage_summary_keeps_four_worst_definitions_separate() -> None:
    summary = _load_summary()
    coverage = np.array([[0.80, 1.00], [0.90, 0.95], [1.00, 0.90]])

    result = summary.compute_coverage_summary(coverage, horizon=2)

    assert result["minimum_stage_seed_mean_coverage"] == pytest.approx(0.90)
    assert result["minimum_stage_seed_mean_simultaneous_lcb"] == pytest.approx(
        0.6515862288280455
    )
    assert result["mean_seed_minimum_stage_coverage"] == pytest.approx(2.6 / 3.0)
    assert result["minimum_seed_stage_coverage"] == pytest.approx(0.80)


def test_coverage_interval_is_unavailable_with_fewer_than_two_selections() -> None:
    summary = _load_summary()

    result = summary.compute_coverage_summary(
        np.repeat(np.array([[0.95]]), 12, axis=1),
        horizon=12,
    )

    assert result["n_selected"] == 1
    assert result["lower"] is None
    assert result["minimum_stage_seed_mean_simultaneous_lcb"] is None


def test_paired_bootstrap_matches_locked_oracle_and_is_order_invariant() -> None:
    summary = _load_summary()
    ratios = np.array([0.8, 0.9, 1.1, 1.2])
    seeds = np.array([0, 1, 2, 3])
    numerator = ratios * 10.0
    denominator = np.full(4, 10.0)

    result = summary.compute_paired_width_inference(
        seeds,
        numerator,
        denominator,
        numerator**2,
        denominator**2,
        comparison_name="locked-oracle",
    )
    reordered = summary.compute_paired_width_inference(
        seeds[::-1],
        numerator[::-1],
        denominator[::-1],
        numerator[::-1] ** 2,
        denominator[::-1] ** 2,
        comparison_name="locked-oracle",
    )

    assert result == reordered
    assert result["n_paired"] == 4
    assert result["micro"]["geometric_mean_ratio"] == pytest.approx(
        0.9873624504488285
    )
    assert result["micro"]["ci_lower"] == pytest.approx(0.8239068575628471)
    assert result["micro"]["ci_upper"] == pytest.approx(1.1489125293076057)
    assert result["patient"]["geometric_mean_ratio"] == pytest.approx(
        result["micro"]["geometric_mean_ratio"] ** 2
    )
    assert result["patient"]["ci_lower"] == pytest.approx(
        result["micro"]["ci_lower"] ** 2
    )
    assert result["patient"]["ci_upper"] == pytest.approx(
        result["micro"]["ci_upper"] ** 2
    )
    assert result["bootstrap"] == {
        "resamples": 10_000,
        "rng_seed": 2_718_281,
        "quantile_method": "linear_percentile_95",
        "paired_index_sha256": result["bootstrap"]["paired_index_sha256"],
    }


@pytest.mark.parametrize(
    "bad",
    [0.0, -0.1, float("nan"), float("inf")],
)
def test_paired_width_rejects_nonpositive_or_nonfinite_selected_width(bad: float) -> None:
    summary = _load_summary()
    numerator = np.array([0.8, bad])

    with pytest.raises(ValueError, match="strictly positive and finite"):
        summary.compute_paired_width_inference(
            np.array([0, 1]),
            numerator,
            np.ones(2),
            np.ones(2),
            np.ones(2),
            comparison_name="invalid-width",
        )


def test_paired_interval_is_unavailable_with_fewer_than_two_pairs() -> None:
    summary = _load_summary()

    result = summary.compute_paired_width_inference(
        np.array([7]),
        np.array([0.8]),
        np.array([1.0]),
        np.array([0.8]),
        np.array([1.0]),
        comparison_name="one-pair",
    )

    assert result["n_paired"] == 1
    assert result["micro"]["geometric_mean_ratio"] == pytest.approx(0.8)
    assert result["micro"]["ci_lower"] is None
    assert result["micro"]["ci_upper"] is None


def _passing_gate_inputs() -> dict[str, object]:
    return {
        "tail_greedy_stage_lcb": np.full(12, 0.91),
        "tail_micro_ratio": 0.89,
        "tail_micro_bootstrap_upper": 0.99,
        "tail_patient_ratio": 0.91,
        "tail_greedy_selection_count": 95,
        "tail_n_paired": 100,
        "standard_greedy_stage_lcb": np.full(12, 0.91),
        "standard_micro_ratio": 1.01,
        "standard_n_paired": 100,
    }


@pytest.mark.parametrize(
    "gate_id,mutation",
    [
        ("tail_stage_lcb", {"tail_greedy_stage_lcb": np.r_[0.899, np.full(11, 0.91)]}),
        ("tail_micro_ratio", {"tail_micro_ratio": 0.9000000001}),
        ("tail_micro_bootstrap_upper", {"tail_micro_bootstrap_upper": 1.0}),
        ("tail_patient_ratio", {"tail_patient_ratio": 0.9200000001}),
        ("tail_selection_count", {"tail_greedy_selection_count": 94}),
        (
            "standard_stage_lcb",
            {"standard_greedy_stage_lcb": np.r_[0.899, np.full(11, 0.91)]},
        ),
        ("standard_micro_ratio", {"standard_micro_ratio": 1.0200001}),
    ],
)
def test_each_primitive_gate_independently_forces_no_go(
    gate_id: str,
    mutation: dict[str, object],
) -> None:
    summary = _load_summary()
    inputs = _passing_gate_inputs()
    inputs.update(mutation)

    result = summary.evaluate_go_no_go(**inputs)

    by_id = {gate["id"]: gate for gate in result["gates"]}
    assert len(result["gates"]) == 7
    assert by_id[gate_id]["passed"] is False
    assert sum(not gate["passed"] for gate in result["gates"]) == 1
    assert result["decision"] == "NO_GO"


def test_literal_non_strict_boundaries_and_selection_95_pass() -> None:
    summary = _load_summary()
    inputs = _passing_gate_inputs()
    inputs.update(
        tail_greedy_stage_lcb=np.full(12, 0.90),
        tail_micro_ratio=0.90,
        tail_micro_bootstrap_upper=0.999999999999,
        tail_patient_ratio=0.92,
        tail_greedy_selection_count=95,
        standard_greedy_stage_lcb=np.full(12, 0.90),
        standard_micro_ratio=1.02,
    )

    result = summary.evaluate_go_no_go(**inputs)

    assert all(gate["passed"] for gate in result["gates"])
    assert result["decision"] == "GO"


def test_unavailable_gate_is_false_and_forces_no_go() -> None:
    summary = _load_summary()
    inputs = _passing_gate_inputs()
    inputs["tail_micro_bootstrap_upper"] = None

    result = summary.evaluate_go_no_go(**inputs)

    gate = next(gate for gate in result["gates"] if gate["id"] == "tail_micro_bootstrap_upper")
    assert gate["available"] is False
    assert gate["passed"] is False
    assert result["decision"] == "NO_GO"


def test_one_pair_makes_every_width_gate_unavailable_and_no_go() -> None:
    summary = _load_summary()
    inputs = _passing_gate_inputs()
    inputs["tail_n_paired"] = 1
    inputs["standard_n_paired"] = 1

    result = summary.evaluate_go_no_go(**inputs)
    by_id = {gate["id"]: gate for gate in result["gates"]}

    for gate_id in (
        "tail_micro_ratio",
        "tail_micro_bootstrap_upper",
        "tail_patient_ratio",
        "standard_micro_ratio",
    ):
        assert by_id[gate_id]["value"] is None
        assert by_id[gate_id]["available"] is False
        assert by_id[gate_id]["passed"] is False
    assert result["decision"] == "NO_GO"


def test_ratio_point_can_pass_while_bootstrap_upper_gate_fails() -> None:
    summary = _load_summary()
    ratios = np.tile(np.array([0.2, 3.9605]), 10)
    width = summary.compute_paired_width_inference(
        np.arange(len(ratios)),
        ratios,
        np.ones(len(ratios)),
        ratios,
        np.ones(len(ratios)),
        comparison_name="wide-ratio",
    )
    inputs = _passing_gate_inputs()
    inputs["tail_micro_ratio"] = width["micro"]["geometric_mean_ratio"]
    inputs["tail_micro_bootstrap_upper"] = width["micro"]["ci_upper"]

    result = summary.evaluate_go_no_go(**inputs)
    by_id = {gate["id"]: gate for gate in result["gates"]}

    assert width["micro"]["geometric_mean_ratio"] == pytest.approx(0.89)
    assert width["micro"]["ci_upper"] > 1.0
    assert by_id["tail_micro_ratio"]["passed"] is True
    assert by_id["tail_micro_bootstrap_upper"]["passed"] is False
    assert result["decision"] == "NO_GO"


def test_complete_100_seed_study_validates_and_preserves_analysis_roles(
    tmp_path: Path,
) -> None:
    summary = _load_summary()
    study = tmp_path / "complete"
    _make_complete_study(study)

    result = summary.load_validate_and_analyze(study)

    assert result["integrity"] == {
        "status": "validated_complete",
        "expected_seeds": 100,
        "completed_seeds": 100,
        "primary_rows": 400,
    }
    assert result["decision"] == "GO"
    assert len(result["gates"]) == 7
    assert all(gate["passed"] for gate in result["gates"])
    tail_greedy = result["primary"]["tail_shift"]["Greedy Sequential Oracle"]
    assert tail_greedy["coverage"]["conditioning"] == (
        "conditional_on_successful_selection"
    )
    assert tail_greedy["coverage"]["minimum_stage_seed_mean_coverage"] == (
        pytest.approx(0.95)
    )
    assert tail_greedy["selection_count"] == 100
    assert result["primary_comparisons"]["tail_shift"]["n_paired"] == 100
    assert result["sensitivity"]["analysis_role"] == "sensitivity_only_non_gating"
    assert result["finite_mdp_sanity"]["non_gating"] is True
    assert result["finite_mdp_sanity"]["in_gate"] is False


def test_relative_runner_output_path_accepts_equivalent_absolute_summary_cli(
    tmp_path: Path,
) -> None:
    study = tmp_path / "relative-runner-path"
    _make_complete_study(study, relative_output_dir=True)
    script = ROOT / "scripts" / "summarize_phase0_oracle.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--input-dir", str(study.resolve())],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "GO"
    assert (study / MANIFEST_NAME).is_file()


@pytest.mark.parametrize("field", ["tuning_coverage", "tuning_width"])
def test_profiled_tuning_vectors_must_match_selected_npz_candidate(
    tmp_path: Path,
    field: str,
) -> None:
    summary = _load_summary()
    study = tmp_path / f"profiled-{field}"
    _make_complete_study(study)
    records_path = study / "seed_00000" / "records.csv"
    records = pd.read_csv(records_path)
    row = (records["scenario"] == "standard") & (
        records["method"] == "Current Profiled Oracle"
    )
    values = json.loads(records.loc[row, field].iloc[0])
    values[0] -= 0.01
    records.loc[row, field] = json.dumps(values)
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="tuning metrics disagree with profiled candidate"):
        summary.load_validate_and_analyze(study)


@pytest.mark.parametrize(
    "failure_stage,indices",
    [(0, [1] * 12), (None, [])],
    ids=("wrong-prefix-length", "missing-failure-stage"),
)
def test_unavailable_greedy_indices_must_equal_the_successful_prefix(
    tmp_path: Path,
    failure_stage: int | None,
    indices: list[int],
) -> None:
    summary = _load_summary()
    study = tmp_path / "invalid-greedy-prefix"
    _make_complete_study(
        study,
        unavailable_primary=(0, "standard", "Greedy Sequential Oracle"),
    )
    metadata_path = study / "seed_00000" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    greedy = metadata["diagnostics"]["standard"]["greedy"]
    greedy["failure_stage"] = failure_stage
    greedy["selected_indices"] = indices
    metadata_path.write_text(json.dumps(metadata))
    records_path = study / "seed_00000" / "records.csv"
    records = pd.read_csv(records_path)
    row = (records["scenario"] == "standard") & (records["method"] == METHODS[1])
    records.loc[row, "failure_stage"] = failure_stage
    records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="unavailable greedy prefix"):
        summary.load_validate_and_analyze(study)


@pytest.mark.parametrize("method", METHODS)
def test_selected_tuning_coverage_must_meet_the_locked_target(
    tmp_path: Path,
    method: str,
) -> None:
    summary = _load_summary()
    study = tmp_path / f"undercovered-{method.split()[0]}"
    _make_complete_study(study)
    seed_dir = study / "seed_00000"
    records_path = seed_dir / "records.csv"
    records = pd.read_csv(records_path)
    row = (records["scenario"] == "standard") & (records["method"] == method)
    coverage = json.loads(records.loc[row, "tuning_coverage"].iloc[0])
    coverage[0] = 0.89
    records.loc[row, "tuning_coverage"] = json.dumps(coverage)
    records.to_csv(records_path, index=False)
    if method == "Current Profiled Oracle":
        surface_path = seed_dir / "surfaces.npz"
        with np.load(surface_path, allow_pickle=False) as archive:
            surfaces = {name: np.asarray(archive[name]) for name in archive.files}
        surfaces["standard_profiled_candidate_coverage"][1, 0] = 0.89
        np.savez_compressed(surface_path, **surfaces)

    with pytest.raises(RuntimeError, match="selected tuning coverage must be at least 0.90"):
        summary.load_validate_and_analyze(study)


@pytest.mark.parametrize("method", ["profiled", "greedy", "common"])
def test_selected_endpoint_must_be_recomputed_from_selected_indices(
    tmp_path: Path,
    method: str,
) -> None:
    summary = _load_summary()
    study = tmp_path / f"endpoint-{method}"
    _make_complete_study(study)
    seed_dir = study / "seed_00000"
    metadata_path = seed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    diagnostic_key = "profiled_common_grid" if method == "common" else method
    metadata["diagnostics"]["standard"][diagnostic_key]["selected_endpoint"] = True
    metadata_path.write_text(json.dumps(metadata))
    if method != "common":
        records_path = seed_dir / "records.csv"
        records = pd.read_csv(records_path)
        record_method = METHODS[0] if method == "profiled" else METHODS[1]
        row = (records["scenario"] == "standard") & (
            records["method"] == record_method
        )
        records.loc[row, "selected_endpoint"] = True
        records.to_csv(records_path, index=False)

    with pytest.raises(RuntimeError, match="selected_endpoint disagrees with selected index"):
        summary.load_validate_and_analyze(study)


@pytest.mark.parametrize("grid", ["profiled", "common"])
def test_candidate_schedules_must_equal_scale_times_frozen_profile(
    tmp_path: Path,
    grid: str,
) -> None:
    summary = _load_summary()
    study = tmp_path / f"schedule-surface-{grid}"
    _make_complete_study(study)
    surface_path = study / "seed_00000" / "surfaces.npz"
    with np.load(surface_path, allow_pickle=False) as archive:
        surfaces = {name: np.asarray(archive[name]) for name in archive.files}
    key = (
        "standard_profiled_candidate_schedules"
        if grid == "profiled"
        else "standard_profiled_common_grid_candidate_schedules"
    )
    surfaces[key][0, 0] += 0.01
    np.savez_compressed(surface_path, **surfaces)

    with pytest.raises(RuntimeError, match="candidate schedules disagree with scale/profile"):
        summary.load_validate_and_analyze(study)


def test_selection_failure_is_valid_but_excluded_from_paired_width_denominator(
    tmp_path: Path,
) -> None:
    summary = _load_summary()
    study = tmp_path / "one-unavailable"
    _make_complete_study(
        study,
        unavailable_primary=(0, "tail_shift", "Greedy Sequential Oracle"),
    )

    result = summary.load_validate_and_analyze(study)

    tail_greedy = result["primary"]["tail_shift"]["Greedy Sequential Oracle"]
    comparison = result["primary_comparisons"]["tail_shift"]
    assert tail_greedy["selection_count"] == 99
    assert tail_greedy["coverage"]["n_selected"] == 99
    assert comparison["n_paired"] == 99
    assert comparison["micro"]["geometric_mean_ratio"] == pytest.approx(0.8)


def test_standard_one_pair_marks_ratio_gate_unavailable_and_forces_no_go(
    tmp_path: Path,
) -> None:
    summary = _load_summary()
    study = tmp_path / "standard-one-pair"
    _make_complete_study(
        study,
        unavailable_primary={
            (seed, "standard", "Greedy Sequential Oracle") for seed in range(1, 100)
        },
    )

    result = summary.load_validate_and_analyze(study)
    gate = next(gate for gate in result["gates"] if gate["id"] == "standard_micro_ratio")

    assert result["primary_comparisons"]["standard"]["n_paired"] == 1
    assert gate["value"] is None
    assert gate["available"] is False
    assert gate["passed"] is False
    assert result["decision"] == "NO_GO"
    summary._render_figure(result, tmp_path / "one-pair-figure")
    svg = (tmp_path / "one-pair-figure.svg").read_text()
    assert "G / Current (n=1)" in svg


def test_finite_mdp_sanity_allows_a_failed_greedy_prefix_as_non_gating(
    tmp_path: Path,
) -> None:
    summary = _load_summary()
    study = tmp_path / "greedy-unavailable-sanity"
    _make_complete_study(study)
    path = study / "finite_mdp_sanity.json"
    sanity = json.loads(path.read_text())
    sanity["greedy_available"] = False
    sanity["greedy"] = None
    sanity["true_finite_grid_gap"] = None
    path.write_text(json.dumps(sanity))

    result = summary.load_validate_and_analyze(study)

    assert result["finite_mdp_sanity"]["greedy_available"] is False
    assert result["finite_mdp_sanity"]["greedy"] is None
    assert result["finite_mdp_sanity"]["true_finite_grid_gap"] is None
    assert result["finite_mdp_sanity"]["in_gate"] is False


def test_common_grid_selection_failure_is_valid_and_sensitivity_stays_paired(
    tmp_path: Path,
) -> None:
    summary = _load_summary()
    study = tmp_path / "common-unavailable"
    _make_complete_study(study)
    seed_dir = study / "seed_00000"
    metadata_path = seed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    common = metadata["diagnostics"]["tail_shift"]["profiled_common_grid"]
    common.update(
        selection_available=False,
        failure_stage=0,
        selected_endpoint=False,
        selected_indices=[],
        selected_schedule=[],
        final_coverage=[],
        final_wilson_lcb=[],
        final_stage_width=[],
        micro_normalized_width=float("nan"),
        patient_normalized_width=float("nan"),
        n_rollouts=0,
    )
    metadata_path.write_text(json.dumps(metadata))
    surface_path = seed_dir / "surfaces.npz"
    with np.load(surface_path, allow_pickle=False) as archive:
        surfaces = {name: np.asarray(archive[name]) for name in archive.files}
    prefix = "tail_shift_profiled_common_grid_"
    for suffix in (
        "selected_schedule",
        "final_coverage",
        "final_wilson_lcb",
        "final_stage_width",
    ):
        surfaces[prefix + suffix] = np.empty(0)
    surfaces[prefix + "micro_normalized_width"] = np.array(float("nan"))
    surfaces[prefix + "patient_normalized_width"] = np.array(float("nan"))
    surfaces[prefix + "n_rollouts"] = np.array(0)
    np.savez_compressed(surface_path, **surfaces)

    result = summary.load_validate_and_analyze(study)
    sensitivity = result["sensitivity"]["scenarios"]["tail_shift"]

    assert sensitivity["common_grid_profiled"]["selection_count"] == 99
    assert sensitivity["greedy_over_common_grid"]["n_paired"] == 99
    assert sensitivity["common_grid_over_exact_current"]["n_paired"] == 99


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_marker", "COMPLETE"),
        ("399_rows", "exactly four primary rows"),
        ("401_rows", "exactly four primary rows"),
        ("duplicate_method", "exactly four primary rows"),
        ("nonintegral_seed", "integer dtype"),
        ("wrong_vector_length", "length 12"),
        ("truncated_npz_member", "surfaces.npz is unreadable"),
        ("stream_collision", "tuning streams must be unique"),
        ("common_grid_disagreement", "common-grid metadata/surface disagreement"),
        ("sanity_in_gate", "finite-MDP sanity must be non-gating"),
        ("partial_seed_path", "partial or unexpected seed path"),
        ("provenance", "source_tree_sha256"),
        ("status_seed_type", "strict integer seeds"),
        ("primary_index_shape", "selected_indices length"),
        ("missing_common_surface", "common-grid surface is missing"),
        ("sanity_undercoverage", "finite-MDP sanity exact coverage is infeasible"),
        ("config_synthetic", "frozen config"),
        ("config_policy", "frozen config"),
        ("config_profile", "frozen config"),
        ("primary_grid_shape", "primary surface schema"),
        ("profiled_index_disagreement", "selected profiled candidate disagrees"),
        ("greedy_index_disagreement", "selected greedy candidate disagrees"),
        ("candidate_nan", "primary surface schema"),
        ("sanity_device", "finite-MDP sanity device"),
        ("sanity_config_hash", "finite-MDP sanity config_sha256"),
        ("sanity_scope", "finite-MDP sanity scope"),
    ],
)
def test_artifact_mutations_fail_closed_before_any_output_is_published(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    summary = _load_summary()
    study = tmp_path / mutation
    _make_complete_study(study)
    seed_dir = study / "seed_00000"

    if mutation == "missing_marker":
        (study / "COMPLETE").unlink()
    elif mutation in {"399_rows", "401_rows", "duplicate_method", "nonintegral_seed"}:
        records = pd.read_csv(seed_dir / "records.csv")
        if mutation == "399_rows":
            records = records.iloc[:-1]
        elif mutation == "401_rows":
            records = pd.concat([records, records.iloc[[0]]], ignore_index=True)
        elif mutation == "duplicate_method":
            records.loc[3, "method"] = records.loc[2, "method"]
        else:
            records["seed"] = 0.5
        records.to_csv(seed_dir / "records.csv", index=False)
    elif mutation == "wrong_vector_length":
        records = pd.read_csv(seed_dir / "records.csv")
        records.loc[0, "final_coverage"] = json.dumps([0.95] * 11)
        records.to_csv(seed_dir / "records.csv", index=False)
    elif mutation == "truncated_npz_member":
        member = io.BytesIO()
        np.save(member, np.arange(4))
        with zipfile.ZipFile(seed_dir / "surfaces.npz", "w") as archive:
            archive.writestr("truncated.npy", member.getvalue()[:-8])
    elif mutation == "stream_collision":
        other = study / "seed_00001"
        records = pd.read_csv(other / "records.csv")
        records.loc[records["scenario"] == "standard", "tuning_seed"] = 10_000
        records.to_csv(other / "records.csv", index=False)
        metadata = json.loads((other / "metadata.json").read_text())
        metadata["diagnostics"]["standard"]["tuning_seed"] = 10_000
        (other / "metadata.json").write_text(json.dumps(metadata))
    elif mutation == "common_grid_disagreement":
        metadata = json.loads((seed_dir / "metadata.json").read_text())
        metadata["diagnostics"]["standard"]["profiled_common_grid"][
            "final_coverage"
        ][0] = 0.5
        (seed_dir / "metadata.json").write_text(json.dumps(metadata))
    elif mutation == "sanity_in_gate":
        sanity_path = study / "finite_mdp_sanity.json"
        sanity = json.loads(sanity_path.read_text())
        sanity["non_gating"] = False
        sanity_path.write_text(json.dumps(sanity))
    elif mutation == "partial_seed_path":
        (study / ".seed_00000-partial").mkdir()
    elif mutation == "status_seed_type":
        status_path = study / "study_status.json"
        status = json.loads(status_path.read_text())
        status["expected_seeds"][1] = 1.0
        status_path.write_text(json.dumps(status))
    elif mutation == "primary_index_shape":
        metadata_path = seed_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["diagnostics"]["standard"]["profiled"]["selected_indices"] = [1, 2]
        metadata_path.write_text(json.dumps(metadata))
    elif mutation == "missing_common_surface":
        surface_path = seed_dir / "surfaces.npz"
        with np.load(surface_path, allow_pickle=False) as archive:
            surfaces = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name != "standard_profiled_common_grid_candidate_coverage"
            }
        np.savez_compressed(surface_path, **surfaces)
    elif mutation == "sanity_undercoverage":
        sanity_path = study / "finite_mdp_sanity.json"
        sanity = json.loads(sanity_path.read_text())
        sanity["exact"]["coverage"][0] = 0.89
        sanity_path.write_text(json.dumps(sanity))
    elif mutation in {"config_synthetic", "config_policy", "config_profile"}:
        config_path = study / "config.yaml"
        config = yaml.safe_load(config_path.read_text())
        if mutation == "config_synthetic":
            config["synthetic"]["feedback_strength"] = 0.9
        elif mutation == "config_policy":
            config["policy"]["temperature"] = 0.9
        else:
            config["profile"]["refinement_strength"] = 0.4
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    elif mutation in {
        "primary_grid_shape",
        "profiled_index_disagreement",
        "greedy_index_disagreement",
        "candidate_nan",
    }:
        if mutation in {"profiled_index_disagreement", "greedy_index_disagreement"}:
            metadata_path = seed_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            key = "profiled" if mutation.startswith("profiled") else "greedy"
            metadata["diagnostics"]["standard"][key]["selected_indices"] = (
                [2] if key == "profiled" else [2] * 12
            )
            metadata_path.write_text(json.dumps(metadata))
        else:
            surface_path = seed_dir / "surfaces.npz"
            with np.load(surface_path, allow_pickle=False) as archive:
                surfaces = {name: np.asarray(archive[name]) for name in archive.files}
            if mutation == "primary_grid_shape":
                surfaces["standard_greedy_stage_grids"] = surfaces[
                    "standard_greedy_stage_grids"
                ][:, :-1]
            else:
                surfaces["standard_profiled_candidate_coverage"][0, 0] = np.nan
            np.savez_compressed(surface_path, **surfaces)
    elif mutation in {"sanity_device", "sanity_config_hash", "sanity_scope"}:
        sanity_path = study / "finite_mdp_sanity.json"
        sanity = json.loads(sanity_path.read_text())
        if mutation == "sanity_device":
            sanity["device"] = "cpu"
        elif mutation == "sanity_config_hash":
            sanity["config_sha256"] = "c" * 64
        else:
            sanity["scope"] = "This establishes globally optimal performance."
        sanity_path.write_text(json.dumps(sanity))
    else:
        metadata_path = study / "study_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["source_tree_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match=match):
        summary.publish_phase0_summary(study)

    assert all(not (study / name).exists() for name in OUTPUT_NAMES)


def test_outputs_are_atomic_complete_deterministic_and_figure_text_is_editable(
    tmp_path: Path,
) -> None:
    summary = _load_summary()
    study = tmp_path / "published"
    _make_complete_study(study)

    first = summary.publish_phase0_summary(study)
    stable_first = {name: (study / name).read_bytes() for name in OUTPUT_NAMES}
    second = summary.publish_phase0_summary(study)

    assert first["decision"] == second["decision"] == "GO"
    assert all((study / name).is_file() for name in OUTPUT_NAMES)
    assert all((study / name).stat().st_size > 0 for name in OUTPUT_NAMES)
    assert stable_first == {
        name: (study / name).read_bytes() for name in stable_first
    }
    decision = json.loads((study / "phase0_decision.json").read_text())
    assert decision["schema_version"] == 1
    assert decision["decision"] == "GO"
    assert len(decision["gates"]) == 7
    manifest = json.loads((study / MANIFEST_NAME).read_text())
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "complete"
    assert set(manifest["files"]) == set(PAYLOAD_NAMES)
    for name in PAYLOAD_NAMES:
        payload = (study / name).read_bytes()
        assert manifest["files"][name]["sha256"] == hashlib.sha256(payload).hexdigest()
        assert manifest["files"][name]["bytes"] == len(payload)
    source = pd.read_csv(study / "phase0_summary.csv")
    assert {
        "row_type",
        "analysis_role",
        "scenario",
        "method",
        "comparator",
        "stage",
        "metric",
        "estimate",
        "lower",
        "upper",
        "n_total",
        "n_selected",
        "n_paired",
        "conditioning",
        "interval_method",
        "threshold",
        "operator",
        "passed",
    } <= set(source.columns)
    markdown = (study / "phase0_summary.md").read_text()
    assert "conditional on successful selection" in markdown
    assert "oracle is not a deployable method" in markdown
    assert "greedy search is not globally optimal" in markdown
    assert "fixed T=12" in markdown
    assert "GO does not establish state of the art" in markdown
    assert "## Selection and endpoint audit" in markdown
    assert "Greedy / common-grid" in markdown
    assert "Common-grid / exact-current" in markdown
    assert "| standard | Common-grid | 100 | 0.000000 |" in markdown
    assert (
        "| standard | 100 | 0.945000 / 0.945000 | "
        "1.0316 (1.0316, 1.0316); n=100 | "
        "0.9500 (0.9500, 0.9500); n=100 |"
    ) in markdown
    assert "Finite-MDP true grid gap: 0.000000" in markdown
    assert "SOTA" not in markdown
    svg = (study / "phase0_radius_and_coverage.svg").read_text()
    assert "<text" in svg
    assert "G / Current (n=100)" in svg
    assert all(gate_id in svg for gate_id in GATE_IDS)
    assert svg.count("PASS") >= 7
    assert (study / "phase0_radius_and_coverage.pdf").read_bytes().startswith(b"%PDF")
    assert (study / "phase0_radius_and_coverage.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    "failed_gate,value",
    [("tail_selection_count", 94), ("standard_micro_ratio", 1.03)],
)
def test_figure_gate_strip_names_each_single_failed_gate(
    tmp_path: Path,
    failed_gate: str,
    value: float,
) -> None:
    summary = _load_summary()
    study = tmp_path / failed_gate
    _make_complete_study(study)
    analysis = summary.load_validate_and_analyze(study)
    for gate in analysis["gates"]:
        if gate["id"] == failed_gate:
            gate["value"] = value
            gate["passed"] = False
    analysis["decision"] = "NO_GO"

    rows = summary._gate_strip_rows(analysis)
    failed = next(row for row in rows if row["id"] == failed_gate)
    assert len(rows) == 7
    assert failed["status"] == "FAIL"
    assert str(value) in failed["detail"]

    output = tmp_path / f"figure-{failed_gate}"
    summary._render_figure(analysis, output)
    svg = output.with_suffix(".svg").read_text()
    assert all(gate_id in svg for gate_id in GATE_IDS)
    assert failed_gate in svg
    assert "FAIL" in svg


def test_bundle_replace_failure_rolls_back_all_payloads_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _load_summary()
    study = tmp_path / "rollback"
    _make_complete_study(study)
    summary.publish_phase0_summary(study)
    old_bundle = {name: (study / name).read_bytes() for name in OUTPUT_NAMES}

    sanity_path = study / "finite_mdp_sanity.json"
    sanity = json.loads(sanity_path.read_text())
    sanity["greedy"]["normalized_width_by_stage"] = [4.1] * 4
    sanity["greedy"]["mean_normalized_width"] = 4.1
    sanity["true_finite_grid_gap"] = 0.1
    sanity_path.write_text(json.dumps(sanity))

    real_replace = summary.os.replace
    replacements = 0

    def fail_third_payload(source: str | Path, destination: str | Path) -> None:
        nonlocal replacements
        source_path, destination_path = Path(source), Path(destination)
        if (
            source_path.parent.name.startswith(".phase0-summary-")
            and destination_path.parent == study
            and destination_path.name in PAYLOAD_NAMES
        ):
            replacements += 1
            if replacements == 3:
                raise OSError("injected bundle replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(summary.os, "replace", fail_third_payload)
    with pytest.raises(OSError, match="injected bundle replace failure"):
        summary.publish_phase0_summary(study)

    assert {name: (study / name).read_bytes() for name in OUTPUT_NAMES} == old_bundle
    manifest = json.loads((study / MANIFEST_NAME).read_text())
    for name in PAYLOAD_NAMES:
        assert manifest["files"][name]["sha256"] == hashlib.sha256(
            (study / name).read_bytes()
        ).hexdigest()


def test_manifest_backup_failure_preserves_the_valid_old_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _load_summary()
    study = tmp_path / "manifest-backup-failure"
    _make_complete_study(study)
    summary.publish_phase0_summary(study)
    old_bundle = {name: (study / name).read_bytes() for name in OUTPUT_NAMES}
    real_replace = summary.os.replace

    def fail_manifest_backup(source: str | Path, destination: str | Path) -> None:
        if Path(source) == study / MANIFEST_NAME:
            raise OSError("injected manifest backup failure")
        real_replace(source, destination)

    monkeypatch.setattr(summary.os, "replace", fail_manifest_backup)
    with pytest.raises(OSError, match="injected manifest backup failure"):
        summary.publish_phase0_summary(study)

    assert {name: (study / name).read_bytes() for name in OUTPUT_NAMES} == old_bundle
    manifest = json.loads((study / MANIFEST_NAME).read_text())
    for name in PAYLOAD_NAMES:
        assert manifest["files"][name]["sha256"] == hashlib.sha256(
            (study / name).read_bytes()
        ).hexdigest()


def test_cli_runs_the_locked_100_seed_entry_and_fails_closed_when_incomplete(
    tmp_path: Path,
) -> None:
    study = tmp_path / "cli"
    _make_complete_study(study)
    script = ROOT / "scripts" / "summarize_phase0_oracle.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--input-dir", str(study)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "GO"
    assert all((study / name).is_file() for name in OUTPUT_NAMES)

    for name in OUTPUT_NAMES:
        (study / name).unlink()
    (study / "COMPLETE").unlink()
    failed = subprocess.run(
        [sys.executable, str(script), "--input-dir", str(study)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "COMPLETE" in failed.stderr
    assert all(not (study / name).exists() for name in OUTPUT_NAMES)
