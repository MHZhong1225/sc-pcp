from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("standard", "tail_shift")
SEEDS = tuple(range(10_000, 10_040))
PAYLOAD_NAMES = {
    "phase0c_summary.csv",
    "phase0c_decision.json",
    "phase0c_summary.md",
    "phase0c_joint_search.pdf",
    "phase0c_joint_search.svg",
    "phase0c_joint_search.png",
}


def _load_summary():
    path = ROOT / "scripts" / "summarize_phase0c_joint_search.py"
    spec = importlib.util.spec_from_file_location("summarize_phase0c_joint_search", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runner():
    path = ROOT / "scripts" / "run_phase0c_joint_search.py"
    spec = importlib.util.spec_from_file_location("run_phase0c_joint_search", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runner_test_support():
    path = ROOT / "tests" / "per_step" / "test_phase0c_runner.py"
    spec = importlib.util.spec_from_file_location("phase0c_runner_test_support", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_study_fact(root: Path, relative: str) -> None:
    manifest_path = root / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = root / relative
    manifest["files"][relative] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


def _rewrite_npz(path: Path, updates: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays.update(updates)
    np.savez_compressed(path, **arrays)


def _rewrite_joint_width_ratios(root: Path, *, b: float, b2: float) -> None:
    for seed in SEEDS:
        seed_dir = root / f"seed_{seed:05d}"
        records_path = seed_dir / "records.csv"
        records = pd.read_csv(records_path)
        surface_updates: dict[str, np.ndarray] = {}
        for scenario in SCENARIOS:
            current_mask = records["scenario"].eq(scenario) & records["method_id"].eq(
                "current_profiled"
            )
            current_width = np.asarray(
                json.loads(records.loc[current_mask, "final_stage_width_json"].iloc[0]),
                dtype=np.float32,
            )
            for method, ratio in (("joint_B", b), ("joint_2B", b2)):
                mask = records["scenario"].eq(scenario) & records["method_id"].eq(method)
                stage_width = np.asarray(current_width * np.float32(ratio), dtype=np.float32)
                compact = json.dumps(stage_width.tolist(), separators=(",", ":"))
                records.loc[mask, "final_stage_width_json"] = compact
                records.loc[mask, "micro_normalized_width"] = float(
                    stage_width.mean(dtype=np.float32)
                )
                records.loc[mask, "patient_normalized_width"] = float(
                    stage_width.mean(dtype=np.float32)
                )
                surface_updates[f"{scenario}_{method}_final_stage_width"] = stage_width
        records.to_csv(records_path, index=False)
        _rewrite_npz(seed_dir / "surfaces.npz", surface_updates)
        _refresh_study_fact(root, f"seed_{seed:05d}/records.csv")
        _refresh_study_fact(root, f"seed_{seed:05d}/surfaces.npz")


@pytest.fixture(scope="module")
def complete_initial_root(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    runner = _load_runner()
    support = _load_runner_test_support()
    root = tmp_path_factory.mktemp("phase0c-summary") / "initial"
    with pytest.MonkeyPatch.context() as monkeypatch:
        config = support._run_initial_fixture(
            runner, monkeypatch, root, seeds=SEEDS
        )
    return SimpleNamespace(root=root, runner=runner, support=support, config=config)


def test_registered_paired_ratio_and_budget_gain_literals() -> None:
    summary = _load_summary()
    current = np.array([1.0, 1.0, 1.0, 1.0])
    b = np.array([0.95, 0.96, 0.94, 0.93])
    b2 = np.array([0.94, 0.95, 0.93, 0.92])

    assert summary.paired_geometric_ratio(b, current) == pytest.approx(
        0.9449338571562925
    )
    assert summary.paired_geometric_ratio(b2, current) == pytest.approx(
        0.9349331496314753
    )
    assert summary.relative_budget_gain(
        0.9449338571562925, 0.9349331496314753
    ) == pytest.approx(0.01058350005037767)


def test_paired_bootstrap_uses_the_registered_shared_index_matrix() -> None:
    summary = _load_summary()
    current = np.ones(4)
    b = np.array([0.95, 0.96, 0.94, 0.93])
    b2 = np.array([0.94, 0.95, 0.93, 0.92])
    indices = np.random.default_rng(2_718_281).integers(0, 4, size=(10_000, 4))

    got = summary.paired_ratio_summary(b, current, bootstrap_indices=indices)
    got_2b = summary.paired_ratio_summary(b2, current, bootstrap_indices=indices)
    delta = summary.paired_delta_summary(b, b2, current, bootstrap_indices=indices)

    assert got == pytest.approx(
        {
            "n_pairs": 4,
            "point_ratio": 0.9449338571562925,
            "bootstrap_ci_lower": 0.9349601758979881,
            "bootstrap_ci_upper": 0.9549869109050657,
        }
    )
    assert got_2b == pytest.approx(
        {
            "n_pairs": 4,
            "point_ratio": 0.9349331496314753,
            "bootstrap_ci_lower": 0.9249597484240125,
            "bootstrap_ci_upper": 0.9449867723941959,
        }
    )
    assert delta == pytest.approx(
        {
            "n_pairs": 4,
            "point_gain": 0.01058350005037767,
            "bootstrap_ci_lower": 0.010471492746840262,
            "bootstrap_ci_upper": 0.010696099932139451,
        }
    )


def test_coverage_summary_locks_registered_lcb_and_gate() -> None:
    summary = _load_summary()
    coverage = np.full((40, 2, 12), 0.95)
    coverage[:, 0, 0] = np.array([0.890 + 0.001 * index for index in range(40)])

    got = summary.coverage_summary(coverage)

    assert got["minimum_simultaneous_lcb"] == pytest.approx(0.9038732857531458)
    assert got["coverage_valid"] is True

    coverage[:, 0, 0] -= 0.004
    shifted = summary.coverage_summary(coverage)
    assert shifted["minimum_simultaneous_lcb"] == pytest.approx(
        0.8998732857531458
    )
    assert shifted["coverage_valid"] is False


def test_coverage_summary_keeps_all_four_minima_distinct() -> None:
    summary = _load_summary()
    coverage = np.full((40, 2, 12), 0.95)
    coverage[:, 0, 0] = np.array([0.890 + 0.001 * index for index in range(40)])
    coverage[:, 1, 1] = np.array([0.880, *([0.960] * 39)])

    got = summary.coverage_summary(coverage)

    assert got["minimum_stage_seed_mean"] == pytest.approx(0.9095)
    assert got["minimum_simultaneous_lcb"] == pytest.approx(0.9038732857531458)
    assert got["mean_seedwise_stage_minimum"] == pytest.approx(0.90925)
    assert got["raw_seed_stage_minimum"] == pytest.approx(0.88)


@pytest.mark.parametrize(
    ("valid", "r_2b", "delta_b", "want"),
    [
        (True, 0.92, -0.10, "PROMISING_ORACLE_DIAGNOSTIC"),
        (True, 0.926, 0.004301075268817208, "STOP_SCALAR_SATURATED"),
        (True, 0.921, 0.005, "EXTENSION_8SP_REQUIRED"),
        (False, 0.80, 0.20, "STOP_SCALAR_UNAVAILABLE"),
    ],
)
def test_initial_decision_uses_exact_registered_boundaries(
    valid: bool, r_2b: float, delta_b: float, want: str
) -> None:
    summary = _load_summary()
    assert summary.decide_initial(valid=valid, r_2b=r_2b, delta_b=delta_b) == want


@pytest.mark.parametrize(
    ("valid", "r_8sp", "want"),
    [
        (True, 0.92, "PROMISING_ORACLE_DIAGNOSTIC"),
        (True, 0.9200001, "STOP_SCALAR_INSUFFICIENT"),
        (False, 0.80, "STOP_SCALAR_UNAVAILABLE"),
    ],
)
def test_extension_decision_uses_exact_registered_boundary(
    valid: bool, r_8sp: float, want: str
) -> None:
    summary = _load_summary()
    assert summary.decide_extension(valid=valid, r_8sp=r_8sp) == want


def test_complete_initial_root_is_deeply_loaded_with_all_seed_denominators(
    complete_initial_root: SimpleNamespace,
) -> None:
    summary = _load_summary()

    analysis = summary.load_validate_analyze(complete_initial_root.root)

    assert analysis["analysis_phase"] == "initial"
    assert analysis["ordered_seeds"] == list(SEEDS)
    assert analysis["decision"] == "STOP_SCALAR_SATURATED"
    assert analysis["validity"] == {
        "initial_valid": True,
        "current_b_pairs": 80,
        "current_2b_pairs": 80,
        "required_pairs_each": 80,
        "failed_primitives": [],
    }
    assert analysis["coverage"]["joint_B"]["coverage_valid"] is True
    assert analysis["coverage"]["joint_2B"]["coverage_valid"] is True
    assert analysis["ratios"]["joint_B"]["n_pairs"] == 40
    assert analysis["ratios"]["joint_2B"]["n_pairs"] == 40
    assert analysis["ratios"]["joint_B"]["point_ratio"] == pytest.approx(
        0.9861110846201578
    )
    assert analysis["ratios"]["joint_2B"]["point_ratio"] == pytest.approx(
        1.138888915379842
    )
    assert analysis["delta_b"]["point_gain"] == pytest.approx(-0.15492963535495904)
    assert analysis["extension_eligibility"] == {
        "all_eligible": True,
        "eligible_scenario_seed_count": 80,
        "required_scenario_seed_count": 80,
        "canonical_state_hash_count": 240,
        "state_hash_manifest_sha256": analysis["extension_eligibility"][
            "state_hash_manifest_sha256"
        ],
    }
    assert len(analysis["extension_eligibility"]["state_hash_manifest_sha256"]) == 64
    assert [item["seed"] for item in analysis["convergence_gain_by_seed"]] == list(
        SEEDS
    )
    assert analysis["convergence_gain_by_seed"][0]["value"] == pytest.approx(
        -0.15492963535495918
    )
    standard_audit = analysis["audit"]["checkpoints"]["joint_2B"][
        "by_scenario"
    ]["standard"]
    assert standard_audit["available_seed_count"] == 40
    assert standard_audit["winner_denominator"] == 40
    assert standard_audit["endpoint_stage_denominator"] == 480
    assert analysis["audit"]["runner"]["initial"]["n_seed_runs"] == 40
    assert analysis["audit"]["runner"]["extension"] is None
    assert len(analysis["_source_rows"]) == 320


def test_initial_summary_publishes_exact_authorization_contract_and_manifest(
    complete_initial_root: SimpleNamespace,
) -> None:
    summary = _load_summary()
    analysis = summary.load_validate_analyze(complete_initial_root.root)

    output = summary.publish_summary(analysis)

    assert output == complete_initial_root.root / "checkpoint_analysis"
    assert {path.name for path in output.iterdir()} == PAYLOAD_NAMES | {
        "phase0c_summary_manifest.json"
    }
    decision = json.loads((output / "phase0c_decision.json").read_text())
    assert set(decision) == {
        "protocol",
        "analysis_phase",
        "decision",
        "parent_study_manifest_sha256",
        "ordered_seeds",
        "source_tree_sha256",
        "experiment_tree_sha256",
        "config_sha256",
        "extension_eligibility",
    }
    assert "checkpoint_decision_sha256" not in decision
    assert decision["decision"] == "STOP_SCALAR_SATURATED"
    manifest = json.loads((output / "phase0c_summary_manifest.json").read_text())
    assert set(manifest) == {
        "protocol",
        "status",
        "analysis_phase",
        "decision",
        "parent_study_manifest_sha256",
        "files",
    }
    assert set(manifest["files"]) == PAYLOAD_NAMES
    assert manifest["files"]["phase0c_decision.json"] == {
        "bytes": (output / "phase0c_decision.json").stat().st_size,
        "sha256": _sha256(output / "phase0c_decision.json"),
    }
    summary.validate_summary_bundle(output)


def _run_initial_with_result_factory(
    runner: object,
    support: object,
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    result_factory: object,
):
    base = support.ExperimentConfig.from_yaml(support.CONFIG_PATH)
    config = base.with_overrides(seeds=SEEDS, devices=("cpu",), output_dir=output)
    smoke = output.parent / f"{output.name}-smoke.json"
    support._write_smoke_manifest(runner, smoke, config)
    support._InlineExecutor.instances = []
    monkeypatch.setattr(runner, "ProcessPoolExecutor", support._InlineExecutor)
    monkeypatch.setattr(runner, "run_phase0c_seed", result_factory)
    runner.run_config(
        config,
        output,
        mode="initial",
        workers_per_device=1,
        candidate_chunk_size=16,
        resume=False,
        smoke_manifest=smoke,
    )
    return config


def _no_feasible_result(support: object, seed: int, device: str):
    result = support._initial_result(seed, device)
    records = [dict(row) for row in result.records]
    surfaces = dict(result.surfaces)
    diagnostics = json.loads(json.dumps(result.diagnostics))
    for row in records:
        row.update(
            {
                "selection_status": "NO_FEASIBLE_START",
                "selection_available": False,
                "tuning_joint_feasible": False,
                "failure_reason": "NO_FEASIBLE_START",
                "chosen_initialization": "",
                "selected_endpoint_stage_count": 0,
                "selected_stage_grid_indices_json": "[]",
                "q_by_time_json": "[]",
                "tuning_coverage_json": "[]",
                "tuning_stage_width_json": "[]",
                "tuning_micro_width": float("nan"),
                "final_coverage_json": "[]",
                "final_wilson_lcb_json": "[]",
                "final_stage_width_json": "[]",
                "micro_normalized_width": float("nan"),
                "patient_normalized_width": float("nan"),
                "n_evaluation_rollouts": 0,
                "schedule_evaluations": 0,
                "committed_updates": 0,
                "converged_at_pair": None,
            }
        )
        prefix = f"{row['scenario']}_{row['method_id']}"
        for suffix in (
            "schedule",
            "tuning_coverage",
            "tuning_stage_width",
            "final_coverage",
            "final_wilson_lcb",
            "final_stage_width",
        ):
            surfaces[f"{prefix}_{suffix}"] = support.torch.empty(0)
    for scenario in SCENARIOS:
        for start_name in support.START_NAMES:
            for suffix in (
                "radii",
                "stage_grid_indices",
                "coverage",
                "normalized_width",
                "completed_sweep_pairs",
                "converged_at_pair",
            ):
                surfaces.pop(f"{scenario}_pair4_{start_name}_{suffix}")
        surfaces[f"{scenario}_active_start_names"] = support.torch.empty(
            0, dtype=support.torch.int64
        )
        surfaces[f"{scenario}_extension_eligible"] = support.torch.tensor(False)
        diagnostics[scenario].update(
            {
                "active_start_names": [],
                "extension_eligible": False,
                "pair4_state_sha256": [],
                "greedy_partial_indices": [],
                "search_status": "NO_FEASIBLE_START",
                "checkpoints": {},
            }
        )
    return support.SeedResult(seed, device, records, surfaces, diagnostics)


def test_exact_schema_wall_cap_is_scientific_unavailability_and_still_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _load_summary()
    runner = _load_runner()
    support = _load_runner_test_support()
    root = tmp_path / "scientifically-unavailable"

    def result_factory(_config, *, seed: int, device: str, **_kwargs):
        if seed == SEEDS[-1]:
            return support._pair2_only_result(seed, device)
        return support._initial_result(seed, device)

    _run_initial_with_result_factory(
        runner, support, monkeypatch, root, result_factory
    )

    analysis = summary.load_validate_analyze(root)
    assert analysis["decision"] == "STOP_SCALAR_UNAVAILABLE"
    assert analysis["validity"]["current_2b_pairs"] == 78
    assert "current_2b_pairs_78_of_80" in analysis["validity"]["failed_primitives"]
    output = summary.publish_summary(analysis)
    assert output == root / "checkpoint_analysis"
    assert (output / "phase0c_summary_manifest.json").is_file()


def test_exact_schema_no_feasible_start_is_scientific_unavailability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _load_summary()
    runner = _load_runner()
    support = _load_runner_test_support()
    root = tmp_path / "no-feasible"

    def result_factory(_config, *, seed: int, device: str, **_kwargs):
        if seed == SEEDS[-1]:
            return _no_feasible_result(support, seed, device)
        return support._initial_result(seed, device)

    _run_initial_with_result_factory(
        runner, support, monkeypatch, root, result_factory
    )

    analysis = summary.load_validate_analyze(root)
    assert analysis["decision"] == "STOP_SCALAR_UNAVAILABLE"
    assert analysis["validity"]["current_b_pairs"] == 78
    assert analysis["validity"]["current_2b_pairs"] == 78
    assert summary.publish_summary(analysis) == root / "checkpoint_analysis"


def test_numeric_extension_branch_without_80_of_80_eligibility_stops_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _load_summary()
    runner = _load_runner()
    support = _load_runner_test_support()
    root = tmp_path / "partial-active"
    support._run_initial_fixture(
        runner,
        monkeypatch,
        root,
        seeds=SEEDS,
        partial_seed=SEEDS[-1],
    )
    _rewrite_joint_width_ratios(root, b=0.95, b2=0.94)

    analysis = summary.load_validate_analyze(root)

    assert analysis["numeric_initial_decision"] == "EXTENSION_8SP_REQUIRED"
    assert analysis["decision"] == "STOP_SCALAR_UNAVAILABLE"
    assert analysis["extension_eligibility"]["eligible_scenario_seed_count"] == 78
    assert "extension_eligibility_78_of_80" in analysis["validity"][
        "failed_primitives"
    ]


def test_fresh_coverage_failure_is_scientific_stop_not_structural_corruption(
    complete_initial_root: SimpleNamespace,
    tmp_path: Path,
) -> None:
    summary = _load_summary()
    root = tmp_path / "coverage-invalid"
    shutil.copytree(complete_initial_root.root, root)
    checkpoint = root / "checkpoint_analysis"
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    coverage = np.full(12, 0.89, dtype=np.float32)
    compact = json.dumps(coverage.tolist(), separators=(",", ":"))
    for seed in SEEDS:
        records_path = root / f"seed_{seed:05d}" / "records.csv"
        records = pd.read_csv(records_path)
        selected = records["scenario"].eq("standard") & records["method_id"].eq(
            "joint_B"
        )
        records.loc[selected, "final_coverage_json"] = compact
        records.to_csv(records_path, index=False)
        surfaces_path = root / f"seed_{seed:05d}" / "surfaces.npz"
        _rewrite_npz(
            surfaces_path, {"standard_joint_B_final_coverage": coverage}
        )
        _refresh_study_fact(root, f"seed_{seed:05d}/records.csv")
        _refresh_study_fact(root, f"seed_{seed:05d}/surfaces.npz")

    analysis = summary.load_validate_analyze(root)

    assert analysis["coverage"]["joint_B"]["coverage_valid"] is False
    assert analysis["decision"] == "STOP_SCALAR_UNAVAILABLE"
    assert "joint_B_coverage_invalid" in analysis["validity"]["failed_primitives"]
    assert summary.publish_summary(analysis) == checkpoint


def _mutate_csv(root: Path, mutate) -> None:
    relative = "seed_10000/records.csv"
    path = root / relative
    records = pd.read_csv(path)
    mutate(records)
    records.to_csv(path, index=False)
    _refresh_study_fact(root, relative)


def _mutate_metadata(root: Path, mutate) -> None:
    relative = "seed_10000/metadata.json"
    path = root / relative
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    _refresh_study_fact(root, relative)


def test_structural_mutations_raise_before_default_outputs_appear(
    complete_initial_root: SimpleNamespace,
    tmp_path: Path,
) -> None:
    summary = _load_summary()
    support = complete_initial_root.support

    def tuning_below_target(root: Path) -> None:
        support._set_joint_tuning_coverage(root / "seed_10000", 0.899999)
        _refresh_study_fact(root, "seed_10000/records.csv")
        _refresh_study_fact(root, "seed_10000/surfaces.npz")

    def mismatched_schedule(root: Path) -> None:
        path = root / "seed_10000/surfaces.npz"
        with np.load(path, allow_pickle=False) as archive:
            schedule = np.array(archive["standard_joint_B_schedule"], copy=True)
        schedule[0] += np.float32(0.01)
        _rewrite_npz(path, {"standard_joint_B_schedule": schedule})
        _refresh_study_fact(root, "seed_10000/surfaces.npz")

    def config_hash(root: Path) -> None:
        path = root / "config.yaml"
        mutated = path.read_text().replace("horizon: 12", "horizon: 13")
        assert mutated != path.read_text()
        path.write_text(mutated)
        _refresh_study_fact(root, "config.yaml")

    mutations = {
        "39-physical-seeds": lambda root: shutil.rmtree(root / "seed_10039"),
        "tuning-below-target": tuning_below_target,
        "zero-width": lambda root: _mutate_csv(
            root, lambda rows: rows.__setitem__("micro_normalized_width", 0.0)
        ),
        "negative-width": lambda root: _mutate_csv(
            root, lambda rows: rows.__setitem__("micro_normalized_width", -1.0)
        ),
        "nan-width": lambda root: _mutate_csv(
            root, lambda rows: rows.__setitem__("micro_normalized_width", np.nan)
        ),
        "wrong-scenario": lambda root: _mutate_csv(
            root, lambda rows: rows.__setitem__("scenario", "wrong")
        ),
        "wrong-method": lambda root: _mutate_csv(
            root, lambda rows: rows.__setitem__("method_id", "wrong")
        ),
        "wrong-stream": lambda root: _mutate_csv(
            root,
            lambda rows: rows.__setitem__(
                "tuning_stream_id", rows["tuning_stream_id"] + 1
            ),
        ),
        "duplicate-stream": lambda root: _mutate_csv(
            root,
            lambda rows: rows.__setitem__(
                "tuning_stream_id",
                pd.read_csv(root / "seed_10001" / "records.csv")[
                    "tuning_stream_id"
                ],
            ),
        ),
        "mismatched-schedule": mismatched_schedule,
        "source-hash": lambda root: _mutate_metadata(
            root, lambda payload: payload.__setitem__("source_tree_sha256", "0" * 64)
        ),
        "seed-hash": lambda root: _mutate_metadata(
            root,
            lambda payload: payload["diagnostics"]["runner_provenance"].__setitem__(
                "execution_sha256", "0" * 64
            ),
        ),
        "config-hash": config_hash,
        "root-completion": lambda root: (root / "COMPLETE").write_text("wrong\n"),
    }

    for name, mutate in mutations.items():
        root = tmp_path / name
        shutil.copytree(complete_initial_root.root, root)
        checkpoint = root / "checkpoint_analysis"
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        mutate(root)
        with pytest.raises(RuntimeError):
            summary.main(["--input-dir", str(root)])
        assert not checkpoint.exists(), name


@pytest.mark.parametrize("fail_at", [1, 2])
def test_summary_directory_swap_rolls_back_prior_valid_bundle_byte_for_byte(
    complete_initial_root: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: int,
) -> None:
    summary = _load_summary()
    analysis = summary.load_validate_analyze(complete_initial_root.root)
    output = tmp_path / "atomic-summary"
    summary.publish_summary(analysis, output)
    before = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    original_replace = summary._replace_path
    calls = 0

    def fail_selected_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("injected install fault")
        original_replace(source, destination)

    monkeypatch.setattr(summary, "_replace_path", fail_selected_replace)
    with pytest.raises(RuntimeError, match="publish"):
        summary.publish_summary(analysis, output)

    after = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    assert after == before
    summary.validate_summary_bundle(output)


def test_summary_bundle_validator_rejects_extra_decision_fields_even_when_rehashed(
    complete_initial_root: SimpleNamespace,
    tmp_path: Path,
) -> None:
    summary = _load_summary()
    analysis = summary.load_validate_analyze(complete_initial_root.root)
    output = summary.publish_summary(analysis, tmp_path / "strict-summary")
    decision_path = output / "phase0c_decision.json"
    decision = json.loads(decision_path.read_text())
    decision["checkpoint_decision_sha256"] = "0" * 64
    decision_path.write_text(json.dumps(decision, sort_keys=True, indent=2) + "\n")
    manifest_path = output / "phase0c_summary_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["phase0c_decision.json"] = {
        "bytes": decision_path.stat().st_size,
        "sha256": _sha256(decision_path),
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="decision"):
        summary.validate_summary_bundle(output)


def test_two_independent_cli_processes_publish_byte_identical_text_artifacts(
    complete_initial_root: SimpleNamespace,
    tmp_path: Path,
) -> None:
    outputs = (tmp_path / "first", tmp_path / "second")
    for output in outputs:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "summarize_phase0c_joint_search.py"),
                "--input-dir",
                str(complete_initial_root.root),
                "--output-dir",
                str(output),
            ],
            check=True,
            cwd=ROOT,
            env={**os.environ, "MPLCONFIGDIR": str(tmp_path / f"mpl-{output.name}")},
        )

    for name in (
        "phase0c_decision.json",
        "phase0c_summary.csv",
        "phase0c_summary.md",
        "phase0c_joint_search.svg",
    ):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()


def test_authorized_extension_is_validated_and_publishes_only_final_analysis(
    complete_initial_root: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _load_summary()
    runner = complete_initial_root.runner
    support = complete_initial_root.support
    parent = tmp_path / "parent"
    extension = tmp_path / "extension"
    shutil.copytree(complete_initial_root.root, parent)
    checkpoint = parent / "checkpoint_analysis"
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    _rewrite_joint_width_ratios(parent, b=0.95, b2=0.94)
    initial_analysis = summary.load_validate_analyze(parent)
    assert initial_analysis["decision"] == "EXTENSION_8SP_REQUIRED"
    assert summary.publish_summary(initial_analysis) == checkpoint

    support._InlineExecutor.instances = []
    monkeypatch.setattr(runner, "ProcessPoolExecutor", support._InlineExecutor)
    monkeypatch.setattr(
        runner,
        "run_phase0c_extension_seed",
        lambda _config, *, seed, device, **_kwargs: support._extension_result(
            seed, device
        ),
    )
    extension_config = complete_initial_root.config.with_overrides(output_dir=extension)
    runner.run_config(
        extension_config,
        extension,
        mode="extension-8sp",
        workers_per_device=1,
        candidate_chunk_size=16,
        resume=False,
        parent_dir=parent,
        decision_json=checkpoint / "phase0c_decision.json",
    )

    final = summary.load_validate_analyze(parent, extension)

    assert final["analysis_phase"] == "final"
    assert final["ratios"]["joint_8SP"]["n_pairs"] == 40
    assert final["coverage"]["joint_8SP"]["coverage_valid"] is True
    assert final["decision"] == "STOP_SCALAR_INSUFFICIENT"
    assert final["audit"]["runner"]["extension"]["n_seed_runs"] == 40
    output = summary.publish_summary(final)
    assert output == parent / "final_analysis"
    assert checkpoint.is_dir()
    assert summary.validate_summary_bundle(output) == output

    wrong_parent = tmp_path / "wrong-parent"
    shutil.copytree(parent, wrong_parent)
    final_output = wrong_parent / "final_analysis"
    if final_output.exists():
        shutil.rmtree(final_output)
    with pytest.raises(RuntimeError, match="parent"):
        summary.main(
            [
                "--input-dir",
                str(wrong_parent),
                "--extension-dir",
                str(extension),
            ]
        )
    assert not final_output.exists()
