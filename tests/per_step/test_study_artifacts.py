from __future__ import annotations

from collections import Counter
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scpcp.artifacts import (
    mark_study_complete,
    source_tree_sha256,
    write_collection_status,
    write_seed_result,
    write_study_metadata,
)
from scpcp.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]


def test_artifacts_are_complete_and_source_hash_is_reproducible(tmp_path: Path) -> None:
    output = tmp_path / "study"
    config = ExperimentConfig(devices=("cpu",), seeds=(0,), output_dir=output)
    write_study_metadata(output, config)
    result = SimpleNamespace(
        seed=0,
        device="cpu",
        records=[{"method": "SC-PCP", "track": "logged_data"}],
        surfaces={"q_grid": torch.tensor([1.0])},
        diagnostics={},
    )

    seed_dir = write_seed_result(result, output, config)
    mark_study_complete(output, config.seeds)
    study_metadata = json.loads((output / "study_metadata.json").read_text())
    seed_metadata = json.loads((seed_dir / "metadata.json").read_text())

    assert (seed_dir / "COMPLETE").is_file()
    assert (output / "COMPLETE").is_file()
    assert study_metadata["source_tree_sha256"] == source_tree_sha256()
    assert seed_metadata["source_tree_sha256"] == source_tree_sha256()
    assert len(source_tree_sha256()) == 64


def test_collection_status_requires_every_setting(tmp_path: Path) -> None:
    expected = ("first", "second")
    write_collection_status(tmp_path, expected, status="running", completed_settings=("first",))
    running = json.loads((tmp_path / "study_status.json").read_text())

    assert running["missing_settings"] == ["second"]
    assert not (tmp_path / "COMPLETE").exists()

    with pytest.raises(RuntimeError, match="missing completed settings"):
        write_collection_status(tmp_path, expected, status="complete", completed_settings=("first",))
    incomplete = json.loads((tmp_path / "study_status.json").read_text())

    assert incomplete["status"] == "incomplete"
    assert incomplete["missing_settings"] == ["second"]
    assert not (tmp_path / "COMPLETE").exists()

    write_collection_status(tmp_path, expected, status="complete", completed_settings=expected)
    complete = json.loads((tmp_path / "study_status.json").read_text())

    assert complete["status"] == "complete"
    assert complete["missing_settings"] == []
    assert (tmp_path / "COMPLETE").is_file()


def test_study_rejects_seed_with_mismatched_source_hash(tmp_path: Path) -> None:
    output = tmp_path / "study"
    config = ExperimentConfig(devices=("cpu",), seeds=(0,), output_dir=output)
    write_study_metadata(output, config)
    result = SimpleNamespace(
        seed=0,
        device="cpu",
        records=[{"method": "SC-PCP", "track": "logged_data"}],
        surfaces={"q_grid": torch.tensor([1.0])},
        diagnostics={},
    )
    seed_dir = write_seed_result(result, output, config)
    metadata_path = seed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_tree_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="source hash differs"):
        mark_study_complete(output, config.seeds)

    status = json.loads((output / "study_status.json").read_text())
    assert status["status"] == "incomplete"
    assert not (output / "COMPLETE").exists()


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    module_name = f"test_tool_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_factorial_settings_only_change_feedback_and_policy_tilt() -> None:
    study = _load_script("run_per_step_study")
    base = ExperimentConfig()

    settings = study.build_factorial_settings(base, ("0", "2"), ("0.25", "1"))

    assert [label for label, _ in settings] == [
        "beta_0__eta_0.25",
        "beta_0__eta_1",
        "beta_2__eta_0.25",
        "beta_2__eta_1",
    ]
    assert {(config.synthetic.feedback_strength, config.policy.tilt) for _, config in settings} == {
        (0.0, 0.25),
        (0.0, 1.0),
        (2.0, 0.25),
        (2.0, 1.0),
    }
    assert all(config.horizon == base.horizon for _, config in settings)
    assert all(config.samples == base.samples for _, config in settings)


def test_track_a_rates_distinguish_target_attainment_certificate_and_abstention() -> None:
    plots = _load_script("plot_per_step")
    records = pd.DataFrame(
        [
            {
                "run_root": "/runs/toy",
                "dataset": "synthetic",
                "horizon": 4,
                "setting": "toy",
                "information_regime": "offline_logged_data",
                "selection_estimand": "per_step",
                "method": "SC-PCP",
                "target_coverage": 0.9,
                "selection_status": "PRACTICAL_CLUSTER_MAX_T_LCB",
                "certificate_formal": False,
                "worst_coverage": 0.95,
                "average_coverage": 0.96,
                "pathwise_coverage": 0.80,
                "mean_log_volume": 1.0,
                "clinical_cost": 0.2,
                "selected_q": 1.0,
                "target_policy_trajectories": 0,
            },
            {
                "run_root": "/runs/toy",
                "dataset": "synthetic",
                "horizon": 4,
                "setting": "toy",
                "information_regime": "offline_logged_data",
                "selection_estimand": "per_step",
                "method": "SC-PCP",
                "target_coverage": 0.9,
                "selection_status": "UNCERTIFIED",
                "certificate_formal": False,
                "worst_coverage": np.nan,
                "average_coverage": np.nan,
                "pathwise_coverage": np.nan,
                "mean_log_volume": np.nan,
                "clinical_cost": np.nan,
                "selected_q": np.nan,
                "target_policy_trajectories": 0,
            },
        ]
    )

    summary = plots.summarize_track_a(records)
    certification = plots.track_a_certification_summary(summary)

    assert summary.loc[0, "fresh_target_met_rate_all_runs"] == 0.5
    assert summary.loc[0, "formal_certificate_rate"] == 0.0
    assert summary.loc[0, "abstention_rate"] == 0.5
    forbidden = {
        "cert_rate",
        "cert_rate_among_evaluated",
        "target_met_rate",
        "primary_per_step_target_met_rate",
        "uncertified_rate",
    }
    assert forbidden.isdisjoint(summary.columns)
    assert forbidden.isdisjoint(certification.columns)


def test_main_results_keep_baselines_and_separate_information_regimes() -> None:
    plots = _load_script("plot_per_step")
    summary = pd.DataFrame(
        [
            {
                "run_root": "/runs/toy",
                "dataset": "synthetic",
                "horizon": 4,
                "setting": "toy",
                "information_regime": "offline_logged_data",
                "selection_estimand": "per_step",
                "method": "SC-PCP",
                "n_runs": 2,
                "n_evaluated": 2,
                "n_abstained": 0,
                "target_coverage": 0.9,
                "fresh_target_met_rate_all_runs": 1.0,
                "abstention_rate": 0.0,
                "formal_certificate_rate": 0.0,
                "mean_worst_coverage": 0.92,
                "se_worst_coverage": 0.01,
                "mean_average_coverage": 0.94,
                "se_average_coverage": 0.01,
                "mean_worst_gap": 0.0,
                "se_worst_gap": 0.0,
                "mean_log_volume": 1.0,
                "se_log_volume": 0.1,
                "mean_median_volume": 2.0,
                "se_median_volume": 0.2,
                "mean_clinical_cost": 0.3,
                "se_clinical_cost": 0.01,
                "mean_target_policy_trajectories": 0.0,
                "mean_evaluation_trajectories": 50_000.0,
                "mean_adaptation_worst_coverage": float("nan"),
                "se_adaptation_worst_coverage": float("nan"),
                "mean_adaptation_average_coverage": float("nan"),
                "mean_adaptation_pathwise_coverage": float("nan"),
            },
            {
                "run_root": "/runs/toy",
                "dataset": "synthetic",
                "horizon": 4,
                "setting": "toy",
                "information_regime": "on_policy_adaptation",
                "selection_estimand": "per_step",
                "method": "ACI-style online",
                "n_runs": 2,
                "n_evaluated": 2,
                "n_abstained": 0,
                "target_coverage": 0.9,
                "fresh_target_met_rate_all_runs": 0.5,
                "abstention_rate": 0.0,
                "formal_certificate_rate": 0.0,
                "mean_worst_coverage": 0.89,
                "se_worst_coverage": 0.01,
                "mean_average_coverage": 0.91,
                "se_average_coverage": 0.01,
                "mean_worst_gap": 0.01,
                "se_worst_gap": 0.01,
                "mean_log_volume": 0.9,
                "se_log_volume": 0.1,
                "mean_median_volume": 1.8,
                "se_median_volume": 0.2,
                "mean_clinical_cost": 0.3,
                "se_clinical_cost": 0.01,
                "mean_target_policy_trajectories": 2_000.0,
                "mean_evaluation_trajectories": 50_000.0,
                "mean_adaptation_worst_coverage": 0.88,
                "se_adaptation_worst_coverage": 0.01,
                "mean_adaptation_average_coverage": 0.90,
                "mean_adaptation_pathwise_coverage": 0.70,
            },
        ]
    )

    main = plots.build_main_results(summary)

    assert set(main["method"]) == {"SC-PCP", "ACI-style online"}
    assert set(main["comparison_panel"]) == {"offline", "online_with_adaptation_data"}
    assert main[["run_root", "dataset", "horizon"]].drop_duplicates().to_dict("records") == [
        {"run_root": "/runs/toy", "dataset": "synthetic", "horizon": 4}
    ]
    assert main.set_index("method").loc["ACI-style online", "mean_target_policy_trajectories"] == 2_000


def test_online_adaptation_round_summary_is_seed_paired_and_stably_empty() -> None:
    plots = _load_script("plot_per_step")
    columns = [
        "run_root",
        "dataset",
        "horizon",
        "setting",
        "seed",
        "method",
        "selection_estimand",
        "adaptation_round_worst_coverage",
    ]
    empty = plots.build_adaptation_round_source(pd.DataFrame(columns=columns))
    assert list(empty.columns) == [
        "run_root",
        "dataset",
        "horizon",
        "setting",
        "method",
        "round",
        "n_runs",
        "mean_round_worst_coverage",
        "se_round_worst_coverage",
    ]

    records = pd.DataFrame(
        [
            {
                "run_root": "/runs/toy",
                "dataset": "synthetic",
                "horizon": 2,
                "setting": "toy",
                "seed": 0,
                "method": "ACI-style online (gamma=0.01)",
                "selection_estimand": "per_step",
                "adaptation_round_worst_coverage": "[0.8, 0.9]",
            },
            {
                "run_root": "/runs/toy",
                "dataset": "synthetic",
                "horizon": 2,
                "setting": "toy",
                "seed": 1,
                "method": "ACI-style online (gamma=0.01)",
                "selection_estimand": "per_step",
                "adaptation_round_worst_coverage": "[1.0, 0.7]",
            },
        ]
    )
    summary = plots.build_adaptation_round_source(records)

    assert summary["round"].tolist() == [1, 2]
    assert summary["n_runs"].tolist() == [2, 2]
    assert summary["mean_round_worst_coverage"].tolist() == pytest.approx([0.9, 0.8])


def test_track_a_summary_never_merges_same_setting_across_provenance() -> None:
    plots = _load_script("plot_per_step")
    conditions = [
        ("/runs/a/shared", "mimic_iv", 12, 0.95),
        ("/runs/b/shared", "eicu", 12, 0.85),
        ("/runs/c/shared", "mimic_iv", 6, 0.75),
        ("/runs/d/shared", "mimic_iv", 12, 0.65),
    ]
    records = pd.DataFrame(
        [
            {
                "run_root": run_root,
                "dataset": dataset,
                "horizon": horizon,
                "setting": "shared",
                "information_regime": "offline_logged_data",
                "selection_estimand": "per_step",
                "method": "SC-PCP",
                "target_coverage": 0.9,
                "selection_status": "PRACTICAL_CLUSTER_MAX_T_LCB",
                "certificate_formal": False,
                "worst_coverage": coverage,
            }
            for run_root, dataset, horizon, coverage in conditions
        ]
    )

    summary = plots.summarize_track_a(records)
    main = plots.build_main_results(summary)

    assert len(summary) == len(conditions)
    assert len(main) == len(conditions)
    keyed = summary.set_index(["run_root", "dataset", "horizon"])
    for run_root, dataset, horizon, coverage in conditions:
        row = keyed.loc[(run_root, dataset, horizon)]
        assert row["n_runs"] == 1
        assert row["mean_worst_coverage"] == coverage
    labels = {
        plots._figure_context_label(run_root, dataset, horizon, "shared")
        for run_root, dataset, horizon, _ in conditions
    }
    assert len(labels) == len(conditions)


def test_factorial_summary_keeps_dataset_context_and_unambiguous_rates() -> None:
    plots = _load_script("plot_per_step")
    rows = []
    for dataset, coverage in (("mimic_iv", 0.95), ("eicu", 0.75)):
        for beta in (0.0, 1.0):
            for eta in (0.5, 1.0):
                rows.append(
                    {
                        "study_root": "/studies/shared_factorial",
                        "dataset": dataset,
                        "horizon": 12,
                        "selection_estimand": "per_step",
                        "method_family": "SC-PCP",
                        "feedback_strength": beta,
                        "policy_tilt": eta,
                        "target_coverage": 0.9,
                        "selection_status": "PRACTICAL_CLUSTER_MAX_T_LCB",
                        "certificate_formal": False,
                        "worst_coverage": coverage,
                    }
                )

    summary = plots.summarize_factorial_track_a(pd.DataFrame(rows))

    assert len(summary) == 8
    assert summary.groupby("dataset")["n_runs"].unique().map(list).to_dict() == {
        "eicu": [1],
        "mimic_iv": [1],
    }
    assert summary.groupby("dataset")["mean_worst_coverage"].unique().map(list).to_dict() == {
        "eicu": [0.75],
        "mimic_iv": [0.95],
    }
    assert summary.groupby("dataset")["fresh_target_met_rate_all_runs"].unique().map(list).to_dict() == {
        "eicu": [0.0],
        "mimic_iv": [1.0],
    }
    assert {"fresh_target_met_rate_all_runs", "formal_certificate_rate", "abstention_rate"}.issubset(
        summary.columns
    )
    assert {"cert_rate", "cert_rate_among_evaluated"}.isdisjoint(summary.columns)


def test_target_attainment_tolerates_float32_roundoff() -> None:
    plots = _load_script("plot_per_step")
    records = pd.DataFrame({"worst_coverage": [0.799999952]})

    assert plots._all_run_target_met_rate(records, 0.8, "worst_coverage") == 1.0


def test_horizon_summary_does_not_mix_datasets() -> None:
    plots = _load_script("plot_per_step")
    diagnostics = pd.DataFrame(
        [
            {
                "study_root": "/tmp/main",
                "dataset": "mimic_cxr",
                "horizon": 6,
                "feedback_strength": 1.0,
                "policy_tilt": 1.0,
                "target_coverage": 0.9,
                "logged_trajectories": np.nan,
                "policy_ratio_cap": 10.0,
            },
            {
                "study_root": "/tmp/main",
                "dataset": "mimic_iv",
                "horizon": 12,
                "feedback_strength": 1.0,
                "policy_tilt": 1.0,
                "target_coverage": 0.9,
                "logged_trajectories": np.nan,
                "policy_ratio_cap": 10.0,
            },
        ]
    )

    assert plots.summarize_cot_iw_horizon_diagnostics(diagnostics).empty


def test_dcov_markers_exclude_nonconstant_stagewise_rows() -> None:
    plots = _load_script("plot_per_step")
    records = pd.DataFrame(
        [
            {
                "track": "empirical_environment",
                "method": "Historical CP",
                "selection_estimand": "per_step",
                "selected_q": 1.0,
                "q_by_time": "",
            },
            {
                "track": "empirical_environment",
                "method": "Repeated recalibration",
                "selection_estimand": "per_step",
                "selected_q": 1.2,
                "q_by_time": "[1.2, 1.2]",
            },
            {
                "track": "empirical_environment",
                "method": "ACI-style online",
                "selection_estimand": "per_step",
                "selected_q": 1.1,
                "q_by_time": "[1.0, 1.2]",
            },
        ]
    )

    points = plots.per_step_scalar_selection_points(records, np.array([0.8, 1.0, 1.2, 1.4]))

    assert [label for label, *_ in points] == ["Historical CP", "Repeated recalibration"]


def test_primary_curve_source_keeps_stagewise_methods() -> None:
    plots = _load_script("plot_per_step")
    records = pd.DataFrame(
        [
            {
                "run_root": "/runs/toy",
                "dataset": "synthetic",
                "horizon": 2,
                "setting": "toy",
                "seed": 0,
                "method": "ACI-style online",
                "selection_estimand": "per_step",
                "q_by_time": "[1.0, 1.2]",
                "per_time_coverage": "[0.90, 0.91]",
            },
        ]
    )

    source = plots.build_per_time_source(records)

    assert source["method"].unique().tolist() == ["ACI-style online"]
    assert source["coverage"].tolist() == [0.90, 0.91]
    assert source[["run_root", "dataset", "horizon"]].drop_duplicates().to_dict("records") == [
        {"run_root": "/runs/toy", "dataset": "synthetic", "horizon": 2}
    ]


def test_surface_selection_ignores_incomplete_seed(tmp_path: Path) -> None:
    plots = _load_script("plot_per_step")
    incomplete = tmp_path / "seed_00000"
    complete = tmp_path / "seed_00001"
    incomplete.mkdir()
    complete.mkdir()
    np.savez(incomplete / "surfaces.npz", q_grid=np.array([0.0]))
    np.savez(complete / "surfaces.npz", q_grid=np.array([1.0]))
    (complete / "COMPLETE").write_text("complete\n")

    assert plots.select_surface_file(tmp_path, None) == complete / "surfaces.npz"
    assert plots.select_surface_file(tmp_path, 0) is None
    assert plots.select_surface_file(tmp_path, 1) == complete / "surfaces.npz"


def test_seed_jobs_pin_every_persistent_worker_to_one_device() -> None:
    for runner in (_load_script("run_per_step"), _load_script("run_per_step_study")):
        worker_devices, jobs = runner._build_seed_jobs(
            (10, 11, 12, 13, 14, 15),
            ("cuda:0", "cuda:1"),
            workers_per_device=2,
        )

        assert worker_devices == ("cuda:0", "cuda:0", "cuda:1", "cuda:1")
        assert jobs == (
            (0, 10, "cuda:0"),
            (1, 11, "cuda:0"),
            (2, 12, "cuda:1"),
            (3, 13, "cuda:1"),
            (0, 14, "cuda:0"),
            (1, 15, "cuda:0"),
        )
        assert all(worker_devices[worker] == device for worker, _, device in jobs)
        with pytest.raises(ValueError, match="workers_per_device must be positive"):
            runner._build_seed_jobs((0,), ("cpu",), workers_per_device=0)


def test_seed_parsers_reject_empty_negative_and_duplicate_designs() -> None:
    main_runner = _load_script("run_per_step")
    study_runner = _load_script("run_per_step_study")

    assert main_runner._parse_seeds("0:200", (9,)) == tuple(range(200))
    assert study_runner.parse_seeds("0:200", (9,)) == tuple(range(200))
    for parse in (main_runner._parse_seeds, study_runner.parse_seeds):
        with pytest.raises(ValueError, match="at least one seed"):
            parse("2:2", (0,))
        with pytest.raises(ValueError, match="nonnegative"):
            parse("-1,0", (0,))
        with pytest.raises(ValueError, match="unique"):
            parse("1,1", (0,))


def test_prespecified_settings_change_only_the_intended_parameters() -> None:
    study = _load_script("run_per_step_study")
    base = ExperimentConfig()

    ratio_settings = dict(study.build_settings("ratio_cap", base, ("1.1", "1.25", "2")))
    for label, cap in (("ratio_cap_1.1", 1.1), ("ratio_cap_1.25", 1.25), ("ratio_cap_2", 2.0)):
        config = ratio_settings[label]
        assert config.policy.policy_ratio_cap == cap
        assert config.cot.weight_cap == pytest.approx(base.cot.rho_cap * cap)
        assert config.policy.action_costs == base.policy.action_costs
        assert config.policy.tilt == base.policy.tilt
        assert config.cot.rho_cap == base.cot.rho_cap

    action_settings = dict(study.build_settings("action_cost", base, ("0", "0.05", "0.20")))
    assert action_settings["action_cost_0"].policy.action_costs == (0.0, 0.0, 0.0)
    assert action_settings["action_cost_0.05"].policy.action_costs == (0.0, 0.025, 0.05)
    assert action_settings["action_cost_0.20"].policy.action_costs == (0.0, 0.1, 0.2)
    assert all(config.policy.tilt == base.policy.tilt for config in action_settings.values())

    aci = dict(study.build_settings("aci_gamma", base, ("0.005", "0.1")))
    assert aci["aci_gamma_0.005"].baselines.aci_gamma == 0.005
    assert aci["aci_gamma_0.1"].baselines.aci_gamma == 0.1
    assert all(config.baselines.multidim_buffer == base.baselines.multidim_buffer for config in aci.values())

    buffers = dict(study.build_settings("multidim_buffer", base, ("250", "500")))
    assert buffers["multidim_buffer_250"].baselines.multidim_buffer == 250
    assert buffers["multidim_buffer_500"].baselines.multidim_buffer == 500
    assert all(config.baselines.aci_gamma == base.baselines.aci_gamma for config in buffers.values())

    depths = dict(study.build_settings("mfcs_depth", base, ("1", "2", "4")))
    assert {label: config.baselines.mfcs_depth for label, config in depths.items()} == {
        "mfcs_depth_1": 1,
        "mfcs_depth_2": 2,
        "mfcs_depth_4": 4,
    }
    assert all(config.baselines.aci_gamma == base.baselines.aci_gamma for config in depths.values())


def test_full200_launcher_has_exact_34_noncentral_tasks() -> None:
    launcher = _load_tool("run_full200_shards")
    tasks = launcher.build_tasks()

    assert len(tasks) == 34
    assert len({(task.section, task.label) for task in tasks}) == 34
    assert all(task.seeds == "0:200" for task in tasks)
    assert Counter(task.section for task in tasks) == Counter(
        {
            "factorial": 15,
            "feedback_extra": 1,
            "horizon": 3,
            "sample_size": 4,
            "ratio_cap": 3,
            "alpha": 2,
            "action_cost": 3,
            "mfcs_depth": 3,
        }
    )

    factorial = {
        (task.extra_args[1], task.extra_args[3])
        for task in tasks
        if task.section == "factorial"
    }
    expected_factorial = {
        (beta, eta)
        for beta in ("0", "0.5", "1", "2")
        for eta in ("0.25", "0.5", "1", "2")
    } - {("1", "1")}
    assert factorial == expected_factorial

    ratio_values = {
        task.extra_args[-1]
        for task in tasks
        if task.section == "ratio_cap"
    }
    assert ratio_values == {"1.1", "1.25", "2"}
    assert {task.extra_args[-1] for task in tasks if task.section == "action_cost"} == {
        "0",
        "0.05",
        "0.2",
    }
    assert {task.extra_args[-1] for task in tasks if task.section == "mfcs_depth"} == {
        "1",
        "2",
        "4",
    }


def test_full200_command_maps_one_task_to_one_gpu_and_four_seed_workers(tmp_path: Path) -> None:
    launcher = _load_tool("run_full200_shards")
    task = launcher.build_tasks()[0]
    command = launcher.build_command(
        task,
        device="cuda:1",
        config=tmp_path / "config.yaml",
        task_dir=tmp_path / "task",
        seed_workers_per_device=4,
    )

    def option(name: str) -> str:
        return command[command.index(name) + 1]

    assert option("--devices") == "cuda:1"
    assert option("--workers-per-device") == "4"
    assert option("--seeds") == "0:200"
    assert option("--study") == "factorial"
    assert option("--feedback-values") == "0"
    assert option("--policy-tilt-values") == "0.25"


def _write_valid_full200_collection(
    task_dir: Path,
    task: object,
    source_hash: str,
) -> Path:
    collection = task_dir / "20260812T000000Z_collection"
    setting = collection / task.label
    setting.mkdir(parents=True)
    seed_list = list(range(200))
    for seed in seed_list:
        seed_dir = setting / f"seed_{seed:05d}"
        seed_dir.mkdir()
        (seed_dir / "COMPLETE").write_text("complete\n")
    expected_study = "feedback_policy" if task.study == "factorial" else task.study
    (collection / "study_manifest.json").write_text(
        json.dumps(
            {
                "study": expected_study,
                "settings": [{"label": task.label}],
                "seeds": seed_list,
                "source_tree_sha256": source_hash,
            }
        )
    )
    (collection / "study_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_settings": [task.label],
                "missing_settings": [],
            }
        )
    )
    (setting / "study_metadata.json").write_text(
        json.dumps(
            {
                "seeds": seed_list,
                "source_tree_sha256": source_hash,
                "execution": {"collection_source_tree_sha256": source_hash},
            }
        )
    )
    (setting / "study_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "expected_seeds": seed_list,
                "completed_seeds": seed_list,
                "missing_seeds": [],
            }
        )
    )
    (setting / "COMPLETE").write_text("complete\n")
    (collection / "COMPLETE").write_text("complete\n")
    return collection


def test_full200_skip_requires_exact_manifest_status_seeds_and_source_hash(tmp_path: Path) -> None:
    launcher = _load_tool("run_full200_shards")
    task = launcher.build_tasks()[0]
    source_hash = "a" * 64
    task_dir = tmp_path / "task"
    collection = _write_valid_full200_collection(task_dir, task, source_hash)

    assert launcher.completed_collection(
        task_dir,
        task=task,
        expected_source_hash=source_hash,
    ) == collection

    manifest_path = collection / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_tree_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest))
    assert launcher.completed_collection(
        task_dir,
        task=task,
        expected_source_hash=source_hash,
    ) is None

    manifest["source_tree_sha256"] = source_hash
    manifest_path.write_text(json.dumps(manifest))
    (collection / task.label / "seed_00199" / "COMPLETE").unlink()
    assert launcher.completed_collection(
        task_dir,
        task=task,
        expected_source_hash=source_hash,
    ) is None


def test_full200_task_lock_prevents_duplicate_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_tool("run_full200_shards")
    task = launcher.build_tasks()[0]
    source_hash = "a" * 64
    output_root = tmp_path / "output"
    task_dir = output_root / task.section / task.label
    task_dir.mkdir(parents=True)
    logs_root = output_root / "logs"
    logs_root.mkdir()
    monkeypatch.setattr(launcher, "source_tree_sha256", lambda: source_hash)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("locked task must not launch a subprocess"),
    )

    with (task_dir / ".launcher.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        outcome = launcher.run_task(
            task,
            device="cuda:0",
            config=tmp_path / "config.yaml",
            output_root=output_root,
            logs_root=logs_root,
            seed_workers_per_device=4,
            expected_source_hash=source_hash,
        )

    assert outcome["status"] == "claimed_elsewhere"


def test_full200_incompatible_complete_blocks_ambiguous_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_tool("run_full200_shards")
    task = launcher.build_tasks()[0]
    source_hash = "a" * 64
    output_root = tmp_path / "output"
    task_dir = output_root / task.section / task.label
    stale = task_dir / "old_collection"
    stale.mkdir(parents=True)
    (stale / "COMPLETE").write_text("complete\n")
    logs_root = output_root / "logs"
    logs_root.mkdir()
    monkeypatch.setattr(launcher, "source_tree_sha256", lambda: source_hash)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("an incompatible COMPLETE must be resolved before rerunning"),
    )

    outcome = launcher.run_task(
        task,
        device="cuda:0",
        config=tmp_path / "config.yaml",
        output_root=output_root,
        logs_root=logs_root,
        seed_workers_per_device=4,
        expected_source_hash=source_hash,
    )

    assert outcome["status"] == "incompatible_complete"
    assert outcome["collections"] == [str(stale)]


def test_study_collection_manifest_freezes_hash_and_publishes_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _load_script("run_per_step_study")
    source_hash = "a" * 64
    base = ExperimentConfig(devices=("cpu",), seeds=(0,))
    output = tmp_path / "collections"
    monkeypatch.setattr(study.ExperimentConfig, "from_yaml", staticmethod(lambda _: base))
    monkeypatch.setattr(study, "resolve_devices", lambda _: ("cpu",))
    monkeypatch.setattr(study, "source_tree_sha256", lambda: source_hash)
    monkeypatch.setattr(
        study,
        "write_study_metadata",
        lambda output_dir, config, execution=None: output_dir.mkdir(parents=True),
    )
    monkeypatch.setattr(
        study,
        "run_setting",
        lambda config, output_dir, workers_per_device=1: (output_dir / "COMPLETE").write_text("complete\n"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_per_step_study.py",
            "--study",
            "ratio_cap",
            "--values",
            "1.1",
            "--output-dir",
            str(output),
            "--devices",
            "cpu",
            "--seeds",
            "0",
            "--workers-per-device",
            "3",
        ],
    )

    study.main()

    collection = next(output.iterdir())
    manifest = json.loads((collection / "study_manifest.json").read_text())
    status = json.loads((collection / "study_status.json").read_text())
    assert manifest["source_tree_sha256"] == source_hash
    assert manifest["workers_per_device"] == 3
    assert status["status"] == "complete"
    assert (collection / "COMPLETE").is_file()


def test_study_source_drift_marks_collection_failed_without_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _load_script("run_per_step_study")
    base = ExperimentConfig(devices=("cpu",), seeds=(0,))
    output = tmp_path / "collections"
    hashes = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(study.ExperimentConfig, "from_yaml", staticmethod(lambda _: base))
    monkeypatch.setattr(study, "resolve_devices", lambda _: ("cpu",))
    monkeypatch.setattr(study, "source_tree_sha256", lambda: next(hashes))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_per_step_study.py",
            "--study",
            "ratio_cap",
            "--values",
            "1.1",
            "--output-dir",
            str(output),
            "--devices",
            "cpu",
            "--seeds",
            "0",
        ],
    )

    with pytest.raises(RuntimeError, match="active source changed"):
        study.main()

    collection = next(output.iterdir())
    status = json.loads((collection / "study_status.json").read_text())
    assert status["status"] == "failed"
    assert not (collection / "COMPLETE").exists()
