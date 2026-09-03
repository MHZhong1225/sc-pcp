from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_controlled_clinical_fidelity_v3_science.py"


def _load_runner():
    name = "test_run_controlled_clinical_fidelity_v3_science"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _tiny_preset():
    protocol = runner.v2.load_extension_config(runner.clinical.V2_CONFIG_PATH)
    return runner.replace(
        protocol.datasets["eicu"],
        seeds=tuple(range(7, 27)),
        bootstrap_seed=987_654,
    )


def _tiny_science_rows(preset) -> list[dict[str, object]]:
    widths = {
        "Standard CP": 3.0,
        "ACI": 2.5,
        "MFCS": 2.2,
        "SPCI": 2.0,
        "PRC": 1.8,
        "SC-PCP": 1.5,
    }
    rows = []
    for gamma in runner.GAMMAS:
        for index, seed in enumerate(preset.seeds):
            coverage = [0.8, 1.0] if index < 10 else [1.0, 0.8]
            coverage.extend([0.95] * (preset.horizon - 2))
            methods = {
                method: {
                    "selection_available": True,
                    "source_coverage": [0.9] * preset.horizon,
                    "target_coverage": coverage,
                    "target_normalized_width": [widths[method]] * preset.horizon,
                    "prefix_ess_fraction": [0.5] * preset.horizon,
                    "maximum_normalized_weight_share": [0.02] * preset.horizon,
                }
                for method in runner.METHODS
            }
            rows.append(
                {
                    "seed": seed,
                    "dataset": preset.name,
                    "gamma": gamma,
                    "methods": methods,
                }
            )
    return rows


def _orchestration_gates():
    return SimpleNamespace(
        source_tree_sha256="source",
        frozen_theta={dataset: object() for dataset in runner.DATASETS},
        anchors={dataset: {} for dataset in runner.DATASETS},
        seed_to_device={
            dataset: {1: "cuda:0"} for dataset in runner.DATASETS
        },
        science_config=object(),
        rng_audit={"new_rng_stream_mapping_sha256": "rng"},
        contract={"gate": "frozen"},
    )


def _overlap_nested_fixture() -> dict[str, object]:
    metrics = {
        "local_ess_p01": 12.0,
        "median_ess_fraction": 0.5,
        "maximum_donor_probability": 0.1,
    }
    prefix = {
        "minimum_ess_fraction": 0.2,
        "maximum_normalized_weight_share": 0.1,
        "maximum_raw_log_weight_span": 2.0,
        "gate_role": "report-only",
    }
    probe = {
        "radius_fraction": 0.5,
        "radius": 1.5,
        "metrics": metrics,
        "passed": True,
        "target_simplex_maximum_error": 0.0,
        "logging_simplex_maximum_error": 0.0,
        "minimum_logging_probability": 0.1,
        "minimum_target_probability": 0.1,
        "policy_probabilities_finite": True,
        "maximum_single_step_target_to_logging_ratio": 2.0,
        "single_step_ratio_cap": 3.0,
        "local_unique_k_minimum": 20.0,
        "local_unique_k_median": 30.0,
        "prefix_overlap_report_only": prefix,
    }
    return {
        "metrics": metrics,
        "diagnostics": {
            "probe_trajectories": 3_000,
            "gamma": -4.0,
            "noise_seed": 1,
            "common_random_numbers_across_radii": True,
            "independent_frozen_stream": True,
            "patient_aggregated": True,
            "episode_weighted_transition_patient_aggregated_diagnostics": True,
            "probes": {"q_mid": deepcopy(probe), "q_high": deepcopy(probe)},
            "worst_metrics": metrics,
            "screen_status": "EMPIRICAL_OVERLAP_SCREEN_PASSED",
            "screen_scope": "fixture",
            "environment_episode_support": {},
        },
    }


def _patch_orchestration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    output = tmp_path / "nonformal_science"
    monkeypatch.setattr(runner, "OUTPUT_ROOT", output)
    monkeypatch.setattr(
        runner,
        "_active_source_snapshot",
        lambda: ("source", {"contract": {}}),
    )
    monkeypatch.setattr(runner, "_science_metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runner,
        "_prepare_root",
        lambda root, *args, **kwargs: root.mkdir(parents=True),
    )
    monkeypatch.setattr(runner, "_require_partial_artifact_subset", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_confirmation_preset",
        lambda gates, dataset: SimpleNamespace(name=dataset, seeds=(1,)),
    )
    return output


def test_science_contract_and_cli_surface_are_frozen(monkeypatch) -> None:
    assert tuple(runner.METHODS) == (
        "Standard CP",
        "ACI",
        "MFCS",
        "SPCI",
        "PRC",
        "SC-PCP",
    )
    assert runner.GAMMAS == (-4.0, -2.0, 0.0, 2.0, 4.0)
    assert runner.PRIMARY_GAMMA == -4.0
    assert runner.PRIMARY_METRIC == "min_t mean_seed(target_coverage_seed_t)"
    assert runner.SCIENCE_CONTRACT["calibration_trajectories"] == 3_000
    assert runner.SCIENCE_CONTRACT["grid_trajectories"] == 1_000
    assert runner.SCIENCE_CONTRACT["evaluation_trajectories"] == 20_000
    assert runner.SCIENCE_CONTRACT["bootstrap_resamples"] == 10_000
    assert runner.SCIENCE_CONTRACT["bootstrap_seed_count"] == 20
    assert runner.SCIENCE_CONTRACT["policy_ratio_cap"] == 3.0
    assert runner.SCIENCE_CONTRACT["common_random_numbers"] == {
        "source_calibration_across_gamma": True,
        "source_reference_across_gamma": True,
        "target_reference_across_methods_and_gamma": True,
        "online_baselines": "independent method streams reused across gamma",
    }
    assert runner.SCIENCE_CONTRACT["target_adaptation_trajectories"] == {
        "Standard CP": 0,
        "ACI": 2_000,
        "MFCS": 0,
        "SPCI": 2_000,
        "PRC": 2_000,
        "SC-PCP": 0,
    }

    tree = ast.parse(RUNNER_PATH.read_text())
    options = {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }
    assert options == {"--devices", "--resume"}

    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH), "--gamma", "-4"])
    with pytest.raises(SystemExit):
        runner.main()


def test_contract_matches_live_frozen_v2_and_confirmation_banks() -> None:
    science = runner.v2.load_extension_config(runner.clinical.V2_CONFIG_PATH)
    fidelity = runner.load_fidelity_v3_config(runner.clinical.CONFIG_PATH)

    runner._validate_science_contract(science, fidelity)
    assert all(len(fidelity.confirmation_seeds[name]) == 20 for name in runner.DATASETS)

    altered = runner.replace(science, reference_trajectories=19_999)
    with pytest.raises((ValueError, RuntimeError), match="budgets|constants differ"):
        runner._validate_science_contract(altered, fidelity)


def test_rng_binding_requires_the_exact_reserved_future_stream_map() -> None:
    mapping = {f"stream_{index}": index for index in range(1_304)}
    audit = {
        "status": "passed_before_launch",
        "collision_count": 0,
        "collisions": {},
        "internal_rng_streams_unique": True,
        "new_rng_stream_count": 1_304,
        "new_rng_stream_mapping_sha256": "a" * 64,
        "new_rng_stream_mapping": mapping,
    }
    runner._validate_rng_binding(audit, deepcopy(audit), deepcopy(audit))

    changed = deepcopy(audit)
    changed["new_rng_stream_mapping"]["stream_0"] = -1
    with pytest.raises(RuntimeError, match="confirmation RNG binding differs"):
        runner._validate_rng_binding(audit, changed, audit)

    collided = deepcopy(audit)
    collided["collision_count"] = 1
    collided["collisions"] = {"stream_0": 0}
    with pytest.raises(RuntimeError, match="confirmation RNG binding differs"):
        runner._validate_rng_binding(audit, audit, collided)


def test_verifier_hard_depends_on_development_before_confirmation(monkeypatch) -> None:
    sentinel = FileNotFoundError("development COMPLETE is absent")
    monkeypatch.setattr(runner.clinical, "verify_parent_v2", lambda root: {})

    def fail_development(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(
        runner.clinical,
        "_verify_development_for_confirmation",
        fail_development,
    )
    with pytest.raises(FileNotFoundError, match="development COMPLETE"):
        runner.verify_gate_bundle(devices=("cuda:0", "cuda:1"))


def test_overlap_firewall_rejects_nested_science_fields() -> None:
    runner._reject_overlap_science_keys(
        {
            "nested": {
                "local_ess": [12.0],
                "screen_status": "passed",
                "donor_bandwidth": 2.0,
                "theta": {"bandwidth": 4.0},
            }
        }
    )
    for forbidden in (
        {"nested": {"stage_coverage": [0.91]}},
        {"nested": {"interval_width": [1.2]}},
        {"nested": {"method_selection": "SC-PCP"}},
        {"nested": {"science_result": True}},
    ):
        with pytest.raises(RuntimeError, match="forbidden science key"):
            runner._reject_overlap_science_keys(forbidden)


@pytest.mark.parametrize(
    "field",
    (
        "minimum_ess_fraction",
        "maximum_normalized_weight_share",
        "maximum_raw_log_weight_span",
    ),
)
def test_overlap_prefix_diagnostics_reject_nonfinite(field: str) -> None:
    result = _overlap_nested_fixture()
    runner._validate_overlap_nested_schemas(result)
    result["diagnostics"]["probes"]["q_mid"][
        "prefix_overlap_report_only"
    ][field] = float("nan")
    with pytest.raises(RuntimeError, match="prefix diagnostics differ"):
        runner._validate_overlap_nested_schemas(result)


def test_phase_payload_locks_top_level_schema(monkeypatch) -> None:
    preset = runner.v2.load_extension_config(runner.clinical.V2_CONFIG_PATH).datasets[
        "eicu"
    ]
    theta = runner.KernelTheta(
        "A00_raw_k100_gaussian_b2",
        "raw",
        100,
        "gaussian_b2",
    )
    anchor = runner.ConfirmationAnchor(
        split_audit={"split": "fresh"},
        kernel_identity={"kernel": "frozen"},
    )
    result = {
        "seed": 7,
        "dataset": preset.name,
        "phase": runner.OVERLAP_PHASE,
        "theta": theta.to_dict(),
        "kernel_context_identity": anchor.kernel_identity,
        "split_audit": anchor.split_audit,
        "confirmation_anchor_identity_sha256": runner.clinical._json_sha256(
            anchor.kernel_identity
        ),
    }
    payload = {
        "protocol": runner.PROTOCOL,
        "phase": runner.OVERLAP_PHASE,
        "dataset": preset.name,
        "seed": 7,
        "device": "cuda:0",
        "source_tree_sha256": "source",
        "gate_contract_sha256": "gate",
        "rng_stream_mapping_sha256": "rng",
        "theta_sha256": runner.clinical._json_sha256(theta.to_dict()),
        "result": result,
    }
    monkeypatch.setattr(runner, "_validate_overlap_result", lambda *args: None)
    runner._validate_phase_payload(
        payload,
        phase=runner.OVERLAP_PHASE,
        preset=preset,
        seed=7,
        device="cuda:0",
        theta=theta,
        anchor=anchor,
        source_hash="source",
        gate_contract_sha256="gate",
        rng_mapping_sha256="rng",
    )

    tampered = deepcopy(payload)
    tampered["stage_coverage"] = [0.91]
    with pytest.raises(RuntimeError, match="schema differs"):
        runner._validate_phase_payload(
            tampered,
            phase=runner.OVERLAP_PHASE,
            preset=preset,
            seed=7,
            device="cuda:0",
            theta=theta,
            anchor=anchor,
            source_hash="source",
            gate_contract_sha256="gate",
            rng_mapping_sha256="rng",
        )


def test_confirmation_context_anchor_is_exact(monkeypatch) -> None:
    base = SimpleNamespace(splits=object())
    anchor = runner.ConfirmationAnchor(
        split_audit={"patient_sets_pairwise_disjoint": True},
        kernel_identity={"theta": "frozen"},
    )
    monkeypatch.setattr(
        runner.v2,
        "_split_audit",
        lambda splits: {"patient_sets_pairwise_disjoint": True},
    )
    runner._assert_confirmation_context(anchor, base, {"theta": "frozen"})
    with pytest.raises(RuntimeError, match="differs from confirmation"):
        runner._assert_confirmation_context(anchor, base, {"theta": "changed"})


def test_global_overlap_marker_requires_all_four_exact_statuses(tmp_path: Path) -> None:
    interpretations = {
        dataset: "EMPIRICAL_OVERLAP_SCREEN_PASSED"
        for dataset in runner.DATASETS
    }
    runner._write_global_overlap_marker(tmp_path, interpretations)
    assert runner._valid_global_overlap_marker(tmp_path)

    summary_path = tmp_path / runner.OVERLAP_PHASE / "summary.json"
    complete_path = tmp_path / runner.OVERLAP_PHASE / "COMPLETE"
    summary = runner._read_json(summary_path)
    summary["low_overlap_consequence"] = "ranking allowed"
    runner._write_json(summary_path, summary)
    runner._write_text(
        complete_path,
        f"global-overlap-complete summary_sha256={runner.clinical._json_sha256(summary)}\n",
    )
    assert not runner._valid_global_overlap_marker(tmp_path)


def test_overlap_hard_failure_never_calls_science(monkeypatch, tmp_path: Path) -> None:
    output = _patch_orchestration(monkeypatch, tmp_path)
    calls = []

    def fake_phase(path, *, phase, preset, **kwargs):
        calls.append((phase, preset.name))
        if phase == runner.OVERLAP_PHASE and preset.name == "eicu":
            raise RuntimeError("structural overlap failure")
        if phase == runner.SCIENCE_PHASE:
            raise AssertionError("science must remain locked")
        return [{"seed": 1, "passed": True}]

    monkeypatch.setattr(runner, "_run_phase", fake_phase)
    with pytest.raises(RuntimeError, match="structural overlap failure"):
        runner.run_post_confirmation_science(
            output,
            gates=_orchestration_gates(),
            devices=("cuda:0",),
            resume=False,
        )
    assert all(phase == runner.OVERLAP_PHASE for phase, _ in calls)
    assert not (output / runner.SCIENCE_PHASE).exists()
    assert not (output / runner.OVERLAP_PHASE / "COMPLETE").exists()


def test_low_overlap_unlocks_only_after_all_four_screens(monkeypatch, tmp_path: Path) -> None:
    output = _patch_orchestration(monkeypatch, tmp_path)
    calls = []

    class ScienceStarted(RuntimeError):
        pass

    def fake_phase(path, *, phase, preset, **kwargs):
        calls.append((phase, preset.name))
        if phase == runner.SCIENCE_PHASE:
            raise ScienceStarted("nonformal science boundary reached")
        return [
            {
                "seed": 1,
                "passed": preset.name != runner.DATASETS[0],
            }
        ]

    monkeypatch.setattr(runner, "_run_phase", fake_phase)
    with pytest.raises(ScienceStarted, match="science boundary"):
        runner.run_post_confirmation_science(
            output,
            gates=_orchestration_gates(),
            devices=("cuda:0",),
            resume=False,
        )
    assert calls[: len(runner.DATASETS)] == [
        (runner.OVERLAP_PHASE, dataset) for dataset in runner.DATASETS
    ]
    assert calls[len(runner.DATASETS)][0] == runner.SCIENCE_PHASE
    assert runner._valid_global_overlap_marker(output)
    global_summary = runner._read_json(
        output / runner.OVERLAP_PHASE / "summary.json"
    )
    assert global_summary["datasets"][runner.DATASETS[0]] == (
        "LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
    )


def test_nonformal_tiny_summary_uses_wsc_and_complete_bootstrap(tmp_path: Path) -> None:
    preset = _tiny_preset()
    rows = _tiny_science_rows(preset)
    bootstrap = runner._ensure_bootstrap_artifacts(tmp_path, preset)
    summary = runner._science_summary(
        rows,
        preset=preset,
        interpretation_status="EMPIRICAL_OVERLAP_SCREEN_PASSED",
        bootstrap_contract=bootstrap,
    )
    audit = runner._coverage_audit(
        rows,
        preset=preset,
        summary=summary,
        interpretation_status="EMPIRICAL_OVERLAP_SCREEN_PASSED",
    )

    uniforms = np.load(tmp_path / "bootstrap_uniforms.npy", allow_pickle=False)
    indices = np.load(tmp_path / "bootstrap_indices.npy", allow_pickle=False)
    assert uniforms.shape == (10_000, 20)
    assert indices.shape == (10_000, 20)
    primary = next(row for row in summary["aggregates"] if row["gamma"] == -4.0)
    wsc = primary["methods"]["SC-PCP"]["target_marginal_worst_coverage"]
    mean_seed_minimum = np.mean(
        [
            min(row["methods"]["SC-PCP"]["target_coverage"])
            for row in rows
            if row["gamma"] == -4.0
        ]
    )
    assert wsc == pytest.approx(0.9)
    assert mean_seed_minimum == pytest.approx(0.8)
    assert audit["formula_verified"] is True
    assert len(audit["records"]) == len(runner.GAMMAS) * len(runner.METHODS)


def test_low_overlap_tiny_curves_have_no_ranking(tmp_path: Path) -> None:
    preset = _tiny_preset()
    rows = _tiny_science_rows(preset)
    bootstrap = runner._ensure_bootstrap_artifacts(tmp_path, preset)
    summary = runner._science_summary(
        rows,
        preset=preset,
        interpretation_status="LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
        bootstrap_contract=bootstrap,
    )
    runner._coverage_audit(
        rows,
        preset=preset,
        summary=summary,
        interpretation_status="LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY",
    )
    for aggregate in summary["aggregates"]:
        assert aggregate["analysis_role"] == "descriptive_signed_control_curve"
        assert aggregate["width_order_among_point_eligible"] == []
        assert aggregate["paired_scpcp_comparisons"] == {
            "status": "EXCLUDED_LOW_DONOR_OVERLAP_DESCRIPTIVE_ONLY"
        }
        assert aggregate["universal_ranking_defined"] is False


def test_bootstrap_resume_rejects_tamper_and_missing_without_repair(
    tmp_path: Path,
) -> None:
    preset = _tiny_preset()
    runner._ensure_bootstrap_artifacts(tmp_path, preset)
    uniforms_path = tmp_path / "bootstrap_uniforms.npy"
    uniforms = np.load(uniforms_path, allow_pickle=False)
    uniforms[0, 0] = 1.0 - uniforms[0, 0]
    runner.v2._write_npy(uniforms_path, uniforms)
    with pytest.raises(RuntimeError, match="bootstrap arrays differ"):
        runner._ensure_bootstrap_artifacts(tmp_path, preset)

    empty = tmp_path / "missing"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="missing from a completed root"):
        runner._ensure_bootstrap_artifacts(
            empty,
            preset,
            create_if_missing=False,
        )
    assert list(empty.iterdir()) == []


def test_partial_root_rejects_unknown_or_symlink_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metadata = {
        "source_snapshot": {
            "archive_path": "provenance/source.tar",
            "manifest_path": "provenance/source.json",
        }
    }
    gates = _orchestration_gates()
    monkeypatch.setattr(
        runner,
        "_confirmation_preset",
        lambda gates, dataset: SimpleNamespace(seeds=(1,)),
    )
    (tmp_path / "metadata.json").write_text("{}")
    runner._require_partial_artifact_subset(tmp_path, metadata, gates)
    (tmp_path / "stage_coverage.json").write_text("{}")
    with pytest.raises(RuntimeError, match="unexpected artifact"):
        runner._require_partial_artifact_subset(tmp_path, metadata, gates)
