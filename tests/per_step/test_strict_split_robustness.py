from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

from scpcp.data import TrajectoryBatch
from scpcp.marginal_prefix import MarginalPrefixSelection
import scpcp.strict_split_robustness as strict


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_strict_split_robustness.py"
    name = "test_run_strict_split_robustness"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _batch(start: int, n: int = 4, horizon: int = 2) -> TrajectoryBatch:
    return TrajectoryBatch(
        states=torch.zeros((n, horizon + 1, 1)),
        actions=torch.zeros((n, horizon), dtype=torch.long),
        outcomes=torch.zeros((n, horizon, 1)),
        patient_ids=torch.arange(start, start + n),
    )


def _selection(n: int, horizon: int = 2, candidates: int = 3) -> MarginalPrefixSelection:
    return MarginalPrefixSelection(
        radii=torch.arange(1, horizon + 1, dtype=torch.float32),
        selected_indices=tuple(range(horizon)),
        estimated_coverage=torch.full((horizon,), 0.9, dtype=torch.float64),
        estimated_normalized_width=torch.ones(horizon, dtype=torch.float64),
        effective_sample_size=torch.full((horizon,), float(n), dtype=torch.float64),
        maximum_raw_log_weight=torch.zeros(horizon, dtype=torch.float64),
        raw_log_weight_span=torch.zeros(horizon, dtype=torch.float64),
        candidate_effective_sample_size=torch.full(
            (horizon, candidates), float(n), dtype=torch.float64
        ),
        candidate_estimated_coverage=torch.full(
            (horizon, candidates), 0.9, dtype=torch.float64
        ),
        candidate_estimated_normalized_width=torch.ones(
            (horizon, candidates), dtype=torch.float64
        ),
        candidate_maximum_raw_log_weight=torch.zeros(
            (horizon, candidates), dtype=torch.float64
        ),
        candidate_raw_log_weight_span=torch.zeros(
            (horizon, candidates), dtype=torch.float64
        ),
        selected_endpoint=False,
        failure_stage=None,
    )


def _row(seed: int, canonical_cov: list[float], strict_cov: list[float]) -> dict:
    horizon = len(canonical_cov)
    variants = {}
    for variant, coverage, width, ess in (
        ("canonical", canonical_cov, 2.0, 0.8),
        ("strict", strict_cov, 2.2, 0.7),
    ):
        variants[variant] = {
            "selection_available": True,
            "calibration_roles": ["D_COT", "D_cert"] if variant == "canonical" else ["D_cert"],
            "calibration_trajectories": 10 if variant == "canonical" else 6,
            "radii": [1.0] * horizon,
            "selected_indices": [0] * horizon,
            "selected_endpoint": False,
            "failure_stage": None,
            "estimated_coverage": [0.9] * horizon,
            "estimated_normalized_width": [width] * horizon,
            "selected_ess": [10.0 * ess] * horizon,
            "selected_ess_fraction": [ess] * horizon,
            "selected_minimum_ess_fraction": ess,
            "candidate_ess": [[10.0 * (ess - 0.05)]] * horizon,
            "candidate_ess_fraction": [[ess - 0.05]] * horizon,
            "candidate_minimum_ess_fraction": ess - 0.05,
            "maximum_raw_log_weight": [0.0] * horizon,
            "raw_log_weight_span": [0.0] * horizon,
            "evaluation": {
                "evaluated": True,
                "evaluation_trajectories": 50_000,
                "evaluation_rng": 17,
                "coverage_by_stage": coverage,
                "normalized_width_by_stage": [width] * horizon,
                "mean_normalized_width": width,
            },
        }
    return {
        "protocol": strict.PROTOCOL,
        "setting": "synthetic_main",
        "seed": seed,
        "horizon": horizon,
        "stage_grid_roles": ["D_COT"],
        "stage_grid_sha256": "a" * 64,
        "stage_grid_shape": [horizon, 3],
        "matched_evaluation_crn": True,
        "evaluation_rng": 17,
        "split_sizes": {"D_COT": 4, "D_cert": 6},
        "variants": variants,
    }


def test_frozen_protocol_locks_all_scientific_fields(tmp_path: Path) -> None:
    observed = strict.load_frozen_config()
    assert observed == strict.FROZEN_CONFIG
    changed = json.loads(json.dumps(observed))
    changed["settings"]["controlled_gamma_minus_2"]["gamma"] = -3.0
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(RuntimeError, match="differs from frozen"):
        strict.load_frozen_config(path)


def test_seed_banks_spacing_parent_binding_and_runtime_provenance() -> None:
    runner = _load_runner()
    config = strict.load_frozen_config()
    assert strict.setting_seeds(config, "synthetic_main") == tuple(range(1000, 1100))
    assert strict.setting_seeds(config, "mimic_iv") == tuple(range(20))
    assert strict.setting_seeds(config, "controlled_gamma_minus_2") == tuple(
        range(99000, 99200, 10)
    )
    mapping = runner.controlled_rng_mapping()
    assert len(mapping) == len(set(mapping.values())) == 101
    assert mapping["controlled/base_99000/task"] == 99000
    assert mapping["controlled/base_99000/outcome_model"] == 99001
    assert mapping["controlled/base_99000/behavior_model"] == 99002
    assert mapping["summary/bootstrap"] == 99900
    parent = runner.validate_parent_snapshot()
    assert parent["manifest_sha256"] == (
        "e6a1bba7f3be47d39357f212824e7720262e7d5212a14628e3b8981088c64e24"
    )
    assert parent["archive_sha256"] == (
        "2116b9929240d8a25092f3c9015c362957bc963532930e5e720dc7ec78b2ea0b"
    )
    environment = runner.runtime_environment()
    assert environment["python"]["version"]
    assert environment["numpy"]["version"]
    assert environment["numpy"]["blas"]
    assert environment["torch"]["version"]


def test_manifest_is_json_normalized_before_resume_comparison() -> None:
    runner = _load_runner()
    config = strict.load_frozen_config()
    manifest = runner.build_manifest(
        protocol_config=config,
        parent_snapshot={"role": "test"},
        source_hash="a" * 64,
        config_contract={
            "controlled_active_config": {
                "seeds": (0, 1),
                "devices": ("cuda:0", "cuda:1"),
            }
        },
        environment={"python": {"version": "test"}},
        devices=("cuda:0", "cuda:1"),
        device_mapping={
            ("synthetic_main", 1000): "cuda:0",
            ("mimic_iv", 0): "cuda:1",
        },
        seed_audit={"status": "test"},
        argv=("runner.py", "--resume"),
        created_at_utc="2026-08-26T00:00:00+00:00",
    )

    persisted = json.loads(json.dumps(manifest))
    assert manifest == persisted
    assert manifest["active_config_contract"]["controlled_active_config"] == {
        "seeds": [0, 1],
        "devices": ["cuda:0", "cuda:1"],
    }


def test_rng_audit_ignores_reservation_declarations_but_rejects_actual_use(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    artifacts = tmp_path / "artifacts"
    source = tmp_path / "source"
    artifacts.mkdir()
    (source / "scripts").mkdir(parents=True)
    (source / "scripts" / "coordinator.py").write_text(
        "EXTERNAL_SEED_RESERVATIONS = {\n"
        "    'strict_split_audit': range(99000, 100000),\n"
        "    'score_robustness': range(100000, 101000),\n"
        "}\n"
    )
    audit = runner.audit_fresh_controlled_rng_ids(
        output_dir=tmp_path / "new",
        artifact_root=artifacts,
        source_root=source,
    )
    assert audit["status"] == "passed_before_launch"
    assert audit["formal_rng_id_count"] == 101
    assert audit["source_actual_use_excludes_reservation_declarations"]

    collision = next(iter(runner.controlled_rng_mapping().values()))
    (source / "scripts" / "prior.py").write_text(f"OLD_RNG = {collision}\n")
    with pytest.raises(RuntimeError, match="collide"):
        runner.audit_fresh_controlled_rng_ids(
            output_dir=tmp_path / "new",
            artifact_root=artifacts,
            source_root=source,
        )


def test_one_grid_is_shared_while_only_selection_batch_changes(monkeypatch) -> None:
    cot = _batch(0, n=3)
    cert = _batch(10, n=5)
    cot_scores = torch.ones((3, 2))
    cert_scores = torch.ones((5, 2))
    grids = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    calls = []

    def fake_selector(batch, scores, **kwargs):
        calls.append((batch.n, scores.shape[0], kwargs["stage_grids"].clone()))
        return _selection(batch.n)

    monkeypatch.setattr(strict, "select_marginal_prefix_schedule", fake_selector)
    results = strict.select_strict_split_pair(
        cot_batch=cot,
        cot_scores=cot_scores,
        certification_batch=cert,
        certification_scores=cert_scores,
        stage_grids=grids,
        target_policy=object(),
        logging_policy=object(),
        outcome_model=object(),
        outcome_sd=torch.ones(1),
        target=0.9,
    )
    assert set(results) == set(strict.VARIANTS)
    assert [(n, score_n) for n, score_n, _ in calls] == [(8, 8), (5, 5)]
    assert torch.equal(calls[0][2], grids)
    assert torch.equal(calls[1][2], grids)


def test_patient_overlap_is_rejected_before_selection() -> None:
    batch = _batch(0)
    with pytest.raises(RuntimeError, match="patient identifiers overlap"):
        strict.select_strict_split_pair(
            cot_batch=batch,
            cot_scores=torch.ones((4, 2)),
            certification_batch=batch,
            certification_scores=torch.ones((4, 2)),
            stage_grids=torch.ones((2, 3)),
            target_policy=object(),
            logging_policy=object(),
            outcome_model=object(),
            outcome_sd=torch.ones(1),
            target=0.9,
        )


def test_summary_uses_min_stage_of_seed_mean_and_joint_paired_bootstrap() -> None:
    rows = [
        _row(1, [0.8, 1.0], [0.9, 1.0]),
        _row(2, [1.0, 0.8], [1.0, 0.9]),
    ]
    summary = strict.summarize_setting(
        rows,
        setting="synthetic_main",
        seeds=(1, 2),
        bootstrap_resamples=200,
        bootstrap_rng=123,
    )
    assert summary["variants"]["canonical"]["target_marginal_worst_coverage"] == pytest.approx(0.9)
    assert summary["variants"]["strict"]["target_marginal_worst_coverage"] == pytest.approx(0.95)
    paired = summary["paired_strict_vs_canonical"]
    assert paired["joint_available_seeds"] == 2
    assert paired["strict_minus_canonical_wsc"] == pytest.approx(0.05)
    assert paired["strict_to_canonical_geometric_width_ratio"] == pytest.approx(1.1)
    assert paired["strict_minus_canonical_selected_minimum_ess_fraction"] == pytest.approx(-0.1)


def test_atomic_task_artifact_is_fail_closed(tmp_path: Path) -> None:
    runner = _load_runner()
    root = tmp_path / "study"
    (root / "synthetic_main").mkdir(parents=True)
    row = _row(1, [0.9, 0.9], [0.9, 0.9])
    destination = runner.write_task_artifact(
        row,
        output_dir=root,
        task=("synthetic_main", 1),
        device="cuda:0",
        manifest_hash="b" * 64,
        source_hash="c" * 64,
        config_contract_hash="d" * 64,
        environment_hash="e" * 64,
    )
    loaded = runner._load_valid_artifact(
        destination,
        setting="synthetic_main",
        seed=1,
        expected_device="cuda:0",
        manifest_hash="b" * 64,
        source_hash="c" * 64,
        config_contract_hash="d" * 64,
        environment_hash="e" * 64,
    )
    assert loaded == row
    metadata = json.loads((destination / "metadata.json").read_text())
    metadata["device"] = "cuda:1"
    (destination / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="metadata contract differs"):
        runner._load_valid_artifact(
            destination,
            setting="synthetic_main",
            seed=1,
            expected_device="cuda:0",
            manifest_hash="b" * 64,
            source_hash="c" * 64,
            config_contract_hash="d" * 64,
            environment_hash="e" * 64,
        )


def _empty_scan_root(tmp_path: Path) -> Path:
    root = tmp_path / "scan"
    root.mkdir()
    for setting in strict.SETTINGS:
        (root / setting).mkdir()
    (root / "manifest.json").write_text("{}\n")
    return root


@pytest.mark.parametrize(
    ("kind", "name"),
    (
        ("directory", ".seed_01000-abandoned"),
        ("file", "notes.txt"),
        ("directory", "cache"),
    ),
)
def test_setting_scan_rejects_every_unexpected_child(
    tmp_path: Path,
    kind: str,
    name: str,
) -> None:
    runner = _load_runner()
    root = _empty_scan_root(tmp_path)
    child = root / "synthetic_main" / name
    child.mkdir() if kind == "directory" else child.write_text("unexpected\n")

    with pytest.raises(RuntimeError, match="unexpected strict-split setting child"):
        runner._reject_unexpected_artifacts(
            root,
            (("synthetic_main", 1000),),
        )


@pytest.mark.parametrize("kind", ("file", "directory"))
def test_root_scan_rejects_every_child_outside_allowlist(
    tmp_path: Path,
    kind: str,
) -> None:
    runner = _load_runner()
    root = _empty_scan_root(tmp_path)
    child = root / "untracked"
    child.mkdir() if kind == "directory" else child.write_text("unexpected\n")

    with pytest.raises(RuntimeError, match="unexpected strict-split root child"):
        runner._reject_unexpected_artifacts(
            root,
            (("synthetic_main", 1000),),
        )


def test_expected_seed_name_must_be_a_real_directory(tmp_path: Path) -> None:
    runner = _load_runner()
    root = _empty_scan_root(tmp_path)
    (root / "synthetic_main" / "seed_01000").write_text("not a directory\n")

    with pytest.raises(RuntimeError, match="seed artifact is not a real directory"):
        runner._reject_unexpected_artifacts(
            root,
            (("synthetic_main", 1000),),
        )
