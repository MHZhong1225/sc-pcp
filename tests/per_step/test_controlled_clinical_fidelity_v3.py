from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from scpcp.controlled_clinical_fidelity_v3 import (
    DATASETS,
    CandidateDatasetSummary,
    KernelTheta,
    load_fidelity_v3_config,
    normalized_seed_ratio,
    select_dataset_candidate,
    select_shared_candidate,
    stage_a_candidates,
    stage_b_candidates,
    summarize_candidate_dataset,
)
from scpcp.controlled_transition import ControlledResidualEnvironment
from scpcp.data import TrajectoryBatch


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/controlled_clinical_fidelity_v3.yaml"


def _load_runner():
    path = ROOT / "scripts/run_controlled_clinical_fidelity_v3.py"
    spec = importlib.util.spec_from_file_location(
        "run_controlled_clinical_fidelity_v3",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class _Representation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def representation(self, states: torch.Tensor) -> torch.Tensor:
        return states[:, :2] + 0.0 * self.anchor

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del actions
        mean = states[:, :2] + 0.0 * self.anchor
        return mean, torch.ones_like(mean)


def _batch() -> TrajectoryBatch:
    current = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [10.0, 10.0],
            [12.0, 12.0],
        ]
    )
    successor = current + 1.0
    return TrajectoryBatch(
        states=torch.stack((current, successor), dim=1),
        actions=torch.tensor([[0], [0], [1], [1]]),
        outcomes=successor[:, None, :],
        patient_ids=torch.arange(4),
    )


def _environment(**overrides: object) -> ControlledResidualEnvironment:
    options = {
        "outcome_model": _Representation(),
        "n_actions": 2,
        "difficulty": torch.zeros((4, 1)),
        "history_length": 1,
        "neighbors": 2,
        "bandwidth": 2.0,
        "ridge": 1e-3,
    }
    options.update(overrides)
    return ControlledResidualEnvironment(_batch(), **options)


def _metrics(ratio: float, *, structural: bool = True) -> dict[str, object]:
    return {
        "maximum_score_ks": 0.10 * ratio,
        "maximum_signed_residual_w1": 0.20 * ratio,
        "maximum_successor_mean_w1": 0.20 * ratio,
        "maximum_successor_q95_w1": 0.40 * ratio,
        "structural_invariants": structural,
    }


def _summary(
    candidate: KernelTheta,
    dataset: str,
    pass_count: int,
    q95: float,
    mean: float,
) -> CandidateDatasetSummary:
    return CandidateDatasetSummary(
        candidate_id=candidate.theta_id,
        dataset=dataset,
        pass_count=pass_count,
        q95_seed_ratio=q95,
        mean_seed_ratio=mean,
        seed_ratios=tuple([mean] * 20),
    )


def _matrix(
    candidates: tuple[KernelTheta, ...],
    *,
    pass_count: int = 19,
) -> dict[str, dict[str, CandidateDatasetSummary]]:
    return {
        candidate.theta_id: {
            dataset: _summary(candidate, dataset, pass_count, 0.9, 0.8)
            for dataset in DATASETS
        }
        for candidate in candidates
    }


def test_config_freezes_development_confirmation_and_firewall() -> None:
    config = load_fidelity_v3_config(CONFIG_PATH)

    assert config.development_seeds["eicu"] == tuple(range(92_000, 92_200, 10))
    assert config.confirmation_seeds == {
        "mimic_iv": tuple(range(111_000, 111_200, 10)),
        "eicu": tuple(range(112_000, 112_200, 10)),
        "inspire": tuple(range(113_000, 113_200, 10)),
        "mimic_cxr": tuple(range(114_000, 114_200, 10)),
    }
    assert config.stagewise_sd_floor == 1e-4
    assert set().union(*map(set, config.development_seeds.values())).isdisjoint(
        set().union(*map(set, config.confirmation_seeds.values()))
    )
    mapping = runner._seed_device_mapping(
        config.confirmation_seeds,
        ("cuda:0", "cuda:1"),
    )
    assert mapping["mimic_iv/base_111000"] == "cuda:0"
    assert mapping["mimic_iv/base_111010"] == "cuda:1"
    assert mapping["mimic_cxr/base_114190"] == "cuda:1"


def test_config_rejects_post_freeze_candidate_change(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    payload["stage_a"]["ridge"]["value"] = 0.01
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="Stage-A ridge"):
        load_fidelity_v3_config(changed)


def test_fresh_roots_and_collision_status_are_fail_closed(tmp_path: Path) -> None:
    assert runner.DEVELOPMENT_ROOT.name == (
        "controlled_clinical_fidelity_v3_development"
    )
    assert runner.CONFIRMATION_ROOT.name == (
        "controlled_clinical_fidelity_v3_confirmation"
    )
    root = tmp_path / "must_not_open"
    with pytest.raises(RuntimeError, match="collision audit did not pass"):
        runner._prepare_root(
            root,
            {"confirmation_rng_audit": {"status": "collision"}},
            {},
            resume=False,
        )
    assert not root.exists()


def test_recursive_result_firewall_rejects_scientific_keys() -> None:
    runner._validate_seed_payload_firewall(
        {"result": {"coverage_generated": False}}
    )
    runner._validate_seed_payload_firewall(
        {
            "result": {
                "coverage_generated": False,
                "candidates": [
                    {
                        "theta": {"bandwidth": 2.0},
                        "context_identity": {"theta": {"bandwidth": 2.0}},
                    }
                ],
            }
        }
    )
    with pytest.raises(RuntimeError, match="coverage firewall"):
        runner._reject_scientific_result_keys(
            {"nested": [{"stage_coverage": [0.9]}]}
        )
    with pytest.raises(RuntimeError, match="coverage firewall"):
        runner._validate_seed_payload_firewall(
            {"result": {"coverage_generated": False, "bandwidth": 2.0}}
        )
    with pytest.raises(RuntimeError, match="coverage firewall"):
        runner._validate_seed_payload_firewall(
            {
                "result": {
                    "coverage_generated": False,
                    "kernel": {"bandwidth": 2.0},
                }
            }
        )
    with pytest.raises(RuntimeError, match="coverage firewall"):
        runner._validate_seed_payload_firewall(
            {
                "result": {
                    "coverage_generated": False,
                    "nested": {"coverage_generated": [0.91]},
                }
            }
        )


@pytest.mark.parametrize(
    "forbidden_key",
    ("stage_coverage", "mean_width", "method_selection", "science"),
)
def test_seed_payload_firewall_rejects_top_level_science_tamper(
    forbidden_key: str,
) -> None:
    protocol = runner.v2.load_extension_config(runner.V2_CONFIG_PATH)
    preset = protocol.datasets["eicu"]
    candidate_hash = "c" * 64
    payload = {
        "protocol": runner.PROTOCOL,
        "phase": "development_stage_a",
        "dataset": "eicu",
        "seed": preset.seeds[0],
        "device": "cuda:0",
        "source_tree_sha256": "a" * 64,
        "candidate_contract_sha256": candidate_hash,
        "result": {
            "seed": preset.seeds[0],
            "dataset": "eicu",
            "coverage_generated": False,
        },
        forbidden_key: [],
    }

    with pytest.raises(RuntimeError, match="coverage firewall"):
        runner._validate_seed_payload(
            payload,
            phase="development_stage_a",
            preset=preset,
            seed=preset.seeds[0],
            device="cuda:0",
            source_hash="a" * 64,
            candidate_hash=candidate_hash,
        )


def test_seed_payload_rejects_unknown_top_level_key() -> None:
    protocol = runner.v2.load_extension_config(runner.V2_CONFIG_PATH)
    preset = protocol.datasets["eicu"]
    payload = {
        "protocol": runner.PROTOCOL,
        "phase": "development_stage_a",
        "dataset": "eicu",
        "seed": preset.seeds[0],
        "device": "cuda:0",
        "source_tree_sha256": "a" * 64,
        "candidate_contract_sha256": "c" * 64,
        "result": {
            "seed": preset.seeds[0],
            "dataset": "eicu",
            "coverage_generated": False,
        },
        "unexpected_note": "tamper",
    }

    with pytest.raises(RuntimeError, match="top-level schema"):
        runner._validate_seed_payload(
            payload,
            phase="development_stage_a",
            preset=preset,
            seed=preset.seeds[0],
            device="cuda:0",
            source_hash="a" * 64,
            candidate_hash="c" * 64,
        )


def test_stage_a_grid_has_exact_order_and_v2_anchor() -> None:
    candidates = stage_a_candidates()

    assert len(candidates) == 12
    assert candidates[0].theta_id == "A00_raw_k100_gaussian_b2__raw_ridge_1e-3"
    assert candidates[-1].theta_id == (
        "A11_stagewise_zscore_k200_uniform__raw_ridge_1e-3"
    )
    assert [candidate.metric for candidate in candidates[:6]] == ["raw"] * 6
    assert [candidate.ridge for candidate in stage_b_candidates(candidates[5])] == [
        "raw_ridge_1e-3",
        "normalized_ridge_1e-3",
        "normalized_ridge_1e-2",
    ]


def test_stagewise_zscore_is_pooled_over_actions_in_float64() -> None:
    environment = _environment(representation_geometry="stagewise_zscore")

    center, scale = environment._metric_transforms[0]
    expected = _batch().states[:, 0].to(torch.float64)
    assert center.dtype == torch.float64
    assert scale.dtype == torch.float64
    assert torch.equal(center, expected.mean(dim=0))
    assert torch.equal(scale, expected.std(dim=0, unbiased=False).clamp_min(1e-4))
    action_zero_representation = environment._libraries[(0, 0)][0]
    expected_action_zero = (
        _batch().states[:2, 0] - center.to(torch.float32)
    ) / scale.to(torch.float32)
    assert torch.equal(action_zero_representation, expected_action_zero)
    assert not torch.allclose(action_zero_representation.mean(dim=0), torch.zeros(2))


def test_uniform_weighting_and_normalized_ridge_contract() -> None:
    environment = _environment(donor_weighting="uniform")
    distances = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    assert torch.equal(environment._base_donor_logits(distances), torch.zeros_like(distances))

    features = torch.tensor([[1.0, 0.0], [1.0, 2.0], [1.0, 4.0]])
    target = torch.tensor([[1.0], [2.0], [5.0]])
    coefficient = ControlledResidualEnvironment._fit_ridge(
        features,
        target,
        1e-2,
        mode="sample_normalized_no_intercept",
    )
    penalty = torch.diag(torch.tensor([0.0, 1.0]))
    expected = torch.linalg.solve(
        features.T @ features / 3 + 1e-2 * penalty,
        features.T @ target / 3,
    )
    assert torch.allclose(coefficient, expected)


def test_normalized_seed_ratio_and_linear_q95_are_exact() -> None:
    assert normalized_seed_ratio(_metrics(0.8)) == pytest.approx(0.8)
    assert math.isinf(normalized_seed_ratio(_metrics(0.1, structural=False)))
    metrics = [_metrics(value) for value in np.linspace(0.1, 1.0, 20)]
    summary = summarize_candidate_dataset("candidate", "eicu", metrics)

    expected = float(np.quantile(np.linspace(0.1, 1.0, 20), 0.95, method="linear"))
    assert summary.pass_count == 20
    assert summary.q95_seed_ratio == pytest.approx(expected)
    failed = summarize_candidate_dataset(
        "candidate",
        "eicu",
        [*metrics[:-1], _metrics(0.1, structural=False)],
    )
    assert failed.pass_count == 19
    assert math.isinf(failed.q95_seed_ratio)
    assert failed.to_dict()["q95_seed_ratio"] is None


def test_selectors_follow_frozen_lexicographic_order_and_record_ties() -> None:
    candidates = stage_a_candidates()[:2]
    summaries = _matrix(candidates)
    summaries[candidates[0].theta_id]["eicu"] = _summary(
        candidates[0], "eicu", 18, 0.8, 0.7
    )
    shared = select_shared_candidate(candidates, summaries)
    fallback = select_dataset_candidate("mimic_iv", candidates, summaries)

    assert shared["winner"]["theta_id"] == candidates[1].theta_id
    assert fallback["winner"]["theta_id"] == candidates[0].theta_id
    assert fallback["substantive_ties_before_candidate_index"] == [
        candidates[0].theta_id
    ]


def test_development_decision_prefers_admissible_shared_theta() -> None:
    stage_a = stage_a_candidates()
    stage_a_summaries = _matrix(stage_a, pass_count=20)
    shared_a = stage_a[1]
    fallback_a = {dataset: stage_a[2] for dataset in DATASETS}
    candidates = stage_b_candidates(shared_a)
    stage_b_summaries = _matrix(candidates, pass_count=18)
    winner = candidates[1]
    for dataset in DATASETS:
        stage_b_summaries[winner.theta_id][dataset] = _summary(
            winner,
            dataset,
            20 if dataset == "mimic_iv" else 19,
            0.9,
            0.8,
        )
    for dataset in DATASETS:
        for candidate in stage_b_candidates(fallback_a[dataset]):
            stage_b_summaries.setdefault(candidate.theta_id, {})[dataset] = _summary(
                candidate, dataset, 19, 0.95, 0.9
            )

    decision = runner._development_decision(
        stage_a=stage_a,
        stage_a_summaries=stage_a_summaries,
        shared_a=shared_a,
        fallback_a=fallback_a,
        stage_b_summaries=stage_b_summaries,
    )

    assert decision["status"] == "DEVELOPMENT_GO_SHARED"
    assert set(value["theta_id"] for value in decision["theta_by_dataset"].values()) == {
        winner.theta_id
    }


def test_development_decision_uses_precomputed_fallback_or_no_go() -> None:
    stage_a = stage_a_candidates()
    stage_a_summaries = _matrix(stage_a, pass_count=20)
    shared_a = stage_a[1]
    fallback_a = {dataset: stage_a[2] for dataset in DATASETS}
    stage_b_summaries = _matrix(stage_b_candidates(shared_a), pass_count=18)
    repaired = ("eicu", "inspire", "mimic_cxr")
    for dataset in repaired:
        for candidate in stage_b_candidates(fallback_a[dataset]):
            stage_b_summaries.setdefault(candidate.theta_id, {})[dataset] = _summary(
                candidate,
                dataset,
                19,
                0.9,
                0.8,
            )
    decision = runner._development_decision(
        stage_a=stage_a,
        stage_a_summaries=stage_a_summaries,
        shared_a=shared_a,
        fallback_a=fallback_a,
        stage_b_summaries=stage_b_summaries,
    )
    assert decision["status"] == "DEVELOPMENT_GO_DATASET_SPECIFIC"
    assert decision["theta_by_dataset"]["mimic_iv"]["theta_id"] == stage_a[0].theta_id

    failed_id = stage_b_candidates(fallback_a[repaired[0]])[0].theta_id
    for candidate_id in list(stage_b_summaries):
        if repaired[0] in stage_b_summaries[candidate_id]:
            stage_b_summaries[candidate_id][repaired[0]] = CandidateDatasetSummary(
                candidate_id=candidate_id,
                dataset=repaired[0],
                pass_count=18,
                q95_seed_ratio=1.1,
                mean_seed_ratio=1.0,
                seed_ratios=tuple([1.0] * 20),
            )
    assert failed_id in stage_b_summaries
    no_go = runner._development_decision(
        stage_a=stage_a,
        stage_a_summaries=stage_a_summaries,
        shared_a=shared_a,
        fallback_a=fallback_a,
        stage_b_summaries=stage_b_summaries,
    )
    assert no_go["status"] == "DEVELOPMENT_NO_GO"
    assert no_go["theta_by_dataset"] == {}


def test_parent_firewall_and_actual_rng_audit() -> None:
    config = load_fidelity_v3_config(CONFIG_PATH)
    with pytest.raises(RuntimeError, match="information firewall"):
        runner._read_parent_bytes(
            ROOT / config.parent_v2_root,
            Path("mimic_iv/science/summary.json"),
        )

    binding = runner.verify_parent_v2((ROOT / config.parent_v2_root).resolve())
    audit = runner.audit_confirmation_rng(config)
    assert binding["information_firewall"]["science_or_coverage_files_opened"] is False
    assert set(binding["parent_dataset_contracts"]) == set(DATASETS)
    assert binding["parent_dataset_contracts_sha256"] == runner._json_sha256(
        binding["parent_dataset_contracts"]
    )
    assert audit["status"] == "passed_before_launch"
    assert audit["collision_count"] == 0
    assert audit["new_rng_stream_count"] == 1304
    assert audit["new_rng_stream_mapping_sha256"] == (
        "2e34aefe0adeae8cfc3499ef3481da09c94f0510b2c044c156ac280344ec6ab7"
    )


@pytest.mark.parametrize(
    ("dataset", "path"),
    (
        ("eicu", ("base_config_sha256",)),
        ("inspire", ("raw_clinical_cache", "sha256")),
        (
            "mimic_cxr",
            ("mimic_cxr_sources", "pretrained_checkpoint_sha256"),
        ),
    ),
)
def test_parent_dataset_contract_tamper_is_fail_closed(
    dataset: str,
    path: tuple[str, ...],
) -> None:
    metadata = json.loads(
        (ROOT / "results/work/controlled_clinical_extension_v2/metadata.json").read_text()
    )
    live = copy.deepcopy(metadata["dataset_contracts"])
    target = live[dataset]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "0" * 64

    with pytest.raises(RuntimeError, match="base config/raw cache/CXR"):
        runner._validate_live_dataset_contracts(metadata, live)


def test_confirmation_rejects_parent_binding_drift() -> None:
    current = {"binding": "current"}
    current_hash = runner._json_sha256(current)
    metadata = {
        "parent_v2_binding": current,
        "parent_v2_binding_sha256": current_hash,
    }
    frozen = {"parent_v2_binding_sha256": current_hash}
    runner._validate_development_parent_binding(metadata, frozen, current)

    with pytest.raises(RuntimeError, match="differs at confirmation"):
        runner._validate_development_parent_binding(
            metadata,
            frozen,
            {"binding": "changed"},
        )


def test_runner_exposes_no_science_or_coverage_phase() -> None:
    assert runner.PHASES == ("audit", "development", "confirmation")
    assert not hasattr(runner, "run_science")
    source = (ROOT / "scripts/run_controlled_clinical_fidelity_v3.py").read_text()
    assert "run_science_seed(" not in source
