from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from scpcp.data import TrajectoryBatch


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "run_controlled_prefix_ablations.py"
    spec = importlib.util.spec_from_file_location(
        "run_controlled_prefix_ablations",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _UniformLoggingPolicy:
    def probabilities(self, states: torch.Tensor) -> torch.Tensor:
        return torch.full((len(states), 2), 0.5, device=states.device)


class _RadiusTargetPolicy:
    def probabilities_for_grid(
        self,
        states: torch.Tensor,
        radii: torch.Tensor,
    ) -> torch.Tensor:
        probability_zero = 0.2 + 0.2 * radii
        probabilities = torch.stack(
            (probability_zero, 1.0 - probability_zero),
            dim=1,
        )
        return probabilities[None, :, :].expand(len(states), -1, -1)


class _UnitScaleModel:
    def __call__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((len(states), 1), device=states.device),
            torch.ones((len(states), 1), device=states.device),
        )


def _batch() -> TrajectoryBatch:
    actions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    return TrajectoryBatch(
        states=torch.zeros((4, 3, 1)),
        actions=actions,
        outcomes=torch.zeros((4, 2, 1)),
        patient_ids=torch.arange(4),
    )


def test_ratio_ablations_delete_only_the_declared_ratio_component() -> None:
    runner = _load_runner()
    scores = torch.tensor(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [1.5, 1.5],
            [1.5, 1.5],
        ]
    )
    grids = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    common = {
        "stage_grids": grids,
        "target_policy": _RadiusTargetPolicy(),
        "logging_policy": _UniformLoggingPolicy(),
        "outcome_model": _UnitScaleModel(),
        "outcome_sd": torch.ones(1),
        "target": 0.45,
    }

    omit = runner.select_ratio_ablation_schedule(
        _batch(),
        scores,
        mode="omit_current",
        **common,
    )
    current = runner.select_ratio_ablation_schedule(
        _batch(),
        scores,
        mode="current_only",
        **common,
    )

    assert torch.equal(omit.radii, torch.tensor([1.0, 2.0]))
    assert torch.allclose(
        omit.estimated_coverage,
        torch.tensor([0.5, 1.0], dtype=torch.float64),
    )
    # The stage-zero ratio selected above is committed before stage one.  If it
    # were discarded, radius 1.0 would have coverage 0.50 and remain feasible.
    assert torch.equal(current.radii, torch.tensor([2.0, 1.0]))
    assert float(current.estimated_coverage[1]) == pytest.approx(0.50)


def test_summary_uses_minimum_of_stagewise_seed_means() -> None:
    runner = _load_runner()
    seeds = (11, 13)
    coverage = {
        11: [0.80, 0.95, *([0.95] * 10)],
        13: [0.90, 0.85, *([0.95] * 10)],
    }
    rows = []
    for seed in seeds:
        methods = {}
        for offset, method in enumerate(runner.METHODS):
            methods[method] = {
                "selection_available": True,
                "selection_selected_endpoint": False,
                "selection_effective_sample_size": [2_000.0] * 12,
                "selection_minimum_candidate_effective_sample_size": 1_500.0,
                "target_coverage": coverage[seed],
                "target_normalized_width": [2.0 + 0.1 * offset] * 12,
                "reference_prefix_ess_fraction": [0.5] * 12,
                "target_q90_to_radius_ratio": [1.0] * 12,
            }
        rows.append({"seed": seed, "gamma": 0.0, "methods": methods})

    summary = runner.summarize(rows, seeds=seeds, gammas=(0.0,))
    full = summary["aggregates"][0]["methods"][runner.FULL_PREFIX]

    assert full["target_coverage_by_stage"][:2] == pytest.approx([0.85, 0.90])
    assert full["target_marginal_worst_coverage"] == pytest.approx(0.85)
    assert full["target_marginal_worst_coverage"] != pytest.approx(
        np.mean([0.80, 0.85])
    )
    assert summary["primary_metric"] == "min_t mean_seed(target_coverage_seed_t)"


def test_resume_rejects_malformed_or_provenance_mismatched_seed(tmp_path: Path) -> None:
    runner = _load_runner()
    seed = runner.CONFIRM_SEEDS[0]
    contract = {
        "protocol": runner.PROTOCOL,
        "source_tree_sha256": "active-source",
        "parent_metadata_sha256": "parent-metadata",
        "parent_summary_sha256": "parent-summary",
        "parent_seed_bundle_sha256": "parent-seeds",
    }
    rows = [
        {
            "seed": seed,
            "gamma": gamma,
            "methods": {method: {} for method in runner.METHODS},
        }
        for gamma in runner.GAMMAS
    ]
    path = tmp_path / f"seed_{seed:05d}.json"
    runner._write_json(path, {**contract, "seed": seed, "rows": rows})

    assert runner._completed_seeds(
        tmp_path,
        seeds=(seed,),
        seed_contract=contract,
    ) == {seed}

    payload = json.loads(path.read_text())
    payload["source_tree_sha256"] = "different-source"
    runner._write_json(path, payload)
    with pytest.raises(RuntimeError, match="malformed or mismatched"):
        runner._completed_seeds(
            tmp_path,
            seeds=(seed,),
            seed_contract=contract,
        )


def test_parent_seed_bundle_hash_covers_names_and_contents(tmp_path: Path) -> None:
    runner = _load_runner()
    seeds = (1, 3)
    for seed in seeds:
        (tmp_path / f"seed_{seed:05d}.json").write_text(str(seed))
    first = runner._seed_bundle_sha256(tmp_path, seeds)
    (tmp_path / "seed_00003.json").write_text("changed")
    second = runner._seed_bundle_sha256(tmp_path, seeds)

    assert first != second
