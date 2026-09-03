from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpcp.conservatism_decomposition import (
    FreshEvaluation,
    canonical_fresh_records,
    load_decomposition_schedules,
    load_standard_decomposition,
    log_ratio_decomposition,
    recover_standard_decomposition,
    select_ordered_point_index,
)


def _surfaces() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    grid = np.arange(1, 7, dtype=np.float32)
    profile = np.array([1.0, 2.0], dtype=np.float32)
    candidates = grid[:, None] * profile[None, :]
    oracle_coverage = np.array(
        [
            [0.70, 0.80],
            [0.85, 0.95],
            [0.90, 0.91],
            [0.93, 0.94],
            [0.95, 0.96],
            [0.97, 0.98],
        ],
        dtype=np.float32,
    )
    phase0 = {
        "standard_profiled_scale_grid": grid.copy(),
        "standard_profile": profile.copy(),
        "standard_profiled_candidate_schedules": candidates.copy(),
        "standard_profiled_candidate_coverage": oracle_coverage,
        "standard_profiled_candidate_normalized_width": np.array(
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
            dtype=np.float32,
        ),
        "standard_profiled_selected_schedule": candidates[2].copy(),
        "standard_greedy_stage_grids": np.array(
            [[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]],
            dtype=np.float32,
        ),
        "standard_greedy_selected_schedule": np.array([3, 8], dtype=np.float32),
    }
    paper = {
        "scale_grid": grid.copy(),
        "stage_profile": profile.copy(),
        "candidate_radii": candidates.copy(),
        "cot_diagonal": np.array(
            [
                [0.70, 0.80],
                [0.85, 0.95],
                [0.91, 0.92],
                [0.92, 0.93],
                [0.94, 0.95],
                [0.96, 0.97],
            ],
            dtype=np.float32,
        ),
        "cot_lower_bounds": np.array(
            [
                [0.65, 0.75],
                [0.80, 0.88],
                [0.85, 0.86],
                [0.89, 0.91],
                [0.91, 0.92],
                [0.94, 0.95],
            ],
            dtype=np.float32,
        ),
        "estimated_candidate_widths": np.array(
            [1.0, 2.0, 3.0, 2.5, 4.0, 5.0], dtype=np.float32
        ),
        "scpcp_selected_radii": candidates[4].copy(),
    }
    return phase0, paper


def test_recovers_strict_standard_acde_selection() -> None:
    phase0, paper = _surfaces()

    result = recover_standard_decomposition(phase0, paper)

    assert result.indices.a == (2, 3)
    assert result.indices.c == 2
    assert result.indices.d == 3
    assert result.indices.e == 4
    assert result.diagnostics.d_ordered_prefix == (5, 4, 3, 2)
    assert result.diagnostics.d_stopped_index == 1
    assert result.diagnostics.e_ordered_prefix == (5, 4)
    assert result.diagnostics.e_stopped_index == 3
    assert np.array_equal(result.schedules.c, paper["candidate_radii"][2])
    assert np.array_equal(result.schedules.d, paper["candidate_radii"][3])
    assert np.array_equal(result.schedules.e, paper["candidate_radii"][4])
    assert not result.schedules.a.flags.writeable


def test_npz_loader_disables_pickle_and_matches_mapping_recovery(tmp_path: Path) -> None:
    phase0, paper = _surfaces()
    phase0_path = tmp_path / "phase0.npz"
    paper_path = tmp_path / "paper.npz"
    np.savez(phase0_path, **phase0)
    np.savez(paper_path, **paper)

    loaded = load_standard_decomposition(phase0_path, paper_path)
    direct = recover_standard_decomposition(phase0, paper)

    assert loaded.indices == direct.indices
    assert np.array_equal(loaded.schedules.d, direct.schedules.d)


def test_runner_loader_uses_fixed_layer_keys_and_tensor_schedules(tmp_path: Path) -> None:
    phase0, paper = _surfaces()
    phase0_dir = tmp_path / "phase0" / "seed_00007"
    paper_dir = tmp_path / "paper" / "seed_00007"
    phase0_dir.mkdir(parents=True)
    paper_dir.mkdir(parents=True)
    np.savez(phase0_dir / "surfaces.npz", **phase0)
    np.savez(paper_dir / "surfaces.npz", **paper)

    loaded = load_decomposition_schedules(
        phase0_dir,
        paper_dir,
        alpha=0.10,
    )

    expected_layers = {
        "A_sequential",
        "C_profiled_oracle",
        "D_cot_point",
        "E_lcb",
    }
    assert set(loaded.schedules) == expected_layers
    assert set(loaded.indices) == expected_layers
    assert loaded.indices == {
        "A_sequential": None,
        "C_profiled_oracle": 2,
        "D_cot_point": 3,
        "E_lcb": 4,
    }
    assert loaded.schedules["D_cot_point"].tolist() == [4.0, 8.0]
    assert loaded.diagnostics["A_stage_indices"] == (2, 3)
    assert loaded.diagnostics["target"] == pytest.approx(0.90)


def test_ordered_point_selector_stops_before_later_narrow_safe_island() -> None:
    selected = select_ordered_point_index(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([[0.95], [0.95], [0.85], [0.95]]),
        np.array([1.0, 2.0, 3.0, 4.0]),
    )

    assert selected.index == 3
    assert selected.passing_prefix == (3,)
    assert selected.stopped_index == 2


def test_ordered_point_selector_uses_width_then_grid_then_index_ties() -> None:
    selected = select_ordered_point_index(
        np.array([1.0, 2.0, 2.0, 4.0]),
        np.full((4, 2), 0.95),
        np.array([3.0, 1.0, 1.0, 4.0]),
    )

    assert selected.index == 1


def test_recovery_rejects_bitwise_family_mismatch() -> None:
    phase0, paper = _surfaces()
    phase0["standard_profile"][0] = np.nextafter(
        phase0["standard_profile"][0], np.float32(2.0)
    )

    with pytest.raises(ValueError, match="not bitwise identical"):
        recover_standard_decomposition(phase0, paper)


def test_recovery_rejects_nan_no_feasible_and_endpoint() -> None:
    phase0, paper = _surfaces()
    paper["cot_diagonal"][2, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        recover_standard_decomposition(phase0, paper)

    phase0, paper = _surfaces()
    paper["cot_diagonal"][:] = 0.89
    with pytest.raises(ValueError, match="no feasible"):
        recover_standard_decomposition(phase0, paper)

    phase0, paper = _surfaces()
    phase0["standard_greedy_selected_schedule"][0] = phase0[
        "standard_greedy_stage_grids"
    ][0, 0]
    with pytest.raises(ValueError, match="endpoint"):
        recover_standard_decomposition(phase0, paper)


def test_recovery_rejects_saved_e_schedule_inconsistent_with_lcb_rule() -> None:
    phase0, paper = _surfaces()
    paper["scpcp_selected_radii"] = paper["candidate_radii"][3].copy()

    with pytest.raises(ValueError, match="saved E schedule"):
        recover_standard_decomposition(phase0, paper)


def test_common_fresh_evaluations_become_canonical_records() -> None:
    phase0, paper = _surfaces()
    selection = recover_standard_decomposition(phase0, paper)
    evaluations = {
        layer: FreshEvaluation(
            coverage=np.array([0.90 + offset, 0.92 + offset]),
            normalized_width=np.array([1.0 + offset, 2.0 + offset]),
            micro_normalized_width=1.5 + offset,
            patient_normalized_width=1.5 + offset,
            n_rollouts=50_000,
        )
        for layer, offset in zip(("A", "C", "D", "E"), (0.0, 0.01, 0.02, 0.03), strict=True)
    }

    records = canonical_fresh_records(7, selection, evaluations)

    assert [record["layer"] for record in records] == ["A", "C", "D", "E"]
    assert all(record["seed"] == 7 for record in records)
    assert all(record["evaluation_scope"] == "common_crn_fresh_target_policy_rollouts" for record in records)
    assert records[0]["selection_indices"] == "[2,3]"
    assert records[2]["selection_indices"] == "[3]"
    assert records[0]["target_met"] is True
    assert records[3]["oracle_evaluation_trajectories"] == 50_000


def test_log_ratio_decomposition_closes_exactly_up_to_roundoff() -> None:
    decomposition = log_ratio_decomposition(
        a_width=1.8328,
        c_width=1.8979,
        d_width=1.9077,
        e_width=1.9562,
    )

    assert decomposition.a_to_c + decomposition.c_to_d + decomposition.d_to_e == pytest.approx(
        decomposition.a_to_e, abs=1e-15
    )
    assert decomposition.closure_error == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("bad_width", [0.0, -1.0, np.nan, np.inf])
def test_log_ratio_decomposition_rejects_invalid_widths(bad_width: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        log_ratio_decomposition(
            a_width=1.0,
            c_width=1.1,
            d_width=bad_width,
            e_width=1.3,
        )
