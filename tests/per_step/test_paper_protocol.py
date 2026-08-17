from __future__ import annotations

import math
from pathlib import Path
import sys

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
import torch

import scpcp.experiment as experiment
from scpcp.config import ExperimentConfig, PaperConfig
from scpcp.data import TrajectoryBatch


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import render_paper_results as renderer


def _batch() -> TrajectoryBatch:
    return TrajectoryBatch(
        states=torch.zeros(3, 3, 1),
        actions=torch.zeros(3, 2, dtype=torch.long),
        outcomes=torch.tensor(
            [
                [[1.0, 2.0], [2.0, 4.0]],
                [[3.0, 6.0], [4.0, 8.0]],
                [[5.0, 10.0], [6.0, 12.0]],
            ]
        ),
        patient_ids=torch.arange(3),
    )


def test_training_outcome_sd_uses_dpred_and_sample_correction() -> None:
    batch = _batch()

    observed = experiment._training_outcome_sd(batch)
    expected = batch.outcomes.reshape(-1, 2).std(dim=0, unbiased=True)

    assert torch.allclose(observed, expected)


def test_paper_rng_streams_are_stable_and_separated() -> None:
    assert experiment._paper_seed(7, 101) == experiment._paper_seed(7, 101)
    assert experiment._paper_seed(7, 101) != experiment._paper_seed(7, 211)
    assert experiment._paper_seed(7, 101) != experiment._paper_seed(8, 101)


def test_paper_config_validates_protocol_controls() -> None:
    config = ExperimentConfig(paper=PaperConfig(mechanism_seed=0))
    config.validate()

    invalid = ExperimentConfig(paper=PaperConfig(mechanism_seed=-1))
    with pytest.raises(ValueError, match="paper.mechanism_seed"):
        invalid.validate()


def test_metric_placeholders_expose_explicit_selection_and_width() -> None:
    placeholders = experiment._metric_placeholders()

    assert placeholders["selection_available"] is False
    assert math.isnan(placeholders["average_normalized_width"])


def test_paper_curve_ci_uses_metric_appropriate_bounds() -> None:
    matrix = np.array([[0.80, 0.90], [1.00, 1.00]])
    coverage_low, coverage_high = renderer.curve_t_ci(matrix, probability=True)

    widths = 4.0 * matrix
    width_low, width_high = renderer.curve_t_ci(widths, probability=False)

    assert np.all(coverage_low >= 0.0)
    assert np.all(coverage_high <= 1.0)
    assert np.all(width_low >= 0.0)
    assert width_high.max() > 1.0


def test_coverage_profile_states_that_curves_are_conditional_on_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = []
    for dataset, feedback_strength in (("synthetic", 2.0), ("mimic_iv", 1.0)):
        rows.append(
            {
                "dataset": dataset,
                "feedback_strength": feedback_strength,
                "method_family": "Standard CP",
                "per_time_coverage": "[0.90, 0.91]",
                "per_time_normalized_width": "[1.20, 1.10]",
            }
        )
    records = pd.DataFrame(rows)
    figure_text: list[str] = []

    def capture_figure(figure: matplotlib.figure.Figure, *_args, **_kwargs) -> None:
        figure_text.extend(text.get_text() for text in figure.texts)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_figure)
    renderer.render_coverage_profiles(records, tmp_path / "coverage.pdf")

    assert renderer.CONDITIONAL_SELECTION_NOTE in figure_text
