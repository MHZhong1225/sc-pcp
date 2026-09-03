from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import shutil
import sys

from matplotlib.container import ErrorbarContainer
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.reporting import render_complete_diagnostics_minimal as renderer


@pytest.fixture(scope="module")
def artifacts() -> renderer.FrozenArtifacts:
    return renderer.load_frozen_artifacts(renderer.RenderConfig())


@pytest.fixture(scope="module")
def source_rows(
    artifacts: renderer.FrozenArtifacts,
) -> dict[str, list[dict[str, object]]]:
    return {
        "exact": renderer.formal.build_exact_source_rows(artifacts.exact_summary),
        "theory": renderer._retarget_rows(
            renderer.theorem.build_theory_source_rows(
                artifacts.horizon_summary,
                artifacts.rq6_summary,
            ),
            renderer.THEORY_STEM,
        ),
        "robustness": renderer._retarget_rows(
            renderer.theorem.build_robustness_source_rows(
                artifacts.propensity_summary,
                artifacts.strict_summary,
            ),
            renderer.ROBUSTNESS_STEM,
        ),
        "prefix": renderer.build_prefix_source_rows(artifacts.prefix_summary),
    }


def test_frozen_loaders_and_source_grids_are_complete(
    artifacts: renderer.FrozenArtifacts,
    source_rows: dict[str, list[dict[str, object]]],
) -> None:
    assert set(artifacts.input_contracts) == {
        "exact_identification",
        "horizon_overlap",
        "ncal_convergence",
        "propensity_robustness",
        "strict_split",
        "prefix_ablation",
    }
    assert {name: len(rows) for name, rows in source_rows.items()} == {
        "exact": 16,
        "theory": 93,
        "robustness": 18,
        "prefix": 100,
    }

    prefix = source_rows["prefix"]
    assert {float(row["gamma"]) for row in prefix} == set(renderer.PREFIX_GAMMAS)
    assert {row["variant"] for row in prefix} == set(renderer.PREFIX_METHODS)
    assert {row["metric"] for row in prefix} == {
        "WSC",
        "Mean normalized width",
        "Late target Q90/radius ratio",
        "Minimum selection ESS/n",
    }
    without_intervals = {
        row["metric"] for row in prefix if row["ci95_lower"] == ""
    }
    assert without_intervals == {
        "Mean normalized width",
        "Minimum selection ESS/n",
    }
    assert all(float(row["selection_rate"]) == 1.0 for row in prefix)


def test_prefix_bundle_is_tree_pinned_and_fails_closed(tmp_path: Path) -> None:
    summary, contract = renderer.validate_prefix_bundle(renderer.DEFAULT_PREFIX_ROOT)
    assert summary["protocol"] == renderer.PREFIX_PROTOCOL
    assert contract["tree_sha256"] == renderer.PREFIX_TREE_SHA256

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    shutil.copyfile(renderer.DEFAULT_PREFIX_ROOT / "COMPLETE", incomplete / "COMPLETE")
    with pytest.raises(RuntimeError, match="root entries differ"):
        renderer.validate_prefix_bundle(incomplete)


@pytest.mark.parametrize(
    ("name", "render", "expected_axes", "panel_letters"),
    [
        ("exact", renderer.render_exact_figure, 2, "a"),
        ("theory", renderer.render_theory_figure, 9, "abcdef"),
        ("robustness", renderer.render_robustness_figure, 6, "abcdef"),
        ("prefix", renderer.render_prefix_figure, 4, "abcd"),
    ],
)
def test_figures_obey_the_minimal_visible_text_contract(
    source_rows: dict[str, list[dict[str, object]]],
    name: str,
    render: object,
    expected_axes: int,
    panel_letters: str,
) -> None:
    renderer.apply_publication_style()
    figure = render(source_rows[name])
    assert figure._suptitle is None
    assert figure.texts == []
    assert len(figure.axes) == expected_axes
    assert all(axis.get_title() == "" for axis in figure.axes)
    visible_annotations = {
        text.get_text() for axis in figure.axes for text in axis.texts
    }
    assert visible_annotations == set(panel_letters)
    all_text = " ".join(text.get_text() for text in figure.findobj(Text))
    for forbidden in (
        "Exact population identification across",
        "Coverage remains",
        "Canonical SC-PCP",
        "diagnostics only",
        "finite-sample",
    ):
        assert forbidden not in all_text
    plt.close(figure)


def test_rendered_panels_preserve_the_frozen_plot_math(
    source_rows: dict[str, list[dict[str, object]]],
) -> None:
    renderer.apply_publication_style()

    exact = renderer.render_exact_figure(source_rows["exact"])
    assert exact.axes[0].images[0].get_array().shape == (4, 4)
    assert len(exact.axes[0].patches) == len(renderer.formal.IDENTIFICATION_CORRECT)
    assert [text.get_text() for text in exact.axes[0].get_legend().texts] == ["Exact"]
    plt.close(exact)

    theory = renderer.render_theory_figure(source_rows["theory"])
    assert all(theory.axes[index].images[0].get_array().shape == (5, 5) for index in range(3))
    assert all(
        sum(isinstance(item, ErrorbarContainer) for item in theory.axes[index].containers)
        == 1
        for index in range(3, 6)
    )
    plt.close(theory)

    robustness = renderer.render_robustness_figure(source_rows["robustness"])
    assert all(
        sum(
            isinstance(item, ErrorbarContainer)
            for item in robustness.axes[index].containers
        )
        == 3
        for index in range(4)
    )
    plt.close(robustness)

    prefix = renderer.render_prefix_figure(source_rows["prefix"])
    assert [text.get_text() for text in prefix.legends[0].texts] == [
        renderer.PREFIX_LABELS[method] for method in renderer.PREFIX_METHODS
    ]
    assert all(len(axis.lines) >= len(renderer.PREFIX_METHODS) for axis in prefix.axes)
    plt.close(prefix)


def test_atomic_bundle_is_deterministic_pdf_only_and_600dpi(tmp_path: Path) -> None:
    outputs = []
    for suffix in ("a", "b"):
        work = tmp_path / f"work_{suffix}"
        paper = tmp_path / f"paper_{suffix}"
        renderer.render_report(
            renderer.RenderConfig(work_output=work, paper_output=paper)
        )
        renderer.validate_rendered_outputs(work, paper)
        outputs.append((work, paper))

    first_work, first_paper = outputs[0]
    second_work, second_paper = outputs[1]
    assert {path.name for path in first_work.iterdir()} == renderer.WORK_FILES
    assert {path.name for path in first_paper.iterdir()} == renderer.PAPER_FILES
    assert all(path.suffix == ".pdf" for path in first_paper.iterdir())

    for name in sorted(renderer.WORK_FILES):
        assert _sha256(first_work / name) == _sha256(second_work / name), name
    for name in sorted(renderer.PAPER_FILES):
        assert _sha256(first_paper / name) == _sha256(second_paper / name), name

    expected_rows = {
        renderer.EXACT_SOURCE: 16,
        renderer.THEORY_SOURCE: 93,
        renderer.ROBUSTNESS_SOURCE: 18,
        renderer.PREFIX_SOURCE: 100,
    }
    for name, expected in expected_rows.items():
        with (first_work / name).open(encoding="utf-8", newline="") as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == expected

    for stem in renderer.FIGURE_STEMS:
        assert _sha256(first_work / f"{stem}.pdf") == _sha256(
            first_paper / f"{stem}.pdf"
        )
        svg = (first_work / f"{stem}.svg").read_text(encoding="utf-8")
        assert "<text" in svg
        assert "Times New Roman" in svg
        with Image.open(first_work / f"{stem}.tiff") as image:
            assert image.info["dpi"][0] == pytest.approx(600, rel=1e-3)
            assert image.info["dpi"][1] == pytest.approx(600, rel=1e-3)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
