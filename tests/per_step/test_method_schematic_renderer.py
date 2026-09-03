from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]


def _load_renderer():
    path = ROOT / "tools" / "render_method_schematic.py"
    spec = importlib.util.spec_from_file_location("render_method_schematic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_method_schematic_semantics_match_the_canonical_claim_boundary() -> None:
    renderer = _load_renderer()
    boxes, arrows = renderer.schematic_elements()
    by_id = {element.element_id: element for element in (*boxes, *arrows)}

    assert renderer.FIGURE_CONTRACT["claim_boundary"] == (
        "plug-in asymptotic per-step marginal coverage"
    )
    assert "entire committed policy prefix" in renderer.FIGURE_CONTRACT[
        "core_conclusion"
    ]
    assert "\\prod_{j=0}^{t}" in by_id["full_prefix_weight"].body
    assert "\\pi_j^{q_j}" in by_id["full_prefix_weight"].body
    assert "\\mu_j" in by_id["full_prefix_weight"].body
    assert "current-action ratio is required" in by_id["post_action_score"].semantic_role
    assert by_id["action_to_score"].source_id == "current_action"
    assert by_id["action_to_score"].target_id == "post_action_score"
    assert "not a global schedule optimizer" in by_id[
        "stagewise_selection"
    ].semantic_role
    assert "not fixed-point iteration" in by_id["commit_and_advance"].note


def test_method_schematic_renderer_writes_exact_editable_bundle(tmp_path: Path) -> None:
    renderer = _load_renderer()
    work = tmp_path / "work"
    paper = tmp_path / "paper"

    renderer.render_bundle(work, paper)

    assert {path.name for path in work.iterdir()} == renderer.WORK_FILES
    assert {path.name for path in paper.iterdir()} == renderer.PAPER_FILES
    assert all(path.suffix == ".pdf" for path in paper.iterdir())

    svg = (work / f"{renderer.FIGURE_STEM}.svg").read_text(encoding="utf-8")
    assert "<text" in svg
    assert "font-family" in svg
    assert "<image" not in svg
    assert re.search(r"Times New Roman|Times|DejaVu Serif", svg)

    paper_pdf = (paper / f"{renderer.FIGURE_STEM}.pdf").read_bytes()
    work_pdf = (work / f"{renderer.FIGURE_STEM}.pdf").read_bytes()
    assert paper_pdf == work_pdf
    assert paper_pdf.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page\b", paper_pdf)) == 1
    assert (work / f"{renderer.FIGURE_STEM}.png").read_bytes().startswith(
        b"\x89PNG"
    )
    assert (work / f"{renderer.FIGURE_STEM}.tiff").read_bytes()[:4] in {
        b"II*\x00",
        b"MM\x00*",
    }

    rows = list(
        csv.DictReader(
            (work / f"{renderer.FIGURE_STEM}_source.csv").open(encoding="utf-8")
        )
    )
    assert {row["panel"] for row in rows} == {"a", "b"}
    assert {row["kind"] for row in rows} == {"box", "arrow"}
    assert len(rows) == len(renderer.schematic_elements()[0]) + len(
        renderer.schematic_elements()[1]
    )

    manifest = json.loads((work / "render_manifest.json").read_text())
    assert manifest["protocol"] == renderer.RENDER_PROTOCOL
    assert manifest["status"] == "complete"
    assert manifest["scientific_seed_reads"] == 0
    assert manifest["canonical_source_modified"] is False
    renderer.validate_bundle(work, paper)


def test_method_schematic_requires_fresh_distinct_outputs(tmp_path: Path) -> None:
    renderer = _load_renderer()
    occupied = tmp_path / "occupied"
    occupied.mkdir()

    try:
        renderer.render_bundle(occupied, tmp_path / "paper")
    except FileExistsError:
        pass
    else:
        raise AssertionError("renderer accepted an occupied output directory")

    same = tmp_path / "same"
    try:
        renderer.render_bundle(same, same)
    except ValueError:
        pass
    else:
        raise AssertionError("renderer accepted one directory for both output roles")
