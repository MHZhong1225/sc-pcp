from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import render_main_suite_figures as renderer


def _load() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    main, stage, metadata = renderer.load_frozen_export(renderer.DEFAULT_INPUT)
    return main, stage, dict(metadata)


def test_pareto_source_retains_all_methods_and_audits_point_frontier() -> None:
    main, _, _ = _load()

    source = renderer.build_pareto_source(main)

    assert len(source) == 30
    assert set(source["method"]) == set(renderer.METHODS)
    scpcp = source[source["method"].eq("SC-PCP")]
    assert scpcp["point_pareto_frontier"].all()
    narrowest = source[source["narrowest_point_eligible"]].set_index("dataset")
    assert narrowest.loc["synthetic", "method"] == "SC-PCP"
    assert narrowest.loc["mimic_iv", "method"] == "SC-PCP"
    assert narrowest.loc["inspire", "method"] == "SC-PCP"
    assert narrowest.loc["mimic_cxr", "method"] == "ACI"
    assert narrowest.loc["eicu", "method"] == "ACI"
    assert (source["point_coverage_eligible"] == source["wsc"].ge(0.90)).all()


def test_stage_profile_source_keeps_six_methods_and_uses_percentage_points() -> None:
    _, stage, _ = _load()

    source = renderer.build_profile_source(stage)

    assert len(source) == 216
    assert set(source["dataset"]) == set(renderer.PROFILE_DATASETS)
    assert set(source["method"]) == set(renderer.METHODS)
    assert source.groupby(["dataset", "method"]).size().eq(12).all()
    assert source.loc[source["method"].eq("SPCI"), "display_role"].eq(
        "muted_comparator"
    ).all()
    assert source.loc[source["method"].eq("PRC"), "display_role"].eq(
        "muted_comparator"
    ).all()
    first = source.iloc[0]
    assert first["coverage_deviation_pp"] == pytest.approx(
        100.0 * (first["coverage_mean"] - 0.90)
    )


def test_frozen_export_validation_fails_after_payload_tampering(tmp_path: Path) -> None:
    copied = tmp_path / "export"
    shutil.copytree(renderer.DEFAULT_INPUT, copied)
    target = copied / "rq1_all_baselines.csv"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash differs"):
        renderer.load_frozen_export(copied)


def test_source_validation_rejects_missing_canonical_method() -> None:
    main, stage, metadata = _load()
    broken = main[~(
        main["dataset"].eq("synthetic") & main["method"].eq("SC-PCP")
    )].copy()

    with pytest.raises(RuntimeError, match="row counts differ"):
        renderer.validate_source_frames(broken, stage, metadata)


def test_python_exports_keep_editable_times_text_and_identical_pdfs(
    tmp_path: Path,
) -> None:
    main, stage, _ = _load()
    pareto = renderer.build_pareto_source(main)
    profiles = renderer.build_profile_source(stage)
    work = tmp_path / "work"
    paper = tmp_path / "paper"
    work.mkdir()
    paper.mkdir()
    renderer.apply_publication_style()

    for stem, figure in (
        (renderer.PARETO_STEM, renderer.render_pareto_figure(pareto)),
        (renderer.PROFILE_STEM, renderer.render_profile_figure(profiles)),
    ):
        renderer.export_figure(
            figure,
            title=f"Test {stem}",
            work_stem=work / stem,
            paper_path=paper / f"{stem}.pdf",
            tiff_dpi=72,
            png_dpi=72,
        )
        svg = (work / f"{stem}.svg").read_text(encoding="utf-8")
        assert "<text" in svg
        assert "Times New Roman" in svg
        assert renderer._file_sha256(work / f"{stem}.pdf") == renderer._file_sha256(
            paper / f"{stem}.pdf"
        )
    assert {path.suffix for path in paper.iterdir()} == {".pdf"}
