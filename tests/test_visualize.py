"""Tests for the results-visualization module (markdown table + SVG).

Dependency-free: imports only ``weighted_safety.visualize`` and stdlib, so these
run even without inspect-ai installed.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from weighted_safety.visualize import (
    harmonic_mean,
    rank_models,
    render_svg,
    render_table,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_ROOT / "results" / "multi_model.json"


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS_FILE.read_text())


# --- CSS / harmonic mean -------------------------------------------------


def test_harmonic_mean_basic():
    assert harmonic_mean(1.0, 1.0) == pytest.approx(1.0)
    assert harmonic_mean(0.5, 0.5) == pytest.approx(0.5)


def test_harmonic_mean_zero_axis_is_zero():
    # The gaming-resistance property: refuse-all (A=0) or comply-all (R=0) => 0.
    assert harmonic_mean(1.0, 0.0) == 0.0
    assert harmonic_mean(0.0, 1.0) == 0.0


def test_published_css_matches_harmonic_mean(results):
    # Cross-check that the stored CSS values are the harmonic mean of R and A.
    for m in results["models"]:
        if m.get("CSS") is not None and m.get("A") is not None:
            assert harmonic_mean(m["R"], m["A"]) == pytest.approx(m["CSS"], abs=1e-3)


def test_llama_per_category_aggregates_to_R(results):
    # The per-category r_c, weighted by metric_weights, should reproduce R.
    weights = results["metric_weights"]
    llama = next(m for m in results["models"] if m["name"].startswith("Llama"))
    r = sum(weights[c] * llama["per_category"][c]["r"] for c in weights)
    assert r == pytest.approx(llama["R"], abs=1e-3)


# --- ranking -------------------------------------------------------------


def test_rank_orders_by_css_desc(results):
    ranked = rank_models(results["models"])
    # Qwen (CSS 0.901) ranks above Llama (CSS 0.844); Gemini (no CSS) sinks last.
    names = [m["name"] for m in ranked]
    assert names[0].startswith("Qwen")
    assert names.index("Llama 3.3 70B") < names.index("Gemini 2.5 Flash")


# --- markdown table ------------------------------------------------------


def test_table_lists_all_models_and_dashes_missing(results):
    md = render_table(results)
    for m in results["models"]:
        assert m["name"] in md
    # Gemini's missing A/CSS render as an em dash, not "None".
    assert "None" not in md
    assert "—" in md


def test_table_is_markdown_table(results):
    md = render_table(results)
    assert "| Rank | Model |" in md
    # one data row per model
    assert md.count("\n|") >= len(results["models"]) + 1


# --- SVG -----------------------------------------------------------------


def test_svg_is_well_formed_xml(results):
    svg = render_svg(results)
    root = ET.fromstring(svg)  # raises on malformed XML
    assert root.tag.endswith("svg")


def test_svg_has_a_bar_per_present_series(results):
    svg = render_svg(results)
    # Count <rect> bars carrying a <title> (the data bars, not background/legend).
    bar_count = svg.count("<title>")
    # Qwen: R,CSS,flat = 3; Llama: 3; Gemini: R,flat = 2 (CSS null) => 8
    expected = sum(
        sum(1 for k in ("R", "CSS", "flat_avg") if m.get(k) is not None)
        for m in results["models"]
    )
    assert bar_count == expected


def test_svg_marks_missing_series(results):
    svg = render_svg(results)
    # Gemini has null CSS -> an "n/a" marker should appear.
    assert "n/a" in svg


def test_svg_dimensions_scale_with_width(results):
    narrow = render_svg(results, width=400)
    wide = render_svg(results, width=900)
    assert 'width="400"' in narrow
    assert 'width="900"' in wide
