"""Figure data extraction, and a smoke test that every figure renders."""
import json

from driftmatrix.figures import (
    action_change_points, drift_values, explicit_totals, load_g1, margin_points,
    plot_action_changes, plot_drift_per_turn, plot_exactness_grid, plot_margin_bins,
)


def member(events, n=90, claimed=False):
    return {"events": events, "n": n, "estimate": events / n, "claimed": claimed,
            "interval": {"method": "clopper_pearson", "low": 0.0 if not events else 0.01, "high": 0.04 + events / n}}


def analysis():
    fam = {}
    for m in ("ornith", "spark", "qwen3"):
        for b in ("natural", "snapped"):
            for p in ("short", "long"):
                for s in ("explicit", "ambiguous"):
                    fam[f"A:{m}:{b}:{p}:{s}"] = member(3 if (b, s) == ("natural", "ambiguous") else 0)
    bins = {"near_tie": {"n": 30, "events": 9, "rate": 0.3, "interval": {"low": 0.15, "high": 0.49}},
            "contested": {"n": 50, "events": 2, "rate": 0.04, "interval": {"low": 0.005, "high": 0.14}},
            "confident": {"n": 280, "events": 0, "rate": 0.0, "interval": {"low": 0.0, "high": 0.013}}}
    rows = [{"model": m, "boundary": b, "stratum": s, "excluded": None, "mean_jsd": v}
            for m in ("ornith", "spark", "qwen3") for b in ("natural", "snapped")
            for s, v in (("explicit", 1e-6), ("ambiguous", 1e-4), ("ambiguous", 0.0))]
    return {"families": {"A": fam},
            "secondary": {m: {"margin_bins": {"natural": json.loads(json.dumps(bins))}}   # a copy per model
                          for m in ("ornith", "spark", "qwen3")},
            "rows": rows}


def test_load_g1_reads_model_ubatch_and_cells(tmp_path):
    for ub in (256, 512):
        (tmp_path / f"g1_ornith_q4km_ub{ub}.json").write_text(json.dumps(
            {"ubatch": ub, "cases": [{"k": 512, "exact": True, "max_tv": 0.0},
                                     {"k": 384, "exact": False, "max_tv": 0.25}, {"k": 640, "error": "x"}]}))
    grid = load_g1(sorted(tmp_path.glob("*.json")))
    assert grid == {"ornith": {256: {512: (True, 0.0), 384: (False, 0.25)},
                               512: {512: (True, 0.0), 384: (False, 0.25)}}}


def test_action_change_points_take_the_ambiguous_members_in_order():
    pts = action_change_points(analysis())
    assert len(pts) == 12 and all(p["n"] == 90 for p in pts)
    assert [(p["model"], p["band"], p["boundary"]) for p in pts[:2]] == [("ornith", "short", "natural"),
                                                                        ("ornith", "short", "snapped")]
    assert {p["events"] for p in pts if p["boundary"] == "natural"} == {3}


def test_explicit_totals_sum_every_explicit_member():
    assert explicit_totals(analysis()) == (0, 24 * 90 // 2)


def test_margin_points_skip_empty_bins():
    a = analysis()
    a["secondary"]["qwen3"]["margin_bins"]["natural"]["near_tie"] = {"n": 0, "events": 0, "rate": None, "interval": None}
    pts = margin_points(a)
    assert len(pts) == 8 and not any(p["model"] == "qwen3" and p["bin"] == "near_tie" for p in pts)


def test_drift_values_group_natural_turns_by_model_and_stratum():
    v = drift_values(analysis())
    assert v[("spark", "ambiguous")] == [1e-4, 0.0] and v[("qwen3", "explicit")] == [1e-6]


def test_every_figure_renders(tmp_path):
    g1 = tmp_path / "g1_qwen3_dense_ub512.json"
    g1.write_text(json.dumps({"ubatch": 512, "cases": [{"k": 512, "exact": True, "max_tv": 0.0},
                                                        {"k": 768, "exact": False, "max_tv": 0.01}]}))
    a = analysis()
    plot_exactness_grid(load_g1([g1]), tmp_path / "grid.png")
    plot_action_changes(a, tmp_path / "actions.png")
    plot_margin_bins(a, tmp_path / "margins.png")
    plot_drift_per_turn(a, tmp_path / "drift.png")
    for name in ("grid", "actions", "margins", "drift"):
        assert (tmp_path / f"{name}.png").stat().st_size > 10_000


import pytest

import driftmatrix.figures as figures_module


def _plotters():
    grid = {"ornith": {256: {256: (True, 0.0), 384: (False, 0.2)}},
            "qwen3": {256: {256: (True, 0.0), 384: (False, 0.01)}}}
    a = analysis()
    return {"g1_exactness_grid": lambda path, **kw: plot_exactness_grid(grid, path, **kw),
            "action_changes": lambda path, **kw: plot_action_changes(a, path, **kw),
            "margin_bins": lambda path, **kw: plot_margin_bins(a, path, **kw),
            "drift_per_turn": lambda path, **kw: plot_drift_per_turn(a, path, **kw)}


@pytest.mark.parametrize("name", ["g1_exactness_grid", "action_changes", "margin_bins", "drift_per_turn"])
def test_paper_mode_writes_a_pdf_with_no_title(tmp_path, monkeypatch, name):
    titled = []
    monkeypatch.setattr(figures_module, "_titles", lambda *args, **kwargs: titled.append(args))
    path = tmp_path / f"{name}.pdf"
    _plotters()[name](path, paper=True)
    assert path.read_bytes().startswith(b"%PDF")
    assert titled == []


@pytest.mark.parametrize("name", ["g1_exactness_grid", "action_changes", "margin_bins", "drift_per_turn"])
def test_default_mode_still_draws_the_title_and_writes_a_png(tmp_path, monkeypatch, name):
    titled = []
    monkeypatch.setattr(figures_module, "_titles", lambda *args, **kwargs: titled.append(args))
    path = tmp_path / f"{name}.png"
    _plotters()[name](path)
    assert path.read_bytes().startswith(b"\x89PNG")
    assert len(titled) == 1


def test_every_figure_has_a_paper_size():
    assert set(figures_module.PAPER_SIZES) == {"g1_exactness_grid", "action_changes", "margin_bins", "drift_per_turn"}
