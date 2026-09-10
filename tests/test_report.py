import pandas as pd
from driftmatrix.report import (
    divergence_table, outcome_census, summary_metrics,
    plot_headline, plot_layer_profile, plot_survival,
)


def _gen_df():
    rows = []
    for arch in (3, 2, 1):
        for align in ("aligned", "misaligned"):
            for turn in range(10):
                diverged = (arch == 3 and align == "misaligned" and turn < 3)
                for arm in ("restore", "recompute"):
                    dirty = diverged and arm == "restore"
                    rows.append({
                        "arch_level": arch, "alignment": align,
                        "prefix_band": "short", "continuation_band": "short",
                        "session_id": "s1", "turn_idx": turn, "replicate": 1,
                        "arm": arm, "mode": "free_running", "position": 0,
                        "outcome": "ok",
                        "chosen_token_id": 99 if dirty else 5,
                        "tool_name": "b" if dirty else "a",
                        "tool_args_hash": "h",
                    })
    return pd.DataFrame(rows)


def test_divergence_table_has_one_row_per_cell():
    t = divergence_table(_gen_df())
    assert set(t.columns) >= {"arch_level", "alignment", "divergence_rate",
                              "tool_disagreement_rate", "n_turns",
                              "ci_low", "ci_high"}
    assert len(t) == 6


def test_divergence_appears_only_in_the_predicted_cell():
    t = divergence_table(_gen_df()).set_index(["arch_level", "alignment"])
    assert t.loc[(3, "misaligned"), "divergence_rate"] > 0
    assert t.loc[(3, "aligned"), "divergence_rate"] == 0
    assert t.loc[(2, "misaligned"), "divergence_rate"] == 0
    assert t.loc[(1, "misaligned"), "divergence_rate"] == 0


def test_confidence_interval_brackets_the_rate():
    t = divergence_table(_gen_df()).set_index(["arch_level", "alignment"])
    row = t.loc[(3, "misaligned")]
    assert row["ci_low"] <= row["divergence_rate"] <= row["ci_high"]
    assert 0.0 <= row["ci_low"] and row["ci_high"] <= 1.0


def test_tool_disagreement_is_tracked_separately():
    t = divergence_table(_gen_df()).set_index(["arch_level", "alignment"])
    assert t.loc[(3, "misaligned"), "tool_disagreement_rate"] > 0
    assert t.loc[(1, "aligned"), "tool_disagreement_rate"] == 0


def test_outcome_census_counts_every_row():
    df = _gen_df()
    df.loc[df.index[:4], "outcome"] = "crashed"
    c = outcome_census(df)
    assert set(c["outcome"]) >= {"ok", "crashed"}
    assert c["count"].sum() == len(df)


def test_summary_metrics_shape_matches_gate_baseline():
    m = summary_metrics(_gen_df(), noise_floor=0.001)
    assert set(m) == {"divergence_rate", "tool_disagreement", "noise_floor"}
    assert 0.0 < m["divergence_rate"] < 1.0


def test_figures_are_written(tmp_path):
    gen = _gen_df()
    t = divergence_table(gen)
    p1 = tmp_path / "headline.png"
    plot_headline(t, p1)
    assert p1.exists() and p1.stat().st_size > 5000
    sd = pd.DataFrame({"layer_idx": [0, 1, 2, 3],
                       "layer_type": ["linear_attention"] * 3 + ["full_attention"],
                       "max_abs_diff": [1e-4, 2e-4, 5e-5, 0.0],
                       "arch_level": [3] * 4})
    p2 = tmp_path / "layers.png"
    plot_layer_profile(sd, p2)
    assert p2.exists() and p2.stat().st_size > 5000
    p3 = tmp_path / "surv.png"
    plot_survival(gen, p3)
    assert p3.exists() and p3.stat().st_size > 5000


def _gen_df_multiposition():
    """Turns with real token sequences so survival has something to measure."""
    rows = []
    for arch in (3, 2, 1):
        for align in ("aligned", "misaligned"):
            for turn in range(10):
                diverges = (arch == 3 and align == "misaligned" and turn < 4)
                div_at = 3 if diverges else None
                for arm in ("restore", "recompute"):
                    for pos in range(8):
                        tok = 100 + pos
                        if div_at is not None and arm == "restore" and pos >= div_at:
                            tok += 500
                        rows.append({
                            "arch_level": arch, "alignment": align,
                            "prefix_band": "short", "continuation_band": "short",
                            "session_id": "s1", "turn_idx": turn, "replicate": 1,
                            "arm": arm, "mode": "free_running", "position": pos,
                            "outcome": "ok", "chosen_token_id": tok,
                            "tool_name": "a", "tool_args_hash": "h",
                        })
    return pd.DataFrame(rows)


def test_first_divergence_table_finds_the_injected_position():
    from driftmatrix.report import first_divergence_table
    fd = first_divergence_table(_gen_df_multiposition())
    hit = fd[(fd["arch_level"] == 3) & (fd["alignment"] == "misaligned")]
    assert set(hit[~hit["censored"]]["first_divergence"]) == {3}
    clean = fd[fd["arch_level"] == 1]
    assert clean["censored"].all()


def test_survival_curve_starts_at_one_and_drops_only_where_predicted():
    from driftmatrix.report import first_divergence_table, _survival_curve
    fd = first_divergence_table(_gen_df_multiposition())
    cap = int(fd["cap"].max())
    l3 = fd[(fd["arch_level"] == 3) & (fd["alignment"] == "misaligned")]
    ks, surv = _survival_curve(l3, cap)
    assert surv[0] == 1.0
    assert surv[-1] < 1.0
    l1 = fd[fd["arch_level"] == 1]
    _, flat = _survival_curve(l1, cap)
    assert (flat == 1.0).all()


def test_survival_figure_renders_with_multiposition_data(tmp_path):
    p = tmp_path / "surv2.png"
    plot_survival(_gen_df_multiposition(), p)
    assert p.exists() and p.stat().st_size > 5000
