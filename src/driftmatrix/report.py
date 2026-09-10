"""Result tables and figures (spec 5.4).

Palette: validated categorical slots 1-3 from the dataviz reference palette.
Slot 3 (aqua) sits below 3:1 on the light surface, so every figure using it
carries direct labels and a companion table - the relief rule.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .stats.bootstrap import paired_bootstrap  # noqa: E402

# --- design tokens ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2de"
WHISKER = "#8a8984"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")   # blue, orange, aqua
MARKERS = ("o", "s", "^")                    # secondary encoding for CVD/print

_CELL = ["arch_level", "alignment", "prefix_band", "continuation_band"]
_ARCH_LABEL = {3: "L3 recurrent", 2: "L2 sliding", 1: "L1 dense"}


def _style(ax):
    """Recessive grid and axes; text wears ink tokens, never a series color."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    return ax


def _fig(nrows=1, ncols=1, figsize=(7.2, 4.2)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    return fig, axes


def _paired(gen: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, session, turn) with both arms side by side."""
    df = gen
    if "mode" in df.columns:
        df = df[df["mode"] == "free_running"]
    first = df[df["position"] == 0]
    idx = _CELL + ["session_id", "turn_idx", "replicate"]
    idx = [c for c in idx if c in first.columns]
    r = first[first["arm"] == "restore"].set_index(idx)
    c = first[first["arm"] == "recompute"].set_index(idx)
    joined = r.join(c, rsuffix="_rc", how="inner")
    joined["diverged"] = joined["chosen_token_id"] != joined["chosen_token_id_rc"]
    joined["tool_disagree"] = (
        (joined["tool_name"] != joined["tool_name_rc"])
        | (joined["tool_args_hash"] != joined["tool_args_hash_rc"]))
    return joined.reset_index()


def divergence_table(gen: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Divergence and tool-disagreement rates per cell, with bootstrap CIs."""
    p = _paired(gen)
    rows = []
    for key, grp in p.groupby(_CELL):
        vals = grp["diverged"].to_numpy(dtype=float)
        ci = paired_bootstrap(vals, n_resamples=2000, seed=seed)
        rows.append(dict(zip(_CELL, key)) | {
            "n_turns": int(len(vals)),
            "divergence_rate": float(vals.mean()),
            "ci_low": max(0.0, ci.low),
            "ci_high": min(1.0, ci.high),
            "tool_disagreement_rate": float(grp["tool_disagree"].mean()),
        })
    return pd.DataFrame(rows).sort_values(_CELL).reset_index(drop=True)


def outcome_census(gen: pd.DataFrame) -> pd.DataFrame:
    return (gen.groupby("outcome").size().reset_index(name="count")
               .sort_values("count", ascending=False).reset_index(drop=True))


def plot_headline(table: pd.DataFrame, path) -> None:
    """Divergence rate by boundary alignment across the architecture ladder.

    The prediction: a visibly non-zero L3 misaligned bar, near zero elsewhere.
    """
    fig, ax = _fig()
    _style(ax)
    levels = sorted(table["arch_level"].unique(), reverse=True)
    aligns = ["aligned", "misaligned"]
    width = 0.36
    gap = 0.02  # 2px-equivalent surface gap between adjacent bars
    for i, al in enumerate(aligns):
        sub = table[table["alignment"] == al].groupby("arch_level")
        vals = [sub["divergence_rate"].mean().get(lvl, 0.0) for lvl in levels]
        los = [sub["ci_low"].mean().get(lvl, 0.0) for lvl in levels]
        his = [sub["ci_high"].mean().get(lvl, 0.0) for lvl in levels]
        err = np.vstack([np.maximum(0, np.array(vals) - los),
                         np.maximum(0, np.array(his) - np.array(vals))])
        xs = [x + i * (width + gap) for x in range(len(levels))]
        ax.bar(xs, vals, width, label=al, color=SERIES[i], edgecolor=SURFACE,
               linewidth=1.5, zorder=2)
        # CI whiskers stay recessive: the bar carries the message, the whisker
        # qualifies it. Dark, thick whiskers read as the primary mark.
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=WHISKER,
                    elinewidth=1.0, capsize=3, zorder=3)
        # label above the whisker, not the bar, so the two never collide
        for x, v, hi in zip(xs, vals, his):
            ax.text(x, max(v, hi) + 0.015, f"{v:.1%}", ha="center", va="bottom",
                    fontsize=8, color=INK_SECONDARY, zorder=4)
    ax.set_xticks([x + (width + gap) / 2 for x in range(len(levels))])
    ax.set_xticklabels([_ARCH_LABEL.get(l, f"level {l}") for l in levels],
                       color=INK_SECONDARY)
    ax.set_ylim(0, max(0.08, float(table["ci_high"].max()) * 1.28))
    ax.set_ylabel("divergence rate", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Restore vs recompute: divergence by boundary alignment",
                 color=INK_PRIMARY, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_layer_profile(sd: pd.DataFrame, path) -> None:
    """Per-layer state difference, coloured and shaped by layer type."""
    fig, ax = _fig()
    _style(ax)
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    for i, (lt, grp) in enumerate(sorted(sd.groupby("layer_type"))):
        ax.scatter(grp["layer_idx"], grp["max_abs_diff"], label=lt,
                   color=SERIES[i % len(SERIES)], marker=MARKERS[i % len(MARKERS)],
                   s=44, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        worst = grp.loc[grp["max_abs_diff"].idxmax()] if len(grp) else None
        if worst is not None and worst["max_abs_diff"] > 0:
            # flip the label inward near the right edge so it cannot overflow
            span = float(sd["layer_idx"].max() - sd["layer_idx"].min()) or 1.0
            near_right = (worst["layer_idx"] - sd["layer_idx"].min()) / span > 0.65
            ax.annotate(f"{lt} max {worst['max_abs_diff']:.2e}",
                        (worst["layer_idx"], worst["max_abs_diff"]),
                        textcoords="offset points",
                        xytext=(-8, 8) if near_right else (8, 8),
                        ha="right" if near_right else "left",
                        fontsize=8, color=INK_SECONDARY)
    ax.set_xlabel("layer index", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("max abs state difference", color=INK_SECONDARY, fontsize=9)
    ax.set_title("State difference by layer type", color=INK_PRIMARY,
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def first_divergence_table(gen: pd.DataFrame) -> pd.DataFrame:
    """Per-turn first-divergence position, right-censored at the token cap.

    Reconstructs each arm's full token sequence, then compares them position by
    position - this is what a survival curve needs, and `_paired` deliberately
    only looks at position 0.
    """
    df = gen
    if "mode" in df.columns:
        df = df[df["mode"] == "free_running"]
    df = df[df["position"].notna()]
    if df.empty:
        return pd.DataFrame(columns=_CELL + ["first_divergence", "censored", "cap"])
    keys = _CELL + ["session_id", "turn_idx", "replicate"]
    keys = [k for k in keys if k in df.columns]
    seqs = (df.sort_values("position")
              .groupby(keys + ["arm"])["chosen_token_id"]
              .apply(list).unstack("arm"))
    rows = []
    for key, row in seqs.iterrows():
        a, b = row.get("restore"), row.get("recompute")
        if not isinstance(a, list) or not isinstance(b, list):
            continue
        cap = max(len(a), len(b))
        pos, censored = cap, True
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                pos, censored = i, False
                break
        else:
            if len(a) != len(b):
                pos, censored = min(len(a), len(b)), False
        rows.append(dict(zip(keys, key if isinstance(key, tuple) else (key,)))
                    | {"first_divergence": pos, "censored": censored, "cap": cap})
    return pd.DataFrame(rows)


def _survival_curve(fd: pd.DataFrame, cap: int):
    """Fraction of turns still token-identical at each position.

    Positions run 0..cap-1. A censored turn carries first_divergence == cap, so
    it must stay above the curve at every plotted k - extending the range to cap
    would drop every censored turn to zero at the final point.
    """
    ks = np.arange(0, max(1, cap))
    if fd.empty:
        return ks, np.ones_like(ks, dtype=float)
    d = fd["first_divergence"].to_numpy()
    return ks, np.array([float((d > k).mean()) for k in ks])


def plot_survival(gen: pd.DataFrame, path) -> None:
    """Small multiples: fraction of turns still token-identical, per level.

    Coincident series are the norm here (levels 1 and 2 should both sit flat at
    1.0), so linestyle carries identity alongside colour - otherwise the upper
    line hides the lower one entirely and the panel reads as a single series.
    """
    fd = first_divergence_table(gen)
    levels = sorted(gen["arch_level"].unique(), reverse=True)
    cap = int(fd["cap"].max()) if len(fd) else 1
    fig, axes = _fig(1, len(levels), figsize=(3.1 * len(levels), 3.4))
    axes = np.atleast_1d(axes)
    styles = ("-", "--")
    for ax, lvl in zip(axes, levels):
        _style(ax)
        sub = fd[fd["arch_level"] == lvl] if len(fd) else fd
        for i, al in enumerate(["aligned", "misaligned"]):
            grp = sub[sub["alignment"] == al] if len(sub) else sub
            ks, surv = _survival_curve(grp, cap)
            ax.plot(ks, surv, color=SERIES[i], linewidth=2, linestyle=styles[i],
                    marker=MARKERS[i], markersize=7, markevery=max(1, cap // 6),
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=al,
                    zorder=2 + i)
        ax.set_ylim(0, 1.06)
        ax.set_title(_ARCH_LABEL.get(lvl, f"level {lvl}"), color=INK_PRIMARY,
                     fontsize=10, loc="left")
        ax.set_xlabel("token position", color=INK_SECONDARY, fontsize=9)
    axes[0].set_ylabel("fraction still identical", color=INK_SECONDARY, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
                    loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def summary_metrics(gen: pd.DataFrame, noise_floor: float = 0.0) -> dict:
    """Flat metric dict for the CI regression gate."""
    p = _paired(gen)
    return {
        "divergence_rate": float(p["diverged"].mean()) if len(p) else 0.0,
        "tool_disagreement": float(p["tool_disagree"].mean()) if len(p) else 0.0,
        "noise_floor": float(noise_floor),
    }
