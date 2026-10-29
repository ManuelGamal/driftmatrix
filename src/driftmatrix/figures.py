"""README, findings and paper figures, drawn only from committed results under docs/findings.

  exactness grid   G1: restore exact or not, by reuse boundary and n_ubatch
  action changes   main study: action-change rate on ambiguous requests, natural vs snapped
  margin bins      main study: action changes by the recompute arm's action margin
  drift per turn   main study: per-turn mean JSD at natural boundaries, by model and stratum

Each figure has a pure extraction function (tested) and a plotting function.
Design tokens and axis styling come from report.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .metrics.margin import BIN_LABELS
from .report import GRID, INK_PRIMARY, INK_SECONDARY, MARKERS, SERIES, SURFACE, WHISKER, _style

ORDER = ("ornith", "spark", "qwen3")
MODEL_LABELS = {"ornith": "Ornith-1.5-9B · hybrid recurrent",
                "spark": "Spark-X2.5-4B · sliding window",
                "qwen3": "Qwen3-4B · dense"}
SHORT_LABELS = {"ornith": "Ornith (hybrid)", "spark": "Spark (sliding window)", "qwen3": "Qwen3 (dense)"}
BIN_NAMES = {"near_tie": "near-tie\n< 0.5 nats", "contested": "contested\n0.5–2 nats", "confident": "confident\n≥ 2 nats"}
JSD_FLOOR = 1e-10        # per-turn drift below this is drawn on the floor line
# Paper figures: no title (the caption carries it), sized in inches at the width they
# print at: one 3.33 in column, or 0.82 of the 7 in text width for g1_exactness_grid.
PAPER_SIZES = {"g1_exactness_grid": (5.74, 3.3), "action_changes": (3.33, 2.75),
               "margin_bins": (3.33, 2.25), "drift_per_turn": (3.33, 2.35)}
# Paper figures print at full size, so README sizes shrink: a 9 pt label prints at 8.1 pt
# and an 8.5 pt annotation at 7.65 pt, close to the 8 pt caption text.
PAPER_FONT_SCALE = 0.9
# The paper's figures follow the closest prior work (Patodiya, arXiv 2609.04748):
# matplotlib's DejaVu Sans, a full black frame with outward ticks, capped error bars,
# a framed legend, no grid, and its palette.
PAPER_ERR = {"elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0, "markersize": 4.5, "markeredgewidth": 0}
# Shorter labels for the paper's narrower figures; the caption names the architectures in full.
PAPER_LABELS = {"ornith": "Ornith (hybrid)", "spark": "Spark (window)", "qwen3": "Qwen3 (dense)"}
# One meaning per color across the figures: orange (red in the paper) is the natural
# boundary, blue the snapped (aligned) boundary, grey the explicit-request baseline.
NATURAL, SNAPPED = SERIES[1], SERIES[0]
PAPER_NATURAL, PAPER_SNAPPED, PAPER_BASELINE = "#c0392b", "#2c7fb8", "#7f8c8d"
PAPER_EXACT_CELL, PAPER_CELL, PAPER_RULE = "#08519c", "#f7fbff", "#d0d0d0"


def use_paper_style() -> None:
    """Draw later figures in the paper's style, with DejaVu Sans embedded as Type 42."""
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})


def _pt(size, paper):
    """Font size in points: as given for the README figure, scaled up for the paper figure."""
    return size * PAPER_FONT_SCALE if paper else size


def _save(fig, path, paper=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if paper:
        fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _print_style(ax):
    """A full black frame with outward ticks, black text, no grid and a framed legend (paper only)."""
    ax.set_facecolor("white")
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(INK_PRIMARY)
        side.set_linewidth(0.8)
    ax.tick_params(which="both", labelsize=_pt(9, True), labelcolor=INK_PRIMARY, color=INK_PRIMARY,
                   direction="out", width=0.8, length=3)
    ax.tick_params(which="minor", length=1.5)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(INK_PRIMARY)
    legend = ax.get_legend()
    if legend:
        legend.set_frame_on(True)
        legend.get_frame().set_edgecolor("0.8")
        for text in legend.get_texts():
            text.set_color(INK_PRIMARY)


def _titles(fig, title, subtitle):
    fig.text(0.01, 1.0, title, ha="left", va="bottom", fontsize=12.5, color=INK_PRIMARY, weight="semibold")
    fig.text(0.01, 0.985, subtitle, ha="left", va="top", fontsize=9, color=INK_SECONDARY)


def _finish(fig, path, title, subtitle, top, paper):
    """Title block and layout for the README figure; neither for the paper figure."""
    if paper:
        for ax in fig.axes:
            _print_style(ax)
        fig.tight_layout()
    else:
        _titles(fig, title, subtitle)
        fig.tight_layout(rect=(0, 0, 1, top))
    _save(fig, path, paper)


# ------------------------------------------------------------------ exactness grid (G1)
def load_g1(paths) -> dict:
    """{model: {n_ubatch: {k: (exact, max_tv)}}} from G1 reports named g1_<model>_*.json."""
    grid = {}
    for path in paths:
        model = Path(path).name.split("_")[1]
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        cells = {c["k"]: (bool(c["exact"]), float(c.get("max_tv", 0.0)))
                 for c in report["cases"] if "exact" in c}
        grid.setdefault(model, {})[int(report["ubatch"])] = cells
    return grid


def plot_exactness_grid(grid, path, paper=False):
    models = [m for m in ORDER if m in grid]
    fig, axes = plt.subplots(1, len(models),
                             figsize=PAPER_SIZES["g1_exactness_grid"] if paper else (3.3 * len(models) + 0.6, 5.6),
                             facecolor=SURFACE, squeeze=False)
    for ax, m in zip(axes[0], models):
        ubs = sorted(grid[m])
        ks = sorted({k for ub in ubs for k in grid[m][ub]})
        ax.set_facecolor(SURFACE)
        for j, ub in enumerate(ubs):
            for i, k in enumerate(ks):
                if k not in grid[m][ub]:
                    continue
                exact, tv = grid[m][ub][k]
                if paper:   # a seamless table with hairline rules
                    fill = PAPER_EXACT_CELL if exact else PAPER_CELL
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fill, edgecolor=PAPER_RULE, linewidth=0.4))
                else:
                    fill = SNAPPED if exact else "#efeee9"
                    ax.add_patch(Rectangle((j + 0.03, i + 0.05), 0.94, 0.9, linewidth=0, facecolor=fill))
                ax.text(j + 0.5, i + 0.5, "exact" if exact else f"{tv:.3f}", ha="center", va="center",
                        fontsize=_pt(8.5, paper), color=SURFACE if exact else (INK_PRIMARY if paper else INK_SECONDARY),
                        weight="semibold" if exact and not paper else "normal")
        ax.set_xlim(0, len(ubs))
        ax.set_ylim(len(ks), 0)
        ax.set_xticks([j + 0.5 for j in range(len(ubs))])
        ax.set_xticklabels([str(ub) for ub in ubs])
        ax.set_yticks([i + 0.5 for i in range(len(ks))])
        ax.set_yticklabels([str(k) for k in ks])
        if not paper:   # the paper keeps the axis at the bottom, clear of the panel title
            ax.xaxis.tick_top()
        ax.tick_params(length=0, colors=INK_SECONDARY, labelsize=_pt(9, paper))
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel("n_ubatch (micro-batch size)", color=INK_SECONDARY, fontsize=_pt(9, paper),
                      labelpad=4 if paper else 8)
        if not paper:
            ax.xaxis.set_label_position("top")
        if paper:
            letter = "(" + "abc"[models.index(m)] + ") "
            ax.set_title(letter + MODEL_LABELS.get(m, m).replace(" · ", ", "), loc="left",
                         fontsize=_pt(9, paper), color=INK_PRIMARY, pad=5)
        else:
            ax.set_title(MODEL_LABELS.get(m, m), loc="left", fontsize=_pt(10, paper), color=INK_PRIMARY, pad=34)
    axes[0][0].set_ylabel("reuse boundary k (tokens)", color=INK_SECONDARY, fontsize=_pt(9, paper))
    _finish(fig, path, "Restore is bit-exact exactly when the reuse boundary is a multiple of n_ubatch",
            "Gate G1 · one ~1.9K-token prompt per model · inexact cells show the worst-position "
            "total variation against recompute", 0.93, paper)


# ----------------------------------------------------------------- action changes (main)
def action_change_points(analysis) -> list:
    """Family A members of the ambiguous stratum, in plotting order."""
    fam, out = analysis["families"]["A"], []
    for m in ORDER:
        for band in ("short", "long"):
            for boundary in ("natural", "snapped"):
                v = fam.get(f"A:{m}:{boundary}:{band}:ambiguous")
                if v is None or v["interval"] is None:
                    continue
                out.append({"model": m, "band": band, "boundary": boundary, "events": v["events"],
                            "n": v["n"], "estimate": v["estimate"], "low": v["interval"]["low"],
                            "high": v["interval"]["high"], "claimed": v["claimed"]})
    return out


def explicit_totals(analysis) -> tuple[int, int]:
    """(action changes, turns) over every explicit-request family A member."""
    members = [v for k, v in analysis["families"]["A"].items() if k.endswith(":explicit")]
    return sum(v["events"] for v in members), sum(v["n"] for v in members)


def plot_action_changes(analysis, path, paper=False):
    points = action_change_points(analysis)
    rows = [(m, b) for m in ORDER for b in ("short", "long") if any(p["model"] == m and p["band"] == b for p in points)]
    y_of, y = {}, 0.0
    for idx, (m, b) in enumerate(rows):
        if idx and rows[idx - 1][0] != m:
            y += 0.6                                       # gap between models
        y_of[(m, b)] = y
        y += 1.0
    fig, ax = plt.subplots(figsize=PAPER_SIZES["action_changes"] if paper else (8.4, 5.0), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    nat, snap = (PAPER_NATURAL, PAPER_SNAPPED) if paper else (NATURAL, SNAPPED)
    styles = {"natural": (nat, MARKERS[1], -0.21 if paper else -0.17,
                          "natural boundary" if paper else "natural turn boundary"),
              "snapped": (snap, MARKERS[0], 0.21 if paper else 0.17,
                          "snapped boundary" if paper else "boundary snapped to a multiple of n_ubatch")}
    for boundary, (color, marker, dy, label) in styles.items():
        sel = [p for p in points if p["boundary"] == boundary]
        ys = [y_of[(p["model"], p["band"])] + dy for p in sel]
        xs = [100 * p["estimate"] for p in sel]
        if paper:
            err = [[x - 100 * p["low"] for x, p in zip(xs, sel)], [100 * p["high"] - x for x, p in zip(xs, sel)]]
            ax.errorbar(xs, ys, xerr=err, fmt=marker, color=color, zorder=3, label=label, **PAPER_ERR)
        else:
            ax.hlines(ys, [100 * p["low"] for p in sel], [100 * p["high"] for p in sel],
                      color=WHISKER, linewidth=1.2, zorder=2)
            ax.scatter(xs, ys, color=color, marker=marker, s=62, edgecolor=SURFACE, linewidth=1.3,
                       zorder=3, label=label)
        for p, yy in zip(sel, ys):
            ax.text(100 * p["high"] + 0.5, yy, f"{p['events']}/{p['n']}", va="center", fontsize=_pt(8.5, paper),
                    color=INK_PRIMARY if p["events"] else INK_SECONDARY)
    ax.set_yticks([y_of[r] for r in rows])
    ax.set_yticklabels([f"{PAPER_LABELS[m]}, {b}" if paper else f"{SHORT_LABELS[m]} · {b} history"
                        for m, b in rows], color=INK_SECONDARY)
    ax.set_ylim(y - 0.4, -0.6)
    pad = 7 if paper else 4      # room for the count label beside the widest interval
    ax.set_xlim(-0.8, max(26.0, 100 * max(p["high"] for p in points) + pad))
    ax.set_xlabel("turns where restore changed the action (%)" if paper
                  else "turns where restore changed the agent's action (%)", color=INK_SECONDARY, fontsize=_pt(9, paper))
    ax.legend(frameon=False, fontsize=_pt(9, paper), labelcolor=INK_SECONDARY, loc="upper center",
              bbox_to_anchor=(0.4, -0.2) if paper else (0.45, -0.13), ncol=2)   # below the axis: every row carries a label
    changes, turns = explicit_totals(analysis)
    _finish(fig, path, "Restore changed the agent's action on ambiguous requests, on every architecture",
            f"Main study · 90 turns per point, both replicates · 95% Clopper–Pearson intervals · "
            f"explicit requests: {changes} changes in {turns:,} restores", 0.94, paper)


# ------------------------------------------------------------------- margin bins (main)
def margin_points(analysis, boundary="natural") -> list:
    out = []
    for m in ORDER:
        bins = analysis["secondary"].get(m, {}).get("margin_bins", {}).get(boundary)
        if not bins:
            continue
        for label in BIN_LABELS:
            v = bins[label]
            if v["n"]:
                out.append({"model": m, "bin": label, "events": v["events"], "n": v["n"], "rate": v["rate"],
                            "low": v["interval"]["low"], "high": v["interval"]["high"]})
    return out


def plot_margin_bins(analysis, path, paper=False):
    points = margin_points(analysis)
    models = [m for m in ORDER if any(p["model"] == m for p in points)]
    fig, ax = plt.subplots(figsize=PAPER_SIZES["margin_bins"] if paper else (8.4, 4.6), facecolor=SURFACE)
    _style(ax)
    width = 0.3 if paper else 0.24
    labels = PAPER_LABELS if paper else SHORT_LABELS
    for i, m in enumerate(models):
        sel = {p["bin"]: p for p in points if p["model"] == m}
        dx = (i - (len(models) - 1) / 2) * width
        for j, label in enumerate(BIN_LABELS):
            if label not in sel:
                continue
            p = sel[label]
            # Every point is a natural-boundary rate, so all wear the natural color; shape names the model.
            if paper:
                r = 100 * p["rate"]
                ax.errorbar([j + dx], [r], yerr=[[r - 100 * p["low"]], [100 * p["high"] - r]], fmt=MARKERS[i],
                            color=PAPER_NATURAL, zorder=3, label=labels[m] if j == 0 else None, **PAPER_ERR)
            else:
                ax.vlines(j + dx, 100 * p["low"], 100 * p["high"], color=WHISKER, linewidth=1.2, zorder=2)
                ax.scatter([j + dx], [100 * p["rate"]], color=NATURAL, marker=MARKERS[i], s=64,
                           edgecolor=SURFACE, linewidth=1.3, zorder=3, label=labels[m] if j == 0 else None)
            lift = 6.0 if paper and i == 1 else 1.8
            ax.text(j + dx, 100 * p["high"] + lift, f"{p['events']}/{p['n']}", ha="center",
                    fontsize=_pt(7.5 if paper else 8, paper), color=INK_SECONDARY)
    total = {label: (sum(p["events"] for p in points if p["bin"] == label),
                     sum(p["n"] for p in points if p["bin"] == label)) for label in BIN_LABELS}
    ax.set_xticks(range(len(BIN_LABELS)))
    ax.set_xticklabels([BIN_NAMES[b] if paper else f"{BIN_NAMES[b]}\nall models {total[b][0]}/{total[b][1]}"
                        for b in BIN_LABELS], color=INK_SECONDARY)
    ax.set_xlim(-0.6, len(BIN_LABELS) - 0.4)
    ax.set_ylim(0, max(70.0, 100 * max(p["high"] for p in points) + 8))
    ax.set_ylabel("action changed (% of turns)" if paper else "turns where restore changed the action (%)",
                  color=INK_SECONDARY, fontsize=_pt(9, paper))
    ax.legend(frameon=False, fontsize=_pt(9, paper), labelcolor=INK_SECONDARY, loc="upper right")
    _finish(fig, path, "Changes happen where the model was already close to a tie",
            "Natural boundaries, all turns · action margin = smallest top-two logprob gap at the decision, "
            "on the recompute arm · 95% Clopper–Pearson intervals", 0.93, paper)


# ---------------------------------------------------------------- drift per turn (main)
def drift_values(analysis, boundary="natural") -> dict:
    """{(model, stratum): [per-turn mean JSD]} over valid turns."""
    out = {}
    for r in analysis["rows"]:
        if r["boundary"] == boundary and r["excluded"] is None:
            out.setdefault((r["model"], r["stratum"]), []).append(r["mean_jsd"])
    return out


def plot_drift_per_turn(analysis, path, seed=0, paper=False):
    values = drift_values(analysis)
    rng = np.random.default_rng(seed)
    models = [m for m in ORDER if any(k[0] == m for k in values)]
    fig, ax = plt.subplots(figsize=PAPER_SIZES["drift_per_turn"] if paper else (8.4, 4.8), facecolor=SURFACE)
    _style(ax)
    ax.set_yscale("log")
    strata = {"explicit": (PAPER_BASELINE if paper else WHISKER, MARKERS[0], -0.18),
              "ambiguous": (PAPER_NATURAL if paper else NATURAL, MARKERS[1], 0.18)}
    for i, m in enumerate(models):
        for stratum, (color, marker, dx) in strata.items():
            vals = np.asarray(values.get((m, stratum), []), dtype=float)
            if vals.size == 0:
                continue
            shown = np.maximum(vals, JSD_FLOOR)
            xs = i + dx + rng.uniform(-0.09, 0.09, size=vals.size)
            ax.scatter(xs, shown, s=5 if paper else 9, color=color, marker=marker, alpha=0.45, linewidth=0, zorder=2,
                       label=f"{stratum} requests" if i == 0 else None)
            med = float(np.median(vals))
            ax.hlines(max(med, JSD_FLOOR), i + dx - 0.13, i + dx + 0.13, color=INK_PRIMARY,
                      linewidth=1.2 if paper else 2, zorder=3)
            if not paper:
                ax.text(i + dx + 0.15, max(med, JSD_FLOOR), f"median {med:.1e}", ha="left", va="center",
                        fontsize=7.5, color=INK_PRIMARY,
                        bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5}, zorder=4)
            floored = int((vals < JSD_FLOOR).sum())
            if floored:
                ax.text(i + dx, JSD_FLOOR * 0.5, f"{floored} < 1e-10", ha="center", va="top", fontsize=_pt(7, paper),
                        color=INK_SECONDARY)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([PAPER_LABELS[m].replace(" (", "\n(") if paper else SHORT_LABELS[m] for m in models],
                       color=INK_SECONDARY)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(JSD_FLOOR * 0.5, 3.0)
    ax.set_ylabel("per-turn mean JSD (nats)" if paper else "per-turn mean JSD, restore vs recompute (nats)",
                  color=INK_SECONDARY, fontsize=_pt(9, paper))
    if paper:   # above the axes: inside, it covers the hybrid's points
        ax.legend(frameon=False, fontsize=_pt(9, paper), labelcolor=INK_SECONDARY, markerscale=2.5,
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    else:
        ax.legend(frameon=False, fontsize=_pt(9, paper), labelcolor=INK_SECONDARY, loc="lower left", markerscale=2.5)
    _finish(fig, path, "Every natural-boundary restore drifted; the hybrid's median drift is the highest",
            "Main study · one point per turn, both prefix bands · snapped restores were exactly 0 on Ornith and "
            "Qwen3, and on Spark at k = 512", 0.93, paper)
