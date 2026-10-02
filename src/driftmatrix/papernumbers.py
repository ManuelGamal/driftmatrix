"""The paper's numbers as LaTeX macros, computed from committed results under docs/findings.

  load_inputs   read analysis.json, the G1 reports and the Spark follow-up reports
  macros        {macro name: LaTeX value}; raises KeyError when an expected result is missing
  render        the text of docs/paper/numbers.tex

Percentages carry no % sign. sci() values need math mode.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

MODELS = {"ornith": "Ornith", "spark": "Spark", "qwen3": "Qwen"}
BOUNDARIES = {"natural": "Nat", "snapped": "Snap"}
BANDS = {"short": "Short", "long": "Long"}
STRATA = {"explicit": "Exp", "ambiguous": "Amb"}
BINS = {"near_tie": "NearTie", "contested": "Contested", "confident": "Confident"}
CONTRASTS = {"C1_snapping_L3": "COne", "C2_snapping_x_architecture": "CTwo", "C3_L3_minus_L1": "CThree",
             "C4_L3_minus_L2": "CFour", "C5_compounding_L3": "CFive", "C6_stratum_L3": "CSix"}
SPARK_UBATCHES = (256, 512, 1024)
SPARK_WINDOW = 512


def count(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def sci(x: float) -> str:
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def half_up(x: float) -> int:
    return int(x + 0.5)


def load_inputs(repo: Path):
    findings = Path(repo) / "docs" / "findings"
    analysis = json.loads((findings / "study" / "analysis.json").read_text(encoding="utf-8"))
    g1: dict[str, list[dict]] = {}
    for path in sorted((findings / "g1").glob("g1_*.json")):
        g1.setdefault(path.name.split("_")[1], []).append(json.loads(path.read_text(encoding="utf-8")))
    spark = {ub: json.loads((findings / "followup" / f"spark_ub{ub}.json").read_text(encoding="utf-8"))
             for ub in SPARK_UBATCHES}
    return analysis, g1, spark


def _family_a(analysis, out):
    fam = analysis["families"]["A"]
    explicit = []
    for m, mn in MODELS.items():
        for b, bn in BOUNDARIES.items():
            for p, pn in BANDS.items():
                for s, sn in STRATA.items():
                    v = fam[f"A:{m}:{b}:{p}:{s}"]
                    name = f"A{mn}{bn}{pn}{sn}"
                    out[name] = str(v["events"])
                    out[name + "N"] = str(v["n"])
                    out[name + "Pct"] = pct(v["estimate"])
                    out[name + "Lo"] = pct(v["interval"]["low"])
                    out[name + "Hi"] = pct(v["interval"]["high"])
                    if s == "explicit":
                        explicit.append(v)
    out["MembersA"] = str(len(fam))
    out["ClaimedA"] = str(sum(bool(v["claimed"]) for v in fam.values()))
    out["ChangesTotal"] = str(sum(v["events"] for v in fam.values()))
    out["ExplicitChanges"] = str(sum(v["events"] for v in explicit))
    out["ExplicitRestores"] = count(sum(v["n"] for v in explicit))
    out["ExplicitTurns"] = count(sum(v["n"] for v in explicit) // len(BOUNDARIES))


def _margins(analysis, out):
    for label, ln in BINS.items():
        events = turns = 0
        for m, mn in MODELS.items():
            v = analysis["secondary"][m]["margin_bins"]["natural"][label]
            out[f"Margin{mn}{ln}"] = str(v["events"])
            out[f"Margin{mn}{ln}N"] = str(v["n"])
            out[f"Margin{mn}{ln}Pct"] = pct(v["events"] / v["n"], 0)
            events, turns = events + v["events"], turns + v["n"]
        rate = events / turns
        out[f"MarginAll{ln}"] = str(events)
        out[f"MarginAll{ln}N"] = str(turns)
        out[f"MarginAll{ln}Pct"] = pct(rate, 0 if rate >= 0.1 else 1)


def _snapping(analysis, out):
    medians = []
    for m, mn in MODELS.items():
        for p, pn in BANDS.items():
            median = half_up(analysis["secondary"][m]["snapping_cost_tokens"][p]["median"])
            out[f"SnapCost{mn}{pn}"] = str(median)
            medians.append(median)
        h1 = analysis["hypotheses"]["H1_mitigation"][m]
        out[f"SnapInexact{mn}"] = str(len(h1["inexact"]))
        out[f"SnapValid{mn}"] = str(h1["valid_snapped_turns"])
    out["SnapCostMin"], out["SnapCostMax"] = str(min(medians)), str(max(medians))


def _family_c(analysis, out):
    for key, name in CONTRASTS.items():
        v = analysis["families"]["C"][key]
        out[name + "Pos"], out[name + "Neg"] = str(v["n_pos"]), str(v["n_neg"])
        out[name + "Est"] = sci(v["estimate"])
        out[name + "Lo"], out[name + "Hi"] = sci(v["interval"]["low"]), sci(v["interval"]["high"])


def _drift(analysis, out):
    for m, mn in MODELS.items():
        for p, pn in BANDS.items():
            for s, sn in STRATA.items():
                v = analysis["secondary"][m]["conditions"][f"natural:{p}:{s}"]
                out[f"Drift{mn}{pn}{sn}"] = sci(v["median_mean_jsd"])


def _g1(g1, out):
    restores = contradictions = 0
    worst = {}
    for m in ("ornith", "qwen3"):
        for report in g1[m]:
            cases = [c for c in report["cases"] if "exact" in c]
            restores += len(cases)
            contradictions += sum(c["exact"] != c["aligned"] for c in cases)
            worst.setdefault(m, []).extend(c["max_tv"] for c in cases if not c["exact"])
    out["GOneRestores"], out["GOneContradictions"] = str(restores), str(contradictions)
    out["GOneRatio"] = str(half_up(statistics.median(worst["ornith"]) / statistics.median(worst["qwen3"])))


def _spark(spark, out):
    cases = contradictions = 0
    for ub in SPARK_UBATCHES:
        proven = [c for c in spark[ub]["cases"] if c.get("restore_used") and "exact" in c]
        cases += len(proven)
        contradictions += sum(c["exact"] != (c["k"] % ub == 0 and c["k"] <= SPARK_WINDOW) for c in proven)
    out["SparkCases"], out["SparkContradictions"] = str(cases), str(contradictions)


def macros(analysis: dict, g1: dict, spark: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    turns = [analysis["determinism"][m]["turns"] for m in MODELS]
    out["TurnsPerModel"] = count(turns[0])
    out["TurnsTotal"] = count(sum(turns))
    out["FloorInexact"] = count(sum(analysis["determinism"][m]["floor_inexact"] for m in MODELS))
    natural = [r for r in analysis["rows"] if r["boundary"] == "natural" and r["excluded"] is None]
    out["NatTurns"] = count(len(natural))
    out["NatInexact"] = count(sum(not r["exact"] for r in natural))
    _family_a(analysis, out)
    _margins(analysis, out)
    _snapping(analysis, out)
    _family_c(analysis, out)
    _drift(analysis, out)
    _g1(g1, out)
    _spark(spark, out)
    return out


def render(macros: dict[str, str]) -> str:
    lines = ["% Generated by scripts/paper_numbers.py from docs/findings. Do not edit.",
             "% Percentages carry no % sign. Values with \\times need math mode."]
    for name in sorted(macros):
        value = macros[name]
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name {name} is not letters only")
        if re.search(r"[%_#&$]", value):
            raise ValueError(f"macro {name} has a value TeX would misread: {value}")
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    return "\n".join(lines) + "\n"
