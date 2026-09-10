"""Main-study analysis, as pre-registered (docs/preregistration/2026-09-10-analysis-plan.md).

Reads the runner's records (servergrid.load_records) for each model and produces:

  turn table     one row per (model, turn, boundary): arm validity, each arm's
                 action, action change in both replicates, exactness, first
                 divergence, drift, the recompute floor, reproducibility, and the
                 action margin (sections 5 and 6)
  families       A (24), B (16) and C (6), each Holm-corrected within itself, with
                 the decision rules of section 8 applied member by member
  hypotheses     H1-H7 as section 8 states them. Per-member claims are the
                 inferential units; a hypothesis is "supported" when every member
                 it names is claimed and "partial" when only some are
  secondary      section 9, intervals only
  exploratory    the architecture contrasts C3 and C4 within each prefix band,
                 committed in docs/findings/2026-09-10-study-manifest.md, because
                 the three models reuse different token lengths for the same turn

Effects are restore minus floor on the same turn (spec 6.1). With a bit-exact
floor they equal the restore arm's own divergence.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np

from .config import ARCH_LEVELS
from .g3 import records_exact
from .metrics.behavioral import ToolCall
from .metrics.distributional import context_matched_drift
from .metrics.margin import BIN_LABELS, decision_margin, margin_bin
from .runner.llamaserver import first_divergence
from .servergrid import restore_record
from .stats.bootstrap import is_self_inconsistent, paired_bootstrap
from .stats.exact import clopper_pearson, sign_test, signs, test_against_floor
from .stats.multiplicity import holm
from .traces.roundtrip import round_trips
from .traces.toolcalls import parse_tool_call_ordered

MODELS = tuple(sorted(ARCH_LEVELS, key=ARCH_LEVELS.get, reverse=True))   # L3, L2, L1
L3, L2, L1 = MODELS
BOUNDARIES = ("natural", "snapped")
BANDS = ("short", "long")
STRATA = ("explicit", "ambiguous")
MAX_EXCLUDED = 0.10          # section 6: more than this and no claim is made for the condition
NONDETERMINISTIC = 0.05      # section 6: floor inexact on more than this share of a model's turns


@dataclass(frozen=True)
class ModelData:
    model: str
    turns: dict
    arms: dict
    template: str
    fmt: str
    specials: dict = field(default_factory=dict)


# ------------------------------------------------------------------------ turn table
def action_of(record, template, fmt, specials=None):
    """(action, faithful). action is (tool name, argument hash), or None for no call."""
    parsed = parse_tool_call_ordered(record.get("content") or "", fmt)
    if parsed is None:
        return None, True
    name, args, block = parsed
    call = ToolCall.from_payload(name, args)
    return (call.name, call.args_hash), round_trips(template, block, name, args, **(specials or {}))


def drift(a, b):
    """(mean JSD, max JSD, first divergence) over context-matched positions."""
    fd = first_divergence(a["tokens"], b["tokens"])
    js, _ = context_matched_drift(a["top"], b["top"], fd,
                                  max_positions=max(len(a["top"]), len(b["top"]), 1))
    return (float(np.mean(js)) if js else 0.0), (float(max(js)) if js else 0.0), fd


def _state_equal(turn, boundary):
    states = turn.get("state_bytes") or {}
    key = "natural" if boundary == "snapped" and turn["snapped_is_natural"] else boundary
    return (states.get(key) or {}).get("equal")


def turn_table(md: ModelData):
    rows, actions = [], {}

    def act(rec):
        if id(rec) not in actions:
            actions[id(rec)] = action_of(rec, md.template, md.fmt, md.specials)
        return actions[id(rec)]

    for conv_id, turn in sorted(md.turns.items()):
        rc = {r: md.arms.get((conv_id, "recompute", None, r)) for r in (1, 2)}
        for boundary in BOUNDARIES:
            rs = {r: restore_record(md.arms, turn, boundary, r) for r in (1, 2)}
            row = {"model": md.model, "conv_id": conv_id, "base_id": turn["base_id"],
                   "prefix_band": turn["prefix_band"], "stratum": turn["stratum"], "boundary": boundary,
                   "k": turn["k_natural"] if boundary == "natural" else turn["k_snapped"],
                   "k_mod_ubatch_natural": turn["k_mod_ubatch"], "excluded": None}
            arms = (rc[1], rc[2], rs[1], rs[2])
            if any(a is None or a["outcome"] != "ok" for a in arms):
                row["excluded"] = "invalid_arm"
                rows.append(row)
                continue
            (c1, f1), (c2, f2), (s1, f3), (s2, f4) = (act(a) for a in arms)
            if not (f1 and f2 and f3 and f4):
                row["excluded"] = "parse_unfaithful"
                rows.append(row)
                continue
            mean_js, max_js, fd = drift(rs[1], rc[1])
            floor_mean, floor_max, _ = drift(rc[2], rc[1])
            margin, _ = decision_margin(rc[1]["top"], rc[1].get("pieces") or ())
            row.update({
                "action_change": int(s1 != c1 and s2 != c2),
                "action_change_any": int(s1 != c1 or s2 != c2),
                "floor_action_change": int(c1 != c2),
                "exact": records_exact(rs[1], rc[1]),
                "exact_both": records_exact(rs[1], rc[1]) and records_exact(rs[2], rc[2]),
                "exact_match": rs[1]["tokens"] == rc[1]["tokens"],
                "first_divergence": fd,
                "output_len": min(len(rs[1]["tokens"]), len(rc[1]["tokens"])),
                "mean_jsd": mean_js, "max_jsd": max_js,
                "floor_exact": records_exact(rc[1], rc[2]),
                "floor_mean_jsd": floor_mean, "floor_max_jsd": floor_max,
                "restore_reproducible": records_exact(rs[1], rs[2]),
                "margin": margin, "margin_bin": margin_bin(margin),
                "state_bytes_equal": _state_equal(turn, boundary),
            })
            rows.append(row)
    return rows


# -------------------------------------------------------------------------- families
def _select(rows, **match):
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def _interval(method, low, high):
    return {"method": method, "low": float(low), "high": float(high)}


def floor_member(cond_rows, treat_key, floor_key, kind):
    """One family A or B member: restore against the floor over a condition's turns."""
    valid = [r for r in cond_rows if r["excluded"] is None]
    total, excluded = len(cond_rows), len(cond_rows) - len(valid)
    member = {"n": len(valid), "excluded": excluded,
              "too_many_excluded": total == 0 or excluded / total > MAX_EXCLUDED}
    if not valid:
        return {**member, "p_value": 1.0, "method": None, "estimate": None, "interval": None, "events": 0}
    t = [float(r[treat_key]) for r in valid]
    f = [float(r[floor_key]) for r in valid]
    floor_exact = all(r["floor_exact"] for r in valid)
    test = test_against_floor(t, f, floor_exact=floor_exact)
    if kind == "rate" and floor_exact:
        x = int(sum(t))
        member.update(estimate=x / len(t), interval=_interval("clopper_pearson", *clopper_pearson(x, len(t))))
    else:
        ci = paired_bootstrap(np.subtract(t, f))
        member.update(estimate=ci.point, interval=_interval("bootstrap", ci.low, ci.high))
    return {**member, "p_value": test.p_value, "method": test.method, "events": test.n_pos}


def _decide_against_floor(members):
    rejected = holm({k: v["p_value"] for k, v in members.items()})
    for k, v in members.items():
        v["holm_reject"] = rejected[k]
        v["claimed"] = bool(rejected[k] and v["interval"] is not None and v["interval"]["low"] > 0
                            and not v["too_many_excluded"])
    return members


def family_a(rows, models):
    members = {}
    for m in models:
        for b in BOUNDARIES:
            for p in BANDS:
                for s in STRATA:
                    cond = _select(rows, model=m, boundary=b, prefix_band=p, stratum=s)
                    members[f"A:{m}:{b}:{p}:{s}"] = floor_member(cond, "action_change", "floor_action_change", "rate")
    return _decide_against_floor(members)


def family_b(rows):
    members = {}
    for b in BOUNDARIES:
        for p in BANDS:
            for s in STRATA:
                cond = _select(rows, model=L3, boundary=b, prefix_band=p, stratum=s)
                for stat in ("mean", "max"):
                    members[f"B:{L3}:{b}:{p}:{s}:{stat}_jsd"] = floor_member(
                        cond, f"{stat}_jsd", f"floor_{stat}_jsd", "mean")
    return _decide_against_floor(members)


def _valid_by_turn(rows, model, boundary):
    return {r["conv_id"]: r for r in rows
            if r["model"] == model and r["boundary"] == boundary and r["excluded"] is None}


def _jsd_effect(r):
    return r["mean_jsd"] - r["floor_mean_jsd"]


def _action_effect(r):
    return r["action_change"] - r["floor_action_change"]


def contrast(diffs):
    diffs = [float(d) for d in diffs]
    if not diffs:
        return {"n": 0, "n_pos": 0, "n_neg": 0, "p_value": 1.0, "estimate": None, "interval": None}
    n_pos, n_neg = signs(diffs, [0.0] * len(diffs))
    ci = paired_bootstrap(diffs)
    return {"n": len(diffs), "n_pos": n_pos, "n_neg": n_neg, "p_value": sign_test(n_pos, n_neg),
            "estimate": ci.point, "interval": _interval("bootstrap", ci.low, ci.high)}


def _paired(first, second, effect):
    return [effect(first[c]) - effect(second[c]) for c in sorted(first) if c in second]


def _within_base(rows_by_turn, pair_key, side_key, high, low, effect):
    by = {(r["base_id"], r[pair_key], r[side_key]): r for r in rows_by_turn.values()}
    return [effect(by[(b, k, high)]) - effect(by[(b, k, low)])
            for (b, k, side) in sorted(by) if side == high and (b, k, low) in by]


def family_c(rows):
    nat3, snap3 = _valid_by_turn(rows, L3, "natural"), _valid_by_turn(rows, L3, "snapped")
    nat2 = _valid_by_turn(rows, L2, "natural")
    nat1, snap1 = _valid_by_turn(rows, L1, "natural"), _valid_by_turn(rows, L1, "snapped")
    c2 = [(_jsd_effect(nat3[c]) - _jsd_effect(snap3[c])) - (_jsd_effect(nat1[c]) - _jsd_effect(snap1[c]))
          for c in sorted(nat3) if c in snap3 and c in nat1 and c in snap1]
    members = {
        "C1_snapping_L3": contrast(_paired(nat3, snap3, _jsd_effect)),
        "C2_snapping_x_architecture": contrast(c2),
        "C3_L3_minus_L1": contrast(_paired(nat3, nat1, _jsd_effect)),
        "C4_L3_minus_L2": contrast(_paired(nat3, nat2, _jsd_effect)),
        "C5_compounding_L3": contrast(_within_base(nat3, "stratum", "prefix_band", "long", "short", _jsd_effect)),
        "C6_stratum_L3": contrast(_within_base(nat3, "prefix_band", "stratum", "ambiguous", "explicit", _action_effect)),
    }
    rejected = holm({k: v["p_value"] for k, v in members.items()})
    for k, v in members.items():
        low = v["interval"]["low"] if v["interval"] else 0.0
        high = v["interval"]["high"] if v["interval"] else 0.0
        v["holm_reject"] = rejected[k]
        v["claimed"] = bool(rejected[k] and low > 0)               # every prediction is > 0
        v["contradicts_prediction"] = bool(rejected[k] and high < 0)
    return members


# ------------------------------------------------------------------- determinism, H1-H7
def determinism(rows, models):
    out = {}
    for m in models:
        turns = [r for r in _select(rows, model=m, boundary="natural") if r["excluded"] is None]
        inexact = sum(not r["floor_exact"] for r in turns)
        out[m] = {"turns": len(turns), "floor_inexact": inexact,
                  "nondeterministic": bool(turns) and inexact / len(turns) > NONDETERMINISTIC}
    return out


def _mark_conditional(families, det):
    shaky = {m for m, d in det.items() if d["nondeterministic"]}
    for name, members in families.items():
        for key, v in members.items():
            involved = {key.split(":")[1]} if name in ("A", "B") else {L3, L2, L1}
            v["conditional"] = bool(involved & shaky)


def _summary(members):
    claimed = sum(v["claimed"] for v in members)
    if members and claimed == len(members):
        return "supported"
    return "partial" if claimed else "not supported"


def hypotheses(rows, fam_a, fam_b, fam_c, models):
    h1 = {}
    for m in models:
        snapped = [r for r in _select(rows, model=m, boundary="snapped") if r["excluded"] is None]
        inexact = [{"conv_id": r["conv_id"], "k": r["k"]} for r in snapped if not r["exact_both"]]
        h1[m] = {"valid_snapped_turns": len(snapped), "inexact": inexact, "holds": bool(snapped) and not inexact}
    pick = lambda fam, prefix: [v for k, v in fam.items() if k.startswith(prefix)]   # noqa: E731
    return {
        "H1_mitigation": h1,
        "H2_distributional_existence": _summary(pick(fam_b, f"B:{L3}:natural:")),
        "H3_behavioural_existence": _summary(pick(fam_a, f"A:{L3}:natural:")),
        "H4_architecture": _summary([fam_c["C3_L3_minus_L1"], fam_c["C4_L3_minus_L2"]]),
        "H5_snapping": _summary([fam_c["C1_snapping_L3"], fam_c["C2_snapping_x_architecture"]]),
        "H6_compounding": _summary([fam_c["C5_compounding_L3"]]),
        "H7_stratum": _summary([fam_c["C6_stratum_L3"]]),
    }


# ------------------------------------------------------------------ secondary analyses
def kaplan_meier(items):
    """Survival curve [(position, share not yet diverged)] from (position, diverged) pairs."""
    curve, survival = [], 1.0
    for t in sorted({d for d, event in items if event}):
        at_risk = sum(1 for d, _ in items if d >= t)
        events = sum(1 for d, event in items if event and d == t)
        survival *= 1.0 - events / at_risk
        curve.append((t, survival))
    return curve


def _rate(rows, key):
    n, x = len(rows), int(sum(r[key] for r in rows))
    if n == 0:
        return {"n": 0, "events": 0, "rate": None, "interval": None}
    return {"n": n, "events": x, "rate": x / n, "interval": _interval("clopper_pearson", *clopper_pearson(x, n))}


def secondary(rows, models):
    out = {}
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        valid = [r for r in mine if r["excluded"] is None]
        s = {"margin_bins": {}, "conditions": {}, "exclusions": {}}
        for b in BOUNDARIES:
            vb = [r for r in valid if r["boundary"] == b]
            s["margin_bins"][b] = {label: _rate([r for r in vb if r["margin_bin"] == label], "action_change")
                                   for label in BIN_LABELS}
            s["nonreproducible_action_changes_" + b] = sum(r["action_change_any"] and not r["action_change"] for r in vb)
            states = [r["state_bytes_equal"] for r in vb]
            s["state_bytes_" + b] = {"equal": states.count(True), "differ": states.count(False),
                                     "unavailable": states.count(None)}
            for p in BANDS:
                for st in STRATA:
                    cond = [r for r in vb if r["prefix_band"] == p and r["stratum"] == st]
                    key = f"{b}:{p}:{st}"
                    med = statistics.median(r["mean_jsd"] for r in cond) if cond else None
                    floor_med = statistics.median(r["floor_mean_jsd"] for r in cond) if cond else None
                    s["conditions"][key] = {
                        "exact_match": _rate(cond, "exact_match"),
                        "exact": _rate(cond, "exact"),
                        "first_divergence_survival": kaplan_meier(
                            [(r["first_divergence"], True) if r["first_divergence"] is not None
                             else (r["output_len"], False) for r in cond]),
                        "median_mean_jsd": med, "floor_median_mean_jsd": floor_med,
                        "self_inconsistent": None if med is None else is_self_inconsistent(med, floor_med),
                    }
                    reasons = [r["excluded"] for r in mine if r["boundary"] == b and r["prefix_band"] == p
                               and r["stratum"] == st and r["excluded"]]
                    s["exclusions"][key] = {x: reasons.count(x) for x in sorted(set(reasons))}
        natural = [r for r in valid if r["boundary"] == "natural"]
        s["alignment"] = {"aligned": _rate([r for r in natural if r["k_mod_ubatch_natural"] == 0], "action_change"),
                          "misaligned": _rate([r for r in natural if r["k_mod_ubatch_natural"] != 0], "action_change")}
        costs = {}
        for p in BANDS:
            c = sorted({r["conv_id"]: r["k_mod_ubatch_natural"] for r in mine if r["prefix_band"] == p}.values())
            costs[p] = {"min": c[0], "median": statistics.median(c), "max": c[-1]} if c else None
        s["snapping_cost_tokens"] = costs
        out[m] = s
    return out


def exploratory(rows):
    """C3 and C4 within each prefix band. Exploratory: not in any Holm family."""
    nat = {m: _valid_by_turn(rows, m, "natural") for m in (L3, L2, L1)}
    out = {}
    for band in BANDS:
        in_band = {m: {c: r for c, r in v.items() if r["prefix_band"] == band} for m, v in nat.items()}
        out[band] = {"L3_minus_L1": contrast(_paired(in_band[L3], in_band[L1], _jsd_effect)),
                     "L3_minus_L2": contrast(_paired(in_band[L3], in_band[L2], _jsd_effect))}
    return out


def run_analysis(datasets) -> dict:
    rows = [r for md in datasets for r in turn_table(md)]
    models = [md.model for md in datasets]
    families = {"A": family_a(rows, models), "B": family_b(rows), "C": family_c(rows)}
    det = determinism(rows, models)
    _mark_conditional(families, det)
    return {"models": models, "determinism": det, "families": families,
            "hypotheses": hypotheses(rows, families["A"], families["B"], families["C"], models),
            "secondary": secondary(rows, models), "exploratory": exploratory(rows), "rows": rows}
