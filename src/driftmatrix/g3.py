"""Gate G3 on pilot turns: can the harness see what it is meant to see? (pre-registration 11.2)

Checked on one model's pilot records, and nothing else is looked at:

  arm proofs         every arm of every turn is `ok`
  snapped exactness  in both replicates, restore at the snapped boundary is bit-exact
                     against recompute: identical tokens and top-k logprobs
  positive control   on the hybrid (level 3) only, at least one natural-boundary
                     restore shows nonzero drift against recompute in replicate 1

Pilot turns use their own seed and never enter the confirmatory analysis.
Floor exactness (recompute against itself) is reported but does not gate.
"""
from __future__ import annotations

from .config import ARCH_LEVELS
from .metrics.distributional import context_matched_drift
from .runner.llamaserver import first_divergence
from .servergrid import restore_record


def records_exact(a, b) -> bool:
    return a["tokens"] == b["tokens"] and a["top"] == b["top"]


def drift_seen(a, b) -> bool:
    fd = first_divergence(a["tokens"], b["tokens"])
    positions = max(len(a["top"]), len(b["top"]), 1)
    _, tv = context_matched_drift(a["top"], b["top"], fd, max_positions=positions)
    return fd is not None or any(x > 0 for x in tv)


def g3_check(meta, turns, arms) -> dict:
    model = meta["model"]
    hybrid = ARCH_LEVELS.get(model) == 3
    invalid = sorted((a["conv_id"], a["arm"], a["boundary"] or "", a["replicate"], a["outcome"])
                     for a in arms.values() if a["outcome"] != "ok")
    missing, snapped_inexact, positive, floor_inexact = [], [], 0, 0
    for conv_id, turn in sorted(turns.items()):
        for rep in (1, 2):
            rc = arms.get((conv_id, "recompute", None, rep))
            rn, rs = restore_record(arms, turn, "natural", rep), restore_record(arms, turn, "snapped", rep)
            if rc is None or rn is None or rs is None:
                missing.append((conv_id, rep))
                continue
            if rc["outcome"] == rs["outcome"] == "ok" and not records_exact(rs, rc):
                snapped_inexact.append({"conv_id": conv_id, "replicate": rep, "k_snapped": turn["k_snapped"]})
            if rep == 1 and rc["outcome"] == rn["outcome"] == "ok" and drift_seen(rn, rc):
                positive += 1
        rc1, rc2 = arms.get((conv_id, "recompute", None, 1)), arms.get((conv_id, "recompute", None, 2))
        if rc1 and rc2 and rc1["outcome"] == rc2["outcome"] == "ok" and not records_exact(rc1, rc2):
            floor_inexact += 1

    if not turns:
        verdict = "NO_TURNS"
    elif invalid or missing:
        verdict = "ARM_PROOFS_FAIL"
    elif snapped_inexact:
        verdict = "SNAPPED_NOT_EXACT"
    elif hybrid and positive == 0:
        verdict = "POSITIVE_CONTROL_SILENT"
    else:
        verdict = "G3_PASS"
    return {"model": model, "turns": len(turns), "invalid_arms": invalid, "missing_arms": missing,
            "snapped_inexact": snapped_inexact, "positive_control_required": hybrid,
            "natural_turns_with_drift": positive, "floor_inexact_turns": floor_inexact, "verdict": verdict}
