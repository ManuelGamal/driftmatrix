"""Holm-Bonferroni within three declared families (spec 6.2).

Family-wise error control, not FDR: these are bug claims and a false positive
is expensive. Families are never pooled - pooling unrelated hypotheses
over-corrects and hides real effects.
"""
from __future__ import annotations
from enum import Enum
from ..config import ALPHA


class Family(str, Enum):
    BEHAVIORAL = "behavioral"
    DISTRIBUTIONAL = "distributional"
    MECHANISM = "mechanism"


FAMILY_SIZES = {
    Family.BEHAVIORAL: 24,
    Family.DISTRIBUTIONAL: 16,
    Family.MECHANISM: 6,    # pre-registration adds hybrid vs sliding-window (spec 6.3.2)
}


def holm(pvalues: dict, alpha: float = ALPHA) -> dict:
    for k, p in pvalues.items():
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value out of range for {k}: {p}")
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out = {}
    still_rejecting = True
    for i, (key, p) in enumerate(ordered):
        if still_rejecting and p <= alpha / (m - i):
            out[key] = True
        else:
            still_rejecting = False
            out[key] = False
    return {k: out[k] for k in pvalues}


def holm_by_family(pvalues: dict, alpha: float = ALPHA) -> dict:
    return {fam: holm(tests, alpha) for fam, tests in pvalues.items()}
