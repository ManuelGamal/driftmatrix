"""Exact tests and intervals for per-turn outcomes (pre-registration section 5).

Turns are the unit. Two situations need different tests.

Against the recompute floor. The floor is the divergence between the recompute
arm's own two replicates on the same turn. If it is zero on every turn, the null
"restore changes nothing" predicts zero divergence on every turn with certainty,
because the engine is deterministic. One diverging turn then rejects it: an
exact binomial test against a rate of 0 gives p = 0 when any turn diverges and
p = 1 otherwise. This is the test stats/power.py assumes. If the floor is
nonzero on any turn, restore and floor are compared by a one-sided exact sign
test on their per-turn difference, ties dropped.

Between two treatment conditions (the mechanism contrasts), neither side is a
floor, so the comparison is a two-sided exact sign test on paired per-turn
differences. For binary outcomes that is the exact McNemar test.

Rates are reported with Clopper-Pearson intervals. A percentile bootstrap of a
proportion with no events returns [0, 0], which cannot bound a null.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

from ..config import ALPHA


def clopper_pearson(x: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Exact two-sided (1 - alpha) interval for a binomial proportion."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= x <= n:
        raise ValueError(f"x must be in [0, n], got {x} of {n}")
    ci = binomtest(x, n).proportion_ci(confidence_level=1.0 - alpha, method="exact")
    return float(ci.low), float(ci.high)


def sign_test(n_pos: int, n_neg: int, alternative: str = "two-sided") -> float:
    """Exact sign test on counts of positive and negative paired differences."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    return float(binomtest(n_pos, n, 0.5, alternative=alternative).pvalue)


def signs(a, b) -> tuple[int, int]:
    """Counts of turns where a > b and where a < b."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"paired arrays differ in shape: {x.shape} vs {y.shape}")
    return int(np.sum(x > y)), int(np.sum(x < y))


@dataclass(frozen=True)
class FloorTest:
    p_value: float
    method: str          # "zero_floor" or "sign_test"
    n: int
    n_pos: int
    n_neg: int


def test_against_floor(treatment, floor) -> FloorTest:
    """Does restore diverge more than the recompute floor on the same turns?"""
    t, f = np.asarray(treatment, dtype=np.float64), np.asarray(floor, dtype=np.float64)
    if t.size == 0:
        raise ValueError("no turns")
    n_pos, n_neg = signs(t, f)
    if not np.any(f):
        return FloorTest(0.0 if np.any(t) else 1.0, "zero_floor", int(t.size), n_pos, n_neg)
    return FloorTest(sign_test(n_pos, n_neg, "greater"), "sign_test", int(t.size), n_pos, n_neg)


test_against_floor.__test__ = False   # not a pytest test despite its name
