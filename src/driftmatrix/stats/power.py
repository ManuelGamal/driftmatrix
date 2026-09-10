"""Power for detecting a rare per-turn divergence rate against a zero floor.

One-sided exact binomial. Under the null the divergence rate is 0, so any
observed divergence is evidence and the critical count is k=1. Power is
therefore 1 - (1 - rate)^n.
"""
from __future__ import annotations
import math
from ..config import ALPHA, TARGET_EFFECT_RATE, TARGET_POWER


def required_turns(rate: float = TARGET_EFFECT_RATE, power: float = TARGET_POWER,
                   alpha: float = ALPHA) -> int:
    if not 0.0 < rate < 1.0:
        raise ValueError(f"rate must be in (0,1), got {rate}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0,1), got {power}")
    return math.ceil(math.log(1.0 - power) / math.log(1.0 - rate))


def achieved_power(n_turns: int, rate: float = TARGET_EFFECT_RATE,
                   alpha: float = ALPHA) -> float:
    if n_turns <= 0:
        return 0.0
    return 1.0 - (1.0 - rate) ** n_turns
