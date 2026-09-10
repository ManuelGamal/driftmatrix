"""Paired bootstrap against a measured noise floor (spec 6.1)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..config import ALPHA, BOOTSTRAP_RESAMPLES, SELF_INCONSISTENCY_MULTIPLIER


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    low: float
    high: float
    n: int


def paired_bootstrap(effects, n_resamples: int = BOOTSTRAP_RESAMPLES,
                     alpha: float = ALPHA, seed: int = 0) -> BootstrapCI:
    """Percentile bootstrap over per-turn effects, paired by turn."""
    x = np.asarray(effects, dtype=np.float64)
    if x.size == 0:
        raise ValueError("effects array is empty")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(float(x.mean()), float(lo), float(hi), int(x.size))


def noise_floor(recompute_r1, recompute_r2) -> float:
    """Median divergence between the recompute arm's own two replicates."""
    a = np.asarray(recompute_r1, dtype=np.float64)
    b = np.asarray(recompute_r2, dtype=np.float64)
    return float(np.median(np.abs(a - b)))


def effect_vector(treatment, floor: float) -> np.ndarray:
    return np.asarray(treatment, dtype=np.float64) - float(floor)


def is_self_inconsistent(cond_median: float, floor_median: float) -> bool:
    return cond_median > SELF_INCONSISTENCY_MULTIPLIER * floor_median
