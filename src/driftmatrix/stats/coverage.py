"""Coverage simulation (spec 9).

Inject an effect of known size, verify detection at the claimed rate; run under
the null, verify the false-positive rate approximates alpha.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..config import ALPHA
from .bootstrap import paired_bootstrap


@dataclass(frozen=True)
class CoverageResult:
    detection_rate: float
    false_positive_rate: float
    n_trials: int


def simulate_coverage(effect_size: float, n_per_trial: int, n_trials: int,
                      noise_sd: float, seed: int) -> CoverageResult:
    rng = np.random.default_rng(seed)
    detections = 0
    false_positives = 0
    for _ in range(n_trials):
        signal = rng.normal(effect_size, noise_sd, size=n_per_trial)
        ci = paired_bootstrap(signal, n_resamples=500, alpha=ALPHA,
                              seed=int(rng.integers(1 << 30)))
        if ci.low > 0.0:
            detections += 1
        null = rng.normal(0.0, noise_sd, size=n_per_trial)
        ci0 = paired_bootstrap(null, n_resamples=500, alpha=ALPHA,
                               seed=int(rng.integers(1 << 30)))
        if ci0.low > 0.0 or ci0.high < 0.0:
            false_positives += 1
    return CoverageResult(detections / n_trials, false_positives / n_trials, n_trials)
