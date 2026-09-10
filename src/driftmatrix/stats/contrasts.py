"""The three contrasts that carry the contribution (spec 6.3)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .bootstrap import BootstrapCI, paired_bootstrap


@dataclass(frozen=True)
class Contrast:
    name: str
    ci: BootstrapCI
    significant: bool


def _contrast(name, diff, seed):
    ci = paired_bootstrap(diff, seed=seed)
    return Contrast(name, ci, ci.low > 0.0 or ci.high < 0.0)


def alignment_effect(aligned, misaligned, seed: int = 0) -> Contrast:
    """Contrast 1: misaligned boundaries diverge, aligned ones do not."""
    n = min(len(aligned), len(misaligned))
    return _contrast("alignment",
                     np.asarray(misaligned[:n], float) - np.asarray(aligned[:n], float),
                     seed)


def alignment_by_architecture(effect_l3, effect_l2, seed: int = 0) -> Contrast:
    """Contrast 2: the alignment effect exists at level 3 and not at level 2.

    This is what attributes divergence to recurrent state rather than to
    interleaved layer structure or to the engine. Without it there is a
    correlation; with it there is a mechanism.
    """
    n = min(len(effect_l3), len(effect_l2))
    return _contrast("alignment_x_architecture",
                     np.asarray(effect_l3[:n], float) - np.asarray(effect_l2[:n], float),
                     seed)


def compounding_trend(short, long, label: str, seed: int = 0) -> Contrast:
    """Contrast 3: divergence grows with prefix length and continuation length."""
    n = min(len(short), len(long))
    return _contrast(f"compounding_{label}",
                     np.asarray(long[:n], float) - np.asarray(short[:n], float),
                     seed)
