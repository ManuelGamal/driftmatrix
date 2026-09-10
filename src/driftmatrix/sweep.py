"""Boundary-offset sweep to locate the kernel chunk size (spec 1.2, gate G1)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .metrics.behavioral import first_divergence
from .runner.arms import run_restore_arm, run_recompute_arm

CANDIDATE_PERIODS = (8, 16, 32, 64, 128, 256, 512)


@dataclass(frozen=True)
class SweepPoint:
    offset: int
    diverged: bool
    first_div_position: int


def sweep_offsets(engine, tokens, offsets, max_new, n_probs, state_dir):
    out = []
    for off in offsets:
        prefix, suffix = tokens[:off], tokens[off:]
        r = run_restore_arm(engine, 0, prefix, suffix, max_new, n_probs,
                            Path(state_dir) / f"s_{off}.bin")
        c = run_recompute_arm(engine, 1, prefix, suffix, max_new, n_probs)
        if r.generation is None or c.generation is None:
            out.append(SweepPoint(off, False, max_new))
            continue
        pos, censored = first_divergence(r.generation.tokens, c.generation.tokens,
                                         cap=max_new)
        out.append(SweepPoint(off, not censored, pos))
    return out


def detect_period(points):
    """Smallest candidate period p where offsets divisible by p never diverge
    and offsets not divisible by p always do."""
    if not points or all(p.diverged for p in points) or not any(p.diverged for p in points):
        return None
    for period in CANDIDATE_PERIODS:
        on = [p for p in points if p.offset % period == 0]
        off = [p for p in points if p.offset % period != 0]
        if on and off and not any(p.diverged for p in on) and all(p.diverged for p in off):
            return period
    return None
