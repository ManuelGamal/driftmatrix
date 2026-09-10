"""Distributional divergence over truncated top-k logprobs (spec 5.2)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class TokenDist:
    token_ids: list
    logprobs: list

    def as_probs(self) -> dict:
        return {int(t): float(np.exp(lp)) for t, lp in zip(self.token_ids, self.logprobs)}


def align_supports(a: TokenDist, b: TokenDist):
    """Re-express both distributions over union(support) + one <other> bucket.

    The engine returns only the top-k tokens, so each distribution is missing
    mass. That missing mass is real and must be represented, not renormalised
    away, or JSD is computed against a distribution neither arm produced.
    """
    pa, pb = a.as_probs(), b.as_probs()
    union = sorted(set(pa) | set(pb))
    va = np.array([pa.get(t, 0.0) for t in union], dtype=np.float64)
    vb = np.array([pb.get(t, 0.0) for t in union], dtype=np.float64)
    ra = max(0.0, 1.0 - float(va.sum()))
    rb = max(0.0, 1.0 - float(vb.sum()))
    p = np.concatenate([va, [ra]])
    q = np.concatenate([vb, [rb]])
    return p / p.sum(), q / q.sum()


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats. Symmetric, bounded by log 2."""
    m = 0.5 * (p + q)

    def _kl(x, y):
        mask = x > _EPS
        return float(np.sum(x[mask] * np.log(x[mask] / np.maximum(y[mask], _EPS))))

    return max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def top1_agree(a: TokenDist, b: TokenDist) -> bool:
    if not a.token_ids or not b.token_ids:
        return False
    return (a.token_ids[int(np.argmax(a.logprobs))]
            == b.token_ids[int(np.argmax(b.logprobs))])


def per_position_jsd(a, b) -> np.ndarray:
    n = min(len(a), len(b))
    return np.array([jsd(*align_supports(a[i], b[i])) for i in range(n)])
