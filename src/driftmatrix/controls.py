"""Synthetic positive control (spec 8).

Perturb restored state by a known relative magnitude and find the smallest
perturbation the pipeline can see. That is the detection floor, expressed in
the units of the mechanism rather than in output-space proxies.
"""
from __future__ import annotations
import tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .metrics.behavioral import first_divergence
from .runner.arms import run_restore_arm, run_recompute_arm


class PerturbEngine:
    """Wraps an Engine; scales restored state tensors by (1 + epsilon)."""

    def __init__(self, inner, epsilon: float):
        self.inner, self.epsilon = inner, epsilon

    def load(self, spec):
        return self.inner.load(spec)

    def reset(self, seq_id):
        return self.inner.reset(seq_id)

    def eval_tokens(self, seq_id, tokens):
        return self.inner.eval_tokens(seq_id, tokens)

    def save_state(self, seq_id, path):
        return self.inner.save_state(seq_id, path)

    def load_state(self, seq_id, path):
        return self.inner.load_state(seq_id, path)

    def generate(self, seq_id, max_tokens, n_probs):
        return self.inner.generate(seq_id, max_tokens, n_probs)

    def layer_types(self):
        return self.inner.layer_types()

    def read_state_tensors(self, path):
        tensors = self.inner.read_state_tensors(path)
        if self.epsilon == 0.0:
            return tensors
        return {k: (v.astype(np.float64) * (1.0 + self.epsilon)).astype(v.dtype)
                for k, v in tensors.items()}


@dataclass(frozen=True)
class DetectionFloor:
    smallest_detected: float
    tested: list


def find_detection_floor(make_engine, epsilons, n_turns: int, seed: int) -> DetectionFloor:
    rng = np.random.default_rng(seed)
    smallest = None
    for eps in sorted(epsilons):
        engine = make_engine(eps)
        diverged = 0
        with tempfile.TemporaryDirectory() as td:
            for t in range(n_turns):
                toks = [int(x) for x in rng.integers(10, 5000, size=64)]
                pre, suf = toks[:48], toks[48:]
                r = run_restore_arm(engine, 0, pre, suf, 8, 5, Path(td) / f"s{t}.bin")
                c = run_recompute_arm(engine, 1, pre, suf, 8, 5)
                if r.generation is None or c.generation is None:
                    continue
                _, censored = first_divergence(r.generation.tokens,
                                               c.generation.tokens, cap=8)
                diverged += 0 if censored else 1
        if diverged > 0 and smallest is None:
            smallest = eps
    return DetectionFloor(smallest, sorted(epsilons))
