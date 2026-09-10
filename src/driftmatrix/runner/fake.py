"""Deterministic in-memory engine for testing without a GPU."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from ..config import LayerType
from ..metrics.distributional import TokenDist
from .engine import GenerationResult

_N_LAYERS = 8


class FakeEngine:
    """Token stream is a pure function of the evaluated context.

    `divergence_at` injects a controlled difference between a restored and a
    directly-evaluated context, so arm logic and detection can be tested.
    """

    def __init__(self, divergence_at=None):
        self.divergence_at = divergence_at
        self._ctx = {}
        self._restored = {}
        self._spec = None

    def load(self, spec):
        self._spec = spec

    def reset(self, seq_id):
        self._ctx[seq_id] = []
        self._restored[seq_id] = False

    def eval_tokens(self, seq_id, tokens):
        self._ctx.setdefault(seq_id, []).extend(tokens)

    def save_state(self, seq_id, path):
        Path(path).write_bytes(pickle.dumps(list(self._ctx.get(seq_id, []))))

    def load_state(self, seq_id, path):
        self._ctx[seq_id] = pickle.loads(Path(path).read_bytes())
        self._restored[seq_id] = True

    def _dist_for(self, seq_id, step, n_probs):
        ctx = self._ctx.get(seq_id, [])
        rng = np.random.default_rng(sum(ctx) % 9973)
        base = int(rng.integers(10, 5000))
        if (self.divergence_at is not None and step == self.divergence_at
                and self._restored.get(seq_id)):
            base += 1
        ids = [base] + [base + j for j in range(1, n_probs)]
        probs = np.array([0.6] + [0.3 / (n_probs - 1)] * (n_probs - 1))
        return base, TokenDist(token_ids=ids, logprobs=list(np.log(probs)))

    def generate(self, seq_id, max_tokens, n_probs):
        ctx = self._ctx.get(seq_id, [])
        rng = np.random.default_rng(sum(ctx) % 9973)
        toks, dists = [], []
        for i in range(max_tokens):
            base = int(rng.integers(10, 5000))
            if (self.divergence_at is not None and i == self.divergence_at
                    and self._restored.get(seq_id)):
                base += 1
            toks.append(base)
            ids = [base] + [base + j for j in range(1, n_probs)]
            probs = np.array([0.6] + [0.3 / (n_probs - 1)] * (n_probs - 1))
            dists.append(TokenDist(token_ids=ids, logprobs=list(np.log(probs))))
        return GenerationResult(tokens=toks, dists=dists,
                                text=" ".join(map(str, toks)))

    def score_tokens(self, seq_id, tokens, n_probs):
        """Teacher-forced scoring: distributions for a fixed token sequence."""
        dists = []
        for step, tok in enumerate(tokens):
            _, dist = self._dist_for(seq_id, step, n_probs)
            dists.append(dist)
            self.eval_tokens(seq_id, [tok])
        return dists

    def read_state_tensors(self, path):
        ctx = pickle.loads(Path(path).read_bytes())
        rng = np.random.default_rng(sum(ctx) % 9973)
        return {i: rng.normal(size=(4, 4)).astype(np.float32) for i in range(_N_LAYERS)}

    def layer_types(self):
        return {i: (LayerType.FULL_ATTENTION if (i + 1) % 4 == 0
                    else LayerType.LINEAR_ATTENTION) for i in range(_N_LAYERS)}
