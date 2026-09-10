"""llama.cpp adapter. Satisfies the same Engine protocol as FakeEngine."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ..config import LayerType
from ..metrics.distributional import TokenDist
from .engine import GenerationResult


def parse_layer_types(config_json: dict):
    """Map layer index -> LayerType from a HuggingFace config.json."""
    n = int(config_json["num_hidden_layers"])
    explicit = config_json.get("layer_types")
    if explicit:
        return {i: LayerType(explicit[i]) for i in range(n)}
    interval = int(config_json.get("full_attention_interval", 0))
    if interval <= 0:
        return {i: LayerType.FULL_ATTENTION for i in range(n)}
    return {i: (LayerType.FULL_ATTENTION if (i + 1) % interval == 0
                else LayerType.LINEAR_ATTENTION) for i in range(n)}


class LlamaCppEngine:
    def __init__(self, n_ctx: int = 16384, seed: int = 1234):
        self.n_ctx, self.seed = n_ctx, seed
        self._llm = None
        self._layer_types = {}
        self._spec = None

    def load(self, spec):
        from llama_cpp import Llama
        self._spec = spec
        self._llm = Llama(model_path=spec.gguf_path, n_ctx=self.n_ctx,
                          seed=self.seed, logits_all=False, verbose=False)
        cfg = Path(spec.gguf_path).with_name("config.json")
        if cfg.exists():
            self._layer_types = parse_layer_types(
                json.loads(cfg.read_text(encoding="utf-8")))

    def reset(self, seq_id):
        self._llm.reset()

    def eval_tokens(self, seq_id, tokens):
        self._llm.eval(list(tokens))

    def save_state(self, seq_id, path):
        state = self._llm.save_state()
        Path(path).write_bytes(bytes(state.llama_state[: state.llama_state_size]))

    def load_state(self, seq_id, path):
        self._llm.load_state(Path(path).read_bytes())

    def generate(self, seq_id, max_tokens, n_probs):
        toks, dists = [], []
        for _ in range(max_tokens):
            logits = np.asarray(self._llm.scores[self._llm.n_tokens - 1],
                                dtype=np.float64)
            top = np.argpartition(-logits, n_probs)[:n_probs]
            top = top[np.argsort(-logits[top])]
            shifted = logits - logits.max()
            logprobs = shifted - np.log(np.exp(shifted).sum())
            dists.append(TokenDist(token_ids=[int(t) for t in top],
                                   logprobs=[float(logprobs[t]) for t in top]))
            nxt = int(top[0])
            if nxt == self._llm.token_eos():
                break
            toks.append(nxt)
            self._llm.eval([nxt])
        return GenerationResult(
            tokens=toks, dists=dists,
            text=self._llm.detokenize(toks).decode("utf-8", "replace"))

    def read_state_tensors(self, path):
        """Best-effort per-layer view of a saved state blob.

        llama.cpp serialises state as one opaque blob. Where per-layer views
        are not exposed by the build, this returns a single pseudo-layer keyed
        -1 holding the raw bytes, which still supports byte-level equality.
        """
        return {-1: np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)}

    def layer_types(self):
        return dict(self._layer_types)
