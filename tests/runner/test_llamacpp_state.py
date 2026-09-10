"""State persistence for the llama.cpp adapter, without loading a model.

Llama.load_state() requires a LlamaState object. This pins that the adapter
round-trips the whole object rather than only its raw byte blob.
"""
import numpy as np
from driftmatrix.runner.llamacpp import LlamaCppEngine


class StateLike:
    """Mirrors LlamaState's fields."""
    def __init__(self, n_tokens, blob):
        self.input_ids = np.arange(n_tokens, dtype=np.intc)
        self.scores = np.zeros((n_tokens, 4), dtype=np.single)
        self.n_tokens = n_tokens
        self.llama_state = blob
        self.llama_state_size = len(blob)
        self.seed = 1234


class RecordingLlm:
    def __init__(self):
        self.loaded = None

    def save_state(self):
        return StateLike(7, b"\x01\x02\x03\x04")

    def load_state(self, state):
        if isinstance(state, (bytes, bytearray)):
            raise TypeError("load_state requires a LlamaState, got bytes")
        self.loaded = state


def _engine():
    e = LlamaCppEngine()
    e._llm = RecordingLlm()
    return e


def test_load_state_receives_a_state_object_not_bytes(tmp_path):
    e = _engine()
    p = tmp_path / "s.bin"
    e.save_state(0, p)
    e.load_state(0, p)
    assert e._llm.loaded is not None
    assert e._llm.loaded.n_tokens == 7
    np.testing.assert_array_equal(e._llm.loaded.input_ids, np.arange(7))


def test_read_state_tensors_exposes_the_raw_blob(tmp_path):
    e = _engine()
    p = tmp_path / "s.bin"
    e.save_state(0, p)
    t = e.read_state_tensors(p)
    np.testing.assert_array_equal(t[-1], np.array([1, 2, 3, 4], dtype=np.uint8))
