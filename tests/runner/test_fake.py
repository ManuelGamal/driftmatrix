import numpy as np
from driftmatrix.config import LayerType
from driftmatrix.runner.fake import FakeEngine


def test_fake_generates_deterministically():
    e = FakeEngine()
    e.reset(0); e.eval_tokens(0, [1, 2, 3])
    a = e.generate(0, max_tokens=8, n_probs=5)
    e.reset(0); e.eval_tokens(0, [1, 2, 3])
    b = e.generate(0, max_tokens=8, n_probs=5)
    assert a.tokens == b.tokens


def test_state_roundtrip_preserves_generation(tmp_path):
    p = tmp_path / "s.bin"
    e = FakeEngine()
    e.reset(0); e.eval_tokens(0, [1, 2, 3]); e.save_state(0, p)
    e.eval_tokens(0, [4, 5]); direct = e.generate(0, 8, 5)
    e.reset(0); e.load_state(0, p); e.eval_tokens(0, [4, 5]); restored = e.generate(0, 8, 5)
    assert direct.tokens == restored.tokens


def test_injected_divergence_appears_at_requested_position(tmp_path):
    p = tmp_path / "s.bin"
    e = FakeEngine(divergence_at=3)
    e.reset(0); e.eval_tokens(0, [1, 2, 3]); e.save_state(0, p)
    e.eval_tokens(0, [4]); direct = e.generate(0, 8, 5)
    e.reset(0); e.load_state(0, p); e.eval_tokens(0, [4]); restored = e.generate(0, 8, 5)
    assert direct.tokens[:3] == restored.tokens[:3]
    assert direct.tokens[3] != restored.tokens[3]


def test_dists_have_requested_width_and_sum_below_one():
    e = FakeEngine()
    e.reset(0); e.eval_tokens(0, [1])
    g = e.generate(0, max_tokens=4, n_probs=5)
    for d in g.dists:
        assert len(d.token_ids) == 5
        assert sum(np.exp(d.logprobs)) <= 1.0 + 1e-9


def test_state_tensors_and_layer_types_are_consistent(tmp_path):
    p = tmp_path / "s.bin"
    e = FakeEngine()
    e.reset(0); e.eval_tokens(0, [1, 2]); e.save_state(0, p)
    assert set(e.read_state_tensors(p)) == set(e.layer_types())
    assert LayerType.LINEAR_ATTENTION in e.layer_types().values()
