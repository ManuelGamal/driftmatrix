import numpy as np
from driftmatrix.runner.fake import FakeEngine
from driftmatrix.controls import PerturbEngine, find_detection_floor


def test_zero_epsilon_is_a_passthrough(tmp_path):
    e = PerturbEngine(FakeEngine(), epsilon=0.0)
    e.reset(0); e.eval_tokens(0, [1, 2, 3]); e.save_state(0, tmp_path / "s.bin")
    e.reset(0); e.load_state(0, tmp_path / "s.bin")
    a = e.generate(0, 8, 5)
    inner = FakeEngine()
    inner.reset(0); inner.eval_tokens(0, [1, 2, 3])
    assert a.tokens == inner.generate(0, 8, 5).tokens


def test_large_epsilon_changes_state_tensors(tmp_path):
    p = tmp_path / "s.bin"
    e = PerturbEngine(FakeEngine(), epsilon=0.5)
    e.reset(0); e.eval_tokens(0, [1, 2]); e.save_state(0, p)
    clean = FakeEngine().read_state_tensors(p)
    dirty = e.read_state_tensors(p)
    assert not np.allclose(clean[0], dirty[0])


def test_detection_floor_finds_the_smallest_detected_epsilon():
    def make(eps):
        return PerturbEngine(
            FakeEngine(divergence_at=0 if eps > 0.05 else None), eps)
    r = find_detection_floor(make, epsilons=[0.001, 0.01, 0.1, 1.0],
                             n_turns=20, seed=1)
    assert r.smallest_detected == 0.1
    assert r.tested == [0.001, 0.01, 0.1, 1.0]


def test_detection_floor_is_none_when_nothing_is_detected():
    r = find_detection_floor(lambda eps: PerturbEngine(FakeEngine(), eps),
                             epsilons=[0.001, 0.01], n_turns=10, seed=2)
    assert r.smallest_detected is None
