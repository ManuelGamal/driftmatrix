from driftmatrix.config import Arm, Outcome
from driftmatrix.runner.fake import FakeEngine
from driftmatrix.runner.arms import run_restore_arm, run_recompute_arm

PRE, SUF = [1, 2, 3, 4], [5, 6]


def test_both_arms_succeed_and_agree_when_engine_is_faithful(tmp_path):
    e = FakeEngine()
    r = run_restore_arm(e, 0, PRE, SUF, 8, 5, tmp_path / "s.bin")
    c = run_recompute_arm(e, 1, PRE, SUF, 8, 5)
    assert r.outcome is Outcome.OK and c.outcome is Outcome.OK
    assert r.arm is Arm.RESTORE and c.arm is Arm.RECOMPUTE
    assert r.generation.tokens == c.generation.tokens


def test_arms_disagree_when_engine_injects_divergence(tmp_path):
    e = FakeEngine(divergence_at=2)
    r = run_restore_arm(e, 0, PRE, SUF, 8, 5, tmp_path / "s.bin")
    c = run_recompute_arm(e, 1, PRE, SUF, 8, 5)
    assert r.generation.tokens != c.generation.tokens


def test_restore_arm_records_its_state_path(tmp_path):
    p = tmp_path / "s.bin"
    r = run_restore_arm(FakeEngine(), 0, PRE, SUF, 4, 5, p)
    assert r.state_path == p and p.exists()


def test_state_io_failure_is_classified_not_raised(tmp_path):
    class BadSave(FakeEngine):
        def save_state(self, seq_id, path):
            raise OSError("disk full")
    r = run_restore_arm(BadSave(), 0, PRE, SUF, 4, 5, tmp_path / "s.bin")
    assert r.outcome is Outcome.STATE_IO_FAILED
    assert r.generation is None
    assert "disk full" in r.error


def test_unsupported_engine_is_classified(tmp_path):
    class NoState(FakeEngine):
        def save_state(self, seq_id, path):
            raise NotImplementedError("state API unavailable")
    r = run_restore_arm(NoState(), 0, PRE, SUF, 4, 5, tmp_path / "s.bin")
    assert r.outcome is Outcome.UNSUPPORTED


def test_generic_engine_crash_is_classified():
    class Boom(FakeEngine):
        def generate(self, *a, **k):
            raise RuntimeError("cuda error")
    c = run_recompute_arm(Boom(), 0, PRE, SUF, 4, 5)
    assert c.outcome is Outcome.CRASHED and "cuda error" in c.error
