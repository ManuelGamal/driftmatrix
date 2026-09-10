from driftmatrix.sweep import SweepPoint, detect_period, sweep_offsets
from driftmatrix.runner.fake import FakeEngine


def test_detects_period_when_divergence_is_periodic():
    pts = [SweepPoint(o, o % 64 != 0, 0 if o % 64 else 32) for o in range(0, 512, 8)]
    assert detect_period(pts) == 64


def test_returns_none_when_nothing_diverges():
    assert detect_period([SweepPoint(o, False, 32) for o in range(0, 512, 8)]) is None


def test_returns_none_when_everything_diverges():
    assert detect_period([SweepPoint(o, True, 0) for o in range(0, 512, 8)]) is None


def test_sweep_runs_both_arms_at_every_offset(tmp_path):
    pts = sweep_offsets(FakeEngine(), list(range(300)), offsets=[64, 128, 192],
                        max_new=4, n_probs=5, state_dir=tmp_path)
    assert [p.offset for p in pts] == [64, 128, 192]
    assert all(p.diverged is False for p in pts)
