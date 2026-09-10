from driftmatrix.gate import write_baseline, check_against_baseline

BASE = {"divergence_rate": 0.02, "noise_floor": 0.001, "tool_disagreement": 0.005}


def test_identical_summary_passes(tmp_path):
    p = tmp_path / "b.json"
    write_baseline(BASE, p)
    r = check_against_baseline(dict(BASE), p)
    assert r.passed is True and r.violations == []


def test_within_tolerance_passes(tmp_path):
    p = tmp_path / "b.json"
    write_baseline(BASE, p)
    assert check_against_baseline({**BASE, "divergence_rate": 0.021}, p,
                                  tolerance=0.10).passed is True


def test_regression_beyond_tolerance_fails_with_named_violation(tmp_path):
    p = tmp_path / "b.json"
    write_baseline(BASE, p)
    r = check_against_baseline({**BASE, "divergence_rate": 0.20}, p, tolerance=0.05)
    assert r.passed is False
    assert any("divergence_rate" in v for v in r.violations)


def test_missing_metric_is_a_violation(tmp_path):
    p = tmp_path / "b.json"
    write_baseline(BASE, p)
    r = check_against_baseline({"noise_floor": 0.001}, p)
    assert r.passed is False
    assert any("missing" in v for v in r.violations)


def test_noise_floor_above_spec_threshold_is_always_a_violation(tmp_path):
    p = tmp_path / "b.json"
    write_baseline(BASE, p)
    r = check_against_baseline({**BASE, "noise_floor": 0.5}, p)
    assert r.passed is False
    assert any("noise_floor" in v for v in r.violations)
