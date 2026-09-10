from driftmatrix.stats.coverage import simulate_coverage


def test_false_positive_rate_approximates_alpha_under_the_null():
    r = simulate_coverage(0.0, n_per_trial=200, n_trials=300, noise_sd=1.0, seed=11)
    assert 0.01 <= r.false_positive_rate <= 0.10


def test_large_effect_is_detected_almost_always():
    r = simulate_coverage(1.0, n_per_trial=200, n_trials=150, noise_sd=1.0, seed=12)
    assert r.detection_rate >= 0.95


def test_tiny_effect_is_rarely_detected():
    r = simulate_coverage(0.01, n_per_trial=50, n_trials=150, noise_sd=1.0, seed=13)
    assert r.detection_rate <= 0.30


def test_result_reports_trial_count():
    assert simulate_coverage(0.5, 100, 50, 1.0, seed=14).n_trials == 50
