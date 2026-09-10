import numpy as np
import pytest
from driftmatrix.stats.bootstrap import (
    paired_bootstrap, noise_floor, effect_vector, is_self_inconsistent,
)


def test_ci_brackets_the_point_estimate():
    x = np.random.default_rng(0).normal(0.5, 0.1, size=200)
    ci = paired_bootstrap(x, n_resamples=2000, seed=1)
    assert ci.low < ci.point < ci.high
    assert ci.point == pytest.approx(x.mean(), abs=1e-12)
    assert ci.n == 200


def test_zero_effect_ci_contains_zero():
    ci = paired_bootstrap(np.zeros(100), n_resamples=1000, seed=2)
    assert ci.low <= 0.0 <= ci.high


def test_large_effect_ci_excludes_zero():
    x = np.random.default_rng(3).normal(1.0, 0.05, size=200)
    assert paired_bootstrap(x, n_resamples=2000, seed=3).low > 0.0


def test_bootstrap_is_reproducible_for_a_seed():
    x = np.random.default_rng(4).normal(0.3, 0.2, size=50)
    assert paired_bootstrap(x, 500, seed=7) == paired_bootstrap(x, 500, seed=7)


def test_noise_floor_is_the_median_self_comparison():
    assert noise_floor(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])) == 0.0
    assert noise_floor(np.array([0.0, 0.0]), np.array([0.4, 0.6])) == pytest.approx(0.5)


def test_effect_vector_subtracts_the_floor():
    np.testing.assert_allclose(effect_vector(np.array([0.5, 0.7]), 0.2),
                               np.array([0.3, 0.5]))


def test_self_inconsistency_uses_the_2x_rule():
    assert is_self_inconsistent(0.21, 0.10) is True
    assert is_self_inconsistent(0.19, 0.10) is False


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap(np.array([]))
