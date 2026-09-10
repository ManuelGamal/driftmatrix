import pytest
from scipy.stats import beta

from driftmatrix.stats.exact import clopper_pearson, sign_test, signs, test_against_floor


def test_clopper_pearson_with_no_events_bounds_the_rate():
    low, high = clopper_pearson(0, 90)
    assert low == 0.0
    assert high == pytest.approx(1 - 0.025 ** (1 / 90))       # about 4.0%


def test_clopper_pearson_matches_beta_quantiles():
    low, high = clopper_pearson(3, 90)
    assert low == pytest.approx(beta.ppf(0.025, 3, 88))
    assert high == pytest.approx(beta.ppf(0.975, 4, 87))


def test_one_event_gives_a_lower_bound_above_zero():
    assert clopper_pearson(1, 90)[0] > 0.0


def test_clopper_pearson_rejects_bad_counts():
    with pytest.raises(ValueError):
        clopper_pearson(1, 0)
    with pytest.raises(ValueError):
        clopper_pearson(5, 4)


def test_sign_test_exact_values():
    assert sign_test(9, 0, "greater") == pytest.approx(2 ** -9)
    assert sign_test(5, 0) == pytest.approx(2 * 2 ** -5)
    assert sign_test(0, 0) == 1.0


def test_signs_counts_paired_directions_and_ignores_ties():
    assert signs([1, 0, 2, 3], [0, 0, 5, 3]) == (1, 1)
    with pytest.raises(ValueError):
        signs([1, 2], [1])


def test_zero_floor_one_diverging_turn_rejects():
    r = test_against_floor([0] * 89 + [1], [0] * 90)
    assert r.method == "zero_floor" and r.p_value == 0.0 and r.n_pos == 1


def test_zero_floor_no_divergence_does_not_reject():
    r = test_against_floor([0.0] * 90, [0.0] * 90)
    assert r.method == "zero_floor" and r.p_value == 1.0


def test_nonzero_floor_falls_back_to_the_one_sided_sign_test():
    r = test_against_floor([1, 1, 1, 0], [0, 0, 0, 1])
    assert r.method == "sign_test"
    assert r.p_value == pytest.approx(sign_test(3, 1, "greater"))


def test_against_floor_needs_turns():
    with pytest.raises(ValueError):
        test_against_floor([], [])


def test_an_inexact_floor_uses_the_sign_test_even_when_its_outcome_is_all_zero():
    # recomputes differed in logprobs but took the same action on every turn
    r = test_against_floor([1, 0, 0, 0], [0, 0, 0, 0], floor_exact=False)
    assert r.method == "sign_test" and r.p_value == pytest.approx(0.5)


def test_floor_exact_cannot_contradict_the_floor():
    with pytest.raises(ValueError, match="contradicts"):
        test_against_floor([1, 0], [0, 1], floor_exact=True)
