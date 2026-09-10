import pytest
from driftmatrix.stats.power import required_turns, achieved_power


def test_required_turns_for_spec_target_is_positive_and_finite():
    assert 0 < required_turns() < 100_000


def test_spec_target_is_reachable_within_the_planned_360_turns():
    assert required_turns() <= 360


def test_rarer_effects_need_more_turns():
    assert required_turns(rate=0.005) > required_turns(rate=0.02)


def test_higher_power_needs_more_turns():
    assert required_turns(power=0.95) > required_turns(power=0.80)


def test_achieved_power_increases_with_turns():
    assert achieved_power(1000) > achieved_power(100)


def test_achieved_power_is_a_probability():
    for n in (10, 100, 1000, 10_000):
        assert 0.0 <= achieved_power(n) <= 1.0


def test_zero_rate_is_rejected():
    with pytest.raises(ValueError, match="rate"):
        required_turns(rate=0.0)
