import pytest
from driftmatrix.stats.multiplicity import Family, FAMILY_SIZES, holm, holm_by_family


def test_family_sizes_match_spec():
    assert FAMILY_SIZES[Family.BEHAVIORAL] == 24
    assert FAMILY_SIZES[Family.DISTRIBUTIONAL] == 16
    assert FAMILY_SIZES[Family.MECHANISM] == 6


def test_holm_matches_hand_computed_example():
    # .001 <= .05/3, .02 <= .05/2, .04 <= .05/1
    assert holm({"a": 0.001, "b": 0.02, "c": 0.04}, alpha=0.05) == \
           {"a": True, "b": True, "c": True}


def test_holm_stops_at_first_failure():
    # .04 > .05/3 -> reject nothing, and no later test may pass
    assert holm({"a": 0.04, "b": 0.045, "c": 0.049}, alpha=0.05) == \
           {"a": False, "b": False, "c": False}


def test_holm_is_less_conservative_than_bonferroni():
    # m=2: Holm compares .04 against alpha/1 = .05 and rejects.
    # Bonferroni would compare it against alpha/2 = .025 and would not.
    out = holm({"a": 0.001, "b": 0.04}, alpha=0.05)
    assert out == {"a": True, "b": True}
    assert 0.04 > 0.05 / 2


def test_families_are_corrected_independently():
    out = holm_by_family({Family.BEHAVIORAL: {"x": 0.02},
                          Family.MECHANISM: {"y": 0.02}}, alpha=0.05)
    assert out[Family.BEHAVIORAL]["x"] is True
    assert out[Family.MECHANISM]["y"] is True


def test_empty_family_is_allowed():
    assert holm_by_family({Family.MECHANISM: {}}) == {Family.MECHANISM: {}}


def test_pvalue_out_of_range_raises():
    with pytest.raises(ValueError, match="p-value"):
        holm({"a": 1.5})
