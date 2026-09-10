from driftmatrix.config import (
    Alignment, PrefixBand, ContinuationBand, Condition, ALL_CONDITIONS,
    PREFIX_TOKENS, CONTINUATION_TOKENS,
)


def test_eight_conditions_all_unique():
    assert len(ALL_CONDITIONS) == 8
    assert {c.condition_id for c in ALL_CONDITIONS} == set(range(8))


def test_condition_id_roundtrips():
    for c in ALL_CONDITIONS:
        assert Condition.from_id(c.condition_id) == c


def test_reference_condition_is_zero():
    c = Condition(Alignment.ALIGNED, PrefixBand.SHORT, ContinuationBand.SHORT)
    assert c.condition_id == 0


def test_band_token_counts_match_spec():
    assert PREFIX_TOKENS[PrefixBand.SHORT] == 1024
    assert PREFIX_TOKENS[PrefixBand.LONG] == 8192
    assert CONTINUATION_TOKENS[ContinuationBand.SHORT] == 32
    assert CONTINUATION_TOKENS[ContinuationBand.LONG] == 256
