import pytest
from driftmatrix.config import Alignment, PrefixBand
from driftmatrix.traces.boundary import pad_to_token_count, boundary_offset


class FakeTok:
    """One token per whitespace-separated word."""

    def encode(self, text):
        return [hash(w) % 1000 for w in text.split()]


def test_padding_hits_exact_token_count():
    tok = FakeTok()
    out = pad_to_token_count("hello world", target=50, tok=tok)
    assert len(tok.encode(out)) == 50


def test_padding_is_noop_when_already_long_enough():
    tok = FakeTok()
    text = " ".join(["w"] * 60)
    assert pad_to_token_count(text, target=50, tok=tok) == text


def test_aligned_offset_is_a_multiple_of_chunk_size():
    off = boundary_offset(PrefixBand.SHORT, Alignment.ALIGNED, chunk_size=64)
    assert off % 64 == 0
    assert off <= 1024


def test_misaligned_offset_is_half_a_chunk_off():
    aligned = boundary_offset(PrefixBand.SHORT, Alignment.ALIGNED, chunk_size=64)
    mis = boundary_offset(PrefixBand.SHORT, Alignment.MISALIGNED, chunk_size=64)
    assert mis % 64 == 32
    assert abs(mis - aligned) == 32


def test_long_band_offset_is_larger_than_short():
    a = boundary_offset(PrefixBand.SHORT, Alignment.ALIGNED, chunk_size=64)
    b = boundary_offset(PrefixBand.LONG, Alignment.ALIGNED, chunk_size=64)
    assert b > a


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError, match="chunk_size"):
        boundary_offset(PrefixBand.SHORT, Alignment.ALIGNED, chunk_size=0)
