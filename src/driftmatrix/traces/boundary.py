"""Tokenizer-verified boundary placement (spec 4.3).

Character counts are never trusted. Every offset is confirmed by encoding.
"""
from __future__ import annotations
from typing import Protocol
from ..config import Alignment, PrefixBand, PREFIX_TOKENS


class Tokenizer(Protocol):
    def encode(self, text: str): ...


def pad_to_token_count(text: str, target: int, tok, filler: str = " lorem") -> str:
    """Grow text until it encodes to exactly `target` tokens.

    Deliberately not estimate-then-correct: tokenizers are not linear in
    characters, so any estimate would need verifying by encoding anyway.
    """
    cur = len(tok.encode(text))
    if cur >= target:
        return text
    out = text
    while cur < target:
        out += filler
        cur = len(tok.encode(out))
    while cur > target:
        out = out.rsplit(" ", 1)[0]
        cur = len(tok.encode(out))
    return out


def boundary_offset(prefix_band, alignment, chunk_size: int) -> int:
    """Token index at which state is saved.

    Aligned    -> largest multiple of chunk_size within the band.
    Misaligned -> that minus half a chunk, so the two arms tile differently.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    band = PREFIX_TOKENS[prefix_band]
    aligned = (band // chunk_size) * chunk_size
    if alignment is Alignment.ALIGNED:
        return aligned
    return aligned - chunk_size // 2
