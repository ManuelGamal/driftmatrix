"""Minimal standard-library GGUF metadata reader.

Reads the header key/value section only - never tensor data - so inspecting a
multi-gigabyte model costs a few megabytes of I/O. Used to confirm a model's
architecture and layer pattern, and to decode token ids from the model's own
vocabulary without loading the model.
"""
from __future__ import annotations

import struct

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
MAX_INLINE_ARRAY = 64


def _rd(f, fmt):
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, raw)[0]


def _rstr(f):
    return f.read(_rd(f, "<Q")).decode("utf-8", "replace")


def _value(f, vtype, capture):
    if vtype == _STRING:
        return _rstr(f)
    if vtype == _ARRAY:
        etype, n = _rd(f, "<I"), _rd(f, "<Q")
        if etype == _STRING:
            if capture:
                return [_rstr(f) for _ in range(n)]
            for _ in range(n):
                f.seek(_rd(f, "<Q"), 1)
            return f"<string array x{n}>"
        if etype in _SCALAR:
            if capture or n <= MAX_INLINE_ARRAY:
                return [_rd(f, _SCALAR[etype]) for _ in range(n)]
            f.seek(struct.calcsize(_SCALAR[etype]) * n, 1)
            return f"<array x{n}>"
        return [_value(f, etype, capture) for _ in range(n)]
    if vtype in _SCALAR:
        return _rd(f, _SCALAR[vtype])
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_metadata(path, capture=()):
    """Return {"version", "n_tensors", "kv"}.

    Large arrays are summarised unless their key is listed in `capture`
    (e.g. "tokenizer.ggml.tokens"), which reads them in full.
    """
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version, n_tensors, n_kv = _rd(f, "<I"), _rd(f, "<Q"), _rd(f, "<Q")
        kv = {}
        for _ in range(n_kv):
            key = _rstr(f)
            kv[key] = _value(f, _rd(f, "<I"), key in capture)
    return {"version": version, "n_tensors": n_tensors, "kv": kv}


def _byte_decoder():
    """Inverse of the GPT-2 byte-to-unicode map used by byte-level BPE vocabs."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, extra = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _byte_decoder()


def decode_token(piece: str) -> str:
    try:
        return bytes(_BYTE_DECODER[c] for c in piece).decode("utf-8", "replace")
    except KeyError:
        return piece    # special tokens are stored literally


def decode(vocab, ids) -> str:
    return "".join(decode_token(vocab[i]) for i in ids)
