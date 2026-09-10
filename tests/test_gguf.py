import struct

import pytest

from driftmatrix.gguf import decode, decode_token, read_metadata


def _s(text):
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _write_gguf(path, kvs):
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    for key, vtype, payload in kvs:
        body += _s(key) + struct.pack("<I", vtype) + payload
    path.write_bytes(body)


def _sample(tmp_path):
    p = tmp_path / "m.gguf"
    vocab = struct.pack("<IQ", 8, 3) + _s("Ġhello") + _s("Ċ") + _s("x")
    pattern = struct.pack("<IQ", 7, 4) + bytes([1, 1, 1, 0])
    _write_gguf(p, [
        ("general.architecture", 8, _s("qwen35")),
        ("qwen35.block_count", 4, struct.pack("<I", 33)),
        ("spark2_5.attention.sliding_window_pattern", 9, pattern),
        ("tokenizer.ggml.tokens", 9, vocab),
    ])
    return p


def test_reads_scalars_strings_and_small_arrays(tmp_path):
    meta = read_metadata(_sample(tmp_path))
    kv = meta["kv"]
    assert meta["version"] == 3 and meta["n_tensors"] == 0
    assert kv["general.architecture"] == "qwen35"
    assert kv["qwen35.block_count"] == 33
    assert kv["spark2_5.attention.sliding_window_pattern"] == [True, True, True, False]


def test_string_arrays_are_summarised_unless_captured(tmp_path):
    p = _sample(tmp_path)
    assert read_metadata(p)["kv"]["tokenizer.ggml.tokens"] == "<string array x3>"
    vocab = read_metadata(p, capture=("tokenizer.ggml.tokens",))["kv"]["tokenizer.ggml.tokens"]
    assert len(vocab) == 3


def test_byte_level_bpe_pieces_decode_to_text(tmp_path):
    vocab = read_metadata(_sample(tmp_path),
                          capture=("tokenizer.ggml.tokens",))["kv"]["tokenizer.ggml.tokens"]
    assert decode(vocab, [0, 1, 2]) == " hello\nx"


def test_special_tokens_pass_through_literally():
    assert decode_token("<think>") == "<think>"


def test_rejects_non_gguf(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"NOPE" + b"\x00" * 20)
    with pytest.raises(ValueError, match="not a GGUF"):
        read_metadata(p)


def test_truncated_header_raises(tmp_path):
    p = tmp_path / "t.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<I", 3))
    with pytest.raises(ValueError, match="truncated"):
        read_metadata(p)
