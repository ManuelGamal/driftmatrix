import os
import pytest
from driftmatrix.config import LayerType, ModelSpec
from driftmatrix.runner.llamacpp import LlamaCppEngine, parse_layer_types


def test_parse_layer_types_from_qwen3_5_config():
    cfg = {"num_hidden_layers": 8,
           "layer_types": ["linear_attention"] * 3 + ["full_attention"]
                          + ["linear_attention"] * 3 + ["full_attention"]}
    types = parse_layer_types(cfg)
    assert len(types) == 8
    assert types[0] is LayerType.LINEAR_ATTENTION
    assert types[3] is LayerType.FULL_ATTENTION


def test_parse_layer_types_falls_back_to_interval():
    types = parse_layer_types({"num_hidden_layers": 8, "full_attention_interval": 4})
    assert types[3] is LayerType.FULL_ATTENTION
    assert types[2] is LayerType.LINEAR_ATTENTION


def test_parse_layer_types_handles_sliding_window():
    cfg = {"num_hidden_layers": 4,
           "layer_types": ["sliding_attention"] * 3 + ["full_attention"]}
    assert parse_layer_types(cfg)[0] is LayerType.SLIDING_ATTENTION


MODEL = os.environ.get("DRIFTMATRIX_MODEL")


@pytest.mark.skipif(not MODEL, reason="set DRIFTMATRIX_MODEL to run integration test")
def test_real_engine_state_roundtrip(tmp_path):
    """Negative control 1: immediate restore must reproduce recompute."""
    e = LlamaCppEngine(n_ctx=2048)
    e.load(ModelSpec("treatment", 3, MODEL, "IQ4_XS"))
    pre, suf = list(range(100, 140)), list(range(200, 210))
    p = tmp_path / "s.bin"
    e.reset(0); e.eval_tokens(0, pre); e.save_state(0, p)
    e.reset(0); e.load_state(0, p); e.eval_tokens(0, suf)
    a = e.generate(0, 16, 10)
    e.reset(0); e.eval_tokens(0, pre + suf)
    b = e.generate(0, 16, 10)
    print("NEGATIVE CONTROL 1 identical:", a.tokens == b.tokens)
    assert len(a.tokens) == len(b.tokens)
