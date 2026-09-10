import numpy as np
import pytest
from driftmatrix.config import LayerType
from driftmatrix.statediff import diff_layer, diff_all


def test_identical_tensors_give_all_zeros():
    x = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    dd = diff_layer(0, LayerType.LINEAR_ATTENTION, x, x.copy())
    assert dd.max_abs_diff == 0.0
    assert dd.max_rel_diff == 0.0
    assert dd.frobenius_norm_diff == 0.0


def test_known_epsilon_is_recovered():
    x = np.ones((2, 2), dtype=np.float32)
    y = x.copy()
    y[1, 1] += 0.25
    dd = diff_layer(3, LayerType.LINEAR_ATTENTION, y, x)
    assert dd.max_abs_diff == pytest.approx(0.25)
    assert dd.max_rel_diff == pytest.approx(0.25)
    assert dd.frobenius_norm_diff == pytest.approx(0.25)


def test_relative_diff_ignores_near_zero_denominators():
    x = np.array([[0.0, 2.0]], dtype=np.float32)
    y = np.array([[1e-9, 2.5]], dtype=np.float32)
    dd = diff_layer(1, LayerType.FULL_ATTENTION, y, x)
    assert np.isfinite(dd.max_rel_diff)
    assert dd.max_rel_diff == pytest.approx(0.25)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        diff_layer(0, LayerType.LINEAR_ATTENTION, np.zeros((2, 2)), np.zeros((3, 2)))


def test_diff_all_preserves_layer_order_and_types():
    r = {0: np.zeros((2, 2)), 1: np.ones((2, 2))}
    c = {0: np.zeros((2, 2)), 1: np.ones((2, 2))}
    types = {0: LayerType.LINEAR_ATTENTION, 1: LayerType.FULL_ATTENTION}
    out = diff_all(r, c, types)
    assert [x.layer_idx for x in out] == [0, 1]
    assert out[0].layer_type is LayerType.LINEAR_ATTENTION
    assert out[1].layer_type is LayerType.FULL_ATTENTION
