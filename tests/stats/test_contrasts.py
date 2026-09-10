import numpy as np
from driftmatrix.stats.contrasts import (
    alignment_effect, alignment_by_architecture, compounding_trend,
)

rng = np.random.default_rng(0)


def test_alignment_effect_detected_when_misaligned_diverges_more():
    c = alignment_effect(rng.normal(0.0, 0.01, 200), rng.normal(0.3, 0.05, 200), seed=1)
    assert c.significant is True and c.ci.low > 0
    assert c.name == "alignment"


def test_alignment_effect_absent_when_both_are_equal():
    c = alignment_effect(rng.normal(0.0, 0.01, 200), rng.normal(0.0, 0.01, 200), seed=2)
    assert c.significant is False


def test_alignment_by_architecture_detects_hybrid_only_effect():
    c = alignment_by_architecture(rng.normal(0.3, 0.05, 200),
                                  rng.normal(0.0, 0.01, 200), seed=3)
    assert c.significant is True and c.ci.low > 0
    assert c.name == "alignment_x_architecture"


def test_alignment_by_architecture_null_when_both_levels_behave_alike():
    c = alignment_by_architecture(rng.normal(0.2, 0.05, 200),
                                  rng.normal(0.2, 0.05, 200), seed=4)
    assert c.significant is False


def test_compounding_trend_labels_and_detects():
    c = compounding_trend(rng.normal(0.1, 0.02, 200), rng.normal(0.4, 0.02, 200),
                          label="prefix_band", seed=5)
    assert c.name == "compounding_prefix_band"
    assert c.significant is True
