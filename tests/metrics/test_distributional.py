import numpy as np
import pytest
from driftmatrix.metrics.distributional import (
    TokenDist, align_supports, jsd, total_variation, top1_agree, per_position_jsd,
)


def d(ids, probs):
    return TokenDist(token_ids=list(ids), logprobs=[float(np.log(p)) for p in probs])


def test_identical_distributions_give_zero():
    a = d([1, 2, 3], [0.5, 0.3, 0.2])
    p, q = align_supports(a, a)
    assert jsd(p, q) == pytest.approx(0.0, abs=1e-12)
    assert total_variation(p, q) == pytest.approx(0.0, abs=1e-12)


def test_disjoint_supports_give_log2():
    p, q = align_supports(d([1], [1.0]), d([2], [1.0]))
    assert jsd(p, q) == pytest.approx(np.log(2), rel=1e-9)


def test_union_support_includes_other_bucket():
    p, q = align_supports(d([1, 2], [0.6, 0.3]), d([2, 3], [0.5, 0.4]))
    assert len(p) == len(q) == 4
    assert p.sum() == pytest.approx(1.0)
    assert q.sum() == pytest.approx(1.0)
    assert p[-1] == pytest.approx(0.1)
    assert q[-1] == pytest.approx(0.1)


def test_jsd_is_symmetric_and_bounded():
    p, q = align_supports(d([1, 2], [0.9, 0.1]), d([1, 2], [0.1, 0.9]))
    assert jsd(p, q) == pytest.approx(jsd(q, p))
    assert 0.0 <= jsd(p, q) <= np.log(2) + 1e-12


def test_top1_agreement():
    assert top1_agree(d([5, 6], [0.7, 0.3]), d([5, 9], [0.6, 0.4])) is True
    assert top1_agree(d([5, 6], [0.7, 0.3]), d([6, 5], [0.8, 0.2])) is False


def test_per_position_jsd_length_matches_shorter_sequence():
    a = [d([1], [1.0])] * 5
    b = [d([1], [1.0])] * 3
    assert per_position_jsd(a, b).shape == (3,)
