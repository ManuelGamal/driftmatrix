"""G1 logic against a fake server in which micro-batch tiling is the mechanism.

The fake tiles prompt evaluation exactly like llama.cpp: a recompute tiles
[0, n) from 0 in steps of n_ubatch; a restore at k tiled [0, k) when the prefix
was first evaluated and tiles [k, n) afresh from k. Output is perturbed exactly
when those tilings differ - so the prediction logic is tested against an engine
where tiling is the only cause, and against engines where it is not.
"""
from driftmatrix.g1 import (
    SPLITS, completions_exact, cross_ubatch_summary, g1_verdict, run_g1,
)
from driftmatrix.runner.llamaserver import Completion, ServerError

PROMPT = " ".join(f"w{i}" for i in range(2000))


def tiling(start, end, ub):
    return tuple(range(start, end, ub))


class TilingServer:
    def __init__(self, ubatch, mode="tiling", flip=True):
        self.ub, self.mode, self.flip = ubatch, mode, flip
        self.cache, self.starts, self.saved = [], (), {}

    def props(self):
        return {"build_info": "fake"}

    def tokenize(self, text, add_special=True):
        return list(range(1, len(text.split()) + 1))

    def slot_erase(self, slot=0):
        self.cache, self.starts = [], ()
        return {}

    def slot_save(self, filename, slot=0):
        self.saved[filename] = (list(self.cache), self.starts)
        return {"n_saved": len(self.cache)}

    def slot_restore(self, filename, slot=0):
        self.cache, self.starts = list(self.saved[filename][0]), self.saved[filename][1]
        self.restored = True
        return {"n_restored": len(self.cache)}

    def complete(self, toks, n_predict, n_probs=10, cache_prompt=True, slot=0, seed=1234):
        n = len(toks)
        reuse = bool(cache_prompt and self.cache and toks[:len(self.cache)] == self.cache)
        if reuse:
            starts = self.starts + tiling(len(self.cache), n, self.ub)
            prompt_n = n - len(self.cache)
        else:
            starts = tiling(0, n, self.ub)
            prompt_n = n
        canonical = tiling(0, n, self.ub)
        if self.mode == "tiling":
            perturbed = starts != canonical
        elif self.mode == "always":
            perturbed = reuse
        else:
            perturbed = False
        base = sum(toks) % 997
        out = [base + i for i in range(n_predict)]
        top = [[(t, -0.1), (t + 1, -2.0)] for t in out]
        if perturbed and n_predict > 1:
            top = [[(t, lp + 1e-6) for t, lp in pos] for pos in top]  # tiny, still inexact
            if self.flip:
                out[1] += 1
                top[1] = [(out[1], -0.1), (out[1] + 1, -2.0)]
        self.cache, self.starts = list(toks), starts
        return Completion(out, top, prompt_n, n - prompt_n, {})


def test_prediction_holds_when_tiling_is_the_mechanism():
    r = run_g1(TilingServer(512), PROMPT, ubatch=512)
    assert r["verdict"] == "PREDICTION_HOLDS"
    for c in r["cases"]:
        assert c["exact"] is c["aligned"]


def test_inexact_without_a_token_flip_is_still_detected():
    r = run_g1(TilingServer(512, flip=False), PROMPT, ubatch=512)
    assert r["verdict"] == "PREDICTION_HOLDS"
    mis = [c for c in r["cases"] if not c["aligned"]]
    assert all(c["first_divergence"] is None and not c["exact"] for c in mis)


def test_exact_set_moves_with_ubatch():
    reports = [run_g1(TilingServer(ub), PROMPT, ubatch=ub) for ub in (256, 512, 1024)]
    s = cross_ubatch_summary(reports)
    assert s["exact_set_tracks_ubatch"] is True
    assert s["grid"][512] == {256: True, 512: True, 1024: False}
    assert s["grid"][768] == {256: True, 512: False, 1024: False}
    assert s["grid"][1024] == {256: True, 512: True, 1024: True}
    assert 512 in s["discriminating_splits"] and 768 in s["discriminating_splits"]


def test_engine_where_every_restore_diverges_falsifies_aligned_exactness():
    r = run_g1(TilingServer(512, mode="always"), PROMPT, ubatch=512)
    assert r["verdict"] == "ALIGNED_NOT_ALWAYS_EXACT"
    assert 512 in r["aligned_not_exact"]


def test_engine_where_tiling_never_matters():
    r = run_g1(TilingServer(512, mode="never"), PROMPT, ubatch=512)
    assert r["verdict"] == "MISALIGNED_SOMETIMES_EXACT"


def test_mismatched_ubatch_argument_is_exposed():
    # server runs -ub 512 but the probe is told 1024: the reading must fail
    r = run_g1(TilingServer(512), PROMPT, ubatch=1024)
    assert r["verdict"] != "PREDICTION_HOLDS"


def test_summary_does_not_claim_tracking_from_one_ubatch():
    s = cross_ubatch_summary([run_g1(TilingServer(512), PROMPT, ubatch=512)])
    assert s["exact_set_tracks_ubatch"] is False


def test_short_prompt_skips_splits_without_room_for_a_suffix():
    short = " ".join(f"w{i}" for i in range(900))
    r = run_g1(TilingServer(512), short, ubatch=512)
    assert all(k > 900 - 64 for k in r["splits_skipped"])
    assert all(c["k"] <= 900 - 64 for c in r["cases"])


def test_all_aligned_splits_cannot_discriminate():
    r = run_g1(TilingServer(256), PROMPT, ubatch=256, splits=(256, 512, 768))
    assert r["verdict"] == "SPLITS_DO_NOT_DISCRIMINATE"


def test_completions_exact_is_bitwise_on_logprobs():
    a = Completion([1, 2], [[(1, -0.1)], [(2, -0.2)]], 10, 0, {})
    b = Completion([1, 2], [[(1, -0.1)], [(2, -0.2000001)]], 10, 0, {})
    assert completions_exact(a, a) is True
    assert completions_exact(a, b) is False


def test_state_api_error_verdict():
    class Refusing(TilingServer):
        def slot_save(self, filename, slot=0):
            raise ServerError(501, "disabled", "/slots/0?action=save")
    r = run_g1(Refusing(512), PROMPT, ubatch=512)
    assert r["verdict"] == "STATE_API_ERROR"


def test_default_splits_include_the_discriminating_boundaries():
    assert {512, 768, 1024}.issubset(SPLITS)
    assert any(k % 256 for k in SPLITS)
