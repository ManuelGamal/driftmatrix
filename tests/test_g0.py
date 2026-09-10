"""G0 probe logic against a scripted fake llama-server.

The fake models the behaviours that matter most: whether a restored slot is
actually reused or silently reprocessed, and whether a host-side prompt cache
lets a "recompute" arm reuse state it never computed.
"""
from driftmatrix.g0 import recompute_was_clean, run_g0, split_points, verdict
from driftmatrix.runner.llamaserver import (
    Completion, ServerError, first_divergence, max_top1_logprob_gap,
    parse_completion, restore_was_used,
)

PROMPT = " ".join(f"w{i}" for i in range(600))


class FakeServer:
    def __init__(self, reprocess_on_restore=False, diverge_after_restore=False,
                 nondeterministic=False, refuse_slots=False, host_cache=False):
        self.reprocess = reprocess_on_restore
        self.diverge = diverge_after_restore
        self.nondet = nondeterministic
        self.refuse = refuse_slots
        self.host_cache = host_cache
        self.host = set()
        self.cache, self.saved, self.restored, self.calls = [], {}, False, 0

    def props(self):
        return {"total_slots": 1}

    def tokenize(self, text, add_special=True):
        return list(range(1, len(text.split()) + 1))

    def slot_erase(self, slot=0):
        self.cache, self.restored = [], False
        return {"n_erased": 1}

    def slot_save(self, filename, slot=0):
        if self.refuse:
            raise ServerError(501, "slot save not enabled", "/slots/0?action=save")
        self.saved[filename] = list(self.cache)
        return {"n_saved": len(self.cache)}

    def slot_restore(self, filename, slot=0):
        self.cache, self.restored = list(self.saved[filename]), True
        return {"n_restored": len(self.cache)}

    def complete(self, toks, n_predict, n_probs=5, cache_prompt=True, slot=0, seed=1234):
        self.calls += 1
        reuse = (cache_prompt and self.cache and toks[:len(self.cache)] == self.cache
                 and not self.reprocess)
        prompt_n = len(toks) - len(self.cache) if reuse else len(toks)
        if self.host_cache and tuple(toks) in self.host:
            prompt_n = 1                       # served from host RAM, ignores erase
        self.host.add(tuple(toks))
        out = [sum(toks) % 997 + i for i in range(n_predict)]
        if self.diverge and self.restored and reuse and n_predict > 3:
            out[3] += 1
        if self.nondet and n_predict > 5:
            out[5] += self.calls
        self.cache = list(toks)
        return Completion(out, [[(t, -0.1)] for t in out], prompt_n,
                          len(toks) - prompt_n, {})


def test_faithful_server_passes():
    assert run_g0(FakeServer(), PROMPT)["verdict"] == "FAITHFUL_AT_TESTED_SPLITS"


def test_silent_reprocessing_is_caught_not_mistaken_for_a_pass():
    r = run_g0(FakeServer(reprocess_on_restore=True), PROMPT)
    assert r["verdict"] == "RESTORE_NOT_USED"
    assert all(c["identical"] for c in r["cases"].values())   # identical, yet invalid


def test_host_cache_leaking_into_recompute_is_caught():
    r = run_g0(FakeServer(host_cache=True), PROMPT)
    assert r["verdict"] == "RECOMPUTE_NOT_CLEAN"
    assert r["floor"]["recompute_clean"][1] is False


def test_divergence_after_real_restore_is_reported():
    r = run_g0(FakeServer(diverge_after_restore=True), PROMPT)
    assert r["verdict"] == "DIVERGENCE_ON_DAY_ONE"
    assert r["cases"]["aligned_64"]["first_divergence"] == 3


def test_nondeterministic_floor_takes_precedence_over_divergence():
    r = run_g0(FakeServer(nondeterministic=True, diverge_after_restore=True), PROMPT)
    assert r["verdict"] == "NONDETERMINISTIC_FLOOR"


def test_refused_slot_api_is_reported():
    assert run_g0(FakeServer(refuse_slots=True), PROMPT)["verdict"] == "STATE_API_UNSUPPORTED"


def test_restore_arm_proves_only_the_suffix_was_evaluated():
    case = run_g0(FakeServer(), PROMPT)["cases"]["offset_32"]
    assert case["restore_used"] is True
    assert case["recompute_clean"] is True
    assert case["restore_prompt_n"] == case["n_suffix"]
    assert case["n_saved"] == case["split_at"]


def test_split_points_are_aligned_and_offset_by_half_a_chunk():
    s = split_points(600)
    assert s["aligned_64"] % 64 == 0
    assert s["offset_32"] - s["aligned_64"] == 32


def test_save_mismatch_verdict():
    rep = {"floor": {"identical": True, "recompute_clean": [True, True]},
           "cases": {"a": {"split_at": 256, "n_saved": 257, "restore_used": True,
                           "recompute_clean": True, "identical": True}}}
    assert verdict(rep) == "SAVE_MISMATCH"


def test_restore_and_recompute_thresholds():
    assert restore_was_used(301, 300, 600) is True
    assert restore_was_used(600, 300, 600) is False
    assert restore_was_used(-1, 300, 600) is None
    assert recompute_was_clean(600, 600) is True
    assert recompute_was_clean(599, 600) is True
    assert recompute_was_clean(12, 600) is False
    assert recompute_was_clean(-1, 600) is None


def test_parse_current_completion_format():
    resp = {"tokens": [7, 8],
            "completion_probabilities": [
                {"id": 7, "logprob": -0.1, "top_logprobs": [{"id": 7, "logprob": -0.1},
                                                            {"id": 9, "logprob": -2.0}]},
                {"id": 8, "logprob": -0.2, "top_logprobs": [{"id": 8, "logprob": -0.2}]}],
            "timings": {"prompt_n": 42}, "tokens_cached": 100}
    c = parse_completion(resp)
    assert c.tokens == [7, 8]
    assert c.top[0][0] == (7, -0.1)
    assert c.prompt_n == 42 and c.tokens_cached == 100


def test_parse_legacy_completion_format_and_token_fallback():
    resp = {"completion_probabilities": [{"content": "a", "probs": [{"tok_str": "a", "prob": 1.0}]}],
            "tokens_evaluated": 12}
    c = parse_completion(resp)
    assert c.top[0][0][0] == "a" and c.prompt_n == 12


def test_logprob_gap_stops_at_first_token_flip():
    a = [[(1, -0.1)], [(2, -0.5)], [(3, -0.1)]]
    b = [[(1, -0.3)], [(9, -0.5)], [(3, -5.0)]]
    assert abs(max_top1_logprob_gap(a, b) - 0.2) < 1e-12


def test_first_divergence_helper():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 5, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2
