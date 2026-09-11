"""Spark window follow-up: the scoring must single out whichever mechanism the engine actually has."""
import pytest

from driftmatrix.followup import (
    HYPOTHESES, SPLITS, WINDOW, compare, evaluate, exact_set, predicted_exact, valid_cases,
)
from driftmatrix.g1 import run_g1
from driftmatrix.runner.llamaserver import Completion

PROMPT = " ".join(f"w{i}" for i in range(4500))


def tiling(start, end, ub):
    return tuple(range(start, end, ub))


class WindowServer:
    """Fake llama-server whose restore exactness follows a chosen mechanism."""

    def __init__(self, ubatch, mechanism, window=WINDOW):
        self.ub, self.mechanism, self.window = ubatch, mechanism, window
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
        return {"n_restored": len(self.cache)}

    def complete(self, toks, n_predict, n_probs=10, cache_prompt=True, slot=0, seed=1234):
        n = len(toks)
        reuse = bool(cache_prompt and self.cache and toks[:len(self.cache)] == self.cache)
        k = len(self.cache) if reuse else 0
        starts = self.starts + tiling(k, n, self.ub) if reuse else tiling(0, n, self.ub)
        misaligned = starts != tiling(0, n, self.ub)
        perturbed = {"tiling_only": misaligned,
                     "window_limit": misaligned or (reuse and k > self.window),
                     "single_microbatch": misaligned or (reuse and k > self.ub)}[self.mechanism]
        out = [sum(toks) % 997 + i for i in range(n_predict)]
        top = [[(t, -0.1), (t + 1, -2.0)] for t in out]
        if perturbed and n_predict:
            top = [[(t, lp + 1e-6) for t, lp in pos] for pos in top]
        self.cache, self.starts = list(toks), starts
        return Completion(out, top, n - k, k, {})


def runs(mechanism):
    return [{**run_g1(WindowServer(ub, mechanism), PROMPT, ubatch=ub, splits=SPLITS), "label": f"ub{ub}"}
            for ub in (256, 512, 1024)]


def test_predictions_written_in_the_docstring():
    assert predicted_exact("window_limit", 256) == [256, 512]
    assert predicted_exact("window_limit", 512) == [512]
    assert predicted_exact("window_limit", 1024) == []
    assert [predicted_exact("single_microbatch", ub) for ub in (256, 512, 1024)] == [[256], [512], [1024]]
    assert predicted_exact("window_regardless_of_alignment", 1024) == [256, 384, 512]


def test_evidence_so_far_cannot_separate_the_window_rules_at_n_ubatch_512():
    # G0 and the main study ran n_ubatch 512 only: exact at 512, inexact at 1024 and 4096
    for name in ("window_limit", "single_microbatch"):
        rule = HYPOTHESES[name]
        assert rule(512, 512) and not rule(1024, 512) and not rule(4096, 512)
    assert predicted_exact("window_limit", 1024) != predicted_exact("single_microbatch", 1024)


@pytest.mark.parametrize("mechanism", ["tiling_only", "window_limit", "single_microbatch"])
def test_scoring_singles_out_the_engines_mechanism(mechanism):
    result = evaluate(runs(mechanism))
    assert result[mechanism]["contradictions"] == [] and result[mechanism]["tested"] > 0
    for other in HYPOTHESES:
        if other != mechanism:
            assert result[other]["contradictions"], other


def test_exact_sets_follow_the_mechanism():
    window = {r["ubatch"]: exact_set(r) for r in runs("window_limit")}
    assert window == {256: [256, 512], 512: [512], 1024: []}


def test_unproven_cases_and_nondeterministic_floors_are_not_scored():
    report = runs("window_limit")[1]
    report["cases"][0]["restore_used"] = False
    assert len(valid_cases(report)[1]) == 1
    assert evaluate([report])["tiling_only"]["tested"] == len(report["cases"]) - 1
    report["floor"]["exact"] = False
    assert evaluate([report])["tiling_only"]["tested"] == 0


def test_compare_reports_what_a_diagnostic_run_changed():
    baseline = run_g1(WindowServer(512, "window_limit"), PROMPT, ubatch=512, splits=SPLITS)
    fixed = run_g1(WindowServer(512, "tiling_only"), PROMPT, ubatch=512, splits=SPLITS)
    assert compare(fixed, baseline) == {"became_exact": [1024, 1536, 2048, 2560, 3072, 3584, 4096],
                                        "became_inexact": [], "proven_in_only_one_run": []}
