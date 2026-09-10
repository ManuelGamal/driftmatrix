"""Gate G1: does the set of bit-exact reuse boundaries track n_ubatch?

G0 found restore bit-exact at reuse boundaries that were multiples of the
server's physical micro-batch (n_ubatch = 512) and not elsewhere. That is what
micro-batch tiling predicts, but a single n_ubatch cannot tell tiling apart
from anything else that happens to repeat every 512 tokens. G1 varies n_ubatch
and checks that the exact set moves with it:

    k = 512   predicted exact at n_ubatch 256 and 512, NOT at 1024
    k = 768   predicted exact at n_ubatch 256 only

"Exact" is binary and strict: identical tokens and bitwise-identical top-k
logprobs at every generated position. The recompute arm is computed once per
server run and reused as the reference for every split.
"""
from __future__ import annotations

from .g0 import SLACK, recompute_was_clean
from .metrics.distributional import context_matched_drift
from .runner.llamaserver import ServerError, first_divergence, restore_was_used

SPLITS = (256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536)
MIN_SUFFIX = 64


def completions_exact(a, b) -> bool:
    """Identical tokens and bitwise-identical top-k logprobs everywhere."""
    if list(a.tokens) != list(b.tokens) or len(a.top) != len(b.top):
        return False
    return all([tuple(x) for x in pa] == [tuple(x) for x in pb]
               for pa, pb in zip(a.top, b.top))


def mismatches(cases):
    """Splits that contradict 'exact iff aligned', in each direction."""
    valid = [c for c in cases if "exact" in c]
    return ([c["k"] for c in valid if c["aligned"] and not c["exact"]],
            [c["k"] for c in valid if not c["aligned"] and c["exact"]])


def g1_verdict(report: dict) -> str:
    floor = report["floor"]
    cases = report["cases"]
    if False in floor["recompute_clean"]:
        return "RECOMPUTE_NOT_CLEAN"
    if not floor["exact"]:
        return "NONDETERMINISTIC_FLOOR"
    if any("error" in c for c in cases):
        return "STATE_API_ERROR"
    if any(c["restore_used"] is False or not c["saved_exact"] for c in cases):
        return "RESTORE_NOT_VALID"
    if not cases or all(c["aligned"] for c in cases) or not any(c["aligned"] for c in cases):
        return "SPLITS_DO_NOT_DISCRIMINATE"
    aligned_not_exact, misaligned_exact = mismatches(cases)
    if not aligned_not_exact and not misaligned_exact:
        return "PREDICTION_HOLDS"
    if aligned_not_exact and not misaligned_exact:
        return "ALIGNED_NOT_ALWAYS_EXACT"
    if misaligned_exact and not aligned_not_exact:
        return "MISALIGNED_SOMETIMES_EXACT"
    return "PREDICTION_FAILS"


def run_g1(client, prompt_text: str, ubatch: int, splits=SPLITS, n_predict: int = 16,
           n_probs: int = 10, slot_file: str = "dm_g1_prefix.bin") -> dict:
    report = {"ubatch": ubatch, "n_predict": n_predict, "errors": [], "cases": []}
    try:
        report["props"] = client.props()
    except Exception as e:  # noqa: BLE001 - diagnostic only
        report["errors"].append(f"props: {e}")

    toks = client.tokenize(prompt_text)
    n = len(toks)
    report["n_tokens"] = n
    usable = [k for k in splits if k <= n - MIN_SUFFIX]
    report["splits_skipped"] = [k for k in splits if k not in usable]

    client.slot_erase()
    rc1 = client.complete(toks, n_predict, n_probs, cache_prompt=False)
    client.slot_erase()
    rc2 = client.complete(toks, n_predict, n_probs, cache_prompt=False)
    report["floor"] = {
        "exact": completions_exact(rc1, rc2),
        "recompute_clean": [recompute_was_clean(rc1.prompt_n, n),
                            recompute_was_clean(rc2.prompt_n, n)],
    }

    for k in usable:
        case = {"k": k, "aligned": k % ubatch == 0}
        try:
            client.slot_erase()
            client.complete(toks[:k], 0, 0, cache_prompt=True)
            case["n_saved"] = client.slot_save(slot_file).get("n_saved")
            client.slot_erase()
            client.slot_restore(slot_file)
            rs = client.complete(toks, n_predict, n_probs, cache_prompt=True)
        except ServerError as e:
            case["error"] = str(e)
            report["cases"].append(case)
            continue
        fd = first_divergence(rs.tokens, rc1.tokens)
        js, tv = context_matched_drift(rs.top, rc1.top, fd, max_positions=n_predict)
        case.update({
            "restore_prompt_n": rs.prompt_n,
            "restore_used": restore_was_used(rs.prompt_n, k, n, SLACK),
            "saved_exact": case["n_saved"] == k,
            "exact": completions_exact(rs, rc1),
            "first_divergence": fd,
            "max_jsd": max(js) if js else 0.0,
            "max_tv": max(tv) if tv else 0.0,
        })
        report["cases"].append(case)

    report["aligned_not_exact"], report["misaligned_exact"] = mismatches(report["cases"])
    report["verdict"] = g1_verdict(report)
    return report


def cross_ubatch_summary(reports) -> dict:
    """Exactness grid over (k, n_ubatch) and whether the exact set moves with n_ubatch."""
    ubs = sorted(r["ubatch"] for r in reports)
    ks = sorted({c["k"] for r in reports for c in r["cases"] if "exact" in c})
    grid = {}
    for k in ks:
        grid[k] = {}
        for r in reports:
            hit = [c["exact"] for c in r["cases"] if c.get("k") == k and "exact" in c]
            grid[k][r["ubatch"]] = hit[0] if hit else None
    discriminating = [k for k in ks
                      if len({v for v in grid[k].values() if v is not None}) > 1]
    tracks = (len(ubs) >= 2 and bool(discriminating)
              and all(r["verdict"] == "PREDICTION_HOLDS" for r in reports))
    return {"ubatches": ubs, "grid": grid, "discriminating_splits": discriminating,
            "exact_set_tracks_ubatch": tracks,
            "verdicts": {r["ubatch"]: r["verdict"] for r in reports}}
