"""Gate G0 against llama-server: does slot state save/restore round-trip?

Stdlib only. Runs these measurements and returns a report with one verdict:

  floor      recompute the full prompt twice; must be identical, or no other
             comparison means anything
  cases      at two split points - one aligned to 64 tokens, one offset by 32 -
             fill the slot with the prefix, save, erase, restore, then complete
             the full prompt; compare against a clean recompute
  proof      each arm's own prompt_n is checked, in both directions: a restore
             arm must have evaluated only the suffix, and a recompute arm must
             have evaluated the whole prompt

Both proofs exist because llama-server can fail silently either way. On hybrid
models it may reprocess a restored prompt from scratch (llama.cpp #20225), and
its host-RAM prompt cache can hand a "recompute" arm state it never computed.
Either failure makes the two arms secretly identical and looks like a pass.

Verdicts, checked in this order because each invalidates the ones after it:
  STATE_API_UNSUPPORTED      slot save/restore endpoints refused
  RECOMPUTE_NOT_CLEAN        a recompute arm reused cached state it never computed
  NONDETERMINISTIC_FLOOR     engine differs from itself with no restore at all
  SAVE_MISMATCH              the saved slot did not hold exactly the prefix
  RESTORE_NOT_USED           server reprocessed the full prompt (#20225 trap)
  DIVERGENCE_ON_DAY_ONE      restore was used and output changed
  FAITHFUL_AT_TESTED_SPLITS  restore was used and output was identical
"""
from __future__ import annotations

from .runner.llamaserver import (
    ServerError, first_divergence, max_top1_logprob_gap, restore_was_used,
)

CHUNK = 64
SLACK = 2  # the server may re-evaluate the last prompt token or two


def split_points(n_tokens: int, chunk: int = CHUNK) -> dict:
    base = max(chunk, (n_tokens // 2 // chunk) * chunk)
    return {"aligned_64": base, "offset_32": base + chunk // 2}


def recompute_was_clean(prompt_n: int, n_full: int):
    """True if the whole prompt was evaluated. None if prompt_n is unreported."""
    if prompt_n is None or prompt_n < 0:
        return None
    return prompt_n >= n_full - SLACK


def verdict(report: dict) -> str:
    cases = report.get("cases", {})
    floor = report.get("floor", {})
    if any("state_api_error" in c for c in cases.values()):
        return "STATE_API_UNSUPPORTED"
    if (False in floor.get("recompute_clean", [])
            or any(c.get("recompute_clean") is False for c in cases.values())):
        return "RECOMPUTE_NOT_CLEAN"
    if not floor.get("identical", True):
        return "NONDETERMINISTIC_FLOOR"
    if any(c.get("n_saved") not in (None, c["split_at"]) for c in cases.values()):
        return "SAVE_MISMATCH"
    if any(c.get("restore_used") is False for c in cases.values()):
        return "RESTORE_NOT_USED"
    if any(not c.get("identical", True) for c in cases.values()):
        return "DIVERGENCE_ON_DAY_ONE"
    return "FAITHFUL_AT_TESTED_SPLITS"


def run_g0(client, prompt_text: str, n_predict: int = 32, n_probs: int = 5,
           slot_file: str = "dm_g0_prefix.bin") -> dict:
    report = {"errors": [], "cases": {}}
    try:
        report["props"] = client.props()
    except Exception as e:  # noqa: BLE001 - props are diagnostic only
        report["errors"].append(f"props: {e}")

    toks = client.tokenize(prompt_text)
    n = len(toks)
    report["n_tokens"] = n

    client.slot_erase()
    rc1 = client.complete(toks, n_predict, n_probs, cache_prompt=False)
    client.slot_erase()
    rc2 = client.complete(toks, n_predict, n_probs, cache_prompt=False)
    report["floor"] = {
        "identical": rc1.tokens == rc2.tokens,
        "first_divergence": first_divergence(rc1.tokens, rc2.tokens),
        "max_top1_logprob_gap": max_top1_logprob_gap(rc1.top, rc2.top),
        "prompt_n": [rc1.prompt_n, rc2.prompt_n],
        "recompute_clean": [recompute_was_clean(rc1.prompt_n, n),
                            recompute_was_clean(rc2.prompt_n, n)],
    }

    for name, k in split_points(n).items():
        case = {"split_at": k, "n_suffix": n - k}
        try:
            client.slot_erase()
            client.complete(toks[:k], 0, 0, cache_prompt=True)
            case["n_saved"] = client.slot_save(slot_file).get("n_saved")
            client.slot_erase()
            case["n_restored"] = client.slot_restore(slot_file).get("n_restored")
            rs = client.complete(toks, n_predict, n_probs, cache_prompt=True)
        except ServerError as e:
            case["state_api_error"] = str(e)
            report["cases"][name] = case
            continue

        client.slot_erase()
        rc = client.complete(toks, n_predict, n_probs, cache_prompt=False)
        case.update({
            "restore_prompt_n": rs.prompt_n,
            "restore_tokens_cached": rs.tokens_cached,
            "recompute_prompt_n": rc.prompt_n,
            "restore_used": restore_was_used(rs.prompt_n, k, n, SLACK),
            "recompute_clean": recompute_was_clean(rc.prompt_n, n),
            "identical": rs.tokens == rc.tokens,
            "first_divergence": first_divergence(rs.tokens, rc.tokens),
            "max_top1_logprob_gap": max_top1_logprob_gap(rs.top, rc.top),
            "restore_tokens": rs.tokens,
            "recompute_tokens": rc.tokens,
        })
        report["cases"][name] = case

    report["verdict"] = verdict(report)
    return report
