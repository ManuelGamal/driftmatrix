"""Main-study grid against llama-server (spec 4.3; pre-registration sections 4, 6 and 11).

For every turn: render offline, tokenize with the server, find the natural reuse
boundary (the longest common token prefix of the history and the full prompt)
and its snapped version, then run the pre-registered arms:

    replicate 1: recompute, restore at the natural boundary, restore at the snapped boundary
    replicate 2: the same again

The slot is erased before every arm. When the natural boundary is already a
multiple of n_ubatch there is no separate snapped arm; the natural run counts in
both conditions (restore_record resolves it).

Every arm is proven, not assumed. A restore is `ok` only if the saved slot held
exactly k tokens and the server evaluated only the suffix; a recompute only if it
evaluated the whole prompt. Otherwise the arm is recorded as save_mismatch,
restore_not_used, recompute_not_clean or proof_unavailable, and the analysis
excludes and counts it. Server errors and timeouts leave the turn incomplete, so
a rerun retries it.

Blinding. The runner computes no treatment outcome - no exactness, drift or
action comparison - and its progress lines report arm validity only, so a run can
be monitored without examining effects. Outcomes are computed afterwards from the
records: g3.py for the pilot, the analysis module for the main study.

Records are JSON lines, appended a whole turn at a time and fsynced, so a run
killed mid-turn resumes at that turn. The first line holds the run's metadata;
a resumed run must match it.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import ENGINE_BUILD, MAX_NEW_TOKENS, STUDY_UBATCH, TOP_K
from .g0 import SLACK, recompute_was_clean
from .runner.llamaserver import ServerError, restore_was_used
from .traces.render import _last_user_index, render, reuse_boundary, snap

PREFIX_SLOT = "dm_study_prefix.bin"
ENGINE_ERRORS = (ServerError, OSError, TimeoutError)     # urllib's URLError is an OSError
FAILED = ("server_error", "timeout")
META_KEYS = ("model", "gguf", "build", "trace_hash", "ubatch", "n_predict", "n_probs")


class StudyConfigError(RuntimeError):
    """The server, model or trace set does not match the run."""


def loaded_model_matches(props, gguf) -> bool:
    path = str((props or {}).get("model_path") or "")
    return bool(path) and Path(path.replace("\\", "/")).name == Path(str(gguf).replace("\\", "/")).name


def plan_turn(tokenize, template, conv, specials, ubatch):
    k, full = reuse_boundary(tokenize, template, conv.messages, conv.tools,
                             enable_thinking=False, **specials)
    head = render(template, conv.messages[:_last_user_index(conv.messages)], conv.tools,
                  add_generation_prompt=False, enable_thinking=False, **specials)
    return {"tokens": full, "k_natural": k, "k_snapped": snap(k, ubatch),
            "history_tokens": len(tokenize(head))}


def _completion_fields(c):
    return {"prompt_n": c.prompt_n, "tokens": list(c.tokens),
            "top": [[[t, lp] for t, lp in pos] for pos in c.top],
            "pieces": list(c.pieces), "content": c.raw.get("content") or ""}


def _recompute(client, toks, n_predict, n_probs):
    client.slot_erase()
    c = client.complete(toks, n_predict, n_probs, cache_prompt=False)
    clean = recompute_was_clean(c.prompt_n, len(toks))
    outcome = "ok" if clean else ("proof_unavailable" if clean is None else "recompute_not_clean")
    return outcome, {"n_saved": None, **_completion_fields(c)}


def _restore(client, toks, k, n_predict, n_probs):
    client.slot_erase()
    client.complete(toks[:k], 0, 0, cache_prompt=True)
    n_saved = client.slot_save(PREFIX_SLOT).get("n_saved")
    client.slot_erase()
    client.slot_restore(PREFIX_SLOT)
    c = client.complete(toks, n_predict, n_probs, cache_prompt=True)
    used = restore_was_used(c.prompt_n, k, len(toks), SLACK)
    if n_saved != k:
        outcome = "save_mismatch"
    elif used is None:
        outcome = "proof_unavailable"
    else:
        outcome = "ok" if used else "restore_not_used"
    return outcome, {"n_saved": n_saved, **_completion_fields(c)}


def compare_files(a, b, chunk: int = 1 << 24) -> dict:
    """Byte-for-byte comparison of two slot files, read in chunks."""
    a, b = Path(a), Path(b)
    size_a, size_b = a.stat().st_size, b.stat().st_size
    differing = abs(size_a - size_b)
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            xa, xb = fa.read(chunk), fb.read(chunk)
            n = min(len(xa), len(xb))
            if n == 0:
                break
            differing += int(np.count_nonzero(np.frombuffer(xa, np.uint8, n) != np.frombuffer(xb, np.uint8, n)))
    return {"equal": differing == 0, "differing_bytes": differing, "bytes_a": size_a, "bytes_b": size_b}


def run_turn(client, conv, template, specials, *, model, ubatch, n_predict, n_probs, state_dir=None):
    attempt = uuid.uuid4().hex
    base = {"model": model, "conv_id": conv.conv_id, "base_id": conv.base_id,
            "prefix_band": conv.prefix_band, "stratum": conv.stratum, "attempt": attempt}
    try:
        plan = plan_turn(client.tokenize, template, conv, specials, ubatch)
    except ENGINE_ERRORS as e:
        return [{"kind": "turn", **base, "complete": False, "error": f"plan: {e}"}]

    toks, k_nat, k_snap = plan["tokens"], plan["k_natural"], plan["k_snapped"]
    snapped_is_natural = k_snap == k_nat
    lines, finals = [], {}
    for rep in (1, 2):
        arms = [("recompute", None, None), ("restore", "natural", k_nat)]
        if not snapped_is_natural:
            arms.append(("restore", "snapped", k_snap))
        for arm, boundary, k in arms:
            t0 = time.monotonic()
            error, fields = None, {}
            try:
                if arm == "recompute":
                    outcome, fields = _recompute(client, toks, n_predict, n_probs)
                elif k <= 0:
                    # restoring an empty prefix is a recompute that passes every proof;
                    # the pre-registered budget rule guarantees k > 0, so this is never valid
                    outcome = "boundary_zero"
                else:
                    outcome, fields = _restore(client, toks, k, n_predict, n_probs)
            except ServerError as e:
                outcome, error = "server_error", str(e)
            except (OSError, TimeoutError) as e:
                outcome, error = "timeout", str(e)
            rec = {"kind": "arm", **base, "replicate": rep, "arm": arm, "boundary": boundary, "k": k,
                   "outcome": outcome, "error": error, "seconds": round(time.monotonic() - t0, 3), **fields}
            if state_dir is not None and rep == 1 and error is None:
                name = f"dm_study_final_{arm}_{boundary or 'none'}.bin"
                try:
                    rec["n_saved_final"] = client.slot_save(name).get("n_saved")
                    finals[(arm, boundary)] = Path(state_dir) / name
                except ENGINE_ERRORS as e:
                    rec["state_save_error"] = str(e)
            lines.append(rec)

    state = None
    if state_dir is not None:
        state = {}
        rc = finals.get(("recompute", None))
        for boundary in ("natural", "snapped"):
            f = finals.get(("restore", boundary))
            if rc is not None and f is not None and rc.exists() and f.exists():
                state[boundary] = compare_files(rc, f)
        for f in [*finals.values(), Path(state_dir) / PREFIX_SLOT]:
            try:
                f.unlink()
            except OSError:
                pass

    lines.append({"kind": "turn", **base, "n_prompt": len(toks), "history_tokens": plan["history_tokens"],
                  "k_natural": k_nat, "k_snapped": k_snap, "k_mod_ubatch": k_nat % ubatch,
                  "snapped_is_natural": snapped_is_natural, "state_bytes": state,
                  "complete": not any(r["outcome"] in FAILED for r in lines), "error": None})
    return lines


def _append(path: Path, lines) -> None:
    needs_newline = path.exists() and path.stat().st_size > 0 and path.read_bytes()[-1:] != b"\n"
    with open(path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")                  # a line cut off by a crash must not swallow the next one
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_records(path):
    """(meta, complete turns by conv_id, arms by (conv_id, arm, boundary, replicate)).

    Only arms from each turn's latest complete attempt are kept; unparseable
    lines (a write cut off by a crash) are skipped.
    """
    meta, turns, by_attempt = None, {}, defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            kind = rec.get("kind")
            if kind == "meta" and meta is None:
                meta = rec
            elif kind == "arm":
                by_attempt[rec["attempt"]].append(rec)
            elif kind == "turn" and rec.get("complete"):
                turns[rec["conv_id"]] = rec
    arms = {}
    for conv_id, turn in turns.items():
        for a in by_attempt[turn["attempt"]]:
            arms[(conv_id, a["arm"], a["boundary"], a["replicate"])] = a
    return meta, turns, arms


def restore_record(arms, turn, boundary, replicate):
    """The restore arm for a boundary; a snapped boundary equal to the natural one reuses its run."""
    if boundary == "snapped" and turn["snapped_is_natural"]:
        boundary = "natural"
    return arms.get((turn["conv_id"], "restore", boundary, replicate))


def run_study(client, turns, template, specials, out_path, *, model, gguf, trace_hash_value,
              expected_build=ENGINE_BUILD, ubatch=STUDY_UBATCH, n_predict=MAX_NEW_TOKENS,
              n_probs=TOP_K, state_dir=None, progress=None, max_consecutive_failures=3) -> dict:
    props = client.props()
    build = props.get("build_info")
    if expected_build is not None and build != expected_build:
        raise StudyConfigError(f"server build {build!r}, pre-registered {expected_build!r}")
    if not loaded_model_matches(props, gguf):
        raise StudyConfigError(f"server has {props.get('model_path')!r} loaded, not {Path(gguf).name}")

    meta = {"kind": "meta", "model": model, "gguf": Path(gguf).name, "build": build,
            "trace_hash": trace_hash_value, "ubatch": ubatch, "n_predict": n_predict,
            "n_probs": n_probs, "state_bytes": state_dir is not None}
    out_path = Path(out_path)
    done = {}
    if out_path.exists() and out_path.stat().st_size > 0:
        existing, done, _ = load_records(out_path)
        if existing is None or any(existing.get(k) != meta[k] for k in META_KEYS):
            raise StudyConfigError(f"{out_path} belongs to a different run; move it aside to start a new one")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _append(out_path, [meta])

    todo = [t for t in turns if t.conv_id not in done]
    counts, consecutive, started = Counter(), 0, time.monotonic()
    for i, conv in enumerate(todo, 1):
        lines = run_turn(client, conv, template, specials, model=model, ubatch=ubatch,
                         n_predict=n_predict, n_probs=n_probs, state_dir=state_dir)
        _append(out_path, lines)
        arm_outcomes = [r["outcome"] for r in lines if r["kind"] == "arm"]
        counts.update(arm_outcomes)
        consecutive = consecutive + 1 if all(o in FAILED for o in arm_outcomes) else 0
        if progress is not None:
            per_turn = (time.monotonic() - started) / i
            progress(f"[{i}/{len(todo)}] {conv.conv_id}  valid arms {arm_outcomes.count('ok')}/{len(arm_outcomes)}"
                     f" | run totals {dict(sorted(counts.items()))}"
                     f" | {per_turn:.0f}s per turn, about {per_turn * (len(todo) - i) / 60:.0f} min left")
        if consecutive >= max_consecutive_failures:
            raise RuntimeError(f"{consecutive} turns in a row failed on the server; is llama-server still running?")
    return {"turns_run": len(todo), "turns_skipped": len(turns) - len(todo), "outcomes": dict(counts)}
