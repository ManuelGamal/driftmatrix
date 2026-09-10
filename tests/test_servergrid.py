"""Main-study runner and G3 against a fake server in which micro-batch tiling is the mechanism."""
import json
from pathlib import Path

import pytest

from driftmatrix.config import ENGINE_BUILD
from driftmatrix.g3 import g3_check, records_exact
from driftmatrix.runner.llamaserver import Completion, ServerError
from driftmatrix.servergrid import (
    StudyConfigError, compare_files, load_records, loaded_model_matches, restore_record, run_study,
)
from driftmatrix.traces.conversations import generate_study_turns, trace_hash
from driftmatrix.traces.render import reuse_boundary

TEMPLATE = ("{% for m in messages %}<{{ m.role }}>{{ m.content }}"
            "{% for tc in (m.tool_calls or []) %}[{{ tc.function.name }} {{ tc.function.arguments | tojson }}]"
            "{% endfor %}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}")
TURNS = generate_study_turns(2, seed=1, round_trips={"short": 1, "long": 2})
HASH = trace_hash(TURNS)
GGUF = "C:/models/fake-Q4_K_M.gguf"


def tiling(start, end, ub):
    return tuple(range(start, end, ub))


class ChatTilingServer:
    def __init__(self, ubatch=64, mode="tiling", flip=True, slot_dir=None, build=ENGINE_BUILD,
                 model_path="C:\\models\\fake-Q4_K_M.gguf", reprocess=False, leaky=False,
                 short_save=False, restore_error=None, tokenize_error=None):
        self.ub, self.mode, self.flip, self.slot_dir = ubatch, mode, flip, slot_dir
        self.build, self.model_path = build, model_path
        self.reprocess, self.leaky, self.short_save = reprocess, leaky, short_save
        self.restore_error, self.tokenize_error = restore_error, tokenize_error
        self.cache, self.starts, self.saved, self.perturbed = [], (), {}, False
        self.tokenize_calls = 0

    def props(self):
        return {"build_info": self.build, "model_path": self.model_path}

    def tokenize(self, text, add_special=True):
        self.tokenize_calls += 1
        if self.tokenize_error:
            raise self.tokenize_error
        return [ord(c) for c in text]

    def slot_erase(self, slot=0):
        self.cache, self.starts, self.perturbed = [], (), False
        return {}

    def slot_save(self, filename, slot=0):
        self.saved[filename] = (list(self.cache), self.starts, self.perturbed)
        if self.slot_dir is not None:
            (Path(self.slot_dir) / filename).write_bytes(json.dumps([self.cache, self.perturbed]).encode())
        return {"n_saved": len(self.cache) - (1 if self.short_save else 0)}

    def slot_restore(self, filename, slot=0):
        if self.restore_error:
            raise self.restore_error
        self.cache, self.starts, self.perturbed = list(self.saved[filename][0]), *self.saved[filename][1:]
        return {"n_restored": len(self.cache)}

    def complete(self, toks, n_predict, n_probs=10, cache_prompt=True, slot=0, seed=1234):
        n = len(toks)
        reuse = bool(cache_prompt and self.cache and not self.reprocess and toks[:len(self.cache)] == self.cache)
        if reuse:
            starts, prompt_n = self.starts + tiling(len(self.cache), n, self.ub), n - len(self.cache)
        else:
            starts, prompt_n = tiling(0, n, self.ub), n
        if self.leaky and not cache_prompt:
            prompt_n = 1
        perturbed = {"tiling": starts != tiling(0, n, self.ub), "always": reuse, "never": False}[self.mode]
        out = [sum(toks) % 997 + i for i in range(n_predict)]
        top = [[(t, -0.1), (t + 1, -2.0)] for t in out]
        if perturbed and n_predict > 1:
            top = [[(t, lp + 1e-6) for t, lp in pos] for pos in top]
            if self.flip:
                out[1] += 1
                top[1] = [(out[1], -0.1), (out[1] + 1, -2.0)]
        self.cache, self.starts, self.perturbed = list(toks) + out, starts, perturbed
        return Completion(out, top, prompt_n, n - prompt_n, {"content": " ".join(map(str, out))},
                          tuple(str(t) for t in out))


def run(server, path, turns=TURNS, **kw):
    kw.setdefault("model", "ornith")
    kw.setdefault("gguf", GGUF)
    kw.setdefault("trace_hash_value", HASH)
    kw.setdefault("n_predict", 8)
    kw.setdefault("ubatch", server.ub)          # the runner must be told the server's -ub
    return run_study(server, turns, TEMPLATE, {}, path, **kw)


def chars(text):
    return [ord(c) for c in text]


def test_a_zero_boundary_is_never_restored(tmp_path):
    class Counting(ChatTilingServer):
        empty_prompts = 0

        def complete(self, toks, *a, **kw):
            Counting.empty_prompts += not toks
            return super().complete(toks, *a, **kw)

    out = tmp_path / "r.jsonl"
    run(Counting(ubatch=1 << 16), out)             # every prompt is shorter than one micro-batch
    meta, turns, arms = load_records(out)
    assert {a["outcome"] for a in arms.values() if a["boundary"] == "snapped"} == {"boundary_zero"}
    assert Counting.empty_prompts == 0
    assert g3_check(meta, turns, arms)["verdict"] == "ARM_PROOFS_FAIL"


def test_every_turn_runs_the_pre_registered_arms_in_order(tmp_path):
    out = tmp_path / "ornith.jsonl"
    summary = run(ChatTilingServer(), out)
    assert summary["turns_run"] == len(TURNS) and summary["turns_skipped"] == 0
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["kind"] == "meta" and lines[0]["build"] == ENGINE_BUILD
    meta, turns, arms = load_records(out)
    assert len(turns) == len(TURNS)
    first = [x for x in lines[1:] if x["conv_id"] == TURNS[0].conv_id and x["kind"] == "arm"]
    assert [(a["replicate"], a["arm"], a["boundary"]) for a in first] == [
        (1, "recompute", None), (1, "restore", "natural"), (1, "restore", "snapped"),
        (2, "recompute", None), (2, "restore", "natural"), (2, "restore", "snapped")]
    assert all(a["outcome"] == "ok" for a in arms.values())


def test_boundaries_are_the_reuse_boundary_and_its_snap(tmp_path):
    out = tmp_path / "r.jsonl"
    run(ChatTilingServer(), out)
    _, turns, arms = load_records(out)
    for conv in TURNS:
        k, full = reuse_boundary(chars, TEMPLATE, conv.messages, conv.tools, enable_thinking=False)
        t = turns[conv.conv_id]
        assert t["k_natural"] == k and t["n_prompt"] == len(full)
        assert t["k_snapped"] % 64 == 0 and 0 <= k - t["k_snapped"] < 64
        assert t["prefix_band"] == conv.prefix_band and t["stratum"] == conv.stratum
        assert restore_record(arms, t, "snapped", 1)["k"] == t["k_snapped"]


def test_aligned_natural_boundary_has_no_separate_snapped_arm(tmp_path):
    out = tmp_path / "r.jsonl"
    run(ChatTilingServer(ubatch=1), out)
    _, turns, arms = load_records(out)
    for t in turns.values():
        assert t["snapped_is_natural"] is True
        assert (t["conv_id"], "restore", "snapped", 1) not in arms
        assert restore_record(arms, t, "snapped", 1) is restore_record(arms, t, "natural", 1)


@pytest.mark.parametrize("server, arm, outcome", [
    (ChatTilingServer(reprocess=True), "restore", "restore_not_used"),
    (ChatTilingServer(leaky=True), "recompute", "recompute_not_clean"),
    (ChatTilingServer(short_save=True), "restore", "save_mismatch"),
])
def test_failed_arm_proofs_are_recorded_as_data(tmp_path, server, arm, outcome):
    out = tmp_path / "r.jsonl"
    run(server, out)
    _, turns, arms = load_records(out)
    assert len(turns) == len(TURNS)                  # deterministic failures complete the turn
    assert {a["outcome"] for a in arms.values() if a["arm"] == arm} == {outcome}


def test_server_errors_leave_turns_incomplete_and_a_rerun_retries_them(tmp_path):
    out = tmp_path / "r.jsonl"
    run(ChatTilingServer(restore_error=ServerError(500, "boom", "/slots/0?action=restore")), out)
    assert load_records(out)[1] == {}
    summary = run(ChatTilingServer(), out)
    assert summary["turns_skipped"] == 0 and len(load_records(out)[1]) == len(TURNS)


def test_resume_skips_complete_turns(tmp_path):
    out = tmp_path / "r.jsonl"
    run(ChatTilingServer(), out, turns=TURNS[:3])
    summary = run(ChatTilingServer(), out)
    assert summary == {"turns_run": len(TURNS) - 3, "turns_skipped": 3, "outcomes": summary["outcomes"]}
    assert len(load_records(out)[1]) == len(TURNS)


def test_a_line_cut_off_by_a_crash_does_not_corrupt_the_resumed_run(tmp_path):
    out = tmp_path / "r.jsonl"
    run(ChatTilingServer(), out, turns=TURNS[:2])
    with open(out, "a", encoding="utf-8") as f:
        f.write('{"kind": "arm", "conv_id": "half a li')
    run(ChatTilingServer(), out)
    _, turns, arms = load_records(out)
    assert len(turns) == len(TURNS) and len(arms) == 6 * len(TURNS)


def test_resume_refuses_a_different_run(tmp_path):
    out = tmp_path / "r.jsonl"
    run(ChatTilingServer(), out, turns=TURNS[:1])
    with pytest.raises(StudyConfigError, match="different run"):
        run(ChatTilingServer(), out, trace_hash_value="other")


def test_wrong_model_or_build_is_refused(tmp_path):
    with pytest.raises(StudyConfigError, match="loaded"):
        run(ChatTilingServer(model_path="C:\\models\\other.gguf"), tmp_path / "a.jsonl")
    with pytest.raises(StudyConfigError, match="build"):
        run(ChatTilingServer(build="b9999-deadbeef"), tmp_path / "b.jsonl")


def test_loaded_model_matches_compares_file_names_across_path_styles():
    assert loaded_model_matches({"model_path": "C:\\a\\m.gguf"}, "/home/x/m.gguf")
    assert not loaded_model_matches({}, "m.gguf")


def test_a_dead_server_aborts_the_run(tmp_path):
    with pytest.raises(RuntimeError, match="in a row"):
        run(ChatTilingServer(tokenize_error=ConnectionRefusedError("down")), tmp_path / "r.jsonl")


def test_progress_reports_validity_only(tmp_path):
    messages = []
    run(ChatTilingServer(), tmp_path / "r.jsonl", progress=messages.append)
    assert len(messages) == len(TURNS)
    for word in ("exact", "drift", "jsd", "diverg", "action", "call"):
        assert not any(word in m.lower() for m in messages)


def test_state_bytes_compare_final_slots_and_clean_up(tmp_path):
    slots = tmp_path / "slots"
    slots.mkdir()
    out = tmp_path / "r.jsonl"
    run(ChatTilingServer(slot_dir=slots), out, state_dir=slots)
    _, turns, _ = load_records(out)
    for t in turns.values():
        assert t["state_bytes"]["natural"]["equal"] is False       # misaligned restore tiles differently
        assert t["state_bytes"]["snapped"]["equal"] is True
    assert list(slots.iterdir()) == []


def test_compare_files_counts_differing_bytes(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_bytes(b"abcdef")
    b.write_bytes(b"abXdef")
    c.write_bytes(b"abcdefgh")
    assert compare_files(a, a, chunk=4) == {"equal": True, "differing_bytes": 0, "bytes_a": 6, "bytes_b": 6}
    assert compare_files(a, b, chunk=4)["differing_bytes"] == 1
    assert compare_files(a, c, chunk=4)["differing_bytes"] == 2


# ------------------------------------------------------------------------------------ G3
def pilot(server, tmp_path, model="ornith"):
    out = tmp_path / f"{model}.jsonl"
    run(server, out, model=model)
    return g3_check(*load_records(out))


def test_g3_passes_when_tiling_is_the_mechanism(tmp_path):
    r = pilot(ChatTilingServer(), tmp_path)
    assert r["verdict"] == "G3_PASS" and r["natural_turns_with_drift"] > 0 and r["floor_inexact_turns"] == 0


def test_g3_detects_drift_without_a_token_flip(tmp_path):
    assert pilot(ChatTilingServer(flip=False), tmp_path)["verdict"] == "G3_PASS"


def test_g3_fails_when_snapped_restores_are_not_exact(tmp_path):
    r = pilot(ChatTilingServer(mode="always"), tmp_path)
    assert r["verdict"] == "SNAPPED_NOT_EXACT" and r["snapped_inexact"]


def test_g3_positive_control_is_required_on_the_hybrid_only(tmp_path):
    assert pilot(ChatTilingServer(mode="never"), tmp_path)["verdict"] == "POSITIVE_CONTROL_SILENT"
    assert pilot(ChatTilingServer(mode="never"), tmp_path, model="qwen3")["verdict"] == "G3_PASS"


def test_g3_fails_on_arm_proofs(tmp_path):
    r = pilot(ChatTilingServer(reprocess=True), tmp_path)
    assert r["verdict"] == "ARM_PROOFS_FAIL" and r["invalid_arms"]


def test_records_exact_is_bitwise_on_logprobs():
    a = {"tokens": [1], "top": [[[1, -0.1]]]}
    assert records_exact(a, {"tokens": [1], "top": [[[1, -0.1]]]})
    assert not records_exact(a, {"tokens": [1], "top": [[[1, -0.1000001]]]})
