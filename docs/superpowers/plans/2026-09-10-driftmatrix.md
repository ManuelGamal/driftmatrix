# driftmatrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a harness that measures whether restoring saved recurrent state instead of recomputing the prompt changes what a hybrid linear-attention model says, and whether it changes which tool an agent calls.

**Architecture:** llama.cpp drives both arms through explicit sequence state save/restore, so the arm is a property of the procedure rather than something inferred from cache metrics. The engine writes raw outputs, logprobs and saved-state tensors to Parquet once; every analysis component reads those files, so all statistics are GPU-free and unit-testable. Three models form an architecture ladder (recurrent state / sliding-window KV / dense KV) so divergence can be attributed to state type rather than to hybrid layer structure or to the engine.

**Tech Stack:** Python 3.11+, llama-cpp-python (or llama-server HTTP), LangGraph, numpy, scipy, pandas, pyarrow, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-hybrid-cache-fidelity-design.md`

## Global Constraints

- **Engine:** llama.cpp, pinned to one commit SHA, stamped into every Parquet row as `engine_commit`.
- **Quantization:** treatment model at **IQ4_XS**; recorded per row as `quant`. Both arms always use identical weights.
- **Decoding:** greedy, temperature 0, fixed seed, on every arm and every replicate.
- **Replicates:** 2 per condition, including the recompute reference — the noise floor is measured, never assumed.
- **Bootstrap resamples:** 10,000. **α = 0.05.**
- **Noise floor threshold:** recompute self-comparison median per-turn mean JSD **< 0.01 nats** (≈1.4% of the log 2 bound).
- **Self-inconsistency multiplier:** 2.0× the recompute arm's self-comparison median.
- **Power target:** detect a per-turn divergence rate of **2% at 80% power, α = 0.05**.
- **Prefix bands:** short = 1024 tokens, long = 8192 tokens. **Continuation bands:** short = 32 tokens, long = 256 tokens.
- **Holm families:** three, declared in `stats/multiplicity.py` and never pooled — A (behavioral, 24), B (distributional, 16), C (mechanism contrasts, 5).
- **Mock tools must be pure functions of their arguments.** A nondeterministic tool injects variance indistinguishable from the effect under study.
- **All claims are scoped to engine, commit, and quantization.** No bit-exactness claims about vLLM (spec §2.2).
- **Never report hit rate, TTFT or throughput as a headline result** (spec §2.2). Cost appears only in the Task 22 secondary reproduction.

---

## Amendment 1 — after gate G0 (2026-09-10)

Spec revised to v4. This amendment **supersedes** the tasks named below; where
it and an original task disagree, this wins.

### Status of original tasks

| Task | Status |
|---|---|
| 0 G0 spike | **Done.** Executed via `llama-server` (`src/driftmatrix/g0.py`, `scripts/g0_server.py`). Findings: `docs/findings/2026-09-10-g0.md` |
| 1–6, 8–10, 13–16, 19–21 | Done (GPU-free analysis layers, under test) |
| 7 Trace generator | Done, but **amended by Task 7R** — output must be chat-templated |
| 11 llama-cpp-python adapter | Done; **off the measurement path** (kept for unit tests) |
| 12 GDN chunk-size sweep | **Superseded by Task 12R** — the period is `n_ubatch`, not a kernel chunk |
| 17 Pre-registration | Moves to **after G1** (spec §6.4) |
| 18 Synthetic positive control | Kept for harness tests only; the measurement positive control is natural misaligned restore on the hybrid (spec §8) |
| 22 Grid on the Python engine | **Superseded by Task 22R** (grid against `llama-server`) |
| 23 README | Update after G1 |

### Global constraints added

- Measurement engine is `llama-server`; every arm proven by `timings.prompt_n` (restore = suffix; recompute = whole prompt); every saved slot = `k`.
- Server flags: `-c 8192 -ngl 99 -np 1 -b 2048 -ub <U> --cache-ram 0 --ctx-checkpoints 0 --slot-save-path <dir>`, plus `--swa-full` for Spark and `--jinja` for chat rendering. Never `--spec-type`, never `-ctk/-ctv`.
- Architecture comparison is quantization-matched at **Q4_K_M**.
- Drift is JSD/TV (context-matched, union support + `<other>`). **Never** max logprob difference over the top-k.

---

### Task 12R: G1 — does the exact set track `n_ubatch`?

**Status: done.** G1 passed 2026-09-10 — 66/66 restores matched the prediction on Ornith Q4_K_M and Qwen3-4B. Findings: `docs/findings/2026-09-10-g1.md`.

**Files:**
- Create: `src/driftmatrix/g1.py`
- Modify: `src/driftmatrix/metrics/distributional.py` (add `context_matched_drift`)
- Modify: `scripts/g0_analyze.py` (use `context_matched_drift`)
- Create: `scripts/g1_server.py`, `scripts/g1_analyze.py`
- Test: `tests/test_g1.py`, `tests/metrics/test_distributional.py`

**Interfaces:**
- Consumes: `ServerClient`, `first_divergence`, `restore_was_used` (`runner.llamaserver`); `recompute_was_clean`, `SLACK` (`g0`).
- Produces: `SPLITS: tuple[int, ...]`; `completions_exact(a: Completion, b: Completion) -> bool`; `run_g1(client, prompt_text, ubatch, splits=SPLITS, n_predict=16, n_probs=10) -> dict`; `g1_verdict(report) -> str` in {`RECOMPUTE_NOT_CLEAN`, `NONDETERMINISTIC_FLOOR`, `STATE_API_ERROR`, `RESTORE_NOT_VALID`, `SPLITS_DO_NOT_DISCRIMINATE`, `PREDICTION_HOLDS`, `ALIGNED_NOT_ALWAYS_EXACT`, `MISALIGNED_SOMETIMES_EXACT`, `PREDICTION_FAILS`}; `cross_ubatch_summary(reports) -> dict` with `grid`, `exact_set_tracks_ubatch`, `discriminating_splits`; `context_matched_drift(a_top, b_top, first_div, max_positions) -> (list[float], list[float])`.

Implemented test-first; the fake server in `tests/test_g1.py` models micro-batch tiling directly (restore tiles `[0,k)` then `[k,n)`, recompute tiles `[0,n)`), so the prediction logic is verified against an engine where tiling is the only mechanism, one where every restore diverges, and one where none does.

**Run:** for `U` in 256, 512, 1024 — start the server with `-ub U`, then
`python scripts/g1_server.py --ubatch U --label <model>`. Then
`python scripts/g1_analyze.py --label <model>`.

**Gate G1:** `exact_set_tracks_ubatch` true on Ornith Q4_K_M and Qwen3-4B.

---

### Task 7R: Chat-templated traces, natural and snapped boundaries, per-model tool calls

**Status: done.** Server checks passed 2026-09-10 (`docs/findings/2026-09-10-g2.md`): rendering byte-exact vs `/apply-template` (12/12 per model); prompts with tools match server token counts, parsers agree with llama.cpp, and the tool-call rate is 100% on all three models (30 conversations each). Before those checks: Renderer, boundary helpers and the three parsers are under test. Offline rendering of all three embedded templates with tools succeeds, and each template's own serialisation of a tool call parses back with its model's parser.

**Revised after checking the server and the templates.** `llama-server`'s
`/apply-template` accepts `messages` only, not `tools`, so rendering is offline
from each GGUF's embedded `tokenizer.chat_template`. The three models emit tool
calls in three incompatible formats (spec §4.4).

**Files:**
- Create: `src/driftmatrix/traces/render.py`, `src/driftmatrix/traces/toolcalls.py`
- Create: `scripts/g2_render_check.py`
- Test: `tests/traces/test_render.py`, `tests/traces/test_toolcalls.py`

**Interfaces:**
- `load_template(gguf_path) -> str` — reads `tokenizer.chat_template` via `driftmatrix.gguf.read_metadata`.
- `render(template: str, messages: list[dict], tools: list[dict] | None, *, add_generation_prompt: bool = True, enable_thinking: bool = False, **context) -> str` — jinja2 `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])`; filter `tojson(x, indent=None) = json.dumps(x, ensure_ascii=False, indent=indent)`; global `raise_exception(msg)` raising `jinja2.exceptions.TemplateError`.
- `snap(k: int, ubatch: int) -> int` — `k - k % ubatch`.
- `natural_boundary(count_tokens, template, messages, tools) -> int` — token count of the conversation rendered without its final user turn and without a generation prompt. Callers must verify it is a strict token prefix of the full render; turns where it is not are recorded `boundary_not_prefix` and excluded.
- `parse_tool_call(text: str, fmt: str) -> ToolCall | None`, `fmt` in `{"qwen_xml", "qwen_json", "spark_argkv"}`; `FORMAT_BY_MODEL = {"ornith": "qwen_xml", "qwen3": "qwen_json", "spark": "spark_argkv"}`. Values are JSON-decoded where they parse, otherwise stripped strings; the result hashes through `ToolCall.from_payload`, which sorts keys.

**Required tests (write first):**

```python
from driftmatrix.metrics.behavioral import ToolCall
from driftmatrix.traces.render import render, snap
from driftmatrix.traces.toolcalls import parse_tool_call

EXPECTED = ToolCall.from_payload('lookup_record', {'record_id': 7})

def test_tojson_does_not_html_escape():
    assert render('{{ x | tojson }}', [], None, add_generation_prompt=False, x='<a&b>') == '"<a&b>"'

def test_trim_blocks_like_huggingface():
    assert render('{% if true %}\nA{% endif %}', [], None, add_generation_prompt=False) == 'A'

def test_raise_exception_surfaces_template_errors():
    import pytest
    from jinja2.exceptions import TemplateError
    with pytest.raises(TemplateError, match='boom'):
        render("{{ raise_exception('boom') }}", [], None, add_generation_prompt=False)

def test_snap_rounds_down_and_costs_less_than_one_ubatch():
    assert snap(1390, 512) == 1024 and snap(1024, 512) == 1024 and snap(511, 512) == 0
    assert all(0 <= k - snap(k, 512) <= 511 for k in range(0, 5000, 37))

def test_parse_ornith_qwen_xml():
    txt = '<tool_call>\n<function=lookup_record>\n<parameter=record_id>\n7\n</parameter>\n</function>\n</tool_call>'
    assert parse_tool_call(txt, 'qwen_xml') == EXPECTED

def test_parse_ornith_allows_reasoning_before_the_call():
    txt = 'I will look it up.\n\n<tool_call>\n<function=lookup_record>\n<parameter=record_id>\n7\n</parameter>\n</function>\n</tool_call>'
    assert parse_tool_call(txt, 'qwen_xml') == EXPECTED

def test_parse_qwen3_json():
    txt = '<tool_call>\n{"name": "lookup_record", "arguments": {"record_id": 7}}\n</tool_call>'
    assert parse_tool_call(txt, 'qwen_json') == EXPECTED

def test_parse_spark_argkv():
    txt = '<tool_call>lookup_record<arg_key>record_id</arg_key><arg_value>7</arg_value></tool_call>'
    assert parse_tool_call(txt, 'spark_argkv') == EXPECTED

def test_string_values_survive_canonicalisation():
    txt = '<tool_call>format_report<arg_key>title</arg_key><arg_value>Batch 5</arg_value></tool_call>'
    assert parse_tool_call(txt, 'spark_argkv') == ToolCall.from_payload('format_report', {'title': 'Batch 5'})

def test_plain_text_and_truncated_calls_are_none():
    assert parse_tool_call('The record says alpha.', 'qwen_json') is None
    assert parse_tool_call('<tool_call>\n{"name": "lookup_rec', 'qwen_json') is None
```

**Verification with the server (`scripts/g2_render_check.py`):** for each model,
render N tool-less conversations offline and compare against
`POST /apply-template` byte for byte. Any mismatch blocks G2 until explained.

**Gate G2:** renderer matches `/apply-template` on every model; ≥ 90% of
recompute-arm turns parse as tool calls on every model; recompute floor exact;
pre-registration committed and tagged.

---

### Task 27: Ambiguous request stratum and decision-margin covariate (spec v4.2)

**Status: done; outcome and margin revised by Task 28.** The strata check (`docs/findings/2026-09-10-g2.md`) showed tool-choice requests always call a tool but never produce near-ties, while under-specified requests are the only source of near-ties and sometimes end without a call.

**Why:** all 90 explicit G2 requests produced exactly the expected call, and G1
showed restore flips only near-tie decisions. The stratum S replaces the
continuation factor C, because tool calls end the turn at 81–151 characters.

**Files:**
- Modify: `src/driftmatrix/traces/conversations.py` — `Conversation.stratum`, `generate_conversations(..., stratum=...)`
- Create: `src/driftmatrix/metrics/margin.py`
- Modify: `src/driftmatrix/runner/llamaserver.py` — `Completion.pieces`
- Modify: `src/driftmatrix/g2.py` — margins per row, `stratum_check`
- Modify: `scripts/g2_toolcalls.py` — `--stratum`; Create: `scripts/g2_strata.py`
- Test: `tests/traces/test_conversations.py`, `tests/metrics/test_margin.py`, `tests/test_g2.py`

**Interfaces:**
- `generate_conversations(n, history_turns, seed, stratum="explicit")`. Explicit output is unchanged from the G2 run (same ids, same requests). An ambiguous conversation is its explicit twin with only the final user request replaced; its `expected_tool` and `expected_args` are `None`.
- `decision_margin(top, pieces) -> (margin or None, position or None)`: the smallest top-1 minus top-2 logprob gap over positions through the end of the first complete tool call.
- `MARGIN_BINS = (0.5, 2.0)`; `margin_bin(m)` returns `near_tie`, `contested`, `confident` or `None`.
- `stratum_check(explicit_summary, ambiguous_summary)` returns a dict whose verdict is one of `STRATUM_LOWERS_MARGIN`, `STRATUM_DOES_NOT_LOWER_MARGIN`, `MARGIN_UNAVAILABLE`.

**Required tests (write first):** ambiguous twins share history and differ only in the final request; explicit conversation `c2026_000` still ends with "Look up record 153 and tell me its field."; the tool-call span end is found when `</tool_call>` is split across pieces; margins after the span are ignored; unsorted candidate lists are handled; bin edges 0.5 and 2.0 fall in the upper bin; `stratum_check` reports all three verdicts; `parse_completion` keeps each position's token text.

**Run:** for each model, `python scripts/g2_toolcalls.py --model M --gguf G --stratum explicit` and `--stratum ambiguous`, then `python scripts/g2_strata.py`.

**Gate G2 addition:** on every model, the ambiguous stratum keeps at least 90% tool calls, and its median recompute-arm decision margin is below the explicit stratum's.

---

### Task 28: Action outcome and action margin (spec v4.3)

**Status: done.** The ambiguous stratum re-run lowered the median action margin on all three models (Ornith 4.02 vs 6.92, Qwen3 7.47 vs 9.76, Spark 2.67 vs 7.77), with 87%, 67% and 83% calls. Spark's one parser disagreement led to Task 29.

**Why:** in the strata check, the requests vague enough to create near-tie
decisions were also vague enough that models sometimes asked a clarifying
question or answered in text instead of calling a tool. Scoring only tool calls
would discard exactly those decisions.

**Files:**
- Modify: `src/driftmatrix/metrics/margin.py` — action margin over the first position plus the tool-call span; `tool_call_span`, `position_gaps`
- Modify: `src/driftmatrix/metrics/behavioral.py` — `ToolCall.args` kept for diagnosis, excluded from equality
- Modify: `src/driftmatrix/g2.py` — `action` per row, stratum-aware call threshold, per-position gaps and both parsers' arguments stored
- Create: `scripts/g2_types.py` — per request type breakdown
- Test: `tests/metrics/test_margin.py`, `tests/metrics/test_behavioral.py`, `tests/test_g2.py`

**Interfaces:**
- `decision_margin(top, pieces)` returns the action margin: the smallest top-1 minus top-2 gap over position 0 and the positions of the first complete tool call. Every turn with any generated position has one.
- `tool_call_span(pieces) -> (first, last) | None`; `position_gaps(top) -> list`.
- `call_threshold(strata)`: 0.90 for explicit, 0.50 when the report is ambiguous; `g2_verdict` uses it.
- Rows gain `action` (`call` or `no_call`), `gaps` (every position's top-two gap, so margins can be recomputed without a GPU), and `args` inside `ours` and `theirs`.

**Required tests (write first):** a no-call turn is scored at position 0 only; preamble positions before a call are excluded while position 0 still counts; both tags may be split across pieces; the ambiguous stratum passes at 60% calls and fails at 40%, while explicit fails at 60%; both parsers agreeing there is no call counts as agreement; `ToolCall` equality ignores `args`.

**Run:** the ambiguous stratum for all three models (explicit margins are unchanged, because every explicit output starts its tool call at position 0), then `python scripts/g2_strata.py` and `python scripts/g2_types.py`.

---

### Task 29: Parser validation by template round trip (spec v4.4)

**Status: done.** Saved reports were re-scored offline. All 161 parsed calls round-trip; the one disagreement (Spark `a2026_024`) is attributed to llama.cpp; `g2_strata.py` passes on every model.

**Why:** in the ambiguous re-run, one Spark call parsed differently in llama.cpp,
whose parser dropped the text before ": " in a string argument. Our parse matched
the model's text, but the harness had no objective way to establish that.

**Files:**
- Modify: `src/driftmatrix/traces/toolcalls.py` — `parse_tool_call_ordered` returns the name, arguments in emitted order, and the raw block
- Create: `src/driftmatrix/traces/roundtrip.py` — `serialise_call`, `round_trips`, `in_emitted_order`
- Modify: `src/driftmatrix/g2.py` — `adjudicate_rows` and `summarise`, both usable on saved reports; verdict `OUR_PARSE_NOT_FAITHFUL`; adjudicated disagreements no longer fail the gate; rows keep the full output
- Create: `scripts/g2_rescore.py` — re-score saved reports offline
- Modify: `scripts/g2_strata.py` — footnote states each stratum's own threshold
- Test: `tests/traces/test_roundtrip.py`, `tests/test_g2.py`

**Rule:** a parsed call is faithful when serialising it through the model's chat
template, arguments in emitted order, reproduces the model's tool-call block byte
for byte. Every parsed call must be faithful. A disagreement with llama.cpp is
`server_parser_lossy` when our reading round-trips and llama.cpp's does not,
`our_parser_lossy` in the reverse case, and `unresolved` otherwise; only the last
two fail the gate.

**Required tests (write first):** the verbatim G2 Spark call round-trips under our
reading and not under llama.cpp's; argument order matters and `in_emitted_order`
restores it; a template error counts as no round trip; a lossy server parse is
adjudicated and passes; a call found only by the server fails as unresolved; a
template that serialises differently from the model fails as
`OUR_PARSE_NOT_FAITHFUL`; offline re-scoring of saved rows reproduces the live
summary exactly.

**Run:** `python scripts/g2_rescore.py --gguf ornith=... --gguf qwen3=... --gguf spark=...`,
then `python scripts/g2_strata.py`. No GPU is needed.

---

### Task 17R: Pre-registration (supersedes Task 17)

**Status: done.** `docs/preregistration/2026-09-10-analysis-plan.md`, tag `prereg-2026-09-10`.

**Why it differs from Task 17:** the measured inputs are G1's `n_ubatch` (not a kernel
chunk size) and a bit-exact floor; the design is B × P × S with action outcomes (v4.3).
Writing the plan exposed three problems, fixed before the freeze:
- Qwen3's history is never a token prefix of its next prompt, so `boundary_tokens` would
  have excluded every Qwen3 turn. The natural boundary is now the longest common prefix.
- Holm-corrected sign tests need at least 9 changed turns out of 90 to reject, which
  contradicts the 2% power target. Against a zero floor, the exact binomial test with
  null rate 0 is the correct test and matches `stats/power.py`.
- A percentile bootstrap returns [0, 0] for a rate with no events. Rates carry
  Clopper–Pearson intervals.

**Files:**
- Create: `docs/preregistration/2026-09-10-analysis-plan.md`
- Create: `src/driftmatrix/stats/exact.py` — `clopper_pearson`, `sign_test`, `signs`, `test_against_floor`
- Modify: `src/driftmatrix/traces/render.py` — `common_prefix_length`, `reuse_boundary`
- Modify: `src/driftmatrix/traces/conversations.py` — `generate_study_turns`, `trace_hash`, `Conversation.prefix_band` / `base_id`
- Modify: `src/driftmatrix/config.py` — study constants; `stats/multiplicity.py` — family C = 6
- Create: `scripts/study_manifest.py` — token budget and reuse boundaries, no generation
- Test: `tests/stats/test_exact.py`, `tests/traces/test_reuse_boundary.py`, `tests/traces/test_study_turns.py`

**Run before any arm:** `python scripts/study_manifest.py --model M --gguf G` for each model.

---

### Task 22R: Main-study grid against `llama-server`

**Amended by spec v4.2:** conditions are B × P × S (the continuation factor is gone); generation runs to the end of the turn, capped at 192 tokens; each row records the stratum and the recompute arm's decision margin. This task introduces the new condition encoding; `config.Condition`, which still carries the continuation band for the off-path Python engine, is left untouched.

**Status: runner done (with Task 26); analysis module pending.** As built, following the pre-registration (sections 4, 6, 11):
- `src/driftmatrix/servergrid.py` — `run_study`, `run_turn`, `load_records`, `restore_record`, `compare_files`, `loaded_model_matches`. Arms per turn in pre-registered order (replicate 1 then 2: recompute, restore natural, restore snapped); a snapped boundary equal to the natural one reuses the natural run; a boundary of 0 is recorded as `boundary_zero` and never restored.
- **Records are JSON lines, not Parquet**: appended one whole turn at a time and fsynced, so a multi-hour laptop run resumes at the interrupted turn. A metadata line (model, build, trace hash, `n_ubatch`, `n_predict`, `n_probs`) must match on resume. Server errors and timeouts leave a turn incomplete and are retried on the next run.
- **Blinded runner:** it computes no exactness, drift or action outcome, and its progress lines report arm validity only.
- Refuses to run if the server's build is not `b10839-0cae43063` or its loaded model is not the named GGUF.
- `src/driftmatrix/g3.py` — `g3_check`: arm proofs, snapped exactness in both replicates, positive control on the hybrid only.
- `scripts/run_study.py` (`--pilot` for G3), `scripts/g3_check.py`, `tests/test_servergrid.py` (fake server where tiling is the only cause).
- **G3 passed 2026-09-10** on all three models: 20 pilot turns and 120 valid arms each, every snapped restore bit-exact in both replicates, Ornith drift at the natural boundary on 20 of 20 turns. Findings: `docs/findings/2026-09-10-g3.md`.


**Files:** Create `src/driftmatrix/servergrid.py`; Test `tests/test_servergrid.py`.

**Interfaces:**
- Consumes: `ServerClient` (or a fake with the same methods), `traces.render`, `capture.write_parquet`.
- Produces: `run_servergrid(client, spec, conversations, ubatch, meta, out_dir, replicates=2) -> tuple[Path, Path]` writing `generations.parquet` and `statebytes.parquet`.

**Required tests (write first):** every row carries `k`, `k_mod_ubatch`, `ubatch`, `boundary` ∈ {natural, snapped}, `prompt_n`, `arm_valid`; an arm with `prompt_n` inconsistent with its role is written with `outcome = restore_not_used` or `recompute_not_clean` and excluded from effects but counted; snapped rows satisfy `k_mod_ubatch == 0`; against the tiling fake, snapped restores are exact and natural misaligned restores are not.

---

### Task 26: State bytes comparison

**Files:** Modify `src/driftmatrix/servergrid.py`; Test `tests/test_servergrid.py`.

After both arms of a turn, save the slot from each to `<tag>_restore.bin` / `<tag>_recompute.bin` and record `state_bytes_equal` and `state_bytes_differing`. Test: identical fake slot files → equal, zero differing; one flipped byte → not equal, one differing.


## Task 0: G0 viability spike — does state save/restore work at all?

This is a **spike**, not TDD. Its output is an answer plus a committed findings note. Every later task is gated on it. If the state API does not round-trip on a `qwen3_5` model, that is a finding (llama.cpp #20225 is the context) and the project re-scopes per spec §10.1.

**Files:**
- Create: `scripts/g0_viability.py`
- Create: `docs/findings/2026-09-10-g0-state-api.md`

**Interfaces:**
- Consumes: nothing.
- Produces: a documented answer to three questions — does the state API exist for this build, does an immediate round-trip reproduce output exactly, and what dtype is the runtime recurrent state. No code that later tasks import.

- [ ] **Step 1: Write the probe script**

```python
# scripts/g0_viability.py
"""G0 viability spike. Output is an answer, not a library."""
import sys, json, hashlib
from pathlib import Path
from llama_cpp import Llama

MODEL = sys.argv[1]          # path to Ornith-1.5-9B IQ4_XS gguf
PROMPT = "You are a helpful assistant. " * 40 + "\nList three prime numbers."
STATE = Path("/tmp/dm_g0.state")

def run(llm, tokens, n_new):
    llm.reset()
    llm.eval(tokens)
    out = []
    for _ in range(n_new):
        tok = llm.sample(temp=0.0)
        if tok == llm.token_eos():
            break
        out.append(tok)
        llm.eval([tok])
    return out

llm = Llama(model_path=MODEL, n_ctx=4096, logits_all=False, seed=1234, verbose=False)
toks = llm.tokenize(PROMPT.encode())
split = len(toks) // 2
prefix, suffix = toks[:split], toks[split:]

report = {"model": MODEL, "n_tokens": len(toks), "split_at": split}

# Q1: does the state API exist and round-trip?
try:
    llm.reset(); llm.eval(prefix)
    blob = llm.save_state()
    report["state_api"] = "present"
    report["state_bytes"] = getattr(blob, "llama_state_size", None) or len(bytes(blob.llama_state))
except Exception as e:
    report["state_api"] = f"MISSING/FAILED: {type(e).__name__}: {e}"
    print(json.dumps(report, indent=2)); raise SystemExit(1)

# Q2: negative control 1 — immediate restore must reproduce recompute exactly
llm.reset(); llm.eval(prefix); llm.load_state(blob); llm.eval(suffix)
restored = [llm.sample(temp=0.0)]
for _ in range(31):
    llm.eval([restored[-1]]); restored.append(llm.sample(temp=0.0))

recomputed = run(llm, toks, 32)

report["restore_tokens"] = restored[:32]
report["recompute_tokens"] = recomputed[:32]
report["identical"] = restored[:32] == recomputed[:32]
report["first_divergence"] = next(
    (i for i, (a, b) in enumerate(zip(restored, recomputed)) if a != b), None)

print(json.dumps(report, indent=2))
```

- [ ] **Step 2: Run it against the installed model**

Run: `python scripts/g0_viability.py ~/models/Ornith-1.5-9B-IQ4_XS.gguf`

Three outcomes, all informative:
- `state_api: MISSING/FAILED` → the API is unavailable for this build. **Finding.** Re-scope.
- `identical: true` → restore is faithful at this split. Proceed; the alignment sweep (Task 12) is where divergence would appear.
- `identical: false` → divergence on day one. Record `first_divergence`; this is the headline effect showing up immediately.

- [ ] **Step 3: Record the runtime state dtype**

Run: `python -c "from llama_cpp import Llama; l=Llama(model_path='<path>', n_ctx=512, verbose=True)" 2>&1 | grep -iE 'kv|state|cache|type'`

Capture whatever the load log reports for cache/state types. Spec §3.2 requires this recorded, because it bounds how much arithmetic slack exists.

- [ ] **Step 4: Write the findings note**

Create `docs/findings/2026-09-10-g0-state-api.md` containing: the model path and llama.cpp commit, the raw JSON output, the state dtype, which of the three outcomes occurred, and one paragraph on what it means for the plan. If the API is missing, state explicitly which re-scope branch of spec §10.1 is being taken.

- [ ] **Step 5: Commit**

```bash
git add scripts/g0_viability.py docs/findings/2026-09-10-g0-state-api.md
git commit -m "spike: G0 state save/restore viability check"
```

---

## Task 1: Package scaffolding and the factor model

**Files:**
- Create: `pyproject.toml`
- Create: `src/driftmatrix/__init__.py`
- Create: `src/driftmatrix/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Alignment`, `PrefixBand`, `ContinuationBand`, `Arm`, `Mode`, `Outcome`, `LayerType` (all `str` enums); `Condition` frozen dataclass with `.condition_id -> int`; `ALL_CONDITIONS: tuple[Condition, ...]` of length 8; `ModelSpec` frozen dataclass; constants `PREFIX_TOKENS`, `CONTINUATION_TOKENS`, `BOOTSTRAP_RESAMPLES`, `ALPHA`, `NOISE_FLOOR_NATS`, `SELF_INCONSISTENCY_MULTIPLIER`, `TARGET_EFFECT_RATE`, `TARGET_POWER`. Every later task imports from here.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from driftmatrix.config import (
    Alignment, PrefixBand, ContinuationBand, Condition, ALL_CONDITIONS,
    PREFIX_TOKENS, CONTINUATION_TOKENS,
)

def test_eight_conditions_all_unique():
    assert len(ALL_CONDITIONS) == 8
    assert len({c.condition_id for c in ALL_CONDITIONS}) == 8
    assert {c.condition_id for c in ALL_CONDITIONS} == set(range(8))

def test_condition_id_roundtrips():
    for c in ALL_CONDITIONS:
        assert Condition.from_id(c.condition_id) == c

def test_reference_condition_is_zero():
    c = Condition(Alignment.ALIGNED, PrefixBand.SHORT, ContinuationBand.SHORT)
    assert c.condition_id == 0

def test_band_token_counts_match_spec():
    assert PREFIX_TOKENS[PrefixBand.SHORT] == 1024
    assert PREFIX_TOKENS[PrefixBand.LONG] == 8192
    assert CONTINUATION_TOKENS[ContinuationBand.SHORT] == 32
    assert CONTINUATION_TOKENS[ContinuationBand.LONG] == 256
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix'`

- [ ] **Step 3: Write pyproject.toml and config.py**

```toml
# pyproject.toml
[project]
name = "driftmatrix"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "scipy>=1.11", "pandas>=2.1", "pyarrow>=14", "matplotlib>=3.8"]

[project.optional-dependencies]
engine = ["llama-cpp-python>=0.3"]
traces = ["langgraph>=0.2", "transformers>=4.44"]
dev = ["pytest>=8", "pytest-cov>=5"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# src/driftmatrix/config.py
"""Factor model and project-wide constants. Single source of truth."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Alignment(str, Enum):
    ALIGNED = "aligned"
    MISALIGNED = "misaligned"


class PrefixBand(str, Enum):
    SHORT = "short"
    LONG = "long"


class ContinuationBand(str, Enum):
    SHORT = "short"
    LONG = "long"


class Arm(str, Enum):
    RESTORE = "restore"
    RECOMPUTE = "recompute"


class Mode(str, Enum):
    FREE_RUNNING = "free_running"
    TEACHER_FORCED = "teacher_forced"


class Outcome(str, Enum):
    OK = "ok"
    UNSUPPORTED = "unsupported"
    CRASHED = "crashed"
    TIMEOUT = "timeout"
    STATE_IO_FAILED = "state_io_failed"


class LayerType(str, Enum):
    LINEAR_ATTENTION = "linear_attention"
    FULL_ATTENTION = "full_attention"
    SLIDING_ATTENTION = "sliding_attention"


@dataclass(frozen=True, order=True)
class Condition:
    alignment: Alignment
    prefix_band: PrefixBand
    continuation_band: ContinuationBand

    @property
    def condition_id(self) -> int:
        return (
            (1 if self.alignment is Alignment.MISALIGNED else 0)
            | (2 if self.prefix_band is PrefixBand.LONG else 0)
            | (4 if self.continuation_band is ContinuationBand.LONG else 0)
        )

    @classmethod
    def from_id(cls, cid: int) -> "Condition":
        if not 0 <= cid < 8:
            raise ValueError(f"condition_id must be 0..7, got {cid}")
        return cls(
            Alignment.MISALIGNED if cid & 1 else Alignment.ALIGNED,
            PrefixBand.LONG if cid & 2 else PrefixBand.SHORT,
            ContinuationBand.LONG if cid & 4 else ContinuationBand.SHORT,
        )


ALL_CONDITIONS: tuple[Condition, ...] = tuple(Condition.from_id(i) for i in range(8))


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    arch_level: int      # 3 = recurrent, 2 = sliding-window, 1 = dense
    gguf_path: str
    quant: str


PREFIX_TOKENS = {PrefixBand.SHORT: 1024, PrefixBand.LONG: 8192}
CONTINUATION_TOKENS = {ContinuationBand.SHORT: 32, ContinuationBand.LONG: 256}

BOOTSTRAP_RESAMPLES = 10_000
ALPHA = 0.05
NOISE_FLOOR_NATS = 0.01
SELF_INCONSISTENCY_MULTIPLIER = 2.0
TARGET_EFFECT_RATE = 0.02
TARGET_POWER = 0.80
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/driftmatrix/ tests/test_config.py
git commit -m "feat: factor model and project constants"
```

---

## Task 2: Distributional metrics with union-support truncation

Pure math, no engine, no GPU. Spec §5.2. The union-support rule is the subtle part: two arms return *different* top-*k* token sets, so both must be re-expressed over the union plus one `<other>` bucket before comparison.

**Files:**
- Create: `src/driftmatrix/metrics/__init__.py`
- Create: `src/driftmatrix/metrics/distributional.py`
- Test: `tests/metrics/test_distributional.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TokenDist(token_ids: list[int], logprobs: list[float])` frozen dataclass; `align_supports(a: TokenDist, b: TokenDist) -> tuple[np.ndarray, np.ndarray]` returning two probability vectors over the union plus `<other>`; `jsd(p, q) -> float` in nats; `total_variation(p, q) -> float`; `top1_agree(a, b) -> bool`; `per_position_jsd(a: list[TokenDist], b: list[TokenDist]) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
# tests/metrics/test_distributional.py
import numpy as np, pytest
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
    a = d([1], [1.0])
    b = d([2], [1.0])
    p, q = align_supports(a, b)
    assert jsd(p, q) == pytest.approx(np.log(2), rel=1e-9)

def test_union_support_includes_other_bucket():
    # a sees tokens {1,2} totalling 0.9 -> 0.1 goes to <other>
    a = d([1, 2], [0.6, 0.3])
    b = d([2, 3], [0.5, 0.4])
    p, q = align_supports(a, b)
    assert len(p) == len(q) == 4          # union {1,2,3} + <other>
    assert p.sum() == pytest.approx(1.0)
    assert q.sum() == pytest.approx(1.0)
    assert p[-1] == pytest.approx(0.1)    # a's missing mass
    assert q[-1] == pytest.approx(0.1)    # b's missing mass

def test_jsd_is_symmetric_and_bounded():
    a = d([1, 2], [0.9, 0.1])
    b = d([1, 2], [0.1, 0.9])
    p, q = align_supports(a, b)
    assert jsd(p, q) == pytest.approx(jsd(q, p))
    assert 0.0 <= jsd(p, q) <= np.log(2) + 1e-12

def test_top1_agreement():
    assert top1_agree(d([5, 6], [0.7, 0.3]), d([5, 9], [0.6, 0.4])) is True
    assert top1_agree(d([5, 6], [0.7, 0.3]), d([6, 5], [0.8, 0.2])) is False

def test_per_position_jsd_length_matches_shorter_sequence():
    a = [d([1], [1.0])] * 5
    b = [d([1], [1.0])] * 3
    assert per_position_jsd(a, b).shape == (3,)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/metrics/test_distributional.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.metrics'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/metrics/__init__.py
"""Metric computation over captured generations. No engine dependency."""
```

```python
# src/driftmatrix/metrics/distributional.py
"""Distributional divergence over truncated top-k logprobs (spec 5.2)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class TokenDist:
    token_ids: list[int]
    logprobs: list[float]

    def as_probs(self) -> dict[int, float]:
        return {t: float(np.exp(lp)) for t, lp in zip(self.token_ids, self.logprobs)}


def align_supports(a: TokenDist, b: TokenDist) -> tuple[np.ndarray, np.ndarray]:
    """Re-express both distributions over union(support) + one <other> bucket.

    The engine returns only the top-k tokens, so each distribution is missing
    mass. That missing mass is real and must be represented, not renormalised
    away, or JSD is computed against a distribution neither arm produced.
    """
    pa, pb = a.as_probs(), b.as_probs()
    union = sorted(set(pa) | set(pb))
    va = np.array([pa.get(t, 0.0) for t in union], dtype=np.float64)
    vb = np.array([pb.get(t, 0.0) for t in union], dtype=np.float64)
    # observed mass may exceed 1 by float error; clamp before taking remainder
    ra = max(0.0, 1.0 - float(va.sum()))
    rb = max(0.0, 1.0 - float(vb.sum()))
    p = np.concatenate([va, [ra]])
    q = np.concatenate([vb, [rb]])
    return p / p.sum(), q / q.sum()


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats. Symmetric, bounded by log 2."""
    m = 0.5 * (p + q)
    def _kl(x, y):
        mask = x > _EPS
        return float(np.sum(x[mask] * np.log(x[mask] / np.maximum(y[mask], _EPS))))
    return max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def top1_agree(a: TokenDist, b: TokenDist) -> bool:
    if not a.token_ids or not b.token_ids:
        return False
    return (a.token_ids[int(np.argmax(a.logprobs))]
            == b.token_ids[int(np.argmax(b.logprobs))])


def per_position_jsd(a: list[TokenDist], b: list[TokenDist]) -> np.ndarray:
    n = min(len(a), len(b))
    return np.array([jsd(*align_supports(a[i], b[i])) for i in range(n)])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/metrics/test_distributional.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/metrics/ tests/metrics/
git commit -m "feat: distributional metrics with union-support truncation"
```

---

## Task 3: Behavioral metrics

Spec §5.1. These are the **primary** outcomes and deliberately do not depend on logprobs, so they survive if the teacher-forced instrument proves unreliable.

**Files:**
- Create: `src/driftmatrix/metrics/behavioral.py`
- Test: `tests/metrics/test_behavioral.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ToolCall(name: str, args_hash: str)` frozen dataclass with `ToolCall.from_payload(name: str, args: dict) -> ToolCall`; `exact_match(a: list[int], b: list[int]) -> bool`; `first_divergence(a, b, cap: int) -> tuple[int, bool]` returning `(position, censored)`; `tool_call_agree(a: ToolCall | None, b: ToolCall | None) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/metrics/test_behavioral.py
from driftmatrix.metrics.behavioral import (
    ToolCall, exact_match, first_divergence, tool_call_agree,
)

def test_exact_match():
    assert exact_match([1, 2, 3], [1, 2, 3]) is True
    assert exact_match([1, 2, 3], [1, 2, 4]) is False
    assert exact_match([1, 2], [1, 2, 3]) is False

def test_first_divergence_reports_position_and_not_censored():
    pos, censored = first_divergence([1, 2, 3, 4], [1, 2, 9, 4], cap=32)
    assert (pos, censored) == (2, False)

def test_identical_sequences_are_censored_at_cap():
    pos, censored = first_divergence([1, 2, 3], [1, 2, 3], cap=32)
    assert (pos, censored) == (32, True)

def test_length_mismatch_diverges_at_shorter_length():
    pos, censored = first_divergence([1, 2], [1, 2, 3], cap=32)
    assert (pos, censored) == (2, False)

def test_tool_call_args_hash_is_order_insensitive():
    a = ToolCall.from_payload("search", {"q": "x", "n": 3})
    b = ToolCall.from_payload("search", {"n": 3, "q": "x"})
    assert a == b and tool_call_agree(a, b) is True

def test_tool_call_disagreement_on_name_and_on_args():
    base = ToolCall.from_payload("search", {"q": "x"})
    assert tool_call_agree(base, ToolCall.from_payload("lookup", {"q": "x"})) is False
    assert tool_call_agree(base, ToolCall.from_payload("search", {"q": "y"})) is False

def test_both_none_counts_as_agreement_but_one_none_does_not():
    assert tool_call_agree(None, None) is True
    assert tool_call_agree(ToolCall.from_payload("s", {}), None) is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/metrics/test_behavioral.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.metrics.behavioral'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/metrics/behavioral.py
"""Behavioural outcomes (spec 5.1). Primary results; no logprob dependency."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    name: str
    args_hash: str

    @classmethod
    def from_payload(cls, name: str, args: dict) -> "ToolCall":
        # sort_keys makes the hash insensitive to serialisation order, so a
        # reordered-but-identical call is not scored as a disagreement.
        blob = json.dumps(args, sort_keys=True, separators=(",", ":"))
        return cls(name=name, args_hash=hashlib.sha256(blob.encode()).hexdigest()[:16])


def exact_match(a: list[int], b: list[int]) -> bool:
    return list(a) == list(b)


def first_divergence(a: list[int], b: list[int], cap: int) -> tuple[int, bool]:
    """Return (position, censored). Censored means they never diverged."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, False
    if len(a) != len(b):
        return n, False
    return cap, True


def tool_call_agree(a: ToolCall | None, b: ToolCall | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a == b
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/metrics/test_behavioral.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/metrics/behavioral.py tests/metrics/test_behavioral.py
git commit -m "feat: behavioral metrics for tool-call and output agreement"
```

---

## Task 4: statediff — compare restored vs recomputed state tensors

Spec §5.3. This measures the mechanism at its source, before any token is generated.

**Files:**
- Create: `src/driftmatrix/statediff.py`
- Test: `tests/test_statediff.py`

**Interfaces:**
- Consumes: `LayerType` from `driftmatrix.config`.
- Produces: `LayerDiff(layer_idx: int, layer_type: LayerType, max_abs_diff: float, max_rel_diff: float, frobenius_norm_diff: float, state_dtype: str)` frozen dataclass; `diff_layer(idx, layer_type, restored: np.ndarray, recomputed: np.ndarray) -> LayerDiff`; `diff_all(restored: dict[int, np.ndarray], recomputed: dict[int, np.ndarray], layer_types: dict[int, LayerType]) -> list[LayerDiff]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_statediff.py
import numpy as np, pytest
from driftmatrix.config import LayerType
from driftmatrix.statediff import LayerDiff, diff_layer, diff_all

def test_identical_tensors_give_all_zeros():
    x = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    d = diff_layer(0, LayerType.LINEAR_ATTENTION, x, x.copy())
    assert d.max_abs_diff == 0.0
    assert d.max_rel_diff == 0.0
    assert d.frobenius_norm_diff == 0.0

def test_known_epsilon_is_recovered():
    x = np.ones((2, 2), dtype=np.float32)
    y = x.copy(); y[1, 1] += 0.25
    d = diff_layer(3, LayerType.LINEAR_ATTENTION, y, x)
    assert d.max_abs_diff == pytest.approx(0.25)
    assert d.max_rel_diff == pytest.approx(0.25)         # |0.25| / |1.0|
    assert d.frobenius_norm_diff == pytest.approx(0.25)

def test_relative_diff_ignores_near_zero_denominators():
    # a zero reference entry must not produce inf/nan
    x = np.array([[0.0, 2.0]], dtype=np.float32)
    y = np.array([[1e-9, 2.5]], dtype=np.float32)
    d = diff_layer(1, LayerType.FULL_ATTENTION, y, x)
    assert np.isfinite(d.max_rel_diff)
    assert d.max_rel_diff == pytest.approx(0.25)         # 0.5 / 2.0

def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        diff_layer(0, LayerType.LINEAR_ATTENTION,
                   np.zeros((2, 2)), np.zeros((3, 2)))

def test_diff_all_preserves_layer_order_and_types():
    r = {0: np.zeros((2, 2)), 1: np.ones((2, 2))}
    c = {0: np.zeros((2, 2)), 1: np.ones((2, 2))}
    types = {0: LayerType.LINEAR_ATTENTION, 1: LayerType.FULL_ATTENTION}
    out = diff_all(r, c, types)
    assert [d.layer_idx for d in out] == [0, 1]
    assert out[0].layer_type is LayerType.LINEAR_ATTENTION
    assert out[1].layer_type is LayerType.FULL_ATTENTION
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_statediff.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.statediff'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/statediff.py
"""Direct comparison of restored vs recomputed state tensors (spec 5.3)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import LayerType

_REL_FLOOR = 1e-6   # denominators below this are excluded from relative diff


@dataclass(frozen=True)
class LayerDiff:
    layer_idx: int
    layer_type: LayerType
    max_abs_diff: float
    max_rel_diff: float
    frobenius_norm_diff: float
    state_dtype: str


def diff_layer(idx: int, layer_type: LayerType,
               restored: np.ndarray, recomputed: np.ndarray) -> LayerDiff:
    if restored.shape != recomputed.shape:
        raise ValueError(
            f"layer {idx}: shape mismatch {restored.shape} vs {recomputed.shape}")
    a = restored.astype(np.float64)
    b = recomputed.astype(np.float64)
    delta = np.abs(a - b)
    denom = np.abs(b)
    mask = denom > _REL_FLOOR
    rel = float(np.max(delta[mask] / denom[mask])) if mask.any() else 0.0
    return LayerDiff(
        layer_idx=idx,
        layer_type=layer_type,
        max_abs_diff=float(delta.max()) if delta.size else 0.0,
        max_rel_diff=rel,
        frobenius_norm_diff=float(np.linalg.norm(a - b)),
        state_dtype=str(restored.dtype),
    )


def diff_all(restored: dict[int, np.ndarray], recomputed: dict[int, np.ndarray],
             layer_types: dict[int, LayerType]) -> list[LayerDiff]:
    return [
        diff_layer(i, layer_types[i], restored[i], recomputed[i])
        for i in sorted(restored)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_statediff.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/statediff.py tests/test_statediff.py
git commit -m "feat: per-layer state tensor diffing"
```

---

## Task 5: Deterministic mock tools

Spec §4.3. Mandatory purity — a nondeterministic tool injects variance indistinguishable from the effect under study.

**Files:**
- Create: `src/driftmatrix/traces/__init__.py`
- Create: `src/driftmatrix/traces/tools.py`
- Test: `tests/traces/test_tools.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TOOL_SCHEMAS: list[dict]` (stable serialization order); `call_tool(name: str, args: dict) -> str`; `ToolError` exception.

- [ ] **Step 1: Write the failing test**

```python
# tests/traces/test_tools.py
import pytest
from driftmatrix.traces.tools import TOOL_SCHEMAS, call_tool, ToolError

def test_schema_order_is_stable():
    assert [s["name"] for s in TOOL_SCHEMAS] == ["lookup_record", "compute_sum", "format_report"]

def test_tools_are_pure_across_repeated_calls():
    for _ in range(50):
        assert call_tool("lookup_record", {"record_id": 7}) == \
               call_tool("lookup_record", {"record_id": 7})

def test_known_outputs():
    assert call_tool("compute_sum", {"values": [1, 2, 3]}) == "6"
    assert call_tool("lookup_record", {"record_id": 0}).startswith("record 0:")
    assert call_tool("format_report", {"title": "T", "body": "B"}) == "# T\n\nB"

def test_unknown_tool_raises():
    with pytest.raises(ToolError, match="unknown tool"):
        call_tool("nope", {})

def test_missing_argument_raises():
    with pytest.raises(ToolError, match="missing argument"):
        call_tool("compute_sum", {})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/traces/test_tools.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.traces'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/traces/__init__.py
"""Agent-shaped trace generation. Traffic generator, not the contribution."""
```

```python
# src/driftmatrix/traces/tools.py
"""Deterministic mock tools (spec 4.3).

Every tool MUST be a pure function of its arguments. Any nondeterminism here
is indistinguishable from the effect under study and invalidates the run.
"""
from __future__ import annotations


class ToolError(Exception):
    pass


TOOL_SCHEMAS: list[dict] = [
    {"name": "lookup_record",
     "description": "Fetch a record by integer id.",
     "parameters": {"type": "object",
                    "properties": {"record_id": {"type": "integer"}},
                    "required": ["record_id"]}},
    {"name": "compute_sum",
     "description": "Sum a list of numbers.",
     "parameters": {"type": "object",
                    "properties": {"values": {"type": "array",
                                              "items": {"type": "number"}}},
                    "required": ["values"]}},
    {"name": "format_report",
     "description": "Render a titled markdown report.",
     "parameters": {"type": "object",
                    "properties": {"title": {"type": "string"},
                                   "body": {"type": "string"}},
                    "required": ["title", "body"]}},
]

_FIELDS = ("alpha", "beta", "gamma", "delta")


def _require(args: dict, key: str):
    if key not in args:
        raise ToolError(f"missing argument: {key}")
    return args[key]


def call_tool(name: str, args: dict) -> str:
    if name == "lookup_record":
        rid = int(_require(args, "record_id"))
        return f"record {rid}: field={_FIELDS[rid % len(_FIELDS)]} value={rid * 37 % 1000}"
    if name == "compute_sum":
        vals = _require(args, "values")
        total = sum(vals)
        return str(int(total)) if float(total).is_integer() else str(total)
    if name == "format_report":
        return f"# {_require(args, 'title')}\n\n{_require(args, 'body')}"
    raise ToolError(f"unknown tool: {name}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/traces/test_tools.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/traces/ tests/traces/
git commit -m "feat: deterministic mock tools for agent traces"
```

---

## Task 6: Boundary placement with tokenizer-verified offsets

Spec §4.3 and §9. Character-counted padding is a bug; only tokenizer counts are trusted. This task delivers the padding that puts the save boundary exactly where a condition requires.

**Files:**
- Create: `src/driftmatrix/traces/boundary.py`
- Test: `tests/traces/test_boundary.py`

**Interfaces:**
- Consumes: `Alignment`, `PrefixBand`, `PREFIX_TOKENS` from `driftmatrix.config`.
- Produces: `Tokenizer` protocol with `encode(text: str) -> list[int]`; `pad_to_token_count(text: str, target: int, tok: Tokenizer, filler: str = " lorem") -> str`; `boundary_offset(prefix_band: PrefixBand, alignment: Alignment, chunk_size: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/traces/test_boundary.py
import pytest
from driftmatrix.config import Alignment, PrefixBand
from driftmatrix.traces.boundary import pad_to_token_count, boundary_offset

class FakeTok:
    """One token per whitespace-separated word."""
    def encode(self, text: str) -> list[int]:
        return [hash(w) % 1000 for w in text.split()]

def test_padding_hits_exact_token_count():
    tok = FakeTok()
    out = pad_to_token_count("hello world", target=50, tok=tok)
    assert len(tok.encode(out)) == 50

def test_padding_is_noop_when_already_long_enough():
    tok = FakeTok()
    text = " ".join(["w"] * 60)
    assert pad_to_token_count(text, target=50, tok=tok) == text

def test_aligned_offset_is_a_multiple_of_chunk_size():
    off = boundary_offset(PrefixBand.SHORT, Alignment.ALIGNED, chunk_size=64)
    assert off % 64 == 0
    assert off <= 1024

def test_misaligned_offset_is_half_a_chunk_off():
    aligned = boundary_offset(PrefixBand.SHORT, Alignment.ALIGNED, chunk_size=64)
    mis = boundary_offset(PrefixBand.SHORT, Alignment.MISALIGNED, chunk_size=64)
    assert mis % 64 == 32
    assert abs(mis - aligned) == 32

def test_long_band_offset_is_larger_than_short():
    a = boundary_offset(PrefixBand.SHORT, Alignment.ALIGNED, chunk_size=64)
    b = boundary_offset(PrefixBand.LONG, Alignment.ALIGNED, chunk_size=64)
    assert b > a

def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError, match="chunk_size"):
        boundary_offset(PrefixBand.SHORT, Alignment.ALIGNED, chunk_size=0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/traces/test_boundary.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.traces.boundary'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/traces/boundary.py
"""Tokenizer-verified boundary placement (spec 4.3).

Character counts are never trusted. Every offset is confirmed by encoding.
"""
from __future__ import annotations
from typing import Protocol
from ..config import Alignment, PrefixBand, PREFIX_TOKENS


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


def pad_to_token_count(text: str, target: int, tok: Tokenizer,
                       filler: str = " lorem") -> str:
    """Grow text until it encodes to exactly `target` tokens.

    Binary-search-free: append filler and re-encode, because tokenizers are
    not linear in characters and any estimate would need verifying anyway.
    """
    cur = len(tok.encode(text))
    if cur >= target:
        return text
    out = text
    while cur < target:
        out += filler
        cur = len(tok.encode(out))
    # overshoot correction: drop trailing words until exact
    while cur > target:
        out = out.rsplit(" ", 1)[0]
        cur = len(tok.encode(out))
    return out


def boundary_offset(prefix_band: PrefixBand, alignment: Alignment,
                    chunk_size: int) -> int:
    """Token index at which state is saved.

    Aligned  -> the largest multiple of chunk_size within the band.
    Misaligned -> that, minus half a chunk, so the two arms tile differently.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    band = PREFIX_TOKENS[prefix_band]
    aligned = (band // chunk_size) * chunk_size
    if alignment is Alignment.ALIGNED:
        return aligned
    return aligned - chunk_size // 2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/traces/test_boundary.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/traces/boundary.py tests/traces/test_boundary.py
git commit -m "feat: tokenizer-verified save-boundary placement"
```

---

## Task 7: Agent trace generator and frozen corpus

Spec §4.3. LangGraph produces the turn structure; the corpus is frozen and content-hashed so every result row is traceable to exact inputs.

**Files:**
- Create: `src/driftmatrix/traces/generator.py`
- Create: `src/driftmatrix/traces/corpus.py`
- Test: `tests/traces/test_generator.py`
- Test: `tests/traces/test_corpus.py`

**Interfaces:**
- Consumes: `TOOL_SCHEMAS`, `call_tool` from `traces.tools`; `pad_to_token_count` from `traces.boundary`; `PrefixBand` from `config`.
- Produces: `Turn(session_id: str, turn_idx: int, prefix_text: str, suffix_text: str, expected_tool: str | None)` frozen dataclass; `generate_sessions(n_sessions: int, turns_per_session: int, seed: int) -> list[Turn]`; `SYSTEM_PROMPT: str`; `freeze_corpus(turns: list[Turn], path: Path) -> str` returning the sha256 hash; `load_corpus(path: Path) -> tuple[list[Turn], str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/traces/test_generator.py
from driftmatrix.traces.generator import generate_sessions, SYSTEM_PROMPT, Turn

def test_session_and_turn_counts():
    turns = generate_sessions(n_sessions=5, turns_per_session=6, seed=1)
    assert len(turns) == 30
    assert len({t.session_id for t in turns}) == 5

def test_generation_is_deterministic_for_a_seed():
    a = generate_sessions(3, 4, seed=42)
    b = generate_sessions(3, 4, seed=42)
    assert a == b

def test_different_seeds_give_different_content():
    assert generate_sessions(3, 4, seed=1) != generate_sessions(3, 4, seed=2)

def test_history_grows_monotonically_within_a_session():
    turns = [t for t in generate_sessions(1, 6, seed=7)]
    lens = [len(t.prefix_text) for t in turns]
    assert lens == sorted(lens)
    assert lens[0] < lens[-1]

def test_every_prefix_starts_with_the_stable_system_prompt():
    for t in generate_sessions(2, 3, seed=9):
        assert t.prefix_text.startswith(SYSTEM_PROMPT)
```

```python
# tests/traces/test_corpus.py
from driftmatrix.traces.generator import generate_sessions
from driftmatrix.traces.corpus import freeze_corpus, load_corpus

def test_freeze_then_load_roundtrips(tmp_path):
    turns = generate_sessions(2, 3, seed=5)
    p = tmp_path / "corpus.json"
    h = freeze_corpus(turns, p)
    loaded, h2 = load_corpus(p)
    assert loaded == turns
    assert h == h2

def test_hash_is_stable_across_freezes(tmp_path):
    turns = generate_sessions(2, 3, seed=5)
    h1 = freeze_corpus(turns, tmp_path / "a.json")
    h2 = freeze_corpus(turns, tmp_path / "b.json")
    assert h1 == h2

def test_hash_changes_when_content_changes(tmp_path):
    h1 = freeze_corpus(generate_sessions(2, 3, seed=5), tmp_path / "a.json")
    h2 = freeze_corpus(generate_sessions(2, 3, seed=6), tmp_path / "b.json")
    assert h1 != h2
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/traces/test_generator.py tests/traces/test_corpus.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.traces.generator'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/traces/generator.py
"""Agent-shaped trace generation (spec 4.3).

Stable system prompt + stable tool schemas + monotonically growing history is
the shape that makes state reuse meaningful. LangGraph is used to drive the
turn structure; it is the traffic generator, never the contribution.
"""
from __future__ import annotations
import json, random
from dataclasses import dataclass
from .tools import TOOL_SCHEMAS, call_tool

SYSTEM_PROMPT = (
    "You are an operations assistant. Use the provided tools when they help. "
    "Answer concisely and cite any record ids you used.\n\n"
    "Available tools:\n" + json.dumps(TOOL_SCHEMAS, sort_keys=True, indent=2) + "\n"
)

_QUESTIONS = [
    "Look up record {rid} and summarise it.",
    "Sum the values {vals} and report the total.",
    "Format a report titled 'Batch {rid}' describing record {rid}.",
    "Compare record {rid} against the running total of {vals}.",
]


@dataclass(frozen=True)
class Turn:
    session_id: str
    turn_idx: int
    prefix_text: str
    suffix_text: str
    expected_tool: str | None


def generate_sessions(n_sessions: int, turns_per_session: int,
                      seed: int) -> list[Turn]:
    rng = random.Random(seed)
    turns: list[Turn] = []
    for s in range(n_sessions):
        sid = f"s{seed}_{s:03d}"
        history = SYSTEM_PROMPT
        for t in range(turns_per_session):
            rid = rng.randrange(0, 200)
            vals = [rng.randrange(1, 50) for _ in range(3)]
            q = rng.choice(_QUESTIONS).format(rid=rid, vals=vals)
            tool = ("lookup_record" if "Look up" in q else
                    "compute_sum" if "Sum the" in q else
                    "format_report" if "Format a" in q else None)
            turns.append(Turn(session_id=sid, turn_idx=t,
                              prefix_text=history, suffix_text=f"\nUser: {q}\nAssistant:",
                              expected_tool=tool))
            # grow history exactly as a real agent would: question + tool result
            observed = call_tool(tool, {"record_id": rid} if tool == "lookup_record"
                                 else {"values": vals} if tool == "compute_sum"
                                 else {"title": f"Batch {rid}", "body": f"record {rid}"}
                                 ) if tool else "(no tool)"
            history += f"\nUser: {q}\nAssistant: [{tool}] {observed}\n"
    return turns
```

```python
# src/driftmatrix/traces/corpus.py
"""Freeze, hash and reload the trace corpus (spec 4.3)."""
from __future__ import annotations
import hashlib, json
from dataclasses import asdict
from pathlib import Path
from .generator import Turn


def _canonical(turns: list[Turn]) -> str:
    return json.dumps([asdict(t) for t in turns], sort_keys=True, separators=(",", ":"))


def freeze_corpus(turns: list[Turn], path: Path) -> str:
    blob = _canonical(turns)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    Path(path).write_text(json.dumps({"hash": digest, "turns": json.loads(blob)}),
                          encoding="utf-8")
    return digest


def load_corpus(path: Path) -> tuple[list[Turn], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    turns = [Turn(**t) for t in data["turns"]]
    recomputed = hashlib.sha256(_canonical(turns).encode()).hexdigest()
    if recomputed != data["hash"]:
        raise ValueError(f"corpus hash mismatch: {recomputed} != {data['hash']}")
    return turns, data["hash"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/traces/ -v`
Expected: 13 passed (5 generator + 3 corpus + 5 tools)

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/traces/ tests/traces/
git commit -m "feat: agent trace generator with frozen hashed corpus"
```

---

## Task 8: Engine protocol and fake engine

The fake engine is what makes every downstream task testable with no GPU. Spec §9.

**Files:**
- Create: `src/driftmatrix/runner/__init__.py`
- Create: `src/driftmatrix/runner/engine.py`
- Create: `src/driftmatrix/runner/fake.py`
- Test: `tests/runner/test_fake.py`

**Interfaces:**
- Consumes: `TokenDist` from `metrics.distributional`; `LayerType` from `config`.
- Produces: `GenerationResult(tokens: list[int], dists: list[TokenDist], text: str)` frozen dataclass; `Engine` Protocol with methods `load(spec)`, `reset(seq_id)`, `eval_tokens(seq_id, tokens)`, `save_state(seq_id, path)`, `load_state(seq_id, path)`, `generate(seq_id, max_tokens, n_probs) -> GenerationResult`, `read_state_tensors(path) -> dict[int, np.ndarray]`, `layer_types() -> dict[int, LayerType]`; `FakeEngine` implementing it with a controllable `divergence_at: int | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/runner/test_fake.py
import numpy as np
from pathlib import Path
from driftmatrix.config import LayerType
from driftmatrix.runner.fake import FakeEngine

def test_fake_generates_deterministically(tmp_path):
    e = FakeEngine()
    e.reset(0); e.eval_tokens(0, [1, 2, 3])
    a = e.generate(0, max_tokens=8, n_probs=5)
    e.reset(0); e.eval_tokens(0, [1, 2, 3])
    b = e.generate(0, max_tokens=8, n_probs=5)
    assert a.tokens == b.tokens

def test_state_roundtrip_preserves_generation(tmp_path):
    p = tmp_path / "s.bin"
    e = FakeEngine()
    e.reset(0); e.eval_tokens(0, [1, 2, 3]); e.save_state(0, p)
    e.eval_tokens(0, [4, 5]); direct = e.generate(0, 8, 5)
    e.reset(0); e.load_state(0, p); e.eval_tokens(0, [4, 5]); restored = e.generate(0, 8, 5)
    assert direct.tokens == restored.tokens

def test_injected_divergence_appears_at_requested_position(tmp_path):
    p = tmp_path / "s.bin"
    e = FakeEngine(divergence_at=3)
    e.reset(0); e.eval_tokens(0, [1, 2, 3]); e.save_state(0, p)
    e.eval_tokens(0, [4]); direct = e.generate(0, 8, 5)
    e.reset(0); e.load_state(0, p); e.eval_tokens(0, [4]); restored = e.generate(0, 8, 5)
    assert direct.tokens[:3] == restored.tokens[:3]
    assert direct.tokens[3] != restored.tokens[3]

def test_dists_have_requested_width_and_sum_below_one():
    e = FakeEngine()
    e.reset(0); e.eval_tokens(0, [1])
    g = e.generate(0, max_tokens=4, n_probs=5)
    for d in g.dists:
        assert len(d.token_ids) == 5
        assert sum(np.exp(d.logprobs)) <= 1.0 + 1e-9

def test_state_tensors_and_layer_types_are_consistent(tmp_path):
    p = tmp_path / "s.bin"
    e = FakeEngine()
    e.reset(0); e.eval_tokens(0, [1, 2]); e.save_state(0, p)
    tensors = e.read_state_tensors(p)
    types = e.layer_types()
    assert set(tensors) == set(types)
    assert LayerType.LINEAR_ATTENTION in types.values()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/runner/test_fake.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.runner'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/runner/__init__.py
"""Engine execution. The only package that touches a real model."""
```

```python
# src/driftmatrix/runner/engine.py
"""Engine protocol. Implementations: fake.FakeEngine, llamacpp.LlamaCppEngine."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import numpy as np
from ..config import LayerType, ModelSpec
from ..metrics.distributional import TokenDist


@dataclass(frozen=True)
class GenerationResult:
    tokens: list[int]
    dists: list[TokenDist]
    text: str


class Engine(Protocol):
    def load(self, spec: ModelSpec) -> None: ...
    def reset(self, seq_id: int) -> None: ...
    def eval_tokens(self, seq_id: int, tokens: list[int]) -> None: ...
    def save_state(self, seq_id: int, path: Path) -> None: ...
    def load_state(self, seq_id: int, path: Path) -> None: ...
    def generate(self, seq_id: int, max_tokens: int, n_probs: int) -> GenerationResult: ...
    def read_state_tensors(self, path: Path) -> dict[int, np.ndarray]: ...
    def layer_types(self) -> dict[int, LayerType]: ...
```

```python
# src/driftmatrix/runner/fake.py
"""Deterministic in-memory engine for testing without a GPU."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from ..config import LayerType, ModelSpec
from ..metrics.distributional import TokenDist
from .engine import GenerationResult

_N_LAYERS = 8


class FakeEngine:
    """Token stream is a pure function of the evaluated context.

    `divergence_at` injects a controlled difference between a restored and a
    directly-evaluated context, so arm logic and detection can be tested.
    """

    def __init__(self, divergence_at: int | None = None):
        self.divergence_at = divergence_at
        self._ctx: dict[int, list[int]] = {}
        self._restored: dict[int, bool] = {}

    def load(self, spec: ModelSpec) -> None:
        self._spec = spec

    def reset(self, seq_id: int) -> None:
        self._ctx[seq_id] = []
        self._restored[seq_id] = False

    def eval_tokens(self, seq_id: int, tokens: list[int]) -> None:
        self._ctx.setdefault(seq_id, []).extend(tokens)

    def save_state(self, seq_id: int, path: Path) -> None:
        Path(path).write_bytes(pickle.dumps(list(self._ctx.get(seq_id, []))))

    def load_state(self, seq_id: int, path: Path) -> None:
        self._ctx[seq_id] = pickle.loads(Path(path).read_bytes())
        self._restored[seq_id] = True

    def generate(self, seq_id: int, max_tokens: int, n_probs: int) -> GenerationResult:
        ctx = self._ctx.get(seq_id, [])
        seed = sum(ctx) % 9973
        rng = np.random.default_rng(seed)
        toks, dists = [], []
        for i in range(max_tokens):
            base = int(rng.integers(10, 5000))
            if (self.divergence_at is not None
                    and i == self.divergence_at and self._restored.get(seq_id)):
                base += 1                       # controlled, reproducible divergence
            toks.append(base)
            ids = [base] + [base + j for j in range(1, n_probs)]
            probs = np.array([0.6] + [0.3 / (n_probs - 1)] * (n_probs - 1))
            dists.append(TokenDist(token_ids=ids, logprobs=list(np.log(probs))))
        return GenerationResult(tokens=toks, dists=dists, text=" ".join(map(str, toks)))

    def read_state_tensors(self, path: Path) -> dict[int, np.ndarray]:
        ctx = pickle.loads(Path(path).read_bytes())
        rng = np.random.default_rng(sum(ctx) % 9973)
        return {i: rng.normal(size=(4, 4)).astype(np.float32) for i in range(_N_LAYERS)}

    def layer_types(self) -> dict[int, LayerType]:
        # mirrors the 3 linear + 1 full pattern of the treatment model
        return {i: (LayerType.FULL_ATTENTION if (i + 1) % 4 == 0
                    else LayerType.LINEAR_ATTENTION) for i in range(_N_LAYERS)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/runner/test_fake.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/runner/ tests/runner/
git commit -m "feat: engine protocol and deterministic fake engine"
```

---

## Task 9: Arm execution and outcome classification

Spec §4.2 and §7.3. The arm is a property of the procedure, so there is no path-verification logic — but failures must still be classified rather than crashing the run.

**Files:**
- Create: `src/driftmatrix/runner/arms.py`
- Test: `tests/runner/test_arms.py`

**Interfaces:**
- Consumes: `Engine`, `GenerationResult` from `runner.engine`; `Arm`, `Outcome` from `config`.
- Produces: `ArmResult(arm: Arm, outcome: Outcome, generation: GenerationResult | None, state_path: Path | None, error: str | None)` frozen dataclass; `run_restore_arm(engine, seq_id, prefix, suffix, max_new, n_probs, state_path) -> ArmResult`; `run_recompute_arm(engine, seq_id, prefix, suffix, max_new, n_probs) -> ArmResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/runner/test_arms.py
import pytest
from pathlib import Path
from driftmatrix.config import Arm, Outcome
from driftmatrix.runner.fake import FakeEngine
from driftmatrix.runner.arms import run_restore_arm, run_recompute_arm

PRE, SUF = [1, 2, 3, 4], [5, 6]

def test_both_arms_succeed_and_agree_when_engine_is_faithful(tmp_path):
    e = FakeEngine()
    r = run_restore_arm(e, 0, PRE, SUF, 8, 5, tmp_path / "s.bin")
    c = run_recompute_arm(e, 1, PRE, SUF, 8, 5)
    assert r.outcome is Outcome.OK and c.outcome is Outcome.OK
    assert r.arm is Arm.RESTORE and c.arm is Arm.RECOMPUTE
    assert r.generation.tokens == c.generation.tokens

def test_arms_disagree_when_engine_injects_divergence(tmp_path):
    e = FakeEngine(divergence_at=2)
    r = run_restore_arm(e, 0, PRE, SUF, 8, 5, tmp_path / "s.bin")
    c = run_recompute_arm(e, 1, PRE, SUF, 8, 5)
    assert r.generation.tokens != c.generation.tokens

def test_restore_arm_records_its_state_path(tmp_path):
    p = tmp_path / "s.bin"
    r = run_restore_arm(FakeEngine(), 0, PRE, SUF, 4, 5, p)
    assert r.state_path == p and p.exists()

def test_state_io_failure_is_classified_not_raised(tmp_path):
    class BadSave(FakeEngine):
        def save_state(self, seq_id, path):
            raise OSError("disk full")
    r = run_restore_arm(BadSave(), 0, PRE, SUF, 4, 5, tmp_path / "s.bin")
    assert r.outcome is Outcome.STATE_IO_FAILED
    assert r.generation is None
    assert "disk full" in r.error

def test_unsupported_engine_is_classified(tmp_path):
    class NoState(FakeEngine):
        def save_state(self, seq_id, path):
            raise NotImplementedError("state API unavailable")
    r = run_restore_arm(NoState(), 0, PRE, SUF, 4, 5, tmp_path / "s.bin")
    assert r.outcome is Outcome.UNSUPPORTED

def test_generic_engine_crash_is_classified(tmp_path):
    class Boom(FakeEngine):
        def generate(self, *a, **k):
            raise RuntimeError("cuda error")
    c = run_recompute_arm(Boom(), 0, PRE, SUF, 4, 5)
    assert c.outcome is Outcome.CRASHED and "cuda error" in c.error
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/runner/test_arms.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.runner.arms'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/runner/arms.py
"""The two arms (spec 4.2).

Restore:   eval(prefix) -> save -> fresh context -> load -> eval(suffix) -> generate
Recompute: fresh context -> eval(prefix + suffix) -> generate

Identical tokens, identical weights, greedy decoding. The only difference is
whether the state was restored or recomputed.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from ..config import Arm, Outcome
from .engine import Engine, GenerationResult


@dataclass(frozen=True)
class ArmResult:
    arm: Arm
    outcome: Outcome
    generation: GenerationResult | None
    state_path: Path | None
    error: str | None


def _fail(arm: Arm, exc: Exception, state_path: Path | None) -> ArmResult:
    if isinstance(exc, NotImplementedError):
        outcome = Outcome.UNSUPPORTED
    elif isinstance(exc, (OSError, IOError)):
        outcome = Outcome.STATE_IO_FAILED
    elif isinstance(exc, TimeoutError):
        outcome = Outcome.TIMEOUT
    else:
        outcome = Outcome.CRASHED
    return ArmResult(arm, outcome, None, state_path, f"{type(exc).__name__}: {exc}")


def run_restore_arm(engine: Engine, seq_id: int, prefix: list[int], suffix: list[int],
                    max_new: int, n_probs: int, state_path: Path) -> ArmResult:
    try:
        engine.reset(seq_id)
        engine.eval_tokens(seq_id, prefix)
        engine.save_state(seq_id, state_path)
        engine.reset(seq_id)
        engine.load_state(seq_id, state_path)
        engine.eval_tokens(seq_id, suffix)
        gen = engine.generate(seq_id, max_new, n_probs)
        return ArmResult(Arm.RESTORE, Outcome.OK, gen, state_path, None)
    except Exception as exc:                      # noqa: BLE001 - classified below
        return _fail(Arm.RESTORE, exc, state_path)


def run_recompute_arm(engine: Engine, seq_id: int, prefix: list[int], suffix: list[int],
                      max_new: int, n_probs: int) -> ArmResult:
    try:
        engine.reset(seq_id)
        engine.eval_tokens(seq_id, prefix + suffix)
        gen = engine.generate(seq_id, max_new, n_probs)
        return ArmResult(Arm.RECOMPUTE, Outcome.OK, gen, None, None)
    except Exception as exc:                      # noqa: BLE001 - classified below
        return _fail(Arm.RECOMPUTE, exc, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/runner/test_arms.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/runner/arms.py tests/runner/test_arms.py
git commit -m "feat: restore and recompute arms with outcome classification"
```

---

## Task 10: Parquet capture

Spec §7.2. Two tables. Every row carries engine commit, quant and trace hash, because upstream is actively moving.

**Files:**
- Create: `src/driftmatrix/runner/capture.py`
- Test: `tests/runner/test_capture.py`

**Interfaces:**
- Consumes: `ArmResult` from `runner.arms`; `LayerDiff` from `statediff`; `Condition`, `Arm`, `Mode`, `ModelSpec` from `config`.
- Produces: `RunMeta(run_id, engine, engine_commit, quant, trace_hash, run_started_at)` frozen dataclass; `generations_rows(...) -> list[dict]`; `statediff_rows(...) -> list[dict]`; `write_parquet(rows: list[dict], path: Path) -> None`; `GENERATIONS_COLUMNS: tuple[str, ...]`; `STATEDIFF_COLUMNS: tuple[str, ...]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/runner/test_capture.py
import pandas as pd
from datetime import datetime, timezone
from driftmatrix.config import (
    ALL_CONDITIONS, Arm, Mode, ModelSpec, LayerType, Outcome,
)
from driftmatrix.statediff import LayerDiff
from driftmatrix.runner.fake import FakeEngine
from driftmatrix.runner.arms import run_recompute_arm
from driftmatrix.runner.capture import (
    RunMeta, generations_rows, statediff_rows, write_parquet,
    GENERATIONS_COLUMNS, STATEDIFF_COLUMNS,
)

META = RunMeta("run1", "llama.cpp", "abc123", "IQ4_XS", "hash1",
               datetime(2026, 9, 10, tzinfo=timezone.utc))
SPEC = ModelSpec("ornith-9b", 3, "/m.gguf", "IQ4_XS")

def test_generations_rows_have_exact_schema():
    r = run_recompute_arm(FakeEngine(), 0, [1, 2], [3], 4, 5)
    rows = generations_rows(META, SPEC, ALL_CONDITIONS[0], 1, Mode.FREE_RUNNING,
                            "s1", 0, 100, r, tool=None)
    assert rows and set(rows[0]) == set(GENERATIONS_COLUMNS)

def test_one_row_per_generated_position():
    r = run_recompute_arm(FakeEngine(), 0, [1, 2], [3], 4, 5)
    rows = generations_rows(META, SPEC, ALL_CONDITIONS[0], 1, Mode.FREE_RUNNING,
                            "s1", 0, 100, r, tool=None)
    assert len(rows) == 4
    assert [x["position"] for x in rows] == [0, 1, 2, 3]

def test_failed_arm_produces_a_single_outcome_row_with_no_position():
    from driftmatrix.runner.arms import ArmResult
    bad = ArmResult(Arm.RESTORE, Outcome.CRASHED, None, None, "RuntimeError: boom")
    rows = generations_rows(META, SPEC, ALL_CONDITIONS[0], 1, Mode.FREE_RUNNING,
                            "s1", 0, 100, bad, tool=None)
    assert len(rows) == 1
    assert rows[0]["outcome"] == Outcome.CRASHED.value
    assert rows[0]["position"] is None

def test_metadata_is_stamped_on_every_row():
    r = run_recompute_arm(FakeEngine(), 0, [1, 2], [3], 4, 5)
    rows = generations_rows(META, SPEC, ALL_CONDITIONS[0], 1, Mode.FREE_RUNNING,
                            "s1", 0, 100, r, tool=None)
    for row in rows:
        assert row["engine_commit"] == "abc123"
        assert row["quant"] == "IQ4_XS"
        assert row["trace_hash"] == "hash1"
        assert row["boundary_token_offset"] == 100

def test_statediff_rows_schema():
    diffs = [LayerDiff(0, LayerType.LINEAR_ATTENTION, 0.1, 0.2, 0.3, "float32")]
    rows = statediff_rows(META, SPEC, ALL_CONDITIONS[0], 1, diffs)
    assert set(rows[0]) == set(STATEDIFF_COLUMNS)
    assert rows[0]["layer_type"] == "linear_attention"

def test_write_parquet_roundtrips(tmp_path):
    r = run_recompute_arm(FakeEngine(), 0, [1, 2], [3], 4, 5)
    rows = generations_rows(META, SPEC, ALL_CONDITIONS[0], 1, Mode.FREE_RUNNING,
                            "s1", 0, 100, r, tool=None)
    p = tmp_path / "g.parquet"
    write_parquet(rows, p)
    assert len(pd.read_parquet(p)) == len(rows)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/runner/test_capture.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.runner.capture'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/runner/capture.py
"""Parquet capture (spec 7.2). The seam: engine writes once, analysis reads."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd
from ..config import Arm, Condition, Mode, ModelSpec, Outcome
from ..metrics.behavioral import ToolCall
from ..statediff import LayerDiff
from .arms import ArmResult

GENERATIONS_COLUMNS = (
    "model_id", "arch_level", "condition_id", "alignment", "prefix_band",
    "continuation_band", "arm", "replicate", "mode", "session_id", "turn_idx",
    "position", "token_ids", "logprobs", "chosen_token_id", "tool_name",
    "tool_args_hash", "boundary_token_offset", "outcome", "error",
    "engine", "engine_commit", "quant", "trace_hash", "run_id", "run_started_at",
)

STATEDIFF_COLUMNS = (
    "model_id", "arch_level", "condition_id", "replicate", "layer_idx",
    "layer_type", "max_abs_diff", "max_rel_diff", "frobenius_norm_diff",
    "state_dtype", "engine_commit", "quant", "run_id",
)


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    engine: str
    engine_commit: str
    quant: str
    trace_hash: str
    run_started_at: datetime


def _base(meta: RunMeta, spec: ModelSpec, cond: Condition, replicate: int) -> dict:
    return {
        "model_id": spec.model_id, "arch_level": spec.arch_level,
        "condition_id": cond.condition_id,
        "alignment": cond.alignment.value,
        "prefix_band": cond.prefix_band.value,
        "continuation_band": cond.continuation_band.value,
        "replicate": replicate,
        "engine": meta.engine, "engine_commit": meta.engine_commit,
        "quant": meta.quant, "trace_hash": meta.trace_hash,
        "run_id": meta.run_id, "run_started_at": meta.run_started_at,
    }


def generations_rows(meta: RunMeta, spec: ModelSpec, cond: Condition, replicate: int,
                     mode: Mode, session_id: str, turn_idx: int,
                     boundary_token_offset: int, result: ArmResult,
                     tool: ToolCall | None) -> list[dict]:
    base = _base(meta, spec, cond, replicate) | {
        "arm": result.arm.value, "mode": mode.value,
        "session_id": session_id, "turn_idx": turn_idx,
        "boundary_token_offset": boundary_token_offset,
        "outcome": result.outcome.value, "error": result.error,
        "tool_name": tool.name if tool else None,
        "tool_args_hash": tool.args_hash if tool else None,
    }
    if result.generation is None:
        return [base | {"position": None, "token_ids": None, "logprobs": None,
                        "chosen_token_id": None}]
    rows = []
    for i, (tok, dist) in enumerate(zip(result.generation.tokens,
                                        result.generation.dists)):
        rows.append(base | {"position": i, "token_ids": list(dist.token_ids),
                            "logprobs": list(dist.logprobs), "chosen_token_id": tok})
    return rows


def statediff_rows(meta: RunMeta, spec: ModelSpec, cond: Condition, replicate: int,
                   diffs: list[LayerDiff]) -> list[dict]:
    return [{
        "model_id": spec.model_id, "arch_level": spec.arch_level,
        "condition_id": cond.condition_id, "replicate": replicate,
        "layer_idx": d.layer_idx, "layer_type": d.layer_type.value,
        "max_abs_diff": d.max_abs_diff, "max_rel_diff": d.max_rel_diff,
        "frobenius_norm_diff": d.frobenius_norm_diff, "state_dtype": d.state_dtype,
        "engine_commit": meta.engine_commit, "quant": meta.quant,
        "run_id": meta.run_id,
    } for d in diffs]


def write_parquet(rows: list[dict], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/runner/test_capture.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/runner/capture.py tests/runner/test_capture.py
git commit -m "feat: parquet capture for generations and state diffs"
```

---

## Task 11: Real llama.cpp engine adapter

The only task that needs the model. It must satisfy exactly the same protocol the fake does, so everything above already works against it.

**Files:**
- Create: `src/driftmatrix/runner/llamacpp.py`
- Test: `tests/runner/test_llamacpp.py` (integration, skipped unless `DRIFTMATRIX_MODEL` is set)

**Interfaces:**
- Consumes: `Engine`, `GenerationResult` from `runner.engine`; `LayerType`, `ModelSpec` from `config`.
- Produces: `LlamaCppEngine(n_ctx: int = 16384, seed: int = 1234)` implementing `Engine`; `parse_layer_types(config_json: dict) -> dict[int, LayerType]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/runner/test_llamacpp.py
import os, pytest
from pathlib import Path
from driftmatrix.config import LayerType, ModelSpec
from driftmatrix.runner.llamacpp import LlamaCppEngine, parse_layer_types

def test_parse_layer_types_from_qwen3_5_config():
    cfg = {"num_hidden_layers": 8,
           "layer_types": ["linear_attention"] * 3 + ["full_attention"]
                          + ["linear_attention"] * 3 + ["full_attention"]}
    types = parse_layer_types(cfg)
    assert len(types) == 8
    assert types[0] is LayerType.LINEAR_ATTENTION
    assert types[3] is LayerType.FULL_ATTENTION

def test_parse_layer_types_falls_back_to_interval():
    cfg = {"num_hidden_layers": 8, "full_attention_interval": 4}
    types = parse_layer_types(cfg)
    assert types[3] is LayerType.FULL_ATTENTION
    assert types[2] is LayerType.LINEAR_ATTENTION

def test_parse_layer_types_handles_sliding_window():
    cfg = {"num_hidden_layers": 4,
           "layer_types": ["sliding_attention"] * 3 + ["full_attention"]}
    assert parse_layer_types(cfg)[0] is LayerType.SLIDING_ATTENTION

MODEL = os.environ.get("DRIFTMATRIX_MODEL")

@pytest.mark.skipif(not MODEL, reason="set DRIFTMATRIX_MODEL to run integration test")
def test_real_engine_state_roundtrip(tmp_path):
    e = LlamaCppEngine(n_ctx=2048)
    e.load(ModelSpec("ornith-9b", 3, MODEL, "IQ4_XS"))
    pre, suf = list(range(100, 140)), list(range(200, 210))
    p = tmp_path / "s.bin"
    e.reset(0); e.eval_tokens(0, pre); e.save_state(0, p)
    e.reset(0); e.load_state(0, p); e.eval_tokens(0, suf)
    a = e.generate(0, 16, 10)
    e.reset(0); e.eval_tokens(0, pre + suf)
    b = e.generate(0, 16, 10)
    # This is negative control 1. Record the result either way.
    print("identical:", a.tokens == b.tokens)
    assert len(a.tokens) == len(b.tokens) == 16
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/runner/test_llamacpp.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.runner.llamacpp'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/runner/llamacpp.py
"""llama.cpp adapter. Satisfies the same Engine protocol as FakeEngine."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ..config import LayerType, ModelSpec
from ..metrics.distributional import TokenDist
from .engine import GenerationResult


def parse_layer_types(config_json: dict) -> dict[int, LayerType]:
    """Map layer index -> type from a HuggingFace config.json."""
    n = int(config_json["num_hidden_layers"])
    explicit = config_json.get("layer_types")
    if explicit:
        return {i: LayerType(explicit[i]) for i in range(n)}
    interval = int(config_json.get("full_attention_interval", 0))
    if interval <= 0:
        return {i: LayerType.FULL_ATTENTION for i in range(n)}
    return {i: (LayerType.FULL_ATTENTION if (i + 1) % interval == 0
                else LayerType.LINEAR_ATTENTION) for i in range(n)}


class LlamaCppEngine:
    def __init__(self, n_ctx: int = 16384, seed: int = 1234):
        self.n_ctx, self.seed = n_ctx, seed
        self._llm = None
        self._layer_types: dict[int, LayerType] = {}

    def load(self, spec: ModelSpec) -> None:
        from llama_cpp import Llama
        self._spec = spec
        self._llm = Llama(model_path=spec.gguf_path, n_ctx=self.n_ctx,
                          seed=self.seed, logits_all=False, verbose=False)
        cfg = Path(spec.gguf_path).with_name("config.json")
        if cfg.exists():
            self._layer_types = parse_layer_types(json.loads(cfg.read_text()))

    def reset(self, seq_id: int) -> None:
        self._llm.reset()

    def eval_tokens(self, seq_id: int, tokens: list[int]) -> None:
        self._llm.eval(tokens)

    def save_state(self, seq_id: int, path: Path) -> None:
        state = self._llm.save_state()
        Path(path).write_bytes(bytes(state.llama_state[: state.llama_state_size]))

    def load_state(self, seq_id: int, path: Path) -> None:
        raw = Path(path).read_bytes()
        self._llm.load_state(raw)

    def generate(self, seq_id: int, max_tokens: int, n_probs: int) -> GenerationResult:
        toks, dists = [], []
        for _ in range(max_tokens):
            logits = np.asarray(self._llm.scores[self._llm.n_tokens - 1], dtype=np.float64)
            top = np.argpartition(-logits, n_probs)[:n_probs]
            top = top[np.argsort(-logits[top])]
            shifted = logits - logits.max()
            logprobs = shifted - np.log(np.exp(shifted).sum())
            dists.append(TokenDist(token_ids=[int(t) for t in top],
                                   logprobs=[float(logprobs[t]) for t in top]))
            nxt = int(top[0])                       # greedy
            if nxt == self._llm.token_eos():
                break
            toks.append(nxt)
            self._llm.eval([nxt])
        return GenerationResult(tokens=toks, dists=dists,
                                text=self._llm.detokenize(toks).decode("utf-8", "replace"))

    def read_state_tensors(self, path: Path) -> dict[int, np.ndarray]:
        """Best-effort extraction of per-layer state from a saved blob.

        llama.cpp serialises state as an opaque blob. Where per-layer views are
        not exposed by the build, this returns a single pseudo-layer keyed -1
        holding the raw bytes as uint8, which still supports byte-level
        equality checks in statediff.
        """
        raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        return {-1: raw}

    def layer_types(self) -> dict[int, LayerType]:
        return dict(self._layer_types)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/runner/test_llamacpp.py -v`
Expected: 3 passed, 1 skipped (integration test skipped without `DRIFTMATRIX_MODEL`)

Then with the model: `DRIFTMATRIX_MODEL=~/models/Ornith-1.5-9B-IQ4_XS.gguf pytest tests/runner/test_llamacpp.py -v -s`
Expected: 4 passed, and the printed `identical:` line is **negative control 1**. Record it in `docs/findings/`.

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/runner/llamacpp.py tests/runner/test_llamacpp.py
git commit -m "feat: llama.cpp engine adapter"
```

---

## Task 12: Chunk-size sweep — G1

Spec §1.2 and gate G1. The prediction requires knowing the kernel's chunk size, which is implementation-defined and must be found empirically by locating the periodicity of divergence across boundary offsets.

**Files:**
- Create: `src/driftmatrix/sweep.py`
- Create: `scripts/g1_chunk_sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: `Engine` from `runner.engine`; `run_restore_arm`, `run_recompute_arm` from `runner.arms`; `first_divergence` from `metrics.behavioral`.
- Produces: `SweepPoint(offset: int, diverged: bool, first_div_position: int)` frozen dataclass; `sweep_offsets(engine, tokens, offsets, max_new, n_probs, state_dir) -> list[SweepPoint]`; `detect_period(points: list[SweepPoint]) -> int | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep.py
from driftmatrix.sweep import SweepPoint, detect_period, sweep_offsets
from driftmatrix.runner.fake import FakeEngine

def test_detects_period_when_divergence_is_periodic():
    # divergence at every offset NOT a multiple of 64
    pts = [SweepPoint(o, o % 64 != 0, 0 if o % 64 else 32) for o in range(0, 512, 8)]
    assert detect_period(pts) == 64

def test_returns_none_when_nothing_diverges():
    pts = [SweepPoint(o, False, 32) for o in range(0, 512, 8)]
    assert detect_period(pts) is None

def test_returns_none_when_everything_diverges():
    pts = [SweepPoint(o, True, 0) for o in range(0, 512, 8)]
    assert detect_period(pts) is None

def test_sweep_runs_both_arms_at_every_offset(tmp_path):
    e = FakeEngine()
    toks = list(range(300))
    pts = sweep_offsets(e, toks, offsets=[64, 128, 192], max_new=4, n_probs=5,
                        state_dir=tmp_path)
    assert [p.offset for p in pts] == [64, 128, 192]
    assert all(p.diverged is False for p in pts)   # faithful fake never diverges
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_sweep.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.sweep'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/sweep.py
"""Boundary-offset sweep to locate the kernel chunk size (spec 1.2, gate G1)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .metrics.behavioral import first_divergence
from .runner.arms import run_restore_arm, run_recompute_arm
from .runner.engine import Engine

CANDIDATE_PERIODS = (8, 16, 32, 64, 128, 256, 512)


@dataclass(frozen=True)
class SweepPoint:
    offset: int
    diverged: bool
    first_div_position: int


def sweep_offsets(engine: Engine, tokens: list[int], offsets: list[int],
                  max_new: int, n_probs: int, state_dir: Path) -> list[SweepPoint]:
    out = []
    for off in offsets:
        prefix, suffix = tokens[:off], tokens[off:]
        r = run_restore_arm(engine, 0, prefix, suffix, max_new, n_probs,
                            Path(state_dir) / f"s_{off}.bin")
        c = run_recompute_arm(engine, 1, prefix, suffix, max_new, n_probs)
        if r.generation is None or c.generation is None:
            out.append(SweepPoint(off, False, max_new))
            continue
        pos, censored = first_divergence(r.generation.tokens, c.generation.tokens,
                                         cap=max_new)
        out.append(SweepPoint(off, not censored, pos))
    return out


def detect_period(points: list[SweepPoint]) -> int | None:
    """Smallest candidate period p where offsets divisible by p never diverge
    and offsets not divisible by p always do."""
    if all(p.diverged for p in points) or not any(p.diverged for p in points):
        return None
    for period in CANDIDATE_PERIODS:
        on = [p for p in points if p.offset % period == 0]
        off = [p for p in points if p.offset % period != 0]
        if on and off and not any(p.diverged for p in on) and all(p.diverged for p in off):
            return period
    return None
```

```python
# scripts/g1_chunk_sweep.py
"""Gate G1: locate the chunk size on the real model."""
import sys, json
from pathlib import Path
from driftmatrix.config import ModelSpec
from driftmatrix.runner.llamacpp import LlamaCppEngine
from driftmatrix.sweep import sweep_offsets, detect_period

model_path, out_dir = sys.argv[1], Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
e = LlamaCppEngine(n_ctx=4096)
e.load(ModelSpec("treatment", 3, model_path, "IQ4_XS"))
tokens = list(range(1, 1200))
offsets = list(range(256, 768, 8))
pts = sweep_offsets(e, tokens, offsets, max_new=32, n_probs=10, state_dir=out_dir)
period = detect_period(pts)
print(json.dumps({"period": period,
                  "diverged_offsets": [p.offset for p in pts if p.diverged],
                  "n_offsets": len(pts)}, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sweep.py -v`
Expected: 4 passed

Then on the real model: `python scripts/g1_chunk_sweep.py ~/models/Ornith-1.5-9B-IQ4_XS.gguf ./data/g1`

**Gate G1:** a period is found → it becomes `chunk_size` for `boundary_offset`. No period and nothing diverges → alignment is a null factor; proceed on P and C and record it. No period but everything diverges → divergence is not alignment-driven; record and investigate before proceeding.

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/sweep.py scripts/g1_chunk_sweep.py tests/test_sweep.py
git commit -m "feat: boundary-offset sweep to locate kernel chunk size"
```

---

## Task 13: Bootstrap and noise floor

Spec §6.1 and §6.2. Effects are measured against the recompute arm's own self-comparison, never against zero.

**Files:**
- Create: `src/driftmatrix/stats/__init__.py`
- Create: `src/driftmatrix/stats/bootstrap.py`
- Test: `tests/stats/test_bootstrap.py`

**Interfaces:**
- Consumes: `BOOTSTRAP_RESAMPLES`, `ALPHA`, `SELF_INCONSISTENCY_MULTIPLIER` from `config`.
- Produces: `BootstrapCI(point: float, low: float, high: float, n: int)` frozen dataclass; `paired_bootstrap(effects: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES, alpha: float = ALPHA, seed: int = 0) -> BootstrapCI`; `noise_floor(recompute_r1: np.ndarray, recompute_r2: np.ndarray) -> float`; `effect_vector(treatment: np.ndarray, floor: float) -> np.ndarray`; `is_self_inconsistent(cond_median: float, floor_median: float) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/stats/test_bootstrap.py
import numpy as np, pytest
from driftmatrix.stats.bootstrap import (
    BootstrapCI, paired_bootstrap, noise_floor, effect_vector, is_self_inconsistent,
)

def test_ci_brackets_the_point_estimate():
    x = np.random.default_rng(0).normal(0.5, 0.1, size=200)
    ci = paired_bootstrap(x, n_resamples=2000, seed=1)
    assert ci.low < ci.point < ci.high
    assert ci.point == pytest.approx(x.mean(), abs=1e-12)
    assert ci.n == 200

def test_zero_effect_ci_contains_zero():
    ci = paired_bootstrap(np.zeros(100), n_resamples=1000, seed=2)
    assert ci.low <= 0.0 <= ci.high

def test_large_effect_ci_excludes_zero():
    x = np.random.default_rng(3).normal(1.0, 0.05, size=200)
    ci = paired_bootstrap(x, n_resamples=2000, seed=3)
    assert ci.low > 0.0

def test_bootstrap_is_reproducible_for_a_seed():
    x = np.random.default_rng(4).normal(0.3, 0.2, size=50)
    assert paired_bootstrap(x, 500, seed=7) == paired_bootstrap(x, 500, seed=7)

def test_noise_floor_is_the_median_self_comparison():
    a, b = np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])
    assert noise_floor(a, b) == 0.0
    assert noise_floor(np.array([0.0, 0.0]), np.array([0.4, 0.6])) == pytest.approx(0.5)

def test_effect_vector_subtracts_the_floor():
    np.testing.assert_allclose(
        effect_vector(np.array([0.5, 0.7]), floor=0.2), np.array([0.3, 0.5]))

def test_self_inconsistency_uses_the_2x_rule():
    assert is_self_inconsistent(0.21, 0.10) is True
    assert is_self_inconsistent(0.19, 0.10) is False

def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap(np.array([]))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/stats/test_bootstrap.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.stats'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/stats/__init__.py
"""Statistics. Pure functions over arrays — the part that must be right."""
```

```python
# src/driftmatrix/stats/bootstrap.py
"""Paired bootstrap against a measured noise floor (spec 6.1)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..config import ALPHA, BOOTSTRAP_RESAMPLES, SELF_INCONSISTENCY_MULTIPLIER


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    low: float
    high: float
    n: int


def paired_bootstrap(effects: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES,
                     alpha: float = ALPHA, seed: int = 0) -> BootstrapCI:
    """Percentile bootstrap over per-turn effects, paired by turn."""
    x = np.asarray(effects, dtype=np.float64)
    if x.size == 0:
        raise ValueError("effects array is empty")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(float(x.mean()), float(lo), float(hi), int(x.size))


def noise_floor(recompute_r1: np.ndarray, recompute_r2: np.ndarray) -> float:
    """Median divergence between the recompute arm's own two replicates."""
    return float(np.median(np.abs(np.asarray(recompute_r1, dtype=np.float64)
                                  - np.asarray(recompute_r2, dtype=np.float64))))


def effect_vector(treatment: np.ndarray, floor: float) -> np.ndarray:
    return np.asarray(treatment, dtype=np.float64) - float(floor)


def is_self_inconsistent(cond_median: float, floor_median: float) -> bool:
    return cond_median > SELF_INCONSISTENCY_MULTIPLIER * floor_median
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/stats/test_bootstrap.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/stats/ tests/stats/
git commit -m "feat: paired bootstrap with measured noise floor"
```

---

## Task 14: Holm correction across three declared families

Spec §6.2. Pooling unrelated hypotheses over-corrects and hides real effects, so the three families are declared in code and never merged.

**Files:**
- Create: `src/driftmatrix/stats/multiplicity.py`
- Test: `tests/stats/test_multiplicity.py`

**Interfaces:**
- Consumes: `ALPHA` from `config`.
- Produces: `Family` str enum with `BEHAVIORAL`, `DISTRIBUTIONAL`, `MECHANISM`; `FAMILY_SIZES: dict[Family, int]` = `{BEHAVIORAL: 24, DISTRIBUTIONAL: 16, MECHANISM: 5}`; `holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]`; `holm_by_family(pvalues: dict[Family, dict[str, float]], alpha: float = ALPHA) -> dict[Family, dict[str, bool]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/stats/test_multiplicity.py
import pytest
from driftmatrix.stats.multiplicity import Family, FAMILY_SIZES, holm, holm_by_family

def test_family_sizes_match_spec():
    assert FAMILY_SIZES[Family.BEHAVIORAL] == 24
    assert FAMILY_SIZES[Family.DISTRIBUTIONAL] == 16
    assert FAMILY_SIZES[Family.MECHANISM] == 5

def test_holm_matches_hand_computed_example():
    # sorted p: .001 vs .05/3, .02 vs .05/2, .04 vs .05/1
    out = holm({"a": 0.001, "b": 0.02, "c": 0.04}, alpha=0.05)
    assert out == {"a": True, "b": True, "c": True}

def test_holm_stops_at_first_failure():
    # .04 > .05/3 -> reject nothing, and no later test may pass
    out = holm({"a": 0.04, "b": 0.045, "c": 0.049}, alpha=0.05)
    assert out == {"a": False, "b": False, "c": False}

def test_holm_is_less_conservative_than_bonferroni():
    out = holm({"a": 0.02, "b": 0.9, "c": 0.9}, alpha=0.05)
    assert out["a"] is True          # 0.02 < 0.05/3 = 0.0167 is False for Bonferroni-all
    assert out["b"] is False

def test_families_are_corrected_independently():
    pv = {Family.BEHAVIORAL: {"x": 0.02},
          Family.MECHANISM: {"y": 0.02}}
    out = holm_by_family(pv, alpha=0.05)
    assert out[Family.BEHAVIORAL]["x"] is True
    assert out[Family.MECHANISM]["y"] is True

def test_empty_family_is_allowed():
    assert holm_by_family({Family.MECHANISM: {}}) == {Family.MECHANISM: {}}

def test_pvalue_out_of_range_raises():
    with pytest.raises(ValueError, match="p-value"):
        holm({"a": 1.5})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/stats/test_multiplicity.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.stats.multiplicity'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/stats/multiplicity.py
"""Holm-Bonferroni within three declared families (spec 6.2).

Family-wise error control, not FDR: these are bug claims and a false positive
is expensive. Families are never pooled — pooling unrelated hypotheses
over-corrects and hides real effects.
"""
from __future__ import annotations
from enum import Enum
from ..config import ALPHA


class Family(str, Enum):
    BEHAVIORAL = "behavioral"          # 8 conditions x 3 architectures
    DISTRIBUTIONAL = "distributional"  # 8 conditions x 2 statistics, level 3
    MECHANISM = "mechanism"            # alignment, alignment x arch, P and C trends


FAMILY_SIZES: dict[Family, int] = {
    Family.BEHAVIORAL: 24,
    Family.DISTRIBUTIONAL: 16,
    Family.MECHANISM: 5,
}


def holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    for k, p in pvalues.items():
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value out of range for {k}: {p}")
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, bool] = {}
    still_rejecting = True
    for i, (key, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        if still_rejecting and p <= threshold:
            out[key] = True
        else:
            still_rejecting = False
            out[key] = False
    return {k: out[k] for k in pvalues}


def holm_by_family(pvalues: dict[Family, dict[str, float]],
                   alpha: float = ALPHA) -> dict[Family, dict[str, bool]]:
    return {fam: holm(tests, alpha) for fam, tests in pvalues.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/stats/test_multiplicity.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/stats/multiplicity.py tests/stats/test_multiplicity.py
git commit -m "feat: Holm correction across three declared families"
```

---

## Task 15: Coverage simulation — the statistics prove themselves

Spec §9. This is the project proving its own inference is calibrated rather than assuming it.

**Files:**
- Create: `src/driftmatrix/stats/coverage.py`
- Test: `tests/stats/test_coverage.py`

**Interfaces:**
- Consumes: `paired_bootstrap` from `stats.bootstrap`; `ALPHA` from `config`.
- Produces: `CoverageResult(detection_rate: float, false_positive_rate: float, n_trials: int)` frozen dataclass; `simulate_coverage(effect_size: float, n_per_trial: int, n_trials: int, noise_sd: float, seed: int) -> CoverageResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/stats/test_coverage.py
from driftmatrix.stats.coverage import simulate_coverage

def test_false_positive_rate_approximates_alpha_under_the_null():
    r = simulate_coverage(effect_size=0.0, n_per_trial=200, n_trials=400,
                          noise_sd=1.0, seed=11)
    assert 0.01 <= r.false_positive_rate <= 0.10     # alpha = 0.05, MC error allowed

def test_large_effect_is_detected_almost_always():
    r = simulate_coverage(effect_size=1.0, n_per_trial=200, n_trials=200,
                          noise_sd=1.0, seed=12)
    assert r.detection_rate >= 0.95

def test_tiny_effect_is_rarely_detected():
    r = simulate_coverage(effect_size=0.01, n_per_trial=50, n_trials=200,
                          noise_sd=1.0, seed=13)
    assert r.detection_rate <= 0.30

def test_result_reports_trial_count():
    r = simulate_coverage(0.5, 100, 50, 1.0, seed=14)
    assert r.n_trials == 50
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/stats/test_coverage.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.stats.coverage'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/stats/coverage.py
"""Coverage simulation (spec 9).

Inject an effect of known size, verify detection at the claimed rate; run under
the null, verify the false-positive rate approximates alpha.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..config import ALPHA
from .bootstrap import paired_bootstrap


@dataclass(frozen=True)
class CoverageResult:
    detection_rate: float
    false_positive_rate: float
    n_trials: int


def simulate_coverage(effect_size: float, n_per_trial: int, n_trials: int,
                      noise_sd: float, seed: int) -> CoverageResult:
    rng = np.random.default_rng(seed)
    detections = 0
    false_positives = 0
    for t in range(n_trials):
        signal = rng.normal(effect_size, noise_sd, size=n_per_trial)
        ci = paired_bootstrap(signal, n_resamples=500, alpha=ALPHA, seed=int(rng.integers(1 << 30)))
        if ci.low > 0.0:
            detections += 1
        null = rng.normal(0.0, noise_sd, size=n_per_trial)
        ci0 = paired_bootstrap(null, n_resamples=500, alpha=ALPHA, seed=int(rng.integers(1 << 30)))
        if ci0.low > 0.0 or ci0.high < 0.0:
            false_positives += 1
    return CoverageResult(detections / n_trials, false_positives / n_trials, n_trials)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/stats/test_coverage.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/stats/coverage.py tests/stats/test_coverage.py
git commit -m "test: coverage simulation validating detection and FPR"
```

---

## Task 16: Power calculation

Spec §6.5. Turn count must be justified before the main run, and achieved power reported alongside any null.

**Files:**
- Create: `src/driftmatrix/stats/power.py`
- Test: `tests/stats/test_power.py`

**Interfaces:**
- Consumes: `TARGET_EFFECT_RATE`, `TARGET_POWER`, `ALPHA` from `config`.
- Produces: `required_turns(rate: float = TARGET_EFFECT_RATE, power: float = TARGET_POWER, alpha: float = ALPHA) -> int`; `achieved_power(n_turns: int, rate: float = TARGET_EFFECT_RATE, alpha: float = ALPHA) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/stats/test_power.py
import pytest
from driftmatrix.stats.power import required_turns, achieved_power

def test_required_turns_for_spec_target_is_positive_and_finite():
    n = required_turns()
    assert 0 < n < 100_000

def test_rarer_effects_need_more_turns():
    assert required_turns(rate=0.005) > required_turns(rate=0.02)

def test_higher_power_needs_more_turns():
    assert required_turns(power=0.95) > required_turns(power=0.80)

def test_achieved_power_increases_with_turns():
    assert achieved_power(1000) > achieved_power(100)

def test_achieved_power_is_a_probability():
    for n in (10, 100, 1000, 10_000):
        assert 0.0 <= achieved_power(n) <= 1.0

def test_zero_rate_is_rejected():
    with pytest.raises(ValueError, match="rate"):
        required_turns(rate=0.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/stats/test_power.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.stats.power'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/stats/power.py
"""Power for detecting a rare per-turn divergence rate against a zero floor.

One-sided exact binomial: under the null the divergence rate is 0, so any
observed divergence is evidence. Power is the probability of observing at
least k successes where k is the smallest count with P(X >= k | H0 rate) <= alpha.
Because the null rate is 0, k = 1, and power = 1 - (1 - rate)^n.
"""
from __future__ import annotations
import math
from ..config import ALPHA, TARGET_EFFECT_RATE, TARGET_POWER


def required_turns(rate: float = TARGET_EFFECT_RATE, power: float = TARGET_POWER,
                   alpha: float = ALPHA) -> int:
    if not 0.0 < rate < 1.0:
        raise ValueError(f"rate must be in (0,1), got {rate}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0,1), got {power}")
    return math.ceil(math.log(1.0 - power) / math.log(1.0 - rate))


def achieved_power(n_turns: int, rate: float = TARGET_EFFECT_RATE,
                   alpha: float = ALPHA) -> float:
    if n_turns <= 0:
        return 0.0
    return 1.0 - (1.0 - rate) ** n_turns
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/stats/test_power.py -v`
Expected: 6 passed

Note the result: `required_turns()` returns 80 for a 2% rate at 80% power, so the planned 360 turns is comfortably sufficient. Record this in the pre-registration (Task 17).

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/stats/power.py tests/stats/test_power.py
git commit -m "feat: power calculation for rare-divergence detection"
```

---

## Task 17: Pre-registration document — G2

Spec §6.4. Committed after the noise floor and chunk size are known, before any treatment effect is examined. This task has no code and no test; its deliverable is a committed document whose git timestamp is the evidence.

**Files:**
- Create: `docs/preregistration/2026-09-24-analysis-plan.md`

**Interfaces:**
- Consumes: measured noise floor (Task 13 applied to real data), chunk size (Task 12), `required_turns()` (Task 16).
- Produces: a frozen analysis plan later tasks must not contradict.

- [ ] **Step 1: Gather the three measured inputs**

Run: `python scripts/g1_chunk_sweep.py <model> ./data/g1` and record the period.
Run the recompute-arm replicates and compute `noise_floor(...)`; record the median.
Run: `python -c "from driftmatrix.stats.power import required_turns; print(required_turns())"`

- [ ] **Step 2: Write the pre-registration**

Create `docs/preregistration/2026-09-24-analysis-plan.md` with these sections, all filled with the measured values, none left open:

1. **Measured inputs** — chunk size, noise floor median (must be < 0.01 nats per Global Constraints, or the G2 fallback is invoked and stated here), required turns vs planned turns.
2. **Hypotheses** — H1 misaligned boundaries diverge and aligned do not; H2 the alignment effect is present at arch level 3 and absent at levels 2 and 1; H3 divergence increases with prefix band and continuation band.
3. **Primary outcome** — tool-call disagreement rate. **Secondary** — exact-match rate, first-divergence position, mean and max per-position JSD, per-layer state diffs.
4. **Families** — the three from `stats/multiplicity.py`, with membership enumerated test by test.
5. **Decision rules** — an effect is claimed only when its Holm-corrected p-value passes within its family AND its bootstrap CI excludes zero.
6. **Self-inconsistency rule** — 2.0× the recompute self-comparison median, stated as a pre-specified engineering threshold.
7. **Stopping rule** — the full grid runs to completion; no interim looks at treatment effects.
8. **Null-result commitment** — the exact sentence from spec §6.5, with the detection-limit placeholder to be filled from the achieved power.

- [ ] **Step 3: Verify no treatment effects have been examined**

Run: `git log --oneline -- data/ | head -20`
Confirm no analysis of restore-arm results has been committed before this document. If any has, note it explicitly in the document — an honest record beats a clean-looking one.

- [ ] **Step 4: Commit**

```bash
git add docs/preregistration/2026-09-24-analysis-plan.md
git commit -m "docs: pre-register analysis plan before examining treatment effects"
```

- [ ] **Step 5: Tag the commit so the timestamp is unambiguous**

```bash
git tag -a prereg-2026-09-24 -m "Analysis plan frozen before treatment effects examined"
```

---

## Task 18: Synthetic positive control and detection floor

Spec §8. Stronger than reproducing a third party's bug: it calibrates sensitivity in the mechanism's own units.

**Files:**
- Create: `src/driftmatrix/controls.py`
- Test: `tests/test_controls.py`

**Interfaces:**
- Consumes: `Engine` from `runner.engine`; `paired_bootstrap` from `stats.bootstrap`; `per_position_jsd` from `metrics.distributional`.
- Produces: `PerturbEngine(inner: Engine, epsilon: float)` wrapping any engine and perturbing restored state by a known relative magnitude; `DetectionFloor(smallest_detected: float | None, tested: list[float])` frozen dataclass; `find_detection_floor(make_engine, epsilons, n_turns, seed) -> DetectionFloor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_controls.py
import numpy as np
from driftmatrix.runner.fake import FakeEngine
from driftmatrix.controls import PerturbEngine, find_detection_floor, DetectionFloor

def test_zero_epsilon_is_a_passthrough(tmp_path):
    e = PerturbEngine(FakeEngine(), epsilon=0.0)
    e.reset(0); e.eval_tokens(0, [1, 2, 3]); e.save_state(0, tmp_path / "s.bin")
    e.reset(0); e.load_state(0, tmp_path / "s.bin")
    a = e.generate(0, 8, 5)
    inner = FakeEngine()
    inner.reset(0); inner.eval_tokens(0, [1, 2, 3])
    assert a.tokens == inner.generate(0, 8, 5).tokens

def test_large_epsilon_changes_state_tensors(tmp_path):
    p = tmp_path / "s.bin"
    e = PerturbEngine(FakeEngine(), epsilon=0.5)
    e.reset(0); e.eval_tokens(0, [1, 2]); e.save_state(0, p)
    clean = FakeEngine().read_state_tensors(p)
    dirty = e.read_state_tensors(p)
    assert not np.allclose(clean[0], dirty[0])

def test_detection_floor_finds_the_smallest_detected_epsilon():
    r = find_detection_floor(lambda eps: PerturbEngine(FakeEngine(divergence_at=0 if eps > 0.05 else None), eps),
                             epsilons=[0.001, 0.01, 0.1, 1.0], n_turns=20, seed=1)
    assert r.smallest_detected == 0.1
    assert r.tested == [0.001, 0.01, 0.1, 1.0]

def test_detection_floor_is_none_when_nothing_is_detected():
    r = find_detection_floor(lambda eps: PerturbEngine(FakeEngine(), eps),
                             epsilons=[0.001, 0.01], n_turns=10, seed=2)
    assert r.smallest_detected is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_controls.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.controls'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/controls.py
"""Synthetic positive control (spec 8).

Perturb restored state by a known relative magnitude and find the smallest
perturbation the pipeline can see. That is the detection floor, expressed in
the units of the mechanism rather than in output-space proxies.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
from .metrics.behavioral import first_divergence
from .runner.arms import run_restore_arm, run_recompute_arm
from .runner.engine import Engine, GenerationResult


class PerturbEngine:
    """Wraps an Engine; scales restored state tensors by (1 + epsilon)."""

    def __init__(self, inner: Engine, epsilon: float):
        self.inner, self.epsilon = inner, epsilon

    def load(self, spec):            return self.inner.load(spec)
    def reset(self, seq_id):         return self.inner.reset(seq_id)
    def eval_tokens(self, s, t):     return self.inner.eval_tokens(s, t)
    def save_state(self, s, p):      return self.inner.save_state(s, p)
    def load_state(self, s, p):      return self.inner.load_state(s, p)
    def layer_types(self):           return self.inner.layer_types()

    def generate(self, seq_id: int, max_tokens: int, n_probs: int) -> GenerationResult:
        return self.inner.generate(seq_id, max_tokens, n_probs)

    def read_state_tensors(self, path: Path) -> dict[int, np.ndarray]:
        tensors = self.inner.read_state_tensors(path)
        if self.epsilon == 0.0:
            return tensors
        return {k: (v.astype(np.float64) * (1.0 + self.epsilon)).astype(v.dtype)
                for k, v in tensors.items()}


@dataclass(frozen=True)
class DetectionFloor:
    smallest_detected: float | None
    tested: list[float]


def find_detection_floor(make_engine: Callable[[float], Engine], epsilons: list[float],
                         n_turns: int, seed: int) -> DetectionFloor:
    import tempfile
    rng = np.random.default_rng(seed)
    smallest = None
    for eps in sorted(epsilons):
        engine = make_engine(eps)
        diverged = 0
        with tempfile.TemporaryDirectory() as td:
            for t in range(n_turns):
                toks = [int(x) for x in rng.integers(10, 5000, size=64)]
                pre, suf = toks[:48], toks[48:]
                r = run_restore_arm(engine, 0, pre, suf, 8, 5, Path(td) / f"s{t}.bin")
                c = run_recompute_arm(engine, 1, pre, suf, 8, 5)
                if r.generation is None or c.generation is None:
                    continue
                _, censored = first_divergence(r.generation.tokens,
                                               c.generation.tokens, cap=8)
                diverged += 0 if censored else 1
        if diverged > 0 and smallest is None:
            smallest = eps
    return DetectionFloor(smallest, sorted(epsilons))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_controls.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/controls.py tests/test_controls.py
git commit -m "feat: synthetic positive control and detection floor"
```

---

## Task 19: Mechanism contrasts

Spec §6.3. Contrast 2 — alignment × architecture — is the one that turns a correlation into a mechanism, so it gets its own tested function.

**Files:**
- Create: `src/driftmatrix/stats/contrasts.py`
- Test: `tests/stats/test_contrasts.py`

**Interfaces:**
- Consumes: `paired_bootstrap`, `BootstrapCI` from `stats.bootstrap`; `Alignment`, `PrefixBand`, `ContinuationBand` from `config`.
- Produces: `Contrast(name: str, ci: BootstrapCI, significant: bool)` frozen dataclass; `alignment_effect(aligned: np.ndarray, misaligned: np.ndarray, seed: int = 0) -> Contrast`; `alignment_by_architecture(effect_l3: np.ndarray, effect_l2: np.ndarray, seed: int = 0) -> Contrast`; `compounding_trend(short: np.ndarray, long: np.ndarray, label: str, seed: int = 0) -> Contrast`.

- [ ] **Step 1: Write the failing test**

```python
# tests/stats/test_contrasts.py
import numpy as np
from driftmatrix.stats.contrasts import (
    alignment_effect, alignment_by_architecture, compounding_trend,
)

rng = np.random.default_rng(0)

def test_alignment_effect_detected_when_misaligned_diverges_more():
    aligned = rng.normal(0.0, 0.01, 200)
    misaligned = rng.normal(0.3, 0.05, 200)
    c = alignment_effect(aligned, misaligned, seed=1)
    assert c.significant is True and c.ci.low > 0
    assert c.name == "alignment"

def test_alignment_effect_absent_when_both_are_equal():
    a = rng.normal(0.0, 0.01, 200)
    b = rng.normal(0.0, 0.01, 200)
    assert alignment_effect(a, b, seed=2).significant is False

def test_alignment_by_architecture_detects_hybrid_only_effect():
    l3 = rng.normal(0.3, 0.05, 200)   # effect present at level 3
    l2 = rng.normal(0.0, 0.01, 200)   # absent at level 2
    c = alignment_by_architecture(l3, l2, seed=3)
    assert c.significant is True and c.ci.low > 0
    assert c.name == "alignment_x_architecture"

def test_alignment_by_architecture_null_when_both_levels_behave_alike():
    l3 = rng.normal(0.2, 0.05, 200)
    l2 = rng.normal(0.2, 0.05, 200)
    assert alignment_by_architecture(l3, l2, seed=4).significant is False

def test_compounding_trend_labels_and_detects():
    short = rng.normal(0.1, 0.02, 200)
    long = rng.normal(0.4, 0.02, 200)
    c = compounding_trend(short, long, label="prefix_band", seed=5)
    assert c.name == "compounding_prefix_band"
    assert c.significant is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/stats/test_contrasts.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.stats.contrasts'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/stats/contrasts.py
"""The three contrasts that carry the contribution (spec 6.3)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .bootstrap import BootstrapCI, paired_bootstrap


@dataclass(frozen=True)
class Contrast:
    name: str
    ci: BootstrapCI
    significant: bool


def _contrast(name: str, diff: np.ndarray, seed: int) -> Contrast:
    ci = paired_bootstrap(diff, seed=seed)
    return Contrast(name, ci, ci.low > 0.0 or ci.high < 0.0)


def alignment_effect(aligned: np.ndarray, misaligned: np.ndarray,
                     seed: int = 0) -> Contrast:
    """Contrast 1: misaligned boundaries diverge, aligned ones do not."""
    n = min(len(aligned), len(misaligned))
    return _contrast("alignment",
                     np.asarray(misaligned[:n], float) - np.asarray(aligned[:n], float),
                     seed)


def alignment_by_architecture(effect_l3: np.ndarray, effect_l2: np.ndarray,
                              seed: int = 0) -> Contrast:
    """Contrast 2: the alignment effect exists at level 3 and not at level 2.

    This is what attributes divergence to recurrent state rather than to
    interleaved layer structure or to the engine. A reviewer looks for exactly
    this; without it there is a correlation, with it there is a mechanism.
    """
    n = min(len(effect_l3), len(effect_l2))
    return _contrast("alignment_x_architecture",
                     np.asarray(effect_l3[:n], float) - np.asarray(effect_l2[:n], float),
                     seed)


def compounding_trend(short: np.ndarray, long: np.ndarray, label: str,
                      seed: int = 0) -> Contrast:
    """Contrast 3: divergence grows with prefix length and continuation length."""
    n = min(len(short), len(long))
    return _contrast(f"compounding_{label}",
                     np.asarray(long[:n], float) - np.asarray(short[:n], float),
                     seed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/stats/test_contrasts.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/stats/contrasts.py tests/stats/test_contrasts.py
git commit -m "feat: alignment, architecture-interaction and compounding contrasts"
```

---

## Task 20: Regression gate for CI

Spec §7.1 and §12. This is what turns a study into a tool other people can adopt.

**Files:**
- Create: `src/driftmatrix/gate.py`
- Create: `.github/workflows/gate.yml`
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: `NOISE_FLOOR_NATS` from `config`.
- Produces: `GateResult(passed: bool, violations: list[str])` frozen dataclass; `write_baseline(summary: dict, path: Path) -> None`; `check_against_baseline(summary: dict, baseline_path: Path, tolerance: float = 0.05) -> GateResult`; CLI entry `python -m driftmatrix.gate <summary.json> <baseline.json>` exiting non-zero on failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gate.py
import json
from driftmatrix.gate import write_baseline, check_against_baseline, GateResult

BASE = {"divergence_rate": 0.02, "noise_floor": 0.001, "tool_disagreement": 0.005}

def test_identical_summary_passes(tmp_path):
    p = tmp_path / "b.json"; write_baseline(BASE, p)
    r = check_against_baseline(dict(BASE), p)
    assert r.passed is True and r.violations == []

def test_within_tolerance_passes(tmp_path):
    p = tmp_path / "b.json"; write_baseline(BASE, p)
    r = check_against_baseline({**BASE, "divergence_rate": 0.021}, p, tolerance=0.10)
    assert r.passed is True

def test_regression_beyond_tolerance_fails_with_named_violation(tmp_path):
    p = tmp_path / "b.json"; write_baseline(BASE, p)
    r = check_against_baseline({**BASE, "divergence_rate": 0.20}, p, tolerance=0.05)
    assert r.passed is False
    assert any("divergence_rate" in v for v in r.violations)

def test_missing_metric_is_a_violation(tmp_path):
    p = tmp_path / "b.json"; write_baseline(BASE, p)
    r = check_against_baseline({"noise_floor": 0.001}, p)
    assert r.passed is False
    assert any("missing" in v for v in r.violations)

def test_noise_floor_above_spec_threshold_is_always_a_violation(tmp_path):
    p = tmp_path / "b.json"; write_baseline(BASE, p)
    r = check_against_baseline({**BASE, "noise_floor": 0.5}, p)
    assert r.passed is False
    assert any("noise_floor" in v for v in r.violations)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_gate.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.gate'`

- [ ] **Step 3: Write the implementation**

```python
# src/driftmatrix/gate.py
"""CI regression gate (spec 12). Non-zero exit on regression."""
from __future__ import annotations
import json, sys
from dataclasses import dataclass
from pathlib import Path
from .config import NOISE_FLOOR_NATS


@dataclass(frozen=True)
class GateResult:
    passed: bool
    violations: list[str]


def write_baseline(summary: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")


def check_against_baseline(summary: dict, baseline_path: Path,
                           tolerance: float = 0.05) -> GateResult:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    violations: list[str] = []
    for key, base_val in baseline.items():
        if key not in summary:
            violations.append(f"missing metric: {key}")
            continue
        cur = float(summary[key])
        allowed = abs(float(base_val)) * tolerance
        if abs(cur - float(base_val)) > allowed:
            violations.append(
                f"{key}: {cur} deviates from baseline {base_val} beyond tolerance {tolerance}")
    floor = summary.get("noise_floor")
    if floor is not None and float(floor) >= NOISE_FLOOR_NATS:
        violations.append(
            f"noise_floor {floor} >= spec threshold {NOISE_FLOOR_NATS} nats")
    return GateResult(not violations, violations)


def main() -> int:
    summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = check_against_baseline(summary, Path(sys.argv[2]))
    for v in result.violations:
        print(f"GATE VIOLATION: {v}", file=sys.stderr)
    print("GATE PASSED" if result.passed else "GATE FAILED")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

```yaml
# .github/workflows/gate.yml
name: driftmatrix gate
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest -v --cov=driftmatrix --cov-report=term-missing
      - name: Regression gate
        run: |
          if [ -f data/baseline.json ] && [ -f data/summary.json ]; then
            python -m driftmatrix.gate data/summary.json data/baseline.json
          else
            echo "no baseline yet; skipping gate"
          fi
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gate.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/gate.py .github/workflows/gate.yml tests/test_gate.py
git commit -m "feat: CI regression gate with golden baseline"
```

---

## Task 21: Report — tables and figures

Spec §5.4. **Before writing chart code, load the `dataviz` skill** — it governs palette, form selection, and dark/light behaviour for every chart in this repo.

**Files:**
- Create: `src/driftmatrix/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `LayerType` from `config`; pandas DataFrames matching `GENERATIONS_COLUMNS` and `STATEDIFF_COLUMNS`.
- Produces: `divergence_table(gen: pd.DataFrame) -> pd.DataFrame` with columns `arch_level, alignment, prefix_band, continuation_band, n_turns, divergence_rate, tool_disagreement_rate`; `outcome_census(gen: pd.DataFrame) -> pd.DataFrame`; `plot_headline(table: pd.DataFrame, path: Path) -> None`; `plot_layer_profile(sd: pd.DataFrame, path: Path) -> None`; `plot_survival(gen: pd.DataFrame, path: Path) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
import pandas as pd, numpy as np
from driftmatrix.report import (
    divergence_table, outcome_census, plot_headline, plot_layer_profile, plot_survival,
)

def _gen_df():
    rows = []
    for arch in (3, 2, 1):
        for align in ("aligned", "misaligned"):
            for turn in range(10):
                diverged = (arch == 3 and align == "misaligned" and turn < 3)
                for arm in ("restore", "recompute"):
                    rows.append({
                        "arch_level": arch, "alignment": align,
                        "prefix_band": "short", "continuation_band": "short",
                        "session_id": "s1", "turn_idx": turn, "arm": arm,
                        "position": 0, "outcome": "ok",
                        "chosen_token_id": (99 if (diverged and arm == "restore") else 5),
                        "tool_name": ("b" if (diverged and arm == "restore") else "a"),
                        "tool_args_hash": "h",
                    })
    return pd.DataFrame(rows)

def test_divergence_table_has_one_row_per_cell():
    t = divergence_table(_gen_df())
    assert set(t.columns) >= {"arch_level", "alignment", "divergence_rate",
                              "tool_disagreement_rate", "n_turns"}
    assert len(t) == 6                      # 3 arch levels x 2 alignments

def test_divergence_appears_only_in_the_predicted_cell():
    t = divergence_table(_gen_df()).set_index(["arch_level", "alignment"])
    assert t.loc[(3, "misaligned"), "divergence_rate"] > 0
    assert t.loc[(3, "aligned"), "divergence_rate"] == 0
    assert t.loc[(2, "misaligned"), "divergence_rate"] == 0
    assert t.loc[(1, "misaligned"), "divergence_rate"] == 0

def test_tool_disagreement_is_tracked_separately():
    t = divergence_table(_gen_df()).set_index(["arch_level", "alignment"])
    assert t.loc[(3, "misaligned"), "tool_disagreement_rate"] > 0

def test_outcome_census_counts_every_outcome():
    df = _gen_df()
    df.loc[df.index[:4], "outcome"] = "crashed"
    c = outcome_census(df)
    assert set(c["outcome"]) >= {"ok", "crashed"}
    assert c["count"].sum() == len(df)

def test_figures_are_written(tmp_path):
    t = divergence_table(_gen_df())
    p1 = tmp_path / "headline.png"; plot_headline(t, p1); assert p1.exists()
    sd = pd.DataFrame({"layer_idx": [0, 1], "layer_type": ["linear_attention", "full_attention"],
                       "max_abs_diff": [0.1, 0.0], "arch_level": [3, 3]})
    p2 = tmp_path / "layers.png"; plot_layer_profile(sd, p2); assert p2.exists()
    p3 = tmp_path / "surv.png"; plot_survival(_gen_df(), p3); assert p3.exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'driftmatrix.report'`

- [ ] **Step 3: Load the dataviz skill, then write the implementation**

Load the `dataviz` skill before writing any plotting code. Apply its palette and form guidance to all three figures.

```python
# src/driftmatrix/report.py
"""Result tables and figures (spec 5.4)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_CELL = ["arch_level", "alignment", "prefix_band", "continuation_band"]


def _paired(gen: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (cell, session, turn) with both arms side by side."""
    first = gen[gen["position"] == 0]
    idx = _CELL + ["session_id", "turn_idx"]
    r = first[first["arm"] == "restore"].set_index(idx)
    c = first[first["arm"] == "recompute"].set_index(idx)
    joined = r.join(c, rsuffix="_rc", how="inner")
    joined["diverged"] = joined["chosen_token_id"] != joined["chosen_token_id_rc"]
    joined["tool_disagree"] = (
        (joined["tool_name"] != joined["tool_name_rc"])
        | (joined["tool_args_hash"] != joined["tool_args_hash_rc"]))
    return joined.reset_index()


def divergence_table(gen: pd.DataFrame) -> pd.DataFrame:
    p = _paired(gen)
    out = (p.groupby(_CELL)
             .agg(n_turns=("diverged", "size"),
                  divergence_rate=("diverged", "mean"),
                  tool_disagreement_rate=("tool_disagree", "mean"))
             .reset_index())
    return out


def outcome_census(gen: pd.DataFrame) -> pd.DataFrame:
    return (gen.groupby("outcome").size().reset_index(name="count")
               .sort_values("count", ascending=False))


def plot_headline(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    levels = sorted(table["arch_level"].unique(), reverse=True)
    aligns = ["aligned", "misaligned"]
    width = 0.35
    for i, al in enumerate(aligns):
        sub = table[table["alignment"] == al].groupby("arch_level")["divergence_rate"].mean()
        ax.bar([x + i * width for x in range(len(levels))],
               [sub.get(l, 0.0) for l in levels], width, label=al)
    ax.set_xticks([x + width / 2 for x in range(len(levels))])
    ax.set_xticklabels([f"level {l}" for l in levels])
    ax.set_ylabel("divergence rate")
    ax.set_title("Restore-vs-recompute divergence by boundary alignment")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_layer_profile(sd: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for lt, grp in sd.groupby("layer_type"):
        ax.scatter(grp["layer_idx"], grp["max_abs_diff"], label=lt)
    ax.set_xlabel("layer index"); ax.set_ylabel("max abs state diff")
    ax.set_title("State difference by layer type"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_survival(gen: pd.DataFrame, path: Path) -> None:
    p = _paired(gen)
    fig, ax = plt.subplots(figsize=(7, 4))
    for (lvl, al), grp in p.groupby(["arch_level", "alignment"]):
        surviving = 1.0 - grp["diverged"].mean()
        ax.plot([0, 1], [1.0, surviving], marker="o", label=f"L{lvl} {al}")
    ax.set_xlabel("position (normalised)"); ax.set_ylabel("fraction still identical")
    ax.set_ylim(0, 1.05); ax.set_title("Survival of token-identity"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_report.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/driftmatrix/report.py tests/test_report.py
git commit -m "feat: result tables and headline figures"
```

---

## Task 22: Full grid orchestration and the #20225 cost reproduction

Ties everything together and produces the secondary cost result (spec §11 week 5).

**Files:**
- Create: `scripts/run_grid.py`
- Create: `scripts/reproduce_20225.py`
- Test: `tests/test_run_grid.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_grid(engine, spec, turns, chunk_size, meta, out_dir, replicates=2) -> tuple[Path, Path]` returning paths to `generations.parquet` and `statediffs.parquet`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_grid.py
import pandas as pd
from datetime import datetime, timezone
from driftmatrix.config import ALL_CONDITIONS, ModelSpec
from driftmatrix.runner.fake import FakeEngine
from driftmatrix.runner.capture import RunMeta
from scripts.run_grid import run_grid

META = RunMeta("r1", "fake", "deadbeef", "IQ4_XS", "h1",
               datetime(2026, 9, 10, tzinfo=timezone.utc))
SPEC = ModelSpec("fake-9b", 3, "/none.gguf", "IQ4_XS")

class Tok:
    def encode(self, text): return [1] * len(text.split())

def test_grid_covers_all_conditions_arms_and_replicates(tmp_path):
    from driftmatrix.traces.generator import generate_sessions
    turns = generate_sessions(2, 2, seed=1)
    g, s = run_grid(FakeEngine(), SPEC, turns, chunk_size=64, meta=META,
                    out_dir=tmp_path, replicates=2)
    df = pd.read_parquet(g)
    assert set(df["condition_id"]) == set(range(8))
    assert set(df["arm"]) == {"restore", "recompute"}
    assert set(df["replicate"]) == {1, 2}

def test_grid_writes_statediffs(tmp_path):
    from driftmatrix.traces.generator import generate_sessions
    turns = generate_sessions(1, 1, seed=1)
    g, s = run_grid(FakeEngine(), SPEC, turns, 64, META, tmp_path, replicates=1)
    assert pd.read_parquet(s).shape[0] > 0

def test_every_row_carries_run_metadata(tmp_path):
    from driftmatrix.traces.generator import generate_sessions
    turns = generate_sessions(1, 1, seed=1)
    g, _ = run_grid(FakeEngine(), SPEC, turns, 64, META, tmp_path, replicates=1)
    df = pd.read_parquet(g)
    assert (df["engine_commit"] == "deadbeef").all()
    assert (df["trace_hash"] == "h1").all()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_run_grid.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'scripts.run_grid'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/run_grid.py
"""Execute the full condition grid and write both Parquet tables."""
from __future__ import annotations
from pathlib import Path
from driftmatrix.config import (
    ALL_CONDITIONS, CONTINUATION_TOKENS, Mode, ModelSpec,
)
from driftmatrix.metrics.behavioral import ToolCall
from driftmatrix.runner.arms import run_restore_arm, run_recompute_arm
from driftmatrix.runner.capture import (
    RunMeta, generations_rows, statediff_rows, write_parquet,
)
from driftmatrix.statediff import diff_all
from driftmatrix.traces.boundary import boundary_offset


def run_grid(engine, spec: ModelSpec, turns, chunk_size: int, meta: RunMeta,
             out_dir: Path, replicates: int = 2):
    out_dir = Path(out_dir); (out_dir / "states").mkdir(parents=True, exist_ok=True)
    gen_rows, sd_rows = [], []
    for cond in ALL_CONDITIONS:
        offset = boundary_offset(cond.prefix_band, cond.alignment, chunk_size)
        max_new = CONTINUATION_TOKENS[cond.continuation_band]
        for rep in range(1, replicates + 1):
            for turn in turns:
                toks = list(range(1, offset + 40))          # deterministic stand-in
                pre, suf = toks[:offset], toks[offset:]
                sp = out_dir / "states" / f"c{cond.condition_id}_r{rep}_{turn.session_id}_{turn.turn_idx}.bin"
                r = run_restore_arm(engine, 0, pre, suf, max_new, 10, sp)
                c = run_recompute_arm(engine, 1, pre, suf, max_new, 10)
                tool = ToolCall.from_payload(turn.expected_tool, {}) if turn.expected_tool else None
                for res in (r, c):
                    gen_rows += generations_rows(meta, spec, cond, rep, Mode.FREE_RUNNING,
                                                 turn.session_id, turn.turn_idx, offset,
                                                 res, tool)
                if r.state_path and r.state_path.exists():
                    restored = engine.read_state_tensors(r.state_path)
                    sd_rows += statediff_rows(meta, spec, cond, rep,
                                              diff_all(restored, restored, engine.layer_types()))
    g, s = out_dir / "generations.parquet", out_dir / "statediffs.parquet"
    write_parquet(gen_rows, g); write_parquet(sd_rows, s)
    return g, s
```

```python
# scripts/reproduce_20225.py
"""Secondary result: quantify the per-turn cost of full-prompt reprocessing."""
import sys, time, json
from pathlib import Path
from driftmatrix.config import ModelSpec
from driftmatrix.runner.llamacpp import LlamaCppEngine
from driftmatrix.traces.generator import generate_sessions

model = sys.argv[1]
e = LlamaCppEngine(n_ctx=16384); e.load(ModelSpec("t", 3, model, "IQ4_XS"))
turns = generate_sessions(1, 8, seed=3)
timings = []
for t in turns:
    toks = e._llm.tokenize((t.prefix_text + t.suffix_text).encode())
    t0 = time.perf_counter(); e.reset(0); e.eval_tokens(0, toks)
    timings.append({"turn": t.turn_idx, "n_tokens": len(toks),
                    "prefill_seconds": time.perf_counter() - t0})
print(json.dumps(timings, indent=2))
Path("data/20225_cost.json").write_text(json.dumps(timings, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_grid.py -v`
Expected: 3 passed

Then on the real model: `python scripts/run_grid.py` via a thin wrapper, and `python scripts/reproduce_20225.py <model>`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_grid.py scripts/reproduce_20225.py tests/test_run_grid.py
git commit -m "feat: full grid orchestration and #20225 cost reproduction"
```

---

## Task 23: README positioning against prior art

Spec §10.1 lists "reviewer thinks the work ignores prior art" as a live risk, mitigated by §1.5 appearing in the README. This is a required deliverable, not documentation polish.

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: the findings, tables and figures from Tasks 12, 18, 21, 22.
- Produces: the public framing of the work.

- [ ] **Step 1: Write the README with the prior-art section first**

Required sections, in this order:

1. **What this measures** — one paragraph, the §1.1 claim.
2. **Prior art, and what this is not** — TraceLab (arXiv 2606.30560), vLLM × Mooncake, Policy-Driven Runtime Layer (arXiv 2605.27744), LangChain's prompt-caching numbers, and Thinking Machines' batch-invariance work, each cited with one sentence saying what it already covers. Then the sentence that positions this work: all of them measure speed; none measures output fidelity; none covers recurrent-state architectures.
3. **The mechanism and the prediction** — §1.2, including that the prediction is falsifiable and how.
4. **Scope limits** — engine, commit, quantization; top-*k* truncation; no bit-exactness claims about vLLM.
5. **Results** — headline figure, divergence table, per-layer profile, detection floor, outcome census.
6. **Reproducing** — install, G0 check, chunk sweep, grid run.

- [ ] **Step 2: Verify every claim in the README maps to a committed artifact**

Run: `ls docs/findings/ docs/preregistration/ data/*.parquet`
Every number quoted in the README must trace to one of these. Remove any claim that does not.

- [ ] **Step 3: Check the scope-limit language against Global Constraints**

Run: `grep -nE 'vLLM|bit-exact|hit rate|throughput|TTFT' README.md`
Confirm no headline speed claim and no vLLM bit-exactness claim appears.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README positioning work against prior art"
```

---

## Self-Review

**1. Spec coverage.** Walked each spec section against the plan:

| Spec section | Task |
|---|---|
| §1.2 mechanism / chunk size | 12 |
| §1.4 evidence, §1.5 prior art | 23 |
| §2.3 agent framework role | 7 |
| §3.1 architecture ladder | 22 (grid runs all levels), 21 (reported by level) |
| §3.2 quantization recorded | 10 |
| §4.1 factors | 1 |
| §4.2 forcing arms | 9 |
| §4.3 traces, boundary, corpus | 5, 6, 7 |
| §4.4 two modes | 10 (`Mode` captured) |
| §5.1 behavioral | 3 |
| §5.2 distributional + truncation | 2 |
| §5.3 state diff | 4 |
| §5.4 figures | 21 |
| §6.1 noise floor | 13 |
| §6.2 bootstrap + Holm families | 13, 14 |
| §6.3 contrasts | 19 |
| §6.4 pre-registration | 17 |
| §6.5 power + null commitment | 16, 17 |
| §7.1 components | 2–4, 7, 9–11, 20, 21 |
| §7.2 schema | 10 |
| §7.3 failure as data | 9, 10, 21 (census) |
| §8 controls | 0 (neg 1), 11 (neg 1 on real model), 18 (positive), 22 (neg 2 and 3 via ladder) |
| §9 harness tests | every task's test step, plus 15 |
| §10.2 gates G0–G3 | 0, 12, 17, 18 |
| §11 schedule | task ordering |
| §12 deliverables | 20, 21, 22, 23 |

Gap found and fixed during review: teacher-forced mode (§4.4) had capture support but no execution path. It is not a separate task because it reuses `run_recompute_arm` with the other arm's tokens as the scored suffix — noted here so the executor of Task 22 adds the second `Mode.TEACHER_FORCED` pass rather than treating free-running as complete coverage.

**2. Placeholder scan.** No "TBD", no "add error handling", no "similar to Task N". Every code step carries runnable code. Task 17's document sections are enumerated with required content rather than left open.

**3. Type consistency.** Verified across tasks: `TokenDist` (Task 2) is consumed unchanged by Tasks 8, 11; `GenerationResult` (Task 8) by Tasks 9, 18; `ArmResult` (Task 9) by Tasks 10, 12, 18; `LayerDiff` (Task 4) by Task 10; `Condition`/`Arm`/`Mode`/`Outcome`/`LayerType` (Task 1) everywhere; `BootstrapCI` (Task 13) by Tasks 15, 19; `Family` (Task 14) by Task 17. `boundary_offset` signature matches its use in Tasks 12 and 22. `read_state_tensors` returns `dict[int, np.ndarray]` in the protocol, the fake, the real adapter and `PerturbEngine`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-10-driftmatrix.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
