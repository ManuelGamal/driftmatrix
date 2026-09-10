# driftmatrix — Design Spec

**Does restoring saved state instead of recomputing change what the model says?**
A state-fidelity audit for hybrid linear-attention models under agent-shaped traffic.

Date: 2026-09-10
Status: design approved, pending spec review
Author: Manuel Gamal

---

## 1. The question

Serving stacks avoid recomputation by saving model state and restoring it later.
On a dense transformer that state is a KV cache: one K/V pair per token, each
computed independently of the others, sliceable at token granularity, and exact
to reuse.

Hybrid models break that. In Qwen3.5-class architectures, 24 of 32 layers are
`linear_attention` layers carrying a **fixed-size recurrent state** that is
rewritten at every token. It cannot be sliced — only snapshotted whole and
restored whole.

**This project measures whether restoring it is faithful.**

### 1.1 The claim under test

> For an identical agent turn, does restoring saved recurrent state produce
> different output than recomputing the prompt from scratch — and does the
> difference change which tool the agent calls?

The second clause is what matters. A shifted logprob is a curiosity. A changed
tool call is a behavioral change caused by an optimization meant to be invisible.

### 1.2 The mechanism, and the falsifiable prediction it yields

Restoring a saved state returns *bit-identical* bytes. So divergence cannot come
from the restore itself. It comes from the fact that **the two arms segment the
prompt differently.**

Let the prompt be `[prefix | suffix]`, with the save boundary between them.

- **Recompute arm** processes `prefix + suffix` in one pass. The chunked
  linear-attention scan tiles the whole sequence, and a chunk may **straddle the
  boundary**.
- **Restore arm** processed `prefix` alone (chunks ending exactly at the
  boundary), saved, restored, and now processes `suffix` as a fresh pass
  (chunking restarts at the boundary).

Different chunk boundaries mean different reduction orders. Floating-point
addition is not associative, so the two produce different numbers.

Why that matters only for recurrent layers:

| | Dense / full attention | `linear_attention` |
|---|---|---|
| Is a stored value a cross-token reduction? | **No.** `K_i = W_k x_i` depends on token *i* alone | **Yes.** `S_i = f(S_{i−1}, k_i, v_i)` folds every prior token into one matrix |
| Does chunking change the stored value? | No — per-token K/V are chunk-invariant | **Yes** — the reduction order over tokens changes the state |
| Where does arithmetic error go? | Stays local to one token's attention output; softmax averaging damps it | Lands **in the state**, then carried forward and re-transformed at every later token — it can **compound** |

This yields a sharp, falsifiable prediction:

> **If the save boundary is aligned to the linear-attention kernel's chunk size,
> both arms tile identically and output should be bit-exact. If it is
> misaligned, the arms tile differently and divergence should appear — on hybrid
> models only, never on dense ones.**

That prediction is the intellectual core of the project. It is testable, it can
be wrong, and either outcome is informative. It also explains why boundary
geometry appears as a factor at all (§4.1) rather than being an arbitrary knob.

The kernel's effective chunk size is implementation-defined and **must be
determined empirically** in week 1 (§11), by sweeping boundary offsets and
locating the periodicity of divergence.

### 1.3 Two different caches — precisely which one is under test

Both are routinely called "the KV cache." Conflating them makes this design
unreadable.

**(a) The within-request decode cache — NOT under test.** While generating token
by token, the engine retains prior tokens' state so it does not re-run attention
over the whole sequence per new token. Mandatory, always on, present in every
engine. **It stays fully enabled on both arms**, including the recompute arm —
that arm prefills from scratch and then decodes normally, like any request.

**(b) Saved state reused across requests — THE TREATMENT.** When a later request
shares a prompt prefix, the engine can restore state saved earlier rather than
recomputing that span. This is what agent traffic exercises every turn: identical
system prompt, identical tool schemas, monotonically growing history.

Measured hit rates in production agent traffic are 94–96% (TraceLab, arXiv
2606.30560, 4,300 Claude Code/Codex sessions), and vLLM × Mooncake reports
1.7%→92.2% on Codex-derived traces. Reuse is not an edge case; it carries
almost all the tokens. That is why its fidelity is load-bearing.

### 1.4 Evidence, with honest confidence levels

Verified individually against the primary source or the GitHub API.

| Source | What it genuinely supports | Confidence |
|---|---|---|
| **llama.cpp #20225** (open) | Primary anchor for this engine. Hybrid agentic sessions reprocess the entire prompt every turn — 15K tokens, ~8 min/turn — because checkpoint/restore logic **written for sliding-window attention** does not work on recurrent state. That sentence is exactly the level-2-vs-level-3 distinction in §3.1. | **Strong — primary** |
| **LMCache hybrid-model docs** | Motivating claim: states generation is **not bit-exact** between cached and fresh runs on GDN backends, and instructs users to abandon token-level comparison. Quantifies nothing. **Caveat: this is vLLM/LMCache-specific**, so it motivates the question but is not evidence about llama.cpp. | **Strong as motivation, out-of-engine as evidence** |
| **vLLM #43587** (open, reproducing 2026-09-06) | Prefix-cache hit 0% on hybrid Qwen3.5-4B; **does not reproduce on the pure-attention sibling**. Architecture-caused by construction. Cross-engine corroboration that hybrids break reuse. | **Strong — corroboration** |
| **vLLM #40696** (open) | A concrete block-size cliff (528 tokens) with a 0%/95% hit-rate discontinuity — independent evidence that **boundary alignment governs reuse behavior** in hybrids. Motivates factor **B**. | **Strong — motivates the alignment factor** |
| **vLLM #26201** (open tracking, filed 2025-10-03) | Umbrella issue for hybrid prefix caching; live and mid-implementation. | **Strong — context** |
| **SGLang #21846** | Competing stack: HiCache hybrid compatibility "remains limited"; agentic KV-cache scheduling for hybrid/linear models on the Qwen3.5 family is Q3-2026 **in progress, not shipped**. | **Strong — industry-wide corroboration** |
| **vLLM RFC #54506** (open) | Adjacent mechanism: M-dependent kernel arithmetic (split-K reduction order) makes verify-pass and plain-decode non-bit-identical on a hybrid model; measured acceptance 100%→72%. Same floating-point class as differing chunk tiling. | **Moderate — analogous mechanism** |
| **vLLM #45238** (open) | Reports 0% hit when the checkpoint lands in the request-unique suffix. **Required caveat:** the reporter later conceded that case was already fixed on main by #37898/#43447 (40/64 hits, not 0/64); independent corroborations continued through Aug 2026 on 0.26–0.28 builds. | **Moderate — cite for geometry, never as a live 0% claim** |
| **vLLM PRs #45845, #46384** | Both **merged**. #46384 adds token-granularity `prefix_match_unit`. Upstream is actively moving — pin versions. | **Strong — forces pinning** |

### 1.5 Prior art — what this project must not pretend to be

**Agent-workload characterization on dense models is crowded.** TraceLab (arXiv
2606.30560), vLLM × Mooncake, and a Policy-Driven Runtime Layer paper (arXiv
2605.27744, +13–37pp hit rate at the engine tier) all cover it. Redoing it is not
a contribution.

**Cache-aware prompt ordering is commoditized.** LangChain published the numbers
(49–80% token cost reduction; "stable content first, volatile last").

**A cache-aware LangGraph/CrewAI runtime layer is scooped**, and scooped deeper,
by arXiv 2605.27744 at the engine tier. Not attempted.

What all of them share: they measure **speed** — hit rate, TTFT, throughput. Not
one measures output fidelity, and none covers recurrent-state architectures.
Mooncake lists hybrid cache offloading as future work. That absence is this
project's territory.

---

## 2. Scope

### 2.1 In scope

- One engine: **llama.cpp**, pinned to a commit, driven via explicit sequence
  state save/restore.
- A three-level architecture ladder (§3.1), all GGUF, all on owned hardware.
- Restore-arm vs recompute-arm comparison under controlled boundary alignment.
- Agent-shaped traces generated with LangGraph.
- A `pip`-installable harness plus a written study.
- Stretch: a vLLM cross-check on a small hybrid, to show the effect is not a
  llama.cpp artifact.

### 2.2 Out of scope

- **Speed, hit rate, TTFT, throughput as headline results.** Crowded (§1.5).
  Cost appears only as the secondary #20225 reproduction.
- **A cache-aware agent runtime or router.** Scooped (§1.5).
- **Prompt-ordering guidance.** Commoditized (§1.5).
- **Engine-internal fixes** to llama.cpp's chunked scan or state serialization —
  C++/CUDA scope beyond this project. Findings are filed upstream; fixes are not
  promised.
- **Batch invariance** — solved by Thinking Machines' work; cited, never claimed.
- **Long-context reasoning collapse** and **state quantization** — the other two
  stressors on the same instrument, deliberately deferred.
- **Bit-exactness claims about vLLM.** The LMCache disclaimer motivates the
  question; this study measures llama.cpp and says so.

### 2.3 The role of the agent framework — stated honestly

LangGraph is the **traffic generator**, not the contribution: stable system
prompt, stable tool schemas, monotonically growing history, deterministic mock
tools, real tool-call round trips. Hand-rolling that shape would be less
credible than using what practitioners run.

An optional week-6 deliverable is a thin "state-risk" reporter surfacing findings
at the framework level. It is labelled a demonstration.

---

## 3. Models and hardware

### 3.1 The architecture ladder

Three levels, one variable: **what kind of state gets saved.**

| Level | Model | Attention mix | State type | Prediction |
|---|---|---|---|---|
| **3 — treatment** | **Ornith-1.5-9B, IQ4_XS** (`qwen3_5`, 32 layers, 3×`linear_attention` + 1×`full_attention`, `full_attention_interval: 4`) | linear + full | **recurrent, non-sliceable** | Divergence when boundary is misaligned |
| **2 — intermediate control** | **Spark-X2.5-4B GGUF** (`spark2_5`, 36 layers, 3×`sliding_attention` + 1×`full_attention`, `sliding_window: 512`) | sliding + full | per-token KV, bounded | **No divergence** — hybrid in structure, conventional in state |
| **1 — baseline** | Qwen3-8B (or 4B) GGUF | full only | per-token KV | **No divergence** — bit-exact by construction |

Level 2 is what makes the finding mechanistic rather than correlational. With only
a dense baseline, a reviewer can object: *"maybe it's interleaved layer structure
in general, not recurrent state."* Level 2 has the same interleaved structure and
a conventional state type. If divergence appears at level 3 and not level 2, the
mechanism is isolated to the recurrent state.

This is also precisely the distinction llama.cpp #20225 draws — checkpoint logic
"written for sliding-window attention" failing on recurrent state. The ladder
turns that sentence into an experiment.

**Naming discipline:** Ornith-1.5-9B is a Qwen3.5-architecture model
(`architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"`),
independently confirmed from its `config.json`. The config's own vocabulary is
`linear_attention`, not `gated_delta`; the GDN naming comes from the Qwen3.5
lineage and its papers. Artifacts use the config's terms.

Ornith-1.5-9B is additionally **natively multimodal** (`vision_config`,
`model_type: qwen3_5_vision`). Multimodal turns are a **stretch** condition, not
part of the core grid — vLLM #43587 shows multimodal and hybrid interacting
badly, which makes it interesting but adds a variable.

### 3.2 Quantization is constant, and that is the point

All three levels run quantized GGUF. Because **both arms use identical weights**,
quantization cannot confound the contrast — it slightly raises the noise floor
and nothing more. Findings are reported as "at IQ4_XS" (~4.25 bits/weight),
which is how these models are actually run locally.

Note the runtime recurrent state is typically kept at higher precision than the
weights; its actual dtype is recorded in week 1 and reported, since it bounds how
much arithmetic slack exists.

### 3.3 Hardware

- **Local: RTX 4060, assumed 8GB.** Ornith-9B at IQ4_XS is ~4.7–4.9GB, leaving
  real context headroom. Spark-4B ~2.2GB. Both fit. This is the primary machine.
- **Kaggle T4, 16GB, sm75.** Secondary — used for scale-out runs and analysis.
  llama.cpp's CUDA backend supports sm75, so the fused-kernel risk that would
  have blocked vLLM largely evaporates. To be confirmed at G0.

Choosing llama.cpp removes the two hardest gates from the previous design: no
building vLLM from main, no 18GB bf16 checkpoint, no FP8 requirement.

---

## 4. Experiment design

### 4.1 Factors

| ID | Factor | Levels |
|---|---|---|
| **T** | Arm (the treatment) | state **restored** · prompt **recomputed** |
| **B** | Save-boundary alignment | **aligned** to the kernel chunk size · **misaligned** (offset by half a chunk) |
| **P** | Prefix length at save point | short (~1K tok) · long (~8K tok) — how much history is folded into the state |
| **C** | Continuation length after restore | short (32 tok) · long (256 tok) — how far error can compound |
| **A** | Architecture (blocking) | level 3 · level 2 · level 1 (§3.1) |

**B** is the mechanism factor (§1.2) and the primary hypothesis carrier.
**P** and **C** test compounding: error should grow with how much state was
accumulated and how long generation continues after restore.

Within each architecture: 2(B) × 2(P) × 2(C) = 8 conditions, each with both arms,
each replicated twice → the noise floor is measured, not assumed, and
"condition vs itself" traverses the same code path as "restore vs recompute."

### 4.2 Forcing the two arms — exactly, not heuristically

llama.cpp exposes sequence state save/restore directly
(`llama_state_seq_save_file` / `llama_state_seq_load_file`, and over HTTP via
llama-server's `/slots/{id}?action=save|restore` with `--slot-save-path`).

- **Restore arm:** feed `prefix` → save state at the boundary → restore into a
  fresh context → feed `suffix` → generate.
- **Recompute arm:** feed `prefix + suffix` in one pass into a fresh context →
  generate.

Identical tokens, identical weights, greedy decoding, fixed seed. The only
difference is whether the state was restored or recomputed.

This is the design's biggest improvement over the previous version. There is no
heuristic cache to coerce, no metrics-based inference about which path ran, and
no need for a `path_unverified` outcome class. The arm is a property of the
procedure, not something inferred afterwards.

It also means the study **does not depend on automatic prompt-cache reuse
working** — which, per #20225, it may not for hybrids.

### 4.3 Agent-shaped traces

Generated with LangGraph:

| Property | Construction |
|---|---|
| Stable system prompt | Identical across all turns and sessions |
| Stable tool schemas | Fixed set, fixed serialization order |
| Growing history | Monotonic accumulation across turns |
| Tool-call round trips | Real calls to **deterministic** mock tools |
| Boundary placement | Save boundary positioned to satisfy **B**, verified by tokenizer count |
| Prefix length | Padded to satisfy **P** |

Deterministic mock tools are mandatory: a nondeterministic tool would inject
variance indistinguishable from the effect under study.

~60 sessions × ~6 turns ≈ 360 turns, frozen and content-hashed; every result row
carries the trace hash.

### 4.4 Two measurement modes

**Free-running — the headline.** Both arms generate at temperature 0. Records
exact-match, first-divergence position, **selected tool call**, final answer.

**Teacher-forced — the instrument.** Each arm scores the other's output so every
position is compared under identical context, giving attributable per-position
distribution differences.

Teacher-forced measures how much the distribution moved; free-running measures
whether it mattered. Reporting only one is the error this design exists to avoid.

---

## 5. Metrics

### 5.1 Behavioral (primary)

- **Tool-call agreement** — same tool, same arguments? Disagreement is the
  finding that matters.
- **Exact-match rate** of the completion.
- **First-divergence position**, right-censored at the generation cap.
- **Final-answer agreement** on the verifiable subset.

### 5.2 Distributional (supporting)

Per-position **Jensen–Shannon divergence** over the returned top-*k* logprobs —
symmetric, bounded by log 2, defined on differing supports. Reported as per-turn
mean (stable) and max (catches a single catastrophic token the mean hides). Also
total variation distance and top-1 agreement.

**Truncation rule.** Where the engine returns only top-*k* logprobs,
renormalize over the **union** of both arms' top-*k* token sets and assign
remaining mass to one `<other>` bucket. *k* is recorded; llama.cpp's
configurable `n_probs` should permit a larger *k* than vLLM's hard cap of 20,
and the value used is reported.

### 5.3 Direct state comparison (available here, and worth taking)

Because state is saved to a file, the **restored and recomputed state tensors can
be compared directly** — max absolute and relative difference per layer, before
any token is generated.

That is a much stronger instrument than output-level inference. It measures the
mechanism at its source, separates "the state differs" from "the difference
changed the output," and gives a per-layer profile showing whether error
concentrates in `linear_attention` layers as predicted. This was impossible in
the vLLM design and is the second major gain from switching engines.

### 5.4 Headline figure

**Divergence rate by boundary alignment, across the three architecture levels**,
with confidence intervals. The prediction is a visibly non-zero level-3
misaligned bar and near-zero everywhere else. Readable in five seconds.

Secondary: per-layer state-difference profile (§5.3), and survival curves of the
fraction of turns still token-identical at position *k*.

---

## 6. Statistics

### 6.1 Effects measured against noise, never zero

Per turn *p* and condition *c*:

```
effect(c, p) = divergence(restore, recompute) − divergence(recompute₁, recompute₂)
```

The subtracted term is the noise floor from the recompute arm's own two
replicates. The bootstrap runs on that **difference**.

**Self-consistency.** Each condition's two replicates are compared. A condition
is flagged **self-inconsistent** when its median per-turn mean JSD exceeds
**twice** the recompute arm's self-comparison median — a pre-specified
engineering threshold, committed with the pre-registration, revisable only with a
reason recorded in git.

Note that if llama.cpp's CUDA kernels are deterministic for fixed batch shapes,
this floor may be exactly zero, which makes the design *stronger*: any nonzero
divergence is then unambiguous. That must be measured, not assumed.

### 6.2 Tests

- **Paired bootstrap over turns** (10,000 resamples), paired by turn.
- **Holm–Bonferroni** within three declared families — pooling unrelated
  hypotheses would over-correct and hide real effects:

  | Family | Members | Size |
  |---|---|---|
  | A — behavioral divergence rates | 8 conditions × 3 architectures | 24 |
  | B — distributional effects | 8 conditions × 2 statistics, level 3 only | 16 |
  | C — mechanism contrasts | alignment effect; alignment × architecture; P and C compounding trends | 5 |

- **CIs on every effect**, significant or not. Nulls reported with intervals,
  never as a bare "no effect."

### 6.3 The contrasts that carry the contribution

1. **Alignment.** Misaligned boundaries diverge; aligned boundaries are
   bit-exact. This is the mechanism prediction of §1.2.
2. **Alignment × architecture.** The alignment effect exists at level 3 and is
   absent at levels 2 and 1. This is what attributes it to recurrent state
   rather than to hybrid structure or to llama.cpp generally.
3. **Compounding.** Divergence increases with prefix length (**P**) and with
   continuation length (**C**).

Contrast 2 is the one a reviewer will look for. Without it there is a
correlation; with it there is a mechanism.

Additivity is a chosen null model, not a law; interaction claims are stated as
"superadditive relative to an additive null on the JSD scale."

### 6.4 Pre-registration

Analysis plan, family definitions, and thresholds committed to git **at the end
of week 2** — after the noise floor and the empirical chunk size are known,
before any treatment effect is examined.

### 6.5 Power, and designing for a null

A power calculation before the main run fixes the turn count needed to detect a
per-turn divergence rate of **2% at 80% power, α = 0.05**. If 360 turns is
insufficient, turn count rises (turns are cheap; sessions are expensive).
Achieved power is reported alongside the result, so a null carries a stated
detection limit rather than an implied one.

**A null is pre-committed as publishable:** *state restore in llama.cpp is
bit-exact for hybrid models at every boundary alignment tested, bounding
divergence below Y% (95% CI) — so LMCache's non-bit-exactness disclaimer does not
generalize to this engine, and the risk is likely specific to speculative-decode
paths (cf. RFC #54506).* Committing to that in advance removes the incentive to
overclaim later.

Note §5.3 makes a *fully* empty result unlikely: even if outputs never diverge,
the per-layer state comparison yields a real measurement of how exact restore is.

---

## 7. Architecture

### 7.1 Components

Load-bearing decision: **engine work happens once, writes raw outputs, state
diffs, and distributions to Parquet; every other component reads those files.**
Analysis reruns are free, analysis code is testable with no GPU, the harness gets
real CI, and Kaggle can generate data while analysis happens locally.

| Component | Responsibility | Depends on | GPU |
|---|---|---|---|
| `traces` | LangGraph trace generator; boundary placement, padding, tokenizer-verified offsets, content hashing | langgraph, tokenizer | no |
| `runner` | Execute both arms via state save/restore, capture outputs, logprobs, and saved-state tensors → Parquet. **Sole owner of the engine dependency.** | llama-cpp-python / llama-server | yes |
| `statediff` | Compare restored vs recomputed state tensors per layer (§5.3) | numpy | no |
| `metrics` | Parquet → behavioral agreement + per-position divergence | numpy, pandas, pyarrow | no |
| `stats` | Noise floor, paired bootstrap, Holm families, contrasts, power | numpy, scipy | no |
| `gate` | CI mode: compare against golden baseline, non-zero exit on regression | metrics, stats | no |
| `report` | Tables, headline figure, per-layer profile, survival curves | metrics, stats, matplotlib | no |

`runner` is deliberately dumb — execute and store, no analysis. All judgment
lives in `stats`, pure functions over arrays: the part that must be right,
isolated into the part easiest to test.

### 7.2 Data schema

Two tables.

**`generations`** — one row per `(model, condition, replicate, arm, mode, session, turn, position)`:

| Field | Type | Notes |
|---|---|---|
| `model_id`, `arch_level` | string, int8 | 1 / 2 / 3 |
| `condition_id` | int8 | Encodes B, P, C |
| `alignment`, `prefix_band`, `continuation_band` | string | Denormalized factor levels |
| `arm` | string | `restore` \| `recompute` |
| `replicate` | int8 | |
| `mode` | string | `free_running` \| `teacher_forced` |
| `session_id`, `turn_idx`, `position` | string, int16, int16 | |
| `token_ids`, `logprobs` | list[int32], list[float32] | Top-*k*, aligned |
| `chosen_token_id` | int32 | |
| `tool_name`, `tool_args_hash` | string | Primary behavioral outcome |
| `boundary_token_offset` | int32 | Exact save position, tokenizer-counted |
| `engine`, `engine_commit`, `quant` | string | Stamped per row |
| `trace_hash`, `run_id`, `run_started_at` | string, string, timestamp | |

**`statediffs`** — one row per `(model, condition, replicate, layer)`:

| Field | Type | Notes |
|---|---|---|
| `layer_idx` | int16 | |
| `layer_type` | string | `linear_attention` \| `full_attention` \| `sliding_attention` |
| `max_abs_diff`, `max_rel_diff`, `frobenius_norm_diff` | float64 | Restored vs recomputed |
| `state_dtype` | string | Runtime precision (§3.2) |

Sizing: ~360 turns × ~256 positions × 8 conditions × 2 arms × 2 replicates × 2
modes ≈ 6M rows in `generations`; `statediffs` is tiny.

**Placement.** Repo on the Windows filesystem (WSL-visible at `/mnt/c/...`);
Parquet and saved-state files on the **WSL-native** filesystem for I/O
throughput, gitignored.

### 7.3 Failure as data

Every run resolves to a recorded outcome: `ok` · `unsupported` (engine refused
the model or the state API) · `crashed` (with captured error) · `timeout` ·
`state_io_failed`.

`unsupported` and `state_io_failed` are findings. "llama.cpp cannot serialize
recurrent state for this architecture" would be a reportable result — and given
#20225, a plausible one.

---

## 8. Validity controls

**Positive control — synthetic, calibrated.** Restore a state, inject a
perturbation of *known magnitude*, and confirm the harness detects the resulting
divergence. Repeat across magnitudes to establish a **detection floor**: the
smallest state perturbation the pipeline can see. This is stronger than
reproducing someone else's bug, because it calibrates sensitivity in the units of
the mechanism and feeds directly into the power analysis (§6.5). If the harness
cannot see a known perturbation, nothing else it reports is trustworthy.

**Negative control 1 — round-trip identity.** Save state and immediately restore
it with no intervening tokens, then continue. Output must be identical. Failure
here means the state serialization itself is lossy, which is a finding but must
be diagnosed before any alignment claim is made.

**Negative control 2 — the dense baseline (level 1).** Per-token K/V are
chunk-invariant, so restore should be bit-exact at every alignment. A large
level-1 effect means the harness is measuring something other than state
restoration.

**Negative control 3 — aligned boundaries at level 3.** Per §1.2, aligned
boundaries should tile identically and be bit-exact even on the hybrid. This is
the sharpest test of the mechanism story: divergence that appears at *aligned*
boundaries too would falsify the chunk-tiling explanation and demand a different
one.

**Noise floor.** Every condition replicated (§6.1).

**Pinning.** One llama.cpp commit, one quantization, stamped per row.

**Deterministic tools.** Mock tools are pure functions of their arguments.

**Pre-registration** before treatment effects are examined (§6.4).

---

## 9. Testing the harness

- **`metrics`** — synthetic distributions with known answers: identical → exactly
  0, disjoint supports → log 2, hand-computed pairs to tolerance. Union-support
  truncation against worked examples.
- **`stats`** — a **coverage simulation**: inject a synthetic effect of known
  size, verify detection at the claimed rate; run under the null, verify the
  false-positive rate approximates α. Holm verified against a reference
  implementation.
- **`statediff`** — identical tensors → all zeros; tensors differing by a known
  epsilon → that epsilon recovered.
- **`runner`** — fake-engine mock returning canned outputs and state blobs, so
  arm execution, outcome classification, and Parquet writing are testable with no
  GPU.
- **`traces`** — hash stability, and assertions that the save boundary lands at
  the intended token offset (tokenizer-counted, not character-counted).
- **`gate`** — golden-file tests: clean baseline passes, seeded regression fails
  with non-zero exit.

---

## 10. Risks and gates

### 10.1 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **llama.cpp cannot save/restore recurrent state for `qwen3_5`** | High | G0 day-1 check. If unsupported, that *is* a finding (with #20225 as context); re-scope to the cost study plus level-1/2 comparison, or move to vLLM with Qwen3.5-4B-AWQ |
| **No divergence at any alignment** | Medium | Pre-committed as publishable (§6.5); §5.3 still yields a real per-layer exactness measurement |
| **Chunk size not discoverable** | Medium | Sweep boundary offsets over a full plausible range (8–512) and look for periodicity; if none appears, report alignment as a null factor and lean on P and C |
| **`spark2_5` unsupported by llama.cpp** | Medium | Level 2 is a control, not the treatment. Substitute any sliding-window model with known GGUF support (Gemma-3-class) |
| **Divergence too rare to detect** | Medium | Power calculation (§6.5); detection floor from the synthetic positive control (§8) |
| **Effect is a llama.cpp artifact, not architectural** | Medium | Stretch vLLM cross-check; and the level-2 control already separates architecture from engine |
| **State files large or slow to write** | Low | Long-prefix conditions dominate; WSL-native storage; save only at the boundary, not per token |
| Scope creep into speed/hit-rate headlines | Medium | §2.2 is a written decision creep must argue against |
| Reviewer thinks prior art was ignored | Medium | §1.5 in the README: TraceLab, Mooncake, 2605.27744, batch-invariance cited up front |
| Overclaiming beyond the tested engine | Medium | §2.2 forbids bit-exactness claims about vLLM; all claims scoped to engine, commit, and quantization |

### 10.2 Gates

| Gate | When | Pass condition | If failed |
|---|---|---|---|
| **G0** | Day 1–2 | Ornith-1.5-9B IQ4_XS loads; sequence state save/restore round-trips; **negative control 1 passes** (immediate restore is identical) | If the state API is unsupported: record as a finding, re-scope per §10.1 row 1 |
| **G1** | End week 1 | Empirical chunk size located (divergence periodicity across a boundary-offset sweep); one condition captured end-to-end to Parquet | If no periodicity: alignment becomes a null factor; proceed on P and C |
| **G2** | End week 2 | Noise floor measured; synthetic positive control establishes a detection floor; **pre-registration committed** | If floor swamps plausible effects: restrict claims to behavioral metrics and state diffs (§5.3) |
| **G3** | End week 4 | Level-1 dense negative control inert **and** synthetic positive control fires | **Stop.** Instrument unvalidated; nothing else is trustworthy until explained |

---

## 11. Schedule

| Week | Work | Gate |
|---|---|---|
| **0** (1–2 days) | Load Ornith IQ4_XS; exercise `llama_state_seq_save/load` (or llama-server slot save/restore); round-trip identity check; record runtime state dtype; pin commit | **G0** |
| 1 | Boundary-offset sweep to find the kernel chunk size; `traces` generator; `runner` both arms; one condition end-to-end | **G1** |
| 2 | Recompute-arm replicates and noise floor; `metrics` + `statediff` + unit tests; synthetic positive control and detection floor; power calculation; **commit pre-registration** | **G2** |
| 3 | Full 8-condition grid at level 3; add levels 2 and 1; `stats` — bootstrap, Holm families, coverage simulation | |
| 4 | Contrasts (alignment, alignment × architecture, compounding); headline figure; per-layer state profile; **controls** | **G3** |
| 5 | Secondary: reproduce #20225's per-turn cost; `gate` CI mode + GitHub Action; stretch — multimodal turns, or vLLM cross-check | |
| 6 | Write-up, figures, README; file findings on #20225 / #26201; optional LangGraph state-risk reporter demo | |

---

## 12. Deliverables and success criteria

**Deliverables**

1. `pip`-installable harness, seven components per §7.1.
2. Test suite including the `stats` coverage simulation.
3. GitHub Action running the regression gate.
4. Committed pre-registration.
5. Study write-up: divergence by alignment across the three levels, with CIs;
   per-layer state-difference profile; compounding trends; outcome census.
6. Documented detection floor from the synthetic positive control.
7. Findings filed on the relevant open upstream issues.
8. Optional: LangGraph state-risk reporter, labelled a demonstration.

**Target headline**

> On a Qwen3.5-architecture hybrid model at IQ4_XS, restoring saved recurrent
> state instead of recomputing changed the output of *X*% of agent turns (95% CI
> [*a*, *b*]) when the save boundary was misaligned with the linear-attention
> chunk size, versus *~0*% when aligned — and *~0*% at any alignment on both a
> sliding-window hybrid and a dense model of comparable size. In *W*% of
> divergent turns the selected tool call changed. Per-layer state differences
> concentrate in `linear_attention` layers, as the chunk-tiling mechanism
> predicts.

**Success criteria — all must hold**

- Synthetic positive control fires; a detection floor is documented.
- All three negative controls behave (round-trip identity, dense baseline,
  aligned level-3 boundaries) — or their failure is diagnosed and reported.
- Noise floor measured and reported, not assumed.
- Every run resolves to a recorded outcome; the census is published.
- Every effect carries a CI, including nulls.
- The coverage simulation demonstrates correct detection and false-positive rates.
- All claims scoped explicitly to engine, commit, and quantization.
- The write-up positions the work against TraceLab / Mooncake / 2605.27744 /
  batch-invariance prior art, and states the top-*k* truncation limit.

Explicitly **not** a success criterion: a high divergence rate. That is an
outcome, not a goal, and making it a goal would create pressure to overclaim.

---

## 13. References

- llama.cpp #20225 — hybrid agentic sessions reprocessing full prompt per turn; checkpoint logic written for sliding-window attention (open; primary anchor)
- LMCache hybrid-model documentation — non-bit-exactness disclaimer (motivating; vLLM-specific)
- vLLM #43587 — hybrid + multimodal prefix cache 0% hit; absent on the pure-attention sibling (open)
- vLLM #40696 — 528-token block-size cliff; motivates the alignment factor (open)
- vLLM #26201 — prefix caching for hybrid models, tracking issue (open)
- vLLM #45238 — checkpoint-boundary geometry; see §1.4 for the required caveat (open)
- vLLM PRs #45845, #46384 — merged partial fixes; `prefix_match_unit`
- vLLM RFC #54506 — M-dependent kernel arithmetic, verify-vs-decode non-invariance (open)
- SGLang #21846 — HiCache hybrid compatibility limited; agentic hybrid scheduling in progress
- TraceLab, arXiv 2606.30560 — agent-workload characterization, dense, speed-focused
- Policy-Driven Runtime Layer for Agentic LLM Serving, arXiv 2605.27744
- vLLM × Mooncake — disaggregated cache results; hybrid offload listed as future work
- Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (Sept 2025)
- LangChain, "Deep Agents Prompt Caching" — commoditized prompt-ordering guidance
- `ornith-ai/Ornith-1.5-9B` — `config.json` confirming `qwen3_5` architecture and layer pattern
- `XHToken/Spark-X2.5-4B` — `config.json` confirming `spark2_5` sliding-window pattern
