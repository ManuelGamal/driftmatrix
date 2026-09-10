# driftmatrix — Design Spec

**Does reusing saved recurrent state change what the model says?**
A cache-fidelity audit for hybrid linear-attention models under agent-shaped traffic.

Date: 2026-09-10
Status: design approved, pending spec review
Author: Manuel Gamal

---

## 1. The question

Modern serving stacks avoid recomputation by saving and restoring model state
between requests. On a dense transformer this is a KV cache: a per-token tensor
that can be sliced, matched at token granularity, and reused with the arithmetic
guarantee that a reused prefix is bit-identical to a recomputed one.

Hybrid models break that guarantee. In Qwen3.5/3.6-class architectures, 24 of 32
layers are **Gated DeltaNet** linear-attention layers carrying a *fixed-size
recurrent state* that is rewritten at every token. Such a state cannot be sliced
— it must be checkpointed whole at a specific position and restored whole. There
is no arithmetic reason for a restored state to equal a freshly computed one.

The vendor of the caching layer says as much. From LMCache's own hybrid-model
documentation, verbatim:

> "Generation is **not bit-exact** between a cached and a fresh run: GDN backends
> do not support vLLM's batch-invariant mode… Validate with a **score-level**
> comparison, not a token-level diff."

That is a disclaimer, not a measurement. Nobody has quantified how often outputs
diverge, by how much, under what geometry, or whether the divergence changes
model *behavior* rather than just its numbers.

**This project measures it, in the workload where it matters most.**

Agent traffic is the heaviest consumer of state reuse: every turn resends an
identical system prompt and identical tool schemas, plus a conversation history
that grows monotonically. It is the best case for caching and therefore the case
where caching is always on.

### 1.1 The claim under test

> For an identical agent turn, does serving from restored recurrent state produce
> a different output than recomputing from scratch — and does the difference
> change which tool the agent calls?

The second clause is the one that matters. A shifted logprob is a curiosity. A
changed tool call is a behavioral divergence caused by an optimization that is
supposed to be invisible.

### 1.2 Evidence, with honest confidence levels

Every item below was verified against the GitHub API or the primary document, not
a summary.

| Source | What it genuinely supports | Confidence |
|---|---|---|
| **LMCache hybrid-model docs** | The named claim under test. States generation is not bit-exact between cached and fresh runs on GDN backends; instructs users to abandon token-level comparison. Quantifies nothing. | **Strong — this is the anchor** |
| **vLLM #43587** (open, still reproducing 2026-09-06) | Prefix-cache hit rate pinned at 0% on hybrid Qwen3.5-4B; **does not reproduce on the pure-attention sibling model**. Architecture-caused by construction. | **Strong** |
| **vLLM #40696** (open) | The block-size cliff: 528-token block on Qwen3.5-4B, 0% hit rate below it, ~95% at 552 tokens. Used here as the positive control. | **Strong** |
| **llama.cpp #20225** (open) | Hybrid agentic sessions reprocess the entire prompt every turn — 15K tokens, ~8 min/turn — because checkpoint/restore logic written for sliding-window attention does not work on recurrent state. | **Strong — secondary target** |
| **vLLM #26201** (open tracking, filed 2025-10-03) | Umbrella issue for hybrid-model prefix caching; live and mid-implementation, not abandoned. | **Strong — context** |
| **vLLM Hybrid KV Cache Manager docs** | Documents per-group intersection of cache hits, and that the design does not support models without full-attention layers or with more than two attention types. Explains the mechanism the bugs exploit. | **Strong — mechanism** |
| **SGLang #21846** | Competing stack states HiCache hybrid compatibility "remains limited"; agentic KV-cache scheduling for hybrid/linear models on the Qwen3.5 family is Q3-2026 **in progress, not shipped**. Independent corroboration that this is unsolved industry-wide. | **Strong — corroboration** |
| **vLLM RFC #54506** (open) | Adjacent mechanism: M-dependent kernel arithmetic (split-K reduction order) makes verify-pass and plain-decode non-bit-identical on a hybrid MoE model; measured draft acceptance 100%→72%, throughput 110→89 tok/s. Same floating-point class as a restore path landing on a different shape. | **Moderate — analogy, not prefix-cache evidence** |
| **vLLM #45238** (open) | Reports catastrophic 0% hit rate when the checkpoint lands in the request-unique suffix. **Nuance that must be stated:** the reporter later conceded that specific case was already fixed on main by #37898/#43447 (40/64 hits, not 0/64). Independent corroborations continued through Aug 2026 on 0.26–0.28 builds — including loading a vision tower disabling caching for pure-text requests, and 261,794 byte-identical queries with 0 hits. | **Moderate — cite for boundary *geometry*, never as a live 0% claim** |
| **vLLM PRs #45845, #46384** | Both **merged** (Jun/Jul 2026). #46384 adds `prefix_match_unit` for token-granularity matching independent of block size. Partial fixes have landed — this is a moving target. | **Strong — forces commit pinning** |

### 1.3 Prior art — what this project must not pretend to be

**Agent-workload characterization on dense models is a crowded field.** TraceLab
(arXiv 2606.30560) analyzed 4,300 Claude Code/Codex sessions and ~350K LLM steps,
reporting 94–96% prefix-cache hit rates. vLLM × Mooncake published 1.7%→92.2%
hit-rate and 46× TTFT gains on Codex-derived traces. A Policy-Driven Runtime
Layer paper (arXiv 2605.27744) implements cache-aware routing at the
serving-engine tier with +13–37pp hit-rate gains. Redoing any of this is not a
contribution.

**Cache-aware prompt ordering is commoditized.** LangChain's own blog published
the numbers (49–80% token cost reduction; "stable content first, volatile last").
Settled advice.

**A cache-aware LangGraph/CrewAI runtime layer is scooped, and scooped deeper**
by arXiv 2605.27744 at the engine tier. Not attempted here.

What every one of those sources has in common: they measure **speed**. Hit rate,
TTFT, throughput, cost. Not one measures output fidelity, and none covers hybrid
architectures. Mooncake explicitly lists hybrid-model cache offloading as future
work. That absence — checked across all of them — is this project's entire
territory.

---

## 2. Scope

### 2.1 In scope

- Primary engine **vLLM**, pinned to a specific commit (Qwen3.5/3.6 requires main).
- One hybrid model and one dense control of comparable size.
- A cache-hit path versus a forced-recompute path, under controlled boundary geometry.
- Agent-shaped synthetic traces generated by a real agent framework.
- A `pip`-installable harness plus a written study.
- Secondary: cross-engine check against llama.cpp, reproducing #20225's cost.

### 2.2 Out of scope

- **Speed, hit rate, TTFT, throughput as headline results.** Crowded (§1.3).
  Hit rate is instrumentation here, never a finding.
- **A cache-aware agent runtime / router.** Scooped (§1.3).
- **Prompt-ordering guidance.** Commoditized (§1.3).
- **Engine-internal fixes** to LMCache/vLLM cache machinery — C++/CUDA scope well
  beyond this project. Findings get filed upstream; fixes are not promised.
- **Batch size / batch invariance** — solved by Thinking Machines' work and
  shipped as `VLLM_BATCH_INVARIANT`. Cited, used as an instrument, never claimed.
- **Long-context reasoning collapse** and **state quantization** — the other two
  stressors on the same instrument. Deliberate follow-on work, not this spec.

### 2.3 The role of the agent framework — stated honestly

LangGraph (or CrewAI) is the **traffic generator**, not the contribution. It
produces realistic agent turn structure: a stable system prompt, stable tool
schemas, monotonically growing message history, and real tool-call round trips.
That shape is what makes the cache-reuse question meaningful, and hand-rolling it
would be less credible than using what practitioners actually run.

An optional week-6 deliverable is a thin "cache-risk" reporter that surfaces
findings at the framework level. It is labelled a demonstration. The contribution
is the measurement.

---

## 3. Models and hardware

### 3.1 Models

| Role | Model | Why |
|---|---|---|
| **Treatment** | Qwen3.5-2B (hybrid GDN + Gated Attention, Apache 2.0) | Smallest current-generation hybrid; fits 8GB at int4 with room for state and cache |
| **Control** | Qwen3-1.7B (dense, official AWQ available) | Same vendor, same tokenizer family, comparable size, **dense**. Architecture is the only variable that changes |
| Stretch | Qwen3.5-4B / Qwen3.6-class | Only if 16GB proves available |

Same-vendor pairing is deliberate: a different vendor's dense model would add
tokenizer and training-data confounds and destroy the comparison.

**Naming discipline:** artifacts say "Qwen3.5/3.6-class hybrid," and the exact
checkpoint is resolved and recorded at build time. Qwen3.6 (Apr 2026) and
Qwen3.8-27B (Aug 2026) are newer members of the same architecture family.

### 3.2 Hardware, and the two real gates

Local: **RTX 4060, assumed 8GB** (specified for the worst case; a 16GB 4060 Ti
only unlocks larger models). Ada / sm89 — fused GDN Triton kernels expected to work.

Kaggle: **T4, 16GB but sm75.** More memory, older architecture.

Two hard gates, both resolved in week 0:

1. **Does vLLM run a GDN hybrid on the T4 at all?** sm75 may not support the
   fused Gated-DeltaNet kernels. If it does not, Kaggle drops to analysis and
   trace generation only, and all engine runs happen on the 4060.
2. **Does vLLM-from-main build and run a hybrid model in 8GB?** vLLM preallocates
   KV memory; `--gpu-memory-utilization` must be tuned. If 8GB proves unworkable,
   fall back to the T4 (if gate 1 passed) or to llama.cpp as primary engine —
   which is defensible, since #20225 is a llama.cpp bug.

---

## 4. Experiment design

### 4.1 Factors

Within each architecture, a 2×2×2 design with replication:

| ID | Factor | Levels |
|---|---|---|
| **T** | Cache path (the treatment) | restored-from-cache · forced cold recompute |
| **G** | Checkpoint boundary geometry | boundary falls in the **shared prefix** · boundary falls in the **request-unique suffix** |
| **L** | Prompt length vs the block-size cliff | below one block (<528 tok) · above two blocks (>1056 tok) |
| **A** | Architecture (blocking variable) | hybrid · dense control |

**G** replicates the trigger geometry described in #45238 — the mechanism is
documented in vLLM's Hybrid KV Cache Manager as per-group intersection of hits,
so where the boundary lands relative to shared content is expected to matter.

**L** exists because of #40696. Without it the study would silently measure
nothing: below the 528-token block size, prefix caching does **not engage at
all**, and a null result would be indistinguishable from a real one.

Every cell runs **twice** — including the reference — so the nondeterminism floor
is measured, not assumed, and "cell vs itself" and "cell vs reference" traverse
identical code paths.

### 4.2 Forcing the two paths

The treatment requires the *same* request served two ways:

- **Cache-hit path:** warm the server with the shared prefix, then issue the turn
  and assert via metrics (`prefix_cache_hits_total`, matched-token count) that a
  hit actually occurred. A cell claiming a hit that did not happen is discarded.
- **Forced-cold path:** the same turn against evicted or cold cache, verified as a
  miss by the same metrics.

Hit-rate instrumentation exists solely to prove which path was taken. It is never
a reported finding (§2.2).

### 4.3 Agent-shaped traces

Generated with LangGraph, structured as real agent turns:

| Property | Construction |
|---|---|
| Stable system prompt | Identical across all turns and sessions |
| Stable tool schemas | Fixed set, fixed serialization order |
| Growing history | Monotonic message accumulation across turns |
| Tool-call round trips | Real calls to deterministic mock tools |
| Prefix length control | Padded to satisfy factor **L** on both sides of the 528-token cliff |

Deterministic mock tools matter: a nondeterministic tool would inject variance
indistinguishable from the effect under study.

~60 sessions × ~6 turns ≈ 360 turns, frozen and content-hashed. Every result row
carries the trace hash.

### 4.4 Two measurement modes

**Free-running — the headline.** Both paths generate normally at temperature 0.
Records exact-match rate, first-divergence position, **selected tool call**, and
final answer.

**Teacher-forced — the instrument.** Each path scores the *other's* output text so
every position is compared under identical context, yielding attributable
per-position distribution differences via `prompt_logprobs`.

Teacher-forced measures how much the distribution moved; free-running measures
whether it mattered. Reporting only one is the error this design exists to avoid.

---

## 5. Metrics

### 5.1 Behavioral (primary — these are the headline)

- **Tool-call agreement.** Did both paths select the same tool with the same
  arguments? Disagreement is the finding that matters.
- **Exact-match rate** of the full completion.
- **First-divergence position**, right-censored at the generation cap.
- **Final-answer agreement** on the subset with a verifiable answer.

### 5.2 Distributional (supporting)

Per-position **Jensen-Shannon divergence** over top-20 logprobs — symmetric,
bounded by log 2, defined on differing supports. Reported as per-turn mean
(stable) and max (catches a single catastrophic token the mean would hide). Also
total variation distance and top-1 agreement rate.

**Truncation rule.** vLLM caps returned logprobs at 20, so distributions are
observed only at the top. Renormalize over the **union** of both paths' top-20
token sets and assign remaining mass to one `<other>` bucket. A stated
limitation, not a hidden one.

### 5.3 Headline figure

**Divergence rate by boundary geometry and length, hybrid versus dense**, with
confidence intervals — a small grouped bar chart where the dense control should
sit at zero and the hybrid bars should not. Readable in five seconds.

Secondary figure: survival curves of the fraction of turns still token-identical
at position *k*, per condition.

---

## 6. Statistics

### 6.1 Effects measured against noise, never against zero

Two runs of an identical configuration are not bit-identical; kernel
nondeterminism guarantees jitter. Per turn *p*:

```
effect(cell c, turn p) = divergence(c, cold) − divergence(cold₁, cold₂)
```

The subtracted term is the noise floor from the cold path's own two replicates.
The bootstrap runs on that **difference**.

**Free finding — self-consistency.** Every cell's two replicates are compared to
each other. A cell is flagged **self-inconsistent** when its median per-turn mean
JSD exceeds **twice** the cold path's self-comparison median — a pre-specified
engineering threshold, committed with the pre-registration and revisable only
with a reason recorded in git.

### 6.2 Tests

- **Paired bootstrap over turns** (10,000 resamples), paired by turn so
  intrinsic turn difficulty is not a confound.
- **Holm–Bonferroni** within three separately declared families — pooling
  unrelated hypotheses would over-correct and hide real effects:

  | Family | Members | Size |
  |---|---|---|
  | A — behavioral divergence rates | 4 geometry×length cells × 2 architectures | 8 |
  | B — distributional effects | 4 cells × 2 statistics (mean, max JSD), hybrid | 8 |
  | C — interaction terms | geometry × length, and each × architecture | 3 |

- **Confidence intervals on every effect**, significant or not. Null results are
  reported with their intervals, never as a bare "no effect."

### 6.3 The interaction that carries the contribution

The claim is not merely "hybrid diverges." It is:

> Divergence is concentrated where the mechanism predicts — when the state
> checkpoint boundary falls in the request-unique region rather than the shared
> prefix — and is absent on a same-size dense model.

A geometry effect present in the hybrid and absent in the dense control is
mechanistic evidence, not a correlation. That combination is what makes this a
finding about *recurrent state* rather than about vLLM in general.

Additivity is a chosen null model, not a law; interaction claims are stated as
"superadditive relative to an additive null on the JSD scale."

### 6.4 Pre-registration

The analysis plan, family definitions, and thresholds are committed to git **at
the end of week 2** — after the noise floor is known, before any treatment effect
is examined. Deciding what counts as significant after seeing results is how a
project talks itself into a finding that is not there.

### 6.5 Statistical power, and designing for a null

Divergence may be rare. Before the main run, a power calculation fixes the turn
count needed to detect a per-turn divergence rate of **2% at 80% power,
α = 0.05**, against a floor of 0%. If 360 turns is insufficient at that target,
the turn count rises until it is met — turns are cheap (more per session),
sessions are expensive (more engine launches). The achieved power is reported
alongside the result, so a null carries a stated detection limit rather than an
implied one.

**A null result is pre-committed as publishable**, phrased as: *contrary to
LMCache's blanket non-bit-exactness disclaimer, no measurable divergence was
found for plain prefix-cache restore in configuration X, bounding the risk at
<Y% (95% CI), which suggests the phenomenon is specific to speculative-decode
paths (cf. RFC #54506) rather than prefix caching.* That is a useful, honest,
citable result, and committing to it in advance removes the incentive to
overclaim later.

---

## 7. Architecture

### 7.1 Components

Load-bearing decision: **GPU work happens once, writes raw captured outputs and
distributions to Parquet, and every other component reads those files.** Analysis
reruns are then free, analysis code is testable with no GPU, the harness can have
real CI, and Kaggle can generate data while analysis happens locally.

| Component | Responsibility | Depends on | GPU |
|---|---|---|---|
| `traces` | LangGraph agent-trace generator; freezing, padding to length factors, content hashing | langgraph | no |
| `runner` | Enumerate cells, force hit/cold paths, verify path via metrics, capture outputs + logprobs → Parquet. **Sole owner of the engine dependency.** | vLLM | yes |
| `metrics` | Parquet → behavioral agreement + per-position divergence | numpy, pandas, pyarrow | no |
| `stats` | Noise floor, paired bootstrap, Holm families, interactions, power | numpy, scipy | no |
| `gate` | CI mode: compare against a golden baseline, non-zero exit on regression | metrics, stats | no |
| `report` | Tables, grouped bar chart, survival curves | metrics, stats, matplotlib | no |

`runner` is deliberately dumb — capture and store, no analysis. All judgment
lives in `stats`, pure functions over arrays: the part that must be right,
isolated into the part easiest to test.

### 7.2 Data schema

One Parquet row per `(model, cell_id, replicate, mode, session_id, turn_idx, position)`:

| Field | Type | Notes |
|---|---|---|
| `model_id`, `architecture` | string | `hybrid` \| `dense` |
| `cell_id` | int8 | Encodes T, G, L |
| `path`, `geometry`, `length_band` | string | Denormalized factor levels |
| `replicate` | int8 | 1 or 2 |
| `mode` | string | `free_running` \| `teacher_forced` |
| `session_id`, `turn_idx` | string, int16 | |
| `position` | int16 | |
| `token_ids`, `logprobs` | list[int32], list[float32] | Top-20, aligned |
| `chosen_token_id` | int32 | |
| `tool_name`, `tool_args_hash` | string | The primary behavioral outcome |
| `cache_hit_verified` | bool | Path assertion from engine metrics |
| `matched_tokens` | int32 | |
| `engine`, `engine_commit` | string | **Stamped per row** — PRs are actively merging |
| `trace_hash`, `run_id`, `run_started_at` | string, string, timestamp | |

Sizing: ~360 turns × ~256 positions × 16 cells × 2 replicates × 2 modes ≈ 6M
position-rows, a few GB of Parquet.

**Placement.** Repo on the Windows filesystem (WSL-visible at `/mnt/c/...`);
Parquet data on the **WSL-native** filesystem for I/O throughput, gitignored.

### 7.3 Failure as data

Every run resolves to a recorded outcome: `ok` · `unsupported` · `crashed` (with
captured error) · `timeout` · `path_unverified` (the claimed hit or miss could not
be confirmed, so the cell is excluded from effects but counted).

`unsupported` and `crashed` are findings. "This engine cannot serve a hybrid
model with prefix caching under configuration X" is a reportable result and one of
several paths guaranteeing a non-empty study.

---

## 8. Validity controls

**Positive control — instrumentation.** Reproduce #40696's block-size cliff: ~0%
hit rate below 528 tokens, high hit rate above it. If the harness cannot see a
documented, mechanically simple effect, its instrumentation is wrong and nothing
else it reports can be trusted. Hard gate (§10).

**Negative control — the dense model.** Prefix caching on dense transformers is
designed to be lossless, so the dense control should show divergence
indistinguishable from its noise floor. A large dense effect means the harness is
measuring something other than state restoration.

**Noise floor.** Every cell replicated; effects are differences against the cold
path's self-comparison (§6.1).

**Commit pinning.** One engine commit SHA, stamped per row. #45845 and #46384
already merged; treating the engine as stable would make results unreproducible.

**Deterministic tools.** Mock tools are pure functions of their arguments.

**Pre-registration** before treatment effects are examined (§6.4).

---

## 9. Testing the harness

- **`metrics`** — synthetic distributions with known answers: identical
  distributions give exactly 0, disjoint supports give log 2, hand-computed pairs
  match to tolerance. Union-support truncation tested against worked examples.
- **`stats`** — a **coverage simulation**: inject a synthetic effect of known size
  and verify detection at the claimed rate; run under the null and verify the
  false-positive rate approximates α. The project proving its own statistics
  rather than assuming them. Holm verified against a reference implementation.
- **`runner`** — fake-engine mock returning canned outputs and metrics, so cell
  enumeration, path verification, outcome classification, and Parquet writing are
  testable with no GPU.
- **`traces`** — hash stability, and assertions that padding actually places
  prompts on the intended side of the 528-token cliff (tokenizer-counted, not
  character-counted).
- **`gate`** — golden-file tests: clean baseline passes, seeded regression fails
  with non-zero exit.

---

## 10. Risks and gates

### 10.1 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **T4 (sm75) cannot run fused GDN kernels** | High | Week-0 gate. If it fails, Kaggle does traces + analysis only; all engine runs on the 4060 |
| **8GB insufficient for vLLM + hybrid model** | High | Week-0 gate. Tune `--gpu-memory-utilization`; fall back to T4, or to llama.cpp as primary (defensible — #20225 is a llama.cpp bug) |
| **Null result** | Medium | Pre-committed as publishable (§6.5); bounds LMCache's unquantified disclaimer either way |
| **Divergence too rare to detect** | Medium | Power calculation before the main run; turn count scales up if needed |
| **Moving target — fixes merging upstream** | Medium | Pinned commit SHA stamped per row; findings framed as "at commit X," never "vLLM is broken" |
| **Cannot reliably force a cache hit** | Medium | Every cell asserts its path via engine metrics; unverifiable cells recorded as `path_unverified` and excluded from effects |
| **`prompt_logprobs` unreliable on hybrid models** | Medium | Behavioral metrics (§5.1) are primary and do not depend on it; teacher-forced is supporting. Crashes recorded as outcomes |
| **Qwen3.5 needs vLLM main — build friction** | Medium | Week-0 gate; llama.cpp fallback |
| Scope creep into speed/hit-rate results | Medium | §2.2 is a written decision creep must argue against |
| Reviewer thinks the work ignores prior art | Medium | §1.3 in the README: TraceLab, Mooncake, 2605.27744, batch-invariance all cited up front |

### 10.2 Gates

| Gate | When | Pass condition | If failed |
|---|---|---|---|
| **G0** | Day 3 | vLLM-from-main serves the hybrid model with prefix caching on 4060 **or** T4; hit-rate metrics readable | Switch primary engine to llama.cpp; re-scope to cross-engine cost reproduction (#20225) |
| **G1** | End week 1 | Traces generated; one cell captured end-to-end to Parquet with verified hit and verified cold paths | Simplify trace structure; drop tool round trips to static histories |
| **G2** | End week 2 | Noise floor measured; cold-path self-comparison median per-turn mean JSD **< 0.01 nats** (≈1.4% of the log 2 bound). Pre-registration committed | Enable `VLLM_BATCH_INVARIANT`, re-measure. If still high, report the floor as a primary finding and restrict claims to behavioral metrics |
| **G3** | End week 4 | **Positive control fires** (#40696 cliff reproduced) **and** dense negative control is inert | **Stop.** Instrumentation unvalidated; no other finding is trustworthy until explained |

---

## 11. Schedule

| Week | Work | Gate |
|---|---|---|
| **0** (3 days) | Engine viability: vLLM main + hybrid model on 4060 and T4; confirm GDN kernels run; confirm hit-rate metrics readable; pin commit | **G0** |
| 1 | `traces` (LangGraph generator, padding, hashing); `runner` forced hit/cold paths; one cell end-to-end | **G1** |
| 2 | Both cold replicates; noise floor; `metrics` + unit tests; power calculation; **commit pre-registration** | **G2** |
| 3 | Full 16-cell grid, hybrid + dense; `stats` — bootstrap, Holm families, coverage simulation | |
| 4 | Interactions; headline figure; survival curves; **positive + negative controls** | **G3** |
| 5 | Cross-engine llama.cpp check + #20225 cost reproduction; `gate` CI mode + GitHub Action | |
| 6 | Write-up, figures, README; file findings on #43587 / #26201; optional LangGraph cache-risk reporter demo | |

---

## 12. Deliverables and success criteria

**Deliverables**

1. `pip`-installable harness, six components per §7.1.
2. Test suite including the `stats` coverage simulation.
3. GitHub Action running the regression gate.
4. Committed pre-registration.
5. Study write-up: divergence rates by geometry and length, hybrid vs dense, all
   with CIs; headline figure; survival curves; recorded outcome census.
6. Reproduction of #40696's cliff as an automated test.
7. Findings filed on the relevant open upstream issues.
8. Optional: thin LangGraph cache-risk reporter, labelled a demonstration.

**Target headline**

> On hybrid model *M* at commit *C*, *X*% of agent turns produced different
> output when served from restored recurrent state versus recomputed (95% CI
> [*a*, *b*]), rising to *Z*% when the state checkpoint boundary fell in the
> request-unique region rather than the shared prefix — versus *~0*% on a
> same-size dense model. In *W*% of divergent turns, the selected tool call
> changed.

**Success criteria — all must hold**

- Positive control fires; dense negative control inert.
- Noise floor measured and reported, not assumed.
- Every run resolves to a recorded outcome; the census is published.
- Every effect carries a CI, including nulls.
- The coverage simulation demonstrates correct detection and false-positive rates.
- Every reported cell has a verified cache path.
- The write-up positions the work against TraceLab / Mooncake / 2605.27744 /
  batch-invariance prior art, and states the top-20 truncation limit.

Explicitly **not** a success criterion: finding a high divergence rate. That is an
outcome, not a goal, and making it a goal would create pressure to overclaim.

---

## 13. References

- LMCache hybrid-model documentation — non-bit-exactness disclaimer (the anchor claim)
- vLLM #43587 — hybrid + multimodal prefix cache 0% hit; absent on the pure-attention sibling (open)
- vLLM #40696 — 528-token block-size cliff (open; positive control)
- vLLM #26201 — prefix caching for hybrid models, tracking issue (open)
- vLLM #45238 — checkpoint-boundary geometry; see §1.2 for the required caveat (open)
- vLLM PRs #45845, #46384 — merged partial fixes; `prefix_match_unit`
- vLLM RFC #54506 — M-dependent kernel arithmetic, verify-vs-decode non-invariance (open)
- vLLM design docs — Hybrid KV Cache Manager
- llama.cpp #20225 — hybrid agentic sessions reprocessing full prompt per turn (open)
- SGLang #21846 — HiCache hybrid compatibility limited; agentic hybrid scheduling in progress
- TraceLab, arXiv 2606.30560 — agent-workload characterization, dense, speed-focused
- Policy-Driven Runtime Layer for Agentic LLM Serving, arXiv 2605.27744
- vLLM × Mooncake — disaggregated cache results; hybrid offload listed as future work
- Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (Sept 2025); vLLM `VLLM_BATCH_INVARIANT`
- LangChain, "Deep Agents Prompt Caching" — commoditized prompt-ordering guidance
