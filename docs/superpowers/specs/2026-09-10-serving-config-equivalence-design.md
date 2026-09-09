# driftmatrix — Design Spec

**A factorial equivalence harness for LLM serving configurations.**

Date: 2026-09-10
Status: design approved, pending spec review
Author: Manuel Gamal

---

## 1. The problem

An LLM serving engine exposes optimizations that are each expected to be
output-preserving: weight quantization, KV-cache quantization, prefix caching,
speculative decoding. Operators turn them on for throughput and memory, on the
assumption that the model still says the same thing.

That assumption is not systematically tested. Two gaps exist in current practice:

**Nobody tests combinations.** Published quantization checks are single-factor
spot comparisons — AWQ vs fp16 accuracy on one task. Feature *interactions* go
untested, and interactions are where the known bugs live.

**Nobody tests the distribution.** vLLM's nightly CI runs task-accuracy
evaluations (GSM8K, GPQA, AIME, BFCL). A configuration can shift the output
distribution substantially while leaving aggregate accuracy inside noise, and
such a shift is invisible to accuracy-based gates.

driftmatrix closes both: it treats serving configuration as a **2⁴ factorial
experiment** with the model's output distribution as the response variable, and
reports main effects and pairwise interactions with confidence intervals
measured against the engine's own nondeterminism floor.

### 1.1 Evidence, with honest confidence levels

Sources were verified individually. What each one actually supports:

| Source | What it genuinely supports | Confidence |
|---|---|---|
| vLLM issue #25262 | Direct anchor. Identical shared-prefix tokens return *different* logprobs when prefix caching is enabled. Closed "not planned" — unfixed. | **Strong — primary evidence** |
| vLLM issue #13105 | Prefix caching + `prompt_logprobs` raises an assertion. A feature-interaction fault, but a **crash**, not silent divergence. | Moderate — supports interaction risk, not silent drift |
| `llm-compressor` issues #2031, #2646 | Maintainers requesting a KL-divergence evaluation tool / cheap KLD metric. Both **open and unresolved**. | **Strong — demand evidence** |
| vLLM blog, "Keeping vLLM Production Quality" (Jul 2026) | Maintainers concede nightly CI does not simulate real user configuration permutations. | **Strong — gap evidence** |
| vLLM issue #9453 | Logprobs vary with sampling parameters inconsistently with OpenAI semantics. Tangential; arguably definitional. | Weak — do not cite as motivation |
| vLLM issue #3032 | 2023 tensor-shape crash in batched `prompt_logprobs`. Likely long fixed. | Weak — do not cite |

The write-up cites only the four strong/moderate rows. The two weak rows are
recorded here so the project does not later re-inflate them into evidence.

### 1.2 Prior art this project must not pretend to supersede

**Batch invariance is solved, by other people, already.** Thinking Machines Lab's
"Defeating Nondeterminism in LLM Inference" (Sept 2025) diagnosed batch-size
nondeterminism as non-batch-invariant reduction kernels and shipped
`batch_invariant_ops`. vLLM has since built this in natively
(`VLLM_BATCH_INVARIANT`, PRs #25603 / #26373, tracking issue #27433, documented
under `docs.vllm.ai/en/latest/features/batch_invariance/`).

Consequences, binding on this project:

1. **Batch size is not a factor in the design.** It is out of scope by
   construction, not by omission.
2. **The README and write-up open by citing this work.** Leading with batch-size
   novelty would signal ignorance of a well-known, well-funded, shipped result.
3. `VLLM_BATCH_INVARIANT` is used as an *instrument*: if the measured
   nondeterminism floor is too large to leave room for real effects, enabling it
   shrinks the floor.

**vLLM already has some logprob correctness testing.** Its CI runs
`check_logprobs_close`, comparing vLLM against HuggingFace top-k logprob overlap
per model. That is a single-config reference comparison. It is not a
configuration-permutation study, has no noise floor, no interaction analysis, and
no statistical testing. driftmatrix is complementary to it, not a replacement.

---

## 2. Scope

### 2.1 In scope

- One engine: **vLLM**, single pinned version, single GPU.
- Four binary factors, 16 cells, 2 replicates per cell.
- Two measurement modes (teacher-forced and free-running, §4.2).
- Two models, with a third as a stretch goal.
- A `pip`-installable tool with a CI regression-gate mode.
- A written study reporting the 16-cell result.

### 2.2 Out of scope

Explicitly excluded, and the reasons, so scope creep has to argue against a
written decision:

- **Batch size / batch invariance** — solved elsewhere (§1.2).
- **SGLang or any second engine** — doubles setup cost, adds nothing to the
  finding. Week-6 stretch only.
- **Throughput or latency benchmarking** — this project is about *correctness*.
  Mixing performance numbers in blurs the claim.
- **Multi-GPU, tensor parallelism** — introduces a second nondeterminism source
  that would confound the floor.
- **Custom kernels, new quantization methods, new allocators** — driftmatrix
  measures existing configurations. It does not propose better ones.
- **Sampling randomness** — all runs are greedy with a fixed seed, because the
  question is whether the *engine* changes the output.

### 2.3 The claim being made

> These configurations shift the model's output distribution beyond the engine's
> own measured nondeterminism floor, and these factor pairs shift it
> superadditively relative to an additive null model.

Not "quantization is harmful." Not "vLLM is broken." The narrow claim is the
defensible one.

---

## 3. Factors

| ID | Factor | Off (reference) | On |
|---|---|---|---|
| F1 | Weight quantization | bf16 | AWQ W4A16 |
| F2 | KV-cache quantization | `--kv-cache-dtype auto` | `--kv-cache-dtype fp8` |
| F3 | Prefix caching | disabled explicitly | enabled |
| F4 | Speculative decoding | off | n-gram / prompt-lookup |

**Reference cell (cell 0):** bf16, KV auto, prefix caching off, speculation off.

Notes that affect implementation:

- **F3 must be disabled explicitly.** Prefix caching is on by default in vLLM
  V1; the reference cell requires an explicit disable flag, not an omission.
- **F4 uses n-gram speculation deliberately** so no matching draft-model
  checkpoint has to be sourced or trusted.
- **F1 requires AWQ checkpoints.** Week 1 verifies published AWQ checkpoints
  exist for both models; if not, they are produced once with `llm-compressor` and
  the exact producing command is committed.

### 3.1 F4 is a negative control in teacher-forced mode

Speculative decoding alters the *generation* path, not prompt scoring. Under
teacher-forced measurement (§4.2) F4 should therefore be **inert** — its main
effect should be statistically indistinguishable from the noise floor.

This is not a flaw in the design; it is a free negative control. A measured F4
effect in teacher-forced mode means the instrument is mis-measuring something,
and invalidates other findings until explained. F4 is a live factor only in
free-running mode, where its lossless-output claim is exactly what gets tested.

---

## 4. Experiment design

### 4.1 Structure

A **2⁴ factorial with 2 replicates**: 16 cells × 2 replicates × 2 models = 64
engine launches. Both measurement modes are captured from a single launch per
(cell, replicate, model), so mode does not multiply launch count.

Replication is what produces the noise floor, and it applies to **every cell
including the reference** — the replicate is an ordinary row in the grid, not a
special case, so "cell vs itself" and "cell vs reference" traverse identical code
paths.

### 4.2 Two measurement modes

**Teacher-forced — the instrument.** Every cell scores the *reference cell's*
generated text rather than generating freely. Every position is then compared
under an identical context, so per-position distribution differences are clean
and attributable. Implemented via `prompt_logprobs` over the reference
continuation appended to the prompt.

**Free-running — the headline.** Each cell generates normally. Records where
output first diverges from the reference and whether the final answer differs.

The two answer different questions. Teacher-forced measures *how much* the
distribution moved; free-running measures *whether it mattered*. Reporting only
one is the error this design exists to avoid.

### 4.3 Prompt suite

200 prompts in four deliberate slices of 50, each stressing one mechanism:

| Slice | Stresses | Construction |
|---|---|---|
| `shared-prefix` | prefix-cache reuse (F3) | Many prompts sharing a long identical preamble, diverging only in the final instruction |
| `numeric-dense` | quantization outlier channels (F1) | Tables, arithmetic, code with dense literals |
| `long-context` | KV-cache quantization (F2) | Long inputs near the context budget |
| `repeated-substring` | n-gram speculation acceptance (F4) | Heavy verbatim repetition, the pattern prompt-lookup exploits |

The suite is **frozen, versioned, and content-hashed**. Every result row carries
the suite hash, so any finding is traceable to exact inputs.

Slice-level reporting is what upgrades "cell 6 diverges" into "cell 6 diverges on
numerically dense prompts, consistent with quantization outlier channels" — a
mechanism rather than an observation.

### 4.4 Generation length

128 new tokens. Long enough for divergence to develop, short enough to bound
storage (§7.2).

---

## 5. Metrics

### 5.1 Handling the top-20 truncation

vLLM's API caps returned logprobs at 20. Distributions are therefore observed
only at the top, and every metric must be defined on truncated support.

**Rule:** for a pair of cells at one position, renormalize over the **union** of
the two cells' top-20 token sets, and assign all remaining probability mass to a
single `<other>` bucket. This is a real limitation, stated plainly in the
write-up, not hidden.

### 5.2 Teacher-forced metrics

Per (cell, prompt), two scalars — both are reported, because they fail
differently:

- **Mean per-position JSD** across the 128 positions. Primary. Stable.
- **Max per-position JSD.** Catches a cell that is fine everywhere except one
  catastrophic token, which the mean would hide.

Jensen-Shannon divergence is preferred over raw KL: symmetric, bounded in
[0, log 2], and defined when supports differ — all three matter here.

Also recorded: total variation distance, top-1 agreement rate, and Spearman
correlation of the top-20 ranking.

### 5.3 Free-running metrics

- **First-divergence position** — the first index where the generated token
  differs from the reference (censored at 128 if never).
- **Exact-match rate** of the full completion.
- **Answer-change rate** on the prompt subset with a verifiable answer.

### 5.4 Headline figure

A **survival curve** per cell: the fraction of prompts still token-identical to
the reference at position *k*. Curves dropping early are cells that change
answers fast. Readable without statistical training, and it answers the
operator's actual question. This is Figure 1.

Note the metric is right-censored at 128 tokens, so the curve is a
Kaplan–Meier-style estimate and must be labelled as such.

---

## 6. Statistics

### 6.1 Effects are measured against noise, never against zero

Two runs of an identical configuration do not produce identical distributions;
kernel nondeterminism guarantees jitter. "Cell X differs from reference" is
therefore vacuous on its own.

Per prompt *p* and cell *c*:

```
effect(c, p) = divergence(c₁, ref₁) − divergence(ref₁, ref₂)
```

The subtracted term is the noise floor, measured from the reference's own two
replicates. The bootstrap runs on this **difference**.

**Free finding — self-consistency.** Every cell's own two replicates are
compared to each other with the same statistic. A cell is flagged
**self-inconsistent** when its median per-prompt mean JSD exceeds **twice** the
reference cell's own self-comparison median. That is a separate reportable
defect (the cell is non-deterministic with itself) and is recorded per cell
alongside its effect estimate.

The 2× multiplier is a pre-specified engineering threshold, not a statistical
test. It is committed with the pre-registration (§6.4) and revisable only with a
documented reason recorded in git history.

### 6.2 Tests

- **Paired bootstrap over prompts** (10,000 resamples), pairing by prompt so
  "some prompts are intrinsically harder" is removed as a confound.
- **Holm–Bonferroni correction**, applied within three separately declared
  families rather than one pooled set — pooling unrelated hypotheses would
  over-correct and hide real effects:

  | Family | Members | Size |
  |---|---|---|
  | A — teacher-forced cell effects | 15 non-reference cells × 2 primary statistics (mean JSD, max JSD) | 30 |
  | B — factorial terms | 4 main effects + 6 pairwise interactions, on mean JSD | 10 |
  | C — free-running effects | 15 cells × first-divergence position | 15 |

  Family-wise error control, not FDR, because these are bug claims and a false
  positive is expensive. The three families are declared in the
  pre-registration; no hypothesis is tested in more than one.
- **Bootstrap CIs** reported for every effect, significant or not. Null results
  are reported with their intervals, never as bare "no effect."

### 6.3 Factorial decomposition — the contribution

From the 16 cells, estimate the four main effects and six pairwise interaction
terms, each with a bootstrap CI.

The interaction terms are what single-factor testing cannot produce:

> Does the divergence under *F1 + F3 together* exceed what F1 alone and F3 alone
> predict?

A positive interaction whose CI excludes zero is a feature-interaction finding —
the class of fault #13105 and #25262 belong to.

**Required honesty.** Additivity is a *chosen null model*, not a law. Divergence
is not a linear response in these factors. The claim is therefore precisely
"superadditive relative to an additive null on the JSD scale," and the write-up
says exactly that. Overclaiming here is the fastest way to lose a technical
reader.

### 6.4 Pre-registration

The analysis plan, statistics, and significance thresholds are committed to git
**at the end of week 2** — after the noise floor is known (thresholds depend on
it), and before any cell effect has been examined.

Choosing what counts as significant after seeing results is how a project talks
itself into a finding that is not there.

---

## 7. Architecture

### 7.1 Components

The load-bearing structural decision: **GPU work happens once, writes raw
captured distributions to Parquet, and every other component reads those files.**

That seam means analysis reruns are free, the analysis code is testable on a
laptop with no GPU, the tool can have real CI, and Kaggle can produce data while
analysis happens locally.

| Component | Responsibility | Depends on | GPU |
|---|---|---|---|
| `suite` | Frozen prompt corpus, loader, content hashing | — | no |
| `runner` | Enumerate grid, launch vLLM per cell, capture logprobs, write Parquet. **Sole owner of the vLLM dependency.** | vLLM | yes |
| `metrics` | Parquet → per-position and per-prompt divergence scalars | numpy, pandas, pyarrow | no |
| `stats` | Noise floor, paired bootstrap, Holm, factorial decomposition | numpy, scipy | no |
| `gate` | CI mode: compare against golden baseline, non-zero exit on regression | metrics, stats | no |
| `report` | Result tables, survival curves, figures | metrics, stats, matplotlib | no |

`runner` is deliberately dumb: capture and store, no analysis. All judgment lives
in `stats`, which is pure functions over arrays — the part that must be right,
isolated into the part that is easiest to test.

### 7.2 Data schema

One Parquet row per `(model, cell_id, replicate, mode, prompt_id, position)`:

| Field | Type | Notes |
|---|---|---|
| `model_id` | string | |
| `cell_id` | int8 | 0–15, bit-encoded over F1..F4 |
| `f1`..`f4` | bool | Denormalized for query convenience |
| `replicate` | int8 | 1 or 2 |
| `mode` | string | `teacher_forced` \| `free_running` |
| `prompt_id` | string | |
| `slice` | string | One of the four suite slices |
| `position` | int16 | |
| `token_ids` | list[int32] | Top-20 token ids |
| `logprobs` | list[float32] | Top-20 logprobs, aligned to `token_ids` |
| `chosen_token_id` | int32 | |
| `vllm_version` | string | **Stamped on every row** — results are unreproducible without it |
| `suite_hash` | string | |
| `run_id` | string | |
| `run_started_at` | timestamp | |

**Sizing.** Teacher-forced: 200 prompts × 128 positions × 64 launches ≈ 1.6M
position-rows, each holding two 20-element lists. Free-running adds a comparable
volume at variable length. Total on the order of 3–4M rows and a few GB of
Parquet — comfortable.

**Placement.** The repository lives on the Windows filesystem (visible to WSL
under `/mnt/c/...`). The Parquet data directory lives on the **WSL-native**
filesystem for I/O throughput, and is gitignored. Only the suite, code, analysis
outputs, and figures are committed.

### 7.3 Failure as data

Some cells may not be enableable at all — flags may be mutually exclusive, or a
combination may crash (as #13105 does). Every launch therefore resolves to one
of four recorded outcomes:

`ok` · `unsupported` (engine refused the flag combination) · `crashed` (with the
captured error) · `timeout`

`unsupported` and `crashed` cells are **findings**, not gaps. "These
configuration combinations cannot be enabled together" is a reportable result and
one of the paths that guarantees a non-empty study.

---

## 8. Validity controls

**Positive control — instrument validation.** The harness must independently
rediscover vLLM #25262 (prefix caching altering logprobs for identical
shared-prefix tokens). The `shared-prefix` suite slice exists partly to trigger
it. If the instrument cannot find a documented live bug, it does not work, and no
other finding it produces is trustworthy. This is a hard gate (§10).

**Negative control.** F4's main effect under teacher-forced measurement must be
indistinguishable from the noise floor (§3.1). A detected effect means the
measurement path is contaminated.

**Noise floor.** Every cell replicated; effects computed as differences against
the reference's self-comparison (§6.1).

**Version pinning.** One pinned vLLM version, stamped into every row.

**Pre-registration.** Analysis plan committed before results are examined (§6.4).

---

## 9. Testing the tool itself

The analysis layers are pure functions and get real tests, not smoke tests.

- **`metrics`** — synthetic distributions with known answers. Identical
  distributions must give exactly 0. Hand-computed JSD pairs must match to
  floating-point tolerance. Disjoint supports must give log 2. Truncation and
  `<other>`-bucket renormalization tested against hand-worked examples.
- **`stats`** — a **coverage simulation**. Inject a synthetic effect of known
  size and verify the test detects it at the claimed rate; run under the null and
  verify the false-positive rate approximates α. This is the project proving its
  own statistics rather than assuming them. Holm correction verified against a
  reference implementation.
- **`runner`** — a fake-engine mock returning canned logprobs, so grid
  enumeration, outcome classification, and Parquet writing are all testable with
  no GPU.
- **`gate`** — golden-file tests: a known-clean baseline must pass, a seeded
  regression must fail with a non-zero exit code.
- **`suite`** — hash stability test, so a silent corpus edit cannot invalidate
  historical results unnoticed.

---

## 10. Risks and go/no-go gates

### 10.1 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Null result** — nothing diverges beyond the floor | High | The tool is the deliverable; findings are the marketing. The positive control guarantees ≥1 finding if the instrument works. `unsupported`/`crashed` cells are also findings. A clean audit still closes #2031/#2646. |
| `prompt_logprobs` is itself buggy — the measurement path runs through the flakiest API surface (#13105, #3032) | High | Crashes are a recorded outcome class, not a failure. Fallback teacher-forcing path via a plain HuggingFace forward pass for cross-checking. |
| Noise floor swamps all effects | High | Enable `VLLM_BATCH_INVARIANT` to shrink the floor and re-measure — using the prior art as an instrument. |
| Top-20 cap limits distributional fidelity | Medium | Union-support renormalization (§5.1), limitation stated explicitly. |
| Cells not enableable | Medium | All 16 probed in week 1; `unsupported` is a recorded outcome. |
| AWQ checkpoints unavailable | Medium | Verified week 1; otherwise produced once via `llm-compressor` with the command committed. |
| vLLM version churn | Medium | Pinned version, stamped per row. |
| Scope creep (second engine, more models) | Medium | §2.2 is a written decision that creep must argue against. |
| Reviewer perceives ignorance of batch-invariance work | Medium | README and write-up open by citing it (§1.2). |

### 10.2 Gates

| Gate | When | Pass condition | If failed |
|---|---|---|---|
| G1 | End week 1 | WSL2 + vLLM running; all 16 cells probed for support; one cell captured end-to-end to Parquet | Fall back to Kaggle-only execution and continue |
| G2 | End week 2 | Reference self-comparison median per-prompt mean JSD **< 0.01 nats** (≈1.4% of the log 2 bound), so effects have headroom | Enable `VLLM_BATCH_INVARIANT`, re-measure. If still above threshold, report the floor itself as the study's primary finding and narrow claims to max-JSD and free-running metrics |
| G3 | End week 4 | Positive control fires — harness independently rediscovers #25262; F4 negative control is inert | **Stop.** Instrument unvalidated; no other finding is trustworthy until explained |

---

## 11. Schedule

| Week | Work | Gate |
|---|---|---|
| 1 | WSL2 + vLLM pinned; probe all 16 cells; AWQ checkpoint check; `suite` v1; `runner` → Parquet for one cell | **G1** |
| 2 | Both reference replicates; noise floor; `metrics` + unit tests; **commit pre-registration** | **G2** |
| 3 | Full 16-cell grid on model 1; `stats` — bootstrap, Holm, coverage simulation | |
| 4 | Factorial decomposition and interactions; survival curves; **positive + negative controls** | **G3** |
| 5 | Model 2; `gate` CI mode + GitHub Action; slice breakdown | |
| 6 | Write-up, figures, README; comment on `llm-compressor` #2031 / #2646 with the artifact | |

Stretch, only if week 6 is genuinely free: SGLang as a second engine, or a third
7–8B AWQ model.

---

## 12. Deliverables and success criteria

**Deliverables**

1. `pip`-installable tool with the six components of §7.1.
2. Test suite including the `stats` coverage simulation.
3. GitHub Action running the regression gate.
4. Committed pre-registration document.
5. Study write-up: 16-cell table, main effects, interaction terms, all with CIs;
   survival curves; slice breakdown.
6. Automated regression test reproducing vLLM #25262.
7. A comment on `llm-compressor` #2031 / #2646 linking the artifact.

**Success criteria — the project succeeds if all of these hold**

- The positive control fires and the negative control stays inert.
- The noise floor is measured and reported, not assumed.
- Every cell resolves to a recorded outcome, including `unsupported`/`crashed`.
- Every reported effect carries a CI, including null effects.
- The `stats` coverage simulation demonstrates correct detection and
  false-positive rates.
- The write-up positions the work correctly against batch-invariance prior art.

Explicitly **not** a success criterion: finding a large number of divergent
configurations. That is an outcome, not a goal, and making it a goal would create
pressure to overclaim.

---

## 13. References

- vLLM issue #25262 — prefix caching alters logprobs for identical shared
  prefixes (open / not planned)
- vLLM issue #13105 — prefix caching + `prompt_logprobs` assertion failure
- `llm-compressor` issue #2031 — KL-divergence evaluation tool request (open)
- `llm-compressor` issue #2646 — cheap KLD metric request (open)
- vLLM blog, "Keeping vLLM Production Quality" (Jul 2026)
- Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (Sept 2025)
- vLLM batch invariance — `VLLM_BATCH_INVARIANT`, PRs #25603 / #26373, issue
  #27433, `docs.vllm.ai/en/latest/features/batch_invariance/`
- vLLM CI `check_logprobs_close` — single-config vLLM-vs-HuggingFace logprob gate
