# Main study — does restoring saved state change what an agent does?

Date: 2026-09-11
Engine: llama.cpp `llama-server` build `b10839-0cae43063`, Windows, CUDA, RTX 4060 Laptop (8 GB)
Pre-registration: `docs/preregistration/2026-09-10-analysis-plan.md` (tag `prereg-2026-09-10`)
Analysis code: commit `4985f42`, committed and tested before the main run (pre-registration 11.3)
Results: [`study/analysis.json`](study/analysis.json) (every family member, hypothesis, secondary analysis and the per-turn table), [`study/changed_turns.json`](study/changed_turns.json) (every turn whose action changed, with both outputs). Raw records (about 70 MB) are kept locally in `data/study/`.
Regenerate: `python scripts/analyze_study.py --gguf ornith=… --gguf qwen3=… --gguf spark=…`

## The answer in four sentences

Restoring saved state instead of recomputing changed the agent's action — which
tool it called, with which arguments, or whether it called one at all — on
**2–12% of ambiguous requests on all three architectures**, and on **none of
1,080 explicit-request turns**. The hybrid recurrent model changed most in the
long prefix band (11 of 90 turns, 12.2%), but the dense and sliding-window
models were **not** immune, contrary to the spec's prediction. Snapping the reuse
boundary down to a multiple of `n_ubatch` made every restore **bit-exact on the
hybrid and dense models**, removing all drift and all action changes. It did
**not** protect the sliding-window model once its reuse boundary was 4,096 tokens.

## Validity

| Check | Result |
|---|---|
| Turns complete, on the pre-registered build and trace hash | 360 / 360 on each model |
| Arm proofs (restore used only its suffix; slot held exactly `k`; recompute evaluated everything) | 6,480 / 6,480 arms valid |
| Recompute floor bit-exact (recompute₁ vs recompute₂) | 1,080 / 1,080 turns — the engine was deterministic throughout |
| Restore reproducible (restore₁ vs restore₂, bitwise) | every turn, every condition |
| Parsed tool calls that fail the template round trip | 0 — no turn excluded |
| Natural boundaries aligned to 512 | 0 / 1,080, as the manifest predicted |

Because the floor was bit-exact on every turn, every family A and B member used
the zero-floor test (section 7): one reproducible change rejects, and the
Clopper–Pearson interval carries the magnitude. No model was flagged
nondeterministic, so no claim is conditional.

## Family A — did restore change the action? (24 members)

Action changes that reproduced in both replicates, out of 90 turns per cell.
**Bold** = claimed under section 8. Every unmarked 0/90 has a 95% upper bound of 4.0%.

| Model | Boundary | Short, explicit | Short, ambiguous | Long, explicit | Long, ambiguous |
|---|---|---|---|---|---|
| Ornith (hybrid) | natural | 0 | **3 (3.3%, CI 0.7–9.4%)** | 0 | **11 (12.2%, CI 6.3–20.8%)** |
| Ornith (hybrid) | snapped | 0 | 0 | 0 | 0 |
| Spark (sliding window) | natural | 0 | **2 (2.2%, CI 0.3–7.8%)** | 0 | **4 (4.4%, CI 1.2–11.0%)** |
| Spark (sliding window) | snapped | 0 | 0 | 0 | **3 (3.3%, CI 0.7–9.4%)** |
| Qwen3 (dense) | natural | 0 | **4 (4.4%, CI 1.2–11.0%)** | 0 | **3 (3.3%, CI 0.7–9.4%)** |
| Qwen3 (dense) | snapped | 0 | 0 | 0 | 0 |

7 of 24 members claimed. No change was seen in only one replicate.

## Family B — did restore change the distribution? (hybrid only, 16 members)

Per-turn mean JSD over context-matched positions: median across turns, with the
mean (the pre-registered estimate) in brackets.

| Boundary | Short, explicit | Short, ambiguous | Long, explicit | Long, ambiguous |
|---|---|---|---|---|
| natural | 2.4e-6 (7.9e-6) | 7.1e-5 (6.0e-3) | 2.0e-6 (1.9e-5) | 1.5e-4 (7.2e-3) |
| snapped | 0 exactly | 0 exactly | 0 exactly | 0 exactly |

All 8 natural-boundary members claimed, for both mean and max JSD; all 8 snapped
members are exactly zero. Every natural restore on every model was inexact, even
where the tokens matched: all 540 explicit natural turns produced identical text,
yet no turn had bitwise-identical logprobs.

For reference, the same medians on the controls at natural boundaries are
short/long explicit 9.0e-8 / 8.6e-8 on Qwen3 and 3.1e-8 / 3.3e-8 on Spark, and
short/long ambiguous 1.4e-5 / 8.0e-6 on Qwen3 and 6.6e-5 / 6.3e-5 on Spark. On
explicit requests the hybrid's median drift is about 25× the dense model's,
matching G1's ~26×.

## Family C — mechanism contrasts (6 members)

| Contrast | Pairs | Turns where the first side drifts more / less | Estimate (mean JSD difference) | 95% CI | Claimed |
|---|---|---|---|---|---|
| C1 snapping on the hybrid (natural − snapped) | 360 | 360 / 0 | 3.3e-3 | [1.6e-3, 5.4e-3] | **yes** |
| C2 snapping × architecture (hybrid vs dense) | 360 | 291 / 69 | 1.1e-3 | [−1.9e-3, 3.6e-3] | no |
| C3 hybrid − dense, natural | 360 | 291 / 69 | 1.1e-3 | [−1.9e-3, 3.6e-3] | no |
| C4 hybrid − sliding window, natural | 360 | 304 / 56 | 3.0e-3 | [1.3e-3, 5.0e-3] | **yes** |
| C5 compounding on the hybrid (long − short) | 180 | 84 / 96 | 6.3e-4 | [−3.0e-3, 4.3e-3] | no |
| C6 stratum on the hybrid (ambiguous − explicit, action change) | 180 | 14 / 0 | 0.078 | [0.039, 0.117] | **yes** |

- **C2 and C3 are numerically identical, and that is correct.** Snapped drift was
  exactly zero on both the hybrid and the dense model, so the interaction reduces
  to the natural-boundary difference.
- **C3 is not claimed, although the hybrid drifted more on 291 of 360 turns.** The
  sign test rejects strongly (p ≈ 1.5e-33), but the pre-registered rule also
  requires the bootstrap interval of the *mean* difference to exclude zero, and it
  does not. A handful of dense-model ambiguous turns drifted heavily: the largest
  per-turn mean JSD was 0.36 on Qwen3, against 0.20 on Ornith. The hybrid drifts
  more on a typical turn; it is not established that it drifts more on average.
- **C5 is not claimed.** Longer prefixes did not increase distributional drift
  on the hybrid. Its long-band action-change count (11 against 3) is a
  descriptive observation only; no test compares the two.

## Hypotheses

| | Hypothesis | Verdict | Basis |
|---|---|---|---|
| H1 | Snapped restores are bit-exact | **holds on Ornith and Qwen3; fails on Spark** | 360/360, 360/360; Spark 180/180 short-band exact, 180/180 long-band inexact |
| H2 | Distributional divergence at natural boundaries on the hybrid | **supported** | all 8 natural family B members |
| H3 | Behavioural divergence at natural boundaries on the hybrid | **partial** | ambiguous members claimed; explicit members 0/90 each |
| H4 | Hybrid diverges more than dense and sliding window | **partial** | C4 claimed, C3 not |
| H5 | Snapping reduces divergence, more on hybrid than dense | **partial** | C1 claimed, C2 not |
| H6 | Divergence compounds with prefix length on the hybrid | **not supported** | C5 |
| H7 | Action changes concentrate in the ambiguous stratum | **supported** | C6; and all 30 changes on every model were ambiguous |

A hypothesis is "supported" only when every member it names is claimed; the
per-member results above are the inferential units.

## Snapping does not protect the sliding-window model beyond its window

Spark's snapped restores were bit-exact at `k = 512` (all 180 short-band turns)
and inexact at `k = 4096` (all 180 long-band turns). At 4,096 snapping made no
visible difference: token-identical outputs 165/180 against 166/180 for natural,
3 action changes against 4, and the same order of drift. State bytes agree: the
snapped slot files were byte-equal on the 180 short-band turns and different on
all 180 long-band turns.

This replicates G0's unexplained observation at scale. There, Spark was exact at
`k = 512` and inexact at `k = 1024`, both multiples of `n_ubatch`, and the note
recorded that Spark's sliding window is also 512 tokens. What the data establish:
**for the sliding-window model, alignment to `n_ubatch` is not sufficient for
exactness at 1,024 or 4,096 tokens.** What they cannot separate is why. Every
snapped boundary above the window was inexact, and the only exact snapped
boundary equals the window. The G3 pilot ran the short band only, so it could not
have seen this. A follow-up that varies `k` and `n_ubatch` independently of the
512-token window on Spark would isolate it (exploratory; not pre-registered).

## Where the changes happen

**At near-ties, as predicted.** Action changes at natural boundaries by the
recompute arm's action margin (section 9):

| Model | Near-tie (< 0.5 nats) | Contested (0.5–2) | Confident (≥ 2) |
|---|---|---|---|
| Ornith | 12 / 32 (38%) | 2 / 59 (3%) | 0 / 269 |
| Spark | 5 / 28 (18%) | 1 / 54 (2%) | 0 / 278 |
| Qwen3 | 4 / 13 (31%) | 2 / 30 (7%) | 1 / 317 |
| All three | **21 / 73 (29%)** | **5 / 143 (3.5%)** | **1 / 864 (0.1%)** |

The pre-registered ordering (near-tie ≥ contested ≥ confident) holds on every
model. Three quarters of the changes come from 7% of turns.

**What changed.** Of the 30 changes (all reproduced in both replicates):

- 11 were the same tool called with different arguments, for example a
  `compute_sum` over a different list of numbers, or `lookup_record` for record
  17 instead of 11.
- 10 were a tool call that became a text answer or clarifying question.
- 9 were a clarifying question that became a tool call.
- None switched to a different tool.

By request type: "Add up the numbers from earlier" 20, "Pull up that record
again" 8, "Write that up for me" 2. The two tool-choice requests ("Give me
something on record N", "Can you total up records A and B?") produced no changes.
That matches G2, where they never produced near-ties.

**Examples** (`study/changed_turns.json` has all 30):

- Ornith, long band, "Pull up that record again": recompute calls
  `lookup_record(73)`; restore answers in text, "The record you're referring to is
  record 73, which has field=beta value=701."
- Ornith, short band, "Write that up for me": recompute asks "Could you clarify
  what you'd like me to write up?"; restore calls `format_report` with a summary.
- Ornith, long band, "Add up the numbers from earlier": both call `compute_sum`,
  but restore drops the value 65 from a 23-number list, so the tool computes a
  different total.

## Exploratory: architecture contrasts within each prefix band

The study manifest committed to this check, because the three models reuse
different token lengths for the same conversation. Within each band, the hybrid
still drifted more than the sliding-window model: short 149/31 turns, CI
[7.1e-4, 5.7e-3]; long 155/25, CI [8.5e-4, 6.3e-3]. Against the dense model the
interval included zero in both bands. The C4 result is therefore not an artefact
of Ornith's longer prompts.

## Against the pre-registered predictions and the spec's target headline

The spec's target headline expected action changes on the hybrid and "~0%" on
quantization-matched dense and sliding-window models. **That part was wrong.**

- **Right:**
  - restore at a natural agent-turn boundary is never bit-exact
  - snapping to `n_ubatch` restores exactness on the hybrid and the dense model
  - changes concentrate at near-tie decisions and in ambiguous requests
  - on a typical turn the hybrid drifts more than both controls
- **Wrong or not shown:**
  - the dense and sliding-window models also changed actions, at similar rates in
    the short band
  - the hybrid's mean drift was not shown to exceed the dense model's
  - drift did not compound with prefix length
  - snapping does not generalise to the sliding-window model

The resulting headline, scoped to this build and quantization:

> On llama.cpp b10839, restoring saved state at a natural agent-turn boundary was
> never bit-exact on hybrid recurrent, sliding-window or dense models. It changed
> the agent's action on 2–12% of ambiguous requests on all three (0 of 1,080
> explicit-request turns), almost always where the model was near a tie. Snapping
> the reuse boundary to a multiple of `n_ubatch` — at a median cost of 158–443
> re-evaluated tokens — made restore bit-exact and removed every change on the
> hybrid and dense models, but not on the sliding-window model at 4,096 tokens.

## Limits

- **One build, one quantization (Q4_K_M), one GPU, `n_ubatch = 512`.** Kernels,
  and so exactness, can differ on other builds and hardware.
- **Synthetic agent traffic** from three mock tools and five ambiguous request
  types. Real agent traffic may contain more or fewer near-ties; the margin table
  is the transferable part.
- **Rates rest on few events** (2–11 per cell). The intervals are wide, and no test
  compares action-change rates between architectures. The spec pre-registered the
  architecture contrasts on JSD.
- **Greedy decoding.** With sampling, drift would shift sampled tokens rather than
  flip greedy ones; this study does not measure that.
- **Spark's inexact snapped boundaries are not explained**, only located.

## Deviations and disclosures

- **No deviations** from the pre-registered protocol: all turns, arms, tests and
  decision rules ran as frozen.
- **The G3 pilot disclosure** (drift counts on the controls seen early) is in
  `2026-09-10-g3.md`.
- **How hypothesis summaries read.** "Supported" requires every member a
  hypothesis names to be claimed. The pre-registration's wording for H3 ("when the
  L3 natural-boundary members … are claimed") is read that way, documented in
  `analysis.py` before the run.
- **Runtime:** 2.15 h (Ornith), 1.07 h (Qwen3) and 0.76 h (Spark) of arm time.
