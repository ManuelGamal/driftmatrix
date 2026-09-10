# driftmatrix — pre-registered analysis plan

Frozen: 2026-09-10, git tag `prereg-2026-09-10`
Spec: `docs/superpowers/specs/2026-09-10-hybrid-cache-fidelity-design.md` (v4.5)
Engine: llama.cpp `llama-server` build `b10839-0cae43063`, Windows CUDA, RTX 4060 Laptop (8 GB)

This document fixes the main study's traffic, conditions, outcomes, tests and
decision rules before any main-study turn has run on any arm. It is not edited
after the tag. Departures are recorded in a dated **Deviations** section of the
main-study findings, each with its reason.

---

## 0. What had been seen at the freeze

| Evidence | What it examined | Role here |
|---|---|---|
| G0, G1 (`docs/findings/2026-09-10-g0.md`, `-g1.md`) | Restore vs recompute on synthetic raw-text prompts, no chat template. Restore was exact exactly at multiples of `n_ubatch` (66/66); hybrid drift about 26× dense by median TV | Informed hypotheses H1, H2, H4. Not confirmatory data |
| G2 (`docs/findings/2026-09-10-g2.md`) | The **recompute path only**, on chat-templated conversations (seed 2026, 3 round trips): rendering, tool-call rates, parsers, action margins by stratum | Validated the measurement path. No restore arm has run on chat-templated traffic |
| Main-study turns (seed 20260910) | Nothing | Confirmatory data |

Main-study data will live in `data/` (gitignored). Every report committed under
`docs/findings/` before this tag belongs to G0–G2.

## 1. Hypotheses

Architectures: **L3** Ornith-1.5-9B (hybrid recurrent), **L2** Spark-X2.5-4B
(sliding window), **L1** Qwen3-4B (dense), all Q4_K_M.

- **H1 Mitigation.** At snapped boundaries restore is bit-exact on every valid turn, on every model.
- **H2 Distributional existence.** At natural boundaries, restore diverges from recompute on L3.
- **H3 Behavioural existence.** At natural boundaries, restore changes the agent's action on L3.
- **H4 Architecture.** At natural boundaries, divergence on L3 exceeds L1 and L2.
- **H5 Snapping.** Snapping reduces divergence on L3, by more than it does on L1.
- **H6 Compounding.** On L3, divergence is larger in the long prefix band than the short one.
- **H7 Stratum.** On L3, action changes are more frequent for ambiguous than explicit requests.

## 2. Engine configuration

```
llama-server --jinja -c 8192 -ngl 99 -np 1 -b 2048 -ub 512 --cache-ram 0 --ctx-checkpoints 0 --slot-save-path <dir>
```

plus `--swa-full` for Spark. Never `--spec-type`, never `-ctk`/`-ctv`. Models are
the Q4_K_M GGUFs used in G1 and G2. Greedy decoding (`temperature 0`, `top_k 1`,
`seed 1234`), at most 192 new tokens, top-10 logprobs at every generated position. The build string is
stamped into every row. A different build is a deviation.

## 3. Traffic

- **Turns:** `generate_study_turns(90, seed=20260910, round_trips={"short": 8, "long": 60})`.
  That gives 90 base conversations × 2 prefix bands × 2 strata = **360 turns per model**,
  with **the same 360 turns on every model**.
- **Trace hash:** `5b5c50d53577f5b25f4326c2dbf868f0a889a6dfaabfbdd460ccfd5773f555d4` (`trace_hash`, stamped into every row).
- **Prefix bands:** the long-band history is a base's 60 completed tool round
  trips; the short-band history is its last 8. Both bands end with the same
  recent history and the same final request. Estimated prompt lengths: short
  ~0.9–1.4K tokens, long ~5.3–6.8K tokens, depending on the model.
- **Strata:** the explicit request is drawn with the history, from the three
  explicit templates in `traces/conversations.py`. The ambiguous twin replaces only
  the final request. Its type is chosen by base index modulo 5, so each type covers
  18 bases, the same bases in both bands:
  1. "Pull up that record again, the one from before."
  2. "Give me something on record N."
  3. "Can you total up records A and B?"
  4. "Add up the numbers from earlier."
  5. "Write that up for me."
- **Rendering:** offline from each GGUF's embedded chat template, with tools and
  `enable_thinking=false`, exactly as validated in G2.
- **Budget check, before any arm runs:** `scripts/study_manifest.py` tokenizes
  every turn on every model; it generates nothing. Every turn must satisfy
  prompt + 192 ≤ 8192 and have a snapped boundary > 0. If a long-band turn
  fails, the long band drops by 5 round trips (60 → 55 → …) for all models. If a
  short-band turn has a snapped boundary of 0, the short band grows by 1 round
  trip for all models. Either change is a recorded deviation, and the trace hash
  is re-stamped.

## 4. Conditions, boundaries and arms

- **Natural boundary `k`:** the longest common token prefix of the history
  rendered on its own and the full prompt (`reuse_boundary`). This is exactly
  what a server reuses. For Qwen3 it ends just before the last assistant message's
  content, because the template renders that message differently once a new user
  turn follows. The history is a full prefix on Ornith and Spark.
- **Snapped boundary:** `k − (k mod 512)`. When `k mod 512 = 0` the two coincide;
  the run is made once and counts in both conditions.
- **Conditions per model:** boundary B (natural, snapped) × prefix band P × stratum S
  = 8, with 90 turns each.
- **Arms per turn, in this order:** recompute₁, restore-natural₁,
  restore-snapped₁, then recompute₂, restore-natural₂, restore-snapped₂. The slot
  is erased before every arm. Restore evaluates `prompt[:k]`, saves, erases,
  restores, then completes with caching on. Recompute completes with caching off.
- **Arm proofs:** a restore arm is valid only if `timings.prompt_n` equals the
  suffix length and the saved slot holds exactly `k` tokens. A recompute arm is
  valid only if it evaluated the whole prompt.
- **Teacher-forced scoring is not part of this analysis.** Its path through
  `llama-server` has not been validated. Any later use is exploratory.

## 5. Outcomes

Within a replicate `r`, restore_r is compared with recompute_r. The **floor** is
the same comparison between recompute₁ and recompute₂.

- **Action:** the first complete tool call parsed with the model's parser (name
  plus canonical arguments), or *no call*. Every parsed call must pass the v4.4
  template round trip. A turn with a call that fails it in any arm is
  `parse_unfaithful`: it is excluded and counted.
- **Primary outcome — action change `A_t`:** 1 if the restore action differs
  from the recompute action **in both replicates**, otherwise 0. A change seen in
  only one replicate is counted and reported separately as non-reproducible.
- **Floor action change `F_t`:** 1 if recompute₁ and recompute₂ take different actions.
- **Exact:** identical generated token ids and bitwise-identical top-10 logprobs
  at every generated position.
- **Exact match:** identical generated token ids.
- **First divergence:** the first position where the token ids differ, right-censored when none do.
- **Drift:** JSD and TV at each context-matched position, from position 0
  through the first divergence inclusive (all positions if the outputs are
  identical). Each is computed over the top-10 union support plus an `<other>`
  bucket. Per turn, **mean JSD** and **max JSD** over those positions. Drift
  analyses use replicate 1; replicate 2 is used to check reproducibility.
- **Action margin (covariate):** from recompute₁. The smallest top-two logprob
  gap over position 0 and the positions of the first complete tool call. Bins:
  near-tie < 0.5 ≤ contested < 2.0 ≤ confident (nats).
- **State bytes:** after both arms, whether the saved slot files are byte-equal,
  and how many bytes differ. Descriptive only.

## 6. Exclusions and validity

- Invalid arms (`restore_not_used`, `recompute_not_clean`, `save_mismatch`,
  `crashed`, `timeout`) and `parse_unfaithful` turns are excluded from the
  comparisons they touch and counted per condition. If more than 10% of a
  condition's turns are excluded, no claim is made for that condition.
- If the floor is nonzero on any turn (tokens or logprobs differ between the two
  recomputes), tests for that condition use the sign test (§7). If it is nonzero
  on more than 5% of a model's turns, that model is reported as nondeterministic,
  and every claim about it is marked conditional.

## 7. Tests

Implemented in `src/driftmatrix/stats/exact.py`, `bootstrap.py` and `multiplicity.py`
at this tag. α = 0.05. Holm–Bonferroni within each family, never pooled.

- **Against the floor (families A and B):** `test_against_floor(treatment, floor)`
  over the condition's valid turns. If the floor is zero on every turn, the exact
  binomial test against a rate of 0 gives p = 0 when any turn diverges and p = 1
  otherwise. The engine is deterministic, so one reproducible divergence cannot
  arise under the null. In that case Holm cannot change a decision, and this is
  stated with the results. If the floor is nonzero on any turn: one-sided exact
  sign test on treatment − floor, ties dropped.
- **Between conditions (family C):** two-sided exact sign test on paired per-turn
  differences, ties dropped. For binary outcomes this is the exact McNemar test.
- **Intervals:** rates use Clopper–Pearson 95% intervals. Differences of means
  use a paired percentile bootstrap over turns (10,000 resamples, seed 0).

### Family A — behavioural (24)

For each model {L3, L2, L1} × B {natural, snapped} × P {short, long} × S
{explicit, ambiguous}: action change `A_t` against the floor `F_t`.

### Family B — distributional (16)

L3 only, 8 conditions × {per-turn mean JSD, per-turn max JSD}, each against its floor counterpart.

### Family C — mechanism contrasts (6)

All on mean JSD at natural boundaries unless stated. Pairs are matched by turn
(the same `conv_id` across models or boundaries) or by base.

| ID | Contrast | Pairs | Predicted sign |
|---|---|---|---|
| C1 | Snapping on L3: natural − snapped | 360 turns | > 0 |
| C2 | Snapping × architecture: (natural − snapped) on L3 − (natural − snapped) on L1 | 360 turns | > 0 |
| C3 | L3 − L1 | 360 turns | > 0 |
| C4 | L3 − L2 | 360 turns | > 0 |
| C5 | Compounding on L3: long − short | 180 (base × stratum) | > 0 |
| C6 | Stratum on L3: ambiguous − explicit, on action change `A_t` | 180 (base × band) | > 0 |

C2 is an interaction on the additive JSD scale.

## 8. Decision rules

- **Families A and B:** an effect is claimed when Holm rejects within the family
  **and** its interval excludes zero:
  - Family A: the Clopper–Pearson lower bound on the rate of `A_t` is above 0. With
    a nonzero floor, the bootstrap interval of `A_t − F_t` must exclude 0 instead.
  - Family B: the bootstrap interval of the per-turn (restore − floor) mean excludes 0.
- **Family C:** an effect is claimed when Holm rejects within the family, the
  bootstrap interval excludes zero, **and** the sign matches the prediction. A
  significant effect in the opposite direction is reported as evidence against
  the hypothesis.
- **H1** holds for a model only if every valid snapped turn is exact. Any
  inexact snapped turn is reported with its `k`, and falsifies H1 for that model.
- **Attribution to recurrent state** (H4) requires both C3 and C4. L2 − L1 is
  reported with its interval, descriptively.
- **H3** is supported when the L3 natural-boundary members of family A are
  claimed. H7 rests on C6 alone.

## 9. Secondary analyses (intervals only, no Holm claims)

- Action-change rate by margin bin, per model and boundary type, with
  Clopper–Pearson intervals. Predicted ordering: near-tie ≥ contested ≥ confident.
- Exact-match rate, and first-divergence curves (Kaplan–Meier, right-censored),
  by condition.
- Natural boundaries split by alignment (`k mod 512 = 0` or not).
- JSD by position within the output, per model: compounding across decoded tokens.
- Snapping cost: the distribution of re-evaluated tokens `k mod 512`.
- State-byte equality counts.
- Rates of non-reproducible action changes.
- **Self-inconsistency label:** a condition is self-inconsistent when its median
  per-turn mean JSD exceeds 2.0× the floor median. With a zero floor that means
  any positive median. This is a pre-specified engineering threshold.

## 10. Power and the null

- With 90 turns per condition, a true reproducible action-change rate of 2% is
  observed at least once with probability 1 − 0.98⁹⁰ = 83.8%. Pooling a model's 360
  natural-boundary turns gives 99.9% (`stats/power.py`).
- With no changes observed, the Clopper–Pearson 95% upper bound is 4.0% for 90
  turns and 1.02% for 360.
- C6 compares rare events and may be underpowered. Its interval is reported either way.
- **Null commitment:** *At natural agent-turn boundaries on llama.cpp build
  b10839-0cae43063, restore changed no agent actions in N turns on Ornith-1.5-9B
  Q4_K_M, bounding the rate below Y% (95% Clopper–Pearson upper bound).*

## 11. Order of work and stopping

1. **Budget check** (§3). Tokenization only.
2. **Gate G3 pilot:** `generate_study_turns(10, seed=20260911, …)`, short band,
   both strata, all three models (trace hash `f94004397c8cecc2e23227eaeb7ddb461647566a4e18b960f2b2725dd8804a65`). G3 passes when
   every arm proof holds, every snapped turn is exact, and at least one L3
   natural-boundary turn shows nonzero drift. Only those checks are looked at.
   Pilot turns never enter the confirmatory analysis. If G3 fails, the harness is
   fixed and the pilot rerun under a new, recorded pilot seed.
3. **Analysis code** is committed, and tested on fake-server data, before the
   main run. Its commit is recorded in the findings.
4. **Main study:** all 360 turns on all three models, run to completion. Until
   all three have finished, monitoring is limited to arm-validity counts and
   progress. No treatment effect is examined.
5. **Analysis** exactly as above. Anything else is labelled exploratory.
