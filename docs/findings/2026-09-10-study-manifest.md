# Main-study manifest — token budget and reuse boundaries

Date: 2026-09-10
Engine: llama.cpp `llama-server` build `b10839-0cae43063`
Raw reports: [`study/`](study/) (file names and build only, no local paths)
Pre-registration: `docs/preregistration/2026-09-10-analysis-plan.md`, section 3

**Verdict: every one of the 360 main-study turns fits the pre-registered budget on
all three models, so the round-trip counts stand unchanged (8 and 60).** Every
prompt plus 192 new tokens fits in the 8192-token context, and every snapped
boundary is above zero.

`scripts/study_manifest.py` tokenized each turn with the server's tokenizer.
Nothing was generated and no restore was run. The trace hash matched the
pre-registration on all three models: `5b5c50d53577f5b2…`.

## Measured

| Model | Band | Prompt tokens (min / median / max) | Natural boundary k (min / median / max) | Snapped k | Re-evaluated by snapping | Natural k aligned |
|---|---|---|---|---|---|---|
| Ornith-1.5-9B | short | 1200 / 1245 / 1283 | 1175 / 1219 / 1255 | 1024 | 151–231 | 0 of 180 |
| Ornith-1.5-9B | long | 6317 / 6439 / 6556 | 6294 / 6416 / 6531 | 6144 | 150–387 | 0 of 180 |
| Qwen3-4B | short | 950 / 992 / 1027 | 918 / 954 / 984 | 512 | 406–472 | 0 of 180 |
| Qwen3-4B | long | 5211 / 5316 / 5427 | 5178 / 5278 / 5392 | 5120 | 58–272 | 0 of 180 |
| Spark-X2.5-4B | short | 742 / 786 / 822 | 723 / 767 / 800 | 512 | 211–288 | 0 of 180 |
| Spark-X2.5-4B | long | 4188 / 4298 / 4407 | 4168 / 4277 / 4388 | 4096 | 72–292 | 0 of 180 |

- **No natural boundary is aligned.** With `n_ubatch = 512`, a boundary is aligned
  about once in 512 turns. All 1,080 natural restores are misaligned, and the
  natural-vs-snapped contrast is never degenerate.
- **One snapped boundary per model and band.** Every boundary in a band falls
  between the same two multiples of 512, so snapping always lands on one value.
  It re-evaluates at most 472 prompt tokens (Qwen3, short band).
- **The prompt extends past k by 18–29 tokens on Ornith, 12–23 on Spark and 25–47
  on Qwen3.** That span is the final request plus the generation prompt. Qwen3's
  is longer because its template re-renders the last assistant message once a new
  user turn follows (pre-registration section 4), so the reusable prefix ends
  before that message's content.

## Against the pre-registration's estimates

Section 3 estimated prompt lengths from character counts: about 0.9–1.4K tokens
short and 5.3–6.8K long. Ornith and Qwen3 fall inside those ranges. **Spark does
not: 742–822 short and 4188–4407 long.** Its tokenizer packs more characters per
token than the estimate assumed.

This is not a deviation. Bands are defined by round trips (8 and 60), the budget
rule passed without adjustment, and no pre-registered test changes. It does bear
on interpretation, and the main-study findings must state it:

- The architecture contrasts C3 (L3 − L1) and C4 (L3 − L2) pair the **same
  conversations**, not the same token lengths. In the long band Ornith reuses about
  6.4K tokens, Qwen3 about 5.3K and Spark about 4.3K. If drift grows with prefix
  length, as H6 predicts on the hybrid, part of the L3 − L2 difference could come
  from length rather than architecture.
- To make that separable, the findings will report C3 and C4 within each band
  alongside the pooled, pre-registered values, and label the per-band values
  exploratory.

## Limits of this record

- These three runs did not check which model the server had loaded. The script
  now refuses to run on a mismatch, and so does the main-study runner. The counts
  differ by model the way the three templates predict, and the budget has at least
  1,400 tokens of headroom on every model, so no conclusion here depends on it.
- The "history not a prefix" column printed no values (a display bug, since fixed).
  The offline render check shows every Qwen3 turn loses its last 38–69 history
  characters to the template, and no Ornith or Spark turn does. The main-study
  records will count this in tokens per turn.
