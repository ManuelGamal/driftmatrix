# Spark window follow-up — results

Date: 2026-09-11
Engine: llama.cpp `llama-server` build `b10839-0cae43063`, Windows, CUDA, RTX 4060 Laptop (8 GB)
Plan and predictions: [`2026-09-11-spark-window-plan.md`](2026-09-11-spark-window-plan.md), committed before any run (`e3de45a`)
Raw reports: [`followup/`](followup/) (no local paths). Regenerate with `python scripts/spark_window_analyze.py --dir docs/findings/followup`
Status: **exploratory** — not pre-registered, one prompt, exact sets rather than rates

**Verdict: under the default configuration, restore on the sliding-window model
is bit-exact only when the reuse boundary is a multiple of `n_ubatch` *and* lies
within the 512-token window.** That rule (`window_limit`) fits all 39 proven
cases across three micro-batch sizes. Its closest rival, a single-micro-batch
limit, fails twice.

Two diagnostic runs located parts of the cause:

- **Turning flash attention off** extends exactness to 2,560 tokens, but not to
  3,072 and beyond.
- **Without `--swa-full`**, llama-server silently discarded every restored state
  and recomputed the whole prompt.

## What ran

`scripts/spark_window_probe.py`, one server configuration per run, on Spark-X2.5-4B Q4_K_M:

- **Prompt:** the first long-band main-study turn (`el20260910_000`), rendered
  through Spark's chat template, 4,268 tokens.
- **Boundaries:** 13, from 256 to 4,096.
- **Per case:** 16 generated tokens with top-10 logprobs.
- **Checks reused from G1:** recompute run twice as the floor, both arm proofs,
  and bitwise exactness. The recompute floor was bit-exact in all five runs.

Server flags were the main study's (`-c 8192 -ngl 99 -np 1 -b 2048 --cache-ram 0
--ctx-checkpoints 0 --slot-save-path`), plus:

| Run | `-ub` | Flash attention | `--swa-full` | Restores proven |
|---|---|---|---|---|
| `ub256` | 256 | auto | yes | 13 / 13 |
| `ub512` | 512 | auto | yes | 13 / 13 |
| `ub1024` | 1024 | auto | yes | 13 / 13 |
| `ub512_fa-off` | 512 | off | yes | 13 / 13 |
| `ub512_no-swa-full` | 512 | auto | no | **0 / 13** |

## Exact grid

E = bit-exact, d = inexact, ! = restore not proven.

| k | `ub256` | `ub512` | `ub1024` | `ub512_fa-off` | `ub512_no-swa-full` |
|---|---|---|---|---|---|
| 256 | **E** | d | d | d | ! |
| 384 | d | d | d | d | ! |
| 512 | **E** | **E** | d | **E** | ! |
| 640 | d | d | d | d | ! |
| 768 | d | d | d | d | ! |
| 1024 | d | d | d | **E** | ! |
| 1280 | d | d | d | d | ! |
| 1536 | d | d | d | **E** | ! |
| 2048 | d | d | d | **E** | ! |
| 2560 | d | d | d | **E** | ! |
| 3072 | d | d | d | d | ! |
| 3584 | d | d | d | d | ! |
| 4096 | d | d | d | d | ! |

## The rules, scored on the three default runs

| Rule | Exact when | Contradictions (of 39 proven cases) |
|---|---|---|
| `window_limit` | aligned and k ≤ 512 | **0** |
| `single_microbatch` | aligned and k ≤ `n_ubatch` | 2: k = 512 exact at `n_ubatch` 256; k = 1024 inexact at `n_ubatch` 1024 |
| `window_regardless_of_alignment` | k ≤ 512 | 6: every misaligned k ≤ 512 was inexact |
| `tiling_only` | aligned | 20 |

Every exact set matches the plan's prediction for `window_limit`: {256, 512} at
`n_ubatch` 256, {512} at 512, and none at 1024. At `n_ubatch` 1024, no boundary is
exact at all, because the smallest aligned boundary is already beyond the window.

## Diagnostics

**Flash attention off** (`ub512` against `ub512_fa-off`):

- **Became exact:** 1,024, 1,536, 2,048 and 2,560.
- **Became inexact:** none.
- **Still inexact beyond that:** 3,072, 3,584 and 4,096, with worst-position TV
  1.9–3.3e-6. With flash attention on auto, drift at those boundaries was
  1.9–3.5e-5, about ten times larger.

The attention kernel's path over cached cells is therefore part of the cause,
but not all of it. This data does not explain what changes between 2,560 and
3,072. With one prompt, a limit on the boundary cannot be told apart from a
limit on the suffix length: 1,708 tokens re-evaluated at k = 2,560 against
1,196 at 3,072. The run did not record which kernel "auto" selected in the
default runs.

**Without `--swa-full`:**

- The server saved every slot with exactly `k` tokens.
- On all 13 restores it still evaluated the whole prompt (`prompt_n` = 4,268).
  The restored state was silently discarded, including at k = 256, inside the
  window.
- Outputs were therefore bit-identical to recompute at every boundary.

**A harness that did not check `prompt_n` would have reported the sliding-window
model as exact everywhere in this configuration.** It is the silent-reprocessing
failure the arm proofs exist for; llama.cpp #20225 reports the same pattern on
hybrid models.

## Magnitude

Every inexact Spark restore here drifted very little: worst-position TV at most
1.2e-4, and no generated token changed within 16 positions on this prompt. That
matches the main study, where Spark's inexact restores changed an action only
at near-tie decisions (3 of 90 long-band ambiguous turns at the snapped boundary).

## What it means

- **Snapping is not a mitigation for sliding-window models once the reusable
  prefix is longer than the window.** In the default configuration, no aligned
  boundary beyond 512 tokens was exact. Every realistic agent prompt is longer
  than that; the main study's Spark prompts were 742–4,407 tokens.
- **The configuration matters more for sliding-window models than for the
  others.** Flash attention changes where exactness ends. `--swa-full` decides
  whether a restore is used at all.
- **The main-study result for Spark is explained in outline.** Its short band
  snapped to 512 (inside the window: exact); its long band snapped to 4,096
  (outside: inexact).

## Limits

- One prompt, one build, one GPU, 16 generated positions; exact sets, not rates.
- The flash-attention-off limit (between 2,560 and 3,072) may depend on this
  prompt's length. It was not varied.
- The mechanism behind the window limit, and behind the remaining inexactness
  with flash attention off, is located, not explained. Tracing it needs
  llama.cpp's sliding-window cache and attention code, which this study did not
  instrument.
