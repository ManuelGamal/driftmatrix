# Spark window follow-up — plan and predictions

Date: 2026-09-11. Written and committed **before** any run. Exploratory: not
pre-registered and without multiplicity control.

## Question

Why does snapping the reuse boundary to a multiple of `n_ubatch` stop giving
bit-exact restores on the sliding-window model, when it works on the dense and
hybrid models?

## Evidence so far

| Source | Spark, `n_ubatch` = 512 | Dense and hybrid |
|---|---|---|
| G0 | exact at k = 512, inexact at k = 1024 | exact at every multiple of 512 tested |
| Main study | exact at k = 512 (180 turns), inexact at k = 4096 (180 turns) | exact at 512–6144 (720 turns each) |

Spark's sliding window is also 512 tokens, and every observation so far used
`n_ubatch = 512`. Several rules fit the evidence equally well.

## Rules and predicted exact sets

Boundaries tested: 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2560, 3072,
3584, 4096, on one long-band main-study turn (`el20260910_000`, about 4.2K tokens).

| Rule | Exact when | `n_ubatch` 256 | `n_ubatch` 512 | `n_ubatch` 1024 |
|---|---|---|---|---|
| `tiling_only` | k is a multiple of `n_ubatch` | every multiple of 256 | every multiple of 512 | 1024, 2048, 3072, 4096 |
| `window_limit` | … and k ≤ 512 (the window) | 256, 512 | 512 | none |
| `single_microbatch` | … and k ≤ `n_ubatch` (the prefix fit in one micro-batch) | 256 | 512 | 1024 |
| `window_regardless_of_alignment` | k ≤ 512, aligned or not | 256, 384, 512 | 256, 384, 512 | 256, 384, 512 |

`tiling_only` is already contradicted by G0 and the main study; it stays in as
the rule that holds on the other two architectures. The two runs that separate
the rest are `n_ubatch` 256 and 1024.

## Runs

Same server flags as the main study
(`-c 8192 -ngl 99 -np 1 -b 2048 --cache-ram 0 --ctx-checkpoints 0 --slot-save-path`)
plus:

| Label | `-ub` | Flash attention | `--swa-full` | Purpose |
|---|---|---|---|---|
| `ub256` | 256 | auto | yes | scored against the rules |
| `ub512` | 512 | auto | yes | scored; reproduces the main study's setting |
| `ub1024` | 1024 | auto | yes | scored |
| `ub512_fa-off` | 512 | off | yes | diagnostic: does the attention kernel matter? |
| `ub512_no-swa-full` | 512 | auto | no | diagnostic: does the full-size SWA cache matter? |

## What each outcome would mean

- **`window_limit` holds.** Exactness on Spark ends at the window, whatever the
  micro-batch size. The mitigation then has to change for sliding-window models:
  snapping cannot help once a reusable prefix is longer than the window.
- **`single_microbatch` holds.** The window is a coincidence. Something goes wrong
  once a restored prefix spans more than one micro-batch, and this build does it
  only on Spark's architecture.
- **Neither holds.** The rule is something else. The grid shows where exactness
  ends, and the diagnostics may still point at a component.
- **Flash attention off changes the exact set:** the attention kernel's path over
  cached cells is involved. **No change:** it is not.
- **Without `--swa-full`:** restores beyond the window may not be provable at all,
  since the cache holds only the window. Any change in the exact set, or unproven
  restores, is reported as found.

Results will be recorded in `docs/findings/2026-09-11-spark-window.md`, whichever
way they come out.
