# Paper handoff — turn driftmatrix into a short paper

Written 2026-10-02 at the end of a long applications session. Start a fresh session in this repo and read this first.

## Goal

A ~6-page paper on driftmatrix, posted to **arXiv by about 2026-10-16** (so the link can go on CVs and into the CISPA / Summer@EPFL / master's applications due in November), then submitted to a **workshop** on ML systems or efficient LLM inference.

Author: Manuel Gamal (Ain Shams University). Commits are authored as the user only — no Claude co-author lines.

## Why it matters

Manuel is a final-year undergraduate (GPA 2.72/4.0) applying to research internships and master's programmes. Several roles he otherwise fits gate on a "publication record". This paper is the single biggest lever on those applications.

## Source material (all in this repo)

- `README.md` — the headline results, already in near-paper form.
- `docs/findings/2026-09-11-main-study.md` — the full main-study write-up.
- `docs/preregistration/2026-09-10-analysis-plan.md` — the pre-registration (cite it; state what was pre-registered vs exploratory).
- `docs/findings/2026-09-10-g0.md` … `g3.md` — the gate studies (bit-exactness, mechanism, n_ubatch fix).
- `docs/findings/2026-09-11-spark-*.md` — the sliding-window follow-ups.
- `docs/figures/*.png` — action_changes, drift_per_turn, g1_exactness_grid, margin_bins.
- `docs/superpowers/specs/2026-09-10-hybrid-cache-fidelity-design.md` — design rationale and related work (TraceLab etc.).

## Headline claims (verify every number against the findings docs before writing it)

- Restoring saved state at a natural agent-turn boundary was never bit-exact (1,080/1,080 turns), on hybrid-recurrent, sliding-window and dense models (llama.cpp build b10839, Q4_K_M, RTX 4060 Laptop, greedy).
- After restore, 2–12% of ambiguous requests got a different action on every architecture; 0 of 540 explicit-request turns changed.
- Changes concentrate near ties: 29% of turns with top-2 margin < 0.5 nats vs 0.1% of confident turns.
- Mechanism: micro-batch tiling; restores are bit-exact iff the reuse boundary aligns with `n_ubatch` (66/66 predictions held).
- Fix: round the reuse boundary down to a multiple of `n_ubatch` → bit-exact on hybrid and dense (median 158–443 re-evaluated tokens); on sliding-window it worked at 512 tokens and failed at 4,096.
- Side finding: a suspected parser bug was the tool grammar; testing pending llama.cpp PR #26472 found a Laguna-XS regression, reported with a failing test and a fix.

## Honesty rules (non-negotiable)

- Report the pre-registered predictions that failed (dense/sliding-window "unaffected" was wrong; the fix fails on sliding-window at 4,096) in a Limitations section. In a paper this adds credibility.
- Separate pre-registered from exploratory analyses explicitly.
- One machine, one quantisation, greedy decoding, three models — say so.
- No claim beyond what the data supports; no inflated framing.

## Suggested shape (~6 pages + appendix)

1. Introduction — agent serving reuses cached state on nearly every request and assumes it is invisible; is it?
2. Background — prefix/KV-cache reuse in llama.cpp; hybrid recurrent vs sliding-window vs dense state.
3. Method — pre-registration, models, agent traffic with real tool calls, ambiguous vs explicit requests, server-verified runs, blinding.
4. Results — exactness, action changes, the margin (near-tie) analysis.
5. Mechanism and fix — micro-batch tiling, the n_ubatch alignment rule, cost of the fix.
6. Limitations, related work, conclusion.

## Open tasks for the new session

1. Check (web) the exact deadlines and page limits of candidate venues for late 2026 / early 2027: EuroMLSys 2027, MLSys 2027 workshops, ICLR 2027 workshops on efficient inference / systems, and any NeurIPS-2026-adjacent venue still open. Pick one primary and one backup.
2. Check arXiv's current endorsement rules for a first-time submitter in cs.LG / cs.DC, and who could endorse (a professor; or a llama.cpp maintainer from the PR #26472 thread).
3. Write the paper in LaTeX under `docs/paper/` (a standard two-column or the venue template), build to PDF, and check every figure and number against the findings docs.
4. Ask Manuel before posting anything publicly or contacting anyone (endorsers, maintainers).

Start with the superpowers brainstorming skill to agree the venue, framing and outline before writing.
