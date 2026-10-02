# driftmatrix paper — Design Spec

Date: 2026-10-02
Status: design agreed in conversation; awaiting review of this document
Brief: `docs/paper/HANDOFF.md`

## 1. Goal

A short paper on the driftmatrix results, written in LaTeX under `docs/paper/`.

- **First use:** an arXiv preprint by about 2026-10-16, so the link can go on
  CVs and into applications due in November.
- **Second use:** a submission to EuroMLSys 2027 (primary). An ICLR 2027
  workshop is the backup, chosen once that workshop list is published.
- **Author:** Manuel Gamal, Ain Shams University, sole author.

Success means: a PDF that builds from the repository with one command, fits the
page limit, and in which every number and every reference has been checked
against its source.

## 2. Constraints

### 2.1 Venue

EuroMLSys 2027's call is not posted yet. The paper assumes the 2026 rules:
6 pages excluding references, two-column ACM SIGPLAN format at 10 pt,
single-blind review, an appendix of any length, deadline in late February.
If the 2027 call differs, the difference is handled as a revision, not now.

### 2.2 Honesty rules

These come from the handoff and are not negotiable.

- Every number is verified against the findings documents or the committed
  reports before it appears.
- Pre-registered and exploratory analyses are labelled separately (section 4).
- Pre-registered predictions that failed are reported in the Limitations
  section and in the hypotheses table.
- The scope is stated plainly, in the abstract and in Limitations: one engine
  build (`b10839`), one quantisation (Q4_K_M), one GPU (RTX 4060 Laptop),
  `n_ubatch = 512`, greedy decoding, three models, synthetic traffic.
- No claim goes beyond the data. No claims about other engines.

### 2.3 Out of scope

- New experiments or new analyses of the raw records.
- Posting to arXiv, submitting to a venue, or contacting anyone (endorsers,
  maintainers). Manuel does these, or approves each one explicitly.
- Changes to the README figures or to any findings document.

### 2.4 Commits

Authored as Manuel only, with no co-author lines. Work happens on the `paper`
branch.

## 3. Framing

The paper leads with the behavioural consequence: restoring cached state changes
what an agent does, at near-tie decisions, on all three architectures. The
mechanism and the mitigation form the second half.

Working title: *Does Restoring a Prompt Cache Change What an Agent Does? A
Pre-Registered Study of Cache Fidelity in llama.cpp*

The paper does not claim that prompt caching *can* change output; llama.cpp
documents that. It claims to have measured when reuse is exact, how the size of
the difference depends on architecture, whether it changes agent actions, and
what a mitigation costs.

Contributions, as the introduction will list them:

1. A pre-registered measurement of restore against recompute on agent-shaped
   traffic, across hybrid recurrent, sliding-window and dense models.
2. The behavioural result: action changes on ambiguous requests on all three
   architectures, none on explicit requests, concentrated at near-ties.
3. The mechanism (micro-batch tiling) and the alignment rule.
4. A mitigation (snapping the reuse boundary), its cost, and where it fails.

## 4. Evidence labels

Defined once in the Method section and applied to every result.

| Label | Covers |
|---|---|
| Confirmatory | Pre-registered families A, B and C; hypotheses H1–H7 |
| Pre-registered secondary | Analyses the plan listed with intervals only and no corrected test: the margin-bin analysis, snapping cost, state-byte counts |
| Exploratory | Gates G0 and G1, which ran before the freeze; the within-band architecture contrasts; the Spark window follow-up |

Consequences the text must respect:

- The margin result (21/73 near-tie against 1/864 confident) is pre-registered
  secondary. It is reported with intervals and not described as a tested claim.
- The mechanism evidence (66 of 66 restores in G1) is exploratory and comes from
  one prompt per model.
- The Spark window rule (0 contradictions in 39 cases) is exploratory and comes
  from one prompt.

## 5. Outline and page budget

Six pages excluding references.

| § | Section | Pages | Content |
|---|---|---|---|
| | Abstract | — | About 150 words, including the scope |
| 1 | Introduction | 0.75 | Agents reuse cached state on nearly every request and assume it is invisible; the question; the four contributions |
| 2 | Background | 0.5 | Prefix reuse and slot save/restore in llama.cpp; `n_ubatch`; hybrid recurrent, sliding-window and dense state |
| 3 | Method | 1.25 | Pre-registration and what had been seen at the freeze; models; traffic (90 bases × 2 bands × 2 strata); natural and snapped boundaries; six arms; arm proofs; outcomes; tests and decision rules; evidence labels |
| 4 | Results | 1.5 | Validity checks; exactness; family A; margin analysis; what changed; distribution drift in brief; verdicts on H1–H7 |
| 5 | Mechanism and mitigation | 1.0 | Micro-batch tiling; the G1 grid; snapping and its cost; the sliding-window failure and the follow-up |
| 6 | Limitations | 0.4 | Failed predictions; scope limits; few events per cell; the unexplained sliding-window cause |
| 7 | Related work | 0.4 | llama.cpp `cache_prompt` note; batch invariance (Thinking Machines Lab, vLLM); agent-cache characterisation (TraceLab and others); hybrid prefix-caching issues on three engines |
| 8 | Conclusion | 0.2 | |
| | Acknowledgements | — | Includes the AI-assistance disclosure that ACM's authorship policy requires |
| | Appendix | any | See below |

Appendix contents:

- Full family B and family C tables.
- The drift-per-turn figure.
- The Spark window exactness grid and the rule comparison.
- Changed-turn examples.
- The measurement safeguards in full (rendering checks, parser round trip,
  blinded runner, gates G0–G3).
- A short note on the tool-grammar finding and llama.cpp PR #26472. The main
  text gives it one sentence, because it is not about cache fidelity.
- Reproduction commands and the pre-registration tag.

## 6. Figures and tables

| Item | Content | Source |
|---|---|---|
| Figure 1 | Action changes by model, history length and boundary, with Clopper–Pearson intervals | `study/analysis.json` |
| Figure 2 | Action changes by margin bin, per model | `study/analysis.json` |
| Figure 3 | G1 exactness grid, hybrid and dense | `g1/*.json` |
| Figure A1 | Per-turn drift by model and stratum | `study/analysis.json` |
| Table 1 | Models, architecture type, prompt lengths, server configuration | findings and pre-registration |
| Table 2 | H1–H7 with verdict and basis | main-study findings |

Paper figures are vector PDFs with no baked-in title or subtitle; the caption
carries that text. They come from a paper-style option added to the existing
plotting code in `src/driftmatrix/figures.py`, so the data path is unchanged. The
option's default keeps the README PNGs byte-for-byte as they are now, and the
existing figure tests must still pass. Fonts and sizes are set for a
single-column width (Figure 3 may span two columns).

## 7. Number integrity

- `scripts/paper_numbers.py` reads `docs/findings/study/analysis.json` and the
  committed gate and follow-up reports, and writes `docs/paper/numbers.tex`, a
  file of LaTeX macros. The prose and tables use the macros.
- The script has tests: it fails loudly on a missing key, and a test pins the
  headline values the handoff lists (for example 1,080 of 1,080 inexact turns,
  11 of 90, 21 of 73, 1 of 864, 66 of 66).
- A number that exists only in a markdown findings document is entered by hand
  with a LaTeX comment naming the file and section it came from.
- Final check before the draft is called done: every number in the built PDF is
  traced to its source, and the result is recorded in
  `docs/paper/VERIFICATION.md`.
- Every reference is checked on the web: it exists, the identifier is right, and
  it says what the paper cites it for. GitHub issues and documentation pages are
  cited with URL and access date.

## 8. Files and build

```
docs/paper/
  main.tex            preamble, title, author, abstract, \input of sections
  sections/           one .tex file per section, plus appendix.tex
  refs.bib
  numbers.tex         generated; committed so the paper builds without Python
  figures/            generated PDFs; committed
  build.ps1           builds both layouts with tools/tectonic
  VERIFICATION.md     the number and reference check
  HANDOFF.md          existing brief
```

- One source builds two PDFs: the ACM workshop layout, and the arXiv layout,
  which uses the `acmart` class's `nonacm` option to drop the copyright block
  and conference footer.
- Author block: Manuel Gamal, Ain Shams University, with the email from the git
  configuration.
- Code link: `https://github.com/manuelgamal/driftmatrix`, with the
  pre-registration tag `prereg-2026-09-10` named in the text.
- Build output goes to `output/pdf/`, which stays untracked.

## 9. Risks

| Risk | Handling |
|---|---|
| arXiv endorsement. Since 2026-01-21 a first-time submitter needs a personal endorsement from an established arXiv author in the domain | Outside this spec's work, but on the critical path for 16 October. Manuel starts it in parallel |
| The text runs over six pages | Appendix absorbs the safeguards detail, family B and C tables and examples; the page budget in section 5 is checked on every build |
| tectonic cannot build `acmart` | Checked first, before any writing. Fallback: a plain two-column article class for arXiv, with the ACM layout built later |
| EuroMLSys 2027 changes its format | Handled as a revision when the call appears |
| A cited source cannot be verified | The citation is dropped or reworded to what can be verified |

## 10. Done means

- Both PDFs build cleanly from a fresh checkout with `build.ps1`.
- The ACM layout is at most six pages before the references.
- `VERIFICATION.md` lists every number and reference as checked.
- The test suite passes, including the new tests for the numbers script and the
  figure option.
- Manuel has read the draft. Posting and submission remain his decisions.
