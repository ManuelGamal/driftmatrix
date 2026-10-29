# driftmatrix Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A six-page paper on the driftmatrix results, in LaTeX under `docs/paper/`, that builds to an ACM-layout PDF and an arXiv-layout PDF with one command, with every number and reference traced to its source.

**Architecture:** Three small pieces of tooling come first: a build script that also enforces the page limit, a paper mode for the existing plotting code, and a script that turns the committed result files into LaTeX macros. The prose is then written section by section against those macros and figures. A final pass records the check of every number and reference.

**Tech Stack:** LaTeX (`acmart` class, `sigplan` format) built with `tools/tectonic/tectonic.exe`; Python 3.10, matplotlib, pytest (all already in the project).

**Spec:** `docs/superpowers/specs/2026-10-02-driftmatrix-paper-design.md`. Read it and `docs/paper/HANDOFF.md` before starting.

## Global Constraints

- Work on the `paper` branch. Commits are authored as Manuel only: **no `Co-Authored-By` lines and no generated-by footers.**
- Nothing is pushed, posted, submitted or sent. No one is contacted.
- No new experiments and no new analyses of raw records. `data/` is not read.
- No findings document, README text or README figure (`docs/figures/*.png`) changes.
- Six pages maximum before the references in the ACM layout. The appendix comes after the references and is unlimited.
- Every number in the paper comes from a macro in `numbers.tex`, or is hand-entered with a LaTeX comment naming its source file and section, in the form `% src: docs/findings/<file>.md, "<section heading>"`.
- Every result is labelled confirmatory, pre-registered secondary, or exploratory (spec section 4).
- Scope is stated in the abstract and in Limitations: llama.cpp build `b10839`, Q4_K_M, one RTX 4060 Laptop GPU, `n_ubatch = 512`, greedy decoding, three models, synthetic traffic.
- Failed pre-registered predictions are reported, in Table 2 and in Limitations.
- The paper does not claim that prompt caching *can* change output (llama.cpp documents that), and makes no claim about other engines.
- Prose style: plain and specific. No "novel", "significant" (except in the statistical sense), "comprehensive", "crucially". State what was measured and what it showed.
- Author block: Manuel Gamal, Ain Shams University, Cairo, Egypt, `manuelaziz27@gmail.com`.
- Run Python from the repository root. Run the test suite with `python -m pytest -q`; it passes today with 329 passed, 1 skipped.

## Review Focus

Failure modes the spec implies that are most likely to reach a reader, each pinned by a check in the task named:

1. **A number in the text disagrees with the data.** The macros come from the committed result files, a test pins the headline values, and a test fails when the committed `numbers.tex` is stale (Task 3). Hand-entered numbers are traced one by one (Task 9).
2. **The paper silently runs over six pages.** The build script reads the page of the `endofbody` label and fails above the limit (Task 1).
3. **An undefined citation or cross-reference prints as "??" or "[?]".** The build script scans the log and fails on any undefined or multiply defined label or citation (Task 1).
4. **A result file is missing a key or a member, and the macro file is written anyway with a gap.** The generator iterates over the expected keys and raises (Task 3).
5. **The paper-figure option changes the README figures.** Default behaviour is tested, and the README PNGs are regenerated and compared with `git status` (Task 2).

## File Structure

| File | Responsibility |
|---|---|
| `scripts/build_paper.py` | Build both layouts with tectonic; fail on page overflow or undefined references |
| `tests/test_build_paper.py` | Tests for the page and log checks |
| `docs/paper/build.ps1` | One-line wrapper around the build script |
| `docs/paper/main.tex` | ACM layout: document class line, then `\input{body}` |
| `docs/paper/arxiv.tex` | arXiv layout: document class with `nonacm`, then `\input{body}` |
| `docs/paper/body.tex` | Preamble, title, author, section inputs, bibliography, appendix |
| `docs/paper/sections/*.tex` | One file per section |
| `docs/paper/refs.bib` | Verified references |
| `src/driftmatrix/figures.py` | Gains a `paper` option on each plot function |
| `scripts/make_figures.py` | Gains `--paper` |
| `docs/paper/figures/*.pdf` | Generated, committed |
| `src/driftmatrix/papernumbers.py` | Result files → macro dictionary → LaTeX text |
| `scripts/paper_numbers.py` | Writes `docs/paper/numbers.tex` |
| `tests/test_papernumbers.py` | Tests for the above, including pinned headline values |
| `docs/paper/numbers.tex` | Generated, committed |
| `docs/paper/VERIFICATION.md` | Record of the reference check and the number check |

---

### Task 1: Build pipeline and skeleton

Proves that tectonic builds `acmart` before any prose exists (spec section 9), and puts the page-limit and undefined-reference checks in place.

**Files:**
- Create: `scripts/build_paper.py`, `tests/test_build_paper.py`, `docs/paper/build.ps1`, `docs/paper/main.tex`, `docs/paper/arxiv.tex`, `docs/paper/body.tex`, `docs/paper/sections/introduction.tex`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `body_pages(aux_text: str) -> int`, `log_problems(log_text: str) -> list[str]`, and the command `python scripts/build_paper.py [--max-pages N]`, which exits non-zero on any failure. Later tasks add files under `docs/paper/sections/` and `\input` lines in `body.tex`.

- [ ] **Step 1: Write the failing tests**

`tests/test_build_paper.py`:

```python
"""Page-limit and undefined-reference checks of the paper build script."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_paper.py"
spec = importlib.util.spec_from_file_location("build_paper", SCRIPT)
build_paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_paper)


def test_body_pages_reads_the_page_of_the_endofbody_label():
    aux = "\\newlabel{sec:intro}{{1}{1}{Introduction}{section.1}{}}\n" \
          "\\newlabel{endofbody}{{8}{6}{Conclusion}{section.8}{}}\n"
    assert build_paper.body_pages(aux) == 6


def test_body_pages_fails_when_the_label_is_missing():
    with pytest.raises(ValueError, match="endofbody"):
        build_paper.body_pages("\\newlabel{sec:intro}{{1}{1}{Introduction}{section.1}{}}\n")


def test_log_problems_finds_undefined_and_duplicate_references():
    log = ("LaTeX Warning: Citation `tracelab' on page 1 undefined on input line 12.\n"
           "LaTeX Warning: Reference `fig:grid' on page 4 undefined on input line 40.\n"
           "LaTeX Warning: Label `tab:h' multiply defined.\n"
           "Overfull \\hbox (2.0pt too wide) in paragraph at lines 3--4\n")
    assert len(build_paper.log_problems(log)) == 3


def test_log_problems_is_empty_for_a_clean_log():
    assert build_paper.log_problems("Output written on main.pdf (6 pages).\n") == []
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_build_paper.py -q`
Expected: FAIL at collection, because `scripts/build_paper.py` does not exist.

- [ ] **Step 3: Write the build script**

`scripts/build_paper.py`:

```python
"""Build the paper in both layouts and check it.

    python scripts/build_paper.py            # ACM layout (main) and arXiv layout (arxiv)
    python scripts/build_paper.py --max-pages 6

Fails when the body of the ACM layout runs past the page limit, or when either
layout has an undefined or multiply defined citation or reference.
PDFs, logs and aux files go to output/pdf/.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "docs" / "paper"
OUT = REPO / "output" / "pdf"
BUNDLED = REPO / "tools" / "tectonic" / "tectonic.exe"
LAYOUTS = ("main", "arxiv")

_LABEL = re.compile(r"\\newlabel\{endofbody\}\{\{[^{}]*\}\{(\d+)\}")
_PROBLEM = re.compile(r"(Citation|Reference) `[^']*' .*undefined|multiply defined")


def body_pages(aux_text: str) -> int:
    """Page on which the body ends, from the endofbody label in a .aux file."""
    found = _LABEL.search(aux_text)
    if not found:
        raise ValueError("label endofbody not found in the .aux file")
    return int(found.group(1))


def log_problems(log_text: str) -> list[str]:
    """Log lines that report an undefined or multiply defined citation or reference."""
    return [line for line in log_text.splitlines() if _PROBLEM.search(line)]


def tectonic() -> str:
    if BUNDLED.exists():
        return str(BUNDLED)
    found = shutil.which("tectonic")
    if not found:
        raise SystemExit("tectonic not found: expected tools/tectonic/tectonic.exe or tectonic on PATH")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in LAYOUTS:
        subprocess.run([tectonic(), "--keep-intermediates", "--keep-logs", "--outdir", str(OUT),
                        str(PAPER / f"{name}.tex")], check=True)
        problems = log_problems((OUT / f"{name}.log").read_text(encoding="utf-8", errors="replace"))
        for line in problems:
            print(f"{name}: {line}")
        pages = body_pages((OUT / f"{name}.aux").read_text(encoding="utf-8", errors="replace"))
        print(f"{name}: body ends on page {pages}")
        if problems or (name == "main" and pages > args.max_pages):
            failed = True
    if failed:
        print(f"FAILED: undefined references, or the ACM body is over {args.max_pages} pages")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_build_paper.py -q`
Expected: 4 passed.

- [ ] **Step 5: Write the LaTeX skeleton**

`docs/paper/main.tex`:

```latex
\documentclass[sigplan,10pt]{acmart}
\input{body}
```

`docs/paper/arxiv.tex`:

```latex
\documentclass[sigplan,10pt,nonacm]{acmart}
\input{body}
```

`docs/paper/body.tex`:

```latex
% Shared by main.tex (ACM layout) and arxiv.tex (arXiv layout).
\settopmatter{printacmref=false}
\setcopyright{none}
\renewcommand\footnotetextcopyrightpermission[1]{}
\pagestyle{plain}

\begin{document}

\title{Does Restoring a Prompt Cache Change What an Agent Does? A Pre-Registered Study of Cache Fidelity in llama.cpp}

\author{Manuel Gamal}
\affiliation{%
  \institution{Ain Shams University}
  \city{Cairo}
  \country{Egypt}}
\email{manuelaziz27@gmail.com}

\maketitle

\input{sections/introduction}

\phantomsection\label{endofbody}

\end{document}
```

`docs/paper/sections/introduction.tex`:

```latex
\section{Introduction}
\label{sec:intro}

Skeleton build.
```

`docs/paper/build.ps1`:

```powershell
python "$PSScriptRoot\..\..\scripts\build_paper.py" @args
exit $LASTEXITCODE
```

Append to `.gitignore`:

```
# Paper build output and the LaTeX engine's cache
output/
.tectonic-cache/
```

- [ ] **Step 6: Build**

Run: `python scripts/build_paper.py`
Expected: tectonic downloads the `acmart` bundle on first use, then prints `main: body ends on page 1` and `arxiv: body ends on page 1`, exit code 0. `output/pdf/main.pdf` and `output/pdf/arxiv.pdf` exist.

Open both PDFs with the Read tool. Expected: two-column layout; `main.pdf` has no copyright block; `arxiv.pdf` has no conference footer.

If tectonic cannot build `acmart`: stop and report to Manuel. The spec's fallback (a plain two-column `article` class for arXiv) is a design change that needs his agreement.

- [ ] **Step 7: Confirm the page check can fail**

Run: `python scripts/build_paper.py --max-pages 0`
Expected: `FAILED: ...` and exit code 1.

- [ ] **Step 8: Commit**

```bash
git add scripts/build_paper.py tests/test_build_paper.py docs/paper/build.ps1 docs/paper/main.tex docs/paper/arxiv.tex docs/paper/body.tex docs/paper/sections/introduction.tex .gitignore docs/paper/HANDOFF.md
git commit -m "Paper build pipeline: two layouts, page limit and undefined-reference checks"
```

---

### Task 2: Paper mode for the figures

**Files:**
- Modify: `src/driftmatrix/figures.py`, `scripts/make_figures.py`, `tests/test_figures.py`
- Create: `docs/paper/figures/g1_exactness_grid.pdf`, `action_changes.pdf`, `margin_bins.pdf`, `drift_per_turn.pdf`

**Interfaces:**
- Produces: each of `plot_exactness_grid`, `plot_action_changes`, `plot_margin_bins`, `plot_drift_per_turn` accepts a keyword argument `paper: bool = False`. With `paper=True` the figure has no title or subtitle, uses the size in `PAPER_SIZES`, and is saved with a white background and tight padding; the file format follows the path's suffix. `python scripts/make_figures.py --paper` writes the four PDFs to `docs/paper/figures/`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_figures.py`:

```python
import pytest

import driftmatrix.figures as figures_module


def _plotters():
    grid = {"ornith": {256: {256: (True, 0.0), 384: (False, 0.2)}},
            "qwen3": {256: {256: (True, 0.0), 384: (False, 0.01)}}}
    a = analysis()
    return {"g1_exactness_grid": lambda path, **kw: plot_exactness_grid(grid, path, **kw),
            "action_changes": lambda path, **kw: plot_action_changes(a, path, **kw),
            "margin_bins": lambda path, **kw: plot_margin_bins(a, path, **kw),
            "drift_per_turn": lambda path, **kw: plot_drift_per_turn(a, path, **kw)}


@pytest.mark.parametrize("name", ["g1_exactness_grid", "action_changes", "margin_bins", "drift_per_turn"])
def test_paper_mode_writes_a_pdf_with_no_title(tmp_path, monkeypatch, name):
    titled = []
    monkeypatch.setattr(figures_module, "_titles", lambda *args, **kwargs: titled.append(args))
    path = tmp_path / f"{name}.pdf"
    _plotters()[name](path, paper=True)
    assert path.read_bytes().startswith(b"%PDF")
    assert titled == []


@pytest.mark.parametrize("name", ["g1_exactness_grid", "action_changes", "margin_bins", "drift_per_turn"])
def test_default_mode_still_draws_the_title_and_writes_a_png(tmp_path, monkeypatch, name):
    titled = []
    monkeypatch.setattr(figures_module, "_titles", lambda *args, **kwargs: titled.append(args))
    path = tmp_path / f"{name}.png"
    _plotters()[name](path)
    assert path.read_bytes().startswith(b"\x89PNG")
    assert len(titled) == 1


def test_every_figure_has_a_paper_size():
    assert set(figures_module.PAPER_SIZES) == {"g1_exactness_grid", "action_changes", "margin_bins", "drift_per_turn"}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_figures.py -q`
Expected: the new tests FAIL with `TypeError: ... unexpected keyword argument 'paper'` and `AttributeError: ... PAPER_SIZES`.

- [ ] **Step 3: Implement paper mode**

In `src/driftmatrix/figures.py`, add after `JSD_FLOOR`:

```python
# Paper figures: no title (the caption carries it), sized in inches for inclusion at
# column width (g1_exactness_grid spans both columns).
PAPER_SIZES = {"g1_exactness_grid": (7.0, 4.4), "action_changes": (4.4, 3.2),
               "margin_bins": (4.4, 2.9), "drift_per_turn": (4.4, 3.0)}
```

Replace `_save` and add `_finish` below `_titles`:

```python
def _save(fig, path, paper=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if paper:
        fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
```

```python
def _finish(fig, path, title, subtitle, top, paper):
    """Title block and layout for the README figure; neither for the paper figure."""
    if paper:
        fig.tight_layout()
    else:
        _titles(fig, title, subtitle)
        fig.tight_layout(rect=(0, 0, 1, top))
    _save(fig, path, paper)
```

`plot_exactness_grid`: change the signature to `def plot_exactness_grid(grid, path, paper=False):`, change the `figsize` argument to

```python
figsize=PAPER_SIZES["g1_exactness_grid"] if paper else (3.3 * len(models) + 0.6, 5.6),
```

and replace its last three statements (`_titles(...)`, `fig.tight_layout(...)`, `_save(...)`) with

```python
    _finish(fig, path, "Restore is bit-exact exactly when the reuse boundary is a multiple of n_ubatch",
            "Gate G1 · one ~1.9K-token prompt per model · inexact cells show the worst-position "
            "total variation against recompute", 0.93, paper)
```

`plot_action_changes`: signature `def plot_action_changes(analysis, path, paper=False):`, figure line

```python
    fig, ax = plt.subplots(figsize=PAPER_SIZES["action_changes"] if paper else (8.4, 5.0), facecolor=SURFACE)
```

and replace its last three statements with

```python
    _finish(fig, path, "Restore changed the agent's action on ambiguous requests, on every architecture",
            f"Main study · 90 turns per point, both replicates · 95% Clopper–Pearson intervals · "
            f"explicit requests: {changes} changes in {turns:,} restores", 0.94, paper)
```

`plot_margin_bins`: signature `def plot_margin_bins(analysis, path, paper=False):`, figure line

```python
    fig, ax = plt.subplots(figsize=PAPER_SIZES["margin_bins"] if paper else (8.4, 4.6), facecolor=SURFACE)
```

and replace its last three statements with

```python
    _finish(fig, path, "Changes happen where the model was already close to a tie",
            "Natural boundaries, all turns · action margin = smallest top-two logprob gap at the decision, "
            "on the recompute arm · 95% Clopper–Pearson intervals", 0.93, paper)
```

`plot_drift_per_turn`: signature `def plot_drift_per_turn(analysis, path, seed=0, paper=False):`, figure line

```python
    fig, ax = plt.subplots(figsize=PAPER_SIZES["drift_per_turn"] if paper else (8.4, 4.8), facecolor=SURFACE)
```

and replace its last three statements with

```python
    _finish(fig, path, "Every natural-boundary restore drifted; the hybrid's median drift is the highest",
            "Main study · one point per turn, both prefix bands · snapped restores were exactly 0 on Ornith and "
            "Qwen3, and on Spark at k = 512", 0.93, paper)
```

Update the module docstring's first line to `"""README, findings and paper figures, drawn only from committed results under docs/findings.`

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_figures.py -q`
Expected: all pass, including the 9 new tests.

- [ ] **Step 5: Add `--paper` to the figure script**

Replace `main` in `scripts/make_figures.py` and add `import argparse` at the top:

```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true",
                        help="write title-free PDFs to docs/paper/figures instead of the README PNGs")
    args = parser.parse_args()
    out = REPO / "docs" / "paper" / "figures" if args.paper else REPO / "docs" / "figures"
    ext = "pdf" if args.paper else "png"
    grid = load_g1(sorted((REPO / "docs" / "findings" / "g1").glob("g1_*.json")))
    analysis = json.loads((REPO / "docs" / "findings" / "study" / "analysis.json").read_text(encoding="utf-8"))
    plot_exactness_grid(grid, out / f"g1_exactness_grid.{ext}", paper=args.paper)
    plot_action_changes(analysis, out / f"action_changes.{ext}", paper=args.paper)
    plot_margin_bins(analysis, out / f"margin_bins.{ext}", paper=args.paper)
    plot_drift_per_turn(analysis, out / f"drift_per_turn.{ext}", paper=args.paper)
    for p in sorted(out.glob(f"*.{ext}")):
        print(f"{p.relative_to(REPO)}  {p.stat().st_size // 1024} KB")
    return 0
```

Also add the `--paper` line to the script's docstring usage.

- [ ] **Step 6: Confirm the README figures are unchanged**

Run: `python scripts/make_figures.py` then `git status --short docs/figures`
Expected: no output from `git status`. If a PNG differs, run `git diff --stat docs/figures` and open the regenerated image: a difference with identical appearance means the installed matplotlib differs from the one that drew the originals; restore the committed files with `git checkout -- docs/figures` and note it in the commit message. A visible difference is a bug in Step 3; fix it.

- [ ] **Step 7: Generate and inspect the paper figures**

Run: `python scripts/make_figures.py --paper`
Expected: four PDFs listed under `docs/paper/figures/`.

Open each PDF with the Read tool and check: no title or subtitle; no label clipped or overlapping; the count annotations (for example `11/90`) are inside the axes. The three single-column figures will be included at 3.33 in wide, so text is scaled by about 0.76: a 9 pt label prints at about 6.8 pt, which is the smallest acceptable size. If labels collide, adjust that figure's entry in `PAPER_SIZES` (wider or taller, keeping the width at or below 4.8 in for single-column figures), rerun, and look again.

- [ ] **Step 8: Run the whole suite and commit**

Run: `python -m pytest -q`
Expected: 342 passed, 1 skipped (329 + 4 from Task 1 + 9).

```bash
git add src/driftmatrix/figures.py scripts/make_figures.py tests/test_figures.py docs/paper/figures
git commit -m "Paper mode for the figures: title-free PDFs at column size"
```

---

### Task 3: Numbers as LaTeX macros

**Files:**
- Create: `src/driftmatrix/papernumbers.py`, `scripts/paper_numbers.py`, `tests/test_papernumbers.py`, `docs/paper/numbers.tex`
- Modify: `docs/paper/body.tex`

**Interfaces:**
- Produces:
  - `load_inputs(repo: Path) -> tuple[dict, dict[str, list[dict]], dict[int, dict]]` returns the analysis, the G1 reports by model, and the Spark follow-up reports by `n_ubatch`.
  - `macros(analysis: dict, g1: dict[str, list[dict]], spark: dict[int, dict]) -> dict[str, str]` maps macro name to LaTeX value.
  - `render(macros: dict[str, str]) -> str` returns the text of `numbers.tex`.
  - In LaTeX, `\Name` for every key. Percentages carry no `%` sign; write `\AOrnithNatLongAmbPct\%`. Scientific values (`sci`) need math mode: `$\COneEst$`.

Macro names (letters only, because TeX macro names cannot contain digits):

| Pattern | Meaning | Example → value |
|---|---|---|
| `TurnsPerModel`, `TurnsTotal` | Turns | `360`, `1{,}080` |
| `FloorInexact` | Turns where the two recomputes differed | `0` |
| `NatTurns`, `NatInexact` | Natural-boundary turns, and how many were not bit-exact | `1{,}080`, `1{,}080` |
| `A<Model><Nat\|Snap><Short\|Long><Exp\|Amb>` plus suffix `N`, `Pct`, `Lo`, `Hi` | Family A events, turns, rate and interval (percent) | `AOrnithNatLongAmb` → `11`, `...Pct` → `12.2`, `...Lo` → `6.3`, `...Hi` → `20.8` |
| `MembersA`, `ClaimedA`, `ChangesTotal` | Family A size, claimed members, all action changes | `24`, `7`, `30` |
| `ExplicitChanges`, `ExplicitRestores`, `ExplicitTurns` | Explicit stratum totals | `0`, `1{,}080`, `540` |
| `Margin<Model\|All><NearTie\|Contested\|Confident>` plus suffix `N`, `Pct` | Margin bins at natural boundaries | `MarginAllNearTie` → `21`, `...N` → `73`, `...Pct` → `29` |
| `SnapCost<Model><Short\|Long>` | Median re-evaluated tokens | `SnapCostQwenShort` → `443` |
| `SnapCostMin`, `SnapCostMax` | Smallest and largest of those medians | `158`, `443` |
| `C<One…Six>` plus suffix `Pos`, `Neg`, `Est`, `Lo`, `Hi` | Family C sign counts, estimate and interval (`sci`) | `CThreePos` → `291` |
| `Drift<Model><Short\|Long><Exp\|Amb>` | Median per-turn mean JSD, natural boundary (`sci`) | `DriftOrnithLongAmb` → `1.5\times10^{-4}` |
| `SnapInexact<Model>`, `SnapValid<Model>` | H1: inexact and valid snapped turns | `SnapInexactSpark` → `180` |
| `GOneRestores`, `GOneContradictions`, `GOneRatio` | G1 restores, cases contradicting the alignment rule, hybrid/dense median ratio | `66`, `0`, `26` |
| `SparkCases`, `SparkContradictions` | Follow-up cases, cases contradicting the window rule | `39`, `0` |

`<Model>` is `Ornith`, `Spark` or `Qwen`.

- [ ] **Step 1: Write the failing tests**

`tests/test_papernumbers.py`:

```python
"""The paper's macros: formatting, loud failure, and the headline values pinned to the committed results."""
import copy
from pathlib import Path

import pytest

from driftmatrix.papernumbers import count, load_inputs, macros, pct, render, sci

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs():
    return load_inputs(REPO)


@pytest.fixture(scope="module")
def m(inputs):
    return macros(*inputs)


def test_formatting():
    assert count(1080) == "1{,}080" and count(90) == "90"
    assert pct(0.12222) == "12.2" and pct(0.2877, 0) == "29"
    assert sci(0.0033044) == "3.3\\times10^{-3}" and sci(-0.0019) == "-1.9\\times10^{-3}" and sci(0.0) == "0"


def test_headline_values_match_the_handoff(m):
    assert (m["NatInexact"], m["NatTurns"]) == ("1{,}080", "1{,}080")
    assert m["FloorInexact"] == "0"
    assert (m["ExplicitChanges"], m["ExplicitTurns"], m["ExplicitRestores"]) == ("0", "540", "1{,}080")
    assert (m["AOrnithNatLongAmb"], m["AOrnithNatLongAmbN"], m["AOrnithNatLongAmbPct"]) == ("11", "90", "12.2")
    assert (m["ASparkNatShortAmb"], m["ASparkNatShortAmbPct"]) == ("2", "2.2")
    assert m["ASparkSnapLongAmb"] == "3"
    assert (m["MarginAllNearTie"], m["MarginAllNearTieN"], m["MarginAllNearTiePct"]) == ("21", "73", "29")
    assert (m["MarginAllConfident"], m["MarginAllConfidentN"], m["MarginAllConfidentPct"]) == ("1", "864", "0.1")
    assert (m["SnapCostMin"], m["SnapCostMax"]) == ("158", "443")
    assert (m["GOneRestores"], m["GOneContradictions"], m["GOneRatio"]) == ("66", "0", "26")
    assert (m["SparkCases"], m["SparkContradictions"]) == ("39", "0")
    assert (m["SnapInexactOrnith"], m["SnapInexactQwen"], m["SnapInexactSpark"]) == ("0", "0", "180")
    assert (m["MembersA"], m["ClaimedA"], m["ChangesTotal"]) == ("24", "7", "30")
    assert (m["CThreePos"], m["CThreeNeg"]) == ("291", "69")


def test_a_missing_family_member_fails_loudly(inputs):
    analysis, g1, spark = inputs
    broken = copy.deepcopy(analysis)
    del broken["families"]["A"]["A:spark:snapped:long:ambiguous"]
    with pytest.raises(KeyError):
        macros(broken, g1, spark)


def test_a_missing_g1_model_fails_loudly(inputs):
    analysis, g1, spark = inputs
    with pytest.raises(KeyError):
        macros(analysis, {"ornith": g1["ornith"]}, spark)


def test_render_writes_one_newcommand_per_macro(m):
    text = render(m)
    assert text.startswith("% Generated by scripts/paper_numbers.py")
    assert "\\newcommand{\\NatInexact}{1{,}080}\n" in text
    assert text.count("\\newcommand") == len(m)


def test_render_rejects_names_tex_cannot_define():
    with pytest.raises(ValueError, match="C1"):
        render({"C1": "3"})


def test_render_rejects_values_that_would_break_the_build():
    with pytest.raises(ValueError, match="Rate"):
        render({"Rate": "12%"})


def test_the_committed_numbers_file_is_current(m):
    committed = (REPO / "docs" / "paper" / "numbers.tex").read_text(encoding="utf-8")
    assert committed == render(m), "run: python scripts/paper_numbers.py"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_papernumbers.py -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'driftmatrix.papernumbers'`.

- [ ] **Step 3: Write the module**

`src/driftmatrix/papernumbers.py`:

```python
"""The paper's numbers as LaTeX macros, computed from committed results under docs/findings.

  load_inputs   read analysis.json, the G1 reports and the Spark follow-up reports
  macros        {macro name: LaTeX value}; raises KeyError when an expected result is missing
  render        the text of docs/paper/numbers.tex

Percentages carry no % sign. sci() values need math mode.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

MODELS = {"ornith": "Ornith", "spark": "Spark", "qwen3": "Qwen"}
BOUNDARIES = {"natural": "Nat", "snapped": "Snap"}
BANDS = {"short": "Short", "long": "Long"}
STRATA = {"explicit": "Exp", "ambiguous": "Amb"}
BINS = {"near_tie": "NearTie", "contested": "Contested", "confident": "Confident"}
CONTRASTS = {"C1_snapping_L3": "COne", "C2_snapping_x_architecture": "CTwo", "C3_L3_minus_L1": "CThree",
             "C4_L3_minus_L2": "CFour", "C5_compounding_L3": "CFive", "C6_stratum_L3": "CSix"}
SPARK_UBATCHES = (256, 512, 1024)
SPARK_WINDOW = 512


def count(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def sci(x: float) -> str:
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def half_up(x: float) -> int:
    return int(x + 0.5)


def load_inputs(repo: Path):
    findings = Path(repo) / "docs" / "findings"
    analysis = json.loads((findings / "study" / "analysis.json").read_text(encoding="utf-8"))
    g1: dict[str, list[dict]] = {}
    for path in sorted((findings / "g1").glob("g1_*.json")):
        g1.setdefault(path.name.split("_")[1], []).append(json.loads(path.read_text(encoding="utf-8")))
    spark = {ub: json.loads((findings / "followup" / f"spark_ub{ub}.json").read_text(encoding="utf-8"))
             for ub in SPARK_UBATCHES}
    return analysis, g1, spark


def _family_a(analysis, out):
    fam = analysis["families"]["A"]
    explicit = []
    for m, mn in MODELS.items():
        for b, bn in BOUNDARIES.items():
            for p, pn in BANDS.items():
                for s, sn in STRATA.items():
                    v = fam[f"A:{m}:{b}:{p}:{s}"]
                    name = f"A{mn}{bn}{pn}{sn}"
                    out[name] = str(v["events"])
                    out[name + "N"] = str(v["n"])
                    out[name + "Pct"] = pct(v["estimate"])
                    out[name + "Lo"] = pct(v["interval"]["low"])
                    out[name + "Hi"] = pct(v["interval"]["high"])
                    if s == "explicit":
                        explicit.append(v)
    out["MembersA"] = str(len(fam))
    out["ClaimedA"] = str(sum(bool(v["claimed"]) for v in fam.values()))
    out["ChangesTotal"] = str(sum(v["events"] for v in fam.values()))
    out["ExplicitChanges"] = str(sum(v["events"] for v in explicit))
    out["ExplicitRestores"] = count(sum(v["n"] for v in explicit))
    out["ExplicitTurns"] = count(sum(v["n"] for v in explicit) // len(BOUNDARIES))


def _margins(analysis, out):
    for label, ln in BINS.items():
        events = turns = 0
        for m, mn in MODELS.items():
            v = analysis["secondary"][m]["margin_bins"]["natural"][label]
            out[f"Margin{mn}{ln}"] = str(v["events"])
            out[f"Margin{mn}{ln}N"] = str(v["n"])
            out[f"Margin{mn}{ln}Pct"] = pct(v["events"] / v["n"], 0)
            events, turns = events + v["events"], turns + v["n"]
        rate = events / turns
        out[f"MarginAll{ln}"] = str(events)
        out[f"MarginAll{ln}N"] = str(turns)
        out[f"MarginAll{ln}Pct"] = pct(rate, 0 if rate >= 0.1 else 1)


def _snapping(analysis, out):
    medians = []
    for m, mn in MODELS.items():
        for p, pn in BANDS.items():
            median = half_up(analysis["secondary"][m]["snapping_cost_tokens"][p]["median"])
            out[f"SnapCost{mn}{pn}"] = str(median)
            medians.append(median)
        h1 = analysis["hypotheses"]["H1_mitigation"][m]
        out[f"SnapInexact{mn}"] = str(len(h1["inexact"]))
        out[f"SnapValid{mn}"] = str(h1["valid_snapped_turns"])
    out["SnapCostMin"], out["SnapCostMax"] = str(min(medians)), str(max(medians))


def _family_c(analysis, out):
    for key, name in CONTRASTS.items():
        v = analysis["families"]["C"][key]
        out[name + "Pos"], out[name + "Neg"] = str(v["n_pos"]), str(v["n_neg"])
        out[name + "Est"] = sci(v["estimate"])
        out[name + "Lo"], out[name + "Hi"] = sci(v["interval"]["low"]), sci(v["interval"]["high"])


def _drift(analysis, out):
    for m, mn in MODELS.items():
        for p, pn in BANDS.items():
            for s, sn in STRATA.items():
                v = analysis["secondary"][m]["conditions"][f"natural:{p}:{s}"]
                out[f"Drift{mn}{pn}{sn}"] = sci(v["median_mean_jsd"])


def _g1(g1, out):
    restores = contradictions = 0
    worst = {}
    for m in ("ornith", "qwen3"):
        for report in g1[m]:
            cases = [c for c in report["cases"] if "exact" in c]
            restores += len(cases)
            contradictions += sum(c["exact"] != c["aligned"] for c in cases)
            worst.setdefault(m, []).extend(c["max_tv"] for c in cases if not c["exact"])
    out["GOneRestores"], out["GOneContradictions"] = str(restores), str(contradictions)
    out["GOneRatio"] = str(half_up(statistics.median(worst["ornith"]) / statistics.median(worst["qwen3"])))


def _spark(spark, out):
    cases = contradictions = 0
    for ub in SPARK_UBATCHES:
        proven = [c for c in spark[ub]["cases"] if c.get("restore_used") and "exact" in c]
        cases += len(proven)
        contradictions += sum(c["exact"] != (c["k"] % ub == 0 and c["k"] <= SPARK_WINDOW) for c in proven)
    out["SparkCases"], out["SparkContradictions"] = str(cases), str(contradictions)


def macros(analysis: dict, g1: dict, spark: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    turns = [analysis["determinism"][m]["turns"] for m in MODELS]
    out["TurnsPerModel"] = count(turns[0])
    out["TurnsTotal"] = count(sum(turns))
    out["FloorInexact"] = count(sum(analysis["determinism"][m]["floor_inexact"] for m in MODELS))
    natural = [r for r in analysis["rows"] if r["boundary"] == "natural" and r["excluded"] is None]
    out["NatTurns"] = count(len(natural))
    out["NatInexact"] = count(sum(not r["exact"] for r in natural))
    _family_a(analysis, out)
    _margins(analysis, out)
    _snapping(analysis, out)
    _family_c(analysis, out)
    _drift(analysis, out)
    _g1(g1, out)
    _spark(spark, out)
    return out


def render(macros: dict[str, str]) -> str:
    lines = ["% Generated by scripts/paper_numbers.py from docs/findings. Do not edit.",
             "% Percentages carry no % sign. Values with \\times need math mode."]
    for name in sorted(macros):
        value = macros[name]
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name {name} is not letters only")
        if re.search(r"[%_#&$]", value):
            raise ValueError(f"macro {name} has a value TeX would misread: {value}")
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    return "\n".join(lines) + "\n"
```

`scripts/paper_numbers.py`:

```python
"""Write docs/paper/numbers.tex from the committed results.

    python scripts/paper_numbers.py

Needs no GPU, server or raw records.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.papernumbers import load_inputs, macros, render  # noqa: E402


def main() -> int:
    values = macros(*load_inputs(REPO))
    out = REPO / "docs" / "paper" / "numbers.tex"
    out.write_text(render(values), encoding="utf-8", newline="\n")
    print(f"{out.relative_to(REPO)}  {len(values)} macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Generate the file, then run the tests**

Run: `python scripts/paper_numbers.py`
Expected: `docs\paper\numbers.tex  <N> macros`.

Run: `python -m pytest tests/test_papernumbers.py -q`
Expected: 8 passed. If a pinned headline value fails, do not edit the expected value: the handoff and the findings say what it should be, so find out why the computation disagrees and report it.

If `test_the_committed_numbers_file_is_current` fails on line endings alone, add the line `docs/paper/numbers.tex text eol=lf` to a new `.gitattributes` file at the repository root and regenerate.

- [ ] **Step 5: Load the macros in the paper**

In `docs/paper/body.tex`, add as the first line after the opening comment:

```latex
\input{numbers}
```

and change the body of `docs/paper/sections/introduction.tex` from `Skeleton build.` to:

```latex
Skeleton build: \NatInexact{} of \NatTurns{} natural-boundary restores were inexact,
and \AOrnithNatLongAmb{} of \AOrnithNatLongAmbN{} (\AOrnithNatLongAmbPct\%) changed the action.
```

Run: `python scripts/build_paper.py`
Expected: exit code 0. Open `output/pdf/main.pdf` and confirm the sentence reads "1,080 of 1,080 ... 11 of 90 (12.2%)".

- [ ] **Step 6: Run the whole suite and commit**

Run: `python -m pytest -q`
Expected: 350 passed, 1 skipped.

```bash
git add src/driftmatrix/papernumbers.py scripts/paper_numbers.py tests/test_papernumbers.py docs/paper/numbers.tex docs/paper/body.tex docs/paper/sections/introduction.tex
git commit -m "Paper numbers as LaTeX macros generated from the committed results"
```

(Add `.gitattributes` to the commit if Step 4 created it.)

---

### Task 4: References, verified

**Files:**
- Create: `docs/paper/refs.bib`, `docs/paper/VERIFICATION.md`
- Modify: `docs/paper/body.tex`

**Interfaces:**
- Produces: the BibTeX keys below, usable with `\cite{key}` in every later task. A key is present in `refs.bib` only if its row in `VERIFICATION.md` says `verified`.

Candidate sources, with the key to use and the claim the paper needs from each:

| Key | Source | The paper cites it for |
|---|---|---|
| `llamacpp-server` | llama.cpp server README, `cache_prompt` option | The note that cached logits are not guaranteed bit-for-bit identical |
| `llamacpp` | llama.cpp repository, build `b10839` | The engine under test |
| `tml-nondeterminism` | Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (2025) | Batch invariance as the general phenomenon |
| `vllm-batch-invariant` | vLLM documentation for `VLLM_BATCH_INVARIANT` | A shipped batch-invariant mode |
| `tracelab` | arXiv 2606.30560 | 94–96% of prompt tokens served from cache in production agent traces |
| `policy-runtime` | arXiv 2605.27744 | Agent-workload cache characterisation measuring speed |
| `vllm-mooncake` | vLLM × Mooncake write-up | The same |
| `lmcache-hybrid` | LMCache hybrid-model documentation | The statement that generation is not bit-exact between cached and fresh runs |
| `llamacpp-20225` | llama.cpp issue #20225 | Hybrid sessions reprocessing the full prompt |
| `vllm-43587`, `vllm-40696`, `vllm-26201` | vLLM issues | Hybrid prefix-caching limitations |
| `sglang-21846` | SGLang issue #21846 | The same, third engine |
| `llamacpp-26472` | llama.cpp PR #26472 | The tool-grammar argument-order fix (appendix) |
| `clopper-pearson` | Clopper and Pearson, Biometrika 1934 | The interval |
| `holm` | Holm, Scandinavian Journal of Statistics 1979 | The correction |
| `ornith`, `spark`, `qwen3` | Each model's model card or paper | The three models and their architectures |
| `prereg` | This repository, tag `prereg-2026-09-10` | The frozen analysis plan |

- [ ] **Step 1: Verify each source on the web**

For each row, fetch the source (WebFetch for a URL or `https://arxiv.org/abs/<id>`; `gh issue view <n> --repo <owner>/<repo>` or `gh pr view` for GitHub items; WebSearch to find a model card or the two statistics papers). Confirm three things: it exists, the identifier and title are right, and it says what the third column needs. For `tracelab`, find the exact figure and the sentence it appears in; the paper may quote the number only as that source states it.

For the model cards, confirm the architecture description the paper relies on: Ornith as hybrid (linear attention with recurrent state plus full attention), Spark as sliding-window attention with a 512-token window plus full attention, Qwen3 as dense. The local source is `docs/superpowers/specs/2026-09-10-hybrid-cache-fidelity-design.md` section 3.1.

- [ ] **Step 2: Record the results**

`docs/paper/VERIFICATION.md` starts with:

```markdown
# Paper verification record

## References

Checked on 2026-10-02 unless a row says otherwise.

| Key | URL or identifier | Exists | Supports the claim | Status |
|---|---|---|---|---|
```

Add one row per candidate. Status is `verified`, `reworded` (with the wording the source does support), or `dropped` (with the reason). A source that cannot be reached or does not say what the paper needs is dropped or reworded. Never keep a citation on the strength of the repository's own notes.

- [ ] **Step 3: Write `refs.bib`**

One entry per verified or reworded key, with authors, title, year, and venue or URL exactly as the source page gives them. Web pages, documentation and GitHub items use this form:

```bibtex
@misc{llamacpp-20225,
  author       = {{llama.cpp contributors}},
  title        = {<issue title exactly as on GitHub>},
  howpublished = {GitHub issue \#20225, ggml-org/llama.cpp},
  year         = {<year opened>},
  url          = {https://github.com/ggml-org/llama.cpp/issues/20225},
  note         = {Accessed 2026-10-02}
}
```

arXiv papers use `@misc` with `eprint = {<id>}`, `archivePrefix = {arXiv}` and the authors as listed on the abstract page.

- [ ] **Step 4: Wire the bibliography into the paper**

In `docs/paper/body.tex`, replace

```latex
\phantomsection\label{endofbody}
```

with

```latex
\phantomsection\label{endofbody}

\bibliographystyle{ACM-Reference-Format}
\bibliography{refs}
```

and, so the skeleton has a citation to resolve, append to `sections/introduction.tex`:

```latex
The engine's documentation notes that cached logits can differ~\cite{llamacpp-server}.
```

(If `llamacpp-server` was dropped in Step 2, cite another verified key here.)

- [ ] **Step 5: Build and commit**

Run: `python scripts/build_paper.py`
Expected: exit code 0, a References section in both PDFs, no "undefined" lines printed.

```bash
git add docs/paper/refs.bib docs/paper/VERIFICATION.md docs/paper/body.tex docs/paper/sections/introduction.tex
git commit -m "Paper references, each checked against its source"
```

---

### Tasks 5–8: writing rules

These apply to every prose task below.

- Read the source documents named in the task before writing. The text restates them; it adds no claim they do not make.
- Use a macro for every number that has one (the table in Task 3). Hand-enter any other number with a `% src:` comment on the line above.
- Label results as the spec's section 4 requires. The Method section defines the three labels; later sections use the words "confirmatory", "pre-registered secondary" and "exploratory" exactly.
- Model names on first use: "Ornith-1.5-9B (hybrid recurrent)", "Spark-X2.5-4B (sliding window)", "Qwen3-4B (dense)"; afterwards "Ornith", "Spark", "Qwen3".
- `n_ubatch` and other flags are typeset with `\texttt{n\_ubatch}`.
- Single-column figures use `\includegraphics[width=\columnwidth]{figures/<name>.pdf}`.
- After each task: `python scripts/build_paper.py` exits 0, and the page count printed for `main` is within the running budget given in the task. Open `output/pdf/main.pdf` and read the new section as rendered.

---

### Task 5: Introduction, Background, Method

**Files:**
- Create: `docs/paper/sections/background.tex`, `docs/paper/sections/method.tex`
- Modify: `docs/paper/sections/introduction.tex` (replace the skeleton text), `docs/paper/body.tex`

**Sources to read:** `README.md`; `docs/preregistration/2026-09-10-analysis-plan.md`; `docs/findings/2026-09-10-study-manifest.md`; `docs/findings/2026-09-10-g2.md`; spec `docs/superpowers/specs/2026-09-10-hybrid-cache-fidelity-design.md` sections 1 and 3.

**Interfaces:**
- Produces: labels `sec:intro`, `sec:background`, `sec:method`, `sec:labels` (the evidence-label paragraph), `tab:setup`.

- [ ] **Step 1: Write the Introduction (about 0.75 page)**

Content, in order:
1. Agent frameworks resend the system prompt, tool schemas and a growing history on every turn, so serving stacks reuse cached state on nearly every request. Cite `tracelab` for the cache share, worded as verified in Task 4.
2. The reuse is assumed to be invisible. The engine's own documentation says cached logits are not guaranteed bit-identical (`llamacpp-server`). What was open: when reuse is exact, how the difference depends on architecture, and whether it changes what an agent does.
3. What this paper does: a pre-registered comparison of restore against recompute on agent-shaped traffic with real tool calls, on three architectures, one engine build.
4. Findings in four sentences, with macros: `\NatInexact` of `\NatTurns` natural-boundary restores inexact; action changes on 2–12% of ambiguous requests on all three models (use the smallest and largest of the six `A<Model>Nat<Band>AmbPct` macros: `\ASparkNatShortAmbPct` and `\AOrnithNatLongAmbPct`) and `\ExplicitChanges` of `\ExplicitRestores` explicit-request restores; concentration at near-ties (`\MarginAllNearTiePct\%` against `\MarginAllConfidentPct\%`); the alignment rule and the snapping mitigation, exact on two of three models at a median cost of `\SnapCostMin`–`\SnapCostMax` tokens.
5. A sentence stating the scope and that two pre-registered predictions failed (the controls were not immune; snapping fails on the sliding-window model at 4,096 tokens), with a forward reference to Section 6.
6. Contributions as a four-item list, worded as in the paper spec section 3.

- [ ] **Step 2: Write the Background (about 0.5 page)**

1. Prefix reuse: a server keeps the model state for a processed prompt prefix and evaluates only the new suffix. In llama.cpp this is `cache_prompt` within a slot and explicit slot save/restore across requests; the study uses save/restore so that each arm can be forced and proven.
2. Micro-batches: prompts are evaluated in physical micro-batches of `n_ubatch` tokens (512 here).
3. What the state is on each architecture: per-token keys and values for all positions (dense); the same within a window for the windowed layers (sliding window); a fixed-size recurrent state that summarises the prefix, alongside full-attention layers (hybrid). One or two sentences each, cited to the model cards.
4. The reuse boundary `k`: the longest common token prefix between the cached prompt and the new one.

- [ ] **Step 3: Write the Method (about 1.25 pages)**

Subsections, each a short paragraph:
1. **Pre-registration.** Hypotheses H1–H7, outcomes, three Holm-corrected families, decision rules and a null-result commitment were frozen at tag `prereg-2026-09-10` before any main-study turn ran; the analysis code was committed and tested before the run. State what had been seen at the freeze (pre-registration section 0): G0 and G1 on raw prompts, G2 on the recompute path only. Cite `prereg`.
2. **Evidence labels** (`\label{sec:labels}`). Define confirmatory, pre-registered secondary and exploratory exactly as in the paper spec section 4.
3. **Models and engine.** Table 1 (`tab:setup`): for each model its architecture, parameter count, and prompt-length and natural-boundary ranges per band, hand-entered from the "Measured" table in `2026-09-10-study-manifest.md` with a `% src:` comment. Below the table or in its caption: build `b10839`, Q4_K_M, RTX 4060 Laptop (8 GB), the server flags from pre-registration section 2, greedy decoding, at most 192 new tokens, top-10 logprobs.
4. **Traffic.** 90 base conversations with real tool round trips against three deterministic mock tools, rendered through each model's own chat template; short (8 round trips) and long (60) histories; explicit and ambiguous final requests, five ambiguous types; `\TurnsPerModel` turns per model, the same turns on every model.
5. **Boundaries and arms.** Natural boundary: the longest common token prefix (and why: Qwen3's template re-renders the previous assistant turn). Snapped boundary: `k − (k mod 512)`. Six arms per turn: recompute, restore-natural, restore-snapped, each twice.
6. **Arm proofs.** A restore counts only if the server's `timings.prompt_n` equals the suffix length and the saved slot holds exactly `k` tokens; a recompute only if it evaluated the whole prompt. All 6,480 arms passed (hand-entered, `% src: docs/findings/2026-09-11-main-study.md, "Validity"`).
7. **Outcomes.** Primary: action change reproduced in both replicates, where an action is the first complete tool call with canonical arguments, or no call. Secondary: bit-exactness; per-turn mean and max JSD at context-matched positions. Covariate: the action margin, with the three bins.
8. **Tests.** The recompute floor is measured; with a zero floor, one reproducible change rejects and the Clopper–Pearson interval carries the size (`clopper-pearson`); family C uses exact sign tests with paired bootstrap intervals; Holm within each family (`holm`).

- [ ] **Step 4: Add the sections to `body.tex`**

After `\input{sections/introduction}` add:

```latex
\input{sections/background}
\input{sections/method}
```

- [ ] **Step 5: Build, read, commit**

Run: `python scripts/build_paper.py`
Expected: exit code 0; `main: body ends on page 3` or lower (budget so far: 2.5 pages).

```bash
git add docs/paper/sections/introduction.tex docs/paper/sections/background.tex docs/paper/sections/method.tex docs/paper/body.tex
git commit -m "Paper: introduction, background and method"
```

---

### Task 6: Results

**Files:**
- Create: `docs/paper/sections/results.tex`
- Modify: `docs/paper/body.tex`

**Sources to read:** `docs/findings/2026-09-11-main-study.md` (all of it); `docs/findings/study/changed_turns.json`.

**Interfaces:**
- Consumes: `sec:labels`; figures `figures/action_changes.pdf`, `figures/margin_bins.pdf`.
- Produces: labels `sec:results`, `fig:actions`, `fig:margins`, `tab:hypotheses`.

- [ ] **Step 1: Write the Results section (about 1.5 pages)**

Paragraphs, in order:
1. **Validity.** All turns complete; all arm proofs passed; the recompute floor was bit-exact on every turn (`\FloorInexact` inexact of `\TurnsTotal`), so every difference is attributable to restore; every restore reproduced in its replicate; no turn excluded.
2. **Exactness (confirmatory, H1 and H2).** `\NatInexact` of `\NatTurns` natural-boundary restores were not bit-exact, including the `\ExplicitTurns` explicit turns whose generated tokens were identical. Snapped restores: exact on every turn on Ornith and Qwen3; on Spark exact in the short band and inexact on all `\SnapInexactSpark` long-band turns.
3. **Action changes (confirmatory, family A).** Figure 1 (`fig:actions`). Give the six natural-boundary ambiguous cells with counts and intervals from the `A...` macros; state `\ExplicitChanges` of `\ExplicitRestores`; `\ClaimedA` of `\MembersA` members claimed; no change seen in only one replicate. State plainly that the dense and sliding-window models changed too, and that no test compares rates between architectures.
4. **Where the changes happen (pre-registered secondary).** Figure 2 (`fig:margins`). Pooled rates per bin from the `MarginAll...` macros; the predicted ordering holds on every model; this analysis has intervals only and no corrected test.
5. **What changed.** Of `\ChangesTotal` changes: 11 same tool with different arguments, 10 a tool call that became text, 9 a clarifying question that became a tool call, none a different tool (hand-entered, `% src: docs/findings/2026-09-11-main-study.md, "Where the changes happen"`). One concrete example in a sentence: the `compute_sum` call that dropped one value from a 23-number list.
6. **Distribution drift (confirmatory, families B and C), briefly.** The hybrid's median drift on explicit requests (`$\DriftOrnithShortExp$`) against the dense model's (`$\DriftQwenShortExp$`); C1 and C4 claimed; C3 not claimed although the hybrid drifted more on `\CThreePos` of 360 turns, because the bootstrap interval of the mean difference includes zero; C5 not claimed. Full tables are in the appendix.
7. **Table 2** (`tab:hypotheses`): H1–H7, one line each, with verdict and basis, taken row by row from the "Hypotheses" table in the main-study findings. Mark each verdict as holds, supported, partial, fails or not supported, using the findings' own word.

Figure captions carry the information the README figure titles carried: what is plotted, 90 turns per point, 95% Clopper–Pearson intervals, and the label (confirmatory for Figure 1, pre-registered secondary for Figure 2).

- [ ] **Step 2: Add the section to `body.tex`**

After `\input{sections/method}` add:

```latex
\input{sections/results}
```

- [ ] **Step 3: Build, read, commit**

Run: `python scripts/build_paper.py`
Expected: exit code 0; `main: body ends on page 4` or lower (budget so far: 4 pages). Check in the PDF that both figures are legible at column width and that Table 2 fits its column.

```bash
git add docs/paper/sections/results.tex docs/paper/body.tex
git commit -m "Paper: results"
```

---

### Task 7: Mechanism and mitigation, Limitations, Related work, Conclusion, Abstract, Acknowledgements

**Files:**
- Create: `docs/paper/sections/mechanism.tex`, `limitations.tex`, `related.tex`, `conclusion.tex`, `abstract.tex`, `acknowledgements.tex` (all under `docs/paper/sections/`)
- Modify: `docs/paper/body.tex`

**Sources to read:** `docs/findings/2026-09-10-g0.md`, `2026-09-10-g1.md`, `2026-09-11-spark-window.md`, `2026-09-11-main-study.md` (sections "Snapping does not protect…", "Against the pre-registered predictions…", "Limits"); README "Prior art".

**Interfaces:**
- Consumes: `figures/g1_exactness_grid.pdf`; the verified keys from Task 4.
- Produces: labels `sec:mechanism`, `fig:grid`, `sec:limits`, `sec:related`, `sec:conclusion`.

- [ ] **Step 1: Write Mechanism and mitigation (about 1 page)**

1. **Tiling.** Restoring a saved state returns identical bytes, so the restore itself adds no difference. Recompute tiles the prompt from position 0 in micro-batches of `n_ubatch`; restore tiled `[0, k)` when the prefix was cached and tiles the suffix afresh from `k`. The tilings coincide only when `k` is a multiple of `n_ubatch`; otherwise floating-point sums accumulate in a different order.
2. **Test (exploratory, gate G1).** Figure 3 (`fig:grid`, a `figure*` spanning both columns). Changing `n_ubatch` moved the exact set with it: `\GOneRestores` restores, `\GOneContradictions` contradicting the rule, on the hybrid and the dense model, one prompt per model. The hybrid's inexact cells were about `\GOneRatio` times larger by median. Say that this gate ran before the pre-registration and is not confirmatory.
3. **Mitigation: snapping (confirmatory, H1 and C1).** Round the reuse boundary down to a multiple of `n_ubatch` and re-evaluate the remainder. On Ornith and Qwen3 every snapped restore was bit-exact and every action change disappeared. Cost (pre-registered secondary): median re-evaluated tokens per model and band from the `SnapCost...` macros, range `\SnapCostMin`–`\SnapCostMax`.
4. **Where it fails.** On Spark, snapping was exact at `k = 512` and inexact at `k = 4096` on all `\SnapInexactSpark` long-band turns, with `\ASparkSnapLongAmb` action changes remaining. Follow-up (exploratory, one prompt): with default flash attention a restore was exact only when the boundary was aligned and within the 512-token window, `\SparkContradictions` contradictions in `\SparkCases` cases; turning flash attention off extended exactness to 2,560 tokens but not beyond; without `--swa-full` the server discarded every restore, which the arm proofs caught (hand-entered values, `% src: docs/findings/2026-09-11-spark-window.md`). The cause inside llama.cpp is located, not explained.

- [ ] **Step 2: Write Limitations (about 0.4 page)**

1. Predictions that failed, stated first: the dense and sliding-window controls were expected to be unaffected and were not; snapping was expected to be exact on every model and fails on the sliding-window model beyond its window; drift did not compound with prefix length (H6); the hybrid's mean drift was not shown to exceed the dense model's (C3).
2. Scope: one build, one quantisation, one GPU, `n_ubatch = 512`; exactness is a property of kernels and can differ elsewhere. No claim about other engines.
3. Synthetic traffic: three mock tools and five ambiguous request types. Real traffic may hold more or fewer near-ties; the margin relationship is the part expected to transfer.
4. Few events: rates rest on 2–11 changes per cell, intervals are wide, and no test compares architectures on action-change rate.
5. Greedy decoding only.
6. The mechanism evidence and the sliding-window rule are exploratory and each rests on one prompt per configuration.

- [ ] **Step 3: Write Related work (about 0.4 page)**

Four short paragraphs, citing only keys verified in Task 4: (1) the engine's own note on `cache_prompt`, and what this paper adds to it; (2) nondeterminism and batch invariance; (3) agent-workload cache characterisation, which measures speed; (4) hybrid prefix-caching limitations reported on three engines, and the LMCache note. If a key was dropped, drop its sentence.

- [ ] **Step 4: Write the Conclusion (about 0.2 page)**

Three or four sentences: the result, the rule, the mitigation and its limit, and what a serving stack can do today (align reuse boundaries to the micro-batch size where the architecture allows it; treat near-tie decisions as the exposed ones). No future-work list beyond one sentence naming the unexplained sliding-window cause and other engines.

- [ ] **Step 5: Write the Abstract (at most 150 words)**

One paragraph: the question; the design (pre-registered, three architectures, `\TurnsPerModel` agent turns per model, restore against recompute); the four findings with their numbers via macros; the scope sentence (build, quantisation, one GPU, greedy decoding). Wrap it in `\begin{abstract} ... \end{abstract}`.

- [ ] **Step 6: Write the Acknowledgements**

`docs/paper/sections/acknowledgements.tex`:

```latex
\begin{acks}
The author used Claude (Anthropic), an AI assistant, to help write the measurement
code and to draft and edit this text. The author designed the study, ran every
experiment, checked every number against the recorded results, and is responsible
for the content.
\end{acks}
```

This wording is a draft for Manuel to correct; it must describe what actually happened. Flag it in the final report.

- [ ] **Step 7: Assemble `body.tex`**

The abstract must come before `\maketitle` in `acmart`. Replace the lines from `\maketitle` to the `endofbody` label with:

```latex
\input{sections/abstract}

\maketitle

\input{sections/introduction}
\input{sections/background}
\input{sections/method}
\input{sections/results}
\input{sections/mechanism}
\input{sections/limitations}
\input{sections/related}
\input{sections/conclusion}
\input{sections/acknowledgements}

\phantomsection\label{endofbody}
```

- [ ] **Step 8: Build and fit to six pages**

Run: `python scripts/build_paper.py`
Expected: exit code 0 and `main: body ends on page 6` or lower.

If the body is over six pages, cut in this order and rebuild after each cut: move the "What changed" example and the distribution-drift paragraph's detail to the appendix; shorten Background item 3; shorten the Method's traffic and outcomes paragraphs to what the Results need; reduce Figure 3's height in `PAPER_SIZES` and regenerate. Do not cut the Limitations section, Table 2, or any evidence label.

- [ ] **Step 9: Commit**

```bash
git add docs/paper/sections docs/paper/body.tex docs/paper/figures
git commit -m "Paper: mechanism and mitigation, limitations, related work, conclusion, abstract"
```

---

### Task 8: Appendix

**Files:**
- Create: `docs/paper/sections/appendix.tex`
- Modify: `docs/paper/body.tex`

**Sources to read:** `docs/findings/2026-09-11-main-study.md`; `2026-09-11-spark-window.md`; `2026-09-11-spark-argument-order.md`; `2026-09-10-g2.md`; `2026-09-10-g3.md`; README "How the measurement is kept honest" and "Reproducing".

**Interfaces:**
- Consumes: `figures/drift_per_turn.pdf`; macros `C...`, `Drift...`.
- Produces: labels `app:families`, `app:drift`, `app:spark`, `app:examples`, `app:safeguards`, `app:grammar`, `app:repro`. Add forward references to these from the main text where it says "in the appendix".

- [ ] **Step 1: Write the appendix**

Sections, each short:
1. **Families B and C in full** (`app:families`). Family B table: median per-turn mean JSD per band and stratum for each model at natural boundaries, from the `Drift...` macros; snapped values exactly zero on Ornith and Qwen3. Family C table: the six contrasts with pairs, sign counts, estimate, interval and whether claimed, from the `C...` macros. Note that C2 and C3 are numerically identical because snapped drift was exactly zero on both models.
2. **Per-turn drift** (`app:drift`). Figure A1.
3. **Sliding-window follow-up** (`app:spark`, exploratory). The 13 × 5 exactness grid and the four-rule comparison, copied cell by cell from `2026-09-11-spark-window.md` with a `% src:` comment.
4. **Changed-turn examples** (`app:examples`). The three examples from the main-study findings.
5. **Measurement safeguards** (`app:safeguards`). Offline rendering checked against the server; parser round trip (all 161 parsed calls reproduce the model's text, `% src: README.md, "How the measurement is kept honest"`); the blinded, resumable runner; gates G0–G3; the GPU-free test suite.
6. **Tool-call grammar** (`app:grammar`). What was first suspected, what it turned out to be, and the upstream PR, cited as `llamacpp-26472` and worded as verified in Task 4. State that the main study generates without a tool grammar and is unaffected.
7. **Reproduction** (`app:repro`). The repository URL, the pre-registration tag, and the commands from the README's "Reproducing" section in a `verbatim` block.

Add one sentence to the Method's arm-proofs paragraph or the Results' validity paragraph pointing to `app:safeguards`, and one sentence in Section 5 or 6 giving the tool-grammar finding in a single sentence with a pointer to `app:grammar`.

- [ ] **Step 2: Add the appendix to `body.tex`**

After the `\bibliography{refs}` line add:

```latex
\appendix
\input{sections/appendix}
```

- [ ] **Step 3: Build, read, commit**

Run: `python scripts/build_paper.py`
Expected: exit code 0; `main: body ends on page 6` or lower (the appendix follows the label and is not counted).

```bash
git add docs/paper/sections docs/paper/body.tex
git commit -m "Paper: appendix"
```

---

### Task 9: Verification pass and final build

**Files:**
- Modify: `docs/paper/VERIFICATION.md`; any section file a check finds wrong

**Interfaces:**
- Consumes: everything above.
- Produces: the completed verification record and the two final PDFs in `output/pdf/`.

- [ ] **Step 1: List every hand-entered number**

Run: `grep -n "% src:" docs/paper/sections/*.tex`

Then read each section file and list every literal number that is not a macro, a section or figure reference, or a year in a citation. Each one must have a `% src:` comment. Add any that are missing.

- [ ] **Step 2: Check each hand-entered number against its source**

Open the source named in each comment and compare. Record the results in `docs/paper/VERIFICATION.md` under a new heading:

```markdown
## Hand-entered numbers

| Value | Where in the paper | Source | Matches |
|---|---|---|---|
```

One row per number. Correct the paper where a value does not match, and note the correction in the row.

- [ ] **Step 3: Check the handoff's headline claims as rendered**

Open `output/pdf/main.pdf` and find each of the six headline claims in `docs/paper/HANDOFF.md` ("Headline claims"). For each, record in `VERIFICATION.md` under `## Headline claims` the sentence as printed and the findings section that supports it. Check specifically: the margin result is labelled pre-registered secondary; the 66-of-66 mechanism result and the Spark window rule are labelled exploratory; "0 of 540 explicit-request turns" and "0 of 1,080 explicit-request restores" are not confused.

- [ ] **Step 4: Check the honesty rules**

Record under `## Honesty rules` a yes or no, with the section number, for each: failed predictions appear in Limitations and Table 2; pre-registered and exploratory analyses are labelled; the scope is in the abstract and in Limitations; no sentence claims that caching *can* change output as a finding; no sentence makes a claim about vLLM, SGLang or any engine other than llama.cpp build `b10839`. Fix any "no".

- [ ] **Step 5: Check figures and macros are current**

Run: `python scripts/paper_numbers.py`, then `python scripts/make_figures.py --paper`, then `git status --short docs/paper/numbers.tex`
Expected: no change to `numbers.tex`. (PDF figures may differ in embedded timestamps; compare them by eye, not by `git status`.)

- [ ] **Step 6: Final build and test run**

Run: `python scripts/build_paper.py`
Expected: exit code 0, `main: body ends on page 6` or lower, no "undefined" lines.

Run: `python -m pytest -q`
Expected: 350 passed, 1 skipped.

Open `output/pdf/main.pdf` and `output/pdf/arxiv.pdf` and read both through once for rendering faults: overfull columns, a figure far from its reference, a table spilling into the margin, a "??".

- [ ] **Step 7: Commit**

```bash
git add docs/paper
git commit -m "Paper: verification record for every number and reference"
```

- [ ] **Step 8: Report to Manuel**

State: where the two PDFs are; the page count; what `VERIFICATION.md` found and corrected; any reference that was dropped or reworded; and the items that are his to decide: the acknowledgement wording, the arXiv endorsement, posting, and submission. Nothing is pushed.
