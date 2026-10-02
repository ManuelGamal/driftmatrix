"""Build the paper in both layouts and check it.

    python scripts/build_paper.py            # ACM layout (main) and arXiv layout (arxiv)
    python scripts/build_paper.py --max-pages 6

Fails when the body of the ACM layout, including its figures and tables, runs
past the page limit, or when either layout has an undefined or multiply defined
citation or reference.
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
_FLOATS = re.compile(r"\\bodyfloats\{(\d+)\}\{(\d+)\}")
_FLOAT = re.compile(r"\\contentsline \{(figure|table)\}\{\\numberline \{(\d+)\}.*\}\{(\d+)\}\{(?:figure|table)\.")
# The summary lines catch warnings whose own line the log wrapped.
_PROBLEM = re.compile(r"(Citation|Reference) `[^']*' .*undefined|multiply defined"
                      r"|There were undefined|There were multiply-defined")


def body_pages(aux_text: str) -> int:
    """Page on which the body ends, from the endofbody label in a .aux file."""
    found = _LABEL.search(aux_text)
    if not found:
        raise ValueError("label endofbody not found in the .aux file")
    return int(found.group(1))


def body_float_pages(aux_text: str) -> list[int]:
    """Pages of the body's figures and tables, including any that floated past the label.

    body.tex writes \\bodyfloats{figures}{tables} at the end of the body; floats numbered
    up to those counts belong to the body, later ones to the appendix.
    """
    found = _FLOATS.search(aux_text)
    if not found:
        raise ValueError("\\bodyfloats not found in the .aux file")
    limit = {"figure": int(found.group(1)), "table": int(found.group(2))}
    return [int(page) for kind, number, page in _FLOAT.findall(aux_text) if int(number) <= limit[kind]]


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
        aux = (OUT / f"{name}.aux").read_text(encoding="utf-8", errors="replace")
        pages = max([body_pages(aux)] + body_float_pages(aux))
        print(f"{name}: body ends on page {pages}")
        if problems or (name == "main" and pages > args.max_pages):
            failed = True
    if failed:
        print(f"FAILED: undefined references, or the ACM body is over {args.max_pages} pages")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
