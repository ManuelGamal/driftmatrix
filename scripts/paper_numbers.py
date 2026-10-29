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
