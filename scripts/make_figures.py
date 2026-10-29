"""Draw the README and findings figures from committed results.

    python scripts/make_figures.py            # README PNGs in docs/figures
    python scripts/make_figures.py --paper    # PDFs in docs/paper/figures

Both sets are drawn in the paper's style with no title; the README and the paper
carry the captions as text. Reads docs/findings/g1/*.json and
docs/findings/study/analysis.json. Needs no GPU, server or raw records.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.figures import (  # noqa: E402
    load_g1, plot_action_changes, plot_drift_per_turn, plot_exactness_grid, plot_margin_bins, use_paper_style,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true",
                        help="write title-free PDFs to docs/paper/figures instead of the README PNGs")
    args = parser.parse_args()
    out = REPO / "docs" / "paper" / "figures" if args.paper else REPO / "docs" / "figures"
    ext = "pdf" if args.paper else "png"
    use_paper_style()
    grid = load_g1(sorted((REPO / "docs" / "findings" / "g1").glob("g1_*.json")))
    analysis = json.loads((REPO / "docs" / "findings" / "study" / "analysis.json").read_text(encoding="utf-8"))
    plot_exactness_grid(grid, out / f"g1_exactness_grid.{ext}", paper=True)
    plot_action_changes(analysis, out / f"action_changes.{ext}", paper=True)
    plot_margin_bins(analysis, out / f"margin_bins.{ext}", paper=True)
    plot_drift_per_turn(analysis, out / f"drift_per_turn.{ext}", paper=True)
    for p in sorted(out.glob(f"*.{ext}")):
        print(f"{p.relative_to(REPO)}  {p.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
