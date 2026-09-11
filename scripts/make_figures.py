"""Draw the README and findings figures from committed results.

    python scripts/make_figures.py

Reads docs/findings/g1/*.json and docs/findings/study/analysis.json and writes
docs/figures/*.png. Needs no GPU, server or raw records.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.figures import (  # noqa: E402
    load_g1, plot_action_changes, plot_drift_per_turn, plot_exactness_grid, plot_margin_bins,
)


def main() -> int:
    out = REPO / "docs" / "figures"
    grid = load_g1(sorted((REPO / "docs" / "findings" / "g1").glob("g1_*.json")))
    analysis = json.loads((REPO / "docs" / "findings" / "study" / "analysis.json").read_text(encoding="utf-8"))
    plot_exactness_grid(grid, out / "g1_exactness_grid.png")
    plot_action_changes(analysis, out / "action_changes.png")
    plot_margin_bins(analysis, out / "margin_bins.png")
    plot_drift_per_turn(analysis, out / "drift_per_turn.png")
    for p in sorted(out.glob("*.png")):
        print(f"{p.relative_to(REPO)}  {p.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
