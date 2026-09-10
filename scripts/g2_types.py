"""Break the G2 ambiguous stratum down by request type, against explicit.

    python scripts/g2_types.py
    python scripts/g2_types.py --dir docs/findings/g2

For each model and each ambiguous request type: how many turns produced a tool
call, and the median action margin over all turns, calls or not. The request type of a
conversation is its index modulo the number of types - the rotation
generate_conversations uses, pinned by tests/traces/test_conversations.py.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.traces.conversations import AMBIGUOUS_TYPES  # noqa: E402

TYPES = [f.__name__[len("_amb_"):] for f in AMBIGUOUS_TYPES]
KIND = {"something_on_record": "tool-choice", "total_records": "tool-choice",
        "that_record": "under-specified", "add_earlier": "under-specified",
        "write_up": "under-specified"}
MODELS = ("ornith", "qwen3", "spark")


def median(values):
    vals = [v for v in values if v is not None]
    return f"{statistics.median(vals):.2f}" if vals else "n/a"


def summarise(label, rows):
    calls = sum(r["tool_call_made"] for r in rows)
    return f"{label:<24}{calls:>4}/{len(rows):<4}{median(r['decision_margin'] for r in rows):>10}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "data"))
    args = ap.parse_args()

    for model in MODELS:
        paths = {st: Path(args.dir) / f"g2_toolcalls_{model}_{st}.json" for st in ("explicit", "ambiguous")}
        if not all(p.exists() for p in paths.values()):
            print(f"{model}: missing reports")
            continue
        explicit, ambiguous = (json.loads(paths[st].read_text(encoding="utf-8"))["rows"]
                               for st in ("explicit", "ambiguous"))
        print(f"\n{model}\n  {'request type':<24}{'calls':>9}{'margin':>10}  kind")
        by_kind = {}
        for i, t in enumerate(TYPES):
            rows = [r for r in ambiguous if int(r["conv_id"].split("_")[1]) % len(TYPES) == i]
            by_kind.setdefault(KIND.get(t, "?"), []).extend(rows)
            print(f"  {summarise(t, rows)}  {KIND.get(t, '?')}")
        for kind, rows in by_kind.items():
            print(f"  {summarise('all ' + kind, rows)}")
        print(f"  {summarise('explicit', explicit)}")
    print("\ncalls = turns producing a parseable tool call; margin = median recompute-arm action margin (nats) over all turns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
