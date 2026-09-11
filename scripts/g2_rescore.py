"""Re-score saved G2 reports offline: template round trip and parser adjudication (spec v4.4).

    python scripts/g2_rescore.py --gguf ornith=<path> --gguf qwen3=<path> --gguf spark=<path>
    python scripts/g2_rescore.py --dir docs/findings/g2 --gguf spark=<path>

Rewrites each g2_toolcalls_<model>_<stratum>.json in place: every row gains its
round-trip result and, where the parsers disagreed, an adjudication; the summary
and verdict are recomputed. No server is needed - rows store the model's output
and llama.cpp's parsed arguments.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.g2 import adjudicate_rows, summarise  # noqa: E402
from driftmatrix.traces.render import load_specials, load_template  # noqa: E402
from driftmatrix.traces.toolcalls import FORMAT_BY_MODEL  # noqa: E402


def _pair(text):
    model, _, path = text.partition("=")
    if model not in FORMAT_BY_MODEL or not path:
        raise argparse.ArgumentTypeError(f"expected MODEL=PATH with MODEL in {sorted(FORMAT_BY_MODEL)}")
    return model, path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "data"))
    ap.add_argument("--gguf", action="append", type=_pair, required=True, help="MODEL=path to its .gguf")
    args = ap.parse_args()

    for model, gguf in args.gguf:
        template, specials = load_template(gguf), load_specials(gguf)
        fmt = FORMAT_BY_MODEL[model]
        for path in sorted(Path(args.dir).glob(f"g2_toolcalls_{model}_*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            before = report["summary"]["verdict"]
            adjudicate_rows(report["rows"], template, fmt, specials)
            report["summary"] = summarise(report["rows"])
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            s = report["summary"]
            rt = "n/a" if s["our_roundtrip_rate"] is None else f"{s['our_roundtrip_rate']:.0%}"
            print(f"{path.name:<38} {before:<27} -> {s['verdict']:<24} round trip {rt:>4} | "
                  f"disagreements {s['parser_disagreements']} "
                  f"(server-side {s['server_parser_lossy']}, unresolved {s['unresolved_disagreements']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
