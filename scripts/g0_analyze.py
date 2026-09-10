"""Regenerate the G0 findings tables from committed probe reports.

    python scripts/g0_analyze.py
    python scripts/g0_analyze.py --decode ornith="C:/.../Ornith-1.5-9B-MTP-IQ4_XS.gguf"

Distribution drift is measured with the spec's metric (spec 5.2): JSD and total
variation over each arm's top-k plus one <other> bucket for the missing mass.

Do NOT compare arms by the maximum logprob difference across the top-k: that
number is dominated by low-probability tail tokens and made models with
negligible drift look as disturbed as the hybrid in the first analysis.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.gguf import decode, read_metadata  # noqa: E402
from driftmatrix.metrics.distributional import (  # noqa: E402
    TokenDist, align_supports, jsd, total_variation,
)

DEFAULT_REPORTS = {
    lab: REPO / "docs" / "findings" / "g0" / f"g0_{lab}_report.json"
    for lab in ("ornith", "spark", "qwen3_dense")
}


def _dist(pos):
    return TokenDist([i for i, _ in pos], [lp for _, lp in pos])


def drift(case):
    """Per-position JSD and TV while both arms still share the same context.

    Context at position p is prompt + generated[:p], identical in both arms
    through the first divergence, so positions 0..first_divergence compare.
    """
    fd = case["first_divergence"]
    upto = min(8 if fd is None else fd + 1,
               len(case["restore_top"]), len(case["recompute_top"]))
    js, tv = [], []
    for p in range(upto):
        P, Q = align_supports(_dist(case["restore_top"][p]), _dist(case["recompute_top"][p]))
        js.append(jsd(P, Q))
        tv.append(total_variation(P, Q))
    return js, tv


def _kv(text):
    label, _, value = text.partition("=")
    return label, value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", type=_kv, default=[],
                    help="LABEL=PATH; defaults to the committed G0 reports")
    ap.add_argument("--decode", action="append", type=_kv, default=[],
                    help="LABEL=GGUF path, to decode that model's output tokens")
    args = ap.parse_args()

    reports = dict(args.report) if args.report else DEFAULT_REPORTS
    loaded = {lab: json.loads(Path(p).read_text(encoding="utf-8"))
              for lab, p in reports.items()}

    print("## Validity\n")
    print("| model | tokens | floor exact | split | restore prompt_n = suffix | recompute clean | saved = k |")
    print("|---|---|---|---|---|---|---|")
    for lab, r in loaded.items():
        for c in r["cases"].values():
            print(f"| {lab} | {r['n_tokens']} | {r['floor']['identical']} | {c['split_at']} | "
                  f"{c['restore_prompt_n'] == c['n_suffix']} | {c['recompute_clean']} | "
                  f"{c['n_saved'] == c['split_at']} |")

    print("\n## Drift\n")
    print("| model | split | ubatch-aligned | tokens identical | first div | max JSD | max TV | JSD by position |")
    print("|---|---|---|---|---|---|---|---|")
    for lab, r in loaded.items():
        for c in r["cases"].values():
            js, tv = drift(c)
            per = " ".join(f"{x:.4f}" for x in js)
            print(f"| {lab} | {c['split_at']} | {c['aligned_to_ubatch']} | {c['identical']} | "
                  f"{c['first_divergence']} | {max(js):.4f} | {max(tv):.4f} | {per} |")

    for lab, gguf in args.decode:
        vocab = read_metadata(gguf, capture=("tokenizer.ggml.tokens",))["kv"]["tokenizer.ggml.tokens"]
        r = loaded[lab]
        print(f"\n## Decoded output: {lab} (first 12 tokens)\n")
        any_case = next(iter(r["cases"].values()))
        print(f"- recompute: `{decode(vocab, any_case['recompute_tokens'][:12])!r}`")
        for c in r["cases"].values():
            print(f"- restore @ {c['split_at']}: `{decode(vocab, c['restore_tokens'][:12])!r}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
