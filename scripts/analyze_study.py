"""Run the pre-registered main-study analysis - only once all three models have finished.

    python scripts/analyze_study.py --gguf ornith=<path> --gguf qwen3=<path> --gguf spark=<path>

Refuses to compute anything unless data/study/<model>.jsonl exists for all three
models, each on the pre-registered build and trace hash, with all 360 turns
complete (pre-registration section 11: no treatment effect is examined before
then). Writes data/study/analysis.json and prints the confirmatory results.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.analysis import MODELS, ModelData, run_analysis  # noqa: E402
from driftmatrix.config import ENGINE_BUILD, ROUND_TRIPS, STUDY_BASES, STUDY_SEED  # noqa: E402
from driftmatrix.servergrid import load_records  # noqa: E402
from driftmatrix.traces.conversations import generate_study_turns, trace_hash  # noqa: E402
from driftmatrix.traces.render import load_specials, load_template  # noqa: E402
from driftmatrix.traces.toolcalls import FORMAT_BY_MODEL  # noqa: E402


def _pair(text):
    model, _, path = text.partition("=")
    if model not in FORMAT_BY_MODEL or not path:
        raise argparse.ArgumentTypeError(f"expected MODEL=PATH with MODEL in {sorted(FORMAT_BY_MODEL)}")
    return model, path


def _fmt_interval(v):
    i = v.get("interval")
    return "n/a" if not i else f"[{i['low']:.4g}, {i['high']:.4g}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "data" / "study"))
    ap.add_argument("--gguf", action="append", type=_pair, required=True, help="MODEL=path to its .gguf")
    args = ap.parse_args()
    ggufs = dict(args.gguf)

    expected = generate_study_turns(STUDY_BASES, STUDY_SEED, ROUND_TRIPS)
    expected_ids, expected_hash = {t.conv_id for t in expected}, trace_hash(expected)
    datasets, problems = [], []
    for model in MODELS:
        path = Path(args.dir) / f"{model}.jsonl"
        if model not in ggufs:
            problems.append(f"{model}: no --gguf given")
            continue
        if not path.exists():
            problems.append(f"{model}: {path.name} missing")
            continue
        meta, turns, arms = load_records(path)
        if meta is None or meta.get("build") != ENGINE_BUILD or meta.get("trace_hash") != expected_hash:
            problems.append(f"{model}: build or trace hash differs from the pre-registration")
        if set(turns) != expected_ids:
            problems.append(f"{model}: {len(set(turns) & expected_ids)} of {len(expected_ids)} turns complete")
        datasets.append(ModelData(model, turns, arms, load_template(ggufs[model]),
                                  FORMAT_BY_MODEL[model], load_specials(ggufs[model])))
    if problems:
        print("Not analysing - the main study is not complete on the pre-registered setup:")
        for p in problems:
            print(f"  {p}")
        return 2

    result = run_analysis(datasets)
    out = Path(args.dir) / "analysis.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print("\nDeterminism (recompute floor):")
    for m, d in result["determinism"].items():
        print(f"  {m:<8}floor inexact on {d['floor_inexact']} of {d['turns']} turns"
              f"{'  -> NONDETERMINISTIC, claims conditional' if d['nondeterministic'] else ''}")
    for name in ("A", "B", "C"):
        fam = result["families"][name]
        print(f"\nFamily {name} ({len(fam)} members, Holm within family)")
        for key, v in fam.items():
            flags = "CLAIMED" if v["claimed"] else ("contradicts prediction" if v.get("contradicts_prediction") else "")
            est = "n/a" if v["estimate"] is None else f"{v['estimate']:.4g}"
            extra = f" excluded {v['excluded']}" if "excluded" in v else ""
            print(f"  {key:<42} n {v['n']:>3}{extra}  estimate {est:>9}  95% {_fmt_interval(v):<24}"
                  f" p {v['p_value']:.3g}  {flags}{' (conditional)' if v.get('conditional') else ''}")
    print("\nHypotheses:")
    for h, verdict in result["hypotheses"].items():
        if h == "H1_mitigation":
            for m, v in verdict.items():
                state = "holds" if v["holds"] else "FAILS"
                detail = "" if v["holds"] else f"; inexact: {v['inexact'][:5]}"
                print(f"  {h} {m:<8}{state} on {v['valid_snapped_turns']} valid snapped turns{detail}")
        else:
            print(f"  {h:<30}{verdict}")
    print(f"\nfull results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
