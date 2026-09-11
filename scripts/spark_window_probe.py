"""Spark window follow-up, one server configuration: exactness at each reuse boundary.

Start llama-server for Spark with the flags of the run (see the follow-up plan in
docs/findings/2026-09-11-spark-window-plan.md), then:

    python scripts/spark_window_probe.py --gguf <spark.gguf> --ubatch 512 --label ub512

--ubatch, --flash-attn and --swa-full must describe how the server was started;
they are recorded, not applied. Uses the first long-band main-study turn
(about 4.2K tokens), rendered through Spark's chat template. Writes
data/followup/spark_<label>.json. Exploratory; combine runs with
scripts/spark_window_analyze.py.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from driftmatrix.config import ENGINE_BUILD, ROUND_TRIPS, STUDY_BASES, STUDY_SEED  # noqa: E402
from driftmatrix.followup import HYPOTHESES, SPLITS  # noqa: E402
from driftmatrix.g1 import run_g1  # noqa: E402
from driftmatrix.runner.llamaserver import ServerClient  # noqa: E402
from driftmatrix.servergrid import loaded_model_matches  # noqa: E402
from driftmatrix.traces.conversations import generate_study_turns  # noqa: E402
from driftmatrix.traces.render import load_specials, load_template, render  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--gguf", required=True, help="the Spark .gguf the server loaded")
    ap.add_argument("--ubatch", type=int, required=True, help="must equal the server's -ub")
    ap.add_argument("--label", required=True, help="e.g. ub512, ub512_fa-off, ub512_no-swa-full")
    ap.add_argument("--flash-attn", choices=("auto", "on", "off"), default="auto", help="as the server was started")
    ap.add_argument("--swa-full", action=argparse.BooleanOptionalAction, default=True, help="as the server was started")
    ap.add_argument("--turn", default=f"el{STUDY_SEED}_000")
    ap.add_argument("--n-predict", type=int, default=16)
    args = ap.parse_args()

    turns = {t.conv_id: t for t in generate_study_turns(STUDY_BASES, STUDY_SEED, ROUND_TRIPS)}
    conv = turns[args.turn]
    prompt = render(load_template(args.gguf), conv.messages, conv.tools, add_generation_prompt=True,
                    enable_thinking=False, **load_specials(args.gguf))

    client = ServerClient(args.url)
    print(f"waiting for {args.url} ...", flush=True)
    client.wait_healthy()
    props = client.props()
    if props.get("build_info") != ENGINE_BUILD or not loaded_model_matches(props, args.gguf):
        print(f"refusing: server has build {props.get('build_info')!r} and model {props.get('model_path')!r}")
        return 1

    report = run_g1(client, prompt, ubatch=args.ubatch, splits=SPLITS, n_predict=args.n_predict)
    report.pop("props", None)                                   # holds local paths
    report.update({"label": args.label, "model": "spark", "gguf": Path(args.gguf).name, "build": ENGINE_BUILD,
                   "turn": args.turn, "config": {"ubatch": args.ubatch, "flash_attn": args.flash_attn,
                                                 "swa_full": args.swa_full}})
    out = REPO / "data" / "followup" / f"spark_{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\nspark {args.label}: n_ubatch {args.ubatch}, flash attention {args.flash_attn}, "
          f"swa-full {args.swa_full}, prompt {report['n_tokens']} tokens, floor exact {report['floor']['exact']}")
    names = ("window_limit", "single_microbatch")
    print(f"\n  {'k':>5}  {'aligned':<8}{'restore proven':<15}{'exact':<7}{'max TV':>9}   predicted by: "
          + ", ".join(names))
    for c in report["cases"]:
        if "error" in c:
            print(f"  {c['k']:>5}  ERROR {c['error'][:80]}")
            continue
        proven = c["restore_used"] is True and c["saved_exact"]
        preds = "  ".join(f"{str(HYPOTHESES[n](c['k'], args.ubatch)):<5}" for n in names)
        print(f"  {c['k']:>5}  {str(c['aligned']):<8}{str(proven):<15}{str(c['exact']):<7}{c['max_tv']:>9.2e}   {preds}")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
