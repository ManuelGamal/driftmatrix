"""Secondary result (plan Task 22): per-turn cost of full-prompt reprocessing.

Reproduces the cost reported in llama.cpp #20225 under controlled conditions.

Usage:  python scripts/reproduce_20225.py /path/to/model.gguf
"""
import json
import sys
import time
from pathlib import Path

from driftmatrix.config import ModelSpec
from driftmatrix.runner.llamacpp import LlamaCppEngine
from driftmatrix.traces.generator import generate_sessions


def main() -> int:
    model = sys.argv[1]
    engine = LlamaCppEngine(n_ctx=16384)
    engine.load(ModelSpec("treatment", 3, model, "IQ4_XS"))

    timings = []
    for turn in generate_sessions(n_sessions=1, turns_per_session=8, seed=3):
        text = (turn.prefix_text + turn.suffix_text).encode()
        toks = engine._llm.tokenize(text)
        start = time.perf_counter()
        engine.reset(0)
        engine.eval_tokens(0, toks)
        timings.append({"turn": turn.turn_idx, "n_tokens": len(toks),
                        "prefill_seconds": round(time.perf_counter() - start, 3)})

    print(json.dumps(timings, indent=2))
    Path("data").mkdir(exist_ok=True)
    Path("data/20225_cost.json").write_text(json.dumps(timings, indent=2),
                                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
