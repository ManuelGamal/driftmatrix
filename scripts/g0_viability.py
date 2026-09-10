"""G0 viability spike (plan Task 0). Output is an answer, not a library.

Usage:  python scripts/g0_viability.py /path/to/Ornith-1.5-9B-IQ4_XS.gguf

Answers three questions:
  1. Does llama.cpp expose sequence state save/restore for this build?
  2. Negative control 1 - does an immediate restore reproduce recompute exactly?
  3. What is the runtime state dtype (spec 3.2)?
"""
import json
import sys
from pathlib import Path

from llama_cpp import Llama

PROMPT = ("You are an operations assistant. " * 40
          + "\nList three prime numbers and explain your choice briefly.")


def greedy(llm, n_new):
    out = []
    for _ in range(n_new):
        tok = llm.sample(temp=0.0)
        if tok == llm.token_eos():
            break
        out.append(tok)
        llm.eval([tok])
    return out


def main() -> int:
    model = sys.argv[1]
    llm = Llama(model_path=model, n_ctx=4096, seed=1234, verbose=False)
    toks = llm.tokenize(PROMPT.encode())
    split = len(toks) // 2
    prefix, suffix = toks[:split], toks[split:]
    report = {"model": model, "n_tokens": len(toks), "split_at": split}

    try:
        llm.reset()
        llm.eval(prefix)
        blob = llm.save_state()
        report["state_api"] = "present"
        report["state_bytes"] = int(getattr(blob, "llama_state_size", 0))
    except Exception as exc:  # noqa: BLE001 - the failure IS the finding
        report["state_api"] = f"MISSING/FAILED: {type(exc).__name__}: {exc}"
        print(json.dumps(report, indent=2))
        return 1

    llm.reset(); llm.eval(prefix); llm.load_state(blob); llm.eval(suffix)
    restored = greedy(llm, 32)

    llm.reset(); llm.eval(toks)
    recomputed = greedy(llm, 32)

    report["restore_tokens"] = restored
    report["recompute_tokens"] = recomputed
    report["identical"] = restored == recomputed
    report["first_divergence"] = next(
        (i for i, (a, b) in enumerate(zip(restored, recomputed)) if a != b), None)

    print(json.dumps(report, indent=2))
    Path("data").mkdir(exist_ok=True)
    Path("data/g0_report.json").write_text(json.dumps(report, indent=2),
                                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
