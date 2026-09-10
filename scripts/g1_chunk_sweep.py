"""Gate G1 (plan Task 12): locate the linear-attention kernel chunk size.

Usage:  python scripts/g1_chunk_sweep.py /path/to/model.gguf ./data/g1

Sweeps the save-boundary offset and looks for the periodicity of divergence.
A period means alignment is the mechanism factor; no period means alignment is
a null factor and the study proceeds on prefix/continuation length.
"""
import json
import sys
from pathlib import Path

from driftmatrix.config import ModelSpec
from driftmatrix.runner.llamacpp import LlamaCppEngine
from driftmatrix.sweep import detect_period, sweep_offsets


def main() -> int:
    model_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = LlamaCppEngine(n_ctx=4096)
    engine.load(ModelSpec("treatment", 3, model_path, "IQ4_XS"))

    tokens = list(range(1, 1200))
    offsets = list(range(256, 768, 8))
    points = sweep_offsets(engine, tokens, offsets, max_new=32, n_probs=10,
                           state_dir=out_dir)
    period = detect_period(points)

    result = {
        "period": period,
        "n_offsets": len(points),
        "diverged_offsets": [p.offset for p in points if p.diverged],
        "points": [{"offset": p.offset, "diverged": p.diverged,
                    "first_div_position": p.first_div_position} for p in points],
    }
    print(json.dumps({k: v for k, v in result.items() if k != "points"}, indent=2))
    (out_dir / "chunk_sweep.json").write_text(json.dumps(result, indent=2),
                                              encoding="utf-8")
    if period is None:
        print("\nNo period found. See plan Task 12 gate G1 for the two branches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
