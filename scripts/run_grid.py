"""Execute the full condition grid (plan Task 22).

Usage:
  python scripts/run_grid.py /path/to/model.gguf <chunk_size> ./data/run1 [arch_level]
"""
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from driftmatrix.config import ModelSpec
from driftmatrix.grid import run_grid
from driftmatrix.runner.capture import RunMeta
from driftmatrix.runner.llamacpp import LlamaCppEngine
from driftmatrix.traces.corpus import freeze_corpus
from driftmatrix.traces.generator import generate_sessions


def _engine_commit() -> str:
    try:
        import llama_cpp
        return getattr(llama_cpp, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    model_path = sys.argv[1]
    chunk_size = int(sys.argv[2])
    out_dir = Path(sys.argv[3])
    arch_level = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    out_dir.mkdir(parents=True, exist_ok=True)

    turns = generate_sessions(n_sessions=60, turns_per_session=6, seed=2026)
    trace_hash = freeze_corpus(turns, out_dir / "corpus.json")

    spec = ModelSpec(Path(model_path).stem, arch_level, model_path, "IQ4_XS")
    engine = LlamaCppEngine(n_ctx=16384)
    engine.load(spec)

    meta = RunMeta(run_id=uuid.uuid4().hex[:12], engine="llama.cpp",
                   engine_commit=_engine_commit(), quant="IQ4_XS",
                   trace_hash=trace_hash,
                   run_started_at=datetime.now(timezone.utc))

    gen_path, sd_path = run_grid(engine, spec, turns, chunk_size, meta, out_dir)
    print(f"generations -> {gen_path}")
    print(f"statediffs  -> {sd_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
