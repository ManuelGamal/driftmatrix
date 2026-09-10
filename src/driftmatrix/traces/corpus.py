"""Freeze, hash and reload the trace corpus (spec 4.3)."""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from .generator import Turn


def _canonical(turns) -> str:
    return json.dumps([asdict(t) for t in turns], sort_keys=True, separators=(",", ":"))


def freeze_corpus(turns, path) -> str:
    blob = _canonical(turns)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    Path(path).write_text(json.dumps({"hash": digest, "turns": json.loads(blob)}),
                          encoding="utf-8")
    return digest


def load_corpus(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    turns = [Turn(**t) for t in data["turns"]]
    recomputed = hashlib.sha256(_canonical(turns).encode()).hexdigest()
    if recomputed != data["hash"]:
        raise ValueError(f"corpus hash mismatch: {recomputed} != {data['hash']}")
    return turns, data["hash"]
