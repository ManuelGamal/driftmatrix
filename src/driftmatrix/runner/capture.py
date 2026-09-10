"""Parquet capture (spec 7.2). The seam: engine writes once, analysis reads."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

GENERATIONS_COLUMNS = (
    "model_id", "arch_level", "condition_id", "alignment", "prefix_band",
    "continuation_band", "arm", "replicate", "mode", "session_id", "turn_idx",
    "position", "token_ids", "logprobs", "chosen_token_id", "tool_name",
    "tool_args_hash", "boundary_token_offset", "outcome", "error",
    "engine", "engine_commit", "quant", "trace_hash", "run_id", "run_started_at",
)

STATEDIFF_COLUMNS = (
    "model_id", "arch_level", "condition_id", "replicate", "layer_idx",
    "layer_type", "max_abs_diff", "max_rel_diff", "frobenius_norm_diff",
    "state_dtype", "engine_commit", "quant", "run_id",
)


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    engine: str
    engine_commit: str
    quant: str
    trace_hash: str
    run_started_at: object


def _base(meta, spec, cond, replicate):
    return {
        "model_id": spec.model_id, "arch_level": spec.arch_level,
        "condition_id": cond.condition_id,
        "alignment": cond.alignment.value,
        "prefix_band": cond.prefix_band.value,
        "continuation_band": cond.continuation_band.value,
        "replicate": replicate,
        "engine": meta.engine, "engine_commit": meta.engine_commit,
        "quant": meta.quant, "trace_hash": meta.trace_hash,
        "run_id": meta.run_id, "run_started_at": meta.run_started_at,
    }


def generations_rows(meta, spec, cond, replicate, mode, session_id, turn_idx,
                     boundary_token_offset, result, tool):
    base = dict(_base(meta, spec, cond, replicate))
    base.update({
        "arm": result.arm.value, "mode": mode.value,
        "session_id": session_id, "turn_idx": turn_idx,
        "boundary_token_offset": boundary_token_offset,
        "outcome": result.outcome.value, "error": result.error,
        "tool_name": tool.name if tool else None,
        "tool_args_hash": tool.args_hash if tool else None,
    })
    if result.generation is None:
        row = dict(base)
        row.update({"position": None, "token_ids": None, "logprobs": None,
                    "chosen_token_id": None})
        return [row]
    rows = []
    for i, (tok, dist) in enumerate(zip(result.generation.tokens,
                                        result.generation.dists)):
        row = dict(base)
        row.update({"position": i, "token_ids": list(dist.token_ids),
                    "logprobs": list(dist.logprobs), "chosen_token_id": tok})
        rows.append(row)
    return rows


def statediff_rows(meta, spec, cond, replicate, diffs):
    return [{
        "model_id": spec.model_id, "arch_level": spec.arch_level,
        "condition_id": cond.condition_id, "replicate": replicate,
        "layer_idx": d.layer_idx, "layer_type": d.layer_type.value,
        "max_abs_diff": d.max_abs_diff, "max_rel_diff": d.max_rel_diff,
        "frobenius_norm_diff": d.frobenius_norm_diff, "state_dtype": d.state_dtype,
        "engine_commit": meta.engine_commit, "quant": meta.quant,
        "run_id": meta.run_id,
    } for d in diffs]


def write_parquet(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
