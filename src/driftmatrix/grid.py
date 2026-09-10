"""Full condition-grid execution (spec 4.1, 4.4, 5.3).

Runs both arms in both measurement modes across all 8 conditions, capturing
generations and per-layer state differences to Parquet.

Ordering matters: a real engine has one global context, so each arm's final
state must be snapshotted before the next arm resets it.
"""
from __future__ import annotations
from pathlib import Path

from .config import ALL_CONDITIONS, CONTINUATION_TOKENS, Mode, Outcome
from .metrics.behavioral import ToolCall
from .runner.arms import ArmResult, run_recompute_arm, run_restore_arm
from .runner.capture import generations_rows, statediff_rows, write_parquet
from .runner.engine import GenerationResult
from .statediff import diff_all
from .traces.boundary import boundary_offset

N_PROBS = 10


def _synthetic_tokens(offset: int, suffix_len: int = 40):
    """Deterministic stand-in token stream whose boundary lands at `offset`.

    Real runs substitute tokenizer output; the grid only needs a sequence of a
    known length with the save boundary in a known place.
    """
    return list(range(1, offset + suffix_len + 1))


def _snapshot(engine, seq_id, path, ok):
    """Save an arm's final state. Absence is recorded, never fatal."""
    if not ok:
        return None
    try:
        engine.save_state(seq_id, path)
        return path
    except Exception:  # noqa: BLE001 - a missing snapshot is data, not a crash
        return None


def _teacher_forced_rows(engine, seq_id, pre, suf, res, other, meta, spec, cond,
                         rep, turn, offset, tool):
    """Score the *other* arm's tokens under this arm's context."""
    engine.reset(seq_id)
    engine.eval_tokens(seq_id, pre + suf)
    dists = engine.score_tokens(seq_id, other.generation.tokens, N_PROBS)
    forced = ArmResult(
        res.arm, res.outcome,
        GenerationResult(tokens=list(other.generation.tokens), dists=dists, text=""),
        res.state_path, res.error,
    )
    return generations_rows(meta, spec, cond, rep, Mode.TEACHER_FORCED,
                            turn.session_id, turn.turn_idx, offset, forced, tool)


def run_grid(engine, spec, turns, chunk_size: int, meta, out_dir,
             replicates: int = 2, modes=(Mode.FREE_RUNNING, Mode.TEACHER_FORCED)):
    out_dir = Path(out_dir)
    states = out_dir / "states"
    states.mkdir(parents=True, exist_ok=True)
    gen_rows, sd_rows = [], []

    for cond in ALL_CONDITIONS:
        offset = boundary_offset(cond.prefix_band, cond.alignment, chunk_size)
        max_new = CONTINUATION_TOKENS[cond.continuation_band]
        for rep in range(1, replicates + 1):
            for turn in turns:
                toks = _synthetic_tokens(offset)
                pre, suf = toks[:offset], toks[offset:]
                tag = f"c{cond.condition_id}_r{rep}_{turn.session_id}_{turn.turn_idx}"
                tool = (ToolCall.from_payload(turn.expected_tool, {})
                        if turn.expected_tool else None)

                r = run_restore_arm(engine, 0, pre, suf, max_new, N_PROBS,
                                    states / f"{tag}.bin")
                final_r = _snapshot(engine, 0, states / f"{tag}_final_restore.bin",
                                    r.outcome is Outcome.OK)

                c = run_recompute_arm(engine, 1, pre, suf, max_new, N_PROBS)
                final_c = _snapshot(engine, 1, states / f"{tag}_final_recompute.bin",
                                    c.outcome is Outcome.OK)

                for res in (r, c):
                    gen_rows += generations_rows(
                        meta, spec, cond, rep, Mode.FREE_RUNNING,
                        turn.session_id, turn.turn_idx, offset, res, tool)

                if (Mode.TEACHER_FORCED in modes and r.generation and c.generation
                        and hasattr(engine, "score_tokens")):
                    gen_rows += _teacher_forced_rows(
                        engine, 0, pre, suf, r, c, meta, spec, cond, rep, turn,
                        offset, tool)
                    gen_rows += _teacher_forced_rows(
                        engine, 1, pre, suf, c, r, meta, spec, cond, rep, turn,
                        offset, tool)

                if final_r is not None and final_c is not None:
                    tr = engine.read_state_tensors(final_r)
                    tc = engine.read_state_tensors(final_c)
                    types = engine.layer_types()
                    common = sorted(set(tr) & set(tc) & set(types))
                    if common:
                        sd_rows += statediff_rows(
                            meta, spec, cond, rep,
                            diff_all({k: tr[k] for k in common},
                                     {k: tc[k] for k in common},
                                     {k: types[k] for k in common}))

    g = out_dir / "generations.parquet"
    s = out_dir / "statediffs.parquet"
    write_parquet(gen_rows, g)
    write_parquet(sd_rows, s)
    return g, s
