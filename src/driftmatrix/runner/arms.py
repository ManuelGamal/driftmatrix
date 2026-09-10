"""The two arms (spec 4.2).

Restore:   eval(prefix) -> save -> fresh context -> load -> eval(suffix) -> generate
Recompute: fresh context -> eval(prefix + suffix) -> generate

Identical tokens, identical weights, greedy decoding. The only difference is
whether the state was restored or recomputed.
"""
from __future__ import annotations
from dataclasses import dataclass
from ..config import Arm, Outcome


@dataclass(frozen=True)
class ArmResult:
    arm: Arm
    outcome: Outcome
    generation: object
    state_path: object
    error: str


def _classify(exc):
    if isinstance(exc, NotImplementedError):
        return Outcome.UNSUPPORTED
    if isinstance(exc, TimeoutError):
        return Outcome.TIMEOUT
    if isinstance(exc, OSError):
        return Outcome.STATE_IO_FAILED
    return Outcome.CRASHED


def _fail(arm, exc, state_path):
    return ArmResult(arm, _classify(exc), None, state_path,
                     f"{type(exc).__name__}: {exc}")


def run_restore_arm(engine, seq_id, prefix, suffix, max_new, n_probs, state_path):
    try:
        engine.reset(seq_id)
        engine.eval_tokens(seq_id, prefix)
        engine.save_state(seq_id, state_path)
        engine.reset(seq_id)
        engine.load_state(seq_id, state_path)
        engine.eval_tokens(seq_id, suffix)
        gen = engine.generate(seq_id, max_new, n_probs)
        return ArmResult(Arm.RESTORE, Outcome.OK, gen, state_path, None)
    except Exception as exc:  # noqa: BLE001 - classified above
        return _fail(Arm.RESTORE, exc, state_path)


def run_recompute_arm(engine, seq_id, prefix, suffix, max_new, n_probs):
    try:
        engine.reset(seq_id)
        engine.eval_tokens(seq_id, list(prefix) + list(suffix))
        gen = engine.generate(seq_id, max_new, n_probs)
        return ArmResult(Arm.RECOMPUTE, Outcome.OK, gen, None, None)
    except Exception as exc:  # noqa: BLE001 - classified above
        return _fail(Arm.RECOMPUTE, exc, None)
