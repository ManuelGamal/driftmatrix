"""Factor model and project-wide constants. Single source of truth."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Alignment(str, Enum):
    ALIGNED = "aligned"
    MISALIGNED = "misaligned"


class PrefixBand(str, Enum):
    SHORT = "short"
    LONG = "long"


class ContinuationBand(str, Enum):
    SHORT = "short"
    LONG = "long"


class Arm(str, Enum):
    RESTORE = "restore"
    RECOMPUTE = "recompute"


class Mode(str, Enum):
    FREE_RUNNING = "free_running"
    TEACHER_FORCED = "teacher_forced"


class Outcome(str, Enum):
    OK = "ok"
    UNSUPPORTED = "unsupported"
    CRASHED = "crashed"
    TIMEOUT = "timeout"
    STATE_IO_FAILED = "state_io_failed"


class LayerType(str, Enum):
    LINEAR_ATTENTION = "linear_attention"
    FULL_ATTENTION = "full_attention"
    SLIDING_ATTENTION = "sliding_attention"


@dataclass(frozen=True, order=True)
class Condition:
    alignment: Alignment
    prefix_band: PrefixBand
    continuation_band: ContinuationBand

    @property
    def condition_id(self) -> int:
        return (
            (1 if self.alignment is Alignment.MISALIGNED else 0)
            | (2 if self.prefix_band is PrefixBand.LONG else 0)
            | (4 if self.continuation_band is ContinuationBand.LONG else 0)
        )

    @classmethod
    def from_id(cls, cid: int) -> "Condition":
        if not 0 <= cid < 8:
            raise ValueError(f"condition_id must be 0..7, got {cid}")
        return cls(
            Alignment.MISALIGNED if cid & 1 else Alignment.ALIGNED,
            PrefixBand.LONG if cid & 2 else PrefixBand.SHORT,
            ContinuationBand.LONG if cid & 4 else ContinuationBand.SHORT,
        )


ALL_CONDITIONS: tuple[Condition, ...] = tuple(Condition.from_id(i) for i in range(8))


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    arch_level: int
    gguf_path: str
    quant: str


PREFIX_TOKENS = {PrefixBand.SHORT: 1024, PrefixBand.LONG: 8192}
CONTINUATION_TOKENS = {ContinuationBand.SHORT: 32, ContinuationBand.LONG: 256}

BOOTSTRAP_RESAMPLES = 10_000
ALPHA = 0.05
NOISE_FLOOR_NATS = 0.01
SELF_INCONSISTENCY_MULTIPLIER = 2.0
TARGET_EFFECT_RATE = 0.02
TARGET_POWER = 0.80

# Main study against llama-server - fixed by docs/preregistration/2026-09-10-analysis-plan.md.
# PREFIX_TOKENS and CONTINUATION_TOKENS above belong to the off-path Python engine.
STUDY_SEED = 20260910
PILOT_SEED = 20260911          # G3 pilot turns; never part of the confirmatory analysis
STUDY_BASES = 90               # x 2 prefix bands x 2 strata = 360 turns per model
PILOT_BASES = 10
ROUND_TRIPS = {"short": 8, "long": 60}
STUDY_UBATCH = 512
STUDY_CONTEXT = 8192
MAX_NEW_TOKENS = 192
TOP_K = 10
ENGINE_BUILD = "b10839-0cae43063"
ARCH_LEVELS = {"ornith": 3, "spark": 2, "qwen3": 1}   # hybrid recurrent, sliding window, dense
