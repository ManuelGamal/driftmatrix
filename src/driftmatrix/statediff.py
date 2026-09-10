"""Direct comparison of restored vs recomputed state tensors (spec 5.3)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import LayerType

_REL_FLOOR = 1e-6


@dataclass(frozen=True)
class LayerDiff:
    layer_idx: int
    layer_type: LayerType
    max_abs_diff: float
    max_rel_diff: float
    frobenius_norm_diff: float
    state_dtype: str


def diff_layer(idx, layer_type, restored, recomputed) -> LayerDiff:
    if restored.shape != recomputed.shape:
        raise ValueError(
            f"layer {idx}: shape mismatch {restored.shape} vs {recomputed.shape}")
    a = restored.astype(np.float64)
    b = recomputed.astype(np.float64)
    delta = np.abs(a - b)
    denom = np.abs(b)
    mask = denom > _REL_FLOOR
    rel = float(np.max(delta[mask] / denom[mask])) if mask.any() else 0.0
    return LayerDiff(
        layer_idx=idx,
        layer_type=layer_type,
        max_abs_diff=float(delta.max()) if delta.size else 0.0,
        max_rel_diff=rel,
        frobenius_norm_diff=float(np.linalg.norm(a - b)),
        state_dtype=str(restored.dtype),
    )


def diff_all(restored, recomputed, layer_types):
    return [diff_layer(i, layer_types[i], restored[i], recomputed[i])
            for i in sorted(restored)]
