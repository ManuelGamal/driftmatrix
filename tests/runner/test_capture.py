import pandas as pd
from datetime import datetime, timezone
from driftmatrix.config import ALL_CONDITIONS, Arm, Mode, ModelSpec, LayerType, Outcome
from driftmatrix.statediff import LayerDiff
from driftmatrix.runner.fake import FakeEngine
from driftmatrix.runner.arms import run_recompute_arm, ArmResult
from driftmatrix.runner.capture import (
    RunMeta, generations_rows, statediff_rows, write_parquet,
    GENERATIONS_COLUMNS, STATEDIFF_COLUMNS,
)

META = RunMeta("run1", "llama.cpp", "abc123", "IQ4_XS", "hash1",
               datetime(2026, 9, 10, tzinfo=timezone.utc))
SPEC = ModelSpec("ornith-9b", 3, "/m.gguf", "IQ4_XS")


def _rows():
    r = run_recompute_arm(FakeEngine(), 0, [1, 2], [3], 4, 5)
    return generations_rows(META, SPEC, ALL_CONDITIONS[0], 1, Mode.FREE_RUNNING,
                            "s1", 0, 100, r, tool=None)


def test_generations_rows_have_exact_schema():
    assert set(_rows()[0]) == set(GENERATIONS_COLUMNS)


def test_one_row_per_generated_position():
    rows = _rows()
    assert len(rows) == 4
    assert [x["position"] for x in rows] == [0, 1, 2, 3]


def test_failed_arm_produces_a_single_outcome_row_with_no_position():
    bad = ArmResult(Arm.RESTORE, Outcome.CRASHED, None, None, "RuntimeError: boom")
    rows = generations_rows(META, SPEC, ALL_CONDITIONS[0], 1, Mode.FREE_RUNNING,
                            "s1", 0, 100, bad, tool=None)
    assert len(rows) == 1
    assert rows[0]["outcome"] == Outcome.CRASHED.value
    assert rows[0]["position"] is None


def test_metadata_is_stamped_on_every_row():
    for row in _rows():
        assert row["engine_commit"] == "abc123"
        assert row["quant"] == "IQ4_XS"
        assert row["trace_hash"] == "hash1"
        assert row["boundary_token_offset"] == 100


def test_statediff_rows_schema():
    diffs = [LayerDiff(0, LayerType.LINEAR_ATTENTION, 0.1, 0.2, 0.3, "float32")]
    rows = statediff_rows(META, SPEC, ALL_CONDITIONS[0], 1, diffs)
    assert set(rows[0]) == set(STATEDIFF_COLUMNS)
    assert rows[0]["layer_type"] == "linear_attention"


def test_write_parquet_roundtrips(tmp_path):
    rows = _rows()
    p = tmp_path / "g.parquet"
    write_parquet(rows, p)
    assert len(pd.read_parquet(p)) == len(rows)
