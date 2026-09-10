import pandas as pd
from datetime import datetime, timezone
from driftmatrix.config import Mode, ModelSpec
from driftmatrix.runner.fake import FakeEngine
from driftmatrix.runner.capture import RunMeta
from driftmatrix.traces.generator import generate_sessions
from driftmatrix.grid import run_grid

META = RunMeta("r1", "fake", "deadbeef", "IQ4_XS", "h1",
               datetime(2026, 9, 10, tzinfo=timezone.utc))
SPEC = ModelSpec("fake-9b", 3, "/none.gguf", "IQ4_XS")
FREE = (Mode.FREE_RUNNING,)


def test_grid_covers_all_conditions_arms_and_replicates(tmp_path):
    turns = generate_sessions(1, 1, seed=1)
    g, _ = run_grid(FakeEngine(), SPEC, turns, 64, META, tmp_path,
                    replicates=2, modes=FREE)
    df = pd.read_parquet(g)
    assert set(df["condition_id"]) == set(range(8))
    assert set(df["arm"]) == {"restore", "recompute"}
    assert set(df["replicate"]) == {1, 2}


def test_grid_captures_both_measurement_modes(tmp_path):
    turns = generate_sessions(1, 1, seed=1)
    g, _ = run_grid(FakeEngine(), SPEC, turns, 64, META, tmp_path, replicates=1)
    assert set(pd.read_parquet(g)["mode"]) == {"free_running", "teacher_forced"}


def test_grid_writes_statediffs(tmp_path):
    turns = generate_sessions(1, 1, seed=1)
    _, s = run_grid(FakeEngine(), SPEC, turns, 64, META, tmp_path,
                    replicates=1, modes=FREE)
    assert pd.read_parquet(s).shape[0] > 0


def test_every_row_carries_run_metadata(tmp_path):
    turns = generate_sessions(1, 1, seed=1)
    g, _ = run_grid(FakeEngine(), SPEC, turns, 64, META, tmp_path,
                    replicates=1, modes=FREE)
    df = pd.read_parquet(g)
    assert (df["engine_commit"] == "deadbeef").all()
    assert (df["trace_hash"] == "h1").all()


def test_faithful_engine_produces_zero_state_difference(tmp_path):
    turns = generate_sessions(1, 1, seed=1)
    _, s = run_grid(FakeEngine(), SPEC, turns, 64, META, tmp_path,
                    replicates=1, modes=FREE)
    assert (pd.read_parquet(s)["max_abs_diff"] == 0.0).all()


def test_boundary_offset_is_recorded_and_matches_alignment(tmp_path):
    turns = generate_sessions(1, 1, seed=1)
    g, _ = run_grid(FakeEngine(), SPEC, turns, 64, META, tmp_path,
                    replicates=1, modes=FREE)
    df = pd.read_parquet(g)
    aligned = df[df["alignment"] == "aligned"]["boundary_token_offset"].unique()
    misaligned = df[df["alignment"] == "misaligned"]["boundary_token_offset"].unique()
    assert all(o % 64 == 0 for o in aligned)
    assert all(o % 64 == 32 for o in misaligned)
