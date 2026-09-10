from driftmatrix.traces.generator import generate_sessions, SYSTEM_PROMPT


def test_session_and_turn_counts():
    turns = generate_sessions(n_sessions=5, turns_per_session=6, seed=1)
    assert len(turns) == 30
    assert len({t.session_id for t in turns}) == 5


def test_generation_is_deterministic_for_a_seed():
    assert generate_sessions(3, 4, seed=42) == generate_sessions(3, 4, seed=42)


def test_different_seeds_give_different_content():
    assert generate_sessions(3, 4, seed=1) != generate_sessions(3, 4, seed=2)


def test_history_grows_monotonically_within_a_session():
    turns = generate_sessions(1, 6, seed=7)
    lens = [len(t.prefix_text) for t in turns]
    assert lens == sorted(lens)
    assert lens[0] < lens[-1]


def test_every_prefix_starts_with_the_stable_system_prompt():
    for t in generate_sessions(2, 3, seed=9):
        assert t.prefix_text.startswith(SYSTEM_PROMPT)
