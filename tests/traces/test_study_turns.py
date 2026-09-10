from collections import Counter

import pytest

from driftmatrix.config import ROUND_TRIPS, STUDY_BASES, STUDY_SEED
from driftmatrix.traces.conversations import (
    AMBIGUOUS_TYPES, SYSTEM, generate_conversations, generate_study_turns, trace_hash,
)
from driftmatrix.traces.tools import call_tool

RT = {"short": 2, "long": 5}


def cells(turns):
    return Counter((t.prefix_band, t.stratum) for t in turns)


def test_every_base_appears_in_every_band_and_stratum():
    turns = generate_study_turns(10, seed=1, round_trips=RT)
    assert len(turns) == 40
    assert cells(turns) == {(b, s): 10 for b in ("short", "long") for s in ("explicit", "ambiguous")}
    assert all(c == 4 for c in Counter(t.base_id for t in turns).values())
    assert len({t.conv_id for t in turns}) == 40


def test_history_lengths_follow_the_bands():
    for t in generate_study_turns(3, seed=1, round_trips=RT):
        assert t.messages[0] == {"role": "system", "content": SYSTEM}
        assert len(t.messages) == 1 + 4 * RT[t.prefix_band] + 1


def test_short_band_is_the_most_recent_history_of_the_long_band():
    turns = generate_study_turns(4, seed=2, round_trips=RT)
    by = {(t.base_id, t.prefix_band, t.stratum): t for t in turns}
    for base in {t.base_id for t in turns}:
        for stratum in ("explicit", "ambiguous"):
            short, long_ = by[base, "short", stratum], by[base, "long", stratum]
            assert short.messages[1:] == long_.messages[-(4 * RT["short"] + 1):]


def test_ambiguous_twin_replaces_only_the_final_request():
    turns = generate_study_turns(5, seed=3, round_trips=RT)
    by = {(t.base_id, t.prefix_band, t.stratum): t for t in turns}
    for (base, band, stratum), t in by.items():
        if stratum == "ambiguous":
            e = by[base, band, "explicit"]
            assert t.messages[:-1] == e.messages[:-1]
            assert t.messages[-1]["content"] != e.messages[-1]["content"]
            assert t.expected_tool is None and e.expected_tool is not None


def test_explicit_expected_calls_are_valid():
    for t in generate_study_turns(20, seed=4, round_trips=RT):
        if t.stratum == "explicit":
            call_tool(t.expected_tool, t.expected_args)


def test_ambiguous_types_rotate_evenly_in_every_band():
    turns = generate_study_turns(STUDY_BASES, seed=STUDY_SEED, round_trips=RT)
    for band in ("short", "long"):
        requests = Counter(t.messages[-1]["content"].split()[0] + t.messages[-1]["content"].split()[1]
                           for t in turns if t.prefix_band == band and t.stratum == "ambiguous")
        assert sorted(requests.values()) == [STUDY_BASES // len(AMBIGUOUS_TYPES)] * len(AMBIGUOUS_TYPES)


def test_study_generation_is_deterministic_and_hashable():
    a = generate_study_turns(6, seed=5, round_trips=RT)
    assert a == generate_study_turns(6, seed=5, round_trips=RT)
    assert trace_hash(a) == trace_hash(generate_study_turns(6, seed=5, round_trips=RT))
    assert trace_hash(a) != trace_hash(generate_study_turns(6, seed=6, round_trips=RT))


def test_study_turns_do_not_disturb_the_g2_conversations():
    generate_study_turns(3, seed=2026, round_trips=RT)
    assert generate_conversations(1, 3, seed=2026)[0].messages[-1]["content"] == \
        "Look up record 153 and tell me its field."


def test_invalid_round_trips_raise():
    with pytest.raises(ValueError):
        generate_study_turns(1, seed=1, round_trips={"short": 6, "long": 5})
    with pytest.raises(ValueError):
        generate_study_turns(1, seed=1, round_trips={"short": 0, "long": 5})


def test_pre_registered_study_constants():
    assert STUDY_BASES == 90 and ROUND_TRIPS == {"short": 8, "long": 60}
    turns = generate_study_turns(STUDY_BASES, seed=STUDY_SEED, round_trips=ROUND_TRIPS)
    assert len(turns) == 360 and set(cells(turns).values()) == {90}
