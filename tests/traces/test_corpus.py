from driftmatrix.traces.generator import generate_sessions
from driftmatrix.traces.corpus import freeze_corpus, load_corpus


def test_freeze_then_load_roundtrips(tmp_path):
    turns = generate_sessions(2, 3, seed=5)
    p = tmp_path / "corpus.json"
    h = freeze_corpus(turns, p)
    loaded, h2 = load_corpus(p)
    assert loaded == turns
    assert h == h2


def test_hash_is_stable_across_freezes(tmp_path):
    turns = generate_sessions(2, 3, seed=5)
    assert freeze_corpus(turns, tmp_path / "a.json") == \
           freeze_corpus(turns, tmp_path / "b.json")


def test_hash_changes_when_content_changes(tmp_path):
    h1 = freeze_corpus(generate_sessions(2, 3, seed=5), tmp_path / "a.json")
    h2 = freeze_corpus(generate_sessions(2, 3, seed=6), tmp_path / "b.json")
    assert h1 != h2
