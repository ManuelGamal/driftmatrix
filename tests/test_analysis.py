"""Pre-registered analysis against synthetic runner records with known answers."""
import pytest

from driftmatrix.analysis import (
    L1, L2, L3, ModelData, action_of, kaplan_meier, run_analysis, turn_table,
)

# Serialises assistant tool calls exactly as the synthetic model emits them (qwen_json).
TEMPLATE = ('{% for m in messages %}{% for tc in (m.tool_calls or []) %}<tool_call>\n{"name": "'
            '{{ tc.function.name }}", "arguments": {{ tc.function.arguments | tojson }}}\n</tool_call>'
            '{% endfor %}{% endfor %}')


def call(record_id):
    return f'<tool_call>\n{{"name": "lookup_record", "arguments": {{"record_id": {record_id}}}}}\n</tool_call>'


def top(tokens, first_lp=-0.05, second_lp=-3.0):
    return [[[t, first_lp], [t + 1, second_lp]] for t in tokens]


def arm(content, tokens, tops, outcome="ok"):
    return {"outcome": outcome, "content": content, "tokens": tokens, "top": tops, "pieces": [content]}


def dataset(model, n_bases=10, change=None, drift=None, floor_noise=False, invalid=(),
            unfaithful=(), snapped_alias=False):
    """change(conv_id, boundary, rep) -> bool; drift(conv_id, boundary) -> logprob shift."""
    change = change or (lambda *_: False)
    drift = drift or (lambda *_: 0.0)
    turns, arms = {}, {}
    for i in range(n_bases):
        for band in ("short", "long"):
            for stratum in ("explicit", "ambiguous"):
                cid = f"{stratum[0]}{band[0]}_{i:02d}"
                turns[cid] = {"conv_id": cid, "base_id": f"b{i}", "prefix_band": band, "stratum": stratum,
                              "k_natural": 1000 + i, "k_snapped": 512 if not snapped_alias else 1000 + i,
                              "k_mod_ubatch": 488 - i, "snapped_is_natural": snapped_alias,
                              "state_bytes": {"natural": {"equal": False}, "snapped": {"equal": True}}}
                content = call(i) if cid not in unfaithful else call(i).replace('": ', '":')
                base = [10 + i, 20, 30]
                for rep in (1, 2):
                    noise = -0.01 if floor_noise and rep == 2 else 0.0
                    arms[(cid, "recompute", None, rep)] = arm(content, base, top(base, -0.05 + noise))
                    boundaries = ("natural",) if snapped_alias else ("natural", "snapped")
                    for b in boundaries:
                        outcome = "restore_not_used" if (cid, b) in invalid else "ok"
                        if change(cid, b, rep):
                            other = [10 + i, 99, 98]
                            arms[(cid, "restore", b, rep)] = arm(call(i + 100), other, top(other), outcome)
                        else:
                            shift = drift(cid, b)
                            arms[(cid, "restore", b, rep)] = arm(content, base, top(base, -0.05 - shift, -3.0 + shift),
                                                                  outcome)
    return ModelData(model, turns, arms, TEMPLATE, "qwen_json")


def analyse(*datasets):
    return run_analysis(list(datasets))


def three(l3=None, l2=None, l1=None):
    return analyse(l3 or dataset(L3), l2 or dataset(L2), l1 or dataset(L1))


def row(result, model, conv_id, boundary):
    return next(r for r in result["rows"] if r["model"] == model and r["conv_id"] == conv_id
                and r["boundary"] == boundary)


def test_action_of_reads_calls_and_no_calls():
    assert action_of({"content": "I need more detail."}, TEMPLATE, "qwen_json") == (None, True)
    action, faithful = action_of({"content": call(7)}, TEMPLATE, "qwen_json")
    assert action[0] == "lookup_record" and faithful


def test_family_sizes_with_all_three_models():
    r = three()
    assert (len(r["families"]["A"]), len(r["families"]["B"]), len(r["families"]["C"])) == (24, 16, 6)


def test_identical_arms_claim_nothing():
    r = three()
    assert not any(v["claimed"] for fam in r["families"].values() for v in fam.values())
    assert r["hypotheses"]["H1_mitigation"][L3]["holds"] is True
    assert r["hypotheses"]["H3_behavioural_existence"] == "not supported"


def test_action_change_counts_only_when_both_replicates_change():
    r = analyse(dataset(L3, change=lambda c, b, rep: c == "es_00" and b == "natural" and rep == 1))
    x = row(r, L3, "es_00", "natural")
    assert x["action_change"] == 0 and x["action_change_any"] == 1
    assert r["secondary"][L3]["nonreproducible_action_changes_natural"] == 1


def test_one_reproducible_change_against_a_zero_floor_is_claimed():
    r = analyse(dataset(L3, change=lambda c, b, rep: c == "es_03" and b == "natural"))
    m = r["families"]["A"][f"A:{L3}:natural:short:explicit"]
    assert m["method"] == "zero_floor" and m["p_value"] == 0.0 and m["events"] == 1
    assert m["interval"]["method"] == "clopper_pearson" and m["interval"]["low"] > 0
    assert m["claimed"] is True
    assert not r["families"]["A"][f"A:{L3}:snapped:short:explicit"]["claimed"]
    assert r["hypotheses"]["H3_behavioural_existence"] == "partial"


def test_an_inexact_floor_switches_to_the_sign_test():
    r = analyse(dataset(L3, floor_noise=True, change=lambda c, b, rep: c == "es_03" and b == "natural"))
    m = r["families"]["A"][f"A:{L3}:natural:short:explicit"]
    assert m["method"] == "sign_test" and m["claimed"] is False         # one turn cannot reject
    assert r["determinism"][L3]["nondeterministic"] is True
    assert m["conditional"] is True


def test_unfaithful_parses_are_excluded_and_too_many_exclusions_block_a_claim():
    bad = {f"es_{i:02d}" for i in range(2)}                              # 2 of 10 turns = 20%
    r = analyse(dataset(L3, unfaithful=bad, change=lambda c, b, rep: c == "es_05" and b == "natural"))
    assert row(r, L3, "es_00", "natural")["excluded"] == "parse_unfaithful"
    m = r["families"]["A"][f"A:{L3}:natural:short:explicit"]
    assert m["excluded"] == 2 and m["too_many_excluded"] and m["holm_reject"] and not m["claimed"]


def test_invalid_arms_are_excluded_and_counted():
    r = analyse(dataset(L3, invalid={("el_01", "natural")}))
    assert row(r, L3, "el_01", "natural")["excluded"] == "invalid_arm"
    assert r["secondary"][L3]["exclusions"]["natural:long:explicit"] == {"invalid_arm": 1}


def test_drift_without_action_change_is_family_b_not_a():
    r = analyse(dataset(L3, drift=lambda c, b: 0.3 if b == "natural" else 0.0))
    b = r["families"]["B"]
    assert all(v["claimed"] for k, v in b.items() if ":natural:" in k)
    assert not any(v["claimed"] for k, v in b.items() if ":snapped:" in k)
    assert not any(v["claimed"] for v in r["families"]["A"].values())
    assert r["hypotheses"]["H2_distributional_existence"] == "supported"
    assert r["secondary"][L3]["conditions"]["natural:short:explicit"]["self_inconsistent"] is True


def test_mechanism_contrasts_follow_the_predicted_pattern():
    l3 = dataset(L3, drift=lambda c, b: (0.4 if "l_" in c[:3] else 0.2) if b == "natural" else 0.0,
                 change=lambda c, b, rep: c.startswith("a") and b == "natural")
    l2 = dataset(L2, drift=lambda c, b: 0.01 if b == "natural" else 0.0)
    l1 = dataset(L1, drift=lambda c, b: 0.01 if b == "natural" else 0.0)
    c = three(l3, l2, l1)["families"]["C"]
    for key in c:
        assert c[key]["claimed"], key
    h = three(l3, l2, l1)["hypotheses"]
    assert (h["H4_architecture"], h["H5_snapping"], h["H6_compounding"], h["H7_stratum"]) == ("supported",) * 4


def test_a_significant_effect_against_the_prediction_is_flagged():
    l1 = dataset(L1, drift=lambda c, b: 0.3 if b == "natural" else 0.0)
    c = three(l1=l1)["families"]["C"]["C3_L3_minus_L1"]
    assert c["contradicts_prediction"] is True and c["claimed"] is False


def test_aligned_turns_reuse_the_natural_run_for_snapped():
    r = analyse(dataset(L3, snapped_alias=True, change=lambda c, b, rep: c == "es_00"))
    assert row(r, L3, "es_00", "snapped")["action_change"] == 1
    assert row(r, L3, "es_00", "natural")["action_change"] == 1
    assert r["secondary"][L3]["alignment"]["aligned"]["n"] == 0          # k_mod_ubatch is never 0 here


def test_h1_fails_on_an_inexact_snapped_turn():
    r = analyse(dataset(L3, drift=lambda c, b: 0.2 if (c, b) == ("al_04", "snapped") else 0.0))
    h1 = r["hypotheses"]["H1_mitigation"][L3]
    assert h1["holds"] is False and h1["inexact"] == [{"conv_id": "al_04", "k": 512}]


def test_margin_bins_and_state_bytes_are_tabulated():
    r = analyse(dataset(L3))
    bins = r["secondary"][L3]["margin_bins"]["natural"]
    assert bins["confident"]["n"] == 40 and bins["near_tie"]["n"] == 0
    assert r["secondary"][L3]["state_bytes_natural"] == {"equal": 0, "differ": 40, "unavailable": 0}
    assert r["secondary"][L3]["snapping_cost_tokens"]["short"] == {"min": 479, "median": 483.5, "max": 488}


def test_exploratory_contrasts_are_reported_per_band():
    l3 = dataset(L3, drift=lambda c, b: 0.3 if b == "natural" else 0.0)
    e = three(l3)["exploratory"]
    assert set(e) == {"short", "long"} and e["long"]["L3_minus_L1"]["n"] == 20


def test_kaplan_meier_with_censoring():
    curve = kaplan_meier([(1, True), (3, False), (3, True), (5, False)])
    assert curve == [(1, 0.75), (3, pytest.approx(0.75 * (1 - 1 / 3)))]


def test_turn_table_rows_cover_every_turn_and_boundary():
    rows = turn_table(dataset(L3, n_bases=3))
    assert len(rows) == 3 * 4 * 2


def test_analysis_consumes_the_runners_own_records(tmp_path):
    from test_servergrid import ChatTilingServer, run          # fake server where tiling is the mechanism

    from driftmatrix.servergrid import load_records

    out = tmp_path / f"{L3}.jsonl"
    run(ChatTilingServer(), out, model=L3)
    _, turns, arms = load_records(out)
    r = run_analysis([ModelData(L3, turns, arms, TEMPLATE, "qwen_json")])
    natural = [x for x in r["rows"] if x["boundary"] == "natural"]
    snapped = [x for x in r["rows"] if x["boundary"] == "snapped"]
    assert natural and all(x["excluded"] is None for x in r["rows"])
    assert all(not x["exact"] and x["first_divergence"] == 1 for x in natural)   # the fake flips token 1
    assert all(x["exact_both"] and x["mean_jsd"] == 0.0 for x in snapped)
    assert all(x["action_change"] == 0 for x in r["rows"])                      # no tool calls either way
    assert r["hypotheses"]["H1_mitigation"][L3]["holds"] is True
    assert all(v["claimed"] for k, v in r["families"]["B"].items() if ":natural:" in k)
