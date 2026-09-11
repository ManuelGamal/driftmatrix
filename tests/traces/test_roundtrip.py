from driftmatrix.traces.roundtrip import in_emitted_order, round_trips, serialise_call
from driftmatrix.traces.toolcalls import parse_tool_call_ordered

# Serialises assistant tool calls the way Spark's template does: string values verbatim, others as JSON.
SPARK_LIKE = ("{% for m in messages %}{% for tc in (m.tool_calls or []) %}"
              "<tool_call>{{ tc.function.name }}{% for k, v in tc.function.arguments.items() %}"
              "<arg_key>{{ k }}</arg_key><arg_value>{{ v if v is string else v | tojson }}</arg_value>"
              "{% endfor %}</tool_call>{% endfor %}{% endfor %}")

# The call from G2's Spark ambiguous stratum, conversation a2026_024, verbatim.
SPARK_024 = ("<tool_call>format_report<arg_key>body</arg_key><arg_value>record 165: field=beta value=105"
             "</arg_value><arg_key>title</arg_key><arg_value>Record 165</arg_value></tool_call>")


def test_ordered_parse_keeps_emitted_order_and_returns_the_block():
    name, args, block = parse_tool_call_ordered("prefix " + SPARK_024 + " suffix", "spark_argkv")
    assert name == "format_report"
    assert list(args) == ["body", "title"]
    assert block == SPARK_024


def test_our_reading_of_the_g2_spark_call_round_trips():
    name, args, block = parse_tool_call_ordered(SPARK_024, "spark_argkv")
    assert round_trips(SPARK_LIKE, block, name, args)


def test_the_chat_endpoints_call_does_not_reproduce_the_model_text():
    # what /v1/chat/completions returned: its tool grammar forced title first, and the model wrote a different body
    grammar_forced = {"body": "field=beta value=105", "title": "Record 165"}
    assert not round_trips(SPARK_LIKE, SPARK_024, "format_report", grammar_forced)


def test_non_string_values_round_trip():
    out = "<tool_call>lookup_record<arg_key>record_id</arg_key><arg_value>7</arg_value></tool_call>"
    name, args, block = parse_tool_call_ordered(out, "spark_argkv")
    assert args == {"record_id": 7} and round_trips(SPARK_LIKE, block, name, args)


def test_argument_order_matters_and_in_emitted_order_restores_it():
    out = ("<tool_call>format_report<arg_key>title</arg_key><arg_value>T</arg_value>"
           "<arg_key>body</arg_key><arg_value>B</arg_value></tool_call>")
    name, args, block = parse_tool_call_ordered(out, "spark_argkv")
    canonical = dict(sorted(args.items()))                  # body first, as ToolCall stores it
    assert not round_trips(SPARK_LIKE, block, name, canonical)
    assert round_trips(SPARK_LIKE, block, name, in_emitted_order(canonical, list(args)))


def test_in_emitted_order_keeps_keys_missing_from_the_order():
    assert list(in_emitted_order({"a": 1, "b": 2, "c": 3}, ["c", "a"])) == ["c", "a", "b"]


def test_template_error_counts_as_no_round_trip():
    assert not round_trips("{{ raise_exception('no') }}", SPARK_024, "format_report", {})


def test_serialise_call_places_the_call_in_an_assistant_turn():
    assert serialise_call(SPARK_LIKE, "x", {"k": "v"}) == \
        "<tool_call>x<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"
