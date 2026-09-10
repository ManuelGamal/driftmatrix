import pytest
from jinja2.exceptions import TemplateError

from driftmatrix.traces.render import boundary_tokens, natural_boundary, render, snap

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

CONVERSATION = [
    {"role": "system", "content": "You are an assistant."},
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "second question"},
]


def whitespace_tokens(text):
    return text.split()


def test_tojson_does_not_html_escape():
    assert render("{{ x | tojson }}", [], None, add_generation_prompt=False, x="<a&b>") == '"<a&b>"'


def test_tojson_preserves_key_order():
    # jinja2's default tojson sorts keys (policy json.dumps_kwargs = {"sort_keys": True}),
    # which would silently reorder every rendered tool schema.
    assert render("{{ x | tojson }}", [], None, add_generation_prompt=False,
                  x={"b": 1, "a": 2}) == '{"b": 1, "a": 2}'


def test_tojson_keeps_non_ascii_and_accepts_indent():
    out = render("{{ x | tojson(indent=2) }}", [], None, add_generation_prompt=False,
                 x={"k": "é"})
    assert '"é"' in out and "\n  " in out


def test_trim_blocks_like_huggingface():
    assert render("{% if true %}\nA{% endif %}", [], None, add_generation_prompt=False) == "A"


def test_lstrip_blocks_like_huggingface():
    assert render("  {% if true %}A{% endif %}", [], None, add_generation_prompt=False) == "A"


def test_raise_exception_surfaces_template_errors():
    with pytest.raises(TemplateError, match="boom"):
        render("{{ raise_exception('boom') }}", [], None, add_generation_prompt=False)


def test_enable_thinking_defaults_false_and_can_be_left_undefined():
    tpl = "{% if enable_thinking is defined %}{{ enable_thinking }}{% else %}undefined{% endif %}"
    assert render(tpl, [], None, add_generation_prompt=False) == "False"
    assert render(tpl, [], None, add_generation_prompt=False, enable_thinking=None) == "undefined"


def test_tools_are_passed_only_when_given():
    tpl = "{% if tools %}{{ tools | length }}{% else %}none{% endif %}"
    assert render(tpl, [], None, add_generation_prompt=False) == "none"
    assert render(tpl, [], [{"name": "a"}, {"name": "b"}], add_generation_prompt=False) == "2"


def test_snap_rounds_down_and_costs_less_than_one_ubatch():
    assert snap(1390, 512) == 1024 and snap(1024, 512) == 1024 and snap(511, 512) == 0
    assert all(0 <= k - snap(k, 512) <= 511 for k in range(0, 5000, 37))


def test_snap_rejects_bad_arguments():
    with pytest.raises(ValueError):
        snap(10, 0)
    with pytest.raises(ValueError):
        snap(-1, 512)


def test_natural_boundary_is_a_token_prefix_of_the_full_render():
    k, full, is_prefix = boundary_tokens(whitespace_tokens, CHATML, CONVERSATION)
    assert is_prefix is True
    assert k == natural_boundary(lambda t: len(whitespace_tokens(t)), CHATML, CONVERSATION)
    assert 0 < k < len(full)


def test_history_that_renders_differently_when_last_is_not_a_prefix():
    tpl = CHATML.replace("{{ m.content }}", "{{ m.content }}{% if loop.last %} LAST{% endif %}")
    _, _, is_prefix = boundary_tokens(whitespace_tokens, tpl, CONVERSATION)
    assert is_prefix is False


def test_conversation_without_history_is_rejected():
    with pytest.raises(ValueError, match="no history"):
        natural_boundary(len, CHATML, [{"role": "user", "content": "only"}])
