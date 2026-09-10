from driftmatrix.traces.render import boundary_tokens, common_prefix_length, reuse_boundary

# The last message gets a marker only while it is last, the way Qwen3's template adds an
# empty think block to an assistant message that follows the final user turn.
UNSTABLE = ("{% for m in messages %}<{{ m.role }}>{% if m.role == 'assistant' and loop.last %}[think]"
            "{% endif %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}")
STABLE = UNSTABLE.replace("{% if m.role == 'assistant' and loop.last %}[think]{% endif %}", "")

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"}, {"role": "user", "content": "v"}]


def chars(text):
    return list(text)


def test_common_prefix_length():
    assert common_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix_length([1, 2], [1, 2, 3]) == 2
    assert common_prefix_length([], [1]) == 0


def test_prefix_stable_template_reuses_the_whole_history():
    k, full = reuse_boundary(chars, STABLE, MESSAGES)
    assert k == len("<system>s<user>u<assistant>a")
    assert "".join(full) == "<system>s<user>u<assistant>a<user>v<assistant>"


def test_unstable_template_reuses_only_the_common_prefix():
    k, full = reuse_boundary(chars, UNSTABLE, MESSAGES)
    assert k == len("<system>s<user>u<assistant>")
    assert "".join(full[:k]) == "<system>s<user>u<assistant>"


def test_boundary_tokens_would_have_excluded_the_unstable_template():
    assert boundary_tokens(chars, UNSTABLE, MESSAGES)[2] is False
    assert boundary_tokens(chars, STABLE, MESSAGES)[2] is True
