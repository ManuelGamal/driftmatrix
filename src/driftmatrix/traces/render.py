"""Offline chat-template rendering (spec 4.4).

llama-server's /apply-template accepts messages but not tools, so prompts that
carry tool definitions are rendered here, from the chat template embedded in
each GGUF, configured the way HuggingFace renders chat templates:

- trim_blocks and lstrip_blocks on, loop controls enabled
- a tojson filter that neither HTML-escapes nor sorts keys. jinja2's built-in
  tojson does both: it turns <, > and & into \\u003c-style escapes, and its
  default policy (json.dumps_kwargs = {"sort_keys": True}) reorders every
  object's keys. Either would silently change every rendered tool schema and
  make the prompt differ from the one the model was built for
- raise_exception available to templates

A rendered prompt is only trusted after scripts/g2_render_check.py has shown
this renderer reproduces the server's own rendering byte for byte.
"""
from __future__ import annotations

import json

from jinja2.exceptions import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from ..gguf import read_metadata


def _tojson(value, indent=None, ensure_ascii=False, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def _raise_exception(message):
    raise TemplateError(message)


def _environment():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    return env


_ENV = _environment()


def load_template(gguf_path) -> str:
    template = read_metadata(gguf_path)["kv"].get("tokenizer.chat_template")
    if not isinstance(template, str):
        raise ValueError(f"no tokenizer.chat_template in {gguf_path}")
    return template


def render(template: str, messages, tools=None, *, add_generation_prompt: bool = True,
           enable_thinking=False, **context) -> str:
    """Render a conversation.

    enable_thinking defaults to False (spec 4.4). Pass None to leave it
    undefined, which is how templates behave when a caller never sets it.
    """
    ctx = dict(context)
    ctx["messages"] = messages
    ctx["add_generation_prompt"] = add_generation_prompt
    if enable_thinking is not None:
        ctx["enable_thinking"] = enable_thinking
    if tools is not None:
        ctx["tools"] = tools
    return _ENV.from_string(template).render(**ctx)


def snap(k: int, ubatch: int) -> int:
    """Round a reuse boundary down to a multiple of n_ubatch.

    Costs k % ubatch re-evaluated prompt tokens - always fewer than ubatch.
    """
    if ubatch <= 0:
        raise ValueError(f"ubatch must be positive, got {ubatch}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    return k - k % ubatch


def _last_user_index(messages) -> int:
    idx = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not idx:
        raise ValueError("conversation has no user turn")
    if idx[-1] == 0:
        raise ValueError("conversation has no history before its final user turn")
    return idx[-1]


def natural_boundary(count_tokens, template, messages, tools=None, **context) -> int:
    """Token length of the conversation before its final user turn.

    This is where a serving stack would have cached state from the previous
    request. Callers must confirm it is a token prefix of the full render; use
    boundary_tokens, which does.
    """
    head = messages[:_last_user_index(messages)]
    return count_tokens(render(template, head, tools, add_generation_prompt=False,
                               **context))


def boundary_tokens(tokenize, template, messages, tools=None, **context):
    """Return (k, full_tokens, is_prefix).

    is_prefix is False when the history renders differently on its own than
    inside the full conversation (some templates change a message's rendering
    depending on what follows it), or when tokenization merges across the
    boundary. Such turns are excluded as boundary_not_prefix.
    """
    full = list(tokenize(render(template, messages, tools, add_generation_prompt=True,
                                **context)))
    head = list(tokenize(render(template, messages[:_last_user_index(messages)], tools,
                                add_generation_prompt=False, **context)))
    k = len(head)
    return k, full, 0 < k < len(full) and full[:k] == head
