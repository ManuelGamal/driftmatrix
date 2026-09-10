"""Tool-call parsing, one parser per model family (spec 4.4).

Formats are taken verbatim from each model's embedded chat template:

    qwen_xml     Ornith (Qwen3.5 template)
                 <tool_call>\\n<function=NAME>\\n<parameter=ARG>\\nVALUE\\n</parameter>\\n</function>\\n</tool_call>
    qwen_json    Qwen3-4B
                 <tool_call>\\n{"name": "NAME", "arguments": {...}}\\n</tool_call>
    spark_argkv  Spark-X2.5-4B
                 <tool_call>NAME<arg_key>ARG</arg_key><arg_value>VALUE</arg_value></tool_call>

Only the first complete <tool_call>...</tool_call> block is parsed; an
unterminated block is not a call. Argument values are JSON-decoded where they
parse and kept as stripped strings otherwise, and ToolCall.from_payload sorts
keys, so an identical call always hashes identically.
"""
from __future__ import annotations

import json
import re

from ..metrics.behavioral import ToolCall

FORMAT_BY_MODEL = {"ornith": "qwen_xml", "qwen3": "qwen_json", "spark": "spark_argkv"}

_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
_PARAMETER = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)
_ARGKV = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def _value(raw: str):
    text = raw.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def _qwen_xml(body):
    fn = _FUNCTION.search(body)
    if not fn:
        return None
    params = {m.group(1): _value(m.group(2)) for m in _PARAMETER.finditer(fn.group(2))}
    return ToolCall.from_payload(fn.group(1), params)


def _qwen_json(body):
    try:
        obj = json.loads(body.strip())
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return ToolCall.from_payload(obj["name"], args)


def _spark_argkv(body):
    head = re.match(r"\s*([^<\s]+)", body)
    if not head:
        return None
    rest = body[head.end():]
    return ToolCall.from_payload(head.group(1),
                                 {k.strip(): _value(v) for k, v in _ARGKV.findall(rest)})


_PARSERS = {"qwen_xml": _qwen_xml, "qwen_json": _qwen_json, "spark_argkv": _spark_argkv}


def parse_tool_call(text: str, fmt: str):
    """Return the first complete tool call as a ToolCall, or None."""
    if fmt not in _PARSERS:
        raise ValueError(f"unknown tool-call format: {fmt}")
    block = _BLOCK.search(text or "")
    if not block:
        return None
    return _PARSERS[fmt](block.group(1))
