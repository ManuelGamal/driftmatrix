"""Deterministic mock tools (spec 4.3).

Every tool MUST be a pure function of its arguments. Any nondeterminism here
is indistinguishable from the effect under study and invalidates the run.
"""
from __future__ import annotations


class ToolError(Exception):
    pass


TOOL_SCHEMAS = [
    {"name": "lookup_record",
     "description": "Fetch a record by integer id.",
     "parameters": {"type": "object",
                    "properties": {"record_id": {"type": "integer"}},
                    "required": ["record_id"]}},
    {"name": "compute_sum",
     "description": "Sum a list of numbers.",
     "parameters": {"type": "object",
                    "properties": {"values": {"type": "array",
                                              "items": {"type": "number"}}},
                    "required": ["values"]}},
    {"name": "format_report",
     "description": "Render a titled markdown report.",
     "parameters": {"type": "object",
                    "properties": {"title": {"type": "string"},
                                   "body": {"type": "string"}},
                    "required": ["title", "body"]}},
]

_FIELDS = ("alpha", "beta", "gamma", "delta")


def _require(args, key):
    if key not in args:
        raise ToolError(f"missing argument: {key}")
    return args[key]


def call_tool(name: str, args: dict) -> str:
    if name == "lookup_record":
        rid = int(_require(args, "record_id"))
        return f"record {rid}: field={_FIELDS[rid % len(_FIELDS)]} value={rid * 37 % 1000}"
    if name == "compute_sum":
        total = sum(_require(args, "values"))
        return str(int(total)) if float(total).is_integer() else str(total)
    if name == "format_report":
        return f"# {_require(args, 'title')}\n\n{_require(args, 'body')}"
    raise ToolError(f"unknown tool: {name}")
