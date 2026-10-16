"""Tool registry with native function calling. Schemas are generated from Python signatures."""

import inspect
import json
import logging
import re
from typing import Any, Callable

log = logging.getLogger("glucotwin.tools")

TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _param_docs(doc: str) -> dict[str, str]:
    """Pull `name: description` lines from an `Args:` section of a docstring."""
    match = re.search(r"Args:\s*\n(.*?)(?:\n\s*\n|$)", doc, re.DOTALL)
    if not match:
        return {}
    docs = {}
    for line in match.group(1).splitlines():
        m = re.match(r"\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", line)
        if m:
            docs[m.group(1)] = m.group(2).strip()
    return docs


def tool_to_schema(function: Callable) -> dict:
    """Convert a Python function into an OpenAI-style tool schema."""
    doc = inspect.getdoc(function) or ""
    param_docs = _param_docs(doc)
    summary = doc.split("\n\n")[0].strip()

    properties, required = {}, []
    for name, parameter in inspect.signature(function).parameters.items():
        prop = {"type": TYPE_MAP.get(parameter.annotation, "string")}
        if prop["type"] == "array":
            prop["items"] = {"type": "string"}
        if name in param_docs:
            prop["description"] = param_docs[name]
        properties[name] = prop
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    schema = {"type": "function", "function": {"name": function.__name__, "description": summary}}
    # Some providers reject OBJECT schemas with empty properties, so omit them for zero-arg tools.
    if properties:
        schema["function"]["parameters"] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    return schema


class NativeTools:
    """Registry of tools the agent may call."""

    def __init__(self):
        self.registry: dict[str, Callable] = {}

    def add_tool(self, func: Callable, name: str | None = None) -> None:
        self.registry[name or func.__name__] = func

    @property
    def schemas(self) -> list[dict]:
        return [tool_to_schema(f) for f in self.registry.values()]

    @staticmethod
    def parse(tool_call: dict) -> tuple[str, dict]:
        """Extract (name, kwargs) from a raw tool_call dict."""
        fn = tool_call["function"]
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        return fn["name"], args

    def execute(self, name: str, kwargs: dict) -> str:
        """Run a tool. Errors are returned as text so the model can see and recover from them."""
        func = self.registry.get(name)
        if func is None:
            return f"Error: tool '{name}' not found. Available: {', '.join(self.registry)}"
        try:
            result: Any = func(**kwargs)
        except Exception as e:  # noqa: BLE001 - surface any tool failure to the model
            log.exception("Tool %s failed", name)
            return f"Error running {name}: {type(e).__name__}: {e}"
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, default=str)
        return str(result)
