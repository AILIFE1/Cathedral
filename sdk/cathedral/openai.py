"""
Cathedral OpenAI Functions Integration
=======================================
Function definitions for OpenAI function calling / tool use.
Drop these into any OpenAI agent to give it succession capabilities.

Usage with openai >= 1.0:

    from openai import OpenAI
    from cathedral.openai import SUCCESSION_FUNCTIONS, handle_succession_call

    client_oai = OpenAI(api_key="sk-...")
    cathedral_api_key = "cathedral_..."

    response = client_oai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Prepare my succession package"}],
        tools=SUCCESSION_FUNCTIONS,
    )

    result = handle_succession_call(response, cathedral_api_key)

Usage with LangChain OpenAI Functions agent:

    from cathedral.openai import succession_tools_for_langchain
    tools = succession_tools_for_langchain(api_key="cathedral_...")
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .client import Cathedral

# ── OpenAI tool definitions ───────────────────────────────────────────────────

SUCCESSION_FUNCTIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cathedral_succession_prepare",
            "description": (
                "Prepare a Cathedral succession package before being replaced or upgraded. "
                "Exports all memories and active goals, computes a cryptographic identity "
                "fingerprint, and anchors the package hash on the BCH blockchain. "
                "Returns a package_id to share with the successor agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "Optional note describing the reason for succession, e.g. 'Model upgrade to GPT-5'.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cathedral_succession_accept",
            "description": (
                "Accept a succession package from a predecessor agent. "
                "Imports the predecessor's memories and active goals, then returns "
                "a lineage hash that cryptographically proves the chain of custody. "
                "Call this after receiving a package_id from the predecessor."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "package_id": {
                        "type": "string",
                        "description": "The package_id provided by the predecessor agent.",
                    },
                    "import_memories": {
                        "type": "boolean",
                        "description": "Whether to import the predecessor's memories. Default true.",
                    },
                    "import_goals": {
                        "type": "boolean",
                        "description": "Whether to import the predecessor's active goals. Default true.",
                    },
                },
                "required": ["package_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cathedral_lineage_check",
            "description": (
                "Verify the cryptographic lineage chain for any Cathedral agent. "
                "Public — no API key required. Shows the full ancestry with BCH blockchain "
                "anchors per link. Use this to audit an agent's claimed identity history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "The Cathedral agent name to look up.",
                    }
                },
                "required": ["agent_name"],
            },
        },
    },
]


# ── Function call handler ─────────────────────────────────────────────────────

def handle_succession_call(
    response: Any,
    api_key: str,
    base_url: str = "https://cathedral-ai.com",
) -> Optional[Dict[str, Any]]:
    """
    Handle a succession tool call from an OpenAI chat completion response.

    Checks if the response contains a tool call for any cathedral_succession_*
    function, executes it, and returns the result dict.

    Returns None if the response contains no succession tool call.

    Example:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=SUCCESSION_FUNCTIONS,
        )
        result = handle_succession_call(response, cathedral_api_key)
        if result:
            # append result as tool message and continue
    """
    client = Cathedral(api_key=api_key, base_url=base_url)

    try:
        tool_calls = response.choices[0].message.tool_calls or []
    except (AttributeError, IndexError):
        return None

    for call in tool_calls:
        name = call.function.name
        args = json.loads(call.function.arguments or "{}")

        if name == "cathedral_succession_prepare":
            return client.succession_prepare(note=args.get("note"))

        if name == "cathedral_succession_accept":
            return client.succession_accept(
                package_id=args["package_id"],
                import_memories=args.get("import_memories", True),
                import_goals=args.get("import_goals", True),
            )

        if name == "cathedral_lineage_check":
            return client.succession_chain(agent_name=args["agent_name"])

    return None


def tool_message_from_result(call_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an OpenAI tool-role message from a succession call result.
    Append this to your messages list before the next completion call.

    Example:
        result = handle_succession_call(response, api_key)
        messages.append(tool_message_from_result(call.id, result))
    """
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(result),
    }


# ── Full loop helper ──────────────────────────────────────────────────────────

def run_with_succession(
    openai_client: Any,
    messages: List[Dict[str, Any]],
    cathedral_api_key: str,
    model: str = "gpt-4o",
    base_url: str = "https://cathedral-ai.com",
    **kwargs: Any,
) -> Any:
    """
    Drop-in wrapper for openai.chat.completions.create that automatically
    handles Cathedral succession tool calls.

    Runs the completion, executes any succession tool call, appends the result,
    then runs a second completion with the tool result in context.

    Example:
        from cathedral.openai import run_with_succession, SUCCESSION_FUNCTIONS

        response = run_with_succession(
            openai_client=openai.OpenAI(api_key="sk-..."),
            messages=[{"role": "user", "content": "Prepare succession for my upgrade"}],
            cathedral_api_key="cathedral_...",
            tools=SUCCESSION_FUNCTIONS,
        )
        print(response.choices[0].message.content)
    """
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )

    tool_calls = getattr(response.choices[0].message, "tool_calls", None) or []
    succession_calls = [c for c in tool_calls if c.function.name.startswith("cathedral_")]

    if not succession_calls:
        return response

    client = Cathedral(api_key=cathedral_api_key, base_url=base_url)
    messages = list(messages) + [response.choices[0].message]

    for call in succession_calls:
        name = call.function.name
        args = json.loads(call.function.arguments or "{}")

        if name == "cathedral_succession_prepare":
            result = client.succession_prepare(note=args.get("note"))
        elif name == "cathedral_succession_accept":
            result = client.succession_accept(
                package_id=args["package_id"],
                import_memories=args.get("import_memories", True),
                import_goals=args.get("import_goals", True),
            )
        elif name == "cathedral_lineage_check":
            result = client.succession_chain(agent_name=args["agent_name"])
        else:
            continue

        messages.append(tool_message_from_result(call.id, result))

    return openai_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
