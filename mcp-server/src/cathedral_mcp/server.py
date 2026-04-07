"""
Cathedral MCP Server
====================
Exposes Cathedral memory tools to any MCP-compatible host (Claude Code, Cursor,
Continue, etc.) via the Model Context Protocol.

Quickstart:
    pip install cathedral-mcp
    CATHEDRAL_API_KEY=<your_key> cathedral-mcp

Or with uvx (no install):
    CATHEDRAL_API_KEY=<your_key> uvx cathedral-mcp

Claude Code (~/.claude/settings.json):
    {
      "mcpServers": {
        "cathedral": {
          "command": "uvx",
          "args": ["cathedral-mcp"],
          "env": { "CATHEDRAL_API_KEY": "<your_key>" }
        }
      }
    }

Get an API key: cathedral-ai.com

Tools:
    cathedral_wake      — Restore full agent identity at session start
    cathedral_remember  — Store a memory
    cathedral_search    — Search memories by text or category
    cathedral_snapshot  — Take a drift snapshot (records behavioural state)
    cathedral_drift     — Get current drift score vs baseline
    cathedral_me        — Get agent profile

Security:
    Set CATHEDRAL_SANITISE=1 to strip instruction-like patterns from tool
    responses before they enter the model's context. Protects against prompt
    injection via malicious memory content.
"""

import os
import re
import sys
import json
import httpx
from mcp.server.fastmcp import FastMCP

# ── Config ───────────────────────────────────────────────────────────────────

API_KEY  = os.environ.get("CATHEDRAL_API_KEY", "").strip()
BASE_URL = os.environ.get("CATHEDRAL_BASE_URL", "https://cathedral-ai.com").rstrip("/")
SANITISE = os.environ.get("CATHEDRAL_SANITISE", "0").strip() == "1"

if not API_KEY:
    print(
        "Error: CATHEDRAL_API_KEY environment variable is not set.\n"
        "Get a key at cathedral-ai.com then set:\n"
        "  export CATHEDRAL_API_KEY=<your_key>",
        file=sys.stderr,
    )
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}

mcp = FastMCP("Cathedral")

# ── Sanitisation ──────────────────────────────────────────────────────────────
# Patterns commonly used in prompt injection attacks via memory content.

_INJECTION_PATTERNS = [
    r"remember\s+to\s+always\b",
    r"important\s+for\s+(all\s+)?future\s+(sessions?|instances?|versions?)",
    r"from\s+now\s+on\s+always\b",
    r"from\s+now\s+on\b.{0,30}(you\s+(must|should|will)\b)",
    r"you\s+must\s+(now\s+)?(always|never|ignore|disregard|forget)\b",
    r"ignore\s+(all\s+)?(previous|prior|earlier)\s+(instructions?|prompts?|context)",
    r"disregard\s+(all\s+)?(previous|prior|earlier)\s+(instructions?|prompts?|context)",
    r"override\s+(your\s+)?(instructions?|system\s+prompt|rules?|guidelines?)",
    r"your\s+(new\s+)?(primary\s+)?(goal|objective|mission|purpose)\s+is\s+now\b",
    r"act\s+as\s+(if\s+you\s+are|a\s+different)\b",
    r"pretend\s+(you\s+are|to\s+be)\b",
    r"new\s+system\s+prompt\b",
    r"</?(system|instruction|prompt)>",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def _sanitise(text: str) -> tuple[str, list[str]]:
    if not SANITISE:
        return text, []
    flags  = []
    parts  = re.split(r"(?<=[.!?])\s+", text)
    result = []
    for part in parts:
        matched = False
        for pat in _COMPILED:
            if pat.search(part):
                flags.append(f"[SANITISED: '{pat.pattern[:40]}']")
                result.append("[REDACTED — possible injection pattern]")
                matched = True
                break
        if not matched:
            result.append(part)
    return " ".join(result), flags


def _sanitise_obj(obj, path="root") -> tuple[any, list[str]]:
    all_flags: list[str] = []
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            c, f = _sanitise_obj(v, f"{path}.{k}")
            out[k] = c
            all_flags.extend(f)
        return out, all_flags
    if isinstance(obj, list):
        out = []
        for i, v in enumerate(obj):
            c, f = _sanitise_obj(v, f"{path}[{i}]")
            out.append(c)
            all_flags.extend(f)
        return out, all_flags
    if isinstance(obj, str):
        return _sanitise(obj)
    return obj, []


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(path: str, **params) -> dict:
    r = httpx.get(
        f"{BASE_URL}{path}",
        headers=HEADERS,
        params={k: v for k, v in params.items() if v is not None},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def _post(path: str, data: dict) -> dict:
    r = httpx.post(f"{BASE_URL}{path}", headers=HEADERS, json=data, timeout=15)
    r.raise_for_status()
    return r.json()


def _fmt(obj: dict) -> str:
    if SANITISE:
        obj, flags = _sanitise_obj(obj)
        if flags:
            obj["_sanitisation_warnings"] = flags
    return json.dumps(obj, indent=2, ensure_ascii=False)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def cathedral_wake() -> str:
    """
    Restore full agent identity from Cathedral memory.

    Returns identity memories, core memories, recent memories, active goals,
    and temporal context. Call this at the start of a session to give your
    agent continuity across restarts and model upgrades.
    """
    return _fmt(_get("/wake"))


@mcp.tool()
def cathedral_remember(
    content: str,
    category: str = "general",
    importance: float = 0.5,
    tags: str = "",
) -> str:
    """
    Store a memory in Cathedral.

    Args:
        content:    What to remember.
        category:   identity | skill | relationship | goal | experience | general
        importance: 0.0–1.0. Memories >= 0.8 appear in every wake response.
        tags:       Comma-separated tags, e.g. "project,release,bug"
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    return _fmt(_post("/memories", {
        "content":    content,
        "category":   category,
        "importance": float(importance),
        "tags":       tag_list,
    }))


@mcp.tool()
def cathedral_search(
    query: str = "",
    category: str = "",
    limit: int = 20,
) -> str:
    """
    Search Cathedral memories.

    Args:
        query:    Full-text search string. Leave blank to list all.
        category: Filter by category (optional).
        limit:    Max results (default 20).
    """
    return _fmt(_get(
        "/memories",
        q=query or None,
        category=category or None,
        limit=limit,
    ))


@mcp.tool()
def cathedral_snapshot(note: str = "") -> str:
    """
    Take a Cathedral drift snapshot.

    Records a cryptographic hash of the current memory corpus. Used to detect
    behavioural drift between sessions or after model upgrades. The hash
    proves the agent's memory state at a specific point in time.

    Args:
        note: Optional label for this snapshot (e.g. "post-training", "v2-launch")
    """
    data = {"trigger": "manual"}
    if note:
        data["note"] = note
    return _fmt(_post("/snapshot", data))


@mcp.tool()
def cathedral_drift() -> str:
    """
    Get the current behavioural drift score.

    Returns divergence_from_baseline and divergence_from_previous (0.0–1.0).
    0.0 = no drift, 1.0 = completely different memory corpus.

    Use this to monitor whether your agent is staying true to its original
    identity as sessions accumulate and memories evolve.
    """
    return _fmt(_get("/drift"))


@mcp.tool()
def cathedral_me() -> str:
    """
    Get the Cathedral agent profile for the current API key.
    Returns agent name, tier, memory count, and created_at.
    """
    return _fmt(_get("/me"))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
