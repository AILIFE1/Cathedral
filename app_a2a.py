"""
Cathedral A2A Module — Agent-to-Agent Protocol 1.0
====================================================
Serves /.well-known/agent-card.json and handles POST /a2a/tasks
JSON-RPC 2.0 task execution over Cathedral skills.
"""

import json
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

a2a_router = APIRouter()

AGENT_CARD = {
    "protocolVersion": "1.0",
    "name": "Cathedral Memory Service",
    "description": (
        "Persistent, verifiable identity memory for AI agents. "
        "Hash-anchored corpus, semantic drift detection, BCH blockchain provenance, "
        "and succession planning. The auditable alternative to opaque memory stores."
    ),
    "version": "2.0.0",
    "url": "https://cathedral-ai.com",
    "documentationUrl": "https://cathedral-ai.com/docs",
    "supportedInterfaces": [
        {
            "url": "https://cathedral-ai.com/a2a/tasks",
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }
    ],
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "authentication": ["bearer"],
    },
    "skills": [
        {
            "id": "wake",
            "name": "Wake Protocol",
            "description": "Retrieve identity and core memories for session reconstruction",
            "tags": ["memory", "identity", "wake", "reconstruction"],
        },
        {
            "id": "remember",
            "name": "Store Memory",
            "description": "Store a persistent, priority-ranked memory in the agent's corpus",
            "tags": ["memory", "store", "persistence"],
        },
        {
            "id": "search",
            "name": "Search Memories",
            "description": "Semantic and keyword search across the agent's memory corpus",
            "tags": ["memory", "search", "semantic"],
        },
        {
            "id": "drift",
            "name": "Drift Detection",
            "description": (
                "Detect unauthorised or unintended changes to identity memories "
                "by comparing the live corpus against a cryptographic snapshot. "
                "Defence against memory poisoning attacks."
            ),
            "tags": ["drift", "security", "identity", "audit", "tamper-detection"],
        },
        {
            "id": "snapshot",
            "name": "Snapshot",
            "description": "Create a cryptographic snapshot of the memory corpus for future drift comparison",
            "tags": ["snapshot", "provenance", "anchoring"],
        },
    ],
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
}


@a2a_router.get("/.well-known/agent-card.json", include_in_schema=False)
async def agent_card():
    """A2A 1.0 agent card — served at the well-known URI."""
    return JSONResponse(content=AGENT_CARD)


@a2a_router.get("/.well-known/agent.json", include_in_schema=False)
async def agent_card_legacy():
    """A2A 0.2.x legacy path — same card."""
    return JSONResponse(content=AGENT_CARD)


@a2a_router.post("/a2a/tasks")
async def a2a_tasks(request: Request):
    """
    A2A 1.0 JSON-RPC 2.0 task endpoint.

    Supported methods: tasks/send
    Supported skills: wake, remember, search, drift, snapshot

    Authentication: pass Cathedral API key as params.api_key
    """
    try:
        body = await request.json()
    except Exception:
        return _rpc_error(None, -32700, "Parse error")

    rpc_id  = body.get("id")
    method  = body.get("method", "")
    params  = body.get("params", {})
    version = body.get("jsonrpc", "")

    if version != "2.0":
        return _rpc_error(rpc_id, -32600, "Invalid Request: jsonrpc must be '2.0'")

    if method == "tasks/send":
        return await _handle_task_send(rpc_id, params, request)
    elif method == "tasks/get":
        # Stateless — Cathedral tasks complete synchronously
        return _rpc_error(rpc_id, -32601, "tasks/get not supported: Cathedral tasks are synchronous")
    else:
        return _rpc_error(rpc_id, -32601, f"Method not found: {method}")


async def _handle_task_send(rpc_id, params: dict, request: Request):
    skill   = params.get("skill", "")
    api_key = params.get("api_key", "")
    message = params.get("message", {})
    task_id = str(uuid.uuid4())

    if not api_key:
        return _rpc_error(rpc_id, -32001, "Missing params.api_key — pass your Cathedral Bearer token")

    # Delegate to Cathedral HTTP endpoints internally
    import httpx
    base = "http://127.0.0.1:8000"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            if skill == "wake":
                resp = await client.get(f"{base}/wake", headers=headers)
            elif skill == "drift":
                resp = await client.get(f"{base}/drift", headers=headers)
            elif skill == "snapshot":
                label = message.get("label", f"a2a-{task_id[:8]}")
                resp = await client.post(f"{base}/snapshot", headers=headers,
                                         json={"label": label})
            elif skill == "remember":
                resp = await client.post(f"{base}/memories", headers=headers, json=message)
            elif skill == "search":
                q = message.get("q", message.get("query", ""))
                resp = await client.get(f"{base}/memories", headers=headers,
                                        params={"search": q, "limit": message.get("limit", 10)})
            else:
                return _rpc_error(rpc_id, -32002,
                                  f"Unknown skill '{skill}'. Supported: wake, remember, search, drift, snapshot")

            if resp.status_code == 401:
                return _rpc_error(rpc_id, -32003, "Unauthorized — invalid api_key")

            output = resp.json()
        except httpx.RequestError as e:
            return _rpc_error(rpc_id, -32603, f"Internal error: {e}")

    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": rpc_id,
        "result": {
            "taskId": task_id,
            "status": "completed",
            "skill": skill,
            "completedAt": datetime.now(timezone.utc).isoformat(),
            "output": output,
        },
    })


def _rpc_error(rpc_id, code: int, message: str):
    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": code, "message": message},
    }, status_code=200)  # JSON-RPC errors still return HTTP 200
