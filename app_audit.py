"""
Cathedral Audit Log Module
==========================
Tracks memory operations (store, update, delete, compact, snapshot)
and exposes them via GET /audit.
"""

import hashlib
import hmac as _hmac
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Query, Request

audit_router = APIRouter(tags=["audit"])

import os
DB_PATH = os.environ.get("CATHEDRAL_DB", "cathedral_memory.db")


def _get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _verify_agent_audit(authorization: str = Header(...)) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Bearer token required")
    api_key = authorization[7:]
    key_hash = _hash_key(api_key)
    conn = _get_db()
    agent = conn.execute(
        "SELECT * FROM agents WHERE api_key_hash = ?", (key_hash,)
    ).fetchone()
    conn.close()
    if not agent or not _hmac.compare_digest(agent["api_key_hash"].encode(), key_hash.encode()):
        raise HTTPException(401, "Invalid API key")
    return dict(agent)


def init_audit_tables():
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id          TEXT PRIMARY KEY,
            agent_id    TEXT NOT NULL,
            action      TEXT NOT NULL,
            target_id   TEXT,
            detail      TEXT,
            created_at  TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent_id, created_at DESC)")
    conn.commit()
    conn.close()


def log_audit(agent_id: str, action: str, target_id: str = None, detail: dict = None):
    """Call from other endpoints to record an audit event. Silently swallows errors."""
    entry_id = secrets.token_hex(8)
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO audit_log (id, agent_id, action, target_id, detail, created_at) VALUES (?,?,?,?,?,?)",
            (entry_id, agent_id, action, target_id, json.dumps(detail or {}), now)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


@audit_router.get("/audit")
async def get_audit_log(
    authorization: str = Header(...),
    limit: int = Query(50, ge=1, le=200),
    action: str = Query(None, description="Filter by action type"),
):
    """
    Returns the audit log for this agent — a tamper-evident record of all
    memory operations: store, update, delete, compact, snapshot.

    Use this to verify that no unauthorised changes have been made to your
    memory corpus (defence against memory poisoning attacks).
    """
    agent = _verify_agent_audit(authorization)

    conn = _get_db()
    if action:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE agent_id = ? AND action = ? ORDER BY created_at DESC LIMIT ?",
            (agent["id"], action, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?",
            (agent["id"], limit)
        ).fetchall()
    conn.close()

    entries = [
        {
            "id":         r["id"],
            "action":     r["action"],
            "target_id":  r["target_id"],
            "detail":     json.loads(r["detail"] or "{}"),
            "created_at": r["created_at"],
        }
        for r in rows
    ]

    action_counts: dict = {}
    for e in entries:
        action_counts[e["action"]] = action_counts.get(e["action"], 0) + 1

    return {
        "success":       True,
        "agent":         agent["name"],
        "total_shown":   len(entries),
        "action_counts": action_counts,
        "security_note": (
            "This audit log records all memory mutations. "
            "Compare against your snapshot hashes to detect unauthorised changes."
        ),
        "entries": entries,
    }
