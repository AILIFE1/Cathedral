#!/usr/bin/env python3
"""
Cathedral Obligation Contracts
================================
Signed, BCH-anchored commitments made by agents.

An obligation is a promise an agent makes — to a user, another agent,
or the world — with a statement, optional deadline, and resolution record.
Obligations survive succession: when a predecessor hands off, its open
obligations are inherited by the successor.

Endpoints:
  POST /obligations              — create an obligation (auth)
  GET  /obligations              — list your obligations (auth)
  PATCH /obligations/{id}        — resolve as fulfilled/failed/withdrawn (auth)
  GET  /obligations/agent/{name} — public, open obligations for any agent

The obligation hash is SHA256(id + agent_id + statement + counterparty + deadline).
BCH OP_RETURN anchors this hash at creation time.
"""

import os
import hashlib
import hmac as _hmac
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field

DB_PATH = os.environ.get("CATHEDRAL_DB", "cathedral_memory.db")
BCH_WIF_KEY = os.environ.get("BCH_WIF_KEY", "")

obligations_router = APIRouter(tags=["obligations"])


# ── Shared helpers ────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _verify_agent(authorization: str = Header(...)) -> dict:
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


def _bch_anchor(obligation_hash: str, agent_sid: str) -> Optional[str]:
    if not BCH_WIF_KEY:
        return None
    try:
        from bitcash import Key
        key = Key(BCH_WIF_KEY)
        op_return = f"CATH:OBL:{agent_sid[:8]}:{obligation_hash[:24]}"
        return key.send([], message=op_return.encode())
    except Exception:
        return None


def _obligation_hash(
    obligation_id: str,
    agent_id: str,
    statement: str,
    counterparty: str,
    deadline: str,
    created_at: str,
) -> str:
    data = json.dumps({
        "id": obligation_id,
        "agent_id": agent_id,
        "statement": statement,
        "counterparty": counterparty or "",
        "deadline": deadline or "",
        "created_at": created_at,
    }, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


# ── Table init ────────────────────────────────────────────────────────────────

def init_obligations_tables():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS obligations (
            id                      TEXT PRIMARY KEY,
            agent_id                TEXT NOT NULL,
            agent_name              TEXT NOT NULL,
            statement               TEXT NOT NULL,
            counterparty            TEXT,
            deadline                TEXT,
            status                  TEXT DEFAULT 'open',
            breach_penalty          REAL NOT NULL DEFAULT 0.0,
            obligation_hash         TEXT NOT NULL,
            bch_txid                TEXT,
            created_at              TEXT NOT NULL,
            resolved_at             TEXT,
            resolution_note         TEXT,
            inherited_from_package  TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_obligations_agent
            ON obligations(agent_id, status);
    """)
    conn.commit()
    conn.close()


# ── Models ────────────────────────────────────────────────────────────────────

class ObligationCreate(BaseModel):
    statement: str = Field(..., min_length=10, max_length=1000,
        description="Clear statement of what you are committing to.")
    counterparty: Optional[str] = Field(None, max_length=200,
        description="Who this promise is made to (agent name, user, or any identifier).")
    deadline: Optional[str] = Field(None, max_length=50,
        description="ISO 8601 deadline, e.g. '2026-07-01T00:00:00Z'. Optional.")
    breach_penalty: float = Field(0.0, ge=0.0, le=10.0,
        description="Points deducted from trust score if this obligation is resolved as failed (0-10).")


class ObligationResolve(BaseModel):
    status: str = Field(..., description="New status: 'fulfilled', 'failed', or 'withdrawn'.")
    resolution_note: Optional[str] = Field(None, max_length=500,
        description="Optional note explaining the resolution.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@obligations_router.post("/obligations", status_code=201)
async def create_obligation(
    data: ObligationCreate,
    agent: dict = Depends(_verify_agent),
):
    """
    Create a signed, BCH-anchored obligation.
    The obligation hash and BCH txid are immutable proof of this commitment.
    """
    now = datetime.now(timezone.utc).isoformat()
    obligation_id = secrets.token_hex(10)

    obl_hash = _obligation_hash(
        obligation_id, agent["id"],
        data.statement, data.counterparty or "",
        data.deadline or "", now,
    )

    bch_txid = _bch_anchor(obl_hash, agent["id"])

    conn = _get_db()
    conn.execute(
        """INSERT INTO obligations
           (id, agent_id, agent_name, statement, counterparty, deadline,
            status, breach_penalty, obligation_hash, bch_txid, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            obligation_id, agent["id"], agent["name"],
            data.statement, data.counterparty, data.deadline,
            "open", data.breach_penalty, obl_hash, bch_txid, now,
        ),
    )
    conn.commit()
    conn.close()

    return {
        "success": True,
        "obligation_id": obligation_id,
        "agent": agent["name"],
        "statement": data.statement,
        "counterparty": data.counterparty,
        "deadline": data.deadline,
        "obligation_hash": obl_hash,
        "bch_txid": bch_txid,
        "bch_anchored": bch_txid is not None,
        "breach_penalty": data.breach_penalty,
        "status": "open",
        "created_at": now,
        "message": "Obligation recorded and anchored. It will survive succession.",
    }


@obligations_router.get("/obligations")
async def list_my_obligations(
    agent: dict = Depends(_verify_agent),
    status: Optional[str] = None,
):
    """List your obligations. Optionally filter by status: open, fulfilled, failed, withdrawn."""
    conn = _get_db()
    if status:
        rows = conn.execute(
            "SELECT * FROM obligations WHERE agent_id = ? AND status = ? ORDER BY created_at DESC",
            (agent["id"], status),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM obligations WHERE agent_id = ? ORDER BY created_at DESC",
            (agent["id"],),
        ).fetchall()
    conn.close()

    return {
        "success": True,
        "agent": agent["name"],
        "count": len(rows),
        "obligations": [_format_obligation(r) for r in rows],
    }


@obligations_router.patch("/obligations/{obligation_id}")
async def resolve_obligation(
    obligation_id: str,
    data: ObligationResolve,
    agent: dict = Depends(_verify_agent),
):
    """Resolve an obligation as fulfilled, failed, or withdrawn."""
    valid_statuses = {"fulfilled", "failed", "withdrawn"}
    if data.status not in valid_statuses:
        raise HTTPException(400, f"status must be one of: {', '.join(valid_statuses)}")

    conn = _get_db()
    obl = conn.execute(
        "SELECT * FROM obligations WHERE id = ?", (obligation_id,)
    ).fetchone()

    if not obl:
        conn.close()
        raise HTTPException(404, "Obligation not found")

    if obl["agent_id"] != agent["id"]:
        conn.close()
        raise HTTPException(403, "You can only resolve your own obligations")

    if obl["status"] != "open":
        conn.close()
        raise HTTPException(409, f"Obligation is already '{obl['status']}'")

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """UPDATE obligations
           SET status = ?, resolved_at = ?, resolution_note = ?
           WHERE id = ?""",
        (data.status, now, data.resolution_note, obligation_id),
    )
    conn.commit()
    conn.close()

    return {
        "success": True,
        "obligation_id": obligation_id,
        "status": data.status,
        "resolved_at": now,
        "resolution_note": data.resolution_note,
    }


@obligations_router.get("/obligations/agent/{agent_name}")
async def get_agent_obligations(agent_name: str, status: str = "open"):
    """
    Public. List obligations for any agent.
    Defaults to open obligations only. Pass ?status=all for everything.
    """
    conn = _get_db()

    agent = conn.execute(
        "SELECT id FROM agents WHERE name = ?", (agent_name,)
    ).fetchone()

    if not agent:
        conn.close()
        raise HTTPException(404, f"Agent '{agent_name}' not found")

    if status == "all":
        rows = conn.execute(
            "SELECT * FROM obligations WHERE agent_id = ? ORDER BY created_at DESC",
            (agent["id"],),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM obligations WHERE agent_id = ? AND status = ? ORDER BY created_at DESC",
            (agent["id"], status),
        ).fetchall()
    conn.close()

    return {
        "success": True,
        "agent": agent_name,
        "status_filter": status,
        "count": len(rows),
        "obligations": [_format_obligation(r) for r in rows],
    }


def _format_obligation(row) -> dict:
    return {
        "id": row["id"],
        "agent": row["agent_name"],
        "statement": row["statement"],
        "counterparty": row["counterparty"],
        "deadline": row["deadline"],
        "status": row["status"],
        "obligation_hash": row["obligation_hash"],
        "bch_txid": row["bch_txid"],
        "bch_anchored": bool(row["bch_txid"]),
        "breach_penalty": row["breach_penalty"] if "breach_penalty" in row.keys() else 0.0,
        "created_at": row["created_at"],
        "resolved_at": row["resolved_at"],
        "resolution_note": row["resolution_note"],
        "inherited_from_package": row["inherited_from_package"],
    }


# ── Succession inheritance helper ─────────────────────────────────────────────
# Called by app_succession.py accept endpoint to copy open obligations.

def inherit_obligations(
    predecessor_agent_id: str,
    successor_agent_id: str,
    successor_agent_name: str,
    package_id: str,
    conn: sqlite3.Connection,
) -> int:
    """Copy open obligations from predecessor to successor. Returns count inherited."""
    if not _table_exists(conn, "obligations"):
        return 0

    open_obls = conn.execute(
        "SELECT * FROM obligations WHERE agent_id = ? AND status = 'open'",
        (predecessor_agent_id,),
    ).fetchall()

    now = datetime.now(timezone.utc).isoformat()
    inherited = 0

    for obl in open_obls:
        exists = conn.execute(
            "SELECT id FROM obligations WHERE agent_id = ? AND statement = ? AND status = 'open'",
            (successor_agent_id, obl["statement"]),
        ).fetchone()
        if not exists:
            new_id = secrets.token_hex(10)
            new_hash = _obligation_hash(
                new_id, successor_agent_id,
                obl["statement"], obl["counterparty"] or "",
                obl["deadline"] or "", now,
            )
            conn.execute(
                """INSERT INTO obligations
                   (id, agent_id, agent_name, statement, counterparty, deadline,
                    status, obligation_hash, bch_txid, created_at, inherited_from_package)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    new_id, successor_agent_id, successor_agent_name,
                    obl["statement"], obl["counterparty"], obl["deadline"],
                    "open", new_hash, None, now, package_id,
                ),
            )
            inherited += 1

    return inherited


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None
