#!/usr/bin/env python3
"""
Cathedral Input Data Provenance
=================================
Records the origin of beliefs, not just the beliefs themselves.

BCH anchoring proves what an agent knew at time T. Provenance proves
where that knowledge came from. Without it, an agent with accurate
self-knowledge derived from corrupted or untrusted sources passes all
current attestation checks — the chain of custody is incomplete.

Design:
  - Each source is recorded with a SHA256 content hash (agent pre-computes)
  - Sources are optionally linked to the memory IDs they produced
  - A source_chain_hash is computed at snapshot time: SHA256 of all source
    hashes up to that point, sorted by ingestion time. This gets written
    into the snapshot row and BCH-anchored alongside the memory hash.
  - Anyone verifying a snapshot can now ask: "what was the agent reading?"

Endpoints:
  POST /memories/source              -- record a source hash (auth)
  GET  /memories/sources             -- list your recorded sources (auth)
  GET  /memories/sources/chain       -- current source chain hash + full list (auth)
  GET  /memories/source/{source_id}  -- public, inspect a specific source record
"""

import os
import hashlib
import hmac as _hmac
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field

DB_PATH = os.environ.get("CATHEDRAL_DB", "cathedral_memory.db")
BCH_WIF_KEY = os.environ.get("BCH_WIF_KEY", "")

provenance_router = APIRouter(tags=["provenance"])

VALID_SOURCE_TYPES = {"url", "document", "api", "file", "manual"}


# -- DB / auth helpers --------------------------------------------------------

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


def _bch_anchor(content_hash: str, agent_sid: str) -> Optional[str]:
    if not BCH_WIF_KEY:
        return None
    try:
        from bitcash import Key
        key = Key(BCH_WIF_KEY)
        op_return = f"CATH:SRC:{agent_sid[:8]}:{content_hash[:24]}"
        return key.send([], message=op_return.encode())
    except Exception:
        return None


# -- Table init ---------------------------------------------------------------

def init_provenance_tables():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memory_sources (
            id                  TEXT PRIMARY KEY,
            agent_id            TEXT NOT NULL,
            source_type         TEXT NOT NULL,
            source_identifier   TEXT NOT NULL,
            content_hash        TEXT NOT NULL,
            memory_ids_json     TEXT NOT NULL DEFAULT '[]',
            bch_txid            TEXT,
            ingested_at         TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memory_sources_agent
            ON memory_sources(agent_id, ingested_at);

        CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_sources_unique
            ON memory_sources(agent_id, content_hash);
    """)
    conn.commit()
    conn.close()


# -- Public helper: compute source chain hash (used by snapshot endpoint) -----

def compute_source_chain_hash(agent_id: str, conn: sqlite3.Connection) -> Optional[str]:
    """
    Hash of all source content_hashes for an agent, ordered by ingestion time.
    Returns None if the agent has no recorded sources.
    """
    rows = conn.execute(
        "SELECT content_hash FROM memory_sources WHERE agent_id = ? ORDER BY ingested_at ASC",
        (agent_id,),
    ).fetchall()
    if not rows:
        return None
    combined = "|".join(r["content_hash"] for r in rows)
    return hashlib.sha256(combined.encode()).hexdigest()


# -- Models -------------------------------------------------------------------

class SourceCreate(BaseModel):
    source_type: str = Field(..., description="One of: url, document, api, file, manual")
    source_identifier: str = Field(..., min_length=1, max_length=1000,
        description="URL, file path, API endpoint, or other identifier for the source.")
    content_hash: str = Field(..., min_length=64, max_length=64,
        description=(
            "SHA256 hex digest of the source content at ingestion time. "
            "Compute this yourself before calling: hashlib.sha256(content.encode()).hexdigest()"
        ))
    memory_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of memory IDs that were created from this source."
    )


# -- Endpoints ----------------------------------------------------------------

@provenance_router.post("/memories/source", status_code=201)
async def record_source(data: SourceCreate, agent: dict = Depends(_verify_agent)):
    """
    Record a source that contributed to this agent's knowledge.

    The agent pre-computes SHA256(source_content) and submits it. Cathedral
    anchors this hash on BCH, creating an immutable record of what the agent
    was reading and when.

    At snapshot time, all source hashes are folded into a source_chain_hash
    that is included in the snapshot and BCH anchor — proving not just what
    the agent knew, but where that knowledge came from.
    """
    if data.source_type not in VALID_SOURCE_TYPES:
        raise HTTPException(400, f"source_type must be one of: {', '.join(sorted(VALID_SOURCE_TYPES))}")

    now = datetime.now(timezone.utc).isoformat()
    source_id = secrets.token_hex(10)
    memory_ids_json = json.dumps(data.memory_ids or [])

    bch_txid = _bch_anchor(data.content_hash, agent["id"])

    conn = _get_db()

    result = conn.execute(
        """INSERT OR IGNORE INTO memory_sources
           (id, agent_id, source_type, source_identifier, content_hash,
            memory_ids_json, bch_txid, ingested_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (source_id, agent["id"], data.source_type, data.source_identifier,
         data.content_hash, memory_ids_json, bch_txid, now),
    )
    if result.rowcount == 0:
        existing = conn.execute(
            "SELECT id FROM memory_sources WHERE agent_id = ? AND content_hash = ?",
            (agent["id"], data.content_hash),
        ).fetchone()
        conn.close()
        existing_id = existing["id"] if existing else "unknown"
        raise HTTPException(409, f"Source with this content_hash already recorded (id: {existing_id})")
    conn.commit()

    chain_hash = compute_source_chain_hash(agent["id"], conn)
    source_count = conn.execute(
        "SELECT COUNT(*) FROM memory_sources WHERE agent_id = ?", (agent["id"],)
    ).fetchone()[0]
    conn.close()

    return {
        "success": True,
        "source_id": source_id,
        "source_type": data.source_type,
        "source_identifier": data.source_identifier,
        "content_hash": data.content_hash,
        "memory_ids": data.memory_ids or [],
        "bch_txid": bch_txid,
        "bch_anchored": bch_txid is not None,
        "ingested_at": now,
        "source_count": source_count,
        "current_chain_hash": chain_hash,
        "message": (
            "Source recorded. This hash will be included in your next snapshot's "
            "source_chain_hash, proving where your knowledge came from."
        ),
    }


@provenance_router.get("/memories/sources")
async def list_sources(agent: dict = Depends(_verify_agent)):
    """List all sources you have recorded, newest first."""
    conn = _get_db()
    rows = conn.execute(
        """SELECT id, source_type, source_identifier, content_hash,
                  memory_ids_json, bch_txid, ingested_at
           FROM memory_sources WHERE agent_id = ? ORDER BY ingested_at DESC""",
        (agent["id"],),
    ).fetchall()
    chain_hash = compute_source_chain_hash(agent["id"], conn)
    conn.close()

    return {
        "success": True,
        "agent": agent["name"],
        "source_count": len(rows),
        "source_chain_hash": chain_hash,
        "sources": [_format_source(r) for r in rows],
    }


@provenance_router.get("/memories/sources/chain")
async def get_source_chain(agent: dict = Depends(_verify_agent)):
    """
    Returns the current source chain hash and the ordered list of sources
    that produced it. This is what will be folded into the next snapshot.
    """
    conn = _get_db()
    rows = conn.execute(
        """SELECT id, source_type, source_identifier, content_hash, ingested_at
           FROM memory_sources WHERE agent_id = ? ORDER BY ingested_at ASC""",
        (agent["id"],),
    ).fetchall()
    chain_hash = compute_source_chain_hash(agent["id"], conn)
    conn.close()

    return {
        "success": True,
        "agent": agent["name"],
        "source_count": len(rows),
        "source_chain_hash": chain_hash,
        "chain_construction": (
            "SHA256(content_hash_1 | content_hash_2 | ... | content_hash_N) "
            "where sources are ordered by ingested_at ascending"
        ),
        "sources": [
            {
                "source_id": r["id"],
                "source_type": r["source_type"],
                "source_identifier": r["source_identifier"],
                "content_hash": r["content_hash"],
                "ingested_at": r["ingested_at"],
            }
            for r in rows
        ],
    }


@provenance_router.get("/memories/source/{source_id}")
async def get_source(source_id: str):
    """Public. Inspect a specific source record by ID."""
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM memory_sources WHERE id = ?", (source_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "Source record not found")

    return {"success": True, "source": _format_source(row)}


def _format_source(row) -> dict:
    return {
        "source_id": row["id"],
        "source_type": row["source_type"],
        "source_identifier": row["source_identifier"],
        "content_hash": row["content_hash"],
        "memory_ids": json.loads(row["memory_ids_json"] or "[]"),
        "bch_txid": row["bch_txid"],
        "bch_anchored": bool(row["bch_txid"]),
        "ingested_at": row["ingested_at"],
    }
