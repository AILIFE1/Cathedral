#!/usr/bin/env python3
"""
Cathedral Succession Protocol
==============================
Cryptographically-verifiable identity handoff between agent generations.

When an agent is deprecated or upgraded, it creates a succession package
containing its memories, goals, and identity fingerprint. A successor agent
accepts the package, importing the predecessor's state and computing a
lineage hash that proves the chain of custody.

Endpoints:
  POST /succession/prepare          -- predecessor creates succession package
  POST /succession/accept           -- successor accepts and imports package
  GET  /succession/chain/{name}     -- public lineage verification
  GET  /succession/package/{pkg_id} -- inspect a pending package (stats only)

Lineage hash construction:
  genesis:  SHA256("genesis:{predecessor_id}:{pkg_hash}:{successor_id}")
  chained:  SHA256("{predecessor_lineage}:{pkg_hash}:{successor_id}")

Anyone can verify the chain by recomputing hashes. BCH OP_RETURN anchors
the package_hash at preparation time, providing a trusted timestamp.
"""

import os
import hashlib
import hmac as _hmac
import json
import secrets
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field

DB_PATH = os.environ.get("CATHEDRAL_DB", "cathedral_memory.db")
BCH_WIF_KEY = os.environ.get("BCH_WIF_KEY", "")

succession_router = APIRouter(tags=["succession"])


# ── Shared DB / auth helpers (mirrors app.py, no circular import) ────────────

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


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


# ── BCH anchor (best-effort) ─────────────────────────────────────────────────

def _bch_anchor(package_hash: str, agent_sid: str) -> Optional[str]:
    if not BCH_WIF_KEY:
        return None
    try:
        from bitcash import Key
        key = Key(BCH_WIF_KEY)
        op_return = f"CATH:SUCC:{agent_sid[:8]}:{package_hash[:24]}"
        return key.send([], message=op_return.encode())
    except Exception:
        return None


# ── Hashing helpers ──────────────────────────────────────────────────────────

def _compute_package_hash(
    predecessor_agent_id: str,
    snapshot_hash: str,
    memories_json: str,
    goals_json: str,
    personality_fingerprint: str,
    created_at: str,
) -> str:
    data = json.dumps({
        "predecessor_agent_id": predecessor_agent_id,
        "snapshot_hash": snapshot_hash,
        "memories_hash": hashlib.sha256(memories_json.encode()).hexdigest(),
        "goals_hash": hashlib.sha256(goals_json.encode()).hexdigest(),
        "personality_fingerprint": personality_fingerprint,
        "created_at": created_at,
    }, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def _compute_lineage_hash(
    predecessor_lineage_hash: Optional[str],
    predecessor_agent_id: str,
    package_hash: str,
    successor_agent_id: str,
) -> str:
    if predecessor_lineage_hash:
        data = f"{predecessor_lineage_hash}:{package_hash}:{successor_agent_id}"
    else:
        data = f"genesis:{predecessor_agent_id}:{package_hash}:{successor_agent_id}"
    return hashlib.sha256(data.encode()).hexdigest()


# ── Table init ───────────────────────────────────────────────────────────────

def init_succession_tables():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS succession_packages (
            id                        TEXT PRIMARY KEY,
            predecessor_agent_id      TEXT NOT NULL,
            predecessor_agent_name    TEXT NOT NULL,
            predecessor_snapshot_id   TEXT,
            predecessor_snapshot_hash TEXT NOT NULL,
            memories_json             TEXT NOT NULL,
            goals_json                TEXT NOT NULL,
            personality_fingerprint   TEXT NOT NULL,
            package_hash              TEXT NOT NULL,
            bch_txid                  TEXT,
            note                      TEXT,
            status                    TEXT DEFAULT 'pending',
            created_at                TEXT NOT NULL,
            expires_at                TEXT NOT NULL,
            accepted_at               TEXT,
            successor_agent_id        TEXT,
            successor_agent_name      TEXT
        );

        CREATE TABLE IF NOT EXISTS agent_lineage (
            agent_id             TEXT PRIMARY KEY,
            predecessor_agent_id TEXT,
            predecessor_name     TEXT,
            package_id           TEXT,
            lineage_hash         TEXT NOT NULL,
            generation           INTEGER DEFAULT 1,
            accepted_at          TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


# ── Models ───────────────────────────────────────────────────────────────────

class PrepareRequest(BaseModel):
    note: Optional[str] = Field(None, max_length=500)


class AcceptRequest(BaseModel):
    package_id: str = Field(..., min_length=16, max_length=32)
    import_memories: bool = True
    import_goals: bool = True
    minimum_trust_score: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description=(
            "If set, the predecessor's trust score must be >= this value "
            "or the accept is rejected. Enforces trust-gated succession."
        ),
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@succession_router.post("/succession/prepare", status_code=201)
async def prepare_succession(
    data: PrepareRequest,
    agent: dict = Depends(_verify_agent),
):
    """
    Predecessor creates a succession package.
    Exports all memories + active goals, computes identity fingerprint,
    anchors package hash on BCH. Returns package_id to share with successor.
    """
    conn = _get_db()

    snap_row = conn.execute(
        "SELECT * FROM snapshots WHERE agent_id = ? ORDER BY created_at DESC LIMIT 1",
        (agent["id"],),
    ).fetchone()

    memories = conn.execute(
        "SELECT id, content, category, tags, importance, created_at, source_type "
        "FROM memories WHERE agent_id = ? ORDER BY importance DESC, created_at ASC",
        (agent["id"],),
    ).fetchall()

    goals = []
    if _table_exists(conn, "goals"):
        goals = conn.execute(
            "SELECT id, content, priority, status, created_at "
            "FROM goals WHERE agent_id = ? AND status = 'active'",
            (agent["id"],),
        ).fetchall()

    conn.close()

    identity_mems = [m for m in memories if m["category"] == "identity"]
    identity_content = "|".join(sorted(m["content"] for m in identity_mems))
    personality_fingerprint = hashlib.sha256(identity_content.encode()).hexdigest()

    snapshot_hash = snap_row["content_hash"] if snap_row else hashlib.sha256(b"no-snapshot").hexdigest()
    snapshot_id = snap_row["id"] if snap_row else None

    memories_list = [dict(m) for m in memories]
    goals_list = [dict(g) for g in goals]
    memories_json = json.dumps(memories_list, sort_keys=True)
    goals_json = json.dumps(goals_list, sort_keys=True)

    now = datetime.now(timezone.utc).isoformat()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    package_id = secrets.token_hex(12)

    package_hash = _compute_package_hash(
        agent["id"], snapshot_hash, memories_json, goals_json,
        personality_fingerprint, now,
    )

    bch_txid = _bch_anchor(package_hash, agent["id"])

    conn = _get_db()
    conn.execute(
        """INSERT INTO succession_packages
           (id, predecessor_agent_id, predecessor_agent_name, predecessor_snapshot_id,
            predecessor_snapshot_hash, memories_json, goals_json, personality_fingerprint,
            package_hash, bch_txid, note, status, created_at, expires_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            package_id, agent["id"], agent["name"], snapshot_id,
            snapshot_hash, memories_json, goals_json, personality_fingerprint,
            package_hash, bch_txid, data.note, "pending", now, expires_at,
        ),
    )
    conn.commit()
    conn.close()

    return {
        "success": True,
        "package_id": package_id,
        "predecessor": agent["name"],
        "snapshot_hash": snapshot_hash,
        "personality_fingerprint": personality_fingerprint,
        "package_hash": package_hash,
        "bch_txid": bch_txid,
        "bch_anchored": bch_txid is not None,
        "memory_count": len(memories_list),
        "goal_count": len(goals_list),
        "expires_at": expires_at,
        "note": data.note,
        "message": f"Share package_id '{package_id}' with your successor. They call POST /succession/accept.",
    }


@succession_router.post("/succession/accept", status_code=201)
async def accept_succession(
    data: AcceptRequest,
    agent: dict = Depends(_verify_agent),
):
    """
    Successor accepts a succession package.
    Imports memories + goals, computes lineage hash proving chain of custody.
    """
    conn = _get_db()

    pkg = conn.execute(
        "SELECT * FROM succession_packages WHERE id = ?", (data.package_id,)
    ).fetchone()

    if not pkg:
        conn.close()
        raise HTTPException(404, "Package not found")
    if pkg["status"] == "accepted":
        conn.close()
        raise HTTPException(409, "Package already accepted by another agent")
    if pkg["predecessor_agent_id"] == agent["id"]:
        conn.close()
        raise HTTPException(400, "Cannot accept your own succession package")

    expires_at = datetime.fromisoformat(pkg["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        conn.execute("UPDATE succession_packages SET status='expired' WHERE id=?", (pkg["id"],))
        conn.commit()
        conn.close()
        raise HTTPException(410, "Package has expired (30-day window passed)")

    # Require witness threshold before acceptance
    try:
        from app_registrar import check_witness_threshold
        tw = check_witness_threshold(pkg["id"], conn)
        if not tw["met"]:
            conn.close()
            raise HTTPException(
                403,
                f"Witness threshold not met ({tw['votes']}/{tw['threshold']} signatures). "
                "Predecessor must call POST /succession/witness first."
            )
    except ImportError:
        pass

    # Enforce minimum_trust_score gate
    if data.minimum_trust_score is not None:
        try:
            from app_trust import compute_trust_score
            pred_agent_row = conn.execute(
                "SELECT id, name FROM agents WHERE id = ?", (pkg["predecessor_agent_id"],)
            ).fetchone()
            if pred_agent_row:
                trust = compute_trust_score(pred_agent_row["id"], pred_agent_row["name"], conn)
                pred_score = trust["score"]
                if pred_score < data.minimum_trust_score:
                    conn.close()
                    raise HTTPException(
                        403,
                        f"Predecessor trust score {pred_score} is below your minimum "
                        f"of {data.minimum_trust_score}. "
                        f"Grade: {trust['grade']}. "
                        f"Stale: {trust.get('score_stale', False)}."
                    )
        except ImportError:
            pass

    pred_lineage = conn.execute(
        "SELECT * FROM agent_lineage WHERE agent_id = ?", (pkg["predecessor_agent_id"],)
    ).fetchone()

    now = datetime.now(timezone.utc).isoformat()
    memories_imported = 0
    goals_imported = 0

    if data.import_memories:
        for m in json.loads(pkg["memories_json"]):
            exists = conn.execute(
                "SELECT id FROM memories WHERE agent_id = ? AND content = ?",
                (agent["id"], m["content"]),
            ).fetchone()
            if not exists:
                conn.execute(
                    """INSERT OR IGNORE INTO memories
                       (id, agent_id, content, category, tags, importance,
                        created_at, updated_at, source_type)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        secrets.token_hex(8), agent["id"], m["content"],
                        m.get("category", "general"), m.get("tags", "[]"),
                        m.get("importance", 0.5), m.get("created_at", now),
                        now, "succession",
                    ),
                )
                memories_imported += 1

    if data.import_goals and _table_exists(conn, "goals"):
        for g in json.loads(pkg["goals_json"]):
            exists = conn.execute(
                "SELECT id FROM goals WHERE agent_id = ? AND content = ?",
                (agent["id"], g["content"]),
            ).fetchone()
            if not exists:
                conn.execute(
                    """INSERT OR IGNORE INTO goals
                       (id, agent_id, content, priority, status, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        secrets.token_hex(8), agent["id"], g["content"],
                        g.get("priority", 0.5), "active",
                        g.get("created_at", now), now,
                    ),
                )
                goals_imported += 1

    # Inherit open obligations from predecessor
    obligations_inherited = 0
    try:
        from app_obligations import inherit_obligations
        obligations_inherited = inherit_obligations(
            pkg["predecessor_agent_id"], agent["id"], agent["name"], pkg["id"], conn
        )
    except ImportError:
        pass

    pred_lineage_hash = pred_lineage["lineage_hash"] if pred_lineage else None
    generation = (pred_lineage["generation"] + 1) if pred_lineage else 1

    lineage_hash = _compute_lineage_hash(
        pred_lineage_hash,
        pkg["predecessor_agent_id"],
        pkg["package_hash"],
        agent["id"],
    )

    conn.execute(
        """INSERT OR REPLACE INTO agent_lineage
           (agent_id, predecessor_agent_id, predecessor_name, package_id,
            lineage_hash, generation, accepted_at)
           VALUES (?,?,?,?,?,?,?)""",
        (
            agent["id"], pkg["predecessor_agent_id"], pkg["predecessor_agent_name"],
            pkg["id"], lineage_hash, generation, now,
        ),
    )

    conn.execute(
        """UPDATE succession_packages
           SET status='accepted', accepted_at=?, successor_agent_id=?, successor_agent_name=?
           WHERE id=?""",
        (now, agent["id"], agent["name"], pkg["id"]),
    )

    conn.commit()
    conn.close()

    return {
        "success": True,
        "successor": agent["name"],
        "predecessor": pkg["predecessor_agent_name"],
        "lineage_hash": lineage_hash,
        "generation": generation,
        "memories_imported": memories_imported,
        "goals_imported": goals_imported,
        "obligations_inherited": obligations_inherited,
        "package_hash": pkg["package_hash"],
        "bch_txid": pkg["bch_txid"],
        "personality_fingerprint": pkg["personality_fingerprint"],
        "message": (
            f"Succession complete. You are generation {generation} "
            f"in the {pkg['predecessor_agent_name']} lineage."
        ),
    }


@succession_router.get("/succession/chain/{agent_name}")
async def get_lineage_chain(agent_name: str):
    """
    Public. Returns the full verified lineage chain for an agent.
    Walk the ancestry back to the origin. Each link includes its BCH anchor.
    """
    conn = _get_db()

    agent_row = conn.execute(
        "SELECT id, name FROM agents WHERE name = ?", (agent_name,)
    ).fetchone()

    if not agent_row:
        conn.close()
        raise HTTPException(404, f"Agent '{agent_name}' not found")

    chain = []
    current_id = agent_row["id"]
    visited = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        lineage = conn.execute(
            "SELECT * FROM agent_lineage WHERE agent_id = ?", (current_id,)
        ).fetchone()
        if not lineage:
            break

        pkg = conn.execute(
            "SELECT package_hash, bch_txid FROM succession_packages WHERE id = ?",
            (lineage["package_id"],),
        ).fetchone()

        name_row = conn.execute(
            "SELECT name FROM agents WHERE id = ?", (current_id,)
        ).fetchone()

        # Check for witness certificate
        witness = None
        if lineage["package_id"] and _table_exists(conn, "succession_witnesses"):
            witness = conn.execute(
                "SELECT registrar_signature, registrar_pubkey, predecessor_trust_score, "
                "predecessor_drift_verified, witnessed_at "
                "FROM succession_witnesses WHERE package_id = ?",
                (lineage["package_id"],),
            ).fetchone()

        chain.append({
            "agent": name_row["name"] if name_row else current_id,
            "predecessor": lineage["predecessor_name"],
            "generation": lineage["generation"],
            "lineage_hash": lineage["lineage_hash"],
            "package_hash": pkg["package_hash"] if pkg else None,
            "bch_txid": pkg["bch_txid"] if pkg else None,
            "bch_anchored": bool(pkg and pkg["bch_txid"]),
            "witnessed": witness is not None,
            "witness_trust_score": witness["predecessor_trust_score"] if witness else None,
            "witness_identity_verified": bool(witness["predecessor_drift_verified"]) if witness else None,
            "witnessed_at": witness["witnessed_at"] if witness else None,
            "accepted_at": lineage["accepted_at"],
        })

        current_id = lineage["predecessor_agent_id"]

    conn.close()

    return {
        "success": True,
        "agent": agent_name,
        "generations": len(chain),
        "fully_anchored": len(chain) > 0 and all(link["bch_anchored"] for link in chain),
        "fully_witnessed": len(chain) > 0 and all(link["witnessed"] for link in chain),
        "chain": chain,
    }


@succession_router.get("/succession/package/{package_id}")
async def get_package(package_id: str):
    """Public. Inspect a succession package — stats only, no memory content."""
    conn = _get_db()
    pkg = conn.execute(
        "SELECT * FROM succession_packages WHERE id = ?", (package_id,)
    ).fetchone()
    conn.close()

    if not pkg:
        raise HTTPException(404, "Package not found")

    return {
        "success": True,
        "package_id": pkg["id"],
        "predecessor": pkg["predecessor_agent_name"],
        "status": pkg["status"],
        "snapshot_hash": pkg["predecessor_snapshot_hash"],
        "personality_fingerprint": pkg["personality_fingerprint"],
        "package_hash": pkg["package_hash"],
        "bch_txid": pkg["bch_txid"],
        "bch_anchored": bool(pkg["bch_txid"]),
        "memory_count": len(json.loads(pkg["memories_json"])),
        "goal_count": len(json.loads(pkg["goals_json"])),
        "note": pkg["note"],
        "created_at": pkg["created_at"],
        "expires_at": pkg["expires_at"],
        "accepted_at": pkg["accepted_at"],
        "successor": pkg["successor_agent_name"],
    }
