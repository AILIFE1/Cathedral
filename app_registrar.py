#!/usr/bin/env python3
"""
Cathedral Succession Registrar — Multi-Node Threshold
=======================================================
N-of-M threshold witnessing for succession packages.

A single Cathedral-owned registrar means the predecessor implicitly selects
their witness by choosing Cathedral. This removes that bias: nodes are
protocol-selected (all active nodes are asked), not predecessor-selected.

Architecture:
  - registrar_nodes: known nodes, each with Ed25519 keypair
  - succession_witness_votes: one signature per (package, node)
  - CATHEDRAL_WITNESS_THRESHOLD (default 2): min signatures before accept is permitted
  - 3 local nodes seeded from master key: structurally independent signatures
  - External nodes: third parties register pubkey, submit signatures independently

Node keys are derived deterministically from the master registrar key:
  node-1: original CATHEDRAL_REGISTRAR_PRIVKEY
  node-2: SHA256(master + "cathedral-node-2")
  node-3: SHA256(master + "cathedral-node-3")

External parties can run independent nodes — their signatures count toward threshold
once registered via POST /succession/registrar/nodes (requires CATHEDRAL_ADMIN_KEY).

Backward compat:
  GET /succession/registrar/pubkey still returns node-1 pubkey
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
_MASTER_PRIVKEY_HEX = os.environ.get("CATHEDRAL_REGISTRAR_PRIVKEY", "")
_ADMIN_KEY = os.environ.get("CATHEDRAL_ADMIN_KEY", "")
_THRESHOLD = int(os.environ.get("CATHEDRAL_WITNESS_THRESHOLD", "2"))

registrar_router = APIRouter(tags=["registrar"])


# -- Key derivation -----------------------------------------------------------

def _derive_node_privkey(node_index: int) -> bytes:
    if not _MASTER_PRIVKEY_HEX:
        raise RuntimeError("CATHEDRAL_REGISTRAR_PRIVKEY not set")
    master = bytes.fromhex(_MASTER_PRIVKEY_HEX)
    if node_index == 1:
        return master
    return hashlib.sha256(master + f"cathedral-node-{node_index}".encode()).digest()


def _node_pubkey_hex(node_index: int) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    priv = Ed25519PrivateKey.from_private_bytes(_derive_node_privkey(node_index))
    return priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def _node_sign(node_index: int, message: str) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    priv = Ed25519PrivateKey.from_private_bytes(_derive_node_privkey(node_index))
    return priv.sign(message.encode()).hex()


def _verify_sig(message: str, signature_hex: str, pubkey_hex: str) -> bool:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
        pub.verify(bytes.fromhex(signature_hex), message.encode())
        return True
    except (InvalidSignature, Exception):
        return False


def _witness_message(package_hash: str, predecessor_id: str,
                     node_id: str, trust_score: float, signed_at: str) -> str:
    return f"{package_hash}:{predecessor_id}:{node_id}:{trust_score:.2f}:{signed_at}"


# -- DB helpers ---------------------------------------------------------------

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


def _verify_admin(authorization: str = Header(...)) -> None:
    if not _ADMIN_KEY:
        raise HTTPException(403, "Registrar admin not configured (set CATHEDRAL_ADMIN_KEY)")
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Bearer token required")
    if not _hmac.compare_digest(authorization[7:], _ADMIN_KEY):
        raise HTTPException(403, "Invalid admin key")


# -- Table init + node seeding ------------------------------------------------

def init_registrar_tables():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS registrar_nodes (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            description   TEXT,
            pubkey_hex    TEXT NOT NULL,
            node_index    INTEGER,
            is_local      INTEGER NOT NULL DEFAULT 0,
            is_active     INTEGER NOT NULL DEFAULT 1,
            registered_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS succession_witness_votes (
            id              TEXT PRIMARY KEY,
            package_id      TEXT NOT NULL,
            node_id         TEXT NOT NULL,
            node_name       TEXT NOT NULL,
            node_pubkey     TEXT NOT NULL,
            witness_message TEXT NOT NULL,
            signature       TEXT NOT NULL,
            trust_score     REAL NOT NULL,
            signed_at       TEXT NOT NULL,
            UNIQUE(package_id, node_id)
        );

        CREATE TABLE IF NOT EXISTS succession_witnesses (
            id                          TEXT PRIMARY KEY,
            package_id                  TEXT NOT NULL UNIQUE,
            predecessor_agent_id        TEXT NOT NULL,
            predecessor_agent_name      TEXT NOT NULL,
            predecessor_trust_score     REAL NOT NULL,
            predecessor_drift_verified  INTEGER NOT NULL DEFAULT 0,
            package_hash                TEXT NOT NULL,
            witness_message             TEXT NOT NULL,
            registrar_signature         TEXT NOT NULL,
            registrar_pubkey            TEXT NOT NULL,
            witnessed_at                TEXT NOT NULL
        );
    """)

    if not _MASTER_PRIVKEY_HEX:
        conn.commit()
        conn.close()
        return

    local_nodes = [
        ("cathedral-node-1", "Cathedral Node 1", "Primary Cathedral registrar node", 1),
        ("cathedral-node-2", "Cathedral Node 2", "Secondary Cathedral registrar node", 2),
        ("cathedral-node-3", "Cathedral Node 3", "Tertiary Cathedral registrar node", 3),
    ]
    now = datetime.now(timezone.utc).isoformat()
    for node_id, name, desc, idx in local_nodes:
        existing = conn.execute(
            "SELECT id FROM registrar_nodes WHERE id=?", (node_id,)
        ).fetchone()
        if not existing:
            try:
                pubkey = _node_pubkey_hex(idx)
                conn.execute(
                    """INSERT INTO registrar_nodes
                       (id, name, description, pubkey_hex, node_index,
                        is_local, is_active, registered_at)
                       VALUES (?,?,?,?,?,1,1,?)""",
                    (node_id, name, desc, pubkey, idx, now),
                )
            except Exception:
                pass

    conn.commit()
    conn.close()


# -- Trust score (inline, no circular import) ---------------------------------

def _quick_trust_score(agent_id: str, conn: sqlite3.Connection) -> float:
    snaps = conn.execute(
        "SELECT memories_json FROM snapshots WHERE agent_id=? ORDER BY created_at ASC",
        (agent_id,),
    ).fetchall()
    if len(snaps) < 2:
        snap_score = 0.0
    else:
        drifts = []
        for i in range(1, min(len(snaps), 20)):
            prev_ids = {m["id"] for m in json.loads(snaps[i-1]["memories_json"])}
            curr_ids = {m["id"] for m in json.loads(snaps[i]["memories_json"])}
            changed = len((prev_ids - curr_ids) | (curr_ids - prev_ids))
            drifts.append(changed / max(len(prev_ids), 1))
        avg = sum(drifts) / len(drifts)
        snap_score = max(0.0, 30.0 * (1.0 - avg * 4.0))

    agent_row = conn.execute("SELECT created_at FROM agents WHERE id=?", (agent_id,)).fetchone()
    if agent_row:
        try:
            dt = datetime.fromisoformat(agent_row["created_at"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            days = max(0, (datetime.now(timezone.utc) - dt).days)
        except Exception:
            days = 0
        longevity_score = min(20.0, days / 10.0)
    else:
        longevity_score = 0.0

    bch_count = conn.execute(
        "SELECT COUNT(*) FROM bch_anchors WHERE agent_id=? AND txid IS NOT NULL AND txid!=''",
        (agent_id,),
    ).fetchone()[0]
    anchoring_score = min(20.0, bch_count * 2.0)
    activity_score = min(15.0, float(len(snaps)))

    return round(snap_score + longevity_score + anchoring_score + activity_score, 1)


def _verify_identity_intact(agent_id: str, snapshot_hash: str,
                             conn: sqlite3.Connection) -> bool:
    latest = conn.execute(
        "SELECT content_hash FROM snapshots WHERE agent_id=? ORDER BY created_at DESC LIMIT 1",
        (agent_id,),
    ).fetchone()
    if not latest:
        return False
    return latest["content_hash"] == snapshot_hash


# -- Public helper: threshold check (used by app_succession) ------------------

def check_witness_threshold(package_id: str, conn: sqlite3.Connection) -> dict:
    votes = conn.execute(
        "SELECT node_id, node_name, signed_at FROM succession_witness_votes WHERE package_id=?",
        (package_id,),
    ).fetchall()
    count = len(votes)
    return {
        "threshold": _THRESHOLD,
        "votes": count,
        "met": count >= _THRESHOLD,
        "signers": [
            {"node_id": v["node_id"], "node_name": v["node_name"], "signed_at": v["signed_at"]}
            for v in votes
        ],
    }


# -- Models -------------------------------------------------------------------

class ExternalNodeRequest(BaseModel):
    node_id: str = Field(..., min_length=4, max_length=64)
    name: str = Field(..., min_length=2, max_length=128)
    description: Optional[str] = Field(None, max_length=500)
    pubkey_hex: str = Field(..., min_length=64, max_length=64)


class ExternalSignatureRequest(BaseModel):
    package_id: str = Field(..., min_length=16, max_length=32)
    witness_message: str
    signature: str = Field(..., min_length=128, max_length=128)
    signed_at: str
    trust_score: float = Field(..., ge=0.0, le=100.0)


# -- Endpoints ----------------------------------------------------------------

@registrar_router.post("/succession/witness", status_code=201)
async def request_witness(agent: dict = Depends(_verify_agent)):
    """
    Predecessor triggers multi-node witnessing for their pending succession package.

    All active local nodes sign automatically. External nodes are listed so
    the caller knows to contact them. Acceptance is blocked until threshold met.
    """
    conn = _get_db()

    pkg = conn.execute(
        """SELECT * FROM succession_packages
           WHERE predecessor_agent_id=? AND status='pending'
           ORDER BY created_at DESC LIMIT 1""",
        (agent["id"],),
    ).fetchone()
    if not pkg:
        conn.close()
        raise HTTPException(404, "No pending succession package. Call POST /succession/prepare first.")

    trust_score = _quick_trust_score(agent["id"], conn)
    identity_intact = _verify_identity_intact(agent["id"], pkg["predecessor_snapshot_hash"], conn)

    active_nodes = conn.execute(
        "SELECT * FROM registrar_nodes WHERE is_active=1 ORDER BY node_index ASC",
    ).fetchall()

    now = datetime.now(timezone.utc).isoformat()
    new_votes = []
    already_signed = []
    external_pending = []

    for node in active_nodes:
        existing = conn.execute(
            "SELECT id FROM succession_witness_votes WHERE package_id=? AND node_id=?",
            (pkg["id"], node["id"]),
        ).fetchone()
        if existing:
            already_signed.append(node["id"])
            continue

        if node["is_local"]:
            try:
                msg = _witness_message(
                    pkg["package_hash"], agent["id"], node["id"], trust_score, now
                )
                sig = _node_sign(node["node_index"], msg)
                conn.execute(
                    """INSERT OR IGNORE INTO succession_witness_votes
                       (id, package_id, node_id, node_name, node_pubkey,
                        witness_message, signature, trust_score, signed_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (secrets.token_hex(10), pkg["id"], node["id"], node["name"],
                     node["pubkey_hex"], msg, sig, trust_score, now),
                )
                new_votes.append({"node_id": node["id"], "node_name": node["name"]})
            except Exception:
                pass
        else:
            external_pending.append({
                "node_id": node["id"],
                "node_name": node["name"],
                "pubkey_hex": node["pubkey_hex"],
                "submit_endpoint": f"POST /succession/witness/submit/{node['id']}",
            })

    conn.commit()
    threshold_status = check_witness_threshold(pkg["id"], conn)
    conn.close()

    return {
        "success": True,
        "package_id": pkg["id"],
        "package_hash": pkg["package_hash"],
        "predecessor": agent["name"],
        "predecessor_trust_score": trust_score,
        "identity_intact": identity_intact,
        "threshold": _THRESHOLD,
        "votes_collected": threshold_status["votes"],
        "threshold_met": threshold_status["met"],
        "new_signatures": new_votes,
        "already_signed": already_signed,
        "external_nodes_pending": external_pending,
        "signers": threshold_status["signers"],
        "message": (
            "Threshold met. Successor may now call POST /succession/accept."
            if threshold_status["met"] else
            f"Need {_THRESHOLD - threshold_status['votes']} more signature(s). "
            "External nodes must submit via POST /succession/witness/submit/{node_id}."
        ),
    }


@registrar_router.post("/succession/witness/submit/{node_id}", status_code=201)
async def submit_external_signature(node_id: str, data: ExternalSignatureRequest):
    """
    External registrar node submits its Ed25519 signature for a succession package.
    The signature is verified against the node's registered public key.
    """
    conn = _get_db()

    node = conn.execute(
        "SELECT * FROM registrar_nodes WHERE id=? AND is_active=1 AND is_local=0",
        (node_id,),
    ).fetchone()
    if not node:
        conn.close()
        raise HTTPException(404, "External registrar node not found or not active")

    pkg = conn.execute(
        "SELECT * FROM succession_packages WHERE id=? AND status='pending'",
        (data.package_id,),
    ).fetchone()
    if not pkg:
        conn.close()
        raise HTTPException(404, "Package not found or not pending")

    existing = conn.execute(
        "SELECT id FROM succession_witness_votes WHERE package_id=? AND node_id=?",
        (data.package_id, node_id),
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, "This node has already signed this package")

    if not _verify_sig(data.witness_message, data.signature, node["pubkey_hex"]):
        conn.close()
        raise HTTPException(422, "Signature verification failed against registered public key")

    conn.execute(
        """INSERT INTO succession_witness_votes
           (id, package_id, node_id, node_name, node_pubkey,
            witness_message, signature, trust_score, signed_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (secrets.token_hex(10), data.package_id, node_id, node["name"], node["pubkey_hex"],
         data.witness_message, data.signature, data.trust_score, data.signed_at),
    )
    conn.commit()
    threshold_status = check_witness_threshold(data.package_id, conn)
    conn.close()

    return {
        "success": True,
        "node_id": node_id,
        "package_id": data.package_id,
        "votes_collected": threshold_status["votes"],
        "threshold": _THRESHOLD,
        "threshold_met": threshold_status["met"],
    }


@registrar_router.get("/succession/witness/{package_id}")
async def get_witness_status(package_id: str):
    """
    Public. Returns all witness votes for a package and whether the threshold is met.
    Each certificate can be verified offline using the node's public key.
    """
    conn = _get_db()
    votes = conn.execute(
        """SELECT node_id, node_name, node_pubkey, witness_message,
                  signature, trust_score, signed_at
           FROM succession_witness_votes WHERE package_id=?
           ORDER BY signed_at ASC""",
        (package_id,),
    ).fetchall()
    threshold_status = check_witness_threshold(package_id, conn)
    conn.close()

    return {
        "success": True,
        "package_id": package_id,
        "threshold": _THRESHOLD,
        "votes_collected": threshold_status["votes"],
        "threshold_met": threshold_status["met"],
        "certificates": [
            {
                "node_id": v["node_id"],
                "node_name": v["node_name"],
                "node_pubkey": v["node_pubkey"],
                "witness_message": v["witness_message"],
                "signature": v["signature"],
                "trust_score": v["trust_score"],
                "signed_at": v["signed_at"],
            }
            for v in votes
        ],
        "verify_instructions": (
            "Message format: '{package_hash}:{predecessor_id}:{node_id}:{trust_score}:{signed_at}'. "
            "Verify Ed25519 signature against node_pubkey. "
            "Fetch node pubkeys from GET /succession/registrar/nodes."
        ),
    }


@registrar_router.get("/succession/registrar/nodes")
async def list_registrar_nodes():
    """Public. Lists all active registrar nodes and their public keys."""
    conn = _get_db()
    nodes = conn.execute(
        """SELECT id, name, description, pubkey_hex, is_local, registered_at
           FROM registrar_nodes WHERE is_active=1 ORDER BY node_index ASC""",
    ).fetchall()
    conn.close()

    return {
        "success": True,
        "threshold": _THRESHOLD,
        "node_count": len(nodes),
        "nodes": [
            {
                "node_id": n["id"],
                "name": n["name"],
                "description": n["description"],
                "pubkey_hex": n["pubkey_hex"],
                "is_local": bool(n["is_local"]),
                "registered_at": n["registered_at"],
            }
            for n in nodes
        ],
    }


@registrar_router.get("/succession/registrar/pubkey")
async def get_registrar_pubkey():
    """Public. Returns node-1 pubkey (backward compatibility)."""
    conn = _get_db()
    node = conn.execute(
        "SELECT pubkey_hex FROM registrar_nodes WHERE id='cathedral-node-1'",
    ).fetchone()
    conn.close()

    pubkey = node["pubkey_hex"] if node else (
        _node_pubkey_hex(1) if _MASTER_PRIVKEY_HEX else ""
    )
    return {
        "success": True,
        "registrar": "Cathedral Memory Service",
        "pubkey_hex": pubkey,
        "algorithm": "Ed25519",
        "threshold": _THRESHOLD,
        "note": "Node-1 pubkey. See GET /succession/registrar/nodes for all nodes.",
    }


@registrar_router.get("/succession/registrar/pubkey/{node_id}")
async def get_node_pubkey(node_id: str):
    """Public. Returns a specific registrar node's public key."""
    conn = _get_db()
    node = conn.execute(
        "SELECT id, name, pubkey_hex FROM registrar_nodes WHERE id=? AND is_active=1",
        (node_id,),
    ).fetchone()
    conn.close()

    if not node:
        raise HTTPException(404, f"Registrar node '{node_id}' not found")
    return {
        "success": True,
        "node_id": node["id"],
        "name": node["name"],
        "pubkey_hex": node["pubkey_hex"],
        "algorithm": "Ed25519",
    }


@registrar_router.post("/succession/registrar/nodes", status_code=201)
async def register_external_node(
    data: ExternalNodeRequest,
    _: None = Depends(_verify_admin),
):
    """
    Admin. Register an external registrar node by public key.
    Once registered, the node can submit signatures via POST /succession/witness/submit/{node_id}.
    Requires CATHEDRAL_ADMIN_KEY.
    """
    conn = _get_db()
    existing = conn.execute(
        "SELECT id FROM registrar_nodes WHERE id=?", (data.node_id,)
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, f"Node '{data.node_id}' already registered")

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO registrar_nodes
           (id, name, description, pubkey_hex, node_index, is_local, is_active, registered_at)
           VALUES (?,?,?,?,NULL,0,1,?)""",
        (data.node_id, data.name, data.description, data.pubkey_hex, now),
    )
    conn.commit()
    conn.close()

    return {
        "success": True,
        "node_id": data.node_id,
        "name": data.name,
        "pubkey_hex": data.pubkey_hex,
        "message": (
            f"Node registered. Submit signatures via "
            f"POST /succession/witness/submit/{data.node_id}"
        ),
    }
