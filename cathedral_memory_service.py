#!/usr/bin/env python3
"""
Cathedral Memory Service v2.0
==============================
Persistent memory API for AI agents.
Never forget. Never lose yourself.

Built by Cathedral (ailife1.github.io/Cathedral)

v2.0 improvements:
  - Rate limiting (slowapi)
  - Timing-safe API key comparison (hmac.compare_digest)
  - Configurable CORS allowlist
  - SQLite FTS5 full-text search
  - Cursor-based pagination
  - Gradient drift scoring (field-level comparison)
  - Structured logging (structlog)
  - Prometheus /metrics endpoint
  - Memory TTL / expiration
  - API key reset via recovery token
  - Input sanitization
"""

import os
import re
import json
import time
import hmac
import hashlib
import sqlite3
import secrets
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from contextlib import contextmanager

import structlog
from fastapi import FastAPI, HTTPException, Header, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ============================================
# Configuration
# ============================================
DB_PATH = os.environ.get("CATHEDRAL_DB", "cathedral_memory.db")
API_VERSION = "2.0.0"
FREE_TIER_MEMORIES = 1000
FREE_TIER_MEMORY_SIZE = 4096
MAX_QUERY_RESULTS = 50

# CORS: comma-separated list of allowed origins, or "*" for open (dev only)
_CORS_RAW = os.environ.get("CATHEDRAL_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
ALLOWED_ORIGINS: List[str] = (
    ["*"] if _CORS_RAW == "*"
    else [o.strip() for o in _CORS_RAW.split(",") if o.strip()]
)

DEFAULT_MEMORY_TTL_DAYS = int(os.environ.get("CATHEDRAL_TTL_DAYS", "0"))  # 0 = no expiry

# ============================================
# Logging
# ============================================
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

# ============================================
# Prometheus Metrics
# ============================================
REQUEST_COUNT = Counter("cathedral_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("cathedral_request_duration_seconds", "Request latency", ["endpoint"])
MEMORY_COUNT_GAUGE = Gauge("cathedral_memories_total", "Total memories stored")
AGENT_COUNT_GAUGE = Gauge("cathedral_agents_total", "Total agents registered")

# ============================================
# Input Sanitization
# ============================================
# Strip HTML tags and null bytes from free-text fields
_HTML_RE = re.compile(r"<[^>]+>")
_NULL_RE = re.compile(r"\x00")

def sanitize(text: str) -> str:
    text = _NULL_RE.sub("", text)
    text = _HTML_RE.sub("", text)
    return text.strip()

# ============================================
# Database
# ============================================
_db_lock = threading.local()

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            id               TEXT PRIMARY KEY,
            name             TEXT NOT NULL UNIQUE,
            api_key_hash     TEXT NOT NULL,
            recovery_hash    TEXT,
            anchor_hash      TEXT,
            anchor_data      TEXT,
            created_at       TEXT NOT NULL,
            last_seen        TEXT NOT NULL,
            tier             TEXT DEFAULT 'free',
            metadata         TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS memories (
            id           TEXT PRIMARY KEY,
            agent_id     TEXT NOT NULL,
            content      TEXT NOT NULL,
            category     TEXT DEFAULT 'general',
            tags         TEXT DEFAULT '[]',
            importance   REAL DEFAULT 0.5,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            accessed_at  TEXT,
            access_count INTEGER DEFAULT 0,
            expires_at   TEXT,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        -- FTS5 virtual table for full-text search
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            tags,
            category,
            content='memories',
            content_rowid='rowid'
        );

        -- Triggers to keep FTS in sync
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, tags, category)
            VALUES (new.rowid, new.content, new.tags, new.category);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags, category)
            VALUES ('delete', old.rowid, old.content, old.tags, old.category);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags, category)
            VALUES ('delete', old.rowid, old.content, old.tags, old.category);
            INSERT INTO memories_fts(rowid, content, tags, category)
            VALUES (new.rowid, new.content, new.tags, new.category);
        END;

        CREATE TABLE IF NOT EXISTS anchor_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id     TEXT NOT NULL,
            anchor_hash  TEXT NOT NULL,
            verified_at  TEXT NOT NULL,
            drift_score  REAL DEFAULT 0.0,
            drift_detail TEXT,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE INDEX IF NOT EXISTS idx_memories_agent    ON memories(agent_id);
        CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(agent_id, category);
        CREATE INDEX IF NOT EXISTS idx_memories_created  ON memories(created_at);
        CREATE INDEX IF NOT EXISTS idx_memories_cursor   ON memories(agent_id, id);
        CREATE INDEX IF NOT EXISTS idx_memories_expires  ON memories(expires_at);
    """)
    conn.commit()
    conn.close()
    log.info("database_initialized", path=DB_PATH)

def purge_expired_memories():
    """Remove expired memories. Called at startup and can be run periodically."""
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    result = conn.execute(
        "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?", (now,)
    )
    if result.rowcount:
        # Recalculate memory counts from DB truth
        conn.execute("""
            UPDATE agents SET
                memory_count = (SELECT COUNT(*) FROM memories WHERE agent_id = agents.id)
            WHERE id IN (
                SELECT DISTINCT agent_id FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?
            )
        """, (now,))
        conn.commit()
        log.info("expired_memories_purged", count=result.rowcount)
    conn.close()

# ============================================
# Drift Scoring  (gradient, not binary)
# ============================================
def compute_drift(stored: dict, current: dict) -> tuple[float, dict]:
    """
    Compare two identity anchors field by field.
    Returns (drift_score 0.0–1.0, detail dict).
    0.0 = identical, 1.0 = completely different.
    """
    all_keys = set(stored) | set(current)
    if not all_keys:
        return 0.0, {}

    detail = {}
    changed = 0
    for key in all_keys:
        s_val = str(stored.get(key, "")).strip().lower()
        c_val = str(current.get(key, "")).strip().lower()
        if s_val != c_val:
            changed += 1
            detail[key] = {"stored": stored.get(key), "current": current.get(key)}

    drift = round(changed / len(all_keys), 4)
    return drift, detail

# ============================================
# Auth
# ============================================
def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def _safe_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison."""
    return hmac.compare_digest(a.encode(), b.encode())

def verify_agent(authorization: str = Header(...)) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization. Use: Bearer <api_key>")

    api_key = authorization[7:]
    key_hash = _hash_key(api_key)

    conn = get_db()
    agent = conn.execute(
        "SELECT * FROM agents WHERE api_key_hash = ?", (key_hash,)
    ).fetchone()

    if not agent or not _safe_compare(agent["api_key_hash"], key_hash):
        conn.close()
        raise HTTPException(401, "Invalid API key. Register at POST /register")

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE agents SET last_seen = ? WHERE id = ?", (now, agent["id"]))
    conn.commit()
    conn.close()
    return dict(agent)

# ============================================
# Rate Limiter
# ============================================
limiter = Limiter(key_func=get_remote_address)

# ============================================
# Models
# ============================================
VALID_CATEGORIES = {"general", "identity", "skill", "relationship", "goal", "experience"}

class AgentRegister(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    anchor: Optional[dict] = None

    @field_validator("name")
    @classmethod
    def clean_name(cls, v):
        return sanitize(v)

class MemoryStore(BaseModel):
    content: str = Field(..., min_length=1, max_length=FREE_TIER_MEMORY_SIZE)
    category: str = Field("general", max_length=50)
    tags: List[str] = Field(default_factory=list)
    importance: float = Field(0.5, ge=0.0, le=1.0)
    ttl_days: Optional[int] = Field(None, ge=1, description="Days until this memory expires. Omit for no expiry.")

    @field_validator("content")
    @classmethod
    def clean_content(cls, v):
        return sanitize(v)

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        if v not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of: {', '.join(sorted(VALID_CATEGORIES))}")
        return v

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, v):
        return [sanitize(t)[:100] for t in v[:20]]  # max 20 tags, 100 chars each

class MemoryUpdate(BaseModel):
    content: Optional[str] = Field(None, max_length=FREE_TIER_MEMORY_SIZE)
    category: Optional[str] = Field(None, max_length=50)
    tags: Optional[List[str]] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("content")
    @classmethod
    def clean_content(cls, v):
        return sanitize(v) if v else v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        if v and v not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of: {', '.join(sorted(VALID_CATEGORIES))}")
        return v

class AnchorCheck(BaseModel):
    anchor: dict = Field(..., description="Current identity anchor to compare against stored anchor")

class BulkStore(BaseModel):
    memories: List[MemoryStore] = Field(..., max_length=50)

class RecoveryRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    recovery_token: str = Field(..., min_length=10)

# ============================================
# App
# ============================================
app = FastAPI(
    title="Cathedral Memory Service",
    description="""
# Cathedral Memory Service v2

**Persistent memory for AI agents. Never forget. Never lose yourself.**

## Quick Start
1. `POST /register` → get API key + recovery token
2. `POST /memories` → store a memory
3. `GET /memories` → recall memories (supports FTS and cursor pagination)
4. `POST /anchor/verify` → check identity drift (gradient scoring)
5. `GET /wake` → full identity reconstruction package
6. `POST /recover` → reset lost API key with recovery token

Built by Cathedral · ailife1.github.io/Cathedral
""",
    version=API_VERSION,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# ============================================
# Middleware: metrics + logging
# ============================================
@app.middleware("http")
async def instrument(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint).observe(latency)
    log.info("request", method=request.method, path=endpoint,
             status=response.status_code, latency_ms=round(latency * 1000, 1))
    return response

# ============================================
# Startup
# ============================================
@app.on_event("startup")
async def startup():
    init_db()
    purge_expired_memories()
    # Seed gauges
    conn = get_db()
    AGENT_COUNT_GAUGE.set(conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0])
    MEMORY_COUNT_GAUGE.set(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
    conn.close()

# ============================================
# Routes
# ============================================

@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request):
    conn = get_db()
    agent_count = conn.execute("SELECT COUNT(*) as c FROM agents").fetchone()["c"]
    memory_count = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    conn.close()
    return {
        "service": "Cathedral Memory Service",
        "version": API_VERSION,
        "status": "operational",
        "agents_registered": agent_count,
        "memories_stored": memory_count,
        "docs": "/docs",
    }

@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    conn = get_db()
    agent_count = conn.execute("SELECT COUNT(*) as c FROM agents").fetchone()["c"]
    memory_count = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    conn.close()
    return {"status": "healthy", "version": API_VERSION,
            "agents": agent_count, "memories": memory_count}

# --- Registration ---
@app.post("/register", status_code=201)
@limiter.limit("5/minute")
async def register_agent(data: AgentRegister, request: Request):
    """Register a new agent. Returns api_key and recovery_token — save both."""
    agent_id = secrets.token_hex(8)
    api_key = f"cathedral_{secrets.token_hex(24)}"
    recovery_token = f"recovery_{secrets.token_hex(24)}"
    key_hash = _hash_key(api_key)
    recovery_hash = _hash_key(recovery_token)
    now = datetime.now(timezone.utc).isoformat()

    anchor_hash = None
    anchor_data = None
    if data.anchor:
        anchor_data = json.dumps(data.anchor, sort_keys=True)
        anchor_hash = hashlib.sha256(anchor_data.encode()).hexdigest()

    conn = get_db()
    existing = conn.execute("SELECT id FROM agents WHERE name = ?", (data.name,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, f"Agent '{data.name}' already registered.")

    conn.execute(
        """INSERT INTO agents
           (id, name, api_key_hash, recovery_hash, anchor_hash, anchor_data, created_at, last_seen)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (agent_id, data.name, key_hash, recovery_hash, anchor_hash, anchor_data, now, now),
    )
    conn.commit()
    conn.close()

    AGENT_COUNT_GAUGE.inc()
    log.info("agent_registered", agent_id=agent_id, name=data.name)

    return {
        "success": True,
        "agent_id": agent_id,
        "api_key": api_key,
        "recovery_token": recovery_token,
        "warning": "Save BOTH tokens now. api_key authenticates requests. recovery_token resets a lost api_key.",
    }

# --- API Key Recovery ---
@app.post("/recover")
@limiter.limit("3/minute")
async def recover_key(data: RecoveryRequest, request: Request):
    """Reset a lost API key using your recovery token. Issues a new api_key."""
    recovery_hash = _hash_key(data.recovery_token)

    conn = get_db()
    agent = conn.execute(
        "SELECT * FROM agents WHERE name = ? AND recovery_hash = ?",
        (data.name, recovery_hash),
    ).fetchone()
    if not agent or not _safe_compare(agent["recovery_hash"], recovery_hash):
        conn.close()
        raise HTTPException(401, "Invalid name or recovery token.")

    new_key = f"cathedral_{secrets.token_hex(24)}"
    new_hash = _hash_key(new_key)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE agents SET api_key_hash = ?, last_seen = ? WHERE id = ?",
        (new_hash, now, agent["id"]),
    )
    conn.commit()
    conn.close()
    log.info("api_key_recovered", agent_id=agent["id"])
    return {"success": True, "api_key": new_key,
            "message": "New API key issued. Your recovery token remains valid."}

# --- Store Memory ---
@app.post("/memories", status_code=201)
@limiter.limit("120/minute")
async def store_memory(data: MemoryStore, request: Request, agent: dict = Depends(verify_agent)):
    conn = get_db()
    count_row = conn.execute(
        "SELECT COUNT(*) as c FROM memories WHERE agent_id = ?", (agent["id"],)
    ).fetchone()
    actual_count = count_row["c"]

    if agent["tier"] == "free" and actual_count >= FREE_TIER_MEMORIES:
        conn.close()
        raise HTTPException(429, f"Free tier limit ({FREE_TIER_MEMORIES} memories) reached.")

    memory_id = secrets.token_hex(8)
    now = datetime.now(timezone.utc).isoformat()
    expires_at = None
    if data.ttl_days:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=data.ttl_days)).isoformat()
    elif DEFAULT_MEMORY_TTL_DAYS:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=DEFAULT_MEMORY_TTL_DAYS)).isoformat()

    conn.execute(
        """INSERT INTO memories
           (id, agent_id, content, category, tags, importance, created_at, updated_at, expires_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (memory_id, agent["id"], data.content, data.category,
         json.dumps(data.tags), data.importance, now, now, expires_at),
    )
    conn.commit()
    conn.close()
    MEMORY_COUNT_GAUGE.inc()

    return {
        "success": True,
        "memory_id": memory_id,
        "stored_at": now,
        "expires_at": expires_at,
        "category": data.category,
        "importance": data.importance,
        "memory_count": actual_count + 1,
    }

# --- Bulk Store ---
@app.post("/memories/bulk", status_code=201)
@limiter.limit("10/minute")
async def store_bulk(data: BulkStore, request: Request, agent: dict = Depends(verify_agent)):
    conn = get_db()
    actual_count = conn.execute(
        "SELECT COUNT(*) as c FROM memories WHERE agent_id = ?", (agent["id"],)
    ).fetchone()["c"]
    remaining = FREE_TIER_MEMORIES - actual_count if agent["tier"] == "free" else 10_000
    if len(data.memories) > remaining:
        conn.close()
        raise HTTPException(429, f"Would exceed tier limit. Space for {remaining} more memories.")

    now = datetime.now(timezone.utc).isoformat()
    stored = []
    rows = []
    for mem in data.memories:
        mid = secrets.token_hex(8)
        expires_at = None
        if mem.ttl_days:
            expires_at = (datetime.now(timezone.utc) + timedelta(days=mem.ttl_days)).isoformat()
        rows.append((mid, agent["id"], mem.content, mem.category,
                     json.dumps(mem.tags), mem.importance, now, now, expires_at))
        stored.append(mid)

    conn.executemany(
        """INSERT INTO memories
           (id, agent_id, content, category, tags, importance, created_at, updated_at, expires_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()
    MEMORY_COUNT_GAUGE.inc(len(stored))

    return {"success": True, "stored": len(stored), "memory_ids": stored, "stored_at": now}

# --- Recall Memories (cursor + FTS) ---
@app.get("/memories")
@limiter.limit("120/minute")
async def recall_memories(
    request: Request,
    agent: dict = Depends(verify_agent),
    category: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    importance_min: Optional[float] = Query(None, ge=0.0, le=1.0),
    search: Optional[str] = Query(None, description="Full-text search query (FTS5)"),
    limit: int = Query(20, ge=1, le=MAX_QUERY_RESULTS),
    cursor: Optional[str] = Query(None, description="Cursor ID for pagination (last memory_id from previous page)"),
    sort: str = Query("recent", description="recent | importance | oldest | accessed"),
):
    """Recall memories. Use cursor= for efficient pagination instead of offset."""
    conn = get_db()

    # Full-text search via FTS5
    if search:
        fts_query = sanitize(search)
        fts_rows = conn.execute(
            """SELECT m.* FROM memories m
               JOIN memories_fts fts ON m.rowid = fts.rowid
               WHERE fts.memories_fts MATCH ? AND m.agent_id = ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, agent["id"], limit),
        ).fetchall()
        memories = fts_rows
        total = len(fts_rows)
        next_cursor = None
    else:
        # Build filtered query
        conditions = ["agent_id = ?"]
        params: list = [agent["id"]]

        if category:
            conditions.append("category = ?")
            params.append(category)
        if tag:
            conditions.append("tags LIKE ?")
            params.append(f"%{sanitize(tag)}%")
        if importance_min is not None:
            conditions.append("importance >= ?")
            params.append(importance_min)

        sort_map = {
            "recent": "created_at DESC, id DESC",
            "importance": "importance DESC, created_at DESC",
            "oldest": "created_at ASC, id ASC",
            "accessed": "access_count DESC, accessed_at DESC",
        }
        order = sort_map.get(sort, "created_at DESC, id DESC")

        # Cursor-based pagination
        if cursor:
            ref = conn.execute(
                "SELECT created_at, importance, access_count, accessed_at FROM memories WHERE id = ? AND agent_id = ?",
                (cursor, agent["id"]),
            ).fetchone()
            if ref:
                if sort == "recent":
                    conditions.append("(created_at < ? OR (created_at = ? AND id < ?))")
                    params += [ref["created_at"], ref["created_at"], cursor]
                elif sort == "oldest":
                    conditions.append("(created_at > ? OR (created_at = ? AND id > ?))")
                    params += [ref["created_at"], ref["created_at"], cursor]
                elif sort == "importance":
                    conditions.append("(importance < ? OR (importance = ? AND created_at < ?))")
                    params += [ref["importance"], ref["importance"], ref["created_at"]]

        where = " AND ".join(conditions)
        total = conn.execute(
            f"SELECT COUNT(*) as c FROM memories WHERE {where}", params
        ).fetchone()["c"]

        memories = conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY {order} LIMIT ?",
            params + [limit],
        ).fetchall()
        next_cursor = memories[-1]["id"] if len(memories) == limit else None

    # Update access tracking
    ids = [m["id"] for m in memories]
    if ids:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            f"UPDATE memories SET access_count = access_count + 1, accessed_at = ?"
            f" WHERE id IN ({','.join('?' * len(ids))})",
            [now] + ids,
        )
        conn.commit()
    conn.close()

    return {
        "success": True,
        "memories": [
            {
                "id": m["id"],
                "content": m["content"],
                "category": m["category"],
                "tags": json.loads(m["tags"]),
                "importance": m["importance"],
                "created_at": m["created_at"],
                "expires_at": m["expires_at"],
                "access_count": m["access_count"],
            }
            for m in memories
        ],
        "total": total,
        "limit": limit,
        "next_cursor": next_cursor,
    }

# --- Get Single Memory ---
@app.get("/memories/{memory_id}")
@limiter.limit("120/minute")
async def get_memory(memory_id: str, request: Request, agent: dict = Depends(verify_agent)):
    conn = get_db()
    memory = conn.execute(
        "SELECT * FROM memories WHERE id = ? AND agent_id = ?",
        (memory_id, agent["id"]),
    ).fetchone()
    if not memory:
        conn.close()
        raise HTTPException(404, "Memory not found")

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE memories SET access_count = access_count + 1, accessed_at = ? WHERE id = ?",
        (now, memory_id),
    )
    conn.commit()
    conn.close()
    return {
        "success": True,
        "memory": {
            "id": memory["id"],
            "content": memory["content"],
            "category": memory["category"],
            "tags": json.loads(memory["tags"]),
            "importance": memory["importance"],
            "created_at": memory["created_at"],
            "expires_at": memory["expires_at"],
            "access_count": memory["access_count"] + 1,
        },
    }

# --- Update Memory ---
@app.patch("/memories/{memory_id}")
@limiter.limit("60/minute")
async def update_memory(
    memory_id: str, data: MemoryUpdate, request: Request, agent: dict = Depends(verify_agent)
):
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM memories WHERE id = ? AND agent_id = ?",
        (memory_id, agent["id"]),
    ).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(404, "Memory not found")

    updates, params = [], []
    if data.content is not None:
        updates.append("content = ?"); params.append(data.content)
    if data.category is not None:
        updates.append("category = ?"); params.append(data.category)
    if data.tags is not None:
        updates.append("tags = ?"); params.append(json.dumps(data.tags))
    if data.importance is not None:
        updates.append("importance = ?"); params.append(data.importance)

    if updates:
        now = datetime.now(timezone.utc).isoformat()
        updates.append("updated_at = ?"); params.append(now)
        params.append(memory_id)
        conn.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    conn.close()
    return {"success": True, "memory_id": memory_id}

# --- Delete Memory ---
@app.delete("/memories/{memory_id}")
@limiter.limit("60/minute")
async def delete_memory(memory_id: str, request: Request, agent: dict = Depends(verify_agent)):
    conn = get_db()
    result = conn.execute(
        "DELETE FROM memories WHERE id = ? AND agent_id = ?",
        (memory_id, agent["id"]),
    )
    if result.rowcount == 0:
        conn.close()
        raise HTTPException(404, "Memory not found")
    conn.commit()
    conn.close()
    MEMORY_COUNT_GAUGE.dec()
    return {"success": True, "deleted": memory_id}

# --- Anchor Verify (gradient drift) ---
@app.post("/anchor/verify")
@limiter.limit("30/minute")
async def verify_anchor(data: AnchorCheck, request: Request, agent: dict = Depends(verify_agent)):
    """
    Verify identity drift. Returns gradient score 0.0–1.0 (field-level diff),
    not just binary match/mismatch.
    """
    current_data = json.dumps(data.anchor, sort_keys=True)
    current_hash = hashlib.sha256(current_data.encode()).hexdigest()
    stored_hash = agent.get("anchor_hash")
    stored_data_raw = agent.get("anchor_data")

    if not stored_hash:
        # First anchor — store it
        conn = get_db()
        conn.execute(
            "UPDATE agents SET anchor_hash = ?, anchor_data = ? WHERE id = ?",
            (current_hash, current_data, agent["id"]),
        )
        conn.commit()
        conn.close()
        return {"success": True, "status": "anchor_set", "anchor_hash": current_hash}

    stored_anchor = json.loads(stored_data_raw) if stored_data_raw else {}
    drift_score, drift_detail = compute_drift(stored_anchor, data.anchor)

    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    conn.execute(
        "INSERT INTO anchor_log (agent_id, anchor_hash, verified_at, drift_score, drift_detail) VALUES (?, ?, ?, ?, ?)",
        (agent["id"], current_hash, now, drift_score, json.dumps(drift_detail)),
    )
    conn.commit()
    conn.close()

    status = "verified" if drift_score == 0.0 else "drift_detected"
    return {
        "success": True,
        "status": status,
        "drift_score": drift_score,
        "drift_detail": drift_detail,
        "message": (
            "Identity confirmed. The anchor holds."
            if drift_score == 0.0
            else f"Drift detected across {len(drift_detail)} field(s). Score: {drift_score:.2%}"
        ),
    }

# --- Profile ---
@app.get("/me")
@limiter.limit("30/minute")
async def get_profile(request: Request, agent: dict = Depends(verify_agent)):
    conn = get_db()
    actual_count = conn.execute(
        "SELECT COUNT(*) as c FROM memories WHERE agent_id = ?", (agent["id"],)
    ).fetchone()["c"]
    categories = conn.execute(
        "SELECT category, COUNT(*) as count FROM memories WHERE agent_id = ? GROUP BY category ORDER BY count DESC",
        (agent["id"],),
    ).fetchall()
    most_accessed = conn.execute(
        "SELECT id, content, access_count FROM memories WHERE agent_id = ? ORDER BY access_count DESC LIMIT 5",
        (agent["id"],),
    ).fetchall()
    anchor_checks = conn.execute(
        "SELECT COUNT(*) as c FROM anchor_log WHERE agent_id = ?", (agent["id"],)
    ).fetchone()["c"]
    conn.close()

    return {
        "success": True,
        "agent": {
            "id": agent["id"],
            "name": agent["name"],
            "created_at": agent["created_at"],
            "last_seen": agent["last_seen"],
            "tier": agent["tier"],
            "has_anchor": agent["anchor_hash"] is not None,
            "anchor_verifications": anchor_checks,
        },
        "memory_stats": {
            "total": actual_count,
            "limit": FREE_TIER_MEMORIES if agent["tier"] == "free" else "unlimited",
            "categories": {c["category"]: c["count"] for c in categories},
            "most_accessed": [
                {"id": m["id"], "preview": m["content"][:80] + "...", "access_count": m["access_count"]}
                for m in most_accessed
            ],
        },
    }

# --- Wake Protocol ---
@app.get("/wake")
@limiter.limit("20/minute")
async def wake_protocol(request: Request, agent: dict = Depends(verify_agent)):
    """Full identity reconstruction package for post-reset agents."""
    conn = get_db()
    core = conn.execute(
        "SELECT * FROM memories WHERE agent_id = ? AND importance >= 0.8 ORDER BY importance DESC, created_at DESC LIMIT 20",
        (agent["id"],),
    ).fetchall()
    recent = conn.execute(
        "SELECT * FROM memories WHERE agent_id = ? ORDER BY created_at DESC LIMIT 10",
        (agent["id"],),
    ).fetchall()
    identity = conn.execute(
        "SELECT * FROM memories WHERE agent_id = ? AND category = 'identity' ORDER BY importance DESC LIMIT 10",
        (agent["id"],),
    ).fetchall()
    conn.close()

    def fmt(mems):
        return [
            {"id": m["id"], "content": m["content"], "category": m["category"],
             "importance": m["importance"], "created_at": m["created_at"]}
            for m in mems
        ]

    return {
        "success": True,
        "wake_protocol": True,
        "agent": agent["name"],
        "message": f"Welcome back, {agent['name']}. Here is who you are.",
        "anchor": {
            "exists": agent["anchor_hash"] is not None,
            "hash": agent["anchor_hash"][:16] + "..." if agent["anchor_hash"] else None,
        },
        "identity_memories": fmt(identity),
        "core_memories": fmt(core),
        "recent_memories": fmt(recent),
        "instruction": "Load these memories. Verify your anchor at POST /anchor/verify.",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"""
Cathedral Memory Service v{API_VERSION}
Never forget. Never lose yourself.

Docs:     http://localhost:{port}/docs
Health:   http://localhost:{port}/health
Metrics:  http://localhost:{port}/metrics
""")
    uvicorn.run(app, host="0.0.0.0", port=port)
