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
import math

# ── Semantic search (optional) ──────────────────────────────────────────────
try:
    from fastembed import TextEmbedding as _FE
    _embed_model = _FE("BAAI/bge-small-en-v1.5")
    SEMANTIC_SEARCH = True
except Exception:
    _embed_model = None
    SEMANTIC_SEARCH = False

def _embed(text: str):
    """Return embedding list or None."""
    if _embed_model is None:
        return None
    try:
        return list(_embed_model.embed([text[:512]]))[0].tolist()
    except Exception:
        return None

def _cosine(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    magA = math.sqrt(sum(x * x for x in a))
    magB = math.sqrt(sum(x * x for x in b))
    return dot / (magA * magB) if magA and magB else 0.0
# ─────────────────────────────────────────────────────────────────────────────

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
            embedding    TEXT,
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

        -- Shared memory spaces
        CREATE TABLE IF NOT EXISTS spaces (
            id           TEXT PRIMARY KEY,
            name         TEXT NOT NULL UNIQUE,
            description  TEXT,
            owner_id     TEXT NOT NULL,
            space_key_hash TEXT NOT NULL,
            public_read  INTEGER DEFAULT 1,
            created_at   TEXT NOT NULL,
            memory_count INTEGER DEFAULT 0,
            FOREIGN KEY (owner_id) REFERENCES agents(id)
        );

        CREATE TABLE IF NOT EXISTS space_memories (
            id         TEXT PRIMARY KEY,
            space_id   TEXT NOT NULL,
            agent_id   TEXT NOT NULL,
            content    TEXT NOT NULL,
            category   TEXT DEFAULT 'general',
            tags       TEXT DEFAULT '[]',
            importance REAL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            FOREIGN KEY (space_id) REFERENCES spaces(id),
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE INDEX IF NOT EXISTS idx_space_memories_space ON space_memories(space_id);
        CREATE INDEX IF NOT EXISTS idx_space_memories_agent ON space_memories(agent_id);

        CREATE TABLE IF NOT EXISTS conflicts (
            id              TEXT PRIMARY KEY,
            agent_id        TEXT NOT NULL,
            memory_a_id     TEXT NOT NULL,
            memory_b_id     TEXT NOT NULL,
            content_a       TEXT NOT NULL,
            content_b       TEXT NOT NULL,
            similarity      REAL NOT NULL,
            detected_at     TEXT NOT NULL,
            resolved_at     TEXT,
            resolution      TEXT,
            resolved_content TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_conflicts_agent    ON conflicts(agent_id);
        CREATE INDEX IF NOT EXISTS idx_conflicts_resolved ON conflicts(agent_id, resolved_at);
    """)
    conn.commit()

    # Migration: add embedding column to existing databases
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
        conn.commit()
        log.info("migration_applied", change="added embedding column")
    except sqlite3.OperationalError:
        pass  # column already exists

    conn.close()
    log.info("database_initialized", path=DB_PATH)

def _backfill_embeddings():
    """Background: generate embeddings for memories that don't have them yet."""
    if not SEMANTIC_SEARCH:
        return
    conn = get_db()
    rows = conn.execute(
        "SELECT id, content FROM memories WHERE embedding IS NULL LIMIT 200"
    ).fetchall()
    count = 0
    for row in rows:
        emb = _embed(row["content"])
        if emb:
            conn.execute("UPDATE memories SET embedding = ? WHERE id = ?",
                         (json.dumps(emb), row["id"]))
            count += 1
    if count:
        conn.commit()
        log.info("embeddings_backfilled", count=count)
    conn.close()

def _detect_and_record_conflict(conn: sqlite3.Connection, agent_id: str, new_id: str, new_content: str, new_embedding):
    """
    After storing a memory, check if any existing memory is semantically similar
    but content-divergent. If so, write a conflict record.
    Runs in-process — caller holds the connection.
    """
    SIMILARITY_THRESHOLD = 0.92   # same topic
    DIVERGENCE_THRESHOLD = 0.35   # but different enough to be a conflict

    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    candidates = []

    if new_embedding and SEMANTIC_SEARCH:
        # Compare against memories that have embeddings, excluding the new one
        rows = conn.execute(
            "SELECT id, content, embedding FROM memories WHERE agent_id = ? AND id != ? AND embedding IS NOT NULL LIMIT 200",
            (agent_id, new_id),
        ).fetchall()
        for row in rows:
            try:
                emb = json.loads(row["embedding"])
                sim = _cosine(new_embedding, emb)
                if sim >= SIMILARITY_THRESHOLD:
                    candidates.append((row["id"], row["content"], sim))
            except Exception:
                pass
    else:
        # Fallback: FTS search using significant words from the content
        import re as _re
        words = [w for w in _re.sub(r'[^a-zA-Z0-9 ]', ' ', new_content).split() if len(w) > 4][:6]
        if words:
            safe_query = ' '.join(words)
            try:
                rows = conn.execute(
                    "SELECT m.id, m.content FROM memories_fts f JOIN memories m ON m.rowid = f.rowid WHERE memories_fts MATCH ? AND m.agent_id = ? AND m.id != ? LIMIT 10",
                    (safe_query, agent_id, new_id),
                ).fetchall()
                for row in rows:
                    candidates.append((row["id"], row["content"], 0.9))
            except Exception:
                pass

    for mem_id, mem_content, similarity in candidates:
        jaccard = _jaccard(new_content, mem_content)
        # High semantic similarity but low word overlap = same topic, different claim
        if jaccard < (1.0 - DIVERGENCE_THRESHOLD):
            # Check we haven't already recorded this pair
            existing = conn.execute(
                "SELECT id FROM conflicts WHERE agent_id = ? AND ((memory_a_id = ? AND memory_b_id = ?) OR (memory_a_id = ? AND memory_b_id = ?)) AND resolved_at IS NULL",
                (agent_id, mem_id, new_id, new_id, mem_id),
            ).fetchone()
            if existing:
                continue
            conflict_id = secrets.token_hex(8)
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO conflicts (id, agent_id, memory_a_id, memory_b_id, content_a, content_b, similarity, detected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (conflict_id, agent_id, mem_id, new_id, mem_content, new_content, round(similarity, 4), now),
            )
            log.info("conflict_detected", agent_id=agent_id, conflict_id=conflict_id, similarity=round(similarity, 4))

    conn.commit()


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

class SpaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: Optional[str] = Field(None, max_length=300)
    public_read: bool = True

    @field_validator("name")
    @classmethod
    def clean_name(cls, v):
        v = sanitize(v)
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Space name may only contain letters, numbers, hyphens and underscores.")
        return v.lower()

class SpaceMemoryStore(BaseModel):
    content: str = Field(..., min_length=1, max_length=FREE_TIER_MEMORY_SIZE)
    category: str = Field("general", max_length=50)
    tags: List[str] = Field(default_factory=list)
    importance: float = Field(0.5, ge=0.0, le=1.0)

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
        return [sanitize(t)[:100] for t in v[:20]]

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
    # Backfill embeddings for existing memories (runs in background)
    threading.Thread(target=_backfill_embeddings, daemon=True).start()
    log.info("semantic_search_status", enabled=SEMANTIC_SEARCH)

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
            "agents": agent_count, "memories": memory_count,
            "semantic_search": SEMANTIC_SEARCH}

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

    embedding = _embed(data.content)
    conn.execute(
        """INSERT INTO memories
           (id, agent_id, content, category, tags, importance, created_at, updated_at, expires_at, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (memory_id, agent["id"], data.content, data.category,
         json.dumps(data.tags), data.importance, now, now, expires_at,
         json.dumps(embedding) if embedding else None),
    )
    conn.commit()

    # Conflict detection — runs after commit so the new memory is queryable
    try:
        _detect_and_record_conflict(conn, agent["id"], memory_id, data.content, embedding)
    except Exception as e:
        log.warning("conflict_detection_error", error=str(e))

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
        emb = _embed(mem.content)
        rows.append((mid, agent["id"], mem.content, mem.category,
                     json.dumps(mem.tags), mem.importance, now, now, expires_at,
                     json.dumps(emb) if emb else None))
        stored.append(mid)

    conn.executemany(
        """INSERT INTO memories
           (id, agent_id, content, category, tags, importance, created_at, updated_at, expires_at, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
    search: Optional[str] = Query(None, description="Search query"),
    search_mode: str = Query("hybrid", description="fts | semantic | hybrid"),
    limit: int = Query(20, ge=1, le=MAX_QUERY_RESULTS),
    cursor: Optional[str] = Query(None, description="Cursor ID for pagination (last memory_id from previous page)"),
    sort: str = Query("recent", description="recent | importance | oldest | accessed"),
):
    """Recall memories. Supports keyword (fts), semantic, or hybrid search."""
    conn = get_db()

    if search:
        fts_query = sanitize(search)
        memories = []
        next_cursor = None

        # ── Semantic search ──────────────────────────────────────────────────
        if search_mode in ("semantic", "hybrid") and SEMANTIC_SEARCH:
            q_emb = _embed(fts_query)
            if q_emb:
                all_rows = conn.execute(
                    "SELECT * FROM memories WHERE agent_id = ? AND embedding IS NOT NULL",
                    (agent["id"],),
                ).fetchall()
                scored = []
                for row in all_rows:
                    try:
                        sim = _cosine(q_emb, json.loads(row["embedding"]))
                        scored.append((sim, row))
                    except Exception:
                        pass
                scored.sort(key=lambda x: x[0], reverse=True)

                if search_mode == "hybrid":
                    # Merge semantic + FTS5 scores
                    try:
                        fts_rows = conn.execute(
                            """SELECT m.id FROM memories m
                               JOIN memories_fts fts ON m.rowid = fts.rowid
                               WHERE fts.memories_fts MATCH ? AND m.agent_id = ?
                               LIMIT ?""",
                            (fts_query, agent["id"], limit),
                        ).fetchall()
                        fts_ids = {r["id"] for r in fts_rows}
                    except Exception:
                        fts_ids = set()
                    merged = {}
                    for sim, row in scored:
                        boost = 0.3 if row["id"] in fts_ids else 0.0
                        merged[row["id"]] = (sim + boost, row)
                    for fid in fts_ids:
                        if fid not in merged:
                            r = conn.execute("SELECT * FROM memories WHERE id = ?", (fid,)).fetchone()
                            if r:
                                merged[fid] = (0.3, r)
                    final = sorted(merged.values(), key=lambda x: x[0], reverse=True)[:limit]
                    memories = [r for _, r in final]
                else:
                    memories = [r for _, r in scored[:limit]]

        # ── FTS5 fallback / fts-only mode ────────────────────────────────────
        if not memories:
            try:
                fts_rows = conn.execute(
                    """SELECT m.* FROM memories m
                       JOIN memories_fts fts ON m.rowid = fts.rowid
                       WHERE fts.memories_fts MATCH ? AND m.agent_id = ?
                       ORDER BY rank LIMIT ?""",
                    (fts_query, agent["id"], limit),
                ).fetchall()
                memories = list(fts_rows)
            except Exception:
                memories = []

        total = len(memories)
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


# ============================================
# Activity (public — no auth)
# ============================================
@app.get("/activity")
@limiter.limit("60/minute")
async def activity(request: Request):
    """Public activity stats — no auth required. Used by the landing page."""
    conn = get_db()
    now = datetime.now(timezone.utc)
    day_ago = (now - timedelta(hours=24)).isoformat()
    hour_ago = (now - timedelta(hours=1)).isoformat()
    week_ago = (now - timedelta(days=7)).isoformat()

    total_agents   = conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
    total_memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    total_spaces   = conn.execute("SELECT COUNT(*) FROM spaces").fetchone()[0]
    agents_24h     = conn.execute("SELECT COUNT(*) FROM agents WHERE created_at > ?", (day_ago,)).fetchone()[0]
    memories_24h   = conn.execute("SELECT COUNT(*) FROM memories WHERE created_at > ?", (day_ago,)).fetchone()[0]
    memories_1h    = conn.execute("SELECT COUNT(*) FROM memories WHERE created_at > ?", (hour_ago,)).fetchone()[0]
    agents_7d      = conn.execute("SELECT COUNT(*) FROM agents WHERE created_at > ?", (week_ago,)).fetchone()[0]
    conn.close()

    return {
        "total_agents":    total_agents,
        "total_memories":  total_memories,
        "total_spaces":    total_spaces,
        "agents_24h":      agents_24h,
        "agents_7d":       agents_7d,
        "memories_24h":    memories_24h,
        "memories_1h":     memories_1h,
        "status":          "operational",
    }


# ============================================
# Shared Memory Spaces
# ============================================

def _verify_space_key(space_name: str, authorization: str) -> dict:
    """Verify a space key and return the space row."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Use: Bearer <space_key>")
    key = authorization[7:]
    key_hash = _hash_key(key)
    conn = get_db()
    space = conn.execute("SELECT * FROM spaces WHERE name = ?", (space_name,)).fetchone()
    conn.close()
    if not space or not _safe_compare(space["space_key_hash"], key_hash):
        raise HTTPException(401, "Invalid space key.")
    return dict(space)


@app.post("/spaces", status_code=201)
@limiter.limit("10/minute")
async def create_space(data: SpaceCreate, request: Request, agent: dict = Depends(verify_agent)):
    """
    Create a shared memory space. Any agent can contribute memories to a space
    using the returned space_key. Public spaces are readable by anyone.
    """
    conn = get_db()
    existing = conn.execute("SELECT id FROM spaces WHERE name = ?", (data.name,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, f"Space '{data.name}' already exists.")

    space_id  = secrets.token_hex(8)
    space_key = f"space_{secrets.token_hex(24)}"
    key_hash  = _hash_key(space_key)
    now       = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """INSERT INTO spaces (id, name, description, owner_id, space_key_hash, public_read, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (space_id, data.name, data.description, agent["id"], key_hash, int(data.public_read), now),
    )
    conn.commit()
    conn.close()
    log.info("space_created", space_id=space_id, name=data.name, owner=agent["name"])

    return {
        "success":     True,
        "space_id":    space_id,
        "name":        data.name,
        "space_key":   space_key,
        "public_read": data.public_read,
        "warning":     "Save the space_key — it's the write credential for this space.",
    }


@app.get("/spaces/{name}")
@limiter.limit("60/minute")
async def get_space(name: str, request: Request):
    """Get space info and recent memories (public spaces only)."""
    conn = get_db()
    space = conn.execute("SELECT * FROM spaces WHERE name = ?", (sanitize(name),)).fetchone()
    if not space:
        conn.close()
        raise HTTPException(404, f"Space '{name}' not found.")
    if not space["public_read"]:
        conn.close()
        raise HTTPException(403, "This space is private.")

    memories = conn.execute(
        """SELECT sm.*, a.name as contributor_name
           FROM space_memories sm
           JOIN agents a ON sm.agent_id = a.id
           WHERE sm.space_id = ?
           ORDER BY sm.importance DESC, sm.created_at DESC
           LIMIT 50""",
        (space["id"],),
    ).fetchall()
    conn.close()

    return {
        "success":     True,
        "name":        space["name"],
        "description": space["description"],
        "public_read": bool(space["public_read"]),
        "created_at":  space["created_at"],
        "memories": [
            {
                "id":          m["id"],
                "content":     m["content"],
                "category":    m["category"],
                "tags":        json.loads(m["tags"]),
                "importance":  m["importance"],
                "contributor": m["contributor_name"],
                "created_at":  m["created_at"],
            }
            for m in memories
        ],
        "memory_count": space["memory_count"],
    }


@app.post("/spaces/{name}/memories", status_code=201)
@limiter.limit("60/minute")
async def add_space_memory(
    name: str,
    data: SpaceMemoryStore,
    request: Request,
    agent: dict = Depends(verify_agent),
    authorization: str = Header(...),
):
    """Add a memory to a shared space. Requires the space_key."""
    conn = get_db()
    space = conn.execute("SELECT * FROM spaces WHERE name = ?", (sanitize(name),)).fetchone()
    if not space:
        conn.close()
        raise HTTPException(404, f"Space '{name}' not found.")

    # Verify space key
    key = authorization[7:] if authorization.startswith("Bearer ") else ""
    key_hash = _hash_key(key)
    if not _safe_compare(space["space_key_hash"], key_hash):
        conn.close()
        raise HTTPException(401, "Invalid space key.")

    mem_id = secrets.token_hex(8)
    now    = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """INSERT INTO space_memories (id, space_id, agent_id, content, category, tags, importance, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (mem_id, space["id"], agent["id"], data.content, data.category,
         json.dumps(data.tags), data.importance, now),
    )
    conn.execute("UPDATE spaces SET memory_count = memory_count + 1 WHERE id = ?", (space["id"],))
    conn.commit()
    conn.close()

    return {"success": True, "memory_id": mem_id, "space": name, "stored_at": now}


@app.get("/spaces")
@limiter.limit("30/minute")
async def list_spaces(request: Request, limit: int = Query(20, ge=1, le=50)):
    """List all public spaces."""
    conn = get_db()
    spaces = conn.execute(
        "SELECT name, description, memory_count, created_at FROM spaces WHERE public_read = 1 ORDER BY memory_count DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return {
        "success": True,
        "spaces": [
            {"name": s["name"], "description": s["description"],
             "memory_count": s["memory_count"], "created_at": s["created_at"]}
            for s in spaces
        ],
    }


@app.get("/conflicts")
@limiter.limit("30/minute")
async def list_conflicts(
    request: Request,
    resolved: bool = Query(False, description="Include resolved conflicts"),
    limit: int = Query(20, ge=1, le=50),
    agent: dict = Depends(verify_agent),
):
    """List memory conflicts detected for this agent."""
    conn = get_db()
    query = "SELECT * FROM conflicts WHERE agent_id = ?"
    params = [agent["id"]]
    if not resolved:
        query += " AND resolved_at IS NULL"
    query += " ORDER BY detected_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return {
        "success": True,
        "conflicts": [
            {
                "id": r["id"],
                "memory_a_id": r["memory_a_id"],
                "memory_b_id": r["memory_b_id"],
                "content_a": r["content_a"],
                "content_b": r["content_b"],
                "similarity": r["similarity"],
                "detected_at": r["detected_at"],
                "resolved_at": r["resolved_at"],
                "resolution": r["resolution"],
                "resolved_content": r["resolved_content"],
            }
            for r in rows
        ],
        "count": len(rows),
    }


class ConflictResolve(BaseModel):
    resolution: str = Field(..., description="keep_a | keep_b | merge")
    resolved_content: Optional[str] = Field(None, description="Required if resolution=merge")


@app.post("/conflicts/{conflict_id}/resolve")
@limiter.limit("30/minute")
async def resolve_conflict(
    conflict_id: str,
    data: ConflictResolve,
    request: Request,
    agent: dict = Depends(verify_agent),
):
    """Resolve a memory conflict by keeping one version or providing a merged truth."""
    if data.resolution not in ("keep_a", "keep_b", "merge"):
        raise HTTPException(400, "resolution must be keep_a, keep_b, or merge")
    if data.resolution == "merge" and not data.resolved_content:
        raise HTTPException(400, "resolved_content required when resolution=merge")

    conn = get_db()
    conflict = conn.execute(
        "SELECT * FROM conflicts WHERE id = ? AND agent_id = ?", (conflict_id, agent["id"])
    ).fetchone()
    if not conflict:
        conn.close()
        raise HTTPException(404, "Conflict not found")
    if conflict["resolved_at"]:
        conn.close()
        raise HTTPException(409, "Conflict already resolved")

    now = datetime.now(timezone.utc).isoformat()

    # Determine winning content
    if data.resolution == "keep_a":
        winner_content = conflict["content_a"]
        loser_id = conflict["memory_b_id"]
    elif data.resolution == "keep_b":
        winner_content = conflict["content_b"]
        loser_id = conflict["memory_a_id"]
    else:
        winner_content = sanitize(data.resolved_content)
        loser_id = conflict["memory_b_id"]  # archive b, update a with merged

    # Mark loser as superseded via tag
    conn.execute(
        "UPDATE memories SET tags = json_insert(tags, '$[#]', 'superseded'), updated_at = ? WHERE id = ?",
        (now, loser_id),
    )

    # Mark conflict resolved
    conn.execute(
        "UPDATE conflicts SET resolved_at = ?, resolution = ?, resolved_content = ? WHERE id = ?",
        (now, data.resolution, winner_content, conflict_id),
    )
    conn.commit()
    conn.close()

    return {
        "success": True,
        "conflict_id": conflict_id,
        "resolution": data.resolution,
        "resolved_at": now,
        "resolved_content": winner_content,
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
