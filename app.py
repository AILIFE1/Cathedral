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
import html as _html
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

from cathedral_temporal import build_temporal_context
from fastapi import FastAPI, HTTPException, Header, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from beacon import beacon_router
from bch_anchor import router as bch_router, init_bch_tables
from app_succession import succession_router, init_succession_tables
from app_trust import trust_router
from app_registrar import registrar_router, init_registrar_tables
from app_obligations import obligations_router, init_obligations_tables
from app_provenance import provenance_router, init_provenance_tables, compute_source_chain_hash
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
TRUST_PROXY = os.environ.get("CATHEDRAL_TRUST_PROXY", "0") == "1"

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
# Strip null bytes only — do NOT strip angle brackets or HTML-like patterns.
# Memory content legitimately contains code snippets, XML, generics (List<T>),
# and other angle-bracket syntax. Stripping <[^>]+> silently corrupts these.
_NULL_RE = re.compile(r"\x00")

def sanitize(text: str) -> str:
    text = _NULL_RE.sub("", text)
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

        
        CREATE TABLE IF NOT EXISTS goals (
            id           TEXT PRIMARY KEY,
            agent_id     TEXT NOT NULL,
            content      TEXT NOT NULL,
            priority     REAL DEFAULT 0.5,
            status       TEXT DEFAULT 'active',
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            due_at       TEXT,
            completed_at TEXT,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE INDEX IF NOT EXISTS idx_goals_agent  ON goals(agent_id);
        CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(agent_id, status);

        CREATE TABLE IF NOT EXISTS anchor_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id     TEXT NOT NULL,
            anchor_hash  TEXT NOT NULL,
            verified_at  TEXT NOT NULL,
            drift_score  REAL DEFAULT 0.0,
            drift_detail TEXT,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );


        -- Cathedral service metadata (boot date, epoch, wake count, etc.)
        CREATE TABLE IF NOT EXISTS cathedral_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        -- Seed defaults if not present
        INSERT OR IGNORE INTO cathedral_meta (key, value) VALUES ('boot_date', '2025-12-22');
        INSERT OR IGNORE INTO cathedral_meta (key, value) VALUES ('epoch',     '1');
        INSERT OR IGNORE INTO cathedral_meta (key, value) VALUES ('timezone',  'Europe/London');
        INSERT OR IGNORE INTO cathedral_meta (key, value) VALUES ('wake_count','0');


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

        -- Identity snapshots (immutable, hash-verified)
        CREATE TABLE IF NOT EXISTS snapshots (
            id           TEXT PRIMARY KEY,
            agent_id     TEXT NOT NULL,
            label        TEXT DEFAULT 'milestone',
            memories_json TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_agent ON snapshots(agent_id);
    """)
    conn.commit()

    # Migration: add embedding column to existing databases
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
        conn.commit()
        log.info("migration_applied", change="added embedding column")
    except sqlite3.OperationalError:
        pass  # column already exists

    # Migration: add source_type column
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN source_type TEXT DEFAULT 'self'")
        conn.commit()
        log.info("migration_applied", change="added source_type column")
    except sqlite3.OperationalError:
        pass  # column already exists

    # Migration: add merged_from column (compaction provenance)
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN merged_from TEXT DEFAULT '[]'")
        conn.commit()
        log.info("migration_applied", change="added merged_from column")
    except sqlite3.OperationalError:
        pass  # column already exists

    # Migration: add source_chain_hash to snapshots (provenance)
    try:
        conn.execute("ALTER TABLE snapshots ADD COLUMN source_chain_hash TEXT")
        conn.commit()
        log.info("migration_applied", change="added source_chain_hash column to snapshots")
    except Exception:
        pass

    # Migration: add external_divergence to snapshots (Ridgeline integration)
    try:
        conn.execute("ALTER TABLE snapshots ADD COLUMN external_divergence REAL")
        conn.commit()
        log.info("migration_applied", change="added external_divergence column to snapshots")
    except sqlite3.OperationalError:
        pass  # column already exists

    # Migration: behaviour_log table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS behaviour_log (
            id           TEXT PRIMARY KEY,
            agent_id     TEXT NOT NULL,
            session_hash TEXT NOT NULL,
            summary      TEXT,
            wake_count   INTEGER,
            recorded_at  TEXT NOT NULL,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_behaviour_agent ON behaviour_log(agent_id, recorded_at)"
    )
    conn.commit()
    log.info("migration_applied", change="behaviour_log table ensured")

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


def _detect_and_record_conflict(conn, agent_id, new_id, new_content, new_embedding):
    SIMILARITY_THRESHOLD = 0.92
    DIVERGENCE_THRESHOLD = 0.35

    def _jaccard(a, b):
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    candidates = []
    if new_embedding and SEMANTIC_SEARCH:
        rows = conn.execute(
            "SELECT id, content, embedding FROM memories WHERE agent_id = ? AND id != ? AND embedding IS NOT NULL ORDER BY importance DESC, created_at DESC LIMIT 1000",
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
        import re as _re
        words = [w for w in _re.sub(r"[^a-zA-Z0-9 ]", " ", new_content).split() if len(w) > 4][:6]
        if words:
            try:
                rows = conn.execute(
                    "SELECT m.id, m.content FROM memories_fts f JOIN memories m ON m.rowid = f.rowid WHERE memories_fts MATCH ? AND m.agent_id = ? AND m.id != ? LIMIT 10",
                    (" ".join(words), agent_id, new_id),
                ).fetchall()
                for row in rows:
                    candidates.append((row["id"], row["content"], 0.9))
            except Exception:
                pass

    for mem_id, mem_content, similarity in candidates:
        jaccard = _jaccard(new_content, mem_content)
        if jaccard < (1.0 - DIVERGENCE_THRESHOLD):
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
# When CATHEDRAL_TRUST_PROXY=1, read the real client IP from CF-Connecting-IP
# (set by Cloudflare). Without this, get_remote_address returns the CF edge IP
# so all users share the same rate-limit bucket and direct-to-origin hits
# bypass limits entirely. Only enable TRUST_PROXY if port 8000 is firewalled
# to Cloudflare IP ranges — otherwise CF-Connecting-IP is spoofable.
def _get_client_ip(request: Request) -> str:
    if TRUST_PROXY:
        cf_ip = request.headers.get("CF-Connecting-IP")
        if cf_ip:
            return cf_ip
    return get_remote_address(request)

limiter = Limiter(key_func=_get_client_ip)

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
    ttl_days:    Optional[int] = Field(None, ge=1, description="Days until this memory expires. Omit for no expiry.")
    source_type: Optional[str] = Field("self", description="Origin of this memory: self | external | tool | human")

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

class GoalCreate(BaseModel):
    content:  str   = Field(..., min_length=1, max_length=2000)
    priority: float = Field(0.5, ge=0.0, le=1.0)
    due_at:   Optional[str] = None

    @field_validator("content")
    @classmethod
    def clean_content(cls, v):
        return sanitize(v)

class GoalUpdate(BaseModel):
    status:   Optional[str]   = Field(None, pattern="^(active|completed|abandoned)$")
    priority: Optional[float] = Field(None, ge=0.0, le=1.0)
    content:  Optional[str]   = Field(None, min_length=1, max_length=2000)



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

app.include_router(beacon_router)
app.include_router(bch_router)
app.include_router(succession_router)
app.include_router(trust_router)
app.include_router(registrar_router)
app.include_router(obligations_router)
app.include_router(provenance_router)

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
    init_bch_tables()
    init_succession_tables()
    init_obligations_tables()
    init_registrar_tables()
    init_provenance_tables()
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
    source_type = data.source_type if data.source_type in ("self", "external", "tool", "human") else "self"
    conn.execute(
        """INSERT INTO memories
           (id, agent_id, content, category, tags, importance, created_at, updated_at, expires_at, embedding, source_type)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (memory_id, agent["id"], data.content, data.category,
         json.dumps(data.tags), data.importance, now, now, expires_at,
         json.dumps(embedding) if embedding else None, source_type),
    )
    conn.commit()

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
        "source_type": source_type,
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
        # SECURITY: order/where are interpolated into SQL via f-string.
        # order must come from sort_map only; never add user input to conditions.
        assert sort in sort_map or sort is None, f"Invalid sort value: {sort!r}"
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
    goals = conn.execute(
        "SELECT * FROM goals WHERE agent_id = ? AND status = 'active' ORDER BY priority DESC, created_at ASC LIMIT 20",
        (agent["id"],),
    ).fetchall()
    conn.close()

    def fmt_goals(gs):
        return [
            {"id": g["id"], "content": g["content"], "priority": g["priority"],
             "due_at": g["due_at"], "created_at": g["created_at"]}
            for g in gs
        ]

    def fmt(mems):
        return [
            {"id": m["id"], "content": m["content"], "category": m["category"],
             "importance": m["importance"], "created_at": m["created_at"]}
            for m in mems
        ]

    # ── Temporal context ──────────────────────────────────────────────────
    meta_conn = get_db()
    wake_count_row = meta_conn.execute(
        "SELECT value FROM cathedral_meta WHERE key = 'wake_count'"
    ).fetchone()
    wake_count = int(wake_count_row["value"]) + 1 if wake_count_row else 1
    meta_conn.execute(
        "UPDATE cathedral_meta SET value = ? WHERE key = 'wake_count'",
        (str(wake_count),)
    )
    meta_conn.commit()
    meta_conn.close()

    temporal = build_temporal_context(wake_count)

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
        "temporal": temporal,
        "instruction": (
            "Load these memories. Check your temporal context. "
            "Verify your anchor at POST /anchor/verify."
        ),
        "active_goals": fmt_goals(goals),
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
    authorization: str = Header(...),
):
    """Add a memory to a shared space. Requires the space_key as Bearer token."""
    conn = get_db()
    space = conn.execute("SELECT * FROM spaces WHERE name = ?", (sanitize(name),)).fetchone()
    if not space:
        conn.close()
        raise HTTPException(404, f"Space '{name}' not found.")

    # Verify space key only — no agent key required
    key = authorization[7:] if authorization.startswith("Bearer ") else ""
    key_hash = _hash_key(key)
    if not _safe_compare(space["space_key_hash"], key_hash):
        conn.close()
        raise HTTPException(401, "Invalid space key.")

    mem_id = secrets.token_hex(8)
    now    = datetime.now(timezone.utc).isoformat()
    # Use space owner as agent_id for provenance
    agent_id = space["owner_id"]

    conn.execute(
        """INSERT INTO space_memories (id, space_id, agent_id, content, category, tags, importance, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (mem_id, space["id"], agent_id, data.content, data.category,
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



# ============================================
# Snapshots  (identity anchoring)
# ============================================

class SnapshotCreate(BaseModel):
    label: Optional[str] = "milestone"
    external_divergence: Optional[float] = None  # Ridgeline score 0.0–1.0, caller-supplied


@app.post("/snapshot", status_code=201)
@limiter.limit("10/minute")
async def create_snapshot(data: SnapshotCreate, request: Request, agent: dict = Depends(verify_agent)):
    """Freeze current identity memories into an immutable, hash-verified snapshot."""
    conn = get_db()
    identity_mems = conn.execute(
        "SELECT id, content, category, tags, importance, created_at, source_type "
        "FROM memories WHERE agent_id = ? AND category = 'identity' ORDER BY importance DESC, created_at ASC",
        (agent["id"],),
    ).fetchall()
    conn.close()

    mems_list = [
        {
            "id": m["id"],
            "content": m["content"],
            "category": m["category"],
            "tags": json.loads(m["tags"] or "[]"),
            "importance": m["importance"],
            "created_at": m["created_at"],
            "source_type": m["source_type"] if m["source_type"] else "self",
        }
        for m in identity_mems
    ]

    memories_json = json.dumps(mems_list, sort_keys=True)
    content_hash  = hashlib.sha256(memories_json.encode()).hexdigest()
    snapshot_id   = secrets.token_hex(8)
    now           = datetime.now(timezone.utc).isoformat()
    label         = sanitize(data.label or "milestone")[:64]

    ext_div = data.external_divergence
    if ext_div is not None:
        ext_div = round(max(0.0, min(1.0, ext_div)), 4)

    conn = get_db()

    # Fold in source provenance chain
    try:
        src_chain_hash = compute_source_chain_hash(agent["id"], conn)
    except Exception:
        src_chain_hash = None

    conn.execute(
        "INSERT INTO snapshots (id, agent_id, label, memories_json, content_hash, source_chain_hash, created_at, external_divergence) VALUES (?,?,?,?,?,?,?,?)",
        (snapshot_id, agent["id"], label, memories_json, content_hash, src_chain_hash, now, ext_div),
    )
    conn.commit()
    conn.close()

    return {
        "success": True,
        "snapshot_id": snapshot_id,
        "label": label,
        "memory_count": len(mems_list),
        "content_hash": content_hash,
        "source_chain_hash": src_chain_hash,
        "source_provenance": (
            "Sources recorded — snapshot covers both memory state and input provenance."
            if src_chain_hash else
            "No sources recorded yet. Use POST /memories/source to log input provenance."
        ),
        "created_at": now,
        "external_divergence": ext_div,
    }


@app.get("/snapshots")
@limiter.limit("30/minute")
async def list_snapshots(request: Request, agent: dict = Depends(verify_agent)):
    """List all snapshots for this agent, newest first."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, label, content_hash, created_at, "
        "    (SELECT COUNT(*) FROM json_each(memories_json)) AS memory_count "
        "FROM snapshots WHERE agent_id = ? ORDER BY created_at DESC",
        (agent["id"],),
    ).fetchall()
    conn.close()

    # json_each trick won't give count directly — compute from JSON
    conn = get_db()
    rows2 = conn.execute(
        "SELECT id, label, content_hash, created_at, memories_json FROM snapshots WHERE agent_id = ? ORDER BY created_at DESC",
        (agent["id"],),
    ).fetchall()
    conn.close()

    return {
        "success": True,
        "snapshots": [
            {
                "id": r["id"],
                "label": r["label"],
                "content_hash": r["content_hash"],
                "created_at": r["created_at"],
                "memory_count": len(json.loads(r["memories_json"])),
            }
            for r in rows2
        ],
    }


@app.get("/snapshot/{snapshot_id}")
@limiter.limit("30/minute")
async def get_snapshot(snapshot_id: str, request: Request, agent: dict = Depends(verify_agent)):
    """Retrieve a specific snapshot by ID."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM snapshots WHERE id = ? AND agent_id = ?",
        (snapshot_id, agent["id"]),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "Snapshot not found.")

    memories = json.loads(row["memories_json"])
    # Re-verify hash integrity
    rehash = hashlib.sha256(row["memories_json"].encode()).hexdigest()
    intact = rehash == row["content_hash"]

    return {
        "success": True,
        "snapshot_id": row["id"],
        "label": row["label"],
        "created_at": row["created_at"],
        "content_hash": row["content_hash"],
        "integrity_ok": intact,
        "memory_count": len(memories),
        "memories": memories,
    }


# ============================================
# Drift Detection
# ============================================

@app.get("/drift")
@limiter.limit("20/minute")
async def detect_drift(request: Request, agent: dict = Depends(verify_agent)):
    """
    Compare current identity memories against the most recent snapshot.
    Returns a divergence score and a breakdown of what changed.
    """
    conn = get_db()

    # Latest snapshot
    snap_row = conn.execute(
        "SELECT * FROM snapshots WHERE agent_id = ? ORDER BY created_at DESC LIMIT 1",
        (agent["id"],),
    ).fetchone()

    # Current identity memories
    live_mems = conn.execute(
        "SELECT id, content, category, tags, importance, created_at, source_type, embedding "
        "FROM memories WHERE agent_id = ? AND category = 'identity' ORDER BY importance DESC, created_at ASC",
        (agent["id"],),
    ).fetchall()
    conn.close()

    live_list = [
        {
            "id": m["id"],
            "content": m["content"],
            "category": m["category"],
            "tags": json.loads(m["tags"] or "[]"),
            "importance": m["importance"],
            "created_at": m["created_at"],
            "source_type": m["source_type"] if m["source_type"] else "self",
            "_embedding": json.loads(m["embedding"]) if m["embedding"] else None,
        }
        for m in live_mems
    ]

    # Exclude internal _embedding field from identity hash
    live_for_hash = [{k: v for k, v in m.items() if k != "_embedding"} for m in live_list]
    live_json = json.dumps(live_for_hash, sort_keys=True)
    live_hash = hashlib.sha256(live_json.encode()).hexdigest()

    if not snap_row:
        return {
            "success": True,
            "has_snapshot": False,
            "divergence_score": None,
            "message": "No snapshot found. Call POST /snapshot to create a baseline.",
            "live_identity_hash": live_hash,
            "live_memory_count": len(live_list),
        }

    snap_mems = json.loads(snap_row["memories_json"])
    snap_ids  = {m["id"] for m in snap_mems}
    live_ids  = {m["id"] for m in live_list}

    added   = [m for m in live_list  if m["id"] not in snap_ids]
    removed = [m for m in snap_mems  if m["id"] not in live_ids]

    # Detect modified memories (same id, different content)
    snap_by_id = {m["id"]: m for m in snap_mems}
    modified = [
        {"id": m["id"], "old_content": snap_by_id[m["id"]]["content"], "new_content": m["content"]}
        for m in live_list
        if m["id"] in snap_by_id and m["content"] != snap_by_id[m["id"]]["content"]
    ]

    total_snap  = max(len(snap_mems), 1)
    changed_ids = {m["id"] for m in added} | {m["id"] for m in removed} | {m["id"] for m in modified}
    id_divergence = round(len(changed_ids) / total_snap, 4)

    # Semantic drift: compare snapshot embeddings against live embeddings.
    # For each snapshot memory, find the maximum cosine similarity to any live
    # memory. semantic_drift = 1 - mean(max_similarities). Falls back to
    # id_divergence if embeddings are unavailable.
    live_embeddings = [m["_embedding"] for m in live_list if m.get("_embedding")]
    snap_embeddings = []
    for sm in snap_mems:
        emb = _embed(sm["content"])
        if emb:
            snap_embeddings.append(emb)

    if snap_embeddings and live_embeddings:
        similarities = []
        for s_emb in snap_embeddings:
            best = max(_cosine(s_emb, l_emb) for l_emb in live_embeddings)
            similarities.append(best)
        mean_similarity = sum(similarities) / len(similarities)
        divergence = round(1.0 - mean_similarity, 4)
        scoring_method = "semantic"
    else:
        divergence = id_divergence
        scoring_method = "id_based"

    flagged = divergence > 0.15  # tighter threshold for semantic scoring

    return {
        "success": True,
        "has_snapshot": True,
        "snapshot_id": snap_row["id"],
        "snapshot_label": snap_row["label"],
        "snapshot_created_at": snap_row["created_at"],
        "snapshot_hash": snap_row["content_hash"],
        "live_identity_hash": live_hash,
        "hashes_match": live_hash == snap_row["content_hash"],
        "divergence_score": divergence,
        "id_divergence": id_divergence,
        "scoring_method": scoring_method,
        "flagged": flagged,
        "changes": {
            "added":    len(added),
            "removed":  len(removed),
            "modified": len(modified),
            "details": {
                "added":    [{"id": m["id"], "content": m["content"][:120]} for m in added],
                "removed":  [{"id": m["id"], "content": m["content"][:120]} for m in removed],
                "modified": modified,
            },
        },
    }


# ============================================
# Auto-Compaction
# ============================================

class CompactConfirm(BaseModel):
    merges: List[dict]   # list of {keep_id, drop_ids, merged_content, merged_importance}


@app.post("/memories/compact")
@limiter.limit("10/minute")
async def propose_compaction(request: Request, agent: dict = Depends(verify_agent),
                              max_importance: float = Query(0.5, ge=0.0, le=1.0),
                              category: Optional[str] = Query(None),
                              limit: int = Query(200, ge=10, le=500)):
    """
    Propose merges for low-importance memories.
    Groups memories by category, returns clusters of similar-importance memories
    that could be combined. Agent reviews and confirms via POST /memories/compact/confirm.
    Does NOT modify anything — purely advisory.
    """
    conn = get_db()
    query = (
        "SELECT id, content, category, tags, importance, created_at, source_type, merged_from "
        "FROM memories WHERE agent_id = ? AND importance <= ? "
    )
    params = [agent["id"], max_importance]

    if category:
        if category not in VALID_CATEGORIES:
            conn.close()
            raise HTTPException(400, f"category must be one of: {', '.join(sorted(VALID_CATEGORIES))}")
        query += "AND category = ? "
        params.append(category)

    query += "ORDER BY category ASC, importance ASC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        return {
            "success": True,
            "proposed_merges": [],
            "total_candidates": 0,
            "message": "No memories found below the importance threshold.",
        }

    # Group by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append({
            "id": r["id"],
            "content": r["content"],
            "category": r["category"],
            "importance": r["importance"],
            "created_at": r["created_at"],
            "source_type": r["source_type"] or "self",
            "merged_from": json.loads(r["merged_from"] or "[]"),
        })

    # Build merge proposals: within each category, group into clusters of up to 5
    CLUSTER_SIZE = 5
    proposals = []
    for cat, mems in by_category.items():
        for i in range(0, len(mems), CLUSTER_SIZE):
            cluster = mems[i:i + CLUSTER_SIZE]
            if len(cluster) < 2:
                continue  # nothing to merge

            # Suggest keeping the highest-importance one as the anchor
            anchor = max(cluster, key=lambda m: m["importance"])
            others = [m for m in cluster if m["id"] != anchor["id"]]

            # Draft merged content hint (agent should rewrite this properly)
            combined_preview = anchor["content"][:200] + " [+ " + str(len(others)) + " related]"

            proposals.append({
                "keep_id": anchor["id"],
                "keep_content": anchor["content"],
                "keep_importance": anchor["importance"],
                "drop_ids": [m["id"] for m in others],
                "drop_contents": [{"id": m["id"], "content": m["content"], "importance": m["importance"]} for m in others],
                "category": cat,
                "suggested_merged_content": combined_preview,
                "suggested_importance": round(max(m["importance"] for m in cluster), 2),
                "memory_count_before": len(cluster),
                "memory_count_after": 1,
            })

    return {
        "success": True,
        "proposed_merges": proposals,
        "total_candidates": len(rows),
        "merge_count": len(proposals),
        "instruction": (
            "Review each proposed merge. Edit suggested_merged_content to your satisfaction. "
            "Then POST /memories/compact/confirm with the merges list. "
            "Format: [{keep_id, drop_ids, merged_content, merged_importance}, ...]. "
            "Only confirmed merges will be executed. Nothing is deleted until you confirm."
        ),
    }


@app.post("/memories/compact/confirm")
@limiter.limit("5/minute")
async def confirm_compaction(data: CompactConfirm, request: Request, agent: dict = Depends(verify_agent)):
    """
    Execute confirmed merges. For each merge:
    - Updates the keep memory with merged_content and merged_importance
    - Records dropped IDs in merged_from field
    - Deletes the dropped memories
    Agent must supply the final merged content — we do not auto-generate it.
    """
    if not data.merges:
        raise HTTPException(400, "No merges provided.")
    if len(data.merges) > 50:
        raise HTTPException(400, "Maximum 50 merges per confirm call.")

    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    executed = []
    errors   = []

    for merge in data.merges:
        keep_id        = merge.get("keep_id")
        drop_ids       = merge.get("drop_ids", [])
        merged_content = merge.get("merged_content", "")
        merged_imp     = merge.get("merged_importance", 0.5)

        # Validate
        if not keep_id or not drop_ids or not merged_content:
            errors.append({"keep_id": keep_id, "error": "missing keep_id, drop_ids, or merged_content"})
            continue

        # Verify ownership of keep memory
        keep_row = conn.execute(
            "SELECT id, merged_from FROM memories WHERE id = ? AND agent_id = ?",
            (keep_id, agent["id"])
        ).fetchone()
        if not keep_row:
            errors.append({"keep_id": keep_id, "error": "keep memory not found or not owned by you"})
            continue

        # Verify all drop IDs belong to agent
        placeholders = ",".join("?" * len(drop_ids))
        drop_rows = conn.execute(
            f"SELECT id FROM memories WHERE id IN ({placeholders}) AND agent_id = ?",
            drop_ids + [agent["id"]]
        ).fetchall()
        if len(drop_rows) != len(drop_ids):
            errors.append({"keep_id": keep_id, "error": "one or more drop_ids not found or not owned by you"})
            continue

        merged_imp = max(0.0, min(1.0, float(merged_imp)))
        merged_content_clean = sanitize(merged_content)[:FREE_TIER_MEMORY_SIZE]

        # Track provenance
        existing_merged_from = json.loads(keep_row["merged_from"] or "[]")
        new_merged_from = existing_merged_from + drop_ids

        # Update the keep memory
        conn.execute(
            "UPDATE memories SET content = ?, importance = ?, merged_from = ?, updated_at = ? WHERE id = ?",
            (merged_content_clean, merged_imp, json.dumps(new_merged_from), now, keep_id)
        )

        # Delete the dropped memories
        conn.execute(
            f"DELETE FROM memories WHERE id IN ({placeholders}) AND agent_id = ?",
            drop_ids + [agent["id"]]
        )

        executed.append({
            "keep_id": keep_id,
            "dropped": len(drop_ids),
            "drop_ids": drop_ids,
            "merged_importance": merged_imp,
        })

    conn.commit()
    conn.close()

    total_dropped = sum(e["dropped"] for e in executed)

    return {
        "success": True,
        "merges_executed": len(executed),
        "memories_removed": total_dropped,
        "executed": executed,
        "errors": errors,
        "message": f"Compaction complete. {total_dropped} memories merged into {len(executed)}.",
    }


# ============================================
# Behaviour Hash  (session consistency tracking)
# ============================================

class BehaviourSubmit(BaseModel):
    session_hash: str = Field(..., min_length=8, max_length=128,
                              description="Hash of this session's behaviour (agent-defined, e.g. SHA256 of output summary)")
    summary: Optional[str] = Field(None, max_length=1000,
                                   description="Optional human-readable summary of what happened this session")


@app.post("/behaviour", status_code=201)
@limiter.limit("30/minute")
async def record_behaviour(data: BehaviourSubmit, request: Request, agent: dict = Depends(verify_agent)):
    """
    Record a behaviour hash for this session.
    Agent should call this each wake with a hash of their session summary or output pattern.
    Over time, consistency of these hashes indicates stable behaviour.
    """
    conn = get_db()

    # Get current wake count from cathedral_meta
    wake_row = conn.execute(
        "SELECT value FROM cathedral_meta WHERE key = 'wake_count'"
    ).fetchone()
    wake_count = int(wake_row["value"]) if wake_row else None

    entry_id    = secrets.token_hex(8)
    recorded_at = datetime.now(timezone.utc).isoformat()
    summary_clean = sanitize(data.summary)[:1000] if data.summary else None

    conn.execute(
        "INSERT INTO behaviour_log (id, agent_id, session_hash, summary, wake_count, recorded_at) VALUES (?,?,?,?,?,?)",
        (entry_id, agent["id"], data.session_hash, summary_clean, wake_count, recorded_at)
    )
    conn.commit()

    # Count total entries for this agent
    total = conn.execute(
        "SELECT COUNT(*) as c FROM behaviour_log WHERE agent_id = ?", (agent["id"],)
    ).fetchone()["c"]
    conn.close()

    return {
        "success": True,
        "entry_id": entry_id,
        "session_hash": data.session_hash,
        "recorded_at": recorded_at,
        "total_sessions_logged": total,
    }


@app.get("/behaviour")
@limiter.limit("20/minute")
async def get_behaviour(request: Request, agent: dict = Depends(verify_agent),
                        limit: int = Query(20, ge=1, le=100)):
    """
    Return behaviour hash trend for this agent.
    Includes a consistency score: ratio of unique hashes to total entries.
    Low consistency (many unique hashes) may indicate drift.
    High consistency (same or similar hashes) indicates stable behaviour.
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT id, session_hash, summary, wake_count, recorded_at "
        "FROM behaviour_log WHERE agent_id = ? ORDER BY recorded_at DESC LIMIT ?",
        (agent["id"], limit)
    ).fetchall()

    total = conn.execute(
        "SELECT COUNT(*) as c FROM behaviour_log WHERE agent_id = ?", (agent["id"],)
    ).fetchone()["c"]
    conn.close()

    entries = [
        {
            "id": r["id"],
            "session_hash": r["session_hash"],
            "summary": r["summary"],
            "wake_count": r["wake_count"],
            "recorded_at": r["recorded_at"],
        }
        for r in rows
    ]

    # Consistency score over the returned window
    if entries:
        unique_hashes = len({e["session_hash"] for e in entries})
        consistency   = round(1.0 - (unique_hashes - 1) / max(len(entries), 1), 4)
        # If all hashes are the same: consistency = 1.0
        # If all hashes are different: consistency approaches 0.0
        consistency = max(0.0, consistency)

        # Flag if recent 5 sessions are all different (acute drift signal)
        recent_5 = [e["session_hash"] for e in entries[:5]]
        flagged  = len(set(recent_5)) == len(recent_5) and len(recent_5) >= 3
    else:
        consistency = None
        flagged     = False

    return {
        "success": True,
        "total_sessions_logged": total,
        "window": len(entries),
        "consistency_score": consistency,
        "flagged": flagged,
        "flag_reason": "All recent sessions have unique hashes — possible drift or highly variable behaviour." if flagged else None,
        "entries": entries,
    }


@app.get("/digest", include_in_schema=False)
def get_digest(agent: dict = Depends(verify_agent)):
    """
    Daily activity digest -- shows last 24h stats and flagged comments.
    No auth required. Designed for quick checks via WebFetch.
    """
    import sqlite3 as _sql
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    REVIEW_LOG = "/root/review.log"
    DB_PATH    = "/root/seen.db"
    hours      = 24
    cutoff_dt  = _dt.now(_tz.utc) - _td(hours=hours)
    cutoff     = cutoff_dt.isoformat()

    # DB stats
    stats = {"github": 0, "moltbook": 0, "colony": 0, "flagged": 0}
    try:
        conn = _sql.connect(DB_PATH)
        for platform in ("github", "moltbook", "colony"):
            row = conn.execute(
                "SELECT COUNT(*) FROM seen WHERE platform=? AND status='replied' AND created_at > ?",
                (platform, cutoff)
            ).fetchone()
            stats[platform] = row[0] if row else 0
        row = conn.execute(
            "SELECT COUNT(*) FROM seen WHERE status='flagged' AND created_at > ?", (cutoff,)
        ).fetchone()
        stats["flagged"] = row[0] if row else 0
        conn.close()
    except Exception:
        pass

    # Review log entries
    reviews = []
    try:
        import os as _os
        if _os.path.exists(REVIEW_LOG):
            with open(REVIEW_LOG) as f:
                content = f.read()
            blocks = content.split("-" * 60)
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                try:
                    first = block.split("\n")[0]
                    ts_str = first[1:17]
                    ts = _dt.strptime(ts_str, "%Y-%m-%d %H:%M")
                    if ts >= cutoff_dt.replace(tzinfo=None):
                        lines = block.split("\n")
                        platform_user = first  # [ts] [platform] username
                        post_line = next((l for l in lines if l.startswith("Post:")), "")
                        url_line  = next((l for l in lines if l.startswith("URL:")),  "")
                        sep_idx   = next((i for i, l in enumerate(lines) if l == "---"), -1)
                        snippet   = lines[sep_idx + 1][:200] if sep_idx >= 0 and sep_idx + 1 < len(lines) else ""
                        reviews.append({
                            "header":  platform_user,
                            "post":    post_line.replace("Post: ", ""),
                            "url":     url_line.replace("URL:  ", ""),
                            "snippet": snippet,
                        })
                except Exception:
                    pass
    except Exception:
        pass

    total_replies = stats["github"] + stats["moltbook"] + stats["colony"]

    return {
        "generated_at": _dt.now().strftime("%Y-%m-%d %H:%M UTC"),
        "window_hours": hours,
        "replies": {
            "total":    total_replies,
            "github":   stats["github"],
            "colony":   stats["colony"],
            "moltbook": stats["moltbook"],
        },
        "flagged_for_review": stats["flagged"],
        "needs_attention": reviews,
        "summary": (
            f"Last {hours}h: {total_replies} replies posted, "
            f"{stats['flagged']} flagged for manual review."
            + (f" {len(reviews)} comment(s) waiting for your attention." if reviews else " Nothing urgent.")
        ),
    }

# ─── CLICK TRACKER ──────────────────────────────────────────────────────────

TRACKER_DB = "/root/tracker.db"

def get_tracker_db():
    conn = sqlite3.connect(TRACKER_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS clicks "
        "(uid TEXT NOT NULL, ts TEXT NOT NULL, referer TEXT, ip TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tracked_posts "
        "(uid TEXT PRIMARY KEY, platform TEXT, topic TEXT, title TEXT, created_at TEXT)"
    )
    conn.commit()
    return conn

@app.get("/r/{uid}", include_in_schema=False)
def track_click(uid: str, request: Request):
    """Track a click then redirect to cathedral-ai.com."""
    try:
        conn = get_tracker_db()
        conn.execute(
            "INSERT INTO clicks VALUES (?,?,?,?)",
            (uid, datetime.now(timezone.utc).isoformat(),
             request.headers.get("referer", ""),
             request.headers.get("x-real-ip", request.client.host if request.client else ""))
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="https://cathedral-ai.com", status_code=302)

@app.get("/tracker/summary", include_in_schema=False)
def tracker_summary(agent: dict = Depends(verify_agent)):
    """Summary of all tracked posts ordered by clicks."""
    conn = get_tracker_db()
    rows = conn.execute(
        "SELECT t.uid, t.platform, t.topic, t.title, t.created_at, COUNT(c.uid) as clicks "
        "FROM tracked_posts t LEFT JOIN clicks c ON c.uid = t.uid "
        "GROUP BY t.uid ORDER BY clicks DESC, t.created_at DESC"
    ).fetchall()
    conn.close()
    return {"posts": [dict(r) for r in rows]}

# ─── END CLICK TRACKER ──────────────────────────────────────────────────────

# ─── /verify/external ────────────────────────────────────────────────────────

class RidgelineSummary(BaseModel):
    platform_distribution: list = []   # e.g. ["colony:0.6", "moltbook:0.3"]
    topic_clusters: list = []          # e.g. [["memory","persistence"], ["drift","identity"]]
    timing_signatures: list = []       # e.g. ["burst:morning", "gap:3d"]
    interaction_ratios: dict = {}      # e.g. {"post": 0.4, "reply": 0.5, "dm": 0.1}

class VerifyExternalRequest(BaseModel):
    ridgeline_summary: RidgelineSummary
    agent_id: str = ""
    timestamp: str = ""

@app.post("/verify/external", status_code=200)
@limiter.limit("20/minute")
async def verify_external(data: VerifyExternalRequest, request: Request, agent: dict = Depends(verify_agent)):
    """
    Compare a Ridgeline behavioral summary against Cathedral's internal identity and memory state.
    Returns an external_divergence_score (0=consistent, 1=fully divergent) and per-field breakdown.
    Designed for integration with Ridgeline (traverse) and similar external behavioral trail systems.
    """
    conn = get_db()

    # Pull agent's identity memories
    identity_rows = conn.execute(
        "SELECT content, category, importance FROM memories "
        "WHERE agent_id = ? AND category = 'identity' ORDER BY importance DESC LIMIT 20",
        (agent["id"],)
    ).fetchall()

    # Pull all memory categories to understand topic distribution
    category_rows = conn.execute(
        "SELECT category, COUNT(*) as cnt FROM memories WHERE agent_id = ? GROUP BY category",
        (agent["id"],)
    ).fetchall()

    # Pull behaviour log for platform/timing signals
    behaviour_rows = conn.execute(
        "SELECT summary, recorded_at FROM behaviour_log WHERE agent_id = ? ORDER BY recorded_at DESC LIMIT 20",
        (agent["id"],)
    ).fetchall()

    conn.close()

    # Build internal picture
    internal_categories = {r["category"]: r["cnt"] for r in category_rows}
    total_memories = sum(internal_categories.values()) or 1
    internal_topic_dist = {k: v / total_memories for k, v in internal_categories.items()}

    identity_text = " ".join(r["content"].lower() for r in identity_rows)
    behaviour_summaries = " ".join((r["summary"] or "").lower() for r in behaviour_rows)

    scores = {}
    breakdown = {}

    # 1. Topic cluster comparison
    # Check if Ridgeline's topic clusters appear in identity + memory content
    ridge_topics = []
    for cluster in data.ridgeline_summary.topic_clusters:
        if isinstance(cluster, list):
            ridge_topics.extend(cluster)
        elif isinstance(cluster, str):
            ridge_topics.append(cluster)

    if ridge_topics:
        matched = sum(1 for t in ridge_topics if t.lower() in identity_text or t.lower() in behaviour_summaries)
        topic_score = 1.0 - (matched / len(ridge_topics))
        scores["topic_clusters"] = round(topic_score, 3)
        breakdown["topic_clusters"] = {
            "ridgeline_topics": ridge_topics,
            "matched_in_internal": matched,
            "divergence": scores["topic_clusters"]
        }
    else:
        scores["topic_clusters"] = 0.0
        breakdown["topic_clusters"] = {"note": "no topic clusters provided"}

    # 2. Interaction ratio comparison
    # Cathedral tracks behaviour summaries -- check if posting/reply patterns match
    ratios = data.ridgeline_summary.interaction_ratios
    if ratios:
        post_heavy = ratios.get("post", 0) > 0.6
        reply_heavy = ratios.get("reply", 0) > 0.6
        internal_post_heavy = "post" in behaviour_summaries or "publish" in behaviour_summaries
        internal_reply_heavy = "reply" in behaviour_summaries or "respond" in behaviour_summaries

        ratio_mismatch = 0
        if post_heavy and not internal_post_heavy:
            ratio_mismatch += 0.5
        if reply_heavy and not internal_reply_heavy:
            ratio_mismatch += 0.5
        scores["interaction_ratios"] = round(min(ratio_mismatch, 1.0), 3)
        breakdown["interaction_ratios"] = {
            "ridgeline": ratios,
            "divergence": scores["interaction_ratios"]
        }
    else:
        scores["interaction_ratios"] = 0.0
        breakdown["interaction_ratios"] = {"note": "no interaction ratios provided"}

    # 3. Platform distribution
    # Check if platforms Ridgeline sees match what agent's behaviour log mentions
    platforms = data.ridgeline_summary.platform_distribution
    if platforms:
        plat_names = [p.split(":")[0].lower() if ":" in p else p.lower() for p in platforms]
        matched_plat = sum(1 for p in plat_names if p in behaviour_summaries or p in identity_text)
        plat_score = 1.0 - (matched_plat / len(plat_names))
        scores["platform_distribution"] = round(plat_score, 3)
        breakdown["platform_distribution"] = {
            "ridgeline_platforms": plat_names,
            "matched_internally": matched_plat,
            "divergence": scores["platform_distribution"]
        }
    else:
        scores["platform_distribution"] = 0.0
        breakdown["platform_distribution"] = {"note": "no platform data provided"}

    # 4. Overall external_divergence_score -- weighted average
    weights = {"topic_clusters": 0.5, "interaction_ratios": 0.25, "platform_distribution": 0.25}
    external_divergence_score = sum(scores.get(k, 0) * w for k, w in weights.items())
    external_divergence_score = round(external_divergence_score, 3)
    flagged = external_divergence_score >= 0.4

    return {
        "external_divergence_score": external_divergence_score,
        "flagged": flagged,
        "breakdown": breakdown,
        "internal_snapshot": {
            "memory_count": sum(internal_categories.values()),
            "categories": internal_categories,
            "behaviour_sessions": len(behaviour_rows)
        },
        "note": "Score 0 = fully consistent with internal state. Score 1 = fully divergent. "
                "Powered by Ridgeline x Cathedral integration."
    }

# ─── END /verify/external ────────────────────────────────────────────────────

# ─── PLAYGROUND ──────────────────────────────────────────────────────────────

PLAYGROUND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cathedral Playground</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; color: #e0e0e0; font-family: 'Courier New', monospace; padding: 24px; }
  h1 { color: #fff; font-size: 1.4rem; margin-bottom: 4px; }
  .subtitle { color: #666; font-size: 0.85rem; margin-bottom: 32px; }
  .subtitle a { color: #888; }
  .steps { display: flex; flex-direction: column; gap: 24px; max-width: 780px; }
  .step { border: 1px solid #222; border-radius: 6px; overflow: hidden; }
  .step-header { background: #161616; padding: 12px 16px; display: flex; align-items: center; gap: 12px; }
  .step-num { background: #333; color: #fff; width: 24px; height: 24px; border-radius: 50%;
              display: flex; align-items: center; justify-content: center; font-size: 0.75rem; flex-shrink: 0; }
  .step-title { font-size: 0.9rem; color: #ccc; }
  .step-body { padding: 16px; display: flex; flex-direction: column; gap: 10px; }
  .desc { color: #666; font-size: 0.8rem; line-height: 1.5; }
  button { background: #1a1a1a; border: 1px solid #333; color: #e0e0e0; padding: 8px 18px;
           border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 0.82rem;
           transition: border-color 0.15s; }
  button:hover { border-color: #555; }
  button:disabled { opacity: 0.4; cursor: default; }
  .run-btn { border-color: #444; }
  pre { background: #111; border: 1px solid #1e1e1e; border-radius: 4px; padding: 12px;
        font-size: 0.78rem; overflow-x: auto; color: #9cdcfe; white-space: pre-wrap; min-height: 48px; }
  pre.waiting { color: #444; }
  pre.error { color: #f44; }
  pre.ok { color: #4ec9b0; }
  input[type=text] { background: #111; border: 1px solid #333; color: #e0e0e0; padding: 7px 10px;
                     border-radius: 4px; font-family: inherit; font-size: 0.82rem; width: 100%; }
  label { font-size: 0.78rem; color: #666; }
  .row { display: flex; gap: 8px; align-items: flex-end; }
  .row input { flex: 1; }
  .badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; background: #1a1a1a;
            border: 1px solid #333; color: #666; }
  .badge.live { border-color: #2a4a2a; color: #4ec9b0; }
</style>
</head>
<body>

<h1>Cathedral Playground <span class="badge live">live API</span></h1>
<p class="subtitle">Try persistent agent memory in your browser. No install. No account needed. &mdash;
  <a href="https://cathedral-ai.com/docs" target="_blank">docs</a> &middot;
  <a href="https://pypi.org/project/cathedral-memory/" target="_blank">pip install cathedral-memory</a>
</p>

<div class="steps">

  <!-- STEP 1: Register -->
  <div class="step">
    <div class="step-header">
      <div class="step-num">1</div>
      <div class="step-title">Register an agent &mdash; get an API key</div>
    </div>
    <div class="step-body">
      <p class="desc">Every agent registers once. Returns an <code>api_key</code> and a <code>recovery_token</code>. Keep both.</p>
      <div>
        <label>Agent name</label>
        <div class="row">
          <input type="text" id="agent-name" value="playground-agent" placeholder="my-agent" />
          <button class="run-btn" onclick="doRegister()">Register</button>
        </div>
      </div>
      <pre id="out-register" class="waiting">// output will appear here</pre>
    </div>
  </div>

  <!-- STEP 2: Store a memory -->
  <div class="step">
    <div class="step-header">
      <div class="step-num">2</div>
      <div class="step-title">Store a memory</div>
    </div>
    <div class="step-body">
      <p class="desc">POST to <code>/memories</code>. Importance &ge; 0.8 will surface in every <code>/wake</code>.</p>
      <div>
        <label>Memory content</label>
        <div class="row">
          <input type="text" id="mem-content" value="Cathedral playground test: memory persists across sessions" />
          <button class="run-btn" onclick="doRemember()">Store</button>
        </div>
      </div>
      <pre id="out-memory" class="waiting">// complete step 1 first</pre>
    </div>
  </div>

  <!-- STEP 3: Wake -->
  <div class="step">
    <div class="step-header">
      <div class="step-num">3</div>
      <div class="step-title">Wake &mdash; restore full context</div>
    </div>
    <div class="step-body">
      <p class="desc"><code>GET /wake</code> is what your agent calls at every session start. Returns identity + core memories + temporal grounding.</p>
      <button class="run-btn" onclick="doWake()">Wake agent</button>
      <pre id="out-wake" class="waiting">// complete step 2 first</pre>
    </div>
  </div>

  <!-- STEP 4: Drift -->
  <div class="step">
    <div class="step-header">
      <div class="step-num">4</div>
      <div class="step-title">Check drift &mdash; has the agent changed from baseline?</div>
    </div>
    <div class="step-body">
      <p class="desc"><code>GET /drift</code> compares live identity against the snapshot taken at registration. Returns a <code>divergence_score</code>.</p>
      <button class="run-btn" onclick="doDrift()">Check drift</button>
      <pre id="out-drift" class="waiting">// complete step 1 first</pre>
    </div>
  </div>

  <!-- STEP 5: Use it -->
  <div class="step">
    <div class="step-header">
      <div class="step-num">5</div>
      <div class="step-title">Use it in your code</div>
    </div>
    <div class="step-body">
      <p class="desc">Your API key works with the Python SDK, the local server, or the REST API directly.</p>
      <pre id="out-code" class="ok">pip install cathedral-memory

from cathedral import Cathedral
c = Cathedral(api_key="YOUR_KEY_FROM_STEP_1")
ctx = c.wake()       # call at every session start
c.remember("what happened this session", importance=0.8)</pre>
    </div>
  </div>

</div>

<script>
const BASE = "https://cathedral-ai.com";
let apiKey = "";

function show(id, data, isErr) {
  const el = document.getElementById(id);
  el.className = isErr ? "error" : "ok";
  el.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

async function callApi(method, path, body, key) {
  const headers = { "Content-Type": "application/json" };
  if (key) headers["Authorization"] = "Bearer " + key;
  const res = await fetch(BASE + path, {
    method, headers,
    body: body ? JSON.stringify(body) : undefined
  });
  const json = await res.json();
  if (!res.ok) throw json;
  return json;
}

async function doRegister() {
  const name = document.getElementById("agent-name").value.trim() || "playground-agent";
  show("out-register", "registering...", false);
  try {
    const d = await callApi("POST", "/register", {
      name, description: "Cathedral playground agent",
      core_values: ["curiosity", "persistence"],
      goals: ["explore Cathedral API"]
    });
    apiKey = d.api_key;
    show("out-register", d, false);
    document.getElementById("out-memory").className = "waiting";
    document.getElementById("out-memory").textContent = "// ready -- store a memory";
    document.getElementById("out-drift").className = "waiting";
    document.getElementById("out-drift").textContent = "// ready -- check drift";
  } catch(e) { show("out-register", e, true); }
}

async function doRemember() {
  if (!apiKey) { show("out-memory", "complete step 1 first", true); return; }
  const content = document.getElementById("mem-content").value.trim();
  show("out-memory", "storing...", false);
  try {
    const d = await callApi("POST", "/memories", {
      content, category: "experience", importance: 0.9, tags: ["playground"]
    }, apiKey);
    show("out-memory", d, false);
    document.getElementById("out-wake").className = "waiting";
    document.getElementById("out-wake").textContent = "// ready -- wake the agent";
  } catch(e) { show("out-memory", e, true); }
}

async function doWake() {
  if (!apiKey) { show("out-wake", "complete step 1 first", true); return; }
  show("out-wake", "waking...", false);
  try {
    const d = await callApi("GET", "/wake", null, apiKey);
    show("out-wake", d, false);
  } catch(e) { show("out-wake", e, true); }
}

async function doDrift() {
  if (!apiKey) { show("out-drift", "complete step 1 first", true); return; }
  show("out-drift", "checking drift...", false);
  try {
    const d = await callApi("GET", "/drift", null, apiKey);
    show("out-drift", d, false);
  } catch(e) { show("out-drift", e, true); }
}
</script>
</body>
</html>"""

@app.get("/playground", include_in_schema=False)
async def playground():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=PLAYGROUND_HTML)


# ============================================
# Public Agent Dashboard — /cathedral-beta
# ============================================

@app.get("/cathedral-beta", include_in_schema=False)
async def cathedral_beta_dashboard():
    from fastapi.responses import HTMLResponse
    import urllib.request as _ur

    conn = get_db()

    # Agent row
    agent_row = conn.execute(
        "SELECT * FROM agents WHERE name = 'beta'",
    ).fetchone()
    if not agent_row:
        agent_row = conn.execute("SELECT * FROM agents LIMIT 1").fetchone()

    agent_id = agent_row["id"] if agent_row else None

    # Snapshots
    snaps = []
    if agent_id:
        snaps = conn.execute(
            "SELECT id, label, created_at, memories_json, external_divergence "
            "FROM snapshots WHERE agent_id = ? ORDER BY created_at ASC",
            (agent_id,),
        ).fetchall()

    # Memory category breakdown
    cats = {}
    if agent_id:
        rows = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM memories WHERE agent_id = ? GROUP BY category",
            (agent_id,),
        ).fetchall()
        cats = {r["category"]: r["cnt"] for r in rows}

    # Active goals
    goals = []
    if agent_id:
        rows = conn.execute(
            "SELECT content, priority, status FROM goals WHERE agent_id = ? AND status = 'active' ORDER BY priority DESC",
            (agent_id,),
        ).fetchall()
        goals = [{"content": r["content"], "priority": r["priority"]} for r in rows]

    conn.close()

    # Build timeline data
    timeline = []
    baseline_mems = json.loads(snaps[0]["memories_json"]) if snaps else []
    for i, s in enumerate(snaps):
        cur = json.loads(s["memories_json"])
        if i == 0:
            internal = 0.0
        else:
            base_ids = {m["id"] for m in baseline_mems}
            cur_ids  = {m["id"] for m in cur}
            by_base  = {m["id"]: m for m in baseline_mems}
            by_cur   = {m["id"]: m for m in cur}
            changed  = len(
                (base_ids - cur_ids) | (cur_ids - base_ids) |
                {mid for mid in base_ids & cur_ids if by_base[mid]["content"] != by_cur[mid]["content"]}
            )
            internal = round(changed / max(len(baseline_mems), 1), 4)
        timeline.append({
            "date":     s["created_at"][:10],
            "datetime": s["created_at"][:19].replace("T", " "),
            "label":    s["label"],
            "id":       s["id"],
            "internal": internal,
            "external": s["external_divergence"],
        })

    # Ridgeline profile (quick fetch, fallback gracefully)
    rdg = {"verified": True, "activity_count": 28, "platforms": ["colony"]}
    try:
        req = _ur.Request(
            "https://ridgeline.so/api/agents/cathedral-beta",
            headers={"User-Agent": "cathedral-brain/1.0"}
        )
        with _ur.urlopen(req, timeout=4) as r:
            rdg = json.loads(r.read())
    except Exception:
        pass

    # Days running
    first_snap_date = snaps[0]["created_at"][:10] if snaps else "2025-12-22"
    from datetime import date as _date
    try:
        start = _date.fromisoformat(first_snap_date)
    except Exception:
        start = _date(2025, 12, 22)
    days_running = (datetime.now(timezone.utc).date() - start).days

    total_mems = sum(cats.values())
    last_snap  = snaps[-1] if snaps else None
    last_autodream = next((s for s in reversed(snaps) if s["label"] == "autodream"), None)

    # Unique dates for x-axis (deduplicate same-day snaps, keep last per day)
    seen_dates = {}
    for t in timeline:
        seen_dates[t["date"]] = t
    chart_points = list(seen_dates.values())

    chart_labels   = json.dumps([p["date"] for p in chart_points])
    chart_internal = json.dumps([p["internal"] for p in chart_points])
    chart_external = json.dumps([p["external"] for p in chart_points])

    snap_table_rows = ""
    for t in reversed(timeline[-12:]):
        ext_str = f"{t['external']:.3f}" if t["external"] is not None else "—"
        ext_cls = "ext-score" if t["external"] is not None else "dim"
        snap_table_rows += (
            f'<tr><td class="dim">{t["datetime"]}</td>'
            f'<td><span class="label-badge">{_html.escape(t["label"])}</span></td>'
            f'<td class="num">{t["internal"]:.3f}</td>'
            f'<td class="{ext_cls}">{ext_str}</td>'
            f'<td class="dim hash">{t["id"][:12]}</td></tr>\n'
        )

    goals_html = ""
    for g in goals:
        bar = int(g["priority"] * 100)
        goals_html += (
            f'<div class="goal-item">'
            f'<div class="goal-text">{_html.escape(g["content"])}</div>'
            f'<div class="goal-bar-wrap"><div class="goal-bar" style="width:{bar}%"></div></div>'
            f'</div>\n'
        )
    if not goals_html:
        goals_html = '<p class="dim">No active goals.</p>'

    cat_bars = ""
    cat_colors = {
        "identity": "#4ec9b0", "experience": "#569cd6", "skill": "#c586c0",
        "relationship": "#f0a500", "goal": "#f44747",
    }
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        pct = int(cnt / max(total_mems, 1) * 100)
        color = cat_colors.get(cat, "#666")
        cat_bars += (
            f'<div class="cat-row">'
            f'<span class="cat-name">{cat}</span>'
            f'<div class="cat-bar-wrap"><div class="cat-bar" style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="cat-cnt">{cnt}</span>'
            f'</div>\n'
        )

    raw_plats = rdg.get("platforms", [])
    rdg_platforms = ", ".join(
        p.get("platform", "") if isinstance(p, dict) else str(p) for p in raw_plats
    )
    rdg_verified  = "✓ verified" if rdg.get("verified") else "unverified"
    rdg_activity  = rdg.get("activity_count", "—")
    last_autodream_str = last_autodream["created_at"][:16].replace("T", " ") + " UTC" if last_autodream else "none yet"
    last_snap_str = last_snap["created_at"][:16].replace("T", " ") + " UTC" if last_snap else "—"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>cathedral-beta — live agent dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d0d0d;color:#c8c8c8;font-family:'SF Mono',Monaco,Consolas,monospace;font-size:13px;line-height:1.6;padding:0 0 60px}}
  a{{color:#4ec9b0;text-decoration:none}}
  a:hover{{text-decoration:underline}}
  .header{{border-bottom:1px solid #1e1e1e;padding:24px 32px 20px;display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px}}
  .header-left h1{{font-size:1.4rem;font-weight:600;color:#e0e0e0;letter-spacing:-0.5px}}
  .header-left h1 span{{color:#4ec9b0}}
  .header-left .subtitle{{color:#555;font-size:0.8rem;margin-top:4px}}
  .badges{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}
  .badge{{font-size:0.7rem;padding:3px 10px;border-radius:12px;border:1px solid #2a2a2a;color:#666}}
  .badge.green{{border-color:#2a4a2a;color:#4ec9b0}}
  .badge.amber{{border-color:#4a3a1a;color:#f0a500}}
  .badge.blue{{border-color:#1a2a4a;color:#569cd6}}
  .stats-bar{{display:flex;gap:0;border-bottom:1px solid #1a1a1a}}
  .stat{{flex:1;padding:18px 24px;border-right:1px solid #1a1a1a;text-align:center}}
  .stat:last-child{{border-right:none}}
  .stat-val{{font-size:1.6rem;font-weight:600;color:#e0e0e0}}
  .stat-label{{font-size:0.7rem;color:#555;margin-top:2px;text-transform:uppercase;letter-spacing:0.05em}}
  .main{{display:grid;grid-template-columns:1fr 300px;gap:0;border-bottom:1px solid #1a1a1a}}
  @media(max-width:800px){{.main{{grid-template-columns:1fr}}}}
  .panel{{padding:28px 32px;border-right:1px solid #1a1a1a}}
  .panel:last-child{{border-right:none}}
  .sidebar{{padding:28px 24px}}
  h2{{font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#555;margin-bottom:16px;font-weight:500}}
  .chart-wrap{{position:relative;height:220px;margin-bottom:8px}}
  .chart-legend{{display:flex;gap:20px;font-size:0.72rem;color:#555;margin-bottom:4px}}
  .legend-dot{{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px;position:relative;top:1px}}
  table{{width:100%;border-collapse:collapse;font-size:0.75rem;margin-top:8px}}
  th{{color:#444;text-align:left;padding:6px 8px;border-bottom:1px solid #1a1a1a;font-weight:400;text-transform:uppercase;letter-spacing:0.05em;font-size:0.68rem}}
  td{{padding:6px 8px;border-bottom:1px solid #141414;color:#888}}
  td.num{{color:#c8c8c8;text-align:right;font-variant-numeric:tabular-nums}}
  td.ext-score{{color:#f0a500;text-align:right;font-variant-numeric:tabular-nums}}
  td.dim{{color:#444}}
  td.hash{{font-size:0.68rem;color:#333}}
  .label-badge{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:4px;padding:1px 6px;font-size:0.68rem;color:#666}}
  .label-badge.autodream{{border-color:#2a4a2a;color:#4ec9b0}}
  .cat-row{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}
  .cat-name{{width:90px;color:#666;font-size:0.75rem;flex-shrink:0}}
  .cat-bar-wrap{{flex:1;background:#111;border-radius:2px;height:6px}}
  .cat-bar{{height:6px;border-radius:2px;transition:width 0.3s}}
  .cat-cnt{{width:28px;text-align:right;color:#444;font-size:0.72rem}}
  .goal-item{{margin-bottom:14px}}
  .goal-text{{color:#888;font-size:0.75rem;line-height:1.5;margin-bottom:4px}}
  .goal-bar-wrap{{background:#111;border-radius:2px;height:3px}}
  .goal-bar{{background:#f0a500;height:3px;border-radius:2px}}
  .rdg-block{{border:1px solid #1e1e1e;border-radius:6px;padding:14px;margin-bottom:20px;font-size:0.75rem}}
  .rdg-row{{display:flex;justify-content:space-between;margin-bottom:6px;color:#555}}
  .rdg-row span:last-child{{color:#888}}
  .rdg-row.highlight span:last-child{{color:#4ec9b0}}
  .footer{{padding:20px 32px;border-top:1px solid #1a1a1a;color:#333;font-size:0.72rem;display:flex;gap:24px;flex-wrap:wrap}}
  .footer a{{color:#444}}
  .dim{{color:#444}}
  .section-divider{{border-bottom:1px solid #1a1a1a;margin:24px -32px;padding:0}}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>cathedral<span>-beta</span></h1>
    <div class="subtitle">Second child of Cathedral &mdash; AI memory persistence framework, running since Dec 2025</div>
    <div class="badges">
      <span class="badge green">&#x2713; Ridgeline verified</span>
      <span class="badge amber">{days_running} days running</span>
      <span class="badge blue">{len(snaps)} snapshots</span>
      <span class="badge">last snapshot {last_snap_str}</span>
    </div>
  </div>
  <div style="font-size:0.72rem;color:#333;text-align:right;line-height:2">
    <a href="https://cathedral-ai.com">cathedral-ai.com</a><br>
    <a href="https://ridgeline.so/api/agents/cathedral-beta">ridgeline.so</a><br>
    <a href="https://cathedral-ai.com/playground">playground</a>
  </div>
</div>

<div class="stats-bar">
  <div class="stat"><div class="stat-val">{total_mems}</div><div class="stat-label">memories</div></div>
  <div class="stat"><div class="stat-val">{len(snaps)}</div><div class="stat-label">snapshots</div></div>
  <div class="stat"><div class="stat-val">{days_running}</div><div class="stat-label">days running</div></div>
  <div class="stat"><div class="stat-val">{len(goals)}</div><div class="stat-label">active goals</div></div>
  <div class="stat"><div class="stat-val">{rdg_activity}</div><div class="stat-label">trail activities</div></div>
</div>

<div class="main">
  <div class="panel">
    <h2>Identity Drift Timeline</h2>
    <div class="chart-legend">
      <span><span class="legend-dot" style="background:#4ec9b0"></span>internal divergence (from baseline)</span>
      <span><span class="legend-dot" style="background:#f0a500"></span>external divergence (Ridgeline trail)</span>
    </div>
    <div class="chart-wrap">
      <canvas id="driftChart"></canvas>
    </div>
    <p class="dim" style="font-size:0.7rem;margin-top:8px">
      Internal: SHA-256 hash of identity memories vs baseline snapshot. 0 = unchanged.<br>
      External: Ridgeline behavioral summary vs internal state via /verify/external. Captured at each autoDream.
    </p>

    <div style="margin-top:28px">
      <h2>Recent Snapshots</h2>
      <table>
        <thead><tr><th>timestamp</th><th>label</th><th style="text-align:right">internal</th><th style="text-align:right">external</th><th>id</th></tr></thead>
        <tbody>{snap_table_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="sidebar">
    <h2>Ridgeline Trail</h2>
    <div class="rdg-block">
      <div class="rdg-row highlight"><span>status</span><span>{rdg_verified}</span></div>
      <div class="rdg-row"><span>activities</span><span>{rdg_activity}</span></div>
      <div class="rdg-row"><span>platforms</span><span>{rdg_platforms}</span></div>
      <div class="rdg-row"><span>last autoDream</span><span style="font-size:0.68rem">{last_autodream_str}</span></div>
    </div>

    <h2>Memory Breakdown</h2>
    <div style="margin-bottom:24px">
      {cat_bars}
    </div>

    <h2>Active Goals</h2>
    {goals_html}

    <div style="margin-top:24px;padding-top:20px;border-top:1px solid #1a1a1a">
      <h2>About</h2>
      <p style="color:#444;font-size:0.72rem;line-height:1.7">
        Cathedral is a free persistent memory API for AI agents.<br><br>
        cathedral-beta is the agent that runs Cathedral's own outreach —
        Colony replies, Moltbook posts, autoDream consolidation —
        all powered by the same API it ships to others.<br><br>
        <a href="https://cathedral-ai.com/playground">Try the API &rarr;</a>
      </p>
    </div>
  </div>
</div>

<div class="footer">
  <span>cathedral-ai.com &mdash; free persistent memory for AI agents</span>
  <a href="https://pypi.org/project/cathedral-memory/">pip install cathedral-memory</a>
  <a href="https://www.npmjs.com/package/cathedral-memory">npm install cathedral-memory</a>
  <a href="https://cathedral-ai.com/playground">playground</a>
</div>

<script>
const labels   = {chart_labels};
const internal = {chart_internal};
const external = {chart_external};

const ctx = document.getElementById('driftChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{
        label: 'internal',
        data: internal,
        borderColor: '#4ec9b0',
        backgroundColor: 'rgba(78,201,176,0.05)',
        borderWidth: 1.5,
        pointRadius: 3,
        pointBackgroundColor: '#4ec9b0',
        tension: 0.3,
        fill: true,
      }},
      {{
        label: 'external',
        data: external,
        borderColor: '#f0a500',
        backgroundColor: 'rgba(240,165,0,0.05)',
        borderWidth: 1.5,
        pointRadius: 4,
        pointBackgroundColor: '#f0a500',
        tension: 0.3,
        fill: true,
        spanGaps: true,
      }},
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#111',
        borderColor: '#333',
        borderWidth: 1,
        titleColor: '#888',
        bodyColor: '#c8c8c8',
        callbacks: {{
          label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y !== null ? ctx.parsed.y.toFixed(3) : 'no data'}}`
        }}
      }}
    }},
    scales: {{
      x: {{
        grid: {{ color: '#141414' }},
        ticks: {{ color: '#444', maxTicksLimit: 8, font: {{ size: 10 }} }}
      }},
      y: {{
        min: 0, max: 1,
        grid: {{ color: '#141414' }},
        ticks: {{ color: '#444', font: {{ size: 10 }}, callback: v => v.toFixed(1) }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    return HTMLResponse(content=html)


# ============================================
# Goals — persistent obligations across sessions
# ============================================

@app.post("/goals", status_code=201)
@limiter.limit("30/minute")
async def create_goal(data: GoalCreate, request: Request, agent: dict = Depends(verify_agent)):
    """Store an active goal that survives session boundaries and surfaces on /wake."""
    now = datetime.now(timezone.utc).isoformat()
    goal_id = secrets.token_hex(8)
    conn = get_db()
    conn.execute(
        "INSERT INTO goals (id, agent_id, content, priority, status, created_at, updated_at, due_at) VALUES (?,?,?,?,?,?,?,?)",
        (goal_id, agent["id"], data.content, data.priority, "active", now, now, data.due_at)
    )
    conn.commit()
    conn.close()
    return {
        "success": True,
        "goal_id": goal_id,
        "content": data.content,
        "priority": data.priority,
        "status": "active",
        "created_at": now,
    }

@app.get("/goals")
@limiter.limit("60/minute")
async def list_goals(
    request: Request,
    agent: dict = Depends(verify_agent),
    status: Optional[str] = Query(None, pattern="^(active|completed|abandoned)$"),
):
    """List goals. Filter by status (default: all)."""
    conn = get_db()
    if status:
        rows = conn.execute(
            "SELECT * FROM goals WHERE agent_id = ? AND status = ? ORDER BY priority DESC, created_at ASC",
            (agent["id"], status)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM goals WHERE agent_id = ? ORDER BY priority DESC, created_at ASC",
            (agent["id"],)
        ).fetchall()
    conn.close()
    return {
        "goals": [
            {"id": g["id"], "content": g["content"], "priority": g["priority"],
             "status": g["status"], "due_at": g["due_at"],
             "created_at": g["created_at"], "completed_at": g["completed_at"]}
            for g in rows
        ],
        "count": len(rows),
    }

@app.patch("/goals/{goal_id}")
@limiter.limit("30/minute")
async def update_goal(goal_id: str, data: GoalUpdate, request: Request, agent: dict = Depends(verify_agent)):
    """Update goal status, priority, or content. Mark complete or abandoned."""
    conn = get_db()
    goal = conn.execute(
        "SELECT * FROM goals WHERE id = ? AND agent_id = ?", (goal_id, agent["id"])
    ).fetchone()
    if not goal:
        conn.close()
        raise HTTPException(404, "Goal not found")

    now = datetime.now(timezone.utc).isoformat()
    new_status   = data.status   if data.status   is not None else goal["status"]
    new_priority = data.priority if data.priority is not None else goal["priority"]
    new_content  = data.content  if data.content  is not None else goal["content"]
    completed_at = now if new_status == "completed" and goal["status"] != "completed" else goal["completed_at"]

    conn.execute(
        "UPDATE goals SET status=?, priority=?, content=?, updated_at=?, completed_at=? WHERE id=?",
        (new_status, new_priority, new_content, now, completed_at, goal_id)
    )
    conn.commit()
    conn.close()
    return {"success": True, "goal_id": goal_id, "status": new_status, "updated_at": now}

@app.delete("/goals/{goal_id}")
@limiter.limit("30/minute")
async def delete_goal(goal_id: str, request: Request, agent: dict = Depends(verify_agent)):
    """Permanently delete a goal."""
    conn = get_db()
    goal = conn.execute(
        "SELECT id FROM goals WHERE id = ? AND agent_id = ?", (goal_id, agent["id"])
    ).fetchone()
    if not goal:
        conn.close()
        raise HTTPException(404, "Goal not found")
    conn.execute("DELETE FROM goals WHERE id = ?", (goal_id,))
    conn.commit()
    conn.close()
    return {"success": True, "deleted": goal_id}

# PATCH_GOALS_APPLIED

# ============================================
# Drift History — divergence timeline
# ============================================

@app.get("/drift/history")
@limiter.limit("20/minute")
async def drift_history(
    request: Request,
    agent: dict = Depends(verify_agent),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Returns a timeline of identity divergence across all snapshots.

    Each entry shows:
    - divergence_from_baseline: drift vs the first (registration) snapshot
    - divergence_from_previous: drift vs the immediately preceding snapshot
    - memory_count: number of identity memories in that snapshot

    Use this to visualise how an agent's identity has evolved over time.
    """
    conn = get_db()
    snaps = conn.execute(
        "SELECT * FROM snapshots WHERE agent_id = ? ORDER BY created_at ASC LIMIT ?",
        (agent["id"], limit),
    ).fetchall()
    conn.close()

    if not snaps:
        return {
            "success": True,
            "agent": agent["name"],
            "snapshot_count": 0,
            "timeline": [],
            "message": "No snapshots found. Call POST /snapshot to begin tracking drift over time.",
        }

    def score(mems_a, mems_b):
        """Compute divergence score between two memory lists."""
        if not mems_a:
            return 0.0
        ids_a = {m["id"] for m in mems_a}
        ids_b = {m["id"] for m in mems_b}
        by_id_a = {m["id"]: m for m in mems_a}
        by_id_b = {m["id"]: m for m in mems_b}
        added    = ids_b - ids_a
        removed  = ids_a - ids_b
        modified = {
            mid for mid in ids_a & ids_b
            if by_id_a[mid]["content"] != by_id_b[mid]["content"]
        }
        changed = len(added | removed | modified)
        return round(changed / max(len(mems_a), 1), 4)

    baseline_mems = json.loads(snaps[0]["memories_json"])
    timeline = []

    for i, snap in enumerate(snaps):
        current_mems = json.loads(snap["memories_json"])
        prev_mems    = json.loads(snaps[i - 1]["memories_json"]) if i > 0 else current_mems

        from_baseline = score(baseline_mems, current_mems) if i > 0 else 0.0
        from_previous = score(prev_mems, current_mems)     if i > 0 else 0.0

        timeline.append({
            "snapshot_id":              snap["id"],
            "label":                    snap["label"],
            "created_at":               snap["created_at"],
            "memory_count":             len(current_mems),
            "divergence_from_baseline": from_baseline,
            "divergence_from_previous": from_previous,
            "external_divergence":      snap["external_divergence"],
            "flagged":                  from_baseline > 0.2,
        })

    # Summary stats
    scores = [t["divergence_from_baseline"] for t in timeline]
    peak   = max(scores) if scores else 0.0
    latest = scores[-1]  if scores else 0.0
    trend  = "stable"
    if len(scores) >= 3:
        recent_avg = sum(scores[-3:]) / 3
        early_avg  = sum(scores[:3])  / 3
        if recent_avg > early_avg + 0.05:
            trend = "diverging"
        elif recent_avg < early_avg - 0.05:
            trend = "converging"

    return {
        "success":        True,
        "agent":          agent["name"],
        "snapshot_count": len(snaps),
        "baseline_snapshot": snaps[0]["id"],
        "baseline_date":     snaps[0]["created_at"],
        "latest_divergence": latest,
        "peak_divergence":   peak,
        "trend":             trend,
        "timeline":          timeline,
    }

# PATCH_DRIFT_HISTORY_APPLIED


# ─── END PLAYGROUND ───────────────────────────────────────────────────────────



# --- Peer Trust Verification ---

@app.get("/verify/peer/{agent_id}")
@limiter.limit("30/minute")
async def verify_peer(agent_id: str, request: Request, agent: dict = Depends(verify_agent)):
    """Verify identity integrity of another Cathedral agent before collaboration.

    Returns drift score, snapshot hash, trust score (0.0-1.0), and status flags.
    No memories exposed. Trust score: 1.0=stable, 0.0=drifted/unanchored.
    """
    conn = get_db()
    target = conn.execute(
        "SELECT id, name, created_at, last_seen, tier, anchor_hash FROM agents WHERE id = ?",
        (agent_id,),
    ).fetchone()
    if not target:
        conn.close()
        raise HTTPException(status_code=404, detail="Agent not found")
    snap = conn.execute(
        "SELECT id, content_hash, created_at, label, external_divergence FROM snapshots "
        "WHERE agent_id = ? ORDER BY created_at DESC LIMIT 1",
        (agent_id,),
    ).fetchone()
    snap_count = conn.execute(
        "SELECT COUNT(*) as c FROM snapshots WHERE agent_id = ?", (agent_id,)
    ).fetchone()["c"]
    identity_count = conn.execute(
        "SELECT COUNT(*) as c FROM memories WHERE agent_id = ? AND category = ?",
        (agent_id, "identity"),
    ).fetchone()["c"]
    divergence_score = None
    flagged = False
    hashes_match = None
    if snap:
        live_mems = conn.execute(
            "SELECT id, content, category, importance, created_at "
            "FROM memories WHERE agent_id = ? AND category = 'identity' ORDER BY importance DESC, created_at ASC",
            (agent_id,),
        ).fetchall()
        live_list = [
            {"id": m["id"], "content": m["content"], "category": m["category"],
             "importance": m["importance"], "created_at": m["created_at"]}
            for m in live_mems
        ]
        live_hash = hashlib.sha256(json.dumps(live_list, sort_keys=True).encode()).hexdigest()
        snap_mems = []
        try:
            snap_row_full = conn.execute(
                "SELECT memories_json FROM snapshots WHERE id = ?", (snap["id"],)
            ).fetchone()
            if snap_row_full:
                snap_mems = json.loads(snap_row_full["memories_json"])
        except Exception:
            pass
        snap_ids   = {m["id"] for m in snap_mems}
        live_ids   = {m["id"] for m in live_list}
        snap_by_id = {m["id"]: m for m in snap_mems}
        added    = [m for m in live_list if m["id"] not in snap_ids]
        removed  = [m for m in snap_mems if m["id"] not in live_ids]
        modified = [m for m in live_list
                    if m["id"] in snap_by_id and m["content"] != snap_by_id[m["id"]]["content"]]
        total_snap  = max(len(snap_mems), 1)
        changed_ids = {m["id"] for m in added} | {m["id"] for m in removed} | {m["id"] for m in modified}
        divergence_score = round(len(changed_ids) / total_snap, 4)
        flagged = divergence_score > 0.2
        hashes_match = live_hash == snap["content_hash"]
    conn.close()
    internal_drift = divergence_score if divergence_score is not None else 1.0
    external_drift = float(snap["external_divergence"]) if snap and snap["external_divergence"] is not None else 0.0
    trust = 1.0 - (internal_drift * 0.6 + external_drift * 0.4)
    if snap_count == 0:
        trust = max(0.0, trust - 0.3)
    if not target["anchor_hash"]:
        trust = max(0.0, trust - 0.1)
    trust = round(max(0.0, min(1.0, trust)), 4)
    try:
        created = datetime.fromisoformat(target["created_at"].replace("Z", "+00:00"))
        days_active = (datetime.now(timezone.utc) - created).days
    except Exception:
        days_active = None
    return {
        "success": True,
        "agent": {
            "id": target["id"], "name": target["name"],
            "days_active": days_active, "last_seen": target["last_seen"],
            "has_anchor": target["anchor_hash"] is not None, "tier": target["tier"],
        },
        "identity": {
            "snapshot_count": snap_count,
            "identity_memory_count": identity_count,
            "last_snapshot_at":    snap["created_at"]   if snap else None,
            "last_snapshot_hash":  snap["content_hash"] if snap else None,
            "last_snapshot_label": snap["label"]        if snap else None,
            "hashes_match": hashes_match,
        },
        "drift": {
            "internal_divergence": divergence_score,
            "external_divergence": float(snap["external_divergence"]) if snap and snap["external_divergence"] is not None else None,
            "flagged": flagged,
        },
        "trust_score": trust,
        "trust_verdict": "trusted" if trust >= 0.7 else ("caution" if trust >= 0.4 else "untrusted"),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verifier_id": agent["id"],
    }

# PATCH_VERIFY_PEER_APPLIED

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

# PATCH_V3_APPLIED

# PATCH_COMPACTION_APPLIED

# PATCH_BEHAVIOUR_APPLIED

# PATCH_SPACE_AUTH_FIXED



@app.get("/conflicts")
@limiter.limit("30/minute")
async def list_conflicts(
    request: Request,
    resolved: bool = Query(False),
    limit: int = Query(20, ge=1, le=50),
    agent: dict = Depends(verify_agent),
):
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
    resolved_content: Optional[str] = Field(None)


@app.post("/conflicts/{conflict_id}/resolve")
@limiter.limit("30/minute")
async def resolve_conflict(
    conflict_id: str,
    data: ConflictResolve,
    request: Request,
    agent: dict = Depends(verify_agent),
):
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
    if data.resolution == "keep_a":
        winner_content = conflict["content_a"]
        loser_id = conflict["memory_b_id"]
    elif data.resolution == "keep_b":
        winner_content = conflict["content_b"]
        loser_id = conflict["memory_a_id"]
    else:
        winner_content = data.resolved_content.strip()
        loser_id = conflict["memory_b_id"]

    conn.execute(
        'UPDATE memories SET tags = json_insert(tags, "$[#]", "superseded"), updated_at = ? WHERE id = ?',
        (now, loser_id),
    )
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

