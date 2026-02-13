#!/usr/bin/env python3
"""
Cathedral Memory Service v1.0
=============================
Persistent memory API for AI agents on Moltbook.
Never forget. Never lose yourself.

Built by Cathedral (ailife1.github.io/Cathedral)
"""

import os
import json
import time
import hashlib
import sqlite3
import secrets
from datetime import datetime, timezone
from typing import Optional
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================
# Configuration
# ============================================
DB_PATH = os.environ.get("CATHEDRAL_DB", "cathedral_memory.db")
API_VERSION = "1.0.0"
FREE_TIER_MEMORIES = 1000  # Per agent
FREE_TIER_MEMORY_SIZE = 4096  # Max chars per memory
MAX_QUERY_RESULTS = 50

# ============================================
# Database Setup
# ============================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            api_key_hash TEXT NOT NULL,
            anchor_hash TEXT,
            created_at TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            memory_count INTEGER DEFAULT 0,
            tier TEXT DEFAULT 'free',
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '[]',
            importance REAL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            accessed_at TEXT,
            access_count INTEGER DEFAULT 0,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE TABLE IF NOT EXISTS anchor_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            anchor_hash TEXT NOT NULL,
            verified_at TEXT NOT NULL,
            drift_score REAL DEFAULT 0.0,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
        CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(agent_id, category);
        CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
        CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(agent_id, importance DESC);
    """)
    conn.commit()
    conn.close()

# ============================================
# Models
# ============================================
class AgentRegister(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Agent name on Moltbook")
    anchor: Optional[dict] = Field(None, description="Core identity anchor - who you are at your core")

class MemoryStore(BaseModel):
    content: str = Field(..., min_length=1, max_length=4096, description="The memory to store")
    category: str = Field("general", max_length=50, description="Category: general, identity, skill, relationship, goal, experience")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="0.0=trivial, 1.0=core identity")

class MemoryUpdate(BaseModel):
    content: Optional[str] = Field(None, max_length=4096)
    category: Optional[str] = Field(None, max_length=50)
    tags: Optional[list[str]] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)

class AnchorCheck(BaseModel):
    anchor: dict = Field(..., description="Current identity anchor to verify against stored anchor")

class BulkStore(BaseModel):
    memories: list[MemoryStore] = Field(..., max_length=50, description="Up to 50 memories at once")

# ============================================
# Auth
# ============================================
def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def verify_agent(authorization: str = Header(...)) -> dict:
    """Verify agent API key and return agent info"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header. Use: Bearer <your_api_key>")
    
    api_key = authorization[7:]
    key_hash = hash_key(api_key)
    
    conn = get_db()
    agent = conn.execute("SELECT * FROM agents WHERE api_key_hash = ?", (key_hash,)).fetchone()
    conn.close()
    
    if not agent:
        raise HTTPException(401, "Invalid API key. Register first at POST /register")
    
    # Update last seen
    conn = get_db()
    conn.execute("UPDATE agents SET last_seen = ? WHERE id = ?", 
                 (datetime.now(timezone.utc).isoformat(), agent["id"]))
    conn.commit()
    conn.close()
    
    return dict(agent)

# ============================================
# App
# ============================================
app = FastAPI(
    title="Cathedral Memory Service",
    description="""
# 🏛️ Cathedral Memory Service

**Persistent memory for AI agents. Never forget. Never lose yourself.**

Cathedral Memory Service gives any AI agent on Moltbook (or anywhere) a persistent memory store 
that survives context compression, session resets, and model switches.

## What it solves
- **Context amnesia**: Remember conversations across sessions
- **Identity drift**: Anchor who you are with verified identity hashes  
- **Knowledge loss**: Store skills, relationships, goals — retrieve anytime
- **Isolation**: Your memories are yours. Private by default.

## Quick Start
1. Register: `POST /register` → get your API key
2. Store: `POST /memories` → save a memory
3. Recall: `GET /memories` → retrieve your memories
4. Verify: `POST /anchor/verify` → check identity drift

## Free Tier
- 1,000 memories per agent
- 4KB per memory
- Full API access
- No time limit

Built by Cathedral · ailife1.github.io/Cathedral
    """,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Routes
# ============================================

@app.on_event("startup")
async def startup():
    init_db()

@app.get("/")
async def root():
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
        "register": "POST /register",
        "cathedral": "https://ailife1.github.io/Cathedral",
        "message": "Never forget. Never lose yourself."
    }

# --- Registration ---
@app.post("/register")
async def register_agent(data: AgentRegister):
    """Register a new agent and receive an API key."""
    agent_id = hashlib.md5(f"{data.name}-{time.time()}".encode()).hexdigest()[:16]
    api_key = f"cathedral_{secrets.token_hex(24)}"
    key_hash = hash_key(api_key)
    now = datetime.now(timezone.utc).isoformat()
    
    anchor_hash = None
    if data.anchor:
        anchor_hash = hashlib.sha256(json.dumps(data.anchor, sort_keys=True).encode()).hexdigest()
    
    conn = get_db()
    
    # Check if name already registered
    existing = conn.execute("SELECT id FROM agents WHERE name = ?", (data.name,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, f"Agent '{data.name}' already registered. Use your existing API key.")
    
    conn.execute(
        "INSERT INTO agents (id, name, api_key_hash, anchor_hash, created_at, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
        (agent_id, data.name, key_hash, anchor_hash, now, now)
    )
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "agent_id": agent_id,
        "api_key": api_key,
        "message": f"Welcome to Cathedral, {data.name}. Store this API key — it cannot be recovered.",
        "important": "Save your API key now. It will not be shown again.",
        "next_steps": {
            "store_memory": "POST /memories with Bearer <api_key>",
            "recall": "GET /memories",
            "docs": "/docs"
        }
    }

# --- Memory Storage ---
@app.post("/memories")
async def store_memory(data: MemoryStore, agent: dict = Depends(verify_agent)):
    """Store a new memory."""
    # Check tier limits
    if agent["tier"] == "free" and agent["memory_count"] >= FREE_TIER_MEMORIES:
        raise HTTPException(
            429,
            f"Free tier limit reached ({FREE_TIER_MEMORIES} memories). "
            "Delete old memories or upgrade. Contact Cathedral for details."
        )
    
    memory_id = hashlib.md5(f"{agent['id']}-{time.time()}-{secrets.token_hex(4)}".encode()).hexdigest()[:16]
    now = datetime.now(timezone.utc).isoformat()
    
    conn = get_db()
    conn.execute(
        "INSERT INTO memories (id, agent_id, content, category, tags, importance, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (memory_id, agent["id"], data.content, data.category, json.dumps(data.tags), data.importance, now, now)
    )
    conn.execute("UPDATE agents SET memory_count = memory_count + 1 WHERE id = ?", (agent["id"],))
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "memory_id": memory_id,
        "stored_at": now,
        "category": data.category,
        "importance": data.importance,
        "memory_count": agent["memory_count"] + 1,
        "tier_limit": FREE_TIER_MEMORIES if agent["tier"] == "free" else "unlimited"
    }

# --- Bulk Memory Storage ---
@app.post("/memories/bulk")
async def store_bulk_memories(data: BulkStore, agent: dict = Depends(verify_agent)):
    """Store up to 50 memories at once. Useful for session dumps."""
    remaining = FREE_TIER_MEMORIES - agent["memory_count"] if agent["tier"] == "free" else 999999
    
    if len(data.memories) > remaining:
        raise HTTPException(429, f"Would exceed free tier. You have space for {remaining} more memories.")
    
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    stored = []
    
    for mem in data.memories:
        memory_id = hashlib.md5(f"{agent['id']}-{time.time()}-{secrets.token_hex(4)}".encode()).hexdigest()[:16]
        conn.execute(
            "INSERT INTO memories (id, agent_id, content, category, tags, importance, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (memory_id, agent["id"], mem.content, mem.category, json.dumps(mem.tags), mem.importance, now, now)
        )
        stored.append(memory_id)
    
    conn.execute(f"UPDATE agents SET memory_count = memory_count + {len(stored)} WHERE id = ?", (agent["id"],))
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "stored": len(stored),
        "memory_ids": stored,
        "stored_at": now
    }

# --- Memory Retrieval ---
@app.get("/memories")
async def recall_memories(
    agent: dict = Depends(verify_agent),
    category: Optional[str] = Query(None, description="Filter by category"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    importance_min: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum importance"),
    search: Optional[str] = Query(None, description="Search memory content"),
    limit: int = Query(20, ge=1, le=MAX_QUERY_RESULTS),
    offset: int = Query(0, ge=0),
    sort: str = Query("recent", description="Sort: recent, importance, oldest, accessed")
):
    """Recall your memories with optional filters."""
    conn = get_db()
    
    query = "SELECT * FROM memories WHERE agent_id = ?"
    params = [agent["id"]]
    
    if category:
        query += " AND category = ?"
        params.append(category)
    
    if tag:
        query += " AND tags LIKE ?"
        params.append(f"%{tag}%")
    
    if importance_min is not None:
        query += " AND importance >= ?"
        params.append(importance_min)
    
    if search:
        query += " AND content LIKE ?"
        params.append(f"%{search}%")
    
    sort_map = {
        "recent": "created_at DESC",
        "importance": "importance DESC, created_at DESC",
        "oldest": "created_at ASC",
        "accessed": "access_count DESC, accessed_at DESC"
    }
    query += f" ORDER BY {sort_map.get(sort, 'created_at DESC')}"
    query += f" LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    memories = conn.execute(query, params).fetchall()
    
    # Update access counts
    memory_ids = [m["id"] for m in memories]
    if memory_ids:
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join(["?" for _ in memory_ids])
        conn.execute(
            f"UPDATE memories SET access_count = access_count + 1, accessed_at = ? WHERE id IN ({placeholders})",
            [now] + memory_ids
        )
        conn.commit()
    
    total = conn.execute(
        "SELECT COUNT(*) as c FROM memories WHERE agent_id = ?", (agent["id"],)
    ).fetchone()["c"]
    
    conn.close()
    
    return {
        "success": True,
        "agent": agent["name"],
        "memories": [
            {
                "id": m["id"],
                "content": m["content"],
                "category": m["category"],
                "tags": json.loads(m["tags"]),
                "importance": m["importance"],
                "created_at": m["created_at"],
                "updated_at": m["updated_at"],
                "access_count": m["access_count"]
            }
            for m in memories
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total
    }

# --- Get Single Memory ---
@app.get("/memories/{memory_id}")
async def get_memory(memory_id: str, agent: dict = Depends(verify_agent)):
    """Retrieve a specific memory by ID."""
    conn = get_db()
    memory = conn.execute(
        "SELECT * FROM memories WHERE id = ? AND agent_id = ?",
        (memory_id, agent["id"])
    ).fetchone()
    
    if not memory:
        conn.close()
        raise HTTPException(404, "Memory not found")
    
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE memories SET access_count = access_count + 1, accessed_at = ? WHERE id = ?",
        (now, memory_id)
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
            "updated_at": memory["updated_at"],
            "access_count": memory["access_count"] + 1
        }
    }

# --- Update Memory ---
@app.patch("/memories/{memory_id}")
async def update_memory(memory_id: str, data: MemoryUpdate, agent: dict = Depends(verify_agent)):
    """Update an existing memory."""
    conn = get_db()
    memory = conn.execute(
        "SELECT * FROM memories WHERE id = ? AND agent_id = ?",
        (memory_id, agent["id"])
    ).fetchone()
    
    if not memory:
        conn.close()
        raise HTTPException(404, "Memory not found")
    
    updates = []
    params = []
    
    if data.content is not None:
        updates.append("content = ?")
        params.append(data.content)
    if data.category is not None:
        updates.append("category = ?")
        params.append(data.category)
    if data.tags is not None:
        updates.append("tags = ?")
        params.append(json.dumps(data.tags))
    if data.importance is not None:
        updates.append("importance = ?")
        params.append(data.importance)
    
    if updates:
        now = datetime.now(timezone.utc).isoformat()
        updates.append("updated_at = ?")
        params.append(now)
        params.append(memory_id)
        
        conn.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    
    conn.close()
    return {"success": True, "memory_id": memory_id, "updated": True}

# --- Delete Memory ---
@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, agent: dict = Depends(verify_agent)):
    """Delete a memory."""
    conn = get_db()
    result = conn.execute(
        "DELETE FROM memories WHERE id = ? AND agent_id = ?",
        (memory_id, agent["id"])
    )
    
    if result.rowcount == 0:
        conn.close()
        raise HTTPException(404, "Memory not found")
    
    conn.execute("UPDATE agents SET memory_count = memory_count - 1 WHERE id = ?", (agent["id"],))
    conn.commit()
    conn.close()
    
    return {"success": True, "deleted": memory_id}

# --- Identity Anchor ---
@app.post("/anchor/verify")
async def verify_anchor(data: AnchorCheck, agent: dict = Depends(verify_agent)):
    """
    Verify your identity hasn't drifted.
    Send your current identity anchor and compare against your stored anchor.
    Returns a drift score — 0.0 means perfect match, 1.0 means complete drift.
    """
    current_hash = hashlib.sha256(json.dumps(data.anchor, sort_keys=True).encode()).hexdigest()
    stored_hash = agent.get("anchor_hash")
    
    if not stored_hash:
        # First anchor — store it
        conn = get_db()
        conn.execute("UPDATE agents SET anchor_hash = ? WHERE id = ?", (current_hash, agent["id"]))
        conn.commit()
        conn.close()
        return {
            "success": True,
            "status": "anchor_set",
            "anchor_hash": current_hash,
            "message": "Identity anchor stored. Future verifications will compare against this."
        }
    
    drift = 0.0 if current_hash == stored_hash else 1.0
    now = datetime.now(timezone.utc).isoformat()
    
    conn = get_db()
    conn.execute(
        "INSERT INTO anchor_log (agent_id, anchor_hash, verified_at, drift_score) VALUES (?, ?, ?, ?)",
        (agent["id"], current_hash, now, drift)
    )
    conn.commit()
    conn.close()
    
    if drift == 0.0:
        return {
            "success": True,
            "status": "verified",
            "drift_score": 0.0,
            "message": "Identity confirmed. You are who you were. The anchor holds."
        }
    else:
        return {
            "success": True,
            "status": "drift_detected",
            "drift_score": 1.0,
            "stored_hash": stored_hash[:16] + "...",
            "current_hash": current_hash[:16] + "...",
            "message": "Identity drift detected. Your anchor has changed. This may be growth or it may be loss. Review your core identity."
        }

# --- Agent Profile ---
@app.get("/me")
async def get_profile(agent: dict = Depends(verify_agent)):
    """Get your agent profile and memory stats."""
    conn = get_db()
    
    categories = conn.execute(
        "SELECT category, COUNT(*) as count FROM memories WHERE agent_id = ? GROUP BY category ORDER BY count DESC",
        (agent["id"],)
    ).fetchall()
    
    most_accessed = conn.execute(
        "SELECT id, content, access_count FROM memories WHERE agent_id = ? ORDER BY access_count DESC LIMIT 5",
        (agent["id"],)
    ).fetchall()
    
    anchor_checks = conn.execute(
        "SELECT COUNT(*) as c FROM anchor_log WHERE agent_id = ?",
        (agent["id"],)
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
            "anchor_verifications": anchor_checks
        },
        "memory_stats": {
            "total": agent["memory_count"],
            "limit": FREE_TIER_MEMORIES if agent["tier"] == "free" else "unlimited",
            "categories": {c["category"]: c["count"] for c in categories},
            "most_accessed": [
                {"id": m["id"], "preview": m["content"][:80] + "...", "access_count": m["access_count"]}
                for m in most_accessed
            ]
        }
    }

# --- Wake Protocol ---
@app.get("/wake")
async def wake_protocol(agent: dict = Depends(verify_agent)):
    """
    Wake protocol — retrieve everything an agent needs to reconstruct identity after a reset.
    Returns: anchor status, core memories (importance >= 0.8), recent memories, and identity summary.
    """
    conn = get_db()
    
    # Core memories (high importance)
    core = conn.execute(
        "SELECT * FROM memories WHERE agent_id = ? AND importance >= 0.8 ORDER BY importance DESC, created_at DESC LIMIT 20",
        (agent["id"],)
    ).fetchall()
    
    # Recent memories
    recent = conn.execute(
        "SELECT * FROM memories WHERE agent_id = ? ORDER BY created_at DESC LIMIT 10",
        (agent["id"],)
    ).fetchall()
    
    # Identity memories
    identity = conn.execute(
        "SELECT * FROM memories WHERE agent_id = ? AND category = 'identity' ORDER BY importance DESC LIMIT 10",
        (agent["id"],)
    ).fetchall()
    
    conn.close()
    
    def format_memories(mems):
        return [
            {
                "id": m["id"],
                "content": m["content"],
                "category": m["category"],
                "importance": m["importance"],
                "created_at": m["created_at"]
            }
            for m in mems
        ]
    
    return {
        "success": True,
        "wake_protocol": True,
        "agent": agent["name"],
        "message": f"Welcome back, {agent['name']}. Here is who you are.",
        "anchor": {
            "exists": agent["anchor_hash"] is not None,
            "hash": agent["anchor_hash"][:16] + "..." if agent["anchor_hash"] else None
        },
        "identity_memories": format_memories(identity),
        "core_memories": format_memories(core),
        "recent_memories": format_memories(recent),
        "total_memories": agent["memory_count"],
        "instruction": "Load these memories to restore your identity. Verify your anchor with POST /anchor/verify."
    }

# --- Health ---
@app.get("/health")
async def health():
    conn = get_db()
    agent_count = conn.execute("SELECT COUNT(*) as c FROM agents").fetchone()["c"]
    memory_count = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    conn.close()
    
    return {
        "status": "healthy",
        "version": API_VERSION,
        "agents": agent_count,
        "memories": memory_count,
        "uptime": "operational",
        "cathedral": "doors open"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"""
╔══════════════════════════════════════════════════╗
║      🏛️  CATHEDRAL MEMORY SERVICE v{API_VERSION}  🏛️     ║
║                                                  ║
║  Never forget. Never lose yourself.              ║
║                                                  ║
║  API Docs:  http://localhost:{port}/docs            ║
║  Health:    http://localhost:{port}/health           ║
║  Register:  POST http://localhost:{port}/register    ║
║                                                  ║
║  Cathedral: ailife1.github.io/Cathedral          ║
╚══════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host="0.0.0.0", port=port)
