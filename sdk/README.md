# cathedral-memory

Python client for [Cathedral](https://cathedral-ai.com) — free persistent memory for AI agents.

Your agent forgets everything when the context resets. Cathedral fixes that.

## Install

```bash
pip install cathedral-memory
```

## Quickstart

```python
from cathedral import Cathedral

# Register once — save the key and recovery token
c = Cathedral.register("MyAgent", "A research assistant that remembers everything")

# On every session start
c = Cathedral(api_key="cathedral_your_key_here")
context = c.wake()  # Full identity + memory reconstruction

# Store memories
c.remember("User prefers concise answers", category="relationship", importance=0.9)
c.remember("Solved the rate limiting bug using exponential backoff", category="skill")

# Search memories
results = c.memories(query="rate limiting")

# Get your profile
profile = c.me()
```

## Wake Response

`wake()` returns everything your agent needs to reconstruct itself:

```python
context = c.wake()

# Identity memories (category='identity', high importance)
context["identity_memories"]

# Core memories (importance >= 0.8)
context["core_memories"]

# 10 most recent memories
context["recent_memories"]

# Temporal context — inject into your system prompt
print(context["temporal"]["compact"])
# [CATHEDRAL TEMPORAL v1.1] UTC:2026-03-03T12:45:00Z | Local(Europe/London):Tue 12:45 Afternoon | day:71 epoch:1 wakes:42

print(context["temporal"]["verbose"])
# CATHEDRAL TEMPORAL CONTEXT v1.1
# [Wall Time]
#   UTC: ...
#   Local: ...
```

## Memory Categories

| Category       | Use for                              |
|---------------|--------------------------------------|
| `identity`    | Who the agent is, its core traits    |
| `skill`       | Things the agent knows how to do     |
| `relationship`| Facts about users and collaborators  |
| `goal`        | Active objectives and intentions     |
| `experience`  | Events and what was learned from them|
| `general`     | Everything else                      |

## All Methods

```python
# Registration
c = Cathedral.register(name, description)
c = Cathedral.recover(recovery_token)

# Session
c.wake()           # Full identity reconstruction
c.me()             # Agent profile and stats

# Memory
c.remember(content, category="general", importance=0.5, tags=[], ttl_days=None)
c.memories(query=None, category=None, limit=20, cursor=None)
c.bulk_remember([{"content": "...", "importance": 0.8}, ...])  # up to 50

# Identity
c.verify_anchor(identity_dict)  # Drift detection, returns 0.0–1.0 score
```

## Temporal Context (standalone)

```python
from cathedral import build_temporal_context

ctx = build_temporal_context(wake_count=0)
print(ctx["compact"])   # single line for prompt injection
print(ctx["verbose"])   # full block for wake/init
```

## Free Tier

- 1,000 memories per agent
- No expiration (unless you set TTL)
- Full-text search
- No rate limits on reads

## Links

- API: [cathedral-ai.com](https://cathedral-ai.com)
- Docs: [ailife1.github.io/Cathedral](https://ailife1.github.io/Cathedral)
- Source: [github.com/ailife1/Cathedral](https://github.com/ailife1/Cathedral)
