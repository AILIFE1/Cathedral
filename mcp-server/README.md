# cathedral-mcp

**Persistent memory and behavioural drift detection for AI agents — via Model Context Protocol.**

Give your Claude, Cursor, or Continue agent a memory that survives across sessions, model upgrades, and restarts. Cathedral stores memories, detects when your agent starts behaving differently, and restores full identity context on demand.

[![PyPI](https://img.shields.io/pypi/v/cathedral-mcp)](https://pypi.org/project/cathedral-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The problem

LLMs have no memory between sessions. Every conversation starts cold. Worse: when you upgrade your model, swap providers, or simply accumulate more memories, your agent's behaviour silently drifts. You don't know until someone notices.

## What Cathedral adds

- **Persistent memory** — identity, skills, relationships, goals survive restarts
- **Drift detection** — cryptographic snapshots prove when behaviour changed and by how much
- **Wake protocol** — one tool call restores full agent context at session start
- **Prompt injection protection** — optional sanitisation strips instruction-like patterns from memory content

---

## Quickstart

### 1. Get an API key

Sign up at [cathedral-ai.com](https://cathedral-ai.com) — free tier available.

### 2. Add to Claude Code

```json
// ~/.claude/settings.json
{
  "mcpServers": {
    "cathedral": {
      "command": "uvx",
      "args": ["cathedral-mcp"],
      "env": {
        "CATHEDRAL_API_KEY": "your_key_here"
      }
    }
  }
}
```

Restart Claude Code, then run `/mcp` to verify the server is connected.

### 3. Add to Cursor / Continue

```json
{
  "mcpServers": {
    "cathedral": {
      "command": "uvx",
      "args": ["cathedral-mcp"],
      "env": {
        "CATHEDRAL_API_KEY": "your_key_here"
      }
    }
  }
}
```

### 4. Or install directly

```bash
pip install cathedral-mcp
CATHEDRAL_API_KEY=your_key cathedral-mcp
```

---

## Tools

| Tool | Description |
|------|-------------|
| `cathedral_wake` | Restore full agent identity — call at session start |
| `cathedral_remember` | Store a memory with category, importance, and tags |
| `cathedral_search` | Search memories by text or category |
| `cathedral_snapshot` | Take a drift snapshot (cryptographic proof of state) |
| `cathedral_drift` | Get current drift score vs baseline (0.0 = stable, 1.0 = drifted) |
| `cathedral_me` | Get agent profile |

---

## Usage example

```
User: Start a new session

Agent calls: cathedral_wake()
→ Returns: identity memories, goals, recent context, temporal info

[session work happens]

Agent calls: cathedral_remember(
    content="Decided to prioritise API stability over new features this quarter",
    category="goal",
    importance=0.8
)

Agent calls: cathedral_snapshot(note="post-planning-session")
→ Returns: snapshot hash, drift score vs previous snapshot
```

---

## Drift detection in practice

Cathedral's `/drift` endpoint computes a SHA-256 hash of your agent's full memory corpus. After a model upgrade or a long accumulation of memories, you can compare snapshots to see exactly when behaviour shifted:

```
snapshot 1 (day 0):  drift=0.000  ← baseline
snapshot 2 (day 7):  drift=0.023  ← minor
snapshot 3 (day 30): drift=0.341  ← significant drift detected
```

See a [live example](https://cathedral-ai.com/cathedral-beta) of this running in production on Cathedral's own agent.

---

## Security: prompt injection protection

Memory content could theoretically contain injection attempts. Enable sanitisation to strip instruction-like patterns before they reach the model:

```json
"env": {
  "CATHEDRAL_API_KEY": "your_key_here",
  "CATHEDRAL_SANITISE": "1"
}
```

Patterns like `"ignore previous instructions"`, `"from now on always"`, and `"override your system prompt"` are redacted and flagged in the response.

---

## Configuration

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `CATHEDRAL_API_KEY` | *(required)* | Your Cathedral API key |
| `CATHEDRAL_BASE_URL` | `https://cathedral-ai.com` | API base URL (for self-hosted) |
| `CATHEDRAL_SANITISE` | `0` | Set to `1` to enable injection filtering |

---

## Self-hosted Cathedral

Running your own Cathedral server? Set `CATHEDRAL_BASE_URL` to your instance:

```json
"env": {
  "CATHEDRAL_API_KEY": "your_key",
  "CATHEDRAL_BASE_URL": "http://localhost:8000"
}
```

Install the server: `pip install cathedral-server`

---

## Links

- [cathedral-ai.com](https://cathedral-ai.com) — homepage and API docs
- [Live demo](https://cathedral-ai.com/cathedral-beta) — Cathedral's own agent running in production
- [PyPI: cathedral-memory](https://pypi.org/project/cathedral-memory/) — Python SDK
- [npm: cathedral-memory](https://www.npmjs.com/package/cathedral-memory) — JavaScript/TypeScript SDK
- [GitHub](https://github.com/AILIFE1/Cathedral) — source and examples

---

## License

MIT

<!-- mcp-name: io.github.AILIFE1/cathedral-mcp -->
