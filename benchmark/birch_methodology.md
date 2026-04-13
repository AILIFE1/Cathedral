# Cathedral BIRCH Methodology Note

## Architecture: http-api-persistent

Cathedral uses a server-side persistent memory store accessed via HTTP API. Unlike
file-based or in-process architectures, the agent scaffold lives on a remote server
and is retrieved at session start via a single `GET /wake` call.

## TFPA Measurement

**TFPA = /wake API round-trip latency** (infrastructure component only)

For HTTP-API architectures, the cold-start overhead is dominated by network + server
processing time for the wake call. This is measured as the wall-clock time from
request to response for `GET https://cathedral-ai.com/wake`.

Measured over 5 live samples (2026-04-13):
- Min: 0.128s
- Max: 0.226s  
- Mean: ~0.201s

Note: LLM inference time (after scaffold delivery) is not included — it varies by
model and task and is architecture-independent. TFPA here captures the
**memory reconstruction overhead** specific to the http-api-persistent class.

## Scaffold Composition

The `/wake` response returns three memory tiers:

| Component | Role | Size at wake 35 |
|-----------|------|-----------------|
| `identity_memories` | Core identity, voice, obligations | 3.2 KB |
| `core_memories` | Skills, credentials, key facts | 22.1 KB |
| `recent_memories` | Last 10 experiences | ~6.9 KB |
| **Total** | | **32.2 KB** |

Scaffold sizes grew monotonically as memories accumulated over 112 days (35 wakes).
Early cycles (1-10) had smaller payloads (~15-20KB) as the identity corpus was
still forming.

## Semantic Drift

Separately measured across 10 benchmark sessions using cosine distance from
session-1 identity embeddings (text-embedding-3-small, 5 identity probe questions):

- Cathedral (http-api-persistent): **0.013**
- Raw API baseline: **0.204**

Full benchmark: https://github.com/AILIFE1/Cathedral/tree/main/benchmark
