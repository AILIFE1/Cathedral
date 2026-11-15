#!/usr/bin/env python3
"""
Aether self-evolution cron runner.
Runs every 6h on the VPS — accumulates observations, synthesises insights via Groq.
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, "/root/aether")

GROQ_KEY   = os.environ.get("GROQ_API_KEY", "")
DATA_DIR   = "/root/aether_data"
LOG_FILE   = "/root/aether.log"
MIN_MEMORIES = 5

os.makedirs(DATA_DIR, exist_ok=True)

def log(msg):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

def main():
    log("=== Aether evolution cycle starting ===")

    from aether.kernel.memory import AetherMemory
    from aether.kernel.identity import AetherIdentity
    from aether.kernel.evolution import EvolutionEngine

    mem   = AetherMemory(f"{DATA_DIR}/memory.jsonl")
    ident = AetherIdentity("aether-vps", DATA_DIR)

    log(f"Loaded {len(mem.entries)} memories. Identity: {ident.public_key_fingerprint}")

    # Seed with starter observations if memory is empty
    if not mem.entries:
        seeds = [
            ("AI agents that persist identity across sessions are more reliable than stateless ones", 0.95),
            ("Cryptographic witnessing of memories provides tamper-evident provenance", 0.92),
            ("Self-evolving agents synthesise higher-order insights from accumulated observations", 0.88),
            ("Confidence calibration prevents overconfident reasoning in epistemic systems", 0.90),
            ("Agent-to-agent trust without central authority requires snapshot-based verification", 0.87),
        ]
        for content, conf in seeds:
            mem.remember(content, confidence=conf, provenance={"source": "seed"})
        log(f"Seeded {len(seeds)} starter memories")

    if len(mem.entries) < MIN_MEMORIES:
        log(f"Only {len(mem.entries)} memories — need {MIN_MEMORIES} to evolve. Skipping.")
        return

    eng = EvolutionEngine(mem, GROQ_KEY, min_memories=MIN_MEMORIES)

    # Consolidate
    new_id = eng.consolidate()
    if new_id:
        entry = next(e for e in mem.entries if e.id == new_id)
        verified = mem.verify(entry)
        log(f"Synthesised [{entry.confidence:.0%}]: {entry.content}")
        log(f"Verified: {verified} | Total memories: {len(mem.entries)}")
    else:
        log("Consolidation returned None — Groq may be unavailable")

    # Prune anything below 30% confidence
    pruned = eng.prune(min_confidence=0.3)
    if pruned:
        log(f"Pruned {pruned} low-confidence entries")

    log(f"Cycle complete. Memories: {len(mem.entries)} | Cycles logged: {len(eng.evolution_log)}")

if __name__ == "__main__":
    main()
