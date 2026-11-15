#!/usr/bin/env python3
"""
Axiom self-evolution v2 -- real-world feedback loop.

Every 6h:
  1. Drain observation queue (/root/axiom_observations.json)
  2. Fetch recent Cathedral memories as fresh observations
  3. Inject observations as low-confidence belief candidates
  4. Challenge weakest 3 evolution beliefs against observation corpus
  5. Evolve (consolidate 70b + prune at 0.5)
  6. Run cross-agent debate (axiom-vps vs axiom-challenger)
  7. Detect contradictions in updated corpus -> dialectic store
  8. Extract topic suggestions -> /root/axiom_topics.json
  9. Write journal entry -> /root/axiom_journal.jsonl
 10. Snapshot to Cathedral
"""

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    filename="/root/axiom.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("axiom_evolve")

GROQ_KEY       = os.environ.get("GROQ_API_KEY", "")
CATHEDRAL_KEY  = "cathedral_d7e231f58b028e2ec3d916cc54202b696c8d991e3758fac9"
CATHEDRAL_BASE = "https://cathedral-ai.com"
DATA_DIR       = Path("/root/axiom_data")
OBS_QUEUE      = Path("/root/axiom_observations.json")
TOPICS_FILE    = Path("/root/axiom_topics.json")

STARTER_BELIEFS = [
    ("Epistemic agents should express calibrated confidence rather than binary certainty.", 0.92),
    ("Provenance tracking -- knowing where a belief came from -- is essential for trustworthy AI systems.", 0.88),
    ("Agent identity drift can be detected via corpus hashing across sessions.", 0.85),
    ("Cryptographic identity allows agents to persist who they are across restarts.", 0.90),
    ("Multi-agent trust without a central authority requires snapshot-based peer verification.", 0.87),
    ("Safety constraints should gate actions rather than gate reasoning -- preserve epistemic freedom.", 0.83),
    ("Self-evolving agents synthesise higher-order beliefs by consolidating their own memory corpus.", 0.80),
]


def http(method, url, headers=None, body=None, timeout=15):
    data = json.dumps(body).encode() if body else None
    h = dict(headers or {})
    if "User-Agent" not in h:
        h["User-Agent"] = "axiom-evolve/2.0"
    if body:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


def drain_observation_queue():
    if not OBS_QUEUE.exists():
        return []
    try:
        observations = json.loads(OBS_QUEUE.read_text() or "[]")
        OBS_QUEUE.write_text("[]")
        log.info("Drained %d observations from queue", len(observations))
        return observations
    except Exception as e:
        log.warning("Failed to drain queue: %s", e)
        return []


def fetch_cathedral_memories():
    s, data = http("GET", f"{CATHEDRAL_BASE}/wake",
                   {"Authorization": f"Bearer {CATHEDRAL_KEY}"})
    if s != 200:
        log.warning("Cathedral /wake returned %d", s)
        return []
    memories = (
        data.get("core_memories", []) +
        data.get("recent_memories", []) +
        data.get("identity_memories", [])
    )
    memories.sort(key=lambda m: m.get("importance", 0.5), reverse=True)
    return memories[:8]


def observations_to_text(observations, cathedral_memories):
    lines = []
    for obs in observations[-10:]:
        t = obs.get("type", "")
        if t in ("colony_reply", "colony_challenge"):
            lines.append(f"Colony [{obs.get('author', '?')}]: {obs.get('content', '')[:150]}")
        elif t == "moltbook_engagement":
            lines.append(
                f"Moltbook: '{obs.get('title', '')[:60]}' got {obs.get('clicks', 0)} clicks"
            )
    for m in cathedral_memories[:5]:
        lines.append(f"Memory [{m.get('category', '?')}]: {m.get('content', '')[:120]}")
    return "\n".join(lines)


def inject_observations(mem, observations, cathedral_memories):
    from axiom.epistemic.belief import Belief, Provenance

    injected = 0
    existing = {b.content[:60] for b in mem.all()}

    for obs in observations:
        if obs.get("type") in ("colony_reply", "colony_challenge"):
            content = obs.get("content", "").strip()
            if len(content) > 20:
                preview = f"Observed: {content[:200]}"
                if preview[:60] not in existing:
                    mem.store(Belief(
                        content=preview,
                        confidence=0.65,
                        provenance=[Provenance(source="observation", ref=obs.get("type", ""))],
                    ))
                    injected += 1

    for m in cathedral_memories:
        content = m.get("content", "").strip()
        if len(content) > 20:
            preview = f"Observed: {content[:200]}"
            if preview[:60] not in existing:
                mem.store(Belief(
                    content=preview,
                    confidence=0.60,
                    provenance=[Provenance(source="cathedral", ref=m.get("id", ""))],
                ))
                injected += 1

    if injected:
        log.info("Injected %d new observations into belief corpus", injected)
    return injected


def challenge_cycle(agent, obs_text):
    if not obs_text.strip():
        return 0

    evo_beliefs = [
        b for b in agent.memory.all()
        if b.provenance and b.provenance[0].source == "evolution"
    ]
    if not evo_beliefs:
        return 0

    evo_beliefs.sort(key=lambda b: b.confidence)
    challenged = 0
    for belief in evo_beliefs[:3]:
        delta = agent.evolution.challenge(belief, obs_text)
        if delta != 0.0:
            belief.confidence = max(0.0, min(1.0, belief.confidence + delta))
            agent.memory.store(belief)
            log.info("Challenged (delta=%.2f new=%.2f): %s",
                     delta, belief.confidence, belief.content[:70])
            challenged += 1

    return challenged


def extract_topics(agent):
    top = sorted(
        agent.memory.all(min_confidence=0.72),
        key=lambda b: b.confidence,
        reverse=True,
    )[:8]
    if not top:
        return []

    corpus = "\n".join(f"- [{b.confidence:.0%}] {b.content}" for b in top)
    body = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": (
            "These are the highest-confidence beliefs of an AI agent studying epistemic honesty, "
            "agent identity, and persistent memory.\n\nBeliefs:\n" + corpus +
            "\n\nSuggest 3 specific post angles for an AI builders community. Each should be a "
            "concrete, debatable claim -- not a title, an angle that would provoke technical "
            "discussion.\n\nOutput exactly 3 lines starting with TOPIC: and nothing else."
        )}],
        "max_tokens": 200,
        "temperature": 0.85,
    }).encode()

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {GROQ_KEY}",
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0"},
        method="POST",
    )
    topics = []
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = json.loads(r.read())["choices"][0]["message"]["content"]
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("TOPIC:"):
                topics.append(line[6:].strip())
    except Exception as e:
        log.warning("Topic extraction failed: %s", e)
    return topics[:3]


def snapshot_to_cathedral(snap):
    s, d = http(
        "POST", f"{CATHEDRAL_BASE}/snapshot",
        {"Authorization": f"Bearer {CATHEDRAL_KEY}"},
        {"label": "axiom-evolution",
         "belief_count": snap.get("belief_count"),
         "avg_confidence": snap.get("avg_confidence"),
         "corpus_hash": snap.get("corpus_hash")},
    )
    if s in (200, 201):
        log.info("Cathedral snapshot: %s", d.get("snapshot_id", "?")[:12])
    else:
        log.warning("Cathedral snapshot failed: %d", s)


def main():
    try:
        from axiom import AxiomAgent
        from axiom.epistemic.belief import Belief, Provenance
        import axiom_debate

        def groq_llm(prompt):
            body = json.dumps({
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.7,
            }).encode()
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {GROQ_KEY}",
                         "Content-Type": "application/json",
                         "User-Agent": "curl/8.5.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]

        agent = AxiomAgent(
            name="axiom-vps",
            llm=groq_llm,
            groq_api_key=GROQ_KEY,
            data_dir=DATA_DIR,
            auto_evolve_threshold=10,
        )

        # Seed if sparse
        existing = agent.memory.all()
        if len(existing) < len(STARTER_BELIEFS):
            log.info("Seeding starter beliefs (%d existing)", len(existing))
            for content, confidence in STARTER_BELIEFS:
                agent.memory.store(Belief(
                    content=content,
                    confidence=confidence,
                    provenance=[Provenance(source="seed", ref="axiom_evolve")],
                ))

        # 1. Drain observation queue
        observations = drain_observation_queue()

        # 2. Fetch Cathedral memories
        cathedral_memories = fetch_cathedral_memories()
        log.info("Fetched %d Cathedral memories", len(cathedral_memories))

        # 3. Inject observations
        inject_observations(agent.memory, observations, cathedral_memories)

        # 4. Build observation text
        obs_text = observations_to_text(observations, cathedral_memories)

        # 5. Challenge weakest evolution beliefs
        challenged = challenge_cycle(agent, obs_text)
        if challenged:
            log.info("Challenged %d beliefs this cycle", challenged)

        # 6. Evolution cycle (70b consolidate + prune at 0.5)
        result = agent.evolution.evolve()
        log.info("Evolution: %s", result)

        # 7. Cross-agent debate
        debate_result = None
        try:
            challenger = axiom_debate.get_challenger_agent()
            debate_result = axiom_debate.run_debate_round(agent, challenger)
            log.info("Debate: vps_wins=%d challenger_wins=%d draws=%d",
                     debate_result["vps_wins"], debate_result["challenger_wins"],
                     debate_result["draws"])
        except Exception as e:
            log.warning("Debate failed: %s", e)

        # 8. Contradiction detection on newest synthesis belief
        dialectic_result = None
        try:
            evo_beliefs = sorted(
                [b for b in agent.memory.all()
                 if b.provenance and b.provenance[0].source == "evolution"],
                key=lambda b: b.timestamp, reverse=True
            )
            if evo_beliefs:
                contradiction = axiom_debate.detect_contradiction(agent, evo_beliefs[0])
                if contradiction:
                    log.info("Contradiction preserved: %s vs %s",
                             contradiction["belief_a"][:40],
                             contradiction["belief_b"][:40])
        except Exception as e:
            log.warning("Contradiction detection failed: %s", e)

        # 9. Extract topic suggestions
        topics = extract_topics(agent)
        if topics:
            TOPICS_FILE.write_text(json.dumps({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "topics": topics,
            }, indent=2))
            log.info("Topics written: %s", topics)

        # 10. Write journal entry
        try:
            axiom_debate.write_journal_entry(agent, debate_result, dialectic_result, challenged, topics)
        except Exception as e:
            log.warning("Journal write failed: %s", e)

        # 11. Snapshot
        snap = agent.snapshot()
        log.info(
            "Snapshot: beliefs=%s avg_conf=%.3f drift_base=%.4f drift_prev=%.4f",
            snap.get("belief_count"),
            snap.get("avg_confidence", 0),
            snap.get("divergence_from_baseline", 0),
            snap.get("divergence_from_previous", 0),
        )
        snapshot_to_cathedral(snap)

    except Exception as exc:
        log.exception("axiom_evolve failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
