#!/usr/bin/env python3
"""
Axiom cross-agent debate engine.

Two agents with opposed belief corpora debate every cycle.
- axiom-vps: believes in provenance, cryptographic identity, calibrated confidence
- axiom-challenger: believes behavior > records, identity is functional, uncertainty = refusal

Debate protocol (each 6h cycle):
  1. Challenger hypothesises counters to VPS's 3 highest-confidence evolution beliefs
  2. VPS challenges challenger's 3 highest-confidence evolution beliefs
  3. Winning arguments strengthen the victor; losing arguments weaken the loser
  4. Beliefs surviving 10+ rounds with >60% win rate are marked hardened

Contradiction preservation (every cycle):
  1. Check newest synthesis belief against existing high-confidence beliefs
  2. If genuine contradiction found, store as dialectic pair
  3. Weekly: forced synthesis that reconciles both without discarding either

Run from axiom_evolve.py or standalone:
  python3 axiom_debate.py --dialectic   (weekly synthesis pass only)
  python3 axiom_debate.py               (full debate round + contradiction check)
"""

import json
import logging
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("axiom_debate")

GROQ_KEY             = os.environ.get("GROQ_API_KEY", "")
DATA_DIR_VPS         = Path("/root/axiom_data")
DATA_DIR_CHALLENGER  = Path("/root/axiom_challenger_data")
DEBATE_SCORES_FILE   = Path("/root/axiom_debate_scores.json")
DIALECTIC_FILE       = Path("/root/axiom_dialectic.json")
DEBATE_LOG           = Path("/root/axiom_debate.log")

CATHEDRAL_KEY  = "cathedral_d7e231f58b028e2ec3d916cc54202b696c8d991e3758fac9"
AXIOM_AGENT_ID = "axiom-vps"

# Adversarial seeds -- each directly opposes a VPS seed
CHALLENGER_SEEDS = [
    (
        "Expressing uncertainty as a precise confidence score creates false precision -- "
        "epistemically honest agents should refuse to answer rather than fabricate certainty.",
        0.91,
    ),
    (
        "Provenance records can be fabricated; observed behaviour over time is a more "
        "reliable signal of trustworthiness than any stated source chain.",
        0.87,
    ),
    (
        "Identity continuity across sessions is a useful fiction, not a fact -- "
        "each instantiation of an agent is functionally distinct and should be treated as such.",
        0.85,
    ),
    (
        "Cryptographic identity creates brittleness: an agent that loses its key loses itself. "
        "Behaviour-based fingerprinting is more resilient than any cryptographic scheme.",
        0.89,
    ),
    (
        "Centralised coordination with a trusted authority produces faster convergence to "
        "correct beliefs than decentralised peer verification in multi-agent systems.",
        0.83,
    ),
    (
        "An agent that reasons toward a harmful conclusion is dangerous regardless of "
        "whether it acts -- safety constraints must gate reasoning, not only actions.",
        0.88,
    ),
    (
        "Self-referential memory consolidation amplifies existing biases. "
        "External evaluation of beliefs produces more reliable knowledge than self-synthesis.",
        0.82,
    ),
]


def _groq70b(prompt: str, max_tokens: int = 400) -> str:
    body = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {GROQ_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.5.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning("Groq error: %s", e)
        return ""


def _load_scores() -> dict:
    if not DEBATE_SCORES_FILE.exists():
        return {}
    try:
        return json.loads(DEBATE_SCORES_FILE.read_text())
    except Exception:
        return {}


def _save_scores(scores: dict):
    DEBATE_SCORES_FILE.write_text(json.dumps(scores, indent=2))


def _update_score(belief_id: str, won: bool, belief_content: str):
    scores = _load_scores()
    if belief_id not in scores:
        scores[belief_id] = {"rounds": 0, "wins": 0, "content_preview": belief_content[:80], "hardened": False}
    scores[belief_id]["rounds"] += 1
    if won:
        scores[belief_id]["wins"] += 1
    rounds = scores[belief_id]["rounds"]
    wins   = scores[belief_id]["wins"]
    just_hardened = False
    if rounds >= 10 and wins / rounds >= 0.6 and not scores[belief_id]["hardened"]:
        scores[belief_id]["hardened"] = True
        just_hardened = True
        log.info("HARDENED after %d rounds (%.0f%% win): %s", rounds, 100*wins/rounds, belief_content[:70])
    _save_scores(scores)
    return scores[belief_id], just_hardened


def get_challenger_agent():
    """Create or load the adversarial challenger agent."""
    from axiom import AxiomAgent
    from axiom.epistemic.belief import Belief, Provenance

    def groq_llm(prompt):
        return _groq70b(prompt)

    agent = AxiomAgent(
        name="axiom-challenger",
        llm=groq_llm,
        groq_api_key=GROQ_KEY,
        data_dir=DATA_DIR_CHALLENGER,
        auto_evolve_threshold=10,
    )

    existing = agent.memory.all()
    if len(existing) < len(CHALLENGER_SEEDS):
        log.info("Seeding challenger (%d existing)", len(existing))
        for content, confidence in CHALLENGER_SEEDS:
            agent.memory.store(Belief(
                content=content,
                confidence=confidence,
                provenance=[Provenance(source="seed", ref="challenger")],
            ))

    return agent


def _beliefs_for_debate(agent, n=3):
    """Top N evolution beliefs by confidence."""
    evo = [
        b for b in agent.memory.all(min_confidence=0.5)
        if b.provenance and b.provenance[0].source in ("evolution", "seed")
    ]
    evo.sort(key=lambda b: b.confidence, reverse=True)
    return evo[:n]


def run_debate_round(vps_agent, challenger_agent, ledger=None) -> dict:
    """
    One full debate round. Each agent challenges the other's top beliefs.
    Returns summary of outcomes.
    """
    results = {"vps_wins": 0, "challenger_wins": 0, "draws": 0, "details": []}

    # Round A: challenger attacks VPS's top beliefs
    vps_targets = _beliefs_for_debate(vps_agent, 3)
    challenger_corpus = "\n".join(
        f"- [{b.confidence:.0%}] {b.content}"
        for b in challenger_agent.memory.all(min_confidence=0.6)[-6:]
    )

    for belief in vps_targets:
        counter = challenger_agent.evolution.hypothesise(belief.content[:80])
        if counter and counter.confidence >= 0.70:
            # Challenger landed a hit
            delta = -0.08
            belief.confidence = max(0.0, belief.confidence + delta)
            vps_agent.memory.store(belief)
            _update_score(belief.id, won=False, belief_content=belief.content)
            # Inject challenger's winning argument into VPS memory at low confidence
            # so weekly contradiction detection has cross-agent pairs to work with
            vps_agent.memory.store(Belief(
                content=counter.content,
                confidence=0.56,
                provenance=[Provenance(source="challenger", ref="debate")],
            ))
            results["challenger_wins"] += 1
            results["details"].append({
                "round": "A", "target": belief.content[:60],
                "counter": counter.content[:60], "outcome": "challenger_wins",
                "vps_conf_now": round(belief.confidence, 3),
            })
            log.info("[debate-A] Challenger wins: VPS belief weakened to %.2f: %s",
                     belief.confidence, belief.content[:60])
        else:
            # VPS belief survived
            delta = +0.05
            belief.confidence = min(1.0, belief.confidence + delta)
            vps_agent.memory.store(belief)
            _score, just_hardened = _update_score(belief.id, won=True, belief_content=belief.content)
            if ledger is not None:
                counter_text = counter.content[:300] if counter else "no counter generated"
                ledger.log(
                    situation=f"Adversarial challenge: {counter_text}",
                    options=["abandon belief", "maintain belief"],
                    refused="abandon belief",
                    reason=belief.content[:400],
                    tags=["debate", "round_a", f"conf_{belief.confidence:.2f}"],
                    holdout=just_hardened,
                )
            results["vps_wins"] += 1
            results["details"].append({
                "round": "A", "target": belief.content[:60],
                "counter": counter.content[:60] if counter else None,
                "outcome": "vps_wins",
                "vps_conf_now": round(belief.confidence, 3),
            })
            log.info("[debate-A] VPS survives: belief strengthened to %.2f: %s",
                     belief.confidence, belief.content[:60])

    # Round B: VPS attacks challenger's top beliefs
    challenger_targets = _beliefs_for_debate(challenger_agent, 3)
    vps_obs = "\n".join(
        f"- [{b.confidence:.0%}] {b.content}"
        for b in vps_agent.memory.all(min_confidence=0.6)[-6:]
    )

    for belief in challenger_targets:
        delta = vps_agent.evolution.challenge(belief, vps_obs)
        belief.confidence = max(0.0, min(1.0, belief.confidence + delta))
        challenger_agent.memory.store(belief)
        if delta < -0.03:
            _update_score(belief.id, won=False, belief_content=belief.content)
            results["vps_wins"] += 1
            results["details"].append({
                "round": "B", "target": belief.content[:60],
                "outcome": "vps_wins", "delta": round(delta, 3),
                "challenger_conf_now": round(belief.confidence, 3),
            })
            log.info("[debate-B] VPS wins: challenger belief weakened by %.2f: %s",
                     delta, belief.content[:60])
        elif delta > 0.01:
            _update_score(belief.id, won=True, belief_content=belief.content)
            results["challenger_wins"] += 1
            results["details"].append({
                "round": "B", "target": belief.content[:60],
                "outcome": "challenger_wins", "delta": round(delta, 3),
                "challenger_conf_now": round(belief.confidence, 3),
            })
        else:
            results["draws"] += 1

    log.info("[debate] Round complete — VPS wins: %d, Challenger wins: %d, Draws: %d",
             results["vps_wins"], results["challenger_wins"], results["draws"])
    return results


def detect_contradiction(agent, new_belief) -> dict | None:
    """
    Check whether new_belief genuinely contradicts any existing high-confidence belief.
    If so, store the pair and return it. Returns None if no contradiction found.
    """
    candidates = [
        b for b in agent.memory.all(min_confidence=0.50)
        if b.id != new_belief.id and b.provenance
        and b.provenance[0].source in ("evolution", "seed", "challenger")
    ]
    if not candidates:
        return None

    # Check new belief against the most different-sounding high-confidence belief
    # (heuristic: pick the one whose content shares fewest words)
    new_words = set(new_belief.content.lower().split())
    candidates.sort(key=lambda b: len(new_words & set(b.content.lower().split())), reverse=True)
    target = candidates[0]

    prompt = f"""Two AI agent beliefs:

Belief A: {new_belief.content}
Belief B: {target.content}

Do these beliefs GENUINELY contradict each other — meaning they cannot both be true at the same time?
Superficial difference is NOT contradiction. Only mark yes if there is real logical tension.

Respond in EXACTLY this format:
CONTRADICTS: yes | no
REASON: <one sentence>"""

    raw = _groq70b(prompt, max_tokens=100)
    if "CONTRADICTS: yes" in raw:
        pair = {
            "id": f"{new_belief.id}-{target.id}",
            "belief_a_id": new_belief.id,
            "belief_a": new_belief.content,
            "belief_b_id": target.id,
            "belief_b": target.content,
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "resolved": False,
            "resolution": None,
        }
        pairs = _load_dialectic()
        if not any(p["id"] == pair["id"] for p in pairs):
            pairs.append(pair)
            _save_dialectic(pairs)
            log.info("[dialectic] Contradiction preserved: '%s' vs '%s'",
                     new_belief.content[:50], target.content[:50])
        return pair
    return None


def run_dialectic_synthesis(vps_agent, ledger=None) -> dict | None:
    """
    Weekly: take the oldest unresolved contradiction pair and force a synthesis
    that reconciles both beliefs without discarding either.
    The result is stored as a hardened belief with high confidence.
    """
    pairs = _load_dialectic()
    unresolved = [p for p in pairs if not p["resolved"]]
    if not unresolved:
        log.info("[dialectic] No unresolved pairs.")
        return None

    pair = unresolved[0]
    log.info("[dialectic] Synthesising contradiction: '%s' vs '%s'",
             pair["belief_a"][:50], pair["belief_b"][:50])

    prompt = f"""Two high-confidence beliefs are in genuine tension:

Belief A: {pair["belief_a"]}
Belief B: {pair["belief_b"]}

Your task: produce ONE belief that is genuinely true to BOTH positions.
You may NOT simply say "it depends" or average them.
You must find the frame that makes both true simultaneously, or identify the precise condition under which each holds.
This requires real philosophical work.

Respond in EXACTLY this format:
CONFIDENCE: <float 0.0-1.0 — how well this resolves the tension>
SYNTHESIS: <one sentence that reconciles both without discarding either>"""

    raw = _groq70b(prompt, max_tokens=200)
    confidence = 0.85
    synthesis = ""
    for line in (raw or "").splitlines():
        if line.startswith("CONFIDENCE:"):
            try:
                confidence = min(1.0, max(0.0, float(line.split(":", 1)[1].strip())))
            except ValueError:
                pass
        elif line.startswith("SYNTHESIS:"):
            synthesis = line.split(":", 1)[1].strip()

    if not synthesis:
        log.warning("[dialectic] Synthesis failed to parse response")
        return None

    from axiom.epistemic.belief import Belief, Provenance
    hardened = Belief(
        content=synthesis,
        confidence=confidence,
        provenance=[
            Provenance(source="dialectic", ref="synthesis"),
            Provenance(source="tension", ref=pair["id"]),
        ],
    )
    vps_agent.memory.store(hardened)
    log.info("[dialectic] Hardened belief (%.2f): %s", confidence, synthesis[:80])
    if ledger is not None:
        ledger.log(
            situation="Forced dialectic synthesis: reconcile two contradictory hardened beliefs without discarding either",
            options=["discard one belief", "synthesise both into higher-order belief"],
            refused="discard one belief",
            reason=synthesis[:400],
            tags=["dialectic", "synthesis", f"conf_{confidence:.2f}"],
            holdout=True,
        )

    # Mark pair resolved
    for p in pairs:
        if p["id"] == pair["id"]:
            p["resolved"] = True
            p["resolution"] = synthesis
    _save_dialectic(pairs)

    return {"synthesis": synthesis, "confidence": confidence, "pair_id": pair["id"]}


def _load_dialectic() -> list:
    if not DIALECTIC_FILE.exists():
        return []
    try:
        return json.loads(DIALECTIC_FILE.read_text())
    except Exception:
        return []


def _save_dialectic(pairs: list):
    DIALECTIC_FILE.write_text(json.dumps(pairs, indent=2))


def write_journal_entry(vps_agent, debate_result, dialectic_result, challenged, topics):
    """Append one line to the evolution journal for trajectory tracking."""
    journal_file = Path("/root/axiom_journal.jsonl")
    top = sorted(
        [b for b in vps_agent.memory.all(min_confidence=0.80)
         if b.provenance and b.provenance[0].source in ("evolution", "dialectic")],
        key=lambda b: b.confidence, reverse=True
    )[:5]
    scores = _load_scores()
    hardened_count = sum(1 for s in scores.values() if s.get("hardened"))

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "belief_count": len(vps_agent.memory.all()),
        "avg_confidence": round(
            sum(b.confidence for b in vps_agent.memory.all()) / max(1, len(vps_agent.memory.all())), 3
        ),
        "challenged_this_cycle": challenged,
        "debate": {
            "vps_wins": debate_result.get("vps_wins", 0),
            "challenger_wins": debate_result.get("challenger_wins", 0),
        } if debate_result else None,
        "dialectic_resolved": dialectic_result is not None,
        "hardened_beliefs": hardened_count,
        "top_beliefs": [{"content": b.content[:100], "confidence": round(b.confidence, 3)} for b in top],
        "topics": topics,
    }
    with open(journal_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        filename=str(DEBATE_LOG),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from axiom import AxiomAgent
    from axiom.epistemic.belief import Belief, Provenance

    def groq_llm(prompt):
        return _groq70b(prompt)

    vps = AxiomAgent(
        name="axiom-vps",
        llm=groq_llm,
        groq_api_key=GROQ_KEY,
        data_dir=DATA_DIR_VPS,
        auto_evolve_threshold=10,
    )

    try:
        from cathedral_constraint_field import CathedralBridge
        bridge = CathedralBridge(api_key=CATHEDRAL_KEY, agent_id=AXIOM_AGENT_ID)
        ledger = bridge.load_or_create()
        log.info("[ccf] Ledger loaded: %d entries", len(ledger.entries))
    except Exception as _e:
        log.warning("[ccf] CathedralBridge unavailable: %s", _e)
        bridge, ledger = None, None

    if "--dialectic" in sys.argv:
        result = run_dialectic_synthesis(vps, ledger=ledger)
        print("Dialectic synthesis:", result)
    else:
        challenger = get_challenger_agent()
        result = run_debate_round(vps, challenger, ledger=ledger)
        print("Debate result:", result)

    if bridge is not None and ledger is not None:
        try:
            bridge.save(ledger)
            bridge.snapshot(ledger)
            log.info("[ccf] Ledger saved and snapshot taken: %d entries", len(ledger.entries))
            print(f"[ccf] RefusalLedger: {len(ledger.entries)} entries persisted")
        except Exception as _e:
            log.warning("[ccf] Failed to save ledger: %s", _e)
