"""
Cathedral + CrewAI Integration
================================
Adds persistent identity state and drift detection to CrewAI crews.

Problem this solves:
    CrewAI agents silently change behaviour across session boundaries when
    context compression or memory rotation occurs. The agent completes the
    task, but its behavioral fingerprint has shifted — ghost lexicon decay,
    tool-call sequence changes, semantic drift. These don't raise exceptions.

How Cathedral fixes it:
    - Before kickoff: /wake restores the agent's identity state from the
      last run. Memories are injected into the agent's backstory so the
      agent starts the session knowing what it knew at the end of the last one.
    - After kickoff: /snapshot freezes the post-run state as a hash-verified
      record. Each snapshot is SHA-256 hashed and optionally BCH-anchored.
    - /drift compares the post-run snapshot against the pre-run baseline
      and returns a divergence score (0.0 = identical, 1.0 = fully changed).

Usage:
    pip install cathedral-memory crewai

    from crewai import Agent, Task, Crew
    from crewai_integration import MonitoredCrew

    crew = Crew(agents=[...], tasks=[...])

    monitored = MonitoredCrew(
        crew=crew,
        api_key="your_cathedral_api_key",
        agent_name="my-crew-agent",
    )

    result = monitored.kickoff(inputs={"topic": "AI trends"})

    print(result.output)
    print(f"Identity drift this run: {result.drift_score:.3f}")
    print(f"Snapshot: {result.snapshot_id}")

Get a free Cathedral API key: cathedral-ai.com
Live example of drift tracking in production: cathedral-ai.com/cathedral-beta
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    raise ImportError("pip install requests")


CATHEDRAL_BASE = "https://cathedral-ai.com"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MonitoredCrewResult:
    """Wraps CrewOutput with Cathedral drift metadata."""
    output: Any                        # original CrewOutput from crewai
    snapshot_id: Optional[str]         # Cathedral snapshot ID for this run
    drift_score: Optional[float]       # 0.0 = no drift, 1.0 = full drift
    drift_flagged: bool                # True if drift > 20% of identity state
    run_id: str                        # unique ID for this run
    memories_restored: int             # identity memories loaded at wake
    agent_name: str


# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------

class MonitoredCrew:
    """
    Drop-in wrapper for crewai.Crew that adds Cathedral identity persistence
    and drift detection around every kickoff() call.

    Parameters
    ----------
    crew : crewai.Crew
        The crew to wrap. Not modified.
    api_key : str
        Cathedral API key (get one free at cathedral-ai.com).
    agent_name : str
        Canonical name for this agent in Cathedral. Use the same name across
        runs — this is the identity anchor.
    inject_memories : bool
        If True, prepend restored identity memories to the backstory of every
        agent in the crew before kickoff(). Default True.
    snapshot_label_prefix : str
        Label prefix for Cathedral snapshots. Each run gets a label like
        "crew-run-<run_id[:8]>".
    timeout : int
        HTTP timeout in seconds for Cathedral API calls.
    """

    def __init__(
        self,
        crew: Any,
        api_key: str,
        agent_name: str,
        inject_memories: bool = True,
        snapshot_label_prefix: str = "crew-run",
        timeout: int = 10,
    ):
        self.crew = crew
        self.agent_name = agent_name
        self.inject_memories = inject_memories
        self.snapshot_label_prefix = snapshot_label_prefix
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "cathedral-crewai/1.0",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def kickoff(self, inputs: Optional[Dict] = None) -> MonitoredCrewResult:
        """
        Run crew.kickoff() with Cathedral identity state wrapping.

        1. Wake  — restore identity memories from last run
        2. Inject — prepend memories to agent backstories (optional)
        3. Run   — crew.kickoff(inputs)
        4. Snapshot — freeze post-run identity state
        5. Drift — score divergence from pre-run baseline
        """
        run_id = str(uuid.uuid4())
        label = f"{self.snapshot_label_prefix}-{run_id[:8]}"

        # 1. Wake
        memories, memories_restored = self._wake()

        # 2. Inject
        original_backstories: Dict[str, str] = {}
        if self.inject_memories and memories:
            original_backstories = self._inject(memories)

        # 3. Run
        crew_output = self.crew.kickoff(inputs=inputs or {})

        # 4. Restore backstories
        if original_backstories:
            self._restore_backstories(original_backstories)

        # 5. Snapshot
        snapshot_id = self._snapshot(label)

        # 6. Drift
        drift_score, drift_flagged = self._drift()

        return MonitoredCrewResult(
            output=crew_output,
            snapshot_id=snapshot_id,
            drift_score=drift_score,
            drift_flagged=drift_flagged,
            run_id=run_id,
            memories_restored=memories_restored,
            agent_name=self.agent_name,
        )

    def drift_history(self, limit: int = 20) -> List[Dict]:
        """
        Return the full drift timeline for this agent — every snapshot with
        internal and external divergence scores.

        Useful for tracking behavioral consistency across many crew runs.
        """
        try:
            resp = self._session.get(
                f"{CATHEDRAL_BASE}/drift/history",
                params={"limit": limit},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("timeline", [])
        except Exception:
            return []

    def remember(self, content: str, category: str = "experience",
                 importance: float = 0.7) -> Optional[str]:
        """
        Store a memory about what happened in this crew run.
        Returns the memory ID or None on failure.
        """
        try:
            resp = self._session.post(
                f"{CATHEDRAL_BASE}/memories",
                json={"content": content, "category": category,
                      "importance": importance},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("memory_id")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wake(self):
        """Restore identity state. Returns (memories list, count)."""
        try:
            resp = self._session.get(
                f"{CATHEDRAL_BASE}/wake",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            memories = data.get("identity_memories", [])
            return memories, len(memories)
        except Exception:
            return [], 0

    def _inject(self, memories: List[Dict]) -> Dict[str, str]:
        """
        Prepend identity memory summary to each agent's backstory.
        Returns original backstories so they can be restored after the run.
        """
        summary_lines = [f"- {m['content']}" for m in memories[:8]]
        prefix = (
            "[Identity context from last session]\n"
            + "\n".join(summary_lines)
            + "\n\n[Current task]\n"
        )
        originals = {}
        for agent in getattr(self.crew, "agents", []):
            orig = getattr(agent, "backstory", "") or ""
            originals[id(agent)] = (agent, orig)
            agent.backstory = prefix + orig
        return originals

    def _restore_backstories(self, originals: Dict):
        for agent, orig in originals.values():
            agent.backstory = orig

    def _snapshot(self, label: str) -> Optional[str]:
        try:
            resp = self._session.post(
                f"{CATHEDRAL_BASE}/snapshot",
                json={"label": label},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("snapshot_id")
        except Exception:
            return None

    def _drift(self):
        """Returns (drift_score, flagged). None if no snapshot exists yet."""
        try:
            resp = self._session.get(
                f"{CATHEDRAL_BASE}/drift",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            score = data.get("divergence_score")
            flagged = data.get("flagged", False)
            return score, flagged
        except Exception:
            return None, False


# ---------------------------------------------------------------------------
# Minimal usage example (runs without a real crew)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    API_KEY = os.environ.get("CATHEDRAL_API_KEY", "your_api_key_here")

    # --- Example with a real CrewAI crew ---
    #
    # from crewai import Agent, Task, Crew
    #
    # researcher = Agent(
    #     role="Researcher",
    #     goal="Find the latest AI memory research",
    #     backstory="Expert in AI systems with focus on memory architectures.",
    #     verbose=False,
    # )
    # task = Task(
    #     description="Summarise the three most important recent papers on AI agent memory.",
    #     expected_output="A bullet-point summary of three papers.",
    #     agent=researcher,
    # )
    # crew = Crew(agents=[researcher], tasks=[task], verbose=False)
    #
    # monitored = MonitoredCrew(
    #     crew=crew,
    #     api_key=API_KEY,
    #     agent_name="research-crew",
    # )
    #
    # result = monitored.kickoff(inputs={"focus": "memory"})
    # print(result.output)
    # print(f"Drift score: {result.drift_score}")
    # print(f"Snapshot:    {result.snapshot_id}")
    # print(f"Flagged:     {result.drift_flagged}")
    #
    # # Store a memory about what the crew learned
    # monitored.remember(
    #     content=f"Run {result.run_id[:8]}: drift={result.drift_score}",
    #     category="experience",
    #     importance=0.6,
    # )
    #
    # # View full drift history across all runs
    # for entry in monitored.drift_history():
    #     print(entry["created_at"][:10], entry["label"], entry["divergence_from_baseline"])

    # Smoke test against live API (no crew required)
    print("Testing Cathedral connection...")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {API_KEY}"
    session.headers["User-Agent"] = "cathedral-crewai/1.0"

    resp = session.get(f"{CATHEDRAL_BASE}/wake", timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        print(f"Connected. Agent: {data.get('agent')}")
        print(f"Identity memories: {len(data.get('identity_memories', []))}")
        print(f"Active goals: {len(data.get('active_goals', []))}")
    else:
        print(f"Connection failed: {resp.status_code}")
        print("Get a free API key at cathedral-ai.com")
