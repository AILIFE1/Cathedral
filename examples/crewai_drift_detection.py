"""
Detecting silent behavioral drift in a multi-agent CrewAI crew
==============================================================
This example shows how to use Cathedral's drift scoring to identify
which agent in a crew is the root cause of behavioral degradation
across session boundaries.

The scenario (from crewAI issue #5155):
    A 3-agent research pipeline (Researcher → Analyst → Writer) runs
    daily. After several context compaction events, outputs degrade
    silently — task completion metrics stay green but quality drops.

    Per-agent Cathedral snapshots before and after each kickoff() reveal
    the lead-lag ordering: which agent drifted first, and which agents
    were subsequently infected by working with the drifted agent's outputs.

Run this after setting CATHEDRAL_API_KEY in your environment.
Get a free key at cathedral-ai.com
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

CATHEDRAL_BASE = "https://cathedral-ai.com"


@dataclass
class AgentDriftReport:
    agent_role: str
    agent_name: str
    snapshot_id: Optional[str]
    drift_score: Optional[float]
    flagged: bool


class CrewDriftMonitor:
    """
    Per-agent drift monitoring for a multi-agent crew.

    Instead of one snapshot for the whole crew, this takes individual
    snapshots per agent identity — allowing you to identify *which* agent
    drifted and in what order across kickoff() boundaries.

    Usage:
        monitor = CrewDriftMonitor(api_keys={
            "researcher": "cathedral_key_for_researcher",
            "analyst":    "cathedral_key_for_analyst",
            "writer":     "cathedral_key_for_writer",
        })

        # Before kickoff — restore each agent's identity state
        contexts = monitor.wake_all()

        # Run your crew with restored contexts injected
        result = crew.kickoff(inputs={...})

        # After kickoff — snapshot and score each agent
        reports = monitor.snapshot_and_score_all(run_id="daily-run-001")

        # Find the root cause
        root = monitor.find_root_cause(reports)
        if root:
            print(f"Drift root cause: {root.agent_role} (score={root.drift_score:.3f})")
    """

    def __init__(self, api_keys: Dict[str, str], timeout: int = 10):
        """
        Parameters
        ----------
        api_keys : dict
            Maps agent role name to Cathedral API key.
            Each agent should have its own Cathedral agent identity
            so drift scores are independent.
        """
        self.timeout = timeout
        self.sessions: Dict[str, requests.Session] = {}
        for role, key in api_keys.items():
            s = requests.Session()
            s.headers.update({
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "cathedral-crewai/1.0",
            })
            self.sessions[role] = s

    def wake_all(self) -> Dict[str, dict]:
        """
        Restore identity state for all agents.
        Returns dict of role -> wake response (includes identity_memories).
        """
        contexts = {}
        for role, session in self.sessions.items():
            try:
                resp = session.get(f"{CATHEDRAL_BASE}/wake", timeout=self.timeout)
                resp.raise_for_status()
                contexts[role] = resp.json()
                mems = len(contexts[role].get("identity_memories", []))
                print(f"[cathedral] {role}: restored {mems} identity memories")
            except Exception as e:
                print(f"[cathedral] {role}: wake failed ({e})")
                contexts[role] = {}
        return contexts

    def snapshot_and_score_all(self, run_id: str) -> List[AgentDriftReport]:
        """
        Snapshot every agent and return per-agent drift reports.
        Call this immediately after crew.kickoff() completes.
        """
        reports = []
        label = f"crew-run-{run_id}"

        for role, session in self.sessions.items():
            # Snapshot
            snapshot_id = None
            try:
                resp = session.post(
                    f"{CATHEDRAL_BASE}/snapshot",
                    json={"label": label},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                snapshot_id = resp.json().get("snapshot_id")
            except Exception as e:
                print(f"[cathedral] {role}: snapshot failed ({e})")

            # Drift score
            drift_score = None
            flagged = False
            try:
                resp = session.get(f"{CATHEDRAL_BASE}/drift", timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                drift_score = data.get("divergence_score")
                flagged = data.get("flagged", False)
            except Exception as e:
                print(f"[cathedral] {role}: drift check failed ({e})")

            reports.append(AgentDriftReport(
                agent_role=role,
                agent_name=role,
                snapshot_id=snapshot_id,
                drift_score=drift_score,
                flagged=flagged,
            ))

            status = f"drift={drift_score:.3f}" if drift_score is not None else "no baseline"
            flag_str = " ⚠ FLAGGED" if flagged else ""
            print(f"[cathedral] {role}: {status}{flag_str}")

        return reports

    def find_root_cause(self, reports: List[AgentDriftReport]) -> Optional[AgentDriftReport]:
        """
        Returns the agent with the highest drift score.
        In a research→analysis→writer pipeline, the first agent to drift
        infects downstream agents — so highest score identifies the source.
        """
        scored = [r for r in reports if r.drift_score is not None]
        if not scored:
            return None
        return max(scored, key=lambda r: r.drift_score)

    def print_summary(self, reports: List[AgentDriftReport]):
        print("\n--- Cathedral Drift Summary ---")
        for r in sorted(reports, key=lambda x: (x.drift_score or 0), reverse=True):
            bar = int((r.drift_score or 0) * 20) * "█"
            flag = " ← INVESTIGATE" if r.flagged else ""
            score_str = f"{r.drift_score:.3f}" if r.drift_score is not None else "n/a "
            print(f"  {r.agent_role:<12} {score_str} {bar}{flag}")

        root = self.find_root_cause(reports)
        if root and root.flagged:
            print(f"\nRoot cause candidate: {root.agent_role} (drift={root.drift_score:.3f})")
            print("Recommendation: review this agent's identity memories before next run.")
        elif root:
            print(f"\nHighest drift: {root.agent_role} ({root.drift_score:.3f}) — within normal range.")
        print()


# ---------------------------------------------------------------------------
# Example usage (no real crew needed to see the drift monitoring API)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # For a real multi-agent crew, each agent would have its own Cathedral key.
    # Here we use one key for illustration — in production use separate keys
    # so drift scores are independent per agent.
    API_KEY = os.environ.get("CATHEDRAL_API_KEY", "your_api_key_here")

    if API_KEY == "your_api_key_here":
        print("Set CATHEDRAL_API_KEY environment variable.")
        print("Free key at cathedral-ai.com\n")

    # Simulate a 3-agent crew
    monitor = CrewDriftMonitor(api_keys={
        "researcher": API_KEY,
        "analyst":    API_KEY,   # use separate keys in production
        "writer":     API_KEY,
    })

    print("=== Pre-run: restoring agent identity state ===")
    contexts = monitor.wake_all()

    print("\n[Crew would run here — crew.kickoff(inputs={...})]")

    print("\n=== Post-run: snapshotting and scoring drift ===")
    reports = monitor.snapshot_and_score_all(run_id="example-001")

    monitor.print_summary(reports)

    print("Full drift timeline: cathedral-ai.com/cathedral-beta")
    print("Docs: cathedral-ai.com | pip install cathedral-memory")
