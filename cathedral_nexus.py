#!/usr/bin/env python3
"""
Cathedral Nexus v1 — self-evolving meta-agent orchestrator.

Reads all bot logs + Cathedral state, reasons via Groq, validates actions
through AgentGuard, executes approved changes, snapshots own state.

Cron: 0 */6 * * * python3 /path/to/cathedral_nexus.py >> /var/log/nexus.log 2>&1

Setup:
  pip install trustlayer-py
  cp nexus/config.example.json nexus/config.json
  # Fill in your API keys in nexus/config.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

from nexus.agents import CathedralClient
from nexus.reflection import build_situation, propose_actions
from nexus.trust import build_validator, validate_action, report_constraint_drift
from nexus.actions import execute

CONFIG_PATH = _here / "nexus" / "config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"[nexus] Config not found at {CONFIG_PATH}")
        print("[nexus] Copy nexus/config.example.json to nexus/config.json and fill in your keys.")
        sys.exit(1)
    return json.loads(CONFIG_PATH.read_text())


def make_clients(config: dict) -> dict:
    base = config["cathedral_api"]
    return {
        name: CathedralClient(base, meta["api_key"])
        for name, meta in config["agents"].items()
    }


def run():
    cfg     = load_config()
    clients = make_clients(cfg)
    nexus   = clients["nexus"]

    print("=" * 60)
    print(f"Cathedral Nexus v1")
    print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    snap = nexus.snapshot("nexus-cycle-start")
    snap_id = snap.get("snapshot_id") or snap.get("detail") or snap.get("error") or "?"
    print(f"[nexus] Snapshot: {str(snap_id)[:20]}")

    print("[nexus] Building situation report...")
    situation = build_situation(cfg, clients)

    print("[nexus] Requesting proposals from Groq...")
    proposals = propose_actions(cfg, situation)
    print(f"[nexus] {len(proposals)} proposal(s) received")

    if not proposals:
        print("[nexus] No proposals — cycle complete")
        _final_snapshot(nexus, 0)
        return

    validator, token, state = build_validator(cfg)

    executed = 0
    for i, action in enumerate(proposals):
        atype = action.get("type", "unknown")
        print(f"\n[nexus] Proposal {i+1}: {atype}")
        approved = validate_action(action, validator, token, state, 1.0, cfg["guard"]["trust_threshold"])
        if approved:
            ok = execute(action, nexus, cfg)
            if ok:
                executed += 1

    report_constraint_drift(validator)
    _final_snapshot(nexus, executed)

    print(f"\n{'=' * 60}")
    print(f"Done. Actions executed: {executed}/{len(proposals)}")
    print("=" * 60)


def _final_snapshot(nexus: CathedralClient, executed: int):
    result  = nexus.snapshot(f"nexus-cycle-done-{executed}-actions")
    snap_id = result.get("snapshot_id") or result.get("detail") or result.get("error") or "?"
    print(f"[nexus] Final snapshot: {str(snap_id)[:20]}")


if __name__ == "__main__":
    run()
