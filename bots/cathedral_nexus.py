#!/usr/bin/env python3
"""
Cathedral Nexus v1 — self-evolving meta-agent orchestrator.

Reads all bot logs + Cathedral state, reasons via Groq, validates actions
through AgentGuard, executes approved changes, snapshots own state.

Cron: 0 */6 * * * python3 /root/cathedral_nexus.py >> /root/nexus.log 2>&1
"""

import json
# SECRETS: loaded from /root/.cathedral_env
import os as _os
_env_file = "/root/.cathedral_env"
if _os.path.exists(_env_file):
    for _line in open(_env_file):
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            _os.environ.setdefault(_k.strip(), _v.strip())
import sys
from datetime import datetime, timezone
from pathlib import Path

# ensure nexus package is importable regardless of where script lives
_here = Path(__file__).resolve().parent
_pkg  = _here if (_here / "nexus").is_dir() else _here / "cathedral-nexus"
sys.path.insert(0, str(_pkg))

from nexus.agents import CathedralClient
from nexus.reflection import build_situation, propose_actions
from nexus.trust import build_validator, validate_action, report_constraint_drift
from nexus.actions import execute

CONFIG_PATH = _pkg / "config.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def make_clients(config: dict) -> dict:
    """Build a CathedralClient for each agent using its own API key."""
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

    # 1. Register nexus is alive
    snap = nexus.snapshot("nexus-cycle-start")
    snap_id = snap.get("snapshot_id") or snap.get("detail") or snap.get("error") or "?"
    print(f"[nexus] Snapshot: {str(snap_id)[:20]}")

    # 2. Build situation report
    print("[nexus] Building situation report...")
    situation = build_situation(cfg, clients)

    # 3. Groq proposes actions
    print("[nexus] Requesting proposals from Groq...")
    proposals = propose_actions(cfg, situation)
    print(f"[nexus] {len(proposals)} proposal(s) received")

    if not proposals:
        print("[nexus] No proposals — cycle complete")
        _final_snapshot(nexus, 0)
        return

    # 4. AgentGuard validator
    validator, token, state = build_validator(cfg)

    # 5. Validate + execute
    executed = 0
    for i, action in enumerate(proposals):
        atype = action.get("type", "unknown")
        print(f"\n[nexus] Proposal {i+1}: {atype}")
        approved = validate_action(action, validator, token, state, 1.0, cfg["guard"]["trust_threshold"])
        if approved:
            ok = execute(action, nexus)
            if ok:
                executed += 1

    # 6. Constraint drift report
    report_constraint_drift(validator)

    # 7. Final snapshot
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
