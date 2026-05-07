"""Default situation builder — reads Cathedral state + log tails per agent."""

from datetime import datetime, timezone
from pathlib import Path


def tail_log(path: str | None, max_lines: int = 60) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return f"[log not found: {path}]"
    return "\n".join(p.read_text(errors="ignore").splitlines()[-max_lines:])


def build_situation(agents: dict, cathedral_url: str = "https://cathedral-ai.com") -> dict:
    """
    Build a situation report from Cathedral state + log tails.

    agents format:
        {
          "brain": {"api_key": "...", "uid": "my-brain", "log": "/var/log/brain.log"},
          ...
        }
    """
    from cathedral_nexus.client import CathedralClient

    situation: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agents": {},
    }
    for name, meta in agents.items():
        client = CathedralClient(cathedral_url, meta["api_key"])
        situation["agents"][name] = {
            "uid":      meta.get("uid", name),
            "drift":    client.drift(),
            "goals":    client.get_goals(),
            "log_tail": tail_log(meta.get("log")),
        }
    return situation
