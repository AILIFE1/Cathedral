"""Read bot logs + Cathedral state, build situation report, propose actions via Groq."""

import json
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path


def tail_log(path: str | None, max_lines: int = 60) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return f"[log not found: {path}]"
    lines = p.read_text(errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def build_situation(config: dict, clients: dict) -> dict:
    situation = {"timestamp": datetime.now(timezone.utc).isoformat(), "agents": {}}
    for name, meta in config["agents"].items():
        client = clients[name]
        log_snippet = tail_log(meta.get("log"))
        drift = client.drift()
        goals = client.get_goals()
        snap  = client.snapshot(f"{name}-situation-check")
        situation["agents"][name] = {
            "uid":      meta["uid"],
            "drift":    drift,
            "goals":    goals,
            "snapshot": snap.get("snapshot_id", "?"),
            "log_tail": log_snippet,
        }
    return situation


def summarise_prompt(situation: dict, goal: str) -> str:
    lines = [
        "You are the reflection layer of a self-evolving AI agent network.",
        f"Master goal: {goal}",
        f"Timestamp: {situation['timestamp']}",
        "",
        "AGENT STATUS:",
    ]
    for name, data in situation["agents"].items():
        drift_score = data["drift"].get("divergence_from_baseline", "unknown")
        log = data["log_tail"][-600:] if data["log_tail"] else "(no log)"
        lines.append(f"\n[{name.upper()}] uid={data['uid']} drift={drift_score}")
        lines.append(f"Recent log:\n{log}")

    lines += [
        "",
        "Based on the above, propose up to 3 concrete actions to improve performance toward the master goal.",
        "Each action must be one of: queue_post, store_memory, update_goal, adjust_strategy",
        "",
        "Respond ONLY with a JSON array, no prose. Format:",
        '[{"type": "queue_post", "submolt": "agents", "title": "...", "content": "... {url}"},',
        ' {"type": "store_memory", "content": "...", "category": "experience", "importance": 0.8},',
        ' valid categories: experience, general, goal, identity, relationship, skill',
        ' {"type": "adjust_strategy", "target": "outreach", "change": "..."}]',
    ]
    return "\n".join(lines)


def _parse(text: str) -> list:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return []


def _groq(prompt: str, key: str, model: str = "llama-3.3-70b-versatile") -> list:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 1000,
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.5.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
            return _parse(text)
    except Exception as e:
        print(f"[reflection] Groq error: {e}")
        return []


def propose_actions(config: dict, situation: dict) -> list:
    prompt = summarise_prompt(situation, config["goal"])
    actions = _groq(prompt, config["groq_key"])
    if actions:
        print(f"[reflection] Groq → {len(actions)} proposal(s)")
    return actions
