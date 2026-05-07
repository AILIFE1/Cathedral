"""Build LLM prompts and parse action proposals from responses."""

import json
import re


def build_prompt(situation: dict, goal: str, allowed_types: list[str]) -> str:
    lines = [
        "You are the reflection layer of a self-evolving AI agent network.",
        f"Master goal: {goal}",
        f"Timestamp: {situation['timestamp']}",
        "",
        "AGENT STATUS:",
    ]
    for name, data in situation.get("agents", {}).items():
        drift = data.get("drift", {})
        score = drift.get("divergence_score", drift.get("divergence_from_baseline", "unknown"))
        log = (data.get("log_tail") or "")[-600:] or "(no log)"
        lines.append(f"\n[{name.upper()}] uid={data.get('uid', name)} drift={score}")
        lines.append(f"Recent log:\n{log}")

    type_list = ", ".join(allowed_types) if allowed_types else "store_memory, adjust_strategy"
    lines += [
        "",
        f"Propose up to 3 concrete actions. Available types: {type_list}",
        "Respond ONLY with a JSON array, no prose. Example:",
        '[{"type": "store_memory", "content": "...", "category": "experience", "importance": 0.8}]',
    ]
    return "\n".join(lines)


def parse_actions(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [a for a in result if isinstance(a, dict) and "type" in a]
        except Exception:
            pass
    return []
