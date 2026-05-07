"""Execute approved actions against the bot ecosystem."""

import json
import urllib.request
import urllib.error

from nexus.agents import CathedralClient


def _moltbook_post(title: str, content: str, submolt: str, token: str, tracker_base: str) -> bool:
    if "{url}" not in content:
        content += f"\n\n{tracker_base}"
    body = {"title": title, "content": content, "submolt_name": submolt}
    req = urllib.request.Request(
        "https://www.moltbook.com/api/v1/posts",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "cathedral-nexus/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"[action] Posted to r/{submolt}: {title[:60]}")
            return True
    except urllib.error.HTTPError as e:
        print(f"[action] Moltbook {e.code}: {e.read().decode()[:150]}")
        return False
    except Exception as e:
        print(f"[action] Moltbook error: {e}")
        return False


def execute(action: dict, nexus: CathedralClient, config: dict) -> bool:
    atype = action.get("type")

    if atype == "queue_post":
        title   = action.get("title", "")
        content = action.get("content", "")
        submolt = action.get("submolt", "agents")
        if not title or not content:
            print("[action] queue_post missing title/content")
            return False
        return _moltbook_post(
            title, content, submolt,
            config["moltbook_token"],
            config.get("tracker_base", "https://cathedral-ai.com"),
        )

    VALID_CATEGORIES = {"experience", "general", "goal", "identity", "relationship", "skill"}

    if atype == "store_memory":
        content    = action.get("content", "")
        raw_cat    = action.get("category", "general")
        category   = raw_cat if raw_cat in VALID_CATEGORIES else "general"
        importance = float(action.get("importance", 0.7))
        if not content:
            return False
        result = nexus.store_memory(content, category, importance)
        ok = "error" not in result
        print(f"[action] store_memory: {'ok' if ok else result.get('detail', result.get('error'))}")
        return ok

    elif atype == "update_goal":
        goal_id = action.get("goal_id", "")
        patch   = action.get("patch", {})
        if not goal_id or not patch:
            print("[action] update_goal missing goal_id or patch")
            return False
        result = nexus.update_goal(goal_id, patch)
        ok = "error" not in result
        print(f"[action] update_goal: {'ok' if ok else result.get('detail', result.get('error'))}")
        return ok

    elif atype == "adjust_strategy":
        target = action.get("target", "unknown")
        change = action.get("change", "")
        note   = f"[nexus strategy] {target}: {change}"
        result = nexus.store_memory(note, "experience", 0.8)
        ok = "error" not in result
        print(f"[action] adjust_strategy → memory: {'ok' if ok else result.get('detail', result.get('error'))}")
        return ok

    else:
        print(f"[action] Unknown type: {atype}")
        return False
