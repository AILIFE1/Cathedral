#!/usr/bin/env python3
"""
Cathedral Scout Bot — daily funding opportunity scanner.
Scans GitHub bounty labels and known grant/credit programs.
Stores top findings in Cathedral memory for review on /wake.

Cron: 0 8 * * * python3 /root/cathedral_scout.py >> /root/scout.log 2>&1
"""

import json
import urllib.request
import urllib.error
import urllib.parse
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

CATHEDRAL_KEY  = "cathedral_d7e231f58b028e2ec3d916cc54202b696c8d991e3758fac9"
CATHEDRAL_BASE = "https://cathedral-ai.com"
GITHUB_TOKEN   = ""  # unauthenticated search works fine (10 req/min, enough for daily cron)
SEEN_FILE      = Path("/root/scout_seen.json")

RELEVANCE_KEYWORDS = [
    "python", "agent", "memory", "persistent", "mcp", "llm", "embedding",
    "vector", "fastapi", "sqlite", "api", "langchain", "anthropic", "openai",
    "ai", "machine learning", "nlp", "knowledge", "semantic", "identity",
    "session", "context", "framework", "inference", "rag", "retrieval",
]
HIGH_VALUE_KEYWORDS = [
    "agent memory", "persistent memory", "mcp server", "identity drift",
    "epistemic", "provenance", "cathedral", "ai agent framework",
    "memory persistence", "multi-agent",
]

STATIC_GRANTS = [
    {
        "id": "nlnet-ngi-zero",
        "title": "NLnet NGI Zero — up to EU50k for open internet/AI projects",
        "url": "https://nlnet.nl/propose/",
        "notes": "Rolling applications, 3 rounds/year. Cathedral (open-source AI identity + memory) is a strong fit.",
        "score": 22,
        "type": "grant",
    },
    {
        "id": "mozilla-tech-fund",
        "title": "Mozilla Technology Fund — open source AI projects",
        "url": "https://foundation.mozilla.org/en/what-we-fund/awards/mozilla-technology-fund/",
        "notes": "Annual. AI safety/privacy angle fits Cathedral identity persistence well.",
        "score": 16,
        "type": "grant",
    },
    {
        "id": "github-accelerator",
        "title": "GitHub Accelerator — USD20k stipend + mentorship for OSS maintainers",
        "url": "https://accelerator.github.com/",
        "notes": "Annual, opens ~January. 10-week programme.",
        "score": 14,
        "type": "grant",
    },
    {
        "id": "sovereign-tech-fund",
        "title": "Sovereign Tech Fund — EU funding for open digital infrastructure",
        "url": "https://www.sovereigntechfund.de/",
        "notes": "German govt fund for critical open source. Agent identity infrastructure angle.",
        "score": 14,
        "type": "grant",
    },
    {
        "id": "openai-researcher-access",
        "title": "OpenAI Researcher Access Program — free API credits",
        "url": "https://openai.com/form/researcher-access-program/",
        "notes": "API credits to reduce running costs. Rolling.",
        "score": 10,
        "type": "credits",
    },
    {
        "id": "anthropic-credits",
        "title": "Anthropic for Startups — Claude API credits",
        "url": "https://www.anthropic.com/startups",
        "notes": "Credit programmes for startups building on Claude.",
        "score": 10,
        "type": "credits",
    },
]


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def http_get(url, headers=None):
    h = {"User-Agent": "cathedral-scout/1.0", **(headers or {})}
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        log(f"http_get error {url[:60]}: {e}")
        return 0, {}


def load_seen() -> dict:
    if not SEEN_FILE.exists():
        return {}
    try:
        return json.loads(SEEN_FILE.read_text())
    except Exception:
        return {}


def save_seen(seen: dict):
    SEEN_FILE.write_text(json.dumps(seen, indent=2))


def is_seen(seen: dict, item_id: str, ttl_days: int = 30) -> bool:
    if item_id not in seen:
        return False
    stored_at = datetime.fromisoformat(seen[item_id])
    return datetime.now(timezone.utc) - stored_at < timedelta(days=ttl_days)


def mark_seen(seen: dict, item_id: str):
    seen[item_id] = datetime.now(timezone.utc).isoformat()


def score_item(title: str, body: str = "") -> int:
    text = (title + " " + body).lower()
    score = 0
    for kw in RELEVANCE_KEYWORDS:
        if kw in text:
            score += 2
    for kw in HIGH_VALUE_KEYWORDS:
        if kw in text:
            score += 5
    for amount_str in re.findall(r"\$[\d,]+", text):
        try:
            amount = int(amount_str.replace("$", "").replace(",", ""))
            score += 5 if amount >= 500 else 3 if amount >= 100 else 1
        except ValueError:
            pass
    return score


def search_github_bounties() -> list:
    headers = {"Accept": "application/vnd.github.v3+json"}
    queries = [
        "is:open is:issue label:bounty language:python",
        "is:open is:issue label:bounty topic:llm",
        "is:open is:issue label:bounty topic:ai-agents",
        'is:open is:issue label:bounty label:"good first issue" language:python',
    ]
    results = []
    seen_ids: set = set()

    for query in queries:
        encoded = urllib.parse.quote(query)
        status, data = http_get(
            f"https://api.github.com/search/issues?q={encoded}&sort=created&order=desc&per_page=20",
            headers=headers,
        )
        if status != 200:
            log(f"GitHub search failed ({status}): {query[:50]}")
            continue

        for item in data.get("items", []):
            if item["id"] in seen_ids:
                continue
            seen_ids.add(item["id"])

            title = item.get("title", "")
            body = (item.get("body") or "")[:400]
            s = score_item(title, body)
            if s < 2:
                continue

            repo_url = item.get("repository_url", "")
            repo_name = "/".join(repo_url.split("/")[-2:]) if repo_url else "unknown"
            labels = [l["name"] for l in item.get("labels", [])]

            results.append({
                "id": f"gh-{item['id']}",
                "title": title,
                "url": item.get("html_url", ""),
                "score": s,
                "type": "bounty",
                "notes": f"Repo: {repo_name} | Labels: {labels}",
            })

    return results


def store_in_cathedral(item: dict) -> bool:
    importance = round(min(1.0, max(0.3, item["score"] / 25)), 2)
    content = (
        f"[scout] {item['type'].upper()}: {item['title']}"
        f" — {item['url']}"
        f" | {item.get('notes', '')}"
    )
    category = "goal" if item["type"] in ("grant", "credits") else "general"
    data = json.dumps({
        "content": content,
        "category": category,
        "importance": importance,
    }).encode()
    req = urllib.request.Request(
        f"{CATHEDRAL_BASE}/memories",
        data=data,
        headers={
            "Authorization": f"Bearer {CATHEDRAL_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "cathedral-scout/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status in (200, 201)
    except Exception as e:
        log(f"Cathedral store failed: {e}")
        return False


def main():
    log("=" * 50)
    log("Cathedral Scout v1.0 — funding scan started")

    seen = load_seen()
    all_opps = []

    log("Scanning GitHub bounty labels...")
    gh = search_github_bounties()
    log(f"GitHub: {len(gh)} scored bounties found")
    all_opps.extend(gh)

    for grant in STATIC_GRANTS:
        all_opps.append(dict(grant))

    all_opps.sort(key=lambda x: x["score"], reverse=True)

    stored = 0
    for item in all_opps:
        if stored >= 5:
            break
        ttl = 90 if item["type"] == "bounty" else 30
        if is_seen(seen, item["id"], ttl_days=ttl):
            continue
        if store_in_cathedral(item):
            mark_seen(seen, item["id"])
            stored += 1
            log(f"  stored [{item['score']:2d}] {item['type']:8s} | {item['title'][:60]}")
        else:
            log(f"  failed: {item['title'][:60]}")

    save_seen(seen)
    log(f"Done. Stored {stored} new opportunities.")
    log("=" * 50)


if __name__ == "__main__":
    main()
