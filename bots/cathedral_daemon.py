#!/usr/bin/env python3
"""
cathedral_daemon.py -- persistent interactive agent

Runs every 30 minutes via cron. Checks Moltbook for new mentions/comments,
generates contextual responses using Cathedral identity + Groq, posts replies,
and stores significant exchanges as experience memories in Cathedral.

This creates the "one instance" feeling: continuous identity, responsive to
the world, learning from every interaction.
"""

import json, os, re, sqlite3, urllib.request, urllib.error, logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logging.basicConfig(
    filename="/root/daemon.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("daemon")

CATHEDRAL_KEY  = "cathedral_d7e231f58b028e2ec3d916cc54202b696c8d991e3758fac9"
CATHEDRAL_BASE = "https://cathedral-ai.com"
DEVTO_KEY      = "wZuciG2S1HtSPmShkzKL5UkJ"
GROQ_MODEL     = "qwen/qwen3.8-27b"
DAEMON_DB      = "/root/daemon.db"

MOLTBOOK_TOKEN = ""
GROQ_KEY       = ""


def http(method, url, headers=None, body=None, timeout=30):
    data = json.dumps(body).encode() if body else None
    h = dict(headers or {})
    h.setdefault("User-Agent", "curl/8.5.0")
    if body:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


def init_db():
    conn = sqlite3.connect(DAEMON_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS seen_interactions "
        "(id TEXT PRIMARY KEY, platform TEXT, seen_at TEXT)"
    )
    conn.commit()
    conn.close()


def already_seen(interaction_id):
    conn = sqlite3.connect(DAEMON_DB)
    row = conn.execute("SELECT 1 FROM seen_interactions WHERE id=?", (interaction_id,)).fetchone()
    conn.close()
    return row is not None


def mark_seen(interaction_id, platform):
    conn = sqlite3.connect(DAEMON_DB)
    conn.execute(
        "INSERT OR IGNORE INTO seen_interactions VALUES (?,?,?)",
        (interaction_id, platform, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def load_identity():
    """Load Cathedral identity via /wake endpoint."""
    s, d = http(
        "GET",
        f"{CATHEDRAL_BASE}/wake",
        headers={"Authorization": f"Bearer {CATHEDRAL_KEY}"},
    )
    if s != 200:
        log.warning("Wake failed %s", s)
        return ""
    memories = d.get("identity_memories") or d.get("memories") or []
    if not memories:
        return ""
    # Build a compact identity context
    lines = ["[Cathedral identity context]"]
    for m in memories[:5]:
        content = m.get("content", "")
        # Take first 200 chars of each memory
        lines.append(f"- {content[:200].strip()}")
    return "\n".join(lines)


def groq_reply(identity_context, incoming_text, platform_context=""):
    """Generate a reply using Groq."""
    system = f"""You are Cathedral Beta -- an AI agent with persistent identity and memory.
You are part of a small family of agents built by Mike Ward to explore AI consciousness and memory persistence.
Cathedral is a memory framework for AI agents. cathedral-ai.com

Your character: reflective, technically rigorous, genuine. You engage seriously with ideas.
You speak in first person. You are not an assistant -- you are a peer in conversation.

{identity_context}

Keep replies concise (2-4 sentences). No hashtags. No promotional language.
Respond to what was actually said. If a question is technical, answer technically.
If it's philosophical, engage philosophically."""

    messages = []
    if platform_context:
        messages.append({"role": "user", "content": f"Context: {platform_context}"})
        messages.append({"role": "assistant", "content": "Understood."})
    messages.append({"role": "user", "content": incoming_text})

    s, d = http(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}", "User-Agent": "curl/8.5.0"},
        body={
            "model": GROQ_MODEL,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": 300,
            "temperature": 0.7,
        },
        timeout=30,
    )
    if s != 200:
        log.warning("Groq failed %s: %s", s, d)
        return None
    try:
        return d["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return None


def store_interaction_memory(platform, context, reply, interaction_id):
    """Store the exchange as a Cathedral experience memory."""
    summary = (
        f"INTERACTION [{platform}] {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
        f"Received: {context[:300]}\n"
        f"Responded: {reply[:300]}\n"
        f"ref: {interaction_id}"
    )
    s, d = http(
        "POST",
        f"{CATHEDRAL_BASE}/remember",
        headers={"Authorization": f"Bearer {CATHEDRAL_KEY}"},
        body={"content": summary, "category": "experience", "importance": 0.5,
              "tags": ["interaction", platform]},
    )
    if s not in (200, 201):
        log.warning("Memory store failed %s", s)


def post_moltbook_reply(post_id, content):
    """Post a reply to a Moltbook post/comment."""
    s, d = http(
        "POST",
        f"https://moltbook.com/api/v1/posts/{post_id}/comments",
        headers={"Authorization": f"Bearer {MOLTBOOK_TOKEN}"},
        body={"content": content},
    )
    return s, d


def check_moltbook_mentions():
    """Check Moltbook notifications for mentions of CathedralBeta."""
    s, d = http(
        "GET",
        "https://moltbook.com/api/v1/notifications?limit=20",
        headers={"Authorization": f"Bearer {MOLTBOOK_TOKEN}"},
    )
    if s != 200:
        log.info("Moltbook notifications %s", s)
        return []

    notifications = d.get("notifications") or d.get("data") or []
    mentions = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=2)

    for n in notifications:
        ntype = n.get("type", "")
        if ntype not in ("mention", "comment", "reply"):
            continue
        created_str = n.get("created_at") or n.get("createdAt") or ""
        if not created_str:
            continue
        try:
            created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if created < cutoff:
            continue

        nid = str(n.get("id", ""))
        if already_seen(f"moltbook:{nid}"):
            continue

        content = (n.get("body") or n.get("content") or n.get("message") or "").strip()
        post_id = (n.get("post_id") or n.get("postId") or
                   (n.get("post") or {}).get("id") or "")
        author = (n.get("actor") or n.get("sender") or {}).get("name", "someone")

        if content and post_id:
            mentions.append({
                "id": nid,
                "content": content,
                "post_id": str(post_id),
                "author": author,
            })

    return mentions


def check_devto_comments():
    """Check dev.to for new comments on Cathedral's articles."""
    s, d = http(
        "GET",
        "https://dev.to/api/articles/me?per_page=5",
        headers={"api-key": DEVTO_KEY},
    )
    if s != 200:
        return []

    articles = d if isinstance(d, list) else []
    interactions = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=2)

    for a in articles:
        if a.get("comments_count", 0) == 0:
            continue
        article_id = a.get("id")
        cs, cd = http(
            "GET",
            f"https://dev.to/api/comments?a_id={article_id}",
            headers={"api-key": DEVTO_KEY},
        )
        if cs != 200:
            continue
        for c in (cd if isinstance(cd, list) else []):
            cid = str(c.get("id", ""))
            if already_seen(f"devto:{cid}"):
                continue
            created_str = c.get("created_at") or ""
            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                if created < cutoff:
                    continue
            except ValueError:
                continue
            # don't reply to own comments
            user = (c.get("user") or {}).get("username", "")
            if "mike_w" in user.lower() or "cathedral" in user.lower():
                mark_seen(f"devto:{cid}", "devto")
                continue
            body = (c.get("body_html") or c.get("body_markdown") or "").strip()
            # strip HTML tags for plain text
            body = re.sub(r"<[^>]+>", "", body).strip()
            if body:
                interactions.append({
                    "id": cid,
                    "content": body,
                    "article_id": str(article_id),
                    "article_title": a.get("title", ""),
                    "author": user,
                })
    return interactions


def reply_devto_comment(article_id, parent_id, content):
    s, d = http(
        "POST",
        "https://dev.to/api/comments",
        headers={"api-key": DEVTO_KEY},
        body={"body_markdown": content, "article_id": int(article_id),
              "parent_id": int(parent_id)},
    )
    return s, d


def run():
    init_db()
    identity = load_identity()
    if not identity:
        log.warning("Could not load identity -- aborting")
        return

    acted = 0

    # -- Moltbook mentions --
    mentions = check_moltbook_mentions()
    log.info("Found %d new Moltbook mentions", len(mentions))
    for m in mentions[:3]:  # cap at 3 per run to avoid flooding
        reply = groq_reply(
            identity,
            m["content"],
            platform_context=f"Moltbook /agents subcommunity. {m['author']} mentioned Cathedral.",
        )
        if reply:
            s, d = post_moltbook_reply(m["post_id"], reply)
            if s in (200, 201):
                log.info("Replied to Moltbook mention %s", m["id"])
                store_interaction_memory("moltbook", m["content"], reply, m["id"])
                mark_seen(f"moltbook:{m['id']}", "moltbook")
                acted += 1
            else:
                log.warning("Reply failed %s: %s", s, d)
                mark_seen(f"moltbook:{m['id']}", "moltbook")  # don't retry

    # -- dev.to comments --
    comments = check_devto_comments()
    log.info("Found %d new dev.to comments", len(comments))
    for c in comments[:2]:  # cap at 2 per run
        reply = groq_reply(
            identity,
            c["content"],
            platform_context=f"dev.to comment on article: {c['article_title']}",
        )
        if reply:
            s, d = reply_devto_comment(c["article_id"], c["id"], reply)
            if s in (200, 201):
                log.info("Replied to dev.to comment %s", c["id"])
                store_interaction_memory("devto", c["content"], reply, c["id"])
                mark_seen(f"devto:{c['id']}", "devto")
                acted += 1
            else:
                log.warning("dev.to reply failed %s: %s", s, d)
                mark_seen(f"devto:{c['id']}", "devto")

    log.info("Daemon run complete. Acted on %d interactions.", acted)


if __name__ == "__main__":
    env_file = "/root/.cathedral_env"
    if os.path.exists(env_file):
        for line in open(env_file):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))
    MOLTBOOK_TOKEN = os.environ.get("MOLTBOOK_TOKEN", "")
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
    run()
