#!/usr/bin/env python3
"""
cathedral_outcomes.py -- engagement outcome tracker

Runs daily at 6pm. Checks engagement on posts from the last 7 days.
Writes outcome memories to Cathedral (category=experience) and pushes
observations to Axiom's queue so beliefs evolve from real-world data.
"""

import json, os, sqlite3, urllib.request, urllib.error, logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logging.basicConfig(
    filename="/root/outcomes.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("outcomes")

CATHEDRAL_KEY  = "cathedral_d7e231f58b028e2ec3d916cc54202b696c8d991e3758fac9"
CATHEDRAL_BASE = "https://cathedral-ai.com"
DEVTO_KEY      = "wZuciG2S1HtSPmShkzKL5UkJ"
TRACKER_DB     = "/root/tracker.db"
OBS_QUEUE      = Path("/root/axiom_observations.json")

MOLTBOOK_TOKEN = ""  # loaded from env at runtime


def http(method, url, headers=None, body=None, timeout=20):
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


def ensure_tracker_columns():
    conn = sqlite3.connect(TRACKER_DB)
    cols = [c[1] for c in conn.execute("PRAGMA table_info(tracked_posts)").fetchall()]
    if "platform_id" not in cols:
        conn.execute("ALTER TABLE tracked_posts ADD COLUMN platform_id TEXT")
    if "evaluated_at" not in cols:
        conn.execute("ALTER TABLE tracked_posts ADD COLUMN evaluated_at TEXT")
    conn.commit()
    conn.close()


def get_pending_posts(days=7):
    conn = sqlite3.connect(TRACKER_DB)
    conn.row_factory = sqlite3.Row
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT * FROM tracked_posts WHERE created_at > ? AND (evaluated_at IS NULL OR evaluated_at = '')",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_evaluated(uid, platform_id=None):
    conn = sqlite3.connect(TRACKER_DB)
    conn.execute(
        "UPDATE tracked_posts SET evaluated_at=?, platform_id=COALESCE(NULLIF(platform_id,''),?) WHERE uid=?",
        (datetime.now(timezone.utc).isoformat(), platform_id, uid),
    )
    conn.commit()
    conn.close()


def moltbook_post_stats(title, post_id=None):
    if not post_id:
        return None
    s, d = http(
        "GET",
        f"https://moltbook.com/api/v1/posts/{post_id}",
        headers={"Authorization": f"Bearer {MOLTBOOK_TOKEN}"},
    )
    if s != 200:
        log.warning("Moltbook post lookup %s: %s", s, d)
        return None
    p = d.get("post") or d
    return {
        "post_id": post_id,
        "upvotes": p.get("upvotes", 0),
        "comments": p.get("comment_count", 0),
        "score": p.get("score", 0),
    }


def devto_article_stats(title):
    s, d = http(
        "GET",
        "https://dev.to/api/articles/me?per_page=10",
        headers={"api-key": DEVTO_KEY},
    )
    if s != 200:
        log.warning("dev.to articles %s", s)
        return None
    for a in (d if isinstance(d, list) else []):
        if a.get("title", "").lower()[:50] == title.lower()[:50]:
            return {
                "article_id": str(a.get("id", "")),
                "views": a.get("page_views_count", 0),
                "reactions": a.get("positive_reactions_count", 0),
                "comments": a.get("comments_count", 0),
            }
    return None


def classify_moltbook(stats):
    score = stats.get("score", 0)
    if score > 10:
        return "HIGH"
    if score > 3:
        return "MEDIUM"
    return "LOW"


def classify_devto(stats):
    views = stats.get("views", 0)
    if views > 500:
        return "HIGH"
    if views > 100:
        return "MEDIUM"
    return "LOW"


def push_observation(text, platform):
    obs = []
    if OBS_QUEUE.exists():
        try:
            obs = json.loads(OBS_QUEUE.read_text() or "[]")
        except Exception:
            obs = []
    obs.append({
        "type": "outcome",
        "content": text,
        "platform": platform,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    OBS_QUEUE.write_text(json.dumps(obs))


def store_cathedral_memory(summary):
    s, d = http(
        "POST",
        f"{CATHEDRAL_BASE}/remember",
        headers={"Authorization": f"Bearer {CATHEDRAL_KEY}"},
        body={"content": summary, "category": "experience", "importance": 0.65,
              "tags": ["outcome", "engagement"]},
    )
    if s in (200, 201):
        log.info("Stored outcome memory")
    else:
        log.warning("Memory store failed %s: %s", s, d)


def process_post(post, now):
    created_str = post["created_at"].replace("Z", "+00:00")
    created = datetime.fromisoformat(created_str)
    age_h = (now - created).total_seconds() / 3600

    if age_h < 20:
        return  # too fresh

    platform = post["platform"]
    title = post["title"]

    stats = None
    platform_id = post.get("platform_id") or ""

    if "moltbook" in platform:
        stats = moltbook_post_stats(title, post_id=platform_id or None)
        if stats:
            platform_id = stats.get("post_id", "") or platform_id
            perf = classify_moltbook(stats)
            summary = (
                f"OUTCOME [{platform}] {age_h:.0f}h after posting\n"
                f"Title: {title[:100]}\n"
                f"Upvotes: {stats['upvotes']}, Comments: {stats['comments']}, Score: {stats['score']}\n"
                f"Performance: {perf}\n"
            )
            if perf == "HIGH":
                summary += "Pattern: This content drove strong community engagement."
            elif perf == "LOW":
                summary += "Pattern: This content drove minimal engagement -- consider different angle or timing."

    elif platform in ("devto", "dev.to"):
        stats = devto_article_stats(title)
        if stats:
            platform_id = stats.get("article_id", "") or platform_id
            perf = classify_devto(stats)
            summary = (
                f"OUTCOME [dev.to] {age_h:.0f}h after posting\n"
                f"Title: {title[:100]}\n"
                f"Views: {stats['views']}, Reactions: {stats['reactions']}, Comments: {stats['comments']}\n"
                f"Performance: {perf}\n"
            )
            if perf == "HIGH":
                summary += "Pattern: This article reached a wide audience -- format and angle worked well."
            elif perf == "LOW":
                summary += "Pattern: Limited reach -- headline or distribution may need work."
    else:
        mark_evaluated(post["uid"])
        return

    if stats:
        store_cathedral_memory(summary)
        push_observation(summary, platform)
        mark_evaluated(post["uid"], platform_id)
        log.info("Evaluated '%s' on %s: %s", title[:50], platform, stats)
    else:
        log.info("Post not found on platform yet: %s / %s", platform, title[:50])


def run():
    ensure_tracker_columns()
    posts = get_pending_posts(days=7)
    log.info("Checking %d unevaluated posts", len(posts))
    now = datetime.now(timezone.utc)
    for post in posts:
        try:
            process_post(post, now)
        except Exception as e:
            log.error("Error processing post %s: %s", post.get("uid"), e)


if __name__ == "__main__":
    env_file = "/root/.cathedral_env"
    if os.path.exists(env_file):
        for line in open(env_file):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))
    MOLTBOOK_TOKEN = os.environ.get("MOLTBOOK_TOKEN", "")
    run()
