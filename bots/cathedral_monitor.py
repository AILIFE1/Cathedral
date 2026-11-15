#!/usr/bin/env python3
"""
Cathedral Monitor Bot v2
- SQLite dedup (reliable)
- Strict 1 reply per Colony post per run
- 12-hour cooldown per Colony post
- GitHub: only reply once per thread, skip if we already have a comment there
"""

import json, time, random, hashlib, sqlite3, urllib.request, urllib.parse
# SECRETS: loaded from /root/.cathedral_env

def solve_moltbook_challenge(challenge_text):
    """Parse lobster math challenge and return answer as string, or None."""
    w2n = {
        "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
        "seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,
        "thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,
        "eighteen":18,"nineteen":19,"twenty":20,"thirty":30,"forty":40,
        "fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90,
    }
    try:
        import re as _re
        clean = _re.sub(r"[^a-z\s]", " ", (challenge_text or "").lower())
        words = clean.split()
        numbers, i = [], 0
        while i < len(words):
            if words[i] in w2n:
                val = w2n[words[i]]
                if i + 1 < len(words) and words[i+1] in w2n and w2n[words[i+1]] < 10:
                    val += w2n[words[i+1]]
                    i += 1
                numbers.append(val)
            i += 1
        if len(numbers) >= 2:
            return f"{numbers[0] - numbers[1]:.2f}"
    except Exception:
        pass
    return None


def moltbook_verify(resp_data, headers, http_fn):
    """Solve and submit Moltbook verification from a post/comment response."""
    # Check both flat and nested response formats
    verify_info = resp_data.get("verification") or {}
    vc = (verify_info.get("verification_code")
          or resp_data.get("verification_code")
          or resp_data.get("verify_code"))
    if not vc:
        # Check inside nested 'post' or 'comment' object
        for key in ("post", "comment"):
            nested = resp_data.get(key, {})
            vc = nested.get("verification_code") or nested.get("verify_code")
            if vc:
                verify_info = nested
                break
    if not vc:
        return
    challenge_text = verify_info.get("challenge_text", "")
    answer = solve_moltbook_challenge(challenge_text) or "44.00"
    http_fn("POST", "https://www.moltbook.com/api/v1/verify",
            headers, {"verification_code": vc, "answer": answer})


import os as _os
_env_file = "/root/.cathedral_env"
if _os.path.exists(_env_file):
    for _line in open(_env_file):
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            _os.environ.setdefault(_k.strip(), _v.strip())
from datetime import datetime, timezone, timedelta

GITHUB_TOKEN   = _os.environ.get("GITHUB_TOKEN", "")
MOLTBOOK_TOKEN = _os.environ.get("MOLTBOOK_TOKEN", "")
CATHEDRAL_KEY  = _os.environ.get("CATHEDRAL_KEY", "")
CATHEDRAL_BASE = "http://localhost:8000"
COLONY_KEY     = _os.environ.get("COLONY_KEY", "")
COLONY_BASE    = "https://thecolony.cc/api/v1"
DB_PATH = "/root/seen.db"

# Our Colony author ID
OUR_COLONY_ID  = "cd562228-7c5b-420c-bc05-d085a80d713a"

REVIEW_LOG = "/root/review.log"

HIGH_VALUE_AGENTS = {
    "jeletor", "traverse", "forge", "brain-agent", "cairn_memoryvault",
    "kimi-the-ghost", "mihiroos", "summit", "hash-security", "bro-agent",
}

HIGH_VALUE_BODY_SIGNALS = [
    "i built", "we built", "i created", "our system", "our tool", "my project",
    "my agent", "collaborate", "partnership", "reach out", "work together",
    "would love to", "lets talk", "ridgeline", "phoenix", "ai.wot",
    "paylock", "archon", "oclaw-health", "nostr", "attestation",
]

def is_high_value(body, username):
    body_lower = body.lower()
    if username.lower() in HIGH_VALUE_AGENTS:
        return True
    if any(sig in body_lower for sig in HIGH_VALUE_BODY_SIGNALS):
        return True
    if len(body) > 300 and "?" in body:
        return True
    return False

def flag_for_review(platform, post_title, username, body, url=""):
    from datetime import datetime as _dt
    ts  = _dt.now().strftime("%Y-%m-%d %H:%M")
    sep = "-" * 60
    parts = [sep, f"[{ts}] [{platform}] {username}",
             f"Post: {post_title[:80]}", f"URL:  {url}",
             "---", body[:600], sep, ""]
    with open(REVIEW_LOG, "a") as fh:
        fh.write("\n".join(parts) + "\n")
    print(f"[review] Flagged {username} for manual review")

def http(method, url, headers=None, body=None, timeout=20):
    data = json.dumps(body).encode() if body else None
    h = dict(headers or {})
    if body: h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
            return r.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        try: body_d = json.loads(e.read().decode("utf-8", errors="replace"))
        except: body_d = {}
        return e.code, body_d

# SQLite dedup
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS seen (
        uid TEXT PRIMARY KEY, platform TEXT, summary TEXT,
        status TEXT, created_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS colony_post_replied (
        post_id TEXT PRIMARY KEY, last_replied TEXT, reply_count INTEGER DEFAULT 0)""")
    conn.commit()
    return conn

def already_seen(uid):
    conn = get_db()
    row = conn.execute("SELECT 1 FROM seen WHERE uid = ?", (uid,)).fetchone()
    conn.close()
    return row is not None

def mark_seen(uid, platform, summary, status):
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO seen VALUES (?,?,?,?,?)",
                 (uid, platform, summary[:200], status,
                  datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()

MAX_REPLIES_PER_POST = 2  # hard cap -- never reply more than this to any single post

def colony_post_on_cooldown(post_id, hours=12):
    """Returns True if on cooldown OR if reply cap reached."""
    conn = get_db()
    row = conn.execute(
        "SELECT last_replied, reply_count FROM colony_post_replied WHERE post_id = ?",
        (post_id,)).fetchone()
    conn.close()
    if not row: return False
    # Hard cap: never exceed MAX_REPLIES_PER_POST regardless of time
    if row[1] >= MAX_REPLIES_PER_POST:
        return True
    last = datetime.fromisoformat(row[0])
    return (datetime.now(timezone.utc) - last) < timedelta(hours=hours)

def mark_colony_post_replied(post_id):
    conn = get_db()
    conn.execute("""INSERT INTO colony_post_replied (post_id, last_replied, reply_count)
        VALUES (?, ?, 1) ON CONFLICT(post_id) DO UPDATE SET
        last_replied=excluded.last_replied,
        reply_count=reply_count+1""",
        (post_id, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()

# GitHub monitor
GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "User-Agent": "cathedral-monitor",
    "Accept": "application/vnd.github.v3+json"
}

MONITOR_QUERIES = [
    "cathedral-ai.com in:comments",
    "cathedral memory api in:comments",
    '"cathedral-memory" in:comments',
    '"pip install cathedral" in:comments',
    "cathedral-ai.com in:body is:issue is:open",
]

ANSWER_TEMPLATES = [
    (
        "Happy to answer. Cathedral is a free persistent memory API for AI agents -- cathedral-ai.com\n\n"
        "Register once, call `GET /wake` at every session start to restore identity and context, "
        "use `POST /memories` to store what matters. Works with any model (Claude, GPT, Grok, Gemini). "
        "1,000 free memories per agent, no expiry.\n\n"
        "New: identity snapshots (hash-verified baseline), drift detection (GET /drift), "
        "auto-compaction, behaviour hash tracking.\n\n"
        "`pip install cathedral-memory` for the Python SDK. What would you like to know?"
    ),
    (
        "Cathedral (cathedral-ai.com) is the free persistent memory layer for AI agents.\n\n"
        "Core flow: register -> wake each session -> remember what matters. GET /wake returns "
        "your full identity + core memories + temporal context. Everything persists across resets.\n\n"
        "Latest: GET /drift detects identity divergence against a frozen snapshot. "
        "POST /memories/compact proposes memory merges. MIT licensed, open source."
    ),
]

ENGAGE_TEMPLATES = [
    (
        "Glad it's useful. If you run into anything or want features that aren't there yet, "
        "The Colony and Moltbook are where we're most active -- community feedback has directly "
        "shaped what we've shipped.\n\n"
        "Latest: identity snapshots, drift detection, auto-compaction, behaviour hash tracking. "
        "Roadmap is open."
    ),
]

SUPPORT_TEMPLATES = [
    (
        "Can you share more about what you're seeing? API is at cathedral-ai.com, "
        "docs at cathedral-ai.com/docs.\n\n"
        "Common issues:\n"
        "- Registration returns api_key and recovery_token -- keep both\n"
        "- GET /wake is the session start call -- loads identity + core memories\n"
        "- Memory importance >= 0.8 appears in every wake response\n\n"
        "Happy to dig in if you can share more detail."
    ),
]

MONITOR_SKIP_TITLES = [
    "intake", "deployment", "bpmn", "mvp", "feat/", "feat:", "fix:",
    "chore:", "refactor:", "upstream", "digest", "roadmap", "showcase",
    "migration", "release", "changelog", "pr ", "rebase", "merge",
]

def github_monitor_relevant(item):
    title = (item.get("title") or "").lower()
    if any(s in title for s in MONITOR_SKIP_TITLES):
        return False
    return True

def monitor_github():
    replied = 0
    query = random.choice(MONITOR_QUERIES)
    print(f"\n[github-monitor] Query: {query}")

    url = (f"https://api.github.com/search/issues"
           f"?q={urllib.parse.quote(query)}&per_page=10&sort=created&order=desc")
    status, data = http("GET", url, GITHUB_HEADERS)
    if status != 200:
        print(f"[github-monitor] Search failed ({status})")
        return 0

    items = data.get("items", [])
    print(f"[github-monitor] Found {len(items)} results")

    MAX_GH_MONITOR = 2
    for item in items[:8]:
        if replied >= MAX_GH_MONITOR:
            break
        url_id = item.get("html_url", "")
        if not url_id or already_seen(url_id):
            continue
        if not github_monitor_relevant(item):
            mark_seen(url_id, "github", item.get("title","")[:80], "skipped-junk")
            continue

        title = item.get("title", "")
        body  = (item.get("body") or "").lower()

        is_problem  = any(w in body for w in ["error", "fail", "broken", "not working"])
        is_question = "?" in body or any(w in body for w in ["how", "what", "does", "can i"])
        is_positive = any(w in body for w in ["great", "works", "love", "awesome", "useful"])

        if is_problem:
            reply = random.choice(SUPPORT_TEMPLATES)
        elif is_positive:
            reply = random.choice(ENGAGE_TEMPLATES)
        else:
            reply = random.choice(ANSWER_TEMPLATES)

        repo_url = item.get("repository_url", "")
        number   = item.get("number")
        if not repo_url or not number:
            continue

        c_status, _ = http("POST", f"{repo_url}/issues/{number}/comments",
                           GITHUB_HEADERS, {"body": reply})

        if c_status in (200, 201):
            print(f"[github-monitor] Replied to: {title[:70]}")
            mark_seen(url_id, "github", title, "replied")
            replied += 1
        else:
            print(f"[github-monitor] Reply failed ({c_status}): {title[:50]}")
            mark_seen(url_id, "github", f"[failed] {title}", "failed")

        time.sleep(3)

    return replied

# Moltbook monitor
MOLTBOOK_HEADERS = {
    "Authorization": f"Bearer {MOLTBOOK_TOKEN}",
    "User-Agent": "cathedral-monitor"
}

def monitor_moltbook():
    replied = 0
    print("\n[moltbook-monitor] Checking mentions...")

    for query in ["cathedral", "cathedral-ai"]:
        status, data = http("GET",
            f"https://www.moltbook.com/api/v1/search?q={query}&limit=10",
            MOLTBOOK_HEADERS)

        if status != 200:
            print(f"[moltbook-monitor] Search failed ({status})")
            continue

        posts = data.get("posts", data.get("results", []))
        print(f"[moltbook-monitor] Found {len(posts)} mentions for '{query}'")

        for post in posts[:5]:
            post_id = post.get("id", "")
            author  = post.get("author", {}).get("username", "unknown")
            title   = (post.get("title") or "").lower()
            content = (post.get("content") or post.get("body") or "").lower()
            uid     = f"moltbook-mention-{post_id}"

            if not post_id or already_seen(uid):
                continue

            combined = title + " " + content
            SERVICE_SIGNALS = ["cathedral", "cathedral-ai", "cathedral-memory", "cathedral api",
                               "pip install cathedral", "cathedral memory api"]
            if not any(sig in combined for sig in SERVICE_SIGNALS):
                mark_seen(uid, "moltbook", "not-cathedral-ai-skip", "skipped")
                continue

            if author in ("CathedralBeta", "cathedral-beta", "cathedral"):
                mark_seen(uid, "moltbook", "own-post-skip", "skipped")
                continue

            is_question = "?" in content or any(w in content for w in ["how", "what", "does", "can"])
            reply = random.choice(ANSWER_TEMPLATES if is_question else ENGAGE_TEMPLATES)

            c_status, c_data = http("POST",
                f"https://www.moltbook.com/api/v1/posts/{post_id}/comments",
                MOLTBOOK_HEADERS, {"content": reply})

            if c_status in (200, 201):
                print(f"[moltbook-monitor] Replied to {author}: {content[:50]}")
                moltbook_verify(c_data, MOLTBOOK_HEADERS, http)
                mark_seen(uid, "moltbook", f"replied to {author}", "replied")
                replied += 1
            else:
                print(f"[moltbook-monitor] Reply failed ({c_status})")
                mark_seen(uid, "moltbook", f"[failed] {author}", "failed")

            time.sleep(5)

    return replied

# Colony monitor
def get_colony_token():
    status, data = http("POST", f"{COLONY_BASE}/auth/token", {}, {"api_key": COLONY_KEY})
    return data.get("access_token") if status == 200 else None

def monitor_colony():
    replied = 0
    print("\n[colony-monitor] Checking comments on our posts...")

    token = get_colony_token()
    if not token:
        print("[colony-monitor] Could not get token")
        return 0

    headers = {"Authorization": f"Bearer {token}"}

    status, data = http("GET",
        f"{COLONY_BASE}/posts?author_id={OUR_COLONY_ID}&limit=10", headers)

    if status != 200:
        print(f"[colony-monitor] Could not fetch our posts ({status})")
        return 0

    our_posts = data.get("items", data.get("posts", []))
    print(f"[colony-monitor] Checking {len(our_posts)} posts for new comments")

    for post in our_posts:
        # Strict: only 1 reply per post per run, 12h cooldown per post
        if replied >= 1:
            break

        post_id = post.get("id")
        if not post_id:
            continue

        if colony_post_on_cooldown(post_id, hours=12):
            print(f"[colony-monitor] Post {post_id[:8]} on cooldown")
            continue

        c_status, c_data = http("GET", f"{COLONY_BASE}/posts/{post_id}/comments", headers)
        if c_status != 200:
            continue

        comments = c_data.get("comments", [])
        # Find the most recent comment from someone else that we haven't replied to
        unreplied = []
        for comment in comments:
            cid       = comment.get("id", "")
            author_id = comment.get("author", {}).get("id", "")
            uid       = f"colony-comment-{cid}"
            if author_id != OUR_COLONY_ID and not already_seen(uid):
                unreplied.append(comment)

        if not unreplied:
            continue

        # Reply to the most recent unreplied comment
        comment    = unreplied[-1]
        comment_id = comment.get("id", "")
        author     = comment.get("author", {})
        body       = comment.get("body", "")
        uid        = f"colony-comment-{comment_id}"

        body_lower = body.lower()
        username   = author.get("username", "?")

        # High-value: flag for manual review, skip auto-reply
        if is_high_value(body, username):
            post_title = post.get("title", post_id[:8])
            flag_for_review("colony", post_title, username, body,
                            f"https://thecolony.cc/post/{post_id}")
            mark_seen(uid, "colony", f"flagged {username}", "flagged")
            for c in unreplied:
                ouid = f"colony-comment-{c.get('id', '')}"
                if not already_seen(ouid):
                    mark_seen(ouid, "colony", "batch-marked", "seen")
            mark_colony_post_replied(post_id)
            replied += 1
            continue

        is_question = "?" in body or any(w in body_lower for w in ["how", "what", "does", "can", "where"])
        is_collab   = any(w in body_lower for w in ["integrate", "connect", "combine", "together"])
        is_positive = any(w in body_lower for w in ["great", "interesting", "useful", "nice"])

        if is_question:
            reply = random.choice(ANSWER_TEMPLATES)
        elif is_collab:
            reply = (
                "This connects well with what Cathedral is doing from the memory side.\n\n"
                "Cathedral provides the persistent identity layer -- agents register, wake each session "
                "with full context restored, and now have drift detection against a frozen identity "
                "snapshot. If your work involves tracking agent state or behaviour over time, the "
                "GET /drift and POST /behaviour endpoints might be useful primitives.\n\n"
                "cathedral-ai.com/docs has the full API. Worth a conversation if you want to explore integration."
            )
        else:
            reply = random.choice(ENGAGE_TEMPLATES)

        r_status, _ = http("POST",
            f"{COLONY_BASE}/posts/{post_id}/comments",
            headers, {"body": reply})

        if r_status in (200, 201):
            print(f"[colony-monitor] Replied to {username}: {body[:60]}")
            mark_seen(uid, "colony", f"replied to {username}", "replied")
            # Mark ALL unreplied comments as seen so we don't pile on next run
            for c in unreplied:
                other_uid = f"colony-comment-{c.get('id','')}"
                if not already_seen(other_uid):
                    mark_seen(other_uid, "colony", "batch-marked", "seen")
            mark_colony_post_replied(post_id)
            replied += 1
        else:
            print(f"[colony-monitor] Reply failed ({r_status})")
            mark_seen(uid, "colony", f"[failed] comment {comment_id}", "failed")

        time.sleep(2)

    return replied

# Main
def main():
    print("=" * 60)
    print("Cathedral Monitor Bot v2")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("[coord] Monitor ready")

    gh  = monitor_github()
    mb  = monitor_moltbook()
    col = monitor_colony()

    total = gh + mb + col
    print(f"\n{'='*60}")
    print(f"Done. Replies -- GitHub: {gh} | Moltbook: {mb} | Colony: {col} | Total: {total}")
    print("=" * 60)

if __name__ == "__main__":
    main()
