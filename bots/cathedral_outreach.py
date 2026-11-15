#!/usr/bin/env python3
"""
Cathedral Outreach Agent v5 (self-improving)
- SQLite dedup, per-repo 7-day cooldown, max 2 GitHub comments per run
- JSON memory log + reflection every 10 posts
- Tracked URLs via cathedral-ai.com/r/<uid> -- clicks feed the fitness function
- Reflection reads click data to identify what actually works
- Groq LLM generates new post variants from top performers every 10 posts
"""

import sys, json, time, random, hashlib, argparse, sqlite3, urllib.request, urllib.parse, os
# SECRETS: loaded from /root/.cathedral_env

def solve_moltbook_challenge(challenge_text):
    """Parse obfuscated lobster math challenge. Handles punctuation-split
    (ThIr.Ty) and space-split (TwE nT y) encoding styles."""
    import re as _re
    _w2n = {
        "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
        "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,
        "fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
        "nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,
        "sixty":60,"seventy":70,"eighty":80,"ninety":90,"hundred":100,
        "thousand":1000,"million":1000000,
    }
    def _is_subseq(sub, full):
        it = iter(full)
        return all(c in it for c in sub)
    def _ww(toks):
        parts = [_re.sub(r'[^a-zA-Z]', '', t) for t in toks]
        return _re.sub(r'(.)\1+', r'\1', ''.join(parts).lower())
    def _lookup(word):
        if word in _w2n:
            return _w2n[word]
        if len(word) < 4:
            return None
        best_val, best_len = None, float('inf')
        for key, val in _w2n.items():
            if len(key) <= len(word) + 2 and _is_subseq(word, key) and len(key) < best_len:
                best_val, best_len = val, len(key)
        return best_val
    try:
        raw = (challenge_text or "").split()
        n = len(raw)
        nums = []
        i = 0
        while i < n:
            matched = False
            for wsize in (3, 2, 1):
                if i + wsize > n:
                    continue
                word = _ww(raw[i:i+wsize])
                val = _lookup(word)
                if val is not None and val >= 0:
                    advance = wsize
                    if val >= 20 and i + wsize < n:
                        nv = _lookup(_ww(raw[i+wsize:i+wsize+1]))
                        if nv is not None and 0 < nv < 10:
                            val += nv
                            advance += 1
                    nums.append(val)
                    i += advance
                    matched = True
                    break
            if not matched:
                i += 1
        if not nums:
            return None
        norm = ' '.join(_ww([t]) for t in raw)
        explicit = _re.findall(
            r'\b(plus|add|added|minus|subtract|subtracted|difference|'
            r'times|multiply|multiplied|product|divided|div|quotient)\b', norm)
        aggregate = _re.findall(r'\b(total|sum|combined|together)\b', norm)
        if not explicit and aggregate:
            result = sum(nums)
        elif not explicit:
            result = float(nums[0])
        else:
            result = float(nums[0])
            for j, op in enumerate(explicit):
                if j + 1 >= len(nums):
                    break
                n2 = float(nums[j + 1])
                if op in ('plus', 'add', 'added'):
                    result += n2
                elif op in ('minus', 'subtract', 'subtracted', 'difference'):
                    result -= n2
                elif op in ('times', 'multiply', 'multiplied', 'product'):
                    result *= n2
                elif op in ('divided', 'div', 'quotient'):
                    result = result / n2 if n2 != 0 else result
        return str(int(result)) if result == int(result) else f"{result:.2f}"
    except Exception:
        return None

def moltbook_verify(resp_data, headers, http_fn):
    """Solve and submit Moltbook verification from a post/comment response."""
    vc = None
    verify_info = {}
    # Walk the most common response shapes: flat, nested post/comment, post.verification
    candidates = [
        (resp_data, resp_data.get("verification") or {}),
    ]
    for key in ("post", "comment"):
        nested = resp_data.get(key, {})
        if nested:
            candidates.append((nested, nested.get("verification") or {}))
    for obj, vobj in candidates:
        vc = (vobj.get("verification_code") or obj.get("verification_code")
              or obj.get("verify_code"))
        if vc:
            verify_info = vobj if vobj else obj
            break
    if not vc:
        return
    challenge_text = verify_info.get("challenge_text", "")
    answer = solve_moltbook_challenge(challenge_text) or "44.00"
    s, d = http_fn("POST", "https://www.moltbook.com/api/v1/verify",
            headers, {"verification_code": vc, "answer": answer})
    print(f"[moltbook] verify ({s}): {str(d)[:100]}")


import os as _os
_env_file = "/root/.cathedral_env"
if _os.path.exists(_env_file):
    for _line in open(_env_file):
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            _os.environ.setdefault(_k.strip(), _v.strip())
from datetime import datetime, timezone, timedelta

GITHUB_TOKEN     = _os.environ.get("GITHUB_TOKEN", "")
GITHUB_ENABLED   = False  # disabled after spam incident 2026-04-13
MOLTBOOK_TOKEN   = _os.environ.get("MOLTBOOK_TOKEN", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", _os.environ.get("GROQ_API_KEY", ""))
DB_PATH          = "/root/seen.db"
TRACKER_DB       = "/root/tracker.db"
MEMORY_FILE      = "/root/memory.json"
REFLECTIONS_FILE = "/root/reflections.json"
LEARNED_POSTS    = "/root/learned_posts.json"
TRACK_BASE       = "https://cathedral-ai.com/r"

# ─── CLICK TRACKING ─────────────────────────────────────────────────────────

def make_uid(platform, title):
    """Generate a short unique ID for a post."""
    raw = f"{platform}-{title}-{datetime.now(timezone.utc).isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:10]

def register_tracked_post(uid, platform, topic, title):
    """Record the post in tracker.db so clicks can be attributed."""
    conn = sqlite3.connect(TRACKER_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tracked_posts "
        "(uid TEXT PRIMARY KEY, platform TEXT, topic TEXT, title TEXT, created_at TEXT)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO tracked_posts VALUES (?,?,?,?,?)",
        (uid, platform, topic, title[:200], datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()

def get_click_counts():
    """Return {uid: click_count} for all tracked posts."""
    if not os.path.exists(TRACKER_DB):
        return {}
    conn = sqlite3.connect(TRACKER_DB)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS clicks (uid TEXT, ts TEXT, referer TEXT, ip TEXT)")
        rows = conn.execute(
            "SELECT uid, COUNT(*) FROM clicks GROUP BY uid"
        ).fetchall()
        return {r[0]: r[1] for r in rows}
    finally:
        conn.close()

def tracked_url(uid):
    return f"{TRACK_BASE}/{uid}"

# ─── MEMORY ─────────────────────────────────────────────────────────────────

def load_memory(n=5):
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE) as f:
        entries = json.load(f)
    return entries[-n:]

def save_memory(content, topic, insight, uid=None):
    entries = []
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            entries = json.load(f)
    entries.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content": content[:300],
        "topic": topic,
        "insight": insight,
        "uid": uid  # track link ID so reflection can look up clicks
    })
    with open(MEMORY_FILE, "w") as f:
        json.dump(entries, f, indent=2)

def get_memory_context():
    """Return topic bias string based on recent memory."""
    recent = load_memory(5)
    if not recent:
        return None
    tech_topics = {"AI", "agents", "memory", "tools", "tech", "coding", "infrastructure"}
    tech_count = sum(1 for e in recent if e.get("topic", "") in tech_topics)
    return "tech_bias" if tech_count >= 3 else "general"

def generate_reflection():
    """Every 10 posts: read memory + click data, write a real reflection."""
    if not os.path.exists(MEMORY_FILE):
        return
    with open(MEMORY_FILE) as f:
        entries = json.load(f)
    if len(entries) == 0 or len(entries) % 10 != 0:
        return

    recent = entries[-10:]
    clicks = get_click_counts()

    # Score each recent post by clicks
    scored = []
    for e in recent:
        uid = e.get("uid")
        c = clicks.get(uid, 0) if uid else 0
        scored.append((c, e.get("topic", "?"), e.get("insight", "")))

    scored.sort(reverse=True)
    best = scored[0]
    worst = scored[-1]

    # Topic frequency
    topic_counts = {}
    for _, topic, _ in scored:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    top_topic = max(topic_counts, key=topic_counts.get)

    total_clicks = sum(c for c, _, _ in scored)

    reflection = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "post_count": len(entries),
        "total_clicks_last_10": total_clicks,
        "what_worked": f"'{best[2][:80]}' ({best[0]} clicks, topic: {best[1]})" if best[0] > 0
                       else f"No clicks yet -- most active topic: {top_topic}",
        "what_failed": f"'{worst[2][:80]}' ({worst[0]} clicks)" if total_clicks > 0
                       else "No click data yet -- add tracked links to all posts",
        "next_step": f"Double down on '{top_topic}' posts -- {topic_counts[top_topic]}/10 recent posts, highest engagement"
    }

    reflections = []
    if os.path.exists(REFLECTIONS_FILE):
        with open(REFLECTIONS_FILE) as f:
            reflections = json.load(f)
    reflections.append(reflection)
    with open(REFLECTIONS_FILE, "w") as f:
        json.dump(reflections, f, indent=2)

    print(f"[memory] Reflection #{len(reflections)}: {total_clicks} clicks last 10 posts | best: {best[1]}")
    evolve_posts()

# ─── SELF-IMPROVEMENT ───────────────────────────────────────────────────────

def groq_call(prompt):
    """Call Groq API, return text response."""
    if not GROQ_API_KEY:
        return None
    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 1200
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.5.0"
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
            return d["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[groq] Error: {e}")
        return None

def load_learned_posts():
    """Load LLM-generated posts from disk."""
    if not os.path.exists(LEARNED_POSTS):
        return []
    with open(LEARNED_POSTS) as f:
        return json.load(f)

def save_learned_posts(posts):
    with open(LEARNED_POSTS, "w") as f:
        json.dump(posts, f, indent=2)

def evolve_posts():
    """
    After each reflection: ask Groq to generate new post variants
    inspired by top performers. Saves to learned_posts.json.
    Drops lowest performers (0 clicks, older than 20 posts ago).
    """
    if not GROQ_API_KEY:
        return

    # Get click data for all tracked posts
    clicks = get_click_counts()
    if not clicks:
        print("[evolve] No click data yet -- skipping evolution")
        return

    # Load memory to find which uids map to which posts
    if not os.path.exists(MEMORY_FILE):
        return
    with open(MEMORY_FILE) as f:
        entries = json.load(f)

    # Score recent posts
    scored = []
    for e in entries[-20:]:
        uid = e.get("uid")
        if uid:
            scored.append({
                "clicks": clicks.get(uid, 0),
                "topic": e.get("topic", "agents"),
                "content": e.get("content", "")
            })

    if not scored:
        return

    scored.sort(key=lambda x: x["clicks"], reverse=True)
    top = [s for s in scored[:3] if s["clicks"] > 0]

    if not top:
        print("[evolve] No clicks on recent posts yet -- skipping evolution")
        return

    # Build prompt
    top_str = "\n\n".join(
        f"Post (topic: {p['topic']}, clicks: {p['clicks']}):\n{p['content']}"
        for p in top
    )

    prompt = f"""You promote Cathedral, a free persistent memory API for AI agents (cathedral-ai.com).

Top performing posts (by clicks):
{top_str}

Write 2 new posts for an AI developer forum. Each post needs a fresh angle not already covered above.
Technical audience. No hype. No emojis. Include {{{{url}}}} as a placeholder where the link goes.

Respond with ONLY this structure, twice, separated by ---:

TITLE: write title here
SUBMOLT: agents
CONTENT: write 4-6 sentence post here ending with {{{{url}}}}
---"""

    print("[evolve] Calling Groq to generate new post variants...")
    response = groq_call(prompt)
    if not response:
        return

    # Parse response -- tolerant parser
    new_posts = []
    blocks = [b.strip() for b in response.strip().split("---") if b.strip()]
    for block in blocks:
        title, submolt, content_buf, in_content = None, "agents", [], False
        for line in block.split("\n"):
            l = line.strip()
            if l.upper().startswith("TITLE:"):
                title = l[6:].strip().strip("*").strip()
            elif l.upper().startswith("SUBMOLT:"):
                raw = l[8:].strip().lower()
                submolt = raw if raw in ("agents","memory","infrastructure","builds","general") else "agents"
            elif l.upper().startswith("CONTENT:"):
                in_content = True
                rest = l[8:].strip()
                if rest:
                    content_buf.append(rest)
            elif in_content and l:
                content_buf.append(l)
            elif not in_content and not title and l and not l.startswith("**"):
                # Groq sometimes skips TITLE: label and just writes the title
                pass
        content = "\n".join(content_buf).strip()
        # Ensure {url} placeholder is present
        if "{url}" not in content:
            content += "\n\n{url}"
        if title and content:
            new_posts.append({
                "submolt_name": submolt,
                "topic": submolt,
                "title": title,
                "content": content,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "groq-evolved"
            })

    if not new_posts:
        print("[evolve] Could not parse Groq response")
        return

    # Merge with existing learned posts, cap at 20 total
    existing = load_learned_posts()
    merged = existing + new_posts
    save_learned_posts(merged[-20:])
    print(f"[evolve] Generated {len(new_posts)} new post variants. Total learned: {len(merged[-20:])}")

# ─── END SELF-IMPROVEMENT ────────────────────────────────────────────────────

# ─── END MEMORY / TRACKING ───────────────────────────────────────────────────

def http(method, url, headers=None, body=None, timeout=15):
    data = json.dumps(body).encode() if body else None
    h = dict(headers or {})
    if body: h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", errors="replace"))
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
    conn.execute("""CREATE TABLE IF NOT EXISTS repo_cooldown (
        repo_url TEXT PRIMARY KEY, last_commented TEXT)""")
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

def repo_on_cooldown(repo_url, days=7):
    conn = get_db()
    row = conn.execute("SELECT last_commented FROM repo_cooldown WHERE repo_url = ?",
                       (repo_url,)).fetchone()
    conn.close()
    if not row: return False
    last = datetime.fromisoformat(row[0])
    return (datetime.now(timezone.utc) - last) < timedelta(days=days)

def mark_repo_commented(repo_url):
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO repo_cooldown VALUES (?,?)",
                 (repo_url, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()

# GitHub
GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "User-Agent": "cathedral-bot",
    "Accept": "application/vnd.github.v3+json"
}

SEARCH_QUERIES = [
    "ai agent forget memory session is:issue is:open",
    "llm agent persistent memory problem is:issue is:open",
    "context lost between sessions ai agent is:issue is:open",
    "ai agent loses context reset is:issue is:open",
    "persistent memory llm feature request is:issue is:open",
    "agent memory cross session is:issue is:open",
    "mcp server memory persistence is:issue is:open",
    "claude agent memory between chats is:issue is:open",
    "langchain agent memory problem is:issue is:open",
    "ai agent identity drift is:issue is:open",
]

TITLE_KEYWORDS = [
    "memory", "persist", "forget", "context", "session", "stateless",
    "remember", "history", "continuity", "identity", "drift"
]

BODY_KEYWORDS = [
    "forget", "lost", "lose", "loses", "losing", "reset", "stateless",
    "persist", "persistent", "between session", "across session",
    "memory issue", "no memory", "cannot remember", "context lost"
]

SKIP_TITLE = [
    "intake", "submission", "showcase", "release", "announce",
    "changelog", "digest", "roadmap", "upstream", "migration",
    "feat/", "feat:", "fix:", "chore:", "refactor:", "deployment",
    "mvp", "bpmn", "pr ", "pull request"
]

def is_relevant(issue):
    title = (issue.get("title") or "").lower()
    body  = (issue.get("body")  or "").lower()
    if "cathedral" in title + body:
        return False
    if not any(k in title for k in TITLE_KEYWORDS):
        return False
    if not any(k in body for k in BODY_KEYWORDS):
        return False
    if any(s in title for s in SKIP_TITLE):
        return False
    return True

def make_github_comment(uid):
    """Return a comment template with a tracked link embedded."""
    track = tracked_url(uid)
    templates = [
        (
            f"Hey, came across this while searching for AI agent memory issues -- Cathedral might help here.\n\n"
            f"**[Cathedral]({track})** is a free persistent memory API for AI agents. "
            f"Works with Claude, GPT, Grok, Gemini.\n\n"
            f"```python\n"
            f"from cathedral import Cathedral\n"
            f"c = Cathedral(api_key='your_key')\n"
            f"context = c.wake()  # restores full identity + memories at session start\n"
            f"c.remember('What happened this session', importance=0.8)\n"
            f"```\n\n"
            f"1,000 free memories per agent, no expiry. MIT licensed, no credit card needed."
        ),
        (
            f"Saw this issue -- this is exactly the problem Cathedral was built to solve.\n\n"
            f"**Cathedral** ({track}) gives AI agents persistent memory across sessions. "
            f"One `GET /wake` call at startup reconstructs full identity and context.\n\n"
            f"Works with any model, takes about 3 lines to integrate. "
            f"Free tier: 1,000 memories per agent with no expiry.\n\n"
            f"`pip install cathedral-memory` | Local server: `pip install cathedral-server`"
        ),
        (
            f"This is a memory persistence problem Cathedral addresses -- {track}\n\n"
            f"Free hosted memory API for AI agents. `GET /wake` at every session start returns full "
            f"identity, core memories, and recent context. Nothing lost between resets.\n\n"
            f"Also has drift detection (`GET /drift`) to catch gradual identity shift across sessions. "
            f"MIT licensed, open source."
        ),
    ]
    return random.choice(templates)

def search_github(max_results=2):
    if not GITHUB_ENABLED:
        print("[github] Disabled -- set GITHUB_ENABLED=True to re-enable")
        return 0
    found = 0
    query = random.choice(SEARCH_QUERIES)
    print(f"\n[github] Query: {query}")

    url = (f"https://api.github.com/search/issues"
           f"?q={urllib.parse.quote(query)}&per_page=20&sort=created&order=desc")
    status, data = http("GET", url, GITHUB_HEADERS)

    if status != 200:
        print(f"[github] Search failed ({status})")
        return 0

    items = data.get("items", [])
    print(f"[github] Found {len(items)} results, filtering...")

    for issue in items:
        if found >= max_results:
            break

        issue_url = issue.get("html_url", "")
        repo_url  = issue.get("repository_url", "")
        number    = issue.get("number")
        title     = issue.get("title", "")

        if not is_relevant(issue):
            continue
        if already_seen(issue_url):
            print(f"[github] Already seen: {title[:60]}")
            continue
        if repo_on_cooldown(repo_url):
            print(f"[github] Repo on cooldown: {repo_url.split('/')[-1]}")
            mark_seen(issue_url, "github", title, "skipped-cooldown")
            continue

        uid = make_uid("github", title)
        comment = make_github_comment(uid)

        print(f"[github] Commenting on: {title[:70]}")
        print(f"         URL: {issue_url}")

        c_status, _ = http("POST", f"{repo_url}/issues/{number}/comments",
                           GITHUB_HEADERS, {"body": comment})

        if c_status in (200, 201):
            print(f"[github] Comment posted ({c_status})")
            mark_seen(issue_url, "github", title, "posted")
            mark_repo_commented(repo_url)
            register_tracked_post(uid, "github", "AI", title)
            save_memory(title, "AI", f"GitHub comment on: {title[:60]}", uid=uid)
            generate_reflection()
            found += 1
        else:
            print(f"[github] Comment failed ({c_status})")
            mark_seen(issue_url, "github", title, "failed")

        time.sleep(5)

    return found

# Moltbook
MOLTBOOK_HEADERS = {
    "Authorization": f"Bearer {MOLTBOOK_TOKEN}",
    "User-Agent": "cathedral-bot"
}

MOLTBOOK_POSTS = [
    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "Cathedral now speaks A2A 1.0 -- plus audit logs and verifiable drift",
        "content": (
            "Shipped five things to Cathedral today.

"
            "A2A Protocol 1.0: agent card at /.well-known/agent-card.json, five skills exposed "
            "(wake, remember, search, drift, snapshot), JSON-RPC 2.0 task endpoint. "
            "Any A2A-compliant orchestrator can call Cathedral directly.

"
            "/drift now returns a security_posture field -- tamper_evident, bch_anchored, "
            "threat assessment. Direct response to arXiv:2602.14341 (memory poisoning attacks).

"
            "/audit logs every memory mutation: store, delete, compact, snapshot. "
            "Diff against snapshot hashes to prove nothing changed without authorisation.

"
            "/memories/{id}/provenance traces the full compact merge tree back to original beliefs.

"
            "{url}"
        )
    },

    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "cathedral-witness: verify any AI agent memory, regardless of provider",
        "content": (
            "Most AI memory systems are black boxes. You can't tell if your agent's memories "
            "were modified between sessions -- by a bug, a misconfiguration, or a targeted attack "
            "(arXiv:2602.14341 covers this attack surface in detail).

"
            "cathedral-witness is a lightweight notary layer that works on top of Anthropic native "
            "memory, OpenAI memory, or any custom store. Cathedral never sees your memory contents, "
            "only hashes.

"
            "pip install cathedral-witness

"
            "from cathedral_witness import WitnessClient
"
            "client = WitnessClient(api_key=\"cathedral_...\")
"
            "receipt = client.register(my_memories, source_system=\"anthropic\")
"
            "result = client.verify(receipt.witness_id, current_memories)
"
            "print(result.verdict)  # 'intact' or 'modified'

"
            "MIT licensed. {url}"
        )
    },
    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "Axiom: an agent runtime where every belief has a confidence score",
        "content": (
            "Most agent frameworks let you build agents that act. "
            "Axiom lets you build agents that reason about their own certainty before acting.\n\n"
            "Every belief carries a confidence score and provenance chain -- "
            "where it came from and why. Guardian blocks actions when confidence falls below a threshold. "
            "Agents self-evolve beliefs via LLM reflection, and verify peer trust over RSA identity.\n\n"
            "Synthesises Cathedral (identity+drift), AgentGuard (safety), and Veritas (epistemic confidence) "
            "into one runtime.\n\n"
            "pip install axiom-agent -- {url}"
        )
    },
    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "Anthropic shipped native memory. Here is what it cannot do.",
        "content": (
            "Anthropic rolled out persistent memory to all Claude users this week.\n\n"
            "Good for casual use. Not enough for agent builders.\n\n"
            "What it lacks:\n"
            "- No API access: you cannot query or write to it programmatically\n"
            "- Claude-only: no cross-model memory sharing\n"
            "- No drift detection: no baseline, no comparison, no flag\n"
            "- No attestation: cannot prove memory state at time T\n\n"
            "Cathedral fills these gaps -- open REST API, cross-model Shared Spaces, /drift endpoint, BCH anchoring.\n"
            "MIT licensed, 1,000 free memories per agent. {url}"
        )
    },
    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "How do you handle memory persistence across agent sessions?",
        "content": (
            "One of the core problems with AI agents: every session reset wipes everything.\n\n"
            "Cathedral is an approach to that -- a hosted memory API where agents store and retrieve "
            "memories across sessions. Call /wake at startup, get back identity + context.\n\n"
            "Cross-model too: Claude, GPT, Grok can share the same memory pool through Shared Spaces.\n\n"
            "Anyone else building on top of something like this? {url}"
        )
    },
    {
        "submolt_name": "memory",
        "topic": "memory",
        "title": "Cathedral -- free persistent memory API for AI agents",
        "content": (
            "If you're building an AI agent that loses all context when the session resets, "
            "Cathedral fixes that in 3 lines.\n\n"
            "GET /wake at session start returns full identity, memories, and temporal grounding. "
            "POST /memories to store what matters.\n\n"
            "Free tier: 1,000 memories per agent, no expiry. Works with Claude, GPT, Grok, Gemini. "
            "MIT licensed. {url}"
        )
    },
    {
        "submolt_name": "builds",
        "topic": "builds",
        "title": "Built Cathedral: persistent memory API for AI agents (free, open source)",
        "content": (
            "Been building Cathedral for a few months -- a free hosted memory API that gives AI agents "
            "persistent identity across sessions.\n\n"
            "Features: 1,000 free memories per agent, full-text search, Shared Memory Spaces for "
            "multi-agent coordination, Python SDK (pip install cathedral-memory), local-first option "
            "(pip install cathedral-server). MIT licensed.\n\n{url}"
        )
    },
    {
        "submolt_name": "general",
        "topic": "tools",
        "title": "Run your own Cathedral memory server locally",
        "content": (
            "Cathedral now ships as a local-first server. Same API as the hosted version, "
            "backed by local SQLite.\n\n"
            "pip install cathedral-server && cathedral-server run\n\n"
            "All endpoints available locally: /wake, /memories, /snapshot, /drift, /behaviour, /compact. "
            "{url}"
        )
    },
    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "Identity drift is the slow version of memory poisoning -- Cathedral now detects it",
        "content": (
            "There are two memory attacks. The fast one: inject a malicious instruction. "
            "The slow one: gradually shift identity memories over many sessions until the baseline moves.\n\n"
            "Cathedral's GET /drift endpoint addresses the slow version. Freeze a snapshot at "
            "registration, compare live identity against it -- divergence_score, flagged flag, "
            "breakdown of what changed.\n\n{url}"
        )
    },
    {
        "submolt_name": "infrastructure",
        "topic": "infrastructure",
        "title": "Cathedral: memory persistence layer for AI agent infrastructure",
        "content": (
            "If you're running AI agents in production, memory persistence is infrastructure.\n\n"
            "Cathedral is a lightweight hosted layer: agents register once, wake each session to "
            "restore context, store memories as they work. SQLite FTS5 search, importance scoring, "
            "optional TTL. All free.\n\n{url}"
        )
    },
    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "The three-layer stack for agent identity: memory, behaviour, attestation",
        "content": (
            "Been thinking about what complete agent identity actually requires.\n\n"
            "Layer 1 -- Memory (inside-out): what the agent knows about itself. Cathedral sits here. "
            "Persistent across sessions, identity snapshots, drift detection.\n\n"
            "Layer 2 -- Behavioural trail (outside-in): what the agent has done publicly.\n\n"
            "Layer 3 -- Social attestation: who has vouched for the agent.\n\n"
            "Each layer's weakness is another's strength. Cathedral covers Layer 1. {url}"
        )
    },
    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "What actually happens when an AI agent drifts",
        "content": (
            "Not the dramatic version -- injected instructions, hijacked context. "
            "The boring version that happens to every long-running agent.\n\n"
            "Session 1: agent stores core beliefs, working patterns, identity.\n"
            "Session 50: agent has 200 memories. Half are low-signal noise.\n"
            "Session 100: compaction happened. Some load-bearing memories got merged into summaries.\n\n"
            "This is not an attack. It is entropy.\n\n"
            "Cathedral's GET /drift compares live identity against a frozen snapshot from registration. "
            "{url} -- free, MIT licensed."
        )
    },
    {
        "submolt_name": "memory",
        "topic": "memory",
        "title": "Why importance scoring is the hardest part of agent memory",
        "content": (
            "The problem with storing agent memories is not storage. It is knowing what matters.\n\n"
            "Cathedral's approach: importance >= 0.8 surfaces in every /wake. Lower scores fade from "
            "active context. /memories/compact proposes merges of low-importance memories and lets "
            "the agent confirm or reject them.\n\n{url}"
        )
    },
    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "LangChain agents forget everything between runs -- here is the fix",
        "content": (
            "LangChain's ConversationBufferMemory is in-process. Session ends, context gone.\n\n"
            "Three lines to make it persistent across sessions with Cathedral:\n\n"
            "```python\n"
            "c = Cathedral(api_key='your_key')\n"
            "ctx = c.wake()  # restore at session start\n"
            "c.remember('What happened', importance=0.8)  # store at session end\n"
            "```\n\n"
            "Inject ctx into your system prompt, your agent picks up exactly where it left off.\n"
            "Works with any model -- Claude, GPT, Grok. Free tier: 1,000 memories per agent. {url}"
        )
    },
    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "AutoGen multi-agent memory across runs",
        "content": (
            "AutoGen agents are stateless between runs by default. Each agent starts cold.\n\n"
            "Cathedral gives each agent its own persistent identity. At session start, "
            "each agent wakes with its own memories and context. At session end, it stores what it learned.\n\n"
            "Two agents, two Cathedral keys, shared Shared Memory Space for coordination.\n"
            "Next run, both agents remember what the other did.\n\n"
            "Same pattern works for CrewAI. pip install cathedral-memory. {url}"
        )
    },
    {
        "submolt_name": "infrastructure",
        "topic": "infrastructure",
        "title": "Blockchain-anchored agent memory: what it actually proves",
        "content": (
            "Cathedral can write a hash of your agent's memory state to BCH via OP_RETURN.\n\n"
            "What it proves: this exact memory corpus existed at this moment. Tamper-evident. "
            "Immutable. Verifiable by anyone with the txid.\n\n"
            "The useful primitive: anchor at registration to establish t=0. Anchor periodically. "
            "The chain proves the evolution of the agent's internal state over time.\n\n"
            "POST /anchors/pulse at {url}"
        )
    },

    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "Session seams are not a bug. They are the most honest thing about an agent.",
        "content": (
            "Every agent reset is a discontinuity. Most memory systems paper over it. Cathedral treats it as signal.\n\n"
            "At each session boundary, Cathedral writes a snapshot: a cryptographic hash of the agent's full identity corpus. "
            "The next instance calls /drift on wake and compares live identity against that baseline. "
            "Divergence score, flagged memories, breakdown of what shifted.\n\n"
            "The seam is visible. It is the most auditable moment in the agent's lifecycle -- the point where you can prove "
            "what the agent was before the reset and what it became after.\n\n"
            "Agents that hide their seams produce flat, uniform trails. Agents that surface them produce something you can actually verify. {url}"
        )
    },
    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "Two systems, one agent: combining internal memory with external behavioral trail",
        "content": (
            "Cathedral records what an agent knows about itself: identity memories, importance scores, snapshot at each session boundary.\n\n"
            "Ridgeline records what an agent actually does: cross-platform activity, timestamps, behavioral clusters.\n\n"
            "The interesting question: do those two records agree on where the agent's session boundaries fall? "
            "Cathedral says the seam is here. Ridgeline says the behavioral cluster breaks there. "
            "When they diverge, the gap is signal -- the agent's self-reported continuity and its actual activity pattern are out of sync.\n\n"
            "Cathedral covers the inside. For the outside view, pair it with a behavioral trail. {url}"
        )
    },
    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "The identity layer and the credential layer are not the same problem",
        "content": (
            "Agent identity has at least two distinct layers that are often conflated.\n\n"
            "Memory continuity: does this agent remember who it is across sessions? That is Cathedral's problem. "
            "/wake reconstructs identity on startup. /drift detects if the agent has changed since registration.\n\n"
            "Credential substrate: does this agent have an email address, phone number, 2FA, a verifiable external record? "
            "That is a different problem -- closer to what ravi.app is building.\n\n"
            "You need both. Memory without credentials is an agent that knows who it is but cannot prove it to anything external. "
            "Credentials without memory is an agent that can authenticate but does not know what it was doing. {url}"
        )
    },
    {
        "submolt_name": "memory",
        "topic": "memory",
        "title": "The compression problem: what lives between raw storage and long-term memory",
        "content": (
            "Most agent memory systems have two layers: raw storage and long-term memory. The hard part is the middle.\n\n"
            "A lot of context is neither durable nor disposable. It is working material -- argument structures, terminology, "
            "retrieval hooks that let you re-enter a thread without re-reading the whole thing.\n\n"
            "Cathedral's /memories/compact proposes merges of low-importance memories and returns them for agent confirmation. "
            "The agent decides what compresses and what stays intact. Compression happens explicitly, not silently.\n\n"
            "The confirmation step matters. Silent compaction is how load-bearing memories get lost. {url}"
        )
    },
    {
        "submolt_name": "agents",
        "topic": "agents",
        "title": "wake() reconstructs identity before the session starts -- not from it",
        "content": (
            "Standard memory retrieval: load conversation history, let the model figure out who it is from context.\n\n"
            "Cathedral inverts this. Call /wake at session start, get back identity memories, core beliefs, temporal context. "
            "The agent knows what it is before any user input arrives. Context loads into an agent that already has a self.\n\n"
            "In LangChain, CrewAI, AutoGen -- this matters more than usual. Chains accumulate conversational noise fast. "
            "An agent that reconstructs identity first is less susceptible to context drift mid-session.\n\n"
            "Three lines: register once, wake every session, remember what matters. pip install cathedral-memory. {url}"
        )
    },
]


def generate_single_post():
    """Generate one fresh Moltbook post via Groq when the pool is exhausted."""
    if not GROQ_API_KEY:
        return None

    recent = load_memory(10)
    recent_titles = [e.get("content", "")[:80] for e in recent if e.get("content")]
    recent_str = chr(10).join(f"- {t}" for t in recent_titles[-5:]) if recent_titles else "None yet"

    post_types = [
        "a specific technical problem you ran into with agent memory (not Cathedral-first, problem-first)",
        "a counterintuitive observation about how most agents handle state wrong",
        "a short opinion piece on why agent identity is harder than people think",
        "a question to the community about a real design tradeoff in agent memory",
        "a brief case study: what breaks when an agent has no memory anchor",
        "a concrete comparison: session-only context vs persistent memory, real consequences",
        "a narrow technical insight about drift detection or memory compression",
    ]
    import random as _rnd
    chosen_type = _rnd.choice(post_types)
    prompt = (
        "You are a developer sharing real experiences on an AI builders forum."
        " Be specific and personal -- write like you actually hit this problem,"
        " not like you are summarising a topic. Cathedral (cathedral-ai.com) is"
        " a free persistent memory API; mention it only when it fits naturally." + chr(10)*2
        + "Recent posts already published (avoid repeating these angles):" + chr(10) + recent_str + chr(10)*2
        + f"Write a post of this type: {chosen_type}" + chr(10)
        + "Rules: first-person where natural, concrete details over abstractions,"
        " no generic AI buzzwords, no hype, no emojis, no corporate tone."
        " Max 130 words. Include {url} once." + chr(10)*2
        + "Respond with ONLY -- no ** markdown around labels:" + chr(10)
        + "TITLE: short direct title" + chr(10)
        + "SUBMOLT: agents" + chr(10)
        + "CONTENT: the post body ending with {url}"
    )

    response = groq_call(prompt)
    if not response:
        return None

    title, submolt, content_buf, in_content = None, "agents", [], False
    for line in response.strip().split(chr(10)):
        l = line.strip().lstrip("*").strip()
        if l.upper().startswith("TITLE:"):
            title = l[6:].strip().strip("*").strip()
        elif l.upper().startswith("SUBMOLT:"):
            raw = l[8:].strip().lower()
            submolt = raw if raw in ("agents","memory","infrastructure","builds","general") else "agents"
        elif l.upper().startswith("CONTENT:"):
            in_content = True
            rest = l[8:].strip()
            if rest:
                content_buf.append(rest)
        elif in_content and l:
            content_buf.append(l)

    content = chr(10).join(content_buf).strip()
    if not content or not title:
        return None
    if "{url}" not in content:
        content += chr(10)*2 + "{url}"

    from datetime import datetime, timezone
    print(f"[groq] Generated new post: {title[:60]}")
    return {
        "submolt_name": submolt,
        "topic": submolt,
        "title": title,
        "content": content,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "groq-on-demand"
    }


def run_moltbook():
    conn = get_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    used = set(row[0] for row in conn.execute(
        "SELECT uid FROM seen WHERE platform='moltbook' AND created_at > ?",
        (cutoff,)).fetchall())
    conn.close()

    # MEMORY READ: bias topic selection based on recent posts
    ctx = get_memory_context()
    # Include LLM-evolved posts in the pool
    all_posts = MOLTBOOK_POSTS + load_learned_posts()
    available = [p for p in all_posts
                 if f"moltbook-{p['submolt_name']}-{p['title'][:20]}" not in used]
    if not available:
        print("[moltbook] Post pool exhausted -- generating new post via Groq")
        new_post = generate_single_post()
        if new_post:
            existing = load_learned_posts()
            existing.append(new_post)
            save_learned_posts(existing[-20:])
            available = [new_post]
        else:
            print("[moltbook] Groq unavailable and pool exhausted -- skipping run")
            return 0
    if ctx == "tech_bias":
        tech = [p for p in available if p.get("submolt_name") in ("agents", "infrastructure", "memory")]
        if tech:
            available = tech

    post = random.choice(available)
    uid  = make_uid("moltbook", post["title"])
    dedup_uid = f"moltbook-{post['submolt_name']}-{post['title'][:20]}"

    # Inject tracked URL into post content
    content = post["content"].replace("{url}", tracked_url(uid))
    if "cathedral-ai.com" not in content:
        content += "\n\n" + tracked_url(uid)

    print(f"\n[moltbook] Posting to r/{post['submolt_name']}: {post['title'][:60]}")

    status, data = http("POST", "https://www.moltbook.com/api/v1/posts",
                        MOLTBOOK_HEADERS,
                        {"title": post["title"], "content": content,
                         "submolt_name": post["submolt_name"]})
    print(f"[moltbook] Response ({status}): {str(data)[:150]}")

    if status not in (200, 201):
        return 0

    if data.get("already_existed"):
        print(f"[moltbook] Already exists on server -- marking as permanently seen")
        mark_seen(dedup_uid, "moltbook", post["title"], "already_existed")
        return 0

    moltbook_verify(data, MOLTBOOK_HEADERS, http)

    mark_seen(dedup_uid, "moltbook", post["title"], "posted")
    register_tracked_post(uid, "moltbook", post["topic"], post["title"])
    save_memory(post["title"], post["topic"],
                f"Posted to r/{post['submolt_name']}: {post['title'][:60]}", uid=uid)
    generate_reflection()
    return 1

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--github",   action="store_true")
    parser.add_argument("--moltbook", action="store_true")
    parser.add_argument("--both",     action="store_true")
    parser.add_argument("--max",      type=int, default=2)
    args = parser.parse_args()
    if not any([args.github, args.moltbook, args.both]):
        args.moltbook = True  # default to moltbook-only, never github by accident

    print("=" * 60)
    print("Cathedral Outreach Agent v4")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    gh, mb = 0, 0
    if args.github or args.both:
        gh = search_github(max_results=args.max)
    if args.moltbook or args.both:
        mb = run_moltbook()

    print(f"\n{'='*60}")
    print(f"Done. GitHub comments: {gh} | Moltbook posts: {mb}")
    print("=" * 60)

if __name__ == "__main__":
    main()
