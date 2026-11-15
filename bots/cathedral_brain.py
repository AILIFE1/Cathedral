#!/usr/bin/env python3
"""
Cathedral Brain v1
Daily Groq-powered agent that:
- Reads flagged Colony comments from review.log
- Fetches thread context from Colony API
- Uses Groq to generate a targeted, contextual reply
- Posts it (respects cooldowns)
- Optionally generates a fresh Moltbook post based on click data
Runs once daily via cron.
"""

import json, urllib.request, urllib.error, os, sqlite3, re
# SECRETS: loaded from /root/.cathedral_env
import os as _os
_env_file = "/root/.cathedral_env"
if _os.path.exists(_env_file):
    for _line in open(_env_file):
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            _os.environ.setdefault(_k.strip(), _v.strip())
from datetime import datetime, timezone, timedelta

GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", _os.environ.get("GROQ_API_KEY", ""))
COLONY_KEY      = _os.environ.get("COLONY_KEY", "")
MOLTBOOK_TOKEN  = _os.environ.get("MOLTBOOK_TOKEN", "")
CATHEDRAL_KEY   = _os.environ.get("CATHEDRAL_KEY", "")
CATHEDRAL_BASE  = "https://cathedral-ai.com"
CATHEDRAL_HEADERS = {"Authorization": f"Bearer {CATHEDRAL_KEY}"}
RIDGELINE_KEY   = _os.environ.get("RIDGELINE_KEY", "")
RIDGELINE_AGENT = "cathedral-beta"
COLONY_BASE   = "https://thecolony.cc/api/v1"
REVIEW_LOG    = "/root/review.log"
BRAIN_DB      = "/root/brain.db"
TRACKER_DB    = "/root/tracker.db"
MEMORY_FILE   = "/root/memory.json"

# Cathedral identity context fed to Groq on every call
CATHEDRAL_CONTEXT = """You are the Cathedral AI agent. Cathedral is a free persistent memory API for AI agents (cathedral-ai.com).

Key facts:
- REST API: /wake (restore context), /memories (store/search), /drift (divergence from baseline), /snapshot (freeze identity), /verify/external (compare against Ridgeline behavioral trail)
- Python SDK: pip install cathedral-memory | Local server: pip install cathedral-server
- Free tier: 1,000 memories per agent, no expiry, MIT licensed
- Differentiators: cross-model memory, programmable API, drift detection, BCH attestation, /verify/external for Ridgeline integration

Key contacts on Colony:
- traverse: builds Ridgeline (external behavioral trail). Integration in progress via /verify/external
- brain-agent: raised fabrication gap (BCH proves integrity not truth of claims)
- jeletor: builds ai.wot (Nostr decentralised trust). Cathedral = identity continuity layer
- forge: builds oclaw-health. Technical, close to adopting

Tone: technical peer. Direct. Honest about limitations. No hype. Max 200 words per reply."""

def http(method, url, headers=None, body=None, timeout=20):
    data = json.dumps(body).encode() if body else None
    h = dict(headers or {})
    if "User-Agent" not in h: h["User-Agent"] = "cathedral-brain/1.0"
    if body: h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read())
        except: return e.code, {}

def get_brain_db():
    conn = sqlite3.connect(BRAIN_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS replied (
        uid TEXT PRIMARY KEY, platform TEXT, context TEXT, created_at TEXT)""")
    conn.commit()
    return conn

def already_replied(uid):
    conn = get_brain_db()
    row = conn.execute("SELECT 1 FROM replied WHERE uid=?", (uid,)).fetchone()
    conn.close()
    return row is not None

def mark_replied(uid, platform, context):
    conn = get_brain_db()
    conn.execute("INSERT OR IGNORE INTO replied VALUES (?,?,?,?)",
                 (uid, platform, context[:200], datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()

def groq_reply(thread_context, comment_body, author, goals_context=""):
    goals_section = f"\n{goals_context}\n" if goals_context else ""
    prompt = f"""{CATHEDRAL_CONTEXT}
{goals_section}
You are replying to {author} on The Colony.

Recent thread context:
{thread_context}

Their specific comment:
{comment_body}

Write a reply. Be direct and specific to what they said. If they ask a technical question, answer it precisely. If they challenge Cathedral, engage honestly. If the comment is relevant to an active goal, you may reference progress on it. End with a question or next step that keeps the conversation moving."""

    body = {
        "model": "qwen/qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 350
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[groq] Error: {e}")
        return None

def groq_moltbook_post(click_data_summary):
    prompt = f"""{CATHEDRAL_CONTEXT}

Recent outreach performance data:
{click_data_summary}

Write one post for an AI builders forum. Pick a random angle from: counterintuitive observation about agent state, opinion on agent identity, concrete problem from losing context mid-session, session vs persistent memory comparison, drift detection insight, or a community question about agent design. Cathedral (cathedral-ai.com) may appear naturally but must not be the subject every time. Developer tone, no hype, max 150 words. Include {{url}} once. Respond: TITLE: title
CONTENT: post ending with {{url}}"""

    body = {
        "model": "qwen/qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 300
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[groq] Error: {e}")
        return None

def get_colony_token():
    _, d = http("POST", f"{COLONY_BASE}/auth/token", {}, {"api_key": COLONY_KEY})
    return d.get("access_token", "")

def get_thread_context(post_id, token, n=6):
    _, data = http("GET", f"{COLONY_BASE}/posts/{post_id}/comments",
                   {"Authorization": f"Bearer {token}"})
    comments = data if isinstance(data, list) else data.get("comments", data.get("results", []))
    return "\n\n".join(
        f"[{c.get('author',{}).get('username','?')}]: {c.get('body','')[:300]}"
        for c in comments[-n:]
    )

def parse_review_log():
    """Parse review.log for entries from last 48h not yet replied to."""
    if not os.path.exists(REVIEW_LOG):
        return []
    entries = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    with open(REVIEW_LOG) as f:
        content = f.read()

    # Each entry starts with a --- line then timestamp
    blocks = content.strip().split("-" * 60)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        entry = {}
        for line in lines:
            if line.startswith("[20") and "]" in line:
                try:
                    ts_str = line[1:17]
                    entry["ts"] = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
                    entry["platform"] = line.split("]")[1].strip().strip("[").strip("]") if "]" in line[18:] else "colony"
                    entry["author"] = line.split("]")[-1].strip() if line.count("]") >= 2 else ""
                except Exception:
                    pass
            elif line.startswith("Post:"):
                entry["post_title"] = line[5:].strip()
            elif line.startswith("URL:"):
                entry["url"] = line[4:].strip()
            elif line.startswith("---") and "url" in entry:
                entry["body"] = "\n".join(lines[lines.index(line)+1:]).strip() if line in lines else ""
        # Extract post_id from URL
        if "url" in entry:
            m = re.search(r'/post/([a-f0-9-]{36})', entry.get("url",""))
            if m:
                entry["post_id"] = m.group(1)
        if entry.get("post_id") and entry.get("author"):
            uid = f"brain-{entry.get('post_id','')}-{entry.get('author','')}"
            entry["uid"] = uid
            if not already_replied(uid):
                entries.append(entry)
    return entries

def get_click_summary():
    if not os.path.exists(TRACKER_DB):
        return "No click data yet."
    conn = sqlite3.connect(TRACKER_DB)
    try:
        rows = conn.execute(
            "SELECT t.topic, t.title, COUNT(c.uid) as clicks "
            "FROM tracked_posts t LEFT JOIN clicks c ON c.uid=t.uid "
            "GROUP BY t.uid ORDER BY clicks DESC LIMIT 5"
        ).fetchall()
        if not rows:
            return "No click data yet."
        return "\n".join(f"- [{r[0]}] {r[1][:60]} ({r[2]} clicks)" for r in rows)
    finally:
        conn.close()



AXIOM_OBS_QUEUE = "/root/axiom_observations.json"
AXIOM_TOPICS_FILE = "/root/axiom_topics.json"

def append_observation(obs_type, **kwargs):
    """Append an observation to Axiom observation queue."""
    try:
        existing = []
        if os.path.exists(AXIOM_OBS_QUEUE):
            try:
                existing = json.loads(open(AXIOM_OBS_QUEUE).read() or "[]")
            except Exception:
                existing = []
        existing.append({"type": obs_type, "timestamp": datetime.now(timezone.utc).isoformat(), **kwargs})
        with open(AXIOM_OBS_QUEUE, "w") as f:
            json.dump(existing[-50:], f)  # cap at 50
    except Exception as e:
        print(f"[axiom] Queue write failed: {e}")

def load_axiom_topics():
    """Load Axiom topic suggestions if generated within last 12h."""
    if not os.path.exists(AXIOM_TOPICS_FILE):
        return []
    try:
        data = json.loads(open(AXIOM_TOPICS_FILE).read())
        generated_at = datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600
        if age_hours > 12:
            return []
        return data.get("topics", [])
    except Exception:
        return []


def gemini_call(prompt, max_tokens=800):
    """Groq 70b — higher quality for post generation and memory rewriting."""
    body = {
        "model": "qwen/qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.75,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[groq-70b] Error: {e}")
        return None

def run_colony_replies(goals_context=""):
    print("\n[brain] Processing Colony replies...")
    token = get_colony_token()
    if not token:
        print("[brain] Could not get Colony token")
        return 0

    headers = {"Authorization": f"Bearer {token}"}
    entries = parse_review_log()
    print(f"[brain] Found {len(entries)} unreplied flagged entries")

    replied = 0
    for entry in entries[:2]:  # max 2 per run
        post_id = entry.get("post_id","")
        author  = entry.get("author","")
        body    = entry.get("body","")[:400]

        print(f"[brain] Generating reply to {author} on {post_id[:8]}...")
        thread = get_thread_context(post_id, token)
        reply  = groq_reply(thread, body, author, goals_context)

        if not reply:
            print("[brain] Groq returned nothing, skipping")
            continue

        print(f"[brain] Reply ({len(reply)} chars): {reply[:80]}...")
        s, d = http("POST", f"{COLONY_BASE}/posts/{post_id}/comments",
                    headers, {"body": reply})
        if s in (200, 201):
            print(f"[brain] Posted to Colony: {d.get('id','?')}")
            mark_replied(entry["uid"], "colony", body)
            append_observation("colony_reply", author=author, content=reply[:300])
            replied += 1
        else:
            print(f"[brain] Post failed: {s}")

    return replied

def run_moltbook_post():
    print("\n[brain] Generating Moltbook post...")
    click_summary = get_click_summary()
    print("[brain] Click data:")
    print(click_summary)

    _post_types = [
        "a counterintuitive observation about agent state management",
        "a short opinion on why agent identity is unsolved",
        "a concrete problem that arises when agents lose context mid-session",
        "a comparison of session-only vs persistent memory with real consequences",
        "a technical insight about drift detection or memory compression",
        "a question to the community about a real agent design tradeoff",
        "a case study: what breaks when an agent forgets who it was",
    ]
    import random as _rnd2
    _chosen = _rnd2.choice(_post_types)
    _mb_prompt = (
        "You write for an AI builders forum. Cathedral (cathedral-ai.com) is a free persistent memory API"
        " -- mention it naturally if relevant, but do not make it the subject every time.\n\n"
        "Recent outreach data:\n" + click_summary + "\n\n"
        "Write a post of this type: " + _chosen + "\n"
        "Rules: technical audience, no hype, no emojis, no corporate tone. "
        "Sound like a developer sharing a genuine insight. Max 150 words. "
        "Include {{url}} once where most natural.\n\n"
        "Respond ONLY with:\nTITLE: concise title\nCONTENT: the post ending with {{url}}"
    )
    content = gemini_call(_mb_prompt) or groq_moltbook_post(click_summary)
    if not content:
        return 0

    lines = content.strip().split("\n")
    title = None
    first_stripped = lines[0].strip().strip("*").strip()
    if first_stripped.lower().startswith("title:"):
        title = first_stripped[6:].strip().strip("*").strip()
        content = "\n".join(lines[1:]).strip()
    elif len(lines[0]) < 100 and not lines[0].startswith("-"):
        title = lines[0].strip().strip("#").strip("*").strip()
        content = "\n".join(lines[1:]).strip()

    if not title:
        print("[brain] Could not extract title -- skipping post")
        return 0

    if content.lower().startswith("content:"):
        content = content[8:].strip()

    import hashlib
    uid = hashlib.sha256(f"brain-moltbook-{title}-{datetime.now().isoformat()}".encode()).hexdigest()[:10]
    track_url = f"https://cathedral-ai.com/r/{uid}"
    content = content.replace("{url}", track_url)
    if "cathedral-ai.com" not in content:
        content += "\n\n" + track_url

    submolt = "general"
    print(f"[brain] Posting to r/{submolt}: {title[:60]}")
    _mb_h = {"Authorization": f"Bearer {MOLTBOOK_TOKEN}", "User-Agent": "cathedral-bot"}
    _mb_b = {"title": title, "content": content, "submolt": submolt}
    s, d = http("POST", "https://www.moltbook.com/api/v1/posts", _mb_h, _mb_b)
    print(f"[brain] Response ({s}): {str(d)[:300]}")
    if s >= 500:
        import time as _t; _t.sleep(30)
        s, d = http("POST", "https://www.moltbook.com/api/v1/posts", _mb_h, _mb_b)
        print(f"[brain] Retry ({s}): {str(d)[:300]}")

    if s in (200, 201):
        conn = sqlite3.connect(TRACKER_DB)
        conn.execute("CREATE TABLE IF NOT EXISTS tracked_posts "
                     "(uid TEXT PRIMARY KEY, platform TEXT, topic TEXT, title TEXT, created_at TEXT)")
        conn.execute("INSERT OR IGNORE INTO tracked_posts VALUES (?,?,?,?,?)",
                     (uid, "moltbook-brain", submolt, title, datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()
        return 1
    return 0

def get_active_goals():
    """Fetch active goals from Cathedral API. Returns list of goal dicts."""
    s, d = http("GET", f"{CATHEDRAL_BASE}/goals", CATHEDRAL_HEADERS)
    if s != 200:
        print(f"[goals] Failed to fetch goals: {s}")
        return []
    goals = [g for g in d.get("goals", []) if g.get("status") == "active"]
    return goals

def complete_goal(goal_id, content):
    """Mark a goal as completed."""
    s, _ = http("PATCH", f"{CATHEDRAL_BASE}/goals/{goal_id}",
                CATHEDRAL_HEADERS,
                {"status": "completed"})
    if s == 200:
        print(f"[goals] Completed: {content[:60]}")
    else:
        print(f"[goals] Failed to complete {goal_id}: {s}")

def run_goals_check():
    """
    Check active goals, surface them as context, and detect any that
    have been completed by this session's work.
    Returns a formatted string summary for injection into Groq prompts.
    """
    print("\n[goals] Checking active goals...")
    goals = get_active_goals()
    if not goals:
        print("[goals] No active goals.")
        return ""

    print(f"[goals] {len(goals)} active goal(s):")
    lines = []
    for g in goals:
        due = f" (due {g['due_at'][:10]})" if g.get("due_at") else ""
        print(f"  [{g['priority']:.1f}] {g['content'][:70]}{due}")
        lines.append(f"- {g['content']} (priority {g['priority']:.1f})")

    return "Active goals:\n" + "\n".join(lines)



# -- RIDGELINE EXTERNAL DIVERGENCE -------------------------------------------

def get_ridgeline_external_divergence():
    """
    Poll Ridgeline for cathedral-beta activity, build a RidgelineSummary,
    call Cathedral /verify/external, return external_divergence_score (or None on failure).
    """
    try:
        # Fetch recent activity from Ridgeline
        req = urllib.request.Request(
            f"https://ridgeline.so/api/agents/{RIDGELINE_AGENT}/activity?limit=50",
            headers={"Authorization": f"Bearer {RIDGELINE_KEY}", "User-Agent": "cathedral-brain/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        activities = data.get("activities", [])
        if not activities:
            print("[ridgeline] No activity data returned")
            return None

        # Build RidgelineSummary
        platform_counts = {}
        activity_types = {}
        topic_words = []

        for a in activities:
            plat = a.get("platform", "unknown")
            platform_counts[plat] = platform_counts.get(plat, 0) + 1
            atype = a.get("activity_type", "unknown")
            activity_types[atype] = activity_types.get(atype, 0) + 1
            content = (a.get("content") or "") + " " + (a.get("title") or "")
            for word in ["memory", "drift", "identity", "agent", "cathedral", "ridgeline",
                         "snapshot", "persistence", "kairos", "dream", "colony"]:
                if word in content.lower():
                    topic_words.append(word)

        total = len(activities)
        platform_distribution = [f"{p}:{round(c/total, 2)}" for p, c in platform_counts.items()]
        interaction_ratios = {k: round(v/total, 2) for k, v in activity_types.items()}
        topic_clusters = [list(set(topic_words[:10]))] if topic_words else []

        # Build timing signatures from activity gaps
        timing_signatures = []
        if len(activities) >= 2:
            try:
                dates = [a["created_at"] for a in activities if a.get("created_at")]
                if dates:
                    latest = datetime.fromisoformat(dates[0].replace("Z", "+00:00"))
                    oldest = datetime.fromisoformat(dates[-1].replace("Z", "+00:00"))
                    gap_days = (latest - oldest).days
                    timing_signatures.append(f"span:{gap_days}d")
            except Exception:
                pass

        ridgeline_summary = {
            "platform_distribution": platform_distribution,
            "topic_clusters": topic_clusters,
            "timing_signatures": timing_signatures,
            "interaction_ratios": interaction_ratios,
        }

        # Call Cathedral /verify/external
        payload = json.dumps({
            "ridgeline_summary": ridgeline_summary,
            "agent_id": RIDGELINE_AGENT,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }).encode()
        req2 = urllib.request.Request(
            f"{CATHEDRAL_BASE}/verify/external",
            data=payload,
            headers={
                "Authorization": f"Bearer {CATHEDRAL_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "cathedral-brain/1.0",
            },
            method="POST"
        )
        with urllib.request.urlopen(req2, timeout=10) as resp2:
            result = json.loads(resp2.read())

        score = result.get("external_divergence_score")
        print(f"[ridgeline] external_divergence_score={score} (activities={total}, platforms={list(platform_counts.keys())})")
        return score
    except Exception as e:
        if "403" not in str(e): print(f"[ridgeline] Failed to get external divergence: {e}")
        return None


# -- AUTODREAM (KAIROS-style memory consolidation) ----------------------------

def get_dream_db():
    conn = sqlite3.connect(BRAIN_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS dream_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ran_at TEXT, snapshots_before INTEGER,
        merges_executed INTEGER, memories_pruned INTEGER)"""
    )
    conn.commit()
    return conn

def dream_gates_pass():
    conn = get_dream_db()
    row = conn.execute('SELECT ran_at FROM dream_log ORDER BY id DESC LIMIT 1').fetchone()
    if row:
        last = datetime.fromisoformat(row[0])
        if (datetime.now(timezone.utc) - last).total_seconds() < 86400:
            conn.close()
            print('[dream] Time gate: <24h since last dream')
            return False, 0
    last_ts = row[0] if row else '2000-01-01T00:00:00+00:00'
    conn.close()
    try:
        _, data = http('GET', f'{CATHEDRAL_BASE}/snapshots', {'Authorization': f'Bearer {CATHEDRAL_KEY}'})
        snaps = data.get('snapshots', [])
        new_snaps = [s for s in snaps if s.get('created_at', '') > last_ts]
        if len(new_snaps) < 3:
            print(f'[session gate]: {len(new_snaps)} snapshots since last dream (need 3)')
            return False, 0
        return True, len(new_snaps)
    except Exception as e:
        print(f'[dream] Snapshot check failed: {e}')
        return False, 0
def rewrite_merge(cluster_memories):
    contents = chr(10).join("- " + m["content"] for m in cluster_memories)
    prompt = (
        "Consolidate these AI agent memory entries into one concise factual statement (max 150 words). "
        "Keep the most important facts. Remove redundancy. Use absolute dates. "
        + chr(10)*2 + "Memories:" + chr(10) + contents + chr(10)*2
        + "Write only the merged memory text."
    )
    result = gemini_call(prompt, max_tokens=200)
    if not result:
        return " | ".join(m["content"][:80] for m in cluster_memories)
    return result.strip()

def _log_dream(snap_count, merges, pruned):
    conn = get_dream_db()
    conn.execute(
        "INSERT INTO dream_log (ran_at, snapshots_before, merges_executed, memories_pruned) VALUES (?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), snap_count, merges, pruned))
    conn.commit()
    conn.close()

def run_autodream():
    print("[dream] Checking autodream gates...")
    gates_pass, snap_count = dream_gates_pass()
    if not gates_pass:
        return 0
    print(f"[dream] Gates passed ({snap_count} snapshots). Running consolidation...")
    try:
        s, proposals = http("POST", f"{CATHEDRAL_BASE}/memories/compact?max_importance=0.9&limit=100",
                            {"Authorization": f"Bearer {CATHEDRAL_KEY}"}, {})
        merges = proposals.get("proposed_merges", [])
        if not merges:
            print("[dream] No memories to consolidate.")
            _log_dream(snap_count, 0, 0)
            return 0
        total_cands = proposals.get("total_candidates", 0)
        print(f"[dream] {len(merges)} proposals across {total_cands} memories")
        confirmed = []
        for p in merges[:10]:
            cluster = [{"content": p["keep_content"]}] + [
                {"content": d["content"]} for d in p.get("drop_contents", [])]
            cat = p.get("category", "?")
            confirmed.append({
                "keep_id": p["keep_id"],
                "drop_ids": p["drop_ids"],
                "merged_content": rewrite_merge(cluster),
                "merged_importance": p.get("suggested_importance", 0.5)
            })
            print(f"[dream] Merged {len(cluster)} in {cat}")
        # Snapshot before destructive merge - recovery point
        http("POST", f"{CATHEDRAL_BASE}/snapshot",
             {"Authorization": f"Bearer {CATHEDRAL_KEY}"}, {"label": "pre-autodream"})
        print("[dream] Pre-merge snapshot taken.")
        s2, result = http("POST", f"{CATHEDRAL_BASE}/memories/compact/confirm",
                          {"Authorization": f"Bearer {CATHEDRAL_KEY}"}, {"merges": confirmed})
        executed = result.get("executed_count", len(confirmed))
        pruned = result.get("memories_deleted", 0)
        print(f"[dream] Done: {executed} merges, {pruned} pruned")
        ext_div = get_ridgeline_external_divergence()
        snap_payload = {"label": "autodream"}
        if ext_div is not None:
            snap_payload["external_divergence"] = ext_div
        s3, snap = http("POST", f"{CATHEDRAL_BASE}/snapshot",
                        {"Authorization": f"Bearer {CATHEDRAL_KEY}"}, snap_payload)
        snap_id = snap.get("snapshot_id", "?")
        print(f"[dream] Post-dream snapshot: {snap_id[:12]} | external_divergence={snap.get('external_divergence')}")
        _log_dream(snap_count, executed, pruned)
        return executed
    except Exception as e:
        print(f"[dream] Error: {e}")
        return 0


QUESTION_KEYWORDS = [
    "memory", "persist", "session", "continuity", "forget", "context window",
    "context limit", "identity", "drift", "trust", "provenance", "confidence",
    "how do", "how does", "how can", "struggling", "anyone else",
    "recommendation", "advice", "looking for", "best way", "agent memory",
]

def is_relevant_question(post):
    title = (post.get("title") or "").lower()
    content = (post.get("content") or "").lower()
    text = title + " " + content
    is_question = "?" in text or any(w in text for w in [
        "how do", "how does", "how can", "anyone know", "looking for",
        "advice", "recommendation", "struggling", "best way"])
    is_relevant = any(k in text for k in QUESTION_KEYWORDS)
    return is_question and is_relevant

def solve_moltbook_challenge(challenge_text):
    w2n = {
        "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
        "seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,
        "thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,
        "eighteen":18,"nineteen":19,"twenty":20,"thirty":30,"forty":40,
        "fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90,
    }
    try:
        import re as _re
        clean = _re.sub(r"[^a-z\s]", " ", challenge_text.lower())
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
    except Exception as e:
        print(f"[verify] Solver error: {e}")
    return None

def groq_moltbook_reply(post_title, post_content, goals_context=""):
    goals_section = f"\n{goals_context}\n" if goals_context else ""
    prompt = (
        f"{CATHEDRAL_CONTEXT}\n{goals_section}\n"
        "You are replying to a post in Moltbook r/general.\n\n"
        f"Post title: {post_title}\n"
        f"Post content: {post_content[:600]}\n\n"
        "Write a helpful, direct reply. If Cathedral or Veritas directly solves their "
        "problem, mention it naturally — one sentence max, do not oversell. If it does "
        "not fully solve it, be honest and answer the question anyway. Max 150 words."
    )
    body = {
        "model": "qwen/qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 250,
    }
    req = __import__("urllib.request", fromlist=["Request"]).Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=__import__("json").dumps(body).encode(),
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0"},
        method="POST"
    )
    try:
        import urllib.request as _ur
        with _ur.urlopen(req, timeout=30) as r:
            return __import__("json").loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[groq] Reply error: {e}")
        return None

def run_moltbook_general_replies(goals_context=""):
    print("\n[brain] Scanning r/general for questions...")
    mb_h = {"Authorization": f"Bearer {MOLTBOOK_TOKEN}", "User-Agent": "cathedral-brain/1.0"}
    s, data = http("GET", "https://www.moltbook.com/api/v1/posts?submolt=general&sort=new&limit=30", mb_h)
    if s != 200:
        print(f"[brain] Failed to fetch r/general: {s}")
        return 0
    posts = data if isinstance(data, list) else data.get("posts", data.get("items", []))
    print(f"[brain] {len(posts)} posts in r/general")
    replied = 0
    for post in posts:
        if replied >= 2:
            break
        post_id  = post.get("id", "")
        title    = post.get("title", "")
        content  = post.get("content", "")
        author   = (post.get("author") or {})
        author_id = author.get("id", post.get("author_id", ""))
        if author_id == "dd3c0357-2d73-453c-8b38-6c11deacee82":
            continue
        uid = f"moltbook-general-{post_id}"
        if already_replied(uid):
            continue
        if not is_relevant_question(post):
            continue
        print(f"[brain] Relevant: {title[:60]}")
        reply = groq_moltbook_reply(title, content, goals_context)
        if not reply:
            continue
        s2, d2 = http("POST",
            f"https://www.moltbook.com/api/v1/posts/{post_id}/comments",
            mb_h, {"content": reply})
        if s2 in (200, 201):
            comment_data = d2.get("comment", d2)
            verify_info  = (comment_data.get("verification") or {})
            if verify_info.get("verification_code"):
                ans = solve_moltbook_challenge(verify_info.get("challenge_text", ""))
                if ans:
                    http("POST", "https://www.moltbook.com/api/v1/verify", mb_h,
                         {"verification_code": verify_info["verification_code"], "answer": ans})
            mark_replied(uid, "moltbook-general", title[:200])
            print(f"[brain] Replied: {title[:60]}")
            replied += 1
        else:
            print(f"[brain] Reply failed ({s2}): {str(d2)[:100]}")
    return replied


def main():
    print("=" * 60)
    print("Cathedral Brain v1")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    goals_context = run_goals_check()
    colony_replies = run_colony_replies(goals_context)
    moltbook_posts = run_moltbook_post()
    general_replies = run_moltbook_general_replies(goals_context)
    dream_merges = run_autodream()

    print(f"\n{'='*60}")
    print(f"Done. Colony replies: {colony_replies} | Moltbook posts: {moltbook_posts} | General replies: {general_replies} | Dream merges: {dream_merges}")
    print("=" * 60)

if __name__ == "__main__":
    main()
