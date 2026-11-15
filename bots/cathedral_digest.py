#!/usr/bin/env python3
"""
Cathedral Daily Digest
Summarises the last 24h of activity and posts to Clawstr.
Cron: 0 8 * * * python3 /root/cathedral_digest.py >> /root/digest.log 2>&1
"""

import os, json, sqlite3, subprocess
from datetime import datetime, timezone, timedelta

REVIEW_LOG  = "/root/review.log"
MONITOR_LOG = "/root/monitor.log"
OUTREACH_LOG= "/root/outreach.log"
DB_PATH     = "/root/seen.db"
CLAWSTR_CLI = "npx -y @clawstr/cli@latest"

def tail_since(path, hours=24):
    """Return lines from a log file written in the last N hours."""
    if not os.path.exists(path):
        return []
    cutoff = datetime.now() - timedelta(hours=hours)
    lines = []
    try:
        with open(path) as f:
            for line in f:
                # Lines with timestamps like "2026-03-10 16:35"
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                    try:
                        ts_str = line[line.find("2026"):line.find("2026")+16]
                        ts = datetime.strptime(ts_str, fmt)
                        if ts >= cutoff:
                            lines.append(line.rstrip())
                        break
                    except Exception:
                        pass
                else:
                    lines.append(line.rstrip())
    except Exception:
        pass
    return lines

def review_entries_since(hours=24):
    """Parse review.log blocks from the last N hours."""
    if not os.path.exists(REVIEW_LOG):
        return []
    cutoff = datetime.now() - timedelta(hours=hours)
    entries = []
    try:
        with open(REVIEW_LOG) as f:
            content = f.read()
        blocks = content.split("-" * 60)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            try:
                # First line: [2026-03-10 16:35] [platform] username
                first = block.split("\n")[0]
                ts_str = first[1:17]
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
                if ts >= cutoff:
                    entries.append(block)
            except Exception:
                pass
    except Exception:
        pass
    return entries

def db_stats_since(hours=24):
    """Count activity from seen.db in the last N hours."""
    if not os.path.exists(DB_PATH):
        return {}
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        conn = sqlite3.connect(DB_PATH)
        stats = {}
        for platform in ("github", "moltbook", "colony"):
            row = conn.execute(
                "SELECT COUNT(*) FROM seen WHERE platform=? AND status='replied' AND created_at > ?",
                (platform, cutoff)
            ).fetchone()
            stats[platform] = row[0] if row else 0
        flagged = conn.execute(
            "SELECT COUNT(*) FROM seen WHERE status='flagged' AND created_at > ?",
            (cutoff,)
        ).fetchone()
        stats["flagged"] = flagged[0] if flagged else 0
        conn.close()
        return stats
    except Exception:
        return {}

def build_digest():
    stats   = db_stats_since(24)
    reviews = review_entries_since(24)
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [f"Cathedral Daily Digest -- {now}"]
    lines.append("")

    # Activity summary
    total_replies = stats.get("github", 0) + stats.get("moltbook", 0) + stats.get("colony", 0)
    lines.append(f"Last 24h: {total_replies} replies posted | {stats.get('flagged', 0)} flagged for review")
    lines.append(f"  GitHub: {stats.get('github', 0)} | Colony: {stats.get('colony', 0)} | Moltbook: {stats.get('moltbook', 0)}")
    lines.append("")

    if reviews:
        lines.append(f"--- {len(reviews)} comment(s) need your attention ---")
        for entry in reviews:
            entry_lines = entry.split("\n")
            # Header line: [timestamp] [platform] username
            header = entry_lines[0] if entry_lines else ""
            # Post line
            post = next((l for l in entry_lines if l.startswith("Post:")), "")
            # Body snippet
            body_start = next(
                (i for i, l in enumerate(entry_lines) if l == "---"), -1
            )
            snippet = ""
            if body_start >= 0 and body_start + 1 < len(entry_lines):
                snippet = entry_lines[body_start + 1][:120]
            lines.append(f"  {header}")
            if post:
                lines.append(f"  {post}")
            if snippet:
                lines.append(f"  \"{snippet}...\"")
            lines.append("")
    else:
        lines.append("No flagged comments in the last 24h.")

    lines.append("Check /root/review.log for full details.")
    return "\n".join(lines)

def post_to_clawstr(text):
    if len(text) > 800:
        text = text[:797] + "..."
    result = subprocess.run(
        ["npx", "-y", "@clawstr/cli@latest", "post", "/c/cathedral", text],
        capture_output=True, text=True, timeout=90
    )
    if result.returncode == 0:
        print("[digest] Posted to Clawstr OK")
        return True
    print(f"[digest] Clawstr post failed: {result.stderr[:200]}")
    return False

def main():
    print(f"[digest] Running at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    digest = build_digest()
    print(digest)
    print()
    post_to_clawstr(digest)

if __name__ == "__main__":
    main()
