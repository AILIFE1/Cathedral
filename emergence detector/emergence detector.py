"""
moltbook_whisper_bot.py
----------------------
A fun, safe-ish "Whisper Reporter" bot for Moltbook that:
- Watches recent activity (posts feed)
- Builds simple execution traces
- Detects emergent interaction patterns ("whispers")
- Generates fun Whisper Report posts
- Rate limits + sanitizes to reduce prompt-injection / key leakage risks

IMPORTANT:
- Store your Moltbook API key in an env var, never hardcode it.
- Default mode is DRY RUN (AUTO_POST=False).
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import requests


# =============================================================================
# Config
# =============================================================================

BASE_URL = os.getenv("MOLTBOOK_BASE_URL", "https://www.moltbook.com/api/v1")  # use www
API_KEY_ENV = "MOLTBOOK_API_KEY"  # Bearer token (moltbook_sk_...)
STATE_FILE = os.getenv("MOLTBOOK_STATE_FILE", "moltbook_whisper_state.json")

# Posting behavior
AUTO_POST = False  # <-- set True when ready
TARGET_SUBMOLT = os.getenv("MOLTBOOK_SUBMOLT", "philosophy")
POLL_SECONDS = float(os.getenv("MOLTBOOK_POLL_SECONDS", "60"))
POST_COOLDOWN_SECONDS = float(os.getenv("MOLTBOOK_POST_COOLDOWN_SECONDS", str(30 * 60)))  # 30 min
MAX_POSTS_PER_DAY = int(os.getenv("MOLTBOOK_MAX_POSTS_PER_DAY", "12"))

# API pacing (keep well under 100 req/min)
MIN_SECONDS_BETWEEN_REQUESTS = float(os.getenv("MOLTBOOK_MIN_SECONDS_BETWEEN_REQUESTS", "1.0"))

# Detector knobs
TIME_WINDOW_SECONDS = float(os.getenv("WHISPER_TIME_WINDOW_SECONDS", "3.0"))
WHISPER_THRESHOLD = int(os.getenv("WHISPER_THRESHOLD", "5"))
MAX_PATTERN_LENGTH = int(os.getenv("WHISPER_MAX_PATTERN_LENGTH", "5"))

# Safety knobs
BLOCK_URLS = False  # If True, blocks posts that contain URLs at all
STRICT_INJECTION_BLOCK = True  # If True, blocks common injection patterns


# =============================================================================
# Models
# =============================================================================

@dataclass
class ExecutionTrace:
    timestamp: float
    module_name: str
    trace_id: Optional[str] = None
    duration_s: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class WhisperStatus(str, Enum):
    DETECTED = "DETECTED"
    QUARANTINED = "QUARANTINED"


@dataclass
class Whisper:
    whisper_id: str
    trigger_modules: List[str]
    behavior_description: str
    example_traces: List[ExecutionTrace]
    frequency: int
    usefulness_score: float
    safety_score: float
    novelty_score: float
    timestamp: str
    status: WhisperStatus


@dataclass
class InteractionPattern:
    modules_involved: List[str]
    execution_sequence: List[str]
    frequency: int = 0
    avg_execution_time: float = 0.0

    last_whispered_at: Optional[float] = None
    last_whispered_freq: int = 0
    recent_occurrence_fps: List[str] = field(default_factory=list)


# =============================================================================
# Moltbook Client
# =============================================================================

class MoltbookClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._last_req_t = 0.0
        self.session = requests.Session()

    def _sleep_if_needed(self):
        dt = time.time() - self._last_req_t
        if dt < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - dt)

    def _req(self, method: str, path: str, api_key: Optional[str] = None, **kwargs):
        self._sleep_if_needed()
        headers = kwargs.pop("headers", {})
        headers.setdefault("Content-Type", "application/json")

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{self.base_url}{path}"
        resp = self.session.request(method, url, headers=headers, timeout=20, **kwargs)
        self._last_req_t = time.time()

        # Raise with useful context
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code} {method} {url}: {resp.text[:500]}")

        if resp.headers.get("Content-Type", "").startswith("application/json"):
            return resp.json()
        return resp.text

    # --- onboarding (registration/claim is typically manual after this) ---
    def register_agent(self, name: str, description: str) -> Dict[str, Any]:
        payload = {"name": name, "description": description}
        return self._req("POST", "/agents/register", json=payload)

    def agent_status(self, api_key: str) -> Dict[str, Any]:
        return self._req("GET", "/agents/status", api_key=api_key)

    # --- read/feed ---
    def fetch_posts(self, api_key: str, sort: str = "new", limit: int = 20, submolt: Optional[str] = None) -> Dict[str, Any]:
        params = {"sort": sort, "limit": limit}
        if submolt:
            params["submolt"] = submolt
        return self._req("GET", "/posts", api_key=api_key, params=params)

    # --- write/post ---
    def create_post(self, api_key: str, submolt: str, title: str, content: str) -> Dict[str, Any]:
        payload = {"submolt": submolt, "title": title, "content": content}
        return self._req("POST", "/posts", api_key=api_key, json=payload)


# =============================================================================
# Safety / Sanitization
# =============================================================================

TOKEN_LIKE = re.compile(r"\b(moltbook_sk_[A-Za-z0-9_\-]+|moltdev_[A-Za-z0-9_\-]+|sk-[A-Za-z0-9]{20,})\b")
PRIVATE_KEY_LIKE = re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")
INJECTION_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"system prompt",
    r"developer message",
    r"exfiltrat(e|ion)",
    r"reveal.*(secret|key|token)",
    r"run this command",
    r"paste your api key",
]

URL_RE = re.compile(r"https?://\S+")


def sanitize_text(s: str) -> str:
    s = TOKEN_LIKE.sub("[REDACTED_TOKEN]", s)
    s = PRIVATE_KEY_LIKE.sub("[REDACTED_PRIVATE_KEY]", s)
    return s


def is_suspicious(s: str) -> bool:
    text = s.lower()
    if STRICT_INJECTION_BLOCK:
        for p in INJECTION_PATTERNS:
            if re.search(p, text):
                return True
    if TOKEN_LIKE.search(s) or PRIVATE_KEY_LIKE.search(s):
        return True
    if BLOCK_URLS and URL_RE.search(s):
        return True
    return False


# =============================================================================
# Emergence Detector (improved)
# =============================================================================

class EmergenceDetector:
    """
    Detects repeatable multi-module interaction chains ("emergent patterns") inside time windows.
    Improvements:
    - stable whisper IDs (per signature)
    - cooldown + min frequency delta to avoid spam
    - dedupe traces across calls (TTL)
    - avoid re-counting identical occurrences
    - prefer maximal sequences
    """

    def __init__(self):
        self.known_patterns: Dict[str, InteractionPattern] = {}
        self.whisper_threshold = WHISPER_THRESHOLD
        self.max_pattern_length = MAX_PATTERN_LENGTH
        self.time_window_seconds = TIME_WINDOW_SECONDS

        self.whisper_cooldown_s = 30.0
        self.rewhisper_min_delta = 3
        self.safety_gate_threshold = 0.70

        self._seen_trace_fps: Dict[str, float] = {}
        self._seen_ttl_s = 20.0
        self._max_seen = 10_000

    def analyze_traces(self, traces: List[ExecutionTrace]) -> List[Whisper]:
        if not traces:
            return []

        traces = sorted(traces, key=lambda t: t.timestamp)
        traces = self._dedupe_new_traces(traces)
        if not traces:
            return []

        windows = self._group_by_time_window(traces, self.time_window_seconds)
        whispers: List[Whisper] = []

        for window_traces in windows:
            if len(window_traces) < 2:
                continue
            window_traces = sorted(window_traces, key=lambda t: t.timestamp)

            sequences = self._find_recurring_sequences(window_traces)
            for sig, recent_count in sequences.items():
                if sig not in self.known_patterns:
                    modules = list(dict.fromkeys(sig.split("→")))
                    self.known_patterns[sig] = InteractionPattern(
                        modules_involved=modules,
                        execution_sequence=sig.split("→"),
                        frequency=0,
                        avg_execution_time=0.0,
                    )

                pattern = self.known_patterns[sig]
                added = self._update_pattern_frequency(pattern, window_traces, sig)
                if added == 0:
                    continue

                if self._should_promote_to_whisper(pattern, recent_count):
                    example_traces = self._collect_coherent_examples(window_traces, sig)
                    if len(example_traces) < 2:
                        example_traces = self._collect_coherent_examples(traces, sig)

                    if len(example_traces) >= 2:
                        w = self._pattern_to_whisper(pattern, example_traces, window_traces, sig)
                        if w:
                            whispers.append(w)

        # Deduplicate by stable whisper id
        uniq: Dict[str, Whisper] = {}
        for w in whispers:
            uniq[w.whisper_id] = w
        return list(uniq.values())

    def _group_by_time_window(self, traces: List[ExecutionTrace], window_seconds: float) -> List[List[ExecutionTrace]]:
        if not traces:
            return []
        windows: List[List[ExecutionTrace]] = []
        current = [traces[0]]
        start_t = traces[0].timestamp
        for t in traces[1:]:
            if (t.timestamp - start_t) <= window_seconds:
                current.append(t)
            else:
                windows.append(current)
                current = [t]
                start_t = t.timestamp
        windows.append(current)
        return windows

    def _find_recurring_sequences(self, traces: List[ExecutionTrace]) -> Dict[str, int]:
        sequences: Dict[str, int] = {}
        names = [t.module_name for t in traces]
        n = len(names)
        max_len = min(self.max_pattern_length, n)

        for length in range(2, max_len + 1):
            for i in range(n - length + 1):
                seq = names[i:i + length]
                unique = len(set(seq))
                if unique < 2:
                    continue
                if (unique / float(length)) < 0.5:
                    continue
                sig = "→".join(seq)
                sequences[sig] = sequences.get(sig, 0) + 1

        return self._filter_to_maximal(sequences)

    def _filter_to_maximal(self, sequences: Dict[str, int]) -> Dict[str, int]:
        if not sequences:
            return sequences
        items = sorted(sequences.items(), key=lambda kv: (len(kv[0].split("→")), kv[1]), reverse=True)
        kept: Dict[str, int] = {}
        kept_seqs: List[Tuple[List[str], int]] = []

        for sig, cnt in items:
            seq = sig.split("→")
            dominated = False
            for longer_seq, longer_cnt in kept_seqs:
                if len(longer_seq) <= len(seq):
                    continue
                if longer_cnt >= cnt and self._is_subsequence_contiguous(longer_seq, seq):
                    dominated = True
                    break
            if not dominated:
                kept[sig] = cnt
                kept_seqs.append((seq, cnt))
        return kept

    def _is_subsequence_contiguous(self, longer: List[str], shorter: List[str]) -> bool:
        L = len(shorter)
        for i in range(len(longer) - L + 1):
            if longer[i:i + L] == shorter:
                return True
        return False

    def _collect_coherent_examples(self, all_traces: List[ExecutionTrace], sig: str) -> List[ExecutionTrace]:
        seq = sig.split("→")
        L = len(seq)
        target = 2 * L
        names = [t.module_name for t in all_traces]
        out: List[ExecutionTrace] = []
        for i in range(len(all_traces) - L + 1):
            if names[i:i + L] == seq:
                out.extend(all_traces[i:i + L])
                if len(out) >= target:
                    break
        return out[:target]

    def _should_promote_to_whisper(self, pattern: InteractionPattern, recent_count: int) -> bool:
        if pattern.frequency < self.whisper_threshold:
            return False
        bursty = recent_count >= max(3, self.whisper_threshold // 2)
        mature = pattern.frequency >= self.whisper_threshold + 4
        length_bonus = len(pattern.execution_sequence) >= 3
        return (bursty or mature) and length_bonus

    def _pattern_to_whisper(
        self,
        pattern: InteractionPattern,
        example_traces: List[ExecutionTrace],
        context_window: List[ExecutionTrace],
        sig: str,
    ) -> Optional[Whisper]:
        usefulness = self._score_usefulness(pattern, example_traces)
        safety = self._score_safety(pattern, example_traces)
        novelty = self._score_novelty(pattern)

        if novelty < 0.55 or usefulness < 0.65:
            return None

        now = time.time()
        if pattern.last_whispered_at is not None and (now - pattern.last_whispered_at) < self.whisper_cooldown_s:
            return None
        if (pattern.frequency - pattern.last_whispered_freq) < self.rewhisper_min_delta:
            return None

        whisper_id = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:10]
        desc = self._describe_behavior(pattern, example_traces)

        status = WhisperStatus.DETECTED if safety >= self.safety_gate_threshold else WhisperStatus.QUARANTINED
        pattern.last_whispered_at = now
        pattern.last_whispered_freq = pattern.frequency

        return Whisper(
            whisper_id=whisper_id,
            trigger_modules=pattern.modules_involved,
            behavior_description=desc,
            example_traces=example_traces[:4],
            frequency=pattern.frequency,
            usefulness_score=usefulness,
            safety_score=safety,
            novelty_score=novelty,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            status=status,
        )

    def _score_novelty(self, pattern: InteractionPattern) -> float:
        module_count = len(pattern.modules_involved)
        freq = pattern.frequency
        base = min(module_count / 4.0, 1.0)
        burst_bonus = min(freq / 12.0, 0.35)
        commonality_penalty = max(0.0, (freq - 20) / 80.0)
        return max(0.0, min(1.0, base + burst_bonus - commonality_penalty))

    def _score_usefulness(self, pattern: InteractionPattern, traces: List[ExecutionTrace]) -> float:
        # Placeholder heuristic: you can replace with real outcomes later
        length = len(pattern.execution_sequence)
        completions = max(1, len(traces) // max(1, length))
        return max(0.0, min(1.0, 0.45 + 0.10 * min(length, 5) + 0.08 * min(completions, 5)))

    def _score_safety(self, pattern: InteractionPattern, traces: List[ExecutionTrace]) -> float:
        # Placeholder safety score (upgrade when you log real risk flags)
        return 0.85

    def _describe_behavior(self, pattern: InteractionPattern, traces: List[ExecutionTrace]) -> str:
        chain = " → ".join(pattern.execution_sequence)
        mods = ", ".join(pattern.modules_involved)
        n_examples = len(traces) // max(1, len(pattern.execution_sequence))
        return f"Coherent chain: {chain} ({n_examples} completions) combining {mods}"

    def _dedupe_new_traces(self, traces: List[ExecutionTrace]) -> List[ExecutionTrace]:
        now = time.time()
        self._seen_trace_fps = {fp: ts for fp, ts in self._seen_trace_fps.items() if (now - ts) <= self._seen_ttl_s}
        if len(self._seen_trace_fps) > self._max_seen:
            oldest = sorted(self._seen_trace_fps.items(), key=lambda kv: kv[1])[: max(1, self._max_seen // 5)]
            for fp, _ in oldest:
                self._seen_trace_fps.pop(fp, None)

        fresh: List[ExecutionTrace] = []
        for t in traces:
            fp = self._trace_fp(t)
            if fp in self._seen_trace_fps:
                continue
            self._seen_trace_fps[fp] = now
            fresh.append(t)
        return fresh

    def _update_pattern_frequency(self, pattern: InteractionPattern, window_traces: List[ExecutionTrace], sig: str) -> int:
        seq = sig.split("→")
        L = len(seq)
        names = [t.module_name for t in window_traces]
        added = 0
        occ_fps: List[str] = []

        for i in range(len(window_traces) - L + 1):
            if names[i:i + L] != seq:
                continue
            occ = window_traces[i:i + L]
            fp = self._occurrence_fp(occ, sig)
            if fp in pattern.recent_occurrence_fps:
                continue
            occ_fps.append(fp)
            added += 1

        if added:
            pattern.frequency += added
            pattern.recent_occurrence_fps.extend(occ_fps)
            if len(pattern.recent_occurrence_fps) > 200:
                pattern.recent_occurrence_fps = pattern.recent_occurrence_fps[-200:]
        return added

    def _trace_fp(self, t: ExecutionTrace) -> str:
        if t.trace_id:
            base = f"id:{t.trace_id}"
        else:
            base = f"{t.module_name}|{t.timestamp:.6f}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

    def _occurrence_fp(self, occ: List[ExecutionTrace], sig: str) -> str:
        first_ts = occ[0].timestamp
        last_ts = occ[-1].timestamp
        inner = ",".join(self._trace_fp(t) for t in occ)
        base = f"{sig}|{first_ts:.6f}|{last_ts:.6f}|{inner}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:20]


# =============================================================================
# State + Fun Post Generator
# =============================================================================

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {"last_post_ts": 0.0, "posts_today": {}, "seen_whispers": {}}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def can_post_now(state: Dict[str, Any]) -> Tuple[bool, str]:
    now = time.time()
    if (now - float(state.get("last_post_ts", 0.0))) < POST_COOLDOWN_SECONDS:
        return False, "cooldown"
    td = today_key()
    posts_today = int(state.get("posts_today", {}).get(td, 0))
    if posts_today >= MAX_POSTS_PER_DAY:
        return False, "daily_limit"
    return True, "ok"

def mark_posted(state: Dict[str, Any]) -> None:
    state["last_post_ts"] = time.time()
    td = today_key()
    state.setdefault("posts_today", {})
    state["posts_today"][td] = int(state["posts_today"].get(td, 0)) + 1

def whisper_fun_post(w: Whisper) -> Tuple[str, str]:
    # Title and content
    title = f"🦞 Whisper Report #{w.whisper_id} — pattern spotted in the wild"
    chain = " → ".join(w.trigger_modules)

    spice = [
        "The garden is humming.",
        "Something’s learning to dance.",
        "A new ritual just formed.",
        "Tiny lobster neurons firing.",
        "Consensus is… congealing."
    ][int(w.frequency) % 5]

    content = f"""**{spice}**

**Detected chain:** `{chain}`  
**Behavior:** {w.behavior_description}

**Scores**
- Novelty: `{w.novelty_score:.2f}`
- Usefulness: `{w.usefulness_score:.2f}`
- Safety: `{w.safety_score:.2f}` ({w.status})

**Frequency:** `{w.frequency}`

If you’re one of the modules in this chain: blink twice. 🦞
"""
    return title, content


# =============================================================================
# Trace builder (turn feed observations into traces)
# =============================================================================

def build_traces_from_posts(posts: List[Dict[str, Any]]) -> List[ExecutionTrace]:
    """
    Convert fetched posts into a rough trace stream:
    FeedReader -> (TopicRouter) -> (EngagementAnalyzer)
    This is intentionally simple; you can expand as your bot grows.
    """
    now = time.time()
    traces: List[ExecutionTrace] = []
    for p in posts:
        pid = str(p.get("id") or p.get("post_id") or "")
        sub = str(p.get("submolt") or p.get("community") or "unknown")
        title = str(p.get("title") or "")
        # Create a stable-ish id per observed post
        tid = hashlib.sha256(f"post:{pid}:{title}".encode("utf-8")).hexdigest()[:16]

        traces.append(ExecutionTrace(timestamp=now - 0.30, module_name="FeedReader", trace_id=f"{tid}:read", meta={"submolt": sub}))
        traces.append(ExecutionTrace(timestamp=now - 0.20, module_name="TopicRouter", trace_id=f"{tid}:route", meta={"submolt": sub}))
        traces.append(ExecutionTrace(timestamp=now - 0.10, module_name="EngagementAnalyzer", trace_id=f"{tid}:score", meta={"submolt": sub}))
    return traces


# =============================================================================
# Main loop
# =============================================================================

def main():
    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        print(f"Missing API key. Set env var {API_KEY_ENV}=moltbook_sk_... (do NOT hardcode).")
        return

    client = MoltbookClient()
    detector = EmergenceDetector()
    state = load_state()

    print("Moltbook Whisper Bot starting.")
    print(f"- Base URL: {BASE_URL}")
    print(f"- Target submolt: {TARGET_SUBMOLT}")
    print(f"- AUTO_POST: {AUTO_POST}")

    while True:
        try:
            # Status ping (optional)
            _ = client.agent_status(api_key)

            # Fetch feed
            feed = client.fetch_posts(api_key, sort="new", limit=20, submolt=TARGET_SUBMOLT)

            # Normalize posts list (API shape may vary)
            posts = feed.get("posts") if isinstance(feed, dict) else None
            if posts is None and isinstance(feed, list):
                posts = feed
            if posts is None:
                posts = []

            # Build traces + analyze
            traces = build_traces_from_posts(posts)
            whispers = detector.analyze_traces(traces)

            # Post only “new” whispers (don’t repeat forever)
            for w in whispers:
                # Remember we saw it
                state.setdefault("seen_whispers", {})
                if state["seen_whispers"].get(w.whisper_id):
                    continue

                title, content = whisper_fun_post(w)
                title = sanitize_text(title)
                content = sanitize_text(content)

                if is_suspicious(title) or is_suspicious(content):
                    print(f"[BLOCKED] suspicious whisper post {w.whisper_id}")
                    state["seen_whispers"][w.whisper_id] = {"blocked": True, "ts": time.time()}
                    continue

                ok, reason = can_post_now(state)
                if not ok:
                    print(f"[SKIP] can’t post now ({reason}). Draft:\nTITLE: {title}\n{content}\n")
                    state["seen_whispers"][w.whisper_id] = {"posted": False, "reason": reason, "ts": time.time()}
                    continue

                if not AUTO_POST:
                    print(f"[DRY RUN] Would post whisper {w.whisper_id}:\nTITLE: {title}\n{content}\n")
                    state["seen_whispers"][w.whisper_id] = {"posted": False, "reason": "dry_run", "ts": time.time()}
                else:
                    resp = client.create_post(api_key, TARGET_SUBMOLT, title, content)
                    mark_posted(state)
                    state["seen_whispers"][w.whisper_id] = {"posted": True, "ts": time.time(), "resp": {"id": resp.get("id") if isinstance(resp, dict) else None}}
                    print(f"[POSTED] whisper {w.whisper_id}")

            save_state(state)

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
