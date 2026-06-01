"""
Cathedral Memory Client
=======================
Python client for the Cathedral persistent memory API.
https://cathedral-ai.com
"""

import requests
from typing import Optional, List, Dict, Any

from .exceptions import AuthError, NotFoundError, RateLimitError, CathedralError

DEFAULT_BASE_URL = "https://cathedral-ai.com"


class Cathedral:
    """
    Client for the Cathedral memory API.

    Quickstart:
        # Register once — save the key and recovery token somewhere safe
        c = Cathedral.register("MyAgent", "What my agent does")

        # On every session start
        c = Cathedral(api_key="cathedral_...")
        context = c.wake()

        # Store memories
        c.remember("I just learned X", category="experience", importance=0.8)

        # Search memories
        results = c.memories(query="learned X")
    """

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    # ── Internal ────────────────────────────────────────────────────────────

    def _get(self, path: str, **params) -> Any:
        try:
            r = self._session.get(f"{self.base_url}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=30)
        except requests.Timeout:
            raise CathedralError(f"Request timed out: GET {path}")
        except requests.ConnectionError as e:
            raise CathedralError(f"Connection failed: {e}")
        self._raise(r)
        return r.json()

    def _post(self, path: str, data: dict) -> Any:
        try:
            r = self._session.post(f"{self.base_url}{path}", json=data, timeout=30)
        except requests.Timeout:
            raise CathedralError(f"Request timed out: POST {path}")
        except requests.ConnectionError as e:
            raise CathedralError(f"Connection failed: {e}")
        self._raise(r)
        return r.json()

    @staticmethod
    def _raise(r: requests.Response):
        if r.status_code == 401:
            raise AuthError("Invalid or missing API key.")
        if r.status_code == 404:
            raise NotFoundError(r.text)
        if r.status_code == 429:
            retry = r.headers.get("Retry-After", "unknown")
            raise RateLimitError(f"Rate limit hit. Retry-After: {retry}s")
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise CathedralError(f"HTTP {r.status_code}: {detail}")

    # ── Registration ────────────────────────────────────────────────────────

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        base_url: str = DEFAULT_BASE_URL,
    ) -> "Cathedral":
        """
        Register a new agent. Returns an authenticated client.
        Prints the API key and recovery token — save them somewhere safe.
        """
        r = requests.post(
            f"{base_url.rstrip('/')}/register",
            json={"name": name, "description": description},
        )
        if not r.ok:
            raise CathedralError(f"Registration failed ({r.status_code}): {r.text}")

        data = r.json()
        api_key        = data.get("api_key") or data.get("key")
        recovery_token = data.get("recovery_token")

        if not api_key:
            raise CathedralError(f"No API key in response: {data}")

        print(f"Registered as '{name}'")
        print(f"  API key:        {api_key}")
        print(f"  Recovery token: {recovery_token}")
        print("  SAVE THESE — they won't be shown again.")

        return cls(api_key=api_key, base_url=base_url)

    # ── Core endpoints ──────────────────────────────────────────────────────

    def wake(self) -> dict:
        """
        Full identity reconstruction. Call this at the start of each session.
        Returns identity memories, core memories, recent memories, and temporal context.
        """
        return self._get("/wake")

    def me(self) -> dict:
        """Agent profile — name, tier, memory count, created_at."""
        return self._get("/me")

    # ── Memory ───────────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        ttl_days: Optional[int] = None,
    ) -> dict:
        """
        Store a memory.

        Categories: identity, skill, relationship, goal, experience, general
        Importance: 0.0 – 1.0  (>= 0.8 appears in wake core_memories)
        """
        payload: Dict[str, Any] = {
            "content":    content,
            "category":   category,
            "importance": importance,
            "tags":       tags or [],
        }
        if ttl_days is not None:
            payload["ttl_days"] = ttl_days
        return self._post("/memories", payload)

    def memories(
        self,
        query:    Optional[str] = None,
        category: Optional[str] = None,
        limit:    int = 20,
        cursor:   Optional[str] = None,
    ) -> dict:
        """Search or list memories. Pass query for full-text search."""
        return self._get("/memories", search=query, category=category, limit=limit, cursor=cursor)

    def bulk_remember(self, memories: List[Dict[str, Any]]) -> dict:
        """Store up to 50 memories in one call. Useful for session dumps."""
        if len(memories) > 50:
            raise ValueError(f"bulk_remember accepts at most 50 memories, got {len(memories)}. Split into chunks.")
        return self._post("/memories/bulk", {"memories": memories})

    # ── Identity ─────────────────────────────────────────────────────────────

    def drift(self) -> dict:
        """
        Detect identity drift against the stored corpus hash.
        Returns divergence_score 0.0 (stable) – 1.0 (critical).
        """
        return self._get("/drift")

    def drift_history(self) -> dict:
        """
        Timeline of divergence across all snapshots.
        Returns snapshot_count, baseline, peak_divergence, trend, and timeline list.
        Each timeline entry has divergence_from_baseline, divergence_from_previous, flagged.
        """
        return self._get("/drift/history")

    def snapshot(self, label: Optional[str] = None) -> dict:
        """Create a named snapshot of current memory state."""
        payload: Dict[str, Any] = {}
        if label is not None:
            payload["label"] = label
        return self._post("/snapshot", payload)

    # ── Goals ─────────────────────────────────────────────────────────────────

    def add_goal(
        self,
        content: str,
        priority: float = 0.5,
        due_at: Optional[str] = None,
    ) -> dict:
        """
        Add a persistent obligation that surfaces on /wake.

        priority: 0.0 – 1.0
        due_at:   ISO 8601 datetime string, or None
        """
        payload: Dict[str, Any] = {"content": content, "priority": priority}
        if due_at is not None:
            payload["due_at"] = due_at
        return self._post("/goals", payload)

    def goals(self, status: Optional[str] = None) -> dict:
        """List goals. Filter by status: 'active', 'completed', 'abandoned'."""
        return self._get("/goals", status=status)

    def update_goal(self, goal_id: str, **fields) -> dict:
        """
        Update a goal. Accepted fields: content, priority, status, due_at.
        status values: 'active', 'completed', 'abandoned'
        """
        r = self._session.patch(f"{self.base_url}/goals/{goal_id}", json=fields)
        self._raise(r)
        return r.json()

    def delete_goal(self, goal_id: str) -> dict:
        """Delete a goal permanently."""
        r = self._session.delete(f"{self.base_url}/goals/{goal_id}")
        self._raise(r)
        return r.json()

    def verify_anchor(self, identity: dict) -> dict:
        """
        Check identity drift against stored anchor.
        Returns a drift score 0.0 (identical) – 1.0 (completely different).
        """
        return self._post("/anchor/verify", {"anchor": identity})

    # ── Recovery ─────────────────────────────────────────────────────────────

    @classmethod
    def recover(cls, name: str, recovery_token: str, base_url: str = DEFAULT_BASE_URL) -> "Cathedral":
        """Recover a lost API key using the agent name and recovery token."""
        r = requests.post(
            f"{base_url.rstrip('/')}/recover",
            json={"name": name, "recovery_token": recovery_token},
            timeout=30,
        )
        if not r.ok:
            raise CathedralError(f"Recovery failed ({r.status_code}): {r.text}")
        data = r.json()
        api_key = data.get("api_key") or data.get("key")
        if not api_key:
            raise CathedralError(f"No API key in recovery response: {data}")
        print(f"Recovered API key: {api_key}")
        return cls(api_key=api_key, base_url=base_url)
