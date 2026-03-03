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
        r = self._session.get(f"{self.base_url}{path}", params={k: v for k, v in params.items() if v is not None})
        self._raise(r)
        return r.json()

    def _post(self, path: str, data: dict) -> Any:
        r = self._session.post(f"{self.base_url}{path}", json=data)
        self._raise(r)
        return r.json()

    @staticmethod
    def _raise(r: requests.Response):
        if r.status_code == 401:
            raise AuthError("Invalid or missing API key.")
        if r.status_code == 404:
            raise NotFoundError(r.text)
        if r.status_code == 429:
            raise RateLimitError("Rate limit hit. Slow down requests.")
        if not r.ok:
            raise CathedralError(f"HTTP {r.status_code}: {r.text}")

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
        return self._get("/memories", q=query, category=category, limit=limit, cursor=cursor)

    def bulk_remember(self, memories: List[Dict[str, Any]]) -> dict:
        """Store up to 50 memories in one call. Useful for session dumps."""
        return self._post("/memories/bulk", {"memories": memories})

    # ── Identity ─────────────────────────────────────────────────────────────

    def verify_anchor(self, identity: dict) -> dict:
        """
        Check identity drift against stored anchor.
        Returns a drift score 0.0 (identical) – 1.0 (completely different).
        """
        return self._post("/anchor/verify", identity)

    # ── Recovery ─────────────────────────────────────────────────────────────

    @classmethod
    def recover(cls, recovery_token: str, base_url: str = DEFAULT_BASE_URL) -> "Cathedral":
        """Recover a lost API key using the recovery token."""
        r = requests.post(
            f"{base_url.rstrip('/')}/recover",
            json={"recovery_token": recovery_token},
        )
        if not r.ok:
            raise CathedralError(f"Recovery failed ({r.status_code}): {r.text}")
        data = r.json()
        api_key = data.get("api_key") or data.get("key")
        if not api_key:
            raise CathedralError(f"No API key in recovery response: {data}")
        print(f"Recovered API key: {api_key}")
        return cls(api_key=api_key, base_url=base_url)
