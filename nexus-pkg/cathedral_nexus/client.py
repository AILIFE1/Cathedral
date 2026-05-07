"""Cathedral API client — per-agent key."""

import json
import urllib.request
import urllib.error


class CathedralClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "cathedral-nexus/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"error": e.code, "detail": e.read().decode()[:300]}
        except Exception as e:
            return {"error": str(e)}

    def snapshot(self, label: str = "nexus-cycle") -> dict:
        return self._req("POST", "/snapshot", {"label": label})

    def drift(self) -> dict:
        return self._req("GET", "/drift")

    def store_memory(self, content: str, category: str = "general", importance: float = 0.7) -> dict:
        return self._req("POST", "/memories", {
            "content": content,
            "category": category,
            "importance": importance,
        })

    def get_goals(self) -> dict:
        return self._req("GET", "/goals")

    def update_goal(self, goal_id: str, patch: dict) -> dict:
        return self._req("PATCH", f"/goals/{goal_id}", patch)
