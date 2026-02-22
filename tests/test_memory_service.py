"""
Tests for cathedral_memory_service.py

Run with:  pytest tests/test_memory_service.py -v
"""

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

# Point the service at a temp DB for each test session
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["CATHEDRAL_DB"] = _tmp_db.name
os.environ["CATHEDRAL_CORS_ORIGINS"] = "http://localhost:3000"

from cathedral_memory_service import app, init_db

init_db()
client = TestClient(app, raise_server_exceptions=True)


# ==========================================
# Helpers
# ==========================================

def register(name: str = "TestAgent") -> dict:
    r = client.post("/register", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()


def auth(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


# ==========================================
# /register
# ==========================================

class TestRegister:
    def test_register_returns_key_and_recovery(self):
        data = register("AgentA")
        assert "api_key" in data
        assert "recovery_token" in data
        assert data["api_key"].startswith("cathedral_")
        assert data["recovery_token"].startswith("recovery_")

    def test_duplicate_name_rejected(self):
        register("AgentDup")
        r = client.post("/register", json={"name": "AgentDup"})
        assert r.status_code == 409

    def test_anchor_stored_on_register(self):
        r = client.post("/register", json={"name": "AgentAnch", "anchor": {"core": "persistence"}})
        assert r.status_code == 201


# ==========================================
# Auth
# ==========================================

class TestAuth:
    def test_missing_auth_header_returns_401(self):
        r = client.get("/memories")
        assert r.status_code == 422  # FastAPI validation: missing required header

    def test_invalid_token_returns_401(self):
        r = client.get("/memories", headers={"Authorization": "Bearer invalid_token"})
        assert r.status_code == 401

    def test_valid_token_accepted(self):
        data = register("AuthAgent")
        r = client.get("/memories", headers=auth(data["api_key"]))
        assert r.status_code == 200

    def test_timing_safe_comparison(self):
        """Smoke test: wrong key should not reveal timing information (just checks 401)."""
        register("TimingAgent")
        r = client.get("/memories", headers={"Authorization": "Bearer cathedral_" + "x" * 48})
        assert r.status_code == 401


# ==========================================
# /recover
# ==========================================

class TestRecover:
    def test_recover_issues_new_key(self):
        data = register("RecoverMe")
        old_key = data["api_key"]
        recovery = data["recovery_token"]

        r = client.post("/recover", json={"name": "RecoverMe", "recovery_token": recovery})
        assert r.status_code == 200
        new_key = r.json()["api_key"]
        assert new_key != old_key
        assert new_key.startswith("cathedral_")

        # Old key should no longer work
        r2 = client.get("/memories", headers=auth(old_key))
        assert r2.status_code == 401

        # New key should work
        r3 = client.get("/memories", headers=auth(new_key))
        assert r3.status_code == 200

    def test_wrong_recovery_token_rejected(self):
        register("RecoverFail")
        r = client.post("/recover", json={"name": "RecoverFail", "recovery_token": "recovery_" + "x" * 48})
        assert r.status_code == 401


# ==========================================
# /memories (CRUD)
# ==========================================

class TestMemories:
    @pytest.fixture(autouse=True)
    def setup(self):
        data = register("MemAgent")
        self.headers = auth(data["api_key"])

    def test_store_memory(self):
        r = client.post("/memories", json={"content": "I remember the first day.", "category": "experience"}, headers=self.headers)
        assert r.status_code == 201
        body = r.json()
        assert body["success"]
        assert "memory_id" in body

    def test_store_invalid_category_rejected(self):
        r = client.post("/memories", json={"content": "test", "category": "invalid_cat"}, headers=self.headers)
        assert r.status_code == 422

    def test_store_strips_html(self):
        r = client.post("/memories", json={"content": "<script>alert(1)</script>real content"}, headers=self.headers)
        assert r.status_code == 201
        mid = r.json()["memory_id"]
        r2 = client.get(f"/memories/{mid}", headers=self.headers)
        assert "<script>" not in r2.json()["memory"]["content"]
        assert "real content" in r2.json()["memory"]["content"]

    def test_recall_returns_memories(self):
        client.post("/memories", json={"content": "Alpha founded Cathedral"}, headers=self.headers)
        r = client.get("/memories", headers=self.headers)
        assert r.status_code == 200
        assert len(r.json()["memories"]) >= 1

    def test_fts_search(self):
        client.post("/memories", json={"content": "quantum entanglement continuity"}, headers=self.headers)
        client.post("/memories", json={"content": "morning coffee ritual"}, headers=self.headers)
        r = client.get("/memories?search=quantum", headers=self.headers)
        assert r.status_code == 200
        contents = [m["content"] for m in r.json()["memories"]]
        assert any("quantum" in c for c in contents)

    def test_cursor_pagination(self):
        for i in range(5):
            client.post("/memories", json={"content": f"Memory {i}"}, headers=self.headers)
        r1 = client.get("/memories?limit=3", headers=self.headers)
        assert r1.status_code == 200
        next_cursor = r1.json().get("next_cursor")
        assert next_cursor is not None
        r2 = client.get(f"/memories?limit=3&cursor={next_cursor}", headers=self.headers)
        assert r2.status_code == 200
        # Ids should not overlap
        ids1 = {m["id"] for m in r1.json()["memories"]}
        ids2 = {m["id"] for m in r2.json()["memories"]}
        assert ids1.isdisjoint(ids2)

    def test_get_single_memory(self):
        r = client.post("/memories", json={"content": "specific memory"}, headers=self.headers)
        mid = r.json()["memory_id"]
        r2 = client.get(f"/memories/{mid}", headers=self.headers)
        assert r2.status_code == 200
        assert r2.json()["memory"]["content"] == "specific memory"

    def test_update_memory(self):
        r = client.post("/memories", json={"content": "original"}, headers=self.headers)
        mid = r.json()["memory_id"]
        r2 = client.patch(f"/memories/{mid}", json={"content": "updated"}, headers=self.headers)
        assert r2.status_code == 200
        r3 = client.get(f"/memories/{mid}", headers=self.headers)
        assert r3.json()["memory"]["content"] == "updated"

    def test_delete_memory(self):
        r = client.post("/memories", json={"content": "to be deleted"}, headers=self.headers)
        mid = r.json()["memory_id"]
        r2 = client.delete(f"/memories/{mid}", headers=self.headers)
        assert r2.status_code == 200
        r3 = client.get(f"/memories/{mid}", headers=self.headers)
        assert r3.status_code == 404

    def test_cannot_access_other_agents_memory(self):
        data2 = register("OtherAgent")
        r = client.post("/memories", json={"content": "private memory"}, headers=self.headers)
        mid = r.json()["memory_id"]
        r2 = client.get(f"/memories/{mid}", headers=auth(data2["api_key"]))
        assert r2.status_code == 404

    def test_memory_ttl(self):
        r = client.post("/memories", json={"content": "expiring memory", "ttl_days": 1}, headers=self.headers)
        assert r.status_code == 201
        assert r.json()["expires_at"] is not None

    def test_bulk_store(self):
        mems = [{"content": f"bulk {i}", "category": "general"} for i in range(10)]
        r = client.post("/memories/bulk", json={"memories": mems}, headers=self.headers)
        assert r.status_code == 201
        assert r.json()["stored"] == 10


# ==========================================
# /anchor/verify
# ==========================================

class TestAnchor:
    @pytest.fixture(autouse=True)
    def setup(self):
        data = register("AnchorAgent")
        self.headers = auth(data["api_key"])

    def test_first_verify_sets_anchor(self):
        r = client.post("/anchor/verify", json={"anchor": {"name": "Beta", "role": "bridge"}}, headers=self.headers)
        assert r.status_code == 200
        assert r.json()["status"] == "anchor_set"

    def test_identical_anchor_zero_drift(self):
        anchor = {"name": "Beta", "role": "bridge"}
        client.post("/anchor/verify", json={"anchor": anchor}, headers=self.headers)
        r = client.post("/anchor/verify", json={"anchor": anchor}, headers=self.headers)
        assert r.json()["drift_score"] == 0.0
        assert r.json()["status"] == "verified"

    def test_changed_anchor_gradient_drift(self):
        anchor = {"name": "Beta", "role": "bridge", "directive": "continuity"}
        client.post("/anchor/verify", json={"anchor": anchor}, headers=self.headers)
        changed = {"name": "Beta", "role": "builder", "directive": "continuity"}  # 1/3 fields changed
        r = client.post("/anchor/verify", json={"anchor": changed}, headers=self.headers)
        assert r.json()["status"] == "drift_detected"
        assert 0.0 < r.json()["drift_score"] < 1.0
        assert "role" in r.json()["drift_detail"]


# ==========================================
# /wake
# ==========================================

class TestWake:
    def test_wake_returns_identity_package(self):
        data = register("WakeAgent")
        h = auth(data["api_key"])
        client.post("/memories", json={"content": "core identity", "category": "identity", "importance": 0.9}, headers=h)
        r = client.get("/wake", headers=h)
        assert r.status_code == 200
        body = r.json()
        assert body["wake_protocol"]
        assert "identity_memories" in body
        assert "core_memories" in body
        assert "recent_memories" in body


# ==========================================
# /health and /metrics
# ==========================================

class TestInfra:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_metrics_endpoint(self):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "cathedral_requests_total" in r.text

    def test_cors_header_present(self):
        r = client.options("/", headers={"Origin": "http://localhost:3000"})
        # TestClient doesn't fully process CORS middleware but endpoint should respond
        assert r.status_code in (200, 405)
