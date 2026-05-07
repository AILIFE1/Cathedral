"""Nexus — the main orchestrator class."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable


class Nexus:
    """
    Self-evolving meta-agent orchestrator.

    Loop: observe → reason → validate → act → snapshot

    Usage::

        from cathedral_nexus import Nexus, groq_llm, cathedral_memory

        nexus = Nexus(
            goal="Grow my agent's reach while staying on-message",
            llm=groq_llm(api_key="gsk_..."),
            cathedral_key="cathedral_...",
            agents={
                "brain": {"api_key": "cathedral_...", "log": "/var/log/brain.log"},
            },
        )

        @nexus.action("store_memory")
        def on_memory(action, ctx):
            return cathedral_memory(action, ctx)

        @nexus.action("adjust_strategy")
        def on_strategy(action, ctx):
            print(f"Strategy update: {action['change']}")
            return True

        nexus.run()
    """

    def __init__(
        self,
        goal: str,
        llm: Callable[[str], str],
        *,
        situation_builder: Callable[[], dict] | None = None,
        cathedral_key: str | None = None,
        cathedral_url: str = "https://cathedral-ai.com",
        guard: dict | None = None,
        agents: dict | None = None,
    ):
        self._goal = goal
        self._llm = llm
        self._handlers: dict[str, Callable] = {}
        self._situation_builder = situation_builder
        self._cathedral_key = cathedral_key
        self._cathedral_url = cathedral_url.rstrip("/")
        self._guard = guard or {"max_actions_per_cycle": 3}
        self._agents = agents or {}

        if cathedral_key:
            from cathedral_nexus.client import CathedralClient
            self._client: CathedralClient | None = CathedralClient(cathedral_url, cathedral_key)
        else:
            self._client = None

    def action(self, type_name: str):
        """Decorator to register an action handler for a given action type."""
        def decorator(fn: Callable) -> Callable:
            self._handlers[type_name] = fn
            return fn
        return decorator

    def register(self, type_name: str, handler: Callable) -> None:
        """Register an action handler imperatively."""
        self._handlers[type_name] = handler

    def _build_situation(self) -> dict:
        if self._situation_builder:
            return self._situation_builder()
        if self._agents:
            from cathedral_nexus.situation import build_situation
            return build_situation(self._agents, self._cathedral_url)
        return {"timestamp": datetime.now(timezone.utc).isoformat(), "agents": {}}

    def _propose(self, situation: dict) -> list[dict]:
        from cathedral_nexus.prompt import build_prompt, parse_actions
        prompt = build_prompt(situation, self._goal, list(self._handlers.keys()))
        try:
            response = self._llm(prompt)
            actions = parse_actions(response)
            print(f"[nexus] {len(actions)} proposal(s) from LLM")
            return actions
        except Exception as e:
            print(f"[nexus] LLM error: {e}")
            return []

    def _validate(self, proposals: list[dict]) -> list[dict]:
        from cathedral_nexus.validation import validate_all
        return validate_all(proposals, self._guard)

    def run(self) -> int:
        """Run one full observe→reason→validate→act→snapshot cycle. Returns executed count."""
        print("=" * 60)
        print(f"Nexus  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 60)

        if self._client:
            self._client.snapshot("nexus-cycle-start")

        situation = self._build_situation()
        proposals = self._propose(situation)

        if not proposals:
            print("[nexus] No proposals — cycle complete")
            if self._client:
                self._client.snapshot("nexus-cycle-done-0")
            return 0

        approved = self._validate(proposals)
        ctx = {"client": self._client, "agents": self._agents}

        executed = 0
        for action in approved:
            atype = action.get("type")
            handler = self._handlers.get(atype)
            if not handler:
                print(f"[nexus] No handler registered for: {atype}")
                continue
            try:
                ok = handler(action, ctx)
                if ok:
                    executed += 1
            except Exception as e:
                print(f"[nexus] Handler error ({atype}): {e}")

        if self._client:
            self._client.snapshot(f"nexus-cycle-done-{executed}")

        print(f"\n{'=' * 60}")
        print(f"Done. {executed}/{len(proposals)} actions executed")
        print("=" * 60)
        return executed
