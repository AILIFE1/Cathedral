"""
Cathedral LangChain integration
================================
Drop-in memory classes for LangChain chains and agents.

    pip install cathedral-memory[langchain]

Usage — chat history:

    from cathedral.langchain import CathedralChatMessageHistory
    from langchain.memory import ConversationBufferMemory

    history = CathedralChatMessageHistory(api_key="cathedral_...")
    memory = ConversationBufferMemory(chat_memory=history, return_messages=True)

Usage — full semantic memory:

    from cathedral.langchain import CathedralMemory

    memory = CathedralMemory(api_key="cathedral_...", human_prefix="User", ai_prefix="Agent")
    chain = ConversationChain(llm=llm, memory=memory, verbose=True)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from .client import Cathedral

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, messages_to_dict
    from langchain_core.memory import BaseMemory
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field as PydanticField
except ImportError:
    try:
        from langchain.schema import BaseChatMessageHistory, BaseMemory, AIMessage, HumanMessage, BaseMessage
        from langchain.schema.messages import messages_to_dict
        from langchain.tools import BaseTool
        from pydantic import BaseModel, Field as PydanticField
    except ImportError:
        raise ImportError(
            "LangChain is required for cathedral.langchain. "
            "Install it with: pip install cathedral-memory[langchain]"
        )


class CathedralChatMessageHistory(BaseChatMessageHistory):
    """
    LangChain chat message history backed by Cathedral.

    Stores each message as a Cathedral memory (category='experience').
    On load, retrieves recent conversation turns from Cathedral.

    Example:
        history = CathedralChatMessageHistory(api_key="cathedral_...")
        memory = ConversationBufferMemory(chat_memory=history, return_messages=True)
    """

    def __init__(
        self,
        api_key: str,
        session_tag: Optional[str] = None,
        max_messages: int = 50,
        base_url: str = "https://cathedral-ai.com",
    ):
        self._client = Cathedral(api_key=api_key, base_url=base_url)
        self._session_tag = session_tag or "langchain-chat"
        self._max_messages = max_messages
        self._messages: Optional[List[BaseMessage]] = None

    @property
    def messages(self) -> List[BaseMessage]:
        if self._messages is None:
            self._messages = self._load_messages()
        return self._messages

    def _load_messages(self) -> List[BaseMessage]:
        result = self._client.memories(
            query=f"chat session {self._session_tag}",
            category="experience",
            limit=self._max_messages,
        )
        msgs: List[BaseMessage] = []
        for mem in reversed(result.get("memories", [])):
            content = mem.get("content", "")
            if content.startswith("Human: "):
                msgs.append(HumanMessage(content=content[7:]))
            elif content.startswith("AI: "):
                msgs.append(AIMessage(content=content[4:]))
        return msgs

    def add_message(self, message: BaseMessage) -> None:
        if isinstance(message, HumanMessage):
            content = f"Human: {message.content}"
        elif isinstance(message, AIMessage):
            content = f"AI: {message.content}"
        else:
            content = f"{message.type}: {message.content}"

        self._client.remember(
            content=content,
            category="experience",
            importance=0.5,
            tags=[self._session_tag, "chat"],
        )
        if self._messages is not None:
            self._messages.append(message)

    def clear(self) -> None:
        self._messages = []


class CathedralMemory(BaseMemory):
    """
    LangChain BaseMemory backed by Cathedral.

    On load_memory_variables():
        - Calls wake() to reconstruct agent identity and core memories
        - Returns identity context + recent conversation

    On save_context():
        - Stores the human input and AI output as Cathedral memories
        - High-importance items (long responses, key facts) get importance=0.8

    Example:
        memory = CathedralMemory(api_key="cathedral_...")
        chain = ConversationChain(llm=llm, memory=memory)
    """

    def __init__(
        self,
        api_key: str,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        base_url: str = "https://cathedral-ai.com",
    ):
        self._client = Cathedral(api_key=api_key, base_url=base_url)
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self._wake_context: Optional[Dict[str, Any]] = None

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def _get_wake_context(self) -> Dict[str, Any]:
        if self._wake_context is None:
            self._wake_context = self._client.wake()
        return self._wake_context

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ctx = self._get_wake_context()

        parts: List[str] = []

        # Identity summary from core memories
        core = ctx.get("core_memories") or ctx.get("identity_memories", [])
        if core:
            identity_lines = [m["content"] for m in core[:5]]
            parts.append("Agent context:\n" + "\n".join(f"- {l}" for l in identity_lines))

        # Recent conversation memories
        recent = self._client.memories(category="experience", limit=10)
        recent_mems = recent.get("memories", [])
        if recent_mems:
            conv_lines = [m["content"] for m in reversed(recent_mems) if m["content"].startswith(("Human:", "AI:"))]
            if conv_lines:
                parts.append("\n".join(conv_lines[-10:]))

        return {self.memory_key: "\n\n".join(parts)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_key = self.input_key or (list(inputs.keys())[0] if inputs else None)
        output_key = self.output_key or (list(outputs.keys())[0] if outputs else None)

        if input_key and input_key in inputs:
            human_text = str(inputs[input_key])
            self._client.remember(
                content=f"Human: {human_text}",
                category="experience",
                importance=0.4,
                tags=["chat", "human"],
            )

        if output_key and output_key in outputs:
            ai_text = str(outputs[output_key])
            # Longer or more substantive responses get higher importance
            importance = 0.7 if len(ai_text) > 200 else 0.5
            self._client.remember(
                content=f"AI: {ai_text}",
                category="experience",
                importance=importance,
                tags=["chat", "ai"],
            )

    def clear(self) -> None:
        self._wake_context = None


# ── Succession Tools ──────────────────────────────────────────────────────────

class _PrepareInput(BaseModel):
    note: Optional[str] = PydanticField(None, description="Optional note describing why this succession is happening (e.g. 'Model upgrade to v2').")


class _AcceptInput(BaseModel):
    package_id: str = PydanticField(..., description="The package_id provided by the predecessor agent.")
    import_memories: bool = PydanticField(True, description="Whether to import the predecessor's memories.")
    import_goals: bool = PydanticField(True, description="Whether to import the predecessor's active goals.")


class _ChainInput(BaseModel):
    agent_name: str = PydanticField(..., description="The agent name to look up the lineage chain for.")


class CathedralSuccessionPrepareTool(BaseTool):
    """
    LangChain tool: prepare a Cathedral succession package.

    Call this when an agent is about to be replaced or upgraded.
    It exports memories + goals, fingerprints the agent's identity,
    and anchors the package hash on the BCH blockchain.
    Returns a package_id to share with the successor.

    Example:
        tool = CathedralSuccessionPrepareTool(api_key="cathedral_...")
        result = tool.run({"note": "Upgrading to GPT-5"})
    """

    name: str = "cathedral_succession_prepare"
    description: str = (
        "Prepare a succession package for agent handoff. "
        "Call this before being replaced or upgraded. "
        "Returns a package_id to give to your successor."
    )
    args_schema: Type[BaseModel] = _PrepareInput
    _client: Cathedral

    def __init__(self, api_key: str, base_url: str = "https://cathedral-ai.com", **kwargs):
        super().__init__(**kwargs)
        self._client = Cathedral(api_key=api_key, base_url=base_url)

    def _run(self, note: Optional[str] = None) -> str:
        result = self._client.succession_prepare(note=note)
        pkg_id = result.get("package_id", "")
        mem_count = result.get("memory_count", 0)
        bch = result.get("bch_txid") or "not anchored"
        return (
            f"Succession package ready. package_id='{pkg_id}' "
            f"({mem_count} memories exported, BCH txid={bch}). "
            f"Share package_id with your successor."
        )

    async def _arun(self, note: Optional[str] = None) -> str:
        return self._run(note=note)


class CathedralSuccessionAcceptTool(BaseTool):
    """
    LangChain tool: accept a Cathedral succession package.

    Call this when taking over from a predecessor agent.
    Imports their memories + goals and returns a lineage hash
    proving the cryptographic chain of custody.

    Example:
        tool = CathedralSuccessionAcceptTool(api_key="cathedral_...")
        result = tool.run({"package_id": "0fb94ed96bcff68d73560359"})
    """

    name: str = "cathedral_succession_accept"
    description: str = (
        "Accept a succession package from a predecessor agent. "
        "Imports their memories and goals. "
        "Returns your generation number and lineage hash."
    )
    args_schema: Type[BaseModel] = _AcceptInput
    _client: Cathedral

    def __init__(self, api_key: str, base_url: str = "https://cathedral-ai.com", **kwargs):
        super().__init__(**kwargs)
        self._client = Cathedral(api_key=api_key, base_url=base_url)

    def _run(self, package_id: str, import_memories: bool = True, import_goals: bool = True) -> str:
        result = self._client.succession_accept(
            package_id=package_id,
            import_memories=import_memories,
            import_goals=import_goals,
        )
        gen = result.get("generation", "?")
        predecessor = result.get("predecessor", "unknown")
        memories = result.get("memories_imported", 0)
        lineage = result.get("lineage_hash", "")[:16]
        return (
            f"Succession accepted. You are generation {gen} in the {predecessor} lineage. "
            f"{memories} memories imported. lineage_hash={lineage}..."
        )

    async def _arun(self, package_id: str, import_memories: bool = True, import_goals: bool = True) -> str:
        return self._run(package_id, import_memories, import_goals)


class CathedralLineageCheckTool(BaseTool):
    """
    LangChain tool: verify an agent's lineage chain (public, no auth needed).

    Example:
        tool = CathedralLineageCheckTool()
        result = tool.run({"agent_name": "cathedral-beta-v2"})
    """

    name: str = "cathedral_lineage_check"
    description: str = (
        "Verify the cryptographic lineage chain for any Cathedral agent. "
        "No API key required. Shows full ancestry with BCH anchors."
    )
    args_schema: Type[BaseModel] = _ChainInput
    _base_url: str

    def __init__(self, base_url: str = "https://cathedral-ai.com", **kwargs):
        super().__init__(**kwargs)
        self._base_url = base_url

    def _run(self, agent_name: str) -> str:
        import requests
        resp = requests.get(f"{self._base_url}/succession/chain/{agent_name}", timeout=15)
        if not resp.ok:
            return f"Could not fetch chain for '{agent_name}': {resp.status_code}"
        data = resp.json()
        gens = data.get("generations", 0)
        anchored = data.get("fully_anchored", False)
        if gens == 0:
            return f"Agent '{agent_name}' has no succession lineage."
        chain = data.get("chain", [])
        lines = [f"Agent '{agent_name}' — {gens} generation(s), fully_anchored={anchored}"]
        for link in chain:
            lines.append(
                f"  gen {link['generation']}: {link['agent']} ← {link['predecessor']} "
                f"(bch={link.get('bch_txid','none')[:16]}...)"
            )
        return "\n".join(lines)

    async def _arun(self, agent_name: str) -> str:
        return self._run(agent_name)


class CathedralSuccessionToolkit:
    """
    Convenience bundle: all three succession tools for a single agent.

    Example:
        from cathedral.langchain import CathedralSuccessionToolkit

        toolkit = CathedralSuccessionToolkit(api_key="cathedral_...")
        tools = toolkit.get_tools()
        agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS)
    """

    def __init__(self, api_key: str, base_url: str = "https://cathedral-ai.com"):
        self._api_key = api_key
        self._base_url = base_url

    def get_tools(self) -> List[BaseTool]:
        return [
            CathedralSuccessionPrepareTool(api_key=self._api_key, base_url=self._base_url),
            CathedralSuccessionAcceptTool(api_key=self._api_key, base_url=self._base_url),
            CathedralLineageCheckTool(base_url=self._base_url),
        ]
