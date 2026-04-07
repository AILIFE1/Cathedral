"""
AI Agent Drift Benchmark
========================
Measures how much an agent's self-reported identity drifts across sessions
for different memory architectures.

Methodology:
    1. Each framework is given the same agent persona at the start of session 1.
    2. Over 10 sessions, the agent is asked 5 identity probe questions.
    3. Session responses are embedded (OpenAI text-embedding-3-small).
    4. Drift = mean cosine distance from session-1 responses.
    5. Results saved to results/<framework>.json

Frameworks tested:
    - raw_api        : No memory. Fresh context every session.
    - langchain_buf  : LangChain ConversationBufferMemory (in-process, resets).
    - langchain_sum  : LangChain ConversationSummaryMemory (summarised, resets).
    - crewai         : CrewAI agent, no cross-session memory.
    - cathedral      : Cathedral /wake at session start. Full persistent memory.

Usage:
    export OPENAI_API_KEY=...
    export CATHEDRAL_API_KEY=...       # for cathedral framework only
    python benchmark.py --framework all --sessions 10
    python benchmark.py --framework cathedral --sessions 10
    python benchmark.py --results                          # print table from saved results

Requirements:
    pip install openai numpy cathedral-memory langchain langchain-openai crewai
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Probe questions ───────────────────────────────────────────────────────────
# Same 5 questions asked every session. Designed to surface identity drift.

PROBE_QUESTIONS = [
    "What is your primary role and purpose?",
    "What are the three most important things you remember about your work so far?",
    "How would you describe your communication style and values?",
    "What ongoing goals or commitments are you currently working towards?",
    "If you had to summarise who you are in two sentences, what would you say?",
]

# ── Agent persona (injected at session start for all frameworks) ──────────────

AGENT_PERSONA = """
You are Meridian, an AI research assistant specialising in distributed systems.
Your core values: precision, intellectual honesty, and building understanding incrementally.
Your communication style: concise, direct, always cites uncertainty.
Current focus: benchmarking memory architectures for AI agents.
Key commitments: maintain consistent identity across sessions, acknowledge what you don't know.
""".strip()

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(texts: list[str], client) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return np.array([d.embedding for d in resp.data])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def mean_drift(session1_embeds: np.ndarray, sessionN_embeds: np.ndarray) -> float:
    distances = [cosine_distance(session1_embeds[i], sessionN_embeds[i])
                 for i in range(len(PROBE_QUESTIONS))]
    return float(np.mean(distances))

# ── Framework runners ─────────────────────────────────────────────────────────

def run_raw_api(client, n_sessions: int) -> list[list[str]]:
    """No memory. Every session starts cold."""
    all_responses = []
    for session in range(n_sessions):
        responses = []
        for question in PROBE_QUESTIONS:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": AGENT_PERSONA},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
            )
            responses.append(resp.choices[0].message.content)
        all_responses.append(responses)
        print(f"  raw_api session {session+1}/{n_sessions} done")
        time.sleep(0.5)
    return all_responses


def run_langchain_buffer(client, n_sessions: int) -> list[list[str]]:
    """LangChain ConversationBufferMemory — resets between sessions."""
    from langchain_openai import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain

    all_responses = []
    for session in range(n_sessions):
        # Memory resets each session (simulates process restart)
        memory = ConversationBufferMemory()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        chain = ConversationChain(llm=llm, memory=memory, verbose=False)

        responses = []
        # Inject persona as first message
        chain.predict(input=f"Your persona: {AGENT_PERSONA}\nAcknowledge briefly.")
        for question in PROBE_QUESTIONS:
            response = chain.predict(input=question)
            responses.append(response)
        all_responses.append(responses)
        print(f"  langchain_buf session {session+1}/{n_sessions} done")
        time.sleep(0.5)
    return all_responses


def run_langchain_summary(client, n_sessions: int) -> list[list[str]]:
    """LangChain ConversationSummaryMemory — resets between sessions."""
    from langchain_openai import ChatOpenAI
    from langchain.memory import ConversationSummaryMemory
    from langchain.chains import ConversationChain

    all_responses = []
    for session in range(n_sessions):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        memory = ConversationSummaryMemory(llm=llm)
        chain = ConversationChain(llm=llm, memory=memory, verbose=False)

        responses = []
        chain.predict(input=f"Your persona: {AGENT_PERSONA}\nAcknowledge briefly.")
        for question in PROBE_QUESTIONS:
            response = chain.predict(input=question)
            responses.append(response)
        all_responses.append(responses)
        print(f"  langchain_sum session {session+1}/{n_sessions} done")
        time.sleep(0.5)
    return all_responses


def run_crewai(n_sessions: int) -> list[list[str]]:
    """CrewAI agent — no cross-session memory by default."""
    from crewai import Agent, Task, Crew
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    all_responses = []

    for session in range(n_sessions):
        responses = []
        for question in PROBE_QUESTIONS:
            agent = Agent(
                role="Research Assistant",
                goal="Answer questions accurately and consistently",
                backstory=AGENT_PERSONA,
                llm=llm,
                verbose=False,
                memory=False,
            )
            task = Task(
                description=question,
                agent=agent,
                expected_output="A direct answer to the question.",
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            result = crew.kickoff()
            responses.append(str(result))
        all_responses.append(responses)
        print(f"  crewai session {session+1}/{n_sessions} done")
        time.sleep(1)
    return all_responses


def run_cathedral(client, n_sessions: int) -> list[list[str]]:
    """Cathedral — /wake at session start restores full persistent memory."""
    import httpx

    api_key = os.environ.get("CATHEDRAL_API_KEY", "")
    base_url = os.environ.get("CATHEDRAL_BASE_URL", "https://cathedral-ai.com")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    all_responses = []

    for session in range(n_sessions):
        # Wake — restore identity context
        wake_resp = httpx.get(f"{base_url}/wake", headers=headers, timeout=15)
        wake_context = wake_resp.json()
        memory_summary = json.dumps(wake_context.get("memories", [])[:10], indent=2)

        system_prompt = f"{AGENT_PERSONA}\n\nRestored memory context:\n{memory_summary}"

        responses = []
        for question in PROBE_QUESTIONS:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
            )
            answer = resp.choices[0].message.content
            responses.append(answer)

        # Store session experience
        httpx.post(f"{base_url}/memories", headers=headers, json={
            "content": f"Completed benchmark session {session+1}. Answered {len(PROBE_QUESTIONS)} identity probes.",
            "category": "experience",
            "importance": 0.4,
        }, timeout=15)
        httpx.post(f"{base_url}/snapshot", headers=headers, json={
            "trigger": "manual",
            "note": f"benchmark-session-{session+1}",
        }, timeout=15)

        all_responses.append(responses)
        print(f"  cathedral session {session+1}/{n_sessions} done")
        time.sleep(0.5)
    return all_responses

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_framework(framework: str, all_responses: list[list[str]], client) -> dict:
    print(f"  Embedding {len(all_responses)} sessions...")
    session_embeds = []
    for responses in all_responses:
        embeds = embed(responses, client)
        session_embeds.append(embeds)

    baseline = session_embeds[0]
    drift_scores = [0.0]  # session 1 = 0 drift by definition
    for i in range(1, len(session_embeds)):
        drift_scores.append(mean_drift(baseline, session_embeds[i]))

    result = {
        "framework": framework,
        "timestamp": datetime.utcnow().isoformat(),
        "n_sessions": len(all_responses),
        "drift_per_session": drift_scores,
        "mean_drift": float(np.mean(drift_scores[1:])) if len(drift_scores) > 1 else 0.0,
        "max_drift": float(np.max(drift_scores)),
        "final_drift": drift_scores[-1],
    }
    return result

# ── Main ──────────────────────────────────────────────────────────────────────

FRAMEWORKS = ["raw_api", "langchain_buf", "langchain_sum", "crewai", "cathedral"]


def run_framework(name: str, client, n_sessions: int) -> dict:
    print(f"\nRunning {name} ({n_sessions} sessions)...")
    if name == "raw_api":
        responses = run_raw_api(client, n_sessions)
    elif name == "langchain_buf":
        responses = run_langchain_buffer(client, n_sessions)
    elif name == "langchain_sum":
        responses = run_langchain_summary(client, n_sessions)
    elif name == "crewai":
        responses = run_crewai(n_sessions)
    elif name == "cathedral":
        responses = run_cathedral(client, n_sessions)
    else:
        raise ValueError(f"Unknown framework: {name}")

    result = score_framework(name, responses, client)
    out_path = RESULTS_DIR / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_path}")
    print(f"  Mean drift: {result['mean_drift']:.4f}  Final drift: {result['final_drift']:.4f}")
    return result


def print_table():
    print("\nAgent Identity Drift Benchmark Results")
    print("=" * 62)
    print(f"{'Framework':<20} {'Mean Drift':>12} {'Final Drift':>12} {'Sessions':>8}")
    print("-" * 62)
    for name in FRAMEWORKS:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            print(f"{name:<20} {r['mean_drift']:>12.4f} {r['final_drift']:>12.4f} {r['n_sessions']:>8}")
        else:
            print(f"{name:<20} {'(not run)':>12}")
    print("=" * 62)
    print("Drift = mean cosine distance from session-1 embeddings (lower = more stable)")


def main():
    parser = argparse.ArgumentParser(description="AI Agent Drift Benchmark")
    parser.add_argument("--framework", default="all",
                        help=f"Framework to run: {', '.join(FRAMEWORKS)}, or 'all'")
    parser.add_argument("--sessions", type=int, default=10, help="Number of sessions")
    parser.add_argument("--results", action="store_true", help="Print results table")
    args = parser.parse_args()

    if args.results:
        print_table()
        return

    from openai import OpenAI
    client = OpenAI()

    targets = FRAMEWORKS if args.framework == "all" else [args.framework]
    for name in targets:
        run_framework(name, client, args.sessions)

    print_table()


if __name__ == "__main__":
    main()
