# Cathedral Integration Examples

Practical integrations showing how to wire Cathedral persistent memory into popular agent frameworks.

## CrewAI

**Problem:** CrewAI agents silently change behaviour across session boundaries when context compression occurs. Ghost lexicon decay, tool-call sequence shifts, and semantic drift don't raise exceptions — they produce subtly wrong outputs that look like correct execution.

**Solution:** Cathedral wraps each `kickoff()` with identity state persistence and drift scoring.

### `crewai_integration.py` — MonitoredCrew

Drop-in wrapper for `crewai.Crew`. Adds `/wake` + `/snapshot` + `/drift` around every `kickoff()` call.

```python
from crewai import Agent, Task, Crew
from crewai_integration import MonitoredCrew

crew = Crew(agents=[researcher, analyst, writer], tasks=[...])

monitored = MonitoredCrew(
    crew=crew,
    api_key="your_cathedral_api_key",
    agent_name="my-research-crew",
)

result = monitored.kickoff(inputs={"topic": "AI memory research"})

print(result.output)
print(f"Identity drift this run: {result.drift_score:.3f}")
print(f"Snapshot: {result.snapshot_id}")
```

### `crewai_drift_detection.py` — CrewDriftMonitor

Per-agent drift monitoring for multi-agent crews. Takes independent snapshots per agent so you can identify *which* agent drifted and find the root cause of crew-level degradation.

```python
monitor = CrewDriftMonitor(api_keys={
    "researcher": "cathedral_key_1",
    "analyst":    "cathedral_key_2",
    "writer":     "cathedral_key_3",
})

contexts = monitor.wake_all()
result = crew.kickoff(inputs={...})
reports = monitor.snapshot_and_score_all(run_id="daily-run-001")

monitor.print_summary(reports)
# Researcher  0.312 ██████  ← INVESTIGATE
# Analyst     0.104 ██
# Writer      0.089 █
# Root cause candidate: researcher (drift=0.312)
```

## How it works

```
Run N:                          Run N+1:
  /wake  ← restore identity       /wake  ← pick up from snapshot N
  kickoff()                        kickoff()
  /snapshot ← freeze state         /snapshot ← freeze state
  /drift  ← score vs baseline      /drift  ← score vs run N snapshot
```

Each snapshot is SHA-256 hashed and optionally BCH blockchain-anchored for tamper-evidence. The drift score (0.0–1.0) measures how much the agent's identity state changed between runs.

Live example running in production: [cathedral-ai.com/cathedral-beta](https://cathedral-ai.com/cathedral-beta)

## Install

```bash
pip install cathedral-memory requests crewai
```

Free API key: [cathedral-ai.com](https://cathedral-ai.com)
