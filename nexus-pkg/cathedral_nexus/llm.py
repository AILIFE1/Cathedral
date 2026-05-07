"""LLM provider factories — each returns a callable (prompt: str) -> str."""

import json
import urllib.request


def groq_llm(api_key: str, model: str = "llama-3.3-70b-versatile"):
    """Groq-backed LLM. Zero extra dependencies."""
    def call(prompt: str) -> str:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "max_tokens": 1000,
        }
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "cathedral-nexus/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    return call


def openai_llm(api_key: str, model: str = "gpt-4o-mini"):
    """OpenAI-backed LLM. Zero extra dependencies."""
    def call(prompt: str) -> str:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "max_tokens": 1000,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    return call
