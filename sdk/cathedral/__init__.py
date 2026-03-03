"""
cathedral-memory
================
Python client for the Cathedral persistent memory API.

    from cathedral import Cathedral

    c = Cathedral.register("MyAgent", "What my agent does")
    c = Cathedral(api_key="cathedral_...")

    context = c.wake()
    c.remember("Something worth keeping", category="experience", importance=0.8)

Docs: https://cathedral-ai.com
"""

from .client import Cathedral
from .temporal import build_temporal_context
from .exceptions import CathedralError, AuthError, NotFoundError, RateLimitError

__version__ = "0.1.0"
__all__ = [
    "Cathedral",
    "build_temporal_context",
    "CathedralError",
    "AuthError",
    "NotFoundError",
    "RateLimitError",
]
