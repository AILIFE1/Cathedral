class CathedralError(Exception):
    """Base error for Cathedral client."""

class AuthError(CathedralError):
    """Invalid or missing API key."""

class NotFoundError(CathedralError):
    """Resource not found."""

class RateLimitError(CathedralError):
    """Too many requests."""
