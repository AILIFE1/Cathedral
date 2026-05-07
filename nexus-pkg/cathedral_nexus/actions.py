"""Built-in action handlers. Register these or write your own."""


def cathedral_memory(action: dict, ctx: dict) -> bool:
    """Store a memory in Cathedral. Requires ctx['client'] to be a CathedralClient."""
    client = ctx.get("client")
    if not client:
        print("[action] cathedral_memory: no Cathedral client in ctx")
        return False
    content = action.get("content", "")
    if not content:
        return False
    VALID = {"experience", "general", "goal", "identity", "relationship", "skill"}
    category = action.get("category", "general")
    if category not in VALID:
        category = "general"
    result = client.store_memory(content, category, float(action.get("importance", 0.7)))
    ok = "error" not in result
    print(f"[action] cathedral_memory: {'ok' if ok else result.get('error', result.get('detail'))}")
    return ok


def log_only(action: dict, ctx: dict) -> bool:
    """Print the action and return True. Useful for testing without side effects."""
    print(f"[action] log_only: {action}")
    return True
