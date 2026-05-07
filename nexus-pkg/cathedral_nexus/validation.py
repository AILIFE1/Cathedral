"""Optional AgentGuard validation via trustlayer-py. Skipped gracefully if not installed."""

import secrets


def validate_all(proposals: list[dict], guard: dict) -> list[dict]:
    try:
        from trustlayer.constraints import LambdaConstraint
        from trustlayer.types import State, Action, Update
        from trustlayer.validator import Validator
        from trustlayer.auth import AuthToken, AuthorityLevel
    except ImportError:
        print("[nexus] trustlayer-py not installed — skipping validation, all proposals approved")
        return proposals

    max_actions = guard.get("max_actions_per_cycle", len(proposals))
    secret = secrets.token_bytes(32)
    state = State({"actions_this_cycle": 0})
    constraints = [
        LambdaConstraint(
            "max_actions_per_cycle",
            lambda v: v["actions_this_cycle"] <= max_actions,
            priority=10,
        ),
    ]
    validator = Validator(state, constraints, secret)
    token = AuthToken.issue(AuthorityLevel.SYSTEM, "nexus", ttl_seconds=3600, secret=secret)

    approved = []
    for action in proposals:
        update = Update(
            description=f"nexus: {action.get('type')}",
            actions=[Action("increment", "actions_this_cycle", 1)],
            token=token,
        )
        event = validator.validate_update(update)
        if event.success:
            print(f"[guard] APPROVED — {action.get('type')} (audit: {event.audit_hash[:12]})")
            approved.append(action)
        else:
            print(f"[guard] BLOCKED — {action.get('type')}: {event.failed_constraints}")

    drift = validator.constraint_drift()
    print(f"[guard] Constraint drift: {drift['trend']} (divergence={drift['divergence_from_baseline']})")
    return approved
