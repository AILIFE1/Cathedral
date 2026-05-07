"""AgentGuard integration — validates proposed actions before execution."""

from trustlayer.constraints import LambdaConstraint
from trustlayer.types import State, Action, Update
from trustlayer.validator import Validator
from trustlayer.auth import AuthToken, AuthorityLevel
import secrets


ALLOWED_TYPES = {"queue_post", "store_memory", "update_goal", "adjust_strategy"}


def build_validator(config: dict) -> tuple[Validator, AuthToken, State]:
    guard_cfg = config["guard"]
    secret = secrets.token_bytes(32)

    state = State({"actions_this_cycle": 0, "trust_ok": True, "goal_locked": True})

    constraints = [
        LambdaConstraint(
            "max_actions_per_cycle",
            lambda v: v["actions_this_cycle"] <= guard_cfg["max_actions_per_cycle"],
            priority=10,
        ),
        LambdaConstraint(
            "trust_threshold_met",
            lambda v: v["trust_ok"],
            priority=5,
        ),
        LambdaConstraint(
            "goal_cannot_be_deleted",
            lambda v: v["goal_locked"],
            priority=1,
        ),
    ]

    validator = Validator(state, constraints, secret)
    token = AuthToken.issue(AuthorityLevel.SYSTEM, "nexus", ttl_seconds=3600, secret=secret)
    return validator, token, state


def report_constraint_drift(validator: Validator) -> None:
    d = validator.constraint_drift()
    trend      = d["trend"]
    divergence = d["divergence_from_baseline"]
    print(f"[guard] Constraint drift: {trend} (divergence={divergence})")
    if trend == "permissive_drift":
        print(f"[guard] WARNING — constraints removed: {d['removed_constraints']}")
    elif d["added_constraints"]:
        print(f"[guard] Constraints added: {d['added_constraints']}")


def validate_action(
    action: dict,
    validator: Validator,
    token: AuthToken,
    state: State,
    trust_score: float,
    trust_threshold: float,
) -> bool:
    if action.get("type") not in ALLOWED_TYPES:
        print(f"[guard] BLOCKED — unknown action type: {action.get('type')}")
        return False

    trust_ok = trust_score >= trust_threshold if isinstance(trust_score, (int, float)) else True

    update = Update(
        description=f"nexus action: {action.get('type')}",
        actions=[
            Action("increment", "actions_this_cycle", 1),
            Action("set", "trust_ok", trust_ok),
        ],
        token=token,
    )
    event = validator.validate_update(update)

    if event.success:
        print(f"[guard] APPROVED — {action.get('type')} (audit: {event.audit_hash[:12]})")
    else:
        print(f"[guard] BLOCKED — {event.failed_constraints}")

    return event.success
