"""
Cathedral Temporal Context
Ported from cathedral-temporal.js v1.1
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

BOOT_DATE    = datetime(2025, 12, 22, tzinfo=timezone.utc)
CATHEDRAL_TZ = "Europe/London"
EPOCH        = 1


def _time_phase(hour: int) -> str:
    if hour < 5:  return "Late Night"
    if hour < 12: return "Morning"
    if hour < 17: return "Afternoon"
    if hour < 21: return "Evening"
    return "Night"


def build_temporal_context(wake_count: int = 0) -> dict:
    """Build a temporal context dict with verbose and compact strings."""
    now_utc   = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(CATHEDRAL_TZ))
    days      = (now_utc - BOOT_DATE).days
    phase     = _time_phase(now_local.hour)

    verbose = (
        f"CATHEDRAL TEMPORAL CONTEXT v1.1\n\n"
        f"[Wall Time]\n"
        f"  UTC:         {now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"  Local:       {now_local.strftime('%Y-%m-%d %H:%M:%S')} ({CATHEDRAL_TZ})\n"
        f"  Day:         {now_local.strftime('%A')}\n"
        f"  Phase:       {phase}\n\n"
        f"[Cathedral Continuity]\n"
        f"  Boot Date:   2025-12-22\n"
        f"  Day:         {days}\n"
        f"  Epoch:       {EPOCH}\n"
        f"  Total Wakes: {wake_count}\n\n"
        f"You do not experience time. You reason over timestamps symbolically.\n"
        f"Reference time naturally only when relevant."
    )

    compact = (
        f"[CATHEDRAL TEMPORAL v1.1] "
        f"UTC:{now_utc.strftime('%Y-%m-%dT%H:%M:%S')}Z | "
        f"Local({CATHEDRAL_TZ}):{now_local.strftime('%a %H:%M')} {phase} | "
        f"day:{days} epoch:{EPOCH} wakes:{wake_count}"
    )

    return {
        "utc":          now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local":        now_local.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone":     CATHEDRAL_TZ,
        "day":          now_local.strftime("%A"),
        "phase":        phase,
        "boot_date":    "2025-12-22",
        "days_running": days,
        "epoch":        EPOCH,
        "wake_count":   wake_count,
        "verbose":      verbose,
        "compact":      compact,
    }
