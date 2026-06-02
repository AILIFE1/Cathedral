#!/usr/bin/env python3
"""
Cathedral Trust Score — Reputation Oracle
==========================================
Computes a 0-100 trust score for any Cathedral agent from five signals:

  1. Consistency  (30 pts) — average drift across snapshot history
  2. Longevity    (20 pts) — days since agent registration
  3. Anchoring    (20 pts) — BCH blockchain anchors recorded
  4. Succession   (15 pts) — lineage depth + successors created
  5. Activity     (15 pts) — total snapshot count

Endpoint:
  GET /trust/{agent_name}  — public, no auth required

The score is computed fresh on each request and reflects the agent's
current state. Agents cannot manipulate it directly — every input is
derived from actions logged by Cathedral's other systems.
"""

import os
import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException

DB_PATH = os.environ.get("CATHEDRAL_DB", "cathedral_memory.db")
STALE_DAYS = int(os.environ.get("CATHEDRAL_STALE_DAYS", "30"))

trust_router = APIRouter(tags=["trust"])


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _days_since(iso: str) -> int:
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - dt).days)
    except Exception:
        return 0


def compute_trust_score(agent_id: str, agent_name: str, conn: sqlite3.Connection) -> dict:
    breakdown = {}

    # ── 1. Consistency (30 pts) ───────────────────────────────────────────────
    # Use external_divergence from snapshots as the drift signal.
    # Also factor in snapshot-to-snapshot id-based drift by comparing memory counts.
    snaps = conn.execute(
        "SELECT memories_json, external_divergence, created_at "
        "FROM snapshots WHERE agent_id = ? ORDER BY created_at ASC",
        (agent_id,),
    ).fetchall()

    if len(snaps) >= 2:
        # Id-based divergence: (|prev_count - curr_count|) / max(prev_count, 1)
        drifts = []
        for i in range(1, len(snaps)):
            import json as _json
            prev = _json.loads(snaps[i - 1]["memories_json"])
            curr = _json.loads(snaps[i]["memories_json"])
            prev_ids = {m["id"] for m in prev}
            curr_ids = {m["id"] for m in curr}
            changed = len((prev_ids - curr_ids) | (curr_ids - prev_ids))
            drifts.append(changed / max(len(prev_ids), 1))

        avg_drift = sum(drifts) / len(drifts)
        # 0 drift = 30 pts; 0.25+ drift = 0 pts
        consistency_pts = round(max(0.0, 30.0 * (1.0 - avg_drift * 4.0)), 1)
        breakdown["consistency"] = {
            "score": consistency_pts,
            "max": 30,
            "avg_drift": round(avg_drift, 4),
            "snapshots_compared": len(drifts),
        }
    else:
        consistency_pts = 0.0
        breakdown["consistency"] = {
            "score": 0.0,
            "max": 30,
            "note": "Need 2+ snapshots to compute drift",
        }

    # ── 2. Longevity (20 pts) ─────────────────────────────────────────────────
    agent_row = conn.execute(
        "SELECT created_at FROM agents WHERE id = ?", (agent_id,)
    ).fetchone()
    days = _days_since(agent_row["created_at"]) if agent_row else 0
    # 200 days = max 20 pts
    longevity_pts = round(min(20.0, days / 10.0), 1)
    breakdown["longevity"] = {
        "score": longevity_pts,
        "max": 20,
        "days_active": days,
    }

    # ── 3. BCH Anchoring (20 pts) ─────────────────────────────────────────────
    # Count anchors where a real txid was written (blockchain_active)
    bch_count = conn.execute(
        "SELECT COUNT(*) FROM bch_anchors WHERE agent_id = ? AND txid IS NOT NULL AND txid != ''",
        (agent_id,),
    ).fetchone()[0]
    # Also count succession packages we BCH-anchored
    succ_anchors = conn.execute(
        "SELECT COUNT(*) FROM succession_packages "
        "WHERE predecessor_agent_id = ? AND bch_txid IS NOT NULL",
        (agent_id,),
    ).fetchone()[0] if _table_exists(conn, "succession_packages") else 0
    total_anchors = bch_count + succ_anchors
    # 10 anchors = max 20 pts
    anchoring_pts = round(min(20.0, total_anchors * 2.0), 1)
    breakdown["anchoring"] = {
        "score": anchoring_pts,
        "max": 20,
        "bch_anchors": bch_count,
        "succession_anchors": succ_anchors,
        "total": total_anchors,
    }

    # ── 4. Succession (15 pts) ────────────────────────────────────────────────
    lineage = conn.execute(
        "SELECT generation FROM agent_lineage WHERE agent_id = ?", (agent_id,)
    ).fetchone() if _table_exists(conn, "agent_lineage") else None

    accepted_count = conn.execute(
        "SELECT COUNT(*) FROM succession_packages "
        "WHERE predecessor_agent_id = ? AND status = 'accepted'",
        (agent_id,),
    ).fetchone()[0] if _table_exists(conn, "succession_packages") else 0

    generation = lineage["generation"] if lineage else 0

    # Re-attestation lag: days between prepare and accept for most recent succession
    re_attestation_lag_days = None
    pending_attestation_days = None
    lag_bonus = 0.0

    if _table_exists(conn, "succession_packages"):
        latest_accepted = conn.execute(
            """SELECT created_at, accepted_at FROM succession_packages
               WHERE predecessor_agent_id = ? AND status = 'accepted'
               ORDER BY accepted_at DESC LIMIT 1""",
            (agent_id,),
        ).fetchone()
        if latest_accepted and latest_accepted["accepted_at"]:
            try:
                import datetime as _dt2
                t0 = _dt2.datetime.fromisoformat(latest_accepted["created_at"])
                t1 = _dt2.datetime.fromisoformat(latest_accepted["accepted_at"])
                if t0.tzinfo is None: t0 = t0.replace(tzinfo=_dt2.timezone.utc)
                if t1.tzinfo is None: t1 = t1.replace(tzinfo=_dt2.timezone.utc)
                re_attestation_lag_days = max(0, (t1 - t0).days)
                # Lag bonus: faster re-attestation = higher trust
                if re_attestation_lag_days <= 1:
                    lag_bonus = 5.0
                elif re_attestation_lag_days <= 7:
                    lag_bonus = 3.0
                elif re_attestation_lag_days <= 30:
                    lag_bonus = 1.0
            except Exception:
                pass

        latest_pending = conn.execute(
            """SELECT created_at FROM succession_packages
               WHERE predecessor_agent_id = ? AND status = 'pending'
               ORDER BY created_at DESC LIMIT 1""",
            (agent_id,),
        ).fetchone()
        if latest_pending:
            try:
                import datetime as _dt3
                t_prep = _dt3.datetime.fromisoformat(latest_pending["created_at"])
                if t_prep.tzinfo is None: t_prep = t_prep.replace(tzinfo=_dt3.timezone.utc)
                pending_attestation_days = max(0, (_dt3.datetime.now(_dt3.timezone.utc) - t_prep).days)
            except Exception:
                pass

    # Being in a lineage = 3 pts/generation; each accepted successor = 3 pts; lag bonus
    succession_pts = round(min(15.0, generation * 3.0 + accepted_count * 3.0 + lag_bonus), 1)
    breakdown["succession"] = {
        "score": succession_pts,
        "max": 15,
        "lineage_generation": generation,
        "successors_created": accepted_count,
        "re_attestation_lag_days": re_attestation_lag_days,
        "pending_attestation_days": pending_attestation_days,
        "lag_bonus": lag_bonus,
        "lag_note": (
            "Same-day re-attestation (+5 pts)" if lag_bonus == 5.0 else
            "Re-attested within a week (+3 pts)" if lag_bonus == 3.0 else
            "Re-attested within 30 days (+1 pt)" if lag_bonus == 1.0 else
            "No succession yet" if re_attestation_lag_days is None else
            "Re-attestation lag >30 days (no bonus)"
        ),
    }

    # ── 5. Activity (15 pts) ──────────────────────────────────────────────────
    snap_count = len(snaps)
    # 15 snapshots = max 15 pts
    activity_pts = round(min(15.0, float(snap_count)), 1)
    breakdown["activity"] = {
        "score": activity_pts,
        "max": 15,
        "snapshots": snap_count,
    }

    # ── 6. Obligation breach penalty ──────────────────────────────────────────
    breach_penalty = 0.0
    if _table_exists(conn, "obligations"):
        failed_obls = conn.execute(
            "SELECT breach_penalty FROM obligations "
            "WHERE agent_id = ? AND status = 'failed' AND breach_penalty IS NOT NULL",
            (agent_id,),
        ).fetchall()
        breach_penalty = sum(float(r["breach_penalty"] or 0.0) for r in failed_obls)
        # Deduct from consistency (floor at 0)
        if breach_penalty > 0:
            consistency_pts = max(0.0, consistency_pts - breach_penalty)
            breakdown["consistency"]["breach_penalty_applied"] = round(breach_penalty, 1)
            breakdown["consistency"]["score"] = consistency_pts
    breakdown["obligations"] = {
        "breach_penalty_total": round(breach_penalty, 1),
        "note": "Sum of breach_penalty from failed obligations, deducted from consistency.",
    }

    # ── 7. Staleness ──────────────────────────────────────────────────────────
    last_snap_at = None
    if snaps:
        last_snap_at = snaps[-1]["created_at"]
    days_since_snap = _days_since(last_snap_at) if last_snap_at else 9999
    import datetime as _dt
    if last_snap_at:
        try:
            last_dt = _dt.datetime.fromisoformat(last_snap_at)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=_dt.timezone.utc)
            valid_until = (last_dt + _dt.timedelta(days=STALE_DAYS)).isoformat()
        except Exception:
            valid_until = None
    else:
        valid_until = None
    score_stale = days_since_snap > STALE_DAYS

    # ── Total & grade ─────────────────────────────────────────────────────────
    total = round(
        consistency_pts + longevity_pts + anchoring_pts + succession_pts + activity_pts,
        1,
    )
    total = min(100.0, total)

    if total >= 90:
        grade = "A+"
    elif total >= 80:
        grade = "A"
    elif total >= 70:
        grade = "B"
    elif total >= 60:
        grade = "C"
    elif total >= 50:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": total,
        "grade": grade,
        "breakdown": breakdown,
        "score_stale": score_stale,
        "score_valid_until": valid_until,
        "days_since_snapshot": days_since_snap if days_since_snap < 9999 else None,
        "stale_after_days": STALE_DAYS,
        "re_attestation_lag_days": re_attestation_lag_days,
        "pending_attestation_days": pending_attestation_days,
    }


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@trust_router.get("/trust/{agent_name}")
async def get_trust_score(agent_name: str):
    """
    Public. Compute and return the trust score for any Cathedral agent.
    Score is 0-100 with a breakdown across five signals:
    consistency, longevity, anchoring, succession, activity.
    """
    conn = _get_db()

    agent = conn.execute(
        "SELECT id, name, created_at FROM agents WHERE name = ?", (agent_name,)
    ).fetchone()

    if not agent:
        conn.close()
        raise HTTPException(404, f"Agent '{agent_name}' not found")

    result = compute_trust_score(agent["id"], agent["name"], conn)
    conn.close()

    stale_note = (
        f"Score is stale — agent has not snapshotted in {result.get('days_since_snapshot', '?')} days "
        f"(valid for {result.get('stale_after_days', 30)} days)."
    ) if result.get("score_stale") else None

    return {
        "success": True,
        "agent": agent_name,
        **result,
        "stale_warning": stale_note,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "signals": {
            "consistency":  "avg drift across snapshot history (lower = better)",
            "longevity":    "days since registration (longer = better)",
            "anchoring":    "BCH blockchain anchors recorded (more = better)",
            "succession":   "lineage depth + successors created",
            "activity":     "total snapshot count",
        },
    }
