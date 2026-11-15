#!/usr/bin/env python3
"""
Cathedral Self-Repairing Maintenance Bot
Runs checks, attempts fixes, verifies repairs, alerts only on failure.

Cron suggestion: 0 * * * * python3 /root/maintenance.py >> /root/maintenance.log 2>&1
"""

import subprocess
# SECRETS: loaded from /root/.cathedral_env
import os as _os
_env_file = "/root/.cathedral_env"
if _os.path.exists(_env_file):
    for _line in open(_env_file):
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            _os.environ.setdefault(_k.strip(), _v.strip())
import urllib.request
import json
import os
import time
import shutil
from datetime import datetime, timezone, timedelta

# ── Config ────────────────────────────────────────────────────────────────────

CATHEDRAL_API   = "https://cathedral-ai.com"
CATHEDRAL_KEY   = _os.environ.get("CATHEDRAL_KEY", "")
SERVICE_NAME    = "cathedral"
LOG_DIR         = "/root"
LOG_MAX_MB      = 50        # rotate logs larger than this
DISK_WARN_PCT   = 80        # warn at 80% disk usage
DISK_CRIT_PCT   = 90        # attempt cleanup at 90%
BOT_MAX_AGE_H   = 25        # alert if a bot hasn't run in this many hours
MAINTENANCE_LOG = "/root/maintenance.log"

BOT_LOGS = {
    "outreach": "/root/outreach.log",
    "brain":    "/root/brain.log",
    "monitor":  "/root/monitor.log",
    "digest":   "/root/digest.log",
}

# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)

def log_repair(check, action, success):
    status = "FIXED" if success else "FAILED"
    log(f"{check}: {action} -> {status}", level=status)

# ── Shell helpers ─────────────────────────────────────────────────────────────

def run(cmd):
    """Run a shell command, return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    return result.returncode, result.stdout.strip(), result.stderr.strip()

def service_active(name):
    code, out, _ = run(f"systemctl is-active {name}")
    return out == "active"

# ── Cathedral API ─────────────────────────────────────────────────────────────

def api_get(path):
    req = urllib.request.Request(
        f"{CATHEDRAL_API}{path}",
        headers={"Authorization": f"Bearer {CATHEDRAL_KEY}",
                 "User-Agent": "cathedral-maintenance/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}

def api_post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{CATHEDRAL_API}{path}",
        data=data,
        headers={"Authorization": f"Bearer {CATHEDRAL_KEY}",
                 "Content-Type": "application/json",
                 "User-Agent": "cathedral-maintenance/1.0"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}

def snapshot_event(event, detail):
    """Record a maintenance event in Cathedral memory."""
    api_post("/memories", {
        "content": f"[maintenance] {event}: {detail}",
        "category": "maintenance",
        "importance": 0.7
    })

# ── Checks + Repairs ──────────────────────────────────────────────────────────

def check_service():
    """Cathedral service running? Restart if not, verify after."""
    if service_active(SERVICE_NAME):
        return True, None

    log(f"Service '{SERVICE_NAME}' is down — attempting restart", level="WARN")
    run(f"systemctl restart {SERVICE_NAME}")
    time.sleep(5)

    if service_active(SERVICE_NAME):
        log_repair("service", f"systemctl restart {SERVICE_NAME}", success=True)
        snapshot_event("service_restart", "Cathedral service was down, restarted successfully")
        return True, None
    else:
        log_repair("service", f"systemctl restart {SERVICE_NAME}", success=False)
        return False, "Cathedral service failed to restart — needs manual investigation"


def check_api_health():
    """Is the API responding? Restart service if health check fails."""
    status, data = api_get("/health")

    if status == 200 and data.get("status") == "healthy":
        return True, None

    log(f"API health check failed (status={status}) — attempting service restart", level="WARN")
    run(f"systemctl restart {SERVICE_NAME}")
    time.sleep(8)

    status2, data2 = api_get("/health")
    ok = status2 == 200 and data2.get("status") == "healthy"
    log_repair("api_health", "service restart after unhealthy response", success=ok)
    if ok:
        snapshot_event("api_restart", f"API was unhealthy (status={status}), restarted and recovered")
    return ok, None if ok else f"API still unhealthy after restart (status={status2})"


def check_disk():
    """Check disk usage. Clean old logs if critical."""
    total, used, free = shutil.disk_usage("/")
    pct = int(used / total * 100)

    if pct < DISK_WARN_PCT:
        return True, None

    log(f"Disk at {pct}% — threshold is {DISK_WARN_PCT}%", level="WARN")

    if pct >= DISK_CRIT_PCT:
        freed = _clean_old_logs()
        total2, used2, _ = shutil.disk_usage("/")
        pct2 = int(used2 / total2 * 100)
        ok = pct2 < DISK_CRIT_PCT
        log_repair("disk", f"cleaned {freed:.1f}MB of old logs, disk now at {pct2}%", success=ok)
        if freed > 0:
            snapshot_event("disk_cleanup", f"Disk was {pct}%, freed {freed:.1f}MB, now {pct2}%")
        if not ok:
            return False, f"Disk at {pct2}% after cleanup — manual intervention needed"

    return True, f"Disk at {pct}% (warning threshold)"


def _clean_old_logs():
    """Delete log files older than 7 days, truncate oversized ones. Return MB freed."""
    freed_bytes = 0
    cutoff = datetime.now() - timedelta(days=7)

    for fname in os.listdir(LOG_DIR):
        if not fname.endswith(".log"):
            continue
        fpath = os.path.join(LOG_DIR, fname)
        if fpath == MAINTENANCE_LOG:
            continue
        try:
            stat = os.stat(fpath)
            size_mb = stat.st_size / 1_048_576
            mtime = datetime.fromtimestamp(stat.st_mtime)

            if mtime < cutoff:
                freed_bytes += stat.st_size
                os.remove(fpath)
                log(f"  Removed old log: {fname} ({size_mb:.1f}MB)")
            elif size_mb > LOG_MAX_MB:
                # Truncate to last 1000 lines
                code, out, _ = run(f"tail -n 1000 {fpath}")
                if code == 0:
                    freed_bytes += stat.st_size - len(out.encode())
                    with open(fpath, "w") as f:
                        f.write(out)
                    log(f"  Truncated oversized log: {fname} ({size_mb:.1f}MB -> kept last 1000 lines)")
        except OSError:
            pass

    return freed_bytes / 1_048_576


def check_bot_activity():
    """Check each bot log to ensure it ran within expected window."""
    issues = []
    now = datetime.now()

    for bot, log_path in BOT_LOGS.items():
        if not os.path.exists(log_path):
            issues.append(f"{bot}: log file missing")
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
        age_h = (now - mtime).total_seconds() / 3600

        if age_h > BOT_MAX_AGE_H:
            issues.append(f"{bot}: last run {age_h:.1f}h ago (limit {BOT_MAX_AGE_H}h)")

    if issues:
        for issue in issues:
            log(f"Bot activity: {issue}", level="WARN")
        # Bots run via cron — we can't safely restart them mid-cycle,
        # but we can flag it. Future: re-trigger cron job.
        return False, "Bot(s) overdue: " + "; ".join(issues)

    return True, None


def check_bot_errors():
    """Scan bot logs for unrecovered errors (errors after the last Done. marker)."""
    import re as _re
    errors_found = []

    for bot, log_path in BOT_LOGS.items():
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path) as f:
                lines = f.readlines()
            recent = lines[-200:]

            # Find the last successful completion marker
            done_indices = [i for i, l in enumerate(recent) if l.strip().startswith("Done.")]
            scan_from = done_indices[-1] + 1 if done_indices else 0

            # Only scan lines after the last Done. — anything before was a recovered error
            check_lines = recent[scan_from:]
            real_errors = [
                l.strip() for l in check_lines
                if (_re.search(r"(ERROR|CRITICAL|Traceback|Exception)", l)
                    or ("error" in l.lower() and not _re.search(r"Response \(2\d\d\)", l)))
            ]
            if real_errors:
                errors_found.append(f"{bot}: {len(real_errors)} error(s) — last: {real_errors[-1][:100]}")
        except OSError:
            pass

    if errors_found:
        for e in errors_found:
            log(f"Bot error detected: {e}", level="WARN")
        return False, "Errors in logs: " + " | ".join(errors_found)

    return True, None


# ── Alert (when self-repair fails) ────────────────────────────────────────────

NTFY_TOPIC = "cathedral-28f03b514e9d"

def alert(check, message):
    """Push notification via ntfy.sh + Cathedral memory on unrecovered failure."""
    log(f"ALERT -- {check}: {message}", level="ERROR")
    snapshot_event(f"alert_{check}", message)
    try:
        import urllib.request as _ur
        req = _ur.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={
                "Title": f"Cathedral Alert: {check}",
                "Priority": "high",
                "Tags": "warning,robot"
            },
            method="POST"
        )
        _ur.urlopen(req, timeout=10)
        log(f"Alert pushed to ntfy: {check}")
    except Exception as e:
        log(f"ntfy push failed: {e}")
def run_maintenance():
    log("=" * 50)
    log("Maintenance cycle started")

    checks = [
        ("service",      check_service),
        ("api_health",   check_api_health),
        ("disk",         check_disk),
        ("bot_activity", check_bot_activity),
        ("bot_errors",   check_bot_errors),
    ]

    passed = 0
    alerted = 0

    for name, fn in checks:
        try:
            ok, issue = fn()
            if ok:
                passed += 1
            else:
                alerted += 1
                if issue:
                    alert(name, issue)
        except Exception as e:
            log(f"Check '{name}' threw an exception: {e}", level="ERROR")
            alerted += 1

    log(f"Maintenance cycle done — {passed}/{len(checks)} checks passed, {alerted} alerts")
    log("=" * 50)


if __name__ == "__main__":
    run_maintenance()
