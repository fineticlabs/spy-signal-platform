#!/usr/bin/env python
"""Pre-market preflight check: validates the entire system is ready for paper trading.

Usage
-----
    python scripts/preflight_check.py

Runs 11 checks and prints a PASS/FAIL summary.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

# ── result tracking ──────────────────────────────────────────────────────────

_results: list[tuple[bool, str, str]] = []  # (passed, name, detail)


def _record(passed: bool, name: str, detail: str) -> None:
    tag = "[PASS]" if passed else "[FAIL]"
    print(f"  {tag} {name}: {detail}")
    _results.append((passed, name, detail))


# ── 1. Python environment ────────────────────────────────────────────────────


def _check_python_env() -> None:
    vi = sys.version_info
    if vi < (3, 12):
        _record(False, "Python environment", f"Python {vi.major}.{vi.minor} < 3.12 required")
        return

    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        _record(False, "Python environment", "venv is not active — run: source .venv/bin/activate")
        return

    missing = []
    for pkg in ("pandas", "numpy", "talib", "alpaca", "structlog", "telegram"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        _record(False, "Python environment", f"Missing packages: {', '.join(missing)}")
    else:
        _record(
            True,
            "Python environment",
            f"Python {vi.major}.{vi.minor}, venv active, all packages OK",
        )


# ── 2. Alpaca connection ─────────────────────────────────────────────────────


def _check_alpaca() -> None:
    try:
        from src.config import get_alpaca_settings
        from src.execution.alpaca_executor import AlpacaExecutor

        settings = get_alpaca_settings()
        executor = AlpacaExecutor(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
            paper=True,
        )
        acct = executor.get_account()

        acct_num = acct.account_number
        is_paper = str(acct_num).startswith("PA")
        equity = float(acct.equity)
        buying_power = float(acct.buying_power)

        # acct.status may be an enum (AccountStatus.ACTIVE) or a string
        status_str = getattr(acct.status, "value", str(acct.status))

        issues = []
        if status_str != "ACTIVE":
            issues.append(f"status={status_str} (expected ACTIVE)")
        if not is_paper:
            issues.append(f"account {acct_num} is NOT paper (no PA prefix)")
        if equity < 50000:
            issues.append(f"equity ${equity:,.2f} < $50,000 minimum")
        if buying_power <= 0:
            issues.append(f"buying power ${buying_power:,.2f} <= $0")

        if issues:
            _record(False, "Alpaca paper account", "; ".join(issues))
        else:
            _record(
                True,
                "Alpaca paper account",
                f"${equity:,.0f} equity, ${buying_power:,.0f} BP",
            )
    except Exception as exc:
        _record(False, "Alpaca paper account", f"{type(exc).__name__}: {exc}")


# ── 3. Telegram bot ──────────────────────────────────────────────────────────


def _check_telegram() -> None:
    try:
        import httpx

        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

        if not bot_token or not chat_id:
            _record(False, "Telegram bot", "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
            return

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = httpx.post(
            url,
            json={"chat_id": chat_id, "text": "Preflight check: Telegram connected"},
            timeout=10,
        )

        if resp.status_code == 200:
            _record(True, "Telegram bot", "Message sent (HTTP 200)")
        else:
            _record(False, "Telegram bot", f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        _record(False, "Telegram bot", f"{type(exc).__name__}: {exc}")


# ── 4. Database ──────────────────────────────────────────────────────────────


def _check_database() -> None:
    db_path = _ROOT / "data" / "spy_signals.db"

    if not db_path.exists():
        _record(False, "Database", f"{db_path} does not exist — run: make backfill")
        return

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # WAL mode
        journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        if journal != "wal":
            _record(False, "Database", f"journal_mode={journal}, expected wal")
            conn.close()
            return

        # Ticker coverage
        rows = conn.execute("SELECT DISTINCT symbol FROM bars").fetchall()
        symbols_in_db = {row[0] for row in rows}

        from src.config import AppSettings

        expected = set(AppSettings().symbols)
        missing = expected - symbols_in_db
        if missing:
            _record(False, "Database", f"Missing tickers: {', '.join(sorted(missing))}")
            conn.close()
            return

        # Total bar count
        total = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]

        # Latest SPY bar
        latest_row = conn.execute(
            "SELECT timestamp FROM bars WHERE symbol='SPY' ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

        if latest_row is None:
            _record(False, "Database", "No SPY bars found")
            conn.close()
            return

        latest_ts = datetime.fromisoformat(latest_row[0])
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.replace(tzinfo=UTC)
        age = datetime.now(UTC) - latest_ts

        conn.close()

        if age > timedelta(days=7):
            _record(
                False,
                "Database",
                f"SPY data stale ({age.days}d old) — run: python scripts/backfill_data.py",
            )
        else:
            latest_str = latest_ts.strftime("%Y-%m-%d")
            _record(
                True,
                "Database",
                f"{len(symbols_in_db)} tickers, {total:,} bars, latest: {latest_str}",
            )
    except Exception as exc:
        _record(False, "Database", f"{type(exc).__name__}: {exc}")


# ── 5. Config ────────────────────────────────────────────────────────────────


def _check_config() -> None:
    try:
        from src.config import AppSettings

        settings = AppSettings()

        issues = []
        if settings.execution_mode != "paper_trade":
            issues.append(f"EXECUTION_MODE={settings.execution_mode!r} (expected 'paper_trade')")
        if len(settings.symbols) != 26:
            issues.append(f"{len(settings.symbols)} symbols (expected 26)")

        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

        if not api_key:
            issues.append("ALPACA_API_KEY not set")
        if not secret:
            issues.append("ALPACA_SECRET_KEY not set")
        if not bot_token:
            issues.append("TELEGRAM_BOT_TOKEN not set")
        if not chat_id:
            issues.append("TELEGRAM_CHAT_ID not set")

        if issues:
            _record(False, "Config", "; ".join(issues))
        else:
            _record(
                True,
                "Config",
                f"{settings.execution_mode} mode, {len(settings.symbols)} symbols",
            )
    except Exception as exc:
        _record(False, "Config", f"{type(exc).__name__}: {exc}")


# ── 6. Launchd jobs ──────────────────────────────────────────────────────────


def _check_launchd() -> None:
    try:
        result = subprocess.run(
            ["launchctl", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [line for line in result.stdout.splitlines() if "fineticlabs" in line]

        if len(lines) < 2:
            _record(
                False,
                "Launchd jobs",
                f"Found {len(lines)}/2 jobs — run: bash scripts/install_launchd.sh",
            )
            return

        bad = []
        for line in lines:
            parts = line.split()
            # launchctl list format: PID  Status  Label
            # Status 0 = OK, 126 = permission denied, - = never ran
            exit_code = parts[1] if len(parts) >= 3 else "?"
            label = parts[-1]
            if exit_code not in ("0", "-"):
                bad.append(f"{label} exit={exit_code}")

        if bad:
            _record(False, "Launchd jobs", f"Bad exit codes: {'; '.join(bad)}")
        else:
            _record(True, "Launchd jobs", f"{len(lines)} jobs loaded")
    except Exception as exc:
        _record(False, "Launchd jobs", f"{type(exc).__name__}: {exc}")


# ── 7. Scripts executable ────────────────────────────────────────────────────


def _check_scripts_executable() -> None:
    scripts_dir = _ROOT / "scripts"
    missing_exec = []
    for name in ("auto_start.sh", "auto_stop.sh"):
        path = scripts_dir / name
        if not path.exists():
            missing_exec.append(f"{name} not found")
        elif not os.access(path, os.X_OK):
            missing_exec.append(f"{name} not executable — run: chmod +x scripts/{name}")

    if missing_exec:
        _record(False, "Scripts executable", "; ".join(missing_exec))
    else:
        _record(True, "Scripts executable", "auto_start.sh, auto_stop.sh OK")


# ── 8. Mac sleep ─────────────────────────────────────────────────────────────


def _check_mac_sleep() -> None:
    try:
        result = subprocess.run(
            ["pmset", "-g"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.splitlines()
        sleep_line = [
            ln for ln in lines if ln.strip().startswith("sleep") and "displaysleep" not in ln
        ]

        if not sleep_line:
            _record(False, "Mac sleep", "Could not parse pmset output")
            return

        # Parse "sleep    0 (sleep prevented by ...)" or "sleep    10"
        val = sleep_line[0].split()[1]
        if val == "0":
            _record(True, "Mac sleep", "sleep=0 (disabled)")
        else:
            _record(
                False,
                "Mac sleep",
                f"sleep={val} min — set to 0 in System Settings > Energy",
            )
    except Exception as exc:
        _record(False, "Mac sleep", f"{type(exc).__name__}: {exc}")


# ── 9. Scanner not running ───────────────────────────────────────────────────


def _check_no_scanner() -> None:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "src.main"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        pids = result.stdout.strip().splitlines()
        # Filter out our own process
        our_pid = str(os.getpid())
        pids = [p for p in pids if p and p != our_pid]

        if pids:
            _record(
                False,
                "No conflicting scanner",
                f"Scanner already running (PID {', '.join(pids)}) — kill first",
            )
        else:
            _record(True, "No conflicting scanner", "No scanner process found")
    except Exception as exc:
        _record(False, "No conflicting scanner", f"{type(exc).__name__}: {exc}")


# ── 10. Disk space ───────────────────────────────────────────────────────────


def _check_disk_space() -> None:
    try:
        usage = shutil.disk_usage(str(_ROOT))
        free_gb = usage.free / (1024**3)

        if free_gb < 1.0:
            _record(False, "Disk space", f"{free_gb:.1f}GB free — need at least 1GB")
        else:
            _record(True, "Disk space", f"{free_gb:.0f}GB free")
    except Exception as exc:
        _record(False, "Disk space", f"{type(exc).__name__}: {exc}")


# ── 11. Pipeline wiring ──────────────────────────────────────────────────────


def _check_pipeline_wiring() -> None:
    """Run the end-to-end pipeline wiring test (no mocks).

    This feeds synthetic bars through the REAL _build_registry, RegimeDetector,
    and ORBStrategy.  It catches:
    - regime.update() not receiving vix/adx
    - snapshot not recognizing new indicators
    - signal chain broken by missing wiring
    """
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/integration/test_pipeline_wiring.py",
                "-q",
                "--tb=short",
                "--no-header",
                "--override-ini=addopts=",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_ROOT),
        )
        output = result.stdout.strip()
        last_line = output.splitlines()[-1] if output.splitlines() else ""

        if result.returncode == 0:
            passed_str = last_line.split(" passed")[0].strip().split()[-1]
            _record(True, "Pipeline wiring", f"{passed_str} passed (real objects, no mocks)")
        else:
            # Show the first failure line for diagnosis
            fail_lines = [ln for ln in output.splitlines() if "FAILED" in ln]
            detail = fail_lines[0] if fail_lines else last_line
            _record(False, "Pipeline wiring", detail)
    except Exception as exc:
        _record(False, "Pipeline wiring", f"{type(exc).__name__}: {exc}")


# ── 12. Websocket connectivity ───────────────────────────────────────────────


def _check_websocket_connectivity() -> None:
    """Open a websocket to Alpaca, authenticate, then disconnect cleanly.

    Catches 'connection limit exceeded' and auth errors before starting
    the scanner, when there's still time to fix them.
    """
    try:
        import asyncio
        import contextlib

        from alpaca.data.enums import DataFeed
        from alpaca.data.live import StockDataStream

        from src.config import get_alpaca_settings

        settings = get_alpaca_settings()

        async def _test_ws() -> str:
            client = StockDataStream(
                api_key=settings.api_key,
                secret_key=settings.secret_key,
                feed=DataFeed.IEX,
            )
            try:
                await client._connect()
                await client._auth()
                # Auth succeeded — clean up
                await client.close()
                return "ok"
            except ValueError as exc:
                return f"auth failed: {exc}"
            except OSError as exc:
                return f"connection failed: {exc}"
            finally:
                with contextlib.suppress(Exception):
                    await client.close()

        result = asyncio.run(_test_ws())
        if result == "ok":
            _record(True, "Websocket connectivity", "Auth + disconnect OK")
        else:
            _record(False, "Websocket connectivity", result)
    except Exception as exc:
        _record(False, "Websocket connectivity", f"{type(exc).__name__}: {exc}")


# ── 13. Full test suite ─────────────────────────────────────────────────────


def _check_tests() -> None:
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-q",
                "--tb=no",
                "--no-header",
                "--override-ini=addopts=",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_ROOT),
        )
        output = result.stdout.strip()
        # Last line like "465 passed in 35.92s" or "1 failed, 464 passed in 36.00s"
        last_line = output.splitlines()[-1] if output.splitlines() else ""

        if result.returncode == 0:
            # Extract count from "465 passed in ..."
            passed_str = last_line.split(" passed")[0].strip().split()[-1]
            _record(True, "Full test suite", f"{passed_str} passed")
        else:
            _record(False, "Full test suite", f"Failures detected: {last_line}")
    except Exception as exc:
        _record(False, "Full test suite", f"{type(exc).__name__}: {exc}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("=" * 60)
    print("  PREFLIGHT CHECK")
    print("=" * 60)
    print()

    _check_python_env()
    _check_alpaca()
    _check_telegram()
    _check_database()
    _check_config()
    _check_launchd()
    _check_scripts_executable()
    _check_mac_sleep()
    _check_no_scanner()
    _check_disk_space()
    _check_websocket_connectivity()
    _check_pipeline_wiring()
    _check_tests()

    # ── summary ──────────────────────────────────────────────────────────────
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)

    print()
    print("=" * 60)
    print("  PREFLIGHT CHECK SUMMARY")
    print("=" * 60)

    for ok, name, _detail in _results:
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"  {tag} {name}")

    print()
    if passed == total:
        print(f"  {passed}/{total} checks passed. System ready for tomorrow.")
    else:
        print(f"  {passed}/{total} checks passed. Fix the failures above before market open.")
    print("=" * 60)
    print()

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
