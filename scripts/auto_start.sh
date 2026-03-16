#!/usr/bin/env bash
# ── auto_start.sh ─────────────────────────────────────────────────────────────
# Activate venv, backfill latest data, start the live scanner.
# Called by launchd at 6:25 AM PT (Mon–Fri) or manually.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$HOME/Desktop/I/Projects/spy-signal-platform"
VENV_DIR="$PROJECT_DIR/.venv"
LOGS_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOGS_DIR/scanner.pid"
TODAY=$(date +%Y-%m-%d)
LOG_FILE="$LOGS_DIR/scanner_${TODAY}.log"

# ── Ensure logs directory exists ──────────────────────────────────────────────
mkdir -p "$LOGS_DIR"

exec >> "$LOG_FILE" 2>&1
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] auto_start.sh BEGIN"
echo "============================================================"

# ── Check if scanner is already running ───────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Scanner already running (PID $OLD_PID). Exiting."
        exit 0
    else
        echo "Stale PID file found (PID $OLD_PID not running). Cleaning up."
        rm -f "$PID_FILE"
    fi
fi

# ── Activate virtual environment ──────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "Activated venv: $(which python)"

cd "$PROJECT_DIR"

# ── Backfill latest data ─────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Running data backfill (last 5 days)..."
python scripts/backfill_data.py --days 5 || {
    echo "WARNING: Backfill failed, continuing with existing data."
}

# ── Start the live scanner ────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Starting live scanner..."
nohup python -m src.main >> "$LOG_FILE" 2>&1 &
SCANNER_PID=$!
echo "$SCANNER_PID" > "$PID_FILE"

echo "Scanner started with PID $SCANNER_PID"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] auto_start.sh DONE"
echo "============================================================"
