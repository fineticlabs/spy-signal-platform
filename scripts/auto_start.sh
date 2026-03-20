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

# ── Kill ALL existing scanner processes (not just the PID file) ─────────────
echo "Cleaning up any existing scanner processes..."
# Kill by PID file first
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Killing scanner PID $OLD_PID..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$OLD_PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi

# Kill any orphaned python processes running src.main
ORPHANS=$(pgrep -f "src.main" 2>/dev/null | grep -v "$$" || true)
if [[ -n "$ORPHANS" ]]; then
    echo "Killing orphaned scanner processes: $ORPHANS"
    echo "$ORPHANS" | xargs kill 2>/dev/null || true
    sleep 2
    echo "$ORPHANS" | xargs kill -9 2>/dev/null || true
fi

# Wait for Alpaca websocket connections to release
echo "Waiting 15s for websocket connections to release..."
sleep 15

# ── Activate virtual environment ──────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "Activated venv: $(which python)"

cd "$PROJECT_DIR"

# ── Pre-start websocket connectivity test ─────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Testing Alpaca API connectivity..."
if python -c "
from src.config import get_alpaca_settings
from alpaca.data.historical import StockHistoricalDataClient
s = get_alpaca_settings()
c = StockHistoricalDataClient(api_key=s.api_key, secret_key=s.secret_key)
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta, timezone
r = StockBarsRequest(symbol_or_symbols='SPY', timeframe=TimeFrame.Minute, start=datetime.now(timezone.utc) - timedelta(hours=1), end=datetime.now(timezone.utc))
bars = c.get_stock_bars(r)
print(f'API OK — got {len(bars.data.get(\"SPY\", []))} bars')
" 2>&1; then
    echo "Alpaca connectivity confirmed."
else
    echo "WARNING: Alpaca API test failed. Scanner may have websocket issues."
fi

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

# ── Post-launch: wait for websocket confirmation ────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Waiting for websocket authentication (up to 90s)..."

WS_CONFIRMED=false
for i in $(seq 1 90); do
    if ! kill -0 "$SCANNER_PID" 2>/dev/null; then
        echo "ERROR: Scanner process died during startup (PID $SCANNER_PID)."
        echo "Check log: tail -30 $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
    if grep -q "websocket_authenticated" "$LOG_FILE" 2>/dev/null; then
        WS_CONFIRMED=true
        echo "Websocket authenticated after ${i}s."
        break
    fi
    sleep 1
done

if $WS_CONFIRMED; then
    # Also check for subscription confirmation
    for i in $(seq 1 30); do
        if grep -q "websocket_subscribed" "$LOG_FILE" 2>/dev/null; then
            echo "Websocket subscribed. Scanner is LIVE."
            break
        fi
        sleep 1
    done
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] auto_start.sh DONE — scanner confirmed live"
else
    echo ""
    echo "WARNING: Websocket did NOT authenticate within 90s."
    echo "Scanner may be stuck. Check: tail -50 $LOG_FILE"
    echo "The scanner will send a Telegram alert if it cannot connect."
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] auto_start.sh DONE — websocket NOT confirmed"
fi
echo "============================================================"
